#include "prism/segmentation.h"

#include <algorithm>

namespace prism::segmentation {
namespace {

// One codepoint of (possibly transformed) text together with its origin in
// the raw input. Synthesized characters — the space restored after sentence
// punctuation and the joining space of merged lines — carry an empty range
// positioned at the insertion point. Tokens never contain whitespace, so
// every codepoint that ends up inside a token has a real, non-empty origin.
struct TrackedCodepoint {
    char32_t value;
    std::size_t source_start;
    std::size_t source_end;

    bool synthesized() const { return source_start == source_end; }
};

bool IsContinuation(unsigned char byte)
{
    return (byte & 0xC0) == 0x80;
}

std::vector<TrackedCodepoint> DecodeUtf8(std::string_view text)
{
    std::vector<TrackedCodepoint> result;
    result.reserve(text.size());

    std::size_t index = 0;
    while (index < text.size()) {
        const auto byte = static_cast<unsigned char>(text[index]);
        char32_t value = 0;
        std::size_t byte_count = 1;

        if (byte < 0x80) {
            value = byte;
        } else if ((byte & 0xE0) == 0xC0
            && index + 1 < text.size()
            && IsContinuation(static_cast<unsigned char>(text[index + 1]))) {
            value = ((byte & 0x1F) << 6) | (static_cast<unsigned char>(text[index + 1]) & 0x3F);
            byte_count = 2;
        } else if ((byte & 0xF0) == 0xE0
            && index + 2 < text.size()
            && IsContinuation(static_cast<unsigned char>(text[index + 1]))
            && IsContinuation(static_cast<unsigned char>(text[index + 2]))) {
            value = ((byte & 0x0F) << 12)
                | ((static_cast<unsigned char>(text[index + 1]) & 0x3F) << 6)
                | (static_cast<unsigned char>(text[index + 2]) & 0x3F);
            byte_count = 3;
        } else if ((byte & 0xF8) == 0xF0
            && index + 3 < text.size()
            && IsContinuation(static_cast<unsigned char>(text[index + 1]))
            && IsContinuation(static_cast<unsigned char>(text[index + 2]))
            && IsContinuation(static_cast<unsigned char>(text[index + 3]))) {
            value = ((byte & 0x07) << 18)
                | ((static_cast<unsigned char>(text[index + 1]) & 0x3F) << 12)
                | ((static_cast<unsigned char>(text[index + 2]) & 0x3F) << 6)
                | (static_cast<unsigned char>(text[index + 3]) & 0x3F);
            byte_count = 4;
        } else {
            value = 0xFFFD;
        }

        result.push_back(TrackedCodepoint{value, index, std::min(index + byte_count, text.size())});
        index += byte_count;
    }
    return result;
}

void AppendUtf8(std::string& target, char32_t value)
{
    if (value <= 0x7F) {
        target.push_back(static_cast<char>(value));
    } else if (value <= 0x7FF) {
        target.push_back(static_cast<char>(0xC0 | ((value >> 6) & 0x1F)));
        target.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else if (value <= 0xFFFF) {
        target.push_back(static_cast<char>(0xE0 | ((value >> 12) & 0x0F)));
        target.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        target.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else {
        target.push_back(static_cast<char>(0xF0 | ((value >> 18) & 0x07)));
        target.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3F)));
        target.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        target.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    }
}

std::string TextOf(
    const std::vector<TrackedCodepoint>& codepoints, std::size_t start, std::size_t end)
{
    std::string text;
    for (auto index = start; index < end; ++index) {
        AppendUtf8(text, codepoints[index].value);
    }
    return text;
}

bool IsWhitespace(char32_t value)
{
    switch (value) {
    case U' ':
    case U'\t':
    case U'\n':
    case U'\r':
    case U'\f':
    case U'\v':
    case 0x00A0:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
        return true;
    default:
        return value >= 0x2000 && value <= 0x200A;
    }
}

bool IsDigit(char32_t value)
{
    return value >= U'0' && value <= U'9';
}

bool IsLetter(char32_t value)
{
    if ((value >= U'a' && value <= U'z') || (value >= U'A' && value <= U'Z')) {
        return true;
    }
    if (value < 0xC0) {
        return false;
    }
    return (value <= 0x24F && value != 0xD7 && value != 0xF7)
        || (value >= 0x0370 && value <= 0x03FF && value != 0x03A2)
        || (value >= 0x0400 && value <= 0x052F);
}

// Mirrors the reference implementation's \w (letters, digits, underscore).
bool IsWordCharacter(char32_t value)
{
    return IsLetter(value) || IsDigit(value) || value == U'_';
}

bool IsLowercaseLetter(char32_t value)
{
    if (value >= U'a' && value <= U'z') {
        return true;
    }
    return (value >= 0xDF && value <= 0xFF && value != 0xF7)
        || (value >= 0x100 && value <= 0x17F && (value % 2) == 1)
        || (value >= 0x3B1 && value <= 0x3C9)
        || (value >= 0x430 && value <= 0x44F);
}

bool IsUppercaseLetter(char32_t value)
{
    if (value >= U'A' && value <= U'Z') {
        return true;
    }
    return (value >= 0xC0 && value <= 0xDE && value != 0xD7)
        || (value >= 0x100 && value <= 0x17F && (value % 2) == 0)
        || (value >= 0x391 && value <= 0x3A9 && value != 0x3A2)
        || (value >= 0x410 && value <= 0x42F);
}

bool IsTerminal(char32_t value)
{
    return value == U'.' || value == U'!' || value == U'?' || value == 0x2026;
}

bool IsOpening(char32_t value)
{
    switch (value) {
    case 0x00AB: // «
    case U'"':
    case U'\'':
    case U'(':
    case U'[':
    case 0x201E: // „
    case 0x201C: // “
        return true;
    default:
        return false;
    }
}

std::string Lowercase(
    const std::vector<TrackedCodepoint>& codepoints, std::size_t start, std::size_t end)
{
    std::string result;
    for (auto index = start; index < end; ++index) {
        auto value = codepoints[index].value;
        if (value >= U'A' && value <= U'Z') {
            value += 32;
        } else if (value >= 0xC0 && value <= 0xDE && value != 0xD7) {
            value += 32;
        }
        AppendUtf8(result, value);
    }
    return result;
}

std::string Lowercase(std::string_view text)
{
    const auto codepoints = DecodeUtf8(text);
    return Lowercase(codepoints, 0, codepoints.size());
}

// Restore spaces lost after sentence punctuation ("veien.Et sekund"). The
// restored space is synthesized: it has no bytes of its own in the input.
std::vector<TrackedCodepoint> RestoreMissingSentenceSpaces(
    const std::vector<TrackedCodepoint>& codepoints)
{
    std::vector<TrackedCodepoint> repaired;
    repaired.reserve(codepoints.size() + 16);

    for (std::size_t index = 0; index < codepoints.size(); ++index) {
        const auto& current = codepoints[index];
        repaired.push_back(current);
        if (index == 0 || index + 1 >= codepoints.size()) {
            continue;
        }
        if (IsTerminal(current.value)
            && IsLowercaseLetter(codepoints[index - 1].value)
            && (IsUppercaseLetter(codepoints[index + 1].value) || IsOpening(codepoints[index + 1].value))) {
            repaired.push_back(TrackedCodepoint{U' ', current.source_end, current.source_end});
        }
    }
    return repaired;
}

// Join wrapped lines into paragraphs without joining layout lines. Joining
// spaces are synthesized; a popped line-break hyphen simply drops out of
// the mapping, so a de-hyphenated token keeps one fragment per line.
std::vector<std::vector<TrackedCodepoint>> MergeWrappedLines(
    const std::vector<TrackedCodepoint>& text)
{
    std::vector<std::vector<TrackedCodepoint>> paragraphs;
    std::vector<TrackedCodepoint> paragraph;

    auto flush = [&]() {
        if (!paragraph.empty()) {
            paragraphs.push_back(paragraph);
        }
        paragraph.clear();
    };

    std::size_t line_start = 0;
    while (line_start <= text.size()) {
        auto line_end = line_start;
        while (line_end < text.size() && text[line_end].value != U'\n') {
            line_end += 1;
        }

        // Collapse internal whitespace runs like " ".join(line.split()).
        std::vector<TrackedCodepoint> line;
        bool previous_space = true;
        for (auto index = line_start; index < line_end; ++index) {
            const auto& codepoint = text[index];
            if (IsWhitespace(codepoint.value)) {
                previous_space = true;
                continue;
            }
            if (previous_space && !line.empty()) {
                line.push_back(
                    TrackedCodepoint{U' ', codepoint.source_start, codepoint.source_start});
            }
            previous_space = false;
            line.push_back(codepoint);
        }

        if (line.empty()) {
            flush();
        } else {
            const bool continues_previous = !paragraph.empty()
                && !IsTerminal(paragraph.back().value) && paragraph.back().value != U':'
                && IsLowercaseLetter(line.front().value);
            if (continues_previous && paragraph.back().value == U'-') {
                paragraph.pop_back();
                paragraph.insert(paragraph.end(), line.begin(), line.end());
            } else if (continues_previous) {
                paragraph.push_back(TrackedCodepoint{
                    U' ', line.front().source_start, line.front().source_start});
                paragraph.insert(paragraph.end(), line.begin(), line.end());
            } else {
                flush();
                paragraph = std::move(line);
            }
        }

        if (line_end >= text.size()) {
            break;
        }
        line_start = line_end + 1;
    }
    flush();
    return paragraphs;
}

bool IsProtectedBoundary(const std::vector<TrackedCodepoint>& codepoints,
    std::size_t match_start, std::size_t match_end, const SegmentationPolicy& policy)
{
    auto preceding_start = match_start;
    while (preceding_start > 0 && !IsWhitespace(codepoints[preceding_start - 1].value)) {
        preceding_start -= 1;
    }
    if (policy.abbreviation_tokens.contains(Lowercase(codepoints, preceding_start, match_end))) {
        return true;
    }

    // Ordinal or list numbers such as "17." in "17. mai" never end a sentence.
    if (match_end - match_start != 1 || codepoints[match_start].value != U'.') {
        return false;
    }
    auto digit_start = match_start;
    while (digit_start > 0 && IsDigit(codepoints[digit_start - 1].value)) {
        digit_start -= 1;
    }
    const auto digit_count = match_start - digit_start;
    if (digit_count < 1 || digit_count > 3) {
        return false;
    }
    return digit_start == 0 || IsWhitespace(codepoints[digit_start - 1].value);
}

std::vector<std::vector<TrackedCodepoint>> SplitParagraphSentences(
    const std::vector<TrackedCodepoint>& codepoints, const SegmentationPolicy& policy)
{
    std::vector<std::vector<TrackedCodepoint>> sentences;
    std::size_t sentence_start = 0;
    std::size_t index = 0;

    auto emit = [&](std::size_t end) {
        auto start = sentence_start;
        while (start < end && IsWhitespace(codepoints[start].value)) {
            start += 1;
        }
        auto last = end;
        while (last > start && IsWhitespace(codepoints[last - 1].value)) {
            last -= 1;
        }
        if (last > start) {
            sentences.emplace_back(codepoints.begin() + static_cast<std::ptrdiff_t>(start),
                codepoints.begin() + static_cast<std::ptrdiff_t>(last));
        }
    };

    while (index < codepoints.size()) {
        if (!IsTerminal(codepoints[index].value)) {
            index += 1;
            continue;
        }
        auto match_end = index + 1;
        while (match_end < codepoints.size() && IsTerminal(codepoints[match_end].value)) {
            match_end += 1;
        }
        const auto match_start = index;
        index = match_end;

        auto remainder_start = match_end;
        while (remainder_start < codepoints.size() && IsWhitespace(codepoints[remainder_start].value)) {
            remainder_start += 1;
        }
        if (remainder_start >= codepoints.size()) {
            continue;
        }
        const auto first = codepoints[remainder_start].value;
        if (!IsUppercaseLetter(first) && !IsOpening(first)) {
            continue;
        }
        // The terminal characters must be followed by whitespace.
        if (match_end >= codepoints.size() || !IsWhitespace(codepoints[match_end].value)) {
            continue;
        }
        if (IsProtectedBoundary(codepoints, match_start, match_end, policy)) {
            continue;
        }
        emit(match_end);
        sentence_start = match_end;
    }
    if (sentence_start < codepoints.size()) {
        emit(codepoints.size());
    }
    return sentences;
}

struct RawToken {
    std::size_t start; // codepoint indices
    std::size_t end;
};

bool MatchesAt(const std::vector<TrackedCodepoint>& codepoints, std::size_t start,
    std::string_view ascii)
{
    if (start + ascii.size() > codepoints.size()) {
        return false;
    }
    for (std::size_t offset = 0; offset < ascii.size(); ++offset) {
        if (codepoints[start + offset].value != static_cast<char32_t>(ascii[offset])) {
            return false;
        }
    }
    return true;
}

bool TryUrl(const std::vector<TrackedCodepoint>& codepoints, std::size_t start, std::size_t& end)
{
    if (!MatchesAt(codepoints, start, "http://") && !MatchesAt(codepoints, start, "https://")
        && !MatchesAt(codepoints, start, "www.")) {
        return false;
    }
    end = start;
    while (end < codepoints.size() && !IsWhitespace(codepoints[end].value)) {
        end += 1;
    }
    return true;
}

bool TryEmail(const std::vector<TrackedCodepoint>& codepoints, std::size_t start, std::size_t& end)
{
    // [^\s@]+ @ [^\s@]+ . \w+
    auto index = start;
    while (index < codepoints.size() && !IsWhitespace(codepoints[index].value)
        && codepoints[index].value != U'@') {
        index += 1;
    }
    if (index == start || index >= codepoints.size() || codepoints[index].value != U'@') {
        return false;
    }
    index += 1;
    // The domain must contain a dot followed by word characters; scan the
    // remaining non-space, non-@ run and verify that shape.
    const auto domain_start = index;
    while (index < codepoints.size() && !IsWhitespace(codepoints[index].value)
        && codepoints[index].value != U'@') {
        index += 1;
    }
    std::size_t match_end = 0;
    for (auto position = domain_start; position < index; ++position) {
        if (codepoints[position].value == U'.') {
            auto word_end = position + 1;
            while (word_end < index && IsWordCharacter(codepoints[word_end].value)) {
                word_end += 1;
            }
            if (word_end > position + 1) {
                match_end = word_end;
            }
        }
    }
    if (match_end == 0 || match_end <= domain_start) {
        return false;
    }
    end = match_end;
    return true;
}

std::size_t ConsumeNumber(const std::vector<TrackedCodepoint>& codepoints, std::size_t start)
{
    // \d+ ( [.,:/-] \d+ )*
    auto index = start;
    while (index < codepoints.size() && IsDigit(codepoints[index].value)) {
        index += 1;
    }
    while (index + 1 < codepoints.size()) {
        const auto separator = codepoints[index].value;
        const bool is_separator = separator == U'.' || separator == U',' || separator == U':'
            || separator == U'/' || separator == U'-';
        if (!is_separator || !IsDigit(codepoints[index + 1].value)) {
            break;
        }
        index += 1;
        while (index < codepoints.size() && IsDigit(codepoints[index].value)) {
            index += 1;
        }
    }
    return index;
}

std::size_t ConsumeWordRun(const std::vector<TrackedCodepoint>& codepoints, std::size_t start)
{
    auto index = start;
    while (index < codepoints.size() && IsWordCharacter(codepoints[index].value)) {
        index += 1;
    }
    return index;
}

// \w+ (\.\w+)+ \.?   — dotted abbreviations such as "f.eks." stay one token.
bool TryDottedAbbreviation(
    const std::vector<TrackedCodepoint>& codepoints, std::size_t start, std::size_t& end)
{
    auto index = ConsumeWordRun(codepoints, start);
    if (index == start) {
        return false;
    }
    std::size_t groups = 0;
    while (index < codepoints.size() && codepoints[index].value == U'.') {
        const auto word_end = ConsumeWordRun(codepoints, index + 1);
        if (word_end == index + 1) {
            break;
        }
        index = word_end;
        groups += 1;
    }
    if (groups == 0) {
        return false;
    }
    if (index < codepoints.size() && codepoints[index].value == U'.') {
        index += 1;
    }
    end = index;
    return true;
}

// \w+ ([-'’]\w+)*
std::size_t ConsumeWord(const std::vector<TrackedCodepoint>& codepoints, std::size_t start)
{
    auto index = ConsumeWordRun(codepoints, start);
    while (index + 1 < codepoints.size()) {
        const auto connector = codepoints[index].value;
        const bool is_connector = connector == U'-' || connector == U'\'' || connector == 0x2019;
        if (!is_connector || !IsWordCharacter(codepoints[index + 1].value)) {
            break;
        }
        index = ConsumeWordRun(codepoints, index + 1);
    }
    return index;
}

std::vector<RawToken> ScanRawTokens(const std::vector<TrackedCodepoint>& codepoints)
{
    std::vector<RawToken> tokens;
    std::size_t position = 0;

    while (position < codepoints.size()) {
        if (IsWhitespace(codepoints[position].value)) {
            position += 1;
            continue;
        }
        std::size_t end = 0;
        if (TryUrl(codepoints, position, end)) {
        } else if (TryEmail(codepoints, position, end)) {
        } else if (IsDigit(codepoints[position].value)) {
            end = ConsumeNumber(codepoints, position);
        } else if (TryDottedAbbreviation(codepoints, position, end)) {
        } else if (IsWordCharacter(codepoints[position].value)) {
            end = ConsumeWord(codepoints, position);
        } else {
            end = position + 1;
        }
        tokens.push_back(RawToken{position, end});
        position = end;
    }
    return tokens;
}

// The token's origin: adjacent codepoint origins merge into one fragment,
// while origins separated by removed input (a popped hyphen, a collapsed
// line break) stay separate fragments. Synthesized codepoints are
// whitespace and never occur inside tokens.
std::vector<Utf8ByteRange> CoalesceSourceRanges(
    const std::vector<TrackedCodepoint>& codepoints, std::size_t start, std::size_t end)
{
    std::vector<Utf8ByteRange> ranges;
    for (auto index = start; index < end; ++index) {
        const auto& codepoint = codepoints[index];
        if (codepoint.synthesized()) {
            continue;
        }
        if (!ranges.empty() && ranges.back().end == codepoint.source_start) {
            ranges.back().end = codepoint.source_end;
        } else {
            ranges.push_back(Utf8ByteRange{codepoint.source_start, codepoint.source_end});
        }
    }
    return ranges;
}

bool GapIsWhitespace(std::string_view text, std::size_t start, std::size_t end)
{
    for (const auto& codepoint : DecodeUtf8(text.substr(start, end - start))) {
        if (!IsWhitespace(codepoint.value)) {
            return false;
        }
    }
    return true;
}

// Sentence ranges cover every token fragment: fragments whose gap in the
// original text is pure whitespace share one range; gaps containing removed
// non-whitespace content split the sentence into several ranges.
std::vector<Utf8ByteRange> SentenceSourceRanges(
    const std::vector<std::vector<Utf8ByteRange>>& token_source_ranges, std::string_view text)
{
    std::vector<Utf8ByteRange> ranges;
    for (const auto& token : token_source_ranges) {
        for (const auto& fragment : token) {
            if (!ranges.empty() && ranges.back().end <= fragment.start
                && GapIsWhitespace(text, ranges.back().end, fragment.start)) {
                ranges.back().end = fragment.end;
            } else {
                ranges.push_back(fragment);
            }
        }
    }
    return ranges;
}

PretokenizedSentence TokenizeWithSpacing(const std::vector<TrackedCodepoint>& codepoints,
    const SegmentationPolicy& policy, std::string_view original_text)
{
    const auto raw = ScanRawTokens(codepoints);

    PretokenizedSentence sentence;
    std::vector<std::pair<std::size_t, std::size_t>> spans;

    std::size_t index = 0;
    while (index < raw.size()) {
        const auto token = raw[index];
        const bool period_is_attached = index + 1 < raw.size()
            && raw[index + 1].end - raw[index + 1].start == 1
            && codepoints[raw[index + 1].start].value == U'.'
            && raw[index + 1].start == token.end;
        bool keeps_period = false;
        if (period_is_attached) {
            const auto text = TextOf(codepoints, token.start, token.end);
            const auto token_length = token.end - token.start;
            bool all_digits = token_length <= 3;
            for (auto position = token.start; all_digits && position < token.end; ++position) {
                all_digits = IsDigit(codepoints[position].value);
            }
            keeps_period = policy.abbreviation_tokens.contains(Lowercase(text) + ".")
                || (all_digits && index + 2 < raw.size());
        }
        const auto token_end = keeps_period ? raw[index + 1].end : token.end;
        sentence.tokens.push_back(TextOf(codepoints, token.start, token_end));
        sentence.token_source_ranges.push_back(
            CoalesceSourceRanges(codepoints, token.start, token_end));
        spans.emplace_back(token.start, token_end);
        index += keeps_period ? 2 : 1;
    }

    sentence.has_space_before.reserve(sentence.tokens.size());
    for (std::size_t position = 0; position < sentence.tokens.size(); ++position) {
        sentence.has_space_before.push_back(
            position != 0 && spans[position - 1].second < spans[position].first);
    }
    sentence.source_ranges = SentenceSourceRanges(sentence.token_source_ranges, original_text);
    return sentence;
}

// Clip the parent's sentence ranges to the covering interval of one chunk's
// token fragments; boundaries stay codepoint boundaries because both inputs
// are codepoint boundaries of the same original text.
std::vector<Utf8ByteRange> ClipSentenceRanges(
    const std::vector<Utf8ByteRange>& ranges, std::size_t cover_start, std::size_t cover_end)
{
    std::vector<Utf8ByteRange> clipped;
    for (const auto& range : ranges) {
        const auto start = std::max(range.start, cover_start);
        const auto end = std::min(range.end, cover_end);
        if (start < end) {
            clipped.push_back(Utf8ByteRange{start, end});
        }
    }
    return clipped;
}

void AppendChunks(std::vector<PretokenizedSentence>& sentences, PretokenizedSentence sentence,
    std::size_t maximum_token_count)
{
    if (sentence.tokens.size() <= maximum_token_count) {
        if (!sentence.tokens.empty()) {
            sentences.push_back(std::move(sentence));
        }
        return;
    }
    const bool has_mapping = !sentence.token_source_ranges.empty();
    for (std::size_t start = 0; start < sentence.tokens.size(); start += maximum_token_count) {
        const auto end = std::min(start + maximum_token_count, sentence.tokens.size());
        PretokenizedSentence chunk;
        chunk.tokens.assign(sentence.tokens.begin() + static_cast<std::ptrdiff_t>(start),
            sentence.tokens.begin() + static_cast<std::ptrdiff_t>(end));
        chunk.has_space_before.assign(
            sentence.has_space_before.begin() + static_cast<std::ptrdiff_t>(start),
            sentence.has_space_before.begin() + static_cast<std::ptrdiff_t>(end));
        chunk.has_space_before[0] = false;
        if (has_mapping) {
            chunk.token_source_ranges.assign(
                sentence.token_source_ranges.begin() + static_cast<std::ptrdiff_t>(start),
                sentence.token_source_ranges.begin() + static_cast<std::ptrdiff_t>(end));
            chunk.source_ranges = ClipSentenceRanges(sentence.source_ranges,
                chunk.token_source_ranges.front().front().start,
                chunk.token_source_ranges.back().back().end);
        }
        sentences.push_back(std::move(chunk));
    }
}

} // namespace

SegmentationPolicy NorwegianPolicy(std::size_t maximum_token_count)
{
    return SegmentationPolicy{
        {
            "adm.", "ang.", "bl.a.", "ca.", "d.v.s.", "dr.", "dvs.", "eks.",
            "ekskl.", "evt.", "f.eks.", "f.o.m.", "fylkeskomm.", "hhv.", "ifm.",
            "iht.", "inkl.", "jf.", "jfr.", "kap.", "kfr.", "kl.", "kr.", "m.a.",
            "m.fl.", "m.m.", "m.v.", "mht.", "mill.", "mrd.", "mv.", "nr.", "osv.",
            "p.g.a.", "pga.", "pkt.", "ref.", "saksnr.", "st.", "t.o.m.", "tlf.",
            "vedr.",
        },
        maximum_token_count,
    };
}

std::vector<PretokenizedSentence> Chunk(
    const PretokenizedSentence& sentence, std::size_t maximum_token_count)
{
    std::vector<PretokenizedSentence> chunks;
    AppendChunks(chunks, sentence, maximum_token_count);
    return chunks;
}

std::vector<PretokenizedSentence> Segment(std::string_view text, const SegmentationPolicy& policy)
{
    const auto repaired = RestoreMissingSentenceSpaces(DecodeUtf8(text));
    std::vector<PretokenizedSentence> sentences;

    for (const auto& paragraph : MergeWrappedLines(repaired)) {
        for (const auto& sentence_codepoints : SplitParagraphSentences(paragraph, policy)) {
            AppendChunks(sentences,
                TokenizeWithSpacing(sentence_codepoints, policy, text),
                policy.maximum_token_count);
        }
    }
    return sentences;
}

} // namespace prism::segmentation
