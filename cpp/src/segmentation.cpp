#include "prism/segmentation.h"

#include <algorithm>
#include <array>

namespace prism::segmentation {
namespace {

struct Codepoint {
    char32_t value;
    std::size_t byte_start;
    std::size_t byte_end;
};

bool IsContinuation(unsigned char byte)
{
    return (byte & 0xC0) == 0x80;
}

std::vector<Codepoint> DecodeUtf8(std::string_view text)
{
    std::vector<Codepoint> result;
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

        result.push_back(Codepoint{value, index, std::min(index + byte_count, text.size())});
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

std::string Lowercase(std::string_view text)
{
    std::string result;
    for (const auto& codepoint : DecodeUtf8(text)) {
        auto value = codepoint.value;
        if (value >= U'A' && value <= U'Z') {
            value += 32;
        } else if (value >= 0xC0 && value <= 0xDE && value != 0xD7) {
            value += 32;
        }
        AppendUtf8(result, value);
    }
    return result;
}

// Restore spaces lost after sentence punctuation ("veien.Et sekund").
std::string RestoreMissingSentenceSpaces(const std::vector<Codepoint>& codepoints, std::string_view text)
{
    std::string repaired;
    repaired.reserve(text.size() + 16);

    for (std::size_t index = 0; index < codepoints.size(); ++index) {
        const auto& current = codepoints[index];
        repaired.append(text.substr(current.byte_start, current.byte_end - current.byte_start));
        if (index == 0 || index + 1 >= codepoints.size()) {
            continue;
        }
        if (IsTerminal(current.value)
            && IsLowercaseLetter(codepoints[index - 1].value)
            && (IsUppercaseLetter(codepoints[index + 1].value) || IsOpening(codepoints[index + 1].value))) {
            repaired.push_back(' ');
        }
    }
    return repaired;
}

// Join wrapped lines into paragraphs without joining layout lines.
std::vector<std::string> MergeWrappedLines(std::string_view text)
{
    std::vector<std::string> paragraphs;
    std::string paragraph;
    char32_t paragraph_last = 0;

    auto flush = [&]() {
        if (!paragraph.empty()) {
            paragraphs.push_back(paragraph);
        }
        paragraph.clear();
        paragraph_last = 0;
    };

    std::size_t line_start = 0;
    while (line_start <= text.size()) {
        const auto line_end = std::min(text.find('\n', line_start), text.size());
        const auto raw_line = text.substr(line_start, line_end - line_start);

        // Collapse internal whitespace runs like " ".join(line.split()).
        std::string line;
        char32_t first_value = 0;
        bool previous_space = true;
        for (const auto& codepoint : DecodeUtf8(raw_line)) {
            if (IsWhitespace(codepoint.value)) {
                previous_space = true;
                continue;
            }
            if (previous_space && !line.empty()) {
                line.push_back(' ');
            }
            previous_space = false;
            if (line.empty()) {
                first_value = codepoint.value;
            }
            AppendUtf8(line, codepoint.value);
        }

        if (line.empty()) {
            flush();
        } else {
            const bool continues_previous = !paragraph.empty()
                && !IsTerminal(paragraph_last) && paragraph_last != U':'
                && IsLowercaseLetter(first_value);
            if (continues_previous && paragraph.back() == '-') {
                paragraph.pop_back();
                paragraph += line;
            } else if (continues_previous) {
                paragraph += ' ';
                paragraph += line;
            } else {
                flush();
                paragraph = line;
            }
            const auto decoded = DecodeUtf8(paragraph);
            paragraph_last = decoded.back().value;
        }

        if (line_end >= text.size()) {
            break;
        }
        line_start = line_end + 1;
    }
    flush();
    return paragraphs;
}

bool IsProtectedBoundary(const std::vector<Codepoint>& codepoints, std::string_view paragraph,
    std::size_t match_start, std::size_t match_end, const SegmentationPolicy& policy)
{
    auto preceding_start = match_start;
    while (preceding_start > 0 && !IsWhitespace(codepoints[preceding_start - 1].value)) {
        preceding_start -= 1;
    }
    const auto word = paragraph.substr(codepoints[preceding_start].byte_start,
        codepoints[match_end - 1].byte_end - codepoints[preceding_start].byte_start);
    if (policy.abbreviation_tokens.contains(Lowercase(word))) {
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

std::vector<std::string> SplitParagraphSentences(std::string_view paragraph, const SegmentationPolicy& policy)
{
    const auto codepoints = DecodeUtf8(paragraph);
    std::vector<std::string> sentences;
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
            sentences.emplace_back(paragraph.substr(codepoints[start].byte_start,
                codepoints[last - 1].byte_end - codepoints[start].byte_start));
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
        if (IsProtectedBoundary(codepoints, paragraph, match_start, match_end, policy)) {
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

bool TryUrl(const std::vector<Codepoint>& codepoints, std::string_view text, std::size_t start, std::size_t& end)
{
    const auto remainder = text.substr(codepoints[start].byte_start);
    if (!remainder.starts_with("http://") && !remainder.starts_with("https://")
        && !remainder.starts_with("www.")) {
        return false;
    }
    end = start;
    while (end < codepoints.size() && !IsWhitespace(codepoints[end].value)) {
        end += 1;
    }
    return true;
}

bool TryEmail(const std::vector<Codepoint>& codepoints, std::size_t start, std::size_t& end)
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
    auto last_word_run = domain_start;
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
        (void)last_word_run;
    }
    if (match_end == 0 || match_end <= domain_start) {
        return false;
    }
    end = match_end;
    return true;
}

std::size_t ConsumeNumber(const std::vector<Codepoint>& codepoints, std::size_t start)
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

std::size_t ConsumeWordRun(const std::vector<Codepoint>& codepoints, std::size_t start)
{
    auto index = start;
    while (index < codepoints.size() && IsWordCharacter(codepoints[index].value)) {
        index += 1;
    }
    return index;
}

// \w+ (\.\w+)+ \.?   — dotted abbreviations such as "f.eks." stay one token.
bool TryDottedAbbreviation(const std::vector<Codepoint>& codepoints, std::size_t start, std::size_t& end)
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
std::size_t ConsumeWord(const std::vector<Codepoint>& codepoints, std::size_t start)
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

std::vector<RawToken> ScanRawTokens(const std::vector<Codepoint>& codepoints, std::string_view text)
{
    std::vector<RawToken> tokens;
    std::size_t position = 0;

    while (position < codepoints.size()) {
        if (IsWhitespace(codepoints[position].value)) {
            position += 1;
            continue;
        }
        std::size_t end = 0;
        if (TryUrl(codepoints, text, position, end)) {
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

std::string Slice(std::string_view text, const std::vector<Codepoint>& codepoints, std::size_t start, std::size_t end)
{
    return std::string(text.substr(codepoints[start].byte_start,
        codepoints[end - 1].byte_end - codepoints[start].byte_start));
}

PretokenizedSentence TokenizeWithSpacing(std::string_view sentence_text, const SegmentationPolicy& policy)
{
    const auto codepoints = DecodeUtf8(sentence_text);
    const auto raw = ScanRawTokens(codepoints, sentence_text);

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
            const auto text = Slice(sentence_text, codepoints, token.start, token.end);
            const auto token_length = token.end - token.start;
            bool all_digits = token_length <= 3;
            for (auto position = token.start; all_digits && position < token.end; ++position) {
                all_digits = IsDigit(codepoints[position].value);
            }
            keeps_period = policy.abbreviation_tokens.contains(Lowercase(text) + ".")
                || (all_digits && index + 2 < raw.size());
        }
        if (keeps_period) {
            sentence.tokens.push_back(Slice(sentence_text, codepoints, token.start, raw[index + 1].end));
            spans.emplace_back(token.start, raw[index + 1].end);
            index += 2;
            continue;
        }
        sentence.tokens.push_back(Slice(sentence_text, codepoints, token.start, token.end));
        spans.emplace_back(token.start, token.end);
        index += 1;
    }

    sentence.has_space_before.reserve(sentence.tokens.size());
    for (std::size_t position = 0; position < sentence.tokens.size(); ++position) {
        sentence.has_space_before.push_back(
            position != 0 && spans[position - 1].second < spans[position].first);
    }
    return sentence;
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
    for (std::size_t start = 0; start < sentence.tokens.size(); start += maximum_token_count) {
        const auto end = std::min(start + maximum_token_count, sentence.tokens.size());
        PretokenizedSentence chunk;
        chunk.tokens.assign(sentence.tokens.begin() + start, sentence.tokens.begin() + end);
        chunk.has_space_before.assign(
            sentence.has_space_before.begin() + start, sentence.has_space_before.begin() + end);
        chunk.has_space_before[0] = false;
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
    const auto repaired = RestoreMissingSentenceSpaces(DecodeUtf8(text), text);
    std::vector<PretokenizedSentence> sentences;

    for (const auto& paragraph : MergeWrappedLines(repaired)) {
        for (const auto& sentence_text : SplitParagraphSentences(paragraph, policy)) {
            AppendChunks(sentences, TokenizeWithSpacing(sentence_text, policy), policy.maximum_token_count);
        }
    }
    return sentences;
}

} // namespace prism::segmentation
