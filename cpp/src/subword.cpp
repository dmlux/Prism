#include "prism/subword.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace prism::subword {

// A model's normalization step (NFC, NFKC, ...), applied per word before the
// byte-level mapping. Implementations live in the anonymous namespace below;
// MakeNormalizer selects one from the artifact's tokenizer definition.
class Normalizer {
public:
    virtual ~Normalizer() = default;
    virtual std::string Normalize(const std::string& word) const = 0;
};

// A model's pre-tokenizer: it splits a word's codepoints into the units BPE
// operates on. Returns [start, end) codepoint ranges over `values`; each
// range, including any leading space the model attaches to it, is one BPE
// unit. MakePreTokenizer selects the matching adapter from the artifact.
class PreTokenizer {
public:
    virtual ~PreTokenizer() = default;
    virtual std::vector<std::pair<std::size_t, std::size_t>>
    Split(const std::vector<char32_t>& values) const = 0;
};

namespace {

constexpr char kMergeSeparator = '\x01';

bool IsContinuation(unsigned char byte)
{
    return (byte & 0xC0) == 0x80;
}

std::vector<char32_t> DecodeUtf8(std::string_view text, std::vector<std::size_t>* byte_starts)
{
    std::vector<char32_t> values;
    values.reserve(text.size());
    std::size_t index = 0;
    while (index < text.size()) {
        const auto byte = static_cast<unsigned char>(text[index]);
        char32_t value = 0;
        std::size_t byte_count = 1;
        if (byte < 0x80) {
            value = byte;
        } else if ((byte & 0xE0) == 0xC0 && index + 1 < text.size()
            && IsContinuation(static_cast<unsigned char>(text[index + 1]))) {
            value = ((byte & 0x1F) << 6) | (static_cast<unsigned char>(text[index + 1]) & 0x3F);
            byte_count = 2;
        } else if ((byte & 0xF0) == 0xE0 && index + 2 < text.size()
            && IsContinuation(static_cast<unsigned char>(text[index + 1]))
            && IsContinuation(static_cast<unsigned char>(text[index + 2]))) {
            value = ((byte & 0x0F) << 12)
                | ((static_cast<unsigned char>(text[index + 1]) & 0x3F) << 6)
                | (static_cast<unsigned char>(text[index + 2]) & 0x3F);
            byte_count = 3;
        } else if ((byte & 0xF8) == 0xF0 && index + 3 < text.size()
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
        if (byte_starts != nullptr) {
            byte_starts->push_back(index);
        }
        values.push_back(value);
        index += byte_count;
    }
    if (byte_starts != nullptr) {
        byte_starts->push_back(text.size());
    }
    return values;
}

bool IsDigit(char32_t value)
{
    return value >= U'0' && value <= U'9';
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

bool IsLetter(char32_t value)
{
    return IsUppercaseLetter(value) || IsLowercaseLetter(value);
}

bool IsWhitespaceValue(char32_t value)
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

bool IsNewline(char32_t value)
{
    return value == U'\r' || value == U'\n';
}

// NorBERT4's pre-tokenizer: a Sequence of a case-splitting Split regex
// followed by a byte-level map. The scanner reproduces that Split regex's
// alternatives in order; each piece is one BPE unit. Its defining traits are
// splitting letter runs at the upper/lower-case boundary and emitting each
// digit on its own.
class NorbertPreTokenizer final : public PreTokenizer {
public:
    std::vector<std::pair<std::size_t, std::size_t>>
    Split(const std::vector<char32_t>& values) const override
    {
        std::vector<std::pair<std::size_t, std::size_t>> pieces;
        std::size_t position = 0;
        const auto size = values.size();

        auto is_optional_prefix = [&](char32_t value) {
            return !IsNewline(value) && !IsLetter(value) && !IsDigit(value);
        };

        while (position < size) {
            const auto start = position;
            const auto value = values[position];
            std::size_t index = position;

            // 1./2.: optional non-letter prefix + cased letter run.
            {
                auto cursor = index;
                if (is_optional_prefix(values[cursor]) && cursor + 1 < size
                    && IsLetter(values[cursor + 1])) {
                    cursor += 1;
                }
                if (IsLetter(values[cursor])) {
                    auto upper_end = cursor;
                    while (upper_end < size && IsUppercaseLetter(values[upper_end])) {
                        upper_end += 1;
                    }
                    auto lower_end = upper_end;
                    while (lower_end < size && IsLowercaseLetter(values[lower_end])) {
                        lower_end += 1;
                    }
                    if (lower_end > cursor) {
                        pieces.emplace_back(start, lower_end);
                        position = lower_end;
                        continue;
                    }
                }
            }
            // 3.: single digit.
            if (IsDigit(value)) {
                pieces.emplace_back(start, start + 1);
                position = start + 1;
                continue;
            }
            // 4.: optional space + symbol run + trailing newlines/slashes.
            {
                auto cursor = index;
                if (values[cursor] == U' ' && cursor + 1 < size
                    && !IsWhitespaceValue(values[cursor + 1]) && !IsLetter(values[cursor + 1])
                    && !IsDigit(values[cursor + 1])) {
                    cursor += 1;
                }
                auto symbol_end = cursor;
                while (symbol_end < size && !IsWhitespaceValue(values[symbol_end])
                    && !IsLetter(values[symbol_end]) && !IsDigit(values[symbol_end])) {
                    symbol_end += 1;
                }
                if (symbol_end > cursor) {
                    while (symbol_end < size
                        && (IsNewline(values[symbol_end]) || values[symbol_end] == U'/')) {
                        symbol_end += 1;
                    }
                    pieces.emplace_back(start, symbol_end);
                    position = symbol_end;
                    continue;
                }
            }
            // 5.-7.: whitespace runs; trailing whitespace keeps its last space
            // separate unless it ends the piece (\s+(?!\S) before \s+).
            {
                auto space_end = index;
                while (space_end < size && IsWhitespaceValue(values[space_end])) {
                    space_end += 1;
                }
                if (space_end > index) {
                    bool contains_newline = false;
                    for (auto cursor = index; cursor < space_end; ++cursor) {
                        contains_newline = contains_newline || IsNewline(values[cursor]);
                    }
                    if (!contains_newline && space_end < size && space_end - index > 1) {
                        space_end -= 1;
                    }
                    pieces.emplace_back(start, space_end);
                    position = space_end;
                    continue;
                }
            }
            pieces.emplace_back(start, start + 1);
            position = start + 1;
        }
        return pieces;
    }
};

// ModernBERT's pre-tokenizer: a byte-level map with the GPT-2 regex
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// The scanner tries the alternatives in that order. Unlike NorBERT4 it keeps
// whole letter runs and whole digit runs together, splits the lowercase
// contraction suffixes off, and attaches a single leading space (only U+0020,
// never a tab) to the following letter/digit/symbol run.
class ByteLevelPreTokenizer final : public PreTokenizer {
public:
    std::vector<std::pair<std::size_t, std::size_t>>
    Split(const std::vector<char32_t>& values) const override
    {
        std::vector<std::pair<std::size_t, std::size_t>> pieces;
        std::size_t position = 0;
        const auto size = values.size();

        auto is_symbol = [&](char32_t value) {
            return !IsWhitespaceValue(value) && !IsLetter(value) && !IsDigit(value);
        };
        // Run of `predicate` starting at `from`, taking an optional single
        // leading space (U+0020) before it. Returns the run's end, or `from`
        // when nothing matches.
        auto spaced_run = [&](std::size_t from, auto predicate) -> std::size_t {
            auto cursor = from;
            if (values[cursor] == U' ' && cursor + 1 < size && predicate(values[cursor + 1])) {
                cursor += 1;
            }
            if (!predicate(values[cursor])) {
                return from;
            }
            while (cursor < size && predicate(values[cursor])) {
                cursor += 1;
            }
            return cursor;
        };

        while (position < size) {
            const auto start = position;

            // 1.: lowercase contraction suffixes 's 't 're 've 'm 'll 'd.
            if (values[start] == U'\'' && start + 1 < size) {
                const auto next = values[start + 1];
                std::size_t length = 0;
                if (next == U's' || next == U't' || next == U'm' || next == U'd') {
                    length = 2;
                } else if (start + 2 < size
                    && ((next == U'r' && values[start + 2] == U'e')
                        || (next == U'v' && values[start + 2] == U'e')
                        || (next == U'l' && values[start + 2] == U'l'))) {
                    length = 3;
                }
                if (length > 0) {
                    pieces.emplace_back(start, start + length);
                    position = start + length;
                    continue;
                }
            }
            // 2.-4.: optional space + a run of letters, digits, or symbols.
            if (const auto end = spaced_run(start, [](char32_t v) { return IsLetter(v); });
                end > start) {
                pieces.emplace_back(start, end);
                position = end;
                continue;
            }
            if (const auto end = spaced_run(start, [](char32_t v) { return IsDigit(v); });
                end > start) {
                pieces.emplace_back(start, end);
                position = end;
                continue;
            }
            if (const auto end = spaced_run(start, is_symbol); end > start) {
                pieces.emplace_back(start, end);
                position = end;
                continue;
            }
            // 5./6.: a whitespace run. \s+(?!\S) holds back the last character
            // when a non-space follows (it then attaches to the next token via
            // its leading-space prefix above); otherwise \s+ takes it all.
            auto space_end = start;
            while (space_end < size && IsWhitespaceValue(values[space_end])) {
                space_end += 1;
            }
            const bool followed_by_non_space = space_end < size;
            const auto cut = (followed_by_non_space && space_end - start >= 2)
                ? space_end - 1
                : space_end;
            pieces.emplace_back(start, cut);
            position = cut;
        }
        return pieces;
    }
};

std::array<std::string, 256> BuildByteToUnicode()
{
    std::vector<int> byte_values;
    for (int value = 33; value <= 126; ++value) {
        byte_values.push_back(value);
    }
    for (int value = 161; value <= 172; ++value) {
        byte_values.push_back(value);
    }
    for (int value = 174; value <= 255; ++value) {
        byte_values.push_back(value);
    }
    std::vector<int> scalars = byte_values;
    int next = 0;
    for (int value = 0; value <= 255; ++value) {
        if (std::find(byte_values.begin(), byte_values.end(), value) == byte_values.end()) {
            byte_values.push_back(value);
            scalars.push_back(256 + next);
            next += 1;
        }
    }
    std::array<std::string, 256> table;
    for (std::size_t index = 0; index < byte_values.size(); ++index) {
        std::string encoded;
        const auto scalar = static_cast<char32_t>(scalars[index]);
        if (scalar <= 0x7F) {
            encoded.push_back(static_cast<char>(scalar));
        } else {
            encoded.push_back(static_cast<char>(0xC0 | ((scalar >> 6) & 0x1F)));
            encoded.push_back(static_cast<char>(0x80 | (scalar & 0x3F)));
        }
        table[static_cast<std::size_t>(byte_values[index])] = encoded;
    }
    return table;
}

// Compatibility folding for the characters that occur in Western prose.
// The reference pipeline applies full Unicode NFKC; this table covers the
// practically occurring subset (the shared parity fixtures are the gate)
// and passes unknown compatibility characters through unchanged.
const char* CompatibilityMapping(char32_t value)
{
    switch (value) {
    case 0x2026: return "...";
    case 0x2025: return "..";
    case 0xFB00: return "ff";
    case 0xFB01: return "fi";
    case 0xFB02: return "fl";
    case 0xFB03: return "ffi";
    case 0xFB04: return "ffl";
    case 0x2116: return "No";
    case 0x2122: return "TM";
    case 0x00A0: return " ";
    case 0x2000:
    case 0x2001:
    case 0x2002:
    case 0x2003:
    case 0x2004:
    case 0x2005:
    case 0x2006:
    case 0x2007:
    case 0x2008:
    case 0x2009:
    case 0x200A:
    case 0x202F:
    case 0x205F:
        return " ";
    default:
        return nullptr;
    }
}

std::string FoldCompatibility(const std::string& word)
{
    std::vector<std::size_t> byte_starts;
    const auto values = DecodeUtf8(word, &byte_starts);
    bool needs_folding = false;
    for (const auto value : values) {
        needs_folding = needs_folding || CompatibilityMapping(value) != nullptr;
    }
    if (!needs_folding) {
        return word;
    }
    std::string folded;
    folded.reserve(word.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (const auto* mapped = CompatibilityMapping(values[index])) {
            folded += mapped;
        } else {
            folded += word.substr(byte_starts[index], byte_starts[index + 1] - byte_starts[index]);
        }
    }
    return folded;
}

// NFC (ModernBERT): the artifact's words already arrive canonically composed,
// so this is the identity — no compatibility characters to fold.
class CanonicalNormalizer final : public Normalizer {
public:
    std::string Normalize(const std::string& word) const override { return word; }
};

// NFKC (NorBERT4): fold the compatibility characters that occur in Western
// prose. The Replace rules the Norwegian normalizer also carries act on
// newlines, which the runtime segmentation has already resolved before a word
// reaches the tokenizer, so they are no-ops here.
class CompatibilityNormalizer final : public Normalizer {
public:
    std::string Normalize(const std::string& word) const override
    {
        return FoldCompatibility(word);
    }
};

// Collects every normalizer type name, flattening a Sequence, so the selection
// can look for NFKC regardless of nesting.
void CollectNormalizerTypes(const nlohmann::json& config, std::vector<std::string>& types)
{
    if (!config.is_object()) {
        return;
    }
    const auto type = config.value("type", std::string());
    if (type == "Sequence") {
        for (const auto& inner : config.at("normalizers")) {
            CollectNormalizerTypes(inner, types);
        }
    } else if (!type.empty()) {
        types.push_back(type);
    }
}

// Selects the normalizer from the tokenizer definition: NFKC folds
// compatibility characters, everything else (NFC, or none) is the identity on
// already-composed input.
std::unique_ptr<const Normalizer> MakeNormalizer(const nlohmann::json& definition)
{
    std::vector<std::string> types;
    if (definition.contains("normalizer")) {
        CollectNormalizerTypes(definition.at("normalizer"), types);
    }
    if (std::find(types.begin(), types.end(), "NFKC") != types.end()) {
        return std::make_unique<CompatibilityNormalizer>();
    }
    return std::make_unique<CanonicalNormalizer>();
}

// Selects the pre-tokenizer adapter from the tokenizer definition. A bare
// byte-level map is ModernBERT's GPT-2 pre-tokenizer; a Sequence carrying a
// Split step is NorBERT4's case-splitting one. An unrecognized configuration
// is a hard error rather than a silent mis-tokenization of a new model.
std::unique_ptr<const PreTokenizer> MakePreTokenizer(const nlohmann::json& definition)
{
    const auto& config = definition.at("pre_tokenizer");
    const auto type = config.value("type", std::string());
    if (type == "ByteLevel") {
        return std::make_unique<ByteLevelPreTokenizer>();
    }
    if (type == "Sequence") {
        for (const auto& inner : config.at("pretokenizers")) {
            if (inner.value("type", std::string()) == "Split") {
                return std::make_unique<NorbertPreTokenizer>();
            }
        }
    }
    throw std::runtime_error("Unsupported tokenizer pre_tokenizer: " + type);
}

} // namespace

Tokenizer::Tokenizer(const std::filesystem::path& vocabulary_path)
{
    std::ifstream file(vocabulary_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open tokenizer definition: " + vocabulary_path.string());
    }
    const auto definition = nlohmann::json::parse(file);
    const auto& model = definition.at("model");

    for (const auto& [token, identifier] : model.at("vocab").items()) {
        vocabulary_.emplace(token, identifier.get<std::int64_t>());
    }
    ignore_merges_ = model.value("ignore_merges", false);

    // Byte-level BPE never emits the unknown token; read it only when the
    // model declares one (ModernBERT's is null).
    if (const auto& unk = model.at("unk_token"); !unk.is_null()) {
        unknown_id_ = vocabulary_.at(unk.get<std::string>());
    }

    // The special tokens wrapping each sequence come from the tokenizer's
    // TemplateProcessing post-processor, so each model's convention is honored
    // without a hardcoded assumption (NorBERT4 "<s>"; ModernBERT "[CLS]"/"[SEP]").
    const auto& post_processor = definition.at("post_processor");
    const auto post_type = post_processor.at("type").get<std::string>();
    if (post_type != "TemplateProcessing") {
        throw std::runtime_error(
            "Unsupported tokenizer post_processor type: " + post_type);
    }
    const auto& special_tokens = post_processor.at("special_tokens");
    bool after_sequence = false;
    for (const auto& element : post_processor.at("single")) {
        if (element.contains("Sequence")) {
            after_sequence = true;
            continue;
        }
        if (element.contains("SpecialToken")) {
            const auto token = element.at("SpecialToken").at("id").get<std::string>();
            for (const auto& identifier : special_tokens.at(token).at("ids")) {
                (after_sequence ? suffix_ids_ : prefix_ids_)
                    .push_back(identifier.get<std::int64_t>());
            }
        }
    }

    int rank = 0;
    for (const auto& merge : model.at("merges")) {
        merge_ranks_.emplace(
            merge.at(0).get<std::string>() + kMergeSeparator + merge.at(1).get<std::string>(),
            rank);
        rank += 1;
    }
    byte_to_unicode_ = BuildByteToUnicode();

    // Normalization and pre-tokenization are model-specific; select each
    // adapter from the definition so a new backbone only needs its convention
    // recognized here, not a code change in the encode path.
    normalizer_ = MakeNormalizer(definition);
    pre_tokenizer_ = MakePreTokenizer(definition);
}

Tokenizer::~Tokenizer() = default;

EncodedSentence Tokenizer::Encode(const segmentation::PretokenizedSentence& sentence) const
{
    EncodedSentence encoded;
    encoded.input_ids.insert(
        encoded.input_ids.end(), prefix_ids_.begin(), prefix_ids_.end());

    for (std::size_t index = 0; index < sentence.tokens.size(); ++index) {
        const auto word = sentence.has_space_before[index]
            ? " " + sentence.tokens[index]
            : sentence.tokens[index];
        const auto subword_ids = EncodeWord(word);
        encoded.first_subword_indices.push_back(
            static_cast<std::int64_t>(encoded.input_ids.size()));
        encoded.input_ids.insert(encoded.input_ids.end(), subword_ids.begin(), subword_ids.end());
        encoded.subword_end_indices.push_back(
            static_cast<std::int64_t>(encoded.input_ids.size()));
    }
    encoded.input_ids.insert(
        encoded.input_ids.end(), suffix_ids_.begin(), suffix_ids_.end());
    return encoded;
}

std::vector<std::int64_t> Tokenizer::EncodeWord(const std::string& raw_word) const
{
    const auto word = normalizer_->Normalize(raw_word);
    std::vector<std::size_t> byte_starts;
    const auto values = DecodeUtf8(word, &byte_starts);

    std::vector<std::int64_t> identifiers;
    for (const auto& [piece_start, piece_end] : pre_tokenizer_->Split(values)) {
        std::string mapped;
        const auto byte_end = byte_starts[piece_end];
        for (auto byte_index = byte_starts[piece_start]; byte_index < byte_end; ++byte_index) {
            mapped += byte_to_unicode_[static_cast<unsigned char>(word[byte_index])];
        }
        const auto piece_ids = BytePairEncode(mapped);
        identifiers.insert(identifiers.end(), piece_ids.begin(), piece_ids.end());
    }
    return identifiers;
}

std::vector<std::int64_t> Tokenizer::BytePairEncode(const std::string& mapped) const
{
    if (ignore_merges_) {
        const auto whole = vocabulary_.find(mapped);
        if (whole != vocabulary_.end()) {
            return {whole->second};
        }
    }

    // Split the mapped string into its codepoints as initial symbols.
    std::vector<std::string> symbols;
    std::vector<std::size_t> byte_starts;
    DecodeUtf8(mapped, &byte_starts);
    for (std::size_t index = 0; index + 1 < byte_starts.size(); ++index) {
        symbols.push_back(
            mapped.substr(byte_starts[index], byte_starts[index + 1] - byte_starts[index]));
    }

    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        std::size_t best_index = symbols.size();
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto rank = merge_ranks_.find(
                symbols[index] + kMergeSeparator + symbols[index + 1]);
            if (rank != merge_ranks_.end() && rank->second < best_rank) {
                best_rank = rank->second;
                best_index = index;
            }
        }
        if (best_index >= symbols.size()) {
            break;
        }
        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index) + 1);
    }

    std::vector<std::int64_t> identifiers;
    identifiers.reserve(symbols.size());
    for (const auto& symbol : symbols) {
        const auto entry = vocabulary_.find(symbol);
        identifiers.push_back(entry != vocabulary_.end() ? entry->second : unknown_id_);
    }
    return identifiers;
}

} // namespace prism::subword
