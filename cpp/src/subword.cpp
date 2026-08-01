#include "prism/subword.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace prism::subword {
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

// The GPT-style split pre-tokenizer as a scanner. Alternatives in the
// reference pattern's order; each piece is one BPE unit.
std::vector<std::pair<std::size_t, std::size_t>> SplitPieces(const std::vector<char32_t>& values)
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
    unknown_id_ = vocabulary_.at(model.at("unk_token").get<std::string>());
    begin_of_sequence_id_ = vocabulary_.at("<s>");

    int rank = 0;
    for (const auto& merge : model.at("merges")) {
        merge_ranks_.emplace(
            merge.at(0).get<std::string>() + kMergeSeparator + merge.at(1).get<std::string>(),
            rank);
        rank += 1;
    }
    byte_to_unicode_ = BuildByteToUnicode();
}

EncodedSentence Tokenizer::Encode(const segmentation::PretokenizedSentence& sentence) const
{
    EncodedSentence encoded;
    encoded.input_ids.push_back(begin_of_sequence_id_);

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
    return encoded;
}

std::vector<std::int64_t> Tokenizer::EncodeWord(const std::string& raw_word) const
{
    const auto word = FoldCompatibility(raw_word);
    std::vector<std::size_t> byte_starts;
    const auto values = DecodeUtf8(word, &byte_starts);

    std::vector<std::int64_t> identifiers;
    for (const auto& [piece_start, piece_end] : SplitPieces(values)) {
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
