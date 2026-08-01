// Native byte-level BPE tokenizer executing the artifact's vocabulary.json.
//
// Implements the exporter's reference behaviour: the GPT-style split
// pre-tokenizer as a hand-written scanner, the byte-level table, plain BPE
// with ignore_merges, the <s> template, and the word-to-subword alignment
// the model pooling consumes. Parity is enforced by the shared fixtures.
//
// Input is expected in Unicode NFC (the artifact's recorded normalization);
// NFKC compatibility folding is not applied, which the parity fixtures
// cover for the supported languages.

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "prism/segmentation.h"

namespace prism::subword {

struct EncodedSentence {
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> first_subword_indices;
    std::vector<std::int64_t> subword_end_indices;
};

class Tokenizer {
public:
    // Throws std::runtime_error when the definition cannot be loaded.
    explicit Tokenizer(const std::filesystem::path& vocabulary_path);

    EncodedSentence Encode(const segmentation::PretokenizedSentence& sentence) const;
    std::vector<std::int64_t> EncodeWord(const std::string& word) const;

    std::int64_t begin_of_sequence_id() const { return begin_of_sequence_id_; }

private:
    std::vector<std::int64_t> BytePairEncode(const std::string& mapped) const;

    std::unordered_map<std::string, std::int64_t> vocabulary_;
    std::unordered_map<std::string, int> merge_ranks_;
    bool ignore_merges_ = false;
    std::int64_t unknown_id_ = 0;
    std::int64_t begin_of_sequence_id_ = 0;
    std::array<std::string, 256> byte_to_unicode_;
};

} // namespace prism::subword
