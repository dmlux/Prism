// Native byte-level BPE tokenizer executing the artifact's vocabulary.json.
//
// Implements the exporter's reference behaviour: the byte-level table, plain
// BPE with ignore_merges, the tokenizer's special-token template, and the
// word-to-subword alignment the model pooling consumes. Parity is enforced by
// the shared fixtures.
//
// Normalization and pre-tokenization are the points where a specific model's
// tokenizer convention enters: both are read from vocabulary.json and served
// by a per-model adapter behind the Normalizer / PreTokenizer interfaces
// (NorBERT4's case-splitting Split regex and NFKC folding; ModernBERT's
// byte-level GPT-2 regex and plain NFC). See subword.cpp.

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "prism/segmentation.h"

namespace prism::subword {

// Model-specific normalization and pre-tokenization strategies. Declared here
// so the tokenizer can own them; defined, together with their per-model
// adapters and the factories that select one from the artifact, in subword.cpp.
class Normalizer;
class PreTokenizer;

struct EncodedSentence {
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> first_subword_indices;
    std::vector<std::int64_t> subword_end_indices;
};

class Tokenizer {
public:
    // Throws std::runtime_error when the definition cannot be loaded.
    explicit Tokenizer(const std::filesystem::path& vocabulary_path);
    ~Tokenizer();

    EncodedSentence Encode(const segmentation::PretokenizedSentence& sentence) const;
    std::vector<std::int64_t> EncodeWord(const std::string& word) const;

private:
    std::vector<std::int64_t> BytePairEncode(const std::string& mapped) const;

    std::unordered_map<std::string, std::int64_t> vocabulary_;
    std::unordered_map<std::string, int> merge_ranks_;
    bool ignore_merges_ = false;
    std::int64_t unknown_id_ = 0;
    // Special tokens wrapping each sequence, read generically from the
    // tokenizer's TemplateProcessing post-processor: NorBERT4 has an "<s>"
    // prefix and no suffix; ModernBERT has a "[CLS]" prefix and "[SEP]" suffix.
    std::vector<std::int64_t> prefix_ids_;
    std::vector<std::int64_t> suffix_ids_;
    std::array<std::string, 256> byte_to_unicode_;
    // The model's normalization and pre-tokenization, selected from the
    // artifact at load time.
    std::unique_ptr<const Normalizer> normalizer_;
    std::unique_ptr<const PreTokenizer> pre_tokenizer_;
};

} // namespace prism::subword
