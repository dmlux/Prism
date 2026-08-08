// Typed view of a versioned Prism model artifact directory.
//
// Exposes the manifest programs (sorted by capacity so callers can pick the
// smallest fitting shape), the tokenizer contract, and the label schema
// including lemma edit rules with the reference semantics of the exporter.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace prism::artifact {

struct FixedShapes {
    int batch_size = 0;
    int subword_count = 0;
    int token_count = 0;
    std::optional<int> character_count;
};

struct Program {
    std::string file_name;
    std::string backend;
    std::string precision;
    FixedShapes shapes;
    // Shared .ptd files holding externally stored weights; empty when the
    // program embeds its weights.
    std::vector<std::string> data_files;
};

struct TokenizerContract {
    std::string file_name;
    std::int64_t padding_token_id = 0;
};

struct MorphologyFeature {
    std::string name;
    std::vector<std::string> values;
    bool allows_multiple_values = false;
};

// A lemma as a reversible edit of the token form; removals and additions
// operate on codepoints of the NFC-normalized token.
struct LemmaEditRule {
    int prefix_removal = 0;
    int suffix_removal = 0;
    std::string prefix_addition;
    std::string suffix_addition;

    // Empty optional when the rule removes more codepoints than the token has.
    std::optional<std::string> Apply(const std::string& token) const;
};

struct Labels {
    std::vector<std::string> upos_labels;
    std::vector<MorphologyFeature> features;
    std::vector<LemmaEditRule> lemma_rules;
    std::vector<std::string> character_vocabulary;
    int maximum_character_count = 32;
};

class Artifact {
public:
    // Throws std::runtime_error when the directory misses required files or
    // the manifest misses required metadata.
    explicit Artifact(const std::filesystem::path& directory);

    const std::filesystem::path& directory() const { return directory_; }
    const TokenizerContract& tokenizer() const { return tokenizer_; }
    const Labels& labels() const { return labels_; }

    // Manifest metadata, straight from manifest.json. language_tags lists
    // the BCP 47 tags the artifact supports (currently for example "nb" and
    // "nn"), in manifest order; consumers decide language support from
    // these tags, never from directory or artifact names.
    const std::string& name() const { return name_; }
    const std::string& version() const { return version_; }
    const std::vector<std::string>& language_tags() const { return language_tags_; }

    // Lowercase, period-terminated abbreviations that protect sentence
    // boundaries during raw-text segmentation, carried in the manifest so the
    // runtime segments per language without a hardcoded policy. Empty for
    // legacy artifacts predating the field.
    const std::vector<std::string>& segmentation_abbreviations() const
    {
        return segmentation_abbreviations_;
    }

    // XNNPACK programs sorted by capacity, smallest first.
    const std::vector<Program>& programs() const { return programs_; }

private:
    std::filesystem::path directory_;
    std::string name_;
    std::string version_;
    std::vector<std::string> language_tags_;
    std::vector<std::string> segmentation_abbreviations_;
    TokenizerContract tokenizer_;
    Labels labels_;
    std::vector<Program> programs_;
};

} // namespace prism::artifact
