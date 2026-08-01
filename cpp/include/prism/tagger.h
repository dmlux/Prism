// Frozen-artifact tagger: raw text or word tokens in, decisions plus
// calibrated confidences out. The lowered program already contains the
// complete decoding policy, so this class only assembles fixed-shape
// batches and applies argmax, the 0.5 threshold for multi-valued
// morphology features, and the lemma edit rules.
//
// Artifacts may ship several fixed-shape programs; sentences are sorted by
// length and every batch runs on the smallest program it fits into, so
// short sentences never pay the padding cost of the largest shapes.
//
// Input is expected in Unicode NFC (the artifact's recorded normalization).

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "prism/artifact.h"
#include "prism/segmentation.h"

namespace prism::tagger {

// One tagged token with calibrated confidences per decision. Multi-valued
// morphology features report the smallest confidence among their selected
// values; features whose decision is "not present" are omitted.
struct TaggedToken {
    std::string text;
    bool has_space_before = false;
    std::string upos;
    double upos_confidence = 0.0;
    std::map<std::string, std::vector<std::string>> features;
    std::map<std::string, double> feature_confidences;
    std::string lemma;
    double lemma_confidence = 0.0;
};

struct TaggedSentence {
    std::vector<TaggedToken> tokens;
};

class Tagger {
public:
    // Throws std::runtime_error when the artifact cannot be loaded.
    explicit Tagger(const std::filesystem::path& artifact_directory);
    ~Tagger();

    Tagger(const Tagger&) = delete;
    Tagger& operator=(const Tagger&) = delete;

    // Segment raw text with the runtime policy, then tag every sentence.
    std::vector<TaggedSentence> TagText(std::string_view text);

    // Tag application-supplied word tokens (space assumed between words).
    std::vector<TaggedSentence> TagPretokenized(
        const std::vector<std::vector<std::string>>& sentences);

    // Tag sentences carrying their own spacing information.
    std::vector<TaggedSentence> Tag(
        const std::vector<segmentation::PretokenizedSentence>& sentences);

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace prism::tagger
