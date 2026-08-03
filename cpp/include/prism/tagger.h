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

    // Ordered, non-overlapping half-open UTF-8 byte ranges locating this
    // token in the exact raw text passed to TagText (see
    // <prism/utf8_byte_range.h>). A token whose source is contiguous has
    // exactly one range; a token assembled from several separated input
    // fragments (for example "språk-\nmodellen" -> "språkmodellen") has one
    // range per contributing fragment, and removed characters such as the
    // line-break hyphen belong to no range. text may differ from the bytes
    // the ranges point to after internal repairs; text, has_space_before,
    // and source_ranges are three distinct pieces of information. Empty for
    // pretokenized input without caller-supplied ranges.
    std::vector<Utf8ByteRange> source_ranges;
};

struct TaggedSentence {
    std::vector<TaggedToken> tokens;

    // Ordered, non-overlapping half-open UTF-8 byte ranges covering every
    // token fragment of this sentence in the exact raw-text input:
    // fragments whose gap in the original is pure whitespace share one
    // range, gaps containing removed non-whitespace content (for example
    // the "-" of a joined line wrap) split the sentence into several
    // ranges. Empty for pretokenized input without caller-supplied ranges.
    std::vector<Utf8ByteRange> source_ranges;
};

class Tagger {
public:
    // Throws std::runtime_error when the artifact cannot be loaded.
    explicit Tagger(const std::filesystem::path& artifact_directory);
    ~Tagger();

    Tagger(const Tagger&) = delete;
    Tagger& operator=(const Tagger&) = delete;

    // The loaded artifact, including its manifest metadata (name, version,
    // language tags). Valid for the lifetime of the tagger.
    const artifact::Artifact& artifact() const;

    // Segment raw text with the runtime policy, then tag every sentence.
    // Every result carries source ranges against the exact input text.
    std::vector<TaggedSentence> TagText(std::string_view text);

    // Tag application-supplied word tokens (space assumed between words).
    // Without raw text there are no source positions: results carry empty
    // source ranges, which Prism never invents.
    std::vector<TaggedSentence> TagPretokenized(
        const std::vector<std::vector<std::string>>& sentences);

    // Tag sentences carrying their own spacing information. Callers who own
    // tokenization and source offsets may fill the sentences' source-range
    // fields; the ranges are validated (matching counts, non-empty, ordered,
    // non-overlapping — codepoint alignment stays the caller's contract,
    // since the raw text is not available here) and returned untouched on
    // the corresponding results. Throws std::invalid_argument on violation.
    std::vector<TaggedSentence> Tag(
        const std::vector<segmentation::PretokenizedSentence>& sentences);

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace prism::tagger
