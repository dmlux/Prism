// Prism runtime segmentation (prism-runtime-segmentation-v1), C++ port.
//
// Raw application text becomes UD-convention word tokens without ever
// dropping user content: headings and fragments stay, over-long sentences
// are chunked, and spaces lost after sentence punctuation are restored.
// The implementation is a hand-written UTF-8 scanner — no regex engine and
// no ICU dependency — matching the reference behaviour of the training
// pipeline, which the shared test fixtures enforce.
//
// Unicode scope: letter and digit classification covers Latin (including
// all Norwegian characters), Greek, and Cyrillic ranges. Scripts outside
// this scope tokenize character-wise, mirroring the documented policy.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "prism/utf8_byte_range.h"

namespace prism::segmentation {

inline constexpr std::string_view kPolicyVersion = "prism-runtime-segmentation-v1";

struct PretokenizedSentence {
    std::vector<std::string> tokens;
    std::vector<bool> has_space_before;

    // Source mapping against the exact raw-text input (see
    // <prism/utf8_byte_range.h> for the offset contract). Segment() always
    // fills both fields; sentences assembled from bare tokens leave them
    // empty, which unambiguously means "no source positions available".
    //
    // token_source_ranges holds, per token, the ordered non-overlapping
    // fragments of the original text the token was built from. A token that
    // survived unchanged has exactly one fragment; a token assembled from
    // several separated fragments (for example a de-hyphenated line wrap)
    // has one fragment per contributing piece. Callers who own their
    // tokenization may fill the fields themselves; Tagger::Tag validates
    // the invariants and passes the ranges through to the results.
    //
    // source_ranges covers the sentence: consecutive token fragments whose
    // gap in the original text is pure whitespace share one range, while
    // gaps containing removed non-whitespace content (for example the "-"
    // of a joined line break) split the sentence into several ranges.
    std::vector<std::vector<Utf8ByteRange>> token_source_ranges;
    std::vector<Utf8ByteRange> source_ranges;

    bool operator==(const PretokenizedSentence&) const = default;
};

struct SegmentationPolicy {
    // Lowercase abbreviations including their trailing period ("f.eks.").
    std::unordered_set<std::string> abbreviation_tokens;
    std::size_t maximum_token_count = 128;
};

// The Norwegian policy matching the Python reference implementation.
SegmentationPolicy NorwegianPolicy(std::size_t maximum_token_count = 128);

// Segment raw UTF-8 text without discarding content.
std::vector<PretokenizedSentence> Segment(std::string_view text, const SegmentationPolicy& policy);

// Split an over-long sentence into chunks of at most maximum_token_count
// tokens (Segment already applies this; callers with pretokenized input
// use it to satisfy a program's fixed token capacity). Source mappings,
// when present, travel with their tokens; each chunk's sentence ranges are
// the parent's ranges clipped to the chunk's own token fragments.
std::vector<PretokenizedSentence> Chunk(
    const PretokenizedSentence& sentence, std::size_t maximum_token_count);

} // namespace prism::segmentation
