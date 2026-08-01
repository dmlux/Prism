// Prism runtime segmentation (prism-runtime-segmentation-v1), C++ port.
//
// Raw application text becomes UD-convention word tokens without ever
// dropping user content: headings and fragments stay, over-long sentences
// are chunked, and spaces lost after sentence punctuation are restored.
// The implementation is a hand-written UTF-8 scanner — no regex engine and
// no ICU dependency — matching the Python and Swift reference behaviour,
// which the shared test fixtures enforce.
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

namespace prism::segmentation {

inline constexpr std::string_view kPolicyVersion = "prism-runtime-segmentation-v1";

struct PretokenizedSentence {
    std::vector<std::string> tokens;
    std::vector<bool> has_space_before;

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

} // namespace prism::segmentation
