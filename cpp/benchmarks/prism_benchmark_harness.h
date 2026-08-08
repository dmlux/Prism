// Shared harness for the per-language Google Benchmark suites.
//
// Each language has its own tiny translation unit (prism_benchmarks_<lang>.cpp)
// that lists its example texts and model artifacts and calls Run() from main().
// The mechanics — reading the fixtures, building the document-scale input,
// registering the segmentation / subword / tagger benchmarks, and honoring
// PRISM_THREADS — live here once.

#pragma once

#include <string>
#include <vector>

namespace prism::benchmarks {

// One checked-in example text: a short label and its file name under
// data/examples/.
struct NamedText {
    std::string name;
    std::string file_name;
};

// One tagger artifact to measure: a label and its directory relative to the
// repository root (for example "models/prism-no-0.2.4").
struct TaggerVariant {
    std::string label;
    std::string artifact_directory;
};

// Everything a single language contributes to the suite. `tagger_text_name`
// selects which of `texts` drives the document-scale tagger runs.
struct LanguageBenchmarks {
    std::vector<NamedText> texts;
    std::vector<TaggerVariant> variants;
    std::string tagger_text_name;
};

// Registers and runs the language's benchmarks; returns a process exit code.
// Variants whose artifact is missing locally are skipped with a note.
int Run(int argc, char** argv, const LanguageBenchmarks& language);

} // namespace prism::benchmarks
