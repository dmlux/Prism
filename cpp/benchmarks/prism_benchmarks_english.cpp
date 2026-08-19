// English performance suite (Google Benchmark). Build and run:
//
//   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DPRISM_BENCHMARKS=ON
//   cmake --build cpp/build --target prism_benchmarks_english
//   cpp/build/prism_benchmarks_english
//
// Inputs are the checked-in CC0 example text under data/examples/ and the local
// English (ModernBERT/Ettin) artifact. The fp32 tagger is measured over a
// ~6,000-token document built from the harbor text; PRISM_THREADS overrides the
// CPU thread count. The shared mechanics live in prism_benchmark_harness.cpp.

#include "prism_benchmark_harness.h"

int main(int argc, char** argv)
{
    const prism::benchmarks::LanguageBenchmarks english{
        {
            {"english", "harbor-english.txt"},
        },
        {
            // Shipped Ettin/ModernBERT reference (fp32).
            {"en-ettin-fp32", "models/prism-en-0.1.0"},
            // PrismBERT A/B: BabyLM GPT-BERT vs RoPE GPT-BERT, fp32 vs int8.
            // Missing artifacts are skipped, so export the ones you want first
            // (see docs/PRISMBERT.md / the export commands).
            {"en-babylm-fp32", "models/prism-en-babylm-fp32"},
            {"en-babylm-int8", "models/prism-en-babylm-int8"},
            {"en-rope-fp32", "models/prism-en-rope-fp32"},
            {"en-rope-int8", "models/prism-en-rope-int8"},
        },
        "english",
    };
    return prism::benchmarks::Run(argc, argv, english);
}
