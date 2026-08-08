// Norwegian performance suite (Google Benchmark). Build and run:
//
//   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DPRISM_BENCHMARKS=ON
//   cmake --build cpp/build --target prism_benchmarks_norwegian
//   cpp/build/prism_benchmarks_norwegian
//
// Inputs are the checked-in CC0 example texts under data/examples/ and the
// local model artifacts. The fp32 and int8 (-fast) taggers are measured over a
// ~6,000-token document built from the Bokmål text; PRISM_THREADS overrides the
// CPU thread count. The shared mechanics live in prism_benchmark_harness.cpp.

#include "prism_benchmark_harness.h"

int main(int argc, char** argv)
{
    const prism::benchmarks::LanguageBenchmarks norwegian{
        {
            {"bokmaal", "skarvholmen-bokmaal.txt"},
            {"nynorsk", "fjellvatnet-nynorsk.txt"},
        },
        {
            {"no-fp32", "models/prism-no-0.2.3"},
            {"no-fast", "models/prism-no-0.2.3-fast"},
        },
        "bokmaal",
    };
    return prism::benchmarks::Run(argc, argv, norwegian);
}
