// Reproducible performance suite for the C++ layer, built on Google
// Benchmark. Everything it needs is checked in or downloadable from the
// releases page, so anyone can run the same measurements on their own
// machine:
//
//   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DPRISM_BENCHMARKS=ON
//   cmake --build cpp/build --target prism_benchmarks
//   cpp/build/prism_benchmarks
//
// The text inputs are the checked-in CC0 example texts under data/examples/
// (see the README there). The tagger benchmarks additionally need the local
// model artifacts (models/prism-no-0.2.3 for fp32, models/prism-no-0.2.3-fast
// for int8); variants whose artifact is missing are skipped with a note.
// PRISM_THREADS overrides the CPU thread count for sweeps.
//
// Measured variants:
//   - runtime segmentation per fixture (no model involved)
//   - subword BPE encoding per fixture (vocabulary only)
//   - tagger construction (artifact load without inference)
//   - TagText: raw text in, including segmentation (fp32 and fast)
//   - Tag(pretokenized): the same document with segmentation prepaid
//     (fp32 and fast)
//
// The document-scale runs repeat the Bokmål fixture until it exceeds 6,000
// tokens, matching the documented document-inference protocol; tokens/s is
// reported through the items-per-second counter.

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <prism/engine.h>
#include <prism/segmentation.h>
#include <prism/subword.h>
#include <prism/tagger.h>

namespace {

const std::string kRoot = PRISM_REPOSITORY_ROOT;

std::string ReadFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::size_t TokenCount(const std::vector<prism::segmentation::PretokenizedSentence>& sentences)
{
    std::size_t count = 0;
    for (const auto& sentence : sentences) {
        count += sentence.tokens.size();
    }
    return count;
}

// Repeat a fixture until the document exceeds the target token count; the
// documented document-inference protocol measures ~6,000-token documents.
std::string BuildDocument(const std::string& text, std::size_t minimum_tokens)
{
    const auto policy = prism::segmentation::NorwegianPolicy();
    std::string document = text;
    while (TokenCount(prism::segmentation::Segment(document, policy)) < minimum_tokens) {
        document += "\n\n";
        document += text;
    }
    return document;
}

void RegisterSegmentationBenchmark(const std::string& name, const std::string& text)
{
    const auto tokens = TokenCount(
        prism::segmentation::Segment(text, prism::segmentation::NorwegianPolicy()));
    benchmark::RegisterBenchmark(
        ("Segment/" + name).c_str(),
        [text, tokens](benchmark::State& state) {
            const auto policy = prism::segmentation::NorwegianPolicy();
            for (auto _ : state) {
                benchmark::DoNotOptimize(prism::segmentation::Segment(text, policy));
            }
            state.SetItemsProcessed(
                static_cast<std::int64_t>(state.iterations() * tokens));
        })
        ->Unit(benchmark::kMillisecond);
}

void RegisterSubwordBenchmark(
    const std::string& name, const std::string& text, const std::string& vocabulary_path)
{
    benchmark::RegisterBenchmark(
        ("SubwordEncode/" + name).c_str(),
        [text, vocabulary_path](benchmark::State& state) {
            const prism::subword::Tokenizer tokenizer(vocabulary_path);
            const auto sentences = prism::segmentation::Segment(
                text, prism::segmentation::NorwegianPolicy());
            const auto tokens = TokenCount(sentences);
            for (auto _ : state) {
                for (const auto& sentence : sentences) {
                    benchmark::DoNotOptimize(tokenizer.Encode(sentence));
                }
            }
            state.SetItemsProcessed(
                static_cast<std::int64_t>(state.iterations() * tokens));
        })
        ->Unit(benchmark::kMillisecond);
}

void RegisterTaggerBenchmarks(
    const std::string& precision, const std::string& artifact, const std::string& document)
{
    benchmark::RegisterBenchmark(
        ("TaggerLoad/" + precision).c_str(),
        [artifact](benchmark::State& state) {
            for (auto _ : state) {
                prism::tagger::Tagger tagger(artifact);
                benchmark::DoNotOptimize(&tagger);
            }
        })
        ->Unit(benchmark::kMillisecond);

    // One shared, warmed tagger per precision: the lowered programs load
    // lazily on first use, and the steady-state numbers are the ones the
    // document-inference protocol gates.
    auto tagger = std::make_shared<prism::tagger::Tagger>(artifact);
    const auto sentences = prism::segmentation::Segment(
        document, prism::segmentation::NorwegianPolicy(
                      static_cast<std::size_t>(
                          tagger->artifact().programs().back().shapes.token_count)));
    const auto tokens = TokenCount(sentences);
    (void)tagger->Tag({sentences.front()});

    benchmark::RegisterBenchmark(
        ("TagText/" + precision).c_str(),
        [tagger, document, tokens](benchmark::State& state) {
            for (auto _ : state) {
                benchmark::DoNotOptimize(tagger->TagText(document));
            }
            state.SetItemsProcessed(
                static_cast<std::int64_t>(state.iterations() * tokens));
        })
        ->Unit(benchmark::kMillisecond);

    benchmark::RegisterBenchmark(
        ("TagPretokenized/" + precision).c_str(),
        [tagger, sentences, tokens](benchmark::State& state) {
            for (auto _ : state) {
                benchmark::DoNotOptimize(tagger->Tag(sentences));
            }
            state.SetItemsProcessed(
                static_cast<std::int64_t>(state.iterations() * tokens));
        })
        ->Unit(benchmark::kMillisecond);
}

} // namespace

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);

    // PRISM_THREADS overrides the CPU backend thread count for sweeps.
    if (const char* threads = std::getenv("PRISM_THREADS")) {
        prism::engine::SetThreadCount(static_cast<std::size_t>(std::atoi(threads)));
    }

    const auto bokmaal = ReadFile(kRoot + "/data/examples/skarvholmen-bokmaal.txt");
    const auto nynorsk = ReadFile(kRoot + "/data/examples/fjellvatnet-nynorsk.txt");
    if (bokmaal.empty() || nynorsk.empty()) {
        std::cerr << "Checked-in example texts are missing under data/examples/.\n";
        return 1;
    }

    RegisterSegmentationBenchmark("bokmaal", bokmaal);
    RegisterSegmentationBenchmark("nynorsk", nynorsk);

    const auto document = BuildDocument(bokmaal, 6000);
    const struct {
        const char* precision;
        std::string artifact;
    } variants[] = {
        {"fp32", kRoot + "/models/prism-no-0.2.3"},
        {"fast", kRoot + "/models/prism-no-0.2.3-fast"},
    };
    bool vocabulary_registered = false;
    for (const auto& variant : variants) {
        if (!std::ifstream(variant.artifact + "/manifest.json")) {
            std::cerr << "note: skipping " << variant.precision
                      << " tagger benchmarks (missing artifact " << variant.artifact
                      << ")\n";
            continue;
        }
        if (!vocabulary_registered) {
            RegisterSubwordBenchmark(
                "bokmaal", bokmaal, variant.artifact + "/vocabulary.json");
            RegisterSubwordBenchmark(
                "nynorsk", nynorsk, variant.artifact + "/vocabulary.json");
            vocabulary_registered = true;
        }
        RegisterTaggerBenchmarks(variant.precision, variant.artifact, document);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
