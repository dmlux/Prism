#include "prism_benchmark_harness.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <prism/artifact.h>
#include <prism/engine.h>
#include <prism/segmentation.h>
#include <prism/subword.h>
#include <prism/tagger.h>

namespace prism::benchmarks {
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

// The runtime segmentation policy for an artifact comes from its manifest
// abbreviations. Every shipped artifact declares them; an empty inventory is a
// hard error rather than a silent fallback that would mis-segment the language.
prism::segmentation::SegmentationPolicy PolicyForArtifact(const std::string& artifact_directory)
{
    const prism::artifact::Artifact artifact(artifact_directory);
    const auto maximum_token_count = static_cast<std::size_t>(
        artifact.programs().back().shapes.token_count);
    const auto& abbreviations = artifact.segmentation_abbreviations();
    if (abbreviations.empty()) {
        throw std::runtime_error(
            "Artifact manifest declares no segmentation abbreviations: "
            + artifact_directory);
    }
    return prism::segmentation::SegmentationPolicy{
        std::unordered_set<std::string>(abbreviations.begin(), abbreviations.end()),
        maximum_token_count,
    };
}

// Repeat a fixture until the document exceeds the target token count; the
// documented document-inference protocol measures ~6,000-token documents.
std::string BuildDocument(
    const std::string& text,
    std::size_t minimum_tokens,
    const prism::segmentation::SegmentationPolicy& policy)
{
    std::string document = text;
    while (TokenCount(prism::segmentation::Segment(document, policy)) < minimum_tokens) {
        document += "\n\n";
        document += text;
    }
    return document;
}

void RegisterSegmentationBenchmark(
    const std::string& name,
    const std::string& text,
    const prism::segmentation::SegmentationPolicy& policy)
{
    const auto tokens = TokenCount(prism::segmentation::Segment(text, policy));
    benchmark::RegisterBenchmark(
        ("Segment/" + name).c_str(),
        [text, policy, tokens](benchmark::State& state) {
            for (auto _ : state) {
                benchmark::DoNotOptimize(prism::segmentation::Segment(text, policy));
            }
            state.SetItemsProcessed(
                static_cast<std::int64_t>(state.iterations() * tokens));
        })
        ->Unit(benchmark::kMillisecond);
}

void RegisterSubwordBenchmark(
    const std::string& name,
    const std::string& text,
    const std::string& vocabulary_path,
    const prism::segmentation::SegmentationPolicy& policy)
{
    benchmark::RegisterBenchmark(
        ("SubwordEncode/" + name).c_str(),
        [text, vocabulary_path, policy](benchmark::State& state) {
            const prism::subword::Tokenizer tokenizer(vocabulary_path);
            const auto sentences = prism::segmentation::Segment(text, policy);
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
    const std::string& precision,
    const std::string& artifact,
    const std::string& document,
    const prism::segmentation::SegmentationPolicy& policy)
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
    const auto sentences = prism::segmentation::Segment(document, policy);
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

int Run(int argc, char** argv, const LanguageBenchmarks& language)
{
    benchmark::Initialize(&argc, argv);

    // PRISM_THREADS overrides the CPU backend thread count for sweeps.
    if (const char* threads = std::getenv("PRISM_THREADS")) {
        prism::engine::SetThreadCount(static_cast<std::size_t>(std::atoi(threads)));
    }

    // The first present artifact supplies the segmentation policy and the
    // subword vocabulary for the whole language.
    std::string available_artifact;
    for (const auto& variant : language.variants) {
        if (std::ifstream(kRoot + "/" + variant.artifact_directory + "/manifest.json")) {
            available_artifact = kRoot + "/" + variant.artifact_directory;
            break;
        }
    }
    if (available_artifact.empty()) {
        std::cerr << "note: no local artifact present; nothing to benchmark\n";
        return 0;
    }
    const auto policy = PolicyForArtifact(available_artifact);

    std::string tagger_text;
    for (const auto& text : language.texts) {
        const auto content = ReadFile(kRoot + "/data/examples/" + text.file_name);
        if (content.empty()) {
            std::cerr << "note: missing example text " << text.name << "\n";
            continue;
        }
        RegisterSegmentationBenchmark(text.name, content, policy);
        RegisterSubwordBenchmark(
            text.name, content, available_artifact + "/vocabulary.json", policy);
        if (text.name == language.tagger_text_name) {
            tagger_text = content;
        }
    }

    if (!tagger_text.empty()) {
        const auto document = BuildDocument(tagger_text, 6000, policy);
        for (const auto& variant : language.variants) {
            const auto artifact = kRoot + "/" + variant.artifact_directory;
            if (!std::ifstream(artifact + "/manifest.json")) {
                std::cerr << "note: skipping " << variant.label
                          << " tagger benchmarks (missing artifact "
                          << variant.artifact_directory << ")\n";
                continue;
            }
            RegisterTaggerBenchmarks(variant.label, artifact, document, policy);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}

} // namespace prism::benchmarks
