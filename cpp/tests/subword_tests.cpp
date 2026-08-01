// Token-by-token parity against the reference tokenizer via the shared
// fixtures: the nine recorded cases (IDs and alignment) and, when the local
// fixtures exist, the full book-chapter subword IDs through the combined
// segmentation + BPE pipeline.

#include "prism/segmentation.h"
#include "prism/subword.h"

#include <chrono>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

namespace {

int failures = 0;

void Check(bool condition, const std::string& label)
{
    if (!condition) {
        failures += 1;
        std::cerr << "FAILED: " << label << "\n";
    }
}

template <typename T>
std::vector<std::int64_t> Ids(const T& array)
{
    std::vector<std::int64_t> result;
    for (const auto& value : array) {
        result.push_back(value.template get<std::int64_t>());
    }
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string root = argc > 1 ? argv[1] : "..";
    const auto vocabulary_path = root + "/models/prism-no-0.2.0/vocabulary.json";
    if (!std::ifstream(vocabulary_path)) {
        std::cout << "SKIP subword tests: local artifact is not present\n";
        return 0;
    }
    const prism::subword::Tokenizer tokenizer(vocabulary_path);

    std::ifstream parity_file(
        root + "/swift/Tests/PrismKitTests/Resources/subword-parity.json");
    Check(static_cast<bool>(parity_file), "parity fixture present");
    const auto parity = nlohmann::json::parse(parity_file);
    for (const auto& expected : parity.at("cases")) {
        prism::segmentation::PretokenizedSentence sentence;
        for (const auto& token : expected.at("tokens")) {
            sentence.tokens.push_back(token.get<std::string>());
        }
        for (const auto& flag : expected.at("has_space_before")) {
            sentence.has_space_before.push_back(flag.get<bool>());
        }
        const auto encoded = tokenizer.Encode(sentence);
        Check(encoded.input_ids == Ids(expected.at("input_ids")),
            "input_ids: " + sentence.tokens.front());
        Check(encoded.first_subword_indices == Ids(expected.at("first_subword_indices")),
            "first_subword_indices: " + sentence.tokens.front());
        Check(encoded.subword_end_indices == Ids(expected.at("subword_end_indices")),
            "subword_end_indices: " + sentence.tokens.front());
    }

    std::ifstream chapter_file(root + "/data/examples/hp7kap1.txt", std::ios::binary);
    std::ifstream oracle_file(root + "/data/examples/hp7kap1-subword-parity.json");
    if (chapter_file && oracle_file) {
        std::stringstream buffer;
        buffer << chapter_file.rdbuf();
        const auto oracle = nlohmann::json::parse(oracle_file);
        const auto& expected_ids = oracle.at("sentence_input_ids");

        const auto started = std::chrono::steady_clock::now();
        const auto sentences = prism::segmentation::Segment(
            buffer.str(), prism::segmentation::NorwegianPolicy());
        std::size_t matches = 0;
        for (std::size_t index = 0; index < sentences.size(); ++index) {
            if (tokenizer.Encode(sentences[index]).input_ids == Ids(expected_ids[index])) {
                matches += 1;
            } else if (matches + 8 > index) { // report only first few
                std::cerr << "chapter sentence " << index << " diverges\n";
            }
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started);
        Check(sentences.size() == expected_ids.size(), "chapter sentence count");
        Check(matches == sentences.size(),
            "chapter subword parity (" + std::to_string(matches) + "/"
                + std::to_string(sentences.size()) + ")");
        std::cout << "chapter segmentation + bpe: " << elapsed.count() / 1000.0 << " ms\n";
    } else {
        std::cout << "SKIP chapter subword parity: local fixture is not present\n";
    }

    if (failures == 0) {
        std::cout << "all subword tests passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
