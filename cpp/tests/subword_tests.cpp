// Token-by-token parity against the reference tokenizer via the shared
// fixtures: the nine recorded cases (IDs and alignment) and the full
// subword IDs of the checked-in CC0 example texts through the combined
// segmentation + BPE pipeline.

#include "prism/segmentation.h"
#include "prism/subword.h"

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

const std::string kRoot = PRISM_REPOSITORY_ROOT;

template <typename T>
std::vector<std::int64_t> Ids(const T& array)
{
    std::vector<std::int64_t> result;
    for (const auto& value : array) {
        result.push_back(value.template get<std::int64_t>());
    }
    return result;
}

class SubwordTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        const auto vocabulary_path = kRoot + "/models/prism-no-0.2.3/vocabulary.json";
        if (!std::ifstream(vocabulary_path)) {
            GTEST_SKIP() << "Local artifact is not present.";
        }
        tokenizer_ = std::make_unique<prism::subword::Tokenizer>(vocabulary_path);
    }

    std::unique_ptr<prism::subword::Tokenizer> tokenizer_;
};

TEST_F(SubwordTokenizerTest, MatchesReferenceCases)
{
    // The shared cross-implementation fixture lives inside the Swift test
    // bundle because SwiftPM requires resources within the target directory;
    // its content is implementation-neutral.
    std::ifstream parity_file(
        kRoot + "/swift/Tests/PrismKitTests/Resources/subword-parity.json");
    ASSERT_TRUE(parity_file) << "Shared parity fixture is missing.";
    const auto parity = nlohmann::json::parse(parity_file);

    for (const auto& expected : parity.at("cases")) {
        prism::segmentation::PretokenizedSentence sentence;
        for (const auto& token : expected.at("tokens")) {
            sentence.tokens.push_back(token.get<std::string>());
        }
        for (const auto& flag : expected.at("has_space_before")) {
            sentence.has_space_before.push_back(flag.get<bool>());
        }
        const auto encoded = tokenizer_->Encode(sentence);
        EXPECT_EQ(encoded.input_ids, Ids(expected.at("input_ids")))
            << "case starting with: " << sentence.tokens.front();
        EXPECT_EQ(encoded.first_subword_indices, Ids(expected.at("first_subword_indices")))
            << "case starting with: " << sentence.tokens.front();
        EXPECT_EQ(encoded.subword_end_indices, Ids(expected.at("subword_end_indices")))
            << "case starting with: " << sentence.tokens.front();
    }
}

TEST_F(SubwordTokenizerTest, FixtureSubwordIdsMatchPythonReference)
{
    // Both checked-in CC0 example texts, replayed through segmentation and
    // BPE against the recorded reference IDs (see data/examples/README.md).
    for (const auto* fixture :
        {"skarvholmen-bokmaal", "fjellvatnet-nynorsk"}) {
        std::ifstream text_file(
            kRoot + "/data/examples/" + fixture + ".txt", std::ios::binary);
        std::ifstream oracle_file(
            kRoot + "/data/examples/" + fixture + "-subword-parity.json");
        ASSERT_TRUE(text_file && oracle_file) << "Checked-in fixture is missing.";
        std::stringstream buffer;
        buffer << text_file.rdbuf();
        const auto oracle = nlohmann::json::parse(oracle_file);
        const auto& expected_ids = oracle.at("sentence_input_ids");

        const auto sentences = prism::segmentation::Segment(
            buffer.str(), prism::segmentation::NorwegianPolicy());
        ASSERT_EQ(sentences.size(), expected_ids.size());
        for (std::size_t index = 0; index < sentences.size(); ++index) {
            EXPECT_EQ(
                tokenizer_->Encode(sentences[index]).input_ids, Ids(expected_ids[index]))
                << fixture << " sentence " << index;
        }
    }
}

} // namespace
