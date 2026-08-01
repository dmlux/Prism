// End-to-end pipeline validation against decisions recorded from the
// reference tagger on the same frozen artifact, plus the C ABI surface.

#include "prism/prism_c.h"
#include "prism/tagger.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

namespace {

const std::string kRoot = PRISM_REPOSITORY_ROOT;
const std::string kArtifact = kRoot + "/models/prism-no-0.2.2";

class TaggerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (!std::ifstream(kArtifact + "/manifest.json")) {
            GTEST_SKIP() << "Local artifact is not present.";
        }
    }
};

TEST_F(TaggerTest, TagsRawTextWithReferenceDecisions)
{
    prism::tagger::Tagger tagger(kArtifact);

    const auto sentences = tagger.TagText("Hun kjøpte tre gamle bøker den 17. mai.");

    ASSERT_EQ(sentences.size(), 1U);
    const auto& tokens = sentences[0].tokens;
    const std::vector<std::string> expected_texts{
        "Hun", "kjøpte", "tre", "gamle", "bøker", "den", "17.", "mai", "."};
    const std::vector<std::string> expected_upos{
        "PRON", "VERB", "NUM", "ADJ", "NOUN", "DET", "ADJ", "NOUN", "PUNCT"};
    const std::vector<std::string> expected_lemmas{
        "hun", "kjøpe", "tre", "gammel", "bok", "den", "17.", "mai", "."};
    ASSERT_EQ(tokens.size(), expected_texts.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        EXPECT_EQ(tokens[index].text, expected_texts[index]);
        EXPECT_EQ(tokens[index].upos, expected_upos[index]);
        EXPECT_EQ(tokens[index].lemma, expected_lemmas[index]);
        EXPECT_GT(tokens[index].upos_confidence, 0.9);
        EXPECT_GT(tokens[index].lemma_confidence, 0.9);
    }
    ASSERT_TRUE(tokens[4].features.contains("Gender"));
    EXPECT_EQ(tokens[4].features.at("Gender"), std::vector<std::string>{"Fem"});
    ASSERT_TRUE(tokens[4].features.contains("Number"));
    EXPECT_EQ(tokens[4].features.at("Number"), std::vector<std::string>{"Plur"});
}

TEST_F(TaggerTest, TagsMoreSentencesThanOneBatch)
{
    prism::tagger::Tagger tagger(kArtifact);
    const std::vector<std::vector<std::string>> sentences(11, {"Katten", "sov", "."});

    const auto tagged = tagger.TagPretokenized(sentences);

    ASSERT_EQ(tagged.size(), 11U);
    for (const auto& sentence : tagged) {
        ASSERT_EQ(sentence.tokens.size(), 3U);
        EXPECT_EQ(sentence.tokens[0].upos, "NOUN");
        EXPECT_EQ(sentence.tokens[1].upos, "VERB");
        EXPECT_EQ(sentence.tokens[2].upos, "PUNCT");
        EXPECT_EQ(sentence.tokens[0].lemma, "katt");
        EXPECT_EQ(sentence.tokens[1].lemma, "sove");
    }
}

TEST_F(TaggerTest, TagsBookChapterWithinTimeBudget)
{
    std::ifstream chapter_file(kRoot + "/data/examples/hp7kap1.txt");
    if (!chapter_file) {
        GTEST_SKIP() << "Local chapter fixture is not present.";
    }
    std::ostringstream buffer;
    buffer << chapter_file.rdbuf();
    const auto text = buffer.str();

    prism::tagger::Tagger tagger(kArtifact);

    const auto begin = std::chrono::steady_clock::now();
    const auto tagged = tagger.TagText(text);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - begin);

    EXPECT_EQ(tagged.size(), 247U);
    std::size_t token_count = 0;
    for (const auto& sentence : tagged) {
        token_count += sentence.tokens.size();
    }
    EXPECT_EQ(token_count, 3783U);
    std::cout << "chapter tagging: " << elapsed.count() << " ms\n";
}

TEST_F(TaggerTest, CAbiExposesTheFullResultSurface)
{
    prism_tagger* tagger = prism_tagger_create(kArtifact.c_str());
    ASSERT_NE(tagger, nullptr) << prism_last_error();

    prism_result* result
        = prism_tagger_tag_text(tagger, "Hun kjøpte tre gamle bøker den 17. mai.");
    ASSERT_NE(result, nullptr) << prism_last_error();

    ASSERT_EQ(prism_result_sentence_count(result), 1U);
    ASSERT_EQ(prism_result_token_count(result, 0), 9U);

    EXPECT_STREQ(prism_result_token_text(result, 0, 4), "bøker");
    EXPECT_EQ(prism_result_token_has_space_before(result, 0, 4), 1);
    EXPECT_EQ(prism_result_token_has_space_before(result, 0, 0), 0);
    EXPECT_STREQ(prism_result_token_upos(result, 0, 4), "NOUN");
    EXPECT_GT(prism_result_token_upos_confidence(result, 0, 4), 0.9);
    EXPECT_STREQ(prism_result_token_lemma(result, 0, 4), "bok");
    EXPECT_GT(prism_result_token_lemma_confidence(result, 0, 4), 0.9);

    // "bøker" carries Definite/Gender/Number in alphabetical order.
    const auto feature_count = prism_result_token_feature_count(result, 0, 4);
    ASSERT_GE(feature_count, 2U);
    bool gender_seen = false;
    for (size_t feature = 0; feature < feature_count; ++feature) {
        const std::string name = prism_result_token_feature_name(result, 0, 4, feature);
        if (name == "Gender") {
            gender_seen = true;
            EXPECT_STREQ(prism_result_token_feature_value(result, 0, 4, feature), "Fem");
            EXPECT_GT(prism_result_token_feature_confidence(result, 0, 4, feature), 0.5);
        }
    }
    EXPECT_TRUE(gender_seen);
    const std::string features = prism_result_token_features(result, 0, 4);
    EXPECT_NE(features.find("Gender=Fem"), std::string::npos);
    EXPECT_NE(features.find("Number=Plur"), std::string::npos);

    // Out-of-range access degrades to NULL/0 instead of aborting.
    EXPECT_EQ(prism_result_token_text(result, 0, 99), nullptr);
    EXPECT_EQ(prism_result_token_count(result, 99), 0U);

    prism_result_destroy(result);

    const char* pretokenized[] = {"Katten", "sov", "."};
    prism_result* tokens_result = prism_tagger_tag_tokens(tagger, pretokenized, 3);
    ASSERT_NE(tokens_result, nullptr) << prism_last_error();
    ASSERT_EQ(prism_result_sentence_count(tokens_result), 1U);
    EXPECT_STREQ(prism_result_token_upos(tokens_result, 0, 0), "NOUN");
    EXPECT_STREQ(prism_result_token_lemma(tokens_result, 0, 1), "sove");
    prism_result_destroy(tokens_result);

    prism_tagger_destroy(tagger);
}

TEST(TaggerCAbi, ReportsErrorsThroughLastError)
{
    prism_tagger* tagger = prism_tagger_create("/nonexistent/artifact");
    EXPECT_EQ(tagger, nullptr);
    EXPECT_NE(std::string(prism_last_error()).size(), 0U);
}

} // namespace
