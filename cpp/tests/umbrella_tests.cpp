// Verifies that <prism> exposes the complete public API in one include.

#include <prism>

#include <gtest/gtest.h>

namespace {

TEST(Umbrella, ExposesEveryPublicNamespace)
{
    const auto policy = prism::segmentation::NorwegianPolicy();
    EXPECT_FALSE(policy.abbreviation_tokens.empty());

    const auto sentences = prism::segmentation::Segment("Katten sov.", policy);
    ASSERT_EQ(sentences.size(), 1U);
    EXPECT_EQ(sentences[0].tokens.size(), 3U);

    prism::artifact::LemmaEditRule rule;
    rule.suffix_removal = 2;
    rule.suffix_addition = "e";
    EXPECT_EQ(rule.Apply("kjøpte"), "kjøpe");

    // The engine, tagger, subword, and C ABI declarations are visible; the
    // remaining suites cover their behaviour against the artifact.
    EXPECT_EQ(sizeof(prism::engine::OutputTensor*), sizeof(void*));
    EXPECT_EQ(sizeof(prism::tagger::Tagger*), sizeof(void*));
    EXPECT_EQ(sizeof(prism::subword::Tokenizer*), sizeof(void*));
    EXPECT_EQ(sizeof(prism_tagger*), sizeof(void*));
}

} // namespace
