// Exercises the shared segmentation fixtures so every implementation of
// prism-runtime-segmentation-v1 stays behaviourally identical, including
// the reference counts of the checked-in CC0 example texts.

#include "prism/segmentation.h"

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

namespace {

using prism::segmentation::NorwegianPolicy;
using prism::segmentation::Segment;
using prism::segmentation::SegmentationPolicy;

std::vector<std::vector<std::string>> Tokens(const std::string& text, const SegmentationPolicy& policy)
{
    std::vector<std::vector<std::string>> result;
    for (const auto& sentence : Segment(text, policy)) {
        result.push_back(sentence.tokens);
    }
    return result;
}

SegmentationPolicy TestPolicy()
{
    return SegmentationPolicy{{"f.eks."}, 8};
}

TEST(RuntimeSegmentation, KeepsFragmentsAndHeadings)
{
    const auto tokens = Tokens("KAPITTEL 1\nHan gjekk heim.\nog so vidare", TestPolicy());
    const std::vector<std::vector<std::string>> expected{
        {"KAPITTEL", "1"},
        {"Han", "gjekk", "heim", "."},
        {"og", "so", "vidare"},
    };
    EXPECT_EQ(tokens, expected);
}

TEST(RuntimeSegmentation, ChunksLongSentencesWithoutLoss)
{
    std::string text;
    std::vector<std::string> words;
    for (int index = 0; index < 19; ++index) {
        words.push_back("ord" + std::to_string(index));
        if (index > 0) {
            text += ' ';
        }
        text += words.back();
    }
    text += '.';

    const auto sentences = Segment(text, TestPolicy());
    ASSERT_EQ(sentences.size(), 3u);
    std::vector<std::string> recovered;
    for (const auto& sentence : sentences) {
        ASSERT_FALSE(sentence.has_space_before.empty());
        EXPECT_FALSE(sentence.has_space_before.front());
        recovered.insert(recovered.end(), sentence.tokens.begin(), sentence.tokens.end());
    }
    words.push_back(".");
    EXPECT_EQ(recovered, words);
}

TEST(RuntimeSegmentation, RestoresMissingSentenceSpaces)
{
    const auto tokens = Tokens(
        "De begynte å gå.De gikk fort.«Noe nytt?» spurte han om f.eks.Dette.", TestPolicy());
    const std::vector<std::vector<std::string>> expected{
        {"De", "begynte", "å", "gå", "."},
        {"De", "gikk", "fort", "."},
        {"«", "Noe", "nytt", "?", "»", "spurte", "han", "om"},
        {"f.eks.", "Dette", "."},
    };
    EXPECT_EQ(tokens, expected);
}

TEST(RuntimeSegmentation, ProtectsAbbreviationsAndOrdinals)
{
    const auto tokens = Tokens("Vi feirar 17. mai med f.eks. kake. Det er fint.", TestPolicy());
    const std::vector<std::vector<std::string>> expected{
        {"Vi", "feirar", "17.", "mai", "med", "f.eks.", "kake", "."},
        {"Det", "er", "fint", "."},
    };
    EXPECT_EQ(tokens, expected);
}

TEST(RuntimeSegmentation, MergesWrappedLinesWithDehyphenation)
{
    const auto tokens = Tokens(
        "Katten\nhennes sov.\nDen hadde vand-\nring i blodet.", NorwegianPolicy(32));
    const std::vector<std::vector<std::string>> expected{
        {"Katten", "hennes", "sov", "."},
        {"Den", "hadde", "vandring", "i", "blodet", "."},
    };
    EXPECT_EQ(tokens, expected);
}

TEST(RuntimeSegmentation, SpacingReflectsAttachedPunctuation)
{
    const auto sentences = Segment("Hun sa «nei», ikke sant?", NorwegianPolicy());
    ASSERT_EQ(sentences.size(), 1u);
    const std::vector<std::string> tokens{
        "Hun", "sa", "«", "nei", "»", ",", "ikke", "sant", "?"};
    EXPECT_EQ(sentences[0].tokens, tokens);
    const std::vector<bool> spacing{
        false, true, true, false, false, false, true, true, false};
    EXPECT_EQ(sentences[0].has_space_before, spacing);
}

// The checked-in CC0 example texts (see data/examples/README.md) pin the
// Python reference implementation's sentence and token counts.
TEST(RuntimeSegmentation, BokmaalFixtureMatchesPythonReferenceCounts)
{
    std::ifstream file(
        std::string(PRISM_REPOSITORY_ROOT) + "/data/examples/skarvholmen-bokmaal.txt",
        std::ios::binary);
    ASSERT_TRUE(file) << "Checked-in fixture is missing.";
    std::stringstream buffer;
    buffer << file.rdbuf();

    const auto sentences = Segment(buffer.str(), NorwegianPolicy());

    std::size_t token_count = 0;
    for (const auto& sentence : sentences) {
        token_count += sentence.tokens.size();
    }
    EXPECT_EQ(sentences.size(), 55u);
    EXPECT_EQ(token_count, 905u);
}

TEST(RuntimeSegmentation, NynorskFixtureMatchesPythonReferenceCounts)
{
    std::ifstream file(
        std::string(PRISM_REPOSITORY_ROOT) + "/data/examples/fjellvatnet-nynorsk.txt",
        std::ios::binary);
    ASSERT_TRUE(file) << "Checked-in fixture is missing.";
    std::stringstream buffer;
    buffer << file.rdbuf();

    const auto sentences = Segment(buffer.str(), NorwegianPolicy());

    std::size_t token_count = 0;
    for (const auto& sentence : sentences) {
        token_count += sentence.tokens.size();
    }
    EXPECT_EQ(sentences.size(), 41u);
    EXPECT_EQ(token_count, 803u);
}

} // namespace
