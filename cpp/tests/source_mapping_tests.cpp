// Source-mapping contract of prism-runtime-segmentation-v1: every token
// and sentence of a raw-text analysis maps onto half-open UTF-8 byte
// ranges of the exact, unmodified input. The expected offsets are written
// as literals and shared with the Swift test suite, which pins byte-offset
// parity across the bindings.

#include "prism/segmentation.h"

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace {

using prism::Utf8ByteRange;
using prism::segmentation::NorwegianPolicy;
using prism::segmentation::PretokenizedSentence;
using prism::segmentation::Segment;
using prism::segmentation::SegmentationPolicy;

using Ranges = std::vector<Utf8ByteRange>;

SegmentationPolicy TestPolicy()
{
    return SegmentationPolicy{{"f.eks."}, 8};
}

// Every boundary must lie on a UTF-8 codepoint boundary of the input.
bool OnCodepointBoundary(std::string_view text, std::size_t offset)
{
    if (offset > text.size()) {
        return false;
    }
    return offset == text.size()
        || (static_cast<unsigned char>(text[offset]) & 0xC0U) != 0x80U;
}

void ExpectValidMapping(std::string_view text, const std::vector<PretokenizedSentence>& sentences)
{
    for (const auto& sentence : sentences) {
        ASSERT_EQ(sentence.token_source_ranges.size(), sentence.tokens.size());
        ASSERT_FALSE(sentence.source_ranges.empty());
        std::size_t previous_end = 0;
        for (const auto& range : sentence.source_ranges) {
            EXPECT_LT(range.start, range.end);
            EXPECT_GE(range.start, previous_end);
            EXPECT_TRUE(OnCodepointBoundary(text, range.start));
            EXPECT_TRUE(OnCodepointBoundary(text, range.end));
            previous_end = range.end;
        }
        for (const auto& token_ranges : sentence.token_source_ranges) {
            ASSERT_FALSE(token_ranges.empty());
            std::size_t previous_token_end = 0;
            for (const auto& range : token_ranges) {
                EXPECT_LT(range.start, range.end);
                EXPECT_GE(range.start, previous_token_end);
                EXPECT_TRUE(OnCodepointBoundary(text, range.start));
                EXPECT_TRUE(OnCodepointBoundary(text, range.end));
                previous_token_end = range.end;
            }
        }
    }
}

TEST(SourceMapping, AsciiTokensMapToExactByteRanges)
{
    const std::string text = "Katten sov. Hunden sov.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 2u);
    ExpectValidMapping(text, sentences);

    const std::vector<std::vector<Ranges>> expected_tokens{
        {{{0, 6}}, {{7, 10}}, {{10, 11}}},
        {{{12, 18}}, {{19, 22}}, {{22, 23}}},
    };
    for (std::size_t sentence = 0; sentence < sentences.size(); ++sentence) {
        EXPECT_EQ(sentences[sentence].token_source_ranges, expected_tokens[sentence]);
    }
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 11}}));
    EXPECT_EQ(sentences[1].source_ranges, (Ranges{{12, 23}}));
}

TEST(SourceMapping, LeadingAndTrailingWhitespaceShiftsRanges)
{
    const std::string text = "  Katten sov.  ";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{2, 8}}, {{9, 12}}, {{12, 13}}}));
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{2, 13}}));
}

TEST(SourceMapping, NorwegianMultibyteLettersCountBytes)
{
    // å and æ occupy two UTF-8 bytes each; the offsets count bytes, not
    // characters.
    const std::string text = "Blåbær smaker godt.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].tokens[0], "Blåbær");
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 8}}, {{9, 15}}, {{16, 20}}, {{20, 21}}}));
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 21}}));
}

TEST(SourceMapping, ConsecutiveMultibyteLettersFormOneToken)
{
    const std::string text = "æøå er bokstaver.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].tokens[0], "æøå");
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 6}}, {{7, 9}}, {{10, 19}}, {{19, 20}}}));
}

TEST(SourceMapping, EmojiBeforeAndBetweenTokens)
{
    const std::string before = "🙂 Katten sov.";
    auto sentences = Segment(before, NorwegianPolicy());
    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(before, sentences);
    EXPECT_EQ(sentences[0].tokens[0], "🙂");
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 4}}, {{5, 11}}, {{12, 15}}, {{15, 16}}}));

    const std::string between = "Katten 🙂 sov.";
    sentences = Segment(between, NorwegianPolicy());
    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(between, sentences);
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 6}}, {{7, 11}}, {{12, 15}}, {{15, 16}}}));
}

TEST(SourceMapping, DecomposedCombiningMarkStaysOnCodepointBoundaries)
{
    // "a" plus combining ring (U+030A) is visually "å" but differently
    // encoded; the scanner works on codepoints, so the combining mark
    // becomes its own token and every boundary stays a codepoint boundary.
    const std::string text = "a\xCC\x8A er fin.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 1}}, {{1, 3}}, {{4, 6}}, {{7, 10}}, {{10, 11}}}));
}

TEST(SourceMapping, RepeatedIdenticalTokensMapToDistinctOccurrences)
{
    // A find()-style reconstruction would collapse onto the first
    // occurrence; the carried mapping assigns each repetition its own bytes.
    const std::string text = "ja ja ja ja.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 2}}, {{3, 5}}, {{6, 8}}, {{9, 11}}, {{11, 12}}}));
}

TEST(SourceMapping, RepeatedIdenticalSentencesMapToDistinctOccurrences)
{
    const std::string text = "Han sov. Han sov.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 2u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 8}}));
    EXPECT_EQ(sentences[1].source_ranges, (Ranges{{9, 17}}));
    EXPECT_EQ(sentences[0].token_source_ranges[0], (Ranges{{0, 3}}));
    EXPECT_EQ(sentences[1].token_source_ranges[0], (Ranges{{9, 12}}));
}

TEST(SourceMapping, RestoredSentenceSpaceKeepsOriginalOffsets)
{
    // "veien.Et" misses a space; the repair splits the sentences without
    // shifting any offset: the restored space owns no input bytes.
    const std::string text = "Han går langs veien.Et sekund senere står han.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 2u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 21}}));
    EXPECT_EQ(sentences[1].source_ranges, (Ranges{{21, 48}}));
    // "Et" starts directly after the period of the first sentence.
    EXPECT_EQ(sentences[1].tokens[0], "Et");
    EXPECT_EQ(sentences[1].token_source_ranges[0], (Ranges{{21, 23}}));
}

TEST(SourceMapping, MultipleRepairedBoundariesStayAligned)
{
    const std::string text = "De gikk.De kom.De sov.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 3u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 8}}));
    EXPECT_EQ(sentences[1].source_ranges, (Ranges{{8, 15}}));
    EXPECT_EQ(sentences[2].source_ranges, (Ranges{{15, 22}}));
}

TEST(SourceMapping, WrappedLineMergesAcrossNewlineWhitespace)
{
    // The merged line wrap is whitespace in the original, so the sentence
    // stays one covering range and the tokens straddle the newline.
    const std::string text = "Katten\nhennes sov.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].tokens,
        (std::vector<std::string>{"Katten", "hennes", "sov", "."}));
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 6}}, {{7, 13}}, {{14, 17}}, {{17, 18}}}));
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 18}}));
}

TEST(SourceMapping, CollapsedWhitespaceRunsKeepTokenOffsets)
{
    const std::string text = "Hun   sov.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 3}}, {{6, 9}}, {{9, 10}}}));
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 10}}));
}

TEST(SourceMapping, DehyphenatedLineWrapKeepsBothFragments)
{
    // The de-hyphenated model token stays "språkmodellen", but its source
    // mapping must point at the two contributing fragments — never at a
    // single invented range that would claim "-\n" as token content.
    const std::string text = "Dette er språk-\nmodellen til laget.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    ASSERT_EQ(sentences[0].tokens[2], "språkmodellen");
    EXPECT_EQ(sentences[0].token_source_ranges[2], (Ranges{{9, 15}, {17, 25}}));
    EXPECT_EQ(text.substr(9, 6), "språk");
    EXPECT_EQ(text.substr(17, 8), "modellen");
    // No token range may cover the removed "-\n".
    for (const auto& token_ranges : sentences[0].token_source_ranges) {
        for (const auto& range : token_ranges) {
            EXPECT_TRUE(range.end <= 15 || range.start >= 17);
        }
    }
    // The sentence splits at the removed hyphen instead of bridging it.
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 15}, {17, 36}}));
}

TEST(SourceMapping, DottedAbbreviationStaysOneContiguousRange)
{
    const std::string text = "Vi har f.eks. kake.";
    const auto sentences = Segment(text, NorwegianPolicy());

    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(text, sentences);
    EXPECT_EQ(sentences[0].tokens[2], "f.eks.");
    EXPECT_EQ(sentences[0].token_source_ranges[2], (Ranges{{7, 13}}));
}

TEST(SourceMapping, UrlAndEmailTokensMapAsSingleRanges)
{
    const std::string url_text = "Se https://prism.no i dag.";
    auto sentences = Segment(url_text, NorwegianPolicy());
    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(url_text, sentences);
    EXPECT_EQ(sentences[0].tokens[1], "https://prism.no");
    EXPECT_EQ(sentences[0].token_source_ranges[1], (Ranges{{3, 19}}));

    const std::string email_text = "Skriv til post@prism.no i dag.";
    sentences = Segment(email_text, NorwegianPolicy());
    ASSERT_EQ(sentences.size(), 1u);
    ExpectValidMapping(email_text, sentences);
    EXPECT_EQ(sentences[0].tokens[2], "post@prism.no");
    EXPECT_EQ(sentences[0].token_source_ranges[2], (Ranges{{10, 23}}));
}

TEST(SourceMapping, ChunkingSlicesTokenRangesAndClipsSentenceRanges)
{
    std::string text;
    for (int index = 0; index < 19; ++index) {
        if (index > 0) {
            text += ' ';
        }
        text += "ord" + std::to_string(index);
    }
    text += '.';

    const auto sentences = Segment(text, TestPolicy());

    ASSERT_EQ(sentences.size(), 3u);
    ExpectValidMapping(text, sentences);
    // Chunk 2 starts at "ord8"; its ranges continue seamlessly where
    // chunk 1 ended, and each chunk's sentence range covers only its own
    // tokens.
    EXPECT_EQ(sentences[1].tokens[0], "ord8");
    EXPECT_EQ(sentences[1].token_source_ranges[0], (Ranges{{40, 44}}));
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 39}}));
    EXPECT_EQ(sentences[1].source_ranges, (Ranges{{40, 85}}));
    EXPECT_EQ(sentences[2].source_ranges, (Ranges{{86, 104}}));
}

TEST(SourceMapping, ChunkHelperSlicesCallerProvidedRanges)
{
    PretokenizedSentence sentence;
    sentence.tokens = {"a", "b", "c"};
    sentence.has_space_before = {false, true, true};
    sentence.token_source_ranges = {{{0, 1}}, {{2, 3}}, {{4, 5}}};
    sentence.source_ranges = {{0, 5}};

    const auto chunks = prism::segmentation::Chunk(sentence, 2);

    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0].token_source_ranges,
        (std::vector<Ranges>{{{0, 1}}, {{2, 3}}}));
    EXPECT_EQ(chunks[0].source_ranges, (Ranges{{0, 3}}));
    EXPECT_EQ(chunks[1].token_source_ranges, (std::vector<Ranges>{{{4, 5}}}));
    EXPECT_EQ(chunks[1].source_ranges, (Ranges{{4, 5}}));
}

TEST(SourceMapping, EveryFixtureTokenStaysAnchored)
{
    for (const auto* fixture : {"skarvholmen-bokmaal", "fjellvatnet-nynorsk"}) {
        std::ifstream file(std::string(PRISM_REPOSITORY_ROOT) + "/data/examples/"
                + fixture + ".txt",
            std::ios::binary);
        ASSERT_TRUE(file) << "Checked-in fixture is missing.";
        std::stringstream buffer;
        buffer << file.rdbuf();
        const auto text = buffer.str();

        const auto sentences = Segment(text, NorwegianPolicy());
        ExpectValidMapping(text, sentences);

        // Document order: sentence ranges never move backwards.
        std::size_t previous_start = 0;
        for (const auto& sentence : sentences) {
            EXPECT_GE(sentence.source_ranges.front().start, previous_start);
            previous_start = sentence.source_ranges.front().start;
        }
    }
}

} // namespace
