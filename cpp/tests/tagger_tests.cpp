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
const std::string kArtifact = kRoot + "/models/prism-no-0.2.3";

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

TEST_F(TaggerTest, RawTextResultsCarrySourceRanges)
{
    prism::tagger::Tagger tagger(kArtifact);
    const std::string text = "Hun kjøpte tre gamle bøker den 17. mai.";

    const auto sentences = tagger.TagText(text);

    ASSERT_EQ(sentences.size(), 1U);
    using Ranges = std::vector<prism::Utf8ByteRange>;
    // ø occupies two UTF-8 bytes; the offsets count bytes of the exact
    // input. These literals are shared with the Swift and Java suites to
    // pin byte-offset parity across the bindings.
    const std::vector<Ranges> expected{
        {{0, 3}}, {{4, 11}}, {{12, 15}}, {{16, 21}}, {{22, 28}},
        {{29, 32}}, {{33, 36}}, {{37, 40}}, {{40, 41}},
    };
    ASSERT_EQ(sentences[0].tokens.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_EQ(sentences[0].tokens[index].source_ranges, expected[index]);
    }
    EXPECT_EQ(sentences[0].source_ranges, (Ranges{{0, 41}}));
    // The mapped bytes reproduce the original token spelling.
    const auto& boker = sentences[0].tokens[4].source_ranges.front();
    EXPECT_EQ(text.substr(boker.start, boker.end - boker.start), "bøker");
}

TEST_F(TaggerTest, BatchSortingKeepsIdenticalSentencesAnchored)
{
    prism::tagger::Tagger tagger(kArtifact);
    // Twenty identical sentences force several batches and length-sorted
    // reordering; every result must still point at its own occurrence.
    std::string text;
    for (int index = 0; index < 20; ++index) {
        if (index > 0) {
            text += ' ';
        }
        text += "Katten sov.";
    }

    const auto sentences = tagger.TagText(text);

    ASSERT_EQ(sentences.size(), 20U);
    for (std::size_t index = 0; index < sentences.size(); ++index) {
        const std::size_t base = index * 12;
        ASSERT_EQ(sentences[index].source_ranges.size(), 1U);
        EXPECT_EQ(sentences[index].source_ranges[0], (prism::Utf8ByteRange{base, base + 11}));
        EXPECT_EQ(sentences[index].tokens[0].source_ranges[0],
            (prism::Utf8ByteRange{base, base + 6}));
    }
}

TEST_F(TaggerTest, PretokenizedInputCarriesNoSourceRanges)
{
    prism::tagger::Tagger tagger(kArtifact);

    const auto tagged = tagger.TagPretokenized({{"Katten", "sov", "."}});

    ASSERT_EQ(tagged.size(), 1U);
    EXPECT_TRUE(tagged[0].source_ranges.empty());
    for (const auto& token : tagged[0].tokens) {
        EXPECT_TRUE(token.source_ranges.empty());
    }
}

TEST_F(TaggerTest, CallerProvidedSourceRangesPassThroughValidated)
{
    prism::tagger::Tagger tagger(kArtifact);

    prism::segmentation::PretokenizedSentence sentence;
    sentence.tokens = {"Katten", "sov", "."};
    sentence.has_space_before = {false, true, false};
    sentence.token_source_ranges = {{{0, 6}}, {{7, 10}}, {{10, 11}}};
    sentence.source_ranges = {{0, 11}};

    const auto tagged = tagger.Tag({sentence});

    ASSERT_EQ(tagged.size(), 1U);
    EXPECT_EQ(tagged[0].source_ranges, sentence.source_ranges);
    EXPECT_EQ(tagged[0].tokens[0].source_ranges, sentence.token_source_ranges[0]);
    EXPECT_EQ(tagged[0].tokens[2].source_ranges, sentence.token_source_ranges[2]);

    // Overlapping, empty, or miscounted ranges are rejected up front.
    auto overlapping = sentence;
    overlapping.token_source_ranges = {{{0, 6}}, {{5, 10}}, {{10, 11}}};
    EXPECT_THROW(tagger.Tag({overlapping}), std::invalid_argument);

    auto empty_range = sentence;
    empty_range.token_source_ranges = {{{0, 6}}, {{7, 7}}, {{10, 11}}};
    EXPECT_THROW(tagger.Tag({empty_range}), std::invalid_argument);

    auto miscounted = sentence;
    miscounted.token_source_ranges = {{{0, 6}}, {{7, 10}}};
    EXPECT_THROW(tagger.Tag({miscounted}), std::invalid_argument);
}

TEST_F(TaggerTest, ExposesArtifactMetadata)
{
    prism::tagger::Tagger tagger(kArtifact);

    EXPECT_EQ(tagger.artifact().name(), "prism-no");
    EXPECT_EQ(tagger.artifact().version(), "0.2.3");
    EXPECT_EQ(tagger.artifact().language_tags(),
        (std::vector<std::string>{"nb", "nn", "no"}));

    // The label inventories mirror labels.json for API consumers.
    const auto& labels = tagger.artifact().labels();
    EXPECT_EQ(labels.upos_labels.size(), 17U);
    EXPECT_NE(std::find(labels.upos_labels.begin(), labels.upos_labels.end(), "NOUN"),
        labels.upos_labels.end());
    EXPECT_EQ(labels.features.size(), 18U);
}

TEST_F(TaggerTest, ReportsTheUposDistributionPerToken)
{
    prism::tagger::Tagger tagger(kArtifact);

    const auto sentences = tagger.TagPretokenized({{"Katten", "sov", "."}});
    ASSERT_EQ(sentences.size(), 1U);
    for (const auto& token : sentences[0].tokens) {
        const auto& distribution = token.upos_distribution;
        // One entry per artifact UPOS label, sorted by descending
        // probability; the first entry is the reported decision.
        ASSERT_EQ(distribution.size(), tagger.artifact().labels().upos_labels.size());
        EXPECT_EQ(distribution[0].upos, token.upos);
        EXPECT_DOUBLE_EQ(distribution[0].probability, token.upos_confidence);
        double sum = 0.0;
        for (std::size_t entry = 0; entry < distribution.size(); ++entry) {
            if (entry > 0) {
                EXPECT_LE(distribution[entry].probability,
                    distribution[entry - 1].probability);
            }
            sum += distribution[entry].probability;
        }
        EXPECT_NEAR(sum, 1.0, 1e-3);
    }
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

TEST_F(TaggerTest, TagsExampleTextsEndToEnd)
{
    // The checked-in CC0 example texts (see data/examples/README.md) with
    // the Python reference implementation's sentence and token counts.
    prism::tagger::Tagger tagger(kArtifact);

    const struct {
        const char* fixture;
        std::size_t sentences;
        std::size_t tokens;
    } expectations[] = {
        {"skarvholmen-bokmaal", 55, 905},
        {"fjellvatnet-nynorsk", 41, 803},
    };
    for (const auto& expected : expectations) {
        std::ifstream text_file(
            kRoot + "/data/examples/" + expected.fixture + ".txt", std::ios::binary);
        ASSERT_TRUE(text_file) << "Checked-in fixture is missing.";
        std::ostringstream buffer;
        buffer << text_file.rdbuf();

        const auto tagged = tagger.TagText(buffer.str());

        EXPECT_EQ(tagged.size(), expected.sentences) << expected.fixture;
        std::size_t token_count = 0;
        for (const auto& sentence : tagged) {
            token_count += sentence.tokens.size();
        }
        EXPECT_EQ(token_count, expected.tokens) << expected.fixture;
    }
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

    // Source ranges: half-open UTF-8 byte ranges against the exact input.
    ASSERT_EQ(prism_result_sentence_source_range_count(result, 0), 1U);
    EXPECT_EQ(prism_result_sentence_source_range_start(result, 0, 0), 0U);
    EXPECT_EQ(prism_result_sentence_source_range_end(result, 0, 0), 41U);
    ASSERT_EQ(prism_result_token_source_range_count(result, 0, 4), 1U);
    EXPECT_EQ(prism_result_token_source_range_start(result, 0, 4, 0), 22U);
    EXPECT_EQ(prism_result_token_source_range_end(result, 0, 4, 0), 28U);

    // Out-of-range access degrades to NULL/0 instead of aborting.
    EXPECT_EQ(prism_result_token_text(result, 0, 99), nullptr);
    EXPECT_EQ(prism_result_token_count(result, 99), 0U);
    EXPECT_EQ(prism_result_sentence_source_range_count(result, 99), 0U);
    EXPECT_EQ(prism_result_sentence_source_range_start(result, 0, 99), 0U);
    EXPECT_EQ(prism_result_sentence_source_range_end(result, 99, 0), 0U);
    EXPECT_EQ(prism_result_token_source_range_count(result, 0, 99), 0U);
    EXPECT_EQ(prism_result_token_source_range_start(result, 0, 4, 99), 0U);
    EXPECT_EQ(prism_result_token_source_range_end(result, 99, 4, 0), 0U);
    EXPECT_EQ(prism_result_sentence_source_range_count(nullptr, 0), 0U);
    EXPECT_EQ(prism_result_token_source_range_count(nullptr, 0, 0), 0U);

    // Artifact metadata from manifest.json, valid for the tagger lifetime.
    // Since 0.2.3 the manifest also declares the BCP 47 macrolanguage "no".
    EXPECT_STREQ(prism_tagger_artifact_name(tagger), "prism-no");
    EXPECT_STREQ(prism_tagger_artifact_version(tagger), "0.2.3");
    ASSERT_EQ(prism_tagger_language_tag_count(tagger), 3U);
    EXPECT_STREQ(prism_tagger_language_tag(tagger, 0), "nb");
    EXPECT_STREQ(prism_tagger_language_tag(tagger, 1), "nn");
    EXPECT_STREQ(prism_tagger_language_tag(tagger, 2), "no");
    EXPECT_EQ(prism_tagger_language_tag(tagger, 3), nullptr);

    // Label inventories mirrored from labels.json.
    const auto upos_labels = prism_tagger_upos_label_count(tagger);
    ASSERT_EQ(upos_labels, 17U);
    EXPECT_NE(prism_tagger_upos_label(tagger, 0), nullptr);
    EXPECT_EQ(prism_tagger_upos_label(tagger, upos_labels), nullptr);
    ASSERT_EQ(prism_tagger_feature_count(tagger), 18U);
    EXPECT_NE(prism_tagger_feature_name(tagger, 0), nullptr);
    ASSERT_GT(prism_tagger_feature_value_count(tagger, 0), 0U);
    EXPECT_NE(prism_tagger_feature_value(tagger, 0, 0), nullptr);
    EXPECT_EQ(prism_tagger_feature_value(tagger, 0, 9999), nullptr);
    EXPECT_EQ(prism_tagger_upos_label_count(nullptr), 0U);
    EXPECT_EQ(prism_tagger_feature_name(nullptr, 0), nullptr);

    // The per-token UPOS distribution: descending (label, probability)
    // entries whose first element is the reported decision.
    const auto entries = prism_result_token_upos_probability_count(result, 0, 0);
    ASSERT_EQ(entries, upos_labels);
    EXPECT_STREQ(prism_result_token_upos_probability_label(result, 0, 0, 0),
        prism_result_token_upos(result, 0, 0));
    EXPECT_DOUBLE_EQ(prism_result_token_upos_probability(result, 0, 0, 0),
        prism_result_token_upos_confidence(result, 0, 0));
    double distribution_sum = 0.0;
    for (size_t entry = 0; entry < entries; ++entry) {
        distribution_sum += prism_result_token_upos_probability(result, 0, 0, entry);
    }
    EXPECT_NEAR(distribution_sum, 1.0, 1e-3);
    EXPECT_EQ(prism_result_token_upos_probability_label(result, 0, 0, entries), nullptr);
    EXPECT_DOUBLE_EQ(prism_result_token_upos_probability(result, 0, 0, entries), 0.0);
    EXPECT_EQ(prism_result_token_upos_probability_count(nullptr, 0, 0), 0U);
    EXPECT_EQ(prism_tagger_artifact_name(nullptr), nullptr);
    EXPECT_EQ(prism_tagger_language_tag_count(nullptr), 0U);

    prism_result_destroy(result);

    const char* pretokenized[] = {"Katten", "sov", "."};
    prism_result* tokens_result = prism_tagger_tag_tokens(tagger, pretokenized, 3);
    ASSERT_NE(tokens_result, nullptr) << prism_last_error();
    ASSERT_EQ(prism_result_sentence_count(tokens_result), 1U);
    EXPECT_STREQ(prism_result_token_upos(tokens_result, 0, 0), "NOUN");
    EXPECT_STREQ(prism_result_token_lemma(tokens_result, 0, 1), "sove");
    // Pretokenized input has no source positions: counts are zero.
    EXPECT_EQ(prism_result_sentence_source_range_count(tokens_result, 0), 0U);
    EXPECT_EQ(prism_result_token_source_range_count(tokens_result, 0, 0), 0U);
    prism_result_destroy(tokens_result);

    prism_tagger_destroy(tagger);
}

// The fast (int8) artifact must reproduce the same reference decisions;
// quality is gated on the development split at export time, and this
// test pins the end-to-end runtime behaviour.
TEST(TaggerFast, TagsRawTextWithReferenceDecisions)
{
    const auto artifact = kRoot + "/models/prism-no-0.2.3-fast";
    if (!std::ifstream(artifact + "/manifest.json")) {
        GTEST_SKIP() << "Local fast artifact is not present.";
    }
    prism::tagger::Tagger tagger(artifact);

    const auto sentences = tagger.TagText("Hun kjøpte tre gamle bøker den 17. mai.");

    ASSERT_EQ(sentences.size(), 1U);
    const auto& tokens = sentences[0].tokens;
    const std::vector<std::string> expected_upos{
        "PRON", "VERB", "NUM", "ADJ", "NOUN", "DET", "ADJ", "NOUN", "PUNCT"};
    const std::vector<std::string> expected_lemmas{
        "hun", "kjøpe", "tre", "gammel", "bok", "den", "17.", "mai", "."};
    ASSERT_EQ(tokens.size(), expected_upos.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        EXPECT_EQ(tokens[index].upos, expected_upos[index]);
        EXPECT_EQ(tokens[index].lemma, expected_lemmas[index]);
        EXPECT_GT(tokens[index].upos_confidence, 0.9);
    }
}

TEST(TaggerCAbi, ReportsErrorsThroughLastError)
{
    prism_tagger* tagger = prism_tagger_create("/nonexistent/artifact");
    EXPECT_EQ(tagger, nullptr);
    EXPECT_NE(std::string(prism_last_error()).size(), 0U);
}

} // namespace
