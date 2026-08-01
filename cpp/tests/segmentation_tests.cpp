// Mirrors python/tests/test_runtime_segmentation.py and the Swift suite so
// all three implementations stay behaviourally identical, including the
// book-chapter reference counts when the local fixture is present.

#include "prism/segmentation.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

using prism::segmentation::NorwegianPolicy;
using prism::segmentation::Segment;
using prism::segmentation::SegmentationPolicy;

namespace {

int failures = 0;

void Check(bool condition, const std::string& label)
{
    if (!condition) {
        failures += 1;
        std::cerr << "FAILED: " << label << "\n";
    }
}

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

void TestKeepsFragmentsAndHeadings()
{
    const auto tokens = Tokens("KAPITTEL 1\nHan gjekk heim.\nog so vidare", TestPolicy());
    Check(tokens
            == std::vector<std::vector<std::string>>{
                {"KAPITTEL", "1"},
                {"Han", "gjekk", "heim", "."},
                {"og", "so", "vidare"},
            },
        "keeps fragments and headings");
}

void TestChunksLongSentencesWithoutLoss()
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
    Check(sentences.size() == 3, "chunk count");
    std::vector<std::string> recovered;
    for (const auto& sentence : sentences) {
        Check(!sentence.has_space_before.empty() && !sentence.has_space_before.front(),
            "chunk starts without space");
        recovered.insert(recovered.end(), sentence.tokens.begin(), sentence.tokens.end());
    }
    words.push_back(".");
    Check(recovered == words, "chunking loses no tokens");
}

void TestRestoresMissingSentenceSpaces()
{
    const auto tokens = Tokens(
        "De begynte å gå.De gikk fort.«Noe nytt?» spurte han om f.eks.Dette.", TestPolicy());
    Check(tokens
            == std::vector<std::vector<std::string>>{
                {"De", "begynte", "å", "gå", "."},
                {"De", "gikk", "fort", "."},
                {"«", "Noe", "nytt", "?", "»", "spurte", "han", "om"},
                {"f.eks.", "Dette", "."},
            },
        "restores missing sentence spaces");
}

void TestProtectsAbbreviationsAndOrdinals()
{
    const auto tokens = Tokens("Vi feirar 17. mai med f.eks. kake. Det er fint.", TestPolicy());
    Check(tokens
            == std::vector<std::vector<std::string>>{
                {"Vi", "feirar", "17.", "mai", "med", "f.eks.", "kake", "."},
                {"Det", "er", "fint", "."},
            },
        "protects abbreviations and ordinals");
}

void TestMergesWrappedLinesWithDehyphenation()
{
    const auto tokens = Tokens(
        "Katten\nhennes sov.\nDen hadde vand-\nring i blodet.", NorwegianPolicy(32));
    Check(tokens
            == std::vector<std::vector<std::string>>{
                {"Katten", "hennes", "sov", "."},
                {"Den", "hadde", "vandring", "i", "blodet", "."},
            },
        "merges wrapped lines with dehyphenation");
}

void TestSpacingReflectsAttachedPunctuation()
{
    const auto sentences = Segment("Hun sa «nei», ikke sant?", NorwegianPolicy());
    Check(sentences.size() == 1, "one sentence");
    Check(sentences[0].tokens
            == std::vector<std::string>{"Hun", "sa", "«", "nei", "»", ",", "ikke", "sant", "?"},
        "attached punctuation tokens");
    Check(sentences[0].has_space_before
            == std::vector<bool>{false, true, true, false, false, false, true, true, false},
        "attached punctuation spacing");
}

void TestChapterParity(const std::string& repository_root)
{
    std::ifstream file(repository_root + "/data/examples/hp7kap1.txt", std::ios::binary);
    if (!file) {
        std::cout << "SKIP chapter parity: local fixture is not present\n";
        return;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const auto text = buffer.str();

    const auto started = std::chrono::steady_clock::now();
    const auto sentences = Segment(text, NorwegianPolicy());
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started);

    std::size_t token_count = 0;
    for (const auto& sentence : sentences) {
        token_count += sentence.tokens.size();
    }
    Check(sentences.size() == 247, "chapter sentence count (" + std::to_string(sentences.size()) + ")");
    Check(token_count == 3783, "chapter token count (" + std::to_string(token_count) + ")");
    std::cout << "chapter: " << sentences.size() << " sentences, " << token_count
              << " tokens in " << elapsed.count() / 1000.0 << " ms\n";
}

} // namespace

int main(int argc, char** argv)
{
    TestKeepsFragmentsAndHeadings();
    TestChunksLongSentencesWithoutLoss();
    TestRestoresMissingSentenceSpaces();
    TestProtectsAbbreviationsAndOrdinals();
    TestMergesWrappedLinesWithDehyphenation();
    TestSpacingReflectsAttachedPunctuation();
    TestChapterParity(argc > 1 ? argv[1] : "..");

    if (failures == 0) {
        std::cout << "all segmentation tests passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
