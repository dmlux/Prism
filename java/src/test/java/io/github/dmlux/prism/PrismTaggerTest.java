package io.github.dmlux.prism;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * End-to-end validation against decisions recorded from the reference
 * tagger on the same frozen artifact.
 *
 * <p>Deliberately a plain {@code main} program without a test framework,
 * so the Java binding carries no dependencies; the CMake test suite runs
 * it with the artifact directory as the only argument.
 */
public final class PrismTaggerTest {

    private PrismTaggerTest() {
    }

    public static void main(String[] args) {
        utf8ByteRangeContract();

        Path artifact = Path.of(args[0]);
        if (!Files.exists(artifact.resolve("manifest.json"))) {
            System.out.println("SKIPPED: local artifact is not present.");
            return;
        }

        expectFailure();

        try (PrismTagger tagger = PrismTagger.load(artifact)) {
            referenceDecisions(tagger);
            sourceRanges(tagger);
            artifactMetadata(tagger);
            pretokenizedBatches(tagger);
        }

        System.out.println("PASSED");
    }

    /** Pure-Java contract of the source-position type; no artifact needed. */
    private static void utf8ByteRangeContract() {
        // UTF-8 byte offsets are not UTF-16 code units: in "🙂å" the å
        // starts at UTF-8 byte 4 but at Java char index 2.
        String text = "🙂å ok";
        check(Utf8ByteRange.utf16OffsetOf(text, 0) == 0, "offset 0");
        check(Utf8ByteRange.utf16OffsetOf(text, 4) == 2, "å starts at char 2");
        check(Utf8ByteRange.utf16OffsetOf(text, 6) == 3, "space starts at char 3");
        try {
            Utf8ByteRange.utf16OffsetOf(text, 2);
            throw new AssertionError("Mid-codepoint offsets must be rejected.");
        } catch (IllegalArgumentException expected) {
            // rejected, not rounded
        }
        try {
            new Utf8ByteRange(5, 5);
            throw new AssertionError("Empty ranges must be rejected.");
        } catch (IllegalArgumentException expected) {
            // a range is never empty
        }
    }

    private static void expectFailure() {
        try (PrismTagger tagger = PrismTagger.load(Path.of("/nonexistent/artifact"))) {
            throw new AssertionError("Loading a nonexistent artifact must fail.");
        } catch (PrismException expected) {
            check(!expected.getMessage().isEmpty(), "error message is empty");
        }
    }

    private static void referenceDecisions(PrismTagger tagger) {
        List<TaggedSentence> sentences =
                tagger.tagText("Hun kjøpte tre gamle bøker den 17. mai.");

        check(sentences.size() == 1, "expected one sentence");
        List<TaggedToken> tokens = sentences.get(0).tokens();
        String[] expectedTexts = {
            "Hun", "kjøpte", "tre", "gamle", "bøker", "den", "17.", "mai", "."
        };
        String[] expectedUpos = {
            "PRON", "VERB", "NUM", "ADJ", "NOUN", "DET", "ADJ", "NOUN", "PUNCT"
        };
        String[] expectedLemmas = {
            "hun", "kjøpe", "tre", "gammel", "bok", "den", "17.", "mai", "."
        };
        check(tokens.size() == expectedTexts.length, "unexpected token count");
        for (int index = 0; index < tokens.size(); index++) {
            TaggedToken token = tokens.get(index);
            check(token.text().equals(expectedTexts[index]), "text " + token.text());
            check(token.upos().equals(expectedUpos[index]), "upos " + token.upos());
            check(token.lemma().equals(expectedLemmas[index]), "lemma " + token.lemma());
            check(token.uposConfidence() > 0.9, "upos confidence");
            check(token.lemmaConfidence() > 0.9, "lemma confidence");
        }
        TaggedToken boker = tokens.get(4);
        check(List.of("Fem").equals(boker.features().get("Gender")), "Gender");
        check(List.of("Plur").equals(boker.features().get("Number")), "Number");
        check(boker.featureConfidences().get("Gender") > 0.5, "Gender confidence");
        check(boker.hasSpaceBefore(), "bøker has a space before it");
        check(!tokens.get(0).hasSpaceBefore(), "first token has no space before it");
    }

    private static void sourceRanges(PrismTagger tagger) {
        String text = "Hun kjøpte tre gamle bøker den 17. mai.";
        List<TaggedSentence> sentences = tagger.tagText(text);

        check(sentences.size() == 1, "expected one sentence");
        // Byte offsets shared with the C++ and Swift suites (parity): ø
        // occupies two UTF-8 bytes.
        long[][] expected = {
            {0, 3}, {4, 11}, {12, 15}, {16, 21}, {22, 28},
            {29, 32}, {33, 36}, {37, 40}, {40, 41},
        };
        List<TaggedToken> tokens = sentences.get(0).tokens();
        check(tokens.size() == expected.length, "unexpected token count");
        for (int index = 0; index < tokens.size(); index++) {
            List<Utf8ByteRange> ranges = tokens.get(index).sourceRanges();
            check(ranges.size() == 1, "expected one range per token");
            check(ranges.get(0).start() == expected[index][0], "start " + index);
            check(ranges.get(0).end() == expected[index][1], "end " + index);
        }
        check(
                List.of(new Utf8ByteRange(0, 41)).equals(sentences.get(0).sourceRanges()),
                "sentence range");

        // The conversion helper maps the UTF-8 offsets onto Java string
        // indices of the exact original text.
        Utf8ByteRange boker = tokens.get(4).sourceRanges().get(0);
        String slice = text.substring(
                Utf8ByteRange.utf16OffsetOf(text, boker.start()),
                Utf8ByteRange.utf16OffsetOf(text, boker.end()));
        check("bøker".equals(slice), "mapped slice " + slice);
    }

    private static void artifactMetadata(PrismTagger tagger) {
        check("prism-no".equals(tagger.artifactName()), "artifact name");
        check("0.2.3".equals(tagger.artifactVersion()), "artifact version");
        check(List.of("nb", "nn", "no").equals(tagger.languageTags()), "language tags");
    }

    private static void pretokenizedBatches(PrismTagger tagger) {
        List<List<String>> sentence = java.util.Collections.nCopies(
                11, List.of("Katten", "sov", "."));

        List<TaggedSentence> tagged = tagger.tagPretokenized(sentence);

        check(tagged.size() == 11, "expected eleven sentences");
        for (TaggedSentence entry : tagged) {
            check(entry.tokens().get(0).upos().equals("NOUN"), "Katten upos");
            check(entry.tokens().get(0).lemma().equals("katt"), "Katten lemma");
            check(entry.tokens().get(1).lemma().equals("sove"), "sov lemma");
            check(entry.tokens().get(2).upos().equals("PUNCT"), "period upos");
            // Pretokenized input has no source positions; Prism never
            // invents ranges.
            check(entry.sourceRanges().isEmpty(), "sentence ranges empty");
            check(entry.tokens().get(0).sourceRanges().isEmpty(), "token ranges empty");
        }
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
