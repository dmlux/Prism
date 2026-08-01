package dev.prism;

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
        Path artifact = Path.of(args[0]);
        if (!Files.exists(artifact.resolve("manifest.json"))) {
            System.out.println("SKIPPED: local artifact is not present.");
            return;
        }

        expectFailure();

        try (PrismTagger tagger = PrismTagger.load(artifact)) {
            referenceDecisions(tagger);
            pretokenizedBatches(tagger);
        }

        System.out.println("PASSED");
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
        }
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
