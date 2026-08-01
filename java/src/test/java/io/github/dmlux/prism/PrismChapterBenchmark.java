package io.github.dmlux.prism;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * End-to-end chapter benchmark: tags a UTF-8 text file twice and reports
 * the cold run (including lazy program loading) and the warm run.
 *
 * <p>Arguments: artifact directory, text file. Point the artifact directory
 * at different manifest variants to compare program configurations.
 */
public final class PrismChapterBenchmark {

    private PrismChapterBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        String text = new String(
                Files.readAllBytes(Path.of(args[1])), StandardCharsets.UTF_8);
        try (PrismTagger tagger = PrismTagger.load(Path.of(args[0]))) {
            System.out.printf("cold: %.1f ms%n", millisecondsFor(tagger, text));
            System.out.printf("warm: %.1f ms%n", millisecondsFor(tagger, text));
        }
    }

    private static double millisecondsFor(PrismTagger tagger, String text) {
        long begin = System.nanoTime();
        List<TaggedSentence> tagged = tagger.tagText(text);
        double milliseconds = (System.nanoTime() - begin) / 1e6;
        int tokens = tagged.stream().mapToInt(sentence -> sentence.tokens().size()).sum();
        System.out.println("tokens: " + tokens);
        return milliseconds;
    }
}
