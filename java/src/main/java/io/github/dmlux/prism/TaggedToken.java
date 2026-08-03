package io.github.dmlux.prism;

import java.util.List;
import java.util.Map;

/**
 * One tagged token with calibrated confidences per decision.
 *
 * <p>Multi-valued morphology features report the smallest confidence among
 * their selected values; features whose decision is "not present" are
 * omitted from both maps. Feature names iterate alphabetically.
 *
 * <p>{@code sourceRanges} locates the token in the exact raw text passed to
 * {@link PrismTagger#tagText} as ordered, non-overlapping {@link
 * Utf8ByteRange UTF-8 byte ranges}. A token whose source is contiguous has
 * exactly one range; a token assembled from several separated input
 * fragments (for example a de-hyphenated line wrap) has one range per
 * fragment. {@code text} may differ from the bytes the ranges point to
 * after internal repairs — {@code text}, {@code hasSpaceBefore}, and
 * {@code sourceRanges} are three distinct pieces of information. The list
 * is empty for pretokenized input, which has no source positions.
 */
public record TaggedToken(
        String text,
        boolean hasSpaceBefore,
        String upos,
        double uposConfidence,
        Map<String, List<String>> features,
        Map<String, Double> featureConfidences,
        String lemma,
        double lemmaConfidence,
        List<Utf8ByteRange> sourceRanges) {

    /** Pre-source-mapping constructor; the token carries no source ranges. */
    public TaggedToken(
            String text,
            boolean hasSpaceBefore,
            String upos,
            double uposConfidence,
            Map<String, List<String>> features,
            Map<String, Double> featureConfidences,
            String lemma,
            double lemmaConfidence) {
        this(text, hasSpaceBefore, upos, uposConfidence, features,
                featureConfidences, lemma, lemmaConfidence, List.of());
    }
}
