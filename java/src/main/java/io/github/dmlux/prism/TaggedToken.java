package io.github.dmlux.prism;

import java.util.List;
import java.util.Map;

/**
 * One tagged token with calibrated confidences per decision.
 *
 * <p>Multi-valued morphology features report the smallest confidence among
 * their selected values; features whose decision is "not present" are
 * omitted from both maps. Feature names iterate alphabetically.
 */
public record TaggedToken(
        String text,
        boolean hasSpaceBefore,
        String upos,
        double uposConfidence,
        Map<String, List<String>> features,
        Map<String, Double> featureConfidences,
        String lemma,
        double lemmaConfidence) {
}
