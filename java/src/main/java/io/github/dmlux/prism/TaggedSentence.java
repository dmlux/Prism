package io.github.dmlux.prism;

import java.util.List;

/** One tagged sentence in original token order. */
public record TaggedSentence(List<TaggedToken> tokens) {
}
