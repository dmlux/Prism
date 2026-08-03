package io.github.dmlux.prism;

import java.util.List;

/**
 * One tagged sentence in original token order.
 *
 * <p>{@code sourceRanges} covers every token fragment of the sentence in
 * the exact raw text passed to {@link PrismTagger#tagText}: fragments whose
 * gap in the original is pure whitespace share one range, gaps containing
 * removed non-whitespace content (for example the "-" of a joined line
 * wrap) split the sentence into several ranges. The list is empty for
 * pretokenized input, which has no source positions.
 */
public record TaggedSentence(List<TaggedToken> tokens, List<Utf8ByteRange> sourceRanges) {

    /** Pre-source-mapping constructor; the sentence carries no source ranges. */
    public TaggedSentence(List<TaggedToken> tokens) {
        this(tokens, List.of());
    }
}
