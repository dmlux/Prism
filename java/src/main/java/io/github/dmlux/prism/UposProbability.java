package io.github.dmlux.prism;

/**
 * One entry of a token's UPOS probability distribution: the calibrated
 * probability that the model assigns this label to the token.
 *
 * <p>See {@link TaggedToken#uposDistribution()}.
 */
public record UposProbability(String upos, double probability) {
}
