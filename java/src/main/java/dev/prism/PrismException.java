package dev.prism;

/** Reported when the artifact cannot be loaded or tagging fails natively. */
public final class PrismException extends RuntimeException {
    public PrismException(String message) {
        super(message);
    }
}
