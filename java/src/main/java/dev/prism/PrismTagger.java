package dev.prism;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Frozen-artifact tagger: raw text or word tokens in, decisions plus
 * calibrated confidences out.
 *
 * <p>The heavy lifting — segmentation, subword tokenization, fixed-shape
 * batching across the artifact's programs, and decoding — runs in the
 * native Prism core; this class marshals text in and results out.
 *
 * <p>Input is expected in Unicode NFC (the artifact's recorded
 * normalization). Instances are not thread-safe; results are immutable.
 *
 * <pre>{@code
 * try (PrismTagger tagger = PrismTagger.load(Path.of("models/prism-no-0.2.0"))) {
 *     for (TaggedSentence sentence : tagger.tagText("Hun kjøpte tre gamle bøker.")) {
 *         for (TaggedToken token : sentence.tokens()) {
 *             System.out.println(token.text() + " " + token.upos() + " " + token.lemma());
 *         }
 *     }
 * }
 * }</pre>
 */
public final class PrismTagger implements AutoCloseable {

    private static volatile boolean nativeLibraryLoaded = false;

    private long handle;

    private PrismTagger(long handle) {
        this.handle = handle;
    }

    /**
     * Loads the native library from an explicit path (for example
     * {@code libprism_jni.dylib} or {@code libprism_jni.so}). Optional:
     * without this call, {@link #load(Path)} resolves {@code prism_jni}
     * through {@code java.library.path}.
     */
    public static synchronized void loadNativeLibrary(Path library) {
        System.load(library.toAbsolutePath().toString());
        nativeLibraryLoaded = true;
    }

    /** Opens the artifact directory (manifest, labels, vocabulary, programs). */
    public static PrismTagger load(Path artifactDirectory) {
        ensureNativeLibrary();
        long handle = nativeCreate(utf8(artifactDirectory.toAbsolutePath().toString()));
        return new PrismTagger(handle);
    }

    /** Segments raw text with the runtime policy, then tags every sentence. */
    public List<TaggedSentence> tagText(String text) {
        return assemble(nativeTagText(requireHandle(), utf8(text)));
    }

    /** Tags application-supplied word tokens (space assumed between words). */
    public List<TaggedSentence> tagPretokenized(List<List<String>> sentences) {
        int total = sentences.stream().mapToInt(List::size).sum();
        byte[][] tokens = new byte[total][];
        int[] tokensPerSentence = new int[sentences.size()];
        int index = 0;
        for (int sentence = 0; sentence < sentences.size(); sentence++) {
            tokensPerSentence[sentence] = sentences.get(sentence).size();
            for (String token : sentences.get(sentence)) {
                tokens[index++] = utf8(token);
            }
        }
        return assemble(nativeTagTokens(requireHandle(), tokens, tokensPerSentence));
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            nativeDestroy(handle);
            handle = 0;
        }
    }

    private static void ensureNativeLibrary() {
        if (!nativeLibraryLoaded) {
            synchronized (PrismTagger.class) {
                if (!nativeLibraryLoaded) {
                    System.loadLibrary("prism_jni");
                    nativeLibraryLoaded = true;
                }
            }
        }
    }

    private long requireHandle() {
        if (handle == 0) {
            throw new PrismException("The tagger has been closed.");
        }
        return handle;
    }

    private static byte[] utf8(String text) {
        return text.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * The native side returns the batch as flat parallel arrays (one JNI
     * transition, no per-token callbacks); this unflattens them into records.
     */
    private static List<TaggedSentence> assemble(Object[] payload) {
        int[] tokensPerSentence = (int[]) payload[0];
        String[] texts = (String[]) payload[1];
        boolean[] hasSpaceBefore = (boolean[]) payload[2];
        String[] upos = (String[]) payload[3];
        double[] uposConfidences = (double[]) payload[4];
        int[] featureCounts = (int[]) payload[5];
        String[] featureNames = (String[]) payload[6];
        String[] featureValues = (String[]) payload[7];
        double[] featureConfidences = (double[]) payload[8];
        String[] lemmas = (String[]) payload[9];
        double[] lemmaConfidences = (double[]) payload[10];

        List<TaggedSentence> sentences = new ArrayList<>(tokensPerSentence.length);
        int token = 0;
        int feature = 0;
        for (int count : tokensPerSentence) {
            List<TaggedToken> tokens = new ArrayList<>(count);
            for (int position = 0; position < count; position++, token++) {
                Map<String, List<String>> features = new LinkedHashMap<>();
                Map<String, Double> confidences = new LinkedHashMap<>();
                for (int entry = 0; entry < featureCounts[token]; entry++, feature++) {
                    features.put(
                            featureNames[feature],
                            List.of(featureValues[feature].split(",")));
                    confidences.put(featureNames[feature], featureConfidences[feature]);
                }
                // Collections.unmodifiableMap keeps the alphabetical
                // feature order; Map.copyOf would not.
                tokens.add(new TaggedToken(
                        texts[token],
                        hasSpaceBefore[token],
                        upos[token],
                        uposConfidences[token],
                        Collections.unmodifiableMap(features),
                        Collections.unmodifiableMap(confidences),
                        lemmas[token],
                        lemmaConfidences[token]));
            }
            sentences.add(new TaggedSentence(List.copyOf(tokens)));
        }
        return List.copyOf(sentences);
    }

    private static native long nativeCreate(byte[] utf8Directory);

    private static native void nativeDestroy(long handle);

    private static native Object[] nativeTagText(long handle, byte[] utf8Text);

    private static native Object[] nativeTagTokens(
            long handle, byte[][] utf8Tokens, int[] tokensPerSentence);
}
