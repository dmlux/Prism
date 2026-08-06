package io.github.dmlux.prism;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
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
    private Object[] artifactMetadata;

    private PrismTagger(long handle) {
        this.handle = handle;
    }

    /**
     * Loads the native library from an explicit path (for example
     * {@code libprism_jni.dylib} or {@code libprism_jni.so}). Optional:
     * without this call, {@link #load(Path)} first extracts a native
     * library embedded in the JAR for the current platform and falls back
     * to resolving {@code prism_jni} through {@code java.library.path}.
     */
    public static synchronized void loadNativeLibrary(Path library) {
        System.load(library.toAbsolutePath().toString());
        nativeLibraryLoaded = true;
    }

    /**
     * Overrides the CPU backend thread count for the whole process (the
     * tagger otherwise installs a measured default). Call before
     * {@link #load(Path)}; returns false when the pool cannot be resized.
     */
    public static boolean setThreadCount(int threadCount) {
        ensureNativeLibrary();
        return nativeSetThreadCount(threadCount);
    }

    /** Opens the artifact directory (manifest, labels, vocabulary, programs). */
    public static PrismTagger load(Path artifactDirectory) {
        ensureNativeLibrary();
        long handle = nativeCreate(utf8(artifactDirectory.toAbsolutePath().toString()));
        return new PrismTagger(handle);
    }

    /**
     * The artifact name recorded in the loaded manifest (for example
     * {@code "prism-no"}).
     */
    public String artifactName() {
        return (String) artifactMetadata()[0];
    }

    /** The artifact version recorded in the loaded manifest. */
    public String artifactVersion() {
        return (String) artifactMetadata()[1];
    }

    /**
     * The BCP 47 language tags the loaded artifact supports (currently for
     * example {@code "nb"} and {@code "nn"}), in manifest order. Decide
     * language support from these values, never from directory names.
     */
    public List<String> languageTags() {
        return List.of((String[]) artifactMetadata()[2]);
    }

    /**
     * Every UPOS tag the loaded artifact can assign, mirrored from its
     * label schema ({@code labels.json}). Inventories differ per language
     * artifact.
     */
    public List<String> uposLabels() {
        return List.of((String[]) artifactMetadata()[3]);
    }

    /**
     * Every morphology feature the loaded artifact can predict, mapped to
     * its possible values, in schema order. Inventories differ per
     * language artifact.
     */
    public Map<String, List<String>> morphologyFeatures() {
        Object[] metadata = artifactMetadata();
        String[] names = (String[]) metadata[4];
        int[] valueCounts = (int[]) metadata[5];
        String[] values = (String[]) metadata[6];
        Map<String, List<String>> features = new LinkedHashMap<>();
        int value = 0;
        for (int index = 0; index < names.length; index++) {
            List<String> featureValues = new ArrayList<>(valueCounts[index]);
            for (int entry = 0; entry < valueCounts[index]; entry++, value++) {
                featureValues.add(values[value]);
            }
            features.put(names[index], List.copyOf(featureValues));
        }
        // Collections.unmodifiableMap keeps the schema order.
        return Collections.unmodifiableMap(features);
    }

    private Object[] artifactMetadata() {
        if (artifactMetadata == null) {
            artifactMetadata = nativeArtifactMetadata(requireHandle());
        }
        return artifactMetadata;
    }

    /**
     * Segments raw text with the runtime policy, then tags every sentence.
     *
     * <p>Every result carries {@link Utf8ByteRange source ranges} against
     * the exact {@code text} argument — UTF-8 byte offsets, not Java string
     * indices; see {@link Utf8ByteRange} for the conversion contract.
     */
    public List<TaggedSentence> tagText(String text) {
        return assemble(nativeTagText(requireHandle(), utf8(text)));
    }

    /**
     * Tags application-supplied word tokens (space assumed between words).
     *
     * <p>Without raw text there are no source positions: the results carry
     * empty {@link Utf8ByteRange source-range} lists, which Prism never
     * invents.
     */
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
                    if (!loadEmbeddedNativeLibrary()) {
                        System.loadLibrary("prism_jni");
                    }
                    nativeLibraryLoaded = true;
                }
            }
        }
    }

    /**
     * Loads the native library embedded in the JAR (the sqlite-jdbc
     * pattern): the resource under
     * {@code /io/github/dmlux/prism/native/<os>-<arch>/} matching the
     * current platform is extracted to a temporary directory and loaded.
     * Returns false when the JAR carries no native for this platform, in
     * which case resolution falls back to {@code java.library.path}.
     */
    private static boolean loadEmbeddedNativeLibrary() {
        String fileName = System.mapLibraryName("prism_jni");
        String resource = "/io/github/dmlux/prism/native/"
                + operatingSystemClassifier() + "-" + architectureClassifier()
                + "/" + fileName;
        try (InputStream stream = PrismTagger.class.getResourceAsStream(resource)) {
            if (stream == null) {
                return false;
            }
            Path directory = Files.createTempDirectory("prism-native");
            Path target = directory.resolve(fileName);
            Files.copy(stream, target);
            target.toFile().deleteOnExit();
            directory.toFile().deleteOnExit();
            System.load(target.toAbsolutePath().toString());
            return true;
        } catch (IOException error) {
            throw new PrismException(
                    "Cannot extract the embedded native library: "
                            + error.getMessage());
        }
    }

    private static String operatingSystemClassifier() {
        String name = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (name.contains("mac") || name.contains("darwin")) {
            return "macos";
        }
        if (name.contains("win")) {
            return "windows";
        }
        return "linux";
    }

    private static String architectureClassifier() {
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        if (arch.equals("aarch64") || arch.equals("arm64")) {
            return "aarch64";
        }
        return "x86_64";
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
        int[] sentenceRangeCounts = (int[]) payload[11];
        long[] sentenceRangeStarts = (long[]) payload[12];
        long[] sentenceRangeEnds = (long[]) payload[13];
        int[] tokenRangeCounts = (int[]) payload[14];
        long[] tokenRangeStarts = (long[]) payload[15];
        long[] tokenRangeEnds = (long[]) payload[16];
        String[] distributionLabels = (String[]) payload[17];
        double[] distributionProbabilities = (double[]) payload[18];
        int totalTokens = 0;
        for (int count : tokensPerSentence) {
            totalTokens += count;
        }
        // Every token contributes the same number of distribution entries.
        int distributionStride = totalTokens == 0 ? 0 : distributionLabels.length / totalTokens;

        List<TaggedSentence> sentences = new ArrayList<>(tokensPerSentence.length);
        int token = 0;
        int feature = 0;
        int sentenceRange = 0;
        int tokenRange = 0;
        for (int sentenceIndex = 0; sentenceIndex < tokensPerSentence.length; sentenceIndex++) {
            int count = tokensPerSentence[sentenceIndex];
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
                List<Utf8ByteRange> tokenRanges = new ArrayList<>(tokenRangeCounts[token]);
                for (int entry = 0; entry < tokenRangeCounts[token]; entry++, tokenRange++) {
                    tokenRanges.add(new Utf8ByteRange(
                            tokenRangeStarts[tokenRange], tokenRangeEnds[tokenRange]));
                }
                List<UposProbability> distribution = new ArrayList<>(distributionStride);
                for (int entry = 0; entry < distributionStride; entry++) {
                    int flat = token * distributionStride + entry;
                    distribution.add(new UposProbability(
                            distributionLabels[flat], distributionProbabilities[flat]));
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
                        lemmaConfidences[token],
                        List.copyOf(tokenRanges),
                        List.copyOf(distribution)));
            }
            List<Utf8ByteRange> sentenceRanges =
                    new ArrayList<>(sentenceRangeCounts[sentenceIndex]);
            for (int entry = 0; entry < sentenceRangeCounts[sentenceIndex];
                    entry++, sentenceRange++) {
                sentenceRanges.add(new Utf8ByteRange(
                        sentenceRangeStarts[sentenceRange], sentenceRangeEnds[sentenceRange]));
            }
            sentences.add(new TaggedSentence(List.copyOf(tokens), List.copyOf(sentenceRanges)));
        }
        return List.copyOf(sentences);
    }

    private static native boolean nativeSetThreadCount(int threadCount);

    private static native long nativeCreate(byte[] utf8Directory);

    private static native void nativeDestroy(long handle);

    private static native Object[] nativeArtifactMetadata(long handle);

    private static native Object[] nativeTagText(long handle, byte[] utf8Text);

    private static native Object[] nativeTagTokens(
            long handle, byte[][] utf8Tokens, int[] tokensPerSentence);
}
