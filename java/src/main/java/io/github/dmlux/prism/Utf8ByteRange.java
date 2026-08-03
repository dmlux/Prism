package io.github.dmlux.prism;

/**
 * The canonical Prism source position: a half-open {@code [start, end)}
 * byte range in the <em>UTF-8 encoding</em> of the exact, unmodified text
 * passed to {@link PrismTagger#tagText}.
 *
 * <p>The offsets never refer to internally repaired, merged, or otherwise
 * transformed intermediate strings, and they are UTF-8 byte offsets — they
 * are <em>not</em> Java string indices. Java strings count UTF-16 code
 * units: {@code "🙂å"} occupies 4 + 2 UTF-8 bytes but 2 + 1 UTF-16 code
 * units, so {@code å} starts at UTF-8 byte offset 4 yet at Java char index
 * 2. Copying the numbers is not a conversion; convert against the original
 * string, for example with {@link #utf16OffsetOf}.
 *
 * <p>Every range Prism emits is non-empty, lies inside the input, has both
 * boundaries on UTF-8 codepoint boundaries, and range lists are ordered and
 * non-overlapping.
 *
 * @param start inclusive UTF-8 byte offset
 * @param end exclusive UTF-8 byte offset, greater than {@code start}
 */
public record Utf8ByteRange(long start, long end) {

    public Utf8ByteRange {
        if (start < 0 || end <= start) {
            throw new IllegalArgumentException(
                    "A Utf8ByteRange must satisfy 0 <= start < end, got ["
                            + start + ", " + end + ").");
        }
    }

    /**
     * Maps a UTF-8 byte offset of the original text onto the corresponding
     * UTF-16 code-unit index of the same string (a Java string index).
     *
     * <p>Walks the string once, so the mapping is unambiguous even for
     * repeated substrings. Apply it to {@link #start} and {@link #end} to
     * obtain a {@code String#substring} compatible index pair.
     *
     * @throws IllegalArgumentException when the offset exceeds the text's
     *     UTF-8 length or does not lie on a UTF-8 codepoint boundary
     */
    public static int utf16OffsetOf(String originalText, long utf8ByteOffset) {
        if (utf8ByteOffset < 0) {
            throw new IllegalArgumentException(
                    "UTF-8 byte offset must not be negative: " + utf8ByteOffset);
        }
        long utf8Position = 0;
        int utf16Position = 0;
        while (utf16Position < originalText.length()) {
            if (utf8Position == utf8ByteOffset) {
                return utf16Position;
            }
            if (utf8Position > utf8ByteOffset) {
                break;
            }
            int codepoint = originalText.codePointAt(utf16Position);
            utf8Position += utf8ByteLengthOf(codepoint);
            utf16Position += Character.charCount(codepoint);
        }
        if (utf8Position == utf8ByteOffset) {
            return utf16Position;
        }
        throw new IllegalArgumentException(
                "UTF-8 byte offset " + utf8ByteOffset
                        + " is out of range or not on a codepoint boundary.");
    }

    private static int utf8ByteLengthOf(int codepoint) {
        if (codepoint < 0x80) {
            return 1;
        }
        if (codepoint < 0x800) {
            return 2;
        }
        if (codepoint < 0x10000) {
            return 3;
        }
        return 4;
    }
}
