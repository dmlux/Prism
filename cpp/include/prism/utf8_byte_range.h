// The canonical Prism source-position type.
//
// A Utf8ByteRange is a half-open [start, end) byte range in the UTF-8
// encoding of the exact, unmodified raw-text input a caller passed to
// Prism. The offsets never refer to internally repaired, merged,
// whitespace-collapsed, or otherwise transformed intermediate strings.
//
// Contract for every range Prism emits:
//   - start is inclusive, end is exclusive, and start < end (never empty);
//   - both offsets lie on UTF-8 codepoint boundaries of the input;
//   - the range lies inside the input;
//   - a list of ranges is ordered and non-overlapping.
//
// The unit is deliberately part of the name: these are UTF-8 byte offsets,
// not UTF-16 code-unit offsets and not codepoint or character indices.
// Consumers whose coordinate system is UTF-16 (for example Java string
// indices or Apple text APIs) must convert against the unchanged original
// text; copying the numbers is not a conversion.

#pragma once

#include <cstddef>

namespace prism {

struct Utf8ByteRange {
    std::size_t start = 0;
    std::size_t end = 0;

    bool operator==(const Utf8ByteRange&) const = default;
};

} // namespace prism
