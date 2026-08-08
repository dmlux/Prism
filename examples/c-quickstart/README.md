# C quickstart

The smallest complete consumer of the stable C ABI (`prism_c.h`) —
the same 30-function surface every foreign-function integration binds
against. Built with CMake FetchContent against a pinned release tag;
the first configure fetches Prism and the pinned ExecuTorch sources
(network required).

Requirements: CMake 3.24+, a C++20 toolchain (the tagger core is C++;
the example source is plain C99), and a Python interpreter with `torch`
installed — the ExecuTorch build resolves its headers through Python
(`python -m pip install "torch>=2.9,<3"`). If that Python is not the
default one on your PATH, pass `-DPython3_EXECUTABLE=/path/to/python`
to the configure step.

```bash
# From this directory, with an unpacked model (see ../README.md):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
build/quickstart ../../prism-no-0.2.4-fast
```

Expected output: the artifact identity line followed by one line per
token (`text  UPOS  lemma  confidence`).

Apple projects can consume the identical ABI without any CMake build
through the prebuilt PrismNative XCFramework — see
`../prism-native-consumer/`.
