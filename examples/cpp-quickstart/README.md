# C++ quickstart

The smallest complete C++ consumer, wired through CMake FetchContent
against a pinned release tag. The first configure fetches Prism and the
pinned ExecuTorch sources (network required); the build compiles them
once into the example's build tree.

Requirements: CMake 3.24+, a C++20 toolchain, and a Python interpreter
with `torch` installed — the ExecuTorch build resolves its headers
through Python (`python -m pip install "torch>=2.9,<3"`). If that
Python is not the default one on your PATH, pass
`-DPython3_EXECUTABLE=/path/to/python` to the configure step.

```bash
# From this directory, with an unpacked model (see ../README.md):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
build/quickstart ../../prism-no-0.2.4-fast
```

Expected output: the artifact identity line followed by one line per
token (`text  UPOS  lemma  confidence`).

Works on macOS and Linux (and is the intended route for Windows once
that leg stabilizes). For Apple projects that would rather consume a
prebuilt binary, see `../prism-native-consumer/`.
