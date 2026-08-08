# Examples

Minimal, runnable quickstart projects — one per language binding, each
containing just a manifest and one source file. They demonstrate the
smallest complete setup that loads a model artifact, tags one sentence,
and prints tokens with UPOS, lemma, and confidence.

| Example | Binding | Build tool |
| --- | --- | --- |
| [`swift-quickstart/`](swift-quickstart/) | PrismKit (Swift) | SwiftPM |
| [`cpp-quickstart/`](cpp-quickstart/) | C++ | CMake (FetchContent) |
| [`c-quickstart/`](c-quickstart/) | C ABI | CMake (FetchContent) |
| [`java-quickstart/`](java-quickstart/) | Java/Kotlin | Maven (Central) |
| [`prism-native-consumer/`](prism-native-consumer/) | PrismNative (Apple binary C ABI) | SwiftPM |

Every example takes the artifact directory as its first command-line
argument. Download and unpack a model once (any example below assumes
this folder):

```bash
curl -LO https://github.com/dmlux/Prism/releases/download/prism-no-0.2.4/prism-no-0.2.4-fast.tar.gz
tar -xzf prism-no-0.2.4-fast.tar.gz
```

`prism-native-consumer` is both an example and the CI validation of the
PrismNative XCFramework; the quickstarts stay as small as possible.
Integration details per binding: [docs/INTEGRATION.md](../docs/INTEGRATION.md).
