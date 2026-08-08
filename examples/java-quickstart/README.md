# Java quickstart

The smallest complete Java consumer: one Maven dependency, one class.
The Central JAR embeds the native libraries for macOS (arm64, x86_64)
and Linux (x86_64, aarch64) — no `java.library.path`, no further setup.
Requires Java 21+.

```bash
# From this directory, with an unpacked model (see ../README.md):
mvn -q compile exec:java -Dexec.args="../../prism-no-0.2.4-fast"
```

Expected output: the artifact identity line followed by one line per
token (`text  UPOS  lemma  confidence`).

Kotlin works identically with
`implementation("io.github.dmlux:prism:0.6.0")`. Without a build tool,
the same JAR is attached to every
[`v*` release](https://github.com/dmlux/Prism/releases):

```bash
javac -cp prism-0.6.0-all-platforms.jar src/main/java/quickstart/Quickstart.java -d out
java -cp out:prism-0.6.0-all-platforms.jar quickstart.Quickstart ../../prism-no-0.2.4-fast
```
