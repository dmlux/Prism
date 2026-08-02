# prism-no 0.2.2 — artifact benchmarks

Two artifacts of the same model version; an application bundles exactly
one. The fp32 artifact is the exact reference; the fast artifact
quantizes linears (dynamic, per-channel) and embeddings (per-channel,
fused into `embedding_byte`) to int8.

| Artifact | Bundle size | Programs | Weights |
| --- | ---: | --- | --- |
| `prism-no-0.2.2` (fp32) | ≈ 94 MB | 4 shapes, 0.7 MiB each | `model.ptd` 83.3 MiB fp32 |
| `prism-no-0.2.2-fast` (int8) | ≈ 45 MB | 4 shapes, 1.5 MiB each | `model.ptd` 33.5 MiB int8 |

Program shapes (batch 8): 24×16, 48×32, 96×64, and 160×96
(subwords×tokens). Runtimes sort sentences by length and run every
batch on the smallest fitting program. Bundle sizes exclude
`fixtures.json`, which is a development aid.

## Quality

### Frozen test-split benchmark (fp32 model)

Evaluated exactly once on the untouched UD test splits with every
policy frozen on development; UDPipe 2.17 via the gold-tokenized LINDAT
reproduction on byte-identical gold files. Official word-level CoNLL
definitions (complete `UFeats` bundles).

| Test cell | Prism | UDPipe 2.17 | Delta |
| --- | ---: | ---: | ---: |
| Bokmål UPOS | **98.7619%** | 98.5717% | +0.1902 pp |
| Bokmål UFeats | 97.1968% | **97.5906%** | -0.3938 pp |
| Bokmål Lemmas | **98.9755%** | 98.8654% | +0.1101 pp |
| Nynorsk UPOS | **98.7688%** | 98.5993% | +0.1695 pp |
| Nynorsk UFeats | 96.9362% | **97.3842%** | -0.4481 pp |
| Nynorsk Lemmas | **98.6760%** | 98.5630% | +0.1130 pp |

Prism wins UPOS and Lemmas on both written standards and stays behind
on exact morphology bundles — at roughly one tenth of UDPipe's model
size for fp32, and about one twentieth for fast.

### fast versus fp32 (development split)

The fast artifact is quality-gated on the development split (67,619
tokens across both standards; the test splits are reserved for the
frozen fp32 evaluation above). Accuracy with the identical production
decoding policy:

| Task | Standard | fp32 | fast | Delta |
| --- | --- | ---: | ---: | ---: |
| UPOS | nb | 99.1724% | 99.1641% | -0.0082 pp |
| UPOS | nn | 98.8384% | 98.8448% | +0.0064 pp |
| UFeats exact | nb | 97.9021% | 97.8883% | -0.0137 pp |
| UFeats exact | nn | 95.3408% | 95.3312% | -0.0096 pp |
| Lemma | nb | 99.2301% | 99.2246% | -0.0055 pp |
| Lemma | nn | 98.8672% | 98.8608% | -0.0064 pp |

Decision flips between fp32 and fast: 0.04–0.27% per task — an order
of magnitude below seed-to-seed training variance.

## End-to-end runtime

Book-chapter fixture (247 sentences / 3,783 tokens; untracked for
copyright reasons), raw text in, tagged sentences with calibrated
confidences out; warm second run of a fresh process. Apple M4 Max, CPU
only. Every binding installs the measured six-thread default
(override: `prism::engine::SetThreadCount`, `prism_set_thread_count`,
`PrismTagger.setThreadCount`, `ComputeThreads.setThreadCount`).

| Variant | Python | Swift | C++ | Java |
| --- | ---: | ---: | ---: | ---: |
| fp32, one program (8×160×96) | 1.6 s* | 4.8 s | 7.5 s | 7.7 s |
| fp32, four shapes | 1.6 s* | 1.5 s | 2.2 s | 2.2 s |
| fast (int8), four shapes | n/a** | **1.5 s** | **1.2 s** | **1.2 s** |

\* Python runs the checkpoint eagerly with dynamic shapes — there are
no fixed-shape programs to select, so both fp32 rows coincide.
\*\* The ExecuTorch Python wheel ships no quantized runtime kernels.

Readings:

- **fast is the recommended deployment artifact**: less than half the
  bundle, native runtimes at or below eager-Python throughput, quality
  within noise of fp32.
- **On Swift, fast matches fp32 in speed rather than beating it** —
  the swiftpm-1.4 runtime's newer XNNPACK has exceptionally fast fp32
  GEMMs; fast's advantage on Swift is therefore the bundle size. On the
  C++ v1.3.1 runtime pin, fast is nearly twice as fast as fp32;
  upgrading the C++ pin is the documented follow-up.
- The thread cap is worth up to 40% on Swift and 24% on C++; the
  runtime's own default oversubscribes every logical core because
  cpuinfo cannot separate performance from efficiency cores on Apple
  Silicon.

## Reproduction

```bash
# C++ (builds with the test suite)
cpp/build/prism_chapter_benchmark models/prism-no-0.2.2 data/examples/hp7kap1.txt
cpp/build/prism_chapter_benchmark models/prism-no-0.2.2-fast data/examples/hp7kap1.txt

# Java
java -Djava.library.path=cpp/build -cp "cpp/build/prism.jar:cpp/build/prism_java_test.jar" \
  io.github.dmlux.prism.PrismChapterBenchmark models/prism-no-0.2.2-fast data/examples/hp7kap1.txt

# Swift (release mode; PRISM_ARTIFACT selects the artifact directory)
cd swift && PRISM_ARTIFACT=models/prism-no-0.2.2-fast \
  swift test -c release --filter ChapterBenchmarkTests

# Thread sweep
PRISM_THREADS=6 cpp/build/prism_chapter_benchmark models/prism-no-0.2.2-fast data/examples/hp7kap1.txt
```

Single-program variants for the first table row are manifest copies
exposing only the largest program (hardlinks plus a filtered
`manifest.json`); the snippet lives in the repository history and in
`docs/benchmarks/prism-no-0.2.1.md`.
