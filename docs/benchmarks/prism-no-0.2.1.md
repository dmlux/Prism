# prism-no 0.2.1 — artifact benchmarks (superseded by 0.2.2)

Historical record of the first cross-binding runtime measurements;
the current numbers live in `prism-no-0.2.2.md`.

## Chapter end-to-end runtime comparison (artifact 0.2.1)

Every binding tags the same local book-chapter fixture (247 sentences,
3,783 tokens; untracked for copyright reasons) end to end: raw text in,
tagged sentences with calibrated confidences out. Hardware: Apple M-series,
CPU only. Values are the warm second run of a fresh process; cold runs
(including lazy program loading) differ by well under 5%. The
"one program" column uses a manifest copy exposing only the large
8×160×96 program; "two programs" is the shipped artifact where batches
additionally qualify for the small 8×48×32 program (automatic
smallest-fit selection, identical predictions in both configurations).

| Binding | Runtime | One program | Two programs |
| --- | --- | ---: | ---: |
| Python | eager PyTorch, dynamic shapes | 1.6 s | 1.6 s |
| Swift (PrismKit) | ExecuTorch XNNPACK | 5.6 s | 2.9 s |
| C++ | ExecuTorch XNNPACK | 8.2 s | 3.9 s |
| Java (JNI over C++) | ExecuTorch XNNPACK | 8.2 s | 3.9 s |

Readings:

- **The second program pays for itself.** Fixed-shape padding dominates
  the ExecuTorch cost; short-sentence batches on the 48×32 program cut
  the chapter roughly in half. With program-data separation the second
  shape costs only 0.7 MB of bundle size.
- **Python eager is the throughput winner but not the deployment
  answer.** Dynamic shapes waste no padding compute and PyTorch's CPU
  kernels are heavily optimized — but the Python runtime requires the
  full torch stack and the raw checkpoint, which is not the offline,
  dependency-free story the native artifact exists for. It sets the
  target the fixed-shape runtimes should approach (more bucket shapes,
  threadpool tuning).
- **Java equals C++.** The JNI bridge (byte-array in, one flat payload
  out) is invisible next to inference.
- **Swift outruns C++ by ~35%** on identical programs; the difference
  sits in the ExecuTorch build/threadpool configuration, documented as
  an open follow-up in `docs/PROJECT_STATUS.md`.

Reproduction:

```bash
# Single-program manifest variant (hardlinks, filtered manifest)
python - << 'PY'
import json, os, shutil
from pathlib import Path
source, target = Path("models/prism-no-0.2.2"), Path("models/prism-no-0.2.2-single")
if target.exists(): shutil.rmtree(target)
target.mkdir(); (target / "LICENSES").mkdir()
for entry in source.iterdir():
    if entry.name not in ("manifest.json", "LICENSES"):
        os.link(entry, target / entry.name)
os.link(source / "LICENSES/README.md", target / "LICENSES/README.md")
manifest = json.loads((source / "manifest.json").read_text())
manifest["programs"] = [p for p in manifest["programs"] if p["file_name"] == "model-xnnpack.pte"]
(target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

# C++ (builds with the test suite)
cpp/build/prism_chapter_benchmark models/prism-no-0.2.2 data/examples/hp7kap1.txt

# Java
java -Djava.library.path=cpp/build -cp "cpp/build/prism.jar:cpp/build/prism_java_test.jar" \
  io.github.dmlux.prism.PrismChapterBenchmark models/prism-no-0.2.2 data/examples/hp7kap1.txt

# Swift (release mode; PRISM_ARTIFACT overrides the artifact directory)
cd swift && swift test -c release --filter ChapterBenchmarkTests
```

### Optimized runtimes: thread cap and four shapes (artifact 0.2.2)

Two measured optimizations follow up on the table above. First, the
ExecuTorch default threadpool spans every logical core (cpuinfo does not
separate performance from efficiency cores on Apple Silicon); on the
small fixed-shape batches that oversubscribes — a sweep on a 16-core
M4 Max shows 6 threads beating the 16-thread default by 24%. The C++
tagger now installs this measured default (`prism::engine::SetThreadCount`,
`prism_set_thread_count`, and `PrismTagger.setThreadCount` override it;
the benchmark tool accepts `PRISM_THREADS`). Second, artifact 0.2.2 adds
two more fixed shapes (8×24×16 and 8×96×64, +0.7 MB each thanks to the
shared `model.ptd`): 150 of the chapter's 247 sentences fit 24×16, so
batches stop paying 48×32 padding. Same chapter, same machine, warm:

| Binding | 0.2.1 two shapes | + thread cap | 0.2.2 four shapes |
| --- | ---: | ---: | ---: |
| Swift (PrismKit) | 2.9 s | n/a (prebuilt default) | **2.5 s** |
| C++ | 3.9 s | 2.9 s | **2.2 s** |
| Java | 3.9 s | 2.9 s | **2.2 s** |

C++ and Java now beat Swift (which cannot cap threads through the
prebuilt frameworks yet) and close most of the gap to eager Python's
1.6 s; the remainder is residual padding and fixed-shape dispatch.
Predictions stay identical across all configurations (every added
program passes the same parity gate against the shared data file).

## Runtime and quality matrix: fp32 versus fast (int8)

Two artifacts of the same model version now exist; an application ships
exactly one:

| Artifact | Bundle size | Programs | Weights |
| --- | ---: | --- | --- |
| `prism-no-0.2.2` (fp32) | ≈ 94 MB | 4 shapes, 0.7 MiB each | `model.ptd` 83.3 MiB fp32 |
| `prism-no-0.2.2-fast` (int8) | ≈ 45 MB | 4 shapes, 1.5 MiB each | `model.ptd` 33.5 MiB int8 |

### Quality (development split, 67,619 tokens, both standards)

The fast artifact's quality gate: the quantized eager twin against the
fp32 adapter with the identical production decoding policy. Accuracy on
all development sentences fitting the 96-token export shape:

| Task | Standard | fp32 | fast | Delta |
| --- | --- | ---: | ---: | ---: |
| UPOS | nb | 99.1724% | 99.1641% | -0.0082 pp |
| UPOS | nn | 98.8384% | 98.8448% | +0.0064 pp |
| UFeats exact | nb | 97.9021% | 97.8883% | -0.0137 pp |
| UFeats exact | nn | 95.3408% | 95.3312% | -0.0096 pp |
| Lemma | nb | 99.2301% | 99.2246% | -0.0055 pp |
| Lemma | nn | 98.8672% | 98.8608% | -0.0064 pp |

Decision flips between fp32 and fast: 0.04–0.27% per task — an order of
magnitude below seed-to-seed training variance. The official test-split
evaluation (and the UDPipe comparison above) belongs to the frozen fp32
model and was not re-run; the fast artifact is quality-gated on the
development split only.

### End-to-end chapter runtime (247 sentences / 3,783 tokens, warm)

All bindings install the measured six-thread default (override:
`prism::engine::SetThreadCount`, `prism_set_thread_count`,
`PrismTagger.setThreadCount`, `ComputeThreads.setThreadCount`). Apple
M4 Max, CPU only; supersedes the tables above.

| Variant | Python | Swift | C++ | Java |
| --- | ---: | ---: | ---: | ---: |
| fp32, one program (8×160×96) | 1.6 s* | 4.8 s | 7.5 s | 7.7 s |
| fp32, four shapes | 1.6 s* | 1.5 s | 2.2 s | 2.2 s |
| fast (int8), four shapes | n/a** | **1.5 s** | **1.2 s** | **1.2 s** |

\* Python runs the checkpoint eagerly with dynamic shapes — there are no
fixed-shape programs to select, so both fp32 rows coincide.
\*\* The ExecuTorch Python wheel ships no quantized runtime kernels.

Readings:

- **fast is the recommended deployment artifact**: less than half the
  bundle, native runtimes at or below eager-Python throughput, quality
  within noise of fp32.
- **Swift's fp32 advantage over C++** (1.5 s vs 2.2 s on identical
  programs) comes from the newer XNNPACK inside the swiftpm-1.4 runtime;
  upgrading the C++ FetchContent pin is the documented follow-up — the
  int8 kernels show no such gap.
- The thread cap is worth up to 40% on Swift and 24% on C++; the
  runtime's own default oversubscribes every logical core because
  cpuinfo cannot separate performance from efficiency cores on Apple
  Silicon.
