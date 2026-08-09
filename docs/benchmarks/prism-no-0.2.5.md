# prism-no 0.2.5 — artifact benchmarks

Two artifacts of the same model version; an application bundles exactly
one. The fp32 artifact is the exact reference; the fast artifact
quantizes linears (dynamic, per-channel) and embeddings (per-channel,
fused into `embedding_byte`) to int8.

| Artifact | Bundle size | Programs | Weights |
| --- | ---: | --- | --- |
| `prism-no-0.2.5` (fp32) | ≈ 94 MB | 4 shapes, 0.7 MiB each | `model.ptd` 83.3 MiB fp32 |
| `prism-no-0.2.5-fast` (int8) | ≈ 45 MB | 4 shapes, 0.6 MiB each | `model.ptd` 33.5 MiB int8 |

Program shapes (batch 8): 24×16, 48×32, 96×64, and 160×96
(subwords×tokens). Runtimes sort sentences by length and run every
batch on the smallest fitting program. Bundle sizes exclude
`fixtures.json`, which is a development aid.

## What changed since 0.2.4

**Only the int8 `-fast` programs changed — the model is numerically the
same.** The fp32 `model.ptd` is byte-identical to 0.2.4 (and thus to
0.2.2); the int8 weights are unchanged. The int8 programs are now
lowered with a single **grouped** XNNPACK partitioner instead of the
former per-op partitioner, so the graph delegates as a few large
subgraphs rather than hundreds of tiny ones. This removes XNNPACK
per-delegate call overhead and makes the fast artifact faster at
byte-identical weights and identical decoding (verified by the C++
fixture-parity suite). The `.pte` program files also shrink (≈ 1.6 MiB
→ ≈ 0.6 MiB each) because there are far fewer delegate payloads.

Every published **quality** number is therefore unchanged — the frozen
test-split results and the fast-versus-fp32 development gate in
[prism-no-0.2.2.md](prism-no-0.2.2.md) apply verbatim (fp32 within
0.014 pp of int8 on every task).

## Speed (reproducible protocol)

Measured under the 0.3.0 protocol on the checked-in CC0 text
`data/examples/skarvholmen-bokmaal.txt` (55 sentences / 905 tokens),
repeated past 6,000 tokens for the document-inference row.

- Machine: Apple M4 Max, macOS, release build, default thread policy.
- C++ suite: Google Benchmark (vendored),
  `--benchmark_repetitions=3 --benchmark_min_time=1x`, median reported.

| Document tag (≈ 6.8k tokens) | fp32 | fast (int8) | speed-up |
| --- | ---: | ---: | ---: |
| `TagText` wall-clock | 3535 ms | **1802 ms** | **1.96×** |
| `TagPretokenized` wall-clock | 3558 ms | **1798 ms** | **1.98×** |
| `TagText` throughput | ≈ 1,850 tok/s | **≈ 3,760 tok/s** | — |
| `TaggerLoad` | 32.1 ms | 34.5 ms | — |

For reference, the previous per-op int8 lowering (0.2.4-fast) reached
≈ 1.81× on the same document; the grouped lowering lifts the fast
artifact to ≈ 1.96× versus fp32 — a ≈ 9 % relative speed-up at zero
quality or size cost.

Reproduce:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DPRISM_BENCHMARKS=ON
cmake --build cpp/build --target prism_benchmarks_norwegian
cpp/build/prism_benchmarks_norwegian --benchmark_repetitions=3 --benchmark_min_time=1x
```
