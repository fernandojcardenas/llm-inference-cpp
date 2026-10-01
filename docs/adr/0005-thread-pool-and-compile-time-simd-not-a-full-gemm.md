# ADR 0005: A thread pool over matmul's output rows, compile-time SIMD, no runtime dispatch

Status: accepted (M4)

## Context

M1-M3's `kernels::matmul` is correct but single-threaded and scalar. M4's job is to make the
existing computation faster without changing what it computes, and to measure that honestly
against both the engine's own prior milestones and against llama.cpp.

## Decisions

### Profile before optimizing

Before writing any M4 code, `tools/bench_kernels.cpp` measured where time actually goes at
this project's real per-layer shapes. It confirmed `matmul` is roughly half of per-token
decode time (the rest: attention score/softmax loops, RoPE, per-step allocation), and that
Qwen2.5's 151,936-entry vocabulary makes its final `lm_head` projection the single largest
matmul in the whole model. This is what decided M4's two changes (thread pool, then SIMD) and
their order, rather than guessing. See docs/performance.md and docs/evidence/m4-profile.txt.

A real finding from this step: the sandbox's usual `build/` (ASan+UBSan, `LLMI_SANITIZE=ON`)
runs this engine's matmul-bound workload about 8x slower than a clean Release build, which
means M3's own sandbox speed evidence was captured on a sanitized binary. Every M4 speed
number uses a dedicated, non-sanitized `build-perf/` configuration instead, documented
explicitly so a reader doesn't compare M3's and M4's sandbox numbers directly.

### Thread pool parallelizes matmul's output rows, not something coarser-grained

`llmi::util::ThreadPool` is a small, persistent pool (`parallel_for` only, no general task
queue) that splits `matmul`'s output dimension across workers plus the calling thread.
Output rows are independent (disjoint writes, no shared accumulator across threads), so this
needs no locking and produces results bit-for-bit identical to the serial loop -- checked
directly (`Kernels.MatmulAboveTheThreadingThresholdMatchesARowByRowReference`) and at a
whole-model level (`KVCache.*` still passing bit-exact against `forward()`).

Parallelizing at a coarser grain (e.g. splitting work across layers, or across the handful of
matmuls within one layer) was considered and rejected: layers are sequentially dependent
(each needs the previous layer's output), and splitting across the 5-7 matmuls inside one
layer would mean uneven, hard-to-balance chunks (the MLP's `gate`/`up`/`down` are much larger
than `wq`/`wk`/`wv`) for no benefit over just parallelizing each matmul's own output rows,
which already scales with however many matmuls exist, of whatever size, including the one
that actually dominates (the final `lm_head` projection).

Trivially small matmuls (most of the attention projections at decode time) skip the thread
pool and run on the calling thread alone: dispatching costs a handful of
mutex/condition-variable round trips, more than tiny work would save. The threshold
(~200,000 multiply-adds) was picked from the same microbenchmark that motivated this ADR, not
guessed.

### SIMD is chosen at compile time, never at runtime

`dot()` has three versions -- AVX2+FMA (x86_64), NEON (arm64), portable scalar fallback --
selected by preprocessor checks at compile time. No CPUID-based runtime dispatch (checking
`__builtin_cpu_supports("avx2")` and branching to one of several compiled variants) was built,
because this project already documents and tests specific target platforms (an x86_64
Linux/macOS dev machine, Apple Silicon in CI's `macos` job) rather than claiming to run
correctly on arbitrary, unknown hardware. Runtime dispatch would add real complexity (compile
every variant, probe the CPU once, pick a function pointer) to cover a platform this project
doesn't target. `LLMI_ENABLE_SIMD=OFF` is the documented escape hatch for an AVX2-less x86_64
CPU: it compiles the same portable scalar fallback used throughout M1-M3, just without the
`-mavx2 -mfma` compiler flags that would otherwise make the binary require AVX2 to run at all.

### Compared against llama.cpp at f32, not a quantized build

M5 (not yet built) is where quantization belongs. Comparing against a quantized llama.cpp
build now would conflate two different questions -- "is this engine's threading and SIMD
competitive" and "does quantization help" -- into one number. Converting the exact same
`testdata/*/model.safetensors` this project already tests against to GGUF at `f32` (no
quantization) isolates the first question: same precision, same model, same machine, same
thread count, so the resulting gap (about 1.5-1.6x slower at generation, bigger at prompt
processing) is attributable to matmul/threading implementation quality, not numeric format.

## Consequences

- A binary built with `LLMI_ENABLE_SIMD=ON` targeting AVX2 will not run (illegal instruction)
  on a CPU without AVX2; such a machine needs `LLMI_ENABLE_SIMD=OFF` and a rebuild, not a
  runtime fallback.
- Prompt processing (multi-row/batched prefill) remains notably slower than llama.cpp, because
  this engine's `matmul` has no row-blocked/tiled GEMM -- it still does independent dot
  products per (row, output) pair, just now threaded and SIMD'd. A blocked GEMM is a natural
  next step, left for later rather than folded into M4's scope.
- Per-step heap allocation inside `forward_cached()` is a secondary, real cost identified
  during profiling but not addressed in M4, since every shape measured showed matmul as the
  larger cost.
