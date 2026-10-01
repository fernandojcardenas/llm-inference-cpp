# Performance: a thread pool and SIMD matmul

M1-M3 built a correct, single-threaded, scalar-float engine. M4 makes the same computation
faster without changing what it computes: a thread pool parallelizes `kernels::matmul`
(nearly all of the forward pass's time), and an AVX2+FMA (x86_64) or NEON (Apple Silicon)
`dot()` replaces the portable scalar inner loop. Every change here is checked against the
single-threaded scalar version it replaces, bit-for-bit in test, and against real generated
text end to end.

## Where the time actually goes

Before writing any of this, `tools/bench_kernels.cpp` measured `kernels::matmul` at the
project's real per-layer shapes (single-token decode): on SmolLM2-135M, matmul alone was
~33 ms of a measured ~73 ms per decode token; on Qwen2.5-0.5B-Instruct, ~137 ms of ~270 ms.
Qwen's 151,936-entry vocabulary makes its final `lm_head` projection the single largest
matmul in the whole forward pass -- bigger than all 24 attention+MLP layers combined. Full
numbers: [docs/evidence/m4-profile.txt](evidence/m4-profile.txt).

That finding set M4's two changes: a thread pool (biggest, most general win -- helps every
matmul, especially the huge one) and SIMD (a secondary multiplier on top).

A real methodology trap, found while profiling: the sandbox's usual `build/` directory has
`LLMI_SANITIZE=ON` (ASan+UBSan) baked in, which is correct for testing but makes this
engine's matmul-bound workload run about 8x slower than a clean Release build on the same
machine. M3's sandbox speed evidence was captured on such a sanitized binary. Every M4 number
in this file and its evidence files is from a dedicated, non-sanitized `build-perf/`
configuration (`-DLLMI_SANITIZE=OFF -DLLMI_BUILD_TESTS=OFF`) instead -- see
[docs/evidence/m4-speed.txt](evidence/m4-speed.txt) for the exact commands.

## Thread pool

`llmi::util::ThreadPool` ([include/llmi/util/thread_pool.hpp](../include/llmi/util/thread_pool.hpp))
is a small, fixed-size pool of persistent worker threads, created once (sized to
`std::thread::hardware_concurrency() - 1`, since the calling thread does a share of the work
too) and reused for every call -- important since `matmul` runs this every layer, every
token. Its only operation is `parallel_for(n, min_total_work, fn)`: split `[0, n)` into
`num_workers() + 1` contiguous chunks and call `fn(begin, end)` once per chunk, one of them on
the calling thread itself, blocking until every chunk returns.

`matmul` parallelizes over its output dimension: different output indices write disjoint
elements of `y` (interleaved by `out`, but never shared), so splitting that range across
threads needs no locking and changes no floating-point operation's order. The result is
bit-for-bit identical to the serial loop, just computed by more than one thread -- checked
directly in `tests/transformer_test.cpp`'s
`Kernels.MatmulAboveTheThreadingThresholdMatchesARowByRowReference` (a shape large enough to
actually split across chunks, unlike the small hand-worked `MatmulWithBias` test) and, at a
whole-model level, by the existing `KVCache.*` tests still passing bit-exact against
`forward()` with threading turned on.

Trivially small matmuls skip the thread pool entirely: dispatching costs a handful of
mutex/condition-variable round trips, which can cost more than the work saves. `matmul` only
threads when `rows * in * out` is at least ~200,000 multiply-adds (picked from
`bench_kernels.cpp`: a 1x576x576 matmul, ~330K MACs, is roughly where threading starts
paying for itself on this 2-core machine). `tests/thread_pool_test.cpp` checks `parallel_for`
itself directly: every index in `[0, n)` is covered by exactly one chunk, for `n` from 0 up to
10,000, for pools with 0, 2 and 3 workers, and across repeated calls (nothing from one
dispatch leaks into the next) -- all under ASan+UBSan, which would flag a data race on the
shared output buffer if the chunking were ever wrong.

## SIMD

`kernels::dot()` (the inner loop of `matmul`, and of attention's score/weighted-sum loops) is
chosen at compile time, not runtime-dispatched: there's no "AVX2-capable CPU, chose not to use
it" case to support, since the project already documents the platforms it targets. Two-wide
accumulator chains (mirroring the original scalar version's 8 independent accumulators, so the
same instruction-level-parallelism idea carries over) with FMA:

- **x86_64, `-mavx2 -mfma`:** two `__m256` (8-wide) accumulators, 16 floats per loop iteration.
- **Apple Silicon (arm64), NEON:** two `float32x4_t` (4-wide) accumulators, 8 floats per loop
  iteration. NEON is simply part of the base ISA on arm64, so no compiler flag is needed --
  `kernels.cpp` enables this path whenever `__ARM_NEON` is predefined, which it always is.
- **Anything else:** falls back to the original portable scalar version.

`LLMI_ENABLE_SIMD` (CMake option, default `ON`) controls whether `-mavx2 -mfma` get added on
x86_64; turning it off (for a CPU without AVX2) compiles the same portable fallback M1-M3
always used. Existing tests already cover this: `Kernels.DotMatchesDoublePrecision` checks
`dot()` against a `double`-precision reference at several sizes (0, 1, 7, 8, 9, 64, 577,
exercising the SIMD loop's full-width iterations, its tail handling, and the empty case) --
unchanged by M4, now passing against whichever `dot()` the platform compiled in.

## Speed-up

Measured on this sandbox (2 logical CPUs), 16 greedily generated tokens, `--kv-cache` (M3),
same tokens produced before and after in every case:

| Model | Before (M1-M3) | After (M4) | Speed-up |
|---|---|---|---|
| SmolLM2-135M-Instruct | 13.64 tok/s | 28.06 tok/s | 2.05x |
| Qwen2.5-0.5B-Instruct | 3.70 tok/s | 8.84 tok/s | 2.39x |

Threading alone (scalar `dot()`, thread-pooled `matmul`) measured separately at about 1.5x on
both models; SIMD contributes roughly the other half on this machine. With more cores (the
Mac's Apple Silicon has more than 2), threading's share should grow -- this sandbox is a
2-core machine, so `num_workers() + 1` tops out at 2 concurrent chunks no matter how large a
matmul gets. Full numbers, including no-cache and repeated runs:
[docs/evidence/m4-speed.txt](evidence/m4-speed.txt).

## Compared with llama.cpp

Same machine, same converted weights (f32, no quantization -- the fairest comparison of raw
matmul and threading work, since quantization changes what's computed, not just how fast),
same prompt and generation lengths, same thread count:

| | llama.cpp (f32) | this engine (f32, M4) | gap |
|---|---|---|---|
| SmolLM2 generation (16 tok) | 46.05 tok/s | 28.06 tok/s | 1.64x |
| Qwen2.5 generation (16 tok) | 13.43 tok/s | 8.84 tok/s | 1.52x |
| SmolLM2 prompt processing (35 tok) | 377.28 tok/s | ~152 tok/s | 2.48x |
| Qwen2.5 prompt processing (36 tok) | 145.45 tok/s | ~53 tok/s | 2.74x |

Generation is the closer, more apples-to-apples comparison: it's the same matrix x vector
matmul shape in both engines, and this engine lands within ~1.5-1.6x of llama.cpp's
years-tuned, hand-written-assembly GEMM kernels using nothing more than a thread pool and
portable AVX2 intrinsics. Prompt processing's bigger gap is an honest, known limitation:
prefill is a multi-row (batched) matmul, and llama.cpp's GEMM blocks and tiles across rows to
reuse each weight tile for every row while it's still in cache. This engine's `matmul` still
parallelizes over output columns regardless of row count, with no such row-blocking -- a
real next step, not implemented here (M4's scope was threading, SIMD and honest measurement,
not a from-scratch blocked GEMM). Full methodology and commands to reproduce:
[docs/evidence/m4-llamacpp-comparison.txt](evidence/m4-llamacpp-comparison.txt).

## What this doesn't do

No cache-blocked/tiled matmul (see above): the prompt-processing gap vs llama.cpp is the
direct cost of this. No quantization (M5). No runtime CPU-feature detection -- `dot()` is
chosen once, at compile time; a binary built with `LLMI_ENABLE_SIMD=ON` on an AVX2 machine
will not run (illegal instruction) on an older CPU without AVX2, and must be rebuilt with
that option off instead. Per-step heap allocation inside `forward_cached()` (a handful of
`std::vector`s sized and freed on every single decode call) is a smaller, real cost noted in
profiling but not addressed here, since it's secondary to matmul at every shape measured.
