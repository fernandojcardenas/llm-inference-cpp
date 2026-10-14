# Performance study on Apple Silicon (M8)

This is the M8 performance study: where single-token decoding spends its time, measured on Apple
Silicon. It is a measurement study, in the spirit of M4 ([docs/performance.md](performance.md)) and M5
([docs/quantization.md](quantization.md)) — it quantifies the engine's behaviour and names the next
optimizations. Of those, the KV-cache layout one has since been built (see that section below).

The numbers come from `tools/bench_apple_silicon.cpp`, which, like `tools/bench_kernels.cpp`, needs no model
weights: it runs the real kernels on random data at the models' real shapes, so it compiles and runs anywhere
the engine does. Two machines: **Apple Silicon** — an arm64 Linux VM on the Mac (Vendor ID Apple, 4 cores),
the engine's NEON kernels; and the **x86 cloud sandbox** (AVX2, 2 cores) for comparison. CPU only — this
engine does no GPU work, so there is no Metal here. Everything is single-threaded. Full output and the exact
command are in [docs/evidence/m8-apple-silicon.txt](evidence/m8-apple-silicon.txt).

## Decoding is memory-bound

Generating one token is a sequence of matrix-by-vector products: every weight is read once and used for two
flops (a multiply and an add), an arithmetic intensity of 2 flops / 4 bytes = **0.5 FLOP/byte**. That is far
below any modern machine's ridge point, so decode speed is set by how fast weights are read from memory, not
by arithmetic.

The cache-busting `lm_head` (108 MiB of weights, far larger than cache) shows it cleanly. On Apple Silicon a
STREAM triad measured **76 GB/s** of memory bandwidth; the `lm_head` float32 matmul ran at **75.5 GB/s and
37.8 GFLOP/s**, and 75.5 × 0.5 = 37.8 — it is reading weights at the machine's full bandwidth, and the flops
come for free. The smaller per-layer matmuls report higher GB/s only because a few-MiB weight partly fits in
cache. The same picture holds on the x86 sandbox at its lower ~10 GB/s, and that bandwidth gap (76 vs 10) is
most of why the Mac decodes several times faster.

## Quantization saves memory but not yet time

Because decode is memory-bound, moving fewer weight bytes should make it faster — that is the whole promise of
quantization. Q8_0 and Q4_0 store 3.6× and 6.4× fewer bytes (measured, matching the theoretical ratio). But
the engine's dequant has no SIMD yet, so the quantized kernels are **compute-bound and run slower than
float32** on both machines: on Apple Silicon the `lm_head` is 1.50 ms in float32 against 3.40 ms for Q8_0.
This is exactly the honest gap M5 documented ([ADR 0006](adr/0006-block-quantization-matches-gguf-no-simd-yet.md)),
now confirmed on Apple Silicon: the memory saving is real, but the speed-up waits on a vectorized dequant
kernel. That remains the single highest-value compute optimization for this engine.

## The KV cache layout leaves bandwidth on the table

Attention reads the whole KV cache every step, so at long context it is the other memory-bound cost. The
engine stores K and V **position-major** — `[position][kv_head][head_dim]`, the layout in
`transformer.hpp` — which means one attention head's keys are strided across memory (each position's row holds
all KV heads). A **head-major** layout, `[kv_head][position][head_dim]`, makes a single head's keys contiguous.

Measured with the same data and the same math, only the layout changed, head-major is **~25–35% faster at long
context** on Apple Silicon and reaches the machine's bandwidth where position-major does not:

| context | KV read / token | position-major | head-major |
|---|---|---|---|
| 1024 | 45 MiB | 3.9 ms (12.1 GB/s) | 3.0 ms (15.7 GB/s) |
| 2048 | 90 MiB | 7.8 ms (12.1 GB/s) | 6.1 ms (15.5 GB/s) |
| 4096 | 180 MiB | 16.2 ms (11.6 GB/s) | 12.0 ms (15.7 GB/s) |

This was the study's concrete next optimization, and it has since been **built**: the engine's KV cache is now
head-major, and `forward_cached()` reads each head contiguously. The change is bit-identical — the KV-cache
tests (which pin `forward_cached()` to the no-cache `forward()` byte-for-byte) and an end-to-end token-identity
check both hold, and clang-tidy and the full suite pass ([docs/evidence/m8-kv-head-major.txt](evidence/m8-kv-head-major.txt)).
(At short context the KV read is small next to the forward pass, so this matters most for long prompts.)

## The tokenizer is not a bottleneck

For completeness, the byte-level BPE tokenizer runs at about **8 million tokens/s encoding** and **100 million
tokens/s decoding** on Apple Silicon (round-trip exact). Tokenization is nowhere near the forward pass in cost;
it needed no optimization and got none.

## What this measured, and didn't

Real kernels at real shapes on real Apple Silicon, single-threaded, CPU only. It did not measure the whole
model end to end (no weights were loaded), multi-threaded scaling (M4 covers the thread pool), or a GPU/Metal
path (the engine has none). The conclusions — memory-bound decode, quantization waiting on a SIMD dequant, a
head-major KV layout worth ~30% at long context — are what the microbenchmarks support.
