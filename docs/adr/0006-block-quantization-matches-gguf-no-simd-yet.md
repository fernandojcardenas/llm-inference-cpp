# ADR 0006: Q8_0/Q4_0 block quantization matching GGUF, no quantized SIMD yet

Status: accepted (M5)

## Context

M1-M4 store every weight as float32. M5's job is to add a smaller, lossy representation
(quantization) and measure, honestly, what it costs in quality and what it buys in memory
and speed -- the same "measure before and after, document what doesn't hold up" standard M4
set for its own thread pool and SIMD work.

## Decisions

### Match GGUF's own Q8_0/Q4_0 exactly, not a format of this engine's own invention

Both formats use GGUF's block size (32 weights/block), the same symmetric per-block scale
convention, and the same bit layout (Q4_0 packs two nibbles per byte with a +8 bias). This
was not the only reasonable choice -- a larger block size trades a little quality for less
scale overhead, for instance -- but matching GGUF means a weight this engine quantizes and
llama.cpp quantizing the identical float32 source are directly comparable, which is exactly
what the roadmap's M5 success criterion ("quality loss measured by perplexity against the
full-precision model and llama.cpp") needs. A format of this engine's own design would make
every comparison in docs/evidence/m5-perplexity.txt an apples-to-oranges one.

### Q4_0's scale takes its sign from the block's extreme value

Q4_0's range (-8..7) is asymmetric: one more negative code than positive, since 0 must be
representable. This project's first Q4_0 implementation scaled by plain `amax/8`
(`amax` = the block's largest |value|) and clamped to [-8, 7] -- which meant whichever sign
happened to hold the block's single most important value always needed the missing code (+8)
and lost up to a full quantization step rounding down to +7, exactly on the value a bad
approximation would hurt the most. This was caught by comparing this engine's own Q4_0
perplexity against llama.cpp's Q4_0 on the identical weights and finding this engine notably
worse (+47.8%/+21.5% quality loss vs llama.cpp's +22.4%/+6.2%, docs/evidence/m5-perplexity.txt
Finding 1) -- a real bug, not an inherent property of 4-bit quantization. Fixed by taking the
scale's sign from the extreme value itself (`scale = signed_extreme / -8`, matching ggml's
own `quantize_row_q4_0_reference`): that value's rounding then always lands on code -8, which
does exist, so it round-trips exactly regardless of its sign
(`tests/quant_test.cpp`'s `Q4_0ExtremeValueInEachBlockRoundTripsExactly`). This closed most,
but not all, of the gap to llama.cpp's own Q4_0 (docs/evidence/m5-perplexity.txt Finding 2) --
the remaining difference is measured and documented, not chased down further here.

### RMSNorm weights, biases and inv_freq stay float32

These are small (hundreds to low thousands of values each, versus millions per matmul
weight) and numerically sensitive -- RMSNorm's weight in particular directly scales every
hidden-state value at every layer, so a quantization error there compounds across the whole
forward pass rather than affecting one weight's contribution to one dot product. GGUF's own
Q8_0/Q4_0 make the same choice. The memory this leaves on the table is negligible next to
what quantizing the matmul weights saves (docs/evidence/m5-speed-and-memory.txt).

### A `Weight` wrapper dispatches per matrix, rather than templating `Transformer`

`Transformer`'s `Layer` struct holds `Weight` (`include/llmi/model/weight.hpp`) instead of
bare `std::vector<float>` for every matmul weight. `Weight` wraps either float32 storage
(calling `kernels::matmul` directly, M1-M4's exact code path) or a `quant::QuantizedMatrix`
(calling `quant::matmul`'s inline-dequant path), chosen once at `Transformer::load()` time.
This keeps `forward()`/`forward_cached()` as ONE implementation regardless of quantization --
the alternative (a separate `QuantizedTransformer` duplicating the whole forward pass) was
considered and rejected: it would double the surface area that M2/M3/M4's correctness work
has to stay right on, for a difference that's really about how one weight matrix is stored,
not how the forward pass is structured. Selecting `quant::Type::F32` (the default) keeps the
original `std::vector<float>` and calls `kernels::matmul` unchanged, so M1-M4's bit-exact
guarantees (the KV cache tests, the threading tests) are completely unaffected unless
quantization is actually requested -- checked directly by every existing test continuing to
pass without modification, plus new tests
(`Transformer.QuantizedForwardStaysCloseToFloat32AndPicksTheSameArgmax`) for the lossy case.

### No SIMD for the quantized matmul (yet)

`quant::matmul` dequantizes each block with a plain scalar loop. This was the single most
important thing M5's own measurement found: this engine's Q8_0/Q4_0 are 1.9-3.4x **slower**
than its own F32, while llama.cpp's Q8_0/Q4_0 are 4-8x **faster** than its own F32, because
llama.cpp's kernels unpack a register's worth of int8/nibbles and multiply-add against
float32 activations in one SIMD instruction sequence, instead of one scalar multiply per
weight (docs/evidence/m5-speed-and-memory.txt). A quantized SIMD dot product -- the natural
next step, mirroring what M4's `dot()` did for float32 -- was not built here: M5's scope was
the format, its correctness and honest measurement, the same boundary M4 drew around
cache-blocked GEMM rather than building everything a fully-optimized engine would have.

## Consequences

- Quantizing a model with this engine today trades memory for *more* time, not less -- the
  opposite of the usual reason to quantize, and worth knowing before choosing `--quant` for
  anything but a memory-constrained scenario specifically.
- Q4_0's quality loss, while now much closer to llama.cpp's own after the scale-sign fix, is
  still measurably larger on both models tested (docs/evidence/m5-perplexity.txt, Finding 2);
  the remaining difference is not fully explained (candidate causes: a different rounding
  function at the .5 boundary, or ggml's half-float block scales versus this engine's
  float32 ones, though the latter would be expected to make llama.cpp's number worse, not
  better, so it's an unlikely sole explanation).
- A SIMD quantized matmul, and K-quants or per-tensor/per-channel scale variants, are natural
  future milestones, left unbuilt here.
