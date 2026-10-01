# Quantization: 8-bit and 4-bit weights

M1-M4 store every weight as float32. M5 adds two smaller, lossy formats -- Q8_0 (8-bit) and
Q4_0 (4-bit), named and laid out the same way GGUF's own Q8_0/Q4_0 are -- so a weight this
engine quantizes and llama.cpp quantizing the identical float32 source are directly
comparable: same granularity, same symmetric-scale convention, same expected error.
Quantizing a model trades memory for quality (a little, measured honestly below) and, in
this engine specifically, for *more* time, not less (also measured honestly below).

## The format

Every matmul weight matrix (`out x in`, the same layout `kernels::matmul`'s `w` has) is split
row by row into blocks of 32 weights (`quant::kBlockSize`). Each block gets its own float32
scale, computed from the block's own largest-magnitude weight -- so weights in the same row
that happen to have very different typical sizes (common across layers and architectures)
each get a scale suited to their own block, not one scale forced across the whole row.

- **Q8_0**: one signed `int8` per weight, value = `q * scale`, `scale = amax / 127`.
- **Q4_0**: one unsigned nibble per weight (two pack into a byte), stored with a +8 bias so
  `q` reconstructs to `-8..7`; value = `(q - 8) * scale`.

Q4_0's range is asymmetric: `-8..7` has one more negative code than positive, since one code
has to represent 0. Scaling by plain `amax/8` would mean whichever sign held the block's
single most important value (the one a bad quantization affects most) always needed the code
that doesn't exist (+8) and lost up to a whole quantization step rounding down to +7 -- a
real bug this project's own first implementation had, found by comparing against llama.cpp's
own Q4_0 on identical weights (docs/evidence/m5-perplexity.txt, Finding 1) and fixed by
taking the scale's *sign* from that extreme value (`scale = signed_extreme / -8`, the same
convention ggml's `quantize_row_q4_0_reference` uses): that value's rounding then always
lands on code -8, which does exist, so it reconstructs with zero error regardless of its
sign (`tests/quant_test.cpp`'s `Q4_0ExtremeValueInEachBlockRoundTripsExactly`).

RMSNorm weights, attention biases and RoPE's `inv_freq` stay float32 regardless of
`--quant`: they're small (a few hundred to a couple thousand values each, versus millions per
matmul weight) and numerically sensitive, the same choice GGUF's own Q8_0/Q4_0 make.

## How a quantized matmul runs

`quant::matmul` (`src/model/quant.cpp`) dequantizes each block inline while computing the dot
product -- scale is a per-block constant, so `sum(x_i * dequant(w_i)) = scale * sum(x_i *
q_i)`, meaning the scale multiply happens once per block rather than once per weight. It
never materializes a full dequantized float row (that would give back F32's memory cost for
no reason), and it parallelizes over output rows through the same shared thread pool M4's
`kernels::matmul` uses, with the same reasoning: different output rows write disjoint memory,
so no locking is needed and small matmuls skip the pool entirely.

What it does *not* have is SIMD: the block loop is a plain scalar `for`, unlike
`kernels::dot()`'s AVX2/NEON paths (M4). That is the direct cause of this milestone's most
important honest finding -- see "What this doesn't do" below.

`Transformer::load()` (`include/llmi/model/transformer.hpp`) takes a `quant::Type` (default
`F32`, unchanged behavior); requesting `Q8_0` or `Q4_0` quantizes every weight a matmul reads
at load time through a small `Weight` wrapper (`include/llmi/model/weight.hpp`) that
dispatches `matmul()`/`row()` (the embedding lookup) to either `kernels::matmul` or
`quant::matmul` depending on how that particular weight was loaded. Selecting `F32` keeps
the original `std::vector<float>` storage and calls `kernels::matmul` directly -- so M1-M4's
bit-exact tests are completely unaffected unless `--quant` is actually requested (ADR 0006).

## Quality: perplexity on real text

Measured on a prefix of wikitext-2-raw-v1 (the standard small perplexity benchmark), scored
the same way llama.cpp's own perplexity tool scores by default (only the second half of each
512-token window, so every scored position has substantial left-context) -- full methodology
and two real findings along the way (the Q4_0 bug above, and a smaller remaining gap against
llama.cpp not fully explained) in
[docs/evidence/m5-perplexity.txt](evidence/m5-perplexity.txt):

| Model | F32 | Q8_0 | Q4_0 |
|---|---|---|---|
| SmolLM2-135M-Instruct | 18.9581 | 18.9922 (+0.18%) | 25.8341 (+36.3%) |
| Qwen2.5-0.5B-Instruct | 13.1518 | 13.1513 (−0.00%) | 14.5068 (+10.3%) |

llama.cpp quantizing the identical float32 weights the identical way: 18.9581 / 19.0250 /
23.2110 (SmolLM2) and 13.1494 / 13.2081 / 13.9677 (Qwen2.5) -- F32 and Q8_0 match closely
(both models, both engines); Q4_0 is where a real, if now much smaller, gap remains.

Q8_0's quality loss is negligible on both models. Q4_0's is real and size-dependent: much
bigger on the smaller SmolLM2 (135M parameters) than on Qwen2.5 (494M) -- consistent with the
well-known pattern that smaller models have less redundancy in their weights to absorb
aggressive rounding, and the same ordering llama.cpp's own numbers show.

## Speed and memory: the honest headline

| | F32 | Q8_0 | Q4_0 |
|---|---|---|---|
| Memory (SmolLM2) | 512.9 MiB | 144.3 MiB (3.56x smaller) | 80.2 MiB (6.40x smaller) |
| Memory (Qwen2.5) | 1884.4 MiB | 530.0 MiB (3.56x smaller) | 294.4 MiB (6.40x smaller) |
| Generation, this engine (SmolLM2) | 23.68 tok/s | 12.51 tok/s | 6.87 tok/s |
| Generation, llama.cpp (SmolLM2) | 46.05 tok/s | 213.63 tok/s | 341.33 tok/s |

Full tables (both models) and methodology: [docs/evidence/m5-speed-and-memory.txt](evidence/m5-speed-and-memory.txt).

Quantizing with this engine trades memory for *more* time, not less: this engine's Q8_0/Q4_0
are 1.9-3.4x **slower** than its own F32, while llama.cpp's are 4-8x **faster** than its own
F32. The difference is entirely `quant::matmul`'s scalar dequantize-and-dot loop, versus
llama.cpp's hand-written SIMD kernels that unpack a register's worth of int8/nibbles and
multiply-add against float32 activations in one instruction sequence. For matmuls this size
on a 2-core machine with no real memory-bandwidth pressure, losing SIMD costs far more than
the smaller footprint saves. This is the same kind of gap M4 drew an honest line around for
cache-blocked GEMM: a quantized SIMD dot product is the natural next step, and isn't built
here (ADR 0006) -- M5's scope was the format, its correctness, and measuring both honestly.

## What this doesn't do

No SIMD for quantized matmul (the speed finding above is the direct cost of this). No
per-channel or per-tensor scale variants (block size 32 only, matching GGUF's own Q8_0/Q4_0
block size for comparability). No K-quants or any mixed-precision scheme -- just the two
simplest, best-understood GGUF formats. No quantization-aware training or calibration data:
both formats quantize a model that was never trained with quantization in mind, the same way
llama.cpp's own Q8_0/Q4_0 do. The small remaining Q4_0-vs-llama.cpp perplexity gap (Finding 2
in the evidence file) is measured, not fully explained.
