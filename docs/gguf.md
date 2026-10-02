# GGUF: the hardened loader (M6)

M1-M5 read Hugging Face's own format, `model.safetensors` + `config.json`. M6 adds a second
reader for GGUF (`include/llmi/model/gguf.hpp`, `src/model/gguf.cpp`): the single-file format
llama.cpp and the wider ggml ecosystem use, carrying metadata, tensor shapes and tensor data in
one file. M6's scope is **the loader only** -- reading a GGUF file safely and correctly and
handing back float32 or `quant::QuantizedMatrix` data an engine could use. It does not wire GGUF
into `Transformer::load()` or `llmi-generate`/`llmi-chat`; that integration (choosing between a
safetensors directory and a single GGUF file at load time) is left to a later milestone. What
exists today is `llmi-gguf-inspect`, a standalone CLI that opens a GGUF file and prints its
metadata, tensor list and (for convertible types) per-tensor value checks.

## Why a second loader, and what it reuses

GGUF and safetensors are unrelated binary formats -- GGUF is a flat custom layout (magic,
counts, then metadata and tensor-info entries read field by field), not JSON-plus-a-data-section
-- so `gguf.cpp` is its own self-contained parser, following the same approach ADR 0001 already
set for safetensors rather than sharing a header-reading layer with it. What it does reuse:
`MappedFile` (memory-mapping, from `safetensors.hpp`) so tensor data is a view, never a copy;
`Result<T>` for errors as values, never exceptions; `llmi::read_float` (F32/F16/BF16 decode) for
`to_f32`; and `quant::QuantizedMatrix` (M5) as the target type for `to_quantized`, so a GGUF Q8_0
or Q4_0 tensor becomes the exact same in-memory representation this engine's own quantizer
produces.

## File layout (v3; this parser accepts no other version)

```
"GGUF" (4 bytes) | version: u32 | tensor_count: u64 | metadata_kv_count: u64
<metadata_kv_count metadata entries>
<tensor_count tensor-info entries>
<padding to `alignment`>
<tensor data, each tensor's bytes starting at its own alignment-padded offset>
```

A metadata entry is a `gguf_string` key (`u64` length + bytes, not null-terminated) followed by
a `u32` value-type tag and the value itself. Scalars (`UInt8`..`Float64`, `Bool`, `String`) are
widened into one `Value` struct (`as_uint`/`as_int`/`as_float`/`as_bool`/`as_string`) so callers
don't need a type-specific accessor per case. `Array` holds one element type and a flat list of
`Value` -- GGUF does not allow an array of arrays, and this parser rejects one (`depth`-bounded
recursion in `read_value`, see docs/gguf-threat-model.md) rather than silently flattening it.

A tensor-info entry is a name (same `gguf_string` encoding, bounded by `Limits::max_name_bytes`,
matching ggml's own `GGML_MAX_NAME`), a dimension count and that many `u64` dimensions
(`ne[0..n_dims)`), a `u32` ggml type code, and a `u64` byte offset relative to the start of the
(alignment-padded) data section.

**GGUF's shape order is reversed from this engine's own convention.** `ne[0]` is the
*fastest-varying* (innermost) dimension -- the opposite of PyTorch/safetensors' `[out, in]`
convention this engine's `TensorInfo::shape` already uses elsewhere. A 2-D weight matrix that
safetensors calls `[out, in]` is `ne = [in, out]` in GGUF. `to_quantized`'s `out`/`in` parameters
are in *this engine's* convention (so callers don't need to remember GGUF's reversal at every
call site); the function itself is what accounts for the difference.

**Default alignment is 32**, overridable by a `general.alignment` metadata entry (must be a
power of two, checked and bounded by `Limits::max_alignment`). Both the header-to-data-section
boundary and every individual tensor's own offset must land on a multiple of the alignment --
not just the first tensor, since each one is independently required to be aligned.

**GGUF permits gaps between tensors; it does not permit overlaps.** This is a real, deliberate
difference from safetensors (ADR 0001), which requires its tensors to tile the data section
exactly with no gaps and no holes. GGUF's own alignment requirement can force a small gap after
a tensor whose size isn't a multiple of `alignment` even when nothing malicious is going on, so
this parser's overlap check (`GGUFFile::parse`, sorting tensors by offset and checking each new
one starts no earlier than the previous one's end) allows gaps and rejects only overlaps. See
ADR 0007 for why this is a deviation from, not a relaxation of, M1's safetensors invariant.

## Element types: which ones convert to usable data, and why only four

ggml has assigned 40+ tensor-type codes over its history (`kTensorTypes` in `gguf.cpp`, every
one of them checked against the reference `gguf` Python package's own `GGML_QUANT_SIZES`
table). This parser **structurally validates every one of them** -- a tensor of any recognized
type gets its element count, byte size, offset and alignment checked exactly like any other, so
a model file with, say, Q6_K tensors loads and reports correctly -- but only four convert to
actual usable values:

- `to_f32`: F32, F16, BF16 (a straight `read_float` decode, reusing M1's float conversion).
- `to_quantized`: Q8_0, Q4_0 (ggml's own 32-weight block formats, matching M5's own
  `quant::Type::Q8_0`/`Q4_0` -- see "The Q4_0 layout surprise" below for why this is not a
  `memcpy`).

Every other type (Q5_0/Q5_1/Q8_1, the whole K-quant family, the IQ family, I8/I16/I32/I64/F64,
TQ1_0/TQ2_0, MXFP4/NVFP4, Q1_0) is recognized, sized and bounds-checked, but `to_f32`/
`to_quantized` return an error for them rather than guessing at a conversion. This is the same
stance M1 took on unsupported safetensors dtypes and architectures: an explicit, declared
limitation is safer than a silent wrong answer. Supporting more of them (K-quants in
particular, since they're llama.cpp's most common default today) is natural future work, not
built here.

## The Q4_0 layout surprise

This is the one real correctness finding of M6, caught by testing against real GGUF files
rather than assumed correct from the written spec. The full comment lives in `gguf.cpp` next to
`to_quantized`; the short version:

GGUF's Q8_0 block (`{ fp16 scale; int8 qs[32]; }`) packs its 32 codes in plain sequential order
-- byte-for-byte identical to this engine's own Q8_0 layout (`quant.cpp`, M5), so converting it
is a direct copy. GGUF's Q4_0 block (`{ fp16 scale; uint8 qs[16]; }`) does **not** pack adjacent
elements into each byte the way this engine's own Q4_0 does. ggml's reference quantizer packs
element `j` into byte `j`'s low nibble and element `j + 16` into that *same byte's* high nibble
-- the block's first and second halves share a byte, not neighbouring elements. A naive port of
this engine's own M5 Q4_0 packing (adjacent pairs: byte `k` holds elements `2k` and `2k+1`) onto
GGUF's bytes would silently produce a tensor with every value in the wrong place. `to_quantized`
unpacks GGUF's half-block pairing explicitly and repacks into this engine's own adjacent-pair
layout, rather than a `memcpy`; see `write_engine_code` and the long comment above it in
`gguf.cpp`, and ADR 0007 for how this was found.

## Checked against reality, twice

1. **A hand-crafted unit test** (`Gguf.ToQuantizedQ4_0UnpacksGgufsHalfBlockPairingNotAdjacentPairs`,
   `tests/gguf_test.cpp`) with manually computed expected values: byte `0x91` at scale 2.0 gives
   element 0 = -14.0 (low nibble `0x1`, code `1-8=-7`, `-7*2=-14`) and element 16 = 2.0 (high
   nibble `0x9`, code `9-8=1`, `1*2=2`).
2. **A full real-file cross-check** (`tools/crosscheck_gguf.py`) against six real GGUF files
   produced by llama.cpp's own converter (`convert_hf_to_gguf.py`) and `llama-quantize` from the
   same SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct weights M1-M5 already use, at all three
   precisions (F32/Q8_0/Q4_0). Checked two independent ways: structure against the reference
   `gguf` Python package, and every tensor's actual dequantized values against a from-scratch
   NumPy reimplementation of ggml's own block layout sharing no code with `gguf.cpp`. All 6
   files, 272/272/272/290/290/290 tensors, **0 mismatches**
   ([evidence](evidence/m6-crosscheck.txt)). This is what caught the layout discrepancy on real
   model weights, not only the synthetic unit test.

## Checked against malformed and adversarial input

20 hand-crafted unit tests (`tests/gguf_test.cpp`) cover specific invalid-input categories --
this is the "regression set of malformed files" the roadmap calls for, expressed as code, the
same role `safetensors_test.cpp`'s equivalent tests already serve for M1. A 60-second libFuzzer
run (`fuzz/fuzz_gguf.cpp`, seeded by `tools/make_fuzz_seeds.py`) found zero crashes with real
coverage (4090 edges; [evidence](evidence/m6-fuzz.txt)). See docs/gguf-threat-model.md for the
specific attack classes both are checking against.

## `llmi-gguf-inspect`

```
$ build/llmi-gguf-inspect model.gguf
version            3
alignment          32
metadata entries   24
tensors            272
architecture       llama
name               SmolLM2-135M-Instruct
tensor data bytes  538004480

$ build/llmi-gguf-inspect model.gguf --tensor-stats | head -1
{"name":"token_embd.weight","type":"Q8_0","shape":[576,49152],"n_elements":28311552,"data_nbytes":30951072,"offset":0,"sum":...,"first":[...]}
```

`--tensor-stats` prints one JSON line per tensor (name, type, shape, element count, byte size,
offset, and for convertible types a value sum and the first 8 values); this is the format
`tools/crosscheck_gguf.py` reads to cross-check structure and values against independent
readers.
