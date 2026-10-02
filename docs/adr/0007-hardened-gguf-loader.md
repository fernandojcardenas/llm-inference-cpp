# ADR 0007: A hardened GGUF loader, scoped to reading -- not wiring into inference

Status: accepted (M6)

## Context

M1 built a hardened safetensors + JSON loader and explicitly flagged ADR 0001's approach as
carrying over to a future GGUF loader, citing llama.cpp's own GGUF CVEs (CVE-2026-27940,
CVE-2026-33298) as the motivating example. M6 is that loader. GGUF is a different binary
format from safetensors -- not JSON plus a data section, but a flat custom layout read field by
field -- so it needed its own parser, but the same standard: every size, offset and count the
file claims about itself is checked before it's trusted, overflow-checked arithmetic
throughout, errors as `Result<T>` values, fuzzed, with a regression set of malformed inputs.

## Decisions

### Scope M6 to the loader itself, not full `Transformer`/`Model` integration

`GGUFFile::parse()`/`open()`, `to_f32()` and `to_quantized()` read a GGUF file and hand back
engine-usable data (a float vector or a `quant::QuantizedMatrix`), and `llmi-gguf-inspect` is a
standalone CLI to exercise and inspect that. None of this is wired into
`Transformer::load()`, `llmi-generate`, or `llmi-chat` -- those still only read a safetensors
directory. This mirrors the boundary M4 drew around cache-blocked GEMM and M5 drew around
quantized SIMD: GGUF support is a large enough piece of surface (one binary format, 40+ tensor
type codes, a metadata system with its own nested-array rules) that doing it right and proving
it right is a milestone on its own; routing an actual forward pass through it, and deciding how
`Transformer` picks between a safetensors directory and a single GGUF file, is better left to
land on top of a loader already known to be correct rather than built and proven at the same
time as the loader itself.

### GGUF's own tensor-packing invariant is gaps-allowed, overlaps-forbidden -- not safetensors' exact tiling

M1's safetensors reader requires tensors to tile the data section exactly: no gaps, no overlaps,
nothing after the last tensor. GGUF's own alignment requirement (every tensor's offset must be a
multiple of `alignment`, independently of every other tensor) means a tensor whose byte size
isn't itself a multiple of the alignment legitimately leaves a small gap before the next
tensor's aligned offset -- this is how real GGUF files produced by llama.cpp's own converter are
laid out, not a sign of something wrong. Requiring exact tiling (porting M1's safetensors rule
unchanged) would have rejected every real file with a non-alignment-multiple tensor size, which
is common. The decision: sort tensors by offset and check only that no tensor starts before the
previous one's claimed end (an overlap); a gap is accepted. This is a real, deliberate deviation
from M1's own invariant, not a relaxation of safety -- an overlap (two tensors claiming the same
bytes) is still exactly as invalid as it is for safetensors, and still rejected
(`Gguf.RejectsOverlappingTensorsButAllowsAlignmentGaps`).

### `general.alignment` is read and validated as part of parsing, not deferred to the caller

GGUF's default alignment (32) can be overridden by a `general.alignment` metadata entry, and
every later offset check (the header-to-data-section boundary, every tensor's own offset) is
relative to whatever alignment the file declares. This is validated eagerly, while the metadata
loop reads it, rather than read-and-stored for some later caller to check: zero, a non-power-of-
two value, or a value past `Limits::max_alignment` are all rejected on the spot
(`Gguf.RejectsBadAlignment`), because every alignment check downstream in `parse()` assumes a
valid power-of-two value without re-checking it at each use site -- deferring the validation
would mean every one of those call sites needs its own defensive check instead of relying on an
invariant `parse()` already established.

### The Q4_0 layout discrepancy: found by testing against real files, not assumed from the spec

GGUF's Q4_0 block (`{ fp16 scale; uint8 qs[16]; }`) packs element `j` into byte `j`'s low nibble
and element `j + 16` into that same byte's high nibble -- the block's two halves share a byte.
This engine's own Q4_0 (`quant.cpp`, M5) packs *adjacent* elements into each byte instead (byte
`k` holds elements `2k`/`2k+1`), a different, self-consistent convention chosen for a format
this engine both writes and reads itself. A GGUF reader that assumed the two layouts matched
and used a direct `memcpy` would silently read every GGUF Q4_0 tensor with its values in the
wrong positions -- row-correct in aggregate statistics (the same 32 values, divided the same
way into halves-worth of bytes) but wrong per element, the kind of bug that would not be obvious
from a sum or a spot check and would only show up as degraded model quality, hard to trace back
to "the loader transposed nibble positions."

This was found exactly the way M5's own scale-sign bug (ADR 0006) was found: by testing against
real reference data rather than trusting a reading of the format, in this case llama.cpp's own
GGUF files for the same two real models M1-M5 already use, cross-checked against an independent
NumPy reimplementation of ggml's actual block layout. The fix is an explicit unpack-then-repack
(`to_quantized`, `write_engine_code`), not a `memcpy`: GGUF's half-block-paired nibbles are
unpacked into signed codes, then written into this engine's adjacent-pair layout one element at
a time. Verified twice -- a hand-crafted unit test with manually computed expected values, and
the full six-file real-model cross-check, zero mismatches on either (docs/gguf.md, [evidence]
(../evidence/m6-crosscheck.txt)).

### Every ggml tensor type is structurally validated; only four convert to usable values

ggml has assigned 40+ tensor-type codes over its history. Rejecting every type this engine
cannot convert to float32 or a `quant::QuantizedMatrix` at the *parse* stage (treating "I can't
use this value" as "the file is invalid") would make the loader reject real, well-formed GGUF
files that simply use a quantization this engine hasn't implemented a converter for yet (any of
the K-quants, for instance -- llama.cpp's own most common default today). Instead, every
recognized type is sized, bounds-checked and validated like any other at parse time (so a file
using an unimplemented type still gets the same structural guarantees), and only `to_f32`/
`to_quantized` -- called once a caller actually wants that tensor's values -- return an explicit
error for a type they don't convert. This mirrors M1's stance on unsupported safetensors
dtypes and architectures: an explicit, declared limitation over a silent wrong answer, drawn at
the point where it actually matters (converting to a value) rather than the point where it
would needlessly reject files that are perfectly valid, just not yet fully supported.

## Consequences

- GGUF models can be inspected and validated today (`llmi-gguf-inspect`), and F32/F16/BF16/
  Q8_0/Q4_0 tensors can be converted to engine-usable data, but no app routes an actual forward
  pass through a GGUF file yet -- that integration is future work, not M6's.
- Supporting more ggml tensor types (the K-quants especially) is natural future work: the
  structural validation already covers them, only `to_f32`/`to_quantized`-equivalent conversion
  functions are missing.
- The gaps-allowed/overlaps-forbidden tensor-packing rule is specific to GGUF and intentionally
  different from safetensors' exact-tiling rule in the same codebase; a reader of both parsers
  side by side should expect that difference rather than read it as an inconsistency.
