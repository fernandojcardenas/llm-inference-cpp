# GGUF threat model

A GGUF file is untrusted input: something downloaded from the internet, exactly the trust
boundary ADR 0001 drew around safetensors, JSON configs and the tokenizer. In 2026 llama.cpp's
own GGUF parser had real integer-overflow bugs that turned a malicious model file into
out-of-bounds reads and writes (CVE-2026-27940, CVE-2026-33298) -- the motivating example ADR
0001 cites for carrying this approach over to M6's GGUF loader. This document is the specific
attack surface `src/model/gguf.cpp` was written against, and how each one is closed and tested.

## The trust boundary

Everything in a GGUF file's header is a claim the file makes about itself: how many tensors it
has, how many metadata entries, how long each string or array is, what each tensor's shape and
byte offset are. None of these is trusted before use. Every count, length, offset and size is
checked against two things before it is allocated, indexed, or used to compute another check:

1. A configurable `Limits` struct (`include/llmi/model/gguf.hpp`) -- a sanity bound independent
   of the file's own claims, so a tiny file can't claim a tensor count of 2^63 and have this
   parser try to act on it before checking the file is even that long.
2. The bytes actually remaining in the buffer (`Cursor::has()`), checked before every read --
   not computed from a claimed length and trusted.

## Specific attack classes and how this parser closes them

**Integer overflow on count x size computations.** A shape like `[2^40, 2^40]` or a tensor
count near `2^64` could overflow a 64-bit multiply and wrap to a small, plausible-looking
number, which a careless implementation would then allocate or index with -- silently
truncating the real size and leaving the rest of a validation blind. Every multiplication that
turns a file-supplied count into a byte size (`numel`, `TensorInfo::size`, the data-section
padding arithmetic) goes through `mul64()`, which returns `nullopt` on overflow rather than a
wrapped value -- the same convention `safetensors.cpp`'s own `mul()` helper already established
for M1, duplicated here per this project's convention that each parser module is self-contained.
Tested by `Gguf.RejectsShapesThatOverflowOrAreNotBlockAligned`.

**Huge count fields causing unbounded allocation before validation.** `tensor_count` and
`metadata_kv_count` are checked against `Limits::max_tensors`/`max_metadata_entries`
*immediately* after being read, before the metadata or tensor-info loops allocate anything
sized by them (a `std::vector::reserve`, for instance). `Limits::max_rank` and
`max_array_len` apply the same check to a tensor's dimension count and a metadata array's
length before either is used to reserve or loop. Tested by `Gguf.RejectsCountsBeyondLimits`.

**Truncation at every stage of the header.** A file can be cut off at any byte boundary --
mid-magic, mid-count, mid-string, mid-tensor-info, mid-offset. Every `Cursor` read checks
`has(n)` first and returns `nullopt` rather than reading past the end of the buffer; every
caller of a `Cursor` read checks the result and fails with a specific message rather than
dereferencing an empty `optional`. Tested exhaustively by
`Gguf.RejectsTruncationAtEveryStageOfTheHeader`, which truncates the same valid file at every
7-byte step and checks every prefix is rejected, never crashes, and never succeeds.

**Malformed or oversized UTF-8 strings.** A `gguf_string`'s claimed length is checked against
both `Limits::max_string_bytes`/`max_name_bytes` and the bytes actually remaining before a
single byte is read or copied (`Cursor::string()`), and the bytes are validated as UTF-8
(`llmi::utf8::valid`) before being accepted -- the same rule this engine already applies to
every other string it parses (JSON, tokenizer vocab). Tested by
`Gguf.RejectsOversizedAndInvalidUtf8Strings`.

**Nested or type-confused metadata arrays.** GGUF's own grammar does not allow an array of
arrays, but a crafted file can still claim its array's element type *is* `Array`. `read_value`
takes a `depth` parameter, incremented on each recursive call for an array element, and rejects
`type == Array` arriving at `depth > 0` -- a file cannot defeat this by nesting arbitrarily deep,
since the first nested array is rejected outright, not merely bounded. An element type outside
the known `ValueType` range is rejected the same way an unknown top-level value type is. Tested
by `Gguf.RejectsNestedArrays` and `Gguf.RejectsUnknownValueAndTensorTypes`.

**Unknown or invalid-but-plausible values.** A `Bool` value's single byte is checked to be
exactly 0 or 1, not treated as "nonzero is true" (`Gguf.RejectsInvalidBoolValues`) -- a stricter
rule than C's own truthiness convention, chosen because a byte that is neither 0 nor 1 means the
file's claim about its own data is wrong in some other way too, and silently coercing it hides
that. An unrecognized tensor type or metadata value type is rejected rather than skipped or
defaulted.

**Tensor offset/size overflow and out-of-bounds reads.** A tensor's byte size is computed with
the same overflow-checked `mul64()` as its element count, then checked against the data
section's actual byte length (`t.offset > data_size || t.size > data_size - t.offset`, written
to avoid a second overflow in the subtraction) before `GGUFFile::data()` ever hands out a span
into it. Tested by `Gguf.RejectsMisalignedOffsetsAndOutOfBoundsOffsets`.

**Misaligned or overlapping tensor regions.** Every tensor's offset must be a multiple of the
file's alignment (default 32, or the validated `general.alignment` override); tensors are then
sorted by offset and checked to never start before the previous one ends. Gaps (alignment
padding between tensors) are allowed; overlaps are not -- see docs/gguf.md and ADR 0007 for why
this deliberately differs from safetensors' exact-tiling rule. Tested by
`Gguf.RejectsOverlappingTensorsButAllowsAlignmentGaps`, which also checks a legitimate
alignment-padding gap is accepted.

**An invalid or adversarial `general.alignment`.** Zero, a non-power-of-two value, or a value
larger than `Limits::max_alignment` are all rejected -- zero would make every later `% alignment`
check divide by zero or always report misalignment nonsensically, and a non-power-of-two value
breaks the `(a & (a - 1)) != 0` fast check every other part of the parser relies on to treat
alignment checks as simple bitwise operations. Tested by `Gguf.RejectsBadAlignment`.

**Duplicate keys and names.** A duplicate metadata key, or two tensors sharing the same name,
each have a well-defined meaning the file is contradicting itself about; both are rejected
outright rather than silently taking the first, the last, or merging them. Tested by
`Gguf.RejectsDuplicateKeysAndNames`.

## What this does not cover

M6 hardens *reading* a GGUF file's structure and converting the four supported element types to
usable data. It does not evaluate whether a model's *weights* are adversarial (a model that
behaves maliciously once loaded and run is a different, much harder problem, out of scope for a
file-format loader). It does not yet convert every ggml tensor type to usable values (see
docs/gguf.md's list of structurally-validated-but-not-convertible types) -- a type this parser
doesn't convert is still fully bounds-checked, just not exposed as float32 or quantized data.

## How this is tested

- 20 unit tests (`tests/gguf_test.cpp`), one or more per attack class above -- the "regression
  set of malformed files" the roadmap asks for, expressed as code rather than as a corpus of
  binary fixtures, matching M1's own precedent for safetensors.
- A 60-second libFuzzer run (`fuzz/fuzz_gguf.cpp`), seeded with valid files
  (`tools/make_fuzz_seeds.py`), checking invariants beyond "doesn't crash": every accepted
  tensor's offset/size stays within the file, alignment is always a power of two, a found
  tensor's `numel()` is always divisible by its type's block size, and no accepted metadata
  value claims to be an array of arrays. Zero crashes, 4090 edges of coverage in 60 seconds
  ([evidence](evidence/m6-fuzz.txt)).
- The real-file cross-check (`tools/crosscheck_gguf.py`) checks correctness on well-formed real
  files, not adversarial ones, but it is what caught the Q4_0 layout discrepancy -- a reminder
  that "rejects bad input" and "computes correct output on good input" are both required and
  neither implies the other.
