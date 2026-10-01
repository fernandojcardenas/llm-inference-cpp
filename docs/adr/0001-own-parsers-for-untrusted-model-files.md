# ADR 0001: Own parsers for untrusted model files

**Status:** accepted (M1)

## Context

An inference engine reads files it did not produce: weights, configs and tokenizers downloaded from the
internet. Parsers for these formats are attack surface. In 2026 llama.cpp's GGUF parser had integer-overflow
bugs that turned a malicious model file into out-of-bounds reads and writes (CVE-2026-27940, CVE-2026-33298).
The safetensors format was designed to be simple to parse safely, but only if the reader checks everything the
format promises.

## Decision

- The engine parses JSON (the safetensors header, `config.json`, `tokenizer.json`) with its own strict parser:
  valid UTF-8 only, depth and node limits, duplicate keys rejected, numbers kept as text so 64-bit offsets are
  read exactly instead of through a double.
- The safetensors reader checks every rule of the format before handing out a pointer: header length against
  the file, offsets within the data section, `end - begin == dtype size × product(shape)` computed with
  overflow checks, no overlaps, no holes, nothing after the last tensor, known dtypes only, no unknown fields.
- Weights are memory-mapped read-only; tensors are views, never copies.
- Errors are values (`Result<T>`), never exceptions or aborts, so a bad file can't take down a server.
- Every parser has a libFuzzer target with invariants beyond "doesn't crash" (for safetensors: accepted
  tensors must tile the input exactly).

## Consequences

- No third-party JSON or safetensors dependency in the engine; the reference libraries are used only to
  cross-check, in Python, outside the engine.
- More code to maintain and test, which the unit tests and fuzzers cover.
- The same approach carries over to the GGUF loader in M6.
