# ADR 0002: Match the reference tokenizer exactly, and report what it hides

**Status:** accepted (M1)

## Context

A model only works on the token ids it was trained with, so the tokenizer must produce exactly what Hugging
Face `tokenizers` produces. Two behaviours of the reference turned up while getting there
([details](../tokenizer.md)): it uses Unicode 16.0 for the pattern's letter and number classes but Unicode 17.0
for digit splitting, and it silently drops bytes that have no token in the vocabulary (six ASCII control
characters in SmolLM2).

## Decision

- Exact agreement is the requirement, tested on every Unicode code point, not a sample.
- Character classes come from the official `UnicodeData.txt` of each version the reference uses, generated
  into source by a script, not from whatever Unicode version the build machine's Python has.
- Dropped bytes are dropped (to match), but counted and reported by `Tokenizer::encode`, so callers can refuse
  lossy input.
- Features the engine doesn't implement are load-time errors. A tokenizer that quietly ignored a normalizer
  would produce wrong ids without any sign of it.
- The reference version is pinned in CI (`tokenizers==0.23.2`); an upgrade that moves its Unicode versions
  will fail the cross-check rather than drift.

## Consequences

- Qwen2.5's tokenizer is refused until its NFC normalizer and split pattern are implemented.
- The Unicode tables must be regenerated if the reference changes versions; the probe script shows which.
