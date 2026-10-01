# Tokenizer: matching the reference exactly

The model was trained on token ids produced by Hugging Face `tokenizers`. One different id changes what the
model sees, so the target is not "close": it is identical ids for every input. This page records how the
engine gets there and what that turned up.

## What it implements

SmolLM2's `tokenizer.json` (like GPT-2's, Llama 3's and Qwen's) describes byte-level BPE:

1. **Added tokens** such as `<|im_start|>` are split out first, leftmost-longest match.
2. **Digits** (`individual_digits: true`): every numeric character becomes its own piece.
3. **ByteLevel pre-tokenizer**: each piece is split with the GPT-2 pattern
   `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
   The engine has no regex library; `gpt2_match` in `src/tokenizer/tokenizer.cpp` implements the alternatives
   in order, including the backtracking that `\s+(?!\S)` implies (a run of spaces before a word leaves its
   last space to prefix the word).
4. Each piece's UTF-8 bytes map to printable characters (GPT-2's `bytes_to_unicode`) and **BPE merges** apply
   lowest rank first, ties broken by position, as the reference does.

Anything else a `tokenizer.json` can contain (normalizers, other pre-tokenizers, byte fallback, dropout,
template post-processors) is rejected at load time. Qwen2.5's tokenizer uses an NFC normalizer and a
different split pattern, so it is refused for now, with a clear error; it is on the [roadmap](roadmap.md).

## How it's checked

`tools/crosscheck_tokenizer.py` runs both tokenizers on the same inputs and compares ids:

- every `.py` file in the Python standard library, each as one document (code, prose, numbers, indentation);
- 50,000 seeded random strings built from the hard cases: all 25 Unicode White_Space characters and look-alikes
  that are not whitespace (zero-width space, U+001C–U+001F), letters and digits from many scripts, combining
  marks, emoji sequences, contractions in both cases, special tokens, control characters, random code points;
- with `--all-codepoints`, **every one of the 1,112,064 Unicode scalar values** in four contexts (`c c x`,
  `c 5`, `a c b`, ` c 1`) that together expose how each step classifies it.

Result: [4,498,842 of 4,498,842 inputs identical](evidence/m1-tokenizer-crosscheck.txt).

## Finding 1: the reference uses two Unicode versions

With letter and number tables from Python's Unicode 15.1 database, the first run matched 50,583 of 50,586
inputs. All three mismatches involved code points new in Unicode 16.0 (Garay U+10D56, Egyptian hieroglyphs
U+13A63 and U+13E77): the reference treats them as letters, so `Y𓹷'm` splits as `Y𓹷` + `'m`, not `Y` +
`𓹷'` + `m`.

Rather than guess the version, `tools/probe_reference_unicode.py` asks the reference directly. For every code
point whose class changed between Unicode 15.1, 16.0 and 17.0, it checks how the pattern and the Digits
splitter treat it ([output](evidence/m1-unicode-probe.txt)):

| Step | New in 16.0 | New in 17.0 | Version | All code points |
|---|---|---|---|---|
| Pattern `\p{L}` | 4,302 of 4,302 treated as letters | 0 of 4,644 | 16.0 | 0 differ |
| Pattern `\p{N}` | 80 of 80 | 0 of 13 | 16.0 | 0 differ |
| Digits splitter | 80 of 80 | 13 of 13 | 17.0 | 0 differ |

The pattern classes come from the regex engine's Unicode tables (16.0); the Digits splitter uses Rust's
`char::is_numeric`, from the Rust standard library (17.0). For example, the 13 numeric characters added in 17.0
(the Tolong Siki digits U+11DE0–U+11DE9 and three Yangqin beat signs U+16FF4–U+16FF6) are split off one by one
as digits but are not `\p{N}` to the pattern.
`tools/gen_unicode_tables.py` now builds both sets of tables from the official `UnicodeData.txt` of each
version, and both are compiled in. These versions belong to `tokenizers` 0.23.2; a newer release may move
them, which the CI cross-check would catch.

## Finding 2: some bytes have no token, and the reference drops them

The tokenizer cannot be lossless unless every byte has a token. SmolLM2's vocabulary has none for 21 bytes:

- 15 that never occur in valid UTF-8 (0xC0, 0xC1, 0xF5–0xFF) or only begin 4-byte sequences for planes 4–11
  (0xF1, 0xF2), which have no assigned characters;
- **6 ASCII control characters: U+0004, U+0006, U+0013, U+0014, U+0016, U+001D.**

With `unk_token: null`, the reference skips such bytes without an error: `"a\x04b"` encodes to the tokens of
`"ab"`, and BPE can even merge the neighbours across the gap. The model never sees the character, and nothing
says so.

The engine reproduces this exactly (the ids must match), but `Tokenizer::encode` reports how many bytes it
dropped, `llmi-tokenize` prints a warning, and the round-trip check in `--jsonl` mode counts lossy lines
separately instead of hiding them. A server built on the engine (M7) can refuse such input. Hidden characters
that the safety layer sees but the model doesn't, or the other way round, are a known way to slip text past
filters, which is why this matters beyond correctness.

## Speed (not a benchmark yet)

On the 4.5-million-input cross-check, `llmi-tokenize` took 9.9 s single-threaded including JSON parsing and
output; Hugging Face took 39.0 s through its Python binding, batched, on 2 cores. The two measurements include
different overheads, so this only shows that tokenization will not be the bottleneck. Proper benchmarks come in
M4.
