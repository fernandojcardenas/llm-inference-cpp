# llm-inference-cpp

An LLM inference engine written from scratch in C++20. It loads small open-weights models (Apache 2.0:
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) and
[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)) and will generate text on an ordinary laptop CPU,
with no GPU and no Python at run time.

Every stage is checked against the reference implementation (Hugging Face `tokenizers`, `safetensors` and
PyTorch) on real model files, in CI, on every push. Model files are untrusted input, so every parser is
bounds-checked and fuzzed.

**Status: milestone 1 of 7 done** (model loading and tokenization). Text generation starts at M2. See the
[roadmap](docs/roadmap.md).

## Why

I'm working toward an M.S. in Computer Science with a focus on large language models. Calling a model
through an API teaches very little about how it works, so this project builds the inference side from
nothing: file formats, tokenizer, the transformer itself, then speed, quantization and serving. Each piece is
checked against a known reference before the next one is built on top of it.

## What works today (M1: loading and tokenization)

- **Model loader.** Reads Hugging Face `config.json` and `model.safetensors`, memory-maps the weights (nothing
  is copied), and checks every tensor against the architecture: 272 tensors for SmolLM2-135M, 290 for
  Qwen2.5-0.5B, each present with the right shape and a float dtype, nothing unexpected. Llama and Qwen2
  layouts (grouped-query attention, optional q/k/v bias, tied embeddings).
- **Hardened file parsing.** The safetensors header and both JSON files go through the engine's own
  strict parser. It rejects offsets out of bounds, overlapping tensors, holes in the data section, sizes that
  don't match dtype × shape (checked without integer overflow), duplicate keys, invalid UTF-8 and excessive
  nesting. Unsupported model features (RoPE scaling, other architectures) are errors, never silently ignored.
- **Byte-level BPE tokenizer.** Reads `tokenizer.json`: special tokens, digit splitting, the GPT-2
  pre-tokenizer pattern written out by hand over Unicode tables, BPE merges by rank, and decoding.

```
$ build/llmi-tokenize testdata/smollm2-135m/tokenizer.json "The year 2026 had 365 days."
```

gives ids `504 713 216 34 32 34 38 761 216 35 38 37 2009 30`, the same as Hugging Face. Each digit is its own
token in this vocabulary.

### Checked against the reference

| Check | Result | Evidence |
|---|---|---|
| Token ids vs Hugging Face `tokenizers` 0.23.2: 586 Python standard-library files, 50,000 random adversarial strings, and **every Unicode code point** in four contexts | **4,498,842 of 4,498,842 inputs identical** (30.5 million tokens) | [output](docs/evidence/m1-tokenizer-crosscheck.txt) |
| Weights vs PyTorch: every tensor's dtype, shape, first values and sum | SmolLM2-135M: 272 tensors, 134,515,008 values identical. Qwen2.5-0.5B: 290 tensors, 494,032,768 values identical | [output](docs/evidence/m1-weights-crosscheck.txt) |
| Which Unicode version each tokenizer step uses, probed over all 1,112,064 code points | Pattern classes: Unicode 16.0, 0 differences. Digit splitting: Unicode 17.0, 0 differences | [output](docs/evidence/m1-unicode-probe.txt) |

Getting to exact agreement found two things the reference does that are easy to miss
([details](docs/tokenizer.md)):

1. **The reference tokenizer uses two Unicode versions at once.** Its regex engine classifies letters and
   numbers by Unicode 16.0; its digit splitter uses Rust's `char::is_numeric`, which is Unicode 17.0. With
   tables from a single version, 3 of 50,586 inputs came out different. The engine now carries both, and
   matches on every code point.
2. **SmolLM2's vocabulary has no token for 21 of the 256 byte values**, six of them ASCII control
   characters (U+0004, U+0006, U+0013, U+0014, U+0016, U+001D). The reference silently drops those
   characters, so the model never sees them. This engine does the same, to stay token-for-token compatible,
   but counts what it drops and reports it, so a caller can refuse such input instead.

### How it's tested

| Check | What it proves | Where |
|---|---|---|
| 34 unit tests (GoogleTest) | UTF-8 edge cases; JSON grammar, escapes, exact 64-bit integers, depth and size limits; every safetensors rejection rule, including shapes that overflow 64 bits; float conversion (F32, F16 incl. subnormals and infinity, BF16); config validation; weight-layout checks; known token ids from the reference; round trips; dropped bytes | `tests/` |
| Tokenizer cross-check | Identical token ids to Hugging Face on 4.5 million inputs | `tools/crosscheck_tokenizer.py`, CI `crosscheck` job |
| Weight cross-check | Every tensor of two models read exactly as PyTorch reads it | `tools/crosscheck_weights.py`, CI `crosscheck` job |
| ASan + UBSan | No memory errors or undefined behaviour, with GCC and Clang | CI `test` job |
| macOS arm64 | Builds and passes with Apple Clang, the target laptop platform | CI `macos` job |
| Four libFuzzer targets | Arbitrary bytes into the JSON parser, the safetensors parser (output must tile the input exactly), the config and tokenizer loaders, and encode/decode (valid UTF-8 must round-trip) | `fuzz/`, CI `fuzz` job |
| clang-tidy | bugprone, cert, performance, modernize and readability checks, warnings as errors | `.clang-tidy`, CI `lint` job |

## Build

Requires CMake 3.24+ and a C++20 compiler (GCC 13, Clang 18 or Apple Clang). GoogleTest is downloaded at
configure time with a pinned SHA-256.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Options: `-DLLMI_SANITIZE=ON` (ASan + UBSan), `-DLLMI_BUILD_FUZZERS=ON` (Clang with libFuzzer),
`-DLLMI_WARNINGS_AS_ERRORS=ON`.

## Use

```
tools/fetch_model.sh smollm2-135m models/smollm2-135m   # pinned revision, SHA-256 checked
build/llmi-inspect models/smollm2-135m                  # architecture, tensors, parameter count
build/llmi-tokenize models/smollm2-135m/tokenizer.json "Hello, world!"
```

To run the cross-checks yourself: `pip install tokenizers==0.23.2 safetensors torch`, then see the commands at
the top of each file in `docs/evidence/`.

## Layout

```
include/llmi/   public headers: util/ (JSON, UTF-8, Result), model/ (safetensors, config, model), tokenizer/
src/            implementation; src/tokenizer/unicode_tables.inc is generated from the Unicode database
apps/           llmi-inspect, llmi-tokenize
tests/          unit tests
fuzz/           libFuzzer targets
tools/          model download, cross-checks, Unicode table generator and probe, fuzz seeds
testdata/       SmolLM2-135M tokenizer.json and config.json (Apache 2.0)
docs/           roadmap, tokenizer notes, architecture decisions, evidence from real runs
```

## Data and licences

Model weights are never committed; `tools/fetch_model.sh` downloads them from Hugging Face at a pinned revision
and checks each file's SHA-256. Third-party material keeps its own licence; see [THIRD-PARTY.md](THIRD-PARTY.md).
Design decisions are recorded in [docs/adr](docs/adr).

## Licence

MIT. See [LICENSE](LICENSE).
