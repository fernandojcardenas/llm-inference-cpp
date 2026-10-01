# llm-inference-cpp

An LLM inference engine written from scratch in C++20. It loads small open-weights models (Apache 2.0:
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) and
[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)) and generates text on an ordinary laptop CPU,
with no GPU and no Python at run time.

Every stage is checked against the reference implementation (Hugging Face `tokenizers`, `safetensors` and
PyTorch) on real model files, in CI, on every push. Model files are untrusted input, so every parser is
bounds-checked and fuzzed.

**Status: milestones 1 through 4 of 7 done** (model loading, tokenization, a forward pass that matches PyTorch
layer by layer, a KV cache, sampling, an interactive chat CLI, and a threaded, SIMD matmul). Next:
quantization. See the [roadmap](docs/roadmap.md).

```
$ echo "What is the capital of France?" | build/llmi-chat models/smollm2-135m-instruct --max-new 20
llmi-chat: models/smollm2-135m-instruct loaded. Type a message and press enter (Ctrl-D to quit).
> The capital of France is Paris.
```

## Why

I'm working toward an M.S. in Computer Science with a focus on large language models. Calling a model
through an API teaches very little about how it works, so this project builds the inference side from
nothing: file formats, tokenizer, the transformer itself, then speed, quantization and serving. Each piece is
checked against a known reference before the next one is built on top of it.

## KV cache, sampling and chat (M3)

`include/llmi/model/transformer.hpp`'s `KVCache` + `Transformer::forward_cached()` keep every layer's past
keys and values, so generating a token only computes that one new position instead of recomputing the whole
sequence. It's checked by reproducing the no-cache baseline exactly — not by a fresh PyTorch cross-check,
since M2 already established the baseline is correct — and measured 14.8x (SmolLM2) and 7.4x (Qwen2.5) faster
on the same short runs ([details](docs/kv-cache.md)).

`llmi::sample()` (`include/llmi/model/sampling.hpp`) adds temperature, top-k and top-p sampling alongside
greedy decoding, checked against a plain softmax with no model involved; the same seed reproduces the same
draws ([details](docs/sampling.md)).

`llmi::chat::ChatTemplate` runs a model's own `chat_template` (a small Jinja2 program in
`tokenizer_config.json`) to format a message list the way that model was trained on, rather than hard-coding
one chat format — checked against `transformers`' own Jinja compiler, 9/9 cases identical on two different
real templates ([details](docs/chat-template.md)).

`build/llmi-chat MODEL_DIR` puts these together: an interactive, multi-turn chat CLI that reuses the KV cache
across turns.

## Performance: threads and SIMD (M4)

`kernels::matmul` is nearly all of the forward pass's time (profiled before optimizing anything:
[details](docs/performance.md)), so M4 makes it faster without changing what it computes. `llmi::util::ThreadPool`
(`include/llmi/util/thread_pool.hpp`) parallelizes matmul's independent output rows across a small, persistent
pool of worker threads — bit-for-bit identical to the serial version, since different output rows write
disjoint memory and the floating-point operations happen in the same order either way. `kernels::dot()` is
chosen at compile time: AVX2+FMA on x86_64, NEON on Apple Silicon (part of the base ISA there, no flag needed),
or the original portable scalar version as a fallback.

Measured on a 2-core machine, same tokens generated before and after: 2.05x (SmolLM2) and 2.39x (Qwen2.5)
faster generation ([details](docs/performance.md), [evidence](docs/evidence/m4-speed.txt)). Compared with
llama.cpp, same machine, same weights, f32 precision in both: within 1.5–1.6x at generation (the same
matrix-vector matmul shape in both engines); 2.5–2.7x slower at prompt processing, an honest gap from
llama.cpp's row-blocked GEMM, which this engine's matmul doesn't have yet
([evidence](docs/evidence/m4-llamacpp-comparison.txt)).

## Forward pass (M2)

`src/model/transformer.cpp` implements the Llama and Qwen2 decoder in float32: token embedding, then per layer
RMSNorm, query/key/value projections (with Qwen2's bias), rotary position embedding, causal grouped-query
attention, output projection, and the SwiGLU MLP, then the final norm and the output layer. Greedy generation
picks the top-scoring token each step. There is no key/value cache yet, so each step recomputes the whole
sequence ([how it works](docs/forward-pass.md)).

It is checked against Hugging Face transformers (PyTorch, float32) on 8 prompts per model: the embeddings, the
output of **every layer**, the final norm and all logits, then 24 greedily generated tokens:

| Model | Worst difference in any layer | Greedy tokens identical | Evidence |
|---|---|---|---|
| SmolLM2-135M (30 layers) | 5.4e-06 of the largest value | **192 of 192** | [output](docs/evidence/m2-forward-smollm2-135m.txt) |
| Qwen2.5-0.5B-Instruct (24 layers, q/k/v bias) | 2.7e-05 | **192 of 192** | [output](docs/evidence/m2-forward-qwen2.5-0.5b.txt) |

To test the check, four realistic bugs were planted one at a time (RoPE pairing, the grouped-query head mapping,
a causal mask that leaks one position, the wrong RoPE base). The layer comparison caught all four with
differences of 0.02 to 17. **Two of them still generated exactly the reference's tokens**: a mask that leaks the
next position changes every position except the last, which is the only one greedy decoding reads. Matching
text alone is not proof of a correct model ([details](docs/forward-pass.md#would-a-bug-get-through),
[output](docs/evidence/m2-planted-bugs.txt)).

Not fast at this stage: generation here is single-threaded plain C++ without a cache (78.9 s against PyTorch's
18.0 s on the SmolLM2 prompt set). M3 (KV cache) and M4 (threads, SIMD) fix that — both reproduce these exact
results, just faster; see their own sections above.

## Loading and tokenization (M1)

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
| 80 unit tests (GoogleTest) | UTF-8 edge cases; JSON grammar, escapes, exact 64-bit integers, depth and size limits; every safetensors rejection rule, including shapes that overflow 64 bits; float conversion (F32, F16 incl. subnormals and infinity, BF16); config validation; weight-layout checks; known token ids from the reference; round trips; dropped bytes; every kernel against hand-worked or double-precision values (RoPE scores depend only on relative position); on a small random model of each architecture: causality, determinism, tracing, bad input, stop tokens, and the KV cache reproducing the no-cache baseline; sampling's filtering checked against a plain softmax; the chat template engine's control flow and whitespace handling; the thread pool's chunking (every index covered exactly once, any worker count) and matmul's threaded output bit-for-bit against a row-by-row reference | `tests/` |
| Forward-pass cross-check | Every layer within 1e-4 of PyTorch and greedy tokens identical, on two architectures | `tools/crosscheck_forward.py`, CI `forward` job |
| Tokenizer cross-check | Identical token ids to Hugging Face on 4.5 million inputs | `tools/crosscheck_tokenizer.py`, CI `crosscheck` job |
| Weight cross-check | Every tensor of two models read exactly as PyTorch reads it | `tools/crosscheck_weights.py`, CI `crosscheck` job |
| Chat template cross-check | Identical prompts to `transformers`' own Jinja compiler, 9 cases on two real templates | `tools/crosscheck_chat_template.py`, CI `crosscheck` job |
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
`-DLLMI_WARNINGS_AS_ERRORS=ON`, `-DLLMI_ENABLE_SIMD=OFF` (portable scalar `dot()` only — for a CPU without
AVX2; on by default, adding `-mavx2 -mfma` on x86_64, nothing needed on Apple Silicon). The default build
above has `LLMI_SANITIZE=OFF`; for speed measurements specifically, also pass `-DLLMI_BUILD_TESTS=OFF` to
skip fetching GoogleTest (docs/performance.md explains why sanitizers specifically must be off to measure
speed, not just avoided as the default).

## Use

```
tools/fetch_model.sh smollm2-135m-instruct models/smollm2-135m-instruct   # pinned revision, SHA-256 checked
build/llmi-inspect models/smollm2-135m-instruct                           # architecture, tensors, parameter count
build/llmi-tokenize models/smollm2-135m-instruct/tokenizer.json "Hello, world!"
build/llmi-generate models/smollm2-135m-instruct --prompt "The capital of France is" --max-new 16 --kv-cache
build/llmi-chat models/smollm2-135m-instruct --temperature 0.8 --top-k 40 --top-p 0.9 --seed 1
build/llmi-generate models/qwen2.5-0.5b-instruct --ids 785,6722,315,9625,374 --max-new 8
```

`llmi-generate --kv-cache` uses the KV cache (M3) instead of recomputing the whole sequence every step (M2);
both give the same tokens, just at different speeds. `llmi-chat` is the interactive chat CLI (M3); it needs
both a chat template and a supported tokenizer, so it works with `smollm2-135m-instruct` today.

Qwen2.5's tokenizer is not supported yet, so its prompt is given as token ids (`785,6722,315,9625,374` is
"The capital of France is") and the output is ids too: `12095,13,1084,374,279,7772,3283,304`, which the
reference tokenizer decodes as " Paris. It is the largest city in". Its chat template engine still works and
is cross-checked (`llmi-chat-template`); only tokenization and `llmi-chat` are blocked on it.

To run the cross-checks yourself: `pip install transformers==5.18.0 tokenizers==0.23.2 safetensors torch`, then
see the commands at the top of each file in `docs/evidence/`.

## Layout

```
include/llmi/   public headers: util/ (JSON, UTF-8, Result, thread pool), model/ (safetensors, config, kernels, transformer, sampling), tokenizer/, chat/ (template)
src/            implementation; src/tokenizer/unicode_tables.inc is generated from the Unicode database
apps/           llmi-inspect, llmi-tokenize, llmi-generate, llmi-chat, llmi-chat-template
tests/          unit tests
fuzz/           libFuzzer targets
tools/          model download, cross-checks, Unicode table generator and probe, fuzz seeds, matmul microbenchmark
testdata/       tokenizer.json/config.json/tokenizer_config.json for the three models (Apache 2.0)
docs/           roadmap, tokenizer, forward-pass, KV cache, sampling and chat-template notes, architecture
                decisions, evidence from real runs
```

## Data and licences

Model weights are never committed; `tools/fetch_model.sh` downloads them from Hugging Face at a pinned revision
and checks each file's SHA-256. Third-party material keeps its own licence; see [THIRD-PARTY.md](THIRD-PARTY.md).
Design decisions are recorded in [docs/adr](docs/adr).

## Licence

MIT. See [LICENSE](LICENSE).
