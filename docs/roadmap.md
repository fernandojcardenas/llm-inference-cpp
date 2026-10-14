# Roadmap

Eight milestones; the first seven are done. Each is finished only when its results are checked against a reference
and reproduced in CI.
Target dates assume about eight hours a week.

| Milestone | What "done" means | Target | Status |
|---|---|---|---|
| **M1 Loading and tokenization** | safetensors and config.json loaded and validated, hardened and fuzzed; byte-level BPE tokenizer with token ids identical to Hugging Face | Oct 2026 | ✅ Done |
| **M2 Forward pass** | RMSNorm, rotary embeddings, grouped-query attention and SwiGLU in plain C++; every layer's output within tolerance of PyTorch; greedy generation identical token for token on a prompt set, in CI | Nov 2026 | ✅ Done (Oct 2026) |
| **M3 KV cache, sampling, chat** | Key/value cache with a measured speed-up; temperature, top-k and top-p sampling, reproducible from a seed; chat templates and a chat CLI | Dec 2026 | ✅ Done (Oct 2026) |
| **M4 Performance** | Thread pool, NEON (Apple Silicon) and AVX2 kernels, profiling; prompt and generation speed compared with llama.cpp on the same machine and model | Jan–Feb 2027 | ✅ Done (Oct 2026) |
| **M5 Quantization** | 8-bit and 4-bit block formats; quality loss measured by perplexity against the full-precision model and llama.cpp; speed and memory table | Mar 2027 | ✅ Done (Oct 2026) |
| **M6 Hardened GGUF loader** | Read GGUF v3 models with every size and offset checked, fuzzed, with a regression set of malformed files and a threat model for loading untrusted model files | Apr 2027 | ✅ Done (Oct 2026) |
| **M7 Server** | OpenAI-compatible chat completions API with streaming, request limits, localhost by default, Docker image, load test | May–Jun 2027 | ✅ Done (Oct 2026) |
| **M8 Constrained decoding and performance study** | Generation constrained to a grammar or JSON schema, with tests that every constrained output parses; a measured performance study on Apple Silicon: tokenizer throughput, Q8_0/Q4_0 against higher precision, a memory-bandwidth roofline and KV-cache layout | Apr–May 2027 | ✅ Done (Oct 2026): constrained decoding, its performance study, the masking optimization, and the Apple-Silicon study (tokenizer throughput, memory-bandwidth roofline, quantization, KV-cache layout) |

Also planned: Qwen2.5's tokenizer (NFC normalization and its split pattern), so Qwen models run end to end;
sharded safetensors (`model.safetensors.index.json`) for larger models.

## M1 results

- Tokenizer: [4,498,842 of 4,498,842 inputs](evidence/m1-tokenizer-crosscheck.txt) give the same token ids as
  Hugging Face, including every Unicode code point. Two findings on the way:
  [the reference uses two Unicode versions, and drops bytes it has no token for](tokenizer.md).
- Weights: [every tensor of SmolLM2-135M and Qwen2.5-0.5B](evidence/m1-weights-crosscheck.txt) is read exactly
  as PyTorch reads it.
- Fuzzing: four targets, clean ([output](evidence/m1-fuzz.txt)).

## M2 results

- Every layer of SmolLM2-135M and Qwen2.5-0.5B within 2.7e-05 of PyTorch; greedy generation identical on
  384 of 384 tokens ([forward-pass.md](forward-pass.md)).
- Four planted bugs all caught by the layer check; two of them would have passed a text-only check.

## M3 results

- KV cache: reproduces the no-cache baseline exactly (`tests/transformer_test.cpp` `KVCache.*`), 14.8x
  (SmolLM2) and 7.4x (Qwen2.5) faster on the same short runs ([kv-cache.md](kv-cache.md)).
- Sampling: temperature, top-k, top-p checked against a plain softmax and against each other on their own,
  with no model involved; same seed reproduces the same draws ([sampling.md](sampling.md)).
- Chat template: a Jinja2 subset covering what Hugging Face's own templates use, checked against
  `transformers`' own Jinja compiler — 9/9 cases identical on both SmolLM2-135M-Instruct and
  Qwen2.5-0.5B-Instruct's real templates ([chat-template.md](chat-template.md)).
- `llmi-chat`: an interactive multi-turn chat CLI, reusing the KV cache across turns by diffing retokenized
  history against what's cached.

## M4 results

- Profiled before optimizing: matmul is ~50% of per-token decode time; Qwen2.5's 151,936-word vocabulary
  makes its final `lm_head` projection the single largest matmul in the model ([evidence](evidence/m4-profile.txt)).
- Thread pool (`llmi::util::ThreadPool`) parallelizes matmul's independent output rows, bit-for-bit identical
  to the serial version; AVX2+FMA (x86_64) / NEON (Apple Silicon) `dot()`, chosen at compile time. 2.05x
  (SmolLM2) and 2.39x (Qwen2.5) faster generation on this sandbox's 2 cores ([evidence](evidence/m4-speed.txt)).
- Compared with llama.cpp, same machine, same weights, f32 (no quantization): within 1.5–1.6x at generation
  (same matmul shape in both engines); 2.5–2.7x slower at prompt processing, an honest gap from llama.cpp's
  row-blocked GEMM, which this engine doesn't have ([evidence](evidence/m4-llamacpp-comparison.txt)).

## M5 results

- Q8_0 (8-bit) and Q4_0 (4-bit) block quantization, matching GGUF's own formats exactly so a weight this
  engine quantizes is directly comparable to llama.cpp quantizing the same source. Found and fixed a real
  bug in the first Q4_0 implementation (a scale-sign convention that clamped the block's single most
  important value) by comparing against llama.cpp's own Q4_0 on identical weights
  ([evidence](evidence/m5-perplexity.txt)).
- Quality on real text (wikitext-2, scored the same way llama.cpp's own perplexity tool does): Q8_0 loses
  under 0.2% perplexity on both models; Q4_0 loses 36.3% (SmolLM2-135M) and 10.3% (Qwen2.5-0.5B) — smaller
  models are more sensitive to 4-bit quantization, the same pattern llama.cpp's own numbers show.
- The honest headline: quantizing with this engine trades memory for *more* time, not less. Q8_0/Q4_0 use
  3.56x/6.40x less memory but run 1.9–3.4x **slower**, because the quantized matmul has no SIMD path yet —
  the opposite of llama.cpp, whose Q8_0/Q4_0 are 4–8x **faster** than its own f32 thanks to hand-written
  SIMD kernels that operate directly on packed weights ([evidence](evidence/m5-speed-and-memory.txt)).

## M6 results

- A from-scratch, hardened GGUF v3 reader (`src/model/gguf.cpp`) — every count, length, offset and size
  the file claims is checked against both configurable limits and the bytes actually remaining before it's
  trusted, with overflow-checked arithmetic throughout, matching the standard ADR 0001 set for M1's
  safetensors reader. Every ggml tensor type (40+ codes) is structurally validated; F32/F16/BF16 convert
  to plain floats and Q8_0/Q4_0 convert to this engine's own `quant::QuantizedMatrix` (M5).
- Found and fixed a real layout bug: GGUF's Q4_0 packs each block's two *halves* into shared bytes, not
  adjacent element pairs the way this engine's own Q4_0 does — caught by cross-checking against real GGUF
  files from llama.cpp's own converter, the same way M5's scale-sign bug was caught, not by inspection
  alone ([ADR 0007](adr/0007-hardened-gguf-loader.md)).
- Checked twice against reality: a hand-crafted unit test with manually computed expected values, and a
  full cross-check against six real GGUF files (two models, three precisions each) read two independent
  ways — structure against the reference `gguf` Python package, values against a from-scratch NumPy
  reimplementation of ggml's block layout. 272/272/272/290/290/290 tensors, **0 mismatches**
  ([evidence](evidence/m6-crosscheck.txt)).
- 20 unit tests covering specific malformed-input categories (truncation at every header stage, overflow,
  nested arrays, bad alignment, overlapping tensors, duplicate keys/names, and more — the regression set
  this milestone calls for), plus a 60-second libFuzzer run with zero crashes and 4090 edges of coverage
  ([evidence](evidence/m6-fuzz.txt)). Full threat model: [docs/gguf-threat-model.md](gguf-threat-model.md).
- Scoped to the loader itself, not wired into `Transformer`/`llmi-generate`/`llmi-chat` yet — see
  [docs/gguf.md](gguf.md) and ADR 0007 for why.

## M7 results

- An OpenAI-compatible `/v1/chat/completions` (streaming via SSE and non-streaming), `/v1/models`
  and `/health` server (`llmi-server`), transported by cpp-httplib (vendored, MIT — this engine's
  first third-party dependency) while this engine's own hardened JSON parser still parses every
  request body, the same trust boundary ADR 0001 drew for every other untrusted input
  ([ADR 0008](adr/0008-server-transport-and-single-generation-slot.md); [server.md](server.md)).
- A real architectural constraint, found by reading `llmi::util::ThreadPool`'s own documented
  contract rather than assumed: `kernels::matmul`'s thread pool cannot be called from more than
  one thread at a time, so every request's actual generation is serialized behind one mutex.
  Confirmed by load-testing a running server with real concurrent HTTP clients: mean latency grew
  3.26x at 4-way concurrency while an 8-way run hit the server's concurrency limit and returned
  429 for exactly the excess requests, not an approximation of it
  ([evidence](evidence/m7-loadtest.txt)).
- Checked against a real reference server: llama.cpp's own `llama-server`, same GGUF file
  (`smollm2-135m-instruct-f32.gguf`, from M6's own cross-check tooling) both ways. Every field
  this engine implements matches llama-server's exactly in name, type and nesting, including an
  exact prompt-token count match through the same chat template; one intentional simplification
  (no role-only priming chunk before the first streamed content delta) is documented, not hidden
  ([evidence](evidence/m7-llamacpp-comparison.txt)).
- A Docker image building just the Release binary, no weights baked in. Building it is not where
  this milestone's real finding was: binding `127.0.0.1` *inside* the container makes a
  `-p 127.0.0.1:PORT:PORT`-published port connect and immediately reset, never serve, because
  Docker's port-publishing reaches a container through its bridge interface, never its loopback —
  caught by actually running the built image against a real model, not assumed from reading
  Docker's own documentation (ADR 0008's Consequences; [server.md](server.md)'s Docker section).

## M8 results (constrained decoding)

- Matcher vs an independent Earley recognizer written in Python: 76,670 strings over 9 grammars (including a
  converted JSON Schema), 0 disagreements ([evidence](evidence/m8-grammar-crosscheck.txt)). Two bugs planted
  on purpose give 95 to 1,012 disagreements each ([evidence](evidence/m8-planted-bugs.txt)).
- SmolLM2-135M-Instruct, 5 schemas x 2 prompts: unconstrained greedy 0 of 10 schema-valid (9 not JSON);
  constrained greedy 10 of 10 and sampled at T=0.8 50 of 50, judged by `jsonschema` 4.26.0
  ([evidence](evidence/m8-constrained-json.txt)).
- Full vocabulary check: 8.49 ms per token on average (20,218 of 100,298 trie nodes visited), against about
  39 ms for the forward pass, on the 2-core test machine.
- A real bug found by the server test: an end-of-text id that also decodes to bytes was allowed as text but
  refused as end-of-text. Fixed ([details](constrained-decoding.md)).
- Details: [constrained-decoding.md](constrained-decoding.md), [ADR 0009](adr/0009-constrained-decoding.md).

## M8 results (performance study, part 2)

- End to end on SmolLM2-135M-Instruct over the five schemas, against a ~60 ms/token forward baseline:
  constrained greedy adds **0.8%** (the greedy shortcut skips ~89% of masks); constrained sampling adds **19%**
  (a full mask every token) ([evidence](evidence/m8-decode-overhead.txt)).
- The mask cost is bimodal: ~0.04 ms at structural positions, but 50-62 ms inside a free-form string where the
  walk visits ~99,500 of 100,298 trie nodes. The cause is ~12 heap allocations per trie node (the matcher
  rebuilds its State per node): a heavy mask churns ~1.2 M allocations / ~70 MB for one token
  ([evidence](evidence/m8-mask-profile.txt), model-free `tools/bench_constrained.cpp`).
- Details and the full tables: [constrained-decoding.md](constrained-decoding.md#performance-study-m8-part-2).

## M8 results (masking optimization, part 3)

- `allowed_tokens()` now memoizes transitions (each distinct matcher state interned, each `(state, byte)`
  computed once) instead of rebuilding the state at every trie node. `allowed_tokens_direct()` keeps the
  original walk as the fallback and test oracle; `step`/`advance`/`normalize` are unchanged.
- Before → after on the same machine: ~12.5 → ~0.25 heap allocations per trie node; a heavy free-form-string
  mask 50-63 ms → 6-9 ms; constrained **sampling overhead 19% → 2.5%** over forward+sampling
  ([evidence](evidence/m8-mask-optimization.txt)).
- Masks are bit-identical to the old walk: a unit test and an independent driver over the real vocabulary
  (under ASan+UBSan) agree 0-for-0, and the real-model constrained-JSON eval still reports every output valid.
- Details: [constrained-decoding.md](constrained-decoding.md#the-optimization-m8-part-3).

## M8 results (Apple-Silicon performance study)

- Where single-token decoding spends time, measured on Apple Silicon (arm64 NEON) and x86 (AVX2) with no model
  weights (`tools/bench_apple_silicon.cpp`, random data at real shapes). Details:
  [performance-apple-silicon.md](performance-apple-silicon.md); evidence
  [m8-apple-silicon.txt](evidence/m8-apple-silicon.txt).
- Decode is **memory-bound** (0.5 FLOP/byte): on Apple Silicon the cache-busting lm_head matmul runs at the
  machine's full ~76 GB/s and 37.8 GFLOP/s (= 76 × 0.5). The ~76-vs-~10 GB/s bandwidth gap to the x86 sandbox
  is most of the speed difference.
- **Quantization** saves 3.6×/6.4× bytes but is slower than f32 (no SIMD dequant yet) on Apple Silicon too —
  the M5/ADR 0006 gap, now measured on arm64. A vectorized dequant is the top compute optimization.
- **KV cache layout**: a head-major layout runs ~25–35% faster at long context than the old position-major
  one (ctx 4096: 16.2 → 12.0 ms/token of attention). **Built**: the engine's KV cache is now head-major,
  bit-identical (KV-cache tests + token-identity; [m8-kv-head-major.txt](evidence/m8-kv-head-major.txt)).
- Tokenizer ≈ 8 M tokens/s encode, 100 M/s decode — not a bottleneck.
