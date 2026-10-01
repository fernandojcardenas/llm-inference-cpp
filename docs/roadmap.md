# Roadmap

Seven milestones, each finished only when its results are checked against a reference and reproduced in CI.
Target dates assume about eight hours a week.

| Milestone | What "done" means | Target | Status |
|---|---|---|---|
| **M1 Loading and tokenization** | safetensors and config.json loaded and validated, hardened and fuzzed; byte-level BPE tokenizer with token ids identical to Hugging Face | Oct 2026 | ✅ Done |
| **M2 Forward pass** | RMSNorm, rotary embeddings, grouped-query attention and SwiGLU in plain C++; every layer's output within tolerance of PyTorch; greedy generation identical token for token on a prompt set, in CI | Nov 2026 | ✅ Done (Oct 2026) |
| **M3 KV cache, sampling, chat** | Key/value cache with a measured speed-up; temperature, top-k and top-p sampling, reproducible from a seed; chat templates and a chat CLI | Dec 2026 | ✅ Done (Oct 2026) |
| **M4 Performance** | Thread pool, NEON (Apple Silicon) and AVX2 kernels, profiling; prompt and generation speed compared with llama.cpp on the same machine and model | Jan–Feb 2027 | ✅ Done (Oct 2026) |
| **M5 Quantization** | 8-bit and 4-bit block formats; quality loss measured by perplexity against the full-precision model and llama.cpp; speed and memory table | Mar 2027 | Planned |
| **M6 Hardened GGUF loader** | Read GGUF v3 models with every size and offset checked, fuzzed, with a regression set of malformed files and a threat model for loading untrusted model files | Apr 2027 | Planned |
| **M7 Server** | OpenAI-compatible chat completions API with streaming, request limits, localhost by default, Docker image, load test | May–Jun 2027 | Planned |

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
