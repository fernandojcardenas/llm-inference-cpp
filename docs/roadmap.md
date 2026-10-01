# Roadmap

Seven milestones, each finished only when its results are checked against a reference and reproduced in CI.
Target dates assume about eight hours a week.

| Milestone | What "done" means | Target | Status |
|---|---|---|---|
| **M1 Loading and tokenization** | safetensors and config.json loaded and validated, hardened and fuzzed; byte-level BPE tokenizer with token ids identical to Hugging Face | Oct 2026 | ✅ Done |
| **M2 Forward pass** | RMSNorm, rotary embeddings, grouped-query attention and SwiGLU in plain C++; every layer's output within tolerance of PyTorch; greedy generation identical token for token on a prompt set, in CI | Nov 2026 | Next |
| **M3 KV cache, sampling, chat** | Key/value cache with a measured speed-up; temperature, top-k and top-p sampling, reproducible from a seed; chat templates and a chat CLI | Dec 2026 | Planned |
| **M4 Performance** | Thread pool, NEON (Apple Silicon) and AVX2 kernels, profiling; prompt and generation speed compared with llama.cpp on the same machine and model | Jan–Feb 2027 | Planned |
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
