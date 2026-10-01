# ADR 0003: A float32 forward pass that matches the reference first, speed later

**Status:** accepted (M2)

## Context

The forward pass is where an inference engine is most likely to be subtly wrong: a rotary embedding applied
to the wrong pair of values, a missing bias, an off-by-one in the causal mask or a grouped-query head mapped to
the wrong key/value head still produce fluent-looking text. Plausible output is not evidence of correctness.

## Decision

- M2 implements the transformer in plain float32 C++ with no optimisation beyond a dot product the compiler can
  vectorise. Weights are converted to float32 at load (BF16 and F16 convert exactly).
- Every operation follows the reference's definition where they could differ: rotate-half RoPE with the
  frequencies computed in float32, RMSNorm as `weight * (x / sqrt(mean(x²) + eps))`, softmax over the causal
  window, ties in greedy decoding resolved to the lowest token id (as `torch.argmax`).
- `tools/crosscheck_forward.py` compares, for every prompt, the token embeddings, the output of every layer,
  the final norm and all logits against Hugging Face transformers (float32, eager attention), and requires
  greedy generation to be identical token for token. It runs on two architectures (Llama: SmolLM2;
  Qwen2: Qwen2.5 with q/k/v biases) in CI.
- The tolerance, 1e-4 relative to the largest reference value of each state, was set before the first run:
  float32 results that differ only in summation order typically agree to about 1e-6, so 1e-4 leaves room for
  rounding. Afterwards, four planted bugs (RoPE pairing, grouped-query head mapping, a causal mask leaking one
  position, the wrong RoPE base) all showed differences of 2e-2 or more ([forward-pass.md](../forward-pass.md)).
- Generated text is compared too, but it is not the main check: two of the planted bugs still produced the
  reference's greedy tokens.
- No KV cache: each generated token recomputes the whole sequence. The cache is M3, and it has to reproduce
  these outputs exactly; this version is the baseline it is checked against.

## Consequences

- Generation is slow (a few tokens per second for the 135M model), which is expected until M3 and M4.
- M3 (cache), M4 (threads, SIMD) and M5 (quantization) are each checked against this baseline as well as
  against the reference, so a speed-up can't silently change results.
