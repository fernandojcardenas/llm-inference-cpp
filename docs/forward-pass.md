# Forward pass: checked layer by layer

M2 turns token ids into next-token scores. This page describes what the engine computes, how it is checked
against PyTorch, and why checking the generated text alone would not have been enough.

## What it computes

For a sequence of token ids, in float32 (`src/model/transformer.cpp`, kernels in `src/model/kernels.cpp`):

1. **Embedding:** each id selects a row of the embedding matrix.
2. **Each layer**, twice adding to the running state ("residual stream"):
   - *Attention:* RMSNorm, then query/key/value projections (plus a bias in Qwen2), rotary position embedding
     on queries and keys, causal attention (position *t* sees positions 0…*t*) with grouped-query heads (SmolLM2:
     9 query heads share 3 key/value heads; Qwen2.5: 14 share 2), then the output projection.
   - *MLP:* RMSNorm, then `down(silu(gate(x)) * up(x))` (SwiGLU).
3. **Output:** a final RMSNorm, then the output matrix (tied to the embedding matrix in both models) gives one
   score per vocabulary entry.

Greedy generation appends the highest-scoring token and repeats. M2 has no key/value cache, so every step
recomputes the whole sequence; that is M3.

## How it's checked

`tools/crosscheck_forward.py` loads the same model in Hugging Face transformers (float32, eager attention) and
captures, through hooks, the embedding output, every layer's output, the final norm and the logits. For 8
prompts (prose, Python, history, science, a chat-formatted question, French, a radio question and a number
sequence) it compares each of those with the engine's (`llmi-generate --dump`), then generates 24 tokens
greedily on both sides and compares them. The tolerance is 1e-4 of the largest value in each state, fixed
before the first run ([ADR 0003](adr/0003-correct-float32-first-then-fast.md)).

| Model | Worst state difference | Worst logits difference | Greedy tokens identical | Evidence |
|---|---|---|---|---|
| SmolLM2-135M (Llama, 30 layers) | 5.4e-06 | 5.6e-06 | **192 of 192** | [output](evidence/m2-forward-smollm2-135m.txt) |
| Qwen2.5-0.5B-Instruct (Qwen2, 24 layers, q/k/v bias) | 2.7e-05 | 1.1e-05 | **192 of 192** | [output](evidence/m2-forward-qwen2.5-0.5b.txt) |

Differences of 1e-6 to 1e-5 are float32 rounding from adding in a different order. Qwen's are larger because
some of its later layers carry a few very large activations, a known trait of the Qwen family; relative to
those, the error is still far inside the tolerance. The closest greedy decision in the runs had the reference's
top two tokens 0.007 apart (SmolLM2, French prompt), and the engine still chose the same one.

## Would a bug get through?

To test the check itself, four realistic bugs were planted one at a time and the cross-check run on two
prompts ([output](evidence/m2-planted-bugs.txt)):

| Planted bug | Worst state difference | Greedy tokens identical |
|---|---|---|
| RoPE pairs taken as (2i, 2i+1) instead of (i, i + d/2) | 0.19 and 0.59 | 3 of 8, 0 of 8 |
| Query head *h* reads key/value head *h* mod 3 instead of *h* / 3 | 9.9 and 17 | 0 of 8, 0 of 8 |
| Attention can see one position into the future | 2.7 and 14 | **8 of 8, 8 of 8** |
| RoPE base 10,000 instead of 100,000 | 0.023 and 0.092 | 3 of 8, **8 of 8** |

Every one fails the layer check, the smallest by more than 200 times the tolerance. Two of them would have
passed a check on generated text alone: a mask that leaks the next position changes every position except the
last, and greedy decoding only reads the last. Fluent, even identical, output is not proof of a correct
model, which is why every layer is compared.

## Speed

Not optimised yet: on the 8-prompt SmolLM2 run, generation took 78.9 s in the engine against 18.0 s in PyTorch
(both recomputing the full sequence each step; PyTorch uses optimised multithreaded matrix code). The KV cache
(M3) and threads and SIMD (M4) address this, and must reproduce these results exactly.
