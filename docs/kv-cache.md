# KV cache: not recomputing the past

M2 generated one token at a time by calling `forward()` on the whole sequence so far — correct, but each step
redoes work the previous step already did. M3's `KVCache` + `Transformer::forward_cached()` keep every
layer's past keys and values and only compute the new tokens.

## How it works

`forward()` computes, for every layer, queries/keys/values for every position, then has each position attend
to all earlier positions. `forward_cached()` does the same thing but only for the *new* positions: it reads
`cache.length()` to find out how many positions are already known, computes queries/keys/values only for the
tokens just given to it, appends the new keys/values into the cache (which already holds every earlier
position's), and attends each new position over the whole cache. Nothing from an earlier call is
recomputed — the cache is exactly the K/V that `forward()` would have produced for the same tokens, filled in
incrementally instead of all at once.

A cache only grows (`reset()` is the only way to shrink it back to empty); it is sized once, for the longest
sequence it will hold (`Transformer::new_cache(max_len)`), capped at the model's context length.

## How it's checked

Unlike the tokenizer or the forward pass, there's no independent reference implementation to run a KV cache
through — what makes a cache correct is that it reproduces what `forward()` already computes, just
incrementally. `tests/transformer_test.cpp` (`KVCache.*`) checks exactly that, on both architectures:

| Test | What it checks |
|---|---|
| `PrefillInOneCallMatchesForward` | one `forward_cached()` call over a whole sequence gives bit-identical logits to `forward()` |
| `OneTokenAtATimeMatchesForward` | feeding the cache one new token per call matches `forward()` recomputing the whole growing sequence every time |
| `PrefillThenStepsMatchesForward` | the realistic pattern — prefill a prompt, then decode one token at a time |
| `GenerateGreedyCachedMatchesGenerateGreedy` | `generate_greedy_cached()` produces the same tokens as `generate_greedy()`, including where generation stops |
| `RejectsOverfullOrUnknownCache` | a full cache is refused rather than overrunning its buffer, and `reset()` makes room again |

Every check compares individual float values with `==`, not a tolerance: the cache either does the same
floating-point operations in the same order as `forward()` or it doesn't, and it does.

## Reusing a cache across a conversation

`llmi-chat` keeps one cache for the whole conversation. Each turn it re-renders the full message history
through the chat template and re-tokenizes it, then finds the longest prefix that already matches what's
cached (`cached_tokens`, kept alongside the cache) and only feeds the new suffix to `forward_cached()`. In the
ordinary case that's just the newest turn's tokens — a 38-token first turn's cache is reused in full on the
second turn, which only computes its own 18 new tokens
([evidence: the common-prefix check in docs/evidence/m3-chat-demo.txt and the cache-reuse numbers in the
roadmap]). If retokenizing the whole history ever disagrees with the cached prefix — possible in principle if
a BPE merge spans the boundary between old and new text — `llmi-chat` falls back to resetting the cache and
reprefilling everything; still correct, just not faster that turn.

## Speed

Measured with `llmi-generate --kv-cache` against the no-cache baseline, same machine, same build, back to
back ([evidence](evidence/m3-speed.txt)):

| Model | No cache | With cache | Speed-up |
|---|---|---|---|
| SmolLM2-135M (19 prompt + 24 new tokens) | 254.4 s | 17.1 s | 14.8x |
| Qwen2.5-0.5B-Instruct (5 prompt + 16 new tokens) | 247.6 s | 33.4 s | 7.4x |

Generated tokens are identical with and without the cache in both runs. The no-cache baseline recomputes the
whole prefix at every step (quadratic total work in the sequence length); the cache makes each step linear in
the sequence length, so the speed-up grows with how long the conversation gets — these short examples
understate it. Absolute seconds aren't comparable to the M2 evidence files, which were captured on different
hardware; only the within-run ratio is the point. Still single-threaded, unoptimised float32 (M4 is threads
and SIMD).
