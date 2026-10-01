# ADR 0004: A KV cache checked against the no-cache baseline, sampling as a separate step, and a scoped chat template engine

**Status:** accepted (M3)

## Context

M3 adds three things needed for an interactive chat CLI: not recomputing the whole sequence on every
generated token, more than greedy decoding, and turning a list of chat messages into the exact prompt string
the model was trained on.

## Decisions

### KV cache: checked against forward(), not just against PyTorch

- `KVCache` holds per-layer key/value history; `Transformer::forward_cached()` computes only the new
  positions and appends their K/V to it, instead of recomputing everything (`forward()`, kept as the M2
  baseline).
- The cache's correctness is defined as reproducing `forward()` exactly, token for token and value for value,
  not as a fresh cross-check against PyTorch: M2 already established that forward() matches the reference, so
  the cache only has to show it computes the same thing faster. `tests/transformer_test.cpp` (`KVCache.*`)
  checks this directly — one call prefilling everything, one token at a time, prefill then steps, and
  `generate_greedy_cached` against `generate_greedy` — rather than re-running the Python cross-check with a
  cache flag.
- A cache only grows by appending (no sliding window, no eviction) and exposes `reset()`; `llmi-chat` reuses
  it across turns by diffing newly-tokenized prompts against what's cached (see docs/kv-cache.md).

### Sampling is a separate, swappable step

- `sampling_probabilities()` turns logits into a probability distribution (temperature, then top-k, then
  top-p, matching Hugging Face's default warper order) and is tested on its own, with no model or random
  numbers involved. `sample()` draws from it with the engine's own `std::mt19937_64`.
- Temperature ≤ 0 is defined as greedy sampling (skip the softmax, return the argmax). This has no reference
  to check against — `transformers`' `do_sample=True` path doesn't accept temperature 0 — so it is documented
  here as this engine's own convention, not a reproduction of anything.
- A seed reproduces a run of *this engine*; it is not claimed to reproduce any particular reference
  implementation's random draws, because that depends on the reference's choice of PRNG algorithm as much as
  the seed. What is checked against a reference is the distribution sample() draws from (softmax then
  filtering), not the draw itself. See docs/sampling.md.

### A minimal Jinja2 subset, not a general template engine

- Chat templates are small Jinja2 programs stored in `tokenizer_config.json`. Rather than embed a general
  Jinja implementation, `llmi::chat::ChatTemplate` implements exactly the constructs Hugging Face's own chat
  templates use: `{{ }}` output, `for`/`if`/`elif`/`else`/`set`, `{%- -%}` whitespace control, attribute and
  index access, string concatenation, comparisons, `and`/`or`/`not`, and `is [not] defined`.
- This engine never calls tools, so `tools` and every message's `tool_calls` are always undefined/empty —
  exactly what a reference caller gets from `apply_chat_template()` without passing `tools`. The branches of
  Qwen2.5's template that build tool-call XML are syntactically parsed (so parsing doesn't fail) but never
  execute.
- Checked against `transformers.utils.chat_template_utils._compile_jinja_template` — the same compiler
  `apply_chat_template()` uses — not a bare `jinja2.Environment()`, because the two can disagree (found during
  development: Jinja2's default environment already matches `keep_trailing_newline=False`, but only testing
  against the actual HF entry point rules out every other default diverging the same way). See
  docs/chat-template.md.

## Consequences

- `llmi-chat` only works end to end with models that have both a chat template (an Instruct tokenizer_config)
  and a tokenizer this engine supports (byte-level BPE, no normalizer) — today that's
  `smollm2-135m-instruct`. Qwen2.5-0.5B-Instruct's chat template engine is still checked and used by
  `llmi-chat-template` and the cross-check, but its tokenizer isn't supported yet (docs/tokenizer.md), so
  `llmi-chat` can't tokenize its prompts.
- Adding a tool-calling-capable chat model later would need the engine to actually model `tool_calls`, not
  just parse past them; today's scope is text-only conversation.
