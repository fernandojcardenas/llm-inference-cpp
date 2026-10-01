# Sampling: temperature, top-k, top-p

Greedy decoding (M2) always picks the highest-scoring token, so the same prompt always gives the same reply.
M3 adds `llmi::sample()`: temperature, then top-k, then top-p (nucleus) filtering — the same order Hugging
Face's default `do_sample=True` path applies them in — followed by a random draw.

## What's checked, and what isn't

The filtering is checked on its own, with no randomness involved. `sampling_probabilities()` turns logits
into a probability distribution and is tested directly (`tests/sampling_test.cpp`):

- it matches a plain double-precision softmax when nothing is filtered;
- temperature above 1 flattens the distribution, below 1 sharpens it, and ≤ 0 collapses it onto the argmax
  (this engine's convention for "sample greedily" — see below);
- top-k keeps exactly the *k* highest-probability entries and renormalises over them;
- top-p keeps the smallest prefix (by descending probability) whose cumulative probability reaches `top_p`,
  while always keeping at least one token even if the very first one already exceeds it;
- draws stay inside whatever set survived the filtering, and the same seed reproduces the same sequence of
  draws.

What isn't checked, and can't honestly be claimed: that a seed reproduces any particular reference
implementation's random draws. `sample()` uses the engine's own `std::mt19937_64`; `transformers`' sampling
uses PyTorch's own generator, a different algorithm entirely, so the same seed does not pick the same token
from the same distribution in both. What the engine does claim, and does check, is that it samples from
*the same distribution* — the probabilities a reference caller would see from the same logits and the same
temperature/top-k/top-p are the ones in `sampling_probabilities()`'s output. The random draw on top of that
is this engine's own, documented as such rather than quietly implied to match.

## Temperature ≤ 0 means greedy

Hugging Face's sampling path doesn't accept `temperature=0` — there's nothing to cross-check there. This
engine defines it anyway, as a convenience so one `--temperature` flag covers both modes, and documents it as
its own choice (ADR 0004): `temperature <= 0` skips the softmax and returns the same token
`generate_greedy()`/`argmax()` would, which `tests/sampling_test.cpp`'s `TemperatureZeroIsGreedy` checks
directly (including picking the *first* maximum on a tie, matching `torch.argmax`).

## Using it

`llmi-chat --temperature 0` (the default) is greedy and reproducible by construction. `--temperature 0.8
--top-k 40 --top-p 0.9 --seed N` samples, and the same seed replays the same conversation
([evidence](evidence/m3-chat-demo.txt) shows both a greedy run and a sampled one).
