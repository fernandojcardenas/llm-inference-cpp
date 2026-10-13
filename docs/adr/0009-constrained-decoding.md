# ADR 0009: Constrained decoding with a GBNF subset, a pushdown matcher and a token trie

Status: accepted (M8, part 1)

## Context

M7's server can answer a chat request, but nothing makes the answer follow a format. Programs that call a
model usually need machine-readable output (JSON with known fields), and small models are bad at producing it
on their own: on SmolLM2-135M-Instruct, 9 of 10 "answer as JSON" prompts did not even produce JSON
([evidence](../evidence/m8-constrained-json.txt)). Constrained decoding fixes this at the source, by never
letting the model pick a token that would break the format.

Four decisions had to be made: the grammar format, the matching algorithm, how the grammar meets sampling,
and what to do with JSON Schema features that are hard to express.

## Decisions

### A subset of llama.cpp's GBNF, not a new format

GBNF is what people writing grammars for local models already know, and grammars written for llama.cpp work
here unchanged as long as they stay inside the subset. The subset covers everything JSON needs (rules,
literals, character classes, groups, `* + ? {m,n}`). The parser rewrites repetitions and groups into plain
rules at parse time, so the matcher handles exactly one shape.

### A pushdown matcher over stacks, the same design as llama.cpp's

The state is a set of stacks of grammar positions. This handles any context-free grammar without left
recursion, works one character at a time (so it can answer "is this a viable prefix?", which is what masking
needs), and is simple enough to check against an independent implementation. Left recursion is rejected at
parse time rather than handled, because it would make expansion loop; every grammar a JSON Schema produces is
right-recursive anyway.

Considered and not chosen:

- **Compiling to a DFA.** Not possible in general: JSON nests, and nesting needs a stack.
- **Earley parsing in the engine.** It handles every grammar, but its per-step cost is higher and harder to
  bound. It is used instead as the independent oracle in `tools/crosscheck_grammar.py`, where a different
  algorithm is the point.

### Grammars and sizes are untrusted

A grammar can come from any HTTP client, so the parser bounds input size, rules, symbols, repetition counts and
nesting, and the matcher bounds stacks, stack depth and expansions per character. Every limit fails closed: a
grammar or input over a limit is rejected, never accepted unchecked. The same rule as ADR 0001 for model
files: untrusted input gets a parser written to refuse it safely, and a fuzz target from the first commit
(`fuzz_grammar`, wired into CI in the same change; the lesson from M6's fuzzer that was never wired in).

### Mask before sampling; bytes, not characters; a trie over the vocabulary

- **Order.** Disallowed tokens get minus infinity before temperature, top-k and top-p, so those filters
  operate on allowed tokens only. The alternative, sampling first and rejecting, would make top-k and top-p
  cut-offs depend on tokens that could never be chosen.
- **Greedy shortcut.** If the unconstrained argmax is allowed, it is also the masked argmax, so the full mask
  is skipped. Exact, not an approximation. No equivalent shortcut is exact for sampling, so sampling always
  masks.
- **Bytes.** The matcher consumes bytes and tracks a half-finished UTF-8 character, accepting it only if some
  allowed character can begin with those bytes. Tokens in byte-level BPE vocabularies routinely end
  mid-character.
- **Trie.** All token byte strings go into one prefix tree, so one rejected byte prunes every token that
  starts with it.
- **End-of-text.** An end-of-text id is allowed exactly when the grammar is complete and is never treated as
  text, even in a vocabulary where the same id has bytes. The server integration test found this case.

### JSON Schema: OpenAI's strict subset, and errors for everything else

Strict mode (every property required, `additionalProperties: false`) makes the output fully determined by the
schema and is what OpenAI's own API requires for guaranteed structure. Keywords outside the supported set
(`pattern`, `format`, `minimum`, `$ref`, ...) are errors, not ignored. A converter that ignored `minimum`
would hand back output that violates the schema the caller gave it, while claiming to enforce it. Whitespace
and number lengths are bounded so a model cannot loop forever on spaces or digits.

## Consequences

- Output structure is guaranteed; content is not. A constrained model can still put the wrong year in a
  valid field.
- Sampling with a grammar costs a full vocabulary check per token; greedy decoding is nearly free because the
  shortcut skips that check whenever the model's own top token is already allowed. The M8 part-2 performance
  study ([docs/constrained-decoding.md](../constrained-decoding.md#performance-study-m8-part-2)) measured this:
  about 0.8% added over the forward pass for greedy, about 19% for sampling, with the mask's cost dominated by
  free-string positions and, underneath, about 12 heap allocations per trie node. Part 3 then memoized the
  masker's transitions (intern each distinct state, compute each `(state, byte)` once) rather than rebuilding
  the state per node: ~12.5 → ~0.25 allocations per node and sampling overhead 19% → 2.5%, with the masks
  byte-identical to the original walk (kept as `allowed_tokens_direct()`, the fallback and test oracle).
- Qwen2.5 cannot be constrained until its tokenizer is supported.
- New code is concentrated in `src/grammar/` (parser, matcher, schema converter, constraint), with small hooks
  in the server request parser, the server loop and `llmi-generate`.
