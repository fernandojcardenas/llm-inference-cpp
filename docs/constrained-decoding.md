# Constrained decoding (M8)

The engine can force a model's output to follow a grammar: any grammar written in a subset of llama.cpp's
GBNF format, or a JSON Schema converted to one. Tokens the grammar does not allow are removed before sampling,
so the output is well-formed by construction rather than by luck. Design decisions are in
[ADR 0009](adr/0009-constrained-decoding.md).

On SmolLM2-135M-Instruct, 0 of 10 unconstrained answers to "answer as JSON" prompts were valid against their
schema (9 were not JSON at all); 60 of 60 constrained answers were, checked by Python's `jsonschema` package
rather than by this engine ([evidence](evidence/m8-constrained-json.txt)). The constraint fixes the *shape* of
the answer, not its truth: the same run has the model give Moby-Dick's year as 1843 (it is 1851), in perfectly
valid JSON.

## Using it

```
build/llmi-generate models/smollm2-135m-instruct --prompt "$PROMPT" --json-schema person.json
build/llmi-generate models/smollm2-135m-instruct --prompt "$PROMPT" --grammar answer.gbnf --temperature 0.8 --seed 1
build/llmi-grammar --schema person.json          # print the grammar a schema becomes
build/llmi-grammar --grammar answer.gbnf          # check a grammar; --jsonl tests strings against it
```

`llmi-server` takes OpenAI's `response_format` field: `{"type": "json_object"}` for any JSON object, or
`{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}` ([server notes](server.md),
[a real request and response](evidence/m8-server-response-format.txt)). A schema outside the supported subset
is a `400` that names the keyword, not a silently weaker constraint.

## How it works

**Grammar** (`include/llmi/grammar/grammar.hpp`). The parser reads rules, string literals, character classes
(including negated classes and `.`), groups, alternatives and the repetitions `* + ? {m} {m,} {m,n}`. It
rewrites every repetition and group into plain rules as it parses, so the rest of the engine only ever sees
sequences of character sets and rule references. Grammars arrive over HTTP, so this is an untrusted-input
parser like the model loaders: input size, rule count, symbol count, repetition counts and group nesting are
all bounded, and it rejects the two things the matcher could not run safely, left recursion (a rule that can
reach itself without reading a character would expand forever) and undefined rules.

**Matcher** (`include/llmi/grammar/matcher.hpp`). The same design as llama.cpp's grammar engine: a pushdown
automaton whose state is a set of stacks of grammar positions, one per way the text so far can still be parsed.
Each stack's top is a character set waiting to be matched; an empty stack means the text so far is complete.
Stacks are deduplicated after every character, and the matcher fails closed (rejects) if their number, their
depth or the work per character passes its limits.

Models emit bytes, not characters, and one token can end halfway through a multi-byte UTF-8 character
(SmolLM2 has tokens that are a single lead byte). The matcher therefore takes bytes and holds an incomplete
character back. It accepts the partial bytes only if some stack could match a character beginning with them,
working out the exact range of code points those bytes can still become.

**Choosing a token** (`include/llmi/grammar/constraint.hpp`). Every token's bytes are arranged in a prefix
tree (100,298 nodes for SmolLM2's 49,152 tokens, built in 0.03 s). Walking that tree with the matcher state
rejects a whole subtree of tokens at the first byte the grammar disallows. Disallowed tokens get a logit of
minus infinity before temperature, top-k and top-p are applied, so sampling is over the allowed tokens with the
model's own probabilities renormalized. Greedy decoding first checks the unconstrained top token: if the
grammar allows it, it is also the top allowed token and the full check is skipped. End-of-text is allowed
exactly when the grammar is complete, and generation stops by itself once the grammar allows nothing more.

**JSON Schema** (`include/llmi/grammar/json_schema.hpp`). Follows OpenAI's strict structured-outputs rules:
every property required, `additionalProperties: false`, properties in schema order. Supported: `type` (one or
a list), `properties`, `required`, `additionalProperties`, `items`, `minItems`, `maxItems`, `minLength`,
`maxLength`, `enum`, `const`, `anyOf`, plus annotations (`title`, `description`, ...). Any other keyword is an
error. Whitespace between tokens is allowed but capped at 20 characters per gap, so a model cannot spend its
whole token budget on spaces. Numbers are bounded too (at most 16 digits before and after the point).

## Checked three ways

1. **An independent recognizer.** `tools/crosscheck_grammar.py` re-implements the grammar format from scratch
   in Python with a different algorithm, an Earley parser, and rewrites repetitions the other way round (left
   recursion and explicit alternatives instead of right recursion). It enumerates strings exhaustively up to a
   length per grammar, plus random sentences and every one-character edit of each. Both implementations judge
   each string complete, viable prefix or rejected: 76,670 strings over 9 grammars, including a converted
   JSON Schema, with 0 disagreements ([evidence](evidence/m8-grammar-crosscheck.txt)). Two bugs planted on
   purpose (one copy too few in `{m,n}`, and a rule reference that drops what follows it) produce 95 to 1,012
   disagreements each, so the check does catch them ([evidence](evidence/m8-planted-bugs.txt)).
2. **A real model, judged by someone else's validator.** `tools/eval_constrained_json.py` runs five schemas
   (person, maintenance record with an enum and an array, yes/no with a confidence, a list of objects, a
   nullable field) with two prompts each, greedy and sampled at temperature 0.8 over five seeds, and validates
   every output with `jsonschema` 4.26.0 ([evidence](evidence/m8-constrained-json.txt)).
3. **Unit, integration and fuzz tests.** Grammar parsing and its limits, matching (including split UTF-8
   characters, rejection leaving state unchanged, and fail-closed limits), the token mask against checking each
   token on its own, every schema rule, and an end-to-end server test where a random-weight model, which writes
   garbage on its own, still returns schema-valid JSON, streamed and not. A new libFuzzer target
   (`fuzz/fuzz_grammar.cpp`) feeds arbitrary grammars, text and schemas and checks that the matcher's answers
   agree with each other. It runs in CI with the other six, and ran 55,119 inputs in 5 minutes here with no
   finding.

The server test found a real bug before it shipped. The test model's end-of-text token (id 10) is also an
ordinary byte, a newline. The mask allowed it as whitespace, but `accept()` treated it as end-of-text and
refused it, because the JSON was not finished yet. End-of-text ids now mean only end-of-text, allowed exactly
when the grammar is complete, even when the same id also decodes to bytes.

## Cost

Measured on this project's 2-core x86 sandbox (Release build, SmolLM2-135M-Instruct, F32 weights):

- Sampled runs check the whole vocabulary at every step: 2,165 checks took 8.49 ms each on average, visiting
  20,218 of 100,298 trie nodes each. A forward pass on the same machine takes about 39 ms per token, so
  sampling with a grammar adds roughly a fifth to generation time.
- Greedy runs mostly skip the check: in the first example below, 1 of 27 tokens needed it.
- The worst case is inside a JSON string, where nearly every token is allowed and the walk visits most of the
  tree. The performance study below measures this per grammar position and finds what drives it.

```
$ build-perf/llmi-generate models/smollm2-135m-instruct --prompt "$P" --max-new 80 --json-schema person.json
{
  "name": "Maria Lopez",
  "age": 34,
  "city": "Detroit"
}
[37 prompt tokens, 27 generated in 1.10 s; grammar complete; deciding allowed tokens took 0.040 s (1 of 27 picks needed a full vocabulary check); vocabulary trie 100298 nodes built in 0.03 s; load 0.77 s]
```

(`$P` is the SmolLM2 chat format, with system prompt "You are a helpful assistant.", around "Extract the person as JSON: Maria Lopez is 34 and lives in Detroit."; `person.json` requires `name` and `city` strings of at most 40 characters and an integer `age`.)

## Performance study (M8, part 2)

The study asks two things: how much does constraining cost in practice, and where does that cost go. Two
measurements answer them, both on the 2-core x86 sandbox at a clean Release build. Absolute times are specific
to that machine and to the mix of outputs each run happens to generate; the ratios and the greedy-shortcut rate
are the portable results.

**How much it costs, end to end** ([evidence](evidence/m8-decode-overhead.txt), `tools/bench_decode_overhead.py`
over the five schemas, using the engine's own `generate_seconds` / `mask_seconds` / `full_masks` to split each
token's time). Against an unconstrained baseline of about 60 ms per token:

| Mode | mask ms/token | full masks | the mask adds |
|---|---|---|---|
| Constrained, greedy | 0.5 | 47 of 419 picks | **0.8%** |
| Constrained, sampled T=0.8 | 12.5 | 1270 of 1270 picks | **19%** |

Greedy is nearly free because the greedy shortcut skips the full vocabulary mask whenever the model's own top
token is already allowed -- about 89% of the time here. Sampling cannot skip it (top-k and top-p must operate on
the allowed set), so it pays a full mask every token. Either way the forward pass, not the grammar, sets the
pace.

**Where the mask's time goes** ([evidence](evidence/m8-mask-profile.txt), `tools/bench_constrained.cpp`, no
model: it drives each grammar along a valid token path on the real vocabulary and times `allowed_tokens()` at
every step). The per-token mask cost is sharply bimodal:

- At structural positions (`{`, `"`, `:`, `,`, digits, enum words) only a handful of tokens fit, so the walk
  visits tens of trie nodes and costs about 0.04 ms -- the median mask.
- Inside a free-form string almost every token is allowed, so the walk visits ~99,500 of the 100,298 trie
  nodes and costs 50-62 ms. A JSON object has only a few such positions, but they dominate its total mask time.

The cause is allocation, not grammar logic: the mask does **about 12 heap allocations per trie node** (the
matcher rebuilds a full State -- a set of grammar-position stacks -- at every node of the walk), so a heavy
free-string mask churns roughly 1.2 million allocations and 70 MB, allocated and freed, for one token.

That points straight at the next optimization -- carry the State down the trie and reuse its buffers instead of
reallocating per node -- which would cut the heavy case most. It is left as the next step rather than rushed in
here: the mask's output must stay bit-identical (it is what the independent Earley cross-check guarantees), so
it deserves its own change, the way M4 deferred a row-blocked GEMM ([docs/performance.md](performance.md)) and
M5 deferred a quantized SIMD kernel ([ADR 0006](adr/0006-block-quantization-matches-gguf-no-simd-yet.md)).

## What's not here

- Qwen2.5 cannot be constrained yet, because its tokenizer is still unsupported (see the README).
- From GBNF: llama.cpp's newer token references (`<[...]>`) and its `%llguidance` mode are not in the subset.
- From JSON Schema: `pattern`, `format`, numeric bounds (`minimum`, `multipleOf`, ...), `$ref`/`$defs`, and
  optional properties. Each is rejected with an error naming it.
- String length counts JSON characters, so a surrogate-pair escape counts as two where JSON Schema counts one
  code point.
- The trie walk reallocates the matcher State at every node rather than reusing it down the tree. The
  performance study above shows this is what makes a free-string mask expensive (~12 allocations per node); the
  fix is the next optimization, kept separate because the mask output must stay bit-identical.
