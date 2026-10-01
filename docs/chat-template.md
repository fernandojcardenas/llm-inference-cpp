# Chat templates: turning messages into the model's own prompt format

An instruction-tuned model expects its prompt formatted a specific way — which special tokens mark a turn,
whether there's a default system message, how multi-turn history is laid out. That format is a small Jinja2
program, shipped in the model's `tokenizer_config.json` as `chat_template`. `llmi::chat::ChatTemplate` reads
and runs it directly, rather than hard-coding ChatML (or any other format) into the engine.

## Why a template interpreter instead of a hard-coded format

SmolLM2-135M-Instruct's and Qwen2.5-0.5B-Instruct's templates both produce ChatML-style output
(`<|im_start|>role\ncontent<|im_end|>`), but they aren't the same program: SmolLM2's inserts its own default
system message only when the first message isn't already one; Qwen2.5's real template also branches on
whether tool definitions were supplied, building a `<tools>` system-prompt block and `<tool_call>` XML for
tool calls. Reading and running the actual template means getting these details right without having guessed
them from a couple of example outputs, and getting them right automatically for a third model with yet
another template.

## What's implemented, and what isn't

`ChatTemplate` is a small, purpose-built Jinja2 subset: `{{ }}` output, `for`/`if`/`elif`/`else`/`set`, `{%-
-%}` whitespace control, `a.b` and `a['b']` access, string concatenation, `==`/`!=`, `and`/`or`/`not`, and `x
is [not] defined`. It is not a general Jinja engine — no macros, no loops over arbitrary filters, no real
`tojson` (the one `| filter` these templates use, inside branches this engine never takes — see below).

This engine never calls tools, so `tools` and every message's `tool_calls` are always undefined, exactly as
when a reference caller invokes `apply_chat_template()` without passing `tools`. Qwen2.5-0.5B-Instruct's real
template has a `{% if tools %}` branch that builds a `<tools>` system prompt and, per assistant message, a
`{% for tool_call in message.tool_calls %}` loop emitting `<tool_call>` XML — both are parsed (so the template
loads without error) but never execute, because both conditions are always false here. That's not a simplified
copy of the template; it's the actual file, behaving the way it does for any caller that doesn't pass tools.

## How it's checked

`tools/crosscheck_chat_template.py` renders the same message lists through this engine
(`llmi-chat-template`) and through `transformers.utils.chat_template_utils._compile_jinja_template` — the
same Jinja compiler `apply_chat_template()` itself calls, not a bare `jinja2.Environment()` with its own
defaults (checked and found to matter: see below) — and requires the output to be byte-identical.

| Case | What it covers |
|---|---|
| single user turn | the default system message each template inserts |
| explicit system prompt | the user's system message replacing the default, not both appearing |
| multi-turn history | several user/assistant turns in sequence |
| with/without `add_generation_prompt` | whether the trailing `<\|im_start\|>assistant\n` is appended |
| newlines, quotes, special-token-looking text, empty content, unicode | content that could break the string
  handling or be mistaken for a template construct |
| a 10-message history | more turns than either template was written assuming |

Run on both models' real `tokenizer_config.json`
([SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)): **9/9 cases identical to the reference, on
both** ([evidence](evidence/m3-chat-template-crosscheck.txt)).

### A reference-compiler detail that mattered

Jinja2's default `Environment()` drops exactly one trailing newline from the template *source* before parsing
(`keep_trailing_newline=False`) — not from the rendered output. Both real templates end with a newline after
their last tag, so without this, this engine's first renders had one extra trailing `\n` that the reference
didn't. `ChatTemplate::parse()` replicates it by stripping a single trailing `\n` (and a preceding `\r`, if
any) from the source before lexing. Confirmed against the actual compiler HF's tokenizer uses
(`transformers.utils.chat_template_utils._compile_jinja_template`), not assumed from the `jinja2` package's
documented defaults, since the two don't have to agree.

## Models this works with today

`ChatTemplate::load()` works on any tokenizer_config.json with a `chat_template` field. `llmi-chat`, the
interactive CLI, additionally needs a tokenizer this engine supports (docs/tokenizer.md): that's
SmolLM2-135M-Instruct today, not Qwen2.5-0.5B-Instruct, whose NFC-normalizing tokenizer isn't supported yet.
`llmi-chat-template` and the cross-check above exercise Qwen2.5's template engine directly, independent of
tokenization, so that gap doesn't limit what's checked — only what `llmi-chat` itself can run end to end.
