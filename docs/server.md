# The server (M7)

M7 adds `llmi-server`: an HTTP server exposing the engine's own
`Transformer`/`Tokenizer`/`ChatTemplate` (M1-M6) as an OpenAI-compatible chat
completions API. The transport is cpp-httplib, vendored as this engine's
first third-party dependency; the architectural decisions behind that
choice, and the concurrency model below, are ADR 0008
([docs/adr/0008-server-transport-and-single-generation-slot.md](adr/0008-server-transport-and-single-generation-slot.md)).

## Running it

```
llmi-server MODEL_DIR [--host HOST] [--port N] [--quant f32|q8_0|q4_0]
            [--max-concurrent N] [--max-tokens N] [--max-body-bytes N]
```

`MODEL_DIR` needs the same files `llmi-chat` requires: `config.json`,
`model.safetensors`, `tokenizer.json`, and a `tokenizer_config.json` with a
chat template. Binds `127.0.0.1` unless `--host` names something else, in
which case a startup warning is printed -- there is no authentication, so
exposing this beyond loopback is a choice the caller makes explicitly, not
a default.

## Endpoints

- `GET /health` -- `200 ok` once the server is accepting requests.
- `GET /v1/models` -- the single loaded model, in the OpenAI `list` shape.
- `POST /v1/chat/completions` -- the main endpoint; see below.

### Request fields

Only the fields this engine can act on are validated; anything else in the
JSON body (`n`, `logprobs`, `user`, ...) is accepted and ignored rather than
rejected, so a client written against the full OpenAI schema isn't broken
by fields this engine doesn't implement.

| field | type | default | notes |
|---|---|---|---|
| `messages` | array, required | -- | each `{role, content}`; `role` must be `system`, `user` or `assistant` |
| `model` | string | the server's configured name | echoed back, not used to select a model (one model per server process) |
| `max_tokens` | integer | 16 | capped at the server's `--max-tokens` ceiling regardless of what's requested (ADR 0008) |
| `temperature` | number, 0-2 | 1.0 | 0 means greedy (this engine's convention, not Hugging Face's -- see docs/sampling.md) |
| `top_p` | number, 0-1 | 1.0 | nucleus sampling |
| `top_k` | integer | 0 (off) | |
| `stream` | boolean | false | Server-Sent Events, see below |
| `stop` | string or array of strings | none | up to 4 sequences; matched against generated text and trimmed from the final non-streaming response |

### Non-streaming response

The OpenAI `chat.completion` shape: `id`, `object`, `created`, `model`,
`choices[0].message.{role,content}`, `choices[0].finish_reason` (`"stop"`
or `"length"` -- this engine never returns a content-filter or tool-call
finish reason, since it does neither), and `usage.{prompt_tokens,
completion_tokens,total_tokens}`.

### Streaming response

Server-Sent Events: one `chat.completion.chunk` JSON object per generated
token as a `data: {...}\n\n` line, a final chunk with an empty `delta` and
the real `finish_reason`, then the literal `data: [DONE]\n\n`. One
intentional simplification against the real OpenAI API and against
llama.cpp's own server (see
[docs/evidence/m7-llamacpp-comparison.txt](evidence/m7-llamacpp-comparison.txt)):
there is no separate role-only priming chunk before the first content
delta -- the first chunk already carries content. A client that only
appends `delta.content` as chunks arrive (the common case) sees no
difference; one that specifically reads `delta.role` off the first chunk
before any content would need this engine to add that chunk, which it
doesn't yet.

Decoding happens one token at a time, but a single token can be one byte of
a multi-byte UTF-8 character (`Tokenizer::decode` documents this directly).
Bytes are only ever emitted, in either response shape, once they form a
complete, valid UTF-8 prefix; anything left incomplete when generation ends
is dropped -- the same convention `Tokenizer::encode` already uses for
bytes it has no token for, applied here to avoid ever writing invalid UTF-8
into a JSON response body.

## Concurrency and request limits (ADR 0008)

HTTP accept, request parsing, and response writing are concurrent
(cpp-httplib's own thread pool). Every call into
`Transformer::forward_cached` is serialized behind one mutex, because
`kernels::matmul` calls `llmi::util::ThreadPool::shared()`, a process-wide
singleton that cannot be called from more than one thread at a time. In
practice this means: **two concurrent requests do not generate tokens in
parallel**, however many CPU cores are available -- the second one's actual
generation waits for the first's to finish. [docs/evidence/m7-loadtest.txt](evidence/m7-loadtest.txt)
measures this directly: latency grows with concurrency while throughput
does not.

Three independent limits, checked in this order, before any model code
runs for a request:

1. **Request body size** (`--max-body-bytes`, default 1 MiB) -- enforced by
   cpp-httplib itself from `Content-Length` before the body is read;
   exceeding it returns `413 Payload Too Large`.
2. **Concurrent request count** (`--max-concurrent`, default 4) -- a fixed
   admission count ahead of the one generation slot; a request beyond it is
   rejected immediately with `429 Too Many Requests` and a JSON error body,
   never queued without bound.
3. **Tokens per request** (`--max-tokens`, default 4096) -- a server-side
   ceiling independent of what any request's own `max_tokens` asks for,
   because under one generation slot, one request's length is every other
   request's wait.

## Docker

```
docker build -t llmi-server .
docker run --rm -p 127.0.0.1:8080:8080 -v /path/to/model:/model:ro llmi-server /model --host 0.0.0.0
```

The image builds the Release binaries and runs `llmi-server` as its
entrypoint; it does not bake in any model weights (this project never
commits weights -- see `tools/fetch_model.sh` and `THIRD-PARTY.md`), so a
model directory is mounted as a read-only volume at container run time.

**`--host 0.0.0.0` here is required, not a loosening of ADR 0008's
loopback-by-default stance** -- found by actually running the image, not
assumed. `llmi-server`'s own default (`127.0.0.1`) means loopback *inside
whatever network namespace it runs in*. Inside a container, that namespace
is the container's own, already isolated from the host and from everything
else by Docker; `docker run -p 127.0.0.1:8080:8080` publishes the
container's port only to the host's own loopback interface, but Docker's
port-publishing reaches a container through its bridge network interface,
never through the container's loopback. A server bound to `127.0.0.1`
*inside* the container is therefore unreachable through a published port
at all (confirmed directly: the host curl gets a reset connection, not
a slow success), regardless of the host-side address in `-p`. Binding
`0.0.0.0` inside the container is correct and not a security loosening:
the container's own network isolation is what makes it safe, and the `-p
127.0.0.1:8080:8080` on the host side is what actually controls exposure
to the rest of the host's network -- the same env-appropriate interpretation
of "loopback by default" as any other containerized server, just not the
same literal bind address as running the binary directly on a host.

## What's not here

No API key or any other authentication -- loopback-by-default is the
mitigation, not a substitute for one; running this reachable beyond
loopback without putting a reverse proxy or auth layer in front of it is a
choice the operator makes, not something this project recommends. No
multi-model routing (one model per process, matching `llmi-chat`'s own
scope). No prompt-prefix caching across requests (each request gets a
fresh `KVCache`, prefilled from scratch, unlike `llmi-chat`'s cross-turn
cache reuse within a single conversation) -- natural future work, not
required for an OpenAI-compatible completions API to be correct.
