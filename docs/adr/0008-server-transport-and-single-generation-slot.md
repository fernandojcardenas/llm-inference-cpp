# ADR 0008: cpp-httplib for transport, a single generation slot, request limits, localhost by default

Status: accepted (M7)

## Context

M7 is an OpenAI-compatible chat completions server: HTTP framing (connections, keep-alive,
chunked transfer encoding for streaming), request parsing, and routing on top of the engine
`Transformer`/`Tokenizer`/`ChatTemplate` that M1-M6 already built and proved correct. Two
questions have to be settled before any server code is written: what handles the HTTP
transport itself, and how concurrent requests interact with an engine that was never built
with concurrent callers in mind.

## Decisions

### Vendor cpp-httplib as a single pinned header, not a hand-rolled HTTP/1.1 implementation

Implementing HTTP/1.1 framing from scratch -- keep-alive, chunked transfer encoding (needed
for Server-Sent Events streaming), header parsing per RFC 7230, timeouts -- is disproportionate
scope for this milestone and is not where this project's own hardening work belongs. ADR 0001's
invariant is "this engine writes its own parsers for untrusted *model and tokenizer* input,
because that's the attack surface with public CVEs against this class of software
(llama.cpp's GGUF and GGUF-adjacent bugs)." HTTP/1.1 framing is a different, generic problem
with mature, widely-deployed solutions; writing a new one here would be scope creep dressed up
as consistency, not an application of the same principle.

cpp-httplib (https://github.com/yhirose/cpp-httplib, MIT) was chosen over alternatives because
it is a single header (`httplib.h`), has no further transitive dependencies, and is a server
*and* client in one file (useful later for `tools/loadtest_server.py`-style tooling and any
in-repo integration test that drives the server as a real HTTP client rather than through a
test-only shortcut). It is vendored at `third_party/cpp-httplib/httplib.h` (pinned to tag
`v0.18.3`, commit `a7bc00e`, SHA-256 `a0a0c13dc086663863dbe6730a19716f7d3744904b6205496e8301750814dbd5`)
rather than fetched at configure time the way GoogleTest is (`FetchContent`, test-only
infrastructure that never ships). A vendored, pinned file means a production build has zero
network dependency and an auditable, reviewable diff of exactly what code is in the binary --
appropriate because, unlike GoogleTest, this is this project's **first third-party dependency
that ships inside the engine itself**. `THIRD-PARTY.md` records it with the same
Material/Where/Licence/Source columns as every other entry.

This does not relax ADR 0001's actual boundary: cpp-httplib owns the wire-level HTTP framing
(where bytes end and headers begin), but the *content* of a request body is untrusted JSON
from the network, parsed by this engine's own hardened `llmi::json::parse` -- the same
depth/node-limited DOM parser M1 built for safetensors headers and config files, not by any
JSON facility cpp-httplib might offer. The trust boundary this project cares about (model and
request *content*) stays exactly where ADR 0001 drew it; only the transport layer around that
content is delegated.

### Generation is serialized behind a single slot; only the HTTP layer is concurrent

`Transformer::forward()`/`forward_cached()` are `const`, mutating only the caller-supplied
`KVCache&` -- in isolation this would let each in-flight HTTP request safely own its own
`KVCache` and call either method concurrently with another request's. It does not, because
both call into `kernels::matmul`, which calls `llmi::util::ThreadPool::shared()` -- a
process-wide singleton. `include/llmi/util/thread_pool.hpp`'s own doc comment is explicit:
`parallel_for` is the pool's only entry point, it blocks the calling thread until every chunk
finishes, and "calling it concurrently from two threads... is not supported." Reading
`src/util/thread_pool.cpp` confirms why: `parallel_for()` mutates shared members (`fn_`,
`ranges_`, `version_`, `pending_`) under one mutex with no per-caller isolation, so a second
concurrent call could overwrite `fn_`/`ranges_` or bump `version_` while workers are still
executing the first call's dispatch -- not a theoretical race, a direct corruption of in-flight
work. This is the same category of constraint M4 (ADR 0005) and M5 (ADR 0006) already
documented honestly about this engine rather than quietly working around: `ThreadPool` was
built and proven correct for M4's single-caller-at-a-time use (one `llmi-generate`/`llmi-chat`
process, one forward pass at a time), and a server is the first caller that could violate that
assumption.

The decision: the server holds one process-wide generation mutex. Any request that needs to
call `forward`/`forward_cached`/`generate_greedy*` acquires it first and holds it for the
duration of that call; a second request's generation work queues behind the first rather than
racing it. This is coarser than it has to be in principle -- a future `ThreadPool` redesign
supporting per-caller instances could let two requests truly generate in parallel on disjoint
thread subsets -- but matches what the engine actually guarantees today, documented as a real,
current limitation rather than an aspiration.

What *is* concurrent: request acceptance, parsing, validation, and response writing (including
streaming a token at a time once it's produced) all happen outside the generation mutex, so
multiple connections can be open, mid-request-parse, or mid-stream-write at once -- only the
actual `forward_cached` call inside each request's turn is serialized against every other
request's.

### Request limits: a bounded queue ahead of the generation slot, a body-size ceiling, a token ceiling

Because generation is serialized, an unbounded number of accepted-but-waiting requests would
let the server accept more work than it can ever keep up with, and a single very large request
could exhaust memory before generation even starts. Three independent limits apply, enforced in
this order for every request before any model code runs:

1. **Request body size.** Capped by a server-configured byte ceiling, checked from the
   `Content-Length` header (or capped incrementally if absent) before the body is fully read
   into memory; exceeding it returns `413 Payload Too Large` without parsing anything.
2. **Queue depth.** A fixed-capacity queue sits in front of the generation mutex; a request that
   would exceed the configured concurrency/queue-depth limit is rejected immediately with
   `429 Too Many Requests` rather than accepted and left waiting indefinitely -- an explicit,
   bounded backpressure policy instead of silent unbounded queuing.
3. **Tokens per request.** `max_tokens` in a request is validated against a server-configured
   ceiling (independent of whatever the client asks for) before generation starts, because an
   unbounded generation length under a serialized single slot would let one request starve
   every other connection for an arbitrarily long time.

All three are request-validation concerns, checked by pure functions over parsed input (no HTTP
or model object involved), consistent with this project's established "testable pure logic,
thin IO shell" pattern from M1-M6 -- the same shape as `llmi::json::parse`'s own limits
(`max_depth`, `max_nodes`) being struct fields checked at specific call sites, not ambient
globals.

### Localhost by default

The server binds to `127.0.0.1` unless a caller explicitly passes a different `--host`. An
OpenAI-compatible chat completions endpoint with no built-in authentication is not something
this project wants exposed to a network by accident; defaulting to loopback means the common
case (running the binary directly on a single development machine) is safe without the caller
having to know to ask for it, while an explicit `--host 0.0.0.0` (or any non-loopback address)
remains possible for whoever deliberately wants it, printed with a warning at startup rather
than silently allowed.

Running inside the Docker image is the one case that *needs* `--host 0.0.0.0`, found by
actually running the image rather than assumed: a container has its own network namespace, so
`127.0.0.1` there means loopback inside the container, and Docker's `-p 127.0.0.1:PORT:PORT`
port-publishing reaches a container through its bridge interface, never through its loopback --
binding `127.0.0.1` inside the container makes the published port connect and immediately reset
rather than serve anything. This isn't a loosening of the loopback-by-default stance: the
container's own isolation is what makes `0.0.0.0` safe there, and `-p 127.0.0.1:PORT:PORT` on
the host side is what actually controls exposure on the host's network, so the real guarantee
(unreachable from outside the host) holds either way, by a different mechanism than the plain
binary's. See docs/server.md's Docker section.

## Consequences

- Two requests submitted at the same time will have their actual token generation run one
  after the other, never truly in parallel, however many CPU cores are available -- a
  deliberate, documented throughput ceiling inherited from `ThreadPool::shared()`'s existing
  single-caller contract, not a bug to be fixed inside M7's scope. The load test (M7, pending)
  is expected to show latency growing with concurrent request count even though HTTP-level
  accept/parse/stream stays concurrent.
- cpp-httplib is now inside the shipped binary, the project's first non-test third-party
  dependency; any future transport-level CVE in cpp-httplib itself is this project's problem to
  track and re-vendor against, the same obligation GoogleTest's test-only `FetchContent` does
  not carry.
- A genuinely concurrent generation path (parallel requests actually overlapping on-CPU) would
  require either a `ThreadPool` redesign (per-caller pool instances or a work-stealing queue
  that tolerates concurrent submitters) or running multiple engine processes behind a
  load balancer -- both out of scope for M7, left as natural future work alongside K-quant
  support (ADR 0007) and a blocked GEMM (ADR 0005).
