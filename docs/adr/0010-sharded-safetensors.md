# ADR 0010: Sharded safetensors, loaded through a hardened index

Status: accepted

## Context

The roadmap lists sharded safetensors (`model.safetensors.index.json`) as planned work, so that
models too large for a single file can run end to end. Hugging Face splits a large checkpoint
into several `model-0000k-of-0000N.safetensors` files and writes an index file naming which
shard holds each tensor:

```
{ "metadata": { "total_size": ... },
  "weight_map": { "<tensor name>": "<shard filename>", ... } }
```

Until now `Model::load()` read exactly one `model.safetensors`. Everything downstream -- the
`Model` tensor check and `Transformer::load()` -- already looks tensors up by name
(`weights.find(name)` / `weights.data(info)`), never by position in a particular file, which is
what makes multi-file weights tractable without touching the forward pass.

The index is another untrusted model file, so ADR 0001's standard applies to it unchanged:
every claim it makes is checked before it is trusted, errors are `Result<T>` values, and it is
fuzzed.

## Decisions

### A `Weights` view over one or more shards, so nothing downstream changes

Rather than teach `Transformer` about shards, the weights became a small `Weights` type
(`include/llmi/model/weights.hpp`) that owns one or more `SafetensorsFile`s and offers the same
by-name surface a single file already did: `find(name)`, `data(info)`, `tensors()`. `Model` now
holds a `Weights`; a single-file model is just a `Weights` with one shard. `to_f32()` in
`transformer.cpp` changed its parameter type from `const SafetensorsFile&` to `const Weights&`
and nothing else moved. Each shard is still parsed and validated on its own by `SafetensorsFile`
(ADR 0001); `Weights` adds the one invariant a single file cannot violate -- a tensor name must
not appear in two shards.

### The shard filename in the index is validated as a plain filename before any file is opened

The index's values are filenames that get appended to the model directory and opened. That is a
path-traversal sink: an index carrying `"../../etc/something.safetensors"` or an absolute path
would otherwise make the loader open a file outside the model directory. `is_safe_shard_filename`
rejects anything that is not a plain filename in that one directory -- no `/` or `\`, not `.` or
`..`, no leading `.` (hidden / traversal) or `-` (so a name can never be read as a command-line
option), characters restricted to `[A-Za-z0-9._-]`, and a required `.safetensors` suffix. This
runs during parsing, so a bad name is rejected before a single shard is opened, and the fuzzer
asserts every accepted index has only safe filenames.

### The index and the shards must describe exactly the same tensors

An index and the shard files on disk can disagree -- a truncated download, a hand-edited index,
a malicious one. `Model::load()` cross-checks the two before building the model: every tensor the
`weight_map` names must actually be present in the shard it points to, and every tensor physically
present in a shard must be attributed to that same shard by the index. Either mismatch is an
error, not a silent merge, so a model only loads when its index and its bytes agree completely.

### A single `model.safetensors` still wins, and is never second-guessed

`Model::load()` loads `model.safetensors` when it is present and only falls back to the index
when it is not. A corrupt single file is a real error, never a reason to go looking for an index;
a directory with neither is a clear error naming both files it looked for.

### Checked against a reference: the same model, split, must load identically

The whole project's test discipline is "checked against a reference" (roadmap). The reference for
sharded loading is the *same model loaded from a single file*: `ShardedModel.LoadsIdenticallyToASingleFile`
writes a tiny model once as one file and once split across two shards, and asserts the loaded
weights are byte-identical and a full forward pass produces bit-identical logits. The failure
paths (index/shard disagreement, a missing shard, an empty directory) are tested too, and the
new index parser has its own hardening unit tests and a libFuzzer target
(`fuzz/fuzz_safetensors_index.cpp`).

## Consequences

- A sharded Llama/Qwen2 model whose tensors match what `expected_tensors()` already requires
  loads and runs with no other change; shards are memory-mapped like any single file, so the
  weights are still views, never copies.
- `Model::weights()` now returns a `Weights`, not a `SafetensorsFile`. The two callers
  (`transformer.cpp`, `llmi-inspect`) use only `find`/`data`/`tensors`, which `Weights` provides;
  `Model::from_parts(ModelConfig, SafetensorsFile)` is kept as a thin wrapper so the existing
  single-file tests and fuzzers are unchanged.
- The index parser does not interpret `metadata` beyond requiring it to be an object if present;
  `total_size` and similar fields are not needed to load the model and are ignored on purpose.
