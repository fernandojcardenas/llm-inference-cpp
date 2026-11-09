# Sharded safetensors: `model.safetensors.index.json`

M1-M8 read a model from a single `model.safetensors` + `config.json`. Models too large for one
file ship their weights split across several shards, with an index file naming which shard holds
each tensor. This engine now reads that layout: if a model directory has no `model.safetensors`
but does have a `model.safetensors.index.json`, `Model::load()` reads the index, opens the shards
it names, and assembles the weights into the same by-name view a single file produces
([ADR 0010](adr/0010-sharded-safetensors.md)). The forward pass is unchanged -- it never knew how
many files the weights came from.

## The format

```
model/
  config.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
```

```json
{
  "metadata": { "total_size": 1234567 },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00002-of-00002.safetensors",
    "model.norm.weight": "model-00002-of-00002.safetensors"
  }
}
```

`weight_map` maps every tensor name to the shard file that holds it. Each shard is itself an
ordinary safetensors file, parsed and validated on its own (ADR 0001). `metadata` is optional and
not interpreted beyond being required to be an object if present -- `total_size` is not needed to
load the model.

## How it is read (`include/llmi/model/safetensors_index.hpp`)

`SafetensorsIndex::parse()` reads the index with the same strict, bounded JSON parser the
safetensors header and `config.json` use, then applies the rules below. `Model::load()` opens each
shard named by `shards()`, wraps them in a `Weights` (the by-name view over one or more shards),
and cross-checks the index against the shards before building the model.

### The index is untrusted, and its filenames are a path-traversal sink

The values in `weight_map` are filenames the loader appends to the model directory and opens.
`is_safe_shard_filename()` rejects anything that is not a plain filename in that one directory,
*before any file is opened*:

- no path separators (`/`, `\`) and no `.`/`..` -- the name cannot escape the model directory;
- no leading `.` (hidden files, `..`) or leading `-` (so a name can never be read as an option);
- characters restricted to `[A-Za-z0-9._-]`;
- a required `.safetensors` suffix.

So an index carrying `"../../etc/passwd"` or `"/abs/path.safetensors"` is rejected at parse time,
not acted on.

### The index and the shards must agree exactly

A download can be truncated or an index hand-edited. Before building the model, `Model::load()`
checks that the index and the bytes on disk describe the same tensors:

- every tensor the index lists is actually present in the shard it names;
- every tensor a shard actually holds is attributed to that shard by the index;
- no tensor name appears in two shards.

Any mismatch is a `Result` error naming the offending tensor, never a silent merge.

### Bounds

`SafetensorsIndexLimits` caps the index file size, the number of `weight_map` entries, the number
of distinct shards, and the tensor-name and filename lengths. The defaults are generous for real
checkpoints (a few hundred shards) and low enough that nothing can overflow.

## Checked against a reference

The test for sharded loading is that it is indistinguishable from single-file loading. `tests/
sharded_model_test.cpp` builds a tiny model, writes it once as a single `model.safetensors` and
once split across two shards with an index, loads both, and asserts:

- byte-identical weights for every tensor, and
- bit-identical logits from a full forward pass.

The failure paths (an index that disagrees with its shards, a missing shard file, a directory
with neither a single file nor an index) are tested too, and the index parser has hardening unit
tests (`tests/safetensors_index_test.cpp`) and a libFuzzer target
(`fuzz/fuzz_safetensors_index.cpp`). Local run: [evidence](evidence/sharded-safetensors.txt).

## Limitations

- The tensors a sharded model contains must still match what `expected_tensors()` requires for a
  supported architecture (Llama / Qwen2); sharding changes only how the weights are stored on
  disk, not which weights the engine knows.
- The index's `metadata` is not interpreted. `total_size` is not checked against the shards'
  actual size (the exact per-tensor cross-check above is stronger than a byte total).
