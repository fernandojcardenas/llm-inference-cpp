#!/usr/bin/env python3
"""Checks constrained JSON output against an independent validator (M8).

For each schema and prompt below, runs llmi-generate on a chat model:
  - unconstrained, greedy (the baseline: does the model produce valid JSON
    on its own?);
  - constrained by --json-schema, greedy;
  - constrained by --json-schema, sampled at temperature 0.8 with several
    seeds (every token then needs a full vocabulary check).
Every output is parsed with Python's json module and validated with the
`jsonschema` package -- a separate implementation from the engine's own
schema-to-grammar converter, so a converter bug that let invalid output
through would show up here.

Usage: eval_constrained_json.py BUILD_DIR MODEL_DIR [--seeds N]
Prints a table and exits 1 if any constrained output is invalid.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jsonschema

SYSTEM = "You are a helpful assistant that answers in JSON."


def obj(props, **extra):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False, **extra}


CASES = [
    ("person",
     obj({"name": {"type": "string", "maxLength": 40}, "age": {"type": "integer"},
          "city": {"type": "string", "maxLength": 40}}),
     ["Extract the person as JSON: Maria Lopez is 34 and lives in Detroit.",
      "Give JSON for this person: Ken Ito, 61 years old, from Ann Arbor."]),
    ("maintenance",
     obj({"vehicle": {"type": "string", "maxLength": 30}, "mileage": {"type": "integer"},
          "status": {"enum": ["ok", "service_due", "out_of_service"]},
          "parts": {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 3}}),
     ["Record as JSON: truck 12 has 84000 miles, needs brakes and an oil change, service is due.",
      "Record as JSON: the van at 15200 miles is fine, nothing to replace."]),
    ("yes-no",
     obj({"answer": {"type": "boolean"}, "confidence": {"type": "number"}}),
     ["Is the sky green? Answer as JSON with a confidence between 0 and 1.",
      "Is water wet? Answer as JSON with a confidence between 0 and 1."]),
    ("shopping list",
     obj({"items": {"type": "array", "minItems": 1, "maxItems": 4,
                    "items": obj({"name": {"type": "string", "maxLength": 20}, "qty": {"type": "integer"}})}}),
     ["Make a JSON shopping list for pancakes.",
      "List three things to buy for a picnic as JSON."]),
    ("optional field",
     obj({"title": {"type": "string", "maxLength": 40}, "year": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}),
     ["Give JSON for the book Moby-Dick with its year.",
      "Give JSON for a song you like; use null if you don't know the year."]),
]


def prompt(text):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def run(build, model, text, schema_file=None, temperature=None, seed=0):
    cmd = [str(build / "llmi-generate"), str(model), "--prompt", prompt(text), "--max-new", "120", "--json"]
    if schema_file is None:
        cmd += ["--kv-cache"]
    else:
        cmd += ["--json-schema", str(schema_file)]
        if temperature is not None:
            cmd += ["--temperature", str(temperature), "--seed", str(seed)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode not in (0, 3):
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


def verdict(text, schema):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return "not JSON"
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError:
        return "JSON, wrong shape"
    return "valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("model")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    build, model = Path(a.build), Path(a.model)
    tok_path = model / "tokenizer.json"
    tok_json = json.loads(tok_path.read_text())

    rows = []
    bad = 0
    mask_s = 0.0
    full = 0
    nodes = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, schema, prompts in CASES:
            sf = Path(tmp) / f"{name}.json"
            sf.write_text(json.dumps(schema))
            for p in prompts:
                base = run(build, model, p)
                ids = base["generated_ids"]
                # Decode the baseline's ids the way the tokenizer does: GPT-2 byte-level
                # pieces back to bytes.
                text = decode(tok_json, ids)
                rows.append((name, "unconstrained, greedy", verdict(text.strip(), schema), text))
                runs = [("constrained, greedy", run(build, model, p, sf))]
                for s in range(1, a.seeds + 1):
                    runs.append((f"constrained, T=0.8 seed {s}", run(build, model, p, sf, 0.8, s)))
                for label, r in runs:
                    v = verdict(r["text"], schema) if r["complete"] else "incomplete (token budget)"
                    if v != "valid":
                        bad += 1
                    if "T=0.8" in label:
                        mask_s += r["mask_seconds"]
                        full += r["full_masks"]
                        nodes += r["trie_nodes_visited"]
                    rows.append((name, label, v, r["text"]))

    by = {}
    for name, label, v, _ in rows:
        kind = label.split(" seed")[0]
        by.setdefault(kind, {}).setdefault(v, 0)
        by[kind][v] += 1
    print(f"model: {model.name}; schemas: {len(CASES)}; prompts per schema: 2; sampled seeds: {a.seeds}")
    for kind, counts in by.items():
        total = sum(counts.values())
        parts = ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))
        print(f"  {kind:28s} {total:3d} outputs: {parts}")
    if full:
        print(f"  sampled runs: {full} full vocabulary checks, {1000 * mask_s / full:.2f} ms each on average, "
              f"{nodes / full:.0f} trie nodes visited each on average")
    print("\nexamples (first prompt of each schema):")
    seen = set()
    for name, label, v, text in rows:
        if (name, label.split(" seed")[0]) in seen or label.endswith("seed 2") or label.endswith("seed 3"):
            continue
        seen.add((name, label.split(" seed")[0]))
        print(f"  [{name} / {label} / {v}] {json.dumps(text)[:160]}")
    print("\nALL CONSTRAINED OUTPUTS VALID" if bad == 0 else f"\n{bad} CONSTRAINED OUTPUTS INVALID")
    return 0 if bad == 0 else 1


def decode(tok_json, ids):
    """Byte-level BPE decode, independent of the engine's tokenizer."""
    vocab = {v: k for k, v in tok_json["model"]["vocab"].items()}
    for added in tok_json.get("added_tokens", []):
        vocab[added["id"]] = "" if added.get("special") else added["content"]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_of = {chr(c): b for b, c in zip(bs, cs)}
    data = bytearray()
    for i in ids:
        piece = vocab.get(i, "")
        for ch in piece:
            if ch in byte_of:
                data.append(byte_of[ch])
    return data.decode("utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())
