#!/usr/bin/env python3
"""Measures the cost constrained decoding adds to generation (M8, part 2).

Runs llmi-generate on a chat model for the same JSON schemas as
tools/eval_constrained_json.py, in three modes:
  - unconstrained, greedy (baseline forward+sample cost per token);
  - constrained by --json-schema, greedy (the greedy shortcut skips the full
    vocabulary mask whenever the model's own top token is already allowed);
  - constrained by --json-schema, sampled at T=0.8 (every token pays a full
    mask, because top-k and top-p cut-offs depend on which tokens survive it).

The engine reports, per run, the total generate time and the time spent in the
mask (`mask_seconds`), plus how many picks needed a full vocabulary check
(`full_masks`). This script aggregates those over all runs and prints where a
token's time goes and what fraction the mask adds. It measures, it does not
optimize: see docs/constrained-decoding.md for the findings.

Usage: bench_decode_overhead.py BUILD_DIR MODEL_DIR [--seeds N]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SYSTEM = "You are a helpful assistant that answers in JSON."


def obj(props, **extra):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False, **extra}


CASES = [
    ("person", obj({"name": {"type": "string", "maxLength": 40}, "age": {"type": "integer"},
                    "city": {"type": "string", "maxLength": 40}}),
     ["Extract the person as JSON: Maria Lopez is 34 and lives in Detroit.",
      "Give JSON for this person: Ken Ito, 61 years old, from Ann Arbor."]),
    ("maintenance", obj({"vehicle": {"type": "string", "maxLength": 30}, "mileage": {"type": "integer"},
                         "status": {"enum": ["ok", "service_due", "out_of_service"]},
                         "parts": {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 3}}),
     ["Record as JSON: truck 12 has 84000 miles, needs brakes and an oil change, service is due.",
      "Record as JSON: the van at 15200 miles is fine, nothing to replace."]),
    ("yes-no", obj({"answer": {"type": "boolean"}, "confidence": {"type": "number"}}),
     ["Is the sky green? Answer as JSON with a confidence between 0 and 1.",
      "Is water wet? Answer as JSON with a confidence between 0 and 1."]),
    ("shopping", obj({"items": {"type": "array", "minItems": 1, "maxItems": 4,
                                "items": obj({"name": {"type": "string", "maxLength": 20},
                                              "qty": {"type": "integer"}})}}),
     ["Make a JSON shopping list for pancakes.", "List three things to buy for a picnic as JSON."]),
    ("optional", obj({"title": {"type": "string", "maxLength": 40},
                      "year": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}),
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


class Agg:
    def __init__(self):
        self.tokens = 0
        self.gen = 0.0
        self.mask = 0.0
        self.full = 0
        self.runs = 0

    def add(self, tokens, gen, mask=0.0, full=0):
        self.tokens += tokens
        self.gen += gen
        self.mask += mask
        self.full += full
        self.runs += 1

    def report(self, label):
        if self.tokens == 0:
            print(f"  {label}: no tokens")
            return
        gen_ms = 1000.0 * self.gen / self.tokens
        mask_ms = 1000.0 * self.mask / self.tokens
        model_ms = gen_ms - mask_ms  # forward + sampling, everything but the mask
        over = (100.0 * self.mask / (self.gen - self.mask)) if (self.gen - self.mask) > 0 else 0.0
        shortcut = 100.0 * (1.0 - self.full / self.tokens) if self.tokens else 0.0
        print(f"  {label}")
        print(f"      {self.runs} runs, {self.tokens} tokens; {gen_ms:6.1f} ms/token total "
              f"= {model_ms:6.1f} forward+sample + {mask_ms:6.2f} mask")
        print(f"      full masks: {self.full} of {self.tokens} picks "
              f"({shortcut:.0f}% skipped by the greedy shortcut); mask adds {over:.1f}% over forward+sample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("model")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    build, model = Path(a.build), Path(a.model)

    base, cg, cs = Agg(), Agg(), Agg()
    with tempfile.TemporaryDirectory() as tmp:
        for name, schema, prompts in CASES:
            sf = Path(tmp) / f"{name}.json"
            sf.write_text(json.dumps(schema))
            for p in prompts:
                b = run(build, model, p)
                base.add(len(b["generated_ids"]), b["generate_seconds"])
                g = run(build, model, p, sf)
                cg.add(g["generated_tokens"], g["generate_seconds"], g["mask_seconds"], g["full_masks"])
                for s in range(1, a.seeds + 1):
                    r = run(build, model, p, sf, 0.8, s)
                    cs.add(r["generated_tokens"], r["generate_seconds"], r["mask_seconds"], r["full_masks"])

    print(f"model: {model.name}; schemas: {len(CASES)}; prompts per schema: 2; sampled seeds: {a.seeds}")
    print("one mask per generated token; times are per generated token\n")
    base.report("unconstrained, greedy (baseline)")
    cg.report("constrained, greedy")
    cs.report("constrained, sampled T=0.8")


if __name__ == "__main__":
    sys.exit(main())
