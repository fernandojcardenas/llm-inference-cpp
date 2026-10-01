#!/usr/bin/env python3
"""Cross-checks llmi's forward pass and greedy generation against Hugging Face
transformers (PyTorch, float32, eager attention) on the same weights.

For each prompt:
  1. one forward pass over the prompt: the token embeddings, the output of
     every layer, the final norm and all logits are compared value by value
     (max absolute difference, relative to the largest reference value);
  2. greedy generation of --new tokens with no stop token: both sides take
     the argmax of the last position's logits and append it. The tokens must
     be identical. If they diverge, the reference's margin between its top two
     logits at that step is printed, to tell a near-tie from a bug.

Prompt token ids come from the reference tokenizer, so models whose
tokenizer llmi does not support yet (Qwen2.5) are checked too.

Usage:
  python3 tools/crosscheck_forward.py MODEL_DIR build/llmi-generate [--new 24] [--tolerance 1e-4] [--limit N]
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, __version__ as transformers_version

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "In 1969, Apollo 11 landed on the Moon. The crew were",
    "Photosynthesis converts light energy into",
    "<|im_start|>user\nWhat is 12 times 7?<|im_end|>\n<|im_start|>assistant\n",
    "Les principales villes de France sont Paris, Lyon,",
    "A ship's radio operator sends a distress call using",
    "1, 1, 2, 3, 5, 8, 13,",
]


def reference_trace(model, ids):
    """Embeddings, every layer's output, the final norm and the logits."""
    captured = []

    def keep(_module, _inputs, output):
        captured.append((output[0] if isinstance(output, tuple) else output)[0].detach().float().numpy().copy())

    inner = model.model
    hooks = [inner.embed_tokens.register_forward_hook(keep)]
    hooks += [layer.register_forward_hook(keep) for layer in inner.layers]
    hooks.append(inner.norm.register_forward_hook(keep))
    with torch.no_grad():
        logits = model(torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
    for h in hooks:
        h.remove()
    return captured, logits


def reference_greedy(model, ids, new):
    ids = list(ids)
    out, margins = [], []
    with torch.no_grad():
        for _ in range(new):
            logits = model(torch.tensor([ids]), use_cache=False).logits[0, -1].float()
            top = torch.topk(logits, 2)
            margins.append(float(top.values[0] - top.values[1]))
            nxt = int(torch.argmax(logits))
            out.append(nxt)
            ids.append(nxt)
    return out, margins


def read_dump(path):
    with open(path, "rb") as f:
        header = json.loads(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4")
    t, h, v, n = header["tokens"], header["hidden"], header["vocab"], header["states"]
    states = [data[i * t * h:(i + 1) * t * h].reshape(t, h) for i in range(n)]
    logits = data[n * t * h:].reshape(t, v)
    return states, logits


def rel_diff(ours, ref):
    return float(np.max(np.abs(ours - ref)) / max(float(np.max(np.abs(ref))), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("binary")
    ap.add_argument("--new", type=int, default=24)
    ap.add_argument("--tolerance", type=float, default=1e-4, help="max relative difference allowed per state")
    ap.add_argument("--limit", type=int, default=len(PROMPTS), help="check only the first N prompts")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    cfg = model.config
    print(f"transformers {transformers_version}, PyTorch {torch.__version__}: {cfg.architectures[0]}, "
          f"{cfg.num_hidden_layers} layers, float32, eager attention")

    failures = 0
    worst_state = worst_logit = 0.0
    worst_where = ""
    total_new = same_new = 0
    t_ref = t_ours = 0.0
    prompts = PROMPTS[:args.limit]
    for prompt in prompts:
        ids = tok.encode(prompt).ids
        ref_states, ref_logits = reference_trace(model, ids)
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            subprocess.run([args.binary, args.model_dir, "--ids", ",".join(map(str, ids)), "--max-new", "0",
                            "--dump", f.name], check=True)
            states, logits = read_dump(f.name)
        if len(states) != len(ref_states):
            print(f"FAIL {prompt[:30]!r}: {len(states)} states vs {len(ref_states)} in the reference")
            failures += 1
            continue
        diffs = [rel_diff(s, r) for s, r in zip(states, ref_states)]
        i_worst = int(np.argmax(diffs))
        if diffs[i_worst] > worst_state:
            worst_state = diffs[i_worst]
            names = ["embeddings"] + [f"layer {k}" for k in range(1, len(states) - 1)] + ["final norm"]
            worst_where = names[i_worst]
        ld = rel_diff(logits, ref_logits)
        worst_logit = max(worst_logit, ld)
        same_argmax = int(np.sum(np.argmax(logits, 1) == np.argmax(ref_logits, 1)))

        t0 = time.perf_counter()
        ref_new, margins = reference_greedy(model, ids, args.new)
        t_ref += time.perf_counter() - t0
        t0 = time.perf_counter()
        run = subprocess.run([args.binary, args.model_dir, "--ids", ",".join(map(str, ids)), "--max-new", str(args.new),
                              "--ignore-eos", "--json"], check=True, capture_output=True, text=True)
        t_ours += time.perf_counter() - t0
        ours_new = json.loads(run.stdout)["generated_ids"]
        total_new += len(ref_new)
        n_same = next((k for k, (a, b) in enumerate(zip(ours_new, ref_new)) if a != b), len(ref_new))
        same_new += n_same

        ok = max(diffs) <= args.tolerance and ld <= args.tolerance and ours_new == ref_new and same_argmax == len(ids)
        failures += not ok
        text = tok.decode(ref_new).replace("\n", "\\n")
        print(f"\n{'PASS' if ok else 'FAIL'} {prompt[:50]!r} ({len(ids)} tokens)")
        print(f"  states: max relative difference {max(diffs):.1e} (at {['embeddings', *[f'layer {k}' for k in range(1, len(diffs) - 1)], 'final norm'][i_worst]}); "
              f"logits {ld:.1e}; argmax same at {same_argmax}/{len(ids)} positions")
        print(f"  greedy: {n_same}/{len(ref_new)} tokens identical; smallest top-2 margin {min(margins):.3f} -> {text[:70]!r}")
        if ours_new != ref_new:
            print(f"  diverged at step {n_same}: reference margin there {margins[n_same]:.4f}; "
                  f"reference {ref_new[n_same:n_same + 4]}, llmi {ours_new[n_same:n_same + 4]}")

    print(f"\n{len(prompts)} prompts: worst state difference {worst_state:.1e} ({worst_where}), worst logits {worst_logit:.1e}; "
          f"greedy tokens identical {same_new}/{total_new}")
    print(f"generation time (no KV cache on either side): reference {t_ref:.1f} s, llmi {t_ours:.1f} s including process start")
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
