#!/usr/bin/env python3
"""Cross-checks how llmi reads model weights against PyTorch + safetensors.

For every tensor in model.safetensors, llmi-inspect --tensor-stats prints
dtype, shape, the first 8 values and the sum of all values (as float, summed
in double precision). This script computes the same with PyTorch and
compares: names, dtypes and shapes must be identical, the first values
bit-identical, and the sums equal to within summation-order rounding.

Usage:
  build/llmi-inspect MODEL_DIR --tensor-stats > stats.jsonl
  python3 tools/crosscheck_weights.py MODEL_DIR/model.safetensors stats.jsonl
"""
import json
import math
import sys

import torch
from safetensors.torch import load_file

DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    reference = load_file(sys.argv[1])
    with open(sys.argv[2], encoding="utf-8") as f:
        ours = {r["name"]: r for r in map(json.loads, f)}

    problems = []
    if set(ours) != set(reference):
        problems.append(f"tensor names differ: only in llmi {sorted(set(ours) - set(reference))[:5]}, "
                        f"only in reference {sorted(set(reference) - set(ours))[:5]}")
    values = 0
    worst = 0.0
    for name in sorted(set(ours) & set(reference)):
        t, r = reference[name], ours[name]
        if DTYPES.get(t.dtype) != r["dtype"] or list(t.shape) != r["shape"]:
            problems.append(f"{name}: reference {t.dtype} {list(t.shape)}, llmi {r['dtype']} {r['shape']}")
            continue
        flat = t.reshape(-1).to(torch.float32)
        values += flat.numel()
        first = flat[:8].tolist()
        if any(struct_bits(a) != struct_bits(b) for a, b in zip(first, r["first"])) or len(first) != len(r["first"]):
            problems.append(f"{name}: first values differ: reference {first}, llmi {r['first']}")
        ref_sum = flat.to(torch.float64).sum().item()
        abs_sum = flat.to(torch.float64).abs().sum().item()
        # Both sums add the same float values in double precision, in different
        # orders; allow that rounding, relative to the sum of magnitudes.
        err = abs(ref_sum - r["sum"]) / max(abs_sum, 1e-300)
        worst = max(worst, err)
        if err > 1e-12:
            problems.append(f"{name}: sum reference {ref_sum!r}, llmi {r['sum']!r} (relative error {err:.2e})")

    print(f"PyTorch {torch.__version__}: {len(reference)} tensors, {values:,} values")
    print(f"names, dtypes, shapes and first values identical; largest relative difference in sums {worst:.1e}"
          if not problems else "")
    for p in problems[:20]:
        print("MISMATCH", p)
    print("FAIL" if problems else "PASS")
    return 1 if problems else 0


def struct_bits(x):
    import struct
    return struct.pack("<f", x) if not math.isnan(x) else b"nan"


if __name__ == "__main__":
    sys.exit(main())
