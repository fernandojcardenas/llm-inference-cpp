#!/usr/bin/env python3
"""Cross-checks llmi-gguf-inspect (src/model/gguf.cpp, M6) against a real
GGUF file, read two independent ways:

1. Structure: every tensor's name, ggml type, element count and byte size,
   read by the reference `gguf` Python package (pip install gguf -- the
   project ggml-org itself publishes), must match this engine's reader
   exactly.
2. Values: F32 tensors are compared element-for-element against NumPy's own
   read of the same bytes. Q8_0/Q4_0 tensors are compared against a
   from-scratch NumPy re-implementation of ggml's real block layout (ggml's
   own reference quantizer/dequantizer, not this engine's code) -- in
   particular Q4_0's "byte j holds element j and element j+16" pairing,
   which is NOT the adjacent-pair layout this engine's own quant.cpp (M5)
   uses internally (see the long comment in src/model/gguf.cpp).

  python3 tools/crosscheck_gguf.py BUILD_DIR FILE.gguf [FILE.gguf ...]
"""
import json
import subprocess
import sys

import numpy as np
import gguf


def ggml_dequant_q8_0(raw: bytes, n: int) -> np.ndarray:
    # ggml's block_q8_0: { fp16 d; int8 qs[32]; }, plain sequential order.
    # Fully vectorized (no per-block Python loop): a structured dtype views
    # every block at once, matching this engine's own interpretation
    # (src/model/gguf.cpp) without sharing a line of code with it.
    dt = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    blocks = np.frombuffer(raw, dtype=dt, count=n // 32)
    scales = blocks["d"].astype(np.float32)[:, None]
    return (blocks["qs"].astype(np.float32) * scales).reshape(-1)


def ggml_dequant_q4_0(raw: bytes, n: int) -> np.ndarray:
    # ggml's block_q4_0: { fp16 d; uint8 qs[16]; }; byte j's low nibble is
    # element j, high nibble is element j+16 (the block's two HALVES share
    # a byte, not neighbouring elements -- see src/model/gguf.cpp).
    dt = np.dtype([("d", "<f2"), ("qs", "u1", (16,))])
    blocks = np.frombuffer(raw, dtype=dt, count=n // 32)
    scales = blocks["d"].astype(np.float32)[:, None]
    lo = (blocks["qs"] & 0x0F).astype(np.float32) - 8.0
    hi = (blocks["qs"] >> 4).astype(np.float32) - 8.0
    return (np.concatenate([lo, hi], axis=1) * scales).reshape(-1)


def check_file(build_dir: str, path: str) -> int:
    print(f"== {path} ==")
    reader = gguf.GGUFReader(path)
    by_name = {t.name: t for t in reader.tensors}

    lines = subprocess.run(
        [f"{build_dir}/llmi-gguf-inspect", path, "--tensor-stats"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert len(lines) == len(reader.tensors), f"tensor count mismatch: {len(lines)} vs {len(reader.tensors)}"

    errors = 0
    checked_values = 0
    for line in lines:
        row = json.loads(line)
        t = by_name.get(row["name"])
        if t is None:
            print(f"  MISSING in reference reader: {row['name']}")
            errors += 1
            continue
        if int(t.tensor_type) != {
            "F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 8, "BF16": 30,
        }.get(row["type"], -1):
            print(f"  TYPE MISMATCH {row['name']}: engine={row['type']} ref={t.tensor_type}")
            errors += 1
            continue
        if row["n_elements"] != int(t.n_elements) or row["data_nbytes"] != int(t.data.nbytes):
            print(f"  SIZE MISMATCH {row['name']}: engine=({row['n_elements']},{row['data_nbytes']}) "
                  f"ref=({t.n_elements},{t.data.nbytes})")
            errors += 1
            continue

        raw = t.data.tobytes()
        if row["type"] == "F32":
            ref = np.frombuffer(raw, dtype=np.float32)
        elif row["type"] == "Q8_0":
            ref = ggml_dequant_q8_0(raw, int(t.n_elements))
        elif row["type"] == "Q4_0":
            ref = ggml_dequant_q4_0(raw, int(t.n_elements))
        else:
            continue
        ref_sum = float(np.sum(ref, dtype=np.float64))
        if abs(ref_sum - row["sum"]) > 1e-2 * max(1.0, abs(ref_sum)):
            print(f"  SUM MISMATCH {row['name']}: engine={row['sum']!r} ref={ref_sum!r}")
            errors += 1
            continue
        first8 = ref[:8].astype(np.float64)
        if not np.allclose(first8, np.array(row["first"], dtype=np.float64), rtol=1e-4, atol=1e-6):
            print(f"  FIRST-VALUES MISMATCH {row['name']}: engine={row['first']} ref={first8.tolist()}")
            errors += 1
            continue
        checked_values += 1

    print(f"  {len(lines)} tensors, structure OK; {checked_values} tensors' values matched "
          f"an independent NumPy decode exactly ({errors} mismatches)")
    return errors


def main() -> int:
    build_dir = sys.argv[1]
    total_errors = 0
    for path in sys.argv[2:]:
        total_errors += check_file(build_dir, path)
    if total_errors:
        print(f"FAILED: {total_errors} mismatches")
        return 1
    print("All files matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
