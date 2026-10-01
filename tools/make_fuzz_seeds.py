#!/usr/bin/env python3
"""Writes small valid seed inputs for the fuzz targets (no dependencies).

  python3 tools/make_fuzz_seeds.py OUT_DIR
creates OUT_DIR/{safetensors,json,config,tokenizer}/seed-*.
"""
import json
import pathlib
import struct
import sys


def safetensors_file(tensors, metadata=None):
    header, data, off = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        data += raw
        off += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + data


def main():
    out = pathlib.Path(sys.argv[1])
    root = pathlib.Path(__file__).resolve().parent.parent
    for d in ("safetensors", "json", "config", "tokenizer"):
        (out / d).mkdir(parents=True, exist_ok=True)
    seeds = {
        "a": {"w": ("F32", [2, 2], struct.pack("<4f", 1, -2, 0.5, 3))},
        "b": {"x": ("BF16", [3], b"\x80\x3f\x00\x40\x00\xc0"), "y": ("F16", [1], b"\x00\x3c"),
              "z": ("I64", [], struct.pack("<q", 7))},
        "c": {"e": ("F32", [0, 4], b"")},
    }
    for k, tensors in seeds.items():
        (out / "safetensors" / f"seed-{k}").write_bytes(safetensors_file(tensors, {"format": "pt"} if k == "a" else None))
    config = (root / "testdata/smollm2-135m/config.json").read_bytes()
    (out / "json" / "seed-config").write_bytes(config)
    (out / "json" / "seed-misc").write_bytes(b'{"a":[1,-2.5e3,true,null,"\\u00e9\\ud83d\\ude42"],"b":{}}')
    (out / "config" / "seed-config").write_bytes(config)
    for i, text in enumerate(["\x01Hello, world! It's 2026.", "\x01<|im_start|>user\nHi<|im_end|>",
                              "\x00naïve café 東京 🙂  \t\n", "\x03a\x04b 123 'll"]):
        (out / "tokenizer" / f"seed-{i}").write_bytes(text.encode())


if __name__ == "__main__":
    main()
