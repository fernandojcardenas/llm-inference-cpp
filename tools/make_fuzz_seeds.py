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


def gguf_string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf_file(tensors, metadata=None, alignment=32):
    """tensors: {name: (type_code, dims, raw_bytes)}. metadata: {key: (value_type, packed_value_bytes)}."""
    out = b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata or {}))
    for key, (vtype, vbytes) in (metadata or {}).items():
        out += gguf_string(key) + struct.pack("<I", vtype) + vbytes
    data, offset, infos = b"", 0, b""
    for name, (type_code, dims, raw) in tensors.items():
        while offset % alignment != 0:  # every tensor's offset is aligned, not only the first
            data += b"\x00"
            offset += 1
        infos += gguf_string(name) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", type_code) + struct.pack("<Q", offset)
        data += raw
        offset += len(raw)
    out += infos
    while len(out) % alignment != 0:
        out += b"\x00"
    return out + data


def main():
    out = pathlib.Path(sys.argv[1])
    root = pathlib.Path(__file__).resolve().parent.parent
    for d in ("safetensors", "gguf", "json", "config", "tokenizer"):
        (out / d).mkdir(parents=True, exist_ok=True)
    seeds = {
        "a": {"w": ("F32", [2, 2], struct.pack("<4f", 1, -2, 0.5, 3))},
        "b": {"x": ("BF16", [3], b"\x80\x3f\x00\x40\x00\xc0"), "y": ("F16", [1], b"\x00\x3c"),
              "z": ("I64", [], struct.pack("<q", 7))},
        "c": {"e": ("F32", [0, 4], b"")},
    }
    for k, tensors in seeds.items():
        (out / "safetensors" / f"seed-{k}").write_bytes(safetensors_file(tensors, {"format": "pt"} if k == "a" else None))

    gguf_seeds = {
        "f32": ({"w": (0, [4], struct.pack("<4f", 1, -2, 0.5, 3))},
                {"general.architecture": (8, gguf_string("llama"))}),
        "q8_0": ({"w": (8, [32], b"\x00\x3c" + bytes(range(32)))}, None),  # scale 1.0, codes 0..31
        "q4_0": ({"w": (2, [32], b"\x00\x40" + bytes([0x91] + [0x88] * 15))}, None),  # scale 2.0
        "multi": ({"a": (0, [2, 2], struct.pack("<4f", 1, 2, 3, 4)),
                   "b": (1, [4], b"\x00\x3c\x00\x40\x00\xc0\x00\x4a")},
                  {"general.alignment": (4, struct.pack("<I", 32)),
                   "tokenizer.ids": (9, struct.pack("<I", 4) + struct.pack("<Q", 3) +
                                     struct.pack("<3I", 1, 2, 3))}),
    }
    for k, (tensors, metadata) in gguf_seeds.items():
        (out / "gguf" / f"seed-{k}").write_bytes(gguf_file(tensors, metadata))
    config = (root / "testdata/smollm2-135m/config.json").read_bytes()
    (out / "json" / "seed-config").write_bytes(config)
    (out / "json" / "seed-misc").write_bytes(b'{"a":[1,-2.5e3,true,null,"\\u00e9\\ud83d\\ude42"],"b":{}}')
    (out / "config" / "seed-config").write_bytes(config)
    for i, text in enumerate(["\x01Hello, world! It's 2026.", "\x01<|im_start|>user\nHi<|im_end|>",
                              "\x00naïve café 東京 🙂  \t\n", "\x03a\x04b 123 'll"]):
        (out / "tokenizer" / f"seed-{i}").write_bytes(text.encode())


if __name__ == "__main__":
    main()
