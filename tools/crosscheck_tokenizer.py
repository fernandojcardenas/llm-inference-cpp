#!/usr/bin/env python3
"""Cross-checks llmi-tokenize against Hugging Face `tokenizers` on the same
tokenizer.json: the token ids must be identical for every input.

Inputs:
  1. every .py file in this Python's standard library, each file as one
     document (real code and prose: indentation, comments, docstrings, numbers);
  2. --random N seeded random strings built to stress the edge cases:
     every Unicode White_Space character, letters and digits from many
     scripts, combining marks, emoji, contractions, special tokens, control
     characters (including the ones with no token) and random code points;
  3. --all-codepoints: every Unicode code point in four probe contexts
     (c c x, c 5, a c b, ' c 1), which together expose how each character is
     classified (letter, number, digit, whitespace or other) by every step.

Usage:
  python3 tools/crosscheck_tokenizer.py TOKENIZER_JSON build/llmi-tokenize [--random 50000] [--seed 1] [--all-codepoints]
"""
import argparse
import json
import pathlib
import random
import subprocess
import sys
import sysconfig
import tempfile
import time

from tokenizers import Tokenizer, __version__ as tokenizers_version

WHITESPACE = [chr(c) for c in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
                               0x2028, 0x2029, 0x202F, 0x205F, 0x3000)]
NOT_WHITESPACE_BUT_CLOSE = ["​", "‌", "‍", "⁠", "﻿", "\x1c", "\x1d", "\x1e", "\x1f", "᠎"]
LETTERS = list("abcxyzABCXYZéñßøłİıΣσςДжאبणกა東京ㅎ가あア") + ["ǅ", "ʰ", "々"]  # Lt, Lm
NUMBERS = list("0123456789") + ["٣", "१", "３", "²", "½", "Ⅳ", "〇", "\U0001d7d8"]
MARKS = ["́", "̈", "َ", "ि", "⃝", "️"]
SYMBOLS = list("!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'") + ["—", "«", "€", "©", "→"]
EMOJI = ["\U0001f642", "\U0001f468‍\U0001f469‍\U0001f467", "\U0001f1fa\U0001f1f8", "❤️"]
CONTRACTIONS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'RE", "'x", "''"]
CONTROLS = [chr(c) for c in range(0x00, 0x20)] + ["\x7f", "\x80", "\x9f"]


def random_codepoint(rng):
    while True:
        cp = rng.randrange(0x110000)
        if not 0xD800 <= cp <= 0xDFFF:
            return chr(cp)


def random_string(rng, specials):
    pools = [WHITESPACE, NOT_WHITESPACE_BUT_CLOSE, LETTERS, NUMBERS, MARKS, SYMBOLS, EMOJI, CONTRACTIONS, CONTROLS,
             specials, [" "], ["\n"]]
    weights = [6, 1, 10, 5, 2, 5, 1, 2, 1, 1, 8, 2]
    parts = []
    for _ in range(rng.randrange(1, 40)):
        if rng.random() < 0.05:
            parts.append(random_codepoint(rng))
        else:
            pool = rng.choices(pools, weights)[0]
            parts.append(rng.choice(pool) * (rng.randrange(1, 4) if rng.random() < 0.2 else 1))
    return "".join(parts)


def stdlib_documents():
    root = pathlib.Path(sysconfig.get_paths()["stdlib"])
    docs = []
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if text:
            docs.append((str(path.relative_to(root)), text))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("binary")
    ap.add_argument("--random", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--all-codepoints", action="store_true")
    args = ap.parse_args()

    hf = Tokenizer.from_file(args.tokenizer)
    specials = [t.content for t in hf.get_added_tokens_decoder().values()]
    rng = random.Random(args.seed)

    inputs = stdlib_documents()
    n_docs = len(inputs)
    inputs += [(f"random #{i}", random_string(rng, specials)) for i in range(args.random)]
    n_probes = 0
    if args.all_codepoints:
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            c = chr(cp)
            for probe in (c + c + "x", c + "5", "a" + c + "b", " " + c + "1"):
                inputs.append((f"U+{cp:04X} probe", probe))
                n_probes += 1

    t0 = time.perf_counter()
    expected = []
    for i in range(0, len(inputs), 20000):  # chunks keep memory bounded
        expected += [e.ids for e in hf.encode_batch([text for _, text in inputs[i:i + 20000]])]
    hf_seconds = time.perf_counter() - t0

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as f:
        for _, text in inputs:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")
        jsonl = f.name
    t0 = time.perf_counter()
    with open(jsonl, encoding="utf-8") as stdin:
        proc = subprocess.run([args.binary, args.tokenizer, "--jsonl"], stdin=stdin, capture_output=True, text=True)
    llmi_seconds = time.perf_counter() - t0
    pathlib.Path(jsonl).unlink()
    sys.stderr.write(proc.stderr)
    got = [json.loads(line) for line in proc.stdout.splitlines()]
    if len(got) != len(inputs):
        print(f"FAIL: llmi-tokenize returned {len(got)} lines for {len(inputs)} inputs (exit {proc.returncode})")
        return 1

    mismatches = [(name, text, e, g) for (name, text), e, g in zip(inputs, expected, got) if e != g]
    total_chars = sum(len(t) for _, t in inputs)
    total_tokens = sum(len(e) for e in expected)
    print(f"tokenizers {tokenizers_version}, Python {sys.version.split()[0]}")
    print(f"inputs: {n_docs} standard-library files + {args.random} random strings (seed {args.seed})"
          + (f" + {n_probes:,} code-point probes (every code point x 4)" if n_probes else ""))
    print(f"characters: {total_chars:,}  tokens: {total_tokens:,}")
    print(f"identical token ids: {len(inputs) - len(mismatches):,} of {len(inputs):,} inputs")
    print(f"time: Hugging Face {hf_seconds:.2f} s in-process (batched, multithreaded), "
          f"llmi-tokenize {llmi_seconds:.2f} s single-threaded including JSON I/O")
    for name, text, e, g in mismatches[:10]:
        print(f"\nMISMATCH {name}: {text[:200]!r}\n  hf:   {e[:40]}\n  llmi: {g[:40]}")
    if mismatches or proc.returncode != 0:
        print(f"\nFAIL: {len(mismatches)} mismatches, llmi-tokenize exit status {proc.returncode}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
