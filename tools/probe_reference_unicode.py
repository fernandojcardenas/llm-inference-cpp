#!/usr/bin/env python3
"""Finds which Unicode version each step of the reference tokenizer uses.

The GPT-2 pattern's \\p{L} and \\p{N} come from the regex engine's tables and
the Digits pre-tokenizer from Rust's char::is_numeric; the two can lag
different Unicode releases. For every code point whose class changed between
the candidate versions, this probes Hugging Face tokenizers directly:

  - letter:  pattern piece count of c c x is 1 only if c is in \\p{L}
  - number:  pattern piece count of c c 5 is 1 only if c is in \\p{N}
  - digit:   Digits splits a c b into 3 pieces only if c is numeric

then checks the chosen versions against every code point.

  curl -sS -o UnicodeData-15.1.txt https://www.unicode.org/Public/15.1.0/ucd/UnicodeData.txt
  curl -sS -o UnicodeData-16.txt https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt
  curl -sS -o UnicodeData-17.txt https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt
  python3 tools/probe_reference_unicode.py UnicodeData-15.1.txt UnicodeData-16.txt UnicodeData-17.txt
"""
import pathlib
import sys

from tokenizers import __version__ as tokenizers_version
from tokenizers import pre_tokenizers as P

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gen_unicode_tables as g  # noqa: E402

pattern = P.ByteLevel(add_prefix_space=False, use_regex=True)
digits = P.Digits(individual_digits=True)
ALL = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]


def pieces(pt, s):
    return len(pt.pre_tokenize_str(s))


PROBES = {
    "letter (pattern \\p{L})": ("L", lambda c: pieces(pattern, chr(c) * 2 + "x") == 1),
    "number (pattern \\p{N})": ("N", lambda c: pieces(pattern, chr(c) * 2 + "5") == 1),
    "digit (Digits pre-tokenizer)": ("N", lambda c: pieces(digits, "a" + chr(c) + "b") == 3),
}


def cpset(path, major):
    out = set()
    for a, b in g.ranges(path, major):
        out.update(range(a, b + 1))
    return out


def main():
    paths = sys.argv[1:]
    names = [pathlib.Path(p).stem.replace("UnicodeData-", "") for p in paths]
    print(f"tokenizers {tokenizers_version}; candidate Unicode versions: {', '.join(names)}")
    for label, (major, probe) in PROBES.items():
        sets = [cpset(p, major) for p in paths]
        print(f"\n{label}")
        for i in range(1, len(sets)):
            added = sorted(sets[i] - sets[i - 1])
            hits = sum(map(probe, added))
            print(f"  {len(added):5d} code points new in {names[i]} (vs {names[i - 1]}): reference agrees on {hits}")
        best = max(range(len(sets)), key=lambda i: sum(probe(c) == (c in sets[i]) for c in
                                                         sorted(sets[-1] ^ sets[0])))
        wrong = [c for c in ALL if probe(c) != (c in sets[best])]
        print(f"  all {len(ALL):,} code points vs Unicode {names[best]}: {len(wrong)} differ"
              + (f" (first: {', '.join(f'U+{c:04X}' for c in wrong[:5])})" if wrong else ""))


if __name__ == "__main__":
    main()
