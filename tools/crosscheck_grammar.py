#!/usr/bin/env python3
"""Cross-checks llmi-grammar's matcher against an independent recognizer (M8).

The engine's matcher (src/grammar/matcher.cpp) runs a grammar as a pushdown
automaton over stacks, with repetitions desugared into right-recursive rules.
This script re-implements the same GBNF subset from scratch in Python with a
different algorithm -- an Earley parser, with repetitions desugared the other
way (left recursion and explicit alternatives) -- and asks both, for every
string it enumerates: is it a complete sentence, a viable prefix, or neither?

Strings are enumerated exhaustively up to a length per grammar, as a tree:
once a prefix is rejected, its one-character extensions are still checked
(both must reject them) but not expanded further. Long sentences are added by
random derivation, plus every single-character edit of each.

Usage: crosscheck_grammar.py BUILD_DIR
Exit status 1 on any disagreement.
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# An independent GBNF parser


class GrammarError(Exception):
    pass


def parse_gbnf(text):
    """Returns (rules, start): rules maps name -> list of productions; a
    production is a list of ('t', ranges, negated) or ('n', name)."""
    rules = {}
    fresh = [0]
    i = 0
    n = len(text)

    def new_name(base):
        fresh[0] += 1
        return f"{base}#{fresh[0]}"

    def skip():
        nonlocal i
        while i < n:
            if text[i] in " \t\r\n":
                i += 1
            elif text[i] == "#":
                while i < n and text[i] != "\n":
                    i += 1
            else:
                break

    def is_name(c):
        return c.isascii() and (c.isalnum() or c in "-_")

    def name():
        nonlocal i
        s = i
        while i < n and is_name(text[i]):
            i += 1
        return text[s:i]

    def rule_start():
        nonlocal i
        save = i
        nm = name()
        skip()
        ok = bool(nm) and text.startswith("::=", i)
        i = save
        return ok

    def char():
        nonlocal i
        c = text[i]
        if c != "\\":
            i += 1
            return ord(c)
        e = text[i + 1]
        i += 2
        simple = {"n": "\n", "r": "\r", "t": "\t"}
        if e in simple:
            return ord(simple[e])
        if e in "\\\"[]-^/":
            return ord(e)
        width = {"x": 2, "u": 4, "U": 8}[e]
        v = int(text[i:i + width], 16)
        i += width
        return v

    def primary(owner):
        nonlocal i
        c = text[i]
        if c == '"':
            i += 1
            out = []
            while text[i] != '"':
                cp = char()
                out.append(("t", [(cp, cp)], False))
            i += 1
            return out
        if c == "[":
            i += 1
            neg = text[i] == "^"
            if neg:
                i += 1
            ranges = []
            while text[i] != "]":
                lo = char()
                hi = lo
                if text[i] == "-" and text[i + 1] != "]":
                    i += 1
                    hi = char()
                ranges.append((lo, hi))
            i += 1
            return [("t", ranges, neg)]
        if c == ".":
            i += 1
            return [("t", [], True)]
        if c == "(":
            i += 1
            alts = alternatives(owner)
            skip()
            assert text[i] == ")"
            i += 1
            g = new_name(owner)
            rules[g] = alts
            return [("n", g)]
        return [("n", name())]

    def number():
        nonlocal i
        s = i
        while text[i].isdigit():
            i += 1
        return int(text[s:i])

    def postfix(owner, unit):
        nonlocal i
        if i >= n or text[i] not in "*+?{":
            return unit
        c = text[i]
        i += 1
        if c == "{":
            skip()
            lo = number()
            skip()
            if text[i] == ",":
                i += 1
                skip()
                hi = None if text[i] == "}" else number()
            else:
                hi = lo
            skip()
            assert text[i] == "}"
            i += 1
        else:
            lo, hi = {"*": (0, None), "+": (1, None), "?": (0, 1)}[c]
        # Desugar differently from the engine: unbounded repetition is left
        # recursive (Earley handles it); bounded repetition is an explicit
        # list of alternatives, one per count.
        r = new_name(owner)
        if hi is None:
            star = new_name(owner)
            rules[star] = [[], [("n", star)] + unit]
            rules[r] = [unit * lo + [("n", star)]]
        else:
            rules[r] = [unit * k for k in range(lo, hi + 1)]
        return [("n", r)]

    def sequence(owner):
        out = []
        while True:
            skip()
            if i >= n or text[i] in "|)" or (is_name(text[i]) and rule_start()):
                return out
            out += postfix(owner, primary(owner))

    def alternatives(owner):
        nonlocal i
        alts = [sequence(owner)]
        while True:
            skip()
            if i < n and text[i] == "|":
                i += 1
                alts.append(sequence(owner))
            else:
                return alts

    skip()
    while i < n:
        nm = name()
        skip()
        if not text.startswith("::=", i):
            raise GrammarError(f"expected ::= after {nm!r}")
        i += 3
        rules[nm] = alternatives(nm)
        skip()
    return rules, "root"


def matches_terminal(sym, cp):
    _, ranges, neg = sym
    return any(lo <= cp <= hi for lo, hi in ranges) != neg


# ---------------------------------------------------------------------------
# Earley recognition, one character at a time


class Earley:
    def __init__(self, rules, start):
        self.rules = rules
        self.start = start
        self.nullable = set()
        changed = True
        while changed:
            changed = False
            for a, prods in rules.items():
                if a in self.nullable:
                    continue
                if any(all(s[0] == "n" and s[1] in self.nullable for s in p) for p in prods):
                    self.nullable.add(a)
                    changed = True

    def closure(self, items, pos, charts):
        """Completes and predicts within one Earley set (in place)."""
        agenda = list(items)
        while agenda:
            item = agenda.pop()
            a, pi, dot, origin = item
            prod = self.rules[a][pi]
            if dot == len(prod):
                for b, pj, d2, o2 in list(charts[origin]) if origin < pos else list(items):
                    p2 = self.rules[b][pj]
                    if d2 < len(p2) and p2[d2] == ("n", a):
                        new = (b, pj, d2 + 1, o2)
                        if new not in items:
                            items.add(new)
                            agenda.append(new)
                continue
            sym = prod[dot]
            if sym[0] == "n":
                for pj in range(len(self.rules[sym[1]])):
                    new = (sym[1], pj, 0, pos)
                    if new not in items:
                        items.add(new)
                        agenda.append(new)
                if sym[1] in self.nullable:
                    new = (a, pi, dot + 1, origin)
                    if new not in items:
                        items.add(new)
                        agenda.append(new)

    def initial(self):
        items = {(self.start, pi, 0, 0) for pi in range(len(self.rules[self.start]))}
        charts = [items]
        self.closure(items, 0, charts)
        return charts

    def step(self, charts, cp):
        pos = len(charts)
        nxt = set()
        for a, pi, dot, origin in charts[-1]:
            prod = self.rules[a][pi]
            if dot < len(prod) and prod[dot][0] == "t" and matches_terminal(prod[dot], cp):
                nxt.add((a, pi, dot + 1, origin))
        new_charts = charts + [nxt]
        self.closure(nxt, pos, new_charts)
        return new_charts

    def verdict(self, charts):
        last = charts[-1]
        if not last:
            return "-"
        for a, pi, dot, origin in last:
            if a == self.start and origin == 0 and dot == len(self.rules[a][pi]):
                return "C"
        return "P"


# ---------------------------------------------------------------------------


def enumerate_cases(earley, alphabet, max_len):
    """(string, verdict) for the tree of strings up to max_len."""
    out = []
    stack = [("", earley.initial())]
    while stack:
        s, charts = stack.pop()
        v = earley.verdict(charts) if s else ("C" if earley.verdict(charts) == "C" else "P")
        out.append((s, v))
        if v == "-" or len(s) >= max_len:
            continue
        for ch in alphabet:
            stack.append((s + ch, earley.step(charts, ord(ch))))
    return out


def random_sentence(rules, start, rng, budget=60):
    """A random derivation, steering to the shortest choices once over budget."""
    shortest = {}
    changed = True
    while changed:
        changed = False
        for a, prods in rules.items():
            for p in prods:
                cost = 0
                for s in p:
                    cost += 1 if s[0] == "t" else shortest.get(s[1], 10**9)
                if cost < shortest.get(a, 10**9):
                    shortest[a] = cost
                    changed = True
    out = []

    def pick_char(sym):
        _, ranges, neg = sym
        if neg:
            for cp in [ord(c) for c in "azZ09 _-xé€"]:
                if matches_terminal(sym, cp):
                    return cp
            raise GrammarError("no sample char")
        lo, hi = rng.choice(ranges)
        return rng.randint(lo, min(hi, lo + 50))

    def expand(a, depth):
        prods = rules[a]
        if len(out) > budget or depth > 40:
            p = min(prods, key=lambda p: sum(1 if s[0] == "t" else shortest.get(s[1], 10**9) for s in p))
        else:
            p = rng.choice(prods)
        for s in p:
            if s[0] == "t":
                out.append(chr(pick_char(s)))
            else:
                expand(s[1], depth + 1)

    expand(start, 0)
    return "".join(out)


def check(build, name, gbnf, alphabet, max_len, samples, rng, is_schema=False):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / ("x.json" if is_schema else "x.gbnf")
        src.write_text(gbnf, encoding="utf-8")
        if is_schema:
            gbnf = subprocess.run([str(build / "llmi-grammar"), "--schema", str(src)], check=True,
                                  capture_output=True, text=True).stdout
        rules, start = parse_gbnf(gbnf)
        earley = Earley(rules, start)
        cases = enumerate_cases(earley, alphabet, max_len) if max_len > 0 else []
        for _ in range(samples):
            s = random_sentence(rules, start, rng)
            edits = [s]
            for k in range(len(s)):
                edits.append(s[:k] + s[k + 1:])
                edits.append(s[:k] + rng.choice(alphabet) + s[k + 1:])
            for t in edits:
                charts = earley.initial()
                for ch in t:
                    charts = earley.step(charts, ord(ch))
                    if not charts[-1]:
                        break
                cases.append((t, earley.verdict(charts) if t else ("C" if earley.verdict(charts) == "C" else "P")))
        lines = "".join(json.dumps(s) + "\n" for s, _ in cases)
        res = subprocess.run([str(build / "llmi-grammar"), "--schema" if is_schema else "--grammar", str(src),
                              "--jsonl"], input=lines, capture_output=True, text=True, check=True)
        got = res.stdout.split("\n")[:-1]
        bad = [(s, want, g) for (s, want), g in zip(cases, got) if want != g]
        counts = {k: sum(1 for _, v in cases if v == k) for k in "CP-"}
        status = "PASS" if not bad and len(got) == len(cases) else "FAIL"
        print(f"{status} {name}: {len(cases)} strings (complete {counts['C']}, prefix {counts['P']}, "
              f"rejected {counts['-']}), {len(bad)} disagreements")
        for s, want, g in bad[:10]:
            print(f"    {s!r}: reference {want}, engine {g}")
        return status == "PASS"


CASES = [
    ("alternatives and classes", 'root ::= "yes" | "no" | [0-9] [a-f]?', "yesno07afz", 4, 0),
    ("balanced parentheses", 'root ::= "(" root ")" root | ""', "()x", 10, 50),
    ("counted repetition", 'root ::= "a"{2,4} "b"{3} "c"{1,}', "abc", 10, 50),
    ("negated classes and any", 'root ::= "\\"" [^"\\\\]* "\\"" .', '"\\a!', 6, 50),
    ("ambiguous splits", 'root ::= x*\nx ::= "a" | "aa" | "aaa"', "ab", 12, 20),
    ("nested groups", 'root ::= ("a" ("b" | "c")* | "d"){1,3} "e"?', "abcde", 6, 50),
    ("unicode classes", 'root ::= [\\u00e9-\\u00eb]+ "\\u20ac" | "\\u65e5\\u672c"', "éëf€日本", 5, 20),
    ("small JSON", 'root ::= "{" ws (pair (ws "," ws pair)*)? ws "}"\n'
                   'pair ::= "\\"" [a-b]{1,2} "\\"" ws ":" ws [0-9]+\nws ::= [ ]{0,2}', '{}",: a0', 5, 200),
]

SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string", "maxLength": 8},
        "kind": {"enum": ["truck", "boat", None]},
        "tags": {"type": "array", "items": {"type": "boolean"}, "maxItems": 2},
        "pos": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 2,
                                              "maxItems": 2}]},
    },
    "required": ["id", "name", "kind", "tags", "pos"],
    "additionalProperties": False,
})


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    build = Path(sys.argv[1])
    rng = random.Random(8)
    ok = True
    for name, gbnf, alphabet, max_len, samples in CASES:
        ok &= check(build, name, gbnf, alphabet, max_len, samples, rng)
    ok &= check(build, "JSON Schema (strict subset)", SCHEMA, '{}[]",:01aeklmnorstu.- ', 0, 300, rng,
                is_schema=True)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
