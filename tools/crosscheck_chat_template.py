#!/usr/bin/env python3
"""Cross-checks llmi's chat template engine against Hugging Face's own
chat-template compiler (the same Jinja compiler apply_chat_template() uses)
on both models' real tokenizer_config.json, across a range of message lists.

Usage:
  python3 tools/crosscheck_chat_template.py MODEL_DIR BINARY [--limit N]

MODEL_DIR must contain tokenizer_config.json. BINARY is the
llmi-chat-template executable.
"""
import argparse
import json
import subprocess
import sys
import tempfile

from transformers.utils.chat_template_utils import _compile_jinja_template

CASES = [
    ("single user turn", [{"role": "user", "content": "What is 12 times 7?"}], True),
    ("single user turn, no generation prompt", [{"role": "user", "content": "Hi"}], False),
    ("explicit system prompt", [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}], True),
    (
        "multi-turn history",
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "Thanks, bye"},
        ],
        True,
    ),
    (
        "content with newlines and quotes",
        [{"role": "user", "content": "Line one\nLine \"two\"\nLine 'three'"}],
        True,
    ),
    (
        "content with special-token-looking text",
        [{"role": "user", "content": "What does <|im_start|> mean?"}],
        True,
    ),
    ("empty content", [{"role": "user", "content": ""}], True),
    (
        "unicode content",
        [{"role": "user", "content": "Bonjour! 你好 こんにちは émotions café"}],
        True,
    ),
    (
        "long multi-turn, no generation prompt at the end",
        [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(10)],
        False,
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("binary")
    ap.add_argument("--limit", type=int, default=len(CASES))
    args = ap.parse_args()

    with open(f"{args.model_dir}/tokenizer_config.json") as f:
        cfg = json.load(f)
    chat_template = cfg["chat_template"]
    ref_tpl = _compile_jinja_template(chat_template)

    failures = 0
    for name, messages, add_gen in CASES[: args.limit]:
        ref = ref_tpl.render(messages=messages, add_generation_prompt=add_gen, tools=None)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            json.dump(messages, f)
            f.flush()
            run_args = [args.binary, f"{args.model_dir}/tokenizer_config.json", f.name]
            if not add_gen:
                run_args.append("--no-generation-prompt")
            result = subprocess.run(run_args, check=True, capture_output=True, text=True)
        ours = result.stdout
        ok = ours == ref
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name!r} ({len(messages)} messages, add_generation_prompt={add_gen})")
        if not ok:
            print(f"  reference: {ref!r}")
            print(f"  llmi:      {ours!r}")

    print(f"\n{args.model_dir}: {len(CASES[:args.limit]) - failures}/{len(CASES[:args.limit])} cases identical to the reference")
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
