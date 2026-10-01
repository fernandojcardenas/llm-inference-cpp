#!/usr/bin/env python3
"""Tokenizes a prefix of the WikiText-2-raw test split with a model's own
Hugging Face tokenizer, for llmi-perplexity (M5's quality-loss evidence).

WikiText-2 is the standard small perplexity benchmark (used by llama.cpp's
own `llama-perplexity` tool among others), so numbers from this script are
directly comparable to numbers llama.cpp reports on the same text. The same
ids file is reused for every --quant run of llmi-perplexity on a given
model, so quantization's effect on perplexity is never confounded by a
difference in tokenization.

Usage:
  python3 tools/make_ppl_ids.py MODEL_DIR --out ids.txt [--target-tokens 4096]
"""
import argparse

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def load_wikitext2_test_text() -> str:
    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = pq.read_table(path).column("text").to_pylist()
    return "".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=4096)
    args = ap.parse_args()

    text = load_wikitext2_test_text()
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    # Grow the text window until tokenizing it yields at least target_tokens
    # ids, rather than tokenizing the whole 1.3 MB test split every time.
    chars = args.target_tokens * 6  # a generous first guess, refined below
    ids = []
    while len(ids) < args.target_tokens and chars < len(text):
        ids = tok.encode(text[:chars])
        chars *= 2
    ids = ids[: args.target_tokens]
    if len(ids) < args.target_tokens:
        raise SystemExit(f"wikitext-2 test split only yielded {len(ids)} tokens, wanted {args.target_tokens}")

    with open(args.out, "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    print(f"wrote {len(ids)} token ids ({chars // 2} source chars) from {tok.__class__.__name__} to {args.out}")


if __name__ == "__main__":
    main()
