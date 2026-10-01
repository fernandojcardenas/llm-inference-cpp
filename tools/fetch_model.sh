#!/usr/bin/env bash
# Downloads a model from Hugging Face at a pinned revision and verifies every
# file against a pinned SHA-256. Weights are never committed to this repo.
#
#   tools/fetch_model.sh smollm2-135m models/smollm2-135m
#   tools/fetch_model.sh qwen2.5-0.5b-instruct models/qwen2.5-0.5b-instruct
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODEL DEST_DIR   (models: smollm2-135m, qwen2.5-0.5b-instruct)" >&2
  exit 2
fi
model=$1
dest=$2

case $model in
  smollm2-135m)  # Apache 2.0
    repo=HuggingFaceTB/SmolLM2-135M
    revision=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
    files=(
      "config.json 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
      "tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
      "model.safetensors 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
    ) ;;
  qwen2.5-0.5b-instruct)  # Apache 2.0
    repo=Qwen/Qwen2.5-0.5B-Instruct
    revision=7ae557604adf67be50417f59c2c2f167def9a775
    files=(
      "config.json 18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
      "tokenizer.json c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
      "model.safetensors fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
    ) ;;
  *)
    echo "unknown model: $model" >&2
    exit 2 ;;
esac

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p "$dest"
for entry in "${files[@]}"; do
  read -r name want <<<"$entry"
  path=$dest/$name
  if [[ -f $path && $(sha256 "$path") == "$want" ]]; then
    echo "ok       $path"
    continue
  fi
  echo "fetching $repo@${revision:0:12} $name"
  curl -fsSL --retry 3 -o "$path.part" "https://huggingface.co/$repo/resolve/$revision/$name"
  got=$(sha256 "$path.part")
  if [[ $got != "$want" ]]; then
    rm -f "$path.part"
    echo "SHA-256 mismatch for $name: expected $want, got $got" >&2
    exit 1
  fi
  mv "$path.part" "$path"
  echo "verified $path"
done
