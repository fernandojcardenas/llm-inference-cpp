# Third-party material

| Material | Where | Licence | Source |
|---|---|---|---|
| SmolLM2-135M `tokenizer.json` and `config.json` | `testdata/smollm2-135m/` (committed) | Apache License 2.0 (`testdata/smollm2-135m/LICENSE`) | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), revision `93efa2f0` |
| SmolLM2-135M weights | downloaded by `tools/fetch_model.sh`, not committed | Apache License 2.0 | as above |
| Qwen2.5-0.5B-Instruct config, tokenizer and weights | downloaded by `tools/fetch_model.sh`, not committed | Apache License 2.0 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), revision `7ae55760` |
| Qwen2.5-0.5B-Instruct `tokenizer_config.json` (chat template only) | `testdata/qwen2.5-0.5b-instruct/` (committed) | Apache License 2.0 (model card) | as above, same revision |
| SmolLM2-135M-Instruct `tokenizer_config.json` (chat template only) | `testdata/smollm2-135m-instruct/` (committed) | Apache License 2.0 (model card) | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f7` |
| Unicode character tables in `src/tokenizer/unicode_tables.inc` | generated from `UnicodeData.txt` 16.0.0 and 17.0.0 | [Unicode License v3](https://www.unicode.org/license.txt) | [unicode.org](https://www.unicode.org/Public/) |
| GoogleTest 1.17.0 | downloaded at configure time (tests only) | BSD 3-Clause | [google/googletest](https://github.com/google/googletest) |

Cross-checks use Hugging Face `tokenizers` and `safetensors` (Apache 2.0) and PyTorch (BSD-style) as
references; none of them is part of the engine.
