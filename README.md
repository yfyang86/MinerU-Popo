# MinerU-Popo: Universal Post-Processing Model for Structured Document Parsing


<p align="center">
  <a href="http://arxiv.org/abs/2605.24973"><img src="https://img.shields.io/badge/arXiv-2605.12882-b31b1b?style=flat-square&logo=arxiv" alt="arXiv" /></a>
  <a href="https://huggingface.co/DreamEternal/MinerU-Popo"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Dataset-HuggingFace-yellow?style=flat-square" alt="Hugging Face dataset" /></a>
  <a href="./LICENSE.txt"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License MIT" /></a>
</p>

<p align="center">
  <b>If you like our project, please give us a star ⭐ on GitHub for the latest update.</b>
</p>

<p align="center">
  📖 <a href="./README.md"><b>English</b></a> &nbsp;|&nbsp; <a href="./README_zh.md"><b>简体中文</b></a>
</p>

![image](./figures/intro.png)

## ✨ Introduction
**MinerU-Popo** is a lightweight and universal framework for POst-Processing OCR outputs, bridging the gap between page-level OCR parsing and document-level semantic structure.
It constructs document tree structures with a 4B post-processing model that performs four subtasks: table truncation analysis, text truncation analysis, title hierarchy analysis, and image-text association analysis. We handle the challenges of cross-page geometric discontinuity, redundant document parsing, and scalability to long documents via:

- **Task-Oriented Data Engine**: Generate representative training data and simplify the task-specific input.
- **Dynamic Chunking and Synchronization**: Process long document by dynamic chunks and reduce deviations across chunks to preserve global consistency.
- **Document Enrichment**: Structurally construct a tree, semantically generate summaries and split long-section nodes.

![image](./figures/overview.png)

## 📊 Performance

### Better Hierarchy (TEDS) after Post-Processing
**Basic OCR** | **Before** | **After**
:---:|:---:|:---:|
 MinerU | 53.7 | **90.6** |
 MonkeyOCR | 48.9 | **87.4** |
 Dolphin | 60.4 | **83.5** |
 PaddleOCR | 59.3 | **82.6** |
 GLM-OCR | 53.5 | **81.8** |

### Advantages Compared to Directly Using Pre-trained Model
**Model** | **TEDS** | **Doc/s**
:---:|:---:|:---:|
 MinerU-Popo | **90.6** | **0.37** |
 Qwen3-VL-2B | 21.2 | 0.22 |
 Qwen3-VL-4B | 56.5 | 0.20 |
 Qwen3-VL-8B | 65.9 | 0.16 |
 Qwen3-VL-32B | 78.0 | 0.04 |

### Benefits for Downstream Retrieval and Analysis (Acc on ViDoRe V3)
**Method** | **C.S.** | **Fin.** | **H.R.** | **Ind.** | **Phar.**
:---:|:---:|:---:|:---:|:---:|:---:|
 MinerU-Popo | **84.4** | 49.5 | **66.8** | 58.7 | **71.6**
 Raw RAG | 82.3 | 48.7 | 63.2 | **60.4** | 64.4
 Visual RAG | 80.7 | **58.4** | 64.8 | 59.7 | 67.6

## ⚙️ Setup

MinerU-Popo is a **Rust-native** engine (the original Python implementation has
been retired; it remains available in the git history for reference). You need a
recent Rust toolchain:

```bash
rustup toolchain install stable   # rustc >= 1.82
cargo build --release             # builds the `popo` binary
```

### Model configuration

Copy `popo.example.toml` to `popo.toml` and set your provider(s). The engine
talks to any OpenAI-compatible endpoint (vLLM/SGLang/TGI serving Popo /
Qwen3-VL) or the Anthropic Messages API:

```toml
[default]
provider = "local_vllm"

[providers.local_vllm]
type = "openai"
api_base = "http://localhost:8000/v1"
api_key = "dummy"
model = "Popo"
```

API keys may be given literally or as `env:VAR` / `${VAR}` to read them from the
environment. Point the CLI at a config with `--config` or `POPO_CONFIG`.

### PDF page images (optional)

The four subtasks are vision-language tasks. To feed the model rendered page
images, build with the `pdfium` feature (requires a `libpdfium` shared library
on your system) and pass `--pdf-dir`:

```bash
cargo build --release --features pdfium
```

Without it, the pipeline runs text-only.

## 💻 Usage

The `popo` binary exposes the pipeline as subcommands. Run the whole thing
end to end:

```bash
popo run \
  --model mineru \
  --input-dir post-process/mineru \
  --work-dir outputs \
  --provider local_vllm \
  --pdf-dir eval_pdf_dir        # optional, needs --features pdfium
```

This produces, under `--work-dir`:

```text
normalized/<model>/<doc>.json   # canonical blocks
inference/<doc>.json            # doc_blocks (4 subtasks applied)
tree/<doc>.json + tree_txt/     # document tree + text preview
final/<doc>.json                # tree with subnode chunking (final result)
```

### Or run the stages individually

```bash
# 1. Normalize OCR/layout outputs (MinerU, MonkeyOCR, Dolphin, PaddleOCR-VL, GLM-OCR)
popo normalize --model mineru --input-dir post-process/mineru --output-dir outputs/normalized

# 2. Run the four post-processing subtasks
popo infer --input-dir outputs/normalized/mineru --output-dir outputs/inference

# 3. Build the document tree (+ cross-page table merge)
popo build-tree --input-dir outputs/inference --output-dir outputs/tree --txt-dir outputs/tree_txt

# 4. Split long/visual nodes into subnodes
popo split-subnode --input-dir outputs/tree --output-dir outputs/final
```

### Evaluation

```bash
popo eval --model mineru --input-dir post-process/mineru \
  --gt-json eval_gt_dir/title.json --output-dir outputs/eval
```

### Other commands

```bash
popo config check          # validate popo.toml
popo model list            # list configured providers
popo model chat "hello"    # one-shot chat against a provider
```

## 🙏 Acknowledgements
- [MinerU](https://github.com/opendatalab/MinerU) and other OCR system (MonkeyOCR, Dolphin, PaddleOCR, GLM-OCR) for page-level parsing.
- [ViDoRe V3](https://huggingface.co/datasets/vidore/vidore-benchmark-v3) and [MMDA](https://huggingface.co/datasets/DreamEternal/MMDA_Bench) as benchmarks.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
