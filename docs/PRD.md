# Product Requirements Document — MinerU-Popo (Rust Native)

> Status: Draft v1 · Owner: Core team · Target branch for development: `dev`
> Scope: Port the current Python post-processing framework to a Rust-native,
> multi-task, high-performance engine with first-class profiling and metrics.
> Python is treated as a reference implementation only and will be fully retired.

---

## 1. Background

MinerU-Popo is a universal **post-processing** framework for structured document
parsing. It bridges the gap between *page-level* OCR/layout parsing (MinerU,
MonkeyOCR, Dolphin, PaddleOCR-VL, GLM-OCR, …) and *document-level* semantic
structure. A 4B post-processing vision-language model (Popo, a Qwen3-VL
fine-tune) performs four subtasks, and a deterministic pipeline assembles a
document tree, enriches it, and evaluates it.

The current implementation is Python. It works but has structural limitations
for production use: GIL-bound concurrency, heavy/fragile dependency chains
(torch, transformers, PyMuPDF, PIL, BeautifulSoup, zss), per-process startup
cost, weak observability, and difficult single-binary deployment.

### Current pipeline (reference implementation)

| Stage | Python entrypoint | Responsibility |
| --- | --- | --- |
| 1. Label normalization | `post_processing/label_normalization.py` | Adapt 5+ OCR model output formats into one canonical block schema; normalize bboxes to `xyxy_01`; render PDF pages for sizing |
| 2. Inference | `post_processing/inference.py` + `run_inference.py` | Run the 4 subtasks via the Popo VLM, with dynamic chunking and cross-chunk synchronization |
| 3. Tree build | `post_processing/get_json_tree.py` + `table_merge_utils.py` | Assemble a typed document tree; merge cross-page tables |
| 4. Enrichment | `generate_metadata.py`, `split_subnode.py` | Generate titles/summaries; split long-section nodes |
| 5. Evaluation | `eval/evaluate.py` | TEDS (tree-edit-distance) hierarchy scoring via `zss` |

### The four subtasks

1. **Text-truncation analysis** — detect text continuing across blocks/pages and
   link them (`<|txt_contd|>`).
2. **Title-hierarchy analysis** — assign heading levels; reconcile levels across
   chunks via bias synchronization.
3. **Image-text association** — bind images/charts/tables to captions/footnotes
   and their owning section.
4. **Table-merge analysis** — heuristic pre-filter (6 checks) + VLM judgment to
   merge cross-page tables at the cell level.

---

## 2. Problem Statement

The Python codebase cannot meet our production targets for **throughput,
deployability, and observability**. Specifically:

- **Concurrency ceiling.** The GIL plus per-chunk `asyncio` / per-doc
  `ThreadPoolExecutor` makes it hard to saturate modern many-core machines or
  to overlap CPU-bound parsing (HTML/table/tree work) with I/O-bound model
  calls efficiently.
- **Deployment weight.** Shipping torch/transformers/CUDA wheels, PyMuPDF, PIL,
  and a Python runtime is large, version-fragile, and slow to cold-start.
- **Observability gaps.** There is no consistent tracing, no per-stage latency
  histograms, no resource metrics, and no standard profiling story. The repo
  pulls in `prometheus-fastapi-instrumentator` but the pipeline itself is not
  instrumented.
- **Two divergent code paths.** `inference.py` (asyncio) and
  `data_engine/add_link.py` (threads, GPT backend) re-implement the same four
  subtasks with subtle differences, doubling maintenance.

## 3. Goals

| # | Goal | Success measure |
| --- | --- | --- |
| G1 | **Rust-native engine** that reproduces the full pipeline (stages 1–5) | Byte-for-byte (or semantically equivalent) tree output vs. Python reference on a golden corpus |
| G2 | **Multi-task, high concurrency** across documents, chunks, and subtasks | Linear-ish scaling to N cores; saturate a model endpoint's concurrency budget |
| G3 | **High performance** end to end | ≥ 5× docs/sec vs. Python on identical hardware and identical model endpoint (excluding model GPU time) |
| G4 | **First-class profiling & metrics** | Tracing spans per stage/subtask; Prometheus metrics; flamegraph + benchmark suite in CI |
| G5 | **Single-binary, easy deploy** | One static-ish binary + config; container image < current footprint |
| G6 | **Full Python retirement** | All five stages + data-engine reach parity; Python deleted from the default path |

## 4. Non-Goals (initial phases)

- **Re-implementing the Popo VLM weights/training in Rust.** Model *inference*
  is consumed via an OpenAI-compatible HTTP endpoint (vLLM/SGLang/TGI). Native
  in-process inference (candle / ONNX / llama.cpp) is an *optional later track*
  (see Roadmap Phase 5), not a launch requirement.
- **Re-implementing OCR/layout parsers.** We consume their JSON outputs.
- **A new training data engine UI.** `data_engine/` is ported for parity, not
  redesigned.
- **Changing the document-tree schema or TEDS definition.** Output contract is
  held stable to allow golden-file comparison.

---

## 5. Users & Use Cases

- **Pipeline operators** running batch post-processing over large document
  corpora; care about throughput, resumability, and cost.
- **Researchers** comparing OCR backends; care about reproducible TEDS scores.
- **Platform/SRE** deploying the engine as a service; care about a single
  binary, health/metrics endpoints, and predictable resource use.
- **Downstream RAG/analytics** consumers reading the document trees.

---

## 6. Functional Requirements

### FR-1 Label normalization
- Read and adapt at least: MinerU, MonkeyOCR, Dolphin, PaddleOCR-VL, GLM-OCR.
- Emit the canonical `NormalizedBlock` schema; normalize all bboxes to
  `xyxy_01` (validated, finite, ordered).
- Pluggable **reader** trait so new OCR backends are added without touching the
  core.
- PDF page sizing (for sources that need it) via a Rust PDF library.

### FR-2 Inference (the 4 subtasks)
- **Dynamic chunking** by page ranges with boundary-aware overlap (parity with
  `adaptive_chunk`: ~50 items/chunk, overlap=1).
- **Cross-chunk synchronization** for title hierarchy (bias averaging).
- Build subtask prompts in the exact `<|id|>…<|page|>…<|box|>…<|content|>…`
  format the model expects; render multi-page bordered images for image-bearing
  tasks.
- Call the model via a configurable OpenAI-compatible endpoint, with retries,
  backoff, timeouts, and concurrency limits.
- Parse model output deterministically (src/tgt pairs, levels, cell-merge
  coordinate lists) with strict + lenient modes.
- Support **resume** (skip docs already produced) and **dry-run** (validate
  inputs without model calls).

### FR-3 Tree build
- Reconstruct the typed tree: text components by title level, attach
  visual/special elements, append page-supplement nodes.
- **Cross-page table merge** at the cell level (heuristic 6-check filter + the
  merge decision applied to HTML tables; parse via a Rust HTML parser).
- Emit both the JSON tree and the indented text preview.

### FR-4 Enrichment
- Generate node metadata (titles/summaries) and split long section nodes by the
  `<|txt_split|>` / `<|txt_contd|>` markers, mirroring `generate_metadata.py`
  and `split_subnode.py`.

### FR-5 Evaluation
- TEDS hierarchy scoring via a native tree-edit-distance implementation
  (replacing `zss`), including GT alignment (box/text/type similarity).

### FR-6 CLI & config
- Single binary with subcommands: `normalize`, `infer`, `build-tree`,
  `enrich`, `eval`, plus an end-to-end `run` that chains them.
- Config via file + env + flags; preserve current env knobs
  (`POPO_MODEL_PATH`, `POPO_INFERENCE_BACKEND`, `POPO_MAX_NEW_TOKENS`, endpoint
  URL/key) with documented mappings.

### FR-7 Data engine parity
- Port `add_link.py` / `api_utils.py` / table utilities so training-data
  generation reaches parity, unifying it with the inference subtask code (one
  implementation, two backends).

---

## 7. Non-Functional Requirements

### NFR-1 Performance
- ≥ 5× document throughput vs. Python at equal model latency.
- Overlap CPU work (parse/tree/table) with model I/O; bounded memory per doc so
  large corpora stream rather than load wholesale.
- Zero-copy / streaming JSON where practical.

### NFR-2 Concurrency model
- `tokio` async runtime. Three nested levels of parallelism with explicit,
  configurable limits:
  - **document-level** (worker pool over the corpus),
  - **chunk-level** (within a doc),
  - **subtask-level** (the 4 tasks per chunk),
  - plus a **global model-endpoint semaphore** to respect server concurrency.

### NFR-3 Profiling & metrics (first-class)
- **Tracing:** `tracing` spans for every stage, subtask, chunk, and model call,
  with structured fields (doc id, page range, token counts, retries).
- **Metrics:** `metrics` facade with a Prometheus exporter — counters
  (docs/blocks/model-calls/errors), histograms (per-stage latency, tokens,
  image bytes), gauges (in-flight, queue depth). Mirrors the intent of the
  retired `prometheus-fastapi-instrumentator`.
- **Profiling:** `pprof`/`flamegraph` integration and a `criterion` benchmark
  suite; optional `tokio-console` support for async stalls.
- **Reports:** per-run JSON summary (counts, timings, p50/p95/p99) emitted to
  stdout and disk, superseding the current ad-hoc `summary` dicts.

### NFR-4 Reliability
- Deterministic output given the same model responses (record/replay model
  fixtures for tests).
- Robust retry/backoff/timeout on every network call; partial-failure isolation
  so one bad doc doesn't sink a batch.

### NFR-5 Portability & deploy
- Single binary; Linux x86_64 + arm64. Container image significantly smaller
  than the current Python+CUDA image (engine carries no CUDA; GPU lives in the
  model server).

### NFR-6 Quality
- Golden-corpus parity tests in CI; unit tests for every parser/heuristic;
  fuzz tests on the prompt/response parsers and bbox/HTML handling.

---

## 8. Architecture (target)

```
                 ┌──────────────────────────────────────────────┐
                 │                 popo (single binary)          │
                 │                                               │
 OCR JSON  ──►  normalize ──► infer ──► build-tree ──► enrich ──►│──► doc tree (JSON + txt)
   (5 readers)     │            │            │           │       │
                   │            │            │           │       │
                   ▼            ▼            ▼           ▼       │
            ┌───────────────────────────────────────────────┐  │
            │  observability: tracing + metrics + profiling  │  │
            └───────────────────────────────────────────────┘  │
                              │                                  │
                              ▼                                  │
                 OpenAI-compatible model endpoint  ◄────────────┘
                 (vLLM / SGLang / TGI serving Popo + Qwen3-VL)
```

### Proposed crate layout (workspace)

| Crate | Responsibility |
| --- | --- |
| `popo-core` | Canonical schema, block/tree types, errors, config |
| `popo-readers` | OCR adapter trait + per-model readers |
| `popo-pdf` | PDF render/sizing + multi-page image stitching |
| `popo-model` | Model-endpoint client: prompt build, HTTP, retries, concurrency, (later) native backends |
| `popo-infer` | Chunking, the 4 subtasks, synchronization, output parsing |
| `popo-tree` | Tree assembly + cross-page table merge + HTML table utils |
| `popo-enrich` | Metadata generation + subnode splitting |
| `popo-eval` | TEDS + tree-edit-distance + GT alignment |
| `popo-obs` | tracing/metrics/profiling wiring, run-summary reporting |
| `popo-cli` | `clap` subcommands, config loading, the single binary |
| `popo-data-engine` | Training-data generation (parity port) |

### Candidate Rust dependencies

| Need | Python today | Rust candidate(s) |
| --- | --- | --- |
| Async runtime | asyncio | `tokio` |
| HTTP client | openai/requests | `reqwest` (+ OpenAI-compatible JSON via `serde`) |
| JSON | json | `serde` / `serde_json` (+ `simd-json` if needed) |
| PDF render | PyMuPDF (fitz) | `pdfium-render`, `mupdf` bindings, or `pdf`/`lopdf` |
| Image | PIL | `image` |
| HTML/table parse | BeautifulSoup | `scraper` / `html5ever` |
| Tree edit distance | zss | native impl in `popo-eval` |
| CLI | argparse | `clap` |
| Tracing | — | `tracing`, `tracing-subscriber` |
| Metrics | prometheus-fastapi-instrumentator | `metrics` + `metrics-exporter-prometheus` |
| Profiling | — | `pprof`, `criterion`, `tokio-console` |
| Errors | exceptions | `thiserror` / `anyhow` |

> **Model-inference note.** "Abandon Python" applies to *our* code. The Popo /
> Qwen3-VL weights are most pragmatically served by an existing inference server
> behind an OpenAI-compatible API; our Rust engine is a pure client. A fully
> self-hosted native runtime (`candle`, `ort`/ONNX, or `llama.cpp` bindings) is
> an explicit later-phase option, kept behind the same `popo-model` trait so the
> rest of the engine is unaffected.

---

## 9. Migration & Parity Strategy

1. **Freeze the contract.** Capture a **golden corpus**: representative inputs
   plus recorded model request/response pairs and the resulting Python outputs
   for every stage. These become replay fixtures and parity oracles.
2. **Port stage-by-stage**, each gated by a parity test against the golden
   corpus, deepest-leverage first (see Roadmap).
3. **Dual-run** during transition: run Rust and Python on the same inputs and
   diff outputs in CI until divergence is zero (or explained).
4. **Cut over** the default path to Rust; mark Python deprecated.
5. **Delete Python** once all stages + data-engine reach parity and the cutover
   has soaked in production.

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| PDF rendering fidelity differs from PyMuPDF (image pixels feed the VLM) | Output drift | Pin a renderer; pixel-diff against fitz on the golden set; allow tolerance bands |
| HTML/table parsing edge cases (colspan/rowspan) | Wrong merges | Port `table_utils` logic 1:1; fuzz + golden tables |
| Non-determinism from the model | Flaky parity | Record/replay fixtures; temperature pinned in tests |
| Floating-point bbox/score differences | Spurious diffs | Match Python rounding (`int(x*1000)` etc.) exactly; epsilon compares |
| Native inference scope creep | Schedule risk | Keep it behind the `popo-model` trait and out of the launch critical path |
| Two-codepath drift recurs in Rust | Maintenance cost | One subtask implementation shared by inference and data-engine from day one |

## 11. Open Questions

- Which PDF renderer is the long-term choice (`pdfium-render` vs `mupdf`),
  weighing license, fidelity vs. fitz, and static-linking?
- Do we need native in-process inference at all, or is an OpenAI-compatible
  endpoint sufficient for all deployments?
- Target hardware profile for the throughput SLO (core count, NIC, model
  endpoint locality)?
- Is exact byte-parity required for tree JSON, or is semantic-equivalence
  (normalized compare) acceptable as the bar?

---

## 12. Success Criteria (launch)

- All five stages + data-engine at parity on the golden corpus.
- ≥ 5× throughput vs. Python at equal model latency.
- Tracing, Prometheus metrics, and a benchmark/flamegraph suite live in CI.
- Single binary + container shipped; Python removed from the default path and
  scheduled for deletion.
