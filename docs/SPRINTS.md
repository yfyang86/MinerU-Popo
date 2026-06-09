# Sprints — MinerU-Popo Rust Port

> Execution plan that turns [`ROADMAP.md`](./ROADMAP.md) into ~2-week sprints.
> Development on `dev`; PRs target `dev`. Every sprint ends green on
> `cargo fmt`, `cargo clippy`, and `cargo test`.

## Overview

| Sprint | Theme | Roadmap phase | Status |
| --- | --- | --- | --- |
| **1** | Workspace bootstrap + **LLM config + model client** + tracing skeleton | Ph 1–2 | ✅ done |
| **2** | **Record/replay backend + Prometheus metrics** (golden corpus deferred) | Ph 0–1 | ✅ done |
| **3** | `popo-readers` (5 OCR adapters) + `normalize` CLI | Ph 3 | ✅ done |
| **4** | `popo-infer`: chunking + **text-truncation** + **title-hierarchy (+sync)** done | Ph 4 | 🚧 subtasks 1–2 |
| 5 | image-text association + table-merge subtasks | Ph 4 | planned |
| 6 | `popo-tree` + cross-page table merge | Ph 5 | planned |
| 7 | `popo-enrich` + `popo-eval` (native TEDS) | Ph 6 | planned |
| 8 | data-engine parity + codepath unification | Ph 7 | planned |
| 9 | hardening, throughput SLO, cutover, **delete Python** | Ph 8 | planned |

---

## Sprint 1 — Model client foundation ✅

**Goal:** A tested, multi-provider model client driven by the project's TOML
config, behind one trait, ready for every later subtask to consume.

### Delivered

- **Cargo workspace** at repo root with `crates/` (coexists with the Python
  reference, which is retired later).
- **`popo-core`** — shared `Error`/`Result`.
- **`popo-obs`** — `tracing` init honoring `POPO_LOG` / `RUST_LOG` (metrics &
  profiling land in Sprint 2).
- **`popo-model`** — the deliverable:
  - `config` — parses the exact provider TOML (`[default]` + `[providers.*]`
    with `type = openai | claude`); default-provider validation; `env:VAR` /
    `${VAR}` secret indirection so keys aren't committed.
  - `message` — provider-neutral `ChatRequest`/`ChatMessage`/`ContentPart`
    with inline base64 image support (for rendered PDF pages).
  - `backend` — `ModelBackend` trait + `OpenAiBackend` (OpenAI-compatible:
    DeepSeek/Kimi/Minimax/vLLM) and `ClaudeBackend` (Anthropic Messages).
  - `client` — `ModelClient`: provider resolution, bounded concurrency
    (`tokio::Semaphore`), retry-with-exponential-backoff on 429/5xx/transport.
- **`popo-cli`** (`popo` binary) — `config check`, `model list`, `model chat`.
- **`popo.example.toml`** — the config template (env-var keys).

### Verification

- `cargo test` — 14 unit tests + 1 doctest green (config parsing, request-body
  building for both protocols, response parsing, retry classification — all
  network-free).
- `cargo clippy --all-targets` — clean.
- Manual: `popo --config popo.example.toml model list` / `config check` work.

### Notes / follow-ups for Sprint 2

- The `ModelBackend` trait is the seam for the **record/replay** backend
  (deterministic parity tests) and, optionally, a native in-process runtime.
- `ClientOptions` (concurrency, attempts, backoff, timeout) will become
  config-driven and wired to metrics.

---

## Sprint 2 — Determinism & metrics ✅

**Goal:** Make model interactions reproducible (the basis for parity testing)
and make the model client observable.

### Delivered

- **Record/replay backends** (`popo-model::backend::replay`):
  - `RecordingBackend` wraps any backend and writes each exchange to a fixture
    directory, keyed by a platform-stable **FNV-1a fingerprint** of
    `(model, messages, max_tokens, temperature)` — images reduced to a hash so
    fixtures stay small.
  - `ReplayBackend` serves responses hermetically from a fixture directory (no
    network), the foundation for golden-corpus parity tests.
  - `ModelClient::from_config_recording` / `from_backend` reuse the shared
    concurrency/retry/metrics layer around either backend.
- **Prometheus metrics** (`popo-obs::metrics`): install a recorder via the
  `metrics` facade; `ModelClient::chat` emits `popo_model_requests_total`,
  `_request_errors_total`, `_retries_total`, `_tokens_total{direction}`, and
  the `popo_model_request_duration_seconds` histogram, all labeled by provider.
- **CLI**: global `--metrics` flag (prints exposition on exit);
  `POPO_MODEL_REPLAY=<dir>` to replay; `model chat --record <dir>` to capture.

### Verification

- 24 tests green: fingerprint stability/distinguishing, record→replay roundtrip
  via disk, replay-miss error, **CLI replay integration test** (runs the built
  binary against an authored fixture), and a **metrics-instrumentation test**
  (local Prometheus recorder asserts counters render). `clippy`/`fmt` clean.

### Deferred to Sprint 3 prelude

- The **golden corpus** itself needs real OCR inputs + recorded Python outputs,
  which aren't available in this environment; the record/replay machinery and
  parity-diff approach are in place to ingest it as soon as the data exists.

---

## Sprint 3 — Label normalization 🚧

**Goal:** Port stage 1 (`label_normalization.py`) — adapt OCR/layout outputs to
the canonical block schema behind a reader trait, with a `normalize` CLI.

### Delivered (this iteration)

- **`popo-core::schema`** — the canonical `NormalizedBlock`, `to_popo_block` /
  `to_popo_pages` projections, and faithful ports of `normalize_text`,
  `normalize_bbox_to_unit`, `sort_blocks`, `reassign_block_ids` (semantics and
  rounding matched to Python for golden-corpus comparison). `serde_json`
  `preserve_order` is enabled so page maps keep numeric order.
- **`popo-readers`** — the `OcrReader` trait + `ReaderResult`, shared content
  extraction (`content`/`text`/`html`/`words` → `lines[].spans[]` fallback),
  and the **MinerU + MonkeyOCR** readers (`middle.json` and `content_list.json`
  paths, `map_mineru_label`, title promotion via `text_level`, `discarded`
  skipping).
- **`popo normalize`** CLI — discovers `<input>/<doc>/…`, writes
  `<output>/<model>/<doc>.json` as `{input_label, pages}` (mirrors
  `run_label_normalization.sh`).

### Verified

- 6 schema tests + 6 reader tests + an **end-to-end CLI smoke** producing the
  exact `{input_label, pages}` inference input. `clippy`/`fmt` clean.

### Added (second iteration)

- **PaddleOCR-VL** reader (`layout_parsing.json` path + `map_paddle_label`).
- **Dolphin** reader (`recognition_json` + `map_dolphin_label`, including the
  `sec_<n>` heading-level convention).
- **GLM-OCR** reader (`model.json` and `words_result` page paths +
  `map_glm_label`), plus shared `iter_model_pages` / optional-int ordering.
- `build_reader` now resolves all five models; 13 reader tests + Dolphin CLI
  smoke green (43 workspace tests total).

### Deferred to the PDF stage

- Source-PDF page-size sourcing: Paddle per-page `*_res.json`, and exact pixel
  sizing for Dolphin/GLM when payloads omit dimensions (bbox falls back today).
  MinerU `model.json` path also lands with that work.

---

## Sprint 4 — Inference core 🚧

**Goal:** Port stage 2 (`inference.py`) — dynamic chunking and the four
subtasks, sharing the model client.

### Delivered (this iteration)

- **`popo-infer::block`** — the mutable `WorkBlock` and `build_doc_blocks`
  (1-based ids in page/document order; `to_output_value` round-trips to the
  `doc_blocks` shape the tree builder reads).
- **`popo-infer::chunk`** — `adaptive_chunk`, a faithful port of the
  boundary-aware, overlap-preserving page chunker (including the
  `last - start > 2` trailing-chunk guard that drops very short spans).
- **`popo-infer::text`** — the **text-truncation** subtask: `merge_rules`,
  sentence helpers, `is_list_item` (regex), `filter_contd`, the
  `Truncation Detection` prompt, `extract_label1` parsing, and `apply_contd`.
- **`run_text_truncation`** — async runner tying it together through
  `ModelClient`, with page images abstracted by `PageImageProvider`
  (`NoImages` today; real rendering arrives with the PDF stage).

### Verified

- 13 infer tests (heuristics, chunking boundary behavior, prompt/parse/apply)
  including a **replay-backed end-to-end** run that applies a `contd` link.
  56 workspace tests total; `clippy`/`fmt` clean.

### Added (second iteration)

- **`popo-infer::title`** — the **title-hierarchy** subtask: `filter_title`, the
  `Title Level Analysis` prompt, `extract_label2`, and the cross-chunk **bias
  synchronization** (`synchronize`) that pins overlapping headings and subtracts
  the averaged offset from a chunk's new levels (banker's rounding, matching
  Python `round`).
- **`run_title_hierarchy`** — async runner processing chunks **sequentially** so
  the synchronization sees them in order; shares a `chat_for_chunk` helper with
  the text-truncation runner. Replay-backed end-to-end test applies levels.
- 18 infer tests; 61 workspace tests total.

### Remaining for Sprint 4 / 5

- Image-text association and table-merge subtasks; the `infer` CLI; and the
  page-image provider backed by the PDF stage.

---

## Working agreements

- One subtask implementation, two backends — do not fork an `inference` vs
  `data_engine` codepath as the Python repo did.
- Every new stage ships with `tracing` spans, unit tests, and a parity check
  against the golden corpus (from Sprint 2 onward).
- Hold the output contract stable to keep golden-file comparison meaningful.
