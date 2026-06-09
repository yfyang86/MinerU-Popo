# Sprints — MinerU-Popo Rust Port

> Execution plan that turns [`ROADMAP.md`](./ROADMAP.md) into ~2-week sprints.
> Development on `dev`; PRs target `dev`. Every sprint ends green on
> `cargo fmt`, `cargo clippy`, and `cargo test`.

## Overview

| Sprint | Theme | Roadmap phase | Status |
| --- | --- | --- | --- |
| **1** | Workspace bootstrap + **LLM config + model client** + tracing skeleton | Ph 1–2 | ✅ done |
| **2** | **Record/replay backend + Prometheus metrics** (golden corpus deferred) | Ph 0–1 | ✅ done |
| **3** | `popo-readers` + `normalize` CLI — **MinerU family done**; Paddle/Dolphin/GLM pending | Ph 3 | 🚧 in progress |
| 4 | `popo-infer`: chunking + text-truncation + title-hierarchy (+sync) | Ph 4 | planned |
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

### Remaining for Sprint 3

- Readers for **PaddleOCR-VL, Dolphin, GLM-OCR**; MinerU `model.json` path;
  PDF page-size sourcing for readers that need it.

---

## Working agreements

- One subtask implementation, two backends — do not fork an `inference` vs
  `data_engine` codepath as the Python repo did.
- Every new stage ships with `tracing` spans, unit tests, and a parity check
  against the golden corpus (from Sprint 2 onward).
- Hold the output contract stable to keep golden-file comparison meaningful.
