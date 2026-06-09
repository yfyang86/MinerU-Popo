# Sprints — MinerU-Popo Rust Port

> Execution plan that turns [`ROADMAP.md`](./ROADMAP.md) into ~2-week sprints.
> Development on `dev`; PRs target `dev`. Every sprint ends green on
> `cargo fmt`, `cargo clippy`, and `cargo test`.

## Overview

| Sprint | Theme | Roadmap phase | Status |
| --- | --- | --- | --- |
| **1** | Workspace bootstrap + **LLM config + model client** + tracing skeleton | Ph 1–2 | ✅ done |
| 2 | Golden corpus + parity harness + record/replay backend + Prometheus metrics | Ph 0–1 | planned |
| 3 | `popo-readers` (label normalization, 5 OCR adapters) + `normalize` CLI | Ph 3 | planned |
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

## Working agreements

- One subtask implementation, two backends — do not fork an `inference` vs
  `data_engine` codepath as the Python repo did.
- Every new stage ships with `tracing` spans, unit tests, and a parity check
  against the golden corpus (from Sprint 2 onward).
- Hold the output contract stable to keep golden-file comparison meaningful.
