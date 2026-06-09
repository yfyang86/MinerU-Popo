# MinerU-Popo — Development Manual

> Engineering reference for the **Rust-native** MinerU-Popo engine: architecture,
> conventions, walkthroughs, gotchas, and future work. Pairs with
> [`PRD.md`](./PRD.md), [`ROADMAP.md`](./ROADMAP.md), and [`SPRINTS.md`](./SPRINTS.md).

---

## 1. What this project is

MinerU-Popo is a **document post-processing engine**. It turns *page-level* OCR /
layout output (MinerU, MonkeyOCR, Dolphin, PaddleOCR-VL, GLM-OCR) into a
*document-level* structure tree, using a 4B vision-language model (Popo, a
Qwen3-VL fine-tune) that performs four subtasks:

1. **Text-truncation** — link text that continues across blocks/pages.
2. **Title-hierarchy** — assign heading levels, reconciled across chunks.
3. **Image-text association** — bind images/tables to captions/footnotes/owners.
4. **Table-merge** — merge cross-page tables.

The engine was ported from a Python reference (now removed; preserved in git
history) to a Rust workspace. The model is consumed over an **OpenAI-compatible
HTTP endpoint** (or the Anthropic API); the orchestration is pure Rust.

### The pipeline

```
OCR JSON ──normalize──► canonical blocks ──infer──► doc_blocks ──build-tree──► tree ──split-subnode──► final tree
  (5 readers)            {input_label,pages}   (4 subtasks)     (+table merge)            (chunked)
                                                                                   eval (TEDS) ◄── readers + GT
```

Run it all with `popo run`, or stage by stage (`popo normalize | infer |
build-tree | split-subnode`), plus `popo eval`.

---

## 2. Architecture

### 2.1 Crate graph

A Cargo workspace under `crates/`. Arrows = "depends on".

```
popo-core   (schema, errors)                 popo-obs (tracing + Prometheus)
   ▲   ▲                                          (standalone)
   │   └──────────────┐
popo-model         popo-readers      popo-table       popo-enrich   popo-eval
(config,            (OcrReader +      (HTML tables,    (subnode      (TEDS,
 backends,           5 adapters)       merge_html)      split)        alignment)
 ModelClient)            ▲                 ▲   ▲            ▲            ▲
   ▲                     │                 │   │            │            │
   └──────── popo-infer ─┘                 │   │            │            │
            (WorkBlock, chunking,──────────┘   │            │            │
             4 subtasks, run_inference)        │            │            │
                 ▲          ▲                  │            │            │
        popo-data-engine  popo-pdf ────────────┘            │            │
        (training cases)  (render + stitch;                 │            │
                           pdfium feature)                  │            │
                 └──────────────┬───────────────────────────┴────────────┘
                            popo-cli  (the `popo` binary, all subcommands)
```

Internal dependencies (precise):

| Crate | Depends on (internal) | External highlights |
| --- | --- | --- |
| `popo-core` | — | serde, thiserror, serde_json |
| `popo-obs` | — | tracing, metrics, metrics-exporter-prometheus |
| `popo-model` | popo-core | reqwest, tokio, async-trait, metrics |
| `popo-readers` | popo-core | serde_json |
| `popo-table` | — | scraper, unicode-normalization |
| `popo-infer` | popo-core, popo-model, popo-table | regex, tokio |
| `popo-tree` | popo-table | serde, serde_json |
| `popo-eval` | popo-core | regex |
| `popo-enrich` | — | regex, serde_json |
| `popo-data-engine` | popo-infer | serde_json |
| `popo-pdf` | popo-infer, popo-core | image, base64; `pdfium-render` (feature) |
| `popo-cli` | all of the above | clap, tokio, anyhow |

### 2.2 Crate responsibilities

- **`popo-core`** — the cross-cutting contract: `Error`/`Result`, and `schema`
  (`NormalizedBlock`, `Bbox`, `normalize_text`, `normalize_bbox_to_unit`,
  `sort_blocks`, `reassign_block_ids`, `to_popo_pages`). Match Python rounding
  here (`(x*1000.0) as i64`) — it feeds parity.
- **`popo-obs`** — `init()` installs a `tracing` subscriber (honors `POPO_LOG`);
  `metrics::install()` returns a Prometheus handle. Library crates emit through
  the `metrics` facade so they don't depend on an exporter.
- **`popo-model`** — everything model: `config` (the `popo.toml` provider TOML),
  `message` (neutral `ChatRequest`/`ChatMessage`/`ContentPart`), `backend`
  (`ModelBackend` trait + `openai`, `claude`, and `replay`/`recording`), and
  `client` (`ModelClient`: provider resolution, bounded concurrency via a
  `tokio::Semaphore`, retry-with-backoff, metrics).
- **`popo-readers`** — `OcrReader` trait + `ReaderResult`, `common` helpers
  (content extraction, `make_block`, `iter_model_pages`), and the five readers.
  `build_reader(model, dir)` is the factory.
- **`popo-table`** — HTML table structure: `parse_table` (scraper),
  colspan/rowspan occupancy, column counts, `detect_table_headers`, span-row
  extraction, `extract_last_coordinates` (a small Python-literal parser), and
  `merge_html`.
- **`popo-infer`** — the heart. `block` (`WorkBlock`, `build_doc_blocks`,
  `doc_blocks_to_json`), `chunk` (`adaptive_chunk`, the `Paged` trait), one
  module per subtask (`text`, `title`, `image`, `table`), and the crate root
  with the `run_*` async runners, `run_inference`, and the `PageImageProvider`
  trait (+ `NoImages`).
- **`popo-tree`** — `build_tree(doc_blocks) -> Component` (supplement remap →
  cross-page table merge → text components → heading-level tree → visual/special
  attachment → supplements) and `tree_to_txt`.
- **`popo-eval`** — native **Zhang-Shasha** TEDS (`teds`), a faithful
  `SequenceMatcher.ratio` port, bbox metrics, alignment, and the parsing of GT
  prompts/labels.
- **`popo-enrich`** — `split_subnode` / `split_tree`: the final tree chunking.
- **`popo-data-engine`** — training-case formatters that **reuse** `popo-infer`'s
  prompt builders + output serializers (the unification), plus `extract_json`.
- **`popo-pdf`** — `stitch_pages_with_border`, `to_jpeg_base64`,
  `crop_unit_bbox`, the `PageRenderer` trait, and `PdfPageImages` (implements
  `popo-infer::PageImageProvider`). `pdfium` feature adds `PdfiumRenderer`.
- **`popo-cli`** — the `popo` binary: one module per subcommand.

### 2.3 Key data types

- **`NormalizedBlock`** (`popo-core`) — reader output: `{block_id, page, bbox
  (xyxy_01), kind, content, order, popo_type, title_level?, source_label?,
  meta}`. Projected to the **popo-pages** map (`{ "1": [popo_block, …] }`) via
  `to_popo_pages`.
- **`WorkBlock`** (`popo-infer::block`) — the mutable inference unit:
  `{id (1-based), page, kind, content, bbox, contd, level, image, table_merge?,
  cell_list?, source}`. `to_output_value` round-trips to the `doc_blocks` array.
- **`Component`** (`popo-tree`) — a tree node: `{type, title, metadata, content,
  level, location, block_ids, children}`. `split_subnode` adds `subnode`.
- **`ModelClient`** / **`ModelBackend`** (`popo-model`) — the single choke point
  for all model I/O.

### 2.4 The model seam

Everything that talks to a model goes through `ModelBackend`:

```
OpenAiBackend  ┐
ClaudeBackend  ├─ dyn ModelBackend ─ ModelClient ─ run_* subtasks
ReplayBackend  ┘   (retry, concurrency, metrics)
RecordingBackend (wraps another, captures fixtures)
```

This is why subtasks are testable offline (replay) and why a future native
in-process runtime (candle/ONNX) would slot in without touching callers.

---

## 3. Repository layout

```
Cargo.toml                 workspace (members + [workspace.dependencies])
popo.example.toml          provider config template (copy to popo.toml)
.github/workflows/ci.yml   fmt + clippy + test gate
crates/
  popo-core/  popo-obs/  popo-model/  popo-readers/  popo-table/
  popo-infer/  popo-tree/  popo-eval/  popo-enrich/  popo-data-engine/
  popo-pdf/   popo-cli/    (src/<subcommand>.rs + tests/)
docs/
  PRD.md  ROADMAP.md  SPRINTS.md  DevelopmentManual.md (this file)
figures/  output_cases/  LICENSE  README.md  README_zh.md
```

---

## 4. Development rules

1. **Toolchain = CI's `stable`.** CI runs `dtolnay/rust-toolchain@stable`
   (currently 1.96). Develop on the same: `rustup update stable && rustup
   default stable`. A newer clippy adds lints your older one won't catch — this
   has bitten us (see §6).
2. **The gate is non-negotiable.** Before every push:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
   CI runs exactly these three. Keep clippy at zero warnings; prefer fixing over
   `#[allow(...)]`, and when you do allow, scope it and comment why.
3. **Branch + PR flow.** Develop on a feature branch; **PR into `dev`** (never
   `master`). Each PR is one logical increment, gated by CI, then merged. Keep
   the local branch tip at *your own* verified commit — do not fast-forward onto
   GitHub's server-side merge commits (they show as "Unverified"; see §6).
4. **Commits.** Conventional style (`feat(rust): …`, `fix(enrich): …`); a
   `!` for breaking/cutover. End the body with the session link. **No backticks
   in `git commit -m` strings** (the shell runs them — see §6).
5. **Faithful ports, with the Python in mind.** When porting, cite the Python
   function in a doc comment, preserve its semantics (including rounding and
   quirks), and add unit tests. Where you deviate, say so explicitly in a
   comment.
6. **One subtask implementation.** Inference and the data engine share
   `popo-infer`'s prompt builders/serializers. Do not fork a parallel
   implementation (that was the original Python's `inference.py` vs
   `add_link.py` problem).
7. **Model I/O only through `ModelBackend`/`ModelClient`.** Don't call HTTP
   directly from a subtask.
8. **Observability is first-class.** New stages get `tracing` spans and (where
   it matters) `metrics`. Don't add a stage with no diagnostics.
9. **Tests are hermetic.** Use `ReplayBackend` for model-dependent tests and
   synthetic fixtures for everything else; no network, no external services.
10. **Native deps are opt-in.** Anything needing a system library (pdfium) lives
    behind a Cargo feature that is **off by default**, so the default build and
    CI never require it.

---

## 5. Walkthroughs

### 5.1 Build & run

```bash
cargo build --release                      # the `popo` binary at target/release/popo
cp popo.example.toml popo.toml             # then fill in a provider/key

# whole pipeline:
popo run --model mineru --input-dir post-process/mineru --work-dir outputs \
  --provider local_vllm --config popo.toml

# or per stage — see README "Usage".
popo config check        # validate config
popo model list          # list providers
popo eval --model mineru --input-dir post-process/mineru \
  --gt-json eval_gt_dir/title.json --output-dir outputs/eval
```

### 5.2 Running with real VLM page images

```bash
cargo build --release --features pdfium     # needs libpdfium on the system
popo infer --input-dir outputs/normalized/mineru --output-dir outputs/inference \
  --pdf-dir eval_pdf_dir
```

Without the feature (or without `--pdf-dir`), inference is text-only via
`NoImages`.

### 5.3 Add a new OCR reader

1. `crates/popo-readers/src/<model>.rs`: a struct + `impl OcrReader`. Map the
   model's labels to `(canonical_type, popo_type)`, normalize bboxes with
   `popo_core::normalize_bbox_to_unit`, build blocks with `common::make_block`.
2. Register it in `build_reader()` (and re-export the type in `lib.rs`).
3. Unit-test the label map and one `read_*` path with a synthetic JSON fixture.
4. The `normalize` CLI and `eval` pick it up automatically.

### 5.4 Add a model backend

1. `crates/popo-model/src/backend/<name>.rs`: implement `ModelBackend`
   (`model()` + `async chat()`), with a **pure** `build_body()` you can unit-test
   without a network, and `parse_body()`.
2. Add a `ProviderType` variant + wire it in `client::build_backend`.
3. Tests: body building and response parsing (see `openai.rs`/`claude.rs`).

### 5.5 Add / modify an inference subtask

Each subtask is split into **pure functions** (filter → prompt → parse → apply)
plus a thin async runner in `lib.rs` that loops chunks through
`chat_for_chunk`. Follow `text.rs`/`title.rs`. Test the pure functions directly,
then add one **replay-backed end-to-end** test (author a `Fixture` whose
fingerprint matches the prompt the runner builds — see the tests in
`popo-infer/src/lib.rs`).

### 5.6 Record / replay model fixtures

```rust
// Capture (against a live endpoint):
let client = ModelClient::from_config_recording(&cfg, provider, "fixtures/", opts)?;
// Replay (hermetic):
let backend = ReplayBackend::from_dir("fixtures/")?;
let client  = ModelClient::from_backend("replay", Arc::new(backend), opts);
```

The fingerprint is a platform-stable FNV-1a over `(model, messages, max_tokens,
temperature)`. The CLI honors `POPO_MODEL_REPLAY=<dir>` for `infer`.

### 5.7 Add a CLI subcommand

Add `crates/popo-cli/src/<cmd>.rs` (an `Args` struct + `run`), then register the
module, the `Command` variant, and the dispatch arm in `main.rs`. Prefer
reusing library crates over inlining logic (see `run.rs`, which composes the
stage commands).

---

## 6. Hints & gotchas (hard-won)

- **Clippy version skew.** CI's `stable` clippy can be newer than your local one
  and reject code that compiled for you (e.g. `explicit_counter_loop`). Keep
  local on `stable`. This caused the only red CI in the port (PR #14).
- **`adaptive_chunk` drops short docs.** Its trailing-chunk guard
  (`last - start > 2`) means a candidate span of ≤2 pages produces **no chunk**,
  so the subtask makes **no model call**. This is faithful to Python and is the
  basis of several offline tests — don't "fix" it.
- **Bbox per-mille strings.** Subtask judge blocks serialize bboxes as
  `[top, left, bottom, right]` scaled by 1000 via `(x*1000.0) as i64` (truncates
  toward zero). The order swap and truncation are intentional (parity).
- **`serde_json` `preserve_order`** is enabled workspace-wide so page maps and
  components keep insertion order (numeric page order, not lexical).
- **GitHub merge commits.** The merge API creates a server-side commit authored
  by the repo owner (`noreply@github.com`) that shows as "Unverified". Don't
  fast-forward your local feature branch onto it; keep your branch at your own
  verified commit (it stays an ancestor of `dev`). A stop-hook enforces verified
  committer identity — set `user.email=noreply@anthropic.com`.
- **No backticks in commit `-m`.** `git commit -m "...\`x\`..."` runs `x` in the
  shell. Use plain text or a heredoc.
- **Title sync runs sequentially.** `run_title_hierarchy` processes chunks in
  order because the bias synchronization mutates shared state — don't parallelize
  it (matches the Python coroutine semantics where the blocking call never
  yields).
- **`SequenceMatcher.ratio` autojunk** (popularity heuristic) only triggers at
  length ≥ 200; the Rust port omits it, which is exact for the short title
  strings it scores. Note this if you reuse `ratio` for long text.
- **pdfium lifetimes.** A `PdfPage` borrows its document; bind it to a variable
  before calling `render_with_config` (chaining off the temporary fails to
  compile, E0716).

---

## 7. Future work

Ordered roughly by leverage. Items 1–2 are the most important.

1. **Golden-corpus parity (highest priority).** The single gap: no real OCR
   outputs + recorded Python results existed in the port environment, so ports
   are *faithful and unit-tested* but **not byte-diffed** against Python. Build a
   golden corpus, record model request/response fixtures (`RecordingBackend`),
   and add a dual-run diff in CI before trusting exact-match output.
2. **Finish the not-yet-ported pieces** (all recoverable from git history):
   - **Metadata generation** (`generate_metadata.py`) — per-node summaries from
     the model over `popo_pdf::crop_unit_bbox` crops; now unblocked by the PDF
     stage. Add to `popo-enrich` + an `enrich` CLI step.
   - **GPT data-engine flow** (`add_linkings_*`) — the judge-block building +
     GPT JSON round-trip that *produces* training results (the formatters that
     consume them are already in `popo-data-engine`).
   - **Semantic table cell-merge** (`merge_table_html`) — colspan reconciliation
     + `cell_list`-driven cell joining; a refinement on the current structural
     row-append in `popo_table::merge_html` (needs a mutable HTML DOM, e.g.
     `kuchikiki`).
   - **Paddle per-page reader** and the **page-number overlay** in the stitched
     image (needs `ab_glyph`/`imageproc` + a bundled font).
3. **Observability buildout** (Roadmap NFR-3): a `/metrics` endpoint for a
   service mode, a `criterion` benchmark suite + flamegraphs, and per-run JSON
   summaries with p50/p95/p99.
4. **Performance pass.** Confirm the ≥5× throughput SLO; tune the document /
   chunk / subtask concurrency limits and the global endpoint semaphore.
5. **Single-binary release & container.** Linux x86_64 + arm64; a slim image;
   wire `enable_pr_auto_merge` once branch protection requires the CI check.
6. **Optional native inference** (Roadmap "Optional Track"): a `candle` / `ort` /
   `llama.cpp` `ModelBackend` behind the existing trait, benchmarked vs the HTTP
   endpoint. No pipeline changes required.

---

## 8. Quick reference

```bash
# the gate (run before every push)
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace

# env
POPO_CONFIG=popo.toml            # config path (or --config)
POPO_LOG=info                    # tracing filter (or RUST_LOG)
POPO_MODEL_REPLAY=fixtures/      # hermetic replay for `infer`

# features
cargo build --features pdfium    # real PDF page images (needs libpdfium)
```
