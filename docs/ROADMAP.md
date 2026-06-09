# Roadmap — MinerU-Popo Rust Native Port

> Companion to [`PRD.md`](./PRD.md). This roadmap sequences the Python → Rust
> migration into shippable phases, each gated by a **parity check** against the
> golden corpus. Development happens on `dev`; PRs target `dev`, not `master`.
>
> Sequencing principle: build the **contract + observability + model client +
> chunking core first**, then port stages deepest-leverage first, dual-run for
> parity, then retire Python.

---

## Phase 0 — Foundations & contract freeze

**Goal:** Make the port measurable and reversible before writing engine code.

- [ ] Set up the Cargo **workspace** with the crate layout from the PRD; CI
      (fmt, clippy, test, build) on `dev`.
- [ ] Define the canonical **schema** in `popo-core` (`NormalizedBlock`, block,
      tree node, run-summary) with `serde`.
- [ ] Build the **golden corpus**: representative inputs across all 5 OCR
      backends + recorded model request/response pairs + Python stage outputs.
- [ ] **Parity harness**: a test runner that diffs Rust vs. recorded Python
      output per stage (byte and/or normalized compare).
- [ ] Decide PDF renderer and HTML parser via spikes (pixel-diff vs. fitz;
      colspan/rowspan correctness).

**Exit:** Workspace builds in CI; golden corpus + parity harness exist and run
(against an empty/stub engine).

---

## Phase 1 — Observability spine (`popo-obs`)

**Goal:** Profiling/metrics are first-class from the start, not bolted on.

- [ ] `tracing` + `tracing-subscriber` with structured spans and JSON logs.
- [ ] `metrics` facade + Prometheus exporter (counters/histograms/gauges per
      PRD NFR-3); optional `/metrics` endpoint for service mode.
- [ ] Run-summary reporter (counts + p50/p95/p99 timings) to stdout + disk,
      superseding the Python `summary` dicts.
- [ ] `criterion` bench harness skeleton; `pprof`/flamegraph wiring;
      `tokio-console` feature flag.

**Exit:** Any later stage gets spans + metrics + a benchmark slot for free.

---

## Phase 2 — Model client & concurrency core (`popo-model`)

**Goal:** A robust, observable, rate-limited client behind a trait — the single
choke point all subtasks share.

- [ ] `ModelBackend` trait; OpenAI-compatible HTTP impl via `reqwest`.
- [ ] Retries + exponential backoff + timeouts; map current env knobs
      (`POPO_*`, endpoint URL/key) to config.
- [ ] **Global endpoint semaphore** + per-level concurrency limits (`tokio`).
- [ ] Record/replay mode driven by the golden fixtures (deterministic tests).
- [ ] Prompt-building primitives + multi-page bordered image stitching
      (`popo-pdf` + `image`).

**Exit:** Can issue concurrent, instrumented, replayable model calls with
bounded concurrency.

---

## Phase 3 — Label normalization (`popo-readers`)  ⟶ first end-user parity

**Goal:** Port Stage 1; prove the reader trait + parity harness on real output.

- [ ] `OcrReader` trait + readers: MinerU, MonkeyOCR, Dolphin, PaddleOCR-VL,
      GLM-OCR.
- [ ] Bbox normalization to `xyxy_01` with the exact validation rules
      (finite, ordered, range) and the same rounding (`int(x*1000)`).
- [ ] PDF page sizing where required.
- [ ] `popo normalize` CLI subcommand.
- [ ] **Parity:** normalized output matches Python on the golden corpus.

**Exit:** Stage 1 at parity; first column of the pipeline is Rust.

---

## Phase 4 — Inference core: the 4 subtasks (`popo-infer`)

**Goal:** Port the heart of the system — chunking, the four subtasks,
synchronization, and output parsing — sharing the Phase-2 model client.

- [ ] **Dynamic chunking** (`adaptive_chunk` parity: ~50 items/chunk, overlap=1,
      boundary-aware).
- [ ] Subtask 1 **text-truncation** (`filter_contd`/`merge_rules` heuristics +
      prompt + parse).
- [ ] Subtask 2 **title-hierarchy** + **cross-chunk synchronization** (bias
      averaging).
- [ ] Subtask 3 **image-text association** (`check_overlap` + prompt + parse).
- [ ] Subtask 4 **table-merge** (6-check pre-filter in `popo-tree`/utils +
      VLM decision + cell-coordinate parse).
- [ ] Strict + lenient output parsers; `resume` + `dry-run`.
- [ ] Document/chunk/subtask parallelism with the limits from NFR-2.
- [ ] `popo infer` CLI subcommand.
- [ ] **Parity:** annotated blocks match Python on the golden corpus.

**Exit:** Stage 2 at parity; the throughput SLO becomes measurable end-to-end on
stages 1–2.

---

## Phase 5 — Tree build + table merge (`popo-tree`)

**Goal:** Port Stage 3 including cross-page table merging and HTML table utils.

- [ ] Text-component assembly by title level; visual/special element attachment;
      page-supplement nodes (`get_json_tree.py` parity).
- [ ] HTML table parsing + occupied-matrix/colspan/rowspan utilities
      (`table_utils` parity) via the chosen Rust HTML parser.
- [ ] Cross-page `merge_cross_page_tables` parity.
- [ ] JSON tree + indented text-preview emit.
- [ ] `popo build-tree` CLI subcommand.
- [ ] **Parity:** tree JSON + txt preview match Python on the golden corpus.

**Exit:** Stages 1–3 are Rust; a document goes OCR-JSON → tree without Python.

---

## Phase 6 — Enrichment + evaluation (`popo-enrich`, `popo-eval`)

**Goal:** Complete the user-facing pipeline and the scoring tool.

- [ ] Metadata generation (`generate_metadata.py` parity) over the tree, using
      the model client.
- [ ] Subnode splitting by `<|txt_split|>`/`<|txt_contd|>` (500-char rule).
- [ ] Native **TEDS**: tree-edit-distance (replace `zss`) + GT alignment
      (box 45% / text 35% / type 20%).
- [ ] `popo enrich` and `popo eval` subcommands.
- [ ] **Parity:** enrichment output + TEDS scores match Python within tolerance.

**Exit:** All five stages at parity; `popo run` chains them end-to-end.

---

## Phase 7 — Data-engine parity & code unification (`popo-data-engine`)

**Goal:** Retire the last Python and remove the duplicate subtask codepath.

- [ ] Port `add_link.py` / `api_utils.py` training-data generation.
- [ ] **Unify**: the data-engine and `popo-infer` share one subtask
      implementation with two model backends (no `inference.py` vs `add_link.py`
      drift).
- [ ] **Parity:** generated training cases match Python.

**Exit:** Nothing in the default path requires Python.

---

## Phase 8 — Hardening, cutover & Python retirement

**Goal:** Make Rust the default and delete Python.

- [ ] Dual-run soak in CI/production; drive divergence to zero.
- [ ] Performance pass: confirm ≥ 5× throughput SLO via the `criterion` suite +
      flamegraphs; tune concurrency limits.
- [ ] Single-binary release (Linux x86_64 + arm64) + slim container image;
      health + `/metrics` endpoints for service mode.
- [ ] Docs: update `README`, migration guide, config mapping from `POPO_*`.
- [ ] **Delete the Python implementation** from the default path.

**Exit:** PRD §12 success criteria met; Python removed.

---

## Optional Track — Native in-process inference

Runs in parallel after Phase 2; **not** on the launch critical path. Behind the
`ModelBackend` trait so it never blocks the pipeline.

- [ ] Evaluate `candle` vs. `ort`/ONNX vs. `llama.cpp` bindings for serving the
      Popo / Qwen3-VL VLM in-process.
- [ ] Prototype + benchmark vs. the HTTP endpoint (latency, throughput, VRAM).
- [ ] Ship as an alternative backend if it wins on TCO/latency.

---

## Sequencing at a glance

```
Phase 0  contract + parity harness      ──┐
Phase 1  observability spine              │ foundations
Phase 2  model client + concurrency     ──┘
Phase 3  normalize        (Stage 1)  ─ parity gate
Phase 4  infer / 4 tasks  (Stage 2)  ─ parity gate  ◄ throughput SLO measurable
Phase 5  tree + tables    (Stage 3)  ─ parity gate
Phase 6  enrich + eval    (Stage 4-5)─ parity gate
Phase 7  data-engine + unify         ─ parity gate
Phase 8  cutover + delete Python     ─ launch
         (Optional: native inference, parallel from Phase 2)
```

## Cross-cutting definition of done (every phase)

- Parity test green on the golden corpus (byte or normalized compare).
- `tracing` spans + metrics emitted for the new code.
- Unit tests + (for parsers/heuristics) fuzz tests.
- A `criterion` benchmark for any performance-sensitive path.
- Docs/config updated; PR opened against `dev`.
