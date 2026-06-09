//! `popo normalize` — adapt OCR/layout outputs into the canonical block schema.
//!
//! Mirrors the reference `run_label_normalization.sh`: read per-document model
//! output from `<input-dir>/<doc_id>/...` and write
//! `<output-dir>/<model>/<doc_id>.json` as `{ "input_label", "pages" }`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use popo_core::to_popo_pages;
use popo_readers::build_reader;
use serde_json::{json, Value};

/// Arguments for the `normalize` subcommand.
#[derive(Args)]
pub struct NormalizeArgs {
    /// OCR model name (e.g. `mineru`, `monkeyocr`).
    #[arg(long)]
    pub model: String,
    /// Directory holding `<doc_id>/...` inputs for this model.
    #[arg(long)]
    pub input_dir: PathBuf,
    /// Output root; results are written under `<output-dir>/<model>/`.
    #[arg(long)]
    pub output_dir: PathBuf,
    /// Limit the number of documents processed (`0` = all).
    #[arg(long, default_value_t = 0)]
    pub doc_limit: usize,
}

/// Execute the normalize command.
pub fn run(args: &NormalizeArgs) -> Result<()> {
    let reader = build_reader(&args.model, &args.input_dir)
        .with_context(|| format!("building reader for model {:?}", args.model))?;

    let mut doc_ids = discover_doc_ids(&args.input_dir)
        .with_context(|| format!("listing documents in {}", args.input_dir.display()))?;
    if args.doc_limit > 0 {
        doc_ids.truncate(args.doc_limit);
    }

    let out_dir = args.output_dir.join(&args.model);
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let mut written = 0usize;
    let mut missing = 0usize;
    for doc_id in &doc_ids {
        let result = reader
            .read_doc(doc_id)
            .with_context(|| format!("reading document {doc_id}"))?;
        if result.status != "ok" {
            tracing::warn!(doc_id, message = %result.message, "skipping document");
            missing += 1;
            continue;
        }
        let payload: Value = json!({
            "input_label": doc_id,
            "pages": to_popo_pages(&result.blocks),
        });
        let path = out_dir.join(format!("{doc_id}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&payload)?)
            .with_context(|| format!("writing {}", path.display()))?;
        written += 1;
    }

    tracing::info!(
        model = %args.model,
        documents = doc_ids.len(),
        written,
        missing,
        output = %out_dir.display(),
        "normalize complete"
    );
    println!(
        "normalize: model={} documents={} written={} missing={} -> {}",
        args.model,
        doc_ids.len(),
        written,
        missing,
        out_dir.display()
    );
    Ok(())
}

/// Each immediate subdirectory of `input_dir` is one document id, sorted.
fn discover_doc_ids(input_dir: &std::path::Path) -> std::io::Result<Vec<String>> {
    let mut ids = Vec::new();
    for entry in std::fs::read_dir(input_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                ids.push(name.to_string());
            }
        }
    }
    ids.sort();
    Ok(ids)
}
