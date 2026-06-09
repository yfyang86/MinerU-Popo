//! `popo infer` — run the four post-processing subtasks over normalized docs.
//!
//! Reads per-document normalized input (`{input_label, pages}` or a bare
//! `pages` map), runs the subtasks through the model client, and writes the
//! `doc_blocks` JSON the tree builder consumes. Page images are not yet wired
//! (the PDF stage is pending), so prompts are text-only via `NoImages`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use popo_infer::{build_doc_blocks, doc_blocks_to_json, run_inference, NoImages};
use popo_model::{ClientOptions, LlmConfig, ModelClient, ReplayBackend};
use serde_json::Value;
use std::sync::Arc;

/// Arguments for the `infer` subcommand.
#[derive(Args)]
pub struct InferArgs {
    /// Directory of normalized `<doc>.json` files (the `normalize` output).
    #[arg(long)]
    pub input_dir: PathBuf,
    /// Directory for the inference `doc_blocks` output.
    #[arg(long)]
    pub output_dir: PathBuf,
    /// Provider name; falls back to `[default].provider`.
    #[arg(long, short)]
    pub provider: Option<String>,
    /// Limit the number of documents processed (`0` = all).
    #[arg(long, default_value_t = 0)]
    pub doc_limit: usize,
}

/// Execute the infer command.
pub async fn run(cli_config: &std::path::Path, args: &InferArgs) -> Result<()> {
    let client = build_client(cli_config, args.provider.as_deref())?;

    let mut docs = list_json(&args.input_dir)
        .with_context(|| format!("listing {}", args.input_dir.display()))?;
    if args.doc_limit > 0 {
        docs.truncate(args.doc_limit);
    }
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;

    let mut written = 0usize;
    for path in &docs {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("doc");
        let (input_label, pages) =
            load_normalized(path).with_context(|| format!("reading {}", path.display()))?;
        let mut blocks = build_doc_blocks(&pages);

        tracing::info!(doc = %input_label, blocks = blocks.len(), "running inference");
        run_inference(&client, &mut blocks, &NoImages)
            .await
            .with_context(|| format!("inference failed for {input_label}"))?;

        let out_path = args.output_dir.join(format!("{stem}.json"));
        std::fs::write(
            &out_path,
            serde_json::to_string_pretty(&doc_blocks_to_json(&blocks))?,
        )
        .with_context(|| format!("writing {}", out_path.display()))?;
        written += 1;
    }

    println!(
        "infer: documents={} written={} -> {}",
        docs.len(),
        written,
        args.output_dir.display()
    );
    Ok(())
}

/// Build the model client, honoring `POPO_MODEL_REPLAY` for hermetic runs.
fn build_client(config_path: &std::path::Path, provider: Option<&str>) -> Result<ModelClient> {
    if let Ok(dir) = std::env::var("POPO_MODEL_REPLAY") {
        let backend = ReplayBackend::from_dir(&dir)
            .with_context(|| format!("loading replay fixtures from {dir}"))?;
        return Ok(ModelClient::from_backend(
            "replay",
            Arc::new(backend),
            ClientOptions::default(),
        ));
    }
    let cfg = LlmConfig::from_path(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    ModelClient::from_config(&cfg, provider).context("building model client")
}

/// Sorted list of `*.json` files in a directory.
fn list_json(dir: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();
    Ok(paths)
}

/// Load a normalized document as `(input_label, pages_map)`. Accepts either the
/// `{input_label, pages}` envelope or a bare `pages` object.
fn load_normalized(path: &std::path::Path) -> Result<(String, serde_json::Map<String, Value>)> {
    let text = std::fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&text)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("doc")
        .to_string();
    if let Some(pages) = value.get("pages").and_then(Value::as_object) {
        let label = value
            .get("input_label")
            .and_then(Value::as_str)
            .unwrap_or(&stem)
            .to_string();
        Ok((label, pages.clone()))
    } else if let Some(map) = value.as_object() {
        Ok((stem, map.clone()))
    } else {
        anyhow::bail!("unexpected normalized JSON shape in {}", path.display())
    }
}
