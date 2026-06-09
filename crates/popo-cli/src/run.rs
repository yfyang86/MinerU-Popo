//! `popo run` — end-to-end pipeline: normalize → infer → build-tree →
//! split-subnode, chaining the individual stage commands under one work dir.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::{build_tree, infer, normalize, split_subnode};

/// Arguments for the `run` subcommand.
#[derive(Args)]
pub struct RunArgs {
    /// OCR model name (selects the reader).
    #[arg(long)]
    pub model: String,
    /// Directory of the model's OCR/layout outputs (`<doc_id>/…`).
    #[arg(long)]
    pub input_dir: PathBuf,
    /// Working directory for all intermediate and final outputs.
    #[arg(long)]
    pub work_dir: PathBuf,
    /// Model provider; falls back to `[default].provider`.
    #[arg(long, short)]
    pub provider: Option<String>,
    /// Directory of source PDFs (`<doc>.pdf`) for VLM page images
    /// (needs `--features pdfium`).
    #[arg(long)]
    pub pdf_dir: Option<PathBuf>,
    /// Limit the number of documents (`0` = all).
    #[arg(long, default_value_t = 0)]
    pub doc_limit: usize,
}

/// Execute the full pipeline.
pub async fn run(config: &std::path::Path, args: &RunArgs) -> Result<()> {
    let normalized = args.work_dir.join("normalized");
    let inference = args.work_dir.join("inference");
    let tree = args.work_dir.join("tree");
    let tree_txt = args.work_dir.join("tree_txt");
    let final_dir = args.work_dir.join("final");

    tracing::info!(stage = "normalize", "starting");
    normalize::run(&normalize::NormalizeArgs {
        model: args.model.clone(),
        input_dir: args.input_dir.clone(),
        output_dir: normalized.clone(),
        doc_limit: args.doc_limit,
    })?;

    tracing::info!(stage = "infer", "starting");
    infer::run(
        config,
        &infer::InferArgs {
            input_dir: normalized.join(&args.model),
            output_dir: inference.clone(),
            provider: args.provider.clone(),
            pdf_dir: args.pdf_dir.clone(),
            doc_limit: args.doc_limit,
        },
    )
    .await?;

    tracing::info!(stage = "build-tree", "starting");
    build_tree::run(&build_tree::BuildTreeArgs {
        input_dir: inference.clone(),
        output_dir: tree.clone(),
        txt_dir: tree_txt,
    })?;

    tracing::info!(stage = "split-subnode", "starting");
    split_subnode::run(&split_subnode::SplitSubnodeArgs {
        input_dir: tree,
        output_dir: final_dir.clone(),
    })?;

    println!("run: pipeline complete -> {}", final_dir.display());
    Ok(())
}
