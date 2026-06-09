//! `popo split-subnode` — final tree chunking (port of `split_subnode.py`).
//!
//! Reads document trees, splits long text nodes into `sub_text` subnodes and
//! moves visual nodes' children under `subnode`, and writes the result.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use serde_json::Value;

/// Arguments for the `split-subnode` subcommand.
#[derive(Args)]
pub struct SplitSubnodeArgs {
    /// Directory of tree JSON files (the `build-tree`/enrich output).
    #[arg(long)]
    pub input_dir: PathBuf,
    /// Directory for the split trees.
    #[arg(long)]
    pub output_dir: PathBuf,
}

/// Execute the split-subnode command.
pub fn run(args: &SplitSubnodeArgs) -> Result<()> {
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;

    let mut files: Vec<PathBuf> = std::fs::read_dir(&args.input_dir)
        .with_context(|| format!("listing {}", args.input_dir.display()))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    files.sort();

    let mut written = 0usize;
    for path in &files {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("doc.json");
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let tree: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let split = popo_enrich::split_tree(tree);
        std::fs::write(
            args.output_dir.join(name),
            serde_json::to_string_pretty(&split)?,
        )
        .with_context(|| format!("writing {name}"))?;
        written += 1;
    }

    println!(
        "split-subnode: documents={} written={} -> {}",
        files.len(),
        written,
        args.output_dir.display()
    );
    Ok(())
}
