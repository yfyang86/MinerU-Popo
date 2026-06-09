//! `popo build-tree` — assemble document trees from inference `doc_blocks`.
//!
//! Port of `get_json_tree.py`'s `main`: read each inference output JSON, build
//! the tree, and write the tree JSON plus the indented text preview.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use serde_json::Value;

/// Arguments for the `build-tree` subcommand.
#[derive(Args)]
pub struct BuildTreeArgs {
    /// Directory of inference `doc_blocks` JSON files (the `infer` output).
    #[arg(long)]
    pub input_dir: PathBuf,
    /// Directory for the document-tree JSON output.
    #[arg(long)]
    pub output_dir: PathBuf,
    /// Directory for the text-preview output.
    #[arg(long)]
    pub txt_dir: PathBuf,
}

/// Execute the build-tree command.
pub fn run(args: &BuildTreeArgs) -> Result<()> {
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;
    std::fs::create_dir_all(&args.txt_dir)
        .with_context(|| format!("creating {}", args.txt_dir.display()))?;

    let mut files: Vec<PathBuf> = std::fs::read_dir(&args.input_dir)
        .with_context(|| format!("listing {}", args.input_dir.display()))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    files.sort();

    let mut built = 0usize;
    for path in &files {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("doc");
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let value: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let blocks = value
            .as_array()
            .cloned()
            .with_context(|| format!("expected a doc_blocks array in {}", path.display()))?;

        let tree = popo_tree::build_tree(blocks);
        let txt = popo_tree::tree_to_txt(&tree);

        std::fs::write(
            args.output_dir.join(format!("{stem}.json")),
            serde_json::to_string_pretty(&tree)?,
        )
        .with_context(|| format!("writing tree json for {stem}"))?;
        std::fs::write(args.txt_dir.join(format!("{stem}.txt")), txt)
            .with_context(|| format!("writing tree txt for {stem}"))?;
        built += 1;
    }

    println!(
        "build-tree: documents={} built={} -> {} (txt: {})",
        files.len(),
        built,
        args.output_dir.display(),
        args.txt_dir.display()
    );
    Ok(())
}
