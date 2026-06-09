//! `popo eval` — title-hierarchy TEDS evaluation.
//!
//! Port of `eval/evaluate.py`'s flow: read the ground-truth title JSON, run the
//! reader over each document, align predicted titles to GT, and score the
//! `content_aware` TEDS. Writes `summary.json` + `details.json`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use popo_eval::{
    build_prediction_nodes, content_aware_nodes, parse_title_labels, parse_title_prompt,
    title_teds_score, GtTitleBlock, PredBlock,
};
use popo_readers::build_reader;
use serde_json::{json, Value};

/// Arguments for the `eval` subcommand.
#[derive(Args)]
pub struct EvalArgs {
    /// OCR model name (selects the reader).
    #[arg(long)]
    pub model: String,
    /// Per-model input directory (`<doc_id>/…`).
    #[arg(long)]
    pub input_dir: PathBuf,
    /// Ground-truth title JSON (benchmark items).
    #[arg(long)]
    pub gt_json: PathBuf,
    /// Output directory for `summary.json` / `details.json`.
    #[arg(long)]
    pub output_dir: PathBuf,
    /// Limit the number of documents (`0` = all).
    #[arg(long, default_value_t = 0)]
    pub doc_limit: usize,
    /// Minimum alignment score for a GT↔prediction match.
    #[arg(long, default_value_t = 0.35)]
    pub min_match_score: f64,
}

/// Execute the eval command.
pub fn run(args: &EvalArgs) -> Result<()> {
    let reader = build_reader(&args.model, &args.input_dir)
        .with_context(|| format!("building reader for {}", args.model))?;

    let text = std::fs::read_to_string(&args.gt_json)
        .with_context(|| format!("reading {}", args.gt_json.display()))?;
    let mut items: Vec<Value> = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", args.gt_json.display()))?;
    if args.doc_limit > 0 {
        items.truncate(args.doc_limit);
    }

    let mut details = Vec::new();
    for item in &items {
        details.push(evaluate_one(item, reader.as_ref(), args.min_match_score));
    }

    let summary = summarize(&details, &args.model);
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;
    std::fs::write(
        args.output_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    std::fs::write(
        args.output_dir.join("details.json"),
        serde_json::to_string_pretty(&Value::Array(details))?,
    )?;

    println!(
        "eval: model={} usable={}/{} score={:.4}",
        args.model,
        summary["usable_docs"].as_u64().unwrap_or(0),
        summary["total_docs"].as_u64().unwrap_or(0),
        summary["avg_score_with_missing_zero"]
            .as_f64()
            .unwrap_or(0.0),
    );
    Ok(())
}

fn conv_value(item: &Value, idx: usize) -> String {
    item.get("conversations")
        .and_then(Value::as_array)
        .and_then(|c| c.get(idx))
        .and_then(|c| c.get("value"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn evaluate_one(item: &Value, reader: &dyn popo_readers::OcrReader, min_match: f64) -> Value {
    let image = item.get("image").and_then(Value::as_str).unwrap_or("");
    let doc_id = Path::new(image)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(image)
        .to_string();

    let gt_blocks = parse_title_prompt(&conv_value(item, 0));
    let gt_block_by_id: HashMap<String, GtTitleBlock> = gt_blocks
        .iter()
        .map(|b| (b.block_id.clone(), b.clone()))
        .collect();
    let gt_nodes_raw = parse_title_labels(&conv_value(item, 1));

    let result = match reader.read_doc(&doc_id) {
        Ok(r) => r,
        Err(e) => {
            return json!({ "doc_id": doc_id, "status": "error", "message": e.to_string(), "score": 0.0 })
        }
    };
    if result.status != "ok" {
        return json!({
            "doc_id": doc_id, "status": result.status, "message": result.message,
            "score": 0.0, "gt_count": gt_nodes_raw.len(), "pred_count": 0, "match_count": 0,
        });
    }

    let preds: Vec<PredBlock> = result
        .blocks
        .iter()
        .filter(|b| b.kind == "title")
        .map(|b| PredBlock {
            block_id: b.block_id.clone(),
            page: b.page,
            bbox: b.bbox,
            kind: b.kind.clone(),
            content: b.content.clone(),
            title_level: b.title_level,
        })
        .collect();

    let (pred_nodes_raw, matches) = build_prediction_nodes(&preds, &gt_blocks, min_match);
    let gt_nodes = content_aware_nodes(&gt_nodes_raw, &gt_block_by_id);
    let pred_nodes = content_aware_nodes(&pred_nodes_raw, &gt_block_by_id);
    let score = title_teds_score(&gt_nodes, &pred_nodes);

    json!({
        "doc_id": doc_id,
        "status": "ok",
        "score": (score * 1_000_000.0).round() / 1_000_000.0,
        "gt_count": gt_nodes_raw.len(),
        "pred_count": pred_nodes_raw.len(),
        "match_count": matches.len(),
        "candidate_count": gt_blocks.len(),
    })
}

fn summarize(details: &[Value], model: &str) -> Value {
    let usable: Vec<&Value> = details.iter().filter(|d| d["status"] == "ok").collect();
    let mean = |vals: &[f64]| -> f64 {
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };
    let usable_scores: Vec<f64> = usable
        .iter()
        .map(|d| d["score"].as_f64().unwrap_or(0.0))
        .collect();
    let all_scores: Vec<f64> = details
        .iter()
        .map(|d| {
            if d["status"] == "ok" {
                d["score"].as_f64().unwrap_or(0.0)
            } else {
                0.0
            }
        })
        .collect();

    json!({
        "model_name": model,
        "mode": "content_aware",
        "total_docs": details.len(),
        "usable_docs": usable.len(),
        "missing_docs": details.len() - usable.len(),
        "avg_score_usable": mean(&usable_scores),
        "avg_score_with_missing_zero": mean(&all_scores),
    })
}
