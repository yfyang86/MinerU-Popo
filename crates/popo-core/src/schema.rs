//! Canonical document schema shared across the pipeline.
//!
//! This is the Rust port of the `NormalizedBlock` contract from the reference
//! Python `label_normalization.py`. Readers produce [`NormalizedBlock`]s; the
//! inference stage consumes the "popo-pages" projection ([`to_popo_pages`]).
//! Bbox helpers and text normalization match the Python semantics exactly so
//! the two implementations can be compared on a golden corpus.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// An axis-aligned bounding box in `xyxy` order: `[left, top, right, bottom]`.
pub type Bbox = [f64; 4];

/// The five canonical block types. Anything else a reader sees collapses to
/// `text` (mirroring the Python `CANONICAL_TYPES` guard).
pub const CANONICAL_TYPES: [&str; 5] = ["title", "text", "image", "table", "caption"];

/// A normalized layout block: the unit of exchange between readers and the rest
/// of the pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedBlock {
    /// Stable id, `"{doc_id}:{order}"`.
    pub block_id: String,
    /// 1-based page index.
    pub page: i64,
    /// Bounding box, normalized to `0..1` when page size is known.
    pub bbox: Bbox,
    /// Canonical type (one of [`CANONICAL_TYPES`]).
    #[serde(rename = "type")]
    pub kind: String,
    /// Normalized text content.
    pub content: String,
    /// Reading order within the document.
    pub order: i64,
    /// Fine-grained "popo" type used by the inference prompts.
    pub popo_type: String,
    /// Heading level, when the block is a title.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title_level: Option<i64>,
    /// The raw source label from the originating OCR model.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub source_label: Option<String>,
    /// Free-form provenance metadata.
    #[serde(skip_serializing_if = "Map::is_empty", default)]
    pub meta: Map<String, Value>,
}

impl NormalizedBlock {
    /// Project to the "popo block" shape the inference stage reads:
    /// `{type, content, bbox, [title_level], [source_label], source_id}`.
    pub fn to_popo_block(&self) -> Value {
        let mut m = Map::new();
        m.insert("type".into(), json!(self.popo_type));
        m.insert("content".into(), json!(self.content));
        m.insert("bbox".into(), json!(self.bbox));
        if let Some(level) = self.title_level {
            m.insert("title_level".into(), json!(level));
        }
        if let Some(label) = &self.source_label {
            m.insert("source_label".into(), json!(label));
        }
        m.insert("source_id".into(), json!(self.block_id));
        Value::Object(m)
    }
}

/// Collapse whitespace the way the Python `normalize_text` does: replace the
/// ideographic space and newlines with spaces, fold runs of whitespace, trim.
pub fn normalize_text(text: &str) -> String {
    let replaced: String = text
        .chars()
        .map(|c| if c == '\u{3000}' || c == '\n' { ' ' } else { c })
        .collect();
    replaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Normalize a bbox to the unit square `0..1` when size information is
/// available, mirroring Python `normalize_bbox_to_unit`. Values already within
/// `±1.5` are assumed normalized and clamped; otherwise an `assumed_scale` or
/// page dimensions are used to divide.
pub fn normalize_bbox_to_unit(
    bbox: &[f64],
    page_width: Option<f64>,
    page_height: Option<f64>,
    assumed_scale: Option<f64>,
) -> Bbox {
    if bbox.len() < 4 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let v = [bbox[0], bbox[1], bbox[2], bbox[3]];
    let clamp = |items: [f64; 4]| items.map(|x| x.clamp(0.0, 1.0));

    let max_value = v.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    if max_value <= 1.5 {
        return clamp(v);
    }
    if let Some(scale) = assumed_scale {
        if scale > 0.0 {
            return clamp(v.map(|x| x / scale));
        }
    }
    let (w, h) = (page_width.unwrap_or(0.0), page_height.unwrap_or(0.0));
    if w > 0.0 && h > 0.0 {
        return clamp([v[0] / w, v[1] / h, v[2] / w, v[3] / h]);
    }
    v
}

/// Sort blocks into reading order: `(page, order, top, left)`.
pub fn sort_blocks(blocks: &mut [NormalizedBlock]) {
    blocks.sort_by(|a, b| {
        (a.page, a.order)
            .cmp(&(b.page, b.order))
            .then(a.bbox[1].total_cmp(&b.bbox[1]))
            .then(a.bbox[0].total_cmp(&b.bbox[0]))
    });
}

/// Reassign `block_id` and `order` sequentially after sorting, as the Python
/// `reassign_block_ids` does at the end of every reader.
pub fn reassign_block_ids(mut blocks: Vec<NormalizedBlock>, doc_id: &str) -> Vec<NormalizedBlock> {
    sort_blocks(&mut blocks);
    for (order, block) in blocks.iter_mut().enumerate() {
        block.block_id = format!("{doc_id}:{order}");
        block.order = order as i64;
    }
    blocks
}

/// Build the page-keyed "popo-pages" projection. Pages are emitted in numeric
/// order (blocks are sorted first), matching the Python `to_popo_pages`.
pub fn to_popo_pages(blocks: &[NormalizedBlock]) -> Map<String, Value> {
    let mut sorted = blocks.to_vec();
    sort_blocks(&mut sorted);
    let mut pages: Map<String, Value> = Map::new();
    for block in &sorted {
        let key = block.page.to_string();
        pages
            .entry(key)
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .expect("page entry is always an array")
            .push(block.to_popo_block());
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_text_folds_whitespace() {
        assert_eq!(normalize_text("  a\n b\u{3000}c  "), "a b c");
        assert_eq!(normalize_text("\t x\t\ty "), "x y");
    }

    #[test]
    fn bbox_unit_passthrough_and_clamp() {
        assert_eq!(
            normalize_bbox_to_unit(&[0.1, 0.2, 0.3, 0.4], None, None, None),
            [0.1, 0.2, 0.3, 0.4]
        );
        // Out-of-range unit values clamp.
        assert_eq!(
            normalize_bbox_to_unit(&[-0.1, 0.0, 1.2, 1.0], None, None, None),
            [0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn bbox_assumed_scale_division() {
        assert_eq!(
            normalize_bbox_to_unit(&[100.0, 250.0, 500.0, 750.0], None, None, Some(1000.0)),
            [0.1, 0.25, 0.5, 0.75]
        );
    }

    #[test]
    fn bbox_page_dimension_division() {
        assert_eq!(
            normalize_bbox_to_unit(
                &[100.0, 200.0, 200.0, 400.0],
                Some(200.0),
                Some(400.0),
                None
            ),
            [0.5, 0.5, 1.0, 1.0]
        );
    }

    #[test]
    fn popo_pages_orders_numerically_and_projects() {
        let mk = |page: i64, order: i64, kind: &str| NormalizedBlock {
            block_id: format!("d:{order}"),
            page,
            bbox: [0.0, 0.0, 1.0, 1.0],
            kind: kind.into(),
            content: "x".into(),
            order,
            popo_type: kind.into(),
            title_level: None,
            source_label: None,
            meta: Map::new(),
        };
        // Page 10 before page 2 by insertion, must come out numerically ordered.
        let blocks = vec![mk(10, 0, "text"), mk(2, 1, "title")];
        let pages = to_popo_pages(&blocks);
        let keys: Vec<&String> = pages.keys().collect();
        assert_eq!(keys, vec!["2", "10"]);
        assert_eq!(pages["2"][0]["type"], "title");
        assert_eq!(pages["2"][0]["source_id"], "d:1");
    }

    #[test]
    fn reassign_renumbers_in_reading_order() {
        let blocks = vec![
            NormalizedBlock {
                block_id: "old".into(),
                page: 2,
                bbox: [0.0, 0.0, 1.0, 1.0],
                kind: "text".into(),
                content: "b".into(),
                order: 99,
                popo_type: "text".into(),
                title_level: None,
                source_label: None,
                meta: Map::new(),
            },
            NormalizedBlock {
                block_id: "old".into(),
                page: 1,
                bbox: [0.0, 0.0, 1.0, 1.0],
                kind: "text".into(),
                content: "a".into(),
                order: 5,
                popo_type: "text".into(),
                title_level: None,
                source_label: None,
                meta: Map::new(),
            },
        ];
        let out = reassign_block_ids(blocks, "doc");
        assert_eq!(out[0].block_id, "doc:0");
        assert_eq!(out[0].content, "a"); // page 1 sorts first
        assert_eq!(out[1].block_id, "doc:1");
    }
}
