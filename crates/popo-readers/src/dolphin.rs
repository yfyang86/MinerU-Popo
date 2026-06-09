//! Dolphin reader.
//!
//! Reads `<doc>/recognition_json/<doc>.json`. Source-PDF page sizes are not
//! available until the PDF stage is ported, so bbox normalization falls back to
//! the unit/identity behavior (matching Python when page sizes are missing).

use std::path::PathBuf;

use popo_core::{normalize_bbox_to_unit, normalize_text, Error, Result};
use serde_json::{Map, Value};

use crate::common::{make_block, read_bbox, sort_by_optional_int};
use crate::{OcrReader, ReaderResult};

const MODEL_NAME: &str = "dolphin";

/// Map a Dolphin element to `(canonical_type, popo_type, title_level)`. Port of
/// `map_dolphin_label`, including the `sec_<n>` heading-level convention.
pub fn map_dolphin_label(label: &str) -> (&'static str, &'static str, Option<i64>) {
    if let Some(level) = parse_sec_level(label) {
        return ("title", "title", Some(level));
    }
    match label {
        "catalogue" => ("title", "title", Some(1)),
        "header" => ("text", "header", None),
        "foot" => ("text", "footer", None),
        "fnote" => ("text", "page_footnote", None),
        "fig" => ("image", "image", None),
        "tab" => ("table", "table", None),
        "cap" => ("caption", "image_caption", None),
        "list" => ("text", "list_item", None),
        "equ" => ("text", "equation", None),
        _ => ("text", "text", None),
    }
}

/// Parse `sec_<n>` where `<n>` is a positive integer with no leading zero
/// (mirrors the Python regex `sec_[1-9]\d*`).
fn parse_sec_level(label: &str) -> Option<i64> {
    let rest = label.strip_prefix("sec_")?;
    let mut chars = rest.chars();
    let first = chars.next()?;
    if !('1'..='9').contains(&first) {
        return None;
    }
    if !chars.all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse::<i64>().ok()
}

/// Reader for Dolphin output.
pub struct DolphinReader {
    root: PathBuf,
}

impl DolphinReader {
    /// Build a Dolphin reader rooted at the per-model input directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn read_data(&self, doc_id: &str, data: &Value) -> ReaderResult {
        let mut blocks = Vec::new();
        let mut order = 0i64;
        let pages = data
            .get("pages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for (fallback0, page) in pages.iter().enumerate() {
            let fallback_index = fallback0 as i64 + 1;
            let page_index = match page.get("page_number").and_then(Value::as_i64) {
                Some(n) if n > 0 => n,
                _ => fallback_index,
            };
            let mut items = page
                .get("elements")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            sort_by_optional_int(&mut items, "reading_order");
            for item in &items {
                let Some(obj) = item.as_object() else {
                    continue;
                };
                let label = obj
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let (canonical, popo, level) = map_dolphin_label(&label);
                let content = normalize_text(obj.get("text").and_then(Value::as_str).unwrap_or(""));
                if content.is_empty() && matches!(canonical, "text" | "title" | "caption") {
                    continue;
                }
                // Page sizes are unavailable without the PDF stage.
                let bbox = normalize_bbox_to_unit(&read_bbox(obj.get("bbox")), None, None, None);
                blocks.push(make_block(
                    doc_id,
                    order,
                    page_index,
                    bbox,
                    canonical,
                    &content,
                    popo,
                    level,
                    Some(label),
                    Map::new(),
                ));
                order += 1;
            }
        }
        ReaderResult::ok(MODEL_NAME, doc_id, blocks)
    }
}

impl OcrReader for DolphinReader {
    fn model_name(&self) -> &str {
        MODEL_NAME
    }

    fn read_doc(&self, doc_id: &str) -> Result<ReaderResult> {
        let json_path = self
            .root
            .join(doc_id)
            .join("recognition_json")
            .join(format!("{doc_id}.json"));
        if !json_path.exists() {
            return Ok(ReaderResult::missing(
                MODEL_NAME,
                doc_id,
                format!("missing {}", json_path.display()),
            ));
        }
        let data = load_json(&json_path)?;
        Ok(self.read_data(doc_id, &data))
    }
}

fn load_json(path: &std::path::Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Io(format!("reading {}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Io(format!("parsing {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sec_levels_and_labels() {
        assert_eq!(map_dolphin_label("sec_1"), ("title", "title", Some(1)));
        assert_eq!(map_dolphin_label("sec_12"), ("title", "title", Some(12)));
        assert_eq!(map_dolphin_label("sec_0"), ("text", "text", None)); // no leading zero
        assert_eq!(map_dolphin_label("catalogue"), ("title", "title", Some(1)));
        assert_eq!(map_dolphin_label("tab"), ("table", "table", None));
        assert_eq!(map_dolphin_label("equ"), ("text", "equation", None));
    }

    #[test]
    fn reads_pages_with_reading_order_and_page_number() {
        let reader = DolphinReader::new("/unused");
        let data = json!({
            "pages": [
                {
                    "page_number": 3,
                    "elements": [
                        { "label": "text", "text": "b", "bbox": [0.0,0.5,1.0,0.6], "reading_order": 2 },
                        { "label": "sec_2", "text": "Heading", "bbox": [0.0,0.0,1.0,0.1], "reading_order": 1 }
                    ]
                }
            ]
        });
        let res = reader.read_data("doc", &data);
        assert_eq!(res.blocks.len(), 2);
        assert_eq!(res.blocks[0].kind, "title");
        assert_eq!(res.blocks[0].title_level, Some(2));
        assert_eq!(res.blocks[0].page, 3);
        assert_eq!(res.blocks[1].content, "b");
    }
}
