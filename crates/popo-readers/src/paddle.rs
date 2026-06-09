//! PaddleOCR-VL reader.
//!
//! Supports the self-contained `layout_parsing.json` path (page width/height
//! come from `prunedResult`). The per-page `*_res.json` path needs source-PDF
//! page sizes and is deferred until the PDF stage exists.

use std::path::PathBuf;

use popo_core::{normalize_bbox_to_unit, normalize_text, Error, Result};
use serde_json::{Map, Value};

use crate::common::{make_block, read_bbox, sort_by_optional_int, SKIP_TYPE};
use crate::{OcrReader, ReaderResult};

const MODEL_NAME: &str = "PaddleOCR-VL-1.5";

/// Map a Paddle `block_label` to `(canonical_type, popo_type)`. Port of
/// `map_paddle_label`.
pub fn map_paddle_label(label: &str) -> (&'static str, &'static str) {
    match label {
        "paragraph_title" | "doc_title" => ("title", "title"),
        "page_title" => ("text", "page_title"),
        "page_number" | "number" => ("text", "page_number"),
        "page_footnote" | "footnote" => ("text", "page_footnote"),
        "header" => ("text", "header"),
        "aside_text" => ("text", "aside_text"),
        "footer" => ("text", "footer"),
        "image" | "chart" | "footer_image" | "header_image" | "seal" => ("image", "image"),
        "table" => ("table", "table"),
        "figure_title" => ("caption", "image_caption"),
        "vision_footnote" => ("caption", "image_footnote"),
        "inline_formula" | "display_formula" => ("text", "equation"),
        _ => ("text", "text"),
    }
}

/// Reader for PaddleOCR-VL output.
pub struct PaddleReader {
    root: PathBuf,
}

impl PaddleReader {
    /// Build a Paddle reader rooted at the per-model input directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn read_layout_parsing(&self, doc_id: &str, data: &Value) -> ReaderResult {
        let mut blocks = Vec::new();
        let mut order = 0i64;
        let pages = data
            .get("result")
            .and_then(|r| r.get("layoutParsingResults"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for (page0, page) in pages.iter().enumerate() {
            let page_index = page0 as i64 + 1;
            let pruned = page.get("prunedResult");
            let page_width = pruned.and_then(|p| p.get("width")).and_then(Value::as_f64);
            let page_height = pruned.and_then(|p| p.get("height")).and_then(Value::as_f64);
            let mut items = pruned
                .and_then(|p| p.get("parsing_res_list"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            sort_by_optional_int(&mut items, "block_order");
            for item in &items {
                let Some(obj) = item.as_object() else {
                    continue;
                };
                let label = obj
                    .get("block_label")
                    .and_then(Value::as_str)
                    .unwrap_or("text")
                    .to_string();
                let (canonical, popo) = map_paddle_label(&label);
                if popo == SKIP_TYPE {
                    continue;
                }
                let content = normalize_text(
                    obj.get("block_content")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
                if content.is_empty() && matches!(canonical, "text" | "title" | "caption") {
                    continue;
                }
                let bbox = normalize_bbox_to_unit(
                    &read_bbox(obj.get("block_bbox")),
                    page_width,
                    page_height,
                    None,
                );
                blocks.push(make_block(
                    doc_id,
                    order,
                    page_index,
                    bbox,
                    canonical,
                    &content,
                    popo,
                    None,
                    Some(label),
                    Map::new(),
                ));
                order += 1;
            }
        }
        ReaderResult::ok(MODEL_NAME, doc_id, blocks)
    }
}

impl OcrReader for PaddleReader {
    fn model_name(&self) -> &str {
        MODEL_NAME
    }

    fn read_doc(&self, doc_id: &str) -> Result<ReaderResult> {
        let doc_root = self.root.join(doc_id);
        let json_path = doc_root.join("layout_parsing.json");
        if json_path.exists() {
            let data = load_json(&json_path)?;
            return Ok(self.read_layout_parsing(doc_id, &data));
        }
        // The per-page *_res.json path requires source-PDF page sizes.
        Ok(ReaderResult::missing(
            MODEL_NAME,
            doc_id,
            format!(
                "missing {} (per-page *_res.json path needs the PDF stage, not yet ported)",
                json_path.display()
            ),
        ))
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
    fn label_mapping() {
        assert_eq!(map_paddle_label("doc_title"), ("title", "title"));
        assert_eq!(map_paddle_label("chart"), ("image", "image"));
        assert_eq!(
            map_paddle_label("figure_title"),
            ("caption", "image_caption")
        );
        assert_eq!(map_paddle_label("display_formula"), ("text", "equation"));
    }

    #[test]
    fn layout_parsing_orders_by_block_order_and_normalizes() {
        let reader = PaddleReader::new("/unused");
        let data = json!({
            "result": { "layoutParsingResults": [
                { "prunedResult": {
                    "width": 1000.0, "height": 2000.0,
                    "parsing_res_list": [
                        { "block_label": "text", "block_content": "second", "block_bbox": [0,1000,1000,1200], "block_order": 2 },
                        { "block_label": "doc_title", "block_content": "first", "block_bbox": [0,0,1000,100], "block_order": 1 }
                    ]
                } }
            ] }
        });
        let res = reader.read_layout_parsing("doc", &data);
        assert_eq!(res.blocks.len(), 2);
        // block_order sorts the title (order 1) first.
        assert_eq!(res.blocks[0].kind, "title");
        assert_eq!(res.blocks[0].content, "first");
        assert_eq!(res.blocks[0].bbox, [0.0, 0.0, 1.0, 0.05]);
        assert_eq!(res.blocks[1].content, "second");
    }
}
