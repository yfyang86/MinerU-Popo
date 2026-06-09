//! GLM-OCR reader.
//!
//! Supports the `<doc>_model.json` path (block layout) and the merged
//! `glm_ocr.json` / `page_*.json` path (`words_result` lines). Image-derived
//! page sizing (PIL) is out of scope; sizes come from payload fields when
//! present, else bbox normalization falls back.

use std::path::PathBuf;

use popo_core::{normalize_bbox_to_unit, normalize_text, Error, NormalizedBlock, Result};
use serde_json::{Map, Value};

use crate::common::{extract_block_content, iter_model_pages, make_block, read_bbox};
use crate::{OcrReader, ReaderResult};

const MODEL_NAME: &str = "glm-ocr";

/// Map a GLM label to `(canonical_type, popo_type)`. Port of `map_glm_label`.
pub fn map_glm_label(label: &str) -> (&'static str, &'static str) {
    match label {
        "doc_title" | "paragraph_title" => ("title", "title"),
        "image" | "chart" | "seal" => ("image", "image"),
        "table" => ("table", "table"),
        "figure_title" => ("caption", "image_caption"),
        "vision_footnote" => ("caption", "image_footnote"),
        "display_formula" | "inline_formula" | "formula" => ("text", "equation"),
        _ => ("text", "text"),
    }
}

/// Reader for GLM-OCR output.
pub struct GlmOcrReader {
    root: PathBuf,
}

impl GlmOcrReader {
    /// Build a GLM-OCR reader rooted at the per-model input directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn load_model_json(&self, doc_id: &str, data: &Value) -> Vec<NormalizedBlock> {
        let mut blocks = Vec::new();
        let mut order = 0i64;
        for (page_index, items) in iter_model_pages(data) {
            for item in &items {
                let Some(obj) = item.as_object() else {
                    continue;
                };
                let label = obj
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("text")
                    .to_string();
                let (canonical, popo) = map_glm_label(&label);
                let content = extract_block_content(obj);
                if content.is_empty() && matches!(canonical, "text" | "title" | "caption") {
                    continue;
                }
                let bbox_src = obj.get("bbox_2d").or_else(|| obj.get("bbox"));
                let bbox = normalize_bbox_to_unit(&read_bbox(bbox_src), None, None, Some(1000.0));
                let mut meta = Map::new();
                meta.insert("source".into(), Value::String("model_json".into()));
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
                    meta,
                ));
                order += 1;
            }
        }
        blocks
    }

    fn load_pages(&self, doc_id: &str, pages: &[Value]) -> Vec<NormalizedBlock> {
        let mut blocks = Vec::new();
        let mut order = 0i64;
        for (page0, page) in pages.iter().enumerate() {
            let page_index = page0 as i64 + 1;
            let (page_width, page_height) = fallback_page_size(page);
            let response = page.get("response").unwrap_or(page);
            let words = response.get("words_result").and_then(Value::as_array);
            let Some(words) = words else { continue };
            for item in words {
                let content =
                    normalize_text(item.get("words").and_then(Value::as_str).unwrap_or(""));
                if content.is_empty() {
                    continue;
                }
                let loc = item.get("location");
                let left = loc
                    .and_then(|l| l.get("left"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let top = loc
                    .and_then(|l| l.get("top"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let width = loc
                    .and_then(|l| l.get("width"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let height = loc
                    .and_then(|l| l.get("height"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let bbox = normalize_bbox_to_unit(
                    &[left, top, left + width, top + height],
                    page_width,
                    page_height,
                    None,
                );
                blocks.push(make_block(
                    doc_id,
                    order,
                    page_index,
                    bbox,
                    "text",
                    &content,
                    "text",
                    None,
                    Some("words_result".into()),
                    Map::new(),
                ));
                order += 1;
            }
        }
        blocks
    }
}

/// Resolve a page's `(width, height)` from common payload keys (Python
/// `_load_fallback_page_size`, minus the PIL image fallback).
fn fallback_page_size(page: &Value) -> (Option<f64>, Option<f64>) {
    let response = page.get("response").unwrap_or(page);
    for holder in [page, response] {
        for (wk, hk) in [
            ("width", "height"),
            ("image_width", "image_height"),
            ("page_width", "page_height"),
        ] {
            let w = holder.get(wk).and_then(Value::as_f64);
            let h = holder.get(hk).and_then(Value::as_f64);
            if let (Some(w), Some(h)) = (w, h) {
                if w > 0.0 && h > 0.0 {
                    return (Some(w), Some(h));
                }
            }
        }
    }
    (None, None)
}

impl OcrReader for GlmOcrReader {
    fn model_name(&self) -> &str {
        MODEL_NAME
    }

    fn read_doc(&self, doc_id: &str) -> Result<ReaderResult> {
        let doc_root = self.root.join(doc_id);
        let model_path = doc_root.join(format!("{doc_id}_model.json"));
        let merged_path = doc_root.join("glm_ocr.json");

        if model_path.exists() {
            let data = load_json(&model_path)?;
            return Ok(ReaderResult::ok(
                MODEL_NAME,
                doc_id,
                self.load_model_json(doc_id, &data),
            ));
        }
        if merged_path.exists() {
            let data = load_json(&merged_path)?;
            if let Some(pages) = data.get("pages").and_then(Value::as_array) {
                if !pages.is_empty() {
                    return Ok(ReaderResult::ok(
                        MODEL_NAME,
                        doc_id,
                        self.load_pages(doc_id, pages),
                    ));
                }
            }
        }
        // `page_*.json` shards are discovered and concatenated.
        let mut page_paths: Vec<PathBuf> = std::fs::read_dir(&doc_root)
            .map(|rd| {
                rd.filter_map(std::result::Result::ok)
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.starts_with("page_") && n.ends_with(".json"))
                            .unwrap_or(false)
                    })
                    .collect()
            })
            .unwrap_or_default();
        page_paths.sort();
        if !page_paths.is_empty() {
            let pages = page_paths
                .iter()
                .map(|p| load_json(p))
                .collect::<Result<Vec<_>>>()?;
            return Ok(ReaderResult::ok(
                MODEL_NAME,
                doc_id,
                self.load_pages(doc_id, &pages),
            ));
        }
        Ok(ReaderResult::missing(
            MODEL_NAME,
            doc_id,
            format!("missing GLM OCR output under {}", doc_root.display()),
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
        assert_eq!(map_glm_label("paragraph_title"), ("title", "title"));
        assert_eq!(map_glm_label("seal"), ("image", "image"));
        assert_eq!(map_glm_label("formula"), ("text", "equation"));
    }

    #[test]
    fn model_json_uses_bbox_2d_with_assumed_scale() {
        let reader = GlmOcrReader::new("/unused");
        let data = json!([
            [
                { "label": "doc_title", "content": "Title", "bbox_2d": [100, 100, 900, 200] },
                { "label": "text", "content": "para", "bbox": [100, 300, 900, 400] }
            ]
        ]);
        let blocks = reader.load_model_json("doc", &data);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].kind, "title");
        assert_eq!(blocks[0].bbox, [0.1, 0.1, 0.9, 0.2]);
    }

    #[test]
    fn pages_words_result_builds_bbox_from_location() {
        let reader = GlmOcrReader::new("/unused");
        let pages = vec![json!({
            "width": 1000.0, "height": 2000.0,
            "response": { "words_result": [
                { "words": "hello", "location": { "left": 100, "top": 200, "width": 300, "height": 100 } },
                { "words": "  ", "location": { "left": 0, "top": 0, "width": 1, "height": 1 } }
            ] }
        })];
        let blocks = reader.load_pages("doc", &pages);
        assert_eq!(blocks.len(), 1); // blank words skipped
        assert_eq!(blocks[0].content, "hello");
        assert_eq!(blocks[0].bbox, [0.1, 0.1, 0.4, 0.15]);
    }
}
