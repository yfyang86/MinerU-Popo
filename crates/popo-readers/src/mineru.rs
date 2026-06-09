//! MinerU and MonkeyOCR readers.
//!
//! MinerU emits one of `<doc>_model.json`, `<doc>_middle.json`, or
//! `<doc>_content_list.json` (tried in that order). MonkeyOCR reuses the
//! `middle.json` path. Port of `MineruReader` / `MonkeyOCRReader`.

use std::path::PathBuf;

use popo_core::{normalize_bbox_to_unit, Error, Result};
use serde_json::{Map, Value};

use crate::common::{extract_block_content, make_block, read_bbox, SKIP_TYPE};
use crate::{OcrReader, ReaderResult};

/// Supplement labels that map to a `text` canonical type but keep a specific
/// popo-type (Python `SUPPLEMENT_LABEL_MAP`).
fn supplement_label(label: &str) -> Option<&'static str> {
    match label {
        "page_title" => Some("page_title"),
        "page_number" | "number" => Some("page_number"),
        "page_footnote" | "footnote" => Some("page_footnote"),
        "header" => Some("header"),
        "aside_text" => Some("aside_text"),
        "footer" => Some("footer"),
        _ => None,
    }
}

/// Map a MinerU source label to `(canonical_type, popo_type)`. Port of
/// `map_mineru_label`.
pub fn map_mineru_label(label: &str) -> (&'static str, &'static str) {
    match label {
        "title" => ("title", "title"),
        "image" => ("image", "image"),
        "table" => ("table", "table"),
        "image_caption" => ("caption", "image_caption"),
        "table_caption" => ("caption", "table_caption"),
        "image_footnote" => ("caption", "image_footnote"),
        "table_footnote" => ("caption", "table_footnote"),
        "equation" | "interline_equation" | "inline_equation" => ("text", "equation"),
        "list" => ("text", "list_item"),
        "discarded" | "image_body" | "table_body" => (SKIP_TYPE, SKIP_TYPE),
        other => {
            if let Some(popo) = supplement_label(other) {
                ("text", popo)
            } else {
                ("text", "text")
            }
        }
    }
}

/// Reader for MinerU output.
pub struct MineruReader {
    root: PathBuf,
    /// Inner directory under `<root>/<doc_id>/` (MinerU uses `vlm`).
    inner_dir: &'static str,
    model_name: &'static str,
}

impl MineruReader {
    /// Build a MinerU reader rooted at the per-model input directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            inner_dir: "vlm",
            model_name: "mineru",
        }
    }

    fn doc_root(&self, doc_id: &str) -> PathBuf {
        let base = self.root.join(doc_id);
        if self.inner_dir.is_empty() {
            base
        } else {
            base.join(self.inner_dir)
        }
    }

    pub(crate) fn read_middle(&self, doc_id: &str, data: &Value) -> ReaderResult {
        let mut blocks = Vec::new();
        let mut order = 0i64;
        let empty = Vec::new();
        let pages = data
            .get("pdf_info")
            .and_then(Value::as_array)
            .unwrap_or(&empty);
        for page in pages {
            let page_index = page.get("page_idx").and_then(Value::as_i64).unwrap_or(0) + 1;
            let page_size = page.get("page_size").and_then(Value::as_array);
            let page_width = page_size.and_then(|s| s.first()).and_then(Value::as_f64);
            let page_height = page_size.and_then(|s| s.get(1)).and_then(Value::as_f64);
            let para_blocks = page
                .get("para_blocks")
                .and_then(Value::as_array)
                .unwrap_or(&empty);
            for item in para_blocks {
                let Some(obj) = item.as_object() else {
                    continue;
                };
                if let Some(block) = self.block_from_item(
                    doc_id,
                    order,
                    page_index,
                    obj,
                    page_width,
                    page_height,
                    None,
                ) {
                    blocks.push(block);
                    order += 1;
                }
            }
        }
        ReaderResult::ok(self.model_name, doc_id, blocks)
    }

    fn read_content_list(&self, doc_id: &str, items: &[Value]) -> ReaderResult {
        let mut blocks = Vec::new();
        let mut order = 0i64;
        for item in items {
            let Some(obj) = item.as_object() else {
                continue;
            };
            if obj.get("bbox").is_none() {
                continue;
            }
            let label = obj
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("text")
                .to_string();
            let (mut canonical, mut popo) = map_mineru_label(&label);
            if popo == SKIP_TYPE {
                continue;
            }
            // A `text_level` promotes the block to a title at that level.
            let mut level = None;
            if let Some(tl) = obj.get("text_level").and_then(Value::as_i64) {
                canonical = "title";
                popo = "title";
                level = Some(tl);
            }
            let content = extract_block_content(obj);
            if content.is_empty() && matches!(canonical, "text" | "title" | "caption") {
                continue;
            }
            let page = obj.get("page_idx").and_then(Value::as_i64).unwrap_or(0) + 1;
            let bbox =
                normalize_bbox_to_unit(&read_bbox(obj.get("bbox")), None, None, Some(1000.0));
            let mut meta = Map::new();
            meta.insert("source".into(), Value::String("content_list".into()));
            blocks.push(make_block(
                doc_id,
                order,
                page,
                bbox,
                canonical,
                &content,
                popo,
                level,
                Some(label),
                meta,
            ));
            order += 1;
        }
        ReaderResult::ok(self.model_name, doc_id, blocks)
    }

    /// Shared per-item construction for the `middle.json`/`model.json` paths.
    #[allow(clippy::too_many_arguments)]
    fn block_from_item(
        &self,
        doc_id: &str,
        order: i64,
        page_index: i64,
        obj: &Map<String, Value>,
        page_width: Option<f64>,
        page_height: Option<f64>,
        assumed_scale: Option<f64>,
    ) -> Option<popo_core::NormalizedBlock> {
        let label = obj
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("text")
            .to_string();
        let (canonical, popo) = map_mineru_label(&label);
        if popo == SKIP_TYPE {
            return None;
        }
        let content = extract_block_content(obj);
        if content.is_empty() && matches!(canonical, "text" | "title" | "caption") {
            return None;
        }
        let bbox = normalize_bbox_to_unit(
            &read_bbox(obj.get("bbox")),
            page_width,
            page_height,
            assumed_scale,
        );
        Some(make_block(
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
        ))
    }
}

impl OcrReader for MineruReader {
    fn model_name(&self) -> &str {
        self.model_name
    }

    fn read_doc(&self, doc_id: &str) -> Result<ReaderResult> {
        let doc_root = self.doc_root(doc_id);
        let middle_path = doc_root.join(format!("{doc_id}_middle.json"));
        let content_list_path = doc_root.join(format!("{doc_id}_content_list.json"));

        if middle_path.exists() {
            let data = load_json(&middle_path)?;
            return Ok(self.read_middle(doc_id, &data));
        }
        if content_list_path.exists() {
            let data = load_json(&content_list_path)?;
            let items = data.as_array().cloned().unwrap_or_default();
            return Ok(self.read_content_list(doc_id, &items));
        }
        Ok(ReaderResult::missing(
            self.model_name,
            doc_id,
            format!("missing {}", middle_path.display()),
        ))
    }
}

/// Reader for MonkeyOCR output: shares MinerU's `middle.json` path but with no
/// inner directory.
pub struct MonkeyOcrReader {
    inner: MineruReader,
}

impl MonkeyOcrReader {
    /// Build a MonkeyOCR reader rooted at the per-model input directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let mut inner = MineruReader::new(root);
        inner.inner_dir = "";
        inner.model_name = "monkeyocr";
        Self { inner }
    }
}

impl OcrReader for MonkeyOcrReader {
    fn model_name(&self) -> &str {
        self.inner.model_name
    }

    fn read_doc(&self, doc_id: &str) -> Result<ReaderResult> {
        let doc_root = self.inner.root.join(doc_id);
        let middle_path = doc_root.join(format!("{doc_id}_middle.json"));
        if !middle_path.exists() {
            return Ok(ReaderResult::missing(
                self.inner.model_name,
                doc_id,
                format!("missing {}", middle_path.display()),
            ));
        }
        let data = load_json(&middle_path)?;
        Ok(self.inner.read_middle(doc_id, &data))
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
    fn label_mapping_matches_python() {
        assert_eq!(map_mineru_label("title"), ("title", "title"));
        assert_eq!(map_mineru_label("table"), ("table", "table"));
        assert_eq!(
            map_mineru_label("image_caption"),
            ("caption", "image_caption")
        );
        assert_eq!(map_mineru_label("equation"), ("text", "equation"));
        assert_eq!(map_mineru_label("list"), ("text", "list_item"));
        assert_eq!(map_mineru_label("discarded"), (SKIP_TYPE, SKIP_TYPE));
        assert_eq!(map_mineru_label("number"), ("text", "page_number"));
        assert_eq!(map_mineru_label("whatever"), ("text", "text"));
    }

    #[test]
    fn content_list_promotes_title_and_skips_discarded() {
        let reader = MineruReader::new("/unused");
        let items = vec![
            json!({ "type": "text", "text": "hi", "bbox": [100, 100, 900, 200], "page_idx": 0, "text_level": 1 }),
            json!({ "type": "discarded", "text": "junk", "bbox": [0, 0, 10, 10], "page_idx": 0 }),
            json!({ "type": "text", "text": "body", "bbox": [100, 300, 900, 400], "page_idx": 1 }),
        ];
        let res = reader.read_content_list("doc", &items);
        assert_eq!(res.status, "ok");
        assert_eq!(res.blocks.len(), 2);
        // First (page 1) is the promoted title.
        assert_eq!(res.blocks[0].kind, "title");
        assert_eq!(res.blocks[0].popo_type, "title");
        assert_eq!(res.blocks[0].title_level, Some(1));
        assert_eq!(res.blocks[0].page, 1);
        // bbox normalized by assumed scale 1000.
        assert_eq!(res.blocks[0].bbox, [0.1, 0.1, 0.9, 0.2]);
        // Second block is on page 2, reassigned to order 1.
        assert_eq!(res.blocks[1].block_id, "doc:1");
        assert_eq!(res.blocks[1].content, "body");
    }

    #[test]
    fn middle_reads_page_size_and_para_blocks() {
        let reader = MineruReader::new("/unused");
        let data = json!({
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [1000.0, 2000.0],
                    "para_blocks": [
                        { "type": "title", "bbox": [100, 200, 900, 300],
                          "lines": [ { "spans": [ { "content": "Heading" } ] } ] }
                    ]
                }
            ]
        });
        let res = reader.read_middle("doc", &data);
        assert_eq!(res.blocks.len(), 1);
        let b = &res.blocks[0];
        assert_eq!(b.kind, "title");
        assert_eq!(b.content, "Heading");
        assert_eq!(b.page, 1);
        assert_eq!(b.bbox, [0.1, 0.1, 0.9, 0.15]);
    }
}
