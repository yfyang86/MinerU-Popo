//! The mutable working block used through the inference stage.
//!
//! Built from the normalized "popo-pages" input, carries the per-block outputs
//! the four subtasks fill in (`contd`, `level`, `image`, `table_merge`), and
//! serializes back to the `doc_blocks` shape the tree builder consumes.

use popo_core::Bbox;
use serde_json::{Map, Value};

/// One document block during inference. Mirrors the Python `doc_blocks` entry:
/// the original popo block plus the mutable subtask outputs.
#[derive(Debug, Clone)]
pub struct WorkBlock {
    /// 1-based id, sequential across the whole document.
    pub id: i64,
    /// 1-based page number.
    pub page: i64,
    /// Popo type (`text`, `list_item`, `title`, `equation`, `image`, …).
    pub kind: String,
    /// Block text content.
    pub content: String,
    /// Bounding box, `xyxy` in `0..1`.
    pub bbox: Bbox,
    /// Text-truncation link: 1-based id of the continuation target, or `-1`.
    pub contd: i64,
    /// Title level, or `-1`.
    pub level: i64,
    /// Image-text association: 1-based id of the linked block, or `-1`.
    pub image: i64,
    /// Cross-page table-merge partner id, or `None` if not a table / unmerged.
    pub table_merge: Option<i64>,
    /// Original block fields, retained for output round-tripping.
    pub source: Map<String, Value>,
}

impl WorkBlock {
    /// Serialize back to a `doc_blocks` entry: the original fields plus
    /// `page`, `id`, and the subtask outputs.
    pub fn to_output_value(&self) -> Value {
        let mut m = self.source.clone();
        m.insert("page".into(), Value::from(self.page));
        m.insert("id".into(), Value::from(self.id));
        m.insert("contd".into(), Value::from(self.contd));
        m.insert("level".into(), Value::from(self.level));
        m.insert("image".into(), Value::from(self.image));
        if let Some(tm) = self.table_merge {
            m.insert("table_merge".into(), Value::from(tm));
        }
        Value::Object(m)
    }
}

/// Read a bbox from a JSON value, defaulting to zeros.
fn read_bbox(v: Option<&Value>) -> Bbox {
    let mut out = [0.0; 4];
    if let Some(arr) = v.and_then(Value::as_array) {
        for (i, slot) in out.iter_mut().enumerate() {
            if let Some(x) = arr.get(i).and_then(Value::as_f64) {
                *slot = x;
            }
        }
    }
    out
}

/// Build the document's working blocks from the normalized `pages` map
/// (`{ "1": [block, …], … }`), assigning 1-based ids in page/document order.
/// Mirrors the id/page assignment at the top of the Python `main`.
pub fn build_doc_blocks(pages: &Map<String, Value>) -> Vec<WorkBlock> {
    let mut blocks = Vec::new();
    let mut id = 1i64;
    for (page_key, page_blocks) in pages {
        let page: i64 = page_key.parse().unwrap_or(0);
        let Some(items) = page_blocks.as_array() else {
            continue;
        };
        for item in items {
            let Some(obj) = item.as_object() else {
                continue;
            };
            let kind = obj
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("text")
                .to_string();
            let content = obj
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let bbox = read_bbox(obj.get("bbox"));
            blocks.push(WorkBlock {
                id,
                page,
                kind,
                content,
                bbox,
                contd: -1,
                level: -1,
                image: -1,
                table_merge: None,
                source: obj.clone(),
            });
            id += 1;
        }
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builds_sequential_ids_across_pages() {
        let pages = json!({
            "1": [
                { "type": "title", "content": "A", "bbox": [0.0, 0.0, 1.0, 0.1] },
                { "type": "text", "content": "b", "bbox": [0.0, 0.2, 1.0, 0.3] }
            ],
            "2": [
                { "type": "text", "content": "c", "bbox": [0.0, 0.0, 1.0, 0.1] }
            ]
        });
        let blocks = build_doc_blocks(pages.as_object().unwrap());
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].id, 1);
        assert_eq!(blocks[0].page, 1);
        assert_eq!(blocks[2].id, 3);
        assert_eq!(blocks[2].page, 2);
        assert_eq!(blocks[0].contd, -1);
    }

    #[test]
    fn output_overlays_computed_fields() {
        let pages = json!({ "1": [ { "type": "text", "content": "x", "bbox": [0.0,0.0,1.0,0.1], "source_id": "d:0" } ] });
        let mut blocks = build_doc_blocks(pages.as_object().unwrap());
        blocks[0].contd = 5;
        blocks[0].level = 2;
        let v = blocks[0].to_output_value();
        assert_eq!(v["id"], 1);
        assert_eq!(v["contd"], 5);
        assert_eq!(v["level"], 2);
        assert_eq!(v["source_id"], "d:0"); // original field retained
    }
}
