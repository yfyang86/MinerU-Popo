//! Shared helpers for readers: content extraction and block construction.

use popo_core::{normalize_text, Bbox, NormalizedBlock, CANONICAL_TYPES};
use serde_json::{Map, Value};

/// Sentinel popo-type meaning "drop this block" (Python `SKIP_TYPE`).
pub const SKIP_TYPE: &str = "__skip__";

/// Pull text content from a block, trying `content`, `text`, `html`, `words`
/// in order, then falling back to `lines[].spans[].content` (Python
/// `extract_block_content` + `extract_middle_text`).
pub fn extract_block_content(block: &Map<String, Value>) -> String {
    for key in ["content", "text", "html", "words"] {
        if let Some(v) = block.get(key) {
            let s = value_to_text(v);
            if !s.is_empty() {
                return normalize_text(&s);
            }
        }
    }
    extract_middle_text(block)
}

/// Join `lines[].spans[].content` into a single normalized string.
pub fn extract_middle_text(block: &Map<String, Value>) -> String {
    let Some(lines) = block.get("lines").and_then(Value::as_array) else {
        return String::new();
    };
    let mut line_texts = Vec::new();
    for line in lines {
        let Some(spans) = line.get("spans").and_then(Value::as_array) else {
            continue;
        };
        let joined: String = spans
            .iter()
            .map(|s| value_to_text(s.get("content").unwrap_or(&Value::Null)))
            .collect();
        let normalized = normalize_text(&joined);
        if !normalized.is_empty() {
            line_texts.push(normalized);
        }
    }
    line_texts.join(" ")
}

/// Coerce a JSON value to text. Strings pass through; arrays of strings join
/// with spaces (covers `words`); other values render via their display form.
fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(value_to_text)
            .collect::<Vec<_>>()
            .join(" "),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Read a bbox array from a JSON value, defaulting to zeros.
pub fn read_bbox(v: Option<&Value>) -> Vec<f64> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
        .unwrap_or_default()
}

/// Construct a [`NormalizedBlock`], collapsing unknown canonical types to
/// `text` and normalizing content (mirrors Python `BaseReader.make_block`).
/// The `bbox` is taken as-is (callers normalize to unit beforehand).
#[allow(clippy::too_many_arguments)]
pub fn make_block(
    doc_id: &str,
    order: i64,
    page: i64,
    bbox: Bbox,
    canonical_type: &str,
    content: &str,
    popo_type: &str,
    title_level: Option<i64>,
    source_label: Option<String>,
    meta: Map<String, Value>,
) -> NormalizedBlock {
    let kind = if CANONICAL_TYPES.contains(&canonical_type) {
        canonical_type
    } else {
        "text"
    };
    NormalizedBlock {
        block_id: format!("{doc_id}:{order}"),
        page,
        bbox,
        kind: kind.to_string(),
        content: normalize_text(content),
        order,
        popo_type: popo_type.to_string(),
        title_level,
        source_label,
        meta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_prefers_content_then_falls_back_to_spans() {
        let b = json!({ "text": "  hello  world " });
        let m = b.as_object().unwrap();
        assert_eq!(extract_block_content(m), "hello world");

        let mid = json!({
            "lines": [
                { "spans": [ { "content": "foo " }, { "content": "bar" } ] },
                { "spans": [ { "content": "baz" } ] }
            ]
        });
        assert_eq!(
            extract_block_content(mid.as_object().unwrap()),
            "foo bar baz"
        );
    }

    #[test]
    fn words_array_joins() {
        let b = json!({ "words": ["a", "b", "c"] });
        assert_eq!(extract_block_content(b.as_object().unwrap()), "a b c");
    }

    #[test]
    fn unknown_canonical_type_collapses_to_text() {
        let blk = make_block(
            "d",
            0,
            1,
            [0.0, 0.0, 1.0, 1.0],
            "weird",
            "x",
            "weird",
            None,
            None,
            Map::new(),
        );
        assert_eq!(blk.kind, "text");
    }
}
