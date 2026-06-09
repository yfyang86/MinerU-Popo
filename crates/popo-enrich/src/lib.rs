//! Tree enrichment.
//!
//! Sprint 7 ships **subnode splitting** (port of `split_subnode.py`): the final
//! pass over an enriched tree that moves a visual node's children into a
//! `subnode` slot and splits long text nodes into ~500-character `sub_text`
//! chunks. Operates on the tree JSON directly. Metadata generation (which needs
//! the model + PDF crops) lands with the PDF stage.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{json, Value};

const VISUAL_TYPES: &[&str] = &["table", "chart", "image", "seal", "image_block"];

fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<\|txt_split\|>|<\|txt_contd\|>").expect("marker regex"))
}

/// Split a tree in place (Python `split_subnode`): every node gains a `subnode`
/// field, visual nodes move their children under it, and long text nodes are
/// chunked into `sub_text` subnodes.
pub fn split_subnode(node: &mut Value) {
    let Some(obj) = node.as_object_mut() else {
        return;
    };
    let typ = obj
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let content = obj
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let children = obj
        .get("children")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    // `true` → recurse into `subnode`; `false` → recurse into `children`.
    let recurse_into_subnode;

    if VISUAL_TYPES.contains(&typ.as_str()) && !children.is_empty() {
        obj.insert("subnode".into(), Value::Array(children));
        obj.insert("children".into(), Value::Array(Vec::new()));
        recurse_into_subnode = true;
    } else if typ == "text" && content.chars().count() > 500 {
        let location = obj
            .get("location")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let block_ids = obj
            .get("block_ids")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let title = obj
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let subnodes = build_text_subnodes(&content, &title, &location, &block_ids);
        obj.insert("subnode".into(), Value::Array(subnodes));
        recurse_into_subnode = false;
    } else {
        obj.insert("subnode".into(), Value::Array(Vec::new()));
        recurse_into_subnode = false;
    }

    let key = if recurse_into_subnode {
        "subnode"
    } else {
        "children"
    };
    if let Some(arr) = obj.get_mut(key).and_then(Value::as_array_mut) {
        for child in arr.iter_mut() {
            split_subnode(child);
        }
    }
}

/// Chunk a long text node's content into `sub_text` components, mirroring the
/// Python segment/index bookkeeping.
fn build_text_subnodes(
    content: &str,
    title: &str,
    location: &[Value],
    block_ids: &[Value],
) -> Vec<Value> {
    let segments: Vec<&str> = marker_re()
        .split(content)
        .filter(|s| !s.is_empty())
        .collect();

    let mut result_chunks: Vec<String> = Vec::new();
    let mut chunk_segments: Vec<Vec<usize>> = Vec::new();
    let mut current_chunk = String::new();
    let mut current_segments: Vec<usize> = Vec::new();
    let mut current_length = 0usize;

    // `segment_index` is 1-based (Python starts at 1).
    for (segment_index, segment) in (1usize..).zip(segments) {
        current_chunk.push_str(segment);
        current_length += segment.chars().count();
        current_segments.push(segment_index);
        if current_length > 500 {
            result_chunks.push(std::mem::take(&mut current_chunk));
            chunk_segments.push(std::mem::take(&mut current_segments));
            current_length = 0;
        }
    }
    if !current_chunk.is_empty() {
        result_chunks.push(current_chunk);
        chunk_segments.push(current_segments);
    }

    if result_chunks.len() <= 1 {
        return Vec::new();
    }

    let mut subnodes = Vec::new();
    for (i, (sub_content, sub_index)) in result_chunks.iter().zip(chunk_segments.iter()).enumerate()
    {
        // Adjust segment indices to address the location/block_ids arrays.
        let indices: Vec<usize> = if title == "Default Title" {
            sub_index.iter().map(|x| x.wrapping_sub(1)).collect()
        } else if !sub_index.contains(&0) {
            std::iter::once(0)
                .chain(sub_index.iter().copied())
                .collect()
        } else {
            sub_index.clone()
        };

        // Skip the chunk if any index is out of range (Python catches and skips).
        if indices
            .iter()
            .any(|&x| x >= location.len() || x >= block_ids.len())
        {
            continue;
        }
        let locs: Vec<Value> = indices.iter().map(|&x| location[x].clone()).collect();
        let ids: Vec<Value> = indices.iter().map(|&x| block_ids[x].clone()).collect();

        subnodes.push(json!({
            "type": "sub_text",
            "title": format!("{title}_{}", i + 1),
            "metadata": "",
            "content": sub_content,
            "level": 1,
            "location": locs,
            "block_ids": ids,
            "children": [],
        }));
    }
    subnodes
}

/// Convenience: split a whole tree value and return it.
pub fn split_tree(mut tree: Value) -> Value {
    split_subnode(&mut tree);
    tree
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visual_node_moves_children_to_subnode() {
        let mut tree = json!({
            "type": "image",
            "content": "<img/>",
            "children": [ { "type": "text", "content": "caption body", "children": [] } ],
            "location": [],
            "block_ids": []
        });
        split_subnode(&mut tree);
        assert_eq!(tree["children"].as_array().unwrap().len(), 0);
        assert_eq!(tree["subnode"].as_array().unwrap().len(), 1);
        // recursion reached the moved child (it gained a subnode field).
        assert!(tree["subnode"][0].get("subnode").is_some());
    }

    #[test]
    fn short_text_gets_empty_subnode() {
        let mut tree = json!({ "type": "text", "content": "short", "children": [], "location": [], "block_ids": [] });
        split_subnode(&mut tree);
        assert_eq!(tree["subnode"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn long_text_splits_into_sub_text_chunks() {
        // Two >500-char body segments (each flushes its own chunk) separated by
        // a split marker, under a title (location[0] is the title).
        let seg = "x".repeat(600);
        let content = format!("{seg}<|txt_split|>{seg}");
        let mut tree = json!({
            "type": "text",
            "title": "Heading",
            "content": content,
            "children": [],
            "location": [ {"page":1}, {"page":1}, {"page":1} ],
            "block_ids": [10, 11, 12]
        });
        split_subnode(&mut tree);
        let subs = tree["subnode"].as_array().unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0]["type"], "sub_text");
        assert_eq!(subs[0]["title"], "Heading_1");
        // first chunk: title index 0 prepended + segment 1 → block_ids [10, 11].
        assert_eq!(subs[0]["block_ids"], json!([10, 11]));
        assert_eq!(subs[1]["block_ids"], json!([10, 12]));
    }
}
