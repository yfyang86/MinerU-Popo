//! Document tree assembly from inference `doc_blocks`.
//!
//! Rust port of `get_json_tree.py`: remap supplements, merge cross-page tables,
//! group text under titles, build the heading-level tree, attach
//! visual/special elements and page supplements, and render the text preview.

use serde::Serialize;
use serde_json::{json, Map, Value};

const SPECIAL_TYPES: &[&str] = &[
    "table_footnote",
    "table",
    "chart",
    "table_caption",
    "image_footnote",
    "image",
    "image_caption",
    "seal",
];
const LARGE_BLOCK_TYPES: &[&str] = &[
    "super",
    "list",
    "ref_block",
    "equation_block",
    "image_block",
];
const SUPPLEMENT_TYPES: &[&str] = &[
    "page_title",
    "page_number",
    "page_footnote",
    "header",
    "aside_text",
    "footer",
];
const VISUAL_TYPES: &[&str] = &["table", "chart", "image", "seal", "image_block"];

fn supplement_source_label(label: &str) -> Option<&'static str> {
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

/// A `{bbox, page}` location entry.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Location {
    /// Bounding box.
    pub bbox: Vec<f64>,
    /// 1-based page number.
    pub page: i64,
}

/// A document-tree node (Python `cp`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct Component {
    /// Node type (`root`, `text`, `table`, `image`, supplement types, …).
    #[serde(rename = "type")]
    pub cp_type: String,
    /// Title / heading text.
    pub title: String,
    /// Free-form metadata (footnotes, supplement content).
    pub metadata: String,
    /// Body content.
    pub content: String,
    /// Heading level / link level.
    pub level: i64,
    /// Source locations.
    pub location: Vec<Location>,
    /// Source block ids.
    pub block_ids: Vec<i64>,
    /// Child nodes.
    pub children: Vec<Component>,
}

impl Component {
    fn new(cp_type: &str, level: i64) -> Self {
        Component {
            cp_type: cp_type.to_string(),
            level,
            ..Default::default()
        }
    }

    fn find_by_id(&mut self, id: i64) -> Option<&mut Component> {
        if self.block_ids.contains(&id) {
            return Some(self);
        }
        for child in self.children.iter_mut() {
            if let Some(found) = child.find_by_id(id) {
                return Some(found);
            }
        }
        None
    }
}

type Element = Map<String, Value>;

fn e_str(e: &Element, k: &str) -> String {
    e.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}
fn e_i64(e: &Element, k: &str, default: i64) -> i64 {
    e.get(k).and_then(Value::as_i64).unwrap_or(default)
}
fn e_bbox(e: &Element) -> Vec<f64> {
    e.get("bbox")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}
fn e_location(e: &Element) -> Location {
    Location {
        bbox: e_bbox(e),
        page: e_i64(e, "page", 0),
    }
}
fn parse_locations(v: &Value) -> Vec<Location> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|o| Location {
                    bbox: o
                        .get("bbox")
                        .and_then(Value::as_array)
                        .map(|b| b.iter().filter_map(Value::as_f64).collect())
                        .unwrap_or_default(),
                    page: o.get("page").and_then(Value::as_i64).unwrap_or(0),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Build the document tree from inference `doc_blocks`.
pub fn build_tree(doc_blocks: Vec<Value>) -> Component {
    let mut elements: Vec<Element> = doc_blocks
        .into_iter()
        .filter_map(|v| v.as_object().cloned())
        .collect();

    // 1. Remap supplement text blocks by their source label.
    for e in &mut elements {
        if e_str(e, "type") == "text" {
            if let Some(mapped) = supplement_source_label(&e_str(e, "source_label")) {
                e.insert("type".into(), json!(mapped));
            }
        }
    }

    // 2. Merge cross-page tables (drops the absorbed partner blocks).
    let mut elements = merge_cross_page_tables(elements);

    // 3. Demote level-less titles to text (mutates elements for later steps).
    for e in &mut elements {
        if e_str(e, "type") == "title" && e_i64(e, "level", -1) < 0 {
            e.insert("type".into(), json!("text"));
        }
    }

    let text_components = get_text_components(&elements);
    let mut root = construct_by_level(text_components);
    add_special_elements(&mut root, &elements);
    add_supplement(&mut root, &elements);
    root
}

fn merge_cross_page_tables(mut elements: Vec<Element>) -> Vec<Element> {
    use std::collections::{HashMap, HashSet};
    let id_to_index: HashMap<i64, usize> = elements
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.get("id").and_then(Value::as_i64).map(|id| (id, i)))
        .collect();
    let mut merged_partner_ids: HashSet<i64> = HashSet::new();

    for idx in 0..elements.len() {
        if e_str(&elements[idx], "type") != "table" {
            continue;
        }
        let partner_id = e_i64(&elements[idx], "table_merge", -1);
        if partner_id < 0 || merged_partner_ids.contains(&partner_id) {
            continue;
        }
        let Some(&partner_index) = id_to_index.get(&partner_id) else {
            continue;
        };
        if partner_index <= idx || e_str(&elements[partner_index], "type") != "table" {
            continue;
        }

        let element_id = e_i64(&elements[idx], "id", -1);
        let prev_html = e_str(&elements[idx], "content");
        let cur_html = e_str(&elements[partner_index], "content");
        let merged = popo_table::merge_html(&prev_html, &cur_html);

        let elem_loc =
            json!({ "bbox": e_bbox(&elements[idx]), "page": e_i64(&elements[idx], "page", 0) });
        let partner_loc = json!({ "bbox": e_bbox(&elements[partner_index]), "page": e_i64(&elements[partner_index], "page", 0) });

        let e = &mut elements[idx];
        e.insert("content".into(), json!(merged));
        e.insert("merged_locations".into(), json!([elem_loc, partner_loc]));
        e.insert("merged_block_ids".into(), json!([element_id, partner_id]));

        // Redirect image links that pointed at the absorbed partner.
        for other in &mut elements {
            if other.get("image").and_then(Value::as_i64) == Some(partner_id) {
                other.insert("image".into(), json!(element_id));
            }
        }
        merged_partner_ids.insert(partner_id);
    }

    elements
        .into_iter()
        .filter(|e| {
            e.get("id")
                .and_then(Value::as_i64)
                .map(|id| !merged_partner_ids.contains(&id))
                .unwrap_or(true)
        })
        .collect()
}

fn get_text_components(elements: &[Element]) -> Vec<Component> {
    let mut text_components = Vec::new();
    let mut contd_list: Vec<i64> = Vec::new();
    let mut cur = Component::new("text", 1);
    cur.title = "Default Title".to_string();

    for e in elements {
        let t = e_str(e, "type");
        if t == "title" {
            let title = e_str(e, "content");
            if cur.title != "Default Title" || !cur.content.is_empty() {
                text_components.push(std::mem::take(&mut cur));
            }
            cur = Component::new("text", e_i64(e, "level", -1));
            cur.title = title;
            cur.location = vec![e_location(e)];
            cur.block_ids = vec![e_i64(e, "id", -1)];
        } else if !SPECIAL_TYPES.contains(&t.as_str())
            && !LARGE_BLOCK_TYPES.contains(&t.as_str())
            && !SUPPLEMENT_TYPES.contains(&t.as_str())
        {
            let contd = e_i64(e, "contd", -1);
            if contd >= 0 {
                contd_list.push(contd);
            }
            let id = e_i64(e, "id", -1);
            let label = if contd_list.contains(&id) {
                "<|txt_contd|>"
            } else {
                "<|txt_split|>"
            };
            let content = e_str(e, "content");
            cur.content = if cur.content.is_empty() {
                content
            } else {
                format!("{}{label}{content}", cur.content)
            };
            cur.location.push(e_location(e));
            cur.block_ids.push(id);
        }
    }
    text_components.push(cur);
    text_components
}

struct ArenaNode {
    comp: Component,
    children: Vec<usize>,
}

fn construct_by_level(text_components: Vec<Component>) -> Component {
    let mut arena = vec![ArenaNode {
        comp: Component::new("root", 0),
        children: Vec::new(),
    }];
    let mut stack: Vec<(usize, i64)> = vec![(0, 0)];

    for cp in text_components {
        let level = if cp.level > 0 { cp.level } else { 100 };
        while stack.last().map(|&(_, l)| l >= level).unwrap_or(false) {
            stack.pop();
        }
        let parent = stack.last().map(|&(i, _)| i).unwrap_or(0);
        let idx = arena.len();
        arena.push(ArenaNode {
            comp: cp,
            children: Vec::new(),
        });
        arena[parent].children.push(idx);
        stack.push((idx, level));
    }

    materialize(&arena, 0)
}

fn materialize(arena: &[ArenaNode], idx: usize) -> Component {
    let mut c = arena[idx].comp.clone();
    c.children = arena[idx]
        .children
        .iter()
        .map(|&ci| materialize(arena, ci))
        .collect();
    c
}

// The nesting/attach loops index two parallel `visuals`/`parent_of` slices at
// once, so range indexing is clearer than iterator gymnastics.
#[allow(clippy::needless_range_loop)]
fn add_special_elements(root: &mut Component, elements: &[Element]) {
    // Build the visual components with their captions/footnotes attached.
    let mut visuals: Vec<Component> = Vec::new();
    for e in elements {
        let t = e_str(e, "type");
        if !VISUAL_TYPES.contains(&t.as_str()) {
            continue;
        }
        let id = e_i64(e, "id", -1);
        let mut vc = Component::new(&t, e_i64(e, "image", -1));
        vc.content = e_str(e, "content");
        vc.location = e
            .get("merged_locations")
            .map(parse_locations)
            .unwrap_or_else(|| vec![e_location(e)]);
        vc.block_ids = e
            .get("merged_block_ids")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_else(|| vec![id]);

        for elem in elements {
            if e_i64(elem, "image", -1) == id {
                let etype = e_str(elem, "type");
                let content = e_str(elem, "content");
                if etype.contains("caption") {
                    vc.title = if vc.title.is_empty() {
                        content
                    } else {
                        format!("{} {content}", vc.title)
                    };
                    vc.location.push(e_location(elem));
                    vc.block_ids.push(e_i64(elem, "id", -1));
                } else if etype.contains("footnote") {
                    vc.metadata = if vc.metadata.is_empty() {
                        content
                    } else {
                        format!("{} {content}", vc.metadata)
                    };
                    vc.location.push(e_location(elem));
                    vc.block_ids.push(e_i64(elem, "id", -1));
                }
            }
        }
        visuals.push(vc);
    }

    // Nest visuals whose link target lands in another visual's block ids.
    let n = visuals.len();
    let mut parent_of: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        for j in 0..n {
            if i != j && visuals[j].block_ids.contains(&visuals[i].level) {
                parent_of[i] = Some(j);
                break;
            }
        }
    }
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, p) in parent_of.iter().enumerate() {
        if let Some(j) = p {
            children_of[*j].push(i);
        }
    }
    fn materialize_visual(
        i: usize,
        visuals: &[Component],
        children_of: &[Vec<usize>],
    ) -> Component {
        let mut c = visuals[i].clone();
        c.children = children_of[i]
            .iter()
            .map(|&ci| materialize_visual(ci, visuals, children_of))
            .collect();
        c
    }

    // Attach top-level visuals to the text tree, one at a time so later visuals
    // can land under earlier ones.
    for i in 0..n {
        if parent_of[i].is_some() {
            continue;
        }
        let vc = materialize_visual(i, &visuals, &children_of);
        let target = if vc.level >= 0 {
            vc.level
        } else {
            let min_id = vc.block_ids.iter().copied().min().unwrap_or(0);
            find_former_title(elements, min_id)
        };
        if let Some(node) = root.find_by_id(target) {
            node.children.push(vc);
        }
    }
}

fn find_former_title(elements: &[Element], idx: i64) -> i64 {
    let mut former = 0;
    for e in elements {
        if e_str(e, "type") == "title" {
            let id = e_i64(e, "id", -1);
            if id < idx && id > former {
                former = id;
            }
        }
    }
    former
}

fn add_supplement(root: &mut Component, elements: &[Element]) {
    let mut exist: Vec<String> = Vec::new();
    for e in elements {
        let t = e_str(e, "type");
        if !SUPPLEMENT_TYPES.contains(&t.as_str()) {
            continue;
        }
        let page = e_i64(e, "page", 0);
        let base = format!("Page {page} - {t}");
        let mut title = base.clone();
        let mut cnt = 0;
        while exist.contains(&title) {
            cnt += 1;
            title = format!("{base} - {cnt}");
        }
        exist.push(title.clone());

        let content = e_str(e, "content");
        let mut supp = Component::new(&t, -1);
        supp.title = title;
        supp.metadata = content.clone();
        supp.content = content;
        supp.location = vec![e_location(e)];
        supp.block_ids = vec![e_i64(e, "id", -1)];
        root.children.push(supp);
    }
}

/// Render the indented text preview (Python `traverse_tree`).
pub fn tree_to_txt(root: &Component) -> String {
    let mut lines = Vec::new();
    fn walk(node: &Component, depth: usize, lines: &mut Vec<String>) {
        let indent = " ".repeat(depth * 4);
        let preview: String = node.content.chars().take(30).collect();
        let suffix = if node.content.chars().count() > 30 {
            "..."
        } else {
            ""
        };
        lines.push(format!("{indent}{}|{preview}{suffix}", node.title));
        for child in &node.children {
            walk(child, depth + 1, lines);
        }
    }
    walk(root, 0, &mut lines);
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(extra: Value) -> Value {
        extra
    }

    #[test]
    fn builds_title_text_hierarchy() {
        let docs = vec![
            blk(
                json!({ "type": "title", "content": "Chapter 1", "level": 1, "bbox": [0.0,0.0,1.0,0.1], "page": 1, "id": 1, "contd": -1, "image": -1 }),
            ),
            blk(
                json!({ "type": "text", "content": "Intro paragraph.", "level": -1, "bbox": [0.0,0.2,1.0,0.3], "page": 1, "id": 2, "contd": -1, "image": -1 }),
            ),
            blk(
                json!({ "type": "title", "content": "Section 1.1", "level": 2, "bbox": [0.0,0.4,1.0,0.5], "page": 1, "id": 3, "contd": -1, "image": -1 }),
            ),
            blk(
                json!({ "type": "text", "content": "Body.", "level": -1, "bbox": [0.0,0.6,1.0,0.7], "page": 1, "id": 4, "contd": -1, "image": -1 }),
            ),
        ];
        let tree = build_tree(docs);
        assert_eq!(tree.cp_type, "root");
        // root → Chapter 1 (level 1) → Section 1.1 (level 2) nested.
        assert_eq!(tree.children.len(), 1);
        let ch1 = &tree.children[0];
        assert_eq!(ch1.title, "Chapter 1");
        assert_eq!(ch1.content, "Intro paragraph.");
        let sec = ch1
            .children
            .iter()
            .find(|c| c.title == "Section 1.1")
            .unwrap();
        assert_eq!(sec.content, "Body.");
    }

    #[test]
    fn attaches_image_to_linked_title_and_adds_supplement() {
        let docs = vec![
            blk(
                json!({ "type": "title", "content": "Figures", "level": 1, "bbox": [0.0,0.0,1.0,0.1], "page": 1, "id": 1, "contd": -1, "image": -1 }),
            ),
            // image linked to title id 1 (image field = 1)
            blk(
                json!({ "type": "image", "content": "<img>", "bbox": [0.1,0.2,0.9,0.5], "page": 1, "id": 2, "contd": -1, "image": 1, "level": -1 }),
            ),
            // caption pointing at the image (image field = 2)
            blk(
                json!({ "type": "image_caption", "content": "Fig 1", "bbox": [0.1,0.55,0.9,0.6], "page": 1, "id": 3, "contd": -1, "image": 2, "level": -1 }),
            ),
            // a page supplement
            blk(
                json!({ "type": "text", "source_label": "page_number", "content": "12", "bbox": [0.9,0.95,1.0,1.0], "page": 1, "id": 4, "contd": -1, "image": -1, "level": -1 }),
            ),
        ];
        let tree = build_tree(docs);
        let figures = &tree.children[0];
        let image = figures
            .children
            .iter()
            .find(|c| c.cp_type == "image")
            .unwrap();
        assert_eq!(image.title, "Fig 1"); // caption attached
        assert_eq!(image.block_ids, vec![2, 3]);
        // supplement attached at root
        let supp = tree
            .children
            .iter()
            .find(|c| c.cp_type == "page_number")
            .unwrap();
        assert_eq!(supp.title, "Page 1 - page_number");
        assert_eq!(supp.content, "12");
    }

    #[test]
    fn cross_page_tables_merge_and_drop_partner() {
        let t1 = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>";
        let t2 = "<table><tr><th>A</th></tr><tr><td>2</td></tr></table>";
        let docs = vec![
            blk(
                json!({ "type": "title", "content": "Tables", "level": 1, "bbox": [0.0,0.0,1.0,0.1], "page": 1, "id": 1, "contd": -1, "image": -1 }),
            ),
            blk(
                json!({ "type": "table", "content": t1, "bbox": [0.1,0.1,0.9,0.5], "page": 1, "id": 2, "contd": -1, "image": -1, "level": -1, "table_merge": 3 }),
            ),
            blk(
                json!({ "type": "table", "content": t2, "bbox": [0.1,0.1,0.9,0.5], "page": 2, "id": 3, "contd": -1, "image": -1, "level": -1, "table_merge": 2 }),
            ),
        ];
        let tree = build_tree(docs);
        let tables: Vec<&Component> = collect_type(&tree, "table");
        assert_eq!(tables.len(), 1, "partner dropped, one merged table");
        assert_eq!(tables[0].block_ids, vec![2, 3]);
        assert!(tables[0].content.contains("<td>1</td>"));
        assert!(tables[0].content.contains("<td>2</td>"));
        // Attached under the preceding title (former-title fallback).
        let title = tree.children.iter().find(|c| c.title == "Tables").unwrap();
        assert!(title.children.iter().any(|c| c.cp_type == "table"));
    }

    #[test]
    fn txt_preview_indents_by_depth() {
        let docs = vec![
            blk(
                json!({ "type": "title", "content": "T", "level": 1, "bbox": [0.0,0.0,1.0,0.1], "page": 1, "id": 1, "contd": -1, "image": -1 }),
            ),
            blk(
                json!({ "type": "text", "content": "hello world this is a fairly long body line", "level": -1, "bbox": [0.0,0.2,1.0,0.3], "page": 1, "id": 2, "contd": -1, "image": -1 }),
            ),
        ];
        let tree = build_tree(docs);
        let txt = tree_to_txt(&tree);
        assert!(txt.contains("\n    T|hello world this is a fairly l..."));
    }

    fn collect_type<'a>(node: &'a Component, t: &str) -> Vec<&'a Component> {
        let mut out = Vec::new();
        if node.cp_type == t {
            out.push(node);
        }
        for c in &node.children {
            out.extend(collect_type(c, t));
        }
        out
    }
}
