//! Table-merge subtask: decide whether the last table on one page continues
//! into the first table on the next.
//!
//! Port of `inference.py::filter_table_merge` plus
//! `data_engine/table_merge_filter.py` (the 6-check heuristic screen) and the
//! `add_table_merge` prompt. HTML table-structure analysis lives in
//! `popo-table`; the model is consulted text-only (no page image).

use std::sync::OnceLock;

use popo_table::{
    detect_table_headers, first_data_row_span_info, full_to_half, last_row_span_info, parse_table,
    row_colspan_total, row_visual_columns, Table,
};
use regex::Regex;
use serde_json::Value;

use crate::block::WorkBlock;

const END_MARKERS: &[&str] = &[
    "(续)",
    "(续表)",
    "(续上表)",
    "(continued)",
    "(cont.)",
    "(cont'd)",
    "(…continued)",
    "续表",
];
const INLINE_MARKERS: &[&str] = &["(continued)"];

/// One screened table pair, ready for the merge prompt.
#[derive(Debug, Clone)]
pub struct MergeInput {
    /// Position of the upper table (previous page).
    pub table1_idx: usize,
    /// Position of the lower table (next page).
    pub table2_idx: usize,
    /// Last-row span info of the upper table.
    pub upper: Vec<Vec<String>>,
    /// First-data-row span info of the lower table.
    pub lower: Vec<Vec<String>>,
}

fn text_kind(kind: &str) -> bool {
    matches!(kind, "text" | "list_item" | "list")
}

/// Caption for a table: the block immediately before it, accepted as an
/// explicit caption type, a title, or short caption-like text
/// (Python `_get_caption_for_table`).
fn caption_for_table(blocks: &[WorkBlock], idx: usize) -> Option<String> {
    if idx == 0 {
        return None;
    }
    let prev = &blocks[idx - 1];
    let content = prev.content.trim().to_string();
    match prev.kind.as_str() {
        "table_caption" | "tab-title" | "tab-caption" | "title" => Some(content),
        "text" if content.chars().count() < 150 => {
            static RE: OnceLock<Regex> = OnceLock::new();
            let re = RE.get_or_init(|| {
                Regex::new(r"(?i)^(?:table|表|exhibit|figure|图)\s*\d+|^\d+(?:\.\d+)*\s+|^[一二三四五六七八九十]+、")
                    .expect("caption regex")
            });
            if re.is_match(&content) {
                Some(content)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Count footnote-like blocks following a table on its page
/// (Python `_get_footnotes_for_table`).
fn footnote_count(blocks: &[WorkBlock], idx: usize) -> usize {
    let page = blocks[idx].page;
    let mut count = 0;
    for b in &blocks[idx + 1..] {
        if b.page != page {
            break;
        }
        if matches!(
            b.kind.as_str(),
            "table_footnote" | "table_caption" | "tab-caption" | "tab-title"
        ) {
            count += 1;
        } else {
            break;
        }
    }
    count
}

fn check_text_between(blocks: &[WorkBlock], t1: usize, t2: usize) -> bool {
    let p1 = blocks[t1].page;
    for b in &blocks[t1 + 1..] {
        if b.page != p1 {
            break;
        }
        if text_kind(&b.kind) {
            return false;
        }
    }
    let p2 = blocks[t2].page;
    for b in blocks[..t2].iter().rev() {
        if b.page != p2 {
            break;
        }
        if text_kind(&b.kind) {
            return false;
        }
    }
    true
}

fn table_number_patterns() -> &'static [(Regex, &'static str)] {
    static PATS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    PATS.get_or_init(|| {
        let raw: &[(&str, &str)] = &[
            (r"[Ee]xhibit\s*(\d+)", "Exhibit"),
            (r"EXHIBIT\s*(\d+)", "Exhibit"),
            (r"[Tt]able\s*([A-Za-z]?\s*-?\s*[\d.]+)", "Table"),
            (r"[Tt]ab\.?\s*([A-Za-z]?\s*-?\s*[\d.]+)", "Table"),
            (r"TABLE\s*([A-Za-z]?\s*-?\s*[\d.]+)", "Table"),
            (r"表\s*([A-Za-z]?\s*-?\s*[\d.]+)", "Table"),
            (r"[Ff]igure\s*(\d+)", "Figure"),
            (r"图\s*(\d+)", "Figure"),
            (r"^([一二三四五六七八九十]+)、", "CN_Num"),
        ];
        raw.iter()
            .map(|(p, t)| (Regex::new(p).expect("table number regex"), *t))
            .collect()
    })
}

fn extract_table_number(caption: &str) -> Option<(&'static str, String)> {
    for (re, label) in table_number_patterns() {
        if let Some(caps) = re.captures(caption) {
            if let Some(m) = caps.get(1) {
                let num = m.as_str().replace([' ', '-'], "").to_uppercase();
                return Some((label, num));
            }
        }
    }
    None
}

fn clean_caption(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"[^\w\x{4e00}-\x{9fa5}]").expect("clean regex"));
    let mut t = full_to_half(text).to_lowercase();
    for marker in END_MARKERS {
        let m = marker.to_lowercase();
        if t.ends_with(&m) {
            t.truncate(t.len() - m.len());
            break;
        }
    }
    re.replace_all(&t, "").into_owned()
}

fn check_caption_consistency(blocks: &[WorkBlock], t1: usize, t2: usize) -> bool {
    let cap1 = caption_for_table(blocks, t1);
    let cap2 = caption_for_table(blocks, t2);
    match (&cap1, &cap2) {
        (None, None) => true,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (Some(c1), Some(c2)) => {
            let r1 = extract_table_number(c1);
            let r2 = extract_table_number(c2);
            if let (Some(a), Some(b)) = (&r1, &r2) {
                return a.0 == b.0 && a.1 == b.1;
            }
            clean_caption(c1) == clean_caption(c2)
        }
    }
}

fn check_continuation_marker(blocks: &[WorkBlock], t2: usize) -> bool {
    let Some(cap2) = caption_for_table(blocks, t2) else {
        return true;
    };
    let cl = full_to_half(&cap2).to_lowercase();
    if END_MARKERS.iter().any(|m| cl.ends_with(&m.to_lowercase())) {
        return true;
    }
    INLINE_MARKERS
        .iter()
        .any(|m| cl.contains(&m.to_lowercase()))
}

fn check_footnote_count(blocks: &[WorkBlock], t1: usize, t2: usize) -> bool {
    let count = footnote_count(blocks, t1);
    if caption_for_table(blocks, t2).is_some() {
        count <= 1
    } else {
        count == 0
    }
}

fn check_width_difference(blocks: &[WorkBlock], t1: usize, t2: usize) -> bool {
    let w1 = blocks[t1].bbox[2] - blocks[t1].bbox[0];
    let w2 = blocks[t2].bbox[2] - blocks[t2].bbox[0];
    let min = w1.min(w2);
    if min <= 0.0 {
        return true;
    }
    (w1 - w2).abs() / min < 0.1
}

/// Last row of `t1` (by structure) vs first data row of `t2` agree in any of
/// effective / colspan / visual column counts (Python `_check_rows_match`).
fn check_rows_match(t1: &Table, t2: &Table) -> bool {
    let last = t1
        .rows
        .iter()
        .enumerate()
        .rev()
        .find(|(_, r)| !r.is_empty());
    let header = detect_table_headers(t1, t2);
    let first = if t2.rows.len() > header {
        Some((header, &t2.rows[header]))
    } else {
        None
    };
    let (Some((last_idx, last_row)), Some((first_idx, first_row))) = (last, first) else {
        return false;
    };
    t1.row_effective_columns(last_idx) == t2.row_effective_columns(first_idx)
        || row_colspan_total(last_row) == row_colspan_total(first_row)
        || row_visual_columns(last_row) == row_visual_columns(first_row)
}

fn check_column_count(blocks: &[WorkBlock], t1: usize, t2: usize) -> bool {
    let s1 = parse_table(&blocks[t1].content);
    let s2 = parse_table(&blocks[t2].content);
    if !s1.has_rows() || !s2.has_rows() {
        return false;
    }
    if s1.total_columns() == s2.total_columns() {
        return true;
    }
    check_rows_match(&s1, &s2)
}

/// Run the 6-check screen in order; reject on the first failure
/// (Python `filter_table_merge_candidates`).
pub fn filter_table_merge_candidates(blocks: &[WorkBlock], t1: usize, t2: usize) -> bool {
    check_text_between(blocks, t1, t2)
        && check_caption_consistency(blocks, t1, t2)
        && check_continuation_marker(blocks, t2)
        && check_footnote_count(blocks, t1, t2)
        && check_width_difference(blocks, t1, t2)
        && check_column_count(blocks, t1, t2)
}

/// Find adjacent-page table pairs that pass screening and prepare their merge
/// prompt rows (Python `filter_table_merge`).
pub fn filter_table_merge(blocks: &[WorkBlock]) -> Vec<MergeInput> {
    // Tables grouped by page, preserving document order within a page.
    let mut by_page: std::collections::BTreeMap<i64, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, b) in blocks.iter().enumerate() {
        if b.kind == "table" {
            by_page.entry(b.page).or_default().push(i);
        }
    }
    if by_page.len() < 2 {
        return Vec::new();
    }

    let pages: Vec<i64> = by_page.keys().copied().collect();
    let mut inputs = Vec::new();
    for w in pages.windows(2) {
        let (p1, p2) = (w[0], w[1]);
        if p2 != p1 + 1 {
            continue;
        }
        let t1 = *by_page[&p1].last().unwrap();
        let t2 = by_page[&p2][0];
        if !filter_table_merge_candidates(blocks, t1, t2) {
            continue;
        }
        let s1 = parse_table(&blocks[t1].content);
        let s2 = parse_table(&blocks[t2].content);
        if !s1.has_rows() || !s2.has_rows() {
            continue;
        }
        let header = detect_table_headers(&s1, &s2);
        inputs.push(MergeInput {
            table1_idx: t1,
            table2_idx: t2,
            upper: last_row_span_info(&s1),
            lower: first_data_row_span_info(&s2, header),
        });
    }
    inputs
}

/// Render a `Vec<Vec<String>>` as a Python list-of-lists repr (single quotes),
/// matching how the Python f-string stringifies the row data.
fn py_repr(rows: &[Vec<String>]) -> String {
    let inner: Vec<String> = rows
        .iter()
        .map(|row| {
            let cells: Vec<String> = row
                .iter()
                .map(|c| format!("'{}'", c.replace('\'', "\\'")))
                .collect();
            format!("[{}]", cells.join(", "))
        })
        .collect();
    format!("[{}]", inner.join(", "))
}

/// Build the table-merge prompt (Python `add_table_merge`).
pub fn build_table_merge_prompt(upper: &[Vec<String>], lower: &[Vec<String>]) -> String {
    format!(
        "\n## Table 1 (Previous Page - Last Table)\n\n**Caption:** :\"\"\n**Last Row(s) Data:**\n{}\n\n---\n\n## Table 2 (Current Page - First Table)\n\n**Caption:** :\"\"\n**First Data Row(s):**\n{}\n",
        py_repr(upper),
        py_repr(lower)
    )
}

/// Apply a merge decision: when the model returns a non-empty cell list, link
/// the two tables and store the cell list on both (Python apply logic).
pub fn apply_merge(blocks: &mut [WorkBlock], mi: &MergeInput, cell_list: &Value) {
    let non_empty = cell_list.as_array().map(|a| !a.is_empty()).unwrap_or(false);
    if !non_empty {
        return;
    }
    let id1 = blocks[mi.table1_idx].id;
    let id2 = blocks[mi.table2_idx].id;
    blocks[mi.table1_idx].table_merge = Some(id2);
    blocks[mi.table2_idx].table_merge = Some(id1);
    blocks[mi.table1_idx].cell_list = Some(cell_list.clone());
    blocks[mi.table2_idx].cell_list = Some(cell_list.clone());
}

#[cfg(test)]
mod tests {
    use super::*;
    use popo_core::Bbox;
    use serde_json::{json, Map};

    fn block(id: i64, page: i64, kind: &str, content: &str, bbox: Bbox) -> WorkBlock {
        WorkBlock {
            id,
            page,
            kind: kind.into(),
            content: content.into(),
            bbox,
            contd: -1,
            level: -1,
            image: -1,
            table_merge: if kind == "table" { Some(-1) } else { None },
            cell_list: None,
            source: Map::new(),
        }
    }

    const TBL: &str = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>";
    const TBL2: &str = "<table><tr><th>A</th><th>B</th></tr><tr><td>3</td><td>4</td></tr></table>";

    #[test]
    fn caption_detection() {
        let blocks = vec![
            block(1, 1, "title", "Table 1: Results", [0.0; 4]),
            block(2, 1, "table", TBL, [0.1, 0.1, 0.9, 0.5]),
        ];
        assert_eq!(
            caption_for_table(&blocks, 1).as_deref(),
            Some("Table 1: Results")
        );
        assert_eq!(caption_for_table(&blocks, 0), None);
    }

    #[test]
    fn candidate_pair_passes_all_checks() {
        // Two compatible tables on consecutive pages, no captions, equal width.
        let blocks = vec![
            block(1, 1, "table", TBL, [0.1, 0.1, 0.9, 0.6]),
            block(2, 2, "table", TBL2, [0.1, 0.1, 0.9, 0.6]),
        ];
        assert!(filter_table_merge_candidates(&blocks, 0, 1));
        let inputs = filter_table_merge(&blocks);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].table1_idx, 0);
        assert_eq!(inputs[0].table2_idx, 1);
    }

    #[test]
    fn rejected_when_text_between_tables() {
        let blocks = vec![
            block(1, 1, "table", TBL, [0.1, 0.1, 0.9, 0.6]),
            block(
                2,
                1,
                "text",
                "a paragraph after the table",
                [0.1, 0.7, 0.9, 0.8],
            ),
            block(3, 2, "table", TBL2, [0.1, 0.1, 0.9, 0.6]),
        ];
        assert!(!filter_table_merge_candidates(&blocks, 0, 2));
        assert!(filter_table_merge(&blocks).is_empty());
    }

    #[test]
    fn rejected_when_widths_differ() {
        let blocks = vec![
            block(1, 1, "table", TBL, [0.1, 0.1, 0.9, 0.6]), // width 0.8
            block(2, 2, "table", TBL2, [0.1, 0.1, 0.5, 0.6]), // width 0.4
        ];
        assert!(!check_width_difference(&blocks, 0, 1));
    }

    #[test]
    fn caption_only_on_table2_rejects() {
        let blocks = vec![
            block(1, 1, "table", TBL, [0.0; 4]),
            block(2, 2, "title", "Table 5: New", [0.0; 4]),
            block(3, 2, "table", TBL2, [0.0; 4]),
        ];
        // table1 (idx 0) has no caption, table2 (idx 2) has one → reject.
        assert!(!check_caption_consistency(&blocks, 0, 2));
    }

    #[test]
    fn apply_merge_links_tables_and_stores_cells() {
        let mut blocks = vec![
            block(1, 1, "table", TBL, [0.0; 4]),
            block(2, 2, "table", TBL2, [0.0; 4]),
        ];
        let mi = MergeInput {
            table1_idx: 0,
            table2_idx: 1,
            upper: vec![],
            lower: vec![],
        };
        apply_merge(&mut blocks, &mi, &json!([[0, 1]]));
        assert_eq!(blocks[0].table_merge, Some(2)); // partner id
        assert_eq!(blocks[1].table_merge, Some(1));
        assert_eq!(blocks[0].cell_list, Some(json!([[0, 1]])));

        // Empty cell list → no merge.
        let mut blocks2 = vec![
            block(1, 1, "table", TBL, [0.0; 4]),
            block(2, 2, "table", TBL2, [0.0; 4]),
        ];
        apply_merge(&mut blocks2, &mi, &json!([]));
        assert_eq!(blocks2[0].table_merge, Some(-1));
    }

    #[test]
    fn prompt_contains_row_data() {
        let p = build_table_merge_prompt(
            &[vec!["1".into(), "2".into()]],
            &[vec!["3".into(), "4".into()]],
        );
        assert!(p.contains("[['1', '2']]"));
        assert!(p.contains("[['3', '4']]"));
        assert!(p.contains("## Table 1"));
    }
}
