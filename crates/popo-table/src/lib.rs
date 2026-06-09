//! HTML table-structure utilities for cross-page table merging.
//!
//! Rust port of the reference `data_engine/table_utils.py`. Tables are parsed
//! once into a flat row/cell model (with `colspan`/`rowspan`), and the column,
//! header, and span-row helpers operate on that model. Shared by the inference
//! table-merge subtask and the tree-builder's cross-page merge.
//!
//! Cell text uses `get_text()` semantics (concatenated descendant text);
//! "stripped" text is approximated by trimming, which is faithful for the flat,
//! single-text-node cells these tables contain.

use std::collections::{HashMap, HashSet};

use scraper::{Html, Selector};
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

/// NFKC normalization (Python `full_to_half`): folds full-width forms to half.
pub fn full_to_half(text: &str) -> String {
    text.nfkc().collect()
}

/// `''.join(full_to_half(text).split())` — normalize and drop all whitespace.
fn collapse_ws(text: &str) -> String {
    full_to_half(text).split_whitespace().collect()
}

/// One table cell.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Column span (default 1).
    pub colspan: usize,
    /// Row span (default 1).
    pub rowspan: usize,
    /// Concatenated descendant text (`get_text()`).
    pub raw: String,
}

impl Cell {
    /// `get_text(strip=True)` — approximated by trimming the raw text.
    pub fn text_stripped(&self) -> &str {
        self.raw.trim()
    }
}

/// A parsed table: rows of cells in document order.
#[derive(Debug, Clone, Default)]
pub struct Table {
    /// Rows, each a list of cells.
    pub rows: Vec<Vec<Cell>>,
}

fn parse_span(attr: Option<&str>) -> usize {
    attr.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(1)
}

/// Parse a table from an HTML fragment. Non-table input yields empty rows.
pub fn parse_table(html: &str) -> Table {
    let doc = Html::parse_fragment(html);
    let Ok(tr_sel) = Selector::parse("tr") else {
        return Table::default();
    };
    let Ok(cell_sel) = Selector::parse("td, th") else {
        return Table::default();
    };
    let mut rows = Vec::new();
    for tr in doc.select(&tr_sel) {
        let mut cells = Vec::new();
        for c in tr.select(&cell_sel) {
            cells.push(Cell {
                colspan: parse_span(c.value().attr("colspan")),
                rowspan: parse_span(c.value().attr("rowspan")),
                raw: c.text().collect(),
            });
        }
        rows.push(cells);
    }
    Table { rows }
}

impl Table {
    /// Whether the table has any rows.
    pub fn has_rows(&self) -> bool {
        !self.rows.is_empty()
    }

    /// Total columns, accounting for colspan/rowspan occupancy
    /// (Python `calculate_table_total_columns`).
    pub fn total_columns(&self) -> usize {
        if self.rows.is_empty() {
            return 0;
        }
        let mut occ: HashSet<(usize, usize)> = HashSet::new();
        let mut max_cols = 0;
        for (row_idx, row) in self.rows.iter().enumerate() {
            let mut col_idx = 0;
            for cell in row {
                while occ.contains(&(row_idx, col_idx)) {
                    col_idx += 1;
                }
                for r in row_idx..row_idx + cell.rowspan {
                    for c in col_idx..col_idx + cell.colspan {
                        occ.insert((r, c));
                    }
                }
                col_idx += cell.colspan;
                max_cols = max_cols.max(col_idx);
            }
        }
        max_cols
    }

    /// Per-row effective column count (`build_table_occupied_matrix`): the
    /// highest occupied column index in the row, plus one.
    pub fn effective_cols_per_row(&self) -> HashMap<usize, usize> {
        let cells = self.occupied_cells();
        let mut out = HashMap::new();
        for row_idx in 0..self.rows.len() {
            let max_col = cells
                .keys()
                .filter(|(r, _)| *r == row_idx)
                .map(|(_, c)| *c)
                .max();
            out.insert(row_idx, max_col.map(|c| c + 1).unwrap_or(0));
        }
        out
    }

    /// Effective columns for one row.
    pub fn row_effective_columns(&self, row_idx: usize) -> usize {
        self.effective_cols_per_row()
            .get(&row_idx)
            .copied()
            .unwrap_or(0)
    }

    /// Occupancy map `(row, col) -> (origin_row, cell_index_in_origin_row)`,
    /// used to walk visual rows while respecting spans.
    fn occupied_cells(&self) -> HashMap<(usize, usize), (usize, usize)> {
        let mut occ: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for (row_idx, row) in self.rows.iter().enumerate() {
            let mut col_idx = 0;
            for (cell_idx, cell) in row.iter().enumerate() {
                while occ.contains_key(&(row_idx, col_idx)) {
                    col_idx += 1;
                }
                for r in row_idx..row_idx + cell.rowspan {
                    for c in col_idx..col_idx + cell.colspan {
                        occ.entry((r, c)).or_insert((row_idx, cell_idx));
                    }
                }
                col_idx += cell.colspan;
            }
        }
        occ
    }
}

/// Sum of colspans in a row (`calculate_row_columns`).
pub fn row_colspan_total(row: &[Cell]) -> usize {
    row.iter().map(|c| c.colspan).sum()
}

/// Raw cell count in a row (`calculate_visual_columns`).
pub fn row_visual_columns(row: &[Cell]) -> usize {
    row.len()
}

/// Count leading header rows shared by two tables (`detect_table_headers`):
/// rows match when cell counts and each cell's colspan/rowspan/normalized-text
/// agree. Stops at the first mismatch.
pub fn detect_table_headers(t1: &Table, t2: &Table) -> usize {
    let min_rows = t1.rows.len().min(t2.rows.len()).min(5);
    let mut header_rows = 0;
    for i in 0..min_rows {
        let c1 = &t1.rows[i];
        let c2 = &t2.rows[i];
        let mut structure_match = c1.len() == c2.len();
        if structure_match {
            for (a, b) in c1.iter().zip(c2.iter()) {
                if a.colspan != b.colspan
                    || a.rowspan != b.rowspan
                    || collapse_ws(&a.raw) != collapse_ws(&b.raw)
                {
                    structure_match = false;
                    break;
                }
            }
        }
        if structure_match {
            header_rows += 1;
        } else {
            break;
        }
    }
    header_rows
}

/// Format the span-info string for a cell (Python span_parts logic).
fn span_label(cell: &Cell, text: &str) -> String {
    let mut parts = Vec::new();
    if cell.rowspan > 1 {
        parts.push(format!("rowspan={}", cell.rowspan));
    }
    if cell.colspan > 1 {
        parts.push(format!("colspan={}", cell.colspan));
    }
    if parts.is_empty() {
        text.to_string()
    } else {
        format!("{}, {}", parts.join(","), text)
    }
}

/// Collect the (deduplicated) cells occupying a given output row, in column
/// order, returning `(cell, label_text)` pairs.
fn row_cells_in_order(
    table: &Table,
    occ: &HashMap<(usize, usize), (usize, usize)>,
    row: usize,
    full_to_half_text: bool,
) -> (Vec<String>, usize) {
    let mut cols: Vec<usize> = occ
        .keys()
        .filter(|(r, _)| *r == row)
        .map(|(_, c)| *c)
        .collect();
    cols.sort_unstable();

    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut data = Vec::new();
    let mut colspan_sum = 0;
    for c in cols {
        let id = occ[&(row, c)];
        if seen.insert(id) {
            let cell = &table.rows[id.0][id.1];
            let stripped = cell.text_stripped();
            let text = if full_to_half_text {
                full_to_half(stripped)
            } else {
                stripped.to_string()
            };
            colspan_sum += cell.colspan;
            data.push(span_label(cell, &text));
        }
    }
    (data, colspan_sum)
}

/// Last visual rows of a table with span info, walking upward while each line is
/// a single full-width cell (`get_visual_last_row_cells_content_with_span_info`).
pub fn last_row_span_info(table: &Table) -> Vec<Vec<String>> {
    if table.rows.is_empty() {
        return Vec::new();
    }
    let total = table.total_columns();
    let occ = table.occupied_cells();
    let last = table.rows.len() - 1;
    let mut result = Vec::new();
    let lower = last.saturating_sub(4); // 5 rows: last .. last-4
    for curr in (lower..=last).rev() {
        let (data, colspan_sum) = row_cells_in_order(table, &occ, curr, false);
        if data.is_empty() {
            continue;
        }
        let cell_count = data.len();
        result.push(data);
        let full_width = cell_count == 1 && colspan_sum >= total;
        if !full_width {
            break;
        }
    }
    result.reverse();
    result
}

/// First data rows of a table (after `header_rows`) with span info
/// (`get_table_first_data_row_cells_with_span_info`).
pub fn first_data_row_span_info(table: &Table, header_rows: usize) -> Vec<Vec<String>> {
    if table.rows.is_empty() {
        return Vec::new();
    }
    let total = table.total_columns();
    let occ = table.occupied_cells();
    let len = table.rows.len();
    let start = if header_rows < len {
        header_rows
    } else {
        len - 1
    };
    let mut result = Vec::new();
    for curr in start..(start + 5).min(len) {
        let (data, colspan_sum) = row_cells_in_order(table, &occ, curr, true);
        if data.is_empty() {
            continue;
        }
        let cell_count = data.len();
        result.push(data);
        let full_width = cell_count == 1 && colspan_sum >= total;
        if !full_width {
            break;
        }
    }
    result
}

/// Extract the last balanced bracketed list from a model response and parse it
/// as a Python literal (`extract_last_coordinates`). Returns `None` if absent
/// or unparseable.
pub fn extract_last_coordinates(text: &str) -> Option<Value> {
    let chars: Vec<char> = text.chars().collect();
    let end = chars.iter().rposition(|&c| c == ']')?;
    let mut balance = 0i32;
    let mut start = None;
    for i in (0..=end).rev() {
        match chars[i] {
            ']' => balance += 1,
            '[' => balance -= 1,
            _ => {}
        }
        if balance == 0 {
            start = Some(i);
            break;
        }
    }
    let start = start?;
    let target: String = chars[start..=end].iter().collect();
    parse_py_literal(&target)
}

/// Minimal Python-literal parser covering the subset cell lists use: nested
/// lists/tuples, integers, floats, quoted strings, and `True`/`False`/`None`.
fn parse_py_literal(s: &str) -> Option<Value> {
    let chars: Vec<char> = s.chars().collect();
    let mut pos = 0;
    let value = parse_value(&chars, &mut pos)?;
    skip_ws(&chars, &mut pos);
    if pos == chars.len() {
        Some(value)
    } else {
        None
    }
}

fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn parse_value(chars: &[char], pos: &mut usize) -> Option<Value> {
    skip_ws(chars, pos);
    if *pos >= chars.len() {
        return None;
    }
    match chars[*pos] {
        '[' | '(' => parse_seq(chars, pos),
        '\'' | '"' => parse_string(chars, pos),
        _ => parse_atom(chars, pos),
    }
}

fn parse_seq(chars: &[char], pos: &mut usize) -> Option<Value> {
    let close = if chars[*pos] == '[' { ']' } else { ')' };
    *pos += 1;
    let mut items = Vec::new();
    loop {
        skip_ws(chars, pos);
        if *pos >= chars.len() {
            return None;
        }
        if chars[*pos] == close {
            *pos += 1;
            return Some(Value::Array(items));
        }
        items.push(parse_value(chars, pos)?);
        skip_ws(chars, pos);
        if *pos < chars.len() && chars[*pos] == ',' {
            *pos += 1;
        }
    }
}

fn parse_string(chars: &[char], pos: &mut usize) -> Option<Value> {
    let quote = chars[*pos];
    *pos += 1;
    let mut out = String::new();
    while *pos < chars.len() {
        let c = chars[*pos];
        if c == '\\' && *pos + 1 < chars.len() {
            *pos += 1;
            out.push(chars[*pos]);
        } else if c == quote {
            *pos += 1;
            return Some(Value::String(out));
        } else {
            out.push(c);
        }
        *pos += 1;
    }
    None
}

fn parse_atom(chars: &[char], pos: &mut usize) -> Option<Value> {
    let start = *pos;
    while *pos < chars.len()
        && !matches!(chars[*pos], ',' | ']' | ')')
        && !chars[*pos].is_whitespace()
    {
        *pos += 1;
    }
    let token: String = chars[start..*pos].iter().collect();
    match token.as_str() {
        "True" => Some(Value::Bool(true)),
        "False" => Some(Value::Bool(false)),
        "None" => Some(Value::Null),
        t => {
            if let Ok(i) = t.parse::<i64>() {
                Some(Value::from(i))
            } else if let Ok(f) = t.parse::<f64>() {
                Some(Value::from(f))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T1: &str = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>";
    const T2: &str = "<table><tr><th>A</th><th>B</th></tr><tr><td>3</td><td>4</td></tr></table>";

    #[test]
    fn parses_rows_and_spans() {
        let t = parse_table("<table><tr><td colspan=2>x</td><td>y</td></tr></table>");
        assert_eq!(t.rows.len(), 1);
        assert_eq!(t.rows[0].len(), 2);
        assert_eq!(t.rows[0][0].colspan, 2);
        assert_eq!(t.rows[0][0].text_stripped(), "x");
    }

    #[test]
    fn total_and_effective_columns() {
        let t = parse_table("<table><tr><td colspan=2>x</td><td>y</td></tr><tr><td>a</td><td>b</td><td>c</td></tr></table>");
        assert_eq!(t.total_columns(), 3);
        let eff = t.effective_cols_per_row();
        assert_eq!(eff[&0], 3);
        assert_eq!(eff[&1], 3);
    }

    #[test]
    fn rowspan_occupies_next_row() {
        let t = parse_table(
            "<table><tr><td rowspan=2>x</td><td>a</td></tr><tr><td>b</td></tr></table>",
        );
        // Row 1: col 0 occupied by the rowspan, so effective cols = 2.
        assert_eq!(t.row_effective_columns(1), 2);
    }

    #[test]
    fn detects_matching_header() {
        let a = parse_table(T1);
        let b = parse_table(T2);
        assert_eq!(detect_table_headers(&a, &b), 1); // first row matches, data rows differ
    }

    #[test]
    fn last_and_first_row_span_info() {
        let a = parse_table(T1);
        let b = parse_table(T2);
        let last = last_row_span_info(&a);
        assert_eq!(last, vec![vec!["1".to_string(), "2".to_string()]]);
        let first = first_data_row_span_info(&b, 1);
        assert_eq!(first, vec![vec!["3".to_string(), "4".to_string()]]);
    }

    #[test]
    fn row_column_helpers() {
        let t = parse_table("<table><tr><td colspan=3>x</td><td>y</td></tr></table>");
        assert_eq!(row_colspan_total(&t.rows[0]), 4);
        assert_eq!(row_visual_columns(&t.rows[0]), 2);
    }

    #[test]
    fn extracts_last_coordinate_list() {
        let v = extract_last_coordinates("Reasoning... final: [[0, 1], [1, 0]]").unwrap();
        assert_eq!(v, serde_json::json!([[0, 1], [1, 0]]));
        assert!(extract_last_coordinates("no brackets here").is_none());
        // Tuples parse as arrays; trailing prose is ignored via last-bracket scan.
        let v2 = extract_last_coordinates("answer = [(1, 2), (3, 4)]").unwrap();
        assert_eq!(v2, serde_json::json!([[1, 2], [3, 4]]));
    }

    #[test]
    fn empty_list_parses() {
        let v = extract_last_coordinates("result: []").unwrap();
        assert_eq!(v, serde_json::json!([]));
    }
}
