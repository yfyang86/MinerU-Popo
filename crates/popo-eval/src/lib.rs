//! Title-hierarchy evaluation via TEDS (tree-edit-distance similarity).
//!
//! Rust port of `eval/evaluate.py`: a native Zhang-Shasha tree-edit-distance
//! (replacing `zss`), a faithful port of Python's `SequenceMatcher.ratio`
//! (Ratcliff-Obershelp), bbox alignment scoring, greedy GT↔prediction matching,
//! and the `content_aware` TEDS score.

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

mod teds;
pub use teds::{count_nodes_excluding_root, title_teds_score, tree_edit_distance};

/// A ground-truth title block parsed from a benchmark prompt.
#[derive(Debug, Clone)]
pub struct GtTitleBlock {
    /// Block id (model-facing).
    pub block_id: String,
    /// 1-based page.
    pub page: i64,
    /// Bbox `[left, top, right, bottom]`.
    pub bbox: [f64; 4],
    /// Normalized content.
    pub content: String,
    /// Position in the prompt.
    pub order: usize,
}

/// A predicted title block (subset of `NormalizedBlock` the eval needs).
#[derive(Debug, Clone)]
pub struct PredBlock {
    /// Block id.
    pub block_id: String,
    /// 1-based page.
    pub page: i64,
    /// Bbox `[left, top, right, bottom]`.
    pub bbox: [f64; 4],
    /// Canonical type (`title`, …).
    pub kind: String,
    /// Content.
    pub content: String,
    /// Title level, if assigned.
    pub title_level: Option<i64>,
}

/// One GT↔prediction match record.
#[derive(Debug, Clone)]
pub struct MatchRecord {
    /// Matched GT id.
    pub gt_id: String,
    /// Matched prediction id.
    pub pred_id: String,
    /// Predicted level.
    pub pred_level: Option<i64>,
    /// Alignment score.
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Text similarity
// ---------------------------------------------------------------------------

/// Normalize and strip to alphanumeric + CJK, lowercased (Python `compact_text`).
pub fn compact_text(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re =
        RE.get_or_init(|| Regex::new(r"[^0-9A-Za-z\x{4e00}-\x{9fff}]+").expect("compact regex"));
    let normalized = popo_core::normalize_text(text);
    re.replace_all(&normalized, "").to_lowercase()
}

/// Ratcliff-Obershelp similarity ratio, matching Python `SequenceMatcher.ratio`
/// for short strings (autojunk, which only triggers at len ≥ 200, is omitted).
pub fn ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    // b2j: char -> sorted indices in b.
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, &c) in b.iter().enumerate() {
        b2j.entry(c).or_default().push(j);
    }
    let matches = matching_total(&a, &b2j, 0, a.len(), 0, b.len());
    2.0 * matches as f64 / total as f64
}

fn find_longest_match(
    a: &[char],
    b2j: &HashMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for (i, _) in a.iter().enumerate().take(ahi).skip(alo) {
        let mut newj2len: HashMap<usize, usize> = HashMap::new();
        if let Some(js) = b2j.get(&a[i]) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = j2len.get(&j.wrapping_sub(1)).copied().unwrap_or(0) + 1;
                newj2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = newj2len;
    }
    (besti, bestj, bestsize)
}

fn matching_total(
    a: &[char],
    b2j: &HashMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> usize {
    let (i, j, k) = find_longest_match(a, b2j, alo, ahi, blo, bhi);
    if k == 0 {
        return 0;
    }
    k + matching_total(a, b2j, alo, i, blo, j) + matching_total(a, b2j, i + k, ahi, j + k, bhi)
}

/// Text similarity over compacted strings, with a substring-containment floor
/// (Python `text_similarity`).
pub fn text_similarity(left: &str, right: &str) -> f64 {
    let l = compact_text(left);
    let r = compact_text(right);
    if l.is_empty() && r.is_empty() {
        return 1.0;
    }
    if l.is_empty() || r.is_empty() {
        return 0.0;
    }
    let mut score = ratio(&l, &r);
    if l.contains(&r) || r.contains(&l) {
        score = score.max(0.9);
    }
    score
}

// ---------------------------------------------------------------------------
// Bbox geometry
// ---------------------------------------------------------------------------

fn intersection(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let left = a[0].max(b[0]);
    let top = a[1].max(b[1]);
    let right = a[2].min(b[2]);
    let bottom = a[3].min(b[3]);
    (right - left).max(0.0) * (bottom - top).max(0.0)
}
fn area(b: &[f64; 4]) -> f64 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}
/// Intersection-over-union of two boxes.
pub fn bbox_iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let inter = intersection(a, b);
    let union = area(a) + area(b) - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}
/// Intersection over the smaller box's area.
pub fn bbox_overlap_smaller(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let inter = intersection(a, b);
    let denom = area(a).min(area(b));
    if denom > 0.0 {
        inter / denom
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

/// Box/text/type weighted alignment score (Python `alignment_score`).
pub fn alignment_score(gt: &GtTitleBlock, pred: &PredBlock) -> f64 {
    if gt.page != pred.page {
        return 0.0;
    }
    let box_score = bbox_iou(&gt.bbox, &pred.bbox).max(bbox_overlap_smaller(&gt.bbox, &pred.bbox));
    let text_score = text_similarity(&gt.content, &pred.content);
    let type_score = if pred.kind == "title" { 1.0 } else { 0.4 };
    0.45 * box_score + 0.35 * text_score + 0.20 * type_score
}

/// Greedy GT↔prediction matching by descending score, with the
/// text/box acceptance floor (Python `align_title_blocks_to_gt`).
pub fn align_title_blocks_to_gt(
    preds: &[PredBlock],
    gts: &[GtTitleBlock],
    min_alignment_score: f64,
) -> (HashMap<String, String>, Vec<MatchRecord>) {
    let gt_by_id: HashMap<&str, &GtTitleBlock> =
        gts.iter().map(|g| (g.block_id.as_str(), g)).collect();
    let pred_by_id: HashMap<&str, &PredBlock> =
        preds.iter().map(|p| (p.block_id.as_str(), p)).collect();

    let mut candidates: Vec<(f64, String, String)> = Vec::new();
    for gt in gts {
        for pred in preds {
            let s = alignment_score(gt, pred);
            if s >= min_alignment_score {
                candidates.push((s, gt.block_id.clone(), pred.block_id.clone()));
            }
        }
    }
    // Sort descending by score, then by ids (mirrors Python's reverse tuple sort).
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
            .then(b.2.cmp(&a.2))
    });

    let mut model_to_gt: HashMap<String, String> = HashMap::new();
    let mut used_gt: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut matches = Vec::new();
    for (score, gt_id, pred_id) in candidates {
        if used_gt.contains(&gt_id) || model_to_gt.contains_key(&pred_id) {
            continue;
        }
        let gt = gt_by_id[gt_id.as_str()];
        let pred = pred_by_id[pred_id.as_str()];
        let text_score = text_similarity(&gt.content, &pred.content);
        let box_score =
            bbox_iou(&gt.bbox, &pred.bbox).max(bbox_overlap_smaller(&gt.bbox, &pred.bbox));
        if text_score < 0.45 && box_score < 0.8 {
            continue;
        }
        used_gt.insert(gt_id.clone());
        model_to_gt.insert(pred_id.clone(), gt_id.clone());
        matches.push(MatchRecord {
            gt_id,
            pred_id,
            pred_level: pred.title_level,
            score: (score * 10000.0).round() / 10000.0,
        });
    }
    (model_to_gt, matches)
}

/// Build prediction `(gt_id, level)` nodes in prediction order
/// (Python `build_prediction_nodes`).
pub fn build_prediction_nodes(
    preds: &[PredBlock],
    gts: &[GtTitleBlock],
    min_match_score: f64,
) -> (Vec<(String, i64)>, Vec<MatchRecord>) {
    let (model_to_gt, matches) = align_title_blocks_to_gt(preds, gts, min_match_score);
    let mut nodes = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for pred in preds {
        let Some(gt_id) = model_to_gt.get(&pred.block_id) else {
            continue;
        };
        if seen.contains(gt_id) {
            continue;
        }
        seen.insert(gt_id.clone());
        let level = match pred.title_level {
            Some(l) if l > 0 => l,
            _ => 1,
        };
        nodes.push((gt_id.clone(), level));
    }
    (nodes, matches)
}

// ---------------------------------------------------------------------------
// Prompt / label parsing
// ---------------------------------------------------------------------------

const TITLE_PREFIX: &str = "<image>\nTitle Level Analysis: ";

/// Parse the benchmark title prompt into GT blocks (Python `parse_title_prompt`).
pub fn parse_title_prompt(prompt: &str) -> Vec<GtTitleBlock> {
    let content = prompt.strip_prefix(TITLE_PREFIX).unwrap_or(prompt);
    let with_nl = format!("\n{content}");
    let mut parts: Vec<&str> = with_nl.split("\n<|id|>").collect();
    if parts.first() == Some(&"") {
        parts.remove(0);
    }

    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"(?s)^([^<]+)<\|page\|>(\d+)<\|box\|>([^<]+)<\|content\|>(.*)")
            .expect("title regex")
    });

    let mut blocks = Vec::new();
    for (order, part) in parts.iter().enumerate() {
        if part.trim().is_empty() {
            continue;
        }
        let Some(caps) = re.captures(part) else {
            continue;
        };
        let raw_box: Vec<f64> = caps[3]
            .split_whitespace()
            .take(4)
            .filter_map(|v| v.parse().ok())
            .collect();
        let bbox = if raw_box.len() == 4 {
            // Prompt serializes top left bottom right; store left top right bottom.
            [raw_box[1], raw_box[0], raw_box[3], raw_box[2]]
        } else {
            [0.0; 4]
        };
        blocks.push(GtTitleBlock {
            block_id: caps[1].trim().to_string(),
            page: caps[2].parse().unwrap_or(0),
            bbox,
            content: popo_core::normalize_text(&caps[4]),
            order,
        });
    }
    blocks
}

/// Parse `<|id|>I<|level|>L` label lines, keeping non-negative levels
/// (Python `parse_title_labels`).
pub fn parse_title_labels(raw: &str) -> Vec<(String, i64)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re =
        RE.get_or_init(|| Regex::new(r"^<\|id\|>([^<]+)<\|level\|>(-?\d+)").expect("label regex"));
    let mut nodes = Vec::new();
    for line in raw.lines() {
        if let Some(caps) = re.captures(line.trim()) {
            if let Ok(level) = caps[2].parse::<i64>() {
                if level >= 0 {
                    nodes.push((caps[1].trim().to_string(), level));
                }
            }
        }
    }
    nodes
}

/// Project `(id, level)` nodes to the `content_aware` TEDS labels: replace each
/// id with its GT block's compacted content (Python `gt_nodes_for_mode`).
pub fn content_aware_nodes(
    raw_nodes: &[(String, i64)],
    block_by_id: &HashMap<String, GtTitleBlock>,
) -> Vec<(String, i64)> {
    raw_nodes
        .iter()
        .map(|(id, level)| {
            let content = block_by_id
                .get(id)
                .map(|b| compact_text(&b.content))
                .unwrap_or_default();
            (content, *level)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_matches_python_examples() {
        // Python: SequenceMatcher(None, "abcd", "abcd").ratio() == 1.0
        assert!((ratio("abcd", "abcd") - 1.0).abs() < 1e-9);
        // SequenceMatcher(None, "abcd", "abxd").ratio() == 0.75 (3 matches of 8)
        assert!((ratio("abcd", "abxd") - 0.75).abs() < 1e-9);
        // disjoint
        assert!((ratio("abc", "xyz") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn text_similarity_substring_floor() {
        assert_eq!(text_similarity("", ""), 1.0);
        assert_eq!(text_similarity("abc", ""), 0.0);
        // containment floor
        assert!(text_similarity("Introduction", "Intro") >= 0.9);
    }

    #[test]
    fn bbox_metrics() {
        let a = [0.0, 0.0, 2.0, 2.0];
        let b = [1.0, 1.0, 3.0, 3.0];
        assert!((bbox_iou(&a, &b) - (1.0 / 7.0)).abs() < 1e-9);
        let inner = [0.0, 0.0, 1.0, 1.0];
        assert!((bbox_overlap_smaller(&a, &inner) - 1.0).abs() < 1e-9);
    }

    fn gt(id: &str, page: i64, bbox: [f64; 4], content: &str, order: usize) -> GtTitleBlock {
        GtTitleBlock {
            block_id: id.into(),
            page,
            bbox,
            content: content.into(),
            order,
        }
    }
    fn pred(id: &str, page: i64, bbox: [f64; 4], content: &str, level: i64) -> PredBlock {
        PredBlock {
            block_id: id.into(),
            page,
            bbox,
            kind: "title".into(),
            content: content.into(),
            title_level: Some(level),
        }
    }

    #[test]
    fn alignment_and_prediction_nodes() {
        let gts = vec![
            gt("g1", 1, [0.1, 0.1, 0.9, 0.2], "Chapter One", 0),
            gt("g2", 1, [0.1, 0.3, 0.9, 0.4], "Section A", 1),
        ];
        let preds = vec![
            pred("p1", 1, [0.1, 0.1, 0.9, 0.2], "Chapter One", 1),
            pred("p2", 1, [0.1, 0.3, 0.9, 0.4], "Section A", 2),
        ];
        let (nodes, matches) = build_prediction_nodes(&preds, &gts, 0.35);
        assert_eq!(matches.len(), 2);
        assert_eq!(nodes, vec![("g1".to_string(), 1), ("g2".to_string(), 2)]);
    }

    #[test]
    fn parse_prompt_and_labels() {
        let prompt =
            "<image>\nTitle Level Analysis: <|id|>5<|page|>2<|box|>100 200 150 800<|content|>Hello";
        let blocks = parse_title_prompt(prompt);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_id, "5");
        assert_eq!(blocks[0].page, 2);
        // top left bottom right -> left top right bottom
        assert_eq!(blocks[0].bbox, [200.0, 100.0, 800.0, 150.0]);
        assert_eq!(blocks[0].content, "Hello");

        let labels = parse_title_labels("<|id|>5<|level|>1\n<|id|>6<|level|>-1");
        assert_eq!(labels, vec![("5".to_string(), 1)]);
    }

    #[test]
    fn teds_identical_trees_score_one() {
        let nodes = vec![
            ("A".to_string(), 1),
            ("B".to_string(), 2),
            ("C".to_string(), 1),
        ];
        let score = title_teds_score(&nodes, &nodes);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn teds_penalizes_structural_difference() {
        let reference = vec![("A".to_string(), 1), ("B".to_string(), 2)];
        let predicted = vec![("A".to_string(), 1), ("B".to_string(), 1)]; // B now a sibling
        let score = title_teds_score(&reference, &predicted);
        assert!((0.0..1.0).contains(&score), "score was {score}");
    }
}
