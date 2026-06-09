//! Text-truncation subtask: detect text that continues across blocks/pages.
//!
//! Port of the Python `filter_contd` / `merge_rules` heuristics, the
//! `Truncation Detection` prompt, and `extract_label1` parsing. The pipeline is
//! split into pure functions so it is testable without a model or images; the
//! async runner that calls the model lives in the crate root.

use std::sync::OnceLock;

use regex::Regex;

use crate::block::WorkBlock;
use crate::chunk::Paged;

/// Sentence-terminating characters (Python `termination_chars`).
const TERMINATION: &[char] = &[
    '.', '。', '?', '!', '？', '！', '¿', '¡', '؟', 'ฯ', '۔', ':', '：', '…', ';', '；',
];

/// Closing punctuation that may trail a terminator (Python `close_chars`).
const CLOSE: &[char] = &['’', '”', '\'', '"', '）', '」', '】', ']', ')'];

/// A candidate block for truncation analysis (Python `filter_contd` output).
#[derive(Debug, Clone)]
pub struct JudgeBlock {
    /// 0-based position of the block in the document (the model-facing id).
    pub idx: i64,
    /// Shortened content (head … tail).
    pub content: String,
    /// 1-based page.
    pub page: i64,
    /// Bbox as `[top, left, bottom, right]` scaled to per-mille integer strings.
    pub bbox: [String; 4],
}

impl Paged for JudgeBlock {
    fn page(&self) -> i64 {
        self.page
    }
}

/// The first sentence: text up to and including the first terminator, trimmed.
pub fn head_sentence(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    if let Some(pos) = text.find(|c| TERMINATION.contains(&c)) {
        // include the terminator char itself
        let end = pos + text[pos..].chars().next().unwrap().len_utf8();
        text[..end].trim().to_string()
    } else {
        text.trim().to_string()
    }
}

/// The last sentence: the final non-empty segment after splitting on
/// terminators, trimmed (Python `get_tail_sentence`).
pub fn tail_sentence(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    let parts: Vec<&str> = text
        .split(|c| TERMINATION.contains(&c))
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    parts
        .last()
        .map(|s| s.to_string())
        .unwrap_or_else(|| text.trim().to_string())
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn char_slice(s: &str, range: std::ops::Range<usize>) -> String {
    s.chars()
        .skip(range.start)
        .take(range.end - range.start)
        .collect()
}

/// Build the shortened preview used in the prompt (Python `filter_contd`).
fn shorten(text: &str) -> String {
    let head = head_sentence(text);
    let tail = tail_sentence(text);
    let short = if head == tail {
        head
    } else {
        format!("{head} ... {tail}")
    };
    if char_len(&short) <= 103 {
        short
    } else {
        let n = char_len(&short);
        format!(
            "{}...{}",
            char_slice(&short, 0..50),
            char_slice(&short, n - 50..n)
        )
    }
}

/// The list-item prefix matcher (Python `is_list_item`). Matches numbering,
/// lettering, CJK enumerations, circled numbers, bullets, roman numerals, etc.
pub fn is_list_item(s: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(concat!(
            r"^(?:",
            r"\d+[.)]|\d+）|\d+．|\d+、|",
            r"\(\d+\)|\([a-zA-Z]\)|（\d+）|",
            r"[a-zA-Z][.)]|",
            r"[一二三四五六七八九十百千万]+、|",
            r"\([一二三四五六七八九十百千万]+\)|（[一二三四五六七八九十百千万]+）|",
            r"[\x{2460}-\x{2473}\x{3251}-\x{325F}]|",
            r"[•▪▫]|",
            r"[IVXLCDM]+\.|",
            r"-|\$|\t|[\[\(「【（]|",
            r"第[一二三四五六七八九十百千万][条节章]",
            r")\s*",
        ))
        .expect("valid list-item regex")
    });
    re.is_match(s)
}

/// Decide whether `str2` plausibly continues `str1` (Python `merge_rules`).
pub fn merge_rules(str1: &str, str2: &str) -> bool {
    if str1.is_empty() || str2.is_empty() {
        return false;
    }
    let trimmed: Vec<char> = str1.trim().chars().collect();
    if let Some(&last) = trimmed.last() {
        if TERMINATION.contains(&last) {
            return false;
        }
        if trimmed.len() > 1 {
            let second_last = trimmed[trimmed.len() - 2];
            if CLOSE.contains(&last) && TERMINATION.contains(&second_last) {
                return false;
            }
        }
    }
    if char_len(str1) < 10 {
        return false;
    }
    if is_list_item(str2) {
        return false;
    }
    if str1.contains('\t') || str2.contains('\t') {
        return false;
    }
    let d1 = str1
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false);
    let d2 = str2
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false);
    if d1 && d2 {
        return false;
    }
    true
}

/// Per-mille integer string for a `0..1` coordinate (Python `int(x*1000)`,
/// truncating toward zero).
fn permille(x: f64) -> String {
    ((x * 1000.0) as i64).to_string()
}

/// Select truncation candidates and build their judge blocks
/// (Python `filter_contd`).
pub fn filter_contd(blocks: &[WorkBlock]) -> Vec<JudgeBlock> {
    let mut potential: Vec<usize> = Vec::new();
    for i in 0..blocks.len() {
        if !matches!(blocks[i].kind.as_str(), "text" | "list_item") {
            continue;
        }
        for pos in (i + 1)..blocks.len() {
            let t = blocks[pos].kind.as_str();
            if t.contains("equation") || t.contains("title") {
                break;
            }
            if matches!(t, "text" | "list_item")
                && merge_rules(&blocks[i].content, &blocks[pos].content)
            {
                if !potential.contains(&i) {
                    potential.push(i);
                }
                if !potential.contains(&pos) {
                    potential.push(pos);
                }
                break;
            }
        }
    }

    potential
        .into_iter()
        .map(|i| {
            let b = &blocks[i];
            JudgeBlock {
                idx: i as i64,
                content: shorten(&b.content),
                page: b.page,
                bbox: [
                    permille(b.bbox[1]),
                    permille(b.bbox[0]),
                    permille(b.bbox[3]),
                    permille(b.bbox[2]),
                ],
            }
        })
        .collect()
}

/// Build the `Truncation Detection` prompt body for a chunk of judge blocks.
pub fn build_contd_prompt(chunk: &[JudgeBlock]) -> String {
    let mut lines = Vec::with_capacity(chunk.len());
    for b in chunk {
        lines.push(format!(
            "<|id|>{}<|page|>{}<|box|>{}<|content|>{}",
            b.idx,
            b.page,
            b.bbox.join(" "),
            b.content
        ));
    }
    format!("<image>\nTruncation Detection: {}", lines.join("\n"))
}

/// Parse `<|src_id|>S<|tgt_id|>T` pairs from a model response, after rewriting
/// the `<|from|>` / `<|to|>` aliases (Python `extract_label1` + the contd
/// replacement). Returns 0-based `(src, tgt)` positions.
pub fn parse_contd_pairs(response: &str) -> Vec<(i64, i64)> {
    let normalized = response
        .replace("<|from|>", "<|src_id|>")
        .replace("<|to|>", "<|tgt_id|>");
    parse_src_tgt(&normalized)
}

/// Shared `<|src_id|>…<|tgt_id|>…` line parser (Python `extract_label1`).
pub fn parse_src_tgt(s: &str) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for line in s.trim().split('\n') {
        if line.is_empty() {
            continue;
        }
        let Some((src_part, tgt_part)) = line.split_once("<|tgt_id|>") else {
            continue;
        };
        let Some(src) = src_part.split("<|src_id|>").nth(1) else {
            continue;
        };
        if let (Ok(s), Ok(t)) = (src.trim().parse::<i64>(), tgt_part.trim().parse::<i64>()) {
            out.push((s, t));
        }
    }
    out
}

/// Apply truncation pairs: `blocks[src].contd = tgt + 1` (Python apply loop).
pub fn apply_contd(blocks: &mut [WorkBlock], pairs: &[(i64, i64)]) {
    for &(src, tgt) in pairs {
        if let Some(b) = usize::try_from(src).ok().and_then(|i| blocks.get_mut(i)) {
            b.contd = tgt + 1;
        }
    }
}

/// Serialize accepted pairs back to the `<|src_id|>S<|tgt_id|>T` form
/// (Python `contd_output`).
pub fn contd_output(pairs: &[(i64, i64)]) -> String {
    pairs
        .iter()
        .map(|(s, t)| format!("<|src_id|>{s}<|tgt_id|>{t}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use popo_core::Bbox;
    use serde_json::Map;

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
            table_merge: None,
            cell_list: None,
            source: Map::new(),
        }
    }

    #[test]
    fn sentence_helpers() {
        assert_eq!(head_sentence("First. Second part"), "First.");
        assert_eq!(tail_sentence("First. Second part"), "Second part");
        assert_eq!(head_sentence("no terminator here"), "no terminator here");
    }

    #[test]
    fn list_item_detection() {
        assert!(is_list_item("1. item"));
        assert!(is_list_item("(a) thing"));
        assert!(is_list_item("• bullet"));
        assert!(is_list_item("第三条 ..."));
        assert!(is_list_item("- dash"));
        assert!(!is_list_item("ordinary sentence"));
    }

    #[test]
    fn merge_rules_basics() {
        // str1 ends mid-sentence, long enough, str2 not a list → continues.
        assert!(merge_rules(
            "this is a long unfinished clause",
            "that completes it"
        ));
        // ends with a period → not a continuation.
        assert!(!merge_rules("this is a complete sentence.", "next one"));
        // str1 too short.
        assert!(!merge_rules("short", "and more text here please"));
        // str2 is a list item.
        assert!(!merge_rules(
            "this is a long unfinished clause",
            "1. a list start"
        ));
        // both start with a digit.
        assert!(!merge_rules(
            "2020 was a long year of events",
            "2021 followed it"
        ));
    }

    #[test]
    fn filter_contd_links_adjacent_text() {
        let blocks = vec![
            block(
                1,
                1,
                "text",
                "this is a long unfinished clause",
                [0.1, 0.2, 0.3, 0.4],
            ),
            block(
                2,
                1,
                "text",
                "that completes the thought nicely",
                [0.1, 0.5, 0.3, 0.7],
            ),
            block(3, 1, "title", "A Heading", [0.0, 0.0, 1.0, 0.05]),
        ];
        let judges = filter_contd(&blocks);
        assert_eq!(judges.len(), 2);
        assert_eq!(judges[0].idx, 0);
        assert_eq!(judges[1].idx, 1);
        // bbox reordered to [top, left, bottom, right] * 1000
        assert_eq!(judges[0].bbox, ["200", "100", "400", "300"]);
    }

    #[test]
    fn parse_and_apply_pairs() {
        let pairs = parse_contd_pairs("<|src_id|>0<|tgt_id|>1\n<|from|>2<|to|>3\ngarbage");
        assert_eq!(pairs, vec![(0, 1), (2, 3)]);

        let mut blocks = vec![
            block(1, 1, "text", "a", [0.0; 4]),
            block(2, 1, "text", "b", [0.0; 4]),
        ];
        apply_contd(&mut blocks, &[(0, 1)]);
        assert_eq!(blocks[0].contd, 2); // tgt + 1
        assert_eq!(blocks[1].contd, -1);
    }

    #[test]
    fn prompt_format() {
        let judges = vec![JudgeBlock {
            idx: 0,
            content: "hello".into(),
            page: 1,
            bbox: ["10".into(), "20".into(), "30".into(), "40".into()],
        }];
        let p = build_contd_prompt(&judges);
        assert_eq!(
            p,
            "<image>\nTruncation Detection: <|id|>0<|page|>1<|box|>10 20 30 40<|content|>hello"
        );
    }
}
