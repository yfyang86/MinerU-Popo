//! Title-hierarchy subtask: assign heading levels, reconciled across chunks.
//!
//! Port of the Python `filter_title`, the `Title Level Analysis` prompt,
//! `extract_label2`, and the cross-chunk **bias synchronization** that keeps
//! heading levels consistent where consecutive chunks overlap.

use crate::block::WorkBlock;
use crate::text::JudgeBlock;
/// An `(id, level)` assignment from the model (Python label-2 pair).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelPair {
    /// 0-based block position (model-facing id).
    pub id: i64,
    /// Heading level.
    pub level: i64,
}

fn permille(x: f64) -> String {
    ((x * 1000.0) as i64).to_string()
}

fn char_take(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Title block kinds eligible for hierarchy analysis.
fn is_title_kind(kind: &str) -> bool {
    matches!(kind, "title" | "TOC-title" | "section-title")
}

/// Build judge blocks for the title blocks (Python `filter_title`).
pub fn filter_title(blocks: &[WorkBlock]) -> Vec<JudgeBlock> {
    blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| is_title_kind(&b.kind))
        .map(|(i, b)| JudgeBlock {
            idx: i as i64,
            content: char_take(&b.content, 50),
            page: b.page,
            bbox: [
                permille(b.bbox[1]),
                permille(b.bbox[0]),
                permille(b.bbox[3]),
                permille(b.bbox[2]),
            ],
        })
        .collect()
}

/// Build the `Title Level Analysis` prompt body for a chunk.
pub fn build_title_prompt(chunk: &[JudgeBlock]) -> String {
    let lines: Vec<String> = chunk
        .iter()
        .map(|b| {
            format!(
                "<|id|>{}<|page|>{}<|box|>{}<|content|>{}",
                b.idx,
                b.page,
                b.bbox.join(" "),
                b.content
            )
        })
        .collect();
    format!("<image>\nTitle Level Analysis: {}", lines.join("\n"))
}

/// Parse `<|id|>I<|level|>L` lines, keeping only non-negative levels
/// (Python `extract_label2`).
pub fn extract_label2(s: &str) -> Vec<LevelPair> {
    let mut out = Vec::new();
    for line in s.trim().split('\n') {
        if line.is_empty() {
            continue;
        }
        let Some((id_part, level_part)) = line.split_once("<|level|>") else {
            continue;
        };
        let Some(id_str) = id_part.split("<|id|>").nth(1) else {
            continue;
        };
        if let (Ok(id), Ok(level)) = (
            id_str.trim().parse::<i64>(),
            level_part.trim().parse::<i64>(),
        ) {
            if level >= 0 {
                out.push(LevelPair { id, level });
            }
        }
    }
    out
}

/// Reconcile a chunk's level assignments against the accumulated `results`,
/// then merge in the chunk's new blocks (Python title synchronization).
///
/// Overlapping blocks (seen in a prior chunk) pin their level to the earlier
/// value and contribute their offset to a bias; that averaged bias is then
/// subtracted from this chunk's *new* positive levels, aligning the chunk to the
/// document-wide frame.
pub fn synchronize(id_pairs: &mut [LevelPair], results: &mut Vec<LevelPair>) {
    let mut bias: Vec<i64> = Vec::new();
    for pair in id_pairs.iter_mut() {
        for exist in results.iter_mut() {
            if pair.id == exist.id {
                if pair.level < 0 || exist.level < 0 {
                    pair.level = -1;
                    exist.level = -1;
                } else {
                    bias.push(pair.level - exist.level);
                    pair.level = exist.level;
                }
                break;
            }
        }
    }

    let avg_bias = if bias.is_empty() {
        0
    } else {
        let mean = bias.iter().sum::<i64>() as f64 / bias.len() as f64;
        // Python `round` is banker's rounding (ties to even).
        mean.round_ties_even() as i64
    };

    for pair in id_pairs.iter() {
        let already = results.iter().any(|e| e.id == pair.id);
        if !already {
            let level = if pair.level > 0 {
                pair.level - avg_bias
            } else {
                pair.level
            };
            results.push(LevelPair { id: pair.id, level });
        }
    }
}

/// Serialize accumulated results to `<|id|>I<|level|>L` lines
/// (Python `title_output`).
pub fn title_output(results: &[LevelPair]) -> String {
    results
        .iter()
        .map(|p| format!("<|id|>{}<|level|>{}", p.id, p.level))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Apply level labels to blocks: `blocks[id].level = level` for each non-negative
/// label (Python apply loop, fed by `extract_label2(title_output)`).
pub fn apply_title(blocks: &mut [WorkBlock], output: &str) {
    for label in extract_label2(output) {
        if let Some(b) = usize::try_from(label.id)
            .ok()
            .and_then(|i| blocks.get_mut(i))
        {
            b.level = label.level;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use popo_core::Bbox;
    use serde_json::Map;

    fn block(page: i64, kind: &str, content: &str, bbox: Bbox) -> WorkBlock {
        WorkBlock {
            id: 0,
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
    fn filter_title_selects_titles_only() {
        let blocks = vec![
            block(1, "text", "body", [0.0; 4]),
            block(1, "title", "Heading One", [0.1, 0.2, 0.3, 0.4]),
        ];
        let judges = filter_title(&blocks);
        assert_eq!(judges.len(), 1);
        assert_eq!(judges[0].idx, 1);
        assert_eq!(judges[0].bbox, ["200", "100", "400", "300"]);
    }

    #[test]
    fn extract_label2_keeps_non_negative() {
        let pairs = extract_label2("<|id|>0<|level|>1\n<|id|>2<|level|>-1\n<|id|>3<|level|>0");
        assert_eq!(
            pairs,
            vec![LevelPair { id: 0, level: 1 }, LevelPair { id: 3, level: 0 }]
        );
    }

    #[test]
    fn synchronize_aligns_overlapping_chunk_levels() {
        // Chunk 1 establishes the frame.
        let mut results = Vec::new();
        let mut c1 = vec![LevelPair { id: 1, level: 1 }, LevelPair { id: 2, level: 2 }];
        synchronize(&mut c1, &mut results);
        assert_eq!(
            results,
            vec![LevelPair { id: 1, level: 1 }, LevelPair { id: 2, level: 2 }]
        );

        // Chunk 2 overlaps on id 2 but labels everything one level deeper.
        let mut c2 = vec![LevelPair { id: 2, level: 3 }, LevelPair { id: 3, level: 4 }];
        synchronize(&mut c2, &mut results);
        // id 2 pinned to its earlier level (2); the +1 bias is subtracted from
        // the new id 3 (4 → 3).
        assert_eq!(
            results,
            vec![
                LevelPair { id: 1, level: 1 },
                LevelPair { id: 2, level: 2 },
                LevelPair { id: 3, level: 3 },
            ]
        );
    }

    #[test]
    fn apply_title_sets_levels() {
        let mut blocks = vec![
            block(1, "title", "A", [0.0; 4]),
            block(1, "text", "b", [0.0; 4]),
            block(1, "title", "C", [0.0; 4]),
        ];
        apply_title(&mut blocks, "<|id|>0<|level|>1\n<|id|>2<|level|>2");
        assert_eq!(blocks[0].level, 1);
        assert_eq!(blocks[1].level, -1);
        assert_eq!(blocks[2].level, 2);
    }
}
