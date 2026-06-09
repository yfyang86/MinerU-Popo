//! Image-text association subtask: link images/tables to their captions,
//! footnotes, and owning blocks.
//!
//! Port of the Python `filter_image` / `check_overlap`, the
//! `Image-Text Correlation Analysis` prompt, and `extract_label1` parsing.
//! Visual blocks fully contained in a large `image_block` are linked directly
//! (the `large_block_linking`); the rest are sent to the model.

use crate::block::WorkBlock;
use crate::chunk::Paged;

/// A candidate visual block for association analysis. Like the truncation judge
/// block but carries a `kind` (the model-facing `<|type|>`).
#[derive(Debug, Clone)]
pub struct ImageJudge {
    /// 0-based block position (model-facing id).
    pub idx: i64,
    /// Normalized type shown to the model.
    pub kind: String,
    /// Content preview (`None` for images/tables).
    pub content: String,
    /// 1-based page.
    pub page: i64,
    /// Bbox as `[top, left, bottom, right]` per-mille integer strings.
    pub bbox: [String; 4],
}

impl Paged for ImageJudge {
    fn page(&self) -> i64 {
        self.page
    }
}

fn permille(x: f64) -> String {
    ((x * 1000.0) as i64).to_string()
}

fn char_take(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Treat `seal` as `image` everywhere (Python mutates the block type).
fn effective_kind(kind: &str) -> &str {
    if kind == "seal" {
        "image"
    } else {
        kind
    }
}

/// Types eligible to be sent for association analysis.
fn is_visual(kind: &str) -> bool {
    matches!(
        effective_kind(kind),
        "image_block"
            | "image"
            | "table"
            | "chart"
            | "table_footnote"
            | "image_footnote"
            | "table_caption"
            | "image_caption"
            | "title"
            | "TOC-title"
            | "section-title"
            | "figure"
            | "fig-title"
            | "fig-caption"
            | "tab-title"
            | "tab-caption"
    )
}

/// Types that may be absorbed into a containing `image_block`.
fn is_overlap_eligible(kind: &str) -> bool {
    matches!(
        effective_kind(kind),
        "image"
            | "chart"
            | "image_footnote"
            | "image_caption"
            | "figure"
            | "fig-title"
            | "fig-caption"
    )
}

/// Normalize a source type to the model-facing type (Python `check_overlap`).
fn normalize_type(kind: &str) -> &str {
    match effective_kind(kind) {
        "image_block" | "chart" | "figure" => "image",
        "title" | "TOC-title" | "section-title" => "title",
        "fig-title" | "fig-caption" => "image_caption",
        "tab-title" | "tab-caption" => "table_caption",
        other => other,
    }
}

/// `inner` is contained within `outer` allowing a 5% slack (Python overlap test).
fn contained(inner: &[f64; 4], outer: &[f64; 4]) -> bool {
    inner[0] >= 0.95 * outer[0]
        && inner[1] >= 0.95 * outer[1]
        && inner[2] <= 1.05 * outer[2]
        && inner[3] <= 1.05 * outer[3]
}

/// Select visual judge blocks and the direct `image_block` containment links.
///
/// Returns `(judges, large_block_linking)` where each linking entry is
/// `(visual_pos, image_block_pos)` (0-based positions in `blocks`).
pub fn filter_image(blocks: &[WorkBlock]) -> (Vec<ImageJudge>, Vec<(usize, usize)>) {
    let visual: Vec<usize> = (0..blocks.len())
        .filter(|&i| is_visual(&blocks[i].kind))
        .collect();
    let large: Vec<usize> = (0..blocks.len())
        .filter(|&i| blocks[i].kind == "image_block")
        .collect();

    let mut judges = Vec::new();
    let mut linking = Vec::new();
    for &i in &visual {
        let b = &blocks[i];
        let mut absorbed = false;
        if is_overlap_eligible(&b.kind) {
            for &j in &large {
                if contained(&b.bbox, &blocks[j].bbox) {
                    linking.push((i, j));
                    absorbed = true;
                    break;
                }
            }
        }
        if absorbed {
            continue;
        }
        let kind = normalize_type(&b.kind).to_string();
        let content = if matches!(kind.as_str(), "image" | "table") {
            "None".to_string()
        } else {
            char_take(&b.content, 50)
        };
        judges.push(ImageJudge {
            idx: i as i64,
            kind,
            content,
            page: b.page,
            bbox: [
                permille(b.bbox[1]),
                permille(b.bbox[0]),
                permille(b.bbox[3]),
                permille(b.bbox[2]),
            ],
        });
    }
    (judges, linking)
}

/// Build the `Image-Text Correlation Analysis` prompt body for a chunk.
pub fn build_image_prompt(chunk: &[ImageJudge]) -> String {
    let lines: Vec<String> = chunk
        .iter()
        .map(|b| {
            format!(
                "<|id|>{}<|type|>{}<|page|>{}<|box|>{}<|content|>{}",
                b.idx,
                b.kind,
                b.page,
                b.bbox.join(" "),
                b.content
            )
        })
        .collect();
    format!(
        "<image>\nImage-Text Correlation Analysis: {}",
        lines.join("\n")
    )
}

/// Apply association pairs then the direct containment links
/// (`blocks[src].image = tgt + 1`), matching the Python order.
pub fn apply_image(blocks: &mut [WorkBlock], pairs: &[(i64, i64)], linking: &[(usize, usize)]) {
    for &(src, tgt) in pairs {
        if let Some(b) = usize::try_from(src).ok().and_then(|i| blocks.get_mut(i)) {
            b.image = tgt + 1;
        }
    }
    for &(i, j) in linking {
        if let Some(b) = blocks.get_mut(i) {
            b.image = j as i64 + 1;
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
            source: Map::new(),
        }
    }

    #[test]
    fn filter_image_emits_judges_with_none_content_for_visuals() {
        let blocks = vec![
            block(1, "image", "ignored", [0.1, 0.1, 0.9, 0.5]),
            block(
                1,
                "image_caption",
                "Figure 1: a plot",
                [0.1, 0.55, 0.9, 0.6],
            ),
            block(1, "text", "body", [0.1, 0.7, 0.9, 0.8]),
        ];
        let (judges, linking) = filter_image(&blocks);
        assert!(linking.is_empty());
        assert_eq!(judges.len(), 2); // text is not visual
        assert_eq!(judges[0].kind, "image");
        assert_eq!(judges[0].content, "None");
        assert_eq!(judges[1].kind, "image_caption");
        assert_eq!(judges[1].content, "Figure 1: a plot");
        assert_eq!(judges[0].bbox, ["100", "100", "500", "900"]);
    }

    #[test]
    fn contained_visuals_become_linking_not_judges() {
        let blocks = vec![
            block(1, "image_block", "", [0.1, 0.1, 0.9, 0.9]),
            block(1, "image", "", [0.2, 0.2, 0.8, 0.8]), // inside the image_block
        ];
        let (judges, linking) = filter_image(&blocks);
        assert_eq!(linking, vec![(1, 0)]);
        // The image_block itself is visual and not contained → still a judge.
        assert_eq!(judges.len(), 1);
        assert_eq!(judges[0].idx, 0);
        assert_eq!(judges[0].kind, "image");
    }

    #[test]
    fn apply_pairs_then_linking() {
        let mut blocks = vec![
            block(1, "image", "", [0.0; 4]),
            block(1, "image_caption", "c", [0.0; 4]),
            block(1, "image", "", [0.0; 4]),
        ];
        // caption (idx 1) links to image (idx 0); plus a containment link 2→0.
        apply_image(&mut blocks, &[(1, 0)], &[(2, 0)]);
        assert_eq!(blocks[1].image, 1); // tgt 0 + 1
        assert_eq!(blocks[2].image, 1); // linking j 0 + 1
        assert_eq!(blocks[0].image, -1);
    }

    #[test]
    fn seal_is_treated_as_image() {
        let blocks = vec![block(1, "seal", "x", [0.1, 0.1, 0.2, 0.2])];
        let (judges, _) = filter_image(&blocks);
        assert_eq!(judges.len(), 1);
        assert_eq!(judges[0].kind, "image");
        assert_eq!(judges[0].content, "None");
    }
}
