//! Training-data generation for the four subtasks (port of the formatters in
//! `data_engine/add_link.py`).
//!
//! Each subtask's *human* prompt is identical to the inference prompt, and the
//! *gpt* response is the inference output serialization — so this crate reuses
//! `popo-infer`'s builders directly (the Sprint 8 "one implementation, two
//! backends" unification) and only adds the training-item envelope plus the
//! table-merge training format, which differs from inference.

use popo_infer::image::{build_image_prompt, ImageJudge};
use popo_infer::text::{build_contd_prompt, contd_output, JudgeBlock};
use popo_infer::title::{build_title_prompt, title_output, LevelPair};
use serde_json::{json, Value};

/// Wrap a `(human, gpt)` pair into the benchmark training item.
fn training_item(human: String, gpt: String, image: &str) -> Value {
    json!({
        "image": image,
        "conversations": [
            { "from": "human", "value": human },
            { "from": "gpt", "value": gpt },
        ],
        "type": "grounding",
    })
}

/// Text-truncation training case (`contd_format`).
pub fn contd_case(chunk: &[JudgeBlock], pairs: &[(i64, i64)], image: &str) -> Value {
    training_item(build_contd_prompt(chunk), contd_output(pairs), image)
}

/// Title-hierarchy training case (`title_format`).
pub fn title_case(chunk: &[JudgeBlock], results: &[LevelPair], image: &str) -> Value {
    training_item(build_title_prompt(chunk), title_output(results), image)
}

/// Image-text association training case (`image_format`).
pub fn image_case(chunk: &[ImageJudge], pairs: &[(i64, i64)], image: &str) -> Value {
    training_item(build_image_prompt(chunk), contd_output(pairs), image)
}

/// A table-merge judge line for the training prompt.
#[derive(Debug, Clone)]
pub struct TableJudge {
    /// 0-based block position.
    pub idx: i64,
    /// 1-based page.
    pub page: i64,
    /// Column count.
    pub col_count: i64,
    /// Caption text.
    pub caption: String,
    /// Last rows of the upper table (serialized as JSON).
    pub last_rows: Value,
    /// First rows of the lower table (serialized as JSON).
    pub first_rows: Value,
}

/// A table-merge training result.
#[derive(Debug, Clone)]
pub struct TableResult {
    /// Source (upper) block id.
    pub src: i64,
    /// Target (lower) block id.
    pub tgt: i64,
    /// Merge decision (0/1).
    pub merge: i64,
    /// Per-cell merge flags.
    pub cell_list: Vec<i64>,
}

/// Table-merge training case (`table_merge_format`) — its prompt format differs
/// from the inference table-merge prompt.
pub fn table_merge_case(blocks: &[TableJudge], results: &[TableResult], image: &str) -> Value {
    let input: Vec<String> = blocks
        .iter()
        .map(|b| {
            format!(
                "<|id|>{}<|page|>{}<|col_count|>{}<|caption|>{}<|last_rows|>{}<|first_rows|>{}",
                b.idx,
                b.page,
                b.col_count,
                b.caption,
                serde_json::to_string(&b.last_rows).unwrap_or_default(),
                serde_json::to_string(&b.first_rows).unwrap_or_default(),
            )
        })
        .collect();
    let output: Vec<String> = results
        .iter()
        .map(|r| {
            let cells = r
                .cell_list
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "<|src_id|>{}<|tgt_id|>{}<|merge|>{}<|cell_list|>{}",
                r.src, r.tgt, r.merge, cells
            )
        })
        .collect();
    training_item(
        format!("<image>\nTable Merge Detection: {}", input.join("\n")),
        output.join("\n"),
        image,
    )
}

/// Extract a JSON value from an LLM response (`llm_generate_json`): prefer a
/// fenced ```json block, else the outermost `[...]`, else the whole string.
pub fn extract_json(output: &str) -> Option<Value> {
    let candidate = if let Some(fenced) = extract_fenced_json(output) {
        fenced
    } else if let (Some(start), Some(end)) = (output.find('['), output.rfind(']')) {
        if end > start {
            output[start..=end].to_string()
        } else {
            output.to_string()
        }
    } else {
        output.to_string()
    };
    serde_json::from_str(&candidate).ok()
}

fn extract_fenced_json(output: &str) -> Option<String> {
    let start = output.find("```json")? + "```json".len();
    let rest = &output[start..];
    let end = rest.find("```")?;
    Some(rest[..end].trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judge(idx: i64) -> JudgeBlock {
        JudgeBlock {
            idx,
            content: format!("c{idx}"),
            page: 1,
            bbox: ["10".into(), "20".into(), "30".into(), "40".into()],
        }
    }

    #[test]
    fn contd_case_reuses_inference_prompt() {
        let chunk = vec![judge(0), judge(1)];
        let item = contd_case(&chunk, &[(0, 1)], "doc.jpg");
        assert_eq!(item["image"], "doc.jpg");
        assert_eq!(item["type"], "grounding");
        assert!(item["conversations"][0]["value"]
            .as_str()
            .unwrap()
            .starts_with("<image>\nTruncation Detection: <|id|>0"));
        assert_eq!(item["conversations"][1]["value"], "<|src_id|>0<|tgt_id|>1");
    }

    #[test]
    fn title_case_serializes_levels() {
        let chunk = vec![judge(0)];
        let item = title_case(&chunk, &[LevelPair { id: 0, level: 1 }], "d.jpg");
        assert!(item["conversations"][0]["value"]
            .as_str()
            .unwrap()
            .starts_with("<image>\nTitle Level Analysis:"));
        assert_eq!(item["conversations"][1]["value"], "<|id|>0<|level|>1");
    }

    #[test]
    fn table_merge_case_format() {
        let blocks = vec![TableJudge {
            idx: 3,
            page: 2,
            col_count: 4,
            caption: "Table 1".into(),
            last_rows: json!([["a", "b"]]),
            first_rows: json!([["c", "d"]]),
        }];
        let results = vec![TableResult {
            src: 3,
            tgt: 4,
            merge: 1,
            cell_list: vec![0, 1, 1],
        }];
        let item = table_merge_case(&blocks, &results, "d.jpg");
        let human = item["conversations"][0]["value"].as_str().unwrap();
        assert!(human.contains("<|col_count|>4<|caption|>Table 1"));
        assert_eq!(
            item["conversations"][1]["value"],
            "<|src_id|>3<|tgt_id|>4<|merge|>1<|cell_list|>0,1,1"
        );
    }

    #[test]
    fn extract_json_variants() {
        assert_eq!(extract_json("```json\n[1, 2]\n```").unwrap(), json!([1, 2]));
        assert_eq!(extract_json("prose [3, 4] tail").unwrap(), json!([3, 4]));
        assert!(extract_json("not json").is_none());
    }
}
