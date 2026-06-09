//! End-to-end test of `popo eval`: a MinerU document whose titles match the
//! ground-truth prompt should score TEDS 1.0. Fully offline.

use std::process::Command;

#[test]
fn cli_eval_scores_matching_titles() {
    let dir = std::env::temp_dir().join(format!("popo-eval-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let input = dir.join("in");
    let vlm = input.join("sample").join("vlm");
    std::fs::create_dir_all(&vlm).unwrap();
    std::fs::write(
        vlm.join("sample_content_list.json"),
        r#"[
            {"type":"text","text":"Chapter One","bbox":[100,100,900,200],"page_idx":0,"text_level":1},
            {"type":"text","text":"Section A","bbox":[100,300,900,360],"page_idx":0,"text_level":2}
        ]"#,
    )
    .unwrap();

    let gt = dir.join("gt.json");
    std::fs::write(
        &gt,
        r#"[{"image":"sample.jpg","conversations":[
            {"value":"<image>\nTitle Level Analysis: <|id|>1<|page|>1<|box|>100 100 200 900<|content|>Chapter One\n<|id|>2<|page|>1<|box|>300 100 360 900<|content|>Section A"},
            {"value":"<|id|>1<|level|>1\n<|id|>2<|level|>2"}
        ]}]"#,
    )
    .unwrap();

    let out = dir.join("out");
    let bin = env!("CARGO_BIN_EXE_popo");
    let status = Command::new(bin)
        .args(["eval", "--model", "mineru"])
        .arg("--input-dir")
        .arg(&input)
        .arg("--gt-json")
        .arg(&gt)
        .arg("--output-dir")
        .arg(&out)
        .output()
        .expect("run popo eval");
    assert!(
        status.status.success(),
        "non-zero exit: stderr={}",
        String::from_utf8_lossy(&status.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["usable_docs"], 1);
    assert_eq!(summary["avg_score_usable"], 1.0);

    let _ = std::fs::remove_dir_all(&dir);
}
