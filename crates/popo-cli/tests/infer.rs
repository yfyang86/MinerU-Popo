//! End-to-end test of `popo infer` on a document that triggers no model calls
//! (single page, text only), so it runs offline. The `local_vllm` provider has
//! a literal api key, so the client builds without environment secrets.

use std::process::Command;

#[test]
fn cli_infer_writes_doc_blocks() {
    let dir = std::env::temp_dir().join(format!("popo-infer-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let input = dir.join("in");
    let output = dir.join("out");
    std::fs::create_dir_all(&input).unwrap();
    std::fs::write(
        input.join("sample.json"),
        r#"{"input_label":"sample","pages":{"1":[
            {"type":"text","content":"this is a long unfinished clause","bbox":[0.1,0.1,0.9,0.2]},
            {"type":"text","content":"that completes the thought nicely","bbox":[0.1,0.3,0.9,0.4]}
        ]}}"#,
    )
    .unwrap();

    // Config from the repo root (two levels up from this crate's manifest dir).
    let config = format!("{}/../../popo.example.toml", env!("CARGO_MANIFEST_DIR"));
    let bin = env!("CARGO_BIN_EXE_popo");
    let out = Command::new(bin)
        .args(["infer", "--config", &config, "--provider", "local_vllm"])
        .arg("--input-dir")
        .arg(&input)
        .arg("--output-dir")
        .arg(&output)
        .output()
        .expect("run popo infer");

    assert!(
        out.status.success(),
        "non-zero exit: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let written = std::fs::read_to_string(output.join("sample.json")).unwrap();
    let blocks: serde_json::Value = serde_json::from_str(&written).unwrap();
    let arr = blocks.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["id"], 1);
    assert_eq!(arr[0]["contd"], -1);
    assert_eq!(arr[0]["level"], -1);
    assert_eq!(arr[0]["image"], -1);

    let _ = std::fs::remove_dir_all(&dir);
}
