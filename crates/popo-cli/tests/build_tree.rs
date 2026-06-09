//! End-to-end test of `popo build-tree`: feed a small `doc_blocks` array and
//! assert the tree JSON and text preview are produced. Fully offline.

use std::process::Command;

#[test]
fn cli_build_tree_writes_tree_and_txt() {
    let dir = std::env::temp_dir().join(format!("popo-tree-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let input = dir.join("inf");
    let tree = dir.join("tree");
    let txt = dir.join("txt");
    std::fs::create_dir_all(&input).unwrap();
    std::fs::write(
        input.join("sample.json"),
        r#"[
            {"type":"title","content":"Chapter 1","bbox":[0,0,1,0.1],"page":1,"id":1,"contd":-1,"level":1,"image":-1},
            {"type":"text","content":"Body text.","bbox":[0,0.2,1,0.3],"page":1,"id":2,"contd":-1,"level":-1,"image":-1}
        ]"#,
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_popo");
    let out = Command::new(bin)
        .arg("build-tree")
        .arg("--input-dir")
        .arg(&input)
        .arg("--output-dir")
        .arg(&tree)
        .arg("--txt-dir")
        .arg(&txt)
        .output()
        .expect("run popo build-tree");
    assert!(
        out.status.success(),
        "non-zero exit: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let tree_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tree.join("sample.json")).unwrap()).unwrap();
    assert_eq!(tree_json["type"], "root");
    assert_eq!(tree_json["children"][0]["title"], "Chapter 1");
    assert_eq!(tree_json["children"][0]["content"], "Body text.");

    let preview = std::fs::read_to_string(txt.join("sample.txt")).unwrap();
    assert!(preview.contains("Chapter 1|Body text."));

    let _ = std::fs::remove_dir_all(&dir);
}
