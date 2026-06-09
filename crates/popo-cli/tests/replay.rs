//! End-to-end test of the CLI replay path: author a fixture with the library
//! fingerprint, then run the built `popo` binary against it with
//! `POPO_MODEL_REPLAY` and assert the recorded text is returned.

use std::process::Command;

use popo_model::{fingerprint, ChatRequest, Fixture, RecordedResponse, RecordedUsage};

#[test]
fn cli_chat_replays_recorded_fixture() {
    let model = "replay-test-model";
    let prompt = "ping";
    // Must match exactly how the CLI builds the request for `model chat`.
    let req = ChatRequest::user_prompt(prompt);
    let fp = fingerprint(model, &req);

    let dir = std::env::temp_dir().join(format!("popo-cli-replay-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fixture = Fixture {
        fingerprint: fp.clone(),
        model: model.to_string(),
        request: vec![format!("model:{model}"), "text:ping".into()],
        max_tokens: None,
        temperature: None,
        response: RecordedResponse {
            text: "pong-from-fixture".into(),
            usage: Some(RecordedUsage {
                input_tokens: 1,
                output_tokens: 1,
            }),
        },
    };
    std::fs::write(
        dir.join(format!("{fp}.json")),
        serde_json::to_string_pretty(&fixture).unwrap(),
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_popo");
    let output = Command::new(bin)
        .env("POPO_MODEL_REPLAY", &dir)
        // Config is irrelevant on the replay path, but the arg is global.
        .args(["model", "chat", prompt])
        .output()
        .expect("failed to run popo binary");

    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        output.status.success(),
        "non-zero exit: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("pong-from-fixture"), "stdout was: {stdout}");
}
