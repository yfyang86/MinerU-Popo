//! Anthropic Messages API backend.
//!
//! Translates the neutral [`ChatRequest`] into the `/v1/messages` schema:
//! system prompts are hoisted into the top-level `system` field, and image
//! parts use the `source: { type: "base64", ... }` form.

use async_trait::async_trait;
use popo_core::{Error, Result};
use serde_json::{json, Value};

use crate::backend::{join_url, ModelBackend};
use crate::message::{ChatRequest, ChatResponse, ContentPart, Role, Usage};

/// Anthropic API version pinned for the Messages API.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Claude requires `max_tokens`; used when the request leaves it unset.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// A configured Anthropic Messages client.
pub struct ClaudeBackend {
    http: reqwest::Client,
    api_base: String,
    api_key: String,
    model: String,
}

impl ClaudeBackend {
    /// Create a backend from a resolved endpoint, key, and model.
    pub fn new(http: reqwest::Client, api_base: &str, api_key: &str, model: &str) -> Self {
        Self {
            http,
            api_base: api_base.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

/// Build the `/v1/messages` request body. Pure and unit-testable.
pub(crate) fn build_body(model: &str, req: &ChatRequest) -> Value {
    // Anthropic carries system prompts out-of-band; collect them separately.
    let mut system_text = String::new();
    let mut messages: Vec<Value> = Vec::new();

    for m in &req.messages {
        if m.role == Role::System {
            for part in &m.content {
                if let ContentPart::Text(t) = part {
                    if !system_text.is_empty() {
                        system_text.push('\n');
                    }
                    system_text.push_str(t);
                }
            }
            continue;
        }
        let parts: Vec<Value> = m
            .content
            .iter()
            .map(|p| match p {
                ContentPart::Text(t) => json!({ "type": "text", "text": t }),
                ContentPart::Image {
                    media_type,
                    data_base64,
                } => json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data_base64
                    }
                }),
            })
            .collect();
        messages.push(json!({ "role": m.role.as_str(), "content": parts }));
    }

    let mut body = json!({
        "model": model,
        "max_tokens": req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": messages,
    });
    if !system_text.is_empty() {
        body["system"] = json!(system_text);
    }
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    body
}

/// Extract assistant text and usage from a Messages API response.
pub(crate) fn parse_body(v: &Value) -> Result<ChatResponse> {
    let blocks = v["content"]
        .as_array()
        .ok_or_else(|| Error::ModelResponse("missing content array".into()))?;
    let mut text = String::new();
    for b in blocks {
        if b.get("type").and_then(Value::as_str) == Some("text") {
            if let Some(t) = b.get("text").and_then(Value::as_str) {
                text.push_str(t);
            }
        }
    }
    let usage = v.get("usage").and_then(|u| {
        Some(Usage {
            input_tokens: u.get("input_tokens")?.as_u64().unwrap_or(0) as u32,
            output_tokens: u.get("output_tokens")?.as_u64().unwrap_or(0) as u32,
        })
    });
    Ok(ChatResponse { text, usage })
}

#[async_trait]
impl ModelBackend for ClaudeBackend {
    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let url = join_url(&self.api_base, "v1/messages");
        let body = build_body(&self.model, req);
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        if !status.is_success() {
            return Err(Error::ModelStatus {
                status: status.as_u16(),
                body: if text.len() > 2000 {
                    format!("{}…", &text[..2000])
                } else {
                    text
                },
            });
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| Error::ModelResponse(format!("invalid JSON: {e}")))?;
        parse_body(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ChatMessage, Role};

    #[test]
    fn hoists_system_and_sets_default_max_tokens() {
        let req = ChatRequest {
            messages: vec![
                ChatMessage::text(Role::System, "be terse"),
                ChatMessage::text(Role::User, "hi"),
            ],
            max_tokens: None,
            temperature: Some(0.5),
        };
        let body = build_body("claude-sonnet-4-20250514", &req);
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
        // System message must not appear in the messages array.
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["temperature"], 0.5);
    }

    #[test]
    fn image_uses_base64_source() {
        let msg = ChatMessage::text(Role::User, "look").with_image("image/png", "ZZ");
        let req = ChatRequest {
            messages: vec![msg],
            max_tokens: Some(64),
            temperature: None,
        };
        let body = build_body("m", &req);
        let img = &body["messages"][0]["content"][1];
        assert_eq!(img["type"], "image");
        assert_eq!(img["source"]["type"], "base64");
        assert_eq!(img["source"]["media_type"], "image/png");
        assert_eq!(img["source"]["data"], "ZZ");
    }

    #[test]
    fn parses_text_blocks_and_usage() {
        let v = json!({
            "content": [
                { "type": "text", "text": "part1 " },
                { "type": "text", "text": "part2" }
            ],
            "usage": { "input_tokens": 10, "output_tokens": 4 }
        });
        let r = parse_body(&v).unwrap();
        assert_eq!(r.text, "part1 part2");
        assert_eq!(r.usage.unwrap().input_tokens, 10);
    }
}
