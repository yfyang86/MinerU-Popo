//! OpenAI-compatible Chat Completions backend.
//!
//! Works against any endpoint that speaks the OpenAI `/chat/completions`
//! schema: OpenAI itself, DeepSeek, Kimi/Moonshot, Minimax, and local vLLM.

use async_trait::async_trait;
use popo_core::{Error, Result};
use serde_json::{json, Value};

use crate::backend::{join_url, ModelBackend};
use crate::message::{ChatRequest, ChatResponse, ContentPart, Usage};

/// A configured OpenAI-compatible client.
pub struct OpenAiBackend {
    http: reqwest::Client,
    api_base: String,
    api_key: String,
    model: String,
}

impl OpenAiBackend {
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

/// Build the `/chat/completions` request body. Pure and side-effect free so it
/// can be unit-tested without a network.
pub(crate) fn build_body(model: &str, req: &ChatRequest) -> Value {
    let messages: Vec<Value> =
        req.messages
            .iter()
            .map(|m| {
                // Text-only messages use the simple string form for maximum
                // compatibility; mixed content uses the typed-parts array.
                let only_text =
                    m.content.len() == 1 && matches!(m.content[0], ContentPart::Text(_));
                if only_text {
                    if let ContentPart::Text(t) = &m.content[0] {
                        return json!({ "role": m.role.as_str(), "content": t });
                    }
                }
                let parts: Vec<Value> = m
                .content
                .iter()
                .map(|p| match p {
                    ContentPart::Text(t) => json!({ "type": "text", "text": t }),
                    ContentPart::Image { media_type, data_base64 } => json!({
                        "type": "image_url",
                        "image_url": { "url": format!("data:{media_type};base64,{data_base64}") }
                    }),
                })
                .collect();
                json!({ "role": m.role.as_str(), "content": parts })
            })
            .collect();

    let mut body = json!({ "model": model, "messages": messages });
    if let Some(mt) = req.max_tokens {
        body["max_tokens"] = json!(mt);
    }
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    body
}

/// Extract the assistant text and usage from a Chat Completions response.
pub(crate) fn parse_body(v: &Value) -> Result<ChatResponse> {
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| Error::ModelResponse("missing choices[0].message.content".into()))?
        .to_string();
    let usage = v.get("usage").and_then(|u| {
        Some(Usage {
            input_tokens: u.get("prompt_tokens")?.as_u64().unwrap_or(0) as u32,
            output_tokens: u.get("completion_tokens")?.as_u64().unwrap_or(0) as u32,
        })
    });
    Ok(ChatResponse { text, usage })
}

#[async_trait]
impl ModelBackend for OpenAiBackend {
    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let url = join_url(&self.api_base, "chat/completions");
        let body = build_body(&self.model, req);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
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
                body: truncate(&text, 2000),
            });
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| Error::ModelResponse(format!("invalid JSON: {e}")))?;
        parse_body(&value)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ChatMessage, Role};

    #[test]
    fn text_only_uses_string_content() {
        let req = ChatRequest::user_prompt("hello").with_temperature(1.0);
        let body = build_body("gpt-4o", &req);
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["temperature"], 1.0);
    }

    #[test]
    fn image_message_uses_parts_array() {
        let msg = ChatMessage::text(Role::User, "describe").with_image("image/jpeg", "AAAA");
        let req = ChatRequest {
            messages: vec![msg],
            max_tokens: Some(128),
            temperature: None,
        };
        let body = build_body("m", &req);
        let parts = &body["messages"][0]["content"];
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/jpeg;base64,AAAA");
        assert_eq!(body["max_tokens"], 128);
    }

    #[test]
    fn parses_response_and_usage() {
        let v = json!({
            "choices": [{ "message": { "content": "hi there" } }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 2 }
        });
        let r = parse_body(&v).unwrap();
        assert_eq!(r.text, "hi there");
        let u = r.usage.unwrap();
        assert_eq!(u.input_tokens, 3);
        assert_eq!(u.output_tokens, 2);
    }
}
