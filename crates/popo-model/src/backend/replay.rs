//! Deterministic record/replay backends.
//!
//! These backends make the engine's model interactions reproducible, which is
//! the foundation for golden-corpus parity tests:
//!
//! * [`RecordingBackend`] wraps a real backend and writes every exchange to a
//!   fixture directory, keyed by a stable fingerprint of the request.
//! * [`ReplayBackend`] serves responses purely from a fixture directory with no
//!   network access, so tests are hermetic and byte-stable.
//!
//! The fingerprint is a platform-stable FNV-1a hash over a canonical rendering
//! of `(model, messages, max_tokens, temperature)`, so the same logical request
//! always maps to the same fixture regardless of which concrete backend
//! produced it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use popo_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::backend::ModelBackend;
use crate::message::{ChatMessage, ChatRequest, ChatResponse, ContentPart, Usage};

/// Token usage as stored in a fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedUsage {
    /// Prompt/input tokens.
    pub input_tokens: u32,
    /// Completion/output tokens.
    pub output_tokens: u32,
}

impl From<&Usage> for RecordedUsage {
    fn from(u: &Usage) -> Self {
        Self {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
        }
    }
}

impl From<&RecordedUsage> for Usage {
    fn from(u: &RecordedUsage) -> Self {
        Self {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
        }
    }
}

/// The recorded response payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedResponse {
    /// Assistant text output.
    pub text: String,
    /// Token usage, if known.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage: Option<RecordedUsage>,
}

/// A single recorded request/response exchange, persisted as one JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixture {
    /// Stable fingerprint of the request (also the file stem).
    pub fingerprint: String,
    /// Model id the request targeted.
    pub model: String,
    /// Canonical, human-readable view of the request (for debugging/review).
    pub request: Vec<String>,
    /// Request sampling cap, if set.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_tokens: Option<u32>,
    /// Request temperature, if set.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub temperature: Option<f32>,
    /// The recorded response.
    pub response: RecordedResponse,
}

/// Render a message's content as canonical, fingerprint-stable strings. Image
/// bytes are reduced to `image:<media_type>:<hash>` so fixtures stay small
/// while still distinguishing different images.
fn canonical_parts(msg: &ChatMessage) -> Vec<String> {
    let mut out = Vec::with_capacity(msg.content.len());
    for part in &msg.content {
        match part {
            ContentPart::Text(t) => out.push(format!("text:{t}")),
            ContentPart::Image {
                media_type,
                data_base64,
            } => out.push(format!(
                "image:{media_type}:{:016x}",
                fnv1a(data_base64.as_bytes())
            )),
        }
    }
    out
}

/// Canonical lines for an entire request: one `role:` header per message
/// followed by its parts, then the sampling parameters.
fn canonical_request(model: &str, req: &ChatRequest) -> Vec<String> {
    let mut lines = vec![format!("model:{model}")];
    for msg in &req.messages {
        lines.push(format!("role:{}", msg.role.as_str()));
        lines.extend(canonical_parts(msg));
    }
    lines.push(format!(
        "max_tokens:{}",
        req.max_tokens.map(|n| n.to_string()).unwrap_or_default()
    ));
    lines.push(format!(
        "temperature:{}",
        req.temperature.map(|t| t.to_string()).unwrap_or_default()
    ));
    lines
}

/// Stable fingerprint for a `(model, request)` pair, as 16 hex chars.
pub fn fingerprint(model: &str, req: &ChatRequest) -> String {
    let joined = canonical_request(model, req).join("\n");
    format!("{:016x}", fnv1a(joined.as_bytes()))
}

/// 64-bit FNV-1a. Chosen over `DefaultHasher` because its output is fixed
/// across platforms and toolchain versions, which fixtures rely on.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// A backend that wraps another and records every exchange to a directory.
pub struct RecordingBackend {
    inner: Arc<dyn ModelBackend>,
    dir: PathBuf,
}

impl RecordingBackend {
    /// Wrap `inner`, writing fixtures into `dir` (created if missing).
    pub fn new(inner: Arc<dyn ModelBackend>, dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::Io(format!("creating fixture dir {}: {e}", dir.display())))?;
        Ok(Self { inner, dir })
    }

    fn write_fixture(&self, req: &ChatRequest, resp: &ChatResponse) -> Result<()> {
        let model = self.inner.model();
        let fp = fingerprint(model, req);
        let fixture = Fixture {
            fingerprint: fp.clone(),
            model: model.to_string(),
            request: canonical_request(model, req),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            response: RecordedResponse {
                text: resp.text.clone(),
                usage: resp.usage.as_ref().map(RecordedUsage::from),
            },
        };
        let path = self.dir.join(format!("{fp}.json"));
        let json = serde_json::to_string_pretty(&fixture)
            .map_err(|e| Error::Io(format!("serializing fixture: {e}")))?;
        std::fs::write(&path, json)
            .map_err(|e| Error::Io(format!("writing {}: {e}", path.display())))
    }
}

#[async_trait]
impl ModelBackend for RecordingBackend {
    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let resp = self.inner.chat(req).await?;
        if let Err(e) = self.write_fixture(req, &resp) {
            tracing::warn!(error = %e, "failed to record fixture");
        }
        Ok(resp)
    }
}

/// A hermetic backend that serves responses from recorded fixtures only.
pub struct ReplayBackend {
    model: String,
    fixtures: HashMap<String, RecordedResponse>,
}

impl ReplayBackend {
    /// Build from an in-memory set of fixtures, bound to `model`.
    pub fn new(model: impl Into<String>, fixtures: Vec<Fixture>) -> Self {
        let model = model.into();
        let map = fixtures
            .into_iter()
            .map(|f| (f.fingerprint, f.response))
            .collect();
        Self {
            model,
            fixtures: map,
        }
    }

    /// Load every `*.json` fixture in `dir`. The backend's model id is taken
    /// from the fixtures (they must all agree); an empty directory is an error.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let entries = std::fs::read_dir(dir)
            .map_err(|e| Error::Io(format!("reading fixture dir {}: {e}", dir.display())))?;
        let mut fixtures = Vec::new();
        let mut model: Option<String> = None;
        for entry in entries {
            let path = entry
                .map_err(|e| Error::Io(format!("reading dir entry: {e}")))?
                .path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|e| Error::Io(format!("reading {}: {e}", path.display())))?;
            let fixture: Fixture = serde_json::from_str(&text)
                .map_err(|e| Error::Io(format!("parsing {}: {e}", path.display())))?;
            match &model {
                Some(m) if m != &fixture.model => {
                    return Err(Error::Config(format!(
                        "fixture dir {} mixes models {m:?} and {:?}",
                        dir.display(),
                        fixture.model
                    )));
                }
                None => model = Some(fixture.model.clone()),
                _ => {}
            }
            fixtures.push(fixture);
        }
        let model = model
            .ok_or_else(|| Error::Config(format!("no fixtures found in {}", dir.display())))?;
        Ok(Self::new(model, fixtures))
    }

    /// Number of loaded fixtures.
    pub fn len(&self) -> usize {
        self.fixtures.len()
    }

    /// Whether no fixtures are loaded.
    pub fn is_empty(&self) -> bool {
        self.fixtures.is_empty()
    }
}

#[async_trait]
impl ModelBackend for ReplayBackend {
    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let fp = fingerprint(&self.model, req);
        match self.fixtures.get(&fp) {
            Some(r) => Ok(ChatResponse {
                text: r.text.clone(),
                usage: r.usage.as_ref().map(Usage::from),
            }),
            None => Err(Error::ModelResponse(format!(
                "no recorded fixture for request fingerprint {fp} (model {})",
                self.model
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ChatMessage, Role};

    fn req(prompt: &str) -> ChatRequest {
        ChatRequest::user_prompt(prompt)
    }

    #[test]
    fn fingerprint_is_stable_and_distinguishing() {
        let a1 = fingerprint("m", &req("hello"));
        let a2 = fingerprint("m", &req("hello"));
        let b = fingerprint("m", &req("world"));
        let c = fingerprint("other", &req("hello"));
        assert_eq!(a1, a2, "same input must yield same fingerprint");
        assert_ne!(a1, b, "different prompt must differ");
        assert_ne!(a1, c, "different model must differ");
        assert_eq!(a1.len(), 16);
    }

    #[test]
    fn image_bytes_affect_fingerprint() {
        let base = ChatMessage::text(Role::User, "look");
        let r1 = ChatRequest {
            messages: vec![base.clone().with_image("image/png", "AAAA")],
            max_tokens: None,
            temperature: None,
        };
        let r2 = ChatRequest {
            messages: vec![base.with_image("image/png", "BBBB")],
            max_tokens: None,
            temperature: None,
        };
        assert_ne!(fingerprint("m", &r1), fingerprint("m", &r2));
    }

    #[tokio::test]
    async fn replay_serves_recorded_response() {
        let request = req("hi");
        let fp = fingerprint("test-model", &request);
        let fixture = Fixture {
            fingerprint: fp,
            model: "test-model".into(),
            request: canonical_request("test-model", &request),
            max_tokens: None,
            temperature: None,
            response: RecordedResponse {
                text: "recorded reply".into(),
                usage: Some(RecordedUsage {
                    input_tokens: 2,
                    output_tokens: 3,
                }),
            },
        };
        let backend = ReplayBackend::new("test-model", vec![fixture]);
        let resp = backend.chat(&request).await.unwrap();
        assert_eq!(resp.text, "recorded reply");
        assert_eq!(resp.usage.unwrap().output_tokens, 3);
    }

    #[tokio::test]
    async fn replay_errors_on_unknown_request() {
        let backend = ReplayBackend::new("test-model", vec![]);
        let err = backend.chat(&req("unseen")).await.unwrap_err();
        assert!(err.to_string().contains("no recorded fixture"));
    }

    #[tokio::test]
    async fn record_then_replay_roundtrip_via_dir() {
        // A trivial stub backend to record from.
        struct Stub;
        #[async_trait]
        impl ModelBackend for Stub {
            fn model(&self) -> &str {
                "stub-model"
            }
            async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse> {
                Ok(ChatResponse {
                    text: "from stub".into(),
                    usage: None,
                })
            }
        }

        let dir = std::env::temp_dir().join(format!("popo-fix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let recorder = RecordingBackend::new(Arc::new(Stub), &dir).unwrap();
        let request = req("roundtrip");
        let recorded = recorder.chat(&request).await.unwrap();
        assert_eq!(recorded.text, "from stub");

        let replay = ReplayBackend::from_dir(&dir).unwrap();
        assert_eq!(replay.len(), 1);
        assert_eq!(replay.model(), "stub-model");
        let replayed = replay.chat(&request).await.unwrap();
        assert_eq!(replayed.text, "from stub");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
