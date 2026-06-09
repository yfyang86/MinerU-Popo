//! The high-level model client: builds backends from configuration and adds
//! shared concerns (a bounded concurrency permit and retry-with-backoff).

use std::sync::Arc;
use std::time::Duration;

use popo_core::{Error, Result};
use tokio::sync::Semaphore;

use crate::backend::{claude::ClaudeBackend, openai::OpenAiBackend, ModelBackend};
use crate::config::{LlmConfig, ProviderConfig, ProviderType};
use crate::message::{ChatRequest, ChatResponse};

/// Tunables for the client's shared behavior.
#[derive(Debug, Clone)]
pub struct ClientOptions {
    /// Maximum number of in-flight requests across all backends.
    pub max_concurrency: usize,
    /// Total attempts per request (1 = no retry).
    pub max_attempts: u32,
    /// Base backoff; doubles each retry.
    pub base_backoff: Duration,
    /// Per-request timeout.
    pub request_timeout: Duration,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            max_concurrency: 8,
            max_attempts: 4,
            base_backoff: Duration::from_millis(500),
            request_timeout: Duration::from_secs(120),
        }
    }
}

/// A model client bound to one resolved provider.
///
/// Cloning is cheap: the underlying backend, HTTP client, and semaphore are
/// shared via `Arc`.
#[derive(Clone)]
pub struct ModelClient {
    backend: Arc<dyn ModelBackend>,
    semaphore: Arc<Semaphore>,
    provider_name: String,
    options: ClientOptions,
}

impl ModelClient {
    /// Build a client for the named provider (or the configured default when
    /// `provider` is `None`), using default [`ClientOptions`].
    pub fn from_config(config: &LlmConfig, provider: Option<&str>) -> Result<Self> {
        Self::from_config_with(config, provider, ClientOptions::default())
    }

    /// Build a client with explicit options.
    pub fn from_config_with(
        config: &LlmConfig,
        provider: Option<&str>,
        options: ClientOptions,
    ) -> Result<Self> {
        let (name, pc) = config.resolve(provider)?;
        let http = reqwest::Client::builder()
            .timeout(options.request_timeout)
            .build()
            .map_err(|e| Error::Transport(format!("building HTTP client: {e}")))?;
        let backend = build_backend(http, pc)?;
        Ok(Self {
            backend,
            semaphore: Arc::new(Semaphore::new(options.max_concurrency)),
            provider_name: name.to_string(),
            options,
        })
    }

    /// The provider name this client is bound to.
    pub fn provider_name(&self) -> &str {
        &self.provider_name
    }

    /// The model id this client will request.
    pub fn model(&self) -> &str {
        self.backend.model()
    }

    /// Run a chat request, honoring the concurrency permit and retrying
    /// transient failures with exponential backoff.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| Error::Transport(format!("semaphore closed: {e}")))?;

        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.backend.chat(req).await {
                Ok(resp) => return Ok(resp),
                Err(e) if attempt < self.options.max_attempts && is_retryable(&e) => {
                    let backoff = self.options.base_backoff * 2u32.pow(attempt - 1);
                    tracing::warn!(
                        provider = %self.provider_name,
                        attempt,
                        backoff_ms = backoff.as_millis() as u64,
                        error = %e,
                        "model call failed; retrying"
                    );
                    tokio::time::sleep(backoff).await;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// Construct the concrete backend for a provider entry, resolving its secret.
pub fn build_backend(http: reqwest::Client, pc: &ProviderConfig) -> Result<Arc<dyn ModelBackend>> {
    let key = pc.resolved_api_key()?;
    let backend: Arc<dyn ModelBackend> = match pc.kind {
        ProviderType::Openai => Arc::new(OpenAiBackend::new(http, &pc.api_base, &key, &pc.model)),
        ProviderType::Claude => Arc::new(ClaudeBackend::new(http, &pc.api_base, &key, &pc.model)),
    };
    Ok(backend)
}

/// Whether an error is worth retrying. Transport failures (timeouts, resets)
/// and 429/5xx statuses are transient; 4xx (other than 429) and parse errors
/// are not.
fn is_retryable(err: &Error) -> bool {
    match err {
        Error::Transport(_) => true,
        Error::ModelStatus { status, .. } => *status == 429 || (500..600).contains(status),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        assert!(is_retryable(&Error::Transport("reset".into())));
        assert!(is_retryable(&Error::ModelStatus {
            status: 429,
            body: String::new()
        }));
        assert!(is_retryable(&Error::ModelStatus {
            status: 503,
            body: String::new()
        }));
        assert!(!is_retryable(&Error::ModelStatus {
            status: 400,
            body: String::new()
        }));
        assert!(!is_retryable(&Error::ModelResponse("bad json".into())));
    }

    #[test]
    fn builds_client_for_default_provider() {
        let cfg = LlmConfig::from_toml_str(
            r#"
[default]
provider = "ds"
[providers.ds]
type = "openai"
api_base = "https://api.deepseek.com"
api_key = "k"
model = "deepseek-v4-flash"
"#,
        )
        .unwrap();
        let client = ModelClient::from_config(&cfg, None).unwrap();
        assert_eq!(client.provider_name(), "ds");
        assert_eq!(client.model(), "deepseek-v4-flash");
    }
}
