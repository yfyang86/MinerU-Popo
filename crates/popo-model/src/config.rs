//! Model provider configuration.
//!
//! Mirrors the TOML format used to configure the engine:
//!
//! ```toml
//! [default]
//! provider = "deepseek"
//!
//! [providers.deepseek]
//! type = "openai"
//! api_base = "https://api.deepseek.com"
//! api_key = "sk-..."
//! model = "deepseek-v4-flash"
//!
//! [providers.claude]
//! type = "claude"
//! api_base = "https://api.anthropic.com"
//! api_key = "sk-ant-..."
//! model = "claude-sonnet-4-20250514"
//! ```
//!
//! `type = "openai"` works for any OpenAI-compatible endpoint (DeepSeek, Kimi,
//! Minimax, local vLLM, …); `type = "claude"` targets the Anthropic Messages
//! API.

use std::collections::BTreeMap;
use std::path::Path;

use popo_core::{Error, Result};
use serde::Deserialize;

/// The wire shape of a provider's `type` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// Any OpenAI-compatible Chat Completions endpoint.
    Openai,
    /// The Anthropic Messages API.
    Claude,
}

/// A single named provider entry under `[providers.<name>]`.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    /// Backend protocol to speak.
    #[serde(rename = "type")]
    pub kind: ProviderType,
    /// Base URL of the endpoint (no trailing path beyond the API root).
    pub api_base: String,
    /// API key. May be a literal, or `env:VAR` / `${VAR}` to read from the
    /// environment (see [`ProviderConfig::resolved_api_key`]).
    pub api_key: String,
    /// Default model id for this provider.
    pub model: String,
}

impl ProviderConfig {
    /// Resolve the API key, expanding `env:VAR` and `${VAR}` indirections from
    /// the process environment. Literal keys are returned unchanged.
    pub fn resolved_api_key(&self) -> Result<String> {
        resolve_secret(&self.api_key)
    }
}

/// The `[default]` table.
#[derive(Debug, Clone, Deserialize)]
pub struct DefaultSection {
    /// Name of the provider to use when none is specified.
    pub provider: String,
}

/// Top-level configuration: a default provider plus the provider table.
#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// Selects the default provider.
    pub default: DefaultSection,
    /// All configured providers, keyed by name.
    pub providers: BTreeMap<String, ProviderConfig>,
}

impl LlmConfig {
    /// Parse a configuration from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let config: LlmConfig =
            toml::from_str(s).map_err(|e| Error::Config(format!("invalid TOML: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Load a configuration from a TOML file on disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?;
        Self::from_toml_str(&text)
    }

    /// Check internal consistency: the default provider must exist.
    pub fn validate(&self) -> Result<()> {
        if self.providers.is_empty() {
            return Err(Error::Config("no [providers.*] entries defined".into()));
        }
        if !self.providers.contains_key(&self.default.provider) {
            return Err(Error::unknown_provider(
                &self.default.provider,
                self.provider_names(),
            ));
        }
        Ok(())
    }

    /// Look up a provider by name.
    pub fn provider(&self, name: &str) -> Result<&ProviderConfig> {
        self.providers
            .get(name)
            .ok_or_else(|| Error::unknown_provider(name, self.provider_names()))
    }

    /// The configured default provider entry.
    pub fn default_provider(&self) -> Result<(&str, &ProviderConfig)> {
        let name = self.default.provider.as_str();
        Ok((name, self.provider(name)?))
    }

    /// Resolve a provider by optional name, falling back to the default.
    pub fn resolve<'a>(&'a self, name: Option<&'a str>) -> Result<(&'a str, &'a ProviderConfig)> {
        match name {
            Some(n) => Ok((n, self.provider(n)?)),
            None => self.default_provider(),
        }
    }

    /// Names of all configured providers, sorted (the map is a BTreeMap).
    pub fn provider_names(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }
}

/// Expand a secret reference. Supports `env:VAR` and `${VAR}`; anything else is
/// treated as a literal value.
fn resolve_secret(raw: &str) -> Result<String> {
    let var = if let Some(rest) = raw.strip_prefix("env:") {
        rest
    } else if let Some(inner) = raw.strip_prefix("${").and_then(|s| s.strip_suffix('}')) {
        inner
    } else {
        return Ok(raw.to_string());
    };
    std::env::var(var)
        .map_err(|_| Error::Config(format!("environment variable {var:?} is not set")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact configuration shape supplied by the project.
    const SAMPLE: &str = r#"
[default]
provider = "deepseek"

[providers.openai]
type = "openai"
api_base = "https://api.openai.com/v1"
api_key = "sk-YOUR-KEY"
model = "gpt-4o"

[providers.claude]
type = "claude"
api_base = "https://api.anthropic.com"
api_key = "sk-ant-YOUR-KEY"
model = "claude-sonnet-4-20250514"

[providers.deepseek]
type = "openai"
api_base = "https://api.deepseek.com"
api_key = "sk-affbaeDEMO-KEY"
model = "deepseek-v4-flash"

[providers.local_vllm]
type = "openai"
api_base = "http://localhost:8000/v1"
api_key = "dummy"
model = "Qwen/Qwen2.5-72B-Instruct"
"#;

    #[test]
    fn parses_sample_config() {
        let cfg = LlmConfig::from_toml_str(SAMPLE).expect("sample config should parse");
        assert_eq!(cfg.default.provider, "deepseek");
        assert_eq!(cfg.providers.len(), 4);

        let (name, ds) = cfg.default_provider().unwrap();
        assert_eq!(name, "deepseek");
        assert_eq!(ds.kind, ProviderType::Openai);
        assert_eq!(ds.api_base, "https://api.deepseek.com");
        assert_eq!(ds.model, "deepseek-v4-flash");

        assert_eq!(cfg.provider("claude").unwrap().kind, ProviderType::Claude);
    }

    #[test]
    fn resolve_falls_back_to_default() {
        let cfg = LlmConfig::from_toml_str(SAMPLE).unwrap();
        assert_eq!(cfg.resolve(None).unwrap().0, "deepseek");
        assert_eq!(cfg.resolve(Some("openai")).unwrap().0, "openai");
    }

    #[test]
    fn unknown_provider_lists_known_ones() {
        let cfg = LlmConfig::from_toml_str(SAMPLE).unwrap();
        let err = cfg.provider("nope").unwrap_err().to_string();
        assert!(err.contains("claude"));
        assert!(err.contains("deepseek"));
    }

    #[test]
    fn default_provider_must_exist() {
        let bad = r#"
[default]
provider = "ghost"
[providers.real]
type = "openai"
api_base = "http://x"
api_key = "k"
model = "m"
"#;
        assert!(LlmConfig::from_toml_str(bad).is_err());
    }

    #[test]
    fn secret_indirection_from_env() {
        std::env::set_var("POPO_TEST_KEY", "secret-123");
        assert_eq!(resolve_secret("env:POPO_TEST_KEY").unwrap(), "secret-123");
        assert_eq!(resolve_secret("${POPO_TEST_KEY}").unwrap(), "secret-123");
        assert_eq!(resolve_secret("sk-literal").unwrap(), "sk-literal");
        assert!(resolve_secret("env:POPO_DEFINITELY_UNSET").is_err());
    }
}
