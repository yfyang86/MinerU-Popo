//! Multi-provider model client for the MinerU-Popo engine.
//!
//! The engine talks to language/vision models exclusively through this crate.
//! Providers are described in TOML ([`config`]); requests use a neutral message
//! model ([`message`]); and each protocol is a [`backend`] behind the
//! [`ModelBackend`] trait. [`ModelClient`] ties them together with provider
//! resolution, bounded concurrency, and retry-with-backoff.
//!
//! ```no_run
//! use popo_model::{LlmConfig, ModelClient, ChatRequest};
//!
//! # async fn run() -> popo_core::Result<()> {
//! let config = LlmConfig::from_path("popo.toml")?;
//! let client = ModelClient::from_config(&config, None)?; // default provider
//! let resp = client.chat(&ChatRequest::user_prompt("hello")).await?;
//! println!("{}", resp.text);
//! # Ok(())
//! # }
//! ```

pub mod backend;
pub mod client;
pub mod config;
pub mod message;

pub use backend::replay::{
    fingerprint, Fixture, RecordedResponse, RecordedUsage, RecordingBackend, ReplayBackend,
};
pub use backend::ModelBackend;
pub use client::{ClientOptions, ModelClient};
pub use config::{LlmConfig, ProviderConfig, ProviderType};
pub use message::{ChatMessage, ChatRequest, ChatResponse, ContentPart, Role, Usage};
