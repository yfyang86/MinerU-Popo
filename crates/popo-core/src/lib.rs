//! Shared types and error definitions for the MinerU-Popo Rust engine.
//!
//! This crate holds cross-cutting primitives that other crates build on. For
//! Sprint 1 that is the central [`Error`] type and its [`Result`] alias; the
//! canonical document/block schema lands in a later sprint.

use thiserror::Error;

/// The crate-wide result alias used across the workspace.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for the engine.
///
/// Each crate maps its failures into one of these variants so that the CLI can
/// present a uniform error surface.
#[derive(Debug, Error)]
pub enum Error {
    /// Configuration could not be loaded or was invalid.
    #[error("configuration error: {0}")]
    Config(String),

    /// A referenced provider/profile was not found in the configuration.
    #[error("unknown provider {name:?}; known providers: {known}")]
    UnknownProvider {
        /// The provider name that was requested.
        name: String,
        /// Comma-separated list of providers that *do* exist.
        known: String,
    },

    /// A network/transport-level failure talking to a model endpoint.
    #[error("model transport error: {0}")]
    Transport(String),

    /// The model endpoint returned a non-success HTTP status.
    #[error("model endpoint returned status {status}: {body}")]
    ModelStatus {
        /// HTTP status code returned by the endpoint.
        status: u16,
        /// Response body (possibly truncated) for diagnostics.
        body: String,
    },

    /// The model response could not be parsed into the expected shape.
    #[error("could not parse model response: {0}")]
    ModelResponse(String),

    /// An I/O failure reading or writing local files.
    #[error("io error: {0}")]
    Io(String),
}

impl Error {
    /// Build an [`Error::UnknownProvider`] from the requested name and the set
    /// of providers that are available.
    pub fn unknown_provider<'a>(name: &str, known: impl IntoIterator<Item = &'a str>) -> Self {
        let mut names: Vec<&str> = known.into_iter().collect();
        names.sort_unstable();
        Error::UnknownProvider {
            name: name.to_string(),
            known: names.join(", "),
        }
    }
}
