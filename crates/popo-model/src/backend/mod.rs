//! Model backends and the [`ModelBackend`] trait they implement.
//!
//! All subtask code in the engine talks to models through this trait, so a new
//! protocol (or a native in-process runtime in a later sprint) can be added
//! without touching callers.

use async_trait::async_trait;
use popo_core::Result;

use crate::message::{ChatRequest, ChatResponse};

pub mod claude;
pub mod openai;

/// A chat-capable model endpoint.
#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// The default model id this backend will use.
    fn model(&self) -> &str;

    /// Issue a single chat completion.
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse>;
}

/// Join an API base with a relative path, collapsing the boundary slash so that
/// both `https://host` and `https://host/` produce one separator.
pub(crate) fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

#[cfg(test)]
mod tests {
    use super::join_url;

    #[test]
    fn join_url_normalizes_slashes() {
        assert_eq!(
            join_url("http://h/v1", "chat/completions"),
            "http://h/v1/chat/completions"
        );
        assert_eq!(
            join_url("http://h/v1/", "/chat/completions"),
            "http://h/v1/chat/completions"
        );
        assert_eq!(
            join_url("https://api.anthropic.com", "v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );
    }
}
