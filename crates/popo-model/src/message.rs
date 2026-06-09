//! Backend-agnostic chat message types.
//!
//! These types are deliberately small and provider-neutral; each backend
//! translates them into its own wire format (OpenAI Chat Completions or the
//! Anthropic Messages API). Multimodal content is supported via inline,
//! base64-encoded images — matching how the Python pipeline feeds rendered PDF
//! pages to the Popo VLM.

/// The role of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// System / developer instruction.
    System,
    /// End-user turn.
    User,
    /// Assistant turn.
    Assistant,
}

impl Role {
    /// The OpenAI/Anthropic wire string for this role.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// A single piece of message content.
#[derive(Debug, Clone)]
pub enum ContentPart {
    /// Plain text.
    Text(String),
    /// An inline image, base64-encoded.
    Image {
        /// MIME type, e.g. `image/jpeg`.
        media_type: String,
        /// Base64-encoded image bytes (no data-URL prefix).
        data_base64: String,
    },
}

/// One message in a conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Who is speaking.
    pub role: Role,
    /// Ordered content parts (text and/or images).
    pub content: Vec<ContentPart>,
}

impl ChatMessage {
    /// Convenience constructor for a text-only message.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(text.into())],
        }
    }

    /// Append an inline base64 image part to this message.
    pub fn with_image(
        mut self,
        media_type: impl Into<String>,
        data_base64: impl Into<String>,
    ) -> Self {
        self.content.push(ContentPart::Image {
            media_type: media_type.into(),
            data_base64: data_base64.into(),
        });
        self
    }
}

/// A chat completion request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// Conversation turns in order.
    pub messages: Vec<ChatMessage>,
    /// Optional cap on generated tokens. Required by Claude; defaulted there if
    /// `None`.
    pub max_tokens: Option<u32>,
    /// Optional sampling temperature.
    pub temperature: Option<f32>,
}

impl ChatRequest {
    /// Build a single-user-turn request from a prompt string.
    pub fn user_prompt(prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::text(Role::User, prompt)],
            max_tokens: None,
            temperature: None,
        }
    }

    /// Set the max-tokens cap.
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }
}

/// Token accounting returned by the endpoint, when available.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    /// Prompt/input tokens.
    pub input_tokens: u32,
    /// Completion/output tokens.
    pub output_tokens: u32,
}

/// A chat completion response.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The assistant's text output (concatenated across content parts).
    pub text: String,
    /// Token usage, if the endpoint reported it.
    pub usage: Option<Usage>,
}
