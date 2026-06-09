//! Inference stage: dynamic chunking and the four post-processing subtasks.
//!
//! Sprint 4 (part 1) ships the working-block model, the `adaptive_chunk`
//! dynamic chunker, and the **text-truncation** subtask end to end through the
//! model client. The title-hierarchy, image-text, and table-merge subtasks
//! land in subsequent iterations behind the same shape.
//!
//! Page-image rendering is abstracted by [`PageImageProvider`] so the subtask
//! logic is testable today (with [`NoImages`]) and gains real rendered pages
//! when the PDF stage exists.

pub mod block;
pub mod chunk;
pub mod text;
pub mod title;

use popo_core::Result;
use popo_model::{ChatMessage, ChatRequest, ModelClient, Role};

pub use block::{build_doc_blocks, WorkBlock};
pub use chunk::{adaptive_chunk, Paged, Range};

/// Supplies a rendered, base64-encoded image for a set of pages.
pub trait PageImageProvider {
    /// Return `(media_type, base64_data)` for `pages`, or `None` to send a
    /// text-only prompt.
    fn image_for_pages(&self, pages: &[i64]) -> Option<(String, String)>;
}

/// A provider that never supplies images (text-only prompts).
pub struct NoImages;

impl PageImageProvider for NoImages {
    fn image_for_pages(&self, _pages: &[i64]) -> Option<(String, String)> {
        None
    }
}

/// The distinct pages present in a chunk, restricted to the chunk's range and
/// sorted (Python `pages = sorted([... if rng[0] <= page <= rng[1]])`).
fn chunk_pages(chunk: &[text::JudgeBlock], rng: &Range) -> Vec<i64> {
    let mut pages: Vec<i64> = chunk
        .iter()
        .map(|b| b.page)
        .filter(|&p| rng[0] <= p && p <= rng[1])
        .collect();
    pages.sort_unstable();
    pages.dedup();
    pages
}

/// Run the text-truncation subtask over `blocks`, mutating their `contd` links
/// in place and returning the serialized `contd_output`.
///
/// Mirrors the Python flow: filter candidates, chunk by page, prompt the model
/// per chunk (with the chunk's page image when available), accumulate unique
/// `(src, tgt)` pairs, then apply them.
pub async fn run_text_truncation(
    client: &ModelClient,
    blocks: &mut [WorkBlock],
    images: &dyn PageImageProvider,
) -> Result<String> {
    let judges = text::filter_contd(blocks);
    let (ranges, chunks) = adaptive_chunk(&judges, 50, 1);

    let mut pairs: Vec<(i64, i64)> = Vec::new();
    for (rng, chunk) in ranges.iter().zip(chunks.iter()) {
        let pages = chunk_pages(chunk, rng);
        let prompt = text::build_contd_prompt(chunk);
        let response = chat_for_chunk(client, prompt, &pages, images).await?;
        for pair in text::parse_contd_pairs(&response) {
            if !pairs.contains(&pair) {
                pairs.push(pair);
            }
        }
    }

    text::apply_contd(blocks, &pairs);
    Ok(text::contd_output(&pairs))
}

/// Send one chunk's prompt (with its page image when available) and return the
/// raw model response text.
async fn chat_for_chunk(
    client: &ModelClient,
    prompt: String,
    pages: &[i64],
    images: &dyn PageImageProvider,
) -> Result<String> {
    let mut message = ChatMessage::text(Role::User, prompt);
    if let Some((media_type, data)) = images.image_for_pages(pages) {
        message = message.with_image(media_type, data);
    }
    let req = ChatRequest {
        messages: vec![message],
        max_tokens: None,
        temperature: None,
    };
    Ok(client.chat(&req).await?.text)
}

/// Run the title-hierarchy subtask over `blocks`, setting their `level` in place
/// and returning the serialized `title_output`.
///
/// Chunks are processed **sequentially** so the cross-chunk bias
/// synchronization (which mutates the accumulated results) sees them in order,
/// matching the Python coroutine semantics.
pub async fn run_title_hierarchy(
    client: &ModelClient,
    blocks: &mut [WorkBlock],
    images: &dyn PageImageProvider,
) -> Result<String> {
    let judges = title::filter_title(blocks);
    let (ranges, chunks) = adaptive_chunk(&judges, 50, 1);

    let mut results: Vec<title::LevelPair> = Vec::new();
    for (rng, chunk) in ranges.iter().zip(chunks.iter()) {
        let pages = chunk_pages(chunk, rng);
        let prompt = title::build_title_prompt(chunk);
        let response = chat_for_chunk(client, prompt, &pages, images).await?;
        let mut id_pairs = title::extract_label2(&response);
        title::synchronize(&mut id_pairs, &mut results);
    }

    let output = title::title_output(&results);
    title::apply_title(blocks, &output);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use popo_model::{ClientOptions, Fixture, RecordedResponse, ReplayBackend};
    use serde_json::json;
    use std::sync::Arc;

    /// A whole-document run of the text-truncation subtask against a replay
    /// backend, asserting the `contd` link is applied.
    #[tokio::test]
    async fn text_truncation_end_to_end_via_replay() {
        // The two linkable blocks span pages 1 and 4 so the chunker keeps them
        // (its trailing-chunk guard drops spans of <= 2 pages).
        let pages = json!({
            "1": [
                { "type": "text", "content": "this is a long unfinished clause", "bbox": [0.1, 0.2, 0.3, 0.4] }
            ],
            "4": [
                { "type": "text", "content": "that completes the thought nicely", "bbox": [0.1, 0.5, 0.3, 0.7] }
            ]
        });
        let mut blocks = build_doc_blocks(pages.as_object().unwrap());

        // Reconstruct the exact prompt the runner will send, so the replay
        // fixture matches by fingerprint.
        let judges = text::filter_contd(&blocks);
        let (ranges, chunks) = adaptive_chunk(&judges, 50, 1);
        assert_eq!(
            chunks.len(),
            1,
            "candidates spanning pages 1..4 → one chunk"
        );
        let prompt = text::build_contd_prompt(&chunks[0]);
        let _ = &ranges;

        let model = "popo-test";
        let req = ChatRequest {
            messages: vec![ChatMessage::text(Role::User, prompt)],
            max_tokens: None,
            temperature: None,
        };
        let fp = popo_model::fingerprint(model, &req);
        let fixture = Fixture {
            fingerprint: fp,
            model: model.into(),
            request: vec![],
            max_tokens: None,
            temperature: None,
            response: RecordedResponse {
                text: "<|src_id|>0<|tgt_id|>1".into(),
                usage: None,
            },
        };
        let backend = ReplayBackend::new(model, vec![fixture]);
        let client =
            ModelClient::from_backend("replay", Arc::new(backend), ClientOptions::default());

        let output = run_text_truncation(&client, &mut blocks, &NoImages)
            .await
            .unwrap();
        assert_eq!(output, "<|src_id|>0<|tgt_id|>1");
        assert_eq!(blocks[0].contd, 2); // linked to block id 2
        assert_eq!(blocks[1].contd, -1);
    }

    /// A whole-document run of the title-hierarchy subtask against a replay
    /// backend, asserting levels are applied.
    #[tokio::test]
    async fn title_hierarchy_end_to_end_via_replay() {
        // Two titles spanning pages 1 and 5 → one chunk.
        let pages = json!({
            "1": [ { "type": "title", "content": "Chapter One", "bbox": [0.1, 0.1, 0.9, 0.2] } ],
            "5": [ { "type": "title", "content": "A Section", "bbox": [0.1, 0.3, 0.9, 0.4] } ]
        });
        let mut blocks = build_doc_blocks(pages.as_object().unwrap());

        let judges = title::filter_title(&blocks);
        let (_ranges, chunks) = adaptive_chunk(&judges, 50, 1);
        assert_eq!(chunks.len(), 1);
        let prompt = title::build_title_prompt(&chunks[0]);

        let model = "popo-test";
        let req = ChatRequest {
            messages: vec![ChatMessage::text(Role::User, prompt)],
            max_tokens: None,
            temperature: None,
        };
        let fp = popo_model::fingerprint(model, &req);
        let fixture = Fixture {
            fingerprint: fp,
            model: model.into(),
            request: vec![],
            max_tokens: None,
            temperature: None,
            response: RecordedResponse {
                text: "<|id|>0<|level|>1\n<|id|>1<|level|>2".into(),
                usage: None,
            },
        };
        let backend = ReplayBackend::new(model, vec![fixture]);
        let client =
            ModelClient::from_backend("replay", Arc::new(backend), ClientOptions::default());

        let output = run_title_hierarchy(&client, &mut blocks, &NoImages)
            .await
            .unwrap();
        assert_eq!(output, "<|id|>0<|level|>1\n<|id|>1<|level|>2");
        assert_eq!(blocks[0].level, 1);
        assert_eq!(blocks[1].level, 2);
    }
}
