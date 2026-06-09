//! OCR/layout output adapters.
//!
//! Each supported page-level parser (MinerU, MonkeyOCR, Dolphin, PaddleOCR-VL,
//! GLM-OCR) has a [`OcrReader`] implementation that maps its native JSON into
//! the canonical [`NormalizedBlock`](popo_core::NormalizedBlock) schema. This
//! is the Rust port of the reference `label_normalization.py`.
//!
//! Sprint 3 ships the MinerU family (MinerU + MonkeyOCR, which share the
//! `middle.json` path); the remaining readers slot in behind the same trait.

use popo_core::{reassign_block_ids, NormalizedBlock, Result};

pub mod common;
pub mod mineru;

pub use mineru::{MineruReader, MonkeyOcrReader};

/// The outcome of reading one document.
#[derive(Debug, Clone)]
pub struct ReaderResult {
    /// The OCR model that produced the input.
    pub model_name: String,
    /// Document id (typically the file stem).
    pub doc_id: String,
    /// `"ok"` or `"missing"`.
    pub status: String,
    /// Normalized blocks in reading order.
    pub blocks: Vec<NormalizedBlock>,
    /// Diagnostic message (e.g. which file was missing).
    pub message: String,
}

impl ReaderResult {
    /// A successful result; block ids/order are reassigned in reading order.
    pub fn ok(model_name: &str, doc_id: &str, blocks: Vec<NormalizedBlock>) -> Self {
        Self {
            model_name: model_name.to_string(),
            doc_id: doc_id.to_string(),
            status: "ok".into(),
            blocks: reassign_block_ids(blocks, doc_id),
            message: String::new(),
        }
    }

    /// A "missing input" result.
    pub fn missing(model_name: &str, doc_id: &str, message: impl Into<String>) -> Self {
        Self {
            model_name: model_name.to_string(),
            doc_id: doc_id.to_string(),
            status: "missing".into(),
            blocks: Vec::new(),
            message: message.into(),
        }
    }
}

/// An adapter from one OCR model's output to canonical blocks.
pub trait OcrReader {
    /// The model name this reader handles.
    fn model_name(&self) -> &str;

    /// Read and normalize a single document by id.
    fn read_doc(&self, doc_id: &str) -> Result<ReaderResult>;
}

/// Build a reader for `model_name` rooted at `input_dir` (the per-model
/// directory that holds `<doc_id>/...`).
pub fn build_reader(
    model_name: &str,
    input_dir: impl Into<std::path::PathBuf>,
) -> Result<Box<dyn OcrReader>> {
    let root = input_dir.into();
    match model_name {
        "mineru" => Ok(Box::new(MineruReader::new(root))),
        "monkeyocr" => Ok(Box::new(MonkeyOcrReader::new(root))),
        other => Err(popo_core::Error::Config(format!(
            "unsupported model {other:?} (supported so far: mineru, monkeyocr)"
        ))),
    }
}
