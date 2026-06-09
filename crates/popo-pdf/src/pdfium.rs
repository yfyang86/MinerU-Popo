//! pdfium-backed page renderer (feature `pdfium`).
//!
//! Requires a `libpdfium` shared library available at runtime (system library
//! or one bound explicitly). Rendering re-opens the document per page, which is
//! simple and lifetime-free; a caching renderer can come later.

use std::path::PathBuf;

use image::RgbImage;
use pdfium_render::prelude::*;

use crate::PageRenderer;

/// A renderer backed by pdfium, rendering pages of one document to a target
/// pixel width.
pub struct PdfiumRenderer {
    pdfium: Pdfium,
    path: PathBuf,
    target_width: u32,
}

impl PdfiumRenderer {
    /// Bind to the system pdfium library and open `path`.
    pub fn open(path: impl Into<PathBuf>, target_width: u32) -> Result<Self, PdfiumError> {
        let pdfium = Pdfium::new(Pdfium::bind_to_system_library()?);
        Ok(Self {
            pdfium,
            path: path.into(),
            target_width,
        })
    }
}

impl PageRenderer for PdfiumRenderer {
    fn render_page(&self, page: i64) -> Option<RgbImage> {
        if page < 1 {
            return None;
        }
        let document = self.pdfium.load_pdf_from_file(&self.path, None).ok()?;
        let pages = document.pages();
        let pdf_page = pages.get((page - 1) as u16).ok()?;
        let rendered = pdf_page
            .render_with_config(&PdfRenderConfig::new().set_target_width(self.target_width as i32))
            .ok()?;

        // Convert via raw RGBA bytes to avoid coupling to pdfium-render's image
        // crate version.
        let width = rendered.width() as u32;
        let height = rendered.height() as u32;
        let rgba = rendered.as_rgba_bytes();
        let mut img = RgbImage::new(width, height);
        for (i, pixel) in img.pixels_mut().enumerate() {
            let o = i * 4;
            if o + 2 < rgba.len() {
                *pixel = image::Rgb([rgba[o], rgba[o + 1], rgba[o + 2]]);
            }
        }
        Some(img)
    }
}
