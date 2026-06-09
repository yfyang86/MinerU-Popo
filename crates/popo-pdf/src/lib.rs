//! PDF page rendering and image stitching for the VLM image inputs.
//!
//! The four inference subtasks consume a rendered, multi-page image of the
//! pages in a chunk. This crate provides:
//!   * the page-stitching, JPEG/base64, and bbox-crop primitives (always
//!     available, no native dependency), and
//!   * a pdfium-backed [`PageRenderer`] behind the `pdfium` feature (requires a
//!     `libpdfium` shared library at runtime).
//!
//! [`PdfPageImages`] adapts a renderer to `popo-infer`'s `PageImageProvider`.

use base64::Engine;
use image::{Rgb, RgbImage};
use popo_infer::PageImageProvider;

#[cfg(feature = "pdfium")]
pub mod pdfium;

/// Renders a 1-based page index to an RGB image.
pub trait PageRenderer {
    /// Render the given (1-based) page to an RGB raster.
    fn render_page(&self, page: i64) -> Option<RgbImage>;
}

/// Stack page images vertically, centered, separated by a solid border — port
/// of Python `concatenate_pdf_pages_with_border` (minus the page-number text).
pub fn stitch_pages_with_border(
    pages: &[RgbImage],
    border_width: u32,
    border: Rgb<u8>,
) -> Option<RgbImage> {
    if pages.is_empty() {
        return None;
    }
    let max_width = pages.iter().map(|p| p.width()).max().unwrap();
    let total_height: u32 =
        pages.iter().map(|p| p.height()).sum::<u32>() + border_width * (pages.len() as u32 - 1);

    let mut canvas = RgbImage::from_pixel(max_width, total_height, Rgb([255, 255, 255]));
    let mut y = 0u32;
    for (i, page) in pages.iter().enumerate() {
        let x = (max_width - page.width()) / 2;
        image::imageops::overlay(&mut canvas, page, x as i64, y as i64);
        y += page.height();
        if i < pages.len() - 1 {
            for by in y..(y + border_width).min(total_height) {
                for bx in 0..max_width {
                    canvas.put_pixel(bx, by, border);
                }
            }
            y += border_width;
        }
    }
    Some(canvas)
}

/// Encode an image as JPEG and base64 (no data-URL prefix).
pub fn to_jpeg_base64(image: &RgbImage) -> Option<String> {
    let mut buf = Vec::new();
    image
        .write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Jpeg,
        )
        .ok()?;
    Some(base64::engine::general_purpose::STANDARD.encode(buf))
}

/// Crop a unit-square (`0..1`) bbox out of an image, clamped to its bounds.
pub fn crop_unit_bbox(image: &RgbImage, bbox: [f64; 4]) -> RgbImage {
    let (w, h) = (image.width() as f64, image.height() as f64);
    let x0 = (bbox[0].clamp(0.0, 1.0) * w).floor() as u32;
    let y0 = (bbox[1].clamp(0.0, 1.0) * h).floor() as u32;
    let x1 = (bbox[2].clamp(0.0, 1.0) * w).ceil() as u32;
    let y1 = (bbox[3].clamp(0.0, 1.0) * h).ceil() as u32;
    let cw = x1
        .saturating_sub(x0)
        .max(1)
        .min(image.width() - x0.min(image.width() - 1));
    let ch = y1
        .saturating_sub(y0)
        .max(1)
        .min(image.height() - y0.min(image.height() - 1));
    image::imageops::crop_imm(
        image,
        x0.min(image.width() - 1),
        y0.min(image.height() - 1),
        cw,
        ch,
    )
    .to_image()
}

/// A [`PageImageProvider`] that renders the chunk's pages and stitches them.
pub struct PdfPageImages<R: PageRenderer> {
    renderer: R,
    border_width: u32,
}

impl<R: PageRenderer> PdfPageImages<R> {
    /// Wrap a renderer with the default 5px black separator border.
    pub fn new(renderer: R) -> Self {
        Self {
            renderer,
            border_width: 5,
        }
    }
}

impl<R: PageRenderer> PageImageProvider for PdfPageImages<R> {
    fn image_for_pages(&self, pages: &[i64]) -> Option<(String, String)> {
        let imgs: Vec<RgbImage> = pages
            .iter()
            .filter_map(|&p| self.renderer.render_page(p))
            .collect();
        let stitched = stitch_pages_with_border(&imgs, self.border_width, Rgb([0, 0, 0]))?;
        let b64 = to_jpeg_base64(&stitched)?;
        Some(("image/jpeg".to_string(), b64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A renderer that returns fixed-size solid-color pages.
    struct StubRenderer {
        size: (u32, u32),
    }
    impl PageRenderer for StubRenderer {
        fn render_page(&self, page: i64) -> Option<RgbImage> {
            let shade = (page * 40) as u8;
            Some(RgbImage::from_pixel(
                self.size.0,
                self.size.1,
                Rgb([shade, shade, shade]),
            ))
        }
    }

    #[test]
    fn stitch_dimensions_account_for_borders() {
        let a = RgbImage::from_pixel(100, 50, Rgb([10, 10, 10]));
        let b = RgbImage::from_pixel(80, 40, Rgb([20, 20, 20]));
        let out = stitch_pages_with_border(&[a, b], 5, Rgb([0, 0, 0])).unwrap();
        assert_eq!(out.width(), 100); // max width
        assert_eq!(out.height(), 50 + 40 + 5); // heights + one border
                                               // Border row is black.
        assert_eq!(*out.get_pixel(0, 50), Rgb([0, 0, 0]));
    }

    #[test]
    fn jpeg_base64_round_trips_to_valid_image() {
        let img = RgbImage::from_pixel(16, 16, Rgb([128, 64, 32]));
        let b64 = to_jpeg_base64(&img).unwrap();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!(decoded.width(), 16);
        assert_eq!(decoded.height(), 16);
    }

    #[test]
    fn crop_unit_bbox_extracts_region() {
        let img = RgbImage::from_pixel(100, 100, Rgb([1, 2, 3]));
        let cropped = crop_unit_bbox(&img, [0.25, 0.5, 0.75, 1.0]);
        assert_eq!(cropped.width(), 50);
        assert_eq!(cropped.height(), 50);
    }

    #[test]
    fn provider_stitches_rendered_pages() {
        let provider = PdfPageImages::new(StubRenderer { size: (64, 48) });
        let (media, b64) = provider.image_for_pages(&[1, 2]).unwrap();
        assert_eq!(media, "image/jpeg");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 48 + 48 + 5);
    }
}
