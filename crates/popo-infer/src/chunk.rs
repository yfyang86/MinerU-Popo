//! Dynamic, boundary-aware page chunking — port of Python `adaptive_chunk`.
//!
//! Long documents are split into page-range chunks of roughly `chunk_size`
//! pages, with the boundary nudged toward the page that carries the most items
//! (so a chunk rarely cuts through a dense region), and a one-page `overlap`
//! kept on each side to preserve cross-chunk context.

/// Anything chunkable exposes its 1-based page number.
pub trait Paged {
    /// The page this item sits on.
    fn page(&self) -> i64;
}

/// A page range `[start, end]` (inclusive), parallel to each returned chunk.
pub type Range = [i64; 2];

/// Split `items` into `(ranges, chunks)`. Empty input yields empty outputs.
pub fn adaptive_chunk<T: Paged + Clone>(
    items: &[T],
    chunk_size: i64,
    overlap: i64,
) -> (Vec<Range>, Vec<Vec<T>>) {
    if items.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut sorted: Vec<T> = items.to_vec();
    sorted.sort_by_key(|x| x.page());
    let pages: Vec<i64> = sorted.iter().map(Paged::page).collect();

    let mut unique_pages: Vec<i64> = pages.clone();
    unique_pages.sort_unstable();
    unique_pages.dedup();

    let first = unique_pages[0];
    let last = *unique_pages.last().unwrap();

    // Locate boundary pages: near each `current_min + chunk_size`, pick the page
    // (within ±5) that carries the most items; otherwise the next page past the
    // target.
    let mut boundaries = Vec::new();
    let mut current_min = first;
    while current_min < last {
        let target = current_min + chunk_size;
        let lo = (target - 5).max(first);
        let hi = (target + 5).min(last);

        let mut best: Option<(i64, usize)> = None; // (page, freq)
        for page in lo..=hi {
            if unique_pages.binary_search(&page).is_ok() {
                let freq = pages.iter().filter(|&&p| p == page).count();
                // `max(freq, key=...)` keeps the first max on ties (ascending page).
                if best.map(|(_, bf)| freq > bf).unwrap_or(true) {
                    best = Some((page, freq));
                }
            }
        }
        let boundary = match best {
            Some((page, _)) => page,
            None => unique_pages
                .iter()
                .copied()
                .find(|&x| x > target)
                .unwrap_or(last),
        };
        boundaries.push(boundary);
        current_min = boundary;
    }

    // Assemble chunks for each boundary, with overlap on both sides.
    let mut ranges = Vec::new();
    let mut chunks = Vec::new();
    let mut prev_boundary = first;
    for &boundary in &boundaries {
        let chunk: Vec<T> = sorted
            .iter()
            .filter(|item| {
                let p = item.page();
                prev_boundary - overlap <= p && p <= boundary + overlap
            })
            .cloned()
            .collect();
        if !chunk.is_empty() {
            let start = if prev_boundary == first {
                prev_boundary
            } else {
                prev_boundary - overlap
            };
            let end = last.min(boundary + overlap);
            chunks.push(chunk);
            ranges.push([start, end]);
        }
        prev_boundary = boundary;
    }

    // Trailing chunk, only kept if it spans more than two pages.
    let last_chunk: Vec<T> = sorted
        .iter()
        .filter(|item| {
            let p = item.page();
            prev_boundary - overlap <= p && p <= last
        })
        .cloned()
        .collect();
    if !last_chunk.is_empty() {
        let start = if prev_boundary == first {
            prev_boundary
        } else {
            prev_boundary - overlap
        };
        if last - start > 2 {
            chunks.push(last_chunk);
            ranges.push([start, last]);
        }
    }

    (ranges, chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Item {
        page: i64,
    }
    impl Paged for Item {
        fn page(&self) -> i64 {
            self.page
        }
    }

    fn items(pages: &[i64]) -> Vec<Item> {
        pages.iter().map(|&page| Item { page }).collect()
    }

    #[test]
    fn empty_input() {
        let (r, c) = adaptive_chunk::<Item>(&[], 50, 1);
        assert!(r.is_empty() && c.is_empty());
    }

    #[test]
    fn single_chunk_small_doc_has_no_boundaries() {
        // first == last (one page) → no boundaries, trailing chunk span 0 → dropped.
        let (r, c) = adaptive_chunk(&items(&[1, 1, 1]), 50, 1);
        assert!(r.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn splits_long_doc_with_overlap() {
        // 120 pages, one item each, chunk_size 50.
        let pages: Vec<i64> = (1..=120).collect();
        let (ranges, chunks) = adaptive_chunk(&items(&pages), 50, 1);
        assert_eq!(ranges.len(), chunks.len());
        assert!(
            ranges.len() >= 2,
            "expected multiple chunks, got {ranges:?}"
        );
        // First range starts at page 1.
        assert_eq!(ranges[0][0], 1);
        // Last range ends at the final page.
        assert_eq!(ranges.last().unwrap()[1], 120);
        // Consecutive ranges overlap (each later range starts before the prior end).
        for w in ranges.windows(2) {
            assert!(w[1][0] <= w[0][1], "ranges should overlap: {:?}", w);
        }
    }

    #[test]
    fn boundary_prefers_denser_page() {
        // Around the target page 51, page 53 is denser; boundary should land there.
        let mut pages: Vec<i64> = (1..=120).collect();
        pages.extend([53, 53, 53, 53]); // make page 53 dense
        let (ranges, _) = adaptive_chunk(&items(&pages), 50, 1);
        // The first boundary's end is boundary+overlap; boundary near 53 → end 54.
        assert_eq!(ranges[0][1], 54);
    }
}
