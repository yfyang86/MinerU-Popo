//! Prometheus metrics wiring.
//!
//! The engine emits metrics through the lightweight [`metrics`] facade so that
//! library crates (e.g. `popo-model`) stay decoupled from any exporter. A
//! binary installs a concrete recorder here; if none is installed, the facade
//! macros are cheap no-ops.
//!
//! ```
//! let handle = popo_obs::metrics::install();
//! metrics::counter!("popo_demo_total").increment(1);
//! assert!(handle.render().contains("popo_demo_total"));
//! ```

pub use metrics_exporter_prometheus::PrometheusHandle;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusRecorder};

/// Build a Prometheus recorder and return both it and its render handle.
///
/// Use this when you need to scope a recorder (for example in tests via
/// [`metrics::with_local_recorder`]) rather than installing it globally.
pub fn recorder() -> (PrometheusRecorder, PrometheusHandle) {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    (recorder, handle)
}

/// Install a global Prometheus recorder and return a handle whose
/// [`PrometheusHandle::render`] yields the current exposition text.
///
/// Installing a global recorder more than once in a process will fail; callers
/// that may double-install should guard with a `OnceCell`/`OnceLock`.
pub fn install() -> PrometheusHandle {
    let (recorder, handle) = recorder();
    // `set_global_recorder` returns Err if one is already installed; surface the
    // existing handle in that case is not possible, so we leak-install best
    // effort and ignore a double-install.
    let _ = metrics::set_global_recorder(recorder);
    handle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_recorder_renders_emitted_metric() {
        let (recorder, handle) = recorder();
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("popo_test_total", "k" => "v").increment(3);
        });
        let text = handle.render();
        assert!(text.contains("popo_test_total"), "render was: {text}");
    }
}
