//! Observability spine for the MinerU-Popo engine.
//!
//! Sprint 1 ships the tracing skeleton: a single [`init`] entry point that
//! installs a `tracing` subscriber honoring the `POPO_LOG` / `RUST_LOG`
//! environment filter. Prometheus metrics and profiling hooks join this crate
//! in Sprint 2.

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize global tracing for a binary.
///
/// The log level is taken from `POPO_LOG`, falling back to `RUST_LOG`, then to
/// the supplied `default_directive` (e.g. `"info"`). Calling this more than
/// once is a no-op; the second call simply returns.
pub fn init(default_directive: &str) {
    let filter = EnvFilter::try_from_env("POPO_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new(default_directive));

    // `try_init` returns Err if a global subscriber is already set; that is a
    // benign double-init, so we ignore it.
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(false))
        .try_init();
}
