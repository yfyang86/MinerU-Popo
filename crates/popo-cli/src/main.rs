//! `popo` — command-line entry point for the MinerU-Popo Rust engine.
//!
//! Model-configuration surface:
//!   * `popo config check`   — validate the TOML config.
//!   * `popo model list`     — list configured providers.
//!   * `popo model chat`     — send a one-shot prompt to a provider.
//!
//! A global `--metrics` flag installs a Prometheus recorder and prints the
//! exposition text to stderr on exit. Set `POPO_MODEL_REPLAY=<dir>` to serve
//! `model chat` from recorded fixtures (no network), or pass
//! `model chat --record <dir>` to capture live responses as fixtures.
//!
//! Pipeline subcommands (`normalize`, `infer`, `build-tree`, `enrich`, `eval`)
//! are added in later sprints.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use popo_model::{ChatRequest, ClientOptions, LlmConfig, ModelClient, ReplayBackend};

mod normalize;

/// MinerU-Popo Rust engine.
#[derive(Parser)]
#[command(name = "popo", version, about)]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(long, env = "POPO_CONFIG", default_value = "popo.toml", global = true)]
    config: PathBuf,

    /// Install a Prometheus recorder and print the exposition to stderr on exit.
    #[arg(long, global = true)]
    metrics: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Configuration utilities.
    #[command(subcommand)]
    Config(ConfigCmd),
    /// Model/provider utilities.
    #[command(subcommand)]
    Model(ModelCmd),
    /// Normalize OCR/layout outputs into the canonical block schema.
    Normalize(normalize::NormalizeArgs),
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Validate the configuration file and print a summary.
    Check,
}

#[derive(Subcommand)]
enum ModelCmd {
    /// List configured providers.
    List,
    /// Send a one-shot prompt to a provider (defaults to the default provider).
    Chat {
        /// The prompt text.
        prompt: String,
        /// Provider name; falls back to `[default].provider`.
        #[arg(long, short)]
        provider: Option<String>,
        /// Maximum tokens to generate.
        #[arg(long)]
        max_tokens: Option<u32>,
        /// Sampling temperature.
        #[arg(long)]
        temperature: Option<f32>,
        /// Record the live response into this fixture directory.
        #[arg(long)]
        record: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    popo_obs::init("info");
    let cli = Cli::parse();
    let metrics_handle = cli.metrics.then(popo_obs::metrics::install);

    let result = run(&cli).await;

    if let Some(handle) = metrics_handle {
        eprintln!("{}", handle.render());
    }
    result
}

async fn run(cli: &Cli) -> Result<()> {
    match &cli.command {
        Command::Config(ConfigCmd::Check) => {
            let cfg = load(&cli.config)?;
            let (default_name, _) = cfg
                .default_provider()
                .context("resolving default provider")?;
            println!(
                "OK: {} provider(s), default = {:?}",
                cfg.providers.len(),
                default_name
            );
        }
        Command::Model(ModelCmd::List) => {
            let cfg = load(&cli.config)?;
            let default_name = cfg.default.provider.clone();
            for (name, p) in &cfg.providers {
                let marker = if *name == default_name {
                    " (default)"
                } else {
                    ""
                };
                println!("{name}{marker}\t{:?}\t{}\t{}", p.kind, p.model, p.api_base);
            }
        }
        Command::Model(ModelCmd::Chat {
            prompt,
            provider,
            max_tokens,
            temperature,
            record,
        }) => {
            let mut req = ChatRequest::user_prompt(prompt.clone());
            req.max_tokens = *max_tokens;
            req.temperature = *temperature;

            let client = build_chat_client(cli, provider.as_deref(), record.as_deref())?;
            tracing::info!(
                provider = client.provider_name(),
                model = client.model(),
                "sending chat"
            );
            let resp = client.chat(&req).await.context("model request failed")?;
            println!("{}", resp.text);
            if let Some(u) = resp.usage {
                tracing::info!(
                    input_tokens = u.input_tokens,
                    output_tokens = u.output_tokens,
                    "usage"
                );
            }
        }
        Command::Normalize(args) => normalize::run(args)?,
    }
    Ok(())
}

fn load(path: &PathBuf) -> Result<LlmConfig> {
    LlmConfig::from_path(path).with_context(|| format!("loading config from {}", path.display()))
}

/// Build the model client for `model chat`.
///
/// When `POPO_MODEL_REPLAY` points at a fixture directory, responses are served
/// hermetically from disk. Otherwise a live client is built from config, and if
/// `record` is set, its responses are captured as fixtures.
fn build_chat_client(
    cli: &Cli,
    provider: Option<&str>,
    record: Option<&std::path::Path>,
) -> Result<ModelClient> {
    if let Ok(dir) = std::env::var("POPO_MODEL_REPLAY") {
        let backend = ReplayBackend::from_dir(&dir)
            .with_context(|| format!("loading replay fixtures from {dir}"))?;
        tracing::info!(replay_dir = %dir, fixtures = backend.len(), "using replay backend");
        return Ok(ModelClient::from_backend(
            "replay",
            Arc::new(backend),
            ClientOptions::default(),
        ));
    }

    let cfg = load(&cli.config)?;
    match record {
        Some(dir) => {
            tracing::info!(record_dir = %dir.display(), "recording responses to fixtures");
            ModelClient::from_config_recording(&cfg, provider, dir, ClientOptions::default())
                .context("building recording model client")
        }
        None => ModelClient::from_config(&cfg, provider).context("building model client"),
    }
}
