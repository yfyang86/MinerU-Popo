//! `popo` — command-line entry point for the MinerU-Popo Rust engine.
//!
//! Sprint 1 exposes the model-configuration surface:
//!   * `popo config check`   — validate the TOML config.
//!   * `popo model list`     — list configured providers.
//!   * `popo model chat`     — send a one-shot prompt to a provider.
//!
//! Pipeline subcommands (`normalize`, `infer`, `build-tree`, `enrich`, `eval`)
//! are added in later sprints.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use popo_model::{ChatRequest, LlmConfig, ModelClient};

/// MinerU-Popo Rust engine.
#[derive(Parser)]
#[command(name = "popo", version, about)]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(long, env = "POPO_CONFIG", default_value = "popo.toml", global = true)]
    config: PathBuf,

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
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    popo_obs::init("info");
    let cli = Cli::parse();

    match cli.command {
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
        }) => {
            let cfg = load(&cli.config)?;
            let client = ModelClient::from_config(&cfg, provider.as_deref())
                .context("building model client")?;
            let mut req = ChatRequest::user_prompt(prompt);
            req.max_tokens = max_tokens;
            req.temperature = temperature;

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
    }
    Ok(())
}

fn load(path: &PathBuf) -> Result<LlmConfig> {
    LlmConfig::from_path(path).with_context(|| format!("loading config from {}", path.display()))
}
