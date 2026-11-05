use crate::{
    IntegrationTestLogMarker, WandBInfo, fetch_data::DataFetcher,
    matformer::{
        MATFORMER_MANIFEST_NAME, MatformerManifest, P2pLoadPlan,
        ensure_matformer_checkpoint_metadata, file_sha256_hex, infer_matformer_checkpoint_metadata,
        plan_p2p_load,
    },
};
use crate::cli::MatformerLoadStrategy;
use psyche_coordinator::{
    Coordinator, HealthChecks,
    model::{self, HttpLLMTrainingDataLocation, LLMTrainingDataLocation},
};
use psyche_core::{
    Barrier, CancellableBarrier, NodeIdentity, OptimizerDefinition, Shuffle, TokenSize, sha256,
};
use psyche_data_provider::{
    DataProvider, DataProviderTcpClient, DummyDataProvider, LocalDataProvider,
    PreprocessedDataProvider, Split, WeightedDataProvider, download_dataset_repo_async,
    download_model_repo_async, download_model_repo_files_async, list_model_repo_files_async,
    http::{FileURLs, HttpDataProvider},
};
use psyche_metrics::ClientMetrics;
use psyche_modeling::{
    AttentionImplementation, AutoConfig, AutoTokenizerError, CausalLM, CommunicatorId,
    DataParallel, DeepseekConfig, DeepseekForCausalLM, Devices, DummyModel, LlamaConfig,
    LlamaForCausalLM, NanoGPTConfig, NanoGPTForCausalLM, LocalTrainer, ModelConfig, ModelLoadError,
    ParallelModels, PretrainedSource, Trainer,
    auto_tokenizer,
};
use psyche_network::{AuthenticatableIdentity, BlobTicket};
use psyche_watcher::{ModelSchemaInfo, OpportunisticData};
use std::{
    collections::{HashMap, HashSet},
    ffi::OsString,
    path::{Component, Path, PathBuf},
    sync::Arc,
};
use tch::{Device, Kind, Tensor};
use thiserror::Error;
use tokenizers::{ModelWrapper, Tokenizer, models::wordlevel::WordLevel};
use tokio::{
    io,
    sync::{mpsc::UnboundedSender, oneshot},
    task::{JoinError, JoinHandle},
};
use tracing::{debug, error, info, warn};

use super::{
    CheckpointConfig, FinishedBroadcast,
    cooldown::{CooldownStepMetadata, MatformerCheckpointInfo},
    evals::ModelTaskRunner, stats::StatsLogger, steps::StepStateMachine,
    train::TrainingStepMetadata, types::DistroBroadcastAndPayload,
    warmup::WarmupStepMetadata, witness::WitnessStepMetadata,
};
use iroh_blobs::api::Tag;

pub struct RunInitConfig<T: NodeIdentity, A: AuthenticatableIdentity> {
    // identity for connecting to the data server
    pub identity: T,
    pub network_identity: A,
    pub private_key: A::PrivateKey,

    // p2p model parameters sharing config
    pub max_concurrent_parameter_requests: usize,

    // model & dataload
    pub device: Devices,
    pub matformer_tier: u8,
    pub matformer_load_strategy: MatformerLoadStrategy,
    pub matformer_helper_fraction: f32,
    pub matformer_helper_rotation_interval: u64,
    pub hub_read_token: Option<String>,
    pub hub_max_concurrent_downloads: usize,
    pub data_parallelism: usize,
    pub tensor_parallelism: usize,
    pub micro_batch_size: usize,
    pub optim_stats_every_n_steps: Option<u32>,
    pub grad_accum_in_fp32: bool,
    pub log_memory_usage: bool,

    // evaluation
    pub eval_task_max_docs: Option<usize>,
    pub eval_tasks: Vec<psyche_eval::Task>,
    pub prompt_task: bool,

    // logging
    pub wandb_info: Option<WandBInfo>,

    // debugging
    pub write_gradients_dir: Option<PathBuf>,

    // checkpointing
    pub checkpoint_config: Option<CheckpointConfig>,

    // configurable dummy training time (in seconds) for this client - relevant just for testing
    pub dummy_training_delay_secs: Option<u64>,

    pub sidecar_port: Option<u16>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_apply_matformer_checkpoint_tier_overrides_auto() {
        let config = json!({
            "intermediate_size": 1024,
            "matformer_tier": 1
        });
        let (uses_sliced, effective_tier) = apply_matformer_checkpoint_tier_overrides(
            &config,
            &MatformerLoadStrategy::Auto,
            false,
            2,
        );
        assert!(uses_sliced);
        assert_eq!(effective_tier, 0);
    }

    #[test]
    fn test_apply_matformer_checkpoint_tier_overrides_universal() {
        let config = json!({
            "intermediate_size": 1024,
            "matformer_tier": 1
        });
        let (uses_sliced, effective_tier) = apply_matformer_checkpoint_tier_overrides(
            &config,
            &MatformerLoadStrategy::Universal,
            false,
            2,
        );
        assert!(uses_sliced);
        assert_eq!(effective_tier, 2);
    }

    #[test]
    fn test_validate_no_double_slicing_universal() {
        let config = json!({
            "intermediate_size": 1024,
            "matformer_tier": 1
        });
        let err = validate_no_double_slicing(
            &config,
            1,
            &MatformerLoadStrategy::Universal,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, InitRunError::DoubleSlicingDetected { .. }));
    }

    #[test]
    fn test_validate_no_double_slicing_auto() {
        let config = json!({
            "intermediate_size": 1024,
            "matformer_tier": 1
        });
        assert!(validate_no_double_slicing(
            &config,
            1,
            &MatformerLoadStrategy::Auto,
            true,
        )
        .is_ok());
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut dir = std::env::temp_dir();
        dir.push(format!("psyche-{prefix}-{nanos}"));
        dir
    }

    #[tokio::test]
    async fn test_resolve_local_manifest_tier() {
        let base = temp_dir("matformer-manifest");
        let tier_dir = base.with_file_name(format!(
            "{}-tier1",
            base.file_name().unwrap().to_string_lossy()
        ));
        let tier_name = tier_dir.file_name().unwrap().to_string_lossy().to_string();
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&tier_dir).unwrap();

        std::fs::write(
            base.join("config.json"),
            r#"{"intermediate_size":1024,"matformer_tier":0,"matformer_base_intermediate_size":1024}"#,
        )
        .unwrap();
        std::fs::write(base.join("model.safetensors"), b"").unwrap();
        std::fs::write(
            tier_dir.join("config.json"),
            r#"{"intermediate_size":512,"matformer_tier":1,"matformer_base_intermediate_size":1024}"#,
        )
        .unwrap();
        std::fs::write(tier_dir.join("model.safetensors"), b"").unwrap();

        let manifest = json!({
            "schema_version": 1,
            "matformer_base_intermediate_size": 1024,
            "common_files": [],
            "tiers": [
                {
                    "tier": 1,
                    "intermediate_size": 512,
                    "files": [
                        format!("../{tier_name}/config.json"),
                        format!("../{tier_name}/model.safetensors")
                    ]
                }
            ],
            "sha256": {}
        });
        std::fs::write(
            base.join(MATFORMER_MANIFEST_NAME),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let resolved =
            resolve_matformer_local_repo_files(&base, 1, MatformerLoadStrategy::Auto)
                .await
                .unwrap()
                .unwrap();
        assert!(resolved.uses_sliced_checkpoint);
        assert_eq!(resolved.matformer_tier_for_loading, 0);
        assert!(resolved
            .repo_files
            .iter()
            .any(|path| path.ends_with("config.json")));

        std::fs::remove_dir_all(&base).ok();
        std::fs::remove_dir_all(&tier_dir).ok();
    }

    #[tokio::test]
    async fn test_resolve_local_manifest_missing_tier_sliced() {
        let base = temp_dir("matformer-manifest-missing");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(
            base.join("config.json"),
            r#"{"intermediate_size":1024,"matformer_tier":0,"matformer_base_intermediate_size":1024}"#,
        )
        .unwrap();
        std::fs::write(base.join("model.safetensors"), b"").unwrap();

        let manifest = json!({
            "schema_version": 1,
            "matformer_base_intermediate_size": 1024,
            "common_files": [],
            "tiers": [],
            "sha256": {}
        });
        std::fs::write(
            base.join(MATFORMER_MANIFEST_NAME),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let err =
            resolve_matformer_local_repo_files(&base, 1, MatformerLoadStrategy::Sliced)
                .await
                .unwrap_err();
        assert!(matches!(
            err,
            InitRunError::MissingMatformerManifestTier { .. }
        ));

        std::fs::remove_dir_all(&base).ok();
    }

    #[tokio::test]
    async fn test_resolve_local_tier_without_base_dir() {
        let base = temp_dir("matformer-tier-only");
        let tier_dir = base.with_file_name(format!(
            "{}-tier1",
            base.file_name().unwrap().to_string_lossy()
        ));
        std::fs::create_dir_all(&tier_dir).unwrap();

        std::fs::write(
            tier_dir.join("config.json"),
            r#"{"intermediate_size":512,"matformer_tier":1,"matformer_base_intermediate_size":1024}"#,
        )
        .unwrap();
        std::fs::write(tier_dir.join("model.safetensors"), b"").unwrap();

        let resolved =
            resolve_matformer_local_repo_files(&base, 1, MatformerLoadStrategy::Auto)
                .await
                .unwrap()
                .unwrap();
        assert!(resolved.uses_sliced_checkpoint);
        assert_eq!(resolved.matformer_tier_for_loading, 0);
        assert_eq!(
            resolved.checkpoint_path.as_deref(),
            Some(tier_dir.to_string_lossy().as_ref())
        );

        std::fs::remove_dir_all(&tier_dir).ok();
    }

    #[test]
    fn test_canonical_schema_hash_matches_full_and_sliced() {
        let parameter_names = vec!["model.layers.0.mlp.gate_proj.weight".to_string()];
        let config_full = json!({
            "intermediate_size": 256
        });
        let config_slice = json!({
            "intermediate_size": 128,
            "matformer_tier": 1,
            "matformer_base_intermediate_size": 256
        });

        let canonical_full = canonicalize_config_for_schema(config_full, 0, false);
        let canonical_slice = canonicalize_config_for_schema(config_slice, 1, true);

        assert_eq!(canonical_full["intermediate_size"], 256);
        assert_eq!(canonical_slice["intermediate_size"], 256);
        assert_eq!(canonical_full["matformer_base_intermediate_size"], 256);
        assert_eq!(canonical_slice["matformer_base_intermediate_size"], 256);

        let hash_full = schema_hash_for_config(
            model::LLMArchitecture::HfLlama,
            &canonical_full,
            &parameter_names,
        );
        let hash_slice = schema_hash_for_config(
            model::LLMArchitecture::HfLlama,
            &canonical_slice,
            &parameter_names,
        );

        assert_eq!(hash_full, hash_slice);
    }

    /// A client that joins over P2P must get the same canonical schema as the first
    /// clients, and it must not depend on the tier of the peer that answers.
    #[test]
    fn test_canonical_schema_hash_is_stable_across_p2p_joins() {
        let parameter_names = vec!["model.layers.0.mlp.gate_proj.weight".to_string()];
        let hub_config = json!({
            "intermediate_size": 256,
            "hidden_size": 64,
            "field_that_the_rust_config_ignores": "kept",
        });
        let hash = |config: serde_json::Value, tier: u8, sliced: bool| {
            schema_hash_for_config(
                model::LLMArchitecture::HfLlama,
                &canonicalize_config_for_schema(config, tier, sliced),
                &parameter_names,
            )
        };
        let expected = hash(hub_config.clone(), 0, false);

        let sliced_peer = plan_p2p_load(hub_config.clone(), 1, false).unwrap();
        assert_eq!(hash(sliced_peer.config.clone(), 1, true), expected);

        for peer_config in [hub_config, sliced_peer.config] {
            for (tier, universal) in [(0, false), (1, false), (2, false), (1, true)] {
                let plan = plan_p2p_load(peer_config.clone(), tier, universal).unwrap();
                assert_eq!(
                    hash(plan.config, tier, plan.uses_sliced_checkpoint),
                    expected,
                    "tier {tier}, universal {universal}"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_manifest_hash_check_rejects_a_changed_file() {
        let dir = std::env::temp_dir().join(format!(
            "matformer-hash-{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        std::fs::write(&file, b"weights").unwrap();
        let good = file_sha256_hex(&file).unwrap();

        verify_manifest_file_hashes(vec![(file.clone(), good.clone())])
            .await
            .unwrap();
        verify_manifest_file_hashes(vec![(file.clone(), good.to_uppercase())])
            .await
            .unwrap();

        std::fs::write(&file, b"other weights").unwrap();
        let err = verify_manifest_file_hashes(vec![(file.clone(), good)])
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InitRunError::MatformerManifestHashMismatch { .. }
        ));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_normalize_manifest_paths_for_repo() {
        let selection = ManifestSelection {
            files: vec![
                "../model-tier1/config.json".to_string(),
                "../model-tier1/model.safetensors".to_string(),
            ],
            config_path: "../model-tier1/config.json".to_string(),
        };
        let normalized =
            normalize_manifest_selection_for_repo("model/matformer_manifest.json", selection)
                .unwrap();
        assert_eq!(
            normalized.files,
            vec![
                "model-tier1/config.json".to_string(),
                "model-tier1/model.safetensors".to_string(),
            ]
        );
        assert_eq!(normalized.config_path, "model-tier1/config.json");
    }

    #[test]
    fn test_normalize_manifest_paths_root_parent_clamps() {
        let selection = ManifestSelection {
            files: vec!["../model-tier1/config.json".to_string()],
            config_path: "../model-tier1/config.json".to_string(),
        };
        let normalized =
            normalize_manifest_selection_for_repo("matformer_manifest.json", selection).unwrap();
        assert_eq!(
            normalized.files,
            vec!["model-tier1/config.json".to_string()]
        );
        assert_eq!(normalized.config_path, "model-tier1/config.json");
    }

    #[test]
    fn test_normalize_manifest_paths_rejects_absolute() {
        let selection = ManifestSelection {
            files: vec!["/etc/passwd".to_string()],
            config_path: "/etc/passwd".to_string(),
        };
        let err =
            normalize_manifest_selection_for_repo("matformer_manifest.json", selection).unwrap_err();
        assert!(matches!(
            err,
            InitRunError::InvalidMatformerManifestPath { .. }
        ));
    }

    #[tokio::test]
    #[ignore]
    async fn test_resolve_hub_manifest_paths() {
        let repo_id = std::env::var("PSYCHE_HF_TEST_REPO")
            .expect("PSYCHE_HF_TEST_REPO must be set for this test");
        let token = std::env::var("PSYCHE_HF_TOKEN")
            .expect("PSYCHE_HF_TOKEN must be set for this test");

        let resolved = resolve_matformer_hub_repo_files(
            &repo_id,
            None,
            1,
            MatformerLoadStrategy::Sliced,
            Some(token),
            4,
        )
        .await
        .expect("expected manifest-based tier resolution");

        assert!(resolved.uses_sliced_checkpoint);
        assert_eq!(resolved.sliced_checkpoint_tier, Some(1));
        assert_eq!(resolved.matformer_tier_for_loading, 0);
        assert_eq!(resolved.manifest_base_intermediate_size, Some(1024));
        assert_eq!(resolved.checkpoint_path.as_deref(), Some("model-tier1"));
        assert!(resolved
            .repo_files
            .iter()
            .any(|path| path.ends_with(Path::new("model-tier1/config.json"))));
    }
}

/// Print MatFormer configuration summary with ASCII art header.
/// Called after model loading to provide clear visibility into tier configuration.
fn print_matformer_summary(
    checkpoint_path: &str,
    cli_tier: u8,
    effective_tier: u8,
    uses_sliced_checkpoint: bool,
    load_strategy: &MatformerLoadStrategy,
    helper_fraction: f32,
    intermediate_size: u64,
    active_intermediate_size: u64,
) {
    let helper_status = if helper_fraction > 0.0 {
        if uses_sliced_checkpoint {
            "Auto-disabled (sliced checkpoint)"
        } else {
            "Enabled"
        }
    } else {
        "Disabled"
    };

    let tier_match = if cli_tier == effective_tier { "✓" } else { "≠" };
    let capacity_pct = (active_intermediate_size * 100) / intermediate_size;

    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║                      MATFORMER CONFIGURATION                     ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Checkpoint:       {:<45} ║", truncate_path(checkpoint_path, 45));
    eprintln!("║  Load strategy:    {:<45} ║", format!("{:?}", load_strategy));
    eprintln!("║  Sliced checkpoint: {:<44} ║", if uses_sliced_checkpoint { "Yes" } else { "No" });
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!("║  CLI tier:         {:<45} ║", cli_tier);
    eprintln!("║  Effective tier:   {:<45} ║", format!("{} {}", effective_tier, tier_match));
    eprintln!("║  Helper mode:      {:<45} ║", helper_status);
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!("║  FFN width:        {:<45} ║", format!("{} / {} ({}%)", active_intermediate_size, intermediate_size, capacity_pct));
    eprintln!("╚══════════════════════════════════════════════════════════════════╝");

    // Validation warnings
    if cli_tier != effective_tier {
        eprintln!();
        eprintln!("[WARNING] CLI tier ({}) differs from effective tier ({})", cli_tier, effective_tier);
        if uses_sliced_checkpoint {
            eprintln!("          Sliced checkpoint detected - using tier 0 to avoid double-slicing.");
            eprintln!("          This is expected behavior for pre-truncated checkpoints.");
        }
    }

    if helper_fraction > 0.0 && uses_sliced_checkpoint {
        eprintln!();
        eprintln!("[INFO] Helper mode requested but auto-disabled for sliced checkpoint.");
        eprintln!("       Sliced checkpoints have no suffix neurons to sample from.");
    }

    if helper_fraction > 0.0 && cli_tier > 0 && !uses_sliced_checkpoint {
        eprintln!();
        eprintln!("[WARNING] Helper mode with tier > 0 on full checkpoint.");
        eprintln!("          This is an advanced configuration. Ensure gradients align correctly.");
    }

    eprintln!();
}

/// Truncate a path string to fit within max_len, adding "..." prefix if needed.
fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - (max_len - 3)..])
    }
}

async fn resolve_matformer_local_repo_path(
    base: &Path,
    tier: u8,
    strategy: MatformerLoadStrategy,
) -> Result<(Option<PathBuf>, bool), InitRunError> {
    let base_exists = tokio::fs::try_exists(base).await?;
    if tier == 0 {
        return Ok((base_exists.then(|| base.to_path_buf()), false));
    }

    let tier_path = base
        .file_name()
        .and_then(|n| n.to_str())
        .map(|name| base.with_file_name(format!("{name}-tier{tier}")))
        .unwrap_or_else(|| base.to_path_buf());
    let tier_exists = tokio::fs::try_exists(&tier_path).await?;

    match strategy {
        MatformerLoadStrategy::Universal => {
            Ok((base_exists.then(|| base.to_path_buf()), false))
        }
        MatformerLoadStrategy::Auto => {
            if tier_exists {
                Ok((Some(tier_path), true))
            } else if base_exists {
                Ok((Some(base.to_path_buf()), false))
            } else {
                Ok((None, false))
            }
        }
        MatformerLoadStrategy::Sliced => {
            if tier_exists {
                Ok((Some(tier_path), true))
            } else if base_exists {
                Err(InitRunError::MissingMatformerTierCheckpoint {
                    path: tier_path.to_string_lossy().to_string(),
                    tier,
                })
            } else {
                Ok((None, false))
            }
        }
    }
}

#[derive(Debug)]
struct ManifestSelection {
    files: Vec<String>,
    config_path: String,
}

#[derive(Debug)]
struct ResolvedRepoFiles {
    repo_files: Vec<PathBuf>,
    uses_sliced_checkpoint: bool,
    matformer_tier_for_loading: u8,
    checkpoint_path: Option<String>,
    manifest_base_intermediate_size: Option<u64>,
    sliced_checkpoint_tier: Option<u8>,
}

/// Compares files with the SHA-256 values in the manifest.
///
/// A manifest with no value for a file does not check that file.
async fn verify_manifest_file_hashes(files: Vec<(PathBuf, String)>) -> Result<(), InitRunError> {
    if files.is_empty() {
        return Ok(());
    }
    info!(
        "Checking SHA-256 of {} MatFormer checkpoint files",
        files.len()
    );
    tokio::task::spawn_blocking(move || {
        for (path, expected) in files {
            let actual = file_sha256_hex(&path)?;
            if !actual.eq_ignore_ascii_case(&expected) {
                return Err(InitRunError::MatformerManifestHashMismatch {
                    path: path.to_string_lossy().to_string(),
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    })
    .await
    .map_err(InitRunError::ModelLoadingThreadCrashed)?
}

fn ensure_manifest_supported(manifest: &MatformerManifest) -> Result<(), InitRunError> {
    if manifest.schema_version != 1 {
        return Err(InitRunError::UnsupportedMatformerManifestVersion {
            version: manifest.schema_version,
        });
    }
    Ok(())
}

fn select_manifest_tier_files(
    manifest: &MatformerManifest,
    tier: u8,
) -> Result<ManifestSelection, InitRunError> {
    ensure_manifest_supported(manifest)?;
    let entry = manifest.tier_entry(tier).ok_or_else(|| {
        InitRunError::MissingMatformerManifestTier {
            tier,
            available: manifest.available_tiers(),
        }
    })?;

    let config_path = entry
        .files
        .iter()
        .find(|path| path.ends_with("config.json"))
        .cloned()
        .ok_or(InitRunError::MatformerManifestMissingConfig { tier })?;

    if !entry
        .files
        .iter()
        .any(|path| path.ends_with(".safetensors"))
    {
        return Err(InitRunError::MatformerManifestMissingWeights { tier });
    }

    let mut files = Vec::new();
    let mut seen = HashSet::new();
    for name in manifest
        .common_files
        .iter()
        .chain(entry.files.iter())
    {
        if seen.insert(name) {
            files.push(name.clone());
        }
    }

    Ok(ManifestSelection {
        files,
        config_path,
    })
}

async fn list_local_repo_files(dir: &Path) -> Result<Vec<PathBuf>, InitRunError> {
    let mut ret = Vec::new();
    let mut read_dir = tokio::fs::read_dir(dir).await?;
    while let Some(dir_entry) = read_dir.next_entry().await? {
        ret.push(dir_entry.path())
    }
    Ok(ret)
}

fn resolve_manifest_local_paths(
    manifest_dir: &Path,
    files: &[String],
) -> (Vec<PathBuf>, Vec<String>) {
    let mut resolved = Vec::with_capacity(files.len());
    let mut missing = Vec::new();
    for name in files {
        let path = manifest_dir.join(Path::new(name));
        if path.is_file() {
            resolved.push(path);
        } else {
            missing.push(name.clone());
        }
    }
    (resolved, missing)
}

fn normalize_repo_relative_path(path: &Path, original: &str) -> Result<PathBuf, InitRunError> {
    let mut parts: Vec<OsString> = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_os_string()),
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.pop().is_none() {
                    // Clamp to repo root if the manifest path tries to traverse above it.
                    continue;
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(InitRunError::InvalidMatformerManifestPath {
                    path: original.to_string(),
                });
            }
        }
    }
    if parts.is_empty() {
        return Err(InitRunError::InvalidMatformerManifestPath {
            path: original.to_string(),
        });
    }
    let mut normalized = PathBuf::new();
    for part in parts {
        normalized.push(part);
    }
    Ok(normalized)
}

fn repo_path_to_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn normalize_manifest_selection_for_repo(
    manifest_path: &str,
    selection: ManifestSelection,
) -> Result<ManifestSelection, InitRunError> {
    let manifest_dir = Path::new(manifest_path)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let mut seen = HashSet::new();
    let mut files = Vec::with_capacity(selection.files.len());
    for entry in selection.files {
        let entry = entry.trim_start_matches("./");
        let combined = manifest_dir.join(entry);
        let normalized = normalize_repo_relative_path(&combined, entry)?;
        let normalized = repo_path_to_string(&normalized);
        if seen.insert(normalized.clone()) {
            files.push(normalized);
        }
    }

    let config_entry = selection.config_path.trim_start_matches("./");
    let config_combined = manifest_dir.join(config_entry);
    let normalized_config = normalize_repo_relative_path(&config_combined, config_entry)?;
    let normalized_config = repo_path_to_string(&normalized_config);

    Ok(ManifestSelection {
        files,
        config_path: normalized_config,
    })
}

async fn resolve_matformer_local_repo_files(
    base: &Path,
    tier: u8,
    strategy: MatformerLoadStrategy,
) -> Result<Option<ResolvedRepoFiles>, InitRunError> {
    let base_exists = tokio::fs::try_exists(base).await?;

    if tier == 0 {
        if !base_exists {
            return Ok(None);
        }
        return Ok(Some(ResolvedRepoFiles {
            repo_files: list_local_repo_files(base).await?,
            uses_sliced_checkpoint: false,
            matformer_tier_for_loading: 0,
            checkpoint_path: Some(base.to_string_lossy().to_string()),
            manifest_base_intermediate_size: None,
            sliced_checkpoint_tier: None,
        }));
    }

    let manifest_path = base.join(MATFORMER_MANIFEST_NAME);
    if base_exists && tokio::fs::try_exists(&manifest_path).await? {
        let manifest_contents = tokio::fs::read_to_string(&manifest_path).await?;
        let manifest: MatformerManifest = serde_json::from_str(&manifest_contents)
            .map_err(InitRunError::MatformerManifestParse)?;

        if !matches!(strategy, MatformerLoadStrategy::Universal) {
            match select_manifest_tier_files(&manifest, tier) {
                Ok(selection) => {
                    let (repo_files, missing) =
                        resolve_manifest_local_paths(base, &selection.files);
                    if missing.is_empty() {
                        verify_manifest_file_hashes(
                            selection
                                .files
                                .iter()
                                .zip(repo_files.iter())
                                .filter_map(|(name, path)| {
                                    manifest
                                        .sha256
                                        .get(name)
                                        .map(|hash| (path.clone(), hash.clone()))
                                })
                                .collect(),
                        )
                        .await?;
                        let config_path = base.join(Path::new(&selection.config_path));
                        let checkpoint_path = config_path
                            .parent()
                            .map(|path| path.to_string_lossy().to_string())
                            .unwrap_or_else(|| config_path.to_string_lossy().to_string());
                        return Ok(Some(ResolvedRepoFiles {
                            repo_files,
                            uses_sliced_checkpoint: true,
                            matformer_tier_for_loading: 0,
                            checkpoint_path: Some(checkpoint_path),
                            manifest_base_intermediate_size: manifest
                                .matformer_base_intermediate_size,
                            sliced_checkpoint_tier: Some(tier),
                        }));
                    }

                    let err = InitRunError::MissingMatformerManifestFiles { missing };
                    if matches!(strategy, MatformerLoadStrategy::Sliced) {
                        return Err(err);
                    }
                    warn!(error = %err, "Manifest files missing; falling back to local lookup");
                }
                Err(err) => {
                    if matches!(strategy, MatformerLoadStrategy::Sliced) {
                        return Err(err);
                    }
                    warn!(error = %err, "Manifest unusable; falling back to local lookup");
                }
            }
        }
    }

    let (maybe_path, sliced_checkpoint) =
        resolve_matformer_local_repo_path(base, tier, strategy).await?;
    if let Some(repo_root) = maybe_path {
        let effective_tier_for_load = if sliced_checkpoint { 0 } else { tier };
        return Ok(Some(ResolvedRepoFiles {
            repo_files: list_local_repo_files(&repo_root).await?,
            uses_sliced_checkpoint: sliced_checkpoint,
            matformer_tier_for_loading: effective_tier_for_load,
            checkpoint_path: Some(repo_root.to_string_lossy().to_string()),
            manifest_base_intermediate_size: None,
            sliced_checkpoint_tier: sliced_checkpoint.then_some(tier),
        }));
    }

    Ok(None)
}

fn manifest_missing_repo_files(repo_files: &[String], expected: &[String]) -> Vec<String> {
    let available: HashSet<&str> = repo_files.iter().map(|file| file.as_str()).collect();
    expected
        .iter()
        .filter(|name| !available.contains(name.as_str()))
        .cloned()
        .collect()
}

async fn resolve_matformer_hub_repo_files(
    repo_id: &str,
    revision: Option<String>,
    tier: u8,
    strategy: MatformerLoadStrategy,
    hub_read_token: Option<String>,
    hub_max_concurrent_downloads: usize,
) -> Result<ResolvedRepoFiles, InitRunError> {
    if tier == 0 || matches!(strategy, MatformerLoadStrategy::Universal) {
        let repo_files = download_model_repo_async(
            repo_id,
            revision,
            None,
            hub_read_token,
            Some(hub_max_concurrent_downloads),
            false,
        )
        .await?;
        return Ok(ResolvedRepoFiles {
            repo_files,
            uses_sliced_checkpoint: false,
            matformer_tier_for_loading: tier,
            checkpoint_path: Some(repo_id.to_string()),
            manifest_base_intermediate_size: None,
            sliced_checkpoint_tier: None,
        });
    }

    let repo_listing = list_model_repo_files_async(
        repo_id,
        revision.clone(),
        None,
        hub_read_token.clone(),
        false,
    )
    .await?;

    let manifest_paths: Vec<String> = repo_listing
        .iter()
        .filter(|path| path.ends_with(MATFORMER_MANIFEST_NAME))
        .cloned()
        .collect();

    if manifest_paths.is_empty() {
        if matches!(strategy, MatformerLoadStrategy::Sliced) {
            return Err(InitRunError::MissingMatformerManifest {
                path: format!("{repo_id}:{MATFORMER_MANIFEST_NAME}"),
            });
        }
        let repo_files = download_model_repo_async(
            repo_id,
            revision,
            None,
            hub_read_token,
            Some(hub_max_concurrent_downloads),
            false,
        )
        .await?;
        return Ok(ResolvedRepoFiles {
            repo_files,
            uses_sliced_checkpoint: false,
            matformer_tier_for_loading: tier,
            checkpoint_path: Some(repo_id.to_string()),
            manifest_base_intermediate_size: None,
            sliced_checkpoint_tier: None,
        });
    }

    if manifest_paths.len() > 1 {
        return Err(InitRunError::MultipleMatformerManifests {
            paths: manifest_paths,
        });
    }

    let manifest_path = manifest_paths[0].clone();
    let manifest_files = download_model_repo_files_async(
        repo_id,
        revision.clone(),
        None,
        hub_read_token.clone(),
        Some(hub_max_concurrent_downloads),
        false,
        &[manifest_path.clone()],
    )
    .await?;
    let manifest_contents = tokio::fs::read_to_string(&manifest_files[0]).await?;
    let manifest: MatformerManifest = serde_json::from_str(&manifest_contents)
        .map_err(InitRunError::MatformerManifestParse)?;

    let selection = match select_manifest_tier_files(&manifest, tier) {
        Ok(selection) => selection,
        Err(err) => {
            if matches!(strategy, MatformerLoadStrategy::Sliced) {
                return Err(err);
            }
            warn!(error = %err, "Manifest tier unavailable; falling back to universal");
            let repo_files = download_model_repo_async(
                repo_id,
                revision,
                None,
                hub_read_token,
                Some(hub_max_concurrent_downloads),
                false,
            )
            .await?;
            return Ok(ResolvedRepoFiles {
                repo_files,
                uses_sliced_checkpoint: false,
                matformer_tier_for_loading: tier,
                checkpoint_path: Some(repo_id.to_string()),
                manifest_base_intermediate_size: None,
                sliced_checkpoint_tier: None,
            });
        }
    };

    // Manifest hash keys are relative to the manifest, and repo paths are not.
    let manifest_dir = Path::new(&manifest_path)
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_path_buf();
    let expected_hashes: Vec<(String, String)> = selection
        .files
        .iter()
        .filter_map(|name| {
            let hash = manifest.sha256.get(name)?;
            let combined = manifest_dir.join(name.trim_start_matches("./"));
            let normalized = normalize_repo_relative_path(&combined, name).ok()?;
            Some((repo_path_to_string(&normalized), hash.clone()))
        })
        .collect();

    let selection = match normalize_manifest_selection_for_repo(&manifest_path, selection) {
        Ok(selection) => selection,
        Err(err) => {
            if matches!(strategy, MatformerLoadStrategy::Sliced) {
                return Err(err);
            }
            warn!(error = %err, "Manifest paths invalid; falling back to universal");
            let repo_files = download_model_repo_async(
                repo_id,
                revision,
                None,
                hub_read_token,
                Some(hub_max_concurrent_downloads),
                false,
            )
            .await?;
            return Ok(ResolvedRepoFiles {
                repo_files,
                uses_sliced_checkpoint: false,
                matformer_tier_for_loading: tier,
                checkpoint_path: Some(repo_id.to_string()),
                manifest_base_intermediate_size: None,
                sliced_checkpoint_tier: None,
            });
        }
    };

    let missing = manifest_missing_repo_files(&repo_listing, &selection.files);
    if !missing.is_empty() {
        let err = InitRunError::MissingMatformerManifestFiles { missing };
        if matches!(strategy, MatformerLoadStrategy::Sliced) {
            return Err(err);
        }
        warn!(error = %err, "Manifest files missing; falling back to universal");
        let repo_files = download_model_repo_async(
            repo_id,
            revision,
            None,
            hub_read_token,
            Some(hub_max_concurrent_downloads),
            false,
        )
        .await?;
        return Ok(ResolvedRepoFiles {
            repo_files,
            uses_sliced_checkpoint: false,
            matformer_tier_for_loading: tier,
            checkpoint_path: Some(repo_id.to_string()),
            manifest_base_intermediate_size: None,
            sliced_checkpoint_tier: None,
        });
    }

    let repo_files = download_model_repo_files_async(
        repo_id,
        revision,
        None,
        hub_read_token,
        Some(hub_max_concurrent_downloads),
        false,
        &selection.files,
    )
    .await?;
    verify_manifest_file_hashes(
        expected_hashes
            .into_iter()
            .filter_map(|(name, hash)| {
                let path = repo_files.iter().find(|path| path.ends_with(&name))?;
                Some((path.clone(), hash))
            })
            .collect(),
    )
    .await?;
    let checkpoint_path = Path::new(&selection.config_path)
        .parent()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or(selection.config_path);
    Ok(ResolvedRepoFiles {
        repo_files,
        uses_sliced_checkpoint: true,
        matformer_tier_for_loading: 0,
        checkpoint_path: Some(checkpoint_path),
        manifest_base_intermediate_size: manifest.matformer_base_intermediate_size,
        sliced_checkpoint_tier: Some(tier),
    })
}

async fn resolve_matformer_repo_files(
    repo_id: &str,
    revision: Option<String>,
    tier: u8,
    strategy: MatformerLoadStrategy,
    hub_read_token: Option<String>,
    hub_max_concurrent_downloads: usize,
) -> Result<ResolvedRepoFiles, InitRunError> {
    if revision.is_none() {
        let local_path = PathBuf::from(repo_id);
        if let Some(resolved) =
            resolve_matformer_local_repo_files(&local_path, tier, strategy).await?
        {
            return Ok(resolved);
        }
    }

    resolve_matformer_hub_repo_files(
        repo_id,
        revision,
        tier,
        strategy,
        hub_read_token,
        hub_max_concurrent_downloads,
    )
    .await
}

fn apply_matformer_checkpoint_tier_overrides(
    checkpoint_config: &serde_json::Value,
    load_strategy: &MatformerLoadStrategy,
    uses_sliced_checkpoint: bool,
    matformer_tier_for_loading: u8,
) -> (bool, u8) {
    let checkpoint_metadata = infer_matformer_checkpoint_metadata(checkpoint_config);
    let checkpoint_tier = checkpoint_metadata.tier;
    let uses_sliced_checkpoint =
        uses_sliced_checkpoint || checkpoint_tier.map(|tier| tier > 0).unwrap_or(false);
    let mut matformer_tier_for_loading = matformer_tier_for_loading;

    if uses_sliced_checkpoint
        && matches!(load_strategy, MatformerLoadStrategy::Auto | MatformerLoadStrategy::Sliced)
    {
        matformer_tier_for_loading = 0;
    }

    (uses_sliced_checkpoint, matformer_tier_for_loading)
}

/// Validate that the requested tier configuration won't cause double-slicing.
/// Returns an error if a sliced checkpoint would be further sliced.
fn validate_no_double_slicing(
    checkpoint_config: &serde_json::Value,
    cli_tier: u8,
    load_strategy: &MatformerLoadStrategy,
    uses_sliced_checkpoint: bool,
) -> Result<(), InitRunError> {
    let checkpoint_metadata = infer_matformer_checkpoint_metadata(checkpoint_config);
    let checkpoint_tier = checkpoint_metadata.tier;

    // If we detected this is a sliced checkpoint (via naming or explicit tier)
    let is_sliced = uses_sliced_checkpoint || checkpoint_tier.map(|t| t > 0).unwrap_or(false);

    if is_sliced && cli_tier > 0 {
        // Check if user is trying to further slice an already-sliced checkpoint
        match load_strategy {
            MatformerLoadStrategy::Universal => {
                // Universal mode bypasses sliced detection - this is dangerous
                return Err(InitRunError::DoubleSlicingDetected {
                    checkpoint_tier: checkpoint_tier.unwrap_or(0),
                    cli_tier,
                    hint: "Use --matformer-load-strategy auto or specify --matformer-tier 0".to_string(),
                });
            }
            MatformerLoadStrategy::Auto | MatformerLoadStrategy::Sliced => {
                // These modes correctly set effective_tier=0 for sliced checkpoints
                // Double-slicing is prevented by the effective_tier logic
            }
        }
    }

    Ok(())
}

fn schema_hash_for_config(
    architecture: model::LLMArchitecture,
    config: &serde_json::Value,
    parameter_names: &[String],
) -> [u8; 32] {
    let arch = format!("{architecture:?}");
    let mut names = parameter_names.to_vec();
    names.sort();
    let config_json =
        serde_json::to_string(config).expect("config value is always serializable");
    let mut data = Vec::with_capacity(arch.len() + config_json.len() + names.len() * 32);
    data.extend_from_slice(arch.as_bytes());
    data.push(0);
    data.extend_from_slice(config_json.as_bytes());
    data.push(0);
    for name in names {
        data.extend_from_slice(name.as_bytes());
        data.push(0);
    }
    sha256(&data)
}

fn canonicalize_config_for_schema(
    mut config: serde_json::Value,
    matformer_tier: u8,
    uses_sliced_checkpoint: bool,
) -> serde_json::Value {
    let mut base_intermediate_size = infer_matformer_checkpoint_metadata(&config)
        .base_intermediate_size;
    if let Some(obj) = config.as_object_mut() {
        if uses_sliced_checkpoint {
            let base = base_intermediate_size
                .or_else(|| obj.get("intermediate_size").and_then(|v| v.as_u64()))
                .or_else(|| {
                    obj.get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .and_then(|size| size.checked_shl(matformer_tier as u32))
                });
            if let Some(base) = base {
                obj.insert("intermediate_size".to_string(), serde_json::Value::from(base));
                base_intermediate_size = Some(base);
            }
        }

        if let Some(base) = base_intermediate_size
            .or_else(|| obj.get("intermediate_size").and_then(|v| v.as_u64()))
        {
            obj.insert(
                "matformer_base_intermediate_size".to_string(),
                serde_json::Value::from(base),
            );
        }
        // ALWAYS set matformer_tier to 0 for canonical comparison
        // This ensures full checkpoints (no field) and sliced checkpoints (field=0)
        // produce the same hash for heterogeneous tier training
        obj.insert("matformer_tier".to_string(), serde_json::Value::from(0));
    }
    config
}

#[derive(Debug, Error)]
pub enum InitRunError {
    #[error("No model provided in Coordinator state, nothing to do.")]
    NoModel,

    #[error("Model is Ephemeral, it's impossible to join this run.")]
    ModelIsEphemeral,

    #[error("failed to read local model info: {0}")]
    LocalModelLoad(#[from] io::Error),

    #[error("failed to read HF model info: {0}")]
    HfModelLoad(#[from] hf_hub::api::tokio::ApiError),

    #[error("model loading thread crashed")]
    ModelLoadingThreadCrashed(JoinError),

    #[error("failed to load model: {0}")]
    ModelLoad(#[from] ModelLoadError),

    #[error("Couldn't load tokenizer: {0}")]
    TokenizerLoad(#[from] AutoTokenizerError),

    // TODO refactor data provider for real errors
    #[error("Couldn't initialize data provider: {0}")]
    DataProviderConnect(anyhow::Error),

    #[error("wandb setup thread crashed")]
    WandbThreadCrashed(JoinError),

    #[error("wandb failed to create run: {0}")]
    WandbLoad(#[from] wandb::ApiError),

    #[error("could not parse config: {0}")]
    FailedToParseConfig(#[from] serde_json::Error),

    #[error("failed to parse MatFormer manifest: {0}")]
    MatformerManifestParse(serde_json::Error),

    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[cfg(feature = "python")]
    #[error("Python distributed error: {0}")]
    PythonDistributedError(#[from] psyche_modeling::PythonDistributedCausalLMError),

    #[cfg(feature = "python")]
    #[error("Python model error: {0}")]
    PythonModelError(#[from] psyche_modeling::PythonCausalLMError),

    #[cfg(feature = "python")]
    #[error("Python distributed trainer error: {0}")]
    PythonDistributedTrainerError(#[from] psyche_modeling::PythonDistributedTrainerError),

    #[error("Required MatFormer tier checkpoint for tier {tier} not found at {path}")]
    MissingMatformerTierCheckpoint { path: String, tier: u8 },

    #[error("Double-slicing detected: checkpoint is tier {checkpoint_tier}, CLI requests tier {cli_tier}. {hint}")]
    DoubleSlicingDetected {
        checkpoint_tier: u8,
        cli_tier: u8,
        hint: String,
    },

    #[error("MatFormer manifest missing at {path}")]
    MissingMatformerManifest { path: String },

    #[error("MatFormer manifest schema version {version} is unsupported")]
    UnsupportedMatformerManifestVersion { version: u32 },

    #[error("MatFormer manifest missing tier {tier}. Available tiers: {available:?}")]
    MissingMatformerManifestTier { tier: u8, available: Vec<u8> },

    #[error("MatFormer manifest missing config.json for tier {tier}")]
    MatformerManifestMissingConfig { tier: u8 },

    #[error("MatFormer manifest missing safetensors for tier {tier}")]
    MatformerManifestMissingWeights { tier: u8 },

    #[error("MatFormer manifest contains invalid path: {path}")]
    InvalidMatformerManifestPath { path: String },

    #[error("MatFormer manifest referenced missing files: {missing:?}")]
    MissingMatformerManifestFiles { missing: Vec<String> },

    #[error("SHA-256 of {path} is {actual}, but the MatFormer manifest says {expected}")]
    MatformerManifestHashMismatch {
        path: String,
        expected: String,
        actual: String,
    },

    #[error("MatFormer tier {tier} leaves no FFN neurons in the model from the peer")]
    InvalidMatformerTierForPeerConfig { tier: u8 },

    #[error("Multiple MatFormer manifests found: {paths:?}")]
    MultipleMatformerManifests { paths: Vec<String> },
}

enum RawLoadedModelType {
    ParallelNativeModels(Vec<Box<dyn CausalLM>>),
    #[cfg(feature = "python")]
    Python(psyche_modeling::PythonCausalLM),
    #[cfg(feature = "python")]
    PythonDistributed(psyche_modeling::PythonDistributedCausalLM),
}

struct RawLoadedModel {
    models: RawLoadedModelType,
    tokenizer: Arc<Tokenizer>,
    model_task_runner: ModelTaskRunner,
    checkpoint_extra_files: Vec<PathBuf>,
    parameter_names: Arc<Vec<String>>,
    matformer_effective_tier: u8,
    matformer_base_intermediate_size: Option<u64>,
    matformer_holds_full_ffn_weights: bool,
    schema_hash_canonical: [u8; 32],
}

type OneshotModelParameterSender = oneshot::Sender<HashMap<String, Tensor>>;
type OneShotModelConfigSender = oneshot::Sender<(String, Tokenizer)>;

pub struct RunInitConfigAndIO<T: NodeIdentity, A: AuthenticatableIdentity> {
    pub init_config: RunInitConfig<T, A>,

    pub tx_health_check: UnboundedSender<HealthChecks<T>>,
    pub tx_witness: UnboundedSender<OpportunisticData>,
    pub tx_checkpoint: UnboundedSender<model::HubRepo>,
    pub tx_schema: UnboundedSender<ModelSchemaInfo>,
    pub tx_model: UnboundedSender<HashMap<String, Tensor>>,
    pub tx_parameters_req: UnboundedSender<(Vec<String>, Option<i64>, OneshotModelParameterSender)>,
    pub tx_config: UnboundedSender<(String, String)>,
    pub tx_distro_result: UnboundedSender<DistroBroadcastAndPayload>,
    pub tx_request_download: UnboundedSender<(BlobTicket, Tag)>,
    pub tx_request_model_config: UnboundedSender<OneShotModelConfigSender>,
    pub tx_broadcast_finished: UnboundedSender<FinishedBroadcast>,

    pub metrics: Arc<ClientMetrics>,
}

struct LoadedCheckpoint {
    source: PretrainedSource<AutoConfig>,
    /// Full config JSON from a peer. The parsed config drops fields that it does not
    /// know, and the schema hash must see the same fields on all clients.
    peer_config_json: Option<serde_json::Value>,
    tokenizer: Arc<Tokenizer>,
    checkpoint_extra_files: Vec<PathBuf>,
    uses_sliced_checkpoint: bool,
    matformer_tier_for_loading: u8,
    matformer_checkpoint_path: Option<String>,
    manifest_base_intermediate_size: Option<u64>,
    sliced_checkpoint_tier: Option<u8>,
}

impl<T: NodeIdentity, A: AuthenticatableIdentity + 'static> RunInitConfigAndIO<T, A> {
    /// Call this on first warmup - when we need to enter the run, we have to load the model, connect to the data server, etc
    pub async fn init_run(
        self,
        state: Coordinator<T>,
    ) -> Result<StepStateMachine<T, A>, InitRunError> {
        let Self {
            init_config,
            tx_witness,
            tx_health_check,
            tx_checkpoint,
            tx_schema,
            tx_model,
            tx_config,
            tx_parameters_req,
            tx_distro_result,
            tx_request_download,
            tx_request_model_config,
            tx_broadcast_finished,
            metrics,
        } = self;

        tch::manual_seed(1337);

        // Check device availability early
        if !init_config.device.is_probably_available() {
            return Err(InitRunError::ModelLoad(
                psyche_modeling::ModelLoadError::UnavailbleDevice(init_config.device),
            ));
        }

        let model::Model::LLM(llm) = state.model;

        if matches!(llm.optimizer, OptimizerDefinition::AdamW { .. })
            && state.config.min_clients > 1
        {
            warn!(
                "This run uses AdamW with min_clients > 1. AdamW clients do not exchange gradients, \
                 so each client trains alone. Use the Distro optimizer for shared training."
            );
        }

        let hub_read_token = init_config.hub_read_token.clone();
        let hub_max_concurrent_downloads = init_config.hub_max_concurrent_downloads;
        let data_future = async {
            debug!("Setting up data provider from {:?}", llm.data_location);
            let data_provider = match llm.data_location {
                LLMTrainingDataLocation::Server(data_server) => DataProvider::Server(
                    DataProviderTcpClient::connect(
                        (&data_server).into(),
                        init_config.network_identity,
                        init_config.private_key,
                    )
                    .await?,
                ),
                LLMTrainingDataLocation::Local(url) => {
                    let url: String = (&url).into();
                    let dir = if std::fs::exists(&url).unwrap_or_default() {
                        PathBuf::from(url)
                    } else {
                        download_dataset_repo_async(
                            url.clone(),
                            None,
                            None,
                            hub_read_token,
                            Some(hub_max_concurrent_downloads),
                            false,
                        )
                        .await?
                        .first()
                        .ok_or(anyhow::anyhow!("No files downloaded for {url}"))?
                        .parent()
                        .unwrap()
                        .into()
                    };
                    DataProvider::Local(LocalDataProvider::new_from_directory(
                        dir,
                        TokenSize::TwoBytes,
                        llm.max_seq_len as usize,
                        Shuffle::DontShuffle,
                    )?)
                }
                LLMTrainingDataLocation::Dummy => {
                    DataProvider::Dummy(DummyDataProvider::new(
                        TokenSize::TwoBytes,
                        llm.max_seq_len as usize,
                        u64::MAX,
                    ))
                }
                LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
                    location,
                    token_size_in_bytes,
                    shuffle,
                }) => {
                    let file_urls = FileURLs::from_location(&location).await?;
                    DataProvider::Http(HttpDataProvider::new(
                        file_urls,
                        token_size_in_bytes,
                        llm.max_seq_len,
                        shuffle,
                    )?)
                }
                LLMTrainingDataLocation::WeightedHttp(config_url) => DataProvider::WeightedHttp(
                    WeightedDataProvider::<HttpDataProvider>::from_config_url(
                        &String::from(&config_url),
                        llm.max_seq_len,
                    )
                    .await?,
                ),
                LLMTrainingDataLocation::Preprocessed(url) => {
                    let url: String = (&url).into();
                    let dir = if std::fs::exists(&url).unwrap_or_default() {
                        PathBuf::from(url)
                    } else {
                        download_dataset_repo_async(
                            url.clone(),
                            None,
                            None,
                            hub_read_token,
                            Some(hub_max_concurrent_downloads),
                            false,
                        )
                        .await?
                        .first()
                        .ok_or(anyhow::anyhow!("No files downloaded for {url}"))?
                        .parent()
                        .unwrap()
                        .into()
                    };
                    DataProvider::Preprocessed(PreprocessedDataProvider::new_from_directory(
                        dir,
                        llm.max_seq_len as usize,
                        Shuffle::DontShuffle,
                        Some(Split::Train),
                        None,
                    )?)
                }
            };
            Ok(data_provider)
        };

        let model_future: JoinHandle<Result<RawLoadedModel, InitRunError>> = match &llm.architecture
        {
            model::LLMArchitecture::HfLlama
            | model::LLMArchitecture::HfDeepseek
            | model::LLMArchitecture::HfNanoGPT
            | model::LLMArchitecture::HfAuto => match &llm.checkpoint {
                model::Checkpoint::Dummy(_) => tokio::spawn(async move {
                    let tokenizer = Arc::new(Tokenizer::new(ModelWrapper::WordLevel(
                        WordLevel::builder().build().unwrap(),
                    )));

                    let mut parameter_names = LlamaConfig::dummy().get_parameter_names();
                    parameter_names.sort();
                    let parameter_names = Arc::new(parameter_names);

                    #[allow(clippy::arc_with_non_send_sync)]
                    let config = PretrainedSource::ConfigAndTensors(
                        AutoConfig::Llama(LlamaConfig::dummy()),
                        Arc::new(psyche_modeling::get_dummy_parameters()),
                    )
                    .serialize_config()?;
                    // A client that joins later gets this config over P2P and must compute the same hash.
                    let schema_hash_canonical = schema_hash_for_config(
                        llm.architecture,
                        &canonicalize_config_for_schema(serde_json::from_str(&config)?, 0, false),
                        &parameter_names,
                    );

                    let model = RawLoadedModel {
                        models: RawLoadedModelType::ParallelNativeModels(
                            (0..(init_config.data_parallelism * init_config.tensor_parallelism))
                                .map(|_| {
                                    if let Some(training_delay) =
                                        init_config.dummy_training_delay_secs
                                    {
                                        Box::new(DummyModel::new(training_delay))
                                            as Box<dyn CausalLM>
                                    } else {
                                        Box::new(DummyModel::default()) as Box<dyn CausalLM>
                                    }
                                })
                                .collect(),
                        ),
                        tokenizer: tokenizer.clone(),
                        checkpoint_extra_files: vec![],
                        model_task_runner: ModelTaskRunner::new(
                            vec![],
                            false,
                            tokenizer.clone(),
                            None,
                            0,
                        ),
                        parameter_names: parameter_names.clone(),
                        matformer_effective_tier: 0,
                        matformer_base_intermediate_size: None,
                        matformer_holds_full_ffn_weights: true,
                        schema_hash_canonical,
                    };
                    let tokenizer = tokenizer.to_string(false).unwrap();
                    info!("Config Uploaded: {}", config);
                    tx_config.send((config, tokenizer)).unwrap();
                    Ok(model)
                }),
                model::Checkpoint::Hub(_) | model::Checkpoint::P2P(_) => tokio::spawn(async move {
                    let checkpoint = llm.checkpoint;
                    let loaded_checkpoint = match checkpoint {
                        model::Checkpoint::Hub(hub_repo) => {
                            let repo_id: String = (&hub_repo.repo_id).into();
                            let revision = hub_repo.revision.map(|bytes| (&bytes).into());
                            let revision_for_log = revision.clone();
                            let resolved = resolve_matformer_repo_files(
                                &repo_id,
                                revision,
                                init_config.matformer_tier,
                                init_config.matformer_load_strategy,
                                init_config.hub_read_token.clone(),
                                init_config.hub_max_concurrent_downloads,
                            )
                            .await?;
                            let repo_files = resolved.repo_files;
                            let checkpoint_path = resolved.checkpoint_path;
                            let effective_tier_for_load = resolved.matformer_tier_for_loading;
                            let sliced_checkpoint = resolved.uses_sliced_checkpoint;
                            let manifest_base_intermediate_size =
                                resolved.manifest_base_intermediate_size;
                            let sliced_checkpoint_tier = resolved.sliced_checkpoint_tier;

                            info!(
                                "Resolved checkpoint {}, revision: {:?}",
                                hub_repo.repo_id, revision_for_log
                            );
                            let checkpoint_extra_files = repo_files
                                .iter()
                                .filter(|file| {
                                    file.ends_with("config.json")
                                        || file.ends_with("tokenizer.json")
                                        || file.ends_with("tokenizer_config.json")
                                        || file.ends_with("special_tokens_map.json")
                                        || file.ends_with("generation_config.json")
                                        || file.ends_with(".py")
                                })
                                .cloned()
                                .collect();
                            let tokenizer = Arc::new(auto_tokenizer(&repo_files)?);
                            LoadedCheckpoint {
                                source: PretrainedSource::<AutoConfig>::RepoFiles(repo_files),
                                peer_config_json: None,
                                tokenizer,
                                checkpoint_extra_files,
                                uses_sliced_checkpoint: sliced_checkpoint,
                                matformer_tier_for_loading: effective_tier_for_load,
                                matformer_checkpoint_path: checkpoint_path,
                                manifest_base_intermediate_size,
                                sliced_checkpoint_tier,
                            }
                        }
                        model::Checkpoint::P2P(_) => {
                            let (tx_model_config_response, rx_model_config_response) =
                                oneshot::channel();
                            info!("Checkpoint is p2p, requesting model config over network");

                            tx_request_model_config
                                .send(tx_model_config_response)
                                .unwrap();

                            let (model_config, tokenizer) =
                                rx_model_config_response.await.unwrap();
                            debug!("Got p2p info, model_config: {}", model_config);

                            let peer_config: serde_json::Value =
                                serde_json::from_str(&model_config)?;
                            let plan = match llm.architecture {
                                model::LLMArchitecture::HfLlama
                                | model::LLMArchitecture::HfNanoGPT => plan_p2p_load(
                                    peer_config,
                                    init_config.matformer_tier,
                                    matches!(
                                        init_config.matformer_load_strategy,
                                        MatformerLoadStrategy::Universal
                                    ),
                                )
                                .ok_or(InitRunError::InvalidMatformerTierForPeerConfig {
                                    tier: init_config.matformer_tier,
                                })?,
                                _ => {
                                    P2pLoadPlan::unchanged(peer_config, init_config.matformer_tier)
                                }
                            };
                            let model_config = serde_json::to_string(&plan.config)?;

                            let model_config = match llm.architecture {
                                model::LLMArchitecture::HfLlama => {
                                    AutoConfig::Llama(serde_json::from_str(&model_config)?)
                                }
                                model::LLMArchitecture::HfDeepseek => {
                                    AutoConfig::Deepseek(serde_json::from_str(&model_config)?)
                                }
                                model::LLMArchitecture::HfNanoGPT => {
                                    AutoConfig::NanoGPT(serde_json::from_str(&model_config)?)
                                }
                                model::LLMArchitecture::HfAuto => {
                                    #[cfg(feature = "python")]
                                    {
                                        AutoConfig::Auto(serde_json::from_str::<
                                            psyche_modeling::PythonModelConfig,
                                        >(
                                            &model_config
                                        )?)
                                    }

                                    #[cfg(not(feature = "python"))]
                                    {
                                        return Err(InitRunError::UnsupportedArchitecture(
                                            "HfAuto".to_string(),
                                        ));
                                    }
                                }
                            };
                            let parameter_names = model_config.get_parameter_names();
                            info!(
                                "Requesting {} parameters over p2p network",
                                parameter_names.len()
                            );

                            let (tx_params_response, rx_params_response) = oneshot::channel();
                            tx_parameters_req
                                .send((
                                    parameter_names,
                                    plan.required_ffn_width,
                                    tx_params_response,
                                ))
                                .unwrap();
                            #[allow(clippy::arc_with_non_send_sync)]
                            let parameters = Arc::new(rx_params_response.await.unwrap());

                            LoadedCheckpoint {
                                source: PretrainedSource::<AutoConfig>::ConfigAndTensors(
                                    model_config,
                                    parameters,
                                ),
                                peer_config_json: Some(plan.config),
                                tokenizer: Arc::new(tokenizer),
                                checkpoint_extra_files: vec![],
                                uses_sliced_checkpoint: plan.uses_sliced_checkpoint,
                                matformer_tier_for_loading: plan.matformer_tier_for_loading,
                                matformer_checkpoint_path: Some("p2p".to_string()),
                                manifest_base_intermediate_size: plan.base_intermediate_size,
                                sliced_checkpoint_tier: plan.sliced_checkpoint_tier,
                            }
                        }
                        _ => unreachable!(),
                    };

                    let source = loaded_checkpoint.source;
                    let peer_config_json = loaded_checkpoint.peer_config_json;
                    let tokenizer = loaded_checkpoint.tokenizer;
                    let checkpoint_extra_files = loaded_checkpoint.checkpoint_extra_files;
                    let mut uses_sliced_checkpoint = loaded_checkpoint.uses_sliced_checkpoint;
                    let mut matformer_tier_for_loading =
                        loaded_checkpoint.matformer_tier_for_loading;
                    let matformer_checkpoint_path = loaded_checkpoint.matformer_checkpoint_path;
                    let manifest_base_intermediate_size =
                        loaded_checkpoint.manifest_base_intermediate_size;
                    let sliced_checkpoint_tier = loaded_checkpoint.sliced_checkpoint_tier;

                    info!("Loading model...");

                    let model_task_runner = ModelTaskRunner::new(
                        init_config.eval_tasks,
                        init_config.prompt_task,
                        tokenizer.clone(),
                        init_config.eval_task_max_docs,
                        // if doing python fsdp we only have one effective dp rank for inference
                        if init_config.data_parallelism > 1
                            && llm.architecture == model::LLMArchitecture::HfAuto
                        {
                            1
                        } else {
                            init_config.data_parallelism
                        },
                    );

                    let serialized_config = source.serialize_config()?;
                    let mut config_json: serde_json::Value = match peer_config_json {
                        Some(config) => config,
                        None => serde_json::from_str(&serialized_config)?,
                    };
                    ensure_matformer_checkpoint_metadata(
                        &mut config_json,
                        manifest_base_intermediate_size,
                        sliced_checkpoint_tier,
                    );
                    (uses_sliced_checkpoint, matformer_tier_for_loading) =
                        apply_matformer_checkpoint_tier_overrides(
                            &config_json,
                            &init_config.matformer_load_strategy,
                            uses_sliced_checkpoint,
                            matformer_tier_for_loading,
                        );
                    // Validate no double-slicing before model load
                    validate_no_double_slicing(
                        &config_json,
                        init_config.matformer_tier,
                        &init_config.matformer_load_strategy,
                        uses_sliced_checkpoint,
                    )?;
                    let mut parameter_names = match llm.architecture {
                        model::LLMArchitecture::HfLlama => {
                            let config: LlamaConfig = serde_json::from_str(&serialized_config)?;
                            config.get_parameter_names()
                        }
                        model::LLMArchitecture::HfDeepseek => {
                            let config: DeepseekConfig =
                                serde_json::from_str(&serialized_config)?;
                            config.get_parameter_names()
                        }
                        model::LLMArchitecture::HfNanoGPT => {
                            let config: NanoGPTConfig =
                                serde_json::from_str(&serialized_config)?;
                            config.get_parameter_names()
                        }
                        model::LLMArchitecture::HfAuto => {
                            #[cfg(feature = "python")]
                            {
                                let config: psyche_modeling::PythonModelConfig =
                                    serde_json::from_str(&serialized_config)?;
                                config.get_parameter_names()
                            }

                            #[cfg(not(feature = "python"))]
                            {
                                return Err(InitRunError::UnsupportedArchitecture(
                                    "HfAuto".to_string(),
                                ));
                            }
                        }
                    };
                    parameter_names.sort();
                    let parameter_names = Arc::new(parameter_names);
                    let attn_implementation: Option<AttentionImplementation> = match llm.data_type {
                        model::LLMTrainingDataType::Finetuning => {
                            #[cfg(feature = "parallelism")]
                            {
                                // use varlen backend if available
                                Some(AttentionImplementation::FlashAttention2)
                            }

                            #[cfg(not(feature = "parallelism"))]
                            None
                        }
                        model::LLMTrainingDataType::Pretraining => None,
                    };

                    let raw_loaded_model_type: RawLoadedModelType =
                        if llm.architecture == model::LLMArchitecture::HfAuto {
                            #[cfg(feature = "python")]
                            {
                                let dp = init_config.data_parallelism;
                                let tp = init_config.tensor_parallelism;

                                tokio::task::spawn_blocking(move || {
                                    if tp != 1 || dp != 1 {
                                        psyche_modeling::PythonDistributedCausalLM::new(
                                            "hf-auto".to_string(),
                                            source.try_into()?,
                                            tch::Device::cuda_if_available(),
                                            attn_implementation.unwrap_or_default(),
                                            psyche_modeling::ParallelismConfig { dp, tp },
                                            Some(llm.max_seq_len as usize),
                                            init_config.sidecar_port,
                                            None,
                                        )
                                        .map(RawLoadedModelType::PythonDistributed)
                                        .map_err(InitRunError::PythonDistributedError)
                                    } else {
                                        let device = init_config.device.device_for_rank(0).ok_or_else(
                                            || ModelLoadError::NoDeviceForRank(0, init_config.device),
                                        )?;
                                        psyche_modeling::PythonCausalLM::new(
                                            "hf-auto",
                                            &source.try_into()?,
                                            device,
                                            attn_implementation.unwrap_or_default(),
                                            None,
                                            Some(llm.max_seq_len as usize),
                                        )
                                        .map(RawLoadedModelType::Python)
                                        .map_err(InitRunError::PythonModelError)
                                    }
                                })
                                .await
                                .map_err(InitRunError::ModelLoadingThreadCrashed)??
                            }

                            #[cfg(not(feature = "python"))]
                            {
                                return Err(InitRunError::UnsupportedArchitecture(
                                    "HfAuto".to_string(),
                                ));
                            }
                        } else {
                            let mut futures: Vec<
                                JoinHandle<Result<Box<dyn CausalLM>, ModelLoadError>>,
                            > = Vec::with_capacity(
                                init_config.data_parallelism * init_config.tensor_parallelism,
                            );
                            let devices = init_config.device.clone();
                            let matformer_tier = matformer_tier_for_loading;
                            let helper_fraction = init_config.matformer_helper_fraction;
                            let helper_rotation_interval =
                                init_config.matformer_helper_rotation_interval;

                            for dp in 0..init_config.data_parallelism {
                                let communicator_id: Option<CommunicatorId> =
                                    match init_config.tensor_parallelism {
                                        0 | 1 => None,
                                        #[cfg(feature = "parallelism")]
                                        _ => Some(tch::CStore::new().into()),
                                        #[cfg(not(feature = "parallelism"))]
                                        _ => unimplemented!(),
                                    };
                                for tp in 0..init_config.tensor_parallelism {
                                    let tensor_parallelism_world =
                                        communicator_id.as_ref().map(|communicator_id| {
                                            (
                                                communicator_id.clone(),
                                                tp,
                                                init_config.tensor_parallelism,
                                            )
                                        });
                                    let source = source.clone();
                                    let rank = dp * init_config.tensor_parallelism + tp;
                                    let devices = devices.clone();
                                    let device = devices.device_for_rank(rank);
                                    futures.push(tokio::task::spawn_blocking(move || {
                                        let device = device.ok_or_else(|| {
                                            ModelLoadError::NoDeviceForRank(rank, devices)
                                        })?;
                                        let kind = match device {
                                            // MPS has incomplete BF16 support; prefer FP32 for correctness.
                                            Device::Mps => Kind::Float,
                                            _ => Kind::BFloat16,
                                        };
                                        match llm.architecture {
                                            model::LLMArchitecture::HfLlama => {
                                                LlamaForCausalLM::from_pretrained_with_matformer_config(
                                                    &source.try_into()?,
                                                    Some(kind),
                                                    attn_implementation,
                                                    Some(device),
                                                    tensor_parallelism_world,
                                                    Some(llm.max_seq_len as usize),
                                                    matformer_tier,
                                                    helper_fraction,
                                                    helper_rotation_interval,
                                                )
                                                .map(|x| Box::new(x) as Box<dyn CausalLM>)
                                            }
                                            model::LLMArchitecture::HfDeepseek => {
                                                DeepseekForCausalLM::from_pretrained(
                                                    &source.try_into()?,
                                                    Some(kind),
                                                    attn_implementation,
                                                    Some(device),
                                                    tensor_parallelism_world,
                                                    Some(llm.max_seq_len as usize),
                                                )
                                                .map(|x| Box::new(x) as Box<dyn CausalLM>)
                                            }
                                            model::LLMArchitecture::HfNanoGPT => {
                                                NanoGPTForCausalLM::from_pretrained_with_matformer_config(
                                                    &source.try_into()?,
                                                    Some(kind),
                                                    attn_implementation,
                                                    Some(device),
                                                    tensor_parallelism_world,
                                                    Some(llm.max_seq_len as usize),
                                                    matformer_tier,
                                                    helper_fraction,
                                                    helper_rotation_interval,
                                                )
                                                .map(|x| Box::new(x) as Box<dyn CausalLM>)
                                            }
                                            model::LLMArchitecture::HfAuto => unreachable!(),
                                        }
                                    }));
                                }
                            }

                            let mut models: Vec<Box<dyn CausalLM>> = Vec::new();
                            for future in futures {
                                let model = future
                                    .await
                                    .map_err(InitRunError::ModelLoadingThreadCrashed)??;
                                models.push(model);
                            }

                            RawLoadedModelType::ParallelNativeModels(models)
                        };

                        // Peers need the base width and the tier of this client, because its
                        // tensors can be narrower than the tensors that they need.
                        let shared_config = serde_json::to_string(&config_json)?;
                        debug!("Config uploaded: {}", shared_config);
                        let serialized_tokenizer = tokenizer.to_string(false).unwrap();
                        tx_config
                            .send((shared_config, serialized_tokenizer))
                            .unwrap();

                        let hidden_size = config_json
                            .get("hidden_size")
                            .and_then(|v| v.as_u64());
                        let intermediate_size = config_json
                            .get("intermediate_size")
                            .and_then(|v| v.as_u64());
                        let active_intermediate_size = intermediate_size.and_then(|h| {
                            let divisor = 1_u64.checked_shl(matformer_tier_for_loading as u32)?;
                            Some(h / divisor)
                        });
                        // Compute base intermediate size (before any tier slicing)
                        let checkpoint_metadata =
                            infer_matformer_checkpoint_metadata(&config_json);
                        let base_intermediate_size = checkpoint_metadata.base_intermediate_size;
                        let num_hidden_layers = config_json
                            .get("num_hidden_layers")
                            .and_then(|v| v.as_u64());
                        let vocab_size = config_json
                            .get("vocab_size")
                            .and_then(|v| v.as_u64());

                        let matformer_holds_full_ffn_weights = !uses_sliced_checkpoint
                            && match (intermediate_size, base_intermediate_size) {
                                (Some(local), Some(base)) => local >= base,
                                _ => true,
                            };

                        let canonical_config = canonicalize_config_for_schema(
                            config_json.clone(),
                            init_config.matformer_tier,
                            uses_sliced_checkpoint,
                        );
                        let schema_hash_canonical = schema_hash_for_config(
                            llm.architecture,
                            &canonical_config,
                            &parameter_names,
                        );
                        let schema_info = ModelSchemaInfo {
                            schema_hash_local: schema_hash_for_config(
                                llm.architecture,
                                &config_json,
                                &parameter_names,
                            ),
                            schema_hash_canonical,
                            parameter_count: u32::try_from(parameter_names.len())
                                .unwrap_or(u32::MAX),
                            matformer_tier: init_config.matformer_tier,
                            uses_sliced_checkpoint,
                        };
                        if tx_schema.send(schema_info).is_err() {
                            warn!("Failed to send schema hash info; channel closed");
                        }

                        match (hidden_size, intermediate_size, active_intermediate_size) {
                            (Some(hidden_size), Some(intermediate_size), Some(active_intermediate_size)) => {
                                info!(
                                    integration_test_log_marker = %IntegrationTestLogMarker::LoadedModel,
                                    checkpoint = %llm.checkpoint,
                                    gpus = init_config.data_parallelism * init_config.tensor_parallelism,
                                    dp = init_config.data_parallelism,
                                    tp = init_config.tensor_parallelism,
                                    matformer_tier = init_config.matformer_tier,
                                    matformer_effective_tier = matformer_tier_for_loading,
                                    matformer_load_strategy = ?init_config.matformer_load_strategy,
                                    matformer_uses_sliced_checkpoint = uses_sliced_checkpoint,
                                    matformer_checkpoint_path = matformer_checkpoint_path.as_deref().unwrap_or("unknown"),
                                    hidden_size = hidden_size,
                                    intermediate_size = intermediate_size,
                                    intermediate_size_active = active_intermediate_size,
                                    num_hidden_layers = num_hidden_layers.unwrap_or_default(),
                                    vocab_size = vocab_size.unwrap_or_default(),
                                    "loaded_model",
                                );

                                // Print MatFormer configuration summary
                                print_matformer_summary(
                                    matformer_checkpoint_path.as_deref().unwrap_or("unknown"),
                                    init_config.matformer_tier,
                                    matformer_tier_for_loading,
                                    uses_sliced_checkpoint,
                                    &init_config.matformer_load_strategy,
                                    init_config.matformer_helper_fraction,
                                    intermediate_size,
                                    active_intermediate_size,
                                );
                            }
                            _ => {
                                info!(
                                    integration_test_log_marker = %IntegrationTestLogMarker::LoadedModel,
                                    checkpoint = %llm.checkpoint,
                                    gpus = init_config.data_parallelism * init_config.tensor_parallelism,
                                    dp = init_config.data_parallelism,
                                    tp = init_config.tensor_parallelism,
                                    matformer_tier = init_config.matformer_tier,
                                    "loaded_model",
                                );
                            }
                        }

                        Ok(RawLoadedModel {
                            models: raw_loaded_model_type,
                            tokenizer,
                            model_task_runner,
                            checkpoint_extra_files,
                            parameter_names,
                            matformer_effective_tier: matformer_tier_for_loading,
                            matformer_base_intermediate_size: base_intermediate_size,
                            matformer_holds_full_ffn_weights,
                            schema_hash_canonical,
                        })
                    }),
                model::Checkpoint::Ephemeral => return Err(InitRunError::ModelIsEphemeral),
            },
        };

        // Extract step logging config before wandb_info is consumed
        let step_logging_enabled = init_config
            .wandb_info
            .as_ref()
            .map(|info| info.step_logging)
            .unwrap_or(false);
        let system_metrics_enabled = init_config
            .wandb_info
            .as_ref()
            .map(|info| info.system_metrics)
            .unwrap_or(false);
        let system_metrics_interval_secs = init_config
            .wandb_info
            .as_ref()
            .map(|info| info.system_metrics_interval_secs)
            .unwrap_or(10);

        let wandb_future: JoinHandle<Result<Option<wandb::Run>, wandb::ApiError>> = tokio::spawn({
            let run_id = String::from(&state.run_id);
            async move {
                match init_config.wandb_info {
                    Some(wandb_info) => {
                        let wandb =
                            wandb::WandB::new(wandb::BackendOptions::new(wandb_info.api_key));
                        let mut run_info = wandb::RunInfo::new(wandb_info.project)
                            .name(wandb_info.run)
                            .config((
                                (
                                    "global_batch_size_start",
                                    state.config.global_batch_size_start,
                                ),
                                ("global_batch_size_end", state.config.global_batch_size_end),
                                (
                                    "global_batch_size_warmup_tokens",
                                    state.config.global_batch_size_warmup_tokens,
                                ),
                                ("total_steps", state.config.total_steps),
                                ("run_id", run_id),
                            ));
                        if let Some(entity) = wandb_info.entity {
                            run_info = run_info.entity(entity);
                        }
                        if let Some(group) = wandb_info.group {
                            run_info = run_info.group(group);
                        }
                        match wandb.new_run(run_info.build()?).await {
                            Ok(run) => Ok(Some(run)),
                            Err(e) => {
                                error!(
                                    "[init_run] Could not connect to wandb. Will continue training without it."
                                );
                                debug!("[init_run] wandb error: {:?}", e);
                                Ok(None)
                            }
                        }
                    }
                    None => {
                        info!(
                            "[init_run] No wandb info provided. Will continue training without it."
                        );
                        Ok(None)
                    }
                }
            }
        });

        let (data, models, wandb_run) = tokio::join!(data_future, model_future, wandb_future);
        let RawLoadedModel {
            models,
            tokenizer,
            checkpoint_extra_files,
            model_task_runner,
            parameter_names,
            matformer_effective_tier,
            matformer_base_intermediate_size,
            matformer_holds_full_ffn_weights,
            schema_hash_canonical,
        } = models.map_err(InitRunError::ModelLoadingThreadCrashed)??;

        // TODO add data fetching for verifying, too..
        let data_provider = data.map_err(InitRunError::DataProviderConnect)?;
        let data_fetcher =
            DataFetcher::<T, A>::new(data_provider, init_config.data_parallelism * 2);

        let trainers: Vec<Trainer> = match models {
            RawLoadedModelType::ParallelNativeModels(models) => {
                let mut tp_models: Vec<Vec<Box<dyn CausalLM>>> = Vec::new();
                for model in models {
                    if tp_models
                        .last()
                        .map(|x| x.len() == init_config.tensor_parallelism)
                        .unwrap_or(true)
                    {
                        tp_models.push(Vec::with_capacity(init_config.tensor_parallelism));
                    }
                    tp_models.last_mut().unwrap().push(model);
                }

                let data_parallel: Option<Vec<(CommunicatorId, Arc<dyn Barrier>)>> =
                    if init_config.data_parallelism > 1 {
                        #[cfg(feature = "parallelism")]
                        {
                            Some(
                                (0..init_config.tensor_parallelism)
                                    .map(|_| {
                                        (
                                            tch::CStore::new().into(),
                                            Arc::new(CancellableBarrier::new(
                                                init_config.tensor_parallelism,
                                            ))
                                                as Arc<dyn Barrier>,
                                        )
                                    })
                                    .collect(),
                            )
                        }

                        #[cfg(not(feature = "parallelism"))]
                        {
                            unimplemented!()
                        }
                    } else {
                        None
                    };

                tp_models
                    .into_iter()
                    .enumerate()
                    .map(|(dp, models)| {
                        let data_parallel = data_parallel.as_ref().map(|data_parallel| {
                            data_parallel
                                .iter()
                                .map(|(id, barrier)| DataParallel {
                                    id: id.clone(),
                                    barrier: barrier.clone(),
                                    rank: dp,
                                    world_size: init_config.data_parallelism,
                                })
                                .collect()
                        });
                        let barrier =
                            Arc::new(CancellableBarrier::new(init_config.tensor_parallelism))
                                as Arc<dyn Barrier>;
                        LocalTrainer::new(
                            ParallelModels {
                                models,
                                barrier,
                                data_parallel,
                            },
                            llm.lr_schedule,
                            llm.optimizer,
                            init_config.micro_batch_size,
                            init_config.optim_stats_every_n_steps,
                            init_config.grad_accum_in_fp32,
                        )
                        .into()
                    })
                    .collect()
            }
            #[cfg(feature = "python")]
            RawLoadedModelType::Python(model) => {
                vec![
                    psyche_modeling::LocalTrainer::new(
                        ParallelModels {
                            models: vec![Box::new(model) as Box<dyn CausalLM>],
                            barrier: Arc::new(psyche_modeling::NopBarrier) as Arc<dyn Barrier>,
                            data_parallel: None,
                        },
                        llm.lr_schedule,
                        llm.optimizer,
                        init_config.micro_batch_size,
                        init_config.optim_stats_every_n_steps,
                        init_config.grad_accum_in_fp32,
                    )
                    .into(),
                ]
            }
            #[cfg(feature = "python")]
            RawLoadedModelType::PythonDistributed(model) => {
                vec![
                    psyche_modeling::PythonDistributedTrainer::new(
                        model,
                        llm.lr_schedule,
                        llm.optimizer,
                        init_config.micro_batch_size,
                        init_config.optim_stats_every_n_steps,
                        init_config.grad_accum_in_fp32,
                    )?
                    .into(),
                ]
            }
        };

        let wandb_run = wandb_run.map_err(InitRunError::WandbThreadCrashed)??;
        // Wrap in Arc for sharing between StatsLogger and TrainingStepMetadata
        let wandb_run = wandb_run.map(Arc::new);

        let stats_logger = StatsLogger::new(
            tokenizer,
            model_task_runner.clone(),
            llm.lr_schedule,
            wandb_run.clone(),
            metrics,
        );

        // Start system metrics logging if enabled
        let _system_metrics_task = if system_metrics_enabled {
            stats_logger.start_system_metrics_logging(system_metrics_interval_secs)
        } else {
            None
        };

        let warmup = WarmupStepMetadata {
            model_task_runner: model_task_runner.clone(),
        };

        let training = TrainingStepMetadata {
            data_fetcher,
            identity: init_config.identity,
            write_gradients_dir: init_config.write_gradients_dir,
            tx_health_check,
            tx_distro_result,
            parameter_names: parameter_names.clone(),
            schema_hash: schema_hash_canonical,
            matformer_tier: init_config.matformer_tier,

            model_task_runner: model_task_runner.clone(),
            log_memory_usage: init_config.log_memory_usage,
            step_logging_enabled,
            wandb_run,
        };

        let witness = WitnessStepMetadata {
            model_task_runner: model_task_runner.clone(),
            identity: init_config.identity,
            tx_witness: tx_witness.clone(),
        };

        let cooldown = CooldownStepMetadata::new(
            tx_checkpoint,
            tx_model,
            init_config.checkpoint_config,
            checkpoint_extra_files,
            MatformerCheckpointInfo {
                effective_tier: matformer_effective_tier,
                base_intermediate_size: matformer_base_intermediate_size,
                holds_full_ffn_weights: matformer_holds_full_ffn_weights,
            },
            model_task_runner,
        );

        Ok(StepStateMachine::new(
            init_config.identity,
            warmup,
            training,
            witness,
            cooldown,
            trainers,
            state,
            tx_request_download,
            tx_witness,
            tx_broadcast_finished,
            stats_logger,
        ))
    }
}
