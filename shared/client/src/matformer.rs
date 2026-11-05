use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    io,
    path::Path,
};

pub const MATFORMER_MANIFEST_NAME: &str = "matformer_manifest.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatformerCheckpointMetadata {
    pub tier: Option<u8>,
    pub base_intermediate_size: Option<u64>,
}

pub fn infer_matformer_checkpoint_metadata(config: &Value) -> MatformerCheckpointMetadata {
    let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_u64());

    if let Some(tier) = config.get("matformer_tier").and_then(|v| v.as_u64()) {
        let base_size = config
            .get("matformer_base_intermediate_size")
            .and_then(|v| v.as_u64())
            .or_else(|| intermediate_size.and_then(|size| size.checked_shl(tier as u32)));
        return MatformerCheckpointMetadata {
            tier: Some(tier as u8),
            base_intermediate_size: base_size,
        };
    }

    if let (Some(base), Some(current)) = (
        config.get("matformer_base_intermediate_size").and_then(|v| v.as_u64()),
        intermediate_size,
    ) {
        if let Some(tier) = compute_tier_from_sizes(base, current) {
            return MatformerCheckpointMetadata {
                tier: Some(tier),
                base_intermediate_size: Some(base),
            };
        }
        return MatformerCheckpointMetadata {
            tier: None,
            base_intermediate_size: Some(base),
        };
    }

    MatformerCheckpointMetadata {
        tier: None,
        base_intermediate_size: intermediate_size,
    }
}

pub fn annotate_matformer_checkpoint_config(
    config: &mut Value,
    base_intermediate_size_override: Option<u64>,
) -> MatformerCheckpointMetadata {
    let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_u64());
    let base_size = base_intermediate_size_override
        .or_else(|| {
            config
                .get("matformer_base_intermediate_size")
                .and_then(|v| v.as_u64())
        })
        .or(intermediate_size);

    let tier = match (base_size, intermediate_size) {
        (Some(base), Some(current)) => compute_tier_from_sizes(base, current).unwrap_or(0),
        _ => 0,
    };

    if let Some(obj) = config.as_object_mut() {
        obj.insert("matformer_tier".to_string(), Value::from(tier));
        if let Some(base_size) = base_size {
            obj.insert(
                "matformer_base_intermediate_size".to_string(),
                Value::from(base_size),
            );
        }
    }

    MatformerCheckpointMetadata {
        tier: Some(tier),
        base_intermediate_size: base_size,
    }
}

pub fn ensure_matformer_checkpoint_metadata(
    config: &mut Value,
    manifest_base_intermediate_size: Option<u64>,
    sliced_checkpoint_tier: Option<u8>,
) -> MatformerCheckpointMetadata {
    let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_u64());
    let existing_base = config
        .get("matformer_base_intermediate_size")
        .and_then(|v| v.as_u64());
    let existing_tier = config
        .get("matformer_tier")
        .and_then(|v| v.as_u64())
        .and_then(|tier| u8::try_from(tier).ok());

    let mut base_intermediate_size = existing_base.or(manifest_base_intermediate_size);
    if base_intermediate_size.is_none() {
        let tier_hint = existing_tier.or(sliced_checkpoint_tier);
        if let (Some(tier_hint), Some(current)) = (tier_hint, intermediate_size) {
            base_intermediate_size = current.checked_shl(tier_hint as u32);
        }
    }

    let mut tier = existing_tier;
    if tier.is_none() {
        if let (Some(base), Some(current)) = (base_intermediate_size, intermediate_size) {
            tier = compute_tier_from_sizes(base, current);
        } else {
            tier = sliced_checkpoint_tier;
        }
    }

    if let Some(obj) = config.as_object_mut() {
        if existing_base.is_none() {
            if let Some(base) = base_intermediate_size {
                obj.insert(
                    "matformer_base_intermediate_size".to_string(),
                    Value::from(base),
                );
            }
        }
        if existing_tier.is_none() {
            if let Some(tier) = tier {
                obj.insert("matformer_tier".to_string(), Value::from(tier));
            }
        }
    }

    MatformerCheckpointMetadata {
        tier,
        base_intermediate_size: base_intermediate_size.or(intermediate_size),
    }
}

fn compute_tier_from_sizes(base: u64, current: u64) -> Option<u8> {
    if base == 0 || current == 0 || base < current {
        return None;
    }
    if base % current != 0 {
        return None;
    }
    let ratio = base / current;
    if ratio.is_power_of_two() {
        Some(ratio.trailing_zeros() as u8)
    } else {
        None
    }
}

/// How a client that joins over P2P builds its local model.
#[derive(Debug, Clone, PartialEq)]
pub struct P2pLoadPlan {
    /// Config for the local model.
    pub config: Value,
    /// FFN width that the downloaded tensors must have.
    pub required_ffn_width: Option<i64>,
    pub uses_sliced_checkpoint: bool,
    pub matformer_tier_for_loading: u8,
    pub base_intermediate_size: Option<u64>,
    pub sliced_checkpoint_tier: Option<u8>,
}

impl P2pLoadPlan {
    /// Plan for an architecture without MatFormer: use the peer config as it is.
    pub fn unchanged(config: Value, cli_tier: u8) -> Self {
        Self {
            config,
            required_ffn_width: None,
            uses_sliced_checkpoint: false,
            matformer_tier_for_loading: cli_tier,
            base_intermediate_size: None,
            sliced_checkpoint_tier: None,
        }
    }
}

/// Selects the local model shape for a client that joins over P2P.
///
/// The first peer that answers can be on any tier, so its width says nothing about
/// the width that this client needs. The base width and the CLI tier decide:
/// a universal client holds full weights, and all other clients hold only their prefix.
///
/// Returns `None` if the tier leaves no FFN neurons.
pub fn plan_p2p_load(peer_config: Value, cli_tier: u8, universal: bool) -> Option<P2pLoadPlan> {
    let Some(base) = infer_matformer_checkpoint_metadata(&peer_config).base_intermediate_size
    else {
        return Some(P2pLoadPlan::unchanged(peer_config, cli_tier));
    };

    let width = base
        .checked_shr(cli_tier as u32)
        .filter(|width| *width > 0)?;
    let holds_full_weights = cli_tier == 0 || universal;
    let local_width = if holds_full_weights { base } else { width };

    let mut config = peer_config;
    if let Some(obj) = config.as_object_mut() {
        obj.insert("intermediate_size".to_string(), Value::from(local_width));
        obj.insert(
            "matformer_base_intermediate_size".to_string(),
            Value::from(base),
        );
        obj.insert(
            "matformer_tier".to_string(),
            Value::from(if holds_full_weights { 0 } else { cli_tier }),
        );
    }

    Some(P2pLoadPlan {
        config,
        required_ffn_width: i64::try_from(local_width).ok(),
        uses_sliced_checkpoint: !holds_full_weights,
        matformer_tier_for_loading: if holds_full_weights { cli_tier } else { 0 },
        base_intermediate_size: Some(base),
        sliced_checkpoint_tier: (!holds_full_weights).then_some(cli_tier),
    })
}

/// Returns the lowercase hex SHA-256 of a file, without reading it all into memory.
pub fn file_sha256_hex(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    io::copy(&mut file, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatformerManifest {
    pub schema_version: u32,
    #[serde(default)]
    pub matformer_base_intermediate_size: Option<u64>,
    #[serde(default)]
    pub common_files: Vec<String>,
    #[serde(default)]
    pub tiers: Vec<MatformerManifestTier>,
    #[serde(default)]
    pub sha256: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatformerManifestTier {
    pub tier: u8,
    #[serde(default)]
    pub intermediate_size: Option<u64>,
    #[serde(default)]
    pub files: Vec<String>,
}

impl MatformerManifest {
    pub fn available_tiers(&self) -> Vec<u8> {
        let mut tiers = self.tiers.iter().map(|entry| entry.tier).collect::<Vec<_>>();
        tiers.sort_unstable();
        tiers.dedup();
        tiers
    }

    pub fn tier_entry(&self, tier: u8) -> Option<&MatformerManifestTier> {
        self.tiers.iter().find(|entry| entry.tier == tier)
    }

    pub fn files_for_tier(&self, tier: u8) -> Option<Vec<String>> {
        let entry = self.tier_entry(tier)?;
        let mut seen = HashSet::new();
        let mut files = Vec::new();
        for name in self
            .common_files
            .iter()
            .chain(entry.files.iter())
        {
            if seen.insert(name) {
                files.push(name.clone());
            }
        }
        Some(files)
    }
}

#[cfg(test)]
mod tests {
    use super::plan_p2p_load;

    fn full_peer() -> serde_json::Value {
        serde_json::json!({"intermediate_size": 1024})
    }

    fn sliced_peer() -> serde_json::Value {
        serde_json::json!({
            "intermediate_size": 512,
            "matformer_tier": 1,
            "matformer_base_intermediate_size": 1024,
        })
    }

    #[test]
    fn p2p_plan_does_not_depend_on_the_tier_of_the_peer() {
        for peer in [full_peer(), sliced_peer()] {
            let full = plan_p2p_load(peer.clone(), 0, false).unwrap();
            assert_eq!(full.config["intermediate_size"], 1024);
            assert_eq!(full.required_ffn_width, Some(1024));
            assert!(!full.uses_sliced_checkpoint);
            assert_eq!(full.matformer_tier_for_loading, 0);

            let narrow = plan_p2p_load(peer.clone(), 1, false).unwrap();
            assert_eq!(narrow.config["intermediate_size"], 512);
            assert_eq!(narrow.config["matformer_tier"], 1);
            assert_eq!(narrow.required_ffn_width, Some(512));
            assert!(narrow.uses_sliced_checkpoint);
            assert_eq!(narrow.matformer_tier_for_loading, 0);
            assert_eq!(narrow.sliced_checkpoint_tier, Some(1));

            let narrower = plan_p2p_load(peer.clone(), 2, false).unwrap();
            assert_eq!(narrower.required_ffn_width, Some(256));

            // A universal client keeps full weights and narrows at forward time.
            let universal = plan_p2p_load(peer, 1, true).unwrap();
            assert_eq!(universal.config["intermediate_size"], 1024);
            assert_eq!(universal.required_ffn_width, Some(1024));
            assert!(!universal.uses_sliced_checkpoint);
            assert_eq!(universal.matformer_tier_for_loading, 1);
        }
    }

    #[test]
    fn p2p_plan_rejects_a_tier_without_neurons() {
        assert!(plan_p2p_load(serde_json::json!({"intermediate_size": 2}), 2, false).is_none());
    }

    #[test]
    fn p2p_plan_passes_through_a_config_without_ffn_width() {
        let config = serde_json::json!({"model_type": "custom"});
        let plan = plan_p2p_load(config.clone(), 1, false).unwrap();
        assert_eq!(plan.config, config);
        assert_eq!(plan.required_ffn_width, None);
        assert_eq!(plan.matformer_tier_for_loading, 1);
    }

    use super::*;

    #[test]
    fn manifest_files_for_tier_combines_common_and_tier_files() {
        let manifest = MatformerManifest {
            schema_version: 1,
            matformer_base_intermediate_size: Some(1024),
            common_files: vec!["tokenizer.json".to_string(), "shared.py".to_string()],
            tiers: vec![MatformerManifestTier {
                tier: 1,
                intermediate_size: Some(512),
                files: vec![
                    "../model-tier1/config.json".to_string(),
                    "../model-tier1/model.safetensors".to_string(),
                ],
            }],
            sha256: HashMap::new(),
        };

        let files = manifest.files_for_tier(1).unwrap();
        assert_eq!(
            files,
            vec![
                "tokenizer.json".to_string(),
                "shared.py".to_string(),
                "../model-tier1/config.json".to_string(),
                "../model-tier1/model.safetensors".to_string(),
            ]
        );
    }
}
