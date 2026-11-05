//! NanoGPT Model Implementation
//!
//! Implementation of the NanoGPT architecture from modded-nanogpt, featuring:
//! - Fused QKVO projections
//! - ReLU² activation in MLP
//! - QK-Norm (RMSNorm on Q,K after RoPE)
//! - SA Lambdas (learnable scalars on Q,K,V,O)
//! - x0 residual (embed added to every layer)
//! - Value embeddings (extra embedding tables mixed into V)
//! - Skip connections (layer 3→6, smear gate)
//! - Logit softcapping
//! - Half-truncate RoPE
//! - Attention gate
//!
//! All features are toggleable via config flags for incremental testing.

use crate::{
    auto_config::{AttentionImplementation, ModelConfig, ModelLoadError, PretrainedSource},
    causal_language_model::{
        CausalLanguageModel, EosToks, LanguageModelConfig, LanguageModelForward,
    },
    matformer_helper::HelperConfig,
    parallelism::{ColumnParallelLinear, Communicator, CommunicatorId, RowParallelLinear},
    rms_norm::RMSNorm,
    rope::{RoPECache, RoPEConfig},
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tch::{
    nn::{self, Module},
    Device, Kind, Tensor,
};

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_true() -> bool {
    true
}

/// Configuration for NanoGPT model
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct NanoGPTConfig {
    // Core dimensions
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,

    // Normalization
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    // RoPE
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub rope_scaling: Option<RoPEConfig>,

    // Tokens
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<EosToks>,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // ========== NanoGPT-specific features ==========

    // Core architecture
    #[serde(default)]
    pub use_fused_qkvo: bool,
    #[serde(default)]
    pub use_relu_squared_mlp: bool,
    #[serde(default)]
    pub mlp_bias: bool,

    // QK-Norm
    #[serde(default)]
    pub use_qk_norm: bool,
    /// If true (default), apply QK-norm before RoPE (modded-nanogpt style).
    /// If false, apply after RoPE.
    #[serde(default = "default_true")]
    pub qk_norm_before_rope: bool,

    // SA Lambdas (learnable scalars on Q, K, V, O)
    #[serde(default)]
    pub use_sa_lambdas: bool,

    // Residual enhancement
    #[serde(default)]
    pub use_x0_residual: bool,
    #[serde(default)]
    pub use_x0_lambdas: bool,
    #[serde(default)]
    pub use_resid_lambdas: bool,

    // Value embeddings
    #[serde(default)]
    pub num_value_embeddings: usize,

    // Skip connections
    #[serde(default)]
    pub use_block_skip: bool,
    pub block_skip_from: Option<usize>,
    pub block_skip_to: Option<usize>,
    #[serde(default)]
    pub use_smear_gate: bool,

    // Output modifications
    #[serde(default)]
    pub use_logit_softcap: bool,
    pub softcap_scale: Option<f64>,
    #[serde(default)]
    pub use_backout: bool,
    pub backout_layer: Option<usize>,

    // RoPE modifications
    #[serde(default)]
    pub use_half_truncate_rope: bool,
    #[serde(default)]
    pub use_key_offset: bool,
    pub key_offset: Option<i64>,
    #[serde(default)]
    pub use_learnable_attn_scale: bool,
    /// Fixed attention scale (overrides 1/sqrt(d)). modded-nanogpt uses 0.1-0.12.
    pub attention_scale: Option<f64>,

    // Attention gate
    #[serde(default)]
    pub use_attention_gate: bool,

    // MatFormer: heterogeneous FFN width training
    // tier 0 = full width, tier N = width / 2^N
    #[serde(default)]
    pub matformer_tier: u8,
    /// Fraction of suffix neurons to help train (0.0-1.0) for stochastic suffix sampling.
    #[serde(default)]
    pub matformer_helper_fraction: f32,
    /// Rounds to keep helper indices fixed before rotating.
    #[serde(default = "default_helper_rotation_interval")]
    pub matformer_helper_rotation_interval: u64,
}

fn default_helper_rotation_interval() -> u64 {
    16
}

impl NanoGPTConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn dummy() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 1000,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            bos_token_id: Some(1),
            eos_token_id: Some(EosToks::Single(2)),
            tie_word_embeddings: false,
            // All NanoGPT features disabled by default
            use_fused_qkvo: false,
            use_relu_squared_mlp: false,
            mlp_bias: false,
            use_qk_norm: false,
            qk_norm_before_rope: true,
            use_sa_lambdas: false,
            use_x0_residual: false,
            use_x0_lambdas: false,
            use_resid_lambdas: false,
            num_value_embeddings: 0,
            use_block_skip: false,
            block_skip_from: None,
            block_skip_to: None,
            use_smear_gate: false,
            use_logit_softcap: false,
            softcap_scale: None,
            use_backout: false,
            backout_layer: None,
            use_half_truncate_rope: false,
            use_key_offset: false,
            key_offset: None,
            use_learnable_attn_scale: false,
            attention_scale: None,
            use_attention_gate: false,
            matformer_tier: 0,
            matformer_helper_fraction: 0.0,
            matformer_helper_rotation_interval: 16,
        }
    }

    /// Get helper config if helper mode is enabled.
    pub fn helper_config(&self) -> Option<crate::matformer_helper::HelperConfig> {
        if self.matformer_helper_fraction > 0.0 && self.matformer_tier > 0 {
            Some(crate::matformer_helper::HelperConfig::new(
                self.matformer_helper_fraction,
                self.matformer_helper_rotation_interval,
            ))
        } else {
            None
        }
    }
}

impl ModelConfig for NanoGPTConfig {
    fn get_parameter_names(&self) -> Vec<String> {
        let mut variables = Vec::new();

        // Embeddings
        variables.push("model.embed_tokens.weight".to_string());

        // Value embeddings if enabled
        for i in 0..self.num_value_embeddings {
            variables.push(format!("model.value_embed_{}.weight", i));
        }

        // Layers
        for layer_idx in 0..self.num_hidden_layers {
            let prefix = format!("model.layers.{}", layer_idx);

            // Attention projections
            if self.use_fused_qkvo {
                variables.push(format!("{}.self_attn.qkvo_proj.weight", prefix));
            } else {
                variables.push(format!("{}.self_attn.q_proj.weight", prefix));
                variables.push(format!("{}.self_attn.k_proj.weight", prefix));
                variables.push(format!("{}.self_attn.v_proj.weight", prefix));
            }
            variables.push(format!("{}.self_attn.o_proj.weight", prefix));

            // QK-Norm
            if self.use_qk_norm {
                variables.push(format!("{}.self_attn.q_norm.weight", prefix));
                variables.push(format!("{}.self_attn.k_norm.weight", prefix));
            }

            // SA Lambdas
            if self.use_sa_lambdas {
                variables.push(format!("{}.self_attn.lambda_q", prefix));
                variables.push(format!("{}.self_attn.lambda_k", prefix));
                variables.push(format!("{}.self_attn.lambda_v", prefix));
                variables.push(format!("{}.self_attn.lambda_o", prefix));
            }

            // Learnable attention scale
            if self.use_learnable_attn_scale {
                variables.push(format!("{}.self_attn.attn_scale", prefix));
            }

            // Attention gate
            if self.use_attention_gate {
                variables.push(format!("{}.self_attn.attn_gate", prefix));
            }

            // MLP
            variables.push(format!("{}.mlp.gate_proj.weight", prefix));
            variables.push(format!("{}.mlp.up_proj.weight", prefix));
            variables.push(format!("{}.mlp.down_proj.weight", prefix));
            if self.mlp_bias {
                variables.push(format!("{}.mlp.gate_proj.bias", prefix));
                variables.push(format!("{}.mlp.up_proj.bias", prefix));
                variables.push(format!("{}.mlp.down_proj.bias", prefix));
            }

            // Norms
            variables.push(format!("{}.input_layernorm.weight", prefix));
            variables.push(format!("{}.post_attention_layernorm.weight", prefix));

            // Residual lambdas
            if self.use_x0_lambdas {
                variables.push(format!("{}.x0_lambda", prefix));
            }
            if self.use_resid_lambdas {
                variables.push(format!("{}.resid_lambda", prefix));
            }

            // Smear gate
            if self.use_smear_gate {
                variables.push(format!("{}.smear_gate", prefix));
            }
        }

        // Final norm and LM head
        variables.push("model.norm.weight".to_string());
        variables.push("lm_head.weight".to_string());

        variables
    }
}

impl LanguageModelConfig for NanoGPTConfig {
    fn matformer_active_ffn_width(&self) -> Option<i64> {
        if self.matformer_tier == 0 {
            return None;
        }
        let divisor = 1_i64.checked_shl(self.matformer_tier as u32)?;
        Some(self.intermediate_size as i64 / divisor)
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    fn set_max_position_embeddings(&mut self, set: usize) {
        self.max_position_embeddings = set;
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn rope_config(&self) -> Option<RoPEConfig> {
        self.rope_scaling.clone()
    }

    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn bos_token_id(&self) -> Option<i64> {
        self.bos_token_id
    }

    fn eos_token_ids(&self) -> Option<EosToks> {
        self.eos_token_id.clone()
    }

    fn logit_softcap_scale(&self) -> Option<f64> {
        if self.use_logit_softcap {
            // Default scale is 30.0 (modded-nanogpt default)
            Some(self.softcap_scale.unwrap_or(30.0))
        } else {
            None
        }
    }
}

// ============================================================================
// MLP
// ============================================================================

#[derive(Debug)]
struct NanoGPTMlp {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
    use_relu_squared: bool,
    /// Prefix size for this tier (None = tier 0, full width)
    matformer_hidden_size: Option<i64>,
    /// Full intermediate size for helper index generation
    full_intermediate_size: i64,
    /// Layer index for deterministic helper sampling
    layer_idx: usize,
    /// Helper config (None = no helper mode)
    helper_config: Option<HelperConfig>,
    /// Shared reference to current round number
    current_round: Arc<AtomicU64>,
    is_tensor_parallel: bool,
}

impl NanoGPTMlp {
    fn new(
        vs: nn::Path,
        hidden_size: i64,
        intermediate_size: i64,
        bias: bool,
        comm: Option<Arc<Communicator>>,
        use_relu_squared: bool,
        matformer_hidden_size: Option<i64>,
        layer_idx: usize,
        helper_config: Option<HelperConfig>,
        current_round: Arc<AtomicU64>,
    ) -> Self {
        let is_tensor_parallel = comm.as_ref().map(|x| x.size()).unwrap_or(1) > 1;
        let tp_size = comm.as_ref().map(|x| x.size()).unwrap_or(1);

        // Validate matformer settings
        if let Some(mf_size) = matformer_hidden_size {
            assert!(
                !bias,
                "mlp_bias is not supported with MatFormer tiers (matformer_tier > 0)"
            );
            assert!(mf_size > 0, "matformer_hidden_size must be > 0");
            assert!(
                mf_size <= intermediate_size,
                "matformer_hidden_size must be <= intermediate_size"
            );
            assert_eq!(
                mf_size % tp_size,
                0,
                "matformer_hidden_size must be divisible by tp_size"
            );
        }
        if let Some(ref config) = helper_config {
            assert!(
                config.helper_fraction <= 0.0,
                "MatFormer helper mode is temporarily disabled (suffix sampling not wired)"
            );
        }

        // Log helper mode configuration
        if let Some(ref config) = helper_config {
            if config.helper_fraction > 0.0 {
                eprintln!(
                    "[NanoGPT MLP layer {}] Helper mode enabled: fraction={}, rotation_interval={}, matformer_hidden={}",
                    layer_idx, config.helper_fraction, config.rotation_interval, matformer_hidden_size.unwrap_or(intermediate_size)
                );
            }
        }

        Self {
            gate_proj: ColumnParallelLinear::new(
                &vs / "gate_proj",
                hidden_size,
                intermediate_size,
                bias,
                false,
                comm.clone(),
            ),
            up_proj: ColumnParallelLinear::new(
                &vs / "up_proj",
                hidden_size,
                intermediate_size,
                bias,
                false,
                comm.clone(),
            ),
            down_proj: RowParallelLinear::new(
                &vs / "down_proj",
                intermediate_size,
                hidden_size,
                bias,
                true,
                comm,
            ),
            use_relu_squared,
            matformer_hidden_size,
            full_intermediate_size: intermediate_size,
            layer_idx,
            helper_config,
            current_round,
            is_tensor_parallel,
        }
    }
}

impl Module for NanoGPTMlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // Tier 0 (full width): use standard forward
        let Some(matformer_hidden_size) = self.matformer_hidden_size else {
            // No MatFormer: use standard forward
            return if self.use_relu_squared {
                // ReLU² activation: relu(x)²
                let gate = self.gate_proj.forward(xs).relu().square();
                let up = self.up_proj.forward(xs);
                self.down_proj.forward(&(gate * up))
            } else {
                // Standard SiLU (SwiGLU) activation
                self.down_proj
                    .forward(&(self.gate_proj.forward(xs).silu() * self.up_proj.forward(xs)))
            };
        };

        // MatFormer: use narrowed weights for sliced FFN
        assert!(
            !self.is_tensor_parallel,
            "matformer_tier is not yet supported with tensor parallelism"
        );

        // Check if we have helper mode enabled
        let use_helper =
            self.helper_config.is_some() && matformer_hidden_size < self.full_intermediate_size;

        if use_helper {
            // Helper mode: use index_select with prefix + stochastic suffix indices
            let helper_config = self.helper_config.as_ref().unwrap();
            let round = self.current_round.load(Ordering::Relaxed);

            let indices = crate::matformer_helper::get_matformer_indices(
                matformer_hidden_size as usize,
                self.full_intermediate_size as usize,
                helper_config,
                round,
                self.layer_idx,
            );

            let indices_tensor = Tensor::from_slice(&indices).to_device(xs.device());

            // Weight shapes:
            // - gate_proj/up_proj: [intermediate_size, hidden_size] - select rows
            // - down_proj: [hidden_size, intermediate_size] - select columns
            let gate_w = self.gate_proj.linear.ws.index_select(0, &indices_tensor);
            let up_w = self.up_proj.linear.ws.index_select(0, &indices_tensor);
            let down_w = self.down_proj.linear.ws.index_select(1, &indices_tensor);

            // Manual matmul with indexed weights
            let gate = xs.matmul(&gate_w.transpose(0, 1));
            let gate = if let Some(ref bias) = self.gate_proj.linear.bs {
                gate + bias.index_select(0, &indices_tensor)
            } else {
                gate
            };
            let up = xs.matmul(&up_w.transpose(0, 1));
            let up = if let Some(ref bias) = self.up_proj.linear.bs {
                up + bias.index_select(0, &indices_tensor)
            } else {
                up
            };

            let hidden = if self.use_relu_squared {
                gate.relu().square() * up
            } else {
                gate.silu() * up
            };

            let out = hidden.matmul(&down_w.transpose(0, 1));
            if let Some(ref bias) = self.down_proj.linear.bs {
                out + bias
            } else {
                out
            }
        } else {
            // Standard MatFormer: use narrow for contiguous prefix (more efficient)
            // Weight shapes:
            // - gate_proj/up_proj: [intermediate_size, hidden_size]
            // - down_proj: [hidden_size, intermediate_size]
            // We narrow along the intermediate dimension:
            // - gate/up: narrow rows (dim 0) to get [matformer_hidden_size, hidden_size]
            // - down: narrow cols (dim 1) to get [hidden_size, matformer_hidden_size]
            let gate_w = self.gate_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
            let up_w = self.up_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
            let down_w = self.down_proj.linear.ws.narrow(1, 0, matformer_hidden_size);

            // Manual matmul with narrowed weights
            let gate = xs.matmul(&gate_w.transpose(0, 1));
            let gate = if let Some(ref bias) = self.gate_proj.linear.bs {
                gate + bias.narrow(0, 0, matformer_hidden_size)
            } else {
                gate
            };
            let up = xs.matmul(&up_w.transpose(0, 1));
            let up = if let Some(ref bias) = self.up_proj.linear.bs {
                up + bias.narrow(0, 0, matformer_hidden_size)
            } else {
                up
            };

            let hidden = if self.use_relu_squared {
                gate.relu().square() * up
            } else {
                gate.silu() * up
            };

            let out = hidden.matmul(&down_w.transpose(0, 1));
            if let Some(ref bias) = self.down_proj.linear.bs {
                out + bias
            } else {
                out
            }
        }
    }
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Debug)]
struct NanoGPTAttention {
    // Fused QKVO projection (when use_fused_qkvo=true)
    qkvo_proj: Option<ColumnParallelLinear>,
    // Separate projections (when use_fused_qkvo=false)
    q_proj: Option<ColumnParallelLinear>,
    k_proj: Option<ColumnParallelLinear>,
    v_proj: Option<ColumnParallelLinear>,
    // Output projection (always present)
    o_proj: RowParallelLinear,

    // QK-Norm: RMSNorm applied to Q and K after RoPE
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,

    // SA Lambdas: Learnable scalars on Q, K, V, O
    lambda_q: Option<Tensor>,
    lambda_k: Option<Tensor>,
    lambda_v: Option<Tensor>,
    lambda_o: Option<Tensor>,

    // Learnable attention scale (replaces 1/sqrt(d))
    attn_scale: Option<Tensor>,

    // Attention gate (sigmoid gate on attention output)
    attn_gate: Option<Tensor>,

    // Dimensions
    n_head: i64,
    n_kvhead: i64,
    head_dim: i64,
    // Sizes for splitting fused projection
    size_q: i64,
    size_kv: i64,

    // Config
    use_fused_qkvo: bool,
    qk_norm_before_rope: bool,
    fixed_attn_scale: Option<f64>,
    attn_implementation: AttentionImplementation,
    tp_size: i64,
    device: Device,
}

impl NanoGPTAttention {
    fn new(
        vs: nn::Path,
        config: &NanoGPTConfig,
        attn_implementation: AttentionImplementation,
        comm: Option<Arc<Communicator>>,
    ) -> Self {
        let n_embd = config.hidden_size as i64;
        let n_head = config.num_attention_heads as i64;
        let n_kvhead = config.num_key_value_heads() as i64;
        let head_dim = config.head_dim() as i64;

        let tp_size = comm.as_ref().map_or(1, |c| c.size());

        // Validate divisibility
        assert!(
            n_head % tp_size == 0,
            "num_attention_heads must be divisible by tensor parallelism size"
        );
        assert!(
            n_kvhead % tp_size == 0,
            "num_key_value_heads must be divisible by tensor parallelism size"
        );

        let size_q = head_dim * n_head;
        let size_kv = head_dim * n_kvhead;

        // Create either fused QKVO or separate Q/K/V projections
        let (qkvo_proj, q_proj, k_proj, v_proj) = if config.use_fused_qkvo {
            // Fused projection: outputs [Q, K, V] concatenated
            let qkvo_size = size_q + size_kv + size_kv; // Q + K + V
            (
                Some(ColumnParallelLinear::new(
                    &vs / "qkvo_proj",
                    n_embd,
                    qkvo_size,
                    false, // no bias
                    false,
                    comm.clone(),
                )),
                None,
                None,
                None,
            )
        } else {
            // Separate projections
            (
                None,
                Some(ColumnParallelLinear::new(
                    &vs / "q_proj",
                    n_embd,
                    size_q,
                    false,
                    false,
                    comm.clone(),
                )),
                Some(ColumnParallelLinear::new(
                    &vs / "k_proj",
                    n_embd,
                    size_kv,
                    false,
                    false,
                    comm.clone(),
                )),
                Some(ColumnParallelLinear::new(
                    &vs / "v_proj",
                    n_embd,
                    size_kv,
                    false,
                    false,
                    comm.clone(),
                )),
            )
        };

        // QK-Norm: RMSNorm on Q and K after RoPE
        // Note: QK-Norm operates on head_dim dimension, applied per-head
        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(RMSNorm::new(&vs / "q_norm", head_dim, config.rms_norm_eps)),
                Some(RMSNorm::new(&vs / "k_norm", head_dim, config.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        // SA Lambdas: Learnable scalars initialized to 1.0
        let (lambda_q, lambda_k, lambda_v, lambda_o) = if config.use_sa_lambdas {
            (
                Some(vs.var("lambda_q", &[1], nn::Init::Const(1.0))),
                Some(vs.var("lambda_k", &[1], nn::Init::Const(1.0))),
                Some(vs.var("lambda_v", &[1], nn::Init::Const(1.0))),
                Some(vs.var("lambda_o", &[1], nn::Init::Const(1.0))),
            )
        } else {
            (None, None, None, None)
        };

        // Learnable attention scale (initialized to 1/sqrt(head_dim))
        let attn_scale = if config.use_learnable_attn_scale {
            let default_scale = 1.0 / (head_dim as f64).sqrt();
            Some(vs.var("attn_scale", &[1], nn::Init::Const(default_scale)))
        } else {
            None
        };

        // Attention gate (sigmoid gate on attention output, initialized to 0 = sigmoid(0) = 0.5)
        let attn_gate = if config.use_attention_gate {
            Some(vs.var("attn_gate", &[1], nn::Init::Const(0.0)))
        } else {
            None
        };

        Self {
            qkvo_proj,
            q_proj,
            k_proj,
            v_proj,
            o_proj: RowParallelLinear::new(&vs / "o_proj", size_q, n_embd, false, true, comm),
            q_norm,
            k_norm,
            lambda_q,
            lambda_k,
            lambda_v,
            lambda_o,
            attn_scale,
            attn_gate,
            n_head,
            n_kvhead,
            head_dim,
            size_q,
            size_kv,
            use_fused_qkvo: config.use_fused_qkvo,
            qk_norm_before_rope: config.qk_norm_before_rope,
            fixed_attn_scale: config.attention_scale,
            attn_implementation,
            tp_size,
            device: vs.device(),
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        value_embed_sum: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        _sequence_lengths: Option<&Vec<Vec<i32>>>,
        rope_cache: &RoPECache,
    ) -> Tensor {
        let (b, t, _c) = x.size3().unwrap();
        let kind = x.kind();

        let local_n_head = self.n_head / self.tp_size;
        let local_n_kvhead = self.n_kvhead / self.tp_size;

        // Local sizes (after tensor parallelism split)
        let local_size_q = self.size_q / self.tp_size;
        let local_size_kv = self.size_kv / self.tp_size;

        // Q, K, V projections (fused or separate)
        let (q, k, v) = if self.use_fused_qkvo {
            // Fused path: single projection, then split
            let qkvo = self.qkvo_proj.as_ref().unwrap().forward(x);
            let splits = qkvo.split_with_sizes(&[local_size_q, local_size_kv, local_size_kv], -1);
            (
                splits[0].shallow_clone(),
                splits[1].shallow_clone(),
                splits[2].shallow_clone(),
            )
        } else {
            // Separate path
            (
                self.q_proj.as_ref().unwrap().forward(x),
                self.k_proj.as_ref().unwrap().forward(x),
                self.v_proj.as_ref().unwrap().forward(x),
            )
        };

        // Add value embeddings to V before reshape
        // value_embed_sum has shape [b, t, hidden_size]
        // V has shape [b, t, local_size_kv] which equals hidden_size when n_kv_heads == n_heads
        let v = if let Some(value_embed_sum) = value_embed_sum {
            // For simplicity, we add directly. This works when local_size_kv == hidden_size
            // (i.e., no GQA or after accounting for TP which maintains the ratio)
            v + value_embed_sum
        } else {
            v
        };

        // Reshape to [b, t, n_heads, head_dim] then transpose to [b, n_heads, t, head_dim]
        let q = q
            .reshape([b, t, local_n_head, self.head_dim])
            .transpose(1, 2);
        let k = k
            .reshape([b, t, local_n_kvhead, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape([b, t, local_n_kvhead, self.head_dim])
            .transpose(1, 2);

        // Apply QK-Norm BEFORE RoPE if configured (modded-nanogpt style)
        // Shape is [b, n_heads, t, head_dim], norm operates on last dim
        let (q, k) = if self.qk_norm_before_rope {
            let q = if let Some(ref q_norm) = self.q_norm {
                q_norm.forward(&q)
            } else {
                q
            };
            let k = if let Some(ref k_norm) = self.k_norm {
                k_norm.forward(&k)
            } else {
                k
            };
            (q, k)
        } else {
            (q, k)
        };

        // Apply RoPE
        let q = rope_cache.apply_rotary_emb(&q, position_ids).to_kind(kind);
        let k = rope_cache.apply_rotary_emb(&k, position_ids).to_kind(kind);

        // Apply QK-Norm AFTER RoPE if configured (legacy style)
        let (q, k) = if !self.qk_norm_before_rope {
            let q = if let Some(ref q_norm) = self.q_norm {
                q_norm.forward(&q)
            } else {
                q
            };
            let k = if let Some(ref k_norm) = self.k_norm {
                k_norm.forward(&k)
            } else {
                k
            };
            (q, k)
        } else {
            (q, k)
        };

        // Apply SA Lambdas: learnable scalars
        let q = if let Some(ref lambda_q) = self.lambda_q {
            &q * lambda_q
        } else {
            q
        };
        let k = if let Some(ref lambda_k) = self.lambda_k {
            &k * lambda_k
        } else {
            k
        };
        let v = if let Some(ref lambda_v) = self.lambda_v {
            &v * lambda_v
        } else {
            v
        };

        // Repeat KV heads if using grouped-query attention
        let n_rep = local_n_head / local_n_kvhead;
        let k = if n_rep > 1 { repeat_kv(&k, n_rep) } else { k };
        let v = if n_rep > 1 { repeat_kv(&v, n_rep) } else { v };

        // Attention computation
        // Priority: fixed_attn_scale > learnable attn_scale > 1/sqrt(d)
        let default_scale = 1.0 / (self.head_dim as f64).sqrt();
        let scale = self.fixed_attn_scale.unwrap_or_else(|| {
            self.attn_scale
                .as_ref()
                .map_or(default_scale, |s| s.double_value(&[]))
        });

        let y = match self.attn_implementation {
            #[cfg(feature = "parallelism")]
            AttentionImplementation::FlashAttention2 => {
                tch::flash_attention_forward(
                    &q.to_kind(Kind::BFloat16),
                    &k.to_kind(Kind::BFloat16),
                    &v.to_kind(Kind::BFloat16),
                    None,
                    None,
                    t,
                    t,
                    0.0,   // dropout
                    scale, // softmax_scale
                    true,  // is_causal
                    false, // return_softmax
                    None,  // window_size_left
                    None,  // window_size_right
                )
                .0
                .to_kind(kind)
            }
            AttentionImplementation::Sdpa => Tensor::scaled_dot_product_attention::<Tensor>(
                &q,
                &k,
                &v,
                None,
                0.0,
                true, // is_causal
                Some(scale),
                false,
            ),
            AttentionImplementation::Eager => {
                // For eager, use tensor scale if available for gradient flow
                let att = if let Some(ref attn_scale) = self.attn_scale {
                    q.matmul(&k.transpose(-2, -1)) * attn_scale
                } else {
                    q.matmul(&k.transpose(-2, -1)) * scale
                };
                let mask = Tensor::ones([t, t], (Kind::Float, self.device))
                    .tril(0)
                    .reshape([1, 1, t, t]);
                let att = att.masked_fill(&mask.eq(0.0), f64::NEG_INFINITY);
                let att = att.softmax(-1, Kind::Float).to_kind(kind);
                att.matmul(&v)
            }
        };

        // Reshape back to [b, t, n_embd]
        let y = y.transpose(1, 2).contiguous().reshape([b, t, -1]);

        // Apply attention gate: sigmoid gate on attention output
        let y = if let Some(ref attn_gate) = self.attn_gate {
            &y * attn_gate.sigmoid()
        } else {
            y
        };

        // Apply lambda_o before output projection
        let y = if let Some(ref lambda_o) = self.lambda_o {
            &y * lambda_o
        } else {
            y
        };

        // Output projection
        self.o_proj.forward(&y)
    }
}

/// Repeat KV heads for grouped-query attention
fn repeat_kv(x: &Tensor, n_rep: i64) -> Tensor {
    if n_rep == 1 {
        return x.shallow_clone();
    }
    let (b, n_kv_heads, t, head_dim) = x.size4().unwrap();
    x.unsqueeze(2)
        .expand([b, n_kv_heads, n_rep, t, head_dim], true)
        .reshape([b, n_kv_heads * n_rep, t, head_dim])
}

// ============================================================================
// Transformer Block
// ============================================================================

#[derive(Debug)]
struct NanoGPTBlock {
    input_norm: RMSNorm,
    attn: NanoGPTAttention,
    post_attn_norm: RMSNorm,
    mlp: NanoGPTMlp,
    // Residual enhancement
    x0_lambda: Option<Tensor>,
    resid_lambda: Option<Tensor>,
    // Smear gate (1-token lookback mixing)
    smear_gate: Option<Tensor>,
}

impl NanoGPTBlock {
    fn new(
        vs: nn::Path,
        config: &NanoGPTConfig,
        layer_idx: usize,
        current_round: Arc<AtomicU64>,
        attn_implementation: AttentionImplementation,
        comm: Option<Arc<Communicator>>,
        matformer_hidden_size: Option<i64>,
    ) -> Self {
        let hidden_size = config.hidden_size as i64;
        let intermediate_size = config.intermediate_size as i64;

        // x0_lambda - learnable scalar for x0 residual contribution
        let x0_lambda = if config.use_x0_lambdas {
            Some(vs.var("x0_lambda", &[1], nn::Init::Const(0.0))) // Start at 0
        } else {
            None
        };

        // resid_lambda - learnable scalar for block output
        let resid_lambda = if config.use_resid_lambdas {
            Some(vs.var("resid_lambda", &[1], nn::Init::Const(1.0))) // Start at 1
        } else {
            None
        };

        // smear_gate - learnable scalar for 1-token lookback mixing
        let smear_gate = if config.use_smear_gate {
            Some(vs.var("smear_gate", &[1], nn::Init::Const(0.0))) // Start at 0 (no mixing)
        } else {
            None
        };

        Self {
            input_norm: RMSNorm::new(&vs / "input_layernorm", hidden_size, config.rms_norm_eps),
            attn: NanoGPTAttention::new(
                &vs / "self_attn",
                config,
                attn_implementation,
                comm.clone(),
            ),
            post_attn_norm: RMSNorm::new(
                &vs / "post_attention_layernorm",
                hidden_size,
                config.rms_norm_eps,
            ),
            mlp: NanoGPTMlp::new(
                &vs / "mlp",
                hidden_size,
                intermediate_size,
                config.mlp_bias,
                comm,
                config.use_relu_squared_mlp,
                matformer_hidden_size,
                layer_idx,
                config.helper_config(),
                current_round,
            ),
            x0_lambda,
            resid_lambda,
            smear_gate,
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        x0: Option<&Tensor>,
        value_embed_sum: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        rope_cache: &RoPECache,
    ) -> Tensor {
        // Pre-norm + attention + residual
        let h = self.attn.forward(
            &self.input_norm.forward(x),
            value_embed_sum,
            position_ids,
            sequence_lengths,
            rope_cache,
        );
        let x = x + h;

        // Pre-norm + MLP + residual
        let h = self.mlp.forward(&self.post_attn_norm.forward(&x));
        let mut out = x + h;

        // Apply resid_lambda
        if let Some(ref resid_lambda) = self.resid_lambda {
            out = &out * resid_lambda;
        }

        // Add x0 contribution
        if let (Some(ref x0_lambda), Some(x0)) = (&self.x0_lambda, x0) {
            out = out + x0 * x0_lambda;
        }

        // Apply smear gate: mix with previous token's state
        // shifted = roll by -1 on sequence dimension (each position gets previous position's value)
        if let Some(ref smear_gate) = self.smear_gate {
            let gate = smear_gate.sigmoid();
            // Roll shifts tokens: position i gets position i-1's value
            // First position gets last position's value (wrapping)
            let shifted = out.roll(&[-1], &[1]);
            out = &out * (1.0 - &gate) + shifted * &gate;
        }

        out
    }
}

// ============================================================================
// Main Model
// ============================================================================

#[derive(Debug)]
pub struct NanoGPT {
    wte: nn::Embedding,
    blocks: Vec<NanoGPTBlock>,
    ln_f: RMSNorm,
    rope_cache: RoPECache,
    // Value embeddings - extra embedding tables mixed into V
    value_embeddings: Option<Vec<nn::Embedding>>,
    config: NanoGPTConfig,
    /// Shared round counter for helper index generation
    current_round: Arc<AtomicU64>,
}

impl NanoGPT {
    pub fn new(
        vs: nn::Path,
        config: &NanoGPTConfig,
        attn_implementation: AttentionImplementation,
        comm: Option<Arc<Communicator>>,
    ) -> Self {
        let model_vs = &vs / "model";
        let current_round = Arc::new(AtomicU64::new(0));

        // Compute MatFormer hidden size from tier
        // tier 0 = full width (None), tier N = width / 2^N
        let matformer_hidden_size: Option<i64> = match config.matformer_tier {
            0 => None,
            tier => {
                let divisor = 1_i64
                    .checked_shl(tier as u32)
                    .expect("matformer_tier too large");
                Some((config.intermediate_size as i64) / divisor)
            }
        };

        // Token embeddings
        let wte = nn::embedding(
            &model_vs / "embed_tokens",
            config.vocab_size as i64,
            config.hidden_size as i64,
            Default::default(),
        );

        // Transformer blocks
        let blocks: Vec<NanoGPTBlock> = (0..config.num_hidden_layers)
            .map(|i| {
                NanoGPTBlock::new(
                    &model_vs / "layers" / i,
                    config,
                    i,
                    current_round.clone(),
                    attn_implementation,
                    comm.clone(),
                    matformer_hidden_size,
                )
            })
            .collect();

        // Final layer norm
        let ln_f = RMSNorm::new(
            &model_vs / "norm",
            config.hidden_size as i64,
            config.rms_norm_eps,
        );

        // RoPE cache - combine rope_scaling with half_truncate setting
        let rope_config = config
            .rope_scaling
            .clone()
            .map(|mut rc| {
                rc.half_truncate = config.use_half_truncate_rope;
                rc
            })
            .or_else(|| {
                // If no rope_scaling but half_truncate is enabled, create minimal config
                if config.use_half_truncate_rope {
                    Some(RoPEConfig {
                        half_truncate: true,
                        ..Default::default()
                    })
                } else {
                    None
                }
            });
        let rope_cache = RoPECache::new(
            &rope_config,
            config.head_dim(),
            config.rope_theta,
            &vs.device(),
        );

        // Value embeddings - extra embedding tables mixed into V
        let value_embeddings = if config.num_value_embeddings > 0 {
            Some(
                (0..config.num_value_embeddings)
                    .map(|i| {
                        nn::embedding(
                            &model_vs / format!("value_embed_{}", i),
                            config.vocab_size as i64,
                            config.hidden_size as i64,
                            Default::default(),
                        )
                    })
                    .collect(),
            )
        } else {
            None
        };

        Self {
            wte,
            blocks,
            ln_f,
            rope_cache,
            value_embeddings,
            config: config.clone(),
            current_round,
        }
    }

    /// Set the current round number for helper index generation.
    /// This should be called before each forward pass during training.
    pub fn set_round(&self, round: u64) {
        self.current_round.store(round, Ordering::Relaxed);
    }

    /// Get the current round number.
    pub fn get_round(&self) -> u64 {
        self.current_round.load(Ordering::Relaxed)
    }
}

impl LanguageModelForward for NanoGPT {
    fn forward(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        _training: bool,
    ) -> Tensor {
        // Embed tokens
        let mut h = self.wte.forward(x);

        // Store x0 for residual connections (if x0_residual or x0_lambdas enabled)
        let x0 = if self.config.use_x0_residual || self.config.use_x0_lambdas {
            Some(h.shallow_clone())
        } else {
            None
        };

        // Compute value embeddings sum (added to V in attention)
        let value_embed_sum = self.value_embeddings.as_ref().map(|embeds| {
            embeds
                .iter()
                .map(|e| e.forward(x))
                .reduce(|a, b| a + b)
                .unwrap()
        });

        // Pass through blocks with optional skip connections and backout
        let mut skip_state: Option<Tensor> = None;
        let mut backout_delta: Option<Tensor> = None;
        let block_skip_from = self.config.block_skip_from;
        let block_skip_to = self.config.block_skip_to;
        let backout_layer = self.config.backout_layer;

        for (i, block) in self.blocks.iter().enumerate() {
            let h_before = h.shallow_clone(); // Save state before block (for backout)

            h = block.forward(
                &h,
                x0.as_ref(),
                value_embed_sum.as_ref(),
                position_ids,
                sequence_lengths,
                &self.rope_cache,
            );

            // Block skip - save state at specified layer
            if self.config.use_block_skip {
                if let Some(from) = block_skip_from {
                    if i == from {
                        skip_state = Some(h.shallow_clone());
                    }
                }
                // Add skip state at specified layer
                if let (Some(to), Some(ref state)) = (block_skip_to, &skip_state) {
                    if i == to {
                        h = &h + state;
                    }
                }
            }

            // Backout - store delta at specified layer
            if self.config.use_backout {
                if let Some(layer) = backout_layer {
                    if i == layer {
                        // Delta = h_after - h_before
                        backout_delta = Some(&h - &h_before);
                    }
                }
            }
        }

        // Apply backout - subtract stored delta before final norm
        let h = if let Some(ref delta) = backout_delta {
            &h - delta
        } else {
            h
        };

        // Final layer norm
        self.ln_f.forward(&h)
    }
}

// ============================================================================
// Type Alias and from_pretrained
// ============================================================================

pub type NanoGPTForCausalLM = CausalLanguageModel<NanoGPT, NanoGPTConfig>;

impl NanoGPTForCausalLM {
    fn builder(
        vs: nn::Path,
        config: &NanoGPTConfig,
        attn_implementation: Option<AttentionImplementation>,
        comm: Option<Arc<Communicator>>,
    ) -> Result<NanoGPT, ModelLoadError> {
        Ok(NanoGPT::new(
            vs,
            config,
            attn_implementation.unwrap_or_default(),
            comm,
        ))
    }

    pub fn from_pretrained(
        source: &PretrainedSource<NanoGPTConfig>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
    ) -> Result<Self, ModelLoadError> {
        Self::from_builder(
            Self::builder,
            source,
            kind,
            attn_implementation,
            device,
            tensor_parallelism_world,
            override_max_position_embeddings,
        )
    }

    pub fn from_pretrained_with_matformer_tier(
        source: &PretrainedSource<NanoGPTConfig>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
        matformer_tier: u8,
    ) -> Result<Self, ModelLoadError> {
        Self::from_pretrained_with_matformer_config(
            source,
            kind,
            attn_implementation,
            device,
            tensor_parallelism_world,
            override_max_position_embeddings,
            matformer_tier,
            0.0,
            16,
        )
    }

    pub fn from_pretrained_with_matformer_config(
        source: &PretrainedSource<NanoGPTConfig>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
        matformer_tier: u8,
        matformer_helper_fraction: f32,
        matformer_helper_rotation_interval: u64,
    ) -> Result<Self, ModelLoadError> {
        Self::from_builder_with_config_overrides(
            Self::builder,
            source,
            kind,
            attn_implementation,
            device,
            tensor_parallelism_world,
            override_max_position_embeddings,
            |config| {
                config.matformer_tier = matformer_tier;
                config.matformer_helper_fraction = matformer_helper_fraction;
                config.matformer_helper_rotation_interval = matformer_helper_rotation_interval;
            },
        )
    }
}

// ============================================================================
// TryFrom implementations for AutoConfig
// ============================================================================

use crate::AutoConfig;

impl TryFrom<AutoConfig> for NanoGPTConfig {
    type Error = ModelLoadError;

    fn try_from(value: AutoConfig) -> Result<Self, Self::Error> {
        match value {
            AutoConfig::NanoGPT(config) => Ok(config),
            _ => Err(ModelLoadError::WrongConfigType),
        }
    }
}

impl TryFrom<PretrainedSource<AutoConfig>> for PretrainedSource<NanoGPTConfig> {
    type Error = ModelLoadError;

    fn try_from(value: PretrainedSource<AutoConfig>) -> Result<Self, Self::Error> {
        match value {
            PretrainedSource::RepoFiles(path_bufs) => Ok(PretrainedSource::RepoFiles(path_bufs)),
            PretrainedSource::ConfigAndTensors(AutoConfig::NanoGPT(config), tensors) => {
                Ok(PretrainedSource::ConfigAndTensors(config, tensors))
            }
            _ => Err(ModelLoadError::WrongConfigType),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::OptimizerConfig;

    #[test]
    fn test_nanogpt_config_dummy() {
        let config = NanoGPTConfig::dummy();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_hidden_layers, 2);
        assert!(!config.use_fused_qkvo);
        assert!(!config.use_relu_squared_mlp);
    }

    #[test]
    fn test_nanogpt_config_parameter_names() {
        let config = NanoGPTConfig::dummy();
        let names = config.get_parameter_names();

        // Should have embed, blocks, norm, lm_head
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(names.contains(&"lm_head.weight".to_string()));

        // Should have layer 0 parameters
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.0.mlp.gate_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.0.input_layernorm.weight".to_string()));
    }

    #[test]
    fn test_nanogpt_forward_basic() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NanoGPTConfig::dummy();
        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let input = Tensor::randint(
            config.vocab_size as i64,
            [2, 16],
            (Kind::Int64, Device::Cpu),
        );
        let output = model.forward(&input, None, None, false);

        assert_eq!(output.size(), vec![2, 16, config.hidden_size as i64]);
    }

    #[test]
    fn test_nanogpt_mlp_relu_squared() {
        use std::sync::{atomic::AtomicU64, Arc};
        let vs = nn::VarStore::new(Device::Cpu);
        let current_round = Arc::new(AtomicU64::new(0));

        // Standard SiLU MLP
        let mlp_silu = NanoGPTMlp::new(
            vs.root() / "silu",
            64,
            128,
            false,
            None,
            false,
            None,
            0,
            None,
            current_round.clone(),
        );

        // ReLU² MLP
        let mlp_relu = NanoGPTMlp::new(
            vs.root() / "relu",
            64,
            128,
            false,
            None,
            true,
            None,
            0,
            None,
            current_round,
        );

        let input = Tensor::randn([2, 16, 64], (Kind::Float, Device::Cpu));

        let out_silu = mlp_silu.forward(&input);
        let out_relu = mlp_relu.forward(&input);

        // Both should produce same shape
        assert_eq!(out_silu.size(), out_relu.size());
        assert_eq!(out_silu.size(), vec![2, 16, 64]);

        // Outputs should be different (different activations)
        let diff = (&out_silu - &out_relu).abs().sum(Kind::Float);
        assert!(diff.double_value(&[]) > 0.0);
    }

    #[test]
    fn test_nanogpt_no_nan_inf() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NanoGPTConfig::dummy();
        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let input = Tensor::randint(
            config.vocab_size as i64,
            [2, 16],
            (Kind::Int64, Device::Cpu),
        );
        let output = model.forward(&input, None, None, false);

        assert!(!bool::try_from(output.isnan().any()).unwrap());
        assert!(!bool::try_from(output.isinf().any()).unwrap());
    }

    /// Verification test: Model can overfit a small batch.
    /// This verifies the full forward + backward pass works correctly.
    #[test]
    fn test_nanogpt_overfit() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create model with dummy config
        let config = NanoGPTConfig::dummy();
        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        // Create lm_head (language model head for logits)
        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        // Enable gradients for all parameters
        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Create optimizer
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        // Generate synthetic training data: 4 sequences of 32 tokens
        tch::manual_seed(42);
        let batch_size = 4;
        let seq_len = 32;
        let data = Tensor::randint(
            config.vocab_size as i64,
            [batch_size, seq_len],
            (Kind::Int64, device),
        );

        // Training loop
        let num_steps = 100;
        let mut losses = Vec::new();

        for _ in 0..num_steps {
            opt.zero_grad();

            // Forward pass
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            // Compute loss: shift for next-token prediction
            let shift_logits = logits.narrow(1, 0, seq_len - 1);
            let shift_labels = data.narrow(1, 1, seq_len - 1);

            let batch_seq = (batch_size * (seq_len - 1)) as i64;
            let flat_logits = shift_logits.reshape([batch_seq, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([batch_seq]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);

            // Record loss
            let loss_val = loss.double_value(&[]);
            losses.push(loss_val);

            // Check for NaN/Inf
            assert!(
                !loss_val.is_nan(),
                "Loss is NaN at step {}",
                losses.len() - 1
            );
            assert!(
                !loss_val.is_infinite(),
                "Loss is infinite at step {}",
                losses.len() - 1
            );

            // Backward and optimize
            loss.backward();
            opt.step();
        }

        // Verify significant improvement (model can memorize small batch)
        let initial_loss = losses[0];
        let final_loss = losses[num_steps - 1];
        let improvement = (initial_loss - final_loss) / initial_loss;

        eprintln!(
            "NanoGPT overfit test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        // Should have at least 50% improvement on small batch
        assert!(
            improvement > 0.5,
            "Model should improve significantly on small batch, got only {:.1}% improvement (initial: {:.4}, final: {:.4})",
            improvement * 100.0,
            initial_loss,
            final_loss
        );

        // Final loss should be quite low (can nearly memorize)
        assert!(
            final_loss < 1.0,
            "Final loss should be low for overfitting, got {:.4}",
            final_loss
        );
    }

    /// Test: Verify fused QKVO projection works correctly
    #[test]
    fn test_nanogpt_fused_qkvo() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with fused QKVO enabled
        let mut config = NanoGPTConfig::dummy();
        config.use_fused_qkvo = true;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        // Create lm_head
        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        // Enable gradients
        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Test forward pass
        let input = Tensor::randint(config.vocab_size as i64, [2, 16], (Kind::Int64, device));
        let hidden = model.forward(&input, None, None, false);
        let logits = lm_head.forward(&hidden);

        // Verify shapes
        assert_eq!(hidden.size(), vec![2, 16, config.hidden_size as i64]);
        assert_eq!(logits.size(), vec![2, 16, config.vocab_size as i64]);

        // Verify no NaN/Inf
        assert!(!bool::try_from(hidden.isnan().any()).unwrap());
        assert!(!bool::try_from(hidden.isinf().any()).unwrap());

        // Verify training works (quick overfit test)
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "Fused QKVO test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "Fused QKVO model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify ReLU² MLP works with full model
    #[test]
    fn test_nanogpt_relu_squared_full_model() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with ReLU² enabled
        let mut config = NanoGPTConfig::dummy();
        config.use_relu_squared_mlp = true;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "ReLU² full model test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "ReLU² model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify QK-Norm works correctly
    #[test]
    fn test_nanogpt_qk_norm() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with QK-Norm enabled
        let mut config = NanoGPTConfig::dummy();
        config.use_qk_norm = true;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Test forward pass
        let input = Tensor::randint(config.vocab_size as i64, [2, 16], (Kind::Int64, device));
        let hidden = model.forward(&input, None, None, false);

        // Verify no NaN/Inf (QK-Norm should help stability)
        assert!(
            !bool::try_from(hidden.isnan().any()).unwrap(),
            "QK-Norm should not produce NaN"
        );
        assert!(
            !bool::try_from(hidden.isinf().any()).unwrap(),
            "QK-Norm should not produce Inf"
        );

        // Verify training works
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            // Verify no NaN during training
            assert!(
                !loss_val.is_nan(),
                "QK-Norm: Loss became NaN at step {}",
                step
            );

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "QK-Norm test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "QK-Norm model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify SA Lambdas work correctly and have gradients
    #[test]
    fn test_nanogpt_sa_lambdas() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with SA Lambdas enabled
        let mut config = NanoGPTConfig::dummy();
        config.use_sa_lambdas = true;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Verify lambda parameters exist
        let var_names: Vec<_> = vs.variables().iter().map(|(n, _)| n.clone()).collect();
        assert!(
            var_names.iter().any(|n| n.contains("lambda_q")),
            "lambda_q should exist in variables"
        );
        assert!(
            var_names.iter().any(|n| n.contains("lambda_k")),
            "lambda_k should exist in variables"
        );
        assert!(
            var_names.iter().any(|n| n.contains("lambda_v")),
            "lambda_v should exist in variables"
        );
        assert!(
            var_names.iter().any(|n| n.contains("lambda_o")),
            "lambda_o should exist in variables"
        );

        // Verify training works
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "SA Lambdas test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "SA Lambdas model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify x0 residual and lambdas work correctly
    #[test]
    fn test_nanogpt_residual_enhancement() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with residual enhancements enabled
        let mut config = NanoGPTConfig::dummy();
        config.use_x0_residual = true;
        config.use_x0_lambdas = true;
        config.use_resid_lambdas = true;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Verify lambda parameters exist
        let var_names: Vec<_> = vs.variables().iter().map(|(n, _)| n.clone()).collect();
        assert!(
            var_names.iter().any(|n| n.contains("x0_lambda")),
            "x0_lambda should exist in variables"
        );
        assert!(
            var_names.iter().any(|n| n.contains("resid_lambda")),
            "resid_lambda should exist in variables"
        );

        // Verify training works
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "Residual enhancement test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss, final_loss, improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "Residual enhancement model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify value embeddings work correctly
    #[test]
    fn test_nanogpt_value_embeddings() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with value embeddings enabled (3 extra tables)
        let mut config = NanoGPTConfig::dummy();
        config.num_value_embeddings = 3;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Verify value embedding parameters exist
        let var_names: Vec<_> = vs.variables().iter().map(|(n, _)| n.clone()).collect();
        assert!(
            var_names.iter().any(|n| n.contains("value_embed_0")),
            "value_embed_0 should exist"
        );
        assert!(
            var_names.iter().any(|n| n.contains("value_embed_1")),
            "value_embed_1 should exist"
        );
        assert!(
            var_names.iter().any(|n| n.contains("value_embed_2")),
            "value_embed_2 should exist"
        );

        // Verify training works
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "Value embeddings test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "Value embeddings model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify skip connections and smear gate work correctly
    #[test]
    fn test_nanogpt_skip_connections() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with skip connections enabled
        // Need at least 2 layers for skip (0 -> 1)
        let mut config = NanoGPTConfig::dummy();
        config.use_block_skip = true;
        config.block_skip_from = Some(0);
        config.block_skip_to = Some(1);
        config.use_smear_gate = true;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Verify smear_gate parameters exist
        let var_names: Vec<_> = vs.variables().iter().map(|(n, _)| n.clone()).collect();
        assert!(
            var_names.iter().any(|n| n.contains("smear_gate")),
            "smear_gate should exist"
        );

        // Verify training works
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "Skip connections test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "Skip connections model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify backout works correctly
    #[test]
    fn test_nanogpt_backout() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with backout enabled (at layer 0)
        let mut config = NanoGPTConfig::dummy();
        config.use_backout = true;
        config.backout_layer = Some(0);

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Verify training works (backout should not break training)
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "Backout test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        // Backout might have different training dynamics, just verify it doesn't break
        assert!(
            improvement > 0.3,
            "Backout model should still train reasonably, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify learnable attention scale works correctly
    #[test]
    fn test_nanogpt_learnable_attn_scale() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with learnable attention scale enabled
        let mut config = NanoGPTConfig::dummy();
        config.use_learnable_attn_scale = true;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Verify attn_scale parameters exist
        let var_names: Vec<_> = vs.variables().iter().map(|(n, _)| n.clone()).collect();
        assert!(
            var_names.iter().any(|n| n.contains("attn_scale")),
            "attn_scale should exist"
        );

        // Verify training works
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "Learnable attn scale test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss, final_loss, improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "Learnable attn scale model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Verify attention gate works correctly and doesn't collapse
    #[test]
    fn test_nanogpt_attention_gate() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        // Create config with attention gate enabled
        let mut config = NanoGPTConfig::dummy();
        config.use_attention_gate = true;

        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        for (_, var) in vs.variables() {
            let _ = var.set_requires_grad(true);
        }

        // Verify attn_gate parameters exist
        let var_names: Vec<_> = vs.variables().iter().map(|(n, _)| n.clone()).collect();
        assert!(
            var_names.iter().any(|n| n.contains("attn_gate")),
            "attn_gate should exist"
        );

        // Get initial gate values
        let initial_gate_values: Vec<f64> = vs
            .variables()
            .iter()
            .filter(|(n, _)| n.contains("attn_gate"))
            .map(|(_, v)| v.double_value(&[]))
            .collect();

        // Verify training works
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        tch::manual_seed(42);
        let data = Tensor::randint(config.vocab_size as i64, [4, 32], (Kind::Int64, device));

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            opt.zero_grad();
            let hidden = model.forward(&data, None, None, true);
            let logits = lm_head.forward(&hidden);

            let shift_logits = logits.narrow(1, 0, 31);
            let shift_labels = data.narrow(1, 1, 31);
            let flat_logits = shift_logits.reshape([4 * 31, config.vocab_size as i64]);
            let flat_labels = shift_labels.reshape([4 * 31]);

            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            let loss_val = loss.double_value(&[]);

            if step == 0 {
                initial_loss = loss_val;
            }
            if step == 49 {
                final_loss = loss_val;
            }

            loss.backward();
            opt.step();
        }

        // Get final gate values and check they haven't collapsed to 0 or 1
        let final_gate_values: Vec<f64> = vs
            .variables()
            .iter()
            .filter(|(n, _)| n.contains("attn_gate"))
            .map(|(_, v)| v.double_value(&[]))
            .collect();

        // Verify gates have moved from initial values (have gradients)
        let gates_changed = initial_gate_values
            .iter()
            .zip(final_gate_values.iter())
            .any(|(init, fin)| (init - fin).abs() > 1e-6);
        assert!(
            gates_changed,
            "Attention gates should receive gradients and change during training"
        );

        // Verify gates haven't collapsed (sigmoid output not too close to 0 or 1)
        for gate_val in &final_gate_values {
            let sigmoid_val = 1.0 / (1.0 + (-gate_val).exp());
            assert!(
                sigmoid_val > 0.01 && sigmoid_val < 0.99,
                "Attention gate should not collapse, got sigmoid = {:.4} from gate = {:.4}",
                sigmoid_val,
                gate_val
            );
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "Attention gate test: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        assert!(
            improvement > 0.5,
            "Attention gate model should train, got {:.1}% improvement",
            improvement * 100.0
        );
    }

    /// Test: Load real FineWeb GPT-2 tokenized data if available
    /// This test is skipped if data/fineweb10B doesn't exist.
    /// Run `python scripts/download_fineweb10B.py --val-only` to download test data.
    #[test]
    fn test_nanogpt_fineweb_data_loading() {
        use std::path::{Path, PathBuf};

        // Navigate from crate directory to workspace root
        let workspace_root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent() // shared
            .unwrap()
            .parent() // workspace root
            .unwrap()
            .into();
        let data_dir = workspace_root.join("data/fineweb10B");
        if !data_dir.exists() {
            eprintln!(
                "Skipping fineweb test: {} not found. Run: python scripts/download_fineweb10B.py --val-only",
                data_dir.display()
            );
            return;
        }

        // Check for validation file
        let val_file = data_dir.join("fineweb_val_000000.bin");
        if !val_file.exists() {
            eprintln!("Skipping fineweb test: validation file not found");
            return;
        }

        // Load raw binary data
        let data = std::fs::read(&val_file).expect("Failed to read validation file");
        assert!(data.len() >= 1024, "Validation file too small");

        // Detect modded-nanogpt format by checking magic number
        const MODDED_NANOGPT_MAGIC: u32 = 20240520;
        const HEADER_SIZE: usize = 1024;

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let data_start = if magic == MODDED_NANOGPT_MAGIC {
            eprintln!(
                "Detected modded-nanogpt format, skipping {} byte header",
                HEADER_SIZE
            );
            HEADER_SIZE
        } else {
            0
        };

        // Parse as uint16 tokens (skip header if present)
        let tokens: Vec<u16> = data[data_start..]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        eprintln!("Loaded {} tokens from {}", tokens.len(), val_file.display());

        // Verify tokens are in valid GPT-2 range (0-50256)
        let max_token = *tokens.iter().max().unwrap();
        let min_token = *tokens.iter().min().unwrap();
        eprintln!("Token range: {} - {}", min_token, max_token);

        assert!(
            max_token <= 50256,
            "Token {} exceeds GPT-2 vocab size 50257",
            max_token
        );

        // Create a small batch from the data
        let seq_len = 64;
        let batch_size = 4;
        let required_tokens = batch_size * (seq_len + 1);

        if tokens.len() < required_tokens {
            eprintln!(
                "Skipping forward pass: need {} tokens, have {}",
                required_tokens,
                tokens.len()
            );
            return;
        }

        // Convert to tensor
        let device = Device::Cpu;
        let batch_tokens: Vec<i64> = tokens[..required_tokens]
            .iter()
            .map(|&t| t as i64)
            .collect();
        let input = Tensor::from_slice(&batch_tokens)
            .reshape([batch_size as i64, (seq_len + 1) as i64])
            .to_device(device);

        // Create 124M config matching modded-nanogpt
        let config = NanoGPTConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            vocab_size: 50257,
            num_hidden_layers: 11,
            num_attention_heads: 6,
            num_key_value_heads: Some(6),
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            // Enable core NanoGPT features
            use_fused_qkvo: false,
            use_relu_squared_mlp: true,
            mlp_bias: false,
            use_qk_norm: true,
            qk_norm_before_rope: true,
            use_sa_lambdas: false,
            use_x0_residual: false,
            use_x0_lambdas: false,
            use_resid_lambdas: false,
            num_value_embeddings: 0,
            use_block_skip: false,
            block_skip_from: None,
            block_skip_to: None,
            use_smear_gate: false,
            use_logit_softcap: false,
            softcap_scale: None,
            use_backout: false,
            backout_layer: None,
            use_half_truncate_rope: false,
            use_key_offset: false,
            key_offset: None,
            use_learnable_attn_scale: false,
            attention_scale: None,
            use_attention_gate: false,
            matformer_tier: 0,
            matformer_helper_fraction: 0.0,
            matformer_helper_rotation_interval: 16,
        };

        let vs = nn::VarStore::new(device);
        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        // Create lm_head
        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        // Forward pass with real data
        let input_ids = input.narrow(1, 0, seq_len as i64);
        let labels = input.narrow(1, 1, seq_len as i64);

        let hidden = model.forward(&input_ids, None, None, false);
        let logits = lm_head.forward(&hidden);

        // Compute loss
        let flat_logits =
            logits.reshape([batch_size as i64 * seq_len as i64, config.vocab_size as i64]);
        let flat_labels = labels.reshape([batch_size as i64 * seq_len as i64]);
        let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
        let loss_val = loss.double_value(&[]);

        eprintln!(
            "NanoGPT 124M on FineWeb data: batch_size={}, seq_len={}, loss={:.4}",
            batch_size, seq_len, loss_val
        );

        // Verify loss is reasonable (untrained model should be ~ln(vocab_size) ≈ 10.8)
        assert!(
            loss_val > 5.0 && loss_val < 15.0,
            "Initial loss {} is outside expected range for untrained model",
            loss_val
        );

        // Verify no NaN/Inf
        assert!(!loss_val.is_nan(), "Loss is NaN");
        assert!(!loss_val.is_infinite(), "Loss is infinite");

        eprintln!("FineWeb data loading test passed!");
    }

    /// Test: Verify NanoGPT 124M config loads from JSON
    #[test]
    fn test_nanogpt_124m_config_from_json() {
        use std::path::{Path, PathBuf};

        // Navigate from crate directory to workspace root
        let workspace_root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent() // shared
            .unwrap()
            .parent() // workspace root
            .unwrap()
            .into();
        let config_path = workspace_root.join("config/nanogpt-124m/config.json");
        if !config_path.exists() {
            eprintln!("Skipping config test: {} not found", config_path.display());
            return;
        }

        let config_str = std::fs::read_to_string(&config_path).expect("Failed to read config");
        let config: NanoGPTConfig =
            serde_json::from_str(&config_str).expect("Failed to parse config");

        // Verify 124M model dimensions
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.num_hidden_layers, 11);
        assert_eq!(config.num_attention_heads, 6);

        // Verify head_dim = hidden_size / num_heads = 128
        let head_dim = config.hidden_size / config.num_attention_heads;
        assert_eq!(head_dim, 128);

        eprintln!("NanoGPT 124M config loaded successfully:");
        eprintln!("  hidden_size: {}", config.hidden_size);
        eprintln!("  num_layers: {}", config.num_hidden_layers);
        eprintln!("  num_heads: {}", config.num_attention_heads);
        eprintln!("  head_dim: {}", head_dim);
        eprintln!("  vocab_size: {}", config.vocab_size);
        eprintln!("  use_relu_squared_mlp: {}", config.use_relu_squared_mlp);
        eprintln!("  use_qk_norm: {}", config.use_qk_norm);
    }

    // ========================================================================
    // MatFormer Tests
    // ========================================================================

    /// Test: MatFormer MLP produces zero gradients in tail weights
    /// This is the key property of MatFormer - unused neurons don't get gradients
    #[test]
    fn test_nanogpt_matformer_mlp_has_zero_tail_grads() {
        use std::sync::{atomic::AtomicU64, Arc};
        let vs = nn::VarStore::new(Device::Cpu);
        let hidden_size = 4;
        let intermediate_size = 8;
        let matformer_hidden = 4; // Use half the FFN width
        let current_round = Arc::new(AtomicU64::new(0));

        let mlp = NanoGPTMlp::new(
            vs.root(),
            hidden_size,
            intermediate_size,
            false, // no bias
            None,  // no comm
            false, // SiLU, not ReLU²
            Some(matformer_hidden),
            0,    // layer_idx
            None, // no helper mode
            current_round,
        );

        // Forward + backward
        let x =
            Tensor::randn([2, 3, hidden_size], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let y = mlp.forward(&x);
        let loss = y.sum(Kind::Float);
        loss.backward();

        // Get gradients
        let gate_grad = mlp.gate_proj.linear.ws.grad();
        let up_grad = mlp.up_proj.linear.ws.grad();
        let down_grad = mlp.down_proj.linear.ws.grad();

        // gate/up: [intermediate_size, hidden_size] => tail rows must have zero grad
        let gate_tail = gate_grad.narrow(0, matformer_hidden, intermediate_size - matformer_hidden);
        let up_tail = up_grad.narrow(0, matformer_hidden, intermediate_size - matformer_hidden);

        // down: [hidden_size, intermediate_size] => tail cols must have zero grad
        let down_tail = down_grad.narrow(1, matformer_hidden, intermediate_size - matformer_hidden);

        let gate_tail_max = gate_tail.abs().max().double_value(&[]);
        let up_tail_max = up_tail.abs().max().double_value(&[]);
        let down_tail_max = down_tail.abs().max().double_value(&[]);

        eprintln!("MatFormer MLP tail gradients:");
        eprintln!("  gate_proj tail max: {:.2e}", gate_tail_max);
        eprintln!("  up_proj tail max: {:.2e}", up_tail_max);
        eprintln!("  down_proj tail max: {:.2e}", down_tail_max);

        assert!(
            gate_tail_max < 1e-10,
            "gate_proj tail should have zero gradients, got max {}",
            gate_tail_max
        );
        assert!(
            up_tail_max < 1e-10,
            "up_proj tail should have zero gradients, got max {}",
            up_tail_max
        );
        assert!(
            down_tail_max < 1e-10,
            "down_proj tail should have zero gradients, got max {}",
            down_tail_max
        );

        eprintln!("MatFormer MLP zero-tail-grads test passed!");
    }

    #[test]
    #[should_panic(expected = "mlp_bias is not supported with MatFormer tiers")]
    fn test_nanogpt_matformer_mlp_bias_not_supported() {
        use std::sync::{atomic::AtomicU64, Arc};
        let vs = nn::VarStore::new(Device::Cpu);
        let hidden_size = 1;
        let intermediate_size = 2;
        let matformer_hidden = 1;
        let current_round = Arc::new(AtomicU64::new(0));

        let _ = NanoGPTMlp::new(
            vs.root(),
            hidden_size,
            intermediate_size,
            true,  // bias enabled
            None,  // no comm
            false, // SiLU, not ReLU²
            Some(matformer_hidden),
            0,    // layer_idx
            None, // no helper mode
            current_round,
        );
    }

    /// Test: MatFormer MLP with ReLU² also has zero tail gradients
    #[test]
    fn test_nanogpt_matformer_mlp_relu_squared_zero_tail_grads() {
        use std::sync::{atomic::AtomicU64, Arc};
        let vs = nn::VarStore::new(Device::Cpu);
        let hidden_size = 4;
        let intermediate_size = 8;
        let matformer_hidden = 4;
        let current_round = Arc::new(AtomicU64::new(0));

        let mlp = NanoGPTMlp::new(
            vs.root(),
            hidden_size,
            intermediate_size,
            false,
            None,
            true, // ReLU²
            Some(matformer_hidden),
            0,    // layer_idx
            None, // no helper mode
            current_round,
        );

        let x =
            Tensor::randn([2, 3, hidden_size], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let y = mlp.forward(&x);
        let loss = y.sum(Kind::Float);
        loss.backward();

        let gate_grad = mlp.gate_proj.linear.ws.grad();
        let gate_tail = gate_grad.narrow(0, matformer_hidden, intermediate_size - matformer_hidden);
        let gate_tail_max = gate_tail.abs().max().double_value(&[]);

        assert!(
            gate_tail_max < 1e-10,
            "ReLU² MatFormer tail should have zero gradients"
        );

        eprintln!("MatFormer MLP ReLU² zero-tail-grads test passed!");
    }

    /// Test: Full model overfit with MatFormer tier 1 (half FFN width)
    #[test]
    fn test_nanogpt_matformer_tier1_overfit() {
        let device = Device::Cpu;

        let config = NanoGPTConfig {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 100,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            use_fused_qkvo: false,
            use_relu_squared_mlp: false,
            mlp_bias: false,
            use_qk_norm: false,
            qk_norm_before_rope: true,
            use_sa_lambdas: false,
            use_x0_residual: false,
            use_x0_lambdas: false,
            use_resid_lambdas: false,
            num_value_embeddings: 0,
            use_block_skip: false,
            block_skip_from: None,
            block_skip_to: None,
            use_smear_gate: false,
            use_logit_softcap: false,
            softcap_scale: None,
            use_backout: false,
            backout_layer: None,
            use_half_truncate_rope: false,
            use_key_offset: false,
            key_offset: None,
            use_learnable_attn_scale: false,
            attention_scale: None,
            use_attention_gate: false,
            matformer_tier: 1, // Half FFN width (128 -> 64)
            matformer_helper_fraction: 0.0,
            matformer_helper_rotation_interval: 16,
        };

        let vs = nn::VarStore::new(device);
        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        // Small overfit dataset
        let batch_size = 4;
        let seq_len = 16;
        let data = Tensor::randint(
            config.vocab_size as i64,
            [batch_size, seq_len + 1],
            (Kind::Int64, device),
        );
        let input_ids = data.narrow(1, 0, seq_len);
        let labels = data.narrow(1, 1, seq_len);

        let initial_loss = {
            let hidden = model.forward(&input_ids, None, None, false);
            let logits = lm_head.forward(&hidden);
            let flat_logits = logits.reshape([batch_size * seq_len, config.vocab_size as i64]);
            let flat_labels = labels.reshape([batch_size * seq_len]);
            flat_logits
                .cross_entropy_for_logits(&flat_labels)
                .double_value(&[])
        };

        // Train for 100 steps
        for _ in 0..100 {
            let hidden = model.forward(&input_ids, None, None, false);
            let logits = lm_head.forward(&hidden);
            let flat_logits = logits.reshape([batch_size * seq_len, config.vocab_size as i64]);
            let flat_labels = labels.reshape([batch_size * seq_len]);
            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            opt.backward_step(&loss);
        }

        let final_loss = {
            let hidden = model.forward(&input_ids, None, None, false);
            let logits = lm_head.forward(&hidden);
            let flat_logits = logits.reshape([batch_size * seq_len, config.vocab_size as i64]);
            let flat_labels = labels.reshape([batch_size * seq_len]);
            flat_logits
                .cross_entropy_for_logits(&flat_labels)
                .double_value(&[])
        };

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "MatFormer tier 1 overfit: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        assert!(
            improvement > 0.90,
            "MatFormer tier 1 should achieve >90% loss reduction, got {:.1}%",
            improvement * 100.0
        );
        assert!(!final_loss.is_nan(), "Final loss is NaN");

        eprintln!("MatFormer tier 1 overfit test passed!");
    }

    /// Test: MatFormer tier 2 (quarter FFN width) also works
    #[test]
    fn test_nanogpt_matformer_tier2_overfit() {
        let device = Device::Cpu;

        let config = NanoGPTConfig {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 100,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            use_fused_qkvo: false,
            use_relu_squared_mlp: false,
            mlp_bias: false,
            use_qk_norm: false,
            qk_norm_before_rope: true,
            use_sa_lambdas: false,
            use_x0_residual: false,
            use_x0_lambdas: false,
            use_resid_lambdas: false,
            num_value_embeddings: 0,
            use_block_skip: false,
            block_skip_from: None,
            block_skip_to: None,
            use_smear_gate: false,
            use_logit_softcap: false,
            softcap_scale: None,
            use_backout: false,
            backout_layer: None,
            use_half_truncate_rope: false,
            use_key_offset: false,
            key_offset: None,
            use_learnable_attn_scale: false,
            attention_scale: None,
            use_attention_gate: false,
            matformer_tier: 2, // Quarter FFN width (128 -> 32)
            matformer_helper_fraction: 0.0,
            matformer_helper_rotation_interval: 16,
        };

        let vs = nn::VarStore::new(device);
        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        let batch_size = 4;
        let seq_len = 16;
        let data = Tensor::randint(
            config.vocab_size as i64,
            [batch_size, seq_len + 1],
            (Kind::Int64, device),
        );
        let input_ids = data.narrow(1, 0, seq_len);
        let labels = data.narrow(1, 1, seq_len);

        let initial_loss = {
            let hidden = model.forward(&input_ids, None, None, false);
            let logits = lm_head.forward(&hidden);
            let flat_logits = logits.reshape([batch_size * seq_len, config.vocab_size as i64]);
            let flat_labels = labels.reshape([batch_size * seq_len]);
            flat_logits
                .cross_entropy_for_logits(&flat_labels)
                .double_value(&[])
        };

        for _ in 0..100 {
            let hidden = model.forward(&input_ids, None, None, false);
            let logits = lm_head.forward(&hidden);
            let flat_logits = logits.reshape([batch_size * seq_len, config.vocab_size as i64]);
            let flat_labels = labels.reshape([batch_size * seq_len]);
            let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
            opt.backward_step(&loss);
        }

        let final_loss = {
            let hidden = model.forward(&input_ids, None, None, false);
            let logits = lm_head.forward(&hidden);
            let flat_logits = logits.reshape([batch_size * seq_len, config.vocab_size as i64]);
            let flat_labels = labels.reshape([batch_size * seq_len]);
            flat_logits
                .cross_entropy_for_logits(&flat_labels)
                .double_value(&[])
        };

        let improvement = (initial_loss - final_loss) / initial_loss;
        eprintln!(
            "MatFormer tier 2 overfit: initial loss = {:.4}, final loss = {:.4}, improvement = {:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        // Tier 2 has less capacity, so we accept slightly lower threshold
        assert!(
            improvement > 0.85,
            "MatFormer tier 2 should achieve >85% loss reduction, got {:.1}%",
            improvement * 100.0
        );
        assert!(!final_loss.is_nan(), "Final loss is NaN");

        eprintln!("MatFormer tier 2 overfit test passed!");
    }

    /// Test: Verify MatFormer tier 0 is equivalent to no MatFormer
    #[test]
    fn test_nanogpt_matformer_tier0_is_full_width() {
        let device = Device::Cpu;

        // Config with tier 0 (should be full width)
        let config = NanoGPTConfig {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 50,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: Some(2),
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            use_fused_qkvo: false,
            use_relu_squared_mlp: false,
            mlp_bias: false,
            use_qk_norm: false,
            qk_norm_before_rope: true,
            use_sa_lambdas: false,
            use_x0_residual: false,
            use_x0_lambdas: false,
            use_resid_lambdas: false,
            num_value_embeddings: 0,
            use_block_skip: false,
            block_skip_from: None,
            block_skip_to: None,
            use_smear_gate: false,
            use_logit_softcap: false,
            softcap_scale: None,
            use_backout: false,
            backout_layer: None,
            use_half_truncate_rope: false,
            use_key_offset: false,
            key_offset: None,
            use_learnable_attn_scale: false,
            attention_scale: None,
            use_attention_gate: false,
            matformer_tier: 0, // Full width
            matformer_helper_fraction: 0.0,
            matformer_helper_rotation_interval: 16,
        };

        let vs = nn::VarStore::new(device);
        let model = NanoGPT::new(vs.root(), &config, AttentionImplementation::Eager, None);

        let lm_head = nn::linear(
            vs.root() / "lm_head",
            config.hidden_size as i64,
            config.vocab_size as i64,
            Default::default(),
        );

        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        let batch_size = 4;
        let seq_len = 8;
        let data = Tensor::randint(
            config.vocab_size as i64,
            [batch_size, seq_len + 1],
            (Kind::Int64, device),
        );
        let input_ids = data.narrow(1, 0, seq_len);
        let labels = data.narrow(1, 1, seq_len);

        // Train and check all weights get gradients (not just subset)
        let hidden = model.forward(&input_ids, None, None, false);
        let logits = lm_head.forward(&hidden);
        let flat_logits = logits.reshape([batch_size * seq_len, config.vocab_size as i64]);
        let flat_labels = labels.reshape([batch_size * seq_len]);
        let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
        opt.backward_step(&loss);

        // Check that all MLP weights have non-zero gradients
        // (This verifies tier 0 uses full width, not sliced)
        for (name, var) in vs.variables() {
            if name.contains("mlp") && name.contains("weight") {
                let grad = var.grad();
                let grad_norm = grad.abs().sum(Kind::Float).double_value(&[]);
                assert!(
                    grad_norm > 0.0,
                    "Tier 0 should have gradients in all MLP weights, but {} has zero grad",
                    name
                );
            }
        }

        eprintln!("MatFormer tier 0 (full width) test passed!");
    }
}
