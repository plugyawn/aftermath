use crate::{
    default_rope, matformer_helper::HelperConfig, parallelism::Communicator,
    AttentionImplementation, AutoConfig, CausalLanguageModel, CausalSelfAttention,
    ColumnParallelLinear, CommunicatorId, EosToks, LanguageModelConfig, LanguageModelForward,
    ModelConfig, ModelLoadError, PretrainedSource, RMSNorm, RoPECache, RoPEConfig,
    RowParallelLinear,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tch::{
    nn::{self, Module},
    Device, Kind, Tensor,
};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub matformer_tier: u8,
    /// Fraction of suffix neurons to help train (0.0-1.0) for stochastic suffix sampling.
    #[serde(default)]
    pub matformer_helper_fraction: f32,
    /// Rounds to keep helper indices fixed before rotating.
    #[serde(default = "default_helper_rotation_interval")]
    pub matformer_helper_rotation_interval: u64,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<EosToks>,
    pub rope_scaling: Option<RoPEConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub attention_bias: Option<bool>,
    /// Sliding window attention size. If set, each token only attends to the
    /// last `sliding_window` tokens instead of the full sequence.
    /// Compatible with Flash Attention 2 only.
    #[serde(default)]
    pub sliding_window: Option<i64>,
}

fn default_helper_rotation_interval() -> u64 {
    16
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn dummy() -> Self {
        Self {
            hidden_size: 1,
            intermediate_size: 1,
            matformer_tier: 0,
            matformer_helper_fraction: 0.0,
            matformer_helper_rotation_interval: 16,
            vocab_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: Some(1),
            rms_norm_eps: 0.00001,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(EosToks::Single(1)),
            rope_scaling: None,
            max_position_embeddings: 2048,
            tie_word_embeddings: false,
            attention_bias: None,
            sliding_window: None,
        }
    }

    /// Get helper config if helper mode is enabled.
    pub fn helper_config(&self) -> Option<HelperConfig> {
        if self.matformer_helper_fraction > 0.0 && self.matformer_tier > 0 {
            Some(HelperConfig::new(
                self.matformer_helper_fraction,
                self.matformer_helper_rotation_interval,
            ))
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Mlp {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
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

impl Mlp {
    fn new(
        vs: nn::Path,
        n_embd: i64,
        n_hidden: i64,
        matformer_hidden_size: Option<i64>,
        layer_idx: usize,
        helper_config: Option<HelperConfig>,
        current_round: Arc<AtomicU64>,
        comm: Option<Arc<Communicator>>,
    ) -> Self {
        let is_tensor_parallel = comm.as_ref().map(|x| x.size()).unwrap_or(1) > 1;
        let tp_size = comm.as_ref().map(|x| x.size()).unwrap_or(1);
        assert_eq!(
            n_hidden % tp_size,
            0,
            "n_hidden must be divisible by tp_size"
        );
        if let Some(ref config) = helper_config {
            assert!(
                config.helper_fraction <= 0.0,
                "MatFormer helper mode is temporarily disabled (suffix sampling not wired)"
            );
        }
        if let Some(matformer_hidden_size) = matformer_hidden_size {
            assert!(
                matformer_hidden_size > 0,
                "matformer_hidden_size must be > 0"
            );
            assert!(
                matformer_hidden_size <= n_hidden,
                "matformer_hidden_size must be <= n_hidden"
            );
            assert_eq!(
                matformer_hidden_size % tp_size,
                0,
                "matformer_hidden_size must be divisible by tp_size"
            );
        }

        let gate_proj = ColumnParallelLinear::new(
            &vs / "gate_proj",
            n_embd,
            n_hidden,
            false,
            false,
            comm.clone(),
        );
        let up_proj = ColumnParallelLinear::new(
            &vs / "up_proj",
            n_embd,
            n_hidden,
            false,
            false,
            comm.clone(),
        );
        let down_proj =
            RowParallelLinear::new(&vs / "down_proj", n_hidden, n_embd, false, true, comm);

        if let Some(ref config) = helper_config {
            eprintln!(
                "[MLP layer {}] Helper mode enabled: fraction={}, rotation_interval={}, matformer_hidden={}",
                layer_idx, config.helper_fraction, config.rotation_interval, matformer_hidden_size.unwrap_or(n_hidden)
            );
        }

        Self {
            gate_proj,
            up_proj,
            down_proj,
            matformer_hidden_size,
            full_intermediate_size: n_hidden,
            layer_idx,
            helper_config,
            current_round,
            is_tensor_parallel,
        }
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // Tier 0 (full width): use standard forward
        let Some(matformer_hidden_size) = self.matformer_hidden_size else {
            return self
                .down_proj
                .forward(&(self.gate_proj.forward(xs).silu() * self.up_proj.forward(xs)));
        };
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
            // - gate_proj/up_proj: [n_hidden, n_embd] - select rows
            // - down_proj: [n_embd, n_hidden] - select columns
            let gate_w = self.gate_proj.linear.ws.index_select(0, &indices_tensor);
            let up_w = self.up_proj.linear.ws.index_select(0, &indices_tensor);
            let down_w = self.down_proj.linear.ws.index_select(1, &indices_tensor);

            let gate = xs.matmul(&gate_w.transpose(0, 1));
            let up = xs.matmul(&up_w.transpose(0, 1));
            let hidden = gate.silu() * up;
            hidden.matmul(&down_w.transpose(0, 1))
        } else {
            // Standard MatFormer: use narrow for contiguous prefix (more efficient)
            // Weight shapes:
            // - gate_proj/up_proj: [n_hidden, n_embd]
            // - down_proj: [n_embd, n_hidden]
            let gate_w = self.gate_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
            let up_w = self.up_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
            let down_w = self.down_proj.linear.ws.narrow(1, 0, matformer_hidden_size);

            let gate = xs.matmul(&gate_w.transpose(0, 1));
            let up = xs.matmul(&up_w.transpose(0, 1));
            let hidden = gate.silu() * up;
            hidden.matmul(&down_w.transpose(0, 1))
        }
    }
}

#[derive(Debug)]
struct Block {
    rms_1: RMSNorm,
    attn: CausalSelfAttention,
    rms_2: RMSNorm,
    mlp: Mlp,
}

impl Block {
    fn new(
        vs: nn::Path,
        config: &LlamaConfig,
        layer_idx: usize,
        current_round: Arc<AtomicU64>,
        attn_implementation: AttentionImplementation,
        comm: Option<Arc<Communicator>>,
    ) -> Self {
        let rms_1 = RMSNorm::new(
            &vs / "input_layernorm",
            config.hidden_size as i64,
            config.rms_norm_eps,
        );
        let attn = CausalSelfAttention::new(
            &vs / "self_attn",
            config.num_attention_heads as i64,
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads) as i64,
            config.hidden_size as i64,
            (config.max_position_embeddings + 1) as i64,
            attn_implementation,
            comm.clone(),
            config.sliding_window,
        );
        let rms_2 = RMSNorm::new(
            &vs / "post_attention_layernorm",
            config.hidden_size as i64,
            config.rms_norm_eps,
        );
        let mlp = Mlp::new(
            &vs / "mlp",
            config.hidden_size as i64,
            config.intermediate_size as i64,
            match config.matformer_tier {
                0 => None,
                tier => {
                    let divisor = 1_i64
                        .checked_shl(tier as u32)
                        .expect("matformer_tier too large");
                    Some((config.intermediate_size as i64) / divisor)
                }
            },
            layer_idx,
            config.helper_config(),
            current_round,
            comm,
        );
        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&(Tensor, i32)>,
        cache: &RoPECache,
    ) -> Tensor {
        let x = self.attn.forward(
            &self.rms_1.forward(x),
            position_ids,
            sequence_lengths,
            cache,
        ) + x;
        self.mlp.forward(&self.rms_2.forward(&x)) + x
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Llama {
    wte: nn::Embedding,
    blocks: Vec<Block>,
    ln_f: RMSNorm,
    attn_implementation: AttentionImplementation,
    rope_cache: RoPECache,
    /// Shared round counter for helper index generation
    current_round: Arc<AtomicU64>,
}

impl Llama {
    pub fn new(
        vs: nn::Path,
        config: &LlamaConfig,
        attn_implementation: AttentionImplementation,
        comm: Option<Arc<Communicator>>,
    ) -> Self {
        let current_round = Arc::new(AtomicU64::new(0));
        let wte = nn::embedding(
            &vs / "model" / "embed_tokens",
            config.vocab_size as i64,
            config.hidden_size as i64,
            Default::default(),
        );
        let ln_f = RMSNorm::new(
            &vs / "model" / "norm",
            config.hidden_size as i64,
            config.rms_norm_eps,
        );
        let blocks = (0..config.num_hidden_layers)
            .map(|i| {
                Block::new(
                    &vs / "model" / "layers" / i,
                    config,
                    i,
                    current_round.clone(),
                    attn_implementation,
                    comm.clone(),
                )
            })
            .collect::<Vec<_>>();
        let rope_cache = RoPECache::new(
            &config.rope_config(),
            config.hidden_size() / config.num_attention_heads(),
            config.rope_theta(),
            &vs.device(),
        );
        Self {
            wte,
            blocks,
            ln_f,
            attn_implementation,
            rope_cache,
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

impl LanguageModelForward for Llama {
    #[allow(unused_variables)]
    fn forward(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        _training: bool,
    ) -> Tensor {
        let sequence_lengths = sequence_lengths.map(|sequence_lengths| {
            #[cfg(feature = "parallelism")]
            {
                if self.attn_implementation == AttentionImplementation::FlashAttention2 {
                    crate::attention::create_cu_seqlens(sequence_lengths, x.device())
                } else {
                    panic!("`sequence_lengths` only supported for FlashAttention2");
                }
            }

            #[cfg(not(feature = "parallelism"))]
            {
                panic!("`sequence_lengths` only supported for FlashAttention2");
            }
        });

        let mut x = self.wte.forward(x);
        for block in &self.blocks {
            x = block.forward(
                &x,
                position_ids,
                sequence_lengths.as_ref(),
                &self.rope_cache,
            );
        }
        self.ln_f.forward(&x)
    }
}

pub type LlamaForCausalLM = CausalLanguageModel<Llama, LlamaConfig>;

impl LlamaForCausalLM {
    fn builder(
        vs: nn::Path,
        config: &LlamaConfig,
        attn_implementation: Option<AttentionImplementation>,
        comm: Option<Arc<Communicator>>,
    ) -> Result<Llama, ModelLoadError> {
        Ok(Llama::new(
            vs,
            config,
            attn_implementation.unwrap_or_default(),
            comm,
        ))
    }

    pub fn from_pretrained(
        source: &PretrainedSource<LlamaConfig>,
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
        source: &PretrainedSource<LlamaConfig>,
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
        source: &PretrainedSource<LlamaConfig>,
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

impl ModelConfig for LlamaConfig {
    fn get_parameter_names(&self) -> Vec<String> {
        let mut variables = Vec::new();
        for layer_idx in 0..self.num_hidden_layers {
            let layer_prefix = format!("model.layers.{}", layer_idx);

            variables.push(format!("{}.self_attn.q_proj.weight", layer_prefix));
            variables.push(format!("{}.self_attn.k_proj.weight", layer_prefix));
            variables.push(format!("{}.self_attn.v_proj.weight", layer_prefix));
            variables.push(format!("{}.self_attn.o_proj.weight", layer_prefix));

            variables.push(format!("{}.mlp.gate_proj.weight", layer_prefix));
            variables.push(format!("{}.mlp.up_proj.weight", layer_prefix));
            variables.push(format!("{}.mlp.down_proj.weight", layer_prefix));

            variables.push(format!("{}.input_layernorm.weight", layer_prefix));
            variables.push(format!("{}.post_attention_layernorm.weight", layer_prefix));

            if self.attention_bias.unwrap_or(false) {
                variables.push(format!("{}.self_attn.q_proj.bias", layer_prefix));
                variables.push(format!("{}.self_attn.k_proj.bias", layer_prefix));
                variables.push(format!("{}.self_attn.v_proj.bias", layer_prefix));
            }
        }

        variables.push("lm_head.weight".to_string());
        variables.push("model.norm.weight".to_string());
        variables.push("model.embed_tokens.weight".to_string());

        variables
    }
}

impl TryFrom<AutoConfig> for LlamaConfig {
    type Error = ModelLoadError;

    fn try_from(value: AutoConfig) -> Result<Self, Self::Error> {
        match value {
            AutoConfig::Llama(llama_config) => Ok(llama_config),
            _ => Err(ModelLoadError::WrongConfigType),
        }
    }
}

impl TryFrom<PretrainedSource<AutoConfig>> for PretrainedSource<LlamaConfig> {
    type Error = ModelLoadError;

    fn try_from(value: PretrainedSource<AutoConfig>) -> Result<Self, Self::Error> {
        match value {
            PretrainedSource::RepoFiles(path_bufs) => Ok(PretrainedSource::RepoFiles(path_bufs)),
            PretrainedSource::ConfigAndTensors(AutoConfig::Llama(config), hash_map) => {
                Ok(PretrainedSource::ConfigAndTensors(config, hash_map))
            }
            _ => Err(ModelLoadError::WrongConfigType),
        }
    }
}

impl LanguageModelConfig for LlamaConfig {
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
}

#[cfg(test)]
mod tests {
    use super::Mlp;
    use std::sync::{atomic::AtomicU64, Arc};
    use tch::nn::Module;
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn matformer_mlp_has_zero_tail_grads() {
        let vs = nn::VarStore::new(Device::Cpu);
        let n_embd = 4;
        let n_hidden = 8;
        let matformer_hidden = 4;
        let current_round = Arc::new(AtomicU64::new(0));

        let mlp = Mlp::new(
            vs.root(),
            n_embd,
            n_hidden,
            Some(matformer_hidden),
            0,    // layer_idx
            None, // helper_config (no helper mode for this test)
            current_round,
            None,
        );

        let xs = Tensor::randn([2, 3, n_embd], (Kind::Float, Device::Cpu));
        let out = mlp.forward(&xs);
        let loss = out.sum(Kind::Float);
        loss.backward();

        let gate_grad = mlp.gate_proj.linear.ws.grad();
        let up_grad = mlp.up_proj.linear.ws.grad();
        let down_grad = mlp.down_proj.linear.ws.grad();

        // gate/up: [n_hidden, n_embd] => tail rows must have zero grad
        let gate_tail = gate_grad.narrow(0, matformer_hidden, n_hidden - matformer_hidden);
        let up_tail = up_grad.narrow(0, matformer_hidden, n_hidden - matformer_hidden);

        // down: [n_embd, n_hidden] => tail cols must have zero grad
        let down_tail = down_grad.narrow(1, matformer_hidden, n_hidden - matformer_hidden);

        let gate_tail_max = gate_tail.abs().max().double_value(&[]);
        let up_tail_max = up_tail.abs().max().double_value(&[]);
        let down_tail_max = down_tail.abs().max().double_value(&[]);

        assert_eq!(gate_tail_max, 0.0);
        assert_eq!(up_tail_max, 0.0);
        assert_eq!(down_tail_max, 0.0);
    }

    #[test]
    #[should_panic(expected = "MatFormer helper mode is temporarily disabled")]
    fn matformer_mlp_helper_mode_has_helper_grads() {
        use crate::matformer_helper::HelperConfig;

        let vs = nn::VarStore::new(Device::Cpu);
        let n_embd = 4;
        let n_hidden = 16;
        let matformer_hidden = 4;
        let current_round = Arc::new(AtomicU64::new(0));
        let helper_config = HelperConfig::new(0.5, 16); // 50% helper

        let _ = Mlp::new(
            vs.root(),
            n_embd,
            n_hidden,
            Some(matformer_hidden),
            0,
            Some(helper_config),
            current_round,
            None,
        );
    }
}
