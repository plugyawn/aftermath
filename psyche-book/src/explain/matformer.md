# MatFormer: Elastic Inference with Nested Models

MatFormer (Matryoshka Transformer) is a technique that enables training a single transformer model that supports multiple compute/quality trade-offs at inference time. Psyche implements MatFormer to allow heterogeneous clients with different hardware capabilities to participate in the same training run.

## Overview

Traditional transformers require running the full model for every inference. MatFormer introduces **nested submodels** where smaller models are strict prefixes of larger ones, sharing weights. This enables:

- **Heterogeneous training**: Clients with different VRAM/compute can train at different capacity levels
- **Elastic inference**: Deploy the same weights at multiple latency/quality points
- **Behavioral consistency**: Smaller models closely match larger model outputs due to shared weights

## How It Works

MatFormer primarily targets the **FFN (Feed-Forward Network)** blocks, which typically account for ~2/3 of transformer parameters. The technique works by:

1. **Prefix slicing**: Smaller models use the first N neurons of the FFN hidden dimension
2. **Shared weights**: The first N neurons are trained by ALL model sizes, making them "universally useful"
3. **Exponential tiers**: Widths are typically h, h/2, h/4, h/8 (powers of 2)

### Weight Structure

For a standard FFN with hidden size `h`:

```
FFN(x) = W_down · activation(W_up · x)

Where:
- W_up:   [hidden_size, embed_dim]  (projects up to hidden)
- W_down: [embed_dim, hidden_size]  (projects back down)
```

MatFormer at tier `t` uses only the first `h / 2^t` neurons:

```
FFN_tier(x) = W_down[:, :h_t] · activation(W_up[:h_t, :] · x)
```

### Tier System

| Tier | FFN Width | Use Case |
|------|-----------|----------|
| 0 | 100% (full) | Maximum quality, high-VRAM GPUs |
| 1 | 50% | Balanced quality/speed |
| 2 | 25% | Lower VRAM devices |
| 3 | 12.5% | Memory-constrained devices |

## Using MatFormer in Psyche

### Client Configuration

Clients specify their MatFormer tier via the `--matformer-tier` flag:

```bash
psyche-centralized-client train \
  --run-id "my-run" \
  --server-addr "server:8080" \
  --matformer-tier 1 \
  --device cuda
```

### End-to-End MatFormer Workflow

This is the full lifecycle used in Psyche today:

1. **Start from a universal checkpoint** (tier 0, full FFN width).
2. **Export tier slices** using `scripts/export_matformer_tiers.py`. This writes
   tier directories and `matformer_manifest.json`.
3. **Publish to local storage or Hugging Face**. The manifest is used to
   enumerate files for each tier.
4. **Clients join with a tier and load strategy**:
   - `auto` tries to load a tier slice via the manifest and falls back to
     universal if missing.
   - `sliced` requires the tier slice and fails fast if missing.
   - `universal` always loads the full checkpoint.
5. **Metadata is enforced or inferred**:
   `matformer_tier` and `matformer_base_intermediate_size` are injected
   (or inferred) so loaders and schema hashes remain consistent.
6. **Heterogeneous training proceeds**:
   clients at smaller tiers update the shared prefix weights. Each client sends
   only the width that it trains. Aggregation aligns all results to the local
   shape, then applies the sign of the sum.

This workflow makes it possible to mix large and small devices in the same run
without requiring every client to download or compute the full model.

### Tiered Checkpoints and Manifests

MatFormer supports **tier-sliced checkpoints** so you can distribute and load only the
subset of weights needed for a given tier. This is especially useful for
heterogeneous training when lower-VRAM devices should download smaller slices.

#### Exporting Tier Slices

Use the export script to generate tier directories and a manifest:

```bash
python scripts/export_matformer_tiers.py --src checkpoints/my-model --tiers 1 2
```

This produces:

- `checkpoints/my-model-tier1/` and `checkpoints/my-model-tier2/` (sliced weights)
- `matformer_manifest.json` in the **universal** checkpoint directory
- Updated `config.json` in each tier directory:
  - `matformer_tier` = the tier number
  - `matformer_base_intermediate_size` = the original full FFN width
  - `intermediate_size` = the sliced width for the tier

The export script validates that base FFN size is divisible by `2**tier` and only
supports a single `.safetensors` shard (for now).

#### Manifest (Schema v1)

`matformer_manifest.json` provides a machine-readable list of files for each tier:

```json
{
  "schema_version": 1,
  "matformer_base_intermediate_size": 1024,
  "common_files": ["tokenizer.json", "tokenizer_config.json"],
  "tiers": [
    {
      "tier": 1,
      "intermediate_size": 512,
      "files": ["../my-model-tier1/config.json", "../my-model-tier1/model.safetensors"]
    }
  ],
  "sha256": { "...": "..." }
}
```

Paths are **relative to the manifest location**. On HF, the client normalizes
these paths to repo-root and rejects absolute paths; `../` is supported for
layouts where tier slices are siblings of the manifest directory.

#### Load Strategy

Use `--matformer-load-strategy` to control how slices are resolved:

- `auto` (default): use a sliced tier if the manifest is complete; otherwise
  fall back to the universal checkpoint.
- `sliced`: require the tier slice; fail fast if missing.
- `universal`: always load the full checkpoint.

The loader detects when a checkpoint is already sliced (via manifest or
`matformer_tier` metadata) and prevents **double-slicing** by forcing the
effective tier to 0 in `auto`/`sliced` modes.

#### Metadata and Schema Compatibility

Tier-sliced checkpoints include:

- `matformer_tier`: the slice tier (0 = full, 1 = half, 2 = quarter, etc.)
- `matformer_base_intermediate_size`: the original full FFN width

If these fields are missing, the client infers them from the manifest or
the selected tier. This metadata is also used to **canonicalize** the schema
hash so heterogeneous runs (mixing tier 0 and tiered checkpoints) agree on
model compatibility.

### What This Unlocks

- **Heterogeneous training at scale**: smaller nodes can participate using
  tier slices without downloading or storing the full model.
- **Checked tier mixing**: schema canonicalization and double-slicing protection
  detect mismatched checkpoints in the same run. They do not verify the tier
  that a client declares.
- **Faster joins and lower bandwidth**: manifest-based loading pulls only
  the needed files for each tier.
- **Operational clarity**: the manifest provides an explicit source of truth
  for which files define a tier, with optional hashes for integrity.

### Heterogeneous Training Example

A training run with clients of varying capabilities:

```
┌─────────────────────────────────────────────────────────────┐
│                     Server                                  │
└─────────────────────────────────────────────────────────────┘
          ▲              ▲              ▲
          │              │              │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  RTX 4090   │  │  RTX 3060   │  │  M1 Mac     │
│  tier=0     │  │  tier=1     │  │  tier=2     │
│  (full)     │  │  (half)     │  │  (quarter)  │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Local Testnet with Multiple Tiers

The local testnet supports heterogeneous tier assignment:

```bash
cargo run -p psyche-centralized-local-testnet -- start \
  --num-clients 3 \
  --client-matformer-tiers 0,1,2 \
  --config-path ./config/my-config
```

This assigns:
- Client 1: tier 0 (full width)
- Client 2: tier 1 (half width)
- Client 3: tier 2 (quarter width)

## Gradient Handling

When clients train at different tiers, gradients are computed only for the active prefix of weights:

- **Tier 0** clients compute gradients for ALL FFN weights
- **Tier 1** clients compute gradients for the first 50% of FFN neurons
- **Tier 2** clients compute gradients for the first 25% of FFN neurons

The shared prefix neurons receive gradient contributions from all clients, making them robust and general-purpose. Exclusive neurons (used only by larger tiers) are more specialized.

### Parameter Naming

Gradient distribution results now include parameter names to correctly aggregate gradients across heterogeneous clients:

```rust
pub struct SerializedDistroResult {
    pub parameter_name: String,  // Identifies which parameter
    pub sparse_idx: ...,
    pub sparse_val: ...,
}
```

## Implementation Details

### LLaMA FFN Implementation

The MatFormer-enabled FFN in `shared/modeling/src/models/llama.rs`:

```rust
impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let Some(matformer_hidden_size) = self.matformer_hidden_size else {
            // Full model path
            return self.down_proj.forward(
                &(self.gate_proj.forward(xs).silu() * self.up_proj.forward(xs)),
            );
        };

        // MatFormer path: use prefix slices via .narrow()
        let gate_w = self.gate_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
        let up_w = self.up_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
        let down_w = self.down_proj.linear.ws.narrow(1, 0, matformer_hidden_size);

        let gate = xs.matmul(&gate_w.transpose(0, 1));
        let up = xs.matmul(&up_w.transpose(0, 1));
        let hidden = gate.silu() * up;
        hidden.matmul(&down_w.transpose(0, 1))
    }
}
```

Key implementation notes:
- `.narrow()` creates views (not copies), preserving gradient flow
- Tier is set at model load time via `from_pretrained_with_matformer_tier()`
- Tensor parallelism is not yet supported with MatFormer tiers > 0

### Client Capabilities Protocol

Clients advertise their tier in the capabilities handshake:

```rust
pub struct ClientCapabilities {
    pub device: String,        // "CUDA(0)", "MPS", "CPU"
    pub matformer_tier: u8,    // Requested tier
}

pub struct TrainingAssignment {
    pub matformer_tier: u8,    // Server-assigned tier
}
```

## Verification

A unit test verifies correct gradient isolation:

```rust
#[test]
fn matformer_mlp_has_zero_tail_grads() {
    // Verifies that exclusive weights (tail) have zero gradient
    // when training at a smaller tier
}
```

Run the test:

```bash
cargo test -p psyche-modeling matformer_mlp_has_zero_tail_grads
```

## Current Limitations

1. **Single tier per client**: Each client trains at one tier. The MatFormer paper samples one tier per step, so the mix of clients in a run acts as the sampling distribution
2. **No tensor parallelism**: MatFormer tiers > 0 require disabling tensor parallelism
3. **No Mix'n'Match**: Per-layer width schedules are not yet implemented
4. **Attention unchanged**: Only FFN width is elastic; attention heads remain fixed
5. **Helper mode disabled**: Stochastic suffix sampling ("helper mode") is wired through the CLI
   but currently hard-disabled until sparse alignment and rotation are finalized
6. **Memory floor**: Attention and embeddings stay on every client. FFN slicing cannot go below this floor. On a checkpoint of 190M parameters the floor is 55% of the full model. On a 7B model it is 36%
7. **Only full-width clients publish checkpoints**: A client with sliced weights has no FFN tail, so it saves checkpoints locally but does not upload them. If all full-width clients leave a run, the FFN tail stops training. A client that needs the tail then cannot join over P2P, and it stops with an error
8. **P2P joins download full-width tensors**: A narrow client that receives a tensor from a wider peer keeps only its prefix. The bandwidth does not fall
9. **Tiers are self-declared**: The coordinator does not store or verify the tier of a client. Each client receives the same batch count, whatever its tier
10. **BF16 weights can lose small updates**: A step has no effect when `lr` is below about 1/256 of the weight magnitude. The `bf16_lost_update_fraction` statistic shows the amount for each parameter. Use `--optim-stats-steps` to log it

## What the Evidence Shows

These results come from one small character model with 209k parameters, 1000 rounds and two seeds. Seed noise was about 0.03 in validation loss, so treat them as direction only.

| Clients | Small-client parameters | Full model | Small model | Small model alone |
| --- | --- | --- | --- | --- |
| 8 big | n/a | 1.824 | n/a | n/a |
| 2 big | n/a | 2.100 | n/a | n/a |
| 2 big + 6 FFN/4 | 47% | 1.880 | 1.921 | 1.966 |
| 2 big + 6 `d_model`/2 | 26% | 2.043 | 2.116 | 2.145 |
| 2 big + 6 `d_model`/4 | 7% | 2.110 | 2.334 | 2.353 |

The theory has these limits:

- **The theory states no objective.** At low signal-to-noise, the mean vote on one coordinate is proportional to the sum of `n_t / sigma_t` times the gradient of tier `t`. These weights change per coordinate, so this field is generally not the gradient of one loss.
- **Sign-SGD does not remove magnitude in raw mode.** The sum uses peer magnitudes before the sign. A peer with gradients 10 times larger controlled 97% of prefix elements in a test. Only `quantize_1bit = true` gives each peer an equal vote.
- **Backprop is separable for one pass only.** The tail output changes the gradient on all shared weights, so the tail couples the two objectives.
- **Same-batch cosine does not show that a sub-model is an accurate estimator.** Clients train on different batches, so they do not share that sampling noise. Use the ratio of progress toward the full gradient, measured on independent batches. In the toy runs this ratio fell from about 0.8 to between 0.35 and 0.62.
- **Small clients help only above a threshold.** With `r` small clients per big client, they help where that ratio exceeds `(sqrt(1 + r) - 1) / r`. This is 0.33 for `r = 3`.
- **Sign agreement adds no information.** It follows from cosine by `1/2 + asin(cos)/pi`.
- **The DCT has no neuron order to use.** On FFN gradients, top-k energy was 0.13 with the DCT, 0.14 with a random basis and 0.19 with no transform.
- **The DeMo convergence theorem does not cover this setup.** It takes the sign after averaging, not 1-bit coefficients.

## References

- [MatFormer: Nested Transformer for Elastic Inference](https://arxiv.org/abs/2310.07707) - Original paper
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) - Related work on nested representations
