# aftermath
## earlier: psyche-elastic
This is a research fork of Nous Research's [**Psyche**](https://nousresearch.com/nous-psyche/), supported by a very generous compute grant from the same team, and an Emergent Ventures grant.

<p align="center" width="50%">
<img src="https://github.com/user-attachments/assets/87c70aab-07c0-4df6-ab4f-b7f23776e427" />
</p>

[Matformer: Nested Transformer for Elastic Inference](https://www.prateekjain.org/publications/all_papers/KuduguntaKDCDTHKFJ23.pdf), and subsequent Gemma-3n models show a way to develop _nested_ Transformers that can be trained collaboratively. 

This means that, even though $A'$ only $\~40$% of the size of the full model $A$ (where $A$ is the full model), it is a good gradient estimator for the full model (over the shared $40$%).
Note that $A'$ can still operate as a fully independent GPT-class model.

This property can be used to develop a distributed training method that slices GPT models _width-wise_ (in case of FFNs) and _head-wise_ (in case of attention), distribute smaller chunks over devices with less VRAM, while larger devices hold the full model, and intelligently gather gradients. This also produces strong small models that are verifiably more powerful than if they were distilled from the larger model later for an equivalent budget (effectively for free, considering devices with low VRAM can otherwise simply not join the training run). 
<img width="1018" height="647" alt="image" src="https://github.com/user-attachments/assets/01574699-0f18-4ce7-b41e-bf4f14432529" />
Highlighted in red above is a submodel on a "satellite", trained in tandem with a larger model.
Both the larger and the smaller submodel outperform a model trained with the same token-budget/time-budget as each of the models independently.
The smaller model occupies only ~70% of the VRAM of the larger, and the difference grows with size. These models are of the ~124M parameter range, trained on a small slice of FineWeb.


## About the code
Aftermath is **built on top of Psyche** (PsycheFoundation/psyche) and extends it with MatFormer tiered checkpoints, manifest-based slicing, and practical tooling for mixed‑hardware training.
For canonical Psyche documentation, see https://docs.psyche.network (upstream). 

This README adds the elastic workflow and fork‑specific details. For more extensive documentation, please host `psyche-book`, available on this repo, and refer to the new Matformer training details.

---
Aftermath offers, out of the box:

- **Support for Heterogeneous training**:
  - smaller devices train smaller tiers; large devices train full tiers; all contribute to shared weights.
  - tiered checkpoints pull only the files needed for a given tier (esp. from HF).
  - schema canonicalization + double‑slicing protection prevents mismatched runs.
  - explicit manifests + metadata give reliable tier detection and validation.

This is possibly through the implementation of a Matryoshka Transformer. Some details, as you look through the code/documentation:
- Tier slices are **prefixes** of the FFN width: tier 1 = 1/2, tier 2 = 1/4, etc.
- Clients at smaller tiers update only prefix weights; shared prefixes receive gradients from all tiers.

---

## Quickstart Guide

### Prereqs
- Rust toolchain
- Python (3.11/3.12 recommended) with **PyTorch** installed
- `tmux` (optional, for local testnet UI)

### Environment

Use the repo helper to point `tch-rs` to your Python torch install:

```bash
source scripts/psyche-env.sh
```

### Local Testnet (fastest path)

```bash
just local-testnet \
  --num-clients 3 \
  --config-path ./config/consilience-match-llama2-20m-fineweb-pretrain-dev/ \
  --client-matformer-tiers 0,1,2
```

If you don’t have `just`, use:

```bash
cargo run -p psyche-centralized-local-testnet -- start \
  --num-clients 3 \
  --config-path ./config/consilience-match-llama2-20m-fineweb-pretrain-dev/ \
  --client-matformer-tiers 0,1,2
```

---

## Tiered Checkpoints & Manifests

### Export Tier Slices

```bash
python scripts/export_matformer_tiers.py --src checkpoints/my-model --tiers 1 2
```

Produces:
- `checkpoints/my-model-tier1/`, `-tier2/` (sliced weights)
- `matformer_manifest.json` in the **universal** directory
- Tier configs with:
  - `matformer_tier`
  - `matformer_base_intermediate_size`
  - `intermediate_size` (sliced width)

### Manifest (Schema v1)

```json
{
  "schema_version": 1,
  "matformer_base_intermediate_size": 1024,
  "common_files": ["tokenizer.json"],
  "tiers": [
    {"tier": 1, "intermediate_size": 512, "files": ["../my-model-tier1/config.json", "../my-model-tier1/model.safetensors"]}
  ],
  "sha256": {"...": "..."}
}
```

Paths are **relative to the manifest location**; the loader normalizes for HF and rejects absolute paths.

### Load Strategy

- `auto` (default): use sliced if complete, otherwise fall back to universal.
- `sliced`: require the tier slice; fail fast if missing.
- `universal`: always load full checkpoint.

The loader prevents **double‑slicing** if a checkpoint is already tiered.

---

## Summary of additions over Psyche

[x] MatFormer tier slices, manifests, and hub selective downloads
[x] Metadata inference + schema canonicalization for mixed tiers
[x] Heterogeneous gradient aggregation normalization
[ ] Parameter tying, aggressive NanoGPT support (ongoing)
[ ] Optimizer and kernel work (Muon, Polar Express) with tests (ongoing)
[x] System metrics logging + fault injection tools
[x] Packaging and local testnet improvements
[ ] **Tensor parallelism with tier > 0** is not supported yet.
[ ] **Helper mode** (suffix sampling) is disabled until sparse alignment/rotation is complete.
[ ] Tier export currently assumes a **single `.safetensors` shard**.

