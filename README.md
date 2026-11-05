# aftermath
## earlier: psyche-elastic
This is a research fork of Nous Research's [**Psyche**](https://nousresearch.com/nous-psyche/), supported by a very generous compute grant from the same team, and an Emergent Ventures grant.

<p align="center" width="50%">
<img src="https://github.com/user-attachments/assets/87c70aab-07c0-4df6-ab4f-b7f23776e427" />
</p>

[Matformer: Nested Transformer for Elastic Inference](https://www.prateekjain.org/publications/all_papers/KuduguntaKDCDTHKFJ23.pdf), and subsequent Gemma-3n models show a way to develop _nested_ Transformers that can be trained collaboratively. 

This means that a smaller sub-model $A'$ can share weights with the full model $A$. $A'$ can help to train the part that they share. How much it helps depends on the model and on the size of $A'$ (refer to [What the evidence shows](https://plugyawn.github.io/aftermath/explain/matformer.html#what-the-evidence-shows)).
Note that $A'$ can still operate as a fully independent GPT-class model.

FFN slicing cannot make $A'$ arbitrarily small, because attention and embeddings stay on every client. On the checkpoint below, the smallest possible $A'$ is 55% of the full model.

This property supports a distributed training method. The method slices GPT models _width-wise_ (in case of FFNs). Devices with less VRAM hold smaller chunks, and larger devices hold the full model. The method gathers gradients from all devices. _Head-wise_ slicing of attention is a planned change. It is not in the code yet.
<img width="1018" height="647" alt="image" src="https://github.com/user-attachments/assets/01574699-0f18-4ce7-b41e-bf4f14432529" />
Highlighted in red above is a submodel on a "satellite", trained in tandem with a larger model.
Both the larger and the smaller submodel show lower `train/loss` than a model trained with the same token-budget/time-budget as each of the models independently. This figure shows training loss, not held-out loss, so it is early evidence only.
The smaller model occupies only ~70% of the VRAM of the larger, and the difference grows with size. Training used a small slice of FineWeb. The checkpoint name says 124M, but `plugyawn/nanogpt-124m-init` has 190,460,160 parameters.

Here is a little infographic I made about my dreams with `Aftermath`:
<img width="1215" height="681" alt="image" src="https://github.com/user-attachments/assets/105b1744-e085-4fe5-a34a-47c8fd67bf81" />

This infographic shows a long-term goal. The code does not support phones. A 4 GB phone holds under 1% of a 70B model. A sub-model of 7% gave the full model no benefit in our small tests.


## About the code
Aftermath is **built on top of Psyche** (PsycheFoundation/psyche) and extends it with MatFormer tiered checkpoints, manifest-based slicing, and practical tooling for mixed‑hardware training.
For canonical Psyche documentation, see https://docs.psyche.network (upstream). 

This README adds the elastic workflow and fork‑specific details. For more extensive documentation, see the [Aftermath docs](https://plugyawn.github.io/aftermath/), including the [MatFormer training details](https://plugyawn.github.io/aftermath/explain/matformer.html). The source is in `psyche-book/`.

---
Aftermath offers, out of the box:

- **Support for Heterogeneous training**:
  - smaller devices train smaller tiers; large devices train full tiers; all contribute to shared weights.
  - tiered checkpoints pull only the files needed for a given tier (esp. from HF). A client that joins over P2P downloads full-width tensors from wider peers and keeps only its prefix.
  - schema canonicalization + double‑slicing protection detect mismatched clients. They do not verify the tier that a client declares.
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

- [x] MatFormer tier slices, manifests, and hub selective downloads
- [x] Metadata inference + schema canonicalization for mixed tiers
- [x] Heterogeneous gradient aggregation normalization
- [ ] Parameter tying, aggressive NanoGPT support (ongoing)
- [ ] Optimizer and kernel work (Muon, Polar Express) with tests (ongoing)
- [x] System metrics logging + fault injection tools
- [x] Packaging and local testnet improvements
- [ ] **Tensor parallelism with tier > 0** is not supported yet.
- [ ] **Helper mode** (suffix sampling) is disabled until sparse alignment/rotation is complete.
- [ ] Tier export currently assumes a **single `.safetensors` shard**.

