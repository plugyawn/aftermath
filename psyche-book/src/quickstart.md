# Quickstart

This page shows how to start a local testnet in which three clients train different tiers of one model.

## Prerequisites

- The Rust toolchain
- Python 3.11 or 3.12, with PyTorch installed
- `tmux` (optional, for the local testnet UI)

## Set the environment

Use this script to point `tch-rs` to your Python installation of PyTorch:

```bash
source scripts/psyche-env.sh
```

## Start the local testnet

```bash
just local-testnet \
  --num-clients 3 \
  --config-path ./config/consilience-match-llama2-20m-fineweb-pretrain-dev/ \
  --client-matformer-tiers 0,1,2
```

If you do not have `just`, use this command:

```bash
cargo run -p psyche-centralized-local-testnet -- start \
  --num-clients 3 \
  --config-path ./config/consilience-match-llama2-20m-fineweb-pretrain-dev/ \
  --client-matformer-tiers 0,1,2
```

Each value in `--client-matformer-tiers` sets the tier of one client. Tier 0 is the full model. Tier 1 uses 1/2 of the FFN width, and tier 2 uses 1/4 of the FFN width.

## Next steps

- To export tiered checkpoints, refer to [Tiered checkpoints and manifests](./explain/matformer.md#tiered-checkpoints-and-manifests).
- For all local testnet options, refer to [Running off-chain](./development/running-offchain.md).
