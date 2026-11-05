# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Psyche is a distributed training system for transformer-based AI models over the internet. It enables collaboration between untrusted parties to train ML models using two deployment architectures:

- **Centralized**: Server-coordinated training via `architectures/centralized/`
- **Decentralized**: Solana blockchain-coordinated training via `architectures/decentralized/`

## Build Commands

```bash
# Enter nix development shell (recommended)
nix develop

# Build all workspace crates
cargo build

# Build with release optimizations
cargo build --release

# Format and lint
nix fmt  # or: cargo clippy --fix --allow-staged --all-targets && cargo fmt

# Build Solana programs (requires anchor)
cd architectures/decentralized/solana-coordinator && anchor build
cd architectures/decentralized/solana-authorizer && anchor build
```

## Running Tests

```bash
# Run all tests
cargo test

# Run a single test
cargo test --release -p <package-name> -- --nocapture <test_name>

# Centralized integration tests
just integration-test                    # all tests
just integration-test <test_name>        # single test

# Decentralized integration tests
just decentralized-integration-tests                    # without Python
USE_PYTHON=1 just decentralized-integration-tests      # with Python

# Solana client tests
cargo test --package psyche-solana-client --features solana-localnet-tests
```

## Local Development

```bash
# Start a local centralized testnet (requires tmux)
just local-testnet

# Headless smoke test
cargo run -p psyche-centralized-local-testnet -- start --headless --headless-exit-after-secs 10 --num-clients 2 --config-path ./config/test --tui false

# Local Solana testnet setup
just setup-solana-localnet-test-run
just start-training-localnet-client

# Start local telemetry stack (Grafana/Prometheus/Loki)
docker compose -f telemetry/docker-compose.yml up
```

## Architecture

### Shared Crates (`shared/`)

Core functionality shared between centralized and decentralized architectures:

- **`psyche-core`**: Core types and primitives (batch IDs, hashes, config)
- **`psyche-coordinator`**: Coordinator state machine, committee selection, data assignment logic
- **`psyche-client`**: Training client implementation (model loading, training loop, checkpointing)
- **`psyche-modeling`**: PyTorch/tch-rs model implementations (LLaMA, attention, trainer)
- **`psyche-network`**: P2P networking via iroh (gossip, blob transfer)
- **`psyche-data-provider`**: Dataset loading (HuggingFace, local, parquet, GCS)
- **`psyche-eval`**: Evaluation utilities
- **`psyche-watcher`**: Run state observation abstractions
- **`psyche-tui`**: Terminal UI components
- **`psyche-metrics`**: OpenTelemetry metrics collection

### Centralized (`architectures/centralized/`)

- **`server`**: Coordinates rounds, assigns batches, collects witnesses
- **`client`**: Trains batches, submits gradients/witnesses
- **`local-testnet`**: Automated multi-client testing with tmux

### Decentralized (`architectures/decentralized/`)

Solana-based coordination:
- **`solana-coordinator`**: Anchor program for run coordination
- **`solana-authorizer`**: Access control program
- **`solana-treasurer`**: Token/reward management
- **`solana-client`**: Client for Solana-coordinated training

### Python Integration (`python/`)

PyO3-based Python bindings for using Python models within the Rust training loop. Enable with `--features python`.

## Configuration

Training runs are configured via TOML files in `config/`:
- `state.toml`: Run configuration (batch sizes, round timing, model config)
- Model configs specify: architecture, data location, optimizer, LR schedule

Example minimal config structure:
```toml
run_id = "test"
run_state = "WaitingForMembers"
[config]
min_clients = 2
global_batch_size_start = 4
# ...
[model.LLM]
architecture = "HfLlama"
# ...
```

## Key Features

- **MatFormer Support**: Heterogeneous training with different FFN widths per client (`--matformer-tier`)
- **Device Support**: CUDA, MPS (macOS Metal), CPU via `--device`
- **Data Parallelism**: Multi-GPU via NCCL (`--features parallelism`)
- **Checkpointing**: HuggingFace Hub integration for model saving/loading

## MatFormer Heterogeneous Training Status

**CURRENT STATUS: RESEARCH** (helper mode disabled for tier > 0)

Mixed-tier training runs in the tests. It has no evidence at real scale yet. For the limits, see `psyche-book/src/explain/matformer.md`.

### How It Works

Each client trains on **different, non-overlapping batches** within a round:
- Tier-0 clients: Train with full FFN width, contribute gradients for all neurons
- Tier-1 clients: Train with half FFN width, contribute gradients for prefix neurons only

This creates **sample rate asymmetry**:
- Prefix neurons: See gradients from ALL tiers (higher effective sample rate)
- Suffix neurons: See gradients from tier-0 only (lower effective sample rate)

### What Holds

1. **Clients send only the width that they train** - A narrow tier never sends the tail, so it cannot change the tail update (`shared/modeling/tests/matformer_distro.rs`)
2. **All clients decode in Float** - The vote does not depend on device or weight precision
3. **MPS weights stay on the BF16 grid** - MPS clients track BF16 clients
4. **The lookahead returns exactly** - Clients with different local errors keep the same consensus weights, and a cancelled step removes its lookahead
5. **Witnesses verify blob hashes** - Commitment verification uses downloaded blob's xshape, not local model

### What Does Not Hold

- Sign-SGD does not remove magnitude in raw mode. The sum uses peer magnitudes before the sign. Use `quantize_1bit = true` for equal votes.
- Backprop is separable for one pass only. The tail output changes gradients on all shared weights.
- AdamW clients do not exchange gradients. Use the Distro optimizer for shared training.

### Recommended Configuration

```bash
# Heterogeneous training (WORKS)
--client-matformer-tiers 0,1,1 \
--client-matformer-helper-fractions 0,0,0  # Required: disable helper mode for tier > 0
```

### Startup Summary

On model load, a MatFormer configuration summary is printed:
```
╔══════════════════════════════════════════════════════════════════╗
║                      MATFORMER CONFIGURATION                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Checkpoint:       ./checkpoints/nanogpt-20m-init-tier1          ║
║  Load strategy:    Auto                                          ║
║  Sliced checkpoint: Yes                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  CLI tier:         1                                             ║
║  Effective tier:   0 ≠                                           ║
║  Helper mode:      Auto-disabled (sliced checkpoint)             ║
╠══════════════════════════════════════════════════════════════════╣
║  FFN width:        512 / 512 (100%)                              ║
╚══════════════════════════════════════════════════════════════════╝
```

Includes contextual warnings for tier mismatches and helper mode configuration.

### Checkpoint Tier Management

Checkpoints now store MatFormer tier metadata:
- `matformer_tier`: The effective tier used during training
- `matformer_base_intermediate_size`: Original FFN width before slicing

This enables:
- **Auto-inference**: Tier can be inferred from stored metadata or size ratio
- **Double-slicing protection**: Error if attempting to further slice a sliced checkpoint with `--matformer-load-strategy universal`

### Known Limitations

| Issue | Status | Mitigation |
|-------|--------|------------|
| Helper mode + sliced checkpoint | Auto-disabled | Code correctly detects and disables |
| Double-slicing | Hard error | `validate_no_double_slicing()` prevents misconfiguration |
| Memory savings (small models) | ~15%, not 50% | Embeddings dominate; larger models save more |
| Sliced client uploads | Blocked | A sliced client has no FFN tail, so it saves locally and does not publish |
| P2P join with no wider peer | Hard error | A client stops when no peer has the FFN width that it needs |
| Tiers | Self-declared | The coordinator does not store or verify a tier |

### Testing Heterogeneous Training

```bash
# Run heterogeneous training
DYLD_LIBRARY_PATH=./.venv/lib/python3.12/site-packages/torch/lib \
  cargo run --release -p psyche-centralized-local-testnet -- start \
    --headless --headless-exit-after-secs 60 --num-clients 3 \
    --config-path ./config/nanogpt-20m-run \
    --client-matformer-tiers 0,1,1 \
    --client-matformer-helper-fractions 0,0,0 \
    --tui false
```

See full technical analysis: `~/.claude/plans/enumerated-scribbling-graham.md`

## Cargo Configuration

Uses `tokio_unstable` feature flag (see `.cargo/config.toml`). The workspace uses a local cargo home (`.cargo-home/`) and target directories (`.cargo-target/`, `.cargo-target-tests/`).
