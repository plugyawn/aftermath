# Aftermath

Aftermath trains one transformer model across computers that have different amounts of VRAM. Small devices train small tiers of the model. Large devices train the full model. All devices contribute to one set of shared weights.

> **Aftermath is a research fork of [Psyche](https://nousresearch.com/nous-psyche/) by Nous Research.** This site documents the fork. Most pages come from the Psyche documentation, so they use the name "Psyche". For the upstream documentation, go to <https://docs.psyche.network>.

## What Aftermath adds to Psyche

- **Heterogeneous training.** Each client trains the [MatFormer](./explain/matformer.md) tier that fits its hardware.
- **Tiered checkpoints.** A client that loads from a checkpoint repository downloads only the files for its tier. A client that joins over P2P downloads full-width tensors from wider peers and keeps only its prefix.
- **Manifests and metadata.** The loader uses these to find the tier and to validate the checkpoint.
- **Schema canonicalization and double-slicing protection.** These checks detect mismatched clients. They do not verify the tier that a client declares.
- **Heterogeneous gradient aggregation.** Aggregation aligns the results of all tiers to one shape, then applies the sign of the sum.
- **Tools.** The fork adds system metrics logging, fault injection tools, and improvements to packaging and the local testnet.

Some features have limits. For the list, refer to [Current Limitations](./explain/matformer.md#current-limitations).

## Where to start

| Your goal                                 | Page                                              |
| ----------------------------------------- | ------------------------------------------------- |
| Run a local testnet with mixed tiers      | [Quickstart](./quickstart.md)                     |
| Learn how nested models train together    | [MatFormer](./explain/matformer.md)               |
| Export tiered checkpoints                 | [Tiered checkpoints and manifests](./explain/matformer.md#tiered-checkpoints-and-manifests) |
| Learn how the training protocol operates  | [Psyche in depth](./explain/index.md)             |
| Set up a development environment          | [Setup & Useful Commands](./development/setup.md) |

## About Psyche

<p align="center" width="100%">
    <img src="./psyche.jpg" alt="Psyche">
</p>

Psyche is a system that enables distributed training of transformer-based AI models over the internet, aiming to foster collaboration between untrusted parties to create state-of-the-art machine learning models.
It leverages a peer-to-peer distributed network for communication and data sharing.

### Introduction to Psyche

<p align="center" width="100%">
    <a href="https://www.youtube.com/watch?v=XMWI3nDk48c">
        <img src="./psyche_youtube.png" alt="Introduction to Psyche video">
    </a>
</p>

### How does it work?

At its core, Psyche is a protocol that coordinates multiple independent clients to train a single machine learning model together. Rather than running on a centralized server farm with high-speed interconnects between every accelerator (GPUs, usually), Psyche distributes the training workload across many independent computers, each contributing a small piece to the overall training process.

Psyche is built to maintain training integrity without requiring participants to trust each other. Through a combination of consensus mechanisms, game theory, and careful protocol design, Psyche will ensure that the trained model remains coherent and consistent despite being trained across disparate machines.
