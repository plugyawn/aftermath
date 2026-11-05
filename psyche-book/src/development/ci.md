# CI

## Documentation site

The `Docs` GitHub Actions workflow (`.github/workflows/docs.yml`) builds this book for each pull request that changes it. A push to `main` also deploys the book to <https://plugyawn.github.io/aftermath/>.

## Upstream Psyche CI

The sections below come from upstream Psyche. The hostnames in them belong to upstream Psyche.

## Overview

We use [Garnix](https://garnix.io/) as our CI provider. It:

- Builds packages in our Nix flakes
- Runs all Nix checks including formatting, lints, & Rust tests.

## Deployment Branches

Some branches are configured for automatic deployment. These branches serve as dedicated testing environments.

### Development Environments

These environments are stateful and accessible via SSH for developer troubleshooting. Public keys are listed in this repo.

| Source Branch         | Purpose                      | Hostname                         |
| --------------------- | ---------------------------- | -------------------------------- |
| `test-deploy-devnet`  | Indexer/frontend for devnet  | `devnet-preview.psyche.network`  |
| `test-deploy-mainnet` | Indexer/frontend for mainnet | `mainnet-preview.psyche.network` |
| `test-deploy-docs`    | Preview docs                 | `docs.preview.psyche.network`    |

### Production Environment

`main` automatically deploys the website/indexer to https://mainnet.psyche.network/ and the docs to https://docs.psyche.network/.

This is a stateful deploy, but with no SSH access for security reasons.
