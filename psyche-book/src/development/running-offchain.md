# Running Psyche offchain

When developing for Psyche, you might not want to spin up all the Solana infrastructure if you're working on a feature like the distributed networking or the training code.

To that end, we maintain a "centralized" client & server package that simply communicate over TCP instead of dealing with code deployed to a Solana network.

There's a `server` package, and a `client` package.
To develop with them, you'd spin up one `server` with whatever [run config](../enduser/run-config.md) you want

## Local Testnet

The local testnet is a helper application designed to easily spin up a Server and multiple clients.
It's useful for doing sample runs on your own hardware, and for development.

### Pre-requisites

Since we want to run many clients and the server we'll need several terminal windows to monitor them. The tool uses [tmux](https://github.com/tmux/tmux/wiki/Installing) to create them.

> If you're using the Nix devShell, tmux is already included.

### Running

A sample invocation that fires up 3 clients to train on a 20m model might look like this:

```bash
just local-testnet \
    --num-clients 3 \
    --config-path ./config/consilience-match-llama2-20m-fineweb-pretrain-dev/
```

### Heterogeneous Training with MatFormer

You can run clients at different [MatFormer](../explain/matformer.md) tiers to simulate heterogeneous hardware:

```bash
just local-testnet \
    --num-clients 3 \
    --config-path ./config/my-config/ \
    --client-matformer-tiers 0,1,2
```

This assigns:
- Client 1: tier 0 (full FFN width)
- Client 2: tier 1 (half FFN width)
- Client 3: tier 2 (quarter FFN width)

The tiers cycle if you have more clients than tier values specified.

### Headless Mode

For CI/automated testing, use headless mode which doesn't require tmux:

```bash
cargo run -p psyche-centralized-local-testnet -- start \
    --headless \
    --headless-exit-after-secs 60 \
    --num-clients 2 \
    --config-path ./config/test \
    --tui false
```

There's a _lot_ of options to configure the local testnet. Check em out below!

<details>
    <summary>Command-line options</summary>

{{#include ../../generated/cli/psyche-centralized-local-testnet.md}}

</details>

## Server & Client

Both of these applications can be spun up individually at your discretion instead of using the local testnet. We include all their command-line options for your reading pleasure:

<details>
    <summary>Client</summary>

{{#include ../../generated/cli/psyche-centralized-client.md}}

</details>

<details>
    <summary>Server</summary>

{{#include ../../generated/cli/psyche-centralized-server.md}}

</details>
