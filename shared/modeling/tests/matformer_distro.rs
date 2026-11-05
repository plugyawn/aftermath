//! Runs mixed MatFormer tiers through the DisTrO optimizer on a real Llama model.
//!
//! Three clients share one run:
//! - `full`: tier 0 with full weights.
//! - `universal`: tier 1 that holds full weights and trains the prefix.
//! - `sliced`: tier 1 that holds only the prefix weights.

use psyche_modeling::{
    CausalLM, Distro, DistroResult, LlamaConfig, LlamaForCausalLM, ModelConfig, PretrainedSource,
    Trainer, matformer_prefix_dim,
};
use std::{collections::HashMap, sync::Arc};
use tch::{Device, Kind, Tensor};

const HIDDEN: i64 = 32;
const FULL_FFN: i64 = 64;
const NARROW_FFN: i64 = 32;
const VOCAB: i64 = 64;
const CHUNK: i64 = 8;
const TOPK: i64 = 4;
const LR: f64 = 1e-3;

/// Deterministic numbers, because tests run in parallel and the torch generator is global.
struct Numbers(u64);

impl Numbers {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// Uniform value in [-0.1, 0.1).
    fn weight(&mut self) -> f32 {
        ((self.next() % 20_000) as f32 - 10_000.0) * 1e-5
    }

    fn tensor(&mut self, shape: &[i64]) -> Tensor {
        let values: Vec<f32> = (0..shape.iter().product::<i64>())
            .map(|_| self.weight())
            .collect();
        Tensor::from_slice(&values).view(shape)
    }

    fn tokens(&mut self, vocab: i64) -> Tensor {
        let values: Vec<i64> = (0..24).map(|_| self.next() as i64 % vocab).collect();
        Tensor::from_slice(&values).view([2, 12])
    }
}

fn config(intermediate_size: i64) -> LlamaConfig {
    let mut config = LlamaConfig::dummy();
    config.hidden_size = HIDDEN as usize;
    config.intermediate_size = intermediate_size as usize;
    config.vocab_size = VOCAB as usize;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 2;
    config.num_key_value_heads = Some(2);
    config.max_position_embeddings = 32;
    config
}

fn shape_of(name: &str, ffn: i64) -> Vec<i64> {
    if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
        vec![ffn, HIDDEN]
    } else if name.ends_with("down_proj.weight") {
        vec![HIDDEN, ffn]
    } else if name.contains("norm") {
        vec![HIDDEN]
    } else if name.contains("embed_tokens") || name.contains("lm_head") {
        vec![VOCAB, HIDDEN]
    } else {
        vec![HIDDEN, HIDDEN]
    }
}

/// Initial weights of the full model. All clients start from these weights.
fn initial_weights() -> HashMap<String, Tensor> {
    let mut numbers = Numbers(42);
    let mut names = config(FULL_FFN).get_parameter_names();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let tensor = numbers.tensor(&shape_of(&name, FULL_FFN));
            (name, tensor)
        })
        .collect()
}

fn narrowed(weights: &HashMap<String, Tensor>, ffn: i64) -> HashMap<String, Tensor> {
    weights
        .iter()
        .map(|(name, tensor)| {
            let tensor = match matformer_prefix_dim(name) {
                Some(dim) => tensor.narrow(dim as i64, 0, ffn).copy(),
                None => tensor.copy(),
            };
            (name.clone(), tensor)
        })
        .collect()
}

struct Client {
    model: LlamaForCausalLM,
    distro: Distro,
    previous: Vec<Vec<DistroResult>>,
}

impl Client {
    fn new(
        weights: HashMap<String, Tensor>,
        intermediate_size: i64,
        tier: u8,
        kind: Kind,
        device: Device,
    ) -> Self {
        let source =
            PretrainedSource::ConfigAndTensors(config(intermediate_size), Arc::new(weights));
        let model = LlamaForCausalLM::from_pretrained_with_matformer_tier(
            &source,
            Some(kind),
            None,
            Some(device),
            None,
            None,
            tier,
        )
        .unwrap();
        model.prepare_for_training();
        let distro = Distro::new(&model, 0.999, CHUNK, TOPK, 0.0);
        Self {
            model,
            distro,
            previous: Vec::new(),
        }
    }

    /// Runs one training step and returns the local result and the loss.
    fn train(&mut self, batch: &Tensor, lr: f64) -> (Vec<DistroResult>, f64) {
        for var in self.model.variables() {
            var.zero_grad();
        }
        let device = self.model.device();
        let batch = batch.to_device(device);
        let (_, loss) = self
            .model
            .forward(&batch, Some(&batch), None, None, None, None);
        let loss = loss.unwrap();
        loss.backward();
        let results = self.distro.generate(&self.model, &self.previous, lr, false);
        self.previous = vec![results.clone()];
        (results, loss.double_value(&[]))
    }

    fn apply(&mut self, results: &[Vec<DistroResult>], lr: f64) {
        self.distro.apply(&self.model, results, lr);
    }

    fn weight(&self, name: &str) -> Tensor {
        self.model
            .variables()
            .find(|var| var.name() == name)
            .unwrap()
            .logical_tensor()
            .detach()
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
    }

    fn weight_names(&self) -> Vec<String> {
        self.model
            .variables()
            .map(|var| var.name().to_string())
            .collect()
    }
}

fn three_clients(kind: Kind) -> (Client, Client, Client) {
    let weights = initial_weights();
    (
        Client::new(narrowed(&weights, FULL_FFN), FULL_FFN, 0, kind, Device::Cpu),
        Client::new(narrowed(&weights, FULL_FFN), FULL_FFN, 1, kind, Device::Cpu),
        Client::new(
            narrowed(&weights, NARROW_FFN),
            NARROW_FFN,
            0,
            kind,
            Device::Cpu,
        ),
    )
}

fn batch(seed: u64) -> Tensor {
    Numbers(seed).tokens(VOCAB)
}

fn on_wire(results: &[DistroResult], quantize: bool) -> Vec<DistroResult> {
    if quantize {
        Trainer::quantize_results(&results.to_vec())
    } else {
        results.to_vec()
    }
}

/// Fraction of elements that differ between two tensors of one shape.
fn differing(a: &Tensor, b: &Tensor) -> f64 {
    a.ne_tensor(b)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[])
}

fn max_abs_difference(a: &Tensor, b: &Tensor) -> f64 {
    (a - b).abs().max().double_value(&[])
}

#[test]
fn universal_and_sliced_tiers_send_the_same_result() {
    for quantize in [false, true] {
        let (_, mut universal, mut sliced) = three_clients(Kind::Float);
        let names = universal.weight_names();
        for round in 0..3 {
            let data = batch(100 + round as u64);
            let (u, _) = universal.train(&data, LR);
            let (s, _) = sliced.train(&data, LR);
            let u = on_wire(&u, quantize);
            let s = on_wire(&s, quantize);
            assert_eq!(u.len(), s.len());
            for ((name, a), b) in names.iter().zip(u.iter()).zip(s.iter()) {
                assert_eq!(a.xshape, b.xshape, "{name}: wire shape differs");
                assert_eq!(a.totalk, b.totalk, "{name}: totalk differs");
                if !quantize {
                    // The zero-coefficient draw in 1-bit mode is random by design.
                    assert!(a.sparse_idx.equal(&b.sparse_idx), "{name}: indices differ");
                    assert!(
                        max_abs_difference(&a.sparse_val, &b.sparse_val) < 1e-5,
                        "{name}"
                    );
                }
            }
            let all = vec![u, s];
            universal.apply(&all, LR);
            sliced.apply(&all, LR);
            universal.distro.error_correction(&universal.model, LR);
            sliced.distro.error_correction(&sliced.model, LR);
        }
    }
}

/// This is the fault that the audit measured: narrow universal clients changed
/// the tail update on about 19% of elements in 1-bit mode.
#[test]
fn narrow_tiers_do_not_change_the_tail_update() {
    for quantize in [false, true] {
        let (mut full, mut universal, mut sliced) = three_clients(Kind::Float);
        let weights = initial_weights();
        let mut full_alone = Client::new(
            narrowed(&weights, FULL_FFN),
            FULL_FFN,
            0,
            Kind::Float,
            Device::Cpu,
        );

        let (f, _) = full.train(&batch(1), LR);
        let (u, _) = universal.train(&batch(2), LR);
        let (s, _) = sliced.train(&batch(3), LR);
        let f = on_wire(&f, quantize);
        let u = on_wire(&u, quantize);
        let s = on_wire(&s, quantize);

        // Same local state as `full`, because the local step does not move weights.
        let (_, _) = full_alone.train(&batch(1), LR);
        full.apply(&[f.clone(), u, s], LR);
        full_alone.apply(&[f], LR);

        let mut checked = 0;
        for name in full.weight_names() {
            let Some(dim) = matformer_prefix_dim(&name) else {
                continue;
            };
            let dim = dim as i64;
            let tail = FULL_FFN - NARROW_FFN;
            let mixed = full.weight(&name).narrow(dim, NARROW_FFN, tail);
            let alone = full_alone.weight(&name).narrow(dim, NARROW_FFN, tail);
            assert_eq!(
                differing(&mixed, &alone),
                0.0,
                "{name}: tail moved (quantize={quantize})"
            );
            checked += 1;
        }
        assert_eq!(checked, 6, "2 layers with 3 FFN matrices each");
    }
}

/// Clients train on different batches, so they have different local errors.
/// They must still hold the same consensus weights, with no rounding difference.
#[test]
fn all_tiers_keep_exactly_the_same_shared_weights_and_the_loss_falls() {
    for (kind, quantize) in [
        (Kind::Float, false),
        (Kind::Float, true),
        (Kind::BFloat16, false),
        (Kind::BFloat16, true),
    ] {
        let (mut full, mut universal, mut sliced) = three_clients(kind);
        let names = full.weight_names();
        let (data_f, data_u, data_s) = (batch(7), batch(8), batch(9));
        let mut first_loss = None;
        let mut last_loss = 0.0;
        for _round in 0..30 {
            let (f, loss) = full.train(&data_f, LR);
            let (u, _) = universal.train(&data_u, LR);
            let (s, _) = sliced.train(&data_s, LR);
            first_loss.get_or_insert(loss);
            last_loss = loss;
            let all = vec![
                on_wire(&f, quantize),
                on_wire(&u, quantize),
                on_wire(&s, quantize),
            ];
            for client in [&mut full, &mut universal, &mut sliced] {
                client.apply(&all, LR);
            }

            for name in &names {
                let f = full.weight(name);
                let u = universal.weight(name);
                let s = sliced.weight(name);
                assert_eq!(differing(&f, &u), 0.0, "{name}: full and universal differ");
                let prefix = match matformer_prefix_dim(name) {
                    Some(dim) => f.narrow(dim as i64, 0, NARROW_FFN),
                    None => f,
                };
                assert_eq!(
                    differing(&prefix, &s),
                    0.0,
                    "{name}: prefix and sliced differ"
                );
            }

            for client in [&mut full, &mut universal, &mut sliced] {
                client.distro.error_correction(&client.model, LR);
            }
        }
        let first_loss = first_loss.unwrap();
        assert!(
            last_loss < first_loss,
            "loss did not fall: {first_loss} -> {last_loss} ({kind:?}, quantize={quantize})"
        );
    }
}

/// A new client must start with an empty error buffer, because a client that
/// joins after the first epoch never gets a call to `zero_optim`.
#[test]
fn a_new_client_starts_with_an_empty_error_buffer() {
    for _ in 0..20 {
        // Fill the allocator with freed nonzero memory first.
        let dirty: Vec<Tensor> = (0..64)
            .map(|_| Tensor::full([64, 32], 1e20, (Kind::Float, Device::Cpu)))
            .collect();
        drop(dirty);

        let weights = narrowed(&initial_weights(), FULL_FFN);
        let mut client = Client::new(weights, FULL_FFN, 0, Kind::Float, Device::Cpu);
        // `Distro::new` clears the gradient buffers, so this step has a zero gradient.
        let (results, _) = client.train(&batch(9), 0.0);
        for result in &results {
            assert_eq!(result.sparse_val.abs().max().double_value(&[]), 0.0);
        }
    }
}

/// Compares a real MPS client with a BF16 client when this machine has MPS.
///
/// This test uses the real chunk size. Very small chunks can cancel exactly, and
/// the sign of that rounding noise can then depend on the device.
#[test]
fn mps_float_client_tracks_bf16_client() {
    if !tch::utils::has_mps() {
        return;
    }
    let (hidden, ffn, vocab) = (128_i64, 256_i64, 256_i64);
    let (chunk, topk) = (64, 8);
    let lr = 4e-4;

    let mut sized = config(ffn);
    sized.hidden_size = hidden as usize;
    sized.vocab_size = vocab as usize;

    let mut numbers = Numbers(43);
    let mut names = sized.get_parameter_names();
    names.sort();
    let weights: HashMap<String, Tensor> = names
        .into_iter()
        .map(|name| {
            let shape: Vec<i64> = shape_of(&name, FULL_FFN)
                .into_iter()
                .map(|dim| match dim {
                    HIDDEN => hidden,
                    FULL_FFN if name.contains("mlp") => ffn,
                    FULL_FFN => vocab,
                    other => other,
                })
                .collect();
            let tensor = numbers.tensor(&shape);
            (name, tensor)
        })
        .collect();
    let build = |kind: Kind, device: Device| {
        let copy: HashMap<String, Tensor> = weights
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.copy()))
            .collect();
        let source = PretrainedSource::ConfigAndTensors(sized.clone(), Arc::new(copy));
        let model = LlamaForCausalLM::from_pretrained_with_matformer_tier(
            &source,
            Some(kind),
            None,
            Some(device),
            None,
            None,
            0,
        )
        .unwrap();
        model.prepare_for_training();
        let distro = Distro::new(&model, 0.999, chunk, topk, 0.0);
        (model, distro)
    };
    let (mps, mut mps_distro) = build(Kind::Float, Device::Mps);
    let (cpu, mut cpu_distro) = build(Kind::BFloat16, Device::Cpu);
    let (peer, mut peer_distro) = build(Kind::Float, Device::Cpu);

    // `apply` writes to the gradient buffer, so each model needs one backward pass first.
    let tokens = numbers.tokens(vocab);
    for model in [&mps, &cpu, &peer] {
        let data = tokens.to_device(model.device());
        let (_, loss) = model.forward(&data, Some(&data), None, None, None, None);
        loss.unwrap().backward();
    }

    for _round in 0..30 {
        for var in peer.variables() {
            var.zero_grad();
        }
        let data = numbers.tokens(vocab);
        let (_, loss) = peer.forward(&data, Some(&data), None, None, None, None);
        loss.unwrap().backward();
        let results = peer_distro.generate(&peer, &[], lr, false);
        let results = vec![Trainer::quantize_results(&results)];
        mps_distro.apply(&mps, &results, lr);
        cpu_distro.apply(&cpu, &results, lr);
        peer_distro.apply(&peer, &results, lr);
    }

    let on_cpu = |model: &LlamaForCausalLM| -> HashMap<String, Tensor> {
        model
            .variables()
            .map(|var| {
                let tensor = var.logical_tensor().detach().to_kind(Kind::Float);
                (var.name().to_string(), tensor.to_device(Device::Cpu))
            })
            .collect()
    };
    let mps_weights = on_cpu(&mps);
    let cpu_weights = on_cpu(&cpu);
    let mut compared = 0.0;
    let mut different = 0.0;
    for (name, tensor) in &cpu_weights {
        let count = tensor.numel() as f64;
        compared += count;
        different += differing(&mps_weights[name], tensor) * count;
    }
    let fraction = different / compared;
    println!("MPS Float against CPU BF16: {fraction} of elements differ after 30 applies");
    assert!(fraction < 1e-4, "MPS client left the BF16 grid: {fraction}");
}
