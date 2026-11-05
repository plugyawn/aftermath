use crate::{CausalLM, StableVariableIterator, Variable};

use std::{cmp::Ordering, collections::HashMap, f64::consts::PI};
use tch::{COptimizer, Device, Kind, Tensor};
use tracing::warn;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PrefixAlignError {
    UnsupportedParameter(String),
    RankMismatch { expected: usize, got: usize },
    ShapeMismatch { full: Vec<i64>, grad: Vec<i64> },
}

/// Returns the dimension that holds the FFN width of a MatFormer parameter.
///
/// Returns `None` for every parameter that has the same shape on all tiers.
pub fn matformer_prefix_dim(name: &str) -> Option<usize> {
    if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
        Some(0)
    } else if name.ends_with("down_proj.weight") {
        Some(1)
    } else {
        None
    }
}

/// Returns true when one width is the other width divided by a power of two.
///
/// All MatFormer tiers use `intermediate_size / 2^tier`, so a peer width that
/// fails this check cannot come from a tier of this model.
pub fn is_matformer_tier_width(a: i64, b: i64) -> bool {
    if a <= 0 || b <= 0 {
        return false;
    }
    let (wide, narrow) = if a >= b { (a, b) } else { (b, a) };
    (0..i64::BITS).any(|tier| wide >> tier == narrow)
}

/// Extract layer index from parameter name like "model.layers.5.mlp.gate_proj.weight"
// TODO: Use this when helper-mode sparse indices are wired (map params -> layer for index lookup).
// TODO: Remove if helper-mode sparse indices are dropped or moved elsewhere.
#[allow(dead_code)]
fn extract_layer_index(name: &str) -> Option<usize> {
    // Look for "layers.N" pattern
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

// TODO: Wire this into DisTrO apply once helper-mode indices are transmitted or computed.
// TODO: Plumb helper indices through DistroResult or compute via helper config + layer index.
/// Align gradient with non-contiguous MatFormer helper indices.
///
/// This function scatters a gradient from a subset of indices back to the full shape.
/// Used when helper mode is enabled and gradients cover prefix + stochastic suffix indices.
///
/// # Arguments
/// * `name` - Parameter name (e.g., "model.layers.0.mlp.gate_proj.weight")
/// * `full_shape` - Full shape of the parameter [intermediate_size, hidden_size] or [hidden_size, intermediate_size]
/// * `grad` - Gradient tensor covering the selected indices
/// * `indices` - Which indices the gradient covers (e.g., [0,1,2,...,767] + [891,1024,...])
///
/// # Returns
/// Full-sized gradient tensor with values scattered to the correct positions
#[allow(dead_code)]
pub fn align_matformer_sparse_grad(
    name: &str,
    full_shape: &[i64],
    grad: Tensor,
    indices: &[i64],
) -> Result<Tensor, PrefixAlignError> {
    let grad_shape = grad.size();

    // If already full size, return as-is
    if grad_shape == full_shape {
        return Ok(grad);
    }

    let Some(prefix_dim) = matformer_prefix_dim(name) else {
        return Err(PrefixAlignError::UnsupportedParameter(name.to_string()));
    };

    if full_shape.len() != 2 || grad_shape.len() != 2 {
        return Err(PrefixAlignError::RankMismatch {
            expected: 2,
            got: grad_shape.len(),
        });
    }

    // Verify the other dimension matches
    let other_dim = if prefix_dim == 0 { 1 } else { 0 };
    if grad_shape[other_dim] != full_shape[other_dim] {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    // Verify grad and indices have compatible sizes
    if grad_shape[prefix_dim] != indices.len() as i64 {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    // Create full-size zero tensor
    let mut full_grad = Tensor::zeros(full_shape, (grad.kind(), grad.device()));

    // Create index tensor
    let indices_tensor = Tensor::from_slice(indices).to_device(grad.device());

    // Scatter gradient values to their correct positions
    if prefix_dim == 0 {
        // gate_proj, up_proj: indices select rows
        let _ = full_grad.index_copy_(0, &indices_tensor, &grad);
    } else {
        // down_proj: indices select columns
        let _ = full_grad.index_copy_(1, &indices_tensor, &grad);
    }

    Ok(full_grad)
}

fn align_matformer_prefix_grad(
    name: &str,
    full_shape: &[i64],
    grad: Tensor,
) -> Result<Tensor, PrefixAlignError> {
    let grad_shape = grad.size();
    if grad_shape == full_shape {
        return Ok(grad);
    }

    let Some(prefix_dim) = matformer_prefix_dim(name) else {
        return Err(PrefixAlignError::UnsupportedParameter(name.to_string()));
    };

    if full_shape.len() != grad_shape.len() {
        return Err(PrefixAlignError::RankMismatch {
            expected: full_shape.len(),
            got: grad_shape.len(),
        });
    }
    if full_shape.len() != 2 {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    let other_dim = if prefix_dim == 0 { 1 } else { 0 };
    if grad_shape[other_dim] != full_shape[other_dim] {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    let prefix_len = full_shape[prefix_dim];
    let grad_len = grad_shape[prefix_dim];
    if grad_len == prefix_len {
        return Ok(grad);
    }

    if !is_matformer_tier_width(prefix_len, grad_len) {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    if grad_len > prefix_len {
        Ok(grad.narrow(prefix_dim as i64, 0, prefix_len))
    } else {
        let expanded = Tensor::zeros(full_shape, (grad.kind(), grad.device()));
        let mut prefix_view = expanded.narrow(prefix_dim as i64, 0, grad_len);
        prefix_view.copy_(&grad);
        Ok(expanded)
    }
}

/// Checks a received result against the local parameter before any decode.
///
/// A result that fails this check would panic in decode or would write to the
/// wrong coefficients, so `Distro::apply` drops it.
fn validate_distro_result(
    name: &str,
    full_shape: &[i64],
    result: &DistroResult,
) -> Result<(), PrefixAlignError> {
    let rank_error = || PrefixAlignError::RankMismatch {
        expected: full_shape.len() * 2,
        got: result.xshape.len(),
    };
    if result.xshape.iter().any(|&dim| dim <= 0) {
        return Err(rank_error());
    }
    let (decoded_shape, totalk, batch_dims) = match (result.xshape.as_slice(), full_shape.len()) {
        (&[y, x, h, w], 2) => (vec![y * h, x * w], h * w, vec![y, x]),
        (&[x, w], 1) => (vec![x * w], w, vec![x]),
        _ => return Err(rank_error()),
    };

    let shape_error = |grad: Vec<i64>| PrefixAlignError::ShapeMismatch {
        full: full_shape.to_vec(),
        grad,
    };

    if decoded_shape != full_shape {
        let Some(prefix_dim) = matformer_prefix_dim(name) else {
            return Err(PrefixAlignError::UnsupportedParameter(name.to_string()));
        };
        let other_dim = 1 - prefix_dim;
        if full_shape.len() != 2
            || decoded_shape[other_dim] != full_shape[other_dim]
            || !is_matformer_tier_width(full_shape[prefix_dim], decoded_shape[prefix_dim])
        {
            return Err(shape_error(decoded_shape));
        }
    }

    if result.totalk.abs() != totalk {
        return Err(shape_error(decoded_shape));
    }

    // `compress_idx` stores indices as raw bytes, except for very large tensors.
    let idx_bytes_per_item = match totalk {
        t if t <= 256 => 1,
        t if t <= 65536 => 2,
        t if t <= 4294967296 => 4,
        _ => 1,
    };
    if totalk <= 4294967296 && result.sparse_idx.kind() != Kind::Uint8 {
        return Err(shape_error(result.sparse_idx.size()));
    }
    let idx_size = result.sparse_idx.size();
    let Some((&idx_last, idx_batch)) = idx_size.split_last() else {
        return Err(shape_error(idx_size));
    };
    let val_size = result.sparse_val.size();
    let values_match = idx_last > 0
        && idx_last % idx_bytes_per_item == 0
        && idx_batch == batch_dims.as_slice()
        && val_size.split_last().map(|(&last, batch)| {
            last == idx_last / idx_bytes_per_item && batch == batch_dims.as_slice()
        }) == Some(true);
    if !values_match {
        return Err(shape_error(val_size));
    }

    let max_idx = decompress_idx(totalk, &result.sparse_idx)
        .max()
        .int64_value(&[]);
    if max_idx >= totalk {
        return Err(shape_error(vec![max_idx]));
    }
    Ok(())
}

/// Rounds a tensor to the BF16 grid and keeps its kind.
fn round_to_bf16(tensor: &Tensor) -> Tensor {
    tensor.to_kind(Kind::BFloat16).to_kind(tensor.kind())
}

/// Rounds a scalar to the BF16 grid.
///
/// BF16 tensor arithmetic rounds a scalar operand first, so this step is
/// necessary to get the same result in a Float tensor.
fn round_scalar_to_bf16(value: f64) -> f64 {
    Tensor::from(value as f32)
        .to_kind(Kind::BFloat16)
        .to_kind(Kind::Double)
        .double_value(&[])
}

pub struct TransformDCT {
    shape_dict: HashMap<i64, i64>,
    f_dict: HashMap<i64, Tensor>,
    b_dict: HashMap<i64, Tensor>,
}

impl TransformDCT {
    pub fn new(variables: StableVariableIterator, target_chunk: i64) -> Self {
        let _no_grad = tch::no_grad_guard();
        let mut transform = Self {
            shape_dict: HashMap::new(),
            f_dict: HashMap::new(),
            b_dict: HashMap::new(),
        };

        // Get all variants of model tensor sizes
        // Generate all possible valid DCT sizes for model tensors
        for variable in variables {
            let size = variable.full_tensor_shape();
            let variable = variable.local_tensor();
            for s in size {
                transform.register_size(s, target_chunk, variable.device(), variable.kind());
            }
        }
        transform
    }

    /// Adds the DCT basis for one tensor dimension.
    ///
    /// A client that holds full weights but trains a narrower tier needs this
    /// for its tier width, because no local variable has that size.
    pub fn register_size(&mut self, s: i64, target_chunk: i64, device: Device, kind: Kind) {
        let _no_grad = tch::no_grad_guard();
        // Get the closest smallest divisor to the targeted DCT size
        let sc = match self.shape_dict.get(&s) {
            Some(sc) => *sc,
            None => {
                let sc = Self::get_smaller_split(s, target_chunk);
                self.shape_dict.insert(s, sc);
                sc
            }
        };

        // Pregenerate DCT basis matrices
        if let std::collections::hash_map::Entry::Vacant(e) = self.f_dict.entry(sc) {
            let i = Tensor::eye(sc, (Kind::Float, device));
            e.insert(Self::dct(&i, true).to_kind(kind).to(device));
            // The decode basis stays in Float. `apply` must decode at the same precision
            // on all clients, because a rounded basis can flip the sign of a small value.
            self.b_dict.insert(sc, Self::idct(&i, true).to(device));
        }
    }

    fn get_prime_divisors(mut n: i64) -> Vec<i64> {
        if n == 0 {
            return Vec::new();
        }
        let mut divisors = Vec::new();
        while n % 2 == 0 {
            divisors.push(2);
            n /= 2;
        }
        while n % 3 == 0 {
            divisors.push(3);
            n /= 3;
        }
        let mut i = 5;
        while i * i <= n {
            for k in [i, i + 2].iter() {
                while n % k == 0 {
                    divisors.push(*k);
                    n /= k;
                }
            }
            i += 6;
        }
        if n > 1 {
            divisors.push(n);
        }
        divisors
    }

    fn get_divisors(n: i64) -> Vec<i64> {
        let mut divisors = Vec::new();
        match n.cmp(&1) {
            Ordering::Equal => {
                divisors.push(1);
            }
            Ordering::Greater => {
                let prime_factors = Self::get_prime_divisors(n);
                divisors = vec![1];
                let mut last_prime = 0;
                let mut factor = 0;
                let mut slice_len = 0;
                // Find all the products that are divisors of n
                for prime in prime_factors {
                    if last_prime != prime {
                        slice_len = divisors.len();
                        factor = prime;
                    } else {
                        factor *= prime;
                    }
                    for i in 0..slice_len {
                        divisors.push(divisors[i] * factor);
                    }
                    last_prime = prime;
                }
                divisors.sort_unstable();
            }
            Ordering::Less => {}
        }
        divisors
    }

    fn get_smaller_split(n: i64, close_to: i64) -> i64 {
        let all_divisors = Self::get_divisors(n);
        for (ix, &val) in all_divisors.iter().enumerate() {
            if val == close_to {
                return val;
            }
            if val > close_to {
                if ix == 0 {
                    return val;
                }
                return all_divisors[ix - 1];
            }
        }
        n
    }

    fn dct_fft_impl(v: &Tensor) -> Tensor {
        v.fft_fft(None, 1, "backward").view_as_real()
    }

    #[allow(unused)]
    fn dct(x: &Tensor, ortho: bool) -> Tensor {
        let x_shape = x.size();
        let n = { *x_shape.last().unwrap() };
        let x = x.contiguous().view([-1, n]);

        let v = Tensor::cat(
            &[x.slice(1, 0, None, 2), x.slice(1, 1, None, 2).flip([1])],
            1,
        );

        let vc = Self::dct_fft_impl(&v);

        let k = -Tensor::arange(n, (Kind::Float, x.device()))
            .unsqueeze(0)
            .g_mul_scalar(PI / (2.0 * n as f64));
        let w_r = k.cos();
        let w_i = k.sin();

        let mut v = vc.select(2, 0) * &w_r - vc.select(2, 1) * &w_i;

        if ortho {
            v.select(1, 0).g_div_scalar_((n as f64).sqrt() * 2.0);
            v.slice(1, 1, None, 1)
                .g_div_scalar_((n as f64 / 2.0).sqrt() * 2.0);
        }

        v.g_mul_scalar_(2.0).view(x_shape.as_slice())
    }

    fn idct_irfft_impl(v: &Tensor) -> Tensor {
        let complex_v = v.view_as_complex();
        let n = v.size()[1];
        complex_v.fft_irfft(Some(n), 1, "backward")
    }

    #[allow(unused)]
    fn idct(x: &Tensor, ortho: bool) -> Tensor {
        let x_shape = x.size();
        let n = { *x_shape.last().unwrap() };

        let mut x_v = x.contiguous().view([-1, n]).f_div_scalar(2.0).unwrap();

        if ortho {
            x_v.slice(1, 0, 1, 1)
                .f_mul_scalar_((n as f64).sqrt() * 2.0)
                .unwrap();
            x_v.slice(1, 1, n, 1)
                .f_mul_scalar_((n as f64 / 2.0).sqrt() * 2.0)
                .unwrap();
        }

        let k = Tensor::arange(n, (Kind::Float, x.device()))
            .f_mul_scalar(PI / (2.0 * n as f64))
            .unwrap()
            .unsqueeze(0);

        let w_r = k.cos();
        let w_i = k.sin();

        let v_t_r = &x_v;
        let v_t_i = Tensor::cat(
            &[
                x_v.slice(1, 0, 1, 1).f_mul_scalar(0.0).unwrap(),
                x_v.flip([1]).slice(1, 0, n - 1, 1).f_neg().unwrap(),
            ],
            1,
        );

        let v_r = v_t_r.f_mul(&w_r).unwrap() - v_t_i.f_mul(&w_i).unwrap();
        let v_i = v_t_r.f_mul(&w_i).unwrap() + v_t_i.f_mul(&w_r).unwrap();

        let v = Tensor::cat(&[v_r.unsqueeze(2), v_i.unsqueeze(2)], 2);

        let v = Self::idct_irfft_impl(&v);

        let mut x = Tensor::zeros(v.size(), (Kind::Float, v.device()));

        x.slice(1, 0, n, 2)
            .f_add_(&v.slice(1, 0, n - (n / 2), 1))
            .unwrap();
        x.slice(1, 1, n, 2)
            .f_add_(&v.flip([1]).slice(1, 0, n / 2, 1))
            .unwrap();

        x.view(x_shape.as_slice())
    }

    fn einsum_2d(x: &Tensor, b: &Tensor, d: Option<&Tensor>) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        match d {
            None => Tensor::einsum("...ij, jb -> ...ib", &[x, b], None::<i64>),
            Some(d_tensor) => {
                // Note: b-c axis output is transposed to chunk DCT in 2D
                Tensor::einsum("...ijkl, jb, ld -> ...ikbd", &[x, b, d_tensor], None::<i64>)
            }
        }
    }

    fn einsum_2d_t(x: &Tensor, b: &Tensor, d: Option<&Tensor>) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        match d {
            None => Tensor::einsum("...ij, jb -> ...ib", &[x, b], None::<i64>),
            Some(d_tensor) => {
                // Note: b-c axis output is transposed to chunk DCT in 2D
                Tensor::einsum("...ijkl, kb, ld -> ...ibjd", &[x, b, d_tensor], None::<i64>)
            }
        }
    }

    pub fn encode(&mut self, x: &Tensor) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        if x.size().len() > 1 {
            // 2D weights
            let n1 = *self.shape_dict.get(&x.size()[0]).unwrap();
            let n2 = *self.shape_dict.get(&x.size()[1]).unwrap();
            let n1w = self.f_dict.get(&n1).unwrap().to_device(x.device());
            let n2w = self.f_dict.get(&n2).unwrap().to_device(x.device());
            self.f_dict.insert(n1, n1w.copy());
            self.f_dict.insert(n2, n2w.copy());

            // Equivalent to rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            let x = x.view([x.size()[0] / n1, n1, x.size()[1] / n2, n2]);
            Self::einsum_2d(&x, &n1w, Some(&n2w))
        } else {
            // 1D weights
            let n1 = *self.shape_dict.get(&x.size()[0]).unwrap();
            let n1w = self.f_dict.get(&n1).unwrap().to_device(x.device());
            self.f_dict.insert(n1, n1w.copy());

            // Equivalent to rearrange(x, "(x w) -> x w", w=n1)
            let x = x.view([-1, n1]);
            Self::einsum_2d(&x, &n1w, None)
        }
    }

    pub fn decode(&mut self, x: &Tensor) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x_shape = x.size();

        if x_shape.len() > 2 {
            // 2D weights
            let n1 = x_shape[2];
            let n2 = x_shape[3];
            let device = x.device();

            if !self.b_dict.contains_key(&n1) {
                let i = Tensor::eye(n1, (Kind::Float, device));
                self.b_dict.insert(n1, Self::idct(&i, true));
            }
            if !self.b_dict.contains_key(&n2) {
                let i = Tensor::eye(n2, (Kind::Float, device));
                self.b_dict.insert(n2, Self::idct(&i, true));
            }
            let n1w = self.b_dict.get(&n1).unwrap().to_device(device);
            let n2w = self.b_dict.get(&n2).unwrap().to_device(device);

            self.b_dict.insert(n1, n1w.copy());
            self.b_dict.insert(n2, n2w.copy());

            let x = Self::einsum_2d_t(x, &n1w.to_kind(x.kind()), Some(&n2w.to_kind(x.kind())));
            let x_shape = x.size();

            // Equivalent to rearrange(x, "y h x w -> (y h) (x w)")
            let (y, h, x_, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
            x.reshape([y * h, x_ * w])
        } else {
            // 1D weights
            let n1 = x_shape[1];
            let device = x.device();

            if !self.b_dict.contains_key(&n1) {
                let i = Tensor::eye(n1, (Kind::Float, device));
                self.b_dict.insert(n1, Self::idct(&i, true));
            }
            let n1w = self.b_dict.get(&n1).unwrap().to_device(device);
            self.b_dict.insert(n1, n1w.copy());

            let x = Self::einsum_2d_t(x, &n1w.to_kind(x.kind()), None);
            let x_shape = x.size();

            // Equivalent to rearrange(x, "x w -> (x w)")
            let (x_, w) = (x_shape[0], x_shape[1]);
            x.reshape([x_ * w])
        }
    }
}

pub struct CompressDCT;

impl CompressDCT {
    fn clamp_topk(x: &Tensor, topk: i64) -> i64 {
        let last_dim = x.size()[x.dim() - 1];

        if topk > last_dim {
            last_dim
        } else if topk < 1 {
            1
        } else {
            topk
        }
    }

    pub fn compress(x: &Tensor, topk: i64) -> (Tensor, Tensor, Vec<i64>, i64) {
        let _no_grad = tch::no_grad_guard();
        let xshape = x.size();
        let x = if xshape.len() > 2 {
            // Equivalent to rearrange(x, "y x h w -> y x (h w)")
            let y = xshape[0];
            let x_dim = xshape[1];
            let h = xshape[2];
            let w = xshape[3];
            x.view([y, x_dim, h * w])
        } else {
            x.shallow_clone()
        };

        let totalk = *x.size().last().unwrap();
        let topk = Self::clamp_topk(&x, topk);

        let idx = x.abs().topk(topk, -1, true, false).1;
        let val = x.gather(-1, &idx, false);

        let idx = compress_idx(totalk, &idx);

        (idx, val, xshape, totalk)
    }

    #[allow(unused)]
    pub fn decompress(
        idx: &Tensor,
        val: &Tensor,
        xshape: &[i64],
        totalk: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let totalk = totalk.abs();

        let idx = decompress_idx(totalk, idx);

        let val = val.to_kind(kind);

        let mut x: Tensor = Tensor::zeros(xshape, (kind, device));

        if xshape.len() > 2 {
            // 2D weights
            // Equivalent to rearrange(x, "y x h w -> y x (h w)")
            let y = xshape[0];
            let x_dim = xshape[1];
            let h = xshape[2];
            let w = xshape[3];
            x = x.view([y, x_dim, h * w]);
        }

        x.internal_scatter_reduce_(-1, &idx, &val, "mean", false);

        x = x.reshape(xshape);

        if x.size().len() > 2 {
            // 2D weights
            // Equivalent to rearrange(x, "y x (h w) -> y x h w", h=xshape[2])
            let y = xshape[0];
            let x_dim = xshape[1];
            let h = xshape[2];
            let w = xshape[3];
            x = x.view([y, x_dim, h, w]);
        }

        x
    }

    pub fn batch_decompress(
        idx: &[Tensor],
        val: &[Tensor],
        xshape: &[i64],
        totalk: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let idx_concat = Tensor::cat(idx, -1).to_device(device);
        let val_concat = Tensor::cat(val, -1).to_device(device);
        // Call the decompress method
        Self::decompress(&idx_concat, &val_concat, xshape, totalk, kind, device)
    }
}

fn compress_idx(max_value: i64, idx: &Tensor) -> Tensor {
    if max_value <= 256 {
        idx.to_kind(Kind::Uint8)
    } else if max_value <= 65536 {
        idx.to_kind(Kind::UInt16).view_dtype(Kind::Uint8)
    } else if max_value <= 4294967296 {
        idx.to_kind(Kind::UInt32).view_dtype(Kind::Uint8)
    } else {
        idx.shallow_clone()
    }
}

fn decompress_idx(max_value: i64, idx: &Tensor) -> Tensor {
    if max_value <= 256 {
        idx.view_dtype(Kind::Uint8)
    } else if max_value <= 65536 {
        idx.view_dtype(Kind::UInt16)
    } else if max_value <= 4294967296 {
        idx.view_dtype(Kind::UInt32)
    } else {
        idx.shallow_clone()
    }
    .to_kind(Kind::Int64)
}

struct State {
    delta: Box<dyn Variable>,
    /// Flat indices and earlier values of the elements that rounding stops
    /// the reverse lookahead step from restoring.
    lookahead_restore: Option<(Tensor, Tensor)>,
}

#[derive(Debug)]
pub struct DistroResult {
    pub sparse_idx: Tensor,
    pub sparse_val: Tensor,
    pub xshape: Vec<i64>,
    pub totalk: i64,
    pub stats: Option<HashMap<String, f64>>,
}

impl Clone for DistroResult {
    fn clone(&self) -> Self {
        Self {
            sparse_idx: self.sparse_idx.shallow_clone(),
            sparse_val: self.sparse_val.shallow_clone(),
            xshape: self.xshape.clone(),
            totalk: self.totalk,
            stats: self.stats.clone(),
        }
    }
}

pub struct Distro {
    sgd: COptimizer,
    compression_decay: f64,
    compression_topk: i64,
    weight_decay: f64,
    state: Vec<State>,
    transform: TransformDCT,
    /// FFN width of this client when it holds full weights but trains a narrower tier.
    ///
    /// The client sends only this prefix. This gives the same wire format as a
    /// client with sliced weights, and it never sends the empty tail.
    active_ffn_width: Option<i64>,
    /// Keep Float weights on the BF16 grid, so this client tracks BF16 peers.
    snap_to_bf16: bool,
    /// Step size of the lookahead that is now in the weights.
    lookahead_lr: Option<f64>,
}

impl Distro {
    pub fn new(
        vs: &dyn CausalLM,
        compression_decay: f64,
        compression_chunk: i64,
        compression_topk: i64,
        weight_decay: f64,
    ) -> Self {
        let _no_grad = tch::no_grad_guard();
        let mut sgd = COptimizer::sgd(0.1, 0.0, 0.0, 0.0, false).unwrap();

        let mut state = Vec::new();
        for variable in vs.variables() {
            state.push(State {
                delta: variable.zeros_like(format!("{}.delta", variable.name())),
                lookahead_restore: None,
            });

            let logical_tensor = variable.logical_tensor();
            sgd.add_parameters(&logical_tensor, 0).unwrap();
            variable.zero_grad();
        }

        let mut transform = TransformDCT::new(vs.variables(), compression_chunk);

        let active_ffn_width = vs.matformer_active_ffn_width();
        let first_variable = vs.variables().next().map(|v| v.logical_tensor());
        if let (Some(width), Some(tensor)) = (active_ffn_width, first_variable.as_ref()) {
            transform.register_size(width, compression_chunk, tensor.device(), tensor.kind());
        }

        // MPS keeps weights in Float, and all other devices keep them in BF16.
        let snap_to_bf16 = first_variable
            .map(|t| t.device() == Device::Mps && t.kind() == Kind::Float)
            .unwrap_or(false);

        let distro = Self {
            sgd,
            compression_decay,
            compression_topk,
            weight_decay,
            state,
            transform,
            active_ffn_width,
            snap_to_bf16,
            lookahead_lr: None,
        };
        distro.snap_variables(vs);
        distro
    }

    fn snap_variables(&self, vars: &dyn CausalLM) {
        if !self.snap_to_bf16 {
            return;
        }
        let _no_grad = tch::no_grad_guard();
        for var in vars.variables() {
            let mut variable = var.logical_tensor();
            let rounded = round_to_bf16(&variable);
            variable.copy_(&rounded);
        }
    }

    fn snap_tensor(&self, tensor: &mut Tensor) {
        if self.snap_to_bf16 {
            let rounded = round_to_bf16(tensor);
            tensor.copy_(&rounded);
        }
    }

    /// Returns the lookahead step for one variable, in the precision of the consensus weights.
    fn lookahead_step(&self, delta: &Tensor, lr: f64) -> Tensor {
        let step = delta.sign().multiply_scalar(lr);
        if self.snap_to_bf16 {
            round_to_bf16(&step)
        } else {
            step
        }
    }

    /// Fraction of elements that one step of size `lr` leaves unchanged in BF16.
    fn bf16_lost_update_fraction(variable: &Tensor, lr: f64) -> f64 {
        let weights = variable.to_kind(Kind::Float);
        let updated = (&weights - lr).to_kind(Kind::BFloat16).to_kind(Kind::Float);
        updated
            .eq_tensor(&round_to_bf16(&weights))
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[])
    }

    pub fn generate(
        &mut self,
        variables: &dyn CausalLM,
        prev_self_results: &[Vec<DistroResult>],
        lr: f64,
        stats: bool,
    ) -> Vec<DistroResult> {
        let _no_grad = tch::no_grad_guard();

        self.undo_lookahead(variables);

        let mut ret = Vec::new();
        for (index, var) in variables.variables().enumerate() {
            let mut variable = var.logical_tensor();

            let grad_energy: Option<f64> = match stats {
                true => Some(
                    variable
                        .grad()
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                _ => None,
            };

            let lost_update_fraction: Option<f64> =
                stats.then(|| Self::bf16_lost_update_fraction(&variable, lr));

            let mut delta = self.state[index].delta.logical_tensor();

            if !prev_self_results.is_empty() {
                let device = variable.device();
                let indicies = prev_self_results
                    .iter()
                    .map(|x| x[index].sparse_idx.to_device(device))
                    .collect::<Vec<_>>();

                let val_kind: Kind = variable.kind();
                let values = prev_self_results
                    .iter()
                    .map(|x| {
                        let sparse_val = x[index].sparse_val.to_device(device);
                        if sparse_val.kind() == Kind::Bool {
                            Self::unpack_tensor_sign_from_boolean(sparse_val, val_kind)
                        } else {
                            sparse_val
                        }
                    })
                    .collect::<Vec<_>>();

                // Decode grad from all nodes
                let decompressed = CompressDCT::batch_decompress(
                    &indicies,
                    &values,
                    &prev_self_results[0][index].xshape,
                    prev_self_results[0][index].totalk,
                    val_kind,
                    device,
                );
                let transmit_grad = self.transform.decode(&decompressed);
                // We may send only the tier prefix, and delta always has the full shape.
                let transmit_grad = align_matformer_prefix_grad(
                    var.name(),
                    &var.full_tensor_shape(),
                    transmit_grad,
                )
                .expect("our own DisTrO result must match our own parameter shape");

                // Remove transmitted from delta
                let _t = delta.g_sub_(&var.shard_other_tensor_like_me(transmit_grad));
            }

            // weight decay
            if self.weight_decay != 0.0 {
                let factor = 1.0 - lr * self.weight_decay;
                let factor = if self.snap_to_bf16 {
                    round_scalar_to_bf16(factor)
                } else {
                    factor
                };
                let _t = variable.g_mul_scalar_(factor);
                self.snap_tensor(&mut variable);
            }

            // decay delta
            if self.compression_decay != 1.0 {
                let _t = delta.g_mul_scalar_(self.compression_decay);
            }

            // add delta to new gradient
            let _t = delta.g_add_(&variable.grad().multiply_scalar(lr));

            // Compress delta
            let full_delta = self.state[index].delta.gather_full_tensor();
            // A narrower tier has no gradient in the tail, so it must not send one:
            // 1-bit values cannot express zero, and raw zeros dilute full-tier peers.
            let delta_to_send = match (self.active_ffn_width, matformer_prefix_dim(var.name())) {
                (Some(width), Some(prefix_dim)) => {
                    full_delta.narrow(prefix_dim as i64, 0, width).contiguous()
                }
                _ => full_delta.shallow_clone(),
            };
            let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(
                &self.transform.encode(&delta_to_send),
                self.compression_topk,
            );

            let delta_energy: Option<f64> = match stats {
                true => Some(
                    full_delta
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                false => None,
            };

            ret.push(DistroResult {
                sparse_idx,
                sparse_val,
                xshape,
                totalk,
                stats: match stats {
                    true => {
                        let name = var.name();
                        Some(HashMap::from([
                            (format!("{name}.delta_energy"), delta_energy.unwrap()),
                            (format!("{name}.grad_energy"), grad_energy.unwrap()),
                            (
                                format!("{name}.bf16_lost_update_fraction"),
                                lost_update_fraction.unwrap(),
                            ),
                        ]))
                    }
                    false => None,
                },
            });
        }
        ret
    }

    pub fn apply(&mut self, vars: &dyn CausalLM, results: &[Vec<DistroResult>], lr: f64) {
        let _no_grad = tch::no_grad_guard();
        if results.is_empty() {
            return;
        }

        for (index, var) in vars.variables().enumerate() {
            let variable = var.logical_tensor();
            let device = variable.device();
            let full_shape = var.full_tensor_shape();

            let usable: Vec<&DistroResult> = results
                .iter()
                .map(|peer| &peer[index])
                .filter(
                    |result| match validate_distro_result(var.name(), &full_shape, result) {
                        Ok(()) => true,
                        Err(err) => {
                            warn!(
                                parameter = var.name(),
                                full_shape = ?full_shape,
                                xshape = ?result.xshape,
                                "Dropping invalid peer result in DisTrO apply: {err:?}"
                            );
                            false
                        }
                    },
                )
                .collect();
            let Some(first) = usable.first() else {
                warn!(
                    parameter = var.name(),
                    "Skipping DisTrO apply: no compatible grads found"
                );
                continue;
            };

            // All clients decode in Float, so the vote does not depend on the weight kind.
            let val_kind = Kind::Float;
            let indicies = usable
                .iter()
                .map(|result| result.sparse_idx.to_device(device))
                .collect::<Vec<_>>();
            let values = usable
                .iter()
                .map(|result| {
                    let sparse_val = result.sparse_val.to_device(device);
                    if sparse_val.kind() == Kind::Bool {
                        Self::unpack_tensor_sign_from_boolean(sparse_val, val_kind)
                    } else {
                        sparse_val
                    }
                })
                .collect::<Vec<_>>();

            let same_shape = usable
                .iter()
                .all(|result| result.xshape == first.xshape && result.totalk == first.totalk);

            let aggregated = if same_shape {
                // Decode grad from all nodes (fast path)
                let decompressed = CompressDCT::batch_decompress(
                    &indicies,
                    &values,
                    &first.xshape,
                    first.totalk,
                    val_kind,
                    device,
                );
                let decoded = self.transform.decode(&decompressed);
                align_matformer_prefix_grad(var.name(), &full_shape, decoded)
                    .expect("validate_distro_result accepted this shape")
            } else {
                // Heterogeneous shapes: decode individually, then align to local shape.
                // We take only the sign of this sum below, so it needs no peer-count division.
                let mut combined = Tensor::zeros(&full_shape, (val_kind, device));
                for ((result, sparse_idx), sparse_val) in
                    usable.iter().zip(indicies.iter()).zip(values.iter())
                {
                    let decompressed = CompressDCT::decompress(
                        sparse_idx,
                        sparse_val,
                        &result.xshape,
                        result.totalk,
                        val_kind,
                        device,
                    );
                    let decoded = self.transform.decode(&decompressed);
                    let aligned = align_matformer_prefix_grad(var.name(), &full_shape, decoded)
                        .expect("validate_distro_result accepted this shape");
                    let _t = combined.g_add_(&aligned);
                }
                combined
            };

            // Sign-SGD. We take the sign before the write, because the write can
            // round a small Float value to zero in a BF16 gradient buffer.
            var.set_grad(aggregated.sign());
        }
        // SGD step
        let effective_lr = if self.snap_to_bf16 {
            round_scalar_to_bf16(lr)
        } else {
            lr
        };
        self.sgd.set_learning_rate(effective_lr).unwrap();
        let _ = self.sgd.step();
        self.snap_variables(vars);
        for var in vars.variables() {
            var.zero_grad();
        }
    }

    /// Moves the weights to the lookahead point for the next forward pass.
    ///
    /// Clients have different local errors, so the consensus weights must come back
    /// exactly. Rounding can stop the reverse step from restoring an element, so
    /// this function also keeps the earlier values of those elements.
    pub fn error_correction(&mut self, vars: &dyn CausalLM, prev_lr: f64) {
        let _no_grad = tch::no_grad_guard();
        self.undo_lookahead(vars);
        for (index, var) in vars.variables().enumerate() {
            let mut variable = var.logical_tensor();
            let before = variable.copy();

            // Apply lookahead, the signed delta, multiplied by lr
            let step = self.lookahead_step(&self.state[index].delta.logical_tensor(), prev_lr);
            let _t = variable.g_sub_(&step);
            self.snap_tensor(&mut variable);

            let mut restored = &variable + &step;
            self.snap_tensor(&mut restored);
            let indices = restored
                .ne_tensor(&before)
                .flatten(0, -1)
                .nonzero()
                .flatten(0, -1);
            self.state[index].lookahead_restore = (indices.numel() > 0).then(|| {
                let values = before.flatten(0, -1).index_select(0, &indices);
                (indices, values)
            });
        }
        self.lookahead_lr = Some(prev_lr);
    }

    /// Returns the weights to the consensus values.
    ///
    /// `generate` calls this function after the backward pass. A step that stops
    /// before `generate` must call it too, or the lookahead stays in the weights.
    pub fn undo_lookahead(&mut self, vars: &dyn CausalLM) {
        let Some(lr) = self.lookahead_lr.take() else {
            return;
        };
        let _no_grad = tch::no_grad_guard();
        for (index, var) in vars.variables().enumerate() {
            let mut variable = var.logical_tensor();
            let step = self.lookahead_step(&self.state[index].delta.logical_tensor(), lr);
            let _t = variable.g_add_(&step);
            self.snap_tensor(&mut variable);
            if let Some((indices, values)) = self.state[index].lookahead_restore.take() {
                let _t = variable.flatten(0, -1).index_copy_(0, &indices, &values);
            }
        }
    }

    pub fn zero_optim(&mut self) {
        for state in &mut self.state {
            let _ = state.delta.logical_tensor().zero_();
        }
    }

    pub fn quantize_nozeros_tensor_to_boolean_sign(tensor: &Tensor) -> Tensor {
        let original_size = tensor.size();
        let tensor = tensor.signbit();
        debug_assert_eq!(tensor.kind(), Kind::Bool);
        debug_assert_eq!(tensor.size(), original_size);
        tensor
    }

    /// Packs values to 1-bit signs for transmission.
    ///
    /// One bit cannot express zero. A fixed sign would give every peer the same
    /// bias on a coefficient that holds no information, so a zero gets a random sign.
    pub fn quantize_tensor_to_boolean_sign(tensor: &Tensor) -> Tensor {
        let signs = Self::quantize_nozeros_tensor_to_boolean_sign(tensor);
        let zeros = tensor.eq(0.0);
        if zeros.any().int64_value(&[]) == 0 {
            return signs;
        }
        let random = Tensor::rand(tensor.size(), (Kind::Float, tensor.device())).lt(0.5);
        random.where_self(&zeros, &signs)
    }

    fn unpack_tensor_sign_from_boolean(tensor: Tensor, unpack_kind: Kind) -> Tensor {
        tensor.to_kind(unpack_kind) * -2 + 1
    }
}

unsafe impl Send for Distro {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{set_torch_rng_seed, Variable};
    use itertools::iproduct;

    impl Variable for Tensor {
        fn name(&self) -> &str {
            unimplemented!()
        }

        fn local_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn logical_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn gather_full_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn shard_other_tensor_like_me(&self, tensor: Tensor) -> Tensor {
            tensor
        }

        fn full_tensor_shape(&self) -> Vec<i64> {
            self.size()
        }

        fn is_sharded(&self) -> bool {
            false
        }

        fn zeros_like(&self, _name: String) -> Box<dyn Variable> {
            Box::new(self.zeros_like())
        }

        fn set_grad(&self, tensor: Tensor) {
            self.grad().copy_(&tensor);
        }

        fn zero_grad(&self) {
            let grad = self.grad();
            if grad.defined() {
                let _ = self.grad().zero_();
            }
        }
    }

    fn vars(vars: Vec<Tensor>) -> StableVariableIterator {
        Box::new(vars.into_iter().map(|x| Box::new(x) as Box<dyn Variable>))
    }

    #[test]
    fn test_get_prime_divisors() {
        assert_eq!(TransformDCT::get_prime_divisors(1), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_prime_divisors(2), vec![2]);
        assert_eq!(TransformDCT::get_prime_divisors(12), vec![2, 2, 3]);
        assert_eq!(TransformDCT::get_prime_divisors(15), vec![3, 5]);
        assert_eq!(TransformDCT::get_prime_divisors(100), vec![2, 2, 5, 5]);
        assert_eq!(TransformDCT::get_prime_divisors(2310), vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_get_divisors() {
        assert_eq!(TransformDCT::get_divisors(1), vec![1]);
        assert_eq!(TransformDCT::get_divisors(2), vec![1, 2]);
        assert_eq!(TransformDCT::get_divisors(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(TransformDCT::get_divisors(15), vec![1, 3, 5, 15]);
        assert_eq!(
            TransformDCT::get_divisors(100),
            vec![1, 2, 4, 5, 10, 20, 25, 50, 100]
        );
    }

    #[test]
    fn test_get_smaller_split() {
        assert_eq!(TransformDCT::get_smaller_split(12, 3), 3);
        assert_eq!(TransformDCT::get_smaller_split(12, 4), 4);
        assert_eq!(TransformDCT::get_smaller_split(12, 5), 4);
        assert_eq!(TransformDCT::get_smaller_split(100, 7), 5);
        assert_eq!(TransformDCT::get_smaller_split(100, 26), 25);
        assert_eq!(TransformDCT::get_smaller_split(100, 101), 100);
        assert_eq!(TransformDCT::get_smaller_split(1, 1), 1);
    }

    #[test]
    fn test_edge_cases() {
        assert_eq!(TransformDCT::get_prime_divisors(0), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_divisors(0), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_smaller_split(0, 1), 0);
    }

    #[test]
    fn test_large_numbers() {
        assert_eq!(
            TransformDCT::get_prime_divisors(1000000007),
            vec![1000000007]
        ); // Large prime
        assert_eq!(TransformDCT::get_divisors(1000000007), vec![1, 1000000007]);
        assert_eq!(TransformDCT::get_smaller_split(1000000007, 500000000), 1);
    }

    #[test]
    fn test_dct() {
        let eye = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [0.5000, 0.6533, 0.5000, 0.2706],
            [0.5000, 0.2706, -0.5000, -0.6533],
            [0.5000, -0.2706, -0.5000, 0.6533],
            [0.5000, -0.6533, 0.5000, -0.2706],
        ]);
        let result = TransformDCT::dct(&eye, true);
        assert!(result.allclose(&truth, 1e-4, 1e-8, false));
    }

    fn _2d_float<T: AsRef<[f64]>>(x: &[T]) -> Tensor {
        Tensor::from_slice2(x).to_kind(Kind::Float).to(Device::Cpu)
    }

    fn _2d_int<T: AsRef<[i64]>>(x: &[T]) -> Tensor {
        Tensor::from_slice2(x).to_kind(Kind::Int64).to(Device::Cpu)
    }

    fn _1d_float(x: &[f64]) -> Tensor {
        Tensor::from_slice(x).to_kind(Kind::Float).to(Device::Cpu)
    }

    fn _1d_int(x: &[i64]) -> Tensor {
        Tensor::from_slice(x).to_kind(Kind::Int64).to(Device::Cpu)
    }

    #[test]
    fn test_idct() {
        let eye = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706],
        ]);
        let result = TransformDCT::idct(&eye, true);
        assert!(result.allclose(&truth, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_compress_2d() {
        let r = _2d_float(&[
            [0.1911, 0.4076, 0.1649, 0.8059],
            [0.2803, 0.9381, 0.9071, 0.2573],
            [0.4070, 0.5765, 0.7226, 0.9486],
            [0.0737, 0.7378, 0.1898, 0.2990],
        ]);
        let truth = (
            _2d_int(&[[3, 1], [1, 2], [3, 2], [1, 3]]),
            _2d_float(&[
                [0.8059, 0.4076],
                [0.9381, 0.9071],
                [0.9486, 0.7226],
                [0.7378, 0.2990],
            ]),
            vec![4i64, 4i64],
            4i64,
        );
        let ret = CompressDCT::compress(&r, 2);
        assert_eq!(truth.0, ret.0);
        assert!(truth.1.allclose(&ret.1, 1e-4, 1e-8, false));
        assert_eq!(truth.2, ret.2);
        assert_eq!(4, ret.3);
    }

    #[test]
    fn test_compress_1d() {
        let r = _1d_float(&[
            0.5223, 0.9625, 0.5487, 0.2152, 0.2161, 0.0363, 0.4944, 0.0974,
        ]);
        let truth = (
            _1d_int(&[1, 2]),
            _1d_float(&[0.9625, 0.5487]),
            vec![8i64],
            8i64,
        );
        let ret = CompressDCT::compress(&r, 2);
        assert_eq!(truth.0, ret.0);
        assert!(truth.1.allclose(&ret.1, 1e-4, 1e-8, false));
        assert_eq!(truth.2, ret.2);
        assert_eq!(8, ret.3);
    }

    #[test]
    fn test_decompress_1d() {
        let p = _1d_float(&[0.0]);
        let idx = _1d_int(&[1, 2]);
        let val = _1d_float(&[0.9625, 0.5487]);
        let xshape = vec![8i64];
        let truth = _1d_float(&[
            0.0000, 0.9625, 0.5487, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        ]);
        let ret = CompressDCT::decompress(&idx, &val, &xshape, i64::MAX, p.kind(), p.device());
        assert!(truth.allclose(&ret, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_decompress_2d() {
        let p = _1d_float(&[0.0]);
        let idx = _2d_int(&[[0, 2], [1, 2], [2, 3], [3, 1]]);
        let val = _2d_float(&[
            [0.8988, 0.5175],
            [0.9882, 0.8945],
            [0.8285, 0.8163],
            [0.9093, 0.7600],
        ]);
        let xshape = vec![4i64, 4i64];
        let truth = _2d_float(&[
            [0.8988, 0.0000, 0.5175, 0.0000],
            [0.0000, 0.9882, 0.8945, 0.0000],
            [0.0000, 0.0000, 0.8285, 0.8163],
            [0.0000, 0.7600, 0.0000, 0.9093],
        ]);
        let ret = CompressDCT::decompress(&idx, &val, &xshape, i64::MAX, p.kind(), p.device());
        assert!(truth.allclose(&ret, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_encode_1d() {
        let a = Tensor::arange(8, (Kind::Float, Device::Cpu));
        let truth = _1d_float(&[
            9.8995e+00,
            -6.4423e+00,
            -4.7684e-07,
            -6.7345e-01,
            2.3842e-07,
            -2.0090e-01,
            -1.1921e-07,
            -5.0702e-02,
        ]);
        let ret = TransformDCT::new(vars(vec![a.copy()]), 64)
            .encode(&a)
            .squeeze();
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_encode_2d() {
        let b = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00, 0.0000e+00, -5.9605e-08],
            [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
            [0.0000e+00, -5.9605e-08, 0.0000e+00, 1.0000e+00],
        ]);
        let ret = TransformDCT::new(vars(vec![b.copy()]), 64)
            .encode(&b)
            .squeeze();
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_decode_1d() {
        let a = Tensor::arange(8, (Kind::Float, Device::Cpu));
        let a_ = _2d_float(&[[
            9.8995e+00,
            -6.4423e+00,
            -4.7684e-07,
            -6.7345e-01,
            2.3842e-07,
            -2.0090e-01,
            -1.1921e-07,
            -5.0702e-02,
        ]]);
        let truth = _1d_float(&[
            -2.2352e-07,
            1.0000e+00,
            2.0000e+00,
            3.0000e+00,
            4.0000e+00,
            5.0000e+00,
            6.0000e+00,
            7.0000e+00,
        ]);
        let ret = TransformDCT::new(vars(vec![a]), 64).decode(&a_);
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_decode_2d() {
        let b = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let b_ = _2d_float(&[
            [1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00, 0.0000e+00, -5.9605e-08],
            [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
            [0.0000e+00, -5.9605e-08, 0.0000e+00, 1.0000e+00],
        ])
        .unsqueeze(0)
        .unsqueeze(0);
        let truth = _2d_float(&[
            [1.0000e+00, 1.4901e-08, 4.4703e-08, 4.4703e-08],
            [2.9802e-08, 1.0000e+00, -2.9802e-08, 4.4703e-08],
            [4.4703e-08, -2.9802e-08, 1.0000e+00, 2.9802e-08],
            [4.4703e-08, 4.4703e-08, 1.4901e-08, 1.0000e+00],
        ]);
        let ret = TransformDCT::new(vars(vec![b]), 64).decode(&b_);
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_align_matformer_prefix_grad_expand_gate() {
        let grad = _2d_float(&[[1.0, 2.0], [3.0, 4.0]]);
        let full_shape = vec![4i64, 2i64];
        let aligned =
            align_matformer_prefix_grad("model.layers.0.mlp.gate_proj.weight", &full_shape, grad)
                .unwrap();
        assert_eq!(aligned.size(), full_shape);
        let prefix = aligned.narrow(0, 0, 2);
        let tail = aligned.narrow(0, 2, 2);
        assert!(prefix.allclose(&_2d_float(&[[1.0, 2.0], [3.0, 4.0]]), 1e-6, 1e-6, false));
        assert!(tail.allclose(&_2d_float(&[[0.0, 0.0], [0.0, 0.0]]), 1e-6, 1e-6, false));
    }

    #[test]
    fn test_align_matformer_prefix_grad_slice_down() {
        let grad = _2d_float(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let full_shape = vec![2i64, 2i64];
        let aligned =
            align_matformer_prefix_grad("model.layers.0.mlp.down_proj.weight", &full_shape, grad)
                .unwrap();
        assert_eq!(aligned.size(), full_shape);
        assert!(aligned.allclose(&_2d_float(&[[1.0, 2.0], [5.0, 6.0]]), 1e-6, 1e-6, false));
    }

    #[test]
    fn test_align_matformer_prefix_grad_rejects_non_mlp() {
        let grad = _2d_float(&[[1.0, 2.0]]);
        let full_shape = vec![2i64, 2i64];
        let err = align_matformer_prefix_grad(
            "model.layers.0.self_attn.q_proj.weight",
            &full_shape,
            grad,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PrefixAlignError::UnsupportedParameter(
                "model.layers.0.self_attn.q_proj.weight".to_string()
            )
        );
    }

    #[test]
    fn test_signed_vals_reconstructs_original_sign() {
        let truth = Tensor::from_slice2(&[
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706],
        ])
        .to_kind(Kind::Float)
        .to(Device::Cpu);

        let signed_truth = truth.sign();

        let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(&truth, i64::MAX);
        let signed_sparse_val = sparse_val.sign();

        let decompressed_signed = CompressDCT::decompress(
            &sparse_idx,
            &signed_sparse_val,
            &xshape,
            totalk,
            truth.kind(),
            Device::Cpu,
        );
        assert!(decompressed_signed.equal(&signed_truth));
    }

    #[test]
    fn test_artifical_distro_results_roundtrip() {
        use tch::{Kind, Tensor};

        /// Generates a dummy estimate_val tensor of shape (r0, r1, k), where r is the remainder shape after DCT chunking
        /// r1 can be set to 0 to simulate a 1D DCT
        fn generate_random_estimate_val(r0: i64, r1: i64, k: i64, dtype: Kind) -> Tensor {
            // Warning: only works if dtype bits size is divisible by 8, should always be true for current torch tensors
            // but who knows what would happen one day... fp4?

            let randbytes = match dtype {
                Kind::BFloat16 => 2,
                Kind::Float => 4,
                Kind::Double => 8,
                _ => panic!("Unsupported dtype"),
            };

            // 1D DCT estimates
            let randsize = if r1 == 0 {
                vec![r0, k * randbytes]
            }
            // 2D DCT estimates
            else {
                vec![r0, r1, k * randbytes]
            };

            Tensor::randint(256, &randsize, (Kind::Uint8, tch::Device::Cpu)).view_dtype(dtype)
        }

        /// Generates a dummy indices tensor when given estimate_val. indices are between 0 and s0*s1 (exclusive),
        /// where s0 and s1 is the DCT chunk shape
        /// s1 can be set to 0 to simulate a 1D DCT
        fn generate_random_estimate_idx(val: &Tensor, s0: i64, s1: i64) -> (Tensor, i64) {
            // Note: Some indices will collide, just like real estimates
            // Warning: At the current moment of writing this test, we assume indices must always be int64
            // for correct torch indexing

            // 1D DCT estimates
            let s1 = if s1 == 0 { 1 } else { s1 };

            let max_value = s0 * s1;
            (
                Tensor::randint(max_value, val.size(), (Kind::Int64, tch::Device::Cpu)),
                max_value,
            )
        }

        set_torch_rng_seed();

        let range_r0 = 1..10;
        let range_r1 = 0..10;
        let range_s0 = [1, 7, 512];
        let range_s1 = [1, 4, 64];
        let range_k = [1, 2, 3, 4, 5, 7, 9, 16, 32, 64, 96, 128];
        let range_dtype = [Kind::BFloat16, Kind::Float];

        for (r0, r1, s0, s1, k, d) in
            iproduct!(range_r0, range_r1, range_s0, range_s1, range_k, range_dtype)
        {
            let val = generate_random_estimate_val(r0, r1, k, d);
            let (idx, max_idx_val) = generate_random_estimate_idx(&val, s0, s1);

            let roundtripped_val = Distro::unpack_tensor_sign_from_boolean(
                Distro::quantize_nozeros_tensor_to_boolean_sign(&val),
                val.kind(),
            );

            // we need to make a reference to compare the compression to.
            // this compression should hold Infinity and +0 and some NaNs as 1
            // and -Infinity and -0 and some NaNs as -1
            let val_signed: Tensor = (-2.0 * val.signbit().to_kind(Kind::Float)) + 1.0;
            assert!(val_signed.equal(&roundtripped_val));

            let roundtripped_idx = decompress_idx(max_idx_val, &compress_idx(max_idx_val, &idx));
            assert!(idx.equal(&roundtripped_idx));
        }
    }
    #[test]
    fn test_1bit_matches_non_quant() {
        set_torch_rng_seed();
        let input = Tensor::rand(
            [51, 35, 5, 13, 6],
            (Kind::BFloat16, Device::cuda_if_available()),
        ) - 0.5;
        // ensure no zeros in our ground truth!
        let input = (&input) + (input.sign() + 0.1);

        let quant = Distro::quantize_nozeros_tensor_to_boolean_sign(&input);
        let unquant = Distro::unpack_tensor_sign_from_boolean(quant, input.kind());

        assert!(input.sign().equal(&unquant));
    }

    use crate::{CausalLM, Communicator, EosToks};
    use std::sync::Arc;

    struct NamedVariable {
        name: String,
        tensor: Tensor,
    }

    impl Variable for NamedVariable {
        fn name(&self) -> &str {
            &self.name
        }
        fn local_tensor(&self) -> Tensor {
            self.tensor.shallow_clone()
        }
        fn logical_tensor(&self) -> Tensor {
            self.tensor.shallow_clone()
        }
        fn gather_full_tensor(&self) -> Tensor {
            self.tensor.shallow_clone()
        }
        fn shard_other_tensor_like_me(&self, tensor: Tensor) -> Tensor {
            tensor
        }
        fn full_tensor_shape(&self) -> Vec<i64> {
            self.tensor.size()
        }
        fn is_sharded(&self) -> bool {
            false
        }
        fn zeros_like(&self, name: String) -> Box<dyn Variable> {
            Box::new(NamedVariable {
                name,
                tensor: self.tensor.zeros_like(),
            })
        }
        fn set_grad(&self, tensor: Tensor) {
            self.tensor.grad().copy_(&tensor);
        }
        fn zero_grad(&self) {
            let mut grad = self.tensor.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
        }
    }

    struct OneTensorModel {
        name: String,
        tensor: Tensor,
    }

    impl OneTensorModel {
        fn new(name: &str, tensor: Tensor) -> Self {
            let tensor = tensor.detach().copy().set_requires_grad(true);
            (tensor.zeros_like() * &tensor).sum(Kind::Float).backward();
            Self {
                name: name.to_string(),
                tensor,
            }
        }
    }

    impl CausalLM for OneTensorModel {
        fn forward(
            &self,
            _x: &Tensor,
            _labels: Option<&Tensor>,
            _position_ids: Option<&Tensor>,
            _sequence_lengths: Option<&Vec<Vec<i32>>>,
            _num_logits_to_keep: Option<i64>,
            _loss_scale: Option<f64>,
        ) -> (Option<Tensor>, Option<Tensor>) {
            unimplemented!()
        }
        fn bos_token_id(&self) -> Option<i64> {
            None
        }
        fn eos_token_ids(&self) -> Option<EosToks> {
            None
        }
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn max_context_length(&self) -> usize {
            0
        }
        fn variables(&self) -> StableVariableIterator {
            Box::new(std::iter::once(Box::new(NamedVariable {
                name: self.name.clone(),
                tensor: self.tensor.shallow_clone(),
            }) as Box<dyn Variable>))
        }
        fn communicator(&self) -> Option<Arc<Communicator>> {
            None
        }
        fn prepare_for_training(&self) {}
        fn clip_grad_norm(&self, _max_grad_norm: f64) {}
    }

    const GATE: &str = "model.layers.0.mlp.gate_proj.weight";

    fn peer_result(model: &OneTensorModel) -> DistroResult {
        let mut distro = Distro::new(model, 0.999, 16, 4, 0.0);
        let grad = Tensor::randn(model.tensor.size(), (Kind::Float, Device::Cpu));
        model.tensor.grad().copy_(&grad);
        distro.generate(model, &[], 1e-3, false).remove(0)
    }

    #[test]
    fn test_is_matformer_tier_width() {
        assert!(is_matformer_tier_width(256, 256));
        assert!(is_matformer_tier_width(256, 128));
        assert!(is_matformer_tier_width(128, 256));
        assert!(is_matformer_tier_width(341, 170));
        assert!(!is_matformer_tier_width(256, 3));
        assert!(!is_matformer_tier_width(256, 999));
        assert!(!is_matformer_tier_width(256, 0));
        assert!(!is_matformer_tier_width(-256, 128));
    }

    #[test]
    fn test_align_matformer_prefix_grad_rejects_non_tier_widths() {
        for rows in [3, 999] {
            let grad = Tensor::ones([rows, 8], (Kind::Float, Device::Cpu));
            let err = align_matformer_prefix_grad(GATE, &[256, 8], grad).unwrap_err();
            assert!(matches!(err, PrefixAlignError::ShapeMismatch { .. }));
        }
    }

    #[test]
    fn test_apply_drops_invalid_results_and_uses_valid_ones() {
        let initial = Tensor::randn([256, 8], (Kind::Float, Device::Cpu)) * 0.02;
        let make_model = || OneTensorModel::new(GATE, initial.copy());
        let good = peer_result(&make_model());

        // Each of these results would panic or corrupt the update without validation.
        let mut wrong_width = good.clone();
        wrong_width.xshape[0] = 3;
        let mut wrong_totalk = good.clone();
        wrong_totalk.totalk = 7;
        let mut wrong_values = good.clone();
        wrong_values.sparse_val = good.sparse_val.narrow(0, 0, 2);
        let mut index_out_of_range = good.clone();
        index_out_of_range.sparse_idx = good.sparse_idx.full_like(255);

        for bad in [wrong_width, wrong_totalk, wrong_values, index_out_of_range] {
            let expected = make_model();
            let mut distro = Distro::new(&expected, 0.999, 16, 4, 0.0);
            distro.apply(&expected, &[vec![good.clone()]], 1e-3);

            let actual = make_model();
            let mut distro = Distro::new(&actual, 0.999, 16, 4, 0.0);
            distro.apply(&actual, &[vec![good.clone()], vec![bad.clone()]], 1e-3);

            assert!(actual.tensor.detach().equal(&expected.tensor.detach()));
            assert!(
                !actual.tensor.detach().equal(&initial),
                "the valid result must apply"
            );

            // A bad result alone leaves the weights unchanged.
            let alone = make_model();
            let mut distro = Distro::new(&alone, 0.999, 16, 4, 0.0);
            distro.apply(&alone, &[vec![bad]], 1e-3);
            assert!(alone.tensor.detach().equal(&initial));
        }
    }

    #[test]
    fn test_1bit_zero_values_have_no_sign_bias() {
        set_torch_rng_seed();
        let input = Tensor::randn([20000], (Kind::Float, Device::Cpu));
        let mut zero_part = input.narrow(0, 0, 10000);
        let _ = zero_part.zero_();

        let quantized = Distro::quantize_tensor_to_boolean_sign(&input);

        // Nonzero values keep their exact sign.
        let nonzero_part = input.narrow(0, 10000, 10000);
        assert!(
            quantized
                .narrow(0, 10000, 10000)
                .equal(&nonzero_part.signbit())
        );

        let negative_fraction = quantized
            .narrow(0, 0, 10000)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        assert!(
            (negative_fraction - 0.5).abs() < 0.03,
            "zeros must not have a fixed sign: {negative_fraction}"
        );

        let no_zeros = input.g_add_scalar(1000.0);
        assert!(Distro::quantize_tensor_to_boolean_sign(&no_zeros).equal(&no_zeros.signbit()));
    }

    /// Float weights with the BF16 grid must equal BF16 weights after every operation.
    #[test]
    fn test_float_with_bf16_grid_equals_bf16_weights() {
        let initial = Tensor::randn([128, 16], (Kind::Float, Device::Cpu)) * 0.05;

        for lr in [4e-4, 4e-5] {
            let float_model = OneTensorModel::new(GATE, initial.to_kind(Kind::Float));
            let bf16_model = OneTensorModel::new(GATE, initial.to_kind(Kind::BFloat16));
            let mut float_distro = Distro::new(&float_model, 0.999, 16, 4, 0.0);
            let mut bf16_distro = Distro::new(&bf16_model, 0.999, 16, 4, 0.0);
            float_distro.snap_to_bf16 = true;
            float_distro.snap_variables(&float_model);
            assert_same_weights(&float_model, &bf16_model, -1, "start");

            for round in 0..200 {
                let source = OneTensorModel::new(GATE, initial.copy());
                let mut result = peer_result(&source);
                result.sparse_val = Distro::quantize_tensor_to_boolean_sign(&result.sparse_val);
                let results = vec![vec![result]];

                // The same local error term gives the same lookahead on both clients.
                let sign = (Tensor::rand([128, 16], (Kind::Float, Device::Cpu)) - 0.5).sign();
                for (distro, kind) in [
                    (&mut float_distro, Kind::Float),
                    (&mut bf16_distro, Kind::BFloat16),
                ] {
                    let _no_grad = tch::no_grad_guard();
                    distro.state[0]
                        .delta
                        .logical_tensor()
                        .copy_(&sign.to_kind(kind));
                }

                float_distro.error_correction(&float_model, lr);
                bf16_distro.error_correction(&bf16_model, lr);
                assert_same_weights(&float_model, &bf16_model, round, "lookahead");

                float_distro.undo_lookahead(&float_model);
                bf16_distro.undo_lookahead(&bf16_model);
                assert_same_weights(&float_model, &bf16_model, round, "lookahead undo");

                float_distro.apply(&float_model, &results, lr);
                bf16_distro.apply(&bf16_model, &results, lr);
                assert_same_weights(&float_model, &bf16_model, round, "apply");
            }
        }
    }

    /// Two clients with different local errors must return to the same weights.
    #[test]
    fn test_lookahead_undo_restores_exact_weights() {
        // Small weights and several step sizes reach every rounding case in BF16.
        let initial = Tensor::randn([256, 64], (Kind::Float, Device::Cpu)) * 0.02;
        for kind in [Kind::BFloat16, Kind::Float] {
            for lr in [1.3e-5, 4e-5, 1.7e-4, 4e-4, 2.9e-3] {
                let model = OneTensorModel::new(GATE, initial.to_kind(kind));
                let start = model.tensor.detach().copy();
                let mut distro = Distro::new(&model, 0.999, 64, 8, 0.0);
                for _ in 0..50 {
                    let sign = (Tensor::rand([256, 64], (Kind::Float, Device::Cpu)) - 0.5).sign();
                    {
                        let _no_grad = tch::no_grad_guard();
                        distro.state[0]
                            .delta
                            .logical_tensor()
                            .copy_(&sign.to_kind(kind));
                    }
                    distro.error_correction(&model, lr);
                    assert!(
                        !model.tensor.detach().equal(&start),
                        "lookahead must move weights"
                    );
                    distro.undo_lookahead(&model);
                    assert!(
                        model.tensor.detach().equal(&start),
                        "kind {kind:?}, lr {lr}"
                    );
                }
            }
        }
    }

    /// A step that stops early must not leave its lookahead in the weights.
    #[test]
    fn test_second_lookahead_does_not_stack_on_a_cancelled_step() {
        let initial = Tensor::randn([64, 16], (Kind::Float, Device::Cpu)) * 0.02;
        let model = OneTensorModel::new(GATE, initial.to_kind(Kind::BFloat16));
        let start = model.tensor.detach().copy();
        let mut distro = Distro::new(&model, 0.999, 16, 4, 0.0);
        let sign = (Tensor::rand([64, 16], (Kind::Float, Device::Cpu)) - 0.5).sign();
        {
            let _no_grad = tch::no_grad_guard();
            distro.state[0]
                .delta
                .logical_tensor()
                .copy_(&sign.to_kind(Kind::BFloat16));
        }
        distro.error_correction(&model, 4e-4);
        let once = model.tensor.detach().copy();
        distro.error_correction(&model, 4e-4);
        assert!(model.tensor.detach().equal(&once));
        distro.undo_lookahead(&model);
        distro.undo_lookahead(&model);
        assert!(model.tensor.detach().equal(&start));
    }

    fn assert_same_weights(a: &OneTensorModel, b: &OneTensorModel, round: i64, stage: &str) {
        let a = a.tensor.detach().to_kind(Kind::Float);
        let b = b.tensor.detach().to_kind(Kind::Float);
        let differing = a
            .ne_tensor(&b)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        assert_eq!(differing, 0.0, "round {round}, after {stage}");
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(
            extract_layer_index("model.layers.0.mlp.gate_proj.weight"),
            Some(0)
        );
        assert_eq!(
            extract_layer_index("model.layers.5.mlp.down_proj.weight"),
            Some(5)
        );
        assert_eq!(
            extract_layer_index("model.layers.12.self_attn.q_proj.weight"),
            Some(12)
        );
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    #[test]
    fn test_align_matformer_sparse_grad_gate_proj() {
        // Test scattering sparse gradients for gate_proj
        // Simulates prefix [0,1] + helper indices [4,6]
        let grad = _2d_float(&[
            [1.0, 2.0], // idx 0
            [3.0, 4.0], // idx 1
            [5.0, 6.0], // idx 4
            [7.0, 8.0], // idx 6
        ]);
        let full_shape = vec![8i64, 2i64];
        let indices: Vec<i64> = vec![0, 1, 4, 6];

        let aligned = align_matformer_sparse_grad(
            "model.layers.0.mlp.gate_proj.weight",
            &full_shape,
            grad,
            &indices,
        )
        .unwrap();

        assert_eq!(aligned.size(), full_shape);

        // Check row 0
        let row0 = aligned.narrow(0, 0, 1);
        assert!(row0.allclose(&_2d_float(&[[1.0, 2.0]]), 1e-6, 1e-6, false));

        // Check row 1
        let row1 = aligned.narrow(0, 1, 1);
        assert!(row1.allclose(&_2d_float(&[[3.0, 4.0]]), 1e-6, 1e-6, false));

        // Check rows 2,3 are zeros
        let row2 = aligned.narrow(0, 2, 1);
        let row3 = aligned.narrow(0, 3, 1);
        assert!(row2.allclose(&_2d_float(&[[0.0, 0.0]]), 1e-6, 1e-6, false));
        assert!(row3.allclose(&_2d_float(&[[0.0, 0.0]]), 1e-6, 1e-6, false));

        // Check row 4
        let row4 = aligned.narrow(0, 4, 1);
        assert!(row4.allclose(&_2d_float(&[[5.0, 6.0]]), 1e-6, 1e-6, false));

        // Check row 5 is zeros
        let row5 = aligned.narrow(0, 5, 1);
        assert!(row5.allclose(&_2d_float(&[[0.0, 0.0]]), 1e-6, 1e-6, false));

        // Check row 6
        let row6 = aligned.narrow(0, 6, 1);
        assert!(row6.allclose(&_2d_float(&[[7.0, 8.0]]), 1e-6, 1e-6, false));
    }

    #[test]
    fn test_align_matformer_sparse_grad_down_proj() {
        // Test scattering sparse gradients for down_proj (columns)
        // Simulates prefix [0,1] + helper indices [4,6]
        let grad = _2d_float(&[
            [1.0, 2.0, 5.0, 7.0], // row 0, cols [0,1,4,6]
            [3.0, 4.0, 6.0, 8.0], // row 1, cols [0,1,4,6]
        ]);
        let full_shape = vec![2i64, 8i64];
        let indices: Vec<i64> = vec![0, 1, 4, 6];

        let aligned = align_matformer_sparse_grad(
            "model.layers.0.mlp.down_proj.weight",
            &full_shape,
            grad,
            &indices,
        )
        .unwrap();

        assert_eq!(aligned.size(), full_shape);

        // Check columns 0,1,4,6 have values, others are zeros
        let col0 = aligned.narrow(1, 0, 1);
        assert!(col0.allclose(&_2d_float(&[[1.0], [3.0]]), 1e-6, 1e-6, false));

        let col4 = aligned.narrow(1, 4, 1);
        assert!(col4.allclose(&_2d_float(&[[5.0], [6.0]]), 1e-6, 1e-6, false));

        let col2 = aligned.narrow(1, 2, 1);
        assert!(col2.allclose(&_2d_float(&[[0.0], [0.0]]), 1e-6, 1e-6, false));
    }

    #[test]
    fn test_align_matformer_sparse_grad_rejects_non_mlp() {
        let grad = _2d_float(&[[1.0, 2.0]]);
        let full_shape = vec![2i64, 2i64];
        let indices: Vec<i64> = vec![0];
        let err = align_matformer_sparse_grad(
            "model.layers.0.self_attn.q_proj.weight",
            &full_shape,
            grad,
            &indices,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PrefixAlignError::UnsupportedParameter(
                "model.layers.0.self_attn.q_proj.weight".to_string()
            )
        );
    }
}

// #[cfg(test)]
// #[cfg(feature = "parallelism")]
// mod tp_tests {
//     use super::*;
//     use crate::tensor_parallelism::CommunicatorId;
//     use crate::{
//         set_suggested_env_vars, set_torch_rng_seed, unsharded_cpu_variables, ColumnParallelLinear,
//     };
//     use std::sync::{Arc, Barrier, Mutex};
//     use tch::{nn, Device, Kind, Tensor, CNCCL};

//     const TEST_LR: f64 = 0.01;
//     const COMPRESSION_DECAY: f64 = 0.99;
//     const COMPRESSION_CHUNK: i64 = 64;
//     const COMPRESSION_TOPK: i64 = 16;
//     const WEIGHT_DECAY: f64 = 0.0;
//     const NUM_STEPS: u32 = 10;

//     fn run_parallel_test<F>(world_size: usize, test_fn: F)
//     where
//         F: Fn(Arc<CommunicatorId>, usize, Arc<Barrier>, Device) -> anyhow::Result<()>
//             + Send
//             + Sync
//             + 'static,
//     {
//         if !tch::utils::has_cuda() || tch::Cuda::device_count() < world_size as i64 {
//             println!(
//                 "Skipping parallel test: requires CUDA and {} GPUs.",
//                 world_size
//             );
//             return;
//         }

//         let barrier = Arc::new(Barrier::new(world_size));
//         let comm_id = Arc::new(CommunicatorId::new());
//         let test_fn = Arc::new(test_fn);

//         let threads: Vec<_> = (0..world_size)
//             .map(|rank| {
//                 let barrier = barrier.clone();
//                 let comm_id = comm_id.clone();
//                 let test_fn = test_fn.clone();
//                 let device = Device::Cuda(rank);

//                 std::thread::spawn(move || {
//                     test_fn(comm_id, rank, barrier, device).unwrap();
//                 })
//             })
//             .collect();

//         for thread in threads {
//             thread.join().expect("Thread panicked");
//         }
//     }

//     // Helper to run a simple training loop step with Distro
//     fn run_distro_step(
//         step_num: u32,
//         model: &dyn nn::Module,
//         input: &Tensor,
//         target: &Tensor,
//         optimizer: &mut Distro,
//         lr: f64,
//         all_rank_results: Arc<Mutex<HashMap<u32, Vec<Vec<DistroResult>>>>>,
//         _rank: usize,
//         _world_size: usize,
//         _comm: &Option<Arc<Communicator>>,
//         barrier: &Arc<Barrier>,
//     ) -> anyhow::Result<Vec<DistroResult>> {
//         optimizer.zero_grad();
//         barrier.wait();

//         let output = model.forward(input);
//         let loss = output.mse_loss(target, tch::Reduction::Mean);
//         barrier.wait();

//         loss.backward();
//         barrier.wait();

//         let current_step_results = optimizer.generate(&vec![], 0.0, lr, false);
//         barrier.wait();

//         {
//             let mut results_map = all_rank_results.lock().unwrap();
//             let step_results = results_map.entry(step_num).or_default();
//             step_results.push(current_step_results.clone());
//         }
//         barrier.wait();

//         let results_to_apply = {
//             let results_map = all_rank_results.lock().unwrap();
//             results_map
//                 .get(&step_num)
//                 .expect(&format!("missing results for current step {step_num}"))
//                 .clone()
//         };
//         barrier.wait();

//         optimizer.apply(&results_to_apply, lr);
//         barrier.wait();

//         Ok(current_step_results)
//     }

//     #[test]
//     fn test_distro_tp_consistency() -> anyhow::Result<()> {
//         const WORLD_SIZE: usize = 8;
//         const BATCH_SIZE: i64 = 4;
//         const SEQ_LEN: i64 = 32;
//         const IN_FEATURES: i64 = 128;
//         const OUT_FEATURES: i64 = 256;

//         set_suggested_env_vars();
//         set_torch_rng_seed();

//         let device = Device::cuda_if_available();
//         if !device.is_cuda() {
//             println!("Skipping TP test as CUDA is not available.");
//             return Ok(());
//         }

//         let input = Arc::new(Mutex::new(Tensor::randn(
//             &[BATCH_SIZE, SEQ_LEN, IN_FEATURES],
//             (Kind::Float, device),
//         )));
//         let target = Arc::new(Mutex::new(Tensor::randn(
//             &[BATCH_SIZE, SEQ_LEN, OUT_FEATURES],
//             (Kind::Float, device),
//         )));

//         // single gpu
//         let (final_weights_non_tp, linear_layer_weights) = {
//             let vs_non_tp = nn::VarStore::new(device);
//             let model_non_tp = nn::linear(
//                 vs_non_tp.root() / "layer",
//                 IN_FEATURES,
//                 OUT_FEATURES,
//                 nn::LinearConfig {
//                     bias: false,
//                     ..Default::default()
//                 },
//             );
//             let original_weights = model_non_tp.ws.copy();

//             let mut optimizer_non_tp = Distro::new(
//                 &vs_non_tp,
//                 COMPRESSION_DECAY,
//                 COMPRESSION_CHUNK,
//                 COMPRESSION_TOPK,
//                 WEIGHT_DECAY,
//                 None,
//             );

//             let dummy_barrier = Arc::new(Barrier::new(1));
//             let dummy_all_results = Arc::new(Mutex::new(HashMap::new()));

//             for step in 0..NUM_STEPS {
//                 let _ = run_distro_step(
//                     step,
//                     &model_non_tp,
//                     &input.lock().unwrap(),
//                     &target.lock().unwrap(),
//                     &mut optimizer_non_tp,
//                     TEST_LR,
//                     dummy_all_results.clone(),
//                     0,
//                     1,
//                     &None,
//                     &dummy_barrier,
//                 )?;
//             }

//             let mut final_weights = HashMap::new();
//             for (name, tensor) in vs_non_tp.variables() {
//                 final_weights.insert(name, tensor.detach().to_device(Device::Cpu));
//             }
//             (final_weights, original_weights)
//         };

//         let final_weights_tp_rank0 = Arc::new(Mutex::new(HashMap::new()));
//         let all_rank_results_tp: Arc<Mutex<HashMap<u32, Vec<Vec<DistroResult>>>>> =
//             Arc::new(Mutex::new(HashMap::new()));

//         {
//             let final_weights_tp_rank0 = final_weights_tp_rank0.clone();
//             let all_rank_results_tp = all_rank_results_tp.clone();
//             let ref_linear_weights = Arc::new(Mutex::new(linear_layer_weights));

//             run_parallel_test(
//                 WORLD_SIZE,
//                 move |comm_id, rank, barrier, device| -> anyhow::Result<()> {
//                     let vs_tp = nn::VarStore::new(device);
//                     let comm = Arc::new(CNCCL::new(
//                         comm_id.clone(),
//                         rank as i64,
//                         WORLD_SIZE as i64,
//                         device,
//                     )?);

//                     let mut model_tp = ColumnParallelLinear::new(
//                         vs_tp.root() / "layer",
//                         IN_FEATURES,
//                         OUT_FEATURES,
//                         false,
//                         true,
//                         Some(comm.clone()),
//                     );

//                     let (input, target) = {
//                         let _no_grad = tch::no_grad_guard();
//                         model_tp.linear.ws.copy_(&tensor_shard(
//                             &ref_linear_weights.lock().unwrap(),
//                             &Shard {
//                                 dim: 0,
//                                 rank,
//                                 world_size: WORLD_SIZE,
//                             },
//                         ));

//                         barrier.wait();

//                         comm.group_start().unwrap();
//                         if rank == 0 {
//                             let input = input.lock().unwrap();
//                             for i in 0..WORLD_SIZE {
//                                 comm.send(&[input.as_ref()], i as i64).unwrap();
//                             }
//                         }
//                         let input = Tensor::zeros(
//                             &[BATCH_SIZE, SEQ_LEN, IN_FEATURES],
//                             (Kind::Float, device),
//                         );
//                         comm.recv(&[input.shallow_clone()], 0).unwrap();
//                         comm.group_end().unwrap();

//                         barrier.wait();

//                         comm.group_start().unwrap();
//                         if rank == 0 {
//                             let target = target.lock().unwrap();
//                             for i in 0..WORLD_SIZE {
//                                 comm.send(&[target.as_ref()], i as i64).unwrap();
//                             }
//                         }
//                         let target = Tensor::zeros(
//                             &[BATCH_SIZE, SEQ_LEN, OUT_FEATURES],
//                             (Kind::Float, device),
//                         );
//                         comm.recv(&[target.shallow_clone()], 0).unwrap();
//                         comm.group_end().unwrap();

//                         barrier.wait();

//                         (input, target)
//                     };

//                     let mut optimizer_tp = Distro::new(
//                         &vs_tp,
//                         COMPRESSION_DECAY,
//                         COMPRESSION_CHUNK,
//                         COMPRESSION_TOPK,
//                         WEIGHT_DECAY,
//                         Some(comm.clone()),
//                     );

//                     for step in 0..NUM_STEPS {
//                         let current_rank_results = run_distro_step(
//                             step,
//                             &model_tp,
//                             &input,
//                             &target,
//                             &mut optimizer_tp,
//                             TEST_LR,
//                             all_rank_results_tp.clone(),
//                             rank,
//                             WORLD_SIZE,
//                             &Some(comm.clone()),
//                             &barrier,
//                         )?;
//                         let _ = current_rank_results;
//                         barrier.wait();
//                     }

//                     let unsharded_vars = unsharded_cpu_variables(&vs_tp, Some(comm.clone()))?;
//                     if rank == 0 {
//                         *final_weights_tp_rank0.lock().unwrap() = unsharded_vars;
//                     }

//                     Ok(())
//                 },
//             );
//         }

//         let final_weights_tp = final_weights_tp_rank0.lock().unwrap();

//         assert_eq!(
//             final_weights_non_tp.len(),
//             final_weights_tp.len(),
//             "Number of parameters differs between TP and non-TP runs."
//         );

//         for (name, non_tp_tensor) in &final_weights_non_tp {
//             let tp_tensor = final_weights_tp
//                 .get(name)
//                 .ok_or_else(|| anyhow::anyhow!("Parameter '{}' missing in TP results", name))?;

//             assert_eq!(
//                 non_tp_tensor.size(),
//                 tp_tensor.size(),
//                 "Shape mismatch for parameter '{}': Non-TP {:?}, TP {:?}",
//                 name,
//                 non_tp_tensor.size(),
//                 tp_tensor.size()
//             );

//             assert!(
//                 non_tp_tensor.allclose(tp_tensor, 1e-5, 1e-4, false),
//                 "Parameter '{}' differs significantly between TP and non-TP runs.\nNon-TP:\n{}\nTP:\n{}", name, non_tp_tensor, tp_tensor
//             );
//         }

//         Ok(())
//     }
// }
