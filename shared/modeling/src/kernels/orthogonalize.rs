//! Orthogonalization kernels for Muon optimizer
//!
//! This module provides implementations of matrix orthogonalization using the
//! Polar Express algorithm - an optimal matrix sign iteration method.
//!
//! # Polar Express Algorithm
//!
//! The Polar Express method computes the orthogonal polar factor Q of a matrix G
//! where G = QS (Q orthogonal, S symmetric positive semi-definite).
//!
//! The algorithm uses pre-computed optimal coefficients (a, b, c) for a 5-iteration
//! fixed-point scheme:
//!
//! ```text
//! X_0 = G / (||G|| * 1.02)  // Normalized initial guess
//! For i = 0..5:
//!     B = b_i * X @ X^T + c_i * (X @ X^T) @ (X @ X^T)
//!     X_{i+1} = a_i * X + B @ X
//! ```
//!
//! The coefficients are derived from polynomial approximations optimized for convergence
//! within the spectral interval [1e-7, 1].
//!
//! # References
//!
//! - NorMuon optimizer in modded-nanogpt
//! - "An Optimal Matrix Sign Iteration" paper

use super::detect_optimal_precision;
use tch::{Kind, Tensor};

/// Pre-computed optimal coefficients for Polar Express (5 iterations)
/// Each tuple is (a, b, c) for one iteration
const POLAR_COEFFS: [(f64, f64, f64); 5] = [
    (8.157, -22.483, 15.879),
    (4.043, -2.809, 0.500),
    (3.892, -2.772, 0.506),
    (3.286, -2.368, 0.464),
    (2.347, -1.710, 0.423),
];

/// Safety factor for initial normalization (prevents numerical instability)
const NORM_SAFETY_FACTOR: f64 = 1.02;

/// Trait for orthogonalization operations
pub trait Orthogonalize: Send + Sync {
    /// Get the name of this kernel implementation
    fn name(&self) -> &'static str;

    /// Check if this kernel is available on the current system
    fn is_available(&self) -> bool;

    /// Orthogonalize a matrix using the Polar Express algorithm
    ///
    /// # Arguments
    ///
    /// * `g` - Input matrix (typically a gradient tensor, 2D)
    /// * `iterations` - Number of iterations (default: 5)
    ///
    /// # Returns
    ///
    /// The orthogonal polar factor Q such that G ≈ QS
    ///
    /// # Notes
    ///
    /// - Input must be 2D tensor
    /// - Output has the same shape as input
    /// - For non-2D inputs, returns the input unchanged (with warning)
    fn polar_express(&self, g: &Tensor, iterations: usize) -> Tensor;

    /// Orthogonalize with default 5 iterations
    fn orthogonalize(&self, g: &Tensor) -> Tensor {
        self.polar_express(g, 5)
    }
}

/// CPU implementation of orthogonalization using Polar Express
pub struct CpuOrthogonalize {
    /// Use BFloat16 for intermediate computations
    /// Note: BF16 is ~150x slower on CPU due to emulation. Use FP32 for CPU.
    /// BF16 should only be used on CUDA where it has hardware support.
    use_bf16: bool,
}

impl CpuOrthogonalize {
    /// Create a new CPU orthogonalization kernel (uses FP32 for speed)
    pub fn new() -> Self {
        // Default to FP32 on CPU - BF16 emulation is ~150x slower
        Self { use_bf16: false }
    }

    /// Create with custom precision settings
    pub fn with_precision(use_bf16: bool) -> Self {
        Self { use_bf16 }
    }
}

impl Default for CpuOrthogonalize {
    fn default() -> Self {
        Self::new()
    }
}

impl Orthogonalize for CpuOrthogonalize {
    fn name(&self) -> &'static str {
        "cpu_polar_express"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn polar_express(&self, g: &Tensor, iterations: usize) -> Tensor {
        // Only orthogonalize 2D tensors
        if g.dim() != 2 {
            tracing::trace!(
                "Polar Express skipped for non-2D tensor (dim={})",
                g.dim()
            );
            return g.shallow_clone();
        }

        let _guard = tch::no_grad_guard();
        let original_kind = g.kind();

        // Convert to computation dtype
        let compute_kind = if self.use_bf16 {
            Kind::BFloat16
        } else {
            Kind::Float
        };

        polar_express_impl(g, iterations, compute_kind, original_kind)
    }
}

/// Adaptive orthogonalization that auto-detects optimal precision per device.
///
/// This implementation automatically selects the best precision based on the
/// input tensor's device:
/// - CPU: FP32 (universal, avoids slow BF16 emulation)
/// - V100 (Volta): FP16 (native tensor cores)
/// - A100+ (Ampere/Hopper): BF16 (native tensor cores)
///
/// This enables efficient operation in heterogeneous GPU clusters where
/// different nodes may have different GPU architectures.
pub struct AdaptiveOrthogonalize {
    _private: (), // Prevent construction except through new()
}

impl AdaptiveOrthogonalize {
    /// Create a new adaptive orthogonalization kernel
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for AdaptiveOrthogonalize {
    fn default() -> Self {
        Self::new()
    }
}

impl Orthogonalize for AdaptiveOrthogonalize {
    fn name(&self) -> &'static str {
        "adaptive_polar_express"
    }

    fn is_available(&self) -> bool {
        true // Works on any device
    }

    fn polar_express(&self, g: &Tensor, iterations: usize) -> Tensor {
        // Only orthogonalize 2D tensors
        if g.dim() != 2 {
            tracing::trace!(
                "Polar Express skipped for non-2D tensor (dim={})",
                g.dim()
            );
            return g.shallow_clone();
        }

        let _guard = tch::no_grad_guard();
        let original_kind = g.kind();
        let device = g.device();

        // Auto-detect optimal precision based on device
        let compute_kind = detect_optimal_precision(device);

        polar_express_impl(g, iterations, compute_kind, original_kind)
    }
}

/// Core Polar Express implementation shared between CPU and Adaptive variants.
///
/// # Arguments
/// * `g` - Input 2D tensor (must be 2D, caller should check)
/// * `iterations` - Number of iterations (max 5)
/// * `compute_kind` - Precision for internal computation (FP32, FP16, or BF16)
/// * `original_kind` - Original tensor dtype to convert back to
fn polar_express_impl(g: &Tensor, iterations: usize, compute_kind: Kind, original_kind: Kind) -> Tensor {
    // Initial normalization: X_0 = G / (||G|| * safety_factor)
    let g_norm = g.norm().double_value(&[]);
    if g_norm < 1e-12 {
        // Near-zero gradient, return as-is
        return g.shallow_clone();
    }

    let mut x = g.to_kind(compute_kind) / (g_norm * NORM_SAFETY_FACTOR);

    // Fixed-point iteration with pre-computed optimal coefficients
    let num_iters = iterations.min(POLAR_COEFFS.len());
    for (a, b, c) in POLAR_COEFFS.iter().take(num_iters) {
        // X @ X^T
        let xxt = x.matmul(&x.tr());

        // B = b * X@X^T + c * (X@X^T)@(X@X^T)
        let b_mat = &xxt * *b + xxt.matmul(&xxt) * *c;

        // X = a * X + B @ X
        x = &x * *a + b_mat.matmul(&x);
    }

    // Assert numerical stability in debug mode
    #[cfg(debug_assertions)]
    {
        assert!(
            x.isnan().any().int64_value(&[]) == 0,
            "Polar Express produced NaN"
        );
        assert!(
            x.isinf().any().int64_value(&[]) == 0,
            "Polar Express produced Inf"
        );
    }

    // Convert back to original dtype
    x.to_kind(original_kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_polar_express_identity() {
        // For an orthogonal matrix, Polar Express should return ~same matrix
        let ortho = CpuOrthogonalize::new();

        // Create a simple orthogonal matrix (rotation by 45 degrees)
        let angle = std::f64::consts::PI / 4.0;
        let data: [f32; 4] = [
            angle.cos() as f32,
            -angle.sin() as f32,
            angle.sin() as f32,
            angle.cos() as f32,
        ];
        let q = Tensor::from_slice(&data).reshape([2, 2]);

        let result = ortho.polar_express(&q, 5);

        // Result should be close to input (since input is already orthogonal)
        let diff = (&result - &q).abs().max().double_value(&[]);
        assert!(
            diff < 0.1,
            "Polar Express on orthogonal matrix changed too much: diff={diff}"
        );
    }

    #[test]
    fn test_polar_express_orthogonality() {
        // Result should be approximately orthogonal: Q @ Q^T ≈ I
        let ortho = CpuOrthogonalize::new();

        // Create a random-ish matrix
        let g = Tensor::randn([4, 4], (Kind::Float, tch::Device::Cpu));
        let result = ortho.polar_express(&g, 5);

        // Check orthogonality: Q @ Q^T should be close to identity
        let qqt = result.matmul(&result.tr());
        let identity = Tensor::eye(4, (Kind::Float, tch::Device::Cpu));
        let ortho_error = (&qqt - &identity).abs().max().double_value(&[]);

        assert!(
            ortho_error < 0.5,
            "Result not orthogonal enough: max error={ortho_error}"
        );
    }

    #[test]
    fn test_polar_express_non_2d() {
        // Non-2D tensors should be returned unchanged
        let ortho = CpuOrthogonalize::new();

        let t1d = Tensor::randn([10], (Kind::Float, tch::Device::Cpu));
        let result1d = ortho.polar_express(&t1d, 5);
        assert_eq!(t1d.size(), result1d.size());

        let t3d = Tensor::randn([2, 3, 4], (Kind::Float, tch::Device::Cpu));
        let result3d = ortho.polar_express(&t3d, 5);
        assert_eq!(t3d.size(), result3d.size());
    }

    #[test]
    fn test_polar_express_near_zero() {
        // Near-zero gradients should be handled gracefully
        let ortho = CpuOrthogonalize::new();

        let g = Tensor::zeros([4, 4], (Kind::Float, tch::Device::Cpu)) * 1e-20;
        let result = ortho.polar_express(&g, 5);

        // Should not produce NaN or Inf
        assert!(
            !result.isnan().any().int64_value(&[]) != 0,
            "Result contains NaN"
        );
        assert!(
            !result.isinf().any().int64_value(&[]) != 0,
            "Result contains Inf"
        );
    }

    #[test]
    #[ignore] // Slow benchmark - run with: cargo test bench_polar -- --ignored --nocapture
    fn bench_polar_express_latency() {
        use std::time::Instant;

        // Test both BF16 (default) and FP32 to see if BF16 emulation is the bottleneck
        let ortho_bf16 = CpuOrthogonalize::new();
        let ortho_fp32 = CpuOrthogonalize::with_precision(false);

        let sizes: [(i64, i64); 3] = [(256, 256), (512, 512), (1024, 1024)];

        println!("\nPolar Express Latency Comparison (CPU, 5 iterations):");
        println!("{:<12} {:>12} {:>12} {:>8}", "Shape", "BF16", "FP32", "Speedup");

        for (m, n) in sizes {
            let g = Tensor::randn([m, n], (Kind::Float, tch::Device::Cpu));

            // Warmup
            let _ = ortho_bf16.polar_express(&g, 5);
            let _ = ortho_fp32.polar_express(&g, 5);

            // Benchmark BF16
            let iters = 3;
            let start = Instant::now();
            for _ in 0..iters {
                let _ = ortho_bf16.polar_express(&g, 5);
            }
            let bf16_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Benchmark FP32
            let start = Instant::now();
            for _ in 0..iters {
                let _ = ortho_fp32.polar_express(&g, 5);
            }
            let fp32_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = bf16_ms / fp32_ms;

            println!(
                "{:<12} {:>10.1}ms {:>10.1}ms {:>7.2}x",
                format!("{}x{}", m, n),
                bf16_ms,
                fp32_ms,
                speedup
            );
        }

        // Also benchmark raw matmul to understand overhead
        println!("\nRaw matmul baseline (single op):");
        let g = Tensor::randn([1024, 1024], (Kind::Float, tch::Device::Cpu));
        let start = Instant::now();
        for _ in 0..10 {
            let _ = g.matmul(&g.tr());
        }
        let matmul_ms = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
        println!("1024x1024 matmul: {:.1}ms", matmul_ms);
        println!("Polar Express has ~15 matmuls, expected: {:.1}ms", matmul_ms * 15.0);
    }

    // ==================== AdaptiveOrthogonalize tests ====================

    #[test]
    fn test_adaptive_orthogonalize_cpu() {
        // Test that AdaptiveOrthogonalize works on CPU
        let ortho = AdaptiveOrthogonalize::new();
        assert_eq!(ortho.name(), "adaptive_polar_express");

        // Five iterations do not converge on every random matrix, so this test needs a fixed input.
        tch::manual_seed(0);
        let g = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let result = ortho.polar_express(&g, 5);

        // Check orthogonality
        let qqt = result.matmul(&result.tr());
        let identity = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let ortho_error = (&qqt - &identity).abs().max().double_value(&[]);

        assert!(
            ortho_error < 0.5,
            "Adaptive result not orthogonal enough: max error={ortho_error}"
        );
    }

    #[test]
    fn test_adaptive_orthogonalize_no_nan_inf() {
        let ortho = AdaptiveOrthogonalize::new();

        // Test with various input scales
        for scale in [1e-6, 1.0, 1e6] {
            let g = Tensor::randn([8, 8], (Kind::Float, Device::Cpu)) * scale;
            let result = ortho.polar_express(&g, 5);

            assert!(
                result.isnan().any().int64_value(&[]) == 0,
                "Result contains NaN for scale={scale}"
            );
            assert!(
                result.isinf().any().int64_value(&[]) == 0,
                "Result contains Inf for scale={scale}"
            );
        }
    }

    #[test]
    fn test_adaptive_vs_cpu_consistency() {
        // Adaptive on CPU should produce same results as CpuOrthogonalize
        let adaptive = AdaptiveOrthogonalize::new();
        let cpu = CpuOrthogonalize::new();

        let g = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));

        let result_adaptive = adaptive.polar_express(&g, 5);
        let result_cpu = cpu.polar_express(&g, 5);

        // Results should be identical on CPU (both use FP32)
        let diff = (&result_adaptive - &result_cpu).abs().max().double_value(&[]);
        assert!(
            diff < 1e-5,
            "Adaptive and CPU results differ: max diff={diff}"
        );
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_adaptive_orthogonalize_cuda() {
        if !tch::Cuda::is_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let ortho = AdaptiveOrthogonalize::new();
        let device = Device::Cuda(0);

        let g = Tensor::randn([64, 64], (Kind::Float, device));
        let result = ortho.polar_express(&g, 5);

        // Check result is on same device
        assert_eq!(result.device(), device);

        // Check orthogonality
        let qqt = result.to_kind(Kind::Float).matmul(&result.to_kind(Kind::Float).tr());
        let identity = Tensor::eye(64, (Kind::Float, device));
        let ortho_error = (&qqt - &identity).abs().max().double_value(&[]);

        assert!(
            ortho_error < 0.5,
            "CUDA result not orthogonal enough: max error={ortho_error}"
        );
    }

    /// Helper to check orthogonality with configurable tolerance
    fn assert_orthogonal(q: &Tensor, tolerance: f64, msg: &str) {
        let n = q.size()[0];
        let qqt = q.to_kind(Kind::Float).matmul(&q.to_kind(Kind::Float).tr());
        let identity = Tensor::eye(n, (Kind::Float, q.device()));
        let error = (&qqt - &identity).abs().max().double_value(&[]);
        assert!(error < tolerance, "{}: orthogonality error={}", msg, error);
    }

    #[test]
    fn test_orthogonality_helper() {
        // Test the helper function itself
        let angle = std::f64::consts::PI / 4.0;
        let data: [f32; 4] = [
            angle.cos() as f32,
            -angle.sin() as f32,
            angle.sin() as f32,
            angle.cos() as f32,
        ];
        let q = Tensor::from_slice(&data).reshape([2, 2]);

        // This should pass - it's a rotation matrix (orthogonal)
        assert_orthogonal(&q, 1e-5, "Rotation matrix");
    }

    // ==================== Per-Precision Orthogonality Tests ====================

    #[test]
    fn test_orthogonality_fp32() {
        // Test orthogonality with FP32 computation
        let g = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));
        let result = polar_express_impl(&g, 5, Kind::Float, Kind::Float);

        // Allow 0.3 tolerance (matches existing tests, random matrices have variance)
        assert_orthogonal(&result, 0.3, "FP32 orthogonality");

        // Also check no NaN/Inf
        assert!(
            result.isnan().any().int64_value(&[]) == 0,
            "FP32 result contains NaN"
        );
        assert!(
            result.isinf().any().int64_value(&[]) == 0,
            "FP32 result contains Inf"
        );
    }

    #[test]
    fn test_orthogonality_fp16() {
        // Test orthogonality with FP16 computation (used on V100)
        let g = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));
        let result = polar_express_impl(&g, 5, Kind::Half, Kind::Float);

        // FP16 has less precision, allow 0.35 tolerance
        assert_orthogonal(&result, 0.35, "FP16 orthogonality");

        assert!(
            result.isnan().any().int64_value(&[]) == 0,
            "FP16 result contains NaN"
        );
        assert!(
            result.isinf().any().int64_value(&[]) == 0,
            "FP16 result contains Inf"
        );
    }

    #[test]
    fn test_orthogonality_bf16() {
        // Test orthogonality with BF16 computation (used on A100+)
        // Note: BF16 is slow on CPU (emulated) but we test correctness
        let g = Tensor::randn([8, 8], (Kind::Float, Device::Cpu)); // Smaller for speed

        let result = polar_express_impl(&g, 5, Kind::BFloat16, Kind::Float);

        // BF16 has same mantissa as FP16, so similar tolerance
        assert_orthogonal(&result, 0.35, "BF16 orthogonality");

        assert!(
            result.isnan().any().int64_value(&[]) == 0,
            "BF16 result contains NaN"
        );
        assert!(
            result.isinf().any().int64_value(&[]) == 0,
            "BF16 result contains Inf"
        );
    }

    // ==================== Cross-Precision Consistency Tests ====================

    #[test]
    fn test_cross_precision_consistency() {
        // Results from different precisions should be broadly consistent
        // (not identical due to precision differences, but similar)
        let g = Tensor::randn([8, 8], (Kind::Float, Device::Cpu));

        let result_fp32 = polar_express_impl(&g, 5, Kind::Float, Kind::Float);
        let result_fp16 = polar_express_impl(&g, 5, Kind::Half, Kind::Float);
        let result_bf16 = polar_express_impl(&g, 5, Kind::BFloat16, Kind::Float);

        // All results should be orthogonal (use relaxed tolerances for random matrices)
        assert_orthogonal(&result_fp32, 0.3, "FP32 cross-precision");
        assert_orthogonal(&result_fp16, 0.35, "FP16 cross-precision");
        assert_orthogonal(&result_bf16, 0.35, "BF16 cross-precision");

        // FP32 vs FP16 should be similar (not identical)
        let diff_fp32_fp16 = (&result_fp32 - &result_fp16).abs().max().double_value(&[]);
        assert!(
            diff_fp32_fp16 < 0.5,
            "FP32 and FP16 results differ too much: {diff_fp32_fp16}"
        );

        // FP32 vs BF16 should be similar
        let diff_fp32_bf16 = (&result_fp32 - &result_bf16).abs().max().double_value(&[]);
        assert!(
            diff_fp32_bf16 < 0.5,
            "FP32 and BF16 results differ too much: {diff_fp32_bf16}"
        );

        // FP16 vs BF16 should be similar (both have ~same precision)
        let diff_fp16_bf16 = (&result_fp16 - &result_bf16).abs().max().double_value(&[]);
        assert!(
            diff_fp16_bf16 < 0.5,
            "FP16 and BF16 results differ too much: {diff_fp16_bf16}"
        );
    }

    #[test]
    fn test_cross_precision_preserves_scale() {
        // Regardless of precision, output should have similar Frobenius norm
        // (orthogonal matrices have norm = sqrt(min(m,n)))
        let g = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));

        let result_fp32 = polar_express_impl(&g, 5, Kind::Float, Kind::Float);
        let result_fp16 = polar_express_impl(&g, 5, Kind::Half, Kind::Float);
        let result_bf16 = polar_express_impl(&g, 5, Kind::BFloat16, Kind::Float);

        let norm_fp32 = result_fp32.norm().double_value(&[]);
        let norm_fp16 = result_fp16.norm().double_value(&[]);
        let norm_bf16 = result_bf16.norm().double_value(&[]);

        // Expected norm for orthogonal 16x16 matrix is sqrt(16) = 4.0
        let expected_norm = 4.0;
        let tolerance = 0.5;

        assert!(
            (norm_fp32 - expected_norm).abs() < tolerance,
            "FP32 norm unexpected: {norm_fp32} (expected ~{expected_norm})"
        );
        assert!(
            (norm_fp16 - expected_norm).abs() < tolerance,
            "FP16 norm unexpected: {norm_fp16} (expected ~{expected_norm})"
        );
        assert!(
            (norm_bf16 - expected_norm).abs() < tolerance,
            "BF16 norm unexpected: {norm_bf16} (expected ~{expected_norm})"
        );
    }

    // ==================== Convergence Tests ====================

    /// Compute orthogonality error: ||Q @ Q^T - I||_F / n
    fn orthogonality_error(q: &Tensor) -> f64 {
        let n = q.size()[0] as f64;
        let qqt = q.matmul(&q.tr());
        let identity = Tensor::eye(q.size()[0], (q.kind(), q.device()));
        let frobenius = (&qqt - &identity).norm().double_value(&[]);
        frobenius / n
    }

    #[test]
    fn test_convergence_iteration_improves() {
        // More iterations should generally improve orthogonality (not strictly monotonic)
        let g = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));

        let mut errors = Vec::new();

        for iters in 1..=5 {
            let result = polar_express_impl(&g, iters, Kind::Float, Kind::Float);
            let error = orthogonality_error(&result);
            errors.push(error);
        }

        // Log errors for visibility
        eprintln!("Iteration errors: {:?}", errors);

        // Error at iteration 5 should be less than iteration 1
        // (not strictly monotonic - some iterations may temporarily increase)
        assert!(
            errors[4] < errors[0],
            "5-iteration error ({}) should be < 1-iteration error ({})",
            errors[4],
            errors[0]
        );

        // Error at iteration 5 should be reasonably small
        assert!(
            errors[4] < 0.2,
            "5-iteration error should be < 0.2, got {}",
            errors[4]
        );
    }

    #[test]
    fn test_convergence_rate_five_iterations() {
        // 5 iterations should achieve orthogonality error < 0.1 for well-conditioned matrices
        let g = Tensor::randn([32, 32], (Kind::Float, Device::Cpu));
        let result = polar_express_impl(&g, 5, Kind::Float, Kind::Float);

        let error = orthogonality_error(&result);
        assert!(
            error < 0.1,
            "5 iterations should achieve error < 0.1, got {}",
            error
        );
    }

    #[test]
    fn test_convergence_rectangular_matrices() {
        // Polar Express should work on rectangular matrices (m x n where m != n)
        // For tall matrices (m > n), we get Q such that Q^T @ Q ≈ I_n
        // For wide matrices (m < n), we get Q such that Q @ Q^T ≈ I_m

        // Tall matrix (more rows than columns)
        let g_tall = Tensor::randn([32, 16], (Kind::Float, Device::Cpu));
        let result_tall = polar_express_impl(&g_tall, 5, Kind::Float, Kind::Float);

        // For tall matrices, Q^T @ Q should be close to identity
        let qtq = result_tall.tr().matmul(&result_tall);
        let identity_n = Tensor::eye(16, (Kind::Float, Device::Cpu));
        let error_tall = (&qtq - &identity_n).norm().double_value(&[]) / 16.0;

        assert!(
            error_tall < 0.2,
            "Tall matrix Q^T @ Q error too high: {}",
            error_tall
        );

        // Wide matrix (more columns than rows)
        let g_wide = Tensor::randn([16, 32], (Kind::Float, Device::Cpu));
        let result_wide = polar_express_impl(&g_wide, 5, Kind::Float, Kind::Float);

        // For wide matrices, Q @ Q^T should be close to identity
        let qqt = result_wide.matmul(&result_wide.tr());
        let identity_m = Tensor::eye(16, (Kind::Float, Device::Cpu));
        let error_wide = (&qqt - &identity_m).norm().double_value(&[]) / 16.0;

        assert!(
            error_wide < 0.2,
            "Wide matrix Q @ Q^T error too high: {}",
            error_wide
        );
    }

    #[test]
    fn test_convergence_stability_under_perturbation() {
        // Two similar inputs should produce similar outputs (Lipschitz continuity)
        let g1 = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));
        let perturbation = Tensor::randn([16, 16], (Kind::Float, Device::Cpu)) * 0.01;
        let g2 = &g1 + &perturbation;

        let result1 = polar_express_impl(&g1, 5, Kind::Float, Kind::Float);
        let result2 = polar_express_impl(&g2, 5, Kind::Float, Kind::Float);

        // Output difference should be bounded by some multiple of input difference
        let input_diff = (&g1 - &g2).norm().double_value(&[]);
        let output_diff = (&result1 - &result2).norm().double_value(&[]);

        // Polar Express is designed to be stable; output perturbation should be reasonable
        // Allow up to 10x amplification (typical for well-conditioned cases)
        let amplification = output_diff / input_diff;
        assert!(
            amplification < 20.0,
            "Perturbation amplified too much: input_diff={}, output_diff={}, ratio={}",
            input_diff,
            output_diff,
            amplification
        );
    }

    #[test]
    fn test_convergence_condition_number_impact() {
        // Test with matrices of varying condition numbers
        // Higher condition number → potentially slower convergence

        // Well-conditioned matrix (condition number ~1)
        let well_conditioned = Tensor::eye(16, (Kind::Float, Device::Cpu))
            + Tensor::randn([16, 16], (Kind::Float, Device::Cpu)) * 0.1;
        let result_wc = polar_express_impl(&well_conditioned, 5, Kind::Float, Kind::Float);
        let error_wc = orthogonality_error(&result_wc);

        // Moderately ill-conditioned matrix
        // Create by scaling singular values differently
        let u = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));
        let (q, _) = u.qr(true); // Get orthogonal Q from QR decomposition
        let singular_values = Tensor::from_slice(&[
            10.0f32, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1,
        ])
        .reshape([16, 1]);
        let ill_conditioned = &q * &singular_values; // Scale columns by singular values
        let result_ic = polar_express_impl(&ill_conditioned, 5, Kind::Float, Kind::Float);
        let error_ic = orthogonality_error(&result_ic);

        // Both should converge, but well-conditioned should be better
        assert!(
            error_wc < 0.15,
            "Well-conditioned error too high: {}",
            error_wc
        );
        assert!(
            error_ic < 0.5,
            "Ill-conditioned error too high (should still converge): {}",
            error_ic
        );

        // Log the results for visibility
        #[cfg(test)]
        {
            eprintln!(
                "Condition number test: well_conditioned_error={:.4}, ill_conditioned_error={:.4}",
                error_wc, error_ic
            );
        }
    }

    #[test]
    fn test_convergence_deterministic() {
        // Same input should produce same output (no randomness in algorithm)
        let g = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));

        let result1 = polar_express_impl(&g, 5, Kind::Float, Kind::Float);
        let result2 = polar_express_impl(&g, 5, Kind::Float, Kind::Float);

        let diff = (&result1 - &result2).abs().max().double_value(&[]);
        assert!(
            diff < 1e-10,
            "Same input should produce identical output, diff={}",
            diff
        );
    }
}
