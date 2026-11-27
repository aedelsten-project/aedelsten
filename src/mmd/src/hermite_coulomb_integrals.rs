// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The Hermite Coulomb integrals of a primitive pair: the Boys values `R^n_{000}` and the tensor `R_{tuv}`, for the
//! full, long- and short-range Coulomb operators (`RangeVersion`) and the AIMP screens.

use crate::hermite_r_tensor::*;
use crate::hermite_expansion::hermite_count;
use arrays::scratch::{Arena, ArenaError};
use crate::hermite_r_generated::r_tensor;
use crate::hermite_r_tensor_rolling::compute_r_tensor_rolling;
use math_utilities::boys::{compute_boys_f0, compute_boys_functions};

/// The Hermite Coulomb integrals `R^n_{000}`, `n = 0..=n_max`, of one primitive pair and one centre, in arena storage.
pub struct HermiteCoulombIntegrals1D<'a> {
    n_max: usize,
    data: &'a mut [f64],
    sr_helper: &'a mut [f64],
}

/// The Hermite Coulomb integrals `R^0_{tuv}`, `t + u + v <= n_max`, of one primitive pair and one centre, with their
/// scratch, in arena storage.
pub struct HermiteCoulombIntegrals4D<'a> {
    n_max: usize,
    tuv_max: usize,
    output: &'a mut [f64],
    shell_a: &'a mut [f64],
    shell_b: &'a mut [f64],
    boys: &'a mut [f64],
    sr_helper: &'a mut [f64],
    // The integrals in a box (`view_box`), `(n_max + 1)^3`, and the box offsets of the bra functions for
    // `offsets_lab` (`usize::MAX`: none yet).
    cube: &'a mut [f64],
    bra_offsets: &'a mut [u32],
    offsets_lab: usize,
}

/// The operator of the integrals: `1/r` (`FullRange`), `erf(omega r)/r` (`LongRange`) or `erfc(omega r)/r`
/// (`ShortRange`).
pub trait RangeVersion {
    /// The integrals `R^0_{tuv}` of `buffer` for the exponent `p`, `(pcx, pcy, pcz) = P - C`.
    fn compute_r_buffer(buffer: &mut HermiteCoulombIntegrals4D, p: f64, pcx: f64, pcy: f64, pcz: f64, rpc_square: f64, omega: f64);
    /// The Boys values `R^n_{000}` of `buffer` alone, `n = 0..=n_max` (`get_boys`), for a caller that builds the
    /// integrals from them itself.
    fn compute_boys(buffer: &mut HermiteCoulombIntegrals4D, p: f64, rpc_square: f64, omega: f64);
    /// The integrals `R^n_{000}` of `buffer`.
    fn compute_r_buffer_1d(buffer: &mut HermiteCoulombIntegrals1D, p: f64, rpc_square: f64, omega: f64);
    /// `R^0_{000}` alone.
    fn compute_r_ssss(p: f64, rpc_square: f64, omega: f64) -> f64;
}

/// The operator `1/r`.
pub struct FullRange;
impl RangeVersion for FullRange {
    fn compute_r_buffer(buffer: &mut HermiteCoulombIntegrals4D, p: f64, pcx: f64, pcy: f64, pcz: f64, rpc_square: f64, _omega: f64) {
        buffer.compute(p, pcx, pcy, pcz, rpc_square);
    }

    fn compute_boys(buffer: &mut HermiteCoulombIntegrals4D, p: f64, rpc_square: f64, _omega: f64) {
        compute_r_only_boys(&mut buffer.boys, buffer.n_max, p, rpc_square);
    }

    fn compute_r_buffer_1d(buffer: &mut HermiteCoulombIntegrals1D, p: f64, rpc_square: f64, _omega: f64) {
        buffer.compute(p, rpc_square);
    }

    #[inline(always)]
    fn compute_r_ssss(p: f64, rpc_square: f64, _omega: f64) -> f64 {
        compute_boys_function_s(p, rpc_square)
    }
}

/// The operator `erf(omega r)/r`.
pub struct LongRange;
impl RangeVersion for LongRange {
    fn compute_r_buffer(buffer: &mut HermiteCoulombIntegrals4D, p: f64, pcx: f64, pcy: f64, pcz: f64, rpc_square: f64, omega: f64) {
        buffer.compute_lr(p, pcx, pcy, pcz, rpc_square, omega);
    }

    fn compute_boys(buffer: &mut HermiteCoulombIntegrals4D, p: f64, rpc_square: f64, omega: f64) {
        compute_r_only_boys_lr(&mut buffer.boys, buffer.n_max, p, rpc_square, omega);
    }
    
    fn compute_r_buffer_1d(buffer: &mut HermiteCoulombIntegrals1D, p: f64, rpc_square: f64, omega: f64) {
        buffer.compute_lr(p, rpc_square, omega);
    }

    #[inline(always)]
    fn compute_r_ssss(p: f64, rpc_square: f64, omega: f64) -> f64 {
        compute_boys_function_s_lr(p, rpc_square, omega)
    }
}

/// The operator `erfc(omega r)/r`.
pub struct ShortRange;
impl RangeVersion for ShortRange {
    fn compute_r_buffer(buffer: &mut HermiteCoulombIntegrals4D, p: f64, pcx: f64, pcy: f64, pcz: f64, rpc_square: f64, omega: f64) {
        buffer.compute_sr(p, pcx, pcy, pcz, rpc_square, omega);
    }

    fn compute_boys(buffer: &mut HermiteCoulombIntegrals4D, p: f64, rpc_square: f64, omega: f64) {
        compute_r_only_boys_sr(&mut buffer.boys, &mut buffer.sr_helper, buffer.n_max, p, rpc_square, omega);
    }
    
    fn compute_r_buffer_1d(buffer: &mut HermiteCoulombIntegrals1D, p: f64, rpc_square: f64, omega: f64) {
        buffer.compute_sr(p, rpc_square, omega);
    }

    #[inline(always)]
    fn compute_r_ssss(p: f64, rpc_square: f64, omega: f64) -> f64 {
        compute_boys_function_s_sr(p, rpc_square, omega)
    }
}

impl<'a> HermiteCoulombIntegrals4D<'a> {
    /// Storage for `n_max` from the arena.
    pub fn from_arena(n_max: usize, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let tuv_max = hermite_count(n_max);
        let output = arena.alloc::<f64>(tuv_max)?;
        let shell_a = arena.alloc::<f64>(tuv_max)?;
        let shell_b = arena.alloc::<f64>(tuv_max)?;
        let boys = arena.alloc::<f64>(n_max + 1)?;
        let sr_helper = arena.alloc::<f64>(n_max + 1)?;
        let cube = arena.alloc::<f64>((n_max + 1).pow(3))?;
        let bra_offsets = arena.alloc::<u32>(tuv_max)?;
        Ok(Self { n_max, tuv_max, output, shell_a, shell_b, boys, sr_helper, cube, bra_offsets, offsets_lab: usize::MAX })
    }

    /// `R^0_{tuv}` from the Boys values: the generated kernel up to `R_TENSOR_L_MAX`, the recursion above.
    #[inline(always)]
    fn tensor(&mut self, pcx: f64, pcy: f64, pcz: f64) {
        if !r_tensor(self.n_max, &self.boys, pcx, pcy, pcz, &mut self.output) {
            compute_r_tensor_rolling(&mut self.output, &self.boys, self.n_max, pcx, pcy, pcz, &mut self.shell_a, &mut self.shell_b);
        }
    }

    /// The integrals of `1/r` for the exponent `p`, `(pcx, pcy, pcz) = P - C` and `rpc_square = |P - C|^2`.
    pub fn compute(&mut self, p: f64, pcx: f64, pcy: f64, pcz: f64, rpc_square: f64) {
        compute_r_only_boys(&mut self.boys, self.n_max, p, rpc_square);
        self.tensor(pcx, pcy, pcz);
    }

    /// The integrals of `erf(omega r)/r`.
    pub fn compute_lr(&mut self, p: f64, pcx: f64, pcy: f64, pcz: f64, rpc_square: f64, omega: f64) {
        compute_r_only_boys_lr(&mut self.boys, self.n_max, p, rpc_square, omega);
        self.tensor(pcx, pcy, pcz);
    }

    /// The integrals of `erfc(omega r)/r`.
    pub fn compute_sr(&mut self, p: f64, pcx: f64, pcy: f64, pcz: f64, rpc_square: f64, omega: f64) {
        compute_r_only_boys_sr(&mut self.boys, &mut self.sr_helper, self.n_max, p, rpc_square, omega);
        self.tensor(pcx, pcy, pcz);
    }

    /// The integrals of a Gaussian-screened `exp(-c r_C^2) / r_C` for the exponent `p`, `(pcx, pcy, pcz) = P - C`
    /// and `rpc_square = s = |P - C|^2`, without the factor `2 pi / q`: those of the radial function
    /// `phi(s) = exp(-mu s) F_0(beta s)` (`q = p + c`, `mu = p c / q`, `beta = p^2 / q`; the product of the Gaussian
    /// of exponent p at P with the screen is `exp(-mu s)` times one of exponent q at Q, `q |Q - C|^2 = beta s`). By
    /// Leibniz its auxiliaries are `2^n phi^(n)(s) = exp(-mu s) Sum_k C(n, k) (-2 mu)^(n - k) (-2 beta)^k
    /// F_k(beta s)`, every term of one sign.
    pub fn compute_screened(&mut self, p: f64, c: f64, pcx: f64, pcy: f64, pcz: f64, rpc_square: f64) {
        let q = p + c;
        let (mu, beta) = (p * c / q, p * p / q);
        // (-2 beta)^k F_k(beta s) into the helper, then the binomial sums into the auxiliaries.
        boys_recursion(&mut self.sr_helper, self.n_max, beta, rpc_square);
        let (damping, m) = ((-mu * rpc_square).exp(), -2.0 * mu);
        for (n, auxiliary) in self.boys.iter_mut().enumerate() {
            // k from n down to 0: C(n, k) m^(n - k) from C(n, n) m^0.
            let (mut sum, mut binomial, mut power) = (0.0, 1.0, 1.0);
            for k in (0..=n).rev() {
                sum += binomial * power * self.sr_helper[k];
                binomial *= k as f64 / (n - k + 1) as f64;
                power *= m;
            }
            *auxiliary = damping * sum;
        }
        self.tensor(pcx, pcy, pcz);
    }

    /// The Boys values `R^n_{000}`, `n = 0..=n_max`, of the last computation.
    pub fn get_boys(&self) -> &[f64] {
        &self.boys
    }

    /// The integrals of the last computation in a box, `R_tuv` at `t + s u + s^2 v` with `s = n_max + 1` (the entries
    /// above `t + u + v = n_max` unset), with `s` and the box offsets of `t + u + v <= lab` in `hermite_index` order:
    /// `R_(t+τ, u+ν, v+φ)` is `box[offsets[tuv] + τ + s ν + s^2 φ]`, so an R matrix column is one loop over the
    /// offsets. The offsets are kept for the next call with the same `lab`.
    pub fn view_box(&mut self, lab: usize) -> (&[f64], &[u32], usize) {
        let (n, s) = (self.n_max, self.n_max + 1);
        // The packed rows (u, v) are consecutive, each contiguous in t.
        let mut i = 0;
        for v in 0..=n {
            for u in 0..=(n - v) {
                let len = n - u - v + 1;
                self.cube[s * u + s * s * v..][..len].copy_from_slice(&self.output[i..i + len]);
                i += len;
            }
        }
        if self.offsets_lab != lab {
            let mut k = 0;
            for v in 0..=lab {
                for u in 0..=(lab - v) {
                    for t in 0..=(lab - v - u) {
                        self.bra_offsets[k] = (t + s * u + s * s * v) as u32;
                        k += 1;
                    }
                }
            }
            self.offsets_lab = lab;
        }
        (&self.cube[..], &self.bra_offsets[..hermite_count(lab)], s)
    }

    /// The integrals, packed by `hermite_index(.., n_max)`.
    pub fn view_tuv(&self) -> HermiteTuvSliceView<'_, f64> {
        HermiteTuvSliceView::new(self.tuv_max, self.n_max, &self.output[..])
    }
}

impl<'a> HermiteCoulombIntegrals1D<'a> {
    /// Zeroed storage for `n_max` from the arena.
    pub fn from_arena(n_max: usize, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let data = arena.alloc::<f64>(n_max + 1)?;
        data.fill(0.0);
        let sr_helper = arena.alloc::<f64>(n_max + 1)?;
        sr_helper.fill(0.0);
        Ok(Self { n_max, data, sr_helper })
    }

    /// The integrals of `1/r` for the exponent `p` and `rpc_square = |P - C|^2`.
    pub fn compute(&mut self, p: f64, rpc_square: f64) {
        compute_r_only_boys(
            &mut self.data,
            self.n_max,
            p,
            rpc_square,
        );
    }

    /// The integrals of `erf(omega r)/r`.
    pub fn compute_lr(&mut self, p: f64, rpc_square: f64, omega: f64) {
        compute_r_only_boys_lr(
            &mut self.data,
            self.n_max,
            p,
            rpc_square,
            omega
        );
    }

    /// The integrals of `erfc(omega r)/r`.
    pub fn compute_sr(&mut self, p: f64, rpc_square: f64, omega: f64) {
        compute_r_only_boys_sr(
            &mut self.data,
            &mut self.sr_helper,
            self.n_max,
            p,
            rpc_square,
            omega
        );
    }

    /// `R^n_{000}`, `n = 0..=n_max`.
    pub fn get_boys(&self) -> &[f64] {
        &self.data
    }
}

/// `boys[n] = R^n_{000} = (-2p)^n F_n(p R_PC^2)`, `n = 0..=n_max`.
pub fn compute_r_only_boys(boys: &mut [f64], n_max: usize, p: f64, rpc_square: f64) {
    boys_recursion(boys, n_max, p, rpc_square)
}

/// `R^n_{000}` of `erf(omega r)/r` into `boys`.
pub fn compute_r_only_boys_lr(boys: &mut [f64], n_max: usize, p: f64, rpc_square: f64, omega: f64) {
    boys_recursion_lr(boys, n_max, p, rpc_square, omega)
}

/// `R^n_{000}` of `erfc(omega r)/r` into `boys`, `scratch` (`n_max + 1` long) for the long-range part.
pub fn compute_r_only_boys_sr(boys: &mut [f64], scratch: &mut [f64], n_max: usize, p: f64, rpc_square: f64, omega: f64) {
    boys_recursion_sr(boys, scratch, n_max, p, rpc_square, omega)
}

/// `R^0_{000} = F_0(p R_PC^2)`.
#[inline(always)]
pub fn compute_boys_function_s(p: f64, rpc_square: f64) -> f64 {
    compute_boys_f0(p * rpc_square)
}

/// `R^0_{000}` of `erf(omega r)/r`, `lr_factor F_0(lr_factor^2 p R_PC^2)`, `lr_factor^2 = omega^2 / (omega^2 + p)`.
#[inline(always)]
pub fn compute_boys_function_s_lr(p: f64, rpc_square: f64, omega: f64) -> f64 {
    let lr_factor_square = omega.powi(2) / (omega.powi(2) + p);
    let x = lr_factor_square * p * rpc_square;
    let lr_factor = lr_factor_square.sqrt();
    lr_factor * compute_boys_f0(x)
}

/// `R^0_{000}` of `erfc(omega r)/r`, the full minus the long-range one.
#[inline(always)]
pub fn compute_boys_function_s_sr(p: f64, rpc_square: f64, omega: f64) -> f64 {
    compute_boys_function_s(p, rpc_square) - compute_boys_function_s_lr(p, rpc_square, omega)
}

// R^{(n)}_{000} = (-2p)^n F_n(x), x = p R_PC^2.
fn boys_recursion(
    boys: &mut [f64],
    n_max: usize,
    p: f64,
    rpc_square: f64
) {
    let boys = &mut boys[..=n_max];
    compute_boys_functions(p * rpc_square, boys);
    let m_two_p = -2.0 * p;
    let mut m_two_p_to_n = 1.0;
    for b in boys.iter_mut() {
        *b *= m_two_p_to_n;
        m_two_p_to_n *= m_two_p;
    }
}

// R^{(n),lr}_{000} = (-2p)^n (p'/p)^{n+1/2} F_n(x') = (-2p)^n lr_factor^{2n+1} F_n(x'), with
// lr_factor^2 = omega^2 / (omega^2 + p) and x' = lr_factor^2 x.
fn boys_recursion_lr(
    boys: &mut [f64],
    n_max: usize,
    p: f64,
    rpc_square: f64,
    omega: f64
) {
    let lr_factor_square = omega.powi(2) / (omega.powi(2) + p);
    let x = lr_factor_square * p * rpc_square;
    let lr_factor = lr_factor_square.sqrt();
    let boys = &mut boys[..=n_max];
    compute_boys_functions(x, boys);
    let factor = -2.0 * p * lr_factor_square;
    let mut factor_to_n = lr_factor;
    for b in boys.iter_mut() {
        *b *= factor_to_n;
        factor_to_n *= factor;
    }
}

// Short-range = full - long-range: F_n at x and at x_lr = lr_factor^2 x, then one pass that scales
// and subtracts. `scratch` holds the long-range values.
fn boys_recursion_sr(
    boys: &mut [f64],
    scratch: &mut [f64],
    n_max: usize,
    p: f64,
    rpc_square: f64,
    omega: f64
) {
    let x = p * rpc_square;
    let lr_factor_square = omega.powi(2) / (omega.powi(2) + p);
    let x_lr = lr_factor_square * x;
    let lr_factor = lr_factor_square.sqrt();
    let boys = &mut boys[..=n_max];
    let scratch = &mut scratch[..=n_max];
    compute_boys_functions(x, boys);
    compute_boys_functions(x_lr, scratch);
    let m_two_p = -2.0 * p;
    let mut m_two_p_to_n = 1.0;
    let mut lr_factor_pow = lr_factor; // lr_factor^(2n+1), starting at n = 0.
    for (b, s) in boys.iter_mut().zip(scratch.iter()) {
        *b = m_two_p_to_n * (*b - lr_factor_pow * *s);
        m_two_p_to_n *= m_two_p;
        lr_factor_pow *= lr_factor_square;
    }
}

#[cfg(test)]
mod sr_recursion_tests {
    use super::*;

    // Full minus long-range, by the two separate recursions; the sweep below puts x and x_lr on both sides of x = 36.
    fn reference_sr(n_max: usize, p: f64, rpc_square: f64, omega: f64) -> Vec<f64> {
        let mut full = vec![0.0; n_max + 1];
        let mut lr = vec![0.0; n_max + 1];
        boys_recursion(&mut full, n_max, p, rpc_square);
        boys_recursion_lr(&mut lr, n_max, p, rpc_square, omega);
        for (f, l) in full.iter_mut().zip(lr.iter()) {
            *f -= *l;
        }
        full
    }

    #[test]
    fn fused_sr_matches_full_minus_long_range() {
        let ps = [0.05, 0.3, 1.0, 3.5, 12.0];
        let omegas = [0.1, 0.3, 0.4, 0.8, 1.5];
        let rpc_squares = [0.0, 0.01, 0.5, 1.0, 4.0, 10.0, 30.0];
        let n_maxes = [0usize, 1, 2, 5, 9, 16];

        for &n_max in &n_maxes {
            for &p in &ps {
                for &omega in &omegas {
                    for &rpc_square in &rpc_squares {
                        let expected = reference_sr(n_max, p, rpc_square, omega);

                        let mut actual = vec![0.0; n_max + 1];
                        let mut scratch = vec![0.0; n_max + 1];
                        boys_recursion_sr(&mut actual, &mut scratch, n_max, p, rpc_square, omega);

                        for n in 0..=n_max {
                            let diff = (actual[n] - expected[n]).abs();
                            let scale = expected[n].abs().max(1.0);
                            assert!(
                                diff / scale < 1e-10,
                                "mismatch at n_max={n_max} n={n} p={p} omega={omega} \
                                 rpc_square={rpc_square}: actual={} expected={}",
                                actual[n], expected[n]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod generated_tensor_tests {
    use super::*;
    use crate::hermite_r_generated::R_TENSOR_L_MAX;

    // Every generated tensor against the recursion on the same Boys values, for exponents, separations (on a centre,
    // and both sides of the Boys function's switch at x = 36) and directions with every component's sign.
    #[test]
    fn generated_tensors_match_the_recursion() {
        let ps = [0.05, 0.7, 3.0, 40.0];
        let separations = [[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [-1.1, 0.4, 2.0], [2.5, -3.0, -1.5]];
        for l in 0..=R_TENSOR_L_MAX {
            let n = hermite_count(l);
            let (mut generated, mut reference, mut a, mut b) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            let mut boys = vec![0.0; l + 1];
            for &p in &ps {
                for pc in &separations {
                    let rpc_square = pc.iter().map(|x| x * x).sum();
                    compute_r_only_boys(&mut boys, l, p, rpc_square);
                    assert!(r_tensor(l, &boys, pc[0], pc[1], pc[2], &mut generated));
                    compute_r_tensor_rolling(&mut reference, &boys, l, pc[0], pc[1], pc[2], &mut a, &mut b);
                    let scale = reference.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                    for (i, (x, y)) in generated.iter().zip(&reference).enumerate() {
                        // 1e-12: the two take different paths through the recursion, which cancels more at l = 12.
                        assert!((x - y).abs() <= 1e-12 * scale, "l = {l}, p = {p}, pc = {pc:?}, [{i}]: {x} against {y}");
                    }
                }
            }
        }
        // Above the generated range the dispatch writes nothing.
        let mut out = vec![1.0; hermite_count(R_TENSOR_L_MAX + 1)];
        assert!(!r_tensor(R_TENSOR_L_MAX + 1, &vec![0.5; R_TENSOR_L_MAX + 2], 0.1, 0.2, 0.3, &mut out));
        assert!(out.iter().all(|&x| x == 1.0));
    }
}

#[cfg(test)]
mod boys_benchmark {
    use super::*;

    // (p, R_PC^2) samples: p log-uniform in [0.05, 50], R^2 uniform in [0, 10], from a fixed LCG.
    fn samples(count: usize) -> Vec<(f64, f64)> {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..count).map(|_| (0.05 * 1000f64.powf(next()), 10.0 * next())).collect()
    }

    // Prints the time per call of the full, long- and short-range R^n_000; not a check.
    //   cargo test -p mmd --release -- --ignored --nocapture benchmark_boys
    #[test]
    #[ignore]
    fn benchmark_boys_recursion() {
        use std::hint::black_box;
        use std::time::Instant;

        let samples = samples(200_000);
        let omega = 0.4;
        let mut boys = vec![0.0; 32];
        let mut scratch = vec![0.0; 32];

        println!();
        println!("{:>6}{:>14}{:>14}{:>14}", "n_max", "full ns/call", "lr ns/call", "sr ns/call");
        for n_max in [0usize, 1, 2, 4, 8, 12, 16, 24, 26, 30] {
            let mut time = |f: &mut dyn FnMut(f64, f64, &mut [f64], &mut [f64])| {
                let mut sum = 0.0;
                for &(p, r2) in &samples[..1000] {
                    f(p, r2, &mut boys, &mut scratch);
                }
                let start = Instant::now();
                for &(p, r2) in &samples {
                    f(black_box(p), black_box(r2), &mut boys, &mut scratch);
                    sum += boys[n_max];
                }
                black_box(sum);
                start.elapsed().as_nanos() as f64 / samples.len() as f64
            };
            let full = time(&mut |p, r2, b, _| {
                if n_max == 0 { b[0] = compute_boys_function_s(p, r2) } else { compute_r_only_boys(b, n_max, p, r2) }
            });
            let lr = time(&mut |p, r2, b, _| {
                if n_max == 0 { b[0] = compute_boys_function_s_lr(p, r2, omega) } else { compute_r_only_boys_lr(b, n_max, p, r2, omega) }
            });
            let sr = time(&mut |p, r2, b, s| {
                if n_max == 0 { b[0] = compute_boys_function_s_sr(p, r2, omega) } else { compute_r_only_boys_sr(b, s, n_max, p, r2, omega) }
            });
            println!("{:>6}{:>14.1}{:>14.1}{:>14.1}", n_max, full, lr, sr);
        }
    }
}
