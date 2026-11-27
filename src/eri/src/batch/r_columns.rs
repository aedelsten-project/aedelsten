// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The R matrices of one primitive quartet (bra x ket Hermite functions, `(-1)^(tau + nu + phi)` and the prefactor
//! included) for the BLAS tiles, the ERI gradient and the spin-orbit integrals: by the generated kernels of
//! `mmd::hermite_r_generated` up to `R_MATRIX_L_MAX`, from the R tensor (in its box) above; and the quartet's
//! prefactor.

use std::f64::consts::PI;

use crate::r_matrix::*;
use arrays::vector_math::{vector_norm_square_3d, vector_subtract_3d};
use mmd::hermite_coulomb_integrals::{RangeVersion, HermiteCoulombIntegrals4D};
use mmd::hermite_expansion::hermite_count;
use mmd::hermite_r_generated::{r_matrix, R_MATRIX_L_MAX};

/// The R matrix of a primitive quartet (exponents and centres in the arguments) into `r`, through `r_buffer` above
/// `R_MATRIX_L_MAX`.
#[inline(always)]
pub(crate) fn construct_r_xxxx_prealloc<R, T>(
    r_buffer: &mut HermiteCoulombIntegrals4D,
    r: &mut T,
    p: f64,
    pos_ab: &[f64; 3],
    lqn_ab: usize,
    q: f64,
    pos_cd: &[f64; 3],
    lqn_cd: usize,
    omega: f64,
)
where
    R: RangeVersion,
    T: RMatrixTraitMut<f64>,
{
    // The R matrix is column-major with one column per ket Hermite function.
    let columns = r.get_data_mut().chunks_exact_mut(hermite_count(lqn_ab));
    construct_r_xxxx_columns::<R>(r_buffer, columns, p, pos_ab, lqn_ab, q, pos_cd, lqn_cd, omega);
}

/// The R matrix of one primitive quartet, `bra_tuv x ket_tuv`, written into `columns`: one per ket Hermite function
/// in order, each `bra_tuv` long. A plain R matrix's columns, or one bra primitive's slot of R matrices stacked for
/// several primitives (`batch_xxxx`). Up to `R_MATRIX_L_MAX` on both sides from the Boys values by the generated
/// kernel, above it the R tensor and `gather_r_columns`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn construct_r_xxxx_columns<'c, R: RangeVersion>(
    r_buffer: &mut HermiteCoulombIntegrals4D,
    columns: impl Iterator<Item = &'c mut [f64]>,
    p: f64,
    pos_ab: &[f64; 3],
    lqn_ab: usize,
    q: f64,
    pos_cd: &[f64; 3],
    lqn_cd: usize,
    omega: f64,
) {
    // Overlap centre.
    let pos_abcd = vector_subtract_3d(&pos_ab, &pos_cd);
    let eta = p * q / (p + q);
    let eri_coeff = compute_eri_coeff(p, q);
    let rpc_square = vector_norm_square_3d(&pos_abcd);

    if lqn_ab <= R_MATRIX_L_MAX && lqn_cd <= R_MATRIX_L_MAX {
        R::compute_boys(r_buffer, eta, rpc_square, omega);
        r_matrix(lqn_ab, lqn_cd, r_buffer.get_boys(), pos_abcd[0], pos_abcd[1], pos_abcd[2], eri_coeff, columns);
    } else {
        R::compute_r_buffer(r_buffer, eta, pos_abcd[0], pos_abcd[1], pos_abcd[2], rpc_square, omega);
        gather_r_columns(r_buffer, columns, lqn_ab, lqn_cd, eri_coeff);
    }
}

/// The R matrix from the R tensor of `r_buffer` (order `lqn_ab + lqn_cd`), into `columns` as
/// `construct_r_xxxx_columns` writes it: element `(tuv, τνφ)` is `eri_coeff (-1)^(τ+ν+φ) R_(t+τ, u+ν, v+φ)`. From the
/// tensor's box (`view_box`): a column is one loop over the bra's box offsets, shifted by the ket function's.
#[inline(always)]
fn gather_r_columns<'c>(
    r_buffer: &mut HermiteCoulombIntegrals4D,
    mut columns: impl Iterator<Item = &'c mut [f64]>,
    lqn_ab: usize,
    lqn_cd: usize,
    eri_coeff: f64,
) {
    let (cube, bra_offsets, s) = r_buffer.view_box(lqn_ab);
    // The ket Hermite functions in order: t fastest, then u, then v.
    for φ in 0..=lqn_cd {
        for ν in 0..=(lqn_cd - φ) {
            for τ in 0..=(lqn_cd - φ - ν) {
                let pref = if ((τ + ν + φ) & 1) == 0 { eri_coeff } else { -eri_coeff };
                let shifted = &cube[τ + s * ν + s * s * φ..];
                let column = columns.next().expect("construct_r_xxxx_columns: one output column per ket Hermite function");
                for (d, &offset) in column.iter_mut().zip(bra_offsets) {
                    *d = pref * box_value(shifted, offset);
                }
            }
        }
    }
}

/// The box value at `offset` of the shifted box, set by construction in `gather_r_columns`: `t + u + v <= lqn_ab`
/// and `τ + ν + φ <= lqn_cd`, so the sum is at most `n_max`. Checked unless built with `unchecked`.
#[inline(always)]
fn box_value(shifted: &[f64], offset: u32) -> f64 {
    #[cfg(feature = "unchecked")]
    {
        unsafe { *shifted.get_unchecked(offset as usize) }
    }
    #[cfg(not(feature = "unchecked"))]
    {
        shifted[offset as usize]
    }
}

/// The prefactor of a primitive quartet's integrals, `2 pi^(5/2) / (p q sqrt(p + q))`.
#[inline(always)]
pub(crate) fn compute_eri_coeff(p: f64, q: f64) -> f64 {
    2.0 * PI.powi(5).sqrt() / (p * q * (p + q).sqrt())
}

#[cfg(test)]
mod generated_r_matrix_tests {
    use super::*;
    use arrays::scratch::{Scratch, ByteUnit};
    use mmd::hermite_coulomb_integrals::{FullRange, LongRange, ShortRange};

    // The R matrix as construct_r_xxxx_columns writes it (the generated kernel within R_MATRIX_L_MAX) against the
    // R tensor and gather_r_columns, for every (l_ab, l_cd) in the range, the three operators, on a centre and off.
    fn compare<R: RangeVersion>(label: &str, omega: f64) {
        let mut scratch = Scratch::new(1, ByteUnit::MiB).unwrap();
        let cases = [(0.9, [0.1, -0.2, 0.3], 2.2, [0.1, -0.2, 0.3]), (1.3, [0.3, -0.4, 0.5], 0.8, [-0.2, 0.6, 0.1]), (15.0, [1.0, 2.0, -1.0], 9.0, [-2.0, -1.5, 2.5])];
        for lqn_ab in 0..=R_MATRIX_L_MAX {
            for lqn_cd in 0..=R_MATRIX_L_MAX {
                let (rows, cols) = (hermite_count(lqn_ab), hermite_count(lqn_cd));
                for &(p, pos_ab, q, pos_cd) in &cases {
                    let arena = scratch.get_arena();
                    let mut r_buffer = HermiteCoulombIntegrals4D::from_arena(lqn_ab + lqn_cd, &arena).unwrap();
                    let (mut generated, mut reference) = (vec![0.0; rows * cols], vec![0.0; rows * cols]);
                    construct_r_xxxx_columns::<R>(&mut r_buffer, generated.chunks_exact_mut(rows), p, &pos_ab, lqn_ab, q, &pos_cd, lqn_cd, omega);
                    let pc = vector_subtract_3d(&pos_ab, &pos_cd);
                    R::compute_r_buffer(&mut r_buffer, p * q / (p + q), pc[0], pc[1], pc[2], vector_norm_square_3d(&pc), omega);
                    gather_r_columns(&mut r_buffer, reference.chunks_exact_mut(rows), lqn_ab, lqn_cd, compute_eri_coeff(p, q));
                    let scale = reference.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                    for (i, (x, y)) in generated.iter().zip(&reference).enumerate() {
                        assert!((x - y).abs() <= 1e-13 * scale, "{label} ({lqn_ab}, {lqn_cd}) p = {p} [{i}]: {x} against {y}");
                    }
                }
            }
        }
    }

    #[test]
    fn generated_r_matrices_match_the_tensor_and_gather() {
        compare::<FullRange>("full range", 0.0);
        compare::<LongRange>("long range", 0.4);
        compare::<ShortRange>("short range", 0.4);
    }
}

#[cfg(test)]
mod gather_benchmark {
    use super::*;
    use crate::r_matrix::RMatrix;
    use arrays::scratch::{Scratch, ByteUnit};
    use mmd::hermite_coulomb_integrals::FullRange;
    use mmd::hermite_r_tensor::HermiteRTensorTrait;

    // Order-sensitive fingerprint of the output bits: identical before and after a change that is meant to
    // be bit-identical.
    fn fingerprint(data: &[f64]) -> u64 {
        data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits()).wrapping_mul(0x0100_0000_01b3))
    }

    fn time<F: FnMut(usize)>(iters: usize, mut f: F) -> f64 {
        for i in 0..iters.min(100) {
            f(i);
        }
        let start = std::time::Instant::now();
        for i in 0..iters {
            f(i);
        }
        start.elapsed().as_nanos() as f64 / iters as f64
    }

    // Not run by default `cargo test` -- timing numbers on a shared/loaded machine are noise
    // otherwise. Run explicitly with:
    //   cargo test -p eri --release -- --ignored --nocapture benchmark_gather
    // The fingerprints identify the outputs; a bit-identical change leaves them unchanged.
    #[test]
    #[ignore]
    fn benchmark_gather() {
        let (p, q, omega) = (1.3, 0.8, 0.0);
        let pos_ab = [0.3, -0.4, 0.5];
        let pos_cd = [-0.2, 0.6, 0.1];
        let mut scratch = Scratch::new(64, ByteUnit::MiB).unwrap();

        println!();
        println!("{:<28}{:>12}{:>20}", "kernel", "ns/call", "fingerprint");
        for (lqn_ab, lqn_cd, iters) in [(4, 4, 200_000), (6, 6, 50_000), (8, 8, 20_000), (12, 12, 2_000)] {
            let arena = scratch.get_arena();
            let mut r_buffer = HermiteCoulombIntegrals4D::from_arena(lqn_ab + lqn_cd, &arena).unwrap();
            let mut r = RMatrix::<f64>::new(lqn_ab, lqn_cd);
            let ns = time(iters, |i| {
                let pos = [pos_ab[0] + i as f64 * 1e-9, pos_ab[1], pos_ab[2]];
                construct_r_xxxx_prealloc::<FullRange, _>(&mut r_buffer, &mut r, p, &pos, lqn_ab, q, &pos_cd, lqn_cd, omega);
                std::hint::black_box(r.get_data()[0]);
            });
            // Boys and the R recursion alone, the part of the above that is not the gather.
            let ns_r = time(iters, |i| {
                let pcx = 0.5 + i as f64 * 1e-9;
                r_buffer.compute(p * q / (p + q), pcx, -1.0, 0.4, pcx * pcx + 1.16);
                std::hint::black_box(r_buffer.view_tuv().get_data()[0]);
            });
            construct_r_xxxx_prealloc::<FullRange, _>(&mut r_buffer, &mut r, p, &pos_ab, lqn_ab, q, &pos_cd, lqn_cd, omega);
            let label = format!("xxxx ({lqn_ab}, {lqn_cd})");
            println!("{:<28}{:>12.1}{:>20x}   (Boys + R alone: {:.1})", label, ns, fingerprint(r.get_data()), ns_r);
        }
    }
}
