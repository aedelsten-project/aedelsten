// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Cartesian Gaussian components and their normalization, and the transformation from Cartesian to real spherical
//! harmonic Gaussians (H. B. Schlegel and M. J. Frisch, Int. J. Quantum Chem. 54, 83 (1995)).

use std::f64::consts::PI;

use arrays::{
    Array3DTraitMut,
};
use arrays::matrix::*;
use arrays::array4d::*;
use arrays::scratch::{Arena, ArenaError};
use complex::Complex;
use constants::L_MAX;
use math_utilities::cartesian::for_each_cartesian_component;
use math_utilities::special_functions::{binomial, compute_factorial};

/// The number of spherical components of angular momentum `l`, `2l + 1`.
#[inline(always)]
pub fn compute_spherical_degeneracy(l: usize) -> usize {
    2 * l + 1
}

/// The number of Cartesian components of angular momentum `l`, `(l + 1)(l + 2) / 2`.
#[inline(always)]
pub fn compute_cartesian_degeneracy(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

/// The Cartesian components `(lx, ly, lz)` of angular momentum `lqn`, in the library's order (`lx` descending, then
/// `ly` descending: xx, xy, xz, yy, yz, zz for d).
#[inline(always)]
pub fn get_cartesian_quantum_numbers(lqn: usize) -> Vec<(usize, usize, usize)> {
    let cart = compute_cartesian_degeneracy(lqn);
    let mut part: Vec<(usize, usize, usize)> = Vec::with_capacity(cart);
    for_each_cartesian_component(lqn, |lx, ly, lz| part.push((lx, ly, lz)));
    part
}

/// The Cartesian components of angular momentum `lqn` as a `3 x n_cart` matrix, one component `(lx, ly, lz)` per
/// column, in the order of `get_cartesian_quantum_numbers`.
#[inline(always)]
pub fn get_cartesian_quantum_numbers_m(lqn: usize) -> Matrix<usize> {
    let cart = compute_cartesian_degeneracy(lqn);
    let mut matrix = Matrix::<usize>::new(3, cart);
    generate_cartesian_quantum_numbers(&mut matrix, lqn);
    matrix
}

/// `get_cartesian_quantum_numbers_m` in arena storage.
#[inline(always)]
pub fn get_cartesian_quantum_numbers_m_arena<'a>(lqn: usize, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, usize>, ArenaError> {
    let cart = compute_cartesian_degeneracy(lqn);
    let mut matrix = MatrixViewMut::<usize>::from_arena(3, cart, arena)?;
    generate_cartesian_quantum_numbers(&mut matrix, lqn);
    Ok(matrix)
}

#[inline(always)]
fn generate_cartesian_quantum_numbers<M: MatrixTraitMut<usize>>(
    matrix: &mut M,
    lqn: usize
) {
    let mut idx = 0;
    for_each_cartesian_component(lqn, |lx, ly, lz| {
        matrix[(0, idx)] = lx;
        matrix[(1, idx)] = ly;
        matrix[(2, idx)] = lz;
        idx += 1;
    });
}

// `n!` for `n <= 2 L_MAX`.
const MAX_FAC: usize = 2 * L_MAX + 1;
const FACTORIALS: [u64; MAX_FAC] = {
    let mut f = [1u64; MAX_FAC];
    let mut i = 1;
    while i < MAX_FAC {
        f[i] = f[i - 1] * i as u64;
        i += 1;
    }
    f
};

/// The normalization constant of the Cartesian Gaussian primitive `x^lx y^ly z^lz exp(-a r^2)`:
/// `[(2a/pi)^(3/2) (8a)^l lx! ly! lz! / ((2lx)! (2ly)! (2lz)!)]^(1/2)`, `l = lx + ly + lz`.
#[inline(always)]
pub fn compute_cart_gaussian_norm(a: f64, lx: usize, ly: usize, lz: usize) -> f64 {
    let l_total = lx + ly + lz;
    let part1 = (2.0 * a / PI).powi(3).sqrt();
    let part2 = (8.0 * a).powi(l_total as i32);
    let f = |n| FACTORIALS[n];
    let part3 = (f(lx) * f(ly) * f(lz)) as f64;
    let part4 = (f(2 * lx) * f(2 * ly) * f(2 * lz)) as f64;
    (part1 * part2 * part3 / part4).sqrt()
}

/// The component a Cartesian normalization with one constant per shell normalizes to unity (see
/// `shared_cartesian_normalization_ratio`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedCartesianReference {
    /// One constant per shell, pinned to the most evenly split partition of the shell's order
    /// into 3 parts (e.g. the xy-type component for d, xyz-type for f).
    MostEvenlySplit,
    /// One constant per shell, pinned to the axis-aligned pure-power component (xx-type for d,
    /// xxx-type for f).
    AxisAligned,
}

impl SharedCartesianReference {
    /// The convention of MOLCAS.
    pub const MOLCAS: Self = Self::MostEvenlySplit;
    /// The convention of DALTON.
    pub const DALTON: Self = Self::AxisAligned;
    /// The convention of DIRAC.
    pub const DIRAC: Self = Self::AxisAligned;

    /// The convention of its C integer (0 most evenly split, 1 axis-aligned).
    pub fn from_c_int(value: i32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::MostEvenlySplit),
            1 => Ok(Self::AxisAligned),
            _ => Err(format!("Invalid SharedCartesianReference: {}", value)),
        }
    }

    /// The convention's C integer.
    pub fn get_c_int(&self) -> i32 {
        match self {
            Self::MostEvenlySplit => 0,
            Self::AxisAligned => 1,
        }
    }

    /// The names the C interface's integer queries accept, with their values.
    pub const NAME_TABLE: &'static [(&'static str, i32)] =
        &[("cartesian_convention_molcas", 0), ("cartesian_convention_dalton", 1), ("cartesian_convention_dirac", 1)];

    // The reference component `(lx, ly, lz)` of a shell of angular momentum `order`.
    fn reference_partition(self, order: usize) -> (usize, usize, usize) {
        match self {
            Self::MostEvenlySplit => {
                let base = order / 3;
                let extra = order % 3;
                (base + (extra >= 1) as usize, base + (extra >= 2) as usize, base)
            }
            Self::AxisAligned => (order, 0, 0),
        }
    }
}

/// The self-overlap of Cartesian component `(lx, ly, lz)` when its shell is normalized with one constant, that of
/// its reference component under `convention`: the ratio of the components' `lx! ly! lz! / ((2lx)! (2ly)! (2lz)!)`,
/// the reference's over this one's. Independent of the exponents (primitive or contracted); 1 for the reference
/// component and for every component of an s or p shell.
pub fn shared_cartesian_normalization_ratio(lx: usize, ly: usize, lz: usize, convention: SharedCartesianReference) -> f64 {
    let (px, py, pz) = convention.reference_partition(lx + ly + lz);
    let f = |n| FACTORIALS[n] as f64;
    let g = |a: usize, b: usize, c: usize| (f(2 * a) * f(2 * b) * f(2 * c)) / (f(a) * f(b) * f(c));
    g(lx, ly, lz) / g(px, py, pz)
}

/// The value and the gradient with respect to the point, `(value, d/dx, d/dy, d/dz)`, of the unnormalized Cartesian
/// Gaussian primitive `px^lx py^ly pz^lz exp(-a (px^2 + py^2 + pz^2))` at `(px, py, pz)`, the point relative to the
/// shell centre. The normalization (`compute_cart_gaussian_norm`) is the caller's.
#[inline(always)]
pub fn compute_cart_gaussian_value_and_gradient(
    px: f64, py: f64, pz: f64,
    lx: usize, ly: usize, lz: usize,
    a: f64,
) -> (f64, f64, f64, f64) {
    let (lx, ly, lz) = (lx as i32, ly as i32, lz as i32);
    let g = (-a * (px * px + py * py + pz * pz)).exp();
    let mx = px.powi(lx);
    let my = py.powi(ly);
    let mz = pz.powi(lz);

    let dmx = (if lx == 0 { 0.0 } else { lx as f64 * px.powi(lx - 1) }) - 2.0 * a * px * mx;
    let dmy = (if ly == 0 { 0.0 } else { ly as f64 * py.powi(ly - 1) }) - 2.0 * a * py * my;
    let dmz = (if lz == 0 { 0.0 } else { lz as f64 * pz.powi(lz - 1) }) - 2.0 * a * pz * mz;

    let value = mx * my * mz * g;
    (value, dmx * my * mz * g, mx * dmy * mz * g, mx * my * dmz * g)
}

/// The transformation of a shell pair of angular momenta `l1`, `l2` from Cartesian to spherical components, the
/// product of the two shells' (`compute_cartesian_to_spherical_matrix`): rows the Cartesian pairs `ca + cb n_cart1`,
/// columns the spherical pairs `sa + sb n_sph1`.
#[inline(always)]
pub fn build_c2s_matrix(l1: usize, l2: usize) -> Matrix<f64> {
    let l1_matrix = compute_cartesian_to_spherical_matrix(l1);
    let l2_matrix = compute_cartesian_to_spherical_matrix(l2);
    let mut combined = Array4D::<f64>::new(
        l1_matrix.rows(),
        l2_matrix.rows(),
        l1_matrix.cols(),
        l2_matrix.cols(),
    );
    for (mut l, col_b) in combined.iter_mut().zip(l2_matrix.iter_col()) {
        for (mut k, col_a) in l.iter_mut().zip(l1_matrix.iter_col()) {
            for (j, b) in k.iter_col_mut().zip(col_b.iter()) {
                for (i, a) in j.iter_mut().zip(col_a.iter()) {
                    *i = a * b;
                }
            }
        }
    }
    combined.into_matrix((2, 2))
}

/// `build_c2s_matrix` in arena storage (the two shells' matrices too).
#[inline(always)]
pub fn build_c2s_matrix_arena<'a>(l1: usize, l2: usize, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError> {
    let l1_matrix = compute_cartesian_to_spherical_matrix_arena(l1, arena)?;
    let l2_matrix = compute_cartesian_to_spherical_matrix_arena(l2, arena)?;
    let mut combined = Array4DViewMut::<f64>::from_arena(
        l1_matrix.rows(),
        l2_matrix.rows(),
        l1_matrix.cols(),
        l2_matrix.cols(),
        arena
    )?;
    for (mut l, col_b) in combined.iter_mut().zip(l2_matrix.iter_col()) {
        for (mut k, col_a) in l.iter_mut().zip(l1_matrix.iter_col()) {
            for (j, b) in k.iter_col_mut().zip(col_b.iter()) {
                for (i, a) in j.iter_mut().zip(col_a.iter()) {
                    *i = a * b;
                }
            }
        }
    }
    Ok(combined.into_matrix((2, 2)))
}

/// The `n_cart x n_sph` transformation of angular momentum `l` from Cartesian components (rows, in the order of
/// `get_cartesian_quantum_numbers`) to real spherical harmonic ones (columns, `m = -l..l`); the identity for `l <= 1`,
/// so p stays x, y, z.
#[inline(always)]
pub fn compute_cartesian_to_spherical_matrix(l: usize) -> Matrix<f64> {
    let cart_c = (compute_cartesian_degeneracy(l)) as usize;
    let sph_c = (compute_spherical_degeneracy(l)) as usize;
    let mut c2s_matrix = Matrix::<f64>::new(cart_c, sph_c);
    cartesian_to_spherical_coeffs(&mut c2s_matrix, l);
    c2s_matrix
}

/// `compute_cartesian_to_spherical_matrix` in arena storage.
#[inline(always)]
pub fn compute_cartesian_to_spherical_matrix_arena<'a>(l: usize, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError> {
    let cart_c = (compute_cartesian_degeneracy(l)) as usize;
    let sph_c = (compute_spherical_degeneracy(l)) as usize;
    let mut c2s_matrix = MatrixViewMut::<f64>::from_arena(cart_c, sph_c, arena)?;
    cartesian_to_spherical_coeffs(&mut c2s_matrix, l);
    Ok(c2s_matrix)
}

// Fills `c2s_matrix` (zeroed, `n_cart x n_sph`) with the real combinations of the complex coefficients: `m > 0`
// `Re(c_m + c_-m) / sqrt 2`, `m < 0` `-Im(c_m - c_-m) / sqrt 2`, `m = 0` `Re c_0`.
#[inline(always)]
fn cartesian_to_spherical_coeffs<M: MatrixTraitMut<f64>>(
    c2s_matrix: &mut M,
    l: usize
) {
    if l > 1 {
        let l: u64 = l as u64;
        let mut ls: usize = 0;
        let sqrt2: f64 = (2.0_f64).sqrt();
        for m in -(l as i64)..=(l as i64) {
            let mut lc: usize = 0;
            for_each_cartesian_component(l as usize, |lx, ly, lz| {
                let lx: u64 = lx as u64;
                let ly: u64 = ly as u64;
                let lz: u64 = lz as u64;
                let c2s: f64;
                if m > 0 {
                    let c2s_complex: Complex<f64> = compute_cart_2_sph_complex(l, m, lx, ly, lz)
                        + compute_cart_2_sph_complex(l, -m, lx, ly, lz);
                    c2s = c2s_complex.re() / sqrt2;
                } else if m < 0 {
                    let c2s_complex: Complex<f64> = compute_cart_2_sph_complex(l, m, lx, ly, lz)
                        - compute_cart_2_sph_complex(l, -m, lx, ly, lz);
                    c2s = -c2s_complex.im() / sqrt2;
                } else {
                    let c2s_complex: Complex<f64> = compute_cart_2_sph_complex(l, m, lx, ly, lz);
                    c2s = c2s_complex.re();
                }
                c2s_matrix[(lc, ls)] = c2s;
                lc += 1;
            });
            ls += 1;
        }
    } else {
        for i in 0..compute_cartesian_degeneracy(l) {
            c2s_matrix[(i, i)] = 1.0;
        }
    }
}

/// The coefficient of the Cartesian component `(lx, ly, lz)` in the complex spherical harmonic Gaussian `(l, m)`
/// (Schlegel and Frisch); zero unless `(lx + ly - |m|) / 2` is an integer.
#[inline(always)]
pub fn compute_cart_2_sph_complex(l: u64, m: i64, lx: u64, ly: u64, lz: u64) -> Complex<f64> {
    let mut c2s: Complex<f64> = Complex::<f64>::new(0.0, 0.0);
    let m_abs: u64 = m.abs() as u64;
    let j: f64 = compute_j(lx as f64, ly as f64, m_abs as f64);
    if integer_test(j) {
        let j: i64 = convert_to_int(j);
        let cg0: f64 = compute_cg0(l, m_abs, lx, ly, lz);
        let cg1: f64 = compute_cg1(l, m_abs, j);
        let cg2: Complex<f64> = compute_cg2(m, m_abs, lx, j);
        c2s = cg2.mulf(cg1 * cg0);
    }
    c2s
}

// Whether `x` is within 1e-15 of an integer.
#[inline(always)]
fn integer_test(x: f64) -> bool {
    let mut test: bool = false;
    let threshold: f64 = 1.0e-15;
    if (x - x.floor()).abs() < threshold {
        test = true;
    } else if (x - x.ceil()).abs() < threshold {
        test = true;
    }
    return test;
}

// The integer nearest `x`.
#[inline(always)]
fn convert_to_int(x: f64) -> i64 {
    let floor_diff: f64 = (x - x.floor()).abs();
    let ceil_diff: f64 = (x - x.ceil()).abs();
    if ceil_diff < floor_diff {
        return x.ceil() as i64;
    } else {
        return x.floor() as i64;
    }
}

#[inline(always)]
fn compute_j(lx: f64, ly: f64, m_abs: f64) -> f64 {
    (lx + ly - m_abs) / 2.0
}

// The normalization factor of the coefficient, the square root of its factorial ratio.
#[inline(always)]
fn compute_cg0(l: u64, m_abs: u64, lx: u64, ly: u64, lz: u64) -> f64 {
    let mut cg0: f64 = 0.0;
    let part1: u64 = compute_factorial(2 * lx)
        * compute_factorial(2 * ly)
        * compute_factorial(2 * lz)
        * compute_factorial(l)
        * compute_factorial(l - m_abs);
    let part1: f64 = part1 as f64;
    let part2: f64 = compute_factorial(2 * l) as f64;
    let part3: f64 = compute_factorial(lx) as f64;
    let part4: f64 = compute_factorial(ly) as f64;
    let part5: f64 = compute_factorial(lz) as f64;
    let part6: f64 = compute_factorial(l + m_abs) as f64;
    let val: f64 = part1 / part2 / part3 / part4 / part5 / part6;
    if val > 0.0 {
        cg0 = val.sqrt();
    }
    cg0
}

// The sum over `i` of the coefficient, over `2^l l!`.
#[inline(always)]
fn compute_cg1(l: u64, m_abs: u64, j: i64) -> f64 {
    let mut cg1: f64 = 0.0;
    let limit: f64 = ((l - m_abs) as f64) / 2.0;
    let limit: u64 = limit.floor() as u64;
    for i in 0..=limit {
        let val: u64 =
            binomial(l as i64, i as i64) * binomial(i as i64, j) * compute_factorial(2 * l - 2 * i);
        let mut val: f64 = (val as f64) / (compute_factorial(l - m_abs - 2 * i) as f64);
        val *= (-1.0_f64).powf(i as f64);
        cg1 += val;
    }
    let denominator: f64 = (2.0_f64).powf(l as f64) * (compute_factorial(l) as f64);
    cg1 / denominator
}

// The sum over `k` of the coefficient, the complex phases `(-1)^(+-(|m| - lx + 2k) / 2)`.
#[inline(always)]
fn compute_cg2(m: i64, m_abs: u64, lx: u64, j: i64) -> Complex<f64> {
    let mut cg2: Complex<f64> = Complex::<f64>::new(0.0, 0.0);
    let mut sign: f64 = 1.0;
    if m < 0 {
        sign = -1.0;
    }
    let c1: Complex<f64> = Complex::<f64>::new(-1.0, 0.0);
    let m_abs_f64: f64 = m_abs as f64;
    for k in 0..=j {
        let tmp: i64 = (lx as i64) - 2 * k;
        let val: f64 = (binomial(j, k) * binomial(m_abs as i64, tmp)) as f64;
        let tmp: f64 = tmp as f64;
        let pow: f64 = sign * (m_abs_f64 - tmp) / 2.0;
        let mut c2: Complex<f64> = c1.powf(pow);
        c2 = c2.mulf(val);
        cg2 += c2;
    }
    cg2
}

#[cfg(test)]
mod shared_cartesian_normalization_tests {
    use super::*;
    const MOLCAS: SharedCartesianReference = SharedCartesianReference::MOLCAS;
    const DALTON: SharedCartesianReference = SharedCartesianReference::DALTON;

    // Component k is get_cartesian_quantum_numbers(order)[k].
    #[test]
    fn molcas_d_shell_self_overlaps_match_the_reference_values() {
        let ratios: Vec<f64> = get_cartesian_quantum_numbers(2).into_iter().map(|(lx, ly, lz)| shared_cartesian_normalization_ratio(lx, ly, lz, MOLCAS)).collect();
        assert_eq!(ratios, [3.0, 1.0, 1.0, 3.0, 1.0, 3.0]);
    }

    #[test]
    fn molcas_f_shell_self_overlaps_match_the_reference_values() {
        let ratios: Vec<f64> = get_cartesian_quantum_numbers(3).into_iter().map(|(lx, ly, lz)| shared_cartesian_normalization_ratio(lx, ly, lz, MOLCAS)).collect();
        assert_eq!(ratios, [15.0, 3.0, 3.0, 3.0, 1.0, 3.0, 15.0, 3.0, 3.0, 15.0]);
    }

    #[test]
    fn dalton_and_dirac_d_shell_self_overlaps_match_the_axis_aligned_reference() {
        let ratios: Vec<f64> = get_cartesian_quantum_numbers(2).into_iter().map(|(lx, ly, lz)| shared_cartesian_normalization_ratio(lx, ly, lz, DALTON)).collect();
        let expected = [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0 / 3.0, 1.0];
        for (r, e) in ratios.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12, "{r} vs {e}");
        }
        assert_eq!(SharedCartesianReference::DIRAC, DALTON);
    }

    #[test]
    fn dalton_and_dirac_f_shell_self_overlaps_match_the_axis_aligned_reference() {
        let ratios: Vec<f64> = get_cartesian_quantum_numbers(3).into_iter().map(|(lx, ly, lz)| shared_cartesian_normalization_ratio(lx, ly, lz, DALTON)).collect();
        let expected = [1.0, 0.2, 0.2, 0.2, 1.0 / 15.0, 0.2, 1.0, 0.2, 0.2, 1.0];
        for (r, e) in ratios.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12, "{r} vs {e}");
        }
    }

    #[test]
    fn s_and_p_shells_need_no_rescaling() {
        for convention in [MOLCAS, DALTON] {
            for order in [0, 1] {
                for (lx, ly, lz) in get_cartesian_quantum_numbers(order) {
                    assert_eq!(shared_cartesian_normalization_ratio(lx, ly, lz, convention), 1.0);
                }
            }
        }
    }
}
