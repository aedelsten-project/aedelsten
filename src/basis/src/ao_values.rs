// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The values, gradients and second derivatives of the AOs of a list of shells at a set of points, without allocating.
//! Per shell, built once: the contraction coefficients with each primitive's radial normalization
//! `[(2a/pi)^(3/2) (4a)^l]^(1/2)` folded in, and per angular momentum the Cartesian-to-output transformation with the
//! component normalization `[2^l lx! ly! lz! / ((2lx)! (2ly)! (2lz)!)]^(1/2)` folded in
//! (`cart2sph::compute_cart_gaussian_norm` is their product). Per shell and point: `exp(-a r^2)` once per primitive,
//! contracted to one radial factor `R_k` per contracted function (and `R'_k`, the coefficients times `-2a`, and
//! `R''_k`, times `4a^2`); the angular factors `Y_s` and their first and second derivatives from power tables. With
//! `d = r - A`: the AO `R_k Y_s`, its gradient `R_k grad Y_s + R'_k Y_s d`, its second derivatives
//! `R_k Y_s,ij + R'_k (d_i Y_s,j + d_j Y_s,i + delta_ij Y_s) + R''_k d_i d_j Y_s`.

use arrays::matrix::MatrixTrait;
use constants::L_MAX;

use crate::contraction::ShellKind;
use crate::shell_types::{BasisShell, HarmonicType};

// The scratch of a shell's points (`5 + n_cont` per point, `n_cont` more with gradients, `2 n_cont` more with second
// derivatives), on the stack.
const SCRATCH_LEN: usize = 8192;
// The most points of one pass over a shell.
const MAX_CHUNK: usize = 128;
const MAX_CART: usize = (L_MAX + 1) * (L_MAX + 2) / 2;

/// One shell of the basis as `get_shells` lists it: its centre (0-based), its angular momentum (negated for a
/// Cartesian shell, as in `BasisFunctionId`), its first AO and its AO count.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShellInfo {
    /// The centre, 0-based.
    pub centre: usize,
    /// The angular momentum, negated for a Cartesian shell.
    pub l: i32,
    /// The first AO.
    pub first_ao: usize,
    /// The AO count.
    pub n_ao: usize,
}

/// Why an AO evaluation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AoValuesError {
    /// The points have a row count other than 3 (x, y, z) or 4 (x, y, z, a weight that is not read).
    PointRows(usize),
    /// The length of the points is not a multiple of their row count.
    PointsLength { rows: usize, len: usize },
    /// A shell index is not below the number of shells.
    ShellIndex { index: usize, n_shells: usize },
    /// An output buffer has the wrong length.
    OutputLength { expected: usize, actual: usize },
}

impl std::fmt::Display for AoValuesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PointRows(rows) => write!(f, "points must have 3 rows (x, y, z) or 4 (x, y, z, weight), not {rows}"),
            Self::PointsLength { rows, len } => write!(f, "{len} point coordinates are not a multiple of {rows} rows"),
            Self::ShellIndex { index, n_shells } => write!(f, "shell {index} is out of range ({n_shells} shells)"),
            Self::OutputLength { expected, actual } => write!(f, "expected an output of length {expected}, got {actual}"),
        }
    }
}

impl std::error::Error for AoValuesError {}

// A primitive's contribution to a contracted function's radial factors.
#[derive(Debug, Clone, Copy)]
struct Term {
    prim: usize,
    cont: usize,
    coefficient: f64,
    gradient_coefficient: f64,
    hessian_coefficient: f64,
}

#[derive(Debug, Clone)]
struct ShellData {
    info: ShellInfo,
    l: usize,
    angular: usize,
    n_cont: usize,
    n_comp: usize,
    exponents: std::ops::Range<usize>,
    terms: std::ops::Range<usize>,
    smallest_exponent: f64,
}

// The components of angular momentum `l` and their transformation to the output components, `(Cartesian, output,
// factor)` for every nonzero factor.
#[derive(Debug, Clone)]
struct AngularTable {
    components: Box<[[usize; 3]]>,
    transform: Box<[(usize, usize, f64)]>,
    // max over output components of the sum of the factors' magnitudes: |Y_s| <= bound r^l.
    bound: f64,
}

/// What the evaluation needs of every shell of a basis, built once (`new`).
#[derive(Debug, Clone)]
pub struct AoEvaluator {
    shells: Box<[ShellData]>,
    exponents: Box<[f64]>,
    terms: Box<[Term]>,
    // Index `2 l` Cartesian, `2 l + 1` spherical.
    angular: Box<[AngularTable]>,
}

fn angular_table(l: usize, harmonic_type: HarmonicType) -> AngularTable {
    let components: Box<[[usize; 3]]> =
        cart2sph::get_cartesian_quantum_numbers(l).into_iter().map(|(x, y, z)| [x, y, z]).collect();
    let factorial = |n: usize| (1..=n).map(|k| k as f64).product::<f64>();
    let norm: Vec<f64> = components
        .iter()
        .map(|&[x, y, z]| {
            (2f64.powi(l as i32) * factorial(x) * factorial(y) * factorial(z)
                / (factorial(2 * x) * factorial(2 * y) * factorial(2 * z)))
                .sqrt()
        })
        .collect();
    let mut transform = Vec::new();
    let n_comp = match harmonic_type {
        HarmonicType::Cartesian => {
            transform.extend(norm.iter().enumerate().map(|(c, &n)| (c, c, n)));
            components.len()
        }
        HarmonicType::Spherical => {
            let c2s = cart2sph::compute_cartesian_to_spherical_matrix(l);
            for s in 0..c2s.cols() {
                for (c, &t) in c2s.get_col(s).iter().enumerate() {
                    if t != 0.0 {
                        transform.push((c, s, norm[c] * t));
                    }
                }
            }
            c2s.cols()
        }
    };
    let mut sums = vec![0.0; n_comp];
    for &(_, s, t) in &transform {
        sums[s] += t.abs();
    }
    AngularTable { components, transform: transform.into_boxed_slice(), bound: sums.iter().copied().fold(0.0, f64::max) }
}

impl AoEvaluator {
    /// The evaluation data of `basis_shells`, their AO order the basis's (`get_basis_index`).
    pub fn new<S: ShellKind>(basis_shells: &[BasisShell<S>]) -> Self {
        let mut shells = Vec::with_capacity(basis_shells.len());
        let mut exponents = Vec::new();
        let mut terms = Vec::new();
        for shell in basis_shells {
            let l = shell.get_lqn();
            let n_prim = shell.get_n_prim();
            let n_cont = shell.get_n_cont();
            let coefficients = S::amplitude_coefficients(shell.get_coefficients(), n_prim);
            let first_term = terms.len();
            for (p, &a) in shell.get_exponents().iter().enumerate() {
                let radial_norm = ((2.0 * a / std::f64::consts::PI).powi(3).sqrt() * (4.0 * a).powi(l as i32)).sqrt();
                for k in 0..n_cont {
                    let c = match &coefficients {
                        Some(matrix) => matrix.get_col(k)[p],
                        None => if p == k { 1.0 } else { 0.0 },
                    };
                    if c != 0.0 {
                        let coefficient = c * radial_norm;
                        terms.push(Term {
                            prim: p,
                            cont: k,
                            coefficient,
                            gradient_coefficient: -2.0 * a * coefficient,
                            hessian_coefficient: 4.0 * a * a * coefficient,
                        });
                    }
                }
            }
            let harmonic_type = shell.get_harmonic_type();
            let n_comp = shell.get_degeneracy();
            let signed_l = if harmonic_type == HarmonicType::Cartesian { -(l as i32) } else { l as i32 };
            shells.push(ShellData {
                info: ShellInfo { centre: shell.get_centre_index(), l: signed_l, first_ao: shell.get_basis_index(), n_ao: n_cont * n_comp },
                l,
                angular: 2 * l + (harmonic_type == HarmonicType::Spherical) as usize,
                n_cont,
                n_comp,
                exponents: exponents.len()..exponents.len() + n_prim,
                terms: first_term..terms.len(),
                smallest_exponent: shell.get_exponents().iter().copied().fold(f64::INFINITY, f64::min),
            });
            exponents.extend_from_slice(shell.get_exponents());
        }
        let max_l = shells.iter().map(|s| s.l).max().unwrap_or(0);
        let angular = (0..=max_l)
            .flat_map(|l| [angular_table(l, HarmonicType::Cartesian), angular_table(l, HarmonicType::Spherical)])
            .collect();
        AoEvaluator { shells: shells.into_boxed_slice(), exponents: exponents.into_boxed_slice(), terms: terms.into_boxed_slice(), angular }
    }

    /// The number of shells.
    pub fn n_shells(&self) -> usize {
        self.shells.len()
    }

    /// Shell `s`.
    pub fn shell_info(&self, s: usize) -> ShellInfo {
        self.shells[s].info
    }

    /// The distance from shell `s`'s centre beyond which each of its AOs and each Cartesian component of their
    /// gradients is below `threshold` in magnitude, from the bound `B sum_p w_p r^l (1 + l + 2 a_p r) exp(-a_p r^2)`
    /// (valid for `r >= 1`), `w_p` the largest normalized coefficient of primitive `p` and `B r^l` a bound on the
    /// angular factors. Never below 1 bohr.
    pub fn shell_extent(&self, s: usize, threshold: f64) -> f64 {
        let shell = &self.shells[s];
        let lf = shell.l as f64;
        let angular_bound = self.angular[shell.angular].bound;
        let exponents = &self.exponents[shell.exponents.clone()];
        let terms = &self.terms[shell.terms.clone()];
        let bound = |r: f64| -> f64 {
            let mut sum = 0.0;
            for (p, &a) in exponents.iter().enumerate() {
                let w = terms.iter().filter(|t| t.prim == p).map(|t| t.coefficient.abs()).fold(0.0, f64::max);
                sum += w * (1.0 + lf + 2.0 * a * r) * (-a * r * r).exp();
            }
            angular_bound * r.powi(shell.l as i32) * sum
        };
        // Every term decreases beyond the outermost maximum, of r^(l+1) exp(-a r^2) at sqrt((l + 1) / (2a)).
        let mut lo = ((lf + 1.0) / (2.0 * shell.smallest_exponent)).sqrt().max(1.0);
        if bound(lo) < threshold {
            return lo;
        }
        let mut hi = 2.0 * lo;
        while bound(hi) >= threshold {
            lo = hi;
            hi *= 2.0;
        }
        while hi - lo > 1e-3 * hi {
            let mid = 0.5 * (lo + hi);
            if bound(mid) < threshold { hi = mid } else { lo = mid }
        }
        hi
    }

    /// The number of AOs of the shells `shells`, checked against the number of shells.
    pub fn count_aos(&self, shells: impl Iterator<Item = usize>) -> Result<usize, AoValuesError> {
        let n_shells = self.shells.len();
        let mut n = 0;
        for index in shells {
            let shell = self.shells.get(index).ok_or(AoValuesError::ShellIndex { index, n_shells })?;
            n += shell.info.n_ao;
        }
        Ok(n)
    }

    /// The values (`values`, `(n_listed, n_points)` column-major), the gradients (`gradients`, three such blocks: x, y,
    /// z) and the second derivatives (`hessians`, six such blocks: xx, xy, xz, yy, yz, zz) of the AOs of `shells`, in
    /// the order of the list, at `points` (`(point_rows, n_points)` column-major, x, y, z in rows 0-2, `point_rows` 3
    /// or 4). `basis_shells` are the shells this was built from (their positions are read here). Any output may be
    /// `None`. Allocates nothing.
    pub fn evaluate<S: ShellKind>(
        &self,
        basis_shells: &[BasisShell<S>],
        shells: impl Iterator<Item = usize> + Clone,
        points: &[f64],
        point_rows: usize,
        mut values: Option<&mut [f64]>,
        mut gradients: Option<[&mut [f64]; 3]>,
        mut hessians: Option<[&mut [f64]; 6]>,
    ) -> Result<(), AoValuesError> {
        if point_rows != 3 && point_rows != 4 {
            return Err(AoValuesError::PointRows(point_rows));
        }
        if points.len() % point_rows != 0 {
            return Err(AoValuesError::PointsLength { rows: point_rows, len: points.len() });
        }
        let n_points = points.len() / point_rows;
        let n_listed = self.count_aos(shells.clone())?;
        let expected = n_listed * n_points;
        let check = |len: usize| if len == expected { Ok(()) } else { Err(AoValuesError::OutputLength { expected, actual: len }) };
        if let Some(values) = &values {
            check(values.len())?;
        }
        if let Some(gradients) = &gradients {
            for g in gradients {
                check(g.len())?;
            }
        }
        if let Some(hessians) = &hessians {
            for h in hessians {
                check(h.len())?;
            }
        }

        let with_hessian = hessians.is_some();
        let with_gradient = gradients.is_some() || with_hessian;
        let mut scratch = [0.0f64; SCRATCH_LEN];
        let mut row = 0;
        for s in shells {
            let shell = &self.shells[s];
            let centre = basis_shells[s].get_position();
            let per_point = 5 + shell.n_cont * (1 + with_gradient as usize + with_hessian as usize);
            let chunk = (SCRATCH_LEN / per_point).min(MAX_CHUNK);
            assert!(chunk > 0, "a shell of {} contracted functions exceeds the AO evaluation's scratch", shell.n_cont);
            let exponents = &self.exponents[shell.exponents.clone()];
            let terms = &self.terms[shell.terms.clone()];
            let table = &self.angular[shell.angular];
            let n_comp = shell.n_comp;
            let radial_len = shell.n_cont * chunk;
            let mut start = 0;
            while start < n_points {
                let n = chunk.min(n_points - start);
                let (geometry, radial_part) = scratch.split_at_mut(5 * chunk);
                let (dx, geometry) = geometry.split_at_mut(chunk);
                let (dy, geometry) = geometry.split_at_mut(chunk);
                let (dz, geometry) = geometry.split_at_mut(chunk);
                let (r2, e) = geometry.split_at_mut(chunk);
                for i in 0..n {
                    let p = &points[(start + i) * point_rows..];
                    dx[i] = p[0] - centre[0];
                    dy[i] = p[1] - centre[1];
                    dz[i] = p[2] - centre[2];
                    r2[i] = dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i];
                }
                let (radial, radial_part) = radial_part.split_at_mut(radial_len);
                let (radial_gradient, radial_part) = radial_part.split_at_mut(if with_gradient { radial_len } else { 0 });
                let radial_hessian = &mut radial_part[..if with_hessian { radial_len } else { 0 }];
                radial.fill(0.0);
                radial_gradient.fill(0.0);
                radial_hessian.fill(0.0);
                let mut current = usize::MAX;
                for term in terms {
                    if term.prim != current {
                        current = term.prim;
                        let a = exponents[current];
                        for i in 0..n {
                            e[i] = (-a * r2[i]).exp();
                        }
                    }
                    let k = term.cont * chunk..term.cont * chunk + n;
                    for (r, &e) in radial[k.clone()].iter_mut().zip(e.iter()) {
                        *r += term.coefficient * e;
                    }
                    if with_gradient {
                        for (r, &e) in radial_gradient[k.clone()].iter_mut().zip(e.iter()) {
                            *r += term.gradient_coefficient * e;
                        }
                    }
                    if with_hessian {
                        for (r, &e) in radial_hessian[k].iter_mut().zip(e.iter()) {
                            *r += term.hessian_coefficient * e;
                        }
                    }
                }

                let mut angular = [0.0; MAX_CART];
                let mut angular_gradient = [[0.0; MAX_CART]; 3];
                let mut angular_hessian = [[0.0; MAX_CART]; 6];
                for i in 0..n {
                    let (x, y, z) = (dx[i], dy[i], dz[i]);
                    let (mut xp, mut yp, mut zp) = ([1.0; L_MAX + 1], [1.0; L_MAX + 1], [1.0; L_MAX + 1]);
                    for k in 1..=shell.l {
                        xp[k] = xp[k - 1] * x;
                        yp[k] = yp[k - 1] * y;
                        zp[k] = zp[k - 1] * z;
                    }
                    // The first and second derivatives of x^l from the power table.
                    let d1 = |p: &[f64; L_MAX + 1], l: usize| if l > 0 { l as f64 * p[l - 1] } else { 0.0 };
                    let d2 = |p: &[f64; L_MAX + 1], l: usize| if l > 1 { (l * (l - 1)) as f64 * p[l - 2] } else { 0.0 };
                    angular[..n_comp].fill(0.0);
                    if with_gradient {
                        angular_gradient.iter_mut().for_each(|g| g[..n_comp].fill(0.0));
                    }
                    if with_hessian {
                        angular_hessian.iter_mut().for_each(|h| h[..n_comp].fill(0.0));
                    }
                    for &(c, s, t) in table.transform.iter() {
                        let [lx, ly, lz] = table.components[c];
                        let (x0, y0, z0) = (xp[lx], yp[ly], zp[lz]);
                        angular[s] += t * x0 * y0 * z0;
                        if with_gradient {
                            let (x1, y1, z1) = (d1(&xp, lx), d1(&yp, ly), d1(&zp, lz));
                            angular_gradient[0][s] += t * x1 * y0 * z0;
                            angular_gradient[1][s] += t * x0 * y1 * z0;
                            angular_gradient[2][s] += t * x0 * y0 * z1;
                            if with_hessian {
                                let h = &mut angular_hessian;
                                h[0][s] += t * d2(&xp, lx) * y0 * z0;
                                h[1][s] += t * x1 * y1 * z0;
                                h[2][s] += t * x1 * y0 * z1;
                                h[3][s] += t * x0 * d2(&yp, ly) * z0;
                                h[4][s] += t * x0 * y1 * z1;
                                h[5][s] += t * x0 * y0 * d2(&zp, lz);
                            }
                        }
                    }
                    let first = (start + i) * n_listed + row;
                    if let Some(values) = values.as_deref_mut() {
                        let out = &mut values[first..first + shell.info.n_ao];
                        for (k, out) in out.chunks_exact_mut(n_comp).enumerate() {
                            let r = radial[k * chunk + i];
                            for (o, &y) in out.iter_mut().zip(&angular[..n_comp]) {
                                *o = r * y;
                            }
                        }
                    }
                    let displacement = [x, y, z];
                    if let Some(gradients) = gradients.as_mut() {
                        for (d, g) in gradients.iter_mut().enumerate() {
                            let out = &mut g[first..first + shell.info.n_ao];
                            for (k, out) in out.chunks_exact_mut(n_comp).enumerate() {
                                let r = radial[k * chunk + i];
                                let rd = radial_gradient[k * chunk + i] * displacement[d];
                                for ((o, &y), &gy) in out.iter_mut().zip(&angular[..n_comp]).zip(&angular_gradient[d][..n_comp]) {
                                    *o = r * gy + rd * y;
                                }
                            }
                        }
                    }
                    if let Some(hessians) = hessians.as_mut() {
                        for (h, (out_h, (a, b))) in
                            hessians.iter_mut().zip([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]).enumerate()
                        {
                            let out = &mut out_h[first..first + shell.info.n_ao];
                            let (da, db) = (displacement[a], displacement[b]);
                            let diagonal = if a == b { 1.0 } else { 0.0 };
                            for (k, out) in out.chunks_exact_mut(n_comp).enumerate() {
                                let r = radial[k * chunk + i];
                                let r1 = radial_gradient[k * chunk + i];
                                let r2 = radial_hessian[k * chunk + i];
                                for (s, o) in out.iter_mut().enumerate() {
                                    let y = angular[s];
                                    *o = r * angular_hessian[h][s]
                                        + r1 * (da * angular_gradient[b][s] + db * angular_gradient[a][s] + diagonal * y)
                                        + r2 * da * db * y;
                                }
                            }
                        }
                    }
                }
                start += n;
            }
            row += shell.info.n_ao;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // The evaluator against the per-point kernels of `amplitudes` (the reference), for every l to L_MAX, both
    // harmonic types and the three contraction kinds; gradients also against finite differences; shell lists,
    // point layouts and extents.
    use super::*;
    use arrays::matrix::{Matrix, MatrixTraitMut};
    use crate::contraction::{Contracted, Segmented, Uncontracted};

    // Per l and harmonic type: three primitives, two contracted functions (one coefficient zero), off the origin.
    fn general_shells() -> Vec<BasisShell<Contracted>> {
        let mut shells = Vec::new();
        for l in 0..=L_MAX {
            for (h, harmonic_type) in [HarmonicType::Cartesian, HarmonicType::Spherical].into_iter().enumerate() {
                let coefficients = Matrix::<f64>::new_from_vec(3, 2, vec![0.4, 0.5, 0.3, 0.0, -0.7, 0.6].into_boxed_slice());
                let position = [0.3 * l as f64 - 0.5, 0.2 * h as f64 + 0.1, -0.4 + 0.1 * l as f64];
                shells.push(BasisShell::for_test(l, harmonic_type, &[3.1, 0.9, 0.25], coefficients, l, position));
            }
        }
        shells
    }

    fn indexed<S: ShellKind>(mut shells: Vec<BasisShell<S>>) -> Vec<BasisShell<S>> {
        crate::fix_basis_indices_with_ids(&mut shells);
        shells
    }

    // `(4, n)`: on two shell centres and off them, a weight row.
    fn points<S: ShellKind>(shells: &[BasisShell<S>]) -> Matrix<f64> {
        let mut columns = vec![shells[0].get_position(), shells[shells.len() - 1].get_position()];
        for i in 0..37 {
            let t = i as f64;
            columns.push([(0.7 * t).sin() * 1.8, (1.3 * t).cos() * 1.5, 0.11 * t - 2.0]);
        }
        let mut grid = Matrix::<f64>::new(4, columns.len());
        for (col, p) in grid.iter_col_mut().zip(&columns) {
            col.copy_from_slice(&[p[0], p[1], p[2], 0.5]);
        }
        grid
    }

    fn assert_close(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: lengths");
        let scale = want.iter().fold(1.0f64, |m, w| m.max(w.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= 1e-12 * scale, "{what}, element {i}: {g} vs reference {w}");
        }
    }

    fn check_against_reference<S: ShellKind>(shells: &[BasisShell<S>]) {
        let evaluator = AoEvaluator::new(shells);
        let grid = points(shells);
        let n_points = grid.cols();
        let n_ao: usize = shells.iter().map(|s| s.get_n_cont() * s.get_degeneracy()).sum();
        let mut reference = Matrix::<f64>::new(n_ao, n_points);
        let mut reference_gradients = [0, 1, 2].map(|_| Matrix::<f64>::new(n_ao, n_points));
        crate::amplitudes::compute_ao_amplitudes(shells, &mut reference, &grid);
        let [rx, ry, rz] = &mut reference_gradients;
        crate::amplitudes::compute_ao_amplitude_gradients(shells, rx, ry, rz, &grid);

        let mut values = vec![0.0; n_ao * n_points];
        let mut gradients = [0, 1, 2].map(|_| vec![0.0; n_ao * n_points]);
        let [gx, gy, gz] = &mut gradients;
        evaluator.evaluate(shells, 0..shells.len(), grid.get_data(), 4, Some(&mut values), Some([gx, gy, gz]), None).unwrap();
        assert_close(&values, reference.get_data(), "values");
        for d in 0..3 {
            assert_close(&gradients[d], reference_gradients[d].get_data(), "gradients");
        }

        // The values alone, the gradients alone.
        let mut alone = vec![0.0; n_ao * n_points];
        evaluator.evaluate(shells, 0..shells.len(), grid.get_data(), 4, Some(&mut alone), None, None).unwrap();
        assert_eq!(alone, values);
        let mut alone = [0, 1, 2].map(|_| vec![0.0; n_ao * n_points]);
        let [ax, ay, az] = &mut alone;
        evaluator.evaluate(shells, 0..shells.len(), grid.get_data(), 4, None, Some([ax, ay, az]), None).unwrap();
        assert_eq!(alone, gradients);
    }

    #[test]
    fn general_shells_match_the_reference() {
        check_against_reference(&indexed(general_shells()));
    }

    #[test]
    fn segmented_shells_match_the_reference() {
        check_against_reference(&indexed(BasisShell::<Segmented>::reduce(&general_shells())));
    }

    #[test]
    fn uncontracted_shells_match_the_reference() {
        check_against_reference(&indexed(BasisShell::<Uncontracted>::reduce(&general_shells())));
    }

    #[test]
    fn gradients_match_finite_differences() {
        let shells = indexed(general_shells());
        let evaluator = AoEvaluator::new(&shells);
        let grid = points(&shells);
        let n_ao = evaluator.count_aos(0..shells.len()).unwrap();
        let h = 1e-5;
        for point in grid.iter_col() {
            let mut gradients = [0, 1, 2].map(|_| vec![0.0; n_ao]);
            let [gx, gy, gz] = &mut gradients;
            evaluator.evaluate(&shells, 0..shells.len(), point, 4, None, Some([gx, gy, gz]), None).unwrap();
            for d in 0..3 {
                let shifted = |step: f64| {
                    let mut p = [point[0], point[1], point[2]];
                    p[d] += step;
                    let mut v = vec![0.0; n_ao];
                    evaluator.evaluate(&shells, 0..shells.len(), &p, 3, Some(&mut v), None, None).unwrap();
                    v
                };
                let (plus, minus) = (shifted(h), shifted(-h));
                for mu in 0..n_ao {
                    let difference = (plus[mu] - minus[mu]) / (2.0 * h);
                    let tolerance = 1e-6 * (1.0 + gradients[d][mu].abs());
                    assert!((gradients[d][mu] - difference).abs() < tolerance, "AO {mu}, direction {d}: {} vs {difference}", gradients[d][mu]);
                }
            }
        }
    }

    // The second derivatives against central differences of the gradients, for every l, both harmonic types and the
    // three contraction kinds; the values and gradients alongside them are those of the path without them, bitwise.
    #[test]
    fn second_derivatives_match_finite_differences_of_the_gradients() {
        let general = indexed(general_shells());
        check_second_derivatives(&general);
        check_second_derivatives(&indexed(BasisShell::<Segmented>::reduce(&general_shells())));
        check_second_derivatives(&indexed(BasisShell::<Uncontracted>::reduce(&general_shells())));
    }

    fn check_second_derivatives<S: ShellKind>(shells: &[BasisShell<S>]) {
        let evaluator = AoEvaluator::new(shells);
        let grid = points(shells);
        let n_ao = evaluator.count_aos(0..shells.len()).unwrap();
        let h = 1e-5;
        let gradients_at = |p: &[f64]| {
            let mut g = [0, 1, 2].map(|_| vec![0.0; n_ao]);
            let [gx, gy, gz] = &mut g;
            evaluator.evaluate(shells, 0..shells.len(), p, 3, None, Some([gx, gy, gz]), None).unwrap();
            g
        };
        for point in grid.iter_col() {
            let mut values = vec![0.0; n_ao];
            let mut gradients = [0, 1, 2].map(|_| vec![0.0; n_ao]);
            let mut hessians = [0, 1, 2, 3, 4, 5].map(|_| vec![0.0; n_ao]);
            let [gx, gy, gz] = &mut gradients;
            let [hxx, hxy, hxz, hyy, hyz, hzz] = &mut hessians;
            evaluator
                .evaluate(shells, 0..shells.len(), point, 4, Some(&mut values), Some([gx, gy, gz]), Some([hxx, hxy, hxz, hyy, hyz, hzz]))
                .unwrap();
            let mut plain_values = vec![0.0; n_ao];
            evaluator.evaluate(shells, 0..shells.len(), point, 4, Some(&mut plain_values), None, None).unwrap();
            assert_eq!(values, plain_values);
            assert_eq!(gradients, gradients_at(&point[..3]));

            for (k, (a, b)) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)].into_iter().enumerate() {
                let shifted = |step: f64| {
                    let mut p = [point[0], point[1], point[2]];
                    p[b] += step;
                    gradients_at(&p)
                };
                let (plus, minus) = (shifted(h), shifted(-h));
                for mu in 0..n_ao {
                    let difference = (plus[a][mu] - minus[a][mu]) / (2.0 * h);
                    let tolerance = 1e-6 * (1.0 + hessians[k][mu].abs());
                    assert!((hessians[k][mu] - difference).abs() < tolerance, "AO {mu}, ({a}, {b}): {} vs {difference}", hessians[k][mu]);
                }
            }
        }
    }

    #[test]
    fn a_shell_list_gives_those_shells_in_its_order() {
        let shells = indexed(general_shells());
        let evaluator = AoEvaluator::new(&shells);
        let grid = points(&shells);
        let n_points = grid.cols();
        let n_ao = evaluator.count_aos(0..shells.len()).unwrap();
        let mut all = vec![0.0; n_ao * n_points];
        evaluator.evaluate(&shells, 0..shells.len(), grid.get_data(), 4, Some(&mut all), None, None).unwrap();

        let list = [7, 2, 11, 0];
        let n_listed = evaluator.count_aos(list.iter().copied()).unwrap();
        let mut listed = vec![0.0; n_listed * n_points];
        evaluator.evaluate(&shells, list.iter().copied(), grid.get_data(), 4, Some(&mut listed), None, None).unwrap();
        for j in 0..n_points {
            let mut row = 0;
            for &s in &list {
                let info = evaluator.shell_info(s);
                for a in 0..info.n_ao {
                    assert_eq!(listed[j * n_listed + row + a], all[j * n_ao + info.first_ao + a]);
                }
                row += info.n_ao;
            }
        }
    }

    #[test]
    fn points_with_three_or_four_rows_and_nothing_else() {
        let shells = indexed(general_shells());
        let evaluator = AoEvaluator::new(&shells);
        let grid = points(&shells);
        let n_points = grid.cols();
        let three: Vec<f64> = grid.iter_col().flat_map(|p| p[..3].to_vec()).collect();
        let n_ao = evaluator.count_aos(0..shells.len()).unwrap();
        let (mut from_four, mut from_three) = (vec![0.0; n_ao * n_points], vec![0.0; n_ao * n_points]);
        evaluator.evaluate(&shells, 0..shells.len(), grid.get_data(), 4, Some(&mut from_four), None, None).unwrap();
        evaluator.evaluate(&shells, 0..shells.len(), &three, 3, Some(&mut from_three), None, None).unwrap();
        assert_eq!(from_four, from_three);

        let two: Vec<f64> = grid.iter_col().flat_map(|p| p[..2].to_vec()).collect();
        let mut out = vec![0.0; n_ao * n_points];
        assert_eq!(evaluator.evaluate(&shells, 0..shells.len(), &two, 2, Some(&mut out), None, None), Err(AoValuesError::PointRows(2)));
        let five: Vec<f64> = grid.iter_col().flat_map(|p| [p[0], p[1], p[2], p[3], 0.0]).collect();
        assert_eq!(evaluator.evaluate(&shells, 0..shells.len(), &five, 5, Some(&mut out), None, None), Err(AoValuesError::PointRows(5)));
        assert_eq!(
            evaluator.evaluate(&shells, [shells.len()].into_iter(), &three, 3, Some(&mut out), None, None),
            Err(AoValuesError::ShellIndex { index: shells.len(), n_shells: shells.len() })
        );
        assert!(matches!(evaluator.evaluate(&shells, 0..shells.len(), &three, 3, Some(&mut out[1..]), None, None), Err(AoValuesError::OutputLength { .. })));
    }

    #[test]
    fn beyond_its_extent_a_shell_is_below_the_threshold() {
        let shells = indexed(general_shells());
        let evaluator = AoEvaluator::new(&shells);
        let threshold = 1e-10;
        let directions = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.48, -0.64], [-0.577, 0.577, 0.577]];
        for s in 0..shells.len() {
            let extent = evaluator.shell_extent(s, threshold);
            let centre = shells[s].get_position();
            let n_ao = evaluator.shell_info(s).n_ao;
            for direction in directions {
                for factor in [1.0, 1.5] {
                    let p: Vec<f64> = (0..3).map(|d| centre[d] + factor * extent * direction[d]).collect();
                    let mut values = vec![0.0; n_ao];
                    let mut gradients = [0, 1, 2].map(|_| vec![0.0; n_ao]);
                    let [gx, gy, gz] = &mut gradients;
                    evaluator.evaluate(&shells, [s].into_iter(), &p, 3, Some(&mut values), Some([gx, gy, gz]), None).unwrap();
                    for v in values.iter().chain(gradients.iter().flatten()) {
                        assert!(v.abs() < threshold, "shell {s} at {factor} x its extent {extent}: {v}");
                    }
                }
            }
            // And not absurdly far out.
            assert!(extent < 40.0, "shell {s}: extent {extent}");
        }
    }
}
