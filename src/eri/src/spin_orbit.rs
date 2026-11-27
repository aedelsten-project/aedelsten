// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Two-electron spin-orbit integrals in the `p x Vp` form of the one-electron picture-change kernels, with the
//! potential of the ket charge distribution in place of `V`:
//!
//! ```text
//!   X_k(ab|cd) = eps_kij ( d_i a  d_j b | c d ) = (d/dA_i d/dB_j - d/dA_j d/dB_i) (ab|cd),   (i, j, k) cyclic,
//! ```
//!
//! the derivatives acting on the bra functions of electron 1 only. With `d/dA = (a/p) d/dP - d/dR` and
//! `d/dB = (b/p) d/dP + d/dR` (`R = B - A`) the terms symmetric in `i, j` cancel and
//!
//! ```text
//!   X_k = sum E^cd (-1)^tau [ (dE^ab/dR_j) R_{t+tau+1_i} - (dE^ab/dR_i) R_{t+tau+1_j} ],
//! ```
//!
//! the `d/dP` derivatives an index shift of the R tensor `R(P - Q)` and the `d/dR` ones the Hermite coefficients'
//! (T. Helgaker and P. R. Taylor, Theor. Chim. Acta 83, 177 (1992)). `X_k` is antisymmetric in `a, b` and symmetric
//! in `c, d`. The spin-other-orbit integrals are the same with bra and ket exchanged.

use arrays::array5d::{Array5D, Array5DTrait};
use arrays::array8d::{Array8DTraitMut, Array8DViewMut};
use arrays::matrix::*;
use arrays::scratch::{Arena, ArenaError};
use basis::contraction::{Contracted, Contraction};
use basis::gto_products::GTOProductTrait;
use basis::shell_pairs::ShellPair;
use mmd::hermite_coulomb_integrals::{FullRange, HermiteCoulombIntegrals4D};
use mmd::hermite_expansion::{hermite_count, hermite_index};
use mmd::ketside_hermite_coefficients::compute_ketside_hermite_derivative_coefficients;

use crate::batch::construct_r_xxxx_prealloc;
use crate::r_matrix::BufferedRMatrix;

/// The bra side of one pair, built once per pair and used for every quartet it is the bra of: per primitive product
/// and component `k`, the `(sph_ab, n_hermite(l_ab + 1))` matrix `B_k` with, for `(i, j, k)` cyclic,
/// `B_k[ab, (t,u,v) + 1_i] += dE/dR_j[(t,u,v), ab]` and `B_k[ab, (t,u,v) + 1_j] -= dE/dR_i[(t,u,v), ab]`: the
/// `dE^ab/dR` coefficients (`compute_ketside_hermite_derivative_coefficients`, same-shell and same-centre pairs
/// included: they do not vanish at `R = 0`) embedded at the Hermite positions of the `d/dP` shift, so that `B_k (R
/// E^cd)` is the primitive quartet's `X_k`.
pub struct SpinOrbitBra {
    rows: usize,
    cols: usize,
    /// `(rows, cols, 3, n_prim)`
    matrices: Box<[f64]>,
}

impl SpinOrbitBra {
    /// The bra side of `pair`; scratch from `scratch`, free again on return.
    pub fn new(pair: &ShellPair<Contracted>, scratch: &mut Arena<'_>) -> Result<Self, ArenaError> {
        let lqn_ab = pair.get_lqn_a() + pair.get_lqn_b();
        let rows = pair.get_degeneracy_a() * pair.get_degeneracy_b();
        let cols = hermite_count(lqn_ab + 1);
        let n_hermite = hermite_count(lqn_ab);
        let n_prim = pair.get_products().len();
        // (tuv, a, b, axis, prim)
        let derivatives = compute_ketside_hermite_derivative_coefficients(pair, 1, scratch)?;
        let derivatives = derivatives.get_data();
        let block = rows * cols;
        let mut matrices = vec![0.0; 3 * block * n_prim].into_boxed_slice();
        let mut s = 0;
        for v in 0..=lqn_ab {
            for u in 0..=(lqn_ab - v) {
                for t in 0..=(lqn_ab - v - u) {
                    debug_assert_eq!(s, hermite_index(t, u, v, lqn_ab));
                    let shifted = [hermite_index(t + 1, u, v, lqn_ab + 1), hermite_index(t, u + 1, v, lqn_ab + 1), hermite_index(t, u, v + 1, lqn_ab + 1)];
                    for ip in 0..n_prim {
                        for k in 0..3 {
                            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                            let target = &mut matrices[(3 * ip + k) * block..][..block];
                            for ab in 0..rows {
                                let de = |axis: usize| derivatives[s + n_hermite * (ab + rows * (axis + 3 * ip))];
                                target[ab + rows * shifted[i]] += de(j);
                                target[ab + rows * shifted[j]] -= de(i);
                            }
                        }
                    }
                    s += 1;
                }
            }
        }
        Ok(SpinOrbitBra { rows, cols, matrices })
    }

    // B_k of primitive product `ip`.
    fn matrix(&self, ip: usize, k: usize) -> MatrixView<'_, f64> {
        let block = self.rows * self.cols;
        MatrixView::new(self.rows, self.cols, &self.matrices[(3 * ip + k) * block..][..block])
    }
}

/// `X_x, X_y, X_z` of the shell quartet `(bra | ket)` into `output[0..3]`, OVERWRITTEN, each with the layout of the
/// ERI energy batch: `(i, j, k, l, c, d, a, b)`, `i..l` the components of the four shells, `(c, d)` and `(a, b)` the
/// ket's and the bra's contracted-pair dimensions (`ShellPair::get_contracted_pair_dims`, positions by
/// `ShellPair::pair_slot`). The pairs are those of `basis::shell_pairs::derive_shell_pairs`, in either shell order and
/// on any centres. A bra pair of a shell with itself holds its contracted pairs `a <= b`, each block over every
/// component pair `(i, j)` exact when no two of its primitive products are merged (a contraction matrix with one
/// nonzero per row and column, as for primitive functions); the pairs `a > b` are the negative transposes.
///
/// Per pair, built once by the caller: `bra_data` (`SpinOrbitBra::new(bra, ..)`) and `ket_coefficients`
/// (`compute_ketside_hermite_derivative_coefficients(ket, 0, ..)`: the ket's `E` by the general route, in its
/// harmonic basis). Scratch from `arena`.
///
/// Per primitive quartet the R matrix at bra order `l_ab + 1`, `R E^cd` and `B_k (R E^cd)`, then the ket and the bra
/// contraction.
// The buffers of one quartet `(bra | ket)`: the R matrix at bra order `l_ab + 1` and its recursion's, `R E^cd`, and
// per component the quartet over the ket primitive pairs `(block, n_prim_cd)`, then, ket contracted, over the bra
// primitive pairs `(block n_slots_cd, n_prim_ab)`.
struct QuartetBuffers<'a> {
    r_buffer: HermiteCoulombIntegrals4D<'a>,
    r: BufferedRMatrix<'a, f64>,
    r_e: &'a mut [f64],
    per_ket: &'a mut [f64],
    per_bra: &'a mut [f64],
}

impl<'a> QuartetBuffers<'a> {
    fn new(bra: &ShellPair<Contracted>, ket: &ShellPair<Contracted>, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let lqn_ab = bra.get_lqn_a() + bra.get_lqn_b();
        let lqn_cd = ket.get_lqn_a() + ket.get_lqn_b();
        let sph_cd = ket.get_degeneracy_a() * ket.get_degeneracy_b();
        let block = bra.get_degeneracy_a() * bra.get_degeneracy_b() * sph_cd;
        let ket_dims = ket.get_contracted_pair_dims();
        let (n_prim_ab, n_prim_cd) = (bra.get_products().len(), ket.get_products().len());
        Ok(QuartetBuffers {
            r_buffer: HermiteCoulombIntegrals4D::from_arena(lqn_ab + 1 + lqn_cd, arena)?,
            r: BufferedRMatrix::<f64>::from_arena(lqn_ab + 1, lqn_cd, arena)?,
            r_e: arena.alloc::<f64>(hermite_count(lqn_ab + 1) * sph_cd)?,
            per_ket: arena.alloc::<f64>(3 * block * n_prim_cd)?,
            per_bra: arena.alloc::<f64>(3 * block * ket_dims.0 * ket_dims.1 * n_prim_ab)?,
        })
    }
}

/// Takes from `arena` the buffers `compute_spin_orbit_quartet` takes for the quartet `(bra | ket)`, computing nothing:
/// for measuring its scratch.
pub fn reserve_spin_orbit_quartet(bra: &ShellPair<Contracted>, ket: &ShellPair<Contracted>, arena: &Arena<'_>) -> Result<(), ArenaError> {
    QuartetBuffers::new(bra, ket, arena).map(|_| ())
}

pub fn compute_spin_orbit_quartet(
    bra: &ShellPair<Contracted>,
    bra_data: &SpinOrbitBra,
    ket: &ShellPair<Contracted>,
    ket_coefficients: &Array5D<f64>,
    output: &mut [Array8DViewMut<'_, f64>; 3],
    arena: &Arena<'_>,
) -> Result<(), ArenaError> {
    let lqn_ab = bra.get_lqn_a() + bra.get_lqn_b();
    let lqn_cd = ket.get_lqn_a() + ket.get_lqn_b();
    let sph_ab = bra.get_degeneracy_a() * bra.get_degeneracy_b();
    let sph_cd = ket.get_degeneracy_a() * ket.get_degeneracy_b();
    let (n_prim_ab, n_prim_cd) = (bra.get_products().len(), ket.get_products().len());
    let (bra_dims, ket_dims) = (bra.get_contracted_pair_dims(), ket.get_contracted_pair_dims());
    let (n_slots_ab, n_slots_cd) = (bra_dims.0 * bra_dims.1, ket_dims.0 * ket_dims.1);
    let n_hermite_cd = hermite_count(lqn_cd);
    let n_raised = hermite_count(lqn_ab + 1); // the Hermite space of the R matrix rows
    let block = sph_ab * sph_cd;
    debug_assert_eq!((bra_data.rows, bra_data.cols, bra_data.matrices.len()), (sph_ab, n_raised, 3 * sph_ab * n_raised * n_prim_ab));
    // (tuv, c, d, 1, prim)
    let e_ket = ket_coefficients.get_data();
    debug_assert_eq!(e_ket.len(), n_hermite_cd * sph_cd * n_prim_cd);

    let QuartetBuffers { mut r_buffer, mut r, r_e, per_ket, per_bra } = QuartetBuffers::new(bra, ket, arena)?;
    let (bra_contraction, ket_contraction) = (bra.contraction(), ket.contraction());

    for (ip, ab) in bra.get_products().iter().enumerate() {
        for (jp, cd) in ket.get_products().iter().enumerate() {
            construct_r_xxxx_prealloc::<FullRange, _>(
                &mut r_buffer, &mut r,
                ab.get_p(), &ab.get_product_centre(), lqn_ab + 1,
                cd.get_p(), &cd.get_product_centre(), lqn_cd,
                0.0,
            );
            let e_cd = MatrixView::<f64>::new(n_hermite_cd, sph_cd, &e_ket[jp * n_hermite_cd * sph_cd..][..n_hermite_cd * sph_cd]);
            r.view_as_matrix().matmul_prealloc(&e_cd, &mut MatrixViewMut::new(n_raised, sph_cd, &mut r_e[..]));
            let r_e = MatrixView::<f64>::new(n_raised, sph_cd, &r_e[..]);
            for k in 0..3 {
                let b_k = bra_data.matrix(ip, k);
                let out = &mut per_ket[(k * n_prim_cd + jp) * block..][..block];
                b_k.matmul_prealloc(&r_e, &mut MatrixViewMut::new(sph_ab, sph_cd, out));
            }
        }
        for k in 0..3 {
            let rows = block * n_slots_cd;
            let per_product = MatrixView::<f64>::new(block, n_prim_cd, &per_ket[k * n_prim_cd * block..][..n_prim_cd * block]);
            let out = &mut per_bra[(k * n_prim_ab + ip) * rows..][..rows];
            ket_contraction.contract(&per_product, &mut MatrixViewMut::new(block, n_slots_cd, out), 1.0);
        }
    }

    let rows = block * n_slots_cd;
    for (k, batch) in output.iter_mut().enumerate() {
        let per_product = MatrixView::<f64>::new(rows, n_prim_ab, &per_bra[k * n_prim_ab * rows..][..n_prim_ab * rows]);
        let data = batch.get_data_mut();
        debug_assert_eq!(data.len(), rows * n_slots_ab);
        bra_contraction.contract(&per_product, &mut MatrixViewMut::new(rows, n_slots_ab, data), 1.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrays::array8d::Array8DTrait;
    use arrays::scratch::{Scratch, ByteUnit};
    use basis::shell_types::{decontract_general_shells, BasisShell};
    use mmd::ketside_hermite_coefficients::compute_ketside_hermite_coefficients;

    use crate::batch::{compute_eri_batch, ERIBatchType};

    // The primitive d block of O x2c-SVPall (centre 0, one primitive, exponent 1.2) and the s and p blocks of H
    // x2c-SVPall (centre 1), each as a general contraction with the identity: mild exponents, so that finite
    // differences in the bra centres are accurate.
    fn shells() -> Vec<BasisShell<Contracted>> {
        // One file per call: the tests run in parallel, and each removes its own.
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("aedelsten_spin_orbit_{}_{call}.inp", std::process::id()));
        std::fs::write(&path, "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 0.0 1.8\nEnd of Basis\n").unwrap();
        let general = basis::setup_basis_system::<Contracted>(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        let (blocks, _) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
        blocks.iter().map(BasisShell::<Contracted>::from_shell).collect()
    }

    // The block of angular momentum `lqn` on centre `centre`, moved to `position`.
    fn shell(shells: &[BasisShell<Contracted>], centre: usize, lqn: usize, position: [f64; 3]) -> BasisShell<Contracted> {
        let mut shell = shells.iter().find(|s| s.get_centre_index() == centre && s.get_lqn() == lqn).unwrap().clone();
        shell.set_position(position);
        shell
    }

    fn pair(a: &BasisShell<Contracted>, b: &BasisShell<Contracted>) -> ShellPair<Contracted> {
        a.form_shell_pair(b, false, None).unwrap()
    }

    fn self_pair(a: &BasisShell<Contracted>) -> ShellPair<Contracted> {
        a.form_shell_pair(a, true, None).unwrap()
    }

    // The functions of each side of a pair (contracted functions times components).
    fn widths(pair: &ShellPair<Contracted>) -> [usize; 2] {
        [pair.get_n_cont_a() * pair.get_degeneracy_a(), pair.get_n_cont_b() * pair.get_degeneracy_b()]
    }

    // A quartet batch (the layout of `compute_spin_orbit_quartet`) as a dense `(a, b, c, d)` tensor over the pairs'
    // functions, `a` fastest; the pairs `a > b` of a self pair from `bra_sign` times the transpose, likewise
    // `ket_sign`.
    fn dense(data: &[f64], bra: &ShellPair<Contracted>, ket: &ShellPair<Contracted>, bra_sign: f64, ket_sign: f64) -> Vec<f64> {
        let [na, nb] = widths(bra);
        let [nc, nd] = widths(ket);
        let (da, db, dc, dd) = (bra.get_degeneracy_a(), bra.get_degeneracy_b(), ket.get_degeneracy_a(), ket.get_degeneracy_b());
        let block = da * db * dc * dd;
        let ket_slots = ket.get_contracted_pair_dims().0 * ket.get_contracted_pair_dims().1;
        let mut out = vec![0.0; na * nb * nc * nd];
        for b in 0..bra.get_n_cont_b() {
            for a in 0..bra.get_n_cont_a() {
                let (bra_slot, bra_factor, bra_swap) = match bra.pair_slot(a.min(b), a.max(b)) {
                    Some(slot) if bra.get_single_centre() => (slot, if a > b { bra_sign } else { 1.0 }, a > b),
                    _ => (bra.pair_slot(a, b).unwrap(), 1.0, false),
                };
                for d in 0..ket.get_n_cont_b() {
                    for c in 0..ket.get_n_cont_a() {
                        let (ket_slot, ket_factor, ket_swap) = match ket.pair_slot(c.min(d), c.max(d)) {
                            Some(slot) if ket.get_single_centre() => (slot, if c > d { ket_sign } else { 1.0 }, c > d),
                            _ => (ket.pair_slot(c, d).unwrap(), 1.0, false),
                        };
                        for l in 0..dd {
                            for k in 0..dc {
                                for j in 0..db {
                                    for i in 0..da {
                                        let (ii, jj) = if bra_swap { (j, i) } else { (i, j) };
                                        let (kk, ll) = if ket_swap { (l, k) } else { (k, l) };
                                        let value = data[ii + da * (jj + db * (kk + dc * ll)) + block * (ket_slot + ket_slots * bra_slot)];
                                        let (fa, fb, fc, fd) = (a * da + i, b * db + j, c * dc + k, d * dd + l);
                                        out[fa + na * (fb + nb * (fc + nc * fd))] = bra_factor * ket_factor * value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    // X_x, X_y, X_z of the quartet as dense tensors.
    fn spin_orbit(bra: &ShellPair<Contracted>, ket: &ShellPair<Contracted>) -> [Vec<f64>; 3] {
        let mut scratch = Scratch::new(64, ByteUnit::MiB).unwrap();
        let mut arena = scratch.get_arena();
        let (bra_dims, ket_dims) = (bra.get_contracted_pair_dims(), ket.get_contracted_pair_dims());
        let batch = || {
            Array8DViewMut::<f64>::from_arena(
                bra.get_degeneracy_a(), bra.get_degeneracy_b(), ket.get_degeneracy_a(), ket.get_degeneracy_b(),
                ket_dims.0, ket_dims.1, bra_dims.0, bra_dims.1, &arena,
            ).unwrap()
        };
        let mut batches = [batch(), batch(), batch()];
        let bra_data = SpinOrbitBra::new(bra, &mut arena).unwrap();
        let ket_coefficients = compute_ketside_hermite_derivative_coefficients(ket, 0, &mut arena).unwrap();
        compute_spin_orbit_quartet(bra, &bra_data, ket, &ket_coefficients, &mut batches, &arena).unwrap();
        batches.map(|b| dense(b.get_data(), bra, ket, -1.0, 1.0))
    }

    // The ordinary ERIs of the quartet as a dense tensor (pairs of distinct shells).
    fn eris(bra: &ShellPair<Contracted>, ket: &ShellPair<Contracted>) -> Vec<f64> {
        let mut scratch = Scratch::new(64, ByteUnit::MiB).unwrap();
        let mut arena = scratch.get_arena();
        let e_bra = compute_ketside_hermite_coefficients(bra, &mut arena).unwrap();
        let e_ket = compute_ketside_hermite_coefficients(ket, &mut arena).unwrap();
        let arena = &mut arena;
        let e_bra = e_bra.mmd_transpose_in(arena).unwrap();
        let batch = compute_eri_batch::<Contracted, Contracted, FullRange>(bra, ket, &e_bra, &e_ket, arena, 0.0).unwrap();
        let data = match &batch {
            ERIBatchType::FourD(b) => arrays::array4d::Array4DTrait::get_data(b),
            ERIBatchType::SixD(b) => arrays::array6d::Array6DTrait::get_data(b),
            ERIBatchType::EightD(b) => arrays::array8d::Array8DTrait::get_data(b),
        };
        dense(data, bra, ket, 1.0, 1.0)
    }

    fn displaced(position: [f64; 3], axis: usize, step: f64) -> [f64; 3] {
        let mut out = position;
        out[axis] += step;
        out
    }

    // X_k = (d/dA_i d/dB_j - d/dA_j d/dB_i)(ab|cd), (i, j, k) cyclic, by central differences of the ordinary ERIs
    // in the positions of `a` and `b`, Richardson-extrapolated from steps h and 2h.
    fn finite_difference(a: &BasisShell<Contracted>, b: &BasisShell<Contracted>, ket: &ShellPair<Contracted>) -> [Vec<f64>; 3] {
        let (pos_a, pos_b) = (a.get_position(), b.get_position());
        let at = |shell: &BasisShell<Contracted>, position: [f64; 3]| {
            let mut moved = shell.clone();
            moved.set_position(position);
            moved
        };
        let mixed = |i: usize, j: usize, h: f64| -> Vec<f64> {
            let value = |si: f64, sj: f64| eris(&pair(&at(a, displaced(pos_a, i, si * h)), &at(b, displaced(pos_b, j, sj * h))), ket);
            let (pp, pm, mp, mm) = (value(1.0, 1.0), value(1.0, -1.0), value(-1.0, 1.0), value(-1.0, -1.0));
            (0..pp.len()).map(|n| (pp[n] - pm[n] - mp[n] + mm[n]) / (4.0 * h * h)).collect()
        };
        let component = |k: usize, h: f64| -> Vec<f64> {
            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
            mixed(i, j, h).iter().zip(mixed(j, i, h)).map(|(x, y)| x - y).collect()
        };
        let h = 2e-3;
        [0, 1, 2].map(|k| component(k, h).iter().zip(component(k, 2.0 * h)).map(|(x1, x2)| (4.0 * x1 - x2) / 3.0).collect())
    }

    fn assert_close(label: &str, x: &[f64], y: &[f64], tolerance: f64) {
        assert_eq!(x.len(), y.len(), "{label}");
        let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let difference = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
        assert!(scale > 1e-3, "{label}: vacuous comparison");
        assert!(difference < tolerance * scale.max(1.0), "{label}: {difference}");
    }

    #[test]
    fn four_centre_quartet_matches_finite_differences_of_eris() {
        // A p-d bra against an s-d ket.
        let shells = shells();
        let a = shell(&shells, 1, 1, [0.1, -0.2, 0.3]);
        let b = shell(&shells, 0, 2, [-0.4, 0.5, 1.1]);
        let c = shell(&shells, 1, 0, [0.7, 0.2, -0.6]);
        let d = shell(&shells, 0, 2, [-0.3, -0.8, 0.4]);
        let (bra, ket) = (pair(&a, &b), pair(&c, &d));
        let analytic = spin_orbit(&bra, &ket);
        let numeric = finite_difference(&a, &b, &ket);
        for k in 0..3 {
            assert_close(&format!("X_{k} (pd|sd)"), &analytic[k], &numeric[k], 1e-9);
        }
    }

    #[test]
    fn one_centre_quartet_matches_finite_differences_of_eris() {
        // The AMFI case: all four shells on one point, the bra of two p blocks, the ket of two d blocks.
        let shells = shells();
        let centre = [0.2, -0.1, 0.4];
        let a = shell(&shells, 1, 1, centre);
        let c = shell(&shells, 0, 2, centre);
        let (bra, ket) = (pair(&a, &a), pair(&c, &c));
        let analytic = spin_orbit(&bra, &ket);
        let numeric = finite_difference(&a, &a, &ket);
        for k in 0..3 {
            assert_close(&format!("X_{k} (pp|dd) one centre"), &analytic[k], &numeric[k], 1e-9);
        }
    }

    #[test]
    fn one_centre_d_bra_matches_finite_differences_of_eris() {
        let shells = shells();
        let centre = [0.2, -0.1, 0.4];
        let a = shell(&shells, 0, 2, centre);
        let c = shell(&shells, 1, 1, centre);
        let (bra, ket) = (pair(&a, &a), pair(&c, &c));
        let analytic = spin_orbit(&bra, &ket);
        let numeric = finite_difference(&a, &a, &ket);
        for k in 0..3 {
            assert_close(&format!("X_{k} (dd|pp) one centre"), &analytic[k], &numeric[k], 1e-9);
        }
    }

    #[test]
    fn antisymmetric_in_the_bra_and_symmetric_in_the_ket() {
        // The bra swapped across angular momenta (d before p), the ket within one (the ket wants l_c <= l_d).
        let shells = shells();
        let a = shell(&shells, 1, 1, [0.1, -0.2, 0.3]);
        let b = shell(&shells, 0, 2, [-0.4, 0.5, 1.1]);
        let c = shell(&shells, 1, 1, [0.7, 0.2, -0.6]);
        let d = shell(&shells, 1, 1, [-0.3, -0.8, 0.4]);
        let reference = spin_orbit(&pair(&a, &b), &pair(&c, &d));
        let [na, nb] = widths(&pair(&a, &b));
        let [nc, nd] = widths(&pair(&c, &d));
        let swapped_bra = spin_orbit(&pair(&b, &a), &pair(&c, &d));
        let swapped_ket = spin_orbit(&pair(&a, &b), &pair(&d, &c));
        for k in 0..3 {
            let mut minus_bra = vec![0.0; reference[k].len()];
            let mut same_ket = vec![0.0; reference[k].len()];
            for fd in 0..nd {
                for fc in 0..nc {
                    for fb in 0..nb {
                        for fa in 0..na {
                            let ket = fc + nc * fd;
                            minus_bra[fa + na * (fb + nb * ket)] = -swapped_bra[k][fb + nb * (fa + na * ket)];
                            same_ket[fa + na * (fb + nb * (fc + nc * fd))] = swapped_ket[k][fa + na * (fb + nb * (fd + nd * fc))];
                        }
                    }
                }
            }
            assert_close(&format!("X_{k} (ba|cd) = -(ab|cd)"), &minus_bra, &reference[k], 1e-13);
            assert_close(&format!("X_{k} (ab|dc) = (ab|cd)"), &same_ket, &reference[k], 1e-13);
        }
    }

    #[test]
    fn self_pairs_match_the_pairs_formed_in_full() {
        // A primitive block's pair with itself stores a <= b; the rest are the negative (bra) or plain (ket)
        // transposes. Formed in full (as two shells), every product is computed.
        let shells = shells();
        let centre = [0.2, -0.1, 0.4];
        let p = shell(&shells, 1, 1, centre);
        let d = shell(&shells, 0, 2, centre);
        let full = spin_orbit(&pair(&p, &p), &pair(&d, &d));
        let self_bra = spin_orbit(&self_pair(&p), &pair(&d, &d));
        let self_both = spin_orbit(&self_pair(&p), &self_pair(&d));
        for k in 0..3 {
            assert_close(&format!("X_{k} self bra"), &self_bra[k], &full[k], 1e-13);
            assert_close(&format!("X_{k} self bra and ket"), &self_both[k], &full[k], 1e-13);
        }
    }
}
