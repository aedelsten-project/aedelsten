// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The batches up to `kernels::L_UNROLL` through the generated kernels: the driver the generated `kernels::dispatch`
//! calls with the kernels of a quartet's `(l_ab, l_cd)` inlined.

use arrays::matrix::*;
use arrays::vector_math::{vector_norm_square_3d, vector_subtract_3d};
use arrays::scratch::{Arena, ArenaError};
use basis::contraction::ShellKind;
use basis::gto_products::GTOProductTrait;
use basis::shell_pairs::ShellPair;
use mmd::hermite_coulomb_integrals::{HermiteCoulombIntegrals4D, RangeVersion};
use mmd::hermite_expansion::hermite_count;

use super::r_columns::compute_eri_coeff;
use super::{accumulates, product_slot};

/// The batch's outputs (`batch_generated`): they outlive it. The primitive batches `(per_ab, slots_ab)` and, with a
/// bra contraction, the contracted ones. Taken only: the batch zeroes what it accumulates into.
pub(crate) struct GeneratedOutputs<'a> {
    /// The primitive batches.
    pub(crate) e_r_e: &'a mut [f64],
    /// The bra-contracted batches.
    pub(crate) after_ab: &'a mut [f64],
}

impl<'a> GeneratedOutputs<'a> {
    /// The outputs of the batch `(pair_ab|pair_cd)`, from `arena`.
    pub(crate) fn new_in<Sab: ShellKind, Scd: ShellKind>(pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let n_ab = pair_ab.get_degeneracy_a() * pair_ab.get_degeneracy_b();
        let n_cd = pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b();
        let (accumulate_ab, accumulate_cd) = (accumulates(&pair_ab.contraction()), accumulates(&pair_cd.contraction()));
        let slots_ab = if accumulate_ab { 1 } else { pair_ab.get_products().len() };
        let slots_cd = if accumulate_cd { 1 } else { pair_cd.get_products().len() };
        let columns_cd = pair_cd.contraction_matrix().map_or(slots_cd, |cc| cc.cols());
        let per_ab = n_ab * n_cd * columns_cd;
        Ok(GeneratedOutputs {
            e_r_e: arena.alloc::<f64>(per_ab * slots_ab)?,
            after_ab: arena.alloc::<f64>(pair_ab.contraction_matrix().map_or(0, |cc| per_ab * cc.cols()))?,
        })
    }
}

/// The batch's scratch (`batch_generated`), in a scope after its outputs and free again when it returns: the R
/// buffer, `X` per cd slot and `X` cd-contracted.
pub(crate) struct GeneratedScratch<'s> {
    r_buffer: HermiteCoulombIntegrals4D<'s>,
    x: &'s mut [f64],
    x_cont: &'s mut [f64],
}

impl<'s> GeneratedScratch<'s> {
    /// The scratch of the batch `(pair_ab|pair_cd)`, from `scratch`.
    pub(crate) fn new_in<Sab: ShellKind, Scd: ShellKind>(pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>, scratch: &Arena<'s>) -> Result<Self, ArenaError> {
        let (lqn_ab, lqn_cd) = (pair_ab.get_lqn_a() + pair_ab.get_lqn_b(), pair_cd.get_lqn_a() + pair_cd.get_lqn_b());
        let n_cd = pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b();
        let accumulate_cd = accumulates(&pair_cd.contraction());
        let slots_cd = if accumulate_cd { 1 } else { pair_cd.get_products().len() };
        let cc_cd = pair_cd.contraction_matrix();
        let columns_cd = cc_cd.map_or(slots_cd, |cc| cc.cols());
        let x_len = hermite_count(lqn_ab) * n_cd;
        Ok(GeneratedScratch {
            r_buffer: HermiteCoulombIntegrals4D::from_arena(lqn_ab + lqn_cd, scratch)?,
            x: scratch.alloc::<f64>(x_len * slots_cd)?,
            x_cont: scratch.alloc::<f64>(if cc_cd.is_some() { x_len * columns_cd } else { 0 })?,
        })
    }
}

/// The batch `(pair_ab|pair_cd)` through the generated kernels of its `(l_ab, l_cd)` (`kernels.rs`): per bra
/// primitive, `X = R E_cd` summed over the ket's primitives into one `(tuv_ab, n_cd)` slab per cd slot (`re`; `re_set`
/// where a slot holds one primitive, which needs no zeroing), the cd contraction applied to `X` when it is dense, then
/// `E_ab X` added into the bra slot (`ebra`); the ab contraction last. Returns `(n_ab, n_cd, n_cont_cd, n_cont_ab)`
/// (the slots for `Summed` and `Identity` sides). `bra` holds the bra pair's coefficients `(n_ab, tuv_ab)` per
/// primitive product, `ket` the ket pair's `(tuv_cd, n_cd)`; `n_ab` and `n_cd` the pairs' components, as their
/// coefficients carry them.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn batch_generated<'a, R, Sab, Scd, K, KS, B>(
    pair_ab: &ShellPair<Sab>,
    pair_cd: &ShellPair<Scd>,
    bra: &[f64],
    ket: &[f64],
    arena: &mut Arena<'a>,
    omega: f64,
    re: K,
    re_set: KS,
    ebra: B,
) -> Result<&'a mut [f64], ArenaError>
where
    R: RangeVersion,
    Sab: ShellKind,
    Scd: ShellKind,
    K: Fn(&[f64], f64, f64, f64, f64, &[f64], &mut [f64]),
    KS: Fn(&[f64], f64, f64, f64, f64, &[f64], &mut [f64]),
    B: Fn(&[f64], usize, &[f64], &mut [f64]),
{
    let (lqn_ab, lqn_cd) = (pair_ab.get_lqn_a() + pair_ab.get_lqn_b(), pair_cd.get_lqn_a() + pair_cd.get_lqn_b());
    let (tuv_ab, tuv_cd) = (hermite_count(lqn_ab), hermite_count(lqn_cd));
    let n_ab = pair_ab.get_degeneracy_a() * pair_ab.get_degeneracy_b();
    let n_cd = pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b();
    let (accumulate_ab, accumulate_cd) = (accumulates(&pair_ab.contraction()), accumulates(&pair_cd.contraction()));
    let slots_ab = if accumulate_ab { 1 } else { pair_ab.get_products().len() };
    let slots_cd = if accumulate_cd { 1 } else { pair_cd.get_products().len() };
    let (cc_ab, cc_cd) = (pair_ab.contraction_matrix(), pair_cd.contraction_matrix());
    let columns_cd = cc_cd.map_or(slots_cd, |cc| cc.cols());

    // The outputs come from the batch's arena (they outlive it), before the primitive loops.
    let per_ab = n_ab * n_cd * columns_cd;
    let GeneratedOutputs { e_r_e, after_ab } = GeneratedOutputs::new_in(pair_ab, pair_cd, arena)?;
    e_r_e.fill(0.0);
    let x_len = tuv_ab * n_cd;
    // An (ss|ss) quartet needs R^0_000 alone: the scalar Boys value.
    let scalar = lqn_ab + lqn_cd == 0;
    arena.scope(|scratch| -> Result<(), ArenaError> {
        let GeneratedScratch { mut r_buffer, x, x_cont } = GeneratedScratch::new_in(pair_ab, pair_cd, scratch)?;
        for (i, (e_ab, ab)) in bra.chunks_exact(n_ab * tuv_ab).zip(pair_ab.get_products()).enumerate() {
            let p = ab.get_p();
            let p_x = ab.get_product_centre();
            // A summed ket adds every primitive into one slot; otherwise each slot is written once (`re_set`), and
            // needs no zeroing.
            if accumulate_cd {
                x.fill(0.0);
            }
            for (j, (e_cd, cd)) in ket.chunks_exact(tuv_cd * n_cd).zip(pair_cd.get_products()).enumerate() {
                let q = cd.get_p();
                let pq = vector_subtract_3d(&p_x, &cd.get_product_centre());
                let (eta, rpc_square) = (p * q / (p + q), vector_norm_square_3d(&pq));
                let pref = compute_eri_coeff(p, q);
                let slot = product_slot(accumulate_cd, j);
                let x = &mut x[slot * x_len..(slot + 1) * x_len];
                let kernel = |boys: &[f64], x: &mut [f64]| {
                    if accumulate_cd {
                        re(boys, pq[0], pq[1], pq[2], pref, e_cd, x)
                    } else {
                        re_set(boys, pq[0], pq[1], pq[2], pref, e_cd, x)
                    }
                };
                if scalar {
                    kernel(&[R::compute_r_ssss(eta, rpc_square, omega)], x);
                } else {
                    R::compute_boys(&mut r_buffer, eta, rpc_square, omega);
                    kernel(r_buffer.get_boys(), x);
                }
            }
            let x_columns: &[f64] = match cc_cd {
                Some(cc_cd) => {
                    MatrixView::new(x_len, slots_cd, &*x).matmul_prealloc(cc_cd, &mut MatrixViewMut::new(x_len, columns_cd, &mut *x_cont));
                    x_cont
                }
                None => x,
            };
            let slot = product_slot(accumulate_ab, i);
            ebra(e_ab, n_ab, x_columns, &mut e_r_e[slot * per_ab..(slot + 1) * per_ab]);
        }
        Ok(())
    })?;
    Ok(match cc_ab {
        Some(cc_ab) => {
            MatrixView::new(per_ab, slots_ab, &*e_r_e).matmul_prealloc(cc_ab, &mut MatrixViewMut::new(per_ab, cc_ab.cols(), &mut *after_ab));
            after_ab
        }
        None => e_r_e,
    })
}

/// `out += E_ab X`: `E_ab` (`n_ab`, `T`), `X` (`T`, columns) and `out` (`n_ab`, columns), column-major, `T` the bra's
/// Hermite functions. Register tiles of 8 bra components by 4 columns (then 4, 2 and 1 of each), the Hermite functions
/// innermost: every `E_ab` value loaded serves 4 columns, every `X` value 8 components.
#[inline(always)]
pub(crate) fn ebra_tiled<const T: usize>(e_ab: &[f64], n_ab: usize, x: &[f64], out: &mut [f64]) {
    let e_ab = &e_ab[..n_ab * T];
    let columns = x.len() / T;
    let mut c = 0;
    while c + 4 <= columns {
        ebra_columns::<T, 4>(e_ab, n_ab, &x[c * T..(c + 4) * T], &mut out[c * n_ab..(c + 4) * n_ab]);
        c += 4;
    }
    while c < columns {
        ebra_columns::<T, 1>(e_ab, n_ab, &x[c * T..(c + 1) * T], &mut out[c * n_ab..(c + 1) * n_ab]);
        c += 1;
    }
}

/// `ebra_tiled` for `C` columns, by row tiles of 8, 4, 2 and 1.
#[inline(always)]
fn ebra_columns<const T: usize, const C: usize>(e_ab: &[f64], n_ab: usize, x: &[f64], out: &mut [f64]) {
    let mut k = 0;
    while k + 8 <= n_ab {
        ebra_tile::<T, C, 8>(e_ab, n_ab, x, out, k);
        k += 8;
    }
    if k + 4 <= n_ab {
        ebra_tile::<T, C, 4>(e_ab, n_ab, x, out, k);
        k += 4;
    }
    if k + 2 <= n_ab {
        ebra_tile::<T, C, 2>(e_ab, n_ab, x, out, k);
        k += 2;
    }
    if k < n_ab {
        ebra_tile::<T, C, 1>(e_ab, n_ab, x, out, k);
    }
}

/// One `M x C` tile of `out` from row `k`, accumulated in registers over the `T` Hermite functions.
#[inline(always)]
fn ebra_tile<const T: usize, const C: usize, const M: usize>(e_ab: &[f64], n_ab: usize, x: &[f64], out: &mut [f64], k: usize) {
    let mut acc = [[0.0; M]; C];
    for j in 0..T {
        let b: &[f64; M] = e_ab[j * n_ab + k..][..M].try_into().unwrap();
        for (acc, x) in acc.iter_mut().zip(x.chunks_exact(T)) {
            let xj = x[j];
            for (a, &b) in acc.iter_mut().zip(b) {
                *a += b * xj;
            }
        }
    }
    for (acc, out) in acc.iter().zip(out.chunks_exact_mut(n_ab)) {
        for (o, a) in out[k..k + M].iter_mut().zip(acc) {
            *o += a;
        }
    }
}
