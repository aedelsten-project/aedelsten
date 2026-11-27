// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! One quartet's two-electron integrals (`compute_eri_batch`): the generated kernels up to `kernels::L_UNROLL`
//! (`generated.rs`, the kernels in `kernels.rs`), the BLAS tiles above (`tiles.rs`), both from the R matrices of
//! `r_columns.rs`.

mod generated;
mod kernels;
mod r_columns;
mod tiles;
#[cfg(test)]
mod tests;

pub(crate) use kernels::re_kernels;
pub(crate) use r_columns::{compute_eri_coeff, construct_r_xxxx_prealloc};

use arrays::array4d::*;
use arrays::array6d::*;
use arrays::array8d::*;
use arrays::scratch::*;

use basis::contraction::{Contraction, ShellKind};
use basis::shell_pairs::*;
use mmd::ketside_hermite_coefficients::{MMDCoefficients, MMDCoefficientsView};
use mmd::hermite_coulomb_integrals::RangeVersion;

/// A quartet's batch, by the components of its pairs.
pub(crate) enum ERIBatchType<'a> {
    /// `(ss|ss)`: `(c, d, a, b)`.
    FourD(Array4DViewMut<'a, f64>),
    /// `(xx|ss)`: `(i, j, c, d, a, b)`.
    SixD(Array6DViewMut<'a, f64>),
    /// `(xx|xx)`: `(i, j, k, l, c, d, a, b)`.
    EightD(Array8DViewMut<'a, f64>),
}

/// The batch of the quartet `(pair_ab|pair_cd)`: by the generated kernels of its `(l_ab, l_cd)` up to
/// `kernels::L_UNROLL`, by the BLAS tiles of `batch_xxxx` above. The two pairs may be of different contraction
/// kinds and in either order; the walks put the pair with more components in the bra (`EriData`), the order the
/// generated kernels are fastest in. The batch is `(n_ab, n_cd, n_cont_cd, n_cont_ab)`, shaped by the pairs'
/// components: an s-s pair on both sides (`FourD`), on the ket side (`SixD`), or neither. Its memory is
/// `eri_batch_buffers`'.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_eri_batch<'a, Sab, Scd, R>(
    pair_ab: &ShellPair<Sab>,
    pair_cd: &ShellPair<Scd>,
    bra_coefficients: &MMDCoefficientsView<'_>,
    ket_coefficients: &MMDCoefficients,
    arena: &mut Arena<'a>,
    omega: f64,
) -> Result<ERIBatchType<'a>, ArenaError>
where
    Sab: ShellKind,
    Scd: ShellKind,
    R: RangeVersion,
{
    let (lqn_ab, lqn_cd) = (pair_ab.get_lqn_a() + pair_ab.get_lqn_b(), pair_cd.get_lqn_a() + pair_cd.get_lqn_b());
    let data = if lqn_ab + lqn_cd <= kernels::L_UNROLL {
        let _span = instrumentation::fine_span("eri::batch_generated");
        kernels::dispatch::<R, _, _>(lqn_ab, lqn_cd, pair_ab, pair_cd, bra_coefficients.get_data(), ket_coefficients.get_data(), arena, omega)
            .expect("a generated kernel up to L_UNROLL")?
    } else {
        let _span = instrumentation::fine_span("eri::batch_xxxx");
        let batch = tiles::batch_xxxx::<R, _, _>(pair_ab, pair_cd, bra_coefficients.as_tensor(), ket_coefficients.as_tensor(), arena, omega)?;
        batch.into_data()
    };
    Ok(batch_of(pair_ab, pair_cd, data))
}

/// `data` shaped as the batch of `(pair_ab|pair_cd)`.
fn batch_of<'a, Sab: ShellKind, Scd: ShellKind>(pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>, data: &'a mut [f64]) -> ERIBatchType<'a> {
    let (deg_a, deg_b) = (pair_ab.get_degeneracy_a(), pair_ab.get_degeneracy_b());
    let (deg_c, deg_d) = (pair_cd.get_degeneracy_a(), pair_cd.get_degeneracy_b());
    let (n_cont_a, n_cont_b) = pair_ab.get_contracted_pair_dims();
    let (n_cont_c, n_cont_d) = pair_cd.get_contracted_pair_dims();
    match (deg_a * deg_b, deg_c * deg_d) {
        (1, 1) => ERIBatchType::FourD(Array4DViewMut::new(n_cont_c, n_cont_d, n_cont_a, n_cont_b, data)),
        (_, 1) => ERIBatchType::SixD(Array6DViewMut::new(deg_a, deg_b, n_cont_c, n_cont_d, n_cont_a, n_cont_b, data)),
        _ => ERIBatchType::EightD(Array8DViewMut::new(deg_a, deg_b, deg_c, deg_d, n_cont_c, n_cont_d, n_cont_a, n_cont_b, data)),
    }
}

/// The memory `compute_eri_batch` takes for `(pair_ab|pair_cd)`, taken and nothing computed (the scratch query): its
/// path's outputs, then its scratch in a scope, as the path takes them. The batch over the outputs, not filled (its
/// values are whatever the memory held: only the walks' digestion reads them, and no allocation of theirs depends on
/// them).
pub(crate) fn eri_batch_buffers<'a, Sab, Scd>(pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>, arena: &mut Arena<'a>) -> Result<ERIBatchType<'a>, ArenaError>
where
    Sab: ShellKind,
    Scd: ShellKind,
{
    let (lqn_ab, lqn_cd) = (pair_ab.get_lqn_a() + pair_ab.get_lqn_b(), pair_cd.get_lqn_a() + pair_cd.get_lqn_b());
    let data = if lqn_ab + lqn_cd <= kernels::L_UNROLL {
        let generated::GeneratedOutputs { e_r_e, after_ab } = generated::GeneratedOutputs::new_in(pair_ab, pair_cd, arena)?;
        arena.scope(|scratch| generated::GeneratedScratch::new_in(pair_ab, pair_cd, scratch).map(|_| ()))?;
        if pair_ab.contraction_matrix().is_some() { after_ab } else { e_r_e }
    } else {
        let tiles::XxxxOutputs { e_r_e_prim, e_r_e_cont } = tiles::XxxxOutputs::new_in(pair_ab, pair_cd, arena)?;
        arena.scope(|scratch| tiles::XxxxScratch::new_in(pair_ab, pair_cd, scratch).map(|_| ()))?;
        if pair_ab.contraction_matrix().is_some() { e_r_e_cont.into_data() } else { e_r_e_prim.into_data() }
    };
    Ok(batch_of(pair_ab, pair_cd, data))
}

/// The slot of product `product`: its own, or the one all products accumulate into.
#[inline(always)]
fn product_slot(accumulate: bool, product: usize) -> usize {
    if accumulate { 0 } else { product }
}

/// Whether every primitive product of a pair of this contraction accumulates into one slot.
#[inline(always)]
fn accumulates<C: Contraction>(_contraction: &C) -> bool {
    C::ACCUMULATES
}

