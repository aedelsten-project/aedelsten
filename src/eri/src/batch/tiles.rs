// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The batches above `kernels::L_UNROLL`: per block of bra primitives the R matrices stacked against each ket
//! primitive's E in one BLAS product, contracted on the cd side, then E_ab per bra primitive and the ab contraction.

use mmd::hermite_expansion::hermite_count;
use arrays::matrix::*;
use arrays::array3d::*;
use arrays::array4d::*;
use arrays::array6d::*;
use arrays::array7d::*;
use arrays::array8d::*;
use arrays::scratch::{checked_product, Arena, ArenaError};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;
use basis::gto_products::GTOProductTrait;

use mmd::hermite_coulomb_integrals::{RangeVersion, HermiteCoulombIntegrals4D};

use super::r_columns::construct_r_xxxx_columns;
use super::accumulates;

/// The batch `(pair_ab|pair_cd)` above `kernels::L_UNROLL`, through the BLAS tiles (see the top of this file).
#[inline(always)]
pub(crate) fn batch_xxxx<'a, R: RangeVersion, Sab: ShellKind, Scd: ShellKind>(
    pair_ab: &ShellPair<Sab>,
    pair_cd: &ShellPair<Scd>,
    bra_array: &Array4DView<'_, f64>,
    ket_array: &Array4D<f64>,
    arena: &mut Arena<'a>,
    omega: f64
) -> Result<Array8DViewMut<'a, f64>, ArenaError> {

    // ab metadata
    let lqn_a = pair_ab.get_lqn_a();
    let lqn_b = pair_ab.get_lqn_b();
    let lqn_ab = lqn_a + lqn_b;
    let sph_a = pair_ab.get_degeneracy_a();
    let sph_b = pair_ab.get_degeneracy_b();
    let (n_cont_a, n_cont_b) = pair_ab.get_contracted_pair_dims();
    let cc_ab = pair_ab.contraction_matrix();

    // cd metadata
    let lqn_c = pair_cd.get_lqn_a();
    let lqn_d = pair_cd.get_lqn_b();
    let lqn_cd = lqn_c + lqn_d;
    let sph_c = pair_cd.get_degeneracy_a();
    let sph_d = pair_cd.get_degeneracy_b();
    let (n_cont_c, n_cont_d) = pair_cd.get_contracted_pair_dims();
    let cc_cd = pair_cd.contraction_matrix();

    // Summed: every product accumulates into one slot. Otherwise one slot per product.
    let (accumulate_ab, accumulate_cd) = (accumulates(&pair_ab.contraction()), accumulates(&pair_cd.contraction()));

    // The outputs come from the batch's arena: they outlive it (`XxxxOutputs`). Everything else comes from a scope
    // over the rest of the arena (`XxxxScratch`), free again when the batch returns, before the walk digests it. ERE
    // in primitive (ab), one slab per ab slot, zeroed: accumulated ab products add up into it.
    let XxxxOutputs { mut e_r_e_prim, mut e_r_e_cont } = XxxxOutputs::new_in(pair_ab, pair_cd, arena)?;
    e_r_e_prim.fill(0.0);
    e_r_e_cont.fill(0.0);

    arena.scope(|scratch| -> Result<(), ArenaError> {
        let XxxxScratch { mut r_buffer, b, mut r_stack, mut re_prim, mut re_cont } = XxxxScratch::new_in(pair_ab, pair_cd, scratch)?;
        r_stack.fill(0.0);
        re_prim.fill(0.0);
        re_cont.fill(0.0);

        {
            let mut bra = bra_array.iter();
            let mut ere_slots = e_r_e_prim.iter_mut();
            let mut ere_shared = if accumulate_ab { ere_slots.next() } else { None };
            for block_ab in pair_ab.get_products().chunks(b) {
                // The slabs are borrowed for the block's products, released before its contraction.
                {
                    let mut slabs = re_prim.iter_mut();
                    let mut shared = if accumulate_cd { slabs.next() } else { None };
                    for (j, (cd, e_ket)) in pair_cd.get_products().iter().zip(ket_array.iter()).enumerate() {
                        let q = cd.get_p();
                        let q_x = cd.get_product_centre();
                        // Build R for every bra primitive of the block, each into its slot of the stack.
                        for (ii, ab) in block_ab.iter().enumerate() {
                            construct_r_xxxx_columns::<R>(
                                &mut r_buffer,
                                r_stack.view_dim1_matrix_mut(ii).iter_mut(),
                                ab.get_p(),
                                &ab.get_product_centre(),
                                lqn_ab,
                                q,
                                &q_x,
                                lqn_cd,
                                omega
                            );
                        }
                        // RE_prim of the block in primitive (cd): one product for the whole block. Summed products
                        // accumulate into one slab.
                        let mut own;
                        let (slab, beta) = match shared.as_mut() {
                            Some(slab) => (slab, if j > 0 { 1.0 } else { 0.0 }),
                            None => {
                                own = slabs.next().expect("one slab per ket primitive");
                                (&mut own, 0.0)
                            }
                        };
                        r_stack.view_as_matrix((2, 1)).matmul_scaled_prealloc(
                            &e_ket.view_as_matrix((1, 2)),
                            &mut slab.view_as_matrix_mut((2, 1)),
                            1.0,
                            beta,
                        );
                    }
                }
                // Transform the block's RE_prim to contracted (cd), RE_cont.
                if let Some(cc_cd) = cc_cd {
                    re_prim.view_as_matrix((3, 1)).matmul_prealloc(cc_cd, &mut re_cont.view_as_matrix_mut((3, 1)));
                }

                // Construct ERE in primitive (ab), per bra primitive of the block: its slot of RE_ket, a strided
                // product operand (the stacking that makes RE_ket one product puts every bra primitive in every
                // column).
                let r_e_ket = if cc_cd.is_some() { re_cont.view_as_array3d((1, 1, 2)) } else { re_prim.view_as_array3d((1, 1, 2)) };
                for (ii, e_bra) in bra.by_ref().take(block_ab.len()).enumerate() {
                    let mut own;
                    let (slot, beta) = match ere_shared.as_mut() {
                        Some(slot) => (slot, 1.0),
                        None => {
                            own = ere_slots.next().expect("one ERE slot per bra primitive");
                            (&mut own, 0.0)
                        }
                    };
                    e_bra.view_as_matrix((2, 1)).matmul_scaled_prealloc(
                        &r_e_ket.view_dim1_matrix(ii),
                        &mut slot.view_as_matrix_mut((2, 4)),
                        1.0,
                        beta,
                    );
                }
            }
        }
        Ok(())
    })?;

    // Transform ERE to contracted (ab) pair.
    match cc_ab {
        Some(cc_ab) => {
            e_r_e_prim.view_as_matrix((6, 1)).matmul_prealloc(
                cc_ab,
                &mut e_r_e_cont.view_as_matrix_mut((6, 2))
            );
            Ok(e_r_e_cont)
        }
        None => Ok(e_r_e_prim.into_reshaped_array8d((sph_a, sph_b, sph_c, sph_d, n_cont_c, n_cont_d, n_cont_a, n_cont_b))),
    }
}

/// `batch_xxxx`'s outputs, which outlive it: ERE per (ab) slot and, with a bra contraction, contracted. Taken only.
pub(crate) struct XxxxOutputs<'a> {
    /// ERE per primitive (ab) slot.
    pub(crate) e_r_e_prim: Array7DViewMut<'a, f64>,
    /// ERE contracted.
    pub(crate) e_r_e_cont: Array8DViewMut<'a, f64>,
}

impl<'a> XxxxOutputs<'a> {
    /// The outputs of the batch `(pair_ab|pair_cd)`, from `arena`.
    pub(crate) fn new_in<Sab: ShellKind, Scd: ShellKind>(pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let (sph_a, sph_b) = (pair_ab.get_degeneracy_a(), pair_ab.get_degeneracy_b());
        let (sph_c, sph_d) = (pair_cd.get_degeneracy_a(), pair_cd.get_degeneracy_b());
        let (n_cont_a, n_cont_b) = pair_ab.get_contracted_pair_dims();
        let (n_cont_c, n_cont_d) = pair_cd.get_contracted_pair_dims();
        let slots_ab = if accumulates(&pair_ab.contraction()) { 1 } else { pair_ab.get_products().len() };
        let n_b = if pair_ab.contraction_matrix().is_some() { n_cont_b } else { 0 };
        let prim = arena.alloc::<f64>(checked_product(&[sph_a, sph_b, sph_c, sph_d, n_cont_c, n_cont_d, slots_ab])?)?;
        let cont = arena.alloc::<f64>(checked_product(&[sph_a, sph_b, sph_c, sph_d, n_cont_c, n_cont_d, n_cont_a, n_b])?)?;
        Ok(XxxxOutputs {
            e_r_e_prim: Array7DViewMut::new(sph_a, sph_b, sph_c, sph_d, n_cont_c, n_cont_d, slots_ab, prim),
            e_r_e_cont: Array8DViewMut::new(sph_a, sph_b, sph_c, sph_d, n_cont_c, n_cont_d, n_cont_a, n_b, cont),
        })
    }
}

/// `batch_xxxx`'s scratch, in a scope after its outputs: the R buffer and, for a block of `b` bra primitives (as many
/// as the space left holds, `xxxx_bra_block`), the stacked R matrices and R . E_ket per cd slot and cd-contracted.
/// Taken only.
pub(crate) struct XxxxScratch<'s> {
    r_buffer: HermiteCoulombIntegrals4D<'s>,
    b: usize,
    r_stack: Array3DViewMut<'s, f64>,
    re_prim: Array4DViewMut<'s, f64>,
    re_cont: Array4DViewMut<'s, f64>,
}

impl<'s> XxxxScratch<'s> {
    /// The scratch of the batch `(pair_ab|pair_cd)`, from `scratch`.
    pub(crate) fn new_in<Sab: ShellKind, Scd: ShellKind>(pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>, scratch: &Arena<'s>) -> Result<Self, ArenaError> {
        let (lqn_ab, lqn_cd) = (pair_ab.get_lqn_a() + pair_ab.get_lqn_b(), pair_cd.get_lqn_a() + pair_cd.get_lqn_b());
        let (tuv_size, ket_tuv) = (hermite_count(lqn_ab), hermite_count(lqn_cd));
        let sph_cd = pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b();
        let slots_cd = if accumulates(&pair_cd.contraction()) { 1 } else { pair_cd.get_products().len() };
        let n_out_cd = pair_cd.contraction_matrix().map_or(0, |cc| cc.cols());
        let r_buffer = HermiteCoulombIntegrals4D::from_arena(lqn_ab + lqn_cd, scratch)?;
        // Blocks of `b` bra primitives share each ket primitive's E_ket in one R . E_ket product, as many as the space
        // left in the scratch holds (see xxxx_bra_block). Every block has `b` slots; the last may fill fewer (the
        // blocks are balanced). Its other slots keep the previous block's R: they go through the block's products
        // like the rest, row by row, and their rows are never read.
        let per_bra = std::mem::size_of::<f64>() * tuv_size * (ket_tuv + sph_cd * (slots_cd + n_out_cd));
        let b = xxxx_bra_block(pair_ab.get_products().len(), scratch.available(), per_bra);
        // The R matrices of a block, stacked: (tuv_size, b, ket_tuv), i.e. a (tuv_size b) x ket_tuv matrix in which
        // every ket column holds the bra primitives' columns in turn. R . E_ket of the block, one slab per cd slot (one
        // in all if the cd products accumulate); the same, cd contracted. A single-centre pair stores only the
        // triangle of its primitive products, and its contraction matrix maps them to the triangle of its contracted
        // pairs (see ShellPair::get_contracted_pair_dims), so R, RE and everything after are built for the unique pairs
        // only. Without a contraction matrix (Identity) the per-product buffers are the result: nothing is contracted.
        let r_stack = scratch.alloc::<f64>(checked_product(&[tuv_size, b, ket_tuv])?)?;
        let re_prim = scratch.alloc::<f64>(checked_product(&[tuv_size, b, sph_cd, slots_cd])?)?;
        let re_cont = scratch.alloc::<f64>(checked_product(&[tuv_size, b, sph_cd, n_out_cd])?)?;
        Ok(XxxxScratch {
            r_buffer,
            b,
            r_stack: Array3DViewMut::new(tuv_size, b, ket_tuv, r_stack),
            re_prim: Array4DViewMut::new(tuv_size, b, sph_cd, slots_cd, re_prim),
            re_cont: Array4DViewMut::new(tuv_size, b, sph_cd, n_out_cd, re_cont),
        })
    }
}

/// The most bra primitives `batch_xxxx` stacks into one R . E_ket product.
const XXXX_MAX_BRA_BLOCK: usize = 16;

/// `batch_xxxx`'s block of bra primitives: as many as fit in `available` bytes at `per_bra` bytes each, at most
/// `XXXX_MAX_BRA_BLOCK`, at least one (if one does not fit, its allocation fails with the size needed). Then balanced:
/// the same number of blocks with sizes as equal as they go, so a short last block leaves fewer slots unused than
/// there are blocks. The block follows the scratch the caller chose: a larger scratch gives larger blocks, up to the
/// cap.
fn xxxx_bra_block(dim_prim_ab: usize, available: usize, per_bra: usize) -> usize {
    // Room for the alignment padding of the block's allocations.
    const SLACK: usize = 256;
    let fits = available.saturating_sub(SLACK) / per_bra.max(1);
    let cap = fits.clamp(1, XXXX_MAX_BRA_BLOCK.min(dim_prim_ab).max(1));
    let blocks = dim_prim_ab.div_ceil(cap).max(1);
    dim_prim_ab.div_ceil(blocks).max(1)
}
