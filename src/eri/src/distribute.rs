// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use arrays::array4d::*;
use arrays::array6d::*;
use arrays::array8d::*;
use arrays::matrix::MatrixView;
use arrays::scratch::{Arena, ArenaError};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;

use crate::batch::ERIBatchType;

/// One shell-quartet batch `(pair_ab|pair_cd)`, handed whole to `TensorChoice::distribute_block`, as computed or with
/// its slots merged into function order (`with_merged_slots`). Read-only; it lives in the walk's arena, so nothing may
/// keep it past the call.
pub struct EriBlock<'b, 'a> {
    source: Source<'b, 'a>,
    bound: Option<f64>,
}

enum Source<'b, 'a> {
    Batch(&'b ERIBatchType<'a>),
    /// A batch in function order, `(s_A, s_B, s_C, s_D)`, every slot weight applied and unstored slots zero.
    Merged { data: &'b [f64], dims: [usize; 4] },
}

impl<'b, 'a> EriBlock<'b, 'a> {
    /// The block of `batch` as computed, with its Schwarz bound when the walk screens.
    pub(crate) fn new(batch: &'b ERIBatchType<'a>, bound: Option<f64>) -> Self {
        EriBlock { source: Source::Batch(batch), bound }
    }

    /// A batch already in function order (`Source::Merged`), `(s_A, s_B, s_C, s_D)`.
    pub(crate) fn merged(data: &'b [f64], sizes: [usize; 4], bound: Option<f64>) -> Self {
        EriBlock { source: Source::Merged { data, dims: sizes }, bound }
    }

    /// The batch itself. A merged batch reaches only accumulators that digest whole blocks
    /// (`DirectFock::merged_slots`), never the per-integral `distribute`.
    pub(crate) fn batch(&self) -> &'b ERIBatchType<'a> {
        match self.source {
            Source::Batch(batch) => batch,
            Source::Merged { .. } => unreachable!("a merged batch handed to the per-integral distribute"),
        }
    }

    /// The Schwarz bound of the batch's integrals (the product of its pairs' bounds), when the walk screens;
    /// `None` from the unscreened `compute_eri_stored`.
    pub fn bound(&self) -> Option<f64> {
        self.bound
    }

    /// `(i, j, k, l, c, d, a, b)`, column-major, `i` fastest: the components of the four shells, then the
    /// contracted-pair slot dimensions of the ket and the bra (`ShellPair::get_contracted_pair_dims`). An `(xx|ss)`
    /// or `(ss|ss)` batch has unit component dimensions where its shells are s shells.
    /// A merged batch (`with_merged_slots`) is `(s_A, s_B, s_C, s_D, 1, 1, 1, 1)`: one slot each side, spanning its
    /// shells' functions.
    pub fn dims(&self) -> [usize; 8] {
        match &self.source {
            Source::Batch(ERIBatchType::EightD(batch)) => batch_dims(batch),
            Source::Batch(ERIBatchType::SixD(b)) => [b.dim0(), b.dim1(), 1, 1, b.dim2(), b.dim3(), b.dim4(), b.dim5()],
            Source::Batch(ERIBatchType::FourD(b)) => [1, 1, 1, 1, b.dim0(), b.dim1(), b.dim2(), b.dim3()],
            Source::Merged { dims: [s_a, s_b, s_c, s_d], .. } => [*s_a, *s_b, *s_c, *s_d, 1, 1, 1, 1],
        }
    }

    /// The integrals, in the layout of `dims`.
    pub fn data(&self) -> &[f64] {
        match &self.source {
            Source::Batch(ERIBatchType::EightD(batch)) => batch.get_data(),
            Source::Batch(ERIBatchType::SixD(batch)) => batch.get_data(),
            Source::Batch(ERIBatchType::FourD(batch)) => batch.get_data(),
            Source::Merged { data, .. } => data,
        }
    }

    /// The batch as a `(n_i n_j) x (n_k n_l c d a b)` matrix (`dims`): the bra components by everything else. One
    /// bra slot's integrals are `n_k n_l c d` consecutive columns (`view_columns`), a slot quartet's `n_k n_l`.
    pub fn as_matrix(&self) -> MatrixView<'_, f64> {
        let [n_i, n_j, ..] = self.dims();
        let data = self.data();
        MatrixView::new(n_i * n_j, data.len() / (n_i * n_j), data)
    }

    /// The `(i, j, k, l)` component block of one slot quartet (`for_each_slot_quartet`), `(n_i, n_j, n_k, n_l)`.
    pub fn slot_block(&self, quartet: &SlotQuartet) -> Array4DView<'_, f64> {
        let [n_i, n_j, n_k, n_l, ..] = self.dims();
        Array4DView::new(n_i, n_j, n_k, n_l, &self.data()[quartet.offset..][..n_i * n_j * n_k * n_l])
    }
}

/// One stored pair of contracted-pair slots of a batch, from `for_each_slot_quartet`.
#[derive(Clone, Copy, Debug)]
pub struct SlotQuartet {
    /// The weight that makes the full stored block exact when each of its elements is applied with all 8 of its
    /// permutations, none deduplicated: 1/2 for an `a == b` slot of a single-centre pair (its `(i, j)` block holds
    /// both orders), the same for the ket, and 1/2 for a batch whose bra pair is its ket pair (it holds both
    /// orders of every slot pair).
    pub weight: f64,
    /// The basis-function index of component 0 of each of the four contracted functions `(a, b, c, d)`.
    pub functions: [usize; 4],
    /// The contracted functions `(a, b, c, d)` within their shells.
    pub contracted: [usize; 4],
    /// The position of the `(i, j, k, l)` sub-block in `EriBlock::data`; it is contiguous, column-major, with the
    /// component counts `dims()[..4]`.
    pub offset: usize,
}

/// Every stored contracted-pair slot of `pair`: `visit([a, b], slot, weight)`, `weight` 1/2 for an `a == b` slot
/// of a single-centre pair (its component block holds both orders), else 1. The slot is the pair's index in the
/// batch's `(c, d)` or `(a, b)` dimensions.
pub fn for_each_pair_slot<F, S>(pair: &ShellPair<S>, mut visit: F)
where
    F: FnMut([usize; 2], usize, f64),
    S: ShellKind,
{
    let single = pair.get_single_centre();
    for b in 0..pair.get_n_cont_b() {
        for a in 0..pair.get_n_cont_a() {
            // A single-centre pair stores its a <= b triangle; a screened-out product is not in the batch.
            if single && a > b {
                continue;
            }
            let Some(slot) = pair.pair_slot(a, b) else { continue };
            visit([a, b], slot, if single && a == b { 0.5 } else { 1.0 });
        }
    }
}

/// 1/2 for a batch whose bra pair is its ket pair (it holds both orders of every slot pair), else 1.
pub fn bra_ket_weight<Sab: ShellKind, Scd: ShellKind>(pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>) -> f64 {
    // The same pair on both sides (pairs of different types never are).
    let self_pair = std::ptr::eq(pair_ab as *const ShellPair<Sab> as *const (), pair_cd as *const ShellPair<Scd> as *const ());
    if self_pair { 0.5 } else { 1.0 }
}

/// Every stored slot pair of the batch `(pair_ab|pair_cd)`, with the weight of `SlotQuartet::weight`. With those
/// weights, summing every element of every sub-block over its 8 permutations gives each AO quartet exactly once,
/// the same as `distribute` applied to the unique integrals with their distinct permutations. Kinds with one
/// contracted function a shell (`ShellKind::ONE_CONTRACTED`) take one visit, without the slot loops.
pub fn for_each_slot_quartet<F, Sab, Scd>(block: &EriBlock<'_, '_>, pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>, mut visit: F)
where
    F: FnMut(SlotQuartet),
    Sab: ShellKind,
    Scd: ShellKind,
{
    let dims = block.dims();
    let (idx_a, idx_b, idx_c, idx_d) = (pair_ab.get_idx_a(), pair_ab.get_idx_b(), pair_cd.get_idx_a(), pair_cd.get_idx_b());
    if let Source::Merged { .. } = block.source {
        // One block over the shells' functions, its weights applied.
        visit(SlotQuartet { weight: 1.0, functions: [idx_a, idx_b, idx_c, idx_d], contracted: [0; 4], offset: 0 });
        return;
    }
    debug_assert_eq!((dims[4], dims[5]), pair_cd.get_contracted_pair_dims());
    debug_assert_eq!((dims[6], dims[7]), pair_ab.get_contracted_pair_dims());
    let [n_i, n_j, n_k, n_l, ..] = dims;
    let stride_cd = n_i * n_j * n_k * n_l;
    let stride_ab = stride_cd * dims[4] * dims[5];
    let self_weight = bra_ket_weight(pair_ab, pair_cd);
    if Sab::ONE_CONTRACTED && Scd::ONE_CONTRACTED {
        // One slot a pair, its contracted functions (0, 0): 1/2 for a single-centre pair (its slot is a == b).
        debug_assert!(pair_ab.pair_slot(0, 0) == Some(0) && pair_cd.pair_slot(0, 0) == Some(0));
        let single = |single_centre: bool| if single_centre { 0.5 } else { 1.0 };
        visit(SlotQuartet {
            weight: self_weight * single(pair_ab.get_single_centre()) * single(pair_cd.get_single_centre()),
            functions: [idx_a, idx_b, idx_c, idx_d],
            contracted: [0; 4],
            offset: 0,
        });
        return;
    }

    for_each_pair_slot(pair_ab, |[a, b], slot_ab, weight_ab| {
        for_each_pair_slot(pair_cd, |[c, d], slot_cd, weight_cd| {
            visit(SlotQuartet {
                weight: self_weight * weight_ab * weight_cd,
                functions: [idx_a + a * n_i, idx_b + b * n_j, idx_c + c * n_k, idx_d + d * n_l],
                contracted: [a, b, c, d],
                offset: slot_cd * stride_cd + slot_ab * stride_ab,
            });
        });
    });
}

/// The memory `with_merged_slots` takes for a batch it merges: the bra slots' offsets and weights and the merged block,
/// `(s_A, s_B, s_C, s_D)`. Taken only.
pub struct MergedSlots<'s> {
    from: &'s mut [usize],
    to: &'s mut [usize],
    weights: &'s mut [f64],
    out: &'s mut [f64],
    dims: [usize; 4],
}

impl<'s> MergedSlots<'s> {
    /// `None` when `block` is merged already or has one slot each side (it is digested as it is).
    pub fn new_in<Sab: ShellKind, Scd: ShellKind>(block: &EriBlock<'_, '_>, pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>, arena: &Arena<'s>) -> Result<Option<Self>, ArenaError> {
        let [n_i, n_j, n_k, n_l, d_c, d_d, d_a, d_b] = block.dims();
        if matches!(block.source, Source::Merged { .. }) || d_a * d_b * d_c * d_d == 1 {
            return Ok(None);
        }
        let (s_a, s_b) = (n_i * pair_ab.get_n_cont_a(), n_j * pair_ab.get_n_cont_b());
        let (s_c, s_d) = (n_k * pair_cd.get_n_cont_a(), n_l * pair_cd.get_n_cont_b());
        let slots = d_a * d_b;
        Ok(Some(MergedSlots {
            from: arena.alloc::<usize>(slots)?,
            to: arena.alloc::<usize>(slots)?,
            weights: arena.alloc::<f64>(slots)?,
            out: arena.alloc::<f64>(s_a * s_b * s_c * s_d)?,
            dims: [s_a, s_b, s_c, s_d],
        }))
    }

    /// The merged block over this memory, not filled (the scratch query: only its dimensions are read).
    pub fn unfilled(&self, bound: Option<f64>) -> EriBlock<'_, 'static> {
        EriBlock::merged(&*self.out, self.dims, bound)
    }

    /// `block` merged into this memory: the slot weights of `for_each_slot_quartet` applied, unstored slots zero.
    fn fill<'m, Sab: ShellKind, Scd: ShellKind>(&'m mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<Sab>, pair_cd: &ShellPair<Scd>) -> EriBlock<'m, 'static> {
        let [n_i, n_j, n_k, n_l, d_c, d_d, ..] = block.dims();
        let [s_a, _, s_c, _] = self.dims;
        let (bra_len, s_ab) = (n_i * n_j, s_a * self.dims[1]);
        let stride_cd = bra_len * n_k * n_l;
        let stride_ab = stride_cd * d_c * d_d;
        let self_weight = bra_ket_weight(pair_ab, pair_cd);
        let data = block.data();
        // The bra slots once: where each starts in the batch and in a merged column, and its weight.
        let mut count = 0;
        for_each_pair_slot(pair_ab, |[a, b], slot_ab, weight_ab| {
            (self.from[count], self.to[count], self.weights[count]) = (slot_ab * stride_ab, b * n_j * s_a + a * n_i, self_weight * weight_ab);
            count += 1;
        });
        let (from, to, weights) = (&self.from[..count], &self.to[..count], &self.weights[..count]);
        let out = &mut *self.out;
        out.fill(0.0);
        for_each_pair_slot(pair_cd, |[c, d], slot_cd, weight_cd| {
            for l in 0..n_l {
                for k in 0..n_k {
                    // Ket component (k, l) of slot (c, d): one column of the merged block, (c n_k + k, d n_l + l).
                    let column = &mut out[s_ab * (c * n_k + k + s_c * (d * n_l + l))..][..s_ab];
                    let from_kl = &data[slot_cd * stride_cd + bra_len * (k + n_k * l)..];
                    // Bra component (i, j) of slot (a, b) at (a n_i + i, b n_j + j): runs of n_i.
                    for ((&from, &to), &weight) in from.iter().zip(to).zip(weights) {
                        let weight = weight * weight_cd;
                        for (j, from_j) in from_kl[from..from + bra_len].chunks_exact(n_i).enumerate() {
                            for (out, &v) in column[to + j * s_a..][..n_i].iter_mut().zip(from_j) {
                                *out = weight * v;
                            }
                        }
                    }
                }
            }
        });
        EriBlock::merged(&*self.out, self.dims, block.bound)
    }
}

/// The batch `(pair_ab|pair_cd)` as `visit` digests it best: a batch with one slot each side as it is; otherwise
/// merged into its shells' function order, `(s_A, s_B, s_C, s_D)` in `scratch` (`MergedSlots`), so that a kernel's
/// runs and products span whole shells. The slot weights of `for_each_slot_quartet` are applied to the integrals and
/// unstored slots are zero, so that summing every element over its 8 permutations gives what the slot quartets give.
pub fn with_merged_slots<Sab, Scd>(
    block: &EriBlock<'_, '_>,
    pair_ab: &ShellPair<Sab>,
    pair_cd: &ShellPair<Scd>,
    scratch: &mut Arena<'_>,
    visit: impl FnOnce(&EriBlock<'_, '_>, &mut Arena<'_>) -> Result<(), ArenaError>,
) -> Result<(), ArenaError>
where
    Sab: ShellKind,
    Scd: ShellKind,
{
    scratch.scope(|merged| match MergedSlots::new_in(block, pair_ab, pair_cd, merged)? {
        None => visit(block, merged),
        Some(mut slots) => {
            let block = slots.fill(block, pair_ab, pair_cd);
            merged.scope(|arena| visit(&block, arena))
        }
    })
}

/// The unique-integral walk of a batch: `distribute` once per unique integral, by the batch's kind.
pub(crate) fn distribute_eri_batch<T, S>(
    eri_tensor: &mut T,
    eri_batch: &ERIBatchType,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>
) where
    T: crate::TensorChoice,
    S: ShellKind,
{
    match eri_batch {
        ERIBatchType::FourD(batch) => distribute_batch_general_ssss(eri_tensor, batch, pair_ab, pair_cd),
        ERIBatchType::SixD(batch) => distribute_batch_general_xxss(eri_tensor, batch, pair_ab, pair_cd),
        ERIBatchType::EightD(batch) => distribute_batch_general(eri_tensor, batch, pair_ab, pair_cd),
    }
}

/// An (xx|xx) batch: `(i, j, k, l, c, d, a, b)`, contracted-pair dimensions last.
#[inline(always)]
fn distribute_batch_general<T, M, S: ShellKind>(
    output: &mut T,
    eri_integrals: &M,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
) where
    T: crate::TensorChoice,
    M: Array8DTrait<f64>,
{
    let data = eri_integrals.get_data();
    for_each_unique_general(batch_dims(eri_integrals), pair_ab, pair_cd, |a, b, c, d, offset| {
        output.distribute(a, b, c, d, data[offset]);
    });
}

/// The dimensions of a generally contracted batch, `(i, j, k, l, c, d, a, b)` as in `Array8DTrait`.
#[inline(always)]
pub(crate) fn batch_dims<M: Array8DTrait<f64>>(batch: &M) -> [usize; 8] {
    [batch.dim0(), batch.dim1(), batch.dim2(), batch.dim3(), batch.dim4(), batch.dim5(), batch.dim6(), batch.dim7()]
}

/// The unique-integral walk of a generally contracted batch of the quartet `(pair_ab|pair_cd)`, shared by
/// the energy distribution (`distribute_batch_general`) and the gradient emission (`eri::gradient`, which
/// reads several derivative batches at the same position). Calls `visit(a, b, c, d, offset)` once per
/// unique integral: `a..d` are the basis-function indices, `offset` the column-major position in a batch
/// of dimensions `dims` (`(i, j, k, l, c, d, a, b)`, the `Array8D` layout, `i` fastest; `(c, d)` and `(a, b)`
/// are the pairs' contracted-pair dimensions, `ShellPair::get_contracted_pair_dims`).
#[inline(always)]
pub(crate) fn for_each_unique_general<F, Sab, Scd>(
    dims: [usize; 8],
    pair_ab: &ShellPair<Sab>,
    pair_cd: &ShellPair<Scd>,
    mut visit: F,
) where
    F: FnMut(usize, usize, usize, usize, usize),
    Sab: ShellKind,
    Scd: ShellKind,
{
    let mut idx_a = pair_ab.get_idx_a();
    let mut idx_b = pair_ab.get_idx_b();
    let mut idx_c = pair_cd.get_idx_a();
    let mut idx_d = pair_cd.get_idx_b();

    let a_reset = idx_a;
    let c_reset = idx_c;
    let d_reset = idx_d;

    // The same pair on both sides (pairs of different types never are).
    let self_pair = std::ptr::eq(pair_ab as *const ShellPair<Sab> as *const (), pair_cd as *const ShellPair<Scd> as *const ());
    let ab_single = pair_ab.get_single_centre();
    let cd_single = pair_cd.get_single_centre();

    // The shells' component counts.
    let [shell_size_a, shell_size_b, shell_size_c, shell_size_d, ..] = dims;
    let (n_a, n_b) = (pair_ab.get_n_cont_a(), pair_ab.get_n_cont_b());
    let (n_c, n_d) = (pair_cd.get_n_cont_a(), pair_cd.get_n_cont_b());
    debug_assert_eq!((dims[4], dims[5]), pair_cd.get_contracted_pair_dims());
    debug_assert_eq!((dims[6], dims[7]), pair_ab.get_contracted_pair_dims());
    // Column-major strides of (j, k, l, c, d, a, b); i has stride 1.
    let mut stride = [0usize; 8];
    stride[0] = 1;
    for axis in 1..8 {
        stride[axis] = stride[axis - 1] * dims[axis - 1];
    }

    for b in 0..n_b {
        let a_bound = if ab_single { b.min(n_a.saturating_sub(1)) } else { n_a.saturating_sub(1) };
        for a in 0..=a_bound {
            let ab_tied = ab_single && a == b;
            let slot_ab = pair_ab.pair_slot(a, b);
            for d in 0..n_d {
                let c_bound = {
                    let bound = n_c.saturating_sub(1);
                    let bound = if self_pair { bound.min(a) } else { bound };
                    if cd_single { bound.min(d) } else { bound }
                };
                for c in 0..=c_bound {
                    let cd_tied = cd_single && c == d;
                    let self_ac_tied = self_pair && a == c;
                    // A block whose product was screened out (uncontracted pairs) is not in the batch.
                    if let (Some(slot_ab), Some(slot_cd)) = (slot_ab, pair_cd.pair_slot(c, d)) {
                        let base_cdab = slot_cd * stride[4] + slot_ab * stride[6];
                        let mut d_count = idx_d;
                        for l in 0..shell_size_d {
                            let mut c_count = idx_c;
                            let k_bound = if cd_tied { l.min(shell_size_c.saturating_sub(1)) } else { shell_size_c.saturating_sub(1) };
                            for k in 0..=k_bound {
                                let mut b_count = idx_b;
                                for j in 0..shell_size_b {
                                    let ab_i_bound = if ab_tied { j.min(shell_size_a.saturating_sub(1)) } else { shell_size_a.saturating_sub(1) };
                                    // Self-pair tiebreak at (a==c): i<=k necessary; if i==k too,
                                    // fall through to comparing (b,j) against (d,l) -- already
                                    // known -- to decide whether i==k itself is still allowed.
                                    let self_i_bound = if !self_ac_tied {
                                        Some(shell_size_a.saturating_sub(1))
                                    } else if b < d || (b == d && j <= l) {
                                        Some(k.min(shell_size_a.saturating_sub(1)))
                                    } else if k == 0 {
                                        None
                                    } else {
                                        Some((k - 1).min(shell_size_a.saturating_sub(1)))
                                    };
                                    if let Some(i_bound) = self_i_bound.map(|s| s.min(ab_i_bound)) {
                                        let base = base_cdab + j * stride[1] + k * stride[2] + l * stride[3];
                                        let mut a_count = idx_a;
                                        for i in 0..=i_bound {
                                            visit(a_count, b_count, c_count, d_count, base + i);
                                            a_count += 1;
                                        }
                                    }
                                    b_count += 1;
                                }
                                c_count += 1;
                            }
                            d_count += 1;
                        }
                    }
                    idx_c += shell_size_c;
                }
                idx_c = c_reset;
                idx_d += shell_size_d;
            }
            idx_c = c_reset;
            idx_d = d_reset;
            idx_a += shell_size_a;
        }
        idx_c = c_reset;
        idx_d = d_reset;
        idx_a = a_reset;
        idx_b += shell_size_b;
    }
}

/// An (xx|ss) batch: `(i, j, c, d, a, b)`.
#[inline(always)]
fn distribute_batch_general_xxss<T, M, S: ShellKind>(
    output: &mut T,
    eri_integrals: &M,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
) where
    T: crate::TensorChoice,
    M: Array6DTrait<f64>,
{
    let mut idx_a = pair_ab.get_idx_a();
    let mut idx_b = pair_ab.get_idx_b();
    let idx_c = pair_cd.get_idx_a();
    let idx_d = pair_cd.get_idx_b();

    let a_reset = idx_a;

    let ab_single = pair_ab.get_single_centre();
    let cd_single = pair_cd.get_single_centre();

    // The shells' component counts.
    let shell_size_a = eri_integrals.dim0();
    let shell_size_b = eri_integrals.dim1();

    let (n_a, n_b) = (pair_ab.get_n_cont_a(), pair_ab.get_n_cont_b());
    let (n_c, n_d) = (pair_cd.get_n_cont_a(), pair_cd.get_n_cont_b());
    debug_assert_eq!((eri_integrals.dim2(), eri_integrals.dim3()), pair_cd.get_contracted_pair_dims());
    debug_assert_eq!((eri_integrals.dim4(), eri_integrals.dim5()), pair_ab.get_contracted_pair_dims());
    let stride_cd = shell_size_a * shell_size_b;
    let stride_ab = stride_cd * eri_integrals.dim2() * eri_integrals.dim3();
    let data = eri_integrals.get_data();

    for b in 0..n_b {
        let a_bound = if ab_single { b.min(n_a.saturating_sub(1)) } else { n_a.saturating_sub(1) };
        for a in 0..=a_bound {
            let ab_tied = ab_single && a == b;
            let mut d_count = idx_d;
            // Blocks whose product was screened out (uncontracted pairs) are not in the batch.
            let Some(slot_ab) = pair_ab.pair_slot(a, b) else {
                idx_a += shell_size_a;
                continue;
            };
            let base_ab = slot_ab * stride_ab;
            for d in 0..n_d {
                let mut c_count = idx_c;
                let c_bound = if cd_single { d.min(n_c.saturating_sub(1)) } else { n_c.saturating_sub(1) };
                for c in 0..=c_bound {
                    let Some(slot_cd) = pair_cd.pair_slot(c, d) else {
                        c_count += 1;
                        continue;
                    };
                    let base = base_ab + slot_cd * stride_cd;
                    let mut b_count = idx_b;
                    for j in 0..shell_size_b {
                        let mut a_count = idx_a;
                        let i_bound = if ab_tied { j.min(shell_size_a.saturating_sub(1)) } else { shell_size_a.saturating_sub(1) };
                        for i in 0..=i_bound {
                            let integral = data[base + i + shell_size_a * j];
                            output.distribute(a_count, b_count, c_count, d_count, integral);
                            a_count += 1;
                        }
                        b_count += 1;
                    }
                    c_count += 1;
                }
                d_count += 1;
            }
            idx_a += shell_size_a;
        }
        idx_a = a_reset;
        idx_b += shell_size_b;
    }
}

/// An (ss|ss) batch: `(c, d, a, b)`.
#[inline(always)]
fn distribute_batch_general_ssss<T, M, S: ShellKind>(
    output: &mut T,
    eri_integrals: &M,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
) where
    T: crate::TensorChoice,
    M: Array4DTrait<f64>,
{
    let idx_a = pair_ab.get_idx_a();
    let idx_b = pair_ab.get_idx_b();
    let idx_c = pair_cd.get_idx_a();
    let idx_d = pair_cd.get_idx_b();

    let self_pair = std::ptr::eq(pair_ab, pair_cd);
    let ab_single = pair_ab.get_single_centre();
    let cd_single = pair_cd.get_single_centre();

    let (dim2, dim3) = (pair_ab.get_n_cont_a(), pair_ab.get_n_cont_b()); // a's, b's range
    let (dim0, dim1) = (pair_cd.get_n_cont_a(), pair_cd.get_n_cont_b()); // c's, d's range
    debug_assert_eq!((eri_integrals.dim0(), eri_integrals.dim1()), pair_cd.get_contracted_pair_dims());
    debug_assert_eq!((eri_integrals.dim2(), eri_integrals.dim3()), pair_ab.get_contracted_pair_dims());
    let stride_ab = eri_integrals.dim0() * eri_integrals.dim1();
    let data = eri_integrals.get_data();

    let mut b_count = idx_b;
    for l in 0..dim3 {
        let mut a_count = idx_a;
        let k_bound = if ab_single { l.min(dim2.saturating_sub(1)) } else { dim2.saturating_sub(1) };
        for k in 0..=k_bound {
            // Blocks whose product was screened out (uncontracted pairs) are not in the batch.
            let Some(slot_ab) = pair_ab.pair_slot(k, l) else {
                a_count += 1;
                continue;
            };
            let base = slot_ab * stride_ab;
            let mut d_count = idx_d;
            for j in 0..dim1 {
                let i_upper = if cd_single { j.min(dim0.saturating_sub(1)) } else { dim0.saturating_sub(1) };
                let i_lower = if self_pair {
                    if l <= j { k } else { k.saturating_add(1) }
                } else {
                    0
                };
                let mut c_count = idx_c + i_lower;
                for i in i_lower..=i_upper {
                    if let Some(slot_cd) = pair_cd.pair_slot(i, j) {
                        output.distribute(a_count, b_count, c_count, d_count, data[base + slot_cd]);
                    }
                    c_count += 1;
                }
                d_count += 1;
            }
            a_count += 1;
        }
        b_count += 1;
    }
}
