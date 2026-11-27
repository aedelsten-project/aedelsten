// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use arrays;
use basis::shell_pairs::ShellPair;

/// Negates one plane of a multi-component pair batch, for the antisymmetric planes of a batch computed with its two
/// shells swapped (see `ArrayBatchKernel::antisymmetric_plane`).
pub(crate) fn negate_plane(batch: &mut arrays::array5d::Array5DViewMut<'_, f64>, plane: usize) {
    use arrays::array5d::{Array5DTrait, Array5DTraitMut};
    use arrays::matrix::MatrixTraitMut;
    let (n_cont_a, n_cont_b) = (batch.dim3(), batch.dim4());
    for cb in 0..n_cont_b {
        for ca in 0..n_cont_a {
            let mut m = batch.view_sub_matrix_mut((plane, ca, cb));
            for v in m.get_data_mut() {
                *v = -*v;
            }
        }
    }
}

/// Adds a pair batch `(comps_a, comps_b, n_cont_a, n_cont_b)` into the upper triangle of `output`.
pub(crate) fn distribute_pair_batch<Mf64, S>(pair: &ShellPair<S>, output: &mut Mf64, integral_batch: &arrays::array4d::Array4DViewMut<'_, f64>)
where
    Mf64: arrays::matrix::MatrixTraitMut<f64>,
    S: basis::contraction::ShellKind,
{
    // Shell pairs are ordered for the ERIs, so (idx_a, idx_b) need not be ascending: a pair with idx_a > idx_b goes
    // to the upper triangle transposed (the lower triangle is filled by one copy at the end). A diagonal block takes
    // the non-transposed route, which reads memory sequentially.
    let start_idx_a = pair.get_idx_a();
    let start_idx_b = pair.get_idx_b();
    let contracted = (pair.get_n_cont_a(), pair.get_n_cont_b(), pair.get_single_centre());
    if start_idx_a > start_idx_b {
        distribute_array4d_to_matrix_impl::<true, _, _>(output, integral_batch, start_idx_a, start_idx_b, contracted, |a, b| pair.pair_slot(a, b));
    } else {
        distribute_array4d_to_matrix_impl::<false, _, _>(output, integral_batch, start_idx_a, start_idx_b, contracted, |a, b| pair.pair_slot(a, b));
    }
}

/// Adds a pair batch `(comps_a, comps_b, n_planes, n_cont_a, n_cont_b)` into the upper triangle of every plane.
pub(crate) fn distribute_pair_batch_planar<Af64, S>(pair: &ShellPair<S>, output: &mut Af64, integral_batch: &arrays::array5d::Array5DViewMut<'_, f64>)
where
    Af64: arrays::array3d::Array3DTraitMut<f64>,
    S: basis::contraction::ShellKind,
{
    let start_idx_a = pair.get_idx_a();
    let start_idx_b = pair.get_idx_b();
    let contracted = (pair.get_n_cont_a(), pair.get_n_cont_b(), pair.get_single_centre());
    if start_idx_a > start_idx_b {
        distribute_array5d_to_array3d_impl::<true, _, _>(output, integral_batch, start_idx_a, start_idx_b, contracted, |a, b| pair.pair_slot(a, b));
    } else {
        distribute_array5d_to_array3d_impl::<false, _, _>(output, integral_batch, start_idx_a, start_idx_b, contracted, |a, b| pair.pair_slot(a, b));
    }
}

// Adds a two-shell batch: every block directly, no triangle.
#[inline(always)]
pub(crate) fn distribute_two_shell_array5d_to_array3d<Mmut64, Af64>(output: &mut Mmut64, integral_batch: &Af64, start_idx_a: usize, start_idx_b: usize)
where
    Mmut64: arrays::array3d::Array3DTraitMut<f64>,
    Af64: arrays::array5d::Array5DTrait<f64>,
{
    let (n_a, n_b) = (integral_batch.dim3(), integral_batch.dim4());
    distribute_array5d_to_array3d_impl::<false, _, _>(output, integral_batch, start_idx_a, start_idx_b, (n_a, n_b, false), |a, b| Some(a + n_a * b));
}

// Adds the integral batch into the upper triangle of the output matrix, starting at the given indices.
#[inline(always)]
fn distribute_matrix_to_matrix<Mmut64, Mf64>(
    output: &mut Mmut64,
    integral_batch: &Mf64,
    start_idx_a: usize,
    start_idx_b: usize,
)
where
    Mmut64: arrays::matrix::MatrixTraitMut<f64>,
    Mf64: arrays::matrix::MatrixTrait<f64>
{
    // The integral batch is (nr_of_b_components, nr_of_a_components).
    debug_assert!(start_idx_a + integral_batch.rows() <= output.rows());
    debug_assert!(start_idx_b + integral_batch.cols() <= output.cols());
    for (out_c, j) in output.iter_mut().skip(start_idx_b).zip(integral_batch.iter()) {
        for (out, val) in out_c.iter_mut().skip(start_idx_a).zip(j.iter()) {
            *out += *val;
        }
    }
}

#[inline(always)]
fn distribute_matrix_to_matrix_transposed<Mmut64, Mf64>(
    output: &mut Mmut64,
    integral_batch: &Mf64,
    start_idx_a: usize,
    start_idx_b: usize,
)
where
    Mmut64: arrays::matrix::MatrixTraitMut<f64>,
    Mf64: arrays::matrix::MatrixTrait<f64>
{
    let size_b = integral_batch.rows();
    let size_a = integral_batch.cols();
    debug_assert!(start_idx_a + size_b <= output.cols());
    debug_assert!(start_idx_b + size_a <= output.rows());
    // The batch is read non-contiguously, by index; the output iterators are truncated to the batch's size.
    for (i, out_c) in output.iter_mut().skip(start_idx_a).take(size_b).enumerate() {
        for (j, out) in out_c.iter_mut().skip(start_idx_b).take(size_a).enumerate() {
            *out += integral_batch[(i, j)]
        }
    }

}

#[inline(always)]
fn distribute_array4d_to_matrix_impl<const TRANSPOSED: bool, Mmut64, Mf64>(
    output: &mut Mmut64,
    integral_batch: &Mf64,
    start_idx_a: usize,
    start_idx_b: usize,
    (n_cont_a, n_cont_b, single_centre): (usize, usize, bool),
    slot: impl Fn(usize, usize) -> Option<usize>,
) where
    Mmut64: arrays::matrix::MatrixTraitMut<f64>,
    Mf64: arrays::array4d::Array4DTrait<f64>,
{
    // The batch is (nr_of_a_components, nr_of_b_components, contracted-pair dimensions), column-major: the
    // (n_cont_a, n_cont_b) grid, the packed unique pairs a <= b of a same-shell pair, or the stored products of a
    // screened uncontracted pair; `slot` places a contracted pair (ShellPair::pair_slot). Each block goes to the
    // output at its (idx_a, idx_b) offset.
    let nr_of_a_components = integral_batch.dim0();
    let nr_of_b_components = integral_batch.dim1();
    let block_len = nr_of_a_components * nr_of_b_components;
    let data = integral_batch.get_data();
    let mut idx_b = start_idx_b;
    for cb in 0..n_cont_b {
        let mut idx_a = start_idx_a;
        let a_end = if single_centre { cb + 1 } else { n_cont_a };
        for ca in 0..a_end {
            // A block whose product was screened out (uncontracted pairs) is not in the batch.
            let Some(slot) = slot(ca, cb) else {
                idx_a += nr_of_a_components;
                continue;
            };
            let offset = slot * block_len;
            let block = arrays::matrix::MatrixView::new(nr_of_a_components, nr_of_b_components, &data[offset..offset + block_len]);
            if TRANSPOSED {
                distribute_matrix_to_matrix_transposed(output, &block, idx_a, idx_b);
            } else {
                distribute_matrix_to_matrix(output, &block, idx_a, idx_b);
            }
            idx_a += nr_of_a_components;
        }
        idx_b += nr_of_b_components;
    }
}

// Places a two-shell batch (comps_a, comps_b, n_cont_a, n_cont_b) in `output`: every block directly, no triangle.
#[inline(always)]
pub(crate) fn distribute_array4d_to_matrix<Mmut64, Mf64>(
    output: &mut Mmut64,
    integral_batch: &Mf64,
    start_idx_a: usize,
    start_idx_b: usize,
) where
    Mmut64: arrays::matrix::MatrixTraitMut<f64>,
    Mf64: arrays::array4d::Array4DTrait<f64>,
{
    let (n_cont_a, n_cont_b) = (integral_batch.dim2(), integral_batch.dim3());
    distribute_array4d_to_matrix_impl::<false, _, _>(
        output, integral_batch, start_idx_a, start_idx_b, (n_cont_a, n_cont_b, false),
        |a, b| Some(basis::shell_pairs::contracted_pair_slot(false, n_cont_a, a, b)),
    )
}


// `distribute_array4d_to_matrix_impl` for a batch (comps_a, comps_b, n_planes, contracted-pair dimensions): every
// plane's blocks into that plane of the output.
#[inline(always)]
fn distribute_array5d_to_array3d_impl<const TRANSPOSED: bool, Mmut64, Af64>(
    output: &mut Mmut64,
    integral_batch: &Af64,
    start_idx_a: usize,
    start_idx_b: usize,
    (n_cont_a, n_cont_b, single_centre): (usize, usize, bool),
    slot: impl Fn(usize, usize) -> Option<usize>,
) where
    Mmut64: arrays::array3d::Array3DTraitMut<f64>,
    Af64: arrays::array5d::Array5DTrait<f64>,
{
    let nr_of_a_components = integral_batch.dim0();
    let nr_of_b_components = integral_batch.dim1();
    let n_planes = integral_batch.dim2();
    let block_len = nr_of_a_components * nr_of_b_components;
    let data = integral_batch.get_data();
    for plane in 0..n_planes {
        let mut out_plane = output.view_sub_matrix_mut(plane);
        let mut idx_b = start_idx_b;
        for cb in 0..n_cont_b {
            let mut idx_a = start_idx_a;
            let a_end = if single_centre { cb + 1 } else { n_cont_a };
            for ca in 0..a_end {
                let Some(slot) = slot(ca, cb) else {
                    idx_a += nr_of_a_components;
                    continue;
                };
                let offset = (plane + n_planes * slot) * block_len;
                let block = arrays::matrix::MatrixView::new(nr_of_a_components, nr_of_b_components, &data[offset..offset + block_len]);
                if TRANSPOSED {
                    distribute_matrix_to_matrix_transposed(&mut out_plane, &block, idx_a, idx_b);
                } else {
                    distribute_matrix_to_matrix(&mut out_plane, &block, idx_a, idx_b);
                }
                idx_a += nr_of_a_components;
            }
            idx_b += nr_of_b_components;
        }
    }
}
