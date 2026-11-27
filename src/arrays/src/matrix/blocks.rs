// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Small blocks of a matrix transformed by small factors on both sides, `left^T A[block] right`: plain loops, no BLAS
//! (a BLAS call costs more than a 3 x 3 tile). For block-sparse transforms that act shell block by shell block, such as
//! a symmetry operation's AO representation. The block of `A` is addressed by its first row and column, so it is read
//! in place, a column piece at a time, without a gather.

use super::*;

/// A signed permutation matrix, as a factor: column `j` has its one nonzero element, `signs[j]` (±1), in row
/// `rows[j]`.
#[derive(Clone, Copy, Debug)]
pub struct SignedPermutation<'a> {
    /// The row of each column's nonzero element.
    pub rows: &'a [usize],
    /// Each column's nonzero element, ±1.
    pub signs: &'a [f64],
}

impl SignedPermutation<'_> {
    /// The number of columns.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether it has no columns.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// `out[out_at + (i, j)] = alpha (left^T A[a_at + (k, l)] right)_ij + beta out[out_at + (i, j)]`, for
/// `i < left.cols()`, `j < right.cols()`: the block of `a` read is `left.rows() x right.rows()` from `a_at`. `scratch`
/// holds at least `left.rows()` f64s. With `beta == 0` the output is overwritten, whatever it held.
#[allow(clippy::too_many_arguments)]
pub fn block_sandwich_prealloc<A, O>(
    left: &MatrixView<'_, f64>,
    a: &A,
    a_at: (usize, usize),
    right: &MatrixView<'_, f64>,
    out: &mut O,
    out_at: (usize, usize),
    alpha: f64,
    beta: f64,
    scratch: &mut [f64],
) where
    A: MatrixTrait<f64>,
    O: MatrixTraitMut<f64>,
{
    let (n_k, n_l) = (left.rows(), right.rows());
    assert!(scratch.len() >= n_k, "block_sandwich_prealloc: scratch of {} for a block of {n_k} rows", scratch.len());
    assert!(a_at.0 + n_k <= a.rows() && a_at.1 + n_l <= a.cols(), "block_sandwich_prealloc: block outside A");
    assert!(out_at.0 + left.cols() <= out.rows() && out_at.1 + right.cols() <= out.cols(), "block_sandwich_prealloc: block outside out");
    let t = &mut scratch[..n_k];
    for (j, right_col) in right.iter_col().enumerate() {
        // t = A[block] right[:, j], a column piece of A at a time.
        t.fill(0.0);
        for (l, &r) in right_col.iter().enumerate() {
            if r != 0.0 {
                let a_piece = &a.get_col(a_at.1 + l)[a_at.0..a_at.0 + n_k];
                t.iter_mut().zip(a_piece).for_each(|(t_k, &a_kl)| *t_k += r * a_kl);
            }
        }
        let out_piece = &mut out.get_col_mut(out_at.1 + j)[out_at.0..out_at.0 + left.cols()];
        for (out_ij, left_col) in out_piece.iter_mut().zip(left.iter_col()) {
            let value = alpha * left_col.iter().zip(t.iter()).map(|(l_ki, t_k)| l_ki * t_k).sum::<f64>();
            *out_ij = if beta == 0.0 { value } else { value + beta * *out_ij };
        }
    }
}

/// `block_sandwich_prealloc` with signed-permutation factors: `out[out_at + (i, j)] = alpha left.signs[i]
/// right.signs[j] A[a_at + (left.rows[i], right.rows[j])] + beta out[out_at + (i, j)]`. A reindexed, signed copy: no
/// arithmetic beyond the signs and the scaling.
#[allow(clippy::too_many_arguments)]
pub fn block_signed_permutation_prealloc<A, O>(
    left: SignedPermutation<'_>,
    a: &A,
    a_at: (usize, usize),
    right: SignedPermutation<'_>,
    out: &mut O,
    out_at: (usize, usize),
    alpha: f64,
    beta: f64,
) where
    A: MatrixTrait<f64>,
    O: MatrixTraitMut<f64>,
{
    assert_eq!(left.rows.len(), left.signs.len(), "block_signed_permutation_prealloc: left rows and signs");
    assert_eq!(right.rows.len(), right.signs.len(), "block_signed_permutation_prealloc: right rows and signs");
    assert!(out_at.0 + left.len() <= out.rows() && out_at.1 + right.len() <= out.cols(), "block_signed_permutation_prealloc: block outside out");
    for (j, (&r_row, &r_sign)) in right.rows.iter().zip(right.signs).enumerate() {
        let a_col = &a.get_col(a_at.1 + r_row)[a_at.0..];
        let out_piece = &mut out.get_col_mut(out_at.1 + j)[out_at.0..out_at.0 + left.len()];
        for (out_ij, (&l_row, &l_sign)) in out_piece.iter_mut().zip(left.rows.iter().zip(left.signs)) {
            // A lookup through the permutation: the index arithmetic is the operation.
            let value = alpha * l_sign * r_sign * a_col[l_row];
            *out_ij = if beta == 0.0 { value } else { value + beta * *out_ij };
        }
    }
}
