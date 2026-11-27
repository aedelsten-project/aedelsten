// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Matrix-matrix and matrix-vector products through BLAS (column-major, explicit leading dimensions).

use super::*;

/// `(rows, cols)` of a stored `rows x cols` operand as BLAS uses it, after the transpose flag.
#[inline(always)]
pub(super) fn logical_gemm_shape(rows: i32, cols: i32, trans: CblasTranspose) -> (i32, i32) {
    match trans {
        CblasTranspose::CblasNoTrans => (rows, cols),
        _ => (cols, rows),
    }
}

/// `c = alpha * op(a) op(b) + beta * c` through dgemm, the operands stored `a_rows x a_cols` and `b_rows x b_cols`
/// with their columns contiguous, `op` each one's transpose flag.
#[inline(always)]
pub(super) fn matrix_matrix_multiplication(
    a: &[f64],
    a_rows: i32,
    a_cols: i32,
    trans_a: CblasTranspose,
    b: &[f64],
    b_rows: i32,
    b_cols: i32,
    trans_b: CblasTranspose,
    c: &mut [f64],
    alpha: f64,
    beta: f64
) {
    let ldc = logical_gemm_shape(a_rows, a_cols, trans_a).0;
    matrix_matrix_multiplication_ld(a, a_rows, a_cols, a_rows, trans_a, b, b_rows, b_cols, b_rows, trans_b, c, ldc, alpha, beta);
}

/// `matrix_matrix_multiplication` with the leading dimensions given explicitly (`lda >= a_rows`: the columns of `a`
/// are `lda` apart; the same for `b`, and `ldc` at least the output's rows for `c`), for blocks of larger buffers
/// (`StridedMatrixView`, `StridedMatrixViewMut`).
#[allow(clippy::too_many_arguments)]
pub(super) fn matrix_matrix_multiplication_ld(
    a: &[f64],
    a_rows: i32,
    a_cols: i32,
    lda: i32,
    trans_a: CblasTranspose,
    b: &[f64],
    b_rows: i32,
    b_cols: i32,
    ldb: i32,
    trans_b: CblasTranspose,
    c: &mut [f64],
    ldc: i32,
    alpha: f64,
    beta: f64
) {
    let (m, k) = logical_gemm_shape(a_rows, a_cols, trans_a);
    let (k2, n) = logical_gemm_shape(b_rows, b_cols, trans_b);
    debug_assert_eq!(k, k2, "inner dimension mismatch: {k} vs {k2}");
    debug_assert!(lda >= a_rows && ldb >= b_rows && ldc >= m);
    // BLAS requires a leading dimension of at least 1, also for an empty matrix.
    let (lda, ldb, ldc) = (lda.max(1), ldb.max(1), ldc.max(1));
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    dgemm(
        CblasColMajor,
        trans_a,
        trans_b,
        m,
        n,
        k,
        alpha,
        a_ptr,
        lda,
        b_ptr,
        ldb,
        beta,
        c_ptr,
        ldc,
    );
}

/// `y = alpha * op(a) x + beta * y`, `a` stored `a_rows x a_cols` with columns `lda` apart, `op` its transpose
/// flag.
#[allow(clippy::too_many_arguments)]
pub(super) fn matrix_vector_multiplication_ld<X, Y>(
    a: &[f64],
    a_rows: i32,
    a_cols: i32,
    lda: i32,
    trans_a: CblasTranspose,
    x: &X,
    y: &mut Y,
    alpha: f64,
    beta: f64,
) where
    X: crate::vector::Vector + ?Sized,
    Y: crate::vector::VectorMut + ?Sized,
{
    debug_assert!(lda >= a_rows);
    let (x, y) = (x.vector_operand(), y.vector_operand_mut());
    dgemv(
        CblasColMajor, trans_a, a_rows, a_cols, alpha, a.as_ptr(), lda.max(1),
        x.data.as_ptr(), x.step as i32, beta, y.data.as_mut_ptr(), y.step as i32,
    );
}
