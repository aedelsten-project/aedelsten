// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! LAPACK wrappers: inverse, least squares, eigensystems, SVD (full, and thin into caller storage).

use super::*;

/// A singular value decomposition `A = U diag(s) V^T`.
pub struct SVD {
    pub(super) u: Matrix<f64>,
    pub(super) s: Box<[f64]>,
    pub(super) vt: Matrix<f64>,
}

impl SVD {
    /// `X` of `A X = other`: `V diag(1/s) U^T other`.
    #[inline(always)]
    pub fn backslash<D>(&self, other: &D) -> Matrix<f64>
    where
        D: MatrixTrait<f64>,
    {
        let other = MatrixView::new(other.rows(), other.cols(), other.get_data());
        let y = self.u.view_transposed().matmul(&other);
        let mut z = Matrix::<f64>::new(self.vt.rows(), self.u.rows());
        for i in 0..self.s.len() {
            z[(i, i)] = 1.0 / self.s[i];
        }
        let z = z.matmul(&y);
        self.vt.view_transposed().matmul(&z)
    }

    /// `x` of `A x = other`: `V diag(1/s) U^T other`, the singular values below 1e-10 dropped.
    #[inline(always)]
    pub fn backslash_col(&self, other: &[f64]) -> Box<[f64]> {
        let y = self.u.view_transposed().matvec(other);
        let mut z = vec![0.0; self.s.len()];
        let eps = 1e-10;
        for (zi, (si, yi)) in z.iter_mut().zip(self.s.iter().zip(y.iter())) {
            *zi = if si.abs() > eps { yi / si } else { 0.0 };
        }
        self.vt.view_transposed().matvec(&z)
    }

}


// In-place LU inversion with caller-provided scratch (`work`: at least m*n values, `pivot`: at least m).
// Returns LAPACK's `info`: 0 on success, nonzero if the matrix is singular (`a` is then unspecified).
#[inline(always)]
pub(super) fn invert_matrix_prealloc(a: &mut [f64], m: usize, n: usize, work: &mut [f64], pivot: &mut [i32]) -> i32 {
    assert!(work.len() >= m * n && pivot.len() >= m, "invert scratch is too small");
    let a_ptr = a.as_ptr();
    let work_ptr = work.as_ptr();
    let ipiv = pivot.as_ptr();
    let lwork = (m * n) as i32;
    let m = m as i32;
    let n = n as i32;
    let lda = m;
    let mut info = 0;
    dgetrf(&m, &n, a_ptr, &lda, ipiv, &mut info);
    if info != 0 {
        return info;
    }
    dgetri(&n, a_ptr, &lda, ipiv, work_ptr, &lwork, &mut info);
    info
}

/// The least-squares solution of `op(A) X = B` by dgels (`a` m x n, `b` k x nrhs, both overwritten: `b` with the
/// solution). Panics if LAPACK reports a failure.
#[inline(always)]
pub(super) fn least_square_fit(
    a: &mut [f64],
    b: &mut [f64],
    m: i32,
    n: i32,
    k: i32,
    nrhs: i32,
    cblas_trans: CblasTranspose,
) {
    let lda = m;
    let ldb = k;
    let mut info = 0;
    // A scratch buffer query.
    let mut query = [0.0f64; 1];
    dgels(cblas_trans, &m, &n, &nrhs, a.as_mut_ptr(), &lda, b.as_mut_ptr(), &ldb, query.as_mut_ptr(), &-1, &mut info);
    assert_eq!(info, 0, "least_square_fit: LAPACK's dgels workspace query failed (info {info})");
    let lwork = (query[0] as i32).max(1);
    let mut work = vec![0.0; lwork as usize];
    dgels(cblas_trans, &m, &n, &nrhs, a.as_mut_ptr(), &lda, b.as_mut_ptr(), &ldb, work.as_mut_ptr(), &lwork, &mut info);
    assert_eq!(info, 0, "least_square_fit: LAPACK's dgels failed (info {info}: a negative value is a bad argument, a positive one a matrix not of full rank)");
}


/// Doubles of `work` that `MatrixMath::eigenvectors_prealloc` needs for an n x n matrix: LAPACK's
/// optimal scratch for dsyev (a scratch buffer query), and never less than its minimum 3n - 1.
pub fn eigenvectors_work_len(n: usize) -> usize {
    let (n_i, lwork) = (n as i32, -1);
    let (a, w, mut work, mut info) = ([0.0f64; 1], [0.0f64; 1], [0.0f64; 1], 0);
    dsyev('V', &CblasUpper, &n_i, a.as_ptr(), &n_i.max(1), w.as_ptr(), work.as_mut_ptr(), &lwork, &mut info);
    let optimal = if info == 0 { work[0] as usize } else { 0 };
    optimal.max((3 * n).saturating_sub(1)).max(1)
}

/// Doubles of `work` that `SplitComplexStruct::eigenvectors_prealloc` (and `eigenvalues_prealloc`) need for an n x n
/// Hermitian matrix: the interleaved copy that zheev works on (`2 n^2`), its real scratch (`3n - 2`), and its complex
/// scratch at LAPACK's optimal size (a scratch buffer query), never less than its minimum `2n - 1` complex elements.
pub fn hermitian_eigenvectors_work_len(n: usize) -> usize {
    let (n_i, lwork) = (n as i32, -1);
    let (mut a, mut w, mut work, mut rwork, mut info) = ([0.0f64; 2], [0.0f64; 1], [0.0f64; 2], [0.0f64; 1], 0);
    zheev('V', &CblasUpper, &n_i, a.as_mut_ptr(), &n_i.max(1), w.as_mut_ptr(), work.as_mut_ptr(), &lwork, rwork.as_mut_ptr(), &mut info);
    let optimal = if info == 0 { work[0] as usize } else { 0 };
    2 * n * n + hermitian_rwork_len(n) + 2 * optimal.max((2 * n).saturating_sub(1)).max(1)
}

/// zheev's real scratch for an n x n matrix: `max(1, 3n - 2)` doubles.
pub(super) fn hermitian_rwork_len(n: usize) -> usize {
    (3 * n).saturating_sub(2).max(1)
}

/// The full SVD (`jobu = jobvt = 'A'`) of the m x n `a` (overwritten): `u` m x m, `s` min(m, n), `vt` n x n.
#[inline(always)]
pub(super) fn svd(a: &mut [f64], m: usize, n: usize, u: &mut [f64], s: &mut [f64], vt: &mut [f64]) {
    let (m_i, n_i) = (m as i32, n as i32);
    let mut info = 0;
    // First call, to get the work size.
    let mut query = [0.0f64; 1];
    dgesvd('A', 'A', &m_i, &n_i, a.as_mut_ptr(), &m_i, s.as_mut_ptr(), u.as_mut_ptr(), &m_i, vt.as_mut_ptr(), &n_i, query.as_mut_ptr(), &-1, &mut info);
    let lwork = query[0] as i32;
    let mut work = vec![0.0; lwork as usize];
    dgesvd('A', 'A', &m_i, &n_i, a.as_mut_ptr(), &m_i, s.as_mut_ptr(), u.as_mut_ptr(), &m_i, vt.as_mut_ptr(), &n_i, work.as_mut_ptr(), &lwork, &mut info);
}

/// Doubles of `work` that `singular_value_decomposition_prealloc` needs for an m x n matrix: the copy dgesvd overwrites
/// (`m n`) and its scratch for the thin decomposition at LAPACK's optimal size (a scratch buffer query), never less
/// than its minimum `max(3 min(m, n) + max(m, n), 5 min(m, n))`.
pub fn svd_work_len(m: usize, n: usize) -> usize {
    let (m_i, n_i, k_i) = (m as i32, n as i32, m.min(n).max(1) as i32);
    let (mut a, mut s, mut u, mut vt, mut query, mut info) = ([0.0f64; 1], [0.0f64; 1], [0.0f64; 1], [0.0f64; 1], [0.0f64; 1], 0);
    dgesvd('S', 'S', &m_i, &n_i, a.as_mut_ptr(), &m_i.max(1), s.as_mut_ptr(), u.as_mut_ptr(), &m_i.max(1), vt.as_mut_ptr(), &k_i, query.as_mut_ptr(), &-1, &mut info);
    let optimal = if info == 0 { query[0] as usize } else { 0 };
    let minimum = (3 * m.min(n) + m.max(n)).max(5 * m.min(n)).max(1);
    m * n + optimal.max(minimum)
}

/// The `work` length `sygv_prealloc` needs for n x n matrices (LAPACK's optimum, from a scratch buffer query).
pub fn sygv_work_len(n: usize) -> usize {
    let (n_i, lwork, itype) = (n as i32, -1, 1);
    let (a, b, w, mut work, mut info) = ([0.0f64; 1], [0.0f64; 1], [0.0f64; 1], [0.0f64; 1], 0);
    dsygv(&itype, 'V', &CblasLower, &n_i, a.as_ptr(), &n_i.max(1), b.as_ptr(), &n_i.max(1), w.as_ptr(), work.as_mut_ptr(), &lwork, &mut info);
    let optimal = if info == 0 { work[0] as usize } else { 0 };
    optimal.max((3 * n).saturating_sub(1)).max(1)
}
