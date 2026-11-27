// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! `MatrixMath` / `MatrixMathMut`: the linear algebra on the matrix types (f64).

use super::*;

/// The linear algebra on a matrix, owned or a view, in the three forms: on the heap, into caller storage
/// (`_prealloc`, the one doing the work) and from an arena (`_arena`). An `_arena` form that needs scratch takes
/// `&mut Arena`: the scratch lives in a nested scope, only the result stays allocated.
pub trait MatrixMath<T>: MatrixTrait<T>
where
    T: Default + Clone + Copy,
{
    /// `other^T self other`.
    fn congruence_transform<D1>(&self, other: &D1) -> Matrix<T>
    where
        D1: MatrixTrait<T>;
    /// `other^T self other` into `output`; `product_scratch` (`self.rows() x other.cols()`) takes `self other`.
    fn congruence_transform_prealloc<D1, D2, D3>(
        &self,
        other: &D1,
        product_scratch: &mut D2,
        output: &mut D3,
    ) where
        D1: MatrixTrait<T>,
        D2: MatrixTraitMut<T>,
        D3: MatrixTraitMut<T>;
    /// `congruence_transform`, from the arena.
    fn congruence_transform_arena<'a, D1>(&self, other: &D1, arena: &mut Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>
    where
        D1: MatrixTrait<T>;
    /// `sum_ij self_ij other_ij`.
    fn frobenius_inner<D1>(&self, other: &D1) -> T
    where
        D1: MatrixTrait<T>;
    /// `alpha (self + self^T)`, square.
    fn symmetrize(&self, alpha: f64) -> Matrix<T>;
    /// `symmetrize` into `output`.
    fn symmetrize_prealloc<D1>(&self, output: &mut D1, alpha: f64)
    where
        D1: MatrixTraitMut<T>;
    /// `symmetrize`, from the arena.
    fn symmetrize_arena<'a>(&self, alpha: f64, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>;
    /// `alpha self^T self`, `cols x cols`; `self` need not be square.
    fn rank_k_update(&self, alpha: f64) -> Matrix<T>;
    /// `rank_k_update` into `output`.
    fn rank_k_update_prealloc<D1>(&self, output: &mut D1, alpha: f64)
    where
        D1: MatrixTraitMut<T>;
    /// `rank_k_update`, from the arena.
    fn rank_k_update_arena<'a>(&self, alpha: f64, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>;
    /// `self^-1` by LU factorization; unspecified for a singular matrix (`invert_prealloc` reports it).
    fn invert(&self) -> Matrix<T>;
    /// `output = self^-1` without allocating: `work` needs rows*cols values and `pivot` `rows` values.
    /// Returns false if the matrix is singular (the output is then unspecified).
    fn invert_prealloc<D1>(&self, output: &mut D1, work: &mut [T], pivot: &mut [i32]) -> bool
    where
        D1: MatrixTraitMut<T>;
    /// `invert`, from the arena.
    fn invert_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>;
    /// `self^-1/2` of a symmetric matrix, from its eigendecomposition. The `_prealloc` forms of this and the
    /// following take the scratch of `eigenvectors_prealloc` (`values`, `vectors`, `work`) and return false if
    /// LAPACK fails or, for the square roots, an eigenvalue is below 1e-14 in magnitude; the heap and arena forms
    /// panic then.
    fn symmetric_inverse_sqrt(&self) -> Matrix<T>;
    /// `symmetric_inverse_sqrt` into `output`.
    fn symmetric_inverse_sqrt_prealloc<D1, D2>(&self, output: &mut D1, values: &mut [T], vectors: &mut D2, work: &mut [T]) -> bool
    where
        D1: MatrixTraitMut<T>,
        D2: MatrixTraitMut<T>;
    /// `symmetric_inverse_sqrt`, from the arena.
    fn symmetric_inverse_sqrt_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>;
    /// `self^1/2` and `self^-1/2` of a symmetric matrix, from one eigendecomposition.
    fn symmetric_sqrt_and_inverse_sqrt(&self) -> (Matrix<T>, Matrix<T>);
    /// `symmetric_sqrt_and_inverse_sqrt` into `sqrt` and `inverse_sqrt`.
    fn symmetric_sqrt_and_inverse_sqrt_prealloc<D1, D2, D3>(
        &self,
        sqrt: &mut D1,
        inverse_sqrt: &mut D2,
        values: &mut [T],
        vectors: &mut D3,
        work: &mut [T],
    ) -> bool
    where
        D1: MatrixTraitMut<T>,
        D2: MatrixTraitMut<T>,
        D3: MatrixTraitMut<T>;
    /// `symmetric_sqrt_and_inverse_sqrt`, from the arena.
    fn symmetric_sqrt_and_inverse_sqrt_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<(MatrixViewMut<'a, T>, MatrixViewMut<'a, T>), ArenaError>;
    /// The pseudo-inverse of a symmetric matrix by canonical orthogonalization: only the eigenvalues above
    /// `relative_threshold` times the largest are inverted, the other directions (negative eigenvalues among them)
    /// are dropped. Built as `M M^T`, so exactly symmetric however ill-conditioned the input.
    fn symmetric_pseudo_inverse(&self, relative_threshold: T) -> Matrix<T>;
    /// `symmetric_pseudo_inverse` into `output`.
    fn symmetric_pseudo_inverse_prealloc<D1, D2>(
        &self,
        relative_threshold: T,
        output: &mut D1,
        values: &mut [T],
        vectors: &mut D2,
        work: &mut [T],
    ) -> bool
    where
        D1: MatrixTraitMut<T>,
        D2: MatrixTraitMut<T>;
    /// `symmetric_pseudo_inverse`, from the arena.
    fn symmetric_pseudo_inverse_arena<'a>(&self, relative_threshold: T, arena: &mut Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>;
    /// The full singular value decomposition `self = U diag(s) V^T`: `U` m x m, `V^T` n x n.
    fn singular_value_decomposition(&self) -> SVD;
    /// The thin singular value decomposition `self = U diag(s) V^T` (m x n, k = min(m, n)) into caller storage: the
    /// singular values into `values[..k]`, descending; `u` m x k, `vt` k x n. `work` holds at least
    /// `svd_work_len(m, n)` values (the copy LAPACK overwrites and its scratch). False when LAPACK fails (its
    /// iteration did not converge).
    fn singular_value_decomposition_prealloc<D1, D2>(&self, values: &mut [T], u: &mut D1, vt: &mut D2, work: &mut [T]) -> bool
    where
        D1: MatrixTraitMut<T>,
        D2: MatrixTraitMut<T>;
    /// `sqrt(sum_ij self_ij^2)`.
    fn frobenius_norm(&self) -> T;
    /// The least-squares solution `x` of `self x = other` (LAPACK's dgels), `cols` values: for more rows than columns
    /// the least-squares fit, for fewer the minimum-norm solution. Panics when `self` is not of full rank.
    fn backslash(&self, other: &[T]) -> Box<[T]>;
    /// The eigenvalues (ascending) of a symmetric matrix.
    fn eigenvalues(&self) -> Box<[T]>;
    /// `eigenvalues` into `values`; `scratch` is n x n, `work` as for `eigenvectors_prealloc`. False if LAPACK fails.
    fn eigenvalues_prealloc<D1>(&self, values: &mut [T], scratch: &mut D1, work: &mut [T]) -> bool
    where
        D1: MatrixTraitMut<T>;
    /// `eigenvalues`, from the arena.
    fn eigenvalues_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<&'a mut [T], ArenaError>;
    /// The eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix.
    fn eigenvectors(&self) -> (Box<[T]>, Matrix<T>);
    /// `eigenvectors`, from the arena.
    fn eigenvectors_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<(&'a mut [T], MatrixViewMut<'a, T>), ArenaError>;
    /// Eigenvalues (ascending) into `values` (n of them) and eigenvectors (columns) into `vectors`
    /// (n x n) of this symmetric matrix, without allocating: `work` holds at least
    /// `eigenvectors_work_len(n)` values. Returns false if LAPACK reports a failure (the outputs are
    /// then unspecified).
    fn eigenvectors_prealloc<D1>(&self, values: &mut [T], vectors: &mut D1, work: &mut [T]) -> bool
    where
        D1: MatrixTraitMut<T>;
    /// `self + other`.
    #[must_use = "`add` returns the sum and leaves `self` unchanged; to add in place, use `add_in_place`"]
    fn add<D1>(&self, other: &D1) -> Matrix<T>
    where
        D1: MatrixTrait<T>;
    /// `self + other` into `output`.
    fn add_prealloc<D1, D2>(&self, other: &D1, output: &mut D2)
    where
        D1: MatrixTrait<T>,
        D2: MatrixTraitMut<T>;
    /// `self + other`, from the arena.
    fn add_arena<'a, D1>(&self, other: &D1, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>
    where
        D1: MatrixTrait<T>;
    /// The generalized symmetric eigenproblem of `self` with `other` (LAPACK's dsygv, problem type `itype` 1-3):
    /// eigenvalues and eigenvectors.
    fn sygv<D1>(&self, other: &D1, itype: usize) -> (Box<[T]>, Matrix<T>)
    where
        D1: MatrixTrait<T>;
    /// `sygv` into `values` and `vectors`: `b_scratch` is n x n, `work` holds at least `sygv_work_len(n)` values.
    /// False if LAPACK fails.
    fn sygv_prealloc<D1, D2, D3>(
        &self,
        other: &D1,
        itype: usize,
        values: &mut [T],
        vectors: &mut D2,
        b_scratch: &mut D3,
        work: &mut [T],
    ) -> bool
    where
        D1: MatrixTrait<T>,
        D2: MatrixTraitMut<T>,
        D3: MatrixTraitMut<T>;
    /// `sygv`, from the arena.
    fn sygv_arena<'a, D1>(&self, other: &D1, itype: usize, arena: &mut Arena<'a>) -> Result<(&'a mut [T], MatrixViewMut<'a, T>), ArenaError>
    where
        D1: MatrixTrait<T>;
}

/// The linear algebra that changes a matrix in place, owned or a mutable view.
pub trait MatrixMathMut<T>: MatrixMath<T> + MatrixTraitMut<T>
where
    T: Default + Clone + Copy,
{
    /// `self += other`.
    fn add_in_place<D1>(&mut self, other: &D1)
    where
        D1: MatrixTrait<T>;
    /// `self += scale other`.
    fn add_scaled_in_place<D1>(&mut self, other: &D1, scale: T)
    where
        D1: MatrixTrait<T>;
    /// `self *= scale`.
    fn scale_in_place(&mut self, scale: T);
    /// `self = alpha (self + self^T)`, in place (`symmetrize` without the output buffer). Square only.
    fn symmetrize_in_place(&mut self, alpha: T);
    /// Part `part` of `parts` of `symmetrize_in_place`: the parts write disjoint elements, cost about the same, and
    /// together give bitwise `symmetrize_in_place`'s result, so that `parts` workers (processes sharing the matrix's
    /// memory) can each do one at the same time. Square only.
    fn symmetrize_in_place_part(&mut self, alpha: T, part: usize, parts: usize);
}

impl<D> MatrixMath<f64> for MatrixStruct<D>
where
    D: DataContainer<T = f64>,
{
    #[inline(always)]
    fn congruence_transform<D1>(&self, other: &D1) -> Matrix<f64>
    where
        D1: MatrixTrait<f64>,
    {
        debug_assert_eq!(self.cols(), other.rows());
        let mut intermediate = Matrix::<f64>::new(self.rows(), other.cols());
        let mut output = Matrix::<f64>::new(other.cols(), intermediate.cols());
        self.congruence_transform_prealloc(other, &mut intermediate, &mut output);
        output
    }

    #[inline(always)]
    fn congruence_transform_prealloc<D1, D2, D3>(
        &self,
        other: &D1,
        product_scratch: &mut D2,
        output: &mut D3,
    ) where
        D1: MatrixTrait<f64>,
        D2: MatrixTraitMut<f64>,
        D3: MatrixTraitMut<f64>,
    {
        debug_assert_eq!(self.cols(), other.rows());
        debug_assert_eq!(product_scratch.rows(), self.rows());
        debug_assert_eq!(product_scratch.cols(), other.cols());
        debug_assert_eq!(output.rows(), other.cols());
        debug_assert_eq!(output.cols(), other.cols());
        matrix_matrix_multiplication(
            self.get_data(), self.rows() as i32, self.cols() as i32, CblasNoTrans,
            other.get_data(), other.rows() as i32, other.cols() as i32, CblasNoTrans,
            product_scratch.get_data_mut(), 1.0, 0.0,
        );
        matrix_matrix_multiplication(
            &other.get_data(), other.rows() as i32, other.cols() as i32, CblasTrans,
            &product_scratch.get_data(), product_scratch.rows() as i32, product_scratch.cols() as i32, CblasNoTrans,
            output.get_data_mut(), 1.0, 0.0
        );
    }

    fn congruence_transform_arena<'a, D1>(&self, other: &D1, arena: &mut Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError>
    where
        D1: MatrixTrait<f64>,
    {
        let mut output = MatrixViewMut::<f64>::from_arena(other.cols(), other.cols(), arena)?;
        arena.scope(|scope| {
            let mut product = MatrixViewMut::<f64>::from_arena(self.rows(), other.cols(), scope)?;
            self.congruence_transform_prealloc(other, &mut product, &mut output);
            Ok::<(), ArenaError>(())
        })?;
        Ok(output)
    }

    #[inline(always)]
    fn symmetrize(&self, alpha: f64) -> Matrix<f64> {
        debug_assert_eq!(self.cols(), self.rows());
        let mut output = Matrix::<f64>::new(self.rows(), self.cols());
        self.symmetrize_prealloc(&mut output, alpha);
        output
    }

    #[inline(always)]
    fn symmetrize_prealloc<D1>(&self, output: &mut D1, alpha: f64)
    where
        D1: MatrixTraitMut<f64>
    {
        debug_assert_eq!(self.cols(), self.rows());
        debug_assert_eq!(output.cols(), output.rows());
        debug_assert_eq!(self.cols(), output.cols());
        for (j, (dst_col_j, src_col_j)) in output.iter_col_mut().zip(self.iter()).enumerate() {
            for (i, (dst, src)) in dst_col_j.iter_mut().zip(src_col_j.iter()).skip(j).enumerate() {
                let src_t = self[(j, j+i)];
                let val = alpha * (src + src_t);
                *dst = val;
            }
        }
        output.copy_lower_to_upper()
    }

    fn symmetrize_arena<'a>(&self, alpha: f64, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError> {
        let mut output = MatrixViewMut::<f64>::from_arena(self.rows(), self.cols(), arena)?;
        self.symmetrize_prealloc(&mut output, alpha);
        Ok(output)
    }

    #[inline(always)]
    fn rank_k_update(&self, alpha: f64) -> Matrix<f64> {
        let mut output = Matrix::<f64>::new(self.cols(), self.cols());
        self.rank_k_update_prealloc(&mut output, alpha);
        output
    }

    #[inline(always)]
    fn rank_k_update_prealloc<D1>(&self, output: &mut D1, alpha: f64)
    where
        D1: MatrixTraitMut<f64>
    {
        debug_assert_eq!(output.cols(), output.rows());
        debug_assert_eq!(self.cols(), output.cols());
        matrix_matrix_multiplication(
            &self.get_data(),
            self.rows() as i32,
            self.cols() as i32,
            CblasTrans,
            &self.get_data(),
            self.rows() as i32,
            self.cols() as i32,
            CblasNoTrans,
            output.get_data_mut(),
            alpha, 0.0
        );
    }

    fn rank_k_update_arena<'a>(&self, alpha: f64, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError> {
        let mut output = MatrixViewMut::<f64>::from_arena(self.cols(), self.cols(), arena)?;
        self.rank_k_update_prealloc(&mut output, alpha);
        Ok(output)
    }

    fn invert(&self) -> Matrix<f64> {
        let mut output = Matrix::<f64>::new(self.cols(), self.rows());
        let (mut work, mut pivot) = (vec![0.0; self.rows() * self.cols()], vec![0; self.rows()]);
        self.invert_prealloc(&mut output, &mut work, &mut pivot);
        output
    }

    #[inline(always)]
    fn invert_prealloc<D1>(&self, output: &mut D1, work: &mut [f64], pivot: &mut [i32]) -> bool
    where
        D1: MatrixTraitMut<f64>,
    {
        debug_assert_eq!(output.rows(), self.cols());
        debug_assert_eq!(output.cols(), self.rows());
        output.get_data_mut().copy_from_slice(self.get_data());
        invert_matrix_prealloc(output.get_data_mut(), self.rows(), self.cols(), work, pivot) == 0
    }

    fn invert_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError> {
        let mut output = MatrixViewMut::<f64>::from_arena(self.cols(), self.rows(), arena)?;
        arena.scope(|scope| {
            let (work, pivot) = (scope.alloc::<f64>(self.rows() * self.cols())?, scope.alloc_zeroed::<i32>(self.rows())?);
            self.invert_prealloc(&mut output, work, pivot);
            Ok::<(), ArenaError>(())
        })?;
        Ok(output)
    }

    fn symmetric_inverse_sqrt(&self) -> Matrix<f64> {
        let n = self.rows();
        let mut output = Matrix::<f64>::new(n, n);
        let (mut values, mut vectors, mut work) = eigen_scratch(n);
        assert!(
            self.symmetric_inverse_sqrt_prealloc(&mut output, &mut values, &mut vectors, &mut work),
            "symmetric_inverse_sqrt: the matrix is singular (or LAPACK failed)"
        );
        output
    }

    fn symmetric_inverse_sqrt_prealloc<D1, D2>(&self, output: &mut D1, values: &mut [f64], vectors: &mut D2, work: &mut [f64]) -> bool
    where
        D1: MatrixTraitMut<f64>,
        D2: MatrixTraitMut<f64>,
    {
        if !self.eigenvectors_prealloc(values, vectors, work) || values.iter().any(|v| v.abs() < 1e-14) {
            return false;
        }
        // S^-1/2 = (V L^-1/4) (V L^-1/4)^T
        scaled_outer_product(values, vectors, output, |v| 1.0 / v.sqrt().sqrt());
        true
    }

    fn symmetric_inverse_sqrt_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError> {
        let n = self.rows();
        let mut output = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let ok = arena.scope(|scope| {
            let (values, mut vectors, work) = eigen_scratch_arena(n, scope)?;
            Ok::<bool, ArenaError>(self.symmetric_inverse_sqrt_prealloc(&mut output, values, &mut vectors, work))
        })?;
        assert!(ok, "symmetric_inverse_sqrt_arena: the matrix is singular (or LAPACK failed)");
        Ok(output)
    }

    fn symmetric_sqrt_and_inverse_sqrt(&self) -> (Matrix<f64>, Matrix<f64>) {
        let n = self.rows();
        let (mut sqrt, mut inverse_sqrt) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
        let (mut values, mut vectors, mut work) = eigen_scratch(n);
        assert!(
            self.symmetric_sqrt_and_inverse_sqrt_prealloc(&mut sqrt, &mut inverse_sqrt, &mut values, &mut vectors, &mut work),
            "symmetric_sqrt_and_inverse_sqrt: the matrix is singular (or LAPACK failed)"
        );
        (sqrt, inverse_sqrt)
    }

    fn symmetric_sqrt_and_inverse_sqrt_prealloc<D1, D2, D3>(
        &self,
        sqrt: &mut D1,
        inverse_sqrt: &mut D2,
        values: &mut [f64],
        vectors: &mut D3,
        work: &mut [f64],
    ) -> bool
    where
        D1: MatrixTraitMut<f64>,
        D2: MatrixTraitMut<f64>,
        D3: MatrixTraitMut<f64>,
    {
        if !self.eigenvectors_prealloc(values, vectors, work) || values.iter().any(|v| v.abs() < 1e-14) {
            return false;
        }
        // The eigenvectors are scaled by L^-1/4 for S^-1/2, then by L^1/2 more (L^1/4 in all) for S^1/2.
        scaled_outer_product(values, vectors, inverse_sqrt, |v| 1.0 / v.sqrt().sqrt());
        scaled_outer_product(values, vectors, sqrt, |v| v.sqrt());
        true
    }

    fn symmetric_sqrt_and_inverse_sqrt_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<(MatrixViewMut<'a, f64>, MatrixViewMut<'a, f64>), ArenaError> {
        let n = self.rows();
        let mut sqrt = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let mut inverse_sqrt = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let ok = arena.scope(|scope| {
            let (values, mut vectors, work) = eigen_scratch_arena(n, scope)?;
            Ok::<bool, ArenaError>(self.symmetric_sqrt_and_inverse_sqrt_prealloc(&mut sqrt, &mut inverse_sqrt, values, &mut vectors, work))
        })?;
        assert!(ok, "symmetric_sqrt_and_inverse_sqrt_arena: the matrix is singular (or LAPACK failed)");
        Ok((sqrt, inverse_sqrt))
    }

    fn symmetric_pseudo_inverse(&self, relative_threshold: f64) -> Matrix<f64> {
        let n = self.rows();
        let mut output = Matrix::<f64>::new(n, n);
        let (mut values, mut vectors, mut work) = eigen_scratch(n);
        assert!(
            self.symmetric_pseudo_inverse_prealloc(relative_threshold, &mut output, &mut values, &mut vectors, &mut work),
            "symmetric_pseudo_inverse: LAPACK failed"
        );
        output
    }

    fn symmetric_pseudo_inverse_prealloc<D1, D2>(
        &self,
        relative_threshold: f64,
        output: &mut D1,
        values: &mut [f64],
        vectors: &mut D2,
        work: &mut [f64],
    ) -> bool
    where
        D1: MatrixTraitMut<f64>,
        D2: MatrixTraitMut<f64>,
    {
        if !self.eigenvectors_prealloc(values, vectors, work) {
            return false;
        }
        let cutoff = relative_threshold * values.iter().cloned().fold(0.0_f64, f64::max);
        scaled_outer_product(values, vectors, output, |v| if v > cutoff { 1.0 / v.sqrt() } else { 0.0 });
        true
    }

    fn symmetric_pseudo_inverse_arena<'a>(&self, relative_threshold: f64, arena: &mut Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError> {
        let n = self.rows();
        let mut output = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let ok = arena.scope(|scope| {
            let (values, mut vectors, work) = eigen_scratch_arena(n, scope)?;
            Ok::<bool, ArenaError>(self.symmetric_pseudo_inverse_prealloc(relative_threshold, &mut output, values, &mut vectors, work))
        })?;
        assert!(ok, "symmetric_pseudo_inverse_arena: LAPACK failed");
        Ok(output)
    }

    #[inline(always)]
    fn singular_value_decomposition(&self) -> SVD {
        // dgesvd overwrites its input: it works on a copy.
        let a = self.get_data();
        let mut a = a.to_vec().into_boxed_slice();
        let m = self.rows();
        let n = self.cols();
        let k = n.min(m);
        let mut u = Matrix::<f64>::new(m, m);
        let u_ptr = u.get_data_mut();
        let mut vt = Matrix::<f64>::new(n, n);
        let vt_ptr = vt.get_data_mut();
        let mut s = vec![0.0; k].into_boxed_slice();
        let s_ptr = s.as_mut();
        svd(&mut a, m, n, u_ptr, s_ptr, vt_ptr);
        SVD { u, s, vt }
    }

    fn singular_value_decomposition_prealloc<D1, D2>(&self, values: &mut [f64], u: &mut D1, vt: &mut D2, work: &mut [f64]) -> bool
    where
        D1: MatrixTraitMut<f64>,
        D2: MatrixTraitMut<f64>,
    {
        let (m, n) = (self.rows(), self.cols());
        let k = m.min(n);
        assert_eq!((u.rows(), u.cols()), (m, k), "singular_value_decomposition_prealloc: U is not {m} x {k}");
        assert_eq!((vt.rows(), vt.cols()), (k, n), "singular_value_decomposition_prealloc: V^T is not {k} x {n}");
        assert!(values.len() >= k, "singular_value_decomposition_prealloc: room for {} of {k} singular values", values.len());
        assert!(work.len() > m * n, "singular_value_decomposition_prealloc: work of {} values, svd_work_len({m}, {n}) needed", work.len());
        if k == 0 {
            return true;
        }
        // dgesvd overwrites its input: it works on a copy at the front of `work`, with the rest as its scratch.
        let (copy, rest) = work.split_at_mut(m * n);
        copy.copy_from_slice(self.get_data());
        let (m_i, n_i, k_i, lwork) = (m as i32, n as i32, k as i32, rest.len() as i32);
        let mut info = 0;
        dgesvd(
            'S', 'S', &m_i, &n_i, copy.as_mut_ptr(), &m_i, values.as_mut_ptr(), u.get_data_mut().as_mut_ptr(), &m_i,
            vt.get_data_mut().as_mut_ptr(), &k_i, rest.as_mut_ptr(), &lwork, &mut info,
        );
        info == 0
    }

    fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        let data = self.get_data();
        for val in data.iter() {
            sum += val * val;
        }
        sum.sqrt()
    }

    #[inline(always)]
    fn backslash(&self, other: &[f64]) -> Box<[f64]> {
        // dgels overwrites its input: it works on a copy.
        let a = self.get_data();
        let mut a = a.to_vec().into_boxed_slice();
        let m = self.rows() as i32;
        let n = self.cols() as i32;
        // b padded to max(m, n) rows, which dgels overwrites with the solution.
        let ldb = m.max(n);
        let k = other.len() as i32;
        let l = 1;
        let mut b = Matrix::<f64>::new(ldb as usize, l as usize);
        for col in b.iter_col_mut() {
            for (row1, row2) in col.iter_mut().take(k as usize).zip(other.iter()) {
                *row1 = *row2;
            }
        }

        let k = b.rows() as i32;
        let nrhs = l;
        let mut b = b.get_data_mut();
        least_square_fit(&mut a, &mut b, m, n, k, nrhs, CblasNoTrans);

        b[..(n as usize)].to_vec().into_boxed_slice()
    }

    fn eigenvalues(&self) -> Box<[f64]> {
        let n = self.rows();
        let (mut values, mut scratch, mut work) = eigen_scratch(n);
        assert!(self.eigenvalues_prealloc(&mut values, &mut scratch, &mut work), "eigenvalues: LAPACK failed");
        values.into_boxed_slice()
    }

    fn eigenvalues_prealloc<D1>(&self, values: &mut [f64], scratch: &mut D1, work: &mut [f64]) -> bool
    where
        D1: MatrixTraitMut<f64>,
    {
        let n = self.rows();
        assert_eq!(self.cols(), n, "eigenvalues_prealloc: the matrix is not square");
        assert_eq!((scratch.rows(), scratch.cols()), (n, n), "eigenvalues_prealloc: the scratch matrix is not n x n");
        assert!(values.len() >= n, "eigenvalues_prealloc: room for {} of {n} eigenvalues", values.len());
        if n == 0 {
            return true;
        }
        // dsyev destroys its input: it works on the scratch copy.
        scratch.get_data_mut().copy_from_slice(self.get_data());
        let (n_i, lwork) = (n as i32, work.len() as i32);
        let mut info = 0;
        dsyev('N', &CblasUpper, &n_i, scratch.get_data_mut().as_mut_ptr(), &n_i, values.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info);
        info == 0
    }

    fn eigenvalues_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<&'a mut [f64], ArenaError> {
        let n = self.rows();
        let values = arena.alloc_zeroed::<f64>(n)?;
        let ok = arena.scope(|scope| {
            let mut scratch = MatrixViewMut::<f64>::from_arena(n, n, scope)?;
            let work = scope.alloc::<f64>(eigenvectors_work_len(n))?;
            Ok::<bool, ArenaError>(self.eigenvalues_prealloc(values, &mut scratch, work))
        })?;
        assert!(ok, "eigenvalues_arena: LAPACK failed");
        Ok(values)
    }

    fn eigenvectors(&self) -> (Box<[f64]>, Matrix<f64>) {
        let n = self.rows();
        let (mut values, mut vectors, mut work) = eigen_scratch(n);
        assert!(self.eigenvectors_prealloc(&mut values, &mut vectors, &mut work), "eigenvectors: LAPACK failed");
        (values.into_boxed_slice(), vectors)
    }

    fn eigenvectors_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<(&'a mut [f64], MatrixViewMut<'a, f64>), ArenaError> {
        let n = self.rows();
        let values = arena.alloc_zeroed::<f64>(n)?;
        let mut vectors = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let ok = arena.scope(|scope| {
            let work = scope.alloc::<f64>(eigenvectors_work_len(n))?;
            Ok::<bool, ArenaError>(self.eigenvectors_prealloc(values, &mut vectors, work))
        })?;
        assert!(ok, "eigenvectors_arena: LAPACK failed");
        Ok((values, vectors))
    }

    fn eigenvectors_prealloc<D1>(&self, values: &mut [f64], vectors: &mut D1, work: &mut [f64]) -> bool
    where
        D1: MatrixTraitMut<f64>,
    {
        let n = self.rows();
        assert_eq!(self.cols(), n, "eigenvectors_prealloc: the matrix is not square");
        assert_eq!((vectors.rows(), vectors.cols()), (n, n), "eigenvectors_prealloc: the eigenvector matrix is not n x n");
        assert!(values.len() >= n, "eigenvectors_prealloc: room for {} of {n} eigenvalues", values.len());
        if n == 0 {
            return true;
        }
        // dsyev overwrites its input with the eigenvectors, so it works on the output directly.
        vectors.get_data_mut().copy_from_slice(self.get_data());
        let (n_i, lwork) = (n as i32, work.len() as i32);
        let mut info = 0;
        dsyev('V', &CblasUpper, &n_i, vectors.get_data_mut().as_mut_ptr(), &n_i, values.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info);
        info == 0
    }

    #[inline(always)]
    fn add<D1>(&self, other: &D1) -> Matrix<f64>
    where
        D1: MatrixTrait<f64>,
    {
        let mut output = Matrix::<f64>::new(self.rows(), self.cols());
        self.add_prealloc(other, &mut output);
        output
    }

    #[inline(always)]
    fn add_prealloc<D1, D2>(&self, other: &D1, output: &mut D2)
    where
        D1: MatrixTrait<f64>,
        D2: MatrixTraitMut<f64>,
    {
        let shape = (self.rows(), self.cols());
        assert_eq!((other.rows(), other.cols()), shape, "add: the summands differ in shape");
        assert_eq!((output.rows(), output.cols()), shape, "add: the output differs in shape");
        for ((out, a), b) in output.get_data_mut().iter_mut().zip(self.get_data()).zip(other.get_data()) {
            *out = a + b;
        }
    }

    #[inline(always)]
    fn add_arena<'a, D1>(&self, other: &D1, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError>
    where
        D1: MatrixTrait<f64>,
    {
        let mut output = MatrixViewMut::<f64>::from_arena(self.rows(), self.cols(), arena)?;
        self.add_prealloc(other, &mut output);
        Ok(output)
    }

    #[inline(always)]
    fn frobenius_inner<D1>(&self, other: &D1) -> f64
    where
        D1: MatrixTrait<f64>,
    {
        let vec1 = self.get_data();
        let vec2 = other.get_data();
        dot(vec1, vec2)
    }

    fn sygv<D1>(&self, other: &D1, itype: usize) -> (Box<[f64]>, Matrix<f64>)
    where
        D1: MatrixTrait<f64>,
    {
        let n = self.rows();
        let (mut values, mut vectors, mut b_scratch) = (vec![0.0; n], Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
        let mut work = vec![0.0; sygv_work_len(n)];
        assert!(self.sygv_prealloc(other, itype, &mut values, &mut vectors, &mut b_scratch, &mut work), "sygv: LAPACK failed");
        (values.into_boxed_slice(), vectors)
    }

    fn sygv_prealloc<D1, D2, D3>(
        &self,
        other: &D1,
        itype: usize,
        values: &mut [f64],
        vectors: &mut D2,
        b_scratch: &mut D3,
        work: &mut [f64],
    ) -> bool
    where
        D1: MatrixTrait<f64>,
        D2: MatrixTraitMut<f64>,
        D3: MatrixTraitMut<f64>,
    {
        let n = self.rows();
        assert_eq!((self.cols(), other.rows(), other.cols()), (n, n, n), "sygv_prealloc: the matrices are not n x n");
        assert_eq!((vectors.rows(), vectors.cols(), b_scratch.rows(), b_scratch.cols()), (n, n, n, n), "sygv_prealloc: the outputs are not n x n");
        assert!(values.len() >= n, "sygv_prealloc: room for {} of {n} eigenvalues", values.len());
        if n == 0 {
            return true;
        }
        // dsygv overwrites A with the eigenvectors and B with its Cholesky factor.
        vectors.get_data_mut().copy_from_slice(self.get_data());
        b_scratch.get_data_mut().copy_from_slice(other.get_data());
        let (itype, n_i, lwork) = (itype as i32, n as i32, work.len() as i32);
        let mut info = 0;
        dsygv(&itype, 'V', &CblasLower, &n_i, vectors.get_data_mut().as_mut_ptr(), &n_i, b_scratch.get_data_mut().as_mut_ptr(), &n_i,
              values.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info);
        info == 0
    }

    fn sygv_arena<'a, D1>(&self, other: &D1, itype: usize, arena: &mut Arena<'a>) -> Result<(&'a mut [f64], MatrixViewMut<'a, f64>), ArenaError>
    where
        D1: MatrixTrait<f64>,
    {
        let n = self.rows();
        let values = arena.alloc_zeroed::<f64>(n)?;
        let mut vectors = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let ok = arena.scope(|scope| {
            let mut b_scratch = MatrixViewMut::<f64>::from_arena(n, n, scope)?;
            let work = scope.alloc::<f64>(sygv_work_len(n))?;
            Ok::<bool, ArenaError>(self.sygv_prealloc(other, itype, values, &mut vectors, &mut b_scratch, work))
        })?;
        assert!(ok, "sygv_arena: LAPACK failed");
        Ok((values, vectors))
    }
}

/// The scratch of `eigenvectors_prealloc` on the heap: values, vectors, work.
fn eigen_scratch(n: usize) -> (Vec<f64>, Matrix<f64>, Vec<f64>) {
    (vec![0.0; n], Matrix::<f64>::new(n, n), vec![0.0; eigenvectors_work_len(n)])
}

/// The scratch of `eigenvectors_prealloc` from an arena.
fn eigen_scratch_arena<'s>(n: usize, arena: &Arena<'s>) -> Result<(&'s mut [f64], MatrixViewMut<'s, f64>, &'s mut [f64]), ArenaError> {
    Ok((arena.alloc_zeroed::<f64>(n)?, MatrixViewMut::<f64>::from_arena(n, n, arena)?, arena.alloc::<f64>(eigenvectors_work_len(n))?))
}

/// `output = (V f(L)) (V f(L))^T`: scales column `i` of `vectors` by `f(values[i])` (in place), then the product.
fn scaled_outer_product<D1, D2>(values: &[f64], vectors: &mut D1, output: &mut D2, f: impl Fn(f64) -> f64)
where
    D1: MatrixTraitMut<f64>,
    D2: MatrixTraitMut<f64>,
{
    for (column, &value) in vectors.iter_col_mut().zip(values) {
        let scale = f(value);
        column.iter_mut().for_each(|x| *x *= scale);
    }
    let v = MatrixView::new(vectors.rows(), vectors.cols(), vectors.get_data());
    v.matmul_prealloc(&v.view_transposed(), output);
}

macro_rules! impl_matrix_math_mut {
    ($arr: ty) => {
impl MatrixMathMut<f64> for $arr {
    fn add_in_place<D1>(&mut self, other: &D1)
    where
        D1: MatrixTrait<f64>,
    {
        let vec1 = self.get_data_mut();
        let vec2 = other.get_data();
        add_in_place(vec1, vec2);
    }

    fn add_scaled_in_place<D1>(&mut self, other: &D1, scale: f64)
    where
        D1: MatrixTrait<f64>,
    {
        let vec1 = self.get_data_mut();
        let vec2 = other.get_data();
        add_scaled_in_place(vec1, vec2, scale);
    }

    #[inline(always)]
    fn scale_in_place(&mut self, scale: f64) {
        let vec1 = self.get_data_mut();
        scale_in_place(vec1, scale);
    }

    fn symmetrize_in_place(&mut self, alpha: f64) {
        assert_eq!(self.rows(), self.cols(), "symmetrize_in_place: the matrix is not square");
        let n = self.rows();
        symmetrize_square_in_place(self.get_data_mut(), n, alpha, 0, 1);
    }

    fn symmetrize_in_place_part(&mut self, alpha: f64, part: usize, parts: usize) {
        assert_eq!(self.rows(), self.cols(), "symmetrize_in_place_part: the matrix is not square");
        assert!(part < parts, "symmetrize_in_place_part: part {part} of {parts}");
        let n = self.rows();
        symmetrize_square_in_place(self.get_data_mut(), n, alpha, part, parts);
    }
}
    };
}

/// The tile edge of `symmetrize_square_in_place`: two tiles of `SYMMETRIZE_TILE^2` doubles (64 KB) stay in cache.
const SYMMETRIZE_TILE: usize = 64;

/// `a = alpha (a + a^T)` for the column-major `n x n` matrix in `a`, by tiles: for each tile pair `(I, J)`, `I >= J`,
/// every lower element of the pair is formed from itself and its mirror and written to both, so the strided reads of
/// the mirror stay within one tile. The same arithmetic per element as the untiled loop (`alpha (a_ij + a_ji)`, the
/// lower element first), so the result is bitwise the same; it only changes the order the elements are visited in.
/// Only the tile columns of part `part` of `parts`: tile column `J` of `T` holds `T - J` tile pairs, so `J` and
/// `T - 1 - J` go to the same part and those pairs of columns are dealt out in turn.
pub(crate) fn symmetrize_square_in_place(a: &mut [f64], n: usize, alpha: f64, part: usize, parts: usize) {
    debug_assert_eq!(a.len(), n * n);
    let tiles = n.div_ceil(SYMMETRIZE_TILE);
    for (j, tile_j) in (0..n).step_by(SYMMETRIZE_TILE).enumerate() {
        if j.min(tiles - 1 - j) % parts != part {
            continue;
        }
        let columns = tile_j..(tile_j + SYMMETRIZE_TILE).min(n);
        for tile_i in (tile_j..n).step_by(SYMMETRIZE_TILE) {
            let rows = tile_i..(tile_i + SYMMETRIZE_TILE).min(n);
            for j in columns.clone() {
                // The lower part of column j in this tile; on the diagonal tile from the diagonal down.
                for i in rows.start.max(j)..rows.end {
                    // a[(j, i)], the mirror: strided, within the tile pair.
                    let value = alpha * (a[i + j * n] + a[j + i * n]);
                    a[i + j * n] = value;
                    a[j + i * n] = value;
                }
            }
        }
    }
}
impl_matrix_math_mut!(Matrix<f64>);
impl_matrix_math_mut!(MatrixViewMut<'_, f64>);
