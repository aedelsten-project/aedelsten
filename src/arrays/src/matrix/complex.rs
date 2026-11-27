// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Complex matrices in split storage: the real and the imaginary part as two real matrices. Every product runs on
//! dgemm without a copy: real x complex and complex x real are two calls, complex x complex four, and the adjoint is
//! a transpose flag and a sign. The one copy is `to_interleaved`, for LAPACK's complex routines and the C/Fortran
//! side.

use super::*;
use super::factor::sealed::BlasOperand;
use super::factor::Operand;

/// A complex matrix as two real matrices of the same shape: owned, a view or a mutable view of both parts.
pub struct SplitComplexStruct<D>
where
    D: DataContainer<T = f64>,
{
    re: MatrixStruct<D>,
    im: MatrixStruct<D>,
}

/// An owned complex matrix.
pub type SplitComplexMatrix = SplitComplexStruct<Box<[f64]>>;
/// A view of a complex matrix.
pub type SplitComplexMatrixView<'a> = SplitComplexStruct<&'a [f64]>;
/// A mutable view of a complex matrix.
pub type SplitComplexMatrixViewMut<'a> = SplitComplexStruct<&'a mut [f64]>;

impl SplitComplexMatrix {
    /// The zero matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { re: Matrix::new(rows, cols), im: Matrix::new(rows, cols) }
    }

    /// From the interleaved layout (`re, im` per element, column-major), as LAPACK's complex routines and Fortran's
    /// `complex*16` store it.
    pub fn from_interleaved(rows: usize, cols: usize, data: &[f64]) -> Self {
        let mut out = Self::new(rows, cols);
        out.set_from_interleaved(data);
        out
    }
}

impl<'a> SplitComplexMatrixViewMut<'a> {
    /// Both parts from the arena, zeroed.
    pub fn from_arena(rows: usize, cols: usize, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        Ok(Self { re: MatrixViewMut::from_arena(rows, cols, arena)?, im: MatrixViewMut::from_arena(rows, cols, arena)? })
    }
}

impl<D> SplitComplexStruct<D>
where
    D: DataContainer<T = f64>,
{
    /// From the two parts, which must have the same shape.
    pub fn from_parts(re: MatrixStruct<D>, im: MatrixStruct<D>) -> Self {
        assert_eq!((re.rows, re.cols), (im.rows, im.cols), "from_parts: the real part is {} x {}, the imaginary part {} x {}", re.rows, re.cols, im.rows, im.cols);
        Self { re, im }
    }

    /// The real and the imaginary part.
    pub fn into_parts(self) -> (MatrixStruct<D>, MatrixStruct<D>) {
        (self.re, self.im)
    }

    /// The real part.
    pub fn re(&self) -> &MatrixStruct<D> {
        &self.re
    }

    /// The imaginary part.
    pub fn im(&self) -> &MatrixStruct<D> {
        &self.im
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.re.rows
    }

    /// The number of columns.
    pub fn cols(&self) -> usize {
        self.re.cols
    }

    /// A view of both parts.
    pub fn view(&self) -> SplitComplexMatrixView<'_> {
        SplitComplexStruct { re: MatrixView::new(self.rows(), self.cols(), self.re.data.resolve()), im: MatrixView::new(self.rows(), self.cols(), self.im.data.resolve()) }
    }

    /// The adjoint (conjugate transpose) as a product operand, no copy.
    pub fn view_adjoint(&self) -> SplitComplexAdjointView<'_> {
        SplitComplexAdjointView { re: self.re.blas_operand(), im: self.im.blas_operand() }
    }

    /// `output = self^H`.
    pub fn adjoint_prealloc<D1>(&self, output: &mut SplitComplexStruct<D1>)
    where
        D1: DataContainerMut<T = f64>,
    {
        assert_eq!((output.rows(), output.cols()), (self.cols(), self.rows()), "adjoint: the output is not {} x {}", self.cols(), self.rows());
        let (rows, cols) = (self.rows(), self.cols());
        let (re, im) = (self.re.data.resolve(), self.im.data.resolve());
        let (out_re, out_im) = (output.re.data.resolve_mut(), output.im.data.resolve_mut());
        for j in 0..cols {
            for i in 0..rows {
                out_re[j + i * cols] = re[i + j * rows];
                out_im[j + i * cols] = -im[i + j * rows];
            }
        }
    }

    /// `self^H`, on the heap.
    pub fn adjoint(&self) -> SplitComplexMatrix {
        let mut output = SplitComplexMatrix::new(self.cols(), self.rows());
        self.adjoint_prealloc(&mut output);
        output
    }

    /// `self^H`, from the arena.
    pub fn adjoint_arena<'a>(&self, arena: &Arena<'a>) -> Result<SplitComplexMatrixViewMut<'a>, ArenaError> {
        let mut output = SplitComplexMatrixViewMut::from_arena(self.cols(), self.rows(), arena)?;
        self.adjoint_prealloc(&mut output);
        Ok(output)
    }

    /// The interleaved layout (`re, im` per element, column-major), into `output` of `2 * rows * cols` values.
    pub fn to_interleaved_prealloc(&self, output: &mut [f64]) {
        let n = self.rows() * self.cols();
        assert_eq!(output.len(), 2 * n, "to_interleaved: {} values for {n} elements", output.len());
        for ((pair, re), im) in output.chunks_exact_mut(2).zip(self.re.data.resolve()).zip(self.im.data.resolve()) {
            (pair[0], pair[1]) = (*re, *im);
        }
    }

    /// The interleaved layout, on the heap.
    pub fn to_interleaved(&self) -> Box<[f64]> {
        let mut output = vec![0.0; 2 * self.rows() * self.cols()].into_boxed_slice();
        self.to_interleaved_prealloc(&mut output);
        output
    }

    /// The interleaved layout, from the arena.
    pub fn to_interleaved_arena<'a>(&self, arena: &Arena<'a>) -> Result<&'a mut [f64], ArenaError> {
        let output = arena.alloc_zeroed::<f64>(2 * self.rows() * self.cols())?;
        self.to_interleaved_prealloc(output);
        Ok(output)
    }
}

impl<D> SplitComplexStruct<D>
where
    D: DataContainerMut<T = f64>,
{
    /// The real part, to write.
    pub fn re_mut(&mut self) -> &mut MatrixStruct<D> {
        &mut self.re
    }

    /// The imaginary part, to write.
    pub fn im_mut(&mut self) -> &mut MatrixStruct<D> {
        &mut self.im
    }

    /// Both parts at once, mutably.
    pub fn parts_mut(&mut self) -> (&mut MatrixStruct<D>, &mut MatrixStruct<D>) {
        (&mut self.re, &mut self.im)
    }

    /// Both parts from the interleaved layout (`re, im` per element, column-major), `2 * rows * cols` values: the
    /// inverse of `to_interleaved_prealloc`.
    pub fn set_from_interleaved(&mut self, data: &[f64]) {
        let (rows, cols) = (self.rows(), self.cols());
        assert_eq!(data.len(), 2 * rows * cols, "from_interleaved: {rows} x {cols} needs {} values, got {}", 2 * rows * cols, data.len());
        let (re, im) = (self.re.data.resolve_mut(), self.im.data.resolve_mut());
        for ((pair, re), im) in data.chunks_exact(2).zip(re.iter_mut()).zip(im.iter_mut()) {
            (*re, *im) = (pair[0], pair[1]);
        }
    }
}

// The eigendecomposition of a Hermitian matrix, in the three forms of the real symmetric one
// (`MatrixMath::eigenvalues`, `eigenvectors` and their `_prealloc`/`_arena` forms). The matrix is assumed Hermitian, as
// the real solver assumes symmetry: only the upper triangle is read, and the input is not changed. Eigenvalues are real
// and ascending; the eigenvectors are the columns, orthonormal (`V^H V = 1`), with `self = V diag(values) V^H`.
// LAPACK's zheev works on an interleaved copy in `work` (it takes no split storage), which it overwrites with the
// eigenvectors; they are split back into `vectors`.
impl<D> SplitComplexStruct<D>
where
    D: DataContainer<T = f64>,
{
    /// The eigenvalues (ascending) into `values` (room for n) and the eigenvectors (columns) into `vectors` (n x n);
    /// `work` holds at least `hermitian_eigenvectors_work_len(n)` doubles. Returns false if LAPACK reports a failure
    /// or `work` is below its minimum (nothing is written past it; the outputs are then unspecified).
    pub fn eigenvectors_prealloc<D1>(&self, values: &mut [f64], vectors: &mut SplitComplexStruct<D1>, work: &mut [f64]) -> bool
    where
        D1: DataContainerMut<T = f64>,
    {
        let n = self.rows();
        assert_eq!(self.cols(), n, "eigenvectors_prealloc: the matrix is not square");
        assert_eq!((vectors.rows(), vectors.cols()), (n, n), "eigenvectors_prealloc: the eigenvector matrix is not n x n");
        assert!(values.len() >= n, "eigenvectors_prealloc: room for {} of {n} eigenvalues", values.len());
        if n == 0 {
            return true;
        }
        match self.zheev_in(work, 'V', values) {
            Some(eigenvectors) => {
                vectors.set_from_interleaved(eigenvectors);
                true
            }
            None => false,
        }
    }

    /// The eigenvalues (ascending) into `values` (room for n); `work` as for `eigenvectors_prealloc`.
    pub fn eigenvalues_prealloc(&self, values: &mut [f64], work: &mut [f64]) -> bool {
        let n = self.rows();
        assert_eq!(self.cols(), n, "eigenvalues_prealloc: the matrix is not square");
        assert!(values.len() >= n, "eigenvalues_prealloc: room for {} of {n} eigenvalues", values.len());
        n == 0 || self.zheev_in(work, 'N', values).is_some()
    }

    /// Eigenvalues and eigenvectors, on the heap.
    pub fn eigenvectors(&self) -> (Box<[f64]>, SplitComplexMatrix) {
        let n = self.rows();
        let (mut values, mut vectors, mut work) = (vec![0.0; n], SplitComplexMatrix::new(n, n), vec![0.0; hermitian_eigenvectors_work_len(n)]);
        assert!(self.eigenvectors_prealloc(&mut values, &mut vectors, &mut work), "eigenvectors: LAPACK failed");
        (values.into_boxed_slice(), vectors)
    }

    /// Eigenvalues, on the heap.
    pub fn eigenvalues(&self) -> Box<[f64]> {
        let n = self.rows();
        let (mut values, mut work) = (vec![0.0; n], vec![0.0; hermitian_eigenvectors_work_len(n)]);
        assert!(self.eigenvalues_prealloc(&mut values, &mut work), "eigenvalues: LAPACK failed");
        values.into_boxed_slice()
    }

    /// Eigenvalues and eigenvectors from the arena; LAPACK's scratch from a scope, free again on return.
    pub fn eigenvectors_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<(&'a mut [f64], SplitComplexMatrixViewMut<'a>), ArenaError> {
        let n = self.rows();
        let values = arena.alloc_zeroed::<f64>(n)?;
        let mut vectors = SplitComplexMatrixViewMut::from_arena(n, n, arena)?;
        let ok = arena.scope(|scope| {
            let work = scope.alloc::<f64>(hermitian_eigenvectors_work_len(n))?;
            Ok::<bool, ArenaError>(self.eigenvectors_prealloc(values, &mut vectors, work))
        })?;
        assert!(ok, "eigenvectors_arena: LAPACK failed");
        Ok((values, vectors))
    }

    /// Eigenvalues from the arena; LAPACK's scratch from a scope, free again on return.
    pub fn eigenvalues_arena<'a>(&self, arena: &mut Arena<'a>) -> Result<&'a mut [f64], ArenaError> {
        let n = self.rows();
        let values = arena.alloc_zeroed::<f64>(n)?;
        let ok = arena.scope(|scope| {
            let work = scope.alloc::<f64>(hermitian_eigenvectors_work_len(n))?;
            Ok::<bool, ArenaError>(self.eigenvalues_prealloc(values, work))
        })?;
        assert!(ok, "eigenvalues_arena: LAPACK failed");
        Ok(values)
    }

    /// zheev on an interleaved copy in `work`: `[copy, 2 n^2 | real scratch, 3n - 2 | complex scratch, the rest]`.
    /// The copy (the eigenvectors with `jobz = 'V'`), or None if `work` cannot hold the three or LAPACK fails (a
    /// complex scratch below its minimum is such a failure: zheev checks it before writing).
    fn zheev_in<'w>(&self, work: &'w mut [f64], jobz: char, values: &mut [f64]) -> Option<&'w [f64]> {
        let n = self.rows();
        let rwork_len = hermitian_rwork_len(n);
        if work.len() < 2 * n * n + rwork_len + 2 {
            return None;
        }
        let (a, rest) = work.split_at_mut(2 * n * n);
        let (rwork, zwork) = rest.split_at_mut(rwork_len);
        self.to_interleaved_prealloc(a);
        let (n_i, lwork) = (n as i32, (zwork.len() / 2) as i32);
        let mut info = 0;
        zheev(jobz, &CblasUpper, &n_i, a.as_mut_ptr(), &n_i, values.as_mut_ptr(), zwork.as_mut_ptr(), &lwork, rwork.as_mut_ptr(), &mut info);
        (info == 0).then_some(&*a)
    }
}

/// The adjoint of a complex matrix, from `view_adjoint`: only a factor, and it cannot be taken again (use the
/// original).
pub struct SplitComplexAdjointView<'a> {
    re: Operand<'a>,
    im: Operand<'a>,
}

// --- products ---

/// The crate-private side of the complex products.
pub(super) mod sealed {
    use super::super::factor::Operand;

    /// A complex operand as BLAS sees it: `re + i * im_sign * im`, each part possibly used transposed. A real operand
    /// has no imaginary part.
    pub struct ComplexOperand<'a> {
        /// The real part.
        pub re: Operand<'a>,
        /// The imaginary part and its sign; none for a real operand.
        pub im: Option<(Operand<'a>, f64)>,
    }

    /// Unnameable outside this crate, so only the arrays types are complex factors.
    pub trait MixedOperand {
        fn mixed_operand(&self) -> ComplexOperand<'_>;
    }
}

use sealed::ComplexOperand;

impl<D: DataContainer<T = f64>> sealed::MixedOperand for SplitComplexStruct<D> {
    fn mixed_operand(&self) -> ComplexOperand<'_> {
        ComplexOperand { re: self.re.blas_operand(), im: Some((self.im.blas_operand(), 1.0)) }
    }
}

impl sealed::MixedOperand for SplitComplexAdjointView<'_> {
    fn mixed_operand(&self) -> ComplexOperand<'_> {
        ComplexOperand { re: transposed(&self.re), im: Some((transposed(&self.im), -1.0)) }
    }
}

/// The same stored matrix, used with the other transpose flag.
fn transposed<'a>(op: &Operand<'a>) -> Operand<'a> {
    Operand { data: op.data, rows: op.rows, cols: op.cols, ld: op.ld, transposed: !op.transposed }
}

macro_rules! real_mixed_operand {
    ($($t:ty),*) => {$(
        impl sealed::MixedOperand for $t {
            fn mixed_operand(&self) -> ComplexOperand<'_> {
                ComplexOperand { re: self.blas_operand(), im: None }
            }
        }
        impl MixedFactor for $t {}
    )*};
}

impl<D: DataContainer<T = f64>> sealed::MixedOperand for MatrixStruct<D> {
    fn mixed_operand(&self) -> ComplexOperand<'_> {
        ComplexOperand { re: self.blas_operand(), im: None }
    }
}
impl<D: DataContainer<T = f64>> MixedFactor for MatrixStruct<D> {}
real_mixed_operand!(StridedMatrixView<'_>, TransposedMatrixView<'_>);

/// The right-hand factor of a complex product: any real factor (a matrix, a strided block, a transposed view) or a
/// complex one (a split complex matrix or its adjoint view).
pub trait MixedFactor: sealed::MixedOperand {}

/// A complex factor on the left of a product: a split complex matrix or its adjoint view. The `_scaled_prealloc`
/// form does the work; `alpha` and `beta` are real.
pub trait ComplexFactor: MixedFactor {
    /// `(rows, cols)` as a factor (an adjoint reports its transposed shape).
    fn shape(&self) -> (usize, usize) {
        self.mixed_operand().re.shape()
    }

    /// `output = alpha * self * other + beta * output`, `other` real or complex.
    fn matmul_scaled_prealloc<F, D>(&self, other: &F, output: &mut SplitComplexStruct<D>, alpha: f64, beta: f64)
    where
        F: MixedFactor + ?Sized,
        D: DataContainerMut<T = f64>,
    {
        complex_product(self.mixed_operand(), other.mixed_operand(), output, alpha, beta);
    }

    /// `output = self * other`.
    fn matmul_prealloc<F, D>(&self, other: &F, output: &mut SplitComplexStruct<D>)
    where
        F: MixedFactor + ?Sized,
        D: DataContainerMut<T = f64>,
    {
        self.matmul_scaled_prealloc(other, output, 1.0, 0.0);
    }

    /// `self * other`, on the heap.
    fn matmul<F: MixedFactor + ?Sized>(&self, other: &F) -> SplitComplexMatrix {
        let mut output = SplitComplexMatrix::new(self.shape().0, other.mixed_operand().re.shape().1);
        self.matmul_prealloc(other, &mut output);
        output
    }

    /// `self * other`, from the arena.
    fn matmul_arena<'a, F: MixedFactor + ?Sized>(&self, other: &F, arena: &Arena<'a>) -> Result<SplitComplexMatrixViewMut<'a>, ArenaError> {
        let mut output = SplitComplexMatrixViewMut::from_arena(self.shape().0, other.mixed_operand().re.shape().1, arena)?;
        self.matmul_prealloc(other, &mut output);
        Ok(output)
    }
}

impl<D: DataContainer<T = f64>> MixedFactor for SplitComplexStruct<D> {}
impl<D: DataContainer<T = f64>> ComplexFactor for SplitComplexStruct<D> {}
impl MixedFactor for SplitComplexAdjointView<'_> {}
impl ComplexFactor for SplitComplexAdjointView<'_> {}

/// `c = alpha * a * b + beta * c` for real operands, into one part of a complex output.
fn gemm(a: &Operand<'_>, b: &Operand<'_>, c: &mut [f64], alpha: f64, beta: f64) {
    matrix_matrix_multiplication_ld(
        a.data, a.rows as i32, a.cols as i32, a.ld as i32, a.trans(),
        b.data, b.rows as i32, b.cols as i32, b.ld as i32, b.trans(),
        c, a.shape().0 as i32, alpha, beta,
    );
}

/// `c = alpha * a * b + beta * c`, `a` and `b` each real or complex, with `(ar + i ai)(br + i bi) = (ar br - ai bi) +
/// i (ar bi + ai br)`: one dgemm per nonzero term, no copy.
pub(super) fn complex_product<D>(a: ComplexOperand<'_>, b: ComplexOperand<'_>, c: &mut SplitComplexStruct<D>, alpha: f64, beta: f64)
where
    D: DataContainerMut<T = f64>,
{
    let ((m, k), (k2, n)) = (a.re.shape(), b.re.shape());
    assert_eq!(k, k2, "matmul: inner dimension mismatch, {m} x {k} times {k2} x {n}");
    assert_eq!((c.rows(), c.cols()), (m, n), "matmul: the output is not {m} x {n}");
    if m == 0 || n == 0 {
        return;
    }
    let (c_re, c_im) = (c.re.data.resolve_mut(), c.im.data.resolve_mut());
    gemm(&a.re, &b.re, c_re, alpha, beta);
    match (&a.im, &b.im) {
        (None, None) => scale_in_place(c_im, beta),
        (Some((ai, sa)), None) => gemm(ai, &b.re, c_im, alpha * sa, beta),
        (None, Some((bi, sb))) => gemm(&a.re, bi, c_im, alpha * sb, beta),
        (Some((ai, sa)), Some((bi, sb))) => {
            gemm(ai, bi, c_re, -alpha * sa * sb, 1.0);
            gemm(&a.re, bi, c_im, alpha * sb, beta);
            gemm(ai, &b.re, c_im, alpha * sa, 1.0);
        }
    }
}
