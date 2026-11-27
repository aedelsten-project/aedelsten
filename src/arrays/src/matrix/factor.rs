// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Product operands. `Factor`: anything that can be multiplied (a matrix, a strided block, a transposed view), with
//! the `matmul` and `matvec` families. `ColumnFactor`: the operands with contiguous columns (not transposed views).

use super::*;
use crate::vector::{Vector, VectorMut};
use super::complex::sealed::ComplexOperand;
use super::complex::{complex_product, ComplexFactor, SplitComplexMatrix, SplitComplexMatrixViewMut, SplitComplexStruct};

/// The crate-private side of the products: the operands as BLAS sees them.
pub(super) mod sealed {
    /// The stored matrix as BLAS sees it: the data from its first element, the stored rows and columns, the
    /// leading dimension (the distance between the starts of two columns) and whether it is used transposed.
    pub struct Operand<'a> {
        /// The data from the first element.
        pub data: &'a [f64],
        /// The stored rows.
        pub rows: usize,
        /// The stored columns.
        pub cols: usize,
        /// The leading dimension.
        pub ld: usize,
        /// Whether it is used transposed.
        pub transposed: bool,
    }

    /// Unnameable outside this crate, so only the arrays types are factors.
    pub trait BlasOperand {
        fn blas_operand(&self) -> Operand<'_>;
    }

    /// The written matrix as BLAS sees it: the data from its first element, its rows and columns, and the leading
    /// dimension.
    pub struct Output<'a> {
        /// The data from the first element.
        pub data: &'a mut [f64],
        /// The rows.
        pub rows: usize,
        /// The columns.
        pub cols: usize,
        /// The leading dimension.
        pub ld: usize,
    }

    /// Unnameable outside this crate, so only the arrays types are product outputs.
    pub trait BlasOutput {
        fn blas_output(&mut self) -> Output<'_>;
    }
}

pub(super) use sealed::Operand;

/// The output of a product: any writable matrix (`Matrix`, `MatrixViewMut`) or a `StridedMatrixViewMut` (a block of
/// a larger matrix, `view_rows_mut`), written in place by the one BLAS call.
pub trait ProductOutput: sealed::BlasOutput {}

impl<M: MatrixTraitMut<f64>> sealed::BlasOutput for M {
    #[inline]
    fn blas_output(&mut self) -> sealed::Output<'_> {
        let (rows, cols) = (self.rows(), self.cols());
        sealed::Output { data: self.get_data_mut(), rows, cols, ld: rows }
    }
}

impl<M: MatrixTraitMut<f64>> ProductOutput for M {}

impl Operand<'_> {
    /// BLAS's transpose flag for the operand.
    #[inline]
    pub(super) fn trans(&self) -> CblasTranspose {
        if self.transposed { CblasTrans } else { CblasNoTrans }
    }

    /// `(rows, cols)` as a factor, after any transpose.
    #[inline]
    pub(super) fn shape(&self) -> (usize, usize) {
        if self.transposed { (self.cols, self.rows) } else { (self.rows, self.cols) }
    }
}

/// A factor of a product, on either side: any matrix, a `StridedMatrixView` or a `TransposedMatrixView`. Every
/// product is one BLAS call, no copy. The `_scaled_prealloc` forms do the work; the others wrap them.
pub trait Factor: sealed::BlasOperand {
    /// `(rows, cols)` as a factor (a transposed view reports its transposed shape).
    #[inline]
    fn shape(&self) -> (usize, usize) {
        self.blas_operand().shape()
    }

    /// `output = alpha * self * other + beta * output`.
    /// `output = alpha * self * other + beta * output`. `output` is a matrix or a strided block of one.
    #[inline]
    fn matmul_scaled_prealloc<F, D>(&self, other: &F, output: &mut D, alpha: f64, beta: f64)
    where
        F: Factor + ?Sized,
        D: ProductOutput + ?Sized,
    {
        let (a, b) = (self.blas_operand(), other.blas_operand());
        let ((m, k), (k2, n)) = (a.shape(), b.shape());
        assert_eq!(k, k2, "matmul: inner dimension mismatch, {m} x {k} times {k2} x {n}");
        let c = output.blas_output();
        assert_eq!((c.rows, c.cols), (m, n), "matmul: the output is not {m} x {n}");
        if m == 0 || n == 0 {
            return;
        }
        matrix_matrix_multiplication_ld(
            a.data, a.rows as i32, a.cols as i32, a.ld as i32, a.trans(),
            b.data, b.rows as i32, b.cols as i32, b.ld as i32, b.trans(),
            c.data, c.ld as i32, alpha, beta,
        );
    }

    /// `output = self * other`.
    #[inline]
    fn matmul_prealloc<F, D>(&self, other: &F, output: &mut D)
    where
        F: Factor + ?Sized,
        D: ProductOutput + ?Sized,
    {
        self.matmul_scaled_prealloc(other, output, 1.0, 0.0);
    }

    /// `self * other`, on the heap.
    #[inline]
    fn matmul<F>(&self, other: &F) -> Matrix<f64>
    where
        F: Factor + ?Sized,
    {
        let mut output = Matrix::<f64>::new(self.shape().0, other.shape().1);
        self.matmul_prealloc(other, &mut output);
        output
    }

    /// `self * other`, from the arena.
    #[inline]
    fn matmul_arena<'a, F>(&self, other: &F, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, f64>, ArenaError>
    where
        F: Factor + ?Sized,
    {
        let mut output = MatrixViewMut::<f64>::from_arena(self.shape().0, other.shape().1, arena)?;
        self.matmul_prealloc(other, &mut output);
        Ok(output)
    }

    /// `y = alpha * self * x + beta * y`. `x` and `y` are any vectors, strided or not (a row, a diagonal).
    #[inline]
    fn matvec_scaled_prealloc<X, Y>(&self, x: &X, y: &mut Y, alpha: f64, beta: f64)
    where
        X: Vector + ?Sized,
        Y: VectorMut + ?Sized,
    {
        let a = self.blas_operand();
        let (m, n) = a.shape();
        assert_eq!(x.len(), n, "matvec: x has {} entries for a {m} x {n} matrix", x.len());
        assert_eq!(y.len(), m, "matvec: y has {} entries for a {m} x {n} matrix", y.len());
        if m == 0 {
            return;
        }
        matrix_vector_multiplication_ld(a.data, a.rows as i32, a.cols as i32, a.ld as i32, a.trans(), x, y, alpha, beta);
    }

    /// `y = self * x`.
    #[inline]
    fn matvec_prealloc<X, Y>(&self, x: &X, y: &mut Y)
    where
        X: Vector + ?Sized,
        Y: VectorMut + ?Sized,
    {
        self.matvec_scaled_prealloc(x, y, 1.0, 0.0);
    }

    /// `self * x`, on the heap.
    #[inline]
    fn matvec<X: Vector + ?Sized>(&self, x: &X) -> Box<[f64]> {
        let mut y = vec![0.0; self.shape().0].into_boxed_slice();
        self.matvec_prealloc(x, &mut y);
        y
    }

    /// `self * x`, from the arena.
    #[inline]
    fn matvec_arena<'a, X: Vector + ?Sized>(&self, x: &X, arena: &Arena<'a>) -> Result<&'a mut [f64], ArenaError> {
        let y = arena.alloc_zeroed::<f64>(self.shape().0)?;
        self.matvec_prealloc(x, y);
        Ok(y)
    }

    /// `output = alpha * self * other + beta * output` with a complex `other`: two dgemm, no copy.
    #[inline]
    fn matmul_complex_scaled_prealloc<C, D>(&self, other: &C, output: &mut SplitComplexStruct<D>, alpha: f64, beta: f64)
    where
        C: ComplexFactor + ?Sized,
        D: DataContainerMut<T = f64>,
    {
        complex_product(ComplexOperand { re: self.blas_operand(), im: None }, other.mixed_operand(), output, alpha, beta);
    }

    /// `output = self * other`, `other` complex.
    #[inline]
    fn matmul_complex_prealloc<C, D>(&self, other: &C, output: &mut SplitComplexStruct<D>)
    where
        C: ComplexFactor + ?Sized,
        D: DataContainerMut<T = f64>,
    {
        self.matmul_complex_scaled_prealloc(other, output, 1.0, 0.0);
    }

    /// `self * other`, `other` complex, on the heap.
    #[inline]
    fn matmul_complex<C: ComplexFactor + ?Sized>(&self, other: &C) -> SplitComplexMatrix {
        let mut output = SplitComplexMatrix::new(self.shape().0, other.shape().1);
        self.matmul_complex_prealloc(other, &mut output);
        output
    }

    /// `self * other`, `other` complex, from the arena.
    #[inline]
    fn matmul_complex_arena<'a, C: ComplexFactor + ?Sized>(&self, other: &C, arena: &Arena<'a>) -> Result<SplitComplexMatrixViewMut<'a>, ArenaError> {
        let mut output = SplitComplexMatrixViewMut::from_arena(self.shape().0, other.shape().1, arena)?;
        self.matmul_complex_prealloc(other, &mut output);
        Ok(output)
    }
}

/// A factor with contiguous columns: any matrix or a `StridedMatrixView`, for signatures that read columns. The method
/// names differ from `MatrixTrait`'s, so both traits can be in scope without ambiguity.
pub trait ColumnFactor: Factor {
    /// Column `j`, contiguous, `rows` long.
    #[inline]
    fn column(&self, j: usize) -> &[f64] {
        let a = self.blas_operand();
        assert!(j < a.cols, "column: column {j} of {}", a.cols);
        &a.data[j * a.ld..j * a.ld + a.rows]
    }

    /// The transpose as a product operand, no copy: BLAS reads it with its transpose flag.
    #[inline]
    fn view_transposed(&self) -> TransposedMatrixView<'_> {
        let a = self.blas_operand();
        TransposedMatrixView { rows: a.rows, cols: a.cols, ld: a.ld, data: a.data }
    }
}

impl<D> sealed::BlasOperand for MatrixStruct<D>
where
    D: DataContainer<T = f64>,
{
    #[inline]
    fn blas_operand(&self) -> Operand<'_> {
        Operand { data: self.get_data(), rows: self.rows(), cols: self.cols(), ld: self.rows(), transposed: false }
    }
}

impl<D> Factor for MatrixStruct<D> where D: DataContainer<T = f64> {}
impl<D> ColumnFactor for MatrixStruct<D> where D: DataContainer<T = f64> {}

impl<D> MatrixStruct<D>
where
    D: DataContainer<T = f64>,
{
    /// The rows `range` (all columns), without a copy: a `StridedMatrixView`, its columns `self.rows()` apart. E.g.
    /// one shell's block of a coefficient matrix, as a factor of a product.
    #[inline]
    pub fn view_rows(&self, range: std::ops::Range<usize>) -> StridedMatrixView<'_> {
        assert!(range.start <= range.end && range.end <= self.rows(), "view_rows: rows {range:?} of {}", self.rows());
        let data = if range.is_empty() || self.cols() == 0 { &[][..] } else { &self.get_data()[range.start..] };
        StridedMatrixView::new(range.len(), self.cols(), self.rows(), data)
    }
}

impl<D> MatrixStruct<D>
where
    D: DataContainerMut<T = f64>,
{
    /// `view_rows`, writable: a `StridedMatrixViewMut`, e.g. one shell's rows of a larger output, written in place
    /// as the output of a product.
    #[inline]
    pub fn view_rows_mut(&mut self, range: std::ops::Range<usize>) -> StridedMatrixViewMut<'_> {
        assert!(range.start <= range.end && range.end <= self.rows(), "view_rows_mut: rows {range:?} of {}", self.rows());
        let (rows, cols) = (self.rows(), self.cols());
        let data = if range.is_empty() || cols == 0 { &mut [][..] } else { &mut self.data.resolve_mut()[range.start..] };
        StridedMatrixViewMut::new(range.len(), cols, rows, data)
    }
}

/// The non-contiguous matrix view: a `rows x cols` block of a larger column-major buffer whose columns are `ld`
/// apart, e.g. a slice of an `Array3D` at a fixed middle index (`Array3DStruct::view_dim1_matrix`) or a range of a
/// matrix's rows (`view_rows`). Only functions of this crate make one. Read-only and only a factor: no indexing and
/// no `get_data`, so nothing can read it as if it were contiguous.
pub struct StridedMatrixView<'a> {
    rows: usize,
    cols: usize,
    ld: usize,
    data: &'a [f64],
}

impl<'a> StridedMatrixView<'a> {
    /// Column `j` of the block is `data[j * ld..j * ld + rows]`.
    #[inline]
    pub(crate) fn new(rows: usize, cols: usize, ld: usize, data: &'a [f64]) -> Self {
        assert!(ld >= rows, "StridedMatrixView: the leading dimension {ld} is smaller than the rows {rows}");
        let needed = if rows == 0 || cols == 0 { 0 } else { ld * (cols - 1) + rows };
        assert!(data.len() >= needed, "StridedMatrixView: {rows} x {cols} with leading dimension {ld} needs {needed} values, got {}", data.len());
        Self { rows, cols, ld, data }
    }

    /// Rows `range` of the block: a block of the same buffer, its columns as far apart.
    #[inline]
    pub fn view_rows(&self, range: std::ops::Range<usize>) -> StridedMatrixView<'a> {
        assert!(range.start <= range.end && range.end <= self.rows, "view_rows: rows {range:?} of {}", self.rows);
        let data = if range.is_empty() || self.cols == 0 { &[][..] } else { &self.data[range.start..] };
        StridedMatrixView::new(range.len(), self.cols, self.ld, data)
    }
}

impl sealed::BlasOperand for StridedMatrixView<'_> {
    #[inline]
    fn blas_operand(&self) -> Operand<'_> {
        Operand { data: self.data, rows: self.rows, cols: self.cols, ld: self.ld, transposed: false }
    }
}

impl Factor for StridedMatrixView<'_> {}
impl ColumnFactor for StridedMatrixView<'_> {}

/// The writable counterpart of `StridedMatrixView`, from `Array3DStruct::view_dim1_matrix_mut`: a `rows x cols`
/// block of a larger column-major buffer whose columns are `ld` apart. Written through its columns only
/// (`iter_mut`), each contiguous: no indexing and no `get_data_mut`, so nothing can write it as if it were
/// contiguous. Only functions of this crate make one.
pub struct StridedMatrixViewMut<'a> {
    rows: usize,
    cols: usize,
    ld: usize,
    data: &'a mut [f64],
}

impl<'a> StridedMatrixViewMut<'a> {
    /// Column `j` of the block is `data[j * ld..j * ld + rows]`.
    #[inline]
    pub(crate) fn new(rows: usize, cols: usize, ld: usize, data: &'a mut [f64]) -> Self {
        assert!(ld >= rows, "StridedMatrixViewMut: the leading dimension {ld} is smaller than the rows {rows}");
        let needed = if rows == 0 || cols == 0 { 0 } else { ld * (cols - 1) + rows };
        assert!(data.len() >= needed, "StridedMatrixViewMut: {rows} x {cols} with leading dimension {ld} needs {needed} values, got {}", data.len());
        Self { rows, cols, ld, data }
    }

    /// `(rows, cols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The columns in order, each contiguous and `rows` long. The last one may end the buffer: it is cut to `rows`,
    /// never read as `ld` long.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [f64]> {
        let rows = self.rows;
        let cols = if rows == 0 { 0 } else { self.cols };
        self.data.chunks_mut(self.ld.max(1)).take(cols).map(move |column| &mut column[..rows])
    }
}

impl sealed::BlasOutput for StridedMatrixViewMut<'_> {
    #[inline]
    fn blas_output(&mut self) -> sealed::Output<'_> {
        sealed::Output { data: &mut *self.data, rows: self.rows, cols: self.cols, ld: self.ld }
    }
}

impl ProductOutput for StridedMatrixViewMut<'_> {}

/// The transpose of a matrix or a strided block, from `view_transposed`: only a factor (its columns are the
/// original's rows, so it has no column access), and it cannot be transposed again (use the original).
pub struct TransposedMatrixView<'a> {
    rows: usize,
    cols: usize,
    ld: usize,
    data: &'a [f64],
}

impl sealed::BlasOperand for TransposedMatrixView<'_> {
    #[inline]
    fn blas_operand(&self) -> Operand<'_> {
        Operand { data: self.data, rows: self.rows, cols: self.cols, ld: self.ld, transposed: true }
    }
}

impl Factor for TransposedMatrixView<'_> {}
