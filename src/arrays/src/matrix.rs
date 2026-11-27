// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The matrix, column-major: owned (`Matrix`), a view (`MatrixView`) and a mutable view (`MatrixViewMut`), one
//! struct over its data container; the products, factorizations and other operations in the submodules.

use crate::{Array3D, Array4D, Array5D, Array7D, Array8D};
use crate::scratch::{Arena, ArenaError};
use crate::array3d::*;
use crate::array4d::*;
use crate::array5d::*;
use crate::array6d::*;
use crate::array7d::*;
use crate::array8d::*;
use crate::blas_interface::{
    dgels, dgemm, dgemv, dgesvd, dgetrf, dgetri, dsyev, dsygv, zheev,
    CblasOrder::CblasColMajor,
    CblasTranspose,
    CblasTranspose::{CblasNoTrans, CblasTrans},
    CblasUplo::{CblasUpper, CblasLower},
};
use crate::vector_math::{dot, scale_in_place, add_in_place, add_scaled_in_place};
use std::debug_assert_eq;
use std::ops::{Index, IndexMut};

mod blocks;
mod complex;
mod factor;
mod lapack;
mod math;
mod products;
mod vectors;
#[cfg(test)]
mod tests;

pub use blocks::{block_sandwich_prealloc, block_signed_permutation_prealloc, SignedPermutation};
pub use complex::{ComplexFactor, MixedFactor, SplitComplexAdjointView, SplitComplexMatrix, SplitComplexMatrixView, SplitComplexMatrixViewMut, SplitComplexStruct};
pub use factor::{ColumnFactor, Factor, ProductOutput, StridedMatrixView, StridedMatrixViewMut, TransposedMatrixView};
pub use lapack::{eigenvectors_work_len, hermitian_eigenvectors_work_len, svd_work_len, sygv_work_len, SVD};
pub use math::{MatrixMath, MatrixMathMut};
use lapack::*;
use products::*;

/// The storage of an array's elements: owned (`Box<[T]>`) or borrowed (`&[T]`, `&mut [T]`).
pub trait DataContainer {
    /// The element type.
    type T;
    /// The elements.
    fn resolve(&self) -> &[Self::T];
}

/// Storage whose elements can be written.
pub trait DataContainerMut: DataContainer {
    /// The elements, to write.
    fn resolve_mut(&mut self) -> &mut [Self::T];
}

impl<T> DataContainer for Box<[T]> {
    type T = T;
    fn resolve(&self) -> &[Self::T] {
        self.as_ref()
    }
}

impl<'a, T> DataContainer for &'a [T] {
    type T = T;
    fn resolve(&self) -> &[Self::T] {
        self
    }
}

impl<'a, T> DataContainer for &'a mut [T] {
    type T = T;
    fn resolve(&self) -> &[Self::T] {
        self
    }
}

impl<T> DataContainerMut for Box<[T]> {
    fn resolve_mut(&mut self) -> &mut [Self::T] {
        self.as_mut()
    }
}

impl<'a, T> DataContainerMut for &'a mut [T] {
    fn resolve_mut(&mut self) -> &mut [Self::T] {
        self
    }
}

/// A `rows x cols` matrix, column-major, over data `D`: owned or borrowed.
#[derive(Debug, PartialEq)]
pub struct MatrixStruct<D>
where
    D: DataContainer,
{
    rows: usize,
    cols: usize,
    data: D,
}
/// An owned matrix.
pub type Matrix<T> = MatrixStruct<Box<[T]>>;
/// A view of a matrix.
pub type MatrixView<'a, T> = MatrixStruct<&'a [T]>;
/// A mutable view of a matrix.
pub type MatrixViewMut<'a, T> = MatrixStruct<&'a mut [T]>;

/// Reading a matrix, owned or a view.
pub trait MatrixTrait<T>: Index<(usize, usize), Output = T> {
    /// The number of rows.
    fn rows(&self) -> usize;
    /// The number of columns.
    fn cols(&self) -> usize;
    /// The elements, column-major.
    fn get_data(&self) -> &[T];
    /// The whole matrix as a view (e.g. to hand an owned matrix to code that takes views).
    fn view(&self) -> MatrixView<'_, T> {
        MatrixView::new(self.rows(), self.cols(), self.get_data())
    }
    /// The elements as a 3-dimensional array of dimensions `dims`, with as many elements; no copy.
    fn view_reshaped_array3d(&self, dims: (usize, usize, usize)) -> Array3DView<'_, T>;
    /// The columns, in order, as slices.
    fn iter(&self) -> std::slice::ChunksExact<'_, T> {
        self.iter_col()
    }
    /// The columns, in order, as slices (as `iter`).
    fn iter_col(&self) -> std::slice::ChunksExact<'_, T>;
    /// Column `l0`.
    fn get_col(&self, l0: usize) -> &[T];
    /// The columns `range` (all rows), without a copy: contiguous, so an ordinary view.
    fn view_columns(&self, range: std::ops::Range<usize>) -> MatrixView<'_, T> {
        let rows = self.rows();
        assert!(range.start <= range.end && range.end <= self.cols(), "view_columns: columns {range:?} of {}", self.cols());
        MatrixView::new(rows, range.len(), &self.get_data()[range.start * rows..range.end * rows])
    }
    /// The transpose.
    fn transpose(&self) -> Matrix<T>
    where
        T: Default + Clone;
    /// The transpose, into `output` (`cols x rows`).
    fn transpose_prealloc<M>(&self, output: &mut M)
    where
        T: Default + Clone,
        M: MatrixTraitMut<T>;
    /// The transpose, from the arena.
    fn transpose_arena<'a>(&self, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>
    where
        T: Default + Clone;
}

/// Writing a matrix, owned or a mutable view.
pub trait MatrixTraitMut<T>: MatrixTrait<T> + IndexMut<(usize, usize), Output = T> {
    /// The elements, column-major, to write.
    fn get_data_mut(&mut self) -> &mut [T];
    /// The whole matrix as a mutable view; on a view, a reborrow.
    fn view_mut(&mut self) -> MatrixViewMut<'_, T> {
        let (rows, cols) = (self.rows(), self.cols());
        MatrixViewMut::new(rows, cols, self.get_data_mut())
    }
    /// As `view_reshaped_array3d`, writable.
    fn view_reshaped_array3d_mut(&mut self, dims: (usize, usize, usize)) -> Array3DViewMut<'_, T>;
    /// As `iter`, writable.
    fn iter_mut(&mut self) -> std::slice::ChunksExactMut<'_, T> {
        self.iter_col_mut()
    }
    /// As `iter_col`, writable.
    fn iter_col_mut(&mut self) -> std::slice::ChunksExactMut<'_, T>;
    /// As `get_col`, writable.
    fn get_col_mut(&mut self, l0: usize) -> &mut [T];
    /// The columns `l0` and `l1` (distinct), both mutable, in that order.
    fn get_two_cols_mut(&mut self, l0: usize, l1: usize) -> (&mut [T], &mut [T]) {
        let (rows, cols) = (self.rows(), self.cols());
        assert!(l0 != l1 && l0 < cols && l1 < cols, "get_two_cols_mut: columns {l0} and {l1} of {cols}");
        let (low, high) = self.get_data_mut().split_at_mut(l0.max(l1) * rows);
        let (low, high) = (&mut low[l0.min(l1) * rows..(l0.min(l1) + 1) * rows], &mut high[..rows]);
        if l0 < l1 { (low, high) } else { (high, low) }
    }
    /// `view_columns`, mutable.
    fn view_columns_mut(&mut self, range: std::ops::Range<usize>) -> MatrixViewMut<'_, T> {
        let rows = self.rows();
        assert!(range.start <= range.end && range.end <= self.cols(), "view_columns_mut: columns {range:?} of {}", self.cols());
        MatrixViewMut::new(rows, range.len(), &mut self.get_data_mut()[range.start * rows..range.end * rows])
    }
    /// Every element set to `val`.
    fn fill(&mut self, val: T)
    where
        T: Default + Clone;
    /// The strict upper triangle set from the lower one (a square matrix), making it symmetric.
    fn copy_lower_to_upper(&mut self)
    where
        T: Default + Clone + Copy;
    /// The strict lower triangle set from the upper one (a square matrix), making it symmetric.
    fn copy_upper_to_lower(&mut self)
    where
        T: Default + Clone + Copy;
}

/// Defines `into_reshaped_arrayNd`, for an owned array (`plain`) or a mutable view (`muta`): its data, without a copy,
/// as an N-dimensional array of the dimensions given as one tuple (`into_reshaped_array3d((d0, d1, d2))`), owned or a
/// mutable view with the same lifetime. Panics unless the dimensions hold as many elements.
#[macro_export]
macro_rules! impl_into_array {
    (@usize $dim:ident) => { usize };
    (plain, $func_name:ident, $ArrayType:ident, $($dim:ident),*) => {
        /// The data, without a copy, as an array of the dimensions given as one tuple, which must hold as many
        /// elements.
        #[inline(always)]
        pub fn $func_name(self, ($($dim),*): ($(crate::impl_into_array!(@usize $dim)),*)) -> $ArrayType<T> {
            let length = self.get_data().len();
            let new_length = [$($dim),*].iter().product::<usize>();
            if new_length == length {
                $ArrayType::<T>::new_from_vec($($dim),*, self.data)
            } else {
                panic!("Matrix dimensions do not match {} dimensions", stringify!($ArrayType));
            }
        }
    };
    (muta, $func_name:ident, $ArrayType:ident, $($dim:ident),*) => {
        /// The data, without a copy, as an array of the dimensions given as one tuple, which must hold as many
        /// elements.
        #[inline(always)]
        pub fn $func_name(self, ($($dim),*): ($(crate::impl_into_array!(@usize $dim)),*)) -> $ArrayType<'a, T> {
            let length = self.get_data().len();
            let new_length = [$($dim),*].iter().product::<usize>();
            if new_length == length {
                $ArrayType::<'a, T>::new($($dim),*, self.data)
            } else {
                panic!("Matrix dimensions do not match {} dimensions", stringify!($ArrayType));
            }
        }
    };
}

impl<D> MatrixStruct<D>
where
    D: DataContainer,
{
    /// The flat position of element `(i, j)`, column-major.
    pub fn compute_index(&self, i: usize, j: usize) -> usize {
        j * self.rows + i
    }
}

impl<T> Matrix<T> {
    /// A `rows x cols` matrix, every element `T::default()`.
    pub fn new(rows: usize, cols: usize) -> Matrix<T>
    where
        T: Default + Clone,
    {
        Matrix {
            rows,
            cols,
            data: vec![Default::default(); cols * rows].into_boxed_slice(),
        }
    }

    /// Takes `vec` (`rows * cols` elements, column-major) as a `rows x cols` matrix.
    pub fn new_from_vec(rows: usize, cols: usize, vec: Box<[T]>) -> Matrix<T> {
        debug_assert_eq!(rows * cols, vec.len());
        Matrix {
            rows,
            cols,
            data: vec,
        }
    }

    /// The elements, column-major.
    pub fn into_data(self) -> Box<[T]> {
        self.data
    }

    impl_into_array!(plain, into_reshaped_array3d, Array3D, dim0, dim1, dim2);
    impl_into_array!(plain, into_reshaped_array4d, Array4D, dim0, dim1, dim2, dim3);
    impl_into_array!(plain, into_reshaped_array5d, Array5D, dim0, dim1, dim2, dim3, dim4);
    impl_into_array!(plain, into_reshaped_array6d, Array6D, dim0, dim1, dim2, dim3, dim4, dim5);
    impl_into_array!(
        plain,
        into_reshaped_array7d,
        Array7D,
        dim0,
        dim1,
        dim2,
        dim3,
        dim4,
        dim5,
        dim6
    );

    impl_into_array!(
        plain,
        into_reshaped_array8d,
        Array8D,
        dim0,
        dim1,
        dim2,
        dim3,
        dim4,
        dim5,
        dim6,
        dim7
    );
}

impl<'a, T> MatrixView<'a, T> {
    /// `data` (`rows * cols` elements, column-major) as a `rows x cols` matrix.
    pub fn new(rows: usize, cols: usize, data: &'a [T]) -> MatrixView<'a, T> {
        debug_assert_eq!(rows * cols, data.len());
        MatrixView { rows, cols, data }
    }
}

impl<'a, T> MatrixViewMut<'a, T> {
    /// `data` (`rows * cols` elements, column-major) as a `rows x cols` matrix.
    pub fn new(rows: usize, cols: usize, data: &'a mut [T]) -> MatrixViewMut<'a, T> {
        debug_assert_eq!(rows * cols, data.len());
        MatrixViewMut { rows, cols, data }
    }

    /// A `rows x cols` matrix, zeroed, from the arena.
    pub fn from_arena(
        rows: usize,
        cols: usize,
        arena: &Arena<'a>,
    ) -> Result<Self, ArenaError>
    where
        T: Default + Clone,
    {
        let matrix_slice = arena.alloc::<T>(crate::scratch::checked_product(&[rows, cols])?)?;

        matrix_slice.fill(T::default());

        Ok(MatrixStruct {
            rows,
            cols,
            data: matrix_slice,
        })
    }

    impl_into_array!(muta, into_reshaped_array3d, Array3DViewMut, dim0, dim1, dim2);
    impl_into_array!(muta, into_reshaped_array4d, Array4DViewMut, dim0, dim1, dim2, dim3);
    impl_into_array!(muta, into_reshaped_array5d, Array5DViewMut, dim0, dim1, dim2, dim3, dim4);
    impl_into_array!(muta, into_reshaped_array6d, Array6DViewMut, dim0, dim1, dim2, dim3, dim4, dim5);
    impl_into_array!(
        muta,
        into_reshaped_array7d,
        Array7DViewMut,
        dim0,
        dim1,
        dim2,
        dim3,
        dim4,
        dim5,
        dim6
    );

    impl_into_array!(
        muta,
        into_reshaped_array8d,
        Array8DViewMut,
        dim0,
        dim1,
        dim2,
        dim3,
        dim4,
        dim5,
        dim6,
        dim7
    );
}

impl<D> Clone for MatrixStruct<D>
where
    D: DataContainer + Clone,
{
    fn clone(&self) -> Self {
        MatrixStruct {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<T, D> MatrixTrait<T> for MatrixStruct<D>
where
    D: DataContainer<T = T>,
{
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn get_data(&self) -> &[T] {
        self.data.resolve()
    }
    
    fn view_reshaped_array3d(&self, dims: (usize, usize, usize)) -> Array3DView<'_, D::T> {
        let (dim0, dim1, dim2) = dims;
        let length = self.rows * self.cols;
        let new_length = dim0 * dim1 * dim2;
        if new_length == length {
            Array3DView::<D::T>::new(dim0, dim1, dim2, &self.get_data())
        } else {
            panic!(
                "Matrix dimensions do not match {} dimensions",
                stringify!(Array3DView)
            );
        }
    }

    fn iter_col(&self) -> std::slice::ChunksExact<'_, D::T> {
        self.get_data().chunks_exact(self.rows)
    }

    fn get_col(&self, l0: usize) -> &[D::T] {
        let offset: usize = self.compute_index(0, l0);
        &self.get_data()[offset..offset + self.rows]
    }

    fn transpose(&self) -> Matrix<T>
    where
        T: Default + Clone,
    {
        let out_rows = self.cols();
        let out_cols = self.rows();
        let mut output = Matrix::<T>::new(out_rows, out_cols);
        for j in 0..self.cols() {
            for i in 0..self.rows() {
                output[(j, i)] = self[(i, j)].clone();
            }
        }
        output
    }

    fn transpose_prealloc<M>(&self, output: &mut M)
    where
        M: MatrixTraitMut<T>,
        T: Default + Clone,
    {
        debug_assert_eq!(output.rows(), self.cols());
        debug_assert_eq!(output.cols(), self.rows());
        for j in 0..self.cols() {
            for i in 0..self.rows() {
                output[(j, i)] = self[(i, j)].clone();
            }
        }
    }

    fn transpose_arena<'a>(&self, arena: &Arena<'a>) -> Result<MatrixViewMut<'a, T>, ArenaError>
    where
        T: Default + Clone,
    {
        let out_rows = self.cols();
        let out_cols = self.rows();
        let mut output = MatrixViewMut::<T>::from_arena(out_rows, out_cols, arena)?;
        for j in 0..self.cols() {
            for i in 0..self.rows() {
                output[(j, i)] = self[(i, j)].clone();
            }
        }
        Ok(output)
    }
}

// `MatrixTraitMut` for the owned matrix and the mutable view.
macro_rules! impl_matrix_trait_mut {
    ($arr: ty) => {
impl<T> MatrixTraitMut<T> for $arr {
    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }

    fn iter_col_mut(&mut self) -> std::slice::ChunksExactMut<'_, T> {
        let rows = self.rows;
        self.get_data_mut().chunks_exact_mut(rows)
    }

    fn view_reshaped_array3d_mut(&mut self, dims: (usize, usize, usize)) -> Array3DViewMut<'_, T> {
        let (dim0, dim1, dim2) = dims;
        let length = self.rows * self.cols;
        let new_length = dim0 * dim1 * dim2;
        let data = self.get_data_mut();
        if new_length == length {
            Array3DViewMut::<T>::new(dim0, dim1, dim2, data)
        } else {
            panic!(
                "Matrix dimensions do not match {} dimensions",
                stringify!(Array3DView)
            );
        }
    }

    fn fill(&mut self, val: T)
    where
        T: Clone,
    {
        self.get_data_mut().fill(val)
    }

    fn get_col_mut(&mut self, l0: usize) -> &mut [T] {
        let rows = self.rows;
        let offset  = self.compute_index(0, l0);
        &mut self.get_data_mut()[offset..offset + rows]
    }

    fn copy_upper_to_lower(&mut self)
    where
        T: Clone + Copy,
    {
        debug_assert_eq!(self.rows, self.cols);
        let n = self.cols;
        for j in 0..n {
            for i in (j + 1)..n {
                self[(i, j)] = self[(j, i)];
            }
        }
    }

    fn copy_lower_to_upper(&mut self)
    where
        T: Clone + Copy,
    {
        debug_assert_eq!(self.rows, self.cols);
        let n = self.cols;
        for j in 0..n {
            for i in 0..j {
                self[(i, j)] = self[(j, i)];
            }
        }
    }
}
    };
}
impl_matrix_trait_mut!(Matrix<T>);
impl_matrix_trait_mut!(MatrixViewMut<'_, T>);




#[cfg(not(feature = "unchecked"))]
impl<D> Index<(usize, usize)> for MatrixStruct<D>
where
    Self: MatrixTrait<D::T>,
    D: DataContainer
{
    type Output = D::T;
    fn index(&self, ij: (usize, usize)) -> &Self::Output {
        debug_check_index!([ij.0, ij.1], [self.rows(), self.cols()]);
        let index = self.compute_index(ij.0, ij.1);
        &self.get_data()[index]
    }
}

#[cfg(feature = "unchecked")]
impl<D> Index<(usize, usize)> for MatrixStruct<D>
where
    Self: MatrixTrait<D::T>,
    D: DataContainer
{
    type Output = D::T;
    fn index(&self, ij: (usize, usize)) -> &Self::Output {
        debug_check_index!([ij.0, ij.1], [self.rows(), self.cols()]);
        let index = self.compute_index(ij.0, ij.1);
        unsafe { self.get_data().get_unchecked(index) }
    }
}

#[cfg(not(feature = "unchecked"))]
impl<D> IndexMut<(usize, usize)> for MatrixStruct<D>
where
    Self: MatrixTraitMut<D::T>,
    D: DataContainer
{
    fn index_mut(&mut self, ij: (usize, usize)) -> &mut D::T {
        debug_check_index!([ij.0, ij.1], [self.rows(), self.cols()]);
        let index = self.compute_index(ij.0, ij.1);
        &mut self.get_data_mut()[index]
    }
}

#[cfg(feature = "unchecked")]
impl<D> IndexMut<(usize, usize)> for MatrixStruct<D>
where
    Self: MatrixTraitMut<D::T>,
    D: DataContainer
{
    fn index_mut(&mut self, ij: (usize, usize)) -> &mut D::T {
        debug_check_index!([ij.0, ij.1], [self.rows(), self.cols()]);
        let index = self.compute_index(ij.0, ij.1);
        unsafe { self.get_data_mut().get_unchecked_mut(index) }
    }
}
