// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The 3-dimensional array, column-major (`dim0` varies fastest): owned (`Array3D`), a view (`Array3DView`) and a
//! mutable view (`Array3DViewMut`). Everything is defined once in `Array3DTrait` / `Array3DTraitMut`, on top of
//! `dims` and the data; the index arithmetic is in `layout`.

use std::ops::{Index, IndexMut};

use crate::layout::{block, element, element_mut, grouped, offset};
use crate::matrix::{DataContainer, DataContainerMut, Matrix, MatrixView, MatrixViewMut, StridedMatrixView, StridedMatrixViewMut};
use crate::array4d::{Array4D, Array4DViewMut};
use crate::array5d::{Array5D, Array5DViewMut};
use crate::array6d::{Array6D, Array6DViewMut};
use crate::array7d::{Array7D, Array7DViewMut};
use crate::array8d::{Array8D, Array8DViewMut};
use crate::vector_math::add_in_place;
use crate::scratch::{checked_product, Arena, ArenaError};

/// The 3-dimensional array, column-major (`dim0` varies fastest), over data `D`: owned or borrowed.
#[derive(Debug, PartialEq)]
pub struct Array3DStruct<D: DataContainer> {
    dims: [usize; 3],
    data: D,
}
/// An owned 3-dimensional array.
pub type Array3D<T> = Array3DStruct<Box<[T]>>;
/// A view of a 3-dimensional array.
pub type Array3DView<'a, T> = Array3DStruct<&'a [T]>;
/// A mutable view of a 3-dimensional array.
pub type Array3DViewMut<'a, T> = Array3DStruct<&'a mut [T]>;

/// Reading a 3-dimensional array, owned or a view.
pub trait Array3DTrait<T>: Index<(usize, usize, usize), Output = T> {
    /// The dimensions.
    fn dims(&self) -> [usize; 3];
    /// The elements, column-major.
    fn get_data(&self) -> &[T];

    /// The whole array as a view (e.g. to hand an owned array to code that takes views).
    fn view(&self) -> Array3DView<'_, T> {
        Array3DStruct { dims: self.dims(), data: self.get_data() }
    }

    /// The size of dimension 0.
    fn dim0(&self) -> usize {
        self.dims()[0]
    }

    /// The size of dimension 1.
    fn dim1(&self) -> usize {
        self.dims()[1]
    }

    /// The size of dimension 2.
    fn dim2(&self) -> usize {
        self.dims()[2]
    }

    /// The matrices along the last dimension, in order.
    fn iter<'a>(&'a self) -> impl Iterator<Item = MatrixView<'a, T>>
    where
        T: 'a,
    {
        let d = self.dims();
        let len: usize = d[..2].iter().product();
        self.get_data().chunks_exact(len).map(move |c| MatrixView::new(d[0], d[1], c))
    }

    /// The contiguous fibre along `dim0` at the other indices.
    fn get_fibre(&self, idx: (usize, usize)) -> &[T] {
        let d = self.dims();
        let start = offset(&d, &[0, idx.0, idx.1]);
        &self.get_data()[start..start + d[0]]
    }

    /// The contiguous matrix at the trailing indices `idx`.
    fn view_sub_matrix(&self, idx: usize) -> MatrixView<'_, T> {
        let d = self.dims();
        MatrixView::new(d[0], d[1], &self.get_data()[block(&d, &[idx])])
    }

    /// The array as a matrix, adjacent dimensions merged (`groups` counts the dimensions in each new one); no copy.
    fn view_as_matrix(&self, groups: (usize, usize)) -> MatrixView<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1]);
        MatrixView::new(d[0], d[1], self.get_data())
    }
}

/// Writing a 3-dimensional array, owned or a mutable view.
pub trait Array3DTraitMut<T>: Array3DTrait<T> + IndexMut<(usize, usize, usize)> {
    /// The elements, column-major, to write.
    fn get_data_mut(&mut self) -> &mut [T];

    /// The whole array as a mutable view; on a view, a reborrow.
    fn view_mut(&mut self) -> Array3DViewMut<'_, T> {
        Array3DStruct { dims: self.dims(), data: self.get_data_mut() }
    }

    /// Every element set to `val`.
    fn fill(&mut self, val: T)
    where
        T: Clone,
    {
        self.get_data_mut().fill(val)
    }

    /// As `iter`, writable.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = MatrixViewMut<'a, T>>
    where
        T: 'a,
    {
        let d = self.dims();
        let len: usize = d[..2].iter().product();
        self.get_data_mut().chunks_exact_mut(len).map(move |c| MatrixViewMut::new(d[0], d[1], c))
    }

    /// As `get_fibre`, writable.
    fn get_fibre_mut(&mut self, idx: (usize, usize)) -> &mut [T] {
        let d = self.dims();
        let start = offset(&d, &[0, idx.0, idx.1]);
        &mut self.get_data_mut()[start..start + d[0]]
    }

    /// As `view_sub_matrix`, writable.
    fn view_sub_matrix_mut(&mut self, idx: usize) -> MatrixViewMut<'_, T> {
        let d = self.dims();
        MatrixViewMut::new(d[0], d[1], &mut self.get_data_mut()[block(&d, &[idx])])
    }

    /// As `view_as_matrix`, writable.
    fn view_as_matrix_mut(&mut self, groups: (usize, usize)) -> MatrixViewMut<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1]);
        MatrixViewMut::new(d[0], d[1], self.get_data_mut())
    }
}

impl<T, D: DataContainer<T = T>> Array3DTrait<T> for Array3DStruct<D> {
    fn dims(&self) -> [usize; 3] {
        self.dims
    }

    fn get_data(&self) -> &[T] {
        self.data.resolve()
    }
}

impl<T, D: DataContainerMut<T = T>> Array3DTraitMut<T> for Array3DStruct<D> {
    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.resolve_mut()
    }
}

impl<D: DataContainer> Index<(usize, usize, usize)> for Array3DStruct<D> {
    type Output = D::T;
    #[inline(always)]
    fn index(&self, (i, j, k): (usize, usize, usize)) -> &D::T {
        debug_check_index!([i, j, k], self.dims);
        let d = &self.dims;
        element(self.data.resolve(), i + d[0] * (j + d[1] * k))
    }
}

impl<D: DataContainerMut> IndexMut<(usize, usize, usize)> for Array3DStruct<D> {
    #[inline(always)]
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut D::T {
        debug_check_index!([i, j, k], self.dims);
        let d = &self.dims;
        element_mut(self.data.resolve_mut(), i + d[0] * (j + d[1] * k))
    }
}

impl<T> Array3D<T> {
    /// An array of the given dimensions, every element `T::default()`.
    pub fn new(dim0: usize, dim1: usize, dim2: usize) -> Self
    where
        T: Default + Clone,
    {
        let dims = [dim0, dim1, dim2];
        Array3D { dims, data: vec![T::default(); dims.iter().product()].into_boxed_slice() }
    }

    /// Takes `data` (exactly as many elements as the dimensions hold, column-major) as an array of the given
    /// dimensions.
    pub fn new_from_vec(dim0: usize, dim1: usize, dim2: usize, data: Box<[T]>) -> Self {
        let dims = [dim0, dim1, dim2];
        debug_assert_eq!(dims.iter().product::<usize>(), data.len(), "Array3D: {dims:?} for {} values", data.len());
        Array3D { dims, data }
    }

    /// The elements, column-major.
    pub fn into_data(self) -> Box<[T]> {
        self.data
    }

    /// As `view_as_matrix`, consuming the array.
    pub fn into_matrix(self, groups: (usize, usize)) -> Matrix<T> {
        let d = grouped(&self.dims, [groups.0, groups.1]);
        Matrix::new_from_vec(d[0], d[1], self.data)
    }

    crate::impl_into_array!(plain, into_reshaped_array4d, Array4D, dim0, dim1, dim2, dim3);
    crate::impl_into_array!(plain, into_reshaped_array5d, Array5D, dim0, dim1, dim2, dim3, dim4);
    crate::impl_into_array!(plain, into_reshaped_array6d, Array6D, dim0, dim1, dim2, dim3, dim4, dim5);
    crate::impl_into_array!(plain, into_reshaped_array7d, Array7D, dim0, dim1, dim2, dim3, dim4, dim5, dim6);
    crate::impl_into_array!(plain, into_reshaped_array8d, Array8D, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
}

impl<'a, T> Array3DView<'a, T> {
    /// `data` (exactly as many elements as the dimensions hold) as an array of the given dimensions.
    pub fn new(dim0: usize, dim1: usize, dim2: usize, data: &'a [T]) -> Self {
        let dims = [dim0, dim1, dim2];
        debug_assert_eq!(dims.iter().product::<usize>(), data.len(), "Array3DView: {dims:?} for {} values", data.len());
        Array3DView { dims, data }
    }
}

impl<'a, T> Array3DViewMut<'a, T> {
    /// `data` (exactly as many elements as the dimensions hold) as an array of the given dimensions.
    pub fn new(dim0: usize, dim1: usize, dim2: usize, data: &'a mut [T]) -> Self {
        let dims = [dim0, dim1, dim2];
        debug_assert_eq!(dims.iter().product::<usize>(), data.len(), "Array3DViewMut: {dims:?} for {} values", data.len());
        Array3DViewMut { dims, data }
    }

    /// Zeroed, from the arena.
    pub fn from_arena(dim0: usize, dim1: usize, dim2: usize, arena: &Arena<'a>) -> Result<Self, ArenaError>
    where
        T: Default + Clone,
    {
        let dims = [dim0, dim1, dim2];
        let data = arena.alloc::<T>(checked_product(&dims)?)?;
        data.fill(T::default());
        Ok(Array3DViewMut { dims, data })
    }

    /// The underlying data, with the view's own lifetime (e.g. to relabel an arena buffer with other dimensions).
    pub fn into_data(self) -> &'a mut [T] {
        self.data
    }

    /// As `view_as_matrix_mut`, with the view's lifetime.
    pub fn into_matrix(self, groups: (usize, usize)) -> MatrixViewMut<'a, T> {
        let d = grouped(&self.dims, [groups.0, groups.1]);
        MatrixViewMut::new(d[0], d[1], self.data)
    }

    crate::impl_into_array!(muta, into_reshaped_array4d, Array4DViewMut, dim0, dim1, dim2, dim3);
    crate::impl_into_array!(muta, into_reshaped_array5d, Array5DViewMut, dim0, dim1, dim2, dim3, dim4);
    crate::impl_into_array!(muta, into_reshaped_array6d, Array6DViewMut, dim0, dim1, dim2, dim3, dim4, dim5);
    crate::impl_into_array!(muta, into_reshaped_array7d, Array7DViewMut, dim0, dim1, dim2, dim3, dim4, dim5, dim6);
    crate::impl_into_array!(muta, into_reshaped_array8d, Array8DViewMut, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
}

impl<D: DataContainerMut<T = f64>> Array3DStruct<D> {
    /// `view_dim1_matrix`, writable: the `dim0 x dim2` matrix at middle index `j`, written column by column
    /// (`StridedMatrixViewMut::iter_mut`), each column contiguous.
    pub fn view_dim1_matrix_mut(&mut self, j: usize) -> StridedMatrixViewMut<'_> {
        let [dim0, dim1, dim2] = self.dims;
        assert!(j < dim1, "view_dim1_matrix_mut: index {j} of {dim1}");
        StridedMatrixViewMut::new(dim0, dim2, dim0 * dim1, &mut self.data.resolve_mut()[j * dim0..])
    }

    /// Adds `other`, of the same dimensions, element by element.
    pub fn add_in_place<B>(&mut self, other: &B)
    where
        B: Array3DTrait<f64>,
    {
        add_in_place(self.get_data_mut(), other.get_data());
    }
}

impl<D: DataContainer<T = f64>> Array3DStruct<D> {
    /// The `dim0 x dim2` matrix at middle index `j` (`self[(i, j, k)]` at row `i`, column `k`): the counterpart of
    /// `view_sub_matrix`, which fixes the last index. Its columns are `dim0 * dim1` apart, so it is not contiguous:
    /// a `StridedMatrixView`, a factor of products, with no copy.
    pub fn view_dim1_matrix(&self, j: usize) -> StridedMatrixView<'_> {
        let [dim0, dim1, dim2] = self.dims;
        assert!(j < dim1, "view_dim1_matrix: index {j} of {dim1}");
        StridedMatrixView::new(dim0, dim2, dim0 * dim1, &self.data.resolve()[j * dim0..])
    }
}

/// `add_in_place` for code generic over owned arrays and mutable views, as `matrix::MatrixMathMut` is for matrices.
pub trait Array3DMathMut<T>: Array3DTraitMut<T> {
    /// Adds `other`, of the same dimensions, element by element.
    fn add_in_place<B: Array3DTrait<T>>(&mut self, other: &B);
}

impl<D: DataContainerMut<T = f64>> Array3DMathMut<f64> for Array3DStruct<D> {
    fn add_in_place<B: Array3DTrait<f64>>(&mut self, other: &B) {
        Array3DStruct::add_in_place(self, other)
    }
}

#[cfg(test)]
mod view_dim1_matrix_tests {
    use super::*;
    use crate::matrix::{ColumnFactor, Factor, MatrixTrait};

    #[test]
    fn a_writable_dim1_matrix_view_writes_exactly_its_slice() {
        // A 4 x 3 x 5 array; slice 1 is written, every other element keeps its value.
        let (d0, d1, d2) = (4, 3, 5);
        let mut array = Array3D::<f64>::new(d0, d1, d2);
        array.fill(-1.0);
        let mut slice = array.view_dim1_matrix_mut(1);
        assert_eq!(slice.shape(), (d0, d2));
        let mut n_columns = 0;
        for (k, column) in slice.iter_mut().enumerate() {
            assert_eq!(column.len(), d0);
            for (i, x) in column.iter_mut().enumerate() {
                *x = (10 * k + i) as f64;
            }
            n_columns += 1;
        }
        assert_eq!(n_columns, d2);
        for k in 0..d2 {
            for j in 0..d1 {
                for i in 0..d0 {
                    let expected = if j == 1 { (10 * k + i) as f64 } else { -1.0 };
                    assert_eq!(array[(i, j, k)], expected, "({i}, {j}, {k})");
                }
            }
        }
        // The last middle index: its last column ends the buffer.
        assert_eq!(array.view_dim1_matrix_mut(d1 - 1).iter_mut().map(|column| column.len()).collect::<Vec<_>>(), [d0; 5]);
    }

    #[test]
    #[should_panic(expected = "view_dim1_matrix_mut")]
    fn a_writable_dim1_matrix_view_out_of_range_panics() {
        Array3D::<f64>::new(2, 3, 4).view_dim1_matrix_mut(3);
    }

    #[test]
    fn a_dim1_matrix_view_multiplies_and_reads_like_a_copied_one() {
        // A 4 x 3 x 5 array: the slice at middle index v is 4 x 5, its columns 12 apart.
        let (d0, d1, d2) = (4, 3, 5);
        let data: Vec<f64> = (0..d0 * d1 * d2).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        let array = Array3DView::new(d0, d1, d2, &data);
        let b = Matrix::<f64>::new_from_vec(d2, 2, (0..d2 * 2).map(|i| 0.5 * i as f64 - 1.0).collect::<Vec<_>>().into_boxed_slice());
        for v in 0..d1 {
            let mut copied = Matrix::<f64>::new(d0, d2);
            for k in 0..d2 {
                for i in 0..d0 {
                    copied[(i, k)] = array[(i, v, k)];
                }
            }
            let slice = array.view_dim1_matrix(v);
            assert_eq!(slice.shape(), (d0, d2));
            for k in 0..d2 {
                assert_eq!(slice.column(k), copied.get_col(k));
            }
            let expected = copied.matmul(&b);
            let mut got = Matrix::<f64>::new(d0, 2);
            slice.matmul_scaled_prealloc(&b, &mut got, 1.0, 0.0);
            for (x, y) in got.get_data().iter().zip(expected.get_data()) {
                assert!((x - y).abs() < 1e-13, "slice {v}: {x} vs {y}");
            }
        }
    }
}
