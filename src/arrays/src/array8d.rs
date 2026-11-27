// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The 8-dimensional array, column-major (`dim0` varies fastest): owned (`Array8D`), a view (`Array8DView`) and a
//! mutable view (`Array8DViewMut`). Everything is defined once in `Array8DTrait` / `Array8DTraitMut`, on top of
//! `dims` and the data; the index arithmetic is in `layout`.

use std::ops::{Index, IndexMut};

use crate::layout::{block, check_reshape, element, element_mut, grouped, offset};
use crate::matrix::{DataContainer, DataContainerMut, Matrix, MatrixView, MatrixViewMut};
use crate::array3d::{Array3D, Array3DView, Array3DViewMut};
use crate::array4d::{Array4D, Array4DView, Array4DViewMut};
use crate::array5d::{Array5D, Array5DView, Array5DViewMut};
use crate::array6d::{Array6D, Array6DView, Array6DViewMut};
use crate::array7d::{Array7D, Array7DView, Array7DViewMut};
use crate::vector_math::add_in_place;
use crate::scratch::{checked_product, Arena, ArenaError};

/// The 8-dimensional array, column-major (`dim0` varies fastest), over data `D`: owned or borrowed.
#[derive(Debug, PartialEq)]
pub struct Array8DStruct<D: DataContainer> {
    dims: [usize; 8],
    data: D,
}
/// An owned 8-dimensional array.
pub type Array8D<T> = Array8DStruct<Box<[T]>>;
/// A view of an 8-dimensional array.
pub type Array8DView<'a, T> = Array8DStruct<&'a [T]>;
/// A mutable view of an 8-dimensional array.
pub type Array8DViewMut<'a, T> = Array8DStruct<&'a mut [T]>;

/// Reading an 8-dimensional array, owned or a view.
pub trait Array8DTrait<T>: Index<(usize, usize, usize, usize, usize, usize, usize, usize), Output = T> {
    /// The dimensions.
    fn dims(&self) -> [usize; 8];
    /// The elements, column-major.
    fn get_data(&self) -> &[T];

    /// The whole array as a view (e.g. to hand an owned array to code that takes views).
    fn view(&self) -> Array8DView<'_, T> {
        Array8DStruct { dims: self.dims(), data: self.get_data() }
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

    /// The size of dimension 3.
    fn dim3(&self) -> usize {
        self.dims()[3]
    }

    /// The size of dimension 4.
    fn dim4(&self) -> usize {
        self.dims()[4]
    }

    /// The size of dimension 5.
    fn dim5(&self) -> usize {
        self.dims()[5]
    }

    /// The size of dimension 6.
    fn dim6(&self) -> usize {
        self.dims()[6]
    }

    /// The size of dimension 7.
    fn dim7(&self) -> usize {
        self.dims()[7]
    }

    /// The 7D arrays along the last dimension, in order.
    fn iter<'a>(&'a self) -> impl Iterator<Item = Array7DView<'a, T>>
    where
        T: 'a,
    {
        let d = self.dims();
        let len: usize = d[..7].iter().product();
        self.get_data().chunks_exact(len).map(move |c| Array7DView::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], c))
    }

    /// The contiguous fibre along `dim0` at the other indices.
    fn get_fibre(&self, idx: (usize, usize, usize, usize, usize, usize, usize)) -> &[T] {
        let d = self.dims();
        let start = offset(&d, &[0, idx.0, idx.1, idx.2, idx.3, idx.4, idx.5, idx.6]);
        &self.get_data()[start..start + d[0]]
    }

    /// The contiguous matrix at the trailing indices `idx`.
    fn view_sub_matrix(&self, idx: (usize, usize, usize, usize, usize, usize)) -> MatrixView<'_, T> {
        let d = self.dims();
        MatrixView::new(d[0], d[1], &self.get_data()[block(&d, &[idx.0, idx.1, idx.2, idx.3, idx.4, idx.5])])
    }

    /// The contiguous 3-dimensional array at the trailing indices `idx`.
    fn view_sub_array3d(&self, idx: (usize, usize, usize, usize, usize)) -> Array3DView<'_, T> {
        let d = self.dims();
        Array3DView::new(d[0], d[1], d[2], &self.get_data()[block(&d, &[idx.0, idx.1, idx.2, idx.3, idx.4])])
    }

    /// The contiguous 4-dimensional array at the trailing indices `idx`.
    fn view_sub_array4d(&self, idx: (usize, usize, usize, usize)) -> Array4DView<'_, T> {
        let d = self.dims();
        Array4DView::new(d[0], d[1], d[2], d[3], &self.get_data()[block(&d, &[idx.0, idx.1, idx.2, idx.3])])
    }

    /// The contiguous 5-dimensional array at the trailing indices `idx`.
    fn view_sub_array5d(&self, idx: (usize, usize, usize)) -> Array5DView<'_, T> {
        let d = self.dims();
        Array5DView::new(d[0], d[1], d[2], d[3], d[4], &self.get_data()[block(&d, &[idx.0, idx.1, idx.2])])
    }

    /// The contiguous 6-dimensional array at the trailing indices `idx`.
    fn view_sub_array6d(&self, idx: (usize, usize)) -> Array6DView<'_, T> {
        let d = self.dims();
        Array6DView::new(d[0], d[1], d[2], d[3], d[4], d[5], &self.get_data()[block(&d, &[idx.0, idx.1])])
    }

    /// The contiguous 7-dimensional array at the trailing indices `idx`.
    fn view_sub_array7d(&self, idx: usize) -> Array7DView<'_, T> {
        let d = self.dims();
        Array7DView::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], &self.get_data()[block(&d, &[idx])])
    }

    /// The array as a matrix, adjacent dimensions merged (`groups` counts the dimensions in each new one); no copy.
    fn view_as_matrix(&self, groups: (usize, usize)) -> MatrixView<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1]);
        MatrixView::new(d[0], d[1], self.get_data())
    }

    /// The array as a 3-dimensional array, adjacent dimensions merged (`groups` counts the dimensions in each new one);
    /// no copy.
    fn view_as_array3d(&self, groups: (usize, usize, usize)) -> Array3DView<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2]);
        Array3DView::new(d[0], d[1], d[2], self.get_data())
    }

    /// The array as a 4-dimensional array, adjacent dimensions merged (`groups` counts the dimensions in each new one);
    /// no copy.
    fn view_as_array4d(&self, groups: (usize, usize, usize, usize)) -> Array4DView<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2, groups.3]);
        Array4DView::new(d[0], d[1], d[2], d[3], self.get_data())
    }

    /// The array as a 5-dimensional array, adjacent dimensions merged (`groups` counts the dimensions in each new one);
    /// no copy.
    fn view_as_array5d(&self, groups: (usize, usize, usize, usize, usize)) -> Array5DView<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2, groups.3, groups.4]);
        Array5DView::new(d[0], d[1], d[2], d[3], d[4], self.get_data())
    }

    /// The array as a 6-dimensional array, adjacent dimensions merged (`groups` counts the dimensions in each new one);
    /// no copy.
    fn view_as_array6d(&self, groups: (usize, usize, usize, usize, usize, usize)) -> Array6DView<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2, groups.3, groups.4, groups.5]);
        Array6DView::new(d[0], d[1], d[2], d[3], d[4], d[5], self.get_data())
    }

    /// The array as a 7-dimensional array, adjacent dimensions merged (`groups` counts the dimensions in each new one);
    /// no copy.
    fn view_as_array7d(&self, groups: (usize, usize, usize, usize, usize, usize, usize)) -> Array7DView<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2, groups.3, groups.4, groups.5, groups.6]);
        Array7DView::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], self.get_data())
    }

    /// The elements as a 3-dimensional array of dimensions `dims`, with as many elements; no copy.
    fn view_reshaped_array3d(&self, dims: (usize, usize, usize)) -> Array3DView<'_, T> {
        let d = [dims.0, dims.1, dims.2];
        check_reshape(&self.dims(), &d);
        Array3DView::new(d[0], d[1], d[2], self.get_data())
    }
}

/// Writing an 8-dimensional array, owned or a mutable view.
pub trait Array8DTraitMut<T>: Array8DTrait<T> + IndexMut<(usize, usize, usize, usize, usize, usize, usize, usize)> {
    /// The elements, column-major, to write.
    fn get_data_mut(&mut self) -> &mut [T];

    /// The whole array as a mutable view; on a view, a reborrow.
    fn view_mut(&mut self) -> Array8DViewMut<'_, T> {
        Array8DStruct { dims: self.dims(), data: self.get_data_mut() }
    }

    /// Every element set to `val`.
    fn fill(&mut self, val: T)
    where
        T: Clone,
    {
        self.get_data_mut().fill(val)
    }

    /// As `iter`, writable.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = Array7DViewMut<'a, T>>
    where
        T: 'a,
    {
        let d = self.dims();
        let len: usize = d[..7].iter().product();
        self.get_data_mut().chunks_exact_mut(len).map(move |c| Array7DViewMut::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], c))
    }

    /// As `get_fibre`, writable.
    fn get_fibre_mut(&mut self, idx: (usize, usize, usize, usize, usize, usize, usize)) -> &mut [T] {
        let d = self.dims();
        let start = offset(&d, &[0, idx.0, idx.1, idx.2, idx.3, idx.4, idx.5, idx.6]);
        &mut self.get_data_mut()[start..start + d[0]]
    }

    /// As `view_sub_matrix`, writable.
    fn view_sub_matrix_mut(&mut self, idx: (usize, usize, usize, usize, usize, usize)) -> MatrixViewMut<'_, T> {
        let d = self.dims();
        MatrixViewMut::new(d[0], d[1], &mut self.get_data_mut()[block(&d, &[idx.0, idx.1, idx.2, idx.3, idx.4, idx.5])])
    }

    /// As `view_sub_array3d`, writable.
    fn view_sub_array3d_mut(&mut self, idx: (usize, usize, usize, usize, usize)) -> Array3DViewMut<'_, T> {
        let d = self.dims();
        Array3DViewMut::new(d[0], d[1], d[2], &mut self.get_data_mut()[block(&d, &[idx.0, idx.1, idx.2, idx.3, idx.4])])
    }

    /// As `view_sub_array4d`, writable.
    fn view_sub_array4d_mut(&mut self, idx: (usize, usize, usize, usize)) -> Array4DViewMut<'_, T> {
        let d = self.dims();
        Array4DViewMut::new(d[0], d[1], d[2], d[3], &mut self.get_data_mut()[block(&d, &[idx.0, idx.1, idx.2, idx.3])])
    }

    /// As `view_sub_array5d`, writable.
    fn view_sub_array5d_mut(&mut self, idx: (usize, usize, usize)) -> Array5DViewMut<'_, T> {
        let d = self.dims();
        Array5DViewMut::new(d[0], d[1], d[2], d[3], d[4], &mut self.get_data_mut()[block(&d, &[idx.0, idx.1, idx.2])])
    }

    /// As `view_sub_array6d`, writable.
    fn view_sub_array6d_mut(&mut self, idx: (usize, usize)) -> Array6DViewMut<'_, T> {
        let d = self.dims();
        Array6DViewMut::new(d[0], d[1], d[2], d[3], d[4], d[5], &mut self.get_data_mut()[block(&d, &[idx.0, idx.1])])
    }

    /// As `view_sub_array7d`, writable.
    fn view_sub_array7d_mut(&mut self, idx: usize) -> Array7DViewMut<'_, T> {
        let d = self.dims();
        Array7DViewMut::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], &mut self.get_data_mut()[block(&d, &[idx])])
    }

    /// As `view_as_matrix`, writable.
    fn view_as_matrix_mut(&mut self, groups: (usize, usize)) -> MatrixViewMut<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1]);
        MatrixViewMut::new(d[0], d[1], self.get_data_mut())
    }

    /// As `view_as_array3d`, writable.
    fn view_as_array3d_mut(&mut self, groups: (usize, usize, usize)) -> Array3DViewMut<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2]);
        Array3DViewMut::new(d[0], d[1], d[2], self.get_data_mut())
    }

    /// As `view_as_array4d`, writable.
    fn view_as_array4d_mut(&mut self, groups: (usize, usize, usize, usize)) -> Array4DViewMut<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2, groups.3]);
        Array4DViewMut::new(d[0], d[1], d[2], d[3], self.get_data_mut())
    }

    /// As `view_as_array5d`, writable.
    fn view_as_array5d_mut(&mut self, groups: (usize, usize, usize, usize, usize)) -> Array5DViewMut<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2, groups.3, groups.4]);
        Array5DViewMut::new(d[0], d[1], d[2], d[3], d[4], self.get_data_mut())
    }

    /// As `view_as_array6d`, writable.
    fn view_as_array6d_mut(&mut self, groups: (usize, usize, usize, usize, usize, usize)) -> Array6DViewMut<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2, groups.3, groups.4, groups.5]);
        Array6DViewMut::new(d[0], d[1], d[2], d[3], d[4], d[5], self.get_data_mut())
    }

    /// As `view_as_array7d`, writable.
    fn view_as_array7d_mut(&mut self, groups: (usize, usize, usize, usize, usize, usize, usize)) -> Array7DViewMut<'_, T> {
        let d = grouped(&self.dims(), [groups.0, groups.1, groups.2, groups.3, groups.4, groups.5, groups.6]);
        Array7DViewMut::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], self.get_data_mut())
    }

    /// As `view_reshaped_array3d`, writable.
    fn view_reshaped_array3d_mut(&mut self, dims: (usize, usize, usize)) -> Array3DViewMut<'_, T> {
        let d = [dims.0, dims.1, dims.2];
        check_reshape(&self.dims(), &d);
        Array3DViewMut::new(d[0], d[1], d[2], self.get_data_mut())
    }
}

impl<T, D: DataContainer<T = T>> Array8DTrait<T> for Array8DStruct<D> {
    fn dims(&self) -> [usize; 8] {
        self.dims
    }

    fn get_data(&self) -> &[T] {
        self.data.resolve()
    }
}

impl<T, D: DataContainerMut<T = T>> Array8DTraitMut<T> for Array8DStruct<D> {
    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.resolve_mut()
    }
}

impl<D: DataContainer> Index<(usize, usize, usize, usize, usize, usize, usize, usize)> for Array8DStruct<D> {
    type Output = D::T;
    #[inline(always)]
    fn index(&self, (i, j, k, l, m, n, o, p): (usize, usize, usize, usize, usize, usize, usize, usize)) -> &D::T {
        debug_check_index!([i, j, k, l, m, n, o, p], self.dims);
        let d = &self.dims;
        element(self.data.resolve(), i + d[0] * (j + d[1] * (k + d[2] * (l + d[3] * (m + d[4] * (n + d[5] * (o + d[6] * p)))))))
    }
}

impl<D: DataContainerMut> IndexMut<(usize, usize, usize, usize, usize, usize, usize, usize)> for Array8DStruct<D> {
    #[inline(always)]
    fn index_mut(&mut self, (i, j, k, l, m, n, o, p): (usize, usize, usize, usize, usize, usize, usize, usize)) -> &mut D::T {
        debug_check_index!([i, j, k, l, m, n, o, p], self.dims);
        let d = &self.dims;
        element_mut(self.data.resolve_mut(), i + d[0] * (j + d[1] * (k + d[2] * (l + d[3] * (m + d[4] * (n + d[5] * (o + d[6] * p)))))))
    }
}

impl<T> Array8D<T> {
    /// An array of the given dimensions, every element `T::default()`.
    pub fn new(dim0: usize, dim1: usize, dim2: usize, dim3: usize, dim4: usize, dim5: usize, dim6: usize, dim7: usize) -> Self
    where
        T: Default + Clone,
    {
        let dims = [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7];
        Array8D { dims, data: vec![T::default(); dims.iter().product()].into_boxed_slice() }
    }

    /// Takes `data` (exactly as many elements as the dimensions hold, column-major) as an array of the given
    /// dimensions.
    pub fn new_from_vec(dim0: usize, dim1: usize, dim2: usize, dim3: usize, dim4: usize, dim5: usize, dim6: usize, dim7: usize, data: Box<[T]>) -> Self {
        let dims = [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7];
        debug_assert_eq!(dims.iter().product::<usize>(), data.len(), "Array8D: {dims:?} for {} values", data.len());
        Array8D { dims, data }
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

    /// As `view_as_array3d`, consuming the array.
    pub fn into_array3d(self, groups: (usize, usize, usize)) -> Array3D<T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2]);
        Array3D::new_from_vec(d[0], d[1], d[2], self.data)
    }

    /// As `view_as_array4d`, consuming the array.
    pub fn into_array4d(self, groups: (usize, usize, usize, usize)) -> Array4D<T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2, groups.3]);
        Array4D::new_from_vec(d[0], d[1], d[2], d[3], self.data)
    }

    /// As `view_as_array5d`, consuming the array.
    pub fn into_array5d(self, groups: (usize, usize, usize, usize, usize)) -> Array5D<T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2, groups.3, groups.4]);
        Array5D::new_from_vec(d[0], d[1], d[2], d[3], d[4], self.data)
    }

    /// As `view_as_array6d`, consuming the array.
    pub fn into_array6d(self, groups: (usize, usize, usize, usize, usize, usize)) -> Array6D<T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2, groups.3, groups.4, groups.5]);
        Array6D::new_from_vec(d[0], d[1], d[2], d[3], d[4], d[5], self.data)
    }

    /// As `view_as_array7d`, consuming the array.
    pub fn into_array7d(self, groups: (usize, usize, usize, usize, usize, usize, usize)) -> Array7D<T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2, groups.3, groups.4, groups.5, groups.6]);
        Array7D::new_from_vec(d[0], d[1], d[2], d[3], d[4], d[5], d[6], self.data)
    }
}

impl<'a, T> Array8DView<'a, T> {
    /// `data` (exactly as many elements as the dimensions hold) as an array of the given dimensions.
    pub fn new(dim0: usize, dim1: usize, dim2: usize, dim3: usize, dim4: usize, dim5: usize, dim6: usize, dim7: usize, data: &'a [T]) -> Self {
        let dims = [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7];
        debug_assert_eq!(dims.iter().product::<usize>(), data.len(), "Array8DView: {dims:?} for {} values", data.len());
        Array8DView { dims, data }
    }
}

impl<'a, T> Array8DViewMut<'a, T> {
    /// `data` (exactly as many elements as the dimensions hold) as an array of the given dimensions.
    pub fn new(dim0: usize, dim1: usize, dim2: usize, dim3: usize, dim4: usize, dim5: usize, dim6: usize, dim7: usize, data: &'a mut [T]) -> Self {
        let dims = [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7];
        debug_assert_eq!(dims.iter().product::<usize>(), data.len(), "Array8DViewMut: {dims:?} for {} values", data.len());
        Array8DViewMut { dims, data }
    }

    /// Zeroed, from the arena.
    pub fn from_arena(dim0: usize, dim1: usize, dim2: usize, dim3: usize, dim4: usize, dim5: usize, dim6: usize, dim7: usize, arena: &Arena<'a>) -> Result<Self, ArenaError>
    where
        T: Default + Clone,
    {
        let dims = [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7];
        let data = arena.alloc::<T>(checked_product(&dims)?)?;
        data.fill(T::default());
        Ok(Array8DViewMut { dims, data })
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

    /// As `view_as_array3d_mut`, with the view's lifetime.
    pub fn into_array3d(self, groups: (usize, usize, usize)) -> Array3DViewMut<'a, T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2]);
        Array3DViewMut::new(d[0], d[1], d[2], self.data)
    }

    /// As `view_as_array4d_mut`, with the view's lifetime.
    pub fn into_array4d(self, groups: (usize, usize, usize, usize)) -> Array4DViewMut<'a, T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2, groups.3]);
        Array4DViewMut::new(d[0], d[1], d[2], d[3], self.data)
    }

    /// As `view_as_array5d_mut`, with the view's lifetime.
    pub fn into_array5d(self, groups: (usize, usize, usize, usize, usize)) -> Array5DViewMut<'a, T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2, groups.3, groups.4]);
        Array5DViewMut::new(d[0], d[1], d[2], d[3], d[4], self.data)
    }

    /// As `view_as_array6d_mut`, with the view's lifetime.
    pub fn into_array6d(self, groups: (usize, usize, usize, usize, usize, usize)) -> Array6DViewMut<'a, T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2, groups.3, groups.4, groups.5]);
        Array6DViewMut::new(d[0], d[1], d[2], d[3], d[4], d[5], self.data)
    }

    /// As `view_as_array7d_mut`, with the view's lifetime.
    pub fn into_array7d(self, groups: (usize, usize, usize, usize, usize, usize, usize)) -> Array7DViewMut<'a, T> {
        let d = grouped(&self.dims, [groups.0, groups.1, groups.2, groups.3, groups.4, groups.5, groups.6]);
        Array7DViewMut::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], self.data)
    }
}

impl<D: DataContainerMut<T = f64>> Array8DStruct<D> {
    /// Adds `other`, of the same dimensions, element by element.
    pub fn add_in_place<B>(&mut self, other: &B)
    where
        B: Array8DTrait<f64>,
    {
        add_in_place(self.get_data_mut(), other.get_data());
    }
}
