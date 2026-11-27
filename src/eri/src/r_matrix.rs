// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The R matrix of a primitive quartet: Hermite Coulomb integrals indexed by the bra's and the ket's Hermite indices
//! `(t, u, v)`, as a `(bra tuv) x (ket tuv)` matrix, the bra index fastest.

use std::ops::{Index, IndexMut};
use arrays::matrix::*;
use mmd::hermite_r_tensor::*;
use mmd::hermite_expansion::*;
use arrays::scratch::{Arena, ArenaError};

/// The R matrix up to `bra_l_max` and `ket_l_max`, over data `D`: owned or borrowed.
#[derive(Debug, PartialEq)]
pub struct RMatrixStruct<D>
where
    D: DataContainer,
{
    bra_tuv_max: usize,
    ket_tuv_max: usize,
    bra_l_max: usize,
    ket_l_max: usize,
    data: D,
}
/// An owned R matrix.
pub type RMatrix<T> = RMatrixStruct<Box<[T]>>;
/// An R matrix in an arena buffer.
pub type BufferedRMatrix<'a, T> = RMatrixStruct<&'a mut [T]>;

/// Reading an R matrix.
pub trait RMatrixTrait<T>: Index<(usize, usize, usize, usize, usize, usize), Output = T> {
    /// The bra's highest Hermite order.
    fn get_bra_l_max(&self) -> usize;
    /// The ket's highest Hermite order.
    fn get_ket_l_max(&self) -> usize;
    /// The values, column-major.
    fn get_data(&self) -> &[T];
}

/// Writing an R matrix.
pub trait RMatrixTraitMut<T>: RMatrixTrait<T> + IndexMut<(usize, usize, usize, usize, usize, usize)> {
    /// The values, column-major, to write.
    fn get_data_mut(&mut self) -> &mut [T];
    /// The column of ket index `(kt, ku, kv)` as a Hermite slice over the bra's `tuv`.
    fn get_tuv_slice_mut(&mut self, kt: usize, ku: usize, kv: usize) -> HermiteTuvSliceViewMut<'_, T>;
}

impl<T, D> RMatrixTrait<T> for RMatrixStruct<D>
where
    D: DataContainer<T = T>,
{
    #[inline(always)]
    fn get_bra_l_max(&self) -> usize {
        self.bra_l_max
    }

    #[inline(always)]
    fn get_ket_l_max(&self) -> usize {
        self.ket_l_max
    }

    #[inline(always)]
    fn get_data(&self) -> &[T] {
        self.data.resolve()
    }
}

impl<T, D> RMatrixTraitMut<T> for RMatrixStruct<D>
where
    D: DataContainerMut<T = T>,
{
    #[inline(always)]
    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.resolve_mut()
    }

    #[inline(always)]
    fn get_tuv_slice_mut(&mut self, kt: usize, ku: usize, kv: usize) -> HermiteTuvSliceViewMut<'_, T> {
        let bra_l_max = self.bra_l_max;
        let bra_tuv_max = self.bra_tuv_max;
        let ket_index = hermite_index(kt, ku, kv, self.ket_l_max);
        let offset = ket_index * bra_tuv_max;
        let data = &mut self.data.resolve_mut()[offset..offset + bra_tuv_max];
        HermiteTuvSliceViewMut::<T>::new(bra_l_max, bra_l_max, data)
    }
}

impl<T, D> RMatrixStruct<D>
where
    D: DataContainer<T = T>,
{
    /// The values as the `(bra tuv) x (ket tuv)` matrix.
    pub fn view_as_matrix(&self) -> MatrixView<'_, T> {
        let rows = self.bra_tuv_max;
        let cols = self.ket_tuv_max;
        MatrixView::<T>::new(rows, cols, self.get_data())
    }

    fn compute_index(
        &self, t: usize, u: usize, v: usize, kt: usize, ku: usize, kv: usize
    ) -> usize {
        let bra_index = hermite_index(t, u, v, self.bra_l_max);
        let ket_index = hermite_index(kt, ku, kv, self.ket_l_max);
        ket_index * self.bra_tuv_max + bra_index
    }
}

impl<T> RMatrix<T> {
    /// A zeroed R matrix up to `bra_l_max` and `ket_l_max`.
    pub fn new(bra_l_max: usize, ket_l_max: usize) -> RMatrix<T>
    where
        T: Default + Clone,
    {
        let bra_tuv_max = hermite_count(bra_l_max);
        let ket_tuv_max = hermite_count(ket_l_max);
        RMatrix {
            bra_tuv_max,
            ket_tuv_max,
            bra_l_max,
            ket_l_max,
            data: vec![Default::default(); bra_tuv_max * ket_tuv_max].into_boxed_slice(),
        }
    }

    /// The values as an owned matrix.
    pub fn into_matrix(self) -> Matrix<T> {
        let rows = self.bra_tuv_max;
        let cols = self.ket_tuv_max;
        Matrix::<T>::new_from_vec(rows, cols, self.data)
    }
}

impl<'a, T> BufferedRMatrix<'a, T> {
    /// A zeroed R matrix up to `bra_l_max` and `ket_l_max`, from the arena.
    pub fn from_arena(
        bra_l_max: usize,
        ket_l_max: usize,
        arena: &Arena<'a>,
    ) -> Result<Self, ArenaError>
    where
        T: Default + Clone,
    {
        let bra_tuv_max = hermite_count(bra_l_max);
        let ket_tuv_max = hermite_count(ket_l_max);
        let buf_len = arrays::scratch::checked_product(&[bra_tuv_max, ket_tuv_max])?;

        let data_buf = arena.alloc::<T>(buf_len)?;

        data_buf.fill(T::default());

        Ok(BufferedRMatrix {
            bra_tuv_max,
            ket_tuv_max,
            bra_l_max,
            ket_l_max,
            data: data_buf,
        })
    }
}

#[cfg(not(feature = "unchecked"))]
impl<T, D> Index<(usize, usize, usize, usize, usize, usize)> for RMatrixStruct<D>
where
    Self: RMatrixTrait<T>,
    D: DataContainer<T = T>,
{
    type Output = T;
    fn index(&self, ij: (usize, usize, usize, usize, usize, usize)) -> &D::T {
        let index = self.compute_index(ij.0, ij.1, ij.2, ij.3, ij.4, ij.5);
        &self.get_data()[index]
    }
}

#[cfg(not(feature = "unchecked"))]
impl<T, D> IndexMut<(usize, usize, usize, usize, usize, usize)> for RMatrixStruct<D>
where
    Self: RMatrixTraitMut<T>,
    D: DataContainer<T = T>,
{
    fn index_mut(&mut self, ij: (usize, usize, usize, usize, usize, usize)) -> &mut D::T {
        let index = self.compute_index(ij.0, ij.1, ij.2, ij.3, ij.4, ij.5);
        &mut self.get_data_mut()[index]
    }
}

#[cfg(feature = "unchecked")]
impl<T, D> Index<(usize, usize, usize, usize, usize, usize)> for RMatrixStruct<D>
where
    Self: RMatrixTrait<T>,
    D: DataContainer<T = T>,
{
    type Output = T;
    fn index(&self, ij: (usize, usize, usize, usize, usize, usize)) -> &D::T {
        let index = self.compute_index(ij.0, ij.1, ij.2, ij.3, ij.4, ij.5);
        unsafe { self.get_data().get_unchecked(index) }
    }
}

#[cfg(feature = "unchecked")]
impl<T, D> IndexMut<(usize, usize, usize, usize, usize, usize)> for RMatrixStruct<D>
where
    Self: RMatrixTraitMut<T>,
    D: DataContainer<T = T>,
{
    fn index_mut(&mut self, ij: (usize, usize, usize, usize, usize, usize)) -> &mut D::T {
        let index = self.compute_index(ij.0, ij.1, ij.2, ij.3, ij.4, ij.5);
        unsafe { self.get_data_mut().get_unchecked_mut(index) }
    }
}
