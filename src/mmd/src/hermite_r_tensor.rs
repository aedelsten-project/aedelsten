// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The Hermite R tensor's storage: blocks of `(t, u, v)`, `t + u + v <= n_max`, packed by `hermite_index`.

use std::ops::Index;
use arrays::matrix::{DataContainer, DataContainerMut};
use crate::hermite_expansion::*;
use arrays::scratch::{Arena, ArenaError};

/// One block of `(t, u, v)`, `t + u + v <= n_max`, packed by `hermite_index`.
#[allow(dead_code)]
pub struct HermiteTuvSliceStruct<D>
where
    D: DataContainer,
{
    tuv_max: usize,
    n_max: usize,
    data: D,
}
/// A block in heap storage.
pub type OwnedHermiteTuvSlice<T> = HermiteTuvSliceStruct<Box<[T]>>;
/// A block borrowed.
pub type HermiteTuvSliceView<'a, T> = HermiteTuvSliceStruct<&'a [T]>;
/// A block borrowed mutably.
pub type HermiteTuvSliceViewMut<'a, T> = HermiteTuvSliceStruct<&'a mut [T]>;

/// The flat data of a Hermite tensor or block.
pub trait HermiteRTensorTrait<T> {
    type T;
    /// The data.
    fn get_data(&self) -> &[T];
}

/// The flat data of a Hermite tensor or block, mutably.
pub trait HermiteRTensorTraitMut<T>: HermiteRTensorTrait<T> {
    type T;
    /// The data.
    fn get_data_mut(&mut self) -> &mut [T];
}

impl<T, D> HermiteRTensorTrait<T> for HermiteTuvSliceStruct<D>
where
    D: DataContainer<T = T>,
{
    type T = D::T;
    fn get_data(&self) -> &[D::T] {
        self.data.resolve()
    }
}

impl<T, D> HermiteRTensorTraitMut<T> for HermiteTuvSliceStruct<D>
where
    D: DataContainerMut<T = T>,
{
    type T = D::T;
    fn get_data_mut(&mut self) -> &mut [D::T] {
        self.data.resolve_mut()
    }
}

impl<D> HermiteTuvSliceStruct<D>
where
    D: DataContainer,
{
    fn compute_index(&self, t: usize, u: usize, v: usize) -> usize {
        hermite_index(t, u, v, self.n_max)
    }
}

impl<T> OwnedHermiteTuvSlice<T> {
    /// A block of default values for `t + u + v <= n_max`.
    pub fn new(n_max: usize) -> OwnedHermiteTuvSlice<T>
    where
        T: Default + Clone,
    {
        let tuv_max = hermite_count(n_max);
        OwnedHermiteTuvSlice {
            tuv_max,
            n_max,
            data: vec![Default::default(); tuv_max].into_boxed_slice(),
        }
    }
}

impl<'a, T> HermiteTuvSliceView<'a, T> {
    /// The block of `n_max` (`tuv_max` entries) over `data`.
    pub fn new(tuv_max: usize, n_max: usize, data: &'a [T]) -> HermiteTuvSliceView<'a, T> {
        HermiteTuvSliceView {
            tuv_max,
            n_max,
            data,
        }
    }

    /// The entries `t = 0..=n_max - u - v` of `(u, v)`, contiguous.
    pub fn get_t_slice(&self, u: usize, v: usize) -> &[T] {
        let start = self.compute_index(0, u, v);
        &self.get_data()[start..start + (self.n_max - u - v + 1)]
    }
}

impl<'a, T> HermiteTuvSliceViewMut<'a, T> {
    /// The block of `n_max` (`tuv_max` entries) over `data`.
    pub fn new(tuv_max: usize, n_max: usize, data: &'a mut [T]) -> HermiteTuvSliceViewMut<'a, T> {
        HermiteTuvSliceViewMut {
            tuv_max,
            n_max,
            data,
        }
    }

    /// A block of default values for `t + u + v <= n_max`, from the arena.
    pub fn from_arena(n_max: usize, arena: &Arena<'a>) -> Result<HermiteTuvSliceViewMut<'a, T>, ArenaError>
    where
        T: Default + Clone,
    {
        let tuv_max = hermite_count(n_max);
        let data = arena.alloc::<T>(tuv_max)?;
        data.fill(Default::default());
        Ok(HermiteTuvSliceViewMut { tuv_max, n_max, data })
    }

    /// The entries `t = 0..=n_max - u - v` of `(u, v)`, contiguous.
    pub fn get_t_slice(&self, u: usize, v: usize) -> &[T] {
        let start = self.compute_index(0, u, v);
        &self.get_data()[start..start + (self.n_max - u - v + 1)]
    }

    /// `get_t_slice`, mutably.
    pub fn get_t_slice_mut(&mut self, u: usize, v: usize) -> &mut [T] {
        let start = self.compute_index(0, u, v);
        let n_max = self.n_max;
        &mut self.get_data_mut()[start..start + (n_max - u - v + 1)]
    }

}

#[cfg(not(feature = "unchecked"))]
impl<'a, D> Index<(usize, usize, usize)> for HermiteTuvSliceStruct<D>
where
    D: DataContainer,
{
    type Output = D::T;
    fn index(&self, ij: (usize, usize, usize)) -> &D::T {
        let index = self.compute_index(ij.0, ij.1, ij.2);
        &self.get_data()[index]
    }
}

#[cfg(feature = "unchecked")]
impl<'a, D> Index<(usize, usize, usize)> for HermiteTuvSliceStruct<D>
where
    D: DataContainer,
{
    type Output = D::T;
    fn index(&self, ij: (usize, usize, usize)) -> &D::T {
        let index = self.compute_index(ij.0, ij.1, ij.2);
        unsafe { self.get_data().get_unchecked(index) }
    }
}
