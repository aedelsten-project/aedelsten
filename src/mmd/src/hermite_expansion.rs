// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The layouts of a shell pair's Hermite expansion coefficients, bra-side and ket-side, and the Hermite index
//! `hermite_index` of `(t, u, v)`.

use arrays::matrix::*;
use arrays::array3d::*;
use arrays::scratch::{Arena, ArenaError};
use std::ops::{Index, IndexMut};

/// The Hermite expansion coefficients of a ket shell pair, `(t, u, v)` fastest: index `(t, u, v, i, j)`, layout
/// `(tuv, i, j)`, `i < dim0`, `j < dim1`, `t + u + v <= l_max`.
pub struct KetExpansionCoefficients<D>
where
    D: DataContainer,
{
    dim0: usize,
    dim1: usize,
    l_max: usize,
    tuv_max: usize,
    data: D,
}
/// Ket coefficients in heap storage.
pub type OwnedKetExpansionCoefficients<T> = KetExpansionCoefficients<Box<[T]>>;
/// Ket coefficients in caller or arena storage.
pub type RefKetExpansionCoefficients<'a, T> = KetExpansionCoefficients<&'a mut [T]>;

/// The Hermite expansion coefficients of a bra shell pair, `(t, u, v)` slowest: index `(i, j, t, u, v)`, layout
/// `(i, j, tuv)`, `i < dim0`, `j < dim1`, `t + u + v <= l_max`.
pub struct BraExpansionCoefficients<D>
where
    D: DataContainer,
{
    dim0: usize,
    dim1: usize,
    l_max: usize,
    tuv_max: usize,
    data: D,
}
/// Bra coefficients in heap storage.
pub type OwnedBraExpansionCoefficients<T> = BraExpansionCoefficients<Box<[T]>>;
/// Bra coefficients in caller storage.
pub type RefBraExpansionCoefficients<'a, T> = BraExpansionCoefficients<&'a mut [T]>;

/// Read access to bra coefficients.
pub trait BraExpansionCoefficientsTrait<T>: Index<(usize, usize, usize, usize, usize), Output = T>{
    type T;
    /// The number of components `i`.
    fn dim0(&self) -> usize;
    /// The number of components `j`.
    fn dim1(&self) -> usize;
    /// The highest `t + u + v`.
    fn l_max(&self) -> usize;
    /// The number of `(t, u, v)`.
    fn tuv_max(&self) -> usize;
    /// The data.
    fn get_data(&self) -> &[T];
    /// The coefficients as a `(dim0 dim1) x tuv_max` matrix.
    fn view_as_matrix(&self) -> MatrixView<'_, T>
    where
        T: Default;
}

/// Write access to bra coefficients.
pub trait BraExpansionCoefficientsTraitMut<T>: BraExpansionCoefficientsTrait<T> + IndexMut<(usize, usize, usize, usize, usize), Output = T>{
    /// The data, mutably.
    fn get_data_mut(&mut self) -> &mut [T];
}

impl<T, D> BraExpansionCoefficientsTrait<T> for BraExpansionCoefficients<D>
where
    D: DataContainerMut<T = T>,
{
    type T = T;
    fn dim0(&self) -> usize {
        self.dim0
    }
    
    fn dim1(&self) -> usize {
        self.dim1
    }
    
    fn l_max(&self) -> usize {
        self.l_max
    }
    
    fn tuv_max(&self) -> usize {
        self.tuv_max
    }

    fn get_data(&self) -> &[T] {
        self.data.resolve()
    }

    fn view_as_matrix(&self) -> MatrixView<'_, T> {
        MatrixView::new(self.dim0 * self.dim1, self.tuv_max, self.get_data())
    }
}

impl<T, D> BraExpansionCoefficientsTraitMut<T> for BraExpansionCoefficients<D>
where
    D: DataContainerMut<T = T>,
{
    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.resolve_mut()
    }
}

impl<T, D> BraExpansionCoefficients<D>
where
    D: DataContainerMut<T = T>,
{
    fn compute_index(&self, i: usize, j: usize, t: usize, u: usize, v: usize) -> usize {
        let index_tuv = hermite_index(t, u, v, self.l_max);
        index_tuv * self.dim0 * self.dim1
            + j * self.dim0
            + i
    }

}


impl OwnedBraExpansionCoefficients<f64> {
    /// Zeroed coefficients.
    pub fn new(dim0: usize, dim1: usize, l_max: usize) -> Self {
        let tuv_max = hermite_count(l_max);
        let data = vec![0.0; dim0 * dim1 * tuv_max].into_boxed_slice();
        Self {
            dim0,
            dim1,
            l_max,
            tuv_max,
            data,
        }
    }

    /// The coefficients as a `(dim0 dim1) x tuv_max` matrix.
    pub fn into_matrix(self) -> Matrix<f64> {
        Matrix::new_from_vec(self.dim0 * self.dim1, self.tuv_max, self.data)
    }

}

impl<'a> RefBraExpansionCoefficients<'a, f64> {
    /// The coefficients over `data`.
    #[inline(always)]
    pub fn new(dim0: usize, dim1: usize, l_max: usize, data: &'a mut [f64]) -> Self {
        let tuv_max = hermite_count(l_max);
        Self {
            dim0,
            dim1,
            l_max,
            tuv_max,
            data,
        }
    }
}

/// Read access to ket coefficients.
pub trait KetExpansionCoefficientsTrait<T>: Index<(usize, usize, usize, usize, usize), Output = T> {
    type T;
    /// The number of components `i`.
    fn dim0(&self) -> usize;
    /// The number of components `j`.
    fn dim1(&self) -> usize;
    /// The highest `t + u + v`.
    fn l_max(&self) -> usize;
    /// The number of `(t, u, v)`.
    fn tuv_max(&self) -> usize;
    /// The entries `t = 0..t_max` of `(u, v)` for the components `(c1, c2)`, contiguous.
    fn get_t_slice(&self, t_max: usize, u: usize, v: usize, c1: usize, c2: usize) -> &[T];
    /// The data.
    fn get_data(&self) -> &[T];
    /// The coefficients as a `tuv_max x (dim0 dim1)` matrix.
    fn view_as_matrix(&self) -> MatrixView<'_, T>;
    /// The coefficients as a `tuv_max x dim0 x dim1` array.
    fn view_as_array3d(&self) -> Array3DView<'_, T>;
}

/// Write access to ket coefficients; the mutable forms of the read methods.
pub trait KetExpansionCoefficientsTraitMut<T>: KetExpansionCoefficientsTrait<T> + IndexMut<(usize, usize, usize, usize, usize), Output = T>{
    fn get_t_slice_mut(&mut self, t_max: usize, u: usize, v: usize, c1: usize, c2: usize) -> &mut [T];
    fn get_data_mut(&mut self) -> &mut [T];
    fn view_as_matrix_mut(&mut self) -> MatrixViewMut<'_, T>;
    fn view_as_array3d_mut(&mut self) -> Array3DViewMut<'_, T>;
}


impl<T, D> KetExpansionCoefficientsTrait<T> for KetExpansionCoefficients<D>
where
    D: DataContainerMut<T = T>,
{
    type T = T;
    fn dim0(&self) -> usize {
        self.dim0
    }
    
    fn dim1(&self) -> usize {
        self.dim1
    }
    
    fn l_max(&self) -> usize {
        self.l_max
    }
    
    fn tuv_max(&self) -> usize {
        self.tuv_max
    }

    fn get_t_slice(&self, t_max: usize, u: usize, v: usize, c1: usize, c2: usize) -> &[T] {
        let offset = self.compute_index(0, u, v, c1, c2);
        let last = self.compute_index(t_max, u, v, c1, c2);
        &self.get_data()[offset..last]
    }

    fn get_data(&self) -> &[T] {
        self.data.resolve()
    }
    
    fn view_as_matrix(&self) -> MatrixView<'_, T> {
        MatrixView::new(self.tuv_max, self.dim0 * self.dim1, self.get_data())
    }

    fn view_as_array3d(&self) -> Array3DView<'_, T> {
        Array3DView::new(self.tuv_max, self.dim0, self.dim1, self.get_data())
    }
}

impl<T, D> KetExpansionCoefficientsTraitMut<T> for KetExpansionCoefficients<D>
where
    D: DataContainerMut<T = T>,
{
    fn get_t_slice_mut(&mut self, t_max: usize, u: usize, v: usize, c1: usize, c2: usize) -> &mut [T] {
        let offset = self.compute_index(0, u, v, c1, c2);
        let last = self.compute_index(t_max, u, v, c1, c2);
        &mut self.get_data_mut()[offset..last]
    }

    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.resolve_mut()
    }

    fn view_as_matrix_mut(&mut self) -> MatrixViewMut<'_, T> {
        MatrixViewMut::new(self.tuv_max, self.dim0 * self.dim1, self.data.resolve_mut())
    }

    fn view_as_array3d_mut(&mut self) -> Array3DViewMut<'_, T> {
        Array3DViewMut::new(self.tuv_max, self.dim0, self.dim1, self.data.resolve_mut())
    }
}

impl<T, D> KetExpansionCoefficients<D> 
where
    D: DataContainerMut<T = T>,
{

    fn compute_index(&self, t: usize, u: usize, v: usize, i: usize, j: usize) -> usize {
        let index_tuv = hermite_index(t, u, v, self.l_max);
        j * self.tuv_max * self.dim0
            + i * self.tuv_max
            + index_tuv
    }
}

impl OwnedKetExpansionCoefficients<f64> {
    /// Zeroed coefficients.
    pub fn new(dim0: usize, dim1: usize, l_max: usize) -> Self {
        let tuv_max = hermite_count(l_max);
        let data = vec![0.0; dim0 * dim1 * tuv_max].into_boxed_slice();
        Self {
            dim0,
            dim1,
            l_max,
            tuv_max,
            data,
        }
    }

    /// The coefficients as a `tuv_max x (dim0 dim1)` matrix.
    pub fn into_matrix(self) -> Matrix<f64> {
        let rows = self.tuv_max;
        let cols = self.dim0 * self.dim1;
        Matrix::<f64>::new_from_vec(rows, cols, self.data)
    }

}

impl<'a> RefKetExpansionCoefficients<'a, f64> {
    /// The coefficients over `data`.
    #[inline(always)]
    pub fn new(dim0: usize, dim1: usize, l_max: usize, data: &'a mut [f64]) -> Self {
        let tuv_max = hermite_count(l_max);
        Self {
            dim0,
            dim1,
            l_max,
            tuv_max,
            data,
        }
    }

    /// Zeroed coefficients from the arena.
    #[inline(always)]
    pub fn from_arena(dim0: usize, dim1: usize, l_max: usize, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let tuv_max = hermite_count(l_max);
        let data = arena.alloc_zeroed(arrays::scratch::checked_product(&[dim0, dim1, tuv_max])?)?;
        Ok(Self {
            dim0,
            dim1,
            l_max,
            tuv_max,
            data,
        })
    }
}

macro_rules! impl_indexing {
    ($TensorType: ty) => {
        #[cfg(feature = "unchecked")]
        impl<D: DataContainerMut> Index<(usize, usize, usize, usize, usize)> for $TensorType {
            type Output = D::T;
            fn index(&self, ij: (usize, usize, usize, usize, usize)) -> &D::T {
                let index = self.compute_index(ij.0, ij.1, ij.2, ij.3, ij.4);
                unsafe { self.get_data().get_unchecked(index) }
            }
        }

        #[cfg(not(feature = "unchecked"))]
        impl<D: DataContainerMut> Index<(usize, usize, usize, usize, usize)> for $TensorType {
            type Output = D::T;
            fn index(&self, ij: (usize, usize, usize, usize, usize)) -> &D::T {
                let index = self.compute_index(ij.0, ij.1, ij.2, ij.3, ij.4);
                &self.get_data()[index]
            }
        }
    };
}
macro_rules! impl_mut_indexing {
    ($TensorType: ty) => {

        impl_indexing!($TensorType);

        #[cfg(feature = "unchecked")]
        impl<D: DataContainerMut> IndexMut<(usize, usize, usize, usize, usize)> for $TensorType {
            fn index_mut(&mut self, ij: (usize, usize, usize, usize, usize)) -> &mut D::T {
                let index = self.compute_index(ij.0, ij.1, ij.2, ij.3, ij.4);
                unsafe { self.get_data_mut().get_unchecked_mut(index) }
            }
        }

        #[cfg(not(feature = "unchecked"))]
        impl<D: DataContainerMut> IndexMut<(usize, usize, usize, usize, usize)> for $TensorType {
            fn index_mut(&mut self, ij: (usize, usize, usize, usize, usize)) -> &mut D::T {
                let index = self.compute_index(ij.0, ij.1, ij.2, ij.3, ij.4);
                &mut self.get_data_mut()[index]
            }
        }
    };
}
impl_mut_indexing!(BraExpansionCoefficients<D>);
impl_indexing!(&BraExpansionCoefficients<D>);
impl_mut_indexing!(&mut BraExpansionCoefficients<D>);
impl_mut_indexing!(KetExpansionCoefficients<D>);
impl_indexing!(&KetExpansionCoefficients<D>);
impl_mut_indexing!(&mut KetExpansionCoefficients<D>);

/// The position of `(t, u, v)`, `t + u + v <= n`, in a packed block: `t` fastest, then `u`, then `v`.
#[inline(always)]
pub fn hermite_index(t: usize, u: usize, v: usize, n: usize) -> usize {
    let m = n - v;
    let t1 = tetrahedral_number(n + 1);
    let t2 = tetrahedral_number(m + 1);
    let t3 = u * (m + 1);
    let t4 = if u >= 2 { u * (u - 1) / 2 } else { 0 };
    t1 - t2 + t3 - t4 + t
}

/// The number of `(t, u, v)` with `t + u + v <= n_max`, `(n_max + 1)(n_max + 2)(n_max + 3) / 6`.
#[inline(always)]
pub fn hermite_count(n_max: usize) -> usize {
    hermite_index(0, 0, n_max, n_max) + 1
}

/// The `n_max` of a packed block of `len` entries, the inverse of `hermite_count`. Panics if there is none.
#[inline(always)]
pub fn hermite_order(len: usize) -> usize {
    (0..).find(|&n| hermite_count(n) == len).expect("not a valid combinations length")
}

// n (n + 1) (n + 2) / 6.
#[inline(always)]
fn tetrahedral_number(n: usize) -> usize {
    n * (n + 1) * (n + 2) / 6
}
