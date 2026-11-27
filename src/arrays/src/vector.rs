// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Vectors as BLAS operands. `Vector`: anything the level-1/2 routines read (a slice, an array, a `Vec`, a
//! `Box<[f64]>`, a `StridedVectorView`); `VectorMut`: anything they write. A strided view is a length, a step and the
//! data, e.g. a matrix row (step = rows) or a diagonal (step = rows + 1); BLAS takes the step as its `incx`/`incy`, so
//! nothing is copied. Vectors carry no orientation: the matrix side of a product carries the transpose.

use crate::blas_interface::{daxpy, ddot};
use std::ops::{Index, IndexMut};

mod sealed {
    /// The vector as BLAS sees it: the data from its first element, the length and the step.
    pub struct VectorOperand<'a> {
        /// The data from the first entry.
        pub data: &'a [f64],
        /// The number of entries.
        pub len: usize,
        /// The distance between two entries.
        pub step: usize,
    }

    /// The written vector as BLAS sees it, as `VectorOperand`.
    pub struct VectorOperandMut<'a> {
        /// The data from the first entry.
        pub data: &'a mut [f64],
        /// The number of entries.
        pub len: usize,
        /// The distance between two entries.
        pub step: usize,
    }

    /// Unnameable outside this crate, so only these types are vectors.
    pub trait BlasVector {
        fn vector_operand(&self) -> VectorOperand<'_>;
    }

    /// Unnameable outside this crate, so only these types are written vectors.
    pub trait BlasVectorMut: BlasVector {
        fn vector_operand_mut(&mut self) -> VectorOperandMut<'_>;
    }
}

/// The values needed to hold `len` entries `step` apart.
#[inline(always)]
fn span(len: usize, step: usize) -> usize {
    if len == 0 { 0 } else { (len - 1) * step + 1 }
}

/// A vector the level-1/2 routines read: a slice, an array, a `Vec`, a `Box<[f64]>`, a `StridedVectorView(Mut)`.
pub trait Vector: sealed::BlasVector {
    /// The number of entries.
    #[inline]
    fn len(&self) -> usize {
        self.vector_operand().len
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `self . other`.
    #[inline]
    fn dot<V: Vector + ?Sized>(&self, other: &V) -> f64 {
        let (x, y) = (self.vector_operand(), other.vector_operand());
        assert_eq!(x.len, y.len, "dot: lengths {} and {}", x.len, y.len);
        if x.len == 0 {
            return 0.0;
        }
        ddot(x.len as i32, x.data.as_ptr(), x.step as i32, y.data.as_ptr(), y.step as i32)
    }

    /// The Euclidean norm.
    #[inline]
    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// The entries, in order.
    #[inline]
    fn iter_entries(&self) -> std::iter::Take<std::iter::StepBy<std::slice::Iter<'_, f64>>> {
        let x = self.vector_operand();
        x.data.iter().step_by(x.step).take(x.len)
    }

    /// A contiguous copy, on the heap.
    #[inline]
    fn to_vec(&self) -> Vec<f64> {
        self.iter_entries().copied().collect()
    }
}

/// A vector the level-1/2 routines write: a mutable slice, a `Vec`, a `Box<[f64]>`, a `StridedVectorViewMut`.
pub trait VectorMut: Vector + sealed::BlasVectorMut {
    /// `self += alpha * x`.
    #[inline]
    fn add_scaled_in_place<V: Vector + ?Sized>(&mut self, x: &V, alpha: f64) {
        let x = x.vector_operand();
        let y = self.vector_operand_mut();
        assert_eq!(x.len, y.len, "add_scaled_in_place: lengths {} and {}", y.len, x.len);
        if y.len == 0 {
            return;
        }
        daxpy(y.len as i32, alpha, x.data.as_ptr(), x.step as i32, y.data.as_mut_ptr(), y.step as i32);
    }

    /// `self *= alpha`.
    #[inline]
    fn scale_in_place(&mut self, alpha: f64) {
        self.iter_entries_mut().for_each(|v| *v *= alpha);
    }

    /// Every entry set to `value`.
    #[inline]
    fn fill_entries(&mut self, value: f64) {
        self.iter_entries_mut().for_each(|v| *v = value);
    }

    /// The entries of `x`, copied in.
    #[inline]
    fn copy_from<V: Vector + ?Sized>(&mut self, x: &V) {
        assert_eq!(self.len(), x.len(), "copy_from: lengths {} and {}", self.len(), x.len());
        self.iter_entries_mut().zip(x.iter_entries()).for_each(|(v, x)| *v = *x);
    }

    /// The entries, in order, mutably.
    #[inline]
    fn iter_entries_mut(&mut self) -> std::iter::Take<std::iter::StepBy<std::slice::IterMut<'_, f64>>> {
        let y = self.vector_operand_mut();
        y.data.iter_mut().step_by(y.step).take(y.len)
    }
}

// --- contiguous containers: step 1 ---

impl sealed::BlasVector for [f64] {
    #[inline]
    fn vector_operand(&self) -> sealed::VectorOperand<'_> {
        sealed::VectorOperand { data: self, len: self.len(), step: 1 }
    }
}

impl sealed::BlasVectorMut for [f64] {
    #[inline]
    fn vector_operand_mut(&mut self) -> sealed::VectorOperandMut<'_> {
        let len = self.len();
        sealed::VectorOperandMut { data: self, len, step: 1 }
    }
}

impl Vector for [f64] {}
impl VectorMut for [f64] {}

macro_rules! contiguous_vector {
    ($t:ty) => {
        impl sealed::BlasVector for $t {
            #[inline]
            fn vector_operand(&self) -> sealed::VectorOperand<'_> {
                self[..].vector_operand()
            }
        }
        impl sealed::BlasVectorMut for $t {
            #[inline]
            fn vector_operand_mut(&mut self) -> sealed::VectorOperandMut<'_> {
                self[..].vector_operand_mut()
            }
        }
        impl Vector for $t {}
        impl VectorMut for $t {}
    };
}

contiguous_vector!(Vec<f64>);
contiguous_vector!(Box<[f64]>);

impl<const N: usize> sealed::BlasVector for [f64; N] {
    #[inline]
    fn vector_operand(&self) -> sealed::VectorOperand<'_> {
        self[..].vector_operand()
    }
}
impl<const N: usize> sealed::BlasVectorMut for [f64; N] {
    #[inline]
    fn vector_operand_mut(&mut self) -> sealed::VectorOperandMut<'_> {
        self[..].vector_operand_mut()
    }
}
impl<const N: usize> Vector for [f64; N] {}
impl<const N: usize> VectorMut for [f64; N] {}

// References forward, so a caller holding a `&[f64]` or `&mut [f64]` passes it as it is.
impl<T: sealed::BlasVector + ?Sized> sealed::BlasVector for &T {
    #[inline]
    fn vector_operand(&self) -> sealed::VectorOperand<'_> {
        (**self).vector_operand()
    }
}
impl<T: sealed::BlasVector + ?Sized> sealed::BlasVector for &mut T {
    #[inline]
    fn vector_operand(&self) -> sealed::VectorOperand<'_> {
        (**self).vector_operand()
    }
}
impl<T: sealed::BlasVectorMut + ?Sized> sealed::BlasVectorMut for &mut T {
    #[inline]
    fn vector_operand_mut(&mut self) -> sealed::VectorOperandMut<'_> {
        (**self).vector_operand_mut()
    }
}
impl<T: Vector + ?Sized> Vector for &T {}
impl<T: Vector + ?Sized> Vector for &mut T {}
impl<T: VectorMut + ?Sized> VectorMut for &mut T {}

// --- the strided views ---

/// `len` entries `step` apart in a larger buffer: a matrix row (`view_row`), a diagonal (`view_diagonal`). Only
/// functions of this crate make one.
#[derive(Clone, Copy)]
pub struct StridedVectorView<'a> {
    data: &'a [f64],
    len: usize,
    step: usize,
}

/// The mutable form of `StridedVectorView`.
pub struct StridedVectorViewMut<'a> {
    data: &'a mut [f64],
    len: usize,
    step: usize,
}

impl<'a> StridedVectorView<'a> {
    /// Entry `i` is `data[i * step]`.
    #[inline]
    pub(crate) fn new(data: &'a [f64], len: usize, step: usize) -> Self {
        assert!(step > 0 || len <= 1, "StridedVectorView: step 0");
        assert!(data.len() >= span(len, step), "StridedVectorView: {len} entries {step} apart need {} values, got {}", span(len, step), data.len());
        Self { data, len, step }
    }
}

impl<'a> StridedVectorViewMut<'a> {
    /// Entry `i` is `data[i * step]`.
    #[inline]
    pub(crate) fn new(data: &'a mut [f64], len: usize, step: usize) -> Self {
        assert!(step > 0 || len <= 1, "StridedVectorViewMut: step 0");
        assert!(data.len() >= span(len, step), "StridedVectorViewMut: {len} entries {step} apart need {} values, got {}", span(len, step), data.len());
        Self { data, len, step }
    }

    /// A read-only view of the same entries.
    #[inline]
    pub fn view(&self) -> StridedVectorView<'_> {
        StridedVectorView { data: self.data, len: self.len, step: self.step }
    }
}

impl sealed::BlasVector for StridedVectorView<'_> {
    #[inline]
    fn vector_operand(&self) -> sealed::VectorOperand<'_> {
        sealed::VectorOperand { data: self.data, len: self.len, step: self.step.max(1) }
    }
}

impl sealed::BlasVector for StridedVectorViewMut<'_> {
    #[inline]
    fn vector_operand(&self) -> sealed::VectorOperand<'_> {
        sealed::VectorOperand { data: self.data, len: self.len, step: self.step.max(1) }
    }
}

impl sealed::BlasVectorMut for StridedVectorViewMut<'_> {
    #[inline]
    fn vector_operand_mut(&mut self) -> sealed::VectorOperandMut<'_> {
        sealed::VectorOperandMut { data: self.data, len: self.len, step: self.step.max(1) }
    }
}

impl Vector for StridedVectorView<'_> {}
impl Vector for StridedVectorViewMut<'_> {}
impl VectorMut for StridedVectorViewMut<'_> {}

impl Index<usize> for StridedVectorView<'_> {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        debug_assert!(i < self.len, "StridedVectorView: entry {i} of {}", self.len);
        crate::layout::element(self.data, i * self.step)
    }
}

impl Index<usize> for StridedVectorViewMut<'_> {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        debug_assert!(i < self.len, "StridedVectorViewMut: entry {i} of {}", self.len);
        crate::layout::element(self.data, i * self.step)
    }
}

impl IndexMut<usize> for StridedVectorViewMut<'_> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        debug_assert!(i < self.len, "StridedVectorViewMut: entry {i} of {}", self.len);
        crate::layout::element_mut(self.data, i * self.step)
    }
}
