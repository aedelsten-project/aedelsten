// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Rows and diagonals as strided vector views: no copy, BLAS reads them with their step.

use super::*;
use crate::vector::{StridedVectorView, StridedVectorViewMut};

impl<D> MatrixStruct<D>
where
    D: DataContainer<T = f64>,
{
    /// Row `i`, its entries `rows()` apart.
    #[inline]
    pub fn view_row(&self, i: usize) -> StridedVectorView<'_> {
        assert!(i < self.rows(), "view_row: row {i} of {}", self.rows());
        let data = if self.cols() == 0 { &[][..] } else { &self.get_data()[i..] };
        StridedVectorView::new(data, self.cols(), self.rows())
    }

    /// The diagonal, `min(rows, cols)` entries `rows() + 1` apart.
    #[inline]
    pub fn view_diagonal(&self) -> StridedVectorView<'_> {
        StridedVectorView::new(self.get_data(), self.rows().min(self.cols()), self.rows() + 1)
    }

}

impl<D> MatrixStruct<D>
where
    D: DataContainerMut<T = f64>,
{
    /// Row `i`, mutably.
    #[inline]
    pub fn view_row_mut(&mut self, i: usize) -> StridedVectorViewMut<'_> {
        let (rows, cols) = (self.rows(), self.cols());
        assert!(i < rows, "view_row_mut: row {i} of {rows}");
        let data = if cols == 0 { &mut [][..] } else { &mut self.data.resolve_mut()[i..] };
        StridedVectorViewMut::new(data, cols, rows)
    }

    /// The diagonal, mutably.
    #[inline]
    pub fn view_diagonal_mut(&mut self) -> StridedVectorViewMut<'_> {
        let (rows, cols) = (self.rows(), self.cols());
        StridedVectorViewMut::new(self.data.resolve_mut(), rows.min(cols), rows + 1)
    }
}
