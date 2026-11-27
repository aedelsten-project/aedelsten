// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The Hermite multipole moments `M^e_t` of a Hermite Gaussian about a point.

use arrays::matrix::*;
use arrays::scratch::{Arena, ArenaError};

/// The Hermite multipole moments `M^e_t` along x, y and z (`(t, e)`, `t, e <= order`) of a Hermite Gaussian centred
/// at P about a point C, without the factor `sqrt(pi/p)`, by
/// `M^e_t = X_PC M^{e-1}_t + t M^{e-1}_{t-1} + M^{e-1}_{t+1} / (2p)` from `M^0_0 = 1`. The rows can be padded.
pub struct HermiteMomentsStruct<M>
where
    M: MatrixTraitMut<f64>
{
    order: usize,
    mm_x: M,
    mm_y: M,
    mm_z: M,
}
/// The moments in arena storage.
pub type HermiteMoments<'a> = HermiteMomentsStruct<MatrixViewMut<'a, f64>>;
/// The moments in heap storage.
pub type HermiteMomentsOwned = HermiteMomentsStruct<Matrix<f64>>;

impl HermiteMomentsOwned {
    /// Zeroed moments up to `order`, with `padding` extra rows.
    pub fn new(order: usize, padding: usize) -> Self {
        let dim = order + 1;
        let mm_x = Matrix::<f64>::new(dim+padding, dim);
        let mm_y = Matrix::<f64>::new(dim+padding, dim);
        let mm_z = Matrix::<f64>::new(dim+padding, dim);
        HermiteMomentsOwned {
            order,
            mm_x,
            mm_y,
            mm_z
        }
    }
}

impl<'a> HermiteMoments<'a> {
    /// `HermiteMomentsOwned::new` in arena storage.
    pub fn from_arena(order: usize, padding: usize, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let dim = order + 1;
        let mm_x = MatrixViewMut::<f64>::from_arena(dim+padding, dim, arena)?;
        let mm_y = MatrixViewMut::<f64>::from_arena(dim+padding, dim, arena)?;
        let mm_z = MatrixViewMut::<f64>::from_arena(dim+padding, dim, arena)?;
        Ok(HermiteMoments {
            order,
            mm_x,
            mm_y,
            mm_z
        })
    }
}

impl<M> HermiteMomentsStruct<M>
where
    M: MatrixTraitMut<f64>
{
    /// The highest order.
    pub fn get_order(&self) -> usize {
        self.order
    }

    /// The x moments, `(t, e)`.
    pub fn get_mm_x(&self) -> &M {
        &self.mm_x
    }

    /// The y moments, `(t, e)`.
    pub fn get_mm_y(&self) -> &M {
        &self.mm_y
    }

    /// The z moments, `(t, e)`.
    pub fn get_mm_z(&self) -> &M {
        &self.mm_z
    }

    /// The moments for `(px, py, pz) = P - C` and the exponent `p`.
    pub fn compute_hermite_moments(&mut self, px: f64, py: f64, pz: f64, p: f64) {
        let HermiteMomentsStruct {
            order,
            mm_x,
            mm_y,
            mm_z
        } = self;
        compute_moment_matrix(mm_x, *order, px, p);
        compute_moment_matrix(mm_y, *order, py, p);
        compute_moment_matrix(mm_z, *order, pz, p);
    }
}


#[inline(always)]
fn compute_moment_matrix<M>(
    moment_matrix: &mut M,
    order: usize,
    px: f64,
    p: f64,
)
where
    M: MatrixTraitMut<f64>
{
    moment_matrix[(0, 0)] = 1.0;
    if order > 0 {
        let two_p_inv = 1.0 / (2.0 * p);
        moment_matrix[(0, 1)] = px;
        moment_matrix[(1, 1)] = 1.0;
        for e in 2..=order {
            for t in 0..e {
                let mut acc = px * moment_matrix[(t, e - 1)];
                if t > 0 {
                    acc += (t as f64) * moment_matrix[(t - 1, e - 1)];
                }
                if t + 1 < e {
                    acc += two_p_inv * moment_matrix[(t + 1, e - 1)];
                }
                moment_matrix[(t, e)] = acc;
            }
            moment_matrix[(e, e)] = (e as f64) * moment_matrix[(e - 1, e - 1)];
        }
    }
}