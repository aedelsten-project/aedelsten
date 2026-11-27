// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use arrays::array3d::{Array3DTrait, Array3DTraitMut, Array3DViewMut};
use arrays::matrix::Factor;
use arrays::matrix::{MatrixTrait, MatrixViewMut};
use arrays::scratch::{Arena, ArenaError};

/// Spherical transformation of every (cart_a, cart_b) slice of a batch. The buffers are allocated up front, before
/// the kernel loop, so an arena overflow does not rerun it.
pub(crate) struct SliceTransformation<'a> {
    spherical: Option<Array3DViewMut<'a, f64>>,
    ca_t: Option<MatrixViewMut<'a, f64>>,
    cb: Option<MatrixViewMut<'a, f64>>,
    half: Option<MatrixViewMut<'a, f64>>,
    comps: (usize, usize),
}

impl<'a> SliceTransformation<'a> {
    /// The buffers for `n_slices` slices of shells of angular momenta `lqn_a` and `lqn_b`, either side spherical or
    /// Cartesian.
    pub(crate) fn new(lqn_a: usize, spherical_a: bool, lqn_b: usize, spherical_b: bool, n_slices: usize, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let cart_a = cart2sph::compute_cartesian_degeneracy(lqn_a);
        let cart_b = cart2sph::compute_cartesian_degeneracy(lqn_b);
        let comps = (
            if spherical_a { cart2sph::compute_spherical_degeneracy(lqn_a) } else { cart_a },
            if spherical_b { cart2sph::compute_spherical_degeneracy(lqn_b) } else { cart_b },
        );
        let spherical = match spherical_a || spherical_b {
            true => Some(Array3DViewMut::<f64>::from_arena(comps.0, comps.1, n_slices, arena)?),
            false => None,
        };
        let ca_t = match spherical_a {
            true => Some(cart2sph::compute_cartesian_to_spherical_matrix_arena(lqn_a, arena)?.transpose_arena(arena)?),
            false => None,
        };
        let cb = match spherical_b {
            true => Some(cart2sph::compute_cartesian_to_spherical_matrix_arena(lqn_b, arena)?),
            false => None,
        };
        let half = match spherical_a && spherical_b {
            true => Some(MatrixViewMut::<f64>::from_arena(cart_a, comps.1, arena)?),
            false => None,
        };
        Ok(Self { spherical, ca_t, cb, half, comps })
    }

    /// The components per side after the transformation.
    pub(crate) fn comps(&self) -> (usize, usize) {
        self.comps
    }

    /// The transformed data of `batch`, or its own data when both sides are Cartesian.
    pub(crate) fn apply(self, batch: Array3DViewMut<'a, f64>) -> &'a mut [f64] {
        let Some(mut spherical) = self.spherical else {
            return batch.into_data();
        };
        let (ca_t, cb, mut half) = (self.ca_t, self.cb, self.half);
        for k in 0..batch.dim2() {
            let src = batch.view_sub_matrix(k);
            let mut dst = spherical.view_sub_matrix_mut(k);
            match (&ca_t, &cb, &mut half) {
                (None, Some(cb), _) => src.matmul_prealloc(cb, &mut dst),
                (Some(ca_t), None, _) => ca_t.matmul_prealloc(&src, &mut dst),
                (Some(ca_t), Some(cb), Some(half)) => {
                    src.matmul_prealloc(cb, half);
                    ca_t.matmul_prealloc(half, &mut dst);
                }
                _ => unreachable!(),
            }
        }
        spherical.into_data()
    }
}
