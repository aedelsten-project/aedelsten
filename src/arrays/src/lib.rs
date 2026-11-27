// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Dense column-major arrays: vectors, matrices (real and split complex) and 3- to 8-dimensional arrays, each owned,
//! a view or a mutable view; their products and factorizations through BLAS and LAPACK, the only calls to either;
//! the BLAS's thread control; and the scratch arena the computations take their temporary memory from.

#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

// Checks each index against its dimension (the flat bounds check misses an index past its dimension that still lands
// inside the buffer). Debug builds only, for the checked and the `unchecked` indexing alike.
macro_rules! debug_check_index {
    ($index:expr, $dims:expr) => {
        debug_assert!($index.iter().zip($dims.iter()).all(|(i, d)| i < d), "index {:?} out of bounds for {:?}", $index, $dims)
    };
}

mod layout;
pub mod matrix;
pub mod array3d;
pub mod array4d;
pub mod array5d;
pub mod array6d;
pub mod array7d;
pub mod array8d;
pub mod scratch;
#[cfg(test)]
mod array_tests;
pub mod blas;
mod blas_interface;
pub mod vector;
pub mod vector_math;

pub use array3d::{
    Array3D,
    Array3DTrait,
    Array3DTraitMut,
    Array3DMathMut,
};
pub use array4d::{
    Array4D,
    Array4DTrait,
    Array4DTraitMut,
};
pub use array5d::Array5D;
pub use array7d::Array7D;
pub use array8d::Array8D;


