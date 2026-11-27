// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Storage for `PolarizableSystem::solve_iterative`, allocated once and reused: the iterative solver itself
// allocates nothing, neither per iteration nor per solve. Six vectors of 3 n_sites doubles: the conjugate-gradient
// residual, its preconditioned form, the direction and its image under the relay matrix, and the dipoles and the
// field of the coupling's n_sites x 3 layout. Two ways to provide them, both giving the same `IterativeScratch`:
//   - `IterativeBuffers`: heap memory owned by the caller and kept alive across solves;
//   - `IterativeScratch::from_arena`: carved out of an arena the caller sized with `required_bytes`.

use arrays::scratch::{Arena, ArenaError};

/// The number of 3 x n_sites vectors the iterative solver keeps.
const VECTORS: usize = 6;

/// The scratch of the iterative solver for a number of sites.
pub struct IterativeScratch<'a> {
    pub(crate) n_sites: usize,
    pub(crate) residual: &'a mut [f64],
    pub(crate) preconditioned: &'a mut [f64],
    pub(crate) direction: &'a mut [f64],
    pub(crate) image: &'a mut [f64],
    pub(crate) dipoles_by_axis: &'a mut [f64],
    pub(crate) field_by_axis: &'a mut [f64],
}

// Splits `n` values off the front of `rest`.
fn take<'a, T>(rest: &mut &'a mut [T], n: usize) -> &'a mut [T] {
    let (head, tail) = std::mem::take(rest).split_at_mut(n);
    *rest = tail;
    head
}

impl<'a> IterativeScratch<'a> {
    /// Bytes an arena must have for `from_arena`, including a small allowance for alignment padding.
    pub fn required_bytes(n_sites: usize) -> usize {
        VECTORS * 3 * n_sites * size_of::<f64>() + 64
    }

    /// Carves the scratch out of `arena` (zero-initialised); nothing is allocated on the heap.
    pub fn from_arena(n_sites: usize, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let mut floats = arena.alloc_zeroed::<f64>(VECTORS * 3 * n_sites)?;
        Ok(Self::from_slices(n_sites, &mut floats))
    }

    fn from_slices(n_sites: usize, floats: &mut &'a mut [f64]) -> Self {
        let block = 3 * n_sites;
        Self {
            n_sites,
            residual: take(floats, block),
            preconditioned: take(floats, block),
            direction: take(floats, block),
            image: take(floats, block),
            dipoles_by_axis: take(floats, block),
            field_by_axis: take(floats, block),
        }
    }

    /// The number of sites it is for.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }
}

/// Heap memory for an `IterativeScratch`, meant to be kept alive and reused across solves: it allocates
/// once, in `new`, 18 n_sites doubles.
#[derive(Debug)]
pub struct IterativeBuffers {
    n_sites: usize,
    floats: Vec<f64>,
}

impl IterativeBuffers {
    /// For `n_sites` sites.
    pub fn new(n_sites: usize) -> Self {
        Self { n_sites, floats: vec![0.0; VECTORS * 3 * n_sites] }
    }

    /// The scratch over these buffers. Stale contents from an earlier solve are harmless: the solver
    /// writes every value before it reads it.
    pub fn scratch(&mut self) -> IterativeScratch<'_> {
        let mut floats: &mut [f64] = &mut self.floats;
        IterativeScratch::from_slices(self.n_sites, &mut floats)
    }
}
