// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

/// The additional argument of the kernels that take none.
pub struct EmptyAdditionalArguments;

/// The origin and order of the moment operator (x - O_x)^lx (y - O_y)^ly (z - O_z)^lz, all Cartesian components of
/// one order: the molecule's own moment integrals. (The environment's point multipoles are sources interacting
/// through 1/r, `ExternalMultipoles`.)
#[derive(Clone, Copy, Debug)]
pub struct MomentOrigin {
    origin: [f64; 3],
    order: usize,
}

impl MomentOrigin {
    /// About `origin` (bohr), of order `order`.
    pub fn new(origin: [f64; 3], order: usize) -> Self {
        Self { origin, order }
    }

    /// The origin, in bohr.
    pub fn get_origin(&self) -> &[f64; 3] {
        &self.origin
    }

    /// The order.
    pub fn get_order(&self) -> usize {
        self.order
    }
}
