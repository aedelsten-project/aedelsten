// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! McMurchie-Davidson integral machinery (L. E. McMurchie and E. R. Davidson, J. Comput. Phys. 26, 218 (1978)): the
//! Hermite expansion coefficients of Gaussian products, the Hermite Coulomb integrals `R_{tuv}`, and the Hermite
//! multipole moments.

#![allow(confusable_idents)]
#![allow(mixed_script_confusables)]

pub mod hermite_coefficients;
pub mod hermite_moment;
pub mod hermite_coulomb_integrals;
pub mod hermite_r_tensor_rolling;
pub mod hermite_r_generated;
pub mod ketside_hermite_coefficients;
pub mod hermite_expansion;
pub mod hermite_r_tensor;
