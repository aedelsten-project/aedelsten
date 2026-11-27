// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Induced dipoles of polarizable sites: the dipole-dipole coupling (with optional Thole damping and
//! fragment exclusions), and the solution of
//!
//!   mu_s = alpha_s (F_s + Sum_(j != s) T_sj mu_j),   i.e.   (alpha^-1 - T) mu = F,
//!
//! the polarizable-embedding model's induced-dipole equations (N. H. List, J. M. H. Olsen and J. Kongsted, Phys.
//! Chem. Chem. Phys. 18, 20234 (2016), eqs. 9-10), with T_ij mu_j the field of dipole j at site i.
//!
//! All vectors are `arrays` matrices of shape 3 x n_sites (column s holds (x, y, z) of site s), the layout of the
//! electronic and static fields of `efield` and `sources`. Sites are `sources::PolarizableSite`s, in input order.
//!
//! The iterative solver splits the coupling over shares (MPI ranks, say) through `Collective`, which the caller
//! implements; the crate itself is MPI-free.

mod coupling;
mod damping;
mod solvers;
mod sym3;
mod system;
mod scratch;

pub use coupling::{pair_rows, Collective, CouplingWork, Whole};
pub use damping::{Damping, THOLE_DEFAULT_A};
pub use solvers::{InitialGuess, SolveReport, SolverOptions};
pub use system::{PolarizableSystem, SystemParameters};
pub use scratch::{IterativeBuffers, IterativeScratch};

/// An error of the induced-dipole setup or solution.
#[derive(Debug, PartialEq)]
pub enum PolarizationError {
    /// A vector is not a 3 x n_sites matrix.
    Shape { what: &'static str, expected: (usize, usize), found: (usize, usize) },
    /// Two sites that are not excluded from each other sit at the same position.
    CoincidentSites { first: usize, second: usize },
    /// A site's polarizability tensor is singular, so its inverse (needed by the relay matrix) does not exist.
    SingularPolarizability { site: usize },
    /// Parameters derived for a different number of sites than the ones given.
    ParametersMismatch { sites: usize, parameters: usize },
    /// The iterative scratch was made for a different number of sites.
    ScratchMismatch { needed: usize, found: usize },
    /// The iterative solver did not reach its threshold.
    NotConverged { iterations: usize, residual: f64 },
    /// The relay matrix has a non-positive eigenvalue: the induction energy is unbounded below. From the
    /// iterative solver, the value is a Rayleigh quotient, an upper bound of the lowest eigenvalue.
    PolarizationCatastrophe { lowest_eigenvalue: f64 },
    /// A sum or broadcast over the shares of the coupling failed.
    Collective(String),
}

impl std::fmt::Display for PolarizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape { what, expected, found } => {
                write!(f, "{what} is {} x {}, expected {} x {}", found.0, found.1, expected.0, expected.1)
            }
            Self::CoincidentSites { first, second } => {
                write!(f, "polarizable sites {first} and {second} coincide and are not excluded from each other")
            }
            Self::SingularPolarizability { site } => write!(f, "the polarizability of site {site} is singular"),
            Self::ParametersMismatch { sites, parameters } => {
                write!(f, "parameters were derived for {parameters} sites, but {sites} sites were given")
            }
            Self::ScratchMismatch { needed, found } => write!(f, "iterative scratch is for {found} sites, need {needed}"),
            Self::NotConverged { iterations, residual } => {
                write!(f, "induced dipoles not converged after {iterations} iterations (residual {residual:.3e})")
            }
            Self::PolarizationCatastrophe { lowest_eigenvalue } => write!(
                f,
                "polarization catastrophe: the relay matrix has eigenvalue {lowest_eigenvalue:.3e} <= 0 (consider damping)"
            ),
            Self::Collective(message) => write!(f, "induced dipoles over the ranks: {message}"),
        }
    }
}

impl std::error::Error for PolarizationError {}

#[cfg(test)]
mod tests;
