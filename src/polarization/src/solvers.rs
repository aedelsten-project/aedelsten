// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use crate::coupling::{Collective, CouplingWork};
use crate::system::check_shape;
use crate::{PolarizableSystem, PolarizationError};
use crate::scratch::IterativeScratch;
use arrays::matrix::{MatrixMath, MatrixTrait, MatrixTraitMut, MatrixView, MatrixViewMut};
use arrays::vector_math::{add_in_place, add_scaled_in_place, dot, norm, scale_in_place};

/// Where the iterative solver starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialGuess {
    /// The dipoles of the uncoupled sites, `alpha F`.
    Static,
    /// The dipoles already in the output matrix, e.g. those of a previous solve.
    Provided,
}

/// Convergence controls of the iterative solver (preconditioned conjugate gradients); by default a threshold of
/// 1e-8 and 100 applications of the coupling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverOptions {
    /// The threshold on the norm of `alpha r`, the residual `r = F - (alpha^-1 - T) mu` preconditioned: the change
    /// of the 3S-vector of dipoles one Jacobi step `mu <- alpha (F + T mu)` would make.
    pub threshold: f64,
    /// The most applications of the coupling before `NotConverged`.
    pub max_iterations: usize,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self { threshold: 1e-8, max_iterations: 100 }
    }
}

/// How an iterative solve went.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveReport {
    /// Number of applications of the coupling (one to start, one per iteration).
    pub iterations: usize,
    /// The norm of `alpha r` at the returned dipoles.
    pub residual: f64,
    /// The induction functional `1/2 mu . (alpha^-1 - T) mu - mu . F` at the returned dipoles, from the residual
    /// (`-1/2 mu . (F + r)`), with no further application of the coupling.
    pub functional: f64,
}

impl<'a> PolarizableSystem<'a> {
    /// Solves `(alpha^-1 - T) mu = F` by dense linear algebra, for the total polarizing field `F` (3 x S)
    /// into `mu` (3 x S). O(S^3) time and O(S^2) memory: the reference for small systems. Fails with
    /// `PolarizationCatastrophe` if the relay matrix is not positive definite (then the induction
    /// energy is unbounded below and the "solution" is meaningless).
    pub fn solve_dense<G, M>(&self, field: &G, mu: &mut M) -> Result<(), PolarizationError>
    where
        G: MatrixTrait<f64>,
        M: MatrixTraitMut<f64>,
    {
        check_shape(field, "field", self.n_sites())?;
        check_shape(mu, "dipoles", self.n_sites())?;
        if self.n_sites() == 0 {
            return Ok(());
        }
        let relay = self.relay_matrix()?;
        let lowest_eigenvalue = relay.eigenvalues().iter().copied().fold(f64::INFINITY, f64::min);
        if lowest_eigenvalue <= 0.0 {
            return Err(PolarizationError::PolarizationCatastrophe { lowest_eigenvalue });
        }
        let solution = relay.backslash(field.get_data());
        let n = 3 * self.n_sites();
        mu.get_data_mut().copy_from_slice(&solution[..n]);
        Ok(())
    }

    /// Solves the same equations iteratively, without ever forming the relay matrix: conjugate gradients on the
    /// symmetric positive definite relay matrix `A = alpha^-1 - T`, preconditioned by `alpha` (M. R. Hestenes and
    /// E. Stiefel, J. Res. Natl. Bur. Stand. 49, 409 (1952)). Each iteration is one application of the O(S^2)
    /// coupling, split over the shares of `collective` (`Whole` for one process), with one sum of 3S values;
    /// the scalars that steer the iteration are share 0's, broadcast, so every share takes the same steps and
    /// returns the same dipoles. With `InitialGuess::Provided` the dipoles in `mu` (e.g. from a previous solve)
    /// are the starting point. A direction with `p . A p <= 0` is `PolarizationCatastrophe`, its Rayleigh
    /// quotient `p . A p / p . p` (an upper bound of the lowest eigenvalue) reported.
    ///
    /// Allocates nothing: all scratch lives in `scratch`, which must have been made for this many sites (see
    /// `IterativeBuffers` and `IterativeScratch::from_arena`).
    pub fn solve_iterative<G, M, C>(
        &self,
        field: &G,
        mu: &mut M,
        guess: InitialGuess,
        options: &SolverOptions,
        scratch: &mut IterativeScratch<'_>,
        collective: &C,
    ) -> Result<SolveReport, PolarizationError>
    where
        G: MatrixTrait<f64>,
        M: MatrixTraitMut<f64>,
        C: Collective,
    {
        let n_sites = self.n_sites();
        check_shape(field, "field", n_sites)?;
        check_shape(mu, "dipoles", n_sites)?;
        if scratch.n_sites != n_sites {
            return Err(PolarizationError::ScratchMismatch { needed: n_sites, found: scratch.n_sites });
        }
        if guess == InitialGuess::Static {
            self.apply_alpha(field, mu)?;
        }
        let r: &mut [f64] = &mut *scratch.residual;
        let z: &mut [f64] = &mut *scratch.preconditioned;
        let p: &mut [f64] = &mut *scratch.direction;
        let q: &mut [f64] = &mut *scratch.image;
        let mut work = CouplingWork::new(&mut *scratch.dipoles_by_axis, &mut *scratch.field_by_axis, n_sites);

        // r = F - (alpha^-1 mu - T mu).
        self.field_of_dipoles(mu, &mut MatrixViewMut::new(3, n_sites, &mut *r), &mut work, collective)?;
        self.relay_from_coupling(mu.get_data(), r);
        scale_in_place(r, -1.0);
        add_in_place(r, field.get_data());
        let mut iterations = 1;
        let (mut residual, mut rz) = self.precondition(r, z, collective)?;
        let mut rz_previous = None;
        loop {
            if residual < options.threshold {
                let functional = -0.5 * (dot(mu.get_data(), field.get_data()) + dot(mu.get_data(), r));
                return Ok(SolveReport { iterations, residual, functional });
            }
            if iterations >= options.max_iterations {
                return Err(PolarizationError::NotConverged { iterations, residual });
            }
            // The direction: p = z + beta p, beta = (r . z) / (r . z) of the step before; p = z at the first.
            match rz_previous {
                None => p.copy_from_slice(z),
                Some(previous) => {
                    scale_in_place(p, rz / previous);
                    add_in_place(p, z);
                }
            }
            // q = A p.
            self.field_of_dipoles(&MatrixView::new(3, n_sites, &*p), &mut MatrixViewMut::new(3, n_sites, &mut *q), &mut work, collective)?;
            self.relay_from_coupling(p, q);
            iterations += 1;
            let mut curvature = [dot(p, q), dot(p, p)];
            collective.broadcast(&mut curvature)?;
            if curvature[0] <= 0.0 {
                return Err(PolarizationError::PolarizationCatastrophe { lowest_eigenvalue: curvature[0] / curvature[1] });
            }
            let step = rz / curvature[0];
            add_scaled_in_place(mu.get_data_mut(), p, step);
            add_scaled_in_place(r, q, -step);
            rz_previous = Some(rz);
            (residual, rz) = self.precondition(r, z, collective)?;
        }
    }

    // z = alpha r; returns (|z|, r . z), share 0's values on every share.
    fn precondition<C: Collective>(&self, r: &[f64], z: &mut [f64], collective: &C) -> Result<(f64, f64), PolarizationError> {
        let n_sites = self.n_sites();
        self.apply_alpha(&MatrixView::new(3, n_sites, r), &mut MatrixViewMut::new(3, n_sites, &mut *z))?;
        let mut values = [norm(z), dot(r, z)];
        collective.broadcast(&mut values)?;
        Ok((values[0], values[1]))
    }
}
