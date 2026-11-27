// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The dipole-dipole coupling `T mu` as the iterative solver applies it: the pair triangle (row i holds the pairs
// (i, j < i)) cut into contiguous runs of rows with nearly equal pair counts, one run per share (an MPI rank, say),
// each pair evaluated once and applied in both directions, the shares' fields summed by a `Collective`. The crate
// itself knows nothing of MPI: the caller supplies the shares and the sum.
//
// The pair loop works on the n_sites x 3 layout (column a holds component a of every site), so that its inner loop
// over j reads positions and dipoles and writes fields contiguously; the 3 x n_sites vectors of the interface are
// transposed in and out once per application, an O(n_sites) cost against the loop's O(n_sites^2).

use crate::damping::Damping;
use crate::system::check_shape;
use crate::{PolarizableSystem, PolarizationError};
use arrays::matrix::{MatrixTrait, MatrixTraitMut, MatrixViewMut};
use std::ops::Range;

/// Below this squared distance (bohr^2) two coupled sites coincide.
const COINCIDENT_SQUARED_DISTANCE: f64 = 1e-16;

/// The shares the coupling is split over and the collectives that join them. Every share must call the
/// solver with the same arguments; the solver then calls `sum` and `broadcast` the same number of times on each.
pub trait Collective {
    /// This share and the number of shares: `(0, 1)` for a single process.
    fn share(&self) -> (usize, usize);
    /// `values` summed over the shares, the result in place on every share.
    fn sum(&self, values: &mut [f64]) -> Result<(), PolarizationError>;
    /// Share 0's `values` on every share, so that all take the same decisions.
    fn broadcast(&self, values: &mut [f64]) -> Result<(), PolarizationError>;
}

/// A single process: the whole coupling, nothing to sum or broadcast.
pub struct Whole;

impl Collective for Whole {
    fn share(&self) -> (usize, usize) {
        (0, 1)
    }
    fn sum(&self, _: &mut [f64]) -> Result<(), PolarizationError> {
        Ok(())
    }
    fn broadcast(&self, _: &mut [f64]) -> Result<(), PolarizationError> {
        Ok(())
    }
}

/// The rows of the pair triangle that share `part` of `parts` evaluates: contiguous, together all rows once,
/// each run holding about `n (n - 1) / (2 parts)` pairs.
pub fn pair_rows(n_sites: usize, part: usize, parts: usize) -> Range<usize> {
    // The first row of run k: the smallest b with b (b - 1) / 2 >= k / parts of the pairs.
    let start = |k: usize| -> usize {
        if k >= parts {
            return n_sites;
        }
        if k == 0 || n_sites < 2 {
            return 0;
        }
        let target = (n_sites * (n_sites - 1) / 2) as f64 * k as f64 / parts as f64;
        let row = ((1.0 + (1.0 + 8.0 * target).sqrt()) / 2.0).ceil() as usize;
        row.min(n_sites)
    };
    start(part)..start(part + 1)
}

/// The work of one application of the coupling: the dipoles and the field on the n_sites x 3 layout.
pub struct CouplingWork<'a> {
    dipoles: MatrixViewMut<'a, f64>,
    field: MatrixViewMut<'a, f64>,
}

impl<'a> CouplingWork<'a> {
    /// Over two slices of `3 n_sites` doubles each.
    pub fn new(dipoles: &'a mut [f64], field: &'a mut [f64], n_sites: usize) -> Self {
        Self { dipoles: MatrixViewMut::new(n_sites, 3, dipoles), field: MatrixViewMut::new(n_sites, 3, field) }
    }
}

impl PolarizableSystem<'_> {
    /// `field = T mu` (3 x n_sites, overwritten): this share's rows of the pair triangle (`pair_rows`), then the
    /// shares summed, so every share holds the whole field. Two coincident sites that are not excluded from each
    /// other are `CoincidentSites` on every share.
    pub fn field_of_dipoles<M, O, C>(&self, dipoles: &M, field: &mut O, work: &mut CouplingWork<'_>, collective: &C) -> Result<(), PolarizationError>
    where
        M: MatrixTrait<f64>,
        O: MatrixTraitMut<f64>,
        C: Collective,
    {
        let n_sites = self.n_sites();
        check_shape(dipoles, "dipoles", n_sites)?;
        check_shape(field, "field", n_sites)?;
        dipoles.transpose_prealloc(&mut work.dipoles);
        work.field.fill(0.0);
        let (part, parts) = collective.share();
        let rows = pair_rows(n_sites, part, parts);
        let parameters = self.parameters();
        let nearest = match (parameters.damping, parameters.has_exclusions) {
            (Damping::None, false) => add_rows(self, rows, &work.dipoles, &mut work.field, |_, _, _| (1.0, 1.0), |_, _| false),
            (Damping::None, true) => add_rows(self, rows, &work.dipoles, &mut work.field, |_, _, _| (1.0, 1.0), |i, j| self.excluded(i, j)),
            (damping @ Damping::Thole { .. }, exclusions) => {
                let scale = &parameters.damping_scale;
                let factors = |i: usize, j: usize, r: f64| damping.factors(r * scale[i] * scale[j]);
                if exclusions {
                    add_rows(self, rows, &work.dipoles, &mut work.field, factors, |i, j| self.excluded(i, j))
                } else {
                    add_rows(self, rows, &work.dipoles, &mut work.field, factors, |_, _| false)
                }
            }
        };
        work.field.transpose_prealloc(field);
        if nearest < COINCIDENT_SQUARED_DISTANCE {
            // Spread the finding to every share through the sum.
            field.get_data_mut()[0] = f64::NAN;
        }
        collective.sum(field.get_data_mut())?;
        if field.get_data().iter().any(|value| !value.is_finite()) {
            if let Some((first, second)) = self.coincident_pair() {
                return Err(PolarizationError::CoincidentSites { first, second });
            }
        }
        Ok(())
    }

    // The first coupled pair closer than COINCIDENT_SQUARED_DISTANCE, searched over all pairs: only on the
    // error path.
    fn coincident_pair(&self) -> Option<(usize, usize)> {
        let positions = &self.parameters().positions;
        let squared_distance = |i: usize, j: usize| (0..3).map(|axis| (positions[(i, axis)] - positions[(j, axis)]).powi(2)).sum::<f64>();
        (0..self.n_sites())
            .flat_map(|i| (0..i).map(move |j| (j, i)))
            .find(|&(j, i)| !self.excluded(i, j) && squared_distance(i, j) < COINCIDENT_SQUARED_DISTANCE)
    }
}

// `field += T dipoles` over the pairs of `rows` (n_sites x 3 layout), both directions of each pair; returns the
// smallest squared distance of a coupled pair met. `factors(i, j, r)` gives the damping factors (l3, l5) of the
// pair and `excluded(i, j)` whether it does not couple; with constant closures the compiler drops both.
//
//   T_ij mu_j = 3 l5 (d . mu_j) d / r^5 - l3 mu_j / r^3,   d = R_i - R_j,
//
// symmetric in i and j, so the same pair gives T_ji mu_i with mu_i in place of mu_j.
#[inline(always)]
fn add_rows(
    system: &PolarizableSystem<'_>,
    rows: Range<usize>,
    dipoles: &MatrixViewMut<'_, f64>,
    field: &mut MatrixViewMut<'_, f64>,
    factors: impl Fn(usize, usize, f64) -> (f64, f64),
    excluded: impl Fn(usize, usize) -> bool,
) -> f64 {
    let positions = &system.parameters().positions;
    let (x, y, z) = (positions.get_col(0), positions.get_col(1), positions.get_col(2));
    let (mu_x, mu_y, mu_z) = (dipoles.get_col(0), dipoles.get_col(1), dipoles.get_col(2));
    let mut columns = field.iter_mut();
    let (Some(field_x), Some(field_y), Some(field_z)) = (columns.next(), columns.next(), columns.next()) else {
        return f64::INFINITY;
    };
    let mut nearest = f64::INFINITY;
    for i in rows {
        let (xi, yi, zi) = (x[i], y[i], z[i]);
        let (ax, ay, az) = (mu_x[i], mu_y[i], mu_z[i]);
        let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
        let positions_j = x[..i].iter().zip(&y[..i]).zip(&z[..i]);
        let dipoles_j = mu_x[..i].iter().zip(&mu_y[..i]).zip(&mu_z[..i]);
        let fields_j = field_x[..i].iter_mut().zip(field_y[..i].iter_mut()).zip(field_z[..i].iter_mut());
        for (j, (((xj, yj), zj), (((bx, by), bz), ((fx, fy), fz)))) in positions_j.zip(dipoles_j.zip(fields_j)).enumerate() {
            if excluded(i, j) {
                continue;
            }
            let (dx, dy, dz) = (xi - xj, yi - yj, zi - zj);
            let r2 = dx * dx + dy * dy + dz * dz;
            nearest = nearest.min(r2);
            let r = r2.sqrt();
            let (l3, l5) = factors(i, j, r);
            let inverse_r = 1.0 / r;
            let inverse_r3 = inverse_r * inverse_r * inverse_r;
            let quadratic = 3.0 * l5 * inverse_r3 * inverse_r * inverse_r;
            let isotropic = l3 * inverse_r3;
            let d_mu_j = dx * bx + dy * by + dz * bz;
            let d_mu_i = dx * ax + dy * ay + dz * az;
            gx += quadratic * d_mu_j * dx - isotropic * bx;
            gy += quadratic * d_mu_j * dy - isotropic * by;
            gz += quadratic * d_mu_j * dz - isotropic * bz;
            *fx += quadratic * d_mu_i * dx - isotropic * ax;
            *fy += quadratic * d_mu_i * dy - isotropic * ay;
            *fz += quadratic * d_mu_i * dz - isotropic * az;
        }
        field_x[i] += gx;
        field_y[i] += gy;
        field_z[i] += gz;
    }
    nearest
}
