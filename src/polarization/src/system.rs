// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use crate::damping::Damping;
use crate::sym3::{self, Packed};
use crate::PolarizationError;
use crate::coupling::{CouplingWork, Whole};
use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use std::borrow::Cow;
use arrays::vector_math::dot;
use sources::{PolarizableSite, SiteTrait};

/// The per-site data derived from the polarizable sites once (positions, inverse polarizabilities, own
/// fragments, damping scales), owned so that it can be kept and reused while the sites are borrowed afresh
/// each time (see `PolarizableSystem::with_parameters`). About 9 doubles per site.
#[derive(Clone, Debug)]
pub struct SystemParameters {
    pub(crate) damping: Damping,
    pub(crate) alpha_inverse: Vec<Packed>,
    // n_sites x 3: column a holds coordinate a of every site, so the coupling's inner loop reads contiguously.
    pub(crate) positions: Matrix<f64>,
    // Own fragment of each site, 0 for none.
    pub(crate) own_fragment: Vec<i32>,
    // Whether any two sites share a non-zero own fragment, i.e. whether the coupling must test pairs at all.
    pub(crate) has_exclusions: bool,
    // (isotropic mean alpha)^(-1/6): the reduced distance of a pair is R * scale_i * scale_j.
    pub(crate) damping_scale: Vec<f64>,
}

impl SystemParameters {
    /// Derived from `sites`; fails for a singular polarizability.
    pub fn new(sites: &[PolarizableSite], damping: Damping) -> Result<Self, PolarizationError> {
        let mut alpha_inverse = Vec::with_capacity(sites.len());
        let mut own_fragment = Vec::with_capacity(sites.len());
        let mut damping_scale = Vec::with_capacity(sites.len());
        let mut positions = Matrix::<f64>::new(sites.len(), 3);
        for (axis, column) in positions.iter_mut().enumerate() {
            for (coordinate, site) in column.iter_mut().zip(sites) {
                *coordinate = site.get_position()[axis];
            }
        }
        for (site_index, site) in sites.iter().enumerate() {
            let alpha = sym3::from_polarizability(site.get_polarizability());
            alpha_inverse.push(sym3::invert(&alpha).ok_or(PolarizationError::SingularPolarizability { site: site_index })?);
            own_fragment.push(site.get_fragments().first().copied().unwrap_or(0));
            let mean_alpha = (alpha[0] + alpha[2] + alpha[5]) / 3.0;
            damping_scale.push(mean_alpha.powf(-1.0 / 6.0));
        }
        let mut fragments: Vec<i32> = own_fragment.iter().copied().filter(|&fragment| fragment != 0).collect();
        fragments.sort_unstable();
        let has_exclusions = fragments.windows(2).any(|pair| pair[0] == pair[1]);
        Ok(Self { damping, alpha_inverse, positions, own_fragment, has_exclusions, damping_scale })
    }

    /// The number of sites they were derived for.
    pub fn n_sites(&self) -> usize {
        self.alpha_inverse.len()
    }
}

/// A set of polarizable sites with their (optionally damped) dipole-dipole coupling.
///
/// Pairs are excluded from the coupling by fragment: two sites with the same non-zero own fragment (the
/// first fragment number of their record) do not polarise each other, the usual intramolecular exclusion
/// of the polarizable-embedding model. Fragment 0 (or no fragment) excludes nothing, a site never couples
/// to itself.
pub struct PolarizableSystem<'a> {
    sites: &'a [PolarizableSite],
    parameters: Cow<'a, SystemParameters>,
}

/// A vector of the 3 x n_sites layout: column `s` is (x, y, z) at site `s`.
pub(crate) fn check_shape<M: MatrixTrait<f64>>(matrix: &M, what: &'static str, n_sites: usize) -> Result<(), PolarizationError> {
    if (matrix.rows(), matrix.cols()) != (3, n_sites) {
        return Err(PolarizationError::Shape { what, expected: (3, n_sites), found: (matrix.rows(), matrix.cols()) });
    }
    Ok(())
}

impl<'a> PolarizableSystem<'a> {
    /// A system that owns its derived parameters.
    pub fn new(sites: &'a [PolarizableSite], damping: Damping) -> Result<Self, PolarizationError> {
        Ok(Self { sites, parameters: Cow::Owned(SystemParameters::new(sites, damping)?) })
    }

    /// A system over parameters derived earlier from these same sites: no allocation, no derivation.
    pub fn with_parameters(sites: &'a [PolarizableSite], parameters: &'a SystemParameters) -> Result<Self, PolarizationError> {
        if parameters.n_sites() != sites.len() {
            return Err(PolarizationError::ParametersMismatch { sites: sites.len(), parameters: parameters.n_sites() });
        }
        Ok(Self { sites, parameters: Cow::Borrowed(parameters) })
    }

    /// The number of sites.
    pub fn n_sites(&self) -> usize {
        self.sites.len()
    }

    /// The sites, in input order.
    pub fn sites(&self) -> &'a [PolarizableSite] {
        self.sites
    }

    /// The damping of the coupling.
    pub fn damping(&self) -> Damping {
        self.parameters.damping
    }

    /// The polarizability tensor of site `s`, packed [xx, xy, xz, yy, yz, zz].
    pub(crate) fn alpha_of(&self, site: usize) -> Packed {
        sym3::from_polarizability(self.sites[site].get_polarizability())
    }

    /// Whether the pair does not couple (same non-zero own fragment).
    pub fn excluded(&self, i: usize, j: usize) -> bool {
        self.parameters.own_fragment[i] != 0 && self.parameters.own_fragment[i] == self.parameters.own_fragment[j]
    }

    /// The (damped) dipole-dipole tensor T_ij, packed, or `None` if the pair is excluded: the field at
    /// site i of a unit dipole at site j is T_ij mu_j. Symmetric in i and j.
    pub fn tensor(&self, i: usize, j: usize) -> Result<Option<Packed>, PolarizationError> {
        if i == j || self.excluded(i, j) {
            return Ok(None);
        }
        let (position_i, position_j) = (self.sites[i].get_position(), self.sites[j].get_position());
        let d = [position_i[0] - position_j[0], position_i[1] - position_j[1], position_i[2] - position_j[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if r2 < 1e-16 {
            return Err(PolarizationError::CoincidentSites { first: i.min(j), second: i.max(j) });
        }
        let r = r2.sqrt();
        let (l3, l5) = self.parameters.damping.factors(r * self.parameters.damping_scale[i] * self.parameters.damping_scale[j]);
        // T_ab = (3 l5 d_a d_b / r^2 - l3 delta_ab) / r^3
        let inverse_r3 = 1.0 / (r * r2);
        let quadratic = 3.0 * l5 * inverse_r3 / r2;
        let isotropic = l3 * inverse_r3;
        Ok(Some([
            quadratic * d[0] * d[0] - isotropic,
            quadratic * d[0] * d[1],
            quadratic * d[0] * d[2],
            quadratic * d[1] * d[1] - isotropic,
            quadratic * d[1] * d[2],
            quadratic * d[2] * d[2] - isotropic,
        ]))
    }

    /// `field += T mu`: the field the dipoles produce at every site, in one process. O(n_sites^2); each pair
    /// is evaluated once and applied in both directions. Allocates its work (2 x 3 n_sites doubles): the
    /// iterative solver applies the coupling with `field_of_dipoles` and its scratch instead.
    pub fn add_field_of_dipoles<M, O>(&self, dipoles: &M, field: &mut O) -> Result<(), PolarizationError>
    where
        M: MatrixTrait<f64>,
        O: MatrixTraitMut<f64>,
    {
        check_shape(dipoles, "dipoles", self.n_sites())?;
        check_shape(field, "field", self.n_sites())?;
        let (mut dipoles_by_axis, mut field_by_axis) = (Matrix::<f64>::new(self.n_sites(), 3), Matrix::<f64>::new(self.n_sites(), 3));
        let mut coupled = Matrix::<f64>::new(3, self.n_sites());
        let mut work = CouplingWork::new(dipoles_by_axis.get_data_mut(), field_by_axis.get_data_mut(), self.n_sites());
        self.field_of_dipoles(dipoles, &mut coupled, &mut work, &Whole)?;
        for (total, value) in field.get_data_mut().iter_mut().zip(coupled.get_data()) {
            *total += value;
        }
        Ok(())
    }

    /// `out = alpha * v`, site by site.
    pub fn apply_alpha<M, O>(&self, v: &M, out: &mut O) -> Result<(), PolarizationError>
    where
        M: MatrixTrait<f64>,
        O: MatrixTraitMut<f64>,
    {
        check_shape(v, "input", self.n_sites())?;
        check_shape(out, "output", self.n_sites())?;
        for site in 0..self.n_sites() {
            let result = sym3::apply(&self.alpha_of(site), v.get_col(site));
            out.get_col_mut(site).copy_from_slice(&result);
        }
        Ok(())
    }

    /// The relay matrix applied to a vector: `out = (alpha^-1 - T) mu`.
    pub fn apply_relay<M, O>(&self, mu: &M, out: &mut O) -> Result<(), PolarizationError>
    where
        M: MatrixTrait<f64>,
        O: MatrixTraitMut<f64>,
    {
        check_shape(mu, "dipoles", self.n_sites())?;
        check_shape(out, "output", self.n_sites())?;
        out.fill(0.0);
        self.add_field_of_dipoles(mu, out)?;
        self.relay_from_coupling(mu.get_data(), out.get_data_mut());
        Ok(())
    }

    /// The relay matrix applied to a vector with the coupling already known: `out = alpha^-1 mu - coupled`
    /// for `coupled = T mu`. `out` may hold `coupled` itself (pass the same storage through `coupled_in_out`).
    pub(crate) fn relay_from_coupling(&self, mu: &[f64], coupled_in_out: &mut [f64]) {
        for ((out, mu_site), alpha_inverse) in coupled_in_out.chunks_exact_mut(3).zip(mu.chunks_exact(3)).zip(&self.parameters.alpha_inverse) {
            let local = sym3::apply(alpha_inverse, mu_site);
            for (value, local) in out.iter_mut().zip(local) {
                *value = local - *value;
            }
        }
    }

    /// The parameters the system was made with.
    pub(crate) fn parameters(&self) -> &SystemParameters {
        &self.parameters
    }

    /// The full relay matrix `alpha^-1 - T` as a dense 3S x 3S matrix (row/column 3s + a is component a of
    /// site s). Only for small systems: it is O(S^2) in memory.
    pub fn relay_matrix(&self) -> Result<Matrix<f64>, PolarizationError> {
        let n = self.n_sites();
        let mut relay = Matrix::<f64>::new(3 * n, 3 * n);
        for i in 0..n {
            let block = sym3::to_matrix(&self.parameters.alpha_inverse[i]);
            for a in 0..3 {
                for b in 0..3 {
                    relay[(3 * i + a, 3 * i + b)] = block[a][b];
                }
            }
            for j in 0..i {
                let Some(t) = self.tensor(i, j)? else { continue };
                let block = sym3::to_matrix(&t);
                for a in 0..3 {
                    for b in 0..3 {
                        relay[(3 * i + a, 3 * j + b)] = -block[a][b];
                        relay[(3 * j + b, 3 * i + a)] = -block[a][b];
                    }
                }
            }
        }
        Ok(relay)
    }

    /// The induction energy at the solution, `-1/2 mu . F`, for the total polarizing field `F`.
    pub fn induction_energy<M, G>(&self, mu: &M, field: &G) -> Result<f64, PolarizationError>
    where
        M: MatrixTrait<f64>,
        G: MatrixTrait<f64>,
    {
        check_shape(mu, "dipoles", self.n_sites())?;
        check_shape(field, "field", self.n_sites())?;
        Ok(-0.5 * dot(mu.get_data(), field.get_data()))
    }

    /// The functional the induced dipoles minimise, `1/2 mu . (alpha^-1 - T) mu - mu . F`; its value at the
    /// solution is `induction_energy`, and its error is second order in the dipoles' (the energy's is first).
    pub fn induction_functional<M, G>(&self, mu: &M, field: &G) -> Result<f64, PolarizationError>
    where
        M: MatrixTrait<f64>,
        G: MatrixTrait<f64>,
    {
        let mut relay_mu = Matrix::<f64>::new(3, self.n_sites());
        self.induction_functional_prealloc(mu, field, &mut relay_mu)
    }

    /// `induction_functional` with `(alpha^-1 - T) mu` made in `relay_mu` (3 x S, overwritten).
    pub fn induction_functional_prealloc<M, G, O>(&self, mu: &M, field: &G, relay_mu: &mut O) -> Result<f64, PolarizationError>
    where
        M: MatrixTrait<f64>,
        G: MatrixTrait<f64>,
        O: MatrixTraitMut<f64>,
    {
        check_shape(field, "field", self.n_sites())?;
        self.apply_relay(mu, relay_mu)?;
        Ok(0.5 * dot(mu.get_data(), relay_mu.get_data()) - dot(mu.get_data(), field.get_data()))
    }
}
