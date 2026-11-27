// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The environment of the QM region: AIMPs (core and embedding) and their operators, external point charges and
// multipoles, polarizable sites and the induced-dipole polarization, the electric field at library-owned points, and
// the classical (nucleus-level) embedding energies.
//
// The electric field: which points to evaluate it at (`SiteSelection`), the electric-field integrals there, and the
// electronic field in chunks of sites (the `efield` crate).
//
// A selection is never materialised as a list of sites: it is a sequence of *segments*, each
// the region's own slice (nuclei, core AIMPs, ...) of its own concrete type, and the kernel is
// instantiated once per site type. Nothing is copied or converted; a selection spanning two
// regions (`Qm`) just visits two slices.
//
// Induced-dipole embedding at the context level, as a small state machine with one call per SCF iteration:
//
//   set_polarization(options)            once per geometry (after set_polarizable_sites, and after any
//                                        external multipoles): derives the per-site parameters, allocates
//                                        every buffer, and (with `StaticFieldPolicy::Cache`) computes the
//                                        static field at the polarizable sites once;
//   compute_induction(density, v_ind)    once per Fock build: the electronic field of the density, the
//                                        induced dipoles (warm-started from the previous call), the
//                                        induction energy, and the induction operator to add to the Fock
//                                        matrix -- the sequence field -> solve -> operator, with its two passes
//                                        over the electric-field integrals, kept inside the library;
//   free_polarization()                  drops the state.
//
// The pieces (polarizing field, solve, operator) are deliberately not exposed: callers cannot get the
// order or the warm start wrong. The buffers made by set_polarization (fields, dipoles, the solver's
// scratch, the integral chunk buffer) are reused for every Fock build; the integral evaluation itself
// works in the shell-pair scratch.
//
// Physics. With F(D) = F_static + F_electronic(D) the field polarising the sites, mu = (alpha^-1 - T)^-1 F
// and E_ind = -1/2 mu . F, the derivative of E_ind with respect to the density is
//   V_pq = dE_ind/dD_pq = - Sum_(s,a) mu_(s,a) t_(a,pq)(R_s),
// t being the electric-field integrals (electron charge included, as in `compute_electronic_field`), the
// 1/2 cancelling because the relay matrix is symmetric. V is what is added to the one-electron
// Hamiltonian; E_ind (not Tr(D V)) is what is added to the total energy. The energy reported is the
// functional the dipoles minimise, 1/2 mu . (alpha^-1 - T) mu - mu . F: equal to -1/2 mu . F at the
// solution, but second order in the dipoles' error where -1/2 mu . F is first order, so the solver's
// threshold does not leave first-order noise in the energy an SCF converges on; and at fixed dipoles its
// derivative with respect to the density is V exactly.

use super::*;
use arrays::array3d::Array3DTraitMut;
use basis::charge_carriers::{ExternalMultipoles, NuclearPotential, Site, SiteTrait, fragment_excludes};
use std::ops::Range;
use arrays::matrix::{Matrix, MatrixView, MatrixViewMut, MatrixTrait, MatrixTraitMut};
use arrays::vector_math::add_scaled_in_place;
use polarization::{Collective, Damping, InitialGuess, IterativeBuffers, PolarizableSystem, PolarizationError, SolveReport, SolverOptions, SystemParameters};

impl AedelstenContext {
    /// Whether the QM region has core AIMPs.
    pub fn has_core_aimp(&self) -> bool { dispatch!(self, has_core_aimp()) }

    /// Whether the input has embedding AIMPs.
    pub fn has_embedding_aimp(&self) -> bool { dispatch!(self, has_embedding_aimp()) }

    /// The number of embedding AIMPs (0 without any).
    pub fn get_n_embedding_aimps(&self) -> usize { dispatch!(self, get_n_embedding_aimps()) }

    /// Every embedding AIMP's site in input order: the atomic number of its block's element (0 when the block's name
    /// is not an element symbol), its position (bohr) and its charge; `output.len()` must equal
    /// `get_n_embedding_aimps()`. The environment's atoms, for a model of the QM region's bonds to it.
    pub fn get_embedding_aimp_sites(&self, output: &mut [basis::EmbeddingAimpSite]) -> Result<(), AedelstenError> {
        dispatch!(self, get_embedding_aimp_sites(output))
    }

    /// Whether external multipoles are set (`set_external_multipoles`).
    pub fn has_external_multipoles(&self) -> bool { dispatch!(self, has_external_multipoles()) }

    /// Whether polarizable sites are set (`set_polarizable_sites`).
    pub fn has_polarizable_sites(&self) -> bool { dispatch!(self, has_polarizable_sites()) }

    /// The energy of the QM region's nuclei and core AIMPs in the embedding AIMPs' screened potentials; 0 without
    /// embedding AIMPs.
    pub fn compute_nuclear_aimp_embedding_energy(&self) -> f64 { dispatch!(self, compute_nuclear_aimp_embedding_energy()) }

    /// The energy of the QM region's nuclei and core AIMPs in the external multipoles' potential; NaN without external
    /// multipoles.
    pub fn compute_nuclear_multipole_interaction_energy(&self) -> f64 { dispatch!(self, compute_nuclear_multipole_interaction_energy()) }

    /// Reads the external point charges and multipoles of the xfield file `xfield`. An error if some are already set
    /// (`free_external_multipoles`), or while a polarization state holds the static field they contribute to.
    pub fn set_external_multipoles(&mut self, xfield: &str) -> Result<(), AedelstenError> { dispatch!(self, set_external_multipoles(xfield)) }

    /// Frees the external multipoles. A static field a polarization state already holds is kept: it was computed with
    /// them.
    pub fn free_external_multipoles(&mut self) { dispatch!(self, free_external_multipoles()) }

    /// Reads the polarizable sites (positions and polarizabilities) of the xfield file `xfield`, a list of its own,
    /// apart from the external multipoles. An error if some are already set (`free_polarizable_sites`).
    pub fn set_polarizable_sites(&mut self, xfield: &str) -> Result<(), AedelstenError> { dispatch!(self, set_polarizable_sites(xfield)) }

    /// Frees the polarizable sites and the polarization state derived from them.
    pub fn free_polarizable_sites(&mut self) { dispatch!(self, free_polarizable_sites()) }

    /// Sets up the induced-dipole polarization of the polarizable sites (`set_polarizable_sites` first, and the
    /// external multipoles before it if any): the sites' parameters, every buffer of the Fock builds' calls, the static
    /// field if `options` keeps it, and how many sites' field integrals each rank keeps under
    /// `options.resident_field_budget` (computed at the first field pass). Collective under MPI. An error if it is
    /// already set up (`free_polarization`).
    pub fn set_polarization(&mut self, options: PolarizationOptions) -> Result<(), AedelstenError> {
        let damping = options.damping;
        dispatch!(self, set_polarization(options)).inspect(|_| crate::citations::polarization(&damping))
    }

    /// Frees the polarization state (its dipoles, buffers and kept integrals).
    pub fn free_polarization(&mut self) { dispatch!(self, free_polarization()) }

    /// Whether a polarization state is set up (`set_polarization`).
    pub fn has_polarization(&self) -> bool { dispatch!(self, has_polarization()) }

    /// `dE_ind/dX` at fixed `density` into a column-major `(3, get_n_gradient_centres())` matrix, OVERWRITTEN, from the
    /// dipoles of the last `compute_induction` (call it with the same density first). See
    /// `context/embedding.rs`. `active` is the frozen-centre mask of the other gradient calls.
    pub fn compute_induction_gradient(&self, gradient: &mut [f64], density: &[f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        dispatch!(self, compute_induction_gradient(gradient, density, active))
    }

    /// One SCF iteration's induction: from the (1-RDM) `density` (`n_cont x n_cont`) computes the induced
    /// dipoles and *overwrites* `v_ind` (`n_cont x n_cont`) with the induction operator to add to the
    /// one-electron Hamiltonian, and returns the energy to add to the total energy. See the module comment.
    pub fn compute_induction(&self, density: &[f64], v_ind: &mut [f64]) -> Result<InductionReport, AedelstenError> {
        dispatch!(self, compute_induction(density, v_ind))
    }

    /// `compute_induction` with the dipoles solved to `threshold` (positive; the norm of the preconditioned residual,
    /// as `SolverOptions::threshold`) for this call alone, in place of the threshold `set_polarization` set up. The
    /// energy is the variational functional of the dipoles reached, its error quadratic in the residual. For a caller
    /// that converges the density and the dipoles together: a looser threshold while the density is far from
    /// converged, the setup's at the end.
    pub fn compute_induction_within(&self, density: &[f64], v_ind: &mut [f64], threshold: f64) -> Result<InductionReport, AedelstenError> {
        dispatch!(self, compute_induction_within(density, v_ind, threshold))
    }

    /// How this rank holds the field integrals of its share of the polarizable sites: how many it keeps in memory (the
    /// rest evaluated in every field pass), and the bytes keeping them all needs (for choosing
    /// `PolarizationOptions::resident_field_budget`). Needs `set_polarization`.
    pub fn get_polarization_storage(&self) -> Result<PolarizationStorage, AedelstenError> {
        dispatch!(self, get_polarization_storage())
    }

    /// The memory budget of this rank's kept field integrals, in place of `PolarizationOptions::resident_field_budget`
    /// (`None` or a budget below one site: none kept): the integrals kept now go, and those of the first sites of the
    /// share that `budget` holds are computed at the next pass and kept after; the rest are evaluated in every pass.
    /// For a host that learns what it can give them only after `set_polarization`. Collective (every rank, each with
    /// its own budget). Needs `set_polarization`.
    pub fn set_resident_field_budget(&mut self, budget: Option<(usize, arrays::scratch::ByteUnit)>) -> Result<(), AedelstenError> {
        dispatch!(self, set_resident_field_budget(budget))
    }

    /// The linear response of the induction to changes of the density: for each of the `n_mat` densities
    /// `D1_m` (`n_cont x n_cont`, consecutive in `densities`; need not be symmetric) *overwrites* the
    /// matching block of `operators` with
    ///   V1_m = -Sum mu1_m . t,   mu1_m = (alpha^-1 - T)^-1 F_electronic(D1_m),
    /// the derivative of the induction operator along D1_m (no static field; the ground-state dipoles of
    /// `compute_induction` are not touched, and the operator is linear in D1, so the same call serves RPA/CI/CC
    /// trial vectors, transition densities and their symmetric or antisymmetric parts alike). The dipoles are
    /// solved with the same solver options as the ground state, from the uncoupled guess. Batches of at most
    /// `PolarizationOptions::response_batch` densities share one evaluation of the integrals per pass.
    pub fn compute_induction_response(&self, densities: &[f64], n_mat: usize, operators: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, compute_induction_response(densities, n_mat, operators))
    }

    /// The induction's couplings between `n_mat` densities `D_m` (consecutive `n_cont x n_cont` blocks of `densities`;
    /// need not be symmetric): *overwrites* `couplings` (`n_mat x n_mat`, column-major) with
    ///   K_mn = -f_m . (alpha^-1 - T)^-1 f_n,   f_m = F_electronic(D_m),
    /// the electrons' field of each density at the sites (no static field) against the dipoles another's induces:
    /// the second derivative of the induction energy along the two densities, equal to `tr(D_m V1_n)` with `V1_n` the
    /// operator `compute_induction_response` gives for `D_n`, without forming any operator. Symmetric to the solver's
    /// threshold (the relay matrix is). The dipoles are solved with the setup's solver options from the uncoupled
    /// guess; the ground state's dipoles are not touched. Batches of at most `PolarizationOptions::response_batch`
    /// densities share one evaluation of the integrals in the field pass (needs `response_batch >= 1`). One-shot: it
    /// allocates the fields of all the densities, `3 n_sites n_mat` doubles. Collective under MPI.
    pub fn compute_induction_couplings(&self, densities: &[f64], n_mat: usize, couplings: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, compute_induction_couplings(densities, n_mat, couplings))
    }

    /// `compute_induction` for `n_states` densities (consecutive `n_cont x n_cont` blocks of `densities`),
    /// each with its *own* induced dipoles: state `m` gets the dipoles, energy `energies[m]` and operator
    /// block `m` of `operators` of its density alone, `E_ind(D_m)` and `V_m = dE_ind/dD_m`. For a
    /// state-average energy the caller weights them, sum_m w_m E_m and sum_m w_m V_m; this differs from
    /// `compute_induction` on the averaged density, because E_ind is quadratic in D. (Both are valid
    /// choices.) Each state's dipoles are kept and warm-started at the next call, so the states must come
    /// in the same order every time. One evaluation of the integrals per pass serves all the states.
    /// Needs `PolarizationOptions::state_slots >= n_states`.
    pub fn compute_induction_states(&self, densities: &[f64], n_states: usize, operators: &mut [f64], energies: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, compute_induction_states(densities, n_states, operators, energies))
    }

    /// `compute_induction_states` with every state's dipoles solved to `threshold` (positive) for this call alone, in
    /// place of the setup's, as `compute_induction_within` does for one density.
    pub fn compute_induction_states_within(&self, densities: &[f64], n_states: usize, operators: &mut [f64], energies: &mut [f64], threshold: f64) -> Result<(), AedelstenError> {
        dispatch!(self, compute_induction_states_within(densities, n_states, operators, energies, threshold))
    }

    /// The effective external field (EEF) operators of the polarizable sites (N. H. List, H. J. Aa. Jensen,
    /// J. Kongsted, Phys. Chem. Chem. Phys. 18, 10070 (2016)): for each direction i = x, y, z of a uniform unit
    /// external field, the sites' induced dipoles `mu^(i) = (alpha^-1 - T)^-1 e_i` (every site in the field `e_i`,
    /// coupled and damped as `set_polarization` set them up) and their operator on the electrons
    /// `V^(i) = -Sum_s mu_s^(i) . t_s`, *overwriting* `operators`: three consecutive n_cont x n_cont blocks, x, y, z.
    /// Added to the dipole-moment integrals of the same component, `V^(i)` makes their matrix elements the response
    /// to the external field rather than to the local one; a transition moment of the sum includes the dipoles the
    /// transition density induces at the sites (the relay matrix is symmetric). Only the external field polarizes
    /// the sites here: no static field, no density. Not picture-changed, as the induction operator. Zero blocks
    /// without polarizable sites. Needs `set_polarization`. One-shot: it allocates three dipole sets (9 doubles per
    /// site) and a field (3 per site). Collective under MPI.
    pub fn compute_effective_external_field(&self, operators: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, compute_effective_external_field(operators)).inspect(|_| crate::citations::effective_external_field())
    }

    /// The induced dipoles of the last `compute_induction` (3 values per polarizable site, in the order of
    /// the polarizable sites; see `get_input_indices`).
    pub fn get_induced_dipoles(&self, output: &mut [f64]) -> Result<(), AedelstenError> { dispatch!(self, get_induced_dipoles(output)) }

    /// The dipoles the next `compute_induction` starts its solve from (`3 n_sites`, as `get_induced_dipoles` gives
    /// them): another geometry's, in an optimization. The solution depends on the start only within the solver's
    /// tolerance; the dipoles are not a solution until that solve (`get_induced_dipoles` and the induction gradient
    /// still need it).
    pub fn set_induced_dipoles(&mut self, dipoles: &[f64]) -> Result<(), AedelstenError> { dispatch!(self, set_induced_dipoles(dipoles)) }

    /// Builds the embedding AIMPs' projection operators (their projectors and spectrally resolved exchange), needed by
    /// their projection and exchange operators; none without embedding AIMPs. An error if they are already built
    /// (`free_embedding_aimp_operators`).
    pub fn set_embedding_aimp_operators(&mut self) -> Result<(), AedelstenError> { dispatch!(self, set_embedding_aimp_operators()) }

    /// Frees the embedding AIMPs' projection operators.
    pub fn free_embedding_aimp_operators(&mut self) { dispatch!(self, free_embedding_aimp_operators()) }

    /// The number of sites in `selection`.
    pub fn count_electric_field_sites(&self, selection: SiteSelection<'_>) -> Result<usize, AedelstenError> {
        dispatch!(self, count_electric_field_sites(selection))
    }

    /// Electric-field integrals `< i | (r - C)/|r - C|^3 | j >` for the sites `site_range` of the
    /// selection (indices in selection order), written to `output` as an
    /// `n_cont x n_cont x 3*site_range.len()` array, (x, y, z) planes per site. The range is what
    /// makes this batchable: a caller with more sites than fit in memory loops over ranges, see
    /// `efield::resolve_field_chunk_size`.
    pub fn compute_electric_field_integrals(
        &self,
        selection: SiteSelection<'_>,
        site_range: std::ops::Range<usize>,
        output: &mut [f64],
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_electric_field_integrals(selection, site_range, output))
    }

    /// For each site of `selection` (in selection order), the position of the record of its input
    /// file it was read from (0-based), or `None` if it did not come from a file (nuclei, AIMPs,
    /// custom sites). Records dropped because they carry nothing for that list (all-zero moments,
    /// zero polarizability) still count, so this maps field/dipole vectors, which are indexed by
    /// the retained sites, back to the lines of the input.
    pub fn get_input_indices(&self, selection: SiteSelection<'_>) -> Result<Vec<Option<usize>>, AedelstenError> {
        dispatch!(self, get_input_indices(selection))
    }

    /// The static (classical) electric field at the selected sites from the enabled sources,
    /// `E = -grad(phi)` with `phi` the potential of positive nuclei/AIMP charges and the external
    /// multipoles, *overwriting* `field` (`3 * n_sites`, (x, y, z) per site). Together with
    /// `compute_electronic_field` this is the total field. A source coinciding with a site is
    /// skipped for that site (see `COINCIDENCE_TOLERANCE`).
    pub fn compute_static_field(&self, selection: SiteSelection<'_>, sources: StaticFieldSources, field: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, compute_static_field(selection, sources, field))
    }

    /// Electronic electric field at the selected sites, `sum_ij D_ij < i | (r - C)/|r - C|^3 | j >`,
    /// for a plain 1-RDM `density` (`n_cont * n_cont`, no charge sign). `field` is a `3 * n_sites`
    /// slice receiving (x, y, z) per site. Nuclear and other classical contributions are not included. The density is
    /// contracted in the Hermite basis of each shell pair, with no integral formed per site; `chunking` sets the sites
    /// handed out over the ranks as one piece.
    pub fn compute_electronic_field(
        &self,
        density: &[f64],
        selection: SiteSelection<'_>,
        chunking: &SiteChunking,
        field: &mut [f64],
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_electronic_field(density, selection, chunking, field))
    }

    /// The core AIMPs' point charges, `-Z/r`. An error without core AIMPs.
    pub fn get_core_aimp_point_charge_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_core_aimp_point_charge_operator())
    }

    /// The core AIMPs' screening corrections, `-Z s(r)/r`. An error without core AIMPs.
    pub fn get_core_aimp_screened_coulomb_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_core_aimp_screened_coulomb_operator())
    }

    /// The core AIMPs' Coulomb part, point charge and screen together, `-Z (1 + s(r))/r`. An error without core AIMPs.
    pub fn get_core_aimp_coulomb_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_core_aimp_coulomb_operator())
    }

    /// The core AIMPs' projection operator, `Sum_sites S W S^T` with `W` the projector of the frozen orbitals. An error
    /// without core AIMPs.
    pub fn get_core_aimp_projection_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_core_aimp_projection_operator())
    }

    /// The core AIMPs' spectrally resolved exchange, `Sum_sites S K_sro S^T`. An error without core AIMPs.
    pub fn get_core_aimp_spectrally_resolved_exchange_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_core_aimp_spectrally_resolved_exchange_operator())
    }

    /// The core AIMPs' projection and spectrally resolved exchange together, in one pass. An error without core AIMPs.
    pub fn get_core_aimp_exchange_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_core_aimp_exchange_operator())
    }

    /// The embedding AIMPs' point charges, `-Z/r`. An error without embedding AIMPs.
    pub fn get_embedding_aimp_point_charge_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_embedding_aimp_point_charge_operator())
    }

    /// The embedding AIMPs' screening corrections, `-Z s(r)/r`. An error without embedding AIMPs.
    pub fn get_embedding_aimp_screened_coulomb_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_embedding_aimp_screened_coulomb_operator())
    }

    /// The embedding AIMPs' Coulomb part, point charge and screen together, `-Z (1 + s(r))/r`. An error without
    /// embedding AIMPs.
    pub fn get_embedding_aimp_coulomb_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_embedding_aimp_coulomb_operator())
    }

    /// The embedding AIMPs' projection operator, `Sum_sites S W S^T`. Needs `set_embedding_aimp_operators`.
    pub fn get_embedding_aimp_projection_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_embedding_aimp_projection_operator())
    }

    /// The embedding AIMPs' spectrally resolved exchange, `Sum_sites S K_sro S^T`. Needs
    /// `set_embedding_aimp_operators`.
    pub fn get_embedding_aimp_spectrally_resolved_exchange_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_embedding_aimp_spectrally_resolved_exchange_operator())
    }

    /// The embedding AIMPs' projection and spectrally resolved exchange together, in one pass. Needs
    /// `set_embedding_aimp_operators`.
    pub fn get_embedding_aimp_exchange_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_embedding_aimp_exchange_operator())
    }

    /// The external point charges and multipoles' operator on the electrons, in the convention
    /// `set_multipole_convention` chose. An error without external multipoles.
    pub fn get_external_multipole_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_external_multipole_operator())
    }

    /// Which interaction of the environment's quadrupoles and higher with the electrons the external-multipole
    /// operator represents: `MultipoleConvention::ContactFree` (the default, PElib/Dalton) or `Exact`. Only that
    /// operator is affected; the classical energies and fields are identical in both.
    pub fn set_multipole_convention(&mut self, convention: MultipoleConvention) { dispatch!(self, set_multipole_convention(convention)) }

    /// The convention of the external-multipole operator (`set_multipole_convention`).
    pub fn get_multipole_convention(&self) -> MultipoleConvention { dispatch!(self, get_multipole_convention()) }
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    pub fn compute_nuclear_multipole_interaction_energy(&self) -> f64 {
        // NaN, not a panic, without external multipoles: they are optional.
        let multipoles = match self.external_multipoles.get_cache() {
            Ok(m) => m,
            Err(_) => return f64::NAN,
        };
        let mut energy = 0.0;
        if let Some(nuclear_potential) = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_nuclear_potential() {
            match multipoles {
                ExternalMultipoles::Charges(multipoles) => {
                    energy += nuclear_potential.compute_nuclear_interaction_energy(multipoles);
                }
                ExternalMultipoles::Multipoles(multipoles) => {
                    energy += nuclear_potential.compute_general_multipole_interaction_energy(multipoles);
                }
            }
        }
        if let Some(core_regions) = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_core_regions() {
            match multipoles {
                ExternalMultipoles::Charges(multipoles) => {
                    energy += basis::charge_carriers::compute_screened_coulomb_interaction(multipoles, core_regions);
                }
                ExternalMultipoles::Multipoles(multipoles) => {
                    energy += basis::charge_carriers::compute_general_multipole_screened_interaction(multipoles, core_regions);
                }
            }
        }
        energy
    }

    pub fn compute_nuclear_aimp_embedding_energy(&self) -> f64 {
        let mut energy = 0.0;
        if let Some(nuclear_potential) = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_nuclear_potential() {
            if let Some(embedding_aimps) = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_embedding_potentials() {
                energy += nuclear_potential.compute_screened_coulomb_interaction(embedding_aimps);
            }
        }
        if let Some(core_regions) = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_core_regions() {
            if let Some(embedding_aimps) = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_embedding_potentials() {
                energy += basis::charge_carriers::compute_screened_coulomb_cross_interaction(core_regions, embedding_aimps);
            }
        }
        energy
    }

    pub fn has_core_aimp(&self) -> bool {
        self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_core_regions().is_some()
    }

    pub fn has_embedding_aimp(&self) -> bool {
        self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_embedding_potentials().is_some()
    }

    pub fn get_n_embedding_aimps(&self) -> usize {
        self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_embedding_potentials().map_or(0, |aimps| aimps.len())
    }

    pub fn get_embedding_aimp_sites(&self, output: &mut [basis::EmbeddingAimpSite]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().map_err(|_| AedelstenError::CacheNotReady("input has been freed (free_input)".to_string()))?;
        let aimps = qm_region.input.get_embedding_potentials().map_or(&[][..], |aimps| aimps.as_slice());
        if output.len() != aimps.len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: aimps.len(), actual: output.len() });
        }
        for (site, aimp) in output.iter_mut().zip(aimps) {
            *site = basis::EmbeddingAimpSite { atomic_number: aimp.get_atomic_number(), position: *aimp.get_position(), charge: aimp.get_point_charge().get_charge() };
        }
        Ok(())
    }

    pub fn has_external_multipoles(&self) -> bool {
        self.external_multipoles.is_ready()
    }

    pub fn has_polarizable_sites(&self) -> bool {
        self.polarizable_sites.is_ready()
    }

    pub fn set_multipole_convention(&mut self, convention: basis::charge_carriers::MultipoleConvention) {
        self.multipole_convention = convention;
    }

    pub fn get_multipole_convention(&self) -> basis::charge_carriers::MultipoleConvention {
        self.multipole_convention
    }

    // External multipoles.
    pub fn set_external_multipoles(&mut self, xfield: &str) -> Result<(), AedelstenError> {
        if self.polarization_caches_static_field() {
            return Err(AedelstenError::CacheAlreadyDerived(
                "The polarization state caches the static field, which depends on the external multipoles. Call free_polarization() before changing them (or set it up with StaticFieldPolicy::Recompute).".to_string(),
            ));
        }
        if self.external_multipoles.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived(
                "External multipoles are already allocated. Call free_external_multipoles() before allocating a new xfield.".to_string()
            ));
        }
        let external_multipoles = basis::charge_carriers::read_xfield(xfield)?;
        self.external_multipoles = AedelstenCacheState::Ready(external_multipoles);
        Ok(())
    }

    pub fn free_external_multipoles(&mut self) {
        // A kept static field was computed when the polarization was set up and stays valid (a host may free the
        // sources once it is kept); only changing the sources is refused (`set_external_multipoles`). With
        // `StaticFieldPolicy::Recompute` the sources present at each call are used.
        self.external_multipoles.debug_double_free_warning("xfield");
        self.external_multipoles.free_cache();
    }

    // Polarizable sites: a separate list from the external multipoles (the two need not
    // correspond one-to-one), read from the polarizability columns of a Molcas-style xfield file.
    pub fn set_polarizable_sites(&mut self, xfield: &str) -> Result<(), AedelstenError> {
        if self.polarizable_sites.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived(
                "Polarizable sites are already allocated. Call free_polarizable_sites() before allocating a new list.".to_string()
            ));
        }
        let polarizable_sites = basis::charge_carriers::read_xfield_polarizable_sites(xfield)?;
        self.polarizable_sites = AedelstenCacheState::Ready(polarizable_sites);
        Ok(())
    }

    pub fn free_polarizable_sites(&mut self) {
        // The polarization state is derived from these sites.
        self.free_polarization();
        self.polarizable_sites.debug_double_free_warning("polarizable sites");
        self.polarizable_sites.free_cache();
    }

    pub fn set_embedding_aimp_operators(&mut self) -> Result<(), AedelstenError> {
        if self.embedding_aimp_operators.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived(
                "Embedding AIMP projection operators have already been derived. Call free_embedding_aimp_operators() before deriving new ones.".to_string()
            ));
        }
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let operators = match qm_region.input.get_embedding_potentials() {
            Some(emb_aimps) => aimp::materialise_aimp_projection_operator(emb_aimps)?,
            None => Vec::new(),
        };
        self.embedding_aimp_operators = AedelstenCacheState::Ready(RefCell::new(operators));
        Ok(())
    }

    pub fn free_embedding_aimp_operators(&mut self) {
        self.embedding_aimp_operators.debug_double_free_warning("embedding AIMP projection operators");
        self.embedding_aimp_operators.free_cache();
    }

    pub fn get_external_multipole_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        Ok(crate::context::one_electron::OneElectronOperator::external_multipole(self.require_external_multipoles()?, self.multipole_convention))
    }

    /// The external multipoles, or the error saying to set them.
    pub(crate) fn require_external_multipoles(&self) -> Result<&basis::charge_carriers::ExternalMultipoles, AedelstenError> {
        self.external_multipoles.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("external multipoles are not resident in memory; call set_external_multipoles() first".to_string())
        })
    }

    pub fn get_core_aimp_point_charge_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        Ok(crate::context::one_electron::OneElectronOperator::core_aimp_point_charge(self.require_core_aimp_regions()?))
    }

    pub fn get_core_aimp_screened_coulomb_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        Ok(crate::context::one_electron::OneElectronOperator::core_aimp_screened_coulomb(self.require_core_aimp_regions()?))
    }

    pub fn get_core_aimp_coulomb_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        Ok(crate::context::one_electron::OneElectronOperator::core_aimp_coulomb(self.require_core_aimp_regions()?))
    }

    pub fn get_embedding_aimp_point_charge_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        Ok(crate::context::one_electron::OneElectronOperator::embedding_aimp_point_charge(self.require_embedding_aimp_regions()?))
    }

    pub fn get_embedding_aimp_screened_coulomb_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        Ok(crate::context::one_electron::OneElectronOperator::embedding_aimp_screened_coulomb(self.require_embedding_aimp_regions()?))
    }

    pub fn get_embedding_aimp_coulomb_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        Ok(crate::context::one_electron::OneElectronOperator::embedding_aimp_coulomb(self.require_embedding_aimp_regions()?))
    }

    /// The core AIMPs, or an error without any.
    pub(crate) fn require_core_aimp_regions(&self) -> Result<&[basis::IndexedAimp], AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        qm_region.input.get_core_regions().map(|v| v.as_slice()).ok_or_else(|| {
            AedelstenError::CacheNotReady(
                "no core AIMP regions in this input; check has_core_aimp() before requesting core AIMP integrals".to_string(),
            )
        })
    }

    /// The embedding AIMPs, or an error without any.
    pub(crate) fn require_embedding_aimp_regions(&self) -> Result<&[basis::Aimp], AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        qm_region.input.get_embedding_potentials().map(|v| v.as_slice()).ok_or_else(|| {
            AedelstenError::CacheNotReady(
                "no embedding AIMP regions in this input; check has_embedding_aimp() before requesting embedding AIMP integrals".to_string(),
            )
        })
    }

    pub fn get_core_aimp_projection_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        let (operators, aimps) = self.core_aimp_exchange_payload()?;
        Ok(crate::context::one_electron::OneElectronOperator::core_aimp_projection(operators, aimps))
    }

    pub fn get_core_aimp_spectrally_resolved_exchange_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        let (operators, aimps) = self.core_aimp_exchange_payload()?;
        Ok(crate::context::one_electron::OneElectronOperator::core_aimp_spectrally_resolved_exchange(operators, aimps))
    }

    pub fn get_core_aimp_exchange_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        let (operators, aimps) = self.core_aimp_exchange_payload()?;
        Ok(crate::context::one_electron::OneElectronOperator::core_aimp_exchange(operators, aimps))
    }

    pub fn get_embedding_aimp_projection_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        let (operators, aimps) = self.embedding_aimp_exchange_payload()?;
        Ok(crate::context::one_electron::OneElectronOperator::embedding_aimp_projection(operators, aimps))
    }

    pub fn get_embedding_aimp_spectrally_resolved_exchange_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        let (operators, aimps) = self.embedding_aimp_exchange_payload()?;
        Ok(crate::context::one_electron::OneElectronOperator::embedding_aimp_spectrally_resolved_exchange(operators, aimps))
    }

    pub fn get_embedding_aimp_exchange_operator(&self) -> Result<crate::context::one_electron::OneElectronOperator<'_>, AedelstenError> {
        let (operators, aimps) = self.embedding_aimp_exchange_payload()?;
        Ok(crate::context::one_electron::OneElectronOperator::embedding_aimp_exchange(operators, aimps))
    }

    /// The core AIMPs' projection operators and sites, or an error without core AIMPs.
    pub(crate) fn core_aimp_exchange_payload(&self) -> Result<(&RefCell<Vec<aimp::AimpProjectionOperator>>, &[basis::IndexedAimp]), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let aimps = qm_region.input.get_core_regions().map(|v| v.as_slice()).ok_or_else(|| {
            AedelstenError::CacheNotReady(
                "no core AIMP regions in this input; check has_core_aimp() before requesting core AIMP exchange integrals".to_string(),
            )
        })?;
        Ok((&qm_region.core_aimp_operators, aimps))
    }

    /// The embedding AIMPs' projection operators and sites, or an error without them or before
    /// `set_embedding_aimp_operators`.
    pub(crate) fn embedding_aimp_exchange_payload(&self) -> Result<(&RefCell<Vec<aimp::AimpProjectionOperator>>, &[basis::Aimp]), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let aimps = qm_region.input.get_embedding_potentials().map(|v| v.as_slice()).ok_or_else(|| {
            AedelstenError::CacheNotReady(
                "no embedding AIMP regions in this input; check has_embedding_aimp() before requesting embedding AIMP exchange integrals".to_string(),
            )
        })?;
        let embedding_aimp_operators = self.embedding_aimp_operators.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady(
                "embedding AIMP projection operators are not resident in memory; call set_embedding_aimp_operators() first".to_string(),
            )
        })?;
        Ok((embedding_aimp_operators, aimps))
    }

    fn visit_site_segments(&self, selection: SiteSelection<'_>, visitor: &mut impl SegmentVisitor) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        match selection {
            SiteSelection::Qm => {
                match qm_region.input.get_nuclear_potential() {
                    Some(NuclearPotential::PointCharge(charges)) => visitor.visit(charges)?,
                    Some(NuclearPotential::GaussianCharge(charges)) => visitor.visit(charges)?,
                    None => {}
                }
                if let Some(core_aimps) = qm_region.input.get_core_regions() {
                    visitor.visit(core_aimps)?;
                }
                Ok(())
            }
            SiteSelection::EmbeddingAimps => {
                let aimps = qm_region.input.get_embedding_potentials().ok_or_else(|| {
                    AedelstenError::CacheNotReady("no embedding AIMP regions in this input; check has_embedding_aimp() first".to_string())
                })?;
                visitor.visit(aimps)
            }
            SiteSelection::ExternalMultipoles => {
                let multipoles = self.external_multipoles.get_cache().map_err(|_| {
                    AedelstenError::CacheNotReady("external multipoles are not resident in memory; call set_external_multipoles() first".to_string())
                })?;
                match multipoles {
                    ExternalMultipoles::Charges(points) => visitor.visit(points),
                    ExternalMultipoles::Multipoles(points) => visitor.visit(points),
                }
            }
            SiteSelection::Polarizable => {
                let sites = self.polarizable_sites.get_cache().map_err(|_| {
                    AedelstenError::CacheNotReady("polarizable sites are not resident in memory; call set_polarizable_sites() first".to_string())
                })?;
                visitor.visit(sites)
            }
            SiteSelection::Custom(sites) => visitor.visit(sites),
        }
    }

    pub fn count_electric_field_sites(&self, selection: SiteSelection<'_>) -> Result<usize, AedelstenError> {
        let mut counter = CountSites(0);
        self.visit_site_segments(selection, &mut counter)?;
        Ok(counter.0)
    }

    pub fn get_input_indices(&self, selection: SiteSelection<'_>) -> Result<Vec<Option<usize>>, AedelstenError> {
        let mut indices = InputIndices(Vec::new());
        self.visit_site_segments(selection, &mut indices)?;
        Ok(indices.0)
    }

    // Integrals for one slice of one segment: `output` is n_cont x n_cont x 3*sites.len(), three
    // planes (x, y, z) per site; of the shell pairs `pairs` alone, on this rank (a piece of a field pass), or with
    // `None` of all of them, shared among the ranks.
    pub(crate) fn compute_electric_field_integrals_chunk<Site>(&self, output: &mut [f64], sites: &[Site], pairs: Option<std::ops::Range<usize>>) -> Result<(), AedelstenError>
    where
        Site: SiteTrait,
    {
        let n = self.get_n_cont();
        super::check_array3d_dimensions(n, 3 * sites.len(), output)?;
        let mut wrapped_output = arrays::array3d::Array3DViewMut::new(n, n, 3 * sites.len(), output);
        let shell_pairs = self.shell_pairs.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
        })?;
        let scratch_cell = self.scratch.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
        })?;
        let mut scratch = scratch_cell.borrow_mut();
        crate::context::one_electron::compute_electric_field_integrals_array3d(
            match &pairs { Some(pairs) => &shell_pairs[pairs.clone()], None => shell_pairs }, sites, &mut scratch, &self.comm, &mut wrapped_output, pairs.is_some(),
        )
    }

    // The electronic fields of `densities` (consecutive n_cont x n_cont matrices) at the sites of `pieces`, added to
    // `fields` (one column of 3 per site of `sites` a density), each piece's shell pairs against its sites on this
    // rank (`oneint::site_contractions::add_electronic_fields`).
    fn add_piece_fields<Site: SiteTrait>(&self, pieces: impl Iterator<Item = oneint::pieces::Piece>, densities: &[f64], sites: &[Site], fields: &mut [f64]) -> Result<(), AedelstenError> {
        let _span = instrumentation::span("polarization::site_fields");
        let n = self.get_n_cont();
        let shell_pairs = self.shell_pairs.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
        })?;
        let scratch_cell = self.scratch.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
        })?;
        let mut scratch = scratch_cell.borrow_mut();
        let column = 3 * sites.len();
        for piece in pieces {
            oneint::site_contractions::add_electronic_fields(
                fields, column, piece.sites.start, densities, n, &shell_pairs[piece.pairs], &sites[piece.sites], &mut scratch,
            )?;
        }
        Ok(())
    }

    // `operator_m += scale * Sum_s mu_(m,s) . <i| (r - C_s) / |r - C_s|^3 |j>` over the sites of `pieces`, for the
    // dipoles `dipoles` (one column of 3 per site a set) into `operators` (consecutive n_cont x n_cont matrices), as
    // the external-multipole operator of the dipoles `-scale mu` on this rank; `sources` holds a dipole per site at
    // its position, overwritten. Adds to the upper triangles only: `mirror_operators` completes them.
    fn add_piece_operators(
        &self,
        pieces: impl Iterator<Item = oneint::pieces::Piece>,
        dipoles: &[f64],
        scale: f64,
        sources: &mut [sources::PointDipole],
        operators: &mut [f64],
    ) -> Result<(), AedelstenError> {
        let _span = instrumentation::span("polarization::site_operators");
        let n = self.get_n_cont();
        let shell_pairs = self.shell_pairs.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
        })?;
        let scratch_cell = self.scratch.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
        })?;
        let mut scratch = scratch_cell.borrow_mut();
        let column = 3 * sources.len();
        for piece in pieces {
            for (mu, operator) in dipoles.chunks_exact(column).zip(operators.chunks_exact_mut(n * n)) {
                for (source, mu_s) in sources[piece.sites.clone()].iter_mut().zip(mu[3 * piece.sites.start..3 * piece.sites.end].chunks_exact(3)) {
                    source.set_dipole([-scale * mu_s[0], -scale * mu_s[1], -scale * mu_s[2]]);
                }
                oneint::site_contractions::add_external_multipole_upper(
                    &mut MatrixViewMut::new(n, n, operator), &shell_pairs[piece.pairs.clone()], &sources[piece.sites.clone()], &mut scratch,
                )?;
            }
        }
        Ok(())
    }

    // The pieces of a field pass over the sites `runs` (ranges in site order) in runs of at most `max_sites`: runs of
    // the shell pairs against runs of the sites (`oneint::pieces`, cut as if the runs were one), in hand-out order.
    fn field_pieces(&self, runs: &[Range<usize>], max_sites: usize) -> Result<Vec<oneint::pieces::Piece>, AedelstenError> {
        let shell_pairs = self.shell_pairs.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
        })?;
        let costs: Vec<f64> = shell_pairs.iter().map(oneint::pieces::pair_cost).collect();
        let n_sites: usize = runs.iter().map(|run| run.len()).sum();
        let joined = oneint::pieces::sorted_pieces(&costs, n_sites, max_sites, oneint::pieces::Cut::SitesFirst, &self.comm);
        // Each piece's sites back from the joined count to the runs, cut where a run ends.
        let mut pieces = Vec::with_capacity(joined.len());
        for piece in joined {
            let (mut offset, mut first) = (0, piece.sites.start);
            for run in runs {
                let end = offset + run.len();
                if first < end && first < piece.sites.end {
                    let last = piece.sites.end.min(end);
                    pieces.push(oneint::pieces::Piece { pairs: piece.pairs.clone(), sites: run.start + (first - offset)..run.start + (last - offset) });
                    first = last;
                }
                offset = end;
            }
        }
        Ok(pieces)
    }

    // The polarization's field pieces (`field_pieces` over the sites no rank keeps, in chunks), made when the shell
    // pairs have changed in number or the kept sites since they were last made.
    fn ensure_field_pieces(&self, state: &mut PolarizationState) -> Result<(), AedelstenError> {
        let n_sites = state.electronic_field.cols();
        if n_sites == 0 {
            return Ok(());
        }
        let n_pairs = self.shell_pairs.get_cache().map(|pairs| pairs.len()).ok();
        if state.field_pieces_pairs != n_pairs || n_pairs.is_none() {
            let size = state.kept_by_rank.len();
            let rest: Vec<Range<usize>> = (0..size)
                .map(|rank| (rank * n_sites / size + state.kept_by_rank[rank], (rank + 1) * n_sites / size))
                .filter(|(start, end)| start < end)
                .map(|(start, end)| start..end)
                .collect();
            state.field_pieces = self.field_pieces(&rest, state.chunk_sites)?;
            state.field_pieces_pairs = n_pairs;
        }
        Ok(())
    }

    // The kept field integrals of this rank's share of the polarizable sites, computed if they are not (the first pass
    // after `set_polarization` or new shell pairs); nothing when they are not kept.
    fn fill_resident_field<Site: SiteTrait>(&self, state: &mut PolarizationState, sites: &[Site]) -> Result<(), AedelstenError> {
        if !state.chunker.is_resident() || state.chunker.is_filled() {
            return Ok(());
        }
        let n_pairs = self.shell_pairs.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
        })?.len();
        state
            .chunker
            .fill_resident(sites, |integrals, chunk| self.compute_electric_field_integrals_chunk(integrals.get_data_mut(), chunk, Some(0..n_pairs)))
            .map_err(integral_error)
    }

    pub fn compute_electric_field_integrals(
        &self,
        selection: SiteSelection<'_>,
        site_range: Range<usize>,
        output: &mut [f64],
    ) -> Result<(), AedelstenError>
    where
    {
        let n = self.get_n_cont();
        let total = self.count_electric_field_sites(selection)?;
        if site_range.start > site_range.end || site_range.end > total {
            return Err(AedelstenError::StringError(format!(
                "site range {}..{} is outside the selection of {total} sites",
                site_range.start, site_range.end
            )));
        }
        super::check_array3d_dimensions(n, 3 * site_range.len(), output)?;
        let mut integrals = IntegralsForRange { ctx: self, range: site_range, output, n_ao_squared: n * n, offset: 0 };
        self.visit_site_segments(selection, &mut integrals)
    }

    pub fn compute_static_field(&self, selection: SiteSelection<'_>, sources: StaticFieldSources, field: &mut [f64]) -> Result<(), AedelstenError> {
        let total = self.count_electric_field_sites(selection)?;
        if field.len() != 3 * total {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * total, actual: field.len() });
        }
        self.static_field_of_sites(selection, sources, field, 0..total)?;
        check_static_field(field)
    }

    /// `compute_static_field` at the polarizable sites, every rank the sources' field at its run of the sites, the
    /// runs summed: collective, every rank calls it alike and gets the whole field.
    fn static_field_over_ranks(&self, field: &mut [f64]) -> Result<(), AedelstenError> {
        let total = self.count_electric_field_sites(SiteSelection::Polarizable)?;
        if field.len() != 3 * total {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * total, actual: field.len() });
        }
        let (rank, size) = (self.comm.rank(), self.comm.size());
        let _span = instrumentation::span("polarization::static_field");
        self.static_field_of_sites(SiteSelection::Polarizable, StaticFieldSources::ALL, field, rank * total / size..(rank + 1) * total / size)?;
        sum_over_ranks(&self.comm, field)?;
        check_static_field(field)
    }

    // The static field at the selection's sites `sites` (indices into the whole selection) into `field`, zero at
    // the others.
    fn static_field_of_sites(&self, selection: SiteSelection<'_>, sources: StaticFieldSources, field: &mut [f64], sites: std::ops::Range<usize>) -> Result<(), AedelstenError> {
        field.fill(0.0);
        let mut static_field = StaticField { ctx: self, sources, field, offset: 0, sites };
        self.visit_site_segments(selection, &mut static_field)
    }


    pub fn compute_electronic_field(
        &self,
        density: &[f64],
        selection: SiteSelection<'_>,
        chunking: &efield::SiteChunking,
        field: &mut [f64],
    ) -> Result<(), AedelstenError>
    where
    {
        let n = self.get_n_cont();
        super::check_matrix_dimensions(n, density)?;
        let total = self.count_electric_field_sites(selection)?;
        if field.len() != 3 * total {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * total, actual: field.len() });
        }
        let mut electronic_field = ElectronicField { ctx: self, density: MatrixView::new(n, n, density), chunking, field, offset: 0 };
        self.visit_site_segments(selection, &mut electronic_field)
    }

    pub fn has_polarization(&self) -> bool {
        self.polarization.is_ready()
    }

    /// Whether the polarization state holds a static field that a change of the sources would make stale.
    pub(crate) fn polarization_caches_static_field(&self) -> bool {
        self.polarization.get_cache().map(|state| state.borrow().caches_static_field()).unwrap_or(false)
    }

    /// After the QM centres moved (`set_centre_positions`): recompute the cached static field (it holds the field of
    /// the QM nuclei and core AIMPs at the sites), in place. The induced dipoles are kept as the warm start of the next
    /// solve. A no-op without polarization or with `StaticFieldPolicy::Recompute`.
    pub(crate) fn refresh_cached_static_field(&mut self) -> Result<(), AedelstenError> {
        let cached = match self.polarization.get_cache() {
            Ok(state) => state.borrow_mut().static_field.take(),
            Err(_) => return Ok(()),
        };
        if let Some(mut field) = cached {
            let result = self.static_field_over_ranks(field.get_data_mut());
            if let Ok(state) = self.polarization.get_cache() {
                state.borrow_mut().static_field = Some(field);
            }
            result?;
        }
        Ok(())
    }

    pub fn set_polarization(&mut self, options: PolarizationOptions) -> Result<(), AedelstenError> {
        if self.polarization.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived(
                "Polarization is already set up. Call free_polarization() before setting it up again.".to_string(),
            ));
        }
        let n_ao = self.get_n_cont();
        let (parameters, n_sites, dipole_sources) = {
            let sites = self.polarizable_sites.get_cache().map_err(|_| {
                AedelstenError::CacheNotReady("polarizable sites are not resident in memory; call set_polarizable_sites() first".to_string())
            })?;
            let dipole_sources: Vec<sources::PointDipole> = sites.iter().map(|site| sources::PointDipole::new(*site.get_position(), [0.0; 3])).collect();
            (SystemParameters::new(sites, options.damping)?, sites.len(), dipole_sources)
        };
        // This rank's share of the sites, every rank a contiguous run, and the packed matrices a call contracts at
        // once.
        let (rank, size_comm) = (self.comm.rank(), self.comm.size());
        let share = rank * n_sites / size_comm..(rank + 1) * n_sites / size_comm;
        let batch = options.response_batch.max(options.state_slots);
        let chunk_sites = efield::resolve_field_chunk_size(&options.chunking, n_ao, n_sites).map_err(|e| AedelstenError::StringError(e.to_string()))?;
        let memory_bytes = options.resident_field_budget.map_or(0, |(size, unit)| size.saturating_mul(unit.bytes_per_unit()));
        let chunker = efield::FieldChunker::with_kept(n_ao, n_sites, share.clone(), batch, memory_bytes).map_err(chunker_error)?;
        let kept_by_rank = self.kept_by_rank(&chunker)?;
        let static_field = match options.static_field {
            StaticFieldPolicy::Cache => {
                let mut cached = Matrix::<f64>::new(3, n_sites);
                self.static_field_over_ranks(cached.get_data_mut())?;
                Some(cached)
            }
            StaticFieldPolicy::Recompute => None,
        };
        let state = PolarizationState {
            parameters,
            solver: options.solver,
            static_policy: options.static_field,
            static_field,
            electronic_field: Matrix::<f64>::new(3, n_sites),
            total_field: Matrix::<f64>::new(3, n_sites),
            dipoles: Matrix::<f64>::new(3, n_sites),
            have_dipoles: false,
            start_given: false,
            buffers: IterativeBuffers::new(n_sites),
            chunker,
            response_field: Matrix::<f64>::new(3 * n_sites, options.response_batch),
            response_dipoles: Matrix::<f64>::new(3 * n_sites, options.response_batch),
            response_batch: options.response_batch,
            state_field: Matrix::<f64>::new(3 * n_sites, options.state_slots),
            state_dipoles: Matrix::<f64>::new(3 * n_sites, options.state_slots),
            state_have_dipoles: vec![false; options.state_slots],
            field_pieces: Vec::new(),
            field_pieces_pairs: None,
            dipole_sources,
            kept_by_rank,
            share,
            batch,
            chunk_sites,
        };
        self.polarization = crate::context::cache::AedelstenCacheState::Ready(std::cell::RefCell::new(state));
        Ok(())
    }

    pub fn free_polarization(&mut self) {
        self.polarization.debug_double_free_warning("polarization");
        self.polarization.free_cache();
    }

    pub fn get_induced_dipoles(&self, output: &mut [f64]) -> Result<(), AedelstenError> {
        let state = self.polarization_state()?.borrow();
        if !state.have_dipoles {
            return Err(AedelstenError::CacheNotReady("no induced dipoles yet; call compute_induction() first".to_string()));
        }
        if output.len() != state.dipoles.get_data().len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: state.dipoles.get_data().len(), actual: output.len() });
        }
        output.copy_from_slice(state.dipoles.get_data());
        Ok(())
    }

    pub fn set_induced_dipoles(&mut self, dipoles: &[f64]) -> Result<(), AedelstenError> {
        let mut state = self.polarization_state()?.borrow_mut();
        if dipoles.len() != state.dipoles.get_data().len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: state.dipoles.get_data().len(), actual: dipoles.len() });
        }
        if dipoles.iter().any(|d| !d.is_finite()) {
            return Err(AedelstenError::StringError("set_induced_dipoles: the dipoles must be finite".to_string()));
        }
        state.dipoles.get_data_mut().copy_from_slice(dipoles);
        state.have_dipoles = false;
        state.start_given = true;
        Ok(())
    }

    // One induced-dipole solve, the coupling split over the context's ranks (every rank calls it alike).
    fn solve_dipoles<G, M>(
        &self,
        system: &PolarizableSystem<'_>,
        field: &G,
        mu: &mut M,
        guess: InitialGuess,
        solver: &SolverOptions,
        buffers: &mut IterativeBuffers,
    ) -> Result<SolveReport, AedelstenError>
    where
        G: arrays::matrix::MatrixTrait<f64>,
        M: arrays::matrix::MatrixTraitMut<f64>,
    {
        let _span = instrumentation::span("polarization::solve");
        Ok(system.solve_iterative(field, mu, guess, solver, &mut buffers.scratch(), &RankShares(&self.comm))?)
    }

    fn polarization_state(&self) -> Result<&std::cell::RefCell<PolarizationState>, AedelstenError> {
        self.polarization
            .get_cache()
            .map_err(|_| AedelstenError::CacheNotReady("polarization is not set up; call set_polarization() first".to_string()))
    }

    pub fn compute_induction(&self, density: &[f64], v_ind: &mut [f64]) -> Result<InductionReport, AedelstenError> {
        self.compute_induction_to(density, v_ind, None)
    }

    /// `compute_induction` with the dipoles solved to `threshold` in place of the setup's.
    pub fn compute_induction_within(&self, density: &[f64], v_ind: &mut [f64], threshold: f64) -> Result<InductionReport, AedelstenError> {
        if !(threshold.is_finite() && threshold > 0.0) {
            return Err(AedelstenError::StringError(format!("compute_induction_within: the threshold must be positive and finite, not {threshold}")));
        }
        self.compute_induction_to(density, v_ind, Some(threshold))
    }

    /// How this rank holds the field integrals of the polarizable sites.
    pub fn get_polarization_storage(&self) -> Result<PolarizationStorage, AedelstenError> {
        let state = self.polarization_state()?.borrow();
        let n_ao = self.get_n_cont();
        Ok(PolarizationStorage {
            sites: state.share.len(),
            sites_in_memory: state.chunker.kept().len(),
            bytes_per_site: efield::FieldChunker::bytes_per_site(n_ao),
            bytes_to_keep: efield::FieldChunker::resident_bytes(n_ao, state.share.len(), state.batch),
        })
    }

    /// See `AedelstenContext::set_resident_field_budget`.
    pub fn set_resident_field_budget(&mut self, budget: Option<(usize, arrays::scratch::ByteUnit)>) -> Result<(), AedelstenError> {
        let n_ao = self.get_n_cont();
        let n_sites = self.polarizable_sites.get_cache().map(|sites| sites.len()).unwrap_or(0);
        let memory_bytes = budget.map_or(0, |(size, unit)| size.saturating_mul(unit.bytes_per_unit()));
        let failed = {
            let mut state = self.polarization_state()?.borrow_mut();
            // The old integrals go before the new ones take their room; if the new ones cannot be allocated, this rank
            // evaluates its whole share.
            state.chunker = efield::FieldChunker::new(n_ao);
            state.field_pieces_pairs = None;
            let (chunker, failed) = match efield::FieldChunker::with_kept(n_ao, n_sites, state.share.clone(), state.batch, memory_bytes) {
                Ok(chunker) => (chunker, None),
                Err(e) => (efield::FieldChunker::new(n_ao), Some(chunker_error(e))),
            };
            state.chunker = chunker;
            failed
        };
        // Every rank takes part in the count, also one that failed.
        let kept = self.kept_by_rank(&self.polarization_state()?.borrow().chunker)?;
        self.polarization_state()?.borrow_mut().kept_by_rank = kept;
        failed.map_or(Ok(()), Err)
    }

    // The number of sites each rank keeps (`chunker`'s here), on every rank. Collective.
    fn kept_by_rank(&self, chunker: &efield::FieldChunker) -> Result<Vec<usize>, AedelstenError> {
        let mut counts = vec![0.0; self.comm.size()];
        counts[self.comm.rank()] = chunker.kept().len() as f64;
        sum_over_ranks(&self.comm, &mut counts)?;
        Ok(counts.into_iter().map(|c| c as usize).collect())
    }

    // The induction, its dipoles solved to `threshold`, the setup's when `None`.
    fn compute_induction_to(&self, density: &[f64], v_ind: &mut [f64], threshold: Option<f64>) -> Result<InductionReport, AedelstenError> {
        let n = self.get_n_cont();
        super::check_matrix_dimensions(n, density)?;
        super::check_matrix_dimensions(n, v_ind)?;
        let mut guard = self.polarization_state()?.borrow_mut();
        let state = &mut *guard;
        self.ensure_field_pieces(state)?;
        let sites = self.polarizable_sites.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("polarizable sites are not resident in memory; call set_polarizable_sites() first".to_string())
        })?;
        self.fill_resident_field(state, sites)?;
        v_ind.fill(0.0);
        if sites.is_empty() {
            return Ok(InductionReport { energy: 0.0, iterations: 0, residual: 0.0 });
        }

        // Pass 1: the electronic field of the density at the sites, the kept ones from their integrals, the others
        // over the pieces handed out.
        let column = 3 * sites.len();
        state.electronic_field.get_data_mut().fill(0.0);
        kept_fields(&mut state.chunker, density, state.electronic_field.get_data_mut(), column);
        let handout = oneint::pieces::Handout::new(&state.field_pieces, &self.comm);
        self.add_piece_fields(handout.iter(), density, sites, state.electronic_field.get_data_mut())?;
        sum_over_ranks(&self.comm, state.electronic_field.get_data_mut())?;

        // The polarizing field: static (kept, or computed now) plus electronic.
        match &state.static_field {
            Some(cached) => state.total_field.get_data_mut().copy_from_slice(cached.get_data()),
            None => self.static_field_over_ranks(state.total_field.get_data_mut())?,
        }
        add_scaled_in_place(state.total_field.get_data_mut(), state.electronic_field.get_data(), 1.0);

        // The induced dipoles, warm-started from the previous call.
        let system = PolarizableSystem::with_parameters(sites, &state.parameters)?;
        let guess = if state.have_dipoles || state.start_given { InitialGuess::Provided } else { InitialGuess::Static };
        let solver = SolverOptions { threshold: threshold.unwrap_or(state.solver.threshold), ..state.solver };
        let report = self.solve_dipoles(&system, &state.total_field, &mut state.dipoles, guess, &solver, &mut state.buffers)?;
        state.have_dipoles = true;
        state.start_given = false;
        let energy = report.functional;

        // Pass 2: the operator, V = -Sum mu . t, likewise.
        kept_operators(&mut state.chunker, state.dipoles.get_data(), column, -1.0, v_ind);
        let handout = oneint::pieces::Handout::new(&state.field_pieces, &self.comm);
        self.add_piece_operators(handout.iter(), state.dipoles.get_data(), -1.0, &mut state.dipole_sources, v_ind)?;
        mirror_operators(n, v_ind);
        sum_over_ranks(&self.comm, v_ind)?;
        Ok(InductionReport { energy, iterations: report.iterations, residual: report.residual })
    }

    pub fn compute_induction_response(&self, densities: &[f64], n_mat: usize, operators: &mut [f64]) -> Result<(), AedelstenError>
    where
    {
        let n = self.get_n_cont();
        let n2 = n * n;
        for (name, len) in [("densities", densities.len()), ("operators", operators.len())] {
            if len != n2 * n_mat {
                return Err(AedelstenError::StringError(format!("{name} has length {len}, expected n_cont^2 * n_mat = {}", n2 * n_mat)));
            }
        }
        let mut guard = self.polarization_state()?.borrow_mut();
        let state = &mut *guard;
        self.ensure_field_pieces(state)?;
        if n_mat > 0 && state.response_batch == 0 {
            return Err(AedelstenError::CacheNotReady(
                "response is not enabled; set PolarizationOptions::response_batch >= 1 in set_polarization()".to_string(),
            ));
        }
        let sites = self.polarizable_sites.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("polarizable sites are not resident in memory; call set_polarizable_sites() first".to_string())
        })?;
        self.fill_resident_field(state, sites)?;
        operators.fill(0.0);
        if sites.is_empty() || n_mat == 0 {
            return Ok(());
        }
        let block = 3 * sites.len();
        let system = PolarizableSystem::with_parameters(sites, &state.parameters)?;
        for (dens, ops) in densities.chunks(n2 * state.response_batch).zip(operators.chunks_mut(n2 * state.response_batch)) {
            let k = dens.len() / n2;
            let fields = &mut state.response_field.get_data_mut()[..block * k];
            fields.fill(0.0);
            kept_fields(&mut state.chunker, dens, fields, block);
            let handout = oneint::pieces::Handout::new(&state.field_pieces, &self.comm);
            self.add_piece_fields(handout.iter(), dens, sites, fields)?;
            sum_over_ranks(&self.comm, fields)?;
            for m in 0..k {
                let field = MatrixView::new(3, sites.len(), &state.response_field.get_data()[m * block..(m + 1) * block]);
                let mut mu = MatrixViewMut::new(3, sites.len(), &mut state.response_dipoles.get_data_mut()[m * block..(m + 1) * block]);
                self.solve_dipoles(&system, &field, &mut mu, InitialGuess::Static, &state.solver, &mut state.buffers)?;
            }
            let mu_all = &state.response_dipoles.get_data()[..block * k];
            kept_operators(&mut state.chunker, mu_all, block, -1.0, ops);
            let handout = oneint::pieces::Handout::new(&state.field_pieces, &self.comm);
            self.add_piece_operators(handout.iter(), mu_all, -1.0, &mut state.dipole_sources, ops)?;
            mirror_operators(n, ops);
            sum_over_ranks(&self.comm, ops)?;
        }
        Ok(())
    }

    /// See `AedelstenContext::compute_induction_couplings`.
    pub fn compute_induction_couplings(&self, densities: &[f64], n_mat: usize, couplings: &mut [f64]) -> Result<(), AedelstenError> {
        let n2 = self.get_n_cont() * self.get_n_cont();
        if densities.len() != n2 * n_mat {
            return Err(AedelstenError::StringError(format!("densities has length {}, expected n_cont^2 * n_mat = {}", densities.len(), n2 * n_mat)));
        }
        if couplings.len() != n_mat * n_mat {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: n_mat * n_mat, actual: couplings.len() });
        }
        let mut guard = self.polarization_state()?.borrow_mut();
        let state = &mut *guard;
        self.ensure_field_pieces(state)?;
        if n_mat > 0 && state.response_batch == 0 {
            return Err(AedelstenError::CacheNotReady(
                "response is not enabled; set PolarizationOptions::response_batch >= 1 in set_polarization()".to_string(),
            ));
        }
        let sites = self.polarizable_sites.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("polarizable sites are not resident in memory; call set_polarizable_sites() first".to_string())
        })?;
        self.fill_resident_field(state, sites)?;
        couplings.fill(0.0);
        if sites.is_empty() || n_mat == 0 {
            return Ok(());
        }
        let block = 3 * sites.len();
        // Every density's field at the sites, the passes in batches of `response_batch` sharing the integrals.
        let mut fields = Matrix::<f64>::new(block, n_mat);
        for (dens, batch) in densities.chunks(n2 * state.response_batch).zip(fields.get_data_mut().chunks_mut(block * state.response_batch)) {
            kept_fields(&mut state.chunker, dens, batch, block);
            let handout = oneint::pieces::Handout::new(&state.field_pieces, &self.comm);
            self.add_piece_fields(handout.iter(), dens, sites, batch)?;
            sum_over_ranks(&self.comm, batch)?;
        }
        // Column n: the dipoles of field n, then K_mn = -f_m . mu_n for every m.
        let system = PolarizableSystem::with_parameters(sites, &state.parameters)?;
        for n in 0..n_mat {
            let field = MatrixView::new(3, sites.len(), fields.get_col(n));
            let mut mu = MatrixViewMut::new(3, sites.len(), &mut state.response_dipoles.get_data_mut()[..block]);
            self.solve_dipoles(&system, &field, &mut mu, InitialGuess::Static, &state.solver, &mut state.buffers)?;
            let mu = &state.response_dipoles.get_data()[..block];
            for (m, f_m) in fields.iter().enumerate() {
                couplings[m + n * n_mat] = -f_m.iter().zip(mu).map(|(f, d)| f * d).sum::<f64>();
            }
        }
        Ok(())
    }

    /// See `AedelstenContext::compute_effective_external_field`.
    pub fn compute_effective_external_field(&self, operators: &mut [f64]) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let n2 = n * n;
        if operators.len() != 3 * n2 {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * n2, actual: operators.len() });
        }
        let mut guard = self.polarization_state()?.borrow_mut();
        let state = &mut *guard;
        self.ensure_field_pieces(state)?;
        let sites = self.polarizable_sites.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("polarizable sites are not resident in memory; call set_polarizable_sites() first".to_string())
        })?;
        self.fill_resident_field(state, sites)?;
        operators.fill(0.0);
        if sites.is_empty() {
            return Ok(());
        }
        let block = 3 * sites.len();
        let system = PolarizableSystem::with_parameters(sites, &state.parameters)?;
        // Column i: the dipoles of the unit field along i, 3 per site.
        let mut dipoles = Matrix::<f64>::new(block, 3);
        let mut field = Matrix::<f64>::new(3, sites.len());
        for (i, mu_i) in dipoles.iter_mut().enumerate() {
            for site_field in field.iter_mut() {
                site_field.fill(0.0);
                site_field[i] = 1.0;
            }
            let mut mu = MatrixViewMut::new(3, sites.len(), mu_i);
            self.solve_dipoles(&system, &field, &mut mu, InitialGuess::Static, &state.solver, &mut state.buffers)?;
        }
        kept_operators(&mut state.chunker, dipoles.get_data(), block, -1.0, operators);
        let handout = oneint::pieces::Handout::new(&state.field_pieces, &self.comm);
        self.add_piece_operators(handout.iter(), dipoles.get_data(), -1.0, &mut state.dipole_sources, operators)?;
        mirror_operators(n, operators);
        sum_over_ranks(&self.comm, operators)?;
        Ok(())
    }

    pub fn compute_induction_states(&self, densities: &[f64], n_states: usize, operators: &mut [f64], energies: &mut [f64]) -> Result<(), AedelstenError> {
        self.compute_induction_states_to(densities, n_states, operators, energies, None)
    }

    /// `compute_induction_states` with the dipoles solved to `threshold` in place of the setup's.
    pub fn compute_induction_states_within(&self, densities: &[f64], n_states: usize, operators: &mut [f64], energies: &mut [f64], threshold: f64) -> Result<(), AedelstenError> {
        if !(threshold.is_finite() && threshold > 0.0) {
            return Err(AedelstenError::StringError(format!("compute_induction_states_within: the threshold must be positive and finite, not {threshold}")));
        }
        self.compute_induction_states_to(densities, n_states, operators, energies, Some(threshold))
    }

    // The state-specific induction, the dipoles solved to `threshold`, the setup's when `None`.
    fn compute_induction_states_to(&self, densities: &[f64], n_states: usize, operators: &mut [f64], energies: &mut [f64], threshold: Option<f64>) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let n2 = n * n;
        for (name, len, expected) in [("densities", densities.len(), n2 * n_states), ("operators", operators.len(), n2 * n_states), ("energies", energies.len(), n_states)] {
            if len != expected {
                return Err(AedelstenError::StringError(format!("{name} has length {len}, expected {expected} (n_states = {n_states})")));
            }
        }
        let mut guard = self.polarization_state()?.borrow_mut();
        let state = &mut *guard;
        self.ensure_field_pieces(state)?;
        if n_states > state.state_have_dipoles.len() {
            return Err(AedelstenError::CacheNotReady(format!(
                "{n_states} states requested but only {} state slots; set PolarizationOptions::state_slots in set_polarization()",
                state.state_have_dipoles.len()
            )));
        }
        let sites = self.polarizable_sites.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("polarizable sites are not resident in memory; call set_polarizable_sites() first".to_string())
        })?;
        self.fill_resident_field(state, sites)?;
        operators.fill(0.0);
        energies.fill(0.0);
        if sites.is_empty() || n_states == 0 {
            return Ok(());
        }
        let block = 3 * sites.len();
        let k = n_states;
        let fields = &mut state.state_field.get_data_mut()[..block * k];
        fields.fill(0.0);
        kept_fields(&mut state.chunker, densities, fields, block);
        let handout = oneint::pieces::Handout::new(&state.field_pieces, &self.comm);
        self.add_piece_fields(handout.iter(), densities, sites, fields)?;
        sum_over_ranks(&self.comm, fields)?;
        match &state.static_field {
            Some(cached) => state.total_field.get_data_mut().copy_from_slice(cached.get_data()),
            None => self.static_field_over_ranks(state.total_field.get_data_mut())?,
        }
        let system = PolarizableSystem::with_parameters(sites, &state.parameters)?;
        let solver = SolverOptions { threshold: threshold.unwrap_or(state.solver.threshold), ..state.solver };
        for m in 0..k {
            let range = m * block..(m + 1) * block;
            add_scaled_in_place(&mut state.state_field.get_data_mut()[range.clone()], state.total_field.get_data(), 1.0);
            let field = MatrixView::new(3, sites.len(), &state.state_field.get_data()[range.clone()]);
            let mut mu = MatrixViewMut::new(3, sites.len(), &mut state.state_dipoles.get_data_mut()[range]);
            let guess = if state.state_have_dipoles[m] { InitialGuess::Provided } else { InitialGuess::Static };
            let report = self.solve_dipoles(&system, &field, &mut mu, guess, &solver, &mut state.buffers)?;
            state.state_have_dipoles[m] = true;
            energies[m] = report.functional;
        }
        let mu_all = &state.state_dipoles.get_data()[..block * k];
        kept_operators(&mut state.chunker, mu_all, block, -1.0, operators);
        let handout = oneint::pieces::Handout::new(&state.field_pieces, &self.comm);
        self.add_piece_operators(handout.iter(), mu_all, -1.0, &mut state.dipole_sources, operators)?;
        mirror_operators(n, operators);
        sum_over_ranks(&self.comm, operators)?;
        Ok(())
    }

    /// `dE_ind/dX` at fixed `density`, for every gradient centre X, into `gradient` (column-major `(3, n_centres)`,
    /// OVERWRITTEN), using the induced dipoles the library holds -- so call `compute_induction` with the same density
    /// first (in an SCF: the converged one). At converged dipoles E_ind is stationary in them, so its derivative is
    /// `-mu . dF_QM/dX` with the dipoles held fixed: the gradient of the energy of the QM charges (the electrons in
    /// `density`, the nuclei, the core AIMPs) in the potential of fixed point dipoles `mu_s` at the sites. The
    /// electronic part is the one-electron external-multipole gradient with the dipoles as sources; the nuclear part
    /// is `add_qm_charges_in_multipoles`. Permanent multipoles, embedding AIMPs and the sites are fixed and give
    /// nothing; the density-derivative (Pulay) part comes with the overlap gradient, since `v_ind` is in the Fock
    /// matrix. `active` is the frozen-centre mask.
    pub fn compute_induction_gradient(&self, gradient: &mut [f64], density: &[f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        let dipoles: Box<[sources::PointMultipole]> = {
            let state = self
                .polarization
                .get_cache()
                .map_err(|e| AedelstenError::CacheNotReady(format!("polarization: {e} (set_polarization())")))?
                .borrow();
            if !state.have_dipoles {
                return Err(AedelstenError::CacheNotReady(
                    "no induced dipoles yet: call compute_induction with this density first".to_string(),
                ));
            }
            let sites = self.polarizable_sites.get_cache().map_err(|e| {
                AedelstenError::CacheNotReady(format!("polarizable sites: {e} (set_polarizable_sites())"))
            })?;
            sites
                .iter()
                .enumerate()
                .map(|(s, site)| {
                    use sources::SiteTrait;
                    sources::PointMultipole::dipole(*site.get_position(), 0.0, [state.dipoles[(0, s)], state.dipoles[(1, s)], state.dipoles[(2, s)]])
                })
                .collect()
        };
        let sources = basis::charge_carriers::ExternalMultipoles::Multipoles(dipoles);
        // Electrons: the one-electron operator of the fixed dipoles contracted with the density (fills `gradient`).
        self.compute_one_electron_gradient(
            gradient,
            density,
            crate::context::one_electron::OneElectronOperator::external_multipole(&sources, self.multipole_convention),
            active,
        )?;
        // Nuclei and core AIMPs in the dipoles' potential.
        let basis::charge_carriers::ExternalMultipoles::Multipoles(dipoles) = &sources else { unreachable!() };
        let n_centres = self.get_n_gradient_centres();
        let mut out = crate::context::gradients::CentreGradient::new(gradient, eri::GradientCentres::new(n_centres, active));
        self.add_qm_charges_in_multipoles(dipoles, &mut out)
    }
}

/// The points at which the electric field is evaluated.
///
/// The library-owned selections resolve, in a fixed order, to the sites of the respective
/// region(s) in the order they are stored, so a site's position in the output is its index in
/// that order.
#[derive(Clone, Copy)]
pub enum SiteSelection<'a> {
    /// All sites of the QM region: the nuclei first, then the core AIMP centres (each part
    /// absent if the input has none, so this can be empty for an input with neither).
    /// Unlike for nuclear attraction, the two are not treated separately here.
    Qm,
    /// The embedding AIMP centres.
    EmbeddingAimps,
    /// The external (xfield) points, see `set_external_multipoles`.
    ExternalMultipoles,
    /// The polarizable sites, see `set_polarizable_sites`, in input order (the site order of the
    /// induced-dipole and field vectors).
    Polarizable,
    /// Arbitrary user-provided points.
    Custom(&'a [Site]),
}

/// Which classical sources contribute to the static field (see `compute_static_field`).
/// A source that does not exist in the input (no embedding AIMPs, no external multipoles, ...)
/// simply contributes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticFieldSources {
    /// The QM region's all-electron nuclei (point or finite). A site must not coincide with one.
    pub nuclei: bool,
    /// The core AIMPs of the QM region, point charge plus screen. A site must not coincide with one.
    pub core_aimps: bool,
    /// The embedding AIMPs, point charge plus screen. An AIMP on the site itself (a polarizable AIMP
    /// centre) is skipped for that site.
    pub embedding_aimps: bool,
    /// The external multipoles of any order (`set_external_multipoles`).
    pub external_multipoles: bool,
}

impl StaticFieldSources {
    /// Every source.
    pub const ALL: Self = Self { nuclei: true, core_aimps: true, embedding_aimps: true, external_multipoles: true };
}

// A source closer than this to a site (Bohr) is taken to be that site's own, and skipped: its field
// there is singular, and a site does not polarise itself. Generous on purpose: the site and the
// source come from different files (the polarizable-site file and the AIMP/multipole data), whose
// coordinates may be written in different units and rounded differently, while genuinely distinct
// sites are never within a milliBohr of each other. Used for the external multipoles (a site's
// own permanent multipole, on top of the fragment exclusions) and the embedding AIMPs (a polarizable
// AIMP centre). NOT applied to QM nuclei or core AIMPs: a site on one of those is an input error,
// reported by `compute_static_field` rather than silently skipped.
const COINCIDENCE_TOLERANCE: f64 = 1e-3;

fn coincident(a: &[f64; 3], b: &[f64; 3]) -> bool {
    (a[0] - b[0]).abs() < COINCIDENCE_TOLERANCE && (a[1] - b[1]).abs() < COINCIDENCE_TOLERANCE && (a[2] - b[2]).abs() < COINCIDENCE_TOLERANCE
}

// One operation to run on each segment of a selection. A trait with a generic method (rather than
// a closure) because each segment has its own concrete site type.
trait SegmentVisitor {
    fn visit<Site: SiteTrait>(&mut self, segment: &[Site]) -> Result<(), AedelstenError>;
}

struct CountSites(usize);

impl SegmentVisitor for CountSites {
    fn visit<Site: SiteTrait>(&mut self, segment: &[Site]) -> Result<(), AedelstenError> {
        self.0 += segment.len();
        Ok(())
    }
}

struct InputIndices(Vec<Option<usize>>);

impl SegmentVisitor for InputIndices {
    fn visit<Site: SiteTrait>(&mut self, segment: &[Site]) -> Result<(), AedelstenError> {
        self.0.extend(segment.iter().map(|site| site.get_input_index()));
        Ok(())
    }
}

// Electric-field integrals for the global site range `range`, written to `output` (3 planes per
// site of the range, in selection order).
struct IntegralsForRange<'o, 'c, S: basis::contraction::ShellKind> {
    ctx: &'c AedelstenContextImpl<S>,
    range: Range<usize>,
    output: &'o mut [f64],
    n_ao_squared: usize,
    // Global index of the first site of the segment about to be visited.
    offset: usize,
}

impl<S: basis::contraction::ShellKind> SegmentVisitor for IntegralsForRange<'_, '_, S>
where
{
    fn visit<Site: SiteTrait>(&mut self, segment: &[Site]) -> Result<(), AedelstenError> {
        let start = self.range.start.max(self.offset);
        let end = self.range.end.min(self.offset + segment.len());
        if start < end {
            let planes = 3 * self.n_ao_squared;
            let first = (start - self.range.start) * planes;
            let last = (end - self.range.start) * planes;
            self.ctx.compute_electric_field_integrals_chunk(&mut self.output[first..last], &segment[start - self.offset..end - self.offset], None)?;
        }
        self.offset += segment.len();
        Ok(())
    }
}

// Electronic field for every site of the selection, chunked within each segment.
struct ElectronicField<'d, 'c, S: basis::contraction::ShellKind> {
    ctx: &'c AedelstenContextImpl<S>,
    density: MatrixView<'d, f64>,
    chunking: &'d efield::SiteChunking,
    field: &'d mut [f64],
    offset: usize,
}

impl<S: basis::contraction::ShellKind> SegmentVisitor for ElectronicField<'_, '_, S>
where
{
    fn visit<Site: SiteTrait>(&mut self, segment: &[Site]) -> Result<(), AedelstenError> {
        let field = &mut self.field[3 * self.offset..3 * (self.offset + segment.len())];
        let ctx = self.ctx;
        let piece_sites = efield::resolve_field_chunk_size(self.chunking, self.density.rows(), segment.len()).map_err(|e| AedelstenError::StringError(e.to_string()))?;
        let pieces = ctx.field_pieces(&[0..segment.len()], piece_sites)?;
        let handout = oneint::pieces::Handout::new(&pieces, &ctx.comm);
        field.fill(0.0);
        ctx.add_piece_fields(handout.iter(), self.density.get_data(), segment, field)?;
        sum_over_ranks(&ctx.comm, field)?;
        self.offset += segment.len();
        Ok(())
    }
}

// Static (classical) field at every site of the selection, from the sources enabled in `sources`.
struct StaticField<'c, 'f, S: basis::contraction::ShellKind> {
    ctx: &'c AedelstenContextImpl<S>,
    sources: StaticFieldSources,
    field: &'f mut [f64],
    offset: usize,
    // The sites (indices into the whole selection) to compute the field at.
    sites: std::ops::Range<usize>,
}

impl<S: basis::contraction::ShellKind> SegmentVisitor for StaticField<'_, '_, S>
where
{
    fn visit<Site: SiteTrait>(&mut self, whole_segment: &[Site]) -> Result<(), AedelstenError> {
        let qm_region = self.ctx.qm_region.get_cache().expect("input has been freed (free_input)");
        // The segment's part within `sites`.
        let (start, end) = (self.offset, self.offset + whole_segment.len());
        let (first, last) = (self.sites.start.clamp(start, end), self.sites.end.clamp(start, end));
        self.offset = end;
        if first >= last {
            return Ok(());
        }
        let segment = &whole_segment[first - start..last - start];
        let field = &mut self.field[3 * first..3 * last];
        // A closure per source list: excludes the pairs where the source sits on the site itself.
        // Only for sources a site may legitimately share a position with (see COINCIDENCE_TOLERANCE).
        macro_rules! exclusion {
            ($sources:expr) => {
                |site: usize, source: usize| coincident(segment[site].get_position(), $sources[source].get_position())
            };
        }
        if self.sources.nuclei {
            match qm_region.input.get_nuclear_potential() {
                Some(NuclearPotential::PointCharge(charges)) => sources::field::add_field_from_charges(segment, charges, sources::field::no_exclusions, field),
                Some(NuclearPotential::GaussianCharge(charges)) => sources::field::add_field_from_finite_charges(segment, charges, sources::field::no_exclusions, field),
                None => {}
            }
        }
        if self.sources.core_aimps {
            if let Some(aimps) = qm_region.input.get_core_regions() {
                sources::field::add_field_from_aimps(segment, aimps, sources::field::no_exclusions, field);
            }
        }
        if self.sources.embedding_aimps {
            if let Some(aimps) = qm_region.input.get_embedding_potentials() {
                sources::field::add_field_from_aimps(segment, aimps, exclusion!(aimps), field);
            }
        }
        if self.sources.external_multipoles {
            if let Ok(multipoles) = self.ctx.external_multipoles.get_cache() {
                // Only the external multipoles carry fragments (AIMPs and nuclei are their own
                // fragments): a source is also excluded from the sites of its fragments.
                macro_rules! fragment_exclusion {
                    ($sources:expr) => {
                        |site: usize, source: usize| {
                            coincident(segment[site].get_position(), $sources[source].get_position())
                                || fragment_excludes(&segment[site], &$sources[source])
                        }
                    };
                }
                match multipoles {
                    ExternalMultipoles::Charges(charges) => sources::field::add_field_from_charges(segment, charges, fragment_exclusion!(charges), field),
                    ExternalMultipoles::Multipoles(points) => sources::field::add_field_from_multipoles(segment, points, fragment_exclusion!(points), field),
                }
            }
        }
        Ok(())
    }
}

// A site on a QM nucleus or core AIMP (never skipped) makes the static field singular; say so instead of returning
// NaN.
fn check_static_field(field: &[f64]) -> Result<(), AedelstenError> {
    if let Some(bad) = field.iter().position(|x| !x.is_finite()) {
        return Err(AedelstenError::StringError(format!(
            "the static field is singular at site {}: it coincides with a QM nucleus or core AIMP (sites may only share a position with external multipoles or embedding AIMPs)",
            bad / 3
        )));
    }
    Ok(())
}

/// Whether the static field at the polarizable sites (nuclei, AIMPs and external multipoles, constant for a
/// geometry) is kept in memory or recomputed at every `compute_induction`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticFieldPolicy {
    /// Computed once by `set_polarization` and kept: 3 doubles per site of memory, but the sources may not
    /// change while the polarization is set (see `set_external_multipoles`).
    Cache,
    /// Computed again at every call: no memory beyond the scratch that is needed anyway, and the sources
    /// may change between calls, at the cost of an O(sites x sources) evaluation each time.
    Recompute,
}

/// The setup of the induced-dipole polarization (`AedelstenContext::set_polarization`).
pub struct PolarizationOptions {
    pub damping: Damping,
    pub solver: SolverOptions,
    pub static_field: StaticFieldPolicy,
    /// The sites a field pass over the sites not kept hands out as one piece over the ranks, with a run of shell
    /// pairs.
    pub chunking: efield::SiteChunking,
    /// The most densities `compute_induction_response` treats in one go (their fields from one pass over the shell
    /// pairs, at 6 doubles per site per density of memory). 0: no response calls.
    pub response_batch: usize,
    /// The number of states `compute_induction_states` can hold: each keeps its own induced dipoles between
    /// calls (6 doubles per site per state of memory). 0: no state calls.
    pub state_slots: usize,
    /// Keep the field integrals between calls, computed once per geometry (on each rank, those of its share of the
    /// sites: `3 n_share n_cont (n_cont + 1) / 2` doubles, and `n_cont (n_cont + 1) / 2` per response or state
    /// density), when they fit this budget; a call then costs a product with them instead of their evaluation.
    /// `None` (or a budget they exceed): the sites not kept are evaluated in every call, each density contracted in
    /// the Hermite basis of the shell pairs and the dipoles' operator as an external-multipole one, with no integral
    /// formed per site.
    pub resident_field_budget: Option<(usize, arrays::scratch::ByteUnit)>,
}

impl Default for PolarizationOptions {
    fn default() -> Self {
        Self {
            damping: Damping::None,
            solver: SolverOptions::default(),
            static_field: StaticFieldPolicy::Cache,
            chunking: efield::SiteChunking::Budget(None),
            response_batch: 0,
            state_slots: 0,
            resident_field_budget: None,
        }
    }
}

/// How this rank holds the field integrals of its share of the polarizable sites
/// (`AedelstenContext::get_polarization_storage`): the first `sites_in_memory` kept in memory, the rest evaluated in
/// every field pass (handed out over the ranks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolarizationStorage {
    /// The sites of this rank's share.
    pub sites: usize,
    pub sites_in_memory: usize,
    /// The bytes of one site's kept integrals (each plane's lower triangle).
    pub bytes_per_site: usize,
    /// The bytes this rank needs to keep its whole share in memory, with the scratch of the packed matrices a call
    /// contracts at once: a `PolarizationOptions::resident_field_budget` of at least this keeps them all.
    pub bytes_to_keep: usize,
}

/// What `compute_induction` returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InductionReport {
    /// E_ind = -1/2 mu . F, to add to the total energy.
    pub energy: f64,
    /// Iterations the dipole solver needed (1 if the previous dipoles already solved the equations).
    pub iterations: usize,
    /// The norm of the last change of the dipoles.
    pub residual: f64,
}

/// The polarization's state between calls: the sites' parameters, the solver's options, the dipoles and fields, the
/// response and state buffers, the field passes' pieces and the kept integrals.
#[derive(Debug)]
pub(crate) struct PolarizationState {
    parameters: SystemParameters,
    solver: SolverOptions,
    static_policy: StaticFieldPolicy,
    // Some exactly for StaticFieldPolicy::Cache.
    static_field: Option<Matrix<f64>>,
    electronic_field: Matrix<f64>,
    total_field: Matrix<f64>,
    dipoles: Matrix<f64>,
    // Whether `dipoles` holds a solution to warm-start from.
    have_dipoles: bool,
    // Whether `dipoles` holds a start given by `set_induced_dipoles` (not a solution).
    start_given: bool,
    buffers: IterativeBuffers,
    chunker: efield::FieldChunker,
    // 3S x response_batch each; empty when response is not enabled.
    response_field: Matrix<f64>,
    response_dipoles: Matrix<f64>,
    response_batch: usize,
    // One column per state slot: the polarizing field and the persistent dipoles of each state.
    state_field: Matrix<f64>,
    state_dipoles: Matrix<f64>,
    state_have_dipoles: Vec<bool>,
    // The field passes' pieces (runs of shell pairs against runs of sites, `oneint::pieces`), in hand-out order, for
    // `field_pieces_pairs` shell pairs; rebuilt when the shell pairs change in number.
    field_pieces: Vec<oneint::pieces::Piece>,
    field_pieces_pairs: Option<usize>,
    // A dipole at each site, the sources of the operator passes over the pieces (their dipoles set by each pass):
    // 6 doubles a site.
    dipole_sources: Vec<sources::PointDipole>,
    // The number of sites each rank keeps (the first of its share): the rest are the field passes' pieces.
    kept_by_rank: Vec<usize>,
    // This rank's share of the sites, the packed matrices a call contracts at once, and the sites a piece holds.
    share: Range<usize>,
    batch: usize,
    chunk_sites: usize,
}

impl PolarizationState {
    /// Whether the static field is kept (`StaticFieldPolicy::Cache`).
    pub(crate) fn caches_static_field(&self) -> bool {
        self.static_policy == StaticFieldPolicy::Cache
    }

    // The matrices' data: a lower bound, the parameters' and the chunker's own memory not counted.
    /// The kept field integrals no longer hold (new shell pairs): the next pass computes them.
    pub(crate) fn invalidate_field_integrals(&mut self) {
        self.chunker.invalidate();
    }

    pub(crate) fn approx_memory_bytes(&self) -> usize {
        let f64_bytes = std::mem::size_of::<f64>();
        let matrices = [
            &self.electronic_field,
            &self.total_field,
            &self.dipoles,
            &self.response_field,
            &self.response_dipoles,
            &self.state_field,
            &self.state_dipoles,
        ];
        let mut bytes: usize = matrices.iter().map(|m| m.get_data().len() * f64_bytes).sum();
        if let Some(static_field) = &self.static_field {
            bytes += static_field.get_data().len() * f64_bytes;
        }
        // The kept field integrals with their buffers, and the dipole sources.
        bytes + self.chunker.bytes() + std::mem::size_of_val(self.dipole_sources.as_slice())
    }
}

// `FieldChunker::add_kept_fields`, timed.
fn kept_fields(chunker: &mut efield::FieldChunker, densities: &[f64], fields: &mut [f64], column: usize) {
    let _span = instrumentation::span("polarization::kept_fields");
    chunker.add_kept_fields(densities, fields, column);
}

// `FieldChunker::add_kept_operators`, timed.
fn kept_operators(chunker: &mut efield::FieldChunker, dipoles: &[f64], column: usize, scale: f64, operators: &mut [f64]) {
    let _span = instrumentation::span("polarization::kept_operators");
    chunker.add_kept_operators(dipoles, column, scale, operators);
}

// Each `n x n` operator of `operators` completed from its upper triangle, which the operator passes over the pieces add
// to (the kept passes add whole symmetric matrices, which this leaves as they are).
fn mirror_operators(n: usize, operators: &mut [f64]) {
    for operator in operators.chunks_exact_mut(n * n) {
        MatrixViewMut::new(n, n, operator).copy_upper_to_lower();
    }
}

fn chunker_error(error: efield::EfieldError<std::convert::Infallible>) -> AedelstenError {
    AedelstenError::StringError(error.to_string())
}

fn integral_error(error: efield::EfieldError<AedelstenError>) -> AedelstenError {
    match error {
        efield::EfieldError::Integrals(inner) => inner,
        other => AedelstenError::StringError(other.to_string()),
    }
}

/// Sums `values`, a field pass's results over this rank's pieces, over the ranks of `comm`. Collective; nothing on
/// one rank.
// The ranks of the context's communicator as the shares of the induced-dipole coupling: each rank its run of rows
// of the pair triangle, the fields summed, the solver's scalars rank 0's.
struct RankShares<'c>(&'c aedelsten_mpi::Communicator);

impl Collective for RankShares<'_> {
    fn share(&self) -> (usize, usize) {
        (self.0.rank(), self.0.size())
    }
    fn sum(&self, values: &mut [f64]) -> Result<(), PolarizationError> {
        if self.0.size() > 1 {
            self.0.sum(values).map_err(|e| PolarizationError::Collective(format!("summing the dipoles' field: {e}")))?;
        }
        Ok(())
    }
    fn broadcast(&self, values: &mut [f64]) -> Result<(), PolarizationError> {
        if self.0.size() > 1 {
            self.0.broadcast(values, 0).map_err(|e| PolarizationError::Collective(format!("broadcasting the solver's scalars: {e}")))?;
        }
        Ok(())
    }
}

fn sum_over_ranks(comm: &aedelsten_mpi::Communicator, values: &mut [f64]) -> Result<(), AedelstenError> {
    if comm.size() > 1 {
        comm.sum(values).map_err(|e| AedelstenError::StringError(format!("summing a field pass over the ranks: {e}")))?;
    }
    Ok(())
}
