// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Geometric gradients: the two-electron gradient (contracted with densities; and, for the crate's tests, the
// derivative tensors themselves) and the classical (nucleus-level) energies' gradients. The one-electron
// gradients are with their operators in one_electron.rs.
//
// Geometric gradients of the classical (nucleus-level) energies: one call per `compute_nuclear_*_energy`, each
// following its energy branch for branch (the pair expressions are differentiated in `sources::energy_gradient`).
// Every call fills a column-major `(3, get_n_gradient_centres())` matrix, overwritten, with the frozen-centre
// mask of the other gradient calls. What moves: the QM nuclei (point or Gaussian) and the core AIMPs, each with
// its centre. What does not: embedding AIMPs, external multipoles, orthogonality ghosts.

use super::*;
use basis::charge_carriers::{ExternalMultipoles, NuclearPotential};

impl AedelstenContext {
    /// Geometric gradients of the three `compute_nuclear_*_energy` above, one call each (the consumer sums what it
    /// uses): `gradient` is a column-major `(3, get_n_gradient_centres())` matrix, OVERWRITTEN; `active` is the
    /// frozen-centre mask of the other gradient calls. The QM nuclei and core AIMPs move with their centres; embedding
    /// AIMPs and external multipoles are fixed. See `context/gradients.rs`.
    pub fn compute_nuclear_repulsion_gradient(&self, gradient: &mut [f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        dispatch!(self, compute_nuclear_repulsion_gradient(gradient, active))
    }

    /// The gradient of `compute_nuclear_aimp_embedding_energy`: the QM nuclei and core AIMPs in the fixed embedding
    /// AIMPs' screened potentials. Zero without embedding AIMPs (as the energy).
    pub fn compute_nuclear_aimp_embedding_gradient(&self, gradient: &mut [f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        dispatch!(self, compute_nuclear_aimp_embedding_gradient(gradient, active))
    }

    /// An error without external multipoles (`set_external_multipoles`).
    pub fn compute_nuclear_multipole_interaction_gradient(&self, gradient: &mut [f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        dispatch!(self, compute_nuclear_multipole_interaction_gradient(gradient, active))
    }

    /// The number of centres the gradient matrices have columns for: real atoms and pseudocharge ghosts;
    /// orthogonality-function ghosts (last in the centre list, not independent degrees of freedom) are outside.
    pub fn get_n_gradient_centres(&self) -> usize { dispatch!(self, get_n_gradient_centres()) }

    /// The two-electron integral derivatives, fed to a `GradientSink` batch by batch, from the resident ERI data
    /// (`set_eri_data`). Every contraction type (general, segmented, primitive).
    ///
    /// `active`: the frozen-centre mask, one flag per gradient centre (`get_n_gradient_centres()`),
    /// `None` = every centre. Frozen centres are never sent to the sink, and work that only feeds them
    /// is skipped. The mask is per centre only: to freeze single coordinates, keep their centre active
    /// and zero those entries afterwards (the work is shared by x, y, z, so a per-axis mask saves nothing).
    pub(crate) fn compute_eri_gradient<G>(
        &self,
        sink: &mut G,
        range_operator: crate::context::two_electron::ERIRangeOperator,
        pairwise_threshold: Option<f64>,
        active: Option<&[bool]>,
    ) -> Result<(), AedelstenError>
    where
        G: eri::GradientSink,
    {
        let n_centres = self.get_n_gradient_centres();
        if let Some(mask) = active {
            if mask.len() != n_centres {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n_centres, actual: mask.len() });
            }
        }
        let centres = eri::GradientCentres::new(n_centres, active);
        dispatch!(self, compute_eri_gradient(sink, range_operator, pairwise_threshold, centres)).inspect(|_| crate::citations::gradients())
    }

    /// `gradient` is `3 x n_centres` and every density of `terms` `n x n`.
    pub(crate) fn check_two_electron_gradient(&self, gradient: &[f64], terms: &[TwoElectronTerm<'_>]) -> Result<(), AedelstenError> {
        let (n, n_centres) = (self.get_n_cont(), self.get_n_gradient_centres());
        if gradient.len() != 3 * n_centres {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * n_centres, actual: gradient.len() });
        }
        for density in terms.iter().flat_map(|term| [term.left, term.right]) {
            if density.len() != n * n {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n * n, actual: density.len() });
            }
        }
        Ok(())
    }

    /// The two-electron energy gradient, `dE_2e/dX` for every gradient centre X, into `gradient`: a column-major
    /// `(3, get_n_gradient_centres())` matrix (`gradient[3 * X + axis]`), OVERWRITTEN. The contracted twin of the
    /// direct J/K builds: `terms` states the energy, `E_2e = 1/2 sum_terms coeff sum_abcd W (ab|cd)` with
    /// `W = left[a, b] right[c, d]` (Coulomb) or `left[a, c] right[b, d]` (exchange), each density `n x n`
    /// (`get_n_cont()`), column-major, borrowed as slices; see `TwoElectronTerm` for RHF and UHF. Only the integral
    /// derivatives: the density-derivative (Pulay) terms come with the one-electron (overlap) gradient. One range
    /// operator per call. `active`: the frozen-centre mask, one flag per gradient centre, `None` = every centre;
    /// frozen centres are left zero and work that only feeds them is skipped (to freeze single coordinates, keep
    /// their centre active and zero those entries afterwards). Requires `set_eri_data` first. Every contraction
    /// type.
    pub fn compute_two_electron_gradient(
        &self,
        gradient: &mut [f64],
        terms: &[TwoElectronTerm<'_>],
        range_operator: crate::context::two_electron::ERIRangeOperator,
        pairwise_threshold: Option<f64>,
        active: Option<&[bool]>,
    ) -> Result<(), AedelstenError> {
        self.check_two_electron_gradient(gradient, terms)?;
        let mut accumulator = ::fock_constructors::DirectEriGradient::new(terms, self.get_n_cont(), gradient);
        self.compute_eri_gradient(&mut accumulator, range_operator, pairwise_threshold, active)
    }

    // The derivative tensors themselves, `d(ab|cd)/dX_axis` for every gradient centre X, into `tensors`
    // (`3 * get_n_gradient_centres()` packed S8 tensors of `n` functions, index `3 * X + axis`), accumulated: zero them
    // first. Unscreened only (`set_eri_data(None)`, no pairwise threshold). `active` as in
    // `compute_two_electron_gradient`. For the crate's tests of the contracted route.
    #[cfg(test)]
    pub(crate) fn compute_eri_gradient_tensors(
        &self,
        tensors: &mut [packed::PackedERITensor<f64>],
        range_operator: crate::context::two_electron::ERIRangeOperator,
        active: Option<&[bool]>,
    ) -> Result<(), AedelstenError> {
        let n_centres = self.get_n_gradient_centres();
        if tensors.len() != 3 * n_centres {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * n_centres, actual: tensors.len() });
        }
        self.compute_eri_gradient(&mut eri::GradientTensors::new(tensors), range_operator, None, active)
    }

}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    pub fn compute_eri_gradient<G>(
        &self,
        sink: &mut G,
        range_operator: crate::context::two_electron::ERIRangeOperator,
        pairwise_threshold: Option<f64>,
        centres: eri::GradientCentres<'_>,
    ) -> Result<(), AedelstenError>
    where
        G: eri::GradientSink,
    {
        let shell_pairs = self.shell_pairs.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("shell pairs: {e} (set_shell_pairs())")))?;
        let eri_data = self.eri_data.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("ERI data is not resident in memory; call set_eri_data() first".to_string())
        })?;
        let scratch_cell = self.scratch.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("scratch: {e} (set_scratch())")))?;
        let mut scratch = scratch_cell.borrow_mut();
        crate::context::two_electron::compute_eri_gradient(
            shell_pairs,
            eri_data,
            sink,
            &mut scratch, &self.comm,
            range_operator,
            pairwise_threshold,
            centres,
        )
    }

    pub fn get_n_gradient_centres(&self) -> usize {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        qm_region.input.get_n_gradient_centres()
    }

    fn prepare_classical_gradient<'a>(&self, gradient: &'a mut [f64], active: Option<&'a [bool]>) -> Result<CentreGradient<'a>, AedelstenError> {
        let n_centres = self.get_n_gradient_centres();
        if gradient.len() != 3 * n_centres {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * n_centres, actual: gradient.len() });
        }
        if let Some(mask) = active {
            if mask.len() != n_centres {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n_centres, actual: mask.len() });
            }
        }
        gradient.fill(0.0);
        Ok(CentreGradient { gradient, centres: eri::GradientCentres::new(n_centres, active) })
    }

    /// The gradient of `compute_nuclear_repulsion_energy`: nucleus-nucleus, nucleus-core AIMP and core AIMP-core AIMP.
    pub fn compute_nuclear_repulsion_gradient(&self, gradient: &mut [f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        let mut out = self.prepare_classical_gradient(gradient, active)?;
        let qm_region = self.qm_region.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("input: {e}")))?;
        let nuclear_potential = qm_region.input.get_nuclear_potential();
        let core_regions = qm_region.input.get_core_regions();
        if let Some(potential) = nuclear_potential {
            potential.compute_nuclear_repulsion_gradient(|i, g| out.add(nucleus_centre(potential, i), g));
        }
        if let (Some(potential), Some(core)) = (nuclear_potential, core_regions) {
            // Both sides move; the two callbacks write into one matrix, so collect through a shared cell.
            let out = std::cell::RefCell::new(&mut out);
            potential.compute_screened_coulomb_interaction_gradient(
                core,
                |i, g| out.borrow_mut().add(nucleus_centre(potential, i), g),
                |j, g| out.borrow_mut().add(core[j].get_idx(), g),
            );
        }
        if let Some(core) = core_regions {
            sources::screened_coulomb_self_interaction_gradient(core, |i, g| out.add(core[i].get_idx(), g));
        }
        Ok(())
    }

    pub fn compute_nuclear_aimp_embedding_gradient(&self, gradient: &mut [f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        let mut out = self.prepare_classical_gradient(gradient, active)?;
        let qm_region = self.qm_region.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("input: {e}")))?;
        let Some(embedding) = qm_region.input.get_embedding_potentials() else { return Ok(()) };
        if let Some(potential) = qm_region.input.get_nuclear_potential() {
            potential.compute_screened_coulomb_interaction_gradient(embedding, |i, g| out.add(nucleus_centre(potential, i), g), |_, _| {});
        }
        if let Some(core) = qm_region.input.get_core_regions() {
            sources::screened_coulomb_cross_interaction_gradient(core, embedding, |i, g| out.add(core[i].get_idx(), g), |_, _| {});
        }
        Ok(())
    }

    /// The gradient of `compute_nuclear_multipole_interaction_energy`: the QM nuclei and core AIMPs in the fixed
    /// external multipoles' potential. An error without external multipoles (the energy is NaN then).
    pub fn compute_nuclear_multipole_interaction_gradient(&self, gradient: &mut [f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        let multipoles = self.external_multipoles.get_cache().map_err(|e| {
            AedelstenError::CacheNotReady(format!("external multipoles: {e} (set_external_multipoles())"))
        })?;
        let mut out = self.prepare_classical_gradient(gradient, active)?;
        let qm_region = self.qm_region.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("input: {e}")))?;
        match multipoles {
            ExternalMultipoles::Charges(charges) => {
                if let Some(potential) = qm_region.input.get_nuclear_potential() {
                    potential.compute_nuclear_interaction_gradient(charges, |i, g| out.add(nucleus_centre(potential, i), g));
                }
                if let Some(core) = qm_region.input.get_core_regions() {
                    sources::screened_coulomb_interaction_gradient(charges, core, |_, _| {}, |j, g| out.add(core[j].get_idx(), g));
                }
            }
            ExternalMultipoles::Multipoles(sources) => self.add_qm_charges_in_multipoles(sources, &mut out)?,
        }
        Ok(())
    }

    /// Adds the gradient of the energy of the QM nuclei and core AIMPs in the potential of fixed point multipoles
    /// (the external multipoles, or the induced dipoles for `compute_induction_gradient`).
    pub(crate) fn add_qm_charges_in_multipoles(&self, multipoles: &[sources::PointMultipole], out: &mut CentreGradient<'_>) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("input: {e}")))?;
        if let Some(potential) = qm_region.input.get_nuclear_potential() {
            potential.compute_general_multipole_interaction_gradient(multipoles, |i, g| out.add(nucleus_centre(potential, i), g));
        }
        if let Some(core) = qm_region.input.get_core_regions() {
            sources::general_multipole_screened_interaction_gradient(multipoles, core, |j, g| out.add(core[j].get_idx(), g));
        }
        Ok(())
    }
}

/// The gradient matrix, restricted to the differentiated centres.
pub(crate) struct CentreGradient<'a> {
    gradient: &'a mut [f64],
    centres: eri::GradientCentres<'a>,
}

impl<'a> CentreGradient<'a> {
    /// `gradient` with its centres.
    pub(crate) fn new(gradient: &'a mut [f64], centres: eri::GradientCentres<'a>) -> Self {
        CentreGradient { gradient, centres }
    }

    /// Adds `g` to centre `centre`'s column, if it is a differentiated centre.
    pub(crate) fn add(&mut self, centre: usize, g: [f64; 3]) {
        if self.centres.contains(centre) {
            for axis in 0..3 {
                self.gradient[3 * centre + axis] += g[axis];
            }
        }
    }
}

/// The centre of member `i` of the nuclear charges.
fn nucleus_centre(potential: &NuclearPotential, i: usize) -> usize {
    match potential {
        NuclearPotential::PointCharge(charges) => charges[i].get_idx(),
        NuclearPotential::GaussianCharge(charges) => charges[i].get_idx(),
    }
}

#[cfg(test)]
#[path = "tests/eri_gradient.rs"]
mod eri_gradient_tests;
