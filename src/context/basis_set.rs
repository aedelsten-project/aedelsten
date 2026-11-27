// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The basis and the centres: basis-function and primitive ids, centre labels and geometry, nuclear charges and
// repulsion, moving the centres (a geometry step), Cartesian rescaling for export, and the AO amplitudes on
// arbitrary points.
//
// Moving the QM centres of an existing context (a geometry step), instead of building a new context: the basis,
// the scratch and the whole environment (external multipoles, polarizable sites, embedding AIMPs, orthogonality
// ghosts) stay; everything derived from the QM geometry is invalidated and has to be set again with its own setter.
//
// Cartesian rescaling for export: from this library's normalization (every Cartesian component normalized on its
// own) to a convention of one constant per Cartesian shell, pinned to a reference component
// (`cart2sph::SharedCartesianReference`, Molcas's and Dalton's). Only for writing AO coefficients for a reader that
// expects such a convention; the calculations never use it. Spherical shells and s and p Cartesian shells have
// factor 1.

use super::*;
use crate::context::cache::AedelstenCacheState;
use basis::shell_types::HarmonicType;
use cart2sph::SharedCartesianReference;
use arrays::matrix::MatrixTraitMut;

impl AedelstenContext {
    /// The number of contracted basis functions (AOs): the dimension of every AO matrix.
    pub fn get_n_cont(&self) -> usize { dispatch!(self, get_n_cont()) }

    /// The repulsion energy of the QM region's charges: nucleus-nucleus, nucleus-core AIMP and core AIMP-core AIMP (the
    /// AIMPs' screened charges).
    pub fn compute_nuclear_repulsion_energy(&self) -> f64 { dispatch!(self, compute_nuclear_repulsion_energy()) }

    /// The total charge of the QM region's nuclei and core AIMPs.
    pub fn compute_nuclear_charge(&self) -> f64 { dispatch!(self, compute_nuclear_charge()) }

    /// The centre of mass of the real atoms (bohr).
    pub fn compute_centre_of_mass(&self) -> [f64; 3] { dispatch!(self, compute_centre_of_mass()) }

    /// Rescales `matrix` (`n_ao` x `n_cols`, column-major, AO axis = rows) in place, from this
    /// library's own Cartesian normalization to `convention`'s -- e.g. an AO x MO coefficient
    /// matrix being written out for a reader that expects `convention`. Returns whether anything
    /// changed (`false` when every factor was exactly 1.0 -- no Cartesian d/f/g/... shells in this
    /// basis, or none under `convention` -- so `matrix` is untouched).
    pub fn rescale_cartesian_ao_rows(&self, convention: SharedCartesianReference, n_cols: usize, matrix: &mut [f64]) -> Result<bool, AedelstenError> {
        dispatch!(self, rescale_cartesian_ao_rows(convention, n_cols, matrix))
    }

    /// The inverse of `rescale_cartesian_ao_rows`: takes an AO x MO coefficient matrix written in
    /// `convention`'s Cartesian normalization (e.g. one read back from a Molcas-layout file) to this
    /// library's own. Same layout and return value (whether anything changed).
    pub fn unrescale_cartesian_ao_rows(&self, convention: SharedCartesianReference, n_cols: usize, matrix: &mut [f64]) -> Result<bool, AedelstenError> {
        dispatch!(self, unrescale_cartesian_ao_rows(convention, n_cols, matrix))
    }

    /// Same as `rescale_cartesian_ao_rows`, but for `matrix` (`n_rows` x `n_ao`, AO axis =
    /// columns) -- e.g. a symmetric AO x AO matrix needs both.
    pub fn rescale_cartesian_ao_columns(&self, convention: SharedCartesianReference, n_rows: usize, matrix: &mut [f64]) -> Result<bool, AedelstenError> {
        dispatch!(self, rescale_cartesian_ao_columns(convention, n_rows, matrix))
    }

    /// One entry per AO (`output.len()` must equal `get_n_cont()`), in AO order. See
    /// `BasisFunctionId`'s own doc comment.
    pub fn get_basis_function_ids(&self, output: &mut [BasisFunctionId]) -> Result<(), AedelstenError> {
        dispatch!(self, get_basis_function_ids(output))
    }

    /// Length `get_primitive_ids`/`get_primitives` need `output` sized to.
    pub fn get_n_primitive_rows(&self) -> usize { dispatch!(self, get_n_primitive_rows()) }

    /// Parallel to `get_primitives` (row `i` of each belongs together); `output.len()` must equal
    /// `get_n_primitive_rows()`. See `PrimitiveId`'s own doc comment.
    pub fn get_primitive_ids(&self, output: &mut [PrimitiveId]) -> Result<(), AedelstenError> {
        dispatch!(self, get_primitive_ids(output))
    }

    /// Parallel to `get_primitive_ids`; `output.len()` must equal `get_n_primitive_rows()`. See
    /// `Primitive`'s own doc comment.
    pub fn get_primitives(&self, output: &mut [Primitive]) -> Result<(), AedelstenError> {
        dispatch!(self, get_primitives(output))
    }

    /// `centre_idx`'s label (`BasisFunctionId`/`PrimitiveId`'s `centre` field is this same index,
    /// 1-based -- pass `centre - 1` here). `None` only for an out-of-range index.
    pub fn get_centre_label(&self, centre_idx: usize) -> Option<CentreLabel> { dispatch!(self, get_centre_label(centre_idx)) }

    /// Number of centres (real atoms, ghost centres and orthogonality ghosts): the length
    /// `get_centre_geometry` needs `output` sized to, and one more than the highest 0-based centre
    /// index (`BasisFunctionId::centre` is 1-based, so its maximum equals this).
    pub fn get_n_centres(&self) -> usize { dispatch!(self, get_n_centres()) }

    /// Atomic number, position (Bohr) and kind of every centre, indexed by 0-based centre index
    /// (pass `centre - 1` when cross-referencing `BasisFunctionId`/`PrimitiveId`, as for
    /// `get_centre_label`); `output.len()` must equal `get_n_centres()`. Ghost centres and
    /// orthogonality ghosts are included -- check `kind` before treating an entry as an atom; see
    /// `CentreGeometry`'s own doc comment.
    pub fn get_centre_geometry(&self, output: &mut [CentreGeometry]) -> Result<(), AedelstenError> {
        dispatch!(self, get_centre_geometry(output))
    }

    /// Every centre's charge, from its nucleus or its core AIMP, 0 for a ghost centre or an orthogonality ghost;
    /// `output.len()` must equal `get_n_centres()`.
    pub fn get_nuclear_charges(&self, output: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, get_nuclear_charges(output))
    }

    /// Moves the gradient centres to `positions`, a column-major `(3, get_n_gradient_centres())` matrix (the layout of
    /// the gradients), for the next geometry step: the basis, scratch and environment stay, and the caches derived
    /// from the geometry are invalidated (set them again: shell pairs, ERI data, primitive shell pairs, X2C). See
    /// `set_centre_positions` below. Every MPI rank must pass the same positions.
    pub fn set_centre_positions(&mut self, positions: &[f64]) -> Result<(), AedelstenError> {
        dispatch!(self, set_centre_positions(positions))
    }

    /// The sum of the shells' primitive AOs; a primitive shared by sibling segments counts once per sibling.
    pub fn get_n_prim(&self) -> usize { dispatch!(self, get_n_prim()) }

    /// The number of distinct primitive AOs: the dimension of the primitive basis (`set_primitive_shell_pairs`).
    pub fn get_n_prim_unique(&self) -> usize { dispatch!(self, get_n_prim_unique()) }

    /// The number of shells.
    pub fn get_n_shells(&self) -> usize { dispatch!(self, get_n_shells()) }

    /// Every shell, in AO order: its centre (0-based), angular momentum (negated for a Cartesian shell), first AO and
    /// AO count. `output` has `get_n_shells()` entries.
    pub fn get_shells(&self, output: &mut [ShellInfo]) -> Result<(), AedelstenError> { dispatch!(self, get_shells(output)) }

    /// Per shell, the distance from its centre beyond which every one of its AOs and every Cartesian component of
    /// their gradients is below `threshold` in magnitude (never below 1 bohr). `output` has `get_n_shells()` entries.
    pub fn compute_shell_extents(&self, threshold: f64, output: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, compute_shell_extents(threshold, output))
    }

    /// The values of the AOs of the shells `shells` (0-based, any order and subset) at the points `points`, and
    /// optionally their gradients. `points` is column-major `(point_rows, n_points)`, `point_rows` 3 (x, y, z) or 4
    /// (x, y, z and a weight that is not read). `values` is column-major `(n_listed, n_points)`, `n_listed` the AOs of
    /// the listed shells in the order of the list; `gradients` three such blocks, x, y, z. Allocates nothing.
    pub fn compute_ao_values(
        &self,
        shells: &[usize],
        points: &[f64],
        point_rows: usize,
        values: &mut [f64],
        gradients: Option<&mut [f64]>,
    ) -> Result<(), AedelstenError> {
        self.compute_ao_values_of(shells.iter().copied(), points, point_rows, values, gradients)
    }

    /// The values, gradients and second derivatives of the AOs of the shells `shells` (0-based, any order and subset)
    /// at the points `points`, laid out as in `compute_ao_values`: `values` `(n_listed, n_points)`, `gradients` three
    /// such blocks (x, y, z), `hessians` six (xx, xy, xz, yy, yz, zz). A shell's extent (`compute_shell_extents`)
    /// bounds its values and gradients, not its second derivatives. Allocates nothing.
    pub fn compute_ao_hessians(
        &self,
        shells: &[usize],
        points: &[f64],
        point_rows: usize,
        values: &mut [f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) -> Result<(), AedelstenError> {
        self.compute_ao_hessians_of(shells.iter().copied(), points, point_rows, values, gradients, hessians)
    }

    // `compute_ao_hessians` for any iterator of shell indices (the C form's, converted on the fly).
    pub(crate) fn compute_ao_hessians_of(
        &self,
        shells: impl Iterator<Item = usize> + Clone,
        points: &[f64],
        point_rows: usize,
        values: &mut [f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_ao_hessians(shells, points, point_rows, values, gradients, hessians))
    }

    // `compute_ao_values` for any iterator of shell indices (the C form's, converted on the fly).
    pub(crate) fn compute_ao_values_of(
        &self,
        shells: impl Iterator<Item = usize> + Clone,
        points: &[f64],
        point_rows: usize,
        values: &mut [f64],
        gradients: Option<&mut [f64]>,
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_ao_values(shells, points, point_rows, values, gradients))
    }
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    // Basis queries. Panic if the input has been freed (free_input).
    pub fn get_n_cont(&self) -> usize {
        self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_n_cont()
    }

    pub fn get_n_prim(&self) -> usize {
        self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_n_prim()
    }

    pub fn get_n_prim_unique(&self) -> usize {
        self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_n_prim_unique()
    }

    pub fn compute_nuclear_repulsion_energy(&self) -> f64 {
        let mut nuclear_energy = 0.0;
        let nuclear_potential = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_nuclear_potential();
        let core_regions = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_core_regions();
        match (nuclear_potential, core_regions) {
            (Some(nuclear_potential), None) => {
                nuclear_energy += nuclear_potential.compute_nuclear_repulsion_energy();
            }
            (None, Some(core_regions)) => {
                nuclear_energy += basis::charge_carriers::compute_screened_coulomb_self_interaction(core_regions);
            }
            (Some(nuclear_potential), Some(core_regions)) => {
                nuclear_energy += nuclear_potential.compute_nuclear_repulsion_energy();
                nuclear_energy += nuclear_potential.compute_screened_coulomb_interaction(core_regions);
                nuclear_energy += basis::charge_carriers::compute_screened_coulomb_self_interaction(core_regions);
            }
            // setup_basis_system() rejects an input with neither a nuclear
            // potential nor core regions (SetupError::NoCoreDescription), so
            // a successfully constructed AedelstenContextImpl can't reach this.
            (None, None) => unreachable!("input has neither a nuclear potential nor core regions; setup_basis_system() should have rejected this"),
        };
        nuclear_energy
    }

    pub fn compute_nuclear_charge(&self) -> f64 {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let input = &qm_region.input;
        let mut charge = match input.get_nuclear_potential() {
            Some(basis::charge_carriers::NuclearPotential::PointCharge(charges)) => charges.iter().map(|c| c.get_charge()).sum(),
            Some(basis::charge_carriers::NuclearPotential::GaussianCharge(charges)) => charges.iter().map(|c| c.get_charge()).sum(),
            None => 0.0,
        };
        if let Some(core_regions) = input.get_core_regions() {
            charge += core_regions.iter().map(|aimp| aimp.get_charge()).sum::<f64>();
        }
        charge
    }

    pub fn compute_centre_of_mass(&self) -> [f64; 3] {
        self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_qm_atoms().compute_centre_of_mass()
    }

    pub fn rescale_cartesian_ao_rows(&self, convention: SharedCartesianReference, n_cols: usize, matrix: &mut [f64]) -> Result<bool, AedelstenError> {
        self.rescale_ao_rows(convention, n_cols, matrix, false)
    }

    pub fn unrescale_cartesian_ao_rows(&self, convention: SharedCartesianReference, n_cols: usize, matrix: &mut [f64]) -> Result<bool, AedelstenError> {
        self.rescale_ao_rows(convention, n_cols, matrix, true)
    }

    // Shared body of the two above: multiplies row r by factor[r], or by 1/factor[r] when `invert`.
    pub(crate) fn rescale_ao_rows(&self, convention: SharedCartesianReference, n_cols: usize, matrix: &mut [f64], invert: bool) -> Result<bool, AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let n_ao = qm_region.input.get_n_cont();
        let expected = n_ao * n_cols;
        if matrix.len() != expected {
            return Err(AedelstenError::InvalidMatrixDimensions { expected, actual: matrix.len() });
        }
        let mut factors = vec![1.0; n_ao];
        compute_cartesian_rescale_factors(
            qm_region.input.get_basis_shells().iter().map(|shell| (shell.get_lqn(), shell.get_harmonic_type(), shell.get_n_cont())),
            convention,
            &mut factors,
        );
        let rescaled = factors.iter().any(|&f| f != 1.0);
        if invert {
            // Every factor is a positive, finite normalization ratio (see compute_cartesian_rescale_factors).
            factors.iter_mut().for_each(|f| *f = 1.0 / *f);
        }
        let mut view = arrays::matrix::MatrixViewMut::new(n_ao, n_cols, matrix);
        rescale_rows(&mut view, &factors);
        Ok(rescaled)
    }

    pub fn rescale_cartesian_ao_columns(&self, convention: SharedCartesianReference, n_rows: usize, matrix: &mut [f64]) -> Result<bool, AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let n_ao = qm_region.input.get_n_cont();
        let expected = n_rows * n_ao;
        if matrix.len() != expected {
            return Err(AedelstenError::InvalidMatrixDimensions { expected, actual: matrix.len() });
        }
        let mut factors = vec![1.0; n_ao];
        compute_cartesian_rescale_factors(
            qm_region.input.get_basis_shells().iter().map(|shell| (shell.get_lqn(), shell.get_harmonic_type(), shell.get_n_cont())),
            convention,
            &mut factors,
        );
        let rescaled = factors.iter().any(|&f| f != 1.0);
        let mut view = arrays::matrix::MatrixViewMut::new(n_rows, n_ao, matrix);
        rescale_columns(&mut view, &factors);
        Ok(rescaled)
    }

    pub fn get_basis_function_ids(&self, output: &mut [BasisFunctionId]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let ids = qm_region.input.get_basis_info().get_basis_function_ids();
        if output.len() != ids.len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: ids.len(), actual: output.len() });
        }
        output.copy_from_slice(ids);
        Ok(())
    }

    pub fn get_n_primitive_rows(&self) -> usize {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        qm_region.input.get_basis_info().get_primitive_ids().len()
    }

    pub fn get_primitive_ids(&self, output: &mut [PrimitiveId]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let ids = qm_region.input.get_basis_info().get_primitive_ids();
        if output.len() != ids.len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: ids.len(), actual: output.len() });
        }
        output.copy_from_slice(ids);
        Ok(())
    }

    pub fn get_primitives(&self, output: &mut [Primitive]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let primitives = qm_region.input.get_basis_info().get_primitives();
        if output.len() != primitives.len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: primitives.len(), actual: output.len() });
        }
        output.copy_from_slice(primitives);
        Ok(())
    }

    pub fn get_centre_label(&self, centre_idx: usize) -> Option<CentreLabel> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        qm_region.input.get_centre_label(centre_idx)
    }

    pub fn get_n_centres(&self) -> usize {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        qm_region.input.get_n_centres()
    }

    pub fn get_centre_geometry(&self, output: &mut [CentreGeometry]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let geometry = qm_region.input.get_centre_geometry();
        if output.len() != geometry.len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: geometry.len(), actual: output.len() });
        }
        output.copy_from_slice(&geometry);
        Ok(())
    }

    pub fn get_nuclear_charges(&self, output: &mut [f64]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let geometry = qm_region.input.get_nuclear_charges()?;
        if output.len() != geometry.len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: geometry.len(), actual: output.len() });
        }
        output.copy_from_slice(&geometry);
        Ok(())
    }

    pub fn get_n_shells(&self) -> usize {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        qm_region.input.get_ao_evaluator().n_shells()
    }

    pub fn get_shells(&self, output: &mut [ShellInfo]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let evaluator = qm_region.input.get_ao_evaluator();
        if output.len() != evaluator.n_shells() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: evaluator.n_shells(), actual: output.len() });
        }
        for (s, info) in output.iter_mut().enumerate() {
            *info = evaluator.shell_info(s);
        }
        Ok(())
    }

    pub fn compute_shell_extents(&self, threshold: f64, output: &mut [f64]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let evaluator = qm_region.input.get_ao_evaluator();
        if output.len() != evaluator.n_shells() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: evaluator.n_shells(), actual: output.len() });
        }
        if !(threshold > 0.0 && threshold.is_finite()) {
            return Err(AedelstenError::StringError(format!("the shell extent threshold must be positive and finite, not {threshold}")));
        }
        for (s, extent) in output.iter_mut().enumerate() {
            *extent = evaluator.shell_extent(s, threshold);
        }
        Ok(())
    }

    pub fn compute_ao_values(
        &self,
        shells: impl Iterator<Item = usize> + Clone,
        points: &[f64],
        point_rows: usize,
        values: &mut [f64],
        gradients: Option<&mut [f64]>,
    ) -> Result<(), AedelstenError> {
        let _span = instrumentation::fine_span("basis::ao_values");
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        Ok(qm_region.input.compute_ao_values(shells, points, point_rows, values, gradients)?)
    }

    pub fn compute_ao_hessians(
        &self,
        shells: impl Iterator<Item = usize> + Clone,
        points: &[f64],
        point_rows: usize,
        values: &mut [f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) -> Result<(), AedelstenError> {
        let _span = instrumentation::fine_span("basis::ao_values");
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        Ok(qm_region.input.compute_ao_hessians(shells, points, point_rows, values, gradients, hessians)?)
    }

    /// Moves every gradient centre to `positions`, a column-major `(3, get_n_gradient_centres())` matrix (the layout
    /// of the gradients). Each centre carries its basis shells, its nucleus and its core AIMP along. Orthogonality
    /// ghosts and embedding AIMPs are the fixed environment and are not in the matrix.
    ///
    /// Invalidated (call the setter again): shell pairs (`set_shell_pairs`), ERI data (`set_eri_data`), primitive
    /// shells, pairs and contraction matrix (`set_primitive_shell_pairs`), the X2C picture change
    /// (`construct_x2c_one_electron_hamiltonian`), the point group and SALCs (`set_symmetry`). Kept: the scratch (sized
    /// by the basis), external multipoles, polarizable sites, the embedding and core AIMP projection operators (both
    /// take their sites' positions at each use). Updated in place: a polarization state's cached static field (the
    /// induced dipoles stay as the next solve's warm start).
    pub fn set_centre_positions(&mut self, positions: &[f64]) -> Result<(), AedelstenError> {
        let qm_region = self.qm_region.get_cache_mut().map_err(|_| {
            AedelstenError::CacheNotReady("input has been freed (free_input)".to_string())
        })?;
        let n_centres = qm_region.input.get_n_gradient_centres();
        let (centres, rest) = positions.as_chunks::<3>();
        if centres.len() != n_centres || !rest.is_empty() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * n_centres, actual: positions.len() });
        }
        // The materialised core AIMP projection operators need nothing: their shells' positions are stamped from
        // the AIMP list (just moved here) at every use, see aimp::compute_aimp_projection_contribution.
        qm_region.input.set_centre_positions(centres).map_err(AedelstenError::StringError)?;

        invalidate_if_derived(&mut self.shell_pairs, "shell pairs", "set_shell_pairs()");
        invalidate_if_derived(&mut self.eri_data, "ERI data", "set_eri_data()");
        invalidate_if_derived(&mut self.primitive_shells, "primitive shells", "set_primitive_shell_pairs()");
        invalidate_if_derived(&mut self.primitive_shell_pairs, "primitive shell pairs", "set_primitive_shell_pairs()");
        invalidate_if_derived(&mut self.contraction_matrix, "contraction matrix", "set_primitive_shell_pairs()");
        invalidate_if_derived(&mut self.picture_change, "X2C picture change", "construct_x2c_one_electron_hamiltonian()");
        invalidate_if_derived(&mut self.x2c_frozen_density, "X2C frozen densities", "set_x2c_frozen_density()");
        invalidate_if_derived(&mut self.symmetry, "point group and SALCs", "set_symmetry()");
        self.refresh_cached_static_field()
    }
}

/// Marks a derived cache stale, naming the setter that rebuilds it. Caches that were never derived stay empty.
fn invalidate_if_derived<C>(cache: &mut AedelstenCacheState<C>, what: &str, setter: &str) {
    if cache.is_already_derived() {
        cache.invalidate(format!(
            "the {what} belong to the previous geometry (the centres were moved by set_centre_positions); call {setter} again"
        ));
    }
}

/// Fills `output` (pre-sized to the basis's total AO count, e.g. `AedelstenContext::get_n_cont()`)
/// with the per-AO factor, in the AO order of `shells` -- `(lqn, harmonic type, n_cont)` per shell, one
/// contiguous `n_cont * degeneracy`-sized block each, contraction outermost, Cartesian/spherical
/// component innermost (see `BasisShell::compute_ao_amplitude`) -- to multiply an AO coefficient by to convert it
/// from this library's convention to `convention`'s.
pub(crate) fn compute_cartesian_rescale_factors(shells: impl IntoIterator<Item = (usize, HarmonicType, usize)>, convention: SharedCartesianReference, output: &mut [f64]) {
    output.fill(1.0);
    let mut ao = 0;
    for (lqn, harmonic_type, n_cont) in shells {
        let degeneracy = match harmonic_type {
            HarmonicType::Cartesian => cart2sph::compute_cartesian_degeneracy(lqn),
            HarmonicType::Spherical => cart2sph::compute_spherical_degeneracy(lqn),
        };
        let block = n_cont * degeneracy;
        if harmonic_type == HarmonicType::Cartesian && lqn >= 2 {
            let components = cart2sph::get_cartesian_quantum_numbers(lqn);
            for cont in 0..n_cont {
                for (idx, &(lx, ly, lz)) in components.iter().enumerate() {
                    output[ao + cont * components.len() + idx] = 1.0 / cart2sph::shared_cartesian_normalization_ratio(lx, ly, lz, convention).sqrt();
                }
            }
        }
        ao += block;
    }
}

/// Scales row `r` of `matrix` by `factors[r]`, for every row -- e.g. an AO x MO coefficient
/// matrix's AO axis.
pub fn rescale_rows<M: MatrixTraitMut<f64>>(matrix: &mut M, factors: &[f64]) {
    assert_eq!(factors.len(), matrix.rows());
    for c in 0..matrix.cols() {
        for r in 0..matrix.rows() {
            matrix[(r, c)] *= factors[r];
        }
    }
}

/// Scales column `c` of `matrix` by `factors[c]`, for every column -- e.g. a symmetric AO x AO
/// matrix needs both `rescale_rows` and `rescale_columns` with the same factors.
pub fn rescale_columns<M: MatrixTraitMut<f64>>(matrix: &mut M, factors: &[f64]) {
    assert_eq!(factors.len(), matrix.cols());
    for c in 0..matrix.cols() {
        for r in 0..matrix.rows() {
            matrix[(r, c)] *= factors[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrays::matrix::Matrix;

    #[test]
    fn factors_are_one_for_s_p_and_spherical_shells() {
        let shells = [
            (0, HarmonicType::Cartesian, 2),
            (1, HarmonicType::Cartesian, 1),
            (2, HarmonicType::Spherical, 1),
        ];
        let mut factors = vec![0.0; 2 * 1 + 3 + 5]; // s (2 cont) + p + spherical d
        compute_cartesian_rescale_factors(shells, SharedCartesianReference::MOLCAS, &mut factors);
        assert!(factors.iter().all(|&f| f == 1.0));
    }

    #[test]
    fn cartesian_d_shell_factors_are_the_inverse_sqrt_of_the_reference_ratios() {
        let shells = [(2, HarmonicType::Cartesian, 1)];
        let mut factors = vec![0.0; 6];
        compute_cartesian_rescale_factors(shells, SharedCartesianReference::MOLCAS, &mut factors);
        let expected: Vec<f64> = [3.0_f64, 1.0, 1.0, 3.0, 1.0, 3.0].iter().map(|r| 1.0 / r.sqrt()).collect();
        for (f, e) in factors.iter().zip(expected.iter()) {
            assert!((f - e).abs() < 1e-12, "{f} vs {e}");
        }
    }

    #[test]
    fn dalton_convention_gives_a_different_rescaling_than_molcas() {
        let shells = [(2, HarmonicType::Cartesian, 1)];
        let mut factors = vec![0.0; 6];
        compute_cartesian_rescale_factors(shells, SharedCartesianReference::DALTON, &mut factors);
        let expected: Vec<f64> = [1.0_f64, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0 / 3.0, 1.0].iter().map(|r| 1.0 / r.sqrt()).collect();
        for (f, e) in factors.iter().zip(expected.iter()) {
            assert!((f - e).abs() < 1e-12, "{f} vs {e}");
        }
    }

    #[test]
    fn multiple_contractions_repeat_the_same_block() {
        let shells = [(2, HarmonicType::Cartesian, 2)];
        let mut factors = vec![0.0; 12];
        compute_cartesian_rescale_factors(shells, SharedCartesianReference::MOLCAS, &mut factors);
        assert_eq!(&factors[0..6], &factors[6..12]);
    }

    #[test]
    fn rescale_rows_and_columns_apply_the_right_axis() {
        let mut m = Matrix::<f64>::new(2, 3);
        for r in 0..2 {
            for c in 0..3 {
                m[(r, c)] = 1.0;
            }
        }
        rescale_rows(&mut m, &[2.0, 3.0]);
        assert_eq!((m[(0, 0)], m[(1, 0)]), (2.0, 3.0));
        rescale_columns(&mut m, &[1.0, 2.0, 3.0]);
        assert_eq!((m[(0, 0)], m[(0, 1)], m[(0, 2)]), (2.0, 4.0, 6.0));
    }
}
