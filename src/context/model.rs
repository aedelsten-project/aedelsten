// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The model basis: the free-atom reference orbitals of the QM atoms (for an atomic frozen core), their overlaps with
// the working basis, the model Fockian, and the overlap of this context's basis with another context's.

use super::*;

impl AedelstenContext {
    /// The model Fockian of the start guess, `F = S_mw^T diag(E) S_mw` (`n x n`, `n = get_n_cont()`, column-major,
    /// OVERWRITTEN): `S_mw` the overlap of the model basis's free-atom reference orbitals with the working basis
    /// (`compute_model_basis_overlaps`), `E` their reference orbital energies.
    pub fn compute_model_fockian(&self, output: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, compute_model_fockian(output))
    }

    /// The number of model functions: the free-atom reference orbitals of the model basis (ANO-RCC, the model
    /// Fockian's), one per component, on the QM centres that are atoms (no ghosts, no pseudocharges); a core AIMP's
    /// share is left out. The length `get_model_functions` needs and the rows of `compute_model_basis_overlaps`.
    pub fn get_n_model_functions(&self) -> Result<usize, AedelstenError> {
        dispatch!(self, get_n_model_functions())
    }

    /// One `ModelFunction` per model function (`output.len()` must equal `get_n_model_functions()`), in the model
    /// basis's AO order: centre, `l`, `m`, the ordinal within its `l` in the reference library (counting what a core
    /// AIMP covers) and the reference orbital energy. See `ModelFunction`'s doc comment.
    pub fn get_model_functions(&self, output: &mut [ModelFunction]) -> Result<(), AedelstenError> {
        dispatch!(self, get_model_functions(output))
    }

    /// The overlaps of the model basis: `s_model_working` (`n_model x n`, `n_model = get_n_model_functions()`,
    /// `n = get_n_cont()`), model functions by working basis functions, and, if given, `s_model_model`
    /// (`n_model x n_model`). Both column-major, OVERWRITTEN. `compute_model_fockian` is `S_mw^T diag(E) S_mw` with
    /// the same `S_mw`.
    pub fn compute_model_basis_overlaps(&self, s_model_working: &mut [f64], s_model_model: Option<&mut [f64]>) -> Result<(), AedelstenError> {
        dispatch!(self, compute_model_basis_overlaps(s_model_working, s_model_model))
    }

    /// The overlap between this context's basis (rows) and `other`'s (columns) -- see
    /// `AedelstenContextImpl::compute_overlap_with` for the layout, the scratch rule and the
    /// checks. Any combination of contraction types is allowed.
    pub fn compute_overlap_with(&self, other: &AedelstenContext, output: &mut [f64]) -> Result<(), AedelstenError> {
        macro_rules! against_other {
            ($ctx:expr) => {
                match other {
                    AedelstenContext::General(o) => $ctx.compute_overlap_with(o, output),
                    AedelstenContext::Segmented(o) => $ctx.compute_overlap_with(o, output),
                    AedelstenContext::Primitive(o) => $ctx.compute_overlap_with(o, output),
                }
            };
        }
        match self {
            AedelstenContext::General(c) => against_other!(c),
            AedelstenContext::Segmented(c) => against_other!(c),
            AedelstenContext::Primitive(c) => against_other!(c),
        }
    }
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    // The model basis of the model Fockian and of the atomic reference orbitals (setup_model_fockian_basis): built
    // from the reference library on each call, which these one-shot queries make once per stage.
    pub(crate) fn model_basis(&self) -> Result<basis::model_fockian::ModelFockianBasis, AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        Ok(basis::model_fockian::setup_model_fockian_basis(&qm_region.input, basis::model_fockian::ANO_RCC)?)
    }

    // `s = <model | other>`, model functions (rows) by `other`'s (columns).
    pub(crate) fn model_basis_overlap<T: basis::contraction::ShellKind>(
        &self,
        model: &basis::model_fockian::ModelFockianBasis,
        other: &[basis::shell_types::BasisShell<T>],
        s: &mut arrays::matrix::MatrixViewMut<'_, f64>,
    ) -> Result<(), AedelstenError> {
        use oneint::OneElectronProjectionIntegrals as _;
        let scratch_cell = self.scratch.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
        })?;
        let mut scratch = scratch_cell.borrow_mut();
        oneint::OneIntProjectionWorkStruct::new(model.get_shells(), other, &mut scratch).compute_overlap_integrals(s)?;
        Ok(())
    }

    pub fn get_n_model_functions(&self) -> Result<usize, AedelstenError> {
        Ok(self.model_basis()?.get_basis_info().get_n_cont())
    }

    pub fn get_model_functions(&self, output: &mut [ModelFunction]) -> Result<(), AedelstenError> {
        let model = self.model_basis()?;
        let functions = model.functions();
        if output.len() != functions.len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: functions.len(), actual: output.len() });
        }
        for (out, function) in output.iter_mut().zip(functions) {
            *out = function;
        }
        Ok(())
    }

    pub fn compute_model_basis_overlaps(&self, s_model_working: &mut [f64], s_model_model: Option<&mut [f64]>) -> Result<(), AedelstenError> {
        let model = self.model_basis()?;
        let (n_model, n) = (model.get_basis_info().get_n_cont(), self.get_n_cont());
        let mut s_mw = AedelstenContext::matrix_of_mut(s_model_working, n_model, n)?;
        let s_mm = s_model_model.map(|s| AedelstenContext::matrix_of_mut(s, n_model, n_model)).transpose()?;
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        self.model_basis_overlap(&model, qm_region.input.get_basis_shells(), &mut s_mw)?;
        if let Some(mut s_mm) = s_mm {
            self.model_basis_overlap(&model, model.get_shells(), &mut s_mm)?;
        }
        Ok(())
    }

    // Model-Fock start-guess Fock matrix: F_model = S^T diag(E) S, where S is the
    // overlap between setup_model_fockian's model basis (rows) and this context's
    // own working basis (columns, expected_n of them), and E is that model basis's
    // own unfolded reference orbital energies (see setup_model_fockian's own doc
    // comment for where both come from).
    pub fn compute_model_fockian(&self, output: &mut [f64]) -> Result<(), AedelstenError> {
        let expected_n = self.get_n_cont();
        check_matrix_dimensions(expected_n, output)?;
        let mut wrapped_output = arrays::matrix::MatrixViewMut::new(expected_n, expected_n, output);

        let model = self.model_basis()?;
        let n_model = model.get_basis_info().get_n_cont();

        // S: model basis (rows) x working basis (columns) overlap.
        let mut s = arrays::matrix::Matrix::<f64>::new(n_model, expected_n);
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        self.model_basis_overlap(&model, qm_region.input.get_basis_shells(), &mut arrays::matrix::MatrixTraitMut::view_mut(&mut s))?;

        // E: model basis's own reference orbital energies, on the diagonal.
        let mut e = arrays::matrix::Matrix::<f64>::new(n_model, n_model);
        e.view_diagonal_mut().copy_from(model.get_energies());

        let mut scratch = arrays::matrix::Matrix::<f64>::new(n_model, expected_n);
        e.congruence_transform_prealloc(&s, &mut scratch, &mut wrapped_output);

        Ok(())
    }

    /// The overlap between this context's basis and `other`'s, which may differ in everything: basis
    /// set, geometry, harmonic types, contraction type (general, segmented, primitive in any
    /// combination). `output` is `self.get_n_cont()` rows by `other.get_n_cont()` columns,
    /// column-major: `output[i + j * n_self] = <self_i | other_j>`: the `S_BA` that projects orbitals from one
    /// basis onto another.
    ///
    /// Needs both contexts' inputs resident, but neither shell pairs nor -- necessarily -- a
    /// scratch: the larger of the two is used if both have one (it must be large enough for the
    /// mixed pairs, or this fails with an arena error naming the size needed). If only one has a
    /// scratch that one is used, so the context that merely describes an old basis need not
    /// allocate anything; if neither has one this is an error.
    ///
    /// No screening is applied: every primitive pair is included, so this is the exact overlap. The
    /// ordinary `get_overlap_operator` path drops primitive pairs below the shell pairs' overlap
    /// threshold, so the two differ from each other by up to about that much.
    pub fn compute_overlap_with<S2: basis::contraction::ShellKind>(&self, other: &AedelstenContextImpl<S2>, output: &mut [f64]) -> Result<(), AedelstenError> {
        use oneint::OneElectronProjectionIntegrals as _;
        let not_resident = |which: &str| {
            AedelstenError::CacheNotReady(format!("the {which} context's input has been freed (free_input)"))
        };
        let region_self = self.qm_region.get_cache().map_err(|_| not_resident("first"))?;
        let region_other = other.qm_region.get_cache().map_err(|_| not_resident("second"))?;
        let (n_self, n_other) = (region_self.input.get_n_cont(), region_other.input.get_n_cont());
        if output.len() != n_self * n_other {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: n_self * n_other, actual: output.len() });
        }

        let scratch_self = self.scratch.get_cache().ok();
        let scratch_other = other.scratch.get_cache().ok();
        let use_self = match (scratch_self, scratch_other) {
            // The larger buffer is the more likely to be enough.
            (Some(a), Some(b)) => a.borrow().byte_len() >= b.borrow().byte_len(),
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => {
                return Err(AedelstenError::CacheNotReady(
                    "neither context has a scratch buffer; call set_scratch() on at least one of them".to_string(),
                ));
            }
        };
        let scratch_cell = if use_self { scratch_self } else { scratch_other }.expect("chosen above");
        let mut scratch = scratch_cell.borrow_mut();

        output.fill(0.0);
        let mut wrapped_output = arrays::matrix::MatrixViewMut::new(n_self, n_other, output);
        let mut work = oneint::OneIntProjectionWorkStruct::new(region_self.input.get_basis_shells(), region_other.input.get_basis_shells(), &mut scratch);
        work.compute_overlap_integrals(&mut wrapped_output)?;
        Ok(())
    }
}
