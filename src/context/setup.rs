// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The context's setup and lifecycle: the scratch, the shell pairs, the primitive shells and their pairs, freeing
// cached state, the memory breakdown.

use super::*;

/// Field order `AedelstenContext::memory_breakdown` fills its output buffer in. `&CStr` rather
/// than `&str` so the exact same table also serves `memory_breakdown_field_name_c` over FFI, with
/// no separate C-side copy to keep in sync.
pub const MEMORY_BREAKDOWN_FIELDS: [&std::ffi::CStr; 14] = [
    c"qm_region",
    c"shell_pairs",
    c"scratch",
    c"external_multipoles",
    c"polarizable_sites",
    c"polarization",
    c"embedding_aimp_operators",
    c"primitive_shells",
    c"primitive_shell_pairs",
    c"contraction_matrix",
    c"eri_data",
    c"picture_change",
    c"symmetry",
    c"x2c_frozen_density",
];

impl AedelstenContext {
    /// The memory this context holds now, field by field, in bytes (0 for a field that is empty or invalidated):
    /// `output` has `MEMORY_BREAKDOWN_FIELDS.len()` entries, filled in that order. The flat buffers (the scratch, the
    /// contraction matrix, the picture change, the X2C frozen densities, the ERI data's Hermite coefficients) are exact; the rest are each
    /// type's own estimate, which may leave out what it holds on the heap.
    pub fn memory_breakdown(&self, output: &mut [usize]) -> Result<(), AedelstenError> { dispatch!(self, memory_breakdown(output)) }

    /// Frees the input (basis, centres, nuclei, AIMPs): nothing that needs it can be called afterwards.
    pub fn free_input(&mut self) { dispatch!(self, free_input()) }

    /// Derives the shell pairs of the basis, dropping primitive pairs whose overlap factor is below `overlap_threshold`
    /// (`None`: none dropped). Needed by every integral over the contracted basis; an error if they are already set.
    pub fn set_shell_pairs(&mut self, overlap_threshold: Option<f64>) -> Result<(), AedelstenError> { dispatch!(self, set_shell_pairs(overlap_threshold)) }

    /// Frees the shell pairs, and with them the ERI data (`set_eri_data`).
    pub fn free_shell_pairs(&mut self) { dispatch!(self, free_shell_pairs()) }

    /// Frees the primitive shells `set_primitive_shell_pairs` made.
    pub fn free_primitive_shells(&mut self) { dispatch!(self, free_primitive_shells()) }

    /// Frees the contraction matrix `set_primitive_shell_pairs` made (primitive to contracted AOs).
    pub fn free_contraction_matrix(&mut self) { dispatch!(self, free_contraction_matrix()) }

    /// Decontracts the basis into its primitive shells and derives their pairs (dropped below `overlap_threshold` as in
    /// `set_shell_pairs`) and the contraction matrix: the primitive basis of X2C, AMFI and the integrals asked for with
    /// `primitive`. Nothing to do for a primitive context, whose basis is primitive already.
    pub fn set_primitive_shell_pairs(&mut self, overlap_threshold: Option<f64>) -> Result<(), AedelstenError> {
        dispatch!(self, set_primitive_shell_pairs(overlap_threshold))
    }

    /// Frees the primitive shell pairs (nothing for a primitive context).
    pub fn free_primitive_shell_pairs(&mut self) { dispatch!(self, free_primitive_shell_pairs()) }

    /// Whether the primitive basis is set: always for a primitive context, otherwise after `set_primitive_shell_pairs`.
    pub fn has_primitive(&self) -> bool { dispatch!(self, has_primitive()) }

    /// The buffer every integral computation of this context takes its scratch from. The library makes no judgment
    /// about its size; the caller chooses it, `scratch_bytes` giving what the chosen operations need from the input
    /// alone. That need depends only on the basis's largest shells, never on the size of the molecule; two-electron
    /// integral gradients need the most, one-electron integrals (X2C included) little; deep, generally contracted or
    /// Cartesian shells raise it steeply. The buffer is committed lazily, so unused space costs address space rather
    /// than resident memory (on systems that overcommit).
    ///
    /// The scratch never grows: a computation that runs out of it fails with an arena error naming the size it
    /// needed at that point (a lower bound: a later, larger item may need more).
    pub fn set_scratch(&mut self, size: (usize, arrays::scratch::ByteUnit)) -> Result<(), AedelstenError> { dispatch!(self, set_scratch(size)) }

    /// Frees the scratch buffer.
    pub fn free_scratch(&mut self) { dispatch!(self, free_scratch()) }

    /// Replaces the scratch buffer by one of `size`, e.g. to shrink it to a measured `get_scratch_peak_bytes` plus a
    /// margin, or to grow it after an arena error (which reports the bytes it needed). The new buffer is committed
    /// lazily; the high-water mark starts over. Needs a scratch buffer (`set_scratch`).
    pub fn resize_scratch(&mut self, size: (usize, arrays::scratch::ByteUnit)) -> Result<(), AedelstenError> {
        dispatch!(self, resize_scratch(size))
    }

    /// The most bytes of the scratch any computation has used so far (its high-water mark, alignment padding
    /// included; see `arrays::scratch::Scratch::peak_bytes`). The buffer is committed lazily, so this is also
    /// about how much of it is resident.
    pub fn get_scratch_peak_bytes(&self) -> Result<usize, AedelstenError> { dispatch!(self, get_scratch_peak_bytes()) }

    /// Splits the context's parallel computations over the ranks of `comm` from now on and returns the communicator
    /// they were split over before (to restore it). Every collective method must then be called by every rank of
    /// `comm`; ranks in different groups may make different calls at the same time. Nothing cached depends on it.
    pub fn set_communicator(&mut self, comm: &aedelsten_mpi::Communicator) -> aedelsten_mpi::Communicator { dispatch!(self, set_communicator(comm)) }

    /// The communicator the context's parallel computations are split over.
    pub fn communicator(&self) -> &aedelsten_mpi::Communicator { dispatch!(self, communicator()) }
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    pub fn memory_breakdown(&self, output: &mut [usize]) -> Result<(), AedelstenError> {
        if output.len() != MEMORY_BREAKDOWN_FIELDS.len() {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: MEMORY_BREAKDOWN_FIELDS.len(), actual: output.len() });
        }
        fn ready_bytes<V>(state: &AedelstenCacheState<V>, bytes: impl FnOnce(&V) -> usize) -> usize {
            match state {
                AedelstenCacheState::Ready(v) => bytes(v),
                _ => 0,
            }
        }
        let f64_bytes = std::mem::size_of::<f64>();
        output[0] = ready_bytes(&self.qm_region, |qm| qm.input.approx_memory_bytes());
        output[1] = ready_bytes(&self.shell_pairs, |pairs| std::mem::size_of_val(pairs.as_slice()));
        output[2] = ready_bytes(&self.scratch, |ws| ws.borrow().byte_len());
        output[3] = ready_bytes(&self.external_multipoles, |v| v.approx_memory_bytes());
        output[4] = ready_bytes(&self.polarizable_sites, |v| std::mem::size_of_val(v.as_ref()));
        output[5] = ready_bytes(&self.polarization, |v| v.borrow().approx_memory_bytes());
        output[6] = ready_bytes(&self.embedding_aimp_operators, |v| v.borrow().iter().map(|op| op.approx_memory_bytes()).sum());
        output[7] = ready_bytes(&self.primitive_shells, |v| std::mem::size_of_val(v.as_ref()));
        output[8] = ready_bytes(&self.primitive_shell_pairs, |v| std::mem::size_of_val(v.as_slice()));
        output[9] = ready_bytes(&self.contraction_matrix, |m| m.get_data().len() * f64_bytes);
        output[10] = ready_bytes(&self.eri_data, |v| v.approx_memory_bytes());
        output[11] = ready_bytes(&self.picture_change, |v| (v.get_u_large().get_data().len() + v.get_u_small().get_data().len()) * f64_bytes);
        output[12] = ready_bytes(&self.symmetry, |v| v.approx_memory_bytes());
        output[13] = ready_bytes(&self.x2c_frozen_density, |v| v.bytes());
        Ok(())
    }

    pub fn free_input(&mut self) {
        self.qm_region.debug_double_free_warning("input");
        self.qm_region.free_cache();
    }

    // Shell pair handling
    pub fn set_shell_pairs(&mut self, overlap_threshold: Option<f64>) -> Result<(), AedelstenError> {
        if self.shell_pairs.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived("Shell pairs have already been derived. Please free the existing shell pairs before setting new ones.".to_string()));
        }
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let shell_pairs = basis::shell_pairs::derive_shell_pairs(qm_region.input.get_basis_shells(), overlap_threshold)?;
        self.shell_pairs = AedelstenCacheState::Ready(shell_pairs);
        // The polarization's kept field integrals were of the shell pairs before.
        if let Ok(state) = self.polarization.get_cache() {
            state.borrow_mut().invalidate_field_integrals();
        }
        Ok(())
    }

    pub fn free_shell_pairs(&mut self) {
        self.shell_pairs.debug_double_free_warning("shell pairs");
        self.shell_pairs.free_cache();
        self.eri_data.free_cache();
    }

    pub fn free_primitive_shells(&mut self) {
        self.primitive_shells.debug_double_free_warning("primitive shells");
        self.primitive_shells.free_cache();
    }

    pub fn free_contraction_matrix(&mut self) {
        self.contraction_matrix.debug_double_free_warning("contraction matrix");
        self.contraction_matrix.free_cache();
    }

    pub fn set_scratch(&mut self, (size, unit): (usize, arrays::scratch::ByteUnit)) -> Result<(), AedelstenError>
    {
        if self.scratch.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived(
                "Scratch is already allocated. Call free_scratch() before allocating a new one.".to_string()
            ));
        }
        let scratch = arrays::scratch::Scratch::new(size, unit)?;
        self.scratch = AedelstenCacheState::Ready(RefCell::new(scratch));
        Ok(())
    }

    pub fn resize_scratch(&mut self, (size, unit): (usize, arrays::scratch::ByteUnit)) -> Result<(), AedelstenError>
    {
        let scratch = self.scratch.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("scratch: {e} (set_scratch())")))?;
        scratch.borrow_mut().resize(size, unit)?;
        Ok(())
    }

    pub fn get_scratch_peak_bytes(&self) -> Result<usize, AedelstenError> {
        let scratch = self.scratch.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("scratch: {e} (set_scratch())")))?;
        Ok(scratch.borrow().peak_bytes())
    }

    pub fn free_scratch(&mut self) {
        self.scratch.debug_double_free_warning("scratch");
        self.scratch.free_cache();
    }

    pub fn set_communicator(&mut self, comm: &aedelsten_mpi::Communicator) -> aedelsten_mpi::Communicator {
        std::mem::replace(&mut self.comm, comm.clone())
    }

    pub fn communicator(&self) -> &aedelsten_mpi::Communicator {
        &self.comm
    }

    /// The scratch, or the error saying to set it.
    pub(crate) fn require_scratch(&self) -> Result<RefMut<'_, arrays::scratch::Scratch>, AedelstenError> {
        let scratch_cell = self.scratch.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
        })?;
        Ok(scratch_cell.borrow_mut())
    }

    /// The primitive shell pairs, or the error saying to set them.
    pub(crate) fn require_primitive_shell_pairs(&self) -> Result<&[basis::shell_pairs::ShellPair<basis::contraction::Uncontracted>], AedelstenError> {
        self.primitive_shell_pairs.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("primitive shell pairs are not resident in memory; call set_primitive_shell_pairs() first".to_string())
        }).map(|pairs| pairs.as_slice())
    }

    /// The primitive shells, or the error saying to set them.
    pub(crate) fn require_primitive_shells(&self) -> Result<&[basis::shell_types::BasisShell<basis::contraction::Uncontracted>], AedelstenError> {
        self.primitive_shells.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("primitive shells are not resident in memory; call set_primitive_shell_pairs() first".to_string())
        }).map(|shells| shells.as_ref())
    }

    pub fn set_primitive_shell_pairs(&mut self, overlap_threshold: Option<f64>) -> Result<(), AedelstenError> {
        if S::PRIMITIVE {
            return Ok(());
        }
        if self.primitive_shell_pairs.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived("Primitive shell pairs have already been derived. Please free the existing primitive shell pairs before setting new ones.".to_string()));
        }
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let (primitive_shells, contraction_matrix) = S::decontract(qm_region.input.get_basis_shells(), qm_region.input.get_basis_info())?;
        let primitive_shell_pairs = basis::shell_pairs::derive_shell_pairs(&primitive_shells, overlap_threshold)?;
        self.primitive_shells = AedelstenCacheState::Ready(primitive_shells);
        self.primitive_shell_pairs = AedelstenCacheState::Ready(primitive_shell_pairs);
        self.contraction_matrix = AedelstenCacheState::Ready(contraction_matrix);
        Ok(())
    }

    pub fn free_primitive_shell_pairs(&mut self) {
        if !S::PRIMITIVE {
            self.primitive_shell_pairs.debug_double_free_warning("primitive shell pairs");
            self.primitive_shell_pairs.free_cache();
        }
    }

    pub fn has_primitive(&self) -> bool {
        S::PRIMITIVE || self.primitive_shell_pairs.is_ready()
    }

    /// The primitive pairs: the decontracted ones, or the context's own for a primitive kind.
    pub(crate) fn primitive_pairs(&self) -> Result<&[basis::shell_pairs::ShellPair<basis::contraction::Uncontracted>], AedelstenError> {
        if S::PRIMITIVE {
            let pairs = self.shell_pairs.get_cache().map_err(|_| {
                AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
            })?;
            return Ok(S::as_uncontracted(pairs, &[]).expect("a primitive kind").0);
        }
        self.require_primitive_shell_pairs()
    }

    /// The primitive shells: the decontracted ones, or the context's own for a primitive kind.
    pub(crate) fn primitive_shells(&self) -> Result<&[basis::shell_types::BasisShell<basis::contraction::Uncontracted>], AedelstenError> {
        if S::PRIMITIVE {
            let shells = self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_basis_shells();
            return Ok(S::as_uncontracted(&[], shells).expect("a primitive kind").1);
        }
        self.require_primitive_shells()
    }
}
