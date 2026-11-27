// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Point-group symmetry of the QM region (see the `symmetry` crate): which centres are equivalent is decided here,
// from what the input says about each centre, and the resulting `Symmetry` is a cache like any other.
//
// Only the QM centres take part. The environment (external multipoles, polarizable sites, embedding AIMPs, fields) is
// the caller's responsibility. If it breaks the symmetry, a matrix that includes it is either refused
// (`Cleaning::Strict`) or averaged over the group (`Cleaning::Average`): the operator of the environment replaced by
// the average of its images under the group, which is the physics wanted when only the environment's far tail is not
// symmetric, and leaves the energy of a symmetric density unchanged.

use super::*;
use crate::context::cache::AedelstenCacheState;
use arrays::matrix::{MatrixTrait, MatrixView, MatrixViewMut};
use basis::charge_carriers::PointChargeTrait;
use std::collections::HashMap;
use crate::context::one_electron::{OneElectronAimpProjectionOperator, OneElectronOperator, OneElectronPlainOperator};
use std::cell::RefCell;
use symmetry::{AoBasis, Irrep, OperatorSymmetry, OrbitalLabel, PointGroupRequest, SymmetrisationReport, Symmetry, SymmetryBasisFunction, SymmetryCentre};

impl AedelstenContext {
    /// Finds the point group of the QM centres (or uses the requested one, see `PointGroupRequest`) and builds the AO
    /// representation of its operations, with the tolerances in `options`. Fails if a centre is more than
    /// `options.geometry_tolerance` Bohr from its symmetric position. Spherical and Cartesian shells alike.
    pub fn set_symmetry(&mut self, request: PointGroupRequest<'_>, options: SymmetryOptions) -> Result<(), AedelstenError> {
        dispatch!(self, set_symmetry(request, options)).inspect(|_| crate::citations::symmetry())
    }

    /// The tolerances symmetry was set with.
    pub fn get_symmetry_options(&self) -> Result<SymmetryOptions, AedelstenError> {
        dispatch!(self, get_symmetry_options())
    }

    /// The report of the last cleaning of a library output with symmetry set: `name` the output (the one-electron
    /// operator kinds' wire names, `oneint_overlap`, `oneint_kinetic`, `oneint_nuclear_attraction`, ...,
    /// `oneint_external_multipole`; `picture_change_` and the same for the scalar picture-change integrals;
    /// `x2c_one_electron_hamiltonian`, `x2c_u_large`, `x2c_u_small`; `spin_orbit_` and the operator kind for the three
    /// spin-orbit components; `oneint_moment` and `picture_change_oneint_moment` for a moment about a fixed point;
    /// `amfi_spin_same_orbit` and `amfi_spin_other_orbit` for the AMFI planes), in the primitive or the contracted
    /// basis. `None` if it was not cleaned since `set_symmetry`.
    pub fn get_symmetrisation_report(&self, name: &str, primitive: bool) -> Result<Option<SymmetrisationReport>, AedelstenError> {
        dispatch!(self, get_symmetrisation_report(name, primitive))
    }

    pub(crate) fn get_symmetry(&self) -> Result<&symmetry::Symmetry, AedelstenError> {
        dispatch!(self, get_symmetry())
    }

    /// The point group in use, e.g. "C2v".
    pub fn get_point_group_name(&self) -> Result<&str, AedelstenError> {
        dispatch!(self, get_point_group_name())
    }

    /// The irreps, in character-table order: every irrep index means a position in this list.
    pub fn get_irreps(&self) -> Result<&[Irrep], AedelstenError> {
        dispatch!(self, get_irreps())
    }

    /// The index of the irrep called `name` (case-insensitive), `None` if the group has none.
    pub fn get_irrep_index(&self, name: &str) -> Result<Option<usize>, AedelstenError> {
        dispatch!(self, get_irrep_index(name))
    }

    /// The index of the totally symmetric irrep (`get_irreps`).
    pub fn get_totally_symmetric_irrep(&self) -> Result<usize, AedelstenError> {
        dispatch!(self, get_totally_symmetric_irrep())
    }

    /// The irrep of the Cartesian monomial x^lx y^ly z^lz about the centre of mass (and so of that moment
    /// component's AO matrix), `None` if it is not a single one-dimensional irrep (see `symmetry::Symmetry`).
    pub fn get_cartesian_irrep(&self, lx: u32, ly: u32, lz: u32) -> Result<Option<usize>, AedelstenError> {
        dispatch!(self, get_cartesian_irrep(lx, ly, lz))
    }

    /// The group's standard x, y and z axes (rows) in the input's coordinates: the axes the irrep names refer to.
    pub fn get_symmetry_standard_axes(&self) -> Result<[[f64; 3]; 3], AedelstenError> {
        dispatch!(self, get_symmetry_standard_axes())
    }

    /// How many AOs (symmetry-adapted functions) each irrep has; a complete set of MOs comes in the same numbers.
    pub fn get_ao_irrep_counts(&self) -> Result<&[usize], AedelstenError> {
        dispatch!(self, get_ao_irrep_counts())
    }

    /// Frees the point group and its AO representation.
    pub fn free_symmetry(&mut self) {
        dispatch!(self, free_symmetry())
    }

    /// Cleans the column-major `(n_ao, n_ao)` totally symmetric AO `matrix` (a Fock matrix, a density, h1) in place:
    /// projects it onto the totally symmetric irrep, which is the average over the group's operations, and reports what
    /// was removed. `Cleaning::Strict` refuses, leaving the matrix alone, above the strict tolerance;
    /// `Cleaning::Average` always averages, and warns above the warning threshold (`SymmetryOptions`). In place, by
    /// orbits of shell-block pairs; scratch (a few tiles, whatever n_ao) from the context's buffer.
    pub fn symmetrise_matrix(&self, matrix: &mut [f64], cleaning: Cleaning) -> Result<SymmetrisationReport, AedelstenError> {
        dispatch!(self, symmetrise_matrix(matrix, cleaning))
    }

    /// Symmetrizes column-major `(n_ao, n_mo)` MO coefficients in place, `n_mo = labels_out.len()` (any number of MOs):
    /// each MO projected onto its dominant irrep, degenerate partners aligned onto the irrep's canonical components.
    /// Writes each MO's (irrep, component) label and returns the largest coefficient norm dropped from an MO. Not
    /// re-orthonormalized. Scratch from the context's buffer.
    pub fn symmetrise_orbitals(&self, orbitals: &mut [f64], labels_out: &mut [OrbitalLabel]) -> Result<f64, AedelstenError> {
        dispatch!(self, symmetrise_orbitals(orbitals, labels_out))
    }

    /// What cleaning `matrix` as `operator` would remove, `matrix` untouched: a diagnostic, also for matrices that are
    /// not totally symmetric (a moment component's irrep from `get_cartesian_irrep`, or `OperatorSymmetry::Auto`).
    pub fn analyse_matrix(&self, matrix: &[f64], operator: OperatorSymmetry) -> Result<SymmetrisationReport, AedelstenError> {
        dispatch!(self, analyse_matrix(matrix, operator))
    }

    /// The norm of each irrep's part of the `(n_ao, n_ao)` `matrix`, one per irrep in `norms_out` (for a degenerate
    /// irrep, its part as a whole): how an unknown operator transforms.
    pub fn irrep_norms(&self, matrix: &[f64], norms_out: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, irrep_norms(matrix, norms_out))
    }

    /// The number of the point group's operations; an error without symmetry.
    pub fn get_symmetry_operation_count(&self) -> Result<usize, AedelstenError> {
        Ok(self.get_symmetry()?.operation_count())
    }

    /// Operation `op` as a 3 x 3 matrix in the input's axes, `m[i][k]` (`symmetry::Symmetry::operation_matrix`): a
    /// displacement or gradient `v` of a centre goes to `sum_k m[i][k] v_k` at the centre's image
    /// (`get_symmetry_centre_images`). Orthogonal. For displacements that keep the symmetry (a geometry
    /// optimization's).
    pub fn get_symmetry_operation_matrix(&self, op: usize) -> Result<[[f64; 3]; 3], AedelstenError> {
        let symmetry = self.get_symmetry()?;
        if op >= symmetry.operation_count() {
            return Err(AedelstenError::StringError(format!("get_symmetry_operation_matrix: operation {op} of {}", symmetry.operation_count())));
        }
        Ok(symmetry.operation_matrix(op))
    }

    /// The number of QM centres the point group acts on (ghosts included; the gradient centres come first).
    pub fn get_n_symmetry_centres(&self) -> Result<usize, AedelstenError> {
        Ok(self.get_symmetry()?.centre_count())
    }

    /// Where operation `op` takes each QM centre: `images[a]` the centre (0-based, the input's order, ghosts included)
    /// it takes centre `a` to. `images` holds `get_n_symmetry_centres()` entries.
    pub fn get_symmetry_centre_images(&self, op: usize, images: &mut [usize]) -> Result<(), AedelstenError> {
        let symmetry = self.get_symmetry()?;
        if op >= symmetry.operation_count() {
            return Err(AedelstenError::StringError(format!("get_symmetry_centre_images: operation {op} of {}", symmetry.operation_count())));
        }
        let n = symmetry.centre_count();
        if images.len() != n {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: n, actual: images.len() });
        }
        for (a, image) in images.iter_mut().enumerate() {
            *image = symmetry.centre_image(a, op);
        }
        Ok(())
    }

    /// Operation `op` applied to MO coefficients (`(n_ao, n_mo)`, column-major, contracted AOs): `out` the coefficients
    /// of each R phi (`symmetry::Symmetry::apply_operation`).
    pub fn apply_symmetry_operation(&self, op: usize, coefficients: &[f64], n_mo: usize, out: &mut [f64]) -> Result<(), AedelstenError> {
        let symmetry = self.get_symmetry()?;
        let n = symmetry.n_ao();
        if coefficients.len() != n * n_mo || out.len() != n * n_mo {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: n * n_mo, actual: coefficients.len().min(out.len()) });
        }
        Ok(symmetry.apply_operation(op, &MatrixView::new(n, n_mo, coefficients), &mut MatrixViewMut::new(n, n_mo, out))?)
    }

    /// The character of irrep `irrep` at operation `op`.
    pub fn get_irrep_character(&self, irrep: usize, op: usize) -> Result<f64, AedelstenError> {
        let symmetry = self.get_symmetry()?;
        check_irrep_and_operation(symmetry, irrep, op, "get_irrep_character")?;
        Ok(symmetry.character(irrep, op))
    }

    /// Gamma(R)_jk of irrep `irrep` at operation `op` into `out` (d x d, column-major, d the irrep's dimension) for
    /// the aligned partner components; `false` when the irrep has none (`symmetry::Symmetry::irrep_matrix`).
    pub fn get_irrep_matrix(&self, irrep: usize, op: usize, out: &mut [f64]) -> Result<bool, AedelstenError> {
        let symmetry = self.get_symmetry()?;
        check_irrep_and_operation(symmetry, irrep, op, "get_irrep_matrix")?;
        let d = symmetry.irreps()[irrep].dimension;
        if out.len() != d * d {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: d * d, actual: out.len() });
        }
        Ok(symmetry.irrep_matrix(irrep, op, out))
    }
}

/// An irrep and an operation of the group, or the error naming what is out of range.
fn check_irrep_and_operation(symmetry: &symmetry::Symmetry, irrep: usize, op: usize, call: &str) -> Result<(), AedelstenError> {
    if irrep >= symmetry.irreps().len() || op >= symmetry.operation_count() {
        return Err(AedelstenError::StringError(format!(
            "{call}: irrep {irrep} of {}, operation {op} of {}",
            symmetry.irreps().len(),
            symmetry.operation_count()
        )));
    }
    Ok(())
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    pub fn set_symmetry(&mut self, request: PointGroupRequest<'_>, options: SymmetryOptions) -> Result<(), AedelstenError> {
        if self.symmetry.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived("Symmetry has already been set. Call free_symmetry() first.".to_string()));
        }
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(options.geometry_tolerance) || !positive(options.strict_tolerance) || options.average_warning.is_some_and(|w| !positive(w)) {
            return Err(AedelstenError::StringError(format!("set_symmetry: tolerances must be positive and finite, got {options:?}")));
        }
        let qm_region = self.qm_region.get_cache().map_err(|_| AedelstenError::CacheNotReady("input has been freed (free_input)".to_string()))?;
        let input = &qm_region.input;
        let info = input.get_basis_info();
        let ids = info.get_basis_function_ids();

        let atoms = input.get_qm_atoms().get_atoms();
        let mut signatures: Vec<CentreSignature> = atoms
            .iter()
            .map(|atom| CentreSignature {
                atomic_number: atom.get_atomic_number(),
                kind: atom.get_kind() as u8,
                core_aimp_charge: None,
                primitives: Vec::new(),
                harmonics: Vec::new(),
            })
            .collect();
        if let Some(core_regions) = input.get_core_regions() {
            for aimp in core_regions {
                signatures[aimp.get_idx()].core_aimp_charge = Some(aimp.get_charge().to_bits());
            }
        }
        // A primitive row's l is never negated (PrimitiveId's doc comment), and Cartesian shells are excluded above.
        for (id, primitive) in info.get_primitive_ids().iter().zip(info.get_primitives()) {
            signatures[id.centre - 1].primitives.push((id.l as i32, id.shell, primitive.exponent.to_bits(), primitive.coefficient.to_bits()));
        }
        for id in ids {
            let harmonics = &mut signatures[id.get_centre() - 1].harmonics;
            if harmonics.last() != Some(&(id.get_l(), id.get_shell())) {
                harmonics.push((id.get_l(), id.get_shell()));
            }
        }
        let mut kinds: HashMap<&CentreSignature, u32> = HashMap::new();
        let centres: Vec<SymmetryCentre> = atoms
            .iter()
            .zip(&signatures)
            .map(|(atom, signature)| {
                let next = kinds.len() as u32;
                SymmetryCentre { kind: *kinds.entry(signature).or_insert(next), mass: atom.get_mass(), position: atom.get_position() }
            })
            .collect();
        // p shells are always Cartesian (l = -1) and are the real p harmonics all the same (BasisFunctionId's encoding:
        // x is -1, y is 0, z is 1; libmsym's real p_m: x +1, y -1, z 0). From d up a Cartesian shell (l = -lqn) is its
        // own representation: its components by position in the shell, the library's order.
        let mut basis: Vec<SymmetryBasisFunction> = Vec::with_capacity(ids.len());
        for (i, id) in ids.iter().enumerate() {
            let function = match (id.get_l(), id.get_m()) {
                (-1, m) => SymmetryBasisFunction { centre: id.get_centre() - 1, radial: id.get_shell(), l: 1, m: [1, -1, 0][(m + 1) as usize], cartesian: false },
                (l, _) if l < -1 => {
                    let same_block = |other: &basis::BasisFunctionId| (other.get_centre(), other.get_shell(), other.get_l()) == (id.get_centre(), id.get_shell(), id.get_l());
                    let position = ids[..i].iter().rev().take_while(|other| same_block(other)).count();
                    SymmetryBasisFunction { centre: id.get_centre() - 1, radial: id.get_shell(), l: (-l) as u32, m: position as i32, cartesian: true }
                }
                (l, m) => SymmetryBasisFunction { centre: id.get_centre() - 1, radial: id.get_shell(), l: l as u32, m, cartesian: false },
            };
            basis.push(function);
        }

        // The primitive basis: the library's own decontraction (sibling segmented shells folded, shared exponents
        // once), the basis of the primitive integrals and of X2C; for a primitive context, the basis itself.
        let primitive_basis: Vec<SymmetryBasisFunction> = if S::PRIMITIVE {
            basis.clone()
        } else {
            let (shells, _) = S::decontract(input.get_basis_shells(), info)?;
            primitive_basis_functions(&shells)?
        };
        let symmetry = Symmetry::new(&centres, &basis, Some(&primitive_basis), request, options.geometry_tolerance)?;

        // The basis may keep apart centres the geometry makes equivalent (equivalent atoms given different basis sets
        // or harmonic conventions, by intent or by accident): that lowers the group, and is said.
        let mut geometric_kinds: HashMap<(u8, u8, Option<u64>), u32> = HashMap::new();
        let geometric: Vec<SymmetryCentre> = centres
            .iter()
            .zip(&signatures)
            .map(|(centre, signature)| {
                let next = geometric_kinds.len() as u32;
                let kind = *geometric_kinds.entry((signature.atomic_number, signature.kind, signature.core_aimp_charge)).or_insert(next);
                SymmetryCentre { kind, ..*centre }
            })
            .collect();
        let full_kinds: Vec<u32> = centres.iter().map(|centre| centre.kind).collect();
        let by_geometry = symmetry::geometric_symmetry(&geometric, &full_kinds)?;
        if !by_geometry.mismatched.is_empty() && self.comm.is_root() {
            let label = |i: usize| input.get_centre_label(i).map_or(format!("centre {}", i + 1), |label| label.to_string());
            let pairs: Vec<String> = by_geometry.mismatched.iter().map(|&(a, b)| format!("{} and {}", label(a), label(b))).collect();
            eprintln!(
                "Warning (set_symmetry): the geometry is {}, but the basis lowers the symmetry: {} are equivalent by position \
                 but have different basis functions or harmonic conventions (Cartesian/spherical). Give symmetry-equivalent \
                 centres the same basis to keep the geometry's group",
                by_geometry.point_group,
                pairs.join(", ")
            );
        }
        self.symmetry = AedelstenCacheState::Ready(SymmetryState { symmetry, options, reports: RefCell::new(Vec::new()) });
        Ok(())
    }

    // Internal: `Symmetry` works on `arrays` types and stays inside the crate; the public queries below return plain
    // data.
    pub(crate) fn get_symmetry(&self) -> Result<&Symmetry, AedelstenError> {
        Ok(&self.get_symmetry_state()?.symmetry)
    }

    fn get_symmetry_state(&self) -> Result<&SymmetryState, AedelstenError> {
        self.symmetry.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("symmetry: {e}; call set_symmetry() first")))
    }

    pub fn get_symmetry_options(&self) -> Result<SymmetryOptions, AedelstenError> {
        Ok(self.get_symmetry_state()?.options)
    }

    pub fn get_point_group_name(&self) -> Result<&str, AedelstenError> {
        Ok(self.get_symmetry()?.point_group())
    }

    pub fn get_irreps(&self) -> Result<&[Irrep], AedelstenError> {
        Ok(self.get_symmetry()?.irreps())
    }

    pub fn get_irrep_index(&self, name: &str) -> Result<Option<usize>, AedelstenError> {
        Ok(self.get_symmetry()?.irrep_index(name))
    }

    pub fn get_totally_symmetric_irrep(&self) -> Result<usize, AedelstenError> {
        Ok(self.get_symmetry()?.totally_symmetric_irrep())
    }

    pub fn get_cartesian_irrep(&self, lx: u32, ly: u32, lz: u32) -> Result<Option<usize>, AedelstenError> {
        Ok(self.get_symmetry()?.cartesian_irrep(lx, ly, lz))
    }

    pub fn get_symmetry_standard_axes(&self) -> Result<[[f64; 3]; 3], AedelstenError> {
        Ok(self.get_symmetry()?.standard_axes())
    }

    pub fn get_ao_irrep_counts(&self) -> Result<&[usize], AedelstenError> {
        Ok(self.get_symmetry()?.ao_irrep_counts())
    }

    pub fn free_symmetry(&mut self) {
        self.symmetry.debug_double_free_warning("symmetry");
        self.symmetry.free_cache();
    }

    // The symmetry and an arena over the main scratch, for the per-call methods' scratch.
    fn with_symmetry_scratch<R>(&self, work: impl FnOnce(&Symmetry, &arrays::scratch::Arena<'_>) -> Result<R, AedelstenError>) -> Result<R, AedelstenError> {
        let symmetry = self.get_symmetry()?;
        let scratch_cell = self.scratch.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
        })?;
        let mut scratch = scratch_cell.borrow_mut();
        let arena = scratch.get_arena();
        work(symmetry, &arena)
    }

    fn check_ao_matrix(symmetry: &Symmetry, len: usize) -> Result<usize, AedelstenError> {
        let n = symmetry.n_ao();
        if len != n * n {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: n * n, actual: len });
        }
        Ok(n)
    }

    pub fn symmetrise_matrix(&self, matrix: &mut [f64], cleaning: Cleaning) -> Result<SymmetrisationReport, AedelstenError> {
        let options = self.get_symmetry_state()?.options;
        self.with_symmetry_scratch(|symmetry, arena| {
            let n = Self::check_ao_matrix(symmetry, matrix.len())?;
            Ok(clean(symmetry, AoBasis::Contracted, &mut MatrixViewMut::new(n, n, matrix), cleaning, options, "symmetrise_matrix", arena)?)
        })
    }

    pub fn get_symmetrisation_report(&self, name: &str, primitive: bool) -> Result<Option<SymmetrisationReport>, AedelstenError> {
        let basis = if primitive { AoBasis::Primitive } else { AoBasis::Contracted };
        let reports = self.get_symmetry_state()?.reports.borrow();
        Ok(reports.iter().find(|(n, b, _)| *n == name && *b == basis).map(|(_, _, report)| *report))
    }

    // Cleans the planes of a Cartesian tensor operator the library produced (`(n, n, components)` in `output`), all
    // components together, as `clean_library_output` does a matrix. Nothing without symmetry, and nothing (no report)
    // for a polar tensor taken about a point the operations move (`origin`): its components mix with other orders
    // there.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn clean_library_tensor(
        &self,
        output: &mut [f64],
        primitive: bool,
        order: usize,
        kind: symmetry::TensorKind,
        origin: Option<[f64; 3]>,
        cleaning: Cleaning,
        name: &'static str,
    ) -> Result<(), AedelstenError> {
        if !self.symmetry.is_ready() {
            return Ok(());
        }
        let state = self.get_symmetry_state()?;
        if let Some(origin) = origin
            && !state.symmetry.is_fixed_point(origin)?
        {
            return Ok(());
        }
        let basis = if primitive { AoBasis::Primitive } else { AoBasis::Contracted };
        let options = state.options;
        let report = self.with_symmetry_scratch(|symmetry, arena| {
            let n = symmetry.n_ao_of(basis).expect("set_symmetry always builds the primitive basis");
            let n_comp = (order + 1) * (order + 2) / 2;
            if output.len() != n * n * n_comp {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n * n * n_comp, actual: output.len() });
            }
            let scale = output.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            let mut planes = arrays::array3d::Array3DViewMut::new(n, n, n_comp, output);
            let result = match cleaning {
                Cleaning::Strict => symmetry.symmetrise_tensor(basis, &mut planes, order, kind, options.strict_tolerance * scale, arena),
                Cleaning::Average => symmetry.average_tensor(basis, &mut planes, order, kind, arena).inspect(|report| warn_if_averaged_beyond(report, options, name)),
            };
            result.map_err(|e| AedelstenError::StringError(format!("{name} is not symmetric under {}: {e}", symmetry.point_group())))
        })?;
        let mut reports = state.reports.borrow_mut();
        match reports.iter_mut().find(|(n, b, _)| *n == name && *b == basis) {
            Some(entry) => entry.2 = report,
            None => reports.push((name, basis, report)),
        }
        Ok(())
    }

    // Fails, naming `name`, if `matrix` (an input the library relies on being symmetric, such as X2C's S, T, V, pVp) is
    // not totally symmetric to the strict tolerance (relative to its largest element). Nothing without symmetry; reads
    // only.
    pub(crate) fn require_symmetric_input(&self, matrix: &[f64], primitive: bool, name: &str, advice: &str) -> Result<(), AedelstenError> {
        if !self.symmetry.is_ready() {
            return Ok(());
        }
        let options = self.get_symmetry_state()?.options;
        let basis = if primitive { AoBasis::Primitive } else { AoBasis::Contracted };
        self.with_symmetry_scratch(|symmetry, arena| {
            let n = symmetry.n_ao_of(basis).expect("set_symmetry always builds the primitive basis");
            if matrix.len() != n * n {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n * n, actual: matrix.len() });
            }
            let report = symmetry.analyse_matrix(basis, &MatrixView::new(n, n, matrix), OperatorSymmetry::TotallySymmetric, arena)?;
            let scale = matrix.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            if report.max_abs_removed > options.strict_tolerance * scale {
                return Err(AedelstenError::StringError(format!(
                    "{name} is not symmetric under {} (an element would change by {:.3e}, tolerance {:.1e} relative): {advice}",
                    symmetry.point_group(),
                    report.max_abs_removed,
                    options.strict_tolerance
                )));
            }
            Ok(())
        })
    }

    // Cleans a matrix the library produced (an integral matrix, the X2C Hamiltonian) when symmetry is set: `cleaning`
    // as its kind calls for, the report kept under `name`. Nothing without symmetry. A strict refusal is an error: an
    // exact operator that is not symmetric means a wrong geometry or basis, or a bug.
    pub(crate) fn clean_library_output(&self, matrix: &mut [f64], primitive: bool, cleaning: Cleaning, name: &'static str) -> Result<(), AedelstenError> {
        if !self.symmetry.is_ready() {
            return Ok(());
        }
        let state = self.get_symmetry_state()?;
        let basis = if primitive { AoBasis::Primitive } else { AoBasis::Contracted };
        let report = self.with_symmetry_scratch(|symmetry, arena| {
            let n = symmetry.n_ao_of(basis).expect("set_symmetry always builds the primitive basis");
            if matrix.len() != n * n {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n * n, actual: matrix.len() });
            }
            clean(symmetry, basis, &mut MatrixViewMut::new(n, n, matrix), cleaning, state.options, name, arena)
                .map_err(|e| AedelstenError::StringError(format!("{name} is not symmetric under {}: {e}", symmetry.point_group())))
        })?;
        let mut reports = state.reports.borrow_mut();
        match reports.iter_mut().find(|(n, b, _)| *n == name && *b == basis) {
            Some(entry) => entry.2 = report,
            None => reports.push((name, basis, report)),
        }
        Ok(())
    }

    pub fn analyse_matrix(&self, matrix: &[f64], operator: OperatorSymmetry) -> Result<SymmetrisationReport, AedelstenError> {
        self.with_symmetry_scratch(|symmetry, arena| {
            let n = Self::check_ao_matrix(symmetry, matrix.len())?;
            Ok(symmetry.analyse_matrix(AoBasis::Contracted, &MatrixView::new(n, n, matrix), operator, arena)?)
        })
    }

    pub fn irrep_norms(&self, matrix: &[f64], norms_out: &mut [f64]) -> Result<(), AedelstenError> {
        self.with_symmetry_scratch(|symmetry, arena| {
            let n = Self::check_ao_matrix(symmetry, matrix.len())?;
            Ok(symmetry.irrep_norms(AoBasis::Contracted, &MatrixView::new(n, n, matrix), norms_out, arena)?)
        })
    }

    pub fn symmetrise_orbitals(&self, orbitals: &mut [f64], labels_out: &mut [OrbitalLabel]) -> Result<f64, AedelstenError> {
        self.with_symmetry_scratch(|symmetry, arena| {
            let (n, n_mo) = (symmetry.n_ao(), labels_out.len());
            if orbitals.len() != n * n_mo {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n * n_mo, actual: orbitals.len() });
            }
            Ok(symmetry.symmetrise_orbitals(&mut MatrixViewMut::new(n, n_mo, orbitals), labels_out, arena)?)
        })
    }
}

/// The tolerances of symmetry, set once with `set_symmetry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryOptions {
    /// Bohr: `set_symmetry` fails if a centre is further than this from its symmetric position.
    pub geometry_tolerance: f64,
    /// `Cleaning::Strict` refuses a matrix whose cleaning would change an element by more than this, relative to the
    /// matrix's largest element (absolute when that is below 1): rounding scales with the size of the elements (an X2C
    /// Hamiltonian of a heavy atom has elements of 1e4).
    pub strict_tolerance: f64,
    /// `Cleaning::Average` warns (stderr, rank 0) when it changes an element by more than this; `None`: never.
    pub average_warning: Option<f64>,
}

/// Default of `SymmetryOptions::strict_tolerance`: well above rounding in integrals, well below anything physical.
pub const DEFAULT_STRICT_TOLERANCE: f64 = 1e-10;

impl Default for SymmetryOptions {
    fn default() -> Self {
        Self { geometry_tolerance: symmetry::DEFAULT_GEOMETRY_TOLERANCE, strict_tolerance: DEFAULT_STRICT_TOLERANCE, average_warning: None }
    }
}

/// What `symmetrise_matrix` does with what is not totally symmetric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cleaning {
    /// The matrix should be symmetric already (S, h1 and F without an environment): rounding is removed, anything above
    /// `SymmetryOptions::strict_tolerance` is refused and the matrix left alone.
    Strict,
    /// The matrix is averaged over the group, whatever that removes (an embedding whose tail is not symmetric); a
    /// warning above `SymmetryOptions::average_warning`.
    Average,
}

// The cache: the group and its representation, the tolerances it was set with, and the report of the last cleaning of
// each library output, per basis (`get_symmetrisation_report`).
#[derive(Debug)]
pub(crate) struct SymmetryState {
    pub(crate) symmetry: Symmetry,
    pub(crate) options: SymmetryOptions,
    reports: RefCell<Vec<(&'static str, AoBasis, SymmetrisationReport)>>,
}

impl SymmetryState {
    /// The memory held, in bytes, roughly.
    pub(crate) fn approx_memory_bytes(&self) -> usize {
        self.symmetry.approx_memory_bytes()
    }
}

// Everything that has to agree for two centres to be images of each other: element, kind (real, ghost, orthogonality
// ghost), core AIMP charge, and the basis on the centre, primitive by primitive, with each shell's harmonic convention.
// Floats by their bits: equivalent centres get their data from the same basis-set and AIMP entries, so it is identical,
// not just close. Where in the input a centre was given does not matter.
#[derive(PartialEq, Eq, Hash)]
struct CentreSignature {
    atomic_number: u8,
    kind: u8,
    core_aimp_charge: Option<u64>,
    // (l, shell ordinal, exponent, coefficient) per primitive row, in order.
    primitives: Vec<(i32, usize, u64, u64)>,
    // (l as in BasisFunctionId: negative for a Cartesian shell, shell ordinal) per contracted shell, in order.
    harmonics: Vec<(i32, usize)>,
}

// The warning `Cleaning::Average` gives when it removed more than the threshold: the environment is less symmetric, as
// seen from the QM region, than the caller expected.
fn warn_if_averaged_beyond(report: &SymmetrisationReport, options: SymmetryOptions, what: &str) {
    if let Some(threshold) = options.average_warning
        && report.max_abs_removed > threshold
        && aedelsten_mpi::rank() == 0
    {
        eprintln!(
            "Warning ({what}): averaging over the point group changed an element by {:.3e} (warning threshold {threshold:.1e}; \
             removed norm {:.3e} of {:.3e}): the environment is not symmetric as seen from the QM region",
            report.max_abs_removed, report.removed_norm, report.norm
        );
    }
}

// A totally symmetric matrix cleaned as `cleaning` says: strict (the tolerance relative to the largest element, refused
// above it) or averaged (a warning above the threshold).
fn clean(
    symmetry: &Symmetry,
    basis: AoBasis,
    matrix: &mut MatrixViewMut<'_, f64>,
    cleaning: Cleaning,
    options: SymmetryOptions,
    what: &str,
    arena: &arrays::scratch::Arena<'_>,
) -> Result<SymmetrisationReport, symmetry::SymmetryError> {
    match cleaning {
        Cleaning::Strict => {
            let scale = matrix.get_data().iter().fold(1.0f64, |m, x| m.max(x.abs()));
            symmetry.symmetrise_matrix(basis, matrix, OperatorSymmetry::TotallySymmetric, options.strict_tolerance * scale, arena)
        }
        Cleaning::Average => {
            let report = symmetry.average_matrix(basis, matrix, OperatorSymmetry::TotallySymmetric, arena)?;
            warn_if_averaged_beyond(&report, options, what);
            Ok(report)
        }
    }
}

// The primitive basis as symmetry AOs: one block per primitive of a shell (the primitive's ordinal in its (centre, l)
// as the radial index; equivalent centres have the same exponents in the same, descending, order), components as in the
// contracted basis (p Cartesian x, y, z; d and up spherical, ascending m, or Cartesian, by position).
fn primitive_basis_functions(shells: &[basis::shell_types::BasisShell<basis::contraction::Uncontracted>]) -> Result<Vec<SymmetryBasisFunction>, AedelstenError> {
    let mut functions = Vec::new();
    let mut ordinals: HashMap<(usize, usize), usize> = HashMap::new();
    for shell in shells {
        if shell.get_basis_index() != functions.len() {
            return Err(AedelstenError::StringError(format!(
                "internal: the decontracted shells are not in AO order (a shell at {} after {} AOs)",
                shell.get_basis_index(),
                functions.len()
            )));
        }
        let (centre, l) = (shell.get_centre_index(), shell.get_lqn());
        let cartesian = l > 1 && shell.get_harmonic_type() == basis::shell_types::HarmonicType::Cartesian;
        let ms: Vec<i32> = if l == 1 {
            vec![1, -1, 0]
        } else if cartesian {
            (0..((l + 1) * (l + 2) / 2) as i32).collect()
        } else {
            (-(l as i32)..=l as i32).collect()
        };
        for _ in 0..shell.get_n_cont() {
            let ordinal = ordinals.entry((centre, l)).or_insert(0);
            *ordinal += 1;
            functions.extend(ms.iter().map(|&m| SymmetryBasisFunction { centre, radial: *ordinal, l: l as u32, m, cartesian }));
        }
    }
    Ok(functions)
}

/// How the library cleans a one-electron operator's AO matrices when symmetry is set: the names the reports are kept
/// under (its plain integrals, its scalar picture-change integrals, its spin-orbit ones), and strict (an operator of
/// the QM region) or averaged (the environment). `None`: not a plain operator (the moments are handled as tensors).
pub(crate) struct OneElectronCleaning {
    pub(crate) name: &'static str,
    pub(crate) picture_change: &'static str,
    pub(crate) spin_orbit: &'static str,
    pub(crate) cleaning: Cleaning,
}

/// How the library cleans `operator`'s matrices (`OneElectronCleaning`); `None` for the moments and other component
/// operators, cleaned as tensors.
pub(crate) fn one_electron_cleaning(operator: &OneElectronOperator<'_>) -> Option<OneElectronCleaning> {
    use Cleaning::{Average, Strict};
    macro_rules! names {
        ($name:literal, $cleaning:expr) => {
            OneElectronCleaning {
                name: $name,
                picture_change: concat!("picture_change_", $name),
                spin_orbit: concat!("spin_orbit_", $name),
                cleaning: $cleaning,
            }
        };
    }
    Some(match operator {
        OneElectronOperator::MatrixOutput(op) => match op {
            OneElectronPlainOperator::Overlap => names!("oneint_overlap", Strict),
            OneElectronPlainOperator::Kinetic => names!("oneint_kinetic", Strict),
            OneElectronPlainOperator::NuclearAttraction(_) => names!("oneint_nuclear_attraction", Strict),
            OneElectronPlainOperator::ExternalMultipole(..) => names!("oneint_external_multipole", Average),
            OneElectronPlainOperator::CoreAimpPointCharge(_) => names!("oneint_core_aimp_point_charge", Strict),
            OneElectronPlainOperator::CoreAimpScreenedCoulomb(_) => names!("oneint_core_aimp_screened_coulomb", Strict),
            OneElectronPlainOperator::CoreAimpCoulomb(_) => names!("oneint_core_aimp_coulomb", Strict),
            OneElectronPlainOperator::EmbeddingAimpPointCharge(_) => names!("oneint_embedding_aimp_point_charge", Average),
            OneElectronPlainOperator::EmbeddingAimpScreenedCoulomb(_) => names!("oneint_embedding_aimp_screened_coulomb", Average),
            OneElectronPlainOperator::EmbeddingAimpCoulomb(_) => names!("oneint_embedding_aimp_coulomb", Average),
            OneElectronPlainOperator::SapScreening(SapFit::HelFem, _) => names!("oneint_sap_helfem", Strict),
            OneElectronPlainOperator::SapScreening(SapFit::Grasp, _) => names!("oneint_sap_grasp", Strict),
        },
        OneElectronOperator::AimpSpecial(op) => match op {
            OneElectronAimpProjectionOperator::CoreAimpProjection(..) => names!("oneint_core_aimp_projection", Strict),
            OneElectronAimpProjectionOperator::CoreAimpSpectrallyResolvedExchange(..) => names!("oneint_core_aimp_spectrally_resolved_exchange", Strict),
            OneElectronAimpProjectionOperator::CoreAimpExchange(..) => names!("oneint_core_aimp_exchange", Strict),
            OneElectronAimpProjectionOperator::EmbeddingAimpProjection(..) => names!("oneint_embedding_aimp_projection", Average),
            OneElectronAimpProjectionOperator::EmbeddingAimpSpectrallyResolvedExchange(..) => {
                names!("oneint_embedding_aimp_spectrally_resolved_exchange", Average)
            }
            OneElectronAimpProjectionOperator::EmbeddingAimpExchange(..) => names!("oneint_embedding_aimp_exchange", Average),
        },
        OneElectronOperator::Array3DOutput(_) => return None,
    })
}
