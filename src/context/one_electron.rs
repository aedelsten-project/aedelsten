// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// One-electron integrals: the operators (overlap, kinetic, moments, nuclear attraction, and the AIMP and external-
// multipole operators built in embedding.rs), their integrals over the contracted or primitive basis, their
// gradients, and contracting primitive-basis matrices to the contracted basis.

use super::*;
use std::cell::RefCell;
use arrays::Array3DTraitMut;
use basis::charge_carriers::NuclearPotential;
use basis::charge_carriers::ExternalMultipoles;
use oneint::OneElectronIntegrals;
use oneint::OneElectronGradientContractions;
use oneint::OneElectronPictureChangeIntegrals;
use aimp::AimpCoulombIntegrals;
use aimp::AimpExchangeIntegrals;
use aimp::AimpCoulombGradientContractions;
use aimp::AimpExchangeGradientContractions;
use crate::error::AedelstenError;
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;
use crate::wire_enum::c_int_enum;

impl AedelstenContext {
    /// Contracts `primitive`, an `(n_prim, n_prim)` matrix over the primitive basis (`n_prim = get_n_prim_unique()`),
    /// to the contracted basis, `C^T M C` into `output` (`n_cont x n_cont`, OVERWRITTEN); a copy for a primitive
    /// context. Needs `set_primitive_shell_pairs`.
    pub fn contract_one_electron_matrix(&self, primitive: &[f64], output: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, contract_one_electron_matrix(primitive, output))
    }

    /// `contract_one_electron_matrix` for each of the `n_planes` planes of `primitive` (`(n_prim, n_prim, n_planes)`).
    pub fn contract_one_electron_array3d(&self, primitive: &[f64], n_planes: usize, output: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, contract_one_electron_array3d(primitive, n_planes, output))
    }

    /// The overlap operator.
    pub fn get_overlap_operator(&self) -> OneElectronOperator<'_> {
        dispatch!(self, get_overlap_operator())
    }

    /// The kinetic-energy operator, `-1/2 nabla^2`.
    pub fn get_kinetic_operator(&self) -> OneElectronOperator<'_> {
        dispatch!(self, get_kinetic_operator())
    }

    /// The Cartesian multipole moments about `origin` (its order sets the components).
    pub fn get_moment_operator(&self, origin: oneint::MomentOrigin) -> OneElectronOperator<'_> {
        dispatch!(self, get_moment_operator(origin))
    }

    /// The attraction of the QM region's nuclei (point or Gaussian, `set_finite_nuclear_model`); zero integrals when
    /// the region has only core AIMPs.
    pub fn get_nuclear_attraction_operator(&self) -> OneElectronOperator<'_> {
        dispatch!(self, get_nuclear_attraction_operator())
    }

    /// The SAP screening `W = Sum_A W_A`: each real atom's electrons as the Gaussian charges of `fit` (S. Lehtola,
    /// J. Chem. Theory Comput. 15, 1593 (2019); S. Lehtola, L. Visscher, E. Engel, J. Chem. Phys. 152, 144105 (2020)),
    /// a repulsive potential that, added to a one-electron Hamiltonian, makes each atom neutral. Independent of the
    /// nuclear model; never picture-changed (the picture-change integrals refuse it). An error if a centre has a core
    /// AIMP (the fits include the core electrons), naming it.
    pub fn get_sap_screening_operator(&self, fit: SapFit) -> Result<OneElectronOperator<'_>, AedelstenError> {
        dispatch!(self, get_sap_screening_operator(fit))
    }

    /// One-electron integrals over the contracted basis, or (`primitive`, after `set_primitive_shell_pairs`) over the
    /// primitive basis.
    pub fn compute_one_electron_integrals(
        &self,
        output: &mut [f64],
        operator: OneElectronOperator<'_>,
        primitive: bool,
    ) -> Result<(), AedelstenError> {
        let sap = operator.is_sap_screening();
        dispatch!(self, compute_one_electron_integrals(output, operator, primitive)).inspect(|_| if sap { crate::citations::sap() })
    }

    /// The gradient of one one-electron operator contracted with a density, for all centres at once:
    /// `gradient[3 * X + c] = sum_mn D_mn dO_mn/dX_c`, into a `(3, get_n_gradient_centres())` column-major
    /// matrix that is overwritten. `density` is `n_cont x n_cont` (only its symmetric part matters; use the
    /// energy-weighted density for the overlap). It includes the basis-function terms and, for operators whose
    /// sources are centres (nuclei, core AIMPs), the source terms; external charges/multipoles and embedding AIMPs
    /// do not move and give no forces here. The consumer sums the operators it needs. Errors for the moment
    /// (property) operator, which has no gradient.
    ///
    /// `active` freezes centres: one flag per gradient centre (`None` = all active). A frozen centre's column is
    /// not computed and is returned as zero, and the work that only feeds frozen columns is skipped -- pairs whose
    /// two centres are both frozen, and the source terms of frozen nuclei and core AIMPs. The mask is per centre only:
    /// to freeze single coordinates, keep their centre active and zero those entries of `gradient` afterwards.
    pub fn compute_one_electron_gradient(
        &self,
        gradient: &mut [f64],
        density: &[f64],
        operator: OneElectronOperator<'_>,
        active: Option<&[bool]>,
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_one_electron_gradient(gradient, density, operator, active)).inspect(|_| crate::citations::gradients())
    }
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    pub fn get_overlap_operator(&self) -> OneElectronOperator<'_> {
        OneElectronOperator::overlap()
    }

    pub fn get_kinetic_operator(&self) -> OneElectronOperator<'_> {
        OneElectronOperator::kinetic()
    }

    pub fn get_moment_operator(&self, origin: oneint::MomentOrigin) -> OneElectronOperator<'_> {
        OneElectronOperator::moment(origin)
    }

    pub fn get_nuclear_attraction_operator(&self) -> OneElectronOperator<'_> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        OneElectronOperator::nuclear_attraction(qm_region.input.get_nuclear_potential())
    }

    pub fn get_sap_screening_operator(&self, fit: SapFit) -> Result<OneElectronOperator<'_>, AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let charges = qm_region.input.derive_sap_charges(fit)?;
        Ok(OneElectronOperator::MatrixOutput(OneElectronPlainOperator::SapScreening(fit, charges)))
    }

    pub fn compute_one_electron_gradient(
        &self,
        gradient: &mut [f64],
        density: &[f64],
        operator: OneElectronOperator<'_>,
        active: Option<&[bool]>,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let n_centres = self.get_n_gradient_centres();
        if density.len() != n * n {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: n * n, actual: density.len() });
        }
        if gradient.len() != 3 * n_centres {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * n_centres, actual: gradient.len() });
        }
        if let Some(mask) = active {
            if mask.len() != n_centres {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n_centres, actual: mask.len() });
            }
        }
        let mut wrapped_gradient = arrays::matrix::MatrixViewMut::new(3, n_centres, gradient);
        let wrapped_density = arrays::matrix::MatrixView::new(n, n, density);
        match operator {
            OneElectronOperator::MatrixOutput(plain_operator) => {
                let shell_pairs = self.shell_pairs.get_cache().map_err(|_| {
                    AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
                })?;
                let scratch_cell = self.scratch.get_cache().map_err(|_| {
                    AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
                })?;
                let mut scratch = scratch_cell.borrow_mut();
                compute_one_electron_gradient(
                    shell_pairs, plain_operator, &mut wrapped_gradient, &wrapped_density, active, &mut scratch, &self.comm,
                )?;
            }
            OneElectronOperator::Array3DOutput(_) => {
                return Err(AedelstenError::StringError("the moment (property) operator has no gradient".to_string()));
            }
            OneElectronOperator::AimpSpecial(op) => {
                let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
                let basis_shells = qm_region.input.get_basis_shells();
                let scratch_cell = self.scratch.get_cache().map_err(|_| {
                    AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
                })?;
                let mut scratch = scratch_cell.borrow_mut();
                compute_one_electron_aimp_special_gradient(basis_shells, op, &mut wrapped_gradient, &wrapped_density, active, &mut scratch, &self.comm)?;
            }
        }
        Ok(())
    }

    /// `C^T M C` of the primitive-basis matrix `primitive` into `output`, by the contraction matrix of
    /// `set_primitive_shell_pairs`.
    pub(crate) fn contract_primitive_matrix(&self, primitive: &[f64], output: &mut [f64]) -> Result<(), AedelstenError> {
        let c = self.contraction_matrix.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("contraction matrix is not resident in memory; call set_primitive_shell_pairs() first".to_string())
        })?;
        let n_prim = c.rows();
        let n_cont = c.cols();
        check_matrix_dimensions(n_prim, primitive)?;
        check_matrix_dimensions(n_cont, output)?;
        let primitive_matrix = arrays::matrix::MatrixView::new(n_prim, n_prim, primitive);
        let mut wrapped_output = arrays::matrix::MatrixViewMut::new(n_cont, n_cont, output);
        let mut product_scratch = arrays::matrix::Matrix::<f64>::new(n_prim, n_cont);
        primitive_matrix.congruence_transform_prealloc(c, &mut product_scratch, &mut wrapped_output);
        Ok(())
    }

    /// `contract_primitive_matrix` plane by plane, for `n_planes` planes.
    pub(crate) fn contract_primitive_array3d(&self, primitive: &[f64], n_planes: usize, output: &mut [f64]) -> Result<(), AedelstenError> {
        let c = self.contraction_matrix.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("contraction matrix is not resident in memory; call set_primitive_shell_pairs() first".to_string())
        })?;
        let n_prim = c.rows();
        let n_cont = c.cols();
        check_array3d_dimensions(n_prim, n_planes, primitive)?;
        check_array3d_dimensions(n_cont, n_planes, output)?;
        let primitive_view = arrays::array3d::Array3DView::new(n_prim, n_prim, n_planes, primitive);
        let mut output_view = arrays::array3d::Array3DViewMut::new(n_cont, n_cont, n_planes, output);
        let mut product_scratch = arrays::matrix::Matrix::<f64>::new(n_prim, n_cont);
        for k in 0..n_planes {
            let primitive_plane = primitive_view.view_sub_matrix(k);
            let mut contracted_plane = output_view.view_sub_matrix_mut(k);
            primitive_plane.congruence_transform_prealloc(c, &mut product_scratch, &mut contracted_plane);
        }
        Ok(())
    }

    pub fn contract_one_electron_matrix(&self, primitive: &[f64], output: &mut [f64]) -> Result<(), AedelstenError> {
        if S::PRIMITIVE {
            check_matrix_dimensions(self.get_n_cont(), primitive)?;
            check_matrix_dimensions(self.get_n_cont(), output)?;
            output.copy_from_slice(primitive);
            return Ok(());
        }
        self.contract_primitive_matrix(primitive, output)
    }

    pub fn contract_one_electron_array3d(&self, primitive: &[f64], n_planes: usize, output: &mut [f64]) -> Result<(), AedelstenError> {
        if S::PRIMITIVE {
            check_array3d_dimensions(self.get_n_cont(), n_planes, primitive)?;
            check_array3d_dimensions(self.get_n_cont(), n_planes, output)?;
            output.copy_from_slice(primitive);
            return Ok(());
        }
        self.contract_primitive_array3d(primitive, n_planes, output)
    }

    /// One-electron integrals over the context's basis, or (`primitive`) over its primitive basis.
    /// With symmetry set (`set_symmetry`), the matrix is cleaned before it is returned: an operator of the QM region
    /// strictly (it must already be symmetric), the environment's averaged over the group, moment integrals as whole
    /// tensors (when taken about a point every operation leaves fixed, such as the centre of mass), the report kept
    /// (`get_symmetrisation_report`).
    pub fn compute_one_electron_integrals(
        &self,
        output: &mut [f64],
        operator: OneElectronOperator<'_>,
        primitive: bool,
    ) -> Result<(), AedelstenError>
    where
    {
        use OneElectronComponentOperator as Component;
        let cleaning = crate::context::point_group::one_electron_cleaning(&operator);
        // The component operators are cleaned as tensors: (order, kind, origin, name). The moment and the angular
        // momentum about an origin that is not a fixed point of the group are left alone (clean_library_tensor).
        let tensor = match &operator {
            OneElectronOperator::Array3DOutput(op) => Some(match op {
                Component::Moment(origin) => (origin.get_order(), symmetry::TensorKind::Polar, Some(*origin.get_origin()), "oneint_moment"),
                Component::AngularMomentum(origin) => (1, symmetry::TensorKind::Axial, Some(*origin), "oneint_angular_momentum"),
                Component::Nabla => (1, symmetry::TensorKind::Polar, None, "oneint_nabla"),
            }),
            _ => None,
        };
        self.compute_one_electron_integrals_uncleaned(output, operator, primitive)?;
        if let Some(cleaning) = cleaning {
            self.clean_library_output(output, primitive, cleaning.cleaning, cleaning.name)?;
        }
        if let Some((order, kind, origin, name)) = tensor {
            use crate::context::point_group::Cleaning::Strict;
            self.clean_library_tensor(output, primitive, order, kind, origin, Strict, name)?;
        }
        Ok(())
    }

    /// `compute_one_electron_integrals` without the symmetry cleaning: over the contracted basis, or (`primitive`) the
    /// primitive one.
    pub(crate) fn compute_one_electron_integrals_uncleaned(
        &self,
        output: &mut [f64],
        operator: OneElectronOperator<'_>,
        primitive: bool,
    ) -> Result<(), AedelstenError> {
        if primitive {
            self.one_electron_integrals_over::<basis::contraction::Uncontracted, basis::contraction::Uncontracted>(
                self.get_n_prim_unique(), || self.primitive_pairs(), || self.primitive_shells(), output, operator,
            )
        } else {
            let pairs = || {
                self.shell_pairs.get_cache().map(|pairs| pairs.as_slice()).map_err(|_| {
                    AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
                })
            };
            let shells = || Ok(self.qm_region.get_cache().expect("input has been freed (free_input)").input.get_basis_shells());
            self.one_electron_integrals_over::<S, S>(self.get_n_cont(), pairs, shells, output, operator)
        }
    }

    /// The integrals of `operator` over a basis of `n` functions, its shell pairs and shells given lazily (only the
    /// operator's kind asks for one or the other), into `output`.
    pub(crate) fn one_electron_integrals_over<'s, K, Kb>(
        &'s self,
        n: usize,
        pairs: impl FnOnce() -> Result<&'s [basis::shell_pairs::ShellPair<K>], AedelstenError>,
        shells: impl FnOnce() -> Result<&'s [basis::shell_types::BasisShell<Kb>], AedelstenError>,
        output: &mut [f64],
        operator: OneElectronOperator<'_>,
    ) -> Result<(), AedelstenError>
    where
        K: basis::contraction::ShellKind,
        Kb: basis::contraction::ShellKind,
    {
        match operator {
            OneElectronOperator::MatrixOutput(op) => {
                check_matrix_dimensions(n, output)?;
                let mut wrapped_output = arrays::matrix::MatrixViewMut::new(n, n, output);
                let shell_pairs = pairs()?;
                let mut scratch = self.require_scratch()?;
                compute_one_electron_integrals_matrix(shell_pairs, op, &mut scratch, &self.comm, &mut wrapped_output)?;
            }
            OneElectronOperator::Array3DOutput(op) => {
                let n_comps = op.get_expected_components();
                check_array3d_dimensions(n, n_comps, output)?;
                let mut wrapped_output = arrays::array3d::Array3DViewMut::new(n, n, n_comps, output);
                let shell_pairs = pairs()?;
                let mut scratch = self.require_scratch()?;
                compute_one_electron_integrals_array3d(shell_pairs, op, &mut scratch, &self.comm, &mut wrapped_output)?;
            }
            OneElectronOperator::AimpSpecial(op) => {
                check_matrix_dimensions(n, output)?;
                let mut wrapped_output = arrays::matrix::MatrixViewMut::new(n, n, output);
                let basis_shells = shells()?;
                let mut scratch = self.require_scratch()?;
                compute_one_electron_aimp_special_integrals(basis_shells, op, &mut scratch, &self.comm, &mut wrapped_output)?;
            }
        }
        Ok(())
    }
}

// The one-electron operators by their C codes and wire names (`oneint_*`).
c_int_enum! {
    pub enum OneElectronOperatorKind {
        Overlap = 0 => "oneint_overlap",
        Kinetic = 1 => "oneint_kinetic",
        NuclearAttraction = 2 => "oneint_nuclear_attraction",
        // The environment's point multipoles (charges, dipoles, quadrupoles, ...), whatever their orders.
        ExternalMultipole = 3 => "oneint_external_multipole",
        CoreAimpPointCharge = 6 => "oneint_core_aimp_point_charge",
        CoreAimpScreenedCoulomb = 7 => "oneint_core_aimp_screened_coulomb",
        EmbeddingAimpPointCharge = 8 => "oneint_embedding_aimp_point_charge",
        EmbeddingAimpScreenedCoulomb = 9 => "oneint_embedding_aimp_screened_coulomb",
        CoreAimpCoulomb = 10 => "oneint_core_aimp_coulomb",
        EmbeddingAimpCoulomb = 11 => "oneint_embedding_aimp_coulomb",
        CoreAimpProjection = 12 => "oneint_core_aimp_projection",
        CoreAimpSpectrallyResolvedExchange = 13 => "oneint_core_aimp_spectrally_resolved_exchange",
        CoreAimpExchange = 14 => "oneint_core_aimp_exchange",
        EmbeddingAimpProjection = 15 => "oneint_embedding_aimp_projection",
        EmbeddingAimpSpectrallyResolvedExchange = 16 => "oneint_embedding_aimp_spectrally_resolved_exchange",
        EmbeddingAimpExchange = 17 => "oneint_embedding_aimp_exchange",
        Moment = 18 => "oneint_moment",
        AngularMomentum = 19 => "oneint_angular_momentum",
        Nabla = 20 => "oneint_nabla",
        SapHelFem = 21 => "oneint_sap_helfem",
        SapGrasp = 22 => "oneint_sap_grasp",
    }
    invalid_error = AedelstenError::InvalidIntegralType;
}

impl OneElectronOperatorKind {
    /// The number of output planes of a component operator (`n x n x planes`): the moment's Cartesian components of
    /// `order`, 3 for the angular momentum and nabla; `None` for an operator with a plain matrix output.
    pub fn component_count(self, order: usize) -> Option<usize> {
        match self {
            OneElectronOperatorKind::Moment => Some(cart2sph::compute_cartesian_degeneracy(order)),
            OneElectronOperatorKind::AngularMomentum | OneElectronOperatorKind::Nabla => Some(3),
            _ => None,
        }
    }
}

// The wire names and the codes of the operator kinds are distinct (the macro derives the rest from one list).
#[cfg(test)]
mod wire_enum_tests {
    use super::OneElectronOperatorKind;
    use std::collections::HashSet;

    #[test]
    fn wire_keys_are_unique_and_carry_the_oneint_prefix() {
        let mut seen = HashSet::new();
        for (key, _) in OneElectronOperatorKind::NAME_TABLE {
            assert!(key.starts_with("oneint_"), "wire key '{key}' is missing the oneint_ prefix");
            assert!(seen.insert(*key), "duplicate wire key '{key}' in OneElectronOperatorKind::NAME_TABLE");
        }
    }

    #[test]
    fn discriminants_are_unique() {
        let mut seen = HashSet::new();
        for (_, disc) in OneElectronOperatorKind::NAME_TABLE {
            assert!(seen.insert(*disc), "duplicate discriminant {disc} in OneElectronOperatorKind::NAME_TABLE");
        }
    }
}

// Enum of enums, nested by output shape (Matrix vs Array3D) and by whether
// the operator needs basis_shells instead of shell_pairs (AimpSpecial) --
// exactly the axis compute_one_electron_integrals needs to dispatch on.
// Callers never spell this nesting themselves: see the OneElectronOperator::*
// constructors just below, which are the intended way to build one.
#[derive(Clone)]
pub enum OneElectronOperator<'a> {
    MatrixOutput(OneElectronPlainOperator<'a>),
    Array3DOutput(OneElectronComponentOperator),
    AimpSpecial(OneElectronAimpProjectionOperator<'a>)
}

// Operator with a regular matrix output.
#[derive(Clone)]
pub enum OneElectronPlainOperator<'a> {
    Overlap,
    Kinetic,
    NuclearAttraction(Option<&'a NuclearPotential>),
    ExternalMultipole(&'a ExternalMultipoles, basis::charge_carriers::MultipoleConvention),
    CoreAimpPointCharge(&'a [basis::IndexedAimp]),
    CoreAimpScreenedCoulomb(&'a [basis::IndexedAimp]),
    CoreAimpCoulomb(&'a [basis::IndexedAimp]),
    EmbeddingAimpPointCharge(&'a [basis::Aimp]),
    EmbeddingAimpScreenedCoulomb(&'a [basis::Aimp]),
    EmbeddingAimpCoulomb(&'a [basis::Aimp]),
    /// The SAP screening of `get_sap_screening_operator`: one series per real atom.
    SapScreening(SapFit, Box<[basis::charge_carriers::IndexedContractedGaussianCharge]>),
}

// Operators with Cartesian -- or possibly other -- components,
// forcing the usage of Array3D instead of matrices.
#[derive(Clone)]
pub enum OneElectronComponentOperator {
    Moment(oneint::MomentOrigin),
    /// About the gauge origin; see `OneElectronOperator::angular_momentum`.
    AngularMomentum([f64; 3]),
    /// See `OneElectronOperator::nabla`.
    Nabla,
}

impl OneElectronComponentOperator {
    /// The number of output planes: the moment's Cartesian components of its order, 3 for the angular momentum and
    /// nabla.
    pub(crate) fn get_expected_components(&self) -> usize {
        match self {
            // The number of Cartesian components of the order.
            OneElectronComponentOperator::Moment(moment_origin) => cart2sph::compute_cartesian_degeneracy(moment_origin.get_order()),
            OneElectronComponentOperator::AngularMomentum(_) | OneElectronComponentOperator::Nabla => 3,
        }
    }
}

/// The AIMPs' nonlocal operators (projection, spectrally resolved exchange, both), core or embedding, with the
/// projection operators they are built from and their sites.
#[derive(Clone)]
pub enum OneElectronAimpProjectionOperator<'a> {
    CoreAimpProjection(&'a RefCell<Vec<aimp::AimpProjectionOperator>>, &'a [basis::IndexedAimp]),
    CoreAimpSpectrallyResolvedExchange(&'a RefCell<Vec<aimp::AimpProjectionOperator>>, &'a [basis::IndexedAimp]),
    CoreAimpExchange(&'a RefCell<Vec<aimp::AimpProjectionOperator>>, &'a [basis::IndexedAimp]),
    EmbeddingAimpProjection(&'a RefCell<Vec<aimp::AimpProjectionOperator>>, &'a [basis::Aimp]),
    EmbeddingAimpSpectrallyResolvedExchange(&'a RefCell<Vec<aimp::AimpProjectionOperator>>, &'a [basis::Aimp]),
    EmbeddingAimpExchange(&'a RefCell<Vec<aimp::AimpProjectionOperator>>, &'a [basis::Aimp]),
}

impl<'a> OneElectronOperator<'a> {
    /// Whether this is the SAP screening (refused by the picture-change integrals).
    pub(crate) fn is_sap_screening(&self) -> bool {
        matches!(self, OneElectronOperator::MatrixOutput(OneElectronPlainOperator::SapScreening(..)))
    }

    /// The overlap.
    pub fn overlap() -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::Overlap)
    }

    /// The kinetic energy, `-1/2 nabla^2`.
    pub fn kinetic() -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::Kinetic)
    }

    /// The attraction of the nuclei of `potential`; zero integrals for `None` (only core AIMPs).
    pub fn nuclear_attraction(potential: Option<&'a NuclearPotential>) -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::NuclearAttraction(potential))
    }

    /// The external point charges and multipoles `multipoles`, in `convention`.
    pub fn external_multipole(multipoles: &'a ExternalMultipoles, convention: basis::charge_carriers::MultipoleConvention) -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::ExternalMultipole(multipoles, convention))
    }

    /// The core AIMPs' point charges, `-Z/r`.
    pub fn core_aimp_point_charge(aimps: &'a [basis::IndexedAimp]) -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::CoreAimpPointCharge(aimps))
    }

    /// The core AIMPs' screening corrections, `-Z s(r)/r`.
    pub fn core_aimp_screened_coulomb(aimps: &'a [basis::IndexedAimp]) -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::CoreAimpScreenedCoulomb(aimps))
    }

    /// The core AIMPs' Coulomb part, `-Z (1 + s(r))/r`.
    pub fn core_aimp_coulomb(aimps: &'a [basis::IndexedAimp]) -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::CoreAimpCoulomb(aimps))
    }

    /// The embedding AIMPs' point charges, `-Z/r`.
    pub fn embedding_aimp_point_charge(aimps: &'a [basis::Aimp]) -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::EmbeddingAimpPointCharge(aimps))
    }

    /// The embedding AIMPs' screening corrections, `-Z s(r)/r`.
    pub fn embedding_aimp_screened_coulomb(aimps: &'a [basis::Aimp]) -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::EmbeddingAimpScreenedCoulomb(aimps))
    }

    /// The embedding AIMPs' Coulomb part, `-Z (1 + s(r))/r`.
    pub fn embedding_aimp_coulomb(aimps: &'a [basis::Aimp]) -> Self {
        OneElectronOperator::MatrixOutput(OneElectronPlainOperator::EmbeddingAimpCoulomb(aimps))
    }

    /// The Cartesian multipole moments about `origin`, of its order.
    pub fn moment(origin: oneint::MomentOrigin) -> Self {
        OneElectronOperator::Array3DOutput(OneElectronComponentOperator::Moment(origin))
    }

    /// The orbital angular momentum about the gauge origin `O`: three real ANTISYMMETRIC matrices (`n x n x 3`, x y z)
    /// `A_x(mu, nu) = <mu| (y - O_y) d/dz - (z - O_z) d/dy |nu>` (y, z cyclic), so that
    /// `<mu| L |nu> = -i A(mu, nu)` for `L = -i (r - O) x nabla`.
    pub fn angular_momentum(origin: [f64; 3]) -> Self {
        OneElectronOperator::Array3DOutput(OneElectronComponentOperator::AngularMomentum(origin))
    }

    /// The linear momentum: three real ANTISYMMETRIC matrices (`n x n x 3`, x y z) `nabla_x(mu, nu) = <mu| d/dx |nu>`,
    /// so that `<mu| p |nu> = -i nabla(mu, nu)`. With the angular momentum,
    /// `A(O') = A(O) + (O - O') x nabla` componentwise.
    pub fn nabla() -> Self {
        OneElectronOperator::Array3DOutput(OneElectronComponentOperator::Nabla)
    }

    /// The core AIMPs' projection operator.
    pub fn core_aimp_projection(operators: &'a RefCell<Vec<aimp::AimpProjectionOperator>>, aimps: &'a [basis::IndexedAimp]) -> Self {
        OneElectronOperator::AimpSpecial(OneElectronAimpProjectionOperator::CoreAimpProjection(operators, aimps))
    }

    /// The core AIMPs' spectrally resolved exchange.
    pub fn core_aimp_spectrally_resolved_exchange(operators: &'a RefCell<Vec<aimp::AimpProjectionOperator>>, aimps: &'a [basis::IndexedAimp]) -> Self {
        OneElectronOperator::AimpSpecial(OneElectronAimpProjectionOperator::CoreAimpSpectrallyResolvedExchange(operators, aimps))
    }

    /// The core AIMPs' projection and spectrally resolved exchange together.
    pub fn core_aimp_exchange(operators: &'a RefCell<Vec<aimp::AimpProjectionOperator>>, aimps: &'a [basis::IndexedAimp]) -> Self {
        OneElectronOperator::AimpSpecial(OneElectronAimpProjectionOperator::CoreAimpExchange(operators, aimps))
    }

    /// The embedding AIMPs' projection operator.
    pub fn embedding_aimp_projection(operators: &'a RefCell<Vec<aimp::AimpProjectionOperator>>, aimps: &'a [basis::Aimp]) -> Self {
        OneElectronOperator::AimpSpecial(OneElectronAimpProjectionOperator::EmbeddingAimpProjection(operators, aimps))
    }

    /// The embedding AIMPs' spectrally resolved exchange.
    pub fn embedding_aimp_spectrally_resolved_exchange(operators: &'a RefCell<Vec<aimp::AimpProjectionOperator>>, aimps: &'a [basis::Aimp]) -> Self {
        OneElectronOperator::AimpSpecial(OneElectronAimpProjectionOperator::EmbeddingAimpSpectrallyResolvedExchange(operators, aimps))
    }

    /// The embedding AIMPs' projection and spectrally resolved exchange together.
    pub fn embedding_aimp_exchange(operators: &'a RefCell<Vec<aimp::AimpProjectionOperator>>, aimps: &'a [basis::Aimp]) -> Self {
        OneElectronOperator::AimpSpecial(OneElectronAimpProjectionOperator::EmbeddingAimpExchange(operators, aimps))
    }
}

/// The integrals of a plain operator over `shell_pairs` into the matrix `output`, the pairs shared among the ranks of
/// `comm`.
pub(crate) fn compute_one_electron_integrals_matrix<S: ShellKind, Mf64>(
    shell_pairs: &[ShellPair<S>],
    operator: OneElectronPlainOperator<'_>,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    output: &mut Mf64,
) -> Result<(), AedelstenError>
where
    Mf64: arrays::matrix::MatrixMathMut<f64>,
{
    let mut workstruct = oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm);
    match operator {
        OneElectronPlainOperator::Overlap => Ok(workstruct.compute_overlap_integrals(output)?),
        OneElectronPlainOperator::Kinetic => Ok(workstruct.compute_kinetic_integrals(output)?),
        OneElectronPlainOperator::NuclearAttraction(potential_type) => {
            // No nuclear potential (only core AIMPs): zero integrals.
            match potential_type {
                Some(basis::charge_carriers::NuclearPotential::PointCharge(charges)) => Ok(workstruct.compute_point_charge_integrals(output, charges)?),
                Some(basis::charge_carriers::NuclearPotential::GaussianCharge(charges)) => Ok(workstruct.compute_finite_charge_integrals(output, charges)?),
                None => Ok(output.fill(0.0)),
            }
        }
        OneElectronPlainOperator::ExternalMultipole(multipoles, convention) => match multipoles {
            ExternalMultipoles::Charges(charges) => Ok(workstruct.compute_point_charge_integrals(output, charges)?),
            ExternalMultipoles::Multipoles(multipoles) => Ok(workstruct.compute_external_multipole_integrals(output, multipoles, convention)?),
        },
        // The AIMPs' Coulomb parts, core and embedding, through the same generic `AimpCoulombIntegrals` methods.
        OneElectronPlainOperator::CoreAimpPointCharge(aimps) => Ok(workstruct.compute_aimp_point_charge_contribution(output, aimps)?),
        OneElectronPlainOperator::CoreAimpScreenedCoulomb(aimps) => Ok(workstruct.compute_aimp_screened_coulomb_contribution(output, aimps)?),
        OneElectronPlainOperator::CoreAimpCoulomb(aimps) => Ok(workstruct.compute_full_aimp_coulomb_contribution(output, aimps)?),
        OneElectronPlainOperator::EmbeddingAimpPointCharge(aimps) => Ok(workstruct.compute_aimp_point_charge_contribution(output, aimps)?),
        OneElectronPlainOperator::EmbeddingAimpScreenedCoulomb(aimps) => Ok(workstruct.compute_aimp_screened_coulomb_contribution(output, aimps)?),
        OneElectronPlainOperator::EmbeddingAimpCoulomb(aimps) => Ok(workstruct.compute_full_aimp_coulomb_contribution(output, aimps)?),
        OneElectronPlainOperator::SapScreening(_, charges) => Ok(workstruct.compute_finite_charge_integrals(output, &charges)?),
    }
}

/// The integrals of a component operator over `shell_pairs` into the planes of `output`, the pairs shared among the
/// ranks of `comm`.
pub(crate) fn compute_one_electron_integrals_array3d<S: ShellKind, Af64>(
    shell_pairs: &[ShellPair<S>],
    operator: OneElectronComponentOperator,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    output: &mut Af64,
) -> Result<(), AedelstenError>
where
    Af64: Array3DTraitMut<f64>,
{
    let mut workstruct = oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm);
    match operator {
        OneElectronComponentOperator::Moment(moment_origin) => Ok(workstruct.compute_moment_integrals(output, &moment_origin)?),
        OneElectronComponentOperator::AngularMomentum(origin) => Ok(workstruct.compute_angular_momentum_integrals(output, &origin)?),
        OneElectronComponentOperator::Nabla => Ok(workstruct.compute_nabla_integrals(output)?),
    }
}

/// The electric-field integrals at `sites`; `local`: every pair on this rank, no communication
/// (`oneint::OneIntWorkStruct::new_local`), else shared among the ranks of `comm` and summed.
pub(crate) fn compute_electric_field_integrals_array3d<S: ShellKind, Af64, Site>(
    shell_pairs: &[ShellPair<S>],
    sites: &[Site],
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    output: &mut Af64,
    local: bool,
) -> Result<(), AedelstenError>
where
    Af64: Array3DTraitMut<f64>,
    Site: basis::charge_carriers::SiteTrait,
{
    let mut workstruct = if local {
        oneint::OneIntWorkStruct::new_local(shell_pairs, scratch)
    } else {
        oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm)
    };
    Ok(workstruct.compute_electric_field_integrals(output, sites)?)
}

/// The gradient of a plain operator contracted with `density` into `gradient` (`(3, n_centres)`), with the source terms
/// of the operators whose sources are centres; `active` the frozen-centre mask.
pub(crate) fn compute_one_electron_gradient<S: ShellKind, Mg, Md>(
    shell_pairs: &[ShellPair<S>],
    operator: OneElectronPlainOperator<'_>,
    gradient: &mut Mg,
    density: &Md,
    active: Option<&[bool]>,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
) -> Result<(), AedelstenError>
where
    Mg: arrays::matrix::MatrixMathMut<f64>,
    Md: arrays::matrix::MatrixTrait<f64>,
{
    let mut workstruct = oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm);
    // Sources bound to a centre (nuclei, core AIMPs): (position in the list, centre index). External
    // charges/multipoles and embedding AIMPs do not move.
    let fixed: &[(usize, usize)] = &[];
    match operator {
        OneElectronPlainOperator::Overlap => Ok(workstruct.contract_overlap_gradient(gradient, density, active)?),
        OneElectronPlainOperator::Kinetic => Ok(workstruct.contract_kinetic_gradient(gradient, density, active)?),
        OneElectronPlainOperator::NuclearAttraction(potential_type) => match potential_type {
            Some(basis::charge_carriers::NuclearPotential::PointCharge(charges)) => {
                let centres: Vec<(usize, usize)> = charges.iter().enumerate().map(|(k, c)| (k, c.get_idx())).collect();
                Ok(workstruct.contract_point_charge_gradient(gradient, density, charges, &centres, active)?)
            }
            Some(basis::charge_carriers::NuclearPotential::GaussianCharge(charges)) => {
                let centres: Vec<(usize, usize)> = charges.iter().enumerate().map(|(k, c)| (k, c.get_idx())).collect();
                Ok(workstruct.contract_finite_charge_gradient(gradient, density, charges, &centres, active)?)
            }
            None => Ok(gradient.fill(0.0)),
        },
        OneElectronPlainOperator::ExternalMultipole(multipoles, convention) => match multipoles {
            ExternalMultipoles::Charges(charges) => Ok(workstruct.contract_point_charge_gradient(gradient, density, charges, fixed, active)?),
            ExternalMultipoles::Multipoles(multipoles) => Ok(workstruct.contract_external_multipole_gradient(gradient, density, multipoles, convention, fixed, active)?),
        },
        OneElectronPlainOperator::CoreAimpPointCharge(aimps) => {
            let centres: Vec<(usize, usize)> = aimps.iter().enumerate().map(|(k, a)| (k, a.get_idx())).collect();
            Ok(workstruct.contract_aimp_point_charge_gradient(gradient, density, aimps, &centres, active)?)
        }
        OneElectronPlainOperator::CoreAimpScreenedCoulomb(aimps) => {
            let centres: Vec<(usize, usize)> = aimps.iter().enumerate().map(|(k, a)| (k, a.get_idx())).collect();
            Ok(workstruct.contract_aimp_screened_coulomb_gradient(gradient, density, aimps, &centres, active)?)
        }
        OneElectronPlainOperator::CoreAimpCoulomb(aimps) => {
            let centres: Vec<(usize, usize)> = aimps.iter().enumerate().map(|(k, a)| (k, a.get_idx())).collect();
            Ok(workstruct.contract_full_aimp_coulomb_gradient(gradient, density, aimps, &centres, active)?)
        }
        OneElectronPlainOperator::EmbeddingAimpPointCharge(aimps) => Ok(workstruct.contract_aimp_point_charge_gradient(gradient, density, aimps, fixed, active)?),
        OneElectronPlainOperator::EmbeddingAimpScreenedCoulomb(aimps) => Ok(workstruct.contract_aimp_screened_coulomb_gradient(gradient, density, aimps, fixed, active)?),
        OneElectronPlainOperator::EmbeddingAimpCoulomb(aimps) => Ok(workstruct.contract_full_aimp_coulomb_gradient(gradient, density, aimps, fixed, active)?),
        OneElectronPlainOperator::SapScreening(_, charges) => {
            let centres: Vec<(usize, usize)> = charges.iter().enumerate().map(|(k, c)| (k, c.get_idx())).collect();
            Ok(workstruct.contract_finite_charge_gradient(gradient, density, &charges, &centres, active)?)
        }
    }
}

/// The derivatives of the picture-change (p.Vp) integrals of `operator` contracted with `density`, as
/// `compute_one_electron_gradient` for the plain ones: the nuclei move with their centres, external sources do not.
pub(crate) fn compute_one_electron_pvp_gradient<S: ShellKind, Mg, Md>(
    shell_pairs: &[ShellPair<S>],
    operator: OneElectronPlainOperator<'_>,
    gradient: &mut Mg,
    density: &Md,
    active: Option<&[bool]>,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
) -> Result<(), AedelstenError>
where
    Mg: arrays::matrix::MatrixMathMut<f64>,
    Md: arrays::matrix::MatrixTrait<f64>,
{
    let mut workstruct = oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm);
    let fixed: &[(usize, usize)] = &[];
    match operator {
        OneElectronPlainOperator::NuclearAttraction(potential_type) => match potential_type {
            Some(basis::charge_carriers::NuclearPotential::PointCharge(charges)) => {
                let centres: Vec<(usize, usize)> = charges.iter().enumerate().map(|(k, c)| (k, c.get_idx())).collect();
                Ok(workstruct.contract_pvp_point_charge_gradient(gradient, density, charges, &centres, active)?)
            }
            Some(basis::charge_carriers::NuclearPotential::GaussianCharge(charges)) => {
                let centres: Vec<(usize, usize)> = charges.iter().enumerate().map(|(k, c)| (k, c.get_idx())).collect();
                Ok(workstruct.contract_pvp_finite_charge_gradient(gradient, density, charges, &centres, active)?)
            }
            None => Ok(gradient.fill(0.0)),
        },
        OneElectronPlainOperator::ExternalMultipole(multipoles, convention) => match multipoles {
            ExternalMultipoles::Charges(charges) => Ok(workstruct.contract_pvp_point_charge_gradient(gradient, density, charges, fixed, active)?),
            ExternalMultipoles::Multipoles(multipoles) => {
                Ok(workstruct.contract_pvp_external_multipole_gradient(gradient, density, multipoles, convention, fixed, active)?)
            }
        },
        // Embedding AIMPs do not move.
        OneElectronPlainOperator::EmbeddingAimpPointCharge(aimps) => Ok(workstruct.contract_pvp_point_charge_gradient(gradient, density, aimps, fixed, active)?),
        OneElectronPlainOperator::EmbeddingAimpScreenedCoulomb(aimps) => Ok(workstruct.contract_pvp_aimp_coulomb_gradient(gradient, density, aimps, true, fixed, active)?),
        OneElectronPlainOperator::EmbeddingAimpCoulomb(aimps) => Ok(workstruct.contract_pvp_aimp_coulomb_gradient(gradient, density, aimps, false, fixed, active)?),
        _ => Err(AedelstenError::StringError("no picture-change gradient for this operator".to_string())),
    }
}

/// The scalar picture-change integrals `p.Vp` of a plain operator over the primitive `shell_pairs`.
pub(crate) fn compute_one_electron_scalar_picture_change_integrals_matrix<S: ShellKind, Mf64>(
    shell_pairs: &[ShellPair<S>],
    operator: OneElectronPlainOperator<'_>,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    output: &mut Mf64,
) -> Result<(), AedelstenError>
where
    Mf64: arrays::matrix::MatrixMathMut<f64>,
{
    let mut workstruct = oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm);
    match operator {
        OneElectronPlainOperator::NuclearAttraction(potential_type) => {
            // No nuclear potential (only core AIMPs): zero integrals.
            match potential_type {
                Some(basis::charge_carriers::NuclearPotential::PointCharge(charges)) => Ok(workstruct.compute_pvp_scalar_point_charge_integrals(output, charges)?),
                Some(basis::charge_carriers::NuclearPotential::GaussianCharge(charges)) => Ok(workstruct.compute_pvp_scalar_finite_charge_integrals(output, charges)?),
                None => Ok(output.fill(0.0)),
            }
        }
        OneElectronPlainOperator::ExternalMultipole(multipoles, convention) => match multipoles {
            ExternalMultipoles::Charges(charges) => Ok(workstruct.compute_pvp_scalar_point_charge_integrals(output, charges)?),
            // Every moment, as the plain operator (compute_external_multipole_integrals).
            ExternalMultipoles::Multipoles(multipoles) => Ok(workstruct.compute_pvp_scalar_external_multipole_integrals(output, multipoles, convention)?),
        },
        // The embedding AIMPs' Coulomb part: the bare charge, the screen, or both. Their exchange has no picture
        // change.
        OneElectronPlainOperator::EmbeddingAimpPointCharge(aimps) => Ok(workstruct.compute_pvp_scalar_point_charge_integrals(output, aimps)?),
        OneElectronPlainOperator::EmbeddingAimpScreenedCoulomb(aimps) => Ok(workstruct.compute_pvp_scalar_aimp_coulomb_integrals(output, aimps, true)?),
        OneElectronPlainOperator::EmbeddingAimpCoulomb(aimps) => Ok(workstruct.compute_pvp_scalar_aimp_coulomb_integrals(output, aimps, false)?),
        _ => Err(AedelstenError::StringError("no matching scalar picture-change operator implemented".to_string())),
    }
}

/// The spin-orbit picture-change integrals of a plain operator: the planes `(p V x p)` x, y, z and the scalar `p.Vp`.
pub(crate) fn compute_one_electron_spin_orbit_picture_change_integrals_matrix<S: ShellKind, Af64>(
    shell_pairs: &[ShellPair<S>],
    operator: OneElectronPlainOperator<'_>,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    output: &mut Af64,
) -> Result<(), AedelstenError>
where
    Af64: Array3DTraitMut<f64>,
{
    let mut workstruct = oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm);
    match operator {
        OneElectronPlainOperator::NuclearAttraction(potential_type) => {
            // No nuclear potential (only core AIMPs): zero integrals.
            match potential_type {
                Some(basis::charge_carriers::NuclearPotential::PointCharge(charges)) => Ok(workstruct.compute_pvp_spin_orbit_point_charge_integrals(output, charges)?),
                Some(basis::charge_carriers::NuclearPotential::GaussianCharge(charges)) => Ok(workstruct.compute_pvp_spin_orbit_finite_charge_integrals(output, charges)?),
                None => Ok(output.fill(0.0)),
            }
        }
        OneElectronPlainOperator::ExternalMultipole(multipoles, convention) => match multipoles {
            ExternalMultipoles::Charges(charges) => Ok(workstruct.compute_pvp_spin_orbit_point_charge_integrals(output, charges)?),
            // Every moment; see the scalar case above.
            ExternalMultipoles::Multipoles(multipoles) => Ok(workstruct.compute_pvp_spin_orbit_external_multipole_integrals(output, multipoles, convention)?),
        },
        OneElectronPlainOperator::EmbeddingAimpPointCharge(aimps) => Ok(workstruct.compute_pvp_spin_orbit_point_charge_integrals(output, aimps)?),
        OneElectronPlainOperator::EmbeddingAimpScreenedCoulomb(aimps) => Ok(workstruct.compute_pvp_spin_orbit_aimp_coulomb_integrals(output, aimps, true)?),
        OneElectronPlainOperator::EmbeddingAimpCoulomb(aimps) => Ok(workstruct.compute_pvp_spin_orbit_aimp_coulomb_integrals(output, aimps, false)?),
        _ => Err(AedelstenError::StringError("no matching spin-orbit picture-change operator implemented".to_string())),
    }
}

/// The scalar picture-change integrals of a component operator, one plane per component.
pub(crate) fn compute_one_electron_scalar_picture_change_integrals_array3d<S: ShellKind, Af64>(
    shell_pairs: &[ShellPair<S>],
    operator: OneElectronComponentOperator,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    output: &mut Af64,
) -> Result<(), AedelstenError>
where
    Af64: Array3DTraitMut<f64>,
{
    let mut workstruct = oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm);
    match operator {
        OneElectronComponentOperator::Moment(moment_origin) => Ok(workstruct.compute_prp_scalar_integrals(output, &moment_origin)?),
        // No picture change for these: spin-orbit state interaction uses the plain operators.
        OneElectronComponentOperator::AngularMomentum(_) | OneElectronComponentOperator::Nabla => {
            Err(AedelstenError::StringError("no picture-change angular-momentum or nabla integrals".to_string()))
        }
    }
}

/// The spin-orbit picture-change integrals of a component operator: per component the planes x, y, z and the scalar
/// one.
pub(crate) fn compute_one_electron_spin_orbit_picture_change_integrals_array3d<S: ShellKind, Af64>(
    shell_pairs: &[ShellPair<S>],
    operator: OneElectronComponentOperator,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    output: &mut Af64,
) -> Result<(), AedelstenError>
where
    Af64: Array3DTraitMut<f64>,
{
    let mut workstruct = oneint::OneIntWorkStruct::new(shell_pairs, scratch, comm);
    match operator {
        OneElectronComponentOperator::Moment(moment_origin) => Ok(workstruct.compute_prp_spin_orbit_integrals(output, &moment_origin)?),
        OneElectronComponentOperator::AngularMomentum(_) | OneElectronComponentOperator::Nabla => {
            Err(AedelstenError::StringError("no picture-change angular-momentum or nabla integrals".to_string()))
        }
    }
}

/// The AIMPs' nonlocal operator `operator` over `basis_shells` into `output`, the sites shared among the ranks.
pub(crate) fn compute_one_electron_aimp_special_integrals<'a, S: ShellKind, Mf64>(
    basis_shells: &'a [basis::shell_types::BasisShell<S>],
    operator: OneElectronAimpProjectionOperator<'_>,
    scratch: &'a mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    output: &mut Mf64,
) -> Result<(), AedelstenError>
where
    Mf64: arrays::matrix::MatrixMathMut<f64>,
{
    let mut workstruct = aimp::AimpProjectionWorkStruct::new(basis_shells, scratch, comm);
    match operator {
        OneElectronAimpProjectionOperator::CoreAimpProjection(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.compute_aimp_projection_contribution(output, &mut projection_operator, aimps)?)
        }
        OneElectronAimpProjectionOperator::CoreAimpSpectrallyResolvedExchange(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.compute_aimp_spectrally_resolved_exchange_contribution(output, &mut projection_operator, aimps)?)
        }
        OneElectronAimpProjectionOperator::CoreAimpExchange(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.compute_full_aimp_exchange_contribution(output, &mut projection_operator, aimps)?)
        }
        OneElectronAimpProjectionOperator::EmbeddingAimpProjection(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.compute_aimp_projection_contribution(output, &mut projection_operator, aimps)?)
        }
        OneElectronAimpProjectionOperator::EmbeddingAimpSpectrallyResolvedExchange(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.compute_aimp_spectrally_resolved_exchange_contribution(output, &mut projection_operator, aimps)?)
        }
        OneElectronAimpProjectionOperator::EmbeddingAimpExchange(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.compute_full_aimp_exchange_contribution(output, &mut projection_operator, aimps)?)
        }
    }
}

/// The gradient of the AIMPs' nonlocal operator `operator` contracted with `density`, as
/// `compute_one_electron_gradient` for the plain ones.
pub(crate) fn compute_one_electron_aimp_special_gradient<'a, S: ShellKind, Mg, Md>(
    basis_shells: &'a [basis::shell_types::BasisShell<S>],
    operator: OneElectronAimpProjectionOperator<'_>,
    gradient: &mut Mg,
    density: &Md,
    active: Option<&[bool]>,
    scratch: &'a mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
) -> Result<(), AedelstenError>
where
    Mg: arrays::matrix::MatrixTraitMut<f64>,
    Md: arrays::matrix::MatrixTrait<f64>,
{
    let mut workstruct = aimp::AimpProjectionWorkStruct::new(basis_shells, scratch, comm);
    match operator {
        OneElectronAimpProjectionOperator::CoreAimpProjection(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.contract_aimp_projection_gradient(gradient, density, &mut projection_operator, aimps, active)?)
        }
        OneElectronAimpProjectionOperator::CoreAimpSpectrallyResolvedExchange(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.contract_aimp_spectrally_resolved_exchange_gradient(gradient, density, &mut projection_operator, aimps, active)?)
        }
        OneElectronAimpProjectionOperator::CoreAimpExchange(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.contract_full_aimp_exchange_gradient(gradient, density, &mut projection_operator, aimps, active)?)
        }
        OneElectronAimpProjectionOperator::EmbeddingAimpProjection(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.contract_aimp_projection_gradient(gradient, density, &mut projection_operator, aimps, active)?)
        }
        OneElectronAimpProjectionOperator::EmbeddingAimpSpectrallyResolvedExchange(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.contract_aimp_spectrally_resolved_exchange_gradient(gradient, density, &mut projection_operator, aimps, active)?)
        }
        OneElectronAimpProjectionOperator::EmbeddingAimpExchange(projection_operator, aimps) => {
            let mut projection_operator = projection_operator.borrow_mut();
            Ok(workstruct.contract_full_aimp_exchange_gradient(gradient, density, &mut projection_operator, aimps, active)?)
        }
    }
}

/// The operator of `kind` (a C code) on the context `ctx`, the moment's and the angular momentum's origin from
/// `moment_origin`; an error for an operator the input lacks.
pub(crate) fn resolve_one_electron_operator(
    ctx: &crate::AedelstenContext,
    kind: OneElectronOperatorKind,
    moment_origin: oneint::MomentOrigin,
) -> Result<OneElectronOperator<'_>, AedelstenError> {
    Ok(match kind {
        OneElectronOperatorKind::Overlap => OneElectronOperator::overlap(),
        OneElectronOperatorKind::Kinetic => OneElectronOperator::kinetic(),
        OneElectronOperatorKind::Moment => OneElectronOperator::moment(moment_origin),
        // The gauge origin travels in the moment origin's place (its order is not read).
        OneElectronOperatorKind::AngularMomentum => OneElectronOperator::angular_momentum(*moment_origin.get_origin()),
        OneElectronOperatorKind::Nabla => OneElectronOperator::nabla(),
        OneElectronOperatorKind::NuclearAttraction => ctx.get_nuclear_attraction_operator(),
        OneElectronOperatorKind::ExternalMultipole => ctx.get_external_multipole_operator()?,
        OneElectronOperatorKind::CoreAimpPointCharge => ctx.get_core_aimp_point_charge_operator()?,
        OneElectronOperatorKind::CoreAimpScreenedCoulomb => ctx.get_core_aimp_screened_coulomb_operator()?,
        OneElectronOperatorKind::CoreAimpCoulomb => ctx.get_core_aimp_coulomb_operator()?,
        OneElectronOperatorKind::EmbeddingAimpPointCharge => ctx.get_embedding_aimp_point_charge_operator()?,
        OneElectronOperatorKind::EmbeddingAimpScreenedCoulomb => ctx.get_embedding_aimp_screened_coulomb_operator()?,
        OneElectronOperatorKind::EmbeddingAimpCoulomb => ctx.get_embedding_aimp_coulomb_operator()?,
        OneElectronOperatorKind::CoreAimpProjection => ctx.get_core_aimp_projection_operator()?,
        OneElectronOperatorKind::CoreAimpSpectrallyResolvedExchange => ctx.get_core_aimp_spectrally_resolved_exchange_operator()?,
        OneElectronOperatorKind::CoreAimpExchange => ctx.get_core_aimp_exchange_operator()?,
        OneElectronOperatorKind::EmbeddingAimpProjection => ctx.get_embedding_aimp_projection_operator()?,
        OneElectronOperatorKind::EmbeddingAimpSpectrallyResolvedExchange => ctx.get_embedding_aimp_spectrally_resolved_exchange_operator()?,
        OneElectronOperatorKind::EmbeddingAimpExchange => ctx.get_embedding_aimp_exchange_operator()?,
        OneElectronOperatorKind::SapHelFem => ctx.get_sap_screening_operator(SapFit::HelFem)?,
        OneElectronOperatorKind::SapGrasp => ctx.get_sap_screening_operator(SapFit::Grasp)?,
    })
}

#[cfg(test)]
mod sap_tests {
    use crate::{AedelstenContext, ContractionType, OneElectronOperator, SapFit};
    use super::OneElectronPlainOperator;
    use basis::charge_carriers::{FiniteChargeTrait, IndexedGaussianCharge, NuclearPotential, SiteTrait};

    /// The SAP screening's contracted series give the integrals of the same charges as separate single-Gaussian nuclei:
    /// the finite-charge kernel summed over each atom's terms.
    #[test]
    fn the_series_equal_their_terms_as_single_gaussian_charges() {
        let dir = std::env::temp_dir().join(format!("aedelsten_sap_terms_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.inp");
        std::fs::write(&path, "Basis set\n    Ne.x2c-SVPall\n    Ne1 0.0 0.1 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 1.2 1.4\nEnd of Basis\n").unwrap();
        let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        ctx.set_shell_pairs(None).unwrap();
        ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
        let n = ctx.get_n_cont();
        for fit in [SapFit::HelFem, SapFit::Grasp] {
            let operator = ctx.get_sap_screening_operator(fit).unwrap();
            let OneElectronOperator::MatrixOutput(OneElectronPlainOperator::SapScreening(_, series)) = &operator else { panic!("a SAP operator") };
            assert_eq!(series.len(), 2);
            let terms: Box<[IndexedGaussianCharge]> = series
                .iter()
                .flat_map(|s| s.get_exponents().iter().zip(s.get_charges()).map(|(&e, &q)| IndexedGaussianCharge::new(s.get_idx(), *s.get_position(), q, e)))
                .collect();
            let separate = NuclearPotential::GaussianCharge(terms);
            let mut expected = vec![0.0; n * n];
            ctx.compute_one_electron_integrals(&mut expected, OneElectronOperator::nuclear_attraction(Some(&separate)), false).unwrap();
            let mut w = vec![0.0; n * n];
            ctx.compute_one_electron_integrals(&mut w, operator, false).unwrap();
            let largest = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let difference = w.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(largest > 1.0 && difference < 1e-12 * largest, "{fit:?}: {difference} of {largest}");
        }
    }
}
