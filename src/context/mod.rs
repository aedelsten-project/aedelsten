// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The context: one input's basis, geometry and cached state, and everything computed from them.
//
// `AedelstenContext` is the public type, an enum over the contraction kinds so that the kind is chosen at run time
// and never appears in a signature. `AedelstenContextImpl<S>` holds the state and does the work, generic over the
// kind. Each topic file holds both halves of its methods: the public forwarders on `AedelstenContext` (one
// `dispatch!` each, with the method's documentation) and the generic implementation below them.
//
//   setup.rs         scratch, shell pairs, primitive shells, memory breakdown, the screening defaults
//   basis_set.rs     basis functions, primitives, centres, geometry, Cartesian rescaling, AO amplitudes
//   model.rs         the model basis (free-atom reference orbitals) and overlaps with another context
//   one_electron.rs  one-electron operators, integrals and their gradients
//   two_electron.rs  ERI data, the stored tensor, the direct Fock builders, the half transform
//   gradients.rs     two-electron and nuclear (classical) gradients
//   embedding.rs     AIMPs, external multipoles, polarization, the electric field
//   relativity.rs    X2C, the picture change, the finite nucleus
//   point_group.rs   point-group symmetry

use crate::*;
use cache::*;
use std::cell::{RefCell, RefMut};
use arrays::array3d::{Array3DTrait, Array3DTraitMut};
use arrays::matrix::{MatrixMath, MatrixTrait};
use arrays::vector::VectorMut;
use basis::charge_carriers::PointChargeTrait;

mod cache;

// The QM region: the input and its core AIMPs' projection operators, one cache so that the two stay together. The
// operators are the `aimp` crate's, which depends on `basis`, so `InputStructure` cannot hold them.
#[derive(Debug)]
pub(crate) struct QmRegion<S: basis::contraction::ShellKind> {
    input: basis::InputStructure<S>,
    core_aimp_operators: RefCell<Vec<aimp::AimpProjectionOperator>>,
}

impl<S: basis::contraction::ShellKind> QmRegion<S> {
    fn new(input: basis::InputStructure<S>) -> Result<Self, AedelstenError> {
        let core_aimp_operators = match input.get_core_regions() {
            Some(core_aimps) => aimp::materialise_aimp_projection_operator(core_aimps)?,
            None => Vec::new(),
        };
        Ok(Self { input, core_aimp_operators: RefCell::new(core_aimp_operators) })
    }
}

// The context of one contraction kind: its state, and the work of every method. Not public: `AedelstenContext` is
// the type outside the crate.
#[derive(Debug)]
pub(crate) struct AedelstenContextImpl<S: basis::contraction::ShellKind> {
    qm_region: AedelstenCacheState<QmRegion<S>>,
    shell_pairs: AedelstenCacheState<Vec<basis::shell_pairs::ShellPair<S>>>,
    scratch: AedelstenCacheState<RefCell<arrays::scratch::Scratch>>,
    external_multipoles: AedelstenCacheState<basis::charge_carriers::ExternalMultipoles>,
    // Which interaction of higher multipoles with the electrons the external-multipole operator represents.
    multipole_convention: basis::charge_carriers::MultipoleConvention,
    polarizable_sites: AedelstenCacheState<Box<[basis::charge_carriers::PolarizableSite]>>,
    polarization: AedelstenCacheState<RefCell<crate::context::embedding::PolarizationState>>,
    embedding_aimp_operators: AedelstenCacheState<RefCell<Vec<aimp::AimpProjectionOperator>>>,
    primitive_shells: AedelstenCacheState<Box<[basis::shell_types::BasisShell<basis::contraction::Uncontracted>]>>,
    primitive_shell_pairs: AedelstenCacheState<Vec<basis::shell_pairs::ShellPair<basis::contraction::Uncontracted>>>,
    contraction_matrix: AedelstenCacheState<arrays::matrix::Matrix<f64>>,
    eri_data: AedelstenCacheState<eri::EriData>,
    picture_change: AedelstenCacheState<x2c::PictureChange>,
    // The densities of the X2C gradient with the decoupling held, see context/relativity.rs.
    x2c_frozen_density: AedelstenCacheState<crate::context::relativity::X2cFrozenDensities>,
    // Point group of the QM region and the AO representation of its operations, see context/point_group.rs.
    symmetry: AedelstenCacheState<crate::context::point_group::SymmetryState>,
    // The ranks the context's parallel computations are split over (setup.rs).
    comm: aedelsten_mpi::Communicator,
}

/// The context of one input: its basis, geometry and embedding, the state derived from them, and every computation
/// over them. The contraction kind (general, segmented, primitive) is chosen when it is made, not named in any type.
#[allow(private_interfaces)] // the variants hold the crate's implementation, which no caller names
pub enum AedelstenContext {
    General(AedelstenContextImpl<basis::contraction::Contracted>),
    Segmented(AedelstenContextImpl<basis::contraction::Segmented>),
    Primitive(AedelstenContextImpl<basis::contraction::Uncontracted>),
}

// The contraction kind of a context made without one (the C interface's default).
pub(crate) fn default_contraction_type() -> basis::shell_types::ContractionType {
    basis::shell_types::ContractionType::General
}

impl AedelstenContext {
    /// The context of the input file `input` (its basis, geometry and embedding), with shells of `contraction_type`.
    /// Its parallel computations are split over the ranks of `comm` (`set_communicator` changes it): every
    /// collective method must then be called by every rank of `comm`. `Communicator::single()` keeps every
    /// computation on this rank; MPI is neither needed nor initialized here.
    pub fn new(input: &str, contraction_type: basis::shell_types::ContractionType, comm: &aedelsten_mpi::Communicator) -> Result<Self, AedelstenError> {
        match contraction_type {
            basis::shell_types::ContractionType::General => Ok(AedelstenContext::General(AedelstenContextImpl::<basis::contraction::Contracted>::new(input, comm)?)),
            basis::shell_types::ContractionType::Segmented => Ok(AedelstenContext::Segmented(AedelstenContextImpl::<basis::contraction::Segmented>::new(input, comm)?)),
            basis::shell_types::ContractionType::Primitive => Ok(AedelstenContext::Primitive(AedelstenContextImpl::<basis::contraction::Uncontracted>::new(input, comm)?)),
        }
    }
}

// A forwarder's body: the method of the same name on whichever contraction kind the context holds.
macro_rules! dispatch {
    ($self:ident, $method:ident($($arg:expr),*)) => {
        match $self {
            AedelstenContext::General(ctx) => ctx.$method($($arg),*),
            AedelstenContext::Segmented(ctx) => ctx.$method($($arg),*),
            AedelstenContext::Primitive(ctx) => ctx.$method($($arg),*),
        }
    };
}

pub(crate) mod basis_set;
pub(crate) mod embedding;
pub(crate) mod gradients;
pub(crate) mod setup;
pub(crate) mod two_electron;
pub(crate) mod model;
pub(crate) mod one_electron;
pub(crate) mod point_group;
pub(crate) mod relativity;
pub(crate) mod scratch_need;

/// An error unless `output` holds an `expected_n x expected_n` matrix.
pub(crate) fn check_matrix_dimensions(expected_n: usize, output: &[f64]) -> Result<(), AedelstenError> {
    if output.len() != expected_n * expected_n {
        return Err(AedelstenError::InvalidMatrixDimensions {
            expected: expected_n * expected_n,
            actual: output.len(),
        });
    }
    Ok(())
}

/// An error unless `output` holds `n_planes` planes of `expected_n x expected_n`.
pub(crate) fn check_array3d_dimensions(expected_n: usize, n_planes: usize, output: &[f64]) -> Result<(), AedelstenError> {
    let expected = expected_n * expected_n * n_planes;
    if output.len() != expected {
        return Err(AedelstenError::InvalidMatrixDimensions {
            expected,
            actual: output.len(),
        });
    }
    Ok(())
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    pub fn new(input: &str, comm: &aedelsten_mpi::Communicator) -> Result<Self, AedelstenError> {
        let input = basis::setup_basis_system::<S>(&input)?;
        let qm_region = QmRegion::new(input)?;

        Ok(Self {
            qm_region: AedelstenCacheState::Ready(qm_region),
            shell_pairs: AedelstenCacheState::Empty,
            scratch: AedelstenCacheState::Empty,
            external_multipoles: AedelstenCacheState::Empty,
            multipole_convention: basis::charge_carriers::MultipoleConvention::default(),
            polarizable_sites: AedelstenCacheState::Empty,
            polarization: AedelstenCacheState::Empty,
            embedding_aimp_operators: AedelstenCacheState::Empty,
            primitive_shells: AedelstenCacheState::Empty,
            primitive_shell_pairs: AedelstenCacheState::Empty,
            contraction_matrix: AedelstenCacheState::Empty,
            eri_data: AedelstenCacheState::Empty,
            picture_change: AedelstenCacheState::Empty,
            x2c_frozen_density: AedelstenCacheState::Empty,
            symmetry: AedelstenCacheState::Empty,
            comm: comm.clone(),
        })
        .inspect(|ctx| {
            let shells = ctx.qm_region.get_cache().expect("input is set at construction").input.get_basis_shells();
            let spherical = shells.iter()
                .any(|shell| shell.get_lqn() >= 2 && matches!(shell.get_harmonic_type(), basis::shell_types::HarmonicType::Spherical));
            crate::citations::context(spherical, ctx.has_core_aimp(), ctx.has_embedding_aimp());
        })
    }
}
