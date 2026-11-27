// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The library's public surface: everything a Rust caller can name is re-exported here, so this list is the whole Rust
// API. `AedelstenContext` (src/context/) is the entry point: one input's basis, geometry and cached state, and every
// computation over them, one file per topic. The C ABI (src/ffi/) mirrors the context topic by topic; the Fortran,
// Python and Julia bindings (src/interfaces/) stand on the C ABI.

mod citations;
mod context;
mod error;
mod ffi;
mod timing;
mod wire_enum;

pub use context::AedelstenContext;

/// The library's version (`<major>.<minor>.<patch>`), every crate of it sharing one; a release is tagged
/// `v<version>`.
pub fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The git commit the library was built from (12 hex digits), `None` when its source tree was no git checkout (an
/// exported tree, or no git where it was built).
pub fn get_source_commit() -> Option<&'static str> {
    option_env!("AEDELSTEN_SOURCE_COMMIT")
}
pub use context::scratch_need::{OperationNeed, ShellShape, ScratchItem, ScratchOperation, SCRATCH_BREAKDOWN_FIELDS, SCRATCH_OPERATION_FIELDS};
pub use context::setup::MEMORY_BREAKDOWN_FIELDS;
// The array types of the public API, the crate whole.
pub use arrays;
pub use eri::packed;
pub use eri::packed::{compute_s4_size, compute_s8_size};
pub use context::two_electron::{ERIAOSymmetry, ERIIndexOrder, ERIRangeOperator};
pub use context::embedding::{SiteSelection, StaticFieldSources};
pub use crate::error::AedelstenError;
pub use context::embedding::{InductionReport, PolarizationOptions, PolarizationStorage, StaticFieldPolicy};
pub use context::basis_set::{rescale_columns, rescale_rows};
pub use cart2sph::SharedCartesianReference;
pub use polarization::{Damping, SolverOptions};
pub use efield::{SiteChunking, resolve_field_chunk_size};
pub use context::one_electron::OneElectronOperator;
pub use crate::timing::{mpi_collective_seconds, profile_enabled, profile_report, timing_report, TimingSpan};
pub use symmetry::{Irrep, OperatorSymmetry, OrbitalLabel, PointGroupRequest, SymmetrisationReport};
pub use context::point_group::{Cleaning, SymmetryOptions, DEFAULT_STRICT_TOLERANCE};
pub use oneint::MomentOrigin;
pub use basis::shell_types::ContractionType;
pub use basis::charge_carriers::{FiniteModel, MultipoleConvention, Site};
pub use basis::ao_values::ShellInfo;
pub use basis::{AtomKind, BasisFunctionId, CentreGeometry, CentreLabel, EmbeddingAimpSite, Primitive, PrimitiveId};
pub use basis::model_fockian::ModelFunction;
pub use basis::sap::SapFit;
// The dense (stored-tensor) Fock builders are plain functions over arrays and stay public. The direct accumulators
// and the walk's sink traits are internal: the direct builds are `AedelstenContext` methods.
pub use ::fock_constructors::{TwoElectronTerm, TwoElectronTermKind};
/// The dense J and K builders from a stored S1 tensor.
pub mod fock_constructors {
    pub use ::fock_constructors::{
        construct_j_and_k_stored, construct_j_and_k_unrestricted_stored, construct_j_stored, construct_k_stored,
        construct_k_unrestricted_stored,
    };
}
