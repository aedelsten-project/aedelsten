// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The scratch the context's operations take, from the input alone: each item's memory taken by its own constructors
// on probe pairs of the basis's shell shapes (`basis::probes`, `eri::need`), nothing computed, with the tables a walk
// holds for the whole basis at the basis's size. The same on every range of the two-electron operator.

use crate::*;
use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use arrays::scratch::{Scratch, ByteUnit};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;

use super::AedelstenContextImpl;

/// An operation the scratch is sized for (`AedelstenContext::scratch_bytes`), on any range of the two-electron
/// operator (the same memory).
#[derive(Clone, Copy, Debug)]
pub enum ScratchOperation {
    /// The Hermite coefficients and Schwarz bounds of the direct walks (`set_eri_data`).
    EriData,
    /// A direct restricted J and K (`compute_fock_direct_restricted`), screened.
    RestrictedFock,
    /// A direct unrestricted J and K (`compute_fock_direct_unrestricted`), screened.
    UnrestrictedFock,
    /// The two-electron gradient (`compute_two_electron_gradient`) of `terms` density products, screened.
    TwoElectronGradient { terms: usize },
    /// J and K of a density of any symmetry (`compute_fock_direct_general`), screened.
    GeneralFock,
    /// J and K of `densities` densities in one walk (`compute_fock_direct_general_multi`), screened.
    GeneralMultiFock { densities: usize },
    /// A CASSCF macro-iteration's walk over `active` orbitals (`compute_fock_direct_active_space`), screened.
    ActiveSpaceFock { active: usize },
    /// A state interaction's walk over `active` orbitals in two bases (`compute_fock_direct_mixed_active_space`).
    MixedActiveSpaceFock { active: usize },
    /// The half transform (`compute_eri_half_transformed` and its `_on_node` and `_sigma` forms) of `left` orbitals,
    /// and `right` ones in a second basis (`None`: one basis), for a batch of `sigma` functions (`None`: all), on
    /// `node_ranks` ranks of one node (1 for the rank-local forms): its buffers
    /// (`half_transform_scratch_bytes`) and the walk after them.
    HalfTransform { left: usize, right: Option<usize>, sigma: Option<usize>, node_ranks: usize },
    /// The AMFI integrals (`compute_amfi_integrals`): the largest atom kind's mean field, over the primitive shells
    /// whatever `primitive` says.
    Amfi,
}

/// The C interface's operations: rows of `SCRATCH_OPERATION_FIELDS` int32, a kind and up to four parameters
/// (unused: any value; `None`: -1): 1 `EriData`, 2 `RestrictedFock`, 3 `UnrestrictedFock`, 4 `TwoElectronGradient`
/// (terms), 5 `GeneralFock`, 6 `GeneralMultiFock` (densities), 7 `ActiveSpaceFock` (active), 8 `MixedActiveSpaceFock`
/// (active), 9 `HalfTransform` (left, right, sigma, node_ranks), 10 `Amfi`.
pub const SCRATCH_OPERATION_FIELDS: usize = 5;

impl ScratchOperation {
    /// The operation of a C row (`SCRATCH_OPERATION_FIELDS`); `None` for an unknown kind or a parameter out of range.
    pub fn from_c(row: [i32; SCRATCH_OPERATION_FIELDS]) -> Option<Self> {
        let count = |value: i32| usize::try_from(value).ok();
        let optional = |value: i32| if value == -1 { Some(None) } else { count(value).map(Some) };
        Some(match row[0] {
            1 => ScratchOperation::EriData,
            2 => ScratchOperation::RestrictedFock,
            3 => ScratchOperation::UnrestrictedFock,
            4 => ScratchOperation::TwoElectronGradient { terms: count(row[1])? },
            5 => ScratchOperation::GeneralFock,
            6 => ScratchOperation::GeneralMultiFock { densities: count(row[1])? },
            7 => ScratchOperation::ActiveSpaceFock { active: count(row[1])? },
            8 => ScratchOperation::MixedActiveSpaceFock { active: count(row[1])? },
            9 => ScratchOperation::HalfTransform {
                left: count(row[1])?,
                right: optional(row[2])?,
                sigma: optional(row[3])?,
                node_ranks: count(row[4]).filter(|&ranks| ranks > 0)?,
            },
            10 => ScratchOperation::Amfi,
            _ => return None,
        })
    }
}

/// The C interface's breakdown: per operation a row of these int64 fields (`OperationNeed::c_fields`), the shells'
/// absent in a pair's item -1.
pub const SCRATCH_BREAKDOWN_FIELDS: [&std::ffi::CStr; 18] = [
    c"bytes",
    c"buffers",
    c"tables",
    c"row",
    c"item_bytes",
    c"item",
    c"l_1",
    c"primitives_1",
    c"contracted_1",
    c"l_2",
    c"primitives_2",
    c"contracted_2",
    c"l_3",
    c"primitives_3",
    c"contracted_3",
    c"l_4",
    c"primitives_4",
    c"contracted_4",
];

impl OperationNeed {
    /// The row of `SCRATCH_BREAKDOWN_FIELDS`.
    pub fn c_fields(&self) -> [i64; SCRATCH_BREAKDOWN_FIELDS.len()] {
        let mut row = [-1; SCRATCH_BREAKDOWN_FIELDS.len()];
        row[..6].copy_from_slice(&[self.bytes as i64, self.buffers as i64, self.tables as i64, self.row as i64, self.item_bytes as i64, self.item as i64]);
        for (fields, shell) in row[6..].chunks_exact_mut(3).zip(&self.shells) {
            fields.copy_from_slice(&[shell.l as i64, shell.primitives as i64, shell.contracted as i64]);
        }
        row
    }
}

impl AedelstenContext {
    /// The bytes of scratch that every operation in `operations` fits in, on this context's basis: each operation's
    /// buffers taken by its own code, nothing computed, on a pair of every two shell shapes the basis has (all their
    /// primitive products kept, so never fewer than an overlap threshold leaves), with the tables a walk holds for the
    /// whole basis at its size. Needs the input only: called before the shell pairs are set, to size the scratch
    /// once. `primitive`: the same operations on the basis's primitive blocks (as `set_primitive_shell_pairs` makes
    /// them, the primitive-basis operations' and the primitive kind's), which need at least what the contracted shells
    /// do, so this covers both.
    pub fn scratch_bytes(&self, operations: &[ScratchOperation], primitive: bool) -> Result<usize, AedelstenError> {
        Ok(self.scratch_breakdown(operations, primitive)?.iter().map(|need| need.bytes).max().unwrap_or(0))
    }

    /// `scratch_bytes` operation by operation, in the order of `operations`: each one's need and what sets it, for
    /// finding which item to make leaner. `scratch_bytes` is the largest `bytes` of these.
    pub fn scratch_breakdown(&self, operations: &[ScratchOperation], primitive: bool) -> Result<Vec<OperationNeed>, AedelstenError> {
        dispatch!(self, scratch_breakdown(operations, primitive))
    }
}

/// An operation's need (`AedelstenContext::scratch_breakdown`) and what sets it: `bytes` is the sum of `buffers`
/// (the half transform's, taken before its walk; 0 otherwise), `tables` (the walk's for the whole basis), `row` (a
/// walk row's own memory; 0 for a pair's item) and `item_bytes` (the item's), on the item's shells (two for a pair's
/// item, the bra's and the ket's for a quartet).
#[derive(Clone, Debug)]
pub struct OperationNeed {
    pub operation: ScratchOperation,
    pub bytes: usize,
    pub buffers: usize,
    pub tables: usize,
    pub row: usize,
    pub item_bytes: usize,
    pub item: ScratchItem,
    pub shells: Vec<ShellShape>,
}

/// The item that sets an operation's need. The values are the C interface's codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ScratchItem {
    /// A pair's ERI data entry: its Hermite coefficients and Schwarz bound.
    EriDataEntry = 1,
    /// A pair's data for the gradient walk.
    GradientPairData = 2,
    /// A pair's derivative Schwarz bound.
    DerivativeBound = 3,
    /// A quartet of a walk's row: its batch and digestion, or its gradient buffers.
    Quartet = 4,
    /// An atom kind's AMFI mean field: its pairs' coefficients and the largest of its quartets.
    AmfiAtomKind = 5,
}

/// A shell of an item: its angular momentum and its primitive and contracted functions' counts (for the primitive
/// query, a primitive block's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShellShape {
    pub l: usize,
    pub primitives: usize,
    pub contracted: usize,
}

impl<S: ShellKind> AedelstenContextImpl<S> {
    pub fn scratch_breakdown(&self, operations: &[ScratchOperation], primitive: bool) -> Result<Vec<OperationNeed>, AedelstenError> {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let shells = qm_region.input.get_basis_shells();
        let amfi = if operations.iter().any(|operation| matches!(operation, ScratchOperation::Amfi)) { self.amfi_scratch_bytes()? } else { 0 };
        if primitive && !S::PRIMITIVE {
            let blocks = S::primitive_blocks(shells);
            let probes: Vec<ShellPair<basis::contraction::Uncontracted>> = basis::probes::probe_pairs(&blocks, [0, 1], [0.0; 3]).into_iter().map(|(pair, _)| pair).collect();
            let basis = BasisSize::of(&blocks, qm_region.input.get_n_prim_unique());
            return operations_bytes(&probes, &basis, operations, amfi);
        }
        let probes: Vec<ShellPair<S>> = basis::probes::probe_pairs(shells, [0, 1], [0.0; 3]).into_iter().map(|(pair, _)| pair).collect();
        operations_bytes(&probes, &BasisSize::of(shells, self.get_n_cont()), operations, amfi)
    }

    // The AMFI integrals' scratch, over the primitive shells `set_primitive_shell_pairs` makes (the basis itself for a
    // primitive kind).
    fn amfi_scratch_bytes(&self) -> Result<usize, AedelstenError> {
        let input = &self.qm_region.get_cache().expect("input has been freed (free_input)").input;
        let shells = input.get_basis_shells();
        if S::PRIMITIVE {
            return Ok(amfi::amfi_scratch_bytes(input, S::as_uncontracted(&[], shells).expect("a primitive kind").1)?);
        }
        let (primitive_shells, _) = S::decontract(shells, input.get_basis_info())?;
        Ok(amfi::amfi_scratch_bytes(input, &primitive_shells)?)
    }
}

/// The sizes of a basis the walks' tables and the half transform's buffers follow.
struct BasisSize {
    n: usize,
    n_shells: usize,
    max_shell_functions: usize,
}

impl BasisSize {
    fn of<K: ShellKind>(shells: &[basis::shell_types::BasisShell<K>], n: usize) -> Self {
        let max_shell_functions = shells.iter().map(|shell| shell.get_n_cont() * shell.get_degeneracy()).max().unwrap_or(0);
        BasisSize { n, n_shells: shells.len(), max_shell_functions }
    }
}

/// The half transform's buffers in bytes (`AedelstenContext::half_transform_scratch_bytes`): its row buffers and
/// transposed coefficients over `n` functions whose largest shell has `max_shell_functions`, and on `node_ranks` ranks
/// of one node the bounds of the `n_shells` shells and the most tiles of either kind of walk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn half_transform_bytes(n: usize, max_shell_functions: usize, n_shells: usize, m_left: usize, m_right: usize, one_basis: bool, n_sigma: usize, node_ranks: usize) -> usize {
    let doubles = ::fock_constructors::HalfTransformedEri::sigma_row_buffer_len(n, n_sigma, m_left, m_right, max_shell_functions, one_basis);
    let bounds = if node_ranks > 1 {
        let tiles = |symmetric| ::fock_constructors::HalfTransformedEri::node_buffer_len(n_shells, n_shells, node_ranks, symmetric);
        n_shells + 1 + tiles(true).max(tiles(false))
    } else {
        0
    };
    8 * doubles + std::mem::size_of::<usize>() * bounds
}

/// `operations`' needs, measured on `probes` for a basis of `basis`'s size; `amfi` the AMFI integrals' (measured on
/// the atom kinds).
fn operations_bytes<K: ShellKind>(probes: &[ShellPair<K>], basis: &BasisSize, operations: &[ScratchOperation], amfi: usize) -> Result<Vec<OperationNeed>, AedelstenError> {
    let (n, n_shells) = (basis.n, basis.n_shells);
    let n_probe = probes.iter().map(|pair| pair.get_idx_b() + pair.get_n_cont_b() * pair.get_degeneracy_b()).max().unwrap_or(0);
        let mut ws = Scratch::new(1, ByteUnit::MiB)?;
        let (eri_data_need, coefficients) = eri::need::eri_data(probes, &mut ws)?;
        let mut needs = Vec::with_capacity(operations.len());
        for operation in operations {
            let mut buffers = 0;
            let need = match *operation {
                ScratchOperation::Amfi => {
                    needs.push(OperationNeed { operation: *operation, bytes: amfi, buffers: 0, tables: 0, row: 0, item_bytes: amfi, item: ScratchItem::AmfiAtomKind, shells: Vec::new() });
                    continue;
                }
                ScratchOperation::EriData => eri_data_need,
                ScratchOperation::RestrictedFock => {
                    let (d, mut j, mut k) = (Matrix::<f64>::new(n_probe, n_probe), Matrix::<f64>::new(n_probe, n_probe), Matrix::<f64>::new(n_probe, n_probe));
                    let output = ::fock_constructors::RestrictedDirectFock::new(d.view(), j.view_mut(), k.view_mut(), 1.0, -0.5);
                    eri::need::direct_fock(&output, probes, &coefficients, n, n_shells, true, &mut ws)?
                }
                ScratchOperation::UnrestrictedFock => {
                    let densities = [(); 3].map(|_| Matrix::<f64>::new(n_probe, n_probe));
                    let mut outputs = [(); 3].map(|_| Matrix::<f64>::new(n_probe, n_probe));
                    let [j, k_alpha, k_beta] = &mut outputs;
                    let output = ::fock_constructors::UnrestrictedDirectFock::new(
                        densities[0].view(), densities[1].view(), densities[2].view(),
                        j.view_mut(), k_alpha.view_mut(), k_beta.view_mut(), 1.0, -1.0,
                    );
                    eri::need::direct_fock(&output, probes, &coefficients, n, n_shells, true, &mut ws)?
                }
                ScratchOperation::TwoElectronGradient { terms } => {
                    let density = vec![0.0; n_probe * n_probe];
                    let kinds = [::fock_constructors::TwoElectronTermKind::Coulomb, ::fock_constructors::TwoElectronTermKind::Exchange];
                    let terms: Vec<_> = (0..terms).map(|t| ::fock_constructors::TwoElectronTerm { kind: kinds[t % 2], left: &density, right: &density, coeff: 1.0 }).collect();
                    // The probe pairs' two atoms.
                    let mut gradient = [0.0; 6];
                    let mut sink = ::fock_constructors::DirectEriGradient::new(&terms, n_probe, &mut gradient);
                    eri::need::eri_gradient::<_, K>(&mut sink, probes, n_shells, true, &mut ws)?
                }
                ScratchOperation::GeneralFock => {
                    let (d, mut j, mut k) = (Matrix::<f64>::new(n_probe, n_probe), Matrix::<f64>::new(n_probe, n_probe), Matrix::<f64>::new(n_probe, n_probe));
                    let output = ::fock_constructors::GeneralDirectFock::new(d.view(), j.view_mut(), k.view_mut(), 1.0, -0.5);
                    eri::need::direct_fock(&output, probes, &coefficients, n, n_shells, true, &mut ws)?
                }
                ScratchOperation::GeneralMultiFock { densities } => {
                    let d = vec![0.0; densities.max(1) * n_probe * n_probe];
                    let (mut j, mut k) = (d.clone(), d.clone());
                    let output = ::fock_constructors::GeneralMultiDirectFock::new(n_probe, &d, &mut j, &mut k, 1.0, -0.5);
                    eri::need::direct_fock(&output, probes, &coefficients, n, n_shells, true, &mut ws)?
                }
                ScratchOperation::ActiveSpaceFock { active } => {
                    let densities = [(); 2].map(|_| Matrix::<f64>::new(n_probe, n_probe));
                    let mut outputs = [(); 4].map(|_| Matrix::<f64>::new(n_probe, n_probe));
                    let [j_inactive, k_inactive, j_active, k_active] = &mut outputs;
                    let c_active = Matrix::<f64>::new(n_probe, active);
                    let mut m = vec![0.0; eri::packed::packed_pair_tensor_len(n_probe, active * (active + 1) / 2)];
                    let jk = ::fock_constructors::InactiveActiveDirectFock::new(
                        densities[0].view(), densities[1].view(),
                        j_inactive.view_mut(), k_inactive.view_mut(), j_active.view_mut(), k_active.view_mut(), 1.0, -0.5, 1.0, -0.5,
                    );
                    let m = eri::packed::PackedPairTensorViewMut::new(n_probe, active * (active + 1) / 2, &mut m);
                    let output = ::fock_constructors::ActiveSpaceDirectFock::new(jk, c_active.view(), m, None);
                    eri::need::direct_fock(&output, probes, &coefficients, n, n_shells, true, &mut ws)?
                }
                ScratchOperation::MixedActiveSpaceFock { active } => {
                    let (d, mut j, mut k) = (Matrix::<f64>::new(n_probe, n_probe), Matrix::<f64>::new(n_probe, n_probe), Matrix::<f64>::new(n_probe, n_probe));
                    let c = Matrix::<f64>::new(n_probe, active);
                    let mut m = vec![0.0; eri::packed::packed_pair_tensor_len(n_probe, active * active)];
                    let jk = ::fock_constructors::GeneralDirectFock::new(d.view(), j.view_mut(), k.view_mut(), 1.0, -0.5);
                    let m = eri::packed::PackedPairTensorViewMut::new(n_probe, active * active, &mut m);
                    let output = ::fock_constructors::ActiveSpaceDirectFock::mixed(jk, c.view(), c.view(), m, None);
                    eri::need::direct_fock(&output, probes, &coefficients, n, n_shells, true, &mut ws)?
                }
                ScratchOperation::HalfTransform { left, right, sigma, node_ranks } => {
                    // The walk takes no scratch per batch from the accumulator, so one orbital per side stands in for
                    // the real counts; its buffers come off the front of the scratch at the real basis's size.
                    let probe_shell_functions = probes.iter().map(|pair| pair.get_n_cont_a() * pair.get_degeneracy_a()).max().unwrap_or(0);
                    let c = Matrix::<f64>::new(n_probe, 1);
                    let mut x = Matrix::<f64>::new(n_probe, n_probe);
                    let buffer_bytes = 8 * ::fock_constructors::HalfTransformedEri::row_buffer_len(n_probe, 1, 1, probe_shell_functions, right.is_none());
                    let mut accumulator_buffers = Scratch::new(buffer_bytes.div_ceil(1024) + 1, ByteUnit::KiB)?;
                    let arena = accumulator_buffers.get_arena();
                    let output = match right {
                        None => ::fock_constructors::HalfTransformedEri::new(c.view(), x.view_mut(), None, false, probe_shell_functions, &arena)?,
                        Some(_) => ::fock_constructors::HalfTransformedEri::mixed(c.view(), c.view(), x.view_mut(), None, false, probe_shell_functions, &arena)?,
                    };
                    let walk = eri::need::direct_fock(&output, probes, &coefficients, n, n_shells, true, &mut ws)?;
                    let one_basis = right.is_none();
                    buffers = half_transform_bytes(n, basis.max_shell_functions, basis.n_shells, left, right.unwrap_or(0), one_basis, sigma.unwrap_or(n), node_ranks);
                    walk
                }
            };
            needs.push(operation_need(*operation, buffers, need, probes));
        }
        Ok(needs)
}

/// An operation's need from its walk's (`need`; `None` for no probes) and its `buffers` before the walk.
fn operation_need<K: ShellKind>(operation: ScratchOperation, buffers: usize, need: Option<eri::need::Need>, probes: &[ShellPair<K>]) -> OperationNeed {
    let Some(need) = need else {
        return OperationNeed { operation, bytes: buffers, buffers, tables: 0, row: 0, item_bytes: 0, item: ScratchItem::EriDataEntry, shells: Vec::new() };
    };
    let shapes = |pair: &ShellPair<K>| {
        [
            ShellShape { l: pair.get_lqn_a(), primitives: pair.get_n_prim_a(), contracted: pair.get_n_cont_a() },
            ShellShape { l: pair.get_lqn_b(), primitives: pair.get_n_prim_b(), contracted: pair.get_n_cont_b() },
        ]
    };
    let mut shells = shapes(&probes[need.pair]).to_vec();
    if let Some(ket) = need.ket {
        shells.extend(shapes(&probes[ket]));
    }
    let item = match need.item {
        eri::need::NeedItem::EriDataEntry => ScratchItem::EriDataEntry,
        eri::need::NeedItem::GradientPairData => ScratchItem::GradientPairData,
        eri::need::NeedItem::DerivativeBound => ScratchItem::DerivativeBound,
        eri::need::NeedItem::Quartet => ScratchItem::Quartet,
    };
    OperationNeed { operation, bytes: buffers + need.bytes, buffers, tables: need.tables, row: need.row, item_bytes: need.item_bytes, item, shells }
}
