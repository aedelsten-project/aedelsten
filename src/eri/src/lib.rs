// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Two-electron repulsion integrals over Gaussian shells by the McMurchie-Davidson scheme (L. E. McMurchie and E. R.
//! Davidson, J. Comput. Phys. 26, 218 (1978)): the walks over shell-pair quartets that hand each batch to a sink
//! (`TensorChoice`: a stored tensor, dense or packed; `DirectFock`: a direct J/K accumulator), their screening data
//! (`EriData`), the first derivatives (`GradientSink`), the full, short- and long-range Coulomb operators, and the
//! spin-orbit integrals (`spin_orbit`).

#![allow(confusable_idents)]
#![allow(mixed_script_confusables)]

pub mod packed;
pub mod r_matrix;
mod batch;
mod distribute;
mod gradient;
pub mod need;
pub mod spin_orbit;
mod walks;

use arrays::array4d::*;
use crate::packed::*;
use aedelsten_mpi::Communicator;
use mmd::hermite_coulomb_integrals::{ShortRange, LongRange, FullRange};

pub use walks::{EriData, EriDataEntry};
pub use gradient::{GradientCentres, GradientScreening, GradientScreeningTerm, GradientSink, GradientTensors, IntegralDerivatives, QuartetAtoms};
pub use distribute::{bra_ket_weight, for_each_pair_slot, for_each_slot_quartet, with_merged_slots, EriBlock, SlotQuartet};

/// What a walk runs on: the shell pairs, the scratch its arena comes from (`W`: the whole scratch, the default, or
/// what is left of it after a caller's own buffers, `arrays::scratch::SubScratch`, e.g. the half transform's
/// row buffers) and the ranks it may be split over (`comm`; a walk whose output wants it is split over them and
/// summed, collectively).
pub struct ERIWorkStruct<'a, S: basis::contraction::ShellKind, W: arrays::scratch::ScratchSource = arrays::scratch::Scratch> {
    shell_pairs: &'a [basis::shell_pairs::ShellPair<S>],
    scratch: &'a mut W,
    comm: &'a Communicator,
}

impl<'a, S: basis::contraction::ShellKind, W: arrays::scratch::ScratchSource> ERIWorkStruct<'a, S, W> {
    /// The walks' input: `shell_pairs`, `scratch`, `comm`.
    pub fn new(shell_pairs: &'a [basis::shell_pairs::ShellPair<S>], scratch: &'a mut W, comm: &'a Communicator) -> Self {
        Self { shell_pairs, scratch, comm }
    }
}

/// The ERI walks of a work struct.
pub trait TwoElectronIntegrals {
    /// Every unique integral of the shell pairs into `output`, unscreened.
    fn compute_eri_stored<T>(&mut self, output: &mut T) -> Result<(), arrays::scratch::ArenaError>
    where
        T: TensorChoice;
    /// `compute_eri_stored` with the short-range operator `erfc(omega r) / r`.
    fn compute_eri_stored_sr<T>(&mut self, output: &mut T, omega: f64) -> Result<(), arrays::scratch::ArenaError>
    where
        T: TensorChoice;
    /// `compute_eri_stored` with the long-range operator `erf(omega r) / r`.
    fn compute_eri_stored_lr<T>(&mut self, output: &mut T, omega: f64) -> Result<(), arrays::scratch::ArenaError>
    where
        T: TensorChoice;

    /// The density-independent ERI data of the shell pairs (Hermite coefficients and Schwarz bounds), the shell pairs
    /// whose bound is below `schwarz_threshold` left out (`None` keeps every pair): made once, reused by every direct
    /// build (`compute_fock_direct_from_data` and its range forms) and gradient walk.
    fn derive_eri_data(&mut self, schwarz_threshold: Option<f64>) -> Result<EriData, arrays::scratch::ArenaError>;

    /// A direct Fock build from `eri_data`: only direct accumulators (`DirectFock`), never a stored tensor.
    /// `pairwise_threshold`: the quartet half of Cauchy-Schwarz screening, beside `derive_eri_data`'s shell-pair half.
    fn compute_fock_direct_from_data<T>(&mut self, output: &mut T, eri_data: &EriData, pairwise_threshold: Option<f64>) -> Result<(), arrays::scratch::ArenaError>
    where
        T: DirectFock;
    /// `compute_fock_direct_from_data` with the short-range operator `erfc(omega r) / r`.
    fn compute_fock_direct_from_data_sr<T>(&mut self, output: &mut T, eri_data: &EriData, omega: f64, pairwise_threshold: Option<f64>) -> Result<(), arrays::scratch::ArenaError>
    where
        T: DirectFock;
    /// `compute_fock_direct_from_data` with the long-range operator `erf(omega r) / r`.
    fn compute_fock_direct_from_data_lr<T>(&mut self, output: &mut T, eri_data: &EriData, omega: f64, pairwise_threshold: Option<f64>) -> Result<(), arrays::scratch::ArenaError>
    where
        T: DirectFock;
}

impl<'a, S: basis::contraction::ShellKind, W: arrays::scratch::ScratchSource> TwoElectronIntegrals for ERIWorkStruct<'a, S, W> {
    fn compute_eri_stored<T>(&mut self, output: &mut T) -> Result<(), arrays::scratch::ArenaError>
    where
        T: TensorChoice
    {
        walks::compute_eri_stored::<T, S, FullRange, _>(
            output,
            self.shell_pairs,
            &mut *self.scratch,
            self.comm,
            0.0
        )
    }

    fn compute_eri_stored_sr<T>(&mut self, output: &mut T, omega: f64) -> Result<(), arrays::scratch::ArenaError>
    where
        T: TensorChoice
    {
        walks::compute_eri_stored::<T, S, ShortRange, _>(
            output,
            self.shell_pairs,
            &mut *self.scratch,
            self.comm,
            omega
        )
    }

    fn compute_eri_stored_lr<T>(&mut self, output: &mut T, omega: f64) -> Result<(), arrays::scratch::ArenaError>
    where
        T: TensorChoice
    {
        walks::compute_eri_stored::<T, S, LongRange, _>(
            output,
            self.shell_pairs,
            &mut *self.scratch,
            self.comm,
            omega
        )
    }

    fn derive_eri_data(&mut self, schwarz_threshold: Option<f64>) -> Result<EriData, arrays::scratch::ArenaError> {
        walks::derive_eri_data::<S, _>(
            self.shell_pairs,
            &mut *self.scratch,
            schwarz_threshold
        )
    }

    fn compute_fock_direct_from_data<T>(&mut self, output: &mut T, eri_data: &EriData, pairwise_threshold: Option<f64>) -> Result<(), arrays::scratch::ArenaError>
    where
        T: DirectFock
    {
        walks::compute_fock_direct_from_data::<T, S, FullRange, _>(
            output,
            self.shell_pairs,
            eri_data,
            &mut *self.scratch,
            self.comm,
            0.0,
            pairwise_threshold
        )
    }

    fn compute_fock_direct_from_data_sr<T>(&mut self, output: &mut T, eri_data: &EriData, omega: f64, pairwise_threshold: Option<f64>) -> Result<(), arrays::scratch::ArenaError>
    where
        T: DirectFock
    {
        walks::compute_fock_direct_from_data::<T, S, ShortRange, _>(
            output,
            self.shell_pairs,
            eri_data,
            &mut *self.scratch,
            self.comm,
            omega,
            pairwise_threshold
        )
    }

    fn compute_fock_direct_from_data_lr<T>(&mut self, output: &mut T, eri_data: &EriData, omega: f64, pairwise_threshold: Option<f64>) -> Result<(), arrays::scratch::ArenaError>
    where
        T: DirectFock
    {
        walks::compute_fock_direct_from_data::<T, S, LongRange, _>(
            output,
            self.shell_pairs,
            eri_data,
            &mut *self.scratch,
            self.comm,
            omega,
            pairwise_threshold
        )
    }
}

/// The ERI gradient walks, on the same `EriData` as the direct walk (its pair list and screening; no derivative data
/// is kept in it), for every contraction type: `d/dA = (a/p) d/dP - d/dR`, `d/dP` an index shift of the R tensor.
/// `sink` receives the derivative integrals batch by batch (`GradientSink`). Unscreened: `EriData` from
/// `derive_eri_data(None)` and `pairwise_threshold: None`. `centres`: the atoms the all-centres walk differentiates
/// (`GradientCentres`: the gradient centres, less the frozen ones).
pub trait TwoElectronIntegralGradients {
    /// The derivatives with respect to every centre of `centres`.
    fn compute_eri_gradient<S: GradientSink>(&mut self, sink: &mut S, eri_data: &EriData, pairwise_threshold: Option<f64>, centres: GradientCentres<'_>) -> Result<(), arrays::scratch::ArenaError>;
    /// `compute_eri_gradient` with the short-range operator `erfc(omega r) / r`.
    fn compute_eri_gradient_sr<S: GradientSink>(&mut self, sink: &mut S, eri_data: &EriData, omega: f64, pairwise_threshold: Option<f64>, centres: GradientCentres<'_>) -> Result<(), arrays::scratch::ArenaError>;
    /// `compute_eri_gradient` with the long-range operator `erf(omega r) / r`.
    fn compute_eri_gradient_lr<S: GradientSink>(&mut self, sink: &mut S, eri_data: &EriData, omega: f64, pairwise_threshold: Option<f64>, centres: GradientCentres<'_>) -> Result<(), arrays::scratch::ArenaError>;
}

// Every contraction type: the kernels are generic over `gradient::GradientPair` (general, segmented, primitive).
impl<'a, S: basis::contraction::ShellKind> TwoElectronIntegralGradients for ERIWorkStruct<'a, S> {
    fn compute_eri_gradient<G: GradientSink>(&mut self, sink: &mut G, eri_data: &EriData, pairwise_threshold: Option<f64>, centres: GradientCentres<'_>) -> Result<(), arrays::scratch::ArenaError> {
        gradient::compute_eri_gradient_from_data::<G, FullRange, S>(sink, self.shell_pairs, eri_data, &mut self.scratch, self.comm, 0.0, pairwise_threshold, centres)
    }

    fn compute_eri_gradient_sr<G: GradientSink>(&mut self, sink: &mut G, eri_data: &EriData, omega: f64, pairwise_threshold: Option<f64>, centres: GradientCentres<'_>) -> Result<(), arrays::scratch::ArenaError> {
        gradient::compute_eri_gradient_from_data::<G, ShortRange, S>(sink, self.shell_pairs, eri_data, &mut self.scratch, self.comm, omega, pairwise_threshold, centres)
    }

    fn compute_eri_gradient_lr<G: GradientSink>(&mut self, sink: &mut G, eri_data: &EriData, omega: f64, pairwise_threshold: Option<f64>, centres: GradientCentres<'_>) -> Result<(), arrays::scratch::ArenaError> {
        gradient::compute_eri_gradient_from_data::<G, LongRange, S>(sink, self.shell_pairs, eri_data, &mut self.scratch, self.comm, omega, pairwise_threshold, centres)
    }
}

/// The sink of an ERI walk: what each computed integral, or batch, goes into.
pub trait TensorChoice {
    /// One unique integral `(ij|kl)`, standing for its 8 permutations.
    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64);

    /// One shell-quartet batch, whole. Default: the unique-integral walk, calling `distribute` once per unique
    /// integral. An accumulator that overrides it digests the stored block itself, through
    /// `for_each_slot_quartet`. Its scratch comes from `scratch`, the walk's arena (the quartet's scope: what it
    /// allocates is free again after this call); running out of it is the walk's error.
    fn distribute_block<S: basis::contraction::ShellKind>(
        &mut self,
        block: &EriBlock<'_, '_>,
        pair_ab: &basis::shell_pairs::ShellPair<S>,
        pair_cd: &basis::shell_pairs::ShellPair<S>,
        _scratch: &mut arrays::scratch::Arena<'_>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Self: Sized,
    {
        distribute::distribute_eri_batch(self, block.batch(), pair_ab, pair_cd);
        Ok(())
    }

    /// Called once before the walk (e.g. to zero an output the walk adds into). Default no-op.
    fn prepare(&mut self) {}

    /// Called once after the walk, on every path and every rank, before `allreduce_sum`. Default no-op.
    fn finish(&mut self) {}

    /// Whether the walks (`compute_fock_direct_from_data`, `compute_eri_stored`) may split their quartets over the
    /// ranks of the walk's communicator (each rank computing and distributing only its own share, then
    /// `allreduce_sum`) rather than every rank computing everything. The walk splits only when the communicator has
    /// more than one rank; a split walk is collective: every rank of the communicator must make it. Default `false`;
    /// the stored S1/S4/S8 tensors below opt in. `Serial` computes a rank-local result instead.
    fn wants_mpi_partition(&self) -> bool {
        false
    }

    /// Combines every rank's own contribution (disjoint, since the split is over shell-pair quartets, not AO
    /// positions) into the complete result on every rank of `comm`. Called once, right after the whole quartet loop
    /// finishes, only for a split walk. Default no-op.
    fn allreduce_sum(&mut self, _comm: &Communicator) {}
}

/// A direct Fock accumulator (J/K-type contraction with densities as each batch arrives): the only output
/// `compute_fock_direct_from_data` accepts.
pub trait DirectFock: TensorChoice {
    /// The densities the integrals are contracted with, for the walk's density-weighted screening; `None` (the
    /// default): the walk screens by the Schwarz bounds alone, as for an accumulator that contracts with something
    /// other than densities.
    fn screening_densities(&self) -> Option<ScreeningDensities<'_>> {
        None
    }

    /// The walk hands a batch with several slots over merged into function order (`with_merged_slots`); `false` (the
    /// default): every batch as it is.
    fn merged_slots(&self) -> bool {
        false
    }

    /// The functions a quartet must reach: the direct walk computes only the quartets one of whose four shells has its
    /// first function in this range; `None` (the default): every quartet.
    fn wanted_functions(&self) -> Option<std::ops::Range<usize>> {
        None
    }

    /// The direct walk digests by rows: a row is one bra pair AB with its ket pairs, the diagonal quartet first. A
    /// row may come in several pieces (contiguous runs of its kets, on one rank or several), each from its own
    /// `begin_row` to its own `end_row`, so `end_row` adds the piece's part to the output. The scratch (in `f64`) an
    /// accumulator keeps for a row whose shells A and B have `rows` functions together, held from `begin_row` to
    /// `end_row`; 0 (the default): every quartet through `distribute_block`.
    fn row_len(&self, _rows: usize) -> usize {
        0
    }

    /// `row_len` for a basis of `n` functions in place of the accumulator's own (the scratch query sizes a basis's
    /// rows on a small accumulator of the same kind); `row_len` is this at the accumulator's own. The default:
    /// `row_len`, for an accumulator whose rows do not depend on `n`.
    fn row_len_at(&self, rows: usize, _n: usize) -> usize {
        self.row_len(rows)
    }

    /// A row (or a piece of it) begins, before its first quartet that is computed; `row` is its scratch, `row_len`
    /// long, uninitialized.
    fn begin_row<S: basis::contraction::ShellKind>(&mut self, _pair_ab: &basis::shell_pairs::ShellPair<S>, _row: &mut [f64])
    where
        Self: Sized,
    {
    }

    /// The scratch one quartet's digestion takes; `()` for an accumulator that takes none.
    type Tiles<'s>;

    /// The scratch `distribute_block_in_row` takes for `block` of `(pair_ab|pair_cd)`, from `arena`: the only memory
    /// the digestion takes (the scratch query takes it alone).
    fn digestion_tiles<'s, S: basis::contraction::ShellKind>(
        &self,
        block: &EriBlock<'_, '_>,
        pair_ab: &basis::shell_pairs::ShellPair<S>,
        pair_cd: &basis::shell_pairs::ShellPair<S>,
        arena: &arrays::scratch::Arena<'s>,
    ) -> Result<Self::Tiles<'s>, arrays::scratch::ArenaError>
    where
        Self: Sized;

    /// One quartet of the row, with `tiles` (`digestion_tiles` of the same block).
    fn distribute_block_in_row<S: basis::contraction::ShellKind>(
        &mut self,
        block: &EriBlock<'_, '_>,
        pair_ab: &basis::shell_pairs::ShellPair<S>,
        pair_cd: &basis::shell_pairs::ShellPair<S>,
        row: &mut [f64],
        tiles: &mut Self::Tiles<'_>,
    ) where
        Self: Sized;

    /// The row (or a piece of it) ends, after its last computed quartet (not called for one with none): its part is
    /// added to the output.
    fn end_row<S: basis::contraction::ShellKind>(&mut self, _pair_ab: &basis::shell_pairs::ShellPair<S>, _row: &mut [f64])
    where
        Self: Sized,
    {
    }
}

/// The densities of a direct Fock build, column-major `(n, n)` over the walk's AOs: those that enter J and those
/// that enter K, `None` where there are fewer (or that part is not built); any number more that enter both, back to
/// back in `both` (a multiple of `n^2` long); and the coefficients of the pair densities whose Coulomb matrices the
/// same walk makes, if any. A quartet is computed if any of them needs it.
pub struct ScreeningDensities<'a> {
    /// The walk's AOs.
    pub n: usize,
    /// The densities that enter J.
    pub coulomb: [Option<&'a [f64]>; 2],
    /// The densities that enter K.
    pub exchange: [Option<&'a [f64]>; 3],
    /// Densities that enter both, back to back.
    pub both: Option<&'a [f64]>,
    /// The coefficients of the pair densities.
    pub pair_densities: Option<ScreeningCoefficients<'a>>,
}

/// The coefficients of the pair densities whose Coulomb matrices an accumulator makes from the walk (the active-space
/// accumulators'): the bra's and the ket's, column-major `(n, m)` (the same slice in one basis). A quartet adds to
/// `K^T M` at most its Schwarz bound times `max(max(K_A, K_B) P_CD, max(K_C, K_D) P_AB)`, `P_XY = max(B_X K_Y, B_Y
/// K_X)`, with the largest coefficient on each shell; it needs the quartet when that is at least `threshold`.
pub struct ScreeningCoefficients<'a> {
    /// The coefficients' columns.
    pub m: usize,
    /// The bra's coefficients.
    pub bra: &'a [f64],
    /// The ket's coefficients.
    pub ket: &'a [f64],
    /// The threshold on what a quartet adds to `K^T M`.
    pub threshold: f64,
}

/// `output`, computed by the serial driver whatever the number of MPI ranks: a rank-local result, from a call that
/// need not be collective.
pub struct Serial<'a, T: TensorChoice>(pub &'a mut T);

impl<T: TensorChoice> TensorChoice for Serial<'_, T> {
    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64) {
        self.0.distribute(i, j, k, l, val);
    }

    fn distribute_block<S: basis::contraction::ShellKind>(
        &mut self,
        block: &EriBlock<'_, '_>,
        pair_ab: &basis::shell_pairs::ShellPair<S>,
        pair_cd: &basis::shell_pairs::ShellPair<S>,
        scratch: &mut arrays::scratch::Arena<'_>,
    ) -> Result<(), arrays::scratch::ArenaError> {
        self.0.distribute_block(block, pair_ab, pair_cd, scratch)
    }

    fn prepare(&mut self) {
        self.0.prepare();
    }

    fn finish(&mut self) {
        self.0.finish();
    }
}

impl<T: DirectFock> DirectFock for Serial<'_, T> {
    fn screening_densities(&self) -> Option<ScreeningDensities<'_>> {
        self.0.screening_densities()
    }

    fn merged_slots(&self) -> bool {
        self.0.merged_slots()
    }

    fn row_len(&self, rows: usize) -> usize {
        self.0.row_len(rows)
    }

    fn row_len_at(&self, rows: usize, n: usize) -> usize {
        self.0.row_len_at(rows, n)
    }

    fn begin_row<S: basis::contraction::ShellKind>(&mut self, pair_ab: &basis::shell_pairs::ShellPair<S>, row: &mut [f64]) {
        self.0.begin_row(pair_ab, row);
    }

    type Tiles<'s> = T::Tiles<'s>;

    fn digestion_tiles<'s, S: basis::contraction::ShellKind>(
        &self,
        block: &EriBlock<'_, '_>,
        pair_ab: &basis::shell_pairs::ShellPair<S>,
        pair_cd: &basis::shell_pairs::ShellPair<S>,
        arena: &arrays::scratch::Arena<'s>,
    ) -> Result<Self::Tiles<'s>, arrays::scratch::ArenaError> {
        self.0.digestion_tiles(block, pair_ab, pair_cd, arena)
    }

    fn distribute_block_in_row<S: basis::contraction::ShellKind>(
        &mut self,
        block: &EriBlock<'_, '_>,
        pair_ab: &basis::shell_pairs::ShellPair<S>,
        pair_cd: &basis::shell_pairs::ShellPair<S>,
        row: &mut [f64],
        tiles: &mut Self::Tiles<'_>,
    ) {
        self.0.distribute_block_in_row(block, pair_ab, pair_cd, row, tiles)
    }

    fn end_row<S: basis::contraction::ShellKind>(&mut self, pair_ab: &basis::shell_pairs::ShellPair<S>, row: &mut [f64]) {
        self.0.end_row(pair_ab, row);
    }
}

/// `allreduce_sum` of the stored tensors.
fn allreduce_tensor(comm: &Communicator, data: &mut [f64]) {
    if let Err(e) = comm.sum(data) {
        panic!("MPI allreduce failed while combining the stored ERI tensor across ranks: {e}");
    }
}

macro_rules! impl_tensor_choice {
    (4D, $Tensor: ty) => {
        impl TensorChoice for $Tensor {
            fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64) {
                // The 8 permutations of (ij|kl), chemists' notation.
                self[(i, j, k, l)] = val;
                self[(i, j, l, k)] = val;
                self[(j, i, k, l)] = val;
                self[(j, i, l, k)] = val;
                self[(k, l, i, j)] = val;
                self[(l, k, i, j)] = val;
                self[(k, l, j, i)] = val;
                self[(l, k, j, i)] = val;
            }

            // Every position is written by exactly one rank under MPI; the others must hold 0 for the sum.
            fn prepare(&mut self) {
                self.get_data_mut().fill(0.0);
            }

            fn wants_mpi_partition(&self) -> bool {
                true
            }

            fn allreduce_sum(&mut self, comm: &Communicator) {
                allreduce_tensor(comm, self.get_data_mut());
            }
        }
    };
}
impl_tensor_choice!(4D, Array4D<f64>);
impl_tensor_choice!(4D, Array4DViewMut<'_, f64>);

// The packed tensors, generic over the packing order (`RowMajor`, `ColumnMajor`) and the symmetry (S8, S4):
// `AOSymmetry::distribute` writes every stored position the symmetry implies for (i,j,k,l), one for S8, the (ij|kl)
// and (kl|ij) pair for S4.
impl<O: IndexOrder, S: AOSymmetry<O>> TensorChoice for PackedERITensor<f64, O, S> {
    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64) {
        S::distribute(self, i, j, k, l, val);
    }

    // As the dense tensors': zeroed for the sum, partitioned under MPI.
    fn prepare(&mut self) {
        self.get_data_mut().fill(0.0);
    }

    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        allreduce_tensor(comm, self.get_data_mut());
    }
}
impl<O: IndexOrder, S: AOSymmetry<O>> TensorChoice for PackedERITensorViewMut<'_, f64, O, S> {
    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64) {
        S::distribute(self, i, j, k, l, val);
    }

    // As the dense tensors': zeroed for the sum, partitioned under MPI.
    fn prepare(&mut self) {
        self.get_data_mut().fill(0.0);
    }

    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        allreduce_tensor(comm, self.get_data_mut());
    }
}
