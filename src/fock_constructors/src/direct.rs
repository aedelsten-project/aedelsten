// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The direct J/K accumulators (`eri::DirectFock`) and the direct two-electron gradient (`eri::GradientSink`): J and
//! K are built from each shell-quartet batch as the walk computes it, with no tensor stored. The J/K accumulators
//! digest whole batches (`TensorChoice::distribute_block`) with the kernels of `digest` (the full stored block, the
//! slot weights of `eri::for_each_slot_quartet`, half-matrices completed in `finish`); their per-integral
//! `distribute` is never called. Each accumulator produces all its outputs (J and K, both spins, several densities)
//! from one walk, so that every integral is computed once. The gradient sink works per unique integral, scattered
//! over its distinct permutations (`unique_permutations`).

use aedelsten_mpi::Communicator;
use arrays::matrix::{MatrixMathMut, MatrixTrait, MatrixTraitMut, MatrixView, MatrixViewMut};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;
use eri::{EriBlock, GradientSink, IntegralDerivatives, QuartetAtoms, ScreeningDensities, TensorChoice};

use crate::digest::{
    begin_coulomb_row, begin_exchange_row, coulomb, coulomb_in_row, DigestTiles, coulomb_row_len, end_coulomb_row, end_exchange_row, exchange,
    exchange_in_row, exchange_row_len, row_functions, take_scratch,
};
use arrays::scratch::{Arena, ArenaError};

/// `distribute` of the J/K accumulators: they digest whole blocks.
fn per_integral_unused() -> ! {
    unreachable!("the direct J/K accumulators digest whole batches (distribute_block); distribute is never called")
}

/// `buf` summed over the ranks of `comm`, panicking with the matrix's name (`label`) if MPI fails: every
/// accumulator's `allreduce_sum` below.
pub(crate) fn allreduce_or_panic(comm: &Communicator, buf: &mut [f64], label: &str) {
    if let Err(e) = comm.sum(buf) {
        panic!("MPI allreduce failed while combining direct-Fock {label} across ranks: {e}");
    }
}

/// The 8 index quartets equivalent to (i,j,k,l) under the 8-fold symmetry of real ERIs.
#[inline(always)]
fn eightfold_permutations(i: usize, j: usize, k: usize, l: usize) -> [(usize, usize, usize, usize); 8] {
    [
        (i, j, k, l),
        (i, j, l, k),
        (j, i, k, l),
        (j, i, l, k),
        (k, l, i, j),
        (l, k, i, j),
        (k, l, j, i),
        (l, k, j, i),
    ]
}

/// The distinct index quartets of the 8-fold orbit of (i,j,k,l), sorted: fewer than 8 when i == j, k == l or
/// {i,j} == {k,l}. On the stack: `perms[..n]`.
fn unique_permutations(i: usize, j: usize, k: usize, l: usize) -> ([(usize, usize, usize, usize); 8], usize) {
    let mut perms = eightfold_permutations(i, j, k, l);
    perms.sort_unstable();
    let mut n = 1;
    for idx in 1..perms.len() {
        if perms[idx] != perms[n - 1] {
            perms[n] = perms[idx];
            n += 1;
        }
    }
    (perms, n)
}

/// Restricted (one density) direct J/K accumulator: `J = coeff_j J[p]`, `K = coeff_k K[p]`, `p` symmetric.
pub struct RestrictedDirectFock<'a> {
    p: MatrixView<'a, f64>,
    j: MatrixViewMut<'a, f64>,
    k: MatrixViewMut<'a, f64>,
    coeff_j: f64,
    coeff_k: f64,
}

impl<'a> RestrictedDirectFock<'a> {
    /// `j`/`k` are zeroed here. A range-separated build takes one walk per range operator, each into its own outputs.
    ///
    /// The densities and outputs of every accumulator here are views (`m.view()`, `m.view_mut()` of an owned
    /// matrix, a buffer from an arena, a column block of a larger matrix), so nothing is copied.
    pub fn new(p: MatrixView<'a, f64>, mut j: MatrixViewMut<'a, f64>, mut k: MatrixViewMut<'a, f64>, coeff_j: f64, coeff_k: f64) -> Self {
        j.fill(0.0);
        k.fill(0.0);
        RestrictedDirectFock { p, j, k, coeff_j, coeff_k }
    }
}

impl eri::DirectFock for RestrictedDirectFock<'_> {
    fn screening_densities(&self) -> Option<ScreeningDensities<'_>> {
        let p = self.p.get_data();
        Some(ScreeningDensities {
            n: self.p.rows(),
            coulomb: [(self.coeff_j != 0.0).then_some(p), None],
            exchange: [(self.coeff_k != 0.0).then_some(p), None, None],
            both: None,
            pair_densities: None,
        })
    }

    // J and K by rows (digest.rs): per density with a part, J's tiles and K's strips in the row's scratch, in order.
    fn row_len(&self, rows: usize) -> usize {
        self.row_len_at(rows, self.p.rows())
    }

    fn row_len_at(&self, rows: usize, n: usize) -> usize {
        usize::from(self.coeff_j != 0.0) * coulomb_row_len(rows)
            + usize::from(self.coeff_k != 0.0) * exchange_row_len(rows, n)
    }

    fn begin_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let (rows, n) = (row_functions(pair_ab), self.p.rows());
        if self.coeff_j != 0.0 {
            begin_coulomb_row(pair_ab, &self.p, take_scratch(&mut row, coulomb_row_len(rows)));
        }
        if self.coeff_k != 0.0 {
            begin_exchange_row(pair_ab, &self.p, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
    }

    type Tiles<'s> = ();

    fn digestion_tiles<'s, S: ShellKind>(&self, _block: &EriBlock<'_, '_>, _pair_ab: &ShellPair<S>, _pair_cd: &ShellPair<S>, _arena: &Arena<'s>) -> Result<(), ArenaError> {
        Ok(())
    }

    // In the row's scratch alone: no tiles.
    fn distribute_block_in_row<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, mut row: &mut [f64], _tiles: &mut ()) {
        let (rows, n) = (row_functions(pair_ab), self.p.rows());
        if self.coeff_j != 0.0 {
            coulomb_in_row(block, pair_ab, pair_cd, &self.p, &mut self.j, take_scratch(&mut row, coulomb_row_len(rows)), self.coeff_j);
        }
        if self.coeff_k != 0.0 {
            exchange_in_row(block, pair_ab, pair_cd, take_scratch(&mut row, exchange_row_len(rows, n)), self.coeff_k);
        }
    }

    fn end_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let (rows, n) = (row_functions(pair_ab), self.p.rows());
        if self.coeff_j != 0.0 {
            end_coulomb_row(pair_ab, &mut self.j, take_scratch(&mut row, coulomb_row_len(rows)));
        }
        if self.coeff_k != 0.0 {
            end_exchange_row(pair_ab, &mut self.k, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
    }
}

impl<'a> TensorChoice for RestrictedDirectFock<'a> {
    fn distribute(&mut self, _: usize, _: usize, _: usize, _: usize, _: f64) {
        per_integral_unused()
    }

    // J_ab = sum_cd D_cd (ab|cd); K_ac = sum_bd D_bd (ab|cd); D symmetric.
    fn distribute_block<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, _scratch: &mut Arena<'_>) -> Result<(), ArenaError> {
        if self.coeff_j != 0.0 {
            coulomb(block, pair_ab, pair_cd, &self.p, &mut self.j, self.coeff_j, None);
        }
        if self.coeff_k != 0.0 {
            exchange(block, pair_ab, pair_cd, &self.p, &mut self.k, self.coeff_k, None);
        }
        Ok(())
    }


    fn finish(&mut self) {
        self.j.symmetrize_in_place(1.0);
        self.k.symmetrize_in_place(1.0);
    }

    // Ranks own disjoint shell quartets: their partial J/K add.
    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        allreduce_or_panic(comm, self.j.get_data_mut(), "J");
        allreduce_or_panic(comm, self.k.get_data_mut(), "K");
    }
}

/// Unrestricted (per-spin densities) direct J/K accumulator: `J = coeff_j J[p_total]`, `K_s = coeff_k K[p_s]`, with
/// `p_total = p_alpha + p_beta` given by the caller.
pub struct UnrestrictedDirectFock<'a> {
    p_alpha: MatrixView<'a, f64>,
    p_beta: MatrixView<'a, f64>,
    p_total: MatrixView<'a, f64>,
    j: MatrixViewMut<'a, f64>,
    k_alpha: MatrixViewMut<'a, f64>,
    k_beta: MatrixViewMut<'a, f64>,
    coeff_j: f64,
    coeff_k: f64,
}

impl<'a> UnrestrictedDirectFock<'a> {
    /// `j`/`k_alpha`/`k_beta` are zeroed here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p_alpha: MatrixView<'a, f64>,
        p_beta: MatrixView<'a, f64>,
        p_total: MatrixView<'a, f64>,
        mut j: MatrixViewMut<'a, f64>,
        mut k_alpha: MatrixViewMut<'a, f64>,
        mut k_beta: MatrixViewMut<'a, f64>,
        coeff_j: f64,
        coeff_k: f64,
    ) -> Self {
        j.fill(0.0);
        k_alpha.fill(0.0);
        k_beta.fill(0.0);
        UnrestrictedDirectFock { p_alpha, p_beta, p_total, j, k_alpha, k_beta, coeff_j, coeff_k }
    }
}

impl eri::DirectFock for UnrestrictedDirectFock<'_> {
    fn screening_densities(&self) -> Option<ScreeningDensities<'_>> {
        let k = self.coeff_k != 0.0;
        Some(ScreeningDensities {
            n: self.p_total.rows(),
            coulomb: [(self.coeff_j != 0.0).then_some(self.p_total.get_data()), None],
            exchange: [k.then_some(self.p_alpha.get_data()), k.then_some(self.p_beta.get_data()), None],
            both: None,
            pair_densities: None,
        })
    }

    // J and K by rows (digest.rs): per density with a part, J's tiles and K's strips in the row's scratch, in order.
    fn row_len(&self, rows: usize) -> usize {
        self.row_len_at(rows, self.p_total.rows())
    }

    fn row_len_at(&self, rows: usize, n: usize) -> usize {
        usize::from(self.coeff_j != 0.0) * coulomb_row_len(rows)
            + usize::from(self.coeff_k != 0.0) * exchange_row_len(rows, n)
            + usize::from(self.coeff_k != 0.0) * exchange_row_len(rows, n)
    }

    fn begin_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let (rows, n) = (row_functions(pair_ab), self.p_total.rows());
        if self.coeff_j != 0.0 {
            begin_coulomb_row(pair_ab, &self.p_total, take_scratch(&mut row, coulomb_row_len(rows)));
        }
        if self.coeff_k != 0.0 {
            begin_exchange_row(pair_ab, &self.p_alpha, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
        if self.coeff_k != 0.0 {
            begin_exchange_row(pair_ab, &self.p_beta, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
    }

    type Tiles<'s> = ();

    fn digestion_tiles<'s, S: ShellKind>(&self, _block: &EriBlock<'_, '_>, _pair_ab: &ShellPair<S>, _pair_cd: &ShellPair<S>, _arena: &Arena<'s>) -> Result<(), ArenaError> {
        Ok(())
    }

    // In the row's scratch alone: no tiles.
    fn distribute_block_in_row<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, mut row: &mut [f64], _tiles: &mut ()) {
        let (rows, n) = (row_functions(pair_ab), self.p_total.rows());
        if self.coeff_j != 0.0 {
            coulomb_in_row(block, pair_ab, pair_cd, &self.p_total, &mut self.j, take_scratch(&mut row, coulomb_row_len(rows)), self.coeff_j);
        }
        if self.coeff_k != 0.0 {
            exchange_in_row(block, pair_ab, pair_cd, take_scratch(&mut row, exchange_row_len(rows, n)), self.coeff_k);
        }
        if self.coeff_k != 0.0 {
            exchange_in_row(block, pair_ab, pair_cd, take_scratch(&mut row, exchange_row_len(rows, n)), self.coeff_k);
        }
    }

    fn end_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let (rows, n) = (row_functions(pair_ab), self.p_total.rows());
        if self.coeff_j != 0.0 {
            end_coulomb_row(pair_ab, &mut self.j, take_scratch(&mut row, coulomb_row_len(rows)));
        }
        if self.coeff_k != 0.0 {
            end_exchange_row(pair_ab, &mut self.k_alpha, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
        if self.coeff_k != 0.0 {
            end_exchange_row(pair_ab, &mut self.k_beta, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
    }
}

impl<'a> TensorChoice for UnrestrictedDirectFock<'a> {
    fn distribute(&mut self, _: usize, _: usize, _: usize, _: usize, _: f64) {
        per_integral_unused()
    }

    fn distribute_block<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, _scratch: &mut Arena<'_>) -> Result<(), ArenaError> {
        if self.coeff_j != 0.0 {
            coulomb(block, pair_ab, pair_cd, &self.p_total, &mut self.j, self.coeff_j, None);
        }
        if self.coeff_k != 0.0 {
            exchange(block, pair_ab, pair_cd, &self.p_alpha, &mut self.k_alpha, self.coeff_k, None);
            exchange(block, pair_ab, pair_cd, &self.p_beta, &mut self.k_beta, self.coeff_k, None);
        }
        Ok(())
    }


    fn finish(&mut self) {
        self.j.symmetrize_in_place(1.0);
        self.k_alpha.symmetrize_in_place(1.0);
        self.k_beta.symmetrize_in_place(1.0);
    }

    // As `RestrictedDirectFock`'s.
    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        allreduce_or_panic(comm, self.j.get_data_mut(), "J");
        allreduce_or_panic(comm, self.k_alpha.get_data_mut(), "K_alpha");
        allreduce_or_panic(comm, self.k_beta.get_data_mut(), "K_beta");
    }
}

/// Direct J/K of ONE density of any symmetry: `J = coeff_j J[p]`, `J_ab = sum_cd p_cd (ab|cd)` (only `p`'s
/// symmetric part enters), and `K = coeff_k K[p]`, `K_ac = sum_bd p_bd (ab|cd)`, so `K[p]^T = K[p^T]`, and K is not
/// symmetrized. For a transition density (state interaction) or the
/// spin-flip density of a stability analysis; a symmetric `p` gives `RestrictedDirectFock`'s J and K. Both outputs
/// are zeroed here.
pub struct GeneralDirectFock<'a> {
    p: MatrixView<'a, f64>,
    j: MatrixViewMut<'a, f64>,
    k: MatrixViewMut<'a, f64>,
    coeff_j: f64,
    coeff_k: f64,
}

impl<'a> GeneralDirectFock<'a> {
    /// The accumulator over the caller's density and outputs, both outputs zeroed here.
    pub fn new(p: MatrixView<'a, f64>, mut j: MatrixViewMut<'a, f64>, mut k: MatrixViewMut<'a, f64>, coeff_j: f64, coeff_k: f64) -> Self {
        j.fill(0.0);
        k.fill(0.0);
        GeneralDirectFock { p, j, k, coeff_j, coeff_k }
    }
}

impl eri::DirectFock for GeneralDirectFock<'_> {
    type Tiles<'s> = DigestTiles<'s>;

    fn digestion_tiles<'s, S: ShellKind>(&self, _block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, arena: &Arena<'s>) -> Result<DigestTiles<'s>, ArenaError> {
        DigestTiles::new_in(pair_ab, pair_cd, arena)
    }

    fn distribute_block_in_row<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, _row: &mut [f64], tiles: &mut DigestTiles<'_>) {
        self.digest(block, pair_ab, pair_cd, tiles)
    }

    fn screening_densities(&self) -> Option<ScreeningDensities<'_>> {
        let p = self.p.get_data();
        Some(ScreeningDensities {
            n: self.p.rows(),
            coulomb: [(self.coeff_j != 0.0).then_some(p), None],
            exchange: [(self.coeff_k != 0.0).then_some(p), None, None],
            both: None,
            pair_densities: None,
        })
    }
}

impl GeneralDirectFock<'_> {
    /// The batch's J and K with `tiles` (`DigestTiles` of the quartet).
    fn digest<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, tiles: &mut DigestTiles<'_>) {
        if self.coeff_j != 0.0 {
            coulomb(block, pair_ab, pair_cd, &self.p, &mut self.j, self.coeff_j, Some(&mut *tiles));
        }
        if self.coeff_k != 0.0 {
            exchange(block, pair_ab, pair_cd, &self.p, &mut self.k, self.coeff_k, Some(&mut *tiles));
        }
    }
}

impl<'a> TensorChoice for GeneralDirectFock<'a> {
    fn distribute(&mut self, _: usize, _: usize, _: usize, _: usize, _: f64) {
        per_integral_unused()
    }

    // J from p's symmetric part, completed in `finish`; K written in full (`symmetric: false`), left alone there.
    fn distribute_block<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, scratch: &mut Arena<'_>) -> Result<(), ArenaError> {
        let mut tiles = DigestTiles::new_in(pair_ab, pair_cd, scratch)?;
        self.digest(block, pair_ab, pair_cd, &mut tiles);
        Ok(())
    }

    fn finish(&mut self) {
        self.j.symmetrize_in_place(1.0);
    }

    // As `RestrictedDirectFock`'s.
    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        allreduce_or_panic(comm, self.j.get_data_mut(), "J");
        allreduce_or_panic(comm, self.k.get_data_mut(), "K");
    }
}

/// Direct J/K of several densities of any symmetry in one walk, each as `GeneralDirectFock`'s: `n_mat` densities
/// back to back in `p` (`n x n` each, column-major), their J and K in the same places of `j` and `k`, with the same
/// coefficients for every density. A batch is computed once and digested for each density in turn; it is skipped
/// only when no density needs it. Both outputs are zeroed here.
pub struct GeneralMultiDirectFock<'a> {
    n: usize,
    p: &'a [f64],
    j: &'a mut [f64],
    k: &'a mut [f64],
    coeff_j: f64,
    coeff_k: f64,
}

impl<'a> GeneralMultiDirectFock<'a> {
    /// `p`, `j` and `k` hold the same number of `n x n` matrices.
    pub fn new(n: usize, p: &'a [f64], j: &'a mut [f64], k: &'a mut [f64], coeff_j: f64, coeff_k: f64) -> Self {
        assert!(n > 0 && p.len() % (n * n) == 0 && j.len() == p.len() && k.len() == p.len(), "densities, J and K of n x n matrices each, as many of each");
        j.fill(0.0);
        k.fill(0.0);
        GeneralMultiDirectFock { n, p, j, k, coeff_j, coeff_k }
    }
}

impl eri::DirectFock for GeneralMultiDirectFock<'_> {
    type Tiles<'s> = DigestTiles<'s>;

    fn digestion_tiles<'s, S: ShellKind>(&self, _block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, arena: &Arena<'s>) -> Result<DigestTiles<'s>, ArenaError> {
        DigestTiles::new_in(pair_ab, pair_cd, arena)
    }

    fn distribute_block_in_row<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, _row: &mut [f64], tiles: &mut DigestTiles<'_>) {
        self.digest(block, pair_ab, pair_cd, tiles)
    }

    fn screening_densities(&self) -> Option<ScreeningDensities<'_>> {
        let built = self.coeff_j != 0.0 || self.coeff_k != 0.0;
        Some(ScreeningDensities { n: self.n, coulomb: [None, None], exchange: [None, None, None], both: built.then_some(self.p), pair_densities: None })
    }
}

impl GeneralMultiDirectFock<'_> {
    /// The batch's J and K with `tiles` (`DigestTiles` of the quartet).
    fn digest<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, tiles: &mut DigestTiles<'_>) {
        let (n, size) = (self.n, self.n * self.n);
        for ((p, j), k) in self.p.chunks_exact(size).zip(self.j.chunks_exact_mut(size)).zip(self.k.chunks_exact_mut(size)) {
            let p = MatrixView::new(n, n, p);
            if self.coeff_j != 0.0 {
                coulomb(block, pair_ab, pair_cd, &p, &mut MatrixViewMut::new(n, n, j), self.coeff_j, Some(&mut *tiles));
            }
            if self.coeff_k != 0.0 {
                exchange(block, pair_ab, pair_cd, &p, &mut MatrixViewMut::new(n, n, k), self.coeff_k, Some(&mut *tiles));
            }
        }
    }
}

impl<'a> TensorChoice for GeneralMultiDirectFock<'a> {
    fn distribute(&mut self, _: usize, _: usize, _: usize, _: usize, _: f64) {
        per_integral_unused()
    }

    // As `GeneralDirectFock`, once for each density.
    fn distribute_block<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, scratch: &mut Arena<'_>) -> Result<(), ArenaError> {
        let mut tiles = DigestTiles::new_in(pair_ab, pair_cd, scratch)?;
        self.digest(block, pair_ab, pair_cd, &mut tiles);
        Ok(())
    }

    fn finish(&mut self) {
        let n = self.n;
        for j in self.j.chunks_exact_mut(n * n) {
            MatrixViewMut::new(n, n, j).symmetrize_in_place(1.0);
        }
    }

    // As `RestrictedDirectFock`'s.
    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        allreduce_or_panic(comm, self.j, "J");
        allreduce_or_panic(comm, self.k, "K");
    }
}

/// Two-density restricted direct J/K accumulator (the inactive and active densities of CASSCF, or any two): each
/// density's `J = coeff_j J[p]` and `K = coeff_k K[p]` as `RestrictedDirectFock`'s, with coefficients of its own,
/// from one walk.
pub struct InactiveActiveDirectFock<'a> {
    p_inactive: MatrixView<'a, f64>,
    p_active: MatrixView<'a, f64>,
    j_inactive: MatrixViewMut<'a, f64>,
    k_inactive: MatrixViewMut<'a, f64>,
    j_active: MatrixViewMut<'a, f64>,
    k_active: MatrixViewMut<'a, f64>,
    coeff_j_inactive: f64,
    coeff_k_inactive: f64,
    coeff_j_active: f64,
    coeff_k_active: f64,
}

impl<'a> InactiveActiveDirectFock<'a> {
    /// All four output matrices are zeroed here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p_inactive: MatrixView<'a, f64>,
        p_active: MatrixView<'a, f64>,
        mut j_inactive: MatrixViewMut<'a, f64>,
        mut k_inactive: MatrixViewMut<'a, f64>,
        mut j_active: MatrixViewMut<'a, f64>,
        mut k_active: MatrixViewMut<'a, f64>,
        coeff_j_inactive: f64,
        coeff_k_inactive: f64,
        coeff_j_active: f64,
        coeff_k_active: f64,
    ) -> Self {
        j_inactive.fill(0.0);
        k_inactive.fill(0.0);
        j_active.fill(0.0);
        k_active.fill(0.0);
        InactiveActiveDirectFock {
            p_inactive, p_active, j_inactive, k_inactive, j_active, k_active,
            coeff_j_inactive, coeff_k_inactive, coeff_j_active, coeff_k_active,
        }
    }
}

impl eri::DirectFock for InactiveActiveDirectFock<'_> {
    fn screening_densities(&self) -> Option<ScreeningDensities<'_>> {
        let (inactive, active) = (self.p_inactive.get_data(), self.p_active.get_data());
        Some(ScreeningDensities {
            n: self.p_inactive.rows(),
            coulomb: [(self.coeff_j_inactive != 0.0).then_some(inactive), (self.coeff_j_active != 0.0).then_some(active)],
            exchange: [(self.coeff_k_inactive != 0.0).then_some(inactive), (self.coeff_k_active != 0.0).then_some(active), None],
            both: None,
            pair_densities: None,
        })
    }

    // J and K by rows (digest.rs): per density with a part, J's tiles and K's strips in the row's scratch, in order.
    fn row_len(&self, rows: usize) -> usize {
        self.row_len_at(rows, self.p_inactive.rows())
    }

    fn row_len_at(&self, rows: usize, n: usize) -> usize {
        usize::from(self.coeff_j_inactive != 0.0) * coulomb_row_len(rows)
            + usize::from(self.coeff_j_active != 0.0) * coulomb_row_len(rows)
            + usize::from(self.coeff_k_inactive != 0.0) * exchange_row_len(rows, n)
            + usize::from(self.coeff_k_active != 0.0) * exchange_row_len(rows, n)
    }

    fn begin_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let (rows, n) = (row_functions(pair_ab), self.p_inactive.rows());
        if self.coeff_j_inactive != 0.0 {
            begin_coulomb_row(pair_ab, &self.p_inactive, take_scratch(&mut row, coulomb_row_len(rows)));
        }
        if self.coeff_j_active != 0.0 {
            begin_coulomb_row(pair_ab, &self.p_active, take_scratch(&mut row, coulomb_row_len(rows)));
        }
        if self.coeff_k_inactive != 0.0 {
            begin_exchange_row(pair_ab, &self.p_inactive, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
        if self.coeff_k_active != 0.0 {
            begin_exchange_row(pair_ab, &self.p_active, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
    }

    type Tiles<'s> = ();

    fn digestion_tiles<'s, S: ShellKind>(&self, _block: &EriBlock<'_, '_>, _pair_ab: &ShellPair<S>, _pair_cd: &ShellPair<S>, _arena: &Arena<'s>) -> Result<(), ArenaError> {
        Ok(())
    }

    // In the row's scratch alone: no tiles.
    fn distribute_block_in_row<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, mut row: &mut [f64], _tiles: &mut ()) {
        let (rows, n) = (row_functions(pair_ab), self.p_inactive.rows());
        if self.coeff_j_inactive != 0.0 {
            coulomb_in_row(block, pair_ab, pair_cd, &self.p_inactive, &mut self.j_inactive, take_scratch(&mut row, coulomb_row_len(rows)), self.coeff_j_inactive);
        }
        if self.coeff_j_active != 0.0 {
            coulomb_in_row(block, pair_ab, pair_cd, &self.p_active, &mut self.j_active, take_scratch(&mut row, coulomb_row_len(rows)), self.coeff_j_active);
        }
        if self.coeff_k_inactive != 0.0 {
            exchange_in_row(block, pair_ab, pair_cd, take_scratch(&mut row, exchange_row_len(rows, n)), self.coeff_k_inactive);
        }
        if self.coeff_k_active != 0.0 {
            exchange_in_row(block, pair_ab, pair_cd, take_scratch(&mut row, exchange_row_len(rows, n)), self.coeff_k_active);
        }
    }

    fn end_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let (rows, n) = (row_functions(pair_ab), self.p_inactive.rows());
        if self.coeff_j_inactive != 0.0 {
            end_coulomb_row(pair_ab, &mut self.j_inactive, take_scratch(&mut row, coulomb_row_len(rows)));
        }
        if self.coeff_j_active != 0.0 {
            end_coulomb_row(pair_ab, &mut self.j_active, take_scratch(&mut row, coulomb_row_len(rows)));
        }
        if self.coeff_k_inactive != 0.0 {
            end_exchange_row(pair_ab, &mut self.k_inactive, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
        if self.coeff_k_active != 0.0 {
            end_exchange_row(pair_ab, &mut self.k_active, take_scratch(&mut row, exchange_row_len(rows, n)));
        }
    }
}

impl<'a> TensorChoice for InactiveActiveDirectFock<'a> {
    fn distribute(&mut self, _: usize, _: usize, _: usize, _: usize, _: f64) {
        per_integral_unused()
    }

    fn distribute_block<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, _scratch: &mut Arena<'_>) -> Result<(), ArenaError> {
        if self.coeff_j_inactive != 0.0 {
            coulomb(block, pair_ab, pair_cd, &self.p_inactive, &mut self.j_inactive, self.coeff_j_inactive, None);
        }
        if self.coeff_k_inactive != 0.0 {
            exchange(block, pair_ab, pair_cd, &self.p_inactive, &mut self.k_inactive, self.coeff_k_inactive, None);
        }
        if self.coeff_j_active != 0.0 {
            coulomb(block, pair_ab, pair_cd, &self.p_active, &mut self.j_active, self.coeff_j_active, None);
        }
        if self.coeff_k_active != 0.0 {
            exchange(block, pair_ab, pair_cd, &self.p_active, &mut self.k_active, self.coeff_k_active, None);
        }
        Ok(())
    }


    fn finish(&mut self) {
        self.j_inactive.symmetrize_in_place(1.0);
        self.k_inactive.symmetrize_in_place(1.0);
        self.j_active.symmetrize_in_place(1.0);
        self.k_active.symmetrize_in_place(1.0);
    }

    // As `RestrictedDirectFock`'s.
    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        allreduce_or_panic(comm, self.j_inactive.get_data_mut(), "J_inactive");
        allreduce_or_panic(comm, self.k_inactive.get_data_mut(), "K_inactive");
        allreduce_or_panic(comm, self.j_active.get_data_mut(), "J_active");
        allreduce_or_panic(comm, self.k_active.get_data_mut(), "K_active");
    }
}

/// Which density pairing a `TwoElectronTerm` weights the integral `(ab|cd)` with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoElectronTermKind {
    /// `left[a, b] * right[c, d]` (Coulomb-like).
    Coulomb,
    /// `left[a, c] * right[b, d]` (exchange-like; `left`, `right` need not be symmetric).
    Exchange,
}

/// One term of the two-electron energy whose geometric derivative `DirectEriGradient` accumulates:
///
/// ```text
///   E_2e = 1/2 sum_terms coeff sum_abcd W_term(a, b, c, d) (ab|cd)
///   Coulomb:  W = left[a, b] right[c, d]        Exchange:  W = left[a, c] right[b, d]
/// ```
///
/// With the direct Fock builders' conventions: restricted `[(Coulomb, P, P, c_j), (Exchange, P, P, c_k)]`,
/// `c_k = -0.5 x`; unrestricted `[(Coulomb, P, P, c_j), (Exchange, P_a, P_a, c_k), (Exchange, P_b, P_b, c_k)]`,
/// `c_k = -x` (`P = P_a + P_b`, `x` the exact-exchange fraction). Generalized (GHF): `E_x = 1/2 c_k sum_st
/// sum_abcd P^st_ac P^st_bd (ab|cd)` over the four spin blocks, so `(Coulomb, P, P, c_j)` with `P = P_aa + P_bb`,
/// `(Exchange, P_aa, P_aa, c_k)`, `(Exchange, P_bb, P_bb, c_k)` and the spin-flip block paired with ITSELF,
/// `(Exchange, P_ab, P_ab, 2 c_k)` (the `P_ba = P_ab^T` term is the same by the permutational symmetry). That
/// is the energy whose derivative `GeneralizedDirectFock`'s `K_ab[a,c] = sum_bd P_ab[b,d] (ab|cd)` is. Terms
/// mixing two densities (CASSCF inactive/active) are written the same way.
///
/// `left` and `right` are the caller's `n x n` column-major densities, borrowed where they are (no copy; two terms
/// may share one).
#[derive(Clone, Copy, Debug)]
pub struct TwoElectronTerm<'a> {
    /// The density pairing.
    pub kind: TwoElectronTermKind,
    /// The first density, `n x n` column-major.
    pub left: &'a [f64],
    /// The second density, `n x n` column-major.
    pub right: &'a [f64],
    /// The term's coefficient.
    pub coeff: f64,
}

/// Direct two-electron energy gradient: contracts ERI *derivative* integrals with the densities of
/// `terms` as they are computed and accumulates `dE_2e/dX` into `gradient`, a column-major
/// `(3, n_centres)` matrix (`gradient[3 * X + axis]`, the layout of the one-electron gradients),
/// OVERWRITTEN (zeroed here). The production counterpart of `eri::GradientTensors`, driven through
/// `AedelstenContext::compute_eri_gradient` (or `compute_two_electron_gradient`, which builds it).
/// Nothing but the `3 x n_centres` result is stored. The weight of an integral is formed once and
/// applied to every (atom, axis) of its quartet.
///
/// Only the integral derivatives: the density-derivative (Pulay, overlap-derivative) terms belong to
/// the one-electron gradients. Range-separated hybrids need one call per range operator; the
/// consumer adds them, as for the direct Fock builders.
pub struct DirectEriGradient<'a> {
    terms: &'a [TwoElectronTerm<'a>],
    n: usize,
    gradient: &'a mut [f64],
    atoms: QuartetAtoms,
}

impl<'a> DirectEriGradient<'a> {
    /// Every density of `terms` is `n x n`.
    pub fn new(terms: &'a [TwoElectronTerm<'a>], n: usize, gradient: &'a mut [f64]) -> Self {
        assert!(gradient.len() % 3 == 0, "the gradient is a (3, n_centres) matrix");
        assert!(terms.iter().all(|term| term.left.len() == n * n && term.right.len() == n * n), "every density is n x n");
        gradient.fill(0.0);
        DirectEriGradient { terms, n, gradient, atoms: QuartetAtoms::default() }
    }
}

impl<'a> GradientSink for DirectEriGradient<'a> {
    fn select_atoms(&mut self, atoms: &[usize]) {
        self.atoms.set(atoms);
    }

    fn screening(&self) -> Option<eri::GradientScreening<'_>> {
        let terms = self
            .terms
            .iter()
            .map(|term| eri::GradientScreeningTerm { exchange: matches!(term.kind, TwoElectronTermKind::Exchange), left: term.left, right: term.right, coeff: term.coeff })
            .collect();
        Some(eri::GradientScreening { n: self.n, terms })
    }

    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, derivatives: IntegralDerivatives<'_>) {
        // The derivatives are those of one canonical integral; the energy sums every distinct
        // permutation of it with that permutation's own density factors. The weight is the same for
        // every (atom, axis), so it is formed once per integral. (No dedup of repeated calls: the
        // shell-quartet loop already delivers each unique integral once.)
        let (perms, n_perms) = unique_permutations(i, j, k, l);
        let mut weight = 0.0;
        for term in self.terms {
            // Views of the caller's slices: no copy, no allocation.
            let (left, right) = (MatrixView::new(self.n, self.n, term.left), MatrixView::new(self.n, self.n, term.right));
            for &(a, b, c, d) in &perms[..n_perms] {
                weight += term.coeff * match term.kind {
                    TwoElectronTermKind::Coulomb => left[(a, b)] * right[(c, d)],
                    TwoElectronTermKind::Exchange => left[(a, c)] * right[(b, d)],
                };
            }
        }
        let half_weight = 0.5 * weight;
        for q in 0..derivatives.len() {
            self.gradient[3 * self.atoms.atom(q) + q % 3] += half_weight * derivatives.get(q);
        }
    }

    // Ranks own disjoint shell quartets: their partial gradients add.
    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        allreduce_or_panic(comm, self.gradient, "ERI gradient");
    }
}
