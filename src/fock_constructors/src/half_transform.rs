// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The half-transformed ERIs from one direct walk:
//!
//!   X(p, nu, r, sigma) = sum_{mu lambda} L_mu,p R_lambda,r (mu nu|lambda sigma),
//!
//! for the columns p of the left coefficients L (n x m_L) and r of the right ones R (n x m_R). Two indices are
//! contracted, two stay in the AO basis.
//!
//! Per batch (AB|CD) of the walk's row AB, with the slot weights of `eri::for_each_slot_quartet`, every stored element
//! w v stands for its 8 permutations, none deduplicated. By the pair that carries each side's contracted index they
//! are two groups of four:
//!
//!   row-R:  L on the ket pair, R on the row pair    (cd|ab), (cd|ba), (dc|ab), (dc|ba)
//!   row-L:  L on the row pair, R on the ket pair    (ab|cd), (ab|dc), (ba|cd), (ba|dc)
//!
//! The contracted index on the row pair is the same for the whole row, so that quarter is deferred to the row's end
//! (`end_row`), after the other quarter has been summed over every ket pair of the row. For row-R, a in A and b in B
//! (the same functions for a single-centre row pair):
//!
//!   T_L(p, nu, a, b) = sum_{CD in row} sum_{mu in C or D} L_mu,p (mu nu|a b)   per quartet, per stored element, one
//!                                                                            axpy of length m_L per ket shell
//!   X(p, nu, r, b) += sum_a T_L(p, nu, a, b) R_a,r                            at the row's end, per b one GEMM into
//!   X(p, nu, r, a) += sum_b T_L(p, nu, a, b) R_b,r                            m_R whole columns of X; per a likewise
//!
//! The per-quartet quarter has an inner dimension of one shell, too small for a GEMM: it is a loop over the batch's
//! stored elements, each one an axpy of a column of L^T (m_L x n, made once per walk: the coefficients of a function
//! contiguous) into a column segment of T_L (the m_L values of one (nu, a, b)), all in cache. No tiles and no scratch.
//!
//! Row-L is the same with L and R swapped: T_R(r, sigma, a, b) from R on the ket shells, then L on the row's shells
//! into m_L whole rows of X. In one orbital basis (L = R), X(p, nu, r, sigma) = X(r, sigma, p, nu) and row-L is the
//! transpose of row-R, so only row-R is formed and `finish` adds the transpose of X as a (m n) x (m n) matrix.
//!
//! A batch of sigma (`sigma_batch`: the sigma of a run of whole shells S, X's columns for them) needs the
//! permutations whose sigma is in S: row-R on the rows with a shell in S (T_L over all their kets, as above; only the
//! columns of S written) and row-L on the kets with a shell in S (T_R only for the sigma of S, the ket shells in S its
//! targets), and no other quartet: the walk skips those (`DirectFock::wanted_functions`). In one basis T_R is T_L's
//! rows of S (L = R), so row-L reads them from T_L, filled for just those rows on a row outside S. Over batches that
//! cover the basis, a quartet's integrals are computed once per batch among its four shells' (at most 4 times), the
//! per-quartet quarter of a row's T_L over all kets once per batch among its two shells' (at most twice) and that of
//! the restricted T once, and each element of X is written by the rows that reach it, as in one walk.
//!
//! On a node (`HalfTransformedEri::on_node`) the ranks share one X in the node's memory instead of each forming its own
//! and summing: X's rows (p, nu) are cut into blocks of nu and its columns (r, sigma) into blocks of sigma, at shell
//! boundaries, and each tile (a block of rows by a block of columns) has a lock of its own. A row's deferred quarter
//! writes the columns of sigma on the row (row-R) or the rows of nu on it (row-L) one tile at a time, under the tile's
//! lock. In one basis over every sigma only row-R writes, so the rows are one block and every shell its own column
//! block: a row's products are as without the node, under one lock per shell. At the end the ranks form X~ + X~^T
//! together, each its part of the tiles of the symmetrization.
//!
//! Cost (N functions, m = m_L = m_R, one basis): the per-quartet quarter m N^4 / 2 over the unique quartets and the
//! row's quarter 2 m^2 N^3 over the rows, each element of X touched once per row that reaches it; the integrals are
//! those of the unique walk.

use std::ops::Range;

use arrays::array3d::Array3DTrait;
use arrays::array4d::{Array4DTrait, Array4DView};
use arrays::matrix::{ColumnFactor, Factor, MatrixMathMut, MatrixTrait, MatrixTraitMut, MatrixView, MatrixViewMut, StridedMatrixView};
use arrays::scratch::{Arena, ArenaError};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;
use eri::{for_each_slot_quartet, EriBlock, TensorChoice};

use aedelsten_mpi::{Communicator, SharedLocks};

use crate::digest::Shells;
use crate::direct::allreduce_or_panic;

/// The coefficients of `X`: one orbital basis on both sides (`L = R`, X symmetric under the pair swap), or a left and
/// a right one.
enum Orbitals<'a> {
    One(MatrixView<'a, f64>),
    Two { left: MatrixView<'a, f64>, right: MatrixView<'a, f64> },
}

/// Where the ranks of a walk put their parts of `X`: each its own whole `X`, summed over the ranks afterwards when the
/// walk is split (`partition`), or one `X` in the shared memory of a node's ranks.
enum Writes<'a> {
    Own { partition: bool },
    Node(NodeTiles<'a>),
}

/// `X`'s tiles on a node (see the top of this file): the first functions of its blocks of nu (rows) and of sigma
/// (columns), each list ending with the end of its last block, and a lock per tile, `row block * column blocks +
/// column block`.
struct NodeTiles<'a> {
    comm: Communicator,
    locks: SharedLocks,
    row_bounds: &'a [usize],
    column_bounds: &'a [usize],
}

impl NodeTiles<'_> {
    /// The block of `bounds` that function `f` is in.
    fn block(bounds: &[usize], f: usize) -> usize {
        bounds.partition_point(|&start| start <= f) - 1
    }

    /// The blocks of `bounds`, each with its functions.
    fn blocks(bounds: &[usize]) -> impl Iterator<Item = (usize, Range<usize>)> + '_ {
        bounds.windows(2).enumerate().map(|(k, w)| (k, w[0]..w[1]))
    }

    fn tile(&self, row_block: usize, column_block: usize) -> usize {
        row_block * (self.column_bounds.len() - 1) + column_block
    }
}

/// `count` blocks of whole shells (`shell_bounds`: each shell's first function, then the end of the last) of about
/// equal function counts, as their first functions followed by the end, into `out` (`count + 1` long; `count` at most
/// the shells).
fn block_bounds(shell_bounds: &[usize], count: usize, out: &mut [usize]) {
    let (first, end) = (shell_bounds[0], *shell_bounds.last().expect("the shell bounds end with the last shell's end"));
    let shells = shell_bounds.len() - 1;
    let mut next = 0;
    for (k, &start) in shell_bounds[..shells].iter().enumerate() {
        // A block begins at the first shell whose start reaches the block's share of the functions, or where only as
        // many shells are left as blocks.
        if next < count && ((start - first) * count >= next * (end - first) || shells - k == count - next) {
            out[next] = start;
            next += 1;
        }
    }
    assert_eq!(next, count, "{count} blocks of {shells} shells");
    out[count] = end;
}

impl Orbitals<'_> {
    fn left(&self) -> &MatrixView<'_, f64> {
        match self {
            Orbitals::One(c) | Orbitals::Two { left: c, .. } => c,
        }
    }

    fn right(&self) -> &MatrixView<'_, f64> {
        match self {
            Orbitals::One(c) | Orbitals::Two { right: c, .. } => c,
        }
    }
}

/// The half-transformed ERIs `X(p, nu, r, sigma) = sum_{mu lambda} L_mu,p R_lambda,r (mu nu|lambda sigma)` (see the
/// top of this file) from one direct walk; no J/K.
///
/// `x` is `X` as the `(m_L n) x (m_R n)` matrix of the column-major `(p, nu, r, sigma)` array: `p` fastest, then
/// `nu`, `r`, `sigma`; zeroed here. For a batch of sigma (`sigma_batch`) it is that matrix's columns of the batch,
/// `(m_L n) x (m_R n_sigma)`. The coefficients are `n x m` views (e.g. a column block of the orbital matrix).
/// `threshold`: a batch whose Schwarz bound times the largest `|L|` on one of its pairs' shells times the largest
/// `|R|` on the other pair's is below it skips the transformation; `None` transforms every batch. `partition`: under
/// MPI, split the rows over the ranks and sum `X` (a collective walk); `false` computes the whole `X` on this rank
/// alone (for a caller that distributes batches of `L` over the ranks itself).
///
/// Memory: `X` is the caller's, `m_L m_R n n_sigma` doubles. The row buffers (see the top of this file) come from the
/// arena given to `new`, at exact size (`row_buffer_len`, `sigma_row_buffer_len`): `T_L`, `s_max^2 m_L n` doubles,
/// and in two bases `T_R`, `s_max^2 m_R n_sigma` (`s_max = max_shell_functions`, the largest shell's function count).
/// The caller takes them off the front of its scratch and runs the walk in the rest
/// (`Arena::into_sub_scratch`), with the transposed coefficients, `m_L n` (and `m_R n`) doubles. No scratch per
/// batch. The walk must be the direct one by rows (`begin_row` before a row's batches, `end_row` after them). Nothing
/// is allocated on the heap.
pub struct HalfTransformedEri<'a> {
    orbitals: Orbitals<'a>,
    x: MatrixViewMut<'a, f64>,
    threshold: Option<f64>,
    writes: Writes<'a>,
    max_shell_functions: usize,
    /// The functions of sigma that `x` holds: all n, or a batch of whole shells (`sigma_batch`).
    sigma: Range<usize>,
    /// One basis and every sigma: only row-R is formed, and `finish` adds X's transpose.
    symmetric: bool,
    /// T_L(p, nu, a, b): (m_L n) x s_max^2, the row's first s_A s_B columns used; zero between rows.
    left_row: MatrixViewMut<'a, f64>,
    /// T_R(r, sigma, a, b) for the sigma of `sigma` (`sigma.start` its first), two bases only: (m_R n_sigma) x
    /// s_max^2.
    right_row: Option<MatrixViewMut<'a, f64>>,
    /// L^T (m_L x n), and R^T (m_R x n) in two bases.
    left_t: MatrixViewMut<'a, f64>,
    right_t: Option<MatrixViewMut<'a, f64>>,
    row: Option<Row>,
}

/// The walk row in hand: the functions of its two shells (the same range twice for a single-centre pair), whether a
/// shell of it is in sigma (row-R: T_L over every ket), and whether a ket shell in sigma has come (row-L).
struct Row {
    a: Range<usize>,
    b: Range<usize>,
    in_sigma: bool,
    kets_in_sigma: bool,
}

impl<'a> HalfTransformedEri<'a> {
    /// The doubles of the row buffers and transposed coefficients `new` (one basis) or `mixed` (two) take from their
    /// arena over `n` basis functions: `(s_max^2 + 1) m_L n`, and `(s_max^2 + 1) m_R n` more in two bases.
    pub fn row_buffer_len(n: usize, m_left: usize, m_right: usize, max_shell_functions: usize, one_basis: bool) -> usize {
        Self::sigma_row_buffer_len(n, n, m_left, m_right, max_shell_functions, one_basis)
    }

    /// The doubles `sigma_batch` (or `on_node` with a batch) takes from its arena for a batch of `n_sigma` of the `n`
    /// functions: `(s_max^2 + 1) m_L n`, and `s_max^2 m_R n_sigma + m_R n` more in two bases.
    pub fn sigma_row_buffer_len(n: usize, n_sigma: usize, m_left: usize, m_right: usize, max_shell_functions: usize, one_basis: bool) -> usize {
        let s2 = max_shell_functions * max_shell_functions;
        (s2 + 1) * n * m_left + if one_basis { 0 } else { (s2 * n_sigma + n) * m_right }
    }

    /// One orbital basis, `c` on both sides: only half of the 8 permutations are transformed, the rest come from
    /// the symmetry of `X` in `finish`. `max_shell_functions`: the largest shell's function count (contracted
    /// functions times components) in the walk's basis, which sizes the row buffers, taken from `buffers` (an arena
    /// that outlives the walk; too small is the `ArenaError` naming the size).
    pub fn new(c: MatrixView<'a, f64>, x: MatrixViewMut<'a, f64>, threshold: Option<f64>, partition: bool, max_shell_functions: usize, buffers: &Arena<'a>) -> Result<Self, ArenaError> {
        Self::with_orbitals(Orbitals::One(c), None, x, threshold, Writes::Own { partition }, max_shell_functions, buffers)
    }

    /// A left and a right basis (`left` may be a column block of `right`, or anything else).
    #[allow(clippy::too_many_arguments)]
    pub fn mixed(
        left: MatrixView<'a, f64>,
        right: MatrixView<'a, f64>,
        x: MatrixViewMut<'a, f64>,
        threshold: Option<f64>,
        partition: bool,
        max_shell_functions: usize,
        buffers: &Arena<'a>,
    ) -> Result<Self, ArenaError> {
        assert_eq!(left.rows(), right.rows(), "the left and the right coefficients have one basis");
        Self::with_orbitals(Orbitals::Two { left, right }, None, x, threshold, Writes::Own { partition }, max_shell_functions, buffers)
    }

    /// X's columns for the sigma of `sigma` only, a run of whole shells (see the top of this file): `x` is `(m_L n) x
    /// (m_R n_sigma)`. `right` `None`: one basis, `left` on both sides. The walk computes only the quartets with a
    /// shell in `sigma` (`DirectFock::wanted_functions`). Otherwise as `new` and `mixed`.
    #[allow(clippy::too_many_arguments)]
    pub fn sigma_batch(
        left: MatrixView<'a, f64>,
        right: Option<MatrixView<'a, f64>>,
        sigma: Range<usize>,
        x: MatrixViewMut<'a, f64>,
        threshold: Option<f64>,
        partition: bool,
        max_shell_functions: usize,
        buffers: &Arena<'a>,
    ) -> Result<Self, ArenaError> {
        Self::with_orbitals(Self::orbitals(left, right), Some(sigma), x, threshold, Writes::Own { partition }, max_shell_functions, buffers)
    }

    fn orbitals(left: MatrixView<'a, f64>, right: Option<MatrixView<'a, f64>>) -> Orbitals<'a> {
        match right {
            None => Orbitals::One(left),
            Some(right) => {
                assert_eq!(left.rows(), right.rows(), "the left and the right coefficients have one basis");
                Orbitals::Two { left, right }
            }
        }
    }

    /// As `new` (`right` `None`: one basis) or `mixed`, or with `sigma` as `sigma_batch`, for the ranks of `comm`,
    /// which share one node's memory, with `x` the same memory on all of them: the walk over `comm`'s ranks writes its
    /// parts into the one `X` (see the top of this file), and nothing is summed afterwards. `shell_bounds`: the first
    /// function of every shell of the basis, ascending, then `n`; `sigma`'s ends are among them. Collective over
    /// `comm` (zeroing `x` in parts, a barrier); its `finish` is too. The tile bounds come from `buffers` too
    /// (`node_buffer_len` `usize`s). Panics if MPI fails.
    #[allow(clippy::too_many_arguments)]
    pub fn on_node(
        left: MatrixView<'a, f64>,
        right: Option<MatrixView<'a, f64>>,
        sigma: Option<Range<usize>>,
        x: MatrixViewMut<'a, f64>,
        threshold: Option<f64>,
        max_shell_functions: usize,
        buffers: &Arena<'a>,
        comm: &Communicator,
        shell_bounds: &[usize],
    ) -> Result<Self, ArenaError> {
        let n_shells = shell_bounds.len() - 1;
        let symmetric = right.is_none() && sigma.is_none();
        let sigma_shells = Self::sigma_shells(shell_bounds, sigma.as_ref());
        let (row_blocks, column_blocks) = Self::node_blocks(n_shells, sigma_shells.len() - 1, comm.size(), symmetric);
        let row_bounds = buffers.alloc::<usize>(row_blocks + 1)?;
        let column_bounds = buffers.alloc::<usize>(column_blocks + 1)?;
        block_bounds(shell_bounds, row_blocks, row_bounds);
        block_bounds(sigma_shells, column_blocks, column_bounds);
        let locks = comm.shared_locks(row_blocks * column_blocks).expect("the locks of X's tiles in the node's shared memory");
        let writes = Writes::Node(NodeTiles { comm: comm.clone(), locks, row_bounds, column_bounds });
        Self::with_orbitals(Self::orbitals(left, right), sigma, x, threshold, writes, max_shell_functions, buffers)
    }

    /// The bounds of `shell_bounds` from `sigma`'s start to its end (all of them for `None`).
    fn sigma_shells<'s>(shell_bounds: &'s [usize], sigma: Option<&Range<usize>>) -> &'s [usize] {
        let Some(sigma) = sigma else { return shell_bounds };
        let (first, last) = (shell_bounds.partition_point(|&f| f < sigma.start), shell_bounds.partition_point(|&f| f < sigma.end));
        assert!(sigma.start < sigma.end && last < shell_bounds.len() && shell_bounds[first] == sigma.start && shell_bounds[last] == sigma.end, "sigma {sigma:?} is not a run of whole shells");
        &shell_bounds[first..=last]
    }

    /// The blocks of rows and of columns of `X`'s tiles on a node of `ranks` ranks over `n_shells` shells,
    /// `sigma_shells` of them X's columns' (see the top of this file): one block of rows and a column block per shell
    /// when `symmetric` (one basis, every sigma), else up to twice the ranks each.
    fn node_blocks(n_shells: usize, sigma_shells: usize, ranks: usize, symmetric: bool) -> (usize, usize) {
        let few = |shells: usize| shells.min(2 * ranks).max(1);
        if symmetric { (1, n_shells.max(1)) } else { (few(n_shells), few(sigma_shells)) }
    }

    /// The `usize`s `on_node` takes from its arena for the tile bounds, beside `row_buffer_len`'s (or
    /// `sigma_row_buffer_len`'s) doubles, over `n_shells` shells, `sigma_shells` of them in X's columns.
    pub fn node_buffer_len(n_shells: usize, sigma_shells: usize, ranks: usize, symmetric: bool) -> usize {
        let (rows, columns) = Self::node_blocks(n_shells, sigma_shells, ranks, symmetric);
        rows + columns + 2
    }

    /// How many times this rank waited for a tile's lock held by another (0 unless `on_node`).
    pub fn lock_waits(&self) -> u64 {
        match &self.writes {
            Writes::Own { .. } => 0,
            Writes::Node(tiles) => tiles.locks.waits(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn with_orbitals(
        orbitals: Orbitals<'a>,
        sigma: Option<Range<usize>>,
        mut x: MatrixViewMut<'a, f64>,
        threshold: Option<f64>,
        writes: Writes<'a>,
        max_shell_functions: usize,
        buffers: &Arena<'a>,
    ) -> Result<Self, ArenaError> {
        let (n, m_left, m_right) = (orbitals.left().rows(), orbitals.left().cols(), orbitals.right().cols());
        let symmetric = sigma.is_none() && matches!(orbitals, Orbitals::One(_));
        let sigma = sigma.unwrap_or(0..n);
        assert!(sigma.start < sigma.end && sigma.end <= n || n == 0, "sigma {sigma:?} is not within the {n} functions");
        assert_eq!((x.rows(), x.cols()), (m_left * n, m_right * sigma.len()), "X is (m_left n) x (m_right n_sigma)");
        // The row buffers, once per walk (see the top of this file): what a row's (a, b) block can hold, zeroed.
        let s2 = max_shell_functions * max_shell_functions;
        let left_row = MatrixViewMut::from_arena(m_left * n, s2, buffers)?;
        let right_row = match orbitals {
            Orbitals::One(_) => None,
            Orbitals::Two { .. } => Some(MatrixViewMut::from_arena(m_right * sigma.len(), s2, buffers)?),
        };
        let left_t = orbitals.left().transpose_arena(buffers)?;
        let right_t = match &orbitals {
            Orbitals::One(_) => None,
            Orbitals::Two { right, .. } => Some(right.transpose_arena(buffers)?),
        };
        match &writes {
            Writes::Own { .. } => x.fill(0.0),
            Writes::Node(tiles) => {
                // Each rank its share of the columns, then every rank waits for all (the walk adds into them).
                let (rank, ranks, columns) = (tiles.comm.rank(), tiles.comm.size(), x.cols());
                x.view_columns_mut(columns * rank / ranks..columns * (rank + 1) / ranks).fill(0.0);
                tiles.comm.shared_memory_barrier().expect("a barrier over the node's ranks");
            }
        }
        Ok(HalfTransformedEri { orbitals, x, threshold, writes, max_shell_functions, sigma, symmetric, left_row, right_row, left_t, right_t, row: None })
    }

    fn begin<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>) {
        assert!(self.row.is_none(), "a row begins before the last one ended");
        let (sa, sb) = (pair_ab.get_n_cont_a() * pair_ab.get_degeneracy_a(), pair_ab.get_n_cont_b() * pair_ab.get_degeneracy_b());
        assert!(sa.max(sb) <= self.max_shell_functions, "a shell has more functions than max_shell_functions");
        let (a, b) = (pair_ab.get_idx_a(), pair_ab.get_idx_b());
        let in_sigma = self.sigma.contains(&a) || self.sigma.contains(&b);
        self.row = Some(Row { a: a..a + sa, b: b..b + sb, in_sigma, kets_in_sigma: false });
    }

    /// The row's deferred quarter (see the top of this file) into X, then the row buffers zeroed.
    fn end(&mut self) {
        let Some(Row { a, b, in_sigma, kets_in_sigma }) = self.row.take() else { return };
        let (n, m_left, m_right) = (self.orbitals.left().rows(), self.orbitals.left().cols(), self.orbitals.right().cols());
        if m_left == 0 || m_right == 0 {
            return;
        }
        let one_basis = matches!(self.orbitals, Orbitals::One(_));
        let used = 0..a.len() * b.len();
        let (left, right) = (self.orbitals.left(), self.orbitals.right());
        let sigma = self.sigma.clone();
        let tiles = match &self.writes {
            Writes::Own { .. } => None,
            Writes::Node(tiles) => Some(tiles),
        };
        // The bounds of one block of all n functions, and of one block of the sigma, for the writes without tiles.
        let (whole, whole_sigma) = ([0, n], [sigma.start, sigma.end]);
        // Row-R: X(p, nu, r, sigma) for sigma on the row (and in the batch) and every (p, nu): blocks of m_R whole
        // columns; on a node, a block of rows at a time, each tile under its lock.
        if in_sigma {
            let t = self.left_row.view_columns(used.clone());
            let t = t.view_reshaped_array3d((m_left * n, a.len(), b.len()));
            let (ra, rb) = (right.view_rows(a.clone()), right.view_rows(b.clone()));
            let row_bounds = tiles.map_or(&whole[..], |tiles| tiles.row_bounds);
            for (shell, other, along_b) in [(&b, &ra, true), (&a, &rb, false)] {
                if !sigma.contains(&shell.start) {
                    continue;
                }
                let column_block = tiles.map(|tiles| NodeTiles::block(tiles.column_bounds, shell.start));
                for (row_block, nu) in NodeTiles::blocks(row_bounds) {
                    let _held = tiles.zip(column_block).map(|(tiles, column_block)| tiles.locks.lock(tiles.tile(row_block, column_block)));
                    let rows = m_left * nu.start..m_left * nu.end;
                    for k in 0..shell.len() {
                        let column = shell.start + k - sigma.start;
                        let mut target = self.x.view_columns_mut(m_right * column..m_right * (column + 1));
                        // The tile's rows of m_R whole columns: a product's output, strided when the rows are a block.
                        let mut target = target.view_rows_mut(rows.clone());
                        if along_b {
                            t.view_sub_matrix(k).view_rows(rows.clone()).matmul_scaled_prealloc(other, &mut target, 1.0, 1.0);
                        } else {
                            t.view_dim1_matrix(k).view_rows(rows.clone()).matmul_scaled_prealloc(other, &mut target, 1.0, 1.0);
                        }
                    }
                }
            }
        }
        // Row-L (all but the symmetric path): X(p, nu, r, sigma) for nu on the row and every (r, sigma) of the batch:
        // blocks of m_L whole rows, from T_R (two bases; its first row the batch's first sigma) or T_L's rows of the
        // batch (one basis); on a node, a block of columns at a time, each tile under its lock.
        if !self.symmetric && kets_in_sigma {
            let (buffer, origin) = match &self.right_row {
                Some(right_row) => (right_row, sigma.start),
                None => (&self.left_row, 0),
            };
            let t = buffer.view_columns(used.clone());
            let t = t.view_reshaped_array3d((buffer.rows(), a.len(), b.len()));
            let (la, lb) = (left.view_rows(a.clone()), left.view_rows(b.clone()));
            let column_bounds = tiles.map_or(&whole_sigma[..], |tiles| tiles.column_bounds);
            for (shell, other, along_b) in [(&b, &la, true), (&a, &lb, false)] {
                let row_block = tiles.map(|tiles| NodeTiles::block(tiles.row_bounds, shell.start));
                for (column_block, block) in NodeTiles::blocks(column_bounds) {
                    let _held = tiles.zip(row_block).map(|(tiles, row_block)| tiles.locks.lock(tiles.tile(row_block, column_block)));
                    let columns = m_right * (block.start - sigma.start)..m_right * (block.end - sigma.start);
                    let t_rows = m_right * (block.start - origin)..m_right * (block.end - origin);
                    for k in 0..shell.len() {
                        let mut target = self.x.view_columns_mut(columns.clone());
                        // The tile's columns of m_L whole rows: a product's output, strided.
                        let mut target = target.view_rows_mut(m_left * (shell.start + k)..m_left * (shell.start + k + 1));
                        if along_b {
                            other.view_transposed().matmul_scaled_prealloc(&t.view_sub_matrix(k).view_rows(t_rows.clone()).view_transposed(), &mut target, 1.0, 1.0);
                        } else {
                            other.view_transposed().matmul_scaled_prealloc(&t.view_dim1_matrix(k).view_rows(t_rows.clone()).view_transposed(), &mut target, 1.0, 1.0);
                        }
                    }
                }
            }
        }
        // The buffers zeroed where the row wrote them: T_L whole on a row in the batch, else (one basis) its rows of
        // the batch; T_R whole.
        if in_sigma {
            self.left_row.view_columns_mut(used.clone()).fill(0.0);
        } else if one_basis && kets_in_sigma {
            let rows = m_left * sigma.start..m_left * sigma.end;
            for column in self.left_row.view_columns_mut(used.clone()).iter_mut() {
                column[rows.clone()].fill(0.0);
            }
        }
        if let (Some(right_row), true) = (&mut self.right_row, kets_in_sigma) {
            right_row.view_columns_mut(used).fill(0.0);
        }
    }

    fn transform<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>) {
        let (m_left, m_right) = (self.orbitals.left().cols(), self.orbitals.right().cols());
        if m_left == 0 || m_right == 0 {
            return;
        }
        let row_in_sigma = self.row.as_ref().expect("HalfTransformedEri digests the direct walk's rows (begin_row before a batch)").in_sigma;
        let one_basis = matches!(self.orbitals, Orbitals::One(_));
        let shells = Shells::of(pair_ab, pair_cd);
        // The ket shells in the batch, the targets of row-L's T; T_L over both ket shells on a row in the batch, else
        // (one basis, where it is row-L's T) only into those.
        let kets = [self.sigma.contains(&shells.start[2]), self.sigma.contains(&shells.start[3])];
        let left_into = if row_in_sigma { [true, true] } else if one_basis { kets } else { [false, false] };
        let right_into = if one_basis { [false, false] } else { kets };
        if left_into == [false, false] && right_into == [false, false] {
            return;
        }
        let left = shell_tiles(self.orbitals.left(), &shells);
        let right = match &self.orbitals {
            Orbitals::One(_) => None,
            Orbitals::Two { right, .. } => Some(shell_tiles(right, &shells)),
        };
        if let (Some(bound), Some(threshold)) = (block.bound(), self.threshold) {
            // The largest coefficient on each shell, read from its tile (a few rows per batch, nothing kept between
            // walks). L sits on either pair and R on the other.
            let largest = |tile: &StridedMatrixView<'_>| (0..tile.shape().1).flat_map(|p| tile.column(p)).fold(0.0_f64, |max, c| max.max(c.abs()));
            let [la, lb, lc, ld] = left.each_ref().map(largest);
            let [ra, rb, rc, rd] = match &right {
                None => [la, lb, lc, ld],
                Some(right) => right.each_ref().map(largest),
            };
            if bound * (la.max(lb) * rc.max(rd)).max(lc.max(ld) * ra.max(rb)) < threshold {
                return;
            }
        }
        if kets != [false, false] {
            self.row.as_mut().expect("a row is open").kets_in_sigma = true;
        }
        let (sa, start, origin) = (shells.size[0], shells.start, self.sigma.start);
        for_each_slot_quartet(block, pair_ab, pair_cd, |q| {
            let offsets: [usize; 4] = std::array::from_fn(|s| q.functions[s] - start[s]);
            let slot = block.slot_block(&q);
            if left_into != [false, false] {
                ket_quarter(&slot, q.weight, offsets, sa, &start, &self.left_t, &mut self.left_row, left_into, 0);
            }
            if let (Some(right_t), Some(right_row), true) = (&self.right_t, &mut self.right_row, right_into != [false, false]) {
                ket_quarter(&slot, q.weight, offsets, sa, &start, right_t, right_row, right_into, origin);
            }
        });
    }
}

/// The quarter over the ket shells of one stored slot block `v` (`(i, j, k, l)`, i fastest, weight `w`, its functions
/// from `offsets` within the shells A, B, C, D, those starting at `start`; `sa` A's functions) into a row buffer `t`
/// (`(p, nu)` x `(a, b)`, a fastest; each column an m x n_t matrix, its column `nu - origin`) from the transposed
/// coefficients `ct` (m x n): with L on C, T(:, d, a, b) += w v(a, b, c, d) ct(:, c) if `into[1]`; with L on D,
/// T(:, c, a, b) += w v(a, b, c, d) ct(:, d) if `into[0]`. `v` is read once, in its order (i innermost, along a
/// fibre); every target and source is a column, the targets of one fibre in consecutive columns of `t`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn ket_quarter(v: &Array4DView<'_, f64>, w: f64, [alpha, beta, gamma, delta]: [usize; 4], sa: usize, start: &[usize; 4], ct: &MatrixViewMut<'_, f64>, t: &mut MatrixViewMut<'_, f64>, [into_c, into_d]: [bool; 2], origin: usize) {
    let m = ct.rows();
    let [n_i, n_j, n_k, n_l] = v.dims();
    for l in 0..n_l {
        let d = start[3] + delta + l;
        let ct_d = ct.get_col(d);
        for k in 0..n_k {
            let c = start[2] + gamma + k;
            let ct_c = ct.get_col(c);
            for j in 0..n_j {
                // The columns T(:, :, a, b) of this b and the fibre's a, each an m x n_t matrix.
                let first = alpha + sa * (beta + j);
                let mut columns = t.view_columns_mut(first..first + n_i);
                for (column, &v_ijkl) in columns.iter_mut().zip(v.get_fibre((j, k, l))) {
                    let n_t = column.len() / m.max(1);
                    let mut t_ab = MatrixViewMut::new(m, n_t, column);
                    let wv = w * v_ijkl;
                    // L on C, into d; L on D, into c.
                    if into_d {
                        axpy(t_ab.get_col_mut(d - origin), wv, ct_c);
                    }
                    if into_c {
                        axpy(t_ab.get_col_mut(c - origin), wv, ct_d);
                    }
                }
            }
        }
    }
}

/// `y += a x`, inlined: the columns are short (m) and the calls many, where a BLAS call's overhead would dominate.
#[inline(always)]
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

/// The rows of `c` on each of the four shells, `(s_X x m)`: views, no copy (product operands that would otherwise
/// be copied per batch).
fn shell_tiles<'c>(c: &'c MatrixView<'_, f64>, shells: &Shells) -> [StridedMatrixView<'c>; 4] {
    std::array::from_fn(|x| c.view_rows(shells.start[x]..shells.start[x] + shells.size[x]))
}

impl eri::DirectFock for HalfTransformedEri<'_> {
    type Tiles<'s> = ();

    // No scratch per batch.
    fn digestion_tiles<'s, S: ShellKind>(&self, _block: &EriBlock<'_, '_>, _pair_ab: &ShellPair<S>, _pair_cd: &ShellPair<S>, _arena: &Arena<'s>) -> Result<(), ArenaError> {
        Ok(())
    }

    fn distribute_block_in_row<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, _row: &mut [f64], _tiles: &mut ()) {
        self.transform(block, pair_ab, pair_cd);
    }

    fn begin_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, _row: &mut [f64]) {
        self.begin(pair_ab);
    }

    fn end_row<S: ShellKind>(&mut self, _pair_ab: &ShellPair<S>, _row: &mut [f64]) {
        self.end();
    }

    // A batch of sigma needs only the quartets with a shell in it.
    fn wanted_functions(&self) -> Option<Range<usize>> {
        (self.sigma != (0..self.orbitals.left().rows())).then(|| self.sigma.clone())
    }
}

impl TensorChoice for HalfTransformedEri<'_> {
    fn distribute(&mut self, _: usize, _: usize, _: usize, _: usize, _: f64) {
        unreachable!("HalfTransformedEri digests whole batches (distribute_block); distribute is never called")
    }

    fn distribute_block<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, _scratch: &mut Arena<'_>) -> Result<(), ArenaError> {
        self.transform(block, pair_ab, pair_cd);
        Ok(())
    }

    // Every row has ended (the walk's `end_row`). Then, on the symmetric path, X = X~ + X~^T, X~ the half the walk
    // formed; linear, so it commutes with the sum over ranks. On a node, once every rank's rows are in, each rank its
    // part of the symmetrization, and every rank waits for all.
    fn finish(&mut self) {
        assert!(self.row.is_none(), "the walk ended inside a row");
        match &self.writes {
            Writes::Own { .. } => {
                if self.symmetric {
                    self.x.symmetrize_in_place(1.0);
                }
            }
            Writes::Node(tiles) => {
                tiles.comm.shared_memory_barrier().expect("a barrier over the node's ranks");
                if self.symmetric {
                    self.x.symmetrize_in_place_part(1.0, tiles.comm.rank(), tiles.comm.size());
                    tiles.comm.shared_memory_barrier().expect("a barrier over the node's ranks");
                }
            }
        }
    }

    // As the J/K accumulators': ranks own disjoint rows, so the partial X simply add; on a node they are in one X.
    fn wants_mpi_partition(&self) -> bool {
        match self.writes {
            Writes::Own { partition } => partition,
            Writes::Node(_) => true,
        }
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        if let Writes::Own { .. } = self.writes {
            allreduce_or_panic(comm, self.x.get_data_mut(), "X (half transformation)");
        }
    }
}
