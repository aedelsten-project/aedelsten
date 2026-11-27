// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Block digestion of one shell-quartet batch into J and K, shared by every direct accumulator of `direct`.
//!
//! The batch `(AB|CD)` is digested whole: every stored element, with the slot weight of
//! `eri::for_each_slot_quartet`, stands for all 8 of its permutations (no deduplication; the weights make that
//! exact). For a symmetric density those 8 fold into half-matrices, completed in `finish` by a transpose-add:
//!
//!   J'_{mu nu} += 2w D_{lambda sigma} v,   J'_{lambda sigma} += 2w D_{mu nu} v,              J = J' + J'^T
//!   K'_{mu lambda} += w D_{nu sigma} v,    K'_{mu sigma} += w D_{nu lambda} v,
//!   K'_{nu lambda} += w D_{mu sigma} v,    K'_{nu sigma} += w D_{mu lambda} v,              K = K' + K'^T
//!
//! A non-symmetric density takes K = K'[D] + K'[D^T]^T instead, both written straight into K (no transpose-add).
//! Its J reads the symmetric part, J[D] = J[(D + D^T)/2] since (mu nu|lambda sigma) = (mu nu|sigma lambda), and
//! J' + J'^T completes it as for a symmetric density.
//!
//! A, B, C, D are the four shells, all their contracted functions. The kernels read the density blocks D[A,B], D[C,D]
//! (J) or D[A,C], D[A,D], D[B,C], D[B,D] (K) and add into the same blocks of the output, each block a set of
//! contiguous column segments: a slot quartet's rows in a block are a segment of a column; its integrals are read
//! column by column, in storage order, as the `n_i x (n_j n_k n_l)` matrix of its component block. The density and
//! J'/K'[D] are read and written in place, at the shells' offsets. What is not a column segment of the density or
//! of the output goes through tiles from the walk's arena (the quartet's scope, each pass in a scope of its own): the
//! symmetric part for J, D^T and K'[D^T]^T for the transposed K pass; the tile's blocks are gathered, the output
//! tile's added into J or K once. J/K are completed with `symmetrize_in_place` (arrays).
//!
//! K of a symmetric density is digested by rows of the walk (`DirectFock::begin_row` ... `end_row`): over a row the
//! bra shells A and B are fixed, and every density block K reads and every output block it writes lies in their rows.
//! So per row the rows of A and of B of the density are gathered once into a strip, `(n_A + n_B) x n` column-major (A's
//! rows over B's), and K' is accumulated in a strip of the same shape, added into K' once at the row's end: the
//! quartets' K blocks are then column segments of these small strips instead of the n x n matrices. The strips are
//! filled from the density's columns and added into K' as columns, through the symmetry of the density and of
//! K' + K'^T, so that both passes over the n x n matrices are contiguous.
//!
//! J of a symmetric density by rows likewise: D[A,B] and J'[A,B] are fixed over a row, so they are a pair of tiles,
//! gathered at the row's start and added into J' at its end; D[C,D] and J'[C,D] are read and written in place. A bra of
//! one slot (segmented shells) spans its shells, so the tiles' block and a quartet's bra components are both the
//! contiguous `(i, j)`: per ket element one run of `n_i n_j` instead of `n_j` runs of `n_i`. In both
//! row kernels every block a quartet touches is cut once and walked by iterators inside, so the innermost loops carry
//! no offsets or bounds of their own.
//!
//! With the `profile` feature the kernels are timed in the fine spans `digest::j_kernel` and `digest::k_kernel`; the
//! two spins' K passes add up under the same label.

use arrays::array4d::Array4DTrait;
use arrays::matrix::{MatrixTrait, MatrixTraitMut, MatrixView, MatrixViewMut};
use arrays::scratch::{Arena, ArenaError};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;
use eri::{for_each_slot_quartet, EriBlock};

/// The four shells of a batch: their first basis function and their size (contracted functions times components).
#[derive(Clone, Copy)]
pub(crate) struct Shells {
    /// Each shell's first basis function.
    pub(crate) start: [usize; 4],
    /// Each shell's size.
    pub(crate) size: [usize; 4],
}

impl Shells {
    /// The shells of the batch `(pair_ab | pair_cd)`.
    pub(crate) fn of<S: ShellKind>(pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>) -> Self {
        Shells {
            start: [pair_ab.get_idx_a(), pair_ab.get_idx_b(), pair_cd.get_idx_a(), pair_cd.get_idx_b()],
            size: [
                pair_ab.get_n_cont_a() * pair_ab.get_degeneracy_a(),
                pair_ab.get_n_cont_b() * pair_ab.get_degeneracy_b(),
                pair_cd.get_n_cont_a() * pair_cd.get_degeneracy_a(),
                pair_cd.get_n_cont_b() * pair_cd.get_degeneracy_b(),
            ],
        }
    }
}

// The shells, as indices into `Shells`.
const A: usize = 0;
const B: usize = 1;
const C: usize = 2;
const D: usize = 3;

/// Where each shell's functions start in the matrices a kernel reads and writes: as rows at `row[X]`, as columns at
/// `col[X]`. A kernel uses the rows of A and C and the columns of B and D (J), or the rows of A and B and the columns
/// of C and D (K).
#[derive(Clone, Copy)]
struct Placement {
    row: [usize; 4],
    col: [usize; 4],
}

impl Placement {
    /// The density and J/K themselves.
    fn in_place(shells: &Shells) -> Self {
        Placement { row: shells.start, col: shells.start }
    }

    /// A tile of the shells `rows` by the shells `cols`, each pair stacked in that order: for J (A, C) by (B, D), for
    /// K (A, B) by (C, D). Its size is `(rows, cols)`.
    fn tile(shells: &Shells, (x0, x1): (usize, usize), (y0, y1): (usize, usize)) -> (Self, (usize, usize)) {
        let (mut row, mut col) = ([0; 4], [0; 4]);
        row[x1] = shells.size[x0];
        col[y1] = shells.size[y0];
        (Placement { row, col }, (shells.size[x0] + shells.size[x1], shells.size[y0] + shells.size[y1]))
    }
}

/// The tiles of the J and K digestion of one quartet whose density is not symmetric (`coulomb`, `exchange`): a density
/// tile and an output tile, each sized for the larger of J's tile, `(s_A + s_C) x (s_B + s_D)`, and K's,
/// `(s_A + s_B) x (s_C + s_D)`, so that one pair serves every call of the quartet. Taken only.
pub struct DigestTiles<'s> {
    p: &'s mut [f64],
    out: &'s mut [f64],
}

impl<'s> DigestTiles<'s> {
    /// The tiles of the quartet `(pair_ab | pair_cd)`, from `arena`.
    pub(crate) fn new_in<S: ShellKind>(pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, arena: &Arena<'s>) -> Result<Self, ArenaError> {
        let [a, b, c, d] = Shells::of(pair_ab, pair_cd).size;
        let len = ((a + c) * (b + d)).max((a + b) * (c + d));
        Ok(DigestTiles { p: arena.alloc::<f64>(len)?, out: arena.alloc::<f64>(len)? })
    }

    /// The density tile and the zeroed output tile, `rows x cols` each.
    fn views(&mut self, rows: usize, cols: usize) -> (MatrixViewMut<'_, f64>, MatrixViewMut<'_, f64>) {
        let out = &mut self.out[..rows * cols];
        out.fill(0.0);
        (MatrixViewMut::new(rows, cols, &mut self.p[..rows * cols]), MatrixViewMut::new(rows, cols, out))
    }
}

/// `J' += coeff (the batch's contribution)`; `j` is completed by `symmetrize_in_place` in `finish`. `tiles`: `None` for
/// a symmetric `p`, digested in place; else (`DigestTiles`) `p` is read through its symmetric part, which is all J
/// depends on.
#[allow(clippy::too_many_arguments)]
pub(crate) fn coulomb<S: ShellKind>(
    block: &EriBlock<'_, '_>,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
    p: &MatrixView<'_, f64>,
    j: &mut MatrixViewMut<'_, f64>,
    coeff: f64,
    tiles: Option<&mut DigestTiles<'_>>,
) {
    let shells = Shells::of(pair_ab, pair_cd);
    let Some(tiles) = tiles else {
        j_kernel(block, pair_ab, pair_cd, p, j, Placement::in_place(&shells), coeff);
        return;
    };
    let (at, (rows, cols)) = Placement::tile(&shells, (A, C), (B, D));
    // Only the blocks [A,B] and [C,D] of the density tile are written below, and only they are read.
    let (mut p_tile, mut j_tile) = tiles.views(rows, cols);
    gather(p, &shells, (A, B), TileOf::SymmetricPart, &mut p_tile, at);
    gather(p, &shells, (C, D), TileOf::SymmetricPart, &mut p_tile, at);
    j_kernel(block, pair_ab, pair_cd, &p_tile.view(), &mut j_tile, at, coeff);
    scatter(j, &shells, (A, B), false, &j_tile, at);
    scatter(j, &shells, (C, D), false, &j_tile, at);
}

/// `K' += coeff (the batch's contribution)`. `tiles`: `None` for a symmetric `p`, which leaves the half-matrix for
/// `symmetrize_in_place`; else (`DigestTiles`) K is written in full (`K'[D] + K'[D^T]^T`) and must not be symmetrized.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exchange<S: ShellKind>(
    block: &EriBlock<'_, '_>,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
    p: &MatrixView<'_, f64>,
    k: &mut MatrixViewMut<'_, f64>,
    coeff: f64,
    tiles: Option<&mut DigestTiles<'_>>,
) {
    let shells = Shells::of(pair_ab, pair_cd);
    k_kernel(block, pair_ab, pair_cd, p, k, Placement::in_place(&shells), coeff);
    let Some(tiles) = tiles else {
        return;
    };
    let (at, (rows, cols)) = Placement::tile(&shells, (A, B), (C, D));
    // Every element of the density tile is written below.
    let (mut p_tile, mut k_tile) = tiles.views(rows, cols);
    for pair in [(A, C), (A, D), (B, C), (B, D)] {
        gather(p, &shells, pair, TileOf::Transposed, &mut p_tile, at);
    }
    k_kernel(block, pair_ab, pair_cd, &p_tile.view(), &mut k_tile, at, coeff);
    for pair in [(A, C), (A, D), (B, C), (B, D)] {
        scatter(k, &shells, pair, true, &k_tile, at);
    }
}

/// The scratch a row of K by strips takes for one density over `n` functions, the row's shells A and B having `rows`
/// functions together: the density's strip and K''s.
pub(crate) fn exchange_row_len(rows: usize, n: usize) -> usize {
    2 * rows * n
}

/// The strips of the row of `pair_ab` in `row` (`exchange_row_len` long): the density's, from the symmetric `p`, and
/// K''s, zeroed.
pub(crate) fn begin_exchange_row<S: ShellKind>(pair_ab: &ShellPair<S>, p: &MatrixView<'_, f64>, row: &mut [f64]) {
    let n = p.rows();
    let (starts, sizes) = row_shells(pair_ab);
    let rows = sizes[0] + sizes[1];
    let (density, k) = row.split_at_mut(rows * n);
    k.fill(0.0);
    // Strip row r is the density's row start + r, its column start + r by symmetry: read contiguously, written with
    // the strip's (short) row stride.
    let mut r = 0;
    for (start, size) in starts.into_iter().zip(sizes) {
        for column in start..start + size {
            for (value, &p_mu) in density[r..].iter_mut().step_by(rows).zip(p.get_col(column)) {
                *value = p_mu;
            }
            r += 1;
        }
    }
}

/// `K' += coeff (the batch's contribution)` into the row's K' strip, from its density strip (`exchange_row_len` long).
/// The same updates as `k_kernel`, with every block a quartet touches cut from the strips once: per pair of ket
/// columns (c, d) the four column segments of A and B in the density strip and in K', walked by iterators inside.
pub(crate) fn exchange_in_row<S: ShellKind>(block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, row: &mut [f64], coeff: f64) {
    let _span = instrumentation::fine_span("digest::k_kernel");
    let shells = Shells::of(pair_ab, pair_cd);
    let rows = shells.size[A] + shells.size[B];
    let n = row.len() / (2 * rows);
    let (density, k) = row.split_at_mut(rows * n);
    let [n_i, n_j, n_k, n_l, ..] = block.dims();
    for_each_slot_quartet(block, pair_ab, pair_cd, |q| {
        let [qa, qb, qc, qd] = q.contracted;
        let w = coeff * q.weight;
        // The slot's rows of A and B in the strips (A's from row 0, B's below them), its columns of C and D.
        let (a, b) = (qa * n_i, shells.size[A] + qb * n_j);
        let (c, d) = (shells.start[C] + qc * n_k, shells.start[D] + qd * n_l);
        let integrals = block.slot_block(&q);
        let integrals = integrals.view_as_matrix((1, 3));
        let mut columns = integrals.iter_col();
        if c == d {
            // C and D the same functions (a single-centre ket slot): the columns of K' are one block.
            for dd in d..d + n_l {
                for cc in c..c + n_k {
                    for (jj, v) in columns.by_ref().take(n_j).enumerate() {
                        let p_col = |x: usize| &density[x * rows..(x + 1) * rows];
                        let (p_c, p_d) = (p_col(cc), p_col(dd));
                        let (p_bc_w, p_bd_w) = (w * p_c[b + jj], w * p_d[b + jj]);
                        let (mut k_bc_sum, mut k_bd_sum) = (0.0, 0.0);
                        for (((x, p_ac_i), p_ad_i), v_i) in (a..a + n_i).zip(&p_c[a..a + n_i]).zip(&p_d[a..a + n_i]).zip(v) {
                            k[cc * rows + x] += p_bd_w * v_i;
                            k[dd * rows + x] += p_bc_w * v_i;
                            k_bc_sum += p_ad_i * v_i;
                            k_bd_sum += p_ac_i * v_i;
                        }
                        k[cc * rows + b + jj] += w * k_bc_sum;
                        k[dd * rows + b + jj] += w * k_bd_sum;
                    }
                }
            }
            return;
        }
        // Disjoint column blocks of C and D: both cut once, the K' ones apart for mutable access.
        let p_cs = &density[c * rows..(c + n_k) * rows];
        let p_ds = &density[d * rows..(d + n_l) * rows];
        let (k_cs, k_ds) = if c < d {
            let (low, high) = k.split_at_mut(d * rows);
            (&mut low[c * rows..(c + n_k) * rows], &mut high[..n_l * rows])
        } else {
            let (low, high) = k.split_at_mut(c * rows);
            (&mut high[..n_k * rows], &mut low[d * rows..(d + n_l) * rows])
        };
        for (p_d, k_d) in p_ds.chunks_exact(rows).zip(k_ds.chunks_exact_mut(rows)) {
            let (p_ad, p_bd) = (&p_d[a..a + n_i], &p_d[b..b + n_j]);
            let (k_d_a, k_d_b) = k_d.split_at_mut(shells.size[A]);
            let (k_ad, k_bd) = (&mut k_d_a[a..a + n_i], &mut k_d_b[b - shells.size[A]..b - shells.size[A] + n_j]);
            for (p_c, k_c) in p_cs.chunks_exact(rows).zip(k_cs.chunks_exact_mut(rows)) {
                let (p_ac, p_bc) = (&p_c[a..a + n_i], &p_c[b..b + n_j]);
                let (k_c_a, k_c_b) = k_c.split_at_mut(shells.size[A]);
                let (k_ac, k_bc) = (&mut k_c_a[a..a + n_i], &mut k_c_b[b - shells.size[A]..b - shells.size[A] + n_j]);
                for ((((v, &p_bd_j), &p_bc_j), k_bc_j), k_bd_j) in
                    columns.by_ref().take(n_j).zip(p_bd).zip(p_bc).zip(k_bc.iter_mut()).zip(k_bd.iter_mut())
                {
                    let (p_bd_w, p_bc_w) = (w * p_bd_j, w * p_bc_j);
                    let (mut k_bc_sum, mut k_bd_sum) = (0.0, 0.0);
                    for ((((k_ac_i, k_ad_i), p_ac_i), p_ad_i), v_i) in
                        k_ac.iter_mut().zip(k_ad.iter_mut()).zip(p_ac).zip(p_ad).zip(v)
                    {
                        *k_ac_i += p_bd_w * v_i;
                        *k_ad_i += p_bc_w * v_i;
                        k_bc_sum += p_ad_i * v_i;
                        k_bd_sum += p_ac_i * v_i;
                    }
                    *k_bc_j += w * k_bc_sum;
                    *k_bd_j += w * k_bd_sum;
                }
            }
        }
    });
}

/// The scratch a row of J takes for one density, the row's shells A and B having `rows` functions together: the
/// density's block D[A,B] and J''s, `n_A x n_B` each (at most `rows^2 / 4`).
pub(crate) fn coulomb_row_len(rows: usize) -> usize {
    2 * (rows * rows / 4 + 1)
}

/// The row's tiles of J in `row` (`coulomb_row_len` long): D[A,B] from `p`, J'[A,B] zeroed.
pub(crate) fn begin_coulomb_row<S: ShellKind>(pair_ab: &ShellPair<S>, p: &MatrixView<'_, f64>, row: &mut [f64]) {
    let ([start_a, start_b], [size_a, size_b]) = row_shells(pair_ab);
    let (density, j) = row.split_at_mut(row.len() / 2);
    j[..size_a * size_b].fill(0.0);
    for (tile, column) in density.chunks_exact_mut(size_a).zip(start_b..start_b + size_b) {
        tile.copy_from_slice(&p.get_col(column)[start_a..start_a + size_a]);
    }
}

/// `J' += coeff (the batch's contribution)`: J'[C,D] in place in `j`, from `p`; J'[A,B] into the row's tile, from its
/// tile of D[A,B]. The same updates as `j_kernel`, with the blocks cut once: per ket column d the segments of D[C,d]
/// and J'[C,d], the bra tiles' columns walked by iterators inside.
pub(crate) fn coulomb_in_row<S: ShellKind>(
    block: &EriBlock<'_, '_>,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
    p: &MatrixView<'_, f64>,
    j: &mut MatrixViewMut<'_, f64>,
    row: &mut [f64],
    coeff: f64,
) {
    let _span = instrumentation::fine_span("digest::j_kernel");
    let shells = Shells::of(pair_ab, pair_cd);
    let size_a = shells.size[A];
    let (density, j_ab) = row.split_at_mut(row.len() / 2);
    let [n_i, n_j, n_k, n_l, ..] = block.dims();
    // A bra of one slot spans its shells: its tiles' block and every slot quartet's bra components are both the
    // contiguous `(i, j)`, one run of `n_i n_j` per ket element.
    let bra_whole = (n_i, n_j) == (size_a, shells.size[B]);
    for_each_slot_quartet(block, pair_ab, pair_cd, |q| {
        let [qa, qb, qc, qd] = q.contracted;
        let w = 2.0 * coeff * q.weight;
        if bra_whole {
            let n_ij = n_i * n_j;
            let (c, d) = (shells.start[C] + qc * n_k, shells.start[D] + qd * n_l);
            let (p_ab, j_ab) = (&density[..n_ij], &mut j_ab[..n_ij]);
            let integrals = block.slot_block(&q);
            for (v_l, dd) in integrals.get_data().chunks_exact(n_ij * n_k).zip(d..d + n_l) {
                let p_cd = &p.get_col(dd)[c..c + n_k];
                let j_cd = &mut j.get_col_mut(dd)[c..c + n_k];
                for ((&p_kl, j_kl_out), v) in p_cd.iter().zip(j_cd.iter_mut()).zip(v_l.chunks_exact(n_ij)) {
                    let p_kl = w * p_kl;
                    let mut j_kl = 0.0;
                    for ((j_x, p_x), v_x) in j_ab.iter_mut().zip(p_ab).zip(v) {
                        *j_x += p_kl * v_x;
                        j_kl += p_x * v_x;
                    }
                    *j_kl_out += w * j_kl;
                }
            }
            return;
        }
        // The slot's rows of A and columns of B in the tiles, its rows of C and columns of D in J' and p.
        let (a, b) = (qa * n_i, qb * n_j);
        let (c, d) = (shells.start[C] + qc * n_k, shells.start[D] + qd * n_l);
        let p_ab = &density[b * size_a..(b + n_j) * size_a];
        let j_ab = &mut j_ab[b * size_a..(b + n_j) * size_a];
        let integrals = block.slot_block(&q);
        let integrals = integrals.view_as_matrix((1, 3));
        let mut columns = integrals.iter_col();
        for dd in d..d + n_l {
            let p_cd = &p.get_col(dd)[c..c + n_k];
            let j_cd = &mut j.get_col_mut(dd)[c..c + n_k];
            for (&p_kl, j_kl_out) in p_cd.iter().zip(j_cd.iter_mut()) {
                let p_kl = w * p_kl;
                let mut j_kl = 0.0;
                for ((p_col, j_col), v) in p_ab.chunks_exact(size_a).zip(j_ab.chunks_exact_mut(size_a)).zip(columns.by_ref().take(n_j)) {
                    for ((j_i, p_i), v_i) in j_col[a..a + n_i].iter_mut().zip(&p_col[a..a + n_i]).zip(v) {
                        *j_i += p_kl * v_i;
                        j_kl += p_i * v_i;
                    }
                }
                *j_kl_out += w * j_kl;
            }
        }
    });
}

/// Adds the row's tile of J'[A,B] into `j`.
pub(crate) fn end_coulomb_row<S: ShellKind>(pair_ab: &ShellPair<S>, j: &mut MatrixViewMut<'_, f64>, row: &[f64]) {
    let ([start_a, start_b], [size_a, size_b]) = row_shells(pair_ab);
    let tile = &row[row.len() / 2..];
    for (values, column) in tile.chunks_exact(size_a).zip(start_b..start_b + size_b) {
        for (out, value) in j.get_col_mut(column)[start_a..start_a + size_a].iter_mut().zip(values) {
            *out += value;
        }
    }
}

/// Adds the row's K' strip into `k`: strip row r into K''s column start + r, which `symmetrize_in_place` completes as
/// it would the row.
pub(crate) fn end_exchange_row<S: ShellKind>(pair_ab: &ShellPair<S>, k: &mut MatrixViewMut<'_, f64>, row: &[f64]) {
    let n = k.rows();
    let (starts, sizes) = row_shells(pair_ab);
    let rows = sizes[0] + sizes[1];
    let strip = &row[rows * n..];
    let mut r = 0;
    for (start, size) in starts.into_iter().zip(sizes) {
        for column in start..start + size {
            for (out, &value) in k.get_col_mut(column).iter_mut().zip(strip[r..].iter().step_by(rows)) {
                *out += value;
            }
            r += 1;
        }
    }
}

/// The functions of a row's shells A and B together.
pub(crate) fn row_functions<S: ShellKind>(pair_ab: &ShellPair<S>) -> usize {
    let (_, sizes) = row_shells(pair_ab);
    sizes[0] + sizes[1]
}

/// The next `len` values of a row's scratch, the rest left in `row`.
pub(crate) fn take_scratch<'r>(row: &mut &'r mut [f64], len: usize) -> &'r mut [f64] {
    let (head, tail) = std::mem::take(row).split_at_mut(len);
    *row = tail;
    head
}

/// The first function and the size of a row's shells A and B.
pub(crate) fn row_shells<S: ShellKind>(pair_ab: &ShellPair<S>) -> ([usize; 2], [usize; 2]) {
    (
        [pair_ab.get_idx_a(), pair_ab.get_idx_b()],
        [pair_ab.get_n_cont_a() * pair_ab.get_degeneracy_a(), pair_ab.get_n_cont_b() * pair_ab.get_degeneracy_b()],
    )
}

/// The J' kernel: reads `p` and adds into `j`, the shells placed at `at` in both.
fn j_kernel<S: ShellKind>(
    block: &EriBlock<'_, '_>,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
    p: &MatrixView<'_, f64>,
    j: &mut MatrixViewMut<'_, f64>,
    at: Placement,
    coeff: f64,
) {
    let _span = instrumentation::fine_span("digest::j_kernel");
    let [n_i, n_j, n_k, n_l, ..] = block.dims();
    for_each_slot_quartet(block, pair_ab, pair_cd, |q| {
        let w = 2.0 * coeff * q.weight;
        // The slot's first row of A and of C, its first column of B and of D.
        let [a, b, c, d] = [
            at.row[A] + q.contracted[0] * n_i,
            at.col[B] + q.contracted[1] * n_j,
            at.row[C] + q.contracted[2] * n_k,
            at.col[D] + q.contracted[3] * n_l,
        ];
        let integrals = block.slot_block(&q);
        let integrals = integrals.view_as_matrix((1, 3));
        let mut columns = integrals.iter_col();
        for ll in d..d + n_l {
            for kk in c..c + n_k {
                let p_kl = w * p[(kk, ll)];
                let mut j_kl = 0.0;
                for (jj, v) in columns.by_ref().take(n_j).enumerate() {
                    let p_col = &p.get_col(b + jj)[a..a + n_i];
                    let j_col = &mut j.get_col_mut(b + jj)[a..a + n_i];
                    for ((j_i, p_i), v_i) in j_col.iter_mut().zip(p_col).zip(v) {
                        *j_i += p_kl * v_i;
                        j_kl += p_i * v_i;
                    }
                }
                j[(kk, ll)] += w * j_kl;
            }
        }
    });
}

/// The K' kernel: reads `p` and adds into `k`, the shells placed at `at` in both.
fn k_kernel<S: ShellKind>(
    block: &EriBlock<'_, '_>,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
    p: &MatrixView<'_, f64>,
    k: &mut MatrixViewMut<'_, f64>,
    at: Placement,
    coeff: f64,
) {
    let _span = instrumentation::fine_span("digest::k_kernel");
    let [n_i, n_j, n_k, n_l, ..] = block.dims();
    for_each_slot_quartet(block, pair_ab, pair_cd, |q| {
        let w = coeff * q.weight;
        // The slot's first row of A and of B, its first column of C and of D.
        let [a, b, c, d] = [
            at.row[A] + q.contracted[0] * n_i,
            at.row[B] + q.contracted[1] * n_j,
            at.col[C] + q.contracted[2] * n_k,
            at.col[D] + q.contracted[3] * n_l,
        ];
        let integrals = block.slot_block(&q);
        let integrals = integrals.view_as_matrix((1, 3));
        let mut columns = integrals.iter_col();
        for dd in d..d + n_l {
            for cc in c..c + n_k {
                for (jj, v) in columns.by_ref().take(n_j).enumerate() {
                    let bb = b + jj;
                    let (p_bd_w, p_bc_w) = (w * p[(bb, dd)], w * p[(bb, cc)]);
                    let p_ac_col = &p.get_col(cc)[a..a + n_i];
                    let p_ad_col = &p.get_col(dd)[a..a + n_i];
                    let (k_bc_sum, k_bd_sum) = if cc == dd {
                        // K'[A,C] and K'[A,D] are one column here (C = D, in place): one update with both weights.
                        let k_col = &mut k.get_col_mut(cc)[a..a + n_i];
                        let mut sum = 0.0;
                        for ((k_i, p_i), v_i) in k_col.iter_mut().zip(p_ac_col).zip(v) {
                            *k_i += (p_bd_w + p_bc_w) * v_i;
                            sum += p_i * v_i;
                        }
                        (sum, sum)
                    } else {
                        let (k_ac_col, k_ad_col) = k.get_two_cols_mut(cc, dd);
                        let (k_ac_col, k_ad_col) = (&mut k_ac_col[a..a + n_i], &mut k_ad_col[a..a + n_i]);
                        let (mut k_bc_sum, mut k_bd_sum) = (0.0, 0.0);
                        for ((((k_ac_i, k_ad_i), p_ac_i), p_ad_i), v_i) in
                            k_ac_col.iter_mut().zip(k_ad_col.iter_mut()).zip(p_ac_col).zip(p_ad_col).zip(v)
                        {
                            *k_ac_i += p_bd_w * v_i;
                            *k_ad_i += p_bc_w * v_i;
                            k_bc_sum += p_ad_i * v_i;
                            k_bd_sum += p_ac_i * v_i;
                        }
                        (k_bc_sum, k_bd_sum)
                    };
                    k[(bb, cc)] += w * k_bc_sum;
                    k[(bb, dd)] += w * k_bd_sum;
                }
            }
        }
    });
}

/// What a density tile holds, for the shells `(X, Y)`.
#[derive(Clone, Copy)]
enum TileOf {
    /// `p^T[X, Y] = p[Y, X]`.
    Transposed,
    /// `(p[X, Y] + p[Y, X]) / 2`, the symmetric part.
    SymmetricPart,
}

/// The block `(x, y)` of the density tile `tile`, the shells placed at `at` in it, from `p`.
fn gather(p: &MatrixView<'_, f64>, shells: &Shells, (x, y): (usize, usize), of: TileOf, tile: &mut MatrixViewMut<'_, f64>, at: Placement) {
    let (start_x, start_y, size_x, size_y) = (shells.start[x], shells.start[y], shells.size[x], shells.size[y]);
    let (row, col) = (at.row[x], at.col[y]);
    for c in 0..size_y {
        let column = &p.get_col(start_y + c)[start_x..start_x + size_x];
        let out = &mut tile.get_col_mut(col + c)[row..row + size_x];
        // A row of p: scattered.
        for (r, (value, direct)) in out.iter_mut().zip(column).enumerate() {
            let transposed = p[(start_y + c, start_x + r)];
            *value = match of {
                TileOf::Transposed => transposed,
                TileOf::SymmetricPart => 0.5 * (direct + transposed),
            };
        }
    }
}

/// `m[X, Y] += tile` for the block `(x, y)` of `tile`, the shells placed at `at` in it (`transposed`:
/// `m[Y, X] += tile^T`).
fn scatter(m: &mut MatrixViewMut<'_, f64>, shells: &Shells, (x, y): (usize, usize), transposed: bool, tile: &MatrixViewMut<'_, f64>, at: Placement) {
    let (start_x, start_y, size_x, size_y) = (shells.start[x], shells.start[y], shells.size[x], shells.size[y]);
    let (row, col) = (at.row[x], at.col[y]);
    for c in 0..size_y {
        let values = &tile.get_col(col + c)[row..row + size_x];
        if transposed {
            // A row of m: scattered.
            for (r, value) in values.iter().enumerate() {
                m[(start_y + c, start_x + r)] += value;
            }
        } else {
            for (out, value) in m.get_col_mut(start_y + c)[start_x..start_x + size_x].iter_mut().zip(values) {
                *out += value;
            }
        }
    }
}
