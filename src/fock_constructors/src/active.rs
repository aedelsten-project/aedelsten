// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The active-space accumulator: the J/K of an inner direct accumulator (`InactiveActiveDirectFock` for CASSCF,
//! `GeneralDirectFock` for a state interaction) plus, from the same walk, the Coulomb matrices of the active pair
//! densities
//!
//!   M(vw, mu nu) = sum_{lambda sigma} P_{lambda sigma, vw} (mu nu|lambda sigma),
//!   P_{lambda sigma, vw} = 1/2 (B_lambda,v K_sigma,w + B_sigma,v K_lambda,w),
//!
//! for active coefficients B (the bra's) and K (the ket's). In one orbital basis (CASSCF), B = K = C and P is
//! symmetric in v <-> w, so only v >= w is formed; in two (a state interaction between runs with different orbitals)
//! every pair (v, w) is. M is symmetric in mu <-> nu and held packed (`eri::packed::PackedPairTensor`, the pairs vw
//! fastest); the consumer makes `T(u, vw, mu) = sum_nu K_nu,u M(vw, mu nu)` and `(tu|vw) = sum_mu B_mu,t T(u, vw, mu)`
//! from it by products. P is symmetric in lambda <-> sigma either way, which is what the kernel below relies on.
//!
//! Per batch (AB|CD), with the slot weights w of `eri::for_each_slot_quartet` on the full stored block, every element
//! stands for its 8 permutations: 4 bra placements, (mu nu), (nu mu), (lambda sigma), (sigma lambda), each with both
//! ket orders, which P's symmetry collapses into a factor 2. With the pair densities Pab[xy, vw] = 2 P_{a_x b_y, vw}
//! over the functions x of A and y of B, and Pcd likewise:
//!
//!   bra AB:  X(xy, vw) += w sum_kl V(xy, kl) Pcd[kl, vw],   M(vw, a_x b_y) += X(xy, vw)
//!   ket CD:  Y(kl, vw) += w sum_xy Pab[xy, vw] V(xy, kl),   M(vw, c_k d_l) += Y(kl, vw)
//!
//! both added into the packed pair once, twice on the diagonal mu = nu (which the two placements of a full matrix both
//! reach). X, Y and the pair densities are indexed by the functions of their shells (not by slot), so the same code
//! takes a batch as computed or merged into function order (`eri::with_merged_slots`, which the walk applies:
//! `merged_slots`); merged, a large batch is two products. The walk goes by rows (`eri::DirectFock::begin_row` ...
//! `end_row`): over a row the bra shells A and B are fixed, so their coefficients and pair densities Pab are made once
//! a row, and X is summed over the row's quartets and added into M once at its end. Per quartet only Pcd and Y remain.
//! Everything is held with vw fastest, so every inner loop is a contiguous run over the pairs. The small slot quartets
//! are loops; from `PRODUCT_MIN` up a bra that spans its shells goes through two products. The J/K of the inner
//! accumulator are digested by rows too, in their own part of the row's scratch.
//!
//! With the `profile` feature the per-quartet kernel is timed in the fine span `active::m_kernel` and the rows' adds
//! into M in `active::m_row`.

use arrays::matrix::{ColumnFactor, Factor, MatrixTrait, MatrixView, MatrixViewMut};
use arrays::scratch::{Arena, ArenaError};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;
use eri::packed::PackedPairTensorViewMut;
use eri::{for_each_slot_quartet, with_merged_slots, DirectFock, EriBlock, ScreeningCoefficients, ScreeningDensities, TensorChoice};

use crate::digest::{row_functions, row_shells, take_scratch, Shells};
use aedelsten_mpi::Communicator;

use crate::direct::{allreduce_or_panic, InactiveActiveDirectFock};

/// The active coefficients of the pair densities (see the top of this file): one orbital basis, `B = K = C`, or a
/// bra's and a ket's.
enum ActiveBases<'a> {
    One(MatrixView<'a, f64>),
    Two { bra: MatrixView<'a, f64>, ket: MatrixView<'a, f64> },
}

impl ActiveBases<'_> {
    /// `K`: the ket's.
    fn ket(&self) -> &MatrixView<'_, f64> {
        match self {
            ActiveBases::One(c) | ActiveBases::Two { ket: c, .. } => c,
        }
    }

    /// `B`, where it differs from `K`.
    fn bra(&self) -> Option<&MatrixView<'_, f64>> {
        match self {
            ActiveBases::One(_) => None,
            ActiveBases::Two { bra, .. } => Some(bra),
        }
    }

    /// The number of pairs `(v, w)`: `v >= w` in one basis, all of them in two.
    fn n_pairs(&self) -> usize {
        let m = self.ket().cols();
        match self {
            ActiveBases::One(_) => m * (m + 1) / 2,
            ActiveBases::Two { .. } => m * m,
        }
    }

    /// How many coefficient sets a row gathers: the ket's, and the bra's in two bases.
    fn count(&self) -> usize {
        match self {
            ActiveBases::One(_) => 1,
            ActiveBases::Two { .. } => 2,
        }
    }
}

/// The direct J/K of `jk`, plus the Coulomb matrices `M` of the active pair densities (see the top of this file) from
/// the same walk.
///
/// `m` is `M(vw, mu nu)`, `n (n + 1)/2 n_pairs` values packed over the function pairs `mu >= nu`, the pairs `vw`
/// fastest, zeroed here: in one basis (`new`) `n_pairs = n_act (n_act + 1)/2` and the pair `v >= w` is at
/// `v (v + 1)/2 + w`; in two (`mixed`) `n_pairs = n_act^2` and the pair `(v, w)` is at `v + w n_act`. The coefficients
/// are `n x n_act` views and `m` a view (as the J/K of `jk`): a coefficient matrix may be the active column block of
/// the whole orbital matrix, `m` the caller's buffer. `active_threshold`: a batch is skipped for `M` (its J/K are
/// still built) when what it adds to `T = K^T M` is below it: its Schwarz bound times
/// `max(max(K_A, K_B) P_CD, max(K_C, K_D) P_AB)`, the largest pair density of one shell pair times the largest ket
/// coefficient on the other, whose functions are the pair `mu nu` it adds to, so that a screened `M` is complete for
/// `T` only; `None` adds every batch. Under MPI, `M` is summed across ranks with the J/K.
pub struct ActiveSpaceDirectFock<'a, JK = InactiveActiveDirectFock<'a>> {
    jk: JK,
    bases: ActiveBases<'a>,
    m: PackedPairTensorViewMut<'a, f64>,
    active_threshold: Option<f64>,
    /// The largest active coefficient on the row's shells A and B: the ket's, then the bra's (the ket's again in one
    /// basis). Set by `begin_row`.
    row_largest: [f64; 4],
}

impl<'a, JK: DirectFock> ActiveSpaceDirectFock<'a, JK> {
    /// One orbital basis, `c_active` (CASSCF): the pairs `v >= w`.
    pub fn new(jk: JK, c_active: MatrixView<'a, f64>, m: PackedPairTensorViewMut<'a, f64>, active_threshold: Option<f64>) -> Self {
        Self::with_bases(jk, ActiveBases::One(c_active), m, active_threshold)
    }

    /// Two orbital bases, `c_bra` and `c_ket` (a state interaction): every pair `(v, w)`.
    pub fn mixed(jk: JK, c_bra: MatrixView<'a, f64>, c_ket: MatrixView<'a, f64>, m: PackedPairTensorViewMut<'a, f64>, active_threshold: Option<f64>) -> Self {
        assert_eq!((c_bra.rows(), c_bra.cols()), (c_ket.rows(), c_ket.cols()), "the bra's and the ket's active coefficients have one shape");
        Self::with_bases(jk, ActiveBases::Two { bra: c_bra, ket: c_ket }, m, active_threshold)
    }

    fn with_bases(jk: JK, bases: ActiveBases<'a>, mut m: PackedPairTensorViewMut<'a, f64>, active_threshold: Option<f64>) -> Self {
        assert_eq!((m.n(), m.n_ket()), (bases.ket().rows(), bases.n_pairs()), "M is packed over the n functions' pairs, n_pairs values each");
        m.get_data_mut().fill(0.0);
        ActiveSpaceDirectFock { jk, bases, m, active_threshold, row_largest: [0.0; 4] }
    }

    /// The scratch M's part of a row takes, its shells A and B having `rows` functions together: the ket's
    /// coefficients on them (and the bra's, in two bases), `n_act` per function; the bra's pair densities and X,
    /// `n_pairs` per pair of functions of A and B, at most `rows^2 / 4` pairs (`row_pairs`).
    fn coulomb_row_len(&self, rows: usize) -> usize {
        let (m, q) = (self.bases.ket().cols(), self.bases.n_pairs());
        if m == 0 {
            return 0;
        }
        self.bases.count() * m * rows + 2 * q * row_pairs(rows)
    }

    /// M's part of a row: the coefficients of A and B gathered and their largest elements, the bra's pair densities
    /// `Pab`, X zeroed.
    fn begin_coulomb_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let (m, q) = (self.bases.ket().cols(), self.bases.n_pairs());
        if m == 0 {
            return;
        }
        let (starts, sizes) = row_shells(pair_ab);
        let rows = sizes[0] + sizes[1];
        let ket = take_scratch(&mut row, m * rows);
        gather_rows(self.bases.ket(), starts, sizes, ket);
        let bra = self.bases.bra().map(|c| {
            let bra = take_scratch(&mut row, m * rows);
            gather_rows(c, starts, sizes, bra);
            &*bra
        });
        let ket = &*ket;
        let split = m * sizes[0];
        let (ka, kb) = (largest(&ket[..split]), largest(&ket[split..]));
        let (ba, bb) = bra.map_or((ka, kb), |bra| (largest(&bra[..split]), largest(&bra[split..])));
        self.row_largest = [ka, kb, ba, bb];

        let (pab, x) = row.split_at_mut(q * row_pairs(rows));
        let functions = sizes[0] * sizes[1];
        pair_densities(ket, bra, m, sizes, &mut pab[..q * functions]);
        x[..q * functions].fill(0.0);
    }

    /// The tiles `coulomb_in_row` takes for `block` of `(pair_ab|pair_cd)`: the ket's (and the bra's) coefficients of C
    /// and D, the pair densities and Y over C and D's functions, and their ket columns for a quartet digested by
    /// products. Empty without active orbitals. Taken only.
    fn m_tiles<'s, S: ShellKind>(&self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, arena: &Arena<'s>) -> Result<MTiles<'s>, ArenaError> {
        let (m, q) = (self.bases.ket().cols(), self.bases.n_pairs());
        if m == 0 {
            return Ok(MTiles::default());
        }
        let shells = Shells::of(pair_ab, pair_cd);
        let [n_i, n_j, n_k, n_l, ..] = block.dims();
        let (sizes, functions) = ([shells.size[2], shells.size[3]], shells.size[2] * shells.size[3]);
        let (n_xy, n_kl) = (n_i * n_j, n_k * n_l);
        let by_products = n_xy * n_kl >= PRODUCT_MIN && n_xy == shells.size[0] * shells.size[1];
        let tile_len = if by_products { q * n_kl } else { 0 };
        Ok(MTiles {
            ket: arena.alloc::<f64>(m * (sizes[0] + sizes[1]))?,
            bra: match self.bases.bra() {
                Some(_) => Some(arena.alloc::<f64>(m * (sizes[0] + sizes[1]))?),
                None => None,
            },
            pcd: arena.alloc::<f64>(q * functions)?,
            y: arena.alloc::<f64>(q * functions)?,
            pcd_tile: arena.alloc::<f64>(tile_len)?,
            y_tile: arena.alloc::<f64>(tile_len)?,
        })
    }

    /// M's part of one quartet (or chunk) of the row: X += its bra half, and its ket half Y added into M, with `tiles`
    /// (`m_tiles` of the same block).
    fn coulomb_in_row<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, row: &mut [f64], tiles: &mut MTiles<'_>) {
        let ActiveSpaceDirectFock { bases, m: coulomb, active_threshold, row_largest, .. } = self;
        let (m, q) = (bases.ket().cols(), bases.n_pairs());
        if m == 0 {
            return;
        }
        let shells = Shells::of(pair_ab, pair_cd);
        let [n_i, n_j, n_k, n_l, ..] = block.dims();
        let rows = shells.size[0] + shells.size[1];
        let (_, rest) = row.split_at_mut(bases.count() * m * rows);
        let (pab, x) = rest.split_at_mut(q * row_pairs(rows));
        let (starts, sizes) = ([shells.start[2], shells.start[3]], [shells.size[2], shells.size[3]]);
        let size_a = shells.size[0];
        let MTiles { ket, bra, pcd, y, pcd_tile, y_tile } = tiles;

        // The coefficients of C and D, C's functions first.
        gather_rows(bases.ket(), starts, sizes, ket);
        if let (Some(c), Some(bra)) = (bases.bra(), bra.as_deref_mut()) {
            gather_rows(c, starts, sizes, bra);
        }
        let (ket, bra) = (&**ket, bra.as_deref());
        if let (Some(bound), Some(threshold)) = (block.bound(), *active_threshold) {
            // |P_xy| <= max(B_x K_y, B_y K_x); the pair the quartet adds to carries K in T = K^T M.
            let [ka, kb, ba, bb] = *row_largest;
            let split = m * sizes[0];
            let (kc, kd) = (largest(&ket[..split]), largest(&ket[split..]));
            let (bc, bd) = bra.map_or((kc, kd), |bra| (largest(&bra[..split]), largest(&bra[split..])));
            let (p_ab, p_cd) = ((ba * kb).max(bb * ka), (bc * kd).max(bd * kc));
            if bound * (ka.max(kb) * p_cd).max(kc.max(kd) * p_ab) < threshold {
                return;
            }
        }
        let _span = instrumentation::fine_span("active::m_kernel");
        let functions = sizes[0] * sizes[1];
        let (pcd, y) = (&mut pcd[..q * functions], &mut y[..q * functions]);
        pair_densities(ket, bra, m, sizes, pcd);
        y.fill(0.0);
        let (pcd_tile, y_tile) = (&mut **pcd_tile, &mut **y_tile);

        // One pass over the integrals: per element V(xy, kl), X[xy] += w V Pcd[kl] and Y[kl] += w V Pab[xy]. A
        // large slot quartet whose bra spans its shells (merged, or one bra slot) as two products instead,
        // X += w Pcd V^T and Y += w Pab V, its ket's columns of Pcd and Y through tiles.
        let data = block.data();
        let (n_xy, n_kl) = (n_i * n_j, n_k * n_l);
        let by_products = n_xy * n_kl >= PRODUCT_MIN && n_xy == size_a * shells.size[1];
        for_each_slot_quartet(block, pair_ab, pair_cd, |slot| {
            let [a, b, c, d] = [0, 1, 2, 3].map(|x| slot.functions[x] - shells.start[x]);
            let sub = &data[slot.offset..][..n_i * n_j * n_k * n_l];
            if by_products {
                let ket_columns = || (0..n_l).flat_map(move |l| (0..n_k).map(move |k| c + k + sizes[0] * (d + l)));
                for (tile, kl) in pcd_tile.chunks_exact_mut(q).zip(ket_columns()) {
                    tile.copy_from_slice(&pcd[q * kl..][..q]);
                }
                let v = MatrixView::new(n_xy, n_kl, sub);
                MatrixView::new(q, n_kl, &*pcd_tile).matmul_scaled_prealloc(&v.view_transposed(), &mut MatrixViewMut::new(q, n_xy, &mut x[..q * n_xy]), slot.weight, 1.0);
                MatrixView::new(q, n_xy, &pab[..q * n_xy]).matmul_scaled_prealloc(&v, &mut MatrixViewMut::new(q, n_kl, &mut *y_tile), slot.weight, 0.0);
                for (tile, kl) in y_tile.chunks_exact(q).zip(ket_columns()) {
                    for (out, value) in y[q * kl..][..q].iter_mut().zip(tile) {
                        *out += value;
                    }
                }
                return;
            }
            for (l, v_l) in sub.chunks_exact(n_i * n_j * n_k).enumerate() {
                for (k, v_kl) in v_l.chunks_exact(n_i * n_j).enumerate() {
                    let kl = c + k + sizes[0] * (d + l);
                    let (pcd_kl, y_kl) = (&pcd[q * kl..][..q], &mut y[q * kl..][..q]);
                    for (j, v_j) in v_kl.chunks_exact(n_i).enumerate() {
                        let xy = a + size_a * (b + j);
                        let (x_j, pab_j) = (&mut x[q * xy..][..q * n_i], &pab[q * xy..][..q * n_i]);
                        for ((&v, x_xy), pab_xy) in v_j.iter().zip(x_j.chunks_exact_mut(q)).zip(pab_j.chunks_exact(q)) {
                            let wv = slot.weight * v;
                            for (((x_e, y_e), &pcd_e), &pab_e) in x_xy.iter_mut().zip(y_kl.iter_mut()).zip(pcd_kl).zip(pab_xy) {
                                *x_e += wv * pcd_e;
                                *y_e += wv * pab_e;
                            }
                        }
                    }
                }
            }
        });

        // The ket half into M. Pairs of functions no stored slot holds have Y zero.
        for (cd, y_cd) in y.chunks_exact(q).enumerate() {
            add_pair(coulomb, starts[0] + cd % sizes[0], starts[1] + cd / sizes[0], y_cd);
        }
    }

    /// M's part of the row's end: X added into M.
    fn end_coulomb_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, row: &mut [f64]) {
        let (m, q) = (self.bases.ket().cols(), self.bases.n_pairs());
        if m == 0 {
            return;
        }
        let _span = instrumentation::fine_span("active::m_row");
        let (starts, sizes) = row_shells(pair_ab);
        let rows = sizes[0] + sizes[1];
        let x = &row[self.bases.count() * m * rows + q * row_pairs(rows)..][..q * sizes[0] * sizes[1]];
        for (ab, x_ab) in x.chunks_exact(q).enumerate() {
            add_pair(&mut self.m, starts[0] + ab % sizes[0], starts[1] + ab / sizes[0], x_ab);
        }
    }
}

/// `M(vw, mu nu) += values[vw]` for the functions `mu` and `nu`, twice for `mu = nu`: a full matrix receives an element
/// at `(mu, nu)` and at `(nu, mu)`.
#[inline(always)]
fn add_pair(m: &mut PackedPairTensorViewMut<'_, f64>, mu: usize, nu: usize, values: &[f64]) {
    if values.iter().all(|&v| v == 0.0) {
        return;
    }
    let scale = if mu == nu { 2.0 } else { 1.0 };
    for (out, value) in m.pair_mut(mu, nu).iter_mut().zip(values) {
        *out += scale * value;
    }
}

/// The slot quartets from this size up (`n_xy n_kl`, bra by ket components) go through two products in
/// `coulomb_in_row`; below it the loops are faster than the products' call.
const PRODUCT_MIN: usize = 256;

/// The most pairs of functions a row whose shells A and B have `rows` functions together can have:
/// `n_A n_B <= rows^2 / 4`.
fn row_pairs(rows: usize) -> usize {
    rows * rows / 4 + 1
}

/// `2 P_{xy, vw}` (`pair_density`) for every function `x` of the first shell and `y` of the second, from their gathered
/// coefficients (`gather_rows`, the ket's and, in two bases, the bra's), `q` per `(x, y)`, `x` fastest.
fn pair_densities(ket: &[f64], bra: Option<&[f64]>, m: usize, sizes: [usize; 2], out: &mut [f64]) {
    let q = out.len() / (sizes[0] * sizes[1]);
    for (xy, out) in out.chunks_exact_mut(q).enumerate() {
        let (x, y) = (xy % sizes[0], sizes[0] + xy / sizes[0]);
        pair_density(function(ket, m, x), function(ket, m, y), bra.map(|bra| (function(bra, m, x), function(bra, m, y))), out);
    }
}

/// The coefficients of `c` (`n x m`) on the functions of two shells, `starts[s]..starts[s] + sizes[s]`, into `out`:
/// one function's `m` after another, the first shell's functions first.
fn gather_rows(c: &MatrixView<'_, f64>, starts: [usize; 2], sizes: [usize; 2], out: &mut [f64]) {
    let m = c.cols();
    for u in 0..m {
        let column = c.get_col(u);
        let mut x = 0;
        for (start, size) in starts.into_iter().zip(sizes) {
            // A transpose: written with the stride m.
            for &value in &column[start..start + size] {
                out[x * m + u] = value;
                x += 1;
            }
        }
    }
}

/// Function `x`'s `m` coefficients in a gathered set (`gather_rows`).
#[inline(always)]
fn function(gathered: &[f64], m: usize, x: usize) -> &[f64] {
    &gathered[x * m..(x + 1) * m]
}

/// The largest magnitude in `values`.
fn largest(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |max, c| max.max(c.abs()))
}

/// `out[vw] = B_x[v] K_y[w] + B_y[v] K_x[w]`, `2 P_{xy, vw}`, for the functions `x` and `y` with the ket's
/// coefficients `ket_x`, `ket_y` and the bra's `bra` (`None`: one basis, B = K): in one basis every `v >= w`, at
/// `v (v + 1)/2 + w`; in two every pair, at `v + w n_act`. Written in order.
#[inline(always)]
fn pair_density(ket_x: &[f64], ket_y: &[f64], bra: Option<(&[f64], &[f64])>, out: &mut [f64]) {
    let Some((bra_x, bra_y)) = bra else {
        let mut out = out.iter_mut();
        for (v, (xv, yv)) in ket_x.iter().zip(ket_y).enumerate() {
            for ((xw, yw), value) in ket_x[..=v].iter().zip(&ket_y[..=v]).zip(out.by_ref()) {
                *value = xv * yw + yv * xw;
            }
        }
        return;
    };
    let m = ket_x.len();
    for ((xw, yw), column) in ket_x.iter().zip(ket_y).zip(out.chunks_exact_mut(m)) {
        for ((value, xv), yv) in column.iter_mut().zip(bra_x).zip(bra_y) {
            *value = xv * yw + yv * xw;
        }
    }
}

// By rows: the inner accumulator's scratch first (its `row_len`), then M's (`coulomb_row_len`).
/// `ActiveSpaceDirectFock::m_tiles`.
#[derive(Default)]
pub struct MTiles<'s> {
    ket: &'s mut [f64],
    bra: Option<&'s mut [f64]>,
    pcd: &'s mut [f64],
    y: &'s mut [f64],
    pcd_tile: &'s mut [f64],
    y_tile: &'s mut [f64],
}

impl<JK: DirectFock> DirectFock for ActiveSpaceDirectFock<'_, JK> {
    // The walk skips a quartet only if the J/K's densities and M's pair densities both allow it; without an active
    // threshold M needs every quartet, and so does a J/K that does not screen.
    fn screening_densities(&self) -> Option<ScreeningDensities<'_>> {
        let threshold = self.active_threshold?;
        let mut densities = self.jk.screening_densities()?;
        let ket = self.bases.ket();
        densities.pair_densities = Some(ScreeningCoefficients {
            m: ket.cols(),
            bra: self.bases.bra().unwrap_or(ket).get_data(),
            ket: ket.get_data(),
            threshold,
        });
        Some(densities)
    }

    fn merged_slots(&self) -> bool {
        true
    }

    fn row_len(&self, rows: usize) -> usize {
        self.jk.row_len(rows) + self.coulomb_row_len(rows)
    }

    fn row_len_at(&self, rows: usize, n: usize) -> usize {
        self.jk.row_len_at(rows, n) + self.coulomb_row_len(rows)
    }

    fn begin_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let jk = take_scratch(&mut row, self.jk.row_len(row_functions(pair_ab)));
        self.jk.begin_row(pair_ab, jk);
        self.begin_coulomb_row(pair_ab, row);
    }

    type Tiles<'s> = (JK::Tiles<'s>, MTiles<'s>);

    fn digestion_tiles<'s, S: ShellKind>(&self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, arena: &Arena<'s>) -> Result<Self::Tiles<'s>, ArenaError> {
        Ok((self.jk.digestion_tiles(block, pair_ab, pair_cd, arena)?, self.m_tiles(block, pair_ab, pair_cd, arena)?))
    }

    fn distribute_block_in_row<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, mut row: &mut [f64], tiles: &mut Self::Tiles<'_>) {
        let jk = take_scratch(&mut row, self.jk.row_len(row_functions(pair_ab)));
        self.jk.distribute_block_in_row(block, pair_ab, pair_cd, jk, &mut tiles.0);
        self.coulomb_in_row(block, pair_ab, pair_cd, row, &mut tiles.1);
    }

    fn end_row<S: ShellKind>(&mut self, pair_ab: &ShellPair<S>, mut row: &mut [f64]) {
        let jk = take_scratch(&mut row, self.jk.row_len(row_functions(pair_ab)));
        self.jk.end_row(pair_ab, jk);
        self.end_coulomb_row(pair_ab, row);
    }
}

impl<JK: DirectFock> TensorChoice for ActiveSpaceDirectFock<'_, JK> {
    fn distribute(&mut self, _: usize, _: usize, _: usize, _: usize, _: f64) {
        unreachable!("ActiveSpaceDirectFock digests whole batches (distribute_block); distribute is never called")
    }

    // Outside a row walk: the J/K as their accumulator digests a lone batch, M as a row of this one quartet, merged as
    // the walk merges it.
    fn distribute_block<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, scratch: &mut Arena<'_>) -> Result<(), ArenaError> {
        self.jk.distribute_block(block, pair_ab, pair_cd, scratch)?;
        scratch.scope(|lone| {
            let row = lone.alloc::<f64>(self.coulomb_row_len(row_functions(pair_ab)))?;
            self.begin_coulomb_row(pair_ab, row);
            with_merged_slots(block, pair_ab, pair_cd, lone, |merged, arena| {
                let mut tiles = self.m_tiles(merged, pair_ab, pair_cd, arena)?;
                self.coulomb_in_row(merged, pair_ab, pair_cd, row, &mut tiles);
                Ok(())
            })?;
            self.end_coulomb_row(pair_ab, row);
            Ok(())
        })
    }

    fn prepare(&mut self) {
        self.jk.prepare();
    }

    fn finish(&mut self) {
        self.jk.finish();
    }

    // As the J/K accumulators': ranks own disjoint shell quartets, so the partial M simply add.
    fn wants_mpi_partition(&self) -> bool {
        true
    }

    fn allreduce_sum(&mut self, comm: &Communicator) {
        self.jk.allreduce_sum(comm);
        allreduce_or_panic(comm, self.m.get_data_mut(), "M (active pair densities' Coulomb matrices)");
    }
}
