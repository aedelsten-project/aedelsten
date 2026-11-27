// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The direct walks' data, built once per geometry: per shell pair surviving the Schwarz threshold its Hermite
//! coefficients and bound (`EriData`, `derive_eri_data`), in segments by component count; the modelled cost of a
//! quartet (`CostFactors`), the pieces of rows the direct walks hand out over the ranks (`Handout`) and the rows a
//! rank owns in the stored tensor's walk (`owned_rows`).

use aedelsten_mpi::{Communicator, Counter};
use arrays::array4d::*;
use arrays::array6d::*;
use arrays::array8d::*;
use arrays::matrix::*;
use arrays::scratch::*;

use basis::contraction::ShellKind;
use basis::shell_pairs::*;
use mmd::ketside_hermite_coefficients::{compute_ketside_hermite_coefficients, MMDCoefficients};
use mmd::hermite_coulomb_integrals::FullRange;

use std::ops::Range;

use crate::batch::{compute_eri_batch, ERIBatchType};

/// One shell pair's entry in `EriData`.
#[derive(Debug)]
pub struct EriDataEntry {
    /// The pair's index in the `shell_pairs` it was derived from.
    pub original_index: usize,
    /// Its Hermite coefficients.
    pub(crate) coeffs: MMDCoefficients,
    /// Its Schwarz bound, `sqrt(max |(ab|ab)|)`.
    pub schwarz_bound: f64,
    // The pair's two shells, indices into `EriData::shells`.
    pub(crate) shells: (usize, usize),
    // The pair's components (`ShellPair::components`), its segment.
    components: usize,
}

/// The data of every shell pair that survived the Schwarz threshold (`derive_eri_data`), each entry with its index
/// into the `shell_pairs` it came from. The entries are in segments of one component count each, in descending order
/// of it (a quartet puts the earlier pair in the bra: the one with more components, as the generated kernels take
/// it), and within a segment in descending order of the Schwarz bound. Derived once per geometry and basis, reused by
/// every direct walk.
#[derive(Debug)]
pub struct EriData {
    entries: Vec<EriDataEntry>,
    max_bound: f64,
    // Every shell of the pairs, `(first AO, AO count)`, by first AO.
    pub(crate) shells: Box<[(usize, usize)]>,
    // Where each segment ends.
    segment_ends: Box<[usize]>,
}

impl EriData {
    fn new(mut entries: Vec<EriDataEntry>, shells: Box<[(usize, usize)]>) -> Self {
        entries.sort_by(|a, b| b.components.cmp(&a.components).then(b.schwarz_bound.total_cmp(&a.schwarz_bound)));
        let mut data = EriData { entries, max_bound: 0.0, shells, segment_ends: Box::new([]) };
        data.index_segments();
        data
    }

    fn index_segments(&mut self) {
        self.max_bound = self.entries.iter().map(|e| e.schwarz_bound).fold(0.0_f64, f64::max);
        let entries = &self.entries;
        self.segment_ends = (0..entries.len()).filter(|&i| i + 1 == entries.len() || entries[i + 1].components != entries[i].components).map(|i| i + 1).collect();
    }

    /// The entries `kets` cut at the segment boundaries, in order: each range within one segment, in descending
    /// order of the bound.
    pub(super) fn segments_in(&self, kets: Range<usize>) -> impl Iterator<Item = Range<usize>> + '_ {
        let mut start = kets.start;
        self.segment_ends.iter().filter(move |&&end| end > kets.start).map_while(move |&end| {
            let range = start..end.min(kets.end);
            start = end;
            (!range.is_empty()).then_some(range)
        })
    }

    /// Bytes held: the entries plus every entry's Hermite coefficients.
    pub fn approx_memory_bytes(&self) -> usize {
        let f64_bytes = std::mem::size_of::<f64>();
        let coeff_bytes: usize = self.entries.iter().map(|entry| entry.coeffs.get_data().len() * f64_bytes).sum();
        std::mem::size_of_val(self.entries.as_slice()) + coeff_bytes
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries, in walk order.
    pub fn entries(&self) -> &[EriDataEntry] {
        &self.entries
    }

    /// The largest Schwarz bound.
    pub fn max_bound(&self) -> f64 {
        self.max_bound
    }

    /// Removes the entries whose bound times the largest bound is below `threshold`.
    pub fn prune(&mut self, threshold: f64) {
        let max_bound = self.max_bound;
        self.entries.retain(|entry| entry.schwarz_bound * max_bound >= threshold);
        self.index_segments();
    }
}

fn schwarz_bound_from_batch(batch: &ERIBatchType) -> f64 {
    let max_abs = match batch {
        ERIBatchType::FourD(t) => t.get_data().iter().fold(0.0_f64, |acc, v| acc.max(v.abs())),
        ERIBatchType::SixD(t) => t.get_data().iter().fold(0.0_f64, |acc, v| acc.max(v.abs())),
        ERIBatchType::EightD(t) => t.get_data().iter().fold(0.0_f64, |acc, v| acc.max(v.abs())),
    };
    max_abs.max(0.0).sqrt()
}

/// A pair's Hermite coefficients (kept, on the heap) and Schwarz bound, its scratch from `arena` (`pair_entry_buffers`
/// takes the same).
pub(crate) fn pair_entry<S: ShellKind>(pair: &ShellPair<S>, arena: &mut Arena<'_>) -> Result<(MMDCoefficients, f64), ArenaError> {
    let ket_coefficients = compute_ketside_hermite_coefficients::<S>(pair, arena)?;
    let schwarz_bound = arena.scope(|item| {
        let bra_coefficients = ket_coefficients.mmd_transpose_in(item)?;
        let eri_batch = compute_eri_batch::<S, S, FullRange>(
            pair, pair,
            &bra_coefficients, &ket_coefficients,
            item, 0.0,
        )?;
        Ok::<f64, ArenaError>(schwarz_bound_from_batch(&eri_batch))
    })?;
    Ok((ket_coefficients, schwarz_bound))
}

/// The memory `pair_entry` takes (the scratch query), in its order: the coefficients (kept, returned) with their
/// scratch, then in a scope the bra's transpose and the diagonal batch's buffers (`eri_batch_buffers`), nothing of the
/// batch computed.
pub(crate) fn pair_entry_buffers<S: ShellKind>(pair: &ShellPair<S>, arena: &mut Arena<'_>) -> Result<MMDCoefficients, ArenaError> {
    let ket_coefficients = compute_ketside_hermite_coefficients::<S>(pair, arena)?;
    arena.scope(|item| {
        ket_coefficients.mmd_transpose_in(item)?;
        crate::batch::eri_batch_buffers(pair, pair, item).map(|_| ())
    })?;
    Ok(ket_coefficients)
}

/// `TwoElectronIntegrals::derive_eri_data`.
pub(crate) fn derive_eri_data<S, W: arrays::scratch::ScratchSource>(
    shell_pairs: &[ShellPair<S>],
    ws: &mut W,
    schwarz_threshold: Option<f64>,
) -> Result<EriData, ArenaError>
where
    S: ShellKind,
{
    let mut shells: Vec<(usize, usize)> = shell_pairs
        .iter()
        .flat_map(|pair| {
            [(pair.get_idx_a(), pair.get_n_cont_a() * pair.get_degeneracy_a()), (pair.get_idx_b(), pair.get_n_cont_b() * pair.get_degeneracy_b())]
        })
        .collect();
    shells.sort_unstable();
    shells.dedup();
    let shell_of = |first_ao: usize| shells.binary_search_by_key(&first_ao, |&(first, _)| first).expect("a shell of the pairs");
    let mut entries = Vec::with_capacity(shell_pairs.len());
    let mut arena = ws.get_arena();
    for (original_index, pair) in shell_pairs.iter().enumerate() {
        let (ket_coefficients, schwarz_bound) = pair_entry::<S>(pair, &mut arena)?;
        if schwarz_threshold.is_some_and(|threshold| schwarz_bound < threshold) {
            continue;
        }
        let pair_shells = (shell_of(pair.get_idx_a()), shell_of(pair.get_idx_b()));
        entries.push(EriDataEntry { original_index, coeffs: ket_coefficients, schwarz_bound, shells: pair_shells, components: pair.components() });
    }
    Ok(EriData::new(entries, shells.into_boxed_slice()))
}

/// The weights of a quartet's modelled cost (`CostFactors::quartet`), relative to one multiplication of `E_ab X` or
/// of the digest: the R integrals (Boys function and recursion) per primitive quartet and Hermite function of
/// `l_ab + l_cd`, one multiplication of `R E_cd`, and the fixed work of a quartet.
const R_WEIGHT: f64 = 150.0;
const R_E_WEIGHT: f64 = 2.0;
const QUARTET_WEIGHT: f64 = 5.0e4;

/// A pair's factors in the modelled cost of a quartet: `p` primitive products, `l = l_a + l_b`, and `p t`, `p t n`,
/// `n c` for `t` Hermite functions of `l`, `n` components and `c` contracted pairs.
#[derive(Clone, Copy, Debug)]
pub(super) struct CostFactors {
    p: f64,
    l: usize,
    pt: f64,
    ptn: f64,
    nc: f64,
}

/// The pair's factors in the modelled cost of a quartet.
pub(super) fn cost_factors<S: ShellKind>(pair: &ShellPair<S>) -> CostFactors {
    let p = pair.get_products().len() as f64;
    let l = pair.get_lqn_a() + pair.get_lqn_b();
    let t = mmd::hermite_expansion::hermite_count(l) as f64;
    let n = pair.components() as f64;
    let (c_a, c_b) = pair.get_contracted_pair_dims();
    CostFactors { p, l, pt: p * t, ptn: p * t * n, nc: n * (c_a * c_b) as f64 }
}

impl CostFactors {
    /// The modelled cost of the quartet of this pair (the bra) and `ket`: the R integrals, `p_ab p_cd T(l_ab + l_cd)`
    /// for `T` Hermite functions; `(p t)_ab (p t n)_cd` multiplications for `R E_cd`; `(p t n)_ab (n c)_cd` for
    /// `E_ab X` and `(n c)_ab (n c)_cd` in the digest; and a fixed part; weighted (`R_WEIGHT` ...).
    fn quartet(&self, ket: &CostFactors) -> f64 {
        let t = mmd::hermite_expansion::hermite_count(self.l + ket.l) as f64;
        R_WEIGHT * self.p * ket.p * t + R_E_WEIGHT * self.pt * ket.ptn + (self.ptn + self.nc) * ket.nc + QUARTET_WEIGHT
    }
}

/// Prefix sums of the ket factors over pairs in order, so that a bra's modelled cost against a run of kets takes one
/// difference per factor: `sums(k, f)` sums pairs `..k` of factor `f`, `p t n`, `n c`, then `p` of the pairs with
/// `l = 0, 1, ...`.
pub(super) struct CostPrefix {
    sums: Matrix<f64>,
}

impl CostPrefix {
    /// The prefix sums of `factors`.
    pub(super) fn new(factors: &[CostFactors]) -> CostPrefix {
        let l_max = factors.iter().map(|f| f.l).max().unwrap_or(0);
        let mut sums = Matrix::<f64>::new(factors.len() + 1, 3 + l_max);
        let mut column_values = |column: usize, value: &dyn Fn(&CostFactors) -> f64| {
            let sum = sums.get_col_mut(column);
            for (k, factor) in factors.iter().enumerate() {
                sum[k + 1] = sum[k] + value(factor);
            }
        };
        column_values(0, &|f| f.ptn);
        column_values(1, &|f| f.nc);
        for l in 0..=l_max {
            column_values(2 + l, &|f| if f.l == l { f.p } else { 0.0 });
        }
        CostPrefix { sums }
    }

    /// The modelled cost of `bra` against the kets `kets` (`CostFactors::quartet` summed).
    pub(super) fn cost(&self, bra: &CostFactors, kets: Range<usize>) -> f64 {
        let sum = |column: usize| {
            let column = self.sums.get_col(column);
            column[kets.end] - column[kets.start]
        };
        let r: f64 = (0..self.sums.cols() - 2).map(|l| mmd::hermite_expansion::hermite_count(bra.l + l) as f64 * sum(2 + l)).sum();
        R_WEIGHT * bra.p * r + R_E_WEIGHT * bra.pt * sum(0) + (bra.ptn + bra.nc) * sum(1) + QUARTET_WEIGHT * kets.len() as f64
    }
}

/// A piece of the direct walk over `EriData`: entry `row` as the bra against the ket entries `kets`, a sub-range of
/// `row..` (the diagonal quartet is in the piece if `kets` starts at `row`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowPiece {
    /// The bra entry.
    pub row: usize,
    /// The ket entries.
    pub kets: Range<usize>,
}

/// The pieces of the walk over `EriData` for the ranks of a communicator, largest modelled cost first, handed out at
/// run time: each rank takes the next piece from a counter the ranks share (`Communicator::counter`) when it is free,
/// until none is left.
pub(crate) struct Handout {
    pieces: Vec<RowPiece>,
    counter: Counter,
}

impl Handout {
    /// The pieces of `row_pieces` for `comm`'s size in descending order of their modelled costs, the same on every
    /// rank, and the counter over `comm`. Collective. Panics if MPI fails.
    pub(crate) fn new<S: ShellKind>(shell_pairs: &[ShellPair<S>], eri_data: &EriData, pairwise_threshold: Option<f64>, comm: &Communicator) -> Handout {
        let (pieces, costs) = row_pieces(shell_pairs, eri_data, pairwise_threshold, comm.size());
        let mut order: Vec<usize> = (0..pieces.len()).collect();
        order.sort_by(|&a, &b| costs[b].total_cmp(&costs[a]));
        let pieces = order.into_iter().map(|k| pieces[k].clone()).collect();
        let counter = comm.counter().expect("the shared counter of the walk's pieces");
        Handout { pieces, counter }
    }

    /// The next piece not yet taken by any rank, `None` when all are. Panics if MPI fails.
    fn next(&self) -> Option<RowPiece> {
        let k = self.counter.next().expect("the next piece from the shared counter");
        self.pieces.get(k).cloned()
    }
}

/// The pieces a walk over `n` entries visits: those this rank takes from `handout`, or every whole row (`None`).
pub(crate) fn walk_pieces(handout: Option<&Handout>, n: usize) -> impl Iterator<Item = RowPiece> + '_ {
    let all = handout.is_none().then(|| (0..n).map(move |row| RowPiece { row, kets: row..n }));
    all.into_iter().flatten().chain(handout.into_iter().flat_map(|handout| std::iter::from_fn(|| handout.next())))
}

/// The largest piece is the modelled cost of the walk divided by this many times the rank count.
const PIECES_PER_RANK: usize = 8;

/// The walk over `eri_data` (row `i`: entry `i` against itself and every later entry) in pieces for `n_ranks` ranks,
/// in row order, with each piece's modelled cost (`CostFactors::quartet`; zero for a quartet the Schwarz bounds remove,
/// `Q_i Q_j < pairwise_threshold`: each segment is sorted, so its kept entries are its first ones). On more than one
/// rank, a row costing more than the walk's total over `PIECES_PER_RANK n_ranks` is cut into that many contiguous
/// runs of kets, at quartet boundaries, of about equal cost; every other row is one piece.
fn row_pieces<S: ShellKind>(shell_pairs: &[ShellPair<S>], eri_data: &EriData, pairwise_threshold: Option<f64>, n_ranks: usize) -> (Vec<RowPiece>, Vec<f64>) {
    let entries = eri_data.entries();
    let n = entries.len();
    let factors: Vec<CostFactors> = entries.iter().map(|entry| cost_factors(&shell_pairs[entry.original_index])).collect();
    let prefix = CostPrefix::new(&factors);
    // Per segment of row `i`'s later entries, its kept ones.
    let kept_segments = |i: usize| {
        eri_data.segments_in(i + 1..n).map(move |segment| {
            let kept = match pairwise_threshold {
                Some(t) => entries[segment.clone()].partition_point(|other| entries[i].schwarz_bound * other.schwarz_bound >= t),
                None => segment.len(),
            };
            segment.start..segment.start + kept
        })
    };
    let row_costs: Vec<f64> = (0..n)
        .map(|i| factors[i].quartet(&factors[i]) + kept_segments(i).map(|kept| prefix.cost(&factors[i], kept)).sum::<f64>())
        .collect();
    let largest = row_costs.iter().sum::<f64>() / (PIECES_PER_RANK * n_ranks) as f64;

    let (mut pieces, mut costs) = (Vec::new(), Vec::new());
    for (i, &row_cost) in row_costs.iter().enumerate() {
        let count = if n_ranks > 1 { (row_cost / largest).ceil().max(1.0) as usize } else { 1 };
        let (mut start, mut running, mut piece_cost, mut cuts) = (i, 0.0, 0.0, 0);
        if count > 1 {
            for j in std::iter::once(i).chain(kept_segments(i).flatten()) {
                let cost = factors[i].quartet(&factors[j]);
                running += cost;
                piece_cost += cost;
                // A cut after the quartet that brings the running cost to the next of `count` equal shares.
                if cuts + 1 < count && running * count as f64 >= (cuts + 1) as f64 * row_cost {
                    pieces.push(RowPiece { row: i, kets: start..j + 1 });
                    costs.push(piece_cost);
                    (start, piece_cost, cuts) = (j + 1, 0.0, cuts + 1);
                }
            }
        } else {
            piece_cost = row_cost;
        }
        // The last piece runs to the row's end; the removed quartets after a last cut join the piece before it.
        match pieces.last_mut() {
            Some(last) if cuts > 0 && piece_cost == 0.0 => last.kets.end = n,
            _ => {
                pieces.push(RowPiece { row: i, kets: start..n });
                costs.push(piece_cost);
            }
        }
    }
    (pieces, costs)
}

/// The rows this rank of `comm` owns, from each row's cost.
pub(super) fn owned_rows(comm: &Communicator, costs: impl Iterator<Item = usize>) -> Vec<bool> {
    let costs: Vec<usize> = costs.collect();
    let n = costs.len();
    let mut owned = vec![false; n];
    for idx in comm.partition(&costs) {
        owned[idx] = true;
    }
    owned
}

#[cfg(test)]
mod tests {
    use super::*;
    use basis::contraction::Contracted;
    use basis::shell_pairs::derive_shell_pairs;

    /// Water in ANO-RCC with general contraction (`test_fixtures/mpi_eri_check.inp`): its pairs and their data.
    fn water() -> (Vec<ShellPair<Contracted>>, EriData) {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_fixtures/mpi_eri_check.inp");
        let system = basis::setup_basis_system::<Contracted>(path).unwrap();
        let pairs = derive_shell_pairs(system.get_basis_shells(), None).unwrap();
        let mut scratch = Scratch::new(64, ByteUnit::MiB).unwrap();
        let data = derive_eri_data(&pairs, &mut scratch, None).unwrap();
        (pairs, data)
    }

    #[test]
    fn segments_in_cuts_a_range_at_the_segment_boundaries() {
        let (_, data) = water();
        let n = data.len();
        let ends = data.segment_ends.to_vec();
        assert!(ends.len() > 1, "the fixture should have several segments");
        for start in 0..=n {
            for end in start..=n {
                let ranges: Vec<Range<usize>> = data.segments_in(start..end).collect();
                assert_eq!(ranges.iter().flat_map(|r| r.clone()).collect::<Vec<_>>(), (start..end).collect::<Vec<_>>());
                for range in &ranges {
                    assert!(!range.is_empty());
                    let segment = ends.partition_point(|&e| e <= range.start);
                    assert!(range.end <= ends[segment], "{range:?} crosses the end of its segment");
                }
            }
        }
    }

    #[test]
    fn the_prefix_sums_give_the_quartets_costs() {
        let (pairs, data) = water();
        let factors: Vec<CostFactors> = data.entries().iter().map(|entry| cost_factors(&pairs[entry.original_index])).collect();
        let prefix = CostPrefix::new(&factors);
        for (i, bra) in factors.iter().enumerate() {
            for (start, end) in [(i, factors.len()), (i, i + 1), (0, factors.len() / 2)] {
                let expected: f64 = factors[start..end].iter().map(|ket| bra.quartet(ket)).sum();
                let cost = prefix.cost(bra, start..end);
                assert!((cost - expected).abs() <= 1e-12 * expected, "bra {i}, kets {start}..{end}: {cost} against {expected}");
            }
        }
    }

    #[test]
    fn the_pieces_cover_every_quartet_once_and_keep_the_cost() {
        let (pairs, data) = water();
        let n = data.len();
        for threshold in [None, Some(1e-3)] {
            let (whole, whole_costs) = row_pieces(&pairs, &data, threshold, 1);
            assert_eq!(whole, walk_pieces(None, n).collect::<Vec<_>>(), "one rank: every row whole");
            for n_ranks in [2, 3, 8, 64] {
                let (pieces, costs) = row_pieces(&pairs, &data, threshold, n_ranks);
                assert!(pieces.len() > n, "{n_ranks} ranks: some row should be cut");
                let (sum, whole_sum) = (costs.iter().sum::<f64>(), whole_costs.iter().sum::<f64>());
                assert!((sum - whole_sum).abs() <= 1e-12 * whole_sum, "{n_ranks} ranks: the cost changed");
                // Per row, contiguous pieces from the diagonal to the last entry.
                let mut next = vec![None; n];
                for piece in &pieces {
                    assert!(!piece.kets.is_empty());
                    let expected = next[piece.row].unwrap_or(piece.row);
                    assert_eq!(piece.kets.start, expected, "{n_ranks} ranks: a gap or an overlap in row {}", piece.row);
                    next[piece.row] = Some(piece.kets.end);
                }
                assert!(next.iter().all(|&end| end == Some(n)), "{n_ranks} ranks: a row not walked to its end");
                assert!(pieces.windows(2).all(|w| (w[0].row, w[0].kets.start) < (w[1].row, w[1].kets.start)), "not in row order");
            }
        }
    }
}
