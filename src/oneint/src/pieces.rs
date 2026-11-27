// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The pieces the one-electron loops share out over the ranks: a run of shell pairs against a run of the operator's
// sites (its charges, multipoles, field points), every operator's integrals being sums over its sites. The sites are
// cut only as far as the costliest pair needs, the pairs within each run of sites. The pieces are handed out at run
// time, largest modelled cost first, from a counter the ranks share.

use std::ops::Range;

use aedelsten_mpi::{Communicator, Counter};

use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;

use crate::additional_arguments::{EmptyAdditionalArguments, MomentOrigin};

/// A kernel's additional arguments as the loops cut them: the number of the operator's sites, and the arguments of
/// a run of them (the same arguments for an operator without sites).
pub trait SiteRuns: Sized {
    /// The number of sites.
    fn n_sites(&self) -> usize;
    /// The arguments of the sites `sites` alone.
    fn run(&self, sites: Range<usize>) -> Self;
}

impl<T> SiteRuns for &[T] {
    fn n_sites(&self) -> usize {
        self.len()
    }

    fn run(&self, sites: Range<usize>) -> Self {
        &self[sites]
    }
}

impl SiteRuns for EmptyAdditionalArguments {
    fn n_sites(&self) -> usize {
        0
    }

    fn run(&self, _: Range<usize>) -> Self {
        EmptyAdditionalArguments
    }
}

impl SiteRuns for &MomentOrigin {
    fn n_sites(&self) -> usize {
        0
    }

    fn run(&self, _: Range<usize>) -> Self {
        self
    }
}

impl SiteRuns for [f64; 3] {
    fn n_sites(&self) -> usize {
        0
    }

    fn run(&self, _: Range<usize>) -> Self {
        *self
    }
}

/// The shell pairs `pairs` (indices into the loop's list) against the sites `sites` (empty for an operator without
/// sites).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    /// The shell pairs, indices into the loop's list.
    pub pairs: Range<usize>,
    /// The operator's sites.
    pub sites: Range<usize>,
}

/// The largest piece is the modelled cost of the loop divided by this many times the rank count.
const PIECES_PER_RANK: usize = 8;

/// A pair's modelled cost per site (or for an operator without sites), for sizing and ordering the pieces only:
/// `150 p t + p t n + n c` for `p` primitive products, `t` Hermite functions of `l_a + l_b`, `n` components and `c`
/// contracted pairs: the source integrals per primitive product, their transformation, and the contraction.
pub fn pair_cost<S: ShellKind>(pair: &ShellPair<S>) -> f64 {
    let p = pair.get_products().len() as f64;
    let t = mmd::hermite_expansion::hermite_count(pair.get_lqn_a() + pair.get_lqn_b()) as f64;
    let n = pair.components() as f64;
    let (c_a, c_b) = pair.get_contracted_pair_dims();
    150.0 * p * t + p * t * n + n * (c_a * c_b) as f64
}

/// Which of the two a loop cuts first: the sites (when a piece's fixed work grows with the run of pairs, as the field
/// passes' contraction of whole planes) or the pairs (when it grows with the runs of sites, as a pair's setup).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    SitesFirst,
    PairsFirst,
}

/// The pieces of the pairs with costs `costs` (per site) against `n_sites` sites for `n_ranks` ranks, in order,
/// with their modelled costs; no run longer than `max_sites` sites. A piece is at most the total over
/// `PIECES_PER_RANK n_ranks` where it can be. `PairsFirst`: the sites cut into runs only as short as the costliest
/// pair needs to fit, each pair of a run a piece of its own on several ranks (with sites; without, the run's pairs
/// into consecutive runs within that size).
/// `SitesFirst`: runs of sites against every pair when one site against every pair fits, else runs of pairs against
/// each site. On one rank, runs of `max_sites` sites against every pair.
pub fn pieces(costs: &[f64], n_sites: usize, max_sites: usize, n_ranks: usize, cut: Cut) -> Vec<(Piece, f64)> {
    let n_pairs = costs.len();
    let per_site: f64 = costs.iter().sum();
    let costliest = costs.iter().fold(0.0_f64, |m, &c| m.max(c));
    let largest = if n_ranks == 1 { f64::INFINITY } else { per_site * n_sites.max(1) as f64 / (PIECES_PER_RANK * n_ranks) as f64 };
    let fitting = |cost: f64| if cost > 0.0 { (largest / cost).min(n_sites as f64) as usize } else { n_sites };
    let run = if n_sites == 0 {
        0
    } else {
        let fits = match cut {
            Cut::PairsFirst => fitting(costliest),
            Cut::SitesFirst => fitting(per_site),
        };
        fits.clamp(1, max_sites.max(1).min(n_sites))
    };
    let site_runs: Vec<Range<usize>> = if n_sites == 0 { vec![0..0] } else { (0..n_sites.div_ceil(run)).map(|k| k * run..((k + 1) * run).min(n_sites)).collect() };
    // With sites and pairs first on several ranks, every pair a piece of its own (as the ERI walk's rows): the many
    // small ones handed out last even out what the costs' model misses.
    let alone = cut == Cut::PairsFirst && n_sites > 0 && n_ranks > 1;
    let mut pieces = Vec::new();
    for sites in site_runs {
        let weight = sites.len().max(1) as f64;
        let (mut start, mut cost) = (0, 0.0);
        for (i, &c) in costs.iter().enumerate() {
            if i > start && (alone || (cost + c) * weight > largest) {
                pieces.push((Piece { pairs: start..i, sites: sites.clone() }, cost * weight));
                (start, cost) = (i, 0.0);
            }
            cost += c;
        }
        if start < n_pairs {
            pieces.push((Piece { pairs: start..n_pairs, sites: sites.clone() }, cost * weight));
        }
    }
    pieces
}

/// The pieces of a loop for the ranks of `comm` (`pieces`), largest modelled cost first: the order a `Handout` gives
/// them out in. The same on every rank.
pub fn sorted_pieces(costs: &[f64], n_sites: usize, max_sites: usize, cut: Cut, comm: &Communicator) -> Vec<Piece> {
    let mut pieces = pieces(costs, n_sites, max_sites, comm.size(), cut);
    pieces.sort_by(|a, b| b.1.total_cmp(&a.1));
    pieces.into_iter().map(|(piece, _)| piece).collect()
}

/// Pieces (`sorted_pieces`) handed out at run time: each rank takes the next from a counter the ranks of a
/// communicator share (`Communicator::counter`) when it is free, until none is left. On one rank, every piece in turn.
pub struct Handout<'p> {
    pieces: &'p [Piece],
    counter: Counter,
}

impl<'p> Handout<'p> {
    /// Hands out `pieces`, the same list on every rank of `comm`. Collective over `comm`. Panics if MPI fails.
    pub fn new(pieces: &'p [Piece], comm: &Communicator) -> Handout<'p> {
        let counter = comm.counter().expect("the shared counter of the one-electron pieces");
        Handout { pieces, counter }
    }

    /// The pieces this rank takes, in the order it takes them. Panics if MPI fails.
    pub fn iter(&self) -> impl Iterator<Item = Piece> + '_ {
        std::iter::from_fn(|| {
            let k = self.counter.next().expect("the next piece from the shared counter");
            self.pieces.get(k).cloned()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every (pair, site) once, whatever the shape.
    fn covered(pieces: &[(Piece, f64)], n_pairs: usize, n_sites: usize) -> bool {
        let mut count = vec![0usize; n_pairs * n_sites.max(1)];
        for (piece, _) in pieces {
            for pair in piece.pairs.clone() {
                if n_sites == 0 {
                    assert!(piece.sites.is_empty());
                    count[pair] += 1;
                }
                for site in piece.sites.clone() {
                    count[pair + n_pairs * site] += 1;
                }
            }
        }
        count.iter().all(|&c| c == 1)
    }

    #[test]
    fn the_pieces_cover_every_pair_and_site_once() {
        let costs: Vec<f64> = (0..37).map(|i| 1.0 + (i * 7 % 11) as f64).collect();
        for n_sites in [0, 1, 3, 50, 2000] {
            for n_ranks in [1, 2, 8, 64] {
                for max_sites in [1, 7, usize::MAX] {
                    for cut in [Cut::SitesFirst, Cut::PairsFirst] {
                    let pieces = pieces(&costs, n_sites, max_sites, n_ranks, cut);
                    assert!(covered(&pieces, costs.len(), n_sites), "{n_sites} sites, {n_ranks} ranks, runs of {max_sites}");
                    assert!(pieces.iter().all(|(piece, _)| piece.sites.len() <= max_sites));
                    let total: f64 = pieces.iter().map(|(_, cost)| cost).sum();
                    let expected = costs.iter().sum::<f64>() * n_sites.max(1) as f64;
                    assert!((total - expected).abs() <= 1e-12 * expected);
                    }
                }
            }
        }
    }

    #[test]
    fn sites_are_cut_only_as_far_as_the_costliest_pair_needs() {
        // Even pairs: every site in one run, every pair a piece.
        let even = pieces(&vec![1.0; 100], 1000, usize::MAX, 4, Cut::PairsFirst);
        assert!(even.len() == 100 && even.iter().all(|(piece, _)| piece.sites == (0..1000) && piece.pairs.len() == 1));
        // One pair a third of the work: the sites cut so that it fits a piece, the pairs cut within each run.
        let mut skewed = vec![1.0; 100];
        skewed[0] = 50.0;
        let cut = pieces(&skewed, 1000, usize::MAX, 4, Cut::PairsFirst);
        let largest = skewed.iter().sum::<f64>() * 1000.0 / 32.0;
        assert!(cut.iter().all(|(piece, cost)| piece.sites.len() < 1000 && (*cost <= largest || piece.pairs.len() == 1)));
        // The runs of sites never longer than asked.
        assert!(pieces(&vec![1.0; 100], 1000, 7, 4, Cut::PairsFirst).iter().all(|(piece, _)| piece.sites.len() <= 7));
        // Sites first: runs of sites against every pair, and with few sites runs of pairs against each.
        let sites = pieces(&vec![1.0; 100], 1000, usize::MAX, 4, Cut::SitesFirst);
        assert!(sites.len() >= 32 && sites.iter().all(|(piece, _)| piece.pairs == (0..100)));
        let few = pieces(&vec![1.0; 100], 2, usize::MAX, 4, Cut::SitesFirst);
        assert!(few.iter().all(|(piece, _)| piece.sites.len() == 1 && piece.pairs.len() < 100));
        // No sites: runs of pairs.
        let none = pieces(&vec![1.0; 100], 0, usize::MAX, 4, Cut::PairsFirst);
        assert!(none.len() >= 32 && none.iter().all(|(piece, _)| piece.sites.is_empty()));
    }
}
