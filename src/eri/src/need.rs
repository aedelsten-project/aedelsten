// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The scratch the walks take, measured on probe pairs (`basis::probes`): each item's memory taken by its own
//! constructors on a scratch buffer that grows until it fits (`Scratch::measure`), with the tables a walk holds for the
//! whole basis sized by the functions it allocates them from. Each measurement says what sets its peak (`Need`).

use arrays::scratch::{Scratch, ScratchError};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;
use mmd::ketside_hermite_coefficients::MMDCoefficients;

/// The item a measured peak is set by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeedItem {
    /// A pair's ERI data entry (`pair_entry_buffers`): its Hermite coefficients and bound.
    EriDataEntry,
    /// A pair's data for the gradient walk.
    GradientPairData,
    /// A pair's derivative Schwarz bound.
    DerivativeBound,
    /// A quartet of a walk's row: its batch and digestion, or its gradient buffers.
    Quartet,
}

/// A measured peak and what sets it: the walk's tables for the whole basis, the row's own memory (none for a pair's
/// item) and the item's, which add up to `bytes`; the item's probe pairs by index, the pair (or the bra) and the ket
/// (`None` for a pair's item).
#[derive(Clone, Copy, Debug)]
pub struct Need {
    /// The peak, in bytes.
    pub bytes: usize,
    /// The walk's tables for the whole basis.
    pub tables: usize,
    /// The row's own memory.
    pub row: usize,
    /// The item's.
    pub item_bytes: usize,
    /// What sets the peak.
    pub item: NeedItem,
    /// The probe pair (or the bra) by index.
    pub pair: usize,
    /// The ket by index; `None` for a pair's item.
    pub ket: Option<usize>,
}

impl Need {
    /// A pair's item alone, of `bytes`.
    pub(crate) fn pair(item: NeedItem, pair: usize, bytes: usize) -> Self {
        Need { bytes, tables: 0, row: 0, item_bytes: bytes, item, pair, ket: None }
    }

    /// A row's quartet `ket` behind the walk's tables, from the parts measured in the row's run: the tables, the bytes
    /// taken when the row's own memory is (`row_end`, the tables included), and the ket of the largest quartet with
    /// its extent. `bytes` is the run's peak.
    pub(crate) fn quartet(bytes: usize, tables: usize, row_end: usize, pair: usize, (ket, item_bytes): (usize, usize)) -> Self {
        Need { bytes, tables, row: row_end - tables, item_bytes, item: NeedItem::Quartet, pair, ket: Some(ket) }
    }

    /// The larger of two, the first on a tie.
    pub fn max(self, other: Option<Need>) -> Need {
        match other {
            Some(other) if other.bytes > self.bytes => other,
            _ => self,
        }
    }
}

/// The larger of `largest` and `need`.
pub(crate) fn keep_largest(largest: &mut Option<Need>, need: Need) {
    *largest = Some(need.max(*largest));
}

/// The scratch `derive_eri_data` takes for the largest of `probes`, and their Hermite coefficients, which the
/// measurements of the direct walks take.
pub fn eri_data<S: ShellKind>(probes: &[ShellPair<S>], ws: &mut Scratch) -> Result<(Option<Need>, Vec<MMDCoefficients>), ScratchError> {
    let mut largest = None;
    let mut coefficients = Vec::with_capacity(probes.len());
    for (i, pair) in probes.iter().enumerate() {
        let mut kept = None;
        let bytes = ws.measure(|arena| {
            kept = Some(crate::walks::pair_entry_buffers(pair, arena)?);
            Ok(())
        })?;
        keep_largest(&mut largest, Need::pair(NeedItem::EriDataEntry, i, bytes));
        coefficients.push(kept.expect("measured"));
    }
    Ok((largest, coefficients))
}

/// The scratch the direct walk takes into `output`, an accumulator of the operation's kind over the probe pairs'
/// functions, for a basis of `n` functions in `n_shells` shells: the density tables if `screened`, and the largest row
/// of `probes` (their coefficients `coefficients`, from `eri_data`) with its largest quartet. The same on every range.
pub fn direct_fock<T, S>(output: &T, probes: &[ShellPair<S>], coefficients: &[MMDCoefficients], n: usize, n_shells: usize, screened: bool, ws: &mut Scratch) -> Result<Option<Need>, ScratchError>
where
    T: crate::DirectFock,
    S: ShellKind,
{
    crate::walks::direct_walk_need::<T, S>(output, probes, coefficients, n, n_shells, screened, ws)
}

/// The scratch the ERI gradient walk takes into `sink`, a sink of the operation's kind over the probe pairs'
/// functions and their two atoms (0 and 1), for a basis of `n_shells` shells, its pairs screened if `screened`.
pub fn eri_gradient<G, S>(sink: &mut G, probes: &[ShellPair<S>], n_shells: usize, screened: bool, ws: &mut Scratch) -> Result<Option<Need>, ScratchError>
where
    G: crate::GradientSink,
    S: ShellKind,
{
    crate::gradient::gradient_walk_need::<G, S>(sink, probes, n_shells, screened, ws)
}
