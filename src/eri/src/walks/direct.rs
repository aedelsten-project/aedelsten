// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The direct Fock walk over `EriData`: per row its diagonal quartet, then its later entries segment by segment,
//! each quartet's Schwarz bound times the largest density element it touches against the threshold; serial or by
//! pieces of rows handed out over the ranks at run time (`eri_data::Handout`).

use aedelsten_mpi::Communicator;
use arrays::scratch::*;

use basis::contraction::ShellKind;
use basis::shell_pairs::*;
use mmd::hermite_coulomb_integrals::RangeVersion;
use mmd::ketside_hermite_coefficients::{MMDCoefficients, MMDCoefficientsView};

use crate::batch::compute_eri_batch;
use crate::distribute::EriBlock;
use super::eri_data::{walk_pieces, Handout, EriData, EriDataEntry, RowPiece};

/// The direct Fock walk (`TwoElectronIntegrals::compute_fock_direct_from_data`).
pub(crate) fn compute_fock_direct_from_data<T, S, R, W: arrays::scratch::ScratchSource>(
    output: &mut T,
    shell_pairs: &[ShellPair<S>],
    eri_data: &EriData,
    ws: &mut W,
    comm: &Communicator,
    omega: f64,
    pairwise_threshold: Option<f64>,
) -> Result<(), ArenaError>
where
    T: crate::DirectFock,
    S: ShellKind,
    R: RangeVersion,
{
    if output.wants_mpi_partition() && comm.size() > 1 {
        compute_fock_direct_from_data_parallel::<T, S, R, _>(output, shell_pairs, eri_data, ws, comm, omega, pairwise_threshold)
    } else {
        compute_fock_direct_from_data_serial::<T, S, R, _>(output, shell_pairs, eri_data, ws, omega, pairwise_threshold)
    }
}

fn compute_fock_direct_from_data_serial<T, S, R, W: arrays::scratch::ScratchSource>(
    output: &mut T,
    shell_pairs: &[ShellPair<S>],
    eri_data: &EriData,
    ws: &mut W,
    omega: f64,
    pairwise_threshold: Option<f64>,
) -> Result<(), ArenaError>
where
    T: crate::DirectFock,
    S: ShellKind,
    R: RangeVersion,
{
    output.prepare();
    compute_fock_direct_rows::<T, S, R, _>(output, shell_pairs, eri_data, None, ws, omega, pairwise_threshold)?;
    let _span = instrumentation::span("eri::direct::finish");
    output.finish();
    Ok(())
}

fn coefficient_tables_len(n_shells: usize) -> usize {
    2 * n_shells
}

fn density_tables_len(n_shells: usize) -> usize {
    2 * n_shells * n_shells
}

/// Per shell, the largest magnitude of the pair densities' coefficients over its functions (`ScreeningCoefficients`),
/// the bra's then the ket's, from the arena.
fn coefficient_tables<'a>(coefficients: &crate::ScreeningCoefficients<'_>, n: usize, shells: &[(usize, usize)], arena: &Arena<'a>) -> Result<&'a [f64], ArenaError> {
    let n_shells = shells.len();
    let tables = arena.alloc_zeroed::<f64>(coefficient_tables_len(n_shells))?;
    for (table, c) in tables.chunks_exact_mut(n_shells).zip([coefficients.bra, coefficients.ket]) {
        assert_eq!(c.len(), n * coefficients.m, "screening coefficients are not (n, m)");
        for column in c.chunks_exact(n) {
            for (largest, &(first, count)) in table.iter_mut().zip(shells) {
                *largest = column[first..first + count].iter().fold(*largest, |m, x| m.max(x.abs()));
            }
        }
    }
    Ok(tables)
}

/// Per pair of shells, the largest magnitude of the J densities (first `n_shells^2`) and of the K densities (the
/// next) over the pair's AO block or its transpose, from the arena; the densities in `both` count in both. Zero for a
/// part without densities.
fn density_tables<'a>(densities: &crate::ScreeningDensities<'_>, shells: &[(usize, usize)], arena: &Arena<'a>) -> Result<&'a [f64], ArenaError> {
    let n_shells = shells.len();
    let n = densities.n;
    let both = densities.both.unwrap_or(&[]);
    assert_eq!(both.len() % (n * n), 0, "screening densities are not (n, n) each");
    let tables = arena.alloc_zeroed::<f64>(density_tables_len(n_shells))?;
    let (coulomb, exchange) = tables.split_at_mut(n_shells * n_shells);
    for (table, matrices) in [(coulomb, &densities.coulomb[..]), (exchange, &densities.exchange[..])] {
        for density in matrices.iter().flatten().copied().chain(both.chunks_exact(n * n)) {
            assert_eq!(density.len(), n * n, "a screening density is not (n, n)");
            for (sb, &(first_b, count_b)) in shells.iter().enumerate() {
                for column in density.chunks_exact(n).skip(first_b).take(count_b) {
                    for (sa, &(first_a, count_a)) in shells.iter().enumerate() {
                        let largest = column[first_a..first_a + count_a].iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                        let entry = &mut table[sb * n_shells + sa];
                        *entry = entry.max(largest);
                    }
                }
            }
        }
        for a in 0..n_shells {
            for b in 0..a {
                let largest = table[a * n_shells + b].max(table[b * n_shells + a]);
                table[a * n_shells + b] = largest;
                table[b * n_shells + a] = largest;
            }
        }
    }
    Ok(tables)
}

/// The direct walk over the pieces of rows this rank takes from `handout` (every whole row if `None`): each piece's
/// diagonal quartet if it has it, then its off-diagonal ones, in one arena. A piece keeps its bra coefficients and the
/// accumulator's row scratch (`DirectFock::row_len`, between `begin_row` and `end_row`) in a scope; each quartet is
/// computed in a scope inside it. With a threshold, a quartet is skipped when its Schwarz bound `Q_ab Q_cd` times the
/// largest density element it touches (`D_ab`, `D_cd` of the J densities, `D_ac`, `D_ad`, `D_bc`, `D_bd` of the K
/// densities; the bound alone if the accumulator has no densities) is below it, and, for an accumulator with pair
/// densities, their bound (`ScreeningCoefficients`) is below its own threshold too; within a segment the entries are in
/// descending order of their bounds, so a row leaves a segment at the first pair whose product with the largest density
/// element is below it. An accumulator with `DirectFock::wanted_functions` gets only the quartets with a shell whose
/// first function is in that range. The quartets computed and those skipped by the density or as unwanted are counted
/// (`eri::direct::quartets`, `eri::direct::density_skipped`, `eri::direct::unwanted`).
fn compute_fock_direct_rows<T, S, R, W: arrays::scratch::ScratchSource>(
    output: &mut T,
    shell_pairs: &[ShellPair<S>],
    eri_data: &EriData,
    handout: Option<&Handout>,
    ws: &mut W,
    omega: f64,
    pairwise_threshold: Option<f64>,
) -> Result<(), ArenaError>
where
    T: crate::DirectFock,
    S: ShellKind,
    R: RangeVersion,
{
    let _span = instrumentation::span("eri::direct::rows");
    let mut arena = ws.get_arena();
    let densities = pairwise_threshold.and(output.screening_densities());
    let tables = densities.as_ref().map(|densities| density_tables(densities, &eri_data.shells, &arena)).transpose()?;
    // The pair densities' coefficient maxima and the factor that puts their bound against the walk's threshold.
    let pair_densities = match (pairwise_threshold, densities.as_ref().and_then(|densities| densities.pair_densities.as_ref())) {
        (Some(t), Some(coefficients)) => Some((coefficient_tables(coefficients, densities.as_ref().map_or(0, |d| d.n), &eri_data.shells, &arena)?, t / coefficients.threshold)),
        _ => None,
    };
    let n_shells = eri_data.shells.len();
    // The largest density element a quartet of `ab` and `cd` touches, or the pair densities' bound scaled to the
    // walk's threshold if larger; 1 without densities.
    let density_factor = |ab: &EriDataEntry, cd: &EriDataEntry| -> f64 {
        let Some(tables) = tables else { return 1.0 };
        let ((a, b), (c, d)) = (ab.shells, cd.shells);
        let coulomb = |p: usize, q: usize| tables[p * n_shells + q];
        let exchange = |p: usize, q: usize| tables[n_shells * n_shells + p * n_shells + q];
        let density = coulomb(a, b).max(coulomb(c, d)).max(exchange(a, c)).max(exchange(a, d)).max(exchange(b, c)).max(exchange(b, d));
        let Some((largest, scale)) = pair_densities else { return density };
        let (bra, ket) = largest.split_at(n_shells);
        let (p_ab, p_cd) = ((bra[a] * ket[b]).max(bra[b] * ket[a]), (bra[c] * ket[d]).max(bra[d] * ket[c]));
        density.max(scale * (ket[a].max(ket[b]) * p_cd).max(ket[c].max(ket[d]) * p_ab))
    };
    let largest_density = tables.map_or(1.0, |tables| {
        let density = tables.iter().fold(0.0_f64, |m, &x| m.max(x));
        let Some((largest, scale)) = pair_densities else { return density };
        let (bra, ket) = largest.split_at(n_shells);
        let (bra_max, ket_max) = (bra.iter().fold(0.0_f64, |m, &x| m.max(x)), ket.iter().fold(0.0_f64, |m, &x| m.max(x)));
        density.max(scale * ket_max * bra_max * ket_max)
    });
    let (mut computed, mut skipped, mut unwanted) = (0u64, 0u64, 0u64);
    let merged = output.merged_slots();
    // Whether a shell's functions are among those the accumulator wants (every shell's without a range).
    let wanted = output.wanted_functions();
    let wanted_shell = |s: usize| wanted.as_ref().is_none_or(|range| range.contains(&eri_data.shells[s].0));

    for RowPiece { row: i, kets } in walk_pieces(handout, eri_data.len()) {
        let entry_ab = &eri_data.entries()[i];
        let row_wanted = wanted_shell(entry_ab.shells.0) || wanted_shell(entry_ab.shells.1);
        let pair_ab = &shell_pairs[entry_ab.original_index];
        let rows = pair_ab.get_n_cont_a() * pair_ab.get_degeneracy_a() + pair_ab.get_n_cont_b() * pair_ab.get_degeneracy_b();
        arena.scope(|row| {
            let bra_coefficients = entry_ab.coeffs.mmd_transpose_in(row)?;
            let strips = row.alloc::<f64>(output.row_len(rows))?;
            let mut begun = false;
            // The diagonal quartet if the piece has it, then its later segments.
            let diagonal = (kets.start == i).then_some((std::slice::from_ref(entry_ab), true));
            let later = eri_data.segments_in(kets.start.max(i + 1)..kets.end).map(|segment| (&eri_data.entries()[segment], false));
            for (entries, is_diagonal) in diagonal.into_iter().chain(later) {
            for entry_cd in entries {
                let bound = entry_ab.schwarz_bound * entry_cd.schwarz_bound;
                if let Some(t) = pairwise_threshold {
                    if !is_diagonal && bound * largest_density < t {
                        break;
                    }
                }
                if !row_wanted && !wanted_shell(entry_cd.shells.0) && !wanted_shell(entry_cd.shells.1) {
                    unwanted += 1;
                    continue;
                }
                if let Some(t) = pairwise_threshold {
                    if tables.is_some() && bound * density_factor(entry_ab, entry_cd) < t {
                        skipped += 1;
                        continue;
                    }
                }
                computed += 1;
                if !begun {
                    output.begin_row(pair_ab, strips);
                    begun = true;
                }
                let pair_cd = &shell_pairs[entry_cd.original_index];
                row.scope(|quartet| digest_quartet::<T, S, R>(output, pair_ab, pair_cd, &bra_coefficients, &entry_cd.coeffs, bound, merged, strips, quartet, omega))?;
            }
            }
            if begun {
                output.end_row(pair_ab, strips);
            }
            Ok::<(), ArenaError>(())
        })?;
    }
    instrumentation::count("eri::direct::quartets", computed);
    instrumentation::count("eri::direct::density_skipped", skipped);
    if wanted.is_some() {
        instrumentation::count("eri::direct::unwanted", unwanted);
    }
    Ok(())
}

/// One quartet of a row: its integrals in `quartet`, digested by `output` with the row's `strips` (merged first if it
/// wants, `with_merged_slots`) and its tiles (`DirectFock::digestion_tiles`). Its memory is `quartet_buffers`'.
#[allow(clippy::too_many_arguments)]
fn digest_quartet<T, S, R>(
    output: &mut T,
    pair_ab: &ShellPair<S>,
    pair_cd: &ShellPair<S>,
    bra_coefficients: &MMDCoefficientsView<'_>,
    ket_coefficients: &MMDCoefficients,
    bound: f64,
    merged: bool,
    strips: &mut [f64],
    quartet: &mut Arena<'_>,
    omega: f64,
) -> Result<(), ArenaError>
where
    T: crate::DirectFock,
    S: ShellKind,
    R: RangeVersion,
{
    let eri_batch = compute_eri_batch::<S, S, R>(pair_ab, pair_cd, bra_coefficients, ket_coefficients, quartet, omega)?;
    let _span = instrumentation::fine_span("eri::direct::distribute");
    let block = EriBlock::new(&eri_batch, Some(bound));
    if merged {
        crate::distribute::with_merged_slots(&block, pair_ab, pair_cd, quartet, |merged, arena| {
            let mut tiles = output.digestion_tiles(merged, pair_ab, pair_cd, arena)?;
            output.distribute_block_in_row(merged, pair_ab, pair_cd, strips, &mut tiles);
            Ok(())
        })
    } else {
        let mut tiles = output.digestion_tiles(&block, pair_ab, pair_cd, quartet)?;
        output.distribute_block_in_row(&block, pair_ab, pair_cd, strips, &mut tiles);
        Ok(())
    }
}

/// The memory `digest_quartet` takes, in its order, nothing computed (the scratch query): the batch's
/// (`eri_batch_buffers`), the merged block's in a scope (`MergedSlots`) when `merged`, and the digestion's tiles.
fn quartet_buffers<T, S>(output: &T, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, merged: bool, quartet: &mut Arena<'_>) -> Result<(), ArenaError>
where
    T: crate::DirectFock,
    S: ShellKind,
{
    let eri_batch = crate::batch::eri_batch_buffers(pair_ab, pair_cd, quartet)?;
    let block = EriBlock::new(&eri_batch, None);
    if !merged {
        return output.digestion_tiles(&block, pair_ab, pair_cd, quartet).map(|_| ());
    }
    quartet.scope(|scope| match crate::distribute::MergedSlots::new_in(&block, pair_ab, pair_cd, scope)? {
        None => output.digestion_tiles(&block, pair_ab, pair_cd, scope).map(|_| ()),
        Some(slots) => {
            let merged = slots.unfilled(None);
            scope.scope(|arena| output.digestion_tiles(&merged, pair_ab, pair_cd, arena).map(|_| ()))
        }
    })
}

/// The scratch the direct walk takes for a basis of `n` functions in `n_shells` shells, measured on `ws`: its
/// density tables (with `screened`), then per row the bra's transposed coefficients and `output.row_len_at(rows, n)`,
/// and in it each quartet's memory (`quartet_buffers`), over the probe pairs `probes` (`basis::probes`, with their
/// coefficients `coefficients`) as bras and kets in the walk's order: a ket of no more components than its bra, the
/// diagonal quartet the pair with itself. `output` is an accumulator of the operation's kind over the probe pairs'
/// functions. Nothing is computed but the transposes.
pub(crate) fn direct_walk_need<T, S>(
    output: &T,
    probes: &[ShellPair<S>],
    coefficients: &[MMDCoefficients],
    n: usize,
    n_shells: usize,
    screened: bool,
    ws: &mut Scratch,
) -> Result<Option<crate::need::Need>, ScratchError>
where
    T: crate::DirectFock,
    S: ShellKind,
{
    let merged = output.merged_slots();
    let mut largest = None;
    // A row at a time behind the tables: one that does not fit is run again alone.
    for (i, pair_ab) in probes.iter().enumerate() {
        let rows = pair_ab.get_n_cont_a() * pair_ab.get_degeneracy_a() + pair_ab.get_n_cont_b() * pair_ab.get_degeneracy_b();
        let (mut tables, mut row_end, mut quartet) = (0, 0, (i, 0));
        let bytes = ws.measure(|arena| {
            if screened {
                arena.alloc::<f64>(density_tables_len(n_shells))?;
                arena.alloc::<f64>(coefficient_tables_len(n_shells))?;
            }
            tables = arena.used();
            arena.scope(|row| {
                coefficients[i].mmd_transpose_in(row)?;
                row.alloc::<f64>(output.row_len_at(rows, n))?;
                row_end = row.used();
                quartet = (i, 0);
                for (j, pair_cd) in probes.iter().enumerate().filter(|(_, cd)| cd.components() <= pair_ab.components()) {
                    let (taken, extent) = row.scope_extent(|scope| quartet_buffers(output, pair_ab, pair_cd, merged, scope));
                    taken?;
                    if extent > quartet.1 {
                        quartet = (j, extent);
                    }
                }
                Ok(())
            })
        })?;
        crate::need::keep_largest(&mut largest, crate::need::Need::quartet(bytes, tables, row_end, i, quartet));
    }
    Ok(largest)
}

fn compute_fock_direct_from_data_parallel<T, S, R, W: arrays::scratch::ScratchSource>(
    output: &mut T,
    shell_pairs: &[ShellPair<S>],
    eri_data: &EriData,
    ws: &mut W,
    comm: &Communicator,
    omega: f64,
    pairwise_threshold: Option<f64>,
) -> Result<(), ArenaError>
where
    T: crate::DirectFock,
    S: ShellKind,
    R: RangeVersion,
{
    let handout = Handout::new(shell_pairs, eri_data, pairwise_threshold, comm);
    output.prepare();
    compute_fock_direct_rows::<T, S, R, _>(output, shell_pairs, eri_data, Some(&handout), ws, omega, pairwise_threshold)?;
    {
        let _span = instrumentation::span("eri::direct::finish");
        output.finish();
    }
    let _span = instrumentation::span("eri::direct::allreduce");
    output.allreduce_sum(comm);
    Ok(())
}
