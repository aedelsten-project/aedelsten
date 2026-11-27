// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The stored ERI tensor's walk: every pair against itself and every later pair (the pairs sorted with more
//! components first, so the bra has more), serial or by rows over the ranks (`eri_data::owned_rows` with the cost
//! model of `eri_data::CostFactors`).

use arrays::scratch::*;

use basis::contraction::ShellKind;
use basis::shell_pairs::*;
use mmd::ketside_hermite_coefficients::{compute_ketside_hermite_coefficients, MMDCoefficients};
use mmd::hermite_coulomb_integrals::RangeVersion;

use crate::batch::compute_eri_batch;
use crate::distribute::EriBlock;
use aedelsten_mpi::Communicator;

use super::eri_data::{cost_factors, owned_rows, CostFactors, CostPrefix};

/// The stored tensor's walk (`TwoElectronIntegrals::compute_eri_stored`).
pub(crate) fn compute_eri_stored<T, S, R, W: arrays::scratch::ScratchSource>(
    output: &mut T,
    shell_pairs: &[ShellPair<S>],
    scratch: &mut W,
    comm: &Communicator,
    omega: f64
) -> Result<(), ArenaError>
where
    T: crate::TensorChoice,
    S: ShellKind,
    R: RangeVersion,
{
    if output.wants_mpi_partition() && comm.size() > 1 {
        compute_eri_stored_parallel::<T, S, R, _>(output, shell_pairs, scratch, comm, omega)
    } else {
        compute_eri_stored_serial::<T, S, R, _>(output, shell_pairs, scratch, omega)
    }
}

fn compute_eri_stored_serial<T, S, R, W: arrays::scratch::ScratchSource>(
    output: &mut T,
    shell_pairs: &[ShellPair<S>],
    scratch: &mut W,
    omega: f64
) -> Result<(), ArenaError>
where
    T: crate::TensorChoice,
    S: ShellKind,
    R: RangeVersion,
{
    compute_eri_stored_rows::<T, S, R, _>(output, shell_pairs, None, scratch, omega)
}

/// Every rank computes its own rows of the pair-quartet walk (`owned_rows`), then the ranks sum. Every rank must
/// call it. `distribute` overwrites, so the stored tensors zero themselves in `prepare`: a position this rank does
/// not own stays 0 for the sum.
fn compute_eri_stored_parallel<T, S, R, W: arrays::scratch::ScratchSource>(
    output: &mut T,
    shell_pairs: &[ShellPair<S>],
    scratch: &mut W,
    comm: &Communicator,
    omega: f64
) -> Result<(), ArenaError>
where
    T: crate::TensorChoice,
    S: ShellKind,
    R: RangeVersion,
{
    // Row `i` pairs `i` with itself and every later pair: its modelled cost (`CostFactors::quartet`) from prefix
    // sums of the ket factors.
    let factors: Vec<CostFactors> = shell_pairs.iter().map(cost_factors).collect();
    let prefix = CostPrefix::new(&factors);
    let n = factors.len();
    let owned = owned_rows(comm, factors.iter().enumerate().map(|(i, bra)| prefix.cost(bra, i..n).round() as usize));
    compute_eri_stored_rows::<T, S, R, _>(output, shell_pairs, Some(&owned), scratch, omega)?;
    let _span = instrumentation::span("eri::in_memory::allreduce");
    output.allreduce_sum(comm);
    Ok(())
}

/// The unscreened walk: row `i` is pair `i` against itself and against every later pair. `owned`: the rows to
/// compute (`None`: all of them); every row's Hermite coefficients are derived regardless, since later rows need
/// them as kets.
fn compute_eri_stored_rows<T, S, R, W: arrays::scratch::ScratchSource>(
    output: &mut T,
    shell_pairs: &[ShellPair<S>],
    owned: Option<&[bool]>,
    scratch: &mut W,
    omega: f64
) -> Result<(), ArenaError>
where
    T: crate::TensorChoice,
    S: ShellKind,
    R: RangeVersion,
{
    let _span = instrumentation::span("eri::in_memory::rows");
    output.prepare();
    // The diagonal quartets, keeping every pair's Hermite coefficients for the kets.
    let eri_data = compute_diagonal_eri_stored::<T, S, R, _>(
        output,
        shell_pairs,
        owned,
        scratch,
        omega
    )?;

    // The off-diagonal quartets.
    compute_off_diagonal_eri_stored::<T, S, R, _>(
        output,
        &eri_data,
        owned,
        scratch,
        omega
    )?;
    output.finish();
    Ok(())
}

struct ERIData<'a, S: ShellKind> {
    shell_pairs: &'a [ShellPair<S>],
    coeffs: Box<[MMDCoefficients]>,
}

// Every pair's diagonal quartet (the owned rows'), and every pair's Hermite coefficients.
fn compute_diagonal_eri_stored<'a, T, S, R, W: arrays::scratch::ScratchSource>(
    eri_tensor: &mut T,
    shell_pairs: &'a [ShellPair<S>],
    owned: Option<&[bool]>,
    ws: &mut W,
    omega: f64
) -> Result<ERIData<'a, S>, ArenaError>
where
    T: crate::TensorChoice,
    S: ShellKind,
    R: RangeVersion,
{
    let mut mmd_coeffs = Vec::with_capacity(shell_pairs.len());
    let mut arena = ws.get_arena();
    for (pair_idx, pair) in shell_pairs.iter().enumerate() {
        let ket_coefficients = compute_ketside_hermite_coefficients::<S>(&pair, &mut arena)?;
        if owned.is_some_and(|owned| !owned[pair_idx]) {
            mmd_coeffs.push(ket_coefficients);
            continue;
        }
        arena.scope(|item| {
            // Only the ket coefficients are kept; the bra's are their transpose.
            let bra_coefficients = ket_coefficients.mmd_transpose_in(item)?;
            let eri_batch = compute_eri_batch::<S, S, R>(
                &pair,
                &pair,
                &bra_coefficients,
                &ket_coefficients,
                item,
                omega
            )?;
            let _span = instrumentation::fine_span("eri::in_memory::distribute");
            eri_tensor.distribute_block(&EriBlock::new(&eri_batch, None), pair, pair, item)
        })?;
        mmd_coeffs.push(ket_coefficients);
    }
    let mmd_coeffs = mmd_coeffs.into_boxed_slice();
    Ok(ERIData {
        shell_pairs,
        coeffs: mmd_coeffs,
    })
}

fn compute_off_diagonal_eri_stored<'a, T, S, R, W: arrays::scratch::ScratchSource>(
    eri_tensor: &mut T,
    eri_data: &ERIData<'a, S>,
    owned: Option<&[bool]>,
    ws: &mut W,
    omega: f64
) -> Result<(), ArenaError>
where
    T: crate::TensorChoice,
    S: ShellKind,
    R: RangeVersion,
{
    let shell_pairs = eri_data.shell_pairs;
    let coefficients = &eri_data.coeffs;
    let mut arena = ws.get_arena();
    for (pair_idx, (pair_ab, bra_coeffs)) in shell_pairs.iter().zip(coefficients.iter()).enumerate() {
        if owned.is_some_and(|owned| !owned[pair_idx]) {
            continue;
        }
        // One row: the bra coefficients stay in the arena for its quartets, each computed in its own scope.
        arena.scope(|row| {
            let bra_coefficients = bra_coeffs.mmd_transpose_in(row)?;
            for (pair_cd, ket_coefficients) in shell_pairs.iter().skip(pair_idx+1).zip(coefficients.iter().skip(pair_idx+1)) {
                row.scope(|quartet| {
                    let eri_batch = compute_eri_batch::<S, S, R>(
                        &pair_ab,
                        &pair_cd,
                        &bra_coefficients,
                        &ket_coefficients,
                        quartet,
                        omega
                    )?;
                    let _span = instrumentation::fine_span("eri::in_memory::distribute");
                    eri_tensor.distribute_block(&EriBlock::new(&eri_batch, None), pair_ab, pair_cd, quartet)
                })?;
            }
            Ok::<(), ArenaError>(())
        })?;
    }
    Ok(())
}
