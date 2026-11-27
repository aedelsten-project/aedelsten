// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// One-electron gradients contracted with a density, for all centres in one pass over the shell pairs: a
// (3, n_centres) matrix
//
//   G[c, X] = sum_mn D_mn dO_mn / dX_c
//
// from the same kernels and `GradientArguments` as the per-centre derivative integrals
// (`one_electron_integral_gradient_loop`).
//
// Per shell pair (a, b) on the centres (A, B) the affected atoms are A, B and, for an operator whose sources are
// centres (nuclei, core AIMPs), each source's centre. The terms are additive, as in the per-centre route:
//   * A == B (which includes a shell with itself): one call with both flags set, where the d/dR terms cancel (see
//     gradient_kernels::centre_gradient_term); that is also what makes the triangular primitive pairs of a
//     same-shell pair valid;
//   * A != B: one call. The kernels form d/dR and d/dP in the same loop, so the resolved mode returns d/dA and d/dP
//     together (six planes) and dB = dP - dA follows after contraction (d/dA + d/dB = d/dP, since a/p + b/p = 1);
//     for operators without a position (overlap, kinetic) dP = 0 and dB = -dA from the three-plane call;
//   * every source: a call with its own term only.
//
// Weights. A pair stores one of the two mirror blocks (mu in a, nu in b) of the symmetric matrix dO/dX, and
// D is symmetrised (D + D^T)/2 once, so an off-diagonal shell pair counts twice and a pair of a shell with
// itself once (within it, likewise, its off-diagonal contracted pairs twice).
//
// Atoms with an index >= the number of columns of the gradient matrix (orthogonality-function ghosts, last in the
// centre list) are not accumulated.

use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;

use crate::gradient_kernels::{ArrayBatchGradientKernel, GradientArguments};
use crate::pieces::SiteRuns;

// 2 for a pair of two shells (it stands for its mirror too), 1 for a shell with itself.
#[inline(always)]
fn pair_weight<S: ShellKind>(pair: &ShellPair<S>) -> f64 {
    if pair.get_single_centre() { 1.0 } else { 2.0 }
}

/// Contracts three planes (x, y, z) of a derivative batch `(sph_a, sph_b, planes, contracted-pair dimensions)`,
/// starting at plane `first_plane`, with the symmetrised density. (A batch has three planes, or six in the resolved
/// mode: d/dA then d/dP.) Contraction ca of shell a occupies the AOs idx_a + ca * sph_a ..., likewise for b; a
/// same-shell pair holds only ca <= cb (ShellPair::pair_slot), each off-diagonal one standing for its mirror too.
fn contract_gradient_batch<S: ShellKind>(pair: &ShellPair<S>, ds: &Matrix<f64>, batch: &arrays::array5d::Array5DViewMut<'_, f64>, first_plane: usize) -> [f64; 3] {
    use arrays::array5d::Array5DTrait;
    let (sph_a, sph_b, n_planes) = (batch.dim0(), batch.dim1(), batch.dim2());
    let (n_cont_a, n_cont_b, single) = (pair.get_n_cont_a(), pair.get_n_cont_b(), pair.get_single_centre());
    let block_len = sph_a * sph_b;
    let data = batch.get_data();
    let mut out = [0.0; 3];
    for (plane, value) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        for cb in 0..n_cont_b {
            let col0 = pair.get_idx_b() + cb * sph_b;
            let a_end = if single { cb + 1 } else { n_cont_a };
            for ca in 0..a_end {
                let row0 = pair.get_idx_a() + ca * sph_a;
                let Some(slot) = pair.pair_slot(ca, cb) else {
                    continue;
                };
                let block = &data[(first_plane + plane + n_planes * slot) * block_len..][..block_len];
                let mut block_sum = 0.0;
                for j in 0..sph_b {
                    for i in 0..sph_a {
                        block_sum += ds[(row0 + i, col0 + j)] * block[i + sph_a * j];
                    }
                }
                sum += if single && ca != cb { 2.0 * block_sum } else { block_sum };
            }
        }
        *value = pair_weight(pair) * sum;
    }
    out
}

/// `(D + D^T) / 2`.
fn symmetrise<Md: MatrixTrait<f64>>(density: &Md) -> Matrix<f64> {
    let n = density.rows();
    debug_assert_eq!(n, density.cols());
    let mut symmetrised = Matrix::<f64>::new(n, n);
    for j in 0..n {
        for i in 0..n {
            symmetrised[(i, j)] = 0.5 * (density[(i, j)] + density[(j, i)]);
        }
    }
    symmetrised
}

/// Fills `gradient` (3 x n_centres, overwritten) with `sum_mn D_mn dO_mn/dX` for every centre X.
///
/// `operator_centres` lists, for an operator whose sources include centres, `(position in the source list, global
/// centre index)` of each such source (nuclei; core AIMPs) -- the argument `operator_gradient_idx` of the per-centre
/// route, for all sources at once. Empty for operators without centre-bound sources. `sourceless` marks an operator
/// that depends on nothing but the basis centres (overlap, kinetic): d/dP = 0 for it, so d/dB = -d/dA.
///
/// `active` (one flag per column of `gradient`; `None` = all) freezes centres: their columns are never accumulated
/// into and so stay zero, and work that only feeds frozen columns is skipped -- a pair whose two centres are both
/// frozen, and the source term of a frozen centre's nucleus or core AIMP, for every pair.
pub(crate) fn one_electron_gradient_contraction_loop<Mg, Md, S, K>(
    gradient: &mut Mg,
    density: &Md,
    shell_pairs: &[ShellPair<S>],
    operator_centres: &[(usize, usize)],
    sourceless: bool,
    active: Option<&[bool]>,
    scratch: &mut arrays::scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<(), arrays::scratch::ArenaError>
where
    Mg: MatrixTraitMut<f64>,
    Md: MatrixTrait<f64>,
    S: ShellKind,
    K: ArrayBatchGradientKernel,
{
    assert_eq!(gradient.rows(), 3, "the gradient matrix is (3, n_centres)");
    let n_centres = gradient.cols();
    if let Some(mask) = active {
        assert_eq!(mask.len(), n_centres, "the active mask has one flag per gradient centre");
    }
    // Centres without a column (orthogonality ghosts) and frozen centres are not accumulated.
    let is_active = |atom: usize| atom < n_centres && active.map_or(true, |mask| mask[atom]);
    gradient.fill(0.0);
    let ds = symmetrise(density);

    let costs: Vec<f64> = shell_pairs.iter().map(crate::pieces::pair_cost).collect();
    let n_sites = if K::SUMS_OVER_SITES { additional_arguments.n_sites() } else { 0 };
    let pieces = crate::pieces::sorted_pieces(&costs, n_sites, usize::MAX, crate::pieces::Cut::PairsFirst, comm);
    let handout = crate::pieces::Handout::new(&pieces, comm);
    for piece in handout.iter() {
        // The piece's sources: its run of the operator's, and the centre-bound ones among them at their places in it.
        let sites = if n_sites > 0 { piece.sites } else { 0..additional_arguments.n_sites() };
        let arguments = additional_arguments.run(sites.clone());
        let run_centres: Vec<(usize, usize)> = operator_centres
            .iter()
            .filter(|&&(k, _)| sites.contains(&k) || additional_arguments.n_sites() == 0)
            .map(|&(k, centre)| (k - sites.start, centre))
            .collect();
        let additional_arguments = &arguments;
        let operator_centres = &run_centres[..];
        for pair in &shell_pairs[piece.pairs] {
            let (centre_a, centre_b) = (pair.get_centre_idx_a(), pair.get_centre_idx_b());
            let (active_a, active_b) = (is_active(centre_a), is_active(centre_b));

            // The source of the operator centred on `atom`, if there is one.
            let source_at = |atom: usize| operator_centres.iter().find(|&&(_, centre)| centre == atom).map(|&(k, _)| k);

            // One kernel call, its planes contracted: the first three (x, y, z), and in the resolved mode the second
            // three (d/dP) as well.
            let mut evaluate = |arguments: GradientArguments| -> Result<([f64; 3], [f64; 3]), arrays::scratch::ArenaError> {
                let arena = &scratch.get_arena();
                let batch = crate::inner_functions::pair_gradient_batch::<K, S>(pair, arguments, arena, additional_arguments)?;
                let _span = instrumentation::fine_span("oneint::contract_density");
                let first = contract_gradient_batch(pair, &ds, &batch, 0);
                let second = if arguments.resolved { contract_gradient_batch(pair, &ds, &batch, 3) } else { [0.0; 3] };
                Ok((first, second))
            };
            let mut add = |atom: usize, value: [f64; 3], sign: f64| {
                if is_active(atom) {
                    for (plane, v) in value.iter().enumerate() {
                        gradient[(plane, atom)] += sign * v;
                    }
                }
            };

            // Basis-function terms.
            let mut merged_a = None;
            if centre_a == centre_b {
                // One call: the d/dR terms cancel, and a source on this atom is folded in (additively). Nothing to
                // do if the atom is frozen: both the basis terms and that source only feed its column.
                if active_a {
                    merged_a = source_at(centre_a);
                    add(centre_a, evaluate(GradientArguments::new(true, true, merged_a))?.0, 1.0);
                }
            } else if !active_a && !active_b {
                // Both centres frozen: no basis-function term to compute.
            } else if sourceless {
                // d/dP = 0: d/dB = -d/dA, one call, whichever of the two centres are active.
                let d_a = evaluate(GradientArguments::new(true, false, None))?.0;
                add(centre_a, d_a, 1.0);
                add(centre_b, d_a, -1.0);
            } else if active_a && active_b {
                // d/dA and d/dP from one evaluation (the kernels form d/dR and d/dP together anyway), and
                // d/dB = d/dP - d/dA after contraction (d/dA + d/dB = d/dP, a/p + b/p = 1). The sources are separate
                // calls below (a folded source would not cancel in d/dP - d/dA).
                let (d_a, d_p) = evaluate(GradientArguments::new_resolved())?;
                add(centre_a, d_a, 1.0);
                add(centre_b, d_p, 1.0);
                add(centre_b, d_a, -1.0);
            } else if active_a {
                // Only A moves: the plain d/dA call, no d/dP needed.
                add(centre_a, evaluate(GradientArguments::new(true, false, None))?.0, 1.0);
            } else {
                add(centre_b, evaluate(GradientArguments::new(false, true, None))?.0, 1.0);
            }
            // Every source (on any atom, several on one) contributes its own term only; sources are additive. Only a
            // source folded into the same-atom call above is skipped, and so is one on a frozen centre.
            for &(k, centre) in operator_centres {
                if Some(k) != merged_a && is_active(centre) {
                    add(centre, evaluate(GradientArguments::new(false, false, Some(k)))?.0, 1.0);
                }
            }
        }
    }

    let _allreduce = instrumentation::span("oneint::allreduce");
    comm.sum(gradient.get_data_mut())
        .expect("MPI allreduce failed while combining one-electron gradients across ranks");
    Ok(())
}
