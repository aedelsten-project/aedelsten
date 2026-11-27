// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The spin-averaged two-electron mean field of one atom kind from the integrals of `eri::spin_orbit`:
//!
//! ```text
//!   W^SSO_k(p, q) = sum_M n_M [ X_k(pq|MM) - 1/2 ( X_k(pM|qM) - X_k(qM|pM) ) ],
//!   W^SOO_k(p, q) = sum_M n_M [            -     ( X_k(pM|qM) - X_k(qM|pM) ) ],
//! ```
//!
//! `p, q` working AOs, `M` the occupied reference orbitals' AOs with `n_M` electrons each: the spin-same-orbit
//! Coulomb and exchange terms, and the spin-other-orbit exchange term (its Coulomb term vanishes for a spin-averaged
//! density). Their sum is the Breit-Pauli mean field.
//! `W_k` is antisymmetric. With a spherical density it couples only functions of equal angular momentum.

use arrays::array3d::{Array3DTraitMut, Array3DViewMut};
use arrays::array8d::Array8DViewMut;
use arrays::array5d::Array5D;
use arrays::scratch::{Arena, ArenaError, Scratch, ScratchError};
use basis::contraction::Contracted;
use basis::shell_pairs::ShellPair;
use basis::shell_types::HarmonicType;
use eri::spin_orbit::SpinOrbitBra;
use mmd::ketside_hermite_coefficients::compute_ketside_hermite_derivative_coefficients;

use crate::atom::AtomKind;
use crate::AmfiError;

// One shell of a pair: its first local AO, components and contracted functions.
#[derive(Clone, Copy)]
struct Side {
    first: usize,
    degeneracy: usize,
    n_cont: usize,
}

impl Side {
    // The local AO of contracted function `cont`, component `m`.
    fn ao(&self, cont: usize, m: usize) -> usize {
        self.first + cont * self.degeneracy + m
    }
}

// A pair's shells `a` and `b`.
fn sides(pair: &ShellPair<Contracted>) -> [Side; 2] {
    [
        Side { first: pair.get_idx_a(), degeneracy: pair.get_degeneracy_a(), n_cont: pair.get_n_cont_a() },
        Side { first: pair.get_idx_b(), degeneracy: pair.get_degeneracy_b(), n_cont: pair.get_n_cont_b() },
    ]
}

// The contracted pairs `(a, b)` a pair holds, with their positions (`ShellPair::pair_slot`): every pair, or `a <= b`
// for a pair of a shell with itself.
fn slots(pair: &ShellPair<Contracted>) -> Vec<(usize, usize, usize)> {
    let [side_a, side_b] = sides(pair);
    let mut slots = Vec::new();
    for b in 0..side_b.n_cont {
        let a_end = if pair.get_single_centre() { b + 1 } else { side_a.n_cont };
        for a in 0..a_end {
            if let Some(slot) = pair.pair_slot(a, b) {
                slots.push((a, b, slot));
            }
        }
    }
    slots
}

// Whether a spherical mean field can couple shells of angular momenta and harmonic types `a` and `b`: equal angular
// momenta, or angular momenta differing by an even number with the higher one Cartesian (whose components include
// the lower one).
fn can_couple(a: (usize, HarmonicType), b: (usize, HarmonicType)) -> bool {
    let (low, high) = if a.0 <= b.0 { (a, b) } else { (b, a) };
    low.0 == high.0 || ((high.0 - low.0) % 2 == 0 && high.1 == HarmonicType::Cartesian)
}

// The kets' coefficients (`E`), built once and kept for every bra; a bra's data (`SpinOrbitBra`) is built in its
// own iteration of the outer loop.
fn ket_data(pairs: &[&ShellPair<Contracted>], arena: &mut Arena<'_>) -> Result<Vec<Array5D<f64>>, ArenaError> {
    pairs.iter().map(|pair| compute_ketside_hermite_derivative_coefficients(pair, 0, arena)).collect()
}

// The three batches of the quartet `(bra | ket)` from `arena`, computed if `COMPUTE`, else only their buffers taken.
fn quartet<'a, const COMPUTE: bool>(
    (bra, bra_data): (&ShellPair<Contracted>, &SpinOrbitBra),
    (ket, ket_coefficients): (&ShellPair<Contracted>, &Array5D<f64>),
    arena: &Arena<'a>,
) -> Result<[Array8DViewMut<'a, f64>; 3], ArenaError> {
    let (bra_dims, ket_dims) = (bra.get_contracted_pair_dims(), ket.get_contracted_pair_dims());
    let batch = || {
        Array8DViewMut::<f64>::from_arena(
            bra.get_degeneracy_a(), bra.get_degeneracy_b(), ket.get_degeneracy_a(), ket.get_degeneracy_b(),
            ket_dims.0, ket_dims.1, bra_dims.0, bra_dims.1, arena,
        )
    };
    let mut batches = [batch()?, batch()?, batch()?];
    if COMPUTE {
        eri::spin_orbit::compute_spin_orbit_quartet(bra, bra_data, ket, ket_coefficients, &mut batches, arena)?;
    } else {
        eri::spin_orbit::reserve_spin_orbit_quartet(bra, ket, arena)?;
    }
    Ok(batches)
}

// The value at components `(i, j, k, l)`, ket slot `ket_slot` and bra slot `bra_slot` of a batch (`quartet`).
fn at(batch: &Array8DViewMut<'_, f64>, [i, j, k, l]: [usize; 4], ket_slot: usize, ket_rows: usize, bra_slot: usize, bra_rows: usize) -> f64 {
    batch[(i, j, k, l, ket_slot % ket_rows, ket_slot / ket_rows, bra_slot % bra_rows, bra_slot / bra_rows)]
}

/// `kind`'s `W^SSO_x, W^SSO_y, W^SSO_z, W^SOO_x, W^SOO_y, W^SOO_z` into `output`, OVERWRITTEN, `(n, n, 6)` over the
/// kind's working AOs (`n = kind.get_n_working_aos()`). Scratch from `scratch`.
pub(crate) fn compute_kind_mean_field(kind: &AtomKind, output: &mut Array3DViewMut<'_, f64>, scratch: &mut Scratch) -> Result<(), AmfiError> {
    Ok(mean_field::<true>(kind, output, &mut scratch.get_arena())?)
}

/// The bytes of scratch `compute_kind_mean_field` takes for `kind`: its own buffers taken on `ws` (which grows until
/// they fit), no integral computed.
pub(crate) fn kind_scratch_bytes(kind: &AtomKind, ws: &mut Scratch) -> Result<usize, ScratchError> {
    ws.measure(|arena| mean_field::<false>(kind, &mut Array3DViewMut::new(0, 0, 0, &mut []), arena))
}

// `compute_kind_mean_field` if `COMPUTE`; else its buffers only, taken in the same order, `output` untouched.
fn mean_field<const COMPUTE: bool>(kind: &AtomKind, output: &mut Array3DViewMut<'_, f64>, arena: &mut Arena<'_>) -> Result<(), ArenaError> {
    if COMPUTE {
        output.get_data_mut().fill(0.0);
    }
    let n_working = kind.get_n_working_aos();
    let occupations = kind.get_occupations();
    let is_working = |first_ao: usize| first_ao < n_working;

    let pairs = kind.get_pairs();
    let coupled = |p: &ShellPair<Contracted>| can_couple((p.get_lqn_a(), p.get_harmonic_a()), (p.get_lqn_b(), p.get_harmonic_b()));
    let working_pairs: Vec<&ShellPair<Contracted>> =
        pairs.iter().filter(|p| is_working(p.get_idx_a()) && is_working(p.get_idx_b()) && coupled(p)).collect();
    let density_self_pairs: Vec<&ShellPair<Contracted>> =
        pairs.iter().filter(|p| !is_working(p.get_idx_a()) && p.get_single_centre()).collect();
    let mixed_pairs: Vec<&ShellPair<Contracted>> =
        pairs.iter().filter(|p| is_working(p.get_idx_a()) != is_working(p.get_idx_b())).collect();
    let density_kets = ket_data(&density_self_pairs, arena)?;
    let mixed_kets = ket_data(&mixed_pairs, arena)?;

    // The Coulomb term, (pq|MM), from the stored contracted pairs of each working pair; every other block, (qp|MM)
    // = -(pq|MM), placed with it.
    for bra in &working_pairs {
        let bra_data = &SpinOrbitBra::new(bra, arena)?;
        let [side_a, side_b] = sides(bra);
        let bra_rows = bra.get_contracted_pair_dims().0;
        let bra_slots = slots(bra);
        for (ket, ket_coefficients) in density_self_pairs.iter().zip(&density_kets) {
            let [side_c, _] = sides(ket);
            let ket_rows = ket.get_contracted_pair_dims().0;
            arena.scope(|scratch| -> Result<(), ArenaError> {
                let batches = quartet::<COMPUTE>((bra, bra_data), (ket, ket_coefficients), scratch)?;
                for (k, batch) in batches.iter().enumerate().filter(|_| COMPUTE) {
                    for &(cont_a, cont_b, bra_slot) in &bra_slots {
                        let whole_block = bra.get_single_centre() && cont_a == cont_b;
                        for cont_c in 0..side_c.n_cont {
                            let ket_slot = ket.pair_slot(cont_c, cont_c).expect("an unscreened pair");
                            for j in 0..side_b.degeneracy {
                                for i in 0..side_a.degeneracy {
                                    let coulomb: f64 = (0..side_c.degeneracy)
                                        .map(|m| occupations[side_c.ao(cont_c, m) - n_working] * at(batch, [i, j, m, m], ket_slot, ket_rows, bra_slot, bra_rows))
                                        .sum();
                                    let (p, q) = (side_a.ao(cont_a, i), side_b.ao(cont_b, j));
                                    output[(p, q, k)] += coulomb;
                                    if !whole_block {
                                        output[(q, p, k)] -= coulomb;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            })?;
        }
    }

    // The exchange terms, -[(pM|qM) - (qM|pM)] with weight 1/2 (spin-same-orbit) and 1 (spin-other-orbit): every
    // ordered combination of two working-density pairs on one density shell with coupled working shells; a bra stored
    // density first is turned round by (Mp| = -(pM|.
    for bra in &mixed_pairs {
        let bra_data = &SpinOrbitBra::new(bra, arena)?;
        let bra_working_is_b = !is_working(bra.get_idx_a());
        let [bra_a, bra_b] = sides(bra);
        let (bra_working, bra_density) = if bra_working_is_b { (bra_b, bra_a) } else { (bra_a, bra_b) };
        let bra_shell = if bra_working_is_b { (bra.get_lqn_b(), bra.get_harmonic_b()) } else { (bra.get_lqn_a(), bra.get_harmonic_a()) };
        let sign = if bra_working_is_b { -1.0 } else { 1.0 };
        let bra_rows = bra.get_contracted_pair_dims().0;
        let bra_slots = slots(bra);
        for (ket, ket_coefficients) in mixed_pairs.iter().zip(&mixed_kets) {
            let ket_working_is_b = !is_working(ket.get_idx_a());
            let [ket_a, ket_b] = sides(ket);
            let (ket_working, ket_density) = if ket_working_is_b { (ket_b, ket_a) } else { (ket_a, ket_b) };
            let ket_shell = if ket_working_is_b { (ket.get_lqn_b(), ket.get_harmonic_b()) } else { (ket.get_lqn_a(), ket.get_harmonic_a()) };
            if ket_density.first != bra_density.first || !can_couple(bra_shell, ket_shell) {
                continue;
            }
            let ket_rows = ket.get_contracted_pair_dims().0;
            let ket_slots = slots(ket);
            arena.scope(|scratch| -> Result<(), ArenaError> {
                let batches = quartet::<COMPUTE>((bra, bra_data), (ket, ket_coefficients), scratch)?;
                for (k, batch) in batches.iter().enumerate().filter(|_| COMPUTE) {
                    for &(bra_cont_a, bra_cont_b, bra_slot) in &bra_slots {
                        let (cont_p, cont_m) = if bra_working_is_b { (bra_cont_b, bra_cont_a) } else { (bra_cont_a, bra_cont_b) };
                        for &(ket_cont_a, ket_cont_b, ket_slot) in &ket_slots {
                            let (cont_q, cont_m_ket) = if ket_working_is_b { (ket_cont_b, ket_cont_a) } else { (ket_cont_a, ket_cont_b) };
                            if cont_m_ket != cont_m {
                                continue;
                            }
                            for q in 0..ket_working.degeneracy {
                                for p in 0..bra_working.degeneracy {
                                    let mut exchange = 0.0;
                                    for m in 0..bra_density.degeneracy {
                                        let (i, j) = if bra_working_is_b { (m, p) } else { (p, m) };
                                        let (r, t) = if ket_working_is_b { (m, q) } else { (q, m) };
                                        exchange += occupations[bra_density.ao(cont_m, m) - n_working]
                                            * at(batch, [i, j, r, t], ket_slot, ket_rows, bra_slot, bra_rows);
                                    }
                                    exchange *= sign;
                                    let (row, col) = (bra_working.ao(cont_p, p), ket_working.ao(cont_q, q));
                                    output[(row, col, k)] -= 0.5 * exchange;
                                    output[(col, row, k)] += 0.5 * exchange;
                                    output[(row, col, 3 + k)] -= exchange;
                                    output[(col, row, 3 + k)] += exchange;
                                }
                            }
                        }
                    }
                }
                Ok(())
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupling_needs_equal_lqn_or_a_cartesian_image() {
        use HarmonicType::{Cartesian, Spherical};
        assert!(can_couple((1, Cartesian), (1, Cartesian)));
        assert!(!can_couple((0, Cartesian), (1, Cartesian)));
        assert!(!can_couple((0, Cartesian), (2, Spherical)));
        assert!(can_couple((2, Cartesian), (0, Cartesian)));
        assert!(can_couple((1, Cartesian), (3, Cartesian)));
        assert!(!can_couple((2, Cartesian), (3, Cartesian)));
    }
}
