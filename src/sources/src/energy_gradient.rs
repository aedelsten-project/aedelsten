// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Geometric gradients of the classical energies in `energy.rs`, one function per energy function there,
//! differentiating the same pair expression.
//!
//! A pair term `W g(|d|)` with `d = r_1 - r_2` has `dE/dr_1 = W B_1(r) d` and `dE/dr_2 = -W B_1(r) d`, where
//! `B_1 = (1/r) dg/dr` (`math_utilities::radial`). A multipole source's potential at a probe differentiates one
//! Cartesian order higher (`PotentialTables::potential_gradient`). Every function hands `dE/dr` of member `i`
//! of a set to that set's callback, `add(i, gradient)`; a set that does not move gets a no-op. Multipole sources
//! (the external multipoles) are always fixed here: only the probes' gradients are formed.

use math_utilities::radial::{Coulomb, ErfCoulomb, GaussianScreenedCoulomb, RadialKernel, ScreenedCoulomb};

use crate::charges::{MultipoleSourceTrait, IndexedGaussianCharge, PointChargeTrait, ScreenedChargeTrait, ScreenedCoulombPotential};
use crate::energy::PotentialTables;

/// `d = p1 - p2` and `|d|`.
fn separation(p1: &[f64; 3], p2: &[f64; 3]) -> ([f64; 3], f64) {
    let d = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
    (d, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
}

fn scaled(d: [f64; 3], factor: f64) -> [f64; 3] {
    [factor * d[0], factor * d[1], factor * d[2]]
}

/// `B_1` of an AIMP's screening correction alone, `sum_i A_i exp(-c_i r^2)/r` (no bare 1/r).
fn screen_b1(screen: &ScreenedCoulombPotential, r: f64) -> f64 {
    screen.get_exponents().iter().zip(screen.get_coefficients()).map(|(&c, &a)| a * GaussianScreenedCoulomb { c }.b(1, r)).sum()
}

/// Adds `W B_1 d` to member `i` of the first set and its negative to member `j` of the second.
fn add_pair(w_b1: f64, d: [f64; 3], i: usize, j: usize, add_1: &mut impl FnMut(usize, [f64; 3]), add_2: &mut impl FnMut(usize, [f64; 3])) {
    add_1(i, scaled(d, w_b1));
    add_2(j, scaled(d, -w_b1));
}

/// The gradient of `compute_classical_coulomb_self_interaction`.
pub(crate) fn classical_coulomb_self_interaction_gradient<T: PointChargeTrait>(charges: &[T], mut add: impl FnMut(usize, [f64; 3])) {
    for (i, charge1) in charges.iter().enumerate() {
        for (j, charge2) in charges.iter().enumerate().take(i) {
            let (d, r) = separation(charge1.get_position(), charge2.get_position());
            let g = scaled(d, charge1.get_charge() * charge2.get_charge() * Coulomb.b(1, r));
            add(i, g);
            add(j, scaled(g, -1.0));
        }
    }
}

/// The gradient of `compute_classical_coulomb_interaction`.
pub(crate) fn classical_coulomb_interaction_gradient<T1: PointChargeTrait, T2: PointChargeTrait>(
    charges1: &[T1],
    charges2: &[T2],
    mut add_1: impl FnMut(usize, [f64; 3]),
    mut add_2: impl FnMut(usize, [f64; 3]),
) {
    for (j, charge2) in charges2.iter().enumerate() {
        for (i, charge1) in charges1.iter().enumerate() {
            let (d, r) = separation(charge1.get_position(), charge2.get_position());
            add_pair(charge1.get_charge() * charge2.get_charge() * Coulomb.b(1, r), d, i, j, &mut add_1, &mut add_2);
        }
    }
}

/// The gradient of `compute_gaussian_coulomb_self_interaction`: `erf(sqrt(zeta) r)/r`, `zeta = e_1 e_2/(e_1 + e_2)`.
pub(crate) fn gaussian_coulomb_self_interaction_gradient(charges: &[IndexedGaussianCharge], mut add: impl FnMut(usize, [f64; 3])) {
    for (i, charge1) in charges.iter().enumerate() {
        for (j, charge2) in charges.iter().enumerate().take(i) {
            let (d, r) = separation(&charge1.position, &charge2.position);
            let zeta = charge1.exp * charge2.exp / (charge1.exp + charge2.exp);
            let g = scaled(d, charge1.charge * charge2.charge * ErfCoulomb { kappa: zeta.sqrt() }.b(1, r));
            add(i, g);
            add(j, scaled(g, -1.0));
        }
    }
}

/// The gradient of `compute_gaussian_coulomb_interaction`: `erf(sqrt(e) r)/r` of the finite nucleus against a point
/// charge.
pub(crate) fn gaussian_coulomb_interaction_gradient<T: PointChargeTrait>(
    nuclei: &[IndexedGaussianCharge],
    charges: &[T],
    mut add_nuclei: impl FnMut(usize, [f64; 3]),
    mut add_charges: impl FnMut(usize, [f64; 3]),
) {
    for (i, nucleus) in nuclei.iter().enumerate() {
        for (j, charge) in charges.iter().enumerate() {
            let (d, r) = separation(&nucleus.position, charge.get_position());
            let w_b1 = nucleus.charge * charge.get_charge() * ErfCoulomb { kappa: nucleus.exp.sqrt() }.b(1, r);
            add_pair(w_b1, d, i, j, &mut add_nuclei, &mut add_charges);
        }
    }
}

/// The gradient of `compute_gaussian_screened_coulomb_interaction`:
/// `q_n q_a (erf(sqrt(e) r) + sum_i A_i exp(-c_i r^2))/r`.
pub(crate) fn gaussian_screened_coulomb_interaction_gradient<T: ScreenedChargeTrait>(
    nuclei: &[IndexedGaussianCharge],
    screened: &[T],
    mut add_nuclei: impl FnMut(usize, [f64; 3]),
    mut add_screened: impl FnMut(usize, [f64; 3]),
) {
    for (i, nucleus) in nuclei.iter().enumerate() {
        for (j, potential) in screened.iter().enumerate() {
            let (d, r) = separation(&nucleus.position, potential.get_position());
            let b1 = ErfCoulomb { kappa: nucleus.exp.sqrt() }.b(1, r) + screen_b1(potential.get_screened_coulomb_potential(), r);
            add_pair(nucleus.charge * potential.get_charge() * b1, d, i, j, &mut add_nuclei, &mut add_screened);
        }
    }
}

/// The gradient of `compute_screened_coulomb_interaction`: `q_p q_a (1 + sum_i A_i exp(-c_i r^2))/r`.
pub fn screened_coulomb_interaction_gradient<T1: PointChargeTrait, T2: ScreenedChargeTrait>(
    charges: &[T1],
    screened: &[T2],
    mut add_charges: impl FnMut(usize, [f64; 3]),
    mut add_screened: impl FnMut(usize, [f64; 3]),
) {
    for (i, charge) in charges.iter().enumerate() {
        for (j, potential) in screened.iter().enumerate() {
            let (d, r) = separation(charge.get_position(), potential.get_position());
            let screen = potential.get_screened_coulomb_potential();
            let b1 = ScreenedCoulomb { exponents: screen.get_exponents(), coefficients: screen.get_coefficients() }.b(1, r);
            add_pair(charge.get_charge() * potential.get_charge() * b1, d, i, j, &mut add_charges, &mut add_screened);
        }
    }
}

/// The gradient of `compute_general_multipole_interaction`: the probes (point charges) move, the multipole sources are
/// fixed.
pub(crate) fn general_multipole_interaction_gradient<T1: PointChargeTrait, T2: MultipoleSourceTrait>(
    probes: &[T1],
    sources: &[T2],
    mut add_probes: impl FnMut(usize, [f64; 3]),
) {
    let mut scratch = PotentialTables::new();
    for (i, probe) in probes.iter().enumerate() {
        for source in sources.iter() {
            let field_gradient = scratch.potential_gradient(source, probe.get_position(), &Coulomb);
            add_probes(i, scaled(field_gradient, probe.get_charge()));
        }
    }
}

/// The gradient of `compute_gaussian_general_multipole_interaction`: finite nuclei as probes (`erf(sqrt(e) r)/r`),
/// sources fixed.
pub(crate) fn gaussian_general_multipole_interaction_gradient<T: MultipoleSourceTrait>(
    nuclei: &[IndexedGaussianCharge],
    sources: &[T],
    mut add_nuclei: impl FnMut(usize, [f64; 3]),
) {
    let mut scratch = PotentialTables::new();
    for (i, nucleus) in nuclei.iter().enumerate() {
        let kernel = ErfCoulomb { kappa: nucleus.exp.sqrt() };
        for source in sources.iter() {
            let field_gradient = scratch.potential_gradient(source, &nucleus.position, &kernel);
            add_nuclei(i, scaled(field_gradient, nucleus.charge));
        }
    }
}

/// The gradient of `compute_general_multipole_screened_interaction`: the screened charges (AIMPs) are the probes and
/// move, the multipole sources are fixed.
pub fn general_multipole_screened_interaction_gradient<T1: MultipoleSourceTrait, T2: ScreenedChargeTrait>(
    sources: &[T1],
    screened: &[T2],
    mut add_screened: impl FnMut(usize, [f64; 3]),
) {
    let mut scratch = PotentialTables::new();
    for (i, potential) in screened.iter().enumerate() {
        let screen = potential.get_screened_coulomb_potential();
        let kernel = ScreenedCoulomb { exponents: screen.get_exponents(), coefficients: screen.get_coefficients() };
        for source in sources.iter() {
            let field_gradient = scratch.potential_gradient(source, potential.get_position(), &kernel);
            add_screened(i, scaled(field_gradient, potential.get_charge()));
        }
    }
}

/// The gradient of `compute_screened_coulomb_self_interaction`: `q_1 q_2 (1 + s_1(r) + s_2(r))/r` per pair, `s` the
/// relative screening correction (see that function).
pub fn screened_coulomb_self_interaction_gradient<T: ScreenedChargeTrait>(screened: &[T], mut add: impl FnMut(usize, [f64; 3])) {
    for (i, potential1) in screened.iter().enumerate() {
        for (j, potential2) in screened.iter().enumerate().take(i) {
            let (d, r) = separation(potential1.get_position(), potential2.get_position());
            let (q1, q2) = (potential1.get_charge(), potential2.get_charge());
            let b1 = q1 * q2
                * (Coulomb.b(1, r) + screen_b1(potential1.get_screened_coulomb_potential(), r) + screen_b1(potential2.get_screened_coulomb_potential(), r));
            let g = scaled(d, b1);
            add(i, g);
            add(j, scaled(g, -1.0));
        }
    }
}

/// The gradient of `compute_screened_coulomb_cross_interaction`, the same pair form between two different sets.
pub fn screened_coulomb_cross_interaction_gradient<T1: ScreenedChargeTrait, T2: ScreenedChargeTrait>(
    screened1: &[T1],
    screened2: &[T2],
    mut add_1: impl FnMut(usize, [f64; 3]),
    mut add_2: impl FnMut(usize, [f64; 3]),
) {
    for (i, potential1) in screened1.iter().enumerate() {
        for (j, potential2) in screened2.iter().enumerate() {
            let (d, r) = separation(potential1.get_position(), potential2.get_position());
            let (q1, q2) = (potential1.get_charge(), potential2.get_charge());
            let b1 = q1 * q2
                * (Coulomb.b(1, r) + screen_b1(potential1.get_screened_coulomb_potential(), r) + screen_b1(potential2.get_screened_coulomb_potential(), r));
            add_pair(b1, d, i, j, &mut add_1, &mut add_2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::*;
    use crate::{IndexedPointCharge, PointMultipole, SiteTrait};

    struct Aimp {
        position: [f64; 3],
        charge: f64,
        screen: ScreenedCoulombPotential,
    }
    impl SiteTrait for Aimp {
        fn get_position(&self) -> &[f64; 3] {
            &self.position
        }
    }
    impl PointChargeTrait for Aimp {
        fn get_charge(&self) -> f64 {
            self.charge
        }
    }
    impl ScreenedChargeTrait for Aimp {
        fn get_screened_coulomb_potential(&self) -> &ScreenedCoulombPotential {
            &self.screen
        }
    }

    const P1: [[f64; 3]; 3] = [[0.1, -0.3, 0.2], [1.4, 0.9, -0.6], [-1.1, 1.3, 1.0]];
    const P2: [[f64; 3]; 2] = [[0.7, -1.2, 2.1], [-1.6, -0.4, -1.3]];
    const Q1: [f64; 3] = [2.0, 3.0, 1.5];
    const Q2: [f64; 2] = [1.7, 2.6];

    fn points(p: &[[f64; 3]], q: &[f64]) -> Vec<IndexedPointCharge> {
        p.iter().zip(q).enumerate().map(|(i, (&p, &q))| IndexedPointCharge::new(i, p, q)).collect()
    }
    fn gaussians(p: &[[f64; 3]], q: &[f64]) -> Vec<IndexedGaussianCharge> {
        p.iter().zip(q).enumerate().map(|(i, (&p, &q))| IndexedGaussianCharge::new(i, p, q, 1.3 + 0.4 * i as f64)).collect()
    }
    fn aimps(p: &[[f64; 3]], q: &[f64]) -> Vec<Aimp> {
        p.iter()
            .zip(q)
            .enumerate()
            .map(|(i, (&p, &q))| Aimp { position: p, charge: q, screen: ScreenedCoulombPotential::new(vec![0.8 + 0.3 * i as f64, 3.1], vec![1.2, -0.45]) })
            .collect()
    }
    fn multipoles() -> Vec<PointMultipole> {
        let mut values = vec![0.0; 20];
        for (k, v) in values.iter_mut().enumerate() {
            *v = 0.3 * ((k * 7 % 11) as f64 - 5.0) / (1.0 + k as f64);
        }
        vec![PointMultipole::from_xfield_moments([3.1, 2.4, -2.2], 3, &values), PointMultipole::from_xfield_moments([-2.8, 3.3, 2.5], 2, &values[..10])]
    }

    /// Central differences of `energy(p1, p2)` for every coordinate of both sets against `gradient(p1, p2)`;
    /// `moves2` = false checks that the second set gets nothing.
    fn check(
        label: &str,
        energy: &dyn Fn(&[[f64; 3]], &[[f64; 3]]) -> f64,
        gradient: &dyn Fn(&[[f64; 3]], &[[f64; 3]], &mut Vec<[f64; 3]>, &mut Vec<[f64; 3]>),
        p1: &[[f64; 3]],
        p2: &[[f64; 3]],
        moves2: bool,
    ) {
        let (mut g1, mut g2) = (vec![[0.0; 3]; p1.len()], vec![[0.0; 3]; p2.len()]);
        gradient(p1, p2, &mut g1, &mut g2);
        let h = 1e-5;
        let mut largest: f64 = 0.0;
        for (set, analytic) in [(0, &g1), (1, &g2)] {
            for member in 0..analytic.len() {
                for axis in 0..3 {
                    let displaced = |delta: f64| {
                        let (mut a, mut b) = (p1.to_vec(), p2.to_vec());
                        if set == 0 { a[member][axis] += delta } else { b[member][axis] += delta }
                        energy(&a, &b)
                    };
                    let fd = if set == 1 && !moves2 { 0.0 } else { (displaced(h) - displaced(-h)) / (2.0 * h) };
                    largest = largest.max(fd.abs());
                    let got = analytic[member][axis];
                    assert!((got - fd).abs() < 1e-7 * (1.0 + fd.abs()), "{label}: set {set} member {member} axis {axis}: analytic {got}, finite difference {fd}");
                }
            }
        }
        assert!(largest > 1e-3, "{label}: the gradient is (nearly) zero, the check is vacuous");
    }

    fn adder(g: &mut Vec<[f64; 3]>) -> impl FnMut(usize, [f64; 3]) + '_ {
        move |i, v| {
            for k in 0..3 {
                g[i][k] += v[k];
            }
        }
    }

    #[test]
    fn every_classical_gradient_matches_finite_differences_of_its_energy() {
        check("point self", &|a, _| compute_classical_coulomb_self_interaction(&points(a, &Q1)),
            &|a, _, g1, _| classical_coulomb_self_interaction_gradient(&points(a, &Q1), adder(g1)), &P1, &[], false);
        check("point cross", &|a, b| compute_classical_coulomb_interaction(&points(a, &Q1), &points(b, &Q2)),
            &|a, b, g1, g2| classical_coulomb_interaction_gradient(&points(a, &Q1), &points(b, &Q2), adder(g1), adder(g2)), &P1, &P2, true);
        check("gaussian self", &|a, _| compute_gaussian_coulomb_self_interaction(&gaussians(a, &Q1)),
            &|a, _, g1, _| gaussian_coulomb_self_interaction_gradient(&gaussians(a, &Q1), adder(g1)), &P1, &[], false);
        check("gaussian-point", &|a, b| compute_gaussian_coulomb_interaction(&gaussians(a, &Q1), &points(b, &Q2)),
            &|a, b, g1, g2| gaussian_coulomb_interaction_gradient(&gaussians(a, &Q1), &points(b, &Q2), adder(g1), adder(g2)), &P1, &P2, true);
        check("gaussian-screened", &|a, b| compute_gaussian_screened_coulomb_interaction(&gaussians(a, &Q1), &aimps(b, &Q2)),
            &|a, b, g1, g2| gaussian_screened_coulomb_interaction_gradient(&gaussians(a, &Q1), &aimps(b, &Q2), adder(g1), adder(g2)), &P1, &P2, true);
        check("point-screened", &|a, b| compute_screened_coulomb_interaction(&points(a, &Q1), &aimps(b, &Q2)),
            &|a, b, g1, g2| screened_coulomb_interaction_gradient(&points(a, &Q1), &aimps(b, &Q2), adder(g1), adder(g2)), &P1, &P2, true);
        check("screened self", &|a, _| compute_screened_coulomb_self_interaction(&aimps(a, &Q1)),
            &|a, _, g1, _| screened_coulomb_self_interaction_gradient(&aimps(a, &Q1), adder(g1)), &P1, &[], false);
        check("screened cross", &|a, b| compute_screened_coulomb_cross_interaction(&aimps(a, &Q1), &aimps(b, &Q2)),
            &|a, b, g1, g2| screened_coulomb_cross_interaction_gradient(&aimps(a, &Q1), &aimps(b, &Q2), adder(g1), adder(g2)), &P1, &P2, true);
        // Multipole sources (octupole and quadrupole) are fixed; only the probes move.
        check("point-multipole", &|a, _| compute_general_multipole_interaction(&points(a, &Q1), &multipoles()),
            &|a, _, g1, _| general_multipole_interaction_gradient(&points(a, &Q1), &multipoles(), adder(g1)), &P1, &[], false);
        check("gaussian-multipole", &|a, _| compute_gaussian_general_multipole_interaction(&gaussians(a, &Q1), &multipoles()),
            &|a, _, g1, _| gaussian_general_multipole_interaction_gradient(&gaussians(a, &Q1), &multipoles(), adder(g1)), &P1, &[], false);
        check("screened-multipole", &|a, _| compute_general_multipole_screened_interaction(&multipoles(), &aimps(a, &Q1)),
            &|a, _, g1, _| general_multipole_screened_interaction_gradient(&multipoles(), &aimps(a, &Q1), adder(g1)), &P1, &[], false);
    }
}
