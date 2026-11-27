// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use crate::charges::{MultipoleSourceTrait, IndexedGaussianCharge, PointChargeTrait};

/// The Coulomb energy of a set of point charges among themselves, each pair once.
pub(crate) fn compute_classical_coulomb_self_interaction<T: PointChargeTrait>(point_charges: &[T]) -> f64 {
    let mut classical_coulomb = 0.0;
    for (j, charge1) in point_charges.iter().enumerate() {
        for charge2 in point_charges.iter().take(j) {
            let dist = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&charge2.get_position(), &charge1.get_position()));
            classical_coulomb += charge1.get_charge() * charge2.get_charge() / dist;
        }
    }
    classical_coulomb
}

/// The Coulomb energy between two sets of point charges.
pub(crate) fn compute_classical_coulomb_interaction<T1: PointChargeTrait, T2: PointChargeTrait>(point_charges1: &[T1], point_charges2: &[T2]) -> f64 {
    let mut classical_coulomb = 0.0;
    for point_charge2 in point_charges2.iter() {
        for point_charge1 in point_charges1.iter() {
            let dist = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&point_charge1.get_position(), &point_charge2.get_position()));
            classical_coulomb += point_charge2.get_charge() * point_charge1.get_charge() / dist;
        }
    }
    classical_coulomb
}
    
/// The Coulomb energy of a set of Gaussian nuclear charge distributions rho_k(r) = q_k (e_k/pi)^(3/2)
/// exp(-e_k r^2) among themselves, each pair once: q_1 q_2 erf(sqrt(zeta) r)/r, zeta = e_1 e_2 / (e_1 + e_2)
/// (`citations.rs`: `VISSCHER_DYALL_1997`).
pub(crate) fn compute_gaussian_coulomb_self_interaction(gaussian_charges: &[IndexedGaussianCharge]) -> f64 {
    let mut coulomb = 0.0;
    for (j, charge1) in gaussian_charges.iter().enumerate() {
        for charge2 in gaussian_charges.iter().take(j) {
            let dist = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&charge2.position, &charge1.position));
            let zeta = charge1.exp * charge2.exp / (charge1.exp + charge2.exp);
            coulomb += charge1.charge * charge2.charge * math_utilities::special_functions::erf(zeta.sqrt() * dist) / dist;
        }
    }
    coulomb
}

/// The Coulomb energy between Gaussian nuclei and point charges: q_n q erf(sqrt(e_n) r)/r.
pub(crate) fn compute_gaussian_coulomb_interaction<T: PointChargeTrait>(gaussian_charges: &[IndexedGaussianCharge], point_charges: &[T]) -> f64 {
    let mut coulomb = 0.0;
    for nucleus in gaussian_charges.iter() {
        for point_charge in point_charges.iter() {
            let dist = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&nucleus.position, point_charge.get_position()));
            coulomb += nucleus.charge * point_charge.get_charge() * math_utilities::special_functions::erf(nucleus.exp.sqrt() * dist) / dist;
        }
    }
    coulomb
}

/// The interaction energy of Gaussian nuclei with screened charges (AIMPs):
/// q_n q_a (erf(sqrt(e_n) r) + s_a(r))/r, the screening correction s_a evaluated at the nuclear position.
pub(crate) fn compute_gaussian_screened_coulomb_interaction<T: crate::charges::ScreenedChargeTrait>(gaussian_charges: &[IndexedGaussianCharge], screened_point_charges: &[T]) -> f64 {
    let mut coulomb = 0.0;
    for nucleus in gaussian_charges.iter() {
        for potential in screened_point_charges.iter() {
            let dist = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&potential.get_position(), &nucleus.position));
            let mut contr = math_utilities::special_functions::erf(nucleus.exp.sqrt() * dist);
            let screen = potential.get_screened_coulomb_potential();
            for (c, coeff) in screen.get_exponents().iter().zip(screen.get_coefficients().iter()) {
                contr += coeff * (-c * dist * dist).exp();
            }
            coulomb += nucleus.charge * potential.get_charge() * contr / dist;
        }
    }
    coulomb
}

/// The interaction energy of point charges with screened charges (AIMPs), each point charge in their full
/// potential: q_p q_a (1 + s_a(r))/r.
pub fn compute_screened_coulomb_interaction<T1: PointChargeTrait, T2: crate::charges::ScreenedChargeTrait>(point_charges: &[T1], screened_point_charges: &[T2]) -> f64 {
    let mut classical_coulomb = 0.0;
    for point_charge in point_charges.iter() {
        for potential in screened_point_charges.iter() {
            let dist = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&potential.get_position(), &point_charge.get_position()));
            let mut contr = 1.0;
            let screen = potential.get_screened_coulomb_potential();
            for (c, coeff) in screen.get_exponents().iter().zip(screen.get_coefficients().iter()) {
                contr += coeff * (-c * dist * dist).exp();
            }
            classical_coulomb += point_charge.get_charge() * potential.get_charge() * contr / dist;
        }
    }
    classical_coulomb
}

/// The potential of point multipole sources in a radial kernel g(r) (1/r, erf(kr)/r, an AIMP's
/// (1 + s(r))/r): at d = field_point - source position, with M_k the bare Cartesian moments,
///   phi = Sum_k (-1)^|k| / k! M_k D^k g(d),      k! = lx! ly! lz!,
/// the derivatives D^k g from `math_utilities::radial`. Holds the derivative tables, one per source order.
pub(crate) struct PotentialTables {
    by_order: Vec<math_utilities::radial::Derivatives>,
}

impl PotentialTables {
    /// An empty scratch; the tables are allocated as sources need them.
    pub(crate) fn new() -> Self {
        Self { by_order: Vec::new() }
    }

    /// The potential of `source` at `field_point` in `kernel`, per unit probe charge.
    fn potential<T: MultipoleSourceTrait>(
        &mut self,
        source: &T,
        field_point: &[f64; 3],
        kernel: &impl math_utilities::radial::RadialKernel,
    ) -> f64 {
        let order = source.get_max_order();
        while self.by_order.len() <= order {
            self.by_order.push(math_utilities::radial::Derivatives::new(self.by_order.len()));
        }
        let d = arrays::vector_math::vector_subtract_3d(field_point, source.get_position());
        let derivatives = &mut self.by_order[order];
        derivatives.compute(kernel, d);
        let factorial = |n: usize| -> f64 { (1..=n).map(|k| k as f64).product() };
        let mut potential = 0.0;
        for k in 0..=order {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            math_utilities::cartesian::for_each_cartesian_component(k, |lx, ly, lz| {
                potential += sign * source.get_moment(lx, ly, lz) / (factorial(lx) * factorial(ly) * factorial(lz)) * derivatives.get(lx, ly, lz);
            });
        }
        potential
    }

    /// The gradient of `potential` with respect to `field_point`: every term differentiated once more,
    /// `d/dd_x D^k g = D^(k + e_x) g`, so the derivatives run to one order above the source.
    pub(crate) fn potential_gradient<T: MultipoleSourceTrait>(
        &mut self,
        source: &T,
        field_point: &[f64; 3],
        kernel: &impl math_utilities::radial::RadialKernel,
    ) -> [f64; 3] {
        let order = source.get_max_order() + 1;
        while self.by_order.len() <= order {
            self.by_order.push(math_utilities::radial::Derivatives::new(self.by_order.len()));
        }
        let d = arrays::vector_math::vector_subtract_3d(field_point, source.get_position());
        let derivatives = &mut self.by_order[order];
        derivatives.compute(kernel, d);
        let factorial = |n: usize| -> f64 { (1..=n).map(|k| k as f64).product() };
        let mut gradient = [0.0; 3];
        for k in 0..order {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            math_utilities::cartesian::for_each_cartesian_component(k, |lx, ly, lz| {
                let weight = sign * source.get_moment(lx, ly, lz) / (factorial(lx) * factorial(ly) * factorial(lz));
                gradient[0] += weight * derivatives.get(lx + 1, ly, lz);
                gradient[1] += weight * derivatives.get(lx, ly + 1, lz);
                gradient[2] += weight * derivatives.get(lx, ly, lz + 1);
            });
        }
        gradient
    }
}

/// The interaction energy of point charges with point multipole sources.
pub(crate) fn compute_general_multipole_interaction<T1: PointChargeTrait, T2: MultipoleSourceTrait>(point_charges: &[T1], sources: &[T2]) -> f64 {
    let mut scratch = PotentialTables::new();
    let mut energy = 0.0;
    for probe in point_charges.iter() {
        for source in sources.iter() {
            energy += probe.get_charge() * scratch.potential(source, probe.get_position(), &math_utilities::radial::Coulomb);
        }
    }
    energy
}

/// The interaction energy of Gaussian nuclei with point multipole sources, each nucleus seeing the sources
/// through erf(sqrt(e_n) r)/r.
pub(crate) fn compute_gaussian_general_multipole_interaction<T: MultipoleSourceTrait>(gaussian_charges: &[IndexedGaussianCharge], sources: &[T]) -> f64 {
    let mut scratch = PotentialTables::new();
    let mut energy = 0.0;
    for nucleus in gaussian_charges.iter() {
        // The Visscher-Dyall potential felt by a point charge: erf(sqrt(exponent) r)/r.
        let kernel = math_utilities::radial::ErfCoulomb { kappa: nucleus.exp.sqrt() };
        for source in sources.iter() {
            energy += nucleus.charge * scratch.potential(source, &nucleus.position, &kernel);
        }
    }
    energy
}

/// The interaction energy of point multipole sources with screened charges (AIMPs), each screened charge
/// seeing the sources through its own (1 + s(r))/r.
pub fn compute_general_multipole_screened_interaction<T1: MultipoleSourceTrait, T2: crate::charges::ScreenedChargeTrait>(
    sources: &[T1],
    screened_point_charges: &[T2],
) -> f64 {
    let mut scratch = PotentialTables::new();
    let mut energy = 0.0;
    for potential in screened_point_charges.iter() {
        let screen = potential.get_screened_coulomb_potential();
        let kernel = math_utilities::radial::ScreenedCoulomb { exponents: screen.get_exponents(), coefficients: screen.get_coefficients() };
        for source in sources.iter() {
            energy += potential.get_charge() * scratch.potential(source, potential.get_position(), &kernel);
        }
    }
    energy
}

/// The interaction energy of a set of screened charges (AIMPs) among themselves, each pair once: each one's
/// charge in the other's full potential, the bare repulsion counted once,
///   Z_1 Z_2 (1 + s_1(r) + s_2(r)) / r,     s = Sum_i A_i exp(-c_i r^2).
pub fn compute_screened_coulomb_self_interaction<T: crate::charges::ScreenedChargeTrait>(screened_point_charges: &[T]) -> f64 {
    let mut classical_coulomb = 0.0;
    for (j, potential1) in screened_point_charges.iter().enumerate() {
        for potential2 in screened_point_charges.iter().take(j) {
            let dist = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&potential2.get_position(), &potential1.get_position()));
            let mut screen1_contr = 0.0;
            let screen1 = potential1.get_screened_coulomb_potential();
            for (c1, coeff1) in screen1.get_exponents().iter().zip(screen1.get_coefficients().iter()) {
                screen1_contr += coeff1 * (-c1 * dist * dist).exp();
            }
            let mut screen2_contr = 0.0;
            let screen2 = potential2.get_screened_coulomb_potential();
            for (c2, coeff2) in screen2.get_exponents().iter().zip(screen2.get_coefficients().iter()) {
                screen2_contr += coeff2 * (-c2 * dist * dist).exp();
            }
            let charge1 = potential1.get_charge();
            let charge2 = potential2.get_charge();
            classical_coulomb += charge1 * charge2 * (1.0 + screen1_contr + screen2_contr) / dist;
        }
    }
    classical_coulomb
}

/// The interaction energy between two sets of screened charges (AIMPs), with the pair term of
/// `compute_screened_coulomb_self_interaction`.
pub fn compute_screened_coulomb_cross_interaction<T1: crate::charges::ScreenedChargeTrait, T2: crate::charges::ScreenedChargeTrait>(screened_point_charges1: &[T1], screened_point_charges2: &[T2]) -> f64 {
    let mut classical_coulomb = 0.0;
    for potential1 in screened_point_charges1.iter() {
        for potential2 in screened_point_charges2.iter() {
            let dist = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&potential2.get_position(), &potential1.get_position()));
            let mut screen1_contr = 0.0;
            let screen1 = potential1.get_screened_coulomb_potential();
            for (c1, coeff1) in screen1.get_exponents().iter().zip(screen1.get_coefficients().iter()) {
                screen1_contr += coeff1 * (-c1 * dist * dist).exp();
            }
            let mut screen2_contr = 0.0;
            let screen2 = potential2.get_screened_coulomb_potential();
            for (c2, coeff2) in screen2.get_exponents().iter().zip(screen2.get_coefficients().iter()) {
                screen2_contr += coeff2 * (-c2 * dist * dist).exp();
            }
            let charge1 = potential1.get_charge();
            let charge2 = potential2.get_charge();
            classical_coulomb += charge1 * charge2 * (1.0 + screen1_contr + screen2_contr) / dist;
        }
    }
    classical_coulomb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IndexedGaussianCharge, IndexedPointCharge, NuclearPotential, PointMultipole, ScreenedChargeTrait, ScreenedCoulombPotential, SiteTrait};

    // A point multipole's classical interaction energy against that of the point charges of a
    // finite-difference stencil realising the same multipole.
    fn point_charge(x: f64, y: f64, z: f64, charge: f64) -> IndexedPointCharge {
        IndexedPointCharge::new(0, [x, y, z], charge)
    }

    #[test]
    fn plain_dipole_energy_matches_finite_difference_point_charges() {
        let probe = NuclearPotential::PointCharge(Box::new([IndexedPointCharge::new(0, [1.0, -2.0, 0.5], 3.0)]));

        let mu = 0.83;
        let source_pos = [4.0, 4.0, 4.0];
        let dipole = PointMultipole::dipole(source_pos, 0.0, [mu, 0.0, 0.0]);
        let e_dipole = probe.compute_general_multipole_interaction_energy(&[dipole]);

        let eps = 1e-4;
        let q = mu / eps;
        let fd_charges = [
            point_charge(source_pos[0] + eps / 2.0, source_pos[1], source_pos[2], q),
            point_charge(source_pos[0] - eps / 2.0, source_pos[1], source_pos[2], -q),
        ];
        let e_fd = probe.compute_nuclear_interaction_energy(&fd_charges);

        assert!((e_dipole - e_fd).abs() < 1e-2 * e_dipole.abs(), "e_dipole={e_dipole:.6e} e_fd={e_fd:.6e}");
    }

    #[test]
    fn plain_quadrupole_off_diagonal_energy_matches_finite_difference_point_charges() {
        let probe = NuclearPotential::PointCharge(Box::new([IndexedPointCharge::new(0, [1.0, -2.0, 0.5], 3.0)]));

        let raw_xz = 0.3;
        let source_pos = [4.0, 4.0, 4.0];
        // Order xx xy xz yy yz zz, unscaled.
        let quad = PointMultipole::quadrupole(source_pos, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, raw_xz, 0.0, 0.0, 0.0]);
        let e_quad = probe.compute_general_multipole_interaction_energy(&[quad]);

        let eps = 1e-3;
        let q = raw_xz / (4.0 * eps * eps);
        let fd_charges = [
            point_charge(source_pos[0] + eps, source_pos[1], source_pos[2] + eps, q),
            point_charge(source_pos[0] + eps, source_pos[1], source_pos[2] - eps, -q),
            point_charge(source_pos[0] - eps, source_pos[1], source_pos[2] + eps, -q),
            point_charge(source_pos[0] - eps, source_pos[1], source_pos[2] - eps, q),
        ];
        let e_fd = probe.compute_nuclear_interaction_energy(&fd_charges);

        assert!((e_quad - e_fd).abs() < 5e-2 * e_quad.abs(), "e_quad={e_quad:.6e} e_fd={e_fd:.6e}");
    }

    #[test]
    fn erf_screened_dipole_energy_matches_finite_difference_point_charges() {
        let nucleus = NuclearPotential::GaussianCharge(Box::new([IndexedGaussianCharge::new(0, [1.0, -2.0, 0.5], 3.0, 0.9)]));

        let mu = 0.83;
        let source_pos = [4.0, 4.0, 4.0];
        let dipole = PointMultipole::dipole(source_pos, 0.0, [mu, 0.0, 0.0]);
        let e_dipole = nucleus.compute_general_multipole_interaction_energy(&[dipole]);

        let eps = 1e-4;
        let q = mu / eps;
        let fd_charges = [
            point_charge(source_pos[0] + eps / 2.0, source_pos[1], source_pos[2], q),
            point_charge(source_pos[0] - eps / 2.0, source_pos[1], source_pos[2], -q),
        ];
        let e_fd = nucleus.compute_nuclear_interaction_energy(&fd_charges);

        assert!((e_dipole - e_fd).abs() < 1e-2 * e_dipole.abs(), "e_dipole={e_dipole:.6e} e_fd={e_fd:.6e}");
    }

    // Order 3, for every kernel: an octupole component M_abc = Sum q a b c is realised by a
    // finite-difference charge stencil whose only nonzero moment below order 5 is that one, and the
    // energy of the single octupole source is compared against the energy of the stencil's charges.

    const SOURCE: [f64; 3] = [4.0, 4.0, 4.0];
    const H: f64 = 0.02;

    // `component` is the position in the file's order-3 block: xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz.
    fn octupole(component: usize) -> PointMultipole {
        let mut file_values = vec![0.0; 20];
        file_values[10 + component] = 1.0;
        PointMultipole::from_xfield_moments(SOURCE, 3, &file_values)
    }

    // (component in the file's order, stencil charges) for zzz, xxy and xyz.
    fn octupole_stencils() -> Vec<(usize, Vec<IndexedPointCharge>)> {
        let at = |dx: f64, dy: f64, dz: f64, q: f64| point_charge(SOURCE[0] + dx * H, SOURCE[1] + dy * H, SOURCE[2] + dz * H, q);
        let zzz = [(-1.5, -1.0), (-0.5, 3.0), (0.5, -3.0), (1.5, 1.0)].iter().map(|&(z, w)| at(0.0, 0.0, z, w / (6.0 * H * H * H))).collect();
        let mut xxy = Vec::new();
        for &(x, a) in &[(-1.0, 1.0), (0.0, -2.0), (1.0, 1.0)] {
            for &(y, b) in &[(-0.5, -1.0), (0.5, 1.0)] {
                xxy.push(at(x, y, 0.0, a * b / (2.0 * H * H * H)));
            }
        }
        let mut xyz = Vec::new();
        for &sx in &[-1.0, 1.0] {
            for &sy in &[-1.0, 1.0] {
                for &sz in &[-1.0, 1.0] {
                    xyz.push(at(sx / 2.0, sy / 2.0, sz / 2.0, sx * sy * sz / (H * H * H)));
                }
            }
        }
        vec![(9, zzz), (1, xxy), (4, xyz)]
    }

    #[test]
    fn plain_octupole_energy_matches_finite_difference_point_charges() {
        let probe = NuclearPotential::PointCharge(Box::new([IndexedPointCharge::new(0, [1.0, -2.0, 0.5], 3.0)]));
        for (component, stencil) in octupole_stencils() {
            let e_octupole = probe.compute_general_multipole_interaction_energy(&[octupole(component)]);
            let e_fd = probe.compute_nuclear_interaction_energy(&stencil);
            assert!(e_octupole.abs() > 1e-8);
            assert!((e_octupole - e_fd).abs() < 1e-3 * e_octupole.abs(), "component {component}: e_octupole={e_octupole:.6e} e_fd={e_fd:.6e}");
        }
    }

    #[test]
    fn erf_screened_octupole_energy_matches_finite_difference_point_charges() {
        let nucleus = NuclearPotential::GaussianCharge(Box::new([IndexedGaussianCharge::new(0, [1.0, -2.0, 0.5], 3.0, 0.9)]));
        for (component, stencil) in octupole_stencils() {
            let e_octupole = nucleus.compute_general_multipole_interaction_energy(&[octupole(component)]);
            let e_fd = nucleus.compute_nuclear_interaction_energy(&stencil);
            assert!(e_octupole.abs() > 1e-8);
            assert!((e_octupole - e_fd).abs() < 1e-3 * e_octupole.abs(), "component {component}: e_octupole={e_octupole:.6e} e_fd={e_fd:.6e}");
        }
    }

    struct FakeAimp {
        position: [f64; 3],
        charge: f64,
        screen: ScreenedCoulombPotential,
    }

    impl SiteTrait for FakeAimp {
        fn get_position(&self) -> &[f64; 3] {
            &self.position
        }
    }
    impl PointChargeTrait for FakeAimp {
        fn get_charge(&self) -> f64 {
            self.charge
        }
    }
    impl ScreenedChargeTrait for FakeAimp {
        fn get_screened_coulomb_potential(&self) -> &ScreenedCoulombPotential {
            &self.screen
        }
    }

    // Two AIMPs interact as each one's charge in the other's full potential, the bare repulsion once:
    // E_12 = E(2 as a point charge in 1) + E(1 as a point charge in 2) - Z_1 Z_2 / r. Non-unit charges, so a
    // screen scaled by the wrong charge shows.
    #[test]
    fn aimp_aimp_energy_is_each_charge_in_the_others_potential() {
        let a = FakeAimp { position: [0.3, -0.2, 0.1], charge: 3.0, screen: ScreenedCoulombPotential::new(vec![0.9, 4.2], vec![1.3, -0.4]) };
        let b = FakeAimp { position: [1.1, 0.8, -0.9], charge: -2.0, screen: ScreenedCoulombPotential::new(vec![0.6, 2.5], vec![-2.1, 0.7]) };
        let point = |aimp: &FakeAimp| IndexedPointCharge::new(0, aimp.position, aimp.charge);
        let r = arrays::vector_math::vector_norm_3d(&arrays::vector_math::vector_subtract_3d(&a.position, &b.position));
        let expected = compute_screened_coulomb_interaction(&[point(&b)], std::slice::from_ref(&a))
            + compute_screened_coulomb_interaction(&[point(&a)], std::slice::from_ref(&b))
            - a.charge * b.charge / r;
        let cross = compute_screened_coulomb_cross_interaction(std::slice::from_ref(&a), std::slice::from_ref(&b));
        assert!((cross - expected).abs() < 1e-12 * expected.abs(), "cross {cross} vs {expected}");
        let pair = [FakeAimp { position: a.position, charge: a.charge, screen: ScreenedCoulombPotential::new(vec![0.9, 4.2], vec![1.3, -0.4]) }, b];
        let own = compute_screened_coulomb_self_interaction(&pair);
        assert!((own - expected).abs() < 1e-12 * expected.abs(), "self {own} vs {expected}");
    }

    #[test]
    fn aimp_screened_octupole_energy_matches_finite_difference_point_charges() {
        // The AIMP probe sits close enough that its screening terms are far from negligible.
        let aimp = FakeAimp { position: [2.5, 3.0, 3.2], charge: 2.0, screen: ScreenedCoulombPotential::new(vec![0.9, 4.2], vec![1.3, -0.4]) };
        for (component, stencil) in octupole_stencils() {
            let e_octupole = compute_general_multipole_screened_interaction(&[octupole(component)], std::slice::from_ref(&aimp));
            let e_fd = compute_screened_coulomb_interaction(&stencil, std::slice::from_ref(&aimp));
            assert!(e_octupole.abs() > 1e-8);
            assert!((e_octupole - e_fd).abs() < 1e-3 * e_octupole.abs(), "component {component}: e_octupole={e_octupole:.6e} e_fd={e_fd:.6e}");
        }
    }
}
