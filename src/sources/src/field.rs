// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The classical electric field at a set of sites, from point charges, finite (Gaussian) nuclei, AIMPs and
//! point multipoles of any order. Each function adds the field E = -grad(phi) of its sources to `field`, which
//! holds (x, y, z) per site in the order of `sites`.
//!
//! Pairs for which `exclude(site_index, source_index)` is true are not evaluated. A site that coincides with a
//! source it is not excluded from is a caller error (the field is singular there, except for a finite nucleus).

use math_utilities::cartesian::for_each_cartesian_component;
use math_utilities::radial::{Coulomb, Derivatives, ErfCoulomb, RadialKernel, ScreenedCoulomb};
use crate::{FiniteChargeTrait, MultipoleSourceTrait, PointChargeTrait, ScreenedChargeTrait, SiteTrait};

/// No pair is excluded.
pub fn no_exclusions(_site: usize, _source: usize) -> bool {
    false
}

fn check_field_length<S>(sites: &[S], field: &[f64]) {
    assert_eq!(field.len(), 3 * sites.len(), "the field holds (x, y, z) for each of the {} sites", sites.len());
}

fn separation(site: &impl SiteTrait, source: &impl SiteTrait) -> [f64; 3] {
    let (a, b) = (site.get_position(), source.get_position());
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

// field += -q d (sum of B_1 of the kernels), i.e. the field of q g(r) with grad g = d B_1(r).
fn add_radial_field(field: &mut [f64], d: [f64; 3], scale: f64) {
    for axis in 0..3 {
        field[axis] -= scale * d[axis];
    }
}

/// The field of point charges (plain 1/r).
pub fn add_field_from_charges<S: SiteTrait, C: PointChargeTrait>(
    sites: &[S],
    charges: &[C],
    exclude: impl Fn(usize, usize) -> bool,
    field: &mut [f64],
) {
    check_field_length(sites, field);
    for (i, site) in sites.iter().enumerate() {
        for (j, charge) in charges.iter().enumerate() {
            if exclude(i, j) {
                continue;
            }
            let d = separation(site, charge);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            add_radial_field(&mut field[3 * i..3 * i + 3], d, charge.get_charge() * Coulomb.b(1, r));
        }
    }
}

/// The field of finite charges: contracted series of spherical Gaussian charge distributions (a Visscher-Dyall nucleus
/// has one term), whose potential is Sum_i q_i erf(sqrt(e_i) r)/r.
pub fn add_field_from_finite_charges<S: SiteTrait, C: FiniteChargeTrait>(
    sites: &[S],
    charges: &[C],
    exclude: impl Fn(usize, usize) -> bool,
    field: &mut [f64],
) {
    check_field_length(sites, field);
    for (i, site) in sites.iter().enumerate() {
        for (j, charge) in charges.iter().enumerate() {
            if exclude(i, j) {
                continue;
            }
            let d = separation(site, charge);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let b1: f64 = charge.get_exponents().iter().zip(charge.get_charges()).map(|(e, q)| q * ErfCoulomb { kappa: e.sqrt() }.b(1, r)).sum();
            add_radial_field(&mut field[3 * i..3 * i + 3], d, b1);
        }
    }
}

/// The full field of AIMPs: the point charge plus the screening correction, i.e. the potential
/// q (1 + Sum_i A_i exp(-c_i r^2))/r of each AIMP (the same one `compute_screened_coulomb_interaction`
/// uses for the energy).
pub fn add_field_from_aimps<S: SiteTrait, A: ScreenedChargeTrait>(
    sites: &[S],
    aimps: &[A],
    exclude: impl Fn(usize, usize) -> bool,
    field: &mut [f64],
) {
    check_field_length(sites, field);
    for (i, site) in sites.iter().enumerate() {
        for (j, aimp) in aimps.iter().enumerate() {
            if exclude(i, j) {
                continue;
            }
            let d = separation(site, aimp);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let screen = aimp.get_screened_coulomb_potential();
            let kernel = ScreenedCoulomb { exponents: screen.get_exponents(), coefficients: screen.get_coefficients() };
            add_radial_field(&mut field[3 * i..3 * i + 3], d, aimp.get_charge() * kernel.b(1, r));
        }
    }
}

/// The field of point multipoles of any order (charge, dipole, quadrupole, ...). With the moments
/// M_k (bare Cartesian moments) about the source, the potential at d = site - source is
/// phi = Sum_k (-1)^|k| / k! M_k D^k (1/r), so E_e = -Sum_k (-1)^|k| / k! M_k D^(k + e) (1/r): a
/// source of order n needs derivatives up to order n + 1.
pub fn add_field_from_multipoles<S: SiteTrait, M: MultipoleSourceTrait>(
    sites: &[S],
    sources: &[M],
    exclude: impl Fn(usize, usize) -> bool,
    field: &mut [f64],
) {
    check_field_length(sites, field);
    // One scratch per order: sources of different orders share nothing but the loop.
    let max_order = sources.iter().map(|source| source.get_max_order()).max().unwrap_or(0);
    let mut derivatives: Vec<Derivatives> = (0..=max_order).map(|order| Derivatives::new(order + 1)).collect();
    let factorial = |n: usize| -> f64 { (1..=n).map(|k| k as f64).product() };

    for (i, site) in sites.iter().enumerate() {
        for (j, source) in sources.iter().enumerate() {
            if exclude(i, j) {
                continue;
            }
            let order = source.get_max_order();
            let d = separation(site, source);
            let table = &mut derivatives[order];
            table.compute(&Coulomb, d);
            let mut e = [0.0; 3];
            for k in 0..=order {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                for_each_cartesian_component(k, |lx, ly, lz| {
                    let moment = source.get_moment(lx, ly, lz);
                    if moment == 0.0 {
                        return;
                    }
                    let weight = sign * moment / (factorial(lx) * factorial(ly) * factorial(lz));
                    e[0] -= weight * table.get(lx + 1, ly, lz);
                    e[1] -= weight * table.get(lx, ly + 1, lz);
                    e[2] -= weight * table.get(lx, ly, lz + 1);
                });
            }
            for axis in 0..3 {
                field[3 * i + axis] += e[axis];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IndexedGaussianCharge, IndexedPointCharge, PointCharge, PointMultipole, ScreenedCoulombPotential, Site};

    const SITES: [[f64; 3]; 3] = [[0.3, -0.4, 0.9], [-1.2, 0.5, 0.1], [2.0, 1.0, -0.7]];

    fn sites() -> Vec<Site> {
        SITES.iter().map(|&p| Site::new(p)).collect()
    }

    // Central-difference -grad of a scalar potential, as an independent reference.
    fn negative_gradient(potential: impl Fn([f64; 3]) -> f64, at: [f64; 3]) -> [f64; 3] {
        let h = 1e-5;
        let mut gradient = [0.0; 3];
        for axis in 0..3 {
            let (mut plus, mut minus) = (at, at);
            plus[axis] += h;
            minus[axis] -= h;
            gradient[axis] = -(potential(plus) - potential(minus)) / (2.0 * h);
        }
        gradient
    }

    fn assert_field_close(field: &[f64], expected: &[[f64; 3]], tolerance: f64) {
        for (i, expected) in expected.iter().enumerate() {
            for axis in 0..3 {
                assert!((field[3 * i + axis] - expected[axis]).abs() < tolerance, "site {i}, axis {axis}: {} vs {}", field[3 * i + axis], expected[axis]);
            }
        }
    }

    fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[test]
    fn point_charges_give_the_coulomb_field() {
        let charges = [PointCharge::new([1.0, 1.0, 1.0], 2.0), PointCharge::new([-1.0, 0.0, 0.5], -0.7)];
        let mut field = vec![0.0; 9];
        add_field_from_charges(&sites(), &charges, no_exclusions, &mut field);
        let expected: Vec<[f64; 3]> = SITES
            .iter()
            .map(|&site| negative_gradient(|p| charges.iter().map(|c| c.get_charge() / distance(p, *c.get_position())).sum(), site))
            .collect();
        assert_field_close(&field, &expected, 1e-8);
    }

    #[test]
    fn contributions_accumulate_and_excluded_pairs_are_skipped() {
        let charges = [PointCharge::new(SITES[0], 5.0), PointCharge::new([1.0, 1.0, 1.0], 2.0)];
        let mut field = vec![0.0; 9];
        // Site 0 coincides with charge 0 and is excluded from it; adding the same charges twice doubles the field.
        let exclude = |site: usize, source: usize| site == 0 && source == 0;
        add_field_from_charges(&sites(), &charges, exclude, &mut field);
        let once = field.clone();
        add_field_from_charges(&sites(), &charges, exclude, &mut field);
        assert!(once.iter().all(|x| x.is_finite()));
        for (a, b) in field.iter().zip(&once) {
            assert!((a - 2.0 * b).abs() < 1e-12);
        }
    }

    #[test]
    fn finite_nuclei_give_the_erf_screened_field_and_are_regular_at_the_centre() {
        let nuclei = [IndexedGaussianCharge::new(0, [0.5, 0.5, 0.0], 3.0, 1.9), IndexedGaussianCharge::new(1, [-1.0, 0.2, 0.3], 1.0, 0.6)];
        let mut field = vec![0.0; 9];
        add_field_from_finite_charges(&sites(), &nuclei, no_exclusions, &mut field);
        let potential = |p: [f64; 3]| -> f64 {
            nuclei
                .iter()
                .map(|n| n.get_charge() * math_utilities::special_functions::erf(n.get_exponents()[0].sqrt() * distance(p, *n.get_position())) / distance(p, *n.get_position()))
                .sum()
        };
        let expected: Vec<[f64; 3]> = SITES.iter().map(|&site| negative_gradient(potential, site)).collect();
        assert_field_close(&field, &expected, 1e-7);

        // At the nucleus itself the field of a Gaussian charge is finite (zero, by symmetry).
        let at_centre = [Site::new([0.5, 0.5, 0.0])];
        let mut centre_field = vec![0.0; 3];
        add_field_from_finite_charges(&at_centre, &nuclei[..1], no_exclusions, &mut centre_field);
        assert!(centre_field.iter().all(|x| x.abs() < 1e-12), "{centre_field:?}");
    }

    /// A contracted series of Gaussian charges has the field of its terms as separate charges.
    #[test]
    fn a_contracted_series_has_the_field_of_its_terms() {
        let (position, exponents, charges) = ([0.5, 0.5, 0.0], [0.4, 2.0, 30.0], [-1.5, 2.25, -3.0]);
        let series = [crate::IndexedContractedGaussianCharge::new(0, position, exponents.into(), charges.into())];
        let terms: Vec<IndexedGaussianCharge> = exponents.iter().zip(charges).map(|(&e, q)| IndexedGaussianCharge::new(0, position, q, e)).collect();
        let (mut field, mut expected) = (vec![0.0; 9], vec![0.0; 9]);
        add_field_from_finite_charges(&sites(), &series, no_exclusions, &mut field);
        add_field_from_finite_charges(&sites(), &terms, no_exclusions, &mut expected);
        assert!(expected.iter().any(|x| x.abs() > 1e-2));
        for (a, b) in field.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        }
        assert_eq!(series[0].get_charge(), -2.25);
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

    #[test]
    fn aimps_give_the_point_charge_plus_screen_field() {
        let aimps = [
            FakeAimp { position: [0.4, 0.4, 0.4], charge: 2.0, screen: ScreenedCoulombPotential::new(vec![0.9, 4.2], vec![1.3, -0.4]) },
            FakeAimp { position: [-1.5, 0.0, 1.0], charge: 1.0, screen: ScreenedCoulombPotential::new(vec![2.5], vec![0.8]) },
        ];
        let mut field = vec![0.0; 9];
        add_field_from_aimps(&sites(), &aimps, no_exclusions, &mut field);
        // The potential exactly as the classical AIMP interaction energy uses it: q (1 + Sum A exp(-c r^2)) / r.
        let potential = |p: [f64; 3]| -> f64 {
            aimps
                .iter()
                .map(|a| {
                    let r = distance(p, a.position);
                    let screen = a.screen.get_exponents().iter().zip(a.screen.get_coefficients()).map(|(c, w)| w * (-c * r * r).exp()).sum::<f64>();
                    a.charge * (1.0 + screen) / r
                })
                .sum()
        };
        let expected: Vec<[f64; 3]> = SITES.iter().map(|&site| negative_gradient(potential, site)).collect();
        assert_field_close(&field, &expected, 1e-7);
    }

    // A point multipole's field, against the field of the point charges whose finite-difference
    // stencil realises it (charges at offsets around the source, so that only one moment below
    // order 5 is nonzero).
    fn stencil_field(source: [f64; 3], charges: &[([f64; 3], f64)], site: [f64; 3]) -> [f64; 3] {
        let point_charges: Vec<PointCharge> = charges.iter().map(|&(offset, q)| PointCharge::new([source[0] + offset[0], source[1] + offset[1], source[2] + offset[2]], q)).collect();
        let mut field = vec![0.0; 3];
        add_field_from_charges(&[Site::new(site)], &point_charges, no_exclusions, &mut field);
        [field[0], field[1], field[2]]
    }

    fn multipole_field(source: &PointMultipole, site: [f64; 3]) -> [f64; 3] {
        let mut field = vec![0.0; 3];
        add_field_from_multipoles(&[Site::new(site)], std::slice::from_ref(source), no_exclusions, &mut field);
        [field[0], field[1], field[2]]
    }

    fn assert_vectors_close(a: [f64; 3], b: [f64; 3], relative: f64) {
        let scale = b.iter().fold(0.0_f64, |w, x| w.max(x.abs()));
        for axis in 0..3 {
            assert!((a[axis] - b[axis]).abs() < relative * scale, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn multipole_fields_match_point_charge_stencils_at_every_order() {
        let source = [0.2, -0.1, 0.4];
        let site = [3.1, 2.2, -2.5];
        let h = 0.02;

        // Dipole along y: +-Q at y = +-h/2, M_y = Q h.
        let q = 1.0 / h;
        let dipole = PointMultipole::dipole(source, 0.0, [0.0, 1.0, 0.0]);
        let stencil = [([0.0, h / 2.0, 0.0], q), ([0.0, -h / 2.0, 0.0], -q)];
        assert_vectors_close(multipole_field(&dipole, site), stencil_field(source, &stencil, site), 1e-3);

        // Quadrupole xz (off-diagonal): (+,-,-,+) at the corners of an x-z square, M_xz = Q h^2.
        let q = 1.0 / (h * h);
        let quadrupole = PointMultipole::quadrupole(source, 0.0, [0.0; 3], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let mut stencil = Vec::new();
        for &sx in &[-1.0, 1.0] {
            for &sz in &[-1.0, 1.0] {
                stencil.push(([sx * h / 2.0, 0.0, sz * h / 2.0], sx * sz * q));
            }
        }
        assert_vectors_close(multipole_field(&quadrupole, site), stencil_field(source, &stencil, site), 1e-3);

        // Octupoles: zzz (file position 9 of the order-3 block), xxy (1), xyz (4). Order 0..2 entries zero.
        let octupole = |component: usize| {
            let mut file_values = vec![0.0; 20];
            file_values[10 + component] = 1.0;
            PointMultipole::from_xfield_moments(source, 3, &file_values)
        };
        let zzz: Vec<([f64; 3], f64)> = [(-1.5, -1.0), (-0.5, 3.0), (0.5, -3.0), (1.5, 1.0)]
            .iter()
            .map(|&(z, w)| ([0.0, 0.0, z * h], w / (6.0 * h * h * h)))
            .collect();
        assert_vectors_close(multipole_field(&octupole(9), site), stencil_field(source, &zzz, site), 1e-3);

        let mut xxy = Vec::new();
        for &(x, a) in &[(-1.0, 1.0), (0.0, -2.0), (1.0, 1.0)] {
            for &(y, b) in &[(-0.5, -1.0), (0.5, 1.0)] {
                xxy.push(([x * h, y * h, 0.0], a * b / (2.0 * h * h * h)));
            }
        }
        assert_vectors_close(multipole_field(&octupole(1), site), stencil_field(source, &xxy, site), 1e-3);

        let mut xyz = Vec::new();
        for &sx in &[-1.0, 1.0] {
            for &sy in &[-1.0, 1.0] {
                for &sz in &[-1.0, 1.0] {
                    xyz.push(([sx * h / 2.0, sy * h / 2.0, sz * h / 2.0], sx * sy * sz / (h * h * h)));
                }
            }
        }
        assert_vectors_close(multipole_field(&octupole(4), site), stencil_field(source, &xyz, site), 1e-3);
    }

    #[test]
    fn a_charge_only_multipole_equals_a_point_charge_and_mixed_orders_coexist() {
        let charge = PointCharge::new([1.0, 1.0, 1.0], 1.7);
        let as_multipole = PointMultipole::new([1.0, 1.0, 1.0], 0, vec![1.7]);
        let mut from_charge = vec![0.0; 9];
        let mut from_multipole = vec![0.0; 9];
        add_field_from_charges(&sites(), &[charge], no_exclusions, &mut from_charge);
        add_field_from_multipoles(&sites(), &[as_multipole], no_exclusions, &mut from_multipole);
        for (a, b) in from_charge.iter().zip(&from_multipole) {
            assert!((a - b).abs() < 1e-13);
        }

        // Sources of different orders in one call (separate scratch buffers per order) sum correctly.
        let mixed = [PointMultipole::new([1.0, 1.0, 1.0], 0, vec![1.7]), PointMultipole::dipole([0.0, 2.0, 0.0], 0.5, [0.1, 0.2, 0.3])];
        let mut together = vec![0.0; 9];
        add_field_from_multipoles(&sites(), &mixed, no_exclusions, &mut together);
        let mut separately = vec![0.0; 9];
        for source in &mixed {
            add_field_from_multipoles(&sites(), std::slice::from_ref(source), no_exclusions, &mut separately);
        }
        for (a, b) in together.iter().zip(&separately) {
            assert!((a - b).abs() < 1e-13);
        }
    }

    #[test]
    fn an_excluded_coincident_multipole_is_skipped() {
        let source = PointMultipole::dipole(SITES[1], 1.0, [0.3, 0.0, 0.0]);
        let mut field = vec![0.0; 9];
        add_field_from_multipoles(&sites(), &[source], |site, _| site == 1, &mut field);
        assert!(field.iter().all(|x| x.is_finite()));
        assert_eq!(&field[3..6], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn indexed_point_charges_work_as_sources_too() {
        // The QM nuclei are IndexedPointCharge.
        let nuclei = [IndexedPointCharge::new(0, [0.0, 0.0, 0.0], 6.0)];
        let mut field = vec![0.0; 9];
        add_field_from_charges(&sites(), &nuclei, no_exclusions, &mut field);
        let expected: Vec<[f64; 3]> = SITES.iter().map(|&s| negative_gradient(|p| 6.0 / distance(p, [0.0; 3]), s)).collect();
        assert_field_close(&field, &expected, 1e-7);
    }
}
