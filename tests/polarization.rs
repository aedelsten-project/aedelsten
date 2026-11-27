// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The induced-dipole embedding at the context level: set_polarization / compute_induction /
// free_polarization, checked against an independent reconstruction of the whole pipeline (static field +
// electronic field, dense solve, contraction of the integrals with the dipoles) and against the defining
// property of the operator, V = dE_ind/dD.

mod common;

use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, Damping, PolarizationOptions, SiteChunking, SiteSelection, SolverOptions, StaticFieldPolicy, StaticFieldSources};
use polarization::PolarizableSystem;
use sources::{Fragments, PolarizableSite, Polarizability};

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/fock_builder_test.inp");

// Tests run in parallel: every file gets a name no other call can produce.
fn write_temp(tag: &str, contents: &str) -> std::path::PathBuf {
    let path = common::temp_path(&format!("pol_{tag}.xfield"));
    std::fs::write(&path, contents).unwrap();
    path
}

// (fragment, position, tensor in the xfield file's order xx xy xz yy yz zz). Sites 0 and 1 share fragment 1
// and do not couple; the rest do. Site 0 is isotropic, the others anisotropic.
const SITES: [(i32, [f64; 3], [f64; 6]); 4] = [
    (1, [6.0, 0.0, 0.0], [2.0, 0.0, 0.0, 2.0, 0.0, 2.0]),
    (1, [6.0, 3.0, 0.5], [3.0, 0.2, 0.1, 2.5, 0.15, 2.0]),
    (2, [-5.0, 4.0, 4.0], [1.5, -0.1, 0.05, 1.8, 0.2, 1.2]),
    (0, [0.0, -6.0, 3.0], [2.2, 0.3, 0.0, 2.0, -0.1, 1.6]),
];

fn polarizable_file() -> String {
    let mut contents = format!("{} Bohr -1 2 1\n", SITES.len());
    for (fragment, p, t) in SITES {
        contents.push_str(&format!("{fragment} {} {} {} {} {} {} {} {} {}\n", p[0], p[1], p[2], t[0], t[1], t[2], t[3], t[4], t[5]));
    }
    contents
}

// The same sites as library objects, for the reference calculation (file order is the internal order).
fn reference_sites() -> Vec<PolarizableSite> {
    SITES
        .iter()
        .map(|&(fragment, p, components)| PolarizableSite::new(p, Polarizability::Anisotropic(components), Fragments::new(&[fragment])))
        .collect()
}

const EXTERNAL: &str = "2 Bohr 1 0 1\n1 8.0 8.0 -2.0 0.5 0.1 0.2 0.3\n2 -7.0 6.0 1.0 -0.4 0.0 0.2 0.0\n";

fn build_context(with_external: bool) -> (AedelstenContext, usize) {
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let pol = write_temp("sites", &polarizable_file());
    ctx.set_polarizable_sites(pol.to_str().unwrap()).unwrap();
    std::fs::remove_file(&pol).ok();
    if with_external {
        let ext = write_temp("external", EXTERNAL);
        ctx.set_external_multipoles(ext.to_str().unwrap()).unwrap();
        std::fs::remove_file(&ext).ok();
    }
    let n = ctx.get_n_cont();
    (ctx, n)
}

// A fixed symmetric density (Hilbert matrix), scaled to a moderate field.
fn density(n: usize) -> Vec<f64> {
    (0..n * n).map(|idx| 0.02 / (1.0 + (idx % n) as f64 + (idx / n) as f64)).collect()
}

fn tight() -> PolarizationOptions {
    PolarizationOptions {
        solver: SolverOptions { threshold: 1e-12, ..SolverOptions::default() },
        chunking: SiteChunking::Sites(3),
        ..PolarizationOptions::default()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// The reference: everything rebuilt from public pieces, with the dense solver.
fn reference(ctx: &AedelstenContext, n: usize, density: &[f64], damping: Damping) -> (f64, Vec<f64>, Matrix<f64>) {
    let sites = reference_sites();
    let count = sites.len();
    let mut field = vec![0.0; 3 * count];
    ctx.compute_static_field(SiteSelection::Polarizable, StaticFieldSources::ALL, &mut field).unwrap();
    let mut electronic = vec![0.0; 3 * count];
    ctx.compute_electronic_field(density, SiteSelection::Polarizable, &SiteChunking::Sites(2), &mut electronic).unwrap();
    let total: Vec<f64> = field.iter().zip(&electronic).map(|(a, b)| a + b).collect();

    let system = PolarizableSystem::new(&sites, damping).unwrap();
    let total_matrix = Matrix::<f64>::new_from_vec(3, count, total.clone().into_boxed_slice());
    let mut mu = Matrix::<f64>::new(3, count);
    system.solve_dense(&total_matrix, &mut mu).unwrap();
    let energy = system.induction_energy(&mu, &total_matrix).unwrap();

    // V = -Sum mu . plane, from the full integral tensor.
    let mut integrals = vec![0.0; n * n * 3 * count];
    ctx.compute_electric_field_integrals(SiteSelection::Polarizable, 0..count, &mut integrals).unwrap();
    let mut operator = vec![0.0; n * n];
    for (plane, &dipole) in integrals.chunks(n * n).zip(mu.get_data()) {
        for (v, x) in operator.iter_mut().zip(plane) {
            *v -= dipole * x;
        }
    }
    (energy, operator, mu)
}

#[test]
fn compute_induction_matches_an_independent_reconstruction() {
    for damping in [Damping::None, Damping::thole()] {
        let (mut ctx, n) = build_context(true);
        let density = density(n);
        let (expected_energy, expected_operator, expected_dipoles) = reference(&ctx, n, &density, damping);

        ctx.set_polarization(PolarizationOptions { damping, ..tight() }).unwrap();
        let mut operator = vec![f64::NAN; n * n];
        let report = ctx.compute_induction(&density, &mut operator).unwrap();

        assert!((report.energy - expected_energy).abs() < 1e-10 * expected_energy.abs().max(1e-3), "{damping:?}: {} vs {expected_energy}", report.energy);
        assert!(report.energy < 0.0, "induction lowers the energy");
        for (a, b) in operator.iter().zip(&expected_operator) {
            assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{damping:?}: operator {a} vs {b}");
        }
        assert!(operator.iter().any(|v| v.abs() > 1e-6), "the operator should not be trivially zero");
        // The model and the dipole solver's conjugate gradients are cited.
        for doi in ["10.1021/ct1003803", "10.6028/jres.049.044"] {
            assert!(instrumentation::citations().into_iter().any(|r| r.reference.doi() == Some(doi)), "{doi} is cited");
        }
        // The operator is symmetric.
        for i in 0..n {
            for j in 0..i {
                assert!((operator[i + j * n] - operator[j + i * n]).abs() < 1e-12);
            }
        }
        let mut dipoles = vec![0.0; 12];
        ctx.get_induced_dipoles(&mut dipoles).unwrap();
        for (a, b) in dipoles.iter().zip(expected_dipoles.get_data()) {
            assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{damping:?}: dipole {a} vs {b}");
        }
    }
}

#[test]
fn the_operator_is_the_derivative_of_the_induction_energy_with_respect_to_the_density() {
    // dE_ind/dD_pq = V_pq, checked by central finite differences along a symmetric direction. This pins the
    // sign, the factor (the 1/2 of E_ind cancelling against the two occurrences of F) and the transposition.
    let (mut ctx, n) = build_context(true);
    let d0 = density(n);
    ctx.set_polarization(PolarizationOptions { damping: Damping::thole(), ..tight() }).unwrap();

    let mut direction = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = ((3 * i + 5 * j) % 7) as f64 - 3.0;
            direction[i + j * n] = value;
            direction[j + i * n] = value;
        }
    }
    let energy_at = |ctx: &AedelstenContext, epsilon: f64| {
        let shifted: Vec<f64> = d0.iter().zip(&direction).map(|(d, x)| d + epsilon * x).collect();
        let mut operator = vec![0.0; n * n];
        ctx.compute_induction(&shifted, &mut operator).unwrap().energy
    };
    let mut operator = vec![0.0; n * n];
    ctx.compute_induction(&d0, &mut operator).unwrap();
    let analytic = dot(&operator, &direction);

    let epsilon = 1e-3;
    let numeric = (energy_at(&ctx, epsilon) - energy_at(&ctx, -epsilon)) / (2.0 * epsilon);
    assert!(analytic.abs() > 1e-8, "the directional derivative should not be trivially zero");
    assert!((analytic - numeric).abs() < 1e-6 * analytic.abs(), "analytic {analytic:.10e} vs finite difference {numeric:.10e}");
}

#[test]
fn the_static_field_policies_agree_and_differ_in_what_may_change() {
    let (mut cached, n) = build_context(true);
    let (mut recomputed, _) = build_context(true);
    let density = density(n);
    cached.set_polarization(PolarizationOptions { static_field: StaticFieldPolicy::Cache, ..tight() }).unwrap();
    recomputed.set_polarization(PolarizationOptions { static_field: StaticFieldPolicy::Recompute, ..tight() }).unwrap();

    let (mut a, mut b) = (vec![0.0; n * n], vec![0.0; n * n]);
    let report_a = cached.compute_induction(&density, &mut a).unwrap();
    let report_b = recomputed.compute_induction(&density, &mut b).unwrap();
    assert_eq!(report_a, report_b);
    assert_eq!(a, b);

    // Freeing the sources: a cached static field is a snapshot, so the state and its results survive it -- which is
    // what lets an SCF driver release the multipoles once the field is cached.
    let other = write_temp("other_external", "1 Bohr 0\n9.0 -3.0 2.0 1.0\n");
    let other_path = other.to_str().unwrap();
    cached.free_external_multipoles();
    assert!(cached.has_polarization());
    let mut after_free = vec![0.0; n * n];
    let report_after_free = cached.compute_induction(&density, &mut after_free).unwrap();
    // (Not bit-identical: the second call is warm-started, so the solver ends on an equally converged iterate.)
    assert!((report_after_free.energy - report_a.energy).abs() < 1e-12, "the cached field does not depend on the freed sources");
    for (x, y) in after_free.iter().zip(&a) {
        assert!((x - y).abs() < 1e-11);
    }
    // ... but *changing* the sources under a cached field is refused, so the snapshot cannot silently go stale.
    assert!(cached.set_external_multipoles(other_path).is_err());
    cached.free_polarization();
    assert!(cached.set_external_multipoles(other_path).is_ok());

    let mut fresh = build_context(false).0;
    fresh.set_polarization(PolarizationOptions { static_field: StaticFieldPolicy::Cache, ..tight() }).unwrap();
    assert!(fresh.set_external_multipoles(other_path).is_err(), "a cached static field must not go stale");
    fresh.free_polarization();
    assert!(fresh.set_external_multipoles(other_path).is_ok());

    // With Recompute the sources present at each call are used, so they may be replaced freely.
    recomputed.free_external_multipoles();
    assert!(recomputed.has_polarization(), "Recompute does not depend on a snapshot");
    recomputed.set_external_multipoles(other_path).unwrap();
    let mut c = vec![0.0; n * n];
    let report_c = recomputed.compute_induction(&density, &mut c).unwrap();
    assert_ne!(report_c.energy, report_b.energy, "the new source must change the induction energy");
    std::fs::remove_file(&other).ok();
}

#[test]
fn the_nuclear_model_cannot_change_under_a_cached_static_field() {
    let (mut ctx, _n) = build_context(false);
    ctx.set_polarization(PolarizationOptions { static_field: StaticFieldPolicy::Cache, ..tight() }).unwrap();
    assert!(ctx.set_finite_nuclear_model(aedelsten::FiniteModel::VisscherDyall).is_err());
    ctx.free_polarization();
    assert!(ctx.set_finite_nuclear_model(aedelsten::FiniteModel::VisscherDyall).is_ok());
}

/// The energy of loosely converged dipoles is the functional's, above the exact energy by exactly
/// 1/2 dmu . (alpha^-1 - T) dmu (second order in their error dmu), where -1/2 mu . F would be off at first order.
#[test]
fn the_reported_energy_is_second_order_in_the_dipoles_error() {
    let (mut ctx, n) = build_context(true);
    let density = density(n);
    let (exact, _, exact_dipoles) = reference(&ctx, n, &density, Damping::None);
    let loose = PolarizationOptions { solver: SolverOptions { threshold: 1e-2, ..SolverOptions::default() }, ..tight() };
    ctx.set_polarization(loose).unwrap();
    let mut operator = vec![0.0; n * n];
    let report = ctx.compute_induction(&density, &mut operator).unwrap();
    let mut dipoles = vec![0.0; 12];
    ctx.get_induced_dipoles(&mut dipoles).unwrap();
    let error: Vec<f64> = dipoles.iter().zip(exact_dipoles.get_data()).map(|(a, b)| a - b).collect();
    let relay = PolarizableSystem::new(&reference_sites(), Damping::None).unwrap().relay_matrix().unwrap();
    let relay_error: Vec<f64> = (0..12).map(|i| (0..12).map(|j| relay[(i, j)] * error[j]).sum()).collect();
    let second_order = 0.5 * dot(&error, &relay_error);
    assert!(dot(&error, &error).sqrt() > 1e-6, "the loose solve should leave an error");
    assert!(second_order > 0.0);
    assert!((report.energy - exact - second_order).abs() < 1e-12, "{} - {exact} against {second_order}", report.energy);
}

#[test]
fn a_threshold_for_one_call_replaces_the_setup_s_for_that_call_alone() {
    let (mut ctx, n) = build_context(true);
    let density = density(n);
    let (exact, _, _) = reference(&ctx, n, &density, Damping::None);
    let mut operator = vec![0.0; n * n];
    // The setup's threshold given per call: the same solve, bit for bit.
    ctx.set_polarization(tight()).unwrap();
    let setup = ctx.compute_induction(&density, &mut operator).unwrap();
    let setup_operator = operator.clone();
    ctx.free_polarization();
    ctx.set_polarization(tight()).unwrap();
    let given = ctx.compute_induction_within(&density, &mut operator, 1e-12).unwrap();
    assert_eq!((given.energy, given.iterations), (setup.energy, setup.iterations));
    assert_eq!(operator, setup_operator);
    // A looser one: fewer iterations, the energy second order in the error; the next call is the setup's again.
    ctx.free_polarization();
    ctx.set_polarization(tight()).unwrap();
    let loose = ctx.compute_induction_within(&density, &mut operator, 1e-3).unwrap();
    assert!(loose.iterations < setup.iterations && loose.residual < 1e-3);
    assert!((loose.energy - exact).abs() < 1e-5 * exact.abs(), "{} against {exact}", loose.energy);
    let after = ctx.compute_induction(&density, &mut operator).unwrap();
    assert!(after.residual < 1e-12 && (after.energy - exact).abs() < 1e-10 * exact.abs());
    for bad in [0.0, -1e-8, f64::NAN, f64::INFINITY] {
        assert!(ctx.compute_induction_within(&density, &mut operator, bad).is_err(), "threshold {bad}");
    }
}

#[test]
fn dipoles_are_warm_started_between_calls() {
    let (mut ctx, n) = build_context(true);
    let d0 = density(n);
    ctx.set_polarization(PolarizationOptions { damping: Damping::thole(), ..PolarizationOptions::default() }).unwrap();

    let mut dipoles = vec![0.0; 12];
    assert!(ctx.get_induced_dipoles(&mut dipoles).is_err(), "no dipoles before the first call");

    let mut operator = vec![0.0; n * n];
    let first = ctx.compute_induction(&d0, &mut operator).unwrap();
    let again = ctx.compute_induction(&d0, &mut operator).unwrap();
    assert!(first.iterations > 1);
    assert_eq!(again.iterations, 1, "the previous dipoles already solve the equations");
    assert!((first.energy - again.energy).abs() < 1e-8 * first.energy.abs());

    // A small change of the density needs fewer iterations than starting from scratch.
    let shifted: Vec<f64> = d0.iter().map(|d| d * 1.01).collect();
    let warm = ctx.compute_induction(&shifted, &mut operator).unwrap();
    assert!(warm.iterations <= first.iterations);
}

#[test]
fn setup_and_call_order_errors_are_reported() {
    // Nothing to polarise yet.
    let mut bare = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    bare.set_shell_pairs(None).unwrap();
    bare.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    assert!(bare.set_polarization(PolarizationOptions::default()).is_err());

    let (mut ctx, n) = build_context(true);
    let density = density(n);
    let mut operator = vec![0.0; n * n];
    assert!(ctx.compute_induction(&density, &mut operator).is_err(), "not set up");
    ctx.set_polarization(PolarizationOptions::default()).unwrap();
    assert!(ctx.set_polarization(PolarizationOptions::default()).is_err(), "already set up");
    assert!(ctx.compute_induction(&density[1..], &mut operator).is_err(), "wrong density size");
    assert!(ctx.compute_induction(&density, &mut operator[1..].to_vec()).is_err(), "wrong operator size");
    assert!(ctx.compute_induction(&density, &mut operator).is_ok());

    // The polarization is derived from the polarizable sites: freeing them frees it.
    ctx.free_polarizable_sites();
    assert!(!ctx.has_polarization());
    assert!(ctx.compute_induction(&density, &mut operator).is_err());
}

#[test]
fn a_singular_polarizability_is_reported_at_setup() {
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    // A rank-one tensor is positive semi-definite, so it reads fine, but it has no inverse for the relay matrix.
    let pol = write_temp("singular", "1 Bohr -1 2\n5.0 5.0 5.0 1.0 0.0 0.0 0.0 0.0 0.0\n");
    ctx.set_polarizable_sites(pol.to_str().unwrap()).unwrap();
    std::fs::remove_file(&pol).ok();
    let error = ctx.set_polarization(PolarizationOptions::default()).unwrap_err();
    assert!(error.to_string().contains("singular"), "{error}");
    assert!(!ctx.has_polarization());
}

fn direction(n: usize, seed: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = (((3 + seed) * i + (5 + 2 * seed) * j) % 7) as f64 - 3.0;
            d[i + j * n] = value;
            d[j + i * n] = value;
        }
    }
    d
}

#[test]
fn the_response_operator_is_the_derivative_of_the_induction_operator() {
    // V is affine in D, so V1[X] = (V(D0 + eX) - V(D0 - eX)) / 2e; the response call has no static field and
    // must leave the ground-state dipoles alone.
    let (mut ctx, n) = build_context(true);
    let d0 = density(n);
    ctx.set_polarization(PolarizationOptions { damping: Damping::thole(), response_batch: 2, ..tight() }).unwrap();
    let x = direction(n, 0);
    let operator_at = |ctx: &AedelstenContext, epsilon: f64| {
        let shifted: Vec<f64> = d0.iter().zip(&x).map(|(d, x)| d + epsilon * x).collect();
        let mut operator = vec![0.0; n * n];
        ctx.compute_induction(&shifted, &mut operator).unwrap();
        operator
    };
    let epsilon = 1e-2;
    let (plus, minus) = (operator_at(&ctx, epsilon), operator_at(&ctx, -epsilon));
    let numeric: Vec<f64> = plus.iter().zip(&minus).map(|(p, m)| (p - m) / (2.0 * epsilon)).collect();

    let mut before = vec![0.0; 3 * 4];
    ctx.get_induced_dipoles(&mut before).unwrap();
    let mut response = vec![0.0; n * n];
    ctx.compute_induction_response(&x, 1, &mut response).unwrap();
    let scale = numeric.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let error = response.iter().zip(&numeric).fold(0.0f64, |a, (r, v)| a.max((r - v).abs()));
    assert!(scale > 1e-8 && error < 1e-8 * scale, "response vs finite difference: error {error:.3e}, scale {scale:.3e}");
    let mut after = vec![0.0; 3 * 4];
    ctx.get_induced_dipoles(&mut after).unwrap();
    assert_eq!(before, after, "the ground-state dipoles must not change");
}

#[test]
fn response_batches_agree_with_single_calls_and_the_operator_is_symmetric() {
    let (mut ctx, n) = build_context(true);
    ctx.set_polarization(PolarizationOptions { damping: Damping::thole(), response_batch: 2, ..tight() }).unwrap();
    let dens: Vec<Vec<f64>> = (0..3).map(|s| direction(n, s)).collect();
    let flat: Vec<f64> = dens.iter().flatten().copied().collect();
    // 3 densities with a batch of 2: exercises the sub-batching.
    let mut together = vec![0.0; 3 * n * n];
    ctx.compute_induction_response(&flat, 3, &mut together).unwrap();
    for (m, d) in dens.iter().enumerate() {
        let mut single = vec![0.0; n * n];
        ctx.compute_induction_response(d, 1, &mut single).unwrap();
        let block = &together[m * n * n..(m + 1) * n * n];
        let error = single.iter().zip(block).fold(0.0f64, |a, (s, b): (&f64, &f64)| a.max((s - b).abs()));
        assert!(error < 1e-12, "density {m}: {error:.3e}");
    }
    // <D2, V1[D1]> = <D1, V1[D2]>: the response kernel is a symmetric bilinear form.
    let (a, b) = (dot(&dens[1], &together[..n * n]), dot(&dens[0], &together[n * n..2 * n * n]));
    assert!((a - b).abs() < 1e-9 * a.abs().max(1.0), "{a:.10e} vs {b:.10e}");
    // An antisymmetric density polarizes nothing.
    let mut anti = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..i {
            anti[i + j * n] = 1.0;
            anti[j + i * n] = -1.0;
        }
    }
    let mut out = vec![1.0; n * n];
    ctx.compute_induction_response(&anti, 1, &mut out).unwrap();
    assert!(out.iter().all(|v| v.abs() < 1e-12));
}

#[test]
fn the_couplings_are_the_fields_against_the_dipoles_of_the_other_densities() {
    // K_mn = -f_m . R f_n against two oracles: the public electronic field and the dense solve, and tr(D_m V1[D_n])
    // through the response operators (the dipoles' operator pass). 3 densities with a batch of 2 (the
    // sub-batching), one nonsymmetric (only its symmetric part polarizes); symmetric; the ground state's dipoles
    // untouched; with the field integrals evaluated and kept alike.
    let (mut ctx, n) = build_context(true);
    let mut dens: Vec<Vec<f64>> = (0..3).map(|s| direction(n, s)).collect();
    for i in 0..n {
        dens[2][i + ((i + 1) % n) * n] += 0.5;
    }
    let flat: Vec<f64> = dens.iter().flatten().copied().collect();
    let sites = reference_sites();
    let system = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    let field_of = |ctx: &AedelstenContext, d: &[f64]| {
        let mut f = vec![0.0; 3 * sites.len()];
        ctx.compute_electronic_field(d, SiteSelection::Polarizable, &SiteChunking::Sites(2), &mut f).unwrap();
        Matrix::<f64>::new_from_vec(3, sites.len(), f.into_boxed_slice())
    };
    let fields: Vec<Matrix<f64>> = dens.iter().map(|d| field_of(&ctx, d)).collect();
    let dense: Vec<f64> = (0..9)
        .map(|index| {
            let (m, k) = (index % 3, index / 3);
            let mut mu = Matrix::<f64>::new(3, sites.len());
            system.solve_dense(&fields[k], &mut mu).unwrap();
            -dot(fields[m].get_data(), mu.get_data())
        })
        .collect();
    let scale = dense.iter().fold(0.0f64, |a, k| a.max(k.abs()));
    assert!(scale > 1e-6, "couplings too small to test: {scale:.3e}");
    ctx.set_polarization(PolarizationOptions { damping: Damping::thole(), response_batch: 2, ..tight() }).unwrap();
    for kept in [false, true] {
        ctx.set_resident_field_budget(kept.then_some((64, arrays::scratch::ByteUnit::MiB))).unwrap();
        ctx.compute_induction(&density(n), &mut vec![0.0; n * n]).unwrap();
        let mut before = vec![0.0; 3 * 4];
        ctx.get_induced_dipoles(&mut before).unwrap();

        let mut couplings = vec![0.0; 9];
        ctx.compute_induction_couplings(&flat, 3, &mut couplings).unwrap();
        let mut operators = vec![0.0; 3 * n * n];
        ctx.compute_induction_response(&flat, 3, &mut operators).unwrap();
        for (m, d_m) in dens.iter().enumerate() {
            for k in 0..3 {
                let value = couplings[m + 3 * k];
                let contracted = dot(d_m, &operators[k * n * n..(k + 1) * n * n]);
                for (name, reference) in [("dense", dense[m + 3 * k]), ("response", contracted)] {
                    assert!((value - reference).abs() < 1e-10 * scale, "kept {}, K({m},{k}) {value:.12e} vs {name} {reference:.12e}", kept);
                }
                assert!((value - couplings[k + 3 * m]).abs() < 1e-10 * scale, "kept {}: K not symmetric at ({m},{k})", kept);
            }
        }
        let mut after = vec![0.0; 3 * 4];
        ctx.get_induced_dipoles(&mut after).unwrap();
        assert_eq!(before, after, "the ground-state dipoles must not change");
    }
    // The sizes are checked, and response must be enabled.
    assert!(ctx.compute_induction_couplings(&flat, 3, &mut [0.0; 8]).is_err());
    assert!(ctx.compute_induction_couplings(&flat, 2, &mut [0.0; 4]).is_err());
    ctx.compute_induction_couplings(&[], 0, &mut []).unwrap();
    ctx.free_polarization();
    ctx.set_polarization(tight()).unwrap();
    assert!(ctx.compute_induction_couplings(&flat, 3, &mut [0.0; 9]).is_err());
}

#[test]
fn response_must_be_enabled_and_sized_correctly() {
    let (mut ctx, n) = build_context(false);
    ctx.set_polarization(tight()).unwrap();
    let d = density(n);
    let mut out = vec![0.0; n * n];
    assert!(ctx.compute_induction_response(&d, 1, &mut out).is_err());
    assert!(ctx.compute_induction_response(&d, 2, &mut out).is_err());
    ctx.compute_induction_response(&[], 0, &mut []).unwrap();
}

#[test]
fn states_get_their_own_dipoles_and_differ_from_the_averaged_density() {
    let (mut ctx, n) = build_context(true);
    ctx.set_polarization(PolarizationOptions { damping: Damping::thole(), state_slots: 3, ..tight() }).unwrap();
    let d0 = density(n);
    let dens: Vec<Vec<f64>> = (0..3).map(|s| d0.iter().zip(direction(n, s)).map(|(d, x)| d + 0.01 * x).collect()).collect();
    let flat: Vec<f64> = dens.iter().flatten().copied().collect();
    let mut ops = vec![0.0; 3 * n * n];
    let mut energies = vec![0.0; 3];
    // Twice: the second call is warm-started and must agree.
    ctx.compute_induction_states(&flat, 3, &mut ops, &mut energies).unwrap();
    let first = energies.clone();
    ctx.compute_induction_states(&flat, 3, &mut ops, &mut energies).unwrap();
    assert!(first.iter().zip(&energies).all(|(a, b)| (a - b).abs() < 1e-10));
    // A threshold for one call: the setup's gives the same; a looser one the energies to about its square.
    let (mut ops_within, mut within) = (vec![0.0; 3 * n * n], vec![0.0; 3]);
    ctx.compute_induction_states_within(&flat, 3, &mut ops_within, &mut within, 1e-12).unwrap();
    assert!(first.iter().zip(&within).all(|(a, b)| (a - b).abs() < 1e-10));
    let shifted: Vec<f64> = flat.iter().map(|d| d * 1.05).collect();
    ctx.compute_induction_states_within(&shifted, 3, &mut ops_within, &mut within, 1e-3).unwrap();
    let mut tight_energies = vec![0.0; 3];
    ctx.compute_induction_states(&shifted, 3, &mut ops_within, &mut tight_energies).unwrap();
    assert!(tight_energies.iter().zip(&within).all(|(a, b)| (a - b).abs() < 1e-5 * a.abs()), "{within:?} against {tight_energies:?}");
    for bad in [0.0, f64::NAN] {
        assert!(ctx.compute_induction_states_within(&flat, 3, &mut ops_within, &mut within, bad).is_err());
    }
    ctx.compute_induction_states(&flat, 3, &mut ops, &mut energies).unwrap();
    for (m, d) in dens.iter().enumerate() {
        let (mut v, v_states) = (vec![0.0; n * n], &ops[m * n * n..(m + 1) * n * n]);
        let report = ctx.compute_induction(d, &mut v).unwrap();
        assert!((report.energy - energies[m]).abs() < 1e-9, "state {m}");
        assert!(v.iter().zip(v_states.iter()).all(|(a, b)| (a - *b).abs() < 1e-9));
    }
    // E_ind is quadratic in D, so the average of the state energies is not the energy of the average.
    let avg: Vec<f64> = (0..n * n).map(|i| dens.iter().map(|d| d[i]).sum::<f64>() / 3.0).collect();
    let mut v = vec![0.0; n * n];
    let e_avg = ctx.compute_induction(&avg, &mut v).unwrap().energy;
    assert!((e_avg - energies.iter().sum::<f64>() / 3.0).abs() > 1e-12);
    // Too many states is an error.
    let mut ops4 = vec![0.0; 4 * n * n];
    assert!(ctx.compute_induction_states(&vec![0.0; 4 * n * n], 4, &mut ops4, &mut [0.0; 4]).is_err());
}

// The effective external field: for each direction i, the dipoles of a uniform unit field at every site from the
// dense solver, and V^(i) = -Sum_s mu_s^(i) . t_s from the full integral tensor; the citation registered.
#[test]
fn the_effective_external_field_matches_an_independent_reconstruction() {
    let (mut ctx, n) = build_context(true);
    let damping = Damping::thole();
    ctx.set_polarization(PolarizationOptions { damping, ..tight() }).unwrap();
    let mut operators = vec![0.0; 3 * n * n];
    ctx.compute_effective_external_field(&mut operators).unwrap();

    let sites = reference_sites();
    let count = sites.len();
    let system = PolarizableSystem::new(&sites, damping).unwrap();
    let mut integrals = vec![0.0; n * n * 3 * count];
    ctx.compute_electric_field_integrals(SiteSelection::Polarizable, 0..count, &mut integrals).unwrap();
    for (i, computed) in operators.chunks(n * n).enumerate() {
        let mut field = Matrix::<f64>::new(3, count);
        for site in field.iter_mut() {
            site[i] = 1.0;
        }
        let mut mu = Matrix::<f64>::new(3, count);
        system.solve_dense(&field, &mut mu).unwrap();
        let mut expected = vec![0.0; n * n];
        for (plane, &dipole) in integrals.chunks(n * n).zip(mu.get_data()) {
            for (v, x) in expected.iter_mut().zip(plane) {
                *v -= dipole * x;
            }
        }
        let scale = expected.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let error = computed.iter().zip(&expected).fold(0.0f64, |a, (c, e)| a.max((c - e).abs()));
        assert!(scale > 1e-6 && error < 1e-9 * scale, "direction {i}: error {error:.3e}, scale {scale:.3e}");
    }
    let doi = "10.1039/c6cp00669h";
    assert!(instrumentation::citations().into_iter().any(|r| r.reference.doi() == Some(doi)), "the EEF is cited");
}

// Reciprocity, which fixes the sign: tr(D V^(i)) is minus the i-component of the total dipole the density's electrons
// induce at the sites, mu(D) - mu(0) from two inductions (the static field cancels in the difference). So the
// electronic dipole moment -tr(D (r + V^(i))) includes the environment's.
#[test]
fn the_effective_external_field_gives_the_dipole_a_density_induces() {
    let (mut ctx, n) = build_context(true);
    ctx.set_polarization(PolarizationOptions { damping: Damping::thole(), ..tight() }).unwrap();
    let mut operators = vec![0.0; 3 * n * n];
    ctx.compute_effective_external_field(&mut operators).unwrap();
    let dipoles_of = |ctx: &AedelstenContext, d: &[f64]| {
        let mut operator = vec![0.0; n * n];
        ctx.compute_induction(d, &mut operator).unwrap();
        let mut mu = vec![0.0; 3 * SITES.len()];
        ctx.get_induced_dipoles(&mut mu).unwrap();
        mu
    };
    let d = density(n);
    let (with, without) = (dipoles_of(&ctx, &d), dipoles_of(&ctx, &vec![0.0; n * n]));
    for (i, v) in operators.chunks(n * n).enumerate() {
        let induced: f64 = with.chunks(3).zip(without.chunks(3)).map(|(a, b)| a[i] - b[i]).sum();
        let traced = dot(&d, v);
        assert!(induced.abs() > 1e-6 && (traced + induced).abs() < 1e-9 * induced.abs().max(1.0), "direction {i}: tr(D V) {traced:.10e} vs -induced {:.10e}", -induced);
    }
}
