// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use super::*;
use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use arrays::scratch::{Scratch, ByteUnit};
use sources::field::{add_field_from_multipoles, no_exclusions};
use sources::{Fragments, PointMultipole, PolarizableSite, Polarizability, Site};

// The iterative solver with freshly allocated buffers (the tests that care about reuse manage their own).
fn solve_iter(
    system: &PolarizableSystem,
    field: &Matrix<f64>,
    mu: &mut Matrix<f64>,
    guess: InitialGuess,
    options: &SolverOptions,
) -> Result<SolveReport, PolarizationError> {
    let mut buffers = IterativeBuffers::new(system.n_sites());
    system.solve_iterative(field, mu, guess, options, &mut buffers.scratch(), &Whole)
}

fn site(position: [f64; 3], alpha: Polarizability, fragments: &[i32]) -> PolarizableSite {
    PolarizableSite::new(position, alpha, Fragments::new(fragments))
}

fn iso(alpha: f64) -> Polarizability {
    Polarizability::Isotropic(alpha)
}

// A deterministic pseudo-random sequence in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

// Twelve sites in a box, mixed isotropic/anisotropic polarizabilities, in fragments of three (a few sites
// left with fragment 0), at spacings around typical molecular ones.
fn cluster() -> Vec<PolarizableSite> {
    cluster_in_a_box(6.0)
}

fn cluster_in_a_box(edge: f64) -> Vec<PolarizableSite> {
    let mut rng = Lcg(7);
    (0..12)
        .map(|s| {
            let position = [edge * rng.next(), edge * rng.next(), edge * rng.next()];
            let alpha = if s % 3 == 0 {
                iso(0.8 + rng.next())
            } else {
                // Diagonally dominant, hence positive definite: xx xy xz yy yz zz.
                Polarizability::Anisotropic([1.2 + rng.next(), 0.1 * rng.next(), 0.1 * rng.next(), 0.9 + rng.next(), 0.1 * rng.next(), 0.7 + rng.next()])
            };
            let fragment = if s >= 9 { 0 } else { 1 + (s / 3) as i32 };
            site(position, alpha, &[fragment])
        })
        .collect()
}

fn random_field(n_sites: usize, seed: u64) -> Matrix<f64> {
    let mut rng = Lcg(seed);
    let mut field = Matrix::<f64>::new(3, n_sites);
    for value in field.get_data_mut() {
        *value = 0.05 * (rng.next() - 0.5);
    }
    field
}

fn max_abs_difference(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
    a.get_data().iter().zip(b.get_data()).fold(0.0_f64, |worst, (x, y)| worst.max((x - y).abs()))
}

#[test]
fn the_tensor_is_the_field_of_a_point_dipole() {
    // Cross-check against the (independently verified) dipole field of the `sources` crate.
    let sites = [site([0.4, -0.3, 0.9], iso(1.0), &[]), site([2.1, 1.0, -1.2], iso(1.0), &[])];
    let system = PolarizableSystem::new(&sites, Damping::None).unwrap();
    let t = system.tensor(0, 1).unwrap().unwrap();
    let dipole = [0.3, -0.7, 0.5];
    let by_tensor = {
        let block = [[t[0], t[1], t[2]], [t[1], t[3], t[4]], [t[2], t[4], t[5]]];
        [0, 1, 2].map(|a| block[a][0] * dipole[0] + block[a][1] * dipole[1] + block[a][2] * dipole[2])
    };
    let mut reference = vec![0.0; 3];
    let source = PointMultipole::dipole(*sites[1].get_position_for_test(), 0.0, dipole);
    add_field_from_multipoles(&[Site::new(*sites[0].get_position_for_test())], &[source], no_exclusions, &mut reference);
    for axis in 0..3 {
        assert!((by_tensor[axis] - reference[axis]).abs() < 1e-13, "axis {axis}: {} vs {}", by_tensor[axis], reference[axis]);
    }
    // Symmetric in the pair.
    assert_eq!(system.tensor(1, 0).unwrap().unwrap(), t);
}

trait PositionForTest {
    fn get_position_for_test(&self) -> &[f64; 3];
}
impl PositionForTest for PolarizableSite {
    fn get_position_for_test(&self) -> &[f64; 3] {
        use sources::SiteTrait;
        self.get_position()
    }
}

#[test]
fn exclusion_follows_fragments_and_fragment_zero_excludes_nothing() {
    let sites = [
        site([0.0, 0.0, 0.0], iso(1.0), &[1]),
        site([1.0, 0.0, 0.0], iso(1.0), &[1]),
        site([2.0, 0.0, 0.0], iso(1.0), &[2]),
        site([3.0, 0.0, 0.0], iso(1.0), &[0]),
        site([4.0, 0.0, 0.0], iso(1.0), &[0]),
        site([5.0, 0.0, 0.0], iso(1.0), &[]),
    ];
    let system = PolarizableSystem::new(&sites, Damping::None).unwrap();
    assert!(system.excluded(0, 1));
    assert!(system.tensor(0, 1).unwrap().is_none());
    assert!(!system.excluded(0, 2));
    // Fragment 0 (a polarizable AIMP centre) and no fragment couple to everything, each other included.
    assert!(!system.excluded(3, 4) && system.tensor(3, 4).unwrap().is_some());
    assert!(!system.excluded(4, 5) && system.tensor(4, 5).unwrap().is_some());
    // A site never couples to itself.
    assert!(system.tensor(2, 2).unwrap().is_none());
}

#[test]
fn coincident_sites_are_an_error_unless_excluded() {
    let coincident = [site([1.0, 1.0, 1.0], iso(1.0), &[]), site([1.0, 1.0, 1.0], iso(1.0), &[])];
    let system = PolarizableSystem::new(&coincident, Damping::None).unwrap();
    assert_eq!(system.tensor(0, 1), Err(PolarizationError::CoincidentSites { first: 0, second: 1 }));
    let same_fragment = [site([1.0, 1.0, 1.0], iso(1.0), &[3]), site([1.0, 1.0, 1.0], iso(1.0), &[3])];
    assert_eq!(PolarizableSystem::new(&same_fragment, Damping::None).unwrap().tensor(0, 1), Ok(None));
    // The coupling and the iterative solver find the pair too, among others; the excluded pair passes.
    let mut sites = cluster_in_a_box(10.0);
    let twin = site(*sites[4].get_position_for_test(), iso(1.0), &[]);
    sites.insert(8, twin);
    let field = random_field(sites.len(), 3);
    let mut out = Matrix::<f64>::new(3, sites.len());
    for damping in [Damping::None, Damping::thole()] {
        let system = PolarizableSystem::new(&sites, damping).unwrap();
        assert_eq!(system.add_field_of_dipoles(&field, &mut out), Err(PolarizationError::CoincidentSites { first: 4, second: 8 }));
        let mut mu = Matrix::<f64>::new(3, sites.len());
        assert_eq!(
            solve_iter(&system, &field, &mut mu, InitialGuess::Static, &SolverOptions::default()),
            Err(PolarizationError::CoincidentSites { first: 4, second: 8 })
        );
        let system = PolarizableSystem::new(&same_fragment, damping).unwrap();
        let mut out = Matrix::<f64>::new(3, 2);
        assert!(system.add_field_of_dipoles(&random_field(2, 1), &mut out).is_ok());
    }
}

#[test]
fn singular_polarizabilities_and_wrong_shapes_are_errors() {
    let singular = [site([0.0; 3], Polarizability::Anisotropic([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), &[])];
    assert_eq!(PolarizableSystem::new(&singular, Damping::None).err(), Some(PolarizationError::SingularPolarizability { site: 0 }));

    let sites = cluster();
    let system = PolarizableSystem::new(&sites, Damping::None).unwrap();
    let wrong = Matrix::<f64>::new(3, 5);
    let mut out = Matrix::<f64>::new(3, sites.len());
    assert!(matches!(system.add_field_of_dipoles(&wrong, &mut out), Err(PolarizationError::Shape { what: "dipoles", .. })));
    assert!(matches!(system.solve_dense(&wrong, &mut out), Err(PolarizationError::Shape { what: "field", .. })));
}

#[test]
fn damping_bounds_the_short_range_coupling() {
    let sites = [site([0.0; 3], iso(1.5), &[]), site([0.6, 0.0, 0.0], iso(1.5), &[])];
    let plain = PolarizableSystem::new(&sites, Damping::None).unwrap().tensor(0, 1).unwrap().unwrap();
    let damped = PolarizableSystem::new(&sites, Damping::thole()).unwrap().tensor(0, 1).unwrap().unwrap();
    for (d, p) in damped.iter().zip(&plain) {
        assert!(d.abs() < p.abs() || p == &0.0, "damped {d} vs plain {p}");
    }
    // Far away the two agree.
    let far = [site([0.0; 3], iso(1.5), &[]), site([60.0, 0.0, 0.0], iso(1.5), &[])];
    let plain = PolarizableSystem::new(&far, Damping::None).unwrap().tensor(0, 1).unwrap().unwrap();
    let damped = PolarizableSystem::new(&far, Damping::thole()).unwrap().tensor(0, 1).unwrap().unwrap();
    for (d, p) in damped.iter().zip(&plain) {
        assert!((d - p).abs() < 1e-12 * p.abs().max(1e-12));
    }
}

#[test]
fn the_matvec_agrees_with_the_dense_relay_matrix() {
    // With fragments (pairs tested for exclusion) and without (the pair loop that tests nothing).
    let without_fragments: Vec<PolarizableSite> =
        cluster().iter().map(|s| site(*s.get_position_for_test(), *s.get_polarizability(), &[])).collect();
    for (sites, damping) in [cluster(), without_fragments].into_iter().flat_map(|sites| [Damping::None, Damping::thole()].map(|d| (sites.clone(), d))) {
        let n = sites.len();
        let system = PolarizableSystem::new(&sites, damping).unwrap();
        let mu = random_field(n, 3);
        let mut relay_mu = Matrix::<f64>::new(3, n);
        system.apply_relay(&mu, &mut relay_mu).unwrap();

        let relay = system.relay_matrix().unwrap();
        let mut dense = Matrix::<f64>::new(3, n);
        for row in 0..3 * n {
            dense.get_data_mut()[row] = (0..3 * n).map(|col| relay[(row, col)] * mu.get_data()[col]).sum();
        }
        assert!(max_abs_difference(&relay_mu, &dense) < 1e-12, "{damping:?}, exclusions {}", system.parameters().has_exclusions);
        // The relay matrix is symmetric.
        for i in 0..3 * n {
            for j in 0..i {
                assert!((relay[(i, j)] - relay[(j, i)]).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn two_sites_on_an_axis_match_the_analytic_solution() {
    // Two equal isotropic sites a distance r apart in a uniform field F. Along the axis T_zz = 2/r^3, so
    // mu = alpha F / (1 - 2 alpha / r^3); across it T_xx = -1/r^3, so mu = alpha F / (1 + alpha / r^3).
    let (alpha, r, f) = (1.3, 4.0, 0.02);
    let sites = [site([0.0, 0.0, 0.0], iso(alpha), &[]), site([0.0, 0.0, r], iso(alpha), &[])];
    let system = PolarizableSystem::new(&sites, Damping::None).unwrap();
    for (axis, factor) in [(2, 1.0 / (1.0 - 2.0 * alpha / r.powi(3))), (0, 1.0 / (1.0 + alpha / r.powi(3)))] {
        let mut field = Matrix::<f64>::new(3, 2);
        for site in 0..2 {
            field.get_col_mut(site)[axis] = f;
        }
        let expected = alpha * f * factor;

        let mut dense = Matrix::<f64>::new(3, 2);
        system.solve_dense(&field, &mut dense).unwrap();
        let mut iterative = Matrix::<f64>::new(3, 2);
        solve_iter(&system, &field, &mut iterative, InitialGuess::Static, &SolverOptions::default()).unwrap();
        for solution in [&dense, &iterative] {
            for site in 0..2 {
                assert!((solution.get_col(site)[axis] - expected).abs() < 1e-9, "axis {axis}: {} vs {expected}", solution.get_col(site)[axis]);
            }
        }
    }
}

#[test]
fn iterative_and_dense_solutions_agree_on_a_damped_fragmented_cluster() {
    let sites = cluster();
    let n = sites.len();
    let system = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    let field = random_field(n, 11);

    let mut dense = Matrix::<f64>::new(3, n);
    system.solve_dense(&field, &mut dense).unwrap();

    let mut iterative = Matrix::<f64>::new(3, n);
    let report = solve_iter(&system, &field, &mut iterative, InitialGuess::Static, &SolverOptions::default()).unwrap();
    assert!(max_abs_difference(&dense, &iterative) < 1e-7, "difference {}", max_abs_difference(&dense, &iterative));
    assert!(report.iterations < 40, "took {} iterations", report.iterations);

    // The solution satisfies the equations: (alpha^-1 - T) mu = F.
    let mut relay_mu = Matrix::<f64>::new(3, n);
    system.apply_relay(&iterative, &mut relay_mu).unwrap();
    assert!(max_abs_difference(&relay_mu, &field) < 1e-6);
}

#[test]
fn conjugate_gradients_end_within_the_dimension_and_report_the_functional() {
    // In exact arithmetic CG ends within 3S steps; on these small clusters it is far below.
    for (edge, damping) in [(14.0, Damping::None), (6.0, Damping::thole())] {
        let sites = cluster_in_a_box(edge);
        let n = sites.len();
        let system = PolarizableSystem::new(&sites, damping).unwrap();
        let field = random_field(n, 5);
        let mut dense = Matrix::<f64>::new(3, n);
        system.solve_dense(&field, &mut dense).unwrap();
        let mut mu = Matrix::<f64>::new(3, n);
        let options = SolverOptions { threshold: 1e-12, ..SolverOptions::default() };
        let report = solve_iter(&system, &field, &mut mu, InitialGuess::Static, &options).unwrap();
        assert!(report.iterations <= 3 * n + 1, "{} iterations for {} unknowns", report.iterations, 3 * n);
        assert!(max_abs_difference(&dense, &mu) < 1e-10, "{damping:?}: {}", max_abs_difference(&dense, &mu));
        let functional = system.induction_functional(&mu, &field).unwrap();
        assert!((report.functional - functional).abs() < 1e-13, "{} vs {functional}", report.functional);
    }
}

#[test]
fn the_pair_rows_cover_the_triangle_once_in_nearly_equal_runs() {
    for (n, parts) in [(0, 3), (1, 2), (2, 4), (7, 3), (1000, 7), (67360, 64)] {
        let runs: Vec<_> = (0..parts).map(|part| pair_rows(n, part, parts)).collect();
        assert_eq!(runs[0].start, 0);
        assert_eq!(runs[parts - 1].end, n);
        for pair in runs.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "n {n}, parts {parts}");
        }
        let pairs = |run: &std::ops::Range<usize>| run.clone().sum::<usize>();
        if n > 100 * parts {
            let total = n * (n - 1) / 2;
            for run in &runs {
                let share = pairs(run) as f64 / (total as f64 / parts as f64);
                assert!((share - 1.0).abs() < 0.01 + parts as f64 * n as f64 / total as f64, "n {n}, parts {parts}: {run:?} {share}");
            }
        }
    }
}

// One share of a split, for checking that the shares' fields add up to the whole: the sum is left to the test.
struct Share(usize, usize);
impl Collective for Share {
    fn share(&self) -> (usize, usize) {
        (self.0, self.1)
    }
    fn sum(&self, _: &mut [f64]) -> Result<(), PolarizationError> {
        Ok(())
    }
    fn broadcast(&self, _: &mut [f64]) -> Result<(), PolarizationError> {
        Ok(())
    }
}

#[test]
fn the_shares_of_the_coupling_add_up_to_the_whole() {
    let sites = cluster();
    let n = sites.len();
    let system = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    let mu = random_field(n, 41);
    let mut whole = Matrix::<f64>::new(3, n);
    system.add_field_of_dipoles(&mu, &mut whole).unwrap();
    let (mut dipoles_by_axis, mut field_by_axis) = (Matrix::<f64>::new(n, 3), Matrix::<f64>::new(n, 3));
    for parts in [2, 3, 5] {
        let mut summed = Matrix::<f64>::new(3, n);
        for part in 0..parts {
            let mut share = Matrix::<f64>::new(3, n);
            let mut work = CouplingWork::new(dipoles_by_axis.get_data_mut(), field_by_axis.get_data_mut(), n);
            system.field_of_dipoles(&mu, &mut share, &mut work, &Share(part, parts)).unwrap();
            for (total, value) in summed.get_data_mut().iter_mut().zip(share.get_data()) {
                *total += value;
            }
        }
        assert!(max_abs_difference(&summed, &whole) < 1e-15, "{parts} shares: {}", max_abs_difference(&summed, &whole));
    }
}

#[test]
fn a_provided_converged_guess_converges_immediately() {
    let sites = cluster();
    let n = sites.len();
    let system = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    let field = random_field(n, 2);
    let mut mu = Matrix::<f64>::new(3, n);
    solve_iter(&system, &field, &mut mu, InitialGuess::Static, &SolverOptions { threshold: 1e-13, ..SolverOptions::default() }).unwrap();
    let report = solve_iter(&system, &field, &mut mu, InitialGuess::Provided, &SolverOptions::default()).unwrap();
    assert_eq!(report.iterations, 1);
}

#[test]
fn excluded_sites_stay_uncoupled() {
    // Two close sites of the same fragment do not polarise each other: mu = alpha F for both.
    let sites = [site([0.0; 3], iso(2.0), &[1]), site([0.5, 0.0, 0.0], iso(3.0), &[1])];
    let system = PolarizableSystem::new(&sites, Damping::None).unwrap();
    let mut field = Matrix::<f64>::new(3, 2);
    field.get_col_mut(0).copy_from_slice(&[0.01, 0.02, 0.03]);
    field.get_col_mut(1).copy_from_slice(&[-0.02, 0.0, 0.04]);
    let mut mu = Matrix::<f64>::new(3, 2);
    system.solve_dense(&field, &mut mu).unwrap();
    for (site, alpha) in [(0, 2.0), (1, 3.0)] {
        for axis in 0..3 {
            assert!((mu.get_col(site)[axis] - alpha * field.get_col(site)[axis]).abs() < 1e-12);
        }
    }
}

#[test]
fn the_induced_dipoles_minimise_the_induction_functional() {
    let sites = cluster();
    let n = sites.len();
    let system = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    let field = random_field(n, 13);
    let mut mu = Matrix::<f64>::new(3, n);
    system.solve_dense(&field, &mut mu).unwrap();

    let minimum = system.induction_functional(&mu, &field).unwrap();
    let energy = system.induction_energy(&mu, &field).unwrap();
    assert!((minimum - energy).abs() < 1e-12, "{minimum} vs {energy}");
    assert!(energy < 0.0, "induction lowers the energy");

    // Any perturbation of the dipoles raises the functional (its Hessian, the relay matrix, is positive definite).
    let perturbation = random_field(n, 17);
    for scale in [1e-3, -1e-3, 1e-2] {
        let mut perturbed = Matrix::<f64>::new(3, n);
        perturbed.get_data_mut().copy_from_slice(mu.get_data());
        for (p, d) in perturbed.get_data_mut().iter_mut().zip(perturbation.get_data()) {
            *p += scale * d;
        }
        assert!(system.induction_functional(&perturbed, &field).unwrap() > minimum);
    }
}

#[test]
fn undamped_close_sites_are_a_polarization_catastrophe_and_damping_softens_it() {
    // 2 alpha / r^3 > 1 along the axis: the relay matrix has a negative eigenvalue.
    let sites = [site([0.0; 3], iso(10.0), &[]), site([0.0, 0.0, 2.0], iso(10.0), &[])];
    let field = random_field(2, 1);
    let mut mu = Matrix::<f64>::new(3, 2);
    let undamped = PolarizableSystem::new(&sites, Damping::None).unwrap();
    assert!(matches!(undamped.solve_dense(&field, &mut mu), Err(PolarizationError::PolarizationCatastrophe { lowest_eigenvalue }) if lowest_eigenvalue < 0.0));
    // The iterative solver meets a direction of negative curvature.
    assert!(matches!(
        solve_iter(&undamped, &field, &mut mu, InitialGuess::Static, &SolverOptions::default()),
        Err(PolarizationError::PolarizationCatastrophe { lowest_eigenvalue }) if lowest_eigenvalue <= 0.0
    ));

    let lowest = |system: &PolarizableSystem| {
        use arrays::matrix::MatrixMath;
        system.relay_matrix().unwrap().eigenvalues().iter().copied().fold(f64::INFINITY, f64::min)
    };
    let damped = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    assert!(lowest(&damped) > lowest(&undamped));
}

#[test]
fn a_scratch_for_the_wrong_system_is_rejected() {
    let sites = cluster();
    let n = sites.len();
    let system = PolarizableSystem::new(&sites, Damping::None).unwrap();
    let field = random_field(n, 1);
    let mut mu = Matrix::<f64>::new(3, n);
    let options = SolverOptions::default();
    let mut wrong_sites = IterativeBuffers::new(n + 1);
    assert_eq!(
        system.solve_iterative(&field, &mut mu, InitialGuess::Static, &options, &mut wrong_sites.scratch(), &Whole),
        Err(PolarizationError::ScratchMismatch { needed: n, found: n + 1 })
    );
}

#[test]
fn a_scratch_carved_from_an_arena_gives_the_same_solution_as_heap_buffers() {
    let sites = cluster();
    let n = sites.len();
    let system = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    let field = random_field(n, 29);
    let options = SolverOptions::default();

    let mut from_heap = Matrix::<f64>::new(3, n);
    let heap_report = solve_iter(&system, &field, &mut from_heap, InitialGuess::Static, &options).unwrap();

    let bytes = IterativeScratch::required_bytes(n);
    let mut arena_memory = Scratch::new(bytes.div_ceil(1024), ByteUnit::KiB).unwrap();
    let arena = arena_memory.get_arena();
    let mut scratch = IterativeScratch::from_arena(n, &arena).unwrap();
    let mut from_arena = Matrix::<f64>::new(3, n);
    let arena_report = system.solve_iterative(&field, &mut from_arena, InitialGuess::Static, &options, &mut scratch, &Whole).unwrap();
    assert_eq!(from_heap.get_data(), from_arena.get_data());
    assert_eq!(heap_report, arena_report);

    // An arena that is too small is a clean error, not a panic.
    let mut tiny = Scratch::new(1, ByteUnit::KiB).unwrap();
    assert!(IterativeScratch::from_arena(n, &tiny.get_arena()).is_err());
}

#[test]
fn stale_scratch_contents_do_not_change_the_result() {
    // One set of buffers used for different problems in a row, as across SCF iterations.
    let sites = cluster();
    let n = sites.len();
    let system = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    let options = SolverOptions::default();
    let mut buffers = IterativeBuffers::new(n);
    let mut previous: Option<Matrix<f64>> = None;
    for seed in [31, 37, 31] {
        let field = random_field(n, seed);
        let mut reused = Matrix::<f64>::new(3, n);
        system.solve_iterative(&field, &mut reused, InitialGuess::Static, &options, &mut buffers.scratch(), &Whole).unwrap();
        let mut fresh = Matrix::<f64>::new(3, n);
        solve_iter(&system, &field, &mut fresh, InitialGuess::Static, &options).unwrap();
        assert_eq!(reused.get_data(), fresh.get_data(), "seed {seed}");
        // The third problem repeats the first, so its solution must too.
        if let Some(first) = &previous {
            if seed == 31 {
                assert_eq!(first.get_data(), reused.get_data());
            }
        } else {
            previous = Some(reused);
        }
    }
}
