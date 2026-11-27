// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Heap allocations on the library's per-call paths. A counting global
// allocator counts this thread's allocations around the second call of each path.
//
// The paths an SCF iterates (the direct J/K build, one-electron integrals) allocate nothing: all their scratch is
// the scratch's. The once-per-geometry paths (the stored ERI tensor, the ERI gradient) keep each shell pair's
// Hermite coefficients on the heap for the call (per-pair results on the heap keep the scratch size independent of
// the molecule), so they may allocate per shell pair, never per
// quartet. `allocation_counts` prints the counts per path.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{TwoElectronTerm, TwoElectronTermKind};
use aedelsten::{AedelstenContext, ContractionType, ERIAOSymmetry, ERIIndexOrder, ERIRangeOperator, OneElectronOperator};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// This thread's allocations during `f`.
fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(|c| c.get());
    f();
    ALLOCATIONS.with(|c| c.get()) - before
}

fn context(name: &str, input: &str) -> AedelstenContext {
    let dir = common::temp_path(&format!("hot_paths_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    ctx.set_shell_pairs(Some(1e-14)).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(Some(1e-12)).unwrap();
    ctx
}

fn density(n: usize) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..=i {
            let v = 0.1 + 0.05 * (((i * 7 + j * 3) % 11) as f64);
            p[(i, j)] = v;
            p[(j, i)] = v;
        }
    }
    p
}

/// `(path, allocations of its second call)` for each hot path. Every buffer the calls write into is made before
/// the counted region.
fn second_call_allocations(ctx: &AedelstenContext) -> Vec<(&'static str, usize)> {
    let n = ctx.get_n_cont();
    let p = density(n);
    let (mut j, mut k) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    let mut overlap = vec![0.0; n * n];
    let mut eri = vec![0.0; aedelsten::compute_s8_size(n)];
    let mut gradient = vec![0.0; 3 * ctx.get_n_gradient_centres()];
    let s8 = ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor };

    let mut direct_jk = || {
        ctx.compute_fock_direct_restricted(p.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, ERIRangeOperator::FullRange, Some(1e-12)).unwrap();
    };
    direct_jk();
    let direct = allocations(&mut direct_jk);

    // A non-symmetric density (its J reads the symmetric part tile by tile, with no n x n copy).
    let mut q = density(n);
    for (i, x) in q.get_data_mut().iter_mut().enumerate() {
        *x += 0.01 * (i % 7) as f64;
    }
    let mut direct_general = || {
        ctx.compute_fock_direct_general(q.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, ERIRangeOperator::FullRange, Some(1e-12)).unwrap();
    };
    direct_general();
    let general = allocations(&mut direct_general);

    // Several non-symmetric densities in one walk (back to back, their J and K likewise).
    let mut stacked = q.get_data().repeat(3);
    for (i, x) in stacked.iter_mut().enumerate() {
        *x += 0.001 * (i % 5) as f64;
    }
    let (mut j_multi, mut k_multi) = (vec![0.0; stacked.len()], vec![0.0; stacked.len()]);
    let mut direct_multi = || {
        ctx.compute_fock_direct_general_multi(&stacked, &mut j_multi, &mut k_multi, 3, 1.0, -0.5, ERIRangeOperator::FullRange, Some(1e-12)).unwrap();
    };
    direct_multi();
    let multi = allocations(&mut direct_multi);

    // The active-space walks, one basis and two, at n_act = 3, and T from M (M, T and the coefficients made up front).
    let m = 3;
    let (c_bra, c_ket): (Vec<f64>, Vec<f64>) = ((0..n * m).map(|i| 0.1 * (i % 5) as f64 - 0.2).collect(), (0..n * m).map(|i| 0.1 * (i % 7) as f64 - 0.3).collect());
    let (mut j_a, mut k_a) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    let mut m_active = vec![0.0; n * (n + 1) / 2 * m * (m + 1) / 2];
    let mut active_space = || {
        ctx.compute_fock_direct_active_space(
            p.get_data(), p.get_data(), j.get_data_mut(), k.get_data_mut(), j_a.get_data_mut(), k_a.get_data_mut(),
            1.0, -0.5, 1.0, -0.5, &c_ket, m, &mut m_active, Some(1e-12), Some(1e-12),
        )
        .unwrap();
    };
    active_space();
    let active = allocations(&mut active_space);
    let mut m_mixed = vec![0.0; n * (n + 1) / 2 * m * m];
    let mut mixed_active_space = || {
        ctx.compute_fock_direct_mixed_active_space(q.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -1.0, &c_bra, &c_ket, m, &mut m_mixed, Some(1e-12), Some(1e-12)).unwrap();
    };
    mixed_active_space();
    let mixed = allocations(&mut mixed_active_space);
    let mut t_active = vec![0.0; m * m * (m + 1) / 2 * n];
    let mut one_index = || ctx.transform_pair_coulomb_one_index(&m_active, m * (m + 1) / 2, &c_ket, m, &mut t_active).unwrap();
    one_index();
    let one_index_count = allocations(&mut one_index);

    // The half transform, one basis (the same slice on both sides) and two (one column of another basis on the left).
    let mut x_one = vec![0.0; m * n * m * n];
    let mut half_one = || ctx.compute_eri_half_transformed(&c_ket, m, &c_ket, m, &mut x_one, Some(1e-12), Some(1e-12), true).unwrap();
    half_one();
    let half_one_count = allocations(&mut half_one);
    let mut x_two = vec![0.0; n * m * n];
    let mut half_two = || ctx.compute_eri_half_transformed(&c_bra[n..2 * n], 1, &c_ket, m, &mut x_two, Some(1e-12), Some(1e-12), true).unwrap();
    half_two();
    let half_two_count = allocations(&mut half_two);
    // A batch of sigma, the second half of the shells, one basis and two.
    let half = infos_first_ao_at_half(&ctx);
    let mut x_sigma_one = vec![0.0; m * n * m * (n - half)];
    let mut sigma_one = || ctx.compute_eri_half_transformed_sigma(&c_ket, m, &c_ket, m, half, n, &mut x_sigma_one, Some(1e-12), Some(1e-12), true).unwrap();
    sigma_one();
    let sigma_one_count = allocations(&mut sigma_one);
    let mut x_sigma_two = vec![0.0; n * m * (n - half)];
    let mut sigma_two = || ctx.compute_eri_half_transformed_sigma(&c_bra[n..2 * n], 1, &c_ket, m, half, n, &mut x_sigma_two, Some(1e-12), Some(1e-12), true).unwrap();
    sigma_two();
    let sigma_two_count = allocations(&mut sigma_two);

    let mut one_electron = || ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();
    one_electron();
    let one = allocations(&mut one_electron);

    // A component operator (three planes, the antisymmetric fold), and the coarse span of each operator call.
    let mut angular_momentum = vec![0.0; 3 * n * n];
    let mut one_electron_components = || ctx.compute_one_electron_integrals(&mut angular_momentum, OneElectronOperator::angular_momentum([0.1, 0.2, 0.3]), false).unwrap();
    one_electron_components();
    let components = allocations(&mut one_electron_components);

    let mut stored = || ctx.compute_eri_stored(&mut eri, s8, ERIRangeOperator::FullRange).unwrap();
    stored();
    let stored_count = allocations(&mut stored);

    let view = p.get_data();
    let terms = [
        TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: view, right: view, coeff: 1.0 },
        TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: view, right: view, coeff: -0.5 },
    ];
    let mut eri_gradient = || ctx.compute_two_electron_gradient(&mut gradient, &terms, ERIRangeOperator::FullRange, Some(1e-12), None).unwrap();
    eri_gradient();
    let gradient_count = allocations(&mut eri_gradient);

    // The AO values and gradients of every other shell, last first, at 300 points (more than one pass per shell).
    let shells: Vec<usize> = (0..ctx.get_n_shells()).rev().step_by(2).collect();
    let mut infos = vec![aedelsten::ShellInfo::default(); ctx.get_n_shells()];
    ctx.get_shells(&mut infos).unwrap();
    let n_listed: usize = shells.iter().map(|&s| infos[s].n_ao).sum();
    let points: Vec<f64> = (0..3 * 300).map(|i| ((i * 37) % 101) as f64 * 0.04 - 2.0).collect();
    let (mut values, mut ao_gradients) = (vec![0.0; n_listed * 300], vec![0.0; 3 * n_listed * 300]);
    let mut ao_values = || ctx.compute_ao_values(&shells, &points, 3, &mut values, Some(&mut ao_gradients)).unwrap();
    ao_values();
    let ao_values_count = allocations(&mut ao_values);
    let mut ao_hessians = vec![0.0; 6 * n_listed * 300];
    let mut second = || ctx.compute_ao_hessians(&shells, &points, 3, &mut values, &mut ao_gradients, &mut ao_hessians).unwrap();
    second();
    let ao_hessians_count = allocations(&mut second);

    vec![
        ("direct J/K build", direct),
        ("direct J/K of a general density", general),
        ("direct J/K of several general densities", multi),
        ("direct active-space walk", active),
        ("direct mixed-basis active-space walk", mixed),
        ("T from the active walk's M", one_index_count),
        ("direct half transform, one basis", half_one_count),
        ("direct half transform, two bases", half_two_count),
        ("direct half transform, a batch of sigma, one basis", sigma_one_count),
        ("direct half transform, a batch of sigma, two bases", sigma_two_count),
        ("one-electron overlap", one),
        ("one-electron angular momentum", components),
        ("stored S8 ERI tensor", stored_count),
        ("direct ERI gradient", gradient_count),
        ("AO values and gradients of a shell list", ao_values_count),
        ("AO second derivatives of a shell list", ao_hessians_count),
    ]
}

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";

#[test]
fn the_iterated_paths_allocate_nothing_and_the_others_only_per_pair() {
    // Shell pairs: water in x2c-SVPall, general, has 7 shells (O s p d, H s p twice), 28 pairs; its quartets number
    // about pairs^2 / 2 (406), so a bound of a few allocations per pair catches anything per quartet.
    for (name, input, pairs) in [("water", WATER, 28)] {
        let ctx = context(name, input);
        for (path, count) in second_call_allocations(&ctx) {
            eprintln!("{name}: {path}: {count} allocations");
            match path {
                "direct J/K build" | "direct J/K of a general density" | "direct J/K of several general densities" | "direct active-space walk" | "direct mixed-basis active-space walk" | "T from the active walk's M" | "direct half transform, one basis" | "direct half transform, two bases" | "direct half transform, a batch of sigma, one basis" | "direct half transform, a batch of sigma, two bases" | "one-electron overlap" | "one-electron angular momentum" | "AO values and gradients of a shell list" | "AO second derivatives of a shell list" => assert_eq!(count, 0, "{name}: {path} allocated {count} times"),
                _ => assert!(count <= 3 * pairs + 2, "{name}: {path} allocated {count} times for {pairs} shell pairs"),
            }
        }
    }
}

#[test]
fn symmetry_cleaning_allocates_nothing() {
    // The per-call symmetry paths take their scratch from the context's buffer (libmsym is not called on them).
    for (name, input) in [("water_symmetry", WATER)] {
        let mut ctx = context(name, input);
        ctx.set_symmetry(aedelsten::PointGroupRequest::Highest, aedelsten::SymmetryOptions::default()).unwrap();
        let n = ctx.get_n_cont();
        let mut overlap = vec![0.0; n * n];
        ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();
        let mut orbitals = vec![0.0; n * n];
        for (i, column) in orbitals.chunks_exact_mut(n).enumerate() {
            column[i] = 1.0;
        }
        let mut labels = vec![aedelsten::OrbitalLabel::default(); n];
        let mut norms = vec![0.0; ctx.get_irreps().unwrap().len()];

        let mut clean = || {
            ctx.symmetrise_matrix(&mut overlap, aedelsten::Cleaning::Strict).unwrap();
        };
        clean();
        assert_eq!(allocations(&mut clean), 0, "{name}: symmetrise_matrix");
        let mut auto = || {
            ctx.symmetrise_matrix(&mut overlap, aedelsten::Cleaning::Average).unwrap();
        };
        auto();
        assert_eq!(allocations(&mut auto), 0, "{name}: symmetrise_matrix (Average)");
        let mut analyse = || {
            ctx.analyse_matrix(&overlap, aedelsten::OperatorSymmetry::TotallySymmetric).unwrap();
        };
        analyse();
        assert_eq!(allocations(&mut analyse), 0, "{name}: analyse_matrix");
        let mut split = || ctx.irrep_norms(&overlap, &mut norms).unwrap();
        split();
        assert_eq!(allocations(&mut split), 0, "{name}: irrep_norms");
        // The library's own integrals, cleaned (strictly) before they are returned.
        let mut integrals = || ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();
        integrals();
        assert_eq!(allocations(&mut integrals), 0, "{name}: one-electron integrals with symmetry");
        // A multipole about the centre of mass, cleaned as a tensor.
        let origin = ctx.compute_centre_of_mass();
        let mut dipole = vec![0.0; 3 * n * n];
        let mut multipole = || {
            ctx.compute_one_electron_integrals(&mut dipole, OneElectronOperator::moment(aedelsten::MomentOrigin::new(origin, 1)), false).unwrap();
        };
        multipole();
        assert_eq!(allocations(&mut multipole), 0, "{name}: multipole integrals with symmetry");
        let mut orbital = || {
            ctx.symmetrise_orbitals(&mut orbitals, &mut labels).unwrap();
        };
        orbital();
        assert_eq!(allocations(&mut orbital), 0, "{name}: symmetrise_orbitals");
    }
}

#[test]
fn the_hermitian_eigensolver_allocates_nothing_after_setup() {
    // The prealloc forms work in the caller's `work` (the interleaved copy and LAPACK's scratch buffers) and write into
    // the caller's outputs: nothing on the heap per call.
    use aedelsten::arrays::matrix::{hermitian_eigenvectors_work_len, SplitComplexMatrix};
    let n = 20;
    let (mut re, mut im) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    for i in 0..n {
        for j in 0..=i {
            let (a, b) = (((i * 7 + j * 3) % 11) as f64 - 5.0, if i == j { 0.0 } else { ((i * 5 + j) % 7) as f64 - 3.0 });
            (re[(i, j)], re[(j, i)], im[(i, j)], im[(j, i)]) = (a, a, b, -b);
        }
    }
    let h = SplitComplexMatrix::from_parts(re, im);
    let (mut values, mut vectors, mut work) = (vec![0.0; n], SplitComplexMatrix::new(n, n), vec![0.0; hermitian_eigenvectors_work_len(n)]);
    assert!(h.eigenvectors_prealloc(&mut values, &mut vectors, &mut work));
    let count = allocations(|| assert!(h.eigenvectors_prealloc(&mut values, &mut vectors, &mut work)));
    assert_eq!(count, 0, "eigenvectors_prealloc allocated {count} times");
    let count = allocations(|| assert!(h.eigenvalues_prealloc(&mut values, &mut work)));
    assert_eq!(count, 0, "eigenvalues_prealloc allocated {count} times");
}

#[test]
fn the_picture_change_integrals_allocate_nothing_after_the_first_call() {
    // The p.Vp kernels' Hermite R-tensor slices come from the scratch (they were a heap allocation per shell pair),
    // for point charges (the nuclei) and for external multipoles (a dipole and a quadrupole site).
    let mut ctx = context("water_ano_pc", WATER);
    ctx.set_primitive_shell_pairs(None).unwrap();
    let xfield = common::temp_path("hot_paths_pc.xfield");
    std::fs::write(&xfield, "2 Bohr 2\n3.0 1.0 -2.0 0.4 0.1 -0.2 0.3 0.05 0.0 0.1 -0.1 0.0 0.05\n-2.5 0.5 1.5 -0.3 0.2 0.1 0.0 0.0 0.1 0.0 0.0 0.0 -0.1\n").unwrap();
    ctx.set_external_multipoles(xfield.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&xfield);
    let n = ctx.get_n_prim_unique();
    let (mut scalar, mut spin_orbit) = (vec![0.0; n * n], vec![0.0; 4 * n * n]);
    for (label, external) in [("nuclear attraction", false), ("external multipoles", true)] {
        let operator = || if external { ctx.get_external_multipole_operator().unwrap() } else { ctx.get_nuclear_attraction_operator() };
        let mut both = || {
            ctx.compute_one_electron_picture_change_integrals(&mut scalar, operator(), false).unwrap();
            ctx.compute_one_electron_picture_change_integrals(&mut spin_orbit, operator(), true).unwrap();
        };
        both();
        let count = allocations(&mut both);
        eprintln!("picture change, {label}: {count} allocations");
        assert_eq!(count, 0, "picture change, {label}: {count} allocations");
    }
}

/// The first AO of the shell that starts nearest the middle of the basis (a batch of sigma's start).
fn infos_first_ao_at_half(ctx: &aedelsten::AedelstenContext) -> usize {
    let mut infos = vec![aedelsten::ShellInfo::default(); ctx.get_n_shells()];
    ctx.get_shells(&mut infos).unwrap();
    let n = ctx.get_n_cont();
    infos.iter().map(|shell| shell.first_ao).filter(|&f| f > 0).min_by_key(|&f| f.abs_diff(n / 2)).expect("more than one shell")
}

/// The induction, the electronic field at the sites and the operator of their dipoles, over sites not kept (the
/// density contracted in the Hermite basis, the dipoles' external-multipole operator) and over kept ones: the second
/// call allocates nothing.
#[test]
fn the_induction_allocates_nothing_after_the_first_call() {
    use aedelsten::{PolarizationOptions, SolverOptions};
    let sites = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/polarization_demo.xfield");
    for (name, resident) in [("water_ano_induction_evaluated", None), ("water_ano_induction_kept", Some((1, arrays::scratch::ByteUnit::GiB)))] {
        let mut ctx = context(name, WATER);
        ctx.set_polarizable_sites(sites).unwrap();
        ctx.set_polarization(PolarizationOptions { solver: SolverOptions { threshold: 1e-10, ..SolverOptions::default() }, resident_field_budget: resident, ..PolarizationOptions::default() }).unwrap();
        let n = ctx.get_n_cont();
        let p = density(n);
        let mut v = vec![0.0; n * n];
        ctx.compute_induction(p.get_data(), &mut v).unwrap();
        let count = allocations(|| {
            ctx.compute_induction(p.get_data(), &mut v).unwrap();
        });
        assert_eq!(count, 0, "{name}: the induction allocated {count} times");
    }
}
