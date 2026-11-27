// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// What a geometry optimization of an embedded QM region reads of its environment and hands from one geometry to the
// next: the embedding AIMPs' sites (`get_embedding_aimp_sites`: element, position, charge, in input order, the
// orthogonality ghosts' AIMPs included) and the induced dipoles' start (`set_induced_dipoles`: the next solve starts
// from them, its solution unchanged within the solver's tolerance, and they are not taken for a solution before it).
// The C forms give the same, and refuse wrong sizes and overlapping outputs.

mod common;

use aedelsten::{AedelstenContext, ContractionType, EmbeddingAimpSite, PolarizationOptions, SolverOptions};

const WATER_IN_AIMPS: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.2217\nEnd of Basis\n\
    Basis set\n    H.x2c-SVPall\n    H1 0.0 1.4309 -0.8867\n    H2 0.0 -1.4309 -0.8867\nEnd of Basis\n\
    Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    orthogonality\n    Na1 2.5 -1.0 3.0\nEnd of Basis\n\
    Basis set\n    O.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    F1 -3.0 0.5 1.0\n    F2 0.5 4.0 -2.5\nEnd of Basis\n";

fn context(input: &str, tag: &str) -> AedelstenContext {
    let path = common::temp_path(&format!("optimization_environment_{tag}.inp"));
    std::fs::write(&path, input).unwrap();
    let ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx
}

const EXPECTED: [EmbeddingAimpSite; 3] = [
    EmbeddingAimpSite { atomic_number: 12, position: [2.5, -1.0, 3.0], charge: 2.0 },
    EmbeddingAimpSite { atomic_number: 8, position: [-3.0, 0.5, 1.0], charge: -2.0 },
    EmbeddingAimpSite { atomic_number: 8, position: [0.5, 4.0, -2.5], charge: -2.0 },
];

#[test]
fn the_embedding_aimps_sites_are_their_elements_positions_and_charges_in_input_order() {
    let ctx = context(WATER_IN_AIMPS, "sites");
    assert_eq!(ctx.get_n_embedding_aimps(), 3);
    let mut sites = [EmbeddingAimpSite { atomic_number: 0, position: [0.0; 3], charge: 0.0 }; 3];
    ctx.get_embedding_aimp_sites(&mut sites).unwrap();
    assert_eq!(sites, EXPECTED);
    let mut short = [EmbeddingAimpSite { atomic_number: 0, position: [0.0; 3], charge: 0.0 }; 2];
    assert!(ctx.get_embedding_aimp_sites(&mut short).is_err());
}

#[test]
fn without_embedding_aimps_there_are_no_sites() {
    let ctx = context("Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\n    H2 0.0 0.0 1.4\nEnd of Basis\n", "none");
    assert_eq!(ctx.get_n_embedding_aimps(), 0);
    ctx.get_embedding_aimp_sites(&mut []).unwrap();
}

/// Water with three polarizable sites (the demos' fixture), the dipoles solved to 1e-12.
fn polarized(tag: &str) -> AedelstenContext {
    let mut ctx = context(
        "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.2217\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4309 -0.8867\n    H2 0.0 -1.4309 -0.8867\nEnd of Basis\n",
        tag,
    );
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_polarizable_sites(concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/polarization_demo.xfield")).unwrap();
    ctx.set_polarization(PolarizationOptions { solver: SolverOptions { threshold: 1e-12, ..SolverOptions::default() }, ..PolarizationOptions::default() }).unwrap();
    ctx
}

fn density(n: usize) -> Vec<f64> {
    (0..n * n).map(|k| if k % (n + 1) == 0 { 0.4 } else { 0.03 / (1.0 + (k % n + k / n) as f64) }).collect()
}

#[test]
fn a_given_start_shortens_the_solve_and_leaves_its_solution() {
    let fresh = polarized("fresh");
    let n = fresh.get_n_cont();
    let d = density(n);
    let mut v_fresh = vec![0.0; n * n];
    let fresh_report = fresh.compute_induction(&d, &mut v_fresh).unwrap();
    let mut solution = vec![0.0; 9];
    fresh.get_induced_dipoles(&mut solution).unwrap();

    let mut started = polarized("started");
    started.set_induced_dipoles(&solution).unwrap();
    // A start is not a solution.
    assert!(started.get_induced_dipoles(&mut [0.0; 9]).is_err());
    let mut v = vec![0.0; n * n];
    let report = started.compute_induction(&d, &mut v).unwrap();
    assert!(report.iterations < fresh_report.iterations, "{} iterations from the solution, {} from the static field", report.iterations, fresh_report.iterations);
    assert!((report.energy - fresh_report.energy).abs() < 1e-10, "{} against {}", report.energy, fresh_report.energy);
    let scale = v_fresh.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(v.iter().zip(&v_fresh).all(|(a, b)| (a - b).abs() < 1e-9 * scale));

    assert!(started.set_induced_dipoles(&solution[..8]).is_err());
    assert!(started.set_induced_dipoles(&[f64::NAN; 9]).is_err());
}

unsafe extern "C" {
    fn get_n_embedding_aimps_c(ctx: *const std::ffi::c_void) -> i64;
    fn get_embedding_aimp_sites_c(ctx: *const std::ffi::c_void, atomic_numbers: *mut i32, positions: *mut f64, charges: *mut f64, n: i64) -> i32;
    fn set_induced_dipoles_c(ctx: *mut std::ffi::c_void, dipoles: *const f64, length: i64) -> i32;
    fn get_induced_dipoles_c(ctx: *const std::ffi::c_void, dipoles: *mut f64, length: i64) -> i32;
}

#[test]
fn the_c_forms_give_what_the_context_gives() {
    let mut ctx = context(WATER_IN_AIMPS, "c_sites");
    let handle = &mut ctx as *mut AedelstenContext as *mut std::ffi::c_void;
    assert_eq!(unsafe { get_n_embedding_aimps_c(handle) }, 3);
    let (mut z, mut xyz, mut q) = ([0i32; 3], [0.0; 9], [0.0; 3]);
    assert_eq!(unsafe { get_embedding_aimp_sites_c(handle, z.as_mut_ptr(), xyz.as_mut_ptr(), q.as_mut_ptr(), 3) }, 0);
    for (k, site) in EXPECTED.iter().enumerate() {
        assert_eq!(z[k], i32::from(site.atomic_number));
        assert_eq!(&xyz[3 * k..3 * k + 3], &site.position);
        assert_eq!(q[k], site.charge);
    }
    assert_ne!(unsafe { get_embedding_aimp_sites_c(handle, z.as_mut_ptr(), xyz.as_mut_ptr(), q.as_mut_ptr(), 2) }, 0);
    // The charges written over the positions.
    assert_ne!(unsafe { get_embedding_aimp_sites_c(handle, z.as_mut_ptr(), xyz.as_mut_ptr(), xyz.as_mut_ptr().add(6), 3) }, 0);

    let fresh = polarized("c_fresh");
    let n = fresh.get_n_cont();
    let d = density(n);
    fresh.compute_induction(&d, &mut vec![0.0; n * n]).unwrap();
    let mut solution = [0.0; 9];
    fresh.get_induced_dipoles(&mut solution).unwrap();
    let mut by_rust = polarized("c_rust");
    by_rust.set_induced_dipoles(&solution).unwrap();
    let rust_report = by_rust.compute_induction(&d, &mut vec![0.0; n * n]).unwrap();
    let mut by_c = polarized("c_c");
    let handle = &mut by_c as *mut AedelstenContext as *mut std::ffi::c_void;
    assert_eq!(unsafe { set_induced_dipoles_c(handle, solution.as_ptr(), 9) }, 0);
    assert_ne!(unsafe { get_induced_dipoles_c(handle, [0.0; 9].as_mut_ptr(), 9) }, 0);
    assert_ne!(unsafe { set_induced_dipoles_c(handle, solution.as_ptr(), 8) }, 0);
    let c_report = by_c.compute_induction(&d, &mut vec![0.0; n * n]).unwrap();
    assert_eq!((c_report.iterations, c_report.energy), (rust_report.iterations, rust_report.energy));
}
