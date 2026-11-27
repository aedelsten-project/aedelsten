// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the induced-dipole embedding, driven the way a Fortran/C/Julia caller would (raw
// pointers, flat buffers, sentinel options) and compared to the Rust API they wrap.

mod common;

use aedelsten::{AedelstenContext, ContractionType, PolarizationOptions, SiteChunking, SolverOptions};
use std::ffi::{CString, c_char};

#[repr(C)]
#[derive(Clone, Copy)]
struct Options {
    damping: i32,
    thole_a: f64,
    threshold: f64,
    max_iterations: i32,
    static_field: i32,
    chunk_sites: i32,
    chunk_budget: i32,
    chunk_budget_unit: i32,
    response_batch: i32,
    state_slots: i32,
    resident_budget: i32,
    resident_budget_unit: i32,
}

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn set_shell_pairs_c(ctx: *mut std::ffi::c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut std::ffi::c_void, size: i32, unit: i32) -> i32;
    fn set_polarizable_sites_c(ctx: *mut std::ffi::c_void, filename: *const c_char) -> i32;
    fn get_n_polarizable_sites_c(ctx: *const std::ffi::c_void) -> i64;
    fn aedelsten_polarization_options_default_c(options: *mut Options);
    fn set_polarization_c(ctx: *mut std::ffi::c_void, options: *const Options) -> i32;
    fn free_polarization_c(ctx: *mut std::ffi::c_void);
    fn compute_induction_c(ctx: *const std::ffi::c_void, d: *const f64, v: *mut f64, e: *mut f64, it: *mut i32) -> i32;
    fn compute_induction_within_c(ctx: *const std::ffi::c_void, d: *const f64, v: *mut f64, threshold: f64, e: *mut f64, it: *mut i32) -> i32;
    fn get_polarization_storage_c(ctx: *const std::ffi::c_void, storage: *mut i64) -> i32;
    fn set_resident_field_budget_c(ctx: *mut std::ffi::c_void, budget: i32, unit: i32) -> i32;
    fn compute_induction_response_c(ctx: *const std::ffi::c_void, d: *const f64, n: i32, v: *mut f64) -> i32;
    fn compute_induction_couplings_c(ctx: *const std::ffi::c_void, d: *const f64, n: i32, k: *mut f64) -> i32;
    fn compute_induction_states_within_c(ctx: *const std::ffi::c_void, d: *const f64, n: i32, v: *mut f64, e: *mut f64, threshold: f64) -> i32;
    fn get_induced_dipoles_c(ctx: *const std::ffi::c_void, mu: *mut f64, len: i64) -> i32;
    fn compute_effective_external_field_c(ctx: *const std::ffi::c_void, v: *mut f64) -> i32;
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/fock_builder_test.inp");
const SITES: &str = "3 Bohr -1 2 1\n1 6.0 0.0 0.0 2 0 0 2 0 2\n2 -5.0 4.0 4.0 1.5 0 0 1.8 0 1.2\n0 0.0 -6.0 3.0 2.2 0.3 0 2.0 0 1.6\n";

#[test]
fn the_c_interface_reproduces_the_rust_api() {
    let path = common::temp_path("ffi_pol.xfield");
    std::fs::write(&path, SITES).unwrap();
    let inp = CString::new(INP).unwrap();
    let xfield = CString::new(path.to_str().unwrap()).unwrap();

    let mut reference = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    reference.set_shell_pairs(None).unwrap();
    reference.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    reference.set_polarizable_sites(path.to_str().unwrap()).unwrap();
    let n = reference.get_n_cont();
    let density: Vec<f64> = (0..n * n).map(|i| 0.02 / (1.0 + (i % n) as f64 + (i / n) as f64)).collect();
    reference
        .set_polarization(PolarizationOptions {
            solver: SolverOptions { threshold: 1e-12, ..SolverOptions::default() },
            chunking: SiteChunking::Sites(2),
            response_batch: 2,
            state_slots: 1,
            ..PolarizationOptions::default()
        })
        .unwrap();
    let mut v_ref = vec![0.0; n * n];
    let report = reference.compute_induction(&density, &mut v_ref).unwrap();
    let mut r_ref = vec![0.0; 2 * n * n];
    let two: Vec<f64> = density.iter().chain(density.iter().map(|d| d * 3.0).collect::<Vec<_>>().iter()).copied().collect();
    reference.compute_induction_response(&two, 2, &mut r_ref).unwrap();
    let mut k_ref = vec![0.0; 4];
    reference.compute_induction_couplings(&two, 2, &mut k_ref).unwrap();
    let mut eef_ref = vec![0.0; 3 * n * n];
    reference.compute_effective_external_field(&mut eef_ref).unwrap();
    let mut loose_ref = vec![0.0; n * n];
    let loose_report = reference.compute_induction_within(&density, &mut loose_ref, 1e-4).unwrap();
    let storage = reference.get_polarization_storage().unwrap();
    let (mut state_ref, mut state_energy_ref) = (vec![0.0; n * n], [0.0]);
    reference.compute_induction_states_within(&density, 1, &mut state_ref, &mut state_energy_ref, 1e-4).unwrap();
    assert!(storage.sites_in_memory == 0 && storage.bytes_to_keep > 0);

    unsafe {
        let ctx = set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL);
        assert!(!ctx.is_null());
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 64, 1), 0);
        // Calls out of order are errors, not crashes.
        assert_ne!(set_polarization_c(ctx, std::ptr::null()), 0);
        assert_eq!(set_polarizable_sites_c(ctx, xfield.as_ptr()), 0);
        assert_eq!(get_n_polarizable_sites_c(ctx), 3);

        let mut options = std::mem::zeroed::<Options>();
        aedelsten_polarization_options_default_c(&mut options);
        options.threshold = 1e-12;
        options.chunk_sites = 2;
        options.response_batch = 2;
        options.state_slots = 1;
        let mut bad = options;
        bad.damping = 7;
        assert_ne!(set_polarization_c(ctx, &bad), 0);
        assert_eq!(set_polarization_c(ctx, &options), 0);

        let mut v = vec![0.0; n * n];
        let (mut energy, mut iterations) = (0.0, 0);
        assert_eq!(compute_induction_c(ctx, density.as_ptr(), v.as_mut_ptr(), &mut energy, &mut iterations), 0);
        assert_eq!(energy, report.energy);
        assert!(v.iter().zip(&v_ref).all(|(a, b)| a == b));

        let mut mu = vec![0.0; 9];
        assert_eq!(get_induced_dipoles_c(ctx, mu.as_mut_ptr(), 9), 0);
        assert_ne!(get_induced_dipoles_c(ctx, mu.as_mut_ptr(), 8), 0);

        let mut r = vec![0.0; 2 * n * n];
        assert_eq!(compute_induction_response_c(ctx, two.as_ptr(), 2, r.as_mut_ptr()), 0);
        assert!(r.iter().zip(&r_ref).all(|(a, b)| a == b));

        let mut k = vec![0.0; 4];
        assert_eq!(compute_induction_couplings_c(ctx, two.as_ptr(), 2, k.as_mut_ptr()), 0);
        assert_eq!(k, k_ref);
        assert_ne!(compute_induction_couplings_c(ctx, two.as_ptr(), -1, k.as_mut_ptr()), 0);
        assert_ne!(compute_induction_couplings_c(ctx, two.as_ptr(), 2, std::ptr::null_mut()), 0);
        assert_ne!(compute_induction_couplings_c(ctx, two.as_ptr(), 2, two.as_ptr() as *mut f64), 0);

        let mut eef = vec![0.0; 3 * n * n];
        assert_eq!(compute_effective_external_field_c(ctx, eef.as_mut_ptr()), 0);
        assert!(eef.iter().zip(&eef_ref).all(|(a, b)| a == b));
        assert_ne!(compute_effective_external_field_c(ctx, std::ptr::null_mut()), 0);

        let mut loose = vec![0.0; n * n];
        assert_eq!(compute_induction_within_c(ctx, density.as_ptr(), loose.as_mut_ptr(), 1e-4, &mut energy, &mut iterations), 0);
        assert_eq!((energy, iterations as usize), (loose_report.energy, loose_report.iterations));
        assert!(loose.iter().zip(&loose_ref).all(|(a, b)| a == b));
        assert_ne!(compute_induction_within_c(ctx, density.as_ptr(), loose.as_mut_ptr(), 0.0, std::ptr::null_mut(), std::ptr::null_mut()), 0);
        assert_ne!(compute_induction_within_c(ctx, density.as_ptr(), density.as_ptr() as *mut f64, 1e-4, std::ptr::null_mut(), std::ptr::null_mut()), 0);

        let (mut state_ops, mut state_energy) = (vec![0.0; n * n], [0.0]);
        assert_eq!(compute_induction_states_within_c(ctx, density.as_ptr(), 1, state_ops.as_mut_ptr(), state_energy.as_mut_ptr(), 1e-4), 0);
        assert_eq!(state_energy, state_energy_ref);
        assert!(state_ops.iter().zip(&state_ref).all(|(a, b)| a == b));
        assert_ne!(compute_induction_states_within_c(ctx, density.as_ptr(), 1, state_ops.as_mut_ptr(), state_energy.as_mut_ptr(), -1.0), 0);

        let mut values = [-1i64; 4];
        assert_eq!(get_polarization_storage_c(ctx, values.as_mut_ptr()), 0);
        assert_eq!(values.map(|v| v as usize), [storage.sites, 0, storage.bytes_per_site, storage.bytes_to_keep]);
        assert_ne!(get_polarization_storage_c(ctx, std::ptr::null_mut()), 0);

        // A budget set afterwards (one GiB holds them all), then none: the same induction as the Rust call's, the
        // storage says so; a negative budget and an unknown unit are refused.
        assert_ne!(set_resident_field_budget_c(ctx, -1, 0), 0);
        assert_ne!(set_resident_field_budget_c(ctx, 1, 9), 0);
        for (budget, kept) in [(1, storage.sites), (0, 0)] {
            assert_eq!(set_resident_field_budget_c(ctx, budget, 0), 0);
            assert_eq!(get_polarization_storage_c(ctx, values.as_mut_ptr()), 0);
            assert_eq!(values[1] as usize, kept);
            assert_eq!(compute_induction_c(ctx, density.as_ptr(), v.as_mut_ptr(), &mut energy, std::ptr::null_mut()), 0);
            assert!((energy - report.energy).abs() < 1e-12 * report.energy.abs());
            assert!(v.iter().zip(&v_ref).all(|(a, b)| (a - b).abs() < 1e-12));
        }

        free_polarization_c(ctx);
        assert_ne!(compute_induction_c(ctx, density.as_ptr(), v.as_mut_ptr(), std::ptr::null_mut(), std::ptr::null_mut()), 0);
        assert_ne!(get_polarization_storage_c(ctx, values.as_mut_ptr()), 0);
        free_aedelsten_context_c(ctx);
    }
    std::fs::remove_file(&path).ok();
}
