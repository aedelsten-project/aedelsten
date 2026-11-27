// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points for point-group symmetry (src/ffi/point_group.rs), driven the way a Fortran/C/Julia caller
// would: set up, query the group and its irreps, clean matrices and orbitals, free.

// The Rust API is used only as the reference the C calls are compared with.
mod common;

use std::ffi::{CStr, CString, c_char, c_void};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn set_shell_pairs_c(ctx: *mut c_void, overlap_threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut c_void, size: i32, unit: i32) -> i32;
    fn get_n_cont_c(ctx: *const c_void) -> i32;
    fn aedelsten_query_f64_c(key: *const c_char) -> f64;

    fn set_symmetry_c(ctx: *mut c_void, point_group: *const c_char, principal_axis: *const f64, geometry_tolerance: f64, strict_tolerance: f64, average_warning: f64) -> i32;
    fn free_symmetry_c(ctx: *mut c_void);
    fn get_point_group_name_c(ctx: *const c_void, buf: *mut c_char, buf_len: i32) -> i32;
    fn get_n_irreps_c(ctx: *const c_void) -> i32;
    fn get_irrep_name_c(ctx: *const c_void, irrep: i32, buf: *mut c_char, buf_len: i32) -> i32;
    fn get_irrep_dimension_c(ctx: *const c_void, irrep: i32) -> i32;
    fn get_irrep_index_c(ctx: *const c_void, name: *const c_char) -> i32;
    fn get_totally_symmetric_irrep_c(ctx: *const c_void) -> i32;
    fn get_cartesian_irrep_c(ctx: *const c_void, lx: i32, ly: i32, lz: i32) -> i32;
    fn get_symmetry_standard_axes_c(ctx: *const c_void, axes: *mut f64) -> i32;
    fn get_symmetry_operation_count_c(ctx: *const c_void) -> i32;
    fn get_symmetry_operation_matrix_c(ctx: *const c_void, op: i32, matrix: *mut f64) -> i32;
    fn get_n_symmetry_centres_c(ctx: *const c_void) -> i32;
    fn get_symmetry_centre_images_c(ctx: *const c_void, op: i32, images: *mut i32, n: i32) -> i32;
    fn symmetrise_matrix_c(ctx: *const c_void, matrix: *mut f64, len: i32, average: i32, report: *mut f64) -> i32;
    fn analyse_matrix_c(ctx: *const c_void, matrix: *const f64, len: i32, irrep: i32, report: *mut f64) -> i32;
    fn get_irrep_norms_c(ctx: *const c_void, matrix: *const f64, len: i32, norms: *mut f64, n_norms: i32) -> i32;
    fn get_symmetry_options_c(ctx: *const c_void, options: *mut f64) -> i32;
    fn get_symmetrisation_report_c(ctx: *const c_void, name: *const c_char, primitive: i32, found: *mut i32, report: *mut f64) -> i32;
    fn set_primitive_shell_pairs_c(ctx: *mut c_void, overlap_threshold: f64) -> i32;
    fn get_n_prim_unique_c(ctx: *const c_void) -> i32;
    fn compute_one_electron_integrals_c(
        ctx: *mut c_void, output: *mut f64, n: i32, operator: i32, n_planes: i32, x: f64, y: f64, z: f64, order: i32, primitive: i32,
    ) -> i32;
    fn compute_one_electron_picture_change_integrals_c(
        ctx: *mut c_void, output: *mut f64, n: i32, operator: i32, n_planes: i32, x: f64, y: f64, z: f64, order: i32, spin_orbit: i32,
    ) -> i32;
    fn construct_x2c_one_electron_hamiltonian_c(ctx: *mut c_void, oneham: *mut f64, s: *const f64, t: *const f64, v: *const f64, pvp: *const f64, n: i32, c_light: f64) -> i32;
    fn has_picture_change_c(ctx: *const c_void) -> i32;
    fn get_picture_change_transforms_c(ctx: *const c_void, u_large: *mut f64, u_small: *mut f64, len: i32) -> i32;
    fn free_picture_change_c(ctx: *mut c_void);
    fn symmetrise_orbitals_c(ctx: *const c_void, orbitals: *mut f64, len: i32, n_mo: i32, irreps: *mut i32, components: *mut i32, dropped: *mut f64) -> i32;
}

// Ethylene, D2h in libmsym's standard orientation.
const ETHYLENE: &str = "Basis set\n    C.x2c-SVPall\n    C1 0.0 0.0 1.26\n    C2 0.0 0.0 -1.26\nEnd of Basis\n\
                        Basis set\n    H.x2c-SVPall\n    H1 0.0 1.75 2.33\n    H2 0.0 -1.75 2.33\n    H3 0.0 1.75 -2.33\n    H4 0.0 -1.75 -2.33\nEnd of Basis\n";

fn context() -> *mut c_void {
    let path = common::temp_path("ffi_symmetry.inp");
    std::fs::write(&path, ETHYLENE).unwrap();
    let name = CString::new(path.to_str().unwrap()).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(name.as_ptr(), 0, aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    std::fs::remove_file(&path).ok();
    assert!(!ctx.is_null());
    assert_eq!(unsafe { set_shell_pairs_c(ctx, -1.0) }, 0);
    assert_eq!(unsafe { set_scratch_c(ctx, 64, 1) }, 0);
    ctx
}

fn name_of(get: impl FnOnce(*mut c_char, i32) -> i32) -> String {
    let mut buf = [0 as c_char; 16];
    assert_eq!(get(buf.as_mut_ptr(), buf.len() as i32), 0);
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap().to_string()
}

#[test]
fn symmetry_through_the_c_interface() {
    let ctx = context();
    let n = unsafe { get_n_cont_c(ctx) };
    assert_eq!(unsafe { get_n_irreps_c(ctx) }, -1, "no symmetry before set_symmetry_c");

    // The full group, default tolerance.
    assert_eq!(unsafe { set_symmetry_c(ctx, std::ptr::null(), std::ptr::null(), -1.0, -1.0, -1.0) }, 0);
    assert_eq!(name_of(|b, l| unsafe { get_point_group_name_c(ctx, b, l) }), "D2h");
    let n_irreps = unsafe { get_n_irreps_c(ctx) };
    assert_eq!(n_irreps, 8);
    for k in 0..n_irreps {
        assert_eq!(unsafe { get_irrep_dimension_c(ctx, k) }, 1);
        let name = name_of(|b, l| unsafe { get_irrep_name_c(ctx, k, b, l) });
        assert_eq!(unsafe { get_irrep_index_c(ctx, CString::new(name).unwrap().as_ptr()) }, k);
    }
    let mut options = [0.0; 3];
    assert_eq!(unsafe { get_symmetry_options_c(ctx, options.as_mut_ptr()) }, 0);
    assert_eq!(options, [1e-8, 1e-10, 0.0]);
    let ag = unsafe { get_totally_symmetric_irrep_c(ctx) };
    assert_eq!(name_of(|b, l| unsafe { get_irrep_name_c(ctx, ag, b, l) }), "A1g");
    let z = unsafe { get_cartesian_irrep_c(ctx, 0, 0, 1) };
    assert_eq!(name_of(|b, l| unsafe { get_irrep_name_c(ctx, z, b, l) }), "B1u");
    let mut axes = [0.0; 9];
    assert_eq!(unsafe { get_symmetry_standard_axes_c(ctx, axes.as_mut_ptr()) }, 0);
    assert!((axes[8].abs() - 1.0).abs() < 1e-12, "standard z is the input's z: {axes:?}");
    // A too small name buffer and an irrep out of range are errors, not truncations.
    let mut tiny = [0 as c_char; 2];
    assert_ne!(unsafe { get_point_group_name_c(ctx, tiny.as_mut_ptr(), 2) }, 0);
    assert_eq!(unsafe { get_irrep_dimension_c(ctx, n_irreps) }, -1);

    // A symmetric matrix (a multiple of the identity, which every operation leaves alone) and a noisy one.
    let len = n * n;
    let mut identity = vec![0.0; len as usize];
    for i in 0..n as usize {
        identity[i * n as usize + i] = 2.0;
    }
    let mut report = [f64::NAN; 4];
    assert_eq!(unsafe { analyse_matrix_c(ctx, identity.as_ptr(), len, ag, report.as_mut_ptr()) }, 0);
    assert_eq!(report[0] as i32, ag);
    assert!(report[1] < 1e-12 && (report[3] - 2.0 * (n as f64).sqrt()).abs() < 1e-10, "{report:?}");
    // Strict (the default tolerance, 1e-10): refused, untouched, a null report fine; averaged: removed and reported.
    let mut noisy = identity.clone();
    noisy[1] += 1e-9;
    let before = noisy.clone();
    assert_ne!(unsafe { symmetrise_matrix_c(ctx, noisy.as_mut_ptr(), len, 0, std::ptr::null_mut()) }, 0);
    assert_eq!(noisy, before);
    assert_eq!(unsafe { symmetrise_matrix_c(ctx, noisy.as_mut_ptr(), len, 1, report.as_mut_ptr()) }, 0);
    assert!(report[1] > 1e-10 && report[0] as i32 == ag, "{report:?}");
    // The identity is symmetric: a strict clean accepts it; an unknown mode is refused.
    assert_eq!(unsafe { symmetrise_matrix_c(ctx, identity.as_mut_ptr(), len, 0, report.as_mut_ptr()) }, 0);
    assert_ne!(unsafe { symmetrise_matrix_c(ctx, identity.as_mut_ptr(), len, 2, report.as_mut_ptr()) }, 0);
    let mut norms = vec![0.0; n_irreps as usize];
    assert_eq!(unsafe { get_irrep_norms_c(ctx, identity.as_ptr(), len, norms.as_mut_ptr(), n_irreps) }, 0);
    assert!(norms.iter().enumerate().all(|(k, &w)| (k as i32 == ag) == (w > 1.0)), "{norms:?}");
    // Wrong sizes are refused.
    assert_ne!(unsafe { analyse_matrix_c(ctx, identity.as_ptr(), len - 1, ag, report.as_mut_ptr()) }, 0);

    // Orbitals: the identity (each AO as an MO) comes back projected onto irreps, labelled; any number of MOs (here
    // the first three).
    let n_mo = 3;
    let mut orbitals = vec![0.0; (n * n_mo) as usize];
    for (i, column) in orbitals.chunks_exact_mut(n as usize).enumerate() {
        column[i] = 1.0;
    }
    let (mut irreps, mut components) = (vec![-1; n_mo as usize], vec![-1; n_mo as usize]);
    let mut dropped = f64::NAN;
    let len = n * n_mo;
    assert_eq!(
        unsafe { symmetrise_orbitals_c(ctx, orbitals.as_mut_ptr(), len, n_mo, irreps.as_mut_ptr(), components.as_mut_ptr(), &mut dropped) },
        0
    );
    assert!(dropped.is_finite());
    assert!(irreps.iter().all(|&k| (0..n_irreps).contains(&k)) && components.iter().all(|&j| j == 0), "{irreps:?} {components:?}");
    assert_eq!(unsafe { symmetrise_orbitals_c(ctx, orbitals.as_mut_ptr(), len, n_mo, irreps.as_mut_ptr(), std::ptr::null_mut(), &mut dropped) }, 0);
    assert!(dropped < 1e-12, "a second pass drops nothing: {dropped:.3e}");

    // Choosing a subgroup by name and axis; then freeing.
    unsafe { free_symmetry_c(ctx) };
    assert_eq!(unsafe { get_n_irreps_c(ctx) }, -1);
    let c2v = CString::new("C2v").unwrap();
    let x_axis = [1.0, 0.0, 0.0];
    assert_eq!(unsafe { set_symmetry_c(ctx, c2v.as_ptr(), x_axis.as_ptr(), -1.0, -1.0, -1.0) }, 0);
    assert_eq!(name_of(|b, l| unsafe { get_point_group_name_c(ctx, b, l) }), "C2v");
    assert_eq!(unsafe { get_cartesian_irrep_c(ctx, 1, 0, 0) }, unsafe { get_totally_symmetric_irrep_c(ctx) }, "C2 along x: x is A1");
    let c3v = CString::new("C3v").unwrap();
    unsafe { free_symmetry_c(ctx) };
    assert_ne!(unsafe { set_symmetry_c(ctx, c3v.as_ptr(), std::ptr::null(), -1.0, -1.0, -1.0) }, 0);

    assert_eq!(unsafe { aedelsten_query_f64_c(c"default_symmetry_geometry_tolerance".as_ptr()) }, 1e-8);
    unsafe { free_aedelsten_context_c(ctx) };
}

// The library's own integrals cleaned with symmetry set, their reports, and X2C through the C interface: inputs from
// the library, outputs averaged and reported, the transforms copied out, the cache freed.
#[test]
fn library_cleaning_and_x2c_through_the_c_interface() {
    let ctx = context();
    assert_eq!(unsafe { set_symmetry_c(ctx, std::ptr::null(), std::ptr::null(), -1.0, -1.0, -1.0) }, 0);
    assert_eq!(unsafe { set_primitive_shell_pairs_c(ctx, -1.0) }, 0);
    let report_of = |name: &CStr, primitive: i32| {
        let (mut found, mut report) = (-1, [f64::NAN; 4]);
        assert_eq!(unsafe { get_symmetrisation_report_c(ctx, name.as_ptr(), primitive, &mut found, report.as_mut_ptr()) }, 0);
        (found == 1).then_some(report)
    };
    assert!(report_of(c"oneint_overlap", 1).is_none());
    let n = unsafe { get_n_prim_unique_c(ctx) };
    let len = (n * n) as usize;
    let integrals = |operator: i32, picture_change: bool| {
        let mut m = vec![0.0; len];
        let status = if picture_change {
            unsafe { compute_one_electron_picture_change_integrals_c(ctx, m.as_mut_ptr(), n, operator, 1, 0.0, 0.0, 0.0, 0, 0) }
        } else {
            unsafe { compute_one_electron_integrals_c(ctx, m.as_mut_ptr(), n, operator, 1, 0.0, 0.0, 0.0, 0, 1) }
        };
        assert_eq!(status, 0);
        m
    };
    let (s, t, v, pvp) = (integrals(0, false), integrals(1, false), integrals(2, false), integrals(2, true));
    let overlap = report_of(c"oneint_overlap", 1).expect("the primitive overlap was cleaned");
    assert!(overlap[1] < 1e-11, "{overlap:?}");
    assert!(report_of(c"oneint_overlap", 0).is_none(), "only the primitive one was computed");

    assert_eq!(unsafe { has_picture_change_c(ctx) }, 0);
    let mut oneham = vec![0.0; len];
    assert_eq!(unsafe { construct_x2c_one_electron_hamiltonian_c(ctx, oneham.as_mut_ptr(), s.as_ptr(), t.as_ptr(), v.as_ptr(), pvp.as_ptr(), n, -1.0) }, 0);
    assert_eq!(unsafe { has_picture_change_c(ctx) }, 1);
    for name in [c"x2c_one_electron_hamiltonian", c"x2c_u_large", c"x2c_u_small"] {
        let report = report_of(name, 1).expect("cleaned");
        assert!(report[1] < 1e-9 * report[3], "{name:?}: {report:?}");
    }
    let (mut u_large, mut u_small) = (vec![0.0; len], vec![0.0; len]);
    assert_eq!(unsafe { get_picture_change_transforms_c(ctx, u_large.as_mut_ptr(), u_small.as_mut_ptr(), len as i32) }, 0);
    assert!(u_large.iter().any(|&x| x != 0.0));
    assert_ne!(unsafe { get_picture_change_transforms_c(ctx, u_large.as_mut_ptr(), u_small.as_mut_ptr(), len as i32 - 1) }, 0);
    unsafe { free_picture_change_c(ctx) };
    assert_eq!(unsafe { has_picture_change_c(ctx) }, 0);
    unsafe { free_aedelsten_context_c(ctx) };
}

// The operations through C equal the Rust calls on the same input, bit for bit: the matrix column-major, the images
// 0-based; out-of-range operations and wrong lengths are errors.
#[test]
fn symmetry_operations_through_the_c_interface() {
    let ctx = context();
    assert_eq!(unsafe { get_symmetry_operation_count_c(ctx) }, -1, "no operations before set_symmetry_c");
    assert_eq!(unsafe { get_n_symmetry_centres_c(ctx) }, -1);
    assert_eq!(unsafe { set_symmetry_c(ctx, std::ptr::null(), std::ptr::null(), -1.0, -1.0, -1.0) }, 0);

    let path = common::temp_path("ffi_symmetry_rust.inp");
    std::fs::write(&path, ETHYLENE).unwrap();
    let mut rust = aedelsten::AedelstenContext::new(path.to_str().unwrap(), aedelsten::ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    rust.set_symmetry(aedelsten::PointGroupRequest::Highest, aedelsten::SymmetryOptions::default()).unwrap();

    let h = unsafe { get_symmetry_operation_count_c(ctx) };
    assert_eq!(h as usize, rust.get_symmetry_operation_count().unwrap());
    assert_eq!(h, 8);
    let n = unsafe { get_n_symmetry_centres_c(ctx) };
    assert_eq!(n as usize, rust.get_n_symmetry_centres().unwrap());
    assert_eq!(n, 6);
    for op in 0..h {
        let mut matrix = [f64::NAN; 9];
        assert_eq!(unsafe { get_symmetry_operation_matrix_c(ctx, op, matrix.as_mut_ptr()) }, 0);
        let expected = rust.get_symmetry_operation_matrix(op as usize).unwrap();
        for i in 0..3 {
            for k in 0..3 {
                assert_eq!(matrix[i + 3 * k].to_bits(), expected[i][k].to_bits(), "operation {op}, element ({i}, {k})");
            }
        }
        let mut images = [-1i32; 6];
        assert_eq!(unsafe { get_symmetry_centre_images_c(ctx, op, images.as_mut_ptr(), n) }, 0);
        let mut expected = [0usize; 6];
        rust.get_symmetry_centre_images(op as usize, &mut expected).unwrap();
        assert!(images.iter().zip(expected).all(|(&a, b)| a as usize == b), "operation {op}: {images:?} against {expected:?}");
    }
    let mut matrix = [0.0; 9];
    let mut images = [0i32; 6];
    assert_ne!(unsafe { get_symmetry_operation_matrix_c(ctx, h, matrix.as_mut_ptr()) }, 0);
    assert_ne!(unsafe { get_symmetry_operation_matrix_c(ctx, -1, matrix.as_mut_ptr()) }, 0);
    assert_ne!(unsafe { get_symmetry_centre_images_c(ctx, h, images.as_mut_ptr(), n) }, 0);
    assert_ne!(unsafe { get_symmetry_centre_images_c(ctx, 0, images.as_mut_ptr(), n - 1) }, 0);
    unsafe { free_aedelsten_context_c(ctx) };
}
