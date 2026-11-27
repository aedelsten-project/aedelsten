// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the centres (their count, geometry, kinds and nuclear charges) and of the inverse Cartesian
// rescaling, driven as a C, Fortran, Python or Julia caller would, against the Rust calls they wrap. Every kind of
// centre: a real atom with a Cartesian d shell, a ghost centre and an orthogonality ghost.

mod common;

use aedelsten::{AedelstenContext, AtomKind, CentreGeometry, ContractionType, SharedCartesianReference};
use std::ffi::{CString, c_char, c_void};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn aedelsten_query_i32_c(key: *const c_char) -> i32;
    fn get_n_centres_c(ctx: *const c_void) -> i64;
    fn get_centre_geometry_c(ctx: *const c_void, atomic_numbers: *mut i32, positions: *mut f64, kinds: *mut i32, n: i64) -> i32;
    fn get_nuclear_charges_c(ctx: *const c_void, charges: *mut f64, n: i64) -> i32;
    fn rescale_cartesian_ao_rows_c(ctx: *mut c_void, convention: i32, n_cols: i32, matrix: *mut f64, len: i32, rescaled: *mut i32) -> i32;
    fn unrescale_cartesian_ao_rows_c(ctx: *mut c_void, convention: i32, n_cols: i32, matrix: *mut f64, len: i32, rescaled: *mut i32) -> i32;
}

const INPUT: &str = "Basis set\n    O.x2c-SVPall\n    Cartesian d\n    O1 0.0 0.0 0.2\nEnd of Basis\n\
                     Basis set\n    H.x2c-SVPall\n    GHOST\n    H1 0.0 1.4 -0.9\nEnd of Basis\n\
                     Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    ORTHOGONALITY\n    Mg1 0.0 0.0 5.0\nEnd of Basis\n";

fn query(key: &str) -> i32 {
    let key = CString::new(key).unwrap();
    unsafe { aedelsten_query_i32_c(key.as_ptr()) }
}

#[test]
fn the_centres_are_the_rust_calls_answers() {
    let path = common::temp_path("ffi_centres.inp");
    std::fs::write(&path, INPUT).unwrap();
    let rust = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let file = CString::new(path.to_str().unwrap()).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(file.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    std::fs::remove_file(&path).ok();
    assert!(!ctx.is_null());

    let n = rust.get_n_centres();
    assert_eq!(n, 3);
    assert_eq!(unsafe { get_n_centres_c(ctx) }, n as i64);
    assert_eq!(unsafe { get_n_centres_c(std::ptr::null()) }, -1);

    let mut expected = vec![CentreGeometry { atomic_number: 0, position: [0.0; 3], kind: AtomKind::Real }; n];
    rust.get_centre_geometry(&mut expected).unwrap();
    let (mut z, mut xyz, mut kinds) = (vec![-1; n], vec![f64::NAN; 3 * n], vec![-1; n]);
    assert_eq!(unsafe { get_centre_geometry_c(ctx, z.as_mut_ptr(), xyz.as_mut_ptr(), kinds.as_mut_ptr(), n as i64) }, 0);
    for (k, centre) in expected.iter().enumerate() {
        assert_eq!(z[k], i32::from(centre.atomic_number), "centre {k}");
        assert_eq!(&xyz[3 * k..3 * k + 3], &centre.position, "centre {k}");
        assert_eq!(kinds[k], centre.kind as i32, "centre {k}");
    }
    // The kinds by name, as a caller reads them.
    assert_eq!(kinds, [query("atom_kind_real"), query("atom_kind_ghost"), query("atom_kind_orthogonality_ghost")]);
    assert_eq!((z[0], z[1], z[2]), (8, 1, 12));
    assert_ne!(unsafe { get_centre_geometry_c(ctx, z.as_mut_ptr(), xyz.as_mut_ptr(), kinds.as_mut_ptr(), n as i64 + 1) }, 0, "a wrong count is refused");

    let mut expected_charges = vec![0.0; n];
    rust.get_nuclear_charges(&mut expected_charges).unwrap();
    let mut charges = vec![f64::NAN; n];
    assert_eq!(unsafe { get_nuclear_charges_c(ctx, charges.as_mut_ptr(), n as i64) }, 0);
    assert_eq!(charges, expected_charges);
    assert_eq!(charges, [8.0, 0.0, 0.0], "the ghost and the orthogonality ghost have no nucleus");
    assert_ne!(unsafe { get_nuclear_charges_c(ctx, charges.as_mut_ptr(), n as i64 - 1) }, 0);

    // Rescaling and its inverse: a round trip, both reporting the Cartesian d shell, the inverse as in Rust.
    let n_ao = rust.get_n_cont();
    let original: Vec<f64> = (0..2 * n_ao).map(|i| 0.1 + i as f64).collect();
    let molcas = SharedCartesianReference::MOLCAS.get_c_int();
    let (mut matrix, mut flag) = (original.clone(), -1);
    assert_eq!(unsafe { rescale_cartesian_ao_rows_c(ctx, molcas, 2, matrix.as_mut_ptr(), matrix.len() as i32, &mut flag) }, 0);
    assert_eq!(flag, 1);
    let mut rust_back = matrix.clone();
    assert!(rust.unrescale_cartesian_ao_rows(SharedCartesianReference::MOLCAS, 2, &mut rust_back).unwrap());
    assert_eq!(unsafe { unrescale_cartesian_ao_rows_c(ctx, molcas, 2, matrix.as_mut_ptr(), matrix.len() as i32, &mut flag) }, 0);
    assert_eq!(flag, 1);
    assert_eq!(matrix, rust_back);
    assert!(matrix.iter().zip(&original).all(|(a, b)| (a - b).abs() < 1e-12 * b.abs()), "the round trip");
    assert_ne!(unsafe { unrescale_cartesian_ao_rows_c(ctx, 99, 2, matrix.as_mut_ptr(), matrix.len() as i32, &mut flag) }, 0, "an unknown convention");

    unsafe { free_aedelsten_context_c(ctx) };
}
