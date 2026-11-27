// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the group's action (the AO irrep counts, an operation applied to orbitals, the characters
// and the irrep matrices), driven as a C, Fortran, Python or Julia caller would, against the Rust calls they wrap.
// Ammonia (C3v, tilted off the axes), so that a two-dimensional irrep (E) has its matrices; the characters obey the
// orthogonality relation and the matrices' traces are the characters; the refusals of an irrep, an operation or a
// size out of range.

use aedelsten::{AedelstenContext, ContractionType, PointGroupRequest, SymmetryOptions};
use std::ffi::{CString, c_char, c_void};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn set_shell_pairs_c(ctx: *mut c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut c_void, size: i32, unit: i32) -> i32;
    fn set_symmetry_c(ctx: *mut c_void, point_group: *const c_char, principal_axis: *const f64, geometry_tolerance: f64, strict_tolerance: f64, average_warning: f64) -> i32;
    fn get_ao_irrep_counts_c(ctx: *const c_void, counts: *mut i32, n: i32) -> i32;
    fn apply_symmetry_operation_c(ctx: *const c_void, op: i32, coefficients: *const f64, out: *mut f64, len: i32, n_mo: i32) -> i32;
    fn get_irrep_character_c(ctx: *const c_void, irrep: i32, op: i32, character: *mut f64) -> i32;
    fn get_irrep_matrix_c(ctx: *const c_void, irrep: i32, op: i32, matrix: *mut f64, len: i32, aligned: *mut i32) -> i32;
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/symmetry_demo.inp");

#[test]
fn the_groups_action_is_the_rust_calls_answer() {
    let mut rust = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    rust.set_shell_pairs(None).unwrap();
    rust.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    rust.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let file = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(file.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());
    assert_eq!(unsafe { set_shell_pairs_c(ctx, -1.0) }, 0);
    assert_eq!(unsafe { set_scratch_c(ctx, 64, 1) }, 0);
    assert_eq!(unsafe { set_symmetry_c(ctx, std::ptr::null(), std::ptr::null(), -1.0, -1.0, -1.0) }, 0);
    assert_eq!(rust.get_point_group_name().unwrap(), "C3v");

    let n = rust.get_n_cont();
    let irreps = rust.get_irreps().unwrap().to_vec();
    let h = rust.get_symmetry_operation_count().unwrap();
    let n_irreps = irreps.len();

    // The AO counts: the Rust call's, together every AO.
    let expected: Vec<i32> = rust.get_ao_irrep_counts().unwrap().iter().map(|&c| c as i32).collect();
    let mut counts = vec![-1; n_irreps];
    assert_eq!(unsafe { get_ao_irrep_counts_c(ctx, counts.as_mut_ptr(), n_irreps as i32) }, 0);
    assert_eq!(counts, expected);
    assert_eq!(counts.iter().sum::<i32>() as usize, n, "every AO in one irrep");
    assert_ne!(unsafe { get_ao_irrep_counts_c(ctx, counts.as_mut_ptr(), n_irreps as i32 + 1) }, 0, "a wrong count");

    // Every operation on three model orbitals: the Rust call's answer.
    let n_mo = 3;
    let c: Vec<f64> = (0..n * n_mo).map(|k| 0.1 * ((k * 7 % 13) as f64) - 0.6).collect();
    for op in 0..h {
        let mut want = vec![0.0; n * n_mo];
        rust.apply_symmetry_operation(op, &c, n_mo, &mut want).unwrap();
        let mut got = vec![f64::NAN; n * n_mo];
        assert_eq!(unsafe { apply_symmetry_operation_c(ctx, op as i32, c.as_ptr(), got.as_mut_ptr(), got.len() as i32, n_mo as i32) }, 0);
        assert_eq!(got, want, "operation {op}");
    }
    let mut out = vec![0.0; n * n_mo];
    assert_ne!(unsafe { apply_symmetry_operation_c(ctx, h as i32, c.as_ptr(), out.as_mut_ptr(), out.len() as i32, n_mo as i32) }, 0, "an operation out of range");
    assert_ne!(unsafe { apply_symmetry_operation_c(ctx, 0, c.as_ptr(), out.as_mut_ptr(), out.len() as i32 - 1, n_mo as i32) }, 0, "a wrong size");

    // The characters (the Rust call's; orthonormal over the group) and the irrep matrices (the Rust call's; their
    // traces the characters, where the irrep's partners are aligned).
    let mut characters = vec![vec![0.0; h]; n_irreps];
    let mut has_two_dimensional = false;
    for (irrep, info) in irreps.iter().enumerate() {
        let d = info.dimension;
        for op in 0..h {
            let mut chi = f64::NAN;
            assert_eq!(unsafe { get_irrep_character_c(ctx, irrep as i32, op as i32, &mut chi) }, 0);
            assert_eq!(chi, rust.get_irrep_character(irrep, op).unwrap());
            characters[irrep][op] = chi;

            let mut want = vec![0.0; d * d];
            let aligned_rust = rust.get_irrep_matrix(irrep, op, &mut want).unwrap();
            let (mut matrix, mut aligned) = (vec![f64::NAN; d * d], -1);
            assert_eq!(unsafe { get_irrep_matrix_c(ctx, irrep as i32, op as i32, matrix.as_mut_ptr(), (d * d) as i32, &mut aligned) }, 0);
            assert_eq!(aligned, aligned_rust as i32);
            if aligned_rust {
                assert_eq!(matrix, want);
                let trace: f64 = (0..d).map(|k| matrix[k + d * k]).sum();
                assert!((trace - chi).abs() < 1e-12, "irrep {irrep}, operation {op}: trace {trace}, character {chi}");
                has_two_dimensional |= d == 2;
            }
        }
        assert_eq!(characters[irrep][0], d as f64, "the identity's character is the dimension");
    }
    assert!(has_two_dimensional, "C3v's E has its matrices");
    for i in 0..n_irreps {
        for j in 0..n_irreps {
            let product: f64 = (0..h).map(|op| characters[i][op] * characters[j][op]).sum::<f64>() / h as f64;
            assert!((product - f64::from(i == j)).abs() < 1e-12, "characters {i}, {j}: {product}");
        }
    }

    let (mut chi, mut aligned, mut matrix) = (0.0, 0, vec![0.0; 4]);
    assert_ne!(unsafe { get_irrep_character_c(ctx, n_irreps as i32, 0, &mut chi) }, 0, "an irrep out of range");
    assert_ne!(unsafe { get_irrep_character_c(ctx, 0, h as i32, &mut chi) }, 0, "an operation out of range");
    assert_ne!(unsafe { get_irrep_matrix_c(ctx, n_irreps as i32, 0, matrix.as_mut_ptr(), 4, &mut aligned) }, 0, "an irrep out of range");
    let e = irreps.iter().position(|i| i.dimension == 2).unwrap();
    assert_ne!(unsafe { get_irrep_matrix_c(ctx, e as i32, 0, matrix.as_mut_ptr(), 1, &mut aligned) }, 0, "room for 1 x 1, E is 2 x 2");

    unsafe { free_aedelsten_context_c(ctx) };
}
