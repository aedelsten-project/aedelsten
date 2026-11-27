// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry point of the overlap between two contexts' bases (compute_overlap_with_c), driven as a C, Fortran,
// Python or Julia caller would, against the Rust call it wraps: water generally contracted with spherical d against
// a displaced water, primitive with Cartesian d, in both orders (the one the transpose of the other); the refusals.

mod common;

use aedelsten::{AedelstenContext, ContractionType};
use std::ffi::{CString, c_char, c_void};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn set_shell_pairs_c(ctx: *mut c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut c_void, size: i32, unit: i32) -> i32;
    fn compute_overlap_with_c(ctx: *const c_void, other: *const c_void, output: *mut f64, len: i64) -> i32;
}

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.2\nEnd of Basis\n\
                     Basis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";
const DISPLACED: &str = "Basis set\n    O.x2c-SVPall\n    Cartesian d\n    O1 0.1 0.0 0.3\nEnd of Basis\n\
                         Basis set\n    H.x2c-SVPall\n    H1 0.0 1.5 -0.8\n    H2 0.1 -1.3 -0.9\nEnd of Basis\n";

struct Pair {
    rust: AedelstenContext,
    c: *mut c_void,
}

fn both(tag: &str, input: &str, contraction: ContractionType) -> Pair {
    let path = common::temp_path(&format!("ffi_overlap_with_{tag}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut rust = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    rust.set_shell_pairs(None).unwrap();
    rust.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let file = CString::new(path.to_str().unwrap()).unwrap();
    let c = unsafe { set_aedelsten_context_c(file.as_ptr(), contraction.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    std::fs::remove_file(&path).ok();
    assert!(!c.is_null());
    assert_eq!(unsafe { set_shell_pairs_c(c, -1.0) }, 0);
    assert_eq!(unsafe { set_scratch_c(c, 64, 1) }, 0);
    Pair { rust, c }
}

#[test]
fn the_overlap_with_another_basis_is_the_rust_calls_answer() {
    let a = both("general", WATER, ContractionType::General);
    let b = both("primitive", DISPLACED, ContractionType::Primitive);
    let (n_a, n_b) = (a.rust.get_n_cont(), b.rust.get_n_cont());
    assert_ne!(n_a, n_b);

    let mut expected_ab = vec![0.0; n_a * n_b];
    a.rust.compute_overlap_with(&b.rust, &mut expected_ab).unwrap();
    let mut ab = vec![f64::NAN; n_a * n_b];
    assert_eq!(unsafe { compute_overlap_with_c(a.c, b.c, ab.as_mut_ptr(), ab.len() as i64) }, 0);
    assert_eq!(ab, expected_ab);
    assert!(ab.iter().any(|&s| s.abs() > 0.5), "an overlap to compare");

    let mut ba = vec![f64::NAN; n_a * n_b];
    assert_eq!(unsafe { compute_overlap_with_c(b.c, a.c, ba.as_mut_ptr(), ba.len() as i64) }, 0);
    for i in 0..n_a {
        for j in 0..n_b {
            assert!((ba[j + n_b * i] - ab[i + n_a * j]).abs() < 1e-14, "the transpose at ({i}, {j})");
        }
    }

    assert_ne!(unsafe { compute_overlap_with_c(a.c, b.c, ab.as_mut_ptr(), ab.len() as i64 - 1) }, 0, "a short output");
    assert_ne!(unsafe { compute_overlap_with_c(a.c, std::ptr::null(), ab.as_mut_ptr(), ab.len() as i64) }, 0, "no other context");
    unsafe {
        free_aedelsten_context_c(a.c);
        free_aedelsten_context_c(b.c);
    }
}
