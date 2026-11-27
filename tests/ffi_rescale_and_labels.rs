// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points for rescale_cartesian_ao_rows/columns and get_centre_label, driven the way a
// Fortran/C/Julia caller would and cross-checked against the Rust-side API they wrap.

use aedelsten::{AedelstenContext, ContractionType};
use std::ffi::{CString, c_char};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn rescale_cartesian_ao_rows_c(ctx: *mut std::ffi::c_void, convention: i32, n_cols: i32, matrix: *mut f64, len: i32, rescaled: *mut i32) -> i32;
    fn get_centre_label_c(ctx: *mut std::ffi::c_void, centre_idx: i32, buf: *mut c_char, buf_len: i32) -> i32;
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");
const MOLCAS: i32 = 0; // SharedCartesianReference::MOLCAS.get_c_int()

#[test]
fn rescale_rows_reports_the_same_rescaled_flag_and_values_as_rust() {
    let path = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());

    let rust_ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n = rust_ctx.get_n_cont();
    let mut rust_matrix = vec![1.0; n];
    let rust_rescaled = rust_ctx.rescale_cartesian_ao_rows(aedelsten::SharedCartesianReference::MOLCAS, 1, &mut rust_matrix).unwrap();

    let mut c_matrix = vec![1.0; n];
    let mut rescaled_flag: i32 = -1;
    let status = unsafe { rescale_cartesian_ao_rows_c(ctx, MOLCAS, 1, c_matrix.as_mut_ptr(), n as i32, &mut rescaled_flag) };
    assert_eq!(status, 0);
    assert_eq!(rescaled_flag, rust_rescaled as i32);
    assert_eq!(c_matrix, rust_matrix);
    assert_eq!(rescaled_flag, 1); // this fixture has a Cartesian d shell

    // A null `rescaled` pointer is allowed (caller doesn't care).
    let mut c_matrix2 = vec![1.0; n];
    let status = unsafe { rescale_cartesian_ao_rows_c(ctx, MOLCAS, 1, c_matrix2.as_mut_ptr(), n as i32, std::ptr::null_mut()) };
    assert_eq!(status, 0);

    unsafe { free_aedelsten_context_c(ctx) };
}

#[test]
fn centre_label_matches_rust_and_rejects_a_too_small_buffer() {
    let path = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());

    let mut buf = [0i8; 16];
    let status = unsafe { get_centre_label_c(ctx, 0, buf.as_mut_ptr(), buf.len() as i32) };
    assert_eq!(status, 0);
    let label = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap();
    assert_eq!(label, "O1");

    // Buffer with no room even for "O1\0" is rejected, not silently truncated.
    let mut tiny = [0i8; 1];
    assert_eq!(unsafe { get_centre_label_c(ctx, 0, tiny.as_mut_ptr(), tiny.len() as i32) }, 1);

    // No centre 1 in a single-atom system.
    let mut buf2 = [0i8; 16];
    assert_eq!(unsafe { get_centre_label_c(ctx, 1, buf2.as_mut_ptr(), buf2.len() as i32) }, 1);

    unsafe { free_aedelsten_context_c(ctx) };
}
