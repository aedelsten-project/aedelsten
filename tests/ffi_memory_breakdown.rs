// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points for AedelstenContext::memory_breakdown, driven the way a Fortran/C/Julia
// caller would (raw pointers, flat buffers) and cross-checked against the Rust API they wrap.

use aedelsten::{AedelstenContext, ContractionType, MEMORY_BREAKDOWN_FIELDS};
use std::ffi::{CStr, CString, c_char};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn memory_breakdown_field_count_c() -> i32;
    fn memory_breakdown_field_name_c(index: i32) -> *const c_char;
    fn memory_breakdown_c(ctx: *mut std::ffi::c_void, output: *mut u64, len: i32) -> i32;
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");

#[test]
fn field_count_and_names_match_the_rust_side_table() {
    let count = unsafe { memory_breakdown_field_count_c() };
    assert_eq!(count as usize, MEMORY_BREAKDOWN_FIELDS.len());

    for (i, expected) in MEMORY_BREAKDOWN_FIELDS.iter().enumerate() {
        let name_ptr = unsafe { memory_breakdown_field_name_c(i as i32) };
        assert!(!name_ptr.is_null());
        let name = unsafe { CStr::from_ptr(name_ptr) };
        assert_eq!(name, *expected, "index {i}");
    }

    // Out of range: null, not a panic or garbage pointer.
    assert!(unsafe { memory_breakdown_field_name_c(-1) }.is_null());
    assert!(unsafe { memory_breakdown_field_name_c(count) }.is_null());
}

#[test]
fn memory_breakdown_c_matches_the_rust_api() {
    let path = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());

    let len = MEMORY_BREAKDOWN_FIELDS.len();
    let mut from_c = vec![0u64; len];
    let status = unsafe { memory_breakdown_c(ctx, from_c.as_mut_ptr(), len as i32) };
    assert_eq!(status, 0);

    let rust_ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let mut from_rust = vec![0usize; len];
    rust_ctx.memory_breakdown(&mut from_rust).unwrap();

    let from_rust_u64: Vec<u64> = from_rust.iter().map(|&b| b as u64).collect();
    assert_eq!(from_c, from_rust_u64);
    // qm_region is resident right after construction on both sides.
    assert!(from_c[0] > 0);

    // Wrong-sized buffer is rejected, not silently truncated/overrun.
    let mut wrong = vec![0u64; len - 1];
    let status = unsafe { memory_breakdown_c(ctx, wrong.as_mut_ptr(), (len - 1) as i32) };
    assert_eq!(status, 1);

    unsafe { free_aedelsten_context_c(ctx) };
}
