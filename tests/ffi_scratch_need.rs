// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points for the scratch need (`scratch_bytes`, `scratch_breakdown`), driven the way a C or Fortran
// caller would (flat int32 rows in, int64 rows out) and compared with the Rust calls they wrap.

use aedelsten::{AedelstenContext, ContractionType, ScratchOperation, SCRATCH_BREAKDOWN_FIELDS, SCRATCH_OPERATION_FIELDS};
use std::ffi::{CStr, CString, c_char, c_void};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn scratch_bytes_c(ctx: *const c_void, operations: *const i32, n_operations: i32, primitive: i32) -> i64;
    fn scratch_breakdown_field_count_c() -> i32;
    fn scratch_breakdown_field_name_c(index: i32) -> *const c_char;
    fn scratch_breakdown_c(ctx: *const c_void, operations: *const i32, n_operations: i32, primitive: i32, output: *mut i64, len: i32) -> i32;
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");

// Every kind once, in C rows and in Rust.
fn operations() -> (Vec<i32>, Vec<ScratchOperation>) {
    let rows = [
        [1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [3, 0, 0, 0, 0],
        [4, 2, 0, 0, 0],
        [5, 0, 0, 0, 0],
        [6, 3, 0, 0, 0],
        [7, 4, 0, 0, 0],
        [8, 4, 0, 0, 0],
        [9, 3, -1, -1, 1],
        [9, 3, 4, 7, 2],
        [10, 0, 0, 0, 0],
    ];
    let rust = vec![
        ScratchOperation::EriData,
        ScratchOperation::RestrictedFock,
        ScratchOperation::UnrestrictedFock,
        ScratchOperation::TwoElectronGradient { terms: 2 },
        ScratchOperation::GeneralFock,
        ScratchOperation::GeneralMultiFock { densities: 3 },
        ScratchOperation::ActiveSpaceFock { active: 4 },
        ScratchOperation::MixedActiveSpaceFock { active: 4 },
        ScratchOperation::HalfTransform { left: 3, right: None, sigma: None, node_ranks: 1 },
        ScratchOperation::HalfTransform { left: 3, right: Some(4), sigma: Some(7), node_ranks: 2 },
        ScratchOperation::Amfi,
    ];
    (rows.concat(), rust)
}

#[test]
fn the_field_names_match_the_rust_table() {
    let count = unsafe { scratch_breakdown_field_count_c() };
    assert_eq!(count as usize, SCRATCH_BREAKDOWN_FIELDS.len());
    for (i, expected) in SCRATCH_BREAKDOWN_FIELDS.iter().enumerate() {
        let name = unsafe { CStr::from_ptr(scratch_breakdown_field_name_c(i as i32)) };
        assert_eq!(name, *expected, "index {i}");
    }
    assert!(unsafe { scratch_breakdown_field_name_c(-1) }.is_null());
    assert!(unsafe { scratch_breakdown_field_name_c(count) }.is_null());
}

#[test]
fn the_c_query_and_breakdown_match_the_rust_calls() {
    let path = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());
    let rust_ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let (rows, rust) = operations();
    let n = rust.len() as i32;
    assert_eq!(rows.len(), SCRATCH_OPERATION_FIELDS * rust.len());
    let fields = SCRATCH_BREAKDOWN_FIELDS.len();

    for primitive in [0, 1] {
        let bytes = unsafe { scratch_bytes_c(ctx, rows.as_ptr(), n, primitive) };
        assert_eq!(bytes as usize, rust_ctx.scratch_bytes(&rust, primitive != 0).unwrap());
        // One operation at a time: its row of the breakdown.
        for (row, operation) in rows.chunks_exact(SCRATCH_OPERATION_FIELDS).zip(&rust) {
            let alone = unsafe { scratch_bytes_c(ctx, row.as_ptr(), 1, primitive) };
            assert_eq!(alone as usize, rust_ctx.scratch_bytes(&[*operation], primitive != 0).unwrap(), "{operation:?}");
        }

        let mut output = vec![0i64; fields * rust.len()];
        let status = unsafe { scratch_breakdown_c(ctx, rows.as_ptr(), n, primitive, output.as_mut_ptr(), output.len() as i32) };
        assert_eq!(status, 0);
        let breakdown = rust_ctx.scratch_breakdown(&rust, primitive != 0).unwrap();
        for (row, need) in output.chunks_exact(fields).zip(&breakdown) {
            assert_eq!(row, need.c_fields());
            assert_eq!(row[0], row[1] + row[2] + row[3] + row[4], "the parts add up");
            assert_eq!(row[5], need.item as i64);
            assert_eq!(row[6], need.shells.first().map_or(-1, |shell| shell.l as i64));
        }
        assert_eq!(output.chunks_exact(fields).map(|row| row[0]).max().unwrap(), bytes);
    }

    // No operations: nothing needed, no rows.
    assert_eq!(unsafe { scratch_bytes_c(ctx, std::ptr::null(), 0, 0) }, 0);
    // Refused: an unknown kind, a negative count, no ranks, a buffer of the wrong length, a null context.
    for bad in [[0, 0, 0, 0, 0], [11, 0, 0, 0, 0], [4, -2, 0, 0, 0], [9, 3, -1, -1, 0], [9, 3, -2, -1, 1]] {
        assert_eq!(unsafe { scratch_bytes_c(ctx, bad.as_ptr(), 1, 0) }, -1, "{bad:?}");
    }
    let mut short = vec![0i64; fields * rust.len() - 1];
    assert_eq!(unsafe { scratch_breakdown_c(ctx, rows.as_ptr(), n, 0, short.as_mut_ptr(), short.len() as i32) }, 1);
    assert_eq!(unsafe { scratch_bytes_c(std::ptr::null(), rows.as_ptr(), n, 0) }, -1);
    unsafe { free_aedelsten_context_c(ctx) };
}
