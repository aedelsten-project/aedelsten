// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points for BasisFunctionId/PrimitiveId/Primitive, driven the way a Fortran/C/Julia
// caller would (raw pointers, flat column-major buffers) and cross-checked against the Rust-side
// struct-array API they wrap.

use aedelsten::{AedelstenContext, ContractionType};
use std::ffi::{CString, c_char};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn get_basis_function_ids_c(ctx: *mut std::ffi::c_void, output: *mut i32, len: i32) -> i32;
    fn get_primitive_ids_c(ctx: *mut std::ffi::c_void, output: *mut i32, len: i32) -> i32;
    fn get_primitives_c(ctx: *mut std::ffi::c_void, output: *mut f64, len: i32) -> i32;
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");

#[test]
fn basis_function_ids_flatten_to_the_documented_column_major_layout() {
    let path = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());

    let rust_ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n = rust_ctx.get_n_cont();
    let mut ids = vec![aedelsten::BasisFunctionId::default(); n];
    rust_ctx.get_basis_function_ids(&mut ids).unwrap();

    let mut flat = vec![0i32; 4 * n];
    let status = unsafe { get_basis_function_ids_c(ctx, flat.as_mut_ptr(), (4 * n) as i32) };
    assert_eq!(status, 0);

    for (i, id) in ids.iter().enumerate() {
        assert_eq!(flat[i], id.get_centre() as i32, "centre[{i}]");
        assert_eq!(flat[n + i], id.get_shell() as i32, "shell[{i}]");
        assert_eq!(flat[2 * n + i], id.get_l(), "l[{i}]");
        assert_eq!(flat[3 * n + i], id.get_m(), "m[{i}]");
    }

    // Wrong-sized buffer rejected.
    let mut wrong = vec![0i32; 4 * n - 1];
    assert_eq!(unsafe { get_basis_function_ids_c(ctx, wrong.as_mut_ptr(), (4 * n - 1) as i32) }, 1);

    unsafe { free_aedelsten_context_c(ctx) };
}

#[test]
fn primitive_ids_and_primitives_flatten_to_the_documented_column_major_layout() {
    let path = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());

    let rust_ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n = rust_ctx.get_n_primitive_rows();
    let mut ids = vec![aedelsten::PrimitiveId { centre: 0, l: 0, shell: 0 }; n];
    rust_ctx.get_primitive_ids(&mut ids).unwrap();
    let mut primitives = vec![aedelsten::Primitive { exponent: 0.0, coefficient: 0.0 }; n];
    rust_ctx.get_primitives(&mut primitives).unwrap();

    let mut flat_ids = vec![0i32; 3 * n];
    assert_eq!(unsafe { get_primitive_ids_c(ctx, flat_ids.as_mut_ptr(), (3 * n) as i32) }, 0);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(flat_ids[i], id.centre as i32, "centre[{i}]");
        assert_eq!(flat_ids[n + i], id.l as i32, "l[{i}]");
        assert_eq!(flat_ids[2 * n + i], id.shell as i32, "shell[{i}]");
    }

    let mut flat_primitives = vec![0.0f64; 2 * n];
    assert_eq!(unsafe { get_primitives_c(ctx, flat_primitives.as_mut_ptr(), (2 * n) as i32) }, 0);
    for (i, p) in primitives.iter().enumerate() {
        assert_eq!(flat_primitives[i], p.exponent, "exponent[{i}]");
        assert_eq!(flat_primitives[n + i], p.coefficient, "coefficient[{i}]");
    }

    unsafe { free_aedelsten_context_c(ctx) };
}
