// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points that free the input, the primitive shells and the contraction matrix, driven as a C, Fortran,
// Python or Julia caller would, against the Rust calls they wrap: each empties its own field of the memory breakdown
// and no other, as the Rust call does.

use aedelsten::{AedelstenContext, ContractionType, MEMORY_BREAKDOWN_FIELDS};
use std::ffi::{CString, c_char, c_void};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn set_primitive_shell_pairs_c(ctx: *mut c_void, overlap_threshold: f64) -> i32;
    fn memory_breakdown_c(ctx: *mut c_void, output: *mut u64, len: i32) -> i32;
    fn free_input_c(ctx: *mut c_void);
    fn free_primitive_shells_c(ctx: *mut c_void);
    fn free_contraction_matrix_c(ctx: *mut c_void);
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");

fn field(name: &str) -> usize {
    MEMORY_BREAKDOWN_FIELDS.iter().position(|f| f.to_str().unwrap() == name).unwrap()
}

fn breakdown_c(ctx: *mut c_void) -> Vec<u64> {
    let mut out = vec![0u64; MEMORY_BREAKDOWN_FIELDS.len()];
    assert_eq!(unsafe { memory_breakdown_c(ctx, out.as_mut_ptr(), out.len() as i32) }, 0);
    out
}

fn breakdown_rust(ctx: &AedelstenContext) -> Vec<u64> {
    let mut out = vec![0usize; MEMORY_BREAKDOWN_FIELDS.len()];
    ctx.memory_breakdown(&mut out).unwrap();
    out.into_iter().map(|b| b as u64).collect()
}

#[test]
fn each_free_empties_its_field_as_the_rust_call_does() {
    let mut rust = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    rust.set_primitive_shell_pairs(None).unwrap();
    let path = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());
    assert_eq!(unsafe { set_primitive_shell_pairs_c(ctx, -1.0) }, 0);

    let start = breakdown_c(ctx);
    assert_eq!(start, breakdown_rust(&rust));
    for name in ["qm_region", "primitive_shells", "contraction_matrix", "primitive_shell_pairs"] {
        assert!(start[field(name)] > 0, "{name} resident before the frees");
    }

    let steps: [(&str, unsafe extern "C" fn(*mut c_void), fn(&mut AedelstenContext)); 3] = [
        ("primitive_shells", free_primitive_shells_c, AedelstenContext::free_primitive_shells),
        ("contraction_matrix", free_contraction_matrix_c, AedelstenContext::free_contraction_matrix),
        ("qm_region", free_input_c, AedelstenContext::free_input),
    ];
    let mut expected = start.clone();
    for (name, free_c, free_rust) in steps {
        unsafe { free_c(ctx) };
        free_rust(&mut rust);
        expected[field(name)] = 0;
        let after = breakdown_c(ctx);
        assert_eq!(after, expected, "after freeing {name}: only its field is emptied");
        assert_eq!(after, breakdown_rust(&rust), "after freeing {name}: as the Rust call");
    }
    assert!(expected[field("primitive_shell_pairs")] > 0, "the primitive shell pairs stay");

    // A null context is ignored.
    unsafe {
        free_input_c(std::ptr::null_mut());
        free_primitive_shells_c(std::ptr::null_mut());
        free_contraction_matrix_c(std::ptr::null_mut());
        free_aedelsten_context_c(ctx);
    }
}
