// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The communicator a context's parallel work is split over: given at creation, replaced by set_communicator, through
// the Rust API and the C entry points (the multi-rank behaviour is examples/mpi_eri_check.rs's).

use aedelsten::{AedelstenContext, ContractionType};
use aedelsten_mpi::{Communicator, COMM_SINGLE_SENTINEL};
use std::ffi::{c_char, CString};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn set_communicator_c(ctx: *mut std::ffi::c_void, comm: i32) -> i32;
    fn aedelsten_query_i32_c(key: *const c_char) -> i32;
    fn aedelsten_mpi_shared_array_c(comm: i32, len: i64, data: *mut *mut f64) -> *mut std::ffi::c_void;
    fn aedelsten_mpi_shared_array_sync_c(handle: *mut std::ffi::c_void) -> i32;
    fn aedelsten_mpi_shared_array_free_c(handle: *mut std::ffi::c_void);
    fn compute_eri_half_transformed_on_node_c(
        ctx: *mut std::ffi::c_void, x_out: *mut f64, c_left: *const f64, m_left: i32, c_right: *const f64, m_right: i32, n: i32,
        threshold: f64, pairwise_threshold: f64,
    ) -> i32;
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/mpi_oneint_check.inp");

#[test]
fn a_context_keeps_its_communicator_until_it_is_replaced() {
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &Communicator::single()).unwrap();
    assert_eq!((ctx.communicator().rank(), ctx.communicator().size()), (0, 1));
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((16, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();
    let mut before = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut before, aedelsten::OneElectronOperator::overlap(), false).unwrap();

    let previous = ctx.set_communicator(&Communicator::single());
    assert_eq!(previous.size(), 1);
    let mut after = vec![f64::NAN; n * n];
    ctx.compute_one_electron_integrals(&mut after, aedelsten::OneElectronOperator::overlap(), false).unwrap();
    assert_eq!(after, before);
}

#[test]
fn the_c_entry_points_take_a_communicator_by_its_handle() {
    let key = CString::new("mpi_comm_single").unwrap();
    assert_eq!(unsafe { aedelsten_query_i32_c(key.as_ptr()) }, COMM_SINGLE_SENTINEL);

    let path = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());
    assert_eq!(unsafe { set_communicator_c(ctx, COMM_SINGLE_SENTINEL) }, 0);
    assert_ne!(unsafe { set_communicator_c(std::ptr::null_mut(), COMM_SINGLE_SENTINEL) }, 0);
    unsafe { free_aedelsten_context_c(ctx) };
}

#[test]
fn the_half_transform_on_a_node_of_one_rank_is_the_half_transform() {
    let path = CString::new(concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/mpi_eri_check.inp")).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((16, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(None).unwrap();
    let (n, m) = (ctx.get_n_cont(), 2);
    let c: Vec<f64> = (0..n * m).map(|k| 0.1 * ((k * 7 + 1) % 5) as f64 - 0.2).collect();
    let mut expected = vec![0.0; m * n * m * n];
    ctx.compute_eri_half_transformed(&c, m, &c, m, &mut expected, None, None, true).unwrap();
    let mut on_node = vec![f64::NAN; m * n * m * n];
    ctx.compute_eri_half_transformed_on_node(&c, m, &c, m, &mut on_node, None, None).unwrap();
    assert_eq!(on_node, expected);

    // Through the C entry points, into a shared array of this process alone.
    let c_ctx = unsafe { set_aedelsten_context_c(path.as_ptr(), ContractionType::General.get_c_int(), COMM_SINGLE_SENTINEL) };
    let mut data = std::ptr::null_mut();
    let handle = unsafe { aedelsten_mpi_shared_array_c(COMM_SINGLE_SENTINEL, (m * n * m * n) as i64, &mut data) };
    assert!(!handle.is_null() && !data.is_null());
    let ctx_ref = unsafe { &mut *(c_ctx as *mut AedelstenContext) };
    ctx_ref.set_shell_pairs(None).unwrap();
    ctx_ref.set_scratch((16, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx_ref.set_eri_data(None).unwrap();
    let code = unsafe { compute_eri_half_transformed_on_node_c(c_ctx, data, c.as_ptr(), m as i32, c.as_ptr(), m as i32, n as i32, -1.0, -1.0) };
    assert_eq!(code, 0);
    assert_eq!(unsafe { aedelsten_mpi_shared_array_sync_c(handle) }, 0);
    assert_eq!(unsafe { std::slice::from_raw_parts(data, m * n * m * n) }, &expected[..]);
    unsafe {
        aedelsten_mpi_shared_array_free_c(handle);
        free_aedelsten_context_c(c_ctx);
    }
}
