// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry point of the scalar X2C transform of a one-electron operator (transform_one_electron_operator_c),
// against the Rust calls it wraps: one plane (transform_one_electron_operator) and three
// (transform_one_electron_operator_array3d, each plane the one-plane transform of that plane), and the refusals
// (no planes, a wrong size, an output over an input). The context drives both, through its address as the handle.

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};

unsafe extern "C" {
    fn transform_one_electron_operator_c(ctx: *const std::ffi::c_void, op_plain: *const f64, op_pxp: *const f64, out: *mut f64, n: i32, n_planes: i32) -> i32;
}

fn plane(n: usize, salt: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            m[i + n * j] = 0.05 * (((i.min(j) * 7 + i.max(j) * 3 + salt) % 11) as f64) - 0.2;
        }
    }
    m
}

#[test]
fn the_scalar_x2c_transform_is_the_rust_calls_answer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_prim_unique();
    let integrals = |operator: OneElectronOperator<'_>| {
        let mut m = vec![0.0; n * n];
        ctx.compute_one_electron_integrals(&mut m, operator, true).unwrap();
        m
    };
    let (s, t, v) = (integrals(OneElectronOperator::overlap()), integrals(OneElectronOperator::kinetic()), integrals(ctx.get_nuclear_attraction_operator()));
    let mut pvp = vec![0.0; n * n];
    ctx.compute_one_electron_picture_change_integrals(&mut pvp, ctx.get_nuclear_attraction_operator(), false).unwrap();
    let mut h = vec![0.0; n * n];
    ctx.construct_x2c_one_electron_hamiltonian(&mut h, &s, &t, &v, &pvp, None).unwrap();
    let handle = &ctx as *const AedelstenContext as *const std::ffi::c_void;

    // One plane.
    let (plain, pxp) = (plane(n, 1), plane(n, 4));
    let mut expected = vec![0.0; n * n];
    ctx.transform_one_electron_operator(&plain, &pxp, &mut expected).unwrap();
    let mut out = vec![f64::NAN; n * n];
    assert_eq!(unsafe { transform_one_electron_operator_c(handle, plain.as_ptr(), pxp.as_ptr(), out.as_mut_ptr(), n as i32, 1) }, 0);
    assert_eq!(out, expected);
    assert!(out.iter().any(|x| x.abs() > 1e-3));

    // Three planes: the array3d call's answer, each plane its own one-plane transform.
    let plains: Vec<f64> = (0..3).flat_map(|k| plane(n, 2 * k + 1)).collect();
    let pxps: Vec<f64> = (0..3).flat_map(|k| plane(n, 2 * k + 5)).collect();
    let mut expected = vec![0.0; 3 * n * n];
    ctx.transform_one_electron_operator_array3d(&plains, &pxps, 3, &mut expected).unwrap();
    let mut out = vec![f64::NAN; 3 * n * n];
    assert_eq!(unsafe { transform_one_electron_operator_c(handle, plains.as_ptr(), pxps.as_ptr(), out.as_mut_ptr(), n as i32, 3) }, 0);
    assert_eq!(out, expected);
    for k in 0..3 {
        let mut single = vec![0.0; n * n];
        let r = k * n * n..(k + 1) * n * n;
        ctx.transform_one_electron_operator(&plains[r.clone()], &pxps[r.clone()], &mut single).unwrap();
        let largest = out[r].iter().zip(&single).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(largest < 1e-13, "plane {k}: {largest:e}");
    }

    let mut out = vec![0.0; n * n];
    assert_ne!(unsafe { transform_one_electron_operator_c(handle, plain.as_ptr(), pxp.as_ptr(), out.as_mut_ptr(), n as i32, 0) }, 0, "no planes");
    assert_ne!(unsafe { transform_one_electron_operator_c(handle, plain.as_ptr(), pxp.as_ptr(), out.as_mut_ptr(), n as i32 - 1, 1) }, 0, "a wrong size");
    let mut aliased = plain.clone();
    assert_ne!(unsafe { transform_one_electron_operator_c(handle, aliased.as_ptr(), pxp.as_ptr(), aliased.as_mut_ptr(), n as i32, 1) }, 0, "the output over an input");
}
