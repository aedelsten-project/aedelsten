// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the X2C picture of operators with their spin part and of AMFI (src/ffi/relativity.rs), driven
// as a C caller would, against the Rust API on an identical context.

mod common;

use aedelsten::{AedelstenContext, ContractionType, MomentOrigin, OneElectronOperator};
use std::ffi::{CString, c_char, c_void};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn set_shell_pairs_c(ctx: *mut c_void, overlap_threshold: f64) -> i32;
    fn set_primitive_shell_pairs_c(ctx: *mut c_void, overlap_threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut c_void, size: i32, unit: i32) -> i32;
    fn get_n_cont_c(ctx: *const c_void) -> i32;
    fn get_n_prim_unique_c(ctx: *const c_void) -> i32;
    fn compute_one_electron_integrals_c(
        ctx: *mut c_void, output: *mut f64, n: i32, operator: i32, n_planes: i32, x: f64, y: f64, z: f64, order: i32, primitive: i32,
    ) -> i32;
    fn compute_one_electron_picture_change_integrals_c(
        ctx: *mut c_void, output: *mut f64, n: i32, operator: i32, n_planes: i32, x: f64, y: f64, z: f64, order: i32, spin_orbit: i32,
    ) -> i32;
    fn construct_x2c_one_electron_hamiltonian_c(ctx: *mut c_void, oneham: *mut f64, s: *const f64, t: *const f64, v: *const f64, pvp: *const f64, n: i32, c_light: f64) -> i32;
    fn transform_one_electron_spin_operator_c(ctx: *const c_void, op_pxp: *const f64, out: *mut f64, n: i32, n_planes: i32) -> i32;
    fn compute_picture_changed_one_electron_operator_c(
        ctx: *const c_void, scalar: *mut f64, spin: *mut f64, n: i32, operator: i32, n_components: i32, x: f64, y: f64, z: f64, order: i32, contracted: i32,
    ) -> i32;
    fn compute_amfi_integrals_c(ctx: *const c_void, output: *mut f64, n: i32) -> i32;
}

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.0\nEnd of Basis\n\
                     Basis set\n    H.x2c-SVPall\n    H1 0.0 1.43 1.11\n    H2 0.0 -1.43 1.11\nEnd of Basis\n";
const NUCLEAR_ATTRACTION: i32 = 2;
const MOMENT: i32 = 18;

// The primitive S, T, V, pVp of `operator`-style calls, then X2C, through C.
fn c_context(path: &str) -> (*mut c_void, i32, i32) {
    let name = CString::new(path).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(name.as_ptr(), 0, aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());
    unsafe {
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_primitive_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 128, 1), 0);
    }
    let (n, m) = unsafe { (get_n_prim_unique_c(ctx), get_n_cont_c(ctx)) };
    let size = (n * n) as usize;
    let mut stvp = [vec![0.0; size], vec![0.0; size], vec![0.0; size], vec![0.0; size]];
    for (operator, out) in [0, 1, NUCLEAR_ATTRACTION].into_iter().zip(stvp.iter_mut()) {
        assert_eq!(unsafe { compute_one_electron_integrals_c(ctx, out.as_mut_ptr(), n, operator, 1, 0.0, 0.0, 0.0, 0, 1) }, 0);
    }
    assert_eq!(unsafe { compute_one_electron_picture_change_integrals_c(ctx, stvp[3].as_mut_ptr(), n, NUCLEAR_ATTRACTION, 1, 0.0, 0.0, 0.0, 0, 0) }, 0);
    let mut oneham = vec![0.0; size];
    let [s, t, v, pvp] = &stvp;
    assert_eq!(unsafe { construct_x2c_one_electron_hamiltonian_c(ctx, oneham.as_mut_ptr(), s.as_ptr(), t.as_ptr(), v.as_ptr(), pvp.as_ptr(), n, -1.0) }, 0);
    (ctx, n, m)
}

fn rust_context(path: &str) -> AedelstenContext {
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    ctx.set_scratch((128, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_prim_unique();
    let mut stvp = [vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]];
    ctx.compute_one_electron_integrals(&mut stvp[0], OneElectronOperator::overlap(), true).unwrap();
    ctx.compute_one_electron_integrals(&mut stvp[1], OneElectronOperator::kinetic(), true).unwrap();
    ctx.compute_one_electron_integrals(&mut stvp[2], ctx.get_nuclear_attraction_operator(), true).unwrap();
    ctx.compute_one_electron_picture_change_integrals(&mut stvp[3], ctx.get_nuclear_attraction_operator(), false).unwrap();
    let mut oneham = vec![0.0; n * n];
    let [s, t, v, pvp] = &stvp;
    ctx.construct_x2c_one_electron_hamiltonian(&mut oneham, s, t, v, pvp, None).unwrap();
    ctx
}

#[test]
fn picture_changed_operators_and_amfi_through_the_c_interface() {
    let path = common::temp_path("ffi_picture_change.inp");
    std::fs::write(&path, WATER).unwrap();
    let (c, n, m) = c_context(path.to_str().unwrap());
    let rust = rust_context(path.to_str().unwrap());
    std::fs::remove_file(&path).ok();
    let size = (n * n) as usize;

    // AMFI.
    let mut amfi_c = vec![0.0; 6 * size];
    assert_eq!(unsafe { compute_amfi_integrals_c(c, amfi_c.as_mut_ptr(), n) }, 0);
    let mut amfi_rust = vec![0.0; 6 * size];
    rust.compute_amfi_integrals(&mut amfi_rust).unwrap();
    assert_eq!(amfi_c, amfi_rust);
    assert!(amfi_c.iter().any(|x| x.abs() > 1e-2));

    // The spin fold of the one-electron spin planes plus AMFI's, as a spin-orbit calculation needs it.
    let mut planes = vec![0.0; 4 * size];
    assert_eq!(unsafe { compute_one_electron_picture_change_integrals_c(c, planes.as_mut_ptr(), n, NUCLEAR_ATTRACTION, 4, 0.0, 0.0, 0.0, 0, 1) }, 0);
    let mut spin: Vec<f64> = planes[..3 * size].iter().zip(&amfi_c[..3 * size]).zip(&amfi_c[3 * size..]).map(|((a, b), c)| a + b + c).collect();
    let mut folded_c = vec![0.0; 3 * size];
    assert_eq!(unsafe { transform_one_electron_spin_operator_c(c, spin.as_ptr(), folded_c.as_mut_ptr(), n, 3) }, 0);
    let mut folded_rust = vec![0.0; 3 * size];
    rust.transform_one_electron_spin_operator(&spin, 3, &mut folded_rust).unwrap();
    assert_eq!(folded_c, folded_rust);
    spin.clear();

    // The dipole operator in the X2C picture, contracted.
    let csize = (m * m) as usize;
    let (mut scalar_c, mut spin_c) = (vec![0.0; 3 * csize], vec![0.0; 9 * csize]);
    let rc = unsafe { compute_picture_changed_one_electron_operator_c(c, scalar_c.as_mut_ptr(), spin_c.as_mut_ptr(), m, MOMENT, 3, 0.1, -0.2, 0.3, 1, 1) };
    assert_eq!(rc, 0);
    let (mut scalar_rust, mut spin_rust) = (vec![0.0; 3 * csize], vec![0.0; 9 * csize]);
    rust.compute_picture_changed_one_electron_operator(rust.get_moment_operator(MomentOrigin::new([0.1, -0.2, 0.3], 1)), true, &mut scalar_rust, &mut spin_rust).unwrap();
    assert_eq!(scalar_c, scalar_rust);
    assert_eq!(spin_c, spin_rust);
    // The primitive basis claimed for buffers of the contracted size is refused (larger buffers, so nothing is
    // touched out of bounds).
    let (mut scalar_big, mut spin_big) = (vec![0.0; 3 * size], vec![0.0; 9 * size]);
    let rc = unsafe { compute_picture_changed_one_electron_operator_c(c, scalar_big.as_mut_ptr(), spin_big.as_mut_ptr(), n, MOMENT, 3, 0.1, -0.2, 0.3, 1, 1) };
    assert_ne!(rc, 0);

    unsafe { free_aedelsten_context_c(c) };
}
