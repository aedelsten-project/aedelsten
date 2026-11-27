// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the contracted one-electron gradient (get_n_gradient_centres_c,
// compute_one_electron_gradient_c), driven the way a Fortran/C caller would (raw pointers, flat column-major
// buffers, integer operator codes) and compared to the Rust API they wrap -- for a plain operator (kinetic) and an
// AIMP-special one (core-AIMP projection), which go through different branches inside the context.

mod common;

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};
use std::ffi::{CString, c_char};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn set_shell_pairs_c(ctx: *mut std::ffi::c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut std::ffi::c_void, size: i32, unit: i32) -> i32;
    fn get_n_gradient_centres_c(ctx: *const std::ffi::c_void) -> i32;
    fn compute_one_electron_gradient_c(
        ctx: *mut std::ffi::c_void,
        gradient: *mut f64,
        n_centres: i32,
        density: *const f64,
        n: i32,
        operator: i32,
        active: *const i32,
    ) -> i32;
}

// OneElectronOperatorKind's C-facing codes (hardcoded, as a real C/Fortran caller has to).
const KIND_KINETIC: i32 = 1;
const KIND_CORE_AIMP_PROJECTION: i32 = 12;
const KIND_MULTIPOLE: i32 = 18;

const INPUT: &str = "Basis set\n    Sc.TEST-CORE-AIMP\n    Sc1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.15 -0.2 4.0\nEnd of Basis\n";

fn density(n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[i + n * j] = 0.2 + 0.05 * (((i.min(j) * 7 + i.max(j) * 3) % 11) as f64);
        }
    }
    d
}

#[test]
fn the_c_interface_reproduces_the_rust_contracted_gradient() {
    let path = common::temp_path("ffi_gradcon.inp");
    std::fs::write(&path, INPUT).unwrap();
    let inp = CString::new(path.to_str().unwrap()).unwrap();

    let mut reference = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    reference.set_shell_pairs(None).unwrap();
    reference.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = reference.get_n_cont();
    let n_centres = reference.get_n_gradient_centres();
    let d = density(n);
    let mut kinetic_ref = vec![0.0; 3 * n_centres];
    reference.compute_one_electron_gradient(&mut kinetic_ref, &d, OneElectronOperator::kinetic(), None).unwrap();
    let mut projection_ref = vec![0.0; 3 * n_centres];
    reference.compute_one_electron_gradient(&mut projection_ref, &d, reference.get_core_aimp_projection_operator().unwrap(), None).unwrap();
    assert!(kinetic_ref.iter().any(|v| v.abs() > 1e-6) && projection_ref.iter().any(|v| v.abs() > 1e-6));

    unsafe {
        let ctx = set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL);
        assert!(!ctx.is_null());
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 64, 1), 0);
        assert_eq!(get_n_gradient_centres_c(ctx), n_centres as i32);

        for (kind, expected) in [(KIND_KINETIC, &kinetic_ref), (KIND_CORE_AIMP_PROJECTION, &projection_ref)] {
            let mut gradient = vec![f64::NAN; 3 * n_centres];
            assert_eq!(compute_one_electron_gradient_c(ctx, gradient.as_mut_ptr(), n_centres as i32, d.as_ptr(), n as i32, kind, std::ptr::null()), 0);
            for (got, want) in gradient.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1e-12, "kind {kind}: {got} vs {want}");
            }
        }

        // A matrix with the wrong number of centres, and the operator that has no gradient, are refused.
        let mut too_wide = vec![0.0; 3 * (n_centres + 1)];
        assert_ne!(compute_one_electron_gradient_c(ctx, too_wide.as_mut_ptr(), n_centres as i32 + 1, d.as_ptr(), n as i32, KIND_KINETIC, std::ptr::null()), 0);
        let mut gradient = vec![0.0; 3 * n_centres];
        assert_ne!(compute_one_electron_gradient_c(ctx, gradient.as_mut_ptr(), n_centres as i32, d.as_ptr(), n as i32, KIND_MULTIPOLE, std::ptr::null()), 0);
        assert_ne!(compute_one_electron_gradient_c(ctx, gradient.as_mut_ptr(), n_centres as i32, d.as_ptr(), n as i32, 9999, std::ptr::null()), 0);

        // A frozen centre (flag 0) comes back as zero, the active one unchanged; all-frozen gives zeros.
        for (mask, kind, expected) in [([1, 0], KIND_KINETIC, &kinetic_ref), ([0, 1], KIND_CORE_AIMP_PROJECTION, &projection_ref)] {
            let mut gradient = vec![f64::NAN; 3 * n_centres];
            assert_eq!(compute_one_electron_gradient_c(ctx, gradient.as_mut_ptr(), n_centres as i32, d.as_ptr(), n as i32, kind, mask.as_ptr()), 0);
            for centre in 0..n_centres {
                for c in 0..3 {
                    let want = if mask[centre] != 0 { expected[c + 3 * centre] } else { 0.0 };
                    assert!((gradient[c + 3 * centre] - want).abs() < 1e-12, "mask {mask:?}, kind {kind}, centre {centre}");
                }
            }
        }
        let frozen = [0, 0];
        let mut gradient = vec![f64::NAN; 3 * n_centres];
        assert_eq!(compute_one_electron_gradient_c(ctx, gradient.as_mut_ptr(), n_centres as i32, d.as_ptr(), n as i32, KIND_KINETIC, frozen.as_ptr()), 0);
        assert!(gradient.iter().all(|v| *v == 0.0));
        free_aedelsten_context_c(ctx);
    }
    std::fs::remove_file(&path).ok();
}
