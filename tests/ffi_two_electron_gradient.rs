// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry point of the contracted two-electron gradient (compute_two_electron_gradient_c), driven the way a
// Fortran/C caller would (raw pointers, stacked column-major densities, integer term codes, 0-based density
// indices) and compared to the Rust API it wraps: restricted HF terms, a non-symmetric (spin-flip) exchange term,
// the frozen-centre mask, and the refusals (bad term kind, density index out of range, wrong number of centres).

mod common;

use aedelsten::{TwoElectronTerm, TwoElectronTermKind};
use aedelsten::{AedelstenContext, ContractionType, ERIRangeOperator};
use std::ffi::{CString, c_char};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn set_shell_pairs_c(ctx: *mut std::ffi::c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut std::ffi::c_void, size: i32, unit: i32) -> i32;
    fn set_eri_data_c(ctx: *mut std::ffi::c_void, schwarz_threshold: f64) -> i32;
    fn compute_two_electron_gradient_c(
        ctx: *mut std::ffi::c_void,
        gradient: *mut f64,
        n_centres: i32,
        densities: *const f64,
        n: i32,
        n_densities: i32,
        n_terms: i32,
        term_kind: *const i32,
        term_left: *const i32,
        term_right: *const i32,
        term_coeff: *const f64,
        range_operator: i32,
        omega: f64,
        pairwise_threshold: f64,
        active: *const i32,
    ) -> i32;
}

const COULOMB: i32 = 0;
const EXCHANGE: i32 = 1;

const INPUT: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 1.5 -0.9\n    H2 -0.2 -1.4 -1.1\nEnd of Basis\n";

fn matrix(n: usize, salt: usize, symmetric: bool) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let (a, b) = if symmetric { (i.min(j), i.max(j)) } else { (i, j) };
            d[i + n * j] = 0.05 * (((a * 7 + b * 3 + salt) % 11) as f64) - 0.1;
        }
    }
    d
}

#[test]
fn the_c_interface_reproduces_the_rust_two_electron_gradient() {
    let path = common::temp_path("ffi_2e_grad.inp");
    std::fs::write(&path, INPUT).unwrap();
    let inp = CString::new(path.to_str().unwrap()).unwrap();

    let mut reference = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    reference.set_shell_pairs(None).unwrap();
    reference.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    reference.set_eri_data(None).unwrap();
    let n = reference.get_n_cont();
    let n_centres = reference.get_n_gradient_centres();
    // Densities stacked as a C/Fortran caller holds them: [P, P_flip].
    let p = matrix(n, 1, true);
    let p_flip = matrix(n, 4, false);
    let densities: Vec<f64> = p.iter().chain(p_flip.iter()).copied().collect();
    let rust_terms = vec![
        TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: &p, right: &p, coeff: 1.0 },
        TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: &p, right: &p, coeff: -0.5 },
        TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: &p_flip, right: &p_flip, coeff: -2.0 },
    ];
    let (kind, left, right, coeff) = ([COULOMB, EXCHANGE, EXCHANGE], [0, 0, 1], [0, 0, 1], [1.0, -0.5, -2.0]);
    let mut expected = vec![0.0; 3 * n_centres];
    reference.compute_two_electron_gradient(&mut expected, &rust_terms, ERIRangeOperator::FullRange, None, None).unwrap();
    assert!(expected.iter().any(|v| v.abs() > 1e-6));
    let mask = [true, false, true];
    let mut expected_masked = vec![0.0; 3 * n_centres];
    reference.compute_two_electron_gradient(&mut expected_masked, &rust_terms, ERIRangeOperator::FullRange, None, Some(&mask)).unwrap();

    unsafe {
        let ctx = set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL);
        assert!(!ctx.is_null());
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 64, 1), 0);
        assert_eq!(set_eri_data_c(ctx, -1.0), 0);
        let call = |gradient: &mut [f64], n_centres: usize, kind: &[i32], left: &[i32], right: &[i32], active: *const i32| {
            compute_two_electron_gradient_c(
                ctx, gradient.as_mut_ptr(), n_centres as i32, densities.as_ptr(), n as i32, 2, kind.len() as i32,
                kind.as_ptr(), left.as_ptr(), right.as_ptr(), coeff.as_ptr(), 0, 0.0, -1.0, active,
            )
        };

        let mut gradient = vec![f64::NAN; 3 * n_centres];
        assert_eq!(call(&mut gradient, n_centres, &kind, &left, &right, std::ptr::null()), 0);
        for (got, want) in gradient.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        let active: Vec<i32> = mask.iter().map(|&m| m as i32).collect();
        let mut gradient = vec![f64::NAN; 3 * n_centres];
        assert_eq!(call(&mut gradient, n_centres, &kind, &left, &right, active.as_ptr()), 0);
        for (got, want) in gradient.iter().zip(&expected_masked) {
            assert!((got - want).abs() < 1e-12, "masked: {got} vs {want}");
        }
        assert!(gradient[3..6].iter().all(|&v| v == 0.0), "the frozen centre's column must be zero");

        let mut gradient = vec![0.0; 3 * n_centres];
        assert_ne!(call(&mut gradient, n_centres, &[COULOMB, EXCHANGE, 7], &left, &right, std::ptr::null()), 0, "unknown term kind");
        assert_ne!(call(&mut gradient, n_centres, &kind, &[0, 0, 2], &right, std::ptr::null()), 0, "density index out of range");
        assert_ne!(call(&mut gradient, n_centres, &kind, &[0, -1, 1], &right, std::ptr::null()), 0, "negative density index");
        let mut too_wide = vec![0.0; 3 * (n_centres + 1)];
        assert_ne!(call(&mut too_wide, n_centres + 1, &kind, &left, &right, std::ptr::null()), 0, "wrong number of centres");

        free_aedelsten_context_c(ctx);
    }
    std::fs::remove_file(&path).ok();
}
