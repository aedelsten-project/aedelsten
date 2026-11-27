// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The SAP screening (`get_sap_screening_operator`): each real atom's electrons as fitted Gaussian charges, through
// `compute_one_electron_integrals` like every other operator.
//
// - far from a neutral atom its potential vanishes: on a ghost centre far away, the screening cancels the atom's
//   nuclear attraction, and the ghost centre carries no fit of its own;
// - symmetric and repulsive (positive diagonal), and the two fit sets differ;
// - cleaned strictly under the point group, which it passes untouched;
// - a core AIMP on a centre is refused, naming it; the picture-change integrals refuse the operator;
// - its C kinds reproduce the Rust integrals bitwise;
// - cited.

mod common;

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator, PointGroupRequest, SapFit, SymmetryOptions};

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1173\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4304 -0.9306\n    H2 0.0 -1.4304 -0.9306\nEnd of Basis\n";

fn context(name: &str, input: &str) -> AedelstenContext {
    let path = common::temp_path(&format!("sap_{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn matrix(ctx: &AedelstenContext, operator: OneElectronOperator<'_>) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut out = vec![f64::NAN; n * n];
    ctx.compute_one_electron_integrals(&mut out, operator, false).unwrap();
    out
}

fn sap(ctx: &AedelstenContext, fit: SapFit) -> Vec<f64> {
    matrix(ctx, ctx.get_sap_screening_operator(fit).unwrap())
}

#[test]
fn far_from_a_neutral_atom_the_screening_cancels_its_nucleus() {
    // Ne at the origin; a ghost H basis 40 bohr away, where the fits' most diffuse charge is fully enclosed.
    let input = "Basis set\n    Ne.x2c-SVPall\n    Ne1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    pseudocharge\n    X1 0.0 0.0 40.0\nEnd of Basis\n";
    let ctx = context("far", input);
    let n = ctx.get_n_cont();
    let ghost = n - 1;
    let v = matrix(&ctx, ctx.get_nuclear_attraction_operator());
    let v_ghost = v[ghost + ghost * n];
    assert!((v_ghost + 10.0 / 40.0).abs() < 1e-3, "the nucleus seen from the ghost: {v_ghost}");
    for fit in [SapFit::HelFem, SapFit::Grasp] {
        let w = sap(&ctx, fit);
        let w_ghost = w[ghost + ghost * n];
        assert!((w_ghost + v_ghost).abs() < 1e-7, "{fit:?}: screening {w_ghost} against the nucleus {v_ghost}");
    }
}

#[test]
fn the_screening_is_symmetric_repulsive_and_differs_between_the_sets() {
    let ctx = context("water", WATER);
    let n = ctx.get_n_cont();
    let helfem = sap(&ctx, SapFit::HelFem);
    let grasp = sap(&ctx, SapFit::Grasp);
    for (fit, w) in [("HelFEM", &helfem), ("GRASP", &grasp)] {
        for i in 0..n {
            assert!(w[i + i * n] > 0.0, "{fit}: diagonal {i} is {}", w[i + i * n]);
            for j in 0..n {
                assert!((w[i + j * n] - w[j + i * n]).abs() < 1e-12, "{fit}: ({i}, {j}) not symmetric");
            }
        }
    }
    let difference = helfem.iter().zip(&grasp).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(difference > 1e-6, "the two sets give the same screening ({difference})");
}

#[test]
fn the_screening_passes_the_point_groups_strict_cleaning() {
    let mut ctx = context("symmetry", WATER);
    let plain = sap(&ctx, SapFit::Grasp);
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let cleaned = sap(&ctx, SapFit::Grasp);
    let report = ctx.get_symmetrisation_report("oneint_sap_grasp", false).unwrap().expect("a cleaning report");
    assert!(report.max_abs_removed < 1e-10, "{report:?}");
    let difference = plain.iter().zip(&cleaned).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(difference < 1e-10);
}

#[test]
fn a_core_aimp_and_the_picture_change_are_refused() {
    let input = "Basis set\n    H.x2c-SVPall\n    H 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    Sc.TEST-CORE-AIMP\n    Sc 3.0 1.0 -2.0\nEnd of Basis\n";
    let ctx = context("core_aimp", input);
    assert!(ctx.has_core_aimp());
    let e = ctx.get_sap_screening_operator(SapFit::HelFem).err().expect("refused").to_string();
    assert!(e.contains("core AIMP") && e.contains("Sc1"), "{e}");

    let mut ctx = context("picture_change", WATER);
    ctx.set_primitive_shell_pairs(None).unwrap();
    let n = ctx.get_n_prim_unique();
    let mut out = vec![0.0; n * n];
    assert!(ctx.compute_one_electron_picture_change_integrals(&mut out, ctx.get_sap_screening_operator(SapFit::Grasp).unwrap(), false).is_err());
}

#[test]
fn the_screening_is_cited() {
    let ctx = context("cited", WATER);
    sap(&ctx, SapFit::HelFem);
    let cited = |doi: &str| instrumentation::citations().into_iter().any(|r| r.reference.doi() == Some(doi));
    assert!(cited("10.1021/acs.jctc.8b01089") && cited("10.1063/5.0004046"));
}

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const std::ffi::c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn set_shell_pairs_c(ctx: *mut std::ffi::c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut std::ffi::c_void, size: i32, unit: i32) -> i32;
    fn aedelsten_query_i32_c(key: *const std::ffi::c_char) -> i32;
    fn compute_one_electron_integrals_c(
        ctx: *mut std::ffi::c_void,
        output: *mut f64,
        n: i32,
        operator: i32,
        n_planes: i32,
        origin_x: f64,
        origin_y: f64,
        origin_z: f64,
        order: i32,
        primitive: i32,
    ) -> i32;
}

#[test]
fn the_c_kinds_reproduce_the_rust_screening() {
    let path = common::temp_path("sap_ffi.inp");
    std::fs::write(&path, WATER).unwrap();
    let reference = context("ffi_reference", WATER);
    let n = reference.get_n_cont();
    let key = |name: &str| unsafe { aedelsten_query_i32_c(std::ffi::CString::new(name).unwrap().as_ptr()) };
    unsafe {
        let inp = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        let ctx = set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL);
        assert!(!ctx.is_null());
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 64, 1), 0);
        for (name, fit) in [("oneint_sap_helfem", SapFit::HelFem), ("oneint_sap_grasp", SapFit::Grasp)] {
            let operator = key(name);
            assert!(operator >= 0, "{name} is not a C kind");
            let mut out = vec![f64::NAN; n * n];
            assert_eq!(compute_one_electron_integrals_c(ctx, out.as_mut_ptr(), n as i32, operator, 1, 0.0, 0.0, 0.0, 0, 0), 0);
            assert_eq!(out, sap(&reference, fit), "{name}");
        }
        free_aedelsten_context_c(ctx);
    }
    std::fs::remove_file(&path).ok();
}
