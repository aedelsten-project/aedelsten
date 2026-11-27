// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the model basis: get_n_model_functions_c,
// get_model_functions_c (flat column-major ids and the energies), compute_model_basis_overlaps_c (with and without
// S_mm) and compute_model_fockian_c, driven as a C/Fortran caller would and compared bitwise with the Rust calls they
// wrap; wrong lengths and overlapping outputs refused.

mod common;

use aedelsten::{AedelstenContext, ContractionType, ModelFunction};
use std::ffi::{c_char, c_void, CString};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn set_shell_pairs_c(ctx: *mut c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut c_void, size: i32, unit: i32) -> i32;
    fn get_n_model_functions_c(ctx: *const c_void) -> i32;
    fn get_model_functions_c(ctx: *const c_void, ids: *mut i32, ids_len: i32, energies: *mut f64, energies_len: i32) -> i32;
    fn compute_model_basis_overlaps_c(ctx: *const c_void, s_mw: *mut f64, s_mw_len: i32, s_mm: *mut f64, s_mm_len: i32) -> i32;
    fn compute_model_fockian_c(ctx: *const c_void, output: *mut f64, len: i32) -> i32;
}

// ANO-RCC only generally contracted; the segmented case on x2c-SVPall.
const SC_AIMP: &str = "Basis set\n    Sc.TEST-CORE-AIMP\n    Sc1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 0.0 3.4\nEnd of Basis\n";
const CO_SVP: &str = "Basis set\n    C.x2c-SVPall\n    C1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    O.x2c-SVPall\n    O1 0.0 0.0 2.13\nEnd of Basis\n";

/// The same input as a Rust context and a C one.
fn contexts(name: &str, input: &str, contraction: ContractionType) -> (AedelstenContext, *mut c_void) {
    let path = common::temp_path(&format!("ffi_model_basis_{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut rust = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    rust.set_shell_pairs(None).unwrap();
    rust.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let c = unsafe { set_aedelsten_context_c(c_path.as_ptr(), contraction.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    let _ = std::fs::remove_file(&path);
    assert!(!c.is_null());
    assert_eq!(unsafe { set_shell_pairs_c(c, -1.0) }, 0);
    assert_eq!(unsafe { set_scratch_c(c, 64, 1) }, 0);
    (rust, c)
}

#[test]
fn the_c_entry_points_give_the_rust_results() {
    for (name, input, contraction) in [("sc", SC_AIMP, ContractionType::General), ("co", CO_SVP, ContractionType::Segmented)] {
        let (rust, c) = contexts(name, input, contraction);
        let n = rust.get_n_cont();
        let n_model = rust.get_n_model_functions().unwrap();
        assert_eq!(unsafe { get_n_model_functions_c(c) }, n_model as i32, "{name}");

        let mut functions = vec![ModelFunction::default(); n_model];
        rust.get_model_functions(&mut functions).unwrap();
        let (mut ids, mut energies) = (vec![-7i32; 4 * n_model], vec![f64::NAN; n_model]);
        assert_eq!(unsafe { get_model_functions_c(c, ids.as_mut_ptr(), (4 * n_model) as i32, energies.as_mut_ptr(), n_model as i32) }, 0);
        for (i, f) in functions.iter().enumerate() {
            assert_eq!(
                [ids[i], ids[n_model + i], ids[2 * n_model + i], ids[3 * n_model + i]],
                [f.get_centre() as i32, f.get_l() as i32, f.get_m(), f.get_shell_ordinal() as i32],
                "{name}: function {i}"
            );
            assert_eq!(energies[i].to_bits(), f.get_energy().to_bits(), "{name}: energy {i}");
        }
        assert_eq!(unsafe { get_model_functions_c(c, ids.as_mut_ptr(), (4 * n_model - 1) as i32, energies.as_mut_ptr(), n_model as i32) }, 1);

        let (mut s_mw, mut s_mm) = (vec![0.0; n_model * n], vec![0.0; n_model * n_model]);
        rust.compute_model_basis_overlaps(&mut s_mw, Some(&mut s_mm)).unwrap();
        let (mut c_mw, mut c_mm) = (vec![f64::NAN; n_model * n], vec![f64::NAN; n_model * n_model]);
        assert_eq!(unsafe { compute_model_basis_overlaps_c(c, c_mw.as_mut_ptr(), c_mw.len() as i32, c_mm.as_mut_ptr(), c_mm.len() as i32) }, 0);
        assert_eq!(c_mw, s_mw, "{name}: S_mw");
        assert_eq!(c_mm, s_mm, "{name}: S_mm");
        let mut alone = vec![f64::NAN; n_model * n];
        assert_eq!(unsafe { compute_model_basis_overlaps_c(c, alone.as_mut_ptr(), alone.len() as i32, std::ptr::null_mut(), 0) }, 0);
        assert_eq!(alone, s_mw, "{name}: S_mw without S_mm");
        assert_eq!(unsafe { compute_model_basis_overlaps_c(c, alone.as_mut_ptr(), alone.len() as i32 - 1, std::ptr::null_mut(), 0) }, 1);
        // S_mm on top of S_mw: refused.
        let mut both = vec![0.0; n_model * n + n_model * n_model];
        assert_eq!(unsafe { compute_model_basis_overlaps_c(c, both.as_mut_ptr(), (n_model * n) as i32, both.as_mut_ptr().add(1), (n_model * n_model) as i32) }, 1);

        let mut fockian = vec![0.0; n * n];
        rust.compute_model_fockian(&mut fockian).unwrap();
        let mut c_fockian = vec![f64::NAN; n * n];
        assert_eq!(unsafe { compute_model_fockian_c(c, c_fockian.as_mut_ptr(), (n * n) as i32) }, 0);
        assert_eq!(c_fockian, fockian, "{name}: the model Fockian");
        assert_eq!(unsafe { compute_model_fockian_c(c, c_fockian.as_mut_ptr(), (n * n - 1) as i32) }, 1);

        unsafe { free_aedelsten_context_c(c) };
    }
}
