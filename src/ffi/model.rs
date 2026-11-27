// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The model basis: the free-atom reference orbitals, their overlaps, the model Fockian.

use super::*;

/// The number of model functions (`AedelstenContext::get_n_model_functions`), or -1 on failure (a null context, a
/// reference library that cannot be read).
#[unsafe(no_mangle)]
pub extern "C" fn get_n_model_functions_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    match real.get_n_model_functions() {
        Ok(n_model) => n_model as i32,
        Err(e) => {
            rank0_eprintln!("Error in get_n_model_functions (C wrapper {}): {}", stringify!(get_n_model_functions_c), e);
            -1
        }
    }
}

/// The model functions (`AedelstenContext::get_model_functions`) in get_basis_function_ids_c's flat layout: `ids`
/// column-major `n_model x 4` (all centres, 0-based as in the Rust `ModelFunction`, then all l, all m, all shell
/// ordinals), `energies` the `n_model` reference orbital energies. `ids_len` must be `4 * n_model`, `energies_len`
/// `n_model` (`get_n_model_functions_c`).
#[unsafe(no_mangle)]
pub extern "C" fn get_model_functions_c(ctx: *const AedelstenContextHandle, ids: *mut i32, ids_len: i32, energies: *mut f64, energies_len: i32) -> i32 {
    if ctx.is_null() || ids.is_null() || energies.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n_model = match real.get_n_model_functions() {
        Ok(n_model) => n_model,
        Err(e) => {
            rank0_eprintln!("Error in get_model_functions (C wrapper {}): {}", stringify!(get_model_functions_c), e);
            return FAILURE;
        }
    };
    if ids_len < 0 || ids_len as usize != 4 * n_model || energies_len < 0 || energies_len as usize != n_model {
        rank0_eprintln!(
            "Error in get_model_functions (C wrapper {}): expected buffers of length {} and {}, got {} and {}",
            stringify!(get_model_functions_c),
            4 * n_model,
            n_model,
            ids_len,
            energies_len
        );
        return FAILURE;
    }
    let mut functions = vec![crate::ModelFunction::default(); n_model];
    if let Err(e) = real.get_model_functions(&mut functions) {
        rank0_eprintln!("Error in get_model_functions (C wrapper {}): {}", stringify!(get_model_functions_c), e);
        return FAILURE;
    }
    let ids = unsafe { std::slice::from_raw_parts_mut(ids, 4 * n_model) };
    let energies = unsafe { std::slice::from_raw_parts_mut(energies, n_model) };
    let (centres, rest) = ids.split_at_mut(n_model);
    let (ls, rest) = rest.split_at_mut(n_model);
    let (ms, ordinals) = rest.split_at_mut(n_model);
    for (f, ((((centre, l), m), ordinal), energy)) in functions.iter().zip(centres.iter_mut().zip(ls.iter_mut()).zip(ms.iter_mut()).zip(ordinals.iter_mut()).zip(energies.iter_mut())) {
        (*centre, *l, *m, *ordinal, *energy) = (f.get_centre() as i32, f.get_l() as i32, f.get_m(), f.get_shell_ordinal() as i32, f.get_energy());
    }
    0
}

/// The model-basis overlaps (`AedelstenContext::compute_model_basis_overlaps`): `s_model_working` (`n_model x n`,
/// `s_mw_len` its length) and, unless `s_model_model` is null, `s_model_model` (`n_model x n_model`), column-major,
/// OVERWRITTEN. The two outputs must not overlap (such a call fails).
#[unsafe(no_mangle)]
pub extern "C" fn compute_model_basis_overlaps_c(ctx: *const AedelstenContextHandle, s_model_working: *mut f64, s_mw_len: i32, s_model_model: *mut f64, s_mm_len: i32) -> i32 {
    if ctx.is_null() || s_model_working.is_null() || s_mw_len < 0 || (!s_model_model.is_null() && s_mm_len < 0) {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let (s_mw_len, s_mm_len) = (s_mw_len as usize, if s_model_model.is_null() { 0 } else { s_mm_len as usize });
    if buffers_overlap(&[(s_model_working as *const f64, s_mw_len), (s_model_model as *const f64, s_mm_len)], &[]) {
        rank0_eprintln!("Overlapping output buffers in C wrapper {}", stringify!(compute_model_basis_overlaps_c));
        return FAILURE;
    }
    let s_mw = unsafe { std::slice::from_raw_parts_mut(s_model_working, s_mw_len) };
    let s_mm = (!s_model_model.is_null()).then(|| unsafe { std::slice::from_raw_parts_mut(s_model_model, s_mm_len) });
    match real.compute_model_basis_overlaps(s_mw, s_mm) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_model_basis_overlaps (C wrapper {}): {}", stringify!(compute_model_basis_overlaps_c), e);
            FAILURE
        }
    }
}

/// The model Fockian (`AedelstenContext::compute_model_fockian`), `n x n`, column-major, OVERWRITTEN; `len` its
/// length.
#[unsafe(no_mangle)]
pub extern "C" fn compute_model_fockian_c(ctx: *const AedelstenContextHandle, output: *mut f64, len: i32) -> i32 {
    if ctx.is_null() || output.is_null() || len < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let output = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
    match real.compute_model_fockian(output) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_model_fockian (C wrapper {}): {}", stringify!(compute_model_fockian_c), e);
            FAILURE
        }
    }
}

/// The overlap between this context's basis (rows) and `other`'s (columns), which may differ in basis set, geometry,
/// harmonics and contraction type: `get_n_cont_c(ctx) x get_n_cont_c(other)`, column-major, OVERWRITTEN, `len`
/// values; unscreened (see `AedelstenContext::compute_overlap_with`, also for the scratch it uses). Returns 0, or 1 on
/// failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_overlap_with_c(ctx: *const AedelstenContextHandle, other: *const AedelstenContextHandle, output: *mut f64, len: i64) -> i32 {
    if ctx.is_null() || other.is_null() || output.is_null() || len < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let other = unsafe { &*(other as *const AedelstenContext) };
    let output = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
    match real.compute_overlap_with(other, output) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_overlap_with (C wrapper {}): {}", stringify!(compute_overlap_with_c), e);
            FAILURE
        }
    }
}
