// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C interface to point-group symmetry (context/point_group.rs, the `symmetry` crate): handle-based like the
// rest of ffi/mod.rs.
//
// `report` is double[4]: the irrep used, the largest change of an element, the norm removed, the norm before.
// Matrices are flat column-major n_cont x n_cont buffers owned by the caller; scratch comes from the scratch
// (set_scratch_c). Return codes and error reporting are those of ffi/mod.rs (0 or FAILURE, message on
// stderr from rank 0).

use super::{buffers_overlap, write_string, AedelstenContextHandle, FAILURE};
use crate::{AedelstenContext, Cleaning, OperatorSymmetry, OrbitalLabel, PointGroupRequest, SymmetrisationReport, SymmetryOptions};
use symmetry::Symmetry;
use std::ffi::{CStr, c_char};

macro_rules! rank0_eprintln {
    ($($arg:tt)*) => {
        if aedelsten_mpi::rank() == 0 {
            eprintln!($($arg)*);
        }
    };
}

fn context<'a>(ctx: *const AedelstenContextHandle) -> Option<&'a AedelstenContext> {
    // SAFETY: a non-null handle is a context made by set_aedelsten_context_c and not yet freed.
    (!ctx.is_null()).then(|| unsafe { &*(ctx as *const AedelstenContext) })
}

fn symmetry<'a>(ctx: *const AedelstenContextHandle, wrapper: &str) -> Option<&'a Symmetry> {
    match context(ctx)?.get_symmetry() {
        Ok(symmetry) => Some(symmetry),
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {wrapper}: {e}");
            None
        }
    }
}

fn operator(irrep: i32) -> OperatorSymmetry {
    if irrep < 0 { OperatorSymmetry::Auto } else { OperatorSymmetry::Irrep(irrep as usize) }
}

fn write_report(report: &SymmetrisationReport, out: *mut f64) {
    if !out.is_null() {
        // SAFETY: the caller passes a double[4] (see the file's comment).
        let out = unsafe { std::slice::from_raw_parts_mut(out, 4) };
        out.copy_from_slice(&[report.irrep as f64, report.max_abs_removed, report.removed_norm, report.norm]);
    }
}

fn report_error(result: Result<(), crate::AedelstenError>, wrapper: &str) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {wrapper}: {e}");
            FAILURE
        }
    }
}

/// Finds the point group of the QM centres (`point_group` null or "") or uses the named one (e.g. "C2v", with an
/// optional principal axis `principal_axis`, `double[3]` or null), and builds the AO representation of its operations
/// (see `AedelstenContext::set_symmetry`). The tolerances are `SymmetryOptions`'; not positive: the default (for
/// `average_warning`: no warning). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_symmetry_c(
    ctx: *mut AedelstenContextHandle,
    point_group: *const c_char,
    principal_axis: *const f64,
    geometry_tolerance: f64,
    strict_tolerance: f64,
    average_warning: f64,
) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    // SAFETY: as in context(), mutably: the caller does not use the handle elsewhere during the call.
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let name = if point_group.is_null() {
        ""
    } else {
        // SAFETY: a non-null point_group is a NUL-terminated string.
        match unsafe { CStr::from_ptr(point_group) }.to_str() {
            Ok(name) => name,
            Err(_) => {
                rank0_eprintln!("Error in C wrapper set_symmetry_c: the point group name is not UTF-8");
                return FAILURE;
            }
        }
    };
    let request = if name.is_empty() {
        PointGroupRequest::Highest
    } else {
        // SAFETY: a non-null principal_axis is a double[3].
        let principal_axis = (!principal_axis.is_null()).then(|| unsafe { *(principal_axis as *const [f64; 3]) });
        PointGroupRequest::Named { name, principal_axis }
    };
    let defaults = SymmetryOptions::default();
    let options = SymmetryOptions {
        geometry_tolerance: if geometry_tolerance > 0.0 { geometry_tolerance } else { defaults.geometry_tolerance },
        strict_tolerance: if strict_tolerance > 0.0 { strict_tolerance } else { defaults.strict_tolerance },
        average_warning: (average_warning > 0.0).then_some(average_warning),
    };
    report_error(real.set_symmetry(request, options), "set_symmetry_c")
}

/// Frees the point group and its AO representation.
#[unsafe(no_mangle)]
pub extern "C" fn free_symmetry_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: as in set_symmetry_c.
    unsafe { &mut *(ctx as *mut AedelstenContext) }.free_symmetry();
}

/// The tolerances symmetry was set with into `options`, `double[3]`: geometry, strict, warning (0: none). Returns 0, or
/// 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_symmetry_options_c(ctx: *const AedelstenContextHandle, options: *mut f64) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if options.is_null() {
        return FAILURE;
    }
    match real.get_symmetry_options() {
        Ok(o) => {
            // SAFETY: the caller passes a double[3].
            let out = unsafe { std::slice::from_raw_parts_mut(options, 3) };
            out.copy_from_slice(&[o.geometry_tolerance, o.strict_tolerance, o.average_warning.unwrap_or(0.0)]);
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in C wrapper get_symmetry_options_c: {e}");
            FAILURE
        }
    }
}

/// The report of the last cleaning of the library output `name` (e.g. "oneint_overlap", "oneint_external_multipole",
/// "x2c_one_electron_hamiltonian"; see `AedelstenContext::get_symmetrisation_report`) in the primitive (`primitive`
/// nonzero) or contracted basis: `found` 1 and `report` (`double[4]`: the irrep, the largest change of an element, the
/// norm removed, the norm before) when there is one, `found` 0 when not. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_symmetrisation_report_c(
    ctx: *const AedelstenContextHandle,
    name: *const c_char,
    primitive: i32,
    found: *mut i32,
    report: *mut f64,
) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if name.is_null() || found.is_null() || report.is_null() {
        return FAILURE;
    }
    // SAFETY: a NUL-terminated name; found one int.
    let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else { return FAILURE };
    match real.get_symmetrisation_report(name, primitive != 0) {
        Ok(result) => {
            unsafe { *found = result.is_some() as i32 };
            if let Some(r) = result {
                write_report(&r, report);
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in C wrapper get_symmetrisation_report_c: {e}");
            FAILURE
        }
    }
}

/// The point group's name (e.g. "C2v"), NUL-terminated into `buf` (capacity `buf_len`). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_point_group_name_c(ctx: *const AedelstenContextHandle, buf: *mut c_char, buf_len: i32) -> i32 {
    match symmetry(ctx, "get_point_group_name_c") {
        Some(symmetry) => write_string(symmetry.point_group(), buf, buf_len, "get_point_group_name_c"),
        None => FAILURE,
    }
}

/// The number of irreps, -1 if no symmetry is set.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_irreps_c(ctx: *const AedelstenContextHandle) -> i32 {
    // No message: "no symmetry set" is an answer here, not an error.
    match context(ctx).map(|real| real.get_symmetry()) {
        Some(Ok(symmetry)) => symmetry.irreps().len() as i32,
        _ => -1,
    }
}

/// Irrep `irrep`'s name (0-based, character-table order), NUL-terminated into `buf` (capacity `buf_len`). Returns 0, or
/// 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_irrep_name_c(ctx: *const AedelstenContextHandle, irrep: i32, buf: *mut c_char, buf_len: i32) -> i32 {
    let Some(symmetry) = symmetry(ctx, "get_irrep_name_c") else { return FAILURE };
    match usize::try_from(irrep).ok().and_then(|k| symmetry.irreps().get(k)) {
        Some(irrep) => write_string(&irrep.name, buf, buf_len, "get_irrep_name_c"),
        None => {
            rank0_eprintln!("Error in C wrapper get_irrep_name_c: no irrep {irrep} in {}", symmetry.point_group());
            FAILURE
        }
    }
}

/// Irrep `irrep`'s dimension, -1 out of range or without symmetry.
#[unsafe(no_mangle)]
pub extern "C" fn get_irrep_dimension_c(ctx: *const AedelstenContextHandle, irrep: i32) -> i32 {
    let Some(symmetry) = symmetry(ctx, "get_irrep_dimension_c") else { return -1 };
    usize::try_from(irrep).ok().and_then(|k| symmetry.irreps().get(k)).map_or(-1, |irrep| irrep.dimension as i32)
}

/// The index of the irrep called `name` (case-insensitive), -1 if there is none or no symmetry is set.
#[unsafe(no_mangle)]
pub extern "C" fn get_irrep_index_c(ctx: *const AedelstenContextHandle, name: *const c_char) -> i32 {
    let Some(symmetry) = symmetry(ctx, "get_irrep_index_c") else { return -1 };
    if name.is_null() {
        return -1;
    }
    // SAFETY: a non-null name is a NUL-terminated string.
    match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(name) => symmetry.irrep_index(name).map_or(-1, |k| k as i32),
        Err(_) => -1,
    }
}

/// The index of the totally symmetric irrep, -1 if no symmetry is set.
#[unsafe(no_mangle)]
pub extern "C" fn get_totally_symmetric_irrep_c(ctx: *const AedelstenContextHandle) -> i32 {
    symmetry(ctx, "get_totally_symmetric_irrep_c").map_or(-1, |symmetry| symmetry.totally_symmetric_irrep() as i32)
}

/// The irrep of the Cartesian monomial `x^lx y^ly z^lz` about the centre of mass, -1 if it is not one one-dimensional
/// irrep or no symmetry is set.
#[unsafe(no_mangle)]
pub extern "C" fn get_cartesian_irrep_c(ctx: *const AedelstenContextHandle, lx: i32, ly: i32, lz: i32) -> i32 {
    if lx < 0 || ly < 0 || lz < 0 {
        return -1;
    }
    symmetry(ctx, "get_cartesian_irrep_c")
        .and_then(|symmetry| symmetry.cartesian_irrep(lx as u32, ly as u32, lz as u32))
        .map_or(-1, |k| k as i32)
}

/// The group's standard axes into `axes`, `(3, 3)` column-major, column k standard axis k in the input's coordinates.
/// Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_symmetry_standard_axes_c(ctx: *const AedelstenContextHandle, axes: *mut f64) -> i32 {
    let Some(symmetry) = symmetry(ctx, "get_symmetry_standard_axes_c") else { return FAILURE };
    if axes.is_null() {
        return FAILURE;
    }
    // SAFETY: the caller passes a double[9]. Column k (column-major) is standard axis k, i.e. row k of standard_axes.
    let out = unsafe { std::slice::from_raw_parts_mut(axes, 9) };
    for (k, axis) in symmetry.standard_axes().iter().enumerate() {
        out[3 * k..3 * k + 3].copy_from_slice(axis);
    }
    0
}

/// The group's order h, -1 if no symmetry is set.
#[unsafe(no_mangle)]
pub extern "C" fn get_symmetry_operation_count_c(ctx: *const AedelstenContextHandle) -> i32 {
    // No message: "no symmetry set" is an answer here, not an error.
    match context(ctx).map(|real| real.get_symmetry_operation_count()) {
        Some(Ok(h)) => h as i32,
        _ => -1,
    }
}

/// Operation `op` (0..h) into `matrix`, `(3, 3)` column-major in the input's axes: a displacement v of a centre goes to
/// `m v` at the centre's image. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_symmetry_operation_matrix_c(ctx: *const AedelstenContextHandle, op: i32, matrix: *mut f64) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if matrix.is_null() {
        return FAILURE;
    }
    let Ok(op) = usize::try_from(op) else {
        rank0_eprintln!("Error in C wrapper get_symmetry_operation_matrix_c: operation {op}");
        return FAILURE;
    };
    match real.get_symmetry_operation_matrix(op) {
        Ok(m) => {
            // SAFETY: the caller passes a double[9]; column-major, element (i, k) at i + 3 k.
            let out = unsafe { std::slice::from_raw_parts_mut(matrix, 9) };
            for (k, column) in out.chunks_exact_mut(3).enumerate() {
                for (i, value) in column.iter_mut().enumerate() {
                    *value = m[i][k];
                }
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in C wrapper get_symmetry_operation_matrix_c: {e}");
            FAILURE
        }
    }
}

/// The number of QM centres the group acts on (ghosts included), -1 if no symmetry is set.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_symmetry_centres_c(ctx: *const AedelstenContextHandle) -> i32 {
    match context(ctx).map(|real| real.get_n_symmetry_centres()) {
        Some(Ok(n)) => n as i32,
        _ => -1,
    }
}

/// Where operation `op` takes each QM centre: `images[a]` the centre (0-based) it takes centre a to; `n` is
/// `get_n_symmetry_centres_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_symmetry_centre_images_c(ctx: *const AedelstenContextHandle, op: i32, images: *mut i32, n: i32) -> i32 {
    let Some(symmetry) = symmetry(ctx, "get_symmetry_centre_images_c") else { return FAILURE };
    let valid = usize::try_from(op).is_ok_and(|op| op < symmetry.operation_count()) && usize::try_from(n) == Ok(symmetry.centre_count());
    if !valid || images.is_null() {
        rank0_eprintln!(
            "Error in C wrapper get_symmetry_centre_images_c: operation {op} of {}, {n} centres of {}",
            symmetry.operation_count(),
            symmetry.centre_count()
        );
        return FAILURE;
    }
    // SAFETY: the caller passes an int32[n].
    let out = unsafe { std::slice::from_raw_parts_mut(images, n as usize) };
    for (a, image) in out.iter_mut().enumerate() {
        *image = symmetry.centre_image(a, op as usize) as i32;
    }
    0
}

/// Cleans the totally symmetric `matrix` (`n_cont x n_cont`, `len` values; F, D, h1) in place: `average` 0 strict
/// (refused above the strict tolerance, the matrix left alone), 1 averaged over the group. `report` (`double[4]`, as in
/// `get_symmetrisation_report_c`) says what was removed. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn symmetrise_matrix_c(ctx: *const AedelstenContextHandle, matrix: *mut f64, len: i32, average: i32, report: *mut f64) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if matrix.is_null() || len < 0 {
        return FAILURE;
    }
    // SAFETY: matrix has len doubles.
    let matrix = unsafe { std::slice::from_raw_parts_mut(matrix, len as usize) };
    let cleaning = match average {
        0 => Cleaning::Strict,
        1 => Cleaning::Average,
        _ => {
            rank0_eprintln!("Error in C wrapper symmetrise_matrix_c: average must be 0 or 1, got {average}");
            return FAILURE;
        }
    };
    report_error(real.symmetrise_matrix(matrix, cleaning).map(|r| write_report(&r, report)), "symmetrise_matrix_c")
}

/// What cleaning `matrix` would remove, the matrix untouched: `irrep` how it transforms (an irrep index, or negative
/// for whichever irrep carries most of it), `report` as in `symmetrise_matrix_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn analyse_matrix_c(ctx: *const AedelstenContextHandle, matrix: *const f64, len: i32, irrep: i32, report: *mut f64) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if matrix.is_null() || len < 0 || report.is_null() {
        return FAILURE;
    }
    // SAFETY: matrix has len doubles.
    let matrix = unsafe { std::slice::from_raw_parts(matrix, len as usize) };
    report_error(real.analyse_matrix(matrix, operator(irrep)).map(|r| write_report(&r, report)), "analyse_matrix_c")
}

/// The norm of each irrep's part of `matrix` (`n_cont x n_cont`) into `norms` (`n_norms`, the number of irreps).
/// Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_irrep_norms_c(ctx: *const AedelstenContextHandle, matrix: *const f64, len: i32, norms: *mut f64, n_norms: i32) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if matrix.is_null() || norms.is_null() || len < 0 || n_norms < 0 {
        return FAILURE;
    }
    // SAFETY: matrix has len doubles, norms n_norms.
    let (matrix, norms) = unsafe { (std::slice::from_raw_parts(matrix, len as usize), std::slice::from_raw_parts_mut(norms, n_norms as usize)) };
    report_error(real.irrep_norms(matrix, norms), "get_irrep_norms_c")
}

/// Symmetrizes the MO coefficients `orbitals` (`(n_cont, n_mo)` column-major, `len` values, any `n_mo`) in place: each
/// MO projected onto its dominant irrep, degenerate partners aligned. Per MO its irrep into `irreps` and, for a
/// degenerate irrep, its partner component into `components` (may be null); the largest coefficient norm dropped into
/// `dropped` (may be null). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn symmetrise_orbitals_c(
    ctx: *const AedelstenContextHandle,
    orbitals: *mut f64,
    len: i32,
    n_mo: i32,
    irreps: *mut i32,
    components: *mut i32,
    dropped: *mut f64,
) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if orbitals.is_null() || irreps.is_null() || len < 0 || n_mo < 0 {
        return FAILURE;
    }
    let n_mo = n_mo as usize;
    // SAFETY: orbitals has len doubles, irreps n_mo ints, a non-null components n_mo ints.
    let orbitals = unsafe { std::slice::from_raw_parts_mut(orbitals, len as usize) };
    let irreps = unsafe { std::slice::from_raw_parts_mut(irreps, n_mo) };
    let components = (!components.is_null()).then(|| unsafe { std::slice::from_raw_parts_mut(components, n_mo) });
    // The labels come back as Rust structs: one n_mo-long conversion buffer to the C caller's int32 arrays.
    let mut labels = vec![OrbitalLabel::default(); n_mo];
    let result = real.symmetrise_orbitals(orbitals, &mut labels).map(|norm| {
        for (out, label) in irreps.iter_mut().zip(&labels) {
            *out = label.irrep as i32;
        }
        if let Some(components) = components {
            for (out, label) in components.iter_mut().zip(&labels) {
                *out = label.component as i32;
            }
        }
        if !dropped.is_null() {
            // SAFETY: a non-null dropped is one double.
            unsafe { *dropped = norm };
        }
    });
    report_error(result, "symmetrise_orbitals_c")
}

/// How many AOs span each irrep into `counts` (`n`, the number of irreps). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_ao_irrep_counts_c(ctx: *const AedelstenContextHandle, counts: *mut i32, n: i32) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if counts.is_null() || n < 0 {
        return FAILURE;
    }
    let result = real.get_ao_irrep_counts().and_then(|found| {
        if found.len() != n as usize {
            return Err(crate::AedelstenError::InvalidMatrixDimensions { expected: found.len(), actual: n as usize });
        }
        // SAFETY: the caller passes an int32[n].
        let out = unsafe { std::slice::from_raw_parts_mut(counts, n as usize) };
        for (o, &count) in out.iter_mut().zip(found) {
            *o = count as i32;
        }
        Ok(())
    });
    report_error(result, "get_ao_irrep_counts_c")
}

/// Operation `op` applied to the MO coefficients `coefficients` (`(n_cont, n_mo)` column-major, `len = n_cont n_mo`):
/// `out` the coefficients of each `R phi`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn apply_symmetry_operation_c(ctx: *const AedelstenContextHandle, op: i32, coefficients: *const f64, out: *mut f64, len: i32, n_mo: i32) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if coefficients.is_null() || out.is_null() || op < 0 || len < 0 || n_mo < 0 {
        return FAILURE;
    }
    if buffers_overlap(&[(out as *const f64, len as usize)], &[(coefficients, len as usize)]) {
        rank0_eprintln!("Error in C wrapper apply_symmetry_operation_c: the output overlaps the coefficients");
        return FAILURE;
    }
    // SAFETY: both buffers hold len doubles.
    let coefficients = unsafe { std::slice::from_raw_parts(coefficients, len as usize) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, len as usize) };
    report_error(real.apply_symmetry_operation(op as usize, coefficients, n_mo as usize, out), "apply_symmetry_operation_c")
}

/// The character of irrep `irrep` at operation `op` into `character`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_irrep_character_c(ctx: *const AedelstenContextHandle, irrep: i32, op: i32, character: *mut f64) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if character.is_null() || irrep < 0 || op < 0 {
        return FAILURE;
    }
    // SAFETY: a non-null character is one double.
    report_error(real.get_irrep_character(irrep as usize, op as usize).map(|chi| unsafe { *character = chi }), "get_irrep_character_c")
}

/// `Gamma(R)` of irrep `irrep` at operation `op` into `matrix` (`(d, d)` column-major, `len = d^2`) for the partner
/// components `symmetrise_orbitals_c` aligns; `aligned` 0 (the matrix untouched) when the irrep has none. Returns 0, or
/// 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_irrep_matrix_c(ctx: *const AedelstenContextHandle, irrep: i32, op: i32, matrix: *mut f64, len: i32, aligned: *mut i32) -> i32 {
    let Some(real) = context(ctx) else { return FAILURE };
    if matrix.is_null() || aligned.is_null() || irrep < 0 || op < 0 || len < 0 {
        return FAILURE;
    }
    // SAFETY: matrix holds len doubles, a non-null aligned one int32.
    let out = unsafe { std::slice::from_raw_parts_mut(matrix, len as usize) };
    report_error(real.get_irrep_matrix(irrep as usize, op as usize, out).map(|found| unsafe { *aligned = found as i32 }), "get_irrep_matrix_c")
}

