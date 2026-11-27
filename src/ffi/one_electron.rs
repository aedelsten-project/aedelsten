// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// One-electron integrals, their gradients, and contraction from the primitive basis.

use super::*;

/// The number of Cartesian components of a moment of order `order`, `(order + 1)(order + 2)/2`.
#[unsafe(no_mangle)]
pub extern "C" fn compute_moment_component_count_c(order: i32) -> i32 {
    cart2sph::compute_cartesian_degeneracy(order.max(0) as usize) as i32
}

/// The number of output planes of a one-electron operator (its wire key's value, `order` read by the moment
/// only): the moment's Cartesian components, 3 for the angular momentum and nabla, 0 for an operator with a plain
/// `n x n` output, -1 for an unknown operator. What `n_planes` of `compute_one_electron_integrals_c` must be for a
/// component operator.
#[unsafe(no_mangle)]
pub extern "C" fn compute_one_electron_component_count_c(operator: i32, order: i32) -> i32 {
    match OneElectronOperatorKind::from_c_int(operator) {
        Ok(kind) => kind.component_count(order.max(0) as usize).map_or(0, |count| count as i32),
        Err(_) => -1,
    }
}

/// The integrals of the one-electron operator `operator` (an `oneint_*` code) into `output`, `n x n` column-major, or
/// `n x n x n_planes` for an operator with components (`compute_one_electron_component_count_c`), over the contracted
/// basis (`n = get_n_cont_c`) or, with `primitive` nonzero, the primitive one (`n = get_n_prim_unique_c`, after
/// `set_primitive_shell_pairs_c`). The moment's origin and order, and the angular momentum's origin, from `origin_*`
/// and `order`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_one_electron_integrals_c(
    ctx: *mut AedelstenContextHandle,
    output: *mut f64,
    n: i32,
    operator: i32,
    n_planes: i32, // Only meaningful for operators with cartesian components, such as the moment integrals.
    origin_x: f64,
    origin_y: f64,
    origin_z: f64,
    order: i32,
    primitive: i32, // Nonzero: evaluate over the decontracted (primitive) basis instead of the contracted one.
) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let kind = match OneElectronOperatorKind::from_c_int(operator) {
        Ok(kind) => kind,
        Err(_) => {
            rank0_eprintln!("Unknown one-electron integral operator {} in C wrapper {}", operator, stringify!(compute_one_electron_integrals_c));
            return FAILURE;
        }
    };
    let n_planes = n_planes.max(1);
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, (n * n * n_planes) as usize) };
    let moment_origin = oneint::MomentOrigin::new([origin_x, origin_y, origin_z], order.max(0) as usize);
    let primitive = primitive != 0;
    let result = crate::context::one_electron::resolve_one_electron_operator(real, kind, moment_origin)
        .and_then(|operator| real.compute_one_electron_integrals(output_slice, operator, primitive));
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_one_electron_integrals (C wrapper {}): {}", stringify!(compute_one_electron_integrals_c), e);
            FAILURE
        }
    }
}

/// The gradient of the one-electron operator `operator` contracted with `density` (`n x n`, `n = get_n_cont_c`) into
/// `gradient`, `3 * n_centres` doubles, a column-major `(3, n_centres)` matrix, OVERWRITTEN (see
/// `AedelstenContext::compute_one_electron_gradient`); `n_centres` is `get_n_gradient_centres_c`, `active` null (every
/// centre moves) or `n_centres` flags, nonzero for a centre that moves. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_one_electron_gradient_c(
    ctx: *mut AedelstenContextHandle,
    gradient: *mut f64,
    n_centres: i32,
    density: *const f64,
    n: i32,
    operator: i32,
    active: *const i32,
) -> i32 {
    if ctx.is_null() || gradient.is_null() || density.is_null() || n < 0 || n_centres < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let kind = match OneElectronOperatorKind::from_c_int(operator) {
        Ok(kind) => kind,
        Err(_) => {
            rank0_eprintln!("Unknown one-electron integral operator {} in C wrapper {}", operator, stringify!(compute_one_electron_gradient_c));
            return FAILURE;
        }
    };
    let gradient_slice = unsafe { std::slice::from_raw_parts_mut(gradient, 3 * n_centres as usize) };
    let density_slice = unsafe { std::slice::from_raw_parts(density, (n as usize) * (n as usize)) };
    let moment_origin = oneint::MomentOrigin::new([0.0; 3], 0);
    let active_flags: Option<Vec<bool>> = if active.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(active, n_centres as usize) }.iter().map(|&flag| flag != 0).collect())
    };
    let result = crate::context::one_electron::resolve_one_electron_operator(real, kind, moment_origin)
        .and_then(|operator| real.compute_one_electron_gradient(gradient_slice, density_slice, operator, active_flags.as_deref()));
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_one_electron_gradient (C wrapper {}): {}", stringify!(compute_one_electron_gradient_c), e);
            FAILURE
        }
    }
}

/// Contracts `primitive` (`n_prim x n_prim` over the primitive basis) to the contracted basis, `C^T M C` into `output`
/// (`n_cont x n_cont`). Needs `set_primitive_shell_pairs_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn contract_one_electron_matrix_c(
    ctx: *mut AedelstenContextHandle,
    primitive: *const f64,
    n_prim: i32,
    output: *mut f64,
    n_cont: i32,
) -> i32 {
    if ctx.is_null() || primitive.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let primitive_slice = unsafe { std::slice::from_raw_parts(primitive, (n_prim * n_prim) as usize) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, (n_cont * n_cont) as usize) };
    match real.contract_one_electron_matrix(primitive_slice, output_slice) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in contract_one_electron_matrix (C wrapper {}): {}", stringify!(contract_one_electron_matrix_c), e);
            FAILURE
        }
    }
}

/// `contract_one_electron_matrix_c` for each of the `n_planes` planes of `primitive`.
#[unsafe(no_mangle)]
pub extern "C" fn contract_one_electron_array3d_c(
    ctx: *mut AedelstenContextHandle,
    primitive: *const f64,
    n_prim: i32,
    n_planes: i32,
    output: *mut f64,
    n_cont: i32,
) -> i32 {
    if ctx.is_null() || primitive.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n_planes_usize = n_planes.max(1) as usize;
    let primitive_slice = unsafe { std::slice::from_raw_parts(primitive, (n_prim * n_prim) as usize * n_planes_usize) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, (n_cont * n_cont) as usize * n_planes_usize) };
    match real.contract_one_electron_array3d(primitive_slice, n_planes_usize, output_slice) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in contract_one_electron_array3d (C wrapper {}): {}", stringify!(contract_one_electron_array3d_c), e);
            FAILURE
        }
    }
}
