// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The relativistic Hamiltonian: the finite nucleus, the picture-change integrals, X2C.

use super::*;

/// Switches the nuclei from point charges to Gaussian charge distributions (Visscher-Dyall). Returns 0, or 1 on failure
/// (while a polarization state holds the static field).
#[unsafe(no_mangle)]
pub extern "C" fn set_finite_nuclear_model_c(ctx: *mut AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    // Visscher-Dyall, the one finite-nucleus model of the C interface.
    let model = basis::charge_carriers::FiniteModel::VisscherDyall;
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    match real.set_finite_nuclear_model(model) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_finite_nuclear_model (C wrapper {}): {}", stringify!(set_finite_nuclear_model_c), e);
            FAILURE
        }
    }
}

/// The picture-change integrals of `operator` over the primitive basis (`n = get_n_prim_unique_c`) into `output`, `n x
/// n x n_planes`: `p.Xp`, one plane per component of the operator (1 for a matrix operator), or with `spin_orbit`
/// nonzero the planes `(p X x p)` x, y, z and the scalar `p.Xp` per component (4 per component), raw (see
/// `AedelstenContext::compute_one_electron_picture_change_integrals`). The origin and order as in
/// `compute_one_electron_integrals_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_one_electron_picture_change_integrals_c(
    ctx: *mut AedelstenContextHandle,
    output: *mut f64,
    n: i32,
    operator: i32,
    n_planes: i32,
    origin_x: f64,
    origin_y: f64,
    origin_z: f64,
    order: i32,
    spin_orbit: i32,
) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let kind = match OneElectronOperatorKind::from_c_int(operator) {
        Ok(kind) => kind,
        Err(_) => {
            rank0_eprintln!("Unknown one-electron integral operator {} in C wrapper {}", operator, stringify!(compute_one_electron_picture_change_integrals_c));
            return FAILURE;
        }
    };
    let n_planes = n_planes.max(1);
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, (n * n * n_planes) as usize) };
    let moment_origin = oneint::MomentOrigin::new([origin_x, origin_y, origin_z], order.max(0) as usize);
    let spin_orbit = spin_orbit != 0;
    let result = crate::context::one_electron::resolve_one_electron_operator(real, kind, moment_origin)
        .and_then(|operator| real.compute_one_electron_picture_change_integrals(output_slice, operator, spin_orbit));
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_one_electron_picture_change_integrals (C wrapper {}): {}", stringify!(compute_one_electron_picture_change_integrals_c), e);
            FAILURE
        }
    }
}

/// The spin-free X2C one-electron Hamiltonian into `oneham` from the primitive-basis `s`, `t`, `v` and `pvp` (`n x n`
/// each, `n = get_n_prim_unique_c`; `compute_one_electron_integrals_c` with `primitive` and
/// `compute_one_electron_picture_change_integrals_c`), with `c_light` (not positive: the default). The decoupling
/// transforms are kept for the operators transformed later; an error if they are already held
/// (`free_picture_change_c`). With symmetry set, the inputs must be totally symmetric. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn construct_x2c_one_electron_hamiltonian_c(
    ctx: *mut AedelstenContextHandle,
    oneham: *mut f64,
    s: *const f64,
    t: *const f64,
    v: *const f64,
    pvp: *const f64,
    n: i32,
    c_light: f64, // <= 0.0 means "use the library default".
) -> i32 {
    if ctx.is_null() || oneham.is_null() || s.is_null() || t.is_null() || v.is_null() || pvp.is_null() || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let n2 = (n * n) as usize;
    let oneham_slice = unsafe { std::slice::from_raw_parts_mut(oneham, n2) };
    let s_slice = unsafe { std::slice::from_raw_parts(s, n2) };
    let t_slice = unsafe { std::slice::from_raw_parts(t, n2) };
    let v_slice = unsafe { std::slice::from_raw_parts(v, n2) };
    let pvp_slice = unsafe { std::slice::from_raw_parts(pvp, n2) };
    let c_light = if c_light > 0.0 { Some(c_light) } else { None };
    match real.construct_x2c_one_electron_hamiltonian(oneham_slice, s_slice, t_slice, v_slice, pvp_slice, c_light) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in construct_x2c_one_electron_hamiltonian (C wrapper {}): {}", stringify!(construct_x2c_one_electron_hamiltonian_c), e);
            FAILURE
        }
    }
}

/// Frees the X2C decoupling transforms.
#[unsafe(no_mangle)]
pub extern "C" fn free_picture_change_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_picture_change();
}

/// 1 when the X2C decoupling transforms are held, 0 when not, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn has_picture_change_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.has_picture_change() as i32
}

/// Copies the X2C decoupling transforms, each `n_prim x n_prim` column-major (`len = n_prim^2` doubles each): `u_large`
/// for plain operators, `u_small` for their pXp counterparts. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_picture_change_transforms_c(ctx: *const AedelstenContextHandle, u_large: *mut f64, u_small: *mut f64, len: i32) -> i32 {
    if ctx.is_null() || u_large.is_null() || u_small.is_null() || len < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let (u_large, u_small) = unsafe { (std::slice::from_raw_parts_mut(u_large, len as usize), std::slice::from_raw_parts_mut(u_small, len as usize)) };
    match real.get_picture_change_transforms(u_large, u_small) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in get_picture_change_transforms (C wrapper {}): {}", stringify!(get_picture_change_transforms_c), e);
            FAILURE
        }
    }
}

/// A one-electron operator in the X2C picture, the scalar part: `u_large^T X u_large + u_small^T (p.Xp) u_small` per
/// plane, from `op_plain` (X) and `op_pxp` (p.Xp), both `n x n x n_planes` column-major (`n = n_prim`; the raw planes
/// of `compute_one_electron_picture_change_integrals_c`), into `out` (same shape), symmetrised; `n_planes` 1 for a
/// matrix operator. Needs the X2C decoupling. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn transform_one_electron_operator_c(
    ctx: *const AedelstenContextHandle,
    op_plain: *const f64,
    op_pxp: *const f64,
    out: *mut f64,
    n: i32,
    n_planes: i32,
) -> i32 {
    if ctx.is_null() || op_plain.is_null() || op_pxp.is_null() || out.is_null() || n < 0 || n_planes < 1 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let len = (n as usize) * (n as usize) * (n_planes as usize);
    if super::buffers_overlap(&[(out as *const f64, len)], &[(op_plain, len), (op_pxp, len)]) {
        rank0_eprintln!("Error in C wrapper {}: the output overlaps an input", stringify!(transform_one_electron_operator_c));
        return FAILURE;
    }
    let (op_plain, op_pxp, out) =
        unsafe { (std::slice::from_raw_parts(op_plain, len), std::slice::from_raw_parts(op_pxp, len), std::slice::from_raw_parts_mut(out, len)) };
    let result = if n_planes == 1 {
        real.transform_one_electron_operator(op_plain, op_pxp, out)
    } else {
        real.transform_one_electron_operator_array3d(op_plain, op_pxp, n_planes as usize, out)
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in transform_one_electron_operator (C wrapper {}): {}", stringify!(transform_one_electron_operator_c), e);
            FAILURE
        }
    }
}

/// The spin planes of a one-electron operator in the X2C picture: `u_small^T X u_small` per plane of `op_pxp` (`n x n x
/// n_planes` column-major, `n = n_prim`; raw `(p X x p)_k` planes, to which other raw spin planes such as AMFI's may be
/// added first) into `out` (same shape), not symmetrised (the planes are antisymmetric). Needs the X2C decoupling.
/// Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn transform_one_electron_spin_operator_c(
    ctx: *const AedelstenContextHandle,
    op_pxp: *const f64,
    out: *mut f64,
    n: i32,
    n_planes: i32,
) -> i32 {
    if ctx.is_null() || op_pxp.is_null() || out.is_null() || n < 0 || n_planes < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let len = (n as usize) * (n as usize) * (n_planes as usize);
    let (op_pxp, out) = unsafe { (std::slice::from_raw_parts(op_pxp, len), std::slice::from_raw_parts_mut(out, len)) };
    match real.transform_one_electron_spin_operator(op_pxp, n_planes as usize, out) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in transform_one_electron_spin_operator (C wrapper {}): {}", stringify!(transform_one_electron_spin_operator_c), e);
            FAILURE
        }
    }
}

/// A one-electron operator of `n_components` components (1 for a matrix operator; the operator and the origin as in
/// `compute_one_electron_picture_change_integrals_c`) in the X2C picture: `scalar` (`n x n x n_components`) and `spin`
/// (`n x n x 3 n_components`, plane `3c + k` the k-th spin plane of component c), column-major, over the contracted
/// basis (`contracted` nonzero, `n = n_cont`) or the primitive one (`n = n_prim`). Needs the X2C decoupling and the
/// primitive shell pairs. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_picture_changed_one_electron_operator_c(
    ctx: *const AedelstenContextHandle,
    scalar: *mut f64,
    spin: *mut f64,
    n: i32,
    operator: i32,
    n_components: i32,
    origin_x: f64,
    origin_y: f64,
    origin_z: f64,
    order: i32,
    contracted: i32,
) -> i32 {
    if ctx.is_null() || scalar.is_null() || spin.is_null() || n < 0 || n_components < 1 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let kind = match OneElectronOperatorKind::from_c_int(operator) {
        Ok(kind) => kind,
        Err(_) => {
            rank0_eprintln!("Unknown one-electron integral operator {} in C wrapper {}", operator, stringify!(compute_picture_changed_one_electron_operator_c));
            return FAILURE;
        }
    };
    let plane = (n as usize) * (n as usize);
    let n_components = n_components as usize;
    let (scalar, spin) = unsafe {
        (std::slice::from_raw_parts_mut(scalar, plane * n_components), std::slice::from_raw_parts_mut(spin, 3 * plane * n_components))
    };
    let moment_origin = oneint::MomentOrigin::new([origin_x, origin_y, origin_z], order.max(0) as usize);
    let result = crate::context::one_electron::resolve_one_electron_operator(real, kind, moment_origin)
        .and_then(|operator| real.compute_picture_changed_one_electron_operator(operator, contracted != 0, scalar, spin));
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_picture_changed_one_electron_operator (C wrapper {}): {}", stringify!(compute_picture_changed_one_electron_operator_c), e);
            FAILURE
        }
    }
}

/// The AMFI planes over the primitive basis into `output` (`n x n x 6` column-major, `n = n_prim`): spin-same-orbit x,
/// y, z, then spin-other-orbit exchange x, y, z, raw (see `AedelstenContext::compute_amfi_integrals`). Needs the
/// primitive shell pairs and the scratch. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_amfi_integrals_c(ctx: *const AedelstenContextHandle, output: *mut f64, n: i32) -> i32 {
    if ctx.is_null() || output.is_null() || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let output = unsafe { std::slice::from_raw_parts_mut(output, (n as usize) * (n as usize) * amfi::N_AMFI_PLANES) };
    match real.compute_amfi_integrals(output) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_amfi_integrals (C wrapper {}): {}", stringify!(compute_amfi_integrals_c), e);
            FAILURE
        }
    }
}

/// Supplies the decoupling transforms (each `len = n_prim^2` doubles, as `get_picture_change_transforms_c` gives them),
/// made with `c_light` (not positive: the default); see `AedelstenContext::set_picture_change_transforms`. Returns 0,
/// or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_picture_change_transforms_c(
    ctx: *mut AedelstenContextHandle,
    u_large: *const f64,
    u_small: *const f64,
    len: i32,
    c_light: f64,
) -> i32 {
    if ctx.is_null() || u_large.is_null() || u_small.is_null() || len < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let (u_large, u_small) = unsafe { (std::slice::from_raw_parts(u_large, len as usize), std::slice::from_raw_parts(u_small, len as usize)) };
    let c_light = if c_light > 0.0 { Some(c_light) } else { None };
    match real.set_picture_change_transforms(u_large, u_small, c_light) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_picture_change_transforms (C wrapper {}): {}", stringify!(set_picture_change_transforms_c), e);
            FAILURE
        }
    }
}

/// Holds the densities of the X2C gradient with the decoupling held, from `density` (`n x n`, `n = n_cont`). See
/// `AedelstenContext::set_x2c_frozen_density`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_x2c_frozen_density_c(ctx: *mut AedelstenContextHandle, density: *const f64, n: i32) -> i32 {
    if ctx.is_null() || density.is_null() || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let density = unsafe { std::slice::from_raw_parts(density, (n as usize) * (n as usize)) };
    match real.set_x2c_frozen_density(density) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_x2c_frozen_density (C wrapper {}): {}", stringify!(set_x2c_frozen_density_c), e);
            FAILURE
        }
    }
}

/// Frees the densities `set_x2c_frozen_density_c` holds.
#[unsafe(no_mangle)]
pub extern "C" fn free_x2c_frozen_density_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_x2c_frozen_density();
}

// The gradient output and the frozen-centre mask of a C gradient call.
fn gradient_arguments<'a>(gradient: *mut f64, n_centres: i32, active: *const i32) -> (&'a mut [f64], Option<Vec<bool>>) {
    let gradient = unsafe { std::slice::from_raw_parts_mut(gradient, 3 * n_centres as usize) };
    let active = (!active.is_null()).then(|| unsafe { std::slice::from_raw_parts(active, n_centres as usize) }.iter().map(|&flag| flag != 0).collect());
    (gradient, active)
}

/// The kinetic part of the X2C one-electron energy's gradient with the decoupling held into `gradient`, `(3,
/// n_centres)` column-major, OVERWRITTEN; `active` null or `n_centres` flags (nonzero: moves). Needs
/// `set_x2c_frozen_density_c`. See `AedelstenContext::compute_x2c_frozen_kinetic_gradient`. Returns 0, or 1 on
/// failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_x2c_frozen_kinetic_gradient_c(
    ctx: *const AedelstenContextHandle,
    gradient: *mut f64,
    n_centres: i32,
    active: *const i32,
) -> i32 {
    if ctx.is_null() || gradient.is_null() || n_centres < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let (gradient, active) = gradient_arguments(gradient, n_centres, active);
    match real.compute_x2c_frozen_kinetic_gradient(gradient, active.as_deref()) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_x2c_frozen_kinetic_gradient (C wrapper {}): {}", stringify!(compute_x2c_frozen_kinetic_gradient_c), e);
            FAILURE
        }
    }
}

/// The part of the X2C one-electron energy's gradient with the decoupling held of the potential `operator` (a
/// one-electron operator code, a matrix operator) into `gradient`, `(3, n_centres)` column-major, OVERWRITTEN;
/// `active` null or `n_centres` flags (nonzero: moves). Needs `set_x2c_frozen_density_c`. See
/// `AedelstenContext::compute_x2c_frozen_potential_gradient`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_x2c_frozen_potential_gradient_c(
    ctx: *const AedelstenContextHandle,
    gradient: *mut f64,
    n_centres: i32,
    operator: i32,
    active: *const i32,
) -> i32 {
    if ctx.is_null() || gradient.is_null() || n_centres < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let kind = match OneElectronOperatorKind::from_c_int(operator) {
        Ok(kind) => kind,
        Err(_) => {
            rank0_eprintln!("Unknown one-electron integral operator {} in C wrapper {}", operator, stringify!(compute_x2c_frozen_potential_gradient_c));
            return FAILURE;
        }
    };
    let (gradient, active) = gradient_arguments(gradient, n_centres, active);
    let result = crate::context::one_electron::resolve_one_electron_operator(real, kind, oneint::MomentOrigin::new([0.0; 3], 0))
        .and_then(|operator| real.compute_x2c_frozen_potential_gradient(gradient, operator, active.as_deref()));
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_x2c_frozen_potential_gradient (C wrapper {}): {}", stringify!(compute_x2c_frozen_potential_gradient_c), e);
            FAILURE
        }
    }
}
