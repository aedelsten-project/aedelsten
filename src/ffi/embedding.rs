// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The environment: AIMPs, external multipoles, the classical embedding energies; and the induced-dipole
// polarization (below).

// The induced-dipole embedding: one flat call per step (`AedelstenContext::set_polarization` and after).
//
// Every matrix is a flat column-major buffer of n_cont^2 doubles (n_mat of them, back to back, for the
// response call) owned by the caller; nothing is retained after a call returns. Return codes and error
// reporting are those of ffi/mod.rs (0 or FAILURE, message on stderr from rank 0).

use super::{buffers_overlap, AedelstenContextHandle, FAILURE};
use crate::wire_enum::c_int_enum;
use crate::AedelstenError;
use crate::{AedelstenContext, Damping, PolarizationOptions, SiteChunking, SiteSelection, SolverOptions, StaticFieldPolicy};

/// The energy of the QM region's nuclei and core AIMPs in the external multipoles' potential; NaN without external
/// multipoles, 0 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn compute_nuclear_multipole_interaction_energy_c(ctx: *mut AedelstenContextHandle) -> f64 {
    if ctx.is_null() {
        return 0.0;
    }
    let real = unsafe { &*(ctx as *mut AedelstenContext) };
    real.compute_nuclear_multipole_interaction_energy()
}

/// The energy of the QM region's nuclei and core AIMPs in the embedding AIMPs' screened potentials; 0 without embedding
/// AIMPs.
#[unsafe(no_mangle)]
pub extern "C" fn compute_nuclear_aimp_embedding_energy_c(
    ctx: *mut AedelstenContextHandle,
) -> f64 {
    if ctx.is_null() {
        return 0.0;
    }
    let real = unsafe { &*(ctx as *mut AedelstenContext) };
    real.compute_nuclear_aimp_embedding_energy()
}

/// 1 when the QM region has core AIMPs, 0 when not, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn has_core_aimp_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.has_core_aimp() as i32
}

/// 1 when the input has embedding AIMPs, 0 when not, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn has_embedding_aimp_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.has_embedding_aimp() as i32
}

/// 1 when external multipoles are set, 0 when not, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn has_external_multipoles_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.has_external_multipoles() as i32
}

/// Reads the external point charges and multipoles of the xfield file `filename_c` (see
/// `AedelstenContext::set_external_multipoles`). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_external_multipoles_c(
    ctx: *mut AedelstenContextHandle,
    filename_c: *const std::ffi::c_char,
) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let filename = match unsafe { std::ffi::CStr::from_ptr(filename_c) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            rank0_eprintln!("Invalid (non-UTF8) filename in C wrapper {}", stringify!(set_external_multipoles_c));
            return FAILURE;
        }
    };
    match real.set_external_multipoles(filename) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_external_multipoles (C wrapper {}): {}", stringify!(set_external_multipoles_c), e);
            FAILURE
        }
    }
}

/// Frees the external multipoles.
#[unsafe(no_mangle)]
pub extern "C" fn free_external_multipoles_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_external_multipoles();
}

/// Builds the embedding AIMPs' projection operators from the input's embedding AIMPs (see
/// `AedelstenContext::set_embedding_aimp_operators`); `filename_c` must be null. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_embedding_aimp_operators_c(
    ctx: *mut AedelstenContextHandle,
    filename_c: *const std::ffi::c_char,
) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    if !filename_c.is_null() {
        if real.has_embedding_aimp() {
            rank0_eprintln!(
                "set_embedding_aimp_operators_c: the main input file already carries embedding AIMP regions; \
                 passing a separate filename too is ambiguous (which source did you mean?) and is rejected"
            );
        } else {
            rank0_eprintln!(
                "set_embedding_aimp_operators_c: loading embedding AIMPs from a separate file is not yet supported; \
                 pass a null filename to use the ones already parsed from the main input file"
            );
        }
        return FAILURE;
    }
    match real.set_embedding_aimp_operators() {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_embedding_aimp_operators (C wrapper {}): {}", stringify!(set_embedding_aimp_operators_c), e);
            FAILURE
        }
    }
}

/// Frees the embedding AIMPs' projection operators.
#[unsafe(no_mangle)]
pub extern "C" fn free_embedding_aimp_operators_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_embedding_aimp_operators();
}

macro_rules! rank0_eprintln {
    ($($arg:tt)*) => {
        if aedelsten_mpi::rank() == 0 {
            eprintln!($($arg)*);
        }
    };
}

/// The options of `set_polarization_c`. A negative value in a numeric field means "the default", so a
/// caller only sets what it cares about after `aedelsten_polarization_options_default_c`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AedelstenPolarizationOptions {
    /// 0: no damping, 1: Thole.
    pub damping: i32,
    /// Thole parameter a; negative: 2.1304.
    pub thole_a: f64,
    /// Convergence threshold of the dipole solver; negative: 1e-8.
    pub threshold: f64,
    /// Maximum solver iterations; negative: the default.
    pub max_iterations: i32,
    /// 0: keep the static field in memory, 1: recompute it at every call.
    pub static_field: i32,
    /// Sites a field pass over the sites not kept hands out as one piece over the ranks; not positive: derive it from
    /// the budget below.
    pub chunk_sites: i32,
    /// The sites per piece as many as a buffer of their dense electric-field integrals (`3 n_cont^2` doubles a site)
    /// of this size holds (in `chunk_budget_unit`, a byte unit); negative: the default.
    pub chunk_budget: i32,
    pub chunk_budget_unit: i32,
    /// The most densities treated at once by `compute_induction_response_c`; 0: no response calls.
    pub response_batch: i32,
    /// The most states `compute_induction_states_c` can hold, each with its own dipoles; 0: no state calls.
    pub state_slots: i32,
    /// Memory budget for keeping the field integrals between calls (in `resident_budget_unit`, a byte unit); 0: never
    /// kept.
    pub resident_budget: i32,
    pub resident_budget_unit: i32,
}

/// Fills `options` with the defaults, to edit before `set_polarization_c`.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_polarization_options_default_c(options: *mut AedelstenPolarizationOptions) {
    if options.is_null() {
        return;
    }
    unsafe {
        *options = AedelstenPolarizationOptions {
            damping: 0,
            thole_a: -1.0,
            threshold: -1.0,
            max_iterations: -1,
            static_field: 0,
            chunk_sites: -1,
            chunk_budget: -1,
            chunk_budget_unit: 1,
            response_batch: 0,
            state_slots: 0,
            resident_budget: 0,
            resident_budget_unit: 1,
        };
    }
}

fn options_from_c(c: &AedelstenPolarizationOptions) -> Result<PolarizationOptions, String> {
    let defaults = SolverOptions::default();
    let damping = match c.damping {
        0 => Damping::None,
        1 if c.thole_a < 0.0 => Damping::thole(),
        1 => Damping::Thole { a: c.thole_a },
        other => return Err(format!("unknown damping {other} (0: none, 1: Thole)")),
    };
    let static_field = match c.static_field {
        0 => StaticFieldPolicy::Cache,
        1 => StaticFieldPolicy::Recompute,
        other => return Err(format!("unknown static field policy {other} (0: cache, 1: recompute)")),
    };
    let chunking = if c.chunk_sites > 0 {
        SiteChunking::Sites(c.chunk_sites as usize)
    } else if c.chunk_budget < 0 {
        SiteChunking::Budget(None)
    } else {
        let unit = arrays::scratch::ByteUnit::from_c_int(c.chunk_budget_unit)?;
        SiteChunking::Budget(Some((c.chunk_budget as usize, unit)))
    };
    let resident_field_budget = match c.resident_budget {
        size if size < 0 => return Err(format!("negative resident_budget {size}")),
        0 => None,
        size => Some((size as usize, arrays::scratch::ByteUnit::from_c_int(c.resident_budget_unit)?)),
    };
    if c.response_batch < 0 || c.state_slots < 0 {
        return Err(format!("negative response_batch {} or state_slots {}", c.response_batch, c.state_slots));
    }
    Ok(PolarizationOptions {
        damping,
        solver: SolverOptions {
            threshold: if c.threshold < 0.0 { defaults.threshold } else { c.threshold },
            max_iterations: if c.max_iterations < 0 { defaults.max_iterations } else { c.max_iterations as usize },
        },
        static_field,
        chunking,
        response_batch: c.response_batch as usize,
        state_slots: c.state_slots as usize,
        resident_field_budget,
    })
}

fn report(result: Result<(), impl std::fmt::Display>, wrapper: &str) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {}", wrapper, e);
            FAILURE
        }
    }
}

// Lifecycle calls take `&mut` on the context; compute calls only `&`, so a caller may run them concurrently
// on one handle in the ways the Rust API allows.
macro_rules! ctx_mut {
    ($ctx:expr) => {
        if $ctx.is_null() { return FAILURE; } else { unsafe { &mut *($ctx as *mut AedelstenContext) } }
    };
}
macro_rules! ctx_ref {
    ($ctx:expr) => {
        if $ctx.is_null() { return FAILURE; } else { unsafe { &*($ctx as *const AedelstenContext) } }
    };
}

/// Reads the polarizable sites of the xfield file `filename_c` (see `AedelstenContext::set_polarizable_sites`). Returns
/// 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_polarizable_sites_c(ctx: *mut AedelstenContextHandle, filename_c: *const std::ffi::c_char) -> i32 {
    let real = ctx_mut!(ctx);
    let filename = match unsafe { std::ffi::CStr::from_ptr(filename_c) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            rank0_eprintln!("Invalid (non-UTF8) filename in C wrapper {}", stringify!(set_polarizable_sites_c));
            return FAILURE;
        }
    };
    report(real.set_polarizable_sites(filename), stringify!(set_polarizable_sites_c))
}

/// Frees the polarizable sites and the polarization state derived from them.
#[unsafe(no_mangle)]
pub extern "C" fn free_polarizable_sites_c(ctx: *mut AedelstenContextHandle) {
    if !ctx.is_null() {
        unsafe { &mut *(ctx as *mut AedelstenContext) }.free_polarizable_sites();
    }
}

/// The number of polarizable sites (the length of the dipole vector is 3 times this), or -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_polarizable_sites_c(ctx: *const AedelstenContextHandle) -> i64 {
    if ctx.is_null() {
        return -1;
    }
    match unsafe { &*(ctx as *const AedelstenContext) }.count_electric_field_sites(SiteSelection::Polarizable) {
        Ok(n) => n as i64,
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {}", stringify!(get_n_polarizable_sites_c), e);
            -1
        }
    }
}

/// Sets up the induced-dipole polarization with `options` (see `AedelstenContext::set_polarization`), once per
/// geometry, after `set_polarizable_sites_c`. Collective under MPI. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_polarization_c(ctx: *mut AedelstenContextHandle, options: *const AedelstenPolarizationOptions) -> i32 {
    let real = ctx_mut!(ctx);
    if options.is_null() {
        return FAILURE;
    }
    let options = match options_from_c(unsafe { &*options }) {
        Ok(o) => o,
        Err(e) => {
            rank0_eprintln!("Invalid options in C wrapper {}: {}", stringify!(set_polarization_c), e);
            return FAILURE;
        }
    };
    report(real.set_polarization(options), stringify!(set_polarization_c))
}

/// Frees the polarization state.
#[unsafe(no_mangle)]
pub extern "C" fn free_polarization_c(ctx: *mut AedelstenContextHandle) {
    if !ctx.is_null() {
        unsafe { &mut *(ctx as *mut AedelstenContext) }.free_polarization();
    }
}

/// dE_ind/dX at fixed `density` (n_cont^2 doubles), from the dipoles of the last `compute_induction_c` (call it with
/// the same density first): `gradient` is `3 * n_centres` doubles, a column-major `(3, n_centres)` matrix, OVERWRITTEN;
/// `n_centres` must be `get_n_gradient_centres_c`; `active` as in `compute_one_electron_gradient_c` (NULL = all).
#[unsafe(no_mangle)]
pub extern "C" fn compute_induction_gradient_c(
    ctx: *const AedelstenContextHandle,
    gradient: *mut f64,
    n_centres: i32,
    density: *const f64,
    active: *const i32,
) -> i32 {
    let real = ctx_ref!(ctx);
    if gradient.is_null() || density.is_null() || n_centres < 0 {
        return FAILURE;
    }
    let n2 = real.get_n_cont() * real.get_n_cont();
    let density = unsafe { std::slice::from_raw_parts(density, n2) };
    let gradient = unsafe { std::slice::from_raw_parts_mut(gradient, 3 * n_centres as usize) };
    let active_flags: Option<Vec<bool>> = if active.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(active, n_centres as usize) }.iter().map(|&flag| flag != 0).collect())
    };
    match real.compute_induction_gradient(gradient, density, active_flags.as_deref()) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {}", stringify!(compute_induction_gradient_c), e);
            FAILURE
        }
    }
}

/// `density` and `v_ind` are n_cont^2 doubles; `energy` and `iterations` (either may be null) receive E_ind
/// and the solver's iteration count.
#[unsafe(no_mangle)]
pub extern "C" fn compute_induction_c(
    ctx: *const AedelstenContextHandle,
    density: *const f64,
    v_ind: *mut f64,
    energy: *mut f64,
    iterations: *mut i32,
) -> i32 {
    let real = ctx_ref!(ctx);
    if density.is_null() || v_ind.is_null() {
        return FAILURE;
    }
    let n2 = real.get_n_cont() * real.get_n_cont();
    let (density, v_ind) = unsafe { (std::slice::from_raw_parts(density, n2), std::slice::from_raw_parts_mut(v_ind, n2)) };
    match real.compute_induction(density, v_ind) {
        Ok(r) => {
            unsafe {
                if !energy.is_null() {
                    *energy = r.energy;
                }
                if !iterations.is_null() {
                    *iterations = r.iterations as i32;
                }
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {}", stringify!(compute_induction_c), e);
            FAILURE
        }
    }
}

/// `compute_induction_c` with the dipoles solved to `threshold` (positive) for this call, in place of the setup's
/// (`AedelstenContext::compute_induction_within`).
#[unsafe(no_mangle)]
pub extern "C" fn compute_induction_within_c(
    ctx: *const AedelstenContextHandle,
    density: *const f64,
    v_ind: *mut f64,
    threshold: f64,
    energy: *mut f64,
    iterations: *mut i32,
) -> i32 {
    let real = ctx_ref!(ctx);
    if density.is_null() || v_ind.is_null() {
        return FAILURE;
    }
    let n2 = real.get_n_cont() * real.get_n_cont();
    if buffers_overlap(&[(v_ind as *const f64, n2)], &[(density, n2)]) {
        rank0_eprintln!("Error in C wrapper {}: the density and the operator overlap", stringify!(compute_induction_within_c));
        return FAILURE;
    }
    let (density, v_ind) = unsafe { (std::slice::from_raw_parts(density, n2), std::slice::from_raw_parts_mut(v_ind, n2)) };
    match real.compute_induction_within(density, v_ind, threshold) {
        Ok(r) => {
            unsafe {
                if !energy.is_null() {
                    *energy = r.energy;
                }
                if !iterations.is_null() {
                    *iterations = r.iterations as i32;
                }
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {}", stringify!(compute_induction_within_c), e);
            FAILURE
        }
    }
}

/// How this rank holds the field integrals of its share of the polarizable sites
/// (`AedelstenContext::get_polarization_storage`): `storage` (4 values) receives the sites of the share, those kept in
/// memory, the bytes of one site's kept integrals, and the bytes that keep the whole share in memory.
#[unsafe(no_mangle)]
pub extern "C" fn get_polarization_storage_c(ctx: *const AedelstenContextHandle, storage: *mut i64) -> i32 {
    let real = ctx_ref!(ctx);
    if storage.is_null() {
        return FAILURE;
    }
    match real.get_polarization_storage() {
        Ok(s) => {
            let out = unsafe { std::slice::from_raw_parts_mut(storage, 4) };
            for (value, count) in out.iter_mut().zip([s.sites, s.sites_in_memory, s.bytes_per_site, s.bytes_to_keep]) {
                *value = i64::try_from(count).unwrap_or(i64::MAX);
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {}", stringify!(get_polarization_storage_c), e);
            FAILURE
        }
    }
}

/// The memory budget of this rank's kept field integrals (`AedelstenContext::set_resident_field_budget`): `budget` in
/// `budget_unit` (a byte unit); 0 keeps none. Collective.
#[unsafe(no_mangle)]
pub extern "C" fn set_resident_field_budget_c(ctx: *mut AedelstenContextHandle, budget: i32, budget_unit: i32) -> i32 {
    let real = ctx_mut!(ctx);
    if budget < 0 {
        return FAILURE;
    }
    let unit = match arrays::scratch::ByteUnit::from_c_int(budget_unit) {
        Ok(unit) => unit,
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {e}", stringify!(set_resident_field_budget_c));
            return FAILURE;
        }
    };
    report(real.set_resident_field_budget((budget > 0).then_some((budget as usize, unit))), stringify!(set_resident_field_budget_c))
}

/// `densities` and `operators` are `n_mat` consecutive n_cont^2 blocks.
#[unsafe(no_mangle)]
pub extern "C" fn compute_induction_response_c(
    ctx: *const AedelstenContextHandle,
    densities: *const f64,
    n_mat: i32,
    operators: *mut f64,
) -> i32 {
    let real = ctx_ref!(ctx);
    if n_mat < 0 || (n_mat > 0 && (densities.is_null() || operators.is_null())) {
        return FAILURE;
    }
    let len = real.get_n_cont() * real.get_n_cont() * n_mat as usize;
    let (densities, operators) = if n_mat == 0 {
        (&[][..], &mut [][..])
    } else {
        unsafe { (std::slice::from_raw_parts(densities, len), std::slice::from_raw_parts_mut(operators, len)) }
    };
    report(real.compute_induction_response(densities, n_mat as usize, operators), stringify!(compute_induction_response_c))
}

/// The induction's couplings between densities (`AedelstenContext::compute_induction_couplings`): `densities` is
/// `n_mat` consecutive n_cont^2 blocks, `couplings` an `n_mat x n_mat` column-major matrix, overwritten; the two must
/// not overlap.
#[unsafe(no_mangle)]
pub extern "C" fn compute_induction_couplings_c(
    ctx: *const AedelstenContextHandle,
    densities: *const f64,
    n_mat: i32,
    couplings: *mut f64,
) -> i32 {
    let real = ctx_ref!(ctx);
    if n_mat < 0 || (n_mat > 0 && (densities.is_null() || couplings.is_null())) {
        return FAILURE;
    }
    let k = n_mat as usize;
    let len = real.get_n_cont() * real.get_n_cont() * k;
    if k > 0 && buffers_overlap(&[(couplings as *const f64, k * k)], &[(densities, len)]) {
        rank0_eprintln!("Error in C wrapper {}: the densities and the couplings overlap", stringify!(compute_induction_couplings_c));
        return FAILURE;
    }
    let (densities, couplings) = if k == 0 {
        (&[][..], &mut [][..])
    } else {
        unsafe { (std::slice::from_raw_parts(densities, len), std::slice::from_raw_parts_mut(couplings, k * k)) }
    };
    report(real.compute_induction_couplings(densities, k, couplings), stringify!(compute_induction_couplings_c))
}

/// The effective external field operators (`AedelstenContext::compute_effective_external_field`): `operators` is
/// three consecutive n_cont^2 blocks, x, y, z, overwritten.
#[unsafe(no_mangle)]
pub extern "C" fn compute_effective_external_field_c(ctx: *const AedelstenContextHandle, operators: *mut f64) -> i32 {
    let real = ctx_ref!(ctx);
    if operators.is_null() {
        return FAILURE;
    }
    let len = 3 * real.get_n_cont() * real.get_n_cont();
    let operators = unsafe { std::slice::from_raw_parts_mut(operators, len) };
    report(real.compute_effective_external_field(operators), stringify!(compute_effective_external_field_c))
}

/// `length` must be 3 times the number of polarizable sites.
#[unsafe(no_mangle)]
pub extern "C" fn get_induced_dipoles_c(ctx: *const AedelstenContextHandle, dipoles: *mut f64, length: i64) -> i32 {
    let real = ctx_ref!(ctx);
    if dipoles.is_null() || length < 0 {
        return FAILURE;
    }
    let out = unsafe { std::slice::from_raw_parts_mut(dipoles, length as usize) };
    report(real.get_induced_dipoles(out), stringify!(get_induced_dipoles_c))
}

/// The start of the next `compute_induction_c`'s solve: `length` (3 times the number of polarizable sites) values,
/// as `get_induced_dipoles_c` gives them.
#[unsafe(no_mangle)]
pub extern "C" fn set_induced_dipoles_c(ctx: *mut AedelstenContextHandle, dipoles: *const f64, length: i64) -> i32 {
    let real = ctx_mut!(ctx);
    if dipoles.is_null() || length < 0 {
        return FAILURE;
    }
    let values = unsafe { std::slice::from_raw_parts(dipoles, length as usize) };
    report(real.set_induced_dipoles(values), stringify!(set_induced_dipoles_c))
}

/// The number of embedding AIMPs, or -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_embedding_aimps_c(ctx: *const AedelstenContextHandle) -> i64 {
    if ctx.is_null() {
        return -1;
    }
    unsafe { &*(ctx as *const AedelstenContext) }.get_n_embedding_aimps() as i64
}

/// Every embedding AIMP's site in input order: `atomic_numbers` (0 when its block's name is not an element symbol),
/// `positions` (x, y, z per site, bohr) and `charges`, `n` sites (`get_n_embedding_aimps_c`). The outputs must not
/// overlap.
#[unsafe(no_mangle)]
pub extern "C" fn get_embedding_aimp_sites_c(ctx: *const AedelstenContextHandle, atomic_numbers: *mut i32, positions: *mut f64, charges: *mut f64, n: i64) -> i32 {
    let real = ctx_ref!(ctx);
    if atomic_numbers.is_null() || positions.is_null() || charges.is_null() || n < 0 {
        return FAILURE;
    }
    let n = n as usize;
    // The atomic numbers' bytes counted in doubles, rounded up.
    let buffers = [(atomic_numbers as *const f64, n.div_ceil(2)), (positions as *const f64, 3 * n), (charges as *const f64, n)];
    if buffers_overlap(&buffers, &[]) {
        rank0_eprintln!("Error in C wrapper {}: the outputs overlap", stringify!(get_embedding_aimp_sites_c));
        return FAILURE;
    }
    let mut sites = vec![crate::EmbeddingAimpSite { atomic_number: 0, position: [0.0; 3], charge: 0.0 }; n];
    if report(real.get_embedding_aimp_sites(&mut sites), stringify!(get_embedding_aimp_sites_c)) != 0 {
        return FAILURE;
    }
    let (z, xyz, q) = unsafe {
        (std::slice::from_raw_parts_mut(atomic_numbers, n), std::slice::from_raw_parts_mut(positions, 3 * n), std::slice::from_raw_parts_mut(charges, n))
    };
    for (site, ((z, xyz), q)) in sites.iter().zip(z.iter_mut().zip(xyz.chunks_exact_mut(3)).zip(q.iter_mut())) {
        *z = i32::from(site.atomic_number);
        xyz.copy_from_slice(&site.position);
        *q = site.charge;
    }
    0
}

/// One state-specific induction per density: `densities` and `operators` are `n_states` consecutive
/// n_cont^2 blocks, `energies` receives `n_states` values (weight them for a state average).
#[unsafe(no_mangle)]
pub extern "C" fn compute_induction_states_c(
    ctx: *const AedelstenContextHandle,
    densities: *const f64,
    n_states: i32,
    operators: *mut f64,
    energies: *mut f64,
) -> i32 {
    let real = ctx_ref!(ctx);
    if n_states < 0 || (n_states > 0 && (densities.is_null() || operators.is_null() || energies.is_null())) {
        return FAILURE;
    }
    let k = n_states as usize;
    let len = real.get_n_cont() * real.get_n_cont() * k;
    let (densities, operators, energies) = if k == 0 {
        (&[][..], &mut [][..], &mut [][..])
    } else {
        unsafe { (std::slice::from_raw_parts(densities, len), std::slice::from_raw_parts_mut(operators, len), std::slice::from_raw_parts_mut(energies, k)) }
    };
    report(real.compute_induction_states(densities, k, operators, energies), stringify!(compute_induction_states_c))
}

/// `compute_induction_states_c` with every state's dipoles solved to `threshold` (positive) for this call, in place of
/// the setup's (`AedelstenContext::compute_induction_states_within`).
#[unsafe(no_mangle)]
pub extern "C" fn compute_induction_states_within_c(
    ctx: *const AedelstenContextHandle,
    densities: *const f64,
    n_states: i32,
    operators: *mut f64,
    energies: *mut f64,
    threshold: f64,
) -> i32 {
    let real = ctx_ref!(ctx);
    if n_states < 0 || (n_states > 0 && (densities.is_null() || operators.is_null() || energies.is_null())) {
        return FAILURE;
    }
    let k = n_states as usize;
    let len = real.get_n_cont() * real.get_n_cont() * k;
    if k > 0 && buffers_overlap(&[(operators as *const f64, len), (energies as *const f64, k)], &[(densities, len)]) {
        rank0_eprintln!("Error in C wrapper {}: the densities, operators and energies overlap", stringify!(compute_induction_states_within_c));
        return FAILURE;
    }
    let (densities, operators, energies) = if k == 0 {
        (&[][..], &mut [][..], &mut [][..])
    } else {
        unsafe { (std::slice::from_raw_parts(densities, len), std::slice::from_raw_parts_mut(operators, len), std::slice::from_raw_parts_mut(energies, k)) }
    };
    report(real.compute_induction_states_within(densities, k, operators, energies, threshold), stringify!(compute_induction_states_within_c))
}

// The electric field at sites, and the multipole convention.
//
// A selection s is one of the site_selection_* constants; p ((3, np), bohr) are the points of
// site_selection_custom, ignored (and may be null) otherwise. The chunking of the electronic field is the
// polarization options': `sites` > 0 a fixed number of sites per piece, else `budget` in `unit` (< 0: the default).

// The site selections of the electric-field calls, by their C codes.
c_int_enum! {
    pub enum SiteSelectionKind {
        Qm = 0 => "site_selection_qm",
        EmbeddingAimps = 1 => "site_selection_embedding_aimps",
        ExternalMultipoles = 2 => "site_selection_external_multipoles",
        Polarizable = 3 => "site_selection_polarizable",
        Custom = 4 => "site_selection_custom",
    }
    invalid_error = (|value: i32| AedelstenError::StringError(format!("unknown site selection {value}")));
}

// The multipole conventions, by their C codes.
c_int_enum! {
    pub enum MultipoleConventionKind {
        ContactFree = 0 => "multipole_convention_contact_free",
        Exact = 1 => "multipole_convention_exact",
    }
    invalid_error = (|value: i32| AedelstenError::StringError(format!("unknown multipole convention {value}")));
}

/// The custom sites of `selection` (none for the other selections).
fn custom_sites(selection: SiteSelectionKind, points: *const f64, n_points: i64) -> Result<Vec<crate::Site>, String> {
    match selection {
        SiteSelectionKind::Custom => {
            if points.is_null() || n_points < 0 {
                return Err("site_selection_custom needs its points".to_string());
            }
            let xyz = unsafe { std::slice::from_raw_parts(points, 3 * n_points as usize) };
            Ok(xyz.chunks_exact(3).map(|p| crate::Site::new([p[0], p[1], p[2]])).collect())
        }
        _ => Ok(Vec::new()),
    }
}

fn site_selection(selection: SiteSelectionKind, sites: &[crate::Site]) -> SiteSelection<'_> {
    match selection {
        SiteSelectionKind::Qm => SiteSelection::Qm,
        SiteSelectionKind::EmbeddingAimps => SiteSelection::EmbeddingAimps,
        SiteSelectionKind::ExternalMultipoles => SiteSelection::ExternalMultipoles,
        SiteSelectionKind::Polarizable => SiteSelection::Polarizable,
        SiteSelectionKind::Custom => SiteSelection::Custom(sites),
    }
}

/// The selection's kind and custom sites from the C arguments, or the error reported and FAILURE.
macro_rules! selection_from_c {
    ($selection:expr, $points:expr, $n_points:expr, $wrapper:expr, $fail:expr) => {{
        let kind = match SiteSelectionKind::from_c_int($selection) {
            Ok(kind) => kind,
            Err(e) => {
                rank0_eprintln!("Error in C wrapper {}: {}", $wrapper, e);
                return $fail;
            }
        };
        match custom_sites(kind, $points, $n_points) {
            Ok(sites) => (kind, sites),
            Err(e) => {
                rank0_eprintln!("Error in C wrapper {}: {}", $wrapper, e);
                return $fail;
            }
        }
    }};
}

/// The number of sites of `selection` (`points`, `n_points`: site_selection_custom's), or -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn count_electric_field_sites_c(ctx: *const AedelstenContextHandle, selection: i32, points: *const f64, n_points: i64) -> i64 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let (kind, sites) = selection_from_c!(selection, points, n_points, stringify!(count_electric_field_sites_c), -1);
    match real.count_electric_field_sites(site_selection(kind, &sites)) {
        Ok(n) => n as i64,
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {}", stringify!(count_electric_field_sites_c), e);
            -1
        }
    }
}

/// For each site of `selection`, the 0-based record of its input file, -1 when it came from none (nuclei, AIMPs,
/// custom sites); `n` the number of sites. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_input_indices_c(ctx: *const AedelstenContextHandle, selection: i32, points: *const f64, n_points: i64, indices: *mut i64, n: i64) -> i32 {
    let real = ctx_ref!(ctx);
    if indices.is_null() || n < 0 {
        return FAILURE;
    }
    let (kind, sites) = selection_from_c!(selection, points, n_points, stringify!(get_input_indices_c), FAILURE);
    let found = match real.get_input_indices(site_selection(kind, &sites)) {
        Ok(found) => found,
        Err(e) => return report(Err(e), stringify!(get_input_indices_c)),
    };
    if found.len() != n as usize {
        return report(Err(format!("{} sites, but room for {n}", found.len())), stringify!(get_input_indices_c));
    }
    let out = unsafe { std::slice::from_raw_parts_mut(indices, n as usize) };
    for (o, f) in out.iter_mut().zip(&found) {
        *o = f.map_or(-1, |i| i as i64);
    }
    0
}

/// The electric-field integrals of the sites `first_site .. first_site + n_sites` of `selection`, an
/// `n_cont x n_cont x 3 n_sites` array (`len` values), (x, y, z) planes per site. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_electric_field_integrals_c(
    ctx: *const AedelstenContextHandle,
    selection: i32,
    points: *const f64,
    n_points: i64,
    first_site: i64,
    n_sites: i64,
    output: *mut f64,
    len: i64,
) -> i32 {
    let real = ctx_ref!(ctx);
    if output.is_null() || first_site < 0 || n_sites < 0 || len < 0 {
        return FAILURE;
    }
    let (kind, sites) = selection_from_c!(selection, points, n_points, stringify!(compute_electric_field_integrals_c), FAILURE);
    let output = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
    let range = first_site as usize..(first_site + n_sites) as usize;
    report(real.compute_electric_field_integrals(site_selection(kind, &sites), range, output), stringify!(compute_electric_field_integrals_c))
}

/// The classical field at the sites of `selection` from the sources switched on (1: on, 0: off), overwriting
/// `field` (3 per site, `len` values). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_static_field_c(
    ctx: *const AedelstenContextHandle,
    selection: i32,
    points: *const f64,
    n_points: i64,
    nuclei: i32,
    core_aimps: i32,
    embedding_aimps: i32,
    external_multipoles: i32,
    field: *mut f64,
    len: i64,
) -> i32 {
    let real = ctx_ref!(ctx);
    if field.is_null() || len < 0 {
        return FAILURE;
    }
    let (kind, sites) = selection_from_c!(selection, points, n_points, stringify!(compute_static_field_c), FAILURE);
    let sources = crate::StaticFieldSources { nuclei: nuclei != 0, core_aimps: core_aimps != 0, embedding_aimps: embedding_aimps != 0, external_multipoles: external_multipoles != 0 };
    let field = unsafe { std::slice::from_raw_parts_mut(field, len as usize) };
    report(real.compute_static_field(site_selection(kind, &sites), sources, field), stringify!(compute_static_field_c))
}

/// The field of the plain density `density` (`n_cont x n_cont`) at the sites of `selection`, into `field` (3 per
/// site, `len` values); the chunking as the polarization options' (`chunk_sites` > 0 a fixed number, else
/// `chunk_budget` in `chunk_budget_unit`, < 0 the default). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_electronic_field_c(
    ctx: *const AedelstenContextHandle,
    density: *const f64,
    selection: i32,
    points: *const f64,
    n_points: i64,
    chunk_sites: i64,
    chunk_budget: i64,
    chunk_budget_unit: i32,
    field: *mut f64,
    len: i64,
) -> i32 {
    let real = ctx_ref!(ctx);
    if density.is_null() || field.is_null() || len < 0 {
        return FAILURE;
    }
    let n = real.get_n_cont();
    if buffers_overlap(&[(field as *const f64, len as usize)], &[(density, n * n)]) {
        rank0_eprintln!("Error in C wrapper {}: the field overlaps the density", stringify!(compute_electronic_field_c));
        return FAILURE;
    }
    let (kind, sites) = selection_from_c!(selection, points, n_points, stringify!(compute_electronic_field_c), FAILURE);
    let chunking = if chunk_sites > 0 {
        SiteChunking::Sites(chunk_sites as usize)
    } else if chunk_budget < 0 {
        SiteChunking::Budget(None)
    } else {
        match arrays::scratch::ByteUnit::from_c_int(chunk_budget_unit) {
            Ok(unit) => SiteChunking::Budget(Some((chunk_budget as usize, unit))),
            Err(e) => return report(Err(e), stringify!(compute_electronic_field_c)),
        }
    };
    let density = unsafe { std::slice::from_raw_parts(density, n * n) };
    let field = unsafe { std::slice::from_raw_parts_mut(field, len as usize) };
    report(real.compute_electronic_field(density, site_selection(kind, &sites), &chunking, field), stringify!(compute_electronic_field_c))
}

/// 1 when polarizable sites are set, 0 when not, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn has_polarizable_sites_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    unsafe { &*(ctx as *const AedelstenContext) }.has_polarizable_sites() as i32
}

/// 1 when the polarization is set up (`set_polarization_c`), 0 when not, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn has_polarization_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    unsafe { &*(ctx as *const AedelstenContext) }.has_polarization() as i32
}

/// Sets how the external multipoles interact with the electrons (a multipole_convention_* constant). Returns 0, or 1
/// on an unknown convention.
#[unsafe(no_mangle)]
pub extern "C" fn set_multipole_convention_c(ctx: *mut AedelstenContextHandle, convention: i32) -> i32 {
    let real = ctx_mut!(ctx);
    let convention = match MultipoleConventionKind::from_c_int(convention) {
        Ok(MultipoleConventionKind::ContactFree) => crate::MultipoleConvention::ContactFree,
        Ok(MultipoleConventionKind::Exact) => crate::MultipoleConvention::Exact,
        Err(e) => return report(Err(e), stringify!(set_multipole_convention_c)),
    };
    real.set_multipole_convention(convention);
    0
}

/// The multipole convention (a multipole_convention_* constant), -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn get_multipole_convention_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    match unsafe { &*(ctx as *const AedelstenContext) }.get_multipole_convention() {
        crate::MultipoleConvention::ContactFree => MultipoleConventionKind::ContactFree.get_c_int(),
        crate::MultipoleConvention::Exact => MultipoleConventionKind::Exact.get_c_int(),
    }
}

