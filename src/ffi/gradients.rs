// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Geometric gradients: the two-electron gradient and the classical (nucleus-level) energies' gradients.

use super::*;

/// The number of centres a gradient matrix has columns for (real atoms and pseudocharge ghosts; orthogonality
/// ghosts, last in the centre list, are outside), or -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_gradient_centres_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.get_n_gradient_centres() as i32
}

/// The shared body of the three classical-energy gradients below.
fn nuclear_gradient_c(
    ctx: *mut AedelstenContextHandle,
    gradient: *mut f64,
    n_centres: i32,
    active: *const i32,
    label: &str,
    call: impl FnOnce(&AedelstenContext, &mut [f64], Option<&[bool]>) -> Result<(), crate::AedelstenError>,
) -> i32 {
    if ctx.is_null() || gradient.is_null() || n_centres < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let gradient_slice = unsafe { std::slice::from_raw_parts_mut(gradient, 3 * n_centres as usize) };
    let active_flags: Option<Vec<bool>> = if active.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(active, n_centres as usize) }.iter().map(|&flag| flag != 0).collect())
    };
    match call(real, gradient_slice, active_flags.as_deref()) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in {} (C wrapper): {}", label, e);
            FAILURE
        }
    }
}

/// The gradient of `compute_nuclear_repulsion_energy` (nuclei, core AIMPs): `gradient` is `3 * n_centres` doubles, a
/// column-major `(3, n_centres)` matrix, OVERWRITTEN; `n_centres` must be `get_n_gradient_centres_c`; `active` as in
/// `compute_one_electron_gradient_c` (NULL = all centres).
#[unsafe(no_mangle)]
pub extern "C" fn compute_nuclear_repulsion_gradient_c(ctx: *mut AedelstenContextHandle, gradient: *mut f64, n_centres: i32, active: *const i32) -> i32 {
    nuclear_gradient_c(ctx, gradient, n_centres, active, "compute_nuclear_repulsion_gradient", |c, g, a| c.compute_nuclear_repulsion_gradient(g, a))
}

/// The gradient of `compute_nuclear_aimp_embedding_energy` (the QM nuclei and core AIMPs in the fixed embedding AIMPs);
/// zero without embedding AIMPs. Arguments as `compute_nuclear_repulsion_gradient_c`.
#[unsafe(no_mangle)]
pub extern "C" fn compute_nuclear_aimp_embedding_gradient_c(ctx: *mut AedelstenContextHandle, gradient: *mut f64, n_centres: i32, active: *const i32) -> i32 {
    nuclear_gradient_c(ctx, gradient, n_centres, active, "compute_nuclear_aimp_embedding_gradient", |c, g, a| c.compute_nuclear_aimp_embedding_gradient(g, a))
}

/// The gradient of `compute_nuclear_multipole_interaction_energy` (the QM nuclei and core AIMPs in the fixed external
/// multipoles); fails without external multipoles. Arguments as `compute_nuclear_repulsion_gradient_c`.
#[unsafe(no_mangle)]
pub extern "C" fn compute_nuclear_multipole_interaction_gradient_c(ctx: *mut AedelstenContextHandle, gradient: *mut f64, n_centres: i32, active: *const i32) -> i32 {
    nuclear_gradient_c(ctx, gradient, n_centres, active, "compute_nuclear_multipole_interaction_gradient", |c, g, a| {
        c.compute_nuclear_multipole_interaction_gradient(g, a)
    })
}

/// The C arguments of a two-electron gradient decoded: the range operator, the pairwise threshold and the terms,
/// borrowing their densities from the caller's buffer in place; the failure reported.
#[allow(clippy::too_many_arguments)]
fn two_electron_gradient_arguments<'a>(
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
    wrapper: &str,
) -> Option<(crate::context::two_electron::ERIRangeOperator, Option<f64>, Vec<fock_constructors::TwoElectronTerm<'a>>)> {
    if n < 0 || n_densities < 0 || n_terms < 0 {
        return None;
    }
    if n_terms > 0 && (densities.is_null() || term_kind.is_null() || term_left.is_null() || term_right.is_null() || term_coeff.is_null()) {
        return None;
    }
    let (n, n_densities, n_terms) = (n as usize, n_densities as usize, n_terms as usize);
    let range_operator = if omega == 0.0 {
        crate::context::two_electron::ERIRangeOperator::FullRange
    } else {
        match crate::context::two_electron::ERIRangeOperator::from_c_int(range_operator, omega) {
            Ok(op) => op,
            Err(_) => {
                rank0_eprintln!("Unknown ERI range-separation operator {} in C wrapper {}", range_operator, wrapper);
                return None;
            }
        }
    };
    let pairwise_threshold = if pairwise_threshold < 0.0 { None } else { Some(pairwise_threshold) };
    let densities_slice: &'a [f64] = if n_terms == 0 { &[] } else { unsafe { std::slice::from_raw_parts(densities, n * n * n_densities) } };
    let (kinds, lefts, rights, coeffs): (&[i32], &[i32], &[i32], &[f64]) = if n_terms == 0 {
        (&[], &[], &[], &[])
    } else {
        unsafe {
            (
                std::slice::from_raw_parts(term_kind, n_terms),
                std::slice::from_raw_parts(term_left, n_terms),
                std::slice::from_raw_parts(term_right, n_terms),
                std::slice::from_raw_parts(term_coeff, n_terms),
            )
        }
    };
    let density = |index: i32| -> Option<&'a [f64]> {
        let index = usize::try_from(index).ok().filter(|&i| i < n_densities)?;
        Some(&densities_slice[index * n * n..(index + 1) * n * n])
    };
    let mut terms = Vec::with_capacity(n_terms);
    for t in 0..n_terms {
        let kind = match kinds[t] {
            0 => fock_constructors::TwoElectronTermKind::Coulomb,
            1 => fock_constructors::TwoElectronTermKind::Exchange,
            other => {
                rank0_eprintln!("Unknown two-electron term kind {} (term {}) in C wrapper {}", other, t, wrapper);
                return None;
            }
        };
        let (Some(left), Some(right)) = (density(lefts[t]), density(rights[t])) else {
            rank0_eprintln!("Density index out of range in term {} (C wrapper {})", t, wrapper);
            return None;
        };
        terms.push(fock_constructors::TwoElectronTerm { kind, left, right, coeff: coeffs[t] });
    }
    Some((range_operator, pairwise_threshold, terms))
}

/// The two-electron energy gradient `dE_2e/dX` for every gradient centre, the contracted twin of the direct J/K
/// builds (see `AedelstenContext::compute_two_electron_gradient`). `gradient` is `3 * n_centres` doubles, a
/// column-major `(3, n_centres)` matrix, OVERWRITTEN; `n_centres` must be `get_n_gradient_centres_c`.
///
/// The energy is `E_2e = 1/2 sum_t coeff[t] sum_abcd W_t (ab|cd)` over `n_terms` terms. `densities` holds
/// `n_densities` column-major `n x n` matrices one after the other (`n * n * n_densities` doubles); term `t` pairs
/// density `term_left[t]` with density `term_right[t]` (0-based indices into `densities`) as `term_kind[t]`:
/// 0 = Coulomb, `W = L[a,b] R[c,d]`; 1 = exchange, `W = L[a,c] R[b,d]` (L, R need not be symmetric). Restricted HF:
/// densities `[P]`, terms `(0, 0, 0, 1.0)` and `(1, 0, 0, -0.5)`; unrestricted: `[P, P_a, P_b]`, `(0, 0, 0, 1.0)`,
/// `(1, 1, 1, -1.0)`, `(1, 2, 2, -1.0)`.
///
/// `range_operator`/`omega`/`pairwise_threshold` as in `compute_fock_direct_restricted_c`; `active` as in
/// `compute_one_electron_gradient_c` (NULL = all centres). Requires `set_eri_data_c` first.
#[unsafe(no_mangle)]
pub extern "C" fn compute_two_electron_gradient_c(
    ctx: *mut AedelstenContextHandle,
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
) -> i32 {
    let wrapper = stringify!(compute_two_electron_gradient_c);
    if ctx.is_null() || gradient.is_null() || n_centres < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let Some((range_operator, pairwise_threshold, terms)) = two_electron_gradient_arguments(
        densities, n, n_densities, n_terms, term_kind, term_left, term_right, term_coeff, range_operator, omega, pairwise_threshold, wrapper,
    ) else {
        return FAILURE;
    };
    let gradient_slice = unsafe { std::slice::from_raw_parts_mut(gradient, 3 * n_centres as usize) };
    let active_flags: Option<Vec<bool>> = if active.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(active, n_centres as usize) }.iter().map(|&flag| flag != 0).collect())
    };
    match real.compute_two_electron_gradient(gradient_slice, &terms, range_operator, pairwise_threshold, active_flags.as_deref()) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_two_electron_gradient (C wrapper {}): {}", wrapper, e);
            FAILURE
        }
    }
}

