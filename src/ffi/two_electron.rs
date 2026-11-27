// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Two-electron integrals: the ERI data, the stored tensor and the dense J/K builders over it, the direct Fock
// builders, the half transform.

use super::*;

/// The doubles of the S8 ERI tensor of this basis, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn compute_s8_size_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n = real.get_n_cont() as i32;
    eri::packed::compute_s8_size(n as usize) as i32
}

/// The doubles of the S4 ERI tensor of this basis, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn compute_s4_size_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n = real.get_n_cont() as i32;
    eri::packed::compute_s4_size(n as usize) as i32
}

/// Derives the ERI data of the direct walks (see `AedelstenContext::set_eri_data`), the pairs that cannot reach
/// `schwarz_threshold` dropped (negative: none). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_eri_data_c(ctx: *mut AedelstenContextHandle, schwarz_threshold: f64) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let schwarz_threshold = if schwarz_threshold < 0.0 { None } else { Some(schwarz_threshold) };
    match real.set_eri_data(schwarz_threshold) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_eri_data (C wrapper {}): {}", stringify!(set_eri_data_c), e);
            FAILURE
        }
    }
}

/// Frees the ERI data.
#[unsafe(no_mangle)]
pub extern "C" fn free_eri_data_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_eri_data();
}

/// Restricted direct J/K (see `AedelstenContext::compute_fock_direct_restricted`). The caller's buffers are used in
/// place, no copies: the outputs must not overlap the density or each other (such a call fails). Requires
/// set_eri_data_c first.
#[unsafe(no_mangle)]
pub extern "C" fn compute_fock_direct_restricted_c(
    ctx: *mut AedelstenContextHandle,
    j_out: *mut f64,
    k_out: *mut f64,
    density: *const f64,
    n: i32,
    j_coeff: f64,
    k_coeff: f64,
    range_operator: i32,
    omega: f64,
    pairwise_threshold: f64,
) -> i32 {
    if ctx.is_null() || j_out.is_null() || k_out.is_null() || density.is_null() || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let n = n as usize;
    let range_operator = if omega == 0.0 {
        crate::context::two_electron::ERIRangeOperator::FullRange
    } else {
        match crate::context::two_electron::ERIRangeOperator::from_c_int(range_operator, omega) {
            Ok(op) => op,
            Err(_) => {
                rank0_eprintln!("Unknown ERI range-separation operator {} in C wrapper {}", range_operator, stringify!(compute_fock_direct_restricted_c));
                return FAILURE;
            }
        }
    };
    let pairwise_threshold = if pairwise_threshold < 0.0 { None } else { Some(pairwise_threshold) };

    if outputs_overlap(&[j_out, k_out], &[density], n * n) {
        rank0_eprintln!("Overlapping density and output buffers in C wrapper {}", stringify!(compute_fock_direct_restricted_c));
        return FAILURE;
    }
    let density = unsafe { std::slice::from_raw_parts(density, n * n) };
    let (j_out, k_out) = unsafe { (std::slice::from_raw_parts_mut(j_out, n * n), std::slice::from_raw_parts_mut(k_out, n * n)) };
    match real.compute_fock_direct_restricted(density, j_out, k_out, j_coeff, k_coeff, range_operator, pairwise_threshold) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_fock_direct (restricted) (C wrapper {}): {}", stringify!(compute_fock_direct_restricted_c), e);
            FAILURE
        }
    }
}

/// Direct J/K of a density of any symmetry (see `AedelstenContext::compute_fock_direct_general`): J from its symmetric
/// part, `K_ac = sum_bd density_bd (ab|cd)`, not symmetrized. Buffers used in place, as in
/// `compute_fock_direct_restricted_c`: the outputs must not overlap the density or each other. Requires
/// set_eri_data_c first.
#[unsafe(no_mangle)]
pub extern "C" fn compute_fock_direct_general_c(
    ctx: *mut AedelstenContextHandle,
    j_out: *mut f64,
    k_out: *mut f64,
    density: *const f64,
    n: i32,
    j_coeff: f64,
    k_coeff: f64,
    range_operator: i32,
    omega: f64,
    pairwise_threshold: f64,
) -> i32 {
    if ctx.is_null() || j_out.is_null() || k_out.is_null() || density.is_null() || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let n = n as usize;
    let range_operator = if omega == 0.0 {
        crate::context::two_electron::ERIRangeOperator::FullRange
    } else {
        match crate::context::two_electron::ERIRangeOperator::from_c_int(range_operator, omega) {
            Ok(op) => op,
            Err(_) => {
                rank0_eprintln!("Unknown ERI range-separation operator {} in C wrapper {}", range_operator, stringify!(compute_fock_direct_general_c));
                return FAILURE;
            }
        }
    };
    let pairwise_threshold = if pairwise_threshold < 0.0 { None } else { Some(pairwise_threshold) };

    if outputs_overlap(&[j_out, k_out], &[density], n * n) {
        rank0_eprintln!("Overlapping density and output buffers in C wrapper {}", stringify!(compute_fock_direct_general_c));
        return FAILURE;
    }
    let density = unsafe { std::slice::from_raw_parts(density, n * n) };
    let (j_out, k_out) = unsafe { (std::slice::from_raw_parts_mut(j_out, n * n), std::slice::from_raw_parts_mut(k_out, n * n)) };
    match real.compute_fock_direct_general(density, j_out, k_out, j_coeff, k_coeff, range_operator, pairwise_threshold) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_fock_direct (general density) (C wrapper {}): {}", stringify!(compute_fock_direct_general_c), e);
            FAILURE
        }
    }
}

/// Direct J/K of `n_mat` densities of any symmetry in one walk (see
/// `AedelstenContext::compute_fock_direct_general_multi`): `density`, `j_out` and `k_out` hold `n_mat` `n x n`
/// matrices back to back, each density's J and K as in `compute_fock_direct_general_c`. The outputs must not overlap
/// the densities or each other. Requires set_eri_data_c first.
#[unsafe(no_mangle)]
pub extern "C" fn compute_fock_direct_general_multi_c(
    ctx: *mut AedelstenContextHandle,
    j_out: *mut f64,
    k_out: *mut f64,
    density: *const f64,
    n: i32,
    n_mat: i32,
    j_coeff: f64,
    k_coeff: f64,
    range_operator: i32,
    omega: f64,
    pairwise_threshold: f64,
) -> i32 {
    if ctx.is_null() || j_out.is_null() || k_out.is_null() || density.is_null() || n < 0 || n_mat < 1 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let (n, n_mat) = (n as usize, n_mat as usize);
    let range_operator = if omega == 0.0 {
        crate::context::two_electron::ERIRangeOperator::FullRange
    } else {
        match crate::context::two_electron::ERIRangeOperator::from_c_int(range_operator, omega) {
            Ok(op) => op,
            Err(_) => {
                rank0_eprintln!("Unknown ERI range-separation operator {} in C wrapper {}", range_operator, stringify!(compute_fock_direct_general_multi_c));
                return FAILURE;
            }
        }
    };
    let pairwise_threshold = if pairwise_threshold < 0.0 { None } else { Some(pairwise_threshold) };

    let len = n * n * n_mat;
    if outputs_overlap(&[j_out, k_out], &[density], len) {
        rank0_eprintln!("Overlapping density and output buffers in C wrapper {}", stringify!(compute_fock_direct_general_multi_c));
        return FAILURE;
    }
    let density = unsafe { std::slice::from_raw_parts(density, len) };
    let (j_out, k_out) = unsafe { (std::slice::from_raw_parts_mut(j_out, len), std::slice::from_raw_parts_mut(k_out, len)) };
    match real.compute_fock_direct_general_multi(density, j_out, k_out, n_mat, j_coeff, k_coeff, range_operator, pairwise_threshold) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_fock_direct (general densities) (C wrapper {}): {}", stringify!(compute_fock_direct_general_multi_c), e);
            FAILURE
        }
    }
}

/// The integrals of a state interaction between runs with different orbitals in one direct walk (see
/// `AedelstenContext::compute_fock_direct_mixed_active_space`), full range: J/K of a density of any symmetry as in
/// `compute_fock_direct_general_c`, and `m_out`, the Coulomb matrices of the active pair densities `M(vw, mu nu)`,
/// `n (n + 1)/2 n_active^2` doubles packed over the function pairs `mu >= nu`, the pair `(v, w)` at `v + w n_active`
/// fastest, from the `n x n_active` coefficients `c_bra`, `c_ket`. A negative threshold is `None`. Buffers used in
/// place: the outputs must not overlap the inputs or each other (such a call fails). Requires set_eri_data_c first.
#[unsafe(no_mangle)]
pub extern "C" fn compute_fock_direct_mixed_active_space_c(
    ctx: *mut AedelstenContextHandle,
    j_out: *mut f64,
    k_out: *mut f64,
    m_out: *mut f64,
    density: *const f64,
    c_bra: *const f64,
    c_ket: *const f64,
    n: i32,
    n_active: i32,
    j_coeff: f64,
    k_coeff: f64,
    active_threshold: f64,
    pairwise_threshold: f64,
) -> i32 {
    if ctx.is_null() || j_out.is_null() || k_out.is_null() || m_out.is_null() || density.is_null() || c_bra.is_null() || c_ket.is_null() || n < 0 || n_active < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let (n, n_active) = (n as usize, n_active as usize);
    let (len_c, len_m) = (n * n_active, n * (n + 1) / 2 * n_active * n_active);
    let threshold = |x: f64| if x < 0.0 { None } else { Some(x) };

    if buffers_overlap(&[(j_out, n * n), (k_out, n * n), (m_out, len_m)], &[(density, n * n), (c_bra, len_c), (c_ket, len_c)]) {
        rank0_eprintln!("Overlapping input and output buffers in C wrapper {}", stringify!(compute_fock_direct_mixed_active_space_c));
        return FAILURE;
    }
    let (density, c_bra, c_ket) = unsafe { (std::slice::from_raw_parts(density, n * n), std::slice::from_raw_parts(c_bra, len_c), std::slice::from_raw_parts(c_ket, len_c)) };
    let (j_out, k_out) = unsafe { (std::slice::from_raw_parts_mut(j_out, n * n), std::slice::from_raw_parts_mut(k_out, n * n)) };
    let m_out = unsafe { std::slice::from_raw_parts_mut(m_out, len_m) };
    match real.compute_fock_direct_mixed_active_space(density, j_out, k_out, j_coeff, k_coeff, c_bra, c_ket, n_active, m_out, threshold(active_threshold), threshold(pairwise_threshold)) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_fock_direct_mixed_active_space (C wrapper {}): {}", stringify!(compute_fock_direct_mixed_active_space_c), e);
            FAILURE
        }
    }
}

/// The half-transformed ERIs from one direct walk (see `AedelstenContext::compute_eri_half_transformed`), full range:
/// `x_out(p, nu, r, sigma) = sum_{mu lambda} c_left(mu, p) c_right(lambda, r) (mu nu|lambda sigma)`, the column-major
/// `m_left x n x m_right x n` array, `p` fastest, OVERWRITTEN; `c_left` is `n x m_left` and `c_right` `n x m_right`,
/// column-major. The same pointer for both (and `m_left == m_right`) takes the symmetric path, at half the work; the
/// two may also overlap otherwise (a column block of the other). A negative threshold is `None`.
/// `partition_over_ranks`: nonzero, under MPI the call is collective and every rank gets the whole X; zero, this rank
/// computes X alone. `x_out` must not overlap the coefficients (such a call fails). Requires set_eri_data_c first.
#[unsafe(no_mangle)]
pub extern "C" fn compute_eri_half_transformed_c(
    ctx: *mut AedelstenContextHandle,
    x_out: *mut f64,
    c_left: *const f64,
    m_left: i32,
    c_right: *const f64,
    m_right: i32,
    n: i32,
    threshold: f64,
    pairwise_threshold: f64,
    partition_over_ranks: i32,
) -> i32 {
    if ctx.is_null() || x_out.is_null() || c_left.is_null() || c_right.is_null() || m_left < 0 || m_right < 0 || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let (m_left, m_right, n) = (m_left as usize, m_right as usize, n as usize);
    let (len_left, len_right, len_x) = (n * m_left, n * m_right, m_left * n * m_right * n);
    let wire = |x: f64| if x < 0.0 { None } else { Some(x) };
    if buffers_overlap(&[(x_out, len_x)], &[(c_left, len_left), (c_right, len_right)]) {
        rank0_eprintln!("Overlapping input and output buffers in C wrapper {}", stringify!(compute_eri_half_transformed_c));
        return FAILURE;
    }
    let (c_left, c_right) = unsafe { (std::slice::from_raw_parts(c_left, len_left), std::slice::from_raw_parts(c_right, len_right)) };
    let x = unsafe { std::slice::from_raw_parts_mut(x_out, len_x) };
    match real.compute_eri_half_transformed(c_left, m_left, c_right, m_right, x, wire(threshold), wire(pairwise_threshold), partition_over_ranks != 0) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_eri_half_transformed (C wrapper {}): {}", stringify!(compute_eri_half_transformed_c), e);
            FAILURE
        }
    }
}

/// The half-transformed ERIs for a batch of sigma (see `AedelstenContext::compute_eri_half_transformed_sigma`):
/// `x_out(p, nu, r, sigma)` for the sigma in `sigma_start..sigma_end` only (0-based AOs, a run of whole shells, not
/// empty), the column-major `m_left x n x m_right x n_sigma` array, `p` fastest, its last index `sigma - sigma_start`,
/// OVERWRITTEN. The other arguments as `compute_eri_half_transformed_c`.
#[unsafe(no_mangle)]
pub extern "C" fn compute_eri_half_transformed_sigma_c(
    ctx: *mut AedelstenContextHandle,
    x_out: *mut f64,
    c_left: *const f64,
    m_left: i32,
    c_right: *const f64,
    m_right: i32,
    n: i32,
    sigma_start: i32,
    sigma_end: i32,
    threshold: f64,
    pairwise_threshold: f64,
    partition_over_ranks: i32,
) -> i32 {
    half_transformed_sigma_c(ctx, x_out, c_left, m_left, c_right, m_right, n, sigma_start, sigma_end, threshold, pairwise_threshold, Some(partition_over_ranks != 0))
}

/// `compute_eri_half_transformed_sigma_c` by the ranks of the context's communicator together into one X in their
/// node's shared memory (see `AedelstenContext::compute_eri_half_transformed_sigma_on_node`), as
/// `compute_eri_half_transformed_on_node_c` is to `compute_eri_half_transformed_c`.
#[unsafe(no_mangle)]
pub extern "C" fn compute_eri_half_transformed_sigma_on_node_c(
    ctx: *mut AedelstenContextHandle,
    x_out: *mut f64,
    c_left: *const f64,
    m_left: i32,
    c_right: *const f64,
    m_right: i32,
    n: i32,
    sigma_start: i32,
    sigma_end: i32,
    threshold: f64,
    pairwise_threshold: f64,
) -> i32 {
    half_transformed_sigma_c(ctx, x_out, c_left, m_left, c_right, m_right, n, sigma_start, sigma_end, threshold, pairwise_threshold, None)
}

// The two C entry points of a batch of sigma: `partition_over_ranks` `None` on a node.
#[allow(clippy::too_many_arguments)]
fn half_transformed_sigma_c(
    ctx: *mut AedelstenContextHandle,
    x_out: *mut f64,
    c_left: *const f64,
    m_left: i32,
    c_right: *const f64,
    m_right: i32,
    n: i32,
    sigma_start: i32,
    sigma_end: i32,
    threshold: f64,
    pairwise_threshold: f64,
    partition_over_ranks: Option<bool>,
) -> i32 {
    let name = if partition_over_ranks.is_some() { "compute_eri_half_transformed_sigma_c" } else { "compute_eri_half_transformed_sigma_on_node_c" };
    if ctx.is_null() || x_out.is_null() || c_left.is_null() || c_right.is_null() || m_left < 0 || m_right < 0 || n < 0 || sigma_start < 0 || sigma_end <= sigma_start {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let (m_left, m_right, n, sigma_start, sigma_end) = (m_left as usize, m_right as usize, n as usize, sigma_start as usize, sigma_end as usize);
    let (len_left, len_right, len_x) = (n * m_left, n * m_right, m_left * n * m_right * (sigma_end - sigma_start));
    let wire = |x: f64| if x < 0.0 { None } else { Some(x) };
    if buffers_overlap(&[(x_out, len_x)], &[(c_left, len_left), (c_right, len_right)]) {
        rank0_eprintln!("Overlapping input and output buffers in C wrapper {name}");
        return FAILURE;
    }
    let (c_left, c_right) = unsafe { (std::slice::from_raw_parts(c_left, len_left), std::slice::from_raw_parts(c_right, len_right)) };
    let x = unsafe { std::slice::from_raw_parts_mut(x_out, len_x) };
    let result = match partition_over_ranks {
        Some(partition) => real.compute_eri_half_transformed_sigma(c_left, m_left, c_right, m_right, sigma_start, sigma_end, x, wire(threshold), wire(pairwise_threshold), partition),
        None => real.compute_eri_half_transformed_sigma_on_node(c_left, m_left, c_right, m_right, sigma_start, sigma_end, x, wire(threshold), wire(pairwise_threshold)),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in the half transform's batch of sigma (C wrapper {name}): {e}");
            FAILURE
        }
    }
}

/// The scratch a half transform takes besides the walk's own, in bytes (see
/// `AedelstenContext::half_transform_scratch_bytes`): `one_basis` nonzero for the same coefficients on both sides,
/// `n_sigma` the batch's AOs (n for the whole X), `node_ranks` the ranks sharing X (1 without). -1 on bad arguments.
#[unsafe(no_mangle)]
pub extern "C" fn half_transform_scratch_bytes_c(ctx: *const AedelstenContextHandle, m_left: i32, m_right: i32, one_basis: i32, n_sigma: i32, node_ranks: i32) -> i64 {
    if ctx.is_null() || m_left < 0 || m_right < 0 || n_sigma < 0 || node_ranks < 1 {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.half_transform_scratch_bytes(m_left as usize, m_right as usize, one_basis != 0, n_sigma as usize, node_ranks as usize) as i64
}

/// `compute_eri_half_transformed_c` by the ranks of the context's communicator together into one X in their node's
/// shared memory (see `AedelstenContext::compute_eri_half_transformed_on_node`): `x_out` must be the same memory on
/// every rank (an `aedelsten_mpi_shared_array_c`), and the ranks on one node. Collective; every rank has the whole X
/// when it returns. Arguments as `compute_eri_half_transformed_c`.
#[unsafe(no_mangle)]
pub extern "C" fn compute_eri_half_transformed_on_node_c(
    ctx: *mut AedelstenContextHandle,
    x_out: *mut f64,
    c_left: *const f64,
    m_left: i32,
    c_right: *const f64,
    m_right: i32,
    n: i32,
    threshold: f64,
    pairwise_threshold: f64,
) -> i32 {
    if ctx.is_null() || x_out.is_null() || c_left.is_null() || c_right.is_null() || m_left < 0 || m_right < 0 || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let (m_left, m_right, n) = (m_left as usize, m_right as usize, n as usize);
    let (len_left, len_right, len_x) = (n * m_left, n * m_right, m_left * n * m_right * n);
    let wire = |x: f64| if x < 0.0 { None } else { Some(x) };
    if buffers_overlap(&[(x_out, len_x)], &[(c_left, len_left), (c_right, len_right)]) {
        rank0_eprintln!("Overlapping input and output buffers in C wrapper {}", stringify!(compute_eri_half_transformed_on_node_c));
        return FAILURE;
    }
    let (c_left, c_right) = unsafe { (std::slice::from_raw_parts(c_left, len_left), std::slice::from_raw_parts(c_right, len_right)) };
    let x = unsafe { std::slice::from_raw_parts_mut(x_out, len_x) };
    match real.compute_eri_half_transformed_on_node(c_left, m_left, c_right, m_right, x, wire(threshold), wire(pairwise_threshold)) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_eri_half_transformed_on_node (C wrapper {}): {}", stringify!(compute_eri_half_transformed_on_node_c), e);
            FAILURE
        }
    }
}

/// Unrestricted direct J/K (see `AedelstenContext::compute_fock_direct_unrestricted`). Buffers used in place, as in
/// `compute_fock_direct_restricted_c`: the outputs must not overlap any density or each other.
#[unsafe(no_mangle)]
pub extern "C" fn compute_fock_direct_unrestricted_c(
    ctx: *mut AedelstenContextHandle,
    j_out: *mut f64,
    k_alpha_out: *mut f64,
    k_beta_out: *mut f64,
    density_alpha: *const f64,
    density_beta: *const f64,
    density_total: *const f64,
    n: i32,
    j_coeff: f64,
    k_coeff: f64,
    range_operator: i32,
    omega: f64,
    pairwise_threshold: f64,
) -> i32 {
    if ctx.is_null() || j_out.is_null() || k_alpha_out.is_null() || k_beta_out.is_null()
        || density_alpha.is_null() || density_beta.is_null() || density_total.is_null() || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let n = n as usize;
    let range_operator = if omega == 0.0 {
        crate::context::two_electron::ERIRangeOperator::FullRange
    } else {
        match crate::context::two_electron::ERIRangeOperator::from_c_int(range_operator, omega) {
            Ok(op) => op,
            Err(_) => {
                rank0_eprintln!("Unknown ERI range-separation operator {} in C wrapper {}", range_operator, stringify!(compute_fock_direct_unrestricted_c));
                return FAILURE;
            }
        }
    };
    let pairwise_threshold = if pairwise_threshold < 0.0 { None } else { Some(pairwise_threshold) };

    if outputs_overlap(&[j_out, k_alpha_out, k_beta_out], &[density_alpha, density_beta, density_total], n * n) {
        rank0_eprintln!("Overlapping density and output buffers in C wrapper {}", stringify!(compute_fock_direct_unrestricted_c));
        return FAILURE;
    }
    let [d_alpha, d_beta, d_total] = [density_alpha, density_beta, density_total].map(|d| unsafe { std::slice::from_raw_parts(d, n * n) });
    let [j, k_alpha, k_beta] = [j_out, k_alpha_out, k_beta_out].map(|out| unsafe { std::slice::from_raw_parts_mut(out, n * n) });
    match real.compute_fock_direct_unrestricted(d_alpha, d_beta, d_total, j, k_alpha, k_beta, j_coeff, k_coeff, range_operator, pairwise_threshold) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_fock_direct (unrestricted) (C wrapper {}): {}", stringify!(compute_fock_direct_unrestricted_c), e);
            FAILURE
        }
    }
}

/// The active-space integrals of a CASSCF macro-iteration in one direct walk, full range (see
/// `AedelstenContext::compute_fock_direct_active_space`): the inactive and active J/K, each with its own coefficients,
/// and `m_out`, the Coulomb matrices of the active pair densities `M(vw, mu nu)`, `n (n + 1)/2 n_active (n_active +
/// 1)/2` doubles packed over the function pairs `mu >= nu` (the pair at `nu n - nu (nu - 1)/2 + mu - nu`), the pair
/// `v >= w` at `v (v + 1)/2 + w` fastest, from the `n x n_active` coefficients `c_active`. A negative threshold is
/// `None`. Buffers used in place: the outputs must not overlap the inputs or each other (the two densities may be
/// one buffer). Requires set_eri_data_c first.
#[unsafe(no_mangle)]
pub extern "C" fn compute_fock_direct_active_space_c(
    ctx: *mut AedelstenContextHandle,
    j_inactive_out: *mut f64,
    k_inactive_out: *mut f64,
    j_active_out: *mut f64,
    k_active_out: *mut f64,
    m_out: *mut f64,
    density_inactive: *const f64,
    density_active: *const f64,
    c_active: *const f64,
    n: i32,
    n_active: i32,
    coeff_j_inactive: f64,
    coeff_k_inactive: f64,
    coeff_j_active: f64,
    coeff_k_active: f64,
    active_threshold: f64,
    pairwise_threshold: f64,
) -> i32 {
    let jk_outputs = [j_inactive_out, k_inactive_out, j_active_out, k_active_out];
    if ctx.is_null() || jk_outputs.iter().any(|p| p.is_null()) || m_out.is_null()
        || density_inactive.is_null() || density_active.is_null() || c_active.is_null() || n < 0 || n_active < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let (n, n_active) = (n as usize, n_active as usize);
    let (len_c, len_m) = (n * n_active, n * (n + 1) / 2 * n_active * (n_active + 1) / 2);
    let threshold = |x: f64| if x < 0.0 { None } else { Some(x) };

    let outputs = [(jk_outputs[0] as *const f64, n * n), (jk_outputs[1], n * n), (jk_outputs[2], n * n), (jk_outputs[3], n * n), (m_out, len_m)];
    if buffers_overlap(&outputs, &[(density_inactive, n * n), (density_active, n * n), (c_active, len_c)]) {
        rank0_eprintln!("Overlapping input and output buffers in C wrapper {}", stringify!(compute_fock_direct_active_space_c));
        return FAILURE;
    }
    let (d_inactive, d_active) = unsafe { (std::slice::from_raw_parts(density_inactive, n * n), std::slice::from_raw_parts(density_active, n * n)) };
    let c_active = unsafe { std::slice::from_raw_parts(c_active, len_c) };
    let [j_inactive, k_inactive, j_active, k_active] = jk_outputs.map(|out| unsafe { std::slice::from_raw_parts_mut(out, n * n) });
    let m = unsafe { std::slice::from_raw_parts_mut(m_out, len_m) };
    match real.compute_fock_direct_active_space(
        d_inactive, d_active, j_inactive, k_inactive, j_active, k_active,
        coeff_j_inactive, coeff_k_inactive, coeff_j_active, coeff_k_active,
        c_active, n_active, m, threshold(active_threshold), threshold(pairwise_threshold),
    ) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_fock_direct_active_space (C wrapper {}): {}", stringify!(compute_fock_direct_active_space_c), e);
            FAILURE
        }
    }
}

/// One index of the active walks' `M` transformed (see `AedelstenContext::transform_pair_coulomb_one_index`):
/// `t_out(u, vw, mu) = sum_nu c(nu, u) M(vw, mu nu)`, `n_columns x n_pairs x n`, from `m`, `n (n + 1)/2 n_pairs`
/// doubles packed as `compute_fock_direct_active_space_c` writes it, and the `n x n_columns` coefficients `c`.
/// Rank-local. The output must not overlap the inputs (such a call fails).
#[unsafe(no_mangle)]
pub extern "C" fn transform_pair_coulomb_one_index_c(
    ctx: *mut AedelstenContextHandle,
    t_out: *mut f64,
    m: *const f64,
    c: *const f64,
    n: i32,
    n_pairs: i32,
    n_columns: i32,
) -> i32 {
    if ctx.is_null() || t_out.is_null() || m.is_null() || c.is_null() || n < 0 || n_pairs < 0 || n_columns < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let (n, n_pairs, n_columns) = (n as usize, n_pairs as usize, n_columns as usize);
    let (len_m, len_c, len_t) = (n * (n + 1) / 2 * n_pairs, n * n_columns, n_columns * n_pairs * n);
    if buffers_overlap(&[(t_out, len_t)], &[(m, len_m), (c, len_c)]) {
        rank0_eprintln!("Overlapping input and output buffers in C wrapper {}", stringify!(transform_pair_coulomb_one_index_c));
        return FAILURE;
    }
    let (m, c) = unsafe { (std::slice::from_raw_parts(m, len_m), std::slice::from_raw_parts(c, len_c)) };
    let t = unsafe { std::slice::from_raw_parts_mut(t_out, len_t) };
    match real.transform_pair_coulomb_one_index(m, n_pairs, c, n_columns, t) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in transform_pair_coulomb_one_index (C wrapper {}): {}", stringify!(transform_pair_coulomb_one_index_c), e);
            FAILURE
        }
    }
}

/// The stored ERI tensor, overwritten in full. Under MPI the call is collective: every rank must make it; each
/// computes its share of the shell-pair quartets and every rank gets the whole tensor.
#[unsafe(no_mangle)]
pub extern "C" fn compute_eri_stored_c(
    ctx: *mut AedelstenContextHandle,
    output: *mut f64,
    n: i32,
    ao_symmetry: i32,
    ao_index_order: i32,
    range_operator: i32,
    omega: f64
) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    // omega 0 is the full range, whatever range_operator says.
    let range_operator = if omega == 0.0 {
        crate::context::two_electron::ERIRangeOperator::FullRange
    } else {
        match crate::context::two_electron::ERIRangeOperator::from_c_int(range_operator, omega) {
            Ok(op) => op,
            Err(_) => {
                rank0_eprintln!("Unknown ERI range-separation operator {} in C wrapper {}", range_operator, stringify!(compute_eri_stored_c));
                return FAILURE;
            }
        }
    };

    let ao_symmetry = match crate::context::two_electron::ERIAOSymmetry::from_c_int(ao_symmetry, ao_index_order) {
        Ok(op) => op,
        Err(_) => {
            rank0_eprintln!("Unknown ERI AO symmetry {} (index order {}) in C wrapper {}", ao_symmetry, ao_index_order, stringify!(compute_eri_stored_c));
            return FAILURE;
        }
    };

    let expected_len = crate::context::two_electron::expected_eri_output_len(n as usize, &ao_symmetry);
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, expected_len) };
    match real.compute_eri_stored(output_slice, ao_symmetry, range_operator) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in compute_eri_stored (C wrapper {}): {}", stringify!(compute_eri_stored_c), e);
            FAILURE
        }
    }
}

/// `j_out = j_coeff J[density]` and `k_out = k_coeff K[density]` from the dense (S1) tensor `eri_dense`, in one pass;
/// every matrix `n x n`, column-major, the outputs overwritten. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_j_and_k_matrix_stored_c(
    j_out: *mut f64,
    k_out: *mut f64,
    density: *const f64,
    eri_dense: *const f64,
    n: i32,
    j_coeff: f64,
    k_coeff: f64,
) -> i32 {
    if j_out.is_null() || k_out.is_null() || density.is_null() || eri_dense.is_null() || n < 0 {
        return FAILURE;
    }
    let n = n as usize;
    let d = arrays::matrix::MatrixView::<f64>::new(n, n, unsafe { std::slice::from_raw_parts(density, n * n) });
    let g = arrays::array4d::Array4DView::<f64>::new(n, n, n, n, unsafe { std::slice::from_raw_parts(eri_dense, n * n * n * n) });
    let mut j = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(j_out, n * n) });
    let mut k = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(k_out, n * n) });
    fock_constructors::construct_j_and_k_stored(&g, &d, &mut j, &mut k, j_coeff, k_coeff);
    0
}

/// `j_out = j_coeff J[density_total]`, `k_alpha_out = k_coeff K[density_alpha]` and `k_beta_out = k_coeff
/// K[density_beta]` from the dense (S1) tensor `eri_dense`, in one pass; the outputs overwritten. Returns 0, or 1 on
/// failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_j_and_k_unrestricted_matrix_stored_c(
    j_out: *mut f64,
    k_alpha_out: *mut f64,
    k_beta_out: *mut f64,
    density_alpha: *const f64,
    density_beta: *const f64,
    density_total: *const f64,
    eri_dense: *const f64,
    n: i32,
    j_coeff: f64,
    k_coeff: f64,
) -> i32 {
    if j_out.is_null() || k_alpha_out.is_null() || k_beta_out.is_null()
        || density_alpha.is_null() || density_beta.is_null() || density_total.is_null()
        || eri_dense.is_null() || n < 0 {
        return FAILURE;
    }
    let n = n as usize;
    let d_alpha = arrays::matrix::MatrixView::<f64>::new(n, n, unsafe { std::slice::from_raw_parts(density_alpha, n * n) });
    let d_beta = arrays::matrix::MatrixView::<f64>::new(n, n, unsafe { std::slice::from_raw_parts(density_beta, n * n) });
    let d_total = arrays::matrix::MatrixView::<f64>::new(n, n, unsafe { std::slice::from_raw_parts(density_total, n * n) });
    let g = arrays::array4d::Array4DView::<f64>::new(n, n, n, n, unsafe { std::slice::from_raw_parts(eri_dense, n * n * n * n) });
    let mut j = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(j_out, n * n) });
    let mut k_alpha = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(k_alpha_out, n * n) });
    let mut k_beta = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(k_beta_out, n * n) });
    fock_constructors::construct_j_and_k_unrestricted_stored(
        &g, &d_alpha, &d_beta, &d_total, &mut j, &mut k_alpha, &mut k_beta, j_coeff, k_coeff,
    );
    0
}

/// `j_out = coeff J[density]` from the dense (S1) tensor `eri_dense`, overwritten. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_j_matrix_stored_c(
    j_out: *mut f64,
    density: *const f64,
    eri_dense: *const f64,
    n: i32,
    coeff: f64,
) -> i32 {
    if j_out.is_null() || density.is_null() || eri_dense.is_null() || n < 0 {
        return FAILURE;
    }
    let n = n as usize;
    let d = arrays::matrix::MatrixView::<f64>::new(n, n, unsafe { std::slice::from_raw_parts(density, n * n) });
    let g = arrays::array4d::Array4DView::<f64>::new(n, n, n, n, unsafe { std::slice::from_raw_parts(eri_dense, n * n * n * n) });
    let mut j = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(j_out, n * n) });
    fock_constructors::construct_j_stored(&g, &d, &mut j, coeff);
    0
}

/// `k_out = coeff K[density]`, `K[a,c] = sum_bd D[b,d] (ab|cd)`, from the dense (S1) tensor `eri_dense`, overwritten.
/// Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_k_matrix_stored_c(
    k_out: *mut f64,
    density: *const f64,
    eri_dense: *const f64,
    n: i32,
    coeff: f64,
) -> i32 {
    if k_out.is_null() || density.is_null() || eri_dense.is_null() || n < 0 {
        return FAILURE;
    }
    let n = n as usize;
    let d = arrays::matrix::MatrixView::<f64>::new(n, n, unsafe { std::slice::from_raw_parts(density, n * n) });
    let g = arrays::array4d::Array4DView::<f64>::new(n, n, n, n, unsafe { std::slice::from_raw_parts(eri_dense, n * n * n * n) });
    let mut k = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(k_out, n * n) });
    fock_constructors::construct_k_stored(&g, &d, &mut k, coeff);
    0
}

/// `k_alpha_out = coeff K[density_alpha]` and `k_beta_out = coeff K[density_beta]` from the dense (S1) tensor
/// `eri_dense`, in one pass; both overwritten. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_k_matrix_unrestricted_stored_c(
    k_alpha_out: *mut f64,
    k_beta_out: *mut f64,
    density_alpha: *const f64,
    density_beta: *const f64,
    eri_dense: *const f64,
    n: i32,
    coeff: f64,
) -> i32 {
    if k_alpha_out.is_null() || k_beta_out.is_null() || density_alpha.is_null() || density_beta.is_null() || eri_dense.is_null() || n < 0 {
        return FAILURE;
    }
    let n = n as usize;
    let eri_slice = unsafe { std::slice::from_raw_parts(eri_dense, n * n * n * n) };

    let d_alpha = arrays::matrix::MatrixView::<f64>::new(n, n, unsafe { std::slice::from_raw_parts(density_alpha, n * n) });
    let d_beta = arrays::matrix::MatrixView::<f64>::new(n, n, unsafe { std::slice::from_raw_parts(density_beta, n * n) });
    let g = arrays::array4d::Array4DView::<f64>::new(n, n, n, n, eri_slice);
    let mut k_alpha = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(k_alpha_out, n * n) });
    let mut k_beta = arrays::matrix::MatrixViewMut::<f64>::new(n, n, unsafe { std::slice::from_raw_parts_mut(k_beta_out, n * n) });
    fock_constructors::construct_k_unrestricted_stored(&g, &d_alpha, &d_beta, &mut k_alpha, &mut k_beta, coeff);

    0
}
