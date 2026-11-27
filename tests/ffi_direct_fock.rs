// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the direct J/K builds (compute_fock_direct_{restricted,unrestricted,general}_c, the
// active-space walks compute_fock_direct_{active_space,mixed_active_space}_c with T from their M,
// transform_pair_coulomb_one_index_c, the half transform compute_eri_half_transformed_c, its batches of sigma
// compute_eri_half_transformed_sigma(_on_node)_c and half_transform_scratch_bytes_c), driven the way a
// C/Fortran caller would (raw pointers, column-major n x n buffers) and compared to the Rust API they wrap. The
// caller's buffers are used in place, so a call whose output overlaps a density or another output is refused.

mod common;

use aedelsten::{AedelstenContext, ContractionType, ERIRangeOperator};
use std::ffi::{CString, c_char};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn set_shell_pairs_c(ctx: *mut std::ffi::c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut std::ffi::c_void, size: i32, unit: i32) -> i32;
    fn set_eri_data_c(ctx: *mut std::ffi::c_void, schwarz_threshold: f64) -> i32;
    fn compute_eri_half_transformed_c(
        ctx: *mut std::ffi::c_void,
        x_out: *mut f64,
        c_left: *const f64,
        m_left: i32,
        c_right: *const f64,
        m_right: i32,
        n: i32,
        threshold: f64,
        pairwise_threshold: f64,
        partition_over_ranks: i32,
    ) -> i32;
    fn compute_eri_half_transformed_sigma_c(
        ctx: *mut std::ffi::c_void,
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
    ) -> i32;
    fn compute_eri_half_transformed_sigma_on_node_c(
        ctx: *mut std::ffi::c_void,
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
    ) -> i32;
    fn half_transform_scratch_bytes_c(ctx: *const std::ffi::c_void, m_left: i32, m_right: i32, one_basis: i32, n_sigma: i32, node_ranks: i32) -> i64;
    fn compute_fock_direct_restricted_c(
        ctx: *mut std::ffi::c_void,
        j_out: *mut f64,
        k_out: *mut f64,
        density: *const f64,
        n: i32,
        j_coeff: f64,
        k_coeff: f64,
        range_operator: i32,
        omega: f64,
        pairwise_threshold: f64,
    ) -> i32;
    fn compute_fock_direct_general_c(
        ctx: *mut std::ffi::c_void,
        j_out: *mut f64,
        k_out: *mut f64,
        density: *const f64,
        n: i32,
        j_coeff: f64,
        k_coeff: f64,
        range_operator: i32,
        omega: f64,
        pairwise_threshold: f64,
    ) -> i32;
    fn compute_fock_direct_general_multi_c(
        ctx: *mut std::ffi::c_void,
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
    ) -> i32;
    fn compute_fock_direct_mixed_active_space_c(
        ctx: *mut std::ffi::c_void,
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
    ) -> i32;
    fn compute_fock_direct_active_space_c(
        ctx: *mut std::ffi::c_void,
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
    ) -> i32;
    fn transform_pair_coulomb_one_index_c(ctx: *mut std::ffi::c_void, t_out: *mut f64, m: *const f64, c: *const f64, n: i32, n_pairs: i32, n_columns: i32) -> i32;
    fn compute_fock_direct_unrestricted_c(
        ctx: *mut std::ffi::c_void,
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
    ) -> i32;
}

const INPUT: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 1.5 -0.9\n    H2 -0.2 -1.4 -1.1\nEnd of Basis\n";

fn symmetric(n: usize, salt: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[i + n * j] = 0.05 * (((i.min(j) * 7 + i.max(j) * 3 + salt) % 11) as f64) - 0.1;
        }
    }
    d
}

#[test]
fn the_c_direct_fock_builds_reproduce_the_rust_ones_in_place_and_refuse_overlaps() {
    let path = common::temp_path("ffi_direct_fock.inp");
    std::fs::write(&path, INPUT).unwrap();
    let inp = CString::new(path.to_str().unwrap()).unwrap();

    let mut reference = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    reference.set_shell_pairs(None).unwrap();
    reference.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    reference.set_eri_data(None).unwrap();
    let n = reference.get_n_cont();
    let (p_alpha, p_beta) = (symmetric(n, 1), symmetric(n, 4));
    let p_total: Vec<f64> = p_alpha.iter().zip(&p_beta).map(|(a, b)| a + b).collect();
    let zero = || vec![0.0; n * n];
    let (mut j_ref, mut k_ref) = (zero(), zero());
    reference.compute_fock_direct_restricted(&p_total, &mut j_ref, &mut k_ref, 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap();
    let (mut ju_ref, mut ka_ref, mut kb_ref) = (zero(), zero(), zero());
    reference
        .compute_fock_direct_unrestricted(&p_alpha, &p_beta, &p_total, &mut ju_ref, &mut ka_ref, &mut kb_ref, 1.0, -1.0, ERIRangeOperator::FullRange, None)
        .unwrap();
    assert!(k_ref.iter().any(|v| v.abs() > 1e-6));
    // A non-symmetric density for the general build: its K is not symmetric.
    let q: Vec<f64> = p_total.iter().enumerate().map(|(i, v)| v + 0.03 * (i % 5) as f64).collect();
    let (mut jg_ref, mut kg_ref) = (zero(), zero());
    reference.compute_fock_direct_general(&q, &mut jg_ref, &mut kg_ref, 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap();
    assert!((0..n).any(|i| (0..n).any(|j| (kg_ref[i + n * j] - kg_ref[j + n * i]).abs() > 1e-6)), "K of the general density came out symmetric");
    // Two general densities in one walk: q and q scaled and shifted.
    let stacked: Vec<f64> = q.iter().copied().chain(q.iter().enumerate().map(|(i, v)| -2.0 * v + 0.01 * (i % 3) as f64)).collect();
    let (mut jgm_ref, mut kgm_ref) = (vec![0.0; 2 * n * n], vec![0.0; 2 * n * n]);
    reference.compute_fock_direct_general_multi(&stacked, &mut jgm_ref, &mut kgm_ref, 2, 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap();
    // The mixed-basis active walk of the same density, two coefficient sets.
    let m = 2;
    let c_bra: Vec<f64> = (0..n * m).map(|i| 0.1 * (i % 5) as f64 - 0.2).collect();
    let c_ket: Vec<f64> = (0..n * m).map(|i| 0.1 * (i % 7) as f64 - 0.3).collect();
    let packed = |pairs: usize| n * (n + 1) / 2 * pairs;
    let (mut jm_ref, mut km_ref, mut mm_ref) = (zero(), zero(), vec![0.0; packed(m * m)]);
    reference.compute_fock_direct_mixed_active_space(&q, &mut jm_ref, &mut km_ref, 1.0, -0.5, &c_bra, &c_ket, m, &mut mm_ref, None, None).unwrap();
    assert!(mm_ref.iter().any(|v| v.abs() > 1e-6));
    // Active space (CASSCF): inactive and active J/K and M for the coefficients c_ket, and T from M.
    let q_pairs = m * (m + 1) / 2;
    let [mut j_in_ref, mut k_in_ref, mut j_act_ref, mut k_act_ref] = [zero(), zero(), zero(), zero()];
    let mut ma_ref = vec![0.0; packed(q_pairs)];
    reference
        .compute_fock_direct_active_space(&p_alpha, &p_beta, &mut j_in_ref, &mut k_in_ref, &mut j_act_ref, &mut k_act_ref, 1.0, -0.5, 1.0, -0.5, &c_ket, m, &mut ma_ref, None, None)
        .unwrap();
    assert!(ma_ref.iter().any(|v| v.abs() > 1e-6) && k_act_ref.iter().any(|v| v.abs() > 1e-6));
    let mut ta_ref = vec![0.0; m * q_pairs * n];
    reference.transform_pair_coulomb_one_index(&ma_ref, q_pairs, &c_ket, m, &mut ta_ref).unwrap();

    unsafe {
        let ctx = set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL);
        assert!(!ctx.is_null());
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 64, 1), 0);
        assert_eq!(set_eri_data_c(ctx, -1.0), 0);
        let n_c = n as i32;

        // Outputs pre-filled with garbage: written in place, in full.
        let (mut j, mut k) = (vec![f64::NAN; n * n], vec![f64::NAN; n * n]);
        assert_eq!(compute_fock_direct_restricted_c(ctx, j.as_mut_ptr(), k.as_mut_ptr(), p_total.as_ptr(), n_c, 1.0, -0.5, 0, 0.0, -1.0), 0);
        assert_eq!((&j, &k), (&j_ref, &k_ref));

        let (mut j, mut k_a, mut k_b) = (vec![f64::NAN; n * n], vec![f64::NAN; n * n], vec![f64::NAN; n * n]);
        let status = compute_fock_direct_unrestricted_c(
            ctx, j.as_mut_ptr(), k_a.as_mut_ptr(), k_b.as_mut_ptr(), p_alpha.as_ptr(), p_beta.as_ptr(), p_total.as_ptr(), n_c, 1.0, -1.0, 0, 0.0, -1.0,
        );
        assert_eq!(status, 0);
        assert_eq!((&j, &k_a, &k_b), (&ju_ref, &ka_ref, &kb_ref));

        let (mut j, mut k) = (vec![f64::NAN; n * n], vec![f64::NAN; n * n]);
        assert_eq!(compute_fock_direct_general_c(ctx, j.as_mut_ptr(), k.as_mut_ptr(), q.as_ptr(), n_c, 1.0, -0.5, 0, 0.0, -1.0), 0);
        assert_eq!((&j, &k), (&jg_ref, &kg_ref));
        let mut buffer = q.clone();
        let d = buffer.as_ptr();
        assert_ne!(compute_fock_direct_general_c(ctx, j.as_mut_ptr(), buffer.as_mut_ptr(), d, n_c, 1.0, -0.5, 0, 0.0, -1.0), 0, "K on the density");
        assert_ne!(compute_fock_direct_general_c(ctx, j.as_mut_ptr(), j.as_mut_ptr(), q.as_ptr(), n_c, 1.0, -0.5, 0, 0.0, -1.0), 0, "J and K on one buffer");
        assert_eq!(buffer, q, "a refused call leaves the density untouched");

        let (mut j, mut k) = (vec![f64::NAN; 2 * n * n], vec![f64::NAN; 2 * n * n]);
        assert_eq!(compute_fock_direct_general_multi_c(ctx, j.as_mut_ptr(), k.as_mut_ptr(), stacked.as_ptr(), n_c, 2, 1.0, -0.5, 0, 0.0, -1.0), 0);
        assert_eq!((&j, &k), (&jgm_ref, &kgm_ref));
        let mut buffer = stacked.clone();
        let d = buffer.as_ptr();
        assert_ne!(compute_fock_direct_general_multi_c(ctx, j.as_mut_ptr(), buffer.as_mut_ptr(), d, n_c, 2, 1.0, -0.5, 0, 0.0, -1.0), 0, "K on the densities");
        assert_ne!(compute_fock_direct_general_multi_c(ctx, j.as_mut_ptr(), j.as_mut_ptr(), stacked.as_ptr(), n_c, 2, 1.0, -0.5, 0, 0.0, -1.0), 0, "J and K on one buffer");
        assert_ne!(compute_fock_direct_general_multi_c(ctx, j.as_mut_ptr(), k.as_mut_ptr(), stacked.as_ptr(), n_c, 0, 1.0, -0.5, 0, 0.0, -1.0), 0, "no densities");
        assert_eq!(buffer, stacked, "a refused call leaves the densities untouched");

        let (mut j, mut k, mut mm) = (vec![f64::NAN; n * n], vec![f64::NAN; n * n], vec![f64::NAN; packed(m * m)]);
        let (m_c, cb, ck) = (m as i32, c_bra.as_ptr(), c_ket.as_ptr());
        assert_eq!(compute_fock_direct_mixed_active_space_c(ctx, j.as_mut_ptr(), k.as_mut_ptr(), mm.as_mut_ptr(), q.as_ptr(), cb, ck, n_c, m_c, 1.0, -0.5, -1.0, -1.0), 0);
        assert_eq!((&j, &k, &mm), (&jm_ref, &km_ref, &mm_ref));
        // M on the ket's coefficients; K straddling the end of M (the buffers differ in length).
        let mut buffer = c_ket.clone();
        buffer.resize(packed(m * m), 0.0);
        let c = buffer.as_ptr();
        assert_ne!(compute_fock_direct_mixed_active_space_c(ctx, j.as_mut_ptr(), k.as_mut_ptr(), buffer.as_mut_ptr(), q.as_ptr(), cb, c, n_c, m_c, 1.0, -0.5, -1.0, -1.0), 0, "M on the ket's coefficients");
        assert_eq!(&buffer[..n * m], &c_ket[..], "a refused call leaves the coefficients untouched");
        let mut joint = vec![0.0; packed(m * m) + n * n];
        let tail = joint.as_mut_ptr().add(packed(m * m) - 1);
        assert_ne!(compute_fock_direct_mixed_active_space_c(ctx, j.as_mut_ptr(), tail, joint.as_mut_ptr(), q.as_ptr(), cb, ck, n_c, m_c, 1.0, -0.5, -1.0, -1.0), 0, "K straddling M");


        let nan = || vec![f64::NAN; n * n];
        let (pa, pb) = (p_alpha.as_ptr(), p_beta.as_ptr());
        let [mut ji, mut ki, mut ja, mut ka] = [nan(), nan(), nan(), nan()];
        let mut ma = vec![f64::NAN; packed(q_pairs)];
        let status = compute_fock_direct_active_space_c(
            ctx, ji.as_mut_ptr(), ki.as_mut_ptr(), ja.as_mut_ptr(), ka.as_mut_ptr(), ma.as_mut_ptr(), pa, pb, ck, n_c, m_c, 1.0, -0.5, 1.0, -0.5, -1.0, -1.0,
        );
        assert_eq!(status, 0);
        assert_eq!((&ji, &ki, &ja, &ka, &ma), (&j_in_ref, &k_in_ref, &j_act_ref, &k_act_ref, &ma_ref));
        let mut ta = vec![f64::NAN; m * q_pairs * n];
        assert_eq!(transform_pair_coulomb_one_index_c(ctx, ta.as_mut_ptr(), ma.as_ptr(), ck, n_c, q_pairs as i32, m_c), 0);
        assert_eq!(ta, ta_ref);
        // T over M or over the coefficients is refused.
        let mut buffer = ma.clone();
        buffer.resize(packed(q_pairs).max(m * q_pairs * n), 0.0);
        let over_m = buffer.as_ptr();
        assert_ne!(transform_pair_coulomb_one_index_c(ctx, buffer.as_mut_ptr(), over_m, ck, n_c, q_pairs as i32, m_c), 0, "T on M");
        let mut buffer = c_ket.clone();
        buffer.resize(m * q_pairs * n, 0.0);
        let c = buffer.as_ptr();
        assert_ne!(transform_pair_coulomb_one_index_c(ctx, buffer.as_mut_ptr(), ma.as_ptr(), c, n_c, q_pairs as i32, m_c), 0, "T on the coefficients");
        // One density for both is fine (inputs may share a buffer); M over the coefficients is refused.
        assert_eq!(compute_fock_direct_active_space_c(ctx, ji.as_mut_ptr(), ki.as_mut_ptr(), ja.as_mut_ptr(), ka.as_mut_ptr(), ma.as_mut_ptr(), pa, pa, ck, n_c, m_c, 1.0, -0.5, 1.0, -0.5, -1.0, -1.0), 0);
        let mut buffer = c_ket.clone();
        buffer.resize(packed(q_pairs), 0.0);
        let c = buffer.as_ptr();
        assert_ne!(
            compute_fock_direct_active_space_c(ctx, ji.as_mut_ptr(), ki.as_mut_ptr(), ja.as_mut_ptr(), ka.as_mut_ptr(), buffer.as_mut_ptr(), pa, pb, c, n_c, m_c, 1.0, -0.5, 1.0, -0.5, -1.0, -1.0),
            0,
            "M on the coefficients"
        );
        let mut j = vec![0.0; n * n];

        // Overlaps: an output on the density, two outputs on one buffer, an output straddling the density.
        let mut buffer = p_total.clone();
        let mut other = zero();
        let d = buffer.as_ptr();
        assert_ne!(compute_fock_direct_restricted_c(ctx, buffer.as_mut_ptr(), other.as_mut_ptr(), d, n_c, 1.0, -0.5, 0, 0.0, -1.0), 0, "J on the density");
        assert_eq!(buffer, p_total, "a refused call leaves the density untouched");
        assert_ne!(compute_fock_direct_restricted_c(ctx, other.as_mut_ptr(), other.as_mut_ptr(), d, n_c, 1.0, -0.5, 0, 0.0, -1.0), 0, "J and K on one buffer");
        let mut wide = vec![0.0; 2 * n * n];
        wide[..n * n].copy_from_slice(&p_total);
        let straddling = wide.as_mut_ptr().add(n * n / 2);
        assert_ne!(compute_fock_direct_restricted_c(ctx, straddling, other.as_mut_ptr(), wide.as_ptr(), n_c, 1.0, -0.5, 0, 0.0, -1.0), 0, "J straddling the density");
        let status = compute_fock_direct_unrestricted_c(
            ctx, j.as_mut_ptr(), k_a.as_mut_ptr(), j.as_mut_ptr(), p_alpha.as_ptr(), p_beta.as_ptr(), p_total.as_ptr(), n_c, 1.0, -1.0, 0, 0.0, -1.0,
        );
        assert_ne!(status, 0, "J and K_beta on one buffer");
        // Densities may share a buffer (they are only read).
        let status = compute_fock_direct_unrestricted_c(
            ctx, j.as_mut_ptr(), k_a.as_mut_ptr(), k_b.as_mut_ptr(), p_total.as_ptr(), p_total.as_ptr(), p_total.as_ptr(), n_c, 1.0, -1.0, 0, 0.0, -1.0,
        );
        assert_eq!(status, 0, "shared densities");

        free_aedelsten_context_c(ctx);
    }
    std::fs::remove_file(&path).ok();
}

// The half transform through its C entry point, bitwise against the Rust call: one basis (the same pointer twice, the
// symmetric path), two bases, a column block of the right coefficients as the left ones (overlapping inputs, allowed),
// the screen and the rank-local flag; X on a coefficient buffer and a wrong n refused.
#[test]
fn the_c_half_transform_reproduces_the_rust_one_and_refuses_overlaps() {
    let path = common::temp_path("ffi_half_transform.inp");
    std::fs::write(&path, INPUT).unwrap();
    let inp = CString::new(path.to_str().unwrap()).unwrap();
    let mut reference = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    reference.set_shell_pairs(None).unwrap();
    reference.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    reference.set_eri_data(None).unwrap();
    let n = reference.get_n_cont();
    let (ml, mr) = (2, 3);
    let left: Vec<f64> = (0..n * ml).map(|i| 0.1 * (i % 5) as f64 - 0.2).collect();
    let right: Vec<f64> = (0..n * mr).map(|i| 0.1 * (i % 7) as f64 - 0.3).collect();
    // (c_left, m_left, c_right, m_right, threshold, partition_over_ranks); in the third case c_left is a column block
    // of c_right, a sub-slice of the same buffer.
    let cases: [(&[f64], usize, &[f64], usize, Option<f64>, bool); 5] = [
        (&right, mr, &right, mr, None, true),
        (&left, ml, &right, mr, None, true),
        (&right[n..3 * n], 2, &right, mr, None, true),
        (&left, ml, &right, mr, Some(1e-8), true),
        (&left, ml, &right, mr, None, false),
    ];
    let expected: Vec<Vec<f64>> = cases
        .iter()
        .map(|&(l, m_l, r, m_r, threshold, partition)| {
            let mut x = vec![0.0; m_l * n * m_r * n];
            reference.compute_eri_half_transformed(l, m_l, r, m_r, &mut x, threshold, None, partition).unwrap();
            x
        })
        .collect();
    assert!(expected[0].iter().any(|v| v.abs() > 1e-3));

    unsafe {
        let ctx = set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL);
        assert!(!ctx.is_null());
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 64, 1), 0);
        assert_eq!(set_eri_data_c(ctx, -1.0), 0);
        for (k, (&(l, m_l, r, m_r, threshold, partition), expected)) in cases.iter().zip(&expected).enumerate() {
            let mut x = vec![f64::NAN; m_l * n * m_r * n];
            let status = compute_eri_half_transformed_c(ctx, x.as_mut_ptr(), l.as_ptr(), m_l as i32, r.as_ptr(), m_r as i32, n as i32, threshold.unwrap_or(-1.0), -1.0, partition as i32);
            assert_eq!(status, 0, "case {k}");
            assert!(x.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()), "case {k}: X differs from the Rust call");
        }
        // X on top of the right coefficients, and a wrong n: refused.
        let mut wide = vec![0.0; n * mr + ml * n * mr * n];
        wide[..n * mr].copy_from_slice(&right);
        let status = compute_eri_half_transformed_c(ctx, wide.as_mut_ptr(), left.as_ptr(), ml as i32, wide.as_ptr(), mr as i32, n as i32, -1.0, -1.0, 1);
        assert_ne!(status, 0, "X on the coefficients");
        let mut x = vec![0.0; ml * n * mr * n];
        let status = compute_eri_half_transformed_c(ctx, x.as_mut_ptr(), left.as_ptr(), ml as i32, right.as_ptr(), mr as i32, n as i32 - 1, -1.0, -1.0, 1);
        assert_ne!(status, 0, "a wrong n");
        free_aedelsten_context_c(ctx);
    }
    std::fs::remove_file(&path).ok();
}

// A batch of sigma through its C entry points, bitwise against the Rust calls: one basis (the same pointer twice) and
// two, the batch of the shells after the first (on one rank the node form is the same walk); the scratch query as
// the Rust one; a batch that splits a shell, an empty one and X on the coefficients refused.
#[test]
fn the_c_sigma_batches_reproduce_the_rust_ones_and_refuse_bad_batches() {
    let path = common::temp_path("ffi_half_transform_sigma.inp");
    std::fs::write(&path, INPUT).unwrap();
    let inp = CString::new(path.to_str().unwrap()).unwrap();
    let mut reference = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    reference.set_shell_pairs(None).unwrap();
    reference.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    reference.set_eri_data(None).unwrap();
    let n = reference.get_n_cont();
    let mut shells = vec![aedelsten::ShellInfo::default(); reference.get_n_shells()];
    reference.get_shells(&mut shells).unwrap();
    let (start, end) = (shells[1].first_ao, n);
    let inside = shells.iter().find(|shell| shell.n_ao > 1).map(|shell| shell.first_ao + 1).expect("a shell of several AOs");
    let (ml, mr) = (2, 3);
    let left: Vec<f64> = (0..n * ml).map(|i| 0.1 * (i % 5) as f64 - 0.2).collect();
    let right: Vec<f64> = (0..n * mr).map(|i| 0.1 * (i % 7) as f64 - 0.3).collect();
    let cases: [(&[f64], usize, &[f64], usize); 2] = [(&right, mr, &right, mr), (&left, ml, &right, mr)];
    let expected: Vec<Vec<f64>> = cases
        .iter()
        .map(|&(l, m_l, r, m_r)| {
            let mut x = vec![0.0; m_l * n * m_r * (end - start)];
            reference.compute_eri_half_transformed_sigma(l, m_l, r, m_r, start, end, &mut x, None, None, true).unwrap();
            x
        })
        .collect();
    assert!(expected[0].iter().any(|v| v.abs() > 1e-3));

    unsafe {
        let ctx = set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL);
        assert!(!ctx.is_null());
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 64, 1), 0);
        assert_eq!(set_eri_data_c(ctx, -1.0), 0);
        let same_bits = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits());
        for (k, (&(l, m_l, r, m_r), expected)) in cases.iter().zip(&expected).enumerate() {
            let mut x = vec![f64::NAN; m_l * n * m_r * (end - start)];
            let status = compute_eri_half_transformed_sigma_c(ctx, x.as_mut_ptr(), l.as_ptr(), m_l as i32, r.as_ptr(), m_r as i32, n as i32, start as i32, end as i32, -1.0, -1.0, 1);
            assert_eq!(status, 0, "case {k}");
            assert!(same_bits(&x, expected), "case {k}: X differs from the Rust call");
            let mut x = vec![f64::NAN; m_l * n * m_r * (end - start)];
            let status = compute_eri_half_transformed_sigma_on_node_c(ctx, x.as_mut_ptr(), l.as_ptr(), m_l as i32, r.as_ptr(), m_r as i32, n as i32, start as i32, end as i32, -1.0, -1.0);
            assert_eq!(status, 0, "case {k} on the node");
            assert!(same_bits(&x, expected), "case {k} on the node: X differs from the Rust call");
        }
        for (one_basis, n_sigma, ranks) in [(true, n, 1), (false, end - start, 1), (false, end - start, 4)] {
            let bytes = half_transform_scratch_bytes_c(ctx, ml as i32, mr as i32, one_basis as i32, n_sigma as i32, ranks as i32);
            assert_eq!(bytes as usize, reference.half_transform_scratch_bytes(ml, mr, one_basis, n_sigma, ranks), "the scratch query");
        }
        assert_eq!(half_transform_scratch_bytes_c(ctx, ml as i32, mr as i32, 0, n as i32, 0), -1, "no ranks");
        // A batch that starts inside a shell, an empty one, and X on the right coefficients: refused.
        let mut x = vec![0.0; ml * n * mr * n];
        let status = compute_eri_half_transformed_sigma_c(ctx, x.as_mut_ptr(), left.as_ptr(), ml as i32, right.as_ptr(), mr as i32, n as i32, inside as i32, n as i32, -1.0, -1.0, 1);
        assert_ne!(status, 0, "a batch inside a shell");
        let status = compute_eri_half_transformed_sigma_c(ctx, x.as_mut_ptr(), left.as_ptr(), ml as i32, right.as_ptr(), mr as i32, n as i32, start as i32, start as i32, -1.0, -1.0, 1);
        assert_ne!(status, 0, "an empty batch");
        let mut wide = vec![0.0; n * mr + ml * n * mr * (end - start)];
        wide[..n * mr].copy_from_slice(&right);
        let status = compute_eri_half_transformed_sigma_c(ctx, wide.as_mut_ptr(), left.as_ptr(), ml as i32, wide.as_ptr(), mr as i32, n as i32, start as i32, end as i32, -1.0, -1.0, 1);
        assert_ne!(status, 0, "X on the coefficients");
        free_aedelsten_context_c(ctx);
    }
    std::fs::remove_file(&path).ok();
}
