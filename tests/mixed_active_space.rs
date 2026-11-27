// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// `compute_fock_direct_mixed_active_space`: the fully active integrals in two orbital
// bases, `(tu|vw)` with t, v the bra's and u, w the ket's, finished from the direct walk's M (T from it by
// `transform_pair_coulomb_one_index`), against the stored S1 tensor transformed with the two coefficient sets; its J/K
// against the dense builders for a non-symmetric density; with one basis, M's pairs v >= w against
// `compute_fock_direct_active_space`; the active screen off and on.

mod common;

use aedelsten::arrays::array4d::Array4DView;
use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, ERIAOSymmetry, ERIRangeOperator};

const WATER_ANO: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";
const WATER_SVP: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";

fn context(name: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let dir = common::temp_path(&format!("mixed_active_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    ctx.set_shell_pairs(Some(1e-30)).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    // A Schwarz bound on every batch, so that the active screen is live.
    ctx.set_eri_data(Some(1e-14)).unwrap();
    ctx
}

fn cases() -> [(&'static str, AedelstenContext); 2] {
    [("water ANO-RCC, general", context("ano", WATER_ANO, ContractionType::General)), ("water x2c-SVPall, segmented", context("svp", WATER_SVP, ContractionType::Segmented))]
}

/// Deterministic `n x m` coefficients; `salt` makes the bra's differ from the ket's.
fn coefficients(n: usize, m: usize, salt: usize) -> Matrix<f64> {
    let mut c = Matrix::<f64>::new(n, m);
    for (u, column) in c.iter_mut().enumerate() {
        for (mu, c_mu) in column.iter_mut().enumerate() {
            *c_mu = 0.2 * (((mu * 5 + u * 3 + 1 + salt) % 7) as f64) - 0.55 + 0.03 * (mu as f64) * if (u + salt) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    c
}

fn nonsymmetric_density(n: usize) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for (j, column) in p.iter_mut().enumerate() {
        for (i, p_ij) in column.iter_mut().enumerate() {
            *p_ij = 0.03 * (((i * 7 + j * 3 + 2) % 11) as f64) - 0.11 + 0.007 * (i as f64) - 0.013 * (j as f64);
        }
    }
    p
}

/// `(tu|vw) = sum C_bra,mu t C_ket,nu u C_bra,lambda v C_ket,sigma w (mu nu|lambda sigma)` from the S1 tensor, at
/// `t + m (u + m (v + m w))`.
fn dense_active(g: &[f64], n: usize, bra: &Matrix<f64>, ket: &Matrix<f64>) -> Vec<f64> {
    let m = bra.cols();
    // One index at a time, the first index innermost: (mu nu lambda sigma) -> (nu lambda sigma t) -> ... -> (t u v w),
    // each step contracting the leading index and appending the new one last.
    let mut current = g.to_vec();
    let mut lead = n * n * n;
    for c in [bra, ket, bra, ket] {
        let mut next = vec![0.0; lead * m];
        for (p, out) in next.chunks_exact_mut(lead).enumerate() {
            for (rest, value) in out.iter_mut().enumerate() {
                *value = current[rest * n..(rest + 1) * n].iter().zip(c.get_col(p)).map(|(x, y)| x * y).sum();
            }
        }
        current = next;
        lead = lead / n * m;
    }
    current
}

/// `(tu|vw)` from `T(u, (v, w), mu)`, pairs at `v + w m`: `sum_mu C_bra,mu t T(u vw, mu)`, at `t + m (u + m (v + m
/// w))`.
fn finish(t: &[f64], n: usize, bra: &Matrix<f64>) -> Vec<f64> {
    let m = bra.cols();
    let mut out = vec![0.0; m * m * m * m];
    for (uvw, tuvw) in out.chunks_exact_mut(m).enumerate().flat_map(|(uvw, chunk)| chunk.iter_mut().enumerate().map(move |(t, x)| ((uvw, t), x))) {
        let (uvw, t_index) = uvw;
        *tuvw = (0..n).map(|mu| bra[(mu, t_index)] * t[uvw + m * m * m * mu]).sum();
    }
    out
}

/// The values of a packed pair tensor over `n` functions with `n_pairs` values a pair.
fn packed(n: usize, n_pairs: usize) -> usize {
    n * (n + 1) / 2 * n_pairs
}

/// T from the walk's M, `T(u, vw, mu) = sum_nu C_ket,nu u M(vw, mu nu)`, every pair `(v, w)`.
fn t_of(ctx: &AedelstenContext, coulomb: &[f64], ket: &Matrix<f64>) -> Vec<f64> {
    let (n, m) = (ket.rows(), ket.cols());
    let mut t = vec![f64::NAN; m * m * m * n];
    ctx.transform_pair_coulomb_one_index(coulomb, m * m, ket.get_data(), m, &mut t).unwrap();
    t
}

fn assert_close(label: &str, x: &[f64], y: &[f64], tolerance: f64) {
    assert_eq!(x.len(), y.len(), "{label}");
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-2, "{label}: vacuous comparison");
    assert!(difference <= tolerance * scale, "{label}: difference {difference} of {scale}");
}

#[test]
fn mixed_active_integrals_and_jk_match_the_stored_tensor() {
    for (name, ctx) in cases() {
        let n = ctx.get_n_cont();
        let mut g = vec![0.0; n * n * n * n];
        ctx.compute_eri_stored(&mut g, ERIAOSymmetry::S1, ERIRangeOperator::FullRange).unwrap();
        let d = nonsymmetric_density(n);
        let (mut j_ref, mut k_ref) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
        let tensor = Array4DView::new(n, n, n, n, &g);
        aedelsten::fock_constructors::construct_j_stored(&tensor, &d, &mut j_ref, 1.0);
        aedelsten::fock_constructors::construct_k_stored(&tensor, &d, &mut k_ref, -1.0);

        for m in [1, 3] {
            let (bra, ket) = (coefficients(n, m, 0), coefficients(n, m, 3));
            let reference = dense_active(&g, n, &bra, &ket);
            // The bases really differ: in one, (tu|vw) = (wv|ut) for real orbitals; here it must not hold.
            if m > 1 {
                let swapped = |t: usize, u: usize, v: usize, w: usize| reference[t + m * (u + m * (v + m * w))] - reference[w + m * (v + m * (u + m * t))];
                assert!((0..m).any(|t| (0..m).any(|u| swapped(t, u, 1, 0).abs() > 1e-6)), "{name}: the two bases give a v <-> w symmetric tensor");
            }
            for threshold in [None, Some(1e-14)] {
                let (mut j, mut k) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
                let mut coulomb = vec![f64::NAN; packed(n, m * m)];
                ctx.compute_fock_direct_mixed_active_space(d.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -1.0, bra.get_data(), ket.get_data(), m, &mut coulomb, threshold, None).unwrap();
                assert_close(&format!("{name}, n_act {m}, screen {threshold:?}: (tu|vw)"), &finish(&t_of(&ctx, &coulomb, &ket), n, &bra), &reference, 1e-12);
                assert_close(&format!("{name}, n_act {m}: J"), j.get_data(), j_ref.get_data(), 1e-12);
                assert_close(&format!("{name}, n_act {m}: K"), k.get_data(), k_ref.get_data(), 1e-12);
            }
        }
    }
}

#[test]
fn the_active_screen_holds_with_disjoint_bases() {
    // The bra's orbitals on the first half of the basis only, the ket's on the second: the pair densities live on
    // the cross pairs, so a screen that bounded P with one basis's coefficients alone would drop them.
    for (name, ctx) in cases() {
        let n = ctx.get_n_cont();
        let m = 2;
        let (mut bra, mut ket) = (coefficients(n, m, 1), coefficients(n, m, 4));
        for (b, k) in bra.iter_mut().zip(ket.iter_mut()) {
            b[n / 2..].fill(0.0);
            k[..n / 2].fill(0.0);
        }
        let mut g = vec![0.0; n * n * n * n];
        ctx.compute_eri_stored(&mut g, ERIAOSymmetry::S1, ERIRangeOperator::FullRange).unwrap();
        let reference = dense_active(&g, n, &bra, &ket);
        let d = nonsymmetric_density(n);
        let (mut j, mut k) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
        let mut coulomb = vec![0.0; packed(n, m * m)];
        ctx.compute_fock_direct_mixed_active_space(d.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -1.0, bra.get_data(), ket.get_data(), m, &mut coulomb, Some(1e-14), None).unwrap();
        assert_close(&format!("{name}: (tu|vw), disjoint bases, screened"), &finish(&t_of(&ctx, &coulomb, &ket), n, &bra), &reference, 1e-12);
    }
}

#[test]
fn one_basis_gives_the_active_space_transform() {
    for (name, ctx) in cases() {
        let n = ctx.get_n_cont();
        let m = 3;
        let q = m * (m + 1) / 2;
        let c = coefficients(n, m, 2);
        let d = nonsymmetric_density(n);
        let (mut j, mut k) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
        let mut m_mixed = vec![0.0; packed(n, m * m)];
        ctx.compute_fock_direct_mixed_active_space(d.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -1.0, c.get_data(), c.get_data(), m, &mut m_mixed, None, None).unwrap();

        let p = Matrix::<f64>::new(n, n);
        let mut scratch: [Matrix<f64>; 4] = std::array::from_fn(|_| Matrix::<f64>::new(n, n));
        let [j_i, k_i, j_a, k_a] = &mut scratch;
        let mut m_active = vec![0.0; packed(n, q)];
        ctx.compute_fock_direct_active_space(
            p.get_data(), p.get_data(), j_i.get_data_mut(), k_i.get_data_mut(), j_a.get_data_mut(), k_a.get_data_mut(),
            1.0, -0.5, 1.0, -0.5, c.get_data(), m, &mut m_active, None, None,
        )
        .unwrap();
        // The pairs v >= w of the square layout, in the triangular one.
        let mut m_triangle = vec![0.0; packed(n, q)];
        for (triangle, square) in m_triangle.chunks_exact_mut(q).zip(m_mixed.chunks_exact(m * m)) {
            for v in 0..m {
                for w in 0..=v {
                    triangle[v * (v + 1) / 2 + w] = square[v + m * w];
                }
            }
        }
        assert_close(&format!("{name}: M, one basis"), &m_triangle, &m_active, 1e-13);
    }
}

#[test]
fn wrong_lengths_are_errors() {
    let ctx = context("lengths", WATER_SVP, ContractionType::Segmented);
    let n = ctx.get_n_cont();
    let m = 2;
    let (d, c) = (vec![0.0; n * n], vec![0.0; n * m]);
    let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    // M sized for the triangular pairs of one basis, not every pair.
    let mut coulomb = vec![0.0; packed(n, m * (m + 1) / 2)];
    assert!(ctx.compute_fock_direct_mixed_active_space(&d, &mut j, &mut k, 1.0, -1.0, &c, &c, m, &mut coulomb, None, None).is_err());
    let mut coulomb = vec![0.0; packed(n, m * m)];
    assert!(ctx.compute_fock_direct_mixed_active_space(&d, &mut j, &mut k, 1.0, -1.0, &c, &c[1..], m, &mut coulomb, None, None).is_err());
    // T from M: M's length against its pairs, T's against its columns.
    let mut t = vec![0.0; m * m * m * n];
    assert!(ctx.transform_pair_coulomb_one_index(&coulomb[1..], m * m, &c, m, &mut t).is_err());
    assert!(ctx.transform_pair_coulomb_one_index(&coulomb, m * m, &c, m, &mut t[1..]).is_err());
}
