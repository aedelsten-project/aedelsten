// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The direct Fock build's density-weighted screening and its pair list sorted by Schwarz bound, against J and K from
// the stored tensor (no screening): a density that decays as a converged one does (the overlap matrix), and a
// difference density (the same, 1e-5 times smaller, as between two late SCF iterations). The quartets computed and
// those skipped by the density are counted in the timing report.

mod common;

use aedelsten::arrays::array4d::Array4DView;
use aedelsten::arrays::matrix::{Matrix, MatrixTrait};
use aedelsten::{AedelstenContext, ContractionType, ERIAOSymmetry, ERIRangeOperator, OneElectronOperator};

// Two waters 6 bohr apart and a hydrogen 14 bohr out: pairs over a wide range of bounds. ANO-RCC generally contracted
// (bounds over shared primitives with coefficients of both signs); x2c-SVPall, a segmented set, for the segmented mode.
const ANO: &str = "Basis set\n    O.ANO-RCC...2s1p\n    O1 0.0 0.0 0.1\n    O2 6.0 0.3 0.0\nEnd of Basis\nBasis set\n    H.ANO-RCC...1s\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\n    H3 6.0 1.6 0.9\n    H4 6.8 -1.0 0.7\n    H5 14.0 0.0 0.0\nEnd of Basis\n";
const SVP: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\n    O2 6.0 0.3 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\n    H3 6.0 1.6 0.9\n    H4 6.8 -1.0 0.7\n    H5 14.0 0.0 0.0\nEnd of Basis\n";

const CASES: [(&str, &str, ContractionType); 2] = [("general", ANO, ContractionType::General), ("segmented", SVP, ContractionType::Segmented)];

fn context(name: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("density_screening_{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);
    ctx.set_shell_pairs(Some(1e-16)).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(None).unwrap();
    ctx
}

fn overlap(ctx: &AedelstenContext, scale: f64, salt: f64) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut s = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut s, OneElectronOperator::overlap(), false).unwrap();
    for i in 0..n {
        for j in 0..n {
            s[i + n * j] *= scale * (1.0 + 0.3 * (salt * (i + j) as f64).sin());
        }
    }
    s
}

// The stored tensor, S1.
fn tensor(ctx: &AedelstenContext) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut dense = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut dense, ERIAOSymmetry::S1, ERIRangeOperator::FullRange).unwrap();
    dense
}

// J and K of `d` from the stored tensor `dense`, `J = J[d]`, `K = -1/2 K[d]`.
fn exact(dense: &[f64], d: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = (d.len() as f64).sqrt().round() as usize;
    let g = Array4DView::new(n, n, n, n, dense);
    let p = Matrix::new_from_vec(n, n, d.to_vec().into_boxed_slice());
    let (mut j, mut k) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    aedelsten::fock_constructors::construct_j_and_k_stored(&g, &p, &mut j, &mut k, 1.0, -0.5);
    (j.get_data().to_vec(), k.get_data().to_vec())
}

// (quartets computed, quartets skipped by the density) on this thread so far.
fn counts() -> (u64, u64) {
    let report = aedelsten::timing_report();
    let calls = |label: &str| report.iter().find(|s| s.get_label() == label).map_or(0, |s| s.get_calls());
    (calls("eri::direct::quartets"), calls("eri::direct::density_skipped"))
}

fn largest_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn screened_builds_match_the_exact_ones_and_skip_more_for_a_difference_density() {
    for (name, input, contraction) in CASES {
        let ctx = context(name, input, contraction);
        let n = ctx.get_n_cont();
        let threshold = 1e-10;
        let dense = tensor(&ctx);
        let mut skipped_of = Vec::new();
        for (label, d) in [("density", overlap(&ctx, 1.0, 0.7)), ("difference density", overlap(&ctx, 1e-5, 1.3))] {
            let (j_exact, k_exact) = exact(&dense, &d);
            let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
            let before = counts();
            ctx.compute_fock_direct_restricted(&d, &mut j, &mut k, 1.0, -0.5, ERIRangeOperator::FullRange, Some(threshold)).unwrap();
            let after = counts();
            let (computed, skipped) = (after.0 - before.0, after.1 - before.1);
            assert!(computed > 0, "{name}, {label}: no quartet computed");
            // Every skipped quartet contributes below the threshold to each element it touches, a few thousand of them.
            let tolerance = 1e4 * threshold;
            let (dj, dk) = (largest_difference(&j, &j_exact), largest_difference(&k, &k_exact));
            assert!(dj < tolerance && dk < tolerance, "{name}, {label}: J off by {dj:e}, K by {dk:e} ({skipped} skipped)");
            eprintln!("{name}, {label}: {computed} quartets computed, {skipped} skipped by the density; J {dj:.1e}, K {dk:.1e}");
            skipped_of.push(skipped);
        }
        assert!(skipped_of[1] > skipped_of[0], "{name}: the difference density skips no more quartets ({skipped_of:?})");
    }
}

#[test]
fn unrestricted_screening_takes_the_larger_spin() {
    let ctx = context("unrestricted", ANO, ContractionType::General);
    let n = ctx.get_n_cont();
    // Alpha a full density, beta a small one: K_beta must still be built from the quartets alpha needs.
    let (alpha, beta) = (overlap(&ctx, 0.5, 0.7), overlap(&ctx, 1e-6, 1.1));
    let total: Vec<f64> = alpha.iter().zip(&beta).map(|(a, b)| a + b).collect();
    let (mut j, mut ka, mut kb) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
    ctx.compute_fock_direct_unrestricted(&alpha, &beta, &total, &mut j, &mut ka, &mut kb, 1.0, -1.0, ERIRangeOperator::FullRange, Some(1e-10))
        .unwrap();
    let dense = tensor(&ctx);
    let (j_exact, _) = exact(&dense, &total);
    let (_, ka_exact) = exact(&dense, &alpha);
    let (_, kb_exact) = exact(&dense, &beta);
    // `exact` gives -1/2 K; the unrestricted coefficient here is -1.
    let twice = |k: Vec<f64>| k.iter().map(|x| 2.0 * x).collect::<Vec<f64>>();
    assert!(largest_difference(&j, &j_exact) < 1e-6);
    assert!(largest_difference(&ka, &twice(ka_exact)) < 1e-6);
    // Relative to beta's own size: screened by the larger spin, its K is as accurate as alpha's in absolute terms.
    assert!(largest_difference(&kb, &twice(kb_exact)) < 1e-6);
}

#[test]
fn a_zero_density_skips_every_quartet() {
    let ctx = context("zero", ANO, ContractionType::General);
    let n = ctx.get_n_cont();
    let zero = vec![0.0; n * n];
    let (mut j, mut k) = (vec![1.0; n * n], vec![1.0; n * n]);
    let before = counts();
    ctx.compute_fock_direct_restricted(&zero, &mut j, &mut k, 1.0, -0.5, ERIRangeOperator::FullRange, Some(1e-12)).unwrap();
    let after = counts();
    assert_eq!(after.0, before.0, "a quartet was computed for a zero density");
    assert!(after.1 > before.1);
    assert!(j.iter().chain(&k).all(|&x| x == 0.0));
}

#[test]
fn the_sorted_walk_without_screening_is_the_exact_build() {
    // A threshold far below every bound: the sorted list and its early ends change only the summation order.
    for (name, input, contraction) in CASES {
        let ctx = context(&format!("sorted_{name}"), input, contraction);
        let n = ctx.get_n_cont();
        let d = overlap(&ctx, 1.0, 0.7);
        let (j_exact, k_exact) = exact(&tensor(&ctx), &d);
        for threshold in [None, Some(1e-30)] {
            let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
            ctx.compute_fock_direct_restricted(&d, &mut j, &mut k, 1.0, -0.5, ERIRangeOperator::FullRange, threshold).unwrap();
            let scale = j_exact.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            assert!(largest_difference(&j, &j_exact) < 1e-12 * scale, "{name}, {threshold:?}: J");
            assert!(largest_difference(&k, &k_exact) < 1e-12 * scale, "{name}, {threshold:?}: K");
        }
    }
}

// The active-space walk (`compute_fock_direct_active_space`) screened by the inactive density and the active
// coefficients together: active orbitals on the first water only, a decaying inactive density. A quartet is skipped
// only if neither the J/K nor T = C^T M (M the pair densities' Coulomb matrices) needs it, so J/K match the exact build
// and T the unscreened walk's, and some quartets are skipped. Without an active threshold the walk keeps every quartet
// M could need.
#[test]
fn the_active_space_walk_skips_what_neither_its_densities_nor_its_transformation_need() {
    for (name, input, contraction) in CASES {
        let ctx = context(&format!("active_{name}"), input, contraction);
        let n = ctx.get_n_cont();
        let threshold = 1e-10;
        let mut ids = vec![aedelsten::BasisFunctionId::default(); n];
        ctx.get_basis_function_ids(&mut ids).unwrap();
        let mut geometry = vec![aedelsten::CentreGeometry { atomic_number: 0, position: [0.0; 3], kind: aedelsten::AtomKind::Real }; ctx.get_n_centres()];
        ctx.get_centre_geometry(&mut geometry).unwrap();
        // Two active orbitals, zero on every function away from the first water (x below 3 bohr).
        let m = 2;
        let mut c = vec![0.0; n * m];
        for (mu, id) in ids.iter().enumerate() {
            if geometry[id.get_centre() - 1].position[0] < 3.0 {
                for u in 0..m {
                    c[mu + n * u] = 0.4 - 0.1 * (((mu * 3 + u * 5) % 7) as f64);
                }
            }
        }
        let d = overlap(&ctx, 1e-3, 0.9);
        let zero = vec![0.0; n * n];
        let q = m * (m + 1) / 2;
        let walk = |active: Option<f64>, pairwise: Option<f64>| {
            let (mut j, mut k, mut ja, mut ka) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
            let mut coulomb = vec![0.0; n * (n + 1) / 2 * q];
            let before = counts();
            ctx.compute_fock_direct_active_space(&d, &zero, &mut j, &mut k, &mut ja, &mut ka, 1.0, -0.5, 0.0, 0.0, &c, m, &mut coulomb, active, pairwise).unwrap();
            let after = counts();
            let mut t = vec![0.0; m * q * n];
            ctx.transform_pair_coulomb_one_index(&coulomb, q, &c, m, &mut t).unwrap();
            (j, k, t, after.1 - before.1)
        };
        let (_, _, t_exact, _) = walk(None, None);
        let (j_exact, k_exact) = exact(&tensor(&ctx), &d);

        let (j, k, t, skipped) = walk(Some(threshold), Some(threshold));
        assert!(skipped > 0, "{name}: nothing skipped");
        let tolerance = 1e4 * threshold;
        let (dj, dk, dt) = (largest_difference(&j, &j_exact), largest_difference(&k, &k_exact), largest_difference(&t, &t_exact));
        assert!(dj < tolerance && dk < tolerance && dt < tolerance, "{name}: J off by {dj:e}, K by {dk:e}, T by {dt:e} ({skipped} skipped)");
        eprintln!("{name}: {skipped} quartets skipped; J {dj:.1e}, K {dk:.1e}, T {dt:.1e}");

        // No active threshold: no density screening either, the Schwarz bounds alone.
        let (_, _, t, skipped) = walk(None, Some(threshold));
        assert_eq!(skipped, 0, "{name}: quartets skipped by the density without an active threshold");
        let scale = t_exact.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(scale > 1e-3 && largest_difference(&t, &t_exact) < tolerance, "{name}: T without an active threshold");
    }
}
