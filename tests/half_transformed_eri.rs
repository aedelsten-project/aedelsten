// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// `compute_eri_half_transformed`: what the context adds to the accumulator
// (checked against dense transformations in fock_constructors' half_transform test): the symmetric path for the same
// slice on both sides and the general one for two buffers, the rank-local walk, loose thresholds, the length checks,
// batches of sigma (and their checks) and the scratch query. X(p, nu, r, sigma) against the stored S1 tensor
// transformed by dense products.

mod common;

use aedelsten::arrays::matrix::{ColumnFactor, Factor, Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, ERIAOSymmetry, ERIRangeOperator};

const WATER_SVP: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";

fn context(name: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let dir = common::temp_path(&format!("half_transformed_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    ctx.set_shell_pairs(Some(1e-30)).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    // A Schwarz bound on every batch, so that the coefficient screen is live.
    ctx.set_eri_data(Some(1e-14)).unwrap();
    ctx
}

/// Deterministic `n x m` coefficients; `salt` makes two sets differ.
fn coefficients(n: usize, m: usize, salt: usize) -> Matrix<f64> {
    let mut c = Matrix::<f64>::new(n, m);
    for (u, column) in c.iter_mut().enumerate() {
        for (mu, c_mu) in column.iter_mut().enumerate() {
            *c_mu = 0.2 * (((mu * 5 + u * 3 + 1 + salt) % 7) as f64) - 0.55 + 0.03 * (mu as f64) * if (u + salt) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    c
}

/// `X(p, nu, r, sigma)` from the S1 tensor by two dense products: `H(p, nu lambda sigma) = L^T g(mu, nu lambda
/// sigma)`, then, for each sigma, `X(p nu, r, sigma) = H(p nu, lambda, sigma) R`.
fn dense_x(g: &[f64], n: usize, left: &Matrix<f64>, right: &Matrix<f64>) -> Vec<f64> {
    let (ml, mr) = (left.cols(), right.cols());
    let g = Matrix::new_from_vec(n, n * n * n, g.into());
    let h = left.view_transposed().matmul(&g);
    let mut x = vec![0.0; ml * n * mr * n];
    for (sigma, x_sigma) in x.chunks_exact_mut(ml * n * mr).enumerate() {
        let h_sigma = Matrix::new_from_vec(ml * n, n, h.get_data()[sigma * ml * n * n..(sigma + 1) * ml * n * n].into());
        x_sigma.copy_from_slice(h_sigma.matmul(right).get_data());
    }
    x
}

fn assert_close(label: &str, x: &[f64], y: &[f64], tolerance: f64) {
    assert_eq!(x.len(), y.len(), "{label}");
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-2, "{label}: vacuous comparison");
    assert!(difference <= tolerance * scale, "{label}: difference {difference} of {scale}");
}

fn stored(ctx: &AedelstenContext) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut g = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut g, ERIAOSymmetry::S1, ERIRangeOperator::FullRange).unwrap();
    g
}

#[test]
fn the_context_paths_match_the_stored_tensor() {
    let ctx = context("paths", WATER_SVP, ContractionType::Segmented);
    let n = ctx.get_n_cont();
    let g = stored(&ctx);
    for m in [1, 4] {
        // One basis: the same slice on both sides, the symmetric path; the same coefficients in two buffers, the
        // general path; the same X.
        let c = coefficients(n, m, 0);
        let reference = dense_x(&g, n, &c, &c);
        let mut x = vec![f64::NAN; m * n * m * n];
        ctx.compute_eri_half_transformed(c.get_data(), m, c.get_data(), m, &mut x, None, None, true).unwrap();
        assert_close(&format!("one basis, m {m}"), &x, &reference, 1e-12);
        let copy = c.clone();
        let mut x = vec![f64::NAN; m * n * m * n];
        ctx.compute_eri_half_transformed(c.get_data(), m, copy.get_data(), m, &mut x, None, None, true).unwrap();
        assert_close(&format!("one basis in two buffers, m {m}"), &x, &reference, 1e-12);
    }

    // Two bases of different widths, and the rank-local walk (here, on one rank, the very same X).
    let (left, right) = (coefficients(n, 2, 1), coefficients(n, 3, 4));
    let reference = dense_x(&g, n, &left, &right);
    let mut x = vec![f64::NAN; 2 * n * 3 * n];
    ctx.compute_eri_half_transformed(left.get_data(), 2, right.get_data(), 3, &mut x, Some(1e-14), None, true).unwrap();
    assert_close("two bases", &x, &reference, 1e-12);
    let mut local = vec![f64::NAN; 2 * n * 3 * n];
    ctx.compute_eri_half_transformed(left.get_data(), 2, right.get_data(), 3, &mut local, Some(1e-14), None, false).unwrap();
    assert_eq!(local, x, "the rank-local walk");
}

#[test]
fn a_loose_threshold_drops_only_what_is_below_it() {
    // What it drops is small against it: each skipped batch adds at most a few times the threshold to an element, and
    // few batches meet one element.
    let ctx = context("loose", WATER_SVP, ContractionType::Segmented);
    let n = ctx.get_n_cont();
    let c = coefficients(n, 3, 2);
    let reference = dense_x(&stored(&ctx), n, &c, &c);
    let mut x = vec![f64::NAN; 3 * n * 3 * n];
    for threshold in [1e-6, 1e-4, 1e-2] {
        ctx.compute_eri_half_transformed(c.get_data(), 3, c.get_data(), 3, &mut x, Some(threshold), None, true).unwrap();
        let difference = x.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("threshold {threshold:e}: difference {difference:.3e}");
        assert!(difference < 1e2 * threshold, "a threshold of {threshold:e} changed X by {difference:e}");
    }
}

#[test]
fn wrong_lengths_are_errors() {
    let ctx = context("lengths", WATER_SVP, ContractionType::Segmented);
    let n = ctx.get_n_cont();
    let c = vec![0.0; n * 2];
    let mut x = vec![0.0; 2 * n * 2 * n];
    assert!(ctx.compute_eri_half_transformed(&c, 2, &c[1..], 2, &mut x, None, None, true).is_err());
    assert!(ctx.compute_eri_half_transformed(&c, 2, &c, 2, &mut x[1..], None, None, true).is_err());
    let mut x = vec![0.0; 2 * n * n];
    assert!(ctx.compute_eri_half_transformed(&c, 2, &c, 2, &mut x, None, None, true).is_err());
    assert!(ctx.compute_eri_half_transformed(&c, 2, &c[..n], 1, &mut x, None, None, true).is_ok());
}

/// The first AO of every shell, ascending, then n.
fn shell_bounds(ctx: &AedelstenContext) -> Vec<usize> {
    let mut shells = vec![aedelsten::ShellInfo::default(); ctx.get_n_shells()];
    ctx.get_shells(&mut shells).unwrap();
    let mut bounds: Vec<usize> = shells.iter().map(|shell| shell.first_ao).chain([ctx.get_n_cont()]).collect();
    bounds.sort_unstable();
    bounds
}

#[test]
fn batches_of_sigma_are_the_columns_of_x() {
    for (name, contraction) in [("sigma_segmented", ContractionType::Segmented), ("sigma_general", ContractionType::General)] {
        let ctx = context(name, WATER_SVP, contraction);
        let n = ctx.get_n_cont();
        let g = stored(&ctx);
        let bounds = shell_bounds(&ctx);
        let (one, left, right) = (coefficients(n, 3, 0), coefficients(n, 2, 1), coefficients(n, 3, 4));
        // One basis (the same slice twice) and two.
        for (label, l, r, same) in [("one basis", &one, &one, true), ("two bases", &left, &right, false)] {
            let reference = dense_x(&g, n, l, r);
            let column = l.cols() * n * r.cols();
            for batches in [bounds.clone(), vec![0, bounds[bounds.len() / 2], n]] {
                for sigma in batches.windows(2).map(|w| w[0]..w[1]) {
                    let mut x = vec![f64::NAN; column * sigma.len()];
                    let r_data = if same { l.get_data() } else { r.get_data() };
                    ctx.compute_eri_half_transformed_sigma(l.get_data(), l.cols(), r_data, r.cols(), sigma.start, sigma.end, &mut x, Some(1e-14), None, true).unwrap();
                    assert_close(&format!("{name}, {label}, sigma {sigma:?}"), &x, &reference[column * sigma.start..column * sigma.end], 1e-12);
                    // On a node of one rank, the same.
                    let mut on_node = vec![f64::NAN; column * sigma.len()];
                    ctx.compute_eri_half_transformed_sigma_on_node(l.get_data(), l.cols(), r_data, r.cols(), sigma.start, sigma.end, &mut on_node, Some(1e-14), None).unwrap();
                    assert_eq!(on_node, x, "{name}, {label}, sigma {sigma:?} on a node");
                }
            }
        }
    }
}

#[test]
fn a_sigma_that_is_not_whole_shells_is_an_error() {
    let ctx = context("sigma_errors", WATER_SVP, ContractionType::Segmented);
    let n = ctx.get_n_cont();
    let bounds = shell_bounds(&ctx);
    // A shell of more than one AO (O 2p), cut after its first.
    let p_shell = bounds.windows(2).find(|w| w[1] - w[0] > 1).expect("a shell of several AOs")[0];
    let c = coefficients(n, 2, 0);
    let call = |start: usize, end: usize| {
        let mut x = vec![0.0; 2 * n * 2 * end.saturating_sub(start)];
        ctx.compute_eri_half_transformed_sigma(c.get_data(), 2, c.get_data(), 2, start, end, &mut x, None, None, true)
    };
    assert!(call(p_shell + 1, n).is_err(), "a start inside a shell");
    assert!(call(0, p_shell + 1).is_err(), "an end inside a shell");
    assert!(call(p_shell, p_shell).is_err(), "an empty batch");
    assert!(call(0, n + 1).is_err(), "beyond the basis");
    assert!(call(p_shell, n).is_ok());
    let mut short = vec![0.0; 2 * n * 2 * (n - p_shell) - 1];
    assert!(ctx.compute_eri_half_transformed_sigma(c.get_data(), 2, c.get_data(), 2, p_shell, n, &mut short, None, None, true).is_err(), "x too short");
}

#[test]
fn the_scratch_query_is_what_the_half_transform_takes() {
    // The scratch's peak over a context's life, after one call at one width of the orbitals: the walk's own scratch
    // and the half transform's, wide enough to be above set_eri_data's. Two widths differ by what the query says.
    let peak = |name: &str, run: &dyn Fn(&AedelstenContext)| {
        let ctx = context(name, WATER_SVP, ContractionType::General);
        run(&ctx);
        ctx.get_scratch_peak_bytes().unwrap()
    };
    let n = context("query_n", WATER_SVP, ContractionType::General).get_n_cont();
    let whole_one = |m: usize| {
        move |ctx: &AedelstenContext| {
            let c = coefficients(n, m, 0);
            ctx.compute_eri_half_transformed(c.get_data(), m, c.get_data(), m, &mut vec![0.0; m * n * m * n], None, None, true).unwrap();
        }
    };
    let ctx = context("query_whole", WATER_SVP, ContractionType::General);
    let query = |m_left, m_right, one_basis, n_sigma| ctx.half_transform_scratch_bytes(m_left, m_right, one_basis, n_sigma, 1);
    assert_eq!(peak("query_whole_3", &whole_one(3)) - peak("query_whole_1", &whole_one(1)), query(3, 3, true, n) - query(1, 1, true, n), "whole X, one basis");
    let start = shell_bounds(&ctx)[2];
    let sigma = |m_left: usize, m_right: usize| {
        move |ctx: &AedelstenContext| {
            let (l, r) = (coefficients(n, m_left, 1), coefficients(n, m_right, 4));
            ctx.compute_eri_half_transformed_sigma(l.get_data(), m_left, r.get_data(), m_right, start, n, &mut vec![0.0; m_left * n * m_right * (n - start)], None, None, true).unwrap();
        }
    };
    let n_sigma = n - start;
    assert_eq!(peak("query_sigma_l44", &sigma(44, 40)) - peak("query_sigma_l40", &sigma(40, 40)), query(44, 40, false, n_sigma) - query(40, 40, false, n_sigma), "a batch of sigma, two bases, the left");
    assert_eq!(peak("query_sigma_r44", &sigma(40, 44)) - peak("query_sigma_r40", &sigma(40, 40)), query(40, 44, false, n_sigma) - query(40, 40, false, n_sigma), "a batch of sigma, two bases, the right");
    let sigma_one = |m: usize| {
        move |ctx: &AedelstenContext| {
            let c = coefficients(n, m, 0);
            ctx.compute_eri_half_transformed_sigma(c.get_data(), m, c.get_data(), m, start, n, &mut vec![0.0; m * n * m * n_sigma], None, None, true).unwrap();
        }
    };
    assert_eq!(peak("query_sigma_one_44", &sigma_one(44)) - peak("query_sigma_one_40", &sigma_one(40)), query(44, 44, true, n_sigma) - query(40, 40, true, n_sigma), "a batch of sigma, one basis");
}
