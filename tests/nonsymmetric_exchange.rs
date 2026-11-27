// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Exchange for a density matrix that is NOT symmetric -- a spin-flip block (P_ab = C_a C_b^T), a transition
// density: the stored (S1) K builder against a brute-force
// K[a,c] = sum_{b,d} D[b,d] (ab|cd) taken from the S8 tensor's own indexing, on real integrals.

use aedelsten::arrays::array4d::Array4DView;
use aedelsten::packed::{compute_s8_size, PackedERITensorView};
use aedelsten::arrays::matrix::Matrix;
use aedelsten::fock_constructors::construct_k_stored;

fn setup() -> (aedelsten::AedelstenContext, usize) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let mut ctx = aedelsten::AedelstenContext::new(path, aedelsten::ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(Some(1e-30)).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();
    (ctx, n)
}

fn s8_and_dense(ctx: &aedelsten::AedelstenContext, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut s8 = vec![0.0; compute_s8_size(n)];
    ctx.compute_eri_stored(
        &mut s8,
        aedelsten::ERIAOSymmetry::S8 { order: aedelsten::ERIIndexOrder::ColumnMajor },
        aedelsten::ERIRangeOperator::FullRange,
    )
    .unwrap();
    let mut dense = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut dense, aedelsten::ERIAOSymmetry::S1, aedelsten::ERIRangeOperator::FullRange).unwrap();
    (s8, dense)
}

/// A deterministic density with no symmetry at all.
fn nonsymmetric_density(n: usize, salt: usize) -> Matrix<f64> {
    let mut d = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = 0.3 * (((i * 7 + j * 3 + salt) % 11) as f64) - 1.1 + 0.07 * (i as f64) - 0.13 * (j as f64);
        }
    }
    d
}

fn symmetric_density(n: usize, salt: usize) -> Matrix<f64> {
    let a = nonsymmetric_density(n, salt);
    let mut d = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = 0.5 * (a[(i, j)] + a[(j, i)]);
        }
    }
    d
}

fn brute_force_k(g: &PackedERITensorView<f64>, d: &Matrix<f64>, n: usize, coeff: f64) -> Matrix<f64> {
    let mut k = Matrix::<f64>::new(n, n);
    for a in 0..n {
        for c in 0..n {
            let mut sum = 0.0;
            for b in 0..n {
                for e in 0..n {
                    sum += d[(b, e)] * g[(a, b, c, e)];
                }
            }
            k[(a, c)] = coeff * sum;
        }
    }
    k
}

fn assert_close(label: &str, x: &Matrix<f64>, y: &Matrix<f64>, n: usize) {
    for i in 0..n {
        for j in 0..n {
            let (u, v) = (x[(i, j)], y[(i, j)]);
            assert!((u - v).abs() < 1e-9 * (1.0 + v.abs()), "{label} mismatch at ({i},{j}): {u} vs {v}");
        }
    }
}

#[test]
fn the_dense_builder_is_correct_for_a_nonsymmetric_density() {
    let (ctx, n) = setup();
    let (s8, dense) = s8_and_dense(&ctx, n);
    let g8 = PackedERITensorView::<f64>::new(n, &s8);
    let g1 = Array4DView::new(n, n, n, n, &dense);
    let d = nonsymmetric_density(n, 0);
    let mut k = Matrix::<f64>::new(n, n);
    construct_k_stored(&g1, &d, &mut k, -1.0);
    assert_close("dense K", &k, &brute_force_k(&g8, &d, n, -1.0), n);
}

#[test]
fn transposing_the_density_transposes_k() {
    let (ctx, n) = setup();
    let (_, dense) = s8_and_dense(&ctx, n);
    let g1 = Array4DView::new(n, n, n, n, &dense);
    let d = nonsymmetric_density(n, 1);
    let mut d_t = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            d_t[(i, j)] = d[(j, i)];
        }
    }
    let mut k = Matrix::<f64>::new(n, n);
    let mut k_of_transpose = Matrix::<f64>::new(n, n);
    construct_k_stored(&g1, &d, &mut k, 1.0);
    construct_k_stored(&g1, &d_t, &mut k_of_transpose, 1.0);
    let mut k_t = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            k_t[(i, j)] = k[(j, i)];
        }
    }
    assert_close("K[D]^T vs K[D^T]", &k_t, &k_of_transpose, n);
}

// Negative control: the tests above are only meaningful if the density's asymmetry really changes K: the K of its
// symmetric part differs from the true K.
#[test]
fn the_asymmetry_of_the_density_changes_k() {
    let (ctx, n) = setup();
    assert!(n >= 4, "the fixture must be large enough to exercise off-diagonal elements (n = {n})");
    let (s8, _) = s8_and_dense(&ctx, n);
    let g8 = PackedERITensorView::<f64>::new(n, &s8);
    let truth = brute_force_k(&g8, &nonsymmetric_density(n, 0), n, -1.0);
    let symmetric_part = brute_force_k(&g8, &symmetric_density(n, 0), n, -1.0);
    let worst = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (truth[(i, j)] - symmetric_part[(i, j)]).abs()).fold(0.0_f64, f64::max);
    assert!(worst > 1e-3, "the symmetric part's K agreed with the true K to {worst}: the asymmetry is not being exercised");
}
