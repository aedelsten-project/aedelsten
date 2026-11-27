// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The context's general direct Fock build against the dense S1 builders: which density and
// matrix goes where. The accumulators themselves are checked in fock_constructors' block_digestion test; the
// restricted and unrestricted builds through the context in density_screening.

use aedelsten::arrays::array4d::Array4DView;
use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};

fn setup() -> (aedelsten::AedelstenContext, usize) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let mut ctx = aedelsten::AedelstenContext::new(path, aedelsten::ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(Some(1e-30)).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(None).unwrap();
    let n = ctx.get_n_cont();
    (ctx, n)
}

// Deterministic, non-trivial symmetric "density matrix" -- not a real SCF
// density (no idempotency/trace requirement needed here), just a fixed,
// asymmetric-across-indices set of values to exercise the accumulator with.
fn symmetric_density(n: usize, salt: usize) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in i..n {
            let v = 0.1 + 0.05 * (((i * 7 + j * 3 + salt) % 11) as f64);
            p[(i, j)] = v;
            p[(j, i)] = v;
        }
    }
    p
}

fn assert_matrices_close(label: &str, direct: &Matrix<f64>, reference: &Matrix<f64>, n: usize) {
    for i in 0..n {
        for j in 0..n {
            let (x, y) = (direct[(i, j)], reference[(i, j)]);
            assert!((x - y).abs() < 1e-9, "{label} mismatch at ({i},{j}): direct {x}, reference {y}");
        }
    }
}

// A non-symmetric density (a transition density, a spin-flip density). The direct
// accumulator scatters every permutation, so it must equal the general dense K (which contracts the
// full tensor) even though D[b,d] != D[d,b].
fn nonsymmetric_density(n: usize, salt: usize) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            p[(i, j)] = 0.3 * (((i * 7 + j * 3 + salt) % 11) as f64) - 1.1 + 0.07 * (i as f64) - 0.13 * (j as f64);
        }
    }
    p
}

// One density of any symmetry (a transition density, the spin-flip density of a stability analysis): J from its
// symmetric part, K not symmetrized. Against the dense builders on the
// stored S1 tensor, which contract the full tensor with any density; full and long range.
#[test]
fn general_direct_fock_matches_dense_reference_for_a_nonsymmetric_density() {
    let (ctx, n) = setup();
    let p = nonsymmetric_density(n, 3);
    for range in [aedelsten::ERIRangeOperator::FullRange, aedelsten::ERIRangeOperator::LongRange { omega: 0.4 }] {
        let mut dense = vec![0.0; n * n * n * n];
        ctx.compute_eri_stored(&mut dense, aedelsten::ERIAOSymmetry::S1, range).unwrap();
        let g = Array4DView::new(n, n, n, n, &dense);
        let mut j_ref = Matrix::<f64>::new(n, n);
        aedelsten::fock_constructors::construct_j_stored(&g, &p, &mut j_ref, 1.0);
        let mut k_ref = Matrix::<f64>::new(n, n);
        aedelsten::fock_constructors::construct_k_stored(&g, &p, &mut k_ref, -0.5);

        let (mut j_direct, mut k_direct) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
        ctx.compute_fock_direct_general(p.get_data(), j_direct.get_data_mut(), k_direct.get_data_mut(), 1.0, -0.5, range, None).unwrap();
        assert_matrices_close("J", &j_direct, &j_ref, n);
        assert_matrices_close("K", &k_direct, &k_ref, n);

        // J is symmetric; K is not (the test would be vacuous if it were).
        let asymmetry = |m: &Matrix<f64>| (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (m[(i, j)] - m[(j, i)]).abs()).fold(0.0_f64, f64::max);
        assert!(asymmetry(&j_direct) < 1e-12, "J not symmetric: {}", asymmetry(&j_direct));
        assert!(asymmetry(&k_direct) > 1e-6, "K came out symmetric ({}): the test does not exercise the non-symmetric case", asymmetry(&k_direct));

        // A zero coefficient skips that matrix and leaves it zeroed.
        ctx.compute_fock_direct_general(p.get_data(), j_direct.get_data_mut(), k_direct.get_data_mut(), 0.0, -0.5, range, None).unwrap();
        assert!(j_direct.get_data().iter().all(|&x| x == 0.0));
        assert_matrices_close("K alone", &k_direct, &k_ref, n);
    }
}

#[test]
fn general_direct_fock_of_a_symmetric_density_is_the_restricted_one() {
    let (ctx, n) = setup();
    let p = symmetric_density(n, 4);
    let (mut j_general, mut k_general) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    let (mut j_restricted, mut k_restricted) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    ctx.compute_fock_direct_general(p.get_data(), j_general.get_data_mut(), k_general.get_data_mut(), 1.0, -0.5, aedelsten::ERIRangeOperator::FullRange, None).unwrap();
    ctx.compute_fock_direct_restricted(p.get_data(), j_restricted.get_data_mut(), k_restricted.get_data_mut(), 1.0, -0.5, aedelsten::ERIRangeOperator::FullRange, None).unwrap();
    for (label, a, b) in [("J", &j_general, &j_restricted), ("K", &k_general, &k_restricted)] {
        for (x, y) in a.get_data().iter().zip(b.get_data()) {
            assert!((x - y).abs() < 1e-12, "{label}: general {x}, restricted {y}");
        }
    }
}

// Several densities of any symmetry in one walk: each density's J and K are its own walk's, bitwise without a
// threshold (the same batches digested in the same order), and to within the screening with one (the walk keeps
// any batch one of the densities needs). A zero coefficient leaves that part zeroed; wrong lengths are refused.
#[test]
fn general_direct_fock_of_several_densities_is_each_densitys_own() {
    let (ctx, n) = setup();
    let densities: Vec<Matrix<f64>> = vec![nonsymmetric_density(n, 3), symmetric_density(n, 4), nonsymmetric_density(n, 7)];
    let size = n * n;
    let mut stacked = vec![0.0; 3 * size];
    for (chunk, p) in stacked.chunks_exact_mut(size).zip(&densities) {
        chunk.copy_from_slice(p.get_data());
    }
    let range = aedelsten::ERIRangeOperator::FullRange;
    for threshold in [None, Some(1e-10)] {
        let (mut j, mut k) = (vec![0.0; 3 * size], vec![0.0; 3 * size]);
        ctx.compute_fock_direct_general_multi(&stacked, &mut j, &mut k, 3, 1.0, -0.5, range, threshold).unwrap();
        for ((p, j), k) in densities.iter().zip(j.chunks_exact(size)).zip(k.chunks_exact(size)) {
            let (mut j_one, mut k_one) = (vec![0.0; size], vec![0.0; size]);
            ctx.compute_fock_direct_general(p.get_data(), &mut j_one, &mut k_one, 1.0, -0.5, range, threshold).unwrap();
            for (label, a, b) in [("J", j, &j_one), ("K", k, &k_one)] {
                for (x, y) in a.iter().zip(b.iter()) {
                    match threshold {
                        None => assert_eq!(x, y, "{label}: one walk for three densities against one each"),
                        Some(t) => assert!((x - y).abs() < 10.0 * t, "{label}: one walk {x}, one each {y}"),
                    }
                }
            }
        }
    }

    let (mut j, mut k) = (vec![1.0; 3 * size], vec![1.0; 3 * size]);
    ctx.compute_fock_direct_general_multi(&stacked, &mut j, &mut k, 3, 0.0, -0.5, range, None).unwrap();
    assert!(j.iter().all(|&x| x == 0.0), "J with a zero coefficient");
    assert!(k.iter().any(|&x| x != 0.0), "K built");

    let mut short = vec![0.0; 3 * size - 1];
    assert!(ctx.compute_fock_direct_general_multi(&stacked, &mut short, &mut k, 3, 1.0, -0.5, range, None).is_err());
    assert!(ctx.compute_fock_direct_general_multi(&stacked, &mut j, &mut k, 2, 1.0, -0.5, range, None).is_err());
    assert!(ctx.compute_fock_direct_general_multi(&[], &mut [], &mut [], 0, 1.0, -0.5, range, None).is_err());
}
