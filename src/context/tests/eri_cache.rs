// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Verifies the cacheable-ERI-data path (AedelstenContext::set_eri_data + compute_fock_direct, backed by
// eri::EriData) against J and K built from the single-shot tensor (compute_eri_stored), and checks that
// Schwarz-threshold elimination and pruning actually drop shell pairs. In the crate: the bounds and the re-prune are
// not public.

use arrays::array4d::Array4DView;
use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};

fn setup() -> (crate::AedelstenContext, usize) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let mut ctx = crate::AedelstenContext::new(path, crate::ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    // overlap_threshold near zero: let every shell pair (however distant) through into shell_pairs, so it's
    // Schwarz screening below -- not the overlap prefilter -- that's actually being exercised.
    ctx.set_shell_pairs(Some(1e-30)).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();
    (ctx, n)
}

fn density(n: usize) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in i..n {
            let v = 0.1 + 0.05 * (((i * 7 + j * 3) % 11) as f64);
            p[(i, j)] = v;
            p[(j, i)] = v;
        }
    }
    p
}

/// J and K from the full S1 tensor.
fn reference_jk(ctx: &crate::AedelstenContext, p: &Matrix<f64>, n: usize) -> (Matrix<f64>, Matrix<f64>) {
    let mut dense = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut dense, crate::ERIAOSymmetry::S1, crate::ERIRangeOperator::FullRange).unwrap();
    let g = Array4DView::new(n, n, n, n, &dense);
    let mut j = Matrix::<f64>::new(n, n);
    let mut k = Matrix::<f64>::new(n, n);
    crate::fock_constructors::construct_j_and_k_stored(&g, p, &mut j, &mut k, 1.0, -0.5);
    (j, k)
}

/// J and K from the resident ERI data.
fn direct_jk(ctx: &crate::AedelstenContext, p: &Matrix<f64>, n: usize, pairwise_threshold: Option<f64>) -> (Matrix<f64>, Matrix<f64>) {
    let mut j = Matrix::<f64>::new(n, n);
    let mut k = Matrix::<f64>::new(n, n);
    ctx.compute_fock_direct_restricted(p.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, crate::ERIRangeOperator::FullRange, pairwise_threshold)
        .unwrap();
    (j, k)
}

fn max_abs_difference(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
    a.get_data().iter().zip(b.get_data()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn eri_data_cache_matches_plain_path_and_supports_schwarz_elimination() {
    let (mut ctx, n) = setup();
    let p = density(n);
    let (j_ref, k_ref) = reference_jk(&ctx, &p, n);

    // No shell pair eliminated: the resident data must reproduce J and K from the full tensor.
    ctx.set_eri_data(None).unwrap();
    let full_bounds = ctx.eri_data_schwarz_bounds().unwrap();
    let (j, k) = direct_jk(&ctx, &p, n, None);
    assert!(max_abs_difference(&j, &j_ref) < 1e-12, "J");
    assert!(max_abs_difference(&k, &k_ref) < 1e-12, "K");
    ctx.free_eri_data();

    // An enormous threshold eliminates every shell pair, so nothing is ever distributed.
    ctx.set_eri_data(Some(1e30)).unwrap();
    assert!(ctx.eri_data_schwarz_bounds().unwrap().is_empty());
    let (j, k) = direct_jk(&ctx, &p, n, None);
    assert!(j.get_data().iter().chain(k.get_data()).all(|v| *v == 0.0));
    ctx.free_eri_data();

    // A threshold just above the smallest observed bound eliminates at least that one shell pair (three
    // chemically identical H atoms make ties possible, so no exact count), and a build from the reduced list
    // (original_index lookups into a shell_pairs slice longer than eri_data itself) runs.
    let min_bound = full_bounds.iter().cloned().fold(f64::INFINITY, f64::min);
    ctx.set_eri_data(Some(min_bound + 1e-12)).unwrap();
    let partial_bounds = ctx.eri_data_schwarz_bounds().unwrap();
    assert!(
        partial_bounds.len() < full_bounds.len(),
        "expected at least one shell pair eliminated: {} vs {}",
        partial_bounds.len(), full_bounds.len(),
    );
    let _ = direct_jk(&ctx, &p, n, None);
}

#[test]
fn pairwise_cauchy_schwarz_screening_skips_negligible_off_diagonal_blocks() {
    let (mut ctx, n) = setup();
    let p = density(n);
    let (j_ref, k_ref) = reference_jk(&ctx, &p, n);
    ctx.set_eri_data(None).unwrap();

    // A threshold of exactly 0.0 can't exceed any real bound(ab)*bound(cd) product: nothing is skipped.
    let (j, k) = direct_jk(&ctx, &p, n, Some(0.0));
    assert!(max_abs_difference(&j, &j_ref) < 1e-12, "J with threshold 0.0");
    assert!(max_abs_difference(&k, &k_ref) < 1e-12, "K with threshold 0.0");

    // A threshold above max_bound^2 skips every off-diagonal block, while the diagonal blocks are never
    // screened; H1-H2, at a bonding distance, has substantial cross terms, so J changes.
    let bounds = ctx.eri_data_schwarz_bounds().unwrap();
    let max_bound = bounds.iter().cloned().fold(0.0_f64, f64::max);
    let (j, _) = direct_jk(&ctx, &p, n, Some(max_bound * max_bound * 10.0));
    assert!(max_abs_difference(&j, &j_ref) > 1e-3, "an enormous pairwise threshold should have skipped every off-diagonal block");
}

#[test]
fn prune_eri_data_uses_the_rigorous_max_bound_criterion() {
    let (mut ctx, n) = setup();

    // Keep everything (schwarz_threshold=None): pruning is the only elimination happening in this test.
    ctx.set_eri_data(None).unwrap();
    let full_bounds = ctx.eri_data_schwarz_bounds().unwrap();
    let max_bound = full_bounds.iter().cloned().fold(0.0_f64, f64::max);

    // threshold=0.0 can't exceed entry.schwarz_bound * max_bound for any real bound: a no-op.
    ctx.prune_eri_data(0.0).unwrap();
    assert_eq!(ctx.eri_data_schwarz_bounds().unwrap().len(), full_bounds.len(), "threshold=0.0 should prune nothing");

    // A threshold just above the smallest entry's rigorous product (bound * max_bound) prunes at least that
    // one entry (ties are possible, see above).
    let min_bound = full_bounds.iter().cloned().fold(f64::INFINITY, f64::min);
    let threshold = min_bound * max_bound + 1e-12;
    ctx.prune_eri_data(threshold).unwrap();
    let pruned_bounds = ctx.eri_data_schwarz_bounds().unwrap();
    assert!(
        pruned_bounds.len() < full_bounds.len(),
        "expected at least one entry pruned: {} vs {}",
        pruned_bounds.len(), full_bounds.len(),
    );

    // Every surviving bound still satisfies the criterion it was just tested against.
    for bound in &pruned_bounds {
        assert!(bound * max_bound >= threshold, "surviving bound {bound} should satisfy bound*max_bound >= threshold ({threshold})");
    }

    // A build from the pruned (reduced-length) EriData runs.
    let _ = direct_jk(&ctx, &density(n), n, None);
}
