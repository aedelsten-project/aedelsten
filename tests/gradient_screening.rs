// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The two-electron gradient's screening (the derivative Schwarz bounds times the largest density product a quartet
// touches) against the unscreened gradient: a density that decays as a converged one does (the overlap matrix), and a
// difference density (the same, 1e-5 times smaller). The quartets skipped are counted in the timing report.

mod common;

use aedelsten::{AedelstenContext, ContractionType, ERIRangeOperator, OneElectronOperator, TwoElectronTerm, TwoElectronTermKind};

// Two waters 6 bohr apart and a hydrogen 14 bohr out: pairs over a wide range of bounds. ANO-RCC generally contracted
// (bounds over shared primitives with coefficients of both signs); x2c-SVPall, a segmented set, for the segmented mode.
const ANO: &str = "Basis set\n    O.ANO-RCC...2s1p\n    O1 0.0 0.0 0.1\n    O2 6.0 0.3 0.0\nEnd of Basis\nBasis set\n    H.ANO-RCC...1s\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\n    H3 6.0 1.6 0.9\n    H4 6.8 -1.0 0.7\n    H5 14.0 0.0 0.0\nEnd of Basis\n";
const SVP: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\n    O2 6.0 0.3 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\n    H3 6.0 1.6 0.9\n    H4 6.8 -1.0 0.7\n    H5 14.0 0.0 0.0\nEnd of Basis\n";

const CASES: [(&str, &str, ContractionType); 2] = [("general", ANO, ContractionType::General), ("segmented", SVP, ContractionType::Segmented)];

fn context(name: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("gradient_screening_{name}.inp"));
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

// The quartets the gradient walk skipped on this thread so far.
fn skipped() -> u64 {
    let report = aedelsten::timing_report();
    report.iter().find(|s| s.get_label() == "eri::gradient::screened").map_or(0, |s| s.get_calls())
}

// The restricted two-electron gradient of density `d`, `1/2 (J - 1/2 K)`, with the threshold.
fn gradient(ctx: &AedelstenContext, d: &[f64], threshold: Option<f64>) -> Vec<f64> {
    let terms = [
        TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: d, right: d, coeff: 1.0 },
        TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: d, right: d, coeff: -0.5 },
    ];
    let mut flat = vec![0.0; 3 * ctx.get_n_gradient_centres()];
    ctx.compute_two_electron_gradient(&mut flat, &terms, ERIRangeOperator::FullRange, threshold, None).unwrap();
    flat
}

fn largest_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn screened_gradients_match_the_unscreened_ones_and_skip_more_for_a_difference_density() {
    for (name, input, contraction) in CASES {
        let ctx = context(name, input, contraction);
        let threshold = 1e-10;
        let mut skipped_of = Vec::new();
        for (label, d) in [("density", overlap(&ctx, 1.0, 0.7)), ("difference density", overlap(&ctx, 1e-5, 1.3))] {
            let exact = gradient(&ctx, &d, None);
            let before = skipped();
            let screened = gradient(&ctx, &d, Some(threshold));
            let count = skipped() - before;
            // Every skipped quartet changes each gradient component it touches by below the threshold, a few
            // thousand of them.
            let difference = largest_difference(&screened, &exact);
            assert!(difference < 1e4 * threshold, "{name}, {label}: the gradient off by {difference:e} ({count} skipped)");
            eprintln!("{name}, {label}: {count} quartets skipped; the gradient off by {difference:.1e}");
            skipped_of.push(count);
        }
        assert!(skipped_of[1] > skipped_of[0], "{name}: the difference density skips no more quartets ({skipped_of:?})");
    }
}

#[test]
fn a_zero_density_gives_a_zero_gradient_and_skips_every_off_diagonal_quartet() {
    let ctx = context("zero", ANO, ContractionType::General);
    let n = ctx.get_n_cont();
    let before = skipped();
    let flat = gradient(&ctx, &vec![0.0; n * n], Some(1e-12));
    assert!(skipped() > before);
    assert!(flat.iter().all(|&g| g == 0.0));
}
