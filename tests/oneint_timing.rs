// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The one-electron integrals' timing spans: one coarse span per operator call, always recorded
// (`aedelsten::timing_report`); with the `profile` feature, the fine per-shell-pair stages (`profile_report`).
// Totals are per thread, and each test runs on a thread of its own.

mod common;

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";

#[test]
fn every_operator_call_is_timed_and_the_shell_pair_stages_with_profile() {
    let path = common::temp_path("oneint_timing.inp");
    std::fs::write(&path, WATER).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::Segmented, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((16, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();

    let mut matrix = vec![0.0; n * n];
    let mut planes = vec![0.0; 3 * n * n];
    for _ in 0..2 {
        ctx.compute_one_electron_integrals(&mut matrix, OneElectronOperator::overlap(), false).unwrap();
    }
    ctx.compute_one_electron_integrals(&mut planes, OneElectronOperator::angular_momentum([0.0; 3]), false).unwrap();

    let calls = |report: &[aedelsten::TimingSpan], label: &str| report.iter().find(|s| s.get_label() == label).map(|s| s.get_calls());
    let coarse = aedelsten::timing_report();
    assert_eq!(calls(&coarse, "oneint::overlap"), Some(2), "{coarse:?}");
    assert_eq!(calls(&coarse, "oneint::angular_momentum"), Some(1), "{coarse:?}");

    let fine = aedelsten::profile_report();
    if aedelsten::profile_enabled() {
        for stage in ["oneint::primitives", "oneint::contract_transform", "oneint::distribute"] {
            assert!(calls(&fine, stage).is_some_and(|c| c > 0), "{stage} missing from {fine:?}");
        }
    } else {
        assert!(fine.is_empty(), "fine spans without the profile feature: {fine:?}");
    }
}
