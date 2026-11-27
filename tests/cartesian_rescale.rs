// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Smoke test for the AedelstenContext::rescale_cartesian_ao_rows/columns wiring -- exact
// correctness of the underlying factors and of rescale_rows/rescale_columns themselves is already
// pinned down by context/basis_set.rs's and cart2sph's own unit tests; this only checks
// that the context-level plumbing (size validation, basis lookup, matrix view construction) is
// wired correctly end to end.

mod common;

use aedelsten::{AedelstenContext, ContractionType, SharedCartesianReference};

// O x2c-SVPall's d functions default to Spherical (no rescaling needed under any convention), so
// this fixture forces them Cartesian -- the only case rescale_cartesian_ao_rows/columns has
// anything to actually do.
const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");

#[test]
fn wrong_sized_matrix_is_rejected() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n_ao = ctx.get_n_cont();
    let mut too_small = vec![1.0; n_ao]; // claims n_cols = 2 below, but only has room for 1
    assert!(ctx.rescale_cartesian_ao_rows(SharedCartesianReference::MOLCAS, 2, &mut too_small).is_err());
    let mut too_small = vec![1.0; n_ao];
    assert!(ctx.rescale_cartesian_ao_columns(SharedCartesianReference::MOLCAS, 2, &mut too_small).is_err());
}

#[test]
fn rows_rescaling_broadcasts_the_same_per_ao_factor_across_every_column() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n_ao = ctx.get_n_cont();
    let n_cols = 3;
    let mut matrix = vec![1.0; n_ao * n_cols];
    let rescaled = ctx.rescale_cartesian_ao_rows(SharedCartesianReference::MOLCAS, n_cols, &mut matrix).unwrap();
    for row in 0..n_ao {
        let factor = matrix[row];
        assert!(factor > 0.0 && factor.is_finite());
        for col in 1..n_cols {
            assert_eq!(matrix[row + col * n_ao], factor, "row {row} col {col}");
        }
    }
    // This basis (O x2c-SVPall) has Cartesian d functions, so the rescale is not a trivial no-op.
    assert!(matrix[0..n_ao].iter().any(|&f| f != 1.0), "expected at least one non-trivial rescale factor");
    assert!(rescaled, "return value should report that rescaling actually happened");
}

#[test]
fn rescaling_reports_false_when_the_basis_needs_no_rescaling() {
    // s/p are always Cartesian and always self-normalized (factor 1.0 under every convention),
    // so a basis with no d/f/g/... shells at all should be a genuine, reported no-op.
    let path = common::temp_path("cartesian_rescale_noop_test.inp");
    std::fs::write(&path, "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\n").unwrap();
    let ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).unwrap();

    let n_ao = ctx.get_n_cont();
    let mut matrix = vec![1.0; n_ao];
    let rescaled = ctx.rescale_cartesian_ao_rows(SharedCartesianReference::MOLCAS, 1, &mut matrix).unwrap();
    assert!(!rescaled);
    assert!(matrix.iter().all(|&f| f == 1.0));
}

#[test]
fn columns_rescaling_broadcasts_the_same_per_ao_factor_across_every_row() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n_ao = ctx.get_n_cont();
    let n_rows = 3;
    let mut matrix = vec![1.0; n_rows * n_ao];
    ctx.rescale_cartesian_ao_columns(SharedCartesianReference::MOLCAS, n_rows, &mut matrix).unwrap();
    for col in 0..n_ao {
        let factor = matrix[col * n_rows];
        for row in 1..n_rows {
            assert_eq!(matrix[row + col * n_rows], factor, "row {row} col {col}");
        }
    }
}

#[test]
fn dalton_and_molcas_conventions_disagree_on_a_cartesian_shell() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n_ao = ctx.get_n_cont();
    let mut molcas = vec![1.0; n_ao];
    ctx.rescale_cartesian_ao_rows(SharedCartesianReference::MOLCAS, 1, &mut molcas).unwrap();
    let mut dalton = vec![1.0; n_ao];
    ctx.rescale_cartesian_ao_rows(SharedCartesianReference::DALTON, 1, &mut dalton).unwrap();
    assert_ne!(molcas, dalton);
}

#[test]
fn unrescaling_undoes_rescaling_and_reports_the_same_thing() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n_ao = ctx.get_n_cont();
    let n_cols = 4;
    // Distinct, non-trivial entries so a wrong factor on any row shows up.
    let original: Vec<f64> = (0..n_ao * n_cols).map(|i| 0.3 + 0.017 * i as f64).collect();

    let mut matrix = original.clone();
    let changed = ctx.rescale_cartesian_ao_rows(SharedCartesianReference::MOLCAS, n_cols, &mut matrix).unwrap();
    assert!(changed && matrix != original, "this basis has Cartesian d functions");

    let changed_back = ctx.unrescale_cartesian_ao_rows(SharedCartesianReference::MOLCAS, n_cols, &mut matrix).unwrap();
    assert!(changed_back);
    for (a, b) in matrix.iter().zip(&original) {
        assert!((a - b).abs() < 1e-14 * b.abs().max(1.0), "{a} vs {b}");
    }

    // Wrong sizes are rejected the same way as in the forward direction.
    let mut too_small = vec![1.0; n_ao];
    assert!(ctx.unrescale_cartesian_ao_rows(SharedCartesianReference::MOLCAS, 2, &mut too_small).is_err());
}
