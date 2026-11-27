// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Verifies the decontracted (primitive) shell-pair pathway -- set_primitive_shell_pairs/
// compute_one_electron_integrals(.., true)/compute_one_electron_picture_change_integrals: a
// contracted context's primitive integrals are those of the primitive context built from the same
// input, and a segmented context's decontraction folds shared exponents into the same primitives.

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};

fn assert_close(a: &[f64], b: &[f64], tol: f64, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch ({} vs {})", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < tol, "{label}: mismatch at index {i}: {x} vs {y}");
    }
}

fn primitive_matrix(path: &str, contraction_type: ContractionType, operator: OneElectronOperator<'_>) -> Vec<f64> {
    let mut ctx = AedelstenContext::new(path, contraction_type, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_prim_unique();
    let mut output = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut output, operator, true).unwrap();
    output
}

#[test]
fn general_context_primitive_integrals_match_the_primitive_context() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let operators: [(&str, fn() -> OneElectronOperator<'static>, f64); 3] = [
        ("overlap", OneElectronOperator::overlap, 1e-10),
        ("kinetic", OneElectronOperator::kinetic, 1e-8),
        ("nuclear attraction", || OneElectronOperator::nuclear_attraction(None), 1e-8),
    ];
    for (label, operator, tolerance) in operators {
        let decontracted = primitive_matrix(path, ContractionType::General, operator());
        let primitive = primitive_matrix(path, ContractionType::Primitive, operator());
        assert_close(&decontracted, &primitive, tolerance, label);
    }
}

// Segmented decontraction is allowed to see the same exponent shared (nonzero) by
// more than one contracted AO in the same (centre, lqn) block -- see
// shell_types::fold_segmented_shells' doc comment -- so this exercises that dedup
// path specifically: the folded primitives must be exactly the general context's.
#[test]
fn segmented_decontraction_folds_shared_exponents() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/segmented_shared_exponent_test.inp");
    let segmented = primitive_matrix(path, ContractionType::Segmented, OneElectronOperator::overlap());
    let general = primitive_matrix(path, ContractionType::General, OneElectronOperator::overlap());
    assert_close(&segmented, &general, 1e-10, "segmented overlap");
}

// compute_one_electron_picture_change_integrals on an already-primitive
// (ContractionType::Primitive) context must work off its own shell_pairs
// (set_primitive_shell_pairs is a no-op there, since there's nothing left to
// decontract), and must
// agree with the General-shell context's own decontracted
// picture-change integrals, since both describe the exact same primitives.
#[test]
fn primitive_contraction_type_picture_change_integrals_match_decontracted_general() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");

    // The pVp picture-change kernel in the scratch of set_scratch.
    let mut prim_ctx = AedelstenContext::new(path, ContractionType::Primitive, &aedelsten_mpi::Communicator::single()).unwrap();
    prim_ctx.set_shell_pairs(None).unwrap();
    prim_ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n_prim = prim_ctx.get_n_prim_unique();
    let mut prim_pvp = vec![0.0; n_prim * n_prim];
    prim_ctx.compute_one_electron_picture_change_integrals(
        &mut prim_pvp,
        prim_ctx.get_nuclear_attraction_operator(),
        false,
    ).unwrap();

    let mut general_ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    general_ctx.set_primitive_shell_pairs(None).unwrap();
    general_ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n_general = general_ctx.get_n_prim_unique();
    let mut general_pvp = vec![0.0; n_general * n_general];
    general_ctx.compute_one_electron_picture_change_integrals(
        &mut general_pvp,
        general_ctx.get_nuclear_attraction_operator(),
        false,
    ).unwrap();

    assert!(prim_pvp.iter().any(|&v| v.abs() > 1e-8), "sanity check: pvp shouldn't be all-zero for a real nuclear potential");

    assert_eq!(n_prim, n_general);
    assert_close(&prim_pvp, &general_pvp, 1e-8, "primitive vs decontracted-general pvp");
}

// The primitive context's own contract_one_electron_matrix is a no-op copy (nothing
// left to contract for an already-primitive context) -- same call shape as
// General/Segmented, just skipping the contraction-matrix sandwich.
#[test]
fn primitive_contraction_type_contract_is_a_no_op_copy() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::Primitive, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();
    let mut primitive_overlap = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut primitive_overlap, OneElectronOperator::overlap(), false).unwrap();

    let mut output = vec![0.0; n * n];
    ctx.contract_one_electron_matrix(&primitive_overlap, &mut output).unwrap();

    assert_eq!(output, primitive_overlap);
}
