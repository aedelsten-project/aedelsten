// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Smoke test for AedelstenContext::memory_breakdown -- not chasing exact byte counts (several
// fields are still shallow, best-effort estimates, see memory_breakdown's own doc comment), just
// that the API reports sane, field-by-field numbers that grow as more of the context is loaded.

use aedelsten::{AedelstenContext, ContractionType, MEMORY_BREAKDOWN_FIELDS};

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");

fn field_index(name: &str) -> usize {
    MEMORY_BREAKDOWN_FIELDS.iter().position(|f| f.to_str() == Ok(name)).unwrap()
}

#[test]
fn wrong_sized_buffer_is_rejected() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let mut too_small = vec![0; MEMORY_BREAKDOWN_FIELDS.len() - 1];
    assert!(ctx.memory_breakdown(&mut too_small).is_err());
}

#[test]
fn qm_region_is_the_only_nonzero_field_right_after_construction() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let mut breakdown = vec![0; MEMORY_BREAKDOWN_FIELDS.len()];
    ctx.memory_breakdown(&mut breakdown).unwrap();

    assert!(breakdown[field_index("qm_region")] > 0, "qm_region should be resident right after AedelstenContext::new");

    // Nothing else has been requested yet, so every other field should report 0.
    for (name, &bytes) in MEMORY_BREAKDOWN_FIELDS.iter().zip(breakdown.iter()) {
        if name.to_str() != Ok("qm_region") {
            assert_eq!(bytes, 0, "{name:?} should be 0 before it's ever requested");
        }
    }
}

#[test]
fn shell_pairs_becomes_nonzero_once_requested() {
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    let mut breakdown = vec![0; MEMORY_BREAKDOWN_FIELDS.len()];
    ctx.memory_breakdown(&mut breakdown).unwrap();
    assert!(breakdown[field_index("shell_pairs")] > 0);
}
