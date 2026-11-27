// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Smoke test for the AedelstenContext-level accessors wrapping BasisInfo's/InputStructure's own
// BasisFunctionId/PrimitiveId/Primitive/CentreLabel data -- exact correctness of the values
// themselves is already pinned down by basis's own molcas_export_id_tests; this only checks that
// the context-level plumbing (size validation, buffer copy) is wired correctly end to end.

use aedelsten::{AedelstenContext, BasisFunctionId, ContractionType, Primitive, PrimitiveId};

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");

#[test]
fn basis_function_ids_round_trip_through_the_context() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n_ao = ctx.get_n_cont();

    let mut wrong = vec![BasisFunctionId::default(); n_ao - 1];
    assert!(ctx.get_basis_function_ids(&mut wrong).is_err());

    let mut ids = vec![BasisFunctionId::default(); n_ao];
    ctx.get_basis_function_ids(&mut ids).unwrap();
    // The fields are private; compare through the getters.
    let first = &ids[0];
    assert_eq!((first.get_centre(), first.get_shell(), first.get_l(), first.get_m()), (1, 1, 0, 0)); // s shell
    assert!(ids.iter().all(|id| id.get_centre() == 1));
}

#[test]
fn primitive_ids_and_primitives_round_trip_through_the_context() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let n = ctx.get_n_primitive_rows();
    assert!(n > 0);

    let mut wrong = vec![Primitive { exponent: 0.0, coefficient: 0.0 }; 1];
    assert!(ctx.get_primitives(&mut wrong).is_err());

    let mut ids = vec![PrimitiveId { centre: 0, l: 0, shell: 0 }; n];
    ctx.get_primitive_ids(&mut ids).unwrap();
    let mut primitives = vec![Primitive { exponent: 0.0, coefficient: 0.0 }; n];
    ctx.get_primitives(&mut primitives).unwrap();

    assert!(ids.iter().all(|id| id.centre == 1));
    assert!(primitives.iter().all(|p| p.coefficient != 0.0 && p.exponent.is_finite()));
}

#[test]
fn centre_label_round_trips_through_the_context() {
    let ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let label = ctx.get_centre_label(0).unwrap();
    assert_eq!(label.to_string(), "O1");
    assert!(ctx.get_centre_label(1).is_none());
}
