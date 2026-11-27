// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// AIMP library entries that carry their own orthogonality functions as basis functions (Molcas legacy, e.g.
// Mg.TEST-EMB-AIMP.Larsson.14s.1s.ECP.MgO-orthogonality) are illegal: those functions lie in the span of the PROJOP
// orbitals, so the resolution basis (primary union PROJOP) would be linearly dependent. The parser rejects them
// structurally (basis_parser's own tests cover every library entry); this checks that the rejection reaches a caller
// building a context, and that the message points to the legal route (the 0s.0s entry with ORTHOGONALITY).

mod common;

use aedelsten::{AedelstenContext, ContractionType};

#[test]
fn a_context_with_an_orthogonality_function_aimp_is_rejected() {
    let path = common::temp_path("orthogonality_function_aimp.inp");
    std::fs::write(
        &path,
        "Basis set\n    Mg.TEST-EMB-AIMP.Larsson.14s.1s.ECP.MgO-orthogonality\n    Na1 0.0 0.0 0.0\nEnd of Basis\n\
         Basis set\n    H.x2c-SVPall\n    H1 0.15 -0.2 4.0\nEnd of Basis\n",
    )
    .unwrap();
    let result = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single());
    std::fs::remove_file(&path).ok();
    let Err(error) = result else {
        panic!("an AIMP entry with orthogonality functions must not give a context");
    };
    let message = format!("{error}");
    assert!(message.contains("Mg.TEST-EMB-AIMP.Larsson.14s.1s.ECP.MgO-orthogonality") && message.contains("ORTHOGONALITY"), "{message}");
}
