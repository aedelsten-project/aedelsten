// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// What a context registers with the citation registry: the method entries where they run, and the reference lines
// of the library entries its basis reads. The registry is process-wide and shared with the other tests of this binary,
// so the test checks what must be there, not what must not.

use aedelsten::{AedelstenContext, ContractionType};

const INPUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");

fn cited(doi: &str) -> Option<instrumentation::Record> {
    instrumentation::citations().into_iter().find(|r| r.reference.doi() == Some(doi))
}

#[test]
fn a_context_and_its_eri_data_are_cited() {
    let mut ctx = AedelstenContext::new(INPUT, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    // McMurchie-Davidson for every context; x2c-SVPall from the basis file's own reference line.
    assert!(cited("10.1016/0021-9991(78)90092-X").is_some());
    let basis = cited("10.1021/acs.jctc.7b00593").expect("the x2c-SVPall entry's reference");
    assert!(basis.topics.iter().any(|t| t == "Basis sets and model potentials"));
    assert!(basis.reference.to_string().starts_with("P. Pollak, F. Weigend."));

    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((16, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(Some(1e-12)).unwrap();
    // The ERI walk (after SHARK) and Schwarz screening.
    assert!(cited("10.1002/jcc.26942").is_some());
    assert!(cited("10.1002/jcc.540100111").is_some());
}
