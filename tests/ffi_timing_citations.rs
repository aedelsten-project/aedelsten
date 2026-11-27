// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the timings (the coarse and fine reports, the span tree, the MPI collectives' time) and of the
// reference list (each reference's text and topics, the BibTeX file, clearing), driven as a C, Fortran, Python or
// Julia caller would, against the Rust calls they wrap. One test: the reference list is the process's.

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};
use std::ffi::{CStr, c_char};

unsafe extern "C" {
    fn aedelsten_mpi_collective_seconds_c() -> f64;
    fn aedelsten_profile_enabled_c() -> i32;
    fn aedelsten_timing_count_c(fine: i32) -> i32;
    fn aedelsten_timing_span_c(fine: i32, index: i32, label: *mut c_char, label_len: i32, seconds: *mut f64, calls: *mut i64) -> i32;
    fn aedelsten_span_tree_count_c() -> i32;
    fn aedelsten_span_tree_node_c(index: i32, label: *mut c_char, label_len: i32, depth: *mut i32, fine: *mut i32, seconds: *mut f64, calls: *mut i64) -> i32;
    fn aedelsten_citation_count_c() -> i32;
    fn aedelsten_citation_length_c(index: i32, topics: i32) -> i64;
    fn aedelsten_citation_c(index: i32, topics: i32, buf: *mut c_char, buf_len: i32) -> i32;
    fn aedelsten_citations_bibtex_length_c() -> i64;
    fn aedelsten_citations_bibtex_c(buf: *mut c_char, buf_len: i32) -> i32;
    fn aedelsten_clear_citations_c();
}

fn text(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap().to_string()
}

// String `index` of the reference list through the C calls: its length, then a buffer of that size plus the NUL.
fn citation(index: i32, topics: i32) -> String {
    let len = unsafe { aedelsten_citation_length_c(index, topics) };
    assert!(len >= 0);
    let mut buf = vec![0 as c_char; len as usize + 1];
    assert_eq!(unsafe { aedelsten_citation_c(index, topics, buf.as_mut_ptr(), buf.len() as i32) }, 0);
    let mut short = vec![0 as c_char; len as usize];
    assert_ne!(unsafe { aedelsten_citation_c(index, topics, short.as_mut_ptr(), short.len() as i32) }, 0, "no room for the NUL");
    text(&buf)
}

#[test]
fn the_timings_and_the_references_are_the_rust_calls_answers() {
    // Something timed and cited: an overlap matrix.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/cartesian_rescale_test.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((16, arrays::scratch::ByteUnit::MiB)).unwrap();
    let mut s = vec![0.0; ctx.get_n_cont() * ctx.get_n_cont()];
    ctx.compute_one_electron_integrals(&mut s, OneElectronOperator::overlap(), false).unwrap();

    assert_eq!(unsafe { aedelsten_mpi_collective_seconds_c() }, aedelsten::mpi_collective_seconds());
    assert_eq!(unsafe { aedelsten_profile_enabled_c() } != 0, aedelsten::profile_enabled());

    // The coarse and fine reports, entry by entry (the spans are this thread's and nothing runs between the reads).
    for (fine, expected) in [(0, aedelsten::timing_report()), (1, aedelsten::profile_report())] {
        assert_eq!(unsafe { aedelsten_timing_count_c(fine) }, expected.len() as i32);
        for (i, span) in expected.iter().enumerate() {
            let (mut label, mut seconds, mut calls) = (vec![0 as c_char; 256], f64::NAN, -1i64);
            assert_eq!(unsafe { aedelsten_timing_span_c(fine, i as i32, label.as_mut_ptr(), 256, &mut seconds, &mut calls) }, 0);
            assert_eq!((text(&label).as_str(), seconds, calls), (span.get_label(), span.get_seconds(), span.get_calls() as i64));
        }
        let (mut label, mut seconds, mut calls) = (vec![0 as c_char; 256], 0.0, 0i64);
        assert_ne!(unsafe { aedelsten_timing_span_c(fine, expected.len() as i32, label.as_mut_ptr(), 256, &mut seconds, &mut calls) }, 0, "out of range");
    }
    assert!(aedelsten::timing_report().iter().any(|s| s.get_label() == "oneint::overlap" && s.get_calls() >= 1));

    // The span tree, node by node.
    let tree = instrumentation::tree();
    assert_eq!(unsafe { aedelsten_span_tree_count_c() }, tree.nodes().len() as i32);
    assert!(!tree.nodes().is_empty());
    for (i, node) in tree.nodes().iter().enumerate() {
        let (mut label, mut depth, mut fine, mut seconds, mut calls) = (vec![0 as c_char; 256], -1, -1, f64::NAN, -1i64);
        assert_eq!(unsafe { aedelsten_span_tree_node_c(i as i32, label.as_mut_ptr(), 256, &mut depth, &mut fine, &mut seconds, &mut calls) }, 0);
        assert_eq!(text(&label), node.label);
        assert_eq!((depth, fine != 0, seconds, calls), (node.depth as i32, node.fine, node.total.as_secs_f64(), node.calls as i64));
    }
    let mut label = vec![0 as c_char; 2];
    let (mut depth, mut fine, mut seconds, mut calls) = (0, 0, 0.0, 0i64);
    assert_ne!(unsafe { aedelsten_span_tree_node_c(0, label.as_mut_ptr(), 2, &mut depth, &mut fine, &mut seconds, &mut calls) }, 0, "a label too long");

    // The references: each one's text and topics, the BibTeX file, then clearing.
    let records = instrumentation::citations::citations();
    assert!(!records.is_empty());
    assert_eq!(unsafe { aedelsten_citation_count_c() }, records.len() as i32);
    for (i, record) in records.iter().enumerate() {
        assert_eq!(citation(i as i32, 0), record.reference.to_string());
        assert_eq!(citation(i as i32, 1), record.topics.join(", "));
    }
    assert_eq!(unsafe { aedelsten_citation_length_c(records.len() as i32, 0) }, -1, "out of range");
    let bibtex = instrumentation::citations::bibtex(&records);
    assert_eq!(unsafe { aedelsten_citations_bibtex_length_c() }, bibtex.len() as i64);
    let mut buf = vec![0 as c_char; bibtex.len() + 1];
    assert_eq!(unsafe { aedelsten_citations_bibtex_c(buf.as_mut_ptr(), buf.len() as i32) }, 0);
    assert_eq!(text(&buf), bibtex);
    unsafe { aedelsten_clear_citations_c() };
    assert!(instrumentation::citations::citations().is_empty());
    assert_eq!(unsafe { aedelsten_citation_count_c() }, 0);
}
