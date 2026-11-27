// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Measurement shared by the library and the programs that use it, one instance per process: span timing as a tree
//! per thread (`spans`), the citation registry (`citations`) and memory use (`memory`).

pub mod citations;
pub mod memory;
pub mod spans;

pub use citations::{bibtex, cite, cite_data, citations, clear_citations, Citation, Record, Reference};
pub use spans::{count, fine_report, fine_span, report, reset, span, tree, FineSpan, Span, SpanNode, SpanTree, Totals, FINE_ENABLED};

