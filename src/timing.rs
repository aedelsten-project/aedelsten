// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Where this process's time went inside the library, for a caller's own timing report. Three sources, in two
// layers:
//
// - Time in MPI collectives (always on): every rank's parallel loop ends in an allreduce that waits for the
//   slowest rank, so a caller's wall-clock time of a library call is the same on every rank. Subtracting the
//   collective time spent during the call leaves this rank's own work.
// - Coarse spans (always on, src/instrumentation's `span`): a handful per library call, e.g. an ERI walk's own
//   rows and its allreduce, an AIMP's core J/K kernel, one per one-electron operator call (`oneint::<operator>`,
//   e.g. `oneint::overlap`, `oneint::contract_kinetic_gradient`, `oneint::projection_overlap`) and its allreduce
//   (`oneint::allreduce`).
// - Fine spans (the `profile` feature, `fine_span`): per shell-quartet batch and per digest call; per one-electron
//   shell pair, its three stages (`oneint::primitives`, the kernel over the primitive products;
//   `oneint::contract_transform`, contraction and spherical transformation; `oneint::distribute`, placement into
//   the output, or `oneint::contract_density` for the gradient contractions), summed over all operators. Empty
//   without the feature.
//
// All per process: under MPI, each rank reads its own, and gathering them is the caller's business.

/// Seconds this process has spent in MPI collectives (communication and waiting for the other ranks) since it
/// started. 0 without the `mpi` feature or on a single rank.
pub fn mpi_collective_seconds() -> f64 {
    aedelsten_mpi::collective_seconds()
}

/// Whether the library was built with the `profile` feature, i.e. whether `profile_report` can return anything.
pub fn profile_enabled() -> bool {
    instrumentation::FINE_ENABLED
}

/// One label's totals from `timing_report` or `profile_report`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimingSpan {
    label: &'static str,
    seconds: f64,
    calls: u64,
}

impl TimingSpan {
    /// The span's label.
    pub fn get_label(&self) -> &'static str {
        self.label
    }

    /// The seconds spent in it, summed over its calls.
    pub fn get_seconds(&self) -> f64 {
        self.seconds
    }

    /// How often it ran.
    pub fn get_calls(&self) -> u64 {
        self.calls
    }
}

fn spans(totals: instrumentation::Totals) -> Vec<TimingSpan> {
    totals.into_iter().map(|(label, total, calls)| TimingSpan { label, seconds: total.as_secs_f64(), calls }).collect()
}

/// The coarse span totals so far (always recorded), in the order the labels were first seen.
pub fn timing_report() -> Vec<TimingSpan> {
    spans(instrumentation::report())
}

/// The fine span totals so far (`profile` builds only; empty otherwise). They nest inside the coarse ones
/// (`eri::batch_*` and `eri::direct::distribute` inside `eri::direct::rows`; the `oneint::` stages inside the
/// `oneint::<operator>` spans).
pub fn profile_report() -> Vec<TimingSpan> {
    spans(instrumentation::fine_report())
}
