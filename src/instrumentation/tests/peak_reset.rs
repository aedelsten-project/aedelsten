// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// `reset_peak` against the process-wide peak, in a binary of its own: the counters are global, so no other test may
// allocate while this one measures.

use instrumentation::memory::{live_bytes, peak_bytes, process_peak_bytes, reset_peak, TrackingAllocator};

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator::new();

const MIB: usize = 1 << 20;

#[test]
fn a_reset_peak_is_the_most_held_since_and_the_process_peak_stays() {
    let large = vec![1u8; 64 * MIB];
    std::hint::black_box(&large);
    drop(large);

    let baseline = live_bytes();
    reset_peak();
    assert_eq!(peak_bytes(), baseline);
    let small = vec![1u8; 8 * MIB];
    std::hint::black_box(&small);
    let (peak, process_peak) = (peak_bytes(), process_peak_bytes());
    drop(small);

    // The test harness's own allocations meanwhile are small.
    assert!(peak >= baseline + 8 * MIB && peak < baseline + 9 * MIB, "peak since the reset {peak}, baseline {baseline}");
    assert!(process_peak >= 64 * MIB, "the process peak {process_peak} keeps the large allocation");
}
