// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! This process's memory, measured rather than summed from the sizes of the structures:
//!
//! - `live_bytes`, `peak_bytes`, `process_peak_bytes`: what is allocated through Rust's allocator now, the most at any
//!   one time since the last `reset_peak` (a stage's or a phase's peak), and the most since the process started.
//!   Counted by `TrackingAllocator`, the system allocator with one atomic add per allocation, which a binary installs
//!   as its `#[global_allocator]` (a library never does); 0 while it is not installed. Memory that C, C++ or Fortran
//!   code allocates does not pass through it.
//! - `peak_rss_bytes`: the operating system's peak resident set size, which covers everything.
//! - `thread_allocations`, `thread_allocated_bytes` (the `thread-counts` feature): this thread's allocations and their
//!   bytes, for tests that check a path allocates nothing, or how much a setup allocates.
//!
//! Sizes are the sizes requested: a zeroed allocation counts in full before it is touched.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static PROCESS_PEAK: AtomicUsize = AtomicUsize::new(0);
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// The system allocator, counting this process's live and peak bytes (and, with the `thread-counts` feature, each
/// thread's allocations). Installed by a binary: `#[global_allocator] static A: TrackingAllocator =
/// TrackingAllocator::new();`.
#[derive(Debug, Default)]
pub struct TrackingAllocator;

impl TrackingAllocator {
    /// The allocator, for a `#[global_allocator]` static.
    pub const fn new() -> Self {
        TrackingAllocator
    }
}

fn add(bytes: usize) {
    #[cfg(feature = "thread-counts")]
    {
        let _ = THREAD_ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        let _ = THREAD_BYTES.try_with(|c| c.set(c.get() + bytes));
    }
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
    PROCESS_PEAK.fetch_max(live, Relaxed);
}

fn sub(bytes: usize) {
    LIVE.fetch_sub(bytes, Relaxed);
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size());
            if !INSTALLED.load(Relaxed) {
                INSTALLED.store(true, Relaxed);
            }
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        sub(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                add(new_size - layout.size());
            } else {
                sub(layout.size() - new_size);
            }
        }
        new_ptr
    }
}

/// Whether a [`TrackingAllocator`] is counting this process's allocations.
pub fn is_tracking() -> bool {
    INSTALLED.load(Relaxed)
}

/// Bytes allocated through Rust's allocator now; 0 without a [`TrackingAllocator`].
pub fn live_bytes() -> usize {
    LIVE.load(Relaxed)
}

/// The most bytes allocated through Rust's allocator at any one time since the last [`reset_peak`] (since the process
/// started, without one); 0 without a [`TrackingAllocator`].
pub fn peak_bytes() -> usize {
    PEAK.load(Relaxed)
}

/// Starts a new [`peak_bytes`] from what is allocated now, so that it gives the most held over the work that follows.
/// [`process_peak_bytes`] is not affected. An allocation by another thread during the reset may be missed by the new
/// peak.
pub fn reset_peak() {
    PEAK.store(LIVE.load(Relaxed), Relaxed);
}

/// The most bytes allocated through Rust's allocator at any one time since the process started, whatever
/// [`reset_peak`] did; 0 without a [`TrackingAllocator`].
pub fn process_peak_bytes() -> usize {
    PROCESS_PEAK.load(Relaxed)
}

/// The operating system's peak resident set size of this process, in bytes; 0 where unavailable.
pub fn peak_rss_bytes() -> usize {
    #[cfg(unix)]
    {
        // struct rusage on the 64-bit Linux and macOS ABIs: two timevals (16 bytes each), then `long ru_maxrss` and
        // 13 more longs.
        #[repr(C)]
        struct Rusage {
            times: [i64; 4],
            maxrss: i64,
            rest: [i64; 13],
        }
        unsafe extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        const RUSAGE_SELF: i32 = 0;
        let mut usage = Rusage { times: [0; 4], maxrss: 0, rest: [0; 13] };
        if unsafe { getrusage(RUSAGE_SELF, &mut usage) } != 0 || usage.maxrss < 0 {
            return 0;
        }
        // Bytes on macOS, kibibytes on Linux.
        if cfg!(target_os = "macos") { usage.maxrss as usize } else { usage.maxrss as usize * 1024 }
    }
    #[cfg(not(unix))]
    0
}

#[cfg(feature = "thread-counts")]
thread_local! {
    static THREAD_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static THREAD_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The allocations (growing reallocations included) this thread has made through a [`TrackingAllocator`] so far.
#[cfg(feature = "thread-counts")]
pub fn thread_allocations() -> usize {
    THREAD_ALLOCATIONS.with(|c| c.get())
}

/// The bytes this thread has allocated through a [`TrackingAllocator`] so far, frees not subtracted.
#[cfg(feature = "thread-counts")]
pub fn thread_allocated_bytes() -> usize {
    THREAD_BYTES.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[global_allocator]
    static ALLOCATOR: TrackingAllocator = TrackingAllocator::new();

    #[test]
    fn the_counters_follow_allocations_reallocations_and_frees() {
        let tracking = TrackingAllocator::new();
        let layout = Layout::from_size_align(1000, 8).unwrap();
        // The global counters are shared with the tests running in parallel: compare this thread's own counts.
        #[cfg(feature = "thread-counts")]
        let (allocations, bytes) = (thread_allocations(), thread_allocated_bytes());
        unsafe {
            let p = tracking.alloc(layout);
            assert!(!p.is_null());
            let p = tracking.realloc(p, layout, 4000);
            let p = tracking.realloc(p, Layout::from_size_align(4000, 8).unwrap(), 500);
            tracking.dealloc(p, Layout::from_size_align(500, 8).unwrap());
        }
        #[cfg(feature = "thread-counts")]
        {
            assert_eq!(thread_allocations() - allocations, 2, "the allocation and the growing reallocation");
            assert_eq!(thread_allocated_bytes() - bytes, 4000);
        }
    }

    #[test]
    fn a_large_allocation_shows_in_the_peaks() {
        assert!(is_tracking());
        let before = peak_bytes();
        // Varying contents, read back: a constant fill never read could be optimized away and never be resident.
        let big: Vec<u8> = (0..64usize << 20).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
        let checksum: u64 = std::hint::black_box(&big).iter().step_by(4096).map(|&b| b as u64).sum();
        assert!(checksum > 0);
        assert!(live_bytes() >= 64 << 20);
        assert!(peak_bytes() >= before.max(64 << 20));
        assert!(process_peak_bytes() >= peak_bytes());
        assert!(peak_rss_bytes() >= 48 << 20, "touched memory shows in the resident peak");
    }
}
