// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The iterative solver allocates nothing: with its scratch prepared, a whole solve -- every iteration,
// the coupling's transposes included -- performs zero heap allocations. Checked with a counting global
// allocator (this file is its own test binary, so the allocator affects nothing else); the count is
// per-thread and only active around the solve.

use arrays::matrix::{Matrix, MatrixTraitMut};
use polarization::{Damping, InitialGuess, IterativeBuffers, PolarizableSystem, SolverOptions, Whole};
use sources::{Fragments, PolarizableSite, Polarizability};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn record() {
    if COUNTING.with(|counting| counting.get()) {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations_during<R>(work: impl FnOnce() -> R) -> (R, usize) {
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|counting| counting.set(true));
    let result = work();
    COUNTING.with(|counting| counting.set(false));
    (result, ALLOCATIONS.with(|count| count.get()))
}

fn cluster() -> Vec<PolarizableSite> {
    // 24 sites on a slightly perturbed grid, in fragments of four, mixed isotropic/anisotropic.
    (0..24)
        .map(|s| {
            let (x, y, z) = ((s % 3) as f64, ((s / 3) % 2) as f64, (s / 6) as f64);
            let jitter = 0.1 * ((s * 7 % 5) as f64 - 2.0);
            let alpha = if s % 2 == 0 {
                Polarizability::Isotropic(1.0 + 0.1 * (s % 4) as f64)
            } else {
                Polarizability::Anisotropic([1.4, 0.1, 0.05, 1.1, 0.02, 0.9])
            };
            PolarizableSite::new([3.0 * x + jitter, 3.0 * y - jitter, 3.0 * z + 0.5 * jitter], alpha, Fragments::new(&[1 + (s / 4) as i32]))
        })
        .collect()
}

#[test]
fn a_prepared_iterative_solve_allocates_nothing() {
    let sites = cluster();
    let n = sites.len();
    let system = PolarizableSystem::new(&sites, Damping::thole()).unwrap();
    let options = SolverOptions::default();
    let mut buffers = IterativeBuffers::new(n);
    let mut scratch = buffers.scratch();
    let mut field = Matrix::<f64>::new(3, n);
    for (i, value) in field.get_data_mut().iter_mut().enumerate() {
        *value = 0.01 * ((i * 13 % 17) as f64 - 8.0);
    }
    let mut mu = Matrix::<f64>::new(3, n);

    let (report, allocations) = allocations_during(|| system.solve_iterative(&field, &mut mu, InitialGuess::Static, &options, &mut scratch, &Whole));
    let report = report.unwrap();
    assert!(report.iterations >= 3, "the test should take several conjugate-gradient steps (took {} iterations)", report.iterations);
    assert_eq!(allocations, 0, "the solve allocated {allocations} times");

    // And again, warm-started, with the same scratch: still nothing.
    let (again, allocations) = allocations_during(|| system.solve_iterative(&field, &mut mu, InitialGuess::Provided, &options, &mut scratch, &Whole));
    again.unwrap();
    assert_eq!(allocations, 0);
}

#[test]
fn the_counting_allocator_does_count() {
    // Guards the test above against a counter that silently never fires.
    let (_, allocations) = allocations_during(|| std::hint::black_box(vec![0.0_f64; 100]));
    assert!(allocations >= 1);
}
