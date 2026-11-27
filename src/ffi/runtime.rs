// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C ABI of the foundation crates: MPI for a host without MPI of its own (init, finalize, the thread level,
// rank and size of a communicator given by its Fortran handle or the world sentinel `mpi_comm_world`), arrays in a
// node's shared memory, and the BLAS's thread count.

use std::ffi::c_char;
use std::sync::Mutex;

use aedelsten_mpi::{Session, ThreadLevel};

use super::{write_string, FAILURE};

/// The session `aedelsten_mpi_init_c` opened, finalized by `aedelsten_mpi_finalize_c`.
static C_SESSION: Mutex<Option<Session>> = Mutex::new(None);

/// Initializes MPI at thread level `required_level` (0 single, 1 funneled, 2 serialized, 3 multiple; also the
/// named constants `mpi_thread_*`) and returns the level provided, or -1 on failure: MPI initialized by someone
/// else (use it directly), or a level below the one required. A second call returns the level of the first.
/// Without the `mpi` build option it returns 3 and initializes nothing.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_init_c(required_level: i32) -> i32 {
    let Some(required) = ThreadLevel::from_i32(required_level) else {
        eprintln!("Unknown MPI thread level {required_level} in C wrapper {}", stringify!(aedelsten_mpi_init_c));
        return -1;
    };
    let mut session = C_SESSION.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(open) = session.as_ref() {
        return open.thread_level() as i32;
    }
    match Session::init(required) {
        Ok(opened) => {
            let level = opened.thread_level() as i32;
            *session = Some(opened);
            level
        }
        Err(e) => {
            eprintln!("Error in Session::init (C wrapper {}): {e}", stringify!(aedelsten_mpi_init_c));
            -1
        }
    }
}

/// Finalizes MPI if `aedelsten_mpi_init_c` initialized it; MPI initialized by the host is left alone. Every rank
/// calls it, after its last MPI use. Returns 0.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_finalize_c() -> i32 {
    drop(C_SESSION.lock().unwrap_or_else(|e| e.into_inner()).take());
    0
}

/// 1 while MPI is initialized and not finalized, else 0 (always 0 without the `mpi` build option).
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_initialized_c() -> i32 {
    aedelsten_mpi::is_initialized() as i32
}

/// The thread level MPI runs at (0-3), or -1 while it is not up (3 without the `mpi` build option).
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_thread_level_c() -> i32 {
    Session::attach(ThreadLevel::Single).map_or(-1, |session| session.thread_level() as i32)
}

/// This rank's index in communicator `comm` (its MPI Fortran handle, or the sentinel `mpi_comm_world`) into `rank`.
/// Returns 0, or 1 when MPI is not up or `comm` is not a communicator.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_rank_c(comm: i32, rank: *mut i32) -> i32 {
    communicator_query(comm, rank, stringify!(aedelsten_mpi_rank_c), |c| c.rank())
}

/// The number of ranks in communicator `comm` (as `aedelsten_mpi_rank_c`) into `size`. Returns 0 or 1.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_size_c(comm: i32, size: *mut i32) -> i32 {
    communicator_query(comm, size, stringify!(aedelsten_mpi_size_c), |c| c.size())
}

fn communicator_query(
    comm: i32,
    out: *mut i32,
    wrapper: &str,
    query: impl Fn(&aedelsten_mpi::Communicator) -> usize,
) -> i32 {
    if out.is_null() {
        return FAILURE;
    }
    match Session::attach(ThreadLevel::Single).and_then(|session| session.communicator(comm)) {
        Ok(communicator) => {
            unsafe { *out = query(&communicator) as i32 };
            0
        }
        Err(e) => {
            eprintln!("Error in C wrapper {wrapper}: {e}");
            FAILURE
        }
    }
}

/// Sets the BLAS's thread count to `k` and returns the previous count, or -1 when the BLAS cannot run `k` threads
/// (the count is then unchanged; the reason on stderr). From the main thread only, outside any threaded region.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_blas_set_threads_c(k: i32) -> i32 {
    let Ok(k) = usize::try_from(k) else {
        rank0_eprintln!("Negative BLAS thread count {k} in C wrapper {}", stringify!(aedelsten_blas_set_threads_c));
        return -1;
    };
    match arrays::blas::set_threads(k) {
        Ok(previous) => previous as i32,
        Err(e) => {
            rank0_eprintln!("Error in C wrapper {}: {e}", stringify!(aedelsten_blas_set_threads_c));
            -1
        }
    }
}

/// The library's version (`crate::get_version`, e.g. "0.5.0"), NUL-terminated into `buf` (capacity `buf_len`). Returns
/// 0, or 1 when it does not fit.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_get_version_c(buf: *mut c_char, buf_len: i32) -> i32 {
    write_string(crate::get_version(), buf, buf_len, stringify!(aedelsten_get_version_c))
}

/// The git commit the library was built from (`crate::get_source_commit`, 12 hex digits), the empty string when its
/// source tree was no git checkout, NUL-terminated into `buf` (capacity `buf_len`). Returns 0, or 1 when it does not
/// fit.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_get_source_commit_c(buf: *mut c_char, buf_len: i32) -> i32 {
    write_string(crate::get_source_commit().unwrap_or(""), buf, buf_len, stringify!(aedelsten_get_source_commit_c))
}

/// The BLAS's thread count now.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_blas_threads_c() -> i32 {
    arrays::blas::info().threads as i32
}

/// The BLAS, its threading and its thread count, as one line for a run's header (e.g. "OpenBLAS (OpenMP build),
/// 1 thread"), NUL-terminated into `buf` (capacity `buf_len`). Returns 0, or 1 when it does not fit.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_blas_info_c(buf: *mut c_char, buf_len: i32) -> i32 {
    write_string(&arrays::blas::info().to_string(), buf, buf_len, stringify!(aedelsten_blas_info_c))
}

/// The cores available to this process (see `arrays::blas::available_cores`) into `count`, and into `bound` 1 when a
/// launcher bound the process to them, 0 when they are the node's. Returns 0, or 1 on a null pointer.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_blas_available_cores_c(count: *mut i32, bound: *mut i32) -> i32 {
    if count.is_null() || bound.is_null() {
        return FAILURE;
    }
    let cores = arrays::blas::available_cores();
    unsafe {
        *count = cores.count as i32;
        *bound = cores.bound as i32;
    }
    0
}

/// The cores available to this process and where they were counted, as one line for a run's header (e.g. "4
/// performance cores, node wide"), NUL-terminated into `buf` (capacity `buf_len`). Returns 0, or 1 when it does not
/// fit.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_blas_available_cores_info_c(buf: *mut c_char, buf_len: i32) -> i32 {
    write_string(&arrays::blas::available_cores().to_string(), buf, buf_len, stringify!(aedelsten_blas_available_cores_info_c))
}

/// An array of `len` doubles in the shared memory of the ranks of communicator `comm` (as `set_aedelsten_context_c`
/// names it), which must share one node (`mpi_comm_single`: this process's own memory): its first double into
/// `data`, the same memory on every rank. Returns an opaque handle, null on failure (the reason on stderr). Collective;
/// its contents are unspecified until written; freed by `aedelsten_mpi_shared_array_free_c`.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_shared_array_c(comm: i32, len: i64, data: *mut *mut f64) -> *mut std::ffi::c_void {
    if data.is_null() || len < 0 {
        return std::ptr::null_mut();
    }
    match super::communicator_of(comm).and_then(|comm| comm.shared_array(len as usize)) {
        Ok(mut array) => {
            unsafe { *data = array.as_mut_slice().as_mut_ptr() };
            Box::into_raw(Box::new(array)) as *mut std::ffi::c_void
        }
        Err(e) => {
            eprintln!("Error in C wrapper {}: {e}", stringify!(aedelsten_mpi_shared_array_c));
            std::ptr::null_mut()
        }
    }
}

/// Every rank's writes to the shared array `handle` made visible to all, and every rank waited for. Collective.
/// Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_shared_array_sync_c(handle: *mut std::ffi::c_void) -> i32 {
    if handle.is_null() {
        return FAILURE;
    }
    let array = unsafe { &*(handle as *const aedelsten_mpi::SharedArray) };
    match array.sync() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error in C wrapper {}: {e}", stringify!(aedelsten_mpi_shared_array_sync_c));
            FAILURE
        }
    }
}

/// Frees the shared array `handle` (from `aedelsten_mpi_shared_array_c`). Collective; null is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_shared_array_free_c(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle as *mut aedelsten_mpi::SharedArray) });
    }
}

// The timings (`crate::timing`) and the reference list (`instrumentation::citations`), as user output: a count, then
// one entry by its index (from 0), each read anew, so that an index stays valid as entries are added after it.

/// Seconds this process has spent in MPI collectives since it started (`crate::mpi_collective_seconds`).
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_mpi_collective_seconds_c() -> f64 {
    crate::mpi_collective_seconds()
}

/// 1 when the library was built with the `profile` feature (the fine timings are recorded), else 0.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_profile_enabled_c() -> i32 {
    crate::profile_enabled() as i32
}

fn timing_spans(fine: i32) -> Vec<crate::TimingSpan> {
    if fine != 0 { crate::profile_report() } else { crate::timing_report() }
}

/// The number of labels in the coarse timing report (`fine` 0, `crate::timing_report`) or the fine one (`fine`
/// nonzero, `crate::profile_report`, empty without the `profile` feature), this thread's.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_timing_count_c(fine: i32) -> i32 {
    timing_spans(fine).len() as i32
}

/// Entry `index` of the report `fine` chooses (as `aedelsten_timing_count_c`): its label NUL-terminated into `label`
/// (capacity `label_len`), its seconds and its calls. Returns 0, or 1 on an index out of range, a null pointer or a
/// label that does not fit.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_timing_span_c(fine: i32, index: i32, label: *mut c_char, label_len: i32, seconds: *mut f64, calls: *mut i64) -> i32 {
    let spans = timing_spans(fine);
    let Some(span) = usize::try_from(index).ok().and_then(|i| spans.get(i)) else {
        rank0_eprintln!("Error in C wrapper {}: index {index} out of range ({} entries)", stringify!(aedelsten_timing_span_c), spans.len());
        return FAILURE;
    };
    if seconds.is_null() || calls.is_null() {
        return FAILURE;
    }
    if write_string(span.get_label(), label, label_len, stringify!(aedelsten_timing_span_c)) != 0 {
        return FAILURE;
    }
    unsafe {
        *seconds = span.get_seconds();
        *calls = span.get_calls() as i64;
    }
    0
}

/// The number of nodes in this thread's span tree (coarse and fine spans, depth first).
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_span_tree_count_c() -> i32 {
    instrumentation::tree().nodes().len() as i32
}

/// Node `index` of this thread's span tree, depth first (each node followed by its children): its label
/// NUL-terminated into `label` (capacity `label_len`), its depth (0 at the top), whether it is fine (1) or coarse (0),
/// its seconds (children included) and its calls. Returns 0, or 1 on an index out of range, a null pointer or a label
/// that does not fit.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_span_tree_node_c(
    index: i32,
    label: *mut c_char,
    label_len: i32,
    depth: *mut i32,
    fine: *mut i32,
    seconds: *mut f64,
    calls: *mut i64,
) -> i32 {
    let tree = instrumentation::tree();
    let Some(node) = usize::try_from(index).ok().and_then(|i| tree.nodes().get(i)) else {
        rank0_eprintln!("Error in C wrapper {}: index {index} out of range ({} nodes)", stringify!(aedelsten_span_tree_node_c), tree.nodes().len());
        return FAILURE;
    };
    if depth.is_null() || fine.is_null() || seconds.is_null() || calls.is_null() {
        return FAILURE;
    }
    if write_string(node.label, label, label_len, stringify!(aedelsten_span_tree_node_c)) != 0 {
        return FAILURE;
    }
    unsafe {
        *depth = node.depth as i32;
        *fine = node.fine as i32;
        *seconds = node.total.as_secs_f64();
        *calls = node.calls as i64;
    }
    0
}

/// The number of references cited so far in this process, by the library and by any code citing through the same
/// registry, in the order first cited.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_citation_count_c() -> i32 {
    instrumentation::citations::citations().len() as i32
}

// Reference `index`'s text (`topics` false: the formatted reference) or topics (true: joined by ", ").
fn citation_part(index: i32, topics: bool) -> Option<String> {
    let records = instrumentation::citations::citations();
    let record = usize::try_from(index).ok().and_then(|i| records.get(i))?;
    Some(if topics { record.topics.join(", ") } else { record.reference.to_string() })
}

/// The length in bytes (NUL not included) of reference `index`'s text (`topics` 0) or of its topics (`topics`
/// nonzero), for sizing the buffer of `aedelsten_citation_c`; -1 on an index out of range.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_citation_length_c(index: i32, topics: i32) -> i64 {
    citation_part(index, topics != 0).map_or(-1, |s| s.len() as i64)
}

/// Reference `index` (as `aedelsten_citation_count_c`), NUL-terminated into `buf` (capacity `buf_len`): its text
/// (`topics` 0: authors, title, journal, volume, pages, year, DOI) or the topics it was cited for (`topics` nonzero,
/// joined by ", "). Returns 0, or 1 on an index out of range or a buffer too small.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_citation_c(index: i32, topics: i32, buf: *mut c_char, buf_len: i32) -> i32 {
    match citation_part(index, topics != 0) {
        Some(text) => write_string(&text, buf, buf_len, stringify!(aedelsten_citation_c)),
        None => {
            rank0_eprintln!("Error in C wrapper {}: index {index} out of range", stringify!(aedelsten_citation_c));
            FAILURE
        }
    }
}

/// The length in bytes (NUL not included) of the references cited so far as a BibTeX file.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_citations_bibtex_length_c() -> i64 {
    instrumentation::citations::bibtex(&instrumentation::citations::citations()).len() as i64
}

/// The references cited so far as a BibTeX file (one entry each, the topics as keywords), NUL-terminated into `buf`
/// (capacity `buf_len`). Returns 0, or 1 when it does not fit.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_citations_bibtex_c(buf: *mut c_char, buf_len: i32) -> i32 {
    let text = instrumentation::citations::bibtex(&instrumentation::citations::citations());
    write_string(&text, buf, buf_len, stringify!(aedelsten_citations_bibtex_c))
}

/// Forgets every reference cited so far (a host running several independent jobs in one process).
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_clear_citations_c() {
    instrumentation::citations::clear_citations();
}
