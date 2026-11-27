// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The MPI layer: a session ([`Session`]: MPI initialized by this program or by its host, at a checked thread
//! level), communicators ([`Communicator`]: the collectives over `f64` buffers, a pairwise exchange, sub-groups,
//! shared-memory windows, arrays and locks in a node's shared memory, and how much of it was mapped at most:
//! [`shared_memory_peak`]), the time spent in them ([`collective_seconds`], and as spans: [`COLLECTIVE_SPAN`]), and the
//! cost-balanced partition of work over ranks.
//!
//! MPI is compiled in only with the `mpi` feature, which needs an `mpicc`-compatible compiler wrapper on `PATH` (or
//! named by `MPICC`) at build time. Without it every function works as the one-rank case.
//!
//! The free functions [`rank`], [`size`], [`barrier`], [`bcast_f64`], [`allreduce_sum_f64`] and
//! [`partition_for_this_rank`] act on `MPI_COMM_WORLD` (rank 0 of 1 while MPI is not up).

use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

mod communicator;
mod shared;
pub use communicator::{Communicator, Counter, SharedWindow};
pub use shared::{SharedArray, SharedLockGuard, SharedLocks};

/// The value standing for `MPI_COMM_WORLD` where a communicator crosses the C ABI as its Fortran handle.
pub const COMM_WORLD_SENTINEL: i32 = -2;

/// The value standing for this process alone ([`Communicator::single`], no MPI needed) where a communicator crosses
/// the C ABI as its Fortran handle.
pub const COMM_SINGLE_SENTINEL: i32 = -3;

/// Nanoseconds this process has spent inside collectives (and pairwise exchanges): the communication itself plus
/// the wait for the slowest rank.
static COLLECTIVE_NANOS: AtomicU64 = AtomicU64::new(0);

/// Seconds this process has spent in collectives and pairwise exchanges so far: the communication and the wait for
/// the other ranks. A caller's phase timing minus this is the rank's own work. 0 without the `mpi` feature.
pub fn collective_seconds() -> f64 {
    COLLECTIVE_NANOS.load(Ordering::Relaxed) as f64 * 1e-9
}

/// Bytes of node-shared memory this process has mapped now, and at most so far: every segment of every live
/// [`SharedWindow`] (all ranks' segments, since each is mapped into every rank's process).
static SHARED_BYTES: AtomicUsize = AtomicUsize::new(0);
static SHARED_PEAK: AtomicUsize = AtomicUsize::new(0);

/// The most node-shared memory this process has had mapped at once so far, in bytes: the shared-memory windows
/// (and so the [`SharedArray`]s and [`SharedLocks`]) of its communicators, all ranks' segments. The node holds these
/// once, but the operating system counts their touched pages in the resident size of every rank that touches them.
/// 0 without the `mpi` feature or on one rank (a shared array is then local memory, on the heap).
pub fn shared_memory_peak() -> usize {
    SHARED_PEAK.load(Ordering::Relaxed)
}

#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
fn shared_mapped(bytes: usize) {
    let now = SHARED_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    SHARED_PEAK.fetch_max(now, Ordering::Relaxed);
}

fn shared_unmapped(bytes: usize) {
    SHARED_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// The label of the span (`instrumentation::span`) each collective and pairwise exchange opens, under the span open
/// on the calling thread: a span tree's nodes of this label are the time in MPI below their parent.
pub const COLLECTIVE_SPAN: &str = "MPI collectives";

#[cfg(feature = "mpi")]
fn timed_collective<T>(f: impl FnOnce() -> T) -> T {
    let _span = instrumentation::span(COLLECTIVE_SPAN);
    let start = std::time::Instant::now();
    let result = f();
    COLLECTIVE_NANOS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}

/// The C shim's functions (`src/shim.c`): each returns 0 on success and -1 on failure, unless it returns a value.
#[cfg(feature = "mpi")]
mod raw {
    use std::ffi::{c_long, c_void};

    unsafe extern "C" {
        pub fn aedelsten_mpi_is_up() -> i32;
        pub fn aedelsten_mpi_was_initialized() -> i32;
        pub fn aedelsten_mpi_init_thread(required: i32, provided: *mut i32) -> i32;
        pub fn aedelsten_mpi_query_thread(provided: *mut i32) -> i32;
        pub fn aedelsten_mpi_init() -> i32;
        pub fn aedelsten_mpi_finalize() -> i32;
        pub fn aedelsten_mpi_comm_world() -> i32;
        pub fn aedelsten_mpi_comm_null() -> i32;
        pub fn aedelsten_mpi_comm_rank(comm: i32, rank: *mut i32) -> i32;
        pub fn aedelsten_mpi_comm_size(comm: i32, size: *mut i32) -> i32;
        pub fn aedelsten_mpi_comm_split(comm: i32, color: i32, key: i32, split: *mut i32) -> i32;
        pub fn aedelsten_mpi_comm_free(comm: i32) -> i32;
        pub fn aedelsten_mpi_barrier(comm: i32) -> i32;
        pub fn aedelsten_mpi_bcast_f64(comm: i32, buf: *mut f64, n: i32, root: i32) -> i32;
        pub fn aedelsten_mpi_allreduce_sum_f64(comm: i32, buf: *mut f64, n: i32) -> i32;
        pub fn aedelsten_mpi_gather_f64(comm: i32, send: *const f64, n: i32, recv: *mut f64, root: i32) -> i32;
        pub fn aedelsten_mpi_allgather_f64(comm: i32, send: *const f64, n: i32, recv: *mut f64) -> i32;
        pub fn aedelsten_mpi_alltoallv_f64(
            comm: i32,
            send: *const f64,
            send_counts: *const i32,
            send_displs: *const i32,
            recv: *mut f64,
            recv_counts: *const i32,
            recv_displs: *const i32,
        ) -> i32;
        pub fn aedelsten_mpi_sendrecv_f64(
            comm: i32,
            send: *const f64,
            n_send: i32,
            to: i32,
            recv: *mut f64,
            n_recv: i32,
            from: i32,
        ) -> i32;
        pub fn aedelsten_mpi_comm_split_node(comm: i32, key: i32, node: *mut i32) -> i32;
        pub fn aedelsten_mpi_shared_window_allocate(comm: i32, n: c_long, mine: *mut *mut f64, handle: *mut *mut c_void) -> i32;
        pub fn aedelsten_mpi_shared_window_query(handle: *mut c_void, rank: i32, base: *mut *mut f64, n: *mut c_long) -> i32;
        pub fn aedelsten_mpi_shared_window_sync(handle: *mut c_void) -> i32;
        pub fn aedelsten_mpi_shared_window_free(handle: *mut c_void) -> i32;
        pub fn aedelsten_mpi_counter_allocate(comm: i32, handle: *mut *mut c_void) -> i32;
        pub fn aedelsten_mpi_counter_next(handle: *mut c_void, value: *mut c_long) -> i32;
        pub fn aedelsten_mpi_counter_free(handle: *mut c_void) -> i32;
    }
}

/// Why an MPI operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MpiError {
    /// An MPI call returned a failure.
    Call { call: &'static str, code: i32 },
    /// MPI is not initialized (or already finalized).
    NotInitialized,
    /// MPI is initialized already, by someone else: [`Session::attach`] to it instead.
    AlreadyInitialized,
    /// MPI runs at a lower thread level than required.
    ThreadLevel { required: ThreadLevel, provided: ThreadLevel },
    /// An argument the operation cannot take.
    Argument(String),
}

impl fmt::Display for MpiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MpiError::Call { call, code } => write!(f, "{call} failed with status code {code}"),
            MpiError::NotInitialized => write!(f, "MPI is not initialized: the program initializes it"),
            MpiError::AlreadyInitialized => write!(f, "MPI is initialized already: attach to it instead"),
            MpiError::ThreadLevel { required, provided } => {
                write!(f, "MPI runs at thread level {provided}, {required} is required")
            }
            MpiError::Argument(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for MpiError {}

#[cfg(feature = "mpi")]
fn check(call: &'static str, code: i32) -> Result<(), MpiError> {
    if code == 0 { Ok(()) } else { Err(MpiError::Call { call, code }) }
}

/// A count as MPI's `int`.
#[cfg(feature = "mpi")]
fn mpi_count(n: usize) -> Result<i32, MpiError> {
    i32::try_from(n).map_err(|_| MpiError::Argument(format!("{n} values are too many for one MPI call")))
}

/// The largest count one `MPI_Allreduce` sums (2^27 doubles, 1 GB).
#[cfg(feature = "mpi")]
const ALLREDUCE_CHUNK: usize = 1 << 27;

/// MPI's thread support levels, in increasing order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum ThreadLevel {
    /// One thread.
    Single = 0,
    /// Several threads; only the main thread calls MPI.
    Funneled = 1,
    /// Several threads call MPI, one at a time.
    Serialized = 2,
    /// Several threads call MPI at once.
    Multiple = 3,
}

impl ThreadLevel {
    /// The level from its number (0-3, as in the C ABI).
    pub fn from_i32(level: i32) -> Option<ThreadLevel> {
        match level {
            0 => Some(ThreadLevel::Single),
            1 => Some(ThreadLevel::Funneled),
            2 => Some(ThreadLevel::Serialized),
            3 => Some(ThreadLevel::Multiple),
            _ => None,
        }
    }

    /// The level's name and number for the C ABI's named-constant query.
    pub const NAME_TABLE: &'static [(&'static str, i32)] = &[
        ("mpi_thread_single", 0),
        ("mpi_thread_funneled", 1),
        ("mpi_thread_serialized", 2),
        ("mpi_thread_multiple", 3),
    ];
}

impl fmt::Display for ThreadLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ThreadLevel::Single => "MPI_THREAD_SINGLE",
            ThreadLevel::Funneled => "MPI_THREAD_FUNNELED",
            ThreadLevel::Serialized => "MPI_THREAD_SERIALIZED",
            ThreadLevel::Multiple => "MPI_THREAD_MULTIPLE",
        })
    }
}

/// MPI, up at a known thread level: the only source of MPI communicators. Made by the program that initializes MPI
/// ([`Session::init`], which finalizes it when dropped), or over MPI its host initialized ([`Session::attach`], which
/// never finalizes). Without the `mpi` feature every constructor succeeds, at [`ThreadLevel::Multiple`], and the
/// world is [`Communicator::single`].
#[derive(Debug)]
pub struct Session {
    provided: ThreadLevel,
    owner: bool,
}

impl Session {
    /// Initializes MPI at thread level `required` at least, and finalizes it when dropped. Refused when MPI was
    /// initialized already, or provides less than `required` (MPI is then finalized again).
    pub fn init(required: ThreadLevel) -> Result<Session, MpiError> {
        #[cfg(feature = "mpi")]
        {
            if unsafe { raw::aedelsten_mpi_was_initialized() } == 1 {
                return Err(MpiError::AlreadyInitialized);
            }
            let mut provided = 0;
            check("MPI_Init_thread", unsafe { raw::aedelsten_mpi_init_thread(required as i32, &mut provided) })?;
            let session = Session { provided: ThreadLevel::from_i32(provided).unwrap_or(ThreadLevel::Single), owner: true };
            session.require(required)?;
            Ok(session)
        }
        #[cfg(not(feature = "mpi"))]
        {
            let _ = required;
            Ok(Session { provided: ThreadLevel::Multiple, owner: true })
        }
    }

    /// MPI as the host initialized it, refused when it is not up or runs below thread level `required`. Never
    /// finalizes.
    pub fn attach(required: ThreadLevel) -> Result<Session, MpiError> {
        #[cfg(feature = "mpi")]
        {
            if unsafe { raw::aedelsten_mpi_is_up() } != 1 {
                return Err(MpiError::NotInitialized);
            }
            let mut provided = 0;
            check("MPI_Query_thread", unsafe { raw::aedelsten_mpi_query_thread(&mut provided) })?;
            let session = Session { provided: ThreadLevel::from_i32(provided).unwrap_or(ThreadLevel::Single), owner: false };
            session.require(required)?;
            Ok(session)
        }
        #[cfg(not(feature = "mpi"))]
        {
            let _ = required;
            Ok(Session { provided: ThreadLevel::Multiple, owner: false })
        }
    }

    /// MPI at thread level `required` at least, initialized now unless it is up already, and finalized when the
    /// process exits: for test binaries, whose tests share one process. Never finalizes when dropped.
    pub fn init_until_exit(required: ThreadLevel) -> Result<Session, MpiError> {
        #[cfg(feature = "mpi")]
        {
            static AT_EXIT: std::sync::Once = std::sync::Once::new();
            static INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _guard = INIT.lock().unwrap_or_else(|e| e.into_inner());
            if unsafe { raw::aedelsten_mpi_was_initialized() } == 0 {
                let mut provided = 0;
                check("MPI_Init_thread", unsafe { raw::aedelsten_mpi_init_thread(required as i32, &mut provided) })?;
                AT_EXIT.call_once(|| {
                    unsafe extern "C" {
                        fn atexit(f: extern "C" fn()) -> i32;
                    }
                    extern "C" fn finalize() {
                        unsafe { raw::aedelsten_mpi_finalize() };
                    }
                    unsafe { atexit(finalize) };
                });
            }
        }
        Session::attach(required)
    }

    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    fn require(&self, required: ThreadLevel) -> Result<(), MpiError> {
        if self.provided < required {
            Err(MpiError::ThreadLevel { required, provided: self.provided })
        } else {
            Ok(())
        }
    }

    /// The thread level MPI runs at.
    pub fn thread_level(&self) -> ThreadLevel {
        self.provided
    }

    /// Whether this session finalizes MPI when dropped.
    pub fn is_owner(&self) -> bool {
        self.owner
    }

    /// All ranks (`MPI_COMM_WORLD`).
    pub fn world(&self) -> Result<Communicator, MpiError> {
        #[cfg(feature = "mpi")]
        {
            self.communicator(unsafe { raw::aedelsten_mpi_comm_world() })
        }
        #[cfg(not(feature = "mpi"))]
        Ok(Communicator::single())
    }

    /// The communicator with MPI Fortran handle `handle` (`MPI_Comm_c2f`; as mpi4py's `Comm.py2f()` and Fortran's
    /// `MPI_Comm%MPI_VAL` give it), or the world for [`COMM_WORLD_SENTINEL`]. Not freed when dropped: it stays its
    /// owner's.
    pub fn communicator(&self, handle: i32) -> Result<Communicator, MpiError> {
        #[cfg(feature = "mpi")]
        {
            if unsafe { raw::aedelsten_mpi_is_up() } != 1 {
                return Err(MpiError::NotInitialized);
            }
            let handle = if handle == COMM_WORLD_SENTINEL { unsafe { raw::aedelsten_mpi_comm_world() } } else { handle };
            if handle == unsafe { raw::aedelsten_mpi_comm_null() } {
                return Err(MpiError::Argument("MPI_COMM_NULL is not a communicator to work on".into()));
            }
            Communicator::from_handle(handle, false)
        }
        #[cfg(not(feature = "mpi"))]
        {
            let _ = handle;
            Ok(Communicator::single())
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        #[cfg(feature = "mpi")]
        if self.owner {
            unsafe { raw::aedelsten_mpi_finalize() };
        }
    }
}

/// Whether MPI is initialized and not yet finalized; `false` without the `mpi` feature.
pub fn is_initialized() -> bool {
    #[cfg(feature = "mpi")]
    {
        unsafe { raw::aedelsten_mpi_is_up() == 1 }
    }
    #[cfg(not(feature = "mpi"))]
    false
}

/// The world communicator while MPI is up, the one-rank group otherwise.
fn world() -> Communicator {
    #[cfg(feature = "mpi")]
    if is_initialized() {
        if let Ok(comm) = Communicator::from_handle(unsafe { raw::aedelsten_mpi_comm_world() }, false) {
            return comm;
        }
    }
    Communicator::single()
}

/// This rank's index in `MPI_COMM_WORLD`; 0 while MPI is not up.
pub fn rank() -> usize {
    world().rank()
}

/// The number of ranks in `MPI_COMM_WORLD`; 1 while MPI is not up.
pub fn size() -> usize {
    world().size()
}

/// Whether `MPI_COMM_WORLD` has more than one rank.
pub fn is_multi_process() -> bool {
    size() > 1
}

/// Returns once every rank of `MPI_COMM_WORLD` has called it.
pub fn barrier() {
    let _ = world().barrier();
}

/// `buf` from `root` on every rank of `MPI_COMM_WORLD`, in place.
pub fn bcast_f64(buf: &mut [f64], root: usize) -> Result<(), MpiError> {
    world().broadcast(buf, root)
}

/// `buf` summed over the ranks of `MPI_COMM_WORLD`, in place, on every rank.
pub fn allreduce_sum_f64(buf: &mut [f64]) -> Result<(), MpiError> {
    world().sum(buf)
}

/// Item indices `0..costs.len()` in `n_ranks` buckets, each item added to the least-loaded bucket so far (the
/// longest-processing-time heuristic, whose bound needs `costs` sorted descending).
pub fn greedy_partition(costs: &[usize], n_ranks: usize) -> Vec<Vec<usize>> {
    assert!(n_ranks > 0, "greedy_partition: n_ranks must be at least 1");
    let mut buckets = vec![Vec::new(); n_ranks];
    let mut bucket_load = vec![0usize; n_ranks];
    for (idx, &cost) in costs.iter().enumerate() {
        let target = bucket_load
            .iter()
            .enumerate()
            .min_by_key(|&(_, load)| load)
            .map(|(i, _)| i)
            .expect("n_ranks > 0");
        buckets[target].push(idx);
        bucket_load[target] += cost;
    }
    buckets
}

/// A counter shared by the ranks of `MPI_COMM_WORLD`: [`Communicator::counter`]. Collective.
pub fn counter() -> Result<Counter, MpiError> {
    world().counter()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The only test that maps or unmaps (the tally is the process's): the peak follows the most mapped at once, and
    /// an unmapping leaves it.
    #[test]
    fn the_shared_memory_peak_is_the_most_mapped_at_once() {
        let before = shared_memory_peak();
        shared_mapped(before + 100);
        shared_mapped(50);
        shared_unmapped(before + 150);
        shared_mapped(20);
        assert_eq!(shared_memory_peak(), before + 150);
        shared_unmapped(20);
    }

    #[test]
    fn serial_defaults_without_mpi_up() {
        assert_eq!(rank(), 0);
        assert_eq!(size(), 1);
        assert!(!is_multi_process());
    }

    #[test]
    fn collectives_change_nothing_on_one_rank() {
        let mut buf = vec![1.0, 2.0, 3.0];
        allreduce_sum_f64(&mut buf).unwrap();
        assert_eq!(buf, vec![1.0, 2.0, 3.0]);
        bcast_f64(&mut buf, 0).unwrap();
        assert_eq!(buf, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn thread_levels_are_ordered_and_round_trip() {
        assert!(ThreadLevel::Single < ThreadLevel::Funneled && ThreadLevel::Serialized < ThreadLevel::Multiple);
        for &(_, level) in ThreadLevel::NAME_TABLE {
            assert_eq!(ThreadLevel::from_i32(level).map(|l| l as i32), Some(level));
        }
        assert_eq!(ThreadLevel::from_i32(4), None);
    }

    #[cfg(not(feature = "mpi"))]
    #[test]
    fn without_mpi_a_session_is_the_one_rank_group() {
        let session = Session::attach(ThreadLevel::Multiple).unwrap();
        let world = session.world().unwrap();
        assert_eq!((world.rank(), world.size()), (0, 1));
        assert_eq!(session.communicator(COMM_WORLD_SENTINEL).unwrap().size(), 1);
    }

    #[test]
    fn greedy_partition_covers_every_index_exactly_once() {
        let costs = vec![10, 9, 8, 1, 1, 1];
        let buckets = greedy_partition(&costs, 3);
        assert_eq!(buckets.len(), 3);
        let mut all: Vec<usize> = buckets.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn greedy_partition_balances_load() {
        let costs = vec![10, 9, 8, 1, 1, 1];
        let buckets = greedy_partition(&costs, 3);
        let loads: Vec<usize> = buckets.iter().map(|b| b.iter().map(|&i| costs[i]).sum()).collect();
        assert!(loads.iter().max().unwrap() - loads.iter().min().unwrap() <= 1);
    }

    #[test]
    fn greedy_partition_single_rank_gets_everything() {
        let buckets = greedy_partition(&[5, 4, 3], 1);
        assert_eq!(buckets[0], vec![0, 1, 2]);
    }
}
