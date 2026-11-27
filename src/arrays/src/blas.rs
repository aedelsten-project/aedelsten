// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The BLAS's thread count, set by the program rather than left to the environment: which variable a BLAS reads
//! depends on how it was built, so an unset or wrong one can run every rank on every core.
//!
//! The library is the one the build links: OpenBLAS (`openblas`), Intel MKL (`mkl`), Accelerate (the default on
//! macOS), or on other systems the system BLAS, recognized at run time as OpenBLAS, MKL, BLIS or Arm Performance
//! Libraries. Any other BLAS is treated as single-threaded.
//!
//! The count is the calling process's. Set it from the main thread only, outside any threaded region (as under
//! `MPI_THREAD_FUNNELED`). Every count set is read back: a library that runs another count than the one asked for
//! (an OpenBLAS clamping to its compiled maximum, a sequential build) is refused with its reason, never ignored.
//! Accelerate's threading is on or off, not a count: one thread turns it off, more turn it on, and Accelerate
//! chooses how many it runs.

use std::ffi::c_void;
use std::fmt;
use std::sync::OnceLock;

/// A BLAS library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Library {
    OpenBlas,
    Mkl,
    Blis,
    ArmPl,
    Accelerate,
    Unknown,
}

impl fmt::Display for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Library::OpenBlas => "OpenBLAS",
            Library::Mkl => "Intel MKL",
            Library::Blis => "BLIS",
            Library::ArmPl => "Arm Performance Libraries",
            Library::Accelerate => "Accelerate",
            Library::Unknown => "an unrecognized BLAS",
        })
    }
}

/// How a BLAS runs its threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threading {
    /// One thread, always.
    Sequential,
    /// Its own threads.
    Pthreads,
    /// OpenMP's threads.
    OpenMp,
    /// On or off; the library chooses how many threads when on (Accelerate).
    OnOff,
    /// Not known.
    Unknown,
}

/// The BLAS this build runs, for a run's header (`Display`: e.g. "OpenBLAS (OpenMP build), 1 thread").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Info {
    /// The library.
    pub library: Library,
    /// How it runs its threads.
    pub threading: Threading,
    /// Threads it runs now (with [`Threading::OnOff`] and on: the cores available).
    pub threads: usize,
    /// Whether [`set_threads`] can change it.
    pub settable: bool,
}

impl fmt::Display for Info {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plural = if self.threads == 1 { "" } else { "s" };
        match (self.library, self.threading) {
            (Library::Unknown, _) => write!(f, "{} (treated as single-threaded)", self.library),
            (_, Threading::OnOff) if self.threads == 1 => write!(f, "{} (single-threaded)", self.library),
            (_, Threading::OnOff) => write!(f, "{} (multi-threaded, the count its own)", self.library),
            (_, Threading::Sequential) => write!(f, "{} (sequential build), 1 thread", self.library),
            (_, Threading::Pthreads) => write!(f, "{} (pthreads build), {} thread{plural}", self.library, self.threads),
            (_, Threading::OpenMp) => write!(f, "{} (OpenMP build), {} thread{plural}", self.library, self.threads),
            (_, Threading::Unknown) => write!(f, "{}, {} thread{plural}", self.library, self.threads),
        }
    }
}

/// A thread count the BLAS cannot run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadCountError {
    /// The library.
    pub library: Library,
    /// The count asked for.
    pub requested: usize,
    /// Why the library cannot run it.
    pub reason: String,
}

impl fmt::Display for ThreadCountError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: cannot run {} BLAS thread(s): {}", self.library, self.requested, self.reason)
    }
}

impl std::error::Error for ThreadCountError {}

/// The BLAS this build runs and its thread count now.
pub fn info() -> Info {
    let backend = backend();
    Info { library: backend.library(), threading: backend.threading(), threads: backend.threads(), settable: backend.settable() }
}

/// Sets the BLAS's thread count to `k` and returns the previous count. Refused, with the count unchanged, when the
/// library cannot run `k` threads.
pub fn set_threads(k: usize) -> Result<usize, ThreadCountError> {
    let backend = backend();
    let refuse = |reason: String| ThreadCountError { library: backend.library(), requested: k, reason };
    if k == 0 {
        return Err(refuse("at least one thread is needed".into()));
    }
    let previous = backend.threads();
    if !backend.settable() {
        return if k == 1 {
            Ok(previous)
        } else {
            Err(refuse(match backend {
                Backend::Unknown(reason) => format!("{reason}; treated as single-threaded"),
                _ => "a sequential build runs one thread".into(),
            }))
        };
    }
    if backend.threading() == Threading::OnOff {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        if k > cores {
            return Err(refuse(format!("it runs at most the {cores} cores")));
        }
    }
    backend.set(k);
    let now = backend.threads();
    let runs_k = match backend.threading() {
        Threading::OnOff => (now == 1) == (k == 1),
        _ => now == k,
    };
    if runs_k {
        Ok(previous)
    } else {
        backend.set(previous);
        Err(refuse(format!("it runs {now} when asked for {k}")))
    }
}

/// Runs `f` with the BLAS at `k` threads, and restores the previous count when `f` returns (or panics).
pub fn with_threads<T>(k: usize, f: impl FnOnce() -> T) -> Result<T, ThreadCountError> {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = set_threads(self.0);
        }
    }
    let _restore = Restore(set_threads(k)?);
    Ok(f())
}

/// Where [`available_cores`] counted the cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreSource {
    /// The physical cores in the process's affinity mask (Linux; hyperthreads of one core counted once).
    AffinityPhysical,
    /// The CPUs in the process's affinity mask, the topology not readable (Linux).
    AffinityLogical,
    /// The node's performance cores (macOS: `hw.perflevel0.physicalcpu`).
    PerformanceCores,
    /// The node's physical cores (macOS without performance levels: `hw.physicalcpu`).
    PhysicalCores,
    /// The logical CPUs the standard library reports.
    LogicalCpus,
}

impl fmt::Display for CoreSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CoreSource::AffinityPhysical => "physical cores in the affinity mask",
            CoreSource::AffinityLogical => "CPUs in the affinity mask",
            CoreSource::PerformanceCores => "performance cores",
            CoreSource::PhysicalCores => "physical cores",
            CoreSource::LogicalCpus => "logical CPUs",
        })
    }
}

/// The cores this process may run on, for choosing its BLAS thread count (`Display`: e.g. "4 performance cores, node
/// wide").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cores {
    /// The number of cores.
    pub count: usize,
    /// Whether a launcher bound the process to a subset of the node's CPUs (its affinity mask is smaller than the
    /// online CPUs). When not, `count` is the node's, shared by the ranks on it.
    pub bound: bool,
    /// Where they were counted.
    pub source: CoreSource,
}

impl fmt::Display for Cores {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}, {}", self.count, self.source, if self.bound { "bound" } else { "node wide" })
    }
}

/// The cores available to this process: on Linux those of its affinity mask, on macOS (which cannot bind) the node's
/// performance cores, elsewhere the logical CPUs.
pub fn available_cores() -> Cores {
    #[cfg(target_os = "linux")]
    if let Some(cores) = linux_affinity_cores() {
        return cores;
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(count) = sysctl_count("hw.perflevel0.physicalcpu") {
            return Cores { count, bound: false, source: CoreSource::PerformanceCores };
        }
        if let Some(count) = sysctl_count("hw.physicalcpu") {
            return Cores { count, bound: false, source: CoreSource::PhysicalCores };
        }
    }
    let count = std::thread::available_parallelism().map_or(1, |n| n.get());
    Cores { count, bound: false, source: CoreSource::LogicalCpus }
}

#[cfg(target_os = "macos")]
fn sysctl_count(name: &str) -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(name: *const std::ffi::c_char, old: *mut c_void, old_len: *mut usize, new: *mut c_void, new_len: usize) -> i32;
    }
    let name = std::ffi::CString::new(name).ok()?;
    let mut value: i32 = 0;
    let mut len = size_of::<i32>();
    let rc = unsafe { sysctlbyname(name.as_ptr(), (&raw mut value).cast(), &mut len, std::ptr::null_mut(), 0) };
    (rc == 0 && len == size_of::<i32>() && value > 0).then_some(value as usize)
}

#[cfg(target_os = "linux")]
fn linux_affinity_cores() -> Option<Cores> {
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, set_size: usize, mask: *mut u64) -> i32;
    }
    // A cpu_set_t of 1024 CPUs.
    let mut mask = [0u64; 16];
    if unsafe { sched_getaffinity(0, size_of_val(&mask), mask.as_mut_ptr()) } != 0 {
        return None;
    }
    let cpus: Vec<usize> = (0..64 * mask.len()).filter(|&cpu| mask[cpu / 64] >> (cpu % 64) & 1 == 1).collect();
    if cpus.is_empty() {
        return None;
    }
    let online = std::fs::read_to_string("/sys/devices/system/cpu/online").ok().and_then(|list| count_cpu_list(list.trim()));
    let bound = online.is_some_and(|online| cpus.len() < online);
    // One core per distinct sibling list: its hyperthreads share it.
    let siblings: Option<std::collections::BTreeSet<String>> = cpus
        .iter()
        .map(|cpu| {
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"))
                .ok()
                .map(|list| list.trim().to_string())
        })
        .collect();
    Some(match siblings {
        Some(cores) => Cores { count: cores.len(), bound, source: CoreSource::AffinityPhysical },
        None => Cores { count: cpus.len(), bound, source: CoreSource::AffinityLogical },
    })
}

/// The CPUs in a list such as `0-3,8,10-11`.
#[cfg(any(target_os = "linux", test))]
fn count_cpu_list(list: &str) -> Option<usize> {
    list.split(',').filter(|part| !part.is_empty()).try_fold(0, |total, part| {
        let count = match part.split_once('-') {
            Some((first, last)) => last.parse::<usize>().ok()?.checked_sub(first.parse::<usize>().ok()?)? + 1,
            None => {
                part.parse::<usize>().ok()?;
                1
            }
        };
        Some(total + count)
    })
}

// Each build constructs the variants of the libraries it can link.
#[allow(dead_code)]
enum Backend {
    OpenBlas { set: unsafe extern "C" fn(i32), get: unsafe extern "C" fn() -> i32, parallel: i32 },
    Mkl { set: unsafe extern "C" fn(i32), get: unsafe extern "C" fn() -> i32 },
    Blis { set: unsafe extern "C" fn(i64), get: unsafe extern "C" fn() -> i64 },
    ArmPl { set: unsafe extern "C" fn(i32), get: unsafe extern "C" fn() -> i32 },
    Accelerate { set: unsafe extern "C" fn(u32) -> i32, get: unsafe extern "C" fn() -> u32 },
    Unknown(&'static str),
}

// Accelerate's BLAS_THREADING values.
const ACCELERATE_MULTI_THREADED: u32 = 0;
const ACCELERATE_SINGLE_THREADED: u32 = 1;

impl Backend {
    fn library(&self) -> Library {
        match self {
            Backend::OpenBlas { .. } => Library::OpenBlas,
            Backend::Mkl { .. } => Library::Mkl,
            Backend::Blis { .. } => Library::Blis,
            Backend::ArmPl { .. } => Library::ArmPl,
            Backend::Accelerate { .. } => Library::Accelerate,
            Backend::Unknown(_) => Library::Unknown,
        }
    }

    fn threading(&self) -> Threading {
        match self {
            Backend::OpenBlas { parallel: 0, .. } => Threading::Sequential,
            Backend::OpenBlas { parallel: 1, .. } => Threading::Pthreads,
            Backend::OpenBlas { parallel: 2, .. } | Backend::ArmPl { .. } => Threading::OpenMp,
            Backend::Accelerate { .. } => Threading::OnOff,
            Backend::Unknown(_) => Threading::Sequential,
            _ => Threading::Unknown,
        }
    }

    fn settable(&self) -> bool {
        !matches!(self.threading(), Threading::Sequential)
    }

    fn threads(&self) -> usize {
        let count = |n: i64| n.max(1) as usize;
        unsafe {
            match self {
                Backend::OpenBlas { get, .. } | Backend::Mkl { get, .. } | Backend::ArmPl { get, .. } => count(get() as i64),
                Backend::Blis { get, .. } => count(get()),
                Backend::Accelerate { get, .. } => {
                    if get() == ACCELERATE_SINGLE_THREADED {
                        1
                    } else {
                        std::thread::available_parallelism().map_or(1, |n| n.get())
                    }
                }
                Backend::Unknown(_) => 1,
            }
        }
    }

    fn set(&self, k: usize) {
        let k32 = i32::try_from(k).unwrap_or(i32::MAX);
        unsafe {
            match self {
                Backend::OpenBlas { set, .. } | Backend::Mkl { set, .. } | Backend::ArmPl { set, .. } => set(k32),
                Backend::Blis { set, .. } => set(k as i64),
                Backend::Accelerate { set, .. } => {
                    set(if k == 1 { ACCELERATE_SINGLE_THREADED } else { ACCELERATE_MULTI_THREADED });
                }
                Backend::Unknown(_) => {}
            }
        }
    }
}

fn backend() -> &'static Backend {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    BACKEND.get_or_init(resolve)
}

#[cfg(feature = "openblas")]
fn resolve() -> Backend {
    unsafe extern "C" {
        fn openblas_set_num_threads(n: i32);
        fn openblas_get_num_threads() -> i32;
        fn openblas_get_parallel() -> i32;
    }
    Backend::OpenBlas { set: openblas_set_num_threads, get: openblas_get_num_threads, parallel: unsafe { openblas_get_parallel() } }
}

#[cfg(all(feature = "mkl", not(feature = "openblas")))]
fn resolve() -> Backend {
    unsafe extern "C" {
        fn MKL_Set_Num_Threads(n: i32);
        fn MKL_Get_Max_Threads() -> i32;
    }
    Backend::Mkl { set: MKL_Set_Num_Threads, get: MKL_Get_Max_Threads }
}

#[cfg(all(target_os = "macos", not(any(feature = "openblas", feature = "mkl"))))]
fn resolve() -> Backend {
    match (symbol("BLASSetThreading"), symbol("BLASGetThreading")) {
        // SAFETY: Accelerate's `int BLASSetThreading(enum BLAS_THREADING)` and `enum BLAS_THREADING
        // BLASGetThreading(void)`, the enum an unsigned int.
        (Some(set), Some(get)) => unsafe { Backend::Accelerate { set: std::mem::transmute(set), get: std::mem::transmute(get) } },
        _ => Backend::Unknown("Accelerate without thread control (before macOS 15)"),
    }
}

#[cfg(not(any(target_os = "macos", feature = "openblas", feature = "mkl")))]
fn resolve() -> Backend {
    detect()
}

/// The BLAS among the libraries loaded into the process, by the thread-control functions it exports.
#[allow(dead_code)]
fn detect() -> Backend {
    // SAFETY: each pair is the library's documented setter and getter, of the signatures transmuted to.
    unsafe {
        if let (Some(set), Some(get), Some(parallel)) =
            (symbol("openblas_set_num_threads"), symbol("openblas_get_num_threads"), symbol("openblas_get_parallel"))
        {
            let parallel: unsafe extern "C" fn() -> i32 = std::mem::transmute(parallel);
            return Backend::OpenBlas { set: std::mem::transmute(set), get: std::mem::transmute(get), parallel: parallel() };
        }
        if let (Some(set), Some(get)) = (symbol("MKL_Set_Num_Threads"), symbol("MKL_Get_Max_Threads")) {
            return Backend::Mkl { set: std::mem::transmute(set), get: std::mem::transmute(get) };
        }
        if let (Some(set), Some(get)) = (symbol("bli_thread_set_num_threads"), symbol("bli_thread_get_num_threads")) {
            return Backend::Blis { set: std::mem::transmute(set), get: std::mem::transmute(get) };
        }
        if let (Some(_), Some(set), Some(get)) =
            (symbol("armpl_version"), symbol("omp_set_num_threads"), symbol("omp_get_max_threads"))
        {
            return Backend::ArmPl { set: std::mem::transmute(set), get: std::mem::transmute(get) };
        }
    }
    Backend::Unknown("not OpenBLAS, MKL, BLIS or Arm Performance Libraries")
}

/// The address of `name` among the symbols loaded into the process.
#[allow(dead_code)]
fn symbol(name: &str) -> Option<*mut c_void> {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
        }
        #[cfg(target_os = "macos")]
        const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
        #[cfg(not(target_os = "macos"))]
        const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();
        let name = std::ffi::CString::new(name).ok()?;
        let address = unsafe { dlsym(RTLD_DEFAULT, name.as_ptr()) };
        (!address.is_null()).then_some(address)
    }
    #[cfg(not(unix))]
    {
        let _ = name;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test: the count is the process's, and the test harness runs tests in parallel.
    #[test]
    fn the_count_is_set_read_back_restored_and_refused_beyond_the_library() {
        let start = info();
        assert!(start.threads >= 1);
        assert!(!start.to_string().is_empty());

        assert!(set_threads(0).is_err());
        let previous = set_threads(1).unwrap();
        assert_eq!(info().threads, 1);
        assert!(info().to_string().contains(&start.library.to_string()) || start.library == Library::Unknown);

        if info().settable {
            let inside = with_threads(2, || info().threads).unwrap();
            assert!(inside >= 2);
            assert_eq!(info().threads, 1, "restored after with_threads");
            let refused = set_threads(1 << 20).unwrap_err();
            assert_eq!(refused.library, start.library);
            assert_eq!(info().threads, 1, "unchanged after a refusal");
        } else {
            assert!(set_threads(2).is_err());
        }
        set_threads(previous).unwrap();
        assert_eq!(info().threads, start.threads);
    }

    #[test]
    fn the_available_cores_are_counted_and_described() {
        let cores = available_cores();
        assert!(cores.count >= 1);
        assert!(cores.count <= std::thread::available_parallelism().map_or(1, |n| n.get()));
        #[cfg(target_os = "macos")]
        assert!(!cores.bound && matches!(cores.source, CoreSource::PerformanceCores | CoreSource::PhysicalCores));
        #[cfg(target_os = "linux")]
        assert!(matches!(cores.source, CoreSource::AffinityPhysical | CoreSource::AffinityLogical));
        assert!(cores.to_string().starts_with(&cores.count.to_string()));
    }

    #[test]
    fn a_cpu_list_is_counted() {
        assert_eq!(count_cpu_list("0-3,8,10-11"), Some(7));
        assert_eq!(count_cpu_list("0"), Some(1));
        assert_eq!(count_cpu_list("3-1"), None);
        assert_eq!(count_cpu_list("x"), None);
    }

    #[cfg(feature = "openblas")]
    #[test]
    fn a_loaded_openblas_is_recognized_by_its_symbols() {
        let found = detect();
        assert_eq!(found.library(), Library::OpenBlas);
        assert_eq!(found.threading(), backend().threading());
        assert_eq!(found.threads(), backend().threads());
    }
}
