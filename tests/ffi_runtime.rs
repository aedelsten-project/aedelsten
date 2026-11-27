// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the foundation crates, driven as a C, Fortran, Python or Julia caller would, against the
// Rust calls they wrap. Without the `mpi` feature (MPI on several ranks: examples/{python,julia,fortran}/mpi_demo).

// Links the library, whose C entry points are declared below.
extern crate aedelsten;

use std::ffi::{CString, c_char};

unsafe extern "C" {
    fn aedelsten_query_i32_c(key: *const c_char) -> i32;
    fn aedelsten_mpi_init_c(required_level: i32) -> i32;
    fn aedelsten_mpi_finalize_c() -> i32;
    fn aedelsten_mpi_initialized_c() -> i32;
    fn aedelsten_mpi_thread_level_c() -> i32;
    fn aedelsten_mpi_rank_c(comm: i32, rank: *mut i32) -> i32;
    fn aedelsten_mpi_size_c(comm: i32, size: *mut i32) -> i32;
    fn aedelsten_blas_set_threads_c(k: i32) -> i32;
    fn aedelsten_blas_threads_c() -> i32;
    fn aedelsten_blas_info_c(buf: *mut c_char, buf_len: i32) -> i32;
    fn aedelsten_get_version_c(buf: *mut c_char, buf_len: i32) -> i32;
    fn aedelsten_get_source_commit_c(buf: *mut c_char, buf_len: i32) -> i32;
    fn aedelsten_blas_available_cores_c(count: *mut i32, bound: *mut i32) -> i32;
    fn aedelsten_blas_available_cores_info_c(buf: *mut c_char, buf_len: i32) -> i32;
}

fn query(key: &str) -> i32 {
    let key = CString::new(key).unwrap();
    unsafe { aedelsten_query_i32_c(key.as_ptr()) }
}

/// A string entry point's answer, `None` when it refuses a buffer of `len` bytes.
fn string_from(entry: unsafe extern "C" fn(*mut c_char, i32) -> i32, len: usize) -> Option<String> {
    let mut buf = vec![0 as c_char; len];
    (unsafe { entry(buf.as_mut_ptr(), len as i32) } == 0).then(|| unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap().to_string())
}

/// The version is the crate's, and a buffer too small is refused; the commit, in a git checkout, is 12 hex digits.
#[test]
fn the_version_and_the_source_commit_are_the_rust_calls_answers() {
    assert_eq!(string_from(aedelsten_get_version_c, 64).as_deref(), Some(aedelsten::get_version()));
    assert_eq!(aedelsten::get_version(), env!("CARGO_PKG_VERSION"));
    assert_eq!(string_from(aedelsten_get_version_c, aedelsten::get_version().len()), None, "no room for the NUL");
    let commit = string_from(aedelsten_get_source_commit_c, 64).unwrap();
    assert_eq!(commit, aedelsten::get_source_commit().unwrap_or(""));
    if let Some(commit) = aedelsten::get_source_commit() {
        assert!(commit.len() == 12 && commit.chars().all(|c| c.is_ascii_hexdigit()), "{commit}");
    }
}

/// Every crate of the workspace takes the workspace's version, so that a consumer's requirement holds for each.
#[test]
fn every_crate_shares_the_workspace_version() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let members: Vec<&str> = workspace.split("members = [").nth(1).unwrap().split(']').next().unwrap().split(',').map(|m| m.trim().trim_matches('"')).filter(|m| !m.is_empty()).collect();
    assert!(members.len() > 10, "{members:?}");
    for member in members.iter().chain(&["."]) {
        let manifest = std::fs::read_to_string(root.join(member).join("Cargo.toml")).unwrap();
        assert!(manifest.contains("\nversion.workspace = true\n"), "{member} has a version of its own");
    }
}

#[test]
fn the_mpi_constants_are_queryable() {
    for (name, level) in aedelsten_mpi::ThreadLevel::NAME_TABLE {
        assert_eq!(query(name), *level, "{name}");
    }
    assert_eq!(query("mpi_comm_world"), aedelsten_mpi::COMM_WORLD_SENTINEL);
    assert_ne!(query("mpi_comm_world"), -1, "the sentinel is not the unknown-key answer");
    assert_eq!(query("mpi_enabled"), cfg!(feature = "mpi") as i32);
}

#[cfg(not(feature = "mpi"))]
#[test]
fn without_mpi_the_c_session_is_one_rank_at_any_thread_level() {
    let world = query("mpi_comm_world");
    unsafe {
        assert_eq!(aedelsten_mpi_init_c(query("mpi_thread_funneled")), 3);
        assert_eq!(aedelsten_mpi_init_c(9), -1, "an unknown level is refused");
        assert_eq!(aedelsten_mpi_initialized_c(), 0);
        assert_eq!(aedelsten_mpi_thread_level_c(), 3);
        let (mut rank, mut size) = (-1, -1);
        assert_eq!(aedelsten_mpi_rank_c(world, &mut rank), 0);
        assert_eq!(aedelsten_mpi_size_c(world, &mut size), 0);
        assert_eq!((rank, size), (0, 1));
        assert_ne!(aedelsten_mpi_rank_c(world, std::ptr::null_mut()), 0, "a null output is refused");
        assert_eq!(aedelsten_mpi_finalize_c(), 0);
    }
}

#[test]
fn the_blas_thread_count_is_set_and_described_as_in_rust() {
    let info = aedelsten::arrays::blas::info();
    let mut buf = [0 as c_char; 256];
    unsafe {
        assert_eq!(aedelsten_blas_info_c(buf.as_mut_ptr(), buf.len() as i32), 0);
        assert_ne!(aedelsten_blas_info_c(buf.as_mut_ptr(), 4), 0, "too small a buffer is refused");
        assert_eq!(aedelsten_blas_info_c(buf.as_mut_ptr(), buf.len() as i32), 0);
        let text = std::ffi::CStr::from_ptr(buf.as_ptr()).to_str().unwrap();
        assert_eq!(text, info.to_string());

        assert_eq!(aedelsten_blas_threads_c(), info.threads as i32);
        assert_eq!(aedelsten_blas_set_threads_c(-1), -1);
        assert_eq!(aedelsten_blas_set_threads_c(0), -1);
        let previous = aedelsten_blas_set_threads_c(1);
        assert_eq!(previous, info.threads as i32);
        assert_eq!(aedelsten_blas_threads_c(), 1);
        assert_eq!(aedelsten_blas_set_threads_c(1 << 20), -1, "more threads than any BLAS runs");
        assert_eq!(aedelsten_blas_threads_c(), 1);
        assert_eq!(aedelsten_blas_set_threads_c(previous), 1);
    }
}

#[test]
fn the_available_cores_are_as_in_rust() {
    let cores = aedelsten::arrays::blas::available_cores();
    let (mut count, mut bound) = (0, -1);
    let mut buf = [0 as c_char; 256];
    unsafe {
        assert_eq!(aedelsten_blas_available_cores_c(&mut count, &mut bound), 0);
        assert_ne!(aedelsten_blas_available_cores_c(std::ptr::null_mut(), &mut bound), 0);
        assert_ne!(aedelsten_blas_available_cores_info_c(buf.as_mut_ptr(), 4), 0, "too small a buffer is refused");
        assert_eq!(aedelsten_blas_available_cores_info_c(buf.as_mut_ptr(), buf.len() as i32), 0);
        let text = std::ffi::CStr::from_ptr(buf.as_ptr()).to_str().unwrap();
        assert_eq!(text, cores.to_string());
    }
    assert_eq!((count, bound), (cores.count as i32, cores.bound as i32));
}
