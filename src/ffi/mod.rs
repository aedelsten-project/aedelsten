// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C ABI. Every entry point is a `#[no_mangle] extern "C"` function over an opaque context handle, in the file of
// its context topic (src/context/), the foundation crates' in `runtime`; this file holds what they share: the handle,
// the failure code, error reporting, the named-constant queries, the context's creation and destruction, the
// buffer-overlap checks.

use crate::AedelstenContext;
use std::ffi::c_char;

use crate::context::one_electron::OneElectronOperatorKind;

/// The status an entry point returns on failure (0 is success).
pub(crate) const FAILURE: i32 = 1;

// How every entry point reports an error: on stderr, from world rank 0 only (every rank meets the same error).
macro_rules! rank0_eprintln {
    ($($arg:tt)*) => {
        if aedelsten_mpi::rank() == 0 {
            eprintln!($($arg)*);
        }
    };
}

/// The opaque handle of a context (`set_aedelsten_context_c`).
#[repr(C)]
pub struct AedelstenContextHandle {}

/// The communicator a C caller names by `handle`: this process alone for `mpi_comm_single` (no MPI needed), else the
/// MPI communicator with that Fortran handle or the world for `mpi_comm_world` (MPI must be up).
pub(crate) fn communicator_of(handle: i32) -> Result<aedelsten_mpi::Communicator, aedelsten_mpi::MpiError> {
    if handle == aedelsten_mpi::COMM_SINGLE_SENTINEL {
        return Ok(aedelsten_mpi::Communicator::single());
    }
    aedelsten_mpi::Session::attach(aedelsten_mpi::ThreadLevel::Single).and_then(|session| session.communicator(handle))
}

/// A context over the input file `filename_c` with shells of `contraction_type_c`, its parallel work split over the
/// ranks of communicator `comm` (a Fortran handle, `mpi_comm_world`, or `mpi_comm_single` for this process alone);
/// null on failure (the reason on stderr).
#[unsafe(no_mangle)]
pub extern "C" fn set_aedelsten_context_c(
    filename_c: *const std::ffi::c_char,
    contraction_type_c: i32,
    comm: i32,
) -> *mut AedelstenContextHandle {
    let filename = match unsafe { std::ffi::CStr::from_ptr(filename_c) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            rank0_eprintln!("Invalid (non-UTF8) filename in C wrapper {}", stringify!(set_aedelsten_context_c));
            return std::ptr::null_mut();
        }
    };

    let contraction_type = match basis::shell_types::ContractionType::from_c_int(contraction_type_c) {
        Ok(ct) => ct,
        Err(_) => {
            rank0_eprintln!("Unknown contraction type {} in C wrapper {}", contraction_type_c, stringify!(set_aedelsten_context_c));
            return std::ptr::null_mut();
        }
    };

    let comm = match communicator_of(comm) {
        Ok(comm) => comm,
        Err(e) => {
            rank0_eprintln!("Communicator {comm} in C wrapper {}: {e}", stringify!(set_aedelsten_context_c));
            return std::ptr::null_mut();
        }
    };

    let ctx = match AedelstenContext::new(filename, contraction_type, &comm) {
        Ok(ctx) => ctx,
        Err(e) => {
            rank0_eprintln!("Error in AedelstenContext::new (C wrapper {}): {}", stringify!(set_aedelsten_context_c), e);
            return std::ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(ctx)) as *mut AedelstenContextHandle
}

/// Frees a context made by `set_aedelsten_context_c`; null is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn free_aedelsten_context_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ctx as *mut AedelstenContext));
    }
}

fn lookup_named_i32_constant(key: &str) -> Option<i32> {
    const TABLES: &[&[(&str, i32)]] = &[
        basis::shell_types::ContractionType::NAME_TABLE,
        basis::AtomKind::NAME_TABLE,
        embedding::SiteSelectionKind::NAME_TABLE,
        embedding::MultipoleConventionKind::NAME_TABLE,
        arrays::scratch::ByteUnit::NAME_TABLE,
        crate::context::one_electron::OneElectronOperatorKind::NAME_TABLE,
        crate::context::two_electron::ERIRangeOperator::NAME_TABLE,
        crate::context::two_electron::ERIAOSymmetry::NAME_TABLE,
        crate::context::two_electron::ERIIndexOrder::NAME_TABLE,
        cart2sph::SharedCartesianReference::NAME_TABLE,
        aedelsten_mpi::ThreadLevel::NAME_TABLE,
    ];
    TABLES.iter().flat_map(|table| table.iter()).find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// The integer constant named `key` (case-insensitive): an enum's code by its wire name (`oneint_overlap`,
/// `eri_full_range`, ...), `default_contraction_type`, the communicator sentinels `mpi_comm_world` and
/// `mpi_comm_single`, or `mpi_enabled`; -1 for an unknown name.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_query_i32_c(key: *const std::ffi::c_char) -> i32 {
    let key = match unsafe { std::ffi::CStr::from_ptr(key) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let key = key.to_lowercase();
    match key.as_str() {
        // These describe library-computed defaults, not enum tags, so they
        // aren't in any NAME_TABLE.
        "default_contraction_type" => crate::context::default_contraction_type().get_c_int(),
        "mpi_comm_world"           => aedelsten_mpi::COMM_WORLD_SENTINEL,
        "mpi_comm_single"          => aedelsten_mpi::COMM_SINGLE_SENTINEL,
        "mpi_enabled"              => cfg!(feature = "mpi") as i32,
        other => lookup_named_i32_constant(other).unwrap_or(-1),
    }
}

/// The floating-point constant named `key` (case-insensitive): `default_symmetry_geometry_tolerance` or
/// `default_symmetry_strict_tolerance`; NaN for an unknown name.
#[unsafe(no_mangle)]
pub extern "C" fn aedelsten_query_f64_c(key: *const std::ffi::c_char) -> f64 {
    let key = match unsafe { std::ffi::CStr::from_ptr(key) }.to_str() {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    match key.to_lowercase().as_str() {
        "default_symmetry_geometry_tolerance" => symmetry::DEFAULT_GEOMETRY_TOLERANCE,
        "default_symmetry_strict_tolerance" => crate::DEFAULT_STRICT_TOLERANCE,
        _ => f64::NAN,
    }
}

/// Whether an output buffer overlaps any other buffer of the call (each `len` doubles). The direct builds zero
/// their outputs before reading the inputs, so an output sharing memory with an input would be wrong, and a Rust
/// `&mut` slice may not alias another slice: such a call is rejected rather than copied around.
fn outputs_overlap(outputs: &[*const f64], inputs: &[*const f64], len: usize) -> bool {
    let range = |p: *const f64| (p as usize, p as usize + len * std::mem::size_of::<f64>());
    outputs.iter().enumerate().any(|(o, &out)| {
        let (start, end) = range(out);
        outputs.iter().enumerate().filter(|&(other, _)| other != o).map(|(_, &p)| p).chain(inputs.iter().copied()).any(|p| {
            let (other_start, other_end) = range(p);
            len > 0 && start < other_end && other_start < end
        })
    })
}

/// `outputs_overlap` for buffers of different lengths, `(pointer, doubles)` each; an empty buffer overlaps nothing.
fn buffers_overlap(outputs: &[(*const f64, usize)], inputs: &[(*const f64, usize)]) -> bool {
    let range = |(p, len): (*const f64, usize)| (p as usize, p as usize + len * std::mem::size_of::<f64>());
    outputs.iter().enumerate().any(|(o, &out)| {
        let (start, end) = range(out);
        outputs.iter().enumerate().filter(|&(other, _)| other != o).map(|(_, &b)| b).chain(inputs.iter().copied()).any(|b| {
            let (other_start, other_end) = range(b);
            start < end && other_start < other_end && start < other_end && other_start < end
        })
    })
}

/// Writes `text` NUL-terminated into `buf` (capacity `buf_len`, NUL included); fails rather than truncates.
pub(crate) fn write_string(text: &str, buf: *mut c_char, buf_len: i32, wrapper: &str) -> i32 {
    if buf.is_null() || buf_len < 0 || text.len() >= buf_len as usize {
        rank0_eprintln!("Error in C wrapper {wrapper}: buffer of length {buf_len} too small for \"{text}\" ({} bytes + NUL)", text.len());
        return FAILURE;
    }
    // SAFETY: buf has buf_len bytes, more than text needs.
    let out = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, buf_len as usize) };
    out[..text.len()].copy_from_slice(text.as_bytes());
    out[text.len()] = 0;
    0
}

mod setup;
mod basis_set;
mod model;
mod one_electron;
mod two_electron;
mod gradients;
mod embedding;
mod relativity;
mod point_group;
mod runtime;
