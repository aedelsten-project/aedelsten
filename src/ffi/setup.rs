// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Setup and lifecycle: shell pairs, primitive shell pairs, the scratch and its need, the communicator, the memory
// breakdown.

use super::*;

/// Derives the shell pairs (see `AedelstenContext::set_shell_pairs`), dropping primitive pairs whose overlap factor is
/// below `overlap_threshold` (negative: none dropped). Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_shell_pairs_c(
    ctx: *mut AedelstenContextHandle,
    overlap_threshold: f64,
) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    // Sentinel value: a negative threshold means no overlap screening (None), since we can't use Option<f64> in C.
    let overlap_threshold = if overlap_threshold < 0.0 {
        None
    } else {
        Some(overlap_threshold)
    };
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    match real.set_shell_pairs(overlap_threshold) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_shell_pairs (C wrapper {}): {}", stringify!(set_shell_pairs_c), e);
            FAILURE
        }
    }
}

/// Frees the shell pairs and the ERI data.
#[unsafe(no_mangle)]
pub extern "C" fn free_shell_pairs_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_shell_pairs();
}

/// Frees the parsed input (the basis, the nuclei, the embedding). Only after the last call that reads it (the
/// operators, the shell pairs, the primitive shell pairs, the embedding): such a call afterwards stops the program.
#[unsafe(no_mangle)]
pub extern "C" fn free_input_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_input();
}

/// Frees the primitive (decontracted) shells made by `set_primitive_shell_pairs_c`; the primitive shell pairs stay.
#[unsafe(no_mangle)]
pub extern "C" fn free_primitive_shells_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_primitive_shells();
}

/// Frees the contraction matrix made by `set_primitive_shell_pairs_c`.
#[unsafe(no_mangle)]
pub extern "C" fn free_contraction_matrix_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_contraction_matrix();
}

/// Decontracts the basis into its primitive shells, their pairs and the contraction matrix (see
/// `AedelstenContext::set_primitive_shell_pairs`); `overlap_threshold` as in `set_shell_pairs_c`. Returns 0, or 1 on
/// failure.
#[unsafe(no_mangle)]
pub extern "C" fn set_primitive_shell_pairs_c(
    ctx: *mut AedelstenContextHandle,
    overlap_threshold: f64,
) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    // Sentinel value: a negative threshold means no overlap screening (None), since we can't use Option<f64> in C.
    let overlap_threshold = if overlap_threshold < 0.0 {
        None
    } else {
        Some(overlap_threshold)
    };
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    match real.set_primitive_shell_pairs(overlap_threshold) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_primitive_shell_pairs (C wrapper {}): {}", stringify!(set_primitive_shell_pairs_c), e);
            FAILURE
        }
    }
}

/// Frees the primitive shell pairs.
#[unsafe(no_mangle)]
pub extern "C" fn free_primitive_shell_pairs_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_primitive_shell_pairs();
}

/// 1 when the primitive basis is set (always for a primitive context), 0 when not, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn has_primitive_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.has_primitive() as i32
}

/// See `AedelstenContext::set_scratch` on choosing the size: the library makes no judgment about it.
#[unsafe(no_mangle)]
pub extern "C" fn set_scratch_c(
    ctx: *mut AedelstenContextHandle,
    size: i32,
    unit: i32,
) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    // The caller gives the size: there is no default.
    if size < 0 {
        rank0_eprintln!("Negative scratch size {} in C wrapper {}: give the size", size, stringify!(set_scratch_c));
        return FAILURE;
    }
    let unit = match arrays::scratch::ByteUnit::from_c_int(unit) {
        Ok(u) => u,
        Err(_) => {
            rank0_eprintln!("Unknown byte unit {} in C wrapper {}", unit, stringify!(set_scratch_c));
            return FAILURE;
        }
    };
    let size = (size as usize, unit);
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    match real.set_scratch(size) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_scratch (C wrapper {}): {}", stringify!(set_scratch_c), e);
            FAILURE
        }
    }
}

/// Frees the scratch buffer.
#[unsafe(no_mangle)]
pub extern "C" fn free_scratch_c(ctx: *mut AedelstenContextHandle) {
    if ctx.is_null() {
        return;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    real.free_scratch();
}

/// Splits the context's parallel work over communicator `comm` from now on (as `set_aedelsten_context_c` names it;
/// see `AedelstenContext::set_communicator`). Returns 0, or 1 when `comm` is not a communicator (the context's is
/// then unchanged).
#[unsafe(no_mangle)]
pub extern "C" fn set_communicator_c(ctx: *mut AedelstenContextHandle, comm: i32) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    match super::communicator_of(comm) {
        Ok(comm) => {
            let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
            real.set_communicator(&comm);
            0
        }
        Err(e) => {
            rank0_eprintln!("Communicator {comm} in C wrapper {}: {e}", stringify!(set_communicator_c));
            FAILURE
        }
    }
}

/// Replaces the scratch buffer (see `AedelstenContext::resize_scratch`); `size` must not be negative.
#[unsafe(no_mangle)]
pub extern "C" fn resize_scratch_c(ctx: *mut AedelstenContextHandle, size: i32, unit: i32) -> i32 {
    if ctx.is_null() {
        return FAILURE;
    }
    if size < 0 {
        rank0_eprintln!("Negative scratch size {} in C wrapper {}: give the size", size, stringify!(resize_scratch_c));
        return FAILURE;
    }
    let size = match arrays::scratch::ByteUnit::from_c_int(unit) {
        Ok(u) => (size as usize, u),
        Err(_) => {
            rank0_eprintln!("Unknown byte unit {} in C wrapper {}", unit, stringify!(resize_scratch_c));
            return FAILURE;
        }
    };
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    match real.resize_scratch(size) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in resize_scratch (C wrapper {}): {}", stringify!(resize_scratch_c), e);
            FAILURE
        }
    }
}

/// The scratch's high-water mark in bytes (see `AedelstenContext::get_scratch_peak_bytes`), or -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_scratch_peak_bytes_c(ctx: *const AedelstenContextHandle) -> i64 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    match real.get_scratch_peak_bytes() {
        Ok(bytes) => bytes as i64,
        Err(e) => {
            rank0_eprintln!("Error in get_scratch_peak_bytes (C wrapper {}): {}", stringify!(get_scratch_peak_bytes_c), e);
            -1
        }
    }
}

/// The number of fields of `memory_breakdown_c`.
#[unsafe(no_mangle)]
pub extern "C" fn memory_breakdown_field_count_c() -> i32 {
    crate::MEMORY_BREAKDOWN_FIELDS.len() as i32
}

/// Field `index`'s name, a static NUL-terminated string not to be freed; null out of range.
#[unsafe(no_mangle)]
pub extern "C" fn memory_breakdown_field_name_c(index: i32) -> *const std::ffi::c_char {
    usize::try_from(index).ok().and_then(|i| crate::MEMORY_BREAKDOWN_FIELDS.get(i)).map_or(std::ptr::null(), |name| name.as_ptr())
}

/// The memory the context holds, field by field, in bytes (see `AedelstenContext::memory_breakdown`); `len` is
/// `memory_breakdown_field_count_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn memory_breakdown_c(ctx: *mut AedelstenContextHandle, output: *mut u64, len: i32) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let mut breakdown = vec![0usize; crate::MEMORY_BREAKDOWN_FIELDS.len()];
    if let Err(e) = real.memory_breakdown(&mut breakdown) {
        rank0_eprintln!("Error in memory_breakdown (C wrapper {}): {}", stringify!(memory_breakdown_c), e);
        return FAILURE;
    }
    if len < 0 || len as usize != breakdown.len() {
        rank0_eprintln!(
            "Error in memory_breakdown (C wrapper {}): expected a buffer of length {}, got {}",
            stringify!(memory_breakdown_c),
            breakdown.len(),
            len
        );
        return FAILURE;
    }
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
    for (dst, src) in output_slice.iter_mut().zip(breakdown.iter()) {
        *dst = *src as u64;
    }
    0
}

/// The operations of a C caller's rows (`SCRATCH_OPERATION_FIELDS` each), or `None` (reported) if one is invalid.
fn scratch_operations(operations: *const i32, n_operations: i32, wrapper: &str) -> Option<Vec<crate::ScratchOperation>> {
    let Ok(n) = usize::try_from(n_operations) else {
        rank0_eprintln!("Error in C wrapper {wrapper}: {n_operations} operations");
        return None;
    };
    if n == 0 {
        return Some(Vec::new());
    }
    if operations.is_null() {
        return None;
    }
    // SAFETY: the caller passes an int32[5 n_operations].
    let rows = unsafe { std::slice::from_raw_parts(operations, crate::SCRATCH_OPERATION_FIELDS * n) };
    rows.chunks_exact(crate::SCRATCH_OPERATION_FIELDS)
        .map(|row| {
            let row: [i32; crate::SCRATCH_OPERATION_FIELDS] = row.try_into().expect("a row");
            let operation = crate::ScratchOperation::from_c(row);
            if operation.is_none() {
                rank0_eprintln!("Error in C wrapper {wrapper}: no operation {row:?}");
            }
            operation
        })
        .collect()
}

/// The bytes of scratch every operation of `operations` fits in, from the input alone (see
/// `AedelstenContext::scratch_bytes`); -1 on failure. `operations` is `int32[5 n_operations]`, a row per operation
/// (column-major `(5, n)` in Fortran): the kind, then up to four parameters, -1 for none: 1 ERI data, 2 restricted
/// Fock, 3 unrestricted Fock, 4 two-electron gradient (terms), 5 general Fock, 6 general Fock of several densities
/// (densities), 7 active-space Fock (active), 8 mixed active-space Fock (active), 9 half transform (left, right, sigma,
/// node_ranks), 10 AMFI (over the primitive shells whatever `primitive` says). `primitive` nonzero: the same operations
/// on the primitive blocks.
#[unsafe(no_mangle)]
pub extern "C" fn scratch_bytes_c(ctx: *const AedelstenContextHandle, operations: *const i32, n_operations: i32, primitive: i32) -> i64 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let Some(operations) = scratch_operations(operations, n_operations, "scratch_bytes_c") else { return -1 };
    match real.scratch_bytes(&operations, primitive != 0) {
        Ok(bytes) => bytes as i64,
        Err(e) => {
            rank0_eprintln!("Error in scratch_bytes (C wrapper scratch_bytes_c): {e}");
            -1
        }
    }
}

/// The number of fields per operation of `scratch_breakdown_c`.
#[unsafe(no_mangle)]
pub extern "C" fn scratch_breakdown_field_count_c() -> i32 {
    crate::SCRATCH_BREAKDOWN_FIELDS.len() as i32
}

/// Field `index`'s name, a static NUL-terminated string not to be freed; null out of range.
#[unsafe(no_mangle)]
pub extern "C" fn scratch_breakdown_field_name_c(index: i32) -> *const std::ffi::c_char {
    usize::try_from(index).ok().and_then(|i| crate::SCRATCH_BREAKDOWN_FIELDS.get(i)).map_or(std::ptr::null(), |name| name.as_ptr())
}

/// `scratch_bytes_c` operation by operation into `output`, `int64[fields n_operations]`, a row of
/// `scratch_breakdown_field_count_c` fields per operation: bytes, buffers, tables, row, item_bytes, item (1 ERI data
/// entry, 2 gradient pair data, 3 derivative bound, 4 quartet, 5 AMFI atom kind), then the l, primitives and contracted
/// functions of the item's four shells (-1 for a pair's last two). `len` is the fields times `n_operations`. Returns 0,
/// or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn scratch_breakdown_c(
    ctx: *const AedelstenContextHandle,
    operations: *const i32,
    n_operations: i32,
    primitive: i32,
    output: *mut i64,
    len: i32,
) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let Some(operations) = scratch_operations(operations, n_operations, "scratch_breakdown_c") else { return FAILURE };
    let fields = crate::SCRATCH_BREAKDOWN_FIELDS.len();
    if usize::try_from(len) != Ok(fields * operations.len()) {
        rank0_eprintln!("Error in C wrapper scratch_breakdown_c: expected a buffer of length {}, got {len}", fields * operations.len());
        return FAILURE;
    }
    match real.scratch_breakdown(&operations, primitive != 0) {
        Ok(breakdown) => {
            // SAFETY: the caller passes an int64[len], checked above.
            let out = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
            for (row, need) in out.chunks_exact_mut(fields).zip(&breakdown) {
                row.copy_from_slice(&need.c_fields());
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in scratch_breakdown (C wrapper scratch_breakdown_c): {e}");
            FAILURE
        }
    }
}
