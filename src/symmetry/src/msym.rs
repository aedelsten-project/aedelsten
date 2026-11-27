// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Raw libmsym bindings, the declarations of `src/libmsym/src/msym.h` that this crate uses (named and laid out as
//! there), and an owning context handle. Crate-private: `lib.rs` is the safe interface.
#![allow(non_camel_case_types)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_void};

pub(crate) type msym_error_t = c_int;
pub(crate) const MSYM_SUCCESS: msym_error_t = 0;

#[repr(C)]
pub(crate) struct msym_context__ {
    _unused: [u8; 0],
}
pub(crate) type msym_context = *mut msym_context__;

pub(crate) type msym_point_group_type_t = c_int;
pub(crate) type msym_basis_type = c_int;
pub(crate) const MSYM_BASIS_TYPE_REAL_SPHERICAL_HARMONIC: msym_basis_type = 0;

#[repr(C)]
pub(crate) struct msym_symmetry_operation_t {
    pub(crate) type_: c_int,
    pub(crate) order: c_int,
    pub(crate) power: c_int,
    pub(crate) orientation: c_int,
    pub(crate) v: [c_double; 3],
    pub(crate) cla: c_int,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct msym_element_t {
    pub(crate) id: *mut c_void,
    pub(crate) m: c_double,
    pub(crate) v: [c_double; 3],
    pub(crate) n: c_int,
    pub(crate) name: [c_char; 4],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct msym_real_spherical_harmonic_t {
    pub(crate) n: c_int,
    pub(crate) l: c_int,
    pub(crate) m: c_int,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) union msym_basis_function_f {
    pub(crate) rsh: msym_real_spherical_harmonic_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct msym_basis_function_t {
    pub(crate) id: *mut c_void,
    pub(crate) type_: msym_basis_type,
    pub(crate) element: *mut msym_element_t,
    pub(crate) f: msym_basis_function_f,
    pub(crate) name: [c_char; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct msym_partner_function_t {
    // SALC index of partner 0, and this function's component (dimension).
    pub(crate) i: c_int,
    pub(crate) d: c_int,
}

#[repr(C)]
pub(crate) struct msym_symmetry_species_t {
    pub(crate) d: c_int,
    pub(crate) r: c_int,
    pub(crate) name: [c_char; 8],
}

#[repr(C)]
pub(crate) struct msym_character_table_t {
    pub(crate) d: c_int,
    pub(crate) classc: *mut c_int,
    pub(crate) sops: *mut *mut msym_symmetry_operation_t,
    pub(crate) s: *mut msym_symmetry_species_t,
    pub(crate) table: *mut c_void, // double[d][d], row = species, column = class
}

#[repr(C)]
pub(crate) struct msym_subgroup_t {
    pub(crate) type_: msym_point_group_type_t,
    pub(crate) n: c_int,
    pub(crate) order: c_int,
    pub(crate) primary: *mut msym_symmetry_operation_t,
    pub(crate) sops: *mut *mut msym_symmetry_operation_t,
    pub(crate) generators: [*mut msym_subgroup_t; 2],
    pub(crate) name: [c_char; 8],
}

unsafe extern "C" {
    fn msymErrorString(error: msym_error_t) -> *const c_char;
    fn msymGetErrorDetails() -> *const c_char;

    fn msymCreateContext() -> msym_context;
    fn msymReleaseContext(ctx: msym_context) -> msym_error_t;

    pub(crate) fn msymSetElements(ctx: msym_context, length: c_int, elements: *mut msym_element_t) -> msym_error_t;
    pub(crate) fn msymGetElements(ctx: msym_context, length: *mut c_int, elements: *mut *mut msym_element_t) -> msym_error_t;
    pub(crate) fn msymSetBasisFunctions(ctx: msym_context, length: c_int, basis: *mut msym_basis_function_t) -> msym_error_t;
    pub(crate) fn msymGetPointGroupName(ctx: msym_context, l: c_int, buf: *mut c_char) -> msym_error_t;
    pub(crate) fn msymGetSubgroups(ctx: msym_context, l: *mut c_int, subgroups: *mut *const msym_subgroup_t) -> msym_error_t;
    pub(crate) fn msymSelectSubgroup(ctx: msym_context, subgroup: *const msym_subgroup_t) -> msym_error_t;
    pub(crate) fn msymGetCharacterTable(ctx: msym_context, ct: *mut *const msym_character_table_t) -> msym_error_t;
    pub(crate) fn msymFindSymmetry(ctx: msym_context) -> msym_error_t;
    pub(crate) fn msymSymmetrizeElements(ctx: msym_context, err: *mut c_double) -> msym_error_t;
    pub(crate) fn msymGenerateSubrepresentationSpaces(ctx: msym_context) -> msym_error_t;
    pub(crate) fn msymGetSALCs(ctx: msym_context, l: c_int, c: *mut c_double, species: *mut c_int, pf: *mut msym_partner_function_t) -> msym_error_t;
    pub(crate) fn msymGetCenterOfMass(ctx: msym_context, v: *mut c_double) -> msym_error_t;
    pub(crate) fn msymGetAlignmentTransform(ctx: msym_context, transform: *mut c_double) -> msym_error_t;
    pub(crate) fn msymGetSymmetryOperations(ctx: msym_context, sopsl: *mut c_int, sops: *mut *const msym_symmetry_operation_t) -> msym_error_t;

    // Not in msym.h: libmsym's internal (symop.h) 3x3 matrix of a symmetry operation, linkable from the static build.
    pub(crate) fn symmetryOperationMatrix(sop: *const msym_symmetry_operation_t, m: *mut c_double);
    // Not in msym.h either (rsh.h): the real-solid-harmonic representation of every operation, for l = 0..=lmax.
    // `lrs` has lmax + 1 entries, entry l with d = 2l + 1 and t a caller-owned double[sopsl][d][d], filled as
    // t[op][m1 + l][m2 + l]. It only reads the operations.
    pub(crate) fn generateRSHRepresentations(sopsl: c_int, sops: *mut msym_symmetry_operation_t, lmax: c_int, lrs: *mut rsh_representations_t) -> msym_error_t;
}

#[repr(C)]
pub(crate) struct rsh_representations_t {
    pub(crate) d: c_int,
    pub(crate) t: *mut c_void,
}

/// libmsym's message for a failed call: the generic text for the code plus the details of the last error. The
/// details live in one global buffer inside libmsym, so they belong to whichever call failed last.
pub(crate) fn error_message(code: msym_error_t) -> String {
    // SAFETY: both return pointers to static NUL-terminated strings owned by libmsym.
    unsafe {
        let generic = CStr::from_ptr(msymErrorString(code)).to_string_lossy();
        let details = CStr::from_ptr(msymGetErrorDetails()).to_string_lossy();
        format!("{generic}: {details}")
    }
}

/// `Ok` on `MSYM_SUCCESS`, otherwise libmsym's message prefixed with what was being done.
pub(crate) fn check(code: msym_error_t, what: &str) -> Result<(), String> {
    if code == MSYM_SUCCESS { Ok(()) } else { Err(format!("libmsym failed to {what}: {}", error_message(code))) }
}

/// A NUL-terminated (or full) C character buffer as a `String`.
pub(crate) fn c_chars_to_string(chars: &[c_char]) -> String {
    let bytes: Vec<u8> = chars.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// An owned libmsym context, released on drop.
#[derive(Debug)]
pub(crate) struct Context(msym_context);

impl Context {
    pub(crate) fn new() -> Result<Self, String> {
        // SAFETY: no preconditions; a null return means allocation failed.
        let ctx = unsafe { msymCreateContext() };
        if ctx.is_null() { Err("libmsym could not allocate a context".to_string()) } else { Ok(Context(ctx)) }
    }

    pub(crate) fn raw(&self) -> msym_context {
        self.0
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: self.0 came from msymCreateContext and is released exactly once. The return code is ignored: nothing
        // can be done about a failed release in a destructor.
        unsafe {
            msymReleaseContext(self.0);
        }
    }
}
