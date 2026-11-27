// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The basis and the centres: dimensions, ids, primitives, centre labels, the geometry step, nuclear charge and
// repulsion, Cartesian rescaling, the AO amplitudes on caller-given points.

use super::*;

/// The centre of mass of the real atoms (bohr) into `x`, `y`, `z`; nothing on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn compute_centre_of_mass_c(ctx: *mut AedelstenContextHandle, x: *mut f64, y: *mut f64, z: *mut f64) {
    if ctx.is_null() {
        return;
    }
    unsafe {
        let real = &*(ctx as *mut AedelstenContext);
        let centre_of_mass = real.compute_centre_of_mass();
        // Slighly different pattern here than typical: ctx return [f64; 3], since it is small and cheap to copy,
        // compared to integral routines that return large arrays and need to be passed a pointer to preallocated
        // memory. This keeps the Rust side of the API ergonomic, while still allowing C to get the values out.
        *x = centre_of_mass[0];
        *y = centre_of_mass[1];
        *z = centre_of_mass[2];
    }
}

/// The repulsion energy of the QM region's nuclei and core AIMPs (see
/// `AedelstenContext::compute_nuclear_repulsion_energy`); 0 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn compute_nuclear_repulsion_energy_c(ctx: *mut AedelstenContextHandle) -> f64 {
    if ctx.is_null() {
        return 0.0;
    }
    let real = unsafe { &*(ctx as *mut AedelstenContext) };
    real.compute_nuclear_repulsion_energy()
}

/// The total charge of the QM region's nuclei and core AIMPs; 0 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn compute_nuclear_charge_c(ctx: *mut AedelstenContextHandle) -> f64 {
    if ctx.is_null() {
        return 0.0;
    }
    let real = unsafe { &*(ctx as *mut AedelstenContext) };
    real.compute_nuclear_charge()
}

/// The number of shells, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_shells_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.get_n_shells() as i32
}

/// Every shell, in AO order, flat and column-major as `get_basis_function_ids_c`: all centres (0-based), then all l
/// (negated for a Cartesian shell), all first AOs (0-based), all AO counts. `len` is `4 * get_n_shells_c`. Returns 0,
/// or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_shells_c(ctx: *const AedelstenContextHandle, output: *mut i32, len: i32) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n = real.get_n_shells();
    if len < 0 || len as usize != 4 * n {
        rank0_eprintln!("Error in get_shells (C wrapper {}): expected a buffer of length {}, got {}", stringify!(get_shells_c), 4 * n, len);
        return FAILURE;
    }
    let mut shells = vec![crate::ShellInfo::default(); n];
    if let Err(e) = real.get_shells(&mut shells) {
        rank0_eprintln!("Error in get_shells (C wrapper {}): {}", stringify!(get_shells_c), e);
        return FAILURE;
    }
    let output = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
    for (s, info) in shells.iter().enumerate() {
        output[s] = info.centre as i32;
        output[n + s] = info.l;
        output[2 * n + s] = info.first_ao as i32;
        output[3 * n + s] = info.n_ao as i32;
    }
    0
}

/// Per shell, the distance from its centre beyond which its AOs and their gradients are below `threshold` (never below
/// 1 bohr; see `AedelstenContext::compute_shell_extents`); `n_shells` is `get_n_shells_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_shell_extents_c(ctx: *const AedelstenContextHandle, threshold: f64, output: *mut f64, n_shells: i32) -> i32 {
    if ctx.is_null() || output.is_null() || n_shells < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let output = unsafe { std::slice::from_raw_parts_mut(output, n_shells as usize) };
    if let Err(e) = real.compute_shell_extents(threshold, output) {
        rank0_eprintln!("Error in compute_shell_extents (C wrapper {}): {}", stringify!(compute_shell_extents_c), e);
        return FAILURE;
    }
    0
}

/// The values of the AOs of the shells `shells` (`n_shells`, 0-based, any order and subset) at `points` (`(point_rows,
/// n_points)`, `point_rows` 3 or 4, the fourth row not read) into `values` (`(n_ao_listed, n_points)`, the listed
/// shells' AOs in the order of the list), and their gradients into `gradients` (three such blocks, x, y, z) unless it
/// is null. Outputs overlapping each other or the inputs are refused. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_ao_values_c(
    ctx: *const AedelstenContextHandle,
    shells: *const i32,
    n_shells: i32,
    points: *const f64,
    point_rows: i32,
    n_points: i32,
    values: *mut f64,
    gradients: *mut f64,
    n_ao_listed: i32,
) -> i32 {
    if ctx.is_null() || shells.is_null() || points.is_null() || values.is_null() || n_shells < 0 || point_rows < 0 || n_points < 0 || n_ao_listed < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let shells = unsafe { std::slice::from_raw_parts(shells, n_shells as usize) };
    let points = unsafe { std::slice::from_raw_parts(points, point_rows as usize * n_points as usize) };
    let n = n_ao_listed as usize * n_points as usize;
    let values = unsafe { std::slice::from_raw_parts_mut(values, n) };
    let gradients = if gradients.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts_mut(gradients, 3 * n) }) };
    let shells = shells.iter().map(|&s| usize::try_from(s).unwrap_or(usize::MAX));
    if let Err(e) = real.compute_ao_values_of(shells, points, point_rows as usize, values, gradients) {
        rank0_eprintln!("Error in compute_ao_values (C wrapper {}): {}", stringify!(compute_ao_values_c), e);
        return FAILURE;
    }
    0
}

/// As `compute_ao_values_c`, with `gradients` three blocks (x, y, z) and `hessians` six (xx, xy, xz, yy, yz, zz),
/// neither null. Outputs overlapping each other or the inputs are refused. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn compute_ao_hessians_c(
    ctx: *const AedelstenContextHandle,
    shells: *const i32,
    n_shells: i32,
    points: *const f64,
    point_rows: i32,
    n_points: i32,
    values: *mut f64,
    gradients: *mut f64,
    hessians: *mut f64,
    n_ao_listed: i32,
) -> i32 {
    if ctx.is_null()
        || shells.is_null()
        || points.is_null()
        || values.is_null()
        || gradients.is_null()
        || hessians.is_null()
        || n_shells < 0
        || point_rows < 0
        || n_points < 0
        || n_ao_listed < 0
    {
        return FAILURE;
    }
    let n = n_ao_listed as usize * n_points as usize;
    let n_point_values = point_rows as usize * n_points as usize;
    // The shell list's bytes as doubles, rounded up, for the overlap check.
    let shell_doubles = (n_shells as usize * size_of::<i32>()).div_ceil(size_of::<f64>());
    let outputs = [(values as *const f64, n), (gradients as *const f64, 3 * n), (hessians as *const f64, 6 * n)];
    if buffers_overlap(&outputs, &[(points, n_point_values), (shells as *const f64, shell_doubles)]) {
        rank0_eprintln!("Error in compute_ao_hessians (C wrapper {}): an output overlaps another buffer", stringify!(compute_ao_hessians_c));
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let shells = unsafe { std::slice::from_raw_parts(shells, n_shells as usize) };
    let points = unsafe { std::slice::from_raw_parts(points, n_point_values) };
    let values = unsafe { std::slice::from_raw_parts_mut(values, n) };
    let gradients = unsafe { std::slice::from_raw_parts_mut(gradients, 3 * n) };
    let hessians = unsafe { std::slice::from_raw_parts_mut(hessians, 6 * n) };
    let shells = shells.iter().map(|&s| usize::try_from(s).unwrap_or(usize::MAX));
    if let Err(e) = real.compute_ao_hessians_of(shells, points, point_rows as usize, values, gradients, hessians) {
        rank0_eprintln!("Error in compute_ao_hessians (C wrapper {}): {}", stringify!(compute_ao_hessians_c), e);
        return FAILURE;
    }
    0
}

/// The number of contracted basis functions (AOs), -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_cont_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.get_n_cont() as i32
}

/// The number of primitive rows of `get_primitive_ids_c` and `get_primitives_c`, -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_primitive_rows_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.get_n_primitive_rows() as i32
}

/// The sum of the shells' primitive AOs (a primitive shared by sibling segments once per sibling), -1 on a null
/// context.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_prim_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.get_n_prim() as i32
}

/// The number of distinct primitive AOs, the dimension of the primitive basis; -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_prim_unique_c(ctx: *const AedelstenContextHandle) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.get_n_prim_unique() as i32
}

/// Moves the gradient centres: `positions` is `3 * n_centres` doubles, a column-major `(3, n_centres)` matrix (the
/// layout of the gradients), `n_centres` must be `get_n_gradient_centres_c`. Each centre carries its basis shells,
/// nucleus and core AIMP; orthogonality ghosts and embedding AIMPs stay fixed. Invalidates the geometry-derived
/// caches: call `set_shell_pairs_c`, `set_eri_data_c` (and primitive shell pairs / X2C, if used) again. Every MPI
/// rank must pass the same positions.
#[unsafe(no_mangle)]
pub extern "C" fn set_centre_positions_c(ctx: *mut AedelstenContextHandle, positions: *const f64, n_centres: i32) -> i32 {
    if ctx.is_null() || positions.is_null() || n_centres < 0 {
        return FAILURE;
    }
    let real = unsafe { &mut *(ctx as *mut AedelstenContext) };
    let positions_slice = unsafe { std::slice::from_raw_parts(positions, 3 * n_centres as usize) };
    match real.set_centre_positions(positions_slice) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in set_centre_positions (C wrapper {}): {}", stringify!(set_centre_positions_c), e);
            FAILURE
        }
    }
}

/// Every AO's id (`BasisFunctionId`), flat and column-major: all centres, then all shells, all l, all m, so that it
/// maps onto a Fortran `integer(4, n)` array. `len` is `4 * get_n_cont_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_basis_function_ids_c(ctx: *mut AedelstenContextHandle, output: *mut i32, len: i32) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n = real.get_n_cont();
    if len < 0 || len as usize != 4 * n {
        rank0_eprintln!(
            "Error in get_basis_function_ids (C wrapper {}): expected a buffer of length {}, got {}",
            stringify!(get_basis_function_ids_c),
            4 * n,
            len
        );
        return FAILURE;
    }
    let mut ids = vec![crate::BasisFunctionId::default(); n];
    if let Err(e) = real.get_basis_function_ids(&mut ids) {
        rank0_eprintln!("Error in get_basis_function_ids (C wrapper {}): {}", stringify!(get_basis_function_ids_c), e);
        return FAILURE;
    }
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
    for (i, id) in ids.iter().enumerate() {
        output_slice[i] = id.get_centre() as i32;
        output_slice[n + i] = id.get_shell() as i32;
        output_slice[2 * n + i] = id.get_l();
        output_slice[3 * n + i] = id.get_m();
    }
    0
}

/// Every primitive row's id (`PrimitiveId`), flat and column-major as `get_basis_function_ids_c`: all centres, then all
/// l, all shells. `len` is `3 * get_n_primitive_rows_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_primitive_ids_c(ctx: *mut AedelstenContextHandle, output: *mut i32, len: i32) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n = real.get_n_primitive_rows();
    if len < 0 || len as usize != 3 * n {
        rank0_eprintln!(
            "Error in get_primitive_ids (C wrapper {}): expected a buffer of length {}, got {}",
            stringify!(get_primitive_ids_c),
            3 * n,
            len
        );
        return FAILURE;
    }
    let mut ids = vec![crate::PrimitiveId { centre: 0, l: 0, shell: 0 }; n];
    if let Err(e) = real.get_primitive_ids(&mut ids) {
        rank0_eprintln!("Error in get_primitive_ids (C wrapper {}): {}", stringify!(get_primitive_ids_c), e);
        return FAILURE;
    }
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
    for (i, id) in ids.iter().enumerate() {
        output_slice[i] = id.centre as i32;
        output_slice[n + i] = id.l as i32;
        output_slice[2 * n + i] = id.shell as i32;
    }
    0
}

/// Every primitive row's exponent and coefficient, parallel to `get_primitive_ids_c`: all exponents, then all
/// coefficients. `len` is `2 * get_n_primitive_rows_c`. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn get_primitives_c(ctx: *mut AedelstenContextHandle, output: *mut f64, len: i32) -> i32 {
    if ctx.is_null() || output.is_null() {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n = real.get_n_primitive_rows();
    if len < 0 || len as usize != 2 * n {
        rank0_eprintln!(
            "Error in get_primitives (C wrapper {}): expected a buffer of length {}, got {}",
            stringify!(get_primitives_c),
            2 * n,
            len
        );
        return FAILURE;
    }
    let mut primitives = vec![crate::Primitive { exponent: 0.0, coefficient: 0.0 }; n];
    if let Err(e) = real.get_primitives(&mut primitives) {
        rank0_eprintln!("Error in get_primitives (C wrapper {}): {}", stringify!(get_primitives_c), e);
        return FAILURE;
    }
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, len as usize) };
    for (i, p) in primitives.iter().enumerate() {
        output_slice[i] = p.exponent;
        output_slice[n + i] = p.coefficient;
    }
    0
}

/// Rescales `matrix` (AO rows, `n_cols` columns, `len` values, column-major) in place from this library's Cartesian
/// normalization to `convention`'s (a `SharedCartesianReference` code: 0 Molcas's, 1 Dalton's); `rescaled`, if not
/// null, is set to 1 when a factor differed from 1, else 0. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn rescale_cartesian_ao_rows_c(
    ctx: *mut AedelstenContextHandle,
    convention: i32,
    n_cols: i32,
    matrix: *mut f64,
    len: i32,
    rescaled: *mut i32,
) -> i32 {
    if ctx.is_null() || matrix.is_null() || n_cols < 0 || len < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let convention = match cart2sph::SharedCartesianReference::from_c_int(convention) {
        Ok(c) => c,
        Err(e) => {
            rank0_eprintln!("Error in rescale_cartesian_ao_rows (C wrapper {}): {}", stringify!(rescale_cartesian_ao_rows_c), e);
            return FAILURE;
        }
    };
    let matrix_slice = unsafe { std::slice::from_raw_parts_mut(matrix, len as usize) };
    match real.rescale_cartesian_ao_rows(convention, n_cols as usize, matrix_slice) {
        Ok(was_rescaled) => {
            if !rescaled.is_null() {
                unsafe { *rescaled = was_rescaled as i32 };
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in rescale_cartesian_ao_rows (C wrapper {}): {}", stringify!(rescale_cartesian_ao_rows_c), e);
            FAILURE
        }
    }
}

/// As `rescale_cartesian_ao_rows_c`, for a matrix of `n_rows` rows whose columns are the AOs.
#[unsafe(no_mangle)]
pub extern "C" fn rescale_cartesian_ao_columns_c(
    ctx: *mut AedelstenContextHandle,
    convention: i32,
    n_rows: i32,
    matrix: *mut f64,
    len: i32,
    rescaled: *mut i32,
) -> i32 {
    if ctx.is_null() || matrix.is_null() || n_rows < 0 || len < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let convention = match cart2sph::SharedCartesianReference::from_c_int(convention) {
        Ok(c) => c,
        Err(e) => {
            rank0_eprintln!("Error in rescale_cartesian_ao_columns (C wrapper {}): {}", stringify!(rescale_cartesian_ao_columns_c), e);
            return FAILURE;
        }
    };
    let matrix_slice = unsafe { std::slice::from_raw_parts_mut(matrix, len as usize) };
    match real.rescale_cartesian_ao_columns(convention, n_rows as usize, matrix_slice) {
        Ok(was_rescaled) => {
            if !rescaled.is_null() {
                unsafe { *rescaled = was_rescaled as i32 };
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in rescale_cartesian_ao_columns (C wrapper {}): {}", stringify!(rescale_cartesian_ao_columns_c), e);
            FAILURE
        }
    }
}

/// Centre `centre_idx`'s label (0-based, see `CentreLabel`), NUL-terminated into `buf` (capacity `buf_len`, the NUL
/// included). Returns 0, or 1 when the centre has no label (out of range, an orthogonality ghost) or the label does not
/// fit; never truncates.
#[unsafe(no_mangle)]
pub extern "C" fn get_centre_label_c(ctx: *mut AedelstenContextHandle, centre_idx: i32, buf: *mut std::ffi::c_char, buf_len: i32) -> i32 {
    if ctx.is_null() || buf.is_null() || centre_idx < 0 || buf_len < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let Some(label) = real.get_centre_label(centre_idx as usize) else {
        rank0_eprintln!("Error in get_centre_label (C wrapper {}): no label for centre {}", stringify!(get_centre_label_c), centre_idx);
        return FAILURE;
    };
    let text = label.to_string();
    let bytes = text.as_bytes();
    // >=, not >: need room for the trailing NUL too.
    if bytes.len() >= buf_len as usize {
        rank0_eprintln!(
            "Error in get_centre_label (C wrapper {}): buffer of length {} too small for \"{}\" ({} bytes + NUL)",
            stringify!(get_centre_label_c),
            buf_len,
            text,
            bytes.len()
        );
        return FAILURE;
    }
    let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, buf_len as usize) };
    buf_slice[..bytes.len()].copy_from_slice(bytes);
    buf_slice[bytes.len()] = 0;
    0
}

/// The number of centres (real atoms, ghost centres and orthogonality ghosts), -1 on a null context.
#[unsafe(no_mangle)]
pub extern "C" fn get_n_centres_c(ctx: *const AedelstenContextHandle) -> i64 {
    if ctx.is_null() {
        return -1;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    real.get_n_centres() as i64
}

/// Every centre's atomic number, position (bohr, column-major (3, n)) and kind (the `atom_kind_*` constants), in
/// centre order, `n` = `get_n_centres_c`. Returns 0, or 1 on a null pointer, overlapping outputs or a wrong `n`.
#[unsafe(no_mangle)]
pub extern "C" fn get_centre_geometry_c(ctx: *const AedelstenContextHandle, atomic_numbers: *mut i32, positions: *mut f64, kinds: *mut i32, n: i64) -> i32 {
    if ctx.is_null() || atomic_numbers.is_null() || positions.is_null() || kinds.is_null() || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let n = n as usize;
    // The integer outputs' bytes counted in doubles, rounded up.
    let buffers = [(atomic_numbers as *const f64, n.div_ceil(2)), (positions as *const f64, 3 * n), (kinds as *const f64, n.div_ceil(2))];
    if buffers_overlap(&buffers, &[]) {
        rank0_eprintln!("Error in C wrapper {}: the outputs overlap", stringify!(get_centre_geometry_c));
        return FAILURE;
    }
    let mut centres = vec![basis::CentreGeometry { atomic_number: 0, position: [0.0; 3], kind: basis::AtomKind::Real }; n];
    if let Err(e) = real.get_centre_geometry(&mut centres) {
        rank0_eprintln!("Error in get_centre_geometry (C wrapper {}): {}", stringify!(get_centre_geometry_c), e);
        return FAILURE;
    }
    let (z, xyz, kind) = unsafe { (std::slice::from_raw_parts_mut(atomic_numbers, n), std::slice::from_raw_parts_mut(positions, 3 * n), std::slice::from_raw_parts_mut(kinds, n)) };
    for (centre, ((z, xyz), kind)) in centres.iter().zip(z.iter_mut().zip(xyz.chunks_exact_mut(3)).zip(kind.iter_mut())) {
        *z = i32::from(centre.atomic_number);
        xyz.copy_from_slice(&centre.position);
        *kind = centre.kind as i32;
    }
    0
}

/// Every centre's nuclear charge (its nucleus's or its core AIMP's; 0 for a ghost centre or an orthogonality ghost),
/// `n` = `get_n_centres_c`. Returns 0, or 1 on a null pointer or a wrong `n`.
#[unsafe(no_mangle)]
pub extern "C" fn get_nuclear_charges_c(ctx: *const AedelstenContextHandle, charges: *mut f64, n: i64) -> i32 {
    if ctx.is_null() || charges.is_null() || n < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let charges = unsafe { std::slice::from_raw_parts_mut(charges, n as usize) };
    match real.get_nuclear_charges(charges) {
        Ok(()) => 0,
        Err(e) => {
            rank0_eprintln!("Error in get_nuclear_charges (C wrapper {}): {}", stringify!(get_nuclear_charges_c), e);
            FAILURE
        }
    }
}

/// The inverse of `rescale_cartesian_ao_rows_c`: `matrix` (AO rows, `n_cols` columns, `len` values) from
/// `convention`'s Cartesian normalization to this library's, in place; `rescaled`, if not null, 1 when a factor
/// differed from 1. Returns 0, or 1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn unrescale_cartesian_ao_rows_c(
    ctx: *mut AedelstenContextHandle,
    convention: i32,
    n_cols: i32,
    matrix: *mut f64,
    len: i32,
    rescaled: *mut i32,
) -> i32 {
    if ctx.is_null() || matrix.is_null() || n_cols < 0 || len < 0 {
        return FAILURE;
    }
    let real = unsafe { &*(ctx as *const AedelstenContext) };
    let convention = match cart2sph::SharedCartesianReference::from_c_int(convention) {
        Ok(c) => c,
        Err(e) => {
            rank0_eprintln!("Error in unrescale_cartesian_ao_rows (C wrapper {}): {}", stringify!(unrescale_cartesian_ao_rows_c), e);
            return FAILURE;
        }
    };
    let matrix_slice = unsafe { std::slice::from_raw_parts_mut(matrix, len as usize) };
    match real.unrescale_cartesian_ao_rows(convention, n_cols as usize, matrix_slice) {
        Ok(was_rescaled) => {
            if !rescaled.is_null() {
                unsafe { *rescaled = was_rescaled as i32 };
            }
            0
        }
        Err(e) => {
            rank0_eprintln!("Error in unrescale_cartesian_ao_rows (C wrapper {}): {}", stringify!(unrescale_cartesian_ao_rows_c), e);
            FAILURE
        }
    }
}
