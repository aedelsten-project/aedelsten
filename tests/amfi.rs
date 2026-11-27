// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// compute_amfi_integrals through the context: the planes of the amfi crate over the context's primitive basis, the
// citation, and the cleaning as two axial vectors with symmetry set.

mod common;

use aedelsten::{AedelstenContext, ContractionType, PointGroupRequest, SymmetryOptions};
use basis::contraction::{Contracted, ShellKind};

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.0\nEnd of Basis\n\
                     Basis set\n    H.x2c-SVPall\n    H1 0.0 1.43 1.11\n    H2 0.0 -1.43 1.11\nEnd of Basis\n";

fn context(tag: &str) -> (AedelstenContext, std::path::PathBuf) {
    let path = common::temp_path(&format!("amfi_{tag}.inp"));
    std::fs::write(&path, WATER).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    ctx.set_scratch((128, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    (ctx, path)
}

#[test]
fn the_context_planes_are_the_amfi_crate_planes_over_the_primitive_basis() {
    let (ctx, path) = context("plain");
    let n = ctx.get_n_prim_unique();
    let mut planes = vec![0.0; n * n * amfi::N_AMFI_PLANES];
    ctx.compute_amfi_integrals(&mut planes).unwrap();
    assert!(ctx.compute_amfi_integrals(&mut vec![0.0; n * n * 3]).is_err(), "the output size is checked");

    let input = basis::setup_basis_system::<Contracted>(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();
    let (shells, _) = Contracted::decontract(input.get_basis_shells(), input.get_basis_info()).unwrap();
    let mut expected = aedelsten::arrays::array3d::Array3D::<f64>::new(n, n, amfi::N_AMFI_PLANES);
    let mut scratch = aedelsten::arrays::scratch::Scratch::new(128, aedelsten::arrays::scratch::ByteUnit::MiB).unwrap();
    amfi::compute_amfi(&input, &shells, &mut aedelsten::arrays::array3d::Array3DTraitMut::view_mut(&mut expected), &mut scratch).unwrap();
    assert_eq!(planes, aedelsten::arrays::array3d::Array3DTrait::get_data(&expected));
    assert!(planes.iter().any(|x| x.abs() > 1e-2));

    let doi = "10.1016/0009-2614(96)00119-4";
    assert!(instrumentation::citations().into_iter().any(|r| r.reference.doi() == Some(doi)), "AMFI is cited");
}

#[test]
fn with_symmetry_the_planes_are_cleaned_as_two_axial_vectors() {
    let (mut ctx, path) = context("symmetry");
    std::fs::remove_file(&path).ok();
    let n = ctx.get_n_prim_unique();
    let mut raw = vec![0.0; n * n * amfi::N_AMFI_PLANES];
    ctx.compute_amfi_integrals(&mut raw).unwrap();
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let mut cleaned = vec![0.0; n * n * amfi::N_AMFI_PLANES];
    ctx.compute_amfi_integrals(&mut cleaned).unwrap();
    // Water is C2v: the planes are invariant already, so cleaning changes them by rounding only (a polar treatment
    // would be refused).
    let difference = raw.iter().zip(&cleaned).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(difference < 1e-10, "{difference}");
    for name in ["amfi_spin_same_orbit", "amfi_spin_other_orbit"] {
        assert!(ctx.get_symmetrisation_report(name, true).unwrap().is_some(), "{name} reported");
    }
}
