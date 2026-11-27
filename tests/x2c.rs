// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Exercises AedelstenContextImpl::construct_x2c_one_electron_hamiltonian end to
// end. Builds the primitive-basis S/T/V/pVp integrals,
// constructs the spin-free X2C one-electron Hamiltonian, and checks the
// resulting picture-change transform is cached the same way as the other
// AedelstenContextImpl state (set_*/get_*/free_* with a CacheAlreadyDerived guard).

use aedelsten::{AedelstenContext, ContractionType, MomentOrigin, OneElectronOperator};
use aedelsten::arrays::array3d::{Array3DTrait, Array3DView};
use aedelsten::arrays::matrix::{Matrix, MatrixView, MatrixMath, MatrixTrait};

// Builds and caches the picture-change transform on `ctx` from its primitive
// S/T/V/pVp integrals (discarding the one-electron Hamiltonian itself --
// callers here only want the cached PictureChange for further transforms).
fn build_picture_change(ctx: &mut AedelstenContext, n: usize, s: &[f64], t: &[f64], v: &[f64], pvp: &[f64]) {
    let mut oneham = vec![0.0; n * n];
    ctx.construct_x2c_one_electron_hamiltonian(&mut oneham, s, t, v, pvp, None).unwrap();
}

// Shared setup: a context with primitive S/T/V/pVp integrals in hand, ready
// for construct_x2c_one_electron_hamiltonian. Returns (ctx, n, s, t, v, pvp).
fn setup_primitive_integrals() -> (AedelstenContext, usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();

    let n = ctx.get_n_prim_unique();
    let mut s = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut s, OneElectronOperator::overlap(), true).unwrap();
    let mut t = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut t, OneElectronOperator::kinetic(), true).unwrap();
    let mut v = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut v, ctx.get_nuclear_attraction_operator(), true).unwrap();
    let mut pvp = vec![0.0; n * n];
    ctx.compute_one_electron_picture_change_integrals(&mut pvp, ctx.get_nuclear_attraction_operator(), false).unwrap();

    (ctx, n, s, t, v, pvp)
}

#[test]
fn construct_x2c_one_electron_hamiltonian_builds_a_symmetric_hamiltonian_and_caches_picture_change() {
    let (mut ctx, n, s, t, v, pvp) = setup_primitive_integrals();

    let mut oneham = vec![0.0; n * n];
    ctx.construct_x2c_one_electron_hamiltonian(&mut oneham, &s, &t, &v, &pvp, None).unwrap();

    // Symmetric by construction (each of T/V/pVp is its own symmetric
    // operator sandwiched the same way on both sides), up to ordinary
    // floating-point rounding.
    for i in 0..n {
        for j in 0..n {
            assert!(
                (oneham[i * n + j] - oneham[j * n + i]).abs() < 1e-8,
                "one-electron Hamiltonian not symmetric at ({i},{j}): {} vs {}",
                oneham[i * n + j], oneham[j * n + i]
            );
        }
    }
    // Not trivially all-zero -- a real, computed Hamiltonian.
    assert!(oneham.iter().any(|&x| x.abs() > 1e-6));

    // The picture-change transform is cached and retrievable, with the
    // large-component transform sized (n_prim, n_prim).
    assert!(ctx.has_picture_change());
    let (u_large, _) = picture_change(&ctx, n);
    assert_eq!((u_large.rows(), u_large.cols()), (n, n));
    // The getter checks its buffers' sizes.
    assert!(ctx.get_picture_change_transforms(&mut vec![0.0; n], &mut vec![0.0; n * n]).is_err());

    // A second call without freeing first must be rejected, same convention
    // as set_scratch/set_eri_data/etc.
    let mut oneham2 = vec![0.0; n * n];
    let err = ctx.construct_x2c_one_electron_hamiltonian(&mut oneham2, &s, &t, &v, &pvp, None);
    assert!(err.is_err(), "expected CacheAlreadyDerived on a second construct_x2c_one_electron_hamiltonian call");

    // free_picture_change releases the cache so a fresh construction works again.
    ctx.free_picture_change();
    assert!(!ctx.has_picture_change());
    assert!(ctx.get_picture_change_transforms(&mut vec![0.0; n * n], &mut vec![0.0; n * n]).is_err());
    ctx.construct_x2c_one_electron_hamiltonian(&mut oneham2, &s, &t, &v, &pvp, None).unwrap();
}

// transform_one_electron_operator is a thin wrapper around
// x2c::PictureChange::transform_one_electron_operator: fetch u_large/u_small
// directly and replicate the same math by hand (congruence_transform, already
// independently tested in arrays) to confirm the wrapper delegates correctly
// -- dimensions derived from get_u_large().rows(), no transposition/argument
// mixups. Not a physically meaningful (plain, pXp) pair (S paired with
// itself), just a mechanics check.
#[test]
fn transform_one_electron_operator_matches_manual_picture_change_transform() {
    let (mut ctx, n, s, t, v, pvp) = setup_primitive_integrals();
    build_picture_change(&mut ctx, n, &s, &t, &v, &pvp);

    let mut via_wrapper = vec![0.0; n * n];
    ctx.transform_one_electron_operator(&s, &s, &mut via_wrapper).unwrap();

    let s_matrix = MatrixView::new(n, n, &s);
    let (u_large, u_small) = picture_change(&ctx, n);
    let transformed_plain = s_matrix.congruence_transform(&u_large);
    let transformed_pxp = s_matrix.congruence_transform(&u_small);
    for i in 0..n {
        for j in 0..n {
            let expected = transformed_plain[(i, j)] + transformed_pxp[(i, j)];
            assert!(
                (via_wrapper[i * n + j] - expected).abs() < 1e-8,
                "mismatch at ({i},{j}): wrapper={} manual={}",
                via_wrapper[i * n + j], expected
            );
        }
    }
}

// transform_one_electron_operator_array3d, exercised with an actual
// multi-component operator (the dipole moment: order 1, 3 Cartesian
// components) rather than a Matrix operator repeated per plane -- this is
// the physically real (plain, pXp) pairing (compute_one_electron_integrals(.., true)'s
// Array3DOutput arm for the plain dipole integrals,
// compute_one_electron_picture_change_integrals's Array3DOutput arm --
// routed to MomentPXPKernel -- for their pXp counterpart), cross-checked
// plane by plane against the same manual congruence_transform composition as
// the plain-Matrix test above.
#[test]
fn transform_one_electron_operator_array3d_handles_a_multipole_operator() {
    let (mut ctx, n, s, t, v, pvp) = setup_primitive_integrals();
    build_picture_change(&mut ctx, n, &s, &t, &v, &pvp);

    let origin = MomentOrigin::new([0.0, 0.0, 0.0], 1); // dipole: 3 components (x, y, z)
    let n_planes = 3;
    let mut plain = vec![0.0; n * n * n_planes];
    ctx.compute_one_electron_integrals(&mut plain, OneElectronOperator::moment(origin), true).unwrap();
    let mut pxp = vec![0.0; n * n * n_planes];
    ctx.compute_one_electron_picture_change_integrals(&mut pxp, OneElectronOperator::moment(origin), false).unwrap();

    let mut via_wrapper = vec![0.0; n * n * n_planes];
    ctx.transform_one_electron_operator_array3d(&plain, &pxp, n_planes, &mut via_wrapper).unwrap();

    let (u_large, u_small) = picture_change(&ctx, n);
    let plain_view = Array3DView::new(n, n, n_planes, &plain);
    let pxp_view = Array3DView::new(n, n, n_planes, &pxp);
    let out_view = Array3DView::new(n, n, n_planes, &via_wrapper);
    for k in 0..n_planes {
        let transformed_plain = plain_view.view_sub_matrix(k).congruence_transform(&u_large);
        let transformed_pxp = pxp_view.view_sub_matrix(k).congruence_transform(&u_small);
        let out_plane = out_view.view_sub_matrix(k);
        for i in 0..n {
            for j in 0..n {
                let expected = transformed_plain[(i, j)] + transformed_pxp[(i, j)];
                assert!(
                    (out_plane[(i, j)] - expected).abs() < 1e-8,
                    "plane {k} mismatch at ({i},{j}): wrapper={} manual={}",
                    out_plane[(i, j)], expected
                );
            }
        }
    }
}

// The X2C decoupling transforms, (u_large, u_small), through the public slice getter.
fn picture_change(ctx: &AedelstenContext, n: usize) -> (Matrix<f64>, Matrix<f64>) {
    let (mut u_large, mut u_small) = (vec![0.0; n * n], vec![0.0; n * n]);
    ctx.get_picture_change_transforms(&mut u_large, &mut u_small).unwrap();
    (Matrix::new_from_vec(n, n, u_large.into_boxed_slice()), Matrix::new_from_vec(n, n, u_small.into_boxed_slice()))
}
