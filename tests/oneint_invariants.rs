// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Internal (self-referential) correctness checks for the low-level
// one-electron integral routines in oneint -- deliberately independent of
// any external reference package: every check here is either an exact
// mathematical identity the *implementation itself* must satisfy (order-0
// multipole == overlap; the dipole origin-shift relation) or a structural
// property (symmetry, the primitive<->contracted round-trip) verifiable
// from first principles, with no golden data to keep in sync. Same
// "independent code path" philosophy as
// tests/gradient_finite_difference.rs, applied to values instead of
// gradients.

use aedelsten::{AedelstenContext, ContractionType, MomentOrigin, OneElectronOperator};
use aedelsten::arrays::matrix::MatrixView;

fn build_context() -> AedelstenContext {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/mpi_oneint_check.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn max_asymmetry(n: usize, m: &[f64]) -> f64 {
    let view = MatrixView::new(n, n, m);
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((view[(i, j)] - view[(j, i)]).abs());
        }
    }
    worst
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn overlap_kinetic_and_nuclear_attraction_are_symmetric() {
    let ctx = build_context();
    let n = ctx.get_n_cont();

    let mut overlap = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();
    assert!(max_asymmetry(n, &overlap) < 1e-14, "overlap is not symmetric");

    let mut kinetic = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut kinetic, OneElectronOperator::kinetic(), false).unwrap();
    assert!(max_asymmetry(n, &kinetic) < 1e-14, "kinetic is not symmetric");

    let mut nuclear = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut nuclear, ctx.get_nuclear_attraction_operator(), false).unwrap();
    assert!(max_asymmetry(n, &nuclear) < 1e-13, "nuclear attraction is not symmetric");
}

#[test]
fn overlap_trace_equals_n_cont() {
    // Every AO is normalized to unit self-overlap, so the overlap matrix's
    // diagonal is all 1s regardless of geometry or basis -- a cheap,
    // geometry-independent check on the normalization step specifically.
    let ctx = build_context();
    let n = ctx.get_n_cont();
    let mut overlap = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();
    let view = MatrixView::new(n, n, &overlap);
    let trace: f64 = (0..n).map(|i| view[(i, i)]).sum();
    assert!((trace - n as f64).abs() < 1e-12, "trace(overlap) = {trace}, expected {n}");
}

#[test]
fn order_zero_multipole_moment_equals_overlap_at_any_origin() {
    // <A|(x-Ox)^0(y-Oy)^0(z-Oz)^0|B> = <A|1|B> = overlap, identically, for
    // *any* origin O -- raising to the 0th power erases the origin
    // dependence entirely. Exercises the multipole/Hermite machinery
    // against the independently implemented overlap kernel, at two
    // different origins (including a non-trivial, off-molecule one), with
    // no external reference needed: the identity is exact by construction.
    let ctx = build_context();
    let n = ctx.get_n_cont();

    let mut overlap = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();

    for origin in [[0.0, 0.0, 0.0], [3.7, -1.2, 5.5]] {
        let mut moment = vec![0.0; n * n];
        ctx.compute_one_electron_integrals(&mut moment, OneElectronOperator::moment(MomentOrigin::new(origin, 0)), false).unwrap();
        let diff = max_abs_diff(&overlap, &moment);
        assert!(diff < 1e-13, "order-0 multipole at origin {origin:?} disagrees with overlap by {diff}");
    }
}

#[test]
fn dipole_moment_satisfies_the_origin_shift_identity() {
    // <A|(r - O2)|B> = <A|(r - O1)|B> - (O2 - O1)*<A|B>, an exact analytic
    // identity for the order-1 (dipole) moment under a shift of origin --
    // holds for any geometry/basis, so this is a genuine check on the
    // multipole kernel's origin handling (a common source of bugs), not a
    // restatement of physics assumed to be true.
    let ctx = build_context();
    let n = ctx.get_n_cont();

    let mut overlap = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();

    let origin1 = [0.0, 0.0, 0.0];
    let origin2 = [1.3, -2.1, 0.6];
    let mut dipole1 = vec![0.0; n * n * 3];
    ctx.compute_one_electron_integrals(&mut dipole1, OneElectronOperator::moment(MomentOrigin::new(origin1, 1)), false).unwrap();
    let mut dipole2 = vec![0.0; n * n * 3];
    ctx.compute_one_electron_integrals(&mut dipole2, OneElectronOperator::moment(MomentOrigin::new(origin2, 1)), false).unwrap();

    let shift = [origin2[0] - origin1[0], origin2[1] - origin1[1], origin2[2] - origin1[2]];
    let mut max_diff = 0.0_f64;
    for dim in 0..3 {
        for k in 0..n * n {
            let predicted = dipole1[dim * n * n + k] - shift[dim] * overlap[k];
            max_diff = max_diff.max((predicted - dipole2[dim * n * n + k]).abs());
        }
    }
    assert!(max_diff < 1e-12, "dipole origin-shift identity violated by {max_diff}");
}

#[test]
fn contracting_primitive_overlap_reproduces_the_contracted_basis_result() {
    // The whole point of the decontraction/contraction machinery is that
    // this round-trip is exact (up to floating point): computing an
    // operator over the primitive basis and contracting it back must
    // reproduce computing it directly over the contracted basis.
    let mut ctx = build_context();
    ctx.set_primitive_shell_pairs(None).unwrap();
    let n_cont = ctx.get_n_cont();
    let n_prim = ctx.get_n_prim_unique();

    let mut standard = vec![0.0; n_cont * n_cont];
    ctx.compute_one_electron_integrals(&mut standard, OneElectronOperator::overlap(), false).unwrap();

    let mut primitive = vec![0.0; n_prim * n_prim];
    ctx.compute_one_electron_integrals(&mut primitive, OneElectronOperator::overlap(), true).unwrap();

    let mut contracted_back = vec![0.0; n_cont * n_cont];
    ctx.contract_one_electron_matrix(&primitive, &mut contracted_back).unwrap();

    let diff = max_abs_diff(&standard, &contracted_back);
    assert!(diff < 1e-10, "overlap: contract(primitive) disagrees with standard by {diff}");
}

