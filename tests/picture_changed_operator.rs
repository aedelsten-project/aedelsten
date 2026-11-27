// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The X2C picture of one-electron operators with their spin part: transform_one_electron_spin_operator and
// compute_picture_changed_one_electron_operator against the sandwiches built from get_picture_change_transforms.

use aedelsten::arrays::matrix::{Matrix, MatrixMath, MatrixTrait, MatrixTraitMut, MatrixView};
use aedelsten::{AedelstenContext, ContractionType, MomentOrigin, OneElectronOperator};

fn context() -> AedelstenContext {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_prim_unique();
    let mut integrals = [vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]];
    ctx.compute_one_electron_integrals(&mut integrals[0], OneElectronOperator::overlap(), true).unwrap();
    ctx.compute_one_electron_integrals(&mut integrals[1], OneElectronOperator::kinetic(), true).unwrap();
    ctx.compute_one_electron_integrals(&mut integrals[2], ctx.get_nuclear_attraction_operator(), true).unwrap();
    ctx.compute_one_electron_picture_change_integrals(&mut integrals[3], ctx.get_nuclear_attraction_operator(), false).unwrap();
    let mut oneham = vec![0.0; n * n];
    let [s, t, v, pvp] = &integrals;
    ctx.construct_x2c_one_electron_hamiltonian(&mut oneham, s, t, v, pvp, None).unwrap();
    ctx
}

fn transforms(ctx: &AedelstenContext) -> (Matrix<f64>, Matrix<f64>) {
    let n = ctx.get_n_prim_unique();
    let (mut u_large, mut u_small) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    ctx.get_picture_change_transforms(u_large.get_data_mut(), u_small.get_data_mut()).unwrap();
    (u_large, u_small)
}

fn sandwich(x: &[f64], u: &Matrix<f64>) -> Vec<f64> {
    let n = u.rows();
    MatrixView::new(n, n, x).congruence_transform(u).get_data().to_vec()
}

fn assert_close(label: &str, x: &[f64], y: &[f64]) {
    assert_eq!(x.len(), y.len(), "{label}");
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    // A plane may vanish identically (the dipole's (p x x p)_x): then both are zero.
    assert!(difference <= 1e-12 * scale.max(1.0), "{label}: {difference} of {scale}");
}

fn max_asymmetry(x: &[f64], n: usize, sign: f64) -> f64 {
    (0..n).flat_map(|j| (0..n).map(move |i| (i, j))).map(|(i, j)| (x[i + n * j] - sign * x[j + n * i]).abs()).fold(0.0, f64::max)
}

#[test]
fn the_spin_fold_is_the_small_component_sandwich_and_keeps_antisymmetry() {
    let ctx = context();
    let n = ctx.get_n_prim_unique();
    let mut raw = vec![0.0; 4 * n * n];
    ctx.compute_one_electron_picture_change_integrals(&mut raw, ctx.get_nuclear_attraction_operator(), true).unwrap();
    let mut folded = vec![0.0; 3 * n * n];
    ctx.transform_one_electron_spin_operator(&raw[..3 * n * n], 3, &mut folded).unwrap();
    let (_, u_small) = transforms(&ctx);
    for k in 0..3 {
        let plane = &folded[k * n * n..(k + 1) * n * n];
        assert_close(&format!("plane {k}"), plane, &sandwich(&raw[k * n * n..(k + 1) * n * n], &u_small));
        assert!(max_asymmetry(plane, n, -1.0) < 1e-14, "plane {k} antisymmetric");
    }
    assert!(ctx.transform_one_electron_spin_operator(&raw[..3 * n * n], 2, &mut folded).is_err(), "sizes are checked");
}

// One operator (`n_components` components): the library's scalar and spin parts against the sandwiches by hand.
fn check_operator(label: &str, operator: impl Fn(&AedelstenContext) -> OneElectronOperator<'_>, n_components: usize) {
    let ctx = context();
    let (n, m) = (ctx.get_n_prim_unique(), ctx.get_n_cont());
    let (u_large, u_small) = transforms(&ctx);
    let mut plain = vec![0.0; n_components * n * n];
    ctx.compute_one_electron_integrals(&mut plain, operator(&ctx), true).unwrap();
    let mut full = vec![0.0; 4 * n_components * n * n];
    ctx.compute_one_electron_picture_change_integrals(&mut full, operator(&ctx), true).unwrap();
    let plane = |data: &[f64], p: usize| data[p * n * n..(p + 1) * n * n].to_vec();

    for contracted in [false, true] {
        let size = if contracted { m } else { n };
        let mut scalar = vec![0.0; n_components * size * size];
        let mut spin = vec![0.0; 3 * n_components * size * size];
        ctx.compute_picture_changed_one_electron_operator(operator(&ctx), contracted, &mut scalar, &mut spin).unwrap();
        let place = |x: Vec<f64>| -> Vec<f64> {
            if !contracted {
                return x;
            }
            let mut out = vec![0.0; m * m];
            ctx.contract_one_electron_matrix(&x, &mut out).unwrap();
            out
        };
        for c in 0..n_components {
            let mut expected = sandwich(&plane(&plain, c), &u_large);
            for (e, s) in expected.iter_mut().zip(sandwich(&plane(&full, 4 * c + 3), &u_small)) {
                *e += s;
            }
            let got = &scalar[c * size * size..(c + 1) * size * size];
            assert_close(&format!("{label} scalar {c}, contracted {contracted}"), got, &place(expected));
            assert!(max_asymmetry(got, size, 1.0) < 1e-12, "{label} scalar {c} symmetric");
            for k in 0..3 {
                let got = &spin[(3 * c + k) * size * size..(3 * c + k + 1) * size * size];
                assert_close(&format!("{label} spin {c} {k}, contracted {contracted}"), got, &place(sandwich(&plane(&full, 4 * c + k), &u_small)));
                assert!(max_asymmetry(got, size, -1.0) < 1e-12, "{label} spin {c} {k} antisymmetric");
            }
        }
        assert!(spin.iter().any(|x| x.abs() > 1e-6) && scalar.iter().any(|x| x.abs() > 1e-6), "{label}: not vacuous");
        assert!(ctx.compute_picture_changed_one_electron_operator(operator(&ctx), contracted, &mut scalar[1..], &mut spin).is_err(), "sizes are checked");
    }
}

#[test]
fn the_nuclear_attraction_in_the_x2c_picture() {
    check_operator("V", |ctx| ctx.get_nuclear_attraction_operator(), 1);
}

#[test]
fn the_dipole_operator_in_the_x2c_picture() {
    check_operator("r", |ctx| ctx.get_moment_operator(MomentOrigin::new([0.1, -0.2, 0.3], 1)), 3);
}
