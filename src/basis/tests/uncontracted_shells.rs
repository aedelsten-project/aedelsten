// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// BasisShell<Uncontracted> (one shell per (centre, l) block, every exponent its own AO) against the contracted basis
// it decontracts: contracting its AO amplitudes and gradients with the contraction matrix gives the contracted ones.

use arrays::matrix::MatrixTrait;
use basis::contraction::ShellKind;
use basis::shell_types::{decontract_general_shells, BasisShell};
use basis::contraction::Contracted;

/// Every AO's amplitude and gradient at `point`, placed by the shells' basis indices.
fn amplitudes<S: ShellKind>(shells: &[BasisShell<S>], n: usize, point: [f64; 3]) -> [Vec<f64>; 4] {
    let mut out = [vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]];
    for shell in shells {
        let first = shell.get_basis_index();
        let value = shell.compute_ao_amplitude(point[0], point[1], point[2]);
        let gradient = shell.compute_ao_amplitude_gradient(point[0], point[1], point[2]);
        for (k, v) in value.iter().enumerate() {
            out[0][first + k] = *v;
            for axis in 0..3 {
                out[1 + axis][first + k] = gradient[axis][k];
            }
        }
    }
    out
}

#[test]
fn uncontracted_shells_reproduce_the_contracted_amplitudes_through_the_contraction_matrix() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_fixtures/eri_cache_test.inp");
    let general = basis::setup_basis_system::<Contracted>(path).unwrap();
    let (uncontracted, contraction) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    let (n_prim, n_cont) = (contraction.rows(), contraction.cols());
    assert!(n_prim > n_cont, "the basis should be contracted");
    assert!(uncontracted.len() < n_prim, "uncontracted shells should be blocks of exponents");

    for point in [[0.1, -0.3, 0.7], [0.5, 0.2, 1.1], [-1.2, 0.4, 29.5]] {
        let u = amplitudes(&uncontracted, n_prim, point);
        let c = amplitudes(general.get_basis_shells(), n_cont, point);
        for component in 0..4 {
            for j in 0..n_cont {
                let transformed: f64 = (0..n_prim).map(|i| contraction[(i, j)] * u[component][i]).sum();
                let expected = c[component][j];
                assert!((transformed - expected).abs() <= 1e-12 * expected.abs().max(1.0), "point {point:?} component {component} AO {j}: {transformed} against {expected}");
            }
        }
    }
}
