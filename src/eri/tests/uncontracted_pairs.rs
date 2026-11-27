// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Uncontracted (Identity) pairs -- a whole (centre, l) block of primitives per shell, no contraction step -- against
// the contracted basis they decontract: with the contraction matrix C on all four indices, the uncontracted ERIs and
// ERI derivative tensors reproduce the contracted ones. Overlap screening (holes in the uncontracted pairs) is checked
// against the unscreened pairs.

mod common;

use arrays::array4d::{Array4D, Array4DTrait};
use arrays::matrix::{Matrix, MatrixTrait};
use basis::shell_pairs::derive_shell_pairs;
use basis::shell_types::decontract_general_shells;
use eri::packed::{PackedERITensor, PackedERITensorTrait};
use eri::{ERIWorkStruct, TwoElectronIntegralGradients, TwoElectronIntegrals};
use basis::contraction::Contracted;

fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// `(ab|cd) -> sum C_ai C_bj C_ck C_dl (ab|cd)`, one index at a time (column-major, first index fastest).
fn to_contracted(x: &[f64], c: &Matrix<f64>) -> Vec<f64> {
    let (n_prim, n_cont) = (c.rows(), c.cols());
    let mut current = x.to_vec();
    let mut dims = [n_prim; 4];
    for axis in 0..4 {
        let mut next_dims = dims;
        next_dims[axis] = n_cont;
        let stride: usize = dims[..axis].iter().product();
        let outer: usize = dims[axis + 1..].iter().product();
        let mut next = vec![0.0; next_dims.iter().product()];
        for o in 0..outer {
            for k in 0..n_cont {
                for i in 0..stride {
                    let mut sum = 0.0;
                    for p in 0..n_prim {
                        sum += c[(p, k)] * current[i + stride * (p + n_prim * o)];
                    }
                    next[i + stride * (k + n_cont * o)] = sum;
                }
            }
        }
        current = next;
        dims = next_dims;
    }
    current
}

fn assert_close(label: &str, x: &[f64], y: &[f64], tolerance: f64) {
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = max_abs_difference(x, y);
    eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-3, "{label}: vacuous comparison");
    assert!(difference < tolerance * scale.max(1.0), "{label}: {difference}");
}

/// The S1 ERIs, then the S1 derivative tensors (index `3 * atom + axis`) of the debug gradient route, unscreened.
fn eris_and_gradients<W: TwoElectronIntegrals + TwoElectronIntegralGradients>(mut work: W, n: usize, n_atoms: usize) -> Vec<Vec<f64>> {
    let mut eris = Array4D::<f64>::new(n, n, n, n);
    work.compute_eri_stored(&mut eris).unwrap();
    let eri_data = work.derive_eri_data(None).unwrap();
    let mut tensors: Vec<Array4D<f64>> = (0..3 * n_atoms).map(|_| Array4D::new(n, n, n, n)).collect();
    work.compute_eri_gradient(&mut eri::GradientTensors::new(&mut tensors), &eri_data, None, eri::GradientCentres::all(n_atoms)).unwrap();
    std::iter::once(eris.get_data().to_vec()).chain(tensors.iter().map(|t| t.get_data().to_vec())).collect()
}

/// The ANO guard: a generally contracted set, its contracted functions sharing every primitive of a
/// block with coefficients of both signs (H ANO-RCC [2s] from 8s), which a segmented set never gives C.
fn ano_hydrogens(tag: &str) -> (std::path::PathBuf, String) {
    let dir = common::temp_path(tag);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("h3_ano.inp");
    std::fs::write(&path, "Basis set\n    H.ANO-RCC...2s\n    H1 0.0 0.0 0.0\n    H2 0.4 -0.3 1.4\n    H3 0.0 0.0 30.0\nEnd of Basis\n").unwrap();
    let text = path.to_str().unwrap().to_string();
    (dir, text)
}

#[test]
fn uncontracted_pairs_reproduce_the_contracted_eris_and_gradients_ano_rcc() {
    let (dir, path) = ano_hydrogens("uncontracted_eri_ano");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (uncontracted, c) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    let (n_prim, n_cont) = (c.rows(), c.cols());
    assert!(n_prim > n_cont, "the basis should be contracted");
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();

    let contracted_pairs = derive_shell_pairs(general.get_basis_shells(), None).unwrap();
    let expected = eris_and_gradients(ERIWorkStruct::new(&contracted_pairs, &mut scratch, &aedelsten_mpi::Communicator::single()), n_cont, 3);
    let uncontracted_pairs = derive_shell_pairs(&uncontracted, None).unwrap();
    let computed = eris_and_gradients(ERIWorkStruct::new(&uncontracted_pairs, &mut scratch, &aedelsten_mpi::Communicator::single()), n_prim, 3);

    assert_close("ERIs", &to_contracted(&computed[0], &c), &expected[0], 1e-12);
    // H1 and H2; H3, 30 bohr away, has derivatives too small to test anything.
    for component in 0..6 {
        let label = format!("dERI atom {} axis {}", component / 3, component % 3);
        assert_close(&label, &to_contracted(&computed[1 + component], &c), &expected[1 + component], 1e-11);
    }
}

#[test]
fn screened_uncontracted_pairs_match_the_unscreened_ones_far_hydroxyl() {
    let dir = common::temp_path("uncontracted");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("oh_far.inp");
    std::fs::write(&path, "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 3.1\nEnd of Basis\n").unwrap();
    let general = basis::setup_basis_system::<Contracted>(path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (uncontracted, c) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    let n = c.rows();
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();

    let screened_pairs = derive_shell_pairs(&uncontracted, Some(1e-6)).unwrap();
    assert!(screened_pairs.iter().any(|p| !p.get_single_centre() && p.get_contracted_pair_dims() != (p.get_n_cont_a(), p.get_n_cont_b())),
        "the threshold should leave a pair with screened-out products");
    let mut screened = PackedERITensor::<f64>::new(n);
    ERIWorkStruct::new(&screened_pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_eri_stored(&mut screened).unwrap();
    let unscreened_pairs = derive_shell_pairs(&uncontracted, None).unwrap();
    let mut unscreened = PackedERITensor::<f64>::new(n);
    ERIWorkStruct::new(&unscreened_pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_eri_stored(&mut unscreened).unwrap();
    // A misplaced product would be off by O(1); the dropped ones are below the threshold.
    assert_close("screened ERIs", screened.get_data(), unscreened.get_data(), 1e-5);

    let screened_gradient = eris_and_gradients(ERIWorkStruct::new(&screened_pairs, &mut scratch, &aedelsten_mpi::Communicator::single()), n, 2);
    let unscreened_gradient = eris_and_gradients(ERIWorkStruct::new(&unscreened_pairs, &mut scratch, &aedelsten_mpi::Communicator::single()), n, 2);
    for component in 0..6 {
        let label = format!("screened dERI atom {} axis {}", component / 3, component % 3);
        assert_close(&label, &screened_gradient[1 + component], &unscreened_gradient[1 + component], 1e-5);
    }
}
