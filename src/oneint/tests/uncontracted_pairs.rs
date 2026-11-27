// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// One-electron integrals on uncontracted (Identity) pairs -- a whole (centre, l) block of primitives per shell, no
// contraction step -- against the contracted basis they decontract: with the contraction matrix C, C^T X C
// reproduces every contracted matrix, and a density D transformed to C D C^T every density-contracted gradient.
// Overlap screening (holes in the uncontracted pairs) is checked against the unscreened pairs.

mod common;

use arrays::matrix::ColumnFactor;
use arrays::matrix::Factor;
use arrays::matrix::{Matrix, MatrixTrait};
use basis::shell_pairs::derive_shell_pairs;
use basis::shell_types::decontract_general_shells;
use oneint::{OneElectronGradientContractions, OneElectronIntegrals, OneElectronPictureChangeIntegrals, OneIntWorkStruct};
use basis::contraction::Contracted;

/// S, T, V, pVp as (label, n x n plane), and the density-contracted overlap and
/// nuclear gradients.
fn all_integrals<S: basis::contraction::ShellKind>(pairs: &[basis::shell_pairs::ShellPair<S>], n: usize, charges: &[basis::charge_carriers::IndexedPointCharge], density: &Matrix<f64>) -> (Vec<(&'static str, Vec<f64>)>, Vec<(&'static str, Vec<f64>)>)
where
    for<'w> OneIntWorkStruct<'w, S>: OneElectronIntegrals + OneElectronGradientContractions + OneElectronPictureChangeIntegrals,
{
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
    let nuclei: Vec<(usize, usize)> = (0..charges.len()).map(|i| (i, i)).collect();
    let mut work = OneIntWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single());
    let mut planes = Vec::new();
    let mut m = Matrix::<f64>::new(n, n);
    work.compute_overlap_integrals(&mut m).unwrap();
    planes.push(("S", m.get_data().to_vec()));
    let mut m = Matrix::<f64>::new(n, n);
    work.compute_kinetic_integrals(&mut m).unwrap();
    planes.push(("T", m.get_data().to_vec()));
    let mut m = Matrix::<f64>::new(n, n);
    work.compute_point_charge_integrals(&mut m, charges).unwrap();
    planes.push(("V", m.get_data().to_vec()));
    let mut m = Matrix::<f64>::new(n, n);
    work.compute_pvp_scalar_point_charge_integrals(&mut m, charges).unwrap();
    planes.push(("pVp", m.get_data().to_vec()));
    let mut contracted = Vec::new();
    let mut g = Matrix::<f64>::new(3, charges.len());
    work.contract_overlap_gradient(&mut g, density, None).unwrap();
    contracted.push(("D.dS/dX", g.get_data().to_vec()));
    let mut g = Matrix::<f64>::new(3, charges.len());
    work.contract_point_charge_gradient(&mut g, density, charges, &nuclei, None).unwrap();
    contracted.push(("D.dV/dX", g.get_data().to_vec()));
    (planes, contracted)
}

/// `C^T X C` for an `n_prim x n_prim` column-major `x`.
fn to_contracted(x: &[f64], c: &Matrix<f64>) -> Vec<f64> {
    let n_prim = c.rows();
    let x = Matrix::<f64>::new_from_vec(n_prim, n_prim, x.to_vec().into_boxed_slice());
    c.view_transposed().matmul(&x).matmul(c).get_data().to_vec()
}

fn assert_close(label: &str, x: &[f64], y: &[f64], tolerance: f64) {
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-3, "{label}: vacuous comparison");
    assert!(difference < tolerance * scale.max(1.0), "{label}: {difference}");
}

fn point_charges(input: &basis::InputStructure<Contracted>) -> Vec<basis::charge_carriers::IndexedPointCharge> {
    match input.get_nuclear_potential() {
        Some(basis::charge_carriers::NuclearPotential::PointCharge(charges)) => charges.to_vec(),
        _ => panic!("expected point-charge nuclei"),
    }
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
fn uncontracted_pairs_reproduce_the_contracted_one_electron_integrals_ano_rcc() {
    let (dir, path) = ano_hydrogens("uncontracted_oneint_ano");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let charges = point_charges(&general);
    let (uncontracted, c) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    let (n_prim, n_cont) = (c.rows(), c.cols());
    assert!(n_prim > n_cont, "the basis should be contracted");

    let mut density = Matrix::<f64>::new(n_cont, n_cont);
    for i in 0..n_cont {
        for j in i..n_cont {
            let v = 0.01 * (1 + (7 * i + 3 * j) % 11) as f64;
            density[(i, j)] = v;
            density[(j, i)] = v;
        }
    }
    let density_prim = c.matmul(&density).matmul(&c.view_transposed());

    let (contracted_planes, contracted_gradients) = all_integrals(&derive_shell_pairs(general.get_basis_shells(), None).unwrap(), n_cont, &charges, &density);
    let (planes, gradients) = all_integrals(&derive_shell_pairs(&uncontracted, None).unwrap(), n_prim, &charges, &density_prim);
    for ((label, x), (_, expected)) in planes.iter().zip(&contracted_planes) {
        assert_close(label, &to_contracted(x, &c), expected, 1e-11);
    }
    for ((label, x), (_, expected)) in gradients.iter().zip(&contracted_gradients) {
        assert_close(label, x, expected, 1e-11);
    }
}

#[test]
fn screened_uncontracted_pairs_match_the_unscreened_ones_far_hydroxyl() {
    let dir = common::temp_path("oneint_uncontracted");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("oh_far.inp");
    std::fs::write(&path, "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 3.1\nEnd of Basis\n").unwrap();
    let general = basis::setup_basis_system::<Contracted>(path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let charges = point_charges(&general);
    let (uncontracted, c) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    let n = c.rows();
    let mut density = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in i..n {
            let v = 0.01 * (1 + (7 * i + 3 * j) % 11) as f64;
            density[(i, j)] = v;
            density[(j, i)] = v;
        }
    }
    // 1e-8: a product dropped at 1e-6 still moves the nuclear-attraction derivative near O by 6e-5.
    let screened_pairs = derive_shell_pairs(&uncontracted, Some(1e-8)).unwrap();
    assert!(screened_pairs.iter().any(|p| {
        !p.get_single_centre() && p.get_contracted_pair_dims() != (p.get_n_cont_a(), p.get_n_cont_b())
    }), "the threshold should leave a pair with screened-out products");
    let (unscreened, unscreened_gradients) = all_integrals(&derive_shell_pairs(&uncontracted, None).unwrap(), n, &charges, &density);
    let (screened, screened_gradients) = all_integrals(&screened_pairs, n, &charges, &density);
    // A misplaced product would be off by O(1); the dropped ones are below the threshold.
    for ((label, x), (_, expected)) in screened.iter().chain(&screened_gradients).zip(unscreened.iter().chain(&unscreened_gradients)) {
        assert_close(label, x, expected, 1e-5);
    }
}
