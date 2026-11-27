// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The direct J/K accumulators (block digestion, digest.rs) against the dense S1 builders, on general, segmented and
// screened uncontracted pairs with s, p and d shells: single-centre slots, diagonal batches, (xx|ss) and (ss|ss)
// batches.

mod common;

use arrays::array4d::{Array4D, Array4DTrait};
use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use basis::contraction::{Contracted, Segmented, ShellKind};
use basis::shell_pairs::{derive_shell_pairs, ShellPair};
use basis::shell_types::decontract_general_shells;
use eri::{ERIWorkStruct, TwoElectronIntegrals};
use fock_constructors::{
    construct_j_stored, construct_k_stored, InactiveActiveDirectFock, RestrictedDirectFock,
    UnrestrictedDirectFock,
};

fn density(n: usize, salt: usize, symmetric: bool) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            p[(i, j)] = 0.3 * (((i * 7 + j * 3 + salt) % 11) as f64) - 1.1 + 0.07 * (i as f64) - 0.13 * (j as f64);
        }
    }
    if symmetric {
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (p[(i, j)] + p[(j, i)]);
                p[(i, j)] = v;
                p[(j, i)] = v;
            }
        }
    }
    p
}

fn sum(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
    let n = a.rows();
    let mut c = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            c[(i, j)] = a[(i, j)] + b[(i, j)];
        }
    }
    c
}

fn assert_close(label: &str, x: &Matrix<f64>, y: &Matrix<f64>) {
    let scale = y.get_data().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = x.get_data().iter().zip(y.get_data()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-2, "{label}: vacuous comparison");
    assert!(difference < 1e-11 * scale, "{label}: difference {difference}");
}

struct Reference {
    g: Array4D<f64>,
}

impl Reference {
    fn j(&self, p: &Matrix<f64>, coeff: f64) -> Matrix<f64> {
        let mut j = Matrix::<f64>::new(p.rows(), p.rows());
        construct_j_stored(&self.g, p, &mut j, coeff);
        j
    }
    fn k(&self, p: &Matrix<f64>, coeff: f64) -> Matrix<f64> {
        let mut k = Matrix::<f64>::new(p.rows(), p.rows());
        construct_k_stored(&self.g, p, &mut k, coeff);
        k
    }
}

fn check<S: ShellKind>(label: &str, pairs: &[ShellPair<S>], n: usize) {
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
    let mut g = Array4D::<f64>::new(n, n, n, n);
    ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_eri_stored(&mut g).unwrap();
    assert!(g.get_data().iter().any(|v| v.abs() > 1e-2), "{label}: vacuous ERIs");
    let reference = Reference { g };
    let eri_data = ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).derive_eri_data(None).unwrap();
    let mut direct = |output: &mut dyn FnMut(&mut ERIWorkStruct<'_, S>)| output(&mut ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()));
    let new = || Matrix::<f64>::new(n, n);

    let (p_a, p_b) = (density(n, 0, true), density(n, 5, true));
    let p_total = sum(&p_a, &p_b);

    // Restricted.
    let (mut j, mut k) = (new(), new());
    direct(&mut |w| w.compute_fock_direct_from_data(&mut RestrictedDirectFock::new(p_a.view(), j.view_mut(), k.view_mut(), 1.0, -0.5), &eri_data, None).unwrap());
    assert_close(&format!("{label} restricted J"), &j, &reference.j(&p_a, 1.0));
    assert_close(&format!("{label} restricted K"), &k, &reference.k(&p_a, -0.5));

    // Unrestricted.
    let (mut j, mut k_a, mut k_b) = (new(), new(), new());
    direct(&mut |w| {
        let mut acc = UnrestrictedDirectFock::new(p_a.view(), p_b.view(), p_total.view(), j.view_mut(), k_a.view_mut(), k_b.view_mut(), 1.0, -1.0);
        w.compute_fock_direct_from_data(&mut acc, &eri_data, None).unwrap()
    });
    assert_close(&format!("{label} unrestricted J"), &j, &reference.j(&p_total, 1.0));
    assert_close(&format!("{label} unrestricted K_alpha"), &k_a, &reference.k(&p_a, -1.0));
    assert_close(&format!("{label} unrestricted K_beta"), &k_b, &reference.k(&p_b, -1.0));

    // Inactive/active, with different coefficients per density.
    let (mut j_i, mut k_i, mut j_a, mut k_a) = (new(), new(), new(), new());
    direct(&mut |w| {
        let mut acc = InactiveActiveDirectFock::new(p_a.view(), p_b.view(), j_i.view_mut(), k_i.view_mut(), j_a.view_mut(), k_a.view_mut(), 1.0, -0.5, 0.7, -0.3);
        w.compute_fock_direct_from_data(&mut acc, &eri_data, None).unwrap()
    });
    assert_close(&format!("{label} inactive J"), &j_i, &reference.j(&p_a, 1.0));
    assert_close(&format!("{label} inactive K"), &k_i, &reference.k(&p_a, -0.5));
    assert_close(&format!("{label} active J"), &j_a, &reference.j(&p_b, 0.7));
    assert_close(&format!("{label} active K"), &k_a, &reference.k(&p_b, -0.3));
}

fn write_input(name: &str, content: &str) -> (std::path::PathBuf, String) {
    let dir = common::temp_path(&format!("block_digestion_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.inp"));
    std::fs::write(&path, content).unwrap();
    let path_str = path.to_str().unwrap().to_string();
    (dir, path_str)
}

#[test]
fn block_digestion_matches_the_dense_builders_water_ano() {
    let (dir, path) = write_input("water", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let n = general.get_basis_info().get_n_cont();
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), n);
}

#[test]
fn block_digestion_matches_the_dense_builders_screened_uncontracted_far_hydroxyl() {
    let (dir, path) = write_input("oh_far", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 3.1\nEnd of Basis\n");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let segmented = basis::setup_basis_system::<Segmented>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (uncontracted, c) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    check("screened uncontracted", &derive_shell_pairs(&uncontracted, Some(1e-6)).unwrap(), c.rows());
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), c.cols());
    check("segmented", &derive_shell_pairs(segmented.get_basis_shells(), None).unwrap(), c.cols());
}
