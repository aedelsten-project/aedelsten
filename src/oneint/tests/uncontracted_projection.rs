// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Projection (overlap) integrals between two basis sets with an uncontracted side (BasisShell<Uncontracted>), against
// the contracted basis it decontracts: transforming the uncontracted side with its contraction matrix C gives the
// projection onto the contracted basis -- general and segmented partners, values and (side b uncontracted, as for AIMP)
// gradients.

mod common;

use arrays::array3d::{Array3D, Array3DTrait};

use arrays::matrix::ColumnFactor;
use arrays::matrix::Factor;
use arrays::matrix::{Matrix, MatrixTrait};
use basis::shell_types::{decontract_general_shells, decontract_segmented_shells, BasisShell};
use oneint::{OneElectronProjectionIntegralGradients, OneElectronProjectionIntegrals, OneIntProjectionWorkStruct};
use basis::contraction::{Contracted, Segmented, ShellKind};

fn project<Sa: ShellKind, Sb: ShellKind>(a: &[BasisShell<Sa>], n_a: usize, b: &[BasisShell<Sb>], n_b: usize) -> Matrix<f64> {
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
    let mut s = Matrix::<f64>::new(n_a, n_b);
    OneIntProjectionWorkStruct::new(a, b, &mut scratch).compute_overlap_integrals(&mut s).unwrap();
    s
}

/// d S_ab / d centre for both centres, as (n_a x n_b) planes.
fn project_gradient<Sa: ShellKind, Sb: ShellKind>(a: &[BasisShell<Sa>], n_a: usize, b: &[BasisShell<Sb>], n_b: usize) -> Vec<Matrix<f64>> {
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
    let mut planes = Vec::new();
    for centre in 0..2 {
        let mut g = Array3D::<f64>::new(n_a, n_b, 3);
        OneIntProjectionWorkStruct::new(a, b, &mut scratch).compute_overlap_gradient_integrals(&mut g, centre).unwrap();
        for axis in 0..3 {
            planes.push(Matrix::new_from_vec(n_a, n_b, g.get_data()[axis * n_a * n_b..(axis + 1) * n_a * n_b].to_vec().into_boxed_slice()));
        }
    }
    planes
}

fn assert_same(label: &str, x: &Matrix<f64>, y: &Matrix<f64>) {
    let scale = y.get_data().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = x.get_data().iter().zip(y.get_data()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-2, "{label}: vacuous");
    assert!(difference < 1e-12 * scale.max(1.0), "{label}: {difference}");
}

#[test]
fn uncontracted_projection_reproduces_the_contracted_projection() {
    let dir = common::temp_path("uncontracted_projection");
    std::fs::create_dir_all(&dir).unwrap();
    let path_a = dir.join("a.inp");
    let path_b = dir.join("b.inp");
    // A segmented set, as the segmented contraction is run on it: x2c-SVPall, on one element and on two.
    std::fs::write(&path_a, "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H2 0.3 -0.2 1.5\nEnd of Basis\n").unwrap();
    std::fs::write(&path_b, "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    He.x2c-SVPall\n    He2 0.3 -0.2 1.5\nEnd of Basis\n").unwrap();
    let (path_a, path_b) = (path_a.to_str().unwrap(), path_b.to_str().unwrap());
    let general_a = basis::setup_basis_system::<Contracted>(path_a).unwrap();
    let segmented_a = basis::setup_basis_system::<Segmented>(path_a).unwrap();
    let general_b = basis::setup_basis_system::<Contracted>(path_b).unwrap();
    let segmented_b = basis::setup_basis_system::<Segmented>(path_b).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    let (uncontracted_a, c_a) = decontract_general_shells(general_a.get_basis_shells(), general_a.get_basis_info()).unwrap();
    let (uncontracted_b, c_b) = decontract_general_shells(general_b.get_basis_shells(), general_b.get_basis_info()).unwrap();
    let (folded_a, c_seg_a) = decontract_segmented_shells(segmented_a.get_basis_shells(), segmented_a.get_basis_info()).unwrap();
    let (ng_a, ng_b) = (c_a.cols(), c_b.cols());
    let (nu_a, nu_b) = (c_a.rows(), c_b.rows());
    let ns_a = c_seg_a.cols();
    let nf_a = c_seg_a.rows();
    let ns_b = segmented_b.get_basis_info().get_n_cont();
    assert!(nu_a > ng_a && nu_b > ng_b, "the bases should be contracted");

    let (ga, gb, sa, sb) = (general_a.get_basis_shells(), general_b.get_basis_shells(), segmented_a.get_basis_shells(), segmented_b.get_basis_shells());
    assert_same("(uncontracted, general)", &c_a.view_transposed().matmul(&project(&uncontracted_a, nu_a, gb, ng_b)), &project(ga, ng_a, gb, ng_b));
    assert_same("(general, uncontracted)", &project(ga, ng_a, &uncontracted_b, nu_b).matmul(&c_b), &project(ga, ng_a, gb, ng_b));
    assert_same("(uncontracted, uncontracted)", &c_a.view_transposed().matmul(&project(&uncontracted_a, nu_a, &uncontracted_b, nu_b)).matmul(&c_b), &project(ga, ng_a, gb, ng_b));
    assert_same("(segmented, uncontracted)", &project(sa, ns_a, &uncontracted_b, nu_b).matmul(&c_b), &project(sa, ns_a, gb, ng_b));
    assert_same("(uncontracted, segmented)", &c_a.view_transposed().matmul(&project(&uncontracted_a, nu_a, sb, ns_b)), &project(ga, ng_a, sb, ns_b));

    // Gradients exist with side b uncontracted (AIMP): side a is checked through its decontraction.
    let uu = project_gradient(&uncontracted_a, nu_a, &uncontracted_b, nu_b);
    let gu = project_gradient(ga, ng_a, &uncontracted_b, nu_b);
    let fu = project_gradient(&folded_a, nf_a, &uncontracted_b, nu_b);
    let su = project_gradient(sa, ns_a, &uncontracted_b, nu_b);
    for k in 0..6 {
        assert_same(&format!("d(general, uncontracted) {k}"), &c_a.view_transposed().matmul(&uu[k]), &gu[k]);
        assert_same(&format!("d(segmented, uncontracted) {k}"), &c_seg_a.view_transposed().matmul(&fu[k]), &su[k]);
    }
}
