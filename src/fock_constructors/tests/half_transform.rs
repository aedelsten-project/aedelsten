// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// HalfTransformedEri: X(p, nu, r, sigma) from the direct walk against a dense transformation of the S1 tensor, on
// general, segmented and screened uncontracted pairs with s, p and d shells: one basis, two different bases, a column
// block of one against the whole, the coefficient screen, and batches of sigma (each shell alone, two halves, all)
// against the dense X's columns.

mod common;

use arrays::array4d::{Array4D, Array4DTrait};
use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use basis::contraction::{Contracted, Segmented, ShellKind};
use basis::shell_pairs::{derive_shell_pairs, ShellPair};
use basis::shell_types::decontract_general_shells;
use eri::{ERIWorkStruct, EriData, TwoElectronIntegrals};
use fock_constructors::HalfTransformedEri;

fn coefficients(n: usize, m: usize, salt: usize) -> Matrix<f64> {
    let mut c = Matrix::<f64>::new(n, m);
    for mu in 0..n {
        for u in 0..m {
            c[(mu, u)] = 0.2 * (((mu * 5 + u * 3 + 1 + salt) % 7) as f64) - 0.55 + 0.03 * (mu as f64) * if (u + salt) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    c
}

/// X(p, nu, r, sigma) = sum_{mu lambda} L_mu,p R_lambda,r g(mu, nu, lambda, sigma), column-major, p fastest.
fn dense_x(g: &[f64], left: &Matrix<f64>, right: &Matrix<f64>) -> Vec<f64> {
    let (n, ml, mr) = (left.rows(), left.cols(), right.cols());
    // H(p, nu, lambda, sigma) = sum_mu L_mu,p g(mu, nu, lambda, sigma)
    let mut h = vec![0.0; ml * n * n * n];
    for nls in 0..n * n * n {
        for p in 0..ml {
            h[p + ml * nls] = (0..n).map(|mu| left[(mu, p)] * g[mu + n * nls]).sum();
        }
    }
    let mut x = vec![0.0; ml * n * mr * n];
    for sigma in 0..n {
        for r in 0..mr {
            for lambda in 0..n {
                let c = right[(lambda, r)];
                for nu in 0..n {
                    for p in 0..ml {
                        x[p + ml * (nu + n * (r + mr * sigma))] += c * h[p + ml * (nu + n * (lambda + n * sigma))];
                    }
                }
            }
        }
    }
    x
}

fn assert_close(label: &str, x: &[f64], y: &[f64], tolerance: f64) {
    assert_eq!(x.len(), y.len(), "{label}");
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-2, "{label}: vacuous comparison");
    assert!(difference < tolerance * scale, "{label}: difference {difference}");
}

/// The largest shell's function count among the pairs' shells: what sizes the accumulator's row buffers.
fn max_shell_functions<S: ShellKind>(pairs: &[ShellPair<S>]) -> usize {
    pairs.iter().map(|p| (p.get_n_cont_a() * p.get_degeneracy_a()).max(p.get_n_cont_b() * p.get_degeneracy_b())).max().unwrap_or(0)
}

/// The walk's X for `left` and `right` (`None`: one basis, `left` on both sides).
fn walk<S: ShellKind>(pairs: &[ShellPair<S>], eri_data: &EriData, scratch: &mut arrays::scratch::Scratch, left: &Matrix<f64>, right: Option<&Matrix<f64>>, threshold: Option<f64>) -> Matrix<f64> {
    let n = left.rows();
    let mr = right.map_or(left.cols(), |r| r.cols());
    let mut x = Matrix::new_from_vec(left.cols() * n, mr * n, vec![f64::NAN; left.cols() * n * mr * n].into_boxed_slice());
    {
        // As the context does: the row buffers off the front of the scratch, the walk in the rest.
        let buffers = scratch.get_arena();
        let mut acc = match right {
            None => HalfTransformedEri::new(left.view(), x.view_mut(), threshold, true, max_shell_functions(pairs), &buffers),
            Some(right) => HalfTransformedEri::mixed(left.view(), right.view(), x.view_mut(), threshold, true, max_shell_functions(pairs), &buffers),
        }
        .unwrap();
        let mut rest = buffers.into_sub_scratch();
        ERIWorkStruct::new(pairs, &mut rest, &aedelsten_mpi::Communicator::single()).compute_fock_direct_from_data(&mut acc, eri_data, None).unwrap();
    }
    x
}

/// The first function of every shell of the pairs, ascending, then `n`.
fn shell_bounds<S: ShellKind>(pairs: &[ShellPair<S>], n: usize) -> Vec<usize> {
    let mut bounds: Vec<usize> = pairs.iter().flat_map(|p| [p.get_idx_a(), p.get_idx_b()]).chain([n]).collect();
    bounds.sort_unstable();
    bounds.dedup();
    bounds
}

/// The walk's X for the sigma of `sigma` only (`right` `None`: one basis): `(m_L n) x (m_R n_sigma)`.
#[allow(clippy::too_many_arguments)]
fn walk_sigma<S: ShellKind>(pairs: &[ShellPair<S>], eri_data: &EriData, scratch: &mut arrays::scratch::Scratch, left: &Matrix<f64>, right: Option<&Matrix<f64>>, sigma: std::ops::Range<usize>, threshold: Option<f64>) -> Matrix<f64> {
    let n = left.rows();
    let mr = right.map_or(left.cols(), |r| r.cols());
    let mut x = Matrix::new_from_vec(left.cols() * n, mr * sigma.len(), vec![f64::NAN; left.cols() * n * mr * sigma.len()].into_boxed_slice());
    {
        let buffers = scratch.get_arena();
        let mut acc = HalfTransformedEri::sigma_batch(left.view(), right.map(|r| r.view()), sigma, x.view_mut(), threshold, true, max_shell_functions(pairs), &buffers).unwrap();
        let mut rest = buffers.into_sub_scratch();
        ERIWorkStruct::new(pairs, &mut rest, &aedelsten_mpi::Communicator::single()).compute_fock_direct_from_data(&mut acc, eri_data, None).unwrap();
    }
    x
}

/// Batches of sigma against the dense X's columns: every shell alone, two halves at a shell boundary, and all of them.
#[allow(clippy::too_many_arguments)]
fn check_sigma<S: ShellKind>(label: &str, pairs: &[ShellPair<S>], eri_data: &EriData, scratch: &mut arrays::scratch::Scratch, g: &[f64], left: &Matrix<f64>, right: Option<&Matrix<f64>>, threshold: Option<f64>) {
    let n = left.rows();
    let reference = dense_x(g, left, right.unwrap_or(left));
    let column = left.cols() * n * right.map_or(left.cols(), |r| r.cols());
    let bounds = shell_bounds(pairs, n);
    let middle = bounds[bounds.len() / 2];
    for batches in [bounds.clone(), vec![0, middle, n], vec![0, n]] {
        for sigma in batches.windows(2).map(|w| w[0]..w[1]) {
            let x = walk_sigma(pairs, eri_data, scratch, left, right, sigma.clone(), threshold);
            assert_close(&format!("{label}, sigma {sigma:?}"), x.get_data(), &reference[column * sigma.start..column * sigma.end], 1e-12);
        }
    }
}

fn check<S: ShellKind>(label: &str, pairs: &[ShellPair<S>], n: usize) {
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
    let mut g = Array4D::<f64>::new(n, n, n, n);
    ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_eri_stored(&mut g).unwrap();
    let eri_data = ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).derive_eri_data(None).unwrap();

    for m in [1, 3] {
        let c = coefficients(n, m, 0);
        let x = walk(pairs, &eri_data, &mut scratch, &c, None, None);
        assert_close(&format!("{label} one basis, m {m}"), x.get_data(), &dense_x(g.get_data(), &c, &c), 1e-12);
    }

    // Two bases of different widths, either way round.
    let (l, r) = (coefficients(n, 2, 1), coefficients(n, 3, 4));
    let x = walk(pairs, &eri_data, &mut scratch, &l, Some(&r), None);
    assert_close(&format!("{label} two bases"), x.get_data(), &dense_x(g.get_data(), &l, &r), 1e-12);
    let x = walk(pairs, &eri_data, &mut scratch, &r, Some(&l), None);
    assert_close(&format!("{label} two bases, swapped"), x.get_data(), &dense_x(g.get_data(), &r, &l), 1e-12);

    // A column block of the orbitals against all of them (MP2's batches); the block as a view of the whole matrix.
    let all = coefficients(n, 4, 2);
    let block = Matrix::new_from_vec(n, 2, all.view_columns(1..3).get_data().into());
    let mut x = Matrix::<f64>::new(2 * n, 4 * n);
    {
        let buffers = scratch.get_arena();
        let mut acc = HalfTransformedEri::mixed(all.view_columns(1..3), all.view(), x.view_mut(), None, true, max_shell_functions(pairs), &buffers).unwrap();
        let mut rest = buffers.into_sub_scratch();
        ERIWorkStruct::new(pairs, &mut rest, &aedelsten_mpi::Communicator::single()).compute_fock_direct_from_data(&mut acc, &eri_data, None).unwrap();
    }
    assert_close(&format!("{label} column block"), x.get_data(), &dense_x(g.get_data(), &block, &all), 1e-12);

    // The screen: orbitals with no weight on the second half of the basis. The batches it skips contribute nothing
    // above the threshold, so X still matches; and so does a mixed walk whose right side has that weight.
    let mut c = coefficients(n, 3, 0);
    for mu in n / 2..n {
        for u in 0..3 {
            c[(mu, u)] = 0.0;
        }
    }
    let x = walk(pairs, &eri_data, &mut scratch, &c, None, Some(1e-14));
    assert_close(&format!("{label} screened, one basis"), x.get_data(), &dense_x(g.get_data(), &c, &c), 1e-12);
    let x = walk(pairs, &eri_data, &mut scratch, &c, Some(&r), Some(1e-14));
    assert_close(&format!("{label} screened, two bases"), x.get_data(), &dense_x(g.get_data(), &c, &r), 1e-12);

    // Batches of sigma: one basis, two bases either way round, and with the screen.
    let one = coefficients(n, 3, 0);
    check_sigma(&format!("{label} one basis"), pairs, &eri_data, &mut scratch, g.get_data(), &one, None, None);
    check_sigma(&format!("{label} two bases"), pairs, &eri_data, &mut scratch, g.get_data(), &l, Some(&r), None);
    check_sigma(&format!("{label} two bases, swapped"), pairs, &eri_data, &mut scratch, g.get_data(), &r, Some(&l), None);
    check_sigma(&format!("{label} screened, one basis"), pairs, &eri_data, &mut scratch, g.get_data(), &c, None, Some(1e-14));
    check_sigma(&format!("{label} screened, two bases"), pairs, &eri_data, &mut scratch, g.get_data(), &c, Some(&r), Some(1e-14));
}

fn write_input(name: &str, content: &str) -> (std::path::PathBuf, String) {
    let dir = common::temp_path(&format!("half_transform_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.inp"));
    std::fs::write(&path, content).unwrap();
    let path_str = path.to_str().unwrap().to_string();
    (dir, path_str)
}

#[test]
fn half_transformation_matches_the_dense_one_water_ano() {
    let (dir, path) = write_input("water", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n");
    // ANO only generally contracted: segmented, its integrals are far too slow.
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let n = general.get_basis_info().get_n_cont();
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), n);
}

#[test]
fn half_transformation_matches_the_dense_one_water_svp_segmented() {
    let (dir, path) = write_input("water_svp", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n");
    let segmented = basis::setup_basis_system::<Segmented>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let n = segmented.get_basis_info().get_n_cont();
    check("segmented", &derive_shell_pairs(segmented.get_basis_shells(), None).unwrap(), n);
}

#[test]
fn half_transformation_matches_the_dense_one_screened_uncontracted_far_hydroxyl() {
    let (dir, path) = write_input("oh_far", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 3.1\nEnd of Basis\n");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let segmented = basis::setup_basis_system::<Segmented>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (uncontracted, c) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    check("screened uncontracted", &derive_shell_pairs(&uncontracted, Some(1e-6)).unwrap(), c.rows());
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), c.cols());
    check("segmented", &derive_shell_pairs(segmented.get_basis_shells(), None).unwrap(), c.cols());
}
