// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// ActiveSpaceDirectFock: the Coulomb matrices of the active pair densities M(vw, mu nu) from the direct walk, and
// T(mu, u, vw) made from them, against dense transformations of the S1 tensor, and its inactive/active J/K against
// the dense builders, on general, segmented and screened uncontracted pairs with s, p and d shells.

mod common;

use arrays::array4d::{Array4D, Array4DTrait};
use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut, MatrixViewMut};
use basis::contraction::{Contracted, Segmented, ShellKind};
use basis::shell_pairs::{derive_shell_pairs, ShellPair};
use basis::shell_types::decontract_general_shells;
use eri::packed::{packed_pair_tensor_len, PackedPairTensor, PackedPairTensorStruct, PackedPairTensorViewMut};
use eri::{ERIWorkStruct, TwoElectronIntegrals};
use fock_constructors::{construct_j_stored, construct_k_stored, ActiveSpaceDirectFock, InactiveActiveDirectFock};

fn coefficients(n: usize, m: usize) -> Matrix<f64> {
    let mut c = Matrix::<f64>::new(n, m);
    for mu in 0..n {
        for u in 0..m {
            c[(mu, u)] = 0.2 * (((mu * 5 + u * 3 + 1) % 7) as f64) - 0.55 + 0.03 * (mu as f64) * if u % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    c
}

fn symmetric_density(n: usize, salt: usize) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..=i {
            let v = 0.1 + 0.05 * (((i * 7 + j * 3 + salt) % 11) as f64);
            p[(i, j)] = v;
            p[(j, i)] = v;
        }
    }
    p
}

/// T(mu, u, vw) = sum_{nu lambda sigma} C_nu,u P_{lambda sigma, vw} (mu nu|lambda sigma), from the S1 tensor, stored
/// as the accumulator stores it: (u, vw, mu).
fn dense_t(g: &[f64], c: &Matrix<f64>) -> Vec<f64> {
    let (n, m) = (c.rows(), c.cols());
    let q = m * (m + 1) / 2;
    // H(mu, u, lambda, sigma) = sum_nu C_nu,u g(mu, nu, lambda, sigma)
    let mut h = vec![0.0; n * m * n * n];
    for ls in 0..n * n {
        for u in 0..m {
            for nu in 0..n {
                let c_nu = c[(nu, u)];
                for mu in 0..n {
                    h[mu + n * (u + m * ls)] += c_nu * g[mu + n * (nu + n * ls)];
                }
            }
        }
    }
    let mut t = vec![0.0; n * m * q];
    for v in 0..m {
        for w in 0..=v {
            let vw = v * (v + 1) / 2 + w;
            for sigma in 0..n {
                for lambda in 0..n {
                    let p = 0.5 * (c[(lambda, v)] * c[(sigma, w)] + c[(sigma, v)] * c[(lambda, w)]);
                    let ls = lambda + n * sigma;
                    for u in 0..m {
                        for mu in 0..n {
                            t[u + m * (vw + q * mu)] += p * h[mu + n * (u + m * ls)];
                        }
                    }
                }
            }
        }
    }
    t
}

/// M(vw, mu nu) = sum_{lambda sigma} P_{lambda sigma, vw} (mu nu|lambda sigma), from the S1 tensor, packed as the
/// accumulator packs it.
fn dense_m(g: &[f64], c: &Matrix<f64>) -> Vec<f64> {
    let (n, m) = (c.rows(), c.cols());
    let q = m * (m + 1) / 2;
    let mut packed = PackedPairTensor::<f64>::new(n, q);
    for nu in 0..n {
        for mu in nu..n {
            for v in 0..m {
                for w in 0..=v {
                    let mut value = 0.0;
                    for sigma in 0..n {
                        for lambda in 0..n {
                            let p = 0.5 * (c[(lambda, v)] * c[(sigma, w)] + c[(sigma, v)] * c[(lambda, w)]);
                            value += p * g[mu + n * (nu + n * (lambda + n * sigma))];
                        }
                    }
                    packed.pair_mut(mu, nu)[v * (v + 1) / 2 + w] = value;
                }
            }
        }
    }
    packed.get_data().to_vec()
}

/// T from M (`transform_one_index_prealloc`), `m x q n`.
fn t_of<D: arrays::matrix::DataContainer<T = f64>>(packed: &PackedPairTensorStruct<D>, c: &Matrix<f64>) -> Matrix<f64> {
    let mut t = Matrix::<f64>::new(c.cols(), packed.n_ket() * packed.n());
    packed.transform_one_index_prealloc(c, &mut t.view_mut());
    t
}

fn assert_close(label: &str, x: &[f64], y: &[f64], tolerance: f64) {
    assert_eq!(x.len(), y.len(), "{label}");
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-2, "{label}: vacuous comparison");
    assert!(difference < tolerance * scale, "{label}: difference {difference}");
}

fn check<S: ShellKind>(label: &str, pairs: &[ShellPair<S>], n: usize) {
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
    let mut g = Array4D::<f64>::new(n, n, n, n);
    ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_eri_stored(&mut g).unwrap();
    let eri_data = ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).derive_eri_data(None).unwrap();

    for m in [1, 3] {
        let c = coefficients(n, m);
        let t_ref = dense_t(g.get_data(), &c);
        let (p_i, p_a) = (symmetric_density(n, 0), symmetric_density(n, 4));
        let new = || Matrix::<f64>::new(n, n);
        let (mut j_i, mut k_i, mut j_a, mut k_a) = (new(), new(), new(), new());
        let q = m * (m + 1) / 2;
        let mut coulomb = vec![f64::NAN; packed_pair_tensor_len(n, q)];
        {
            let jk = InactiveActiveDirectFock::new(p_i.view(), p_a.view(), j_i.view_mut(), k_i.view_mut(), j_a.view_mut(), k_a.view_mut(), 1.0, -0.5, 1.0, -0.5);
            let mut acc = ActiveSpaceDirectFock::new(jk, c.view(), PackedPairTensorViewMut::new(n, q, &mut coulomb), None);
            ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_fock_direct_from_data(&mut acc, &eri_data, None).unwrap();
        }
        assert_close(&format!("{label} n_act {m} M"), &coulomb, &dense_m(g.get_data(), &c), 1e-12);
        let t = t_of(&eri::packed::PackedPairTensorView::new(n, q, &coulomb), &c);
        assert_close(&format!("{label} n_act {m} T"), t.get_data(), &t_ref, 1e-12);

        let (mut j_ref, mut k_ref) = (new(), new());
        construct_j_stored(&g, &p_a, &mut j_ref, 1.0);
        construct_k_stored(&g, &p_a, &mut k_ref, -0.5);
        assert_close(&format!("{label} n_act {m} active J"), j_a.get_data(), j_ref.get_data(), 1e-12);
        assert_close(&format!("{label} n_act {m} active K"), k_a.get_data(), k_ref.get_data(), 1e-12);
        construct_k_stored(&g, &p_i, &mut k_ref, -0.5);
        assert_close(&format!("{label} n_act {m} inactive K"), k_i.get_data(), k_ref.get_data(), 1e-12);
    }

    // The active screen: orbitals with no weight on the second half of the basis, and a threshold. The batches it
    // skips contribute nothing to T above the threshold (M, which they would have added to where the coefficients
    // vanish, is complete only for T), so T still matches.
    let mut c = coefficients(n, 3);
    for mu in n / 2..n {
        for u in 0..3 {
            c[(mu, u)] = 0.0;
        }
    }
    let t_ref = dense_t(g.get_data(), &c);
    let p = symmetric_density(n, 0);
    let new = || Matrix::<f64>::new(n, n);
    let (mut j_i, mut k_i, mut j_a, mut k_a) = (new(), new(), new(), new());
    let mut coulomb = PackedPairTensor::<f64>::new(n, 6);
    {
        let jk = InactiveActiveDirectFock::new(p.view(), p.view(), j_i.view_mut(), k_i.view_mut(), j_a.view_mut(), k_a.view_mut(), 1.0, -0.5, 1.0, -0.5);
        let mut acc = ActiveSpaceDirectFock::new(jk, c.view(), PackedPairTensorViewMut::new(n, 6, coulomb.get_data_mut()), Some(1e-14));
        ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_fock_direct_from_data(&mut acc, &eri_data, None).unwrap();
    }
    assert_close(&format!("{label} screened T"), t_of(&coulomb, &c).get_data(), &t_ref, 1e-12);
}

fn write_input(name: &str, content: &str) -> (std::path::PathBuf, String) {
    let dir = common::temp_path(&format!("active_space_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.inp"));
    std::fs::write(&path, content).unwrap();
    let path_str = path.to_str().unwrap().to_string();
    (dir, path_str)
}

#[test]
fn active_transformation_matches_the_dense_one_water_ano() {
    let (dir, path) = write_input("water", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let n = general.get_basis_info().get_n_cont();
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), n);
}

#[test]
fn active_transformation_matches_the_dense_one_screened_uncontracted_far_hydroxyl() {
    let (dir, path) = write_input("oh_far", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 3.1\nEnd of Basis\n");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let segmented = basis::setup_basis_system::<Segmented>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (uncontracted, c) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    check("screened uncontracted", &derive_shell_pairs(&uncontracted, Some(1e-6)).unwrap(), c.rows());
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), c.cols());
    check("segmented", &derive_shell_pairs(segmented.get_basis_shells(), None).unwrap(), c.cols());
}

// The accumulators on views: the densities and the J/K outputs from an arena, the active coefficients as a column block
// of the whole orbital matrix, `M` from the arena too. The walk is the same, so the results equal the owned ones
// exactly.
#[test]
fn views_from_an_arena_and_a_column_block_give_the_owned_results() {
    let (dir, path) = write_input("views", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let n = general.get_basis_info().get_n_cont();
    let pairs = derive_shell_pairs(general.get_basis_shells(), None).unwrap();
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
    let eri_data = ERIWorkStruct::new(&pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).derive_eri_data(None).unwrap();

    let (n_inactive, m) = (2, 3);
    let q = m * (m + 1) / 2;
    let orbitals = coefficients(n, n);
    let c_active = Matrix::new_from_vec(n, m, orbitals.view_columns(n_inactive..n_inactive + m).get_data().into());
    let (p_i, p_a) = (symmetric_density(n, 0), symmetric_density(n, 4));

    // Owned.
    let new = || Matrix::<f64>::new(n, n);
    let (mut j_i, mut k_i, mut j_a, mut k_a) = (new(), new(), new(), new());
    let mut coulomb = PackedPairTensor::<f64>::new(n, q);
    {
        let jk = InactiveActiveDirectFock::new(p_i.view(), p_a.view(), j_i.view_mut(), k_i.view_mut(), j_a.view_mut(), k_a.view_mut(), 1.0, -0.5, 1.0, -0.5);
        let mut acc = ActiveSpaceDirectFock::new(jk, c_active.view(), PackedPairTensorViewMut::new(n, q, coulomb.get_data_mut()), Some(1e-14));
        ERIWorkStruct::new(&pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_fock_direct_from_data(&mut acc, &eri_data, None).unwrap();
    }

    // Views: the buffers in an arena of their own (the walk has the scratch's), filled with garbage.
    let mut stage = arrays::scratch::Scratch::new(1, arrays::scratch::ByteUnit::MiB).unwrap();
    let arena = stage.get_arena();
    let garbage = |rows, cols| MatrixViewMut::new(rows, cols, arena.alloc_with(rows * cols, |_| f64::NAN).unwrap());
    let mut densities = MatrixViewMut::new(n, 2 * n, arena.alloc(2 * n * n).unwrap());
    densities.view_columns_mut(0..n).get_data_mut().copy_from_slice(p_i.get_data());
    densities.view_columns_mut(n..2 * n).get_data_mut().copy_from_slice(p_a.get_data());
    let (mut vj_i, mut vk_i, mut vj_a, mut vk_a) = (garbage(n, n), garbage(n, n), garbage(n, n), garbage(n, n));
    let mut vm = PackedPairTensorViewMut::new(n, q, arena.alloc_with(packed_pair_tensor_len(n, q), |_| f64::NAN).unwrap());
    {
        let jk = InactiveActiveDirectFock::new(
            densities.view_columns(0..n),
            densities.view_columns(n..2 * n),
            vj_i.view_mut(),
            vk_i.view_mut(),
            vj_a.view_mut(),
            vk_a.view_mut(),
            1.0,
            -0.5,
            1.0,
            -0.5,
        );
        let mut acc = ActiveSpaceDirectFock::new(jk, orbitals.view_columns(n_inactive..n_inactive + m), PackedPairTensorViewMut::new(n, q, vm.get_data_mut()), Some(1e-14));
        ERIWorkStruct::new(&pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_fock_direct_from_data(&mut acc, &eri_data, None).unwrap();
    }

    for (label, view, owned) in [("J_i", &vj_i, &j_i), ("K_i", &vk_i, &k_i), ("J_a", &vj_a, &j_a), ("K_a", &vk_a, &k_a)] {
        assert_eq!(view.get_data(), owned.get_data(), "{label}");
    }
    assert_eq!(vm.get_data(), coulomb.get_data(), "M");
}
