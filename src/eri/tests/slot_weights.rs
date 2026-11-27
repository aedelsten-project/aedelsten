// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The block hook's slot walk (`for_each_slot_quartet`): every stored element applied with all 8 of its
// permutations and the slot weight must give each AO quartet exactly once. Checked against the unique-integral walk
// (`distribute`, the default `distribute_block`) for the count of each quartet, and against the S1 tensor for the
// values, on general, segmented and screened uncontracted pairs (single-centre slots, diagonal batches, s/p/d).

mod common;

use arrays::array4d::{Array4D, Array4DTrait};
use basis::contraction::{Contracted, Segmented, ShellKind};
use basis::shell_pairs::{derive_shell_pairs, ShellPair};
use arrays::matrix::MatrixTrait;
use basis::shell_types::decontract_general_shells;
use eri::{for_each_slot_quartet, EriBlock, ERIWorkStruct, TensorChoice, TwoElectronIntegrals};

fn permutations(m: usize, n: usize, l: usize, s: usize) -> [(usize, usize, usize, usize); 8] {
    [(m, n, l, s), (m, n, s, l), (n, m, l, s), (n, m, s, l), (l, s, m, n), (s, l, m, n), (l, s, n, m), (s, l, n, m)]
}

/// The unique walk: each `distribute` call adds 1 to every distinct permutation of its quartet.
struct UniqueCounts {
    n: usize,
    counts: Vec<f64>,
}

impl TensorChoice for UniqueCounts {
    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, _val: f64) {
        let mut perms = permutations(i, j, k, l);
        perms.sort_unstable();
        let n = self.n;
        for (idx, &(m, nu, la, s)) in perms.iter().enumerate() {
            if idx == 0 || perms[idx - 1] != (m, nu, la, s) {
                self.counts[m + n * (nu + n * (la + n * s))] += 1.0;
            }
        }
    }
}

/// The block walk: every stored element, all 8 permutations, weighted.
struct BlockSums {
    n: usize,
    values: Vec<f64>,
    counts: Vec<f64>,
}

impl TensorChoice for BlockSums {
    fn distribute(&mut self, _: usize, _: usize, _: usize, _: usize, _: f64) {
        unreachable!("distribute_block is overridden");
    }

    fn distribute_block<S: ShellKind>(&mut self, block: &EriBlock<'_, '_>, pair_ab: &ShellPair<S>, pair_cd: &ShellPair<S>, _scratch: &mut arrays::scratch::Arena<'_>) -> Result<(), arrays::scratch::ArenaError> {
        let [n_i, n_j, n_k, n_l, ..] = block.dims();
        let data = block.data();
        let n = self.n;
        for_each_slot_quartet(block, pair_ab, pair_cd, |q| {
            for l in 0..n_l {
                for k in 0..n_k {
                    for j in 0..n_j {
                        for i in 0..n_i {
                            let v = data[q.offset + i + n_i * (j + n_j * (k + n_k * l))];
                            let [a, b, c, d] = q.functions;
                            for (m, nu, la, s) in permutations(a + i, b + j, c + k, d + l) {
                                let idx = m + n * (nu + n * (la + n * s));
                                self.values[idx] += q.weight * v;
                                self.counts[idx] += q.weight;
                            }
                        }
                    }
                }
            }
        });
        Ok(())
    }
}

fn check<S: ShellKind>(label: &str, pairs: &[ShellPair<S>], n: usize) {
    let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
    let mut reference = Array4D::<f64>::new(n, n, n, n);
    ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_eri_stored(&mut reference).unwrap();
    let mut unique = UniqueCounts { n, counts: vec![0.0; n.pow(4)] };
    ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_eri_stored(&mut unique).unwrap();
    let mut block = BlockSums { n, values: vec![0.0; n.pow(4)], counts: vec![0.0; n.pow(4)] };
    ERIWorkStruct::new(pairs, &mut scratch, &aedelsten_mpi::Communicator::single()).compute_eri_stored(&mut block).unwrap();

    assert!(unique.counts.iter().any(|&c| c == 1.0), "{label}: vacuous");
    for (idx, (&u, &b)) in unique.counts.iter().zip(&block.counts).enumerate() {
        assert!(u <= 1.0, "{label}: the unique walk reached quartet {idx} {u} times");
        assert!((u - b).abs() < 1e-14, "{label}: quartet {idx} counted {b} by the block walk, {u} by the unique walk");
    }
    let scale = reference.get_data().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = reference.get_data().iter().zip(&block.values).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    eprintln!("{label}: n {n}, max {scale:.3e}, difference {difference:.3e}");
    assert!(scale > 1e-2 && difference < 1e-12 * scale, "{label}: {difference}");
}

fn write_input(name: &str, content: &str) -> (std::path::PathBuf, String) {
    let dir = common::temp_path(&format!("slot_weights_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.inp"));
    std::fs::write(&path, content).unwrap();
    let path_str = path.to_str().unwrap().to_string();
    (dir, path_str)
}

#[test]
fn slot_weights_reproduce_the_unique_walk_water_ano() {
    let (dir, path) = write_input("water", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    let n = general.get_basis_info().get_n_cont();
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), n);
}

#[test]
fn slot_weights_reproduce_the_unique_walk_screened_uncontracted_far_hydroxyl() {
    let (dir, path) = write_input("oh_far", "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 3.1\nEnd of Basis\n");
    let general = basis::setup_basis_system::<Contracted>(&path).unwrap();
    let segmented = basis::setup_basis_system::<Segmented>(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (uncontracted, c) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();

    let screened = derive_shell_pairs(&uncontracted, Some(1e-6)).unwrap();
    assert!(screened.iter().any(|p| !p.get_single_centre() && p.get_contracted_pair_dims() != (p.get_n_cont_a(), p.get_n_cont_b())),
        "the threshold should leave a pair with screened-out products");
    check("screened uncontracted", &screened, c.rows());
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), c.cols());
    check("segmented", &derive_shell_pairs(segmented.get_basis_shells(), None).unwrap(), c.cols());
}
