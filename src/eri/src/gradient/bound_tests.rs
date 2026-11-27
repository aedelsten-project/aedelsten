// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! `derivative_schwarz_bounds` against the derivatives themselves: for every quartet of a small molecule, the largest
//! derivative of any of its integrals with respect to any atom is at most `Q'_ab Q_cd + Q_ab Q'_cd`.

use aedelsten_mpi::Communicator;
use arrays::scratch::{Scratch, ByteUnit};
use basis::contraction::{Contracted, Segmented, ShellKind};
use basis::shell_pairs::{derive_shell_pairs, ShellPair};
use basis::shell_types::decontract_general_shells;
use mmd::hermite_coulomb_integrals::FullRange;

use super::second_order::{derivative_bound_from_full_blocks, derivative_schwarz_bounds};
use super::{compute_quartet, pair_data, pair_data_to_order, BraRows, GradientCentres, GradientSink, IntegralDerivatives};
use crate::{ERIWorkStruct, TwoElectronIntegrals};

/// The largest magnitude of any derivative the quartet emits.
struct Largest(f64);

impl GradientSink for Largest {
    fn select_atoms(&mut self, _atoms: &[usize]) {}

    fn distribute(&mut self, _i: usize, _j: usize, _k: usize, _l: usize, derivatives: IntegralDerivatives<'_>) {
        for q in 0..derivatives.len() {
            self.0 = self.0.max(derivatives.get(q).abs());
        }
    }
}

/// Every quartet of `pairs` against the bounds; returns the largest ratio of a quartet's derivative to its bound.
fn check<S: ShellKind>(label: &str, pairs: &[ShellPair<S>], n_centres: usize) -> f64 {
    let comm = Communicator::single();
    let mut scratch = Scratch::new(256, ByteUnit::MiB).unwrap();
    let eri_data = ERIWorkStruct::new(pairs, &mut scratch, &comm).derive_eri_data(None).unwrap();
    let schwarz: Vec<f64> = (0..pairs.len())
        .map(|index| eri_data.entries().iter().find(|entry| entry.original_index == index).unwrap().schwarz_bound)
        .collect();
    let mut arena = scratch.get_arena();
    let data: Vec<_> = pairs.iter().map(|pair| pair_data(pair, &mut arena).unwrap()).collect();
    let derivative = derivative_schwarz_bounds::<FullRange, S>(pairs.iter().zip(&data), &mut arena, 0.0).unwrap();
    let mut worst: f64 = 0.0;
    for (i, pair_ab) in pairs.iter().enumerate() {
        for (j, pair_cd) in pairs.iter().enumerate().skip(i) {
            let mut largest = Largest(0.0);
            arena.scope(|scratch| {
                let bra_rows = BraRows::new_in(pair_ab, scratch)?;
                compute_quartet::<_, FullRange, S, S>(&mut largest, pair_ab, &data[i], &bra_rows, pair_cd, &data[j], GradientCentres::all(n_centres), scratch, 0.0)
            })
            .unwrap();
            let bound = derivative[i] * schwarz[j] + schwarz[i] * derivative[j];
            assert!(largest.0 <= bound * (1.0 + 1e-10), "{label}, pairs {i} {j}: derivative {:e} above the bound {bound:e}", largest.0);
            if bound > 0.0 {
                worst = worst.max(largest.0 / bound);
            }
        }
    }
    eprintln!("{label}: {} pairs, the largest derivative at most {worst:.3} of its bound", pairs.len());
    worst
}

#[test]
fn the_derivative_schwarz_bounds_bound_every_quartets_derivatives() {
    let dir = std::env::temp_dir().join(format!("aedelsten_derivative_bounds_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("water_dimer.inp");
    // Two waters 5.5 bohr apart, ANO-RCC generally contracted, s to d on O.
    std::fs::write(
        &path,
        "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\n    O2 5.5 0.3 -0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 1.5 -0.9\n    H2 -0.2 -1.4 -1.1\n    H3 6.1 1.6 0.8\n    H4 6.3 -1.2 0.6\nEnd of Basis\n",
    )
    .unwrap();
    let general = basis::setup_basis_system::<Contracted>(path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let n_centres = 6;
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap(), n_centres);
    let (uncontracted, _) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    check("uncontracted", &derive_shell_pairs(&uncontracted, None).unwrap(), n_centres);
}

/// The diagonal-only bound (one bra product at a time) against the full blocks' diagonal, pair by pair: equal to
/// rounding, for general and segmented contraction and the pairs of a shell with itself among them.
fn same_as_full_blocks<S: ShellKind>(label: &str, pairs: &[ShellPair<S>]) {
    let mut scratch = Scratch::new(512, ByteUnit::MiB).unwrap();
    let mut arena = scratch.get_arena();
    for (index, pair) in pairs.iter().enumerate() {
        let (diagonal, full) = arena.scope(|scratch| {
            let first = pair_data(pair, scratch).unwrap();
            let second = pair_data_to_order(pair, scratch, 2).unwrap();
            let diagonal = derivative_schwarz_bounds::<FullRange, S>(std::iter::once((pair, &first)), scratch, 0.0).unwrap()[0];
            let full = scratch.scope(|inner| derivative_bound_from_full_blocks::<FullRange, S>(pair, &second, inner).unwrap());
            (diagonal, full)
        });
        assert!((diagonal - full).abs() <= 1e-12 * full.max(1.0), "{label}, pair {index}: {diagonal:e} against the full blocks' {full:e}");
    }
}

#[test]
fn the_diagonal_bound_equals_the_full_blocks_diagonal() {
    let dir = std::env::temp_dir().join(format!("aedelsten_diagonal_bounds_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let general_path = dir.join("general.inp");
    std::fs::write(&general_path, "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 1.5 -0.9\n    H2 -0.2 -1.4 -1.1\nEnd of Basis\n").unwrap();
    let segmented_path = dir.join("segmented.inp");
    std::fs::write(&segmented_path, "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 1.5 -0.9\n    H2 -0.2 -1.4 -1.1\nEnd of Basis\n").unwrap();
    let general = basis::setup_basis_system::<Contracted>(general_path.to_str().unwrap()).unwrap();
    let segmented = basis::setup_basis_system::<Segmented>(segmented_path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    same_as_full_blocks("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap());
    same_as_full_blocks("segmented", &derive_shell_pairs(segmented.get_basis_shells(), None).unwrap());
}
