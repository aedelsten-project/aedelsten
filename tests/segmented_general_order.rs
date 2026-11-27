// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// A segmented basis set (x2c-SVPall) gives the same functions in the same order whether the context is built general
// or segmented: the basis-function labels, the overlap matrix and the stored two-electron integrals agree element by
// element, with spherical and with Cartesian d shells (oxygen's d shell; hydrogen's p, the same in either).

mod common;

use aedelsten::{AedelstenContext, BasisFunctionId, ContractionType, ERIAOSymmetry, ERIRangeOperator, OneElectronOperator};

const WATER: &str = "Basis set\n    O.x2c-SVPall\n{harmonics}    O1 0.0 0.0 0.2217\nEnd of Basis\n\
                     Basis set\n    H.x2c-SVPall\n{harmonics}    H1 0.0 1.4309 -0.8867\n    H2 0.0 -1.4309 -0.8867\nEnd of Basis\n";

fn context(name: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("segmented_general_order_{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn labels(ctx: &AedelstenContext) -> Vec<BasisFunctionId> {
    let mut ids = vec![BasisFunctionId::default(); ctx.get_n_cont()];
    ctx.get_basis_function_ids(&mut ids).unwrap();
    ids
}

fn overlap(ctx: &AedelstenContext) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut s = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut s, OneElectronOperator::overlap(), false).unwrap();
    s
}

fn eri(ctx: &AedelstenContext) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut dense = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut dense, ERIAOSymmetry::S1, ERIRangeOperator::FullRange).unwrap();
    dense
}

fn largest_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn a_segmented_set_has_one_order_general_or_segmented() {
    for (name, harmonics, n_expected) in [("spherical", "", 24), ("cartesian", "    Cartesian d\n", 25)] {
        let input = WATER.replace("{harmonics}", harmonics);
        let general = context(&format!("{name}_general"), &input, ContractionType::General);
        let segmented = context(&format!("{name}_segmented"), &input, ContractionType::Segmented);
        // O [3s2p1d] and two H [2s1p]: 3 + 6 + 5 (6 Cartesian) + 2 x (2 + 3).
        assert_eq!((general.get_n_cont(), segmented.get_n_cont()), (n_expected, n_expected), "{name}");
        assert_eq!(labels(&general), labels(&segmented), "{name}: the functions' labels and order");
        let (s_general, s_segmented) = (overlap(&general), overlap(&segmented));
        assert!(s_general.iter().any(|&x| x.abs() > 0.1), "{name}: an overlap to compare");
        assert!(largest_difference(&s_general, &s_segmented) < 1e-13, "{name}: overlap {:e}", largest_difference(&s_general, &s_segmented));
        let (g_general, g_segmented) = (eri(&general), eri(&segmented));
        assert!(largest_difference(&g_general, &g_segmented) < 1e-12, "{name}: ERIs {:e}", largest_difference(&g_general, &g_segmented));
    }
}
