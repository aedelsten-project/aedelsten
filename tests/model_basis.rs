// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The model basis's reference orbitals and overlaps: `get_model_functions` for C, Ni
// and a core-AIMP Sc (its first ordinals the AIMP's projection-basis columns), `compute_model_basis_overlaps` against
// `compute_model_fockian` (bitwise, from the same S_mw), S_mm symmetric with orthonormal one-centre blocks, and the
// length checks.

mod common;

use aedelsten::arrays::matrix::{Matrix, MatrixMath, MatrixTrait};
use aedelsten::{AedelstenContext, ContractionType, ModelFunction};

fn context(name: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("model_basis_{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

// Working bases: ANO-RCC only generally contracted; the segmented case on x2c-SVPall.
const CO_ANO: &str = "Basis set\n    C.x2c-SVPall\n    C1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    O.x2c-SVPall\n    O1 0.0 0.0 2.13\nEnd of Basis\n";
const CO_SVP: &str = "Basis set\n    C.x2c-SVPall\n    C1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    O.x2c-SVPall\n    O1 0.0 0.0 2.13\nEnd of Basis\n";
const NI_ANO: &str = "Basis set\n    Ni.x2c-SVPall\n    Ni1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 0.0 2.8\nEnd of Basis\n";
const SC_AIMP: &str = "Basis set\n    Sc.TEST-CORE-AIMP\n    Sc1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 0.0 3.4\nEnd of Basis\n";

fn functions(ctx: &AedelstenContext) -> Vec<ModelFunction> {
    let n_model = ctx.get_n_model_functions().unwrap();
    let mut functions = vec![ModelFunction::default(); n_model];
    ctx.get_model_functions(&mut functions).unwrap();
    functions
}

/// The distinct (l, ordinal, energy) orbitals of one centre, in AO order.
fn orbitals(functions: &[ModelFunction], centre: usize) -> Vec<(usize, usize, f64)> {
    let mut orbitals: Vec<(usize, usize, f64)> = functions.iter().filter(|f| f.get_centre() == centre).map(|f| (f.get_l(), f.get_shell_ordinal(), f.get_energy())).collect();
    orbitals.dedup();
    orbitals
}

#[test]
fn the_model_functions_follow_the_reference_library() {
    // C and O: 1s, 2s, 2p (ANO-RCC's energy-bearing columns), the same whatever the working basis.
    for (name, input, contraction) in [("co_ano", CO_ANO, ContractionType::General), ("co_svp", CO_SVP, ContractionType::Segmented)] {
        let ctx = context(name, input, contraction);
        let f = functions(&ctx);
        assert_eq!(f.len(), 2 * (2 + 3), "{name}");
        for centre in 0..2 {
            let o = orbitals(&f, centre);
            assert_eq!(o.iter().map(|&(l, k, _)| (l, k)).collect::<Vec<_>>(), [(0, 0), (0, 1), (1, 0)], "{name}, centre {centre}");
            assert!(o[0].2 < o[1].2 && o.iter().all(|&(_, _, e)| e < 0.0), "{name}, centre {centre}: {o:?}");
        }
        // p in the Cartesian encoding x, y, z of BasisFunctionId.
        assert_eq!(f[2..5].iter().map(|f| f.get_m()).collect::<Vec<_>>(), [-1, 0, 1]);
    }

    // Ni: 1s-4s, 2p-3p, 3d; the d components spherical, -2..=2; then the H 1s on centre 1.
    let f = functions(&context("ni", NI_ANO, ContractionType::General));
    assert_eq!(f.len(), 4 + 2 * 3 + 5 + 1);
    let ni = orbitals(&f, 0);
    assert_eq!(ni.iter().map(|&(l, k, _)| (l, k)).collect::<Vec<_>>(), [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (2, 0)]);
    assert_eq!(f.iter().filter(|f| f.get_l() == 2).map(|f| f.get_m()).collect::<Vec<_>>(), [-2, -1, 0, 1, 2]);
    assert_eq!(orbitals(&f, 1).iter().map(|&(l, k, _)| (l, k)).collect::<Vec<_>>(), [(0, 0)]);
    // The 1s energy of H in ANO-RCC.
    assert_eq!(orbitals(&f, 1)[0].2, -0.5);

    // Sc with its core AIMP: the projection basis holds 1s, 2s and 2p, so s starts at ordinal 2, p at 1; d at 0. The
    // energies are ANO-RCC's 3s, 4s, 3p, 3d (as the model Fockian's own unit test reads them).
    let ctx = context("sc", SC_AIMP, ContractionType::General);
    assert!(ctx.has_core_aimp());
    let sc = orbitals(&functions(&ctx), 0);
    assert_eq!(sc, [(0, 2, -2.5908), (0, 3, -0.2116), (1, 1, -1.5712), (2, 0, -0.3376)]);
}

#[test]
fn the_overlaps_give_the_model_fockian_and_orthonormal_atoms() {
    for (name, input, contraction) in [("co_ano", CO_ANO, ContractionType::General), ("co_svp", CO_SVP, ContractionType::Segmented), ("ni", NI_ANO, ContractionType::General), ("sc", SC_AIMP, ContractionType::General)] {
        let ctx = context(name, input, contraction);
        let (n, f) = (ctx.get_n_cont(), functions(&ctx));
        let n_model = f.len();
        let (mut s_mw, mut s_mm) = (vec![f64::NAN; n_model * n], vec![f64::NAN; n_model * n_model]);
        ctx.compute_model_basis_overlaps(&mut s_mw, Some(&mut s_mm)).unwrap();

        // The model Fockian is S_mw^T diag(E) S_mw, bitwise, the same product from the same S_mw.
        let s = Matrix::new_from_vec(n_model, n, s_mw.clone().into_boxed_slice());
        let mut e = Matrix::<f64>::new(n_model, n_model);
        for (i, function) in f.iter().enumerate() {
            e[(i, i)] = function.get_energy();
        }
        let (mut scratch, mut expected) = (Matrix::<f64>::new(n_model, n), Matrix::<f64>::new(n, n));
        e.congruence_transform_prealloc(&s, &mut scratch, &mut expected);
        let mut fockian = vec![0.0; n * n];
        ctx.compute_model_fockian(&mut fockian).unwrap();
        assert_eq!(fockian, expected.get_data(), "{name}: the model Fockian from S_mw");

        // Without S_mm, the same S_mw.
        let mut alone = vec![f64::NAN; n_model * n];
        ctx.compute_model_basis_overlaps(&mut alone, None).unwrap();
        assert_eq!(alone, s_mw, "{name}: S_mw alone");

        // S_mm: symmetric; on one centre the identity (ANO-RCC's contractions are orthonormal per centre and l, and
        // different l or m are orthogonal by symmetry).
        let s_mm = Matrix::new_from_vec(n_model, n_model, s_mm.into_boxed_slice());
        let mut worst = 0.0_f64;
        for j in 0..n_model {
            for i in 0..n_model {
                assert!((s_mm[(i, j)] - s_mm[(j, i)]).abs() < 1e-14, "{name}: S_mm not symmetric at ({i}, {j})");
                if f[i].get_centre() == f[j].get_centre() {
                    worst = worst.max((s_mm[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
        }
        eprintln!("{name}: n_model {n_model}, one-centre blocks of S_mm differ from the identity by {worst:.2e}");
        assert!(worst < 1e-6, "{name}: one-centre S_mm off the identity by {worst}");
        // Two centres: some overlap between them.
        assert!((0..n_model).any(|j| (0..n_model).any(|i| f[i].get_centre() != f[j].get_centre() && s_mm[(i, j)].abs() > 1e-3)), "{name}");
    }
}

#[test]
fn wrong_lengths_are_errors() {
    let ctx = context("lengths", CO_SVP, ContractionType::Segmented);
    let (n, n_model) = (ctx.get_n_cont(), ctx.get_n_model_functions().unwrap());
    assert!(ctx.get_model_functions(&mut vec![ModelFunction::default(); n_model + 1]).is_err());
    assert!(ctx.compute_model_basis_overlaps(&mut vec![0.0; n_model * n - 1], None).is_err());
    assert!(ctx.compute_model_basis_overlaps(&mut vec![0.0; n_model * n], Some(&mut vec![0.0; n_model * n_model + 1])).is_err());
    assert!(ctx.compute_model_basis_overlaps(&mut vec![0.0; n_model * n], Some(&mut vec![0.0; n_model * n_model])).is_ok());
}
