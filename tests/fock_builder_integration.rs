// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// End-to-end check of a realistic setup -- QM region + pure embedding AIMP
// (Mg, TEST-EMB-AIMP: an SCEI entry with zero basis functions, zero
// orthogonality functions, so its whole contribution is the AIMP
// local Coulomb potential and its associated exchange term, nothing else)
// + an external point-charge field -- assembling the full Fock matrix
// (H_core + AIMP Coulomb/exchange + point-charge attraction + the direct
// two-electron G(P) piece already verified in tests/direct_fock.rs) the way
// a real SCF driver built on aedelsten would.
//
// Non-relativistic: built independently through ContractionType::Segmented
// and ContractionType::General from the same input, and checked to agree
// (not bitwise -- they process the same contraction data through different
// code paths). Relativistic: H_core is the X2C one-electron Hamiltonian,
// which only exists in the primitive basis, so there's no Segmented/General
// axis for it -- it's built once via set_primitive_shell_pairs and
// contracted into ContractionType::General, full stop. Primitive itself is
// checked separately via the contract-back round trip (same pattern as
// tests/oneint_invariants.rs), not against a second contraction scheme.
//
// Gradients are unaffected -- tests/gradient_finite_difference.rs remains
// the mechanism for those.

mod common;

use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, FiniteModel, OneElectronOperator};

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/fock_builder_test.inp");
const XFIELD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/fock_builder_test.xfield");

fn setup(contraction_type: ContractionType) -> AedelstenContext {
    setup_from(INP, contraction_type)
}

fn setup_from(inp: &str, contraction_type: ContractionType) -> AedelstenContext {
    let mut ctx = AedelstenContext::new(inp, contraction_type, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_external_multipoles(XFIELD).unwrap();
    ctx.set_embedding_aimp_operators().unwrap();
    ctx
}

fn symmetric_density(n: usize) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in i..n {
            let v = 0.1 + 0.05 * (((i * 7 + j * 3) % 11) as f64);
            p[(i, j)] = v;
            p[(j, i)] = v;
        }
    }
    p
}

// Sums every non-relativistic H_core contribution the fixture actually has:
// kinetic + nuclear attraction (the two real H nuclei -- the AIMP centre Na1
// carries no nuclear charge of its own, its electrostatics are entirely the
// AIMP Coulomb term below) + the external point-charge field + the
// embedding AIMP's Coulomb and exchange terms.
fn nonrelativistic_h_core(ctx: &AedelstenContext, n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n * n];
    let mut term = vec![0.0; n * n];

    ctx.compute_one_electron_integrals(&mut h, OneElectronOperator::kinetic(), false).unwrap();

    ctx.compute_one_electron_integrals(&mut term, ctx.get_nuclear_attraction_operator(), false).unwrap();
    for k in 0..n * n { h[k] += term[k]; }

    ctx.compute_one_electron_integrals(&mut term, ctx.get_external_multipole_operator().unwrap(), false).unwrap();
    for k in 0..n * n { h[k] += term[k]; }

    ctx.compute_one_electron_integrals(&mut term, ctx.get_embedding_aimp_coulomb_operator().unwrap(), false).unwrap();
    for k in 0..n * n { h[k] += term[k]; }

    ctx.compute_one_electron_integrals(&mut term, ctx.get_embedding_aimp_exchange_operator().unwrap(), false).unwrap();
    for k in 0..n * n { h[k] += term[k]; }

    h
}

// H_core + the direct two-electron G(P) piece -- the full Fock matrix a
// real SCF iteration would build.
fn total_fock(ctx: &mut AedelstenContext, h_core: &[f64], n: usize) -> Matrix<f64> {
    ctx.set_eri_data(None).unwrap();
    let p = symmetric_density(n);
    let mut j = Matrix::<f64>::new(n, n);
    let mut k = Matrix::<f64>::new(n, n);
    ctx.compute_fock_direct_restricted(p.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, aedelsten::ERIRangeOperator::FullRange, None).unwrap();

    let mut fock = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for l in 0..n {
            fock[(i, l)] = h_core[i * n + l] + j[(i, l)] + k[(i, l)];
        }
    }
    fock
}

fn max_diff(a: &Matrix<f64>, b: &Matrix<f64>, n: usize) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    worst
}

fn checksum(m: &Matrix<f64>, n: usize) -> (f64, f64) {
    let mut trace = 0.0;
    let mut frob = 0.0_f64;
    for i in 0..n {
        trace += m[(i, i)];
        for j in 0..n {
            frob += m[(i, j)] * m[(i, j)];
        }
    }
    (trace, frob.sqrt())
}

// Golden checks: unlike every other test in this file, these are not self-consistency identities but numeric values
// of a run of this code, the net for a systematic error that hits every path alike (a wrong constant, a wrong
// universal sign). The tolerance is tight but not zero, for rounding of another compiler, BLAS or optimization
// level.
const GOLDEN_TOL: f64 = 1e-9;

#[test]
fn nonrelativistic_fock_matches_todays_golden_values() {
    let mut ctx = setup(ContractionType::General);
    let n = ctx.get_n_cont();
    let h_core = nonrelativistic_h_core(&ctx, n);
    let fock = total_fock(&mut ctx, &h_core, n);
    let (trace, frob) = checksum(&fock, n);
    eprintln!("golden: {trace:.15e} {frob:.15e} {:.15e} {:.15e} {:.15e}", fock[(0, 0)], fock[(0, n - 1)], fock[(n / 2, n / 2)]);

    assert!((trace - 1.860719088014250e1).abs() < GOLDEN_TOL, "trace = {trace:.15e}");
    assert!((frob - 6.682615091368389e0).abs() < GOLDEN_TOL, "frobenius norm = {frob:.15e}");
    assert!((fock[(0, 0)] - 1.075476332170807e0).abs() < GOLDEN_TOL, "fock[0][0] = {:.15e}", fock[(0, 0)]);
    assert!((fock[(0, n - 1)] - -1.009657142719985e-1).abs() < GOLDEN_TOL, "fock[0][{}] = {:.15e}", n - 1, fock[(0, n - 1)]);
    assert!((fock[(n / 2, n / 2)] - 9.163754081405313e-1).abs() < GOLDEN_TOL, "fock[{0}][{0}] = {1:.15e}", n / 2, fock[(n / 2, n / 2)]);
}

#[test]
fn relativistic_x2c_fock_matches_todays_golden_values() {
    let mut ctx = setup(ContractionType::General);
    ctx.set_finite_nuclear_model(FiniteModel::VisscherDyall).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    let n = ctx.get_n_cont();
    let n_prim = ctx.get_n_prim_unique();

    let mut s = vec![0.0; n_prim * n_prim];
    ctx.compute_one_electron_integrals(&mut s, OneElectronOperator::overlap(), true).unwrap();
    let mut t = vec![0.0; n_prim * n_prim];
    ctx.compute_one_electron_integrals(&mut t, OneElectronOperator::kinetic(), true).unwrap();
    let mut v = vec![0.0; n_prim * n_prim];
    ctx.compute_one_electron_integrals(&mut v, ctx.get_nuclear_attraction_operator(), true).unwrap();
    let mut pvp = vec![0.0; n_prim * n_prim];
    ctx.compute_one_electron_picture_change_integrals(&mut pvp, ctx.get_nuclear_attraction_operator(), false).unwrap();
    let mut oneham_prim = vec![0.0; n_prim * n_prim];
    ctx.construct_x2c_one_electron_hamiltonian(&mut oneham_prim, &s, &t, &v, &pvp, None).unwrap();

    let mut h_core = vec![0.0; n * n];
    ctx.contract_one_electron_matrix(&oneham_prim, &mut h_core).unwrap();
    let mut term = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut term, ctx.get_external_multipole_operator().unwrap(), false).unwrap();
    for k in 0..n * n { h_core[k] += term[k]; }
    ctx.compute_one_electron_integrals(&mut term, ctx.get_embedding_aimp_coulomb_operator().unwrap(), false).unwrap();
    for k in 0..n * n { h_core[k] += term[k]; }
    ctx.compute_one_electron_integrals(&mut term, ctx.get_embedding_aimp_exchange_operator().unwrap(), false).unwrap();
    for k in 0..n * n { h_core[k] += term[k]; }

    let fock = total_fock(&mut ctx, &h_core, n);
    let (trace, frob) = checksum(&fock, n);
    eprintln!("golden: {trace:.15e} {frob:.15e} {:.15e} {:.15e} {:.15e}", fock[(0, 0)], fock[(0, n - 1)], fock[(n / 2, n / 2)]);
    let asymmetry = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (fock[(i, j)] - fock[(j, i)]).abs()).fold(0.0_f64, f64::max);
    assert!(asymmetry < 1e-9, "relativistic Fock matrix is not symmetric: max asymmetry {asymmetry}");

    assert!((trace - 1.860642976369107e1).abs() < GOLDEN_TOL, "trace = {trace:.15e}");
    assert!((frob - 6.682332164576276e0).abs() < GOLDEN_TOL, "frobenius norm = {frob:.15e}");
    assert!((fock[(0, 0)] - 1.075446800346697e0).abs() < GOLDEN_TOL, "fock[0][0] = {:.15e}", fock[(0, 0)]);
    assert!((fock[(0, n - 1)] - -1.009652117482007e-1).abs() < GOLDEN_TOL, "fock[0][{}] = {:.15e}", n - 1, fock[(0, n - 1)]);
    assert!((fock[(n / 2, n / 2)] - 9.163458763164203e-1).abs() < GOLDEN_TOL, "fock[{0}][{0}] = {1:.15e}", n / 2, fock[(n / 2, n / 2)]);
}

#[test]
fn nonrelativistic_fock_matches_between_segmented_and_general_contraction() {
    // A segmented set (x2c-SVPall water) with the fixture's embedding AIMP: segmented mode on a general
    // contraction such as ANO-RCC is not a meaningful use (every contracted function carries all primitives).
    let dir = common::temp_path("fock_segmented");
    std::fs::create_dir_all(&dir).unwrap();
    let inp = dir.join("water.inp");
    std::fs::write(&inp, "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\nBasis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    Na1 -1.63 2.07 3.28\nEnd of Basis\n").unwrap();
    let inp = inp.to_str().unwrap().to_string();
    let mut ctx_seg = setup_from(&inp, ContractionType::Segmented);
    let mut ctx_gen = setup_from(&inp, ContractionType::General);
    let _ = std::fs::remove_dir_all(&dir);
    let n = ctx_seg.get_n_cont();
    assert_eq!(n, ctx_gen.get_n_cont(), "Segmented and General should produce the same AO count");

    let h_seg = nonrelativistic_h_core(&ctx_seg, n);
    let h_gen = nonrelativistic_h_core(&ctx_gen, n);

    let fock_seg = total_fock(&mut ctx_seg, &h_seg, n);
    let fock_gen = total_fock(&mut ctx_gen, &h_gen, n);

    // Segmented and General sum the same contraction in different floating-point orders.
    let diff = max_diff(&fock_seg, &fock_gen, n);
    assert!(diff < 1e-7, "Segmented vs General Fock matrices disagree by {diff}");
}

#[test]
fn primitive_h_core_terms_contract_back_to_the_general_contracted_result() {
    let mut ctx_gen = setup(ContractionType::General);
    ctx_gen.set_primitive_shell_pairs(None).unwrap();
    let n_cont = ctx_gen.get_n_cont();
    let n_prim = ctx_gen.get_n_prim_unique();

    for (label, operator) in [
        ("kinetic", OneElectronOperator::kinetic()),
        ("nuclear_attraction", ctx_gen.get_nuclear_attraction_operator()),
        ("embedding_aimp_coulomb", ctx_gen.get_embedding_aimp_coulomb_operator().unwrap()),
        ("embedding_aimp_exchange", ctx_gen.get_embedding_aimp_exchange_operator().unwrap()),
    ] {
        let mut standard = vec![0.0; n_cont * n_cont];
        ctx_gen.compute_one_electron_integrals(&mut standard, operator, false).unwrap();

        // Rebuild the same operator constructor (OneElectronOperator borrows
        // from ctx, so it can't be reused after the call above moved it).
        let operator = match label {
            "kinetic" => OneElectronOperator::kinetic(),
            "nuclear_attraction" => ctx_gen.get_nuclear_attraction_operator(),
            "embedding_aimp_coulomb" => ctx_gen.get_embedding_aimp_coulomb_operator().unwrap(),
            "embedding_aimp_exchange" => ctx_gen.get_embedding_aimp_exchange_operator().unwrap(),
            _ => unreachable!(),
        };
        let mut primitive = vec![0.0; n_prim * n_prim];
        ctx_gen.compute_one_electron_integrals(&mut primitive, operator, true).unwrap();

        let mut contracted_back = vec![0.0; n_cont * n_cont];
        ctx_gen.contract_one_electron_matrix(&primitive, &mut contracted_back).unwrap();

        let diff = standard.iter().zip(contracted_back.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(diff < 1e-9, "{label}: contract(primitive) disagrees with standard by {diff}");
    }
}

// --- ERI range decomposition ---
// An identity of the operator, independent of the contraction scheme: one basis (eri_cache_test.inp) is enough.

fn simple_setup() -> AedelstenContext {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/eri_cache_test.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(None).unwrap();
    ctx
}

// 1/r = erf(wr)/r + erfc(wr)/r exactly, for any w -- an identity on the
// operator itself, not an approximation, so FullRange must equal
// LongRange{omega} + ShortRange{omega} elementwise for any omega.
#[test]
fn full_range_eri_equals_long_range_plus_short_range() {
    let ctx = simple_setup();
    let n = ctx.get_n_cont();
    let omega = 0.35;

    let mut full = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut full, aedelsten::ERIAOSymmetry::S1, aedelsten::ERIRangeOperator::FullRange).unwrap();
    let mut long = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut long, aedelsten::ERIAOSymmetry::S1, aedelsten::ERIRangeOperator::LongRange { omega }).unwrap();
    let mut short = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut short, aedelsten::ERIAOSymmetry::S1, aedelsten::ERIRangeOperator::ShortRange { omega }).unwrap();

    let diff = full.iter().zip(long.iter().zip(short.iter())).map(|(f, (l, s))| (f - (l + s)).abs()).fold(0.0, f64::max);
    assert!(diff < 1e-10, "FullRange != LongRange + ShortRange (omega={omega}), max diff {diff}");
}

