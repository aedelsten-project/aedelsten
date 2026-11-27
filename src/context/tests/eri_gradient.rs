// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Verifies the ERI gradients (eri::gradient) against central finite differences of the ERIs themselves -- a different
// code path (the energy kernels at displaced geometries), so agreement is not circular. In the crate: the debug route
// is not public.
//   * the debug route (`compute_eri_gradient_tensors`, one S8 ERI tensor per atom and axis) against finite
//     differences of the S8 tensor, integral by integral;
//   * the production route (`compute_two_electron_gradient`) against finite differences of the two-electron
//     energy 1/2 sum P (J + K) at fixed densities, restricted and unrestricted, full/short/long range;
//   * the production accumulator against the debug tensors contracted by brute force;
//   * translational invariance (the gradients over the atoms sum to zero).
//
// Systems: water and HF-H in x2c-SVPall (bundled; spherical d on O and F, general contraction),
// HF-H also segmented and primitive.

use arrays::array4d::Array4DView;
use crate::packed::{PackedERITensor, PackedERITensorTrait};
use arrays::matrix::{Matrix, MatrixTrait, MatrixView};
use crate::fock_constructors::{construct_j_and_k_stored, construct_k_stored};
use crate::{AedelstenContext, TwoElectronTerm, TwoElectronTermKind, ContractionType, ERIAOSymmetry, ERIIndexOrder, ERIRangeOperator};

// A path in the temporary directory no other call in this process returns.
fn temp_path(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("aedelsten_eri_gradient_{}_{call}_{name}", std::process::id()))
}

const STEP: f64 = 2.0e-4; // central-difference step, bohr

type Geometry = Vec<(&'static str, [f64; 3])>;

struct System {
    basis: &'static str, // the basis-set name after the element, e.g. "x2c-SVPall"
    geometry: Geometry,
    contraction: ContractionType,
}

/// The same system in another contraction type.
fn with_contraction(system: System, contraction: ContractionType) -> System {
    System { contraction, ..system }
}

fn water() -> System {
    System { basis: "x2c-SVPall", geometry: vec![("O", [0.0, 0.1, 0.2]), ("H", [0.3, 1.5, -0.9]), ("H", [-0.2, -1.4, -1.1])], contraction: ContractionType::General }
}

fn hydrogen_fluoride_hydrogen() -> System {
    System { basis: "x2c-SVPall", geometry: vec![("F", [0.0, 0.1, 0.2]), ("H", [0.3, 1.5, -0.9]), ("H", [-0.2, -1.4, -2.3])], contraction: ContractionType::General }
}

fn build(basis: &str, geometry: &Geometry, tag: &str, contraction: ContractionType) -> AedelstenContext {
    let mut input = String::new();
    for (i, (element, position)) in geometry.iter().enumerate() {
        input += &format!(
            "Basis set\n    {element}.{basis}\n    {element}{} {:.15} {:.15} {:.15}\nEnd of Basis\n",
            i + 1, position[0], position[1], position[2]
        );
    }
    let path = temp_path(&format!("eri_grad_{tag}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn displaced(geometry: &Geometry, atom: usize, axis: usize, delta: f64) -> Geometry {
    let mut g = geometry.clone();
    g[atom].1[axis] += delta;
    g
}

/// Central finite difference of `quantity` with respect to every (atom, axis), index `3 * atom + axis`.
fn finite_differences(system: &System, quantity: &dyn Fn(&AedelstenContext) -> Vec<f64>) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for atom in 0..system.geometry.len() {
        for axis in 0..3 {
            let plus = quantity(&build(system.basis, &displaced(&system.geometry, atom, axis, STEP), &format!("p{atom}{axis}"), system.contraction));
            let minus = quantity(&build(system.basis, &displaced(&system.geometry, atom, axis, -STEP), &format!("m{atom}{axis}"), system.contraction));
            out.push(plus.iter().zip(&minus).map(|(p, m)| (p - m) / (2.0 * STEP)).collect());
        }
    }
    out
}

fn s8_eri(ctx: &AedelstenContext, range: ERIRangeOperator) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut out = vec![0.0; crate::compute_s8_size(n)];
    ctx.compute_eri_stored(&mut out, ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor }, range).unwrap();
    out
}

fn dense_eri(ctx: &AedelstenContext, range: ERIRangeOperator) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut out = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut out, ERIAOSymmetry::S1, range).unwrap();
    out
}

// Deterministic, non-trivial symmetric matrix; `scale` keeps the energies moderate.
fn symmetric_density(n: usize, salt: usize, scale: f64) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in i..n {
            let v = scale * (0.1 + 0.05 * (((i * 7 + j * 3 + salt) % 11) as f64));
            p[(i, j)] = v;
            p[(j, i)] = v;
        }
    }
    p
}

fn view(m: &Matrix<f64>) -> &[f64] {
    m.get_data()
}

/// The terms of `1/2 sum [P_ab P_cd + coeff_k sum_x Px_ac Px_bd] (ab|cd)`.
fn terms<'a>(p_coulomb: &'a Matrix<f64>, p_exchange: &[&'a Matrix<f64>], coeff_k: f64) -> Vec<TwoElectronTerm<'a>> {
    let mut terms = vec![TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: view(p_coulomb), right: view(p_coulomb), coeff: 1.0 }];
    for p in p_exchange {
        terms.push(TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: view(p), right: view(p), coeff: coeff_k });
    }
    terms
}

/// A flat `(3, n_atoms)` gradient as an `(n_atoms, 3)` matrix, `m[(atom, axis)]`.
fn by_atom(flat: &[f64]) -> Matrix<f64> {
    let n_atoms = flat.len() / 3;
    let mut m = Matrix::<f64>::new(n_atoms, 3);
    for atom in 0..n_atoms {
        for axis in 0..3 {
            m[(atom, axis)] = flat[3 * atom + axis];
        }
    }
    m
}

/// The contracted route, `compute_two_electron_gradient`.
fn two_electron_gradient(ctx: &AedelstenContext, p_coulomb: &Matrix<f64>, p_exchange: &[&Matrix<f64>], coeff_k: f64, range: ERIRangeOperator) -> Matrix<f64> {
    let mut flat = vec![0.0; 3 * ctx.get_n_gradient_centres()];
    ctx.compute_two_electron_gradient(&mut flat, &terms(p_coulomb, p_exchange, coeff_k), range, None, None).unwrap();
    by_atom(&flat)
}

fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn debug_tensors(ctx: &mut AedelstenContext, n_atoms: usize, range: ERIRangeOperator) -> Vec<PackedERITensor<f64>> {
    let n = ctx.get_n_cont();
    ctx.set_eri_data(None).unwrap();
    let mut tensors: Vec<PackedERITensor<f64>> = (0..3 * n_atoms).map(|_| PackedERITensor::new(n)).collect();
    ctx.compute_eri_gradient_tensors(&mut tensors, range, None).unwrap();
    tensors
}

fn check_debug_tensors_against_finite_differences(system: &System, range: ERIRangeOperator, tolerance: f64) {
    let n_atoms = system.geometry.len();
    let mut ctx = build(system.basis, &system.geometry, "debug", system.contraction);
    let tensors = debug_tensors(&mut ctx, n_atoms, range);
    let reference = finite_differences(system, &|c| s8_eri(c, range));
    for (index, expected) in reference.iter().enumerate() {
        let error = max_abs_difference(tensors[index].get_data(), expected);
        let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        eprintln!("atom {} axis {}: max |finite difference| {scale:.3e}, max error {error:.3e}", index / 3, index % 3);
        assert!(scale > 1e-3, "atom {} axis {}: the reference is (almost) zero, the check would be vacuous", index / 3, index % 3);
        assert!(error < tolerance, "atom {} axis {}: max |analytic - finite difference| = {error}", index / 3, index % 3);
    }
    // The four derivatives of every integral sum to zero, so the atoms' tensors cancel.
    for axis in 0..3 {
        let mut sum = vec![0.0; tensors[0].get_data().len()];
        for atom in 0..n_atoms {
            for (s, v) in sum.iter_mut().zip(tensors[3 * atom + axis].get_data()) {
                *s += v;
            }
        }
        let residual = sum.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(residual < 1e-9, "translational invariance, axis {axis}: {residual}");
    }
}

#[test]
fn debug_tensors_match_finite_differences_water() {
    check_debug_tensors_against_finite_differences(&water(), ERIRangeOperator::FullRange, 1e-7);
}

#[test]
fn debug_tensors_match_finite_differences_ccpvdz_with_spherical_d() {
    check_debug_tensors_against_finite_differences(&hydrogen_fluoride_hydrogen(), ERIRangeOperator::FullRange, 1e-7);
}

#[test]
fn debug_tensors_match_finite_differences_for_range_separated_operators() {
    let system = water();
    check_debug_tensors_against_finite_differences(&system, ERIRangeOperator::LongRange { omega: 0.4 }, 1e-7);
    check_debug_tensors_against_finite_differences(&system, ERIRangeOperator::ShortRange { omega: 0.4 }, 1e-7);
}

// E_2e = 1/2 sum_abcd [ coeff_j P_ab P_cd + coeff_k sum_x Px_ac Px_bd ] (ab|cd), via the dense builders.
fn two_electron_energy(ctx: &AedelstenContext, range: ERIRangeOperator, p_coulomb: &Matrix<f64>, p_exchange: &[&Matrix<f64>], coeff_k: f64) -> f64 {
    let n = ctx.get_n_cont();
    let dense = dense_eri(ctx, range);
    let g = Array4DView::new(n, n, n, n, &dense);
    let mut j = Matrix::<f64>::new(n, n);
    let mut k = Matrix::<f64>::new(n, n);
    construct_j_and_k_stored(&g, p_coulomb, &mut j, &mut k, 1.0, 0.0);
    let mut energy = 0.0;
    for a in 0..n {
        for b in 0..n {
            energy += 0.5 * p_coulomb[(a, b)] * j[(a, b)];
        }
    }
    for p in p_exchange {
        construct_k_stored(&g, *p, &mut k, coeff_k);
        for a in 0..n {
            for b in 0..n {
                energy += 0.5 * p[(a, b)] * k[(a, b)];
            }
        }
    }
    energy
}

fn check_direct_gradient_against_finite_differences(system: &System, range: ERIRangeOperator, unrestricted: bool, tolerance: f64) {
    let n_atoms = system.geometry.len();
    let mut ctx = build(system.basis, &system.geometry, "direct", system.contraction);
    let n = ctx.get_n_cont();
    let p_alpha = symmetric_density(n, 0, 0.3);
    let p_beta = symmetric_density(n, 5, 0.2);
    let mut p_total = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            p_total[(i, j)] = if unrestricted { p_alpha[(i, j)] + p_beta[(i, j)] } else { p_alpha[(i, j)] };
        }
    }
    let (exchange, coeff_k): (Vec<&Matrix<f64>>, f64) = if unrestricted { (vec![&p_alpha, &p_beta], -1.0) } else { (vec![&p_alpha], -0.5) };

    ctx.set_eri_data(None).unwrap();
    let gradient = two_electron_gradient(&ctx, &p_total, &exchange, coeff_k, range);

    let reference = finite_differences(system, &|c| vec![two_electron_energy(c, range, &p_total, &exchange, coeff_k)]);
    for atom in 0..n_atoms {
        for axis in 0..3 {
            let (got, expected) = (gradient[(atom, axis)], reference[3 * atom + axis][0]);
            eprintln!("atom {atom} axis {axis}: analytic {got:.8}, finite difference {expected:.8}");
            assert!((got - expected).abs() < tolerance, "atom {atom} axis {axis}: analytic {got}, finite difference {expected}");
        }
    }
    for axis in 0..3 {
        let sum: f64 = (0..n_atoms).map(|atom| gradient[(atom, axis)]).sum();
        assert!(sum.abs() < 1e-9, "translational invariance, axis {axis}: {sum}");
    }
}

#[test]
fn direct_gradient_matches_finite_differences_restricted_and_unrestricted() {
    let system = water();
    check_direct_gradient_against_finite_differences(&system, ERIRangeOperator::FullRange, false, 1e-6);
    check_direct_gradient_against_finite_differences(&system, ERIRangeOperator::FullRange, true, 1e-6);
}

#[test]
fn direct_gradient_matches_finite_differences_range_separated() {
    let system = water();
    check_direct_gradient_against_finite_differences(&system, ERIRangeOperator::LongRange { omega: 0.4 }, false, 1e-6);
    check_direct_gradient_against_finite_differences(&system, ERIRangeOperator::ShortRange { omega: 0.4 }, false, 1e-6);
}

#[test]
fn direct_gradient_matches_finite_differences_ccpvdz_with_spherical_d() {
    check_direct_gradient_against_finite_differences(&hydrogen_fluoride_hydrogen(), ERIRangeOperator::FullRange, false, 1e-6);
}

// The accumulator's permutation weights, independent of any finite difference: contract the debug
// tensors with the densities by brute force.
#[test]
fn direct_gradient_equals_the_debug_tensors_contracted_by_brute_force() {
    let system = hydrogen_fluoride_hydrogen();
    let n_atoms = system.geometry.len();
    let mut ctx = build(system.basis, &system.geometry, "contract", system.contraction);
    let n = ctx.get_n_cont();
    let p = symmetric_density(n, 3, 0.3);
    let tensors = debug_tensors(&mut ctx, n_atoms, ERIRangeOperator::FullRange);

    let gradient = two_electron_gradient(&ctx, &p, &[&p], -0.5, ERIRangeOperator::FullRange);
    for atom in 0..n_atoms {
        for axis in 0..3 {
            let t = &tensors[3 * atom + axis];
            let mut brute = 0.0;
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        for d in 0..n {
                            brute += 0.5 * (p[(a, b)] * p[(c, d)] - 0.5 * p[(a, c)] * p[(b, d)]) * t[(a, b, c, d)];
                        }
                    }
                }
            }
            assert!((gradient[(atom, axis)] - brute).abs() < 1e-9, "atom {atom} axis {axis}: {} vs {brute}", gradient[(atom, axis)]);
        }
    }
}

// Segmented and primitive contraction go through the same kernels (gradient::Contraction: the coefficient in each
// product, the products accumulated straight into the single contracted pair). On HF-H in x2c-SVPall (a segmented
// set): the derivative integrals and the contracted gradient (restricted and unrestricted).
#[test]
fn segmented_and_primitive_contractions_match_finite_differences() {
    for contraction in [ContractionType::Segmented, ContractionType::Primitive] {
        // The contraction changes only the integrals, not how they are contracted with the densities or ranges.
        let fhh = || with_contraction(hydrogen_fluoride_hydrogen(), contraction);
        check_debug_tensors_against_finite_differences(&fhh(), ERIRangeOperator::FullRange, 1e-6);
        check_direct_gradient_against_finite_differences(&fhh(), ERIRangeOperator::FullRange, true, 1e-6);
    }
}

// --- The term list: densities that are not symmetric (a spin-flip block, which pairs with itself) and
// terms that pair two different densities (inactive/active), against finite differences of the energy those
// terms state, evaluated from the dense ERI tensor. ---

fn nonsymmetric_density(n: usize, salt: usize, scale: f64) -> Matrix<f64> {
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            p[(i, j)] = scale * (0.05 * (((i * 5 + j * 11 + salt) % 13) as f64) - 0.2);
        }
    }
    p
}

fn transpose(m: &Matrix<f64>) -> Matrix<f64> {
    let mut t = Matrix::<f64>::new(m.cols(), m.rows());
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            t[(j, i)] = m[(i, j)];
        }
    }
    t
}

/// `1/2 sum_terms coeff sum_abcd W (ab|cd)` by brute force over the dense tensor.
fn term_energy(ctx: &AedelstenContext, terms: &[TwoElectronTerm<'_>]) -> f64 {
    let n = ctx.get_n_cont();
    let dense = dense_eri(ctx, ERIRangeOperator::FullRange);
    let g = Array4DView::new(n, n, n, n, &dense);
    let mut energy = 0.0;
    for d in 0..n {
        for c in 0..n {
            for b in 0..n {
                for a in 0..n {
                    let integral = g[(a, b, c, d)];
                    for term in terms {
                        let (left, right) = (MatrixView::new(n, n, term.left), MatrixView::new(n, n, term.right));
                        let w = match term.kind {
                            TwoElectronTermKind::Coulomb => left[(a, b)] * right[(c, d)],
                            TwoElectronTermKind::Exchange => left[(a, c)] * right[(b, d)],
                        };
                        energy += 0.5 * term.coeff * w * integral;
                    }
                }
            }
        }
    }
    energy
}

#[test]
fn nonsymmetric_and_mixed_density_terms_match_finite_differences() {
    let system = hydrogen_fluoride_hydrogen();
    let n_atoms = system.geometry.len();
    let mut ctx = build(system.basis, &system.geometry, "terms", system.contraction);
    let n = ctx.get_n_cont();
    let p_a = symmetric_density(n, 0, 0.3);
    let p_b = symmetric_density(n, 5, 0.2);
    let p_ab = nonsymmetric_density(n, 3, 0.4);
    let p_ba = transpose(&p_ab);
    let make = || {
        vec![
            TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: view(&p_a), right: view(&p_b), coeff: 0.7 },
            TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: view(&p_ab), right: view(&p_ba), coeff: -0.6 },
            TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: view(&p_a), right: view(&p_b), coeff: -0.3 },
            TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: view(&p_ab), right: view(&p_a), coeff: 0.4 },
            TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: view(&p_ab), right: view(&p_ab), coeff: -2.0 },
        ]
    };
    ctx.set_eri_data(None).unwrap();
    let mut flat = vec![0.0; 3 * n_atoms];
    ctx.compute_two_electron_gradient(&mut flat, &make(), ERIRangeOperator::FullRange, None, None).unwrap();
    let reference = finite_differences(&system, &|c| vec![term_energy(c, &make())]);
    for atom in 0..n_atoms {
        for axis in 0..3 {
            let (got, expected) = (flat[3 * atom + axis], reference[3 * atom + axis][0]);
            eprintln!("atom {atom} axis {axis}: analytic {got:.8}, finite difference {expected:.8}");
            assert!(expected.abs() > 1e-4, "atom {atom} axis {axis}: the reference is (almost) zero, the check would be vacuous");
            assert!((got - expected).abs() < 1e-6, "atom {atom} axis {axis}: analytic {got}, finite difference {expected}");
        }
    }

    // The output is overwritten, and a mis-sized gradient or density is an error.
    let mut dirty = vec![123.0; 3 * n_atoms];
    ctx.compute_two_electron_gradient(&mut dirty, &make(), ERIRangeOperator::FullRange, None, None).unwrap();
    assert!(max_abs_difference(&dirty, &flat) < 1e-12);
    let mut too_long = vec![0.0; 3 * n_atoms + 3];
    assert!(ctx.compute_two_electron_gradient(&mut too_long, &make(), ERIRangeOperator::FullRange, None, None).is_err());
    let small = Matrix::<f64>::new(n - 1, n - 1);
    let wrong = [TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: view(&small), right: view(&small), coeff: 1.0 }];
    assert!(ctx.compute_two_electron_gradient(&mut flat, &wrong, ERIRangeOperator::FullRange, None, None).is_err());
}

// --- Orthogonality ghosts: centres last in the list, not gradient centres. The ghost carries basis functions
// (its orthogonality functions), so its shells enter the ERI loops, but its position is fixed: the matrix is
// (3, n_real), the ghost's derivatives are never emitted (nor folded into another atom), and the real atoms'
// gradient is the finite difference with the ghost held in place. ---

fn h2_with_ghost(h: [[f64; 3]; 2], tag: &str) -> AedelstenContext {
    let input = format!(
        "Basis set\n    H.x2c-SVPall\n    H1 {:.15} {:.15} {:.15}\n    H2 {:.15} {:.15} {:.15}\nEnd of Basis\n\
         Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    orthogonality\n    Na1 1.9 0.4 0.3\nEnd of Basis\n",
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2]
    );
    let path = temp_path(&format!("eri_grad_ghost_{tag}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

#[test]
fn orthogonality_ghosts_are_outside_the_two_electron_gradient() {
    let h = [[0.0, 0.0, 0.0], [0.3, 0.2, 1.4]];
    let mut ctx = h2_with_ghost(h, "base");
    assert_eq!(ctx.get_n_gradient_centres(), 2);
    let n = ctx.get_n_cont();
    assert!(n > 2, "the ghost should carry orthogonality functions");
    ctx.set_eri_data(None).unwrap();
    let p = symmetric_density(n, 1, 0.3);
    let got = two_electron_gradient(&ctx, &p, &[&p], -0.5, ERIRangeOperator::FullRange);

    for atom in 0..2 {
        for axis in 0..3 {
            let energy = |delta: f64| {
                let mut displaced = h;
                displaced[atom][axis] += delta;
                two_electron_energy(&h2_with_ghost(displaced, "fd"), ERIRangeOperator::FullRange, &p, &[&p], -0.5)
            };
            let expected = (energy(STEP) - energy(-STEP)) / (2.0 * STEP);
            let analytic = got[(atom, axis)];
            eprintln!("H{} axis {axis}: analytic {analytic:.8}, finite difference {expected:.8}", atom + 1);
            assert!((analytic - expected).abs() < 1e-7, "H{} axis {axis}: {analytic} vs {expected}", atom + 1);
        }
    }
    // The ghost is fixed, so the H atoms' gradients do not cancel: its derivative was dropped, not folded.
    let residual = (0..3).map(|axis| (got[(0, axis)] + got[(1, axis)]).abs()).fold(0.0, f64::max);
    assert!(residual > 1e-6, "the H gradients cancel; the ghost's share seems to have been folded into them");

    let mut too_wide = vec![0.0; 9];
    let terms = terms(&p, &[&p], -0.5);
    assert!(ctx.compute_two_electron_gradient(&mut too_wide, &terms, ERIRangeOperator::FullRange, None, None).is_err());
}

// --- The frozen-centre mask on the all-centres route: for every subset of active centres, the active
// centres' derivative tensors are the unmasked ones and the frozen centres receive nothing. Four atoms,
// so that quartets with four distinct centres (every one of A, B, C, D derived differently) are covered. ---

fn hydrogen_fluoride_dimer_ccpvdz() -> System {
    System {
        basis: "x2c-SVPall",
        geometry: vec![("F", [0.0, 0.1, 0.2]), ("H", [0.3, 1.5, -0.9]), ("F", [0.2, -1.4, -3.1]), ("H", [-0.8, -2.2, -4.0])],
        contraction: ContractionType::General,
    }
}

fn check_every_mask(system: &System) {
    let n_atoms = system.geometry.len();
    let mut ctx = build(system.basis, &system.geometry, "mask", system.contraction);
    let n = ctx.get_n_cont();
    let all = debug_tensors(&mut ctx, n_atoms, ERIRangeOperator::FullRange);
    for subset in 0..(1usize << n_atoms) {
        let mask: Vec<bool> = (0..n_atoms).map(|atom| subset & (1 << atom) != 0).collect();
        let mut tensors: Vec<PackedERITensor<f64>> = (0..3 * n_atoms).map(|_| PackedERITensor::new(n)).collect();
        ctx.compute_eri_gradient_tensors(&mut tensors, ERIRangeOperator::FullRange, Some(&mask)).unwrap();
        for atom in 0..n_atoms {
            for axis in 0..3 {
                let got = tensors[3 * atom + axis].get_data();
                let error = if mask[atom] {
                    max_abs_difference(got, all[3 * atom + axis].get_data())
                } else {
                    got.iter().fold(0.0, |m: f64, v| m.max(v.abs()))
                };
                assert!(error < 1e-11, "mask {mask:?}, atom {atom} axis {axis}: off by {error}");
            }
        }
    }
    let mut tensors: Vec<PackedERITensor<f64>> = (0..3 * n_atoms).map(|_| PackedERITensor::new(n)).collect();
    let short_mask = vec![true; n_atoms - 1];
    assert!(ctx.compute_eri_gradient_tensors(&mut tensors, ERIRangeOperator::FullRange, Some(&short_mask)).is_err());
}

#[test]
fn the_frozen_centre_mask_keeps_the_active_centres_and_zeroes_the_frozen_ones() {
    check_every_mask(&water());
    check_every_mask(&hydrogen_fluoride_dimer_ccpvdz());
}

#[test]
fn debug_tensors_match_finite_differences_with_four_distinct_centres() {
    check_debug_tensors_against_finite_differences(&hydrogen_fluoride_dimer_ccpvdz(), ERIRangeOperator::FullRange, 1e-6);
}
