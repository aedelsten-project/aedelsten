// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The tiled, orbit-based projection against the dense per-operation one, on synthetic molecules (no
// integrals: any matrix has a projection), for every irrep: Abelian groups in their standard orientation (the
// signed-permutation fast path), a rotated molecule (dense rotation blocks), degenerate irreps (C3v's E, Td's T) and
// a group with complex-pair irreps (C3h). Plus the timing of both (ignored; run with --release --ignored --nocapture).

use super::*;
use arrays::scratch::{Scratch, ByteUnit};

// Heavy centres (kind 1) carry s, s, p, d, f; hydrogens (kind 0) s and p. Components as aedelsten orders them: p as
// x, y, z (m = +1, -1, 0), d and up ascending.
fn basis(kinds: &[u32], shells: &[(u32, usize)], light: &[(u32, usize)]) -> Vec<SymmetryBasisFunction> {
    basis_of(kinds, shells, light, false)
}

// The same with Cartesian d and up (components in the library's order: lx descending, then ly descending).
fn basis_of(kinds: &[u32], shells: &[(u32, usize)], light: &[(u32, usize)], cartesian: bool) -> Vec<SymmetryBasisFunction> {
    let mut functions = Vec::new();
    for (centre, &kind) in kinds.iter().enumerate() {
        let mut radial_of_l = [0usize; 8];
        for &(l, count) in if kind == 0 { light } else { shells } {
            for _ in 0..count {
                radial_of_l[l as usize] += 1;
                let radial = radial_of_l[l as usize];
                let ms: Vec<i32> = if l == 1 {
                    vec![1, -1, 0]
                } else if cartesian {
                    (0..((l + 1) * (l + 2) / 2) as i32).collect()
                } else {
                    (-(l as i32)..=l as i32).collect()
                };
                functions.extend(ms.into_iter().map(|m| SymmetryBasisFunction { centre, radial, l, m, cartesian: cartesian && l > 1 }));
            }
        }
    }
    functions
}

const HEAVY: [(u32, usize); 4] = [(0, 2), (1, 1), (2, 1), (3, 1)];
const LIGHT: [(u32, usize); 2] = [(0, 1), (1, 1)];

fn symmetry(kinds: &[u32], positions: &[[f64; 3]], request: PointGroupRequest<'_>) -> Symmetry {
    let centres: Vec<SymmetryCentre> =
        kinds.iter().zip(positions).map(|(&kind, &position)| SymmetryCentre { kind, mass: if kind == 0 { 1.0 } else { 12.0 }, position }).collect();
    Symmetry::new(&centres, &basis(kinds, &HEAVY, &LIGHT), None, request, DEFAULT_GEOMETRY_TOLERANCE).unwrap()
}

// A deterministic, unstructured, non-symmetric matrix.
fn noise(n: usize) -> Matrix<f64> {
    let mut m = Matrix::<f64>::new(n, n);
    let mut state = 12345u64;
    for x in m.get_data_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
    }
    m
}

fn rotate(p: [f64; 3]) -> [f64; 3] {
    let (a, b): (f64, f64) = (0.3, 0.7);
    let q = [a.cos() * p[0] - a.sin() * p[1], a.sin() * p[0] + a.cos() * p[1], p[2]];
    [q[0] + 0.4, b.cos() * q[1] - b.sin() * q[2] - 1.1, b.sin() * q[1] + b.cos() * q[2] + 2.3]
}

fn molecules() -> Vec<(&'static str, Symmetry)> {
    let water = |f: &dyn Fn([f64; 3]) -> [f64; 3]| [f([0.0, 0.0, 0.2217]), f([1.4309, 0.0, -0.8867]), f([-1.4309, 0.0, -0.8867])];
    let third = 2.0 * std::f64::consts::PI / 3.0;
    let ammonia: Vec<[f64; 3]> = std::iter::once([0.0, 0.0, 0.13]).chain((0..3).map(|k| [1.77 * (k as f64 * third).cos(), 1.77 * (k as f64 * third).sin(), -0.68])).collect();
    let methane = [[0.0; 3], [1.2, 1.2, 1.2], [1.2, -1.2, -1.2], [-1.2, 1.2, -1.2], [-1.2, -1.2, 1.2]];
    let ethylene = [[0.0, 0.0, 1.26], [0.0, 0.0, -1.26], [0.0, 1.75, 2.33], [0.0, -1.75, 2.33], [0.0, 1.75, -2.33], [0.0, -1.75, -2.33]];
    // B(OH)3, planar C3h: the H off the B-O axes, all the same way round.
    let mut boric = vec![[0.0; 3]];
    for k in 0..3 {
        let t = k as f64 * third;
        boric.push([2.6 * t.cos(), 2.6 * t.sin(), 0.0]);
        boric.push([3.0 * (t + 0.5).cos(), 3.0 * (t + 0.5).sin(), 0.0]);
    }
    vec![
        ("water, standard orientation", symmetry(&[1, 0, 0], &water(&|p| p), PointGroupRequest::Highest)),
        ("water, rotated", symmetry(&[1, 0, 0], &water(&rotate), PointGroupRequest::Highest)),
        ("ammonia", symmetry(&[1, 0, 0, 0], &ammonia, PointGroupRequest::Highest)),
        ("methane", symmetry(&[1, 0, 0, 0, 0], &methane, PointGroupRequest::Highest)),
        ("ethylene", symmetry(&[1, 1, 0, 0, 0, 0], &ethylene, PointGroupRequest::Highest)),
        ("boric acid", symmetry(&[1, 1, 0, 1, 0, 1, 0], &boric, PointGroupRequest::Highest)),
    ]
}

fn scratch(symmetry: &Symmetry) -> Scratch {
    // The dense reference needs 4 n^2.
    let bytes = 8 * (4 * symmetry.n_ao() * symmetry.n_ao() + symmetry.matrix_scratch_len()) + (1 << 20);
    Scratch::new(bytes.div_ceil(1 << 20), ByteUnit::MiB).unwrap()
}

#[test]
fn the_groups_are_the_expected_ones() {
    let groups: Vec<String> = molecules().iter().map(|(_, s)| s.point_group().to_string()).collect();
    assert_eq!(groups, ["C2v", "C2v", "C3v", "Td", "D2h", "C3h"]);
}

#[test]
fn tiled_projection_matches_the_dense_one_for_every_irrep() {
    for (what, symmetry) in molecules() {
        let n = symmetry.n_ao();
        let matrix = noise(n);
        let mut scratch = scratch(&symmetry);
        let mut arena = scratch.get_arena();
        for k in 0..symmetry.irreps().len() {
            let name = &symmetry.irreps()[k].name;
            let (dense, dense_report) = arena.scope(|arena| {
                let mut cleaned = MatrixViewMut::<f64>::from_arena(n, n, arena).unwrap();
                let report = symmetry.project_dense(&matrix.view(), OperatorSymmetry::Irrep(k), &mut cleaned, arena).unwrap();
                (cleaned.get_data().to_vec(), report)
            });
            let mut tiled = matrix.clone();
            let report = arena.scope(|arena| symmetry.average_matrix(AoBasis::Contracted, &mut tiled, OperatorSymmetry::Irrep(k), arena).unwrap());
            let difference = vector_math::distance(&dense, tiled.get_data());
            assert!(difference < 1e-12 * dense_report.norm, "{what}, {name}: tiled and dense differ by {difference:.3e}");
            assert!((report.removed_norm - dense_report.removed_norm).abs() < 1e-10, "{what}, {name}: {report:?} vs {dense_report:?}");
            assert!((report.max_abs_removed - dense_report.max_abs_removed).abs() < 1e-12, "{what}, {name}: {report:?} vs {dense_report:?}");
            // analyse_matrix reports the same and changes nothing.
            let analysed = arena.scope(|arena| symmetry.analyse_matrix(AoBasis::Contracted, &matrix, OperatorSymmetry::Irrep(k), arena).unwrap());
            assert!((analysed.removed_norm - report.removed_norm).abs() < 1e-10, "{what}, {name}");
        }
        // The projections onto all irreps add up to the matrix (they are complementary), and a projected matrix is
        // unchanged by a second projection.
        let mut sum = Matrix::<f64>::new(n, n);
        for k in 0..symmetry.irreps().len() {
            let mut part = matrix.clone();
            arena.scope(|arena| symmetry.average_matrix(AoBasis::Contracted, &mut part, OperatorSymmetry::Irrep(k), arena).unwrap());
            vector_math::add_in_place(sum.get_data_mut(), part.get_data());
            let again = arena.scope(|arena| symmetry.average_matrix(AoBasis::Contracted, &mut part, OperatorSymmetry::Irrep(k), arena).unwrap());
            assert!(again.max_abs_removed < 1e-12, "{what}: projecting twice removed {:.3e}", again.max_abs_removed);
        }
        let difference = vector_math::distance(sum.get_data(), matrix.get_data());
        assert!(difference < 1e-11, "{what}: the irreps' parts add up to the matrix only within {difference:.3e}");
    }
}

#[test]
fn tiled_irrep_norms_match_the_dense_ones() {
    for (what, symmetry) in molecules() {
        let n = symmetry.n_ao();
        let matrix = noise(n);
        let mut scratch = scratch(&symmetry);
        let mut arena = scratch.get_arena();
        let d = symmetry.irreps().len();
        let (mut dense, mut tiled) = (vec![0.0; d], vec![0.0; d]);
        arena.scope(|arena| symmetry.irrep_norms_dense(&matrix.view(), &mut dense, arena).unwrap());
        arena.scope(|arena| symmetry.irrep_norms(AoBasis::Contracted, &matrix, &mut tiled, arena).unwrap());
        assert!(vector_math::distance(&dense, &tiled) < 1e-10, "{what}: {dense:?} vs {tiled:?}");
        // The parts are orthogonal: their squared norms add up to the matrix's.
        let total = tiled.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((total - vector_math::norm(matrix.get_data())).abs() < 1e-10, "{what}");
    }
}

#[test]
fn a_strict_clean_refuses_without_touching_and_accepts_noise() {
    let (_, symmetry) = molecules().swap_remove(2);
    let n = symmetry.n_ao();
    let mut scratch = scratch(&symmetry);
    let mut arena = scratch.get_arena();
    // A symmetric matrix: the projection of noise.
    let mut symmetric = noise(n);
    arena.scope(|arena| symmetry.average_matrix(AoBasis::Contracted, &mut symmetric, OperatorSymmetry::TotallySymmetric, arena).unwrap());
    // On a hydrogen's last AO (a pair of two N s functions would be totally symmetric already).
    let mut noisy = symmetric.clone();
    noisy[(n - 1, 0)] += 1e-9;
    let report = arena.scope(|arena| symmetry.symmetrise_matrix(AoBasis::Contracted, &mut noisy, OperatorSymmetry::TotallySymmetric, 1e-8, arena).unwrap());
    assert!(report.max_abs_removed > 1e-10 && vector_math::distance(noisy.get_data(), symmetric.get_data()) < 1e-9);
    let mut asymmetric = noise(n);
    let before = asymmetric.clone();
    assert!(arena.scope(|arena| symmetry.symmetrise_matrix(AoBasis::Contracted, &mut asymmetric, OperatorSymmetry::TotallySymmetric, 1e-8, arena)).is_err());
    assert_eq!(asymmetric.get_data(), before.get_data());
}

// Timings of the dense and the tiled projection (totally symmetric) on a large synthetic ring: D6h and its subgroups
// D2h, C2v, Cs (h = 24, 8, 4, 2), the molecule in the standard orientation (the fast path where the group allows).
#[test]
#[ignore]
fn timings() {
    let mut kinds = Vec::new();
    let mut positions = Vec::new();
    for k in 0..6 {
        let t = k as f64 * std::f64::consts::PI / 3.0;
        kinds.push(1);
        positions.push([2.64 * t.cos(), 2.64 * t.sin(), 0.0]);
        kinds.push(0);
        positions.push([4.69 * t.cos(), 4.69 * t.sin(), 0.0]);
    }
    let centres: Vec<SymmetryCentre> =
        kinds.iter().zip(&positions).map(|(&kind, &position)| SymmetryCentre { kind, mass: if kind == 0 { 1.0 } else { 12.0 }, position }).collect();
    for (label, heavy, light) in [
        ("contracted-size", vec![(0, 4), (1, 3), (2, 2), (3, 1)], vec![(0, 3), (1, 2), (2, 1)]),
        ("primitive-size", vec![(0, 20), (1, 15), (2, 10), (3, 5)], vec![(0, 10), (1, 6), (2, 3)]),
    ] {
        let functions = basis(&kinds, &heavy, &light);
        let n = functions.len();
        for group in ["D6h", "D2h", "C2v", "Cs"] {
            let request = PointGroupRequest::Named { name: group, principal_axis: Some([0.0, 0.0, 1.0]) };
            let symmetry = Symmetry::new(&centres, &functions, None, request, DEFAULT_GEOMETRY_TOLERANCE).unwrap();
            let matrix = noise(n);
            let mut scratch = scratch(&symmetry);
            let mut arena = scratch.get_arena();
            let start = std::time::Instant::now();
            arena.scope(|arena| {
                let mut cleaned = MatrixViewMut::<f64>::from_arena(n, n, arena).unwrap();
                symmetry.project_dense(&matrix.view(), OperatorSymmetry::TotallySymmetric, &mut cleaned, arena).unwrap();
            });
            let dense = start.elapsed();
            let mut tiled = matrix.clone();
            let start = std::time::Instant::now();
            arena.scope(|arena| symmetry.average_matrix(AoBasis::Contracted, &mut tiled, OperatorSymmetry::TotallySymmetric, arena).unwrap());
            let average = start.elapsed();
            let start = std::time::Instant::now();
            arena.scope(|arena| symmetry.analyse_matrix(AoBasis::Contracted, &matrix, OperatorSymmetry::TotallySymmetric, arena).unwrap());
            let report_pass = start.elapsed();
            let fast = symmetry.shapes.iter().all(|shape| shape.permutations.exact.iter().all(|&e| e));
            eprintln!(
                "{label:16} n = {n:5} {group:4} h = {:2} fast path {fast:5}: dense {:8.2} ms, tiled one pass {:8.2} ms (a report pass {:8.2} ms)",
                symmetry.order(),
                dense.as_secs_f64() * 1e3,
                average.as_secs_f64() * 1e3,
                report_pass.as_secs_f64() * 1e3,
            );
        }
    }
}

// A second, primitive basis (more shells, the same kinds of centre): its own blocks and images, the same group;
// cleaning there agrees with a symmetry built on that basis alone, and the contracted side is untouched by it.
#[test]
fn a_primitive_basis_gets_its_own_representation() {
    let water = [[0.0, 0.0, 0.2217], [1.4309, 0.0, -0.8867], [-1.4309, 0.0, -0.8867]];
    let kinds = [1, 0, 0];
    let centres: Vec<SymmetryCentre> = kinds.iter().zip(&water).map(|(&kind, &position)| SymmetryCentre { kind, mass: 1.0 + kind as f64, position }).collect();
    let contracted = basis(&kinds, &HEAVY, &LIGHT);
    let primitive = basis(&kinds, &[(0, 6), (1, 4), (2, 2), (3, 1)], &[(0, 4), (1, 2)]);
    let both = Symmetry::new(&centres, &contracted, Some(&primitive), PointGroupRequest::Highest, DEFAULT_GEOMETRY_TOLERANCE).unwrap();
    let alone = Symmetry::new(&centres, &primitive, None, PointGroupRequest::Highest, DEFAULT_GEOMETRY_TOLERANCE).unwrap();
    assert_eq!(both.n_ao_of(AoBasis::Primitive), Some(primitive.len()));
    assert_eq!(both.n_ao_of(AoBasis::Contracted), Some(contracted.len()));
    assert_eq!(alone.n_ao_of(AoBasis::Primitive), None);

    let mut scratch = scratch(&alone);
    let mut arena = scratch.get_arena();
    let matrix = noise(primitive.len());
    let (mut via_both, mut via_alone) = (matrix.clone(), matrix.clone());
    arena.scope(|arena| both.average_matrix(AoBasis::Primitive, &mut via_both, OperatorSymmetry::TotallySymmetric, arena).unwrap());
    arena.scope(|arena| alone.average_matrix(AoBasis::Contracted, &mut via_alone, OperatorSymmetry::TotallySymmetric, arena).unwrap());
    assert!(vector_math::distance(via_both.get_data(), via_alone.get_data()) < 1e-13);
    // The wrong size for a basis is refused.
    let mut small = noise(contracted.len());
    assert!(arena.scope(|arena| both.average_matrix(AoBasis::Primitive, &mut small, OperatorSymmetry::TotallySymmetric, arena)).is_err());
}

// Tensor averaging is a projection (a second one removes nothing), and the axial and the polar treatments differ where
// the group has improper operations (C2v's mirrors) but agree where it has none (C2, a subgroup).
#[test]
fn tensor_averaging_is_a_projection_and_tells_axial_from_polar() {
    let water = [[0.0, 0.0, 0.2217], [1.4309, 0.0, -0.8867], [-1.4309, 0.0, -0.8867]];
    for (group, differ) in [("C2v", true), ("C2", false)] {
        let symmetry = symmetry(&[1, 0, 0], &water, PointGroupRequest::Named { name: group, principal_axis: None });
        let n = symmetry.n_ao();
        let mut scratch = scratch(&symmetry);
        let mut arena = scratch.get_arena();
        let noisy: Vec<f64> = noise(n).get_data().iter().chain(noise(n).get_data().iter().rev()).chain(noise(n).get_data()).copied().collect();
        let mut averaged = Vec::new();
        for kind in [TensorKind::Polar, TensorKind::Axial] {
            let mut data = noisy.clone();
            let mut planes = Array3DViewMut::new(n, n, 3, &mut data);
            let first = arena.scope(|arena| symmetry.average_tensor(AoBasis::Contracted, &mut planes, 1, kind, arena).unwrap());
            assert!(first.max_abs_removed > 1e-3, "{group} {kind:?}: noise is not symmetric");
            let second = arena.scope(|arena| symmetry.average_tensor(AoBasis::Contracted, &mut planes, 1, kind, arena).unwrap());
            assert!(second.max_abs_removed < 1e-12, "{group} {kind:?}: projecting twice removed {:.3e}", second.max_abs_removed);
            averaged.push(data);
        }
        let difference = vector_math::distance(&averaged[0], &averaged[1]);
        assert_eq!(difference > 1e-6, differ, "{group}: polar and axial differ by {difference:.3e}");
    }
}

// Cartesian d and f shells: their D(R) blocks are not orthogonal (each component normalized on its own), so the cleaned
// tile is carried back with true inverses. Tiled against dense for every irrep, the irreps' parts adding up to the
// matrix, and a second projection removing nothing; standard orientation (signed permutations) and rotated (dense).
#[test]
fn cartesian_shells_project_like_the_dense_reference() {
    let water = [[0.0, 0.0, 0.2217], [1.4309, 0.0, -0.8867], [-1.4309, 0.0, -0.8867]];
    let third = 2.0 * std::f64::consts::PI / 3.0;
    let ammonia: Vec<[f64; 3]> = std::iter::once([0.0, 0.0, 0.13]).chain((0..3).map(|k| [1.77 * (k as f64 * third).cos(), 1.77 * (k as f64 * third).sin(), -0.68])).collect();
    for (what, kinds, positions) in [
        ("water", vec![1, 0, 0], water.to_vec()),
        ("water, rotated", vec![1, 0, 0], water.iter().map(|&p| rotate(p)).collect()),
        ("ammonia", vec![1, 0, 0, 0], ammonia),
    ] {
        let centres: Vec<SymmetryCentre> =
            kinds.iter().zip(&positions).map(|(&kind, &position)| SymmetryCentre { kind, mass: 1.0 + kind as f64, position }).collect();
        let functions = basis_of(&kinds, &HEAVY, &[(0, 1), (1, 1), (2, 1)], true);
        let symmetry = Symmetry::new(&centres, &functions, None, PointGroupRequest::Highest, DEFAULT_GEOMETRY_TOLERANCE).unwrap();
        let n = symmetry.n_ao();
        assert_eq!(symmetry.ao_irrep_counts().iter().sum::<usize>(), n, "{what}");
        let matrix = noise(n);
        let mut scratch = scratch(&symmetry);
        let mut arena = scratch.get_arena();
        let mut sum = Matrix::<f64>::new(n, n);
        for k in 0..symmetry.irreps().len() {
            let dense = arena.scope(|arena| {
                let mut cleaned = MatrixViewMut::<f64>::from_arena(n, n, arena).unwrap();
                symmetry.project_dense(&matrix.view(), OperatorSymmetry::Irrep(k), &mut cleaned, arena).unwrap();
                cleaned.get_data().to_vec()
            });
            let mut tiled = matrix.clone();
            arena.scope(|arena| symmetry.average_matrix(AoBasis::Contracted, &mut tiled, OperatorSymmetry::Irrep(k), arena).unwrap());
            let difference = vector_math::distance(&dense, tiled.get_data());
            assert!(difference < 1e-11 * vector_math::norm(&dense).max(1.0), "{what}, irrep {k}: tiled and dense differ by {difference:.3e}");
            let again = arena.scope(|arena| symmetry.average_matrix(AoBasis::Contracted, &mut tiled, OperatorSymmetry::Irrep(k), arena).unwrap());
            assert!(again.max_abs_removed < 1e-11, "{what}, irrep {k}: projecting twice removed {:.3e}", again.max_abs_removed);
            vector_math::add_in_place(sum.get_data_mut(), tiled.get_data());
        }
        let difference = vector_math::distance(sum.get_data(), matrix.get_data());
        assert!(difference < 1e-10, "{what}: the irreps' parts add up to the matrix only within {difference:.3e}");
    }
}
