// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Point-group symmetry through the context (set_symmetry / symmetrise_matrix / symmetrise_orbitals) against real
// integrals: matrices that are symmetric by construction (overlap, kinetic, nuclear attraction, dipole about the centre
// of mass) must come out of the cleaning unchanged in any orientation of the molecule, including groups with degenerate
// irreps; noise must be removed and reported; asymmetry beyond the tolerance refused; and the point group must follow
// what makes centres equivalent (geometry, basis set).

mod common;

use aedelsten::arrays::matrix::{Factor, Matrix, MatrixMath, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, Cleaning, ContractionType, MomentOrigin, OneElectronOperator, OperatorSymmetry, OrbitalLabel, PointGroupRequest, SymmetryOptions};

const TIGHT: f64 = 1e-11;

fn context(atoms: &[(&str, &str, [f64; 3])]) -> AedelstenContext {
    context_of(atoms, ContractionType::General)
}

fn context_of(atoms: &[(&str, &str, [f64; 3])], contraction: ContractionType) -> AedelstenContext {
    let mut input = String::new();
    for (label, basis, p) in atoms {
        input.push_str(&format!("Basis set\n    {basis}\n    {label} {:.16} {:.16} {:.16}\nEnd of Basis\n", p[0], p[1], p[2]));
    }
    let path = common::temp_path("symmetry.inp");
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn rotate(p: [f64; 3]) -> [f64; 3] {
    // An arbitrary rotation (about z by 0.3, then about x by 0.7) plus a shift: nothing aligned with anything.
    let (a, b): (f64, f64) = (0.3, 0.7);
    let q = [a.cos() * p[0] - a.sin() * p[1], a.sin() * p[0] + a.cos() * p[1], p[2]];
    [q[0] + 0.4, b.cos() * q[1] - b.sin() * q[2] - 1.1, b.sin() * q[1] + b.cos() * q[2] + 2.3]
}

fn water(transform: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<(&'static str, &'static str, [f64; 3])> {
    vec![
        ("O1", "O.x2c-SVPall", transform([0.0, 0.0, 0.2217])),
        ("H1", "H.x2c-SVPall", transform([0.0, 1.4309, -0.8867])),
        ("H2", "H.x2c-SVPall", transform([0.0, -1.4309, -0.8867])),
    ]
}

fn ammonia(transform: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<(&'static str, &'static str, [f64; 3])> {
    let h = |t: f64| transform([1.77 * t.cos(), 1.77 * t.sin(), -0.68]);
    let third = 2.0 * std::f64::consts::PI / 3.0;
    vec![
        ("N1", "N.x2c-SVPall", transform([0.0, 0.0, 0.13])),
        ("H1", "H.x2c-SVPall", h(0.0)),
        ("H2", "H.x2c-SVPall", h(third)),
        ("H3", "H.x2c-SVPall", h(2.0 * third)),
    ]
}

fn ethylene() -> Vec<(&'static str, &'static str, [f64; 3])> {
    vec![
        ("C1", "C.x2c-SVPall", [0.0, 0.0, 1.26]),
        ("C2", "C.x2c-SVPall", [0.0, 0.0, -1.26]),
        ("H1", "H.x2c-SVPall", [0.0, 1.75, 2.33]),
        ("H2", "H.x2c-SVPall", [0.0, -1.75, 2.33]),
        ("H3", "H.x2c-SVPall", [0.0, 1.75, -2.33]),
        ("H4", "H.x2c-SVPall", [0.0, -1.75, -2.33]),
    ]
}

fn one_electron(ctx: &AedelstenContext, operator: OneElectronOperator<'_>) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut m = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut m, operator, false).unwrap();
    m
}

// S, T and V: totally symmetric, so cleaning must change nothing.
fn assert_s_t_v_unchanged(ctx: &AedelstenContext, what: &str) {
    for (name, mut m) in [
        ("S", one_electron(ctx, OneElectronOperator::overlap())),
        ("T", one_electron(ctx, OneElectronOperator::kinetic())),
        ("V", one_electron(ctx, ctx.get_nuclear_attraction_operator())),
    ] {
        let before = m.clone();
        let report = ctx.symmetrise_matrix(&mut m, Cleaning::Strict).unwrap();
        assert!(report.max_abs_removed < TIGHT, "{what}: cleaning {name} removed {:.3e}", report.max_abs_removed);
        let changed = before.iter().zip(&m).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(changed < TIGHT, "{what}: {name} changed by {changed:.3e}");
    }
}

#[test]
fn symmetric_matrices_survive_cleaning_in_any_orientation() {
    for (what, atoms, group) in [
        ("water in yz", water(|p| p), "C2v"),
        ("water in xz", water(|p| [p[1], p[0], p[2]]), "C2v"),
        ("water rotated and shifted", water(rotate), "C2v"),
        ("ammonia", ammonia(|p| p), "C3v"),
        ("ammonia rotated and shifted", ammonia(rotate), "C3v"),
        ("ethylene", ethylene(), "D2h"),
    ] {
        let mut ctx = context(&atoms);
        ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
        assert_eq!(ctx.get_point_group_name().unwrap(), group, "{what}");
        assert_s_t_v_unchanged(&ctx, what);
    }
}

#[test]
fn dipole_components_are_cleaned_as_their_own_irreps() {
    let mut ctx = context(&water(|p| p));
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let n = ctx.get_n_cont();
    let origin = ctx.compute_centre_of_mass();
    let mut dipole = vec![0.0; 3 * n * n];
    ctx.compute_one_electron_integrals(&mut dipole, OneElectronOperator::moment(MomentOrigin::new(origin, 1)), false).unwrap();

    let names: Vec<String> = ctx.get_irreps().unwrap().iter().map(|i| i.name.clone()).collect();
    let mut irreps = Vec::new();
    for component in dipole.chunks_exact_mut(n * n) {
        let before = component.to_vec();
        // As totally symmetric, x and y (B1/B2 in C2v) are almost entirely removed: refused, nothing touched.
        let as_a1 = ctx.symmetrise_matrix(component, Cleaning::Strict);
        if as_a1.is_err() {
            assert_eq!(before, component, "a refused cleaning must leave the matrix alone");
        }
        let report = ctx.analyse_matrix(component, OperatorSymmetry::Auto).unwrap();
        assert!(report.max_abs_removed < TIGHT, "dipole cleaned as {}: removed {:.3e}", names[report.irrep], report.max_abs_removed);
        irreps.push((names[report.irrep].clone(), as_a1.is_ok()));
    }
    // z is along C2: A1, accepted as totally symmetric too. x and y: the two B irreps, refused as A1.
    assert_eq!(irreps[2], ("A1".to_string(), true));
    assert!(irreps[0].0.starts_with('B') && irreps[1].0.starts_with('B') && irreps[0].0 != irreps[1].0, "{irreps:?}");
    assert!(!irreps[0].1 && !irreps[1].1);
}

#[test]
fn operator_irreps_are_exposed() {
    // libmsym's standard orientation of C2v has the molecular plane in xz: water in xz is in it, water in yz is turned
    // by 90 degrees about z (still axis-aligned, so x and y stay pure irreps, the B's swapped), the rotated one is not.
    let mut standard = context(&water(|p| [p[1], p[0], p[2]]));
    standard.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let mut turned = context(&water(|p| p));
    turned.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let mut rotated = context(&water(rotate));
    rotated.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();

    let name = |ctx: &AedelstenContext, irrep: Option<usize>| irrep.map(|k| ctx.get_irreps().unwrap()[k].name.clone());
    let irrep = |ctx: &AedelstenContext, l: (u32, u32, u32)| name(ctx, ctx.get_cartesian_irrep(l.0, l.1, l.2).unwrap());
    for ctx in [&standard, &turned] {
        assert_eq!(irrep(ctx, (0, 0, 0)).as_deref(), Some("A1"));
        assert_eq!(irrep(ctx, (0, 0, 1)).as_deref(), Some("A1"));
        assert_eq!(irrep(ctx, (1, 1, 0)).as_deref(), Some("A2"));
        assert_eq!(irrep(ctx, (2, 0, 0)).as_deref(), Some("A1"));
    }
    // x is in the plane in the standard orientation and B1 there (libmsym's convention); in the turned frame x is
    // perpendicular to the plane, so the B's swap.
    let (x_standard, y_standard) = (irrep(&standard, (1, 0, 0)).unwrap(), irrep(&standard, (0, 1, 0)).unwrap());
    assert!(x_standard.starts_with('B') && y_standard.starts_with('B') && x_standard != y_standard);
    assert_eq!(irrep(&turned, (1, 0, 0)), Some(y_standard));
    assert_eq!(irrep(&turned, (0, 1, 0)), Some(x_standard));
    // Rotated: x, y and z are mixtures; only the totally symmetric s stays one irrep.
    assert_eq!(irrep(&rotated, (1, 0, 0)), None);
    assert_eq!(irrep(&rotated, (0, 0, 1)), None);
    assert_eq!(irrep(&rotated, (0, 0, 0)).as_deref(), Some("A1"));

    let irreps = standard.get_irreps().unwrap();
    assert_eq!(standard.get_irrep_index("b2").unwrap(), irreps.iter().position(|i| i.name == "B2"));
    assert_eq!(standard.get_irrep_index("E").unwrap(), None);
    assert_eq!(irreps[standard.get_totally_symmetric_irrep().unwrap()].name, "A1");

    // C3v: z and z^2 are A1, x and y belong to E together, so neither alone.
    let mut nh3 = context(&ammonia(|p| p));
    nh3.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    assert_eq!(irrep(&nh3, (0, 0, 1)).as_deref(), Some("A1"));
    assert_eq!(irrep(&nh3, (0, 0, 2)).as_deref(), Some("A1"));
    assert_eq!(irrep(&nh3, (1, 0, 0)), None);
    assert_eq!(irrep(&nh3, (0, 1, 0)), None);
    // A matrix still splits into irreps there (a degenerate one as a whole): the overlap is all A1.
    let n = nh3.get_n_cont();
    let irreps = nh3.get_irreps().unwrap();
    let mut norms = vec![0.0; irreps.len()];
    let overlap = one_electron(&nh3, OneElectronOperator::overlap());
    nh3.irrep_norms(&overlap, &mut norms).unwrap();
    for (irrep, &norm) in irreps.iter().zip(&norms) {
        if irrep.name == "A1" {
            assert!((norm - aedelsten::arrays::vector_math::norm(&overlap)).abs() < 1e-9, "{norms:?}");
        } else {
            assert!(norm < 1e-6, "{}: {norm:.3e}", irrep.name);
        }
    }
    let _ = n;
}

#[test]
fn dipole_irreps_agree_between_cartesian_irrep_irrep_norms_and_auto() {
    let mut ctx = context(&water(|p| [p[1], p[0], p[2]]));
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let n = ctx.get_n_cont();
    let origin = ctx.compute_centre_of_mass();
    let mut dipole = vec![0.0; 3 * n * n];
    ctx.compute_one_electron_integrals(&mut dipole, OneElectronOperator::moment(MomentOrigin::new(origin, 1)), false).unwrap();
    let names: Vec<String> = ctx.get_irreps().unwrap().iter().map(|i| i.name.clone()).collect();
    let mut norms = vec![0.0; names.len()];
    for (axis, component) in dipole.chunks_exact_mut(n * n).enumerate() {
        let l = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][axis];
        let expected = ctx.get_cartesian_irrep(l.0, l.1, l.2).unwrap().unwrap();
        // All of the matrix is in that irrep.
        ctx.irrep_norms(component, &mut norms).unwrap();
        for (k, &norm) in norms.iter().enumerate() {
            if k != expected {
                assert!(norm < TIGHT, "axis {axis}: {} carries {norm:.3e}", names[k]);
            }
        }
        // analyse_matrix reports, and changes nothing; the explicit irrep and Auto agree.
        let before = component.to_vec();
        let analysed = ctx.analyse_matrix(component, OperatorSymmetry::Irrep(expected)).unwrap();
        assert_eq!(before, component);
        assert!(analysed.max_abs_removed < TIGHT);
        let auto = ctx.analyse_matrix(component, OperatorSymmetry::Auto).unwrap();
        assert_eq!(auto.irrep, expected);
    }
}

#[test]
fn noise_is_refused_when_strict_and_averaged_away_otherwise() {
    let mut ctx = context(&ammonia(rotate));
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let clean = one_electron(&ctx, OneElectronOperator::kinetic());
    let mut noisy = clean.clone();
    for (k, v) in noisy.iter_mut().enumerate() {
        *v += 1e-7 * (((k * 7919) % 13) as f64 / 13.0 - 0.5);
    }
    // Strict (tolerance 1e-10 by default): refused, untouched.
    let before = noisy.clone();
    assert!(ctx.symmetrise_matrix(&mut noisy, Cleaning::Strict).is_err());
    assert_eq!(noisy, before);
    // Averaged: removed and reported; what is left is symmetric (a strict clean accepts it) and within the noise of
    // the clean matrix.
    let report = ctx.symmetrise_matrix(&mut noisy, Cleaning::Average).unwrap();
    assert!(report.max_abs_removed > 1e-9 && report.max_abs_removed < 1e-7, "{report:?}");
    let again = ctx.symmetrise_matrix(&mut noisy, Cleaning::Strict).unwrap();
    assert!(again.max_abs_removed < 1e-14, "{again:?}");
    let off = clean.iter().zip(&noisy).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(off < 1e-7);
    // A looser strict tolerance accepts the noise.
    ctx.free_symmetry();
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions { strict_tolerance: 1e-6, ..SymmetryOptions::default() }).unwrap();
    assert!(ctx.symmetrise_matrix(&mut before.clone(), Cleaning::Strict).is_ok());
    // Nonsense tolerances are refused.
    ctx.free_symmetry();
    assert!(ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions { strict_tolerance: -1.0, ..SymmetryOptions::default() }).is_err());
}

#[test]
fn subgroups_are_selected_by_name_and_axis() {
    let mut ctx = context(&ethylene());
    let named = |name: &'static str, principal_axis| PointGroupRequest::Named { name, principal_axis };
    for (request, group) in [
        (named("d2h", None), "D2h"),
        (named("C2v", None), "C2v"),
        (named("C2v", Some([-2.0, 0.0, 0.0])), "C2v"),
        (named("C2h", Some([1.0, 0.0, 0.0])), "C2h"),
        (named("D2", None), "D2"),
    ] {
        ctx.set_symmetry(request, SymmetryOptions::default()).unwrap();
        assert_eq!(ctx.get_point_group_name().unwrap(), group);
        let axes = ctx.get_symmetry_standard_axes().unwrap();
        if let PointGroupRequest::Named { principal_axis: Some(axis), .. } = request {
            let z = axes[2];
            assert!((z[0] * axis[0] + z[1] * axis[1] + z[2] * axis[2]).abs() > 0.999 * axis.iter().map(|a| a * a).sum::<f64>().sqrt());
        } else if group == "C2v" {
            assert!(axes[2][2].abs() > 0.999, "C2v by default has its C2 along z");
        }
        assert_s_t_v_unchanged(&ctx, group);
        ctx.free_symmetry();
    }
    // Not a subgroup; a subgroup in no orientation along the axis asked for.
    assert!(ctx.set_symmetry(named("C3v", None), SymmetryOptions::default()).is_err());
    assert!(ctx.set_symmetry(named("C2v", Some([1.0, 1.0, 0.0])), SymmetryOptions::default()).is_err());
    assert!(ctx.get_point_group_name().is_err());
}

#[test]
fn asymmetric_geometry_and_inequivalent_centres() {
    // 1e-5 off: recognized as C2v by libmsym's tolerances, then refused by the geometry check.
    let mut atoms = water(|p| p);
    atoms[1].2[1] += 1e-5;
    let mut ctx = context(&atoms);
    let err = ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap_err().to_string();
    assert!(err.contains("not exactly C2v-symmetric"), "{err}");
    // ... and accepted with a looser tolerance.
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions { geometry_tolerance: 1e-4, ..SymmetryOptions::default() }).unwrap();

    // Same geometry, different basis sets on the two hydrogens: they are not equivalent, only the plane is left.
    let mut atoms = water(|p| p);
    atoms[2].1 = "H.ANO-RCC-VDZP";
    let mut ctx = context(&atoms);
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    assert_eq!(ctx.get_point_group_name().unwrap(), "Cs");
    assert_s_t_v_unchanged(&ctx, "water with inequivalent hydrogens");
}

#[test]
fn orbitals_are_symmetrized_and_labelled() {
    for atoms in [water(|p| p), water(rotate), ammonia(rotate)] {
        let mut ctx = context(&atoms);
        ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
        let n = ctx.get_n_cont();
        // Core-Hamiltonian orbitals: C = S^-1/2 U, U the eigenvectors of S^-1/2 H S^-1/2.
        let s = Matrix::new_from_vec(n, n, one_electron(&ctx, OneElectronOperator::overlap()).into_boxed_slice());
        let t = one_electron(&ctx, OneElectronOperator::kinetic());
        let v = one_electron(&ctx, ctx.get_nuclear_attraction_operator());
        let h = Matrix::new_from_vec(n, n, t.iter().zip(&v).map(|(a, b)| a + b).collect());
        let x = s.symmetric_inverse_sqrt();
        let (_, u) = h.congruence_transform(&x).eigenvectors();
        let mut c = x.matmul(&u);

        let mut labels = vec![OrbitalLabel::default(); n];
        let before = c.clone();
        let dropped = ctx.symmetrise_orbitals(c.get_data_mut(), &mut labels).unwrap();
        assert!(dropped < 1e-8, "symmetrizing exact eigenvectors dropped {dropped:.3e}");
        // Still S-orthonormal, and every irrep's MOs span the same space as before (degenerate partners may have been
        // rotated among themselves): the same projector C_k C_k^T.
        let overlap = s.congruence_transform(&c);
        for j in 0..n {
            for i in 0..n {
                assert!((overlap[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-9, "C^T S C ({i}, {j}) = {}", overlap[(i, j)]);
            }
        }
        let irreps = ctx.get_irreps().unwrap();
        let projector = |m: &Matrix<f64>, keep: &dyn Fn(OrbitalLabel) -> bool| {
            let mut p = Matrix::<f64>::new(n, n);
            for o in (0..n).filter(|&o| keep(labels[o])) {
                for j in 0..n {
                    for i in 0..n {
                        p[(i, j)] += m[(i, o)] * m[(j, o)];
                    }
                }
            }
            p
        };
        let max_diff = |a: &Matrix<f64>, b: &Matrix<f64>| a.get_data().iter().zip(b.get_data()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        for k in 0..irreps.len() {
            let diff = max_diff(&projector(&before, &|l| l.irrep == k), &projector(&c, &|l| l.irrep == k));
            assert!(diff < 1e-9, "{}: the MOs' span moved by {diff:.3e}", irreps[k].name);
            // As many MOs as the irrep has AOs; for a degenerate irrep, as many of each component, and each component's
            // MOs are the images of the others' (the same projector after the transfer: same count, checked here).
            let counts: Vec<usize> = (0..irreps[k].dimension).map(|j| labels.iter().filter(|l| l.irrep == k && l.component == j).count()).collect();
            assert_eq!(counts.iter().sum::<usize>(), ctx.get_ao_irrep_counts().unwrap()[k], "{} in {}", irreps[k].name, ctx.get_point_group_name().unwrap());
            assert!(counts.iter().all(|&c| c == counts[0]), "{}: components {counts:?}", irreps[k].name);
        }

        // Fewer MOs than AOs (the lowest half): the same labels and coefficients as the full set's.
        let half = n / 2;
        let mut part = before.view_columns(0..half).get_data().to_vec();
        let mut part_labels = vec![OrbitalLabel::default(); half];
        ctx.symmetrise_orbitals(&mut part, &mut part_labels).unwrap();
        assert_eq!(part_labels.iter().map(|l| l.irrep).collect::<Vec<_>>(), labels[..half].iter().map(|l| l.irrep).collect::<Vec<_>>());
    }
}

// Cartesian shells from d up (each component normalized on its own, so D(R) is not orthogonal there): the library's
// integrals clean strictly in both bases, dipoles survive as a tensor, and core-Hamiltonian orbitals of rotated water
// and of ammonia (E: partner alignment, in the invariant metric) come out symmetric, S-orthonormal and labelled.
#[test]
fn cartesian_shells_are_supported() {
    for (what, atoms) in [("water, rotated", water(rotate)), ("ammonia", ammonia(|p| p))] {
        let mut input = String::new();
        for (label, basis, p) in &atoms {
            input.push_str(&format!("Basis set\n    {basis}\n    Cartesian all\n    {label} {:.16} {:.16} {:.16}\nEnd of Basis\n", p[0], p[1], p[2]));
        }
        let path = common::temp_path("symmetry_cartesian.inp");
        std::fs::write(&path, input).unwrap();
        let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
        std::fs::remove_file(&path).ok();
        ctx.set_shell_pairs(None).unwrap();
        ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
        let mut ids = vec![aedelsten::BasisFunctionId::default(); ctx.get_n_cont()];
        ctx.get_basis_function_ids(&mut ids).unwrap();
        assert!(ids.iter().any(|id| id.get_l() < -1), "{what}: expected Cartesian d shells");

        ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
        ctx.set_primitive_shell_pairs(None).unwrap();
        for primitive in [false, true] {
            let n = if primitive { ctx.get_n_prim_unique() } else { ctx.get_n_cont() };
            for (name, operator) in [
                ("oneint_overlap", OneElectronOperator::overlap()),
                ("oneint_kinetic", OneElectronOperator::kinetic()),
                ("oneint_nuclear_attraction", ctx.get_nuclear_attraction_operator()),
            ] {
                let mut m = vec![0.0; n * n];
                ctx.compute_one_electron_integrals(&mut m, operator, primitive).unwrap();
                let report = ctx.get_symmetrisation_report(name, primitive).unwrap().unwrap();
                assert!(report.max_abs_removed < TIGHT, "{what}, primitive {primitive}, {name}: {report:?}");
            }
        }
        let n = ctx.get_n_cont();
        let mut dipole = vec![0.0; 3 * n * n];
        ctx.compute_one_electron_integrals(&mut dipole, OneElectronOperator::moment(MomentOrigin::new(ctx.compute_centre_of_mass(), 1)), false).unwrap();
        assert!(ctx.get_symmetrisation_report("oneint_moment", false).unwrap().unwrap().max_abs_removed < TIGHT, "{what}");

        let s = Matrix::new_from_vec(n, n, one_electron(&ctx, OneElectronOperator::overlap()).into_boxed_slice());
        let t = one_electron(&ctx, OneElectronOperator::kinetic());
        let v = one_electron(&ctx, ctx.get_nuclear_attraction_operator());
        let h = Matrix::new_from_vec(n, n, t.iter().zip(&v).map(|(a, b)| a + b).collect());
        let x = s.symmetric_inverse_sqrt();
        let (_, u) = h.congruence_transform(&x).eigenvectors();
        let mut c = x.matmul(&u);
        let mut labels = vec![OrbitalLabel::default(); n];
        let dropped = ctx.symmetrise_orbitals(c.get_data_mut(), &mut labels).unwrap();
        assert!(dropped < 1e-8, "{what}: dropped {dropped:.3e}");
        let overlap = s.congruence_transform(&c);
        for j in 0..n {
            for i in 0..n {
                assert!((overlap[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-9, "{what}: C^T S C ({i}, {j}) = {}", overlap[(i, j)]);
            }
        }
        let irreps = ctx.get_irreps().unwrap();
        for k in 0..irreps.len() {
            let counts: Vec<usize> = (0..irreps[k].dimension).map(|j| labels.iter().filter(|l| l.irrep == k && l.component == j).count()).collect();
            assert_eq!(counts.iter().sum::<usize>(), ctx.get_ao_irrep_counts().unwrap()[k], "{what}: {}", irreps[k].name);
            assert!(counts.iter().all(|&c| c == counts[0]), "{what}: {} components {counts:?}", irreps[k].name);
        }
    }
}

#[test]
fn moving_the_centres_invalidates_the_symmetry() {
    let mut ctx = context(&water(|p| p));
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let mut positions = vec![0.0; 3 * ctx.get_n_gradient_centres()];
    let mut geometry = vec![aedelsten::CentreGeometry { atomic_number: 0, position: [0.0; 3], kind: aedelsten::AtomKind::Real }; ctx.get_n_centres()];
    ctx.get_centre_geometry(&mut geometry).unwrap();
    for (chunk, centre) in positions.chunks_exact_mut(3).zip(&geometry) {
        chunk.copy_from_slice(&centre.position);
    }
    positions[4] += 0.1;
    ctx.set_centre_positions(&positions).unwrap();
    assert!(ctx.get_point_group_name().is_err());
    ctx.free_symmetry();
    ctx.set_shell_pairs(None).ok();
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    assert_eq!(ctx.get_point_group_name().unwrap(), "Cs");
}

// With symmetry set, the library cleans its own one-electron integrals: strictly (S, T, V are exact), in the
// contracted and in the primitive basis (from the library's own decontraction: a segmented basis repeats exponents
// across its contracted functions, and the primitive basis has each once), the report kept per output and basis.
#[test]
fn library_integrals_are_cleaned_strictly_in_both_bases() {
    // Segmented: water in x2c-SVPall, a segmented set (ANO-RCC is generally contracted).
    let segmented: Vec<_> = water(rotate).into_iter().map(|(label, _, p)| (label, if label == "O1" { "O.x2c-SVPall" } else { "H.x2c-SVPall" }, p)).collect();
    for (atoms, contraction) in [(water(rotate), ContractionType::General), (segmented, ContractionType::Segmented)] {
        let mut ctx = context_of(&atoms, contraction);
        assert!(ctx.get_symmetrisation_report("oneint_overlap", false).is_err(), "no symmetry, no reports");
        ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
        ctx.set_primitive_shell_pairs(None).unwrap();
        for primitive in [false, true] {
            let n = if primitive { ctx.get_n_prim_unique() } else { ctx.get_n_cont() };
            assert!(ctx.get_symmetrisation_report("oneint_overlap", primitive).unwrap().is_none());
            for (name, operator) in [
                ("oneint_overlap", OneElectronOperator::overlap()),
                ("oneint_kinetic", OneElectronOperator::kinetic()),
                ("oneint_nuclear_attraction", ctx.get_nuclear_attraction_operator()),
            ] {
                let mut m = vec![0.0; n * n];
                ctx.compute_one_electron_integrals(&mut m, operator, primitive).unwrap();
                let report = ctx.get_symmetrisation_report(name, primitive).unwrap().unwrap();
                assert!(report.max_abs_removed < TIGHT, "{contraction:?}, primitive {primitive}, {name}: {report:?}");
                if !primitive {
                    // Already symmetric: a strict clean by the caller removes nothing more.
                    assert!(ctx.symmetrise_matrix(&mut m, Cleaning::Strict).unwrap().max_abs_removed < 1e-14);
                }
            }
        }
    }
}

// X2C with symmetry set: the inputs must be symmetric (checked before the decoupling, which is nonlinear), the
// Hamiltonian and both decoupling transforms come out averaged (their asymmetry is the decoupling's rounding), the
// amounts reported. An environment added to V without averaging is refused, naming V.
#[test]
fn x2c_checks_its_inputs_and_averages_its_outputs() {
    let mut ctx = context(&water(|p| p));
    let xfield = common::temp_path("symmetry_x2c_xfield.xfield");
    std::fs::write(&xfield, "1 Bohr 0\n 3.0 2.0 1.5 0.5\n").unwrap();
    ctx.set_external_multipoles(xfield.to_str().unwrap()).unwrap();
    std::fs::remove_file(&xfield).ok();
    ctx.set_primitive_shell_pairs(None).unwrap();
    let n = ctx.get_n_prim_unique();
    // The environment's raw (unaveraged) operator, before symmetry is set.
    let mut raw_environment = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut raw_environment, ctx.get_external_multipole_operator().unwrap(), true).unwrap();

    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let integrals = |operator, picture_change: bool| {
        let mut m = vec![0.0; n * n];
        if picture_change {
            ctx.compute_one_electron_picture_change_integrals(&mut m, operator, false).unwrap();
        } else {
            ctx.compute_one_electron_integrals(&mut m, operator, true).unwrap();
        }
        m
    };
    let (s, t) = (integrals(OneElectronOperator::overlap(), false), integrals(OneElectronOperator::kinetic(), false));
    let v = integrals(ctx.get_nuclear_attraction_operator(), false);
    let pvp = integrals(ctx.get_nuclear_attraction_operator(), true);
    // The library's own environment integrals with symmetry set: averaged.
    let environment = integrals(ctx.get_external_multipole_operator().unwrap(), false);
    let pvp_report = ctx.get_symmetrisation_report("picture_change_oneint_nuclear_attraction", true).unwrap().unwrap();
    assert!(pvp_report.max_abs_removed < 1e-9 * pvp_report.norm.max(1.0), "{pvp_report:?}");

    // The environment added unaveraged: refused before the decoupling, naming V, nothing kept.
    let v_raw: Vec<f64> = v.iter().zip(&raw_environment).map(|(a, b)| a + b).collect();
    let mut oneham = vec![0.0; n * n];
    let err = ctx.construct_x2c_one_electron_hamiltonian(&mut oneham, &s, &t, &v_raw, &pvp, None).unwrap_err().to_string();
    assert!(err.starts_with("V is not symmetric under C2v"), "{err}");
    assert!(!ctx.has_picture_change());

    // Averaged first: accepted.
    let v_averaged: Vec<f64> = v.iter().zip(&environment).map(|(a, b)| a + b).collect();
    ctx.construct_x2c_one_electron_hamiltonian(&mut oneham, &s, &t, &v_averaged, &pvp, None).unwrap();
    for name in ["x2c_one_electron_hamiltonian", "x2c_u_large", "x2c_u_small"] {
        let report = ctx.get_symmetrisation_report(name, true).unwrap().unwrap();
        assert!(report.norm > 0.0 && report.max_abs_removed < 1e-9 * report.norm, "{name}: {report:?}");
    }
    // The outputs are symmetric now: transforming a symmetric operator keeps it symmetric to rounding.
    let mut transformed = vec![0.0; n * n];
    ctx.transform_one_electron_operator(&s, &s, &mut transformed).unwrap();
    let mut contracted = vec![0.0; ctx.get_n_cont() * ctx.get_n_cont()];
    ctx.contract_one_electron_matrix(&transformed, &mut contracted).unwrap();
    assert!(ctx.symmetrise_matrix(&mut contracted, Cleaning::Strict).is_ok());
}

// An external point charge off every symmetry element: its operator is averaged over the group (the environment
// replaced by the average of its images), not refused, and the report says how much went.
#[test]
fn an_asymmetric_environment_is_averaged_and_reported() {
    let mut ctx = context(&water(|p| p));
    let xfield = common::temp_path("symmetry_xfield.xfield");
    std::fs::write(&xfield, "1 Bohr 0\n 3.0 2.0 1.5 0.5\n").unwrap();
    ctx.set_external_multipoles(xfield.to_str().unwrap()).unwrap();
    std::fs::remove_file(&xfield).ok();
    let n = ctx.get_n_cont();
    let mut raw = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut raw, ctx.get_external_multipole_operator().unwrap(), false).unwrap();

    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    let mut averaged = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut averaged, ctx.get_external_multipole_operator().unwrap(), false).unwrap();
    let report = ctx.get_symmetrisation_report("oneint_external_multipole", false).unwrap().unwrap();
    assert!(report.max_abs_removed > 1e-4, "an asymmetric charge leaves a visible asymmetry: {report:?}");
    // The average is the projection of the raw matrix, and symmetric now.
    let mut projected = raw.clone();
    let by_hand = ctx.symmetrise_matrix(&mut projected, Cleaning::Average).unwrap();
    assert!((by_hand.removed_norm - report.removed_norm).abs() < 1e-12);
    assert!(projected.iter().zip(&averaged).all(|(a, b)| (a - b).abs() < 1e-13));
    assert!(ctx.symmetrise_matrix(&mut averaged, Cleaning::Strict).unwrap().max_abs_removed < 1e-13);
    // Its trace with a symmetric density is unchanged (Tr(D G(V)) = Tr(D V)): the overlap stands in for one.
    let mut overlap = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();
    let trace = |m: &[f64]| m.iter().zip(&overlap).map(|(a, b)| a * b).sum::<f64>();
    assert!((trace(&raw) - trace(&averaged)).abs() < 1e-11 * trace(&raw).abs().max(1.0));
}

// Multipole integrals about the centre of mass are cleaned as whole tensors (components mixed by the operations): real
// integrals of a symmetric molecule must come out unchanged in any orientation and group, rotated water (no component a
// pure irrep), ammonia (x and y in E together) and ethylene, dipoles and quadrupoles. About a point the operations
// move, they are left alone.
#[test]
fn multipole_tensors_survive_cleaning_in_any_orientation() {
    for (what, atoms) in [("water, rotated", water(rotate)), ("ammonia, rotated", ammonia(rotate)), ("ethylene", ethylene())] {
        let mut ctx = context(&atoms);
        let n = ctx.get_n_cont();
        let origin = ctx.compute_centre_of_mass();
        let mut raw = Vec::new();
        for order in [1, 2] {
            let mut m = vec![0.0; n * n * (order + 1) * (order + 2) / 2];
            ctx.compute_one_electron_integrals(&mut m, OneElectronOperator::moment(MomentOrigin::new(origin, order)), false).unwrap();
            raw.push(m);
        }
        ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
        for (order, raw) in [1, 2].into_iter().zip(&raw) {
            let mut m = vec![0.0; raw.len()];
            ctx.compute_one_electron_integrals(&mut m, OneElectronOperator::moment(MomentOrigin::new(origin, order)), false).unwrap();
            let report = ctx.get_symmetrisation_report("oneint_moment", false).unwrap().unwrap();
            assert!(report.max_abs_removed < TIGHT, "{what}, order {order}: {report:?}");
            let changed = raw.iter().zip(&m).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(changed < TIGHT, "{what}, order {order}: changed by {changed:.3e}");
        }
        // About a point off the symmetry elements: not cleaned, no new report.
        ctx.free_symmetry();
        ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
        let mut m = vec![0.0; n * n * 3];
        let off = [origin[0] + 0.3, origin[1] - 0.2, origin[2] + 0.1];
        ctx.compute_one_electron_integrals(&mut m, OneElectronOperator::moment(MomentOrigin::new(off, 1)), false).unwrap();
        assert!(ctx.get_symmetrisation_report("oneint_moment", false).unwrap().is_none(), "{what}");
    }
}

// The spin-orbit integrals (x, y, z: an axial vector, an extra sign under improper operations; then the scalar pVp)
// come out of a symmetric molecule unchanged: under the mirror planes of C2v, C3v and D2h a polar treatment would flip
// them and be refused.
#[test]
fn spin_orbit_integrals_are_cleaned_as_an_axial_vector() {
    for (what, atoms) in [("water, rotated", water(rotate)), ("ammonia", ammonia(|p| p)), ("ethylene", ethylene())] {
        let mut ctx = context(&atoms);
        ctx.set_primitive_shell_pairs(None).unwrap();
        let n = ctx.get_n_prim_unique();
        let mut raw = vec![0.0; 4 * n * n];
        ctx.compute_one_electron_picture_change_integrals(&mut raw, ctx.get_nuclear_attraction_operator(), true).unwrap();
        assert!(raw[..3 * n * n].iter().any(|x| x.abs() > 1e-6), "{what}: spin-orbit integrals should not vanish");
        ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
        let mut so = vec![0.0; 4 * n * n];
        ctx.compute_one_electron_picture_change_integrals(&mut so, ctx.get_nuclear_attraction_operator(), true).unwrap();
        for name in ["spin_orbit_oneint_nuclear_attraction", "picture_change_oneint_nuclear_attraction"] {
            let report = ctx.get_symmetrisation_report(name, true).unwrap().unwrap();
            assert!(report.max_abs_removed < 1e-10 * report.norm.max(1.0), "{what}, {name}: {report:?}");
        }
        // Relative to the largest element (tight primitives make them large): rounding, not a sign flip.
        let changed = raw.iter().zip(&so).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let largest = raw.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(changed < 1e-12 * largest, "{what}: changed by {changed:.3e} of {largest:.3e}");
    }
}

// A context with a card (e.g. "Cartesian d") per atom's basis block.
fn context_with_cards(atoms: &[(&str, &str, &str, [f64; 3])]) -> AedelstenContext {
    let mut input = String::new();
    for (label, basis, card, p) in atoms {
        let card = if card.is_empty() { String::new() } else { format!("    {card}\n") };
        input.push_str(&format!("Basis set\n    {basis}\n{card}    {label} {:.16} {:.16} {:.16}\nEnd of Basis\n", p[0], p[1], p[2]));
    }
    let path = common::temp_path("symmetry_cards.inp");
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

// Mixed conventions, any way: Cartesian d with spherical f on the heavy atom, spherical hydrogens (with d). Every block
// has its own shape, so nothing requires a convention throughout: integrals strict in both bases, orbitals symmetric
// and labelled. ANO-RCC-VTZP: f on O and N, d on H (x2c-SVPall has neither on light atoms).
#[test]
fn mixed_cartesian_and_spherical_shells_are_supported() {
    let h_basis = "H.ANO-RCC-VTZP";
    let heavy_basis = |label: &str| if label.starts_with('N') { "N.ANO-RCC-VTZP" } else { "O.ANO-RCC-VTZP" };
    for (what, atoms) in [("water, rotated", water(rotate)), ("ammonia", ammonia(|p| p))] {
        let with_cards: Vec<(&str, &str, &str, [f64; 3])> = atoms
            .iter()
            .map(|&(label, _, p)| if label.starts_with('H') { (label, h_basis, "", p) } else { (label, heavy_basis(label), "Cartesian d", p) })
            .collect();
        let mut ctx = context_with_cards(&with_cards);
        let mut ids = vec![aedelsten::BasisFunctionId::default(); ctx.get_n_cont()];
        ctx.get_basis_function_ids(&mut ids).unwrap();
        assert!(ids.iter().any(|id| id.get_l() == -2) && ids.iter().any(|id| id.get_l() == 3) && ids.iter().any(|id| id.get_l() == 2), "{what}: expected Cartesian d, spherical f and spherical d");
        ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
        assert_eq!(ctx.get_point_group_name().unwrap(), if what == "ammonia" { "C3v" } else { "C2v" });
        ctx.set_primitive_shell_pairs(None).unwrap();
        for primitive in [false, true] {
            let n = if primitive { ctx.get_n_prim_unique() } else { ctx.get_n_cont() };
            for (name, operator) in [("oneint_overlap", OneElectronOperator::overlap()), ("oneint_nuclear_attraction", ctx.get_nuclear_attraction_operator())] {
                let mut m = vec![0.0; n * n];
                ctx.compute_one_electron_integrals(&mut m, operator, primitive).unwrap();
                assert!(ctx.get_symmetrisation_report(name, primitive).unwrap().unwrap().max_abs_removed < TIGHT, "{what}, {name}");
            }
        }
        let n = ctx.get_n_cont();
        let s = Matrix::new_from_vec(n, n, one_electron(&ctx, OneElectronOperator::overlap()).into_boxed_slice());
        let t = one_electron(&ctx, OneElectronOperator::kinetic());
        let v = one_electron(&ctx, ctx.get_nuclear_attraction_operator());
        let h = Matrix::new_from_vec(n, n, t.iter().zip(&v).map(|(a, b)| a + b).collect());
        let x = s.symmetric_inverse_sqrt();
        let (_, u) = h.congruence_transform(&x).eigenvectors();
        let mut c = x.matmul(&u);
        let mut labels = vec![OrbitalLabel::default(); n];
        assert!(ctx.symmetrise_orbitals(c.get_data_mut(), &mut labels).unwrap() < 1e-8, "{what}");
        let overlap = s.congruence_transform(&c);
        let off = (0..n).flat_map(|j| (0..n).map(move |i| (i, j))).map(|(i, j)| (overlap[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max);
        assert!(off < 1e-9, "{what}: C^T S C off by {off:.3e}");
    }
}

// Symmetry-equivalent hydrogens given different harmonic conventions (here only H2 Cartesian): their bases differ, so
// they are not equivalent and the group is the one the basis allows, Cs, not the geometry's C2v (a warning says so).
// ANO-RCC-VTZP on the hydrogens: a d shell to make Cartesian (x2c-SVPall's H has none).
#[test]
fn equivalent_centres_with_different_conventions_lower_the_group() {
    let atoms = water(|p| p);
    let h_basis = "H.ANO-RCC-VTZP";
    let with_cards: Vec<(&str, &str, &str, [f64; 3])> = atoms
        .iter()
        .map(|&(label, basis, p)| match label {
            "H1" => (label, h_basis, "", p),
            "H2" => (label, h_basis, "Cartesian d", p),
            _ => (label, basis, "", p),
        })
        .collect();
    let mut ctx = context_with_cards(&with_cards);
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    assert_eq!(ctx.get_point_group_name().unwrap(), "Cs");
    let n = ctx.get_n_cont();
    let mut m = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut m, OneElectronOperator::overlap(), false).unwrap();
    // The same, both hydrogens alike in separate blocks: C2v (where in the input a centre is given does not matter).
    let alike: Vec<(&str, &str, &str, [f64; 3])> =
        atoms.iter().map(|&(label, basis, p)| if label.starts_with('H') { (label, h_basis, "Cartesian d", p) } else { (label, basis, "", p) }).collect();
    let mut ctx = context_with_cards(&alike);
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    assert_eq!(ctx.get_point_group_name().unwrap(), "C2v");
}

// Each operation's Cartesian matrix and centre images: orthogonal, the images a permutation of equivalent centres, and
// every vector between two centres carried onto the vector between their images (so no centre of mass is needed).
// Ammonia turned out of every axis, C3v; one operation is the identity.
#[test]
fn operation_matrices_carry_the_centres_onto_their_images() {
    let atoms = ammonia(rotate);
    let mut ctx = context(&atoms);
    ctx.set_symmetry(PointGroupRequest::Highest, SymmetryOptions::default()).unwrap();
    assert_eq!(ctx.get_point_group_name().unwrap(), "C3v");
    let h = ctx.get_symmetry_operation_count().unwrap();
    assert_eq!(h, 6);
    let n = atoms.len();
    assert_eq!(ctx.get_n_symmetry_centres().unwrap(), n);
    let mut identities = 0;
    for op in 0..h {
        let m = ctx.get_symmetry_operation_matrix(op).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| m[k][i] * m[k][j]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12, "operation {op}: not orthogonal");
            }
        }
        let mut images = vec![0; n];
        ctx.get_symmetry_centre_images(op, &mut images).unwrap();
        let mut sorted = images.clone();
        sorted.sort();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "operation {op}: images {images:?} not a permutation");
        assert!(images.iter().enumerate().all(|(a, &b)| atoms[a].1 == atoms[b].1), "operation {op} maps a centre onto another kind");
        if images.iter().enumerate().all(|(a, &b)| a == b) && (0..3).all(|i| (m[i][i] - 1.0).abs() < 1e-12) {
            identities += 1;
        }
        for a in 0..n {
            for b in 0..n {
                let d: [f64; 3] = std::array::from_fn(|k| atoms[a].2[k] - atoms[b].2[k]);
                let image: [f64; 3] = std::array::from_fn(|k| atoms[images[a]].2[k] - atoms[images[b]].2[k]);
                for i in 0..3 {
                    let moved: f64 = (0..3).map(|k| m[i][k] * d[k]).sum();
                    assert!((moved - image[i]).abs() < 1e-8, "operation {op}: centres {a}, {b} not carried onto {}, {}", images[a], images[b]);
                }
            }
        }
    }
    assert_eq!(identities, 1);
    assert!(ctx.get_symmetry_operation_matrix(h).is_err());
    assert!(ctx.get_symmetry_centre_images(0, &mut [0; 2]).is_err());
}
