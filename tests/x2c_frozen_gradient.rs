// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The X2C one-electron Hamiltonian with the decoupling held, and its derivative: transforms supplied at the geometry
// they were derived at give the derived Hamiltonian; the frozen gradient (`set_x2c_frozen_density`, then the kinetic
// part and one part per potential in the transformation, summed) against central differences of tr(D h_X2C) with the
// transforms of the reference geometry supplied at every displaced one. F, H, H in x2c-SVPall (F's
// d shell for the third derivatives of the p.Vp kernel), point and Gaussian nuclei, external point charges, a
// quadrupole site and embedding AIMPs' Coulomb part (bare charge and screen) inside the transformation, a frozen
// centre.

mod common;

use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};

const POSITIONS: [[f64; 3]; 3] = [[0.0, 0.05, 0.2217], [0.0, 1.4309, -0.8867], [0.1, -1.5309, -0.8067]];
const STEP: f64 = 1e-4;

/// What is in the transformation besides the nuclei.
#[derive(Clone, Copy, PartialEq)]
enum Environment {
    None,
    Charges,
    Quadrupole,
    /// Two O embedding AIMPs (TEST-EMB-AIMP, MgO's), their Coulomb part.
    Aimps,
}

/// The embedding AIMPs of `Environment::Aimps` (bohr).
const AIMPS: [[f64; 3]; 2] = [[-3.0, 0.5, 1.0], [0.5, 4.0, -2.5]];

fn context(name: &str, finite: bool, environment: Environment) -> AedelstenContext {
    let [f, h1, h2] = POSITIONS;
    let mut input = format!(
        "Basis set\n    F.x2c-SVPall\n    F1 {} {} {}\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 {} {} {}\n    H2 {} {} {}\nEnd of Basis\n",
        f[0], f[1], f[2], h1[0], h1[1], h1[2], h2[0], h2[1], h2[2]
    );
    if environment == Environment::Aimps {
        let [a1, a2] = AIMPS;
        input.push_str(&format!(
            "Basis set\n    O.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    A1 {} {} {}\n    A2 {} {} {}\nEnd of Basis\n",
            a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]
        ));
    }
    let path = common::temp_path(&format!("x2c_frozen_{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);
    if finite {
        ctx.set_finite_nuclear_model(aedelsten::FiniteModel::VisscherDyall).unwrap();
    }
    let sites = match environment {
        Environment::None | Environment::Aimps => None,
        Environment::Charges => Some("2 Bohr 0\n 3.0 0.5 -0.4 0.6\n -2.5 -2.0 1.5 -0.8\n"),
        Environment::Quadrupole => Some("1 Bohr 2\n 2.5 1.0 1.5 0.4 0.1 -0.2 0.3 0.5 0.2 -0.1 -0.3 0.4 -0.2\n"),
    };
    if let Some(text) = sites {
        let xfield = common::temp_path(&format!("x2c_frozen_{name}.xfield"));
        std::fs::write(&xfield, text).unwrap();
        ctx.set_external_multipoles(xfield.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&xfield);
    }
    ctx.set_scratch((16, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

/// The contracted X2C one-electron Hamiltonian at the context's geometry, the multipoles and the embedding AIMPs'
/// Coulomb part in the transformation if it has them; derived, or with `transforms` (U_L, U_S) held.
fn hamiltonian(ctx: &mut AedelstenContext, transforms: Option<&(Vec<f64>, Vec<f64>)>) -> Matrix<f64> {
    ctx.set_primitive_shell_pairs(None).unwrap();
    let n = ctx.get_n_prim_unique();
    let integrals = |ctx: &AedelstenContext, op: OneElectronOperator<'_>| {
        let mut m = Matrix::<f64>::new(n, n);
        ctx.compute_one_electron_integrals(m.get_data_mut(), op, true).unwrap();
        m
    };
    let s = integrals(ctx, OneElectronOperator::overlap());
    let t = integrals(ctx, OneElectronOperator::kinetic());
    let mut v = integrals(ctx, ctx.get_nuclear_attraction_operator());
    let mut pvp = Matrix::<f64>::new(n, n);
    ctx.compute_one_electron_picture_change_integrals(pvp.get_data_mut(), ctx.get_nuclear_attraction_operator(), false).unwrap();
    let environment = if ctx.has_external_multipoles() {
        Some(ctx.get_external_multipole_operator().unwrap())
    } else if ctx.has_embedding_aimp() {
        Some(ctx.get_embedding_aimp_coulomb_operator().unwrap())
    } else {
        None
    };
    if let Some(operator) = environment {
        let extra = integrals(ctx, operator.clone());
        let mut extra_pvp = Matrix::<f64>::new(n, n);
        ctx.compute_one_electron_picture_change_integrals(extra_pvp.get_data_mut(), operator, false).unwrap();
        for (a, b) in v.get_data_mut().iter_mut().zip(extra.get_data()) {
            *a += b;
        }
        for (a, b) in pvp.get_data_mut().iter_mut().zip(extra_pvp.get_data()) {
            *a += b;
        }
    }
    if let Some((u_large, u_small)) = transforms {
        ctx.set_picture_change_transforms(u_large, u_small, None).unwrap();
    }
    let mut h = Matrix::<f64>::new(n, n);
    ctx.construct_x2c_one_electron_hamiltonian(h.get_data_mut(), s.get_data(), t.get_data(), v.get_data(), pvp.get_data(), None).unwrap();
    let m = ctx.get_n_cont();
    let mut h_c = Matrix::<f64>::new(m, m);
    ctx.contract_one_electron_matrix(h.get_data(), h_c.get_data_mut()).unwrap();
    h_c
}

fn transforms(ctx: &AedelstenContext) -> (Vec<f64>, Vec<f64>) {
    let n = ctx.get_n_prim_unique();
    let (mut u_large, mut u_small) = (vec![0.0; n * n], vec![0.0; n * n]);
    ctx.get_picture_change_transforms(&mut u_large, &mut u_small).unwrap();
    (u_large, u_small)
}

/// A symmetric density: 0.1 on the diagonal and a small off-diagonal part.
fn density(n: usize) -> Matrix<f64> {
    let mut d = Matrix::<f64>::new(n, n);
    for j in 0..n {
        for i in 0..n {
            d[(i, j)] = if i == j { 0.1 } else { 0.02 / (1.0 + (i + j) as f64) };
        }
    }
    d
}

fn energy(h: &Matrix<f64>, d: &Matrix<f64>) -> f64 {
    h.get_data().iter().zip(d.get_data()).map(|(a, b)| a * b).sum()
}

/// The potentials `hamiltonian` puts into the transformation: the nuclei, and the multipoles or the embedding AIMPs'
/// Coulomb part if the context has them.
fn potentials(ctx: &AedelstenContext) -> Vec<OneElectronOperator<'_>> {
    let mut potentials = vec![ctx.get_nuclear_attraction_operator()];
    if ctx.has_external_multipoles() {
        potentials.push(ctx.get_external_multipole_operator().unwrap());
    } else if ctx.has_embedding_aimp() {
        potentials.push(ctx.get_embedding_aimp_coulomb_operator().unwrap());
    }
    potentials
}

/// The frozen gradient of tr(D h_X2C): the densities set, the kinetic part and each potential's part summed, the
/// densities freed.
fn frozen_gradient(ctx: &mut AedelstenContext, d: &Matrix<f64>, active: Option<&[bool]>) -> Vec<f64> {
    ctx.set_x2c_frozen_density(d.get_data()).unwrap();
    let n_centres = ctx.get_n_gradient_centres();
    let mut total = vec![0.0; 3 * n_centres];
    ctx.compute_x2c_frozen_kinetic_gradient(&mut total, active).unwrap();
    let mut part = vec![0.0; 3 * n_centres];
    for operator in potentials(ctx) {
        ctx.compute_x2c_frozen_potential_gradient(&mut part, operator, active).unwrap();
        for (t, p) in total.iter_mut().zip(&part) {
            *t += p;
        }
    }
    ctx.free_x2c_frozen_density();
    total
}

#[test]
fn supplied_transforms_give_the_derived_hamiltonian() {
    let mut derived = context("derived", false, Environment::Charges);
    let h = hamiltonian(&mut derived, None);
    let held = transforms(&derived);
    let mut supplied = context("supplied", false, Environment::Charges);
    let h_supplied = hamiltonian(&mut supplied, Some(&held));
    let scale = h.get_data().iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (a, b) in h.get_data().iter().zip(h_supplied.get_data()) {
        assert!((a - b).abs() < 1e-11 * scale, "{a} against {b}");
    }
}

/// The analytic frozen gradient against central differences, every coordinate of every centre, the second centre
/// frozen when `frozen`.
fn check(name: &str, finite: bool, environment: Environment, frozen: bool) {
    let mut ctx = context(name, finite, environment);
    let _ = hamiltonian(&mut ctx, None);
    let held = transforms(&ctx);
    let d = density(ctx.get_n_cont());
    // The atoms, then the embedding AIMPs if they are gradient centres (fixed).
    let n_centres = ctx.get_n_gradient_centres();
    let mut active = vec![false; n_centres];
    active[..3].copy_from_slice(&[true, !frozen, true]);
    let mut reference: Vec<f64> = POSITIONS.iter().flatten().copied().collect();
    reference.extend(AIMPS.iter().flatten().take(3 * (n_centres - 3)));
    let analytic = frozen_gradient(&mut ctx, &d, Some(&active));

    let mut worst = 0.0f64;
    for centre in 0..3 {
        for axis in 0..3 {
            if !active[centre] {
                assert_eq!(analytic[3 * centre + axis], 0.0, "{name}: the frozen centre's gradient");
                continue;
            }
            let displaced = |sign: f64| {
                let mut positions = reference.clone();
                positions[3 * centre + axis] += sign * STEP;
                let mut moved = context(&format!("{name}_{centre}{axis}"), finite, environment);
                moved.set_centre_positions(&positions).unwrap();
                energy(&hamiltonian(&mut moved, Some(&held)), &d)
            };
            let difference = (displaced(1.0) - displaced(-1.0)) / (2.0 * STEP);
            let value = analytic[3 * centre + axis];
            worst = worst.max((value - difference).abs());
            assert!((value - difference).abs() < 1e-7 * (1.0 + value.abs()), "{name}, centre {centre}, axis {axis}: {value} against {difference}");
        }
    }
    eprintln!("{name}: largest difference {worst:e}");
}

#[test]
fn the_frozen_gradient_matches_finite_differences_with_point_nuclei() {
    check("point", false, Environment::None, false);
}

#[test]
fn the_frozen_gradient_matches_finite_differences_with_gaussian_nuclei() {
    check("gaussian", true, Environment::None, false);
}

#[test]
fn the_frozen_gradient_matches_finite_differences_with_charges_in_the_transformation() {
    check("charges", false, Environment::Charges, true);
}

#[test]
fn the_frozen_gradient_matches_finite_differences_with_a_quadrupole_in_the_transformation() {
    check("quadrupole", false, Environment::Quadrupole, false);
}

#[test]
fn the_frozen_gradient_matches_finite_differences_with_embedding_aimps_in_the_transformation() {
    check("aimps", true, Environment::Aimps, true);
}

#[test]
fn a_frozen_gradient_cites_the_x2c_gradient() {
    let mut ctx = context("cite", false, Environment::None);
    let _ = hamiltonian(&mut ctx, None);
    let d = density(ctx.get_n_cont());
    let _ = frozen_gradient(&mut ctx, &d, None);
    let cited = instrumentation::citations();
    assert!(cited.iter().any(|r| r.reference.doi() == Some("10.1063/1.3624397")));
}

/// The densities: three n_prim x n_prim matrices in the memory breakdown; dropped by a new decoupling, by freeing it
/// and by moving the centres, after which the gradient calls are errors until they are set again; a component
/// operator refused.
#[test]
fn the_held_densities_are_reported_and_dropped_with_the_decoupling() {
    let mut ctx = context("held", false, Environment::None);
    let _ = hamiltonian(&mut ctx, None);
    let held = transforms(&ctx);
    let d = density(ctx.get_n_cont());
    let field = aedelsten::MEMORY_BREAKDOWN_FIELDS.iter().position(|name| name.to_str() == Ok("x2c_frozen_density")).unwrap();
    let held_bytes = |ctx: &AedelstenContext| {
        let mut breakdown = vec![0usize; aedelsten::MEMORY_BREAKDOWN_FIELDS.len()];
        ctx.memory_breakdown(&mut breakdown).unwrap();
        breakdown[field]
    };
    let n = ctx.get_n_prim_unique();
    let mut gradient = vec![0.0; 9];
    assert!(ctx.compute_x2c_frozen_kinetic_gradient(&mut gradient, None).is_err(), "before the densities are set");

    ctx.set_x2c_frozen_density(d.get_data()).unwrap();
    assert_eq!(held_bytes(&ctx), 3 * n * n * std::mem::size_of::<f64>());
    ctx.compute_x2c_frozen_kinetic_gradient(&mut gradient, None).unwrap();
    assert!(ctx.compute_x2c_frozen_potential_gradient(&mut gradient, OneElectronOperator::nabla(), None).is_err(), "a component operator");
    ctx.free_x2c_frozen_density();
    assert_eq!(held_bytes(&ctx), 0);

    let drops: [(&str, &dyn Fn(&mut AedelstenContext)); 3] = [
        ("free_picture_change", &|ctx| ctx.free_picture_change()),
        ("set_picture_change_transforms", &|ctx| ctx.set_picture_change_transforms(&held.0, &held.1, None).unwrap()),
        ("set_centre_positions", &|ctx| {
            let positions: Vec<f64> = POSITIONS.iter().flatten().copied().collect();
            ctx.set_centre_positions(&positions).unwrap()
        }),
    ];
    for (what, drop) in drops {
        let mut ctx = context(&format!("held_{what}"), false, Environment::None);
        let _ = hamiltonian(&mut ctx, None);
        ctx.set_x2c_frozen_density(d.get_data()).unwrap();
        drop(&mut ctx);
        assert_eq!(held_bytes(&ctx), 0, "{what}");
        let error = ctx.compute_x2c_frozen_kinetic_gradient(&mut gradient, None).unwrap_err().to_string();
        assert!(error.contains("set_x2c_frozen_density"), "{what}: {error}");
    }
}

/// `oneint_external_multipole`'s code.
const EXTERNAL_MULTIPOLE: i32 = 3;

unsafe extern "C" {
    fn set_picture_change_transforms_c(ctx: *mut std::ffi::c_void, u_large: *const f64, u_small: *const f64, len: i32, c_light: f64) -> i32;
    fn set_x2c_frozen_density_c(ctx: *mut std::ffi::c_void, density: *const f64, n: i32) -> i32;
    fn free_x2c_frozen_density_c(ctx: *mut std::ffi::c_void);
    fn compute_x2c_frozen_kinetic_gradient_c(ctx: *const std::ffi::c_void, gradient: *mut f64, n_centres: i32, active: *const i32) -> i32;
    fn compute_x2c_frozen_potential_gradient_c(ctx: *const std::ffi::c_void, gradient: *mut f64, n_centres: i32, operator: i32, active: *const i32) -> i32;
}

/// The C forms: each part bitwise as the context's; supplied transforms through C give the same; an unknown operator
/// code, a density of the wrong size and transforms of the wrong size refused.
#[test]
fn the_c_forms_give_what_the_context_gives() {
    let mut ctx = context("c_forms", false, Environment::Charges);
    let _ = hamiltonian(&mut ctx, None);
    let held = transforms(&ctx);
    let n = ctx.get_n_cont();
    let d = density(n);
    let mask = [true, false, true];
    ctx.set_x2c_frozen_density(d.get_data()).unwrap();
    let mut kinetic = vec![0.0; 9];
    ctx.compute_x2c_frozen_kinetic_gradient(&mut kinetic, Some(&mask)).unwrap();
    let mut multipoles = vec![0.0; 9];
    let operator = ctx.get_external_multipole_operator().unwrap();
    ctx.compute_x2c_frozen_potential_gradient(&mut multipoles, operator, Some(&mask)).unwrap();
    ctx.free_x2c_frozen_density();

    let handle = &mut ctx as *mut AedelstenContext as *mut std::ffi::c_void;
    let len = held.0.len() as i32;
    assert_eq!(unsafe { set_picture_change_transforms_c(handle, held.0.as_ptr(), held.1.as_ptr(), len, 0.0) }, 0);
    assert_eq!(unsafe { set_x2c_frozen_density_c(handle, d.get_data().as_ptr(), n as i32) }, 0);
    let active = [1i32, 0, 1];
    let mut gradient = vec![0.0; 9];
    assert_eq!(unsafe { compute_x2c_frozen_kinetic_gradient_c(handle, gradient.as_mut_ptr(), 3, active.as_ptr()) }, 0);
    assert_eq!(gradient, kinetic);
    assert_eq!(unsafe { compute_x2c_frozen_potential_gradient_c(handle, gradient.as_mut_ptr(), 3, EXTERNAL_MULTIPOLE, active.as_ptr()) }, 0);
    assert_eq!(gradient, multipoles);
    assert_eq!(unsafe { compute_x2c_frozen_potential_gradient_c(handle, gradient.as_mut_ptr(), 3, -7, std::ptr::null()) }, 1);
    unsafe { free_x2c_frozen_density_c(handle) };
    assert_eq!(unsafe { compute_x2c_frozen_kinetic_gradient_c(handle, gradient.as_mut_ptr(), 3, std::ptr::null()) }, 1);
    assert_eq!(unsafe { set_x2c_frozen_density_c(handle, d.get_data().as_ptr(), n as i32 - 1) }, 1);
    assert_eq!(unsafe { set_picture_change_transforms_c(handle, held.0.as_ptr(), held.1.as_ptr(), len - 1, 0.0) }, 1);
}
