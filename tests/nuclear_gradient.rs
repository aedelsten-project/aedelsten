// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The gradients of the classical energies (compute_nuclear_repulsion_gradient, compute_nuclear_aimp_embedding_gradient,
// compute_nuclear_multipole_interaction_gradient) against central finite differences of the energy functions
// themselves, moving one gradient centre at a time with set_centre_positions. Point and Gaussian (Visscher-Dyall)
// nuclei, core AIMPs (screened, with each other and with the nuclei), embedding AIMPs and an orthogonality ghost
// (fixed), and external point charges (order 0) and multipoles up to octupoles (order 3). Plus the frozen-centre mask.

mod common;

use aedelsten::{AedelstenContext, ContractionType, FiniteModel};

fn write_temp(tag: &str, extension: &str, contents: &str) -> std::path::PathBuf {
    let path = common::temp_path(&format!("nucgrad_{tag}.{extension}"));
    std::fs::write(&path, contents).unwrap();
    path
}

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 1.5 -0.9\n    H2 -0.2 -1.4 -1.1\nEnd of Basis\n";

// Two core AIMPs (Sc, the QM atoms carrying projectors) and an H; then an orthogonality ghost (fixed, outside the
// gradient matrix) and embedding AIMPs (fixed, no centre), in the order the input requires.
const AIMP_SYSTEM: &str = "Basis set\n    Sc.TEST-CORE-AIMP\n    Sc1 0.0 0.0 0.0\n    Sc2 0.4 -0.3 5.1\nEnd of Basis\n\
    Basis set\n    H.x2c-SVPall\n    H1 0.15 -0.2 2.6\nEnd of Basis\n\
    Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    orthogonality\n    Na5 -2.0 -3.0 1.0\nEnd of Basis\n\
    Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    Na3 4.0 1.0 -1.5\n    Na4 -3.5 2.0 2.5\nEnd of Basis\n";

const CHARGES: &str = "2 Bohr 0\n5.0 1.0 -2.0 -0.8\n-4.0 3.0 3.5 1.1\n";
const OCTUPOLES: &str = "2 Bohr 3\n5.0 1.0 -2.0 -0.8 0.3 -0.2 0.5 0.4 0.1 -0.3 0.2 0.05 -0.1 0.2 -0.1 0.3 0.05 -0.2 0.1 0.15 -0.05 0.1 0.2\n\
    -4.0 3.0 3.5 1.1 -0.4 0.2 0.1 0.3 -0.2 0.1 -0.1 0.2 0.3 -0.1 0.2 0.1 -0.3 0.05 0.2 -0.15 0.1 0.05 -0.2\n";

fn context(input: &str, finite: bool, xfield: Option<&str>) -> AedelstenContext {
    let path = write_temp("inp", "inp", input);
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    if finite {
        ctx.set_finite_nuclear_model(FiniteModel::VisscherDyall).unwrap();
    }
    if let Some(contents) = xfield {
        let path = write_temp("xfield", "xfield", contents);
        ctx.set_external_multipoles(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
    }
    ctx
}

fn positions(ctx: &AedelstenContext) -> Vec<f64> {
    let mut geometry = vec![aedelsten::CentreGeometry { atomic_number: 0, position: [0.0; 3], kind: aedelsten::AtomKind::Real }; ctx.get_n_centres()];
    ctx.get_centre_geometry(&mut geometry).unwrap();
    geometry[..ctx.get_n_gradient_centres()].iter().flat_map(|c| c.position).collect()
}

type Energy = fn(&AedelstenContext) -> f64;
type Gradient = fn(&AedelstenContext, &mut [f64], Option<&[bool]>) -> Result<(), aedelsten::AedelstenError>;

fn check(label: &str, ctx: &mut AedelstenContext, energy: Energy, gradient: Gradient) {
    let n = ctx.get_n_gradient_centres();
    let base = positions(ctx);
    let mut analytic = vec![f64::NAN; 3 * n];
    gradient(ctx, &mut analytic, None).unwrap();
    let h = 1e-5;
    let mut largest: f64 = 0.0;
    for q in 0..3 * n {
        let mut at = |delta: f64| {
            let mut p = base.clone();
            p[q] += delta;
            ctx.set_centre_positions(&p).unwrap();
            energy(ctx)
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        largest = largest.max(fd.abs());
        assert!((analytic[q] - fd).abs() < 1e-7 * (1.0 + fd.abs()), "{label}: centre {} axis {}: analytic {}, finite difference {fd}", q / 3, q % 3, analytic[q]);
    }
    ctx.set_centre_positions(&base).unwrap();
    assert!(largest > 1e-3, "{label}: the gradient is (nearly) zero, the check is vacuous");

    // Frozen centres: their columns are zero, the others unchanged, for every subset.
    for subset in 0..(1usize << n) {
        let mask: Vec<bool> = (0..n).map(|x| subset & (1 << x) != 0).collect();
        let mut masked = vec![f64::NAN; 3 * n];
        gradient(ctx, &mut masked, Some(&mask)).unwrap();
        for q in 0..3 * n {
            let expected = if mask[q / 3] { analytic[q] } else { 0.0 };
            assert!((masked[q] - expected).abs() < 1e-14, "{label}: mask {mask:?}, entry {q}");
        }
    }
    let mut wrong = vec![0.0; 3 * n + 3];
    assert!(gradient(ctx, &mut wrong, None).is_err(), "{label}: a mis-sized matrix must be refused");
}

fn repulsion(ctx: &AedelstenContext) -> f64 {
    ctx.compute_nuclear_repulsion_energy()
}
fn repulsion_gradient(ctx: &AedelstenContext, g: &mut [f64], active: Option<&[bool]>) -> Result<(), aedelsten::AedelstenError> {
    ctx.compute_nuclear_repulsion_gradient(g, active)
}
fn embedding(ctx: &AedelstenContext) -> f64 {
    ctx.compute_nuclear_aimp_embedding_energy()
}
fn embedding_gradient(ctx: &AedelstenContext, g: &mut [f64], active: Option<&[bool]>) -> Result<(), aedelsten::AedelstenError> {
    ctx.compute_nuclear_aimp_embedding_gradient(g, active)
}
fn multipole(ctx: &AedelstenContext) -> f64 {
    ctx.compute_nuclear_multipole_interaction_energy()
}
fn multipole_gradient(ctx: &AedelstenContext, g: &mut [f64], active: Option<&[bool]>) -> Result<(), aedelsten::AedelstenError> {
    ctx.compute_nuclear_multipole_interaction_gradient(g, active)
}

#[test]
fn water_point_and_finite_nuclei() {
    for finite in [false, true] {
        for xfield in [CHARGES, OCTUPOLES] {
            let mut ctx = context(WATER, finite, Some(xfield));
            check(&format!("water repulsion (finite {finite})"), &mut ctx, repulsion, repulsion_gradient);
            check(&format!("water multipoles (finite {finite})"), &mut ctx, multipole, multipole_gradient);
        }
    }
}

#[test]
fn core_aimps_embedding_aimps_and_multipoles() {
    for finite in [false, true] {
        for xfield in [CHARGES, OCTUPOLES] {
            let mut ctx = context(AIMP_SYSTEM, finite, Some(xfield));
            assert_eq!(ctx.get_n_gradient_centres(), 3, "Sc1, Sc2 and H1; the orthogonality ghost is not a gradient centre");
            check(&format!("AIMP repulsion (finite {finite})"), &mut ctx, repulsion, repulsion_gradient);
            check(&format!("AIMP embedding (finite {finite})"), &mut ctx, embedding, embedding_gradient);
            check(&format!("AIMP multipoles (finite {finite})"), &mut ctx, multipole, multipole_gradient);
        }
    }
}

#[test]
fn without_sources_the_embedding_gradient_is_zero_and_the_multipole_gradient_an_error() {
    let ctx = context(WATER, false, None);
    let mut g = vec![1.0; 9];
    ctx.compute_nuclear_aimp_embedding_gradient(&mut g, None).unwrap();
    assert!(g.iter().all(|&v| v == 0.0));
    assert!(ctx.compute_nuclear_multipole_interaction_gradient(&mut g, None).is_err());
}
