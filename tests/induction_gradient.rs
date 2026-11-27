// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// compute_induction_gradient (dE_ind/dX at fixed density, from the converged induced dipoles) against central finite
// differences of the induction energy at fixed density: every gradient centre is moved with set_centre_positions, the
// shell pairs are rebuilt, and the dipoles are re-solved (compute_induction). Point and Gaussian nuclei, a cached and a
// recomputed static field, permanent external multipoles (fixed, but they polarize the sites), core AIMPs and embedding
// AIMPs. Plus the frozen-centre mask, and the refusal without dipoles.

mod common;

use aedelsten::{AedelstenContext, ContractionType, FiniteModel, PolarizationOptions, SolverOptions, StaticFieldPolicy};

fn write_temp(tag: &str, extension: &str, contents: &str) -> std::path::PathBuf {
    let path = common::temp_path(&format!("indgrad_{tag}.{extension}"));
    std::fs::write(&path, contents).unwrap();
    path
}

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 1.5 -0.9\n    H2 -0.2 -1.4 -1.1\nEnd of Basis\n";

const AIMP_SYSTEM: &str = "Basis set\n    Sc.TEST-CORE-AIMP\n    Sc1 0.0 0.0 0.0\nEnd of Basis\n\
    Basis set\n    H.x2c-SVPall\n    H1 0.15 -0.2 2.6\nEnd of Basis\n\
    Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    Na3 4.0 1.0 -1.5\n    Na4 -3.5 2.0 2.5\nEnd of Basis\n";

// Polarizable sites (fragment, position, anisotropic tensor xx xy xz yy yz zz), 4-6 Bohr from the QM atoms.
const SITES: &str = "3 Bohr -1 2 1\n1 4.5 0.5 0.3 2.0 0.1 0.0 2.2 0.0 1.8\n2 -3.8 2.9 1.5 1.5 -0.1 0.05 1.8 0.2 1.2\n3 0.4 -4.6 2.4 2.2 0.3 0.0 2.0 -0.1 1.6\n";
// Permanent multipoles (order 2) on other fragments.
const EXTERNAL: &str = "2 Bohr 2\n6.0 -2.0 1.0 0.4 0.1 -0.2 0.3 0.2 0.05 -0.1 0.3 0.0 -0.2\n-5.0 -3.0 -2.5 -0.3 0.2 0.1 -0.1 0.1 0.0 0.2 -0.3 0.1 0.2\n";

fn context(input: &str, finite: bool, external: bool, policy: StaticFieldPolicy) -> AedelstenContext {
    let path = write_temp("inp", "inp", input);
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    if finite {
        ctx.set_finite_nuclear_model(FiniteModel::VisscherDyall).unwrap();
    }
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    if external {
        let path = write_temp("external", "xfield", EXTERNAL);
        ctx.set_external_multipoles(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
    }
    let path = write_temp("sites", "xfield", SITES);
    ctx.set_polarizable_sites(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_polarization(PolarizationOptions {
        solver: SolverOptions { threshold: 1e-13, ..SolverOptions::default() },
        static_field: policy,
        ..PolarizationOptions::default()
    })
    .unwrap();
    ctx
}

fn density(n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[i + n * j] = 0.3 * (0.05 * (((i.min(j) * 7 + i.max(j) * 3 + 1) % 11) as f64) - 0.1) + if i == j { 0.4 } else { 0.0 };
        }
    }
    d
}

fn positions(ctx: &AedelstenContext) -> Vec<f64> {
    let mut geometry = vec![aedelsten::CentreGeometry { atomic_number: 0, position: [0.0; 3], kind: aedelsten::AtomKind::Real }; ctx.get_n_centres()];
    ctx.get_centre_geometry(&mut geometry).unwrap();
    geometry[..ctx.get_n_gradient_centres()].iter().flat_map(|c| c.position).collect()
}

fn induction_energy(ctx: &AedelstenContext, d: &[f64]) -> f64 {
    let n = ctx.get_n_cont();
    let mut v_ind = vec![0.0; n * n];
    ctx.compute_induction(d, &mut v_ind).unwrap().energy
}

fn check(label: &str, mut ctx: AedelstenContext) {
    let n_centres = ctx.get_n_gradient_centres();
    let d = density(ctx.get_n_cont());
    let base = positions(&ctx);
    induction_energy(&ctx, &d);
    let mut analytic = vec![f64::NAN; 3 * n_centres];
    ctx.compute_induction_gradient(&mut analytic, &d, None).unwrap();

    let h = 1e-4;
    let mut largest: f64 = 0.0;
    for q in 0..3 * n_centres {
        let mut at = |delta: f64| {
            let mut p = base.clone();
            p[q] += delta;
            ctx.set_centre_positions(&p).unwrap();
            ctx.set_shell_pairs(None).unwrap();
            induction_energy(&ctx, &d)
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        largest = largest.max(fd.abs());
        eprintln!("{label}: centre {} axis {}: analytic {:.10}, finite difference {fd:.10}", q / 3, q % 3, analytic[q]);
        assert!((analytic[q] - fd).abs() < 1e-7 + 1e-6 * fd.abs(), "{label}: centre {} axis {}: analytic {}, finite difference {fd}", q / 3, q % 3, analytic[q]);
    }
    assert!(largest > 1e-4, "{label}: the gradient is (nearly) zero, the check is vacuous");

    // Back at the reference geometry: frozen centres give zero columns and leave the others unchanged.
    ctx.set_centre_positions(&base).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    induction_energy(&ctx, &d);
    for subset in 0..(1usize << n_centres) {
        let mask: Vec<bool> = (0..n_centres).map(|x| subset & (1 << x) != 0).collect();
        let mut masked = vec![f64::NAN; 3 * n_centres];
        ctx.compute_induction_gradient(&mut masked, &d, Some(&mask)).unwrap();
        for q in 0..3 * n_centres {
            let expected = if mask[q / 3] { analytic[q] } else { 0.0 };
            assert!((masked[q] - expected).abs() < 1e-10, "{label}: mask {mask:?}, entry {q}: {} vs {expected}", masked[q]);
        }
    }
}

#[test]
fn water_point_nuclei_cached_static_field() {
    check("water, point nuclei, cached field", context(WATER, false, false, StaticFieldPolicy::Cache));
}

#[test]
fn water_finite_nuclei_permanent_multipoles_recomputed_static_field() {
    check("water, Gaussian nuclei, multipoles", context(WATER, true, true, StaticFieldPolicy::Recompute));
}

#[test]
fn core_and_embedding_aimps() {
    check("core and embedding AIMPs", context(AIMP_SYSTEM, false, true, StaticFieldPolicy::Cache));
}

#[test]
fn without_dipoles_the_gradient_is_refused() {
    let ctx = context(WATER, false, false, StaticFieldPolicy::Cache);
    let n = ctx.get_n_cont();
    let mut g = vec![0.0; 9];
    assert!(ctx.compute_induction_gradient(&mut g, &density(n), None).is_err());
}
