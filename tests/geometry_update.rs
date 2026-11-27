// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// `set_centre_positions` (moving the QM centres of an existing context) against a context built fresh at the new
// geometry: after the move and re-running the geometry setters (shell pairs, ERI data), every output must be the
// fresh context's -- one-electron matrices (incl. the core AIMP operators, whose projectors move with their atom),
// ERIs, one- and two-electron gradients, the nuclear repulsion energy, the centre geometry (orthogonality ghosts
// stay put) -- and a polarization state with a cached static field must give the fresh induction. Also: stale
// caches are refused (not silently used), and malformed positions are rejected.

mod common;

use aedelsten::{TwoElectronTerm, TwoElectronTermKind};
use aedelsten::{
    AedelstenContext, AtomKind, CentreGeometry, ContractionType, ERIAOSymmetry, ERIIndexOrder, ERIRangeOperator, OneElectronOperator,
    PolarizationOptions, SolverOptions,
};

fn write_input(tag: &str, contents: &str) -> std::path::PathBuf {
    let path = common::temp_path(&format!("geom_{tag}.inp"));
    std::fs::write(&path, contents).unwrap();
    path
}

fn context(input: &str, tag: &str) -> AedelstenContext {
    let path = write_input(tag, input);
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn water(p: &[[f64; 3]]) -> String {
    format!(
        "Basis set\n    O.x2c-SVPall\n    O1 {:.15} {:.15} {:.15}\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 {:.15} {:.15} {:.15}\n    H2 {:.15} {:.15} {:.15}\nEnd of Basis\n",
        p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], p[2][0], p[2][1], p[2][2]
    )
}

// A core AIMP (Sc, projectors on the moving atom), an H, and an embedding AIMP's orthogonality ghost (fixed, last).
fn core_aimp_with_ghost(p: &[[f64; 3]]) -> String {
    format!(
        "Basis set\n    Sc.TEST-CORE-AIMP\n    Sc1 {:.15} {:.15} {:.15}\nEnd of Basis\n\
         Basis set\n    H.x2c-SVPall\n    H1 {:.15} {:.15} {:.15}\nEnd of Basis\n\
         Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    orthogonality\n    Na2 2.5 -1.0 3.0\nEnd of Basis\n",
        p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2]
    )
}

fn flat(p: &[[f64; 3]]) -> Vec<f64> {
    p.iter().flatten().copied().collect()
}

fn density(n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[i + n * j] = 0.05 * (((i.min(j) * 7 + i.max(j) * 3 + 1) % 11) as f64) - 0.1;
        }
    }
    d
}

fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn geometry(ctx: &AedelstenContext) -> Vec<CentreGeometry> {
    let mut g = vec![CentreGeometry { atomic_number: 0, position: [0.0; 3], kind: AtomKind::Real }; ctx.get_n_centres()];
    ctx.get_centre_geometry(&mut g).unwrap();
    g
}

/// Every output the moved context shares with the fresh one: named values to compare.
fn outputs(ctx: &AedelstenContext, with_core_aimp: bool) -> Vec<(String, Vec<f64>)> {
    let n = ctx.get_n_cont();
    let n_centres = ctx.get_n_gradient_centres();
    let d = density(n);
    let mut out = Vec::new();
    out.push(("centres".to_string(), geometry(ctx).iter().flat_map(|c| c.position).collect()));
    out.push(("nuclear repulsion".to_string(), vec![ctx.compute_nuclear_repulsion_energy()]));
    type Make<'c> = Box<dyn Fn() -> OneElectronOperator<'c> + 'c>;
    let mut operators: Vec<(&str, Make<'_>)> = vec![
        ("overlap", Box::new(|| ctx.get_overlap_operator())),
        ("kinetic", Box::new(|| ctx.get_kinetic_operator())),
        ("nuclear attraction", Box::new(|| ctx.get_nuclear_attraction_operator())),
    ];
    if with_core_aimp {
        operators.push(("core AIMP coulomb", Box::new(|| ctx.get_core_aimp_coulomb_operator().unwrap())));
        operators.push(("core AIMP projection", Box::new(|| ctx.get_core_aimp_projection_operator().unwrap())));
    }
    for (name, make) in operators {
        let mut matrix = vec![0.0; n * n];
        ctx.compute_one_electron_integrals(&mut matrix, make(), false).unwrap();
        out.push((name.to_string(), matrix));
        let mut gradient = vec![0.0; 3 * n_centres];
        ctx.compute_one_electron_gradient(&mut gradient, &d, make(), None).unwrap();
        out.push((format!("{name} gradient"), gradient));
    }
    let mut eri = vec![0.0; aedelsten::compute_s8_size(n)];
    ctx.compute_eri_stored(&mut eri, ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor }, ERIRangeOperator::FullRange).unwrap();
    out.push(("ERI".to_string(), eri));
    let p = &d[..];
    let terms = [
        TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: p, right: p, coeff: 1.0 },
        TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: p, right: p, coeff: -0.5 },
    ];
    let mut gradient = vec![0.0; 3 * n_centres];
    ctx.compute_two_electron_gradient(&mut gradient, &terms, ERIRangeOperator::FullRange, None, None).unwrap();
    out.push(("two-electron gradient".to_string(), gradient));
    out
}

fn check_move(make_input: &dyn Fn(&[[f64; 3]]) -> String, start: &[[f64; 3]], moved: &[[f64; 3]], with_core_aimp: bool) {
    let mut ctx = context(&make_input(start), "start");
    ctx.set_eri_data(None).unwrap();
    let before = outputs(&ctx, with_core_aimp);
    let ghosts_before: Vec<CentreGeometry> = geometry(&ctx).into_iter().filter(|c| c.kind == AtomKind::OrthogonalityGhost).collect();

    ctx.set_centre_positions(&flat(moved)).unwrap();
    // The geometry-derived caches are stale now: refused with the reason, never silently reused.
    let n = ctx.get_n_cont();
    let mut matrix = vec![0.0; n * n];
    assert!(ctx.compute_one_electron_integrals(&mut matrix, ctx.get_overlap_operator(), false).is_err());
    let mut eri = vec![0.0; aedelsten::compute_s8_size(n)];
    let stale = ctx.compute_eri_stored(&mut eri, ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor }, ERIRangeOperator::FullRange);
    let message = format!("{}", stale.unwrap_err());
    assert!(message.contains("set_centre_positions") && message.contains("set_shell_pairs"), "unhelpful error: {message}");

    ctx.set_shell_pairs(None).unwrap();
    ctx.set_eri_data(None).unwrap();
    let mut fresh = context(&make_input(moved), "fresh");
    fresh.set_eri_data(None).unwrap();
    let got = outputs(&ctx, with_core_aimp);
    let expected = outputs(&fresh, with_core_aimp);
    for ((name, got), (_, expected)) in got.iter().zip(&expected) {
        let error = max_abs_difference(got, expected);
        assert!(error <= 1e-13, "{name}: moved context differs from a fresh one by {error}");
    }
    // The move changed things (the check is not vacuous) ...
    let changed = got.iter().zip(&before).filter(|((_, a), (_, b))| max_abs_difference(a, b) > 1e-6).count();
    assert!(changed >= got.len() - 1, "only {changed} of {} outputs changed with the geometry", got.len());
    // ... and the orthogonality ghosts did not move.
    let ghosts_after: Vec<CentreGeometry> = geometry(&ctx).into_iter().filter(|c| c.kind == AtomKind::OrthogonalityGhost).collect();
    assert_eq!(ghosts_before, ghosts_after);
}

#[test]
fn a_moved_context_equals_a_fresh_one() {
    let start = [[0.0, 0.1, 0.2], [0.3, 1.5, -0.9], [-0.2, -1.4, -1.1]];
    let moved = [[0.05, 0.12, 0.18], [0.4, 1.45, -0.8], [-0.25, -1.5, -1.0]];
    check_move(&water, &start, &moved, false);
}

#[test]
fn core_aimps_move_with_their_atom_and_orthogonality_ghosts_stay() {
    let start = [[0.0, 0.0, 0.0], [0.15, -0.2, 4.0]];
    let moved = [[0.1, -0.05, 0.2], [0.3, -0.1, 3.7]];
    check_move(&core_aimp_with_ghost, &start, &moved, true);
}

#[test]
fn malformed_positions_are_rejected_and_leave_the_context_untouched() {
    let start = [[0.0, 0.1, 0.2], [0.3, 1.5, -0.9], [-0.2, -1.4, -1.1]];
    let mut ctx = context(&water(&start), "malformed");
    let before = geometry(&ctx);
    assert!(ctx.set_centre_positions(&flat(&start)[..8]).is_err(), "too short");
    assert!(ctx.set_centre_positions(&[flat(&start), vec![0.0; 3]].concat()).is_err(), "too long");
    let mut nan = flat(&start);
    nan[4] = f64::NAN;
    assert!(ctx.set_centre_positions(&nan).is_err(), "not finite");
    assert_eq!(geometry(&ctx), before);
    let n = ctx.get_n_cont();
    let mut matrix = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut matrix, ctx.get_overlap_operator(), false).unwrap();
}

// Polarizable sites (fixed environment) around a moving water; the cached static field holds the field of the QM
// nuclei at the sites, so it has to follow the move.
const SITES: &str = "3 Bohr -1 2 1\n1 6.0 0.0 0.0 2.0 0.0 0.0 2.0 0.0 2.0\n2 -5.0 4.0 4.0 1.5 -0.1 0.05 1.8 0.2 1.2\n3 0.0 -6.0 3.0 2.2 0.3 0.0 2.0 -0.1 1.6\n";

fn polarized(p: &[[f64; 3]], tag: &str) -> AedelstenContext {
    let mut ctx = context(&water(p), tag);
    let sites = write_input(&format!("{tag}_sites"), SITES);
    ctx.set_polarizable_sites(sites.to_str().unwrap()).unwrap();
    std::fs::remove_file(&sites).ok();
    ctx.set_polarization(PolarizationOptions {
        solver: SolverOptions { threshold: 1e-12, ..SolverOptions::default() },
        ..PolarizationOptions::default()
    })
    .unwrap();
    ctx
}

#[test]
fn a_cached_static_field_follows_the_move() {
    let start = [[0.0, 0.1, 0.2], [0.3, 1.5, -0.9], [-0.2, -1.4, -1.1]];
    let moved = [[0.3, -0.2, 0.5], [0.6, 1.2, -0.5], [0.1, -1.7, -0.8]];
    let mut ctx = polarized(&start, "pol_start");
    let n = ctx.get_n_cont();
    let d = density(n);
    let mut v_ind = vec![0.0; n * n];
    let first = ctx.compute_induction(&d, &mut v_ind).unwrap();

    ctx.set_centre_positions(&flat(&moved)).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    let report = ctx.compute_induction(&d, &mut v_ind).unwrap();

    let fresh = polarized(&moved, "pol_fresh");
    let mut v_fresh = vec![0.0; n * n];
    let expected = fresh.compute_induction(&d, &mut v_fresh).unwrap();
    assert!((first.energy - expected.energy).abs() > 1e-6, "the move should change the induction energy");
    assert!((report.energy - expected.energy).abs() < 1e-10, "E_ind {} vs fresh {}", report.energy, expected.energy);
    let error = max_abs_difference(&v_ind, &v_fresh);
    assert!(error < 1e-10, "v_ind differs from a fresh context by {error}");
}
