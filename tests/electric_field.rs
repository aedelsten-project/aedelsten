// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Verifies the electric-field integrals (oneint::kernels::ElectricFieldIntegrals)
//   < i | (r - C) / |r - C|^3 | j >
// against central finite differences of the already-verified point-charge path:
// a unit external point charge at C gives V(C) = -< i | 1/|r - C| | j > (nuclear-attraction
// sign convention), and the field integral is d/dC < 1/|r - C| > = -dV/dC. The two share
// only the shell-pair/Hermite-coefficient machinery, so agreement checks the R-tensor
// index shifts, the per-site/per-component plane ordering and the overall sign.
//
// Geometry: fock_builder_test.inp's QM region (x2c-SVPall + an inert AIMP centre),
// with the sites placed at generic positions, one of them close to a nucleus so the
// field is strongly non-uniform.

mod common;

use aedelsten::{AedelstenContext, ContractionType, Site, SiteChunking, SiteSelection, StaticFieldSources};
use aedelsten::arrays::scratch::ByteUnit;

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/fock_builder_test.inp");
const H_STEP: f64 = 1.0e-4;
const TOL: f64 = 1.0e-6;

const SITES: [[f64; 3]; 3] = [[6.0, 6.0, 6.0], [1.3, -0.7, 2.1], [0.5, -1.5, 1.0]];

fn build_context() -> (AedelstenContext, usize) {
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();
    (ctx, n)
}

// -<i|1/|r-C||j> for a unit point charge at `position`, through the external-multipole path.
fn point_charge_potential(ctx: &mut AedelstenContext, n: usize, position: [f64; 3], tag: &str) -> Vec<f64> {
    let path = common::temp_path(&format!("efield_fd_{}.xfield", tag));
    std::fs::write(&path, format!("1 Bohr 0\n{:.15} {:.15} {:.15} 1.0\n", position[0], position[1], position[2])).unwrap();
    ctx.set_external_multipoles(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();
    let mut output = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut output, ctx.get_external_multipole_operator().unwrap(), false).unwrap();
    ctx.free_external_multipoles();
    output
}

#[test]
fn electric_field_integrals_match_finite_difference_of_point_charge_potential() {
    let (mut ctx, n) = build_context();
    let sites: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();

    let mut field = vec![0.0; n * n * 3 * sites.len()];
    ctx.compute_electric_field_integrals(SiteSelection::Custom(&sites), 0..sites.len(), &mut field).unwrap();

    for (i_site, position) in SITES.iter().enumerate() {
        for component in 0..3 {
            let mut plus = *position;
            let mut minus = *position;
            plus[component] += H_STEP;
            minus[component] -= H_STEP;
            let v_plus = point_charge_potential(&mut ctx, n, plus, "plus");
            let v_minus = point_charge_potential(&mut ctx, n, minus, "minus");

            let plane_start = (3 * i_site + component) * n * n;
            let plane = &field[plane_start..plane_start + n * n];

            let mut worst = 0.0_f64;
            let mut signal = 0.0_f64;
            for ((f, vp), vm) in plane.iter().zip(&v_plus).zip(&v_minus) {
                let reference = -(vp - vm) / (2.0 * H_STEP);
                worst = worst.max((f - reference).abs());
                signal = signal.max(reference.abs());
            }
            println!("site {i_site} component {component}: diff={worst:.3e} signal={signal:.3e}");
            assert!(signal > 1e-6, "site {i_site} component {component}: reference field is trivially zero");
            assert!(
                worst < TOL,
                "site {i_site} component {component}: max |analytic - finite difference| = {worst:.3e} (signal {signal:.3e})"
            );
        }
    }
}

#[test]
fn electric_field_integral_planes_are_symmetric() {
    let (ctx, n) = build_context();
    let sites: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();
    let mut field = vec![0.0; n * n * 3 * sites.len()];
    ctx.compute_electric_field_integrals(SiteSelection::Custom(&sites), 0..sites.len(), &mut field).unwrap();
    for plane in field.chunks(n * n) {
        for i in 0..n {
            for j in 0..i {
                assert_eq!(plane[i + j * n], plane[j + i * n]);
            }
        }
    }
}

// ---- Site selection and the chunked density contraction -------------------------------

// Deterministic symmetric, non-trivial "density" (a Hilbert matrix); only the contraction
// is being tested here, not that it is a physical 1-RDM.
fn test_density(n: usize) -> Vec<f64> {
    (0..n * n).map(|idx| 1.0 / (1.0 + (idx % n) as f64 + (idx / n) as f64)).collect()
}

// Reference: full integral tensor for `selection`, contracted plane by plane with the density.
fn reference_field(ctx: &AedelstenContext, n: usize, density: &[f64], selection: SiteSelection) -> Vec<f64> {
    let count = ctx.count_electric_field_sites(selection).unwrap();
    let mut integrals = vec![0.0; n * n * 3 * count];
    ctx.compute_electric_field_integrals(selection, 0..count, &mut integrals).unwrap();
    integrals.chunks(n * n).map(|plane| plane.iter().zip(density).map(|(x, d)| x * d).sum()).collect()
}

fn assert_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-10 * (1.0 + y.abs()), "{x} vs {y}");
    }
}

#[test]
fn library_owned_selections_have_the_expected_sizes() {
    let (ctx, _n) = build_context();
    // The fixture has two QM nuclei (H1, H2) and one embedding AIMP (Na1), but no core AIMP,
    // so the core-AIMP part of `Qm` is not exercised here.
    assert!(!ctx.has_core_aimp());
    assert_eq!(ctx.count_electric_field_sites(SiteSelection::Qm).unwrap(), 2);
    assert_eq!(ctx.count_electric_field_sites(SiteSelection::EmbeddingAimps).unwrap(), 1);
    let custom: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();
    assert_eq!(ctx.count_electric_field_sites(SiteSelection::Custom(&custom)).unwrap(), 3);
    // Lists that were never set are errors, not empty selections.
    assert!(ctx.count_electric_field_sites(SiteSelection::ExternalMultipoles).is_err());
    assert!(ctx.count_electric_field_sites(SiteSelection::Polarizable).is_err());
}

#[test]
fn integrals_for_a_site_range_match_the_same_planes_of_the_full_tensor() {
    let (ctx, n) = build_context();
    let custom: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();
    let selection = SiteSelection::Custom(&custom);

    let mut full = vec![0.0; n * n * 3 * custom.len()];
    ctx.compute_electric_field_integrals(selection, 0..custom.len(), &mut full).unwrap();

    for range in [0..1, 1..3, 2..3, 0..3] {
        let mut part = vec![f64::NAN; n * n * 3 * range.len()];
        ctx.compute_electric_field_integrals(selection, range.clone(), &mut part).unwrap();
        // Not bit-identical: the BLAS-backed transformations block differently for different plane
        // counts, so roundoff differs. A wrong plane would differ at the 1e-3 level.
        let reference = &full[range.start * 3 * n * n..range.end * 3 * n * n];
        for (a, b) in part.iter().zip(reference) {
            assert!((a - b).abs() < 1e-13, "range {range:?}: {a} vs {b}");
        }
    }

    // Ranges outside the selection, and outputs of the wrong size, are errors.
    let mut output = vec![0.0; n * n * 3];
    assert!(ctx.compute_electric_field_integrals(selection, 2..4, &mut output).is_err());
    assert!(ctx.compute_electric_field_integrals(selection, 0..2, &mut output).is_err());
}

#[test]
fn electronic_field_matches_manual_contraction_for_every_chunking() {
    let (ctx, n) = build_context();
    let density = test_density(n);
    let custom: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();

    for selection in [SiteSelection::Qm, SiteSelection::EmbeddingAimps, SiteSelection::Custom(&custom)] {
        let expected = reference_field(&ctx, n, &density, selection);
        let bytes_for_two_sites = 2 * 3 * n * n * 8;
        for chunking in [
            SiteChunking::Sites(1),
            SiteChunking::Sites(2),
            SiteChunking::Sites(1000),
            SiteChunking::Budget(Some((bytes_for_two_sites.div_ceil(1024), ByteUnit::KiB))),
            SiteChunking::Budget(None),
        ] {
            let mut field = vec![f64::NAN; expected.len()];
            ctx.compute_electronic_field(&density, selection, &chunking, &mut field).unwrap();
            assert_close(&field, &expected);
        }
    }
}

#[test]
fn electronic_field_reports_bad_shapes_and_budgets() {
    let (ctx, n) = build_context();
    let density = test_density(n);
    let sites: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();

    let mut too_short = vec![0.0; 3 * sites.len() - 1];
    assert!(ctx.compute_electronic_field(&density, SiteSelection::Custom(&sites), &SiteChunking::Sites(1), &mut too_short).is_err());

    let mut field = vec![0.0; 3 * sites.len()];
    assert!(ctx.compute_electronic_field(&density[1..], SiteSelection::Custom(&sites), &SiteChunking::Sites(1), &mut field).is_err());

    // A budget that cannot hold a single site's integrals must be an error, not a silent overshoot.
    let tiny = SiteChunking::Budget(Some((1, ByteUnit::KiB)));
    assert!(ctx.compute_electronic_field(&density, SiteSelection::Custom(&sites), &tiny, &mut field).is_err());
}

// ---- Polarizable sites ----------------------------------------------------------------

fn write_temp_xfield(tag: &str, contents: &str) -> std::path::PathBuf {
    let path = common::temp_path(&format!("efield_pol_{}.xfield", tag));
    std::fs::write(&path, contents).unwrap();
    path
}

#[test]
fn polarizable_sites_are_a_separate_list_and_a_valid_selection() {
    let (mut ctx, n) = build_context();
    let density = test_density(n);

    // Not set yet: an error, not an empty list.
    assert!(!ctx.has_polarizable_sites());
    assert!(ctx.count_electric_field_sites(SiteSelection::Polarizable).is_err());

    // Three polarizable sites (charge + dipole + isotropic polarizability per line, with a
    // fragment column), and a *different* number of external multipoles: the lists are independent.
    let pol_path = write_temp_xfield("pol", "3 Bohr 1 1 1\n1 6.0 6.0 6.0 0.0 0.0 0.0 0.0 1.5\n1 1.3 -0.7 2.1 0.0 0.0 0.0 0.0 2.5\n2 0.5 -1.5 1.0 0.0 0.0 0.0 0.0 3.5\n");
    let ext_path = write_temp_xfield("ext", "1 Bohr 0\n8.0 8.0 8.0 1.0\n");
    ctx.set_polarizable_sites(pol_path.to_str().unwrap()).unwrap();
    ctx.set_external_multipoles(ext_path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&pol_path).ok();
    std::fs::remove_file(&ext_path).ok();

    assert!(ctx.has_polarizable_sites());
    assert!(ctx.set_polarizable_sites("unused").is_err(), "setting twice without freeing must fail");
    assert_eq!(ctx.count_electric_field_sites(SiteSelection::Polarizable).unwrap(), 3);
    assert_eq!(ctx.count_electric_field_sites(SiteSelection::ExternalMultipoles).unwrap(), 1);

    // Same positions as SITES, in the same order, so it must agree with the custom-site field.
    let custom: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();
    let mut from_polarizable = vec![0.0; 9];
    let mut from_custom = vec![0.0; 9];
    let chunking = SiteChunking::Sites(2);
    ctx.compute_electronic_field(&density, SiteSelection::Polarizable, &chunking, &mut from_polarizable).unwrap();
    ctx.compute_electronic_field(&density, SiteSelection::Custom(&custom), &chunking, &mut from_custom).unwrap();
    assert_close(&from_polarizable, &from_custom);

    ctx.free_polarizable_sites();
    assert!(!ctx.has_polarizable_sites());
    assert!(ctx.count_electric_field_sites(SiteSelection::Polarizable).is_err());
}

#[test]
fn polarizable_sites_reject_a_file_without_polarizabilities() {
    let (mut ctx, _n) = build_context();
    let path = write_temp_xfield("nopol", "1 Bohr 0\n1.0 1.0 1.0 1.0\n");
    assert!(ctx.set_polarizable_sites(path.to_str().unwrap()).is_err());
    std::fs::remove_file(&path).ok();
    assert!(!ctx.has_polarizable_sites());
}

#[test]
fn reading_one_list_never_sets_up_the_other() {
    let (mut ctx, _n) = build_context();
    // A file that carries BOTH multipoles and a polarizability on every line.
    let both = "2 Bohr 1 1\n1.0 2.0 3.0 0.5 0.1 0.2 0.3 7.5\n-1.0 0.0 4.0 -0.5 0.0 0.0 0.4 9.0\n";

    let path = write_temp_xfield("indep_a", both);
    ctx.set_external_multipoles(path.to_str().unwrap()).unwrap();
    assert!(ctx.has_external_multipoles());
    assert!(!ctx.has_polarizable_sites(), "setting external multipoles must not set polarizable sites");
    ctx.free_external_multipoles();

    ctx.set_polarizable_sites(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();
    assert!(ctx.has_polarizable_sites());
    assert!(!ctx.has_external_multipoles(), "setting polarizable sites must not set external multipoles");
}

// ---- A selection spanning two regions (nuclei, then core AIMPs) -----------------------------

// H (all-electron nucleus) plus a Sc centre carrying a core AIMP: `Qm` is then two segments of two
// different site types, which is what the segment offsets in the range/chunk logic must get right.
fn build_core_aimp_context() -> (AedelstenContext, usize) {
    let path = common::temp_path("efield_core_aimp.inp");
    std::fs::write(
        &path,
        "Basis set\n    H.x2c-SVPall\n    H 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    Sc.TEST-CORE-AIMP\n    Sc 3.0 1.0 -2.0\nEnd of Basis\n",
    )
    .unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();
    (ctx, n)
}

#[test]
fn qm_selection_spans_nuclei_and_core_aimps_in_order() {
    let (ctx, n) = build_core_aimp_context();
    assert!(ctx.has_core_aimp());
    let count = ctx.count_electric_field_sites(SiteSelection::Qm).unwrap();
    assert_eq!(count, 2, "one nucleus (H) and one core AIMP centre (Sc)");

    let mut qm = vec![0.0; n * n * 3 * count];
    ctx.compute_electric_field_integrals(SiteSelection::Qm, 0..count, &mut qm).unwrap();

    // The first site is the H nucleus at the origin (unit-independent), i.e. the nuclei segment
    // comes first.
    let origin = [Site::new([0.0, 0.0, 0.0])];
    let mut at_origin = vec![0.0; n * n * 3];
    ctx.compute_electric_field_integrals(SiteSelection::Custom(&origin), 0..1, &mut at_origin).unwrap();
    for (a, b) in qm[..3 * n * n].iter().zip(&at_origin) {
        assert!((a - b).abs() < 1e-13, "{a} vs {b}");
    }

    // The core-AIMP segment alone (a range starting exactly at the segment boundary) agrees with
    // the same planes of the full tensor.
    let mut second = vec![f64::NAN; n * n * 3];
    ctx.compute_electric_field_integrals(SiteSelection::Qm, 1..2, &mut second).unwrap();
    for (a, b) in second.iter().zip(&qm[3 * n * n..]) {
        assert!((a - b).abs() < 1e-13, "{a} vs {b}");
    }

    // A site on the core AIMP centre (Sc) is an error for the static field, like one on a nucleus.
    let on_core_aimp = [Site::new([3.0, 1.0, -2.0])];
    let core_only = StaticFieldSources { nuclei: false, core_aimps: true, embedding_aimps: false, external_multipoles: false };
    let mut static_field = vec![0.0; 3];
    assert!(ctx.compute_static_field(SiteSelection::Custom(&on_core_aimp), core_only, &mut static_field).is_err());
    assert!(ctx.compute_static_field(SiteSelection::Custom(&on_core_aimp), StaticFieldSources { core_aimps: false, ..core_only }, &mut static_field).is_ok());

    // The contracted field agrees with the manual contraction of the full tensor for chunks that
    // do and do not straddle the two segments.
    let density = test_density(n);
    let expected: Vec<f64> = qm.chunks(n * n).map(|plane| plane.iter().zip(&density).map(|(x, d)| x * d).sum()).collect();
    for chunking in [SiteChunking::Sites(1), SiteChunking::Sites(100)] {
        let mut field = vec![f64::NAN; 3 * count];
        ctx.compute_electronic_field(&density, SiteSelection::Qm, &chunking, &mut field).unwrap();
        assert_close(&field, &expected);
    }
}

// ---- The static (classical) field -----------------------------------------------------------

// Fixture geometry (fock_builder_test.inp, Bohr): two unit nuclei.
const H1: [f64; 3] = [0.37, -1.82, 0.94];
const H2: [f64; 3] = [2.15, 0.68, -1.41];

fn coulomb_field(charges: &[([f64; 3], f64)], at: [f64; 3]) -> [f64; 3] {
    let mut field = [0.0; 3];
    for (position, charge) in charges {
        let d = [at[0] - position[0], at[1] - position[1], at[2] - position[2]];
        let r3 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).powf(1.5);
        for axis in 0..3 {
            field[axis] += charge * d[axis] / r3;
        }
    }
    field
}

const NUCLEI_ONLY: StaticFieldSources =
    StaticFieldSources { nuclei: true, core_aimps: false, embedding_aimps: false, external_multipoles: false };
const EMBEDDING_ONLY: StaticFieldSources =
    StaticFieldSources { nuclei: false, core_aimps: false, embedding_aimps: true, external_multipoles: false };
const EXTERNAL_ONLY: StaticFieldSources =
    StaticFieldSources { nuclei: false, core_aimps: false, embedding_aimps: false, external_multipoles: true };

fn static_field(ctx: &AedelstenContext, sites: &[Site], sources: StaticFieldSources) -> Vec<f64> {
    let mut field = vec![f64::NAN; 3 * sites.len()];
    ctx.compute_static_field(SiteSelection::Custom(sites), sources, &mut field).unwrap();
    field
}

#[test]
fn nuclear_static_field_is_the_coulomb_field_of_the_nuclei() {
    let (ctx, _n) = build_context();
    let sites: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();
    let field = static_field(&ctx, &sites, NUCLEI_ONLY);
    for (i, &site) in SITES.iter().enumerate() {
        let expected = coulomb_field(&[(H1, 1.0), (H2, 1.0)], site);
        for axis in 0..3 {
            assert!((field[3 * i + axis] - expected[axis]).abs() < 1e-12, "site {i} axis {axis}");
        }
    }
}

#[test]
fn a_site_on_a_nucleus_is_an_error_but_one_on_an_embedding_aimp_is_fine() {
    let (ctx, _n) = build_context();
    // On a QM nucleus the field is singular and never silently skipped: an error, not NaN.
    let on_nucleus = [Site::new(H1), Site::new(SITES[0])];
    let mut field = vec![0.0; 6];
    assert!(ctx.compute_static_field(SiteSelection::Custom(&on_nucleus), NUCLEI_ONLY, &mut field).is_err());
    // With the nuclei switched off, the same sites are fine.
    ctx.compute_static_field(SiteSelection::Custom(&on_nucleus), EMBEDDING_ONLY, &mut field).unwrap();
    assert!(field.iter().all(|x| x.is_finite()));

    // A site on the (single) embedding AIMP -- a polarizable AIMP centre -- skips that AIMP's own field.
    let on_aimp = [Site::new([-1.63, 2.07, 3.28])];
    let mut own = vec![f64::NAN; 3];
    ctx.compute_static_field(SiteSelection::Custom(&on_aimp), EMBEDDING_ONLY, &mut own).unwrap();
    assert_eq!(own, vec![0.0; 3], "the only embedding AIMP is the site's own");
}

#[test]
fn the_full_static_field_is_the_sum_of_its_sources() {
    let (mut ctx, _n) = build_context();
    // An octupole source (nOrd = 3), so that the multipole path is the arbitrary-order one.
    let mut values = vec!["0.0"; 20];
    values[10 + 4] = "1.0"; // xyz
    values[0] = "0.5"; // and a charge
    let path = write_temp_xfield("static_sum", &format!("1 Bohr 3\n4.0 3.0 -5.0 {}\n", values.join(" ")));
    ctx.set_external_multipoles(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();

    let sites: Vec<Site> = SITES.iter().map(|&p| Site::new(p)).collect();
    let all = static_field(&ctx, &sites, StaticFieldSources::ALL);
    let nuclei = static_field(&ctx, &sites, NUCLEI_ONLY);
    let embedding = static_field(&ctx, &sites, EMBEDDING_ONLY);
    let external = static_field(&ctx, &sites, EXTERNAL_ONLY);
    assert!(embedding.iter().any(|x| x.abs() > 1e-8), "the embedding AIMP should act");
    assert!(external.iter().any(|x| x.abs() > 1e-8), "the external octupole should act");
    for i in 0..all.len() {
        assert!((all[i] - (nuclei[i] + embedding[i] + external[i])).abs() < 1e-12);
    }
    // Sources that are switched off contribute nothing.
    let none = static_field(
        &ctx,
        &sites,
        StaticFieldSources { nuclei: false, core_aimps: false, embedding_aimps: false, external_multipoles: false },
    );
    assert!(none.iter().all(|&x| x == 0.0));
}

#[test]
fn the_embedding_aimp_field_has_a_pure_coulomb_tail() {
    let (ctx, _n) = build_context();
    // Far from the AIMP its screening terms are gone and only the (unknown) charge Q is left, so
    // r^2 * |E| must be the same at two large distances along one direction.
    let na = [-1.63, 2.07, 3.28];
    let direction = [0.6, 0.0, 0.8];
    let tail = |distance: f64| -> f64 {
        let site = Site::new([na[0] + distance * direction[0], na[1] + distance * direction[1], na[2] + distance * direction[2]]);
        let field = static_field(&ctx, &[site], EMBEDDING_ONLY);
        distance * distance * (field[0] * field[0] + field[1] * field[1] + field[2] * field[2]).sqrt()
    };
    let (near, far) = (tail(30.0), tail(60.0));
    assert!(near > 1e-3, "the embedding AIMP should carry a charge");
    assert!((near - far).abs() < 1e-9 * near, "{near} vs {far}");
}

#[test]
fn a_polarizable_site_on_an_external_multipole_is_finite() {
    let (mut ctx, _n) = build_context();
    let ext = write_temp_xfield("static_coincident_ext", "2 Bohr 1\n5.0 5.0 5.0 1.0 0.1 0.2 0.3\n-3.0 2.0 1.0 -0.5 0.0 0.0 0.4\n");
    // The first polarizable site sits exactly on the first multipole.
    let pol = write_temp_xfield("static_coincident_pol", "2 Bohr -1 1\n5.0 5.0 5.0 2.0\n0.0 0.0 8.0 1.0\n");
    ctx.set_external_multipoles(ext.to_str().unwrap()).unwrap();
    ctx.set_polarizable_sites(pol.to_str().unwrap()).unwrap();
    std::fs::remove_file(&ext).ok();
    std::fs::remove_file(&pol).ok();

    let mut field = vec![f64::NAN; 6];
    ctx.compute_static_field(SiteSelection::Polarizable, StaticFieldSources::ALL, &mut field).unwrap();
    assert!(field.iter().all(|x| x.is_finite()), "{field:?}");
    let external_only = {
        let mut f = vec![f64::NAN; 6];
        ctx.compute_static_field(SiteSelection::Polarizable, EXTERNAL_ONLY, &mut f).unwrap();
        f
    };
    // Site 0 feels only the second multipole (the first is its own): a charge -0.5 plus the dipole
    // (0, 0, 0.4), whose field is (3 (p.d) d / r^5 - p / r^3) in closed form.
    let d: [f64; 3] = [5.0 + 3.0, 5.0 - 2.0, 5.0 - 1.0];
    let p = [0.0, 0.0, 0.4];
    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let p_dot_d = p[0] * d[0] + p[1] * d[1] + p[2] * d[2];
    let charge_part = coulomb_field(&[([-3.0, 2.0, 1.0], -0.5)], [5.0, 5.0, 5.0]);
    for axis in 0..3 {
        let expected = charge_part[axis] + 3.0 * p_dot_d * d[axis] / r2.powf(2.5) - p[axis] / r2.powf(1.5);
        assert!((external_only[axis] - expected).abs() < 1e-12, "axis {axis}: {} vs {expected}", external_only[axis]);
    }
}

#[test]
fn static_field_rejects_a_wrong_output_length() {
    let (ctx, _n) = build_context();
    let sites = [Site::new(SITES[0]), Site::new(SITES[1])];
    let mut too_short = vec![0.0; 5];
    assert!(ctx.compute_static_field(SiteSelection::Custom(&sites), StaticFieldSources::ALL, &mut too_short).is_err());
}

#[test]
fn fragments_exclude_sources_from_the_static_field_of_their_own_fragment() {
    let (mut ctx, _n) = build_context();
    // Three charge sources (nFrag = 2, so two leading integers): A in fragment 1, B in fragment 2, and
    // a junction charge C listed for fragments 1 and 3. Fragment 0 in the second column is "none".
    let a = ([5.0, 5.0, 5.0], 1.0);
    let b = ([-4.0, 2.0, 1.0], -0.5);
    let c = ([2.0, -6.0, 3.0], 0.75);
    let ext = write_temp_xfield(
        "frag_ext",
        &format!(
            "3 Bohr 0 0 2\n1 0 {} {} {} {}\n2 0 {} {} {} {}\n1 3 {} {} {} {}\n",
            a.0[0], a.0[1], a.0[2], a.1, b.0[0], b.0[1], b.0[2], b.1, c.0[0], c.0[1], c.0[2], c.1
        ),
    );
    // Polarizable sites in fragments 1, 2, 3 and one without fragments (nFrag = 1 needs a column, so use 9).
    let positions = [[0.0, 0.0, 8.0], [3.0, -3.0, 0.0], [-2.0, 4.0, -5.0], [1.0, 1.0, 1.0]];
    let pol = write_temp_xfield(
        "frag_pol",
        &format!(
            "4 Bohr -1 1 1\n1 {} {} {} 1.0\n2 {} {} {} 1.0\n3 {} {} {} 1.0\n9 {} {} {} 1.0\n",
            positions[0][0], positions[0][1], positions[0][2], positions[1][0], positions[1][1], positions[1][2],
            positions[2][0], positions[2][1], positions[2][2], positions[3][0], positions[3][1], positions[3][2]
        ),
    );
    ctx.set_external_multipoles(ext.to_str().unwrap()).unwrap();
    ctx.set_polarizable_sites(pol.to_str().unwrap()).unwrap();
    std::fs::remove_file(&ext).ok();
    std::fs::remove_file(&pol).ok();

    let mut field = vec![f64::NAN; 12];
    ctx.compute_static_field(SiteSelection::Polarizable, EXTERNAL_ONLY, &mut field).unwrap();

    // Expected: site in fragment 1 is excluded from A (frag 1) and C (frags 1, 3), feels B; fragment 2
    // is excluded from B, feels A and C; fragment 3 is excluded from C (junction), feels A and B; the
    // site with fragment 9 is excluded from nothing.
    let expected_sources: [&[([f64; 3], f64)]; 4] = [&[b], &[a, c], &[a, b], &[a, b, c]];
    for (i, sources) in expected_sources.iter().enumerate() {
        let expected = coulomb_field(sources, positions[i]);
        for axis in 0..3 {
            assert!((field[3 * i + axis] - expected[axis]).abs() < 1e-12, "site {i} axis {axis}: {} vs {}", field[3 * i + axis], expected[axis]);
        }
    }
}

// The combined polarizable-AIMP + PE model: the polarizable list holds a polarizable AIMP centre (fragment 0,
// slightly displaced from the AIMP, as if written by a different file) plus environment sites in
// fragments, while the multipole list holds only environment multipoles -- different counts, no
// site-by-site correspondence, and nothing else linking the two lists.
#[test]
fn polarizable_aimp_and_environment_sites_coexist_with_a_different_multipole_list() {
    let (mut ctx, _n) = build_context();
    let a = ([5.0, 5.0, 5.0], 1.0); // environment multipole, fragment 1
    let b = ([-4.0, 2.0, 1.0], -0.5); // environment multipole, fragment 2
    let ext = write_temp_xfield(
        "combined_ext",
        &format!("2 Bohr 0 0 1\n1 {} {} {} {}\n2 {} {} {} {}\n", a.0[0], a.0[1], a.0[2], a.1, b.0[0], b.0[1], b.0[2], b.1),
    );
    // Three polarizable sites: the AIMP centre (Na1 at (-1.63, 2.07, 3.28), written 1e-4 off), and one site in each
// fragment.
    let aimp_site = [-1.63 + 1e-4, 2.07, 3.28];
    let env1 = [0.0, 0.0, 8.0];
    let env2 = [3.0, -3.0, 0.0];
    let pol = write_temp_xfield(
        "combined_pol",
        &format!(
            "3 Bohr -1 1 1\n0 {} {} {} 2.0\n1 {} {} {} 1.0\n2 {} {} {} 1.0\n",
            aimp_site[0], aimp_site[1], aimp_site[2], env1[0], env1[1], env1[2], env2[0], env2[1], env2[2]
        ),
    );
    ctx.set_external_multipoles(ext.to_str().unwrap()).unwrap();
    ctx.set_polarizable_sites(pol.to_str().unwrap()).unwrap();
    std::fs::remove_file(&ext).ok();
    std::fs::remove_file(&pol).ok();
    assert_eq!(ctx.count_electric_field_sites(SiteSelection::ExternalMultipoles).unwrap(), 2);
    assert_eq!(ctx.count_electric_field_sites(SiteSelection::Polarizable).unwrap(), 3);

    let field_of = |sources: StaticFieldSources| {
        let mut field = vec![f64::NAN; 9];
        ctx.compute_static_field(SiteSelection::Polarizable, sources, &mut field).unwrap();
        field
    };

    // Multipoles: the AIMP site (fragment 0) is excluded from none of them; each environment site
    // is excluded from the multipole of its own fragment only.
    let external = field_of(EXTERNAL_ONLY);
    let expected: [(&[([f64; 3], f64)], [f64; 3]); 3] = [(&[a, b], aimp_site), (&[b], env1), (&[a], env2)];
    for (i, (sources, position)) in expected.iter().enumerate() {
        let expected_field = coulomb_field(sources, *position);
        for axis in 0..3 {
            assert!((external[3 * i + axis] - expected_field[axis]).abs() < 1e-12, "site {i} axis {axis}");
        }
    }

    // The AIMP: skipped at its own site despite the 1e-4 displacement, felt by the environment sites.
    let embedding = field_of(EMBEDDING_ONLY);
    assert_eq!(&embedding[..3], &[0.0, 0.0, 0.0], "the AIMP does not polarise its own site");
    assert!(embedding[3..].iter().all(|x| x.is_finite()) && embedding[3..6].iter().any(|x| x.abs() > 1e-8));

    // Nothing here coincides with a QM nucleus, and the total is the sum of the parts.
    let all = field_of(StaticFieldSources::ALL);
    let nuclei = field_of(NUCLEI_ONLY);
    for i in 0..9 {
        assert!((all[i] - (nuclei[i] + embedding[i] + external[i])).abs() < 1e-12);
    }
}

// One xfield file feeding both lists: an AIMP-like record with zero moments, a multipole-only record and
// a polarizable-only record each end up only in the list they belong to, and the input indices map
// the retained sites back to the file's records.
#[test]
fn a_single_file_feeds_both_lists_and_records_map_back_through_input_indices() {
    let (mut ctx, _n) = build_context();
    let contents = "5 Bohr 1 1 1\n\
        0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0\n\
        1 5.0 5.0 5.0 1.0 0.1 0.2 0.3 1.5\n\
        1 6.0 6.0 6.0 -0.5 0.0 0.0 0.0 0.0\n\
        2 7.0 7.0 7.0 0.0 0.0 0.0 0.0 3.0\n\
        2 8.0 8.0 8.0 0.0 0.0 0.0 0.0 0.0\n";
    let path = write_temp_xfield("one_file", contents);
    ctx.set_external_multipoles(path.to_str().unwrap()).unwrap();
    ctx.set_polarizable_sites(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(ctx.count_electric_field_sites(SiteSelection::ExternalMultipoles).unwrap(), 2);
    assert_eq!(ctx.count_electric_field_sites(SiteSelection::Polarizable).unwrap(), 3);
    assert_eq!(ctx.get_input_indices(SiteSelection::ExternalMultipoles).unwrap(), [Some(1), Some(2)]);
    assert_eq!(ctx.get_input_indices(SiteSelection::Polarizable).unwrap(), [Some(0), Some(1), Some(3)]);

    // Sites that do not come from a file have no input index.
    let custom = [Site::new([0.0; 3])];
    assert_eq!(ctx.get_input_indices(SiteSelection::Custom(&custom)).unwrap(), [None]);
    assert_eq!(ctx.get_input_indices(SiteSelection::Qm).unwrap(), [None, None]);

    // The static field at the polarizable sites sees exactly the two retained multipole sources
    // (fragment 1 records excluded at the fragment-1 site, its own position skipped).
    let mut field = vec![f64::NAN; 9];
    ctx.compute_static_field(SiteSelection::Polarizable, EXTERNAL_ONLY, &mut field).unwrap();
    assert!(field.iter().all(|x| x.is_finite()));
}

#[test]
fn unphysical_polarizabilities_are_rejected_when_the_list_is_set() {
    let (mut ctx, _n) = build_context();
    let bad = write_temp_xfield("bad_alpha", "1 Bohr -1 1\n0 0 0 -2.0\n");
    assert!(ctx.set_polarizable_sites(bad.to_str().unwrap()).is_err());
    std::fs::remove_file(&bad).ok();
    assert!(!ctx.has_polarizable_sites());
}
