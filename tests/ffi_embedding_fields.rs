// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The C entry points of the electric field at sites (the count, input indices, field integrals, static and
// electronic fields, for every selection) and of the multipole convention, driven as a C, Fortran, Python or Julia
// caller would, against the Rust calls they wrap. Water with embedding AIMPs, external multipoles and polarizable
// sites, so that every selection has sites.

mod common;

use aedelsten::{AedelstenContext, ContractionType, MultipoleConvention, Site, SiteChunking, SiteSelection, StaticFieldSources};
use std::ffi::{CString, c_char, c_void};

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const c_char, contraction_type: i32, comm: i32) -> *mut c_void;
    fn free_aedelsten_context_c(ctx: *mut c_void);
    fn aedelsten_query_i32_c(key: *const c_char) -> i32;
    fn set_shell_pairs_c(ctx: *mut c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut c_void, size: i32, unit: i32) -> i32;
    fn set_external_multipoles_c(ctx: *mut c_void, filename: *const c_char) -> i32;
    fn set_polarizable_sites_c(ctx: *mut c_void, filename: *const c_char) -> i32;
    fn has_polarizable_sites_c(ctx: *const c_void) -> i32;
    fn has_polarization_c(ctx: *const c_void) -> i32;
    fn count_electric_field_sites_c(ctx: *const c_void, selection: i32, points: *const f64, n_points: i64) -> i64;
    fn get_input_indices_c(ctx: *const c_void, selection: i32, points: *const f64, n_points: i64, indices: *mut i64, n: i64) -> i32;
    fn compute_electric_field_integrals_c(ctx: *const c_void, selection: i32, points: *const f64, n_points: i64, first: i64, count: i64, output: *mut f64, len: i64) -> i32;
    fn compute_static_field_c(ctx: *const c_void, selection: i32, points: *const f64, n_points: i64, nuclei: i32, core: i32, emb: i32, ext: i32, field: *mut f64, len: i64) -> i32;
    fn compute_electronic_field_c(ctx: *const c_void, density: *const f64, selection: i32, points: *const f64, n_points: i64, sites: i64, budget: i64, unit: i32, field: *mut f64, len: i64) -> i32;
    fn set_multipole_convention_c(ctx: *mut c_void, convention: i32) -> i32;
    fn get_multipole_convention_c(ctx: *const c_void) -> i32;
}

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/embedding_aimp_sites.inp");
const EXTERNAL: &str = "2 Bohr 1\n5.0 5.0 5.0 1.0 0.1 0.2 0.3\n-3.0 2.0 1.0 -0.5 0.0 0.0 0.4\n";
const POLARIZABLE: &str = "3 Bohr 1 1 1\n1 6.0 6.0 6.0 0.0 0.0 0.0 0.0 1.5\n1 1.3 -0.7 2.1 0.0 0.0 0.0 0.0 2.5\n2 0.5 -1.5 1.0 0.0 0.0 0.0 0.0 3.5\n";
const POINTS: [f64; 6] = [1.0, 2.0, 3.0, -2.0, 0.5, 4.0];

fn query(key: &str) -> i32 {
    let key = CString::new(key).unwrap();
    unsafe { aedelsten_query_i32_c(key.as_ptr()) }
}

fn assert_close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    let scale = b.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
    let largest = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(largest <= 1e-13 * scale, "{what}: {largest:e}");
}

#[test]
fn the_fields_at_sites_are_the_rust_calls_answers() {
    let external = common::temp_path("ffi_embedding_fields_ext.xfield");
    let polarizable = common::temp_path("ffi_embedding_fields_pol.xfield");
    std::fs::write(&external, EXTERNAL).unwrap();
    std::fs::write(&polarizable, POLARIZABLE).unwrap();

    let mut rust = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    rust.set_shell_pairs(None).unwrap();
    rust.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    rust.set_external_multipoles(external.to_str().unwrap()).unwrap();
    rust.set_polarizable_sites(polarizable.to_str().unwrap()).unwrap();

    let inp = CString::new(INP).unwrap();
    let ctx = unsafe { set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!ctx.is_null());
    assert_eq!(unsafe { set_shell_pairs_c(ctx, -1.0) }, 0);
    assert_eq!(unsafe { set_scratch_c(ctx, 64, query("byte_unit_mib")) }, 0);
    let (ext_c, pol_c) = (CString::new(external.to_str().unwrap()).unwrap(), CString::new(polarizable.to_str().unwrap()).unwrap());
    assert_eq!(unsafe { has_polarizable_sites_c(ctx) }, 0);
    assert_eq!(unsafe { set_external_multipoles_c(ctx, ext_c.as_ptr()) }, 0);
    assert_eq!(unsafe { set_polarizable_sites_c(ctx, pol_c.as_ptr()) }, 0);
    std::fs::remove_file(&external).ok();
    std::fs::remove_file(&polarizable).ok();
    assert_eq!(unsafe { has_polarizable_sites_c(ctx) }, 1);
    assert_eq!(unsafe { has_polarization_c(ctx) }, i32::from(rust.has_polarization()));
    assert_eq!(unsafe { has_polarization_c(std::ptr::null()) }, -1);

    let n = rust.get_n_cont();
    let density: Vec<f64> = (0..n * n).map(|i| 0.02 / (1.0 + (i % n) as f64 + (i / n) as f64)).collect();
    let custom: Vec<Site> = POINTS.chunks_exact(3).map(|p| Site::new([p[0], p[1], p[2]])).collect();
    let selections = [
        ("site_selection_qm", SiteSelection::Qm, 3),
        ("site_selection_embedding_aimps", SiteSelection::EmbeddingAimps, 3),
        ("site_selection_external_multipoles", SiteSelection::ExternalMultipoles, 2),
        ("site_selection_polarizable", SiteSelection::Polarizable, 3),
        ("site_selection_custom", SiteSelection::Custom(&custom), 2),
    ];
    for (key, selection, n_expected) in selections {
        let s = query(key);
        let (points, n_points) = if key == "site_selection_custom" { (POINTS.as_ptr(), 2) } else { (std::ptr::null(), 0) };
        let n_sites = rust.count_electric_field_sites(selection).unwrap();
        assert_eq!(n_sites, n_expected, "{key}");
        assert_eq!(unsafe { count_electric_field_sites_c(ctx, s, points, n_points) }, n_sites as i64, "{key}");

        let expected: Vec<i64> = rust.get_input_indices(selection).unwrap().iter().map(|i| i.map_or(-1, |i| i as i64)).collect();
        let mut indices = vec![-7; n_sites];
        assert_eq!(unsafe { get_input_indices_c(ctx, s, points, n_points, indices.as_mut_ptr(), n_sites as i64) }, 0, "{key}");
        assert_eq!(indices, expected, "{key}");

        // The integrals of the last sites only: the range is honoured.
        let (first, count) = (1, n_sites - 1);
        let mut expected = vec![0.0; n * n * 3 * count];
        rust.compute_electric_field_integrals(selection, first..n_sites, &mut expected).unwrap();
        let mut integrals = vec![f64::NAN; expected.len()];
        assert_eq!(
            unsafe { compute_electric_field_integrals_c(ctx, s, points, n_points, first as i64, count as i64, integrals.as_mut_ptr(), integrals.len() as i64) },
            0,
            "{key}"
        );
        assert_close(&integrals, &expected, &format!("{key}: integrals"));

        // The static field of every source, and of the nuclei alone (the QM selection's sites are nuclei and
        // embedding-AIMP centres: there the field of the nuclei is singular, so that selection takes none).
        if key != "site_selection_qm" {
            for (nuclei, emb, ext) in [(1, 1, 1), (1, 0, 0)] {
                let sources = StaticFieldSources { nuclei: nuclei == 1, core_aimps: true, embedding_aimps: emb == 1, external_multipoles: ext == 1 };
                let mut expected = vec![0.0; 3 * n_sites];
                rust.compute_static_field(selection, sources, &mut expected).unwrap();
                let mut field = vec![f64::NAN; 3 * n_sites];
                assert_eq!(unsafe { compute_static_field_c(ctx, s, points, n_points, nuclei, 1, emb, ext, field.as_mut_ptr(), field.len() as i64) }, 0, "{key}");
                assert_close(&field, &expected, &format!("{key}: static field {nuclei}{emb}{ext}"));
            }
        }

        // The electronic field, chunked by sites and by the default budget.
        for (chunk_sites, chunking) in [(2, SiteChunking::Sites(2)), (0, SiteChunking::Budget(None))] {
            let mut expected = vec![0.0; 3 * n_sites];
            rust.compute_electronic_field(&density, selection, &chunking, &mut expected).unwrap();
            let mut field = vec![f64::NAN; 3 * n_sites];
            assert_eq!(
                unsafe { compute_electronic_field_c(ctx, density.as_ptr(), s, points, n_points, chunk_sites, -1, 0, field.as_mut_ptr(), field.len() as i64) },
                0,
                "{key}"
            );
            assert_close(&field, &expected, &format!("{key}: electronic field"));
        }
    }

    // Refusals: an unknown selection, custom without points, a short output.
    assert_eq!(unsafe { count_electric_field_sites_c(ctx, 99, std::ptr::null(), 0) }, -1);
    assert_eq!(unsafe { count_electric_field_sites_c(ctx, query("site_selection_custom"), std::ptr::null(), 2) }, -1);
    let mut short = vec![0.0; 2];
    assert_ne!(unsafe { compute_static_field_c(ctx, query("site_selection_polarizable"), std::ptr::null(), 0, 1, 1, 1, 1, short.as_mut_ptr(), 2) }, 0);

    // The multipole convention: the default, a change, and an unknown value refused.
    let (contact_free, exact) = (query("multipole_convention_contact_free"), query("multipole_convention_exact"));
    assert_eq!(rust.get_multipole_convention(), MultipoleConvention::ContactFree);
    assert_eq!(unsafe { get_multipole_convention_c(ctx) }, contact_free);
    assert_eq!(unsafe { set_multipole_convention_c(ctx, exact) }, 0);
    assert_eq!(unsafe { get_multipole_convention_c(ctx) }, exact);
    assert_ne!(unsafe { set_multipole_convention_c(ctx, 99) }, 0);
    assert_eq!(unsafe { get_multipole_convention_c(ctx) }, exact);

    unsafe { free_aedelsten_context_c(ctx) };
}
