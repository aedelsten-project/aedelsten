// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The polarization with its field integrals kept between calls (`PolarizationOptions::resident_field_budget`): every
// pass gives what the passes over the sites not kept give (the induction, its response, the per-state inductions, the
// effective external field); after a move and new shell pairs, what a context made at the new geometry gives; a budget
// too small keeps nothing and changes nothing; the memory breakdown counts the kept integrals. The C options select it.

mod common;

use aedelsten::{AedelstenContext, ContractionType, PolarizationOptions, SolverOptions};
use arrays::scratch::ByteUnit;

const WATER: [[f64; 3]; 3] = [[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]];
const SITES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/polarization_demo.xfield");

// Water in `oxygen` and `hydrogen` (a basis label, with any line that follows it in a basis block).
fn water(positions: &[[f64; 3]; 3], oxygen: &str, hydrogen: &str) -> String {
    let [o, h1, h2] = positions;
    format!(
        "Basis set\n    O.{oxygen}\n    O1 {} {} {}\nEnd of Basis\nBasis set\n    H.{hydrogen}\n    H1 {} {} {}\n    H2 {} {} {}\nEnd of Basis\n",
        o[0], o[1], o[2], h1[0], h1[1], h1[2], h2[0], h2[1], h2[2]
    )
}

fn context(positions: &[[f64; 3]; 3], tag: &str, resident: Option<(usize, ByteUnit)>) -> AedelstenContext {
    context_of(&water(positions, "x2c-SVPall", "x2c-SVPall"), ContractionType::General, tag, resident)
}

fn context_of(input: &str, contraction: ContractionType, tag: &str, resident: Option<(usize, ByteUnit)>) -> AedelstenContext {
    let path = common::temp_path(&format!("polarization_resident_{tag}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, ByteUnit::MiB)).unwrap();
    ctx.set_polarizable_sites(SITES).unwrap();
    ctx.set_polarization(PolarizationOptions {
        solver: SolverOptions { threshold: 1e-12, ..SolverOptions::default() },
        response_batch: 2,
        state_slots: 2,
        resident_field_budget: resident,
        ..PolarizationOptions::default()
    })
    .unwrap();
    ctx
}

fn densities(n: usize, k: usize) -> Vec<f64> {
    (0..n * n * k).map(|i| if i % (n * n) % (n + 1) == 0 { 0.3 } else { 0.02 * (((i * 7) % 13) as f64 - 6.0) / (1.0 + (i % n) as f64) }).collect()
}

/// Every pass, as one vector of numbers.
fn passes(ctx: &AedelstenContext) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let d = densities(n, 2);
    let mut out = Vec::new();
    let mut v = vec![0.0; n * n];
    let report = ctx.compute_induction(&d[..n * n], &mut v).unwrap();
    out.push(report.energy);
    out.extend(&v);
    let mut response = vec![0.0; 2 * n * n];
    ctx.compute_induction_response(&d, 2, &mut response).unwrap();
    out.extend(&response);
    let (mut states, mut energies) = (vec![0.0; 2 * n * n], vec![0.0; 2]);
    ctx.compute_induction_states(&d, 2, &mut states, &mut energies).unwrap();
    out.extend(&states);
    out.extend(&energies);
    let mut eef = vec![0.0; 3 * n * n];
    ctx.compute_effective_external_field(&mut eef).unwrap();
    out.extend(&eef);
    out
}

fn assert_close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let worst = a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(worst < 1e-12 * scale.max(1.0), "{what}: largest difference {worst:e} of {scale:e}");
}

#[test]
fn kept_integrals_give_what_the_chunked_passes_give() {
    let chunked = context(&WATER, "chunked", None);
    let kept = context(&WATER, "kept", Some((1, ByteUnit::GiB)));
    let (a, b) = (passes(&chunked), passes(&kept));
    assert_close(&a, &b, "kept against chunked");
    // A second round from the kept integrals, the same again.
    assert_close(&a, &passes(&kept), "kept, second round");
}

/// The sites not kept go through the density contracted in the Hermite basis and the dipoles' external-multipole
/// operator; the kept ones through their integrals. Both give the same passes for every contraction kind (a
/// segmented set to g functions, a primitive one, a general one with Cartesian d functions).
#[test]
fn the_evaluated_passes_give_what_the_kept_integrals_give_for_every_contraction_kind() {
    for (tag, input, contraction) in [
        ("segmented", water(&WATER, "x2c-SVPall", "x2c-SVPall"), ContractionType::Segmented),
        ("primitive", water(&WATER, "x2c-SVPall", "x2c-SVPall"), ContractionType::Primitive),
        ("cartesian", water(&WATER, "ANO-RCC...3s2p2d1f\n    Cartesian d", "ANO-RCC...2s1p"), ContractionType::General),
    ] {
        let evaluated = context_of(&input, contraction, &format!("{tag}_evaluated"), None);
        let kept = context_of(&input, contraction, &format!("{tag}_kept"), Some((1, ByteUnit::GiB)));
        assert_eq!(kept.get_polarization_storage().unwrap().sites_in_memory, 3, "{tag}: all kept");
        assert_eq!(evaluated.get_polarization_storage().unwrap().sites_in_memory, 0, "{tag}: none kept");
        assert_close(&passes(&kept), &passes(&evaluated), tag);
    }
}

/// The budget set after the polarization (`set_resident_field_budget`): all kept, two, then none, each giving the
/// same passes and the storage its counts.
#[test]
fn a_budget_set_later_keeps_what_it_holds() {
    let mut ctx = context(&WATER, "later", None);
    let expected = passes(&ctx);
    let n = ctx.get_n_cont();
    let per_site = 3 * n * (n + 1) / 2 * 8;
    let two = (2 * per_site + 2 * per_site / 3).div_ceil(1024);
    for (budget, kept) in [(Some((1, ByteUnit::GiB)), 3), (Some((two, ByteUnit::KiB)), 2), (None, 0)] {
        ctx.set_resident_field_budget(budget).unwrap();
        assert_eq!(ctx.get_polarization_storage().unwrap().sites_in_memory, kept, "{kept} to keep");
        assert_close(&expected, &passes(&ctx), &format!("{kept} kept after the polarization"));
        assert_close(&expected, &passes(&ctx), &format!("{kept} kept, second round"));
    }
}

#[test]
fn after_a_move_and_new_shell_pairs_the_kept_integrals_are_the_new_geometry_s() {
    let moved = [[0.05, -0.1, 0.25], [0.0, 1.5, -0.8], [0.1, -1.4, -0.95]];
    let fresh = context(&moved, "fresh", None);
    let mut kept = context(&WATER, "moving", Some((1, ByteUnit::GiB)));
    passes(&kept);
    kept.set_centre_positions(&moved.iter().flatten().copied().collect::<Vec<_>>()).unwrap();
    kept.set_shell_pairs(None).unwrap();
    assert_close(&passes(&fresh), &passes(&kept), "moved");
}

#[test]
fn a_budget_too_small_keeps_nothing() {
    let chunked = context(&WATER, "small_chunked", None);
    let small = context(&WATER, "small", Some((1, ByteUnit::KiB)));
    let kept = context(&WATER, "large", Some((1, ByteUnit::GiB)));
    assert_close(&passes(&chunked), &passes(&small), "small budget");
    let polarization = |ctx: &AedelstenContext| {
        let mut breakdown = [0usize; aedelsten::MEMORY_BREAKDOWN_FIELDS.len()];
        ctx.memory_breakdown(&mut breakdown).unwrap();
        let k = aedelsten::MEMORY_BREAKDOWN_FIELDS.iter().position(|f| f.to_str() == Ok("polarization")).unwrap();
        breakdown[k]
    };
    // The kept integrals: 3 sites x 3 planes x n (n + 1) / 2 doubles at least, which keeping nothing does not hold.
    let n = kept.get_n_cont();
    assert!(polarization(&kept) >= polarization(&small) + 9 * n * (n + 1) / 2 * 8);
}

#[test]
fn the_storage_reports_what_is_kept_and_what_keeping_it_needs() {
    let chunked = context(&WATER, "storage_chunked", None);
    let n = chunked.get_n_cont();
    // 3 sites x 3 planes of n (n + 1) / 2 doubles, and the scratch of 2 packed matrices.
    let per_site = 3 * n * (n + 1) / 2 * 8;
    let expected = aedelsten::PolarizationStorage { sites: 3, sites_in_memory: 0, bytes_per_site: per_site, bytes_to_keep: 3 * per_site + 2 * per_site / 3 };
    assert_eq!(chunked.get_polarization_storage().unwrap(), expected);
    let kib = expected.bytes_to_keep.div_ceil(1024);
    let all = context(&WATER, "storage_all", Some((kib, ByteUnit::KiB))).get_polarization_storage().unwrap();
    assert_eq!(all, aedelsten::PolarizationStorage { sites_in_memory: 3, ..expected });
    // Room for two sites: two kept, one evaluated.
    let two = (2 * per_site + 2 * per_site / 3).div_ceil(1024);
    let partial = context(&WATER, "storage_two", Some((two, ByteUnit::KiB))).get_polarization_storage().unwrap();
    assert_eq!(partial.sites_in_memory, 2);
}

/// What fits in memory, the rest evaluated: every pass as the passes over no kept site give, a second round from what
/// is kept, after a move the new geometry's.
#[test]
fn what_fits_is_kept_and_the_rest_evaluated() {
    let chunked = context(&WATER, "partial_chunked", None);
    let expected = passes(&chunked);
    let n = chunked.get_n_cont();
    let per_site = 3 * n * (n + 1) / 2 * 8;
    let scratch = 2 * per_site / 3;
    // Budgets in KiB; the sites they hold beside the scratch.
    for memory_kib in [1, 2, 3] {
        let memory_sites = ((memory_kib * 1024usize).saturating_sub(scratch) / per_site).min(3);
        let mut ctx = context(&WATER, &format!("partial_{memory_kib}"), Some((memory_kib, ByteUnit::KiB)));
        let storage = ctx.get_polarization_storage().unwrap();
        assert_eq!(storage.sites_in_memory, memory_sites, "{memory_kib} KiB");
        let what = format!("{memory_sites} kept");
        assert_close(&expected, &passes(&ctx), &what);
        assert_close(&expected, &passes(&ctx), &format!("{what}, second round"));
        if memory_sites > 0 && memory_sites < 3 {
            let moved = [[0.05, -0.1, 0.25], [0.0, 1.5, -0.8], [0.1, -1.4, -0.95]];
            ctx.set_centre_positions(&moved.iter().flatten().copied().collect::<Vec<_>>()).unwrap();
            ctx.set_shell_pairs(None).unwrap();
            assert_close(&passes(&context(&moved, "partial_fresh", None)), &passes(&ctx), &format!("{what}, moved"));
        }
    }
}

#[repr(C)]
struct COptions {
    damping: i32,
    thole_a: f64,
    threshold: f64,
    max_iterations: i32,
    static_field: i32,
    chunk_sites: i32,
    chunk_budget: i32,
    chunk_budget_unit: i32,
    response_batch: i32,
    state_slots: i32,
    resident_budget: i32,
    resident_budget_unit: i32,
}

unsafe extern "C" {
    fn aedelsten_polarization_options_default_c(options: *mut COptions);
    fn set_polarization_c(ctx: *mut std::ffi::c_void, options: *const COptions) -> i32;
    fn free_polarization_c(ctx: *mut std::ffi::c_void);
}

#[test]
fn the_c_options_keep_the_integrals() {
    let mut ctx = context(&WATER, "c", None);
    let expected = passes(&ctx);
    let handle = &mut ctx as *mut AedelstenContext as *mut std::ffi::c_void;
    let mut options = std::mem::MaybeUninit::<COptions>::uninit();
    let mut options = unsafe {
        aedelsten_polarization_options_default_c(options.as_mut_ptr());
        options.assume_init()
    };
    assert_eq!((options.resident_budget, options.resident_budget_unit), (0, 1));
    options.threshold = 1e-12;
    options.response_batch = 2;
    options.state_slots = 2;
    options.resident_budget = 1;
    options.resident_budget_unit = 0;
    unsafe { free_polarization_c(handle) };
    assert_eq!(unsafe { set_polarization_c(handle, &options) }, 0);
    let mut breakdown = [0usize; aedelsten::MEMORY_BREAKDOWN_FIELDS.len()];
    ctx.memory_breakdown(&mut breakdown).unwrap();
    assert_close(&expected, &passes(&ctx), "C options");
    options.resident_budget = -1;
    unsafe { free_polarization_c(handle) };
    assert_ne!(unsafe { set_polarization_c(handle, &options) }, 0);
}
