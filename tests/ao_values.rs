// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The AO values (and their derivatives) of a list of shells through the context and its C forms: the shells as
// `get_shells` lists them, a list against the whole basis, `(3, n)` and `(4, n)` points, the extents, the errors; in
// the three contraction types. The kernel itself is tested against the per-point reference in `basis::ao_values`.

mod common;

use aedelsten::{AedelstenContext, ContractionType, ShellInfo};

// x2c-SVPall water-like: Cartesian d on F, s and p on the H: general and primitive contractions; water for the
// segmented mode.
const INPUT: &str = "Basis set\n    F.x2c-SVPall\n    Cartesian d\n    F1 0.0 0.0 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";
const SVP: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";

fn context(name: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("ao_values_{name}.inp"));
    let input = if matches!(contraction, ContractionType::Segmented) { SVP } else { INPUT };
    std::fs::write(&path, input).unwrap();
    let ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);
    ctx
}

const CONTRACTIONS: [(&str, ContractionType); 3] =
    [("general", ContractionType::General), ("segmented", ContractionType::Segmented), ("primitive", ContractionType::Primitive)];

// `(4, n)`: on the three centres and around them.
fn points() -> Vec<f64> {
    let mut points = vec![0.0, 0.0, 0.2, 1.0, 0.0, 1.4, -0.9, 1.0, 0.0, -1.4, -0.9, 1.0];
    for i in 0..60 {
        let t = i as f64;
        points.extend_from_slice(&[(0.9 * t).sin() * 2.5, (0.4 * t).cos() * 2.0, 0.07 * t - 2.1, 0.3]);
    }
    points
}

fn shells(ctx: &AedelstenContext) -> Vec<ShellInfo> {
    let mut shells = vec![ShellInfo::default(); ctx.get_n_shells()];
    ctx.get_shells(&mut shells).unwrap();
    shells
}

#[test]
fn the_shells_cover_the_aos_in_order() {
    for (name, contraction) in CONTRACTIONS {
        let ctx = context(name, contraction);
        let mut ids = vec![aedelsten::BasisFunctionId::default(); ctx.get_n_cont()];
        ctx.get_basis_function_ids(&mut ids).unwrap();
        let mut next = 0;
        for shell in shells(&ctx) {
            assert_eq!(shell.first_ao, next, "{name}");
            for id in &ids[shell.first_ao..shell.first_ao + shell.n_ao] {
                assert_eq!(id.get_centre(), shell.centre + 1, "{name}: centre");
                assert_eq!(id.get_l(), shell.l, "{name}: l");
            }
            next += shell.n_ao;
        }
        assert_eq!(next, ctx.get_n_cont(), "{name}");
    }
}

#[test]
fn a_shell_list_is_the_basis_gathered_in_its_order() {
    for (name, contraction) in CONTRACTIONS {
        let ctx = context(name, contraction);
        let shells = shells(&ctx);
        let points = points();
        let n_points = points.len() / 4;
        let n = ctx.get_n_cont();
        let mut all = vec![0.0; n * n_points];
        let mut all_gradients = vec![0.0; 3 * n * n_points];
        let every: Vec<usize> = (0..shells.len()).collect();
        ctx.compute_ao_values(&every, &points, 4, &mut all, Some(&mut all_gradients)).unwrap();

        let list: Vec<usize> = (0..shells.len()).rev().step_by(2).collect();
        let n_listed: usize = list.iter().map(|&s| shells[s].n_ao).sum();
        let mut values = vec![0.0; n_listed * n_points];
        let mut gradients = vec![0.0; 3 * n_listed * n_points];
        ctx.compute_ao_values(&list, &points, 4, &mut values, Some(&mut gradients)).unwrap();
        for j in 0..n_points {
            let mut row = 0;
            for &s in &list {
                for a in 0..shells[s].n_ao {
                    let mu = shells[s].first_ao + a;
                    assert_eq!(values[j * n_listed + row + a], all[j * n + mu], "{name}");
                    for d in 0..3 {
                        assert_eq!(gradients[(d * n_points + j) * n_listed + row + a], all_gradients[(d * n_points + j) * n + mu], "{name}");
                    }
                }
                row += shells[s].n_ao;
            }
        }

        // (3, n) points give the same values.
        let three: Vec<f64> = points.chunks_exact(4).flat_map(|p| p[..3].to_vec()).collect();
        let mut from_three = vec![0.0; n_listed * n_points];
        ctx.compute_ao_values(&list, &three, 3, &mut from_three, None).unwrap();
        assert_eq!(from_three, values, "{name}");
    }
}

#[test]
fn wrong_points_shells_and_buffers_are_errors() {
    let ctx = context("errors", ContractionType::General);
    let n_shells = ctx.get_n_shells();
    let points = points();
    let n_points = points.len() / 4;
    let n = ctx.get_n_cont();
    let every: Vec<usize> = (0..n_shells).collect();
    let mut values = vec![0.0; n * n_points];
    let two: Vec<f64> = points.chunks_exact(4).flat_map(|p| p[..2].to_vec()).collect();
    assert!(ctx.compute_ao_values(&every, &two, 2, &mut values, None).is_err());
    let five: Vec<f64> = points.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2], p[3], 0.0]).collect();
    assert!(ctx.compute_ao_values(&every, &five, 5, &mut values, None).is_err());
    assert!(ctx.compute_ao_values(&[n_shells], &points, 4, &mut values, None).is_err());
    assert!(ctx.compute_ao_values(&every, &points, 4, &mut values[1..], None).is_err());
    let mut short = vec![0.0; 3 * n * n_points - 1];
    assert!(ctx.compute_ao_values(&every, &points, 4, &mut values, Some(&mut short)).is_err());
    let (mut gradients, mut hessians) = (vec![0.0; 3 * n * n_points], vec![0.0; 6 * n * n_points]);
    assert!(ctx.compute_ao_hessians(&every, &points, 4, &mut values, &mut gradients, &mut hessians[1..]).is_err());
    assert!(ctx.compute_ao_hessians(&every, &points, 4, &mut values, &mut gradients[1..], &mut hessians).is_err());
    assert!(ctx.compute_ao_hessians(&every, &points, 4, &mut values, &mut gradients, &mut hessians).is_ok());
}

#[test]
fn beyond_its_extent_a_shell_is_below_the_threshold() {
    for (name, contraction) in CONTRACTIONS {
        let ctx = context(name, contraction);
        let shells = shells(&ctx);
        let mut extents = vec![0.0; shells.len()];
        ctx.compute_shell_extents(1e-12, &mut extents).unwrap();
        let mut geometry = vec![aedelsten::CentreGeometry { atomic_number: 0, position: [0.0; 3], kind: aedelsten::AtomKind::Real }; ctx.get_n_centres()];
        ctx.get_centre_geometry(&mut geometry).unwrap();
        for (s, shell) in shells.iter().enumerate() {
            let c = geometry[shell.centre].position;
            let r = extents[s];
            let points = [c[0] + r, c[1], c[2], c[0], c[1] - r, c[2], c[0] + 0.6 * r, c[1] + 0.8 * r, c[2]];
            let mut values = vec![0.0; 3 * shell.n_ao];
            let mut gradients = vec![0.0; 9 * shell.n_ao];
            ctx.compute_ao_values(&[s], &points, 3, &mut values, Some(&mut gradients)).unwrap();
            for v in values.iter().chain(&gradients) {
                assert!(v.abs() < 1e-12, "{name}: shell {s} at its extent {r}: {v}");
            }
        }
    }
}

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const std::ffi::c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn get_n_shells_c(ctx: *const std::ffi::c_void) -> i32;
    fn get_shells_c(ctx: *const std::ffi::c_void, output: *mut i32, len: i32) -> i32;
    fn compute_shell_extents_c(ctx: *const std::ffi::c_void, threshold: f64, output: *mut f64, n_shells: i32) -> i32;
    fn compute_ao_values_c(
        ctx: *const std::ffi::c_void,
        shells: *const i32,
        n_shells: i32,
        points: *const f64,
        point_rows: i32,
        n_points: i32,
        values: *mut f64,
        gradients: *mut f64,
        n_ao_listed: i32,
    ) -> i32;
    fn compute_ao_hessians_c(
        ctx: *const std::ffi::c_void,
        shells: *const i32,
        n_shells: i32,
        points: *const f64,
        point_rows: i32,
        n_points: i32,
        values: *mut f64,
        gradients: *mut f64,
        hessians: *mut f64,
        n_ao_listed: i32,
    ) -> i32;
}

#[test]
fn the_c_forms_give_what_the_context_gives() {
    let path = common::temp_path("ao_values_ffi.inp");
    std::fs::write(&path, INPUT).unwrap();
    let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    let c_ctx = unsafe { set_aedelsten_context_c(c_path.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL) };
    assert!(!c_ctx.is_null());
    let ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);

    let shells = shells(&ctx);
    let n = shells.len();
    assert_eq!(unsafe { get_n_shells_c(c_ctx) }, n as i32);
    let mut flat = vec![0i32; 4 * n];
    assert_eq!(unsafe { get_shells_c(c_ctx, flat.as_mut_ptr(), 4 * n as i32) }, 0);
    for (s, shell) in shells.iter().enumerate() {
        assert_eq!([flat[s], flat[n + s], flat[2 * n + s], flat[3 * n + s]], [shell.centre as i32, shell.l, shell.first_ao as i32, shell.n_ao as i32]);
    }
    assert_eq!(unsafe { get_shells_c(c_ctx, flat.as_mut_ptr(), 4 * n as i32 - 1) }, 1);

    let mut extents = vec![0.0; n];
    let mut c_extents = vec![0.0; n];
    ctx.compute_shell_extents(1e-10, &mut extents).unwrap();
    assert_eq!(unsafe { compute_shell_extents_c(c_ctx, 1e-10, c_extents.as_mut_ptr(), n as i32) }, 0);
    assert_eq!(extents, c_extents);

    let points = points();
    let n_points = points.len() / 4;
    let list = [2usize, 0, 3];
    let c_list = [2i32, 0, 3];
    let n_listed: usize = list.iter().map(|&s| shells[s].n_ao).sum();
    let (mut values, mut gradients) = (vec![0.0; n_listed * n_points], vec![0.0; 3 * n_listed * n_points]);
    let (mut c_values, mut c_gradients) = (vec![0.0; n_listed * n_points], vec![0.0; 3 * n_listed * n_points]);
    ctx.compute_ao_values(&list, &points, 4, &mut values, Some(&mut gradients)).unwrap();
    let status = unsafe {
        compute_ao_values_c(c_ctx, c_list.as_ptr(), 3, points.as_ptr(), 4, n_points as i32, c_values.as_mut_ptr(), c_gradients.as_mut_ptr(), n_listed as i32)
    };
    assert_eq!(status, 0);
    assert_eq!((&values, &gradients), (&c_values, &c_gradients));
    // No gradients (a null pointer), a negative shell index.
    let status = unsafe {
        compute_ao_values_c(c_ctx, c_list.as_ptr(), 3, points.as_ptr(), 4, n_points as i32, c_values.as_mut_ptr(), std::ptr::null_mut(), n_listed as i32)
    };
    assert_eq!(status, 0);
    assert_eq!(values, c_values);
    let negative = [-1i32];
    let status = unsafe {
        compute_ao_values_c(c_ctx, negative.as_ptr(), 1, points.as_ptr(), 4, n_points as i32, c_values.as_mut_ptr(), std::ptr::null_mut(), n_listed as i32)
    };
    assert_eq!(status, 1);

    // The second derivatives: the C form bitwise as the context's; values and gradients as compute_ao_values gives
    // them; an output overlapping another is refused.
    let mut hessians = vec![0.0; 6 * n_listed * n_points];
    let (mut h_values, mut h_gradients) = (vec![0.0; n_listed * n_points], vec![0.0; 3 * n_listed * n_points]);
    ctx.compute_ao_hessians(&list, &points, 4, &mut h_values, &mut h_gradients, &mut hessians).unwrap();
    assert_eq!((&h_values, &h_gradients), (&values, &gradients));
    let mut c_hessians = vec![0.0; 6 * n_listed * n_points];
    let status = unsafe {
        compute_ao_hessians_c(
            c_ctx,
            c_list.as_ptr(),
            3,
            points.as_ptr(),
            4,
            n_points as i32,
            c_values.as_mut_ptr(),
            c_gradients.as_mut_ptr(),
            c_hessians.as_mut_ptr(),
            n_listed as i32,
        )
    };
    assert_eq!(status, 0);
    assert_eq!((&values, &gradients, &hessians), (&c_values, &c_gradients, &c_hessians));
    let status = unsafe {
        compute_ao_hessians_c(
            c_ctx,
            c_list.as_ptr(),
            3,
            points.as_ptr(),
            4,
            n_points as i32,
            c_hessians.as_mut_ptr(),
            c_gradients.as_mut_ptr(),
            c_hessians.as_mut_ptr(),
            n_listed as i32,
        )
    };
    assert_eq!(status, 1);
    let status = unsafe {
        compute_ao_hessians_c(
            c_ctx,
            c_list.as_ptr(),
            3,
            points.as_ptr(),
            4,
            n_points as i32,
            c_values.as_mut_ptr(),
            c_gradients.as_mut_ptr(),
            std::ptr::null_mut(),
            n_listed as i32,
        )
    };
    assert_eq!(status, 1);

    unsafe { free_aedelsten_context_c(c_ctx) };
}

// Water in x2c-SVPall (F standing in for O: the same 3s2p1d shells, 24 AOs) at 68k points, values and gradients, every
// shell, as a DFT grid evaluates them. Run with `--release --ignored --nocapture`.
#[test]
#[ignore]
fn timing_of_water_at_68k_points() {
    let input = "Basis set\n    F.x2c-SVPall\n    F1 0.0 0.0 0.2217\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4309 -0.8867\n    H2 0.0 -1.4309 -0.8867\nEnd of Basis\n";
    let path = common::temp_path("ao_values_timing.inp");
    std::fs::write(&path, input).unwrap();
    let ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);
    let n = ctx.get_n_cont();
    let n_points = 68_000;
    // Points on spheres around the centres, radii from 0.01 to 20 bohr as a DFT grid's.
    let mut points = Vec::with_capacity(3 * n_points);
    let centres = [[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]];
    for i in 0..n_points {
        let c = centres[i % 3];
        let t = i as f64;
        let r = 0.01 + 20.0 * (t * 0.618).fract().powi(2);
        let (theta, phi) = ((t * 0.37).fract() * std::f64::consts::PI, (t * 0.73).fract() * 2.0 * std::f64::consts::PI);
        points.extend_from_slice(&[c[0] + r * theta.sin() * phi.cos(), c[1] + r * theta.sin() * phi.sin(), c[2] + r * theta.cos()]);
    }
    let every: Vec<usize> = (0..ctx.get_n_shells()).collect();
    let (mut values, mut gradients) = (vec![0.0; n * n_points], vec![0.0; 3 * n * n_points]);
    // In blocks of 256 points.
    let mut run = |with_gradients: bool| {
        let start = std::time::Instant::now();
        for (block, (v, g)) in points.chunks(3 * 256).zip(values.chunks_mut(256 * n).zip(gradients.chunks_mut(3 * 256 * n))) {
            let m = block.len() / 3;
            let g = if with_gradients { Some(&mut g[..3 * m * n]) } else { None };
            ctx.compute_ao_values(&every, block, 3, &mut v[..m * n], g).unwrap();
        }
        start.elapsed().as_secs_f64()
    };
    run(true);
    let (values_only, with_gradients) = (run(false), run(true));
    eprintln!("water/x2c-SVPall, {n} AOs, {n_points} points: values {values_only:.4} s, values and gradients {with_gradients:.4} s");
}
