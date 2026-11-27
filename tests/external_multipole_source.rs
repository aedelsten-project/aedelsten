// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Verifies compute_external_multipole_integrals / MultipoleSourceMode (the
// kernel backing ExternalMultipoles::Multipoles) against an
// independent reference: a real point-multipole source, built from a handful
// of ordinary point charges via a finite-difference stencil, computed
// through the already-verified PointMode/CoulombKernel<T> path. Agreement
// between the two, in the point-charge-separation -> 0 limit, checks both
// the R-tensor index shift and -- for the quadrupole case -- the off-
// diagonal multinomial scaling the kernel applies.
//
// Geometry: fock_builder_test.inp's QM region (x2c-SVPall + an inert
// AIMP centre nothing here touches), asymmetric and angular-momentum-rich
// enough that a wrong R-tensor index would show up as more than a uniform
// scale error. The multipole source itself sits at (6,6,6) Bohr, well away
// from every QM centre, so R(t,u,v) stays well-conditioned.

mod common;

use aedelsten::{AedelstenContext, ContractionType};

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/fock_builder_test.inp");

fn setup(xfield: &str) -> (AedelstenContext, usize) {
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_external_multipoles(xfield).unwrap();
    let n = ctx.get_n_cont();
    (ctx, n)
}

fn compute_external_multipole(xfield: &str) -> (Vec<f64>, usize) {
    let (ctx, n) = setup(xfield);
    let mut output = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut output, ctx.get_external_multipole_operator().unwrap(), false).unwrap();
    (output, n)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0_f64, |worst, (x, y)| worst.max((x - y).abs()))
}

#[test]
fn point_dipole_matches_a_finite_difference_pair_of_point_charges() {
    const DIPOLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/general_multipole_dipole_e4.xfield");
    const FD_E4: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/general_multipole_dipole_fd_e4.xfield");
    const FD_E5: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/general_multipole_dipole_fd_e5.xfield");

    let (dipole, n) = compute_external_multipole(DIPOLE);
    assert!(dipole.iter().any(|&x| x.abs() > 1e-8), "dipole integral is trivially all-zero");

    let (fd_e4, n4) = compute_external_multipole(FD_E4);
    let (fd_e5, n5) = compute_external_multipole(FD_E5);
    assert_eq!(n, n4);
    assert_eq!(n, n5);

    let diff_e4 = max_diff(&dipole, &fd_e4);
    let diff_e5 = max_diff(&dipole, &fd_e5);
    let signal = dipole.iter().fold(0.0_f64, |worst, &x| worst.max(x.abs()));
    println!("dipole: diff(1e-4)={diff_e4:.3e} diff(1e-5)={diff_e5:.3e} signal={signal:.3e}");

    // A wrong sign or a wrong overall scale shows up as an O(1) fraction of
    // the signal itself, not a small residual -- both epsilons must agree
    // with the analytic dipole kernel to a small fraction of the signal.
    // (Not asserting diff shrinks monotonically from 1e-4 to 1e-5: at that
    // point separation, +-Q with Q ~ 1/epsilon starts catastrophically
    // cancelling in the underlying integral evaluation itself, so floating-
    // point rounding -- not finite-difference truncation -- dominates and
    // can go either way; both are already ~1e-10 smaller than the signal,
    // far below anything a sign or scale bug would produce.)
    assert!(
        diff_e4 < 1.0e-2 * signal,
        "epsilon=1e-4 finite-difference disagreement too large relative to the signal: diff={diff_e4:.3e}, signal={signal:.3e}"
    );
    assert!(
        diff_e5 < 1.0e-2 * signal,
        "epsilon=1e-5 finite-difference disagreement too large relative to the signal: diff={diff_e5:.3e}, signal={signal:.3e}"
    );
}

#[test]
fn point_quadrupole_off_diagonal_matches_a_finite_difference_charge_stencil() {
    const QUAD_XZ: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/general_multipole_quad_xz.xfield");
    const QUAD_XZ_FD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/general_multipole_quad_xz_fd.xfield");

    let (quad, _n) = compute_external_multipole(QUAD_XZ);
    assert!(quad.iter().any(|&x| x.abs() > 1e-8), "quadrupole integral is trivially all-zero");

    let (fd, _n2) = compute_external_multipole(QUAD_XZ_FD);
    let diff = max_diff(&quad, &fd);
    let signal = quad.iter().fold(0.0_f64, |worst, &x| worst.max(x.abs()));
    println!("off-diagonal (xz): diff={diff:.6e} signal={signal:.6e} ratio={:.6}", diff / signal);

    // Single epsilon (1e-3) here: the point is to catch a systematic
    // sign/scale error (which would show up as an O(1) fraction of the
    // signal, e.g. exactly 2x from a missing/doubled multinomial factor),
    // not to establish the finite-difference stencil's own convergence order.
    assert!(
        diff < 5.0e-2 * signal,
        "quadrupole (off-diagonal) finite-difference disagreement too large relative to the signal: \
         diff={diff:.3e}, signal={signal:.3e} -- check MultipoleSourceMode's off-diagonal scaling factor"
    );
}

// NOTE: the AO operator of an environment multipole is CONTACT-FREE (harmonic part of each order), see below. The
// stencil tests in this file therefore only use components whose trace parts vanish (dipole, xz quadrupole, xyz
// octupole), where the raw point-charge stencil is exactly comparable; components with trace are covered by the
// harmonic-stencil test at the end of the file.

// ---- Octupoles (order 3) --------------------------------------------------------------
//
// Same idea as above, one order up: an octupole component M_abc = Sum q a b c (about the source
// position) is realised by a finite-difference charge stencil whose *only* nonzero moment below
// order 5 is that one, and the AO integrals of the single octupole source (nOrd = 3 xfield) are
// compared against the integrals of the stencil's point charges. Each case checks a different
// part of the order-3 machinery: the multinomial factor 1/(lx! ly! lz!) (1/6 for zzz, 1/2 for xxy,
// 1 for xyz) and the file's component order (x descending, then y descending: xxx xxy xxz xyy xyz
// xzz yyy yyz yzz zzz).

fn octupole_source_file(component_in_file_order: usize) -> String {
    // x y z, then the 20 entries of orders 0..=3, everything zero except the requested octupole
    // component (component 0..10 of the last block) which is 1.
    let mut values = vec!["0.0"; 20];
    values[10 + component_in_file_order] = "1.0";
    format!("1 Bohr 3\n6.0 6.0 6.0 {}\n", values.join(" "))
}

fn stencil_file(charges: &[([f64; 3], f64)]) -> String {
    let mut contents = format!("{} Bohr 0\n", charges.len());
    for (position, charge) in charges {
        contents.push_str(&format!("{:.15} {:.15} {:.15} {:.15}\n", position[0], position[1], position[2], charge));
    }
    contents
}

fn compute_from_temp_xfield(tag: &str, contents: &str) -> Vec<f64> {
    let path = common::temp_path(&format!("octupole_{}.xfield", tag));
    std::fs::write(&path, contents).unwrap();
    let (output, _n) = compute_external_multipole(path.to_str().unwrap());
    std::fs::remove_file(&path).ok();
    output
}

fn assert_octupole_matches_stencil(tag: &str, component_in_file_order: usize, charges: &[([f64; 3], f64)]) {
    let octupole = compute_from_temp_xfield(&format!("{tag}_oct"), &octupole_source_file(component_in_file_order));
    let stencil = compute_from_temp_xfield(&format!("{tag}_fd"), &stencil_file(charges));
    let signal = octupole.iter().fold(0.0_f64, |worst, &x| worst.max(x.abs()));
    let diff = max_diff(&octupole, &stencil);
    println!("octupole {tag}: diff={diff:.3e} signal={signal:.3e} ratio={:.3e}", diff / signal);
    assert!(signal > 1e-8, "octupole {tag}: integral is trivially all-zero");
    assert!(
        diff < 1.0e-3 * signal,
        "octupole {tag}: disagreement with the point-charge stencil too large: diff={diff:.3e}, signal={signal:.3e}"
    );
}

const H: f64 = 0.02;
const C: [f64; 3] = [6.0, 6.0, 6.0];

#[test]
fn point_octupole_xyz_matches_a_finite_difference_charge_stencil() {
    // Eight charges Q sx sy sz at the corners (sx, sy, sz) h/2: M_xyz = Q h^3, all lower moments vanish.
    let q = 1.0 / (H * H * H);
    let mut charges = Vec::new();
    for &sx in &[-1.0, 1.0] {
        for &sy in &[-1.0, 1.0] {
            for &sz in &[-1.0, 1.0] {
                charges.push(([C[0] + sx * H / 2.0, C[1] + sy * H / 2.0, C[2] + sz * H / 2.0], sx * sy * sz * q));
            }
        }
    }
    assert_octupole_matches_stencil("xyz", 4, &charges);
}

// ---- Contact-free convention and all components ---------------------------------------------------------

// The AO operator of an environment multipole is contact-free: it only sees the harmonic (traceless) part of each
// order (sources::multipole explains why, and tests/pe_validation.rs shows it is what PElib/Dalton compute). Two
// crisp consequences, checked here on the real fixture geometry.

fn multipole_operator(xfield_contents: &str, tag: &str) -> Vec<f64> {
    compute_from_temp_xfield(tag, xfield_contents)
}

#[test]
fn pure_trace_moments_do_not_interact_with_the_electrons() {
    // A purely isotropic quadrupole (M_xx = M_yy = M_zz = 1, i.e. Q = r^2 delta) and a pure-trace octupole (T_abc =
    // delta_ab v_c + delta_ac v_b + delta_bc v_a with v = (1, 0, 0): M_xxx = 3, M_xyy = M_xzz = 1) are entirely trace,
    // so their harmonic part is zero and so is their operator, even close to the molecule. (The file order of the
    // order-2 block is xx xy xz yy yz zz, of the order-3 block xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz.)
    let quadrupole = "1 Bohr 2\n1.0 0.5 0.5 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0\n";
    let octupole = "1 Bohr 3\n1.0 0.5 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0\n";
    for (tag, contents) in [("trace_q", quadrupole), ("trace_o", octupole)] {
        let operator = multipole_operator(contents, tag);
        let largest = operator.iter().fold(0.0_f64, |w, x| w.max(x.abs()));
        assert!(largest < 1e-12, "{tag}: a pure-trace multipole must not interact with the electrons (largest element {largest:e})");
    }
    // ... while a traceless quadrupole of the same size does.
    let traceless = "1 Bohr 2\n1.0 0.5 0.5 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0\n";
    let operator = multipole_operator(traceless, "traceless_q");
    assert!(operator.iter().fold(0.0_f64, |w, x| w.max(x.abs())) > 1e-3);
}

#[test]
fn raw_and_traceless_input_give_the_same_operator() {
    // Adding trace to a quadrupole (M_xx, M_yy, M_zz all shifted by 0.7) changes nothing.
    let base = "1 Bohr 2\n1.0 0.5 0.5 0.0 0.0 0.0 0.0 0.9 0.2 0.1 -0.4 0.3 -0.5\n";
    let shifted = "1 Bohr 2\n1.0 0.5 0.5 0.0 0.0 0.0 0.0 1.6 0.2 0.1 0.3 0.3 0.2\n";
    let a = multipole_operator(base, "base_q");
    let b = multipole_operator(shifted, "shifted_q");
    assert!(max_diff(&a, &b) < 1e-12 * (1.0 + a.iter().fold(0.0_f64, |w, x| w.max(x.abs()))));
}

// Every one of the 20 components (orders 0..3) against the point-charge stencil realising exactly that
// component, far from the molecule (where the electron density, hence any contact term, is negligible). The
// stencil for (a, b, c) is the product of a-th, b-th and c-th finite-difference stencils along the axes.
fn file_order(order: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for lx in (0..=order).rev() {
        for ly in (0..=order - lx).rev() {
            out.push((lx, ly, order - lx - ly));
        }
    }
    out
}

fn axis_stencil(a: usize, h: f64) -> Vec<(f64, f64)> {
    let binomial = |n: usize, k: usize| (1..=k).fold(1.0, |acc, i| acc * (n + 1 - i) as f64 / i as f64);
    (0..=a).map(|i| ((i as f64 - a as f64 / 2.0) * h, if (a - i) % 2 == 0 { 1.0 } else { -1.0 } * binomial(a, i))).collect()
}


// The exact stencil of a whole moment vector's HARMONIC part: for every component (lx, ly, lz) of the contact-free
// moments, the product of the a-th, b-th and c-th finite-difference stencils, weighted by that moment. Real point
// charges, so the reference is the exact (contact-including) interaction of a distribution whose own moments are
// the harmonic ones -- traceless up to O(h^2), hence with no contact term of its own.
fn harmonic_stencil_file(source: [f64; 3], file_values: &[f64; 20], h: f64) -> String {
    let factorial = |n: usize| (1..=n).map(|k| k as f64).product::<f64>();
    let multipole = sources::PointMultipole::from_xfield_moments(source, 3, file_values);
    let mut lines = Vec::new();
    for order in 0..=3usize {
        math_utilities::cartesian::for_each_cartesian_component(order, |lx, ly, lz| {
            let moment = sources::MultipoleSourceTrait::get_contact_free_moment(&multipole, lx, ly, lz);
            if moment == 0.0 {
                return;
            }
            for (dx, wx) in axis_stencil(lx, h) {
                for (dy, wy) in axis_stencil(ly, h) {
                    for (dz, wz) in axis_stencil(lz, h) {
                        let q = moment * wx * wy * wz / (factorial(lx) * factorial(ly) * factorial(lz) * h.powi(order as i32));
                        lines.push(format!("{} {} {} {}", source[0] + dx, source[1] + dy, source[2] + dz, q));
                    }
                }
            }
        });
    }
    format!("{} Bohr 0\n{}\n", lines.len(), lines.join("\n"))
}

// The contact-free operator of EVERY component of orders 0..3, including the ones that carry trace (xx, zzz, xxy, ...),
// against the exact point-charge stencil of its harmonic part, close to the molecule -- where the contact term that
// the operator leaves out would be large if it leaked in (the raw stencil differs from it by 5% for the zz
// quadrupole even 8 Bohr away, for these diffuse basis functions).
#[test]
fn every_multipole_component_matches_the_stencil_of_its_harmonic_part_near_the_molecule() {
    let h = 0.01;
    for source in [[1.2, -0.5, 0.3], [3.4, 1.9, -1.1]] {
        for order in 0..=3usize {
            for (index, (lx, ly, lz)) in file_order(order).into_iter().enumerate() {
                let mut values = [0.0; 20];
                values[[0, 1, 4, 10][order] + index] = 1.0;
                let text: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                let analytic = compute_from_temp_xfield("harm_mp", &format!("1 Bohr 3\n{} {} {} {}\n", source[0], source[1], source[2], text.join(" ")));
                let stencil = compute_from_temp_xfield("harm_fd", &harmonic_stencil_file(source, &values, h));
                let signal = analytic.iter().fold(0.0_f64, |w, v| w.max(v.abs()));
                assert!(signal > 1e-9, "component ({lx},{ly},{lz}) is trivially zero");
                let relative = max_diff(&analytic, &stencil) / signal;
                assert!(
                    relative < 1e-3,
                    "source {source:?}, component ({lx},{ly},{lz}) of order {order}: relative disagreement {relative:e}"
                );
            }
        }
    }
}

// ---- The exact convention ---------------------------------------------------------------------------------
//
// MultipoleConvention::Exact is the interaction of the moments as a real charge distribution (true derivatives of
// 1/r, contact terms included); ContactFree, the default, is the PElib/Dalton one. They differ only in the trace
// parts of orders >= 2, through the electron density AT the multipole's site.

use aedelsten::MultipoleConvention;

fn operator_with(convention: MultipoleConvention, xfield_contents: &str, tag: &str) -> Vec<f64> {
    let path = common::temp_path(&format!("conv_{}.xfield", tag));
    std::fs::write(&path, xfield_contents).unwrap();
    let (mut ctx, n) = setup(path.to_str().unwrap());
    ctx.set_multipole_convention(convention);
    std::fs::remove_file(&path).ok();
    let mut output = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut output, ctx.get_external_multipole_operator().unwrap(), false).unwrap();
    output
}

#[test]
fn contact_free_is_the_default_convention() {
    let (ctx, _n) = setup(concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/general_multipole_quad_xz.xfield"));
    assert_eq!(ctx.get_multipole_convention(), MultipoleConvention::ContactFree);
}

#[test]
fn a_pure_trace_quadrupole_is_exactly_the_contact_term_in_the_exact_convention() {
    // M_xx = M_yy = M_zz = 1 at C: the interaction is -1/2 Laplacian(1/|r-C|) = +2 pi delta(r-C) (the operator's
    // sign convention: a point charge Q gives -Q/|r-C|), so the operator is 2 pi chi_p(C) chi_q(C) -- checked against
    // the AO amplitudes at C. Nothing at all in the contact-free convention.
    let c = [1.2, -0.5, 0.3];
    let contents = format!("1 Bohr 2\n{} {} {} 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0\n", c[0], c[1], c[2]);
    let exact = operator_with(MultipoleConvention::Exact, &contents, "trace_exact");
    let contact_free = operator_with(MultipoleConvention::ContactFree, &contents, "trace_free");
    assert!(contact_free.iter().fold(0.0_f64, |w, x| w.max(x.abs())) < 1e-12);

    let (ctx, n) = setup(concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/general_multipole_quad_xz.xfield"));
    let every: Vec<usize> = (0..ctx.get_n_shells()).collect();
    let mut amplitudes = vec![0.0; n];
    ctx.compute_ao_values(&every, &c, 3, &mut amplitudes, None).unwrap();
    let mut worst = 0.0_f64;
    let mut largest = 0.0_f64;
    for p in 0..n {
        for q in 0..n {
            let expected = 2.0 * std::f64::consts::PI * amplitudes[p] * amplitudes[q];
            worst = worst.max((exact[p + q * n] - expected).abs());
            largest = largest.max(expected.abs());
        }
    }
    assert!(largest > 1e-3, "the AO amplitudes at the site should not all vanish");
    assert!(worst < 1e-10 * largest.max(1.0), "exact operator differs from 2 pi chi chi by {worst:e} (scale {largest:e})");
}

// The exact operator of every component of orders 0..3 equals the RAW point-charge stencil (real charges), close to
// the molecule.
#[test]
fn every_multipole_component_matches_its_raw_stencil_in_the_exact_convention() {
    let factorial = |n: usize| (1..=n).map(|k| k as f64).product::<f64>();
    let h = 0.01;
    let source = [1.2, -0.5, 0.3];
    for order in 0..=3usize {
        for (index, (lx, ly, lz)) in file_order(order).into_iter().enumerate() {
            let mut values = vec!["0.0"; 20];
            values[[0, 1, 4, 10][order] + index] = "1.0";
            let contents = format!("1 Bohr 3\n{} {} {} {}\n", source[0], source[1], source[2], values.join(" "));
            let analytic = operator_with(MultipoleConvention::Exact, &contents, "raw_mp");

            let mut lines = Vec::new();
            for (dx, wx) in axis_stencil(lx, h) {
                for (dy, wy) in axis_stencil(ly, h) {
                    for (dz, wz) in axis_stencil(lz, h) {
                        let q = wx * wy * wz / (factorial(lx) * factorial(ly) * factorial(lz) * h.powi(order as i32));
                        lines.push(format!("{} {} {} {}", source[0] + dx, source[1] + dy, source[2] + dz, q));
                    }
                }
            }
            let stencil = compute_from_temp_xfield("raw_fd", &format!("{} Bohr 0\n{}\n", lines.len(), lines.join("\n")));
            let signal = analytic.iter().fold(0.0_f64, |w, v| w.max(v.abs()));
            let relative = max_diff(&analytic, &stencil) / signal;
            assert!(relative < 1e-3, "component ({lx},{ly},{lz}) of order {order}: relative disagreement {relative:e}");
        }
    }
}

#[test]
fn both_conventions_agree_when_the_moments_are_traceless() {
    // xz quadrupole and xyz octupole have no trace: the two operators are identical.
    let xz = "1 Bohr 2\n1.2 -0.5 0.3 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0\n";
    let xyz = "1 Bohr 3\n1.2 -0.5 0.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0\n";
    for (tag, contents) in [("agree_xz", xz), ("agree_xyz", xyz)] {
        let a = operator_with(MultipoleConvention::Exact, contents, &format!("{tag}_e"));
        let b = operator_with(MultipoleConvention::ContactFree, contents, &format!("{tag}_c"));
        assert!(max_diff(&a, &b) < 1e-12 * (1.0 + a.iter().fold(0.0_f64, |w, x| w.max(x.abs()))), "{tag}");
    }
}
