// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The X2C picture-change integrals of external point multipoles (p.Vp and sigma.p V sigma.p of the potential of the
// environment's charges, dipoles, quadrupoles, ...), against a finite-difference stencil of point charges whose own
// picture-change integrals go through the point-charge kernels -- the same references
// tests/external_multipole_source.rs uses for the plain operator. Every component of orders 1 to 3, scalar and
// spin-orbit, in both conventions: the exact one against the raw stencil of the moment, the contact-free one against
// the stencil of its harmonic part.

mod common;

use aedelsten::{AedelstenContext, ContractionType, MultipoleConvention};

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/fock_builder_test.inp");
// The stencil's error is O(h^2) (checked: it falls fourfold per halving, 0.02 -> 0.01 -> 0.005).
const H: f64 = 0.005;
const SOURCE: [f64; 3] = [1.2, -0.5, 0.3];

/// The scalar picture-change integrals followed by the four spin-orbit planes (x, y, z, scalar), over the primitive
/// basis.
fn picture_change(xfield_contents: &str, tag: &str, convention: MultipoleConvention) -> (Vec<f64>, Vec<f64>) {
    let path = common::temp_path(&format!("pc_multipole_{tag}.xfield"));
    std::fs::write(&path, xfield_contents).unwrap();
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_external_multipoles(path.to_str().unwrap()).unwrap();
    ctx.set_multipole_convention(convention);
    std::fs::remove_file(&path).ok();
    let n = ctx.get_n_prim_unique();
    let (mut scalar, mut spin_orbit) = (vec![0.0; n * n], vec![0.0; 4 * n * n]);
    ctx.compute_one_electron_picture_change_integrals(&mut scalar, ctx.get_external_multipole_operator().unwrap(), false).unwrap();
    ctx.compute_one_electron_picture_change_integrals(&mut spin_orbit, ctx.get_external_multipole_operator().unwrap(), true).unwrap();
    (scalar, spin_orbit)
}

/// The xfield components of one order, in the file's order (x descending, then y descending).
fn file_order(order: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for lx in (0..=order).rev() {
        for ly in (0..=order - lx).rev() {
            out.push((lx, ly, order - lx - ly));
        }
    }
    out
}

/// The a-th central difference along one axis: offsets and weights.
fn axis_stencil(a: usize, h: f64) -> Vec<(f64, f64)> {
    let binomial = |n: usize, k: usize| (1..=k).fold(1.0, |acc, i| acc * (n + 1 - i) as f64 / i as f64);
    (0..=a).map(|i| ((i as f64 - a as f64 / 2.0) * h, if (a - i) % 2 == 0 { 1.0 } else { -1.0 } * binomial(a, i))).collect()
}

/// Point charges whose moments up to order 3 are `moments(lx, ly, lz)` (to O(h^2)).
fn stencil_file(moments: impl Fn(usize, usize, usize) -> f64) -> String {
    let factorial = |n: usize| (1..=n).map(|k| k as f64).product::<f64>();
    let mut lines = Vec::new();
    for order in 0..=3usize {
        for (lx, ly, lz) in file_order(order) {
            let moment = moments(lx, ly, lz);
            if moment == 0.0 {
                continue;
            }
            for (dx, wx) in axis_stencil(lx, H) {
                for (dy, wy) in axis_stencil(ly, H) {
                    for (dz, wz) in axis_stencil(lz, H) {
                        let q = moment * wx * wy * wz / (factorial(lx) * factorial(ly) * factorial(lz) * H.powi(order as i32));
                        lines.push(format!("{} {} {} {}", SOURCE[0] + dx, SOURCE[1] + dy, SOURCE[2] + dz, q));
                    }
                }
            }
        }
    }
    format!("{} Bohr 0\n{}\n", lines.len(), lines.join("\n"))
}

fn relative_difference(a: &[f64], b: &[f64]) -> (f64, f64) {
    let signal = b.iter().fold(0.0_f64, |w, v| w.max(v.abs()));
    (a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / signal, signal)
}

#[test]
fn every_multipole_component_matches_its_point_charge_stencil_through_the_picture_change() {
    for convention in [MultipoleConvention::Exact, MultipoleConvention::ContactFree] {
        for order in 1..=3usize {
            for (index, (lx, ly, lz)) in file_order(order).into_iter().enumerate() {
                let mut values = [0.0; 20];
                values[[0, 1, 4, 10][order] + index] = 1.0;
                let text: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                let source = format!("1 Bohr 3\n{} {} {} {}\n", SOURCE[0], SOURCE[1], SOURCE[2], text.join(" "));
                let (scalar, spin_orbit) = picture_change(&source, "mp", convention);

                // Exact: the raw moment itself. Contact-free: its harmonic part, the moments the operator reads.
                let multipole = sources::PointMultipole::from_xfield_moments(SOURCE, 3, &values);
                let moments = |a: usize, b: usize, c: usize| match convention {
                    MultipoleConvention::Exact => sources::MultipoleSourceTrait::get_moment(&multipole, a, b, c),
                    MultipoleConvention::ContactFree => sources::MultipoleSourceTrait::get_contact_free_moment(&multipole, a, b, c),
                };
                let (scalar_ref, spin_orbit_ref) = picture_change(&stencil_file(moments), "fd", convention);

                // The spin-orbit planes x, y, z on their own (the fourth, scalar, plane would dominate the comparison).
                let vector = spin_orbit.len() * 3 / 4;
                let (scalar_diff, scalar_signal) = relative_difference(&scalar, &scalar_ref);
                let (so_diff, so_signal) = relative_difference(&spin_orbit[..vector], &spin_orbit_ref[..vector]);
                eprintln!("{convention:?} ({lx},{ly},{lz}): scalar {scalar_diff:.2e} of {scalar_signal:.2e}, spin-orbit {so_diff:.2e} of {so_signal:.2e}");
                assert!(scalar_signal > 1e-6 && so_signal > 1e-6, "({lx},{ly},{lz}): trivially zero");
                assert!(scalar_diff < 1e-3, "{convention:?} ({lx},{ly},{lz}): scalar off the stencil by {scalar_diff:e}");
                assert!(so_diff < 1e-3, "{convention:?} ({lx},{ly},{lz}): spin-orbit off the stencil by {so_diff:e}");
            }
        }
    }
}

#[test]
fn charges_given_as_multipoles_match_the_point_charge_kernels() {
    // An order-1 file whose dipoles are zero: the multipole path, which must reduce to the point-charge one.
    let as_multipoles = "2 Bohr 1\n1.2 -0.5 0.3 0.7 0.0 0.0 0.0\n-0.4 2.1 1.6 -0.3 0.0 0.0 0.0\n";
    let as_charges = "2 Bohr 0\n1.2 -0.5 0.3 0.7\n-0.4 2.1 1.6 -0.3\n";
    for convention in [MultipoleConvention::Exact, MultipoleConvention::ContactFree] {
        let (scalar, spin_orbit) = picture_change(as_multipoles, "q_mp", convention);
        let (scalar_ref, spin_orbit_ref) = picture_change(as_charges, "q_pc", convention);
        let (scalar_diff, _) = relative_difference(&scalar, &scalar_ref);
        let (so_diff, _) = relative_difference(&spin_orbit, &spin_orbit_ref);
        assert!(scalar_diff < 1e-13 && so_diff < 1e-13, "{convention:?}: {scalar_diff:e}, {so_diff:e}");
    }
}
