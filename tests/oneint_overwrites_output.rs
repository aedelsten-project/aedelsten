// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Every one-electron entry point leaves in `output` its integrals and nothing else, whatever the buffer held: a call
// into a buffer filled with a non-zero value equals a call into a zeroed one, bit for bit. The atoms are far apart, so
// the overlap screening drops shell pairs (and primitive products) whose blocks no kernel writes. The multi-rank
// counterpart is examples/mpi_oneint_check.rs.

mod common;

use aedelsten::{AedelstenContext, ContractionType, MomentOrigin, OneElectronOperator, Site, SiteSelection};

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/screened_pairs_test.inp");
const DIRT: f64 = 0.75;
const OVERLAP_THRESHOLD: f64 = 1e-12;

fn context() -> AedelstenContext {
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(Some(OVERLAP_THRESHOLD)).unwrap();
    ctx.set_primitive_shell_pairs(Some(OVERLAP_THRESHOLD)).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    let path = common::temp_path("overwrites_output.xfield");
    std::fs::write(&path, "2 Bohr 0\n 0.0 -2.0 25.0  0.6\n 0.5  1.0 75.0 -0.9\n").unwrap();
    ctx.set_external_multipoles(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_embedding_aimp_operators().unwrap();
    ctx
}

fn with_x2c(ctx: &mut AedelstenContext) {
    let n = ctx.get_n_prim_unique();
    let mut integrals = [vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]];
    ctx.compute_one_electron_integrals(&mut integrals[0], OneElectronOperator::overlap(), true).unwrap();
    ctx.compute_one_electron_integrals(&mut integrals[1], OneElectronOperator::kinetic(), true).unwrap();
    ctx.compute_one_electron_integrals(&mut integrals[2], ctx.get_nuclear_attraction_operator(), true).unwrap();
    ctx.compute_one_electron_picture_change_integrals(&mut integrals[3], ctx.get_nuclear_attraction_operator(), false).unwrap();
    let mut oneham = vec![0.0; n * n];
    let [s, t, v, pvp] = &integrals;
    ctx.construct_x2c_one_electron_hamiltonian(&mut oneham, s, t, v, pvp, None).unwrap();
}

// `compute` into a zeroed buffer and into one filled with DIRT: the same bits (an exact zero compared without its
// sign).
fn assert_overwrites(label: &str, len: usize, compute: impl Fn(&mut [f64])) {
    let mut clean = vec![0.0; len];
    compute(&mut clean);
    let mut dirty = vec![DIRT; len];
    compute(&mut dirty);
    let differing = clean.iter().zip(&dirty).filter(|(a, b)| (**a + 0.0).to_bits() != (**b + 0.0).to_bits()).count();
    assert_eq!(differing, 0, "{label}: {differing} of {len} elements depend on what the buffer held");
}

// H2's functions are the last; their overlap with H1's, 100 bohr away, is exactly zero only if screened out.
#[test]
fn the_fixture_has_screened_out_pairs() {
    let ctx = context();
    for primitive in [false, true] {
        let n = if primitive { ctx.get_n_prim_unique() } else { ctx.get_n_cont() };
        let mut overlap = vec![0.0; n * n];
        ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), primitive).unwrap();
        // H2 in x2c-SVPall: [2s1p], 5 contracted functions over 7 primitives (4s, 1p).
        let h2 = if primitive { n - 7 } else { n - 5 };
        let between = (0..h2).flat_map(|i| (h2..n).map(move |j| (i, j)));
        assert!(between.clone().all(|(i, j)| overlap[i + n * j] == 0.0), "primitive {primitive}: no screened-out block");
    }
}

#[test]
fn the_one_electron_integrals_overwrite_their_output() {
    let ctx = context();
    for primitive in [false, true] {
        let n = if primitive { ctx.get_n_prim_unique() } else { ctx.get_n_cont() };
        let basis = if primitive { "primitive" } else { "contracted" };
        let matrix_operators = [
            ("overlap", OneElectronOperator::overlap()),
            ("kinetic", OneElectronOperator::kinetic()),
            ("nuclear attraction", ctx.get_nuclear_attraction_operator()),
            ("external multipoles", ctx.get_external_multipole_operator().unwrap()),
            ("embedding AIMP point charges", ctx.get_embedding_aimp_point_charge_operator().unwrap()),
            ("embedding AIMP screened Coulomb", ctx.get_embedding_aimp_screened_coulomb_operator().unwrap()),
            ("embedding AIMP Coulomb", ctx.get_embedding_aimp_coulomb_operator().unwrap()),
            ("embedding AIMP projection", ctx.get_embedding_aimp_projection_operator().unwrap()),
            ("embedding AIMP exchange", ctx.get_embedding_aimp_exchange_operator().unwrap()),
        ];
        for (name, operator) in matrix_operators {
            assert_overwrites(&format!("{name}, {basis}"), n * n, |out| {
                ctx.compute_one_electron_integrals(out, operator.clone(), primitive).unwrap()
            });
        }
        let component_operators = [
            ("quadrupole moment", ctx.get_moment_operator(MomentOrigin::new([0.1, 0.2, 50.0], 2)), 6),
            ("angular momentum", OneElectronOperator::angular_momentum([0.3, -0.2, 50.5]), 3),
            ("nabla", OneElectronOperator::nabla(), 3),
        ];
        for (name, operator, planes) in component_operators {
            assert_overwrites(&format!("{name}, {basis}"), planes * n * n, |out| {
                ctx.compute_one_electron_integrals(out, operator.clone(), primitive).unwrap()
            });
        }
    }
}

#[test]
fn the_electric_field_integrals_overwrite_their_output() {
    let ctx = context();
    let n = ctx.get_n_cont();
    let sites: Vec<Site> = [[0.0, 1.0, 5.0], [0.0, 0.0, 50.0], [1.0, -1.0, 95.0]].iter().map(|&p| Site::new(p)).collect();
    assert_overwrites("electric field", 3 * sites.len() * n * n, |out| {
        ctx.compute_electric_field_integrals(SiteSelection::Custom(&sites), 0..sites.len(), out).unwrap()
    });
}

#[test]
fn the_gradients_overwrite_their_output() {
    let ctx = context();
    let n = ctx.get_n_cont();
    let n_centres = ctx.get_n_gradient_centres();
    let density: Vec<f64> = (0..n * n).map(|k| 0.1 + 0.01 * (k % 7) as f64).collect();
    let operators = [
        ("overlap", OneElectronOperator::overlap()),
        ("kinetic", OneElectronOperator::kinetic()),
        ("nuclear attraction", ctx.get_nuclear_attraction_operator()),
        ("external multipoles", ctx.get_external_multipole_operator().unwrap()),
        ("embedding AIMP Coulomb", ctx.get_embedding_aimp_coulomb_operator().unwrap()),
        ("embedding AIMP exchange", ctx.get_embedding_aimp_exchange_operator().unwrap()),
    ];
    for (name, operator) in operators {
        assert_overwrites(&format!("{name} gradient"), 3 * n_centres, |out| {
            ctx.compute_one_electron_gradient(out, &density, operator.clone(), None).unwrap()
        });
    }
}

#[test]
fn the_picture_change_integrals_overwrite_their_output() {
    let mut ctx = context();
    with_x2c(&mut ctx);
    let n = ctx.get_n_prim_unique();
    let operators = [
        ("nuclear attraction", ctx.get_nuclear_attraction_operator()),
        ("external multipoles", ctx.get_external_multipole_operator().unwrap()),
    ];
    for (name, operator) in operators {
        assert_overwrites(&format!("{name}, scalar picture change"), n * n, |out| {
            ctx.compute_one_electron_picture_change_integrals(out, operator.clone(), false).unwrap()
        });
        assert_overwrites(&format!("{name}, spin-orbit picture change"), 4 * n * n, |out| {
            ctx.compute_one_electron_picture_change_integrals(out, operator.clone(), true).unwrap()
        });
    }
    let dipole = ctx.get_moment_operator(MomentOrigin::new([0.0, 0.0, 50.0], 1));
    assert_overwrites("dipole, scalar picture change", 3 * n * n, |out| {
        ctx.compute_one_electron_picture_change_integrals(out, dipole.clone(), false).unwrap()
    });
    assert_overwrites("dipole, spin-orbit picture change", 3 * 4 * n * n, |out| {
        ctx.compute_one_electron_picture_change_integrals(out, dipole.clone(), true).unwrap()
    });
    let nc = ctx.get_n_cont();
    for contracted in [false, true] {
        let m = if contracted { nc } else { n };
        assert_overwrites(&format!("picture-changed dipole, scalar part, contracted {contracted}"), 3 * m * m, |out| {
            let mut spin = vec![0.0; 9 * m * m];
            ctx.compute_picture_changed_one_electron_operator(dipole.clone(), contracted, out, &mut spin).unwrap()
        });
        assert_overwrites(&format!("picture-changed dipole, spin part, contracted {contracted}"), 9 * m * m, |out| {
            let mut scalar = vec![0.0; 3 * m * m];
            ctx.compute_picture_changed_one_electron_operator(dipole.clone(), contracted, &mut scalar, out).unwrap()
        });
    }
    assert_overwrites("AMFI", 6 * n * n, |out| ctx.compute_amfi_integrals(out).unwrap());
}
