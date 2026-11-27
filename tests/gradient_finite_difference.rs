// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The one-electron gradient contracted with a density (AedelstenContext::compute_one_electron_gradient,
// `G[c, X] = sum_mn D_mn dO_mn/dX_c` for every centre X) against central finite differences of
// `sum_mn D_mn O_mn`, the value integrals (compute_one_electron_integrals) at displaced geometries: a different code
// path (the value kernels share nothing with the gradient kernels but the shell pairs and Hermite coefficients), so
// agreement is not circular. Every moveable centre of each system, each axis.
//
// Systems:
// - three H atoms, H1 and H2 real nuclei, H3 a pseudocharge ghost (basis functions, no charge): a pair of two centres
//   that are both a basis centre and a charge source, a source on a third atom, a centre with no source of its own;
//   point and finite (Visscher-Dyall) nuclei; general, segmented and primitive contraction;
// - the core-AIMP Sc test entry (test_fixtures/basis/TEST-CORE-AIMP: its own valence basis, so a moveable centre
//   that is the AIMP's source) next to H: each part of the AIMP's Coulomb and exchange operators; general and
//   primitive contraction (its valence basis is generally contracted);
// - an embedding AIMP (no centre of its own, fixed) next to H: only H moves;
// - two H next to an external dipole or quadrupole (fixed).

mod common;

use std::io::Write;

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};

const H_STEP: f64 = 1.0e-4; // central-difference step, bohr
const TOL: f64 = 1.0e-6; // relative to 1 + |finite difference|: the difference's truncation error, O(H_STEP^2)

// A fixed, non-trivial density with an antisymmetric part (which must not matter: the operators are symmetric).
fn density(n: usize, salt: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let symmetric = 0.2 + 0.05 * (((i.min(j) * 7 + i.max(j) * 3 + salt) % 11) as f64);
            let antisymmetric = 0.03 * (((i * 5 + j * 2 + salt) % 7) as f64 - 3.0) * if i < j { 1.0 } else { -1.0 };
            d[i + n * j] = symmetric + antisymmetric;
        }
    }
    d
}

// `sum_mn D_mn O_mn` of the operator at `positions`.
fn contracted_value<const N: usize>(
    positions: &[[f64; 3]; N],
    tag: &str,
    d: &[f64],
    build: &dyn Fn(&[[f64; 3]; N], &str) -> AedelstenContext,
    operator: &dyn Fn(&AedelstenContext) -> OneElectronOperator<'_>,
) -> f64 {
    let ctx = build(positions, tag);
    let n = ctx.get_n_cont();
    let mut out = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut out, operator(&ctx), false).unwrap();
    d.iter().zip(&out).map(|(d, o)| d * o).sum()
}

// The gradient at `base` against central differences, for each `(position, centre)` of `moves`: the entry of
// `positions` displaced and the gradient centre it is (they differ when a block before it consumes no centre, as an
// embedding AIMP's). Fixed sources (external multipoles, embedding AIMPs) stay in place in both.
fn assert_gradient_matches_finite_differences<const N: usize>(
    label: &str,
    base: [[f64; 3]; N],
    moves: &[(usize, usize)],
    build: &dyn Fn(&[[f64; 3]; N], &str) -> AedelstenContext,
    operator: &dyn Fn(&AedelstenContext) -> OneElectronOperator<'_>,
) {
    let ctx = build(&base, &format!("{label}_base"));
    let d = density(ctx.get_n_cont(), label.len());
    let mut gradient = vec![f64::NAN; 3 * ctx.get_n_gradient_centres()];
    ctx.compute_one_electron_gradient(&mut gradient, &d, operator(&ctx), None).unwrap();
    let mut largest: f64 = 0.0;
    let mut worst: f64 = 0.0;
    for &(position, centre) in moves {
        for axis in 0..3 {
            let mut plus = base;
            plus[position][axis] += H_STEP;
            let mut minus = base;
            minus[position][axis] -= H_STEP;
            let fd = (contracted_value(&plus, &format!("{label}_p{position}{axis}"), &d, build, operator)
                - contracted_value(&minus, &format!("{label}_m{position}{axis}"), &d, build, operator))
                / (2.0 * H_STEP);
            let analytic = gradient[3 * centre + axis];
            largest = largest.max(fd.abs());
            worst = worst.max((analytic - fd).abs() / (1.0 + fd.abs()));
            assert!(
                (analytic - fd).abs() < TOL * (1.0 + fd.abs()),
                "{label}: centre {centre} axis {axis}: analytic {analytic}, finite difference {fd}, difference {:e}",
                (analytic - fd).abs()
            );
        }
    }
    eprintln!("{label}: max |finite difference| {largest:.3e}, max relative difference {worst:.1e}");
    assert!(largest > 1e-6, "{label}: the finite differences are (almost) zero, the check would be vacuous");
}

// --- three H atoms, the last a pseudocharge ghost ---

// H1 and H2 share a block; H3, a pseudocharge (a per-block flag), has its own.
fn write_input(path: &std::path::Path, positions: &[[f64; 3]; 3]) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "Basis set\n    H.x2c-SVPall").unwrap();
    writeln!(f, "    H1 {:.15} {:.15} {:.15}", positions[0][0], positions[0][1], positions[0][2]).unwrap();
    writeln!(f, "    H2 {:.15} {:.15} {:.15}", positions[1][0], positions[1][1], positions[1][2]).unwrap();
    writeln!(f, "End of Basis\nBasis set\n    H.x2c-SVPall\n    pseudocharge").unwrap();
    writeln!(f, "    H3 {:.15} {:.15} {:.15}", positions[2][0], positions[2][1], positions[2][2]).unwrap();
    writeln!(f, "End of Basis").unwrap();
}

// `finite`: Visscher-Dyall Gaussian nuclei instead of point charges.
fn build_context(positions: &[[f64; 3]; 3], tag: &str, finite: bool, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("grad_fd_{tag}.inp"));
    write_input(&path, positions);
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    if finite {
        ctx.set_finite_nuclear_model(aedelsten::FiniteModel::VisscherDyall).unwrap();
    }
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn geometry() -> [[f64; 3]; 3] {
    [
        [0.0, 0.0, 0.0],   // H1, real nucleus
        [0.15, -0.2, 1.4], // H2, real nucleus (off-axis, breaks accidental symmetry)
        [0.3, 0.4, 3.0],   // H3, pseudocharge ghost
    ]
}

const H_MOVES: [(usize, usize); 3] = [(0, 0), (1, 1), (2, 2)];

fn check_hydrogens(label: &str, finite: bool, contraction: ContractionType, operator: &dyn Fn(&AedelstenContext) -> OneElectronOperator<'_>) {
    assert_gradient_matches_finite_differences(label, geometry(), &H_MOVES, &|p, tag| build_context(p, tag, finite, contraction), operator);
}

#[test]
fn overlap_gradient_matches_finite_differences() {
    check_hydrogens("overlap", false, ContractionType::General, &|_| OneElectronOperator::overlap());
}

#[test]
fn kinetic_gradient_matches_finite_differences() {
    check_hydrogens("kinetic", false, ContractionType::General, &|_| OneElectronOperator::kinetic());
}

#[test]
fn nuclear_attraction_gradient_matches_finite_differences() {
    check_hydrogens("nucattr", false, ContractionType::General, &|ctx| ctx.get_nuclear_attraction_operator());
}

#[test]
fn finite_nuclear_attraction_gradient_matches_finite_differences() {
    check_hydrogens("finite_nucattr", true, ContractionType::General, &|ctx| ctx.get_nuclear_attraction_operator());
}

#[test]
fn sap_screening_gradients_match_finite_differences() {
    check_hydrogens("sap_grasp", false, ContractionType::General, &|ctx| ctx.get_sap_screening_operator(aedelsten::SapFit::Grasp).unwrap());
    check_hydrogens("sap_helfem", false, ContractionType::General, &|ctx| ctx.get_sap_screening_operator(aedelsten::SapFit::HelFem).unwrap());
}

#[test]
fn segmented_and_primitive_gradients_match_finite_differences() {
    for (contraction, tag) in [(ContractionType::Segmented, "segmented"), (ContractionType::Primitive, "primitive")] {
        check_hydrogens(&format!("{tag}_overlap"), false, contraction, &|_| OneElectronOperator::overlap());
        check_hydrogens(&format!("{tag}_kinetic"), false, contraction, &|_| OneElectronOperator::kinetic());
        check_hydrogens(&format!("{tag}_nucattr"), false, contraction, &|ctx| ctx.get_nuclear_attraction_operator());
    }
}

// --- core and embedding AIMPs ---

const CORE_AIMP_SC_BASIS: &str = "Sc.TEST-CORE-AIMP";
const EMBEDDING_AIMP_MG_BASIS: &str = "Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO";

// The AIMP at `positions[0]`, H at `positions[1]`. A core AIMP is a real centre and comes first; an embedding AIMP has
// no centre of its own and comes after every real atom.
fn write_aimp_input(path: &std::path::Path, positions: &[[f64; 3]; 2], aimp_basis: &str) {
    let mut f = std::fs::File::create(path).unwrap();
    let label = if aimp_basis == EMBEDDING_AIMP_MG_BASIS { "Mg1" } else { "Sc1" };
    let aimp = format!("Basis set\n    {aimp_basis}\n    {label} {:.15} {:.15} {:.15}\nEnd of Basis", positions[0][0], positions[0][1], positions[0][2]);
    let h = format!("Basis set\n    H.x2c-SVPall\n    H1 {:.15} {:.15} {:.15}\nEnd of Basis", positions[1][0], positions[1][1], positions[1][2]);
    if aimp_basis == EMBEDDING_AIMP_MG_BASIS {
        writeln!(f, "{h}\n{aimp}").unwrap();
    } else {
        writeln!(f, "{aimp}\n{h}").unwrap();
    }
}

fn build_aimp_context(positions: &[[f64; 3]; 2], tag: &str, aimp_basis: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("aimp_grad_fd_{tag}.inp"));
    write_aimp_input(&path, positions, aimp_basis);
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn aimp_geometry() -> [[f64; 3]; 2] {
    [
        [0.0, 0.0, 0.0],   // Sc1, the core-AIMP centre (valence and AIMP core)
        [0.15, -0.2, 4.0], // H1, real nucleus (off-axis, far enough to keep the overlap with Sc modest)
    ]
}

fn check_core_aimp(label: &str, contraction: ContractionType, operator: &dyn Fn(&AedelstenContext) -> OneElectronOperator<'_>) {
    assert_gradient_matches_finite_differences(label, aimp_geometry(), &[(0, 0), (1, 1)], &|p, tag| build_aimp_context(p, tag, CORE_AIMP_SC_BASIS, contraction), operator);
}

#[test]
fn core_aimp_coulomb_gradients_match_finite_differences() {
    check_core_aimp("core_aimp_point_charge", ContractionType::General, &|ctx| ctx.get_core_aimp_point_charge_operator().unwrap());
    check_core_aimp("core_aimp_screened_coulomb", ContractionType::General, &|ctx| ctx.get_core_aimp_screened_coulomb_operator().unwrap());
}

#[test]
fn core_aimp_coulomb_gradient_is_the_point_charge_plus_the_screened_coulomb_gradient() {
    let ctx = build_aimp_context(&aimp_geometry(), "core_aimp_full", CORE_AIMP_SC_BASIS, ContractionType::General);
    let d = density(ctx.get_n_cont(), 4);
    let gradient = |operator: OneElectronOperator<'_>| {
        let mut out = vec![0.0; 3 * ctx.get_n_gradient_centres()];
        ctx.compute_one_electron_gradient(&mut out, &d, operator, None).unwrap();
        out
    };
    let full = gradient(ctx.get_core_aimp_coulomb_operator().unwrap());
    let point_charge = gradient(ctx.get_core_aimp_point_charge_operator().unwrap());
    let screened = gradient(ctx.get_core_aimp_screened_coulomb_operator().unwrap());
    assert!(full.iter().any(|x| x.abs() > 1e-3), "vacuous comparison");
    for (k, ((f, p), s)) in full.iter().zip(&point_charge).zip(&screened).enumerate() {
        assert!((f - (p + s)).abs() < 1e-11, "element {k}: full {f}, point charge {p} + screened {s}");
    }
}

#[test]
fn core_aimp_exchange_gradients_match_finite_differences() {
    check_core_aimp("core_aimp_projection", ContractionType::General, &|ctx| ctx.get_core_aimp_projection_operator().unwrap());
    check_core_aimp("core_aimp_ksro", ContractionType::General, &|ctx| ctx.get_core_aimp_spectrally_resolved_exchange_operator().unwrap());
    check_core_aimp("core_aimp_exchange", ContractionType::General, &|ctx| ctx.get_core_aimp_exchange_operator().unwrap());
}

#[test]
fn core_aimp_gradients_match_finite_differences_for_a_primitive_basis() {
    check_core_aimp("primitive_core_aimp_coulomb", ContractionType::Primitive, &|ctx| ctx.get_core_aimp_coulomb_operator().unwrap());
    check_core_aimp("primitive_core_aimp_projection", ContractionType::Primitive, &|ctx| ctx.get_core_aimp_projection_operator().unwrap());
}

#[test]
fn embedding_aimp_screened_coulomb_gradient_matches_finite_differences() {
    // The embedding AIMP at positions[0] stays; H1, at positions[1], is gradient centre 0.
    let geometry = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    assert_gradient_matches_finite_differences(
        "embedding_aimp_screened_coulomb",
        geometry,
        &[(1, 0)],
        &|p, tag| build_aimp_context(p, tag, EMBEDDING_AIMP_MG_BASIS, ContractionType::General),
        &|ctx| ctx.get_embedding_aimp_screened_coulomb_operator().unwrap(),
    );
}

// --- external multipoles (fixed) ---

fn check_external_multipoles(label: &str, xfield: &str) {
    let xfield_path = common::temp_path(&format!("multipole_grad_fd_{label}.xfield"));
    std::fs::write(&xfield_path, xfield).unwrap();
    let build = |positions: &[[f64; 3]; 2], tag: &str| {
        let path = common::temp_path(&format!("multipole_grad_fd_{tag}.inp"));
        let [h1, h2] = positions;
        std::fs::write(&path, format!("Basis set\n    H.x2c-SVPall\n    H1 {:.15} {:.15} {:.15}\n    H2 {:.15} {:.15} {:.15}\nEnd of Basis\n", h1[0], h1[1], h1[2], h2[0], h2[1], h2[2])).unwrap();
        let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
        std::fs::remove_file(&path).ok();
        ctx.set_shell_pairs(None).unwrap();
        ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
        ctx.set_external_multipoles(xfield_path.to_str().unwrap()).unwrap();
        ctx
    };
    let geometry = [[0.1, 0.2, 0.3], [0.0, 0.0, 1.5]];
    assert_gradient_matches_finite_differences(label, geometry, &[(0, 0), (1, 1)], &build, &|ctx| ctx.get_external_multipole_operator().unwrap());
    std::fs::remove_file(&xfield_path).ok();
}

#[test]
fn external_multipole_dipole_gradient_matches_finite_differences() {
    check_external_multipoles("external_dipole", "1 Bohr 1\n2.5 -1.3 0.7 0.4 0.6 -0.3 0.2\n");
}

#[test]
fn external_multipole_quadrupole_gradient_matches_finite_differences() {
    check_external_multipoles("external_quadrupole", "1 Bohr 2\n2.5 -1.3 0.7 0.4 0.6 -0.3 0.2 0.9 -0.4 0.25 0.5 -0.15 -1.4\n");
}
