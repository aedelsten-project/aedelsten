// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The one-electron gradient contracted with a density for all centres at once
// (AedelstenContext::compute_one_electron_gradient, a (3, n_centres) matrix): what the finite differences
// (tests/gradient_finite_difference.rs) do not show. Orthogonality ghosts are no gradient centres; the property
// operator has none; operators whose sources are all centres give gradients summing to zero over the centres; frozen
// centres are zero and leave the active ones unchanged, for every mask.

mod common;

use std::io::Write;

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};

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

// --- three H atoms, the last a pseudocharge ghost ---

fn hydrogen_context(finite: bool, tag: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("gradcon_h_{}.inp", tag));
    // The test sets: x2c-SVPall segmented, ANO-RCC at VDZP (H [2s1p]) general and primitive.
    let basis = if matches!(contraction, ContractionType::Segmented) { "H.x2c-SVPall" } else { "H.x2c-SVPall" };
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "Basis set\n    {basis}\n    H1 0.0 0.0 0.0\n    H2 0.15 -0.2 1.4\nEnd of Basis").unwrap();
        writeln!(f, "Basis set\n    {basis}\n    pseudocharge\n    H3 0.3 0.4 3.0\nEnd of Basis").unwrap();
    }
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    if finite {
        ctx.set_finite_nuclear_model(aedelsten::FiniteModel::VisscherDyall).unwrap();
    }
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

// --- core and embedding AIMPs ---

const CORE_AIMP_SC_BASIS: &str = "Sc.TEST-CORE-AIMP";
const EMBEDDING_AIMP_NA_BASIS: &str = "Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO";

fn aimp_context(aimp_basis: &str, aimp: [f64; 3], h: [f64; 3], tag: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("gradcon_aimp_{}.inp", tag));
    {
        let mut f = std::fs::File::create(&path).unwrap();
        let label = if aimp_basis == EMBEDDING_AIMP_NA_BASIS { "Na1" } else { "Sc1" };
        let write_aimp = |f: &mut std::fs::File| writeln!(f, "Basis set\n    {aimp_basis}\n    {label} {} {} {}\nEnd of Basis", aimp[0], aimp[1], aimp[2]).unwrap();
        let write_h = |f: &mut std::fs::File| writeln!(f, "Basis set\n    H.x2c-SVPall\n    H1 {} {} {}\nEnd of Basis", h[0], h[1], h[2]).unwrap();
        // A pure embedding AIMP has no centre of its own and must come after every real atom.
        if aimp_basis == EMBEDDING_AIMP_NA_BASIS {
            write_h(&mut f);
            write_aimp(&mut f);
        } else {
            write_aimp(&mut f);
            write_h(&mut f);
        }
    }
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

// --- orthogonality ghosts are outside the gradient matrix ---

#[test]
fn orthogonality_ghosts_are_not_gradient_centres() {
    let path = common::temp_path("gradcon_ghost.inp");
    std::fs::write(
        &path,
        format!("Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    {EMBEDDING_AIMP_NA_BASIS}\n    orthogonality\n    Na1 3.0 0.0 0.0\nEnd of Basis\n"),
    )
    .unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();

    // H1 is the only gradient centre; the orthogonality ghost (centre 1) is not a column.
    assert_eq!(ctx.get_n_gradient_centres(), 1);
    let n = ctx.get_n_cont();
    let d = density(n, 1);
    let mut gradient = vec![0.0; 3];
    ctx.compute_one_electron_gradient(&mut gradient, &d, OneElectronOperator::overlap(), None).unwrap();
    assert!(gradient.iter().all(|g| g.is_finite()) && gradient.iter().any(|g| g.abs() > 1e-6), "H1's overlap gradient");
    // A matrix with a column for the ghost is a dimension error, not silently accepted.
    let mut too_wide = vec![0.0; 6];
    assert!(ctx.compute_one_electron_gradient(&mut too_wide, &d, OneElectronOperator::overlap(), None).is_err());
}

#[test]
fn the_multipole_property_operator_has_no_gradient() {
    let ctx = hydrogen_context(false, "property", ContractionType::General);
    let n = ctx.get_n_cont();
    let d = density(n, 2);
    let mut gradient = vec![0.0; 3 * ctx.get_n_gradient_centres()];
    let operator = OneElectronOperator::moment(aedelsten::MomentOrigin::new([0.0; 3], 1));
    assert!(ctx.compute_one_electron_gradient(&mut gradient, &d, operator, None).is_err());
}

// Moving every centre together changes nothing for an operator whose sources are all centres (overlap, kinetic,
// nuclear attraction -- finite nuclei too) or that has no sources, so the gradient sums to zero over the
// centres: a check of the source terms alone.
#[test]
fn gradients_of_operators_with_centre_bound_sources_sum_to_zero_over_the_centres() {
    for finite in [false, true] {
        let ctx = hydrogen_context(finite, if finite { "sum_finite" } else { "sum_point" }, ContractionType::General);
        let n = ctx.get_n_cont();
        let n_centres = ctx.get_n_gradient_centres();
        let d = density(n, 3);
        let operators: [(&str, Box<dyn Fn(&AedelstenContext) -> OneElectronOperator<'_>>); 3] = [
            ("overlap", Box::new(|_| OneElectronOperator::overlap())),
            ("kinetic", Box::new(|_| OneElectronOperator::kinetic())),
            ("nuclear attraction", Box::new(|c| c.get_nuclear_attraction_operator())),
        ];
        for (label, operator) in operators {
            let mut g = vec![0.0; 3 * n_centres];
            ctx.compute_one_electron_gradient(&mut g, &d, operator(&ctx), None).unwrap();
            assert!(g.iter().any(|v| v.abs() > 1e-4), "{label}: vacuous");
            for c in 0..3 {
                let sum: f64 = (0..n_centres).map(|x| g[c + 3 * x]).sum();
                // the pseudocharge ghost carries basis functions but no nucleus, and moves with the others
                assert!(sum.abs() < 1e-10, "{label} (finite nuclei: {finite}), axis {c}: {sum}");
            }
        }
    }
}

// --- frozen centres -----------------------------------------------------------------------------------------
//
// With an `active` mask a frozen centre's column is zero (not computed) and the active columns are exactly what the
// unmasked call gives, for every mask, for operators of every kind: no sources (overlap), sources that are centres
// (nuclear attraction, core AIMP), fixed sources (external multipoles) and the AIMP exchange contraction.

fn check_masks(ctx: &AedelstenContext, label: &str, operator: &dyn Fn(&AedelstenContext) -> OneElectronOperator<'_>) {
    let n = ctx.get_n_cont();
    let n_centres = ctx.get_n_gradient_centres();
    let d = density(n, label.len() + 5);
    let mut full = vec![0.0; 3 * n_centres];
    ctx.compute_one_electron_gradient(&mut full, &d, operator(ctx), None).unwrap();
    assert!(full.iter().any(|v| v.abs() > 1e-6), "{label}: vacuous");
    for bits in 0..(1usize << n_centres) {
        let mask: Vec<bool> = (0..n_centres).map(|centre| bits >> centre & 1 == 1).collect();
        let mut masked = vec![f64::NAN; 3 * n_centres];
        ctx.compute_one_electron_gradient(&mut masked, &d, operator(ctx), Some(&mask)).unwrap();
        for centre in 0..n_centres {
            for c in 0..3 {
                let want = if mask[centre] { full[c + 3 * centre] } else { 0.0 };
                assert!((masked[c + 3 * centre] - want).abs() < 1e-11, "{label}, mask {mask:?}, centre {centre} axis {c}: {} vs {want}", masked[c + 3 * centre]);
            }
        }
    }
    // A mask of the wrong length is a dimension error.
    let mut out = vec![0.0; 3 * n_centres];
    assert!(ctx.compute_one_electron_gradient(&mut out, &d, operator(ctx), Some(&vec![true; n_centres + 1])).is_err());
}

#[test]
fn frozen_centres_are_zero_and_the_active_ones_unchanged() {
    let hydrogens = hydrogen_context(false, "mask_point", ContractionType::General);
    check_masks(&hydrogens, "overlap", &|_| OneElectronOperator::overlap());
    check_masks(&hydrogens, "kinetic", &|_| OneElectronOperator::kinetic());
    check_masks(&hydrogens, "nuclear attraction", &|c| c.get_nuclear_attraction_operator());
    let finite = hydrogen_context(true, "mask_finite", ContractionType::General);
    check_masks(&finite, "finite nuclear attraction", &|c| c.get_nuclear_attraction_operator());

    let core = aimp_context(CORE_AIMP_SC_BASIS, [0.0, 0.0, 0.0], [0.15, -0.2, 4.0], "mask_core", ContractionType::General);
    check_masks(&core, "core aimp coulomb", &|c| c.get_core_aimp_coulomb_operator().unwrap());
    check_masks(&core, "core aimp exchange", &|c| c.get_core_aimp_exchange_operator().unwrap());
    check_masks(&core, "core aimp projection", &|c| c.get_core_aimp_projection_operator().unwrap());

    let embedding = aimp_context(EMBEDDING_AIMP_NA_BASIS, [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], "mask_embedding", ContractionType::General);
    check_masks(&embedding, "embedding aimp screened coulomb", &|c| c.get_embedding_aimp_screened_coulomb_operator().unwrap());
}

#[test]
fn frozen_centres_with_external_multipoles() {
    let input = common::temp_path("gradcon_mask_mp.inp");
    let xfield_path = common::temp_path("gradcon_mask_mp.xfield");
    std::fs::write(&input, "Basis set\n    H.x2c-SVPall\n    H1 0.1 0.2 0.3\n    H2 0.0 0.0 1.5\n    H3 1.0 -0.5 3.0\nEnd of Basis\n").unwrap();
    std::fs::write(&xfield_path, "1 Bohr 1\n2.5 -1.3 0.7 0.4 0.6 -0.3 0.2\n").unwrap();
    let mut ctx = AedelstenContext::new(input.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_external_multipoles(xfield_path.to_str().unwrap()).unwrap();
    check_masks(&ctx, "external dipole", &|c| c.get_external_multipole_operator().unwrap());
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&xfield_path).ok();
}
