// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// AedelstenContext::compute_overlap_with: the overlap between two contexts' bases, which may differ
// in basis set, geometry, harmonic types and contraction type. It is what restarting from orbitals
// in a different basis needs (S_BA in C_orth = X^T S_BA C_A).
//
// The reference is always something Aedelsten can already compute on its own: the ordinary overlap
// of one context, or the off-diagonal block of the ordinary overlap of a *combined* system that
// holds both bases' centres (whose overlap matrix is [[S_BB, S_BA], [S_AB, S_AA]] by construction).

mod common;

use aedelsten::{AedelstenContext, AedelstenError, ContractionType};

const CONTRACTION_TYPES: [ContractionType; 3] = [
    ContractionType::General,
    ContractionType::Segmented,
    ContractionType::Primitive,
];

fn context(tag: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("mixed_overlap_{}.inp", tag));
    std::fs::write(&path, input).unwrap();
    let ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    std::fs::remove_file(&path).ok();
    ctx
}

/// Like `context`, with what the ordinary overlap and the mixed one need to run.
fn ready_context(tag: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    ready_context_screened(tag, input, contraction, None)
}

/// `screening`: the shell-pair overlap screening threshold (`None`: none).
fn ready_context_screened(tag: &str, input: &str, contraction: ContractionType, screening: Option<f64>) -> AedelstenContext {
    let mut ctx = context(tag, input, contraction);
    ctx.set_shell_pairs(screening).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

fn ordinary_overlap(ctx: &AedelstenContext) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut s = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut s, ctx.get_overlap_operator(), false).unwrap();
    s
}

fn mixed_overlap(rows: &AedelstenContext, cols: &AedelstenContext) -> Vec<f64> {
    let mut s = vec![f64::NAN; rows.get_n_cont() * cols.get_n_cont()];
    rows.compute_overlap_with(cols, &mut s).unwrap();
    s
}

fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

const WATER: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.2217\nEnd of Basis\n\
                     Basis set\n    H.x2c-SVPall\n    H1 0.0 1.4309 -0.8867\n    H2 0.0 -1.4309 -0.8867\nEnd of Basis\n";

fn is_primitive(c: ContractionType) -> bool {
    matches!(c, ContractionType::Primitive)
}

#[test]
fn the_same_basis_gives_the_ordinary_overlap_for_every_pair_of_contraction_types() {
    let reference_contracted = ordinary_overlap(&ready_context("ref_c", WATER, ContractionType::General));
    let reference_primitive = ordinary_overlap(&ready_context("ref_p", WATER, ContractionType::Primitive));

    for (i, &rows_type) in CONTRACTION_TYPES.iter().enumerate() {
        for (j, &cols_type) in CONTRACTION_TYPES.iter().enumerate() {
            let rows = ready_context(&format!("r{i}{j}"), WATER, rows_type);
            let cols = ready_context(&format!("c{i}{j}"), WATER, cols_type);
            let s = mixed_overlap(&rows, &cols);
            let label = format!("{rows_type:?} x {cols_type:?}");
            assert_eq!(s.len(), rows.get_n_cont() * cols.get_n_cont(), "{label}");
            assert!(s.iter().all(|v| v.is_finite()), "{label}: every element was written");

            match (is_primitive(rows_type), is_primitive(cols_type)) {
                // Contracted x contracted: general and segmented contraction describe the same AOs.
                (false, false) => assert!(max_difference(&s, &reference_contracted) < 1e-12, "{label}"),
                // Primitive x primitive: the same primitive AOs.
                (true, true) => assert!(max_difference(&s, &reference_primitive) < 1e-12, "{label}"),
                // Contracted x primitive: a rectangular block; the two orientations are transposes.
                _ => {
                    let transposed = mixed_overlap(&cols, &rows);
                    let (n_rows, n_cols) = (rows.get_n_cont(), cols.get_n_cont());
                    for a in 0..n_rows {
                        for b in 0..n_cols {
                            assert!((s[a + b * n_rows] - transposed[b + a * n_cols]).abs() < 1e-12, "{label} ({a},{b})");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn contracted_by_primitive_overlap_reproduces_the_contracted_overlap_through_the_contraction() {
    // <g_i|g_j> = sum_p C[j,p] <g_i|p>, so contracting the primitive columns with each function's own
    // coefficients must give the ordinary contracted overlap. x2c-SVPall is segmented (each primitive in
    // one function), so the coefficient of primitive p in function j is read off `get_primitives` by
    // walking the rows in order.
    let contracted = ready_context("cp_c", WATER, ContractionType::General);
    let primitive = ready_context("cp_p", WATER, ContractionType::Primitive);
    let (n_c, n_p) = (contracted.get_n_cont(), primitive.get_n_cont());
    let s_cp = mixed_overlap(&contracted, &primitive);

    // One (exponent, coefficient) row per primitive of each contracted AO, in AO order, with
    // PrimitiveId.shell/centre/l saying which AO group it belongs to.
    let rows = contracted.get_n_primitive_rows();
    let mut ids = vec![aedelsten::PrimitiveId { centre: 0, l: 0, shell: 0 }; rows];
    let mut prims = vec![aedelsten::Primitive { exponent: 0.0, coefficient: 0.0 }; rows];
    contracted.get_primitive_ids(&mut ids).unwrap();
    contracted.get_primitives(&mut prims).unwrap();
    // Every contracted shell here is a single contraction of `k` consecutive primitives; the
    // primitive context lists the same primitives, shell by shell, in the same order.
    let expected = ordinary_overlap(&contracted);

    let mut got = vec![0.0; n_c * n_c];
    // Map: contracted AO j (a group of (2l+1) functions sharing radial coefficients) uses primitives
    // [start, start + k); its m-components use the matching primitive components. Build C explicitly.
    let mut c = vec![0.0; n_c * n_p]; // C[j, p], column-major (n_c rows)
    let mut ao = 0usize; // contracted AO index
    let mut prim_ao = 0usize; // primitive AO index
    let mut r = 0usize;
    while r < rows {
        let (centre, l, shell) = (ids[r].centre, ids[r].l, ids[r].shell);
        let mut k = 0;
        while r + k < rows && ids[r + k].centre == centre && ids[r + k].l == l && ids[r + k].shell == shell {
            k += 1;
        }
        let degeneracy = 2 * l + 1;
        for m in 0..degeneracy {
            for q in 0..k {
                // primitive q of this shell, component m: primitive AOs are listed primitive-major
                // within a shell, component-minor
                c[(ao + m) + (prim_ao + q * degeneracy + m) * n_c] = prims[r + q].coefficient;
            }
        }
        ao += degeneracy;
        prim_ao += k * degeneracy;
        r += k;
    }
    assert_eq!((ao, prim_ao), (n_c, n_p), "the walk covered every AO");

    for i in 0..n_c {
        for j in 0..n_c {
            got[i + j * n_c] = (0..n_p).map(|p| s_cp[i + p * n_c] * c[j + p * n_c]).sum();
        }
    }
    // The primitive AOs carry their own normalization, so the coefficients in `get_primitives` are
    // the contraction coefficients of *normalized* primitives, exactly what this needs.
    assert!(max_difference(&got, &expected) < 1e-10, "max difference {}", max_difference(&got, &expected));
}

/// A combined system: `first` and `second` are the two contexts' (single) basis blocks; the
/// combined overlap's off-diagonal block is what `compute_overlap_with` must return.
fn assert_matches_combined(tag: &str, first: &str, second: &str, contraction: ContractionType) {
    let a = ready_context(&format!("{tag}_a"), first, contraction);
    let b = ready_context(&format!("{tag}_b"), second, contraction);
    // The reference's shell-pair screening is switched off: the ordinary overlap drops primitive pairs below the
    // overlap threshold, the mixed one applies no screening, so against a screened reference they differ by up to
    // that much.
    let combined = ready_context_screened(&format!("{tag}_ab"), &format!("{first}{second}"), contraction, Some(0.0));
    let (n_a, n_b) = (a.get_n_cont(), b.get_n_cont());
    let s = ordinary_overlap(&combined);
    let n = n_a + n_b;
    assert_eq!(combined.get_n_cont(), n);

    let a_by_b = mixed_overlap(&a, &b);
    let b_by_a = mixed_overlap(&b, &a);
    for i in 0..n_a {
        for j in 0..n_b {
            assert!((a_by_b[i + j * n_a] - s[i + (n_a + j) * n]).abs() < 1e-12, "{tag} A x B ({i},{j})");
            assert!((b_by_a[j + i * n_b] - s[(n_a + j) + i * n]).abs() < 1e-12, "{tag} B x A ({j},{i})");
        }
    }
    // Not vacuous: the two bases really do overlap.
    assert!(a_by_b.iter().any(|v| v.abs() > 1e-3), "{tag}: expected some overlap");
}

#[test]
fn different_basis_sets_at_different_positions_match_the_off_diagonal_block_of_the_combined_system() {
    let sto = "Basis set\n    H.x2c-SVPall\n    Ha 0.0 0.0 0.0\nEnd of Basis\n";
    let dz = "Basis set\n    H.x2c-SVPall\n    Hb 0.3 -0.2 1.7\nEnd of Basis\n";
    for (i, contraction) in [ContractionType::General, ContractionType::Segmented].into_iter().enumerate() {
        assert_matches_combined(&format!("sto_dz{i}"), sto, dz, contraction);
    }
}

#[test]
fn cartesian_and_spherical_d_shells_in_different_bases_overlap_correctly() {
    // Different harmonic types for the same l, and general contractions with many primitives.
    let cartesian = "Basis set\n    H.x2c-SVPall\n    Cartesian d\n    Ha 0.0 0.0 0.0\nEnd of Basis\n";
    let spherical = "Basis set\n    H.x2c-SVPall\n    Hb 0.5 0.4 1.1\nEnd of Basis\n";
    assert_matches_combined("cart_sph", cartesian, spherical, ContractionType::General);
    assert_matches_combined("sph_cart", spherical, cartesian, ContractionType::General);
}

#[test]
fn a_context_overlapped_with_itself_is_its_ordinary_overlap() {
    let ctx = ready_context("self", WATER, ContractionType::General);
    assert!(max_difference(&mixed_overlap(&ctx, &ctx), &ordinary_overlap(&ctx)) < 1e-12);
}

#[test]
fn only_one_context_needs_a_scratch_and_neither_is_an_error() {
    let with_scratch = ready_context("ws_a", WATER, ContractionType::General);
    // A context that only describes an old basis: input resident, nothing else.
    let bare = context("ws_b", WATER, ContractionType::General);
    let expected = mixed_overlap(&with_scratch, &with_scratch);

    assert!(max_difference(&mixed_overlap(&with_scratch, &bare), &expected) < 1e-12, "the other one is bare");
    assert!(max_difference(&mixed_overlap(&bare, &with_scratch), &expected) < 1e-12, "this one is bare");

    let mut out = vec![0.0; bare.get_n_cont() * bare.get_n_cont()];
    match bare.compute_overlap_with(&bare, &mut out) {
        Err(AedelstenError::CacheNotReady(message)) => assert!(message.contains("scratch"), "{message}"),
        other => panic!("expected CacheNotReady, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn with_two_scratchs_the_result_does_not_depend_on_which_is_used() {
    // The larger basis's scratch is the one used; either way the numbers are the same.
    let small = ready_context("two_small", "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\n", ContractionType::General);
    let large = ready_context("two_large", "Basis set\n    H.x2c-SVPall\n    H1 0.2 0.0 0.9\nEnd of Basis\n", ContractionType::General);
    let small_by_large = mixed_overlap(&small, &large);
    let large_by_small = mixed_overlap(&large, &small);
    let (n_s, n_l) = (small.get_n_cont(), large.get_n_cont());
    for i in 0..n_s {
        for j in 0..n_l {
            assert!((small_by_large[i + j * n_s] - large_by_small[j + i * n_l]).abs() < 1e-13);
        }
    }
}

#[test]
fn a_wrongly_sized_output_is_rejected() {
    let a = ready_context("size_a", WATER, ContractionType::General);
    let b = ready_context("size_b", "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\n", ContractionType::General);
    let mut too_small = vec![0.0; a.get_n_cont() * b.get_n_cont() - 1];
    assert!(matches!(a.compute_overlap_with(&b, &mut too_small), Err(AedelstenError::InvalidMatrixDimensions { .. })));
    // Square is the wrong shape for a rectangular pair.
    let mut square = vec![0.0; a.get_n_cont() * a.get_n_cont()];
    assert!(a.compute_overlap_with(&b, &mut square).is_err());
}
