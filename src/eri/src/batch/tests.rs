// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The generated kernels (`kernels.rs`, through `generated::batch_generated`) against the BLAS tiles of `batch_xxxx`,
// for every quartet they cover, in both orders of the pairs: s to f shells, general (Dense), segmented (Summed) and
// uncontracted (Identity) pairs, the full and both range-separated operators; quartets of mixed contraction kinds
// (`mixed_kinds`); the per-class benchmark of the two paths.

use basis::contraction::{Contracted, Segmented, ShellKind};
use basis::shell_pairs::{derive_shell_pairs, ShellPair};
use basis::shell_types::decontract_general_shells;
use mmd::hermite_coulomb_integrals::{FullRange, LongRange, RangeVersion, ShortRange};
use mmd::ketside_hermite_coefficients::{compute_ketside_hermite_coefficients, MMDCoefficients};

use super::kernels::{dispatch, L_UNROLL};
use super::tiles::batch_xxxx;

/// The largest difference and the largest element over every covered quartet of `pairs` with operator `R`.
fn compare<R: RangeVersion, S: ShellKind>(pairs: &[ShellPair<S>], omega: f64) -> (f64, f64, usize) {
    let mut scratch = arrays::scratch::Scratch::new(256, arrays::scratch::ByteUnit::MiB).unwrap();
    let (mut difference, mut largest, mut quartets) = (0.0_f64, 0.0_f64, 0);
    let coefficients: Vec<MMDCoefficients> = pairs.iter().map(|pair| compute_ketside_hermite_coefficients(pair, &mut scratch.get_arena()).unwrap()).collect();
    for (pair_ab, bra) in pairs.iter().zip(&coefficients) {
        for (pair_cd, ket) in pairs.iter().zip(&coefficients) {
            let (lqn_ab, lqn_cd) = (pair_ab.get_lqn_a() + pair_ab.get_lqn_b(), pair_cd.get_lqn_a() + pair_cd.get_lqn_b());
            if lqn_ab + lqn_cd > L_UNROLL {
                continue;
            }
            let mut arena = scratch.get_arena();
            let arena = &mut arena;
            let bra_view = bra.mmd_transpose_in(arena).unwrap();
            let reference = batch_xxxx::<R, S, S>(pair_ab, pair_cd, bra_view.as_tensor(), ket.as_tensor(), arena, omega).unwrap();
            let reference = arrays::array8d::Array8DTrait::get_data(&reference).to_vec();
            let generated = dispatch::<R, S, S>(lqn_ab, lqn_cd, pair_ab, pair_cd, bra_view.get_data(), ket.get_data(), arena, omega).unwrap().unwrap();
            assert_eq!(generated.len(), reference.len(), "({lqn_ab}, {lqn_cd}): batch size");
            for (g, r) in generated.iter().zip(&reference) {
                difference = difference.max((g - r).abs());
                largest = largest.max(r.abs());
            }
            quartets += 1;
        }
    }
    (difference, largest, quartets)
}

fn check<S: ShellKind>(label: &str, pairs: &[ShellPair<S>]) {
    for (operator, (difference, largest, quartets)) in [
        ("full", compare::<FullRange, S>(pairs, 0.0)),
        ("long range", compare::<LongRange, S>(pairs, 0.4)),
        ("short range", compare::<ShortRange, S>(pairs, 0.4)),
    ] {
        eprintln!("{label}, {operator}: {quartets} quartets, largest {largest:.3e}, difference {difference:.3e}");
        assert!(quartets > 0 && largest > 1e-2, "{label}, {operator}: vacuous");
        assert!(difference <= 1e-13 * largest, "{label}, {operator}: difference {difference:e} of {largest:e}");
    }
}

#[test]
fn generated_kernels_equal_the_blas_tiles() {
    assert_eq!(L_UNROLL, 6);
    let dir = std::env::temp_dir().join(format!("aedelsten_generated_ere_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("oh.inp");
    // ANO-RCC: general contraction, s to f on O (l_ab + l_cd up to 6 among the covered quartets).
    std::fs::write(
        &path,
        "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 1.9\nEnd of Basis\n",
    )
    .unwrap();
    // Segmented on a segmented set (ANO-RCC is generally contracted): CuH in x2c-SVPall, s to f on Cu.
    let segmented_path = dir.join("cuh.inp");
    std::fs::write(
        &segmented_path,
        "Basis set\n    Cu.x2c-SVPall\n    Cu1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 2.9\nEnd of Basis\n",
    )
    .unwrap();
    let path = path.to_str().unwrap();
    let general = basis::setup_basis_system::<Contracted>(path).unwrap();
    let segmented = basis::setup_basis_system::<Segmented>(segmented_path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (uncontracted, _) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
    check("general", &derive_shell_pairs(general.get_basis_shells(), None).unwrap());
    check("segmented", &derive_shell_pairs(segmented.get_basis_shells(), None).unwrap());
    check("uncontracted", &derive_shell_pairs(&uncontracted, None).unwrap());
}

/// Per class `(l_a, l_b | l_c, l_d)` in the walks' order (the pair with more components in the bra), the time of the
/// generated kernels and of the BLAS tiles over every covered quartet of the molecule in `AEDELSTEN_BENCH_INPUT`
/// (`AEDELSTEN_BENCH_KIND`: segmented, the default, contracted or uncontracted), each quartet `AEDELSTEN_BENCH_REPEATS`
/// times (default 20; the tiles not at all with `AEDELSTEN_BENCH_SKIP_TILES`), the classes by the generated kernels'
/// time. Run with `--release --ignored --nocapture`.
#[test]
#[ignore]
fn benchmark_generated_kernels_against_the_blas_tiles() {
    let input = std::env::var("AEDELSTEN_BENCH_INPUT").expect("AEDELSTEN_BENCH_INPUT: an input file");
    // Basis files are looked up in the current directory: the input's.
    std::env::set_current_dir(std::path::Path::new(&input).parent().unwrap()).unwrap();
    match std::env::var("AEDELSTEN_BENCH_KIND").as_deref().unwrap_or("segmented") {
        "segmented" => bench_pairs(&derive_shell_pairs(basis::setup_basis_system::<Segmented>(&input).unwrap().get_basis_shells(), None).unwrap()),
        "contracted" => bench_pairs(&derive_shell_pairs(basis::setup_basis_system::<Contracted>(&input).unwrap().get_basis_shells(), None).unwrap()),
        "uncontracted" => {
            let general = basis::setup_basis_system::<Contracted>(&input).unwrap();
            let (uncontracted, _) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
            bench_pairs(&derive_shell_pairs(&uncontracted, None).unwrap())
        }
        kind => panic!("AEDELSTEN_BENCH_KIND: segmented, contracted or uncontracted, not {kind}"),
    }
}

fn bench_pairs<S: ShellKind>(pairs: &[ShellPair<S>]) {
    use std::collections::BTreeMap;
    use std::time::Instant;
    let repeats: usize = std::env::var("AEDELSTEN_BENCH_REPEATS").ok().and_then(|r| r.parse().ok()).unwrap_or(20);
    let tile_repeats = if std::env::var("AEDELSTEN_BENCH_SKIP_TILES").is_ok() { 0 } else { repeats };
    let mut scratch = arrays::scratch::Scratch::new(256, arrays::scratch::ByteUnit::MiB).unwrap();
    let coefficients: Vec<MMDCoefficients> = pairs.iter().map(|pair| compute_ketside_hermite_coefficients(pair, &mut scratch.get_arena()).unwrap()).collect();
    // (class) -> (quartets, generated, BLAS tiles)
    let mut times: BTreeMap<(usize, usize, usize, usize), (usize, f64, f64)> = BTreeMap::new();
    let mut primitive_quartets = 0;
    for (i, (pair_ab, bra)) in pairs.iter().zip(&coefficients).enumerate() {
        for (pair_cd, ket) in pairs[i..].iter().zip(&coefficients[i..]) {
            let (lqn_ab, lqn_cd) = (pair_ab.get_lqn_a() + pair_ab.get_lqn_b(), pair_cd.get_lqn_a() + pair_cd.get_lqn_b());
            if lqn_ab + lqn_cd > L_UNROLL {
                continue;
            }
            primitive_quartets += pair_ab.get_products().len() * pair_cd.get_products().len();
            let mut arena = scratch.get_arena();
            let arena = &mut arena;
            let bra_view = bra.mmd_transpose_in(arena).unwrap();
            let started = Instant::now();
            for _ in 0..repeats {
                arena.scope(|quartet| {
                    std::hint::black_box(dispatch::<FullRange, S, S>(lqn_ab, lqn_cd, pair_ab, pair_cd, bra_view.get_data(), ket.get_data(), quartet, 0.0).unwrap().unwrap());
                    Ok::<(), arrays::scratch::ArenaError>(())
                })
                .unwrap();
            }
            let generated = started.elapsed().as_secs_f64();
            let started = Instant::now();
            for _ in 0..tile_repeats {
                arena.scope(|quartet| {
                    std::hint::black_box(batch_xxxx::<FullRange, S, S>(pair_ab, pair_cd, bra_view.as_tensor(), ket.as_tensor(), quartet, 0.0).unwrap());
                    Ok::<(), arrays::scratch::ArenaError>(())
                })
                .unwrap();
            }
            let tiles = started.elapsed().as_secs_f64();
            let entry = times.entry((pair_ab.get_lqn_a(), pair_ab.get_lqn_b(), pair_cd.get_lqn_a(), pair_cd.get_lqn_b())).or_insert((0, 0.0, 0.0));
            *entry = (entry.0 + 1, entry.1 + generated, entry.2 + tiles);
        }
    }
    let mut rows: Vec<_> = times.into_iter().collect();
    rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
    let (total_generated, total_tiles) = rows.iter().fold((0.0, 0.0), |(g, t), (_, (_, a, b))| (g + a, t + b));
    eprintln!("class          quartets   generated (s)  BLAS tiles (s)  ratio");
    for ((la, lb, lc, ld), (count, generated, tiles)) in &rows {
        eprintln!("({la}{lb}|{lc}{ld})  {count:>12}  {generated:>13.4}  {tiles:>14.4}  {:>5.2}", generated / tiles);
    }
    eprintln!("total          {:>10}  {total_generated:>13.4}  {total_tiles:>14.4}  {:>5.2}", "", total_generated / total_tiles);
    eprintln!("primitive quartets {primitive_quartets}, generated {:.2} ns each", total_generated / repeats as f64 / primitive_quartets as f64 * 1e9);
}

mod mixed_kinds {
    // Quartets whose bra and ket pairs are of different contraction kinds (Dense, Summed, Identity), against same-kind
    // quartets through the contraction matrices -- contracting the ket of (X X|U U) with C_G gives (X X|G G), of
    // (X X|F F) with C_S gives (X X|S S), for G/S the general/segmented basis and U/F their primitive blocks -- and
    // against the bra-ket swapped quartets.

    use arrays::matrix::{Matrix, MatrixTrait};
    use basis::shell_pairs::derive_shell_pairs;
    use basis::shell_types::{decontract_general_shells, decontract_segmented_shells};
    use mmd::hermite_coulomb_integrals::FullRange;
    use mmd::ketside_hermite_coefficients::compute_ketside_hermite_coefficients;

    use crate::batch::{compute_eri_batch, ERIBatchType};
    use basis::contraction::ShellKind;
    use basis::shell_pairs::ShellPair;
    use basis::contraction::{Contracted, Segmented};

    /// `(mu nu|lambda sigma)` over all AOs, `mu` fastest; `n_ab` AOs on the bra basis, `n_cd` on the ket basis.
    fn tensor<Sab: ShellKind, Scd: ShellKind>(pairs_ab: &[ShellPair<Sab>], n_ab: usize, pairs_cd: &[ShellPair<Scd>], n_cd: usize) -> Vec<f64> {
        let mut out = vec![0.0; n_ab * n_ab * n_cd * n_cd];
        let index = |m: usize, n: usize, l: usize, s: usize| m + n_ab * (n + n_ab * (l + n_cd * s));
        let mut scratch = arrays::scratch::Scratch::new(64, arrays::scratch::ByteUnit::MiB).unwrap();
        for pair_ab in pairs_ab {
            for pair_cd in pairs_cd {
                // The walks' order (more components in the bra) and the other one: the dispatch takes both.
                if pair_ab.components() >= pair_cd.components() {
                    quartet(&mut scratch, pair_ab, pair_cd, &mut |m, n, l, s, v| out[index(m, n, l, s)] = v);
                } else {
                    quartet(&mut scratch, pair_cd, pair_ab, &mut |l, s, m, n, v| out[index(m, n, l, s)] = v);
                }
            }
        }
        out
    }

    /// Every element of the batch `(pair_ab|pair_cd)` to `put(mu, nu, lambda, sigma, value)`, with the images under
    /// a <-> b and c <-> d.
    fn quartet<Sab: ShellKind, Scd: ShellKind>(
        scratch: &mut arrays::scratch::Scratch,
        pair_ab: &ShellPair<Sab>,
        pair_cd: &ShellPair<Scd>,
        put: &mut impl FnMut(usize, usize, usize, usize, f64),
    ) {
        {
            let mut arena = scratch.get_arena();
            let bra_ket = compute_ketside_hermite_coefficients(pair_ab, &mut arena).unwrap();
            let ket = compute_ketside_hermite_coefficients(pair_cd, &mut arena).unwrap();
            let arena = &mut arena;
            let bra = bra_ket.mmd_transpose_in(arena).unwrap();
            let batch = compute_eri_batch::<Sab, Scd, FullRange>(
                pair_ab, pair_cd, &bra, &ket, arena, 0.0,
            ).unwrap();
            let data = match &batch {
                ERIBatchType::FourD(b) => arrays::array4d::Array4DTrait::get_data(b),
                ERIBatchType::SixD(b) => arrays::array6d::Array6DTrait::get_data(b),
                ERIBatchType::EightD(b) => arrays::array8d::Array8DTrait::get_data(b),
            };
            let (deg_a, deg_b, deg_c, deg_d) = (pair_ab.get_degeneracy_a(), pair_ab.get_degeneracy_b(), pair_cd.get_degeneracy_a(), pair_cd.get_degeneracy_b());
            let block = deg_a * deg_b * deg_c * deg_d;
            let dims_cd = pair_cd.get_contracted_pair_dims();
            let stride_ab = block * dims_cd.0 * dims_cd.1;
            for b in 0..pair_ab.get_n_cont_b() {
                for a in 0..pair_ab.get_n_cont_a() {
                    if pair_ab.get_single_centre() && a > b {
                        continue;
                    }
                    let slot_ab = pair_ab.pair_slot(a, b).unwrap();
                    for d in 0..pair_cd.get_n_cont_b() {
                        for c in 0..pair_cd.get_n_cont_a() {
                            if pair_cd.get_single_centre() && c > d {
                                continue;
                            }
                            let slot_cd = pair_cd.pair_slot(c, d).unwrap();
                            for l in 0..deg_d {
                                for k in 0..deg_c {
                                    for j in 0..deg_b {
                                        for i in 0..deg_a {
                                            let value = data[i + deg_a * (j + deg_b * (k + deg_c * l)) + block * slot_cd + stride_ab * slot_ab];
                                            let mu = pair_ab.get_idx_a() + a * deg_a + i;
                                            let nu = pair_ab.get_idx_b() + b * deg_b + j;
                                            let lambda = pair_cd.get_idx_a() + c * deg_c + k;
                                            let sigma = pair_cd.get_idx_b() + d * deg_d + l;
                                            for (m, n) in [(mu, nu), (nu, mu)] {
                                                for (p, q) in [(lambda, sigma), (sigma, lambda)] {
                                                    put(m, n, p, q, value);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Contracts the two ket indices of an `(n_ab, n_ab, n_prim, n_prim)` tensor with `c` (`n_prim x n_cont`).
    fn contract_ket(x: &[f64], n_ab: usize, c: &Matrix<f64>) -> Vec<f64> {
        let (n_prim, n_cont) = (c.rows(), c.cols());
        let bra = n_ab * n_ab;
        let mut half = vec![0.0; bra * n_cont * n_prim];
        for s in 0..n_prim {
            for k in 0..n_cont {
                for p in 0..n_prim {
                    let w = c[(p, k)];
                    for m in 0..bra {
                        half[m + bra * (k + n_cont * s)] += w * x[m + bra * (p + n_prim * s)];
                    }
                }
            }
        }
        let mut out = vec![0.0; bra * n_cont * n_cont];
        for l in 0..n_cont {
            for s in 0..n_prim {
                let w = c[(s, l)];
                for m in 0..bra * n_cont {
                    out[m + bra * n_cont * l] += w * half[m + bra * n_cont * s];
                }
            }
        }
        out
    }

    /// `(ab|cd) -> (cd|ab)` for an `(n_ab, n_ab, n_cd, n_cd)` tensor.
    fn swap_bra_ket(x: &[f64], n_ab: usize, n_cd: usize) -> Vec<f64> {
        let (bra, ket) = (n_ab * n_ab, n_cd * n_cd);
        let mut out = vec![0.0; x.len()];
        for k in 0..ket {
            for m in 0..bra {
                out[k + ket * m] = x[m + bra * k];
            }
        }
        out
    }

    fn assert_close(label: &str, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len(), "{label}");
        let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let difference = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("{label}: max {scale:.3e}, difference {difference:.3e}");
        assert!(scale > 1e-2, "{label}: vacuous comparison");
        assert!(difference < 1e-11 * scale.max(1.0), "{label}: {difference}");
    }

    #[test]
    fn quartets_of_mixed_contraction_kinds_match_the_contracted_same_kind_quartets() {
        let dir = std::env::temp_dir().join(format!("aedelsten_mixed_kinds_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("oh_far.inp");
        // A segmented set (x2c-SVPall), O bringing a d shell, H far out.
        std::fs::write(&path, "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.3 -0.2 3.1\nEnd of Basis\n").unwrap();
        let path = path.to_str().unwrap();
        let general = basis::setup_basis_system::<Contracted>(path).unwrap();
        let segmented = basis::setup_basis_system::<Segmented>(path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        let (uncontracted, c_g) = decontract_general_shells(general.get_basis_shells(), general.get_basis_info()).unwrap();
        let (folded, c_s) = decontract_segmented_shells(segmented.get_basis_shells(), segmented.get_basis_info()).unwrap();
        let g = derive_shell_pairs(general.get_basis_shells(), None).unwrap();
        let s = derive_shell_pairs(segmented.get_basis_shells(), None).unwrap();
        let u = derive_shell_pairs(&uncontracted, None).unwrap();
        let f = derive_shell_pairs(&folded, None).unwrap();
        let (n_g, n_s, n_u, n_f) = (c_g.cols(), c_s.cols(), c_g.rows(), c_s.rows());
        assert!(n_u > n_g && n_f > n_s, "the bases should be contracted");

        let gg_uu = tensor(&g, n_g, &u, n_u);
        let gg_gg = tensor(&g, n_g, &g, n_g);
        assert_close("(Dense|Identity)", &contract_ket(&gg_uu, n_g, &c_g), &gg_gg);
        let ss_ff = tensor(&s, n_s, &f, n_f);
        assert_close("(Summed|Identity)", &contract_ket(&ss_ff, n_s, &c_s), &tensor(&s, n_s, &s, n_s));
        let gg_ss = tensor(&g, n_g, &s, n_s);
        assert_close("(Dense|Summed)", &gg_ss, &contract_ket(&tensor(&g, n_g, &f, n_f), n_g, &c_s));
        assert_close("(Summed|Dense)", &tensor(&s, n_s, &g, n_g), &swap_bra_ket(&gg_ss, n_g, n_s));
        assert_close("(Identity|Summed)", &tensor(&f, n_f, &s, n_s), &swap_bra_ket(&ss_ff, n_s, n_f));
        assert_close("(Identity|Dense)", &tensor(&u, n_u, &g, n_g), &swap_bra_ket(&gg_uu, n_g, n_u));
    }
}
