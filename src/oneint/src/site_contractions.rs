// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The two contractions of the electric-field integrals <A| (r - C_s) / |r - C_s|^3 |B> over a run of sites, with no
// integral formed per site:
//
//   - the electronic field of densities D_m at the sites, F_(m,s) = Sum_AB D_m,AB <A| (r - C_s) / |r - C_s|^3 |B>:
//     per primitive product the density is carried back through the spherical transformation and the contraction
//     (their transposes) onto the product's Cartesian components and from there onto its Hermite functions,
//       h_tuv = Sum_ab W_ab N_a N_b E^ab_t E^ab_u E^ab_v,
//     so that each site costs one R tensor and three dots: F_(s,k) = -(2 pi / p) Sum_tuv h_tuv R_(t,u,v)+e_k(P - C_s);
//   - the operator of dipoles at the sites, Sum_s mu_s . <A| (r - C_s) / |r - C_s|^3 |B>, the transpose: the
//     external-multipole operator of the dipoles (`add_external_multipole_upper`).
//
// Both run over the pairs and sites they are given on this rank alone and add to the caller's output, so a caller
// sharing out pieces of pairs and sites sums over the ranks once.
//
// Weights, as for the density-contracted gradients: a pair stores one of the two mirror blocks of a symmetric
// operator and the density enters symmetrised, (D + D^T)/2, so an off-diagonal shell pair counts twice and a pair of
// a shell with itself once (within it, likewise, its off-diagonal contracted pairs twice).

use arrays::matrix::{ColumnFactor, Factor, MatrixTrait, MatrixTraitMut, MatrixViewMut};
use mmd::hermite_coefficients::{HermiteCoefficientsTrait, HermiteCoefficientsTraitMut};
use arrays::scratch::{Arena, ArenaError};
use basis::contraction::{Contraction, ShellKind};
use basis::gto_products::GTOProductTrait;
use basis::shell_pairs::ShellPair;
use basis::shell_types::HarmonicType;
use basis::charge_carriers::{MultipoleSourceTrait, SiteTrait};

use crate::inner_functions::{accumulates, weight};
use crate::kernels::{fill_norm_matrix, CoulombKernel, MultipoleSourceMode};

/// Adds the electronic fields of the densities `densities` (`k` consecutive `n x n` matrices, column-major; need not
/// be symmetric) at the sites `sites` to `fields`: density `m`'s field at site `i` to
/// `fields[m column + 3 (first_site + i) ..][..3]` (x, y, z). Over the pairs `shell_pairs` on this rank only.
#[allow(clippy::too_many_arguments)]
pub fn add_electronic_fields<S: ShellKind, Site: SiteTrait>(
    fields: &mut [f64],
    column: usize,
    first_site: usize,
    densities: &[f64],
    n: usize,
    shell_pairs: &[ShellPair<S>],
    sites: &[Site],
    scratch: &mut arrays::scratch::Scratch,
) -> Result<(), ArenaError> {
    let k = densities.len() / (n * n).max(1);
    assert_eq!(densities.len(), k * n * n, "the densities are whole n x n matrices");
    assert_eq!(fields.len(), k * column, "one column of fields per density");
    assert!(3 * (first_site + sites.len()) <= column, "the sites lie within a column");
    if k == 0 || sites.is_empty() {
        return Ok(());
    }
    for pair in shell_pairs {
        let arena = &scratch.get_arena();
        add_pair_fields(pair, fields, column, first_site, densities, n, k, sites, arena)?;
    }
    Ok(())
}

// The densities' blocks of a pair on its contracted pairs, weighted and carried back to Cartesian components:
// `cart_a cart_b x n_slots` per density, density after density.
#[allow(clippy::too_many_arguments)]
fn cartesian_density_blocks<'a, S: ShellKind>(pair: &ShellPair<S>, densities: &[f64], n: usize, k: usize, arena: &Arena<'a>) -> Result<&'a mut [f64], ArenaError> {
    let (lqn_a, lqn_b) = (pair.get_lqn_a(), pair.get_lqn_b());
    let (cart_a, cart_b) = (cart2sph::compute_cartesian_degeneracy(lqn_a), cart2sph::compute_cartesian_degeneracy(lqn_b));
    let spherical_a = pair.get_harmonic_a() == HarmonicType::Spherical;
    let spherical_b = pair.get_harmonic_b() == HarmonicType::Spherical;
    let comps_a = if spherical_a { cart2sph::compute_spherical_degeneracy(lqn_a) } else { cart_a };
    let comps_b = if spherical_b { cart2sph::compute_spherical_degeneracy(lqn_b) } else { cart_b };
    let (dim0, dim1) = pair.get_contracted_pair_dims();
    let n_slots = dim0 * dim1;
    let block = cart_a * cart_b;

    let blocks = arena.alloc_zeroed(block * n_slots * k)?;
    let mut gathered = MatrixViewMut::<f64>::from_arena(comps_a, comps_b, arena)?;
    let mut half = MatrixViewMut::<f64>::from_arena(comps_a, cart_b, arena)?;
    let to_spherical_a = if spherical_a { Some(cart2sph::compute_cartesian_to_spherical_matrix_arena(lqn_a, arena)?) } else { None };
    let to_spherical_b = if spherical_b { Some(cart2sph::compute_cartesian_to_spherical_matrix_arena(lqn_b, arena)?) } else { None };

    let (n_cont_a, n_cont_b, single) = (pair.get_n_cont_a(), pair.get_n_cont_b(), pair.get_single_centre());
    let pair_weight = if single { 1.0 } else { 2.0 };
    let n2 = n * n;
    for (density, density_blocks) in densities.chunks_exact(n2).zip(blocks.chunks_exact_mut(block * n_slots)) {
        for cb in 0..n_cont_b {
            let col0 = pair.get_idx_b() + cb * comps_b;
            let a_end = if single { cb + 1 } else { n_cont_a };
            for ca in 0..a_end {
                let Some(slot) = pair.pair_slot(ca, cb) else {
                    continue;
                };
                let row0 = pair.get_idx_a() + ca * comps_a;
                let factor = 0.5 * pair_weight * if single && ca != cb { 2.0 } else { 1.0 };
                for (j, gathered_col) in gathered.iter_mut().enumerate() {
                    let col = col0 + j;
                    for (i, value) in gathered_col.iter_mut().enumerate() {
                        let row = row0 + i;
                        // The transposed partner, D_(col, row).
                        *value = factor * (density[row + n * col] + density[col + n * row]);
                    }
                }
                // Back through the spherical transformation, its transpose: W = C_a . D . C_b^T.
                let mut target = MatrixViewMut::new(cart_a, cart_b, &mut density_blocks[slot * block..(slot + 1) * block]);
                match (&to_spherical_a, &to_spherical_b) {
                    (None, None) => target.get_data_mut().copy_from_slice(gathered.get_data()),
                    (None, Some(c_b)) => gathered.matmul_prealloc(&c_b.view_transposed(), &mut target),
                    (Some(c_a), None) => c_a.matmul_prealloc(&gathered, &mut target),
                    (Some(c_a), Some(c_b)) => {
                        gathered.matmul_prealloc(&c_b.view_transposed(), &mut half);
                        c_a.matmul_prealloc(&half, &mut target);
                    }
                }
            }
        }
    }
    Ok(blocks)
}

#[allow(clippy::too_many_arguments)]
fn add_pair_fields<S: ShellKind, Site: SiteTrait>(
    pair: &ShellPair<S>,
    fields: &mut [f64],
    column: usize,
    first_site: usize,
    densities: &[f64],
    n: usize,
    k: usize,
    sites: &[Site],
    arena: &Arena<'_>,
) -> Result<(), ArenaError> {
    let (lqn_a, lqn_b) = (pair.get_lqn_a(), pair.get_lqn_b());
    let lqn_ab = lqn_a + lqn_b;
    let (cart_a, cart_b) = (cart2sph::compute_cartesian_degeneracy(lqn_a), cart2sph::compute_cartesian_degeneracy(lqn_b));
    let block = cart_a * cart_b;
    let (dim0, dim1) = pair.get_contracted_pair_dims();
    let per_density = block * dim0 * dim1;

    let blocks = cartesian_density_blocks(pair, densities, n, k, arena)?;
    let cart_comps_a = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_a, arena)?;
    let cart_comps_b = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_b, arena)?;
    let mut hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, 0, arena)?;
    // One order past lqn_ab: each component reads R shifted by one in its own direction.
    let mut r_buffer = mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D::from_arena(lqn_ab + 1, arena)?;
    let tuv_size = mmd::hermite_expansion::hermite_count(lqn_ab);
    let hermite_densities = arena.alloc_zeroed(tuv_size * k)?;
    let expanded = arena.alloc_zeroed(block)?;
    let mut norm_matrix = MatrixViewMut::<f64>::from_arena(cart_a, cart_b, arena)?;

    let contraction = pair.contraction();
    let accumulate = accumulates(&contraction);
    let x_ab = pair.get_separation();
    for (i_product, product) in pair.get_products().iter().enumerate() {
        let (a, b, p) = (product.get_a(), product.get_b(), product.get_p());
        hermite_coefficients.compute(a, b, p, product.get_μ(), &x_ab);
        fill_norm_matrix(&mut norm_matrix, a, b, &cart_comps_a, &cart_comps_b, weight::<S>(product));
        let e_x = hermite_coefficients.get_e_x(0);
        let e_y = hermite_coefficients.get_e_y(0);
        let e_z = hermite_coefficients.get_e_z(0);

        // The product's Hermite densities, one per density.
        for (density_blocks, h) in blocks.chunks_exact(per_density).zip(hermite_densities.chunks_exact_mut(tuv_size)) {
            let w = contraction.expand(density_blocks, if accumulate { 0 } else { i_product }, &mut *expanded);
            h.fill(0.0);
            for (j, lmn_b) in cart_comps_b.iter_col().enumerate() {
                let (lx2, ly2, lz2) = (lmn_b[0], lmn_b[1], lmn_b[2]);
                for (i, lmn_a) in cart_comps_a.iter_col().enumerate() {
                    let w_ij = w[i + cart_a * j] * norm_matrix[(i, j)];
                    if w_ij == 0.0 {
                        continue;
                    }
                    let (lx1, ly1, lz1) = (lmn_a[0], lmn_a[1], lmn_a[2]);
                    for v in 0..=lz1 + lz2 {
                        let e_v = w_ij * e_z[(v, lz1, lz2)];
                        for u in 0..=ly1 + ly2 {
                            let e_uv = e_v * e_y[(u, ly1, ly2)];
                            let base = mmd::hermite_expansion::hermite_index(0, u, v, lqn_ab);
                            for t in 0..=lx1 + lx2 {
                                h[base + t] += e_uv * e_x[(t, lx1, lx2)];
                            }
                        }
                    }
                }
            }
        }

        // Each site: one R tensor at P - C, three dots per density.
        let centre = product.get_product_centre();
        let constant_factor = -2.0 * std::f64::consts::PI / p;
        for (i_site, site) in sites.iter().enumerate() {
            let dist = arrays::vector_math::vector_subtract_3d(&centre, site.get_position());
            r_buffer.compute(p, dist[0], dist[1], dist[2], arrays::vector_math::vector_norm_square_3d(&dist));
            let r_view = r_buffer.view_tuv();
            let offset = 3 * (first_site + i_site);
            for (h, field_column) in hermite_densities.chunks_exact(tuv_size).zip(fields.chunks_exact_mut(column)) {
                let (mut f_x, mut f_y, mut f_z) = (0.0, 0.0, 0.0);
                // (v, u, t) ascending visits the (t, u, v) in hermite_index order.
                let mut idx = 0;
                for v in 0..=lqn_ab {
                    for u in 0..=(lqn_ab - v) {
                        let (row, row_y, row_z) = (r_view.get_t_slice(u, v), r_view.get_t_slice(u + 1, v), r_view.get_t_slice(u, v + 1));
                        for t in 0..=(lqn_ab - u - v) {
                            let h_tuv = h[idx];
                            f_x += h_tuv * row[t + 1];
                            f_y += h_tuv * row_y[t];
                            f_z += h_tuv * row_z[t];
                            idx += 1;
                        }
                    }
                }
                let out = &mut field_column[offset..offset + 3];
                out[0] += constant_factor * f_x;
                out[1] += constant_factor * f_y;
                out[2] += constant_factor * f_z;
            }
        }
    }
    Ok(())
}

/// Adds the external-multipole operator of `sources` over the pairs `shell_pairs` on this rank only to the upper
/// triangle of `output` (`n x n`): the caller mirrors it once its pieces are done. The contact-free moments, which
/// up to order 1 are the moments.
pub fn add_external_multipole_upper<S: ShellKind, P: MultipoleSourceTrait, Mf64: MatrixTraitMut<f64>>(
    output: &mut Mf64,
    shell_pairs: &[ShellPair<S>],
    sources: &[P],
    scratch: &mut arrays::scratch::Scratch,
) -> Result<(), ArenaError> {
    for pair in shell_pairs {
        let arena = &scratch.get_arena();
        let batch = crate::inner_functions::pair_batch::<CoulombKernel<P, MultipoleSourceMode<crate::kernels::ContactFree>>, S>(pair, arena, &sources)?;
        crate::distribute::distribute_pair_batch(pair, output, &batch);
    }
    Ok(())
}
