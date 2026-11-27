// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The ket-side Hermite coefficients of a shell pair (`MMDCoefficients`), in its harmonic basis and with its
//! contraction, their bra-side transpose, and their derivatives with respect to the separation.

use crate::hermite_expansion::*;
use arrays::matrix::*;
use arrays::array3d::*;
use arrays::array4d::*;
use arrays::array5d::*;
use math_utilities::cartesian::for_each_cartesian_component;
use arrays::scratch::{Arena, ArenaError};

use basis::contraction::{Contraction, ShellKind, Strategy};
use basis::shell_pairs::ShellPair;
use basis::gto_products::GTOProductTrait;

use crate::hermite_coefficients::*;

/// The ket-side Hermite coefficients of a shell pair, `(tuv, c, d, primitive pair)`: normalized, in the pair's
/// harmonic basis (`c`, `d` its components), and scaled by the contraction where the pair's kind carries it there.
#[derive(Debug)]
pub struct MMDCoefficients(Array4D<f64>);

impl MMDCoefficients {
    /// The tensor `(tuv, c, d, primitive pair)`.
    pub fn as_tensor(&self) -> &Array4D<f64> {
        &self.0
    }

    /// Its elements, column-major: one `(tuv, c, d)` slab per primitive pair.
    pub fn get_data(&self) -> &[f64] {
        self.0.get_data()
    }

    /// The bra-side layout of these (ket-side) coefficients, `(tuv, c, d, prim) -> (c, d, tuv, prim)`, in `arena`
    /// (the walks keep it for one row of quartets).
    pub fn mmd_transpose_in<'v, 'a: 'v>(&'v self, arena: &Arena<'a>) -> Result<MMDCoefficientsView<'v>, ArenaError> {
        let t = &self.0;
        let (tuv_size, x_size, y_size, prim_size) = (t.dim0(), t.dim1(), t.dim2(), t.dim3());
        // Every element is written below, so no zero fill (`from_arena`'s).
        let data = arena.alloc::<f64>(arrays::scratch::checked_product(&[x_size, y_size, tuv_size, prim_size])?)?;
        let mut transposed = Array4DViewMut::new(x_size, y_size, tuv_size, prim_size, data);
        for p in 0..prim_size {
            for tuv in 0..tuv_size {
                for y in 0..y_size {
                    for x in 0..x_size {
                        transposed[(x, y, tuv, p)] = t[(tuv, x, y, p)];
                    }
                }
            }
        }
        Ok(MMDCoefficientsView(Array4DView::new(x_size, y_size, tuv_size, prim_size, transposed.into_data())))
    }
}

/// Bra-side coefficients (`MMDCoefficients::mmd_transpose_in`), `(c, d, tuv, primitive pair)`, borrowed from an arena.
#[derive(Debug)]
pub struct MMDCoefficientsView<'a>(Array4DView<'a, f64>);

impl<'a> MMDCoefficientsView<'a> {
    /// The tensor `(c, d, tuv, primitive pair)`.
    pub fn as_tensor(&self) -> &Array4DView<'a, f64> {
        &self.0
    }

    /// Its elements, column-major: one `(c, d, tuv)` slab per primitive pair.
    pub fn get_data(&self) -> &[f64] {
        self.0.get_data()
    }
}

/// The ket-side Hermite coefficients of `pair`, any angular momenta in either order: the Cartesian coefficients per
/// primitive pair by the recursion, then the harmonic transform. The result is on the heap; the scratch comes from
/// `scratch` and is freed on return.
pub fn compute_ketside_hermite_coefficients<S: ShellKind>(pair: &ShellPair<S>, scratch: &mut Arena<'_>) -> Result<MMDCoefficients, ArenaError> {
    Ok(MMDCoefficients(scratch.scope(|scratch| ketside_general(pair, scratch))?))
}

/// The Cartesian ket coefficients of one primitive pair (exponents `c`, `d`, `q = c + d`, `cd = c d / q`, separation
/// `dist_cd`) into `e_ket`: the products of the x, y and z Hermite coefficients for every pair of components of
/// `c_comps`, `d_comps` (`3 x n` each), normalized and scaled by the contraction coefficient `cc`. Computes
/// `hermite_coefficients` first.
#[inline(always)]
pub fn compute_e_ket_cartesian_prealloc<C, T, E, M>(
    e_ket: &mut T,
    hermite_coefficients: &mut E,
    c: f64,
    d: f64,
    q: f64,
    cd: f64,
    dist_cd: &[f64; 3],
    c_comps: &M,
    d_comps: &M,
    cc: f64,
) where
    C: Contraction,
    T: KetExpansionCoefficientsTraitMut<f64>,
    E: HermiteCoefficientsTraitMut,
    M: MatrixTrait<usize>
{
    hermite_coefficients.compute(c, d, q, cd, dist_cd);
    let e_cd_x = hermite_coefficients.get_e_x(0);
    let e_cd_y = hermite_coefficients.get_e_y(0);
    let e_cd_z = hermite_coefficients.get_e_z(0);

    for (idx4, lmn_d) in d_comps.iter_col().enumerate() {
        let lx4 = lmn_d[0];
        let ly4 = lmn_d[1];
        let lz4 = lmn_d[2];
        let norm4 = C::prescale(cart2sph::compute_cart_gaussian_norm(d, lx4, ly4, lz4), cc);
        for (idx3, lmn_c) in c_comps.iter_col().enumerate() {
            let lx3 = lmn_c[0];
            let ly3 = lmn_c[1];
            let lz3 = lmn_c[2];
            let norm3 = norm4 * cart2sph::compute_cart_gaussian_norm(c, lx3, ly3, lz3);
            let φ_max = lz3 + lz4 + 1;
            let ν_max = ly3 + ly4 + 1;
            let τ_max = lx3 + lx4 + 1;
            let z_slice = e_cd_z.get_fibre((lz3, lz4));
            let y_slice = e_cd_y.get_fibre((ly3, ly4));
            let x_slice = e_cd_x.get_fibre((lx3, lx4));
            for (φ, e_φ) in z_slice.iter().take(φ_max).enumerate() {
                let e_φ = norm3 * e_φ;
                for (ν, e_ν) in y_slice.iter().take(ν_max).enumerate() {
                    let e_νφ = e_ν * e_φ;
                    let e_ket_t_slice = e_ket.get_t_slice_mut(τ_max, ν, φ, idx3, idx4);
                    for (e_τ_out, e_τ) in e_ket_t_slice.iter_mut().zip(x_slice.iter()) {
                        *e_τ_out = e_τ * e_νφ;
                    }
                }
            }
        }
    }
}

/// `compute_e_ket_cartesian_prealloc` from Hermite coefficients already computed, with the norms of the component
/// pairs given (`normalisation`, `(c, d)`).
#[inline(always)]
pub fn compute_e_ket_cartesian_prealloc_w_norm<C, T, E, M, Mf64>(
    e_ket: &mut T,
    hermite_coefficients: &mut E,
    c_comps: &M,
    d_comps: &M,
    normalisation: &Mf64,
    cc: f64,
) where
    C: Contraction,
    T: KetExpansionCoefficientsTraitMut<f64>,
    E: HermiteCoefficientsTrait,
    M: MatrixTrait<usize>,
    Mf64: MatrixTrait<f64>
{
    let e_cd_x = hermite_coefficients.get_e_x(0);
    let e_cd_y = hermite_coefficients.get_e_y(0);
    let e_cd_z = hermite_coefficients.get_e_z(0);

    for (idx4, lmn_d) in d_comps.iter_col().enumerate() {
        let lx4 = lmn_d[0];
        let ly4 = lmn_d[1];
        let lz4 = lmn_d[2];
        for (idx3, lmn_c) in c_comps.iter_col().enumerate() {
            let lx3 = lmn_c[0];
            let ly3 = lmn_c[1];
            let lz3 = lmn_c[2];
            let norm3 = C::prescale(normalisation[(idx3, idx4)], cc);
            let φ_max = lz3 + lz4 + 1;
            let ν_max = ly3 + ly4 + 1;
            let τ_max = lx3 + lx4 + 1;
            let z_slice = e_cd_z.get_fibre((lz3, lz4));
            let y_slice = e_cd_y.get_fibre((ly3, ly4));
            let x_slice = e_cd_x.get_fibre((lx3, lx4));
            for (φ, e_φ) in z_slice.iter().take(φ_max).enumerate() {
                let e_φ = norm3 * e_φ;
                for (ν, e_ν) in y_slice.iter().take(ν_max).enumerate() {
                    let e_νφ = e_ν * e_φ;
                    let e_ket_t_slice = e_ket.get_t_slice_mut(τ_max, ν, φ, idx3, idx4);
                    for (e_τ_out, e_τ) in e_ket_t_slice.iter_mut().zip(x_slice.iter()) {
                        *e_τ_out = e_τ * e_νφ;
                    }
                }
            }
        }
    }
}


/// The Cartesian to spherical transform of a pair's ket coefficients `(tuv, c, d)`, by the pair's harmonic types: `d`
/// alone (`CS`), `c` alone (`SC`), or both by the combined matrix (`SS`).
pub enum KetHarmonicTransform<M> {
    /// `d` spherical, `c` Cartesian: `d`'s matrix.
    CS(M),
    /// `c` spherical, `d` Cartesian: `c`'s matrix.
    SC(M),
    /// Both spherical: `build_c2s_matrix(l_c, l_d)`.
    SS(M),
}

impl<'a> KetHarmonicTransform<MatrixViewMut<'a, f64>> {
    /// The transform of `pair`, its matrices from `arena`; `None` for a (Cartesian, Cartesian) pair.
    pub fn new_in<S: ShellKind>(pair: &ShellPair<S>, arena: &Arena<'a>) -> Result<Option<Self>, ArenaError> {
        use basis::shell_types::HarmonicType::{Cartesian, Spherical};
        let (lqn_c, lqn_d) = (pair.get_lqn_a(), pair.get_lqn_b());
        Ok(match (pair.get_harmonic_a(), pair.get_harmonic_b()) {
            (Cartesian, Cartesian) => None,
            (Cartesian, Spherical) => Some(KetHarmonicTransform::CS(cart2sph::compute_cartesian_to_spherical_matrix_arena(lqn_d, arena)?)),
            (Spherical, Cartesian) => Some(KetHarmonicTransform::SC(cart2sph::compute_cartesian_to_spherical_matrix_arena(lqn_c, arena)?)),
            (Spherical, Spherical) => Some(KetHarmonicTransform::SS(cart2sph::build_c2s_matrix_arena(lqn_c, lqn_d, arena)?)),
        })
    }
}

impl<M: MatrixTrait<f64> + Factor> KetHarmonicTransform<M> {
    /// `out` `(tuv, deg_c, deg_d)` from the Cartesian `cart` `(tuv, cart_c, cart_d)`.
    pub fn apply<A, B>(&self, cart: &A, out: &mut B)
    where
        A: Array3DTrait<f64>,
        B: Array3DTraitMut<f64>,
    {
        match self {
            // d is the outermost axis: (tuv * cart_c, cart_d) @ (cart_d, deg_d).
            KetHarmonicTransform::CS(d_trans) => {
                cart.view_as_matrix((2, 1)).matmul_prealloc(d_trans, &mut out.view_as_matrix_mut((2, 1)))
            }
            // c sits between tuv and d: one (tuv, cart_c) @ (cart_c, deg_c) per d slice.
            KetHarmonicTransform::SC(c_trans) => {
                for (mut out_mat, in_mat) in out.iter_mut().zip(cart.iter()) {
                    in_mat.matmul_prealloc(c_trans, &mut out_mat);
                }
            }
            // (tuv, cart_c * cart_d) @ (cart_c * cart_d, deg_c * deg_d).
            KetHarmonicTransform::SS(c2s) => {
                cart.view_as_matrix((1, 2)).matmul_prealloc(c2s, &mut out.view_as_matrix_mut((1, 2)))
            }
        }
    }
}

// The coefficients of `pair` for any angular momenta: the Cartesian coefficients per primitive pair, then the
// harmonic transform, or straight into the output for a (Cartesian, Cartesian) pair.
#[inline(always)]
fn ketside_general<S: ShellKind>(pair: &ShellPair<S>, scratch: &Arena<'_>) -> Result<Array4D<f64>, ArenaError> {
    let lqn_c = pair.get_lqn_a() as usize;
    let c_comps = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_c, scratch)?;
    let cart_c = c_comps.cols();
    let deg_c = pair.get_degeneracy_a();

    let lqn_d = pair.get_lqn_b() as usize;
    let d_comps = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_d, scratch)?;
    let cart_d = d_comps.cols();
    let deg_d = pair.get_degeneracy_b();

    let dist_cd = pair.get_separation();
    let tuv_size = hermite_count(lqn_c + lqn_d);
    let prim_dim = pair.get_products().len();
    let mut hermite_coefficients = HermiteCoefficients::from_arena(lqn_c, lqn_d, 0, scratch)?;
    let mut e_array = Array4D::<f64>::new(tuv_size, deg_c, deg_d, prim_dim);

    let transform = KetHarmonicTransform::new_in(pair, scratch)?;
    // Cartesian scratch, empty when there is no transform.
    let mut e_ket_cart = RefKetExpansionCoefficients::from_arena(if transform.is_some() { cart_c } else { 0 }, cart_d, lqn_c + lqn_d, scratch)?;
    for (mut e_ket, product) in e_array.iter_mut().zip(pair.get_products().iter()) {
        let c = product.get_a();
        let d = product.get_b();
        let q = product.get_p();
        let cd = product.get_μ();
        let cc = S::product_coefficient(product);
        match &transform {
            None => {
                let mut e_ket_direct = RefKetExpansionCoefficients::new(cart_c, cart_d, lqn_c + lqn_d, e_ket.get_data_mut());
                compute_e_ket_cartesian_prealloc::<Strategy<'_, S>, RefKetExpansionCoefficients<f64>, _, _>(
                    &mut e_ket_direct,
                    &mut hermite_coefficients,
                    c, d, q, cd, &dist_cd, &c_comps, &d_comps, cc,
                );
            }
            Some(transform) => {
                compute_e_ket_cartesian_prealloc::<Strategy<'_, S>, RefKetExpansionCoefficients<f64>, _, _>(
                    &mut e_ket_cart,
                    &mut hermite_coefficients,
                    c, d, q, cd, &dist_cd, &c_comps, &d_comps, cc,
                );
                transform.apply(&e_ket_cart.view_as_array3d(), &mut e_ket);
            }
        }
    }

    Ok(e_array)
}


/// Chosen derivative orders of computed coefficients presented as order 0, one per axis, so that the Cartesian
/// builder (which reads `get_e_x(0)` etc.) assembles `d^n E / dR_x^nx dR_y^ny dR_z^nz = E^(nx)_x E^(ny)_y E^(nz)_z`
/// without a copy.
struct AxisGenerations<'a, H: HermiteCoefficientsTrait> {
    coefficients: &'a H,
    generation: [usize; 3],
}

impl<H: HermiteCoefficientsTrait> HermiteCoefficientsTrait for AxisGenerations<'_, H> {
    fn get_e_x(&self, _order: usize) -> Array3DView<'_, f64> {
        self.coefficients.get_e_x(self.generation[0])
    }
    fn get_e_y(&self, _order: usize) -> Array3DView<'_, f64> {
        self.coefficients.get_e_y(self.generation[1])
    }
    fn get_e_z(&self, _order: usize) -> Array3DView<'_, f64> {
        self.coefficients.get_e_z(self.generation[2])
    }
    fn get_lqn_a(&self) -> usize {
        self.coefficients.get_lqn_a()
    }
    fn get_lqn_b(&self) -> usize {
        self.coefficients.get_lqn_b()
    }
}

/// The derivatives of order `order` of `pair`'s ket-side Hermite coefficients with respect to its separation
/// `R = B - A`, normalized and scaled by the contraction as `compute_ketside_hermite_coefficients`' (which they are
/// for `order = 0`), per primitive pair: `(tuv, c, d, variant, primitive pair)`, the variants
/// `d^order E / dR_x^nx dR_y^ny dR_z^nz` with `(nx, ny, nz)` in the library's Cartesian order of order `order`
/// (`math_utilities::cartesian::for_each_cartesian_component`: `x, y, z`; `xx, xy, xz, yy, yz, zz`; ...), the pair in
/// either shell order. The result is on the heap; the scratch comes from `scratch` and is freed on return.
pub fn compute_ketside_hermite_derivative_coefficients<S: ShellKind>(
    pair: &ShellPair<S>,
    order: usize,
    scratch: &mut Arena<'_>,
) -> Result<Array5D<f64>, ArenaError> {
    scratch.scope(|scratch| {
        let (lqn_c, lqn_d) = (pair.get_lqn_a(), pair.get_lqn_b());
        let lqn_cd = lqn_c + lqn_d;
        let c_comps = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_c, scratch)?;
        let d_comps = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_d, scratch)?;
        let (cart_c, cart_d) = (c_comps.cols(), d_comps.cols());
        let (deg_c, deg_d) = (pair.get_degeneracy_a(), pair.get_degeneracy_b());
        let dist_cd = pair.get_separation();
        let transform = KetHarmonicTransform::new_in(pair, scratch)?;

        let mut hermite_coefficients = HermiteCoefficients::from_arena(lqn_c, lqn_d, order, scratch)?;
        // The builder writes only the support of each component pair, the same for every variant: zeroed once.
        let mut e_cartesian = RefKetExpansionCoefficients::from_arena(if transform.is_some() { cart_c } else { 0 }, cart_d, lqn_cd, scratch)?;
        e_cartesian.get_data_mut().fill(0.0);
        let mut normalisation = MatrixViewMut::<f64>::from_arena(cart_c, cart_d, scratch)?;
        let mut e_array = Array5D::<f64>::new(hermite_count(lqn_cd), deg_c, deg_d, (order + 1) * (order + 2) / 2, pair.get_products().len());

        for (ip, product) in pair.get_products().iter().enumerate() {
            let (c, d, q, cd) = (product.get_a(), product.get_b(), product.get_p(), product.get_μ());
            let cc = S::product_coefficient(product);
            hermite_coefficients.compute(c, d, q, cd, &dist_cd);
            for generation in 1..=order {
                hermite_coefficients.compute_derivative(generation, c, d, q, cd, &dist_cd);
            }
            for (idx_d, lmn_d) in d_comps.iter_col().enumerate() {
                let norm_d = cart2sph::compute_cart_gaussian_norm(d, lmn_d[0], lmn_d[1], lmn_d[2]);
                for (idx_c, lmn_c) in c_comps.iter_col().enumerate() {
                    normalisation[(idx_c, idx_d)] = norm_d * cart2sph::compute_cart_gaussian_norm(c, lmn_c[0], lmn_c[1], lmn_c[2]);
                }
            }
            let mut variant = 0;
            for_each_cartesian_component(order, |nx, ny, nz| {
                let mut selected = AxisGenerations { coefficients: &hermite_coefficients, generation: [nx, ny, nz] };
                let mut e_ket = e_array.view_sub_array3d_mut((variant, ip));
                match &transform {
                    None => {
                        let mut e_ket_direct = RefKetExpansionCoefficients::new(cart_c, cart_d, lqn_cd, e_ket.get_data_mut());
                        compute_e_ket_cartesian_prealloc_w_norm::<Strategy<'_, S>, _, _, _, _>(
                            &mut e_ket_direct, &mut selected, &c_comps, &d_comps, &normalisation, cc,
                        );
                    }
                    Some(transform) => {
                        compute_e_ket_cartesian_prealloc_w_norm::<Strategy<'_, S>, _, _, _, _>(
                            &mut e_cartesian, &mut selected, &c_comps, &d_comps, &normalisation, cc,
                        );
                        transform.apply(&e_cartesian.view_as_array3d(), &mut e_ket);
                    }
                }
                variant += 1;
            });
        }
        Ok(e_array)
    })
}

#[cfg(test)]
mod general_ket_harmonic_tests {
    use super::*;
    use basis::contraction::{NoContraction, Uncontracted};
    use basis::shell_pairs::ShellPair;
    use basis::shell_types::HarmonicType;
    use basis::gto_products::GTOProduct;

    // A pair of one uncontracted product.
    fn make_pair(lqn_a: usize, lqn_b: usize, harmonic_a: HarmonicType, harmonic_b: HarmonicType) -> ShellPair<Uncontracted> {
        let product = GTOProduct::new(1.3, 0.8, [0.0, 0.0, 0.0]);
        ShellPair::<Uncontracted>::new(
            lqn_a, lqn_b, 1, 1, 1, 1, harmonic_a, harmonic_b,
            0, 0, 0, 0,
            0.15, -0.2, 0.35,
            false,
            NoContraction::default(),
            vec![product].into_boxed_slice(),
        )
    }

    /// The single product's `(tuv, components)` coefficient matrix.
    fn coefficients(lqn_c: usize, lqn_d: usize, harmonic_c: HarmonicType, harmonic_d: HarmonicType) -> Matrix<f64> {
        let mut scratch = arrays::scratch::Scratch::new(1, arrays::scratch::ByteUnit::MiB).unwrap();
        ketside_general(&make_pair(lqn_c, lqn_d, harmonic_c, harmonic_d), &scratch.get_arena()).unwrap().into_matrix((1, 3))
    }

    // The CC, CS, SC and SS branches of ketside_general against each other for a d-f pair: SS against CC transformed
    // by build_c2s_matrix, CS and SC transformed on their Cartesian axis by indexing.
    #[test]
    fn cc_cs_sc_and_ss_are_mutually_consistent_for_a_df_pair() {
        let lqn_c = 2; // d
        let lqn_d = 3; // f
        let cart_c = cart2sph::compute_cartesian_degeneracy(lqn_c);
        let cart_d = cart2sph::compute_cartesian_degeneracy(lqn_d);
        let deg_c = cart2sph::compute_spherical_degeneracy(lqn_c);
        let deg_d = cart2sph::compute_spherical_degeneracy(lqn_d);
        let tuv_size = hermite_count(lqn_c + lqn_d);

        let cc = coefficients(lqn_c, lqn_d, HarmonicType::Cartesian, HarmonicType::Cartesian);
        let cs = coefficients(lqn_c, lqn_d, HarmonicType::Cartesian, HarmonicType::Spherical);
        let sc = coefficients(lqn_c, lqn_d, HarmonicType::Spherical, HarmonicType::Cartesian);
        let ss = coefficients(lqn_c, lqn_d, HarmonicType::Spherical, HarmonicType::Spherical);

        assert_eq!((cc.rows(), cc.cols()), (tuv_size, cart_c * cart_d));
        assert_eq!((cs.rows(), cs.cols()), (tuv_size, cart_c * deg_d));
        assert_eq!((sc.rows(), sc.cols()), (tuv_size, deg_c * cart_d));
        assert_eq!((ss.rows(), ss.cols()), (tuv_size, deg_c * deg_d));

        let d_trans = cart2sph::compute_cartesian_to_spherical_matrix(lqn_d);
        let c_trans = cart2sph::compute_cartesian_to_spherical_matrix(lqn_c);
        let combined = cart2sph::build_c2s_matrix(lqn_c, lqn_d);

        let ss_from_cc = cc.matmul(&combined);
        for t in 0..tuv_size {
            for col in 0..deg_c * deg_d {
                assert!(
                    (ss_from_cc[(t, col)] - ss[(t, col)]).abs() < 1e-10,
                    "cc->ss mismatch at t={t} col={col}: {} vs {}", ss_from_cc[(t, col)], ss[(t, col)]
                );
            }
        }

        for t in 0..tuv_size {
            for c_out in 0..deg_c {
                for d_idx in 0..deg_d {
                    let mut acc = 0.0;
                    for c_in in 0..cart_c {
                        acc += cs[(t, c_in + d_idx * cart_c)] * c_trans[(c_in, c_out)];
                    }
                    let expected = ss[(t, c_out + d_idx * deg_c)];
                    assert!(
                        (acc - expected).abs() < 1e-10,
                        "cs->ss mismatch at t={t} c_out={c_out} d_idx={d_idx}: {acc} vs {expected}"
                    );
                }
            }
        }

        for t in 0..tuv_size {
            for c_idx in 0..deg_c {
                for d_out in 0..deg_d {
                    let mut acc = 0.0;
                    for d_in in 0..cart_d {
                        acc += sc[(t, c_idx + d_in * deg_c)] * d_trans[(d_in, d_out)];
                    }
                    let expected = ss[(t, c_idx + d_out * deg_c)];
                    assert!(
                        (acc - expected).abs() < 1e-10,
                        "sc->ss mismatch at t={t} c_idx={c_idx} d_out={d_out}: {acc} vs {expected}"
                    );
                }
            }
        }
    }
}


#[cfg(test)]
mod derivative_coefficient_tests {
    use super::*;
    use basis::contraction::{NoContraction, Uncontracted};
    use basis::gto_products::GTOProduct;
    use basis::shell_types::HarmonicType;

    // A pair of one uncontracted product at separation `r`.
    fn make_pair(lqn_c: usize, lqn_d: usize, harmonic: HarmonicType, r: [f64; 3]) -> ShellPair<Uncontracted> {
        let product = GTOProduct::new(1.3, 0.8, [0.0, 0.0, 0.0]);
        ShellPair::<Uncontracted>::new(
            lqn_c, lqn_d, 1, 1, 1, 1, harmonic, harmonic, 0, 0, 0, 0, r[0], r[1], r[2], false,
            NoContraction::default(), vec![product].into_boxed_slice(),
        )
    }

    fn derivatives(pair: &ShellPair<Uncontracted>, order: usize) -> Vec<f64> {
        let mut scratch = arrays::scratch::Scratch::new(1, arrays::scratch::ByteUnit::MiB).unwrap();
        compute_ketside_hermite_derivative_coefficients(pair, order, &mut scratch.get_arena()).unwrap().get_data().to_vec()
    }

    const R: [f64; 3] = [0.15, -0.2, 0.35];

    // Order 0 is compute_ketside_hermite_coefficients, always in the pair's harmonic basis.
    #[test]
    fn order_zero_matches_every_class_of_the_coefficients() {
        for harmonic in [HarmonicType::Spherical, HarmonicType::Cartesian] {
            for (lqn_c, lqn_d) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (0, 3), (2, 3)] {
                let pair = make_pair(lqn_c, lqn_d, harmonic, R);
                let mut scratch = arrays::scratch::Scratch::new(1, arrays::scratch::ByteUnit::MiB).unwrap();
                let expected = compute_ketside_hermite_coefficients(&pair, &mut scratch.get_arena()).unwrap().get_data().to_vec();
                let computed = derivatives(&pair, 0);
                assert_eq!(computed.len(), expected.len(), "({lqn_c}, {lqn_d}) {harmonic:?}");
                for (n, (x, y)) in computed.iter().zip(&expected).enumerate() {
                    assert!((x - y).abs() < 1e-13, "({lqn_c}, {lqn_d}) {harmonic:?} [{n}]: {x} vs {y}");
                }
            }
        }
    }

    // Order n + 1 against central differences of order n in the separation, n = 0, 1, the variants in the order of
    // for_each_cartesian_component; also at R = 0, where the first derivative does not vanish.
    #[test]
    fn orders_one_and_two_match_finite_differences() {
        let h = 1e-5;
        for r in [R, [0.0; 3]] {
            for (lqn_c, lqn_d) in [(0, 1), (1, 1), (0, 2), (2, 2), (1, 3)] {
                for order in 0..2 {
                    let pair = make_pair(lqn_c, lqn_d, HarmonicType::Spherical, r);
                    let higher = derivatives(&pair, order + 1);
                    let lower = derivatives(&pair, order);
                    let block = lower.len() / ((order + 1) * (order + 2) / 2);
                    let mut lower_variants = Vec::new();
                    for_each_cartesian_component(order, |nx, ny, nz| lower_variants.push([nx, ny, nz]));
                    let mut variant = 0;
                    let mut max_value = 0.0_f64;
                    for_each_cartesian_component(order + 1, |nx, ny, nz| {
                        let n = [nx, ny, nz];
                        // Any axis this variant differentiates, and the lower variant it is the derivative of.
                        let axis = (0..3).find(|&axis| n[axis] > 0).unwrap();
                        let mut base = n;
                        base[axis] -= 1;
                        let from = lower_variants.iter().position(|&m| m == base).unwrap();
                        let shifted = |step: f64| {
                            let mut moved = r;
                            moved[axis] += step;
                            derivatives(&make_pair(lqn_c, lqn_d, HarmonicType::Spherical, moved), order)
                        };
                        let (plus, minus) = (shifted(h), shifted(-h));
                        for s in 0..block {
                            let numeric = (plus[from * block + s] - minus[from * block + s]) / (2.0 * h);
                            let analytic = higher[variant * block + s];
                            max_value = max_value.max(analytic.abs());
                            assert!((numeric - analytic).abs() < 1e-8, "({lqn_c}, {lqn_d}) R {r:?} variant {n:?} [{s}]: {analytic} vs {numeric}");
                        }
                        variant += 1;
                    });
                    assert!(max_value > 1e-2, "({lqn_c}, {lqn_d}) order {}: vacuous", order + 1);
                }
            }
        }
    }
}
