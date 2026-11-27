// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The Hermite expansion coefficients `E^{ij}_t` of a primitive pair along each axis, and their derivatives with
//! respect to the separation.

use arrays::array3d::*;
use arrays::scratch::{Arena, ArenaError};

/// The Hermite expansion coefficients `E^{ij}_t` of a shell pair along x, y and z, for `i <= l1_max`, `j <= l2_max`,
/// and their derivatives with respect to the separation up to a fixed order: one `(t, i, j)` array per order.
pub struct HermiteCoefficientsStruct<M, G = Vec<M>>
where
    M: Array3DTraitMut<f64>,
    G: std::ops::DerefMut<Target = [M]>,
{
    l1_max: usize,
    l2_max: usize,
    e_x: G,
    e_y: G,
    e_z: G,
}
/// The coefficients in arena storage.
pub type HermiteCoefficients<'a> = HermiteCoefficientsStruct<Array3DViewMut<'a, f64>, &'a mut [Array3DViewMut<'a, f64>]>;
/// The coefficients in heap storage.
pub type HermiteCoefficientsOwned = HermiteCoefficientsStruct<Array3D<f64>>;

// One derivative order's array as a view.
#[inline(always)]
pub(crate) fn reborrow<M: Array3DTraitMut<f64>>(m: &M) -> Array3DView<'_, f64> {
    Array3DView::new(m.dim0(), m.dim1(), m.dim2(), m.get_data())
}

// One derivative order's array as a mutable view.
#[inline(always)]
pub(crate) fn reborrow_mut<M: Array3DTraitMut<f64>>(m: &mut M) -> Array3DViewMut<'_, f64> {
    Array3DViewMut::new(m.dim0(), m.dim1(), m.dim2(), m.get_data_mut())
}

/// Read access to the coefficients.
pub trait HermiteCoefficientsTrait {
    /// The x coefficients' `derivative_order`-th derivative, indexed `(t, i, j)`; likewise `get_e_y`, `get_e_z`.
    fn get_e_x(&self, derivative_order: usize) -> Array3DView<'_, f64>;
    fn get_e_y(&self, derivative_order: usize) -> Array3DView<'_, f64>;
    fn get_e_z(&self, derivative_order: usize) -> Array3DView<'_, f64>;
    /// The highest `i`.
    fn get_lqn_a(&self) -> usize;
    /// The highest `j`.
    fn get_lqn_b(&self) -> usize;
}

/// Computing the coefficients of the primitive pair with exponents `a`, `b`, `p = a + b`, `ab = a b / p`, and
/// separation `dist_ab = B - A`.
pub trait HermiteCoefficientsTraitMut: HermiteCoefficientsTrait {
    /// The coefficients (order 0).
    fn compute(&mut self, a: f64, b: f64, p: f64, ab: f64, dist_ab: &[f64; 3]);
    /// The first derivative; order 0 must be computed.
    fn compute_gradient(&mut self, a: f64, b: f64, p: f64, ab: f64, dist_ab: &[f64; 3]);
    /// The second derivative; orders 0 and 1 must be computed.
    fn compute_laplacian(&mut self, a: f64, b: f64, p: f64, ab: f64, dist_ab: &[f64; 3]);
    /// The `order`-th derivative with respect to the separation, `order >= 1`; orders `0..order` must be computed.
    fn compute_derivative(&mut self, order: usize, a: f64, b: f64, p: f64, ab: f64, dist_ab: &[f64; 3]);
}

impl HermiteCoefficientsOwned {
    /// Zeroed coefficients for `i <= l1_max`, `j <= l2_max` and derivative orders `0..=derivative_order`.
    #[inline(always)]
    pub fn new(l1_max: usize, l2_max: usize, derivative_order: usize) -> Self {
        let e_x = allocate_hermite_coefficient_generations(l1_max, l2_max, derivative_order);
        let e_y = allocate_hermite_coefficient_generations(l1_max, l2_max, derivative_order);
        let e_z = allocate_hermite_coefficient_generations(l1_max, l2_max, derivative_order);
        Self {
            l1_max,
            l2_max,
            e_x,
            e_y,
            e_z,
        }
    }
}

impl<'a> HermiteCoefficients<'a> {
    /// `HermiteCoefficientsOwned::new` in arena storage.
    #[inline(always)]
    pub fn from_arena(l1_max: usize, l2_max: usize, derivative_order: usize, arena: &Arena<'a>) -> Result<Self, ArenaError> {
        let e_x = allocate_hermite_coefficient_generations_arena(l1_max, l2_max, derivative_order, arena)?;
        let e_y = allocate_hermite_coefficient_generations_arena(l1_max, l2_max, derivative_order, arena)?;
        let e_z = allocate_hermite_coefficient_generations_arena(l1_max, l2_max, derivative_order, arena)?;
        Ok(Self {
            l1_max,
            l2_max,
            e_x,
            e_y,
            e_z,
        })
    }
}

impl<M, G> HermiteCoefficientsTrait for HermiteCoefficientsStruct<M, G>
where
    M: Array3DTraitMut<f64>,
    G: std::ops::DerefMut<Target = [M]>,
{
    #[inline(always)]
    fn get_e_x(&self, derivative_order: usize) -> Array3DView<'_, f64> {
        debug_assert!(derivative_order < self.e_x.len());
        reborrow(&self.e_x[derivative_order])
    }

    #[inline(always)]
    fn get_e_y(&self, derivative_order: usize) -> Array3DView<'_, f64> {
        debug_assert!(derivative_order < self.e_y.len());
        reborrow(&self.e_y[derivative_order])
    }

    #[inline(always)]
    fn get_e_z(&self, derivative_order: usize) -> Array3DView<'_, f64> {
        debug_assert!(derivative_order < self.e_z.len());
        reborrow(&self.e_z[derivative_order])
    }

    #[inline(always)]
    fn get_lqn_a(&self) -> usize {
        self.l1_max
    }

    #[inline(always)]
    fn get_lqn_b(&self) -> usize {
        self.l2_max
    }
}

impl<M, G> HermiteCoefficientsTraitMut for HermiteCoefficientsStruct<M, G>
where
    M: Array3DTraitMut<f64>,
    G: std::ops::DerefMut<Target = [M]>,
{
    #[inline(always)]
    fn compute(&mut self, a: f64, b: f64, p: f64, ab: f64, dist_ab: &[f64; 3]) {
        let HermiteCoefficientsStruct {
            l1_max,
            l2_max,
            e_x,
            e_y,
            e_z,
        } = self;
        let xpa = (b / p) * dist_ab[0];
        let xpb = -(a / p) * dist_ab[0];
        let kab = (-ab * dist_ab[0] * dist_ab[0]).exp();
        compute_hermite_coefficients_prealloc(
            &mut reborrow_mut(&mut e_x[0]),
            *l1_max,
            *l2_max,
            xpa,
            xpb,
            p,
            kab,
        );
        let xpa = (b / p) * dist_ab[1];
        let xpb = -(a / p) * dist_ab[1];
        let kab = (-ab * dist_ab[1] * dist_ab[1]).exp();
        compute_hermite_coefficients_prealloc(
            &mut reborrow_mut(&mut e_y[0]),
            *l1_max,
            *l2_max,
            xpa,
            xpb,
            p,
            kab,
        );
        let xpa = (b / p) * dist_ab[2];
        let xpb = -(a / p) * dist_ab[2];
        let kab = (-ab * dist_ab[2] * dist_ab[2]).exp();
        compute_hermite_coefficients_prealloc(
            &mut reborrow_mut(&mut e_z[0]),
            *l1_max,
            *l2_max,
            xpa,
            xpb,
            p,
            kab,
        );
    }

    #[inline(always)]
    fn compute_gradient(&mut self, a: f64, b: f64, p: f64, ab: f64, dist_ab: &[f64; 3]) {
        self.compute_derivative(1, a, b, p, ab, dist_ab);
    }

    #[inline(always)]
    fn compute_laplacian(&mut self, a: f64, b: f64, p: f64, ab: f64, dist_ab: &[f64; 3]) {
        self.compute_derivative(2, a, b, p, ab, dist_ab);
    }

    #[inline(always)]
    fn compute_derivative(
        &mut self,
        order: usize,
        a: f64,
        b: f64,
        p: f64,
        _ab: f64,
        dist_ab: &[f64; 3],
    ) {
        let HermiteCoefficientsStruct {
            l1_max,
            l2_max,
            e_x,
            e_y,
            e_z,
        } = self;
        debug_assert!(order >= 1 && order < e_x.len());
        compute_derivative_axis(order, e_x, *l1_max, *l2_max, a, b, p, dist_ab[0]);
        compute_derivative_axis(order, e_y, *l1_max, *l2_max, a, b, p, dist_ab[1]);
        compute_derivative_axis(order, e_z, *l1_max, *l2_max, a, b, p, dist_ab[2]);
    }
}

// The `order`-th derivative along one axis from orders `order - 1` and `order - 2`.
#[inline(always)]
fn compute_derivative_axis<M: Array3DTraitMut<f64>>(
    order: usize,
    e: &mut [M],
    l1_max: usize,
    l2_max: usize,
    a: f64,
    b: f64,
    p: f64,
    rx: f64,
) {
    let (lower, upper) = e.split_at_mut(order);
    let mut e_out = reborrow_mut(&mut upper[0]);
    let e_prev1 = reborrow(&lower[order - 1]);
    let e_prev2 = if order >= 2 { Some(reborrow(&lower[order - 2])) } else { None };
    compute_e_derivative(order, &mut e_out, &e_prev1, e_prev2.as_ref(), l1_max, l2_max, a, b, p, rx);
}

// One zeroed `(t, i, j)` array per derivative order.
#[inline(always)]
fn allocate_hermite_coefficient_generations(l1_max: usize, l2_max: usize, derivative_order: usize) -> Vec<Array3D<f64>> {
    let t_max = l1_max + l2_max;
    let dim1 = t_max + 1;
    let dim2 = l1_max + 1;
    let dim3 = l2_max + 1;
    (0..=derivative_order).map(|_| Array3D::<f64>::new(dim1, dim2, dim3)).collect()
}

// One zeroed `(t, i, j)` array per derivative order, as views of one arena block; the views are in the arena too.
#[inline(always)]
pub(crate) fn allocate_hermite_coefficient_generations_arena<'a>(l1_max: usize, l2_max: usize, derivative_order: usize, arena: &Arena<'a>) -> Result<&'a mut [Array3DViewMut<'a, f64>], ArenaError> {
    let t_max = l1_max + l2_max;
    let dim1 = t_max + 1;
    let dim2 = l1_max + 1;
    let dim3 = l2_max + 1;
    let block_len = arrays::scratch::checked_product(&[dim1, dim2, dim3])?;
    let n_generations = derivative_order + 1;
    let total = arrays::scratch::checked_product(&[block_len, n_generations])?;
    let data = arena.alloc::<f64>(total)?;
    data.fill(0.0);

    let mut remaining = Some(data);
    arena.alloc_with(n_generations, |_| {
        let (chunk, rest) = remaining.take().expect("one chunk per generation").split_at_mut(block_len);
        remaining = Some(rest);
        Array3DViewMut::<f64>::new(dim1, dim2, dim3, chunk)
    })
}

/// The coefficients `E^{ij}_t` along one axis into `e_buffer` (`(t, i, j)`) by the McMurchie-Davidson recursion
/// (L. E. McMurchie and E. R. Davidson, J. Comput. Phys. 26, 218 (1978)), from `E^{00}_0 = kab`.
#[inline(always)]
pub fn compute_hermite_coefficients_prealloc(
    e_buffer: &mut Array3DViewMut<f64>,
    l1_max: usize,
    l2_max: usize,
    xpa: f64,
    xpb: f64,
    p: f64,
    kab: f64,
) {
    e_buffer[(0, 0, 0)] = kab;
    let two_p_inv = 1.0 / (2.0 * p);
    update_l1_terms(e_buffer, l1_max, xpa, two_p_inv);
    update_l2_terms(e_buffer, l2_max, xpb, two_p_inv);
    update_cross_terms(e_buffer, l1_max, l2_max, xpb, two_p_inv);
}

/// The n-th derivative with respect to the separation R of the coefficients along one axis, n >= 1, from orders
/// n - 1 and n - 2. R enters the recursion through X_PA = (b/p) R, X_PB = -(a/p) R and the Gaussian
/// exp(-mu R^2) of E^{00}_0, mu = ab/p, so only two terms of each Leibniz sum remain:
///   E^{(n)}(0,0,0)   = -2 mu [ R E^{(n-1)}(0,0,0) + (n-1) E^{(n-2)}(0,0,0) ]
///   E^{(n)}_0(i,0)   = (b/p) [ R E^{(n)}_0(i-1,0) + n E^{(n-1)}_0(i-1,0) ] + E^{(n)}_1(i-1,0)
///   E^{(n)}_t(i,0)   = i/(2 p t) E^{(n)}_{t-1}(i-1,0)     (t >= 1)
/// and likewise for the j and the cross terms. `e_prev2` is `None` for n = 1, where its factor n - 1 is zero.
#[inline(always)]
fn compute_e_derivative(
    order: usize,
    mut e_out: &mut Array3DViewMut<f64>,
    e_prev1: &Array3DView<f64>,
    e_prev2: Option<&Array3DView<f64>>,
    l1_max: usize,
    l2_max: usize,
    a: f64,
    b: f64,
    p: f64,
    rx: f64,
) {
    debug_assert!(order >= 1);
    let order_f64 = order as f64;
    let prev2_000 = e_prev2.map_or(0.0, |e| e[(0, 0, 0)]);
    e_out[(0, 0, 0)] = (-2.0 * a * b / p) * (rx * e_prev1[(0, 0, 0)] + (order_f64 - 1.0) * prev2_000);
    update_l1_terms_grad(&mut e_out, e_prev1, l1_max, b, p, rx, order_f64);
    update_l2_terms_grad(&mut e_out, e_prev1, l2_max, a, p, rx, order_f64);
    update_cross_terms_grad(&mut e_out, e_prev1, l1_max, l2_max, a, p, rx, order_f64);
}

/// `E^{i0}_t` for `i = 1..=l1_max` from `E^{00}_0`.
#[inline(always)]
pub(crate) fn update_l1_terms(
    e: &mut Array3DViewMut<f64>,
    l1_max: usize,
    xpa: f64,
    two_p_inv: f64
) {
    for i in 1..=l1_max {
        let i_f64 = i as f64;
        e[(0, i, 0)] = xpa * e[(0, i - 1, 0)] + e[(1, i - 1, 0)];
        for t in 1..=i {
            let t_f64 = t as f64;
            e[(t, i, 0)] = two_p_inv * i_f64 * e[(t - 1, i - 1, 0)] / t_f64;
        }
    }
}

/// `E^{0j}_t` for `j = 1..=l2_max` from `E^{00}_0`.
#[inline(always)]
pub(crate) fn update_l2_terms(
    e: &mut Array3DViewMut<f64>,
    l2_max: usize,
    xpb: f64,
    two_p_inv: f64
) {
    for i in 1..=l2_max {
        let i_f64 = i as f64;
        e[(0, 0, i)] = xpb * e[(0, 0, i - 1)] + e[(1, 0, i - 1)];
        for t in 1..=i {
            let t_f64 = t as f64;
            e[(t, 0, i)] = two_p_inv * i_f64 * e[(t - 1, 0, i - 1)] / t_f64;
        }
    }
}

/// `E^{ij}_t` for `i, j >= 1` from the `i` and `j` terms.
#[inline(always)]
pub(crate) fn update_cross_terms(
    e: &mut Array3DViewMut<f64>,
    l1_max: usize,
    l2_max: usize,
    xpb: f64,
    two_p_inv: f64,
) {
    for i in 1..=l1_max {
        let i_f64 = i as f64;
        for j in 1..=l2_max {
            let j_f64 = j as f64;
            e[(0, i, j)] = xpb * e[(0, i, j - 1)] + e[(1, i, j - 1)];
            for t in 1..=(i + j) {
                let t_f64 = t as f64;
                e[(t, i, j)] = two_p_inv
                    * (i_f64 * e[(t - 1, i - 1, j)] 
                    + j_f64 * e[(t - 1, i, j - 1)])
                    / t_f64;
            }
        }
    }
}
/// `update_l1_terms` for the `order`-th derivative, `e` the derivative of order `order - 1`.
#[inline(always)]
pub(crate) fn update_l1_terms_grad(
    e_gradient: &mut Array3DViewMut<f64>,
    e: &Array3DView<f64>,
    l1_max: usize,
    b: f64,
    p: f64,
    rx: f64,
    order: f64,
) {
    for i in 1..=l1_max {
        let i_f64 = i as f64;
        e_gradient[(0, i, 0)] = (b / p)
            * (rx * e_gradient[(0, i - 1, 0)] + order * e[(0, i - 1, 0)])
            + e_gradient[(1, i - 1, 0)];
        for t in 1..=i {
            let t_f64 = t as f64;
            e_gradient[(t, i, 0)] = i_f64 * e_gradient[(t - 1, i - 1, 0)] / (2.0 * p * t_f64);
        }
    }
}

/// `update_l2_terms` for the `order`-th derivative, `e` the derivative of order `order - 1`.
#[inline(always)]
pub(crate) fn update_l2_terms_grad(
    e_gradient: &mut Array3DViewMut<f64>,
    e: &Array3DView<f64>,
    l2_max: usize,
    a: f64,
    p: f64,
    rx: f64,
    order: f64,
) {
    for i in 1..=l2_max {
        let i_f64 = i as f64;
        e_gradient[(0, 0, i)] = -(a / p)
            * (rx * e_gradient[(0, 0, i - 1)] + order * e[(0, 0, i - 1)])
            + e_gradient[(1, 0, i - 1)];
        for t in 1..=i {
            let t_f64 = t as f64;
            e_gradient[(t, 0, i)] = i_f64 * e_gradient[(t - 1, 0, i - 1)] / (2.0 * p * t_f64);
        }
    }
}

/// `update_cross_terms` for the `order`-th derivative, `e` the derivative of order `order - 1`.
#[inline(always)]
pub(crate) fn update_cross_terms_grad(
    e_gradient: &mut Array3DViewMut<f64>,
    e: &Array3DView<f64>,
    l1_max: usize,
    l2_max: usize,
    a: f64,
    p: f64,
    rx: f64,
    order: f64,
) {
    for i in 1..=l1_max {
        let i_f64 = i as f64;
        for j in 1..=l2_max {
            let j_f64 = j as f64;
            e_gradient[(0, i, j)] = -(a / p)
                * (rx * e_gradient[(0, i, j - 1)] + order * e[(0, i, j - 1)])
                + e_gradient[(1, i, j - 1)];
            for t in 1..=(i + j) {
                let t_f64 = t as f64;
                e_gradient[(t, i, j)] = (i_f64 * e_gradient[(t - 1, i - 1, j)]
                    + j_f64 * e_gradient[(t - 1, i, j - 1)])
                    / (2.0 * p * t_f64);
            }
        }
    }
}