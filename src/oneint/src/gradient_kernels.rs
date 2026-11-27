// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use mmd::hermite_r_tensor::{HermiteRTensorTrait, HermiteRTensorTraitMut, HermiteTuvSliceViewMut};
use basis::charge_carriers::PointChargeTrait;
use mmd::hermite_coefficients::{HermiteCoefficientsTrait, HermiteCoefficientsTraitMut};

use crate::kernels::{BuildRtuv3D, PointMode};

// Derivative integrals with respect to the basis centres and the operator's sources. Notation: P the product
// centre, R = B - A the separation of the basis centres. The centre derivatives (T. Helgaker and P. R. Taylor,
// Theor. Chim. Acta 83, 177 (1992); `citations.rs`: `HELGAKER_TAYLOR_1992`, with the sign of d/dR for R = B - A):
//  d/dA_x = (a/p) (d/dP_x) - (d/dR_x)
//  d/dB_x = (b/p) (d/dP_x) + (d/dR_x)

#[derive(Clone, Copy)]
// Which terms a gradient kernel forms: the derivative with respect to the basis centre A, B, or both (the atom
// carries both shells), and to the source at position `grad_op` in the source list. The terms are additive.
pub(crate) struct GradientArguments {
    pub(crate) grad_a: bool,
    pub(crate) grad_b: bool,
    pub(crate) grad_op: Option<usize>,
    /// Return BOTH centre combinations from one evaluation, as six planes: d/dA (planes 0-2, the (true, false) call)
    /// and d/dP (planes 3-5, the (true, true) call: d/dR cancelled, a/p + b/p = 1). The kernels form d/dR and d/dP in
    /// the same loop anyway, so the second combination is a byproduct, and d/dB = d/dP - d/dA follows after
    /// contraction. Only for kernels with sources (d/dP = 0 for overlap and kinetic) and only together with
    /// `grad_a && !grad_b && grad_op.is_none()`.
    pub(crate) resolved: bool,
}

impl GradientArguments {
    /// The given terms, three planes.
    pub fn new(grad_a: bool, grad_b: bool, grad_op: Option<usize>) -> Self {
        Self { grad_a, grad_b, grad_op, resolved: false }
    }

    /// d/dA and d/dP of the basis-function terms together (six planes), no source term.
    pub fn new_resolved() -> Self {
        Self { grad_a: true, grad_b: false, grad_op: None, resolved: true }
    }
}

// kernels::OneElectronIntegralKernel for derivative integrals: `compute` also takes the `GradientArguments`.
pub(crate) trait OneElectronIntegralGradientKernel {
    type AdditionalArguments<'c>: crate::pieces::SiteRuns where Self: 'c;
    /// Whether the derivative integrals are a sum over the operator's sites (and so may be computed by runs of sites
    /// and added).
    const SUMS_OVER_SITES: bool = true;
    type KernelBatch<'b>;
    type KernelBuffer<'b>;
    fn compute<'b, 'k, 'c, Musize>(
        cartesian_batch: &mut Self::KernelBatch<'b>,
        kernel_buffer: &mut Self::KernelBuffer<'k>,
        a: f64,
        b: f64,
        p: f64,
        ab: f64,
        cart_a_comps: &Musize,
        cart_b_comps: &Musize,
        r_ab: &[f64; 3],
        p_ab: &[f64; 3],
        additional_arguments: &Self::AdditionalArguments<'c>,
        gradient_arguments: &GradientArguments,
        weight: f64,
    )
    where
        Musize: arrays::matrix::MatrixTrait<usize>;

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError>;

}

// Gradient kernels' batches are stacks of planes: d/dx, d/dy, d/dz of the selected terms (six in the resolved mode).
pub(crate) trait ArrayBatchGradientKernel:
    for<'b> OneElectronIntegralGradientKernel<KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>>
{
    fn n_planes<'c>(additional_arguments: &Self::AdditionalArguments<'c>, gradient_arguments: &GradientArguments) -> usize;
}

// The basis-centre term from d/dR and d/dP: d/dA, d/dB, or, for both together (an atom carrying both shells),
// their sum d/dP (the d/dR terms cancel and a/p + b/p = 1).
#[inline(always)]
pub(crate) fn centre_gradient_term(
    grad_a: bool,
    grad_b: bool,
    ddr: [f64; 3],
    ddp: [f64; 3],
    a_div_p: f64,
    b_div_p: f64,
) -> [f64; 3] {
    match (grad_a, grad_b) {
        (false, false) => [0.0, 0.0, 0.0],
        (true, true) => ddp,
        (true, false) => [
            a_div_p * ddp[0] - ddr[0],
            a_div_p * ddp[1] - ddr[1],
            a_div_p * ddp[2] - ddr[2],
        ],
        (false, true) => [
            b_div_p * ddp[0] + ddr[0],
            b_div_p * ddp[1] + ddr[1],
            b_div_p * ddp[2] + ddr[2],
        ],
    }
}

/// The derivatives of the overlap with respect to the basis centres (d/dP = 0).
pub struct OverlapGradient;
impl OneElectronIntegralGradientKernel for OverlapGradient {
    type AdditionalArguments<'c> = crate::additional_arguments::EmptyAdditionalArguments;
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = crate::kernels::OverlapBuffer<'b>;
    fn compute<'b, 'k, 'c, Musize>(
        cartesian_batch: &mut Self::KernelBatch<'b>,
        kernel_buffer: &mut Self::KernelBuffer<'k>,
        a: f64,
        b: f64,
        p: f64,
        ab: f64,
        cart_a_comps: &Musize,
        cart_b_comps: &Musize,
        r_ab: &[f64; 3],
        _p_ab: &[f64; 3],
        _additional_arguments: &Self::AdditionalArguments<'c>,
        gradient_arguments: &GradientArguments,
        weight: f64,
    ) where
        Musize: arrays::matrix::MatrixTrait<usize>
    {
        let crate::kernels::OverlapBuffer {
            hermite_coefficients,
        } = kernel_buffer;

        let coeff = weight * crate::kernels::compute_oneint_coeff(p);
        hermite_coefficients.compute(a, b, p, ab, r_ab);
        hermite_coefficients.compute_gradient(a, b, p, ab, r_ab);
        let e_x = hermite_coefficients.get_e_x(0);
        let e_y = hermite_coefficients.get_e_y(0);
        let e_z = hermite_coefficients.get_e_z(0);
        let f_x = hermite_coefficients.get_e_x(1);
        let f_y = hermite_coefficients.get_e_y(1);
        let f_z = hermite_coefficients.get_e_z(1);

        let a_div_p = a / p;
        let b_div_p = b / p;
        // No operator: d/dP = 0 and grad_op is never set.
        let ddp = [0.0, 0.0, 0.0];

        for (j, lmn_b) in cart_b_comps.iter_col().enumerate() {
            let lx2 = lmn_b[0];
            let ly2 = lmn_b[1];
            let lz2 = lmn_b[2];
            let norm2 = coeff * cart2sph::compute_cart_gaussian_norm(b, lx2, ly2, lz2);
            for (i, lmn_a) in cart_a_comps.iter_col().enumerate() {
                let lx1 = lmn_a[0];
                let ly1 = lmn_a[1];
                let lz1 = lmn_a[2];
                let norm1 = norm2 * cart2sph::compute_cart_gaussian_norm(a, lx1, ly1, lz1);
                let ddr = [
                    f_x[(0, lx1, lx2)] * e_y[(0, ly1, ly2)] * e_z[(0, lz1, lz2)],
                    e_x[(0, lx1, lx2)] * f_y[(0, ly1, ly2)] * e_z[(0, lz1, lz2)],
                    e_x[(0, lx1, lx2)] * e_y[(0, ly1, ly2)] * f_z[(0, lz1, lz2)],
                ];
                let grad = centre_gradient_term(
                    gradient_arguments.grad_a,
                    gradient_arguments.grad_b,
                    ddr,
                    ddp,
                    a_div_p,
                    b_div_p,
                );
                cartesian_batch[(i, j, 0)] += norm1 * grad[0];
                cartesian_batch[(i, j, 1)] += norm1 * grad[1];
                cartesian_batch[(i, j, 2)] += norm1 * grad[2];
            }
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        _additional_arguments: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError>
    {
        let derivative_order = 1;
        let hermite_coefficients =
            mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
        Ok(crate::kernels::OverlapBuffer{
            hermite_coefficients,
        })
    }

}

impl ArrayBatchGradientKernel for OverlapGradient {
    fn n_planes<'c>(_additional_arguments: &Self::AdditionalArguments<'c>, gradient_arguments: &GradientArguments) -> usize {
        debug_assert!(!gradient_arguments.resolved, "d/dP is identically zero here: no resolved mode");
        3
    }
}

/// The derivatives of the kinetic energy with respect to the basis centres. With T_ab of `KineticKernel` and the
/// Hermite coefficients' derivatives E, F, G, H of orders 0 to 3 along each axis:
///   d/dR_x T_raw = H_x E_y E_z + F_x G_y E_z + F_x E_y G_z
///   d/dR_y T_raw = G_x F_y E_z + E_x H_y E_z + E_x F_y G_z
///   d/dR_z T_raw = G_x E_y F_z + E_x G_y F_z + E_x E_y H_z
/// (T_raw without the -1/2, which is in the normalization). No operator: d/dP = 0 and grad_op is never set.
pub struct KineticGradient;
impl OneElectronIntegralGradientKernel for KineticGradient {
    type AdditionalArguments<'c> = crate::additional_arguments::EmptyAdditionalArguments;
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = crate::kernels::OverlapBuffer<'b>;
    fn compute<'b, 'k, 'c, Musize>(
        cartesian_batch: &mut Self::KernelBatch<'b>,
        kernel_buffer: &mut Self::KernelBuffer<'k>,
        a: f64,
        b: f64,
        p: f64,
        ab: f64,
        cart_a_comps: &Musize,
        cart_b_comps: &Musize,
        r_ab: &[f64; 3],
        _p_ab: &[f64; 3],
        _additional_arguments: &Self::AdditionalArguments<'c>,
        gradient_arguments: &GradientArguments,
        weight: f64,
    ) where
        Musize: arrays::matrix::MatrixTrait<usize>
    {
        let crate::kernels::OverlapBuffer {
            hermite_coefficients,
        } = kernel_buffer;

        let coeff = weight * crate::kernels::compute_oneint_coeff(p);
        hermite_coefficients.compute(a, b, p, ab, r_ab);
        hermite_coefficients.compute_gradient(a, b, p, ab, r_ab);
        hermite_coefficients.compute_laplacian(a, b, p, ab, r_ab);
        hermite_coefficients.compute_derivative(3, a, b, p, ab, r_ab);

        let e_x = hermite_coefficients.get_e_x(0);
        let e_y = hermite_coefficients.get_e_y(0);
        let e_z = hermite_coefficients.get_e_z(0);
        let f_x = hermite_coefficients.get_e_x(1);
        let f_y = hermite_coefficients.get_e_y(1);
        let f_z = hermite_coefficients.get_e_z(1);
        let g_x = hermite_coefficients.get_e_x(2);
        let g_y = hermite_coefficients.get_e_y(2);
        let g_z = hermite_coefficients.get_e_z(2);
        let h_x = hermite_coefficients.get_e_x(3);
        let h_y = hermite_coefficients.get_e_y(3);
        let h_z = hermite_coefficients.get_e_z(3);

        let a_div_p = a / p;
        let b_div_p = b / p;
        let ddp = [0.0, 0.0, 0.0];

        for (j, lmn_b) in cart_b_comps.iter_col().enumerate() {
            let lx2 = lmn_b[0];
            let ly2 = lmn_b[1];
            let lz2 = lmn_b[2];
            let norm2 = -0.5 * coeff * cart2sph::compute_cart_gaussian_norm(b, lx2, ly2, lz2);
            for (i, lmn_a) in cart_a_comps.iter_col().enumerate() {
                let lx1 = lmn_a[0];
                let ly1 = lmn_a[1];
                let lz1 = lmn_a[2];
                let norm1 = norm2 * cart2sph::compute_cart_gaussian_norm(a, lx1, ly1, lz1);

                let e0x = e_x[(0, lx1, lx2)];
                let e0y = e_y[(0, ly1, ly2)];
                let e0z = e_z[(0, lz1, lz2)];
                let f0x = f_x[(0, lx1, lx2)];
                let f0y = f_y[(0, ly1, ly2)];
                let f0z = f_z[(0, lz1, lz2)];
                let g0x = g_x[(0, lx1, lx2)];
                let g0y = g_y[(0, ly1, ly2)];
                let g0z = g_z[(0, lz1, lz2)];
                let h0x = h_x[(0, lx1, lx2)];
                let h0y = h_y[(0, ly1, ly2)];
                let h0z = h_z[(0, lz1, lz2)];

                let ddr = [
                    h0x * e0y * e0z + f0x * g0y * e0z + f0x * e0y * g0z,
                    g0x * f0y * e0z + e0x * h0y * e0z + e0x * f0y * g0z,
                    g0x * e0y * f0z + e0x * g0y * f0z + e0x * e0y * h0z,
                ];
                let grad = centre_gradient_term(
                    gradient_arguments.grad_a,
                    gradient_arguments.grad_b,
                    ddr,
                    ddp,
                    a_div_p,
                    b_div_p,
                );
                cartesian_batch[(i, j, 0)] += norm1 * grad[0];
                cartesian_batch[(i, j, 1)] += norm1 * grad[1];
                cartesian_batch[(i, j, 2)] += norm1 * grad[2];
            }
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        _additional_arguments: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError>
    {
        let derivative_order = 3;
        let hermite_coefficients =
            mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
        Ok(crate::kernels::OverlapBuffer{
            hermite_coefficients,
        })
    }

}

impl ArrayBatchGradientKernel for KineticGradient {
    fn n_planes<'c>(_additional_arguments: &Self::AdditionalArguments<'c>, gradient_arguments: &GradientArguments) -> usize {
        debug_assert!(!gradient_arguments.resolved, "d/dP is identically zero here: no resolved mode");
        3
    }
}

/// The derivatives of CoulombKernel's potential, < A | -Sum_C q_C g(|r - C|) | B > (point charges, Gaussian
/// charges or point multipoles by `M`), as `GradientArguments` selects:
///   - grad_a / grad_b: the dependence on the basis centres, from R_tuv summed over every source;
///   - grad_op = Some(idx): the dependence on source idx's position, from that source's R_tuv alone, by
///     d/dC_x = -d/dP_x (R_tuv depends on C only through P - C).
pub(crate) struct CoulombGradientKernel<T, M = PointMode>(std::marker::PhantomData<(T, M)>);

// CoulombGradientKernel's scratch.
pub(crate) struct CoulombGradientBuffer<'a> {
    hermite_coefficients: mmd::hermite_coefficients::HermiteCoefficients<'a>,
    r_buffer: mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D<'a>,
    r_tuv: HermiteTuvSliceViewMut<'a, f64>,
    r_tuv_op: HermiteTuvSliceViewMut<'a, f64>,
}

impl<T, M> OneElectronIntegralGradientKernel for CoulombGradientKernel<T, M>
where
    T: PointChargeTrait,
    M: BuildRtuv3D<T> + 'static,
{
    type AdditionalArguments<'c> = &'c [T] where T: 'c;
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = CoulombGradientBuffer<'b>;
    fn compute<'b, 'k, 'c, Musize>(
        cartesian_batch: &mut Self::KernelBatch<'b>,
        kernel_buffer: &mut Self::KernelBuffer<'k>,
        a: f64,
        b: f64,
        p: f64,
        ab: f64,
        cart_a_comps: &Musize,
        cart_b_comps: &Musize,
        dist_ab: &[f64; 3],
        pos_ab: &[f64; 3],
        additional_arguments: &Self::AdditionalArguments<'c>,
        gradient_arguments: &GradientArguments,
        weight: f64,
    )
    where
        Musize: arrays::matrix::MatrixTrait<usize>
    {
        let CoulombGradientBuffer {
            hermite_coefficients,
            r_buffer,
            r_tuv,
            r_tuv_op,
        } = kernel_buffer;

        // The summed R_tuv and the order-1 Hermite coefficients F only for a basis-centre term.
        let need_centre_term = gradient_arguments.grad_a || gradient_arguments.grad_b;
        // The d/dR terms only survive when exactly one centre is differentiated: for both together they cancel
        // (centre_gradient_term's (true, true) case), so neither the order-1 coefficients nor ddr are needed.
        let need_ddr = gradient_arguments.grad_a != gradient_arguments.grad_b;

        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        if need_ddr {
            hermite_coefficients.compute_gradient(a, b, p, ab, dist_ab);
        }

        let e_x = hermite_coefficients.get_e_x(0);
        let e_y = hermite_coefficients.get_e_y(0);
        let e_z = hermite_coefficients.get_e_z(0);
        let f_x = hermite_coefficients.get_e_x(1);
        let f_y = hermite_coefficients.get_e_y(1);
        let f_z = hermite_coefficients.get_e_z(1);

        if need_centre_term {
            r_tuv.get_data_mut().fill(0.0);
            M::build(r_tuv, r_buffer, p, pos_ab, additional_arguments);
        }
        // The single source's R_tuv, for the operator term.
        if let Some(idx) = gradient_arguments.grad_op {
            r_tuv_op.get_data_mut().fill(0.0);
            M::build(r_tuv_op, r_buffer, p, pos_ab, &additional_arguments[idx..=idx]);
        }

        let a_div_p = a / p;
        let b_div_p = b / p;

        // hermite_index(t,u,v,n) = hermite_index(0,u,v,n) + t: the per-(u,v) base is computed once, outside the t loop.
        let lqn_ab = mmd::hermite_expansion::hermite_order(r_tuv.get_data().len());
        let r_tuv_data = r_tuv.get_data();
        let r_tuv_op_data = r_tuv_op.get_data();

        for (j, lmn_b) in cart_b_comps.iter_col().enumerate() {
            let lx2 = lmn_b[0];
            let ly2 = lmn_b[1];
            let lz2 = lmn_b[2];
            let norm2 = weight * cart2sph::compute_cart_gaussian_norm(b, lx2, ly2, lz2);
            for (i, lmn_a) in cart_a_comps.iter_col().enumerate() {
                let lx1 = lmn_a[0];
                let ly1 = lmn_a[1];
                let lz1 = lmn_a[2];
                let norm1 = norm2 * cart2sph::compute_cart_gaussian_norm(a, lx1, ly1, lz1);
                let v_max = lz1 + lz2 + 1;
                let u_max = ly1 + ly2 + 1;
                let t_max = lx1 + lx2 + 1;

                let mut grad = [0.0, 0.0, 0.0];
                let mut grad_p = [0.0, 0.0, 0.0]; // d/dP, resolved (six-plane) mode only

                if need_centre_term {
                    let mut ddr = [0.0, 0.0, 0.0];
                    let mut ddp = [0.0, 0.0, 0.0];
                    for v in 0..v_max {
                        let e_v = e_z[(v, lz1, lz2)];
                        for u in 0..u_max {
                            let e_u = e_y[(u, ly1, ly2)];
                            let base_00 = mmd::hermite_expansion::hermite_index(0, u, v, lqn_ab);
                            let base_u1 = mmd::hermite_expansion::hermite_index(0, u + 1, v, lqn_ab);
                            let base_v1 = mmd::hermite_expansion::hermite_index(0, u, v + 1, lqn_ab);
                            for t in 0..t_max {
                                let e_t = e_x[(t, lx1, lx2)];
                                let r_tuv_val = r_tuv_data[base_00 + t];
                                if need_ddr {
                                    ddr[0] += f_x[(t, lx1, lx2)] * e_u * e_v * r_tuv_val;
                                    ddr[1] += e_t * f_y[(u, ly1, ly2)] * e_v * r_tuv_val;
                                    ddr[2] += e_t * e_u * f_z[(v, lz1, lz2)] * r_tuv_val;
                                }
                                ddp[0] += e_t * e_u * e_v * r_tuv_data[base_00 + t + 1];
                                ddp[1] += e_t * e_u * e_v * r_tuv_data[base_u1 + t];
                                ddp[2] += e_t * e_u * e_v * r_tuv_data[base_v1 + t];
                            }
                        }
                    }
                    let centre = centre_gradient_term(
                        gradient_arguments.grad_a,
                        gradient_arguments.grad_b,
                        ddr,
                        ddp,
                        a_div_p,
                        b_div_p,
                    );
                    grad[0] += centre[0];
                    grad[1] += centre[1];
                    grad[2] += centre[2];
                    if gradient_arguments.resolved {
                        // Same ddr and ddp, the other combination: both centres together, d/dR cancelled.
                        grad_p = centre_gradient_term(true, true, ddr, ddp, a_div_p, b_div_p);
                    }
                }

                if gradient_arguments.grad_op.is_some() {
                    let mut ddp_op = [0.0, 0.0, 0.0];
                    for v in 0..v_max {
                        let e_v = e_z[(v, lz1, lz2)];
                        for u in 0..u_max {
                            let e_u = e_y[(u, ly1, ly2)];
                            let base_00 = mmd::hermite_expansion::hermite_index(0, u, v, lqn_ab);
                            let base_u1 = mmd::hermite_expansion::hermite_index(0, u + 1, v, lqn_ab);
                            let base_v1 = mmd::hermite_expansion::hermite_index(0, u, v + 1, lqn_ab);
                            for t in 0..t_max {
                                let e_t = e_x[(t, lx1, lx2)];
                                ddp_op[0] += e_t * e_u * e_v * r_tuv_op_data[base_00 + t + 1];
                                ddp_op[1] += e_t * e_u * e_v * r_tuv_op_data[base_u1 + t];
                                ddp_op[2] += e_t * e_u * e_v * r_tuv_op_data[base_v1 + t];
                            }
                        }
                    }
                    // d/dC = -d/dP for this charge's own (unsummed) contribution.
                    grad[0] -= ddp_op[0];
                    grad[1] -= ddp_op[1];
                    grad[2] -= ddp_op[2];
                }

                cartesian_batch[(i, j, 0)] += norm1 * grad[0];
                cartesian_batch[(i, j, 1)] += norm1 * grad[1];
                cartesian_batch[(i, j, 2)] += norm1 * grad[2];
                if gradient_arguments.resolved {
                    cartesian_batch[(i, j, 3)] += norm1 * grad_p[0];
                    cartesian_batch[(i, j, 4)] += norm1 * grad_p[1];
                    cartesian_batch[(i, j, 5)] += norm1 * grad_p[2];
                }
            }
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        let derivative_order = 1;
        // R_tuv one order past lqn_a + lqn_b for the d/dP and d/dC shifts; a multipole source's extra_headroom only in
        // r_buffer.
        let lqn_ab = lqn_a + lqn_b + 1;
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
        let r_buffer = mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D::from_arena(lqn_ab + M::extra_headroom(additional_arguments), arena)?;
        let r_tuv = HermiteTuvSliceViewMut::from_arena(lqn_ab, arena)?;
        let r_tuv_op = HermiteTuvSliceViewMut::from_arena(lqn_ab, arena)?;
        Ok(CoulombGradientBuffer {
            hermite_coefficients,
            r_buffer,
            r_tuv,
            r_tuv_op,
        })
    }

}

impl<T, M> ArrayBatchGradientKernel for CoulombGradientKernel<T, M>
where
    T: PointChargeTrait,
    M: BuildRtuv3D<T> + 'static,
{
    fn n_planes<'c>(_additional_arguments: &Self::AdditionalArguments<'c>, gradient_arguments: &GradientArguments) -> usize {
        if gradient_arguments.resolved { 6 } else { 3 }
    }
}
