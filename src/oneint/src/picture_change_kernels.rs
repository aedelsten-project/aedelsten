// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Picture-change kernels: p.Vp and the full sigma.p V sigma.p for point charges, finite charges and point
// multipoles, and their moment counterparts p.mp and sigma.p m sigma.p.

use arrays;
use arrays::matrix::{MatrixTrait, MatrixTraitMut};
use arrays::array3d::Array3DTraitMut;
use mmd::hermite_r_tensor::{HermiteRTensorTrait, HermiteRTensorTraitMut};
use basis::charge_carriers::PointChargeTrait;
use mmd::hermite_coefficients::{HermiteCoefficientsTrait, HermiteCoefficientsTraitMut};
use mmd::hermite_moment::HermiteMoments;

use crate::kernels::*;

/// p.Vp, <A| p.V p |B> = nabla_A . nabla_B <A|V|B>, for the potential V of CoulombKernel (point charges, Gaussian
/// charges or point multipoles by `M`), as the mixed second derivative along each axis
///   (d^2/dA_x dB_x) = (a b)/p^2 (d^2/dP_x^2) + (a - b)/p (d^2/dP_x dR_x) - (d^2/dR_x^2),
/// P the product centre, R = B - A; the P derivatives are shifts of the R tensor R^{tuv}(P - C), the R derivatives
/// the Hermite coefficients' (T. Helgaker and P. R. Taylor, Theor. Chim. Acta 83, 177 (1992)).
pub(crate) struct PvpKernel<T, M = PointMode>(std::marker::PhantomData<(T, M)>);

// PvpKernel's and PvpFullKernel's scratch.
pub(crate) struct PvpBuffer<'a> {
    hermite_coefficients: mmd::hermite_coefficients::HermiteCoefficients<'a> ,
    r_buffer: mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D<'a>,
    r_tuv: mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'a, f64>,
}

impl<T, M> OneElectronIntegralKernel for PvpKernel<T, M>
where
    T: PointChargeTrait,
    M: BuildRtuv3D<T> + 'static,
{
    type AdditionalArguments<'c> = &'c [T] where T: 'c;
    type KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>;
    type KernelBuffer<'b> = PvpBuffer<'b>;
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
        weight: f64,
    )
    where
        Musize: MatrixTrait<usize>
    {
        let PvpBuffer {
            hermite_coefficients,
            r_buffer,
            r_tuv,
        } = kernel_buffer;

        // Set info about common centre.
        r_tuv.get_data_mut().fill(0.0);
        M::build(
            r_tuv,
            r_buffer,
            p,
            pos_ab,
            additional_arguments,
        );
        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_gradient(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_laplacian(a, b, p, ab, dist_ab);

        let e_ab_x = hermite_coefficients.get_e_x(0);
        let e_ab_y = hermite_coefficients.get_e_y(0);
        let e_ab_z = hermite_coefficients.get_e_z(0);
        let f_ab_x = hermite_coefficients.get_e_x(1);
        let f_ab_y = hermite_coefficients.get_e_y(1);
        let f_ab_z = hermite_coefficients.get_e_z(1);
        let g_ab_x = hermite_coefficients.get_e_x(2);
        let g_ab_y = hermite_coefficients.get_e_y(2);
        let g_ab_z = hermite_coefficients.get_e_z(2);

        let t1 = (a*b) / p.powi(2);
        let t2 = (a-b) / p;
        let r = r_tuv;
        // hermite_index(t,u,v,n) = hermite_index(0,u,v,n) + t: the per-(u,v) base is computed once, outside the t loop.
        let lqn_ab = mmd::hermite_expansion::hermite_order(r.get_data().len());
        let r_data = r.get_data();
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

                let mut ddp_ddr = 0.0;
                let mut d2dr2 = 0.0;
                let mut d2dp2 = 0.0;

                for v in 0..v_max {
                    let e_v = e_ab_z[(v, lz1, lz2)];
                    let f_v = f_ab_z[(v, lz1, lz2)];
                    let g_v = g_ab_z[(v, lz1, lz2)];
                    for u in 0..u_max {
                        let e_u = e_ab_y[(u, ly1, ly2)];
                        let f_u = f_ab_y[(u, ly1, ly2)];
                        let g_u = g_ab_y[(u, ly1, ly2)];
                        let base_00 = mmd::hermite_expansion::hermite_index(0, u, v, lqn_ab);
                        let base_u1 = mmd::hermite_expansion::hermite_index(0, u + 1, v, lqn_ab);
                        let base_v1 = mmd::hermite_expansion::hermite_index(0, u, v + 1, lqn_ab);
                        let base_u2 = mmd::hermite_expansion::hermite_index(0, u + 2, v, lqn_ab);
                        let base_v2 = mmd::hermite_expansion::hermite_index(0, u, v + 2, lqn_ab);
                        for t in 0..t_max {
                            let e_t = e_ab_x[(t, lx1, lx2)];
                            let f_t = f_ab_x[(t, lx1, lx2)];
                            let g_t = g_ab_x[(t, lx1, lx2)];
                            let r_000 = r_data[base_00 + t];
                            let pr_x = f_t * e_u * e_v * r_data[base_00 + t + 1];
                            let pr_y = e_t * f_u * e_v * r_data[base_u1 + t];
                            let pr_z = e_t * e_u * f_v * r_data[base_v1 + t];
                            ddp_ddr += pr_x + pr_y + pr_z;
                            let rr_x = g_t * e_u * e_v * r_000;
                            let rr_y = e_t * g_u * e_v * r_000;
                            let rr_z = e_t * e_u * g_v * r_000;
                            d2dr2 += rr_x + rr_y + rr_z;
                            let pp_x = e_t * e_u * e_v * r_data[base_00 + t + 2];
                            let pp_y = e_t * e_u * e_v * r_data[base_u2 + t];
                            let pp_z = e_t * e_u * e_v * r_data[base_v2 + t];
                            d2dp2 += pp_x + pp_y + pp_z;
                        }
                    }
                }
                cartesian_batch[(i, j)] += norm1 * (t1 * d2dp2 + t2 * ddp_ddr - d2dr2);
            }
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        // The R tensor one order past lqn_a + lqn_b, for the mixed second derivatives.
        let lqn_a = lqn_a + 1;
        let lqn_b = lqn_b + 1;
        let lqn_ab = lqn_a + lqn_b;
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, 2, arena)?;
        // A multipole source reads its R-tensor `extra_headroom` orders past what the operator itself needs (see
        // BuildRtuv3D); 0 for point and finite charges.
        let r_buffer = mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D::from_arena(lqn_ab + M::extra_headroom(additional_arguments), arena)?;
        let r_tuv = mmd::hermite_r_tensor::HermiteTuvSliceViewMut::from_arena(lqn_ab, arena)?;
        Ok(PvpBuffer {
            hermite_coefficients,
            r_buffer,
            r_tuv,
        })
    }

}

/// The four real components of sigma.p V sigma.p = p.Vp + i sigma.(p x Vp) in one pass: planes 0, 1, 2 the
/// antisymmetric (p x Vp)_x,y,z, plane 3 the scalar p.Vp (as PvpKernel); the factor i and the spinor assembly are
/// left to the caller. By the same mixed derivatives as PvpKernel, for the cross directions,
///   (p x Vp)_z = sum_tuv e_v^z [ e_t^x f_u^y r(t+1,u,v) - f_t^x e_u^y r(t,u+1,v) ]
/// and cyclically for x and y.
pub(crate) struct PvpFullKernel<T, M = PointMode>(std::marker::PhantomData<(T, M)>);

impl<T, M> OneElectronIntegralKernel for PvpFullKernel<T, M>
where
    T: PointChargeTrait,
    M: BuildRtuv3D<T> + 'static,
{
    type AdditionalArguments<'c> = &'c [T] where T: 'c;
    // Component-last: component k occupies one contiguous cart_a*cart_b block.
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = PvpBuffer<'b>;

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
        weight: f64,
    )
    where
        Musize: MatrixTrait<usize>
    {
        let PvpBuffer {
            hermite_coefficients,
            r_buffer,
            r_tuv,
        } = kernel_buffer;

        r_tuv.get_data_mut().fill(0.0);
        M::build(
            r_tuv,
            r_buffer,
            p,
            pos_ab,
            additional_arguments,
        );
        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_gradient(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_laplacian(a, b, p, ab, dist_ab); // needed for the scalar (G) term

        let e_ab_x = hermite_coefficients.get_e_x(0);
        let e_ab_y = hermite_coefficients.get_e_y(0);
        let e_ab_z = hermite_coefficients.get_e_z(0);
        let f_ab_x = hermite_coefficients.get_e_x(1);
        let f_ab_y = hermite_coefficients.get_e_y(1);
        let f_ab_z = hermite_coefficients.get_e_z(1);
        let g_ab_x = hermite_coefficients.get_e_x(2);
        let g_ab_y = hermite_coefficients.get_e_y(2);
        let g_ab_z = hermite_coefficients.get_e_z(2);

        let t1 = (a*b) / p.powi(2);
        let t2 = (a-b) / p;
        let r = r_tuv;
        // The per-(u,v) base of hermite_index computed once, outside the t loop.
        let lqn_ab = mmd::hermite_expansion::hermite_order(r.get_data().len());
        let r_data = r.get_data();
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

                let mut cross_x = 0.0; // (p x Vp)_x: active (y,z), spectator x
                let mut cross_y = 0.0; // (p x Vp)_y: active (z,x), spectator y
                let mut cross_z = 0.0; // (p x Vp)_z: active (x,y), spectator z
                // The scalar p.Vp, as PvpKernel.
                let mut ddp_ddr = 0.0;
                let mut d2dr2 = 0.0;
                let mut d2dp2 = 0.0;

                for v in 0..v_max {
                    let e_v = e_ab_z[(v, lz1, lz2)];
                    let f_v = f_ab_z[(v, lz1, lz2)];
                    let g_v = g_ab_z[(v, lz1, lz2)];
                    for u in 0..u_max {
                        let e_u = e_ab_y[(u, ly1, ly2)];
                        let f_u = f_ab_y[(u, ly1, ly2)];
                        let g_u = g_ab_y[(u, ly1, ly2)];
                        let base_00 = mmd::hermite_expansion::hermite_index(0, u, v, lqn_ab);
                        let base_u1 = mmd::hermite_expansion::hermite_index(0, u + 1, v, lqn_ab);
                        let base_v1 = mmd::hermite_expansion::hermite_index(0, u, v + 1, lqn_ab);
                        let base_u2 = mmd::hermite_expansion::hermite_index(0, u + 2, v, lqn_ab);
                        let base_v2 = mmd::hermite_expansion::hermite_index(0, u, v + 2, lqn_ab);
                        for t in 0..t_max {
                            let e_t = e_ab_x[(t, lx1, lx2)];
                            let f_t = f_ab_x[(t, lx1, lx2)];
                            let g_t = g_ab_x[(t, lx1, lx2)];

                            let r_000 = r_data[base_00 + t];
                            let r_t1_00 = r_data[base_00 + t + 1];
                            let r_0_u1_0 = r_data[base_u1 + t];
                            let r_0_0_v1 = r_data[base_v1 + t];

                            cross_z += e_v * (e_t * f_u * r_t1_00 - f_t * e_u * r_0_u1_0);
                            cross_x += e_t * (e_u * f_v * r_0_u1_0 - f_u * e_v * r_0_0_v1);
                            cross_y += e_u * (e_v * f_t * r_0_0_v1 - f_v * e_t * r_t1_00);

                            let pr_x = f_t * e_u * e_v * r_t1_00;
                            let pr_y = e_t * f_u * e_v * r_0_u1_0;
                            let pr_z = e_t * e_u * f_v * r_0_0_v1;
                            ddp_ddr += pr_x + pr_y + pr_z;
                            let rr_x = g_t * e_u * e_v * r_000;
                            let rr_y = e_t * g_u * e_v * r_000;
                            let rr_z = e_t * e_u * g_v * r_000;
                            d2dr2 += rr_x + rr_y + rr_z;
                            let pp_x = e_t * e_u * e_v * r_data[base_00 + t + 2];
                            let pp_y = e_t * e_u * e_v * r_data[base_u2 + t];
                            let pp_z = e_t * e_u * e_v * r_data[base_v2 + t];
                            d2dp2 += pp_x + pp_y + pp_z;
                        }
                    }
                }
                cartesian_batch[(i, j, 0)] += norm1 * cross_x;
                cartesian_batch[(i, j, 1)] += norm1 * cross_y;
                cartesian_batch[(i, j, 2)] += norm1 * cross_z;
                cartesian_batch[(i, j, 3)] += norm1 * (t1 * d2dp2 + t2 * ddp_ddr - d2dr2);
            }
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        let lqn_a = lqn_a + 1;
        let lqn_b = lqn_b + 1;
        let lqn_ab = lqn_a + lqn_b;
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, 2, arena)?;
        // A multipole source reads its R-tensor `extra_headroom` orders past what the operator itself needs (see
        // BuildRtuv3D); 0 for point and finite charges.
        let r_buffer = mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D::from_arena(lqn_ab + M::extra_headroom(additional_arguments), arena)?;
        let r_tuv = mmd::hermite_r_tensor::HermiteTuvSliceViewMut::from_arena(lqn_ab, arena)?;
        Ok(PvpBuffer {
            hermite_coefficients,
            r_buffer,
            r_tuv,
        })
    }

}

impl<T, M> ArrayBatchKernel for PvpFullKernel<T, M>
where
    T: PointChargeTrait,
    M: BuildRtuv3D<T> + 'static,
{
    fn n_planes<'c>(_additional_arguments: &Self::AdditionalArguments<'c>) -> usize {
        4
    }

    // Planes 0, 1, 2 are (p x Vp)_x,y,z, antisymmetric matrices; plane 3 is p.Vp, symmetric.
    fn antisymmetric_plane(plane: usize) -> bool {
        plane < 3
    }
}

/// p.mp, <A| p . (x-Ox)^lx(y-Oy)^ly(z-Oz)^lz p |B>, for every Cartesian component of one order about one origin:
/// PvpKernel's mixed derivatives with the moment's Hermite moments in place of the R tensor.
pub struct MomentPXPKernel;
impl OneElectronIntegralKernel for MomentPXPKernel {
    type AdditionalArguments<'c> = &'c crate::additional_arguments::MomentOrigin;
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = MomentBuffer<'b>;

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
        weight: f64,
    )
    where
        Musize: MatrixTrait<usize>,
    {
        let MomentBuffer {
            hermite_coefficients,
            hermite_moments,
        } = kernel_buffer;

        let coeff = weight * compute_oneint_coeff(p);
        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_gradient(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_laplacian(a, b, p, ab, dist_ab);

        let order = hermite_moments.get_order();
        let origin = additional_arguments.get_origin();
        hermite_moments.compute_hermite_moments(
            pos_ab[0] - origin[0],
            pos_ab[1] - origin[1],
            pos_ab[2] - origin[2],
            p,
        );

        let mut xyz = 0;
        math_utilities::cartesian::for_each_cartesian_component(order, |lx, ly, lz| {
            general_pxp::<Musize>(
                &mut cartesian_batch.view_sub_matrix_mut(xyz),
                hermite_coefficients,
                hermite_moments,
                cart_a_comps,
                cart_b_comps,
                a,
                b,
                p,
                coeff,
                lx,
                ly,
                lz,
            );
            xyz += 1;
        });
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        let derivative_order = 2;
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
        let order = additional_arguments.get_order();
        let hermite_moments = HermiteMoments::from_arena(order, derivative_order, arena)?;
        Ok(MomentBuffer {
            hermite_coefficients,
            hermite_moments,
        })
    }

}

impl ArrayBatchKernel for MomentPXPKernel {
    fn n_planes<'c>(additional_arguments: &Self::AdditionalArguments<'c>) -> usize {
        moment_component_count(additional_arguments.get_order())
    }
}

/// sigma.p m sigma.p for every Cartesian component of one order about one origin: pieces 0, 1, 2 the (p x mp)_x,y,z,
/// piece 3 the scalar p.mp, in plane k = 4 * component + piece.
pub struct MomentFullKernel;
impl OneElectronIntegralKernel for MomentFullKernel {
    type AdditionalArguments<'c> = &'c crate::additional_arguments::MomentOrigin;
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = MomentBuffer<'b>;

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
        weight: f64,
    )
    where
        Musize: MatrixTrait<usize>,
    {
        let MomentBuffer {
            hermite_coefficients,
            hermite_moments,
        } = kernel_buffer;

        let coeff = weight * compute_oneint_coeff(p);
        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_gradient(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_laplacian(a, b, p, ab, dist_ab);

        let order = hermite_moments.get_order();
        let origin = additional_arguments.get_origin();
        hermite_moments.compute_hermite_moments(
            pos_ab[0] - origin[0],
            pos_ab[1] - origin[1],
            pos_ab[2] - origin[2],
            p,
        );

        let mut xyz = 0;
        math_utilities::cartesian::for_each_cartesian_component(order, |lx, ly, lz| {
            general_pxp_full::<Musize>(
                cartesian_batch,
                xyz,
                hermite_coefficients,
                hermite_moments,
                cart_a_comps,
                cart_b_comps,
                a,
                b,
                p,
                coeff,
                lx,
                ly,
                lz,
            );
            xyz += 1;
        });
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        let derivative_order = 2;
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
        let order = additional_arguments.get_order();
        let hermite_moments = HermiteMoments::from_arena(order, derivative_order, arena)?;
        Ok(MomentBuffer {
            hermite_coefficients,
            hermite_moments,
        })
    }

}

impl ArrayBatchKernel for MomentFullKernel {
    fn n_planes<'c>(additional_arguments: &Self::AdditionalArguments<'c>) -> usize {
        4 * moment_component_count(additional_arguments.get_order())
    }

    // Plane k = 4 * component + piece: pieces 0, 1, 2 are the vector (p x mp) terms, antisymmetric;
    // piece 3 is the scalar term, symmetric.
    fn antisymmetric_plane(plane: usize) -> bool {
        plane % 4 != 3
    }
}

// p.mp of one moment component: PvpKernel's mixed derivatives per axis, against the Hermite moments mm_x, mm_y,
// mm_z, which do not couple across axes.
#[inline(always)]
fn general_pxp<M: MatrixTrait<usize>>(
    oneint_batch: &mut arrays::matrix::MatrixViewMut<f64>,
    hermite_coefficients: &mmd::hermite_coefficients::HermiteCoefficients<'_>,
    hermite_moments: &HermiteMoments<'_>,
    cart_a_comps: &M,
    cart_b_comps: &M,
    a: f64,
    b: f64,
    p: f64,
    coeff: f64,
    order_x: usize,
    order_y: usize,
    order_z: usize
) {
    let e_ab_x = hermite_coefficients.get_e_x(0);
    let e_ab_y = hermite_coefficients.get_e_y(0);
    let e_ab_z = hermite_coefficients.get_e_z(0);
    let f_ab_x = hermite_coefficients.get_e_x(1);
    let f_ab_y = hermite_coefficients.get_e_y(1);
    let f_ab_z = hermite_coefficients.get_e_z(1);
    let g_ab_x = hermite_coefficients.get_e_x(2);
    let g_ab_y = hermite_coefficients.get_e_y(2);
    let g_ab_z = hermite_coefficients.get_e_z(2);
    let mm_x = hermite_moments.get_mm_x();
    let mm_y = hermite_moments.get_mm_y();
    let mm_z = hermite_moments.get_mm_z();

    let t1 = (a*b) / p.powi(2);
    let t2 = (a-b) / p;
    for (xb, lmn_b) in oneint_batch.iter_col_mut().zip(cart_b_comps.iter_col()) {
        let lx2 = lmn_b[0];
        let ly2 = lmn_b[1];
        let lz2 = lmn_b[2];
        let norm2 = coeff * cart2sph::compute_cart_gaussian_norm(b, lx2, ly2, lz2);
        for (xa, lmn_a) in xb.iter_mut().zip(cart_a_comps.iter_col()) {
            let lx1 = lmn_a[0];
            let ly1 = lmn_a[1];
            let lz1 = lmn_a[2];
            let norm1 = norm2 * cart2sph::compute_cart_gaussian_norm(a, lx1, ly1, lz1);

            let t_max = order_x.min(lx1 + lx2);
            let u_max = order_y.min(ly1 + ly2);
            let v_max = order_z.min(lz1 + lz2);

            let mut sx = 0.0;
            let mut d2dr2_x = 0.0;
            let mut d2dpdr_x = 0.0;
            let mut d2dp2_x = 0.0;
            for t in 0..=t_max {
                sx       += e_ab_x[(t, lx1, lx2)] * mm_x[(t, order_x)];
                d2dr2_x  += g_ab_x[(t, lx1, lx2)] * mm_x[(t, order_x)];
                d2dpdr_x += f_ab_x[(t, lx1, lx2)] * mm_x[(t+1, order_x)];
                d2dp2_x  += e_ab_x[(t, lx1, lx2)] * mm_x[(t+2, order_x)];
            }

            let mut sy = 0.0;
            let mut d2dr2_y = 0.0;
            let mut d2dpdr_y = 0.0;
            let mut d2dp2_y = 0.0;
            for u in 0..=u_max {
                sy       += e_ab_y[(u, ly1, ly2)] * mm_y[(u, order_y)];
                d2dr2_y  += g_ab_y[(u, ly1, ly2)] * mm_y[(u, order_y)];
                d2dpdr_y += f_ab_y[(u, ly1, ly2)] * mm_y[(u+1, order_y)];
                d2dp2_y  += e_ab_y[(u, ly1, ly2)] * mm_y[(u+2, order_y)];
            }

            let mut sz = 0.0;
            let mut d2dr2_z = 0.0;
            let mut d2dpdr_z = 0.0;
            let mut d2dp2_z = 0.0;
            for v in 0..=v_max {
                sz       += e_ab_z[(v, lz1, lz2)] * mm_z[(v, order_z)];
                d2dr2_z  += g_ab_z[(v, lz1, lz2)] * mm_z[(v, order_z)];
                d2dpdr_z += f_ab_z[(v, lz1, lz2)] * mm_z[(v+1, order_z)];
                d2dp2_z  += e_ab_z[(v, lz1, lz2)] * mm_z[(v+2, order_z)];
            }

            let mx = (t1 * d2dp2_x + t2 * d2dpdr_x - d2dr2_x) * sy * sz;
            let my = sx * (t1 * d2dp2_y + t2 * d2dpdr_y - d2dr2_y) * sz;
            let mz = sx * sy * (t1 * d2dp2_z + t2 * d2dpdr_z - d2dr2_z);

            *xa += norm1 * (mx + my + mz);
        }
    }
}

// sigma.p m sigma.p of one moment component: the scalar term as general_pxp, and with
//   S0_a = sum_t e_t^a mm_a(t,order_a)   (= sx/sy/sz below)
//   S1_a = sum_t e_t^a mm_a(t+1,order_a)
//   G1_a = sum_t f_t^a mm_a(t,order_a)
// the vector terms
//   (p x mp)_z = S0_z * (S1_x*G1_y - G1_x*S1_y)
// and cyclically for x and y.
#[inline(always)]
fn general_pxp_full<M: MatrixTrait<usize>>(
    cartesian_batch: &mut arrays::array3d::Array3DViewMut<f64>,
    xyz: usize,
    hermite_coefficients: &mmd::hermite_coefficients::HermiteCoefficients<'_>,
    hermite_moments: &HermiteMoments<'_>,
    cart_a_comps: &M,
    cart_b_comps: &M,
    a: f64,
    b: f64,
    p: f64,
    coeff: f64,
    order_x: usize,
    order_y: usize,
    order_z: usize
) {
    let e_ab_x = hermite_coefficients.get_e_x(0);
    let e_ab_y = hermite_coefficients.get_e_y(0);
    let e_ab_z = hermite_coefficients.get_e_z(0);
    let f_ab_x = hermite_coefficients.get_e_x(1);
    let f_ab_y = hermite_coefficients.get_e_y(1);
    let f_ab_z = hermite_coefficients.get_e_z(1);
    let g_ab_x = hermite_coefficients.get_e_x(2);
    let g_ab_y = hermite_coefficients.get_e_y(2);
    let g_ab_z = hermite_coefficients.get_e_z(2);
    let mm_x = hermite_moments.get_mm_x();
    let mm_y = hermite_moments.get_mm_y();
    let mm_z = hermite_moments.get_mm_z();

    let t1 = (a*b) / p.powi(2);
    let t2 = (a-b) / p;
    let base_k = 4 * xyz;

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

            let t_max = order_x.min(lx1 + lx2);
            let u_max = order_y.min(ly1 + ly2);
            let v_max = order_z.min(lz1 + lz2);

            let mut sx = 0.0;
            let mut d2dr2_x = 0.0;
            let mut d2dpdr_x = 0.0;
            let mut d2dp2_x = 0.0;
            let mut s1_x = 0.0;
            let mut g1_x = 0.0;
            for t in 0..=t_max {
                let e = e_ab_x[(t, lx1, lx2)];
                let f = f_ab_x[(t, lx1, lx2)];
                let g = g_ab_x[(t, lx1, lx2)];
                sx       += e * mm_x[(t, order_x)];
                d2dr2_x  += g * mm_x[(t, order_x)];
                d2dpdr_x += f * mm_x[(t+1, order_x)];
                d2dp2_x  += e * mm_x[(t+2, order_x)];
                s1_x     += e * mm_x[(t+1, order_x)];
                g1_x     += f * mm_x[(t, order_x)];
            }

            let mut sy = 0.0;
            let mut d2dr2_y = 0.0;
            let mut d2dpdr_y = 0.0;
            let mut d2dp2_y = 0.0;
            let mut s1_y = 0.0;
            let mut g1_y = 0.0;
            for u in 0..=u_max {
                let e = e_ab_y[(u, ly1, ly2)];
                let f = f_ab_y[(u, ly1, ly2)];
                let g = g_ab_y[(u, ly1, ly2)];
                sy       += e * mm_y[(u, order_y)];
                d2dr2_y  += g * mm_y[(u, order_y)];
                d2dpdr_y += f * mm_y[(u+1, order_y)];
                d2dp2_y  += e * mm_y[(u+2, order_y)];
                s1_y     += e * mm_y[(u+1, order_y)];
                g1_y     += f * mm_y[(u, order_y)];
            }

            let mut sz = 0.0;
            let mut d2dr2_z = 0.0;
            let mut d2dpdr_z = 0.0;
            let mut d2dp2_z = 0.0;
            let mut s1_z = 0.0;
            let mut g1_z = 0.0;
            for v in 0..=v_max {
                let e = e_ab_z[(v, lz1, lz2)];
                let f = f_ab_z[(v, lz1, lz2)];
                let g = g_ab_z[(v, lz1, lz2)];
                sz       += e * mm_z[(v, order_z)];
                d2dr2_z  += g * mm_z[(v, order_z)];
                d2dpdr_z += f * mm_z[(v+1, order_z)];
                d2dp2_z  += e * mm_z[(v+2, order_z)];
                s1_z     += e * mm_z[(v+1, order_z)];
                g1_z     += f * mm_z[(v, order_z)];
            }

            let mx = (t1 * d2dp2_x + t2 * d2dpdr_x - d2dr2_x) * sy * sz;
            let my = sx * (t1 * d2dp2_y + t2 * d2dpdr_y - d2dr2_y) * sz;
            let mz = sx * sy * (t1 * d2dp2_z + t2 * d2dpdr_z - d2dr2_z);
            let scalar = norm1 * (mx + my + mz);

            let cross_x = norm1 * sx * (s1_y * g1_z - g1_y * s1_z);
            let cross_y = norm1 * sy * (s1_z * g1_x - g1_z * s1_x);
            let cross_z = norm1 * sz * (s1_x * g1_y - g1_x * s1_y);

            cartesian_batch[(i, j, base_k)]     += cross_x;
            cartesian_batch[(i, j, base_k + 1)] += cross_y;
            cartesian_batch[(i, j, base_k + 2)] += cross_z;
            cartesian_batch[(i, j, base_k + 3)] += scalar;
        }
    }
}

/// The derivatives of p.Vp (PvpKernel) with respect to the basis centres and the potential's sources. With
/// alpha = a/p, beta = b/p and, per axis k, M_k = alpha beta d^2/dP_k^2 + (alpha - beta) d^2/dP_k dR_k - d^2/dR_k^2
/// (PvpKernel's p.Vp = sum_k M_k V), the centres as A = P - beta R, B = P + alpha R give
///   d/dA_x = alpha d/dP_x - d/dR_x,   d/dB_x = beta d/dP_x + d/dR_x,   d/dA_x + d/dB_x = d/dP_x,
/// and a source at C, on which only the R tensor R^{tuv}(P - C) depends, d/dC_x = -d/dP_x of its own term. So the
/// kernel forms G^P_x = d/dP_x sum_k M_k V (one more shift of the R tensor) and G^R_x = d/dR_x sum_k M_k V (the
/// Hermite coefficients' derivatives to third order) and combines them as `centre_gradient_term` does (T. Helgaker
/// and P. R. Taylor, Theor. Chim. Acta 83, 177 (1992)).
pub(crate) struct PvpGradientKernel<T, M = PointMode>(std::marker::PhantomData<(T, M)>);

// PvpGradientKernel's scratch.
pub(crate) struct PvpGradientBuffer<'a> {
    hermite_coefficients: mmd::hermite_coefficients::HermiteCoefficients<'a>,
    r_buffer: mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D<'a>,
    r_tuv: mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'a, f64>,
    r_tuv_op: mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'a, f64>,
}

/// sum_tuv E^(n_x)_t E^(n_y)_u E^(n_z)_v R_{t + m_x, u + m_y, v + m_z}: the Hermite coefficients' derivatives of
/// orders `n` along each axis (`e[axis][order]`), the R tensor `r` (of order `lqn`) shifted by `m`.
#[inline(always)]
fn shifted_sum(e: &[[arrays::array3d::Array3DView<'_, f64>; 4]; 3], l: [(usize, usize); 3], n: [usize; 3], m: [usize; 3], r: &[f64], lqn: usize) -> f64 {
    let (ex, ey, ez) = (&e[0][n[0]], &e[1][n[1]], &e[2][n[2]]);
    let mut sum = 0.0;
    for v in 0..=l[2].0 + l[2].1 {
        let e_v = ez[(v, l[2].0, l[2].1)];
        if e_v == 0.0 {
            continue;
        }
        for u in 0..=l[1].0 + l[1].1 {
            let e_uv = ey[(u, l[1].0, l[1].1)] * e_v;
            if e_uv == 0.0 {
                continue;
            }
            let base = mmd::hermite_expansion::hermite_index(0, u + m[1], v + m[2], lqn) + m[0];
            for t in 0..=l[0].0 + l[0].1 {
                sum += ex[(t, l[0].0, l[0].1)] * e_uv * r[base + t];
            }
        }
    }
    sum
}

/// G^P_x (`d/dP`) or G^R_x (`!d/dP`) for the three x, from the R tensor `r`; see PvpGradientKernel.
#[inline(always)]
fn pvp_gradient_terms(
    e: &[[arrays::array3d::Array3DView<'_, f64>; 4]; 3],
    l: [(usize, usize); 3],
    r: &[f64],
    lqn: usize,
    alpha_beta: f64,
    alpha_minus_beta: f64,
    d_dp: bool,
) -> [f64; 3] {
    let unit = |k: usize, times: usize| -> [usize; 3] { std::array::from_fn(|axis| if axis == k { times } else { 0 }) };
    let add = |a: [usize; 3], b: [usize; 3]| -> [usize; 3] { std::array::from_fn(|axis| a[axis] + b[axis]) };
    std::array::from_fn(|x| {
        let ex = unit(x, 1);
        (0..3)
            .map(|k| {
                let (k1, k2) = (unit(k, 1), unit(k, 2));
                if d_dp {
                    alpha_beta * shifted_sum(e, l, [0; 3], add(ex, k2), r, lqn) + alpha_minus_beta * shifted_sum(e, l, k1, add(ex, k1), r, lqn)
                        - shifted_sum(e, l, k2, ex, r, lqn)
                } else {
                    alpha_beta * shifted_sum(e, l, ex, k2, r, lqn) + alpha_minus_beta * shifted_sum(e, l, add(ex, k1), k1, r, lqn)
                        - shifted_sum(e, l, add(ex, k2), [0; 3], r, lqn)
                }
            })
            .sum()
    })
}

impl<T, M> crate::gradient_kernels::OneElectronIntegralGradientKernel for PvpGradientKernel<T, M>
where
    T: PointChargeTrait,
    M: BuildRtuv3D<T> + 'static,
{
    type AdditionalArguments<'c> = &'c [T] where T: 'c;
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = PvpGradientBuffer<'b>;
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
        gradient_arguments: &crate::gradient_kernels::GradientArguments,
        weight: f64,
    )
    where
        Musize: MatrixTrait<usize>,
    {
        let PvpGradientBuffer { hermite_coefficients, r_buffer, r_tuv, r_tuv_op } = kernel_buffer;
        let need_centre_term = gradient_arguments.grad_a || gradient_arguments.grad_b;
        // The d/dR terms survive only when exactly one centre is differentiated.
        let need_ddr = gradient_arguments.grad_a != gradient_arguments.grad_b;

        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        for order in 1..=(if need_ddr { 3 } else { 2 }) {
            hermite_coefficients.compute_derivative(order, a, b, p, ab, dist_ab);
        }
        if need_centre_term {
            r_tuv.get_data_mut().fill(0.0);
            M::build(r_tuv, r_buffer, p, pos_ab, additional_arguments);
        }
        if let Some(idx) = gradient_arguments.grad_op {
            r_tuv_op.get_data_mut().fill(0.0);
            M::build(r_tuv_op, r_buffer, p, pos_ab, &additional_arguments[idx..=idx]);
        }
        // Order 3 is read only for G^R, when it has been computed.
        let e = [
            std::array::from_fn(|order| hermite_coefficients.get_e_x(order)),
            std::array::from_fn(|order| hermite_coefficients.get_e_y(order)),
            std::array::from_fn(|order| hermite_coefficients.get_e_z(order)),
        ];
        let (alpha, beta) = (a / p, b / p);
        let (alpha_beta, alpha_minus_beta) = (alpha * beta, alpha - beta);
        let lqn = mmd::hermite_expansion::hermite_order(r_tuv.get_data().len());
        let (r_data, r_op_data) = (r_tuv.get_data(), r_tuv_op.get_data());

        for (j, lmn_b) in cart_b_comps.iter_col().enumerate() {
            let norm2 = weight * cart2sph::compute_cart_gaussian_norm(b, lmn_b[0], lmn_b[1], lmn_b[2]);
            for (i, lmn_a) in cart_a_comps.iter_col().enumerate() {
                let norm1 = norm2 * cart2sph::compute_cart_gaussian_norm(a, lmn_a[0], lmn_a[1], lmn_a[2]);
                let l = [(lmn_a[0], lmn_b[0]), (lmn_a[1], lmn_b[1]), (lmn_a[2], lmn_b[2])];
                let mut grad = [0.0; 3];
                let mut grad_p = [0.0; 3];
                if need_centre_term {
                    let ddp = pvp_gradient_terms(&e, l, r_data, lqn, alpha_beta, alpha_minus_beta, true);
                    let ddr = if need_ddr { pvp_gradient_terms(&e, l, r_data, lqn, alpha_beta, alpha_minus_beta, false) } else { [0.0; 3] };
                    grad = crate::gradient_kernels::centre_gradient_term(gradient_arguments.grad_a, gradient_arguments.grad_b, ddr, ddp, alpha, beta);
                    if gradient_arguments.resolved {
                        grad_p = ddp;
                    }
                }
                if gradient_arguments.grad_op.is_some() {
                    let ddp_op = pvp_gradient_terms(&e, l, r_op_data, lqn, alpha_beta, alpha_minus_beta, true);
                    for (g, d) in grad.iter_mut().zip(ddp_op) {
                        *g -= d;
                    }
                }
                for (plane, g) in grad.iter().chain(grad_p.iter()).enumerate().take(if gradient_arguments.resolved { 6 } else { 3 }) {
                    cartesian_batch[(i, j, plane)] += norm1 * g;
                }
            }
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        // The R tensor three orders past lqn_a + lqn_b: the mixed second derivatives and one more for d/dP.
        let lqn_ab = lqn_a + lqn_b + 3;
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, 3, arena)?;
        let r_buffer = mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D::from_arena(lqn_ab + M::extra_headroom(additional_arguments), arena)?;
        let r_tuv = mmd::hermite_r_tensor::HermiteTuvSliceViewMut::from_arena(lqn_ab, arena)?;
        let r_tuv_op = mmd::hermite_r_tensor::HermiteTuvSliceViewMut::from_arena(lqn_ab, arena)?;
        Ok(PvpGradientBuffer { hermite_coefficients, r_buffer, r_tuv, r_tuv_op })
    }
}

impl<T, M> crate::gradient_kernels::ArrayBatchGradientKernel for PvpGradientKernel<T, M>
where
    T: PointChargeTrait,
    M: BuildRtuv3D<T> + 'static,
{
    fn n_planes<'c>(_additional_arguments: &Self::AdditionalArguments<'c>, gradient_arguments: &crate::gradient_kernels::GradientArguments) -> usize {
        if gradient_arguments.resolved { 6 } else { 3 }
    }
}
