// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use arrays;
use arrays::matrix::{MatrixTrait, MatrixTraitMut};
use arrays::array3d::Array3DTraitMut;
use mmd::hermite_r_tensor::{HermiteRTensorTrait, HermiteRTensorTraitMut};
use basis::charge_carriers::{FiniteChargeTrait, MultipoleSourceTrait, PointChargeTrait};
use mmd::hermite_coefficients::{HermiteCoefficientsTrait, HermiteCoefficientsTraitMut};
use mmd::hermite_moment::HermiteMoments;

/// A primitive-pair kernel of a one-electron operator.
pub trait OneElectronIntegralKernel {
    /// What the operator needs besides the basis (sources, sites, an origin), cut into runs of sites by the loops
    /// when the operator is a sum over its sites (`SUMS_OVER_SITES`).
    type AdditionalArguments<'c>: crate::pieces::SiteRuns where Self: 'c;
    /// Whether the integrals are a sum over the sites (and so may be computed by runs of sites and added); `false`
    /// for planes per site.
    const SUMS_OVER_SITES: bool = true;
    /// The Cartesian batch of one primitive product: a matrix (cart_a, cart_b), or planes of it.
    type KernelBatch<'b>;
    /// The kernel's scratch, allocated once per shell pair.
    type KernelBuffer<'b>;
    /// Adds the integrals of one primitive product, times `weight`, to `cartesian_batch`: exponents `a` and `b`,
    /// `p = a + b`, `ab = a b / p`, the Cartesian exponents of the two shells' components as columns of
    /// `cart_a_comps` and `cart_b_comps`.
    fn compute<'b, 'k, 'c, Musize>(
        cartesian_batch: &mut Self::KernelBatch<'b>,
        kernel_buffer: &mut Self::KernelBuffer<'k>,
        a: f64,
        b: f64,
        p: f64,
        ab: f64,
        cart_a_comps: &Musize,
        cart_b_comps: &Musize,
        // The separation of the two centres, B - A (ShellPair::get_separation), then the product centre P: in this
        // order, as every kernel names them (dist_ab, pos_ab).
        dist_ab: &[f64; 3],
        pos_ab: &[f64; 3],
        additional_arguments: &Self::AdditionalArguments<'c>,
        weight: f64,
    )
    where
        Musize: arrays::matrix::MatrixTrait<usize>;

    /// The scratch for shells of angular momenta `lqn_a` and `lqn_b`, from the arena.
    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError>;

}

// Kernels whose batch is a stack of planes (Array3DViewMut) rather than one matrix; the number of planes can depend
// on the additional arguments (the moment order, the number of sites).
pub(crate) trait ArrayBatchKernel: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>> {
    fn n_planes<'c>(additional_arguments: &Self::AdditionalArguments<'c>) -> usize;

    /// Whether plane `plane` of the output is an ANTISYMMETRIC matrix (A[j,i] = -A[i,j]) instead of a
    /// symmetric one. The driver stores only the upper triangle of every plane and mirrors it, so an
    /// antisymmetric plane needs the mirror negated, and the block of a shell pair that was computed
    /// with its two shells swapped (to keep the higher angular momentum second) needs negating as
    /// well: <b|O|a> = -<a|O|b>^T. The vector (spin-orbit) planes of the "full" sigma.p V sigma.p
    /// kernels are of this kind. Default: all planes symmetric.
    fn antisymmetric_plane(_plane: usize) -> bool {
        false
    }
}

/// The overlap kernel's scratch: the Hermite expansion coefficients.
pub struct OverlapBuffer<'b> {
    /// The coefficients of the current primitive product.
    pub hermite_coefficients: mmd::hermite_coefficients::HermiteCoefficients<'b>,
}

/// The overlap, <A|B>.
pub struct OverlapKernel;
impl OneElectronIntegralKernel for OverlapKernel {
    type AdditionalArguments<'c> = crate::additional_arguments::EmptyAdditionalArguments;
    type KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>;
    type KernelBuffer<'b> = OverlapBuffer<'b>;
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
        _pos_ab: &[f64; 3],
        _additional_arguments: &Self::AdditionalArguments<'c>,
        weight: f64,
    ) where
        Musize: arrays::matrix::MatrixTrait<usize>
    {
        let OverlapBuffer {
            hermite_coefficients,
        } = kernel_buffer;

        let coeff = weight * compute_oneint_coeff(p);
        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        let e_ab_x = hermite_coefficients.get_e_x(0);
        let e_ab_y = hermite_coefficients.get_e_y(0);
        let e_ab_z = hermite_coefficients.get_e_z(0);
        for (xb, lmn_b) in cartesian_batch.iter_col_mut().zip(cart_b_comps.iter_col()) {
            let lx2 = lmn_b[0];
            let ly2 = lmn_b[1];
            let lz2 = lmn_b[2];
            let norm2 = coeff * cart2sph::compute_cart_gaussian_norm(b, lx2, ly2, lz2);
            for (xa, lmn_a) in xb.iter_mut().zip(cart_a_comps.iter_col()) {
                let lx1 = lmn_a[0];
                let ly1 = lmn_a[1];
                let lz1 = lmn_a[2];
                let norm1 = norm2 * cart2sph::compute_cart_gaussian_norm(a, lx1, ly1, lz1);
                *xa +=
                    norm1 * e_ab_x[(0, lx1, lx2)] * e_ab_y[(0, ly1, ly2)] * e_ab_z[(0, lz1, lz2)];
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
        let derivative_order = 0; // Only given a variable to clarify the purpose of the variable. The derivative order is 0 for overlap integrals.
        let hermite_coefficients =
            mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
        Ok(OverlapBuffer {
            hermite_coefficients,
        })
    }


}

/// The kinetic energy, <A| -nabla^2/2 |B>, from the second derivatives G of the Hermite coefficients:
///      T_ab = -(1/2) (G_ij,x * E_ij,y * E_ij,z + E_ij,x * G_ij,y * E_ij,z + E_ij,x * E_ij,y * G_ij,z)
pub struct KineticKernel;
impl OneElectronIntegralKernel for KineticKernel {
    type AdditionalArguments<'c> = crate::additional_arguments::EmptyAdditionalArguments;
    type KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>;
    type KernelBuffer<'b> = OverlapBuffer<'b>;
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
        _pos_ab: &[f64; 3],
        _additional_arguments: &Self::AdditionalArguments<'c>,
        weight: f64,
    ) where
        Musize: arrays::matrix::MatrixTrait<usize>
    {
        let OverlapBuffer {
            hermite_coefficients,
            ..
        } = kernel_buffer;

        let coeff = weight * compute_oneint_coeff(p);
        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_gradient(a, b, p, ab, dist_ab); // Gradient necessary for laplacian.
        hermite_coefficients.compute_laplacian(a, b, p, ab, dist_ab);
        let e_ab_x = hermite_coefficients.get_e_x(0);
        let e_ab_y = hermite_coefficients.get_e_y(0);
        let e_ab_z = hermite_coefficients.get_e_z(0);
        let g_ab_x = hermite_coefficients.get_e_x(2);
        let g_ab_y = hermite_coefficients.get_e_y(2);
        let g_ab_z = hermite_coefficients.get_e_z(2);
        for (xb, lmn_b) in cartesian_batch.iter_col_mut().zip(cart_b_comps.iter_col()) {
            let lx2 = lmn_b[0];
            let ly2 = lmn_b[1];
            let lz2 = lmn_b[2];
            let norm2 = -0.5 * coeff * cart2sph::compute_cart_gaussian_norm(b, lx2, ly2, lz2);
            for (xa, lmn_a) in xb.iter_mut().zip(cart_a_comps.iter_col()) {
                let lx1 = lmn_a[0];
                let ly1 = lmn_a[1];
                let lz1 = lmn_a[2];
                let norm1 = norm2 * cart2sph::compute_cart_gaussian_norm(a, lx1, ly1, lz1);
                let tx = g_ab_x[(0, lx1, lx2)] * e_ab_y[(0, ly1, ly2)] * e_ab_z[(0, lz1, lz2)];
                let ty = e_ab_x[(0, lx1, lx2)] * g_ab_y[(0, ly1, ly2)] * e_ab_z[(0, lz1, lz2)];
                let tz = e_ab_x[(0, lx1, lx2)] * e_ab_y[(0, ly1, ly2)] * g_ab_z[(0, lz1, lz2)];
                *xa += norm1 * (tx + ty + tz);
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
        let derivative_order = 2; // Only given a variable to clarify the purpose of the variable. The derivative order is 2 for kinetic integrals.
        let hermite_coefficients =
            mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
        Ok(OverlapBuffer {
            hermite_coefficients,
        })
    }

}

// Point charges.
pub(crate) struct PointMode;
// Gaussian charge distributions.
pub(crate) struct FiniteMode;
// AIMP screens, exp(-c r_C^2) / r_C per screening Gaussian, as radial operators of P - C
// (`HermiteCoulombIntegrals4D::compute_screened`).
pub(crate) struct ScreenMode;
// The AIMPs' whole Coulomb potential: the bare -Z/r and the screen, summed into one R tensor.
pub(crate) struct AimpCoulombMode;

// The potential energy of an electron in the field of the sources; `M` selects the source model and builds the
// summed R_tuv: point charges (PointMode), Gaussian charges (FiniteMode), point multipoles (MultipoleSourceMode).
pub(crate) struct CoulombKernel<T, M = PointMode>(std::marker::PhantomData<(T, M)>);

// CoulombKernel's scratch.
pub(crate) struct PointChargeBuffer<'a> {
    hermite_coefficients: mmd::hermite_coefficients::HermiteCoefficients<'a> ,
    r_buffer: mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D<'a>,
    r_tuv: &'a mut [f64],
    lqn_ab: usize,
}

// Builds the R_tuv of the sources, summed and scaled, for CoulombKernel.
pub(crate) trait BuildRtuv<T> {
    fn build(
        r_tuv: &mut [f64],
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        sources: &[T],
    );

    // The orders past lqn_a + lqn_b that `build` reads the R tensor at: n for multipole sources of order up to n,
    // which read R(t+n, u+n, v+n). 0 for point and finite charges.
    fn extra_headroom(_sources: &[T]) -> usize { 0 }
}
impl<T, M> OneElectronIntegralKernel for CoulombKernel<T, M>
where
    T: PointChargeTrait,
    M: BuildRtuv<T> + 'static,
{
    type AdditionalArguments<'c> = &'c [T] where T: 'c;
    type KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>;
    type KernelBuffer<'b> = PointChargeBuffer<'b>;
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
        Musize: arrays::matrix::MatrixTrait<usize>
    {
        let PointChargeBuffer {
            hermite_coefficients,
            r_buffer,
            r_tuv,
            lqn_ab,
        } = kernel_buffer;
        let lqn_ab = *lqn_ab;

        r_tuv.fill(0.0);
        M::build(r_tuv, r_buffer, p, pos_ab, additional_arguments);

        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        let e_x = hermite_coefficients.get_e_x(0);
        let e_y = hermite_coefficients.get_e_y(0);
        let e_z = hermite_coefficients.get_e_z(0);

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
                let mut sum = 0.0;
                for v in 0..v_max {
                    let e_v = e_z[(v, lz1, lz2)];
                    for u in 0..u_max {
                        let e_u = e_y[(u, ly1, ly2)];
                        let base = mmd::hermite_expansion::hermite_index(0, u, v, lqn_ab);
                        for t in 0..t_max {
                            let e_t = e_x[(t, lx1, lx2)];
                            sum += e_t * e_u * e_v * r_tuv[base + t];
                        }
                    }
                }
                cartesian_batch[(i, j)] += norm1 * sum;
            }
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        additional_arguments: &Self::AdditionalArguments<'c>
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        let lqn_ab = lqn_a + lqn_b;
        let tuv_size = mmd::hermite_expansion::hermite_count(lqn_ab);
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, 0, arena)?;
        let r_buffer = mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D::from_arena(lqn_ab + M::extra_headroom(additional_arguments), arena)?;
        let r_tuv = arena.alloc_zeroed(tuv_size)?;
        Ok(PointChargeBuffer {
            hermite_coefficients,
            r_buffer,
            r_tuv,
            lqn_ab,
        })
    }

}

impl<T: basis::charge_carriers::PointChargeTrait> BuildRtuv<T> for PointMode {
    #[inline(always)]
    fn build(
        r_tuv: &mut [f64],
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        point_charges: &[T]
    ) {
        let constant_factor = -2.0 * std::f64::consts::PI / p;
        for point_charge in point_charges.iter() {
            let dist_abc = arrays::vector_math::vector_subtract_3d(&pos_ab, point_charge.get_position());
            let rpc2 = arrays::vector_math::vector_norm_square_3d(&dist_abc);
            let prefac = constant_factor * point_charge.get_charge();
            r_buffer.compute(p, dist_abc[0], dist_abc[1], dist_abc[2], rpc2);
            let r_view = r_buffer.view_tuv();
            let r_slice = r_view.get_data();
            arrays::vector_math::add_scaled_in_place(r_tuv, &r_slice, prefac);
        }
    }
}

impl<T: basis::charge_carriers::FiniteChargeTrait> BuildRtuv<T> for FiniteMode {
    #[inline(always)]
    fn build(
        r_tuv: &mut [f64],
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        finite_charges: &[T]
    ) {
        let constant_factor = -2.0 * std::f64::consts::PI / p;
        for charge in finite_charges.iter() {
            let dist_abc = arrays::vector_math::vector_subtract_3d(&pos_ab, charge.get_position());
            let rpc2 = arrays::vector_math::vector_norm_square_3d(&dist_abc);
            for (&q, &charge_i) in charge.get_exponents().iter().zip(charge.get_charges()) {
                let eta = p * q / (p + q);
                let prefac = constant_factor * charge_i * (q / (p + q)).sqrt();
                r_buffer.compute(eta, dist_abc[0], dist_abc[1], dist_abc[2], rpc2);
                let r_view = r_buffer.view_tuv();
                let r_slice = r_view.get_data();
                arrays::vector_math::add_scaled_in_place(r_tuv, r_slice, prefac);
            }
        }
    }
}

/// The screens' R_tuv summed over every AIMP and screening Gaussian: -Z A_i (2 pi / q) times the R tensor of
/// exp(-mu s) F_0(beta s), so the pair's Hermite coefficients are contracted once for all of them.
#[inline(always)]
fn build_screens<T: basis::charge_carriers::ScreenedChargeTrait>(
    r_tuv: &mut [f64],
    r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
    p: f64,
    pos_ab: &[f64; 3],
    aimps: &[T],
) {
    let twopi = 2.0 * std::f64::consts::PI;
    for aimp in aimps {
        let dist = arrays::vector_math::vector_subtract_3d(pos_ab, aimp.get_position());
        let rpc2 = arrays::vector_math::vector_norm_square_3d(&dist);
        let screen = aimp.get_screened_coulomb_potential();
        let charge_factor = -twopi * aimp.get_charge();
        for (&c, &coefficient) in screen.get_exponents().iter().zip(screen.get_coefficients()) {
            r_buffer.compute_screened(p, c, dist[0], dist[1], dist[2], rpc2);
            let r_view = r_buffer.view_tuv();
            arrays::vector_math::add_scaled_in_place(r_tuv, r_view.get_data(), charge_factor * coefficient / (p + c));
        }
    }
}

impl<T: basis::charge_carriers::ScreenedChargeTrait> BuildRtuv<T> for ScreenMode {
    #[inline(always)]
    fn build(
        r_tuv: &mut [f64],
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        aimps: &[T],
    ) {
        build_screens(r_tuv, r_buffer, p, pos_ab, aimps);
    }
}

impl<T: basis::charge_carriers::ScreenedChargeTrait> BuildRtuv3D<T> for ScreenMode {
    #[inline(always)]
    fn build(
        r_tuv: &mut mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'_, f64>,
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        aimps: &[T],
    ) {
        build_screens(r_tuv.get_data_mut(), r_buffer, p, pos_ab, aimps);
    }
}

impl<T: basis::charge_carriers::ScreenedChargeTrait> BuildRtuv3D<T> for AimpCoulombMode {
    #[inline(always)]
    fn build(
        r_tuv: &mut mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'_, f64>,
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        aimps: &[T],
    ) {
        <PointMode as BuildRtuv3D<T>>::build(r_tuv, r_buffer, p, pos_ab, aimps);
        build_screens(r_tuv.get_data_mut(), r_buffer, p, pos_ab, aimps);
    }
}

/// The Cartesian Gaussians' normalizations N_a N_b times `weight`, per component pair of a primitive product.
#[inline(always)]
pub(crate) fn fill_norm_matrix<Mf64, Musize>(
    norm_matrix: &mut Mf64,
    a: f64,
    b: f64,
    cart_a_comps: &Musize,
    cart_b_comps: &Musize,
    weight: f64,
) where
    Mf64: arrays::matrix::MatrixTraitMut<f64>,
    Musize: MatrixTrait<usize>,
{
    for (j, lmn_b) in cart_b_comps.iter_col().enumerate() {
        let norm2 = weight * cart2sph::compute_cart_gaussian_norm(b, lmn_b[0], lmn_b[1], lmn_b[2]);
        for (i, lmn_a) in cart_a_comps.iter_col().enumerate() {
            norm_matrix[(i, j)] = norm2 * cart2sph::compute_cart_gaussian_norm(a, lmn_a[0], lmn_a[1], lmn_a[2]);
        }
    }
}

// BuildRtuv into the (t,u,v)-indexable slice of the gradient kernels.
pub(crate) trait BuildRtuv3D<T> {
    fn build(
        r_tuv: &mut mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'_, f64>,
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        sources: &[T],
    );

    // As BuildRtuv::extra_headroom.
    fn extra_headroom(_sources: &[T]) -> usize { 0 }
}

impl<T: PointChargeTrait> BuildRtuv3D<T> for PointMode {
    #[inline(always)]
    fn build(
        r_tuv: &mut mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'_, f64>,
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64, pos_ab: &[f64; 3],
        charges: &[T]
    ) {
        let constant_factor = -2.0 * std::f64::consts::PI / p;
        let r_tuv_slice = r_tuv.get_data_mut();
        for charge in charges.iter() {
            let dist_abc = arrays::vector_math::vector_subtract_3d(&pos_ab, charge.get_position());
            let rpc2 = arrays::vector_math::vector_norm_square_3d(&dist_abc);
            let prefac = constant_factor * charge.get_charge();
            r_buffer.compute(p, dist_abc[0], dist_abc[1], dist_abc[2], rpc2);
            let r_view = r_buffer.view_tuv();
            let r_slice = r_view.get_data();
            arrays::vector_math::add_scaled_in_place(r_tuv_slice, &r_slice, prefac);
        }
    }
}

impl<T: FiniteChargeTrait> BuildRtuv3D<T> for FiniteMode {
    #[inline(always)]
    fn build(
        r_tuv: &mut mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'_, f64>,
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        finite_charges: &[T]
    ) {
        let constant_factor = -2.0 * std::f64::consts::PI / p;
        let r_tuv_slice = r_tuv.get_data_mut();
        for charge in finite_charges.iter() {
            let dist_abc = arrays::vector_math::vector_subtract_3d(&pos_ab, charge.get_position());
            let rpc2 = arrays::vector_math::vector_norm_square_3d(&dist_abc);
            for (&q, &charge_i) in charge.get_exponents().iter().zip(charge.get_charges()) {
                let eta = p * q / (p + q);
                let prefac = constant_factor * charge_i * (q / (p + q)).sqrt();
                r_buffer.compute(eta, dist_abc[0], dist_abc[1], dist_abc[2], rpc2);
                let r_view = r_buffer.view_tuv();
                let r_slice = r_view.get_data();
                arrays::vector_math::add_scaled_in_place(r_tuv_slice, &r_slice, prefac);
            }
        }
    }
}

// Which moments MultipoleSourceMode contracts with the shifted R tensor: the contact-free (harmonic) ones or the
// raw ones (the exact interaction, contact terms included).
pub(crate) trait MomentConvention: 'static {
    fn moment<T: MultipoleSourceTrait>(source: &T, lx: usize, ly: usize, lz: usize) -> f64;
}
// The harmonic moments.
pub(crate) struct ContactFree;
// The raw moments.
pub(crate) struct Exact;
impl MomentConvention for ContactFree {
    #[inline(always)]
    fn moment<T: MultipoleSourceTrait>(source: &T, lx: usize, ly: usize, lz: usize) -> f64 {
        source.get_contact_free_moment(lx, ly, lz)
    }
}
impl MomentConvention for Exact {
    #[inline(always)]
    fn moment<T: MultipoleSourceTrait>(source: &T, lx: usize, ly: usize, lz: usize) -> f64 {
        source.get_moment(lx, ly, lz)
    }
}

// Point multipoles: a moment (lx, ly, lz) of a source reads the R tensor shifted to R(t+lx, u+ly, v+lz), and every
// moment of every source sums into the one R_tuv.
pub(crate) struct MultipoleSourceMode<C = ContactFree>(std::marker::PhantomData<C>);

impl<T: MultipoleSourceTrait, C: MomentConvention> BuildRtuv<T> for MultipoleSourceMode<C> {
    fn extra_headroom(sources: &[T]) -> usize {
        sources.iter().map(|source| source.get_max_order()).max().unwrap_or(0)
    }

    #[inline(always)]
    fn build(
        r_tuv: &mut [f64],
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        sources: &[T],
    ) {
        // r_tuv is of order lqn_ab; only r_buffer carries the extra_headroom the shift reads.
        let lqn_ab = mmd::hermite_expansion::hermite_order(r_tuv.len());
        let constant_factor = -2.0 * std::f64::consts::PI / p;

        for source in sources.iter() {
            let dist_abc = arrays::vector_math::vector_subtract_3d(&pos_ab, source.get_position());
            let rpc2 = arrays::vector_math::vector_norm_square_3d(&dist_abc);
            r_buffer.compute(p, dist_abc[0], dist_abc[1], dist_abc[2], rpc2);
            let r_view = r_buffer.view_tuv();

            for order in 0..=source.get_max_order() {
                // Each derivative with respect to the source's position rather than the field point flips the sign.
                let sign = if order % 2 == 1 { -1.0 } else { 1.0 };
                math_utilities::cartesian::for_each_cartesian_component(order, |lx, ly, lz| {
                    let moment = C::moment(source, lx, ly, lz);
                    if moment == 0.0 {
                        return;
                    }
                    // The Taylor expansion of 1/|r-C| weights the moment (lx, ly, lz) by 1/(lx! ly! lz!).
                    let factorial_lx = math_utilities::special_functions::compute_factorial(lx as u64) as f64;
                    let factorial_ly = math_utilities::special_functions::compute_factorial(ly as u64) as f64;
                    let factorial_lz = math_utilities::special_functions::compute_factorial(lz as u64) as f64;
                    let scale = constant_factor * sign * moment / (factorial_lx * factorial_ly * factorial_lz);
                    // (v, u, t) ascending visits the (t,u,v) in hermite_index order, so a running counter is the
                    // destination index.
                    let mut out_idx = 0usize;
                    for v in 0..=lqn_ab {
                        for u in 0..=(lqn_ab - v) {
                            let r_row = r_view.get_t_slice(u + ly, v + lz);
                            for t in 0..=(lqn_ab - u - v) {
                                r_tuv[out_idx] += scale * r_row[t + lx];
                                out_idx += 1;
                            }
                        }
                    }
                });
            }
        }
    }
}

// As the BuildRtuv impl above. Every term depends on P and C only through P - C, so d/dC = -d/dP holds as for
// point charges.
impl<T: MultipoleSourceTrait, C: MomentConvention> BuildRtuv3D<T> for MultipoleSourceMode<C> {
    fn extra_headroom(sources: &[T]) -> usize {
        sources.iter().map(|source| source.get_max_order()).max().unwrap_or(0)
    }

    #[inline(always)]
    fn build(
        r_tuv: &mut mmd::hermite_r_tensor::HermiteTuvSliceViewMut<'_, f64>,
        r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
        p: f64,
        pos_ab: &[f64; 3],
        sources: &[T],
    ) {
        let r_tuv_slice = r_tuv.get_data_mut();
        let lqn_ab = mmd::hermite_expansion::hermite_order(r_tuv_slice.len());
        let constant_factor = -2.0 * std::f64::consts::PI / p;

        for source in sources.iter() {
            let dist_abc = arrays::vector_math::vector_subtract_3d(&pos_ab, source.get_position());
            let rpc2 = arrays::vector_math::vector_norm_square_3d(&dist_abc);
            r_buffer.compute(p, dist_abc[0], dist_abc[1], dist_abc[2], rpc2);
            let r_view = r_buffer.view_tuv();

            for order in 0..=source.get_max_order() {
                let sign = if order % 2 == 1 { -1.0 } else { 1.0 };
                math_utilities::cartesian::for_each_cartesian_component(order, |lx, ly, lz| {
                    let moment = C::moment(source, lx, ly, lz);
                    if moment == 0.0 {
                        return;
                    }
                    // The Taylor expansion's 1/(lx! ly! lz!).
                    let factorial_lx = math_utilities::special_functions::compute_factorial(lx as u64) as f64;
                    let factorial_ly = math_utilities::special_functions::compute_factorial(ly as u64) as f64;
                    let factorial_lz = math_utilities::special_functions::compute_factorial(lz as u64) as f64;
                    let scale = constant_factor * sign * moment / (factorial_lx * factorial_ly * factorial_lz);
                    // A running counter over (v, u, t) ascending is the destination index.
                    let mut out_idx = 0usize;
                    for v in 0..=lqn_ab {
                        for u in 0..=(lqn_ab - v) {
                            let r_row = r_view.get_t_slice(u + ly, v + lz);
                            for t in 0..=(lqn_ab - u - v) {
                                r_tuv_slice[out_idx] += scale * r_row[t + lx];
                                out_idx += 1;
                            }
                        }
                    }
                });
            }
        }
    }
}

/// R-tensor combinations for the electric field of a unit electron at `site`,
/// evaluated at the Hermite centre P. Component `k` of `r_field` is overwritten
/// (not accumulated) with
///   -2π/p · R(t+δ_kx, u+δ_ky, v+δ_kz)
/// for every (t,u,v), i.e. the Hermite expansion of the operator
///   (r - C) / |r - C|^3
/// (d/dC of 1/|r-C|), which is the field at C from a charge -1 at r.
/// The unshifted R comes from the same Coulomb recursion as CoulombKernel,
/// so `r_buffer` must have been built one order past lqn_ab.
#[inline(always)]
fn build_field_at_site<T: basis::charge_carriers::SiteTrait>(
    r_field: &mut [&mut [f64]; 3],
    r_buffer: &mut mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D,
    p: f64,
    pos_ab: &[f64; 3],
    site: &T,
) {
    const SHIFTS: [[usize; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let lqn_ab = mmd::hermite_expansion::hermite_order(r_field[0].len());
    let constant_factor = -2.0 * std::f64::consts::PI / p;

    let dist_abc = arrays::vector_math::vector_subtract_3d(&pos_ab, site.get_position());
    let rpc2 = arrays::vector_math::vector_norm_square_3d(&dist_abc);
    r_buffer.compute(p, dist_abc[0], dist_abc[1], dist_abc[2], rpc2);
    let r_view = r_buffer.view_tuv();

    for (r_out, [dt, du, dv]) in r_field.iter_mut().zip(SHIFTS) {
        for v in 0..=lqn_ab {
            for u in 0..=(lqn_ab - v) {
                let out_start = mmd::hermite_expansion::hermite_index(0, u, v, lqn_ab);
                let r_row = r_view.get_t_slice(u + du, v + dv);
                for t in 0..=(lqn_ab - u - v) {
                    r_out[out_start + t] = constant_factor * r_row[t + dt];
                }
            }
        }
    }
}

/// Calculate the electric-field integrals
///   < A | (r - C_s) / |r - C_s|^3 | B >
/// for every site s, i.e. the field at site s due to a single electron
/// (charge -1) distributed as the AO product |A B|.
///
/// The output has three planes per site, component-last, ordered
/// (site 0: x, y, z; site 1: x, y, z; ...). The field at a site is obtained
/// by contracting these with the density matrix outside of this kernel.
pub(crate) struct ElectricFieldIntegrals<T>(std::marker::PhantomData<T>);

// ElectricFieldIntegrals' scratch.
pub(crate) struct ElectricFieldBuffer<'a> {
    hermite_coefficients: mmd::hermite_coefficients::HermiteCoefficients<'a>,
    r_buffer: mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D<'a>,
    r_field: [&'a mut [f64]; 3],
    norm_matrix: arrays::matrix::MatrixViewMut<'a, f64>,
    lqn_ab: usize,
}

impl<T> OneElectronIntegralKernel for ElectricFieldIntegrals<T>
where
    T: basis::charge_carriers::SiteTrait,
{
    type AdditionalArguments<'c> = &'c [T] where T: 'c;
    const SUMS_OVER_SITES: bool = false;
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = ElectricFieldBuffer<'b>;
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
        sites: &Self::AdditionalArguments<'c>,
        weight: f64,
    )
    where
        Musize: arrays::matrix::MatrixTrait<usize>
    {
        let ElectricFieldBuffer {
            hermite_coefficients,
            r_buffer,
            r_field,
            norm_matrix,
            lqn_ab,
        } = kernel_buffer;
        let lqn_ab = *lqn_ab;

        // Once per primitive product, shared by every site and component.
        fill_norm_matrix(norm_matrix, a, b, cart_a_comps, cart_b_comps, weight);

        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        let e_x = hermite_coefficients.get_e_x(0);
        let e_y = hermite_coefficients.get_e_y(0);
        let e_z = hermite_coefficients.get_e_z(0);

        for (i_site, site) in sites.iter().enumerate() {
            build_field_at_site(r_field, r_buffer, p, pos_ab, site);
            for (component, r_component) in r_field.iter().enumerate() {
                let mut plane = cartesian_batch.view_sub_matrix_mut(3 * i_site + component);
                for (j, lmn_b) in cart_b_comps.iter_col().enumerate() {
                    let lx2 = lmn_b[0];
                    let ly2 = lmn_b[1];
                    let lz2 = lmn_b[2];
                    for (i, lmn_a) in cart_a_comps.iter_col().enumerate() {
                        let lx1 = lmn_a[0];
                        let ly1 = lmn_a[1];
                        let lz1 = lmn_a[2];
                        let norm1 = norm_matrix[(i, j)];
                        let v_max = lz1 + lz2 + 1;
                        let u_max = ly1 + ly2 + 1;
                        let t_max = lx1 + lx2 + 1;
                        let mut sum = 0.0;
                        for v in 0..v_max {
                            let e_v = e_z[(v, lz1, lz2)];
                            for u in 0..u_max {
                                let e_u = e_y[(u, ly1, ly2)];
                                let base = mmd::hermite_expansion::hermite_index(0, u, v, lqn_ab);
                                for t in 0..t_max {
                                    let e_t = e_x[(t, lx1, lx2)];
                                    sum += e_t * e_u * e_v * r_component[base + t];
                                }
                            }
                        }
                        plane[(i, j)] += norm1 * sum;
                    }
                }
            }
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        _additional_arguments: &Self::AdditionalArguments<'c>
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        let cart_comps_a = cart2sph::compute_cartesian_degeneracy(lqn_a);
        let cart_comps_b = cart2sph::compute_cartesian_degeneracy(lqn_b);
        let lqn_ab = lqn_a + lqn_b;
        let tuv_size = mmd::hermite_expansion::hermite_count(lqn_ab);
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, 0, arena)?;
        // +1 headroom on the R-tensor order: each field component reads R
        // shifted by one in its own direction.
        let r_buffer = mmd::hermite_coulomb_integrals::HermiteCoulombIntegrals4D::from_arena(lqn_ab + 1, arena)?;
        let r_field = [
            arena.alloc_zeroed(tuv_size)?,
            arena.alloc_zeroed(tuv_size)?,
            arena.alloc_zeroed(tuv_size)?,
        ];
        let norm_matrix = arrays::matrix::MatrixViewMut::<f64>::from_arena(cart_comps_a, cart_comps_b, arena)?;
        Ok(ElectricFieldBuffer {
            hermite_coefficients,
            r_buffer,
            r_field,
            norm_matrix,
            lqn_ab,
        })
    }

}

impl<T> ArrayBatchKernel for ElectricFieldIntegrals<T>
where
    T: basis::charge_carriers::SiteTrait,
{
    fn n_planes<'c>(sites: &Self::AdditionalArguments<'c>) -> usize {
        3 * sites.len()
    }
}

// (pi/p)^(3/2).
pub(crate) fn compute_oneint_coeff(p: f64) -> f64 {
    let sq = (std::f64::consts::PI / p).sqrt();
    sq * sq * sq
}

/// MomentKernel's scratch.
pub struct MomentBuffer<'a> {
    pub(crate) hermite_coefficients: mmd::hermite_coefficients::HermiteCoefficients<'a>,
    pub(crate) hermite_moments: HermiteMoments<'a>,
}

/// <A|(x-Ox)^lx(y-Oy)^ly(z-Oz)^lz|B> for every Cartesian (lx,ly,lz) of one order about one origin, one plane per
/// component in the canonical order: order 1 -> x, y, z; order 2 -> xx, xy, xz, yy, yz, zz.
pub struct MomentKernel;
impl OneElectronIntegralKernel for MomentKernel {
    type AdditionalArguments<'c> = &'c crate::additional_arguments::MomentOrigin;
    // Component-last: component k occupies one contiguous cart_a*cart_b block.
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
            general_moment::<Musize>(
                &mut cartesian_batch.view_sub_matrix_mut(xyz),
                hermite_coefficients,
                hermite_moments,
                cart_a_comps,
                cart_b_comps,
                a,
                b,
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
        let derivative_order = 0;
        let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
        let order = additional_arguments.get_order();
        let hermite_moments = HermiteMoments::from_arena(order, derivative_order, arena)?;
        Ok(MomentBuffer {
            hermite_coefficients,
            hermite_moments,
        })
    }

}

// The planes of a moment of this order: its Cartesian components (1, 3, 6, ...).
pub(crate) fn moment_component_count(order: usize) -> usize {
    cart2sph::compute_cartesian_degeneracy(order)
}

impl ArrayBatchKernel for MomentKernel {
    fn n_planes<'c>(additional_arguments: &Self::AdditionalArguments<'c>) -> usize {
        moment_component_count(additional_arguments.get_order())
    }
}

#[inline(always)]
fn general_moment<M: MatrixTrait<usize>>(
    oneint_batch: &mut arrays::matrix::MatrixViewMut<f64>,
    hermite_coefficients: &mmd::hermite_coefficients::HermiteCoefficients<'_>,
    hermite_moments: &HermiteMoments<'_>,
    cart_a_comps: &M,
    cart_b_comps: &M,
    a: f64,
    b: f64,
    coeff: f64,
    order_x: usize,
    order_y: usize,
    order_z: usize
) {
    let e_ab_x = hermite_coefficients.get_e_x(0);
    let e_ab_y = hermite_coefficients.get_e_y(0);
    let e_ab_z = hermite_coefficients.get_e_z(0);
    let mm_x = hermite_moments.get_mm_x();
    let mm_y = hermite_moments.get_mm_y();
    let mm_z = hermite_moments.get_mm_z();

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
            for t in 0..=t_max {
                sx += e_ab_x[(t, lx1, lx2)] * mm_x[(t, order_x)];
            }

            let mut sy = 0.0;
            for u in 0..=u_max {
                sy += e_ab_y[(u, ly1, ly2)] * mm_y[(u, order_y)];
            }

            let mut sz = 0.0;
            for v in 0..=v_max {
                sz += e_ab_z[(v, lz1, lz2)] * mm_z[(v, order_z)];
            }

            *xa += norm1 * sx * sy * sz;
        }
    }
}

// The linear and orbital angular momentum integrals, real and antisymmetric, from the overlap's Hermite coefficients
// and their first derivatives (no ladder shifts):
//
//   nabla_x = <a| d/dx |b>,                                      p = -i nabla
//   A_x     = <a| (y - O_y) d/dz - (z - O_z) d/dy |b>, (x y z cyclic),   L = -i (r - O) x nabla = -i A
//
// Both factor into one-dimensional integrals per Cartesian direction (the 3D prefactor (pi/p)^{3/2} taken out):
//   overlap   S = E_0,
//   moment    M = E_0 X_PO + E_1          (<(x - O_x)>, from the Hermite moments: int x_P Lambda_t = delta_t1),
//   derivative D = -E'_0                  (<i| d/dx |j>),
// where E' is the derivative of the coefficients with respect to the separation `dist_ab = B - A`
// (HermiteCoefficients::compute_gradient), i.e. d/dB_x of <i|j>, which is -<i| d/dx |j>. So nabla_x = D_x S_y S_z
// and A_x = S_x (M_y D_z - D_y M_z).
//
// A same-shell pair's block is symmetric in its primitive products for both, as the packed storage of such a
// pair (products i <= j) requires.
pub struct NablaKernel;
impl OneElectronIntegralKernel for NablaKernel {
    type AdditionalArguments<'c> = crate::additional_arguments::EmptyAdditionalArguments;
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = OverlapBuffer<'b>;

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
        _additional_arguments: &Self::AdditionalArguments<'c>,
        weight: f64,
    ) where
        Musize: MatrixTrait<usize>,
    {
        let factors = MomentumFactors::compute(&mut kernel_buffer.hermite_coefficients, a, b, p, ab, dist_ab, pos_ab, &[0.0; 3]);
        let coeff = weight * compute_oneint_coeff(p);
        for k in 0..3 {
            let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
            fill_cartesian_plane(&mut cartesian_batch.view_sub_matrix_mut(k), cart_a_comps, cart_b_comps, a, b, coeff, |la, lb| {
                factors.derivative(k, la, lb) * factors.overlap(k1, la, lb) * factors.overlap(k2, la, lb)
            });
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        _additional_arguments: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        momentum_buffer(lqn_a, lqn_b, arena)
    }
}

impl ArrayBatchKernel for NablaKernel {
    fn n_planes<'c>(_additional_arguments: &Self::AdditionalArguments<'c>) -> usize {
        3
    }

    fn antisymmetric_plane(_plane: usize) -> bool {
        true
    }
}

/// `A_k = <a| ((r - O) x nabla)_k |b>` about the gauge origin `O` (the additional argument). See `NablaKernel`.
pub struct AngularMomentumKernel;
impl OneElectronIntegralKernel for AngularMomentumKernel {
    type AdditionalArguments<'c> = [f64; 3];
    type KernelBatch<'b> = arrays::array3d::Array3DViewMut<'b, f64>;
    type KernelBuffer<'b> = OverlapBuffer<'b>;

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
        origin: &Self::AdditionalArguments<'c>,
        weight: f64,
    ) where
        Musize: MatrixTrait<usize>,
    {
        let factors = MomentumFactors::compute(&mut kernel_buffer.hermite_coefficients, a, b, p, ab, dist_ab, pos_ab, origin);
        let coeff = weight * compute_oneint_coeff(p);
        for k in 0..3 {
            // A_k = S_k (M_k1 D_k2 - D_k1 M_k2), (k, k1, k2) cyclic.
            let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
            fill_cartesian_plane(&mut cartesian_batch.view_sub_matrix_mut(k), cart_a_comps, cart_b_comps, a, b, coeff, |la, lb| {
                factors.overlap(k, la, lb)
                    * (factors.moment(k1, la, lb) * factors.derivative(k2, la, lb) - factors.derivative(k1, la, lb) * factors.moment(k2, la, lb))
            });
        }
    }

    fn kernel_buffer_allocation<'b, 'c>(
        lqn_a: usize,
        lqn_b: usize,
        arena: &arrays::scratch::Arena<'b>,
        _origin: &Self::AdditionalArguments<'c>,
    ) -> Result<Self::KernelBuffer<'b>, arrays::scratch::ArenaError> {
        momentum_buffer(lqn_a, lqn_b, arena)
    }
}

impl ArrayBatchKernel for AngularMomentumKernel {
    fn n_planes<'c>(_origin: &Self::AdditionalArguments<'c>) -> usize {
        3
    }

    fn antisymmetric_plane(_plane: usize) -> bool {
        true
    }
}

/// The Hermite coefficients and their first derivatives: the overlap's buffer, one derivative order up.
fn momentum_buffer<'b>(lqn_a: usize, lqn_b: usize, arena: &arrays::scratch::Arena<'b>) -> Result<OverlapBuffer<'b>, arrays::scratch::ArenaError> {
    let derivative_order = 1;
    let hermite_coefficients = mmd::hermite_coefficients::HermiteCoefficients::from_arena(lqn_a, lqn_b, derivative_order, arena)?;
    Ok(OverlapBuffer { hermite_coefficients })
}

/// The one-dimensional overlap, moment and derivative factors of one primitive product, per direction (see
/// `NablaKernel`), read from its Hermite coefficients.
struct MomentumFactors<'h> {
    e: [arrays::array3d::Array3DView<'h, f64>; 3],
    e_derivative: [arrays::array3d::Array3DView<'h, f64>; 3],
    x_po: [f64; 3],
}

impl<'h> MomentumFactors<'h> {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn compute(
        hermite_coefficients: &'h mut mmd::hermite_coefficients::HermiteCoefficients<'_>,
        a: f64,
        b: f64,
        p: f64,
        ab: f64,
        dist_ab: &[f64; 3],
        pos_ab: &[f64; 3],
        origin: &[f64; 3],
    ) -> Self {
        hermite_coefficients.compute(a, b, p, ab, dist_ab);
        hermite_coefficients.compute_gradient(a, b, p, ab, dist_ab);
        let hermite_coefficients = &*hermite_coefficients;
        MomentumFactors {
            e: [hermite_coefficients.get_e_x(0), hermite_coefficients.get_e_y(0), hermite_coefficients.get_e_z(0)],
            e_derivative: [hermite_coefficients.get_e_x(1), hermite_coefficients.get_e_y(1), hermite_coefficients.get_e_z(1)],
            x_po: std::array::from_fn(|k| pos_ab[k] - origin[k]),
        }
    }

    /// `S = E_0` along direction `k`, for the Cartesian exponents `la`, `lb` of the two functions.
    #[inline(always)]
    fn overlap(&self, k: usize, la: &[usize], lb: &[usize]) -> f64 {
        self.e[k][(0, la[k], lb[k])]
    }

    /// `M = E_0 X_PO + E_1`, `<(x - O_x)>` along direction `k` (`E_1` is zero, and not stored, for two s factors).
    #[inline(always)]
    fn moment(&self, k: usize, la: &[usize], lb: &[usize]) -> f64 {
        let (i, j) = (la[k], lb[k]);
        let e_1 = if i + j > 0 { self.e[k][(1, i, j)] } else { 0.0 };
        self.e[k][(0, i, j)] * self.x_po[k] + e_1
    }

    /// `D = -E'_0`, `<i| d/dx |j>` along direction `k`.
    #[inline(always)]
    fn derivative(&self, k: usize, la: &[usize], lb: &[usize]) -> f64 {
        -self.e_derivative[k][(0, la[k], lb[k])]
    }
}

/// `plane += coeff N_a N_b value(lmn_a, lmn_b)` over every Cartesian component pair, `N` the Cartesian Gaussians'
/// normalization.
#[inline(always)]
fn fill_cartesian_plane<M: MatrixTrait<usize>>(
    plane: &mut arrays::matrix::MatrixViewMut<'_, f64>,
    cart_a_comps: &M,
    cart_b_comps: &M,
    a: f64,
    b: f64,
    coeff: f64,
    value: impl Fn(&[usize], &[usize]) -> f64,
) {
    for (xb, lmn_b) in plane.iter_col_mut().zip(cart_b_comps.iter_col()) {
        let norm_b = coeff * cart2sph::compute_cart_gaussian_norm(b, lmn_b[0], lmn_b[1], lmn_b[2]);
        for (xa, lmn_a) in xb.iter_mut().zip(cart_a_comps.iter_col()) {
            let norm = norm_b * cart2sph::compute_cart_gaussian_norm(a, lmn_a[0], lmn_a[1], lmn_a[2]);
            *xa += norm * value(lmn_a, lmn_b);
        }
    }
}

#[cfg(test)]
mod screened_coulomb_tests {
    // The AIMP screens' kernels (`CoulombKernel`, `CoulombGradientKernel` in `ScreenMode`) against an oracle that
    // shares nothing with them: 1/r = (2/sqrt(pi)) Int_0^inf exp(-u^2 r^2) du makes each screening Gaussian, for
    // every u, an s Gaussian of exponent c + u^2 at C, so the integral is a three-centre overlap, a product of 1D
    // integrals done exactly from the Gaussian moments; only the u integral is numerical (Gauss-Legendre after
    // u = t / (1 - t), smooth up to t = 1). The derivatives come from the Gaussians' own: d/dA of
    // (x - A)^i exp(-a (x - A)^2) is 2a (x - A)^(i+1) e - i (x - A)^(i-1) e, and d/dC of the screen 2 gamma (x - C) e.

    use super::*;
    use arrays::matrix::MatrixViewMut;
    use arrays::scratch::{Scratch, ByteUnit};
    use basis::charge_carriers::{ScreenedChargeTrait, ScreenedCoulombPotential, SiteTrait};
    use crate::gradient_kernels::{CoulombGradientKernel, GradientArguments, OneElectronIntegralGradientKernel};

    struct Screen {
        position: [f64; 3],
        charge: f64,
        screen: ScreenedCoulombPotential,
    }
    impl SiteTrait for Screen {
        fn get_position(&self) -> &[f64; 3] {
            &self.position
        }
    }
    impl PointChargeTrait for Screen {
        fn get_charge(&self) -> f64 {
            self.charge
        }
    }
    impl ScreenedChargeTrait for Screen {
        fn get_screened_coulomb_potential(&self) -> &ScreenedCoulombPotential {
            &self.screen
        }
    }

    /// The highest angular momentum tested (g).
    const L_MAX: usize = 4;

    /// Two AIMPs, each a screen of four Gaussians from diffuse to tight: one at `position`, one of another charge
    /// elsewhere.
    fn sites(position: [f64; 3]) -> [Screen; 2] {
        let screen = || ScreenedCoulombPotential::new(vec![0.05, 0.9, 7.5, 60.0], vec![-1.3, 0.4, -0.8, 0.25]);
        [
            Screen { position, charge: 2.0, screen: screen() },
            Screen { position: [position[1], -position[2], 0.6 * position[0]], charge: -1.5, screen: screen() },
        ]
    }

    /// (A, B, the first AIMP): a site off the pair's axis, one close to the product centre, one far away, and a
    /// one-centre pair.
    const GEOMETRIES: [([f64; 3], [f64; 3], [f64; 3]); 4] = [
        ([0.0, 0.0, 0.0], [0.3, -1.1, 0.8], [1.4, 0.6, -0.9]),
        ([0.2, 0.1, -0.3], [-0.4, 0.9, 0.5], [0.05, 0.42, 0.11]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.7], [6.0, -7.5, 3.2]),
        ([0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.7, -0.2, 0.4]),
    ];

    /// The pair's exponents (a, b): balanced, diffuse with tight, tight with diffuse.
    const EXPONENTS: [(f64, f64); 3] = [(1.3, 0.7), (0.08, 12.0), (35.0, 0.25)];

    /// Gauss-Legendre nodes and weights on [lower, upper].
    fn gauss_legendre(n: usize, lower: f64, upper: f64) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
                let mut derivative = 0.0;
                for _ in 0..100 {
                    let (mut p0, mut p1) = (1.0, x);
                    for k in 2..=n {
                        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                        p0 = p1;
                        p1 = p2;
                    }
                    derivative = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                    let step = p1 / derivative;
                    x -= step;
                    if step.abs() < 1e-16 {
                        break;
                    }
                }
                let weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
                let half = 0.5 * (upper - lower);
                (lower + half * (x + 1.0), half * weight)
            })
            .collect()
    }

    fn binomial(n: usize, k: usize) -> f64 {
        (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
    }

    /// Along one axis, `Int (x - A)^i (x - B)^j (x - C)^k exp(-a (x-A)^2 - b (x-B)^2 - g (x-C)^2) dx` for every
    /// `i, j <= L_MAX + 1`, `k <= 1`, at `[i][j][k]`: the Gaussians combined about their centre X, the polynomial
    /// expanded about X, the moments `Int y^m exp(-G y^2) dy` exact.
    fn overlaps_1d(a: f64, xa: f64, b: f64, xb: f64, g: f64, xc: f64) -> [[[f64; 2]; L_MAX + 2]; L_MAX + 2] {
        let total = a + b + g;
        let x = (a * xa + b * xb + g * xc) / total;
        let prefactor = (-(a * b * (xa - xb).powi(2) + a * g * (xa - xc).powi(2) + b * g * (xb - xc).powi(2)) / total).exp();
        let n_moments = 2 * (L_MAX + 1) + 2;
        let mut moments = vec![0.0; n_moments + 1];
        // m_0 = sqrt(pi / G), m_(k+2) = (k + 1) / (2 G) m_k, odd moments zero.
        moments[0] = (std::f64::consts::PI / total).sqrt();
        for k in (2..=n_moments).step_by(2) {
            moments[k] = moments[k - 2] * (k - 1) as f64 / (2.0 * total);
        }
        let (da, db, dc) = (x - xa, x - xb, x - xc);
        let mut out = [[[0.0; 2]; L_MAX + 2]; L_MAX + 2];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                for (k, value) in cell.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for ki in 0..=i {
                        for kj in 0..=j {
                            for kk in 0..=k {
                                let m = ki + kj + kk;
                                if m % 2 == 0 {
                                    sum += binomial(i, ki) * da.powi((i - ki) as i32)
                                        * binomial(j, kj) * db.powi((j - kj) as i32)
                                        * binomial(k, kk) * dc.powi((k - kk) as i32)
                                        * moments[m];
                                }
                            }
                        }
                    }
                    *value = prefactor * sum;
                }
            }
        }
        out
    }

    /// The oracle's integrals of one pair of shells (Cartesian, normalized, column-major `na x nb`): the value, and
    /// the x, y, z planes of d/dA, d/dB and of d/dC for each site.
    struct Oracle {
        value: Vec<f64>,
        d_a: Vec<f64>,
        d_b: Vec<f64>,
        d_c: Vec<Vec<f64>>,
    }

    /// The oracle for every pair of shells up to `L_MAX`, at `[la][lb]`.
    fn oracle(a: f64, b: f64, pos_a: [f64; 3], pos_b: [f64; 3], sites: &[Screen]) -> Vec<Vec<Oracle>> {
        let carts: Vec<_> = (0..=L_MAX).map(cart2sph::get_cartesian_quantum_numbers_m).collect();
        let mut out: Vec<Vec<Oracle>> = (0..=L_MAX)
            .map(|la| {
                (0..=L_MAX)
                    .map(|lb| {
                        let n = carts[la].cols() * carts[lb].cols();
                        Oracle { value: vec![0.0; n], d_a: vec![0.0; 3 * n], d_b: vec![0.0; 3 * n], d_c: vec![vec![0.0; 3 * n]; sites.len()] }
                    })
                    .collect()
            })
            .collect();
        // u = t / (1 - t) on panels of t, du = dt / (1 - t)^2.
        let nodes: Vec<(f64, f64)> = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.92), (0.92, 0.98), (0.98, 1.0)]
            .iter()
            .flat_map(|&(lower, upper)| gauss_legendre(64, lower, upper))
            .map(|(t, w)| (t / (1.0 - t), w / (1.0 - t).powi(2)))
            .collect();
        let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
        for (site_index, site) in sites.iter().enumerate() {
            let screen = site.get_screened_coulomb_potential();
            for (&c, &coefficient) in screen.get_exponents().iter().zip(screen.get_coefficients()) {
                for &(u, w) in &nodes {
                    let g = c + u * u;
                    let axes: Vec<_> = (0..3).map(|k| overlaps_1d(a, pos_a[k], b, pos_b[k], g, site.position[k])).collect();
                    let weight = -site.charge * coefficient * two_over_sqrt_pi * w;
                    for la in 0..=L_MAX {
                        for lb in 0..=L_MAX {
                            let (ca, cb) = (&carts[la], &carts[lb]);
                            let (na, nb) = (ca.cols(), cb.cols());
                            let target = &mut out[la][lb];
                            for (jb, nb_lmn) in cb.iter_col().enumerate() {
                                for (ia, na_lmn) in ca.iter_col().enumerate() {
                                    let norm = weight * cart2sph::compute_cart_gaussian_norm(a, na_lmn[0], na_lmn[1], na_lmn[2])
                                        * cart2sph::compute_cart_gaussian_norm(b, nb_lmn[0], nb_lmn[1], nb_lmn[2]);
                                    let s = |k: usize, i: usize, j: usize, kc: usize| axes[k][i][j][kc];
                                    let plain = |k: usize| s(k, na_lmn[k], nb_lmn[k], 0);
                                    let at = ia + na * jb;
                                    target.value[at] += norm * plain(0) * plain(1) * plain(2);
                                    for k in 0..3 {
                                        let (i, j) = (na_lmn[k], nb_lmn[k]);
                                        let others: f64 = (0..3).filter(|&m| m != k).map(plain).product();
                                        let d_a = 2.0 * a * s(k, i + 1, j, 0) - if i > 0 { i as f64 * s(k, i - 1, j, 0) } else { 0.0 };
                                        let d_b = 2.0 * b * s(k, i, j + 1, 0) - if j > 0 { j as f64 * s(k, i, j - 1, 0) } else { 0.0 };
                                        let d_c = 2.0 * g * s(k, i, j, 1);
                                        let plane = at + na * nb * k;
                                        target.d_a[plane] += norm * d_a * others;
                                        target.d_b[plane] += norm * d_b * others;
                                        target.d_c[site_index][plane] += norm * d_c * others;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The largest difference relative to the largest reference value.
    fn relative(reference: &[f64], candidate: &[f64]) -> f64 {
        let size = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        reference.iter().zip(candidate).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs())) / size.max(1e-300)
    }

    /// The kernels against the oracle for every pair of l up to g, two AIMPs at once: the values and the derivatives
    /// with respect to A, B and each AIMP; and the translational sum rule d/dA + d/dB + Sum_C d/dC = 0 of the
    /// kernels themselves.
    #[test]
    fn the_screen_kernels_equal_the_quadrature_of_the_overlaps() {
        let (mut largest_value, mut largest_derivative, mut largest_sum): (f64, f64, f64) = (0.0, 0.0, 0.0);
        for &(pos_a, pos_b, pos_c) in &GEOMETRIES {
            for &(a, b) in &EXPONENTS {
                let sites = sites(pos_c);
                let sites = &sites[..];
                let reference = oracle(a, b, pos_a, pos_b, sites);
                let p = a + b;
                let ab = a * b / p;
                let dist_ab = [pos_b[0] - pos_a[0], pos_b[1] - pos_a[1], pos_b[2] - pos_a[2]];
                let pos_p = [(a * pos_a[0] + b * pos_b[0]) / p, (a * pos_a[1] + b * pos_b[1]) / p, (a * pos_a[2] + b * pos_b[2]) / p];
                for la in 0..=L_MAX {
                    for lb in 0..=L_MAX {
                        let case = format!("l {la} {lb}, exponents {a} {b}, C {pos_c:?}");
                        let mut scratch = Scratch::new(16, ByteUnit::MiB).unwrap();
                        let arena = scratch.get_arena();
                        let cart_a = cart2sph::get_cartesian_quantum_numbers_m(la);
                        let cart_b = cart2sph::get_cartesian_quantum_numbers_m(lb);
                        let (na, nb) = (cart_a.cols(), cart_b.cols());
                        let expected = &reference[la][lb];

                        let mut value = vec![0.0; na * nb];
                        let mut buffer = CoulombKernel::<Screen, ScreenMode>::kernel_buffer_allocation(la, lb, &arena, &sites).unwrap();
                        CoulombKernel::<Screen, ScreenMode>::compute(&mut MatrixViewMut::new(na, nb, &mut value[..]), &mut buffer, a, b, p, ab, &cart_a, &cart_b, &dist_ab, &pos_p, &sites, 1.0);
                        let difference = relative(&expected.value, &value);
                        largest_value = largest_value.max(difference);
                        assert!(difference < 1e-11, "value, {case}: {difference:e}");

                        let mut buffer = CoulombGradientKernel::<Screen, ScreenMode>::kernel_buffer_allocation(la, lb, &arena, &sites).unwrap();
                        let mut planes = |arguments: GradientArguments| {
                            let mut out = vec![0.0; 3 * na * nb];
                            CoulombGradientKernel::<Screen, ScreenMode>::compute(&mut arrays::array3d::Array3DViewMut::new(na, nb, 3, &mut out[..]), &mut buffer, a, b, p, ab, &cart_a, &cart_b, &dist_ab, &pos_p, &sites, &arguments, 1.0);
                            out
                        };
                        let mut sum = vec![0.0; 3 * na * nb];
                        let mut compare = |what: String, expected: &[f64], got: Vec<f64>| {
                            let difference = relative(expected, &got);
                            largest_derivative = largest_derivative.max(difference);
                            assert!(difference < 1e-10, "{what}, {case}: {difference:e}");
                            sum.iter_mut().zip(&got).for_each(|(s, g)| *s += g);
                        };
                        compare("d/dA".into(), &expected.d_a, planes(GradientArguments::new(true, false, None)));
                        compare("d/dB".into(), &expected.d_b, planes(GradientArguments::new(false, true, None)));
                        for (index, expected) in expected.d_c.iter().enumerate() {
                            compare(format!("d/dC_{index}"), expected, planes(GradientArguments::new(false, false, Some(index))));
                        }
                        let scale = expected.d_a.iter().chain(&expected.d_b).fold(0.0_f64, |m, v| m.max(v.abs()));
                        let residual = sum.iter().fold(0.0_f64, |m, v| m.max(v.abs())) / scale.max(1e-300);
                        largest_sum = largest_sum.max(residual);
                        assert!(residual < 1e-12, "sum rule, {case}: {residual:e}");
                    }
                }
            }
        }
        eprintln!("largest relative differences: values {largest_value:e}, derivatives {largest_derivative:e}, sum rule {largest_sum:e}");
    }

    /// The picture-change integrals of one pair of shells by the same quadrature: the scalar p.Vp =
    /// Sum_k <d_k a| V |d_k b> and the three planes `<d_y a| V |d_z b> - <d_z a| V |d_y b>` (x, cyclic), with
    /// d/dx of (x - A)^i exp(-a (x - A)^2) = i (x - A)^(i-1) e - 2a (x - A)^(i+1) e. `terms` are Gaussian-screened
    /// charges `(position, c, weight)`, `weight exp(-c r_C^2) / r_C`; `c = 0` is a bare charge.
    struct PictureChangeOracle {
        pvp: Vec<f64>,
        cross: Vec<f64>,
    }

    fn picture_change_oracle(a: f64, b: f64, pos_a: [f64; 3], pos_b: [f64; 3], terms: &[([f64; 3], f64, f64)]) -> Vec<Vec<PictureChangeOracle>> {
        let carts: Vec<_> = (0..=L_MAX).map(cart2sph::get_cartesian_quantum_numbers_m).collect();
        let mut out: Vec<Vec<PictureChangeOracle>> = (0..=L_MAX)
            .map(|la| (0..=L_MAX).map(|lb| {
                let n = carts[la].cols() * carts[lb].cols();
                PictureChangeOracle { pvp: vec![0.0; n], cross: vec![0.0; 3 * n] }
            }).collect())
            .collect();
        let nodes: Vec<(f64, f64)> = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.92), (0.92, 0.98), (0.98, 1.0)]
            .iter()
            .flat_map(|&(lower, upper)| gauss_legendre(64, lower, upper))
            .map(|(t, w)| (t / (1.0 - t), w / (1.0 - t).powi(2)))
            .collect();
        let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
        for &(position, c, weight) in terms {
            for &(u, w) in &nodes {
                let g = c + u * u;
                let axes: Vec<_> = (0..3).map(|k| overlaps_1d(a, pos_a[k], b, pos_b[k], g, position[k])).collect();
                let scale = weight * two_over_sqrt_pi * w;
                for la in 0..=L_MAX {
                    for lb in 0..=L_MAX {
                        let (ca, cb) = (&carts[la], &carts[lb]);
                        let (na, nb) = (ca.cols(), cb.cols());
                        let target = &mut out[la][lb];
                        for (jb, lmn_b) in cb.iter_col().enumerate() {
                            for (ia, lmn_a) in ca.iter_col().enumerate() {
                                let norm = scale * cart2sph::compute_cart_gaussian_norm(a, lmn_a[0], lmn_a[1], lmn_a[2])
                                    * cart2sph::compute_cart_gaussian_norm(b, lmn_b[0], lmn_b[1], lmn_b[2]);
                                let s = |k: usize, i: usize, j: usize| axes[k][i][j][0];
                                let lower = |k: usize, i: usize, j: usize, on_a: bool| {
                                    let n = if on_a { i } else { j };
                                    if n == 0 { 0.0 } else if on_a { n as f64 * s(k, i - 1, j) } else { n as f64 * s(k, i, j - 1) }
                                };
                                // Per axis: plain, d on a, d on b, d on both.
                                let plain = |k: usize| s(k, lmn_a[k], lmn_b[k]);
                                let d_a = |k: usize| {
                                    let (i, j) = (lmn_a[k], lmn_b[k]);
                                    lower(k, i, j, true) - 2.0 * a * s(k, i + 1, j)
                                };
                                let d_b = |k: usize| {
                                    let (i, j) = (lmn_a[k], lmn_b[k]);
                                    lower(k, i, j, false) - 2.0 * b * s(k, i, j + 1)
                                };
                                let d_ab = |k: usize| {
                                    let (i, j) = (lmn_a[k], lmn_b[k]);
                                    let both_lower = if i > 0 && j > 0 { (i * j) as f64 * s(k, i - 1, j - 1) } else { 0.0 };
                                    let a_lower = if i > 0 { i as f64 * s(k, i - 1, j + 1) } else { 0.0 };
                                    let b_lower = if j > 0 { j as f64 * s(k, i + 1, j - 1) } else { 0.0 };
                                    both_lower - 2.0 * b * a_lower - 2.0 * a * b_lower + 4.0 * a * b * s(k, i + 1, j + 1)
                                };
                                let at = ia + na * jb;
                                let [px, py, pz] = [plain(0), plain(1), plain(2)];
                                target.pvp[at] += norm * (d_ab(0) * py * pz + px * d_ab(1) * pz + px * py * d_ab(2));
                                target.cross[at] += norm * px * (d_a(1) * d_b(2) - d_a(2) * d_b(1));
                                target.cross[at + na * nb] += norm * py * (d_a(2) * d_b(0) - d_a(0) * d_b(2));
                                target.cross[at + 2 * na * nb] += norm * pz * (d_a(0) * d_b(1) - d_a(1) * d_b(0));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The p.Vp kernels (`PvpKernel`, `PvpFullKernel`) against the quadrature, for every pair of l up to g: point
    /// charges (`PointMode`, which pins the planes' convention), the AIMP screens (`ScreenMode`) and the AIMPs' whole
    /// Coulomb potential (`AimpCoulombMode`), two AIMPs at once.
    #[test]
    fn the_picture_change_kernels_equal_the_quadrature_of_the_overlaps() {
        use crate::picture_change_kernels::{PvpFullKernel, PvpKernel};
        #[allow(clippy::too_many_arguments)]
        fn run<M: BuildRtuv3D<Screen> + 'static>(
            label: &str, la: usize, lb: usize, a: f64, b: f64, pos_a: [f64; 3], pos_b: [f64; 3], sites: &[Screen],
            expected: &PictureChangeOracle, largest: &mut f64,
        ) {
            let mut scratch = Scratch::new(16, ByteUnit::MiB).unwrap();
            let arena = scratch.get_arena();
            let p = a + b;
            let ab = a * b / p;
            let dist_ab = [pos_b[0] - pos_a[0], pos_b[1] - pos_a[1], pos_b[2] - pos_a[2]];
            let pos_p = [(a * pos_a[0] + b * pos_b[0]) / p, (a * pos_a[1] + b * pos_b[1]) / p, (a * pos_a[2] + b * pos_b[2]) / p];
            let cart_a = cart2sph::get_cartesian_quantum_numbers_m(la);
            let cart_b = cart2sph::get_cartesian_quantum_numbers_m(lb);
            let (na, nb) = (cart_a.cols(), cart_b.cols());
            let mut scalar = vec![0.0; na * nb];
            let mut buffer = PvpKernel::<Screen, M>::kernel_buffer_allocation(la, lb, &arena, &sites).unwrap();
            PvpKernel::<Screen, M>::compute(&mut MatrixViewMut::new(na, nb, &mut scalar[..]), &mut buffer, a, b, p, ab, &cart_a, &cart_b, &dist_ab, &pos_p, &sites, 1.0);
            let mut full = vec![0.0; 4 * na * nb];
            let mut buffer = PvpFullKernel::<Screen, M>::kernel_buffer_allocation(la, lb, &arena, &sites).unwrap();
            PvpFullKernel::<Screen, M>::compute(&mut arrays::array3d::Array3DViewMut::new(na, nb, 4, &mut full[..]), &mut buffer, a, b, p, ab, &cart_a, &cart_b, &dist_ab, &pos_p, &sites, 1.0);
            let case = format!("{label}, l {la} {lb}, exponents {a} {b}");
            // The planes against the scale of the case's p.Vp: they vanish by symmetry for some pairs (s with s).
            let scale = expected.pvp.iter().chain(&expected.cross).fold(0.0_f64, |m, v| m.max(v.abs()));
            for (what, reference, got) in [
                ("p.Vp", &expected.pvp[..], &scalar[..]),
                ("p.Vp plane of the full kernel", &expected.pvp[..], &full[3 * na * nb..]),
                ("(p x Vp) planes", &expected.cross[..], &full[..3 * na * nb]),
            ] {
                let difference = reference.iter().zip(got).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs())) / scale.max(1e-300);
                *largest = largest.max(difference);
                assert!(difference < 1e-10, "{what}, {case}: {difference:e}");
            }
        }
        let mut largest: f64 = 0.0;
        for &(pos_a, pos_b, pos_c) in &GEOMETRIES {
            for &(a, b) in &EXPONENTS {
                let sites = sites(pos_c);
                let bare: Vec<_> = sites.iter().map(|site| (site.position, 0.0, -site.charge)).collect();
                let screens: Vec<_> = sites
                    .iter()
                    .flat_map(|site| {
                        let screen = site.get_screened_coulomb_potential();
                        screen.get_exponents().iter().zip(screen.get_coefficients()).map(move |(&c, &coefficient)| (site.position, c, -site.charge * coefficient))
                    })
                    .collect();
                let both: Vec<_> = bare.iter().chain(&screens).copied().collect();
                let (bare, screens, both) = (picture_change_oracle(a, b, pos_a, pos_b, &bare), picture_change_oracle(a, b, pos_a, pos_b, &screens), picture_change_oracle(a, b, pos_a, pos_b, &both));
                for la in 0..=L_MAX {
                    for lb in 0..=L_MAX {
                        run::<PointMode>("point charges", la, lb, a, b, pos_a, pos_b, &sites, &bare[la][lb], &mut largest);
                        run::<ScreenMode>("screens", la, lb, a, b, pos_a, pos_b, &sites, &screens[la][lb], &mut largest);
                        run::<AimpCoulombMode>("whole AIMP Coulomb", la, lb, a, b, pos_a, pos_b, &sites, &both[la][lb], &mut largest);
                    }
                }
            }
        }
        eprintln!("largest relative difference {largest:e}");
    }
}
