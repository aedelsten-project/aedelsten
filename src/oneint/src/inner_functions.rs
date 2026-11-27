// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Inner functions: the integral batch of one pair of shells, as a Cartesian loop over primitive products, the
// contraction, and the spherical transformation of each contracted slice. Two sources of products:
//   - preset shell pairs (ShellPair), products precomputed and screened: the path for every integral within
//     one basis;
//   - two shells (BasisShell), products formed on the fly, each shell contracting its own side: any two shells, e.g.
//     from different basis sets (projection).

use arrays::array3d::{Array3DTrait, Array3DTraitMut};
use arrays::matrix::Factor;
use arrays::matrix::{MatrixTrait};
use arrays::scratch::ArenaError;
use basis::contraction::{Contraction, ShellKind, Strategy};
use basis::gto_products::GTOProductTrait;
use basis::shell_pairs::ShellPair;
use basis::shell_types::{BasisShell, HarmonicType};

use crate::gradient_kernels::{self, ArrayBatchGradientKernel, GradientArguments};
use crate::kernels::{ArrayBatchKernel, OneElectronIntegralKernel};
use crate::spherical_transformation::SliceTransformation;

// --- Over preset shell pairs ---

/// The batch `(comps_a, comps_b, n_cont_a, n_cont_b)` of a matrix kernel.
pub(crate) fn pair_batch<'a, K, S: ShellKind>(
    pair: &ShellPair<S>,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array4d::Array4DViewMut<'a, f64>, ArenaError>
where
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>,
{
    let (n_cont_a, n_cont_b) = pair.get_contracted_pair_dims();
    let finish = PairContraction::new(pair, 1, arena)?;
    let (comps_a, comps_b) = finish.output_comps();
    let cartesian_batch = {
        let _span = instrumentation::fine_span("oneint::primitives");
        pair_cartesian_loop::<K, S>(pair, arena, additional_arguments)?
    };
    let batch = {
        let _span = instrumentation::fine_span("oneint::contract_transform");
        finish.apply(cartesian_batch, pair)
    };
    Ok(arrays::array4d::Array4DViewMut::new(comps_a, comps_b, n_cont_a, n_cont_b, batch))
}

/// The batch `(comps_a, comps_b, n_planes, n_cont_a, n_cont_b)` of a kernel with several planes.
pub(crate) fn pair_batch_planar<'a, K: ArrayBatchKernel, S: ShellKind>(
    pair: &ShellPair<S>,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array5d::Array5DViewMut<'a, f64>, ArenaError> {
    let (n_cont_a, n_cont_b) = pair.get_contracted_pair_dims();
    let n_planes = K::n_planes(additional_arguments);
    let finish = PairContraction::new(pair, n_planes, arena)?;
    let (comps_a, comps_b) = finish.output_comps();
    let cartesian_batch = {
        let _span = instrumentation::fine_span("oneint::primitives");
        pair_cartesian_loop_planar::<K, S>(pair, arena, additional_arguments)?
    };
    let batch = {
        let _span = instrumentation::fine_span("oneint::contract_transform");
        finish.apply(cartesian_batch, pair)
    };
    Ok(arrays::array5d::Array5DViewMut::new(comps_a, comps_b, n_planes, n_cont_a, n_cont_b, batch))
}

/// The derivative batch `(comps_a, comps_b, n_planes, n_cont_a, n_cont_b)` of a gradient kernel.
pub(crate) fn pair_gradient_batch<'a, K: ArrayBatchGradientKernel, S: ShellKind>(
    pair: &ShellPair<S>,
    gradient_arguments: GradientArguments,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array5d::Array5DViewMut<'a, f64>, ArenaError> {
    let (n_cont_a, n_cont_b) = pair.get_contracted_pair_dims();
    let n_planes = K::n_planes(additional_arguments, &gradient_arguments);
    let finish = PairContraction::new(pair, n_planes, arena)?;
    let (comps_a, comps_b) = finish.output_comps();
    let cartesian_batch = {
        let _span = instrumentation::fine_span("oneint::primitives");
        pair_cartesian_loop_gradient_planar::<K, S>(pair, gradient_arguments, arena, additional_arguments)?
    };
    let batch = {
        let _span = instrumentation::fine_span("oneint::contract_transform");
        finish.apply(cartesian_batch, pair)
    };
    Ok(arrays::array5d::Array5DViewMut::new(comps_a, comps_b, n_planes, n_cont_a, n_cont_b, batch))
}

/// Whether the pair's products all accumulate into one batch (`Summed`), and each product's kernel weight.
#[inline(always)]
pub(crate) fn accumulates<C: Contraction>(_contraction: &C) -> bool {
    C::ACCUMULATES
}

#[inline(always)]
pub(crate) fn weight<S: ShellKind>(product: &S::Product) -> f64 {
    <Strategy<'_, S> as Contraction>::prescale(1.0, S::product_coefficient(product))
}

// Contract first, then transform each (plane, contracted pair) slice: cheaper than transforming every product.
// Buffers are allocated before the kernel loop, so an arena overflow does not rerun it.
struct PairContraction<'a> {
    n_planes: usize,
    contracted: Option<arrays::array3d::Array3DViewMut<'a, f64>>,
    transformation: crate::spherical_transformation::SliceTransformation<'a>,
}

impl<'a> PairContraction<'a> {
    fn new<S: ShellKind>(shell_pair: &ShellPair<S>, n_planes: usize, arena: &arrays::scratch::Arena<'a>) -> Result<Self, ArenaError> {
        use basis::shell_types::HarmonicType::Spherical;
        let (lqn_a, lqn_b) = (shell_pair.get_lqn_a(), shell_pair.get_lqn_b());
        let (n_cont_a, n_cont_b) = shell_pair.get_contracted_pair_dims();
        let n_slices = n_planes * n_cont_a * n_cont_b;
        let contraction = shell_pair.contraction();
        let contracted = match contracts(&contraction) && !accumulates(&contraction) {
            true => Some(arrays::array3d::Array3DViewMut::<f64>::from_arena(
                cart2sph::compute_cartesian_degeneracy(lqn_a), cart2sph::compute_cartesian_degeneracy(lqn_b), n_slices, arena)?),
            false => None,
        };
        let transformation = crate::spherical_transformation::SliceTransformation::new(
            lqn_a, shell_pair.get_harmonic_a() == Spherical, lqn_b, shell_pair.get_harmonic_b() == Spherical, n_slices, arena)?;
        Ok(Self { n_planes, contracted, transformation })
    }

    fn output_comps(&self) -> (usize, usize) {
        self.transformation.comps()
    }

    /// `cartesian` is (cart_a, cart_b, planes x slots); returns (comps_a, comps_b, planes x contracted pairs).
    fn apply<S: ShellKind>(self, cartesian: arrays::array3d::Array3DViewMut<'a, f64>, shell_pair: &ShellPair<S>) -> &'a mut [f64] {
        let batch = match self.contracted {
            Some(mut contracted) => {
                let rows = cartesian.dim0() * cartesian.dim1() * self.n_planes;
                let per_product = arrays::matrix::MatrixView::new(rows, cartesian.get_data().len() / rows, cartesian.get_data());
                let n_contracted = contracted.get_data().len() / rows;
                shell_pair.contraction().contract(&per_product, &mut arrays::matrix::MatrixViewMut::new(rows, n_contracted, contracted.get_data_mut()), 1.0);
                contracted
            }
            None => cartesian,
        };
        self.transformation.apply(batch)
    }
}

#[inline(always)]
fn contracts<C: Contraction>(_contraction: &C) -> bool {
    C::CONTRACTS
}

/// Every product into its own slot, or all into one (`Summed`, weighted by the product coefficient).
fn pair_cartesian_loop<'a, K, S: ShellKind>(
    shell_pair: &ShellPair<S>,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array3d::Array3DViewMut<'a, f64>, ArenaError>
where
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>
{
    // A
    let lqn_a = shell_pair.get_lqn_a();
    let cart_size_a = cart2sph::compute_cartesian_degeneracy(lqn_a);
    let cart_comps_a = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_a, arena)?;
    // B
    let lqn_b = shell_pair.get_lqn_b();
    let cart_size_b = cart2sph::compute_cartesian_degeneracy(lqn_b);
    let cart_comps_b = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_b, arena)?;
    // Common
    let x_ab = shell_pair.get_separation();
    let contraction = shell_pair.contraction();
    let accumulate = accumulates(&contraction);
    let n_slots = if accumulate { 1 } else { shell_pair.get_products().len() };
    // Batches.
    let mut cartesian_batch = arrays::array3d::Array3DViewMut::<f64>::from_arena(cart_size_a, cart_size_b, n_slots, arena)?;
    let mut kernel_buffer = K::kernel_buffer_allocation(lqn_a, lqn_b, arena, additional_arguments)?;
    for (i, product) in shell_pair.get_products().iter().enumerate() {
        let mut slot = cartesian_batch.view_sub_matrix_mut(if accumulate { 0 } else { i });
        K::compute(
            &mut slot,
            &mut kernel_buffer,
            product.get_a(),
            product.get_b(),
            product.get_p(),
            product.get_μ(),
            &cart_comps_a,
            &cart_comps_b,
            &x_ab,
            &product.get_product_centre(),
            additional_arguments,
            weight::<S>(product),
        );
    }
    Ok(cartesian_batch)
}

/// `pair_cartesian_loop` for kernels with several planes: (cart_a, cart_b, planes x slots).
fn pair_cartesian_loop_planar<'a, K, S: ShellKind>(
    shell_pair: &ShellPair<S>,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array3d::Array3DViewMut<'a, f64>, ArenaError>
where
    K: ArrayBatchKernel,
{
    // A
    let lqn_a = shell_pair.get_lqn_a();
    let cart_size_a = cart2sph::compute_cartesian_degeneracy(lqn_a);
    let cart_comps_a = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_a, arena)?;
    // B
    let lqn_b = shell_pair.get_lqn_b();
    let cart_size_b = cart2sph::compute_cartesian_degeneracy(lqn_b);
    let cart_comps_b = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_b, arena)?;
    // Common
    let x_ab = shell_pair.get_separation();
    let n_planes = K::n_planes(additional_arguments);
    let contraction = shell_pair.contraction();
    let accumulate = accumulates(&contraction);
    let n_slots = if accumulate { 1 } else { shell_pair.get_products().len() };
    let slot_size = cart_size_a * cart_size_b * n_planes;
    // Batches.
    let mut cartesian_batch = arrays::array3d::Array3DViewMut::<f64>::from_arena(cart_size_a, cart_size_b, n_planes * n_slots, arena)?;
    let mut kernel_buffer = K::kernel_buffer_allocation(lqn_a, lqn_b, arena, additional_arguments)?;
    for (i, product) in shell_pair.get_products().iter().enumerate() {
        let offset = if accumulate { 0 } else { i * slot_size };
        let mut slot = arrays::array3d::Array3DViewMut::new(cart_size_a, cart_size_b, n_planes, &mut cartesian_batch.get_data_mut()[offset..offset + slot_size]);
        K::compute(
            &mut slot,
            &mut kernel_buffer,
            product.get_a(),
            product.get_b(),
            product.get_p(),
            product.get_μ(),
            &cart_comps_a,
            &cart_comps_b,
            &x_ab,
            &product.get_product_centre(),
            additional_arguments,
            weight::<S>(product),
        );
    }
    Ok(cartesian_batch)
}

fn pair_cartesian_loop_gradient_planar<'a, K, S: ShellKind>(
    shell_pair: &ShellPair<S>,
    gradient_arguments: crate::gradient_kernels::GradientArguments,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array3d::Array3DViewMut<'a, f64>, ArenaError>
where
    K: gradient_kernels::ArrayBatchGradientKernel,
{
    // A
    let lqn_a = shell_pair.get_lqn_a();
    let cart_size_a = cart2sph::compute_cartesian_degeneracy(lqn_a);
    let cart_comps_a = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_a, arena)?;
    // B
    let lqn_b = shell_pair.get_lqn_b();
    let cart_size_b = cart2sph::compute_cartesian_degeneracy(lqn_b);
    let cart_comps_b = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_b, arena)?;
    // Common
    let x_ab = shell_pair.get_separation();
    let n_planes = K::n_planes(additional_arguments, &gradient_arguments);
    let contraction = shell_pair.contraction();
    let accumulate = accumulates(&contraction);
    let n_slots = if accumulate { 1 } else { shell_pair.get_products().len() };
    let slot_size = cart_size_a * cart_size_b * n_planes;
    // Batches.
    let mut cartesian_batch = arrays::array3d::Array3DViewMut::<f64>::from_arena(cart_size_a, cart_size_b, n_planes * n_slots, arena)?;
    let mut kernel_buffer = K::kernel_buffer_allocation(lqn_a, lqn_b, arena, additional_arguments)?;
    for (i, product) in shell_pair.get_products().iter().enumerate() {
        let offset = if accumulate { 0 } else { i * slot_size };
        let mut slot = arrays::array3d::Array3DViewMut::new(cart_size_a, cart_size_b, n_planes, &mut cartesian_batch.get_data_mut()[offset..offset + slot_size]);
        K::compute(
            &mut slot,
            &mut kernel_buffer,
            product.get_a(),
            product.get_b(),
            product.get_p(),
            product.get_μ(),
            &cart_comps_a,
            &cart_comps_b,
            &x_ab,
            &product.get_product_centre(),
            additional_arguments,
            &gradient_arguments,
            weight::<S>(product),
        );
    }
    Ok(cartesian_batch)
}

// --- Over two shells ---
//
// Shells of any contraction kinds. With no pair to carry the contraction, each side brings its own: a segmented shell
// sums its primitives into one slot (its coefficients weight the kernel), a general shell contracts one slot per
// primitive with its coefficient matrix, an uncontracted one keeps them.

/// Slots per side, and each primitive's slot and kernel weight.
#[derive(Clone, Copy)]
struct Side<'s> {
    summed: Option<&'s [f64]>,
    n_slots: usize,
}

impl<'s> Side<'s> {
    fn of<S: ShellKind>(shell: &'s BasisShell<S>) -> Self {
        let summed = shell.summed_coefficients();
        Self { summed, n_slots: if summed.is_some() { 1 } else { shell.get_exponents().len() } }
    }
    #[inline(always)]
    fn slot(&self, primitive: usize) -> usize {
        if self.summed.is_some() { 0 } else { primitive }
    }
    #[inline(always)]
    fn weight(&self, primitive: usize) -> f64 {
        self.summed.map_or(1.0, |c| c[primitive])
    }
}

/// (comps_a, comps_b, n_cont_a, n_cont_b).
pub(crate) fn two_shell_batch<'a, K, Sa: ShellKind, Sb: ShellKind>(
    shell_a: &BasisShell<Sa>,
    shell_b: &BasisShell<Sb>,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array4d::Array4DViewMut<'a, f64>, ArenaError>
where
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>,
{
    let finish = TwoShellContraction::new(shell_a, shell_b, 1, arena)?;
    let (comps_a, comps_b) = finish.transformation.comps();
    let cartesian_batch = {
        let _span = instrumentation::fine_span("oneint::primitives");
        two_shell_cartesian_loop::<K, Sa, Sb>(shell_a, shell_b, arena, additional_arguments)?
    };
    let batch = {
        let _span = instrumentation::fine_span("oneint::contract_transform");
        finish.apply(cartesian_batch, shell_a, shell_b)
    };
    Ok(arrays::array4d::Array4DViewMut::new(comps_a, comps_b, shell_a.get_n_cont(), shell_b.get_n_cont(), batch))
}

/// (comps_a, comps_b, n_planes, n_cont_a, n_cont_b).
pub(crate) fn two_shell_gradient_batch<'a, K, Sa: ShellKind, Sb: ShellKind>(
    shell_a: &BasisShell<Sa>,
    shell_b: &BasisShell<Sb>,
    centre_gradient_idx: usize,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array5d::Array5DViewMut<'a, f64>, ArenaError>
where
    K: ArrayBatchGradientKernel,
{
    let gradient_arguments = GradientArguments::new(
        shell_a.get_centre_index() == centre_gradient_idx,
        shell_b.get_centre_index() == centre_gradient_idx,
        None,
    );
    let n_planes = K::n_planes(additional_arguments, &gradient_arguments);
    let finish = TwoShellContraction::new(shell_a, shell_b, n_planes, arena)?;
    let (comps_a, comps_b) = finish.transformation.comps();
    let cartesian_batch = {
        let _span = instrumentation::fine_span("oneint::primitives");
        two_shell_cartesian_loop_gradient::<K, Sa, Sb>(shell_a, shell_b, &gradient_arguments, n_planes, arena, additional_arguments)?
    };
    let batch = {
        let _span = instrumentation::fine_span("oneint::contract_transform");
        finish.apply(cartesian_batch, shell_a, shell_b)
    };
    Ok(arrays::array5d::Array5DViewMut::new(comps_a, comps_b, n_planes, shell_a.get_n_cont(), shell_b.get_n_cont(), batch))
}

// Contract side a, then side b, then transform each (plane, contracted pair) slice; buffers allocated before the
// kernel loop.
struct TwoShellContraction<'a> {
    block: usize,
    contracted_a: Option<arrays::array3d::Array3DViewMut<'a, f64>>,
    contracted_b: Option<arrays::array3d::Array3DViewMut<'a, f64>>,
    transformation: SliceTransformation<'a>,
}

impl<'a> TwoShellContraction<'a> {
    fn new<Sa: ShellKind, Sb: ShellKind>(shell_a: &BasisShell<Sa>, shell_b: &BasisShell<Sb>, n_planes: usize, arena: &arrays::scratch::Arena<'a>) -> Result<Self, ArenaError> {
        let (lqn_a, lqn_b) = (shell_a.get_lqn(), shell_b.get_lqn());
        let (cart_a, cart_b) = (cart2sph::compute_cartesian_degeneracy(lqn_a), cart2sph::compute_cartesian_degeneracy(lqn_b));
        let (n_cont_a, n_cont_b) = (shell_a.get_n_cont(), shell_b.get_n_cont());
        let contracted_a = match shell_a.coefficient_matrix() {
            Some(_) => Some(arrays::array3d::Array3DViewMut::<f64>::from_arena(cart_a, cart_b, n_planes * n_cont_a * Side::of(shell_b).n_slots, arena)?),
            None => None,
        };
        let contracted_b = match shell_b.coefficient_matrix() {
            Some(_) => Some(arrays::array3d::Array3DViewMut::<f64>::from_arena(cart_a, cart_b, n_planes * n_cont_a * n_cont_b, arena)?),
            None => None,
        };
        let transformation = SliceTransformation::new(
            lqn_a, shell_a.get_harmonic_type() == HarmonicType::Spherical,
            lqn_b, shell_b.get_harmonic_type() == HarmonicType::Spherical,
            n_planes * n_cont_a * n_cont_b, arena,
        )?;
        Ok(Self { block: cart_a * cart_b * n_planes, contracted_a, contracted_b, transformation })
    }

    /// `cartesian` is (cart_a, cart_b, planes x slots_a x slots_b).
    fn apply<Sa: ShellKind, Sb: ShellKind>(self, cartesian: arrays::array3d::Array3DViewMut<'a, f64>, shell_a: &BasisShell<Sa>, shell_b: &BasisShell<Sb>) -> &'a mut [f64] {
        let block = self.block;
        let mut batch = cartesian;
        if let (Some(ca), Some(mut contracted)) = (shell_a.coefficient_matrix(), self.contracted_a) {
            let (n_prim_a, n_cont_a) = (ca.rows(), ca.cols());
            for (primitive, out) in batch.get_data().chunks_exact(block * n_prim_a).zip(contracted.get_data_mut().chunks_exact_mut(block * n_cont_a)) {
                arrays::matrix::MatrixView::new(block, n_prim_a, primitive)
                    .matmul_prealloc(ca, &mut arrays::matrix::MatrixViewMut::new(block, n_cont_a, out));
            }
            batch = contracted;
        }
        if let (Some(cb), Some(mut contracted)) = (shell_b.coefficient_matrix(), self.contracted_b) {
            let rows = batch.get_data().len() / cb.rows();
            arrays::matrix::MatrixView::new(rows, cb.rows(), batch.get_data())
                .matmul_prealloc(cb, &mut arrays::matrix::MatrixViewMut::new(rows, cb.cols(), contracted.get_data_mut()));
            batch = contracted;
        }
        self.transformation.apply(batch)
    }
}

fn two_shell_cartesian_loop<'a, K, Sa: ShellKind, Sb: ShellKind>(
    shell_a: &BasisShell<Sa>,
    shell_b: &BasisShell<Sb>,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array3d::Array3DViewMut<'a, f64>, ArenaError>
where
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>,
{
    let (lqn_a, lqn_b) = (shell_a.get_lqn(), shell_b.get_lqn());
    let (pos_a, pos_b) = (shell_a.get_position(), shell_b.get_position());
    let (cart_size_a, cart_size_b) = (cart2sph::compute_cartesian_degeneracy(lqn_a), cart2sph::compute_cartesian_degeneracy(lqn_b));
    let cart_comps_a = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_a, arena)?;
    let cart_comps_b = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_b, arena)?;
    let x_ab = arrays::vector_math::vector_subtract_3d(&pos_b, &pos_a);
    let (side_a, side_b) = (Side::of(shell_a), Side::of(shell_b));
    let mut cartesian_batch = arrays::array3d::Array3DViewMut::<f64>::from_arena(cart_size_a, cart_size_b, side_a.n_slots * side_b.n_slots, arena)?;
    let mut kernel_buffer = K::kernel_buffer_allocation(lqn_a, lqn_b, arena, additional_arguments)?;
    for (j, &b) in shell_b.get_exponents().iter().enumerate() {
        for (i, &a) in shell_a.get_exponents().iter().enumerate() {
            let p = a + b;
            let p_x = basis::gto_products::gaussian_product_centre(&pos_a, a, &pos_b, b, p);
            let μ = basis::gto_products::GTOProduct::new(a, b, p_x).get_μ();
            let mut slot = cartesian_batch.view_sub_matrix_mut(side_a.slot(i) + side_a.n_slots * side_b.slot(j));
            K::compute(
                &mut slot,
                &mut kernel_buffer,
                a,
                b,
                p,
                μ,
                &cart_comps_a,
                &cart_comps_b,
                &x_ab,
                &p_x,
                additional_arguments,
                side_a.weight(i) * side_b.weight(j),
            );
        }
    }
    Ok(cartesian_batch)
}

fn two_shell_cartesian_loop_gradient<'a, K, Sa: ShellKind, Sb: ShellKind>(
    shell_a: &BasisShell<Sa>,
    shell_b: &BasisShell<Sb>,
    gradient_arguments: &GradientArguments,
    n_planes: usize,
    arena: &arrays::scratch::Arena<'a>,
    additional_arguments: &K::AdditionalArguments<'_>,
) -> Result<arrays::array3d::Array3DViewMut<'a, f64>, ArenaError>
where
    K: ArrayBatchGradientKernel,
{
    let (lqn_a, lqn_b) = (shell_a.get_lqn(), shell_b.get_lqn());
    let (pos_a, pos_b) = (shell_a.get_position(), shell_b.get_position());
    let (cart_size_a, cart_size_b) = (cart2sph::compute_cartesian_degeneracy(lqn_a), cart2sph::compute_cartesian_degeneracy(lqn_b));
    let cart_comps_a = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_a, arena)?;
    let cart_comps_b = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_b, arena)?;
    let x_ab = arrays::vector_math::vector_subtract_3d(&pos_b, &pos_a);
    let (side_a, side_b) = (Side::of(shell_a), Side::of(shell_b));
    let slot_size = cart_size_a * cart_size_b * n_planes;
    let mut cartesian_batch = arrays::array3d::Array3DViewMut::<f64>::from_arena(cart_size_a, cart_size_b, n_planes * side_a.n_slots * side_b.n_slots, arena)?;
    let mut kernel_buffer = K::kernel_buffer_allocation(lqn_a, lqn_b, arena, additional_arguments)?;
    for (j, &b) in shell_b.get_exponents().iter().enumerate() {
        for (i, &a) in shell_a.get_exponents().iter().enumerate() {
            let p = a + b;
            let p_x = basis::gto_products::gaussian_product_centre(&pos_a, a, &pos_b, b, p);
            let μ = basis::gto_products::GTOProduct::new(a, b, p_x).get_μ();
            let offset = (side_a.slot(i) + side_a.n_slots * side_b.slot(j)) * slot_size;
            let mut slot = arrays::array3d::Array3DViewMut::new(cart_size_a, cart_size_b, n_planes, &mut cartesian_batch.get_data_mut()[offset..offset + slot_size]);
            K::compute(
                &mut slot,
                &mut kernel_buffer,
                a,
                b,
                p,
                μ,
                &cart_comps_a,
                &cart_comps_b,
                &x_ab,
                &p_x,
                additional_arguments,
                gradient_arguments,
                side_a.weight(i) * side_b.weight(j),
            );
        }
    }
    Ok(cartesian_batch)
}
