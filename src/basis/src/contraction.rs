// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The contraction kinds of a basis (general, segmented, uncontracted) and the contraction step of a shell pair.
//! Every kernel computes per primitive product, then contracts to the contracted pairs; only that step depends on the
//! kind. `per_product` is `rows x n_products` (one column per product; any `ColumnFactor`: a matrix, or a slice of an
//! `Array3D` at a fixed middle index), `out` is `rows x n_cont_a n_cont_b`.

use arrays::matrix::*;

use crate::BasisInfo;
use crate::shell_pairs::ShellPair;
use crate::shell_types::{BasisError, BasisShell};
use arrays::vector_math::add_scaled_in_place;

/// The contraction step of one shell pair, resolved at compile time (`Strategy`).
pub trait Contraction {
    /// Every product contributes with unit weight to the one contracted pair (the coefficients are already in E):
    /// kernels then accumulate each product straight into the contracted buffer (GEMM with beta = 1) instead of
    /// storing every product and calling `contract`.
    const ACCUMULATES: bool = false;
    /// There is a contraction step at all: `false` for `Identity`, whose per-product results are the contracted
    /// ones, so kernels skip the step and use the per-product buffer as the result.
    const CONTRACTS: bool = true;
    /// Each product's coefficient (`ShellKind::product_coefficient`) is folded into its Hermite
    /// coefficients E when they are built.
    const PRESCALED: bool = false;
    /// `value` as it enters E for a product with coefficient `coefficient`.
    #[inline(always)]
    fn prescale(value: f64, coefficient: f64) -> f64 {
        if Self::PRESCALED { coefficient * value } else { value }
    }
    /// Contracted functions per shell.
    fn contracted(&self) -> (usize, usize);
    /// Prepares `contract_weighted` for the per-product weights `fraction` (called once per pair and call).
    fn set_weights(&mut self, _fraction: &[f64]) {}
    /// `out = alpha * per_product . C`.
    fn contract(&self, per_product: &impl ColumnFactor, out: &mut impl MatrixTraitMut<f64>, alpha: f64);
    /// `out = alpha * per_product . C[products, :] + beta * out`: a run of the pair's products (`per_product` their
    /// columns, in order), so that a kernel holds a block of them at a time and accumulates the contracted result.
    fn contract_rows(&self, per_product: &impl ColumnFactor, products: std::ops::Range<usize>, out: &mut impl MatrixTraitMut<f64>, alpha: f64, beta: f64);
    /// `out = per_product . diag(fraction) . C`, `fraction` as given to `set_weights`.
    fn contract_weighted(&self, per_product: &impl ColumnFactor, fraction: &[f64], out: &mut impl MatrixTraitMut<f64>);
    /// The transpose of `contract` for one product, `contracted . C[product, :]^T`, `contracted` being
    /// `rows x n_cont_a n_cont_b` (one column per contracted pair): written to `out` (`rows` long) and returned, or a
    /// column of `contracted` itself where that is the product's.
    fn expand<'s>(&self, contracted: &'s [f64], product: usize, out: &'s mut [f64]) -> &'s [f64];
}

/// General contraction: a GEMM with the pair's contraction matrix (borrowed, not copied). The weighted
/// variant uses that matrix with each product row scaled, built once per call.
pub struct Dense<'a> {
    pub(crate) matrix: &'a Matrix<f64>,
    pub(crate) contracted: (usize, usize),
    pub(crate) weighted: Option<Matrix<f64>>,
}

impl Contraction for Dense<'_> {
    #[inline]
    fn contracted(&self) -> (usize, usize) {
        self.contracted
    }
    fn set_weights(&mut self, fraction: &[f64]) {
        let cc = self.matrix;
        let mut weighted = Matrix::<f64>::new(cc.rows(), cc.cols());
        for col in 0..cc.cols() {
            for row in 0..cc.rows() {
                weighted[(row, col)] = fraction[row] * cc[(row, col)];
            }
        }
        self.weighted = Some(weighted);
    }
    fn contract(&self, per_product: &impl ColumnFactor, out: &mut impl MatrixTraitMut<f64>, alpha: f64) {
        per_product.matmul_scaled_prealloc(self.matrix, out, alpha, 0.0);
    }
    fn contract_rows(&self, per_product: &impl ColumnFactor, products: std::ops::Range<usize>, out: &mut impl MatrixTraitMut<f64>, alpha: f64, beta: f64) {
        per_product.matmul_scaled_prealloc(&self.matrix.view_rows(products), out, alpha, beta);
    }
    fn contract_weighted(&self, per_product: &impl ColumnFactor, _fraction: &[f64], out: &mut impl MatrixTraitMut<f64>) {
        per_product.matmul_scaled_prealloc(self.weighted.as_ref().expect("Dense::set_weights precedes contract_weighted"), out, 1.0, 0.0);
    }
    fn expand<'s>(&self, contracted: &'s [f64], product: usize, out: &'s mut [f64]) -> &'s [f64] {
        out.fill(0.0);
        for (c, column) in contracted.chunks_exact(out.len()).enumerate() {
            let coefficient = self.matrix[(product, c)];
            if coefficient != 0.0 {
                add_scaled_in_place(out, column, coefficient);
            }
        }
        out
    }
}

/// Segmented contraction: one contracted function per shell, each product's coefficient already folded
/// into its Hermite coefficients (`PRESCALED`), so contracting is a plain (weighted) sum.
pub struct Summed;

/// `out = sum_p weight(p) * column p`.
fn sum_columns(per_product: &impl ColumnFactor, out: &mut impl MatrixTraitMut<f64>, weight: impl Fn(usize) -> f64) {
    let out = out.get_data_mut();
    for (o, &x) in out.iter_mut().zip(per_product.column(0)) {
        *o = weight(0) * x;
    }
    for p in 1..per_product.shape().1 {
        add_scaled_in_place(out, per_product.column(p), weight(p));
    }
}

impl Contraction for Summed {
    const ACCUMULATES: bool = true;
    const PRESCALED: bool = true;
    #[inline]
    fn contracted(&self) -> (usize, usize) {
        (1, 1)
    }
    fn contract(&self, per_product: &impl ColumnFactor, out: &mut impl MatrixTraitMut<f64>, alpha: f64) {
        sum_columns(per_product, out, |_| alpha);
    }
    fn contract_rows(&self, per_product: &impl ColumnFactor, _products: std::ops::Range<usize>, out: &mut impl MatrixTraitMut<f64>, alpha: f64, beta: f64) {
        let out = out.get_data_mut();
        if beta != 1.0 {
            out.iter_mut().for_each(|o| *o *= beta);
        }
        for p in 0..per_product.shape().1 {
            add_scaled_in_place(out, per_product.column(p), alpha);
        }
    }
    fn contract_weighted(&self, per_product: &impl ColumnFactor, fraction: &[f64], out: &mut impl MatrixTraitMut<f64>) {
        sum_columns(per_product, out, |p| fraction[p]);
    }
    // Every product takes the one contracted pair with unit weight (its coefficient is in E).
    fn expand<'s>(&self, contracted: &'s [f64], _product: usize, out: &'s mut [f64]) -> &'s [f64] {
        &contracted[..out.len()]
    }
}

/// No contraction: each product is its own contracted pair (`contracted` counts them). No matrix is stored and
/// no GEMM is run; kernels skip the step (`CONTRACTS`), and `contract` is a scaled copy for those that do not.
pub struct Identity {
    pub(crate) contracted: (usize, usize),
}

/// `out = weight(p) * column p` for every product `p`.
fn scale_columns(per_product: &impl ColumnFactor, out: &mut impl MatrixTraitMut<f64>, weight: impl Fn(usize) -> f64) {
    let (rows, cols) = per_product.shape();
    let out = out.get_data_mut();
    for p in 0..cols {
        let w = weight(p);
        for (o, &x) in out[p * rows..(p + 1) * rows].iter_mut().zip(per_product.column(p)) {
            *o = w * x;
        }
    }
}

impl Contraction for Identity {
    const CONTRACTS: bool = false;
    #[inline]
    fn contracted(&self) -> (usize, usize) {
        self.contracted
    }
    fn contract(&self, per_product: &impl ColumnFactor, out: &mut impl MatrixTraitMut<f64>, alpha: f64) {
        scale_columns(per_product, out, |_| alpha);
    }
    // Each product is its own contracted pair: the run's columns go to the same columns of `out`.
    fn contract_rows(&self, per_product: &impl ColumnFactor, products: std::ops::Range<usize>, out: &mut impl MatrixTraitMut<f64>, alpha: f64, beta: f64) {
        let rows = per_product.shape().0;
        let out = &mut out.get_data_mut()[products.start * rows..products.end * rows];
        for (p, column) in out.chunks_exact_mut(rows).enumerate() {
            for (o, &x) in column.iter_mut().zip(per_product.column(p)) {
                *o = if beta == 0.0 { alpha * x } else { beta * *o + alpha * x };
            }
        }
    }
    fn contract_weighted(&self, per_product: &impl ColumnFactor, fraction: &[f64], out: &mut impl MatrixTraitMut<f64>) {
        scale_columns(per_product, out, |p| fraction[p]);
    }
    // Each product is its own contracted pair.
    fn expand<'s>(&self, contracted: &'s [f64], product: usize, out: &'s mut [f64]) -> &'s [f64] {
        let rows = out.len();
        &contracted[product * rows..(product + 1) * rows]
    }
}

/// What a pair stores for its contraction step, and the strategy it gives.
pub trait ContractionData: std::fmt::Debug {
    type Strategy<'a>: Contraction where Self: 'a;
    /// The contraction step, for `contracted` functions per shell.
    fn strategy(&self, contracted: (usize, usize)) -> Self::Strategy<'_>;
    /// The contraction matrix, if there is one (`None`: kernels skip the contraction).
    #[inline(always)]
    fn matrix(&self) -> Option<&Matrix<f64>> {
        None
    }
    /// For a pair with screened-out products: the column of the stored product `(a, b)` at `a + n_a b`, `u32::MAX`
    /// if screened out. `None`: every pair is stored, in the order of `shell_pairs::contracted_pair_slot`.
    #[inline(always)]
    fn placement(&self) -> Option<&[u32]> {
        None
    }
}

impl ContractionData for Matrix<f64> {
    type Strategy<'a> = Dense<'a>;
    fn strategy(&self, contracted: (usize, usize)) -> Dense<'_> {
        Dense { matrix: self, contracted, weighted: None }
    }
    #[inline(always)]
    fn matrix(&self) -> Option<&Matrix<f64>> {
        Some(self)
    }
}

/// The contraction step of a pair of kind `S`.
pub type Strategy<'a, S> = <<S as ShellKind>::PairData as ContractionData>::Strategy<'a>;

/// The contraction kind of a basis shell and its pairs (zero-sized marker): what each stores for the contraction.
pub trait ShellKind: Clone + std::fmt::Debug + 'static {
    /// What a shell stores for its contraction.
    type Coefficients: Clone + std::fmt::Debug;
    /// The shell's `(n_prim, n_cont)` coefficient matrix, `None` when uncontracted.
    fn coefficient_matrix(coefficients: &Self::Coefficients) -> Option<&Matrix<f64>>;
    /// The per-primitive coefficients of a shell with one contracted function (segmented).
    #[inline(always)]
    fn summed_coefficients(_coefficients: &Self::Coefficients) -> Option<&[f64]> {
        None
    }
    /// What a pair stores for its contraction step.
    type PairData: ContractionData + 'static;
    /// A pair's primitive product.
    type Product: crate::gto_products::GTOProductTrait + std::fmt::Debug;
    /// The product's coefficient, folded into E when the strategy is `PRESCALED`; not used otherwise.
    #[inline(always)]
    fn product_coefficient(_product: &Self::Product) -> f64 {
        0.0
    }
    /// Contracted functions of a shell.
    fn n_cont(coefficients: &Self::Coefficients, n_prim: usize) -> usize;
    /// (exponent, coefficient) of every primitive of contracted function `cont`, zero coefficients left out.
    fn contracted_primitives<'s>(exponents: &'s [f64], coefficients: &'s Self::Coefficients, cont: usize) -> impl Iterator<Item = (f64, f64)> + 's;
    /// The `(n_prim, n_cont)` coefficients contracting the primitive AO amplitudes, `None` when uncontracted.
    fn amplitude_coefficients(coefficients: &Self::Coefficients, n_prim: usize) -> Option<MatrixView<'_, f64>>;
    /// The pair of shells `a` and `b`, `None` if every product is screened out by `overlap_threshold`.
    fn form_shell_pair(a: &BasisShell<Self>, b: &BasisShell<Self>, single_centre: bool, overlap_threshold: Option<f64>) -> Option<ShellPair<Self>>;
    /// The shells of this kind from generally contracted ones.
    fn reduce(general_shells: &[BasisShell<Contracted>]) -> Vec<BasisShell<Self>>;
    /// The shells are their own primitive basis.
    const PRIMITIVE: bool = false;
    /// Every shell has exactly one contracted function, so every pair has one slot (slot 0).
    const ONE_CONTRACTED: bool = false;
    /// The primitive blocks of `shells` and the contraction matrix (rows: primitive AOs, columns: AOs).
    fn decontract(shells: &[BasisShell<Self>], basis_info: &BasisInfo) -> Result<(Box<[BasisShell<Uncontracted>]>, Matrix<f64>), BasisError>;
    /// The primitive blocks `decontract` gives, without the contraction matrix and before their first AOs are set;
    /// the shells themselves for a primitive kind.
    fn primitive_blocks(shells: &[BasisShell<Self>]) -> Vec<BasisShell<Uncontracted>>;
    /// The pairs and shells of a primitive kind as uncontracted ones.
    fn as_uncontracted<'s>(
        _pairs: &'s [ShellPair<Self>],
        _shells: &'s [BasisShell<Self>],
    ) -> Option<(&'s [ShellPair<Uncontracted>], &'s [BasisShell<Uncontracted>])> {
        None
    }
}

/// A contracted basis (general contraction): shells store their `(n_prim, n_cont)` coefficient matrix, pairs
/// their contraction matrix (`Dense`).
#[derive(Clone, Debug)]
pub struct Contracted;

impl ShellKind for Contracted {
    type Coefficients = std::rc::Rc<Matrix<f64>>;
    type PairData = Matrix<f64>;
    type Product = crate::gto_products::GTOProduct;
    #[inline(always)]
    fn coefficient_matrix(coefficients: &Self::Coefficients) -> Option<&Matrix<f64>> {
        Some(coefficients)
    }
    fn n_cont(coefficients: &Self::Coefficients, _n_prim: usize) -> usize {
        coefficients.cols()
    }
    fn contracted_primitives<'s>(exponents: &'s [f64], coefficients: &'s Self::Coefficients, cont: usize) -> impl Iterator<Item = (f64, f64)> + 's {
        exponents.iter().copied().zip(coefficients.get_col(cont).iter().copied()).filter(|&(_, c)| c != 0.0)
    }
    fn amplitude_coefficients(coefficients: &Self::Coefficients, n_prim: usize) -> Option<MatrixView<'_, f64>> {
        Some(MatrixView::new(n_prim, coefficients.cols(), coefficients.get_data()))
    }
    fn form_shell_pair(a: &BasisShell<Self>, b: &BasisShell<Self>, single_centre: bool, overlap_threshold: Option<f64>) -> Option<ShellPair<Self>> {
        a.form_pair(b, single_centre, overlap_threshold)
    }
    fn reduce(general_shells: &[BasisShell<Contracted>]) -> Vec<BasisShell<Self>> {
        general_shells.to_vec()
    }
    fn decontract(shells: &[BasisShell<Self>], basis_info: &BasisInfo) -> Result<(Box<[BasisShell<Uncontracted>]>, Matrix<f64>), BasisError> {
        crate::shell_types::decontract_general_shells(shells, basis_info)
    }
    fn primitive_blocks(shells: &[BasisShell<Self>]) -> Vec<BasisShell<Uncontracted>> {
        BasisShell::<Uncontracted>::blocks_of(shells)
    }
}

/// An uncontracted basis, every primitive its own function: nothing to contract (`Identity`).
#[derive(Clone, Debug)]
pub struct Uncontracted;

impl ShellKind for Uncontracted {
    type Coefficients = ();
    type PairData = NoContraction;
    type Product = crate::gto_products::GTOProduct;
    #[inline(always)]
    fn coefficient_matrix(_coefficients: &()) -> Option<&Matrix<f64>> {
        None
    }
    fn n_cont(_coefficients: &(), n_prim: usize) -> usize {
        n_prim
    }
    fn contracted_primitives<'s>(exponents: &'s [f64], _coefficients: &'s (), cont: usize) -> impl Iterator<Item = (f64, f64)> + 's {
        std::iter::once((exponents[cont], 1.0))
    }
    fn amplitude_coefficients(_coefficients: &(), _n_prim: usize) -> Option<MatrixView<'_, f64>> {
        None
    }
    fn form_shell_pair(a: &BasisShell<Self>, b: &BasisShell<Self>, single_centre: bool, overlap_threshold: Option<f64>) -> Option<ShellPair<Self>> {
        a.form_pair(b, single_centre, overlap_threshold)
    }
    fn reduce(general_shells: &[BasisShell<Contracted>]) -> Vec<BasisShell<Self>> {
        BasisShell::<Uncontracted>::blocks_of(general_shells)
    }
    const PRIMITIVE: bool = true;
    fn decontract(_shells: &[BasisShell<Self>], _basis_info: &BasisInfo) -> Result<(Box<[BasisShell<Uncontracted>]>, Matrix<f64>), BasisError> {
        unreachable!("an uncontracted basis is its own primitive basis")
    }
    fn primitive_blocks(shells: &[BasisShell<Self>]) -> Vec<BasisShell<Uncontracted>> {
        shells.to_vec()
    }
    fn as_uncontracted<'s>(
        pairs: &'s [ShellPair<Self>],
        shells: &'s [BasisShell<Self>],
    ) -> Option<(&'s [ShellPair<Uncontracted>], &'s [BasisShell<Uncontracted>])> {
        Some((pairs, shells))
    }
}

/// Nothing to contract (`Identity`); only, for a pair with screened-out products, where the stored ones sit.
#[derive(Debug, Default)]
pub struct NoContraction {
    pub(crate) placement: Option<Box<[u32]>>,
}

impl ContractionData for NoContraction {
    type Strategy<'a> = Identity;
    fn strategy(&self, contracted: (usize, usize)) -> Identity {
        Identity { contracted }
    }
    #[inline(always)]
    fn placement(&self) -> Option<&[u32]> {
        self.placement.as_deref()
    }
}

/// A segmented basis: one contracted function per shell, a coefficient per primitive. Each pair product carries the
/// product of its coefficients, folded into E (`Summed`).
#[derive(Clone, Debug)]
pub struct Segmented;

impl ShellKind for Segmented {
    const ONE_CONTRACTED: bool = true;
    type Coefficients = std::rc::Rc<[f64]>;
    type PairData = SummedPair;
    type Product = crate::gto_products::CGTOProduct;
    #[inline(always)]
    fn coefficient_matrix(_coefficients: &Self::Coefficients) -> Option<&Matrix<f64>> {
        None
    }
    #[inline(always)]
    fn summed_coefficients(coefficients: &Self::Coefficients) -> Option<&[f64]> {
        Some(coefficients)
    }
    #[inline(always)]
    fn product_coefficient(product: &Self::Product) -> f64 {
        crate::gto_products::ContractedGTOProductTrait::get_cc(product)
    }
    fn n_cont(_coefficients: &Self::Coefficients, _n_prim: usize) -> usize {
        1
    }
    fn contracted_primitives<'s>(exponents: &'s [f64], coefficients: &'s Self::Coefficients, _cont: usize) -> impl Iterator<Item = (f64, f64)> + 's {
        exponents.iter().copied().zip(coefficients.iter().copied()).filter(|&(_, c)| c != 0.0)
    }
    fn amplitude_coefficients(coefficients: &Self::Coefficients, n_prim: usize) -> Option<MatrixView<'_, f64>> {
        Some(MatrixView::new(n_prim, 1, coefficients))
    }
    fn form_shell_pair(a: &BasisShell<Self>, b: &BasisShell<Self>, single_centre: bool, overlap_threshold: Option<f64>) -> Option<ShellPair<Self>> {
        a.form_pair(b, single_centre, overlap_threshold)
    }
    fn reduce(general_shells: &[BasisShell<Contracted>]) -> Vec<BasisShell<Self>> {
        general_shells.iter().flat_map(BasisShell::<Segmented>::segments_of).collect()
    }
    fn decontract(shells: &[BasisShell<Self>], basis_info: &BasisInfo) -> Result<(Box<[BasisShell<Uncontracted>]>, Matrix<f64>), BasisError> {
        crate::shell_types::decontract_segmented_shells(shells, basis_info)
    }
    fn primitive_blocks(shells: &[BasisShell<Self>]) -> Vec<BasisShell<Uncontracted>> {
        crate::shell_types::segmented_primitive_blocks(shells)
    }
}

/// Nothing stored for the contraction step: the coefficients are in the products (`Summed`).
#[derive(Debug, Default)]
pub struct SummedPair;

impl ContractionData for SummedPair {
    type Strategy<'a> = Summed;
    fn strategy(&self, _contracted: (usize, usize)) -> Summed {
        Summed
    }
}
