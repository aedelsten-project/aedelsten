// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Packed ERI tensors: `(ij|kl)` stored once per permutational class, 8-fold (`S8`) or 4-fold (`S4`), in a row- or
//! column-major pair packing (`IndexOrder`); and `PackedPairTensor`, a set of symmetric matrices over the function
//! pairs.

use arrays::matrix::*;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// How an unordered pair of indices (p >= q) is packed into one triangular index, and a pair of pairs into a
/// triangular (`S8`) or square (`S4`) one: `PackedERITensorStruct`'s `O`, chosen at compile time.
pub trait IndexOrder {
    /// Unordered pair (i, j) -> triangular pair index. `n` is the total
    /// number of orbitals; row-major packings ignore it.
    fn pair_index(i: usize, j: usize, n: usize) -> usize;
    /// Ordered pair-of-pairs (p >= q) -> linear index into `S8`'s
    /// triangular-of-triangular storage. `n_pairs` is the number of AO
    /// pairs; row-major packings ignore it.
    fn combine(p: usize, q: usize, n_pairs: usize) -> usize;
    /// Pair-of-pairs (p, q), with no assumption on their relative order,
    /// -> linear index into `S4`'s full `n_pairs` x `n_pairs` storage.
    fn combine_square(p: usize, q: usize, n_pairs: usize) -> usize;
}

/// Row-major triangular packing: `i (i + 1)/2 + j`, then `p (p + 1)/2 + q` (S8) or `p n_pairs + q` (S4): the
/// ordering of PySCF's `aosym="s8"` and `aosym="s4"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RowMajor;
impl IndexOrder for RowMajor {
    #[inline(always)]
    fn pair_index(i: usize, j: usize, _n: usize) -> usize {
        let (p, q) = if i >= j { (i, j) } else { (j, i) };
        p * (p + 1) / 2 + q
    }

    #[inline(always)]
    fn combine(p: usize, q: usize, _n_pairs: usize) -> usize {
        p * (p + 1) / 2 + q
    }

    #[inline(always)]
    fn combine_square(p: usize, q: usize, n_pairs: usize) -> usize {
        p * n_pairs + q
    }
}

/// Column-major triangular packing, the default: the smaller index selects a block and the larger sweeps it
/// contiguously, so that `(m,0,0,0)` for `m = 0, 1, 2, ...` is at `0, 1, 2, ...`. For S4's square storage, `Matrix`'s
/// convention: the first pair index sweeps a block selected by the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColumnMajor;
impl IndexOrder for ColumnMajor {
    #[inline(always)]
    fn pair_index(i: usize, j: usize, n: usize) -> usize {
        let (p, q) = if i >= j { (i, j) } else { (j, i) };
        q * n - q * q.saturating_sub(1) / 2 + (p - q)
    }

    #[inline(always)]
    fn combine(p: usize, q: usize, n_pairs: usize) -> usize {
        q * n_pairs - q * q.saturating_sub(1) / 2 + (p - q)
    }

    #[inline(always)]
    fn combine_square(p: usize, q: usize, n_pairs: usize) -> usize {
        q * n_pairs + p
    }
}

/// Which permutational symmetry of the ERIs is folded into storage, on top of the `O: IndexOrder` pair packing:
/// `PackedERITensorStruct`'s `S`, chosen at compile time.
pub trait AOSymmetry<O: IndexOrder> {
    /// Total number of stored elements, given `size` (number of orbitals)
    /// and `size_half = size*(size+1)/2` (number of unordered AO pairs).
    fn storage_size(size: usize, size_half: usize) -> usize;
    /// Map the two (already triangularly packed) pair indices to a linear
    /// storage index.
    fn combine_index(ij: usize, kl: usize, n_pairs: usize) -> usize;
    /// Writes `val` to every storage location this symmetry implies for (i, j, k, l): one where `combine_index`
    /// canonicalises bra and ket (`S8`, the default), also the mirrored (k, l, i, j) where they stay distinct (`S4`).
    #[inline(always)]
    fn distribute<T, Tensor>(tensor: &mut Tensor, i: usize, j: usize, k: usize, l: usize, val: T)
    where
        T: Copy,
        Tensor: PackedERITensorTraitMut<T>,
    {
        tensor.set_value(i, j, k, l, val);
    }
}

/// Full 8-fold permutational symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
/// = (kl|ij) = .... Storage is a triangular-of-triangular pack of the
/// `n_pairs` unordered AO pairs, `size_half (size_half + 1)/2` elements. For real integrals over one AO basis, where
/// bra and ket are interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct S8;
impl<O: IndexOrder> AOSymmetry<O> for S8 {
    #[inline(always)]
    fn storage_size(_size: usize, size_half: usize) -> usize {
        size_half * (size_half + 1) / 2
    }

    #[inline(always)]
    fn combine_index(ij: usize, kl: usize, n_pairs: usize) -> usize {
        let (p, q) = if ij >= kl { (ij, kl) } else { (kl, ij) };
        O::combine(p, q, n_pairs)
    }
}

/// 4-fold permutational symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk),
/// but the bra pair (ij) and ket pair (kl) are kept distinct rather than
/// folded onto each other. Storage is the full `n_pairs` x `n_pairs` square
/// of AO pairs: for real integrals over one AO basis every value twice, for bra and ket spaces that are not
/// interchangeable each once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct S4;
impl<O: IndexOrder> AOSymmetry<O> for S4 {
    #[inline(always)]
    fn storage_size(_size: usize, size_half: usize) -> usize {
        size_half * size_half
    }

    #[inline(always)]
    fn combine_index(ij: usize, kl: usize, n_pairs: usize) -> usize {
        O::combine_square(ij, kl, n_pairs)
    }

    #[inline(always)]
    fn distribute<T, Tensor>(tensor: &mut Tensor, i: usize, j: usize, k: usize, l: usize, val: T)
    where
        T: Copy,
        Tensor: PackedERITensorTraitMut<T>,
    {
        tensor.set_value(i, j, k, l, val);
        tensor.set_value(k, l, i, j, val);
    }
}

/// A packed ERI tensor over `size` functions, symmetry `S`, packing `O`, over data `D`: owned or borrowed.
#[derive(Debug, PartialEq, Clone)]
pub struct PackedERITensorStruct<D: DataContainer, O: IndexOrder = ColumnMajor, S: AOSymmetry<O> = S8> {
    size: usize,
    size_half: usize,
    full_size: usize,
    data: D,
    _order: PhantomData<O>,
    _symmetry: PhantomData<S>,
}
/// An owned packed tensor.
pub type PackedERITensor<T, O = ColumnMajor, S = S8> = PackedERITensorStruct<Box<[T]>, O, S>;
/// A view of a packed tensor.
pub type PackedERITensorView<'a, T, O = ColumnMajor, S = S8> = PackedERITensorStruct<&'a [T], O, S>;
/// A mutable view of a packed tensor.
pub type PackedERITensorViewMut<'a, T, O = ColumnMajor, S = S8> = PackedERITensorStruct<&'a mut [T], O, S>;

/// Reading a packed tensor, owned or a view.
pub trait PackedERITensorTrait<T>: Index<(usize, usize, usize, usize), Output = T> {
    /// The number of stored values.
    fn get_size(&self) -> usize;
    /// The stored values.
    fn get_data(&self) -> &[T];
}

/// Writing a packed tensor, owned or a mutable view.
pub trait PackedERITensorTraitMut<T>: PackedERITensorTrait<T> + IndexMut<(usize, usize, usize, usize)> {
    /// `(ij|kl) = value` (one stored position).
    fn set_value(&mut self, i: usize, j: usize, k: usize, l: usize, value: T);
    /// The stored values, to write.
    fn get_data_mut(&mut self) -> &mut [T];
}

impl<D: DataContainer, O: IndexOrder, S: AOSymmetry<O>> PackedERITensorStruct<D, O, S> {
    /// The packed index of the unordered pair (i, j).
    pub fn pair_index(&self, i: usize, j: usize) -> usize {
        O::pair_index(i, j, self.size)
    }

    /// The stored position of `(ij|kl)`.
    pub fn compute_index(&self, mut i: usize, mut j: usize,
                             mut k: usize, mut l: usize) -> usize {
        // Each pair with its larger index first.
        if i < j { std::mem::swap(&mut i, &mut j); }
        if k < l { std::mem::swap(&mut k, &mut l); }

        let ij = self.pair_index(i, j);
        let kl = self.pair_index(k, l);

        S::combine_index(ij, kl, self.size_half)
    }
}

impl<T, D, O: IndexOrder, S: AOSymmetry<O>> PackedERITensorTrait<T> for PackedERITensorStruct<D, O, S>
where
    D: DataContainer<T = T>,
{
    #[inline(always)]
    fn get_size(&self) -> usize {
        self.full_size
    }

    #[inline(always)]
    fn get_data(&self) -> &[T] {
        self.data.resolve()
    }
}

impl<T, O: IndexOrder, S: AOSymmetry<O>> PackedERITensorTraitMut<T> for PackedERITensor<T, O, S> {
    #[inline(always)]
    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.resolve_mut()
    }

    #[inline(always)]
    fn set_value(&mut self, i: usize, j: usize, k: usize, l: usize, value: T) {
        let idx = self.compute_index(i, j, k, l);
        self.data[idx] = value;
    }
}

impl<T, O: IndexOrder, S: AOSymmetry<O>> PackedERITensorTraitMut<T> for PackedERITensorViewMut<'_, T, O, S> {
    #[inline(always)]
    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.resolve_mut()
    }

    #[inline(always)]
    fn set_value(&mut self, i: usize, j: usize, k: usize, l: usize, value: T) {
        let idx = self.compute_index(i, j, k, l);
        self.data[idx] = value;
    }
}

/// The stored values of an S8 tensor over `size` functions.
pub fn compute_s8_size(size: usize) -> usize {
    let size_half = size * (size + 1) / 2;
    size_half * (size_half + 1) / 2
}

/// The stored values of an S4 tensor over `size` functions.
pub fn compute_s4_size(size: usize) -> usize {
    let size_half = size * (size + 1) / 2;
    size_half * size_half
}

impl<T, O: IndexOrder, S: AOSymmetry<O>> PackedERITensor<T, O, S> {
    /// A zeroed tensor over `size` functions.
    #[inline(always)]
    pub fn new(size: usize) -> PackedERITensor<T, O, S>
    where
        T: Default + Clone,
    {
        let size_half = size * (size + 1) / 2;
        let full_size = S::storage_size(size, size_half);
        PackedERITensorStruct {
            size: size,
            size_half: size_half,
            full_size: full_size,
            data: vec![T::default(); full_size].into_boxed_slice(),
            _order: PhantomData,
            _symmetry: PhantomData,
        }
    }
}

impl<'a, T, O: IndexOrder, S: AOSymmetry<O>> PackedERITensorViewMut<'a, T, O, S> {
    /// `data`, exactly the stored values of a tensor over `size` functions, as one.
    pub fn new(size: usize, data: &'a mut [T]) -> Self
    where
        T: Default + Clone,
    {
        let size_half = size * (size + 1) / 2;
        let full_size = S::storage_size(size, size_half);
        debug_assert_eq!(full_size, data.len());
        Self {
            size: size,
            size_half: size_half,
            full_size: full_size,
            data,
            _order: PhantomData,
            _symmetry: PhantomData,
        }
    }
}

impl<'a, T, O: IndexOrder, S: AOSymmetry<O>> PackedERITensorView<'a, T, O, S> {
    /// `data`, exactly the stored values of a tensor over `size` functions, as one.
    pub fn new(size: usize, data: &'a [T]) -> Self
    where
        T: Default + Clone,
    {
        let size_half = size * (size + 1) / 2;
        let full_size = S::storage_size(size, size_half);
        debug_assert_eq!(full_size, data.len());
        Self {
            size: size,
            size_half: size_half,
            full_size: full_size,
            data,
            _order: PhantomData,
            _symmetry: PhantomData,
        }
    }
}

#[cfg(not(feature = "unchecked"))]
impl<D: DataContainer, O: IndexOrder, S: AOSymmetry<O>> Index<(usize, usize, usize, usize)> for PackedERITensorStruct<D, O, S> {
    type Output = D::T;
    fn index(&self, indices: (usize, usize, usize, usize)) -> &Self::Output {
        let (i, j, k, l) = indices;
        let idx = self.compute_index(i, j, k, l);
        &self.get_data()[idx]
    }
}

#[cfg(not(feature = "unchecked"))]
impl<D, O: IndexOrder, S: AOSymmetry<O>> IndexMut<(usize, usize, usize, usize)> for PackedERITensorStruct<D, O, S>
where
    Self: PackedERITensorTraitMut<D::T>,
    D: DataContainer,
{
    fn index_mut(&mut self, indices: (usize, usize, usize, usize)) -> &mut Self::Output {
        let (i, j, k, l) = indices;
        let idx = self.compute_index(i, j, k, l);
        &mut self.get_data_mut()[idx]
    }
}

#[cfg(feature = "unchecked")]
impl<D: DataContainer, O: IndexOrder, S: AOSymmetry<O>> Index<(usize, usize, usize, usize)> for PackedERITensorStruct<D, O, S> {
    type Output = D::T;
    fn index(&self, indices: (usize, usize, usize, usize)) -> &Self::Output {
        let (i, j, k, l) = indices;
        let idx = self.compute_index(i, j, k, l);
        unsafe { self.get_data().get_unchecked(idx) }
    }
}

#[cfg(feature = "unchecked")]
impl<D, O: IndexOrder, S: AOSymmetry<O>> IndexMut<(usize, usize, usize, usize)> for PackedERITensorStruct<D, O, S>
where
    Self: PackedERITensorTraitMut<D::T>,
    D: DataContainer,
{
    fn index_mut(&mut self, indices: (usize, usize, usize, usize)) -> &mut Self::Output {
        let (i, j, k, l) = indices;
        let idx = self.compute_index(i, j, k, l);
        unsafe { self.get_data_mut().get_unchecked_mut(idx) }
    }
}

/// `n_ket` values for every unordered pair of the `n` basis functions `mu >= nu`: a set of `n_ket` symmetric `n x n`
/// matrices `M(k, mu nu)`, stored as S4 stores its bra pairs (`ColumnMajor::pair_index`) with the ket index `k`
/// fastest, at `ColumnMajor::combine_square(k, pair, n_ket)`. Each pair's `n_ket` values are contiguous, and so is
/// each column `nu` of the triangle (`mu = nu..n`). Holds `n (n + 1)/2 n_ket` values. E.g. the Coulomb matrices of
/// the active pair densities, `M(vw, mu nu) = sum_{lambda sigma} P_{lambda sigma, vw} (mu nu|lambda sigma)`.
#[derive(Debug, PartialEq, Clone)]
pub struct PackedPairTensorStruct<D: DataContainer> {
    n: usize,
    n_ket: usize,
    data: D,
}
/// An owned packed pair tensor.
pub type PackedPairTensor<T> = PackedPairTensorStruct<Box<[T]>>;
/// A view of a packed pair tensor.
pub type PackedPairTensorView<'a, T> = PackedPairTensorStruct<&'a [T]>;
/// A mutable view of a packed pair tensor.
pub type PackedPairTensorViewMut<'a, T> = PackedPairTensorStruct<&'a mut [T]>;

/// The number of values a `PackedPairTensor` over `n` functions with `n_ket` values a pair holds.
pub fn packed_pair_tensor_len(n: usize, n_ket: usize) -> usize {
    n * (n + 1) / 2 * n_ket
}

impl<T: Default + Clone> PackedPairTensor<T> {
    /// Zeroed, on the heap.
    pub fn new(n: usize, n_ket: usize) -> Self {
        PackedPairTensorStruct { n, n_ket, data: vec![T::default(); packed_pair_tensor_len(n, n_ket)].into_boxed_slice() }
    }
}

impl<'a, T> PackedPairTensorView<'a, T> {
    /// Over `data`, `packed_pair_tensor_len(n, n_ket)` long.
    pub fn new(n: usize, n_ket: usize, data: &'a [T]) -> Self {
        assert_eq!(data.len(), packed_pair_tensor_len(n, n_ket), "a packed pair tensor over {n} functions with {n_ket} values a pair");
        PackedPairTensorStruct { n, n_ket, data }
    }
}

impl<'a, T> PackedPairTensorViewMut<'a, T> {
    /// Over `data`, `packed_pair_tensor_len(n, n_ket)` long.
    pub fn new(n: usize, n_ket: usize, data: &'a mut [T]) -> Self {
        assert_eq!(data.len(), packed_pair_tensor_len(n, n_ket), "a packed pair tensor over {n} functions with {n_ket} values a pair");
        PackedPairTensorStruct { n, n_ket, data }
    }
}

impl<T, D: DataContainer<T = T>> PackedPairTensorStruct<D> {
    /// The number of basis functions `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of values a pair, `n_ket`.
    pub fn n_ket(&self) -> usize {
        self.n_ket
    }

    /// The values.
    pub fn get_data(&self) -> &[T] {
        self.data.resolve()
    }

    /// Where the values of the pair `(mu, nu)` (either order) start.
    #[inline(always)]
    pub fn pair_offset(&self, mu: usize, nu: usize) -> usize {
        ColumnMajor::combine_square(0, ColumnMajor::pair_index(mu, nu, self.n), self.n_ket)
    }

    /// The `n_ket` values of the pair `(mu, nu)` (either order).
    #[inline(always)]
    pub fn pair(&self, mu: usize, nu: usize) -> &[T] {
        &self.get_data()[self.pair_offset(mu, nu)..][..self.n_ket]
    }

    /// Column `nu` of the triangle: the pairs `(mu, nu)`, `mu = nu..n`, `n_ket` values each, as the
    /// `n_ket x (n - nu)` column-major matrix `M(k, mu)`.
    #[inline]
    pub fn column(&self, nu: usize) -> &[T] {
        &self.get_data()[self.pair_offset(nu, nu)..][..(self.n - nu) * self.n_ket]
    }
}

impl<T, D: DataContainerMut<T = T>> PackedPairTensorStruct<D> {
    /// The values, to write.
    pub fn get_data_mut(&mut self) -> &mut [T] {
        self.data.resolve_mut()
    }

    /// The `n_ket` values of the pair `(mu, nu)` (either order), writable.
    #[inline(always)]
    pub fn pair_mut(&mut self, mu: usize, nu: usize) -> &mut [T] {
        let (at, n_ket) = (self.pair_offset(mu, nu), self.n_ket);
        &mut self.get_data_mut()[at..][..n_ket]
    }
}

impl<D: DataContainer<T = f64>> PackedPairTensorStruct<D> {
    /// One index transformed: `T(u, k, mu) = sum_nu C(nu, u) M(k, mu nu)`, `c` `n x m`, into `t`, `m x n_ket n` (the
    /// `m x n_ket x n` tensor with `u` fastest, then `k`), overwritten. Two products a column `nu` of the triangle:
    /// the pairs `mu >= nu` into `T(:, :, mu)` with `C(nu, :)`, and `mu > nu` into `T(:, :, nu)` with `C(mu, :)`.
    pub fn transform_one_index_prealloc<C>(&self, c: &MatrixStruct<C>, t: &mut MatrixViewMut<'_, f64>)
    where
        C: DataContainer<T = f64>,
    {
        let (n, n_ket, m) = (self.n, self.n_ket, c.cols());
        assert_eq!(c.rows(), n, "the coefficients have a row per basis function");
        assert_eq!((t.rows(), t.cols()), (m, n_ket * n), "T is m x n_ket n");
        t.fill(0.0);
        if m == 0 || n_ket == 0 {
            return;
        }
        for nu in 0..n {
            let column = MatrixView::new(n_ket, n - nu, self.column(nu));
            // The rows of C as product operands, read with C's stride: the pair's other index runs over them.
            c.view_rows(nu..nu + 1).view_transposed().matmul_scaled_prealloc(
                &MatrixView::new(1, n_ket * (n - nu), column.get_data()),
                &mut t.view_columns_mut(n_ket * nu..n_ket * n),
                1.0,
                1.0,
            );
            if nu + 1 < n {
                c.view_rows(nu + 1..n).view_transposed().matmul_scaled_prealloc(
                    &column.view_columns(1..n - nu).view_transposed(),
                    &mut t.view_columns_mut(n_ket * nu..n_ket * (nu + 1)),
                    1.0,
                    1.0,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small deterministic "random" f64 generator so the tests don't need an
    // external RNG dependency.
    fn lcg_next(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn fill_and_check<O: IndexOrder>(n: usize) {
        let mut s8: PackedERITensor<f64, O, S8> = PackedERITensor::new(n);
        let mut s4: PackedERITensor<f64, O, S4> = PackedERITensor::new(n);

        let mut state = 0x243F6A8885A308D3u64;
        for i in 0..n {
            for j in 0..=i {
                for k in 0..n {
                    for l in 0..=k {
                        let val = lcg_next(&mut state);
                        // Only feed each symmetry-distinct quartet once, exactly
                        // like the real ERI computation does via `TensorChoice`.
                        <S8 as AOSymmetry<O>>::distribute(&mut s8, i, j, k, l, val);
                        <S4 as AOSymmetry<O>>::distribute(&mut s4, i, j, k, l, val);
                    }
                }
            }
        }

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        assert_eq!(
                            s8[(i, j, k, l)], s4[(i, j, k, l)],
                            "S4/S8 mismatch at ({i},{j},{k},{l}) for n={n}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn s4_matches_s8_row_major() {
        fill_and_check::<RowMajor>(5);
    }

    #[test]
    fn s4_matches_s8_column_major() {
        fill_and_check::<ColumnMajor>(5);
    }

    #[test]
    fn packed_pairs_are_s4_column_major_with_the_ket_fastest() {
        let (n, n_ket) = (7, 3);
        let tensor: PackedPairTensor<f64> = PackedPairTensor::new(n, n_ket);
        let mut seen = vec![false; packed_pair_tensor_len(n, n_ket)];
        for nu in 0..n {
            for mu in nu..n {
                let at = tensor.pair_offset(mu, nu);
                assert_eq!(at, tensor.pair_offset(nu, mu));
                for k in 0..n_ket {
                    assert_eq!(at + k, ColumnMajor::combine_square(k, ColumnMajor::pair_index(mu, nu, n), n_ket));
                    assert!(!seen[at + k]);
                    seen[at + k] = true;
                }
                assert_eq!(at, tensor.pair_offset(nu, nu) + (mu - nu) * n_ket, "column {nu} is contiguous");
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn the_one_index_transform_is_the_dense_one() {
        let (n, n_ket, m) = (9, 4, 3);
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut tensor: PackedPairTensor<f64> = PackedPairTensor::new(n, n_ket);
        for value in tensor.get_data_mut() {
            *value = lcg_next(&mut state) - 0.5;
        }
        let mut c = Matrix::<f64>::new(n, m);
        for value in c.get_data_mut() {
            *value = lcg_next(&mut state) - 0.5;
        }
        let mut t = Matrix::<f64>::new(m, n_ket * n);
        tensor.transform_one_index_prealloc(&c, &mut t.view_mut());
        for mu in 0..n {
            for k in 0..n_ket {
                for u in 0..m {
                    let dense: f64 = (0..n).map(|nu| c[(nu, u)] * tensor.pair(mu, nu)[k]).sum();
                    assert!((t[(u, k + n_ket * mu)] - dense).abs() < 1e-14, "T({u}, {k}, {mu}): {} against {dense}", t[(u, k + n_ket * mu)]);
                }
            }
        }
    }

    #[test]
    fn s4_storage_size_is_square_of_pairs() {
        let n = 6;
        let size_half = n * (n + 1) / 2;
        assert_eq!(compute_s4_size(n), size_half * size_half);
        assert_eq!(compute_s8_size(n), size_half * (size_half + 1) / 2);
    }
}
