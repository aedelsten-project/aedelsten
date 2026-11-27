// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Two-electron integrals: the ERI data (shell-pair Hermite coefficients and Schwarz bounds), the stored ERI tensor,
// the direct Fock builders (restricted, unrestricted, general, active space, mixed active space) and the
// half-transformed ERIs, all from one direct walk.

use super::*;
use arrays::scratch;
use basis::contraction::ShellKind;
use basis::shell_pairs::ShellPair;
use eri::TwoElectronIntegrals;

impl AedelstenContext {
    /// The stored ERI tensor (S1, S4 or S8), overwritten in full. Under MPI the call is collective: every rank must
    /// make it; each computes its share of the shell-pair quartets and every rank gets the whole tensor.
    pub fn compute_eri_stored(
        &self,
        output: &mut [f64],
        ao_symmetry: ERIAOSymmetry,
        range_operator: ERIRangeOperator,
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_eri_stored(output, ao_symmetry, range_operator)).inspect(|_| crate::citations::eri(false))
    }

    /// Derives the ERI data of the direct walks from the shell pairs (`set_shell_pairs`), in the scratch: each pair's
    /// Hermite coefficients and Schwarz bound. With `schwarz_threshold`, the pairs that cannot reach it are dropped:
    /// those whose bound, or whose bound times the largest bound, is below it (`None`: none). Needed by the direct
    /// Fock builds, the half transforms and the two-electron gradient; an error if it is already set.
    pub fn set_eri_data(&mut self, schwarz_threshold: Option<f64>) -> Result<(), AedelstenError> {
        dispatch!(self, set_eri_data(schwarz_threshold)).inspect(|_| crate::citations::eri(schwarz_threshold.is_some()))
    }

    /// Frees the ERI data.
    pub fn free_eri_data(&mut self) {
        dispatch!(self, free_eri_data())
    }

    // The surviving pairs' Schwarz bounds, and a second prune: for the crate's tests.
    #[cfg(test)]
    pub(crate) fn eri_data_schwarz_bounds(&self) -> Result<Vec<f64>, AedelstenError> {
        dispatch!(self, eri_data_schwarz_bounds())
    }

    #[cfg(test)]
    pub(crate) fn prune_eri_data(&mut self, threshold: f64) -> Result<(), AedelstenError> {
        dispatch!(self, prune_eri_data(threshold))
    }

    // The direct walks with a sink. Crate-private: the public entry points below and the C interface build the
    // sink (`fock_constructors`' accumulators, `DirectEriGradient`, `GradientTensors`) themselves.
    pub(crate) fn compute_fock_direct<U>(
        &self,
        output: &mut U,
        range_operator: ERIRangeOperator,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError>
    where
        U: eri::DirectFock,
    {
        dispatch!(self, compute_fock_direct(output, range_operator, pairwise_threshold))
    }

    /// `data` as a column-major `rows x cols` matrix, if it has exactly that many elements.
    pub(crate) fn matrix_of(data: &[f64], rows: usize, cols: usize) -> Result<arrays::matrix::MatrixView<'_, f64>, AedelstenError> {
        if data.len() != rows * cols {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: rows * cols, actual: data.len() });
        }
        Ok(arrays::matrix::MatrixView::new(rows, cols, data))
    }

    /// `matrix_of`, mutable.
    pub(crate) fn matrix_of_mut(data: &mut [f64], rows: usize, cols: usize) -> Result<arrays::matrix::MatrixViewMut<'_, f64>, AedelstenError> {
        if data.len() != rows * cols {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: rows * cols, actual: data.len() });
        }
        Ok(arrays::matrix::MatrixViewMut::new(rows, cols, data))
    }

    /// `data` as a packed pair tensor over `n` functions with `n_ket` values a pair, if it has exactly that many
    /// elements.
    pub(crate) fn pair_tensor_of(data: &[f64], n: usize, n_ket: usize) -> Result<eri::packed::PackedPairTensorView<'_, f64>, AedelstenError> {
        let expected = eri::packed::packed_pair_tensor_len(n, n_ket);
        if data.len() != expected {
            return Err(AedelstenError::InvalidMatrixDimensions { expected, actual: data.len() });
        }
        Ok(eri::packed::PackedPairTensorView::new(n, n_ket, data))
    }

    /// `pair_tensor_of`, mutable.
    pub(crate) fn pair_tensor_of_mut(data: &mut [f64], n: usize, n_ket: usize) -> Result<eri::packed::PackedPairTensorViewMut<'_, f64>, AedelstenError> {
        let expected = eri::packed::packed_pair_tensor_len(n, n_ket);
        if data.len() != expected {
            return Err(AedelstenError::InvalidMatrixDimensions { expected, actual: data.len() });
        }
        Ok(eri::packed::PackedPairTensorViewMut::new(n, n_ket, data))
    }

    /// Direct (no stored tensor) restricted J and K from the resident ERI data (`set_eri_data` first):
    /// `J = coeff_j J[d]`, `K = coeff_k K[d]` for a symmetric density `d`, in one walk; `coeff_k` is `-0.5` times the
    /// exact-exchange fraction. `j` and `k` are OVERWRITTEN; a zero coefficient skips that matrix (left zeroed).
    /// Every matrix is `n x n` (`get_n_cont()`), column-major. Range-separated hybrids: one call per range
    /// operator, added by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_fock_direct_restricted(
        &self,
        d: &[f64],
        j: &mut [f64],
        k: &mut [f64],
        coeff_j: f64,
        coeff_k: f64,
        range_operator: ERIRangeOperator,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let d = Self::matrix_of(d, n, n)?;
        let (j, k) = (Self::matrix_of_mut(j, n, n)?, Self::matrix_of_mut(k, n, n)?);
        let mut accumulator = ::fock_constructors::RestrictedDirectFock::new(d, j, k, coeff_j, coeff_k);
        self.compute_fock_direct(&mut accumulator, range_operator, pairwise_threshold)
    }

    /// `compute_fock_direct_restricted` for spin densities: `J = coeff_j J[d_total]` (`d_total = d_alpha + d_beta`,
    /// formed by the caller), `K_s = coeff_k K[d_s]`; `coeff_k` is `-1` times the exact-exchange fraction.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_fock_direct_unrestricted(
        &self,
        d_alpha: &[f64],
        d_beta: &[f64],
        d_total: &[f64],
        j: &mut [f64],
        k_alpha: &mut [f64],
        k_beta: &mut [f64],
        coeff_j: f64,
        coeff_k: f64,
        range_operator: ERIRangeOperator,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let (d_alpha, d_beta, d_total) = (Self::matrix_of(d_alpha, n, n)?, Self::matrix_of(d_beta, n, n)?, Self::matrix_of(d_total, n, n)?);
        let (j, k_alpha, k_beta) = (Self::matrix_of_mut(j, n, n)?, Self::matrix_of_mut(k_alpha, n, n)?, Self::matrix_of_mut(k_beta, n, n)?);
        let mut accumulator = ::fock_constructors::UnrestrictedDirectFock::new(d_alpha, d_beta, d_total, j, k_alpha, k_beta, coeff_j, coeff_k);
        self.compute_fock_direct(&mut accumulator, range_operator, pairwise_threshold)
    }

    /// `compute_fock_direct_restricted` for a density `d` of any symmetry (a transition density, the spin-flip density
    /// of a stability analysis): `J = coeff_j J[d]`, `J_ab = sum_cd d_cd (ab|cd)` (only d's symmetric part enters, and
    /// J is symmetric), and `K = coeff_k K[d]`, `K_ac = sum_bd d_bd (ab|cd)`, which is not symmetric for a
    /// non-symmetric `d` (`K[d]^T = K[d^T]`) and is not symmetrized. A symmetric `d` gives the restricted J and K. `j`
    /// and `k` are OVERWRITTEN; a zero coefficient skips that matrix (left zeroed). Every matrix is `n x n`,
    /// column-major. Range-separated: one call per range operator, added by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_fock_direct_general(
        &self,
        d: &[f64],
        j: &mut [f64],
        k: &mut [f64],
        coeff_j: f64,
        coeff_k: f64,
        range_operator: ERIRangeOperator,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let d = Self::matrix_of(d, n, n)?;
        let (j, k) = (Self::matrix_of_mut(j, n, n)?, Self::matrix_of_mut(k, n, n)?);
        let mut accumulator = ::fock_constructors::GeneralDirectFock::new(d, j, k, coeff_j, coeff_k);
        self.compute_fock_direct(&mut accumulator, range_operator, pairwise_threshold)
    }

    /// `compute_fock_direct_general` for `n_mat` densities in one walk: `d` holds them back to back (`n x n` each,
    /// column-major, `n_mat n^2` doubles), `j` and `k` their J and K in the same places, the same coefficients for
    /// every density. Each integral batch is computed once for all of them. With a threshold, a batch is skipped when
    /// no density needs it (the screen takes the largest density element over the set), so a density's J and K equal
    /// its own `compute_fock_direct_general` to within the screening. `j` and `k` are OVERWRITTEN; a zero coefficient
    /// skips that part (left zeroed). Range-separated: one call per range operator, added by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_fock_direct_general_multi(
        &self,
        d: &[f64],
        j: &mut [f64],
        k: &mut [f64],
        n_mat: usize,
        coeff_j: f64,
        coeff_k: f64,
        range_operator: ERIRangeOperator,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let expected = n * n * n_mat;
        for actual in [d.len(), j.len(), k.len()] {
            if actual != expected || n_mat == 0 {
                return Err(AedelstenError::InvalidMatrixDimensions { expected, actual });
            }
        }
        let mut accumulator = ::fock_constructors::GeneralMultiDirectFock::new(n, d, j, k, coeff_j, coeff_k);
        self.compute_fock_direct(&mut accumulator, range_operator, pairwise_threshold)
    }

    /// The active-space integrals of a CASSCF macro-iteration in one direct walk (full range): the inactive and
    /// active J/K, each against its own coefficients as in `compute_fock_direct_restricted`, and the Coulomb matrices
    /// of the active pair densities, `M(vw, mu nu) = sum_{lambda sigma} P_{lambda sigma, vw} (mu nu|lambda sigma)`,
    /// `P_{lambda sigma, vw} = 1/2 (C_lambda,v C_sigma,w + C_sigma,v C_lambda,w)`, `v >= w`. The caller makes
    /// `T(u, vw, mu) = sum_nu C_nu,u M(vw, mu nu)` with `transform_pair_coulomb_one_index` and
    /// `(pu|vw) = sum_mu T(u vw, mu) C_mu,p` with one product.
    ///
    /// Densities and J/K are `n x n`; `c_active` is `n x n_active`, column-major (e.g. the active columns of the
    /// orbital matrix, a contiguous slice); `m` holds `n (n + 1)/2 n_pairs` doubles, `n_pairs = n_active (n_active +
    /// 1)/2`: packed over the function pairs `mu >= nu` (the pair at `nu n - nu (nu - 1)/2 + mu - nu`), the pairs `vw`
    /// fastest (`v >= w` at `v (v + 1)/2 + w`). Every output OVERWRITTEN. `active_threshold`: a batch whose Schwarz
    /// bound times the largest pair density of one of its shell pairs times the largest coefficient on the other is
    /// below it adds nothing to `M` (its J/K are still built): the bound of what it adds to T. A screened `M` is
    /// complete for T (and `(pu|vw)`) only, not on pairs `mu nu` where `c_active` vanishes. `None` adds every batch.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_fock_direct_active_space(
        &self,
        d_inactive: &[f64],
        d_active: &[f64],
        j_inactive: &mut [f64],
        k_inactive: &mut [f64],
        j_active: &mut [f64],
        k_active: &mut [f64],
        coeff_j_inactive: f64,
        coeff_k_inactive: f64,
        coeff_j_active: f64,
        coeff_k_active: f64,
        c_active: &[f64],
        n_active: usize,
        m: &mut [f64],
        active_threshold: Option<f64>,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let (d_inactive, d_active) = (Self::matrix_of(d_inactive, n, n)?, Self::matrix_of(d_active, n, n)?);
        let (j_inactive, k_inactive) = (Self::matrix_of_mut(j_inactive, n, n)?, Self::matrix_of_mut(k_inactive, n, n)?);
        let (j_active, k_active) = (Self::matrix_of_mut(j_active, n, n)?, Self::matrix_of_mut(k_active, n, n)?);
        let c_active = Self::matrix_of(c_active, n, n_active)?;
        let m = Self::pair_tensor_of_mut(m, n, n_active * (n_active + 1) / 2)?;
        let jk = ::fock_constructors::InactiveActiveDirectFock::new(
            d_inactive, d_active, j_inactive, k_inactive, j_active, k_active,
            coeff_j_inactive, coeff_k_inactive, coeff_j_active, coeff_k_active,
        );
        let mut accumulator = ::fock_constructors::ActiveSpaceDirectFock::new(jk, c_active, m, active_threshold);
        self.compute_fock_direct(&mut accumulator, ERIRangeOperator::FullRange, pairwise_threshold)
    }

    /// The integrals of a state interaction between two runs with different orbitals in one direct walk (full range):
    /// J and K of a density `d` of any symmetry as in `compute_fock_direct_general`, and the Coulomb matrices of the
    /// active pair densities in the bra's and the ket's orbitals,
    /// `M(vw, mu nu) = sum_{lambda sigma} P_{lambda sigma, vw} (mu nu|lambda sigma)`,
    /// `P_{lambda sigma, vw} = 1/2 (c_bra,lambda v c_ket,sigma w + c_bra,sigma v c_ket,lambda w)`, for every pair
    /// `(v, w)` (there is no `v <-> w` symmetry when the bases differ). The caller makes
    /// `T(u, vw, mu) = sum_nu c_ket,nu u M(vw, mu nu)` with `transform_pair_coulomb_one_index` and
    /// `(tu|vw) = sum_mu T(u vw, mu) c_bra,mu t` with one product (`t`, `v` the bra's orbitals, `u`, `w` the ket's).
    /// With `c_bra = c_ket`, the pairs `v >= w` of `M` are `compute_fock_direct_active_space`'s.
    ///
    /// `d`, `j`, `k` are `n x n`; `c_bra` and `c_ket` are `n x n_active`, column-major; `m` holds
    /// `n (n + 1)/2 n_active^2` doubles, packed as in `compute_fock_direct_active_space`, the pair `(v, w)` at
    /// `v + w n_active`. Every output OVERWRITTEN; a zero `coeff_j` or `coeff_k` skips that matrix (left zeroed).
    /// `active_threshold` as in `compute_fock_direct_active_space`, with the larger of the two bases' coefficients.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_fock_direct_mixed_active_space(
        &self,
        d: &[f64],
        j: &mut [f64],
        k: &mut [f64],
        coeff_j: f64,
        coeff_k: f64,
        c_bra: &[f64],
        c_ket: &[f64],
        n_active: usize,
        m: &mut [f64],
        active_threshold: Option<f64>,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let d = Self::matrix_of(d, n, n)?;
        let (j, k) = (Self::matrix_of_mut(j, n, n)?, Self::matrix_of_mut(k, n, n)?);
        let (c_bra, c_ket) = (Self::matrix_of(c_bra, n, n_active)?, Self::matrix_of(c_ket, n, n_active)?);
        let m = Self::pair_tensor_of_mut(m, n, n_active * n_active)?;
        let jk = ::fock_constructors::GeneralDirectFock::new(d, j, k, coeff_j, coeff_k);
        let mut accumulator = ::fock_constructors::ActiveSpaceDirectFock::mixed(jk, c_bra, c_ket, m, active_threshold);
        self.compute_fock_direct(&mut accumulator, ERIRangeOperator::FullRange, pairwise_threshold)
    }

    /// One index of the active walks' `M` transformed: `T(u, vw, mu) = sum_nu c_nu,u M(vw, mu nu)`. `m` holds
    /// `n (n + 1)/2 n_pairs` doubles, packed as `compute_fock_direct_active_space` makes it; `c` is `n x n_columns`,
    /// column-major; `t` is `n_columns x n_pairs x n` (`u` fastest, then `vw`), OVERWRITTEN. Rank-local.
    pub fn transform_pair_coulomb_one_index(&self, m: &[f64], n_pairs: usize, c: &[f64], n_columns: usize, t: &mut [f64]) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let m = Self::pair_tensor_of(m, n, n_pairs)?;
        let c = Self::matrix_of(c, n, n_columns)?;
        let mut t = Self::matrix_of_mut(t, n_columns, n_pairs * n)?;
        m.transform_one_index_prealloc(&c, &mut t);
        Ok(())
    }

    /// The half-transformed ERIs from one direct walk (full range, no J/K):
    /// `X(p, nu, r, sigma) = sum_{mu lambda} c_left,mu p c_right,lambda r (mu nu|lambda sigma)`, two indices in the
    /// orbitals and two left in the AO basis (MRPT2's and integral-direct MP2's integral classes, by two more quarter
    /// transformations in the caller).
    ///
    /// `c_left` is `n x m_left` and `c_right` `n x m_right`, column-major (e.g. column blocks of the orbital matrix,
    /// contiguous slices). `x` holds `m_left n m_right n` doubles, OVERWRITTEN: the column-major array `(p, nu, r,
    /// sigma)`, `p` fastest, then `nu`, `r`, `sigma` (equally, the `(m_left n) x (m_right n)` matrix with rows `p +
    /// m_left nu` and columns `r + m_right sigma`). Passing the same slice as `c_left` and `c_right` (and `m_left ==
    /// m_right`) takes the symmetric path, `X(p, nu, r, sigma) = X(r, sigma, p, nu)`, at half the work. `threshold`: a
    /// batch whose Schwarz bound times the largest `|c_left|` on one of its shell pairs times the largest `|c_right|`
    /// on the other is below it is skipped; `None` keeps every batch. `partition_over_ranks`: under MPI, `true` makes
    /// the call collective (every rank must make it; the quartets are split over the ranks and every rank gets the
    /// whole `X`), `false` computes the whole `X` on this rank alone, with no communication (for a caller that
    /// distributes batches of `c_left` over the ranks itself). Without MPI it makes no difference.
    ///
    /// The scratch (`set_scratch`) holds the walk's row buffers besides its usual scratch:
    /// `(s_max^2 + 1) m_left n` doubles, plus `(s_max^2 + 1) m_right n` in two bases, `s_max` the largest shell's
    /// function count (contracted functions times components). Too small a scratch buffer is an error naming the size
    /// needed.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_eri_half_transformed(
        &self,
        c_left: &[f64],
        m_left: usize,
        c_right: &[f64],
        m_right: usize,
        x: &mut [f64],
        threshold: Option<f64>,
        pairwise_threshold: Option<f64>,
        partition_over_ranks: bool,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let one_basis = std::ptr::eq(c_left, c_right) && m_left == m_right;
        let (left, right) = (Self::matrix_of(c_left, n, m_left)?, Self::matrix_of(c_right, n, m_right)?);
        let x = Self::matrix_of_mut(x, m_left * n, m_right * n)?;
        let right = (!one_basis).then_some(right);
        dispatch!(self, compute_eri_half_transformed(left, right, None, x, threshold, pairwise_threshold, partition_over_ranks))
    }

    /// `compute_eri_half_transformed` by the ranks of the context's communicator together, into one `X` in their
    /// node's shared memory: the ranks must share one node's memory (`set_communicator` with a node's ranks,
    /// `aedelsten_mpi::Communicator::node`), and `x` must be the same memory on every one of them (e.g. an
    /// `aedelsten_mpi::Communicator::shared_array` of the same communicator). The walk's quartets are split over the
    /// ranks, each writes its part straight into `x`, and every rank has the whole `X` when the call returns: one `X`
    /// for the node instead of one per rank, and nothing summed. Collective over the communicator; the coefficients
    /// and thresholds must be the same on every rank, and a rank may still read `x` (a previous `X`) when another
    /// calls: the call waits for all before writing it. On one rank it is `compute_eri_half_transformed`. An error when
    /// the ranks are on several nodes or `x` is not shared among them.
    ///
    /// The scratch holds the row buffers as for `compute_eri_half_transformed`, and a few more `usize`s per shell.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_eri_half_transformed_on_node(
        &self,
        c_left: &[f64],
        m_left: usize,
        c_right: &[f64],
        m_right: usize,
        x: &mut [f64],
        threshold: Option<f64>,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let n = self.get_n_cont();
        let one_basis = std::ptr::eq(c_left, c_right) && m_left == m_right;
        let (left, right) = (Self::matrix_of(c_left, n, m_left)?, Self::matrix_of(c_right, n, m_right)?);
        let x = Self::matrix_of_mut(x, m_left * n, m_right * n)?;
        let right = (!one_basis).then_some(right);
        dispatch!(self, compute_eri_half_transformed_on_node(left, right, None, x, threshold, pairwise_threshold))
    }

    /// `compute_eri_half_transformed` for a batch of sigma: `X(p, nu, r, sigma)` for the sigma in
    /// `sigma_start..sigma_end` only, a run of whole shells (each end a shell's first AO, `get_shells`, or n; not
    /// empty). `x` holds `m_left n m_right n_sigma` doubles (`n_sigma = sigma_end - sigma_start`), OVERWRITTEN: `X`'s
    /// columns for those sigma, the `(m_left n) x (m_right n_sigma)` matrix with rows `p + m_left nu` and columns
    /// `r + m_right (sigma - sigma_start)`. The walk computes only the quartets with a shell in the batch: batches that
    /// cover the basis compute each quartet once per batch among its four shells', at most four times in all,
    /// however many batches there are. The same slice as `c_left` and `c_right` (and `m_left == m_right`) is one
    /// basis, at less work. The other arguments as for `compute_eri_half_transformed`; the scratch beyond the walk's
    /// own: `half_transform_scratch_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_eri_half_transformed_sigma(
        &self,
        c_left: &[f64],
        m_left: usize,
        c_right: &[f64],
        m_right: usize,
        sigma_start: usize,
        sigma_end: usize,
        x: &mut [f64],
        threshold: Option<f64>,
        pairwise_threshold: Option<f64>,
        partition_over_ranks: bool,
    ) -> Result<(), AedelstenError> {
        let (left, right, x) = self.half_transform_sigma_operands(c_left, m_left, c_right, m_right, sigma_start..sigma_end, x)?;
        dispatch!(self, compute_eri_half_transformed(left, right, Some(sigma_start..sigma_end), x, threshold, pairwise_threshold, partition_over_ranks))
    }

    /// `compute_eri_half_transformed_sigma` by the ranks of the context's communicator together, into one `x` in their
    /// node's shared memory, as `compute_eri_half_transformed_on_node` is to `compute_eri_half_transformed`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_eri_half_transformed_sigma_on_node(
        &self,
        c_left: &[f64],
        m_left: usize,
        c_right: &[f64],
        m_right: usize,
        sigma_start: usize,
        sigma_end: usize,
        x: &mut [f64],
        threshold: Option<f64>,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let (left, right, x) = self.half_transform_sigma_operands(c_left, m_left, c_right, m_right, sigma_start..sigma_end, x)?;
        dispatch!(self, compute_eri_half_transformed_on_node(left, right, Some(sigma_start..sigma_end), x, threshold, pairwise_threshold))
    }

    /// The scratch a half transform takes besides the walk's own, in bytes (`set_scratch` holds both): its
    /// row buffers and transposed coefficients for `m_left` and `m_right` orbitals, `one_basis` (the same slice on
    /// both sides) or two, and `n_sigma` functions of sigma (n for `compute_eri_half_transformed`), and on
    /// `node_ranks` ranks sharing one `X` (the `_on_node` forms; 1 for the others) the shells' and the tiles' bounds,
    /// at most. In doubles, `(s_max^2 + 1) m_left n`, and `s_max^2 m_right n_sigma + m_right n` more in two bases,
    /// `s_max` the largest shell's AO count.
    pub fn half_transform_scratch_bytes(&self, m_left: usize, m_right: usize, one_basis: bool, n_sigma: usize, node_ranks: usize) -> usize {
        dispatch!(self, half_transform_scratch_bytes(m_left, m_right, one_basis, n_sigma, node_ranks))
    }

    // The coefficients and X of a batch of sigma as views, checked: `right` `None` for one basis (the same slice).
    #[allow(clippy::type_complexity)]
    fn half_transform_sigma_operands<'c>(
        &self,
        c_left: &'c [f64],
        m_left: usize,
        c_right: &'c [f64],
        m_right: usize,
        sigma: std::ops::Range<usize>,
        x: &'c mut [f64],
    ) -> Result<(arrays::matrix::MatrixView<'c, f64>, Option<arrays::matrix::MatrixView<'c, f64>>, arrays::matrix::MatrixViewMut<'c, f64>), AedelstenError> {
        let n = self.get_n_cont();
        if !(sigma.start < sigma.end && sigma.end <= n) {
            return Err(AedelstenError::StringError(format!("the half transform's sigma {sigma:?} is not a run of the {n} AOs")));
        }
        let one_basis = std::ptr::eq(c_left, c_right) && m_left == m_right;
        let (left, right) = (Self::matrix_of(c_left, n, m_left)?, Self::matrix_of(c_right, n, m_right)?);
        let x = Self::matrix_of_mut(x, m_left * n, m_right * sigma.len())?;
        Ok((left, (!one_basis).then_some(right), x))
    }
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    // The half transform's walk (`AedelstenContext::compute_eri_half_transformed`, or `_sigma` for a batch of sigma;
    // `right` `None` for one basis). Its row buffers are taken off the front of the scratch, sized by the largest
    // shell's function count, and the walk runs in the rest of it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_eri_half_transformed<'c>(
        &self,
        left: arrays::matrix::MatrixView<'c, f64>,
        right: Option<arrays::matrix::MatrixView<'c, f64>>,
        sigma: Option<std::ops::Range<usize>>,
        x: arrays::matrix::MatrixViewMut<'c, f64>,
        threshold: Option<f64>,
        pairwise_threshold: Option<f64>,
        partition_over_ranks: bool,
    ) -> Result<(), AedelstenError> {
        let shell_pairs = self.shell_pairs.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("shell pairs: {e} (set_shell_pairs())")))?;
        let eri_data = self.eri_data.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("ERI data is not resident in memory; call set_eri_data() first".to_string())
        })?;
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let max_shell_functions = qm_region.input.get_basis_shells().iter().map(|shell| shell.get_n_cont() * shell.get_degeneracy()).max().unwrap_or(0);
        if let Some(sigma) = &sigma {
            Self::check_sigma(qm_region.input.get_basis_shells(), self.get_n_cont(), sigma)?;
        }
        let scratch_cell = self.scratch.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("scratch: {e} (set_scratch())")))?;
        let mut scratch = scratch_cell.borrow_mut();
        let buffers = scratch.get_arena();
        let mut accumulator = match (right, sigma) {
            (None, None) => ::fock_constructors::HalfTransformedEri::new(left, x, threshold, partition_over_ranks, max_shell_functions, &buffers)?,
            (Some(right), None) => ::fock_constructors::HalfTransformedEri::mixed(left, right, x, threshold, partition_over_ranks, max_shell_functions, &buffers)?,
            (right, Some(sigma)) => ::fock_constructors::HalfTransformedEri::sigma_batch(left, right, sigma, x, threshold, partition_over_ranks, max_shell_functions, &buffers)?,
        };
        let mut rest = buffers.into_sub_scratch();
        compute_fock_direct(shell_pairs, eri_data, &mut accumulator, &mut rest, &self.comm, ERIRangeOperator::FullRange, pairwise_threshold)?;
        Ok(())
    }

    // `compute_eri_half_transformed` into one X in the shared memory of the communicator's ranks (one node): checked
    // to be shared, then the walk over the ranks with X's tiles under locks (`HalfTransformedEri::on_node`). The
    // shells' first functions and the tile bounds come from the front of the scratch with the row buffers.
    pub(crate) fn compute_eri_half_transformed_on_node<'c>(
        &self,
        left: arrays::matrix::MatrixView<'c, f64>,
        right: Option<arrays::matrix::MatrixView<'c, f64>>,
        sigma: Option<std::ops::Range<usize>>,
        mut x: arrays::matrix::MatrixViewMut<'c, f64>,
        threshold: Option<f64>,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let comm = &self.comm;
        if comm.size() == 1 {
            return self.compute_eri_half_transformed(left, right, sigma, x, threshold, pairwise_threshold, false);
        }
        let mpi = |e: aedelsten_mpi::MpiError| AedelstenError::StringError(format!("the half transform on a node: {e}"));
        if !comm.is_one_node().map_err(mpi)? {
            return Err(AedelstenError::StringError("the half transform on a node: the communicator's ranks are on several nodes".to_string()));
        }
        Self::check_shared(comm, arrays::matrix::MatrixTraitMut::get_data_mut(&mut x)).map_err(mpi)?;
        let shell_pairs = self.shell_pairs.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("shell pairs: {e} (set_shell_pairs())")))?;
        let eri_data = self.eri_data.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("ERI data is not resident in memory; call set_eri_data() first".to_string())
        })?;
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let shells = qm_region.input.get_basis_shells();
        let max_shell_functions = shells.iter().map(|shell| shell.get_n_cont() * shell.get_degeneracy()).max().unwrap_or(0);
        if let Some(sigma) = &sigma {
            Self::check_sigma(shells, self.get_n_cont(), sigma)?;
        }
        let scratch_cell = self.scratch.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("scratch: {e} (set_scratch())")))?;
        let mut scratch = scratch_cell.borrow_mut();
        let buffers = scratch.get_arena();
        let shell_bounds = buffers.alloc::<usize>(shells.len() + 1)?;
        for (bound, shell) in shell_bounds.iter_mut().zip(shells) {
            *bound = shell.get_basis_index();
        }
        shell_bounds[shells.len()] = self.get_n_cont();
        shell_bounds.sort_unstable();
        let mut accumulator = ::fock_constructors::HalfTransformedEri::on_node(left, right, sigma, x, threshold, max_shell_functions, &buffers, comm, shell_bounds)?;
        let mut rest = buffers.into_sub_scratch();
        compute_fock_direct(shell_pairs, eri_data, &mut accumulator, &mut rest, comm, ERIRangeOperator::FullRange, pairwise_threshold)?;
        instrumentation::count("half_transform::node_lock_waits", accumulator.lock_waits());
        Ok(())
    }

    // An error unless `sigma` is a run of whole shells of `shells` (over `n` functions), not empty.
    fn check_sigma(shells: &[basis::shell_types::BasisShell<S>], n: usize, sigma: &std::ops::Range<usize>) -> Result<(), AedelstenError> {
        let on_a_shell = |f: usize| f == n || shells.iter().any(|shell| shell.get_basis_index() == f);
        if sigma.start < sigma.end && sigma.end <= n && on_a_shell(sigma.start) && on_a_shell(sigma.end) {
            Ok(())
        } else {
            Err(AedelstenError::StringError(format!("the half transform's sigma {sigma:?} is not a run of whole shells of the {n} AOs")))
        }
    }

    // `AedelstenContext::half_transform_scratch_bytes` (`scratch_need::half_transform_bytes` on this basis).
    pub(crate) fn half_transform_scratch_bytes(&self, m_left: usize, m_right: usize, one_basis: bool, n_sigma: usize, node_ranks: usize) -> usize {
        let qm_region = self.qm_region.get_cache().expect("input has been freed (free_input)");
        let shells = qm_region.input.get_basis_shells();
        let max_shell_functions = shells.iter().map(|shell| shell.get_n_cont() * shell.get_degeneracy()).max().unwrap_or(0);
        super::scratch_need::half_transform_bytes(self.get_n_cont(), max_shell_functions, shells.len(), m_left, m_right, one_basis, n_sigma, node_ranks)
    }

    // Whether `x` is the same memory on every rank of `comm`: once every rank has stopped reading it, each marks its
    // own element and all read all marks. Collective; `x`'s contents are lost.
    fn check_shared(comm: &aedelsten_mpi::Communicator, x: &mut [f64]) -> Result<(), aedelsten_mpi::MpiError> {
        let (rank, size) = (comm.rank(), comm.size());
        let mut unshared = [0.0];
        comm.shared_memory_barrier()?;
        if x.len() >= size {
            x[rank] = (rank + 1) as f64;
            comm.shared_memory_barrier()?;
            unshared[0] = x[..size].iter().enumerate().any(|(r, &mark)| mark != (r + 1) as f64) as i32 as f64;
            comm.shared_memory_barrier()?;
        }
        comm.sum(&mut unshared)?;
        if unshared[0] > 0.0 {
            return Err(aedelsten_mpi::MpiError::Argument(format!("X is not one memory shared by the {size} ranks")));
        }
        Ok(())
    }

    pub fn set_eri_data(&mut self, schwarz_threshold: Option<f64>) -> Result<(), AedelstenError>
    where
    {
        if self.eri_data.is_already_derived() {
            return Err(AedelstenError::CacheAlreadyDerived("ERI data has already been derived. Please free the existing ERI data before setting new ones.".to_string()));
        }
        let shell_pairs = self.shell_pairs.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("shell pairs are not resident in memory; call set_shell_pairs() first".to_string())
        })?;
        let scratch_cell = self.scratch.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("scratch is not resident in memory; call set_scratch() first".to_string())
        })?;
        let mut scratch = scratch_cell.borrow_mut();
        let eri_data = derive_eri_data(shell_pairs, &mut scratch, schwarz_threshold)?;
        drop(scratch);
        self.eri_data = AedelstenCacheState::Ready(eri_data);
        if schwarz_threshold.is_some() {
            self.prune_eri_data(schwarz_threshold.unwrap())?
        };
        Ok(())
    }

    pub fn free_eri_data(&mut self) {
        self.eri_data.debug_double_free_warning("ERI data");
        self.eri_data.free_cache();
    }

    #[cfg(test)]
    pub(crate) fn eri_data_schwarz_bounds(&self) -> Result<Vec<f64>, AedelstenError> {
        let eri_data = self.eri_data.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("ERI data is not resident in memory; call set_eri_data() first".to_string())
        })?;
        Ok(eri_data.entries().iter().map(|entry| entry.schwarz_bound).collect())
    }

    // Drops the ERI data's pairs whose bound times the largest cannot reach `threshold`.
    pub(crate) fn prune_eri_data(&mut self, threshold: f64) -> Result<(), AedelstenError> {
        let eri_data = self.eri_data.get_cache_mut().map_err(|_| {
            AedelstenError::CacheNotReady("ERI data is not resident in memory; call set_eri_data() first".to_string())
        })?;
        eri_data.prune(threshold);
        Ok(())
    }

    pub fn compute_fock_direct<U>(
        &self,
        output: &mut U,
        range_operator: ERIRangeOperator,
        pairwise_threshold: Option<f64>,
    ) -> Result<(), AedelstenError>
    where
        U: eri::DirectFock,
    {
        let shell_pairs = self.shell_pairs.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("shell pairs: {e} (set_shell_pairs())")))?;
        let eri_data = self.eri_data.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("ERI data is not resident in memory; call set_eri_data() first".to_string())
        })?;
        let scratch_cell = self.scratch.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("scratch: {e} (set_scratch())")))?;
        let mut scratch = scratch_cell.borrow_mut();
        compute_fock_direct(
            shell_pairs,
            eri_data,
            output,
            &mut *scratch, &self.comm,
            range_operator,
            pairwise_threshold,
        )?;
        Ok(())
    }

    pub fn compute_eri_stored(
        &self,
        output: &mut [f64],
        ao_symmetry: ERIAOSymmetry,
        range_operator: ERIRangeOperator,
    ) -> Result<(), AedelstenError>
    where
    {
        self.check_eri_dimensions(&ao_symmetry, output)?;
        let n = self.get_n_cont();
        let shell_pairs = self.shell_pairs.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("shell pairs: {e} (set_shell_pairs())")))?;
        let scratch_cell = self.scratch.get_cache().map_err(|e| AedelstenError::CacheNotReady(format!("scratch: {e} (set_scratch())")))?;
        let mut scratch = scratch_cell.borrow_mut();
        compute_electron_repulsion_integrals(
            output,
            n,
            shell_pairs,
            &mut scratch,
            &self.comm,
            ao_symmetry,
            range_operator,
        )?;
        Ok(())
    }

    /// An error unless `output` has the length of a stored ERI tensor of `ao_symmetry` over this basis.
    pub(crate) fn check_eri_dimensions(&self, ao_symmetry: &ERIAOSymmetry, output: &[f64]) -> Result<(), AedelstenError> {
        let expected = expected_eri_output_len(self.get_n_cont(), ao_symmetry);
        if output.len() != expected {
            return Err(AedelstenError::InvalidMatrixDimensions {
                expected,
                actual: output.len(),
            });
        }
        Ok(())
    }
}

const AEDELSTEN_ERI_FULL_RANGE: i32 = 0;
const AEDELSTEN_ERI_LONG_RANGE: i32 = 1;
const AEDELSTEN_ERI_SHORT_RANGE: i32 = 2;
/// The two-electron operator: the full Coulomb `1/r12`, its long-range part `erf(omega r12)/r12` or its short-range
/// part `erfc(omega r12)/r12`.
#[derive(Debug, Clone, Copy)]
pub enum ERIRangeOperator {
    FullRange,
    LongRange{ omega: f64 },
    ShortRange{ omega: f64 },
}

impl ERIRangeOperator {
    /// The value of a C code (`NAME_TABLE`); an error for an unknown code.
    pub fn from_c_int(value: i32, omega: f64) -> Result<Self, crate::error::AedelstenError> {
        match value {
            AEDELSTEN_ERI_FULL_RANGE => Ok(ERIRangeOperator::FullRange),
            AEDELSTEN_ERI_LONG_RANGE => Ok(ERIRangeOperator::LongRange{ omega }),
            AEDELSTEN_ERI_SHORT_RANGE => Ok(ERIRangeOperator::ShortRange{ omega }),
            _ => Err(crate::error::AedelstenError::InvalidIntegralType(value)),
        }
    }

    /// The C code.
    pub fn get_c_int(&self) -> i32 {
        match self {
            ERIRangeOperator::FullRange => AEDELSTEN_ERI_FULL_RANGE,
            ERIRangeOperator::LongRange{ .. } => AEDELSTEN_ERI_LONG_RANGE,
            ERIRangeOperator::ShortRange{ .. } => AEDELSTEN_ERI_SHORT_RANGE,
        }
    }

    /// The C codes by name, for the named-constant query of the C interface.
    pub const NAME_TABLE: &'static [(&'static str, i32)] = &[
        ("eri_full_range", AEDELSTEN_ERI_FULL_RANGE),
        ("eri_long_range", AEDELSTEN_ERI_LONG_RANGE),
        ("eri_short_range", AEDELSTEN_ERI_SHORT_RANGE),
    ];
}

const AEDELSTEN_ERI_INDEX_ORDER_ROW_MAJOR: i32 = 0;
const AEDELSTEN_ERI_INDEX_ORDER_COLUMN_MAJOR: i32 = 1;
/// The packing order of the S4 and S8 ERI tensors. `RowMajor`: the pair index `i (i + 1)/2 + j` (`i >= j`), then
/// `p (p + 1)/2 + q` over the pairs for S8 or `p n_pairs + q` for S4 (PySCF's `aosym` order). `ColumnMajor`, the
/// default: the smaller (first) of two indices picks the block and the other runs contiguously within it, as the dense
/// arrays are laid out.
#[derive(Debug, Clone, Copy)]
pub enum ERIIndexOrder {
    RowMajor,
    ColumnMajor,
}

impl ERIIndexOrder {
    /// The value of a C code (`NAME_TABLE`); an error for an unknown code.
    pub fn from_c_int(value: i32) -> Result<Self, crate::error::AedelstenError> {
        match value {
            AEDELSTEN_ERI_INDEX_ORDER_ROW_MAJOR => Ok(ERIIndexOrder::RowMajor),
            AEDELSTEN_ERI_INDEX_ORDER_COLUMN_MAJOR => Ok(ERIIndexOrder::ColumnMajor),
            _ => Err(crate::error::AedelstenError::InvalidIntegralType(value)),
        }
    }

    /// The C code.
    pub fn get_c_int(&self) -> i32 {
        match self {
            ERIIndexOrder::RowMajor => AEDELSTEN_ERI_INDEX_ORDER_ROW_MAJOR,
            ERIIndexOrder::ColumnMajor => AEDELSTEN_ERI_INDEX_ORDER_COLUMN_MAJOR,
        }
    }

    /// The C codes by name, for the named-constant query of the C interface.
    pub const NAME_TABLE: &'static [(&'static str, i32)] = &[
        ("eri_index_order_row_major", AEDELSTEN_ERI_INDEX_ORDER_ROW_MAJOR),
        ("eri_index_order_column_major", AEDELSTEN_ERI_INDEX_ORDER_COLUMN_MAJOR),
    ];
}

const AEDELSTEN_ERI_AO_SYMMETRY_S1: i32 = 0;
const AEDELSTEN_ERI_AO_SYMMETRY_S4: i32 = 1;
const AEDELSTEN_ERI_AO_SYMMETRY_S8: i32 = 2;
/// The storage of a stored ERI tensor: `S1` dense (`n^4`), `S4` the 4-fold packing (pairs within the bra and the ket,
/// bra and ket kept apart) and `S8` the 8-fold one (also bra and ket interchanged), each packed in `order`.
#[derive(Debug, Clone, Copy)]
pub enum ERIAOSymmetry {
    S1,
    S4 { order: ERIIndexOrder },
    S8 { order: ERIIndexOrder },
}

impl ERIAOSymmetry {
    /// The value of a C code (`NAME_TABLE`); an error for an unknown code.
    pub fn from_c_int(value: i32, order: i32) -> Result<Self, crate::error::AedelstenError> {
        match value {
            AEDELSTEN_ERI_AO_SYMMETRY_S1 => Ok(ERIAOSymmetry::S1),
            AEDELSTEN_ERI_AO_SYMMETRY_S4 => Ok(ERIAOSymmetry::S4 { order: ERIIndexOrder::from_c_int(order)? }),
            AEDELSTEN_ERI_AO_SYMMETRY_S8 => Ok(ERIAOSymmetry::S8 { order: ERIIndexOrder::from_c_int(order)? }),
            _ => Err(crate::error::AedelstenError::InvalidIntegralType(value)),
        }
    }

    /// The C code.
    pub fn get_c_int(&self) -> i32 {
        match self {
            ERIAOSymmetry::S1 => AEDELSTEN_ERI_AO_SYMMETRY_S1,
            ERIAOSymmetry::S4 { .. } => AEDELSTEN_ERI_AO_SYMMETRY_S4,
            ERIAOSymmetry::S8 { .. } => AEDELSTEN_ERI_AO_SYMMETRY_S8,
        }
    }

    /// The C codes by name, for the named-constant query of the C interface.
    pub const NAME_TABLE: &'static [(&'static str, i32)] = &[
        ("eri_ao_symmetry_s1", AEDELSTEN_ERI_AO_SYMMETRY_S1),
        ("eri_ao_symmetry_s4", AEDELSTEN_ERI_AO_SYMMETRY_S4),
        ("eri_ao_symmetry_s8", AEDELSTEN_ERI_AO_SYMMETRY_S8),
    ];
}

/// The length of a stored ERI tensor of `ao_symmetry` over `n` functions.
pub(crate) fn expected_eri_output_len(n: usize, ao_symmetry: &ERIAOSymmetry) -> usize {
    match ao_symmetry {
        ERIAOSymmetry::S1 => n * n * n * n,
        ERIAOSymmetry::S4 { .. } => eri::packed::compute_s4_size(n),
        ERIAOSymmetry::S8 { .. } => eri::packed::compute_s8_size(n),
    }
}

/// The stored ERI tensor of `shell_pairs` (`n` functions) into `output`, in `ao_symmetry`'s storage, the quartets
/// shared among the ranks of `comm` and the tensor summed on every rank.
pub(crate) fn compute_electron_repulsion_integrals<S: ShellKind>(
    output: &mut [f64],
    n: usize,
    shell_pairs: &[ShellPair<S>],
    scratch: &mut scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    ao_symmetry: ERIAOSymmetry,
    range_operator: ERIRangeOperator,
) -> Result<(), scratch::ArenaError> {
    match ao_symmetry {
        ERIAOSymmetry::S1 => {
            // Dense storage has no symmetry reduction, so index order doesn't apply.
            let mut wrapped_output = arrays::array4d::Array4DViewMut::new(n, n, n, n, output);
            eri_stored(shell_pairs, &mut wrapped_output, scratch, comm, range_operator)
        }
        ERIAOSymmetry::S4 { order: ERIIndexOrder::RowMajor } => {
            let mut wrapped_output: eri::packed::PackedERITensorViewMut<f64, eri::packed::RowMajor, eri::packed::S4> =
                eri::packed::PackedERITensorViewMut::new(n, output);
            eri_stored(shell_pairs, &mut wrapped_output, scratch, comm, range_operator)
        }
        ERIAOSymmetry::S4 { order: ERIIndexOrder::ColumnMajor } => {
            let mut wrapped_output: eri::packed::PackedERITensorViewMut<f64, eri::packed::ColumnMajor, eri::packed::S4> =
                eri::packed::PackedERITensorViewMut::new(n, output);
            eri_stored(shell_pairs, &mut wrapped_output, scratch, comm, range_operator)
        }
        ERIAOSymmetry::S8 { order: ERIIndexOrder::RowMajor } => {
            let mut wrapped_output: eri::packed::PackedERITensorViewMut<f64, eri::packed::RowMajor> =
                eri::packed::PackedERITensorViewMut::new(n, output);
            eri_stored(shell_pairs, &mut wrapped_output, scratch, comm, range_operator)
        }
        ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor } => {
            let mut wrapped_output: eri::packed::PackedERITensorViewMut<f64, eri::packed::ColumnMajor> =
                eri::packed::PackedERITensorViewMut::new(n, output);
            eri_stored(shell_pairs, &mut wrapped_output, scratch, comm, range_operator)
        }
    }
}

fn eri_stored<S: ShellKind, T: eri::TensorChoice>(
    shell_pairs: &[ShellPair<S>],
    output: &mut T,
    scratch: &mut scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    range_operator: ERIRangeOperator,
) -> Result<(), scratch::ArenaError> {
    let mut workstruct = eri::ERIWorkStruct::new(shell_pairs, scratch, comm);
    match range_operator {
        ERIRangeOperator::FullRange => workstruct.compute_eri_stored(output),
        ERIRangeOperator::LongRange { omega } => workstruct.compute_eri_stored_lr(output, omega),
        ERIRangeOperator::ShortRange { omega } => workstruct.compute_eri_stored_sr(output, omega),
    }
}

/// The ERI data of `shell_pairs`, made on this rank.
pub(crate) fn derive_eri_data<S: ShellKind>(
    shell_pairs: &[ShellPair<S>],
    scratch: &mut scratch::Scratch,
    schwarz_threshold: Option<f64>,
) -> Result<eri::EriData, scratch::ArenaError> {
    eri::ERIWorkStruct::new(shell_pairs, scratch, &aedelsten_mpi::Communicator::single()).derive_eri_data(schwarz_threshold)
}

/// One direct walk over `eri_data` into the accumulator `output`, on `range_operator`, the quartets below the threshold
/// skipped.
pub(crate) fn compute_fock_direct<S: ShellKind, U: eri::DirectFock, W: scratch::ScratchSource>(
    shell_pairs: &[ShellPair<S>],
    eri_data: &eri::EriData,
    output: &mut U,
    scratch: &mut W,
    comm: &aedelsten_mpi::Communicator,
    range_operator: ERIRangeOperator,
    pairwise_threshold: Option<f64>,
) -> Result<(), scratch::ArenaError> {
    let mut workstruct = eri::ERIWorkStruct::new(shell_pairs, scratch, comm);
    match range_operator {
        ERIRangeOperator::FullRange => workstruct.compute_fock_direct_from_data(output, eri_data, pairwise_threshold),
        ERIRangeOperator::LongRange { omega } => workstruct.compute_fock_direct_from_data_lr(output, eri_data, omega, pairwise_threshold),
        ERIRangeOperator::ShortRange { omega } => workstruct.compute_fock_direct_from_data_sr(output, eri_data, omega, pairwise_threshold),
    }
}

/// ERI gradients with respect to every centre of `centres` (`select(atom, axis)` per atom).
pub(crate) fn compute_eri_gradient<S: ShellKind, G: eri::GradientSink>(
    shell_pairs: &[ShellPair<S>],
    eri_data: &eri::EriData,
    sink: &mut G,
    scratch: &mut scratch::Scratch,
    comm: &aedelsten_mpi::Communicator,
    range_operator: ERIRangeOperator,
    pairwise_threshold: Option<f64>,
    centres: eri::GradientCentres<'_>,
) -> Result<(), crate::error::AedelstenError> {
    use eri::TwoElectronIntegralGradients;
    let mut workstruct = eri::ERIWorkStruct::new(shell_pairs, scratch, comm);
    match range_operator {
        ERIRangeOperator::FullRange => workstruct.compute_eri_gradient(sink, eri_data, pairwise_threshold, centres),
        ERIRangeOperator::LongRange { omega } => workstruct.compute_eri_gradient_lr(sink, eri_data, omega, pairwise_threshold, centres),
        ERIRangeOperator::ShortRange { omega } => workstruct.compute_eri_gradient_sr(sink, eri_data, omega, pairwise_threshold, centres),
    }?;
    Ok(())
}

#[cfg(test)]
#[path = "tests/eri_cache.rs"]
mod eri_cache_tests;
