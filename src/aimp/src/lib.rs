// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The ab initio model potentials (AIMPs; S. Huzinaga, L. Seijo, Z. Barandiarán and M. Klobukowski, J. Chem. Phys.
//! 86, 2132 (1987); `citations.rs`: `HUZINAGA_AIMP_1987`): the Coulomb part (point charge plus screening), the
//! projection operator and the spectrally resolved exchange of the frozen core, their derivative integrals with respect
//! to one centre and their density-contracted gradients, and the setup of the per-kind kernels.

use std::collections::HashMap;
use arrays::matrix::Factor;
use std::sync::{Arc, LazyLock, Mutex};

use aedelsten_mpi::{Communicator, Counter};
use arrays::matrix::{MatrixMath, MatrixMathMut, MatrixTrait, MatrixTraitMut};
use basis::{BasisInfo, basis_parser::InputBasisShell, contraction::ShellKind, shell_pairs::ShellPair, shell_types::BasisShell};
use eri::TwoElectronIntegrals;
use oneint::OneElectronIntegrals;

#[derive(Debug)]
#[allow(dead_code)] // J_sro is not used, but reserved for future use.
/// The operators of one AIMP kind in its primitive resolution basis: the projector W = C diag(gamma) C^T of the
/// frozen orbitals and the spectrally resolved exchange K_sro (and Coulomb J_sro) of the frozen core.
pub struct AimpProjectionKernel {
    basis_info: BasisInfo,
    w: arrays::matrix::Matrix<f64>,
    k_sro: arrays::matrix::Matrix<f64>,
    j_sro: arrays::matrix::Matrix<f64>,
}

/// One AIMP kind's kernel and primitive shells, for the sites [site_start, site_end) of the AIMP list.
#[derive(Debug)]
pub struct AimpProjectionOperator {
    primitive_shells: Vec<basis::shell_types::BasisShell<basis::contraction::Uncontracted>>,
    site_start: usize,
    site_end: usize,
    kernel: Arc<AimpProjectionKernel>,
}

impl AimpProjectionKernel {
    /// The memory held, in bytes.
    pub fn approx_memory_bytes(&self) -> usize {
        let f64_bytes = std::mem::size_of::<f64>();
        self.basis_info.approx_memory_bytes() + (self.w.get_data().len() + self.k_sro.get_data().len() + self.j_sro.get_data().len()) * f64_bytes
    }
}

impl AimpProjectionOperator {
    /// The memory held, in bytes, the shared kernel counted in full.
    pub fn approx_memory_bytes(&self) -> usize {
        std::mem::size_of_val(self.primitive_shells.as_slice()) + self.kernel.approx_memory_bytes()
    }
}

/// An error in the AIMP setup: library data incomplete, or one passed on from the integrals.
#[derive(Debug)]
pub enum AimpParsingError {
    MissingEnergyBlock,
    IncompleteEnergyContruction,
    MissingOccupationBlock,
    PassedError(String),
}

impl std::fmt::Display for AimpParsingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AimpParsingError::MissingEnergyBlock => write!(f, "AIMP projection shell is missing its orbital energy block"),
            AimpParsingError::IncompleteEnergyContruction => write!(f, "number of unfolded AIMP orbital energies does not match the number of frozen AOs"),
            AimpParsingError::MissingOccupationBlock => write!(f, "AIMP projection shell is missing its occupation block"),
            AimpParsingError::PassedError(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for AimpParsingError {}

impl From<String> for AimpParsingError {
    fn from(error: String) -> Self {
        AimpParsingError::PassedError(error)
    }
}

impl From<arrays::scratch::ScratchError> for AimpParsingError {
    fn from(error: arrays::scratch::ScratchError) -> Self {
        AimpParsingError::PassedError(format!("Scratch error: {}", error))
    }
}

impl From<arrays::scratch::ArenaError> for AimpParsingError {
    fn from(error: arrays::scratch::ArenaError) -> Self {
        AimpParsingError::PassedError(format!("Arena error: {}", error))
    }
}

/// The AIMPs' Coulomb part, -Z (1 + s(r))/r, as one-electron integrals; `output` is (n_AO, n_AO).
pub trait AimpCoulombIntegrals: oneint::OneElectronIntegrals {
    /// The point charges, -Z/r.
    fn compute_aimp_point_charge_contribution<Mf64, T>(&mut self, output: &mut Mf64, screened_charges: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        T: basis::AimpTrait;
    /// The screening corrections, -Z s(r)/r.
    fn compute_aimp_screened_coulomb_contribution<Mf64, T>(&mut self, output: &mut Mf64, screened_charges: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        T: basis::AimpTrait;
    /// Both.
    fn compute_full_aimp_coulomb_contribution<Mf64, T>(&mut self, output: &mut Mf64, screened_charges: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixMathMut<f64>,
        T: basis::AimpTrait,
    {
        let mut temp_matrix = arrays::matrix::Matrix::<f64>::new(output.rows(), output.cols());
        self.compute_aimp_point_charge_contribution(&mut temp_matrix, screened_charges)?;
        self.compute_aimp_screened_coulomb_contribution(output, screened_charges)?;
        output.add_in_place(&temp_matrix);
        Ok(())
    }
}

impl<W: oneint::OneElectronIntegrals> AimpCoulombIntegrals for W {
    fn compute_aimp_point_charge_contribution<Mf64, T>(&mut self, output: &mut Mf64, screened_charges: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        T: basis::AimpTrait,
    {
        self.compute_point_charge_integrals(output, screened_charges)?;
        Ok(())
    }
    fn compute_aimp_screened_coulomb_contribution<Mf64, T>(&mut self, output: &mut Mf64, screened_charges: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        T: basis::AimpTrait,
    {
        self.compute_screened_coulomb_integrals(output, screened_charges)?;
        Ok(())
    }
}

/// The gradients of `AimpCoulombIntegrals` for all centres at once, contracted with a density, as
/// `oneint::OneElectronGradientContractions`: `gradient` (3, n_centres) receives Sum_mn D_mn dV_mn/dX.
/// `operator_centres` pairs each AIMP that is a centre (core AIMPs) with that centre's index; embedding AIMPs do not
/// move and pass an empty slice.
pub trait AimpCoulombGradientContractions: oneint::OneElectronGradientContractions {
    /// The point charges'.
    fn contract_aimp_point_charge_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, aimps: &[T], operator_centres: &[(usize, usize)], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait;
    /// The screening corrections'.
    fn contract_aimp_screened_coulomb_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, aimps: &[T], operator_centres: &[(usize, usize)], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait;
    /// Both.
    fn contract_full_aimp_coulomb_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, aimps: &[T], operator_centres: &[(usize, usize)], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixMathMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait,
    {
        let mut temp = arrays::matrix::Matrix::<f64>::new(gradient.rows(), gradient.cols());
        self.contract_aimp_point_charge_gradient(&mut temp, density, aimps, operator_centres, active)?;
        self.contract_aimp_screened_coulomb_gradient(gradient, density, aimps, operator_centres, active)?;
        gradient.add_in_place(&temp);
        Ok(())
    }
}

impl<W: oneint::OneElectronGradientContractions> AimpCoulombGradientContractions for W {
    fn contract_aimp_point_charge_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, aimps: &[T], operator_centres: &[(usize, usize)], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait,
    {
        oneint::OneElectronGradientContractions::contract_point_charge_gradient(self, gradient, density, aimps, operator_centres, active)?;
        Ok(())
    }
    fn contract_aimp_screened_coulomb_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, aimps: &[T], operator_centres: &[(usize, usize)], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait,
    {
        oneint::OneElectronGradientContractions::contract_screened_coulomb_gradient(self, gradient, density, aimps, operator_centres, active)?;
        Ok(())
    }
}

/// The AIMPs' nonlocal operators, Sum_sites S M S^T with S the overlap between the basis and a site's primitive
/// resolution basis: `output` is (n_AO, n_AO). Every site's primitives are placed at the site, from `input_aimps`.
pub trait AimpExchangeIntegrals {
    /// The projection operator, M = W.
    fn compute_aimp_projection_contribution<Mf64, T>(&mut self, output: &mut Mf64, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixMathMut<f64>,
        T: basis::AimpTrait;
    /// The spectrally resolved exchange, M = K_sro.
    fn compute_aimp_spectrally_resolved_exchange_contribution<Mf64, T>(&mut self, output: &mut Mf64, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixMathMut<f64>,
        T: basis::AimpTrait;
    /// Both, M = W + K_sro, in one pass.
    fn compute_full_aimp_exchange_contribution<Mf64, T>(&mut self, output: &mut Mf64, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixMathMut<f64>,
        T: basis::AimpTrait;
}

/// The basis shells and a scratch buffer: the AIMPs' nonlocal operators in that basis.
pub struct AimpProjectionWorkStruct<'a, S: ShellKind> {
    shells_a: &'a [BasisShell<S>],
    scratch: &'a mut arrays::scratch::Scratch,
    comm: Communicator,
}

impl<'a, S: ShellKind> AimpProjectionWorkStruct<'a, S> {
    /// In the basis `shells_a`, with `scratch` as scratch; the sites are shared among the ranks of `comm` and the
    /// results summed on every rank (collective over `comm`).
    pub fn new(shells_a: &'a [BasisShell<S>], scratch: &'a mut arrays::scratch::Scratch, comm: &Communicator) -> Self {
        Self { shells_a, scratch, comm: comm.clone() }
    }

    /// The one-electron integrals between the basis and `shells_b`.
    pub fn derive_oneint_workstruct<'b, Sb: ShellKind>(&'b mut self, shells_b: &'b [BasisShell<Sb>]) -> oneint::OneIntProjectionWorkStruct<'b, S, Sb> {
        oneint::OneIntProjectionWorkStruct::new(self.shells_a, shells_b, self.scratch)
    }
}

/// The AIMP sites of the projection loops in pieces, each a run of one operator's sites: on one rank every
/// operator's sites in turn; under MPI runs of at most the total cost over `8 n_ranks` (a site costing its operator's
/// primitive count), handed out largest first from a counter the ranks share (`Communicator::counter`), so that each
/// rank takes the next when it is free.
struct AimpSites {
    pieces: Vec<(usize, std::ops::Range<usize>)>,
    counter: Counter,
}

impl AimpSites {
    /// The pieces of the sites of `projection_operator` for the ranks of `comm`. Collective over `comm`. Panics if MPI
    /// fails.
    fn new(projection_operator: &[AimpProjectionOperator], comm: &Communicator) -> AimpSites {
        let n_ranks = comm.size();
        let cost = |op: &AimpProjectionOperator| op.kernel.basis_info.get_n_prim().max(1);
        let total: usize = projection_operator.iter().map(|op| cost(op) * (op.site_end - op.site_start)).sum();
        let largest = total.div_ceil(8 * n_ranks).max(1);
        let mut pieces: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
        for (k, op) in projection_operator.iter().enumerate() {
            let run = if n_ranks == 1 { (op.site_end - op.site_start).max(1) } else { (largest / cost(op)).max(1) };
            let mut start = op.site_start;
            while start < op.site_end {
                let end = (start + run).min(op.site_end);
                pieces.push((k, start..end));
                start = end;
            }
        }
        if n_ranks > 1 {
            pieces.sort_by_key(|(k, sites)| std::cmp::Reverse(cost(&projection_operator[*k]) * sites.len()));
        }
        let counter = comm.counter().expect("the shared counter of the AIMP sites");
        AimpSites { pieces, counter }
    }

    /// The pieces this rank takes, `(operator, its sites)`, in the order it takes them. Panics if MPI fails.
    fn iter(&self) -> impl Iterator<Item = (usize, std::ops::Range<usize>)> + '_ {
        std::iter::from_fn(|| {
            let k = self.counter.next().expect("the next AIMP sites from the shared counter");
            self.pieces.get(k).cloned()
        })
    }
}

use oneint::OneElectronProjectionIntegrals;
use oneint::OneElectronProjectionIntegralGradients;
impl<S: ShellKind> AimpProjectionWorkStruct<'_, S> {
    fn accumulate_aimp_exchange_contribution<Mf64, T>(
        &mut self,
        output: &mut Mf64,
        projection_operator: &mut [AimpProjectionOperator],
        input_aimps: &[T],
        select_proj_mat: impl Fn(&AimpProjectionKernel) -> arrays::matrix::Matrix<f64>,
    ) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixMathMut<f64>,
        T: basis::AimpTrait,
    {
        debug_assert_eq!(output.rows(), output.cols());
        output.fill(0.0);

        let handout = AimpSites::new(projection_operator, &self.comm);

        let mut fully_transformed = arrays::matrix::Matrix::<f64>::new(output.rows(), output.cols());
        for (k, sites) in handout.iter() {
            let op = &mut projection_operator[k];
            let aimp_size = op.kernel.basis_info.get_n_prim();
            let mut s = arrays::matrix::Matrix::<f64>::new(output.rows(), aimp_size);
            let mut half_transformed = arrays::matrix::Matrix::<f64>::new(output.cols(), aimp_size);
            let mut s_inv = arrays::matrix::Matrix::<f64>::new(aimp_size, output.cols());
            let proj_mat = select_proj_mat(&op.kernel);
            // Each site's centre index and position from input_aimps.
            for site in &input_aimps[sites] {
                for shell in op.primitive_shells.iter_mut() {
                    shell.set_centre_index(site.get_centre_index());
                    shell.set_position(*site.get_position());
                }
                let mut oneint_workstruct = self.derive_oneint_workstruct(&op.primitive_shells);
                oneint_workstruct.compute_overlap_integrals(&mut s)?;
                s.transpose_prealloc(&mut s_inv);
                // Compute the projection: S * W * S^T
                s.matmul_prealloc(&proj_mat, &mut half_transformed);
                half_transformed.matmul_prealloc(&s_inv, &mut fully_transformed);
                output.add_in_place(&fully_transformed);
            }
        }

        if self.comm.size() > 1 {
            self.comm.sum(output.get_data_mut())
                .expect("MPI allreduce failed while combining AIMP exchange contributions across ranks");
        }
        Ok(())
    }
}

impl<S: ShellKind> AimpExchangeIntegrals for AimpProjectionWorkStruct<'_, S> {
    fn compute_aimp_projection_contribution<Mf64, T>(&mut self, output: &mut Mf64, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixMathMut<f64>,
        T: basis::AimpTrait,
    {
        self.accumulate_aimp_exchange_contribution(output, projection_operator, input_aimps, |kernel| kernel.w.clone())
    }
    fn compute_aimp_spectrally_resolved_exchange_contribution<Mf64, T>(&mut self, output: &mut Mf64, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixMathMut<f64>,
        T: basis::AimpTrait,
    {
        self.accumulate_aimp_exchange_contribution(output, projection_operator, input_aimps, |kernel| kernel.k_sro.clone())
    }
    fn compute_full_aimp_exchange_contribution<Mf64, T>(&mut self, output: &mut Mf64, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T]) -> Result<(), AimpParsingError>
    where
        Mf64: arrays::matrix::MatrixMathMut<f64>,
        T: basis::AimpTrait,
    {
        self.accumulate_aimp_exchange_contribution(output, projection_operator, input_aimps, |kernel| kernel.k_sro.add(&kernel.w))
    }
}

/// The gradients of `AimpExchangeIntegrals` for all centres at once, contracted with a density: `gradient`
/// (3, n_centres); `active` (one flag per centre, `None` = all) freezes centres.
///
/// Proj = sum_sites S M S^T with M (W or K_sro) independent of every position, so with the symmetrised density D
///   G_X = sum_mn D_mn dProj_mn/dX = sum_(m,p) (dS_mp/dX) Z_mp,     Z = D S (M + M^T),
/// where S is the overlap between the QM basis (rows m) and the site's projection primitives (columns p) and dS/dX
/// its derivative with respect to centre X: nonzero only for X a centre carrying QM shells (the rows) or the
/// site's own centre (the columns). Z needs only S and two small products, so the (n, n) projector derivative is
/// never formed. The overlap gradients are the per-centre ones of the integral-level route.
pub trait AimpExchangeGradientContractions {
    /// The projection operator's.
    fn contract_aimp_projection_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait;
    /// The spectrally resolved exchange's.
    fn contract_aimp_spectrally_resolved_exchange_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait;
    /// Both.
    fn contract_full_aimp_exchange_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait;
}

impl<S: ShellKind> AimpProjectionWorkStruct<'_, S> {
    fn accumulate_aimp_exchange_gradient_contraction<Mg, Md, T>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        projection_operator: &mut [AimpProjectionOperator],
        input_aimps: &[T],
        active: Option<&[bool]>,
        select_proj_mat: impl Fn(&AimpProjectionKernel) -> arrays::matrix::Matrix<f64>,
    ) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait,
    {
        use arrays::array3d::{Array3DTrait, Array3DTraitMut};
        assert_eq!(gradient.rows(), 3, "the gradient matrix is (3, n_centres)");
        let n_centres = gradient.cols();
        if let Some(mask) = active {
            assert_eq!(mask.len(), n_centres, "the active mask has one flag per gradient centre");
        }
        let is_active = |centre: usize| centre < n_centres && active.map_or(true, |mask| mask[centre]);
        gradient.fill(0.0);
        let rows = density.rows();

        let mut ds = arrays::matrix::Matrix::<f64>::new(rows, rows);
        for j in 0..rows {
            for i in 0..rows {
                ds[(i, j)] = 0.5 * (density[(i, j)] + density[(j, i)]);
            }
        }
        // The centres that carry QM shells, each once.
        let mut shell_centres: Vec<usize> = self.shells_a.iter().map(|s| s.get_centre_index()).collect();
        shell_centres.sort_unstable();
        shell_centres.dedup();
        let any_shell_centre_active = shell_centres.iter().any(|&centre| is_active(centre));

        let handout = AimpSites::new(projection_operator, &self.comm);

        for (k, sites) in handout.iter() {
            let op = &mut projection_operator[k];
            let aimp_size = op.kernel.basis_info.get_n_prim();
            let mut s = arrays::matrix::Matrix::<f64>::new(rows, aimp_size);
            let mut half = arrays::matrix::Matrix::<f64>::new(rows, aimp_size);
            let mut z = arrays::matrix::Matrix::<f64>::new(rows, aimp_size);
            let mut ds_dx = arrays::array3d::Array3D::<f64>::new(rows, aimp_size, 3);
            let proj_mat = select_proj_mat(&op.kernel);
            let mut proj_sym = arrays::matrix::Matrix::<f64>::new(aimp_size, aimp_size);
            proj_mat.transpose_prealloc(&mut proj_sym);
            proj_sym.add_in_place(&proj_mat);

            for site in &input_aimps[sites] {
                let site_centre = site.get_centre_index();
                // Nothing to do for a site none of whose centres (the QM shells' and its own) is active.
                if !any_shell_centre_active && !is_active(site_centre) {
                    continue;
                }
                for shell in op.primitive_shells.iter_mut() {
                    shell.set_centre_index(site_centre);
                    shell.set_position(*site.get_position());
                }
                let mut oneint_workstruct = self.derive_oneint_workstruct(&op.primitive_shells);
                oneint_workstruct.compute_overlap_integrals(&mut s)?;
                ds.matmul_prealloc(&s, &mut half);
                half.matmul_prealloc(&proj_sym, &mut z);

                // Embedding AIMPs have no centre index of their own (a sentinel past every column) and do
                // not move; core AIMPs sit on a centre, which may or may not also carry QM shells.
                let extra = if shell_centres.binary_search(&site_centre).is_err() { Some(site_centre) } else { None };
                for &centre in shell_centres.iter().chain(extra.iter()) {
                    if !is_active(centre) {
                        continue;
                    }
                    ds_dx.fill(0.0);
                    oneint_workstruct.compute_overlap_gradient_integrals(&mut ds_dx, centre)?;
                    for plane in 0..3 {
                        let block = ds_dx.view_sub_matrix(plane);
                        let dot: f64 = block.get_data().iter().zip(z.get_data()).map(|(x, y)| x * y).sum();
                        gradient[(plane, centre)] += dot;
                    }
                }
            }
        }

        if self.comm.size() > 1 {
            self.comm.sum(gradient.get_data_mut())
                .expect("MPI allreduce failed while combining AIMP exchange gradients across ranks");
        }
        Ok(())
    }
}

impl<S: ShellKind> AimpExchangeGradientContractions for AimpProjectionWorkStruct<'_, S> {
    fn contract_aimp_projection_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait,
    {
        self.accumulate_aimp_exchange_gradient_contraction(gradient, density, projection_operator, input_aimps, active, |kernel| kernel.w.clone())
    }
    fn contract_aimp_spectrally_resolved_exchange_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait,
    {
        self.accumulate_aimp_exchange_gradient_contraction(gradient, density, projection_operator, input_aimps, active, |kernel| kernel.k_sro.clone())
    }
    fn contract_full_aimp_exchange_gradient<Mg, Md, T>(&mut self, gradient: &mut Mg, density: &Md, projection_operator: &mut [AimpProjectionOperator], input_aimps: &[T], active: Option<&[bool]>) -> Result<(), AimpParsingError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        T: basis::AimpTrait,
    {
        self.accumulate_aimp_exchange_gradient_contraction(gradient, density, projection_operator, input_aimps, active, |kernel| kernel.k_sro.add(&kernel.w))
    }
}


// The size the AIMP setup's own scratch starts at, in MB; it grows on demand (`with_enough_scratch`).
const AIMP_SCRATCH_MIB: usize = 20;

// The kernels built so far in this process, by the content of their resolution shells: a kernel depends on nothing
// else, so every context with the same AIMP kind shares one.
static KERNELS: LazyLock<Mutex<HashMap<Vec<u64>, Arc<AimpProjectionKernel>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

// The resolution shells as bits: equal keys, bit-identical kernels.
fn kernel_key(resolution_shells: &[InputBasisShell]) -> Vec<u64> {
    let mut key = Vec::new();
    let push_all = |key: &mut Vec<u64>, values: Option<&[f64]>| match values {
        Some(values) => {
            key.push(values.len() as u64);
            key.extend(values.iter().map(|x| x.to_bits()));
        }
        None => key.push(u64::MAX),
    };
    for shell in resolution_shells {
        key.push(shell.get_lqn() as u64);
        push_all(&mut key, Some(shell.get_exponents()));
        key.push(shell.get_coefficients().cols() as u64);
        push_all(&mut key, Some(shell.get_coefficients().get_data()));
        push_all(&mut key, shell.get_energies().map(|e| &e[..]));
        push_all(&mut key, shell.get_occupations().map(|o| &o[..]));
    }
    key
}

/// The projection operators of a list of AIMPs, one per run of consecutive sites of the same kind, the kernel built
/// once per kind and process. Uses a scratch buffer of its own, not the caller's.
pub fn materialise_aimp_projection_operator<T: basis::AimpTrait>(input_aimps: &[T]) -> Result<Vec<AimpProjectionOperator>, AimpParsingError>{
    let mut scratch = None;
    let mut aimp_projection_operators = Vec::new();

    let mut i = 0;
    while i < input_aimps.len() {
        let identity = input_aimps[i].get_projection_basis_identity();
        let mut j = i + 1;
        while j < input_aimps.len() && input_aimps[j].get_projection_basis_identity() == identity {
            j += 1;
        }
        // W and K_sro/J_sro in the one resolution basis (valence plus PROJOP primitives for a core AIMP).
        let resolution_shells = input_aimps[i].get_resolution_basis();

        let key = kernel_key(&resolution_shells);
        let cached = KERNELS.lock().unwrap().get(&key).cloned();
        let kernel = match cached {
            Some(kernel) => kernel,
            None => {
                if scratch.is_none() {
                    scratch = Some(arrays::scratch::Scratch::new(AIMP_SCRATCH_MIB, arrays::scratch::ByteUnit::MiB)?);
                }
                let kernel = Arc::new(build_aimp_projection_kernel(&resolution_shells, scratch.as_mut().unwrap())?);
                Arc::clone(KERNELS.lock().unwrap().entry(key).or_insert(kernel))
            }
        };

        // The primitive shells; their centre index and position are set per site when the operator is applied.
        let mut primitive_shells = basis::convert_input_basis_to_used_basis::<basis::contraction::Uncontracted>(
            &resolution_shells,
            basis::shell_types::HarmonicType::default_for_lqn,
            input_aimps[i].get_centre_index(),
            *input_aimps[i].get_position()
        );
        basis::fix_basis_indices(&mut primitive_shells);

        aimp_projection_operators.push(AimpProjectionOperator {
            primitive_shells,
            site_start: i,
            site_end: j,
            kernel,
        });
        i = j;
    }
    Ok(aimp_projection_operators)
}

fn build_aimp_projection_kernel(resolution_shells: &[InputBasisShell], scratch: &mut arrays::scratch::Scratch) -> Result<AimpProjectionKernel, AimpParsingError> {
    let mut primitive_shells = basis::convert_input_basis_to_used_basis::<basis::contraction::Uncontracted>(
        resolution_shells,
        basis::shell_types::HarmonicType::default_for_lqn,
        0,
        [0.0, 0.0, 0.0]
    );
    let basis_info = basis::fix_basis_indices(&mut primitive_shells);
    let n_prim = basis_info.get_n_prim();
    let (c, e, occ) = construct_aimp_coefficient_matrix_and_energy_vector(resolution_shells, n_prim)?;
    let (d, w) = construct_aimp_density_matrices(&c, &e, &occ);
    let (j_sro, k_sro) = compute_j_and_k_sro(&primitive_shells, &basis_info, &d, scratch)?;
    Ok(AimpProjectionKernel { basis_info, w, k_sro, j_sro })
}

// The orbital energy weighted by the occupied fraction of its shell: gamma = e occ / (2 (2l + 1)).
fn projection_gamma(raw_energy: f64, occupation: f64, lqn: usize) -> f64 {
    let full_shell = 2.0 * (2 * lqn + 1) as f64;
    raw_energy * (occupation / full_shell)
}

// The frozen orbitals as coefficients over the primitive resolution basis, with their weighted energies (gamma) and
// occupations, unfolded per m_l: column k is one orbital of one m_l, nonzero only on the primitives of that l and
// m_l, in the basis order (l first, then the primitives, then m_l). For [4s,2p] -> (2s1p):
//
//     | 1s1 | 2s1 | 0.0  | 0.0  | 0.0  |
//     | 1s2 | 2s2 | 0.0  | 0.0  | 0.0  |
//     | 1s3 | 2s3 | 0.0  | 0.0  | 0.0  |
//     | 1s4 | 2s4 | 0.0  | 0.0  | 0.0  |
//     | 0.0 | 0.0 | 1px1 | 0.0  | 0.0  |
//     | 0.0 | 0.0 | 0.0  | 1py1 | 0.0  |
//     | 0.0 | 0.0 | 0.0  | 0.0  | 1pz1 |
//     | 0.0 | 0.0 | 1px2 | 0.0  | 0.0  |
//     | 0.0 | 0.0 | 0.0  | 1py2 | 0.0  |
//     | 0.0 | 0.0 | 0.0  | 0.0  | 1pz2 |
fn construct_aimp_coefficient_matrix_and_energy_vector(
    projection_shells: &[InputBasisShell],
    n_prim: usize
) -> Result<(arrays::matrix::Matrix<f64>, Vec<f64>, Vec<f64>), AimpParsingError> {
    // The number of frozen orbitals, unfolded.
    let mut n_frozen_aos = 0;
    for shell in projection_shells.iter() {
        let lqn = shell.get_lqn();
        let degeneracy = cart2sph::compute_spherical_degeneracy(lqn);
        let n_shell_aos = shell.get_coefficients().cols();
        n_frozen_aos += degeneracy * n_shell_aos;
    }

    // Construct C and fill e and occ.
    let mut e = Vec::with_capacity(n_frozen_aos);
    let mut occ = Vec::with_capacity(n_frozen_aos);
    let mut c = arrays::matrix::Matrix::<f64>::new(n_prim, n_frozen_aos);
    let mut ao_offset = 0;
    let mut lqn_offset = 0;
    for shell in projection_shells.iter() {
        let lqn = shell.get_lqn();
        let degeneracy = cart2sph::compute_spherical_degeneracy(lqn);
        let energies = match shell.get_energies() {
            Some(energies) => energies,
            None => return Err(AimpParsingError::MissingEnergyBlock),
        };
        let occupations = match shell.get_occupations() {
            Some(occupations) => occupations,
            None => return Err(AimpParsingError::MissingOccupationBlock),
        };
        let coeffs = shell.get_coefficients();
        for ((aimp_col, e_in), occ_in) in coeffs.iter_col().zip(energies.iter()).zip(occupations.iter()) {
            // The occupation spread evenly over the 2l + 1 m_l columns.
            let occ_per_ml = occ_in / degeneracy as f64;
            let gamma_in = projection_gamma(*e_in, *occ_in, lqn);
            for (ml, ao) in (0..degeneracy).zip(c.iter_col_mut().skip(ao_offset)) {
                e.push(gamma_in);
                occ.push(occ_per_ml);
                for (c_out, c_in) in ao.iter_mut()
                        .skip(lqn_offset+ml)
                        .step_by(degeneracy)
                        .zip(aimp_col.iter()) {
                    *c_out = *c_in;
                }
            }
            ao_offset += degeneracy;
        }
        lqn_offset += degeneracy * coeffs.rows();
    }

    // One energy per frozen orbital.
    let e = if n_frozen_aos == e.len() {
        e
    } else {
        return Err(AimpParsingError::IncompleteEnergyContruction)
    };


    Ok((c, e, occ))
}


// The frozen density D and the projector W from the unfolded orbitals.
fn construct_aimp_density_matrices(c: &arrays::matrix::Matrix<f64>, e: &[f64], occ: &[f64]) -> (arrays::matrix::Matrix<f64>, arrays::matrix::Matrix<f64>) {
    let n_prim = c.rows();
    // Regular density matrix, \sum_k occ_k c_{j,k}^{*} * c_{i,k}
    let mut d = arrays::matrix::Matrix::<f64>::new(n_prim, n_prim);
    // Energy-weighted density matrix, \sum_k \gamma_k c_{j,k}^{*} * c_{i,k}, gamma_k with the occupation already in.
    let mut w = arrays::matrix::Matrix::<f64>::new(n_prim, n_prim);

    for ((gamma_k, occ_k), ao) in e.iter().zip(occ.iter()).zip(c.iter_col()) {
        for (c_jk, (d_col, w_col)) in ao.iter().zip(d.iter_col_mut().zip(w.iter_col_mut())) {
            for (c_ik, (d_ij, w_ij)) in ao.iter().zip(d_col.iter_mut().zip(w_col.iter_mut())) {
                *d_ij += occ_k * c_jk * c_ik;
                *w_ij += gamma_k * c_jk * c_ik;
            }
        }
    }

    (d, w)
}

// J_sro and K_sro of the frozen density in the resolution basis.
fn compute_j_and_k_sro(
    primitive_shells: &[basis::shell_types::BasisShell<basis::contraction::Uncontracted>],
    basis_info: &BasisInfo,
    d: &arrays::matrix::Matrix<f64>,
    scratch: &mut arrays::scratch::Scratch,
) -> Result<(arrays::matrix::Matrix<f64>, arrays::matrix::Matrix<f64>), AimpParsingError> {
    let shell_pairs = basis::shell_pairs::derive_shell_pairs(primitive_shells, None)?;
    let n_prim = basis_info.get_n_prim();
    let (s_inv, j_core, k_core) = with_enough_scratch(scratch, |scratch| {
        let s_inv = compute_s_inv(&shell_pairs, scratch, basis_info)?;
        let (j_core, k_core) = compute_j_and_k_core(&shell_pairs, d, n_prim, scratch)?;
        Ok((s_inv, j_core, k_core))
    })?;

    // Spectrally resolved Coulomb/Exchange operator, J/K_sro = S^{-1} * J/K_core * S^{-1}, formed in full and then
    // symmetrized.
    let mut half_trans = arrays::matrix::Matrix::<f64>::new(n_prim, n_prim);
    let mut resolve = |core: &arrays::matrix::Matrix<f64>| {
        s_inv.matmul_prealloc(core, &mut half_trans);
        half_trans.matmul(&s_inv).symmetrize(0.5)
    };
    Ok((resolve(&j_core), resolve(&k_core)))
}

/// J_core and K_core of the frozen core density in the resolution basis, built directly from the shell quartets, no
/// stored ERI tensor.
fn compute_j_and_k_core(
    shell_pairs: &[ShellPair<basis::contraction::Uncontracted>],
    d: &arrays::matrix::Matrix<f64>,
    n_prim: usize,
    scratch: &mut arrays::scratch::Scratch,
) -> Result<(arrays::matrix::Matrix<f64>, arrays::matrix::Matrix<f64>), arrays::scratch::ArenaError> {
    let _span = instrumentation::span("aimp::core_jk");
    let mut j_core = arrays::matrix::Matrix::<f64>::new(n_prim, n_prim);
    let mut k_core = arrays::matrix::Matrix::<f64>::new(n_prim, n_prim);
    let single = Communicator::single();
    let mut workstruct = eri::ERIWorkStruct::new(shell_pairs, scratch, &single);
    // No screening.
    let eri_data = workstruct.derive_eri_data(None)?;
    // J +1 and K -0.5 against the total density, as restricted HF; the sign is folded into the coefficient so the
    // operator is additive.
    let mut jk = fock_constructors::RestrictedDirectFock::new(d.view(), j_core.view_mut(), k_core.view_mut(), 1.0, -0.5);
    // On every rank alone.
    workstruct.compute_fock_direct_from_data(&mut jk, &eri_data, None)?;
    Ok((j_core, k_core))
}

/// Runs `setup`; if the scratch is too small, resizes it to at least what the error names (and at least double, so
/// the retries are few) and runs `setup` again from the start. Only for the AIMP setup, which is pure (nothing is
/// written out before it succeeds); the resized scratch is kept for the next AIMP kind.
fn with_enough_scratch<R>(
    scratch: &mut arrays::scratch::Scratch,
    mut setup: impl FnMut(&mut arrays::scratch::Scratch) -> Result<R, arrays::scratch::ArenaError>,
) -> Result<R, AimpParsingError> {
    loop {
        match setup(scratch) {
            Err(arrays::scratch::ArenaError::OutOfMemory { needed, used, .. }) => {
                let bytes = (used + needed).max(2 * scratch.byte_len());
                scratch.resize(bytes.div_ceil(1024), arrays::scratch::ByteUnit::KiB)?;
            }
            other => return Ok(other?),
        }
    }
}

// The pseudo-inverse of the resolution basis' overlap.
fn compute_s_inv(
    shell_pairs: &[ShellPair<basis::contraction::Uncontracted>],
    scratch: &mut arrays::scratch::Scratch,
    basis_info: &BasisInfo
) -> Result<arrays::matrix::Matrix<f64>, arrays::scratch::ArenaError> {
    let n_prim = basis_info.get_n_prim();
    let mut s = arrays::matrix::Matrix::<f64>::new(n_prim, n_prim);
    let mut workstruct = oneint::OneIntWorkStruct::new_local(shell_pairs, scratch);
    workstruct.compute_overlap_integrals(&mut s)?;
    // A canonical-orthogonalization pseudo-inverse: the projector onto the numerically spanned space.
    Ok(s.symmetric_pseudo_inverse(AIMP_RESOLUTION_THRESHOLD))
}

/// Eigenvalues of the AIMP resolution-basis overlap below this fraction of the largest are dropped.
const AIMP_RESOLUTION_THRESHOLD: f64 = 1e-12;

#[cfg(test)]
mod tests {
    use super::*;
    use arrays::scratch::{Scratch, ByteUnit};

    #[test]
    fn a_too_small_aimp_scratch_is_resized_until_the_setup_fits() {
        let mut scratch = Scratch::new(1, ByteUnit::KiB).unwrap();
        let mut runs = 0;
        // 100 000 f64 in two pieces: fails in 1 KB, then in at least one larger size, then fits.
        let sum = with_enough_scratch(&mut scratch, |scratch| {
            runs += 1;
            let arena = scratch.get_arena();
            let a = arena.alloc::<f64>(40_000)?;
            let b = arena.alloc::<f64>(60_000)?;
            a.fill(1.0);
            b.fill(2.0);
            Ok(a.iter().sum::<f64>() + b.iter().sum::<f64>())
        })
        .unwrap();
        assert_eq!(sum, 160_000.0);
        assert!(runs >= 2);
        assert!(scratch.byte_len() >= 800_000, "the resized scratch is kept");
        // Other errors are passed on, not retried.
        let mut overflow_runs = 0;
        let result = with_enough_scratch(&mut scratch, |scratch| {
            overflow_runs += 1;
            scratch.get_arena().alloc::<f64>(usize::MAX).map(|_| ())
        });
        assert!(result.is_err() && overflow_runs == 1, "{:?}", result.map_err(|e| format!("{e:?}")));
    }
}
