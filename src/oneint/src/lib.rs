// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! One-electron integrals over Gaussian shells by the McMurchie-Davidson scheme (L. E. McMurchie and E. R.
//! Davidson, J. Comput. Phys. 26, 218 (1978)): overlap, kinetic energy, the potentials of point and finite nuclei,
//! AIMP screens and external multipoles, moments, the electric field, linear and angular momentum, and the
//! picture-change operators p.Vp and sigma.p V sigma.p; the derivatives of the first group contracted with a density
//! for all centres, and the overlap derivatives between two shell lists with respect to one centre. Under MPI pieces of the shell pairs and of
//! the operator's sites are handed out over the ranks (`pieces`) and every rank receives the complete result.

mod additional_arguments;
pub use additional_arguments::MomentOrigin;
mod distribute;
mod inner_functions;
mod kernels;
mod picture_change_kernels;
mod gradient_contraction;
mod gradient_kernels;
#[cfg(test)]
mod shift_oracle;
pub mod pieces;
use pieces::SiteRuns;
mod spherical_transformation;
pub mod site_contractions;

use aedelsten_mpi::Communicator;
use arrays;
use basis;

use kernels::OneElectronIntegralKernel;
use kernels::ArrayBatchKernel;
use arrays::matrix::MatrixTraitMut as _;

/// The shell pairs of one basis and a scratch buffer: the integrals within that basis.
pub struct OneIntWorkStruct<'a, S: basis::contraction::ShellKind> {
    shell_pairs: &'a [basis::shell_pairs::ShellPair<S>],
    scratch: &'a mut arrays::scratch::Scratch,
    comm: Communicator,
}

impl<'a, S: basis::contraction::ShellKind> OneIntWorkStruct<'a, S> {
    /// Over `shell_pairs`, with `scratch` as scratch; the pairs are shared among the ranks of `comm` and the
    /// results summed on every rank (collective over `comm`).
    pub fn new(shell_pairs: &'a [basis::shell_pairs::ShellPair<S>], scratch: &'a mut arrays::scratch::Scratch, comm: &Communicator) -> Self {
        Self { shell_pairs, scratch, comm: comm.clone() }
    }

    /// As `new`, but every pair on this rank and no communication: for a caller that shares out other work (sites).
    pub fn new_local(shell_pairs: &'a [basis::shell_pairs::ShellPair<S>], scratch: &'a mut arrays::scratch::Scratch) -> Self {
        Self { shell_pairs, scratch, comm: Communicator::single() }
    }
}

/// One-electron integrals within one basis. Each fills every element of `output`, (n_AO, n_AO) or planes of
/// that, in the shell pairs' AO order. An `ArenaError` means a shell pair's batch did not fit the scratch;
/// `output` is then invalid.
pub trait OneElectronIntegrals {
    /// The overlap, <A|B>.
    fn compute_overlap_integrals<Mf64>(&mut self, output: &mut Mf64) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>;
    /// The kinetic energy, <A| -nabla^2/2 |B>.
    fn compute_kinetic_integrals<Mf64>(&mut self, output: &mut Mf64) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>;
    /// The potential energy of an electron in the field of point charges, <A| -Sum_C q_C / |r - C| |B>.
    fn compute_point_charge_integrals<Mf64, P>(&mut self, output: &mut Mf64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::PointChargeTrait;
    /// The potential energy of an electron in the field of point multipoles, every moment up to each source's
    /// order, in the given convention.
    fn compute_external_multipole_integrals<Mf64, P>(
        &mut self,
        output: &mut Mf64,
        sources: &[P],
        convention: basis::charge_carriers::MultipoleConvention,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait;
    /// The potential energy of an electron in the field of Gaussian nuclei,
    /// <A| -Sum_C q_C erf(sqrt(e_C) |r - C|) / |r - C| |B>.
    fn compute_finite_charge_integrals<Mf64, P>(&mut self, output: &mut Mf64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::FiniteChargeTrait;
    /// The AIMPs' screening corrections alone, <A| -Sum_C Z_C Sum_i A_i exp(-c_i |r - C|^2) / |r - C| |B>; the
    /// bare -Z_C / |r - C| is a point-charge integral.
    fn compute_screened_coulomb_integrals<Mf64, T>(&mut self, output: &mut Mf64, aimps: &[T]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        T: basis::charge_carriers::ScreenedChargeTrait;
    /// The moments <A| (x - O_x)^lx (y - O_y)^ly (z - O_z)^lz |B> about the origin, one plane per Cartesian
    /// component of the origin's order, in the canonical order.
    fn compute_moment_integrals<Af64>(&mut self, output: &mut Af64, origin: &MomentOrigin) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>;
    /// The electric-field integrals <A| (r - C_s) / |r - C_s|^3 |B> at every site s: three planes per site,
    /// ordered (site 0: x, y, z; site 1: x, y, z; ...).
    fn compute_electric_field_integrals<Af64, S>(&mut self, output: &mut Af64, sites: &[S]) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        S: basis::charge_carriers::SiteTrait;
    /// The linear momentum, nabla_k = <A| d/dk |B> (p = -i nabla), three real antisymmetric planes x, y, z.
    fn compute_nabla_integrals<Af64>(&mut self, output: &mut Af64) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>;
    /// The orbital angular momentum about `origin`, A_k = <A| ((r - O) x nabla)_k |B> (L = -i A), three real
    /// antisymmetric planes x, y, z.
    fn compute_angular_momentum_integrals<Af64>(&mut self, output: &mut Af64, origin: &[f64; 3]) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>;
}

impl<'a, S: basis::contraction::ShellKind> OneElectronIntegrals for OneIntWorkStruct<'a, S> {
    fn compute_overlap_integrals<Mf64>(&mut self, output: &mut Mf64) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::overlap");
        one_electron_integral_loop::<
            Mf64,
            S,
            kernels::OverlapKernel
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &additional_arguments::EmptyAdditionalArguments {}
            )
    }

    fn compute_kinetic_integrals<Mf64>(&mut self, output: &mut Mf64) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::kinetic");
        one_electron_integral_loop::<
            Mf64,
            S,
            kernels::KineticKernel
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &additional_arguments::EmptyAdditionalArguments {}
            )
    }

    fn compute_point_charge_integrals<Mf64, P>(&mut self, output: &mut Mf64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::PointChargeTrait
    {
        let _span = instrumentation::span("oneint::point_charge");
        one_electron_integral_loop::<
            Mf64,
            S,
            kernels::CoulombKernel<P>
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &charges
            )
    }

    fn compute_external_multipole_integrals<Mf64, P>(
        &mut self,
        output: &mut Mf64,
        sources: &[P],
        convention: basis::charge_carriers::MultipoleConvention,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait
    {
        let _span = instrumentation::span("oneint::external_multipole");
        match convention {
            basis::charge_carriers::MultipoleConvention::ContactFree => one_electron_integral_loop::<
                Mf64,
                S,
                kernels::CoulombKernel<P, kernels::MultipoleSourceMode<kernels::ContactFree>>
                >(
                    output,
                    self.shell_pairs,
                    &mut self.scratch, &self.comm,
                    &sources
                ),
            basis::charge_carriers::MultipoleConvention::Exact => one_electron_integral_loop::<
                Mf64,
                S,
                kernels::CoulombKernel<P, kernels::MultipoleSourceMode<kernels::Exact>>
                >(
                    output,
                    self.shell_pairs,
                    &mut self.scratch, &self.comm,
                    &sources
                ),
        }
    }

    fn compute_finite_charge_integrals<Mf64, P>(&mut self, output: &mut Mf64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::FiniteChargeTrait
    {
        let _span = instrumentation::span("oneint::finite_charge");
        one_electron_integral_loop::<
            Mf64,
            S,
            kernels::CoulombKernel<P, kernels::FiniteMode>
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &charges
            )
    }

    fn compute_screened_coulomb_integrals<Mf64, T>(&mut self, output: &mut Mf64, aimps: &[T]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        T: basis::charge_carriers::ScreenedChargeTrait
    {
        let _span = instrumentation::span("oneint::screened_coulomb");
        one_electron_integral_loop::<
            Mf64,
            S,
            kernels::CoulombKernel<T, kernels::ScreenMode>
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &aimps
            )
    }

    fn compute_moment_integrals<Af64>(&mut self, output: &mut Af64, origin: &MomentOrigin) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::moment");
        one_electron_integral_loop_array::<
            Af64,
            S,
            kernels::MomentKernel
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &origin
            )
    }

    fn compute_electric_field_integrals<Af64, Site>(&mut self, output: &mut Af64, sites: &[Site]) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        Site: basis::charge_carriers::SiteTrait
    {
        let _span = instrumentation::span("oneint::electric_field");
        one_electron_integral_loop_array::<
            Af64,
            S,
            kernels::ElectricFieldIntegrals<Site>
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &sites
            )
    }

    fn compute_nabla_integrals<Af64>(&mut self, output: &mut Af64) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::nabla");
        one_electron_integral_loop_array::<Af64, S, kernels::NablaKernel>(output, self.shell_pairs, &mut self.scratch, &self.comm, &additional_arguments::EmptyAdditionalArguments {})
    }

    fn compute_angular_momentum_integrals<Af64>(&mut self, output: &mut Af64, origin: &[f64; 3]) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::angular_momentum");
        one_electron_integral_loop_array::<Af64, S, kernels::AngularMomentumKernel>(output, self.shell_pairs, &mut self.scratch, &self.comm, origin)
    }

}

/// The derivative integrals of the one-electron operators for all centres at once, contracted with a density:
/// `gradient` (3, n_centres) receives Sum_mn D_mn dO_mn/dX. `operator_centres` pairs each source of the operator
/// that is bound to a centre (nuclei, core AIMPs) with that centre's global index, `(position in the source list,
/// centre index)`; empty for external sources and embedding AIMPs, which do not move. `active` (one flag per centre,
/// `None` = all) freezes centres: their columns stay zero.
pub trait OneElectronGradientContractions {
    /// The overlap's.
    fn contract_overlap_gradient<Mg, Md>(&mut self, gradient: &mut Mg, density: &Md, active: Option<&[bool]>) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>;

    /// The kinetic energy's.
    fn contract_kinetic_gradient<Mg, Md>(&mut self, gradient: &mut Mg, density: &Md, active: Option<&[bool]>) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>;

    /// The point-charge potential's.
    fn contract_point_charge_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        charges: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::PointChargeTrait;

    /// The point-multipole potential's.
    fn contract_external_multipole_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        sources: &[P],
        convention: basis::charge_carriers::MultipoleConvention,
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait;

    /// p.Vp of the point-charge potential's (the picture-change integrals' derivatives).
    fn contract_pvp_point_charge_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        charges: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::PointChargeTrait;

    /// p.Vp of the AIMP screens' potential's (`screened_coulomb`), or of the AIMPs' whole Coulomb potential's, the
    /// bare charge and the screen.
    fn contract_pvp_aimp_coulomb_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        aimps: &[P],
        screen_only: bool,
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::ScreenedChargeTrait;

    /// p.Vp of the Gaussian-nucleus potential's.
    fn contract_pvp_finite_charge_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        charges: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::FiniteChargeTrait;

    /// p.Vp of the point-multipole potential's, in the given convention.
    fn contract_pvp_external_multipole_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        sources: &[P],
        convention: basis::charge_carriers::MultipoleConvention,
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait;

    /// The Gaussian-nucleus potential's.
    fn contract_finite_charge_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        charges: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::FiniteChargeTrait;

    /// The AIMP screening corrections'.
    fn contract_screened_coulomb_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        aimps: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::ScreenedChargeTrait;
}

impl<'a, S: basis::contraction::ShellKind> OneElectronGradientContractions for OneIntWorkStruct<'a, S> {
    fn contract_overlap_gradient<Mg, Md>(&mut self, gradient: &mut Mg, density: &Md, active: Option<&[bool]>) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
    {
        let _span = instrumentation::span("oneint::contract_overlap_gradient");
        gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, gradient_kernels::OverlapGradient>(
            gradient, density, self.shell_pairs, &[], true, active, &mut self.scratch, &self.comm, &additional_arguments::EmptyAdditionalArguments {},
        )
    }

    fn contract_kinetic_gradient<Mg, Md>(&mut self, gradient: &mut Mg, density: &Md, active: Option<&[bool]>) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
    {
        let _span = instrumentation::span("oneint::contract_kinetic_gradient");
        gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, gradient_kernels::KineticGradient>(
            gradient, density, self.shell_pairs, &[], true, active, &mut self.scratch, &self.comm, &additional_arguments::EmptyAdditionalArguments {},
        )
    }

    fn contract_point_charge_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        charges: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::PointChargeTrait,
    {
        let _span = instrumentation::span("oneint::contract_point_charge_gradient");
        gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, gradient_kernels::CoulombGradientKernel<P>>(
            gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &charges,
        )
    }

    fn contract_external_multipole_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        sources: &[P],
        convention: basis::charge_carriers::MultipoleConvention,
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait,
    {
        let _span = instrumentation::span("oneint::contract_external_multipole_gradient");
        match convention {
            basis::charge_carriers::MultipoleConvention::ContactFree => gradient_contraction::one_electron_gradient_contraction_loop::<
                Mg, Md, S, gradient_kernels::CoulombGradientKernel<P, kernels::MultipoleSourceMode<kernels::ContactFree>>
            >(gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &sources),
            basis::charge_carriers::MultipoleConvention::Exact => gradient_contraction::one_electron_gradient_contraction_loop::<
                Mg, Md, S, gradient_kernels::CoulombGradientKernel<P, kernels::MultipoleSourceMode<kernels::Exact>>
            >(gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &sources),
        }
    }

    fn contract_pvp_point_charge_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        charges: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::PointChargeTrait,
    {
        let _span = instrumentation::span("oneint::contract_pvp_point_charge_gradient");
        gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, picture_change_kernels::PvpGradientKernel<P>>(
            gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &charges,
        )
    }

    fn contract_pvp_aimp_coulomb_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        aimps: &[P],
        screen_only: bool,
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::ScreenedChargeTrait,
    {
        let _span = instrumentation::span("oneint::contract_pvp_aimp_coulomb_gradient");
        if screen_only {
            gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, picture_change_kernels::PvpGradientKernel<P, kernels::ScreenMode>>(
                gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &aimps,
            )
        } else {
            gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, picture_change_kernels::PvpGradientKernel<P, kernels::AimpCoulombMode>>(
                gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &aimps,
            )
        }
    }

    fn contract_pvp_finite_charge_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        charges: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::FiniteChargeTrait,
    {
        let _span = instrumentation::span("oneint::contract_pvp_finite_charge_gradient");
        gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, picture_change_kernels::PvpGradientKernel<P, kernels::FiniteMode>>(
            gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &charges,
        )
    }

    fn contract_pvp_external_multipole_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        sources: &[P],
        convention: basis::charge_carriers::MultipoleConvention,
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait,
    {
        let _span = instrumentation::span("oneint::contract_pvp_external_multipole_gradient");
        match convention {
            basis::charge_carriers::MultipoleConvention::ContactFree => gradient_contraction::one_electron_gradient_contraction_loop::<
                Mg, Md, S, picture_change_kernels::PvpGradientKernel<P, kernels::MultipoleSourceMode<kernels::ContactFree>>
            >(gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &sources),
            basis::charge_carriers::MultipoleConvention::Exact => gradient_contraction::one_electron_gradient_contraction_loop::<
                Mg, Md, S, picture_change_kernels::PvpGradientKernel<P, kernels::MultipoleSourceMode<kernels::Exact>>
            >(gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &sources),
        }
    }

    fn contract_finite_charge_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        charges: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::FiniteChargeTrait,
    {
        let _span = instrumentation::span("oneint::contract_finite_charge_gradient");
        gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, gradient_kernels::CoulombGradientKernel<P, kernels::FiniteMode>>(
            gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &charges,
        )
    }

    fn contract_screened_coulomb_gradient<Mg, Md, P>(
        &mut self,
        gradient: &mut Mg,
        density: &Md,
        aimps: &[P],
        operator_centres: &[(usize, usize)],
        active: Option<&[bool]>,
    ) -> Result<(), arrays::scratch::ArenaError>
    where
        Mg: arrays::matrix::MatrixTraitMut<f64>,
        Md: arrays::matrix::MatrixTrait<f64>,
        P: basis::charge_carriers::ScreenedChargeTrait,
    {
        let _span = instrumentation::span("oneint::contract_screened_coulomb_gradient");
        gradient_contraction::one_electron_gradient_contraction_loop::<Mg, Md, S, gradient_kernels::CoulombGradientKernel<P, kernels::ScreenMode>>(
            gradient, density, self.shell_pairs, operator_centres, false, active, &mut self.scratch, &self.comm, &aimps,
        )
    }
}

/// The picture-change integrals of the X2C transformation. The scalar ones are p.Vp, <A| p.V p |B>, of the
/// potential V of the corresponding plain operator; the spin-orbit ones are the four real components of
/// sigma.p V sigma.p = p.Vp + i sigma.(p x Vp): planes 0, 1, 2 the antisymmetric (p x Vp)_x,y,z, plane 3 p.Vp.
pub trait OneElectronPictureChangeIntegrals {
    /// p.Vp of the point-charge potential.
    fn compute_pvp_scalar_point_charge_integrals<Mf64, P>(&mut self, output: &mut Mf64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::PointChargeTrait;
    /// p.Vp of the Gaussian-nucleus potential.
    fn compute_pvp_scalar_finite_charge_integrals<Mf64, P>(&mut self, output: &mut Mf64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::FiniteChargeTrait;
    /// sigma.p V sigma.p of the point-charge potential, four planes.
    fn compute_pvp_spin_orbit_point_charge_integrals<Af64, P>(&mut self, output: &mut Af64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        P: basis::charge_carriers::PointChargeTrait;
    /// sigma.p V sigma.p of the Gaussian-nucleus potential, four planes.
    fn compute_pvp_spin_orbit_finite_charge_integrals<Af64, P>(&mut self, output: &mut Af64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        P: basis::charge_carriers::FiniteChargeTrait;
    /// p.Vp of the AIMP screens' potential (`screen_only`), or of the AIMPs' whole Coulomb potential, the bare
    /// charge and the screen.
    fn compute_pvp_scalar_aimp_coulomb_integrals<Mf64, P>(&mut self, output: &mut Mf64, aimps: &[P], screen_only: bool) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::ScreenedChargeTrait;
    /// sigma.p V sigma.p of the same, four planes.
    fn compute_pvp_spin_orbit_aimp_coulomb_integrals<Af64, P>(&mut self, output: &mut Af64, aimps: &[P], screen_only: bool) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        P: basis::charge_carriers::ScreenedChargeTrait;
    /// p.Vp of the point-multipole potential, in the given convention.
    fn compute_pvp_scalar_external_multipole_integrals<Mf64, P>(&mut self, output: &mut Mf64, sources: &[P], convention: basis::charge_carriers::MultipoleConvention) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait;
    /// sigma.p V sigma.p of the point-multipole potential, in the given convention, four planes.
    fn compute_pvp_spin_orbit_external_multipole_integrals<Af64, P>(&mut self, output: &mut Af64, sources: &[P], convention: basis::charge_carriers::MultipoleConvention) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait;
    /// p.mp of the moment operator m about the origin, one plane per Cartesian component of its order.
    fn compute_prp_scalar_integrals<Af64>(&mut self, output: &mut Af64, origin: &MomentOrigin) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>;
    /// sigma.p m sigma.p of the moment operator m about the origin: four planes per Cartesian component, plane
    /// 4 * component + piece.
    fn compute_prp_spin_orbit_integrals<Af64>(&mut self, output: &mut Af64, origin: &MomentOrigin) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>;
}

impl<'a, S: basis::contraction::ShellKind> OneElectronPictureChangeIntegrals for OneIntWorkStruct<'a, S> {
    fn compute_pvp_scalar_point_charge_integrals<Mf64, P>(&mut self, output: &mut Mf64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::PointChargeTrait
    {
        let _span = instrumentation::span("oneint::pvp_scalar_point_charge");
        one_electron_integral_loop::<
            Mf64,
            S,
            picture_change_kernels::PvpKernel<P>
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &charges
            )
    }

    fn compute_pvp_scalar_finite_charge_integrals<Mf64, P>(&mut self, output: &mut Mf64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::FiniteChargeTrait
    {
        let _span = instrumentation::span("oneint::pvp_scalar_finite_charge");
        one_electron_integral_loop::<
            Mf64,
            S,
            picture_change_kernels::PvpKernel<P, kernels::FiniteMode>
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &charges
            )
    }

    fn compute_pvp_spin_orbit_point_charge_integrals<Af64, P>(&mut self, output: &mut Af64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        P: basis::charge_carriers::PointChargeTrait
    {
        let _span = instrumentation::span("oneint::pvp_spin_orbit_point_charge");
        one_electron_integral_loop_array::<
            Af64,
            S,
            picture_change_kernels::PvpFullKernel<P>
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &charges
            )
    }

    fn compute_pvp_spin_orbit_finite_charge_integrals<Af64, P>(&mut self, output: &mut Af64, charges: &[P]) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        P: basis::charge_carriers::FiniteChargeTrait
    {
        let _span = instrumentation::span("oneint::pvp_spin_orbit_finite_charge");
        one_electron_integral_loop_array::<
            Af64,
            S,
            picture_change_kernels::PvpFullKernel<P, kernels::FiniteMode>
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &charges
            )
    }

    fn compute_pvp_scalar_aimp_coulomb_integrals<Mf64, P>(&mut self, output: &mut Mf64, aimps: &[P], screen_only: bool) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::ScreenedChargeTrait
    {
        let _span = instrumentation::span("oneint::pvp_scalar_aimp_coulomb");
        if screen_only {
            one_electron_integral_loop::<Mf64, S, picture_change_kernels::PvpKernel<P, kernels::ScreenMode>>(
                output, self.shell_pairs, &mut self.scratch, &self.comm, &aimps,
            )
        } else {
            one_electron_integral_loop::<Mf64, S, picture_change_kernels::PvpKernel<P, kernels::AimpCoulombMode>>(
                output, self.shell_pairs, &mut self.scratch, &self.comm, &aimps,
            )
        }
    }

    fn compute_pvp_spin_orbit_aimp_coulomb_integrals<Af64, P>(&mut self, output: &mut Af64, aimps: &[P], screen_only: bool) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        P: basis::charge_carriers::ScreenedChargeTrait
    {
        let _span = instrumentation::span("oneint::pvp_spin_orbit_aimp_coulomb");
        if screen_only {
            one_electron_integral_loop_array::<Af64, S, picture_change_kernels::PvpFullKernel<P, kernels::ScreenMode>>(
                output, self.shell_pairs, &mut self.scratch, &self.comm, &aimps,
            )
        } else {
            one_electron_integral_loop_array::<Af64, S, picture_change_kernels::PvpFullKernel<P, kernels::AimpCoulombMode>>(
                output, self.shell_pairs, &mut self.scratch, &self.comm, &aimps,
            )
        }
    }

    fn compute_pvp_scalar_external_multipole_integrals<Mf64, P>(&mut self, output: &mut Mf64, sources: &[P], convention: basis::charge_carriers::MultipoleConvention) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait
    {
        let _span = instrumentation::span("oneint::pvp_scalar_external_multipole");
        use basis::charge_carriers::MultipoleConvention::{ContactFree, Exact};
        use kernels::MultipoleSourceMode;
        match convention {
            ContactFree => one_electron_integral_loop::<Mf64, S, picture_change_kernels::PvpKernel<P, MultipoleSourceMode<kernels::ContactFree>>>(output, self.shell_pairs, &mut self.scratch, &self.comm, &sources),
            Exact => one_electron_integral_loop::<Mf64, S, picture_change_kernels::PvpKernel<P, MultipoleSourceMode<kernels::Exact>>>(output, self.shell_pairs, &mut self.scratch, &self.comm, &sources),
        }
    }

    fn compute_pvp_spin_orbit_external_multipole_integrals<Af64, P>(&mut self, output: &mut Af64, sources: &[P], convention: basis::charge_carriers::MultipoleConvention) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>,
        P: basis::charge_carriers::MultipoleSourceTrait
    {
        let _span = instrumentation::span("oneint::pvp_spin_orbit_external_multipole");
        use basis::charge_carriers::MultipoleConvention::{ContactFree, Exact};
        use kernels::MultipoleSourceMode;
        match convention {
            ContactFree => one_electron_integral_loop_array::<Af64, S, picture_change_kernels::PvpFullKernel<P, MultipoleSourceMode<kernels::ContactFree>>>(output, self.shell_pairs, &mut self.scratch, &self.comm, &sources),
            Exact => one_electron_integral_loop_array::<Af64, S, picture_change_kernels::PvpFullKernel<P, MultipoleSourceMode<kernels::Exact>>>(output, self.shell_pairs, &mut self.scratch, &self.comm, &sources),
        }
    }

    fn compute_prp_scalar_integrals<Af64>(&mut self, output: &mut Af64, origin: &MomentOrigin) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::prp_scalar");
        one_electron_integral_loop_array::<
            Af64,
            S,
            picture_change_kernels::MomentPXPKernel
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &origin
            )
    }

    fn compute_prp_spin_orbit_integrals<Af64>(&mut self, output: &mut Af64, origin: &MomentOrigin) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::prp_spin_orbit");
        one_electron_integral_loop_array::<
            Af64,
            S,
            picture_change_kernels::MomentFullKernel
            >(
                output,
                self.shell_pairs,
                &mut self.scratch, &self.comm,
                &origin
            )
    }
}

// The pieces of a loop over `shell_pairs` against `n_sites` sites, in the order they are handed out
// (`pieces::sorted_pieces`).
fn pair_pieces<S: basis::contraction::ShellKind>(shell_pairs: &[basis::shell_pairs::ShellPair<S>], n_sites: usize, comm: &Communicator) -> Vec<pieces::Piece> {
    let costs: Vec<f64> = shell_pairs.iter().map(pieces::pair_cost).collect();
    pieces::sorted_pieces(&costs, n_sites, usize::MAX, pieces::Cut::PairsFirst, comm)
}

fn one_electron_integral_loop<Mf64, S, K>(
    output: &mut Mf64,
    shell_pairs: &[basis::shell_pairs::ShellPair<S>],
    scratch: &mut arrays::scratch::Scratch,
    comm: &Communicator,
    additional_arguments: &K::AdditionalArguments<'_>
) -> Result<(), arrays::scratch::ArenaError>
where
    Mf64: arrays::matrix::MatrixTraitMut<f64>,
    S: basis::contraction::ShellKind,
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>,
{
    // The loops write only the blocks of the pairs in the list (and of their products not screened out).
    output.fill(0.0);
    if comm.size() > 1 {
        one_electron_integrals_parallel_loop::<Mf64, S, K>(
            output,
            shell_pairs,
            scratch,
            comm,
            additional_arguments
        )?;
    } else {
        one_electron_integrals_serial_loop::<Mf64, S, K>(
            output,
            shell_pairs,
            scratch,
            additional_arguments
        )?;
    }
    output.copy_upper_to_lower();
    Ok(())
}

// `one_electron_integrals_serial_loop` under MPI: each rank computes its own shell pairs into `output`, zeroed by
// the caller.
// Distinct pairs write disjoint blocks of the upper triangle, so an allreduce sum assembles it on every rank.
fn one_electron_integrals_parallel_loop<Mf64, S, K>(
    output: &mut Mf64,
    shell_pairs: &[basis::shell_pairs::ShellPair<S>],
    scratch: &mut arrays::scratch::Scratch,
    comm: &Communicator,
    additional_arguments: &K::AdditionalArguments<'_>
) -> Result<(), arrays::scratch::ArenaError>
where
    Mf64: arrays::matrix::MatrixTraitMut<f64>,
    S: basis::contraction::ShellKind,
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>,
{
    let n_sites = if K::SUMS_OVER_SITES { additional_arguments.n_sites() } else { 0 };
    let pieces = pair_pieces(shell_pairs, n_sites, comm);
    let handout = pieces::Handout::new(&pieces, comm);
    for piece in handout.iter() {
        let arguments = additional_arguments.run(if K::SUMS_OVER_SITES { piece.sites } else { 0..additional_arguments.n_sites() });
        for pair in &shell_pairs[piece.pairs] {
            let arena = &scratch.get_arena();
            let batch = inner_functions::pair_batch::<K, S>(pair, arena, &arguments)?;
            let _span = instrumentation::fine_span("oneint::distribute");
            distribute::distribute_pair_batch(pair, output, &batch);
        }
    }
    let _allreduce = instrumentation::span("oneint::allreduce");
    comm.sum(output.get_data_mut())
        .expect("MPI allreduce failed while combining one-electron integrals across ranks");
    Ok(())
}

// Stops at the first shell pair whose batch does not fit the scratch and returns the error; `output` is then
// partially written.
fn one_electron_integrals_serial_loop<Mf64, S, K>(
    output: &mut Mf64,
    shell_pairs: &[basis::shell_pairs::ShellPair<S>],
    scratch: &mut arrays::scratch::Scratch,
    additional_arguments: &K::AdditionalArguments<'_>
) -> Result<(), arrays::scratch::ArenaError>
where
    Mf64: arrays::matrix::MatrixTraitMut<f64>,
    S: basis::contraction::ShellKind,
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>,
{
    for pair in shell_pairs.iter() {
        // A fresh arena over the whole scratch for every pair.
        {
            let arena = &scratch.get_arena();
            let batch = inner_functions::pair_batch::<K, S>(pair, arena, additional_arguments)?;
            let _span = instrumentation::fine_span("oneint::distribute");
            distribute::distribute_pair_batch(pair, output, &batch);
        }
    }
    Ok(())
}

// `one_electron_integral_loop` for the kernels with several planes (`ArrayBatchKernel`).
fn one_electron_integral_loop_array<Af64, S, K>(
    output: &mut Af64,
    shell_pairs: &[basis::shell_pairs::ShellPair<S>],
    scratch: &mut arrays::scratch::Scratch,
    comm: &Communicator,
    additional_arguments: &K::AdditionalArguments<'_>
) -> Result<(), arrays::scratch::ArenaError>
where
    Af64: arrays::array3d::Array3DTraitMut<f64>,
    S: basis::contraction::ShellKind,
    K: ArrayBatchKernel
{
    output.fill(0.0);
    if comm.size() > 1 {
        one_electron_integrals_parallel_loop_array::<Af64, S, K>(
            output,
            shell_pairs,
            scratch,
            comm,
            additional_arguments
        )?;
    } else {
        one_electron_integrals_serial_loop_array::<Af64, S, K>(
            output,
            shell_pairs,
            scratch,
            additional_arguments
        )?;
    }
    // Each plane's upper triangle mirrored to its lower one; an antisymmetric plane's strictly lower triangle is
    // then negated (the diagonal blocks of same-shell pairs, placed whole, included).
    let n_planes = output.dim2();
    for k in 0..n_planes {
        let mut plane = output.view_sub_matrix_mut(k);
        plane.copy_upper_to_lower();
        if K::antisymmetric_plane(k) {
            let n = arrays::matrix::MatrixTrait::rows(&plane);
            for j in 0..n {
                for i in (j + 1)..n {
                    plane[(i, j)] = -plane[(i, j)];
                }
            }
        }
    }
    Ok(())
}

fn one_electron_integrals_serial_loop_array<Af64, S, K>(
    output: &mut Af64,
    shell_pairs: &[basis::shell_pairs::ShellPair<S>],
    scratch: &mut arrays::scratch::Scratch,
    additional_arguments: &K::AdditionalArguments<'_>
) -> Result<(), arrays::scratch::ArenaError>
where
    Af64: arrays::array3d::Array3DTraitMut<f64>,
    S: basis::contraction::ShellKind,
    K: ArrayBatchKernel
{
    let n_planes = K::n_planes(additional_arguments);
    for pair in shell_pairs.iter() {
        {
            let arena = &scratch.get_arena();
            let mut batch = inner_functions::pair_batch_planar::<K, S>(pair, arena, additional_arguments)?;
            negate_antisymmetric_planes_of_swapped_pair::<S, K>(pair, &mut batch, n_planes);
            let _span = instrumentation::fine_span("oneint::distribute");
            distribute::distribute_pair_batch_planar(pair, output, &batch);
        }
    }
    Ok(())
}

/// A pair whose first shell has the higher basis index was computed as <a|O|b> but is placed as the
/// transpose into the upper triangle; for an antisymmetric plane that transpose is -<b|O|a>, so the
/// batch is negated first (see `ArrayBatchKernel::antisymmetric_plane`).
fn negate_antisymmetric_planes_of_swapped_pair<S, K>(pair: &basis::shell_pairs::ShellPair<S>, batch: &mut arrays::array5d::Array5DViewMut<'_, f64>, n_planes: usize)
where
    S: basis::contraction::ShellKind,
    K: ArrayBatchKernel,
{
    if pair.get_idx_a() > pair.get_idx_b() {
        for k in 0..n_planes {
            if K::antisymmetric_plane(k) {
                distribute::negate_plane(batch, k);
            }
        }
    }
}

// `one_electron_integrals_parallel_loop` for the kernels with several planes, all planes in one allreduce.
fn one_electron_integrals_parallel_loop_array<Af64, S, K>(
    output: &mut Af64,
    shell_pairs: &[basis::shell_pairs::ShellPair<S>],
    scratch: &mut arrays::scratch::Scratch,
    comm: &Communicator,
    additional_arguments: &K::AdditionalArguments<'_>
) -> Result<(), arrays::scratch::ArenaError>
where
    Af64: arrays::array3d::Array3DTraitMut<f64>,
    S: basis::contraction::ShellKind,
    K: ArrayBatchKernel
{
    let n_planes = K::n_planes(additional_arguments);
    let n_sites = if K::SUMS_OVER_SITES { additional_arguments.n_sites() } else { 0 };
    let pieces = pair_pieces(shell_pairs, n_sites, comm);
    let handout = pieces::Handout::new(&pieces, comm);
    for piece in handout.iter() {
        let arguments = additional_arguments.run(if K::SUMS_OVER_SITES { piece.sites } else { 0..additional_arguments.n_sites() });
        for pair in &shell_pairs[piece.pairs] {
            let arena = &scratch.get_arena();
            let mut batch = inner_functions::pair_batch_planar::<K, S>(pair, arena, &arguments)?;
            negate_antisymmetric_planes_of_swapped_pair::<S, K>(pair, &mut batch, n_planes);
            let _span = instrumentation::fine_span("oneint::distribute");
            distribute::distribute_pair_batch_planar(pair, output, &batch);
        }
    }
    let _allreduce = instrumentation::span("oneint::allreduce");
    comm.sum(output.get_data_mut())
        .expect("MPI allreduce failed while combining one-electron integrals across ranks");
    Ok(())
}

/// Two shell lists, possibly of different bases, and a scratch buffer: the integrals between them.
pub struct OneIntProjectionWorkStruct<'a, Sa: basis::contraction::ShellKind, Sb: basis::contraction::ShellKind> {
    shells_a: &'a [basis::shell_types::BasisShell<Sa>],
    shells_b: &'a [basis::shell_types::BasisShell<Sb>],
    scratch: &'a mut arrays::scratch::Scratch,
}

impl<'a, Sa: basis::contraction::ShellKind, Sb: basis::contraction::ShellKind> OneIntProjectionWorkStruct<'a, Sa, Sb> {
    /// Between `shells_a` and `shells_b`, with `scratch` as scratch.
    pub fn new(shells_a: &'a [basis::shell_types::BasisShell<Sa>], shells_b: &'a [basis::shell_types::BasisShell<Sb>], scratch: &'a mut arrays::scratch::Scratch) -> Self {
        Self { shells_a, shells_b, scratch }
    }
}

/// One-electron integrals between two shell lists: `output` is (n_AO of a, n_AO of b).
pub trait OneElectronProjectionIntegrals {
    /// The overlap, <A|B>.
    fn compute_overlap_integrals<Mf64>(&mut self, output: &mut Mf64) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>;
}

impl<'a, Sa: basis::contraction::ShellKind, Sb: basis::contraction::ShellKind> OneElectronProjectionIntegrals for OneIntProjectionWorkStruct<'a, Sa, Sb> {
    fn compute_overlap_integrals<Mf64>(&mut self, output: &mut Mf64) -> Result<(), arrays::scratch::ArenaError>
    where
        Mf64: arrays::matrix::MatrixTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::projection_overlap");
        one_electron_two_shell_integral_loop::<Mf64, Sa, Sb, kernels::OverlapKernel>(output, self.shells_a, self.shells_b, &mut self.scratch, &additional_arguments::EmptyAdditionalArguments {})
    }
}

// Integrals between two shell lists, of any contraction kinds, over shells rather than shell pairs.
fn one_electron_two_shell_integral_loop<Mf64, Sa, Sb, K>(
    output: &mut Mf64,
    basis_shells_a: &[basis::shell_types::BasisShell<Sa>],
    basis_shells_b: &[basis::shell_types::BasisShell<Sb>],
    scratch: &mut arrays::scratch::Scratch,
    additional_arguments: &K::AdditionalArguments<'_>
) -> Result<(), arrays::scratch::ArenaError>
where
    Mf64: arrays::matrix::MatrixTraitMut<f64>,
    Sa: basis::contraction::ShellKind,
    Sb: basis::contraction::ShellKind,
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>
{
    // Not divided among MPI ranks: every rank computes every pair of shells. The batches are added.
    output.fill(0.0);
    let serial = true;
    if serial {
        one_electron_two_shell_integrals_serial_loop::<Mf64, Sa, Sb, K>(
            output,
            basis_shells_a,
            basis_shells_b,
            scratch,
            additional_arguments
        )?;
    } else {
        unimplemented!("Parallel loop not yet implemented for one-electron integrals.");
    }
    // `output` is rectangular, every element computed: no triangle to mirror.
    Ok(())
}

fn one_electron_two_shell_integrals_serial_loop<Mf64, Sa, Sb, K>(
    output: &mut Mf64,
    basis_shells_a: &[basis::shell_types::BasisShell<Sa>],
    basis_shells_b: &[basis::shell_types::BasisShell<Sb>],
    scratch: &mut arrays::scratch::Scratch,
    additional_arguments: &K::AdditionalArguments<'_>
 ) -> Result<(), arrays::scratch::ArenaError>
where
    Mf64: arrays::matrix::MatrixTraitMut<f64>,
    Sa: basis::contraction::ShellKind,
    Sb: basis::contraction::ShellKind,
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = arrays::matrix::MatrixViewMut<'b, f64>>
{
    for shell_a in basis_shells_a.iter() {
        for shell_b in basis_shells_b.iter() {
            // A fresh arena over the whole scratch for every pair.
            {
                let arena = &scratch.get_arena();
                let batch = inner_functions::two_shell_batch::<K, Sa, Sb>(shell_a, shell_b, arena, additional_arguments)?;
                let _span = instrumentation::fine_span("oneint::distribute");
                distribute::distribute_array4d_to_matrix(output, &batch, shell_a.get_basis_index(), shell_b.get_basis_index());
            }
        }
    }
    Ok(())
}

/// Derivative integrals between two shell lists with respect to one centre: `output` (n_AO of a, n_AO of b, 3).
pub trait OneElectronProjectionIntegralGradients {
    /// The derivatives of the overlap with respect to the centre with the global index `centre_gradient_idx`.
    fn compute_overlap_gradient_integrals<Af64>(&mut self, output: &mut Af64, centre_gradient_idx: usize) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>;
}

impl<'a, Sa: basis::contraction::ShellKind, Sb: basis::contraction::ShellKind> OneElectronProjectionIntegralGradients for OneIntProjectionWorkStruct<'a, Sa, Sb> {
    fn compute_overlap_gradient_integrals<Af64>(&mut self, output: &mut Af64, centre_gradient_idx: usize) -> Result<(), arrays::scratch::ArenaError>
    where
        Af64: arrays::array3d::Array3DTraitMut<f64>
    {
        let _span = instrumentation::span("oneint::projection_overlap_gradient");
        one_electron_two_shell_gradient_integral_loop::<Af64, Sa, Sb, gradient_kernels::OverlapGradient>(output, self.shells_a, self.shells_b, centre_gradient_idx, &mut self.scratch, &additional_arguments::EmptyAdditionalArguments {})
    }
}

fn one_electron_two_shell_gradient_integral_loop<Af64, Sa, Sb, K>(
    output: &mut Af64,
    basis_shells_a: &[basis::shell_types::BasisShell<Sa>],
    basis_shells_b: &[basis::shell_types::BasisShell<Sb>],
    centre_gradient_idx: usize,
    scratch: &mut arrays::scratch::Scratch,
    additional_arguments: &K::AdditionalArguments<'_>
) -> Result<(), arrays::scratch::ArenaError>
where
    Af64: arrays::array3d::Array3DTraitMut<f64>,
    Sa: basis::contraction::ShellKind,
    Sb: basis::contraction::ShellKind,
    K: gradient_kernels::ArrayBatchGradientKernel
{
    output.fill(0.0);
    for shell_a in basis_shells_a.iter() {
        for shell_b in basis_shells_b.iter() {
            {
                let arena = &scratch.get_arena();
                let batch = inner_functions::two_shell_gradient_batch::<K, Sa, Sb>(shell_a, shell_b, centre_gradient_idx, arena, additional_arguments)?;
                let _span = instrumentation::fine_span("oneint::distribute");
                distribute::distribute_two_shell_array5d_to_array3d(output, &batch, shell_a.get_basis_index(), shell_b.get_basis_index());
            }
        }
    }
    Ok(())
}
