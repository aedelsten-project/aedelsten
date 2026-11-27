// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use crate::charges::{MultipoleSourceTrait, IndexedGaussianCharge, IndexedPointCharge, PointChargeTrait};
use crate::energy::*;

/// The nuclei of the QM region, as point charges or as Gaussian charge distributions.
#[derive(Debug, Clone)]
pub enum NuclearPotential {
    PointCharge(Box<[IndexedPointCharge]>),
    GaussianCharge(Box<[IndexedGaussianCharge]>),
}

/// The finite-nucleus models.
pub enum FiniteModel {
    /// Spherical Gaussian nuclear charge distributions (`citations.rs`: `VISSCHER_DYALL_1997`).
    VisscherDyall,
}

impl NuclearPotential {
    /// The nuclear repulsion energy, in hartree.
    pub fn compute_nuclear_repulsion_energy(&self) -> f64 {
        match self {
            NuclearPotential::PointCharge(atom_charges) => compute_classical_coulomb_self_interaction(atom_charges),
            NuclearPotential::GaussianCharge(atom_charges) => compute_gaussian_coulomb_self_interaction(atom_charges),
        }
    }

    /// The interaction energy of the nuclei with `point_charges`.
    pub fn compute_nuclear_interaction_energy<T: PointChargeTrait>(&self, point_charges: &[T]) -> f64 {
        match self {
            NuclearPotential::PointCharge(atom_charges) => compute_classical_coulomb_interaction(atom_charges, point_charges),
            NuclearPotential::GaussianCharge(atom_charges) => compute_gaussian_coulomb_interaction(atom_charges, point_charges),
        }
    }

    /// The interaction energy of the nuclei with screened charges (AIMPs), each nucleus in their full potential.
    pub fn compute_screened_coulomb_interaction<T: crate::ScreenedChargeTrait>(&self, screened_point_charge: &[T]) -> f64 {
        match self {
            NuclearPotential::PointCharge(atom_charges) => compute_screened_coulomb_interaction(atom_charges, screened_point_charge),
            NuclearPotential::GaussianCharge(atom_charges) => compute_gaussian_screened_coulomb_interaction(atom_charges, screened_point_charge),
        }
    }

    /// The interaction energy of the nuclei with point multipole sources.
    pub fn compute_general_multipole_interaction_energy<T: MultipoleSourceTrait>(&self, sources: &[T]) -> f64 {
        match self {
            NuclearPotential::PointCharge(atom_charges) => compute_general_multipole_interaction(atom_charges, sources),
            NuclearPotential::GaussianCharge(atom_charges) => compute_gaussian_general_multipole_interaction(atom_charges, sources),
        }
    }

    // The geometric gradients of the energies above: `add_*(i, g)` receives dE/dr of member i of that set; the
    // multipole sources and plain point charges passed in are fixed.

    /// The gradient of `compute_nuclear_repulsion_energy`.
    pub fn compute_nuclear_repulsion_gradient(&self, add: impl FnMut(usize, [f64; 3])) {
        match self {
            NuclearPotential::PointCharge(atom_charges) => crate::energy_gradient::classical_coulomb_self_interaction_gradient(atom_charges, add),
            NuclearPotential::GaussianCharge(atom_charges) => crate::energy_gradient::gaussian_coulomb_self_interaction_gradient(atom_charges, add),
        }
    }

    /// The nuclei's gradient of `compute_nuclear_interaction_energy`.
    pub fn compute_nuclear_interaction_gradient<T: PointChargeTrait>(&self, point_charges: &[T], add_nuclei: impl FnMut(usize, [f64; 3])) {
        match self {
            NuclearPotential::PointCharge(atom_charges) => {
                crate::energy_gradient::classical_coulomb_interaction_gradient(atom_charges, point_charges, add_nuclei, |_, _| {})
            }
            NuclearPotential::GaussianCharge(atom_charges) => {
                crate::energy_gradient::gaussian_coulomb_interaction_gradient(atom_charges, point_charges, add_nuclei, |_, _| {})
            }
        }
    }

    /// The gradient of `compute_screened_coulomb_interaction`, of the nuclei and of the screened charges.
    pub fn compute_screened_coulomb_interaction_gradient<T: crate::ScreenedChargeTrait>(
        &self,
        screened_point_charge: &[T],
        add_nuclei: impl FnMut(usize, [f64; 3]),
        add_screened: impl FnMut(usize, [f64; 3]),
    ) {
        match self {
            NuclearPotential::PointCharge(atom_charges) => {
                crate::energy_gradient::screened_coulomb_interaction_gradient(atom_charges, screened_point_charge, add_nuclei, add_screened)
            }
            NuclearPotential::GaussianCharge(atom_charges) => {
                crate::energy_gradient::gaussian_screened_coulomb_interaction_gradient(atom_charges, screened_point_charge, add_nuclei, add_screened)
            }
        }
    }

    /// The nuclei's gradient of `compute_general_multipole_interaction_energy`.
    pub fn compute_general_multipole_interaction_gradient<T: MultipoleSourceTrait>(&self, sources: &[T], add_nuclei: impl FnMut(usize, [f64; 3])) {
        match self {
            NuclearPotential::PointCharge(atom_charges) => crate::energy_gradient::general_multipole_interaction_gradient(atom_charges, sources, add_nuclei),
            NuclearPotential::GaussianCharge(atom_charges) => {
                crate::energy_gradient::gaussian_general_multipole_interaction_gradient(atom_charges, sources, add_nuclei)
            }
        }
    }
}
