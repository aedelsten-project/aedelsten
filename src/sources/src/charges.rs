// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use crate::sites::{Fragments, NO_INPUT_INDEX, SiteTrait, input_index_from_storage};

/// A site carrying a charge.
pub trait PointChargeTrait: SiteTrait {
    /// The charge, in units of e.
    fn get_charge(&self) -> f64;
}

/// A point charge with a screening correction on top of its bare 1/r, i.e. the potential per unit
/// charge (1 + Sum_i A_i exp(-c_i r^2))/r: what an AIMP looks like to a classical charge.
pub trait ScreenedChargeTrait: PointChargeTrait {
    /// The screening correction.
    fn get_screened_coulomb_potential(&self) -> &ScreenedCoulombPotential;
}

/// A contracted series of spherical Gaussian charge distributions on one site, Sum_i q_i (e_i/pi)^(3/2) exp(-e_i r^2),
/// whose potential is Sum_i q_i erf(sqrt(e_i) r)/r: a Visscher-Dyall finite nucleus (one term), or an atom's electrons
/// fitted to Gaussian charges. `get_charge` is the total, Sum_i q_i.
pub trait FiniteChargeTrait: PointChargeTrait {
    /// The exponents e_i.
    fn get_exponents(&self) -> &[f64];
    /// The charges q_i, one per exponent.
    fn get_charges(&self) -> &[f64];
}

/// A point source with Cartesian moments up to a maximum order: a point charge (order 0) or a point
/// multipole. The moments are bare Cartesian moment tensor components, Sum q (x-Cx)^lx (y-Cy)^ly (z-Cz)^lz,
/// without the 1/(lx! ly! lz!) of the Taylor expansion.
pub trait MultipoleSourceTrait: PointChargeTrait {
    /// The highest order of this source's moments.
    fn get_max_order(&self) -> usize;
    /// The moment (lx, ly, lz), for lx + ly + lz <= `get_max_order()`.
    fn get_moment(&self, lx: usize, ly: usize, lz: usize) -> f64;

    /// The moment with the trace parts of its order removed (its harmonic part): what the electrons interact
    /// with in the contact-free convention (`MultipoleConvention::ContactFree`). Equal to `get_moment` up to
    /// order 1.
    fn get_contact_free_moment(&self, lx: usize, ly: usize, lz: usize) -> f64 {
        debug_assert!(lx + ly + lz <= 1, "sources of higher order must provide their contact-free moments");
        self.get_moment(lx, ly, lz)
    }
}

impl MultipoleSourceTrait for PointCharge {
    fn get_max_order(&self) -> usize { 0 }
    fn get_moment(&self, lx: usize, ly: usize, lz: usize) -> f64 {
        match (lx, ly, lz) {
            (0, 0, 0) => self.charge,
            _ => unreachable!("PointCharge has no moment beyond order 0"),
        }
    }
}

/// A point charge belonging to a centre of the molecule (a point nucleus).
#[derive(Debug, Clone)]
pub struct IndexedPointCharge {
    index: usize,
    position: [f64; 3],
    charge: f64,
}

impl IndexedPointCharge {
    /// A charge on centre `index`.
    pub fn new(index: usize, position: [f64; 3], charge: f64) -> Self {
        Self { index, position, charge }
    }

    /// The centre's index.
    pub fn get_idx(&self) -> usize {
        self.index
    }

    /// Moves the charge with its centre (geometry updates).
    pub fn set_position(&mut self, position: [f64; 3]) {
        self.position = position;
    }
}

impl SiteTrait for IndexedPointCharge {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }
}

impl PointChargeTrait for IndexedPointCharge {
    fn get_charge(&self) -> f64 {
        self.charge
    }
}

/// An environment point charge.
#[derive(Clone, Debug)]
pub struct PointCharge {
    position: [f64; 3],
    charge: f64,
    fragments: Fragments,
    input_index: usize,
}

impl PointCharge {
    /// A charge at `position` (bohr), without fragments.
    pub fn new(position: [f64; 3], charge: f64) -> Self {
        Self { position, charge, fragments: Fragments::NONE, input_index: NO_INPUT_INDEX }
    }

    /// The fragments whose polarizing field this charge must not contribute to.
    pub fn with_fragments(mut self, fragments: Fragments) -> Self {
        self.fragments = fragments;
        self
    }

    /// The record of the input file this charge was read from (see `SiteTrait::get_input_index`).
    pub fn with_input_index(mut self, index: usize) -> Self {
        self.input_index = index;
        self
    }
}

impl SiteTrait for PointCharge {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }

    fn get_fragments(&self) -> &[i32] {
        self.fragments.as_slice()
    }

    fn get_input_index(&self) -> Option<usize> {
        input_index_from_storage(self.input_index)
    }
}

impl PointChargeTrait for PointCharge {
    fn get_charge(&self) -> f64 {
        self.charge
    }
}

/// The screening correction of an AIMP, s(r) = Sum_i A_i exp(-c_i r^2): the AIMP of charge Z has the potential
/// Z (1 + s(r))/r, so the coefficients A_i are relative to Z.
#[derive(Clone, Debug)]
pub struct ScreenedCoulombPotential {
    exponents: Box<[f64]>,
    coefficients: Box<[f64]>,
}

impl ScreenedCoulombPotential {
    /// From the exponents c_i and the coefficients A_i, of equal length.
    pub fn new(exponents: Vec<f64>, coefficients: Vec<f64>) -> Self {
        Self {
            exponents: exponents.into_boxed_slice(),
            coefficients: coefficients.into_boxed_slice(),
        }
    }

    /// The exponents c_i.
    pub fn get_exponents(&self) -> &[f64] {
        &self.exponents
    }

    /// The coefficients A_i.
    pub fn get_coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

/// A Gaussian charge distribution belonging to a centre of the molecule (a finite nucleus).
#[derive(Debug, Clone)]
pub struct IndexedGaussianCharge {
    pub(crate) index: usize,
    pub(crate) position: [f64; 3],
    pub(crate) charge: f64,
    pub(crate) exp: f64,
}

impl IndexedGaussianCharge {
    /// A charge on centre `index` with the Gaussian exponent `exp`.
    pub fn new(index: usize, position: [f64; 3], charge: f64, exp: f64) -> Self {
        Self { index, position, charge, exp }
    }

    /// The centre's index.
    pub fn get_idx(&self) -> usize {
        self.index
    }

    /// Moves the charge with its centre (geometry updates).
    pub fn set_position(&mut self, position: [f64; 3]) {
        self.position = position;
    }
}

impl SiteTrait for IndexedGaussianCharge {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }
}

impl PointChargeTrait for IndexedGaussianCharge {
    fn get_charge(&self) -> f64 {
        self.charge
    }
}

impl FiniteChargeTrait for IndexedGaussianCharge {
    fn get_exponents(&self) -> &[f64] {
        std::slice::from_ref(&self.exp)
    }

    fn get_charges(&self) -> &[f64] {
        std::slice::from_ref(&self.charge)
    }
}

/// A contracted series of Gaussian charge distributions belonging to a centre of the molecule (an atom's electrons as
/// fitted Gaussian charges). Holds `2 n_terms` values.
#[derive(Debug, Clone)]
pub struct IndexedContractedGaussianCharge {
    index: usize,
    position: [f64; 3],
    exponents: Box<[f64]>,
    charges: Box<[f64]>,
    total: f64,
}

impl IndexedContractedGaussianCharge {
    /// The series `charges[i]` (e_i/pi)^(3/2) exp(-`exponents[i]` r^2) on centre `index`; the two of equal length.
    pub fn new(index: usize, position: [f64; 3], exponents: Box<[f64]>, charges: Box<[f64]>) -> Self {
        assert_eq!(exponents.len(), charges.len(), "one charge per exponent");
        let total = charges.iter().sum();
        Self { index, position, exponents, charges, total }
    }

    /// The centre's index.
    pub fn get_idx(&self) -> usize {
        self.index
    }

    /// Moves the series with its centre (geometry updates).
    pub fn set_position(&mut self, position: [f64; 3]) {
        self.position = position;
    }
}

impl SiteTrait for IndexedContractedGaussianCharge {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }
}

impl PointChargeTrait for IndexedContractedGaussianCharge {
    fn get_charge(&self) -> f64 {
        self.total
    }
}

impl FiniteChargeTrait for IndexedContractedGaussianCharge {
    fn get_exponents(&self) -> &[f64] {
        &self.exponents
    }

    fn get_charges(&self) -> &[f64] {
        &self.charges
    }
}
