// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The centres of a system: atoms, ghost centres and orthogonality ghosts.

use crate::charge_carriers::PointChargeTrait;

/// The centres of a system, in input order.
#[derive(Debug, Clone)]
pub struct Molecule {
    atoms: Box<[Atom]>,
}

impl Molecule {
    /// The system of `atoms`.
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            atoms: atoms.into_boxed_slice(),
        }
    }

    /// `get_atoms`, mutably.
    pub(crate) fn get_atoms_mut(&mut self) -> &mut [Atom] {
        &mut self.atoms
    }

    /// Every centre, in input order; a centre's position here is its centre index (`BasisShell::get_centre_index`,
    /// `IndexedAimp::get_idx`). Real atoms and ghost centres come first, then the orthogonality ghosts.
    pub fn get_atoms(&self) -> &[Atom] {
        &self.atoms
    }

    /// The real atoms only, in input order; ghost centres and orthogonality ghosts excluded.
    pub fn real_atoms(&self) -> impl Iterator<Item = &Atom> {
        self.atoms.iter().filter(|a| a.get_kind() == AtomKind::Real)
    }

    /// The Gaussian nuclei of the centres of `point_charges` under `model`, with their charges.
    pub fn derive_finite_nuclei(
        &self,
        point_charges: Box<[crate::charge_carriers::IndexedPointCharge]>,
        model: crate::charge_carriers::FiniteModel,
    ) -> Box<[crate::charge_carriers::IndexedGaussianCharge]> {
        match model {
            crate::charge_carriers::FiniteModel::VisscherDyall => {
                return self.derive_visscher_dyall_nuclear_exponents(point_charges);
            }
        }
    }

    fn derive_visscher_dyall_nuclear_exponents(
        &self,
        point_charges: Box<[crate::charge_carriers::IndexedPointCharge]>,
    ) -> Box<[crate::charge_carriers::IndexedGaussianCharge]> {
        // A charge's index is its centre index.
        let mut nuclear_exponents = Vec::with_capacity(point_charges.len());
        for charge in point_charges.iter() {
            let atom_idx = charge.get_idx();
            let atom = &self.atoms[atom_idx];
            // The mass number A, not the atomic number.
            let a = constants::MASS_NUMBERS[atom.get_atomic_number() as usize] as f64;
            let exponent = compute_visscher_dyall_nuclear_exponent(a);
            nuclear_exponents.push(crate::charge_carriers::IndexedGaussianCharge::new(
                atom_idx,
                atom.get_position(),
                charge.get_charge(),
                exponent,
            ));
        }
        nuclear_exponents.into_boxed_slice()
    }

    /// The centre of mass of the real atoms.
    pub fn compute_centre_of_mass(&self) -> [f64; 3] {
        let mut total_mass = 0.0;
        let mut centre_of_mass = [0.0; 3];
        for atom in self.real_atoms() {
            let pos = atom.get_position();
            let mass = atom.get_mass();
            centre_of_mass[0] += mass * pos[0];
            centre_of_mass[1] += mass * pos[1];
            centre_of_mass[2] += mass * pos[2];
            total_mass += mass
        };
        centre_of_mass[0] /= total_mass;
        centre_of_mass[1] /= total_mass;
        centre_of_mass[2] /= total_mass;
        centre_of_mass
    }
}

/// What a centre is. Centres are in input order, `Real` and `Ghost` first, interleaved, then `OrthogonalityGhost`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomKind {
    /// An actual atom: a nucleus (or core AIMP) and basis functions.
    Real = 0,
    /// Basis functions but no nucleus (`GHOST` or `PSEUDOCHARGE` in the input).
    Ghost = 1,
    /// The orthogonality functions of an embedding AIMP (`ORTHOGONALITY`), at its position: the AIMP's element, no
    /// mass, no nucleus.
    OrthogonalityGhost = 2,
}

impl AtomKind {
    /// The kinds by the names a C caller queries them under (`aedelsten_query_i32_c`).
    pub const NAME_TABLE: &'static [(&'static str, i32)] = &[("atom_kind_real", 0), ("atom_kind_ghost", 1), ("atom_kind_orthogonality_ghost", 2)];
}

/// A centre: its element, position and kind.
#[derive(Debug, Clone)]
pub struct Atom {
    element: u8,
    position: [f64; 3],
    kind: AtomKind,
}

impl Atom {
    /// A real atom.
    pub fn new(symbol: &str, position: [f64; 3]) -> Result<Self, crate::SetupError> {
        Self::new_with_kind(symbol, position, AtomKind::Real)
    }

    /// A centre of kind `kind`; an error for an unknown element symbol.
    pub fn new_with_kind(symbol: &str, position: [f64; 3], kind: AtomKind) -> Result<Self, crate::SetupError> {
        let element = constants::ELEMENT_SYMBOLS
            .iter()
            .position(|&s| s == symbol)
            .ok_or_else(|| crate::SetupError::UnkownElementSymbol { symbol: symbol.to_string() })? as u8;
        Ok(Self { element, position, kind })
    }

    /// The atomic number.
    pub fn get_atomic_number(&self) -> u8 {
        self.element
    }

    /// The position.
    pub fn get_position(&self) -> [f64; 3] {
        self.position
    }

    /// Sets the position.
    pub(crate) fn set_position(&mut self, position: [f64; 3]) {
        self.position = position;
    }

    /// The kind.
    pub fn get_kind(&self) -> AtomKind {
        self.kind
    }

    /// Whether it is a real atom.
    pub fn is_real(&self) -> bool {
        self.kind == AtomKind::Real
    }

    /// The element's standard atomic weight in u.
    pub fn get_mass(&self) -> f64 {
        constants::AVERAGE_MASSES[self.element as usize]
    }
}

// The exponent of the Gaussian nucleus of mass number `a`, 3 / (2 r_rms^2) with r_rms = (0.836 a^(1/3) + 0.570) fm, in
// bohr^-2 (L. Visscher and K. G. Dyall, At. Data Nucl. Data Tables 67, 207 (1997)).
#[inline(always)]
fn compute_visscher_dyall_nuclear_exponent(a: f64) -> f64 {
    let a3 = a.powf(1.0 / 3.0);
    let r = 0.836 * a3 + 0.570;
    let r = r * 1.0e-15;
    let r = r / constants::R_BOHR;
    let chi = 2.0 * r.powi(2);
    3.0 / chi
}

#[cfg(test)]
mod finite_nucleus_tests {
    use super::compute_visscher_dyall_nuclear_exponent;

    fn exponent_of(z: usize) -> f64 {
        compute_visscher_dyall_nuclear_exponent(constants::MASS_NUMBERS[z] as f64)
    }

    /// The exponent comes from the mass number: Ni (Z = 28, A = 58) gives 2.89964e8 bohr^-2 (OpenMolcas's, for
    /// Ni.x2c-SVPAll), Z would give another.
    #[test]
    fn the_exponent_comes_from_the_mass_number() {
        let ni = exponent_of(28);
        assert!((ni / 2.89964e8 - 1.0).abs() < 5e-6, "Ni: {ni:e}");
        assert!((compute_visscher_dyall_nuclear_exponent(28.0) / ni - 1.0).abs() > 0.4, "Z would give another nucleus");
    }

    /// Spot checks where the most abundant isotope is not near 2Z, or the element has none.
    #[test]
    fn mass_numbers_are_the_most_abundant_isotopes() {
        let expected = [(1, 1), (8, 16), (18, 40), (19, 39), (27, 59), (28, 58), (29, 63), (43, 98), (52, 130), (82, 208), (92, 238), (94, 244)];
        for (z, a) in expected {
            assert_eq!(constants::MASS_NUMBERS[z], a, "Z = {z}");
        }
    }
}
