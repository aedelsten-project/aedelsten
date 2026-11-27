// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The superposition of atomic potentials (SAP): each atom's electrons, their Coulomb and local exchange potential, as
//! fitted Gaussian charges (S. Lehtola, J. Chem. Theory Comput. 15, 1593 (2019); S. Lehtola, L. Visscher, E. Engel,
//! J. Chem. Phys. 152, 144105 (2020)). An atom's charges add up to minus its atomic number, so with its nucleus it is
//! neutral; the nucleus is not part of the fit.

use crate::charge_carriers::IndexedContractedGaussianCharge;
use crate::molecule::AtomKind;

/// The fitted atoms: four-component (GRASP, finite nuclei) or nonrelativistic (HelFEM), hydrogen to oganesson.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SapFit {
    HelFem,
    Grasp,
}

/// One element's fit: the exponents and the charges, of equal length.
struct ElementFit {
    exponents: Box<[f64]>,
    charges: Box<[f64]>,
}

/// A fit set's elements, indexed by atomic number (index 0 empty).
type FitTable = Box<[Option<ElementFit>]>;

static HELFEM: std::sync::OnceLock<FitTable> = std::sync::OnceLock::new();
static GRASP: std::sync::OnceLock<FitTable> = std::sync::OnceLock::new();

impl SapFit {
    fn table(self) -> &'static FitTable {
        match self {
            SapFit::HelFem => HELFEM.get_or_init(|| parse_fits(include_str!("../data/SAP-HELFEM"))),
            SapFit::Grasp => GRASP.get_or_init(|| parse_fits(include_str!("../data/SAP-GRASP"))),
        }
    }

    /// Element `atomic_number`'s exponents and charges, `None` if the set has no fit for it.
    pub fn element(self, atomic_number: u8) -> Option<(&'static [f64], &'static [f64])> {
        self.table().get(atomic_number as usize)?.as_ref().map(|fit| (&*fit.exponents, &*fit.charges))
    }

    /// The set's name.
    pub fn name(self) -> &'static str {
        match self {
            SapFit::HelFem => "HelFEM",
            SapFit::Grasp => "GRASP",
        }
    }
}

/// A bundled fit file: `#` comment lines, then per element its atomic number and term count, and one line per term, the
/// exponent and the charge. The files are part of the build, so a malformed one is a panic.
fn parse_fits(text: &str) -> FitTable {
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let mut table: Vec<Option<ElementFit>> = (0..constants::ELEMENT_SYMBOLS.len()).map(|_| None).collect();
    let number = |s: Option<&str>| -> f64 { s.and_then(|s| s.parse().ok()).expect("a bundled SAP fit file is malformed") };
    while let Some(header) = lines.next() {
        let mut fields = header.split_whitespace();
        let z = number(fields.next()) as usize;
        let n = number(fields.next()) as usize;
        let (mut exponents, mut charges) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for line in lines.by_ref().take(n) {
            let mut fields = line.split_whitespace();
            exponents.push(number(fields.next()));
            charges.push(number(fields.next()));
        }
        assert_eq!(exponents.len(), n, "a bundled SAP fit file is malformed");
        table[z] = Some(ElementFit { exponents: exponents.into_boxed_slice(), charges: charges.into_boxed_slice() });
    }
    table.into_boxed_slice()
}

impl<S: crate::contraction::ShellKind> crate::InputStructure<S> {
    /// The SAP electron charges of every real atom, one series per atom at its position (ghost centres and
    /// orthogonality ghosts have none). An error for a centre with a core AIMP (the fit includes the core electrons)
    /// and for an element without a fit, each naming the centre.
    pub fn derive_sap_charges(&self, fit: SapFit) -> Result<Box<[IndexedContractedGaussianCharge]>, String> {
        if let Some(core) = self.core_regions.as_ref().and_then(|regions| regions.first()) {
            let label = self.get_centre_label(core.get_idx()).map_or_else(String::new, |l| l.to_string());
            return Err(format!("the SAP guess is all-electron only: centre {label} has a core AIMP"));
        }
        let mut series = Vec::new();
        for (centre, atom) in self.qm_atoms.get_atoms().iter().enumerate().filter(|(_, a)| a.get_kind() == AtomKind::Real) {
            let (exponents, charges) = fit.element(atom.get_atomic_number()).ok_or_else(|| {
                let label = self.get_centre_label(centre).map_or_else(String::new, |l| l.to_string());
                format!("no {} SAP fit for centre {label}", fit.name())
            })?;
            series.push(IndexedContractedGaussianCharge::new(centre, atom.get_position(), exponents.into(), charges.into()));
        }
        Ok(series.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every element of both sets is there, each neutral with its nucleus: its charges add up to minus Z.
    #[test]
    fn every_element_has_charges_adding_to_minus_z() {
        for fit in [SapFit::HelFem, SapFit::Grasp] {
            for z in 1..=118u8 {
                let (exponents, charges) = fit.element(z).unwrap_or_else(|| panic!("{} has no fit for Z = {z}", fit.name()));
                assert_eq!(exponents.len(), charges.len());
                assert!(exponents.iter().all(|&e| e > 0.0));
                let total: f64 = charges.iter().sum();
                assert!((total + z as f64).abs() < 1e-8, "{} Z = {z}: charges add up to {total}", fit.name());
            }
            assert!(fit.element(0).is_none() && fit.element(119).is_none());
        }
    }

    /// The relativistic fits hold the contracted cores: Pt's tightest charge is tighter in GRASP's set than in
    /// HelFEM's, while hydrogen's are alike.
    #[test]
    fn the_four_component_cores_are_tighter() {
        let tightest = |fit: SapFit, z: u8| fit.element(z).unwrap().0.iter().cloned().fold(0.0, f64::max);
        assert!(tightest(SapFit::Grasp, 78) > 5.0 * tightest(SapFit::HelFem, 78));
        assert_eq!(tightest(SapFit::Grasp, 1), tightest(SapFit::HelFem, 1));
    }
}
