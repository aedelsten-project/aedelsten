// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Atomic mean-field spin-orbit integrals (AMFI): the two-electron part of the Breit-Pauli spin-orbit operator,
//! spin-same-orbit and spin-other-orbit, averaged over the spherical ground-state density of each atom and kept on
//! that atom (B. A. Hess, C. M. Marian, U. Wahlgren, O. Gropen, Chem. Phys. Lett. 251, 365 (1996)). The atomic
//! density is that of the occupied orbitals of a reference library (`REFERENCE_LIBRARY`) with the occupations of
//! `occupations::AtomicOccupations`; the operator is computed over the primitive basis on each atom, once per atom
//! kind (element and primitive basis).

mod atom;
mod mean_field;
pub mod occupations;

use arrays::array3d::{Array3D, Array3DTrait, Array3DTraitMut, Array3DViewMut};
use arrays::scratch::{ArenaError, Scratch, ScratchError};
use basis::contraction::{ShellKind, Uncontracted};
use basis::shell_types::BasisShell;

/// The planes of `compute_amfi`'s output: the spin-same-orbit `x, y, z`, then the spin-other-orbit `x, y, z`.
pub const N_AMFI_PLANES: usize = 6;

/// The reference library whose leading orbitals define the atomic densities.
pub const REFERENCE_LIBRARY: &str = basis::model_fockian::ANO_RCC;

/// Why the AMFI integrals could not be computed.
#[derive(Debug)]
pub enum AmfiError {
    /// No configuration for the element.
    UnsupportedElement { atomic_number: u8 },
    /// The reference library has `found` orbitals at `lqn` for the element, the configuration occupies `needed`.
    MissingReferenceOrbitals { atomic_number: u8, lqn: usize, needed: usize, found: usize },
    /// The working AOs of a centre are not contiguous in AO order.
    NonContiguousCentre { centre: usize },
    /// The shell pairs of an atom kind could not be formed.
    ShellPairs(String),
    /// The centre has a core AIMP.
    CoreAimpCentre { centre: usize },
    /// `output` is not `(n, n, N_AMFI_PLANES)` with `n` the AOs of the primitive basis.
    OutputDimensions { expected: [usize; 3], actual: [usize; 3] },
    /// Reading the reference library failed.
    Setup(basis::SetupError),
    /// The scratch is too small.
    Arena(ArenaError),
    /// A scratch buffer for measuring could not be made.
    Scratch(ScratchError),
}

impl std::fmt::Display for AmfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AmfiError::UnsupportedElement { atomic_number } => {
                write!(f, "AMFI: no ground-state configuration for atomic number {atomic_number}")
            }
            AmfiError::MissingReferenceOrbitals { atomic_number, lqn, needed, found } => write!(
                f,
                "AMFI: the reference library {REFERENCE_LIBRARY} has {found} orbitals at l = {lqn} for atomic number \
                 {atomic_number}, the ground-state configuration occupies {needed}"
            ),
            AmfiError::NonContiguousCentre { centre } => {
                write!(f, "AMFI: the working AOs of centre {centre} are not contiguous in AO order")
            }
            AmfiError::ShellPairs(message) => write!(f, "AMFI: {message}"),
            AmfiError::CoreAimpCentre { centre } => write!(f, "AMFI: centre {centre} has a core AIMP, which is not supported"),
            AmfiError::OutputDimensions { expected, actual } => {
                write!(f, "AMFI: output dimensions {actual:?}, expected {expected:?}")
            }
            AmfiError::Setup(source) => write!(f, "AMFI: {source}"),
            AmfiError::Arena(source) => write!(f, "AMFI: {source}"),
            AmfiError::Scratch(source) => write!(f, "AMFI: {source}"),
        }
    }
}

impl std::error::Error for AmfiError {}

impl From<basis::SetupError> for AmfiError {
    fn from(source: basis::SetupError) -> Self {
        AmfiError::Setup(source)
    }
}

impl From<ArenaError> for AmfiError {
    fn from(source: ArenaError) -> Self {
        AmfiError::Arena(source)
    }
}

impl From<ScratchError> for AmfiError {
    fn from(source: ScratchError) -> Self {
        AmfiError::Scratch(source)
    }
}

/// The bytes of scratch `compute_amfi` takes on `input` over `shells` (as there): the largest of its atom kinds', each
/// kind's buffers taken by its own code, no integral computed.
pub fn amfi_scratch_bytes<S: ShellKind>(input: &basis::InputStructure<S>, shells: &[BasisShell<Uncontracted>]) -> Result<usize, AmfiError> {
    let mut ws = Scratch::new(1, arrays::scratch::ByteUnit::MiB)?;
    let mut largest = 0;
    for kind in atom::atom_kinds(input, shells, REFERENCE_LIBRARY)? {
        largest = largest.max(mean_field::kind_scratch_bytes(&kind, &mut ws)?);
    }
    Ok(largest)
}

/// The AMFI operator of `input` over its primitive basis `shells` (`ShellKind::decontract`'s blocks, or the basis
/// itself when it is uncontracted, in AO order) into `output`, OVERWRITTEN: `(n, n, N_AMFI_PLANES)`, `n` the AOs of
/// `shells`, real antisymmetric matrices in the convention of the one-electron `p x Vp` spin-orbit planes (no
/// prefactor; the factor `i` and the spin assembly left to the caller). Planes 0, 1, 2: the spin-same-orbit mean field
/// `x, y, z` (Coulomb and exchange); planes 3, 4, 5: the spin-other-orbit exchange `x, y, z`
/// (`mean_field::compute_kind_mean_field`). The Breit-Pauli operator is `h = i sum_k sigma_k W_k` with
/// `W_k = (plane k + plane 3 + k) / (4c^2)`. Block diagonal by atom, each block computed once per atom kind (element
/// and primitive basis); zero on atoms without occupied orbitals (hydrogen) and on ghost centres. Scratch from
/// `scratch`.
pub fn compute_amfi<S: ShellKind>(
    input: &basis::InputStructure<S>,
    shells: &[BasisShell<Uncontracted>],
    output: &mut Array3DViewMut<'_, f64>,
    scratch: &mut Scratch,
) -> Result<(), AmfiError> {
    let n: usize = shells.iter().map(|s| s.get_degeneracy() * s.get_n_cont()).sum();
    if output.dims() != [n, n, N_AMFI_PLANES] {
        return Err(AmfiError::OutputDimensions { expected: [n, n, N_AMFI_PLANES], actual: output.dims() });
    }
    output.get_data_mut().fill(0.0);
    for kind in atom::atom_kinds(input, shells, REFERENCE_LIBRARY)? {
        let n_kind = kind.get_n_working_aos();
        let mut block = Array3D::<f64>::new(n_kind, n_kind, N_AMFI_PLANES);
        mean_field::compute_kind_mean_field(&kind, &mut block.view_mut(), scratch)?;
        for &first in kind.get_first_aos() {
            for k in 0..N_AMFI_PLANES {
                for q in 0..n_kind {
                    for p in 0..n_kind {
                        output[(first + p, first + q, k)] = block[(p, q, k)];
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use basis::contraction::Contracted;

    // OH over its primitive basis: every plane antisymmetric, zero on hydrogen (no core) and between the atoms, not
    // zero on oxygen, and isotropic (a spherical density: the three components of each part equally large).
    #[test]
    fn oxygen_mean_field_is_antisymmetric_one_centre_and_isotropic() {
        let path = std::env::temp_dir().join(format!("amfi_oh_{}.inp", std::process::id()));
        std::fs::write(&path, "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 0.3 1.8\nEnd of Basis\n").unwrap();
        let input = basis::setup_basis_system::<Contracted>(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        let (shells, _) = Contracted::decontract(input.get_basis_shells(), input.get_basis_info()).unwrap();
        let n: usize = shells.iter().map(|s| s.get_degeneracy() * s.get_n_cont()).sum();
        let n_oxygen: usize = shells.iter().filter(|s| s.get_centre_index() == 0).map(|s| s.get_degeneracy() * s.get_n_cont()).sum();
        let mut output = Array3D::<f64>::new(n, n, N_AMFI_PLANES);
        let mut scratch = Scratch::new(256, arrays::scratch::ByteUnit::MiB).unwrap();
        compute_amfi(&input, &shells, &mut output.view_mut(), &mut scratch).unwrap();

        let mut norms = [0.0; N_AMFI_PLANES];
        for (k, norm) in norms.iter_mut().enumerate() {
            for q in 0..n {
                for p in 0..n {
                    let value = output[(p, q, k)];
                    assert!((value + output[(q, p, k)]).abs() < 1e-12, "plane {k} ({p}, {q}) not antisymmetric");
                    if p >= n_oxygen || q >= n_oxygen {
                        assert_eq!(value, 0.0, "plane {k} ({p}, {q}) outside the oxygen block");
                    }
                    *norm += value * value;
                }
            }
        }
        let norms = norms.map(f64::sqrt);
        eprintln!("plane norms {norms:?}");
        for part in [0, 3] {
            assert!(norms[part] > 1e-2, "part {part} vanishes");
            for k in 1..3 {
                assert!((norms[part + k] - norms[part]).abs() < 1e-12 * norms[part], "part {part}: {norms:?}");
            }
        }
    }
}
