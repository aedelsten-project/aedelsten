// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The atom kinds of the mean field: per element and primitive working basis on the atom, the shell pairs of one
//! shell list holding the working primitives and the occupied reference orbitals, and the atoms of that kind.

use arrays::matrix::MatrixTrait;
use basis::contraction::{Contracted, ShellKind, Uncontracted};
use basis::shell_pairs::ShellPair;
use basis::shell_types::{BasisShell, HarmonicType};

use crate::occupations::{AtomicOccupations, MAX_OCCUPIED_LQN};
use crate::AmfiError;

/// One atom kind: an element with one primitive working basis. Its shell list holds the working primitive blocks (as
/// general contractions with the identity matrix, in the primitive basis's AO order), then the occupied reference
/// orbitals (the leading columns of the reference library, s and p Cartesian, d and up spherical); their AOs are
/// numbered locally from 0, working AOs first.
/// `pairs` are that list's pairs (`derive_shell_pairs`). `first_aos` holds, per atom of the kind, the AO of its first
/// working function in the primitive basis; an atom's working AOs are contiguous there.
pub(crate) struct AtomKind {
    working_shells: Vec<BasisShell<Contracted>>,
    pairs: Vec<ShellPair<Contracted>>,
    n_working_aos: usize,
    occupations: Box<[f64]>,
    first_aos: Vec<usize>,
}

impl AtomKind {
    /// The shell pairs.
    pub(crate) fn get_pairs(&self) -> &[ShellPair<Contracted>] {
        &self.pairs
    }

    /// The working AOs of one atom: local AOs `0..n_working_aos`; the density AOs follow.
    pub(crate) fn get_n_working_aos(&self) -> usize {
        self.n_working_aos
    }

    /// Per density AO (local AO minus `get_n_working_aos()`), the electrons in it.
    pub(crate) fn get_occupations(&self) -> &[f64] {
        &self.occupations
    }

    /// Per atom of the kind, the primitive-basis AO of its first working function.
    pub(crate) fn get_first_aos(&self) -> &[usize] {
        &self.first_aos
    }

    // Whether `working` (one atom's working shells, as general contractions) is this kind's working basis: the same
    // angular momenta, harmonic types, exponents and coefficients, in the same order.
    fn has_working_basis(&self, working: &[BasisShell<Contracted>]) -> bool {
        let same_shell = |a: &BasisShell<Contracted>, b: &BasisShell<Contracted>| {
            a.get_lqn() == b.get_lqn()
                && a.get_harmonic_type() == b.get_harmonic_type()
                && a.get_exponents() == b.get_exponents()
                && a.coefficient_matrix().map(|m| m.get_data()) == b.coefficient_matrix().map(|m| m.get_data())
        };
        self.working_shells.len() == working.len() && self.working_shells.iter().zip(working).all(|(a, b)| same_shell(a, b))
    }
}

/// The atom kinds of `input` over its primitive basis `shells` (`ShellKind::decontract`'s blocks, or the basis itself
/// when it is uncontracted, in AO order): every real atom (no ghost centre, no orthogonality ghost) with working
/// shells and occupied orbitals (`AtomicOccupations::for_element`), grouped by element and working basis, with the
/// occupied orbitals from the reference library `reference_library`. Errors: a core AIMP on such an atom; fewer
/// orbitals in the library at some angular momentum than the configuration occupies; an atom's working AOs not
/// contiguous.
pub(crate) fn atom_kinds<S: ShellKind>(
    input: &basis::InputStructure<S>,
    shells: &[BasisShell<Uncontracted>],
    reference_library: &str,
) -> Result<Vec<AtomKind>, AmfiError> {
    let mut kinds: Vec<(u8, AtomKind)> = Vec::new();
    for (centre, atom) in input.get_qm_atoms().get_atoms().iter().enumerate() {
        if atom.get_kind() != basis::AtomKind::Real {
            continue;
        }
        let working: Vec<BasisShell<Contracted>> =
            shells.iter().filter(|s| s.get_centre_index() == centre).map(BasisShell::<Contracted>::from_shell).collect();
        if working.is_empty() {
            continue;
        }
        let atomic_number = atom.get_atomic_number();
        let occupations = AtomicOccupations::for_element(atomic_number)?;
        if occupations.is_empty() {
            continue;
        }
        if input.get_core_regions().is_some_and(|regions| regions.iter().any(|r| r.get_idx() == centre)) {
            return Err(AmfiError::CoreAimpCentre { centre });
        }

        let first_ao = working[0].get_basis_index();
        let mut next_ao = first_ao;
        for shell in &working {
            if shell.get_basis_index() != next_ao {
                return Err(AmfiError::NonContiguousCentre { centre });
            }
            next_ao += shell.get_degeneracy() * shell.get_n_cont();
        }

        match kinds.iter_mut().find(|(z, kind)| *z == atomic_number && kind.has_working_basis(&working)) {
            Some((_, kind)) => kind.first_aos.push(first_ao),
            None => {
                let mut kind = new_kind(atomic_number, &occupations, working, atom.get_position(), centre, reference_library)?;
                kind.first_aos.push(first_ao);
                kinds.push((atomic_number, kind));
            }
        }
    }
    Ok(kinds.into_iter().map(|(_, kind)| kind).collect())
}

// The kind of element `atomic_number` with the working shells `working` of its atom at `centre`, `position`: the
// occupied orbitals of `occupations` appended from `reference_library`, the AOs renumbered locally.
fn new_kind(
    atomic_number: u8,
    occupations: &AtomicOccupations,
    mut shells: Vec<BasisShell<Contracted>>,
    position: [f64; 3],
    centre: usize,
    reference_library: &str,
) -> Result<AtomKind, AmfiError> {
    let n_occupied = occupations.n_occupied_per_lqn();
    let reference = basis::model_fockian::read_leading_columns(atomic_number, reference_library, &n_occupied)?;
    for (l, &needed) in n_occupied.iter().enumerate() {
        let found: usize = reference.iter().filter(|s| s.get_lqn() == l).map(|s| s.get_coefficients().cols()).sum();
        if found != needed {
            return Err(AmfiError::MissingReferenceOrbitals { atomic_number, lqn: l, needed, found });
        }
    }

    let working_shells = shells.clone();
    let n_working_aos = shells.iter().map(|s| s.get_degeneracy() * s.get_n_cont()).sum();
    let mut density_occupations = Vec::new();
    let mut ordinal_per_lqn = [0usize; MAX_OCCUPIED_LQN + 1];
    for shell in &reference {
        let l = shell.get_lqn();
        let density_shell = BasisShell::<Contracted>::new(shell, HarmonicType::default_for_lqn(l), centre, position);
        let degeneracy = density_shell.get_degeneracy();
        for _ in 0..shell.get_coefficients().cols() {
            let occupation = occupations.get_occupation(l, ordinal_per_lqn[l]);
            density_occupations.extend(std::iter::repeat_n(occupation, degeneracy));
            ordinal_per_lqn[l] += 1;
        }
        shells.push(density_shell);
    }
    basis::fix_basis_indices(&mut shells);
    let pairs = basis::shell_pairs::derive_shell_pairs(&shells, None).map_err(AmfiError::ShellPairs)?;

    Ok(AtomKind {
        working_shells,
        pairs,
        n_working_aos,
        occupations: density_occupations.into_boxed_slice(),
        first_aos: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use basis::contraction::Segmented;

    fn input_of<S: ShellKind>(name: &str, text: &str) -> basis::InputStructure<S> {
        let path = std::env::temp_dir().join(format!("amfi_atoms_{name}_{}.inp", std::process::id()));
        std::fs::write(&path, text).unwrap();
        let input = basis::setup_basis_system::<S>(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        input
    }

    const NICKEL_HYDRIDE: &str = "Basis set\n    Ni.x2c-SVPall\n    Ni1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 0.0 2.8\nEnd of Basis\n";

    // Every element of the reference library has the orbitals its configuration occupies.
    #[test]
    fn the_reference_library_has_every_occupied_orbital() {
        for z in 2..=96u8 {
            let occupations = AtomicOccupations::for_element(z).unwrap();
            let n_occupied = occupations.n_occupied_per_lqn();
            let shells = basis::model_fockian::read_leading_columns(z, crate::REFERENCE_LIBRARY, &n_occupied).unwrap();
            for (l, &needed) in n_occupied.iter().enumerate() {
                let found: usize = shells.iter().filter(|s| s.get_lqn() == l).map(|s| s.get_coefficients().cols()).sum();
                assert_eq!(found, needed, "Z = {z}, l = {l}");
            }
        }
    }

    // The primitive blocks of `input`'s basis.
    fn primitive_blocks<S: ShellKind>(input: &basis::InputStructure<S>) -> Box<[BasisShell<Uncontracted>]> {
        S::decontract(input.get_basis_shells(), input.get_basis_info()).unwrap().0
    }

    // Ni (ANO-RCC 6s5p3d) and H: one kind, Ni, over Ni's primitives, holding its 28 electrons.
    #[test]
    fn nickel_hydride_has_one_kind_over_the_nickel_primitives() {
        let input = input_of::<Contracted>("nih", NICKEL_HYDRIDE);
        let blocks = primitive_blocks(&input);
        let kinds = atom_kinds(&input, &blocks, crate::REFERENCE_LIBRARY).unwrap();
        assert_eq!(kinds.len(), 1);
        let nickel = &kinds[0];

        let nickel_blocks: Vec<_> = blocks.iter().filter(|b| b.get_centre_index() == 0).collect();
        let n_prim_aos: usize = nickel_blocks.iter().map(|b| b.get_degeneracy() * b.get_n_prim()).sum();
        assert_eq!(nickel.get_first_aos(), [0]);
        assert_eq!(nickel.get_n_working_aos(), n_prim_aos);
        // 1s-4s, 2p-3p, 3d: 4 + 2 * 3 + 5 AOs.
        assert_eq!(nickel.get_occupations().len(), 15);
        let electrons: f64 = nickel.get_occupations().iter().sum();
        assert!((electrons - 28.0).abs() < 1e-12);
        // 4 working blocks (s, p, d, f) and 3 density shells: every unordered pair, the self pairs as triangles.
        assert_eq!(nickel.get_pairs().len(), 7 * 8 / 2);
        assert_eq!(nickel.get_pairs().iter().filter(|p| p.get_single_centre()).count(), 7);
    }

    // O atoms whose bases share their primitives (two ANO-RCC truncations: a general set's case, so ANO-RCC here) are
    // one kind; an O atom with other primitives (x2c-SVPall) is another; C is a third.
    #[test]
    fn atoms_are_grouped_by_element_and_primitive_basis() {
        let text = "Basis set\n    O.ANO-RCC...3s2p1d\n    O1 0.0 0.0 0.0\n    O2 0.0 0.0 2.4\nEnd of Basis\n\
                    Basis set\n    C.x2c-SVPall\n    C1 0.0 2.2 1.2\nEnd of Basis\n\
                    Basis set\n    O.ANO-RCC...4s3p2d\n    O3 0.0 4.4 1.2\nEnd of Basis\n\
                    Basis set\n    O.x2c-SVPall\n    O4 0.0 -2.2 1.2\nEnd of Basis\n";
        let input = input_of::<Contracted>("grouped", text);
        let blocks = primitive_blocks(&input);
        let kinds = atom_kinds(&input, &blocks, crate::REFERENCE_LIBRARY).unwrap();
        let first_aos: Vec<Vec<usize>> = kinds.iter().map(|k| k.get_first_aos().to_vec()).collect();
        let atom_start = |centre: usize| blocks.iter().find(|b| b.get_centre_index() == centre).unwrap().get_basis_index();
        assert_eq!(first_aos, [vec![atom_start(0), atom_start(1), atom_start(3)], vec![atom_start(2)], vec![atom_start(4)]]);
        // x2c-SVPall O: 7 s, 4 p and 1 d primitives.
        assert_eq!(kinds[2].get_n_working_aos(), 7 + 4 * 3 + 5);
    }

    // A segmented basis's primitive blocks and an uncontracted basis give the same O kind: x2c-SVPall O, 7 s, 4 p and
    // 1 d primitives.
    #[test]
    fn segmented_and_uncontracted_bases_give_the_same_primitives() {
        let text = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 1.1\n    H2 0.0 -1.4 1.1\nEnd of Basis\n";
        let segmented = input_of::<Segmented>("svp_segmented", text);
        let kinds = atom_kinds(&segmented, &primitive_blocks(&segmented), crate::REFERENCE_LIBRARY).unwrap();
        assert_eq!(kinds.len(), 1);
        assert_eq!(kinds[0].get_n_working_aos(), 7 + 4 * 3 + 5);

        let uncontracted = input_of::<Uncontracted>("svp_uncontracted", text);
        let kinds = atom_kinds(&uncontracted, uncontracted.get_basis_shells(), crate::REFERENCE_LIBRARY).unwrap();
        assert_eq!(kinds.len(), 1);
        assert_eq!(kinds[0].get_n_working_aos(), 7 + 4 * 3 + 5);
    }

    // `<converted | original>` equals `<original | original>`: the general contraction of `from_shell` is the same
    // functions as the shells it is made from.
    fn assert_conversion_keeps_the_functions<S: ShellKind>(input: &basis::InputStructure<S>) {
        use oneint::OneElectronProjectionIntegrals as _;
        let original = input.get_basis_shells();
        let converted: Vec<BasisShell<Contracted>> = original.iter().map(BasisShell::<Contracted>::from_shell).collect();
        let n = input.get_n_cont();
        let mut scratch = arrays::scratch::Scratch::new(16, arrays::scratch::ByteUnit::MiB).unwrap();
        let mut reference = arrays::matrix::Matrix::<f64>::new(n, n);
        let mut mixed = arrays::matrix::Matrix::<f64>::new(n, n);
        oneint::OneIntProjectionWorkStruct::new(original, original, &mut scratch)
            .compute_overlap_integrals(&mut arrays::matrix::MatrixTraitMut::view_mut(&mut reference))
            .unwrap();
        oneint::OneIntProjectionWorkStruct::new(&converted, original, &mut scratch)
            .compute_overlap_integrals(&mut arrays::matrix::MatrixTraitMut::view_mut(&mut mixed))
            .unwrap();
        for (r, m) in reference.get_data().iter().zip(mixed.get_data()) {
            assert!((r - m).abs() < 1e-12, "{r} vs {m}");
        }
    }

    #[test]
    fn conversion_to_a_general_contraction_keeps_the_functions() {
        let water = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 1.1\n    H2 0.0 -1.4 1.1\nEnd of Basis\n";
        assert_conversion_keeps_the_functions(&input_of::<Segmented>("convert_segmented", water));
        assert_conversion_keeps_the_functions(&input_of::<Uncontracted>("convert_uncontracted", water));
        assert_conversion_keeps_the_functions(&input_of::<Contracted>("convert_general", NICKEL_HYDRIDE));
    }

    #[test]
    fn a_core_aimp_centre_is_an_error() {
        let input = input_of::<Contracted>(
            "core_aimp",
            "Basis set\n    Sc.TEST-CORE-AIMP\n    Sc1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 0.0 3.4\nEnd of Basis\n",
        );
        let result = atom_kinds(&input, &primitive_blocks(&input), crate::REFERENCE_LIBRARY);
        assert!(matches!(result, Err(AmfiError::CoreAimpCentre { centre: 0 })));
    }
}
