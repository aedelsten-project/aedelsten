// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The model basis of free-atom reference orbitals for a model-Fock start guess: per QM centre, the shells of a
//! reference library (ANO-RCC) with their orbital energies, as contracted shells with an energy per AO.

use arrays::matrix::{MatrixTrait, MatrixTraitMut};

use crate::basis_parser::{self, InputBasisShell};
use crate::shell_types::{BasisShell, HarmonicType};
use crate::contraction::Contracted;

/// The reference library of the model basis.
pub const ANO_RCC: &str = "ANO-RCC";

/// One QM centre's reference shells: the library's shells for its element, each trimmed to its energy-bearing columns
/// and, on a core-AIMP centre, without the columns the AIMP's core covers. Can be empty.
#[derive(Debug, Clone)]
pub struct ModelFockianCentreBasis {
    centre_index: usize,
    shells: Vec<InputBasisShell>,
    first_ordinals: Vec<usize>,
}

impl ModelFockianCentreBasis {
    /// The centre.
    pub fn get_centre_index(&self) -> usize {
        self.centre_index
    }

    /// The trimmed shells.
    pub fn get_shells(&self) -> &[InputBasisShell] {
        &self.shells
    }

    /// Per shell of `get_shells`, the ordinal of its first kept column among the reference library's
    /// energy-bearing columns at its lqn (0-based), counting the columns a core AIMP dropped: on a centre whose
    /// AIMP projection basis has `k` columns at that lqn, the first kept column of the library's first shell there
    /// is `k`.
    pub fn get_first_ordinals(&self) -> &[usize] {
        &self.first_ordinals
    }
}

// The number of a shell's `n_energies` energy-bearing columns left once the first `n_core` are dropped; `None` if none.
fn columns_to_keep(n_energies: usize, n_core: usize) -> Option<usize> {
    if n_core >= n_energies { None } else { Some(n_energies - n_core) }
}

/// The path of the reference library `basis_file` (`crate::find_library_file`).
pub fn locate_reference_library(basis_file: &str) -> Result<std::path::PathBuf, crate::SetupError> {
    crate::find_library_file(basis_file)?
        .ok_or_else(|| crate::SetupError::BasisFileNotFound { element: String::new(), basis_file: basis_file.to_string() })
}

// The entry of element `atomic_number` in the library `basis_file` at `basis_path`.
fn parse_reference_element(
    atomic_number: u8,
    basis_file: &str,
    basis_path: &std::path::Path,
) -> Result<basis_parser::ParsedBasisSet, crate::SetupError> {
    let symbol = constants::ELEMENT_SYMBOLS[atomic_number as usize];
    let basis_path_str = basis_path.to_str().unwrap();
    let spec = format!("{symbol}.{basis_file}");
    basis_parser::parse_basis_file(&spec, basis_path_str).map_err(|source| crate::SetupError::BasisFileParseError {
        element: symbol.to_string(),
        path: basis_path_str.to_string(),
        source,
    })
}

/// The leading columns of element `atomic_number`'s valence shells in the reference library `basis_file`
/// (`locate_reference_library`): at each lqn `l < n_columns.len()`, the first `n_columns[l]` columns, counted across
/// the element's shells at that lqn in library order; shells with no column left, and every shell at a higher lqn,
/// are left out. Fewer columns than asked for where the library has fewer.
pub fn read_leading_columns(
    atomic_number: u8,
    basis_file: &str,
    n_columns: &[usize],
) -> Result<Vec<InputBasisShell>, crate::SetupError> {
    let basis_path = locate_reference_library(basis_file)?;
    let parsed = parse_reference_element(atomic_number, basis_file, &basis_path)?;
    let mut remaining = n_columns.to_vec();
    let mut shells = Vec::new();
    for shell in parsed.get_valence().into_iter().flatten() {
        let Some(left) = remaining.get_mut(shell.get_lqn()) else { continue };
        let coeffs = shell.get_coefficients();
        let n_keep = coeffs.cols().min(*left);
        if n_keep == 0 {
            continue;
        }
        *left -= n_keep;
        let mut trimmed = arrays::matrix::Matrix::<f64>::new(coeffs.rows(), n_keep);
        for (dst_col, src_col) in trimmed.iter_col_mut().zip(coeffs.iter_col()) {
            dst_col.copy_from_slice(src_col);
        }
        let energies = shell.get_energies().map(|e| e[..n_keep.min(e.len())].to_vec().into_boxed_slice());
        shells.push(InputBasisShell::new(shell.get_lqn(), shell.get_exponents().to_vec().into_boxed_slice(), trimmed, energies, None));
    }
    Ok(shells)
}

/// The reference shells of every QM centre of `input` from the library `basis_file` (the current directory first, then
/// the bundled library). Orthogonality ghosts and ghost centres are left out. On a core-AIMP centre, per angular
/// momentum, the first `n` energy-bearing columns are dropped, `n` the column count of the AIMP's projection basis
/// there; the library's energies are taken to be listed core first.
pub fn read_model_fockian_basis<S: crate::contraction::ShellKind>(
    input: &crate::InputStructure<S>,
    basis_file: &str,
) -> Result<Vec<ModelFockianCentreBasis>, crate::SetupError> {
    let basis_path = locate_reference_library(basis_file)?;

    let mut result = Vec::new();
    for (centre_idx, atom) in input.get_qm_atoms().get_atoms().iter().enumerate() {
        if input.is_orthogonality_ghost_centre(centre_idx) || input.is_pseudocharge_centre(centre_idx) {
            continue;
        }

        // Per lqn, the columns the centre's core AIMP covers.
        let core_columns_per_lqn: std::collections::HashMap<usize, usize> = input
            .get_core_regions()
            .and_then(|regions| regions.iter().find(|r| r.get_idx() == centre_idx))
            .map(|aimp| {
                let mut counts = std::collections::HashMap::new();
                for shell in aimp.get_projection_basis() {
                    *counts.entry(shell.get_lqn()).or_insert(0) += shell.get_coefficients().cols();
                }
                counts
            })
            .unwrap_or_default();

        let parsed = parse_reference_element(atom.get_atomic_number(), basis_file, &basis_path)?;

        let mut shells = Vec::new();
        let mut first_ordinals = Vec::new();
        // Per lqn, the library's energy-bearing columns in the shells before this one: the base of its ordinals.
        let mut library_columns_per_lqn: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        if let Some(valence) = parsed.get_valence() {
            for shell in valence {
                let Some(energies) = shell.get_energies() else { continue };
                let earlier = library_columns_per_lqn.entry(shell.get_lqn()).or_insert(0);
                let base = *earlier;
                *earlier += energies.len();
                let n_core = core_columns_per_lqn.get(&shell.get_lqn()).copied().unwrap_or(0);
                let Some(n_keep) = columns_to_keep(energies.len(), n_core) else { continue };
                first_ordinals.push(base + n_core);
                let coeffs = shell.get_coefficients();
                let mut trimmed = arrays::matrix::Matrix::<f64>::new(coeffs.rows(), n_keep);
                for (dst_col, src_col) in trimmed.iter_col_mut().zip(coeffs.iter_col().skip(n_core)) {
                    dst_col.copy_from_slice(src_col);
                }
                shells.push(InputBasisShell::new(
                    shell.get_lqn(),
                    shell.get_exponents().to_vec().into_boxed_slice(),
                    trimmed,
                    Some(energies[n_core..].to_vec().into_boxed_slice()),
                    None,
                ));
            }
        }

        result.push(ModelFockianCentreBasis { centre_index: centre_idx, shells, first_ordinals });
    }

    Ok(result)
}

/// The model basis: contracted shells (s and p Cartesian, d and up spherical) and, per AO, its reference orbital's
/// energy and ordinal.
pub struct ModelFockianBasis {
    shells: Vec<BasisShell<Contracted>>,
    energies: Vec<f64>,
    ordinals: Vec<usize>,
    basis_info: crate::BasisInfo,
}

impl ModelFockianBasis {
    /// The shells.
    pub fn get_shells(&self) -> &[BasisShell<Contracted>] {
        &self.shells
    }

    /// Per AO, its reference orbital's energy (hartree).
    pub fn get_energies(&self) -> &[f64] {
        &self.energies
    }

    /// Per AO, its reference orbital's ordinal within its lqn in the reference library, counting the columns a
    /// core AIMP dropped (`ModelFockianCentreBasis::get_first_ordinals`), unfolded as `get_energies`.
    pub fn get_ordinals(&self) -> &[usize] {
        &self.ordinals
    }

    /// One `ModelFunction` per AO, in AO order.
    pub fn functions(&self) -> impl ExactSizeIterator<Item = ModelFunction> + '_ {
        let ids = self.basis_info.get_basis_function_ids();
        ids.iter().zip(&self.ordinals).zip(&self.energies).map(|((id, &shell_ordinal), &energy)| ModelFunction {
            centre: id.get_centre() - 1,
            l: id.get_l().unsigned_abs() as usize,
            m: id.get_m(),
            shell_ordinal,
            energy,
        })
    }

    /// The model basis's AO data.
    pub fn get_basis_info(&self) -> &crate::BasisInfo {
        &self.basis_info
    }
}

/// The model basis of `input` from the library `basis_file` (`read_model_fockian_basis`): one contracted shell per
/// reference shell, and the energies and ordinals unfolded per column, then per component, in AO order.
pub fn setup_model_fockian_basis<S: crate::contraction::ShellKind>(
    input: &crate::InputStructure<S>,
    basis_file: &str,
) -> Result<ModelFockianBasis, crate::SetupError> {
    let centres = read_model_fockian_basis(input, basis_file)?;
    let atoms = input.get_qm_atoms();

    let mut shells = Vec::new();
    let mut energies = Vec::new();
    let mut ordinals = Vec::new();
    for centre in &centres {
        let position = atoms.get_atoms()[centre.get_centre_index()].get_position();
        for (shell, &first) in centre.get_shells().iter().zip(centre.get_first_ordinals()) {
            let degeneracy = cart2sph::compute_spherical_degeneracy(shell.get_lqn());
            for (column, &energy) in shell.get_energies().expect("read_model_fockian_basis only ever keeps energy-bearing shells").iter().enumerate() {
                for _ in 0..degeneracy {
                    energies.push(energy);
                    ordinals.push(first + column);
                }
            }
            let harmonic_type = HarmonicType::default_for_lqn(shell.get_lqn());
            shells.push(BasisShell::<Contracted>::new(shell, harmonic_type, centre.get_centre_index(), position));
        }
    }

    // With the basis function ids, which give `ModelFunction`'s `m`.
    let basis_info = crate::fix_basis_indices_with_ids(&mut shells);
    debug_assert_eq!(basis_info.get_n_cont(), energies.len());

    Ok(ModelFockianBasis { shells, energies, ordinals, basis_info })
}

/// One function of the model basis: one component (spherical for d and up, Cartesian for s and p) of a free-atom
/// reference orbital of the library.
///
/// `centre` is the 0-based centre index (`get_centre_geometry`'s). `l` is the lqn. `m` is the component as
/// `BasisFunctionId::m` gives it for a shell of the same harmonic type in the working basis: `-l..=l` for d and up;
/// for s and p, which the model basis keeps Cartesian (the real p functions x, y, z), its Cartesian encoding, `-1, 0,
/// 1` for x, y, z. `shell_ordinal` is the orbital's position among the library's energy-bearing columns at `l`
/// (0-based), counting those a core AIMP covers: on a centre whose AIMP projection basis has `k` columns at `l`, the
/// first model function there has ordinal `k`. `energy` is the library's orbital energy (Hartree).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelFunction {
    centre: usize,
    l: usize,
    m: i32,
    shell_ordinal: usize,
    energy: f64,
}

impl ModelFunction {
    /// The 0-based centre.
    pub fn get_centre(&self) -> usize {
        self.centre
    }

    /// The angular momentum.
    pub fn get_l(&self) -> usize {
        self.l
    }

    /// The component, as `BasisFunctionId::m` gives it.
    pub fn get_m(&self) -> i32 {
        self.m
    }

    /// The orbital's position among the library's energy-bearing columns at `l`, counting those a core AIMP covers.
    pub fn get_shell_ordinal(&self) -> usize {
        self.shell_ordinal
    }

    /// The orbital energy (hartree).
    pub fn get_energy(&self) -> f64 {
        self.energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A real H as centre 0, a ghost H as centre 1.
    fn write_fixture() -> std::path::PathBuf {
        let path = std::env::temp_dir().join("model_fockian_test.inp");
        std::fs::write(
            &path,
            "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\n\
             Basis set\n    H.x2c-SVPall\n    PSEUDOCHARGE\n    H2 0.0 0.0 2.0\nEnd of Basis\n",
        ).unwrap();
        path
    }

    #[test]
    fn excludes_pseudocharge_centres_and_reads_energy_bearing_columns() {
        let path = write_fixture();
        let input = crate::setup_basis_system::<crate::contraction::Contracted>(path.to_str().unwrap()).unwrap();

        // From the bundled library: "ANO-RCC" is not in the test's working directory.
        let centres = read_model_fockian_basis(&input, ANO_RCC).unwrap();

        assert_eq!(centres.len(), 1);
        assert_eq!(centres[0].get_centre_index(), 0);

        let shells = centres[0].get_shells();
        // H's ANO-RCC block: one energy-bearing s column (1s, -0.5), no p, d, f energies.
        assert_eq!(shells.len(), 1);
        assert_eq!(shells[0].get_lqn(), 0);
        assert_eq!(shells[0].get_energies().unwrap().as_ref(), &[-0.5]);
        assert_eq!(shells[0].get_coefficients().cols(), 1);
    }

    // A real H as centre 0, an orthogonality ghost as centre 1.
    #[test]
    fn skips_a_trailing_orthogonality_ghost_and_keeps_the_real_atom_under_its_own_centre_index() {
        let path = std::env::temp_dir().join(format!("model_fockian_ghost_last_{}.inp", std::process::id()));
        std::fs::write(
            &path,
            "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\n\
             Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    ORTHOGONALITY\n    Na 5.0 0.0 0.0\nEnd of Basis\n",
        ).unwrap();
        let input = crate::setup_basis_system::<crate::contraction::Contracted>(path.to_str().unwrap()).unwrap();
        assert!(input.is_orthogonality_ghost_centre(1));

        let centres = read_model_fockian_basis(&input, ANO_RCC).unwrap();
        assert_eq!(centres.len(), 1, "only the real H");
        assert_eq!(centres[0].get_centre_index(), 0);
        assert_eq!(centres[0].get_shells()[0].get_energies().unwrap().as_ref(), &[-0.5]);

        let model = setup_model_fockian_basis(&input, ANO_RCC).unwrap();
        assert_eq!(model.get_shells()[0].get_position(), [0.0, 0.0, 0.0]);
    }

    // Without a core AIMP the ordinals start at 0 on every lqn and run over the energy-bearing columns: Ni in
    // ANO-RCC has 1s-4s, 2p-3p and 3d.
    #[test]
    fn ordinals_without_a_core_aimp_count_from_zero() {
        let path = std::env::temp_dir().join(format!("model_fockian_ni_{}.inp", std::process::id()));
        std::fs::write(&path, "Basis set\n    Ni.x2c-SVPall\n    Ni 0.0 0.0 0.0\nEnd of Basis\n").unwrap();
        let input = crate::setup_basis_system::<crate::contraction::Contracted>(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        let model = setup_model_fockian_basis(&input, ANO_RCC).unwrap();
        let per_l = |l: usize| -> Vec<usize> {
            let mut ordinals: Vec<usize> = model.functions().filter(|f| f.get_l() == l).map(|f| f.get_shell_ordinal()).collect();
            ordinals.dedup();
            ordinals
        };
        assert_eq!(per_l(0), [0, 1, 2, 3]);
        assert_eq!(per_l(1), [0, 1]);
        assert_eq!(per_l(2), [0]);
        assert_eq!(model.get_basis_info().get_n_cont(), 4 + 2 * 3 + 5);
    }

    #[test]
    fn columns_to_keep_boundary_cases() {
        assert_eq!(columns_to_keep(4, 0), Some(4));
        assert_eq!(columns_to_keep(4, 2), Some(2));
        assert_eq!(columns_to_keep(4, 4), None);
        assert_eq!(columns_to_keep(4, 6), None);
        assert_eq!(columns_to_keep(0, 0), None);
    }

    // Sc.TEST-CORE-AIMP, a core AIMP whose projection basis covers 1s, 2s and 2p. One file per test (the tests run in
    // parallel).
    fn write_core_aimp_fixture(test: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("model_fockian_core_aimp_{}_{}.inp", std::process::id(), test));
        std::fs::write(&path, "Basis set\n    Sc.TEST-CORE-AIMP\n    Sc 0.0 0.0 0.0\nEnd of Basis\n").unwrap();
        path
    }

    #[test]
    fn drops_core_covered_columns_for_a_real_core_aimp_centre() {
        let path = write_core_aimp_fixture("drops_core_covered_columns");
        let input = crate::setup_basis_system::<crate::contraction::Contracted>(path.to_str().unwrap()).unwrap();
        assert!(input.get_core_regions().is_some(), "Sc.TEST-CORE-AIMP should have parsed into a core region");

        let centres = read_model_fockian_basis(&input, ANO_RCC).unwrap();

        assert_eq!(centres.len(), 1);
        let shells = centres[0].get_shells();

        let s = shells.iter().find(|s| s.get_lqn() == 0).expect("s shell");
        assert_eq!(s.get_energies().unwrap().as_ref(), &[-2.5908, -0.2116]); // 3s, 4s
        assert_eq!(s.get_coefficients().cols(), 2);

        let p = shells.iter().find(|s| s.get_lqn() == 1).expect("p shell");
        assert_eq!(p.get_energies().unwrap().as_ref(), &[-1.5712]); // 3p
        assert_eq!(p.get_coefficients().cols(), 1);

        let d = shells.iter().find(|s| s.get_lqn() == 2).expect("d shell");
        assert_eq!(d.get_energies().unwrap().as_ref(), &[-0.3376]); // 3d
        assert_eq!(d.get_coefficients().cols(), 1);

        assert_eq!(shells.len(), 3);
    }

    #[test]
    fn setup_model_fockian_basis_unfolds_energies_in_lockstep_with_shells() {
        let path = write_core_aimp_fixture("unfolds_energies");
        let input = crate::setup_basis_system::<crate::contraction::Contracted>(path.to_str().unwrap()).unwrap();

        let model = setup_model_fockian_basis(&input, ANO_RCC).unwrap();

        // s 2 x 1, p 1 x 3, d 1 x 5.
        assert_eq!(model.get_energies().len(), 10);
        assert_eq!(model.get_basis_info().get_n_cont(), 10);

        assert_eq!(&model.get_energies()[0..2], &[-2.5908, -0.2116]);
        assert_eq!(&model.get_energies()[2..5], &[-1.5712, -1.5712, -1.5712]);
        assert_eq!(&model.get_energies()[5..10], &[-0.3376; 5]);

        // The library ordinals count the AIMP's core: 1s, 2s and 2p are covered, so s starts at 2, p at 1, d at 0.
        assert_eq!(model.get_ordinals(), &[2, 3, 1, 1, 1, 0, 0, 0, 0, 0]);
        let functions: Vec<ModelFunction> = model.functions().collect();
        assert_eq!(functions.len(), 10);
        assert!(functions.iter().all(|f| f.get_centre() == 0));
        assert_eq!(functions.iter().map(|f| f.get_l()).collect::<Vec<_>>(), [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]);
        // p Cartesian (x, y, z), d spherical (-2..=2), as BasisFunctionId encodes them.
        assert_eq!(functions.iter().map(|f| f.get_m()).collect::<Vec<_>>(), [0, 0, -1, 0, 1, -2, -1, 0, 1, 2]);
        assert_eq!(functions.iter().map(|f| f.get_energy()).collect::<Vec<_>>(), model.get_energies());

        assert_eq!(model.get_shells().len(), 3);
        for shell in model.get_shells() {
            let expected = if shell.get_lqn() <= 1 { HarmonicType::Cartesian } else { HarmonicType::Spherical };
            assert_eq!(shell.get_harmonic_type(), expected);
            assert_eq!(shell.get_centre_index(), 0);
        }
    }
}
