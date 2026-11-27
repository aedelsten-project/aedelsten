// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The input system: the input file and the basis libraries read into centres, basis shells, nuclei, core and
//! embedding AIMPs, and the AO bookkeeping (`BasisInfo`).

#[cfg(test)]
mod amplitudes;
pub mod ao_values;
pub mod basis_parser;
pub mod contraction;
pub mod gto_products;
pub mod model_fockian;
pub mod probes;
pub mod sap;
mod molecule;
pub mod shell_pairs;
pub mod shell_types;
mod xfield;

/// The classical sources (`sources`) and the xfield readers.
pub mod charge_carriers {
    pub use crate::xfield::{read_xfield, read_xfield_polarizable_sites};
    pub use sources::*;
}

pub use molecule::AtomKind;

use std::rc::Rc;
use crate::basis_parser::{InputBasisShell, InputFileParseError};
use crate::charge_carriers::{IndexedPointCharge, PointCharge, PointChargeTrait, SiteTrait};

// The bundled basis and AIMP libraries, `src/basis/data/` of the source tree.
const DEFAULT_BASIS_LIBRARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/");

/// The variable naming further library directories (separated as `PATH` is), searched after the current directory and
/// before the bundled library: a host's own basis and AIMP files, the tests' fixtures.
pub const BASIS_LIBRARY_VARIABLE: &str = "AEDELSTEN_BASIS_LIBRARY";

/// The library file `basis_file` (its name matched case-insensitively): in the current directory, then in the
/// directories of `AEDELSTEN_BASIS_LIBRARY`, then in the bundled library: its installed copy
/// (`<prefix>/share/aedelsten/basis/`, `cargo install-aedelsten install`), then the source tree's.
pub fn find_library_file(basis_file: &str) -> Result<Option<std::path::PathBuf>, SetupError> {
    let current = std::env::current_dir().map_err(SetupError::CurrentDirError)?;
    let named: Vec<std::path::PathBuf> = std::env::var_os(BASIS_LIBRARY_VARIABLE).map(|v| std::env::split_paths(&v).collect()).unwrap_or_default();
    Ok(std::iter::once(current)
        .chain(named)
        .chain(installed_basis_library())
        .chain(std::iter::once(std::path::PathBuf::from(DEFAULT_BASIS_LIBRARY)))
        .find_map(|dir| find_basis_file_case_insensitive(&dir, basis_file)))
}

// The installed copy of the bundled library, `<prefix>/share/aedelsten/basis/`, where `<prefix>/lib/` holds the loaded
// library (or the executable it is linked into).
fn installed_basis_library() -> Option<std::path::PathBuf> {
    let object = loaded_object_path()?;
    Some(object.parent()?.parent()?.join("share").join("aedelsten").join("basis"))
}

// The file of the shared library or executable holding this code, symbolic links resolved.
#[cfg(unix)]
fn loaded_object_path() -> Option<std::path::PathBuf> {
    use std::os::raw::{c_char, c_int, c_void};
    use std::os::unix::ffi::OsStrExt;
    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }
    unsafe extern "C" {
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }
    let mut info = DlInfo { dli_fname: std::ptr::null(), dli_fbase: std::ptr::null_mut(), dli_sname: std::ptr::null(), dli_saddr: std::ptr::null_mut() };
    let address = loaded_object_path as fn() -> Option<std::path::PathBuf> as *const c_void;
    // SAFETY: `dladdr` reads the address and fills `info`; the name it gives is a NUL-terminated string owned by the
    // dynamic loader, valid while the object stays loaded, and copied below.
    let name = unsafe {
        if dladdr(address, &mut info) == 0 || info.dli_fname.is_null() {
            return None;
        }
        std::ffi::CStr::from_ptr(info.dli_fname)
    };
    let path = std::path::PathBuf::from(std::ffi::OsStr::from_bytes(name.to_bytes()));
    Some(std::fs::canonicalize(&path).unwrap_or(path))
}

#[cfg(not(unix))]
fn loaded_object_path() -> Option<std::path::PathBuf> {
    None
}

// The file `basis_file` in `dir`, its name matched case-insensitively (the exact case first).
fn find_basis_file_case_insensitive(dir: &std::path::Path, basis_file: &str) -> Option<std::path::PathBuf> {
    let exact = dir.join(basis_file);
    if exact.exists() {
        return Some(exact);
    }
    std::fs::read_dir(dir).ok()?.filter_map(Result::ok).find_map(|entry| {
        let name = entry.file_name();
        (name.to_str()?.eq_ignore_ascii_case(basis_file)).then(|| entry.path())
    })
}

/// An AIMP: a screened charge with a projection basis. Implemented by core (`IndexedAimp`) and embedding (`Aimp`)
/// AIMPs.
pub trait AimpTrait: sources::ScreenedChargeTrait {
    /// The projection (PROJOP) shells.
    fn get_projection_basis(&self) -> &[crate::basis_parser::InputBasisShell];
    /// The address of the projection shells, shared by every AIMP of one basis block: the AIMP's kind.
    fn get_projection_basis_identity(&self) -> *const Vec<crate::basis_parser::InputBasisShell>;
    /// The centre of a core AIMP, `usize::MAX` for an embedding AIMP; for labelling shells only.
    fn get_centre_index(&self) -> usize;

    /// The primitives the exchange operator is resolved in (S. Huzinaga, L. Seijo, Z. Barandiarán and M. Klobukowski,
    /// J. Chem. Phys. 86, 2132 (1987), eqs. 7-10): for a core AIMP the valence and the projection primitives, the
    /// valence-only ones with zero projection coefficients; for an embedding AIMP the projection primitives.
    fn get_resolution_basis(&self) -> Vec<crate::basis_parser::InputBasisShell>;
}

/// A core AIMP: the core potential of a QM atom, on its centre, with the centre's valence shells.
#[derive(Debug, Clone)]
pub struct IndexedAimp {
    point_charge: IndexedPointCharge,
    screen: Rc<crate::charge_carriers::ScreenedCoulombPotential>,
    projection_shells: Rc<Vec<crate::basis_parser::InputBasisShell>>,
    // The centre's valence shells, for get_resolution_basis.
    valence_shells: Rc<Vec<crate::basis_parser::InputBasisShell>>,
}

impl IndexedAimp {
    /// The centre.
    pub fn get_idx(&self) -> usize {
        self.point_charge.get_idx()
    }

    /// Moves the core AIMP with its centre (geometry updates).
    pub(crate) fn set_position(&mut self, position: [f64; 3]) {
        self.point_charge.set_position(position);
    }

    /// The position.
    pub fn get_position(&self) -> &[f64; 3] {
        &self.point_charge.get_position()
    }

    /// The screening of its charge.
    pub fn get_screened_coulomb_potential(&self) -> &crate::charge_carriers::ScreenedCoulombPotential {
        &self.screen
    }

    /// The projection (PROJOP) shells.
    pub fn get_projection_basis(&self) -> &[crate::basis_parser::InputBasisShell] {
        &self.projection_shells
    }

    /// Its charge on its centre.
    pub fn get_point_charge(&self) -> &IndexedPointCharge {
        &self.point_charge
    }

    /// The address of the projection shells, shared by every AIMP of one basis block: the AIMP's kind.
    pub fn get_projection_basis_identity(&self) -> *const Vec<crate::basis_parser::InputBasisShell> {
        Rc::as_ptr(&self.projection_shells)
    }
}

impl SiteTrait for IndexedAimp {
    fn get_position(&self) -> &[f64; 3] {
        self.point_charge.get_position()
    }
}

impl PointChargeTrait for IndexedAimp {
    fn get_charge(&self) -> f64 {
        self.point_charge.get_charge()
    }
}

impl sources::ScreenedChargeTrait for IndexedAimp {
    fn get_screened_coulomb_potential(&self) -> &crate::charge_carriers::ScreenedCoulombPotential {
        &self.screen
    }
}

impl AimpTrait for IndexedAimp {
    fn get_projection_basis(&self) -> &[crate::basis_parser::InputBasisShell] {
        &self.projection_shells
    }

    fn get_projection_basis_identity(&self) -> *const Vec<crate::basis_parser::InputBasisShell> {
        Rc::as_ptr(&self.projection_shells)
    }

    fn get_centre_index(&self) -> usize {
        self.point_charge.get_idx()
    }

    fn get_resolution_basis(&self) -> Vec<crate::basis_parser::InputBasisShell> {
        crate::basis_parser::merge_resolution_shells(&self.projection_shells, &self.valence_shells)
    }
}

/// An embedding AIMP: a fixed potential of the environment, with no centre of its own.
#[derive(Debug, Clone)]
pub struct Aimp {
    point_charge: PointCharge,
    screen: Rc<crate::charge_carriers::ScreenedCoulombPotential>,
    projection_shells: Rc<Vec<crate::basis_parser::InputBasisShell>>,
    // The atomic number of its block's element; 0 when the block's name is not an element symbol.
    atomic_number: u8,
}

impl Aimp {
    /// The position.
    pub fn get_position(&self) -> &[f64; 3] {
        self.point_charge.get_position()
    }

    /// The atomic number of its block's element; 0 when the block's name is not an element symbol.
    pub fn get_atomic_number(&self) -> u8 {
        self.atomic_number
    }

    /// Its charge.
    pub fn get_point_charge(&self) -> &PointCharge {
        &self.point_charge
    }
}

impl SiteTrait for Aimp {
    fn get_position(&self) -> &[f64; 3] {
        self.point_charge.get_position()
    }
}

impl PointChargeTrait for Aimp {
    fn get_charge(&self) -> f64 {
        self.point_charge.get_charge()
    }
}

impl sources::ScreenedChargeTrait for Aimp {
    fn get_screened_coulomb_potential(&self) -> &crate::charge_carriers::ScreenedCoulombPotential {
        &self.screen
    }
}

impl AimpTrait for Aimp {
    fn get_projection_basis(&self) -> &[crate::basis_parser::InputBasisShell] {
        &self.projection_shells
    }

    fn get_projection_basis_identity(&self) -> *const Vec<crate::basis_parser::InputBasisShell> {
        Rc::as_ptr(&self.projection_shells)
    }

    fn get_centre_index(&self) -> usize {
        usize::MAX
    }

    fn get_resolution_basis(&self) -> Vec<crate::basis_parser::InputBasisShell> {
        self.projection_shells.as_ref().clone()
    }
}

/// One AO's MOLCAS `BASIS_FUNCTION_IDS` row, `(centre, shell, l, m)`, centre and shell 1-based. `l` is the shell's
/// angular momentum, negated for a Cartesian shell. `m` is `-l..=l` for a spherical shell, and `T(ly+lz) - (lx+ly)`,
/// `T(n) = n(n+1)/2`, for the Cartesian component `(lx, ly, lz)`. `shell` is the AO's ordinal among the contracted AOs
/// of its (centre, l).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasisFunctionId {
    centre: usize,
    shell: usize,
    l: i32,
    m: i32,
}

impl BasisFunctionId {
    /// The 1-based centre.
    pub fn get_centre(&self) -> usize {
        self.centre
    }
    
    /// The 1-based ordinal among the contracted AOs of the (centre, l).
    pub fn get_shell(&self) -> usize {
        self.shell
    }

    /// The angular momentum, negative for a Cartesian shell.
    pub fn get_l(&self) -> i32 {
        self.l
    }

    /// The component.
    pub fn get_m(&self) -> i32 {
        self.m
    }
}

impl Default for BasisFunctionId {
    fn default() -> Self {
        BasisFunctionId { centre: 0, shell: 0, l: 0, m: 0 }
    }
}

/// One row of MOLCAS `PRIMITIVE_IDS`, `(centre, l, shell)`: centre and shell 1-based and numbered as in
/// `BasisFunctionId`, `l` never negated.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveId {
    /// The centre, 1-based.
    pub centre: usize,
    /// The angular momentum.
    pub l: usize,
    /// The shell, 1-based.
    pub shell: usize,
}

/// One row of MOLCAS `PRIMITIVES`, parallel to `BasisInfo::get_primitive_ids`: an exponent and its contraction
/// coefficient on the independently normalized primitive.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Primitive {
    /// The exponent.
    pub exponent: f64,
    /// The contraction coefficient.
    pub coefficient: f64,
}

/// A centre's label, displayed as "{element_symbol}{count}" (e.g. "H1", "O2").
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CentreLabel {
    /// The centre (0-based).
    pub centre: usize,
    /// The atomic number; 0 for a ghost centre.
    pub atomic_number: u8,
    /// Its number among the centres of its element, from 1, in input order.
    pub count: usize,
}

impl CentreLabel {
    /// The element's symbol; `"X"` for atomic number 0 (a ghost centre).
    pub fn element_symbol(&self) -> &'static str {
        if self.atomic_number == 0 { "X" } else { constants::ELEMENT_SYMBOLS[self.atomic_number as usize] }
    }
}

impl std::fmt::Display for CentreLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.element_symbol(), self.count)
    }
}

/// One centre's atomic number, position (bohr) and kind, for export (`CENTER_ATNUMS`, `CENTER_COORDINATES` of a
/// MOLCAS HDF5 file); one per centre index, ghost centres and orthogonality ghosts included. An orthogonality ghost has
/// its parent AIMP's element.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CentreGeometry {
    /// The atomic number.
    pub atomic_number: u8,
    /// The position, bohr.
    pub position: [f64; 3],
    /// Real, ghost or orthogonality ghost.
    pub kind: crate::molecule::AtomKind,
}

/// An embedding AIMP's site: the atomic number of its block's element (0 when the block's name is not an element
/// symbol), its position (bohr) and its charge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingAimpSite {
    /// The atomic number of the block's element.
    pub atomic_number: u8,
    /// The position, bohr.
    pub position: [f64; 3],
    /// The charge.
    pub charge: f64,
}

/// The AO counts and bounds of a basis, and, for the user-facing basis, the export identities.
#[derive(Debug)]
pub struct BasisInfo {
    n_cont: usize,
    // The sum of the shells' primitive counts (times components): an exponent shared by sibling segments counts
    // once per sibling.
    n_prim: usize,
    // The distinct (centre, l, exponent) primitives (times components): the primitive basis.
    n_prim_unique: usize,
    // One per AO, in AO order.
    basis_function_ids: Box<[BasisFunctionId]>,
    // One per (shell, contracted function, contributing primitive).
    primitive_ids: Box<[PrimitiveId]>,
    primitives: Box<[Primitive]>,
}

impl BasisInfo {
    /// The number of AOs.
    pub fn get_n_cont(&self) -> usize {
        self.n_cont
    }

    /// The sum of the shells' primitive AOs; a primitive shared by sibling segments counts once per sibling.
    pub fn get_n_prim(&self) -> usize {
        self.n_prim
    }

    /// The number of distinct primitive AOs: the dimension of the primitive basis.
    pub fn get_n_prim_unique(&self) -> usize {
        self.n_prim_unique
    }

    /// One per AO; empty unless built by `fix_basis_indices_with_ids` (the user-facing basis).
    pub fn get_basis_function_ids(&self) -> &[BasisFunctionId] {
        &self.basis_function_ids
    }

    /// One per contributing primitive; empty unless built by `fix_basis_indices_with_ids`.
    pub fn get_primitive_ids(&self) -> &[PrimitiveId] {
        &self.primitive_ids
    }

    /// Parallel to `get_primitive_ids`.
    pub fn get_primitives(&self) -> &[Primitive] {
        &self.primitives
    }

    /// The bytes it holds.
    pub fn approx_memory_bytes(&self) -> usize {
        std::mem::size_of::<BasisInfo>()
            + std::mem::size_of_val(self.basis_function_ids.as_ref())
            + std::mem::size_of_val(self.primitive_ids.as_ref())
            + std::mem::size_of_val(self.primitives.as_ref())
    }
}

/// The system read from an input file: the centres, the basis shells, the nuclei, the core and the embedding AIMPs.
#[derive(Debug)]
pub struct InputStructure<S: crate::contraction::ShellKind> {
    qm_atoms: crate::molecule::Molecule,
    basis_shells: Vec<shell_types::BasisShell<S>>,
    basis_info: BasisInfo,
    nuclear_potentials: Option<crate::charge_carriers::NuclearPotential>, // None when every centre has a core AIMP
    core_regions: Option<Vec<IndexedAimp>>,
    embedding_potentials: Option<Vec<Aimp>>,
    ao_evaluator: ao_values::AoEvaluator,
}

impl<S: crate::contraction::ShellKind> InputStructure<S> {
    /// The bytes it holds: shallow for the shells (their data is shared) and the potentials, exact for `BasisInfo`.
    pub fn approx_memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + std::mem::size_of_val(self.basis_shells.as_slice()) + self.basis_info.approx_memory_bytes()
    }

    /// Every centre.
    pub fn get_qm_atoms(&self) -> &crate::molecule::Molecule {
        &self.qm_atoms
    }

    /// The basis shells.
    pub fn get_basis_shells(&self) -> &[shell_types::BasisShell<S>] {
        &self.basis_shells
    }

    /// The number of AOs.
    pub fn get_n_cont(&self) -> usize {
        self.basis_info.n_cont
    }

    /// `BasisInfo::get_n_prim`.
    pub fn get_n_prim(&self) -> usize {
        self.basis_info.n_prim
    }

    /// `BasisInfo::get_n_prim_unique`.
    pub fn get_n_prim_unique(&self) -> usize {
        self.basis_info.n_prim_unique
    }

    /// The nuclei without a core AIMP, point or Gaussian.
    pub fn get_nuclear_potential(&self) -> Option<&crate::charge_carriers::NuclearPotential> {
        self.nuclear_potentials.as_ref()
    }

    /// Makes point nuclei Gaussian under `model`; nothing if they are already Gaussian.
    pub fn set_finite_nuclear_model(&mut self, model: crate::charge_carriers::FiniteModel) {
        let point_charges = self.nuclear_potentials.take();
        match point_charges {
            Some(crate::charge_carriers::NuclearPotential::PointCharge(point_charges)) => {
                let new_nuclear_potential = self.qm_atoms.derive_finite_nuclei(point_charges, model);
                self.nuclear_potentials = Some(crate::charge_carriers::NuclearPotential::GaussianCharge(new_nuclear_potential));
            },
            Some(crate::charge_carriers::NuclearPotential::GaussianCharge(gaussian_charges)) => {
                self.nuclear_potentials = Some(crate::charge_carriers::NuclearPotential::GaussianCharge(gaussian_charges));
            },
            None => {
                self.nuclear_potentials = None;
            }
        }
    }

    /// The basis's AO data.
    pub fn get_basis_info(&self) -> &BasisInfo {
        &self.basis_info
    }

    /// The values (`(n_listed, n_points)`) and optionally the gradients (three such blocks, x, y, z) of the AOs of
    /// `shells` at `points` (`(point_rows, n_points)`), see `AoEvaluator::evaluate`.
    pub fn compute_ao_values(
        &self,
        shells: impl Iterator<Item = usize> + Clone,
        points: &[f64],
        point_rows: usize,
        values: &mut [f64],
        gradients: Option<&mut [f64]>,
    ) -> Result<(), ao_values::AoValuesError> {
        let gradients = match gradients {
            Some(gradients) => {
                let block = gradients.len() / 3;
                if gradients.len() != values.len() * 3 {
                    return Err(ao_values::AoValuesError::OutputLength { expected: 3 * values.len(), actual: gradients.len() });
                }
                let (x, rest) = gradients.split_at_mut(block);
                let (y, z) = rest.split_at_mut(block);
                Some([x, y, z])
            }
            None => None,
        };
        self.ao_evaluator.evaluate(&self.basis_shells, shells, points, point_rows, Some(values), gradients, None)
    }

    /// The values (`(n_listed, n_points)`), the gradients (three such blocks, x, y, z) and the second derivatives (six
    /// such blocks, xx, xy, xz, yy, yz, zz) of the AOs of `shells` at `points` (`(point_rows, n_points)`), see
    /// `AoEvaluator::evaluate`.
    pub fn compute_ao_hessians(
        &self,
        shells: impl Iterator<Item = usize> + Clone,
        points: &[f64],
        point_rows: usize,
        values: &mut [f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) -> Result<(), ao_values::AoValuesError> {
        if gradients.len() != 3 * values.len() {
            return Err(ao_values::AoValuesError::OutputLength { expected: 3 * values.len(), actual: gradients.len() });
        }
        if hessians.len() != 6 * values.len() {
            return Err(ao_values::AoValuesError::OutputLength { expected: 6 * values.len(), actual: hessians.len() });
        }
        let block = values.len();
        let mut g = gradients.chunks_exact_mut(block.max(1));
        let gradients: [&mut [f64]; 3] = std::array::from_fn(|_| g.next().unwrap_or_default());
        let mut h = hessians.chunks_exact_mut(block.max(1));
        let hessians: [&mut [f64]; 6] = std::array::from_fn(|_| h.next().unwrap_or_default());
        self.ao_evaluator.evaluate(&self.basis_shells, shells, points, point_rows, Some(values), Some(gradients), Some(hessians))
    }

    /// The AO evaluation data of the shells.
    pub fn get_ao_evaluator(&self) -> &ao_values::AoEvaluator {
        &self.ao_evaluator
    }

    /// The embedding AIMPs.
    pub fn get_embedding_potentials(&self) -> Option<&Vec<Aimp>> {
        self.embedding_potentials.as_ref()
    }

    /// The core AIMPs.
    pub fn get_core_regions(&self) -> Option<&Vec<IndexedAimp>> {
        self.core_regions.as_ref()
    }

    /// The number of centres gradients are taken with respect to: every centre but the orthogonality ghosts, which come
    /// last, so this is also the index of the first of them.
    pub fn get_n_gradient_centres(&self) -> usize {
        let atoms = self.qm_atoms.get_atoms();
        let n = atoms.iter().filter(|a| a.get_kind() != crate::molecule::AtomKind::OrthogonalityGhost).count();
        debug_assert!(atoms[n..].iter().all(|a| a.get_kind() == crate::molecule::AtomKind::OrthogonalityGhost));
        n
    }

    /// Moves every gradient centre (real atoms and ghosts; `positions[X]` for centre X, one per
    /// `get_n_gradient_centres()`) together with everything placed on it: its atom, its basis shells, its nucleus
    /// (point or Gaussian) and its core AIMP. Orthogonality ghosts and embedding AIMPs are the fixed environment
    /// and do not move. Nothing derived from the geometry is updated here: the caller invalidates it.
    pub fn set_centre_positions(&mut self, positions: &[[f64; 3]]) -> Result<(), String>
    where
    {
        let n_centres = self.get_n_gradient_centres();
        if positions.len() != n_centres {
            return Err(format!("expected {n_centres} centre positions (the gradient centres), got {}", positions.len()));
        }
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err("centre positions must be finite".to_string());
        }
        for (atom, &position) in self.qm_atoms.get_atoms_mut().iter_mut().zip(positions) {
            atom.set_position(position);
        }
        for shell in self.basis_shells.iter_mut() {
            let centre = shell.get_centre_index();
            if centre < n_centres {
                shell.set_position(positions[centre]);
            }
        }
        match &mut self.nuclear_potentials {
            Some(sources::NuclearPotential::PointCharge(charges)) => {
                for charge in charges.iter_mut().filter(|c| c.get_idx() < n_centres) {
                    charge.set_position(positions[charge.get_idx()]);
                }
            }
            Some(sources::NuclearPotential::GaussianCharge(charges)) => {
                for charge in charges.iter_mut().filter(|c| c.get_idx() < n_centres) {
                    charge.set_position(positions[charge.get_idx()]);
                }
            }
            None => {}
        }
        if let Some(core_aimps) = &mut self.core_regions {
            for aimp in core_aimps.iter_mut().filter(|a| a.get_idx() < n_centres) {
                aimp.set_position(positions[aimp.get_idx()]);
            }
        }
        Ok(())
    }

    /// Whether centre `centre_idx` is an orthogonality ghost.
    pub fn is_orthogonality_ghost_centre(&self, centre_idx: usize) -> bool {
        self.qm_atoms.get_atoms().get(centre_idx).is_some_and(|a| a.get_kind() == crate::molecule::AtomKind::OrthogonalityGhost)
    }

    /// Whether centre `centre_idx` is a ghost centre (`GHOST` or `PSEUDOCHARGE` in the input).
    pub fn is_pseudocharge_centre(&self, centre_idx: usize) -> bool {
        self.qm_atoms.get_atoms().get(centre_idx).is_some_and(|a| a.get_kind() == crate::molecule::AtomKind::Ghost)
    }

    /// Centre `centre_idx`'s label (`BasisFunctionId::centre - 1` is this index), `None` out of range: `{symbol}{n}`,
    /// `n` 1-based per symbol in input order. A real atom by its element, a ghost centre "X"; an orthogonality ghost by
    /// its parent AIMP's element, counted with the real atoms of that element.
    pub fn get_centre_label(&self, centre_idx: usize) -> Option<CentreLabel> {
        use crate::molecule::AtomKind;
        let atoms = self.qm_atoms.get_atoms();
        let label_element = |a: &crate::molecule::Atom| if a.get_kind() == AtomKind::Ghost { 0 } else { a.get_atomic_number() };
        let atomic_number = label_element(atoms.get(centre_idx)?);
        let count = atoms[..=centre_idx].iter().filter(|a| label_element(a) == atomic_number).count();
        Some(CentreLabel { centre: centre_idx, atomic_number, count })
    }

    /// Number of centres of every kind: one more than the highest centre index that exists.
    pub fn get_n_centres(&self) -> usize {
        self.qm_atoms.get_atoms().len()
    }

    /// Every centre's atomic number, position and kind, indexed by centre index. See `CentreGeometry`.
    pub fn get_centre_geometry(&self) -> Vec<CentreGeometry> {
        self.qm_atoms
            .get_atoms()
            .iter()
            .map(|a| CentreGeometry { atomic_number: a.get_atomic_number(), position: a.get_position(), kind: a.get_kind() })
            .collect()
    }

    /// Every centre's charge, from its nucleus or its core AIMP (an error if both or, for a real atom, neither); 0 for
    /// a ghost centre or an orthogonality ghost.
    pub fn get_nuclear_charges(&self) -> Result<Box<[f64]>, String> {
        let atoms = self.qm_atoms.get_atoms();
        let mut nuclear_charges = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            let nuclear_charge = match self.get_nuclear_potential() {
                Some(sources::NuclearPotential::PointCharge(pcs)) => {
                    pcs.iter().find(|pc| pc.get_idx() == i).map(|pc| pc.get_charge())
                },
                Some(sources::NuclearPotential::GaussianCharge(pcs)) => {
                    pcs.iter().find(|pc| pc.get_idx() == i).map(|pc| pc.get_charge())
                },
                None => None,
            };
            let core_charge = self
                .get_core_regions()
                .and_then(|aimps| aimps.iter().find(|aimp| aimp.get_idx() == i))
                .map(|aimp| aimp.get_charge());
            match (nuclear_charge, core_charge) {
                (Some(charge), None) | (None, Some(charge)) => nuclear_charges.push(charge),
                (Some(_), Some(_)) => {
                    return Err(format!("Centre {i} has both a nucleus and a core AIMP, something's broken in the input."))
                },
                (None, None) if atom.get_kind() != crate::molecule::AtomKind::Real => nuclear_charges.push(0.0),
                (None, None) => {
                    return Err(format!("Centre {i} has neither a nucleus nor a core AIMP, something's broken in the input."))
                },
            }
        }
        Ok(nuclear_charges.into_boxed_slice())
    }

}

/// An error reading the input system.
#[derive(Debug)]
pub enum SetupError {
    InputParseError(InputFileParseError),
    CurrentDirError(std::io::Error),
    UnkownElementSymbol { symbol: String },
    BasisFileNotFound { element: String, basis_file: String },
    BasisFileParseError { element: String, path: String, source: InputFileParseError },
    /// No basis shell.
    EmptyElectronicBasis,
    /// No nucleus and no core AIMP.
    NoCoreDescription,
    /// The ORTHOGONALITY option on a basis set with valence shells; it applies to pure embedding AIMPs only.
    OrthogonalityRequiresPureEmbeddingAimp { element: String },
    /// A basis block out of order (see `setup_basis_system`): `found` the block's kind, `after` the kind it may not
    /// follow.
    OutOfOrderBlock { element: String, found: &'static str, after: &'static str },
    /// A shell above `L_MAX`, the highest angular momentum the build supports.
    AngularMomentumAboveLMax { element: String, path: String, l: usize },
}

impl std::fmt::Display for SetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetupError::InputParseError(source) => write!(f, "failed to parse input file: {}", source),
            SetupError::CurrentDirError(source) => write!(f, "failed to determine current directory: {}", source),
            SetupError::UnkownElementSymbol { symbol } => write!(f, "unknown element symbol '{}'", symbol),
            SetupError::BasisFileNotFound { element, basis_file } => {
                write!(f, "basis file '{}' for element '{}' not found", basis_file, element)
            },
            SetupError::BasisFileParseError { element, path, source } => {
                write!(f, "failed to parse basis file '{}' for element '{}': {}", path, element, source)
            },
            SetupError::EmptyElectronicBasis => {
                write!(f, "empty electronic basis: at least one valence shell and one nuclear/core potential is required")
            },
            SetupError::NoCoreDescription => write!(f, "no core description found"),
            SetupError::OrthogonalityRequiresPureEmbeddingAimp { element } => {
                write!(f, "'orthogonality' option given for element '{}', but its basis set already declares valence shells -- a real core AIMP already has genuine orbitals to orthogonalize against and needs no derived ghost orbitals", element)
            }
            SetupError::OutOfOrderBlock { element, found, after } => {
                let article = |kind: &str| if kind.starts_with(|c| "aeiou".contains(c)) { "an" } else { "a" };
                write!(
                    f,
                    "basis block for element '{}' is {} {}, but it comes after {} {}. Blocks must appear in this order: \
                     real atoms and ghost centres (in any mix), then orthogonality ghosts, then pure embedding AIMPs -- \
                     so that a centre's index (e.g. the atom number a gradient refers to) can be read straight off the input",
                    element, article(found), found, article(after), after
                )
            }
            SetupError::AngularMomentumAboveLMax { element, path, l } => write!(
                f,
                "the basis of element '{}' ('{}') has a shell of angular momentum {}, above this build's L_MAX = {}; \
                 rebuild with the environment variable L_MAX={} (or higher)",
                element, path, l, constants::L_MAX, l
            ),
        }
    }
}

impl std::error::Error for SetupError {}

/// An error if a shell of `basis` (element `element`, read from `path`) is above `L_MAX`: the tables sized by it
/// (factorials, AO powers, the Boys function's orders) would be indexed past their end.
fn check_angular_momentum<'s>(basis: impl IntoIterator<Item = &'s InputBasisShell>, element: &str, path: &str) -> Result<(), SetupError> {
    match basis.into_iter().map(InputBasisShell::get_lqn).max() {
        Some(l) if l > constants::L_MAX => Err(SetupError::AngularMomentumAboveLMax { element: element.to_string(), path: path.to_string(), l }),
        _ => Ok(()),
    }
}

/// The system of `input_file`: its basis blocks read from the basis libraries (`find_library_file`: the current
/// directory, `AEDELSTEN_BASIS_LIBRARY`'s directories, the bundled library) into centres, shells, point nuclei, core and embedding AIMPs. The blocks must come in the order:
/// real atoms and ghost centres, interleaved; orthogonality ghosts; pure embedding AIMPs (no centre); centres are
/// numbered in input order.
pub fn setup_basis_system<S: crate::contraction::ShellKind>(
    input_file: &str,
) -> Result<InputStructure<S>, SetupError> {
    let parsed_input = basis_parser::parse_input_file(input_file).map_err(SetupError::InputParseError)?;

    let mut atoms = Vec::new();
    let mut basis_shells = Vec::new();
    let mut nuclear_potentials = Vec::new();
    let mut core_regions = Vec::new();
    let mut embedding_potentials = Vec::new();
    let mut order_stage = 0;
    let mut order_stage_name = "real atom or ghost centre";

    let mut cached_centre_idx = 0;
    for block in parsed_input.get_basis_blocks().iter() {
        let basis_file = block.get_basis_file();
        let basis_path = find_library_file(basis_file)?
            .ok_or_else(|| SetupError::BasisFileNotFound {
                element: block.get_element().to_string(),
                basis_file: basis_file.to_string(),
            })?;
        let parsed_basis = basis_parser::parse_basis_file(block.get_basis_set_spec(), basis_path.to_str().unwrap())
            .map_err(|source| SetupError::BasisFileParseError {
                element: block.get_element().to_string(),
                path: basis_path.display().to_string(),
                source,
            })?;

        // Every shell the block brings, valence and AIMP projection alike, within L_MAX before any is normalized.
        let shells = parsed_basis.get_valence().into_iter().flatten().chain(parsed_basis.get_core().into_iter().flat_map(|core| core.get_shells()));
        check_angular_momentum(shells, block.get_element(), &basis_path.display().to_string())?;

        if block.is_orthogonality() && parsed_basis.get_valence().is_some() {
            return Err(SetupError::OrthogonalityRequiresPureEmbeddingAimp {
                element: block.get_element().to_string(),
            });
        }

        // The block's stage in the order, known once its basis file is read.
        let (block_stage, block_kind) = if parsed_basis.get_valence().is_some() {
            if block.is_pseudocharge() { (0, "ghost centre") } else { (0, "real atom") }
        } else if block.is_orthogonality() {
            (1, "orthogonality ghost")
        } else {
            (2, "pure embedding AIMP")
        };
        if block_stage < order_stage {
            return Err(SetupError::OutOfOrderBlock {
                element: block.get_element().to_string(),
                found: block_kind,
                after: order_stage_name,
            });
        }
        order_stage = block_stage;
        order_stage_name = match block_stage { 0 => "real atom or ghost centre", 1 => "orthogonality ghost", _ => "pure embedding AIMP" };

        if let Some(basis) = parsed_basis.get_valence() {

            let used_shells = convert_input_basis_to_used_basis::<S>(
                basis,
                |lqn| block.resolve_harmonic_type(lqn),
                cached_centre_idx,
                block.get_centres()[0].get_position()
            );

            for (i, centre) in block.get_centres().iter().enumerate() {
                let kind = if block.is_pseudocharge() { crate::molecule::AtomKind::Ghost } else { crate::molecule::AtomKind::Real };
                let atom = crate::molecule::Atom::new_with_kind(block.get_element(), centre.get_position(), kind)?;
                let centre_idx = cached_centre_idx + i;
                debug_assert_eq!(atoms.len(), centre_idx);
                atoms.push(atom);
                let mut shells = used_shells.clone();
                let position = centre.get_position();
                for shell in shells.iter_mut() {
                    shell.set_centre_index(centre_idx);
                    shell.set_position(position);
                }
                basis_shells.extend(shells);
            }

            // Point nuclei, except on ghost centres and core-AIMP centres.
            if !block.is_pseudocharge() && parsed_basis.get_core().is_none() {
                // The input's charge, else the basis file's.
                let charge = block.get_charge().unwrap_or_else(|| parsed_basis.get_charge());
                for (i, centre) in block.get_centres().iter().enumerate() {
                    nuclear_potentials.push(charge_carriers::IndexedPointCharge::new(
                        cached_centre_idx + i,
                        centre.get_position(),
                        charge,
                    ));
                }
            } else if let Some(core) = parsed_basis.get_core() {
                // An AIMP's charge is the basis file's.
                let charge = parsed_basis.get_charge();
                let screen = Rc::new(core.get_potential().clone());
                let projection_shells = Rc::new(core.get_shells().to_vec());
                // The valence shells, for AimpTrait::get_resolution_basis.
                let valence_shells = Rc::new(basis.clone());
                for (i, centre) in block.get_centres().iter().enumerate() {
                    core_regions.push(IndexedAimp {
                        point_charge: charge_carriers::IndexedPointCharge::new(
                            cached_centre_idx + i,
                            centre.get_position(),
                            charge
                        ),
                        screen: Rc::clone(&screen),
                        projection_shells: Rc::clone(&projection_shells),
                        valence_shells: Rc::clone(&valence_shells),
                    });
                }
            }

            cached_centre_idx += block.get_centres().len();

        } else if let Some(embedding) = parsed_basis.get_core() {
            // An AIMP's charge is the basis file's.
            let charge = parsed_basis.get_charge();
            let atomic_number = constants::ELEMENT_SYMBOLS.iter().position(|&symbol| symbol == block.get_element()).unwrap_or(0) as u8;
            let screen = Rc::new(embedding.get_potential().clone());
            let projection_shells = Rc::new(embedding.get_shells().to_vec());
            for centre in block.get_centres().iter() {
                embedding_potentials.push(Aimp {
                    point_charge: PointCharge::new(centre.get_position(), charge),
                    screen: Rc::clone(&screen),
                    projection_shells: Rc::clone(&projection_shells),
                    atomic_number,
                });
            }

            if block.is_orthogonality() {
                // Orthogonality ghosts: the AIMP's orthogonality functions on centres of their own, no nucleus.
                let ghost_shells = embedding.get_orthogonality_shells();
                let used_shells = convert_input_basis_to_used_basis::<S>(
                    &ghost_shells,
                    |lqn| block.resolve_harmonic_type(lqn),
                    cached_centre_idx,
                    block.get_centres()[0].get_position()
                );
                for (i, centre) in block.get_centres().iter().enumerate() {
                    let centre_idx = cached_centre_idx + i;
                    let mut shells = used_shells.clone();
                    let position = centre.get_position();
                    for shell in shells.iter_mut() {
                        shell.set_centre_index(centre_idx);
                        shell.set_position(position);
                    }
                    basis_shells.extend(shells);
                    debug_assert_eq!(atoms.len(), centre_idx);
                    atoms.push(crate::molecule::Atom::new_with_kind(
                        block.get_element(),
                        position,
                        crate::molecule::AtomKind::OrthogonalityGhost,
                    )?);
                }
                cached_centre_idx += block.get_centres().len();
            }
        }
    }

    if basis_shells.is_empty() {
        return Err(SetupError::EmptyElectronicBasis);
    }
    let basis_info = fix_basis_indices_with_ids(&mut basis_shells);
    
    if nuclear_potentials.is_empty() && core_regions.is_empty() {
        return Err(SetupError::NoCoreDescription);
    }

    let ao_evaluator = ao_values::AoEvaluator::new(&basis_shells);
    Ok(InputStructure {
        qm_atoms: crate::molecule::Molecule::new(atoms),
        ao_evaluator,
        basis_shells,
        basis_info,
        nuclear_potentials: if nuclear_potentials.is_empty() {
            None
        } else {
            Some(crate::charge_carriers::NuclearPotential::PointCharge(nuclear_potentials.into_boxed_slice()))
        },
        core_regions: if core_regions.is_empty() { None } else { Some(core_regions) },
        embedding_potentials: if embedding_potentials.is_empty() { None } else { Some(embedding_potentials) },
    })
}

/// The shells of kind `S` of the parsed shells `basis` on centre `idx` at `position`, each shell's harmonic type
/// `resolve_harmonic_type(l)`.
pub fn convert_input_basis_to_used_basis<S: crate::contraction::ShellKind>(
    basis: &[InputBasisShell],
    resolve_harmonic_type: impl Fn(usize) -> shell_types::HarmonicType,
    idx: usize,
    position: [f64; 3]
) -> Vec<shell_types::BasisShell<S>> {
    let general_shells: Vec<shell_types::BasisShell<crate::contraction::Contracted>> = basis
        .iter()
        .map(|input_basis| {
            shell_types::BasisShell::<crate::contraction::Contracted>::new(
                input_basis,
                resolve_harmonic_type(input_basis.get_lqn()),
                idx,
                position,
            )
        })
        .collect();
    shell_types::BasisShell::<S>::reduce(&general_shells)
}

/// Assigns each shell its first AO and returns the basis's counts, without the export identities (they need every
/// shell on a real centre). For helper bases; `fix_basis_indices_with_ids` for the user-facing one.
pub fn fix_basis_indices<S: crate::contraction::ShellKind>(basis: &mut [shell_types::BasisShell<S>]) -> BasisInfo {
    assign_basis_indices(basis, false)
}

/// `fix_basis_indices` with the export identities (`BasisInfo::get_basis_function_ids`, `get_primitive_ids`,
/// `get_primitives`); every shell must be on a real centre.
pub fn fix_basis_indices_with_ids<S: crate::contraction::ShellKind>(basis: &mut [shell_types::BasisShell<S>]) -> BasisInfo {
    assign_basis_indices(basis, true)
}

// The first AO of each shell, the counts and, `with_ids`, the export identities.
fn assign_basis_indices<S: crate::contraction::ShellKind>(basis: &mut [shell_types::BasisShell<S>], with_ids: bool) -> BasisInfo {
    let mut n_cont = 0;
    let mut n_prim = 0;
    let mut basis_function_ids: Vec<BasisFunctionId> = Vec::new();
    let mut primitive_ids: Vec<PrimitiveId> = Vec::new();
    let mut primitives: Vec<Primitive> = Vec::new();
    // The next contracted-AO ordinal per (centre, lqn).
    let mut next_shell_ordinal: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for shell in basis.iter_mut() {
        shell.set_basis_index(n_cont);
        n_cont += shell.get_n_cont() * shell.get_degeneracy();
        n_prim += shell.get_n_prim() * shell.get_degeneracy();

        if !with_ids {
            continue;
        }

        let lqn = shell.get_lqn();
        let centre = shell.get_centre_index() + 1;
        let l = if shell.get_harmonic_type() == shell_types::HarmonicType::Cartesian { -(lqn as i32) } else { lqn as i32 };
        let cartesian_components = (l < 0).then(|| cart2sph::get_cartesian_quantum_numbers(lqn));
        // Contracted function outer, component inner: the AO order.
        for cont in 0..shell.get_n_cont() {
            let ordinal = next_shell_ordinal.entry((centre, lqn)).or_insert(0);
            *ordinal += 1;
            let shell_ordinal = *ordinal;
            match &cartesian_components {
                Some(components) => {
                    for &(lx, ly, lz) in components {
                        let m = (((ly + lz) * (ly + lz + 1)) / 2) as i32 - (lx + ly) as i32;
                        basis_function_ids.push(BasisFunctionId { centre, shell: shell_ordinal, l, m });
                    }
                }
                None => {
                    for m in -(lqn as i32)..=(lqn as i32) {
                        basis_function_ids.push(BasisFunctionId { centre, shell: shell_ordinal, l, m });
                    }
                }
            }
            for (exponent, coefficient) in shell.contracted_primitives(cont) {
                primitive_ids.push(PrimitiveId { centre, l: lqn, shell: shell_ordinal });
                primitives.push(Primitive { exponent, coefficient });
            }
        }
    }

    // Each (centre, l, exponent) once.
    let n_prim_unique = shell_types::group_unique_exponents(basis)
        .values()
        .map(|(degeneracy, exponents)| degeneracy * exponents.len())
        .sum();

    BasisInfo {
        n_cont,
        n_prim,
        n_prim_unique,
        basis_function_ids: basis_function_ids.into_boxed_slice(),
        primitive_ids: primitive_ids.into_boxed_slice(),
        primitives: primitives.into_boxed_slice(),
    }
}

#[cfg(test)]
mod case_insensitive_basis_file_tests {
    use super::*;

    // Against the bundled library.
    #[test]
    fn finds_the_real_file_regardless_of_requested_case() {
        let dir = std::path::Path::new(DEFAULT_BASIS_LIBRARY);
        for requested in ["ANO-RCC", "ano-rcc", "Ano-Rcc", "aNo-RcC"] {
            let found = find_basis_file_case_insensitive(dir, requested);
            assert!(found.is_some(), "'{}' should resolve to the real ANO-RCC file", requested);
            assert!(found.unwrap().exists());
        }
    }

    // A file outside the bundled library is found through AEDELSTEN_BASIS_LIBRARY (the tests' fixtures, set by
    // .cargo/config.toml), and only through it.
    #[test]
    fn a_library_file_is_found_through_the_variable() {
        assert!(find_basis_file_case_insensitive(std::path::Path::new(DEFAULT_BASIS_LIBRARY), "TEST-CORE-AIMP").is_none());
        let found = find_library_file("TEST-CORE-AIMP").unwrap().expect("the fixture through AEDELSTEN_BASIS_LIBRARY");
        assert!(found.ends_with("test_fixtures/basis/TEST-CORE-AIMP"), "{}", found.display());
    }

    // The installed copy is looked for beside the object holding the code: here the test executable,
    // `target/<profile>/deps/<test>`, so `target/<profile>/share/aedelsten/basis`.
    #[test]
    fn the_installed_library_is_found_from_the_loaded_object() {
        let object = loaded_object_path().expect("dladdr names the test executable");
        assert!(object.is_file(), "{}", object.display());
        let installed = installed_basis_library().unwrap();
        assert_eq!(installed, object.parent().unwrap().parent().unwrap().join("share/aedelsten/basis"));
    }

    #[test]
    fn returns_none_for_a_file_that_does_not_exist_under_any_casing() {
        let dir = std::path::Path::new(DEFAULT_BASIS_LIBRARY);
        assert!(find_basis_file_case_insensitive(dir, "not-a-real-basis-set").is_none());
    }
}

#[cfg(test)]
mod molcas_export_id_tests {
    use super::*;

    fn setup(tag: &str, body: &str) -> InputStructure<crate::contraction::Contracted> {
        let path = std::env::temp_dir().join(format!("aedelsten_basis_function_id_test_{}_{}.inp", std::process::id(), tag));
        std::fs::write(&path, body).unwrap();
        let result = setup_basis_system::<crate::contraction::Contracted>(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        result
    }

    // O x2c-SVPall (s, p, one d) with its d shell Cartesian.
    const INP: &str = "Basis set\n    O.x2c-SVPall\n    Cartesian d\n    O1 0.37 -1.82 0.94\nEnd of Basis\n";

    #[test]
    fn s_and_p_shells_are_reported_as_cartesian_with_the_expected_m_values() {
        let structure = setup("s_and_p", INP);
        let ids = structure.get_basis_info().get_basis_function_ids();
        // s: l = 0 either way.
        assert_eq!(ids[0], BasisFunctionId { centre: 1, shell: 1, l: 0, m: 0 });
        // p, Cartesian (l = -1), x y z.
        let p_start = ids.iter().position(|id| id.l == -1).expect("expected a Cartesian p shell");
        assert_eq!(&ids[p_start..p_start + 3], &[
            BasisFunctionId { centre: 1, shell: 1, l: -1, m: -1 }, // x
            BasisFunctionId { centre: 1, shell: 1, l: -1, m: 0 },  // y
            BasisFunctionId { centre: 1, shell: 1, l: -1, m: 1 },  // z
        ]);
    }

    #[test]
    fn cartesian_d_shell_matches_the_reference_m_values() {
        let structure = setup("cartesian_d", INP);
        let ids = structure.get_basis_info().get_basis_function_ids();
        let d_start = ids.iter().position(|id| id.l == -2).expect("expected a Cartesian d shell");
        // xx xy xz yy yz zz.
        let expected_m = [-2, -1, 0, 1, 2, 3];
        for (i, &m) in expected_m.iter().enumerate() {
            assert_eq!(ids[d_start + i], BasisFunctionId { centre: 1, shell: 1, l: -2, m }, "component {i}");
        }
    }

    // Cu x2c-SVPall's one f shell, forced Cartesian.
    #[test]
    fn cartesian_f_shell_matches_the_reference_m_values() {
        let inp = "Basis set\n    Cu.x2c-SVPall\n    Cartesian f\n    Cu1 0.0 0.0 0.0\nEnd of Basis\n";
        let structure = setup("cartesian_f", inp);
        let ids = structure.get_basis_info().get_basis_function_ids();
        let f_start = ids.iter().position(|id| id.l == -3).expect("expected a Cartesian f shell");
        // xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz.
        let expected_m = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6];
        for (i, &m) in expected_m.iter().enumerate() {
            assert_eq!(ids[f_start + i], BasisFunctionId { centre: 1, shell: 1, l: -3, m }, "component {i}");
        }
    }

    #[test]
    fn spherical_f_shell_uses_the_ascending_m_range() {
        let inp = "Basis set\n    Cu.x2c-SVPall\n    Cu1 0.0 0.0 0.0\nEnd of Basis\n";
        let structure = setup("spherical_f", inp);
        let ids = structure.get_basis_info().get_basis_function_ids();
        let f_start = ids.iter().position(|id| id.l == 3).expect("expected a Spherical f shell");
        for (i, m) in (-3..=3).enumerate() {
            assert_eq!(ids[f_start + i], BasisFunctionId { centre: 1, shell: 1, l: 3, m }, "component {i}");
        }
    }

    #[test]
    fn shell_ordinals_are_one_based_and_local_to_each_centre_lqn_pair() {
        let structure = setup("ordinals", INP);
        let ids = structure.get_basis_info().get_basis_function_ids();
        assert!(ids.iter().all(|id| id.centre == 1 && id.shell >= 1));
    }

    #[test]
    fn primitive_ids_use_the_plain_positive_l_unlike_basis_function_ids() {
        let structure = setup("primitive_l_sign", INP);
        let info = structure.get_basis_info();
        // The Cartesian d shell: l = -2 in BasisFunctionId, 2 in PrimitiveId.
        assert!(info.get_basis_function_ids().iter().any(|id| id.l == -2));
        assert!(info.get_primitive_ids().iter().any(|id| id.l == 2));
        assert!(!info.get_primitive_ids().iter().any(|id| id.l as i32 == -2));
    }

    #[test]
    fn every_shell_ordinal_seen_in_basis_function_ids_has_at_least_one_primitive() {
        let structure = setup("shell_coverage", INP);
        let info = structure.get_basis_info();
        let bf_shells: std::collections::HashSet<(usize, usize)> =
            info.get_basis_function_ids().iter().map(|id| (id.centre, id.l.unsigned_abs() as usize)).collect();
        let prim_shells: std::collections::HashSet<(usize, usize)> = info.get_primitive_ids().iter().map(|id| (id.centre, id.l)).collect();
        assert_eq!(bf_shells, prim_shells);
    }

    #[test]
    fn primitives_and_primitive_ids_are_parallel_and_h_ano_rcc_s_shell_is_genuinely_contracted() {
        let structure = setup("s_contraction", INP);
        let info = structure.get_basis_info();
        assert_eq!(info.get_primitive_ids().len(), info.get_primitives().len());

        let s_shell_primitive_count = info.get_primitive_ids().iter().filter(|id| id.centre == 1 && id.l == 0 && id.shell == 1).count();
        assert!(s_shell_primitive_count > 1, "expected more than one primitive, got {s_shell_primitive_count}");

        assert!(info.get_primitives().iter().all(|p| p.coefficient != 0.0));
    }

    #[test]
    fn centre_labels_use_a_per_element_counter_not_a_global_one() {
        let body = "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\n\
                     Basis set\n    O.x2c-SVPall\n    O1 1.0 0.0 0.0\nEnd of Basis\n\
                     Basis set\n    H.x2c-SVPall\n    H2 2.0 0.0 0.0\nEnd of Basis\n";
        let structure = setup("centre_labels", body);
        assert_eq!(structure.get_centre_label(0).unwrap().to_string(), "H1");
        assert_eq!(structure.get_centre_label(1).unwrap().to_string(), "O1");
        assert_eq!(structure.get_centre_label(2).unwrap().to_string(), "H2");
        assert_eq!(structure.get_centre_label(3), None);
    }

    // A helper basis on no centre (usize::MAX) is numbered without panicking and without identities.
    #[test]
    fn helper_bases_can_be_numbered_without_a_centre_and_get_no_identity_lists() {
        let structure = setup("helper_basis", INP);
        let mut shells = structure.get_basis_shells().to_vec();
        for shell in shells.iter_mut() {
            shell.set_centre_index(usize::MAX);
        }
        let info = fix_basis_indices(&mut shells);
        assert_eq!(info.get_n_cont(), structure.get_n_cont());
        assert!(info.get_basis_function_ids().is_empty());
        assert!(info.get_primitive_ids().is_empty() && info.get_primitives().is_empty());

        let with_ids = structure.get_basis_info();
        assert_eq!(with_ids.get_n_cont(), info.get_n_cont());
        assert_eq!(with_ids.get_n_prim(), info.get_n_prim());
        assert_eq!(with_ids.get_basis_function_ids().len(), with_ids.get_n_cont());
    }

    #[test]
    fn centre_geometry_is_indexed_by_centre_index() {
        let body = "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\n\
                     Basis set\n    O.x2c-SVPall\n    O1 1.0 0.5 -2.0\nEnd of Basis\n\
                     Basis set\n    H.x2c-SVPall\n    H2 2.0 0.0 0.0\nEnd of Basis\n";
        let structure = setup("centre_geometry", body);
        assert_eq!(structure.get_n_centres(), 3);
        let g = structure.get_centre_geometry();
        assert_eq!(g.iter().map(|c| c.atomic_number).collect::<Vec<_>>(), vec![1, 8, 1]);
        assert_eq!(g[1].position, [1.0, 0.5, -2.0]);
        for (i, c) in g.iter().enumerate() {
            assert_eq!(structure.get_centre_label(i).unwrap().atomic_number, c.atomic_number);
        }
    }

    #[test]
    fn centre_geometry_includes_orthogonality_ghost_centres_at_their_own_index() {
        // The atom is centre 0, the orthogonality ghost centre 1.
        let body = "Basis set\n    H.x2c-SVPall\n    H1 0.0 1.0 0.0\nEnd of Basis\n\
                     Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    ORTHOGONALITY\n    Na 5.0 0.0 0.0\nEnd of Basis\n";
        let structure = setup("centre_geometry_ghost", body);

        assert!(!structure.is_orthogonality_ghost_centre(0));
        assert!(structure.is_orthogonality_ghost_centre(1));
        assert_eq!(structure.get_qm_atoms().get_atoms().len(), 2);
        assert_eq!(structure.get_qm_atoms().real_atoms().count(), 1);

        assert_eq!(structure.get_n_centres(), 2);
        let g = structure.get_centre_geometry();
        assert_eq!(g[0], CentreGeometry { atomic_number: 1, position: [0.0, 1.0, 0.0], kind: AtomKind::Real });
        assert_eq!(g[1], CentreGeometry { atomic_number: 12, position: [5.0, 0.0, 0.0], kind: AtomKind::OrthogonalityGhost });
        assert_eq!(structure.get_centre_label(0).unwrap().to_string(), "H1");
        assert_eq!(structure.get_centre_label(1).unwrap().to_string(), "Mg1");

        let max_centre = structure.get_basis_info().get_basis_function_ids().iter().map(|id| id.centre).max().unwrap();
        assert!(max_centre <= structure.get_n_centres());
    }

}

#[cfg(test)]
mod ghost_centre_tests {
    use super::*;
    use crate::charge_carriers::{FiniteChargeTrait, NuclearPotential};

    const H1: &str = "Basis set\n    H.x2c-SVPall\n    H1 0.0 0.0 0.0\nEnd of Basis\n";
    const H2: &str = "Basis set\n    H.x2c-SVPall\n    H2 0.0 0.0 1.4\nEnd of Basis\n";
    const GHOST_H: &str = "Basis set\n    H.x2c-SVPall\n    GHOST\n    H3 0.0 0.0 2.8\nEnd of Basis\n";
    const PSEUDO_H: &str = "Basis set\n    H.x2c-SVPall\n    PSEUDOCHARGE\n    H3 0.0 0.0 2.8\nEnd of Basis\n";
    const ORTH_GHOST: &str = "Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    ORTHOGONALITY\n    Na 5.0 0.0 0.0\nEnd of Basis\n";
    const PURE_EMBEDDING: &str = "Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n    Na 6.0 0.0 0.0\nEnd of Basis\n";

    fn try_setup(tag: &str, blocks: &[&str]) -> Result<InputStructure<crate::contraction::Contracted>, SetupError> {
        let path = std::env::temp_dir().join(format!("aedelsten_ghost_centre_test_{}_{}.inp", std::process::id(), tag));
        std::fs::write(&path, blocks.concat()).unwrap();
        let result = setup_basis_system::<crate::contraction::Contracted>(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        result
    }

    fn setup(tag: &str, blocks: &[&str]) -> InputStructure<crate::contraction::Contracted> {
        try_setup(tag, blocks).unwrap()
    }

    #[test]
    fn real_atoms_and_ghost_centres_may_be_interleaved_and_keep_their_input_order_indices() {
        let s = setup("interleaved", &[H1, GHOST_H, H2, ORTH_GHOST, PURE_EMBEDDING]);
        assert_eq!(s.get_n_centres(), 4); // H, ghost H, H, orthogonality ghost
        assert_eq!(s.get_centre_geometry().iter().map(|c| c.atomic_number).collect::<Vec<_>>(), vec![1, 1, 1, 12]);
        assert_eq!(
            s.get_centre_geometry().iter().map(|c| c.kind).collect::<Vec<_>>(),
            vec![AtomKind::Real, AtomKind::Ghost, AtomKind::Real, AtomKind::OrthogonalityGhost]
        );
        for (i, atom) in s.get_qm_atoms().get_atoms().iter().enumerate() {
            assert_eq!(atom.get_position(), s.get_centre_geometry()[i].position);
        }
        assert_eq!(s.get_centre_geometry()[1].position, [0.0, 0.0, 2.8]);
        assert!(s.is_pseudocharge_centre(1));
        assert!(!s.is_pseudocharge_centre(2));
        assert!(s.is_orthogonality_ghost_centre(3));
    }

    #[test]
    fn ghost_and_orthogonality_ghost_centres_have_no_nuclear_charge() {
        let s = setup("nuclear_charges", &[H1, GHOST_H, H2, ORTH_GHOST, PURE_EMBEDDING]);
        assert_eq!(&*s.get_nuclear_charges().unwrap(), &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn ghost_and_pseudocharge_are_synonyms() {
        let a = setup("kw_ghost", &[H1, GHOST_H]);
        let b = setup("kw_pseudo", &[H1, PSEUDO_H]);
        assert!(a.is_pseudocharge_centre(1) && b.is_pseudocharge_centre(1));
        assert_eq!(a.get_centre_geometry(), b.get_centre_geometry());
        assert_eq!(a.get_nuclear_potential().unwrap().compute_nuclear_repulsion_energy(), b.get_nuclear_potential().unwrap().compute_nuclear_repulsion_energy());
    }

    #[test]
    fn a_shell_above_l_max_is_an_error_that_names_the_rebuild() {
        let shell = |l: usize| crate::basis_parser::InputBasisShell::new(l, vec![1.0].into(), arrays::matrix::Matrix::new_from_vec(1, 1, vec![1.0].into()), None, None);
        let within = [shell(0), shell(constants::L_MAX)];
        assert!(check_angular_momentum(&within, "Ac", "cc-pVQZ-X2C").is_ok());
        let above = [shell(1), shell(constants::L_MAX + 1)];
        match check_angular_momentum(&above, "Ac", "cc-pVQZ-X2C") {
            Err(e @ SetupError::AngularMomentumAboveLMax { l, .. }) => {
                assert_eq!(l, constants::L_MAX + 1);
                assert!(e.to_string().contains(&format!("L_MAX={}", constants::L_MAX + 1)), "{e}");
            }
            other => panic!("expected AngularMomentumAboveLMax, got {:?}", other),
        }
    }

    #[test]
    fn blocks_out_of_order_are_errors_that_name_both_kinds() {
        let expect = |tag: &str, blocks: &[&str], found: &str, after: &str| match try_setup(tag, blocks) {
            Err(SetupError::OutOfOrderBlock { found: f, after: a, .. }) => assert_eq!((f, a), (found, after), "{tag}"),
            other => panic!("{tag}: expected OutOfOrderBlock, got {:?}", other.map(|_| ())),
        };
        expect("orth_before_atom", &[ORTH_GHOST, H1], "real atom", "orthogonality ghost");
        expect("orth_before_ghost", &[H1, ORTH_GHOST, GHOST_H], "ghost centre", "orthogonality ghost");
        expect("embedding_before_atom", &[PURE_EMBEDDING, H1], "real atom", "pure embedding AIMP");
        expect("embedding_before_orth", &[H1, PURE_EMBEDDING, ORTH_GHOST], "orthogonality ghost", "pure embedding AIMP");
        // The message spells out the rule.
        let message = try_setup("message", &[ORTH_GHOST, H1]).err().unwrap().to_string();
        assert!(message.contains("real atoms and ghost centres") && message.contains("orthogonality ghosts") && message.contains("pure embedding"));
    }

    #[test]
    fn a_ghost_centre_changes_neither_centre_of_mass_nor_nuclear_repulsion_nor_nuclear_charge() {
        let with = setup("neutral_with", &[H1, H2, ORTH_GHOST]);
        let without = setup("neutral_without", &[H1, H2]);
        assert!(with.is_orthogonality_ghost_centre(2));

        assert_eq!(with.get_qm_atoms().compute_centre_of_mass(), without.get_qm_atoms().compute_centre_of_mass());

        let repulsion = |s: &InputStructure<crate::contraction::Contracted>| s.get_nuclear_potential().unwrap().compute_nuclear_repulsion_energy();
        assert!(repulsion(&without) > 0.0, "sanity: two protons repel");
        assert_eq!(repulsion(&with), repulsion(&without));

        let charge = |s: &InputStructure<crate::contraction::Contracted>| match s.get_nuclear_potential().unwrap() {
            NuclearPotential::PointCharge(c) => c.iter().map(|c| c.get_charge()).sum::<f64>(),
            NuclearPotential::GaussianCharge(c) => c.iter().map(|c| c.get_charge()).sum::<f64>(),
        };
        assert_eq!(charge(&with), charge(&without));
    }

    #[test]
    fn finite_nuclear_model_gives_each_atom_its_own_index_position_and_exponent() {
        let mut structure = setup("finite_nuclei", &[H1, H2, ORTH_GHOST]);
        structure.set_finite_nuclear_model(crate::charge_carriers::FiniteModel::VisscherDyall);

        let Some(NuclearPotential::GaussianCharge(charges)) = structure.get_nuclear_potential() else {
            panic!("expected Gaussian nuclei after set_finite_nuclear_model");
        };
        assert_eq!(charges.iter().map(|c| c.get_idx()).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(*charges[0].get_position(), [0.0, 0.0, 0.0]);
        assert_eq!(*charges[1].get_position(), [0.0, 0.0, 1.4]);
        assert_eq!(charges[0].get_exponents(), charges[1].get_exponents());
    }

    #[test]
    fn centre_of_mass_excludes_both_kinds_of_ghost() {
        let plain = setup("com_plain", &[H1, H2]);
        let with_ghosts = setup("com_ghosts", &[H1, H2, GHOST_H, ORTH_GHOST]);
        assert_eq!(with_ghosts.get_qm_atoms().real_atoms().count(), 2);
        assert_eq!(with_ghosts.get_qm_atoms().compute_centre_of_mass(), plain.get_qm_atoms().compute_centre_of_mass());
    }

    #[test]
    fn atom_kinds_are_recorded_per_centre() {
        let s = setup("kinds", &[H1, GHOST_H, ORTH_GHOST]);
        assert!(!s.is_pseudocharge_centre(0) && !s.is_orthogonality_ghost_centre(0));
        assert!(s.is_pseudocharge_centre(1) && s.is_orthogonality_ghost_centre(2));
    }

    #[test]
    fn labels_are_x_for_ghost_centres_and_the_parent_element_for_orthogonality_ghosts() {
        let s = setup("labels", &[H1, GHOST_H, H2, ORTH_GHOST]);
        let label = |i| s.get_centre_label(i).unwrap().to_string();
        assert_eq!(label(0), "H1");
        assert_eq!(label(1), "X1");
        assert_eq!(label(2), "H2");
        assert_eq!(label(3), "Mg1");
        assert_eq!(s.get_centre_label(4), None);

        let real_mg = "Basis set\n    Mg.x2c-SVPall\n    Mg1 1.0 0.0 0.0\nEnd of Basis\n";
        let t = setup("labels_two", &[real_mg, GHOST_H, GHOST_H, ORTH_GHOST]);
        let label = |i| t.get_centre_label(i).unwrap().to_string();
        assert_eq!((label(0).as_str(), label(1).as_str(), label(2).as_str(), label(3).as_str()), ("Mg1", "X1", "X2", "Mg2"));
    }
}
