// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Point-group symmetry of a molecule's AO basis: the point group and its irreps through libmsym (MIT, vendored in
//! src/libmsym), and on top of them symmetry cleaning of AO matrices and MO coefficients by group averaging with the
//! AO representation D(R) of every symmetry operation R.
//!
//! Symmetry is implicit: nothing is ever transformed to a symmetry-blocked basis for the caller, matrices stay full
//! AO matrices, and cleaning removes what does not belong.
//!
//! libmsym never moves or rotates the molecule: it works relative to the centre of mass, and the symmetry operations
//! are expressed in the input's own axes, whatever its orientation. It does choose a standard orientation of the group
//! (principal axis on z, ...), which only decides the irrep names: B1/B2, B1g/B2g/B3g, ... refer to the standard axes,
//! see `Symmetry::standard_axes`.
//!
//! D(R) maps each AO block (one shell's angular components, for one contraction) onto the block of the same radial
//! function on the image centre, rotated by libmsym's real-solid-harmonic representation of R. With
//! `(O_R f)(r) = f(R^-1 r)` and `O_R χ_μ = Σ_ν χ_ν D(R)_νμ`, an AO matrix of an operator transforms as
//! `M -> D(R)^T M D(R)` and MO coefficients as `c -> D(R) c`. D(R) is orthogonal and never formed: `M D(R)` is one
//! product per block of columns, and `D(R)^T M D(R) = ((M D(R))^T D(R))^T`.
//!
//! The method is the standard one of group theory: projection and transfer operators built from characters and
//! irrep matrices (F. A. Cotton, Chemical Applications of Group Theory, ch. 6).

mod msym;

use arrays::array3d::{Array3D, Array3DTrait, Array3DTraitMut, Array3DView, Array3DViewMut};
use arrays::matrix::{
    block_sandwich_prealloc, block_signed_permutation_prealloc, ColumnFactor, Factor, Matrix, MatrixMath, MatrixTrait, MatrixTraitMut, MatrixView,
    MatrixViewMut, SignedPermutation,
};
use arrays::vector_math;
use arrays::scratch::{Arena, ArenaError};
use msym::*;
use std::collections::HashMap;
use std::os::raw::{c_char, c_int};
use std::ptr;

/// How far (Bohr) an atom may lie from its symmetry-equivalent position. Detection itself uses libmsym's own, much
/// looser, tolerances (about 1e-3), so a geometry that is almost symmetric is recognized and then rejected here with
/// the offending displacement, rather than silently treated as asymmetric.
pub const DEFAULT_GEOMETRY_TOLERANCE: f64 = 1e-8;

// How far two unit axes may be from parallel and still count as the same axis.
const AXIS_TOLERANCE: f64 = 1e-6;

// How far (Bohr) an operation's image of a centre may be from the centre it is matched with. Equivalent centres are
// within the geometry tolerance of their exact images; distinct centres are never this close.
const IMAGE_TOLERANCE: f64 = 1e-3;

// How far libmsym's partner functions may be from spanning an invariant space under D(R): a check, at setup, that D(R)
// is the representation libmsym's own SALCs transform under.
const REPRESENTATION_TOLERANCE: f64 = 1e-8;

/// A centre as libmsym sees it. Centres are equivalent (may be mapped onto each other by a symmetry operation) only
/// if they have the same `kind`: give each distinct combination of element, basis set, nucleus or core AIMP and
/// ghost status its own kind. `mass` only places the centre of mass; any value works as long as it is the same
/// within a kind.
#[derive(Debug, Clone, Copy)]
pub struct SymmetryCentre {
    /// Its kind: only centres of one kind are equivalent.
    pub kind: u32,
    /// Its mass, for the centre of mass.
    pub mass: f64,
    /// Its position.
    pub position: [f64; 3],
}

/// An AO: on `centre` (0-based), with angular momentum `l`. Spherical (`cartesian` false): a real solid harmonic with
/// real component `m` (for p: x is +1, y is -1, z is 0; for d and up the standard ascending -l..=l convention without
/// the Condon-Shortley phase). Cartesian (`cartesian` true, d and up; p is the same functions as spherical p): the
/// component x^lx y^ly z^lz in position `m` of the library's order (lx descending, then ly descending), each
/// component normalized on its own. `radial` (1-based) tells AOs with the same `(centre, l, m)` apart and must be
/// unique per `(centre, l, m)`, and the same for AOs that are images of each other on equivalent centres. The
/// components of one `(centre, radial, l)` (2l + 1 spherical, (l+1)(l+2)/2 Cartesian) must be consecutive, in the
/// same order on equivalent centres.
#[derive(Debug, Clone, Copy)]
pub struct SymmetryBasisFunction {
    /// The centre, 0-based.
    pub centre: usize,
    /// The radial function, 1-based.
    pub radial: usize,
    /// The angular momentum.
    pub l: u32,
    /// The component.
    pub m: i32,
    /// Whether the AO is Cartesian.
    pub cartesian: bool,
}

// The index of a block shape (an l, spherical or Cartesian) in `Symmetry::shapes`.
fn shape_id(l: usize, cartesian: bool) -> usize {
    2 * l + cartesian as usize
}

// The number of components of a block shape.
fn shape_len(shape: usize) -> usize {
    let l = shape / 2;
    if shape % 2 == 1 { (l + 1) * (l + 2) / 2 } else { 2 * l + 1 }
}

/// Which point group to use.
#[derive(Debug, Clone, Copy)]
pub enum PointGroupRequest<'a> {
    /// The full point group of the molecule.
    Highest,
    /// A named group (e.g. "C2v", case-insensitive): the full group or one of its subgroups. A subgroup can occur in
    /// several orientations (D2h has three C2v, one per C2 axis); `principal_axis` (any length, either sign) picks the
    /// one whose standard z axis (the principal rotation axis; for Cs the mirror plane's normal) lies along it. Without
    /// it, the orientation with its principal axis along the input's z is taken, or the only one there is.
    Named { name: &'a str, principal_axis: Option<[f64; 3]> },
}

/// An irreducible representation of the point group.
#[derive(Debug, Clone)]
pub struct Irrep {
    /// The irrep's name, as libmsym gives it.
    pub name: String,
    /// Dimension of the (real) representation.
    pub dimension: usize,
    /// How many complex irreps it is the sum of: 1 normally, 2 for the real pair of complex-conjugate irreps that
    /// libmsym calls one E in groups like C3, C4h, S4, T. Partner components are only defined (and only aligned) for 1.
    pub reality: usize,
}

/// How an AO matrix transforms, i.e. which part of it `symmetrise_matrix` keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorSymmetry {
    /// Totally symmetric (overlap, Hamiltonian, Fock, density of a totally symmetric state, ...).
    TotallySymmetric,
    /// Transforms as irrep `index` of `Symmetry::irreps`, e.g. a dipole component (`Symmetry::cartesian_irrep` gives
    /// it). In a degenerate irrep, the part of the matrix in that irrep as a whole is kept (all its components).
    Irrep(usize),
    /// Whichever irrep carries most of the matrix (see `Symmetry::irrep_norms`).
    Auto,
}

/// What `symmetrise_matrix` removed (or would have removed: when it refused, and from `analyse_matrix`). Norms are
/// Frobenius norms in the AO basis, in the matrix's own units.
#[derive(Debug, Clone, Copy)]
pub struct SymmetrisationReport {
    /// The irrep the matrix was taken to transform as (index into `Symmetry::irreps`).
    pub irrep: usize,
    /// Largest absolute change of a single AO element.
    pub max_abs_removed: f64,
    /// Norm of the removed part.
    pub removed_norm: f64,
    /// Norm of the matrix before cleaning.
    pub norm: f64,
}

/// An MO's symmetry label: its irrep, and for a degenerate irrep which of its partner components it is (libmsym's
/// canonical components, in the group's standard orientation; always 0 for a one-dimensional irrep, and for an irrep
/// with `reality` 2).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct OrbitalLabel {
    /// The irrep, an index into `Symmetry::irreps`.
    pub irrep: usize,
    /// The partner component.
    pub component: usize,
}

/// A failure of the symmetry setup or a refused cleaning, with its message.
#[derive(Debug)]
pub struct SymmetryError(pub String);

impl std::fmt::Display for SymmetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SymmetryError {}

impl From<String> for SymmetryError {
    fn from(msg: String) -> Self {
        SymmetryError(msg)
    }
}

impl From<ArenaError> for SymmetryError {
    fn from(error: ArenaError) -> Self {
        SymmetryError(format!("the scratch is too small for the symmetry tables: {error}"))
    }
}

// One shell's angular components for one radial function: consecutive AOs from `offset`, of a block shape (`shape_id`).
#[derive(Debug, Clone, Copy)]
struct AngularBlock {
    offset: usize,
    shape: usize,
}

impl AngularBlock {
    fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + shape_len(self.shape)
    }
}

/// Which AO basis a matrix is in: the contracted one (`n_cont`), or the primitive one (`n_prim_unique`, the basis of
/// the primitive integrals and of X2C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoBasis {
    Contracted,
    Primitive,
}

// One AO basis's side of the representation: its blocks, and the block each operation takes each block to.
#[derive(Debug)]
struct AoRepresentation {
    n_ao: usize,
    blocks: Vec<AngularBlock>,
    // (n_blocks, h).
    images: Matrix<usize>,
    // How many AOs (symmetry-adapted functions) each irrep has.
    ao_irrep_counts: Vec<usize>,
}

impl AoRepresentation {
    fn image(&self, op: usize, b: usize) -> &AngularBlock {
        &self.blocks[self.images[(b, op)]]
    }

    // Whether (b1, b2) is the smallest pair of its orbit (by column, then row).
    fn is_representative(&self, b1: usize, b2: usize) -> bool {
        (0..self.images.cols()).all(|op| (self.images[(b2, op)], self.images[(b1, op)]) >= (b2, b1))
    }

    // The angular blocks of `basis` (the consecutive components of one (centre, radial, l), spherical or Cartesian),
    // recording each shape's component order in `m_order` (it must be the same in every block of a shape, in either
    // basis; Cartesian components in the library's order), and the blocks each operation maps them onto, from where it
    // maps the centres (`centre_images`, (n_centres, h)).
    fn new(basis: &[SymmetryBasisFunction], centre_images: &Matrix<usize>, m_order: &mut [Option<Vec<usize>>]) -> Result<Self, SymmetryError> {
        let n_ao = basis.len();
        let mut blocks = Vec::new();
        let mut block_of: HashMap<(usize, usize, u32, bool), usize> = HashMap::new();
        let mut offset = 0;
        while offset < n_ao {
            let first = basis[offset];
            let l = first.l as usize;
            let cartesian = first.cartesian && l > 1;
            let shape = shape_id(l, cartesian);
            let len = shape_len(shape);
            let key = |bf: &SymmetryBasisFunction| (bf.centre, bf.radial, bf.l, bf.cartesian && bf.l > 1);
            let block = basis.get(offset..offset + len).filter(|block| block.iter().all(|bf| key(bf) == key(&first)));
            let Some(block) = block else {
                return Err(SymmetryError(format!("the {len} components of the AO block at {offset} ({first:?}) are not consecutive")));
            };
            // Spherical: m + l per position; Cartesian: the position in the library's order, which must be the order.
            let order: Vec<usize> = block.iter().map(|bf| if cartesian { bf.m as usize } else { (bf.m + first.l as i32) as usize }).collect();
            if cartesian && order.iter().enumerate().any(|(i, &m)| i != m) {
                return Err(SymmetryError(format!("the Cartesian components of the AO block at {offset} ({first:?}) are not in the library's order")));
            }
            match &m_order[shape] {
                Some(known) if *known != order => {
                    return Err(SymmetryError(format!("the l = {l} blocks do not all order their components the same way")));
                }
                Some(_) => {}
                None => m_order[shape] = Some(order),
            }
            block_of.insert(key(&first), blocks.len());
            blocks.push(AngularBlock { offset, shape });
            offset += len;
        }
        let mut images = Matrix::<usize>::new(blocks.len(), centre_images.cols());
        for (op, (column, centre_image)) in images.iter_mut().zip(centre_images.iter()).enumerate() {
            for (image, block) in column.iter_mut().zip(&blocks) {
                let first = basis[block.offset];
                *image = *block_of.get(&(centre_image[first.centre], first.radial, first.l, block.shape % 2 == 1)).ok_or_else(|| {
                    SymmetryError(format!("operation {op}: centre {} has no AO block matching {first:?}", centre_image[first.centre] + 1))
                })?;
            }
        }
        Ok(AoRepresentation { n_ao, blocks, images, ao_irrep_counts: Vec::new() })
    }
}

// Per l, which of D(op)'s blocks are signed permutations, and as which: column j of the block has its nonzero sign in
// row rows[(j, op)] (and the transposed block likewise).
#[derive(Debug)]
struct BlockPermutations {
    exact: Vec<bool>,
    rows: Matrix<usize>,
    signs: Matrix<f64>,
    transposed_rows: Matrix<usize>,
    transposed_signs: Matrix<f64>,
}

impl BlockPermutations {
    fn of(rotation: &Array3D<f64>) -> Self {
        let (d, h) = (rotation.dim0(), rotation.dim2());
        let mut permutations = BlockPermutations {
            exact: vec![false; h],
            rows: Matrix::new(d, h),
            signs: Matrix::new(d, h),
            transposed_rows: Matrix::new(d, h),
            transposed_signs: Matrix::new(d, h),
        };
        for (op, slab) in rotation.iter().enumerate() {
            let mut exact = true;
            for (j, column) in slab.iter_col().enumerate() {
                let nonzero: Vec<usize> = (0..d).filter(|&i| column[i].abs() > 1e-12).collect();
                if nonzero.len() != 1 || (column[nonzero[0]].abs() - 1.0).abs() > 1e-12 {
                    exact = false;
                    break;
                }
                let (i, sign) = (nonzero[0], column[nonzero[0]].signum());
                permutations.rows[(j, op)] = i;
                permutations.signs[(j, op)] = sign;
                permutations.transposed_rows[(i, op)] = j;
                permutations.transposed_signs[(i, op)] = sign;
            }
            permutations.exact[op] = exact;
        }
        permutations
    }

    // D(op)'s block (or its transpose) as a signed permutation; meaningful only where `exact[op]`.
    fn factor(&self, op: usize, transposed: bool) -> SignedPermutation<'_> {
        let (rows, signs) = if transposed { (&self.transposed_rows, &self.transposed_signs) } else { (&self.rows, &self.signs) };
        SignedPermutation { rows: rows.get_col(op), signs: signs.get_col(op) }
    }
}

// The matrix a tile pass works on: read only (`analyse_matrix`, the report pass of a strict clean) or written in place.
trait Tiles {
    fn read(&self) -> MatrixView<'_, f64>;
    fn write(&mut self, at: (usize, usize), tile: &MatrixView<'_, f64>);
}

struct ReadOnly<'a>(MatrixView<'a, f64>);

impl Tiles for ReadOnly<'_> {
    fn read(&self) -> MatrixView<'_, f64> {
        self.0.view()
    }

    fn write(&mut self, _: (usize, usize), _: &MatrixView<'_, f64>) {}
}

struct Writable<'a, 'm>(&'a mut MatrixViewMut<'m, f64>);

impl Tiles for Writable<'_, '_> {
    fn read(&self) -> MatrixView<'_, f64> {
        self.0.view()
    }

    // The cleaned tile is the result: it goes into the caller's matrix, column piece by column piece.
    fn write(&mut self, at: (usize, usize), tile: &MatrixView<'_, f64>) {
        for (j, column) in tile.iter_col().enumerate() {
            self.0.get_col_mut(at.1 + j)[at.0..at.0 + column.len()].copy_from_slice(column);
        }
    }
}

// The planes of a Cartesian tensor's AO matrices (`(n_ao, n_ao, n_components)`), read only or written in place.
trait PlaneTiles {
    fn read(&self, plane: usize) -> MatrixView<'_, f64>;
    fn write(&mut self, plane: usize, at: (usize, usize), tile: &MatrixView<'_, f64>);
}

struct ReadOnlyPlanes<'a>(Array3DView<'a, f64>);

impl PlaneTiles for ReadOnlyPlanes<'_> {
    fn read(&self, plane: usize) -> MatrixView<'_, f64> {
        self.0.view_sub_matrix(plane)
    }

    fn write(&mut self, _: usize, _: (usize, usize), _: &MatrixView<'_, f64>) {}
}

struct WritablePlanes<'a, 'm>(&'a mut Array3DViewMut<'m, f64>);

impl PlaneTiles for WritablePlanes<'_, '_> {
    fn read(&self, plane: usize) -> MatrixView<'_, f64> {
        self.0.view_sub_matrix(plane)
    }

    // The cleaned tile is the result: it goes into the caller's plane, column piece by column piece.
    fn write(&mut self, plane: usize, at: (usize, usize), tile: &MatrixView<'_, f64>) {
        let mut target = self.0.view_sub_matrix_mut(plane);
        for (j, column) in tile.iter_col().enumerate() {
            target.get_col_mut(at.1 + j)[at.0..at.0 + column.len()].copy_from_slice(column);
        }
    }
}

/// How the components of a Cartesian tensor operator transform: as the monomials of its order (a multipole: `Polar`),
/// or, for order 1, as an axial vector (the spin-orbit components: `Axial`, an extra det(R) under improper operations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    Polar,
    Axial,
}

// The position of x^lx y^ly z^(n-lx-ly) among the components of order n, in the library's order (lx descending, then ly
// descending: the multipole planes').
fn component_index(n: usize, lx: usize, ly: usize) -> usize {
    (n - lx) * (n - lx + 1) / 2 + (n - lx - ly)
}

// One block shape's representation: (d, d, h) each, d its number of components.
#[derive(Debug)]
struct Shape {
    // D(op)'s block, rows the image block's components, columns the block's, in the AO order of the components.
    rotations: Array3D<f64>,
    // Their inverses, for carrying a cleaned tile back to an image: the transposes for spherical shells (orthogonal),
    // true inverses for Cartesian ones.
    inverses: Array3D<f64>,
    // Which blocks are signed permutations, for the fast path.
    permutations: BlockPermutations,
    // The invariant metric's block, (1/h) Σ_op D(op)^T D(op), for a Cartesian shape (its blocks are not orthogonal);
    // `None` for a spherical one (the identity).
    metric: Option<Matrix<f64>>,
}

impl Shape {
    // D(op)'s block, or its inverse.
    fn block(&self, op: usize, inverse: bool) -> MatrixView<'_, f64> {
        if inverse { self.inverses.view_sub_matrix(op) } else { self.rotations.view_sub_matrix(op) }
    }
}

// The normalization of the Cartesian component x^lx y^ly z^lz up to a factor common to its shell:
// sqrt(lx! ly! lz! / ((2lx)! (2ly)! (2lz)!)), as every component is normalized on its own.
fn cartesian_norm(lx: usize, ly: usize, lz: usize) -> f64 {
    let factorial = |n: usize| (1..=n).map(|k| k as f64).product::<f64>();
    ((factorial(lx) * factorial(ly) * factorial(lz)) / (factorial(2 * lx) * factorial(2 * ly) * factorial(2 * lz))).sqrt()
}

// The transformation of the order-`order` Cartesian monomials under each of `operations`, into `out` ((n, n h), block
// op in columns op n..(op + 1) n, n the number of components): monomial_a(R^T r) = Σ_b T(R)_ba monomial_b(r), times
// det(R) for an axial tensor. The polynomial is expanded factor by factor (each a linear form, a row of R^T), in two
// buffers of n f64 (`scratch`, 2 n).
fn monomial_transformations(operations: &[([[f64; 3]; 3], usize)], order: usize, kind: TensorKind, out: &mut MatrixViewMut<'_, f64>, scratch: &mut [f64]) {
    let n = (order + 1) * (order + 2) / 2;
    let (current, next) = scratch.split_at_mut(n);
    for (op, (m, _)) in operations.iter().enumerate() {
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let sign = if kind == TensorKind::Axial { det.signum() } else { 1.0 };
        let mut a = 0;
        for lx in (0..=order).rev() {
            for ly in (0..=order - lx).rev() {
                let lz = order - lx - ly;
                // (R^T r)_i = Σ_j m[j][i] r_j: x is factor i = 0, y 1, z 2; multiply them in, degree by degree.
                current[0] = 1.0;
                let mut degree = 0;
                for i in std::iter::repeat_n(0, lx).chain(std::iter::repeat_n(1, ly)).chain(std::iter::repeat_n(2, lz)) {
                    let width = (degree + 2) * (degree + 3) / 2;
                    next[..width].fill(0.0);
                    for px in (0..=degree).rev() {
                        for py in (0..=degree - px).rev() {
                            let c = current[component_index(degree, px, py)];
                            if c != 0.0 {
                                next[component_index(degree + 1, px + 1, py)] += c * m[0][i];
                                next[component_index(degree + 1, px, py + 1)] += c * m[1][i];
                                next[component_index(degree + 1, px, py)] += c * m[2][i];
                            }
                        }
                    }
                    degree += 1;
                    current[..width].copy_from_slice(&next[..width]);
                }
                for b in 0..n {
                    out[(b, op * n + a)] = sign * current[b];
                }
                a += 1;
            }
        }
    }
}

/// The point group of a molecule and the AO representation of its operations.
///
/// Persistent data, built once by `new`: the rotation blocks and their transposes and permutations (a few h (2l+1)^2
/// per l up to lmax), the image table (n_blocks x h), the character and projection-weight tables (h x n_irreps), and
/// the partner transfer weights (h d^2 per degenerate irrep); none scales as n_ao^2. The per-call methods take their
/// scratch from an arena, never n_ao^2: `matrix_scratch_len()` f64s for the matrix methods (a few tiles),
/// `orbital_scratch_len()` for `symmetrise_orbitals` (without alignment padding).
///
/// Matrices are cleaned by orbits of block pairs: each D(R) maps the block pair (b1, b2) to (R b1, R b2), rotated, so
/// the pairs fall into orbits of at most h members, and an orbit's projection depends only on its own blocks. A pass
/// visits every orbit once, through its representative (the smallest pair of the orbit), in place.
#[derive(Debug)]
pub struct Symmetry {
    ctx: Context,
    point_group: String,
    irreps: Vec<Irrep>,
    // (n_irreps, n_classes).
    characters: Matrix<f64>,
    totally_symmetric: usize,
    // Every symmetry operation as a 3x3 matrix in the input's axes (acting on positions relative to the centre of
    // mass), with its class (character-table column).
    operations: Vec<([[f64; 3]; 3], usize)>,
    // (n_centres, h): the centre (0-based) each operation takes each centre to.
    centre_images: Matrix<usize>,
    standard_axes: [[f64; 3]; 3],
    // The contracted basis, and the primitive one when it was asked for (`new`'s `primitive_basis`).
    ao: AoRepresentation,
    primitive: Option<AoRepresentation>,
    // Per block shape (`shape_id`: an l, spherical or Cartesian), D(op)'s blocks and what is derived from them.
    shapes: Vec<Shape>,
    // The stand-in libmsym gets for a basis with Cartesian shells (spherical ones of the same l on the same centres),
    // for reading libmsym's SALCs; `None` when the basis is libmsym's own.
    stand_in: Option<AoRepresentation>,
    // The largest block.
    max_block: usize,
    // (h, n_irreps): the weight of each operation in each irrep's projector, d χ(R) / (h r).
    projector_weights: Matrix<f64>,
    // Per irrep, when it is degenerate, absolutely irreducible and present in the basis: (h, d^2), column j + d k the
    // weights (d/h) Γ(R)_jk of the transfer operator P_jk, Γ(R) in libmsym's canonical components.
    transfer_weights: Vec<Option<Matrix<f64>>>,
}

// A libmsym context with the elements and basis functions set and the full point group found.
fn detect(centres: &[SymmetryCentre], basis: &[SymmetryBasisFunction]) -> Result<Context, String> {
    let ctx = Context::new()?;
    // Equivalence in libmsym needs equal name, nuclear charge and mass. The kind goes into the charge (any positive
    // number is accepted when mass and name are given too), so it alone decides equivalence.
    let mut elements: Vec<msym_element_t> = centres
        .iter()
        .map(|c| msym_element_t {
            id: ptr::null_mut(),
            m: if c.mass > 0.0 { c.mass } else { 1.0 },
            v: c.position,
            n: c.kind as c_int + 1,
            name: [b'X' as c_char, 0, 0, 0],
        })
        .collect();
    // SAFETY: elements outlives every call below that dereferences the pointers into it (libmsym copies the elements
    // in msymSetElements and only compares basis-function element pointers against this array in
    // msymSetBasisFunctions, translating them to its own copy).
    unsafe {
        check(msymSetElements(ctx.raw(), elements.len() as c_int, elements.as_mut_ptr()), "set the elements")?;
        let mut functions = Vec::with_capacity(basis.len());
        for bf in basis {
            if bf.centre >= elements.len() || bf.radial == 0 {
                return Err(format!("invalid basis function {bf:?} for {} centres", elements.len()));
            }
            functions.push(msym_basis_function_t {
                id: ptr::null_mut(),
                type_: MSYM_BASIS_TYPE_REAL_SPHERICAL_HARMONIC,
                element: elements.as_mut_ptr().add(bf.centre),
                f: msym_basis_function_f { rsh: msym_real_spherical_harmonic_t { n: (bf.radial + bf.l as usize) as c_int, l: bf.l as c_int, m: bf.m } },
                name: [0; 8],
            });
        }
        check(msymSetBasisFunctions(ctx.raw(), functions.len() as c_int, functions.as_mut_ptr()), "set the basis functions")?;
        check(msymFindSymmetry(ctx.raw()), "find the point group")?;
    }
    Ok(ctx)
}

fn point_group_name(ctx: &Context) -> Result<String, String> {
    let mut buf = [0 as c_char; 8];
    // SAFETY: buf has the length passed.
    check(unsafe { msymGetPointGroupName(ctx.raw(), buf.len() as c_int, buf.as_mut_ptr()) }, "get the point group name")?;
    Ok(c_chars_to_string(&buf))
}

// Names of the subgroups libmsym lists for the context's current group, in its order.
fn subgroup_names(ctx: &Context) -> Result<Vec<String>, String> {
    let mut n: c_int = 0;
    let mut subgroups: *const msym_subgroup_t = ptr::null();
    // SAFETY: on success libmsym returns an array of n subgroups it owns, valid until the group changes.
    unsafe {
        check(msymGetSubgroups(ctx.raw(), &mut n, &mut subgroups), "list the subgroups")?;
        Ok((0..n as usize).map(|i| c_chars_to_string(&(*subgroups.add(i)).name)).collect())
    }
}

fn select_subgroup(ctx: &Context, index: usize) -> Result<(), String> {
    let mut n: c_int = 0;
    let mut subgroups: *const msym_subgroup_t = ptr::null();
    // SAFETY: as in subgroup_names; index is checked against n.
    unsafe {
        check(msymGetSubgroups(ctx.raw(), &mut n, &mut subgroups), "list the subgroups")?;
        if index >= n as usize {
            return Err(format!("subgroup {index} out of range ({n} subgroups)"));
        }
        check(msymSelectSubgroup(ctx.raw(), subgroups.add(index)), "select the subgroup")
    }
}

// The group's standard x, y and z axes in the input's coordinates: the rows of libmsym's rotation from the input's axes
// to the standard orientation.
fn standard_axes(ctx: &Context) -> Result<[[f64; 3]; 3], String> {
    let mut t = [[0.0f64; 3]; 3];
    // SAFETY: t is a double[3][3].
    check(unsafe { msymGetAlignmentTransform(ctx.raw(), t.as_mut_ptr().cast()) }, "get the alignment transform")?;
    Ok(t)
}

fn parallel(a: [f64; 3], b: [f64; 3]) -> bool {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norms = (a.iter().map(|x| x * x).sum::<f64>() * b.iter().map(|x| x * x).sum::<f64>()).sqrt();
    norms > 0.0 && (dot.abs() / norms - 1.0).abs() < AXIS_TOLERANCE
}

// Symmetrizes libmsym's copy of the geometry and fails if any centre had to move more than `tolerance`.
fn check_geometry(ctx: &Context, centres: &[SymmetryCentre], tolerance: f64, group: &str) -> Result<(), String> {
    let mut err = 0.0;
    let mut n: c_int = 0;
    let mut elements: *mut msym_element_t = ptr::null_mut();
    // SAFETY: on success msymGetElements returns libmsym's n external elements, absolute coordinates after
    // msymSymmetrizeElements.
    let symmetric = unsafe {
        check(msymSymmetrizeElements(ctx.raw(), &mut err), "symmetrize the geometry")?;
        check(msymGetElements(ctx.raw(), &mut n, &mut elements), "get the symmetrized geometry")?;
        std::slice::from_raw_parts(elements, n as usize)
    };
    let mut moved = String::new();
    for (i, (centre, element)) in centres.iter().zip(symmetric).enumerate() {
        let d = (0..3).map(|k| (centre.position[k] - element.v[k]).powi(2)).sum::<f64>().sqrt();
        if d > tolerance {
            moved.push_str(&format!("\n  centre {} is {d:.3e} Bohr from its {group}-symmetric position", i + 1));
        }
    }
    if moved.is_empty() {
        Ok(())
    } else {
        Err(format!("the geometry is not exactly {group}-symmetric (tolerance {tolerance:.1e} Bohr):{moved}"))
    }
}

/// What the geometry alone allows, against what the basis does: from `geometric_symmetry`.
#[derive(Debug, Clone)]
pub struct GeometricSymmetry {
    /// The point group of the centres told apart by everything but their basis.
    pub point_group: String,
    /// The group's order.
    pub order: usize,
    /// Pairs of centres (0-based, smaller first) that an operation of that group maps onto each other but whose bases
    /// (or harmonic conventions) differ: what lowers the symmetry below the geometry's.
    pub mismatched: Vec<(usize, usize)>,
}

/// The point group of `centres` with their kinds as given (`centres`: told apart by element, kind and core charge
/// only), and the pairs it maps onto each other whose `full_kinds` (the basis included) differ. For telling a user that
/// the basis, not the geometry, lowered the symmetry.
pub fn geometric_symmetry(centres: &[SymmetryCentre], full_kinds: &[u32]) -> Result<GeometricSymmetry, SymmetryError> {
    // Detection needs a basis; an s function per centre (the group depends on the centres only, except for linear
    // molecules, where it is the highest finite group anyway).
    let basis: Vec<SymmetryBasisFunction> = (0..centres.len()).map(|centre| SymmetryBasisFunction { centre, radial: 1, l: 0, m: 0, cartesian: false }).collect();
    let ctx = detect(centres, &basis)?;
    let point_group = point_group_name(&ctx)?;
    let mut centre_of_mass = [0.0; 3];
    let mut mismatched = Vec::new();
    // SAFETY: libmsym's operations, valid while the group does not change; a double[3] and a double[3][3].
    let order = unsafe {
        check(msymGetCenterOfMass(ctx.raw(), centre_of_mass.as_mut_ptr()), "get the centre of mass")?;
        let mut order: c_int = 0;
        let mut sops: *const msym_symmetry_operation_t = ptr::null();
        check(msymGetSymmetryOperations(ctx.raw(), &mut order, &mut sops), "get the symmetry operations")?;
        for i in 0..order as usize {
            let mut m = [[0.0f64; 3]; 3];
            symmetryOperationMatrix(sops.add(i), m.as_mut_ptr().cast());
            for (a, centre) in centres.iter().enumerate() {
                let p: [f64; 3] = std::array::from_fn(|k| centre.position[k] - centre_of_mass[k]);
                let q: [f64; 3] = std::array::from_fn(|i| centre_of_mass[i] + (0..3).map(|k| m[i][k] * p[k]).sum::<f64>());
                let image = centres
                    .iter()
                    .position(|c| c.kind == centre.kind && vector_math::vector_norm_3d(&vector_math::vector_subtract_3d(&c.position, &q)) < IMAGE_TOLERANCE);
                if let Some(b) = image
                    && full_kinds[a] != full_kinds[b]
                {
                    let pair = (a.min(b), a.max(b));
                    if !mismatched.contains(&pair) {
                        mismatched.push(pair);
                    }
                }
            }
        }
        order as usize
    };
    mismatched.sort();
    Ok(GeometricSymmetry { point_group, order, mismatched })
}

impl Symmetry {
    /// Finds the point group of `centres`, picks the requested group and builds the AO representation of its
    /// operations on `basis`. Fails if any centre is more than `geometry_tolerance` (Bohr) from its symmetric position
    /// in that group.
    pub fn new(
        centres: &[SymmetryCentre],
        basis: &[SymmetryBasisFunction],
        primitive_basis: Option<&[SymmetryBasisFunction]>,
        request: PointGroupRequest<'_>,
        geometry_tolerance: f64,
    ) -> Result<Self, SymmetryError> {
        // libmsym knows only real solid harmonics: a Cartesian shell is given to it as the spherical shell of its l.
        let stand_in_basis: Option<Vec<SymmetryBasisFunction>> = basis.iter().any(|bf| bf.cartesian && bf.l > 1).then(|| {
            let mut spherical = Vec::new();
            for (i, bf) in basis.iter().enumerate() {
                let first_of_block = i == 0 || {
                    let previous = basis[i - 1];
                    (previous.centre, previous.radial, previous.l, previous.cartesian) != (bf.centre, bf.radial, bf.l, bf.cartesian)
                };
                if !(bf.cartesian && bf.l > 1) {
                    spherical.push(*bf);
                } else if first_of_block {
                    spherical.extend((-(bf.l as i32)..=bf.l as i32).map(|m| SymmetryBasisFunction { m, cartesian: false, ..*bf }));
                }
            }
            spherical
        });
        let basis_for_libmsym = stand_in_basis.as_deref().unwrap_or(basis);
        let ctx = detect(centres, basis_for_libmsym)?;
        let full_group = point_group_name(&ctx)?;
        let ctx = match request {
            PointGroupRequest::Highest => ctx,
            PointGroupRequest::Named { name, .. } if name.eq_ignore_ascii_case(&full_group) => ctx,
            PointGroupRequest::Named { name, principal_axis } => {
                let names = subgroup_names(&ctx)?;
                let candidates: Vec<usize> = (0..names.len()).filter(|&i| names[i].eq_ignore_ascii_case(name)).collect();
                if candidates.is_empty() {
                    let mut available = names.clone();
                    available.sort();
                    available.dedup();
                    return Err(SymmetryError(format!(
                        "{name} is not {full_group} or one of its subgroups (available: {full_group}, {})",
                        available.join(", ")
                    )));
                }
                // Selecting a subgroup cannot be undone, so each orientation gets a fresh context. libmsym lists the
                // subgroups in the same order for the same input.
                let mut orientations = Vec::with_capacity(candidates.len());
                for &index in &candidates {
                    let ctx = detect(centres, basis_for_libmsym)?;
                    select_subgroup(&ctx, index)?;
                    let z = standard_axes(&ctx)?[2];
                    orientations.push((ctx, z));
                }
                let wanted = principal_axis.unwrap_or([0.0, 0.0, 1.0]);
                let position = match orientations.iter().position(|(_, z)| parallel(*z, wanted)) {
                    Some(position) => position,
                    None if principal_axis.is_none() && orientations.len() == 1 => 0,
                    None => {
                        let axes: Vec<String> = orientations.iter().map(|(_, z)| format!("{z:.6?}")).collect();
                        return Err(SymmetryError(format!(
                            "{name} is a subgroup of {full_group}, but none of its orientations has its principal axis \
                             along {wanted:?}; choose one by its principal axis: {}",
                            axes.join(", ")
                        )));
                    }
                };
                orientations.swap_remove(position).0
            }
        };
        let point_group = point_group_name(&ctx)?;
        check_geometry(&ctx, centres, geometry_tolerance, &point_group)?;

        // SAFETY: the character table is owned by libmsym and valid while the group does not change (it does not
        // after this point); its arrays have the lengths its d says. Its table is double[irrep][class].
        let (irreps, characters) = unsafe {
            let mut table: *const msym_character_table_t = ptr::null();
            check(msymGetCharacterTable(ctx.raw(), &mut table), "get the character table")?;
            let table = &*table;
            let d = table.d as usize;
            let irreps: Vec<Irrep> = (0..d)
                .map(|i| {
                    let s = &*table.s.add(i);
                    Irrep { name: c_chars_to_string(&s.name), dimension: s.d as usize, reality: s.r.max(1) as usize }
                })
                .collect();
            let rows = std::slice::from_raw_parts(table.table as *const f64, d * d);
            let mut characters = Matrix::<f64>::new(d, d);
            for (irrep, row) in rows.chunks_exact(d).enumerate() {
                for (class, &chi) in row.iter().enumerate() {
                    characters[(irrep, class)] = chi;
                }
            }
            (irreps, characters)
        };
        let n_irreps = irreps.len();
        let totally_symmetric = (0..n_irreps)
            .find(|&k| (0..n_irreps).all(|class| (characters[(k, class)] - 1.0).abs() < 1e-10))
            .ok_or_else(|| SymmetryError(format!("no totally symmetric irrep in the character table of {point_group}")))?;

        // The operations, their 3x3 matrices and classes, and libmsym's real-solid-harmonic representations up to
        // lmax, double[h][2l+1][2l+1] each, row-major, indexed by m + l.
        let lmax = basis.iter().chain(primitive_basis.into_iter().flatten()).map(|bf| bf.l as usize).max().unwrap_or(0);
        // SAFETY: libmsym returns its order-long array of operations, valid while the group does not change.
        // generateRSHRepresentations only reads them, and fills the buffers allocated here.
        let (operations, rsh) = unsafe {
            let mut order: c_int = 0;
            let mut sops: *const msym_symmetry_operation_t = ptr::null();
            check(msymGetSymmetryOperations(ctx.raw(), &mut order, &mut sops), "get the symmetry operations")?;
            let h = order as usize;
            let mut rsh: Vec<Box<[f64]>> = (0..=lmax).map(|l| vec![0.0; h * (2 * l + 1) * (2 * l + 1)].into_boxed_slice()).collect();
            let mut lrs: Vec<rsh_representations_t> =
                rsh.iter_mut().enumerate().map(|(l, t)| rsh_representations_t { d: (2 * l + 1) as c_int, t: t.as_mut_ptr().cast() }).collect();
            check(
                generateRSHRepresentations(order, sops as *mut msym_symmetry_operation_t, lmax as c_int, lrs.as_mut_ptr()),
                "generate the real solid harmonic representations",
            )?;
            let operations: Vec<([[f64; 3]; 3], usize)> = (0..h)
                .map(|i| {
                    let sop = sops.add(i);
                    let mut m = [[0.0f64; 3]; 3];
                    symmetryOperationMatrix(sop, m.as_mut_ptr().cast());
                    (m, (*sop).cla as usize)
                })
                .collect();
            (operations, rsh)
        };
        let h = operations.len();
        let mut centre_of_mass = [0.0; 3];
        // SAFETY: a double[3].
        check(unsafe { msymGetCenterOfMass(ctx.raw(), centre_of_mass.as_mut_ptr()) }, "get the centre of mass")?;

        // Where each operation takes each centre: R (p - cm) + cm, matched among the centres of the same kind.
        let mut centre_images = Matrix::<usize>::new(centres.len(), h);
        for (op, ((m, _), column)) in operations.iter().zip(centre_images.iter_mut()).enumerate() {
            for (a, (image, centre)) in column.iter_mut().zip(centres).enumerate() {
                let p: [f64; 3] = std::array::from_fn(|k| centre.position[k] - centre_of_mass[k]);
                let q: [f64; 3] = std::array::from_fn(|i| centre_of_mass[i] + (0..3).map(|k| m[i][k] * p[k]).sum::<f64>());
                *image = centres
                    .iter()
                    .position(|c| c.kind == centre.kind && vector_math::vector_norm_3d(&vector_math::vector_subtract_3d(&c.position, &q)) < IMAGE_TOLERANCE)
                    .ok_or_else(|| SymmetryError(format!("operation {op} maps centre {} onto no equivalent centre", a + 1)))?;
            }
        }
        // The blocks and their images in each basis, and the component order of each l (m + l per position), the same
        // in every block of either basis.
        let mut m_order: Vec<Option<Vec<usize>>> = vec![None; 2 * (lmax + 1)];
        let ao = AoRepresentation::new(basis, &centre_images, &mut m_order)?;
        let primitive = primitive_basis.map(|primitive| AoRepresentation::new(primitive, &centre_images, &mut m_order)).transpose()?;
        let stand_in = stand_in_basis.as_deref().map(|stand_in| AoRepresentation::new(stand_in, &centre_images, &mut m_order)).transpose()?;

        // Per block shape, D(op)'s blocks. Spherical: libmsym's t[op][m1 + l][m2 + l], permuted to the AO component
        // order. Cartesian: O_R χ_a = N_a monomial_a(R^T r) g = Σ_b (N_a / N_b) T(R)_ba χ_b, the monomial
        // transformation between components normalized each on its own.
        let shapes: Vec<Shape> = (0..2 * (lmax + 1))
            .map(|shape| {
                let (l, d) = (shape / 2, shape_len(shape));
                let mut rotations = Array3D::<f64>::new(d, d, h);
                if shape % 2 == 0 {
                    if let Some(order) = &m_order[shape] {
                        let t = &rsh[l];
                        for (mut slab, t_op) in rotations.iter_mut().zip(t.chunks_exact(d * d)) {
                            for (j, column) in slab.iter_col_mut().enumerate() {
                                for (i, value) in column.iter_mut().enumerate() {
                                    *value = t_op[order[i] * d + order[j]];
                                }
                            }
                        }
                    }
                } else if m_order[shape].is_some() {
                    let mut transforms = Matrix::<f64>::new(d, d * h);
                    let mut polynomial = vec![0.0; 2 * d];
                    monomial_transformations(&operations, l, TensorKind::Polar, &mut transforms.view_mut(), &mut polynomial);
                    let mut norms = Vec::with_capacity(d);
                    for lx in (0..=l).rev() {
                        for ly in (0..=l - lx).rev() {
                            norms.push(cartesian_norm(lx, ly, l - lx - ly));
                        }
                    }
                    for (op, mut slab) in rotations.iter_mut().enumerate() {
                        for (a, column) in slab.iter_col_mut().enumerate() {
                            for (b, value) in column.iter_mut().enumerate() {
                                *value = norms[a] / norms[b] * transforms[(b, op * d + a)];
                            }
                        }
                    }
                }
                let mut inverses = Array3D::<f64>::new(d, d, h);
                for (slab, mut out) in rotations.iter().zip(inverses.iter_mut()) {
                    if shape % 2 == 0 {
                        slab.transpose_prealloc(&mut out);
                    } else if m_order[shape].is_some() {
                        out.get_data_mut().copy_from_slice(slab.invert().get_data());
                    }
                }
                let metric = (shape % 2 == 1 && m_order[shape].is_some()).then(|| {
                    let mut w = Matrix::<f64>::new(d, d);
                    for slab in rotations.iter() {
                        slab.view_transposed().matmul_scaled_prealloc(&slab, &mut w, 1.0 / h as f64, 1.0);
                    }
                    w
                });
                Shape { permutations: BlockPermutations::of(&rotations), rotations, inverses, metric }
            })
            .collect();
        let max_block = std::iter::once(&ao).chain(&primitive).chain(&stand_in).flat_map(|rep| &rep.blocks).map(|b| shape_len(b.shape)).max().unwrap_or(1);

        let mut projector_weights = Matrix::<f64>::new(h, n_irreps);
        for (k, column) in projector_weights.iter_mut().enumerate() {
            let factor = irreps[k].dimension as f64 / (h as f64 * irreps[k].reality as f64);
            for (weight, &(_, class)) in column.iter_mut().zip(&operations) {
                *weight = factor * characters[(k, class)];
            }
        }

        let mut symmetry = Symmetry {
            point_group,
            irreps,
            characters,
            totally_symmetric,
            operations,
            centre_images,
            standard_axes: standard_axes(&ctx)?,
            ao,
            primitive,
            shapes,
            stand_in,
            max_block,
            projector_weights,
            transfer_weights: Vec::new(),
            ctx,
        };
        symmetry.ao.ao_irrep_counts = (0..n_irreps).map(|k| symmetry.count_irrep(&symmetry.ao, k)).collect::<Result<_, _>>()?;
        if let Some(primitive) = &symmetry.primitive {
            let counts = (0..n_irreps).map(|k| symmetry.count_irrep(primitive, k)).collect::<Result<_, _>>()?;
            symmetry.primitive.as_mut().expect("checked").ao_irrep_counts = counts;
        }
        symmetry.transfer_weights = symmetry.transfer_weights_from_salcs()?;
        Ok(symmetry)
    }

    // D(op)'s block for `block`, (2l+1, 2l+1).
    fn rotation(&self, op: usize, block: &AngularBlock) -> MatrixView<'_, f64> {
        self.shapes[block.shape].rotations.view_sub_matrix(op)
    }

    fn representation(&self, basis: AoBasis) -> Result<&AoRepresentation, SymmetryError> {
        match basis {
            AoBasis::Contracted => Ok(&self.ao),
            AoBasis::Primitive => self.primitive.as_ref().ok_or_else(|| SymmetryError("no primitive basis was given to Symmetry::new".to_string())),
        }
    }

    // The number of AOs of irrep k: the trace of its projector, Σ_R w_k(R) tr D(R) (only blocks an operation maps onto
    // themselves contribute).
    fn count_irrep(&self, rep: &AoRepresentation, k: usize) -> Result<usize, SymmetryError> {
        let mut count = 0.0;
        for (op, &weight) in self.projector_weights.get_col(k).iter().enumerate() {
            let trace: f64 = (0..rep.blocks.len())
                .filter(|&b| rep.images[(b, op)] == b)
                .map(|b| {
                    let rotation = self.rotation(op, &rep.blocks[b]);
                    (0..rotation.rows()).map(|i| rotation[(i, i)]).sum::<f64>()
                })
                .sum();
            count += weight * trace;
        }
        if (count - count.round()).abs() > 1e-6 || count < -0.5 {
            return Err(SymmetryError(format!(
                "internal: the AO representation of {} has {count} functions of {}, not an integer",
                self.point_group, self.irreps[k].name
            )));
        }
        Ok(count.round() as usize)
    }

    // libmsym's SALCs, read once: the first partner set of each irrep present checks that every D(R) maps it into its
    // own span (that D(R) is the representation libmsym's SALCs transform under), and gives the irrep matrices
    // Γ(R)_jk = s_j^T D(R) s_k of the degenerate, absolutely irreducible irreps, as transfer-operator weights.
    // One-shot setup: the n^2 SALC matrix lives only for this call.
    fn transfer_weights_from_salcs(&self) -> Result<Vec<Option<Matrix<f64>>>, SymmetryError> {
        // libmsym's basis: the stand-in when the basis has Cartesian shells (Γ(R) depends only on the group).
        let rep = self.stand_in.as_ref().unwrap_or(&self.ao);
        let n = rep.n_ao;
        let h = self.operations.len();
        let mut salcs = Matrix::<f64>::new(n, n);
        let mut species = vec![0 as c_int; n];
        let mut partners = vec![msym_partner_function_t::default(); n];
        // libmsym cannot build SALCs in a group with complex characters (Cn and Cnh for n >= 3, Sn, T, Th: "cannot
        // construct splitting field"). Nothing but the partner alignment needs them: there, no irrep matrices, so no
        // check of D(R) against libmsym's SALCs and no partner alignment (impossible for the real pairs of complex
        // irreps anyway; lost for T and Th's three-dimensional irreps). SAFETY: a call on the context.
        let generated = unsafe { msymGenerateSubrepresentationSpaces(self.ctx.raw()) };
        if generated != MSYM_SUCCESS && self.irreps.iter().any(|irrep| irrep.reality > 1) {
            return Ok(vec![None; self.irreps.len()]);
        }
        check(generated, "generate the SALCs")?;
        // SAFETY: libmsym fills double[n][n] row by row, one SALC per row: exactly column-major (n_ao, n_salc).
        check(
            unsafe { msymGetSALCs(self.ctx.raw(), n as c_int, salcs.get_data_mut().as_mut_ptr(), species.as_mut_ptr(), partners.as_mut_ptr()) },
            "get the SALCs",
        )?;
        let mut weights = Vec::with_capacity(self.irreps.len());
        let mut image = vec![0.0; n];
        for (k, irrep) in self.irreps.iter().enumerate() {
            let Some(first) = (0..n).find(|&i| species[i] as usize == k && partners[i].d == 0) else {
                weights.push(None);
                continue;
            };
            let d = irrep.dimension;
            let set = salcs.view_columns(first..first + d);
            let mut transfer = Matrix::<f64>::new(h, d * d);
            for op in 0..h {
                for (kk, s_k) in set.iter_col().enumerate() {
                    self.apply_to_vector(rep, op, s_k, &mut image);
                    // Γ(R)_jk, and what of D(R) s_k lies outside the set's span.
                    for (j, s_j) in set.iter_col().enumerate() {
                        let gamma = vector_math::dot(s_j, &image);
                        transfer[(op, j + d * kk)] = d as f64 / h as f64 * gamma;
                        vector_math::add_scaled_in_place(&mut image, s_j, -gamma);
                    }
                    let off = vector_math::norm(&image);
                    if off > REPRESENTATION_TOLERANCE {
                        return Err(SymmetryError(format!(
                            "internal: operation {op} of {} takes a SALC of {} out of its partner space by {off:.3e}",
                            self.point_group, irrep.name
                        )));
                    }
                }
            }
            weights.push((d > 1 && irrep.reality == 1).then_some(transfer));
        }
        Ok(weights)
    }

    // <a, b> in the invariant metric W = (1/h) Σ_R D(R)^T D(R) of the contracted basis: block-diagonal, the identity on
    // spherical blocks (D(R) is orthogonal there), the shape's metric block on Cartesian ones. Transfer and projection
    // operators are isometries and self-adjoint in it, so it replaces the coefficients' Euclidean metric wherever the
    // orbitals are measured.
    fn metric_dot(&self, a: &[f64], b: &[f64]) -> f64 {
        self.ao
            .blocks
            .iter()
            .map(|block| {
                let (a, b) = (&a[block.range()], &b[block.range()]);
                match &self.shapes[block.shape].metric {
                    None => vector_math::dot(a, b),
                    Some(w) => w.iter_col().zip(b).map(|(column, &b_j)| b_j * vector_math::dot(column, a)).sum(),
                }
            })
            .sum()
    }

    // Whether every block of every basis is orthogonal (no Cartesian shells).
    fn orthogonal(&self) -> bool {
        self.shapes.iter().all(|shape| shape.metric.is_none())
    }

    // out = D(op) c: block b of c, rotated, into the image block.
    fn apply_to_vector(&self, rep: &AoRepresentation, op: usize, c: &[f64], out: &mut [f64]) {
        for (b, block) in rep.blocks.iter().enumerate() {
            self.rotation(op, block).matvec_prealloc(&c[block.range()], &mut out[rep.image(op, b).range()]);
        }
    }

    // out[out_at] = alpha r1^T a[a_at] r2 + beta out[out_at], r1 and r2 D(op)'s blocks for the l of `b1` and `b2`
    // (transposed: their transposes, which map an image back), the signed-permutation kernel when both are
    // permutations.
    #[allow(clippy::too_many_arguments)]
    fn transform_tile<A: MatrixTrait<f64>, O: MatrixTraitMut<f64>>(
        &self,
        op: usize,
        (s1, s2): (usize, usize),
        inverse: bool,
        a: &A,
        a_at: (usize, usize),
        out: &mut O,
        out_at: (usize, usize),
        alpha: f64,
        beta: f64,
        scratch: &mut [f64],
    ) {
        let (p1, p2) = (&self.shapes[s1].permutations, &self.shapes[s2].permutations);
        if p1.exact[op] && p2.exact[op] {
            // A signed permutation's inverse is its transpose.
            block_signed_permutation_prealloc(p1.factor(op, inverse), a, a_at, p2.factor(op, inverse), out, out_at, alpha, beta);
        } else {
            let (left, right) = (self.shapes[s1].block(op, inverse), self.shapes[s2].block(op, inverse));
            block_sandwich_prealloc(&left, a, a_at, &right, out, out_at, alpha, beta, scratch);
        }
    }

    /// The point group in use, e.g. "C2v".
    pub fn point_group(&self) -> &str {
        &self.point_group
    }

    /// The group's standard x, y and z axes (rows) in the input's coordinates: what the irrep names refer to (e.g.
    /// which of B1/B2 is symmetric under the reflection in the xz plane). The identity when the molecule is in the
    /// group's standard orientation.
    pub fn standard_axes(&self) -> [[f64; 3]; 3] {
        self.standard_axes
    }

    /// Its irreps, in character-table order (the order of every irrep index in this crate).
    pub fn irreps(&self) -> &[Irrep] {
        &self.irreps
    }

    /// How many AOs (symmetry-adapted functions) each irrep has; MOs of a complete set come in the same numbers.
    pub fn ao_irrep_counts(&self) -> &[usize] {
        &self.ao.ao_irrep_counts
    }

    /// The index of the irrep called `name` (e.g. "B1", "A''", case-insensitive).
    pub fn irrep_index(&self, name: &str) -> Option<usize> {
        self.irreps.iter().position(|irrep| irrep.name.eq_ignore_ascii_case(name))
    }

    /// The index of the totally symmetric irrep.
    pub fn totally_symmetric_irrep(&self) -> usize {
        self.totally_symmetric
    }

    /// The irrep of the Cartesian monomial x^lx y^ly z^lz about the centre of mass, and so of the AO matrix of that
    /// operator (a dipole or quadrupole component about the centre of mass, ...): what to pass as
    /// `OperatorSymmetry::Irrep`. `None` if it is not a one-dimensional irrep by itself: in a degenerate irrep (x and y
    /// in C3v belong to E together), or mixed because the molecule is not in the group's standard orientation (see
    /// `standard_axes`).
    pub fn cartesian_irrep(&self, lx: u32, ly: u32, lz: u32) -> Option<usize> {
        let monomial = |p: [f64; 3]| p[0].powi(lx as i32) * p[1].powi(ly as i32) * p[2].powi(lz as i32);
        // Generic points: no monomial vanishes at any of them.
        const POINTS: [[f64; 3]; 3] = [[0.31, 0.57, 0.83], [-0.74, 0.29, 0.46], [0.52, -0.66, 0.37]];
        let d = self.irreps.len();
        // The character of each operation R on the monomial f: (R f)(r) = f(R^T r) = χ(R) f(r) at every point, or none.
        let mut characters = vec![f64::NAN; d];
        for (m, class) in &self.operations {
            let rotated = |p: [f64; 3]| monomial([0, 1, 2].map(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2]));
            let chi = rotated(POINTS[0]) / monomial(POINTS[0]);
            let scale = POINTS.iter().map(|&p| monomial(p).abs()).fold(0.0, f64::max);
            if POINTS.iter().any(|&p| (rotated(p) - chi * monomial(p)).abs() > 1e-10 * scale) {
                return None;
            }
            characters[*class] = chi;
        }
        (0..d).find(|&k| self.irreps[k].dimension == 1 && (0..d).all(|c| (self.characters[(k, c)] - characters[c]).abs() < 1e-10))
    }

    /// Number of AOs of the contracted basis (the orbitals' basis).
    pub fn n_ao(&self) -> usize {
        self.ao.n_ao
    }

    /// Number of AOs of `basis`, `None` for a primitive basis that was not given to `new`.
    pub fn n_ao_of(&self, basis: AoBasis) -> Option<usize> {
        self.representation(basis).ok().map(|rep| rep.n_ao)
    }

    /// Number of symmetry operations.
    pub fn order(&self) -> usize {
        self.operations.len()
    }

    /// Heap bytes held (the operations' AO representation and the tables; libmsym's own context is not counted).
    pub fn approx_memory_bytes(&self) -> usize {
        let f64_bytes = std::mem::size_of::<f64>();
        self.shapes
            .iter()
            .map(|shape| {
                2 * shape.rotations.get_data().len() * f64_bytes
                    + shape.permutations.rows.get_data().len() * 2 * (f64_bytes + std::mem::size_of::<usize>())
                    + shape.metric.as_ref().map_or(0, |w| w.get_data().len() * f64_bytes)
            })
            .sum::<usize>()
            + std::iter::once(&self.ao).chain(&self.primitive).map(|rep| rep.images.get_data().len() * std::mem::size_of::<usize>()).sum::<usize>()
            + (self.characters.get_data().len() + self.projector_weights.get_data().len()) * f64_bytes
            + self.transfer_weights.iter().flatten().map(|t| t.get_data().len() * f64_bytes).sum::<usize>()
    }

    /// f64s of scratch the matrix methods (`symmetrise_matrix`, `average_matrix`, `analyse_matrix`, `irrep_norms`) take
    /// from their arena: a few tiles, whatever n_ao (plus 2 h usize for the orbit's members).
    pub fn matrix_scratch_len(&self) -> usize {
        let h = self.operations.len();
        (h + 1) * self.max_block * self.max_block + self.max_block + h + self.irreps.len()
    }

    fn check_square(&self, what: &str, basis: AoBasis, rows: usize, cols: usize) -> Result<&AoRepresentation, SymmetryError> {
        let rep = self.representation(basis)?;
        let n = rep.n_ao;
        if (rows, cols) != (n, n) {
            return Err(SymmetryError(format!("{what}: the matrix is {rows}x{cols}, the {basis:?} basis has {n} AOs")));
        }
        Ok(rep)
    }

    /// The norm of the part of `matrix` (`(n_ao, n_ao)`) in each irrep, into `norms_out` (one per irrep): how an
    /// unknown operator transforms. For a degenerate irrep, the part in that irrep as a whole. A diagnostic: h tile
    /// transforms per block pair.
    pub fn irrep_norms<M>(&self, basis: AoBasis, matrix: &M, norms_out: &mut [f64], arena: &Arena<'_>) -> Result<(), SymmetryError>
    where
        M: MatrixTrait<f64>,
    {
        let rep = self.check_square("irrep_norms", basis, matrix.rows(), matrix.cols())?;
        let d = self.irreps.len();
        if norms_out.len() != d {
            return Err(SymmetryError(format!("irrep_norms: {} has {d} irreps, got {} slots", self.point_group, norms_out.len())));
        }
        self.irrep_norms_into(rep, &matrix.view(), norms_out, arena)
    }

    // |P_Γ M|^2 = <M, P_Γ M> (the projector is self-adjoint) = Σ_R w_Γ(R) <M, D(R)^T M D(R)>: the overlaps of every
    // operation, block pair by block pair, then one product with the projector weights.
    fn irrep_norms_into(&self, rep: &AoRepresentation, matrix: &MatrixView<'_, f64>, norms_out: &mut [f64], arena: &Arena<'_>) -> Result<(), SymmetryError> {
        if !self.orthogonal() {
            return self.irrep_norms_explicit(rep, matrix, norms_out, arena);
        }
        let overlaps = arena.alloc_zeroed::<f64>(self.operations.len())?;
        let tile_data = arena.alloc::<f64>(self.max_block * self.max_block)?;
        let scratch = arena.alloc::<f64>(self.max_block)?;
        for (b2, block2) in rep.blocks.iter().enumerate() {
            for (b1, block1) in rep.blocks.iter().enumerate() {
                let (d1, d2) = (block1.range().len(), block2.range().len());
                for (op, overlap) in overlaps.iter_mut().enumerate() {
                    let mut tile = MatrixViewMut::new(d1, d2, &mut tile_data[..d1 * d2]);
                    let at = (rep.image(op, b1).offset, rep.image(op, b2).offset);
                    self.transform_tile(op, (block1.shape, block2.shape), false, matrix, at, &mut tile, (0, 0), 1.0, 0.0, scratch);
                    *overlap += tile
                        .iter_col()
                        .enumerate()
                        .map(|(j, column)| vector_math::dot(column, &matrix.get_col(block2.offset + j)[block1.range()]))
                        .sum::<f64>();
                }
            }
        }
        self.projector_weights.view_transposed().matvec_prealloc(&*overlaps, norms_out);
        norms_out.iter_mut().for_each(|norm| *norm = norm.max(0.0).sqrt());
        Ok(())
    }

    // The norms of the irreps' parts each from its projection, tile by tile (|P_Γ M|^2 summed over the block pairs):
    // for a basis with Cartesian shells, whose D(R) are not orthogonal, so the projector is not self-adjoint in the
    // Frobenius metric. h tile transforms per block pair and irrep.
    fn irrep_norms_explicit(&self, rep: &AoRepresentation, matrix: &MatrixView<'_, f64>, norms_out: &mut [f64], arena: &Arena<'_>) -> Result<(), SymmetryError> {
        let tile_data = arena.alloc::<f64>(self.max_block * self.max_block)?;
        let scratch = arena.alloc::<f64>(self.max_block)?;
        norms_out.fill(0.0);
        for (b2, block2) in rep.blocks.iter().enumerate() {
            for (b1, block1) in rep.blocks.iter().enumerate() {
                let (d1, d2) = (block1.range().len(), block2.range().len());
                for (norm, weights) in norms_out.iter_mut().zip(self.projector_weights.iter_col()) {
                    let mut tile = MatrixViewMut::new(d1, d2, &mut tile_data[..d1 * d2]);
                    tile.get_data_mut().fill(0.0);
                    for (op, &weight) in weights.iter().enumerate().filter(|&(_, &w)| w != 0.0) {
                        let at = (rep.image(op, b1).offset, rep.image(op, b2).offset);
                        self.transform_tile(op, (block1.shape, block2.shape), false, matrix, at, &mut tile, (0, 0), weight, 1.0, scratch);
                    }
                    *norm += vector_math::dot(tile.get_data(), tile.get_data());
                }
            }
        }
        norms_out.iter_mut().for_each(|norm| *norm = norm.sqrt());
        Ok(())
    }

    fn resolve(&self, rep: &AoRepresentation, matrix: &MatrixView<'_, f64>, operator: OperatorSymmetry, arena: &Arena<'_>) -> Result<usize, SymmetryError> {
        let d = self.irreps.len();
        Ok(match operator {
            OperatorSymmetry::TotallySymmetric => self.totally_symmetric,
            OperatorSymmetry::Irrep(k) if k >= d => {
                return Err(SymmetryError(format!("irrep {k} out of range, {} has {d}", self.point_group)));
            }
            OperatorSymmetry::Irrep(k) => k,
            OperatorSymmetry::Auto => {
                let norms = arena.alloc::<f64>(d)?;
                self.irrep_norms_into(rep, matrix, norms, arena)?;
                (0..d).max_by(|&a, &b| norms[a].total_cmp(&norms[b])).unwrap_or(self.totally_symmetric)
            }
        })
    }

    // One pass over the orbits: projects every orbit of block pairs onto `irrep`, writes it (when `tiles` is
    // writable) and reports what the projection removes. A one-dimensional irrep (the totally symmetric one among
    // them): the representative's projection G = Σ_R w(R) r1^T M[R b1, R b2] r2, and each member's block the
    // representative's carried over, χ(R) r1 G r2^T. A degenerate one: every member's projection directly, all before
    // any is written.
    fn project_by_orbits<T: Tiles>(&self, rep: &AoRepresentation, tiles: &mut T, irrep: usize, arena: &Arena<'_>) -> Result<SymmetrisationReport, SymmetryError> {
        let h = self.operations.len();
        let tile_len = self.max_block * self.max_block;
        let projection = arena.alloc::<f64>(tile_len)?;
        let carried = arena.alloc::<f64>(h * tile_len)?;
        let scratch = arena.alloc::<f64>(self.max_block)?;
        // The orbit's distinct members, as (b1, b2).
        let members = arena.alloc::<(usize, usize)>(h)?;
        let one_dimensional = self.irreps[irrep].dimension == 1;
        let weights = self.projector_weights.get_col(irrep);

        let mut report = SymmetrisationReport { irrep, max_abs_removed: 0.0, removed_norm: 0.0, norm: vector_math::norm(tiles.read().get_data()) };
        let mut removed_squared = 0.0;
        for (b2, block2) in rep.blocks.iter().enumerate() {
            for (b1, block1) in rep.blocks.iter().enumerate() {
                if !rep.is_representative(b1, b2) {
                    continue;
                }
                let (d1, d2) = (block1.range().len(), block2.range().len());
                let ls = (block1.shape, block2.shape);
                let mut n_members = 0;
                for op in 0..h {
                    let member = (rep.images[(b1, op)], rep.images[(b2, op)]);
                    if !members[..n_members].contains(&member) {
                        members[n_members] = member;
                        n_members += 1;
                    }
                }
                // The new block of every member, into `carried`, before any is written.
                if one_dimensional {
                    let mut g = MatrixViewMut::new(d1, d2, &mut projection[..d1 * d2]);
                    g.get_data_mut().fill(0.0);
                    for (op, &weight) in weights.iter().enumerate().filter(|&(_, &w)| w != 0.0) {
                        let at = (rep.image(op, b1).offset, rep.image(op, b2).offset);
                        self.transform_tile(op, ls, false, &tiles.read(), at, &mut g, (0, 0), weight, 1.0, scratch);
                    }
                    for (m, member) in members[..n_members].iter().enumerate() {
                        // An operation carrying the representative onto this member, and its character.
                        let op = (0..h).find(|&op| (rep.images[(b1, op)], rep.images[(b2, op)]) == *member).expect("a member is an image");
                        let chi = self.characters[(irrep, self.operations[op].1)];
                        let mut out = MatrixViewMut::new(d1, d2, &mut carried[m * tile_len..m * tile_len + d1 * d2]);
                        self.transform_tile(op, ls, true, &g.view(), (0, 0), &mut out, (0, 0), chi, 0.0, scratch);
                    }
                } else {
                    for (m, &(m1, m2)) in members[..n_members].iter().enumerate() {
                        let mut out = MatrixViewMut::new(d1, d2, &mut carried[m * tile_len..m * tile_len + d1 * d2]);
                        out.get_data_mut().fill(0.0);
                        for (op, &weight) in weights.iter().enumerate().filter(|&(_, &w)| w != 0.0) {
                            let at = (rep.image(op, m1).offset, rep.image(op, m2).offset);
                            self.transform_tile(op, ls, false, &tiles.read(), at, &mut out, (0, 0), weight, 1.0, scratch);
                        }
                    }
                }
                // What changes, then the write.
                for (m, &(m1, m2)) in members[..n_members].iter().enumerate() {
                    let new = MatrixView::new(d1, d2, &carried[m * tile_len..m * tile_len + d1 * d2]);
                    let (r1, r2) = (rep.blocks[m1].range(), rep.blocks[m2].offset);
                    let old = tiles.read();
                    for (j, column) in new.iter_col().enumerate() {
                        for (&x_new, &x_old) in column.iter().zip(&old.get_col(r2 + j)[r1.clone()]) {
                            let removed = (x_old - x_new).abs();
                            report.max_abs_removed = report.max_abs_removed.max(removed);
                            removed_squared += removed * removed;
                        }
                    }
                    tiles.write((r1.start, r2), &new);
                }
            }
        }
        report.removed_norm = removed_squared.sqrt();
        Ok(report)
    }

    /// What `symmetrise_matrix` would remove from `matrix`, without changing it.
    pub fn analyse_matrix<M>(&self, basis: AoBasis, matrix: &M, operator: OperatorSymmetry, arena: &Arena<'_>) -> Result<SymmetrisationReport, SymmetryError>
    where
        M: MatrixTrait<f64>,
    {
        let rep = self.check_square("analyse_matrix", basis, matrix.rows(), matrix.cols())?;
        let irrep = self.resolve(rep, &matrix.view(), operator, arena)?;
        self.project_by_orbits(rep, &mut ReadOnly(matrix.view()), irrep, arena)
    }

    /// Removes from the AO matrix `matrix` (`(n_ao, n_ao)`, any matrix, symmetric or not) everything that does not
    /// transform as `operator` says, and reports what was removed. Refuses, leaving `matrix` unchanged, if a single
    /// element would change by more than `tolerance`: then the asymmetry is not noise, but something (often the
    /// environment, or a field) breaks the symmetry. Two passes: the report, then the write.
    ///
    /// The cleaned matrix is the projection onto the irrep, `(d/(h r)) Σ_R χ(R) D(R)^T M D(R)`; for a totally
    /// symmetric matrix the group average `(1/h) Σ_R D(R)^T M D(R)`.
    pub fn symmetrise_matrix<M>(
        &self,
        basis: AoBasis,
        matrix: &mut M,
        operator: OperatorSymmetry,
        tolerance: f64,
        arena: &Arena<'_>,
    ) -> Result<SymmetrisationReport, SymmetryError>
    where
        M: MatrixTraitMut<f64>,
    {
        let rep = self.check_square("symmetrise_matrix", basis, matrix.rows(), matrix.cols())?;
        let irrep = self.resolve(rep, &matrix.view(), operator, arena)?;
        let report = self.project_by_orbits(rep, &mut ReadOnly(matrix.view()), irrep, arena)?;
        if report.max_abs_removed > tolerance {
            return Err(SymmetryError(format!(
                "symmetrise_matrix: cleaning the matrix as {} of {} would change an element by {:.3e} (tolerance {tolerance:.1e}; \
                 removed norm {:.3e} of {:.3e}). Something breaks the symmetry",
                self.irreps[report.irrep].name, self.point_group, report.max_abs_removed, report.removed_norm, report.norm
            )));
        }
        self.project_by_orbits(rep, &mut Writable(&mut matrix.view_mut()), irrep, arena)
    }

    /// `symmetrise_matrix` without a tolerance: projects (for a totally symmetric matrix, averages over the group) in
    /// one pass and reports what it removed. For an operator whose asymmetry is physics to be averaged out (an
    /// embedding whose far tail is not symmetric), not an error.
    pub fn average_matrix<M>(&self, basis: AoBasis, matrix: &mut M, operator: OperatorSymmetry, arena: &Arena<'_>) -> Result<SymmetrisationReport, SymmetryError>
    where
        M: MatrixTraitMut<f64>,
    {
        let rep = self.check_square("average_matrix", basis, matrix.rows(), matrix.cols())?;
        let irrep = self.resolve(rep, &matrix.view(), operator, arena)?;
        self.project_by_orbits(rep, &mut Writable(&mut matrix.view_mut()), irrep, arena)
    }

    /// Whether `point` (Bohr) is a fixed point of every operation (the centre of mass is): the origin a multipole must
    /// be taken about for its components to transform among themselves.
    pub fn is_fixed_point(&self, point: [f64; 3]) -> Result<bool, SymmetryError> {
        let centre = self.centre_of_mass()?;
        let p: [f64; 3] = std::array::from_fn(|k| point[k] - centre[k]);
        let scale = vector_math::vector_norm_3d(&p).max(1.0);
        Ok(self.operations.iter().all(|(m, _)| {
            let image: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| m[i][k] * p[k]).sum::<f64>());
            vector_math::vector_norm_3d(&vector_math::vector_subtract_3d(&image, &p)) < 1e-8 * scale
        }))
    }

    // One pass over the orbits for a Cartesian tensor: every member's every component computed directly,
    // (1/h) Σ_R Σ_b T(R)_ba r1^T M_b[R m1, R m2] r2, before any is written. The projection onto the tensors that
    // transform as their components should (a multipole about a fixed point is one already, up to rounding).
    fn project_tensor_by_orbits<T: PlaneTiles>(
        &self,
        rep: &AoRepresentation,
        tiles: &mut T,
        order: usize,
        kind: TensorKind,
        arena: &Arena<'_>,
    ) -> Result<SymmetrisationReport, SymmetryError> {
        let h = self.operations.len();
        let n_comp = (order + 1) * (order + 2) / 2;
        let tile_len = self.max_block * self.max_block;
        let mut transforms = MatrixViewMut::<f64>::from_arena(n_comp, n_comp * h, arena)?;
        let polynomial = arena.alloc::<f64>(2 * n_comp)?;
        monomial_transformations(&self.operations, order, kind, &mut transforms, polynomial);
        let image = arena.alloc::<f64>(tile_len)?;
        let carried = arena.alloc::<f64>(h * n_comp * tile_len)?;
        let scratch = arena.alloc::<f64>(self.max_block)?;
        let members = arena.alloc::<(usize, usize)>(h)?;

        let norm = (0..n_comp).map(|c| vector_math::dot(tiles.read(c).get_data(), tiles.read(c).get_data())).sum::<f64>().sqrt();
        let mut report = SymmetrisationReport { irrep: self.totally_symmetric, max_abs_removed: 0.0, removed_norm: 0.0, norm };
        let mut removed_squared = 0.0;
        for (b2, block2) in rep.blocks.iter().enumerate() {
            for (b1, block1) in rep.blocks.iter().enumerate() {
                if !rep.is_representative(b1, b2) {
                    continue;
                }
                let (d1, d2) = (block1.range().len(), block2.range().len());
                let ls = (block1.shape, block2.shape);
                let mut n_members = 0;
                for op in 0..h {
                    let member = (rep.images[(b1, op)], rep.images[(b2, op)]);
                    if !members[..n_members].contains(&member) {
                        members[n_members] = member;
                        n_members += 1;
                    }
                }
                carried[..n_members * n_comp * tile_len].fill(0.0);
                for (m, &(m1, m2)) in members[..n_members].iter().enumerate() {
                    for op in 0..h {
                        let at = (rep.image(op, m1).offset, rep.image(op, m2).offset);
                        let transform = transforms.view_columns(op * n_comp..(op + 1) * n_comp);
                        for b in 0..n_comp {
                            let mut t = MatrixViewMut::new(d1, d2, &mut image[..d1 * d2]);
                            self.transform_tile(op, ls, false, &tiles.read(b), at, &mut t, (0, 0), 1.0 / h as f64, 0.0, scratch);
                            for (a, column) in transform.iter_col().enumerate() {
                                let weight = column[b];
                                if weight != 0.0 {
                                    let start = (m * n_comp + a) * tile_len;
                                    vector_math::add_scaled_in_place(&mut carried[start..start + d1 * d2], &image[..d1 * d2], weight);
                                }
                            }
                        }
                    }
                }
                for (m, &(m1, m2)) in members[..n_members].iter().enumerate() {
                    let (r1, r2) = (rep.blocks[m1].range(), rep.blocks[m2].offset);
                    for a in 0..n_comp {
                        let start = (m * n_comp + a) * tile_len;
                        let new = MatrixView::new(d1, d2, &carried[start..start + d1 * d2]);
                        let old = tiles.read(a);
                        for (j, column) in new.iter_col().enumerate() {
                            for (&x_new, &x_old) in column.iter().zip(&old.get_col(r2 + j)[r1.clone()]) {
                                let removed = (x_old - x_new).abs();
                                report.max_abs_removed = report.max_abs_removed.max(removed);
                                removed_squared += removed * removed;
                            }
                        }
                        tiles.write(a, (r1.start, r2), &new);
                    }
                }
            }
        }
        report.removed_norm = removed_squared.sqrt();
        Ok(report)
    }

    fn check_tensor<'s>(&'s self, what: &str, basis: AoBasis, dims: [usize; 3], order: usize) -> Result<&'s AoRepresentation, SymmetryError> {
        let rep = self.representation(basis)?;
        let n_comp = (order + 1) * (order + 2) / 2;
        if dims != [rep.n_ao, rep.n_ao, n_comp] {
            return Err(SymmetryError(format!("{what}: the tensor is {dims:?}, an order-{order} tensor in the {basis:?} basis is {:?}", [rep.n_ao, rep.n_ao, n_comp])));
        }
        Ok(rep)
    }

    /// Averages a Cartesian tensor operator over the group, all its components together: `planes` the AO matrices of
    /// its components, `(n_ao, n_ao, (order + 1) (order + 2) / 2)`, in the library's component order (lx descending,
    /// then ly descending), the multipole about a fixed point (`is_fixed_point`). Whatever the orientation of the
    /// molecule and the group (degenerate irreps included: the components mix). One pass, in place; reports what it
    /// removed.
    pub fn average_tensor(&self, basis: AoBasis, planes: &mut Array3DViewMut<'_, f64>, order: usize, kind: TensorKind, arena: &Arena<'_>) -> Result<SymmetrisationReport, SymmetryError> {
        let rep = self.check_tensor("average_tensor", basis, planes.dims(), order)?;
        self.project_tensor_by_orbits(rep, &mut WritablePlanes(planes), order, kind, arena)
    }

    /// `average_tensor` refused (and `planes` left alone) if an element would change by more than `tolerance`.
    pub fn symmetrise_tensor(
        &self,
        basis: AoBasis,
        planes: &mut Array3DViewMut<'_, f64>,
        order: usize,
        kind: TensorKind,
        tolerance: f64,
        arena: &Arena<'_>,
    ) -> Result<SymmetrisationReport, SymmetryError> {
        let rep = self.check_tensor("symmetrise_tensor", basis, planes.dims(), order)?;
        let report = self.project_tensor_by_orbits(rep, &mut ReadOnlyPlanes(planes.view()), order, kind, arena)?;
        if report.max_abs_removed > tolerance {
            return Err(SymmetryError(format!(
                "symmetrise_tensor: averaging the order-{order} tensor over {} would change an element by {:.3e} (tolerance {tolerance:.1e}; \
                 removed norm {:.3e} of {:.3e}). Something breaks the symmetry",
                self.point_group, report.max_abs_removed, report.removed_norm, report.norm
            )));
        }
        self.project_tensor_by_orbits(rep, &mut WritablePlanes(planes), order, kind, arena)
    }

    // images = [D(R_0) c, D(R_1) c, ...], (n_ao, h).
    fn images_of(&self, c: &[f64], images: &mut MatrixViewMut<'_, f64>) {
        for (op, image) in images.iter_mut().enumerate() {
            self.apply_to_vector(&self.ao, op, c, image);
        }
    }

    /// Symmetrizes the MO coefficients `orbitals` (`(n_ao, n_mo)`, one MO per column, any number of MOs) in place:
    /// each MO is projected onto the irrep it has most of, and the MOs of a degenerate irrep are grouped into partner
    /// sets and replaced by the partner functions of libmsym's canonical components (which leaves the space a set spans
    /// alone). Writes each MO's label to `labels_out` and returns the largest norm, in coefficients, of the part of an
    /// MO outside its irrep: what was dropped.
    ///
    /// The result is not re-orthonormalized: re-orthonormalize (in the AO metric, within each label) if the returned
    /// norm is not negligible.
    pub fn symmetrise_orbitals<M>(&self, orbitals: &mut M, labels_out: &mut [OrbitalLabel], arena: &Arena<'_>) -> Result<f64, SymmetryError>
    where
        M: MatrixTraitMut<f64>,
    {
        let (n, n_mo) = (self.ao.n_ao, orbitals.cols());
        if orbitals.rows() != n || labels_out.len() != n_mo {
            return Err(SymmetryError(format!(
                "symmetrise_orbitals: needs ({n}, n_mo) coefficients and n_mo labels, got ({}, {n_mo}) and {}",
                orbitals.rows(),
                labels_out.len()
            )));
        }
        let mut images = MatrixViewMut::<f64>::from_arena(n, self.operations.len(), arena)?;
        let mut projected = MatrixViewMut::<f64>::from_arena(n, self.irreps.len(), arena)?;

        // Each MO onto its dominant irrep: all irreps' projections at once, P_Γ c = Σ_R w_Γ(R) D(R) c.
        let mut dropped = 0.0f64;
        for (c, label) in orbitals.iter_mut().zip(labels_out.iter_mut()) {
            self.images_of(c, &mut images);
            images.matmul_prealloc(&self.projector_weights, &mut projected);
            let irrep = projected
                .iter()
                .enumerate()
                .filter(|&(k, _)| self.ao.ao_irrep_counts[k] > 0)
                .map(|(k, column)| (k, self.metric_dot(column, column)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(self.totally_symmetric, |(k, _)| k);
            let kept = projected.get_col(irrep);
            dropped = dropped.max(vector_math::distance(c, kept));
            // The projection is the result, and goes into the caller's column.
            c.copy_from_slice(kept);
            *label = OrbitalLabel { irrep, component: 0 };
        }

        // Degenerate partners, irrep by irrep.
        let assigned = arena.alloc_zeroed::<u8>(n_mo)?;
        for (irrep, transfer) in self.transfer_weights.iter().enumerate() {
            if let Some(transfer) = transfer {
                self.align_partners(orbitals, labels_out, irrep, transfer, assigned, &mut images, arena)?;
            }
        }
        Ok(dropped)
    }

    // Groups the MOs of a degenerate irrep into partner sets and replaces each set by the partner functions of the
    // canonical components: for an MO, its largest component k* gives v_j = P_{j k*} c (one product with the transfer
    // weights), the other unassigned MOs of the irrep with the largest share in the span of the v_j join it, and the
    // set's MOs become the v_j, at the set's mean norm in the invariant metric (in an irreducible invariant space it
    // is proportional to the AO metric, so every AO-normalized vector there has the same norm in it), signs kept.
    #[allow(clippy::too_many_arguments)]
    fn align_partners<M: MatrixTraitMut<f64>>(
        &self,
        orbitals: &mut M,
        labels: &mut [OrbitalLabel],
        irrep: usize,
        transfer: &Matrix<f64>,
        assigned: &mut [u8],
        images: &mut MatrixViewMut<'_, f64>,
        arena: &Arena<'_>,
    ) -> Result<(), SymmetryError> {
        let n = self.ao.n_ao;
        let d = self.irreps[irrep].dimension;
        let mut partners = MatrixViewMut::<f64>::from_arena(n, d, arena)?;
        let component = arena.alloc::<f64>(n)?;
        // set[j]: the MO taking component j; taken[j]: whether one has.
        let set = arena.alloc::<usize>(d)?;
        let taken = arena.alloc::<u8>(d)?;
        for first in 0..labels.len() {
            if labels[first].irrep != irrep || assigned[first] != 0 {
                continue;
            }
            self.images_of(orbitals.get_col(first), images);
            let mut star = (0, -1.0);
            for j in 0..d {
                images.matvec_prealloc(transfer.get_col(j + d * j), &mut *component);
                let weight = self.metric_dot(component, component);
                if weight > star.1 {
                    star = (j, weight);
                }
            }
            let star = star.0;
            images.matmul_prealloc(&transfer.view_columns(d * star..d * star + d), &mut partners);

            taken.fill(0);
            set[star] = first;
            taken[star] = 1;
            assigned[first] = 1;
            let mut complete = true;
            for _ in 1..d {
                // The unassigned MO of the irrep most inside the partners' span (1 for a partner of the same level; MOs
                // of other levels are orthogonal in the AO metric, not in coefficients, so the share in a single v_j
                // would not tell them apart). The v_j are mutually orthogonal, so the share is a sum over them.
                let mut candidate: Option<(usize, f64)> = None;
                for (o, label) in labels.iter().enumerate() {
                    if label.irrep != irrep || assigned[o] != 0 {
                        continue;
                    }
                    let c = orbitals.get_col(o);
                    let share = partners.iter().map(|v| self.metric_dot(v, c).powi(2) / self.metric_dot(v, v).max(f64::MIN_POSITIVE)).sum::<f64>()
                        / self.metric_dot(c, c).max(f64::MIN_POSITIVE);
                    if candidate.is_none_or(|(_, best)| share > best) {
                        candidate = Some((o, share));
                    }
                }
                let Some((o, _)) = candidate else {
                    complete = false;
                    break;
                };
                // Its component: the free one it overlaps most.
                let c = orbitals.get_col(o);
                let j = partners
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| taken[j] == 0)
                    .map(|(j, v)| (j, self.metric_dot(v, c).powi(2) / self.metric_dot(v, v).max(f64::MIN_POSITIVE)))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or(0, |(j, _)| j);
                set[j] = o;
                taken[j] = 1;
                assigned[o] = 1;
            }
            for (j, (&mo, &has)) in set.iter().zip(taken.iter()).enumerate() {
                if has != 0 {
                    labels[mo].component = j;
                }
            }
            if !complete {
                // Fewer MOs of the irrep left than its dimension: labelled, left as projected.
                continue;
            }
            let mean = set.iter().map(|&mo| self.metric_dot(orbitals.get_col(mo), orbitals.get_col(mo)).sqrt()).sum::<f64>() / d as f64;
            for (&mo, v) in set.iter().zip(partners.iter()) {
                let column = orbitals.get_col_mut(mo);
                let sign = if self.metric_dot(v, column) < 0.0 { -1.0 } else { 1.0 };
                // The partner function is the result, and goes into the caller's column.
                column.copy_from_slice(v);
                vector_math::scale_in_place(column, sign * mean / self.metric_dot(v, v).sqrt());
            }
        }
        Ok(())
    }

    /// The number of the group's operations (its order h); operations are numbered 0..h in a fixed order, the same
    /// for every query below.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Operation `op` applied to each column of `coefficients` (MO coefficients over the contracted AOs, `(n_ao, m)`),
    /// into the same column of `out`: the coefficients of R phi, so `C^T S out` is the operation's matrix over a set
    /// of orbitals the group maps onto itself, `R phi_q = sum_p U_pq phi_p` (the convention of `irrep_matrix`).
    /// Allocates nothing.
    pub fn apply_operation<A: MatrixTrait<f64>, O: MatrixTraitMut<f64>>(&self, op: usize, coefficients: &A, out: &mut O) -> Result<(), SymmetryError> {
        let n = self.ao.n_ao;
        if op >= self.operations.len() || coefficients.rows() != n || out.rows() != n || out.cols() != coefficients.cols() {
            return Err(SymmetryError(format!(
                "apply_operation: operation {op} of {}, ({n}, m) coefficients and output; got ({}, {}) and ({}, {})",
                self.operations.len(),
                coefficients.rows(),
                coefficients.cols(),
                out.rows(),
                out.cols()
            )));
        }
        for (c, image) in coefficients.iter_col().zip(out.iter_mut()) {
            self.apply_to_vector(&self.ao, op, c, image);
        }
        Ok(())
    }

    /// The character of irrep `irrep` at operation `op` (a pair of complex-conjugate irreps: their sum, real).
    pub fn character(&self, irrep: usize, op: usize) -> f64 {
        self.characters[(irrep, self.operations[op].1)]
    }

    /// The matrix Gamma(R)_jk of irrep `irrep` at operation `op` (d x d, column-major, into `out`), for the components
    /// `symmetrise_orbitals` aligns partners to: R f_k = sum_j Gamma_jk f_j. `false` (out untouched) when the irrep's
    /// partners are not aligned (one-dimensional irreps have Gamma = the character; complex pairs and T, Th none).
    pub fn irrep_matrix(&self, irrep: usize, op: usize, out: &mut [f64]) -> bool {
        let Some(Some(transfer)) = self.transfer_weights.get(irrep) else {
            return false;
        };
        let d = self.irreps[irrep].dimension;
        let scale = self.operations.len() as f64 / d as f64;
        // The operation's row of the weights (h x d^2): one element of each column.
        for (target, column) in out[..d * d].iter_mut().zip(transfer.iter()) {
            *target = scale * column[op];
        }
        true
    }

    /// Operation `op` as a 3 x 3 matrix in the input's axes, `m[i][k]`: a position `p` goes to
    /// `c + sum_k m[i][k] (p - c)_k` with `c` the centre of mass, a displacement or a gradient `v` of a centre to
    /// `sum_k m[i][k] v_k` at its image (`centre_image`). Orthogonal.
    pub fn operation_matrix(&self, op: usize) -> [[f64; 3]; 3] {
        self.operations[op].0
    }

    /// The number of centres the group was found for.
    pub fn centre_count(&self) -> usize {
        self.centre_images.rows()
    }

    /// The centre (0-based, in the order the centres were given) operation `op` takes `centre` to.
    pub fn centre_image(&self, centre: usize, op: usize) -> usize {
        self.centre_images[(centre, op)]
    }

    /// The centre of mass libmsym works relative to (Bohr).
    pub fn centre_of_mass(&self) -> Result<[f64; 3], SymmetryError> {
        let mut v = [0.0; 3];
        // SAFETY: v is a double[3].
        check(unsafe { msymGetCenterOfMass(self.ctx.raw(), v.as_mut_ptr()) }, "get the centre of mass")?;
        Ok(v)
    }
}

// The dense per-operation projection: n^2 scratch, every operation applied to the whole matrix
// (`M D(R)` one product per column block, the left factor through a transpose). The reference the tiled version is
// tested against, and its timing baseline.
#[cfg(test)]
impl Symmetry {
    // out = alpha M D(op) + beta out: block b of the columns is M's columns of b's image times b's rotation.
    fn multiply_right(&self, op: usize, matrix: &MatrixView<'_, f64>, out: &mut MatrixViewMut<'_, f64>, alpha: f64, beta: f64) {
        for (b, block) in self.ao.blocks.iter().enumerate() {
            let source = matrix.view_columns(self.ao.image(op, b).range());
            source.matmul_scaled_prealloc(&self.rotation(op, block), &mut out.view_columns_mut(block.range()), alpha, beta);
        }
    }

    // out = alpha (D(op)^T M D(op))^T + beta out = alpha D(op)^T M^T D(op) + beta out, through `product` and
    // `transposed` (n x n scratch).
    #[allow(clippy::too_many_arguments)]
    fn transform_transposed(
        &self,
        op: usize,
        matrix: &MatrixView<'_, f64>,
        product: &mut MatrixViewMut<'_, f64>,
        transposed: &mut MatrixViewMut<'_, f64>,
        out: &mut MatrixViewMut<'_, f64>,
        alpha: f64,
        beta: f64,
    ) {
        self.multiply_right(op, matrix, product, 1.0, 0.0);
        product.transpose_prealloc(transposed);
        self.multiply_right(op, &transposed.view(), out, alpha, beta);
    }

    // |P_Γ M|^2 = <M, P_Γ M> (the projector is self-adjoint) = Σ_R w_Γ(R) <M, D(R)^T M D(R)>: one transform per
    // operation, then one product with the projector weights. <M, D^T M D> = <M^T, D^T M^T D>, the transform as
    // `transform_transposed` makes it.
    fn irrep_norms_dense(&self, matrix: &MatrixView<'_, f64>, norms_out: &mut [f64], arena: &Arena<'_>) -> Result<(), SymmetryError> {
        let n = self.ao.n_ao;
        let mut transposed_matrix = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let mut product = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let mut transposed = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let mut transformed = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let overlaps = arena.alloc::<f64>(self.operations.len())?;
        matrix.transpose_prealloc(&mut transposed_matrix);
        for (op, overlap) in overlaps.iter_mut().enumerate() {
            self.transform_transposed(op, matrix, &mut product, &mut transposed, &mut transformed, 1.0, 0.0);
            *overlap = transposed_matrix.frobenius_inner(&transformed);
        }
        self.projector_weights.view_transposed().matvec_prealloc(&*overlaps, norms_out);
        norms_out.iter_mut().for_each(|norm| *norm = norm.max(0.0).sqrt());
        Ok(())
    }

    // Projects `matrix` onto `operator` into `cleaned` and reports what the projection removed.
    fn project_dense(
        &self,
        matrix: &MatrixView<'_, f64>,
        operator: OperatorSymmetry,
        cleaned: &mut MatrixViewMut<'_, f64>,
        arena: &Arena<'_>,
    ) -> Result<SymmetrisationReport, SymmetryError> {
        let d = self.irreps.len();
        let irrep = match operator {
            OperatorSymmetry::TotallySymmetric => self.totally_symmetric,
            OperatorSymmetry::Irrep(k) if k >= d => {
                return Err(SymmetryError(format!("irrep {k} out of range, {} has {d}", self.point_group)));
            }
            OperatorSymmetry::Irrep(k) => k,
            OperatorSymmetry::Auto => {
                let norms = arena.alloc::<f64>(d)?;
                self.irrep_norms_dense(matrix, norms, arena)?;
                (0..d).max_by(|&a, &b| norms[a].total_cmp(&norms[b])).unwrap_or(self.totally_symmetric)
            }
        };
        // P_Γ M = Σ_R w_Γ(R) D(R)^T M D(R), accumulated transposed and transposed back once.
        let n = self.ao.n_ao;
        let mut product = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let mut transposed = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        let mut acc = MatrixViewMut::<f64>::from_arena(n, n, arena)?;
        acc.get_data_mut().fill(0.0);
        for (op, &weight) in self.projector_weights.get_col(irrep).iter().enumerate() {
            if weight != 0.0 {
                self.transform_transposed(op, matrix, &mut product, &mut transposed, &mut acc, weight, 1.0);
            }
        }
        acc.transpose_prealloc(cleaned);

        let mut report = SymmetrisationReport { irrep, max_abs_removed: 0.0, removed_norm: 0.0, norm: 0.0 };
        for (&old, &new) in matrix.get_data().iter().zip(cleaned.get_data()) {
            let removed = (old - new).abs();
            report.max_abs_removed = report.max_abs_removed.max(removed);
            report.removed_norm += removed * removed;
        }
        report.removed_norm = report.removed_norm.sqrt();
        report.norm = vector_math::norm(matrix.get_data());
        Ok(report)
    }

}

#[cfg(test)]
mod tests;
