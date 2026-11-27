// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! ERI gradients, for every contraction type (general, segmented, primitive; see `basis::contraction`).
//!
//! What is computed. For every shell-pair quartet `(ab|cd)` the derivative of the integral batch
//! with respect to the *atoms* the four shells sit on, as batches with exactly the layout of the
//! energy batch (`ERIBatchType::EightD`). Because a derivative with respect to an atom is the
//! derivative of a function of the geometry, it has the same 8-fold permutational symmetry as the
//! integral itself, so the unique-integral walk of `distribute` (`for_each_unique_general`) applies
//! unchanged. A `GradientSink` is told the atoms of a quartet once and then receives, per unique
//! integral, the derivatives for every (atom, axis) together, so density weights are computed once
//! per integral.
//!
//! The scheme, per primitive quartet, with `R = B - A` (`X_AB` convention of this code base) and the
//! Hermite expansion `I = sum E^ab_t (-1)^tau E^cd_tau R_{t+tau}(P - Q)`:
//!
//! ```text
//!   S_x = sum E^ab E^cd (-1)^tau R_{t+tau+1_x}      d/dP_x, an index shift of the R tensor
//!   B_x = sum (dE^ab/dR_x) E^cd (-1)^tau R_{t+tau}  bra R-derivative
//!   K_x = sum E^ab (dE^cd/dR_x) (-1)^tau R_{t+tau}  ket R-derivative
//!
//!   dI/dA = (a/p) S - B      dI/dB = (b/p) S + B
//!   dI/dC = -(c/q) S - K     dI/dD = -(d/q) S + K        (sum of the four = 0)
//! ```
//!
//! One R-matrix `M` with the bra Hermite order raised by one (`l_ab + 1`) serves every shift: the
//! shifted bra coefficients are simply embedded at shifted row positions. Contraction weights that
//! depend on the primitive pair (`a/p`, `c/q`) are applied as scalings on the bra matrices and as a
//! row-scaled contraction matrix on the ket side, so the expensive part (`M * E_ket` per primitive
//! quartet) is shared.
//!
//! Which slots are computed. A pair whose two shells sit on the same atom is `merged`: the two
//! slots are one atom, the `dE/dR` terms cancel and only `S` survives (also required for
//! correctness: a same-shell pair stores only the triangular half of its primitive pairs, which is
//! only valid for quantities symmetric under `a <-> b`). Only the {P, A, C} batches are built from
//! Hermite products: `P = S`, `A` if the bra is split, `C` if the ket
//! is split; after contraction `B = P - A` and `D = -P - C` (a merged ket: `-P`), which needs no
//! exponents. Slots on the same atom are summed before they reach the sink. A quartet whose four
//! shells all sit on one atom has zero gradient and is skipped outright.
//!
//! Frozen centres (`active` mask): a frozen atom is never sent to the sink; a quartet touching only
//! frozen atoms is skipped; `A` (the bra `dE/dR` products) is skipped when both bra atoms are frozen,
//! `C` (the ket `dE/dR` products) when both ket atoms are, `P` when nothing needs it.
//!
//! Screening: a quartet is skipped when its derivative Schwarz bound times the largest density product it touches
//! is below the threshold (the sink's densities, `GradientSink::screening`). The per-pair coefficient variants are held
//! on the heap for one call.

use aedelsten_mpi::Communicator;
use arrays::array3d::{Array3DTrait, Array3DTraitMut, Array3DView};
use arrays::array4d::{Array4DTrait, Array4DTraitMut, Array4DViewMut};
use arrays::array5d::{Array5D, Array5DTrait, Array5DTraitMut};
use crate::r_matrix::*;
use arrays::matrix::*;
use mmd::hermite_expansion::{hermite_count, hermite_index, KetExpansionCoefficientsTrait, RefKetExpansionCoefficients};
use arrays::scratch::{Arena, ArenaError, Scratch, ScratchError};
use basis::gto_products::GTOProductTrait;
use basis::contraction::{Contraction, ShellKind, Strategy};
use basis::shell_pairs::ShellPair;
use arrays::vector_math::{add_in_place, vector_norm_square_3d, vector_subtract_3d};
use mmd::hermite_coefficients::{HermiteCoefficients, HermiteCoefficientsTrait, HermiteCoefficientsTraitMut};
use mmd::hermite_coulomb_integrals::{HermiteCoulombIntegrals4D, RangeVersion};
use mmd::ketside_hermite_coefficients::{compute_e_ket_cartesian_prealloc_w_norm, KetHarmonicTransform};

use crate::distribute::for_each_unique_general;
use crate::walks::{walk_pieces, Handout, EriData, RowPiece};
use crate::batch::{compute_eri_coeff, construct_r_xxxx_prealloc, re_kernels};
use crate::TensorChoice;

/// The derivatives of one unique integral with respect to every component of the current quartet:
/// component `q` belongs to atom `atoms[q / 3]` and axis `q % 3` of the last `GradientSink::select_atoms`.
/// A strided view into the kernel's derivative batches (one batch per component), no copy.
#[derive(Clone, Copy)]
pub struct IntegralDerivatives<'a> {
    data: &'a [f64],
    offset: usize,
    stride: usize,
    len: usize,
}

impl IntegralDerivatives<'_> {
    /// The number of components, `3 * atoms.len()`.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no components.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `d(ij|kl)/dX_axis` for component `q = 3 * slot + axis`.
    #[inline(always)]
    pub fn get(&self, q: usize) -> f64 {
        debug_assert!(q < self.len);
        self.data[self.offset + q * self.stride]
    }
}

/// Receives ERI *derivative* integrals, one shell-pair quartet at a time. Per quartet the kernel first
/// calls `select_atoms` with the atoms the quartet's derivatives belong to (distinct, at most four; the
/// shells on one atom already summed; frozen atoms left out), then `distribute(i, j, k, l, derivatives)`
/// once per unique integral with the derivatives for every (atom, axis) of that list. So a sink that
/// weights each integral with densities computes the weight once per integral, not once per atom and
/// axis. Derivatives with respect to atoms keep the 8-fold permutational symmetry of the integrals, so
/// `(i, j, k, l)` runs over the unique integrals exactly as in the energy code; contributions that are
/// identically zero (all four shells on one atom) are not sent at all.
pub trait GradientSink {
    fn select_atoms(&mut self, atoms: &[usize]);
    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, derivatives: IntegralDerivatives<'_>);

    /// As `TensorChoice::wants_mpi_partition`: whether the quartet loop may be split over the ranks of the walk's
    /// communicator.
    fn wants_mpi_partition(&self) -> bool {
        false
    }

    /// As `TensorChoice::allreduce_sum`, called once after a split loop.
    fn allreduce_sum(&mut self, _comm: &Communicator) {}

    /// The densities the derivative integrals are weighed with, for the walk's density screening; `None` (the
    /// default): the walk screens by the derivative bounds alone.
    fn screening(&self) -> Option<GradientScreening<'_>> {
        None
    }
}

/// One term of the densities a gradient sink weighs the derivative integrals with: the integral `(ab|cd)` enters with
/// `coeff left[a, b] right[c, d]` (`exchange` false) or `coeff left[a, c] right[b, d]` (`exchange` true), and so do its
/// permutations; `left` and `right` `n x n`, column-major.
pub struct GradientScreeningTerm<'a> {
    /// Whether the term pairs the densities exchange-like.
    pub exchange: bool,
    /// The first density.
    pub left: &'a [f64],
    /// The second density.
    pub right: &'a [f64],
    /// The term's coefficient.
    pub coeff: f64,
}

/// The terms a gradient sink weighs the derivative integrals with (`GradientSink::screening`), each density `n x n`.
pub struct GradientScreening<'a> {
    /// The functions.
    pub n: usize,
    /// The terms.
    pub terms: Vec<GradientScreeningTerm<'a>>,
}

/// The walk's screening of a quartet: skipped when its derivative bound `Q'_ab Q_cd + Q_ab Q'_cd`
/// (`derivative_schwarz_bounds`, the entries' Schwarz bounds) times the largest density product it touches is below
/// `threshold`. The density products per term from the largest magnitudes of its densities over every pair of shells
/// (block or its transpose); 1 for a sink without densities.
struct DerivativeScreen<'s> {
    threshold: f64,
    /// Per entry of the walk's `EriData`.
    derivative: Vec<f64>,
    /// Per term, the left then the right density's maxima, `n_shells^2` each.
    tables: &'s [f64],
    /// Per term, `|coeff|` and whether it is an exchange term.
    terms: Vec<(f64, bool)>,
    n_shells: usize,
}

impl<'s> DerivativeScreen<'s> {
    /// `pair_data` the walk's, by entry of `eri_data`.
    fn new<'a, G: GradientSink, R: RangeVersion, S: ShellKind>(
        sink: &G,
        shell_pairs: &'a [ShellPair<S>],
        pair_data: &[PairData<Strategy<'a, S>>],
        eri_data: &EriData,
        threshold: f64,
        arena: &mut Arena<'s>,
        omega: f64,
    ) -> Result<Self, ArenaError> {
        let pairs = eri_data.entries().iter().zip(pair_data).map(|(entry, data)| (&shell_pairs[entry.original_index], data));
        let derivative = second_order::derivative_schwarz_bounds::<R, S>(pairs, arena, omega)?;
        let shells = &eri_data.shells;
        let n_shells = shells.len();
        let screening = sink.screening();
        let terms = screening.as_ref().map_or(&[][..], |screening| &screening.terms[..]);
        let tables = arena.alloc_zeroed::<f64>(screen_tables_len(terms.len(), n_shells))?;
        for (term, table) in terms.iter().zip(tables.chunks_exact_mut(2 * n_shells * n_shells)) {
            let n = screening.as_ref().map_or(0, |screening| screening.n);
            let (left, right) = table.split_at_mut(n_shells * n_shells);
            block_maxima(term.left, n, shells, left);
            block_maxima(term.right, n, shells, right);
        }
        let terms = terms.iter().map(|term| (term.coeff.abs(), term.exchange)).collect();
        Ok(DerivativeScreen { threshold, derivative, tables, terms, n_shells })
    }

    /// Whether the quartet of entries `i` and `j` of `eri_data` is below the threshold.
    fn skips(&self, eri_data: &EriData, i: usize, j: usize) -> bool {
        let (ab, cd) = (&eri_data.entries()[i], &eri_data.entries()[j]);
        let bound = self.derivative[i] * cd.schwarz_bound + ab.schwarz_bound * self.derivative[j];
        if self.terms.is_empty() {
            return bound < self.threshold;
        }
        let ((a, b), (c, d)) = (ab.shells, cd.shells);
        let n = self.n_shells;
        let largest = self.terms.iter().enumerate().fold(0.0_f64, |largest, (index, &(coeff, exchange))| {
            let table = &self.tables[2 * index * n * n..2 * (index + 1) * n * n];
            let (left, right) = table.split_at(n * n);
            let pair = |p: usize, q: usize, r: usize, s: usize| left[p * n + q] * right[r * n + s];
            let product = if exchange {
                pair(a, c, b, d).max(pair(a, d, b, c)).max(pair(b, c, a, d)).max(pair(b, d, a, c))
            } else {
                pair(a, b, c, d).max(pair(c, d, a, b))
            };
            largest.max(coeff * product)
        });
        bound * largest < self.threshold
    }
}

/// The screen's tables: per term, the left and the right density's maxima per pair of shells.
fn screen_tables_len(terms: usize, n_shells: usize) -> usize {
    2 * terms * n_shells * n_shells
}

/// Into `table` (`n_shells^2`, zeroed), the largest magnitude of `density` (`n x n`) over every pair of `shells`'
/// blocks, or the block's transpose if larger.
fn block_maxima(density: &[f64], n: usize, shells: &[(usize, usize)], table: &mut [f64]) {
    let n_shells = shells.len();
    for (sb, &(first_b, count_b)) in shells.iter().enumerate() {
        for column in density.chunks_exact(n).skip(first_b).take(count_b) {
            for (sa, &(first_a, count_a)) in shells.iter().enumerate() {
                let largest = column[first_a..first_a + count_a].iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                let entry = &mut table[sb * n_shells + sa];
                *entry = entry.max(largest);
            }
        }
    }
    for a in 0..n_shells {
        for b in 0..a {
            let largest = table[a * n_shells + b].max(table[b * n_shells + a]);
            table[a * n_shells + b] = largest;
            table[b * n_shells + a] = largest;
        }
    }
}

/// The centres a gradient is taken with respect to: the first `n_centres` atoms (the orthogonality-function
/// ghosts come last in the centre list and are never differentiated) and, of those, the ones flagged in
/// `active` (`None` = all of them). Frozen or excluded atoms are never sent to a sink, and work that only
/// feeds them is skipped.
#[derive(Clone, Copy, Debug)]
pub struct GradientCentres<'a> {
    n_centres: usize,
    active: Option<&'a [bool]>,
}

impl<'a> GradientCentres<'a> {
    /// `active`, if given, has one flag per centre (`n_centres` of them).
    pub fn new(n_centres: usize, active: Option<&'a [bool]>) -> Self {
        if let Some(mask) = active {
            assert_eq!(mask.len(), n_centres, "the active mask needs one flag per gradient centre");
        }
        GradientCentres { n_centres, active }
    }

    /// Every one of the first `n_centres` atoms.
    pub fn all(n_centres: usize) -> Self {
        GradientCentres { n_centres, active: None }
    }

    /// Whether `atom` is differentiated.
    #[inline(always)]
    pub fn contains(&self, atom: usize) -> bool {
        atom < self.n_centres && self.active.map_or(true, |mask| mask[atom])
    }
}

/// The atoms of one quartet, as told by `select_atoms` (at most four), without allocating.
#[derive(Clone, Copy, Default)]
pub struct QuartetAtoms {
    atoms: [usize; 4],
    len: usize,
}

impl QuartetAtoms {
    /// The quartet's atoms, at most four.
    pub fn set(&mut self, atoms: &[usize]) {
        debug_assert!(atoms.len() <= 4);
        self.atoms[..atoms.len()].copy_from_slice(atoms);
        self.len = atoms.len();
    }

    /// The atom of component `q = 3 * slot + axis`.
    #[inline(always)]
    pub fn atom(&self, q: usize) -> usize {
        self.atoms[q / 3]
    }

    /// The atoms.
    pub fn as_slice(&self) -> &[usize] {
        &self.atoms[..self.len]
    }
}

/// Debug sink: one ERI tensor per `(atom, axis)`, at index `3 * atom + axis`. Works with every
/// `TensorChoice` that stores plain values, e.g. `PackedERITensor<f64>` (S8) or `Array4D<f64>`; the
/// derivative tensors have the full 8-fold symmetry, so S8 storage is exact.
pub struct GradientTensors<'a, T: TensorChoice> {
    tensors: &'a mut [T],
    atoms: QuartetAtoms,
}

impl<'a, T: TensorChoice> GradientTensors<'a, T> {
    /// `tensors.len()` must be `3 * n_centres`.
    pub fn new(tensors: &'a mut [T]) -> Self {
        assert!(tensors.len() % 3 == 0, "GradientTensors needs 3 tensors (x, y, z) per centre");
        Self { tensors, atoms: QuartetAtoms::default() }
    }
}

impl<T: TensorChoice> GradientSink for GradientTensors<'_, T> {
    fn select_atoms(&mut self, atoms: &[usize]) {
        self.atoms.set(atoms);
    }

    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, derivatives: IntegralDerivatives<'_>) {
        for q in 0..derivatives.len() {
            let index = 3 * self.atoms.atom(q) + q % 3;
            self.tensors[index].distribute(i, j, k, l, derivatives.get(q));
        }
    }
}

/// Emits one quartet: `batches` holds `3 * atoms.len()` contracted derivative batches of `chunk` values
/// each, component `q` at `q * chunk`, each with the layout of the energy batch (`dims`).
fn emit_quartet<G: GradientSink, Sab: ShellKind, Scd: ShellKind>(
    sink: &mut G,
    atoms: &[usize],
    batches: &[f64],
    chunk: usize,
    dims: [usize; 8],
    pair_ab: &ShellPair<Sab>,
    pair_cd: &ShellPair<Scd>,
) {
    sink.select_atoms(atoms);
    let len = 3 * atoms.len();
    for_each_unique_general(dims, pair_ab, pair_cd, |i, j, k, l, offset| {
        sink.distribute(i, j, k, l, IntegralDerivatives { data: batches, offset, stride: chunk, len });
    });
}

// --- Per-pair Hermite data ---------------------------------------------------------------------

/// Presents chosen derivative generations of a `HermiteCoefficientsOwned` as if they were the
/// order-0 coefficients, so that the ket-tensor builder (which reads `get_e_x(0)` etc.) can
/// assemble `dE/dR_x = E'_x E_y E_z` (generation 1 on x, 0 on y and z) without a copy.
struct AxisGenerations<'a, H: HermiteCoefficientsTrait> {
    coefficients: &'a H,
    generation: [usize; 3],
}

impl<H: HermiteCoefficientsTrait> HermiteCoefficientsTrait for AxisGenerations<'_, H> {
    fn get_e_x(&self, _order: usize) -> Array3DView<'_, f64> {
        self.coefficients.get_e_x(self.generation[0])
    }
    fn get_e_y(&self, _order: usize) -> Array3DView<'_, f64> {
        self.coefficients.get_e_y(self.generation[1])
    }
    fn get_e_z(&self, _order: usize) -> Array3DView<'_, f64> {
        self.coefficients.get_e_z(self.generation[2])
    }
    fn get_lqn_a(&self) -> usize {
        self.coefficients.get_lqn_a()
    }
    fn get_lqn_b(&self) -> usize {
        self.coefficients.get_lqn_b()
    }
}

/// Everything one shell pair contributes to gradients, derived once per gradient call and used
/// both as bra (once per outer iteration) and as ket (once per quartet).
///
/// `coefficients` is `(n_hermite, sph_a, sph_b, n_var, n_prim)`, in the harmonic basis of the pair.
/// Variant 0 is `E` (identical to what the energy code uses), variants 1..=3 are `dE/dR_x`,
/// `dE/dR_y`, `dE/dR_z`, and for the second order 4..=9 the second derivatives `xx, xy, xz, yy, yz, zz`
/// (`VARIANTS`). Merged pairs (see the module doc) keep variant 0 only. The variants of one
/// primitive pair are adjacent, so the ket side multiplies all of them by R in one GEMM.
pub(crate) struct PairData<C> {
    split: bool,
    coefficients: Array5D<f64>,
    /// `a/p` for every primitive pair (`b/p = 1 - a/p`).
    fraction: Box<[f64]>,
    /// The contraction step (see `Contraction`); weighted by `fraction` for split pairs, where the ket
    /// contracts `(c/q) E` as well as `E`.
    contraction: C,
}

/// The derivative generations of `E` along x, y and z of each coefficient variant (`PairData`): E, the first
/// derivatives, then the second.
const VARIANTS: [[usize; 3]; 10] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]];

/// `pair_data_to_order` for the gradient: E and its first derivatives.
fn pair_data<'a, S: ShellKind>(pair: &'a ShellPair<S>, scratch: &mut Arena<'_>) -> Result<PairData<Strategy<'a, S>>, ArenaError> {
    pair_data_to_order(pair, scratch, 1)
}

/// The pair's coefficient variants up to the derivative order `order` (1 or 2) with respect to the separation. The
/// result is on the heap (kept for the whole call); the scratch of computing it comes from `scratch`, and is free
/// again on return.
fn pair_data_to_order<'a, S: ShellKind>(pair: &'a ShellPair<S>, scratch: &mut Arena<'_>, order: usize) -> Result<PairData<Strategy<'a, S>>, ArenaError> {
    debug_assert!(order == 1 || order == 2);
    scratch.scope(|scratch| {
        let split = pair.get_centre_idx_a() != pair.get_centre_idx_b();
        let n_var = if !split { 1 } else if order == 1 { 4 } else { 10 };
        let (lqn_a, lqn_b) = (pair.get_lqn_a(), pair.get_lqn_b());
        let lqn_ab = lqn_a + lqn_b;
        let comps_a = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_a, scratch)?;
        let comps_b = cart2sph::get_cartesian_quantum_numbers_m_arena(lqn_b, scratch)?;
        let (cart_a, cart_b) = (comps_a.cols(), comps_b.cols());
        let (sph_a, sph_b) = (pair.get_degeneracy_a(), pair.get_degeneracy_b());
        let n_hermite = hermite_count(lqn_ab);
        let n_prim = pair.get_products().len();
        let separation = pair.get_separation();
        let transform = KetHarmonicTransform::new_in(pair, scratch)?;

        let mut hermite = HermiteCoefficients::from_arena(lqn_a, lqn_b, if split { order } else { 0 }, scratch)?;
        // Cartesian scratch, only when a shell is spherical (else the coefficients are built in place).
        let mut e_cartesian = RefKetExpansionCoefficients::from_arena(if transform.is_some() { cart_a } else { 0 }, cart_b, lqn_ab, scratch)?;
        let mut normalisation = MatrixViewMut::<f64>::from_arena(cart_a, cart_b, scratch)?;
        let mut coefficients = Array5D::<f64>::new(n_hermite, sph_a, sph_b, n_var, n_prim);
        let mut fraction = vec![0.0; n_prim].into_boxed_slice();

        for (ip, product) in pair.get_products().iter().enumerate() {
            let (a, b, p, mu) = (product.get_a(), product.get_b(), product.get_p(), product.get_μ());
            // The product's own coefficient (segmented contraction; ignored for the others).
            let cc = S::product_coefficient(product);
            fraction[ip] = a / p;
            hermite.compute(a, b, p, mu, &separation);
            if split {
                for generation in 1..=order {
                    hermite.compute_derivative(generation, a, b, p, mu, &separation);
                }
            }
            for (idx_b, lmn_b) in comps_b.iter_col().enumerate() {
                let norm_b = cart2sph::compute_cart_gaussian_norm(b, lmn_b[0], lmn_b[1], lmn_b[2]);
                for (idx_a, lmn_a) in comps_a.iter_col().enumerate() {
                    normalisation[(idx_a, idx_b)] =
                        norm_b * cart2sph::compute_cart_gaussian_norm(a, lmn_a[0], lmn_a[1], lmn_a[2]);
                }
            }
            for (var, &generation) in VARIANTS.iter().enumerate().take(n_var) {
                // Each axis' factor of E at its derivative generation with respect to R.
                let mut selected = AxisGenerations { coefficients: &hermite, generation };
                // The support written is the same for every variant, so no refill is needed.
                let mut e = coefficients.view_sub_array3d_mut((var, ip));
                match &transform {
                    None => {
                        let mut e_direct = RefKetExpansionCoefficients::new(cart_a, cart_b, lqn_ab, e.get_data_mut());
                        compute_e_ket_cartesian_prealloc_w_norm::<Strategy<'a, S>, _, _, _, _>(
                            &mut e_direct, &mut selected, &comps_a, &comps_b, &normalisation, cc,
                        );
                    }
                    Some(transform) => {
                        compute_e_ket_cartesian_prealloc_w_norm::<Strategy<'a, S>, _, _, _, _>(
                            &mut e_cartesian, &mut selected, &comps_a, &comps_b, &normalisation, cc,
                        );
                        transform.apply(&e_cartesian.view_as_array3d(), &mut e);
                    }
                }
            }
        }

        let mut contraction = pair.contraction();
        if split {
            contraction.set_weights(&fraction);
        }
        Ok(PairData { split, coefficients, fraction, contraction })
    })
}

// --- Quartet kernel -----------------------------------------------------------------------------

/// Where the bra's Hermite functions `(t, u, v)` of order `l_ab` sit among the rows of an R matrix of order
/// `l_ab + 1` (R E_ket has one row per Hermite function of that order), unshifted and shifted along each axis: the
/// d/dP term of the bra reads R E_ket at `(t + 1, u, v)`, `(t, u + 1, v)` or `(t, u, v + 1)`. The order-`l_ab + 1`
/// layout is not an extension of the order-`l_ab` one, hence the tables. Depends on `l_ab` only; the walk makes it
/// once per bra pair, in the row's arena scope.
pub(crate) struct BraRows<'s> {
    /// `same[s]`: the row of the bra's Hermite function `s`.
    same: &'s [usize],
    /// `shifted[axis][s]`, flattened: the row of function `s` raised by one along `axis`.
    shifted: &'s [usize],
}

impl<'s> BraRows<'s> {
    fn new_in<S: ShellKind>(pair: &ShellPair<S>, arena: &Arena<'s>) -> Result<Self, ArenaError> {
        let lqn_ab = pair.get_lqn_a() + pair.get_lqn_b();
        let tuv_size = hermite_count(lqn_ab);
        let same = arena.alloc::<usize>(tuv_size)?;
        let shifted = arena.alloc::<usize>(3 * tuv_size)?;
        let mut s = 0;
        for v in 0..=lqn_ab {
            for u in 0..=(lqn_ab - v) {
                for t in 0..=(lqn_ab - v - u) {
                    same[s] = hermite_index(t, u, v, lqn_ab + 1);
                    shifted[s] = hermite_index(t + 1, u, v, lqn_ab + 1);
                    shifted[tuv_size + s] = hermite_index(t, u + 1, v, lqn_ab + 1);
                    shifted[2 * tuv_size + s] = hermite_index(t, u, v + 1, lqn_ab + 1);
                    s += 1;
                }
            }
        }
        Ok(BraRows { same, shifted })
    }

    /// The rows of the bra's own Hermite functions.
    fn same(&self) -> &[usize] {
        self.same
    }

    /// The rows of the bra's Hermite functions raised by one along `axis`.
    fn shifted(&self, axis: usize) -> &[usize] {
        let tuv_size = self.same.len();
        &self.shifted[axis * tuv_size..(axis + 1) * tuv_size]
    }
}

/// `target = alpha source[rows, :]`, or `target += alpha source[rows, :]` (`add`): `source` column-major with
/// `rows_source` rows, `target` column-major with `rows.len()` rows, as many columns as `source`. A gather into
/// contiguous rows, so that the bra's products read them as one dense operand.
fn gather_rows(source: &[f64], rows_source: usize, rows: &[usize], alpha: f64, target: &mut [f64], add: bool) {
    for (column, out) in source.chunks_exact(rows_source).zip(target.chunks_exact_mut(rows.len())) {
        for (o, &row) in out.iter_mut().zip(rows) {
            *o = if add { *o + alpha * column[row] } else { alpha * column[row] };
        }
    }
}

/// R E_ket of one bra primitive, contracted over the ket's primitives, each `(rows, sph_c, sph_d, n_cont_cd)` with the
/// rows the bra's Hermite functions: R E (`plain`) and R (c/q) E (`weighted`, if asked for) at the bra's order raised
/// by one (the d/dP terms read them at shifted rows), R dE/dR along x, y, z (`derivatives`, if asked for, adjacent) at
/// the bra's own order (only the bra's own rows read them). The order-`l_ab` R is the order-`l_ab + 1` one restricted
/// to its rows of order `l_ab`. `prim_*` hold the per-primitive products of a ket that does not accumulate,
/// `(rows sph_cd, n_prim_cd)` and `(rows sph_cd, 3, n_prim_cd)`; `r_own` the R matrix's rows of the bra's order, above
/// `L_UNROLL`.
struct KetProducts<'s> {
    plain: &'s mut [f64],
    weighted: &'s mut [f64],
    derivatives: &'s mut [f64],
    prim_plain: &'s mut [f64],
    prim_derivatives: &'s mut [f64],
    r_own: &'s mut [f64],
}

impl<'s> KetProducts<'s> {
    /// The buffers for a bra pair of angular momentum `lqn_ab` and the ket `pair_cd`, with the derivatives and the
    /// weighted term as asked for.
    fn new_in<'a, Scd: ShellKind>(
        lqn_ab: usize,
        pair_cd: &'a ShellPair<Scd>,
        ket: &PairData<Strategy<'a, Scd>>,
        derivatives: bool,
        weighted: bool,
        arena: &Arena<'s>,
    ) -> Result<Self, ArenaError> {
        let lqn_cd = pair_cd.get_lqn_a() + pair_cd.get_lqn_b();
        let sph_cd = pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b();
        let (n_cont_c, n_cont_d) = ket.contraction.contracted();
        let columns = sph_cd * n_cont_c * n_cont_d;
        let (raised, own) = (hermite_count(lqn_ab + 1), hermite_count(lqn_ab));
        let n_prim = if <Strategy<'a, Scd> as Contraction>::ACCUMULATES { 0 } else { pair_cd.get_products().len() };
        let generated = re_kernels(lqn_ab + 1, lqn_cd).is_some();
        Ok(KetProducts {
            plain: arena.alloc::<f64>(raised * columns)?,
            weighted: arena.alloc::<f64>(if weighted { raised * columns } else { 0 })?,
            derivatives: arena.alloc::<f64>(if derivatives { 3 * own * columns } else { 0 })?,
            prim_plain: arena.alloc::<f64>(raised * sph_cd * n_prim)?,
            prim_derivatives: arena.alloc::<f64>(if derivatives { 3 * own * sph_cd * n_prim } else { 0 })?,
            r_own: arena.alloc::<f64>(if derivatives && !generated { own * hermite_count(lqn_cd) } else { 0 })?,
        })
    }

    /// The products of the bra primitive of exponent `p` at `p_x` (a pair of angular momentum `lqn_ab`, its rows
    /// `bra_rows`). Per primitive quartet by the generated kernels of `(l_ab + 1, l_cd)` and `(l_ab, l_cd)` (R from
    /// the Boys values, applied at once), above `L_UNROLL` by the R matrix and BLAS products.
    #[allow(clippy::too_many_arguments)]
    fn compute<'a, R: RangeVersion, Scd: ShellKind>(
        &mut self,
        r_buffer: &mut HermiteCoulombIntegrals4D<'_>,
        r: &mut BufferedRMatrix<'_, f64>,
        p: f64,
        p_x: &[f64; 3],
        lqn_ab: usize,
        bra_rows: &BraRows<'_>,
        pair_cd: &'a ShellPair<Scd>,
        ket: &PairData<Strategy<'a, Scd>>,
        omega: f64,
    ) {
        let lqn_cd = pair_cd.get_lqn_a() + pair_cd.get_lqn_b();
        let sph_cd = pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b();
        let (raised, own) = (hermite_count(lqn_ab + 1), hermite_count(lqn_ab));
        let (derivatives, weighted) = (!self.derivatives.is_empty(), !self.weighted.is_empty());
        let accumulates_cd = <Strategy<'a, Scd> as Contraction>::ACCUMULATES;
        let kernels = re_kernels(lqn_ab + 1, lqn_cd).zip(re_kernels(lqn_ab, lqn_cd));
        for (jp, cd) in pair_cd.get_products().iter().enumerate() {
            // The pair's coefficient variants E, dE/dR_x, dE/dR_y, dE/dR_z, adjacent per primitive pair.
            let ket_block = ket.coefficients.view_sub_array4d(jp);
            let hermite_cd = ket_block.dim0();
            let e_plain = &ket_block.get_data()[..hermite_cd * sph_cd];
            let e_derivatives = if derivatives { &ket_block.get_data()[hermite_cd * sph_cd..4 * hermite_cd * sph_cd] } else { &[][..] };
            let plain_length = if accumulates_cd { 0 } else { raised * sph_cd };
            let derivatives_length = if accumulates_cd || !derivatives { 0 } else { 3 * own * sph_cd };
            let prim_plain = &mut self.prim_plain[jp * plain_length..][..plain_length];
            let prim_derivatives = &mut self.prim_derivatives[jp * derivatives_length..][..derivatives_length];
            match kernels {
                Some(((add_raised, set_raised), (add_own, set_own))) => {
                    let (q, q_x) = (cd.get_p(), cd.get_product_centre());
                    let pq = vector_subtract_3d(p_x, &q_x);
                    let pref = compute_eri_coeff(p, q);
                    R::compute_boys(r_buffer, p * q / (p + q), vector_norm_square_3d(&pq), omega);
                    let boys = r_buffer.get_boys();
                    let (x, y, z) = (pq[0], pq[1], pq[2]);
                    if accumulates_cd {
                        // One contracted pair: summed over the ket's primitives as they come.
                        let (raised_kernel, own_kernel) = if jp == 0 { (set_raised, set_own) } else { (add_raised, add_own) };
                        raised_kernel(boys, x, y, z, pref, e_plain, self.plain);
                        if weighted {
                            raised_kernel(boys, x, y, z, pref * ket.fraction[jp], e_plain, self.weighted);
                        }
                        if derivatives {
                            own_kernel(boys, x, y, z, pref, e_derivatives, self.derivatives);
                        }
                    } else {
                        set_raised(boys, x, y, z, pref, e_plain, prim_plain);
                        if derivatives {
                            set_own(boys, x, y, z, pref, e_derivatives, prim_derivatives);
                        }
                    }
                }
                None => {
                    construct_r_xxxx_prealloc::<R, _>(r_buffer, r, p, p_x, lqn_ab + 1, cd.get_p(), &cd.get_product_centre(), lqn_cd, omega);
                    let r_raised = r.view_as_matrix();
                    let e_plain = MatrixView::<f64>::new(hermite_cd, sph_cd, e_plain);
                    let e_derivatives = MatrixView::<f64>::new(hermite_cd, if derivatives { 3 * sph_cd } else { 0 }, e_derivatives);
                    if derivatives {
                        gather_rows(r_raised.get_data(), raised, bra_rows.same(), 1.0, self.r_own, false);
                    }
                    let r_own = MatrixView::<f64>::new(if derivatives { own } else { 0 }, hermite_cd, &*self.r_own);
                    if accumulates_cd {
                        let beta = if jp == 0 { 0.0 } else { 1.0 };
                        r_raised.matmul_scaled_prealloc(&e_plain, &mut MatrixViewMut::new(raised, sph_cd, &mut *self.plain), 1.0, beta);
                        if weighted {
                            r_raised.matmul_scaled_prealloc(&e_plain, &mut MatrixViewMut::new(raised, sph_cd, &mut *self.weighted), ket.fraction[jp], beta);
                        }
                        if derivatives {
                            r_own.matmul_scaled_prealloc(&e_derivatives, &mut MatrixViewMut::new(own, 3 * sph_cd, &mut *self.derivatives), 1.0, beta);
                        }
                    } else {
                        r_raised.matmul_prealloc(&e_plain, &mut MatrixViewMut::new(raised, sph_cd, prim_plain));
                        if derivatives {
                            r_own.matmul_prealloc(&e_derivatives, &mut MatrixViewMut::new(own, 3 * sph_cd, prim_derivatives));
                        }
                    }
                }
            }
        }
        if !accumulates_cd {
            // To contracted (cd): R E, R (c/q) E (its contraction weighted by c/q), the three R dE/dR (each the
            // middle-index slice of (rows sph_cd, 3, ket primitives), not contiguous).
            let n_prim_cd = pair_cd.get_products().len();
            let columns = self.plain.len() / (raised * sph_cd);
            let per_product = MatrixView::new(raised * sph_cd, n_prim_cd, &*self.prim_plain);
            ket.contraction.contract(&per_product, &mut MatrixViewMut::new(raised * sph_cd, columns, &mut *self.plain), 1.0);
            if weighted {
                ket.contraction.contract_weighted(&per_product, &ket.fraction, &mut MatrixViewMut::new(raised * sph_cd, columns, &mut *self.weighted));
            }
            if derivatives {
                let by_axis = Array3DView::new(own * sph_cd, 3, n_prim_cd, &*self.prim_derivatives);
                for (axis, out) in self.derivatives.chunks_exact_mut(own * sph_cd * columns).enumerate() {
                    ket.contraction.contract(&by_axis.view_dim1_matrix(axis), &mut MatrixViewMut::new(own * sph_cd, columns, out), 1.0);
                }
            }
        }
    }

    /// R E, `(raised rows) x columns`.
    fn plain(&self) -> &[f64] {
        self.plain
    }

    /// R (c/q) E, `(raised rows) x columns`.
    fn weighted(&self) -> &[f64] {
        self.weighted
    }

    /// R dE/dR along `axis`, `(own rows) x columns`.
    fn derivative(&self, axis: usize) -> &[f64] {
        let length = self.derivatives.len() / 3;
        &self.derivatives[axis * length..(axis + 1) * length]
    }
}

/// `target = alpha source` or `target += alpha source` (`add`).
fn scaled_into(source: &[f64], alpha: f64, target: &mut [f64], add: bool) {
    for (t, s) in target.iter_mut().zip(source) {
        *t = if add { *t + alpha * s } else { alpha * s };
    }
}

/// The bra products a gradient quartet holds before contracting them (`compute_quartet`): the energy's `batch_xxxx`
/// stacks up to 16; the contraction stays one product per block with this many rows of the contraction matrix.
const BRA_PRODUCT_BLOCK: usize = 32;

/// What a gradient quartet computes and the sizes of its buffers, from its pairs and the wanted centres alone
/// (`quartet_layout`).
#[derive(Clone, Copy)]
struct QuartetLayout {
    bra_slots: usize,
    n_slots: usize,
    atoms: [usize; 4],
    wanted: [bool; 4],
    need_a: bool,
    need_c: bool,
    need_p: bool,
    k_p: usize,
    k_c: usize,
    k_a: usize,
    n_direct: usize,
    n_wide: usize,
    lqn_ab: usize,
    sph_a: usize,
    sph_b: usize,
    n_cont_a: usize,
    n_cont_b: usize,
    dim_prim_ab: usize,
    tuv_size_shifted: usize,
    tuv_size: usize,
    lqn_cd: usize,
    sph_c: usize,
    sph_d: usize,
    n_cont_c: usize,
    n_cont_d: usize,
    sph_ab: usize,
    cols_cd: usize,
    accumulates_ab: bool,
    n_cont_batches: usize,
    operand: usize,
    block: usize,
    n_stored_ab: usize,
}

/// The quartet's layout: `None` when it computes nothing (all four shells on one atom, or no wanted centre on it).
fn quartet_layout<'a, Sab: ShellKind, Scd: ShellKind>(
    pair_ab: &'a ShellPair<Sab>,
    bra: &PairData<Strategy<'a, Sab>>,
    pair_cd: &'a ShellPair<Scd>,
    ket: &PairData<Strategy<'a, Scd>>,
    centres: GradientCentres<'_>,
) -> Option<QuartetLayout> {
    // Slots, in order: the bra's (one if merged, else A then B), then the ket's. Slot atoms.
    let bra_slots = if bra.split { 2 } else { 1 };
    let ket_slots = if ket.split { 2 } else { 1 };
    let n_slots = bra_slots + ket_slots;
    let mut atoms = [0usize; 4];
    atoms[0] = pair_ab.get_centre_idx_a();
    if bra.split {
        atoms[1] = pair_ab.get_centre_idx_b();
    }
    atoms[bra_slots] = pair_cd.get_centre_idx_a();
    if ket.split {
        atoms[bra_slots + 1] = pair_cd.get_centre_idx_b();
    }
    // Translational invariance: all four shells on one atom means a zero gradient.
    if atoms[..n_slots].iter().all(|&atom| atom == atoms[0]) {
        return None;
    }
    let mut wanted = [false; 4];
    for slot in 0..n_slots {
        wanted[slot] = centres.contains(atoms[slot]);
    }
    if !wanted[..n_slots].iter().any(|&w| w) {
        return None;
    }

    // Which of P, A, C are needed. A serves A and B (B = P - A), C serves C and D (D = -P - C); P
    // serves a merged slot directly, B and D, and the primitive-level A.
    let need_a = bra.split && (wanted[0] || wanted[1]);
    let need_c = ket.split && (wanted[bra_slots] || wanted[bra_slots + 1]);
    let need_p = need_a
        || (!bra.split && wanted[0])
        || (!ket.split && wanted[bra_slots])
        || (ket.split && wanted[bra_slots + 1]);
    // Primitive-level batches: P, then C (the two from one product), then A, as far as needed (three axes each).
    let k_p = 0;
    let k_c = need_p as usize;
    let k_a = k_c + need_c as usize;
    let n_direct = 3 * (k_a + need_a as usize);
    let n_wide = 3 * k_a; // the P and C batches

    // ab metadata
    let lqn_ab = pair_ab.get_lqn_a() + pair_ab.get_lqn_b();
    let sph_a = pair_ab.get_degeneracy_a();
    let sph_b = pair_ab.get_degeneracy_b();
    let (n_cont_a, n_cont_b) = bra.contraction.contracted();
    let dim_prim_ab = pair_ab.get_products().len();
    let tuv_size_shifted = hermite_count(lqn_ab + 1); // the Hermite space of the R-matrix rows
    let tuv_size = hermite_count(lqn_ab); // the bra's own

    // cd metadata
    let lqn_cd = pair_cd.get_lqn_a() + pair_cd.get_lqn_b();
    let sph_c = pair_cd.get_degeneracy_a();
    let sph_d = pair_cd.get_degeneracy_b();
    let (n_cont_c, n_cont_d) = ket.contraction.contracted();

    // Flattened (sph_a sph_b), (sph_c sph_d n_cont_c n_cont_d): rows and columns of the bra step.
    let sph_ab = sph_a * sph_b;
    let cols_cd = sph_c * sph_d * n_cont_c * n_cont_d;
    let accumulates_ab = <Strategy<'a, Sab> as Contraction>::ACCUMULATES;

    let n_cont_batches = 3 * n_slots;
    let operand = tuv_size * cols_cd;
    let block = sph_ab * cols_cd;
    let contracts = <Strategy<'a, Sab> as Contraction>::CONTRACTS;
    let n_stored_ab = if accumulates_ab { 1 } else if contracts { dim_prim_ab.min(BRA_PRODUCT_BLOCK) } else { dim_prim_ab };
    Some(QuartetLayout { bra_slots, n_slots, atoms, wanted, need_a, need_c, need_p, k_p, k_c, k_a, n_direct, n_wide, lqn_ab, sph_a, sph_b, n_cont_a, n_cont_b, dim_prim_ab, tuv_size_shifted, tuv_size, lqn_cd, sph_c, sph_d, n_cont_c, n_cont_d, sph_ab, cols_cd, accumulates_ab, n_cont_batches, operand, block, n_stored_ab })
}

/// A gradient quartet's buffers: everything `run_quartet` uses, taken here and nowhere else (the scratch query takes
/// them alone).
struct QuartetBuffers<'s> {
    r_buffer: HermiteCoulombIntegrals4D<'s>,
    r: BufferedRMatrix<'s, f64>,
    ket_products: KetProducts<'s>,
    gathered: &'s mut [f64],
    e_r_e_prim: &'s mut [f64],
    summed: &'s mut [f64],
    e_r_e_cont: Array4DViewMut<'s, f64>,
}

impl<'s> QuartetBuffers<'s> {
    fn new_in<'a, Scd: ShellKind>(layout: &QuartetLayout, pair_cd: &'a ShellPair<Scd>, ket: &PairData<Strategy<'a, Scd>>, arena: &Arena<'s>) -> Result<Self, ArenaError> {
        let QuartetLayout { lqn_ab, lqn_cd, need_a, need_c, n_wide, operand, block, n_direct, n_stored_ab, accumulates_ab, sph_ab, cols_cd, n_cont_a, n_cont_b, n_cont_batches, .. } = *layout;
        let r_buffer = HermiteCoulombIntegrals4D::from_arena(lqn_ab + 1 + lqn_cd, arena)?;
        let r = BufferedRMatrix::<f64>::from_arena(lqn_ab + 1, lqn_cd, arena)?;
        let ket_products = KetProducts::new_in(lqn_ab, pair_cd, ket, need_c, need_c, arena)?;
        // The gathered rows of R E_ket: the wide product's operand (G, then Z, per axis), then (R E)[same rows] for A.
        let gathered = arena.alloc::<f64>(operand * (n_wide + need_a as usize))?;
        // Per bra primitive pair the n_direct batches `(sph_ab, cols_cd)` are adjacent, held a block of products at a
        // time and contracted per block (accumulated), so the memory does not grow with the bra's primitives; the
        // identity strategy's products are its result and are all kept. An accumulating strategy (one contracted pair)
        // adds each primitive's into `summed` instead.
        let e_r_e_prim = arena.alloc::<f64>(block * n_direct * n_stored_ab)?;
        let summed = arena.alloc_zeroed::<f64>(if accumulates_ab { block * n_direct } else { 0 })?;
        // The contracted batches, one per (slot, axis); P is the merged bra's slot, or, when the bra is split, held in
        // B's place until the slots derived from it are formed (B = P - A last, in place). Not zeroed: every batch read
        // is written first (the computed ones by their first block, B and D when wanted), and the slots not wanted are
        // never read or emitted.
        let e_r_e_cont = Array4DViewMut::<f64>::new(sph_ab, cols_cd, n_cont_a * n_cont_b, n_cont_batches, arena.alloc::<f64>(sph_ab * cols_cd * n_cont_a * n_cont_b * n_cont_batches)?);
        Ok(QuartetBuffers { r_buffer, r, ket_products, gathered, e_r_e_prim, summed, e_r_e_cont })
    }
}

/// Computes and emits the gradient batches of one shell-pair quartet. `bra` and `ket` are the
/// `PairData` of `pair_ab` and `pair_cd`; `centres` are the atoms to differentiate (see
/// `compute_eri_gradient_from_data`).
///
/// Read it against `batch::tiles::batch_xxxx`, which it mirrors: `KetProducts` holds R times the ket
/// coefficients (per ket primitive pair, then contracted), `e_r_e_prim`/`e_r_e_cont` those times the
/// bra coefficients (per bra primitive pair, then contracted). Where `batch_xxxx` has one `E`, there are
/// several here, one per term of the derivative (see the module doc):
///
/// ```text
///   ket, by R of order l_ab + 1 x l_cd:   E,  (c/q) E
///   ket, by R of order l_ab x l_cd:       dE/dR_x, dE/dR_y, dE/dR_z
///   bra, at order l_ab:                   E (against R E_ket at shifted rows for d/dP),  dE/dR
/// ```
///
/// Per bra primitive the rows of the contracted R E_ket the bra needs are gathered (`BraRows`): along each axis
/// `G = (R E)[shifted rows]` and `Z = -((R (c/q) E)[shifted rows] + R dE/dR)`, so that
/// `P = E^T G`, `C = E^T Z` (all axes in one product) and `A = (a/p) P - dE/dR^T (R E)[same rows]`.
///
/// Only the {P, A, C} batches are built from Hermite products: `P = S` for every quartet, `A` if the bra is split,
/// `C` if the ket is split. `B = P - A` and `D = -P - C` follow after contraction, without exponents.
fn compute_quartet<'a, G, R, Sab, Scd>(
    sink: &mut G,
    pair_ab: &'a ShellPair<Sab>,
    bra: &PairData<Strategy<'a, Sab>>,
    bra_rows: &BraRows<'_>,
    pair_cd: &'a ShellPair<Scd>,
    ket: &PairData<Strategy<'a, Scd>>,
    centres: GradientCentres<'_>,
    arena: &Arena<'_>,
    omega: f64,
) -> Result<(), ArenaError>
where
    G: GradientSink,
    R: RangeVersion,
    Sab: ShellKind,
    Scd: ShellKind,
{
    let Some(layout) = quartet_layout(pair_ab, bra, pair_cd, ket, centres) else {
        return Ok(());
    };
    let buffers = QuartetBuffers::new_in(&layout, pair_cd, ket, arena)?;
    run_quartet::<G, R, Sab, Scd>(sink, pair_ab, bra, bra_rows, pair_cd, ket, &layout, buffers, omega);
    Ok(())
}

/// The quartet's derivatives into `buffers` (`QuartetBuffers::new_in` of `layout`), emitted to `sink`.
#[allow(clippy::too_many_arguments)]
fn run_quartet<'a, G, R, Sab, Scd>(
    sink: &mut G,
    pair_ab: &'a ShellPair<Sab>,
    bra: &PairData<Strategy<'a, Sab>>,
    bra_rows: &BraRows<'_>,
    pair_cd: &'a ShellPair<Scd>,
    ket: &PairData<Strategy<'a, Scd>>,
    layout: &QuartetLayout,
    buffers: QuartetBuffers<'_>,
    omega: f64,
) where
    G: GradientSink,
    R: RangeVersion,
    Sab: ShellKind,
    Scd: ShellKind,
{
    let QuartetLayout { bra_slots, n_slots, atoms, wanted, need_a, need_c, need_p, k_p, k_c, k_a, n_direct, n_wide, lqn_ab, sph_a, sph_b, n_cont_a, n_cont_b, dim_prim_ab, tuv_size_shifted, tuv_size, sph_c, sph_d, n_cont_c, n_cont_d, sph_ab, cols_cd, accumulates_ab, operand, block, n_stored_ab, .. } = *layout;
    let QuartetBuffers { mut r_buffer, mut r, mut ket_products, gathered, e_r_e_prim, summed, mut e_r_e_cont } = buffers;
    let p_batch = if bra.split { 1 } else { 0 };

    for (ip, ab) in pair_ab.get_products().iter().enumerate() {
        let p = ab.get_p();
        let p_x = ab.get_product_centre();

        ket_products.compute::<R, Scd>(&mut r_buffer, &mut r, p, &p_x, lqn_ab, bra_rows, pair_cd, ket, omega);

        // E R E in primitive (ab): the rows of R E_ket gathered, P and C in one product, A from P.
        let x_plain = ket_products.plain();
        let (wide, same_rows) = gathered.split_at_mut(operand * n_wide);
        for axis in 0..3 {
            if need_p {
                gather_rows(x_plain, tuv_size_shifted, bra_rows.shifted(axis), 1.0, &mut wide[(3 * k_p + axis) * operand..][..operand], false);
            }
            if need_c {
                // d/dC = -(c/q) [E shifted . R E] - [E . R dE/dR]
                let target = &mut wide[(3 * k_c + axis) * operand..][..operand];
                gather_rows(ket_products.weighted(), tuv_size_shifted, bra_rows.shifted(axis), -1.0, target, false);
                scaled_into(ket_products.derivative(axis), -1.0, target, true);
            }
        }
        let slot = if accumulates_ab { 0 } else { ip % n_stored_ab };
        let out = &mut e_r_e_prim[slot * block * n_direct..(slot + 1) * block * n_direct];
        let e_ab = bra.coefficients.view_sub_array3d((0, ip));
        if n_wide > 0 {
            e_ab.view_as_matrix((1, 2)).view_transposed().matmul_prealloc(
                &MatrixView::new(tuv_size, n_wide * cols_cd, &*wide),
                &mut MatrixViewMut::new(sph_ab, n_wide * cols_cd, &mut out[..n_wide * block]),
            );
        }
        if need_a {
            // d/dA = (a/p) d/dP - [dE/dR . R E]
            gather_rows(x_plain, tuv_size_shifted, bra_rows.same(), 1.0, same_rows, false);
            let (front, a_batches) = out.split_at_mut(3 * k_a * block);
            for (axis, a_batch) in a_batches.chunks_exact_mut(block).enumerate() {
                for (a, &p_term) in a_batch.iter_mut().zip(&front[(3 * k_p + axis) * block..][..block]) {
                    *a = bra.fraction[ip] * p_term;
                }
                let de_ab = bra.coefficients.view_sub_array3d((1 + axis, ip));
                de_ab.view_as_matrix((1, 2)).view_transposed().matmul_scaled_prealloc(
                    &MatrixView::new(tuv_size, cols_cd, &*same_rows),
                    &mut MatrixViewMut::new(sph_ab, cols_cd, a_batch),
                    -1.0,
                    1.0,
                );
            }
        }
        if accumulates_ab {
            add_in_place(summed, out);
        } else if slot + 1 == n_stored_ab || ip + 1 == dim_prim_ab {
            // E R E to contracted (ab) for the block's products, into the batch of its slot: P into the merged bra
            // slot, or behind the slots if the bra is split; A into slot A; C into slot C.
            let first = ip - slot;
            let per_product = Array3DView::new(block, n_direct, slot + 1, &e_r_e_prim[..block * n_direct * (slot + 1)]);
            for (computed, k, batch) in [(need_p, k_p, p_batch), (need_a, k_a, 0), (need_c, k_c, bra_slots)] {
                if !computed {
                    continue;
                }
                for axis in 0..3 {
                    let mut target = e_r_e_cont.view_sub_array3d_mut(3 * batch + axis);
                    let beta = if first == 0 { 0.0 } else { 1.0 };
                    bra.contraction.contract_rows(&per_product.view_dim1_matrix(3 * k + axis), first..ip + 1, &mut target.view_as_matrix_mut((2, 1)), 1.0, beta);
                }
            }
        }
    }

    if accumulates_ab {
        for (computed, k, batch) in [(need_p, k_p, p_batch), (need_a, k_a, 0), (need_c, k_c, bra_slots)] {
            if !computed {
                continue;
            }
            for axis in 0..3 {
                // A copy: the direct batches are in P, C, A order, the slots' batches in slot order.
                e_r_e_cont.view_sub_array3d_mut(3 * batch + axis).get_data_mut().copy_from_slice(&summed[(3 * k + axis) * block..][..block]);
            }
        }
    }
    // The other slots, by translational invariance: D = -P - C, a merged ket -P, then B = P - A in P's place.
    let chunk = sph_ab * cols_cd * n_cont_a * n_cont_b; // one contracted (slot, axis) batch
    let batches = e_r_e_cont.get_data_mut();
    for axis in 0..3 {
        let start = |slot: usize| (slot * 3 + axis) * chunk;
        let last = n_slots - 1;
        if wanted[last] {
            if ket.split {
                combine_within(batches, start(last), &[(start(p_batch), -1.0), (start(bra_slots), -1.0)], chunk, 0.0);
            } else {
                combine_within(batches, start(last), &[(start(p_batch), -1.0)], chunk, 0.0);
            }
        }
        if bra.split && wanted[1] {
            let (a, p) = batches.split_at_mut(start(1));
            arrays::vector_math::add_scaled_in_place(&mut p[..chunk], &a[start(0)..start(0) + chunk], -1.0);
        }
    }

    // Sum the active slots that sit on the same atom (into the first of them), then hand every
    // remaining (atom, axis) batch to the sink through the shared unique-integral distribution loops.
    // Slots of frozen atoms were never computed and are neither summed nor emitted.
    let mut alive = wanted;
    for slot in 0..n_slots {
        if !alive[slot] {
            continue;
        }
        for other in (slot + 1)..n_slots {
            if alive[other] && atoms[other] == atoms[slot] {
                for axis in 0..3 {
                    let (lower, upper) = batches.split_at_mut((other * 3 + axis) * chunk);
                    add_in_place(&mut lower[(slot * 3 + axis) * chunk..(slot * 3 + axis + 1) * chunk], &upper[..chunk]);
                }
                alive[other] = false;
            }
        }
    }
    // Move the remaining slots to the front (component q = 3 * position + axis), then emit once.
    let mut emitted = [0usize; 4];
    let mut n_emitted = 0;
    for slot in 0..n_slots {
        if !alive[slot] {
            continue;
        }
        if slot != n_emitted {
            batches.copy_within(3 * slot * chunk..3 * (slot + 1) * chunk, 3 * n_emitted * chunk);
        }
        emitted[n_emitted] = atoms[slot];
        n_emitted += 1;
    }
    let dims = [sph_a, sph_b, sph_c, sph_d, n_cont_c, n_cont_d, n_cont_a, n_cont_b];
    emit_quartet(sink, &emitted[..n_emitted], batches, chunk, dims, pair_ab, pair_cd);
}

/// `data[target..target + len] = beta * data[target..] + sum_k coef_k * data[source_k..source_k + len]`,
/// for ranges of one buffer that do not overlap the target. `beta = 0` ignores what the target held
/// (arena memory is not clean).
fn combine_within(data: &mut [f64], target: usize, sources: &[(usize, f64)], len: usize, beta: f64) {
    debug_assert!(sources.iter().all(|&(source, _)| source + len <= target || target + len <= source));
    for i in 0..len {
        let mut value = if beta == 0.0 { 0.0 } else { beta * data[target + i] };
        for &(source, coef) in sources {
            value += coef * data[source + i];
        }
        data[target + i] = value;
    }
}

// --- Driver ---------------------------------------------------------------------------------------

/// Walks every unique shell-pair quartet surviving `eri_data`'s pair screening (and the pairwise
/// Schwarz product against `pairwise_threshold`, as the energy code does) and feeds the gradient
/// batches to `sink`. The per-pair derivative data is built once here and dropped on return. If the
/// sink wants an MPI partition, the rows are handed out over the ranks in pieces exactly like the
/// energy loop's (`Handout`) and `allreduce_sum` is called at the end.
///
/// `centres` are the atoms differentiated (the gradient centres, less the frozen ones): other atoms are
/// never sent to the sink, quartets touching none of them are skipped, and the bra (ket) R-derivative
/// products are skipped when neither bra (ket) atom is among them.
pub(crate) fn compute_eri_gradient_from_data<G, R, S>(
    sink: &mut G,
    shell_pairs: &[ShellPair<S>],
    eri_data: &EriData,
    ws: &mut Scratch,
    comm: &Communicator,
    omega: f64,
    pairwise_threshold: Option<f64>,
    centres: GradientCentres<'_>,
) -> Result<(), ArenaError>
where
    G: GradientSink,
    R: RangeVersion,
    S: ShellKind,
{
    let entries = eri_data.entries();
    let mut arena = ws.get_arena();
    // Every pair's coefficients, kept on the heap for the call; their scratch is from the arena.
    let pair_data = entries
        .iter()
        .map(|entry| pair_data(&shell_pairs[entry.original_index], &mut arena))
        .collect::<Result<Vec<PairData<Strategy<'_, S>>>, ArenaError>>()?;
    let handout = (sink.wants_mpi_partition() && comm.size() > 1).then(|| Handout::new(shell_pairs, eri_data, pairwise_threshold, comm));

    let screen = pairwise_threshold.map(|t| DerivativeScreen::new::<G, R, S>(sink, shell_pairs, &pair_data, eri_data, t, &mut arena, omega)).transpose()?;
    let mut skipped = 0u64;
    for RowPiece { row: i, kets } in walk_pieces(handout.as_ref(), entries.len()) {
        let entry_ab = &entries[i];
        let pair_ab = &shell_pairs[entry_ab.original_index];
        // One piece of a row: the bra's row tables in the piece's scope, each quartet in a scope inside it.
        arena.scope(|row| {
            let bra_rows = BraRows::new_in(pair_ab, row)?;
            for (j, entry_cd) in entries.iter().enumerate().take(kets.end).skip(kets.start) {
                if j != i && screen.as_ref().is_some_and(|screen| screen.skips(eri_data, i, j)) {
                    skipped += 1;
                    continue;
                }
                // `pair_cd` must be the very same reference as `pair_ab` on the diagonal: the
                // distribution loops detect a self-pair by pointer identity.
                let pair_cd = &shell_pairs[entry_cd.original_index];
                row.scope(|quartet| {
                    compute_quartet::<G, R, S, S>(sink, pair_ab, &pair_data[i], &bra_rows, pair_cd, &pair_data[j], centres, quartet, omega)
                })?;
            }
            Ok::<(), ArenaError>(())
        })?;
    }

    instrumentation::count("eri::gradient::screened", skipped);
    if handout.is_some() {
        sink.allreduce_sum(comm);
    }
    Ok(())
}

/// The scratch `compute_eri_gradient_from_data` takes into `sink`, a sink of the operation's kind over the probe
/// pairs' functions and their two atoms, for a basis of `n_shells` shells, measured on `ws`: each of `probes`' pair
/// data and (with `screened`) derivative bound, each on its own as the walk takes them; then the screen's tables
/// and in them the largest row of `probes` with its largest quartet, the kets of no more components than their bra.
pub(crate) fn gradient_walk_need<G, S>(
    sink: &mut G,
    probes: &[ShellPair<S>],
    n_shells: usize,
    screened: bool,
    ws: &mut Scratch,
) -> Result<Option<crate::need::Need>, ScratchError>
where
    G: GradientSink,
    S: ShellKind,
{
    use crate::need::{keep_largest, Need, NeedItem};
    let mut largest = None;
    let mut data = Vec::with_capacity(probes.len());
    for (i, pair) in probes.iter().enumerate() {
        let mut kept = None;
        let bytes = ws.measure(|arena| {
            kept = Some(pair_data(pair, arena)?);
            Ok(())
        })?;
        keep_largest(&mut largest, Need::pair(NeedItem::GradientPairData, i, bytes));
        data.push(kept.expect("measured"));
    }
    if screened {
        for (i, (pair, pair_data)) in probes.iter().zip(&data).enumerate() {
            let bytes = ws.measure(|arena| second_order::derivative_bound_buffers(pair, pair_data, arena))?;
            keep_largest(&mut largest, Need::pair(NeedItem::DerivativeBound, i, bytes));
        }
    }
    let terms = sink.screening().map_or(0, |screening| screening.terms.len());
    let centres = GradientCentres::all(2);
    // A row at a time behind the tables: one that does not fit is run again alone.
    for (i, pair_ab) in probes.iter().enumerate() {
        let (mut tables, mut row_end, mut quartet) = (0, 0, (i, 0));
        let bytes = ws.measure(|arena| {
            if screened {
                arena.alloc::<f64>(screen_tables_len(terms, n_shells))?;
            }
            tables = arena.used();
            arena.scope(|row| {
                // The row's bra tables, taken as a row of the walk takes them.
                BraRows::new_in(pair_ab, row)?;
                row_end = row.used();
                quartet = (i, 0);
                for (j, pair_cd) in probes.iter().enumerate().filter(|(_, cd)| cd.components() <= pair_ab.components()) {
                    // The quartet's buffers alone (`QuartetBuffers::new_in`, all `compute_quartet` takes).
                    let (taken, extent) = row.scope_extent(|scope| match quartet_layout(pair_ab, &data[i], pair_cd, &data[j], centres) {
                        Some(layout) => QuartetBuffers::new_in(&layout, pair_cd, &data[j], scope).map(|_| ()),
                        None => Ok(()),
                    });
                    taken?;
                    if extent > quartet.1 {
                        quartet = (j, extent);
                    }
                }
                Ok(())
            })
        })?;
        keep_largest(&mut largest, Need::quartet(bytes, tables, row_end, i, quartet));
    }
    Ok(largest)
}

mod second_order;
#[cfg(test)]
mod bound_tests;
#[cfg(test)]
mod block_benchmark;
#[cfg(test)]
mod shift_oracle;
