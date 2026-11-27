// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Second derivatives of the two-electron integrals with respect to the atoms of one shell-pair quartet. With
//! `d/dA = (a/p) d/dP - d/dR` on the bra, `d/dC = -(c/q) d/dP - d/dS` on the ket (`d/dQ = -d/dP` by translational
//! invariance), per primitive quartet the six blocks in the coordinates {A, P, C}:
//!
//! ```text
//!   [A_y A_x] = (a/p)^2 [P_y P_x] - (a/p) ([P_y R_x] + [P_x R_y]) + [R_y R_x]
//!   [C_y C_x] = (c/q)^2 [P_y P_x] + (c/q) ([P_y S_x] + [P_x S_y]) + [S_y S_x]
//!   [A_y C_x] = -(a c / p q) [P_y P_x] - (a/p) [P_y S_x] + (c/q) [R_y P_x] + [R_y S_x]
//!   [A_y P_x] = (a/p) [P_y P_x] - [R_y P_x]
//!   [C_y P_x] = -(c/q) [P_y P_x] - [S_y P_x]
//!   [P_y P_x]
//! ```
//!
//! `d/dP` raises the bra's Hermite index (two of them: the R matrix of the bra's order raised by two), `d/dR` and
//! `d/dS` act on the bra's and the ket's Hermite coefficients (T. Helgaker and P. R. Taylor, Theor. Chim. Acta 83,
//! 177 (1992)). The blocks are contracted, and every pair of atoms follows from them by additions: with
//! `B = P - A`, `D = -P - C` (a pair on one atom: `P` for the bra, `-P` for the ket), the slots on one atom summed.

use super::*;
#[cfg(test)]
use arrays::vector_math::add_scaled_in_place;

/// The blocks, in this order, each with its 3 x 3 axis pairs `(y, x)` at `3 y + x`.
const AA: usize = 0;
const AP: usize = 1;
const AC: usize = 2;
const PP: usize = 3;
const CP: usize = 4;
const CC: usize = 5;
const N_BATCHES: usize = 54;

/// The shift of a Hermite index: none, one along an axis (1 + axis), or two (`4 + VARIANTS` order of the pair).
fn shift_of(axes: &[usize]) -> usize {
    match axes {
        [] => 0,
        [axis] => 1 + axis,
        [y, x] => second_variant(*y, *x),
        _ => unreachable!("at most two shifts"),
    }
}

/// The coefficient variant of the second derivative along `y` and `x` (`VARIANTS`), the same index as the double
/// shift `shift_of(&[y, x])`.
fn second_variant(y: usize, x: usize) -> usize {
    match (y.min(x), y.max(x)) {
        (0, 0) => 4,
        (0, 1) => 5,
        (0, 2) => 6,
        (1, 1) => 7,
        (1, 2) => 8,
        _ => 9,
    }
}

/// The derivatives of one integral with respect to two coordinates of the quartet's atoms: `get(q_y, q_x)`, with
/// component `q = 3 slot + axis` of the atoms of the last `SecondDerivativeSink::select_atoms`.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct IntegralSecondDerivatives<'a> {
    data: &'a [f64],
    offset: usize,
    stride: usize,
    components: usize,
}

#[cfg(test)]
impl IntegralSecondDerivatives<'_> {
    /// The number of components, `3 * atoms.len()`.
    pub(crate) fn components(&self) -> usize {
        self.components
    }

    /// `d2(ij|kl) / dX_(q_y) dX_(q_x)`.
    pub(crate) fn get(&self, q_y: usize, q_x: usize) -> f64 {
        self.data[self.offset + (q_y * self.components + q_x) * self.stride]
    }
}

/// Receives the second derivatives of the integrals of one quartet: the atoms first (distinct, the shells on one
/// atom summed), then every unique integral with its derivatives for every pair of (atom, axis).
#[cfg(test)]
pub(crate) trait SecondDerivativeSink {
    fn select_atoms(&mut self, atoms: &[usize]);
    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, derivatives: IntegralSecondDerivatives<'_>);
}

/// A slot's derivative in the basis {A, P, C}.
fn slot_basis(slot_is_bra: bool, split: bool, first: bool) -> [f64; 3] {
    match (slot_is_bra, split, first) {
        (true, true, true) => [1.0, 0.0, 0.0],
        (true, true, false) => [-1.0, 1.0, 0.0],
        (true, false, _) => [0.0, 1.0, 0.0],
        (false, true, true) => [0.0, 0.0, 1.0],
        (false, true, false) => [0.0, -1.0, -1.0],
        (false, false, _) => [0.0, -1.0, 0.0],
    }
}

/// The stored batch of `[b1_y b2_x]` for basis derivatives `b1`, `b2` in {A = 0, P = 1, C = 2}.
fn basis_batch(b1: usize, b2: usize, y: usize, x: usize) -> usize {
    let (block, y, x) = match (b1, b2) {
        (0, 0) => (AA, y, x),
        (0, 1) => (AP, y, x),
        (1, 0) => (AP, x, y),
        (0, 2) => (AC, y, x),
        (2, 0) => (AC, x, y),
        (1, 1) => (PP, y, x),
        (2, 1) => (CP, y, x),
        (1, 2) => (CP, x, y),
        _ => (CC, y, x),
    };
    9 * block + 3 * y + x
}

/// The buffer of `fill_shifted_rows` for order `lqn`.
fn shifted_rows_in<'s>(lqn: usize, arena: &Arena<'s>) -> Result<&'s mut [usize], ArenaError> {
    arena.alloc::<usize>(10 * hermite_count(lqn))
}

/// The rows, in the order-`(lqn + 2)` Hermite layout, of the order-`lqn` functions shifted by each `shift_of`:
/// `(hermite_count(lqn), 10)`, into `rows` (`shifted_rows_in`).
fn fill_shifted_rows(rows: &mut [usize], lqn: usize) {
    let tuv = hermite_count(lqn);
    let mut s = 0;
    for v in 0..=lqn {
        for u in 0..=(lqn - v) {
            for t in 0..=(lqn - v - u) {
                for (shift, generation) in VARIANTS.iter().enumerate() {
                    rows[shift * tuv + s] = hermite_index(t + generation[0], u + generation[1], v + generation[2], lqn + 2);
                }
                s += 1;
            }
        }
    }
}

/// Issues every block's terms of one bra primitive (`a/p = a_over_p`): `add(variant, term, shift, alpha, batch)` is to
/// add `alpha E_variant^T K_term[rows shifted by shift]` into `batch`, with the bra's coefficient variants
/// (`VARIANTS`) and the ket terms K of `second_blocks`: E; for a split ket (c/q) E, (c/q)^2 E, dE/dS_x,y,z,
/// (c/q) dE/dS_x,y,z, d2E/dS (six). `diagonal`: only the blocks of the diagonal second derivatives
/// (`diagonal_weights`: AP, AC, PP, CP), which read the first derivatives of E alone.
fn block_terms(bra_split: bool, ket_split: bool, a_over_p: f64, diagonal: bool, mut add: impl FnMut(usize, usize, usize, f64, usize)) {
    let (k_e, k_ge, k_gge) = (0, 1, 2);
    let (k_de, k_gde, k_dde) = (|x: usize| 3 + x, |x: usize| 6 + x, |y: usize, x: usize| 9 + second_variant(y, x) - 4);
    for y in 0..3 {
        for x in 0..3 {
            let (yx, ey, ex) = (shift_of(&[y, x]), shift_of(&[y]), shift_of(&[x]));
            let at = |block_kind: usize| 9 * block_kind + 3 * y + x;
            add(0, k_e, yx, 1.0, at(PP));
            if bra_split && !diagonal {
                add(0, k_e, yx, a_over_p * a_over_p, at(AA));
                add(1 + x, k_e, ey, -a_over_p, at(AA));
                add(1 + y, k_e, ex, -a_over_p, at(AA));
                add(second_variant(y, x), k_e, 0, 1.0, at(AA));
            }
            if bra_split {
                add(0, k_e, yx, a_over_p, at(AP));
                add(1 + y, k_e, ex, -1.0, at(AP));
            }
            if ket_split && !diagonal {
                add(0, k_gge, yx, 1.0, at(CC));
                add(0, k_gde(x), ey, 1.0, at(CC));
                add(0, k_gde(y), ex, 1.0, at(CC));
                add(0, k_dde(y, x), 0, 1.0, at(CC));
            }
            if ket_split {
                add(0, k_ge, yx, -1.0, at(CP));
                add(0, k_de(y), ex, -1.0, at(CP));
            }
            if bra_split && ket_split {
                add(0, k_ge, yx, -a_over_p, at(AC));
                add(0, k_de(x), ey, -a_over_p, at(AC));
                add(1 + y, k_ge, ex, 1.0, at(AC));
                add(1 + y, k_de(x), 0, 1.0, at(AC));
            }
        }
    }
}

/// The weights over the 54 blocks of the diagonal second derivative `(d_X ab|d_X ab)` along `k`, one per slot X of a
/// pair (`[A_k C_k]`, `[B_k D_k]`; `-[P_k P_k]` for a pair on one atom).
fn diagonal_weights(split: bool, k: usize) -> impl Iterator<Item = [f64; N_BATCHES]> {
    let slots: &'static [(bool, bool)] = if split { &[(true, true), (true, false)] } else { &[(false, true)] };
    slots.iter().map(move |&(split, first)| {
        let (u, v) = (slot_basis(true, split, first), slot_basis(false, split, first));
        let mut weights = [0.0; N_BATCHES];
        for (b1, &u_b) in u.iter().enumerate().filter(|(_, c)| **c != 0.0) {
            for (b2, &v_b) in v.iter().enumerate().filter(|(_, c)| **c != 0.0) {
                weights[basis_batch(b1, b2, k, k)] += u_b * v_b;
            }
        }
        weights
    })
}

/// Ket term `index` of `terms`, a `(rows, columns)` matrix.
fn term_mut(terms: &mut [f64], index: usize, (rows, columns): (usize, usize)) -> MatrixViewMut<'_, f64> {
    MatrixViewMut::new(rows, columns, &mut terms[index * rows * columns..(index + 1) * rows * columns])
}

/// Computes and emits the second derivatives of the quartet `(pair_ab|pair_cd)`; `bra` and `ket` their pair data of
/// order two. Correct before fast: R matrices and BLAS products throughout.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_second_quartet<'a, G, R, Sab, Scd>(
    sink: &mut G,
    pair_ab: &'a ShellPair<Sab>,
    bra: &PairData<Strategy<'a, Sab>>,
    pair_cd: &'a ShellPair<Scd>,
    ket: &PairData<Strategy<'a, Scd>>,
    arena: &Arena<'_>,
    omega: f64,
) -> Result<(), ArenaError>
where
    G: SecondDerivativeSink,
    R: RangeVersion,
    Sab: ShellKind,
    Scd: ShellKind,
{
    // The slots (bra first) and their atoms and derivatives in the basis {A, P, C}.
    let mut slots: Vec<(usize, [f64; 3])> = Vec::with_capacity(4);
    slots.push((pair_ab.get_centre_idx_a(), slot_basis(true, bra.split, true)));
    if bra.split {
        slots.push((pair_ab.get_centre_idx_b(), slot_basis(true, true, false)));
    }
    slots.push((pair_cd.get_centre_idx_a(), slot_basis(false, ket.split, true)));
    if ket.split {
        slots.push((pair_cd.get_centre_idx_b(), slot_basis(false, true, false)));
    }
    if slots.iter().all(|&(atom, _)| atom == slots[0].0) {
        return Ok(());
    }
    let mut atoms: Vec<usize> = Vec::with_capacity(4);
    for &(atom, _) in &slots {
        if !atoms.contains(&atom) {
            atoms.push(atom);
        }
    }
    let n_components = 3 * atoms.len();
    let contracted = second_blocks::<R, Sab, Scd>(pair_ab, bra, pair_cd, ket, arena, omega)?;
    let (n_cont_a, n_cont_b) = bra.contraction.contracted();
    let (n_cont_c, n_cont_d) = ket.contraction.contracted();
    let chunk = pair_ab.get_degeneracy_a() * pair_ab.get_degeneracy_b() * pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b() * n_cont_a * n_cont_b * n_cont_c * n_cont_d;

    // Every pair of (atom, axis): the slots on the atoms, each a combination of the blocks.
    let output = arena.alloc_zeroed::<f64>(chunk * n_components * n_components)?;
    let mut weights = [0.0; N_BATCHES];
    for (atom_y, &first) in atoms.iter().enumerate() {
        for (atom_x, &second) in atoms.iter().enumerate() {
            for y in 0..3 {
                for x in 0..3 {
                    weights.fill(0.0);
                    for &(slot_atom_y, u) in slots.iter().filter(|(atom, _)| *atom == first) {
                        for &(slot_atom_x, v) in slots.iter().filter(|(atom, _)| *atom == second) {
                            debug_assert!(slot_atom_y == first && slot_atom_x == second);
                            for (b1, &u_b) in u.iter().enumerate().filter(|(_, c)| **c != 0.0) {
                                for (b2, &v_b) in v.iter().enumerate().filter(|(_, c)| **c != 0.0) {
                                    weights[basis_batch(b1, b2, y, x)] += u_b * v_b;
                                }
                            }
                        }
                    }
                    let target = (3 * atom_y + y) * n_components + 3 * atom_x + x;
                    let out = &mut output[target * chunk..(target + 1) * chunk];
                    for (batch, &weight) in weights.iter().enumerate().filter(|(_, w)| **w != 0.0) {
                        add_scaled_in_place(out, &contracted[batch * chunk..(batch + 1) * chunk], weight);
                    }
                }
            }
        }
    }

    sink.select_atoms(&atoms);
    let dims = [pair_ab.get_degeneracy_a(), pair_ab.get_degeneracy_b(), pair_cd.get_degeneracy_a(), pair_cd.get_degeneracy_b(), n_cont_c, n_cont_d, n_cont_a, n_cont_b];
    for_each_unique_general(dims, pair_ab, pair_cd, |i, j, k, l, offset| {
        sink.distribute(i, j, k, l, IntegralSecondDerivatives { data: &*output, offset, stride: chunk, components: n_components });
    });
    Ok(())
}

/// The contracted blocks of the quartet `(pair_ab|pair_cd)`, `(chunk, 54)` with `chunk` the batch of the energy's
/// layout: per block (`AA`, `AP`, `AC`, `PP`, `CP`, `CC`) its axis pairs `(y, x)` at `9 block + 3 y + x`; the blocks of
/// a pair on one atom that do not exist (A of a merged bra, C of a merged ket) are zero. Every block, from pair data of
/// order two (`for_each_bra_product`), the bra's products stored and then contracted.
#[cfg(test)]
fn second_blocks<'a, 's, R, Sab, Scd>(
    pair_ab: &'a ShellPair<Sab>,
    bra: &PairData<Strategy<'a, Sab>>,
    pair_cd: &'a ShellPair<Scd>,
    ket: &PairData<Strategy<'a, Scd>>,
    arena: &Arena<'s>,
    omega: f64,
) -> Result<&'s [f64], ArenaError>
where
    R: RangeVersion,
    Sab: ShellKind,
    Scd: ShellKind,
{
    let block = pair_ab.get_degeneracy_a() * pair_ab.get_degeneracy_b() * pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b() * {
        let (n_cont_c, n_cont_d) = ket.contraction.contracted();
        n_cont_c * n_cont_d
    };
    let n_prim_ab = pair_ab.get_products().len();
    let accumulates_ab = <Strategy<'a, Sab> as Contraction>::ACCUMULATES;
    let n_stored_ab = if accumulates_ab { 1 } else { n_prim_ab };
    let prim = arena.alloc_zeroed::<f64>(block * N_BATCHES * n_stored_ab)?;
    let buffers = BraProductBuffers::new_in(pair_ab, pair_cd, ket, false, arena)?;
    for_each_bra_product::<R, Sab, Scd>(pair_ab, bra, pair_cd, ket, false, buffers, omega, |ip, out| {
        let slot = if accumulates_ab { 0 } else { ip };
        add_scaled_in_place(&mut prim[slot * block * N_BATCHES..(slot + 1) * block * N_BATCHES], out, 1.0);
    });

    // Contracted over the bra's primitives (an accumulating bra is there already): `(block n_cont_ab, batch)`.
    let (n_cont_a, n_cont_b) = bra.contraction.contracted();
    let chunk = block * n_cont_a * n_cont_b;
    let prim: &'s [f64] = prim;
    if accumulates_ab {
        return Ok(prim);
    }
    let contracted = arena.alloc::<f64>(chunk * N_BATCHES)?;
    let per_product = Array3DView::new(block, N_BATCHES, n_prim_ab, prim);
    for (batch, out) in contracted.chunks_exact_mut(chunk).enumerate() {
        bra.contraction.contract(&per_product.view_dim1_matrix(batch), &mut MatrixViewMut::new(block, n_cont_a * n_cont_b, out), 1.0);
    }
    Ok(contracted)
}

/// `for_each_bra_product`'s buffers, the only memory it takes (the scratch query takes them alone).
struct BraProductBuffers<'s> {
    rows: &'s mut [usize],
    r_buffer: HermiteCoulombIntegrals4D<'s>,
    r: BufferedRMatrix<'s, f64>,
    ket_prim: &'s mut [f64],
    terms: &'s mut [f64],
    scaled: &'s mut [f64],
    gathered: &'s mut [f64],
    out: &'s mut [f64],
}

impl<'s> BraProductBuffers<'s> {
    fn new_in<'a, Sab: ShellKind, Scd: ShellKind>(pair_ab: &'a ShellPair<Sab>, pair_cd: &'a ShellPair<Scd>, ket: &PairData<Strategy<'a, Scd>>, diagonal: bool, arena: &Arena<'s>) -> Result<Self, ArenaError> {
        let ket_split = ket.split;
        let lqn_ab = pair_ab.get_lqn_a() + pair_ab.get_lqn_b();
        let lqn_cd = pair_cd.get_lqn_a() + pair_cd.get_lqn_b();
        let (tuv, raised) = (hermite_count(lqn_ab), hermite_count(lqn_ab + 2));
        let sph_ab = pair_ab.get_degeneracy_a() * pair_ab.get_degeneracy_b();
        let sph_cd = pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b();
        let (n_cont_c, n_cont_d) = ket.contraction.contracted();
        let cols_cd = sph_cd * n_cont_c * n_cont_d;
        let block = sph_ab * cols_cd;
        let n_prim_cd = pair_cd.get_products().len();
        let n_var_ket = if !ket_split { 1 } else if diagonal { 4 } else { 10 };
        let n_terms = if ket_split { 15 } else { 1 };
        Ok(BraProductBuffers {
            rows: shifted_rows_in(lqn_ab, arena)?,
            r_buffer: HermiteCoulombIntegrals4D::from_arena(lqn_ab + 2 + lqn_cd, arena)?,
            r: BufferedRMatrix::<f64>::from_arena(lqn_ab + 2, lqn_cd, arena)?,
            ket_prim: arena.alloc::<f64>(raised * sph_cd * n_var_ket * n_prim_cd)?,
            terms: arena.alloc::<f64>(raised * cols_cd * n_terms)?,
            // R E of each ket primitive scaled by its c/q: contracted with the c/q weights (the contraction's own),
            // (c/q)^2 E.
            scaled: arena.alloc::<f64>(if ket_split && !diagonal { raised * sph_cd * n_prim_cd } else { 0 })?,
            gathered: arena.alloc::<f64>(tuv * cols_cd)?,
            out: arena.alloc::<f64>(block * N_BATCHES)?,
        })
    }
}

/// The second-derivative blocks of `(pair_ab|pair_cd)` one bra product at a time: `visit(ip, out)` with `out` product
/// `ip`'s 54 batches (`(sph_ab, cols_cd)` each, `cols_cd` the ket's components and contracted pairs; the layout of
/// `second_blocks` before the bra's contraction), the ket contracted already. `diagonal`: only AP, AC, PP and CP
/// (`block_terms`), from pair data of order one or two; else every block, from pair data of order two. Correct before
/// fast: R matrices and BLAS products throughout. Its memory is `buffers` (`BraProductBuffers::new_in` for the same
/// pairs and `diagonal`).
#[allow(clippy::too_many_arguments)]
fn for_each_bra_product<'a, R, Sab, Scd>(
    pair_ab: &'a ShellPair<Sab>,
    bra: &PairData<Strategy<'a, Sab>>,
    pair_cd: &'a ShellPair<Scd>,
    ket: &PairData<Strategy<'a, Scd>>,
    diagonal: bool,
    buffers: BraProductBuffers<'_>,
    omega: f64,
    mut visit: impl FnMut(usize, &[f64]),
)
where
    R: RangeVersion,
    Sab: ShellKind,
    Scd: ShellKind,
{
    let (bra_split, ket_split) = (bra.split, ket.split);
    let lqn_ab = pair_ab.get_lqn_a() + pair_ab.get_lqn_b();
    let lqn_cd = pair_cd.get_lqn_a() + pair_cd.get_lqn_b();
    let (tuv, raised) = (hermite_count(lqn_ab), hermite_count(lqn_ab + 2));
    let sph_ab = pair_ab.get_degeneracy_a() * pair_ab.get_degeneracy_b();
    let sph_cd = pair_cd.get_degeneracy_a() * pair_cd.get_degeneracy_b();
    let (n_cont_c, n_cont_d) = ket.contraction.contracted();
    let cols_cd = sph_cd * n_cont_c * n_cont_d;
    let block = sph_ab * cols_cd;
    let n_prim_cd = pair_cd.get_products().len();
    let n_var_ket = if !ket_split { 1 } else if diagonal { 4 } else { 10 };

    let BraProductBuffers { rows, mut r_buffer, mut r, ket_prim, terms, scaled, gathered, out } = buffers;
    fill_shifted_rows(rows, lqn_ab);
    let rows_of = |shift: usize| &rows[shift * tuv..(shift + 1) * tuv];

    for (ip, ab) in pair_ab.get_products().iter().enumerate() {
        let (p, p_x) = (ab.get_p(), ab.get_product_centre());
        let a_over_p = bra.fraction[ip];
        for (jp, cd) in pair_cd.get_products().iter().enumerate() {
            construct_r_xxxx_prealloc::<R, _>(&mut r_buffer, &mut r, p, &p_x, lqn_ab + 2, cd.get_p(), &cd.get_product_centre(), lqn_cd, omega);
            let ket_block = ket.coefficients.view_sub_array4d(jp);
            let width = sph_cd * n_var_ket;
            let e_ket = MatrixView::<f64>::new(ket_block.dim0(), width, &ket_block.get_data()[..ket_block.dim0() * width]);
            r.view_as_matrix().matmul_prealloc(&e_ket, &mut MatrixViewMut::new(raised, width, &mut ket_prim[jp * raised * width..(jp + 1) * raised * width]));
        }
        {
            let per_product = Array3DView::new(raised * sph_cd, n_var_ket, n_prim_cd, &*ket_prim);
            let shape = (raised * sph_cd, n_cont_c * n_cont_d);
            ket.contraction.contract(&per_product.view_dim1_matrix(0), &mut term_mut(terms, 0, shape), 1.0);
            if ket_split {
                ket.contraction.contract_weighted(&per_product.view_dim1_matrix(0), &ket.fraction, &mut term_mut(terms, 1, shape));
                for axis in 0..3 {
                    ket.contraction.contract(&per_product.view_dim1_matrix(1 + axis), &mut term_mut(terms, 3 + axis, shape), 1.0);
                }
                if !diagonal {
                    for (jp, column) in scaled.chunks_exact_mut(raised * sph_cd).enumerate() {
                        scaled_into(per_product.view_dim1_matrix(0).column(jp), ket.fraction[jp], column, false);
                    }
                    ket.contraction.contract_weighted(&MatrixView::new(raised * sph_cd, n_prim_cd, &*scaled), &ket.fraction, &mut term_mut(terms, 2, shape));
                    for axis in 0..3 {
                        ket.contraction.contract_weighted(&per_product.view_dim1_matrix(1 + axis), &ket.fraction, &mut term_mut(terms, 6 + axis, shape));
                    }
                    for second in 0..6 {
                        ket.contraction.contract(&per_product.view_dim1_matrix(4 + second), &mut term_mut(terms, 9 + second, shape), 1.0);
                    }
                }
            }
        }
        let term_of = |index: usize| &terms[index * raised * cols_cd..(index + 1) * raised * cols_cd];

        // The bra: every block's terms, `alpha left^T K[rows]` added into the product's batches.
        out.fill(0.0);
        let add = |variant: usize, term: usize, shift: usize, alpha: f64, batch: usize| {
            gather_rows(term_of(term), raised, rows_of(shift), 1.0, gathered, false);
            let left = bra.coefficients.view_sub_array3d((variant, ip));
            left.view_as_matrix((1, 2)).view_transposed().matmul_scaled_prealloc(
                &MatrixView::new(tuv, cols_cd, &*gathered),
                &mut MatrixViewMut::new(sph_ab, cols_cd, &mut out[batch * block..(batch + 1) * block]),
                alpha,
                1.0,
            );
        };
        block_terms(bra_split, ket_split, a_over_p, diagonal, add);
        visit(ip, out);
    }
}

/// For each pair of `pairs`, the bound `Q'` with `|(d_X ab|cd)| <= Q' (cd|cd)^(1/2)` for X either centre of the pair
/// (both, for a pair on one atom), so that a quartet's derivatives with respect to any atom are bounded by
/// `Q'_ab Q_cd + Q_ab Q'_cd` (`Q` the Schwarz bounds). A centre derivative of a charge distribution is a charge
/// distribution, and the Coulomb metric is positive definite, so Cauchy-Schwarz gives `|(d_X ab|cd)| <=
/// (d_X ab|d_X ab)^(1/2) (cd|cd)^(1/2)`; `(d_X ab|d_X ab)` is a second derivative of `(ab|a'b')` with respect to the
/// bra's centre and the ket's same one: `[A_k C_k]` and `[B_k D_k]`, or `-[P_k P_k]` for a pair on one atom. The
/// largest over the pair's functions and the three axes, square-rooted. From each pair's data of order one (or two),
/// the blocks in a scope of `arena`.
pub(crate) fn derivative_schwarz_bounds<'a, 'p, R, S>(
    pairs: impl Iterator<Item = (&'a ShellPair<S>, &'p PairData<Strategy<'a, S>>)>,
    arena: &mut Arena<'_>,
    omega: f64,
) -> Result<Vec<f64>, ArenaError>
where
    R: RangeVersion,
    S: ShellKind,
    Strategy<'a, S>: 'p,
{
    pairs
        .map(|(pair, data)| {
            arena.scope(|scratch| {
                if S::PRIMITIVE {
                    let buffers = PrimitiveBoundBuffers::new_in(pair, data, scratch)?;
                    return Ok(primitive_derivative_bound::<R, S>(pair, data, buffers, omega));
                }
                let buffers = ContractedBoundBuffers::new_in(pair, data, scratch)?;
                Ok(contracted_derivative_bound::<R, S>(pair, data, buffers, omega))
            })
        })
        .collect()
}

/// The buffers `derivative_schwarz_bounds` takes for `pair` (the scratch query).
pub(crate) fn derivative_bound_buffers<'a, S: ShellKind>(pair: &'a ShellPair<S>, data: &PairData<Strategy<'a, S>>, arena: &Arena<'_>) -> Result<(), ArenaError> {
    if S::PRIMITIVE {
        PrimitiveBoundBuffers::new_in(pair, data, arena).map(|_| ())
    } else {
        ContractedBoundBuffers::new_in(pair, data, arena).map(|_| ())
    }
}

/// `contracted_derivative_bound`'s buffers: the per-product core's, the contraction coefficients and the sums.
struct ContractedBoundBuffers<'s> {
    products: BraProductBuffers<'s>,
    identity: &'s mut [f64],
    coefficients: &'s mut [f64],
    sums: &'s mut [f64],
}

impl<'s> ContractedBoundBuffers<'s> {
    fn new_in<'a, S: ShellKind>(pair: &'a ShellPair<S>, data: &PairData<Strategy<'a, S>>, arena: &Arena<'s>) -> Result<Self, ArenaError> {
        let sph = pair.get_degeneracy_a() * pair.get_degeneracy_b();
        let (n_a, n_b) = data.contraction.contracted();
        let n_prim = pair.get_products().len();
        let n_sets = 3 * if data.split { 2 } else { 1 };
        Ok(ContractedBoundBuffers {
            products: BraProductBuffers::new_in(pair, pair, data, true, arena)?,
            identity: arena.alloc::<f64>(n_prim * n_prim)?,
            coefficients: arena.alloc::<f64>(n_prim * n_a * n_b)?,
            sums: arena.alloc::<f64>(n_sets * sph * n_a * n_b)?,
        })
    }
}

/// `derivative_schwarz_bounds` of a contracted pair: the diagonal of `(d_X ab|d_X ab)` alone, one bra product at a time
/// (`for_each_bra_product` on the pair against itself, the ket contracted): each product's batches, weighted per
/// slot and axis (`diagonal_weights`), read at the ket columns of each bra contracted pair's own functions and added
/// with the product's contraction coefficient for that pair. Nothing of size `n_cont_ab^2` is formed.
fn contracted_derivative_bound<'a, R: RangeVersion, S: ShellKind>(pair: &'a ShellPair<S>, data: &PairData<Strategy<'a, S>>, buffers: ContractedBoundBuffers<'_>, omega: f64) -> f64 {
    let ContractedBoundBuffers { products, identity, coefficients, sums } = buffers;
    let (sph_a, sph_b) = (pair.get_degeneracy_a(), pair.get_degeneracy_b());
    let sph = sph_a * sph_b;
    let (n_a, n_b) = data.contraction.contracted();
    let n_cont = n_a * n_b;
    let n_prim = pair.get_products().len();
    let block = sph * sph * n_cont;
    // The contraction coefficient of product p for contracted pair c, `(n_prim, n_cont)`: the strategy's own
    // contraction of the identity.
    identity.fill(0.0);
    for p in 0..n_prim {
        identity[p + n_prim * p] = 1.0;
    }
    data.contraction.contract(&MatrixView::new(n_prim, n_prim, &*identity), &mut MatrixViewMut::new(n_prim, n_cont, &mut *coefficients), 1.0);
    // The weight sets: per axis, one per slot of the pair.
    let mut sets = [[0.0; N_BATCHES]; 6];
    let mut n_sets = 0;
    for k in 0..3 {
        for weights in diagonal_weights(data.split, k) {
            sets[n_sets] = weights;
            n_sets += 1;
        }
    }
    // Per weight set, the diagonal `(set, function pair, contracted pair)`.
    sums.fill(0.0);
    for_each_bra_product::<R, S, S>(pair, data, pair, data, true, products, omega, |ip, out| {
        for (set, weights) in sets[..n_sets].iter().enumerate() {
            for c in 0..n_cont {
                let coefficient = coefficients[ip + n_prim * c];
                if coefficient == 0.0 {
                    continue;
                }
                for f in 0..sph {
                    // Row f, the ket column of the same function f of contracted pair c.
                    let at = f + sph * (f + sph * c);
                    sums[(set * n_cont + c) * sph + f] += coefficient * diagonal(weights, out, block, at);
                }
            }
        }
    });
    sums.iter().fold(0.0_f64, |largest, &value| largest.max(value)).sqrt()
}

/// `derivative_schwarz_bounds` of a contracted pair the long way: the full contracted blocks of the pair against
/// itself (`second_blocks`, from pair data of order two), their diagonal read off. The reference
/// `contracted_derivative_bound` is checked against.
#[cfg(test)]
pub(super) fn derivative_bound_from_full_blocks<'a, R: RangeVersion, S: ShellKind>(pair: &'a ShellPair<S>, data: &PairData<Strategy<'a, S>>, arena: &Arena<'_>) -> Result<f64, ArenaError> {
    let blocks = second_blocks::<R, S, S>(pair, data, pair, data, arena, 0.0)?;
    let (sph_a, sph_b) = (pair.get_degeneracy_a(), pair.get_degeneracy_b());
    let (n_a, n_b) = data.contraction.contracted();
    let chunk = sph_a * sph_b * sph_a * sph_b * n_a * n_b * n_a * n_b;
    let mut largest: f64 = 0.0;
    for k in 0..3 {
        for weights in diagonal_weights(data.split, k) {
            for c_b in 0..n_b {
                for c_a in 0..n_a {
                    for i_b in 0..sph_b {
                        for i_a in 0..sph_a {
                            let at = i_a + sph_a * (i_b + sph_b * (i_a + sph_a * (i_b + sph_b * (c_a + n_a * (c_b + n_b * (c_a + n_a * c_b))))));
                            largest = largest.max(diagonal(&weights, blocks, chunk, at));
                        }
                    }
                }
            }
        }
    }
    Ok(largest.sqrt())
}

/// `sum_batch weights[batch] blocks[batch chunk + at]`.
fn diagonal(weights: &[f64; N_BATCHES], blocks: &[f64], chunk: usize, at: usize) -> f64 {
    weights.iter().enumerate().filter(|(_, w)| **w != 0.0).map(|(batch, w)| w * blocks[batch * chunk + at]).sum()
}

/// `derivative_schwarz_bounds` of a pair of primitives (every product its own function): each product's diagonal
/// quartet alone, the product against itself, the diagonal blocks as `second_blocks` forms them.
/// `primitive_derivative_bound`'s buffers, the only memory it takes.
struct PrimitiveBoundBuffers<'s> {
    rows: &'s mut [usize],
    r_buffer: HermiteCoulombIntegrals4D<'s>,
    r: BufferedRMatrix<'s, f64>,
    products: &'s mut [f64],
    terms: &'s mut [f64],
    gathered: &'s mut [f64],
    out: &'s mut [f64],
}

impl<'s> PrimitiveBoundBuffers<'s> {
    fn new_in<'a, S: ShellKind>(pair: &'a ShellPair<S>, data: &PairData<Strategy<'a, S>>, arena: &Arena<'s>) -> Result<Self, ArenaError> {
        let split = data.split;
        let lqn = pair.get_lqn_a() + pair.get_lqn_b();
        let (tuv, raised) = (hermite_count(lqn), hermite_count(lqn + 2));
        let sph = pair.get_degeneracy_a() * pair.get_degeneracy_b();
        let n_var = if split { 4 } else { 1 };
        Ok(PrimitiveBoundBuffers {
            rows: shifted_rows_in(lqn, arena)?,
            r_buffer: HermiteCoulombIntegrals4D::from_arena(2 * lqn + 2, arena)?,
            r: BufferedRMatrix::<f64>::from_arena(lqn + 2, lqn, arena)?,
            products: arena.alloc::<f64>(raised * sph * n_var)?,
            terms: arena.alloc::<f64>(raised * sph * if split { 6 } else { 1 })?,
            gathered: arena.alloc::<f64>(tuv * sph)?,
            out: arena.alloc::<f64>(sph * sph * N_BATCHES)?,
        })
    }
}

fn primitive_derivative_bound<'a, R: RangeVersion, S: ShellKind>(pair: &'a ShellPair<S>, data: &PairData<Strategy<'a, S>>, buffers: PrimitiveBoundBuffers<'_>, omega: f64) -> f64 {
    let split = data.split;
    let lqn = pair.get_lqn_a() + pair.get_lqn_b();
    let (tuv, raised) = (hermite_count(lqn), hermite_count(lqn + 2));
    let (sph_a, sph_b) = (pair.get_degeneracy_a(), pair.get_degeneracy_b());
    let sph = sph_a * sph_b;
    let n_var = if split { 4 } else { 1 };
    let chunk = sph * sph;
    let PrimitiveBoundBuffers { rows, mut r_buffer, mut r, products, terms, gathered, out } = buffers;
    fill_shifted_rows(rows, lqn);
    let mut largest: f64 = 0.0;
    for (ip, product) in pair.get_products().iter().enumerate() {
        let (p, p_x) = (product.get_p(), product.get_product_centre());
        construct_r_xxxx_prealloc::<R, _>(&mut r_buffer, &mut r, p, &p_x, lqn + 2, p, &p_x, lqn, omega);
        let ket_block = data.coefficients.view_sub_array4d(ip);
        let e_ket = MatrixView::<f64>::new(ket_block.dim0(), sph * n_var, &ket_block.get_data()[..ket_block.dim0() * sph * n_var]);
        r.view_as_matrix().matmul_prealloc(&e_ket, &mut MatrixViewMut::new(raised, sph * n_var, &mut *products));
        // The ket terms of this one product (its own function: no contraction), as `block_terms` reads them.
        let fraction = data.fraction[ip];
        let variant = |var: usize| &products[var * raised * sph..(var + 1) * raised * sph];
        // The terms the diagonal blocks read (`block_terms`): E, (c/q) E, dE/dS_x,y,z; (c/q)^2 E (term 2) is not read.
        let mut sources = [(0_usize, 1.0_f64); 6];
        let n_sources = if split { 6 } else { 1 };
        if split {
            sources[1] = (0, fraction);
            sources[2] = (0, 0.0);
            for axis in 0..3 {
                sources[3 + axis] = (1 + axis, 1.0);
            }
        }
        for (term, &(var, weight)) in terms.chunks_exact_mut(raised * sph).zip(&sources[..n_sources]) {
            scaled_into(variant(var), weight, term, false);
        }
        out.fill(0.0);
        block_terms(split, split, fraction, true, |variant, term, shift, alpha, batch| {
            gather_rows(&terms[term * raised * sph..(term + 1) * raised * sph], raised, &rows[shift * tuv..(shift + 1) * tuv], 1.0, gathered, false);
            let left = data.coefficients.view_sub_array3d((variant, ip));
            left.view_as_matrix((1, 2)).view_transposed().matmul_scaled_prealloc(
                &MatrixView::new(tuv, sph, &*gathered),
                &mut MatrixViewMut::new(sph, sph, &mut out[batch * chunk..(batch + 1) * chunk]),
                alpha,
                1.0,
            );
        });
        for k in 0..3 {
            for weights in diagonal_weights(split, k) {
                for i_b in 0..sph_b {
                    for i_a in 0..sph_a {
                        largest = largest.max(diagonal(&weights, out, chunk, i_a + sph_a * (i_b + sph_b * (i_a + sph_a * i_b))));
                    }
                }
            }
        }
    }
    largest.sqrt()
}
