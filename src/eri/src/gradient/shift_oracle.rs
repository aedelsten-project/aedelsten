// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! An oracle for the derivatives of the two-electron integrals with respect to the atoms, from the value batches
//! alone: along one axis, d/dA of `G_i = (x - A)^i exp(-a (x - A)^2)` is `2a G_(i+1) - i G_(i-1)`, so any derivative of
//! a primitive integral `(ab|cd)` is a short sum of unnormalized value integrals at shifted Cartesian exponents (the
//! second applies the first twice); a derivative with respect to an atom is the sum over the shells on it. No
//! derivative code enters it.
//!
//! The oracle is checked against the gradient walk's quartet kernel (`compute_quartet`) directly, and its second
//! derivatives against central differences of that kernel: four primitive shells of one Cartesian function each, on
//! four atoms, on two (both pairs on one atom each, the merged slots) and with a pair on one atom.

use arrays::array4d::Array4DTrait;
use arrays::array6d::Array6DTrait;
use arrays::array8d::Array8DTrait;
use arrays::matrix::{Matrix, MatrixTrait};
use arrays::scratch::{Scratch, ByteUnit};
use basis::basis_parser::InputBasisShell;
use basis::contraction::Contracted;
use basis::shell_pairs::{derive_shell_pairs, ShellPair};
use basis::shell_types::{BasisShell, HarmonicType};
use mmd::hermite_coulomb_integrals::FullRange;
use mmd::ketside_hermite_coefficients::compute_ketside_hermite_coefficients;

use super::second_order::{compute_second_quartet, IntegralSecondDerivatives, SecondDerivativeSink};
use super::{compute_quartet, pair_data, pair_data_to_order, BraRows, GradientCentres, GradientSink, IntegralDerivatives};
use crate::batch::{compute_eri_batch, ERIBatchType};
use crate::distribute::for_each_unique_general;

/// The Cartesian exponents of the four functions of an integral, in the order of the quartet's shells.
type Components = [[usize; 3]; 4];

/// Four primitive shells: their exponents, the atoms they sit on (zero-based, consecutive) and those atoms'
/// positions.
#[derive(Clone, Copy, Debug)]
struct Quartet {
    exponents: [f64; 4],
    atoms: [usize; 4],
    positions: [[f64; 3]; 4],
}

impl Quartet {
    fn n_atoms(&self) -> usize {
        self.atoms.iter().max().unwrap() + 1
    }

    /// The quartet with atom `atom` (every shell on it) moved by `step` along `axis`.
    fn displaced(&self, atom: usize, axis: usize, step: f64) -> Quartet {
        let mut moved = *self;
        for (position, &on) in moved.positions.iter_mut().zip(&self.atoms) {
            if on == atom {
                position[axis] += step;
            }
        }
        moved
    }
}

/// Four atoms; both pairs on one atom each (the merged bra and ket); the bra on one atom, the ket split.
fn quartets() -> [Quartet; 3] {
    let (a, b, c, d) = ([0.0, 0.0, 0.0], [0.3, -1.1, 0.8], [1.4, 0.6, -0.9], [-0.7, 0.9, 1.3]);
    [
        Quartet { exponents: [1.3, 0.7, 0.45, 2.1], atoms: [0, 1, 2, 3], positions: [a, b, c, d] },
        Quartet { exponents: [0.9, 3.2, 0.6, 1.7], atoms: [0, 0, 1, 1], positions: [a, a, c, c] },
        Quartet { exponents: [5.5, 0.25, 1.1, 0.8], atoms: [0, 0, 1, 2], positions: [b, b, a, d] },
    ]
}

/// The Cartesian exponents of a shell of angular momentum l, in the kernels' order.
fn components(l: usize) -> Vec<[usize; 3]> {
    cart2sph::get_cartesian_quantum_numbers_m(l).iter_col().map(|lmn| [lmn[0], lmn[1], lmn[2]]).collect()
}

fn norm(exponent: f64, lmn: [usize; 3]) -> f64 {
    cart2sph::compute_cart_gaussian_norm(exponent, lmn[0], lmn[1], lmn[2])
}

/// The quartet's shells of angular momenta `l`: Cartesian, one normalized primitive each, their first AOs in order.
fn shells(quartet: &Quartet, l: [usize; 4]) -> Vec<BasisShell<Contracted>> {
    let mut shells: Vec<_> = (0..4)
        .flat_map(|k| {
            let input = InputBasisShell::new(l[k], vec![quartet.exponents[k]].into(), Matrix::new_from_vec(1, 1, vec![1.0].into()), None, None);
            basis::convert_input_basis_to_used_basis::<Contracted>(&[input], |_| HarmonicType::Cartesian, quartet.atoms[k], quartet.positions[k])
        })
        .collect();
    basis::fix_basis_indices(&mut shells);
    shells
}

/// The pair of shells `first` and `second` (in either order) among `pairs`.
fn pair_of<'p>(pairs: &'p [ShellPair<Contracted>], shells: &[BasisShell<Contracted>], first: usize, second: usize) -> &'p ShellPair<Contracted> {
    let wanted = [shells[first].get_basis_index(), shells[second].get_basis_index()];
    pairs
        .iter()
        .find(|pair| {
            let (a, b) = (pair.get_idx_a(), pair.get_idx_b());
            (a, b) == (wanted[0], wanted[1]) || (b, a) == (wanted[0], wanted[1])
        })
        .unwrap()
}

/// Which shell, and which of its Cartesian functions, each AO of `shells` is.
struct AoMap {
    shell: Vec<usize>,
    component: Vec<[usize; 3]>,
}

impl AoMap {
    fn new(shells: &[BasisShell<Contracted>]) -> Self {
        let (mut shell, mut component) = (Vec::new(), Vec::new());
        for (k, s) in shells.iter().enumerate() {
            assert_eq!(s.get_basis_index(), shell.len());
            for lmn in components(s.get_lqn()) {
                shell.push(k);
                component.push(lmn);
            }
        }
        Self { shell, component }
    }

    /// The components of the integral over the AOs `aos` (one on each shell, in any order).
    fn key(&self, aos: [usize; 4]) -> Components {
        let mut key = [[usize::MAX; 3]; 4];
        for ao in aos {
            key[self.shell[ao]] = self.component[ao];
        }
        assert!(key.iter().all(|lmn| lmn[0] != usize::MAX), "an integral over four distinct shells");
        key
    }
}

/// The normalized value integrals `(AB|CD)` of the quartet's shells of angular momenta `l`, by the energy batch, at
/// `position(components)`.
fn values(scratch: &mut Scratch, quartet: &Quartet, l: [usize; 4]) -> Vec<f64> {
    let shells = shells(quartet, l);
    let pairs = derive_shell_pairs(&shells, None).unwrap();
    let (ab, cd) = (pair_of(&pairs, &shells, 0, 1), pair_of(&pairs, &shells, 2, 3));
    let map = AoMap::new(&shells);
    let mut arena = scratch.get_arena();
    let bra = compute_ketside_hermite_coefficients(ab, &mut arena).unwrap();
    let ket = compute_ketside_hermite_coefficients(cd, &mut arena).unwrap();
    let bra_view = bra.mmd_transpose_in(&arena).unwrap();
    let batch = compute_eri_batch::<Contracted, Contracted, FullRange>(ab, cd, &bra_view, &ket, &mut arena, 0.0).unwrap();
    let data = match &batch {
        ERIBatchType::FourD(b) => b.get_data(),
        ERIBatchType::SixD(b) => b.get_data(),
        ERIBatchType::EightD(b) => b.get_data(),
    };
    let dims = [ab.get_degeneracy_a(), ab.get_degeneracy_b(), cd.get_degeneracy_a(), cd.get_degeneracy_b(), 1, 1, 1, 1];
    let mut out = vec![0.0; n_keys(l)];
    for_each_unique_general(dims, ab, cd, |i, j, k, m, offset| {
        out[position(&map.key([i, j, k, m]))] = data[offset];
    });
    out
}

/// The number of integrals of a quartet of angular momenta `l`.
fn n_keys(l: [usize; 4]) -> usize {
    l.iter().map(|&l| (l + 1) * (l + 2) / 2).product()
}

/// Records every derivative a quartet emits, at `3 (n_atoms position(components) + atom) + axis`; zero for an atom
/// not emitted.
struct Recorder {
    map: AoMap,
    n_atoms: usize,
    atoms: Vec<usize>,
    derivatives: Vec<f64>,
}

impl GradientSink for Recorder {
    fn select_atoms(&mut self, atoms: &[usize]) {
        self.atoms = atoms.to_vec();
    }

    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, derivatives: IntegralDerivatives<'_>) {
        let at = 3 * self.n_atoms * position(&self.map.key([i, j, k, l]));
        for q in 0..derivatives.len() {
            let slot = if self.n_atoms == 1 { 0 } else { self.atoms[q / 3] };
            self.derivatives[at + 3 * slot + q % 3] = derivatives.get(q);
        }
    }
}

/// The derivatives of `(AB|CD)` with respect to every atom by the gradient walk's quartet kernel, as `Recorder` holds
/// them.
fn gradient(scratch: &mut Scratch, quartet: &Quartet, l: [usize; 4]) -> Vec<f64> {
    let shells = shells(quartet, l);
    let pairs = derive_shell_pairs(&shells, None).unwrap();
    let (ab, cd) = (pair_of(&pairs, &shells, 0, 1), pair_of(&pairs, &shells, 2, 3));
    let n_atoms = quartet.n_atoms();
    let mut recorder = Recorder { map: AoMap::new(&shells), n_atoms, atoms: Vec::new(), derivatives: vec![0.0; 3 * n_atoms * n_keys(l)] };
    let mut arena = scratch.get_arena();
    let bra = pair_data(ab, &mut arena).unwrap();
    let ket = pair_data(cd, &mut arena).unwrap();
    let bra_rows = BraRows::new_in(ab, &arena).unwrap();
    compute_quartet::<_, FullRange, Contracted, Contracted>(&mut recorder, ab, &bra, &bra_rows, cd, &ket, GradientCentres::all(quartet.n_atoms()), &arena, 0.0).unwrap();
    recorder.derivatives
}

/// The position of the Cartesian function `lmn` in its shell, in the kernels' order (x's exponent descending, then
/// y's): `r (r + 1) / 2 + n` with `r = l - l_x` (`cartesian_order_is_the_kernels` checks it).
fn cartesian_index(lmn: [usize; 3]) -> usize {
    let r = lmn[1] + lmn[2];
    r * (r + 1) / 2 + lmn[2]
}

/// The highest angular momentum the oracle reads on a shell: two above the highest tested.
const L_READ: usize = 6;

/// The unnormalized primitive integrals of one quartet, made as they are asked for: per quartet of angular momenta
/// (at `l_a + (L_READ + 1) (l_b + ...)`), a dense table over the four shells' Cartesian functions, the first shell's
/// fastest.
struct Unnormalized {
    quartet: Quartet,
    tables: Vec<Option<Vec<f64>>>,
}

impl Unnormalized {
    fn new(quartet: Quartet) -> Self {
        Self { quartet, tables: vec![None; (L_READ + 1).pow(4)] }
    }

    fn get(&mut self, scratch: &mut Scratch, key: &Components) -> f64 {
        let l = key.map(|lmn| lmn[0] + lmn[1] + lmn[2]);
        let slot = l.iter().rev().fold(0, |index, &l| index * (L_READ + 1) + l);
        let quartet = self.quartet;
        let table = self.tables[slot].get_or_insert_with(|| {
            let mut table = vec![0.0; l.iter().map(|&l| (l + 1) * (l + 2) / 2).product()];
            let normalized = values(scratch, &quartet, l);
            for key in keys(l) {
                let scale: f64 = (0..4).map(|k| norm(quartet.exponents[k], key[k])).product();
                table[position(&key)] = normalized[position(&key)] / scale;
            }
            table
        });
        table[position(key)]
    }

    /// The derivative of the normalized integral `key` with respect to each shell's centre along each of its `axes`
    /// (at most two per shell).
    fn derivative(&mut self, scratch: &mut Scratch, key: &Components, axes: &[&[usize]; 4]) -> f64 {
        let terms: [Terms; 4] = std::array::from_fn(|k| shifted(self.quartet.exponents[k], key[k], axes[k]));
        let mut sum = 0.0;
        for (a, ca) in terms[0].iter() {
            for (b, cb) in terms[1].iter() {
                for (c, cc) in terms[2].iter() {
                    for (d, cd) in terms[3].iter() {
                        sum += ca * cb * cc * cd * self.get(scratch, &[a, b, c, d]);
                    }
                }
            }
        }
        let scale: f64 = (0..4).map(|k| norm(self.quartet.exponents[k], key[k])).product();
        scale * sum
    }

    /// The derivative with respect to the atoms and axes of `along` (one or two (atom, axis)): the sum over the
    /// shells on each atom.
    fn atom_derivative(&mut self, scratch: &mut Scratch, key: &Components, along: &[(usize, usize)]) -> f64 {
        let atoms = self.quartet.atoms;
        let mut sum = 0.0;
        let (first_atom, first_axis) = along[0];
        for k in (0..4).filter(|&k| atoms[k] == first_atom) {
            match along.get(1) {
                None => {
                    let mut axes: [&[usize]; 4] = [&[]; 4];
                    let single = [first_axis];
                    axes[k] = &single;
                    sum += self.derivative(scratch, key, &axes);
                }
                Some(&(second_atom, second_axis)) => {
                    for m in (0..4).filter(|&m| atoms[m] == second_atom) {
                        let (one, other, both) = ([first_axis], [second_axis], [first_axis, second_axis]);
                        let mut axes: [&[usize]; 4] = [&[]; 4];
                        if k == m {
                            axes[k] = &both;
                        } else {
                            axes[k] = &one;
                            axes[m] = &other;
                        }
                        sum += self.derivative(scratch, key, &axes);
                    }
                }
            }
        }
        sum
    }
}

/// The index of `key` in a dense table of its angular momenta.
fn position(key: &Components) -> usize {
    key.iter().rev().fold(0, |index, lmn| {
        let l = lmn[0] + lmn[1] + lmn[2];
        index * ((l + 1) * (l + 2) / 2) + cartesian_index(*lmn)
    })
}

/// At most four (exponents, coefficient) terms: a Cartesian Gaussian differentiated along up to two axes.
struct Terms {
    items: [([usize; 3], f64); 4],
    len: usize,
}

impl Terms {
    fn iter(&self) -> impl Iterator<Item = ([usize; 3], f64)> + '_ {
        self.items[..self.len].iter().copied()
    }
}

/// The derivative of the unnormalized Cartesian Gaussian of exponents `lmn` and exponent `exponent` with respect to
/// its centre along each of `axes` (at most two) in turn.
fn shifted(exponent: f64, lmn: [usize; 3], axes: &[usize]) -> Terms {
    let mut terms = Terms { items: [([0; 3], 0.0); 4], len: 1 };
    terms.items[0] = (lmn, 1.0);
    for &k in axes {
        let mut next = Terms { items: [([0; 3], 0.0); 4], len: 0 };
        for (l, c) in terms.iter() {
            let mut up = l;
            up[k] += 1;
            next.items[next.len] = (up, 2.0 * exponent * c);
            next.len += 1;
            if l[k] > 0 {
                let mut down = l;
                down[k] -= 1;
                next.items[next.len] = (down, -(l[k] as f64) * c);
                next.len += 1;
            }
        }
        terms = next;
    }
    terms
}

#[test]
fn cartesian_order_is_the_kernels() {
    for l in 0..=6 {
        for (index, lmn) in components(l).into_iter().enumerate() {
            assert_eq!(cartesian_index(lmn), index, "l {l}, {lmn:?}");
        }
    }
}

/// Every integral's components for the angular momenta `l`, in a fixed order.
fn keys(l: [usize; 4]) -> Vec<Components> {
    let lists: Vec<_> = l.iter().map(|&l| components(l)).collect();
    let mut out = Vec::new();
    for &a in &lists[0] {
        for &b in &lists[1] {
            for &c in &lists[2] {
                for &d in &lists[3] {
                    out.push([a, b, c, d]);
                }
            }
        }
    }
    out
}

/// The largest difference of `pairs` (expected, got) relative to the largest expected value, at least `floor`.
fn relative(pairs: &[(f64, f64)], floor: f64) -> f64 {
    let scale = pairs.iter().fold(floor, |m, (expected, _)| m.max(expected.abs()));
    let scale = if scale > 0.0 { scale } else { 1.0 };
    pairs.iter().fold(0.0_f64, |m, (expected, got)| m.max((expected - got).abs())) / scale
}

/// A scratch buffer sized for the largest quartet the checks run up to `l_max`: the value batches up to two above on
/// one shell or one above on two, and the quartet kernel at `l_max`.
fn sized_scratch(l_max: usize) -> Scratch {
    let mut probe = Scratch::new(256, ByteUnit::MiB).unwrap();
    let quartet = quartets()[0];
    for l in [[l_max + 2, l_max, l_max, l_max], [l_max + 1, l_max + 1, l_max, l_max], [l_max + 1, l_max, l_max + 1, l_max]] {
        for rotation in 0..4 {
            let l = std::array::from_fn(|k| l[(k + rotation) % 4]);
            values(&mut probe, &quartet, l);
        }
    }
    gradient(&mut probe, &quartet, [l_max; 4]);
    Scratch::new(probe.peak_bytes().div_ceil(1024), ByteUnit::KiB).unwrap()
}

/// Every quartet of angular momenta up to `l_max`.
fn momenta(l_max: usize) -> impl Iterator<Item = [usize; 4]> {
    (0..(l_max + 1).pow(4)).map(move |i| std::array::from_fn(|k| (i / (l_max + 1).pow(k as u32)) % (l_max + 1)))
}

/// The quartet kernel's first derivatives against the oracle, every quartet of angular momenta up to g.
#[test]
fn the_shift_oracle_gives_the_eri_first_derivatives() {
    let l_max = 4;
    let scratch = &mut sized_scratch(l_max);
    let mut largest: f64 = 0.0;
    for quartet in quartets() {
        let mut oracle = Unnormalized::new(quartet);
        for l in momenta(l_max) {
            let kernel = gradient(scratch, &quartet, l);
            let mut pairs = Vec::new();
            let mut floor: f64 = 0.0;
            for key in keys(l) {
                floor = floor.max(oracle.get(scratch, &key).abs() * (0..4).map(|k| norm(quartet.exponents[k], key[k])).product::<f64>());
                for atom in 0..quartet.n_atoms() {
                    for axis in 0..3 {
                        let expected = oracle.atom_derivative(scratch, &key, &[(atom, axis)]);
                        pairs.push((expected, kernel[3 * (quartet.n_atoms() * position(&key) + atom) + axis]));
                    }
                }
            }
            let difference = relative(&pairs, floor);
            largest = largest.max(difference);
            assert!(difference < 1e-10, "l {l:?}, {quartet:?}: {difference:e}");
        }
    }
    eprintln!("ERI first derivatives against the oracle: largest relative difference {largest:e}");
}

/// The oracle's second derivatives against central differences of the quartet kernel, every quartet of angular
/// momenta up to f.
#[test]
fn the_shift_oracle_gives_the_eri_second_derivatives() {
    let l_max = 3;
    let scratch = &mut sized_scratch(l_max);
    let mut largest: f64 = 0.0;
    for quartet in quartets() {
        let mut oracle = Unnormalized::new(quartet);
        let h = 2e-3 / quartet.exponents.iter().fold(1.0_f64, |m, &e| m.max(e)).sqrt();
        for l in momenta(l_max) {
            let mut pairs = Vec::new();
            let mut floor: f64 = 0.0;
            let keys = keys(l);
            for key in &keys {
                floor = floor.max(oracle.get(scratch, key).abs() * (0..4).map(|k| norm(quartet.exponents[k], key[k])).product::<f64>());
            }
            for by in 0..quartet.n_atoms() {
                for y in 0..3 {
                    let at = |scratch: &mut Scratch, steps: f64| gradient(scratch, &quartet.displaced(by, y, steps * h), l);
                    let (m2, m1, p1, p2) = (at(scratch, -2.0), at(scratch, -1.0), at(scratch, 1.0), at(scratch, 2.0));
                    for key in &keys {
                        for atom in 0..quartet.n_atoms() {
                            for x in 0..3 {
                                let at = 3 * (quartet.n_atoms() * position(key) + atom) + x;
                                let difference = (m2[at] - 8.0 * m1[at] + 8.0 * p1[at] - p2[at]) / (12.0 * h);
                                let expected = oracle.atom_derivative(scratch, key, &[(by, y), (atom, x)]);
                                pairs.push((expected, difference));
                            }
                        }
                    }
                }
            }
            let difference = relative(&pairs, floor);
            largest = largest.max(difference);
            assert!(difference < 1e-8, "l {l:?}, {quartet:?}: {difference:e}");
        }
    }
    eprintln!("ERI second derivatives against differences of the kernel: largest relative difference {largest:e}");
}

/// Records every second derivative a quartet emits, at `(3 n_atoms)^2 position(components) + (3 atom_y + y) 3 n_atoms
/// + 3 atom_x + x`; zero for an atom not emitted.
struct SecondRecorder {
    map: AoMap,
    n_atoms: usize,
    atoms: Vec<usize>,
    derivatives: Vec<f64>,
}

impl SecondDerivativeSink for SecondRecorder {
    fn select_atoms(&mut self, atoms: &[usize]) {
        self.atoms = atoms.to_vec();
    }

    fn distribute(&mut self, i: usize, j: usize, k: usize, l: usize, derivatives: IntegralSecondDerivatives<'_>) {
        let n = 3 * self.n_atoms;
        let at = n * n * position(&self.map.key([i, j, k, l]));
        for q_y in 0..derivatives.components() {
            for q_x in 0..derivatives.components() {
                let row = 3 * self.atoms[q_y / 3] + q_y % 3;
                let column = 3 * self.atoms[q_x / 3] + q_x % 3;
                self.derivatives[at + row * n + column] = derivatives.get(q_y, q_x);
            }
        }
    }
}

/// The second derivatives of `(AB|CD)` with respect to every pair of atom coordinates by the second-order quartet
/// kernel, as `SecondRecorder` holds them.
fn second(scratch: &mut Scratch, quartet: &Quartet, l: [usize; 4]) -> Vec<f64> {
    let shells = shells(quartet, l);
    let pairs = derive_shell_pairs(&shells, None).unwrap();
    let (ab, cd) = (pair_of(&pairs, &shells, 0, 1), pair_of(&pairs, &shells, 2, 3));
    let n_atoms = quartet.n_atoms();
    let n = 3 * n_atoms;
    let mut recorder = SecondRecorder { map: AoMap::new(&shells), n_atoms, atoms: Vec::new(), derivatives: vec![0.0; n * n * n_keys(l)] };
    let mut arena = scratch.get_arena();
    let bra = pair_data_to_order(ab, &mut arena, 2).unwrap();
    let ket = pair_data_to_order(cd, &mut arena, 2).unwrap();
    compute_second_quartet::<_, FullRange, Contracted, Contracted>(&mut recorder, ab, &bra, cd, &ket, &arena, 0.0).unwrap();
    recorder.derivatives
}

/// The second-order quartet kernel against the oracle, every pair of atom coordinates, every quartet of angular
/// momenta up to f.
#[test]
fn the_shift_oracle_gives_the_eri_second_derivatives_of_the_second_order_kernel() {
    let l_max = 3;
    // Sized for the value batches the oracle reads and the second-order kernel's largest quartet.
    let probe = sized_scratch(l_max);
    let mut peak = probe.peak_bytes();
    let mut big = Scratch::new(256, ByteUnit::MiB).unwrap();
    second(&mut big, &quartets()[0], [l_max; 4]);
    peak = peak.max(big.peak_bytes());
    drop((probe, big));
    let scratch = &mut Scratch::new(peak.div_ceil(1024), ByteUnit::KiB).unwrap();
    let mut largest: f64 = 0.0;
    for quartet in quartets() {
        let mut oracle = Unnormalized::new(quartet);
        let n = 3 * quartet.n_atoms();
        for l in momenta(l_max) {
            let kernel = second(scratch, &quartet, l);
            let mut pairs = Vec::new();
            let mut floor: f64 = 0.0;
            for key in keys(l) {
                floor = floor.max(oracle.get(scratch, &key).abs() * (0..4).map(|k| norm(quartet.exponents[k], key[k])).product::<f64>());
                for row in 0..n {
                    for column in 0..n {
                        let expected = oracle.atom_derivative(scratch, &key, &[(row / 3, row % 3), (column / 3, column % 3)]);
                        pairs.push((expected, kernel[n * n * position(&key) + row * n + column]));
                    }
                }
            }
            let difference = relative(&pairs, floor);
            largest = largest.max(difference);
            assert!(difference < 1e-10, "l {l:?}, {quartet:?}: {difference:e}");
        }
    }
    eprintln!("second-order ERI kernel against the oracle: largest relative difference {largest:e}");
}

