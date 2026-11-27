// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Shell pairs: two shells, their primitive products and their contraction step.

use arrays::matrix::*;
use cart2sph;

use crate::contraction::{Contracted, ContractionData, ShellKind, Strategy};
use crate::shell_types;

/// Which shells of a pair are spherical.
pub enum SphericalTransformationType {
    /// Both Cartesian.
    None,
    /// a Cartesian, b spherical.
    CS,
    /// a spherical, b Cartesian.
    SC,
    /// Both spherical.
    SS,
}

/// Position of the contracted pair `(a, b)` in a batch's contracted-pair dimension: `a + n_a b`, or, for a
/// same-shell pair (`a <= b` only, `b` outer), `b(b+1)/2 + a`.
#[inline(always)]
pub fn contracted_pair_slot(single_centre: bool, n_a: usize, a: usize, b: usize) -> usize {
    if single_centre { b * (b + 1) / 2 + a } else { a + n_a * b }
}

/// A pair of two blocks of primitives (one shell each) and what its kind stores for the contraction step: the
/// contraction matrix (`Contracted`, `Dense`) or nothing (`Uncontracted`, `Identity`: every stored primitive
/// product is its own pair, in the order of `contracted_pair_slot`).
#[derive(Debug)]
pub struct ShellPair<S: ShellKind = Contracted> {
    lqn_a: usize,
    lqn_b: usize,
    n_prim_a: usize,
    n_prim_b: usize,
    n_cont_a: usize,
    n_cont_b: usize,
    harmonic_a: shell_types::HarmonicType,
    harmonic_b: shell_types::HarmonicType,
    idx_a: usize,
    idx_b: usize,
    centre_idx_a: usize,
    centre_idx_b: usize,
    x: f64, // B - A
    y: f64,
    z: f64,
    single_centre: bool,
    contraction: S::PairData,
    products: Box<[S::Product]>,
}


impl<S: ShellKind> ShellPair<S> {
    /// The contraction step.
    pub fn contraction(&self) -> Strategy<'_, S> {
        self.contraction.strategy(self.get_contracted_pair_dims())
    }
    /// The contraction matrix, `None` for a kind without one.
    #[inline(always)]
    pub fn contraction_matrix(&self) -> Option<&Matrix<f64>> {
        self.contraction.matrix()
    }
    /// The contracted functions of shell a.
    pub fn get_n_cont_a(&self) -> usize {
        self.n_cont_a
    }
    /// The contracted functions of shell b.
    pub fn get_n_cont_b(&self) -> usize {
        self.n_cont_b
    }
    /// The contracted-pair dimensions of a batch: `(n_cont_a, n_cont_b)`, `(n(n+1)/2, 1)` for a same-shell pair
    /// (its unique contracted pairs, see `contracted_pair_slot`), `(n_products, 1)` for a pair with screened-out
    /// products (the stored ones).
    pub fn get_contracted_pair_dims(&self) -> (usize, usize) {
        match self.contraction.placement() {
            Some(_) => (self.products.len(), 1),
            None if self.single_centre => (self.n_cont_a * (self.n_cont_a + 1) / 2, 1),
            None => (self.n_cont_a, self.n_cont_b),
        }
    }
    /// The position of the contracted pair `(a, b)` in a batch's contracted-pair dimension, `None` if it was
    /// screened out.
    #[inline(always)]
    pub fn pair_slot(&self, a: usize, b: usize) -> Option<usize> {
        match self.contraction.placement() {
            Some(placement) => {
                let slot = placement[a + self.n_cont_a * b];
                (slot != u32::MAX).then_some(slot as usize)
            }
            None => Some(contracted_pair_slot(self.single_centre, self.n_cont_a, a, b)),
        }
    }
    /// The angular momentum of shell a.
    pub fn get_lqn_a(&self) -> usize {
        self.lqn_a
    }
    /// The angular momentum of shell b.
    pub fn get_lqn_b(&self) -> usize {
        self.lqn_b
    }
    /// The harmonic type of shell a.
    pub fn get_harmonic_a(&self) -> shell_types::HarmonicType {
        self.harmonic_a
    }
    /// The harmonic type of shell b.
    pub fn get_harmonic_b(&self) -> shell_types::HarmonicType {
        self.harmonic_b
    }
    /// The first AO of shell a.
    pub fn get_idx_a(&self) -> usize {
        self.idx_a
    }
    /// The first AO of shell b.
    pub fn get_idx_b(&self) -> usize {
        self.idx_b
    }
    /// The stored primitive products, b outer.
    pub fn get_products(&self) -> &[S::Product] {
        &self.products
    }
    /// `B - A`.
    pub fn get_separation(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    /// The centre of shell a.
    pub fn get_centre_idx_a(&self) -> usize {
        self.centre_idx_a
    }
    /// The centre of shell b.
    pub fn get_centre_idx_b(&self) -> usize {
        self.centre_idx_b
    }
    /// Whether shell a *is* shell b (the same shell, not just the same atom): idx_a == idx_b.
    pub fn get_single_centre(&self) -> bool {
        self.single_centre
    }
    /// The primitives of shell a.
    pub fn get_n_prim_a(&self) -> usize {
        self.n_prim_a
    }
    /// The primitives of shell b.
    pub fn get_n_prim_b(&self) -> usize {
        self.n_prim_b
    }

    /// The pair's workload estimate: Cartesian components of a times those of b times the stored products.
    pub fn estimate_workload(&self) -> usize {
        let lqn_a = self.get_lqn_a();
        let lqn_b = self.get_lqn_b();
        let n_prim_pairs = self.get_products().len();
        let shell_size_1 = (lqn_a + 1) * (lqn_a + 2) / 2;
        let shell_size_2 = (lqn_b + 1) * (lqn_b + 2) / 2;
        shell_size_1 * shell_size_2 * n_prim_pairs
    }
    /// The pair's components: those of shell a times those of shell b, in their harmonic bases (an s-s pair 1).
    pub fn components(&self) -> usize {
        self.get_degeneracy_a() * self.get_degeneracy_b()
    }
    /// The components of shell a, Cartesian or spherical by its harmonic type.
    pub fn get_degeneracy_a(&self) -> usize {
        match self.get_harmonic_a() {
            shell_types::HarmonicType::Cartesian => cart2sph::compute_cartesian_degeneracy(self.get_lqn_a()),
            shell_types::HarmonicType::Spherical => cart2sph::compute_spherical_degeneracy(self.get_lqn_a()),
        }
    }
    /// The components of shell b, Cartesian or spherical by its harmonic type.
    pub fn get_degeneracy_b(&self) -> usize {
        match self.get_harmonic_b() {
            shell_types::HarmonicType::Cartesian => cart2sph::compute_cartesian_degeneracy(self.get_lqn_b()),
            shell_types::HarmonicType::Spherical => cart2sph::compute_spherical_degeneracy(self.get_lqn_b()),
        }
    }
}

impl<S: ShellKind> ShellPair<S> {
    /// The pair of the given data; `(x, y, z) = B - A`, `idx_*` the shells' first AOs.
    pub fn new(
        lqn_a: usize,
        lqn_b: usize,
        n_prim_a: usize,
        n_prim_b: usize,
        n_cont_a: usize,
        n_cont_b: usize,
        harmonic_a: shell_types::HarmonicType,
        harmonic_b: shell_types::HarmonicType,
        idx_a: usize,
        idx_b: usize,
        centre_idx_a: usize,
        centre_idx_b: usize,
        x: f64,
        y: f64,
        z: f64,
        single_centre: bool,
        contraction: S::PairData,
        products: Box<[S::Product]>,
    ) -> Self {
        Self {
            lqn_a,
            lqn_b,
            n_prim_a,
            n_prim_b,
            n_cont_a,
            n_cont_b,
            harmonic_a,
            harmonic_b,
            idx_a,
            idx_b,
            centre_idx_a,
            centre_idx_b,
            x,
            y,
            z,
            single_centre,
            contraction,
            products,
        }
    }
}

/// The pairs `(i, j)`, `j >= i`, of `basis_shells`, the shell of lower angular momentum as a; pairs with no product
/// left after screening by `overlap_threshold` are left out. Sorted by class (XX, PP, SP, SS), then by workload,
/// descending. An error if no pair is left.
pub fn derive_shell_pairs<S: ShellKind>(
    basis_shells: &[shell_types::BasisShell<S>],
    overlap_threshold: Option<f64>
) -> Result<Vec<ShellPair<S>>, String> {
    let n_shells = basis_shells.len();
    let n_pairs = n_shells * (n_shells + 1) / 2;
    let mut shell_pairs = Vec::with_capacity(n_pairs);
    for (i, shell) in basis_shells.iter().enumerate() {
        for (j, other_shell) in basis_shells.iter().enumerate().skip(i) {
            let single_centre = i == j;
            let shell_pair = match shell.form_shell_pair(other_shell, single_centre, overlap_threshold) {
                Some(pair) => pair,
                None => continue,
            };

            shell_pairs.push(shell_pair);
        }
    }

    if shell_pairs.is_empty() {
        return Err("No valid shell pairs found -- this suggest error in basis set construction, or too aggressive overlap threshold".to_string());
    }

    shell_pairs.shrink_to_fit();

    // The pairs with more components first: a quartet puts the earlier pair in the bra.
    shell_pairs.sort_by(|a, b| b.components().cmp(&a.components()).then_with(|| b.estimate_workload().cmp(&a.estimate_workload())));

    Ok(shell_pairs)
}
