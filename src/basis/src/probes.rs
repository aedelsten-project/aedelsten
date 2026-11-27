// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Probe pairs: from the shells alone, a pair of every two shell shapes a basis has, on which the scratch an
//! integral item takes is measured. A shape is what that scratch depends on: angular momentum, harmonic type,
//! primitives and contracted functions; positions enter only through the overlap screening, which removes products,
//! so a probe pair, formed without a threshold, keeps all of them. Each pair of shapes is placed three ways: on two
//! centres, on one centre (two shells of an atom, R = 0) and, for a shape with itself, as one shell's pair with
//! itself. A probe pair another of its kind exceeds in no dimension is left out.

use crate::contraction::ShellKind;
use crate::shell_pairs::ShellPair;
use crate::shell_types::{BasisShell, HarmonicType};

/// B's offset from A on two centres, bohr: about 1, off every axis.
const APART: [f64; 3] = [0.6, 0.5, 0.6];

/// Where a probe pair's two shells are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// On two centres, `APART` from each other.
    Apart,
    /// Two shells on one centre.
    OneCentre,
    /// One shell with itself.
    SameShell,
}

/// The probe pairs of `shells`, placed with A at `origin` on centre `centres[0]` and B on `centres[1]` (on
/// `centres[0]` when on one centre), A's functions first from 0 and B's after them; in the order `derive_shell_pairs`
/// gives, more components first. With their placements.
pub fn probe_pairs<S: ShellKind>(shells: &[BasisShell<S>], centres: [usize; 2], origin: [f64; 3]) -> Vec<(ShellPair<S>, Placement)> {
    let shapes = representatives(shells);
    let mut pairs: Vec<(Dimensions, ShellPair<S>, Placement)> = Vec::new();
    for a in &shapes {
        for b in &shapes {
            let same = std::ptr::eq(a, b);
            for placement in [Placement::Apart, Placement::OneCentre, Placement::SameShell] {
                if placement == Placement::SameShell && !same {
                    continue;
                }
                let dimensions = Dimensions::of(a, b, placement);
                if pairs.iter().any(|(other, _, p)| *p == placement && other.exceeds(&dimensions)) {
                    continue;
                }
                pairs.retain(|(other, _, p)| *p != placement || !dimensions.exceeds(other));
                pairs.push((dimensions, placed_pair(a, b, placement, centres, origin), placement));
            }
        }
    }
    let mut pairs: Vec<(ShellPair<S>, Placement)> = pairs.into_iter().map(|(_, pair, placement)| (pair, placement)).collect();
    pairs.sort_by(|(a, _), (b, _)| b.components().cmp(&a.components()).then_with(|| b.estimate_workload().cmp(&a.estimate_workload())));
    pairs
}

// One shell of every shape, the first met.
fn representatives<S: ShellKind>(shells: &[BasisShell<S>]) -> Vec<BasisShell<S>> {
    let mut kept: Vec<BasisShell<S>> = Vec::new();
    for shell in shells {
        if !kept.iter().any(|k| shape(k) == shape(shell)) {
            kept.push(shell.clone());
        }
    }
    kept
}

fn shape<S: ShellKind>(shell: &BasisShell<S>) -> (usize, HarmonicType, usize, usize) {
    (shell.get_lqn(), shell.get_harmonic_type(), shell.get_n_prim(), shell.get_n_cont())
}

// The shapes of a probe pair, A's and B's; one pair exceeds another of the same angular momenta and harmonic types
// if it has at least as many primitives and contracted functions on each side, and is not the same.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Dimensions {
    a: (usize, HarmonicType, usize, usize),
    b: (usize, HarmonicType, usize, usize),
}

impl Dimensions {
    fn of<S: ShellKind>(a: &BasisShell<S>, b: &BasisShell<S>, placement: Placement) -> Self {
        let b = if placement == Placement::SameShell { a } else { b };
        Dimensions { a: shape(a), b: shape(b) }
    }

    fn exceeds(&self, other: &Self) -> bool {
        let side = |s: (usize, HarmonicType, usize, usize), o: (usize, HarmonicType, usize, usize)| s.0 == o.0 && s.1 == o.1 && s.2 >= o.2 && s.3 >= o.3;
        self != other && side(self.a, other.a) && side(self.b, other.b)
    }
}

fn placed_pair<S: ShellKind>(a: &BasisShell<S>, b: &BasisShell<S>, placement: Placement, centres: [usize; 2], origin: [f64; 3]) -> ShellPair<S> {
    let mut a = a.clone();
    a.set_position(origin);
    a.set_centre_index(centres[0]);
    a.set_basis_index(0);
    if placement == Placement::SameShell {
        return S::form_shell_pair(&a, &a, true, None).expect("a pair without a threshold");
    }
    let mut b = b.clone();
    match placement {
        Placement::Apart => {
            b.set_position(std::array::from_fn(|k| origin[k] + APART[k]));
            b.set_centre_index(centres[1]);
        }
        _ => {
            b.set_position(origin);
            b.set_centre_index(centres[0]);
        }
    }
    b.set_basis_index(a.get_n_cont() * a.get_degeneracy());
    S::form_shell_pair(&a, &b, false, None).expect("a pair without a threshold")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_parser::InputBasisShell;
    use crate::contraction::{Contracted, Segmented, Uncontracted};
    use arrays::matrix::Matrix;

    fn shell(lqn: usize, exponents: &[f64], columns: &[&[f64]], centre: usize) -> BasisShell<Contracted> {
        let data: Vec<f64> = columns.iter().flat_map(|c| c.iter().copied()).collect();
        let coefficients = Matrix::new_from_vec(exponents.len(), columns.len(), data.into_boxed_slice());
        let input = InputBasisShell::new(lqn, exponents.into(), coefficients, None, None);
        BasisShell::<Contracted>::new(&input, HarmonicType::default_for_lqn(lqn), centre, [0.0, 0.0, centre as f64])
    }

    // An O with an s of 5 primitives and 2 functions and a p of 4 and 1, two H with an s of 3 and 1.
    fn water() -> Vec<BasisShell<Contracted>> {
        let o_s: [&[f64]; 2] = [&[0.1, 0.2, 0.3, 0.2, 0.1], &[0.0, 0.1, -0.3, 0.5, 0.4]];
        let h_s: [&[f64]; 1] = [&[0.2, 0.5, 0.4]];
        vec![
            shell(0, &[500.0, 80.0, 20.0, 5.0, 1.0], &o_s, 0),
            shell(1, &[30.0, 7.0, 2.0, 0.5], &[&[0.1, 0.3, 0.5, 0.3]], 0),
            shell(0, &[13.0, 2.0, 0.4], &h_s, 1),
            shell(0, &[13.0, 2.0, 0.4], &h_s, 2),
        ]
    }

    #[test]
    fn every_pair_of_shapes_is_probed_unless_another_exceeds_it() {
        let pairs = probe_pairs(&water(), [3, 4], [1.0, 2.0, 3.0]);
        let count = |placement| pairs.iter().filter(|(_, p)| *p == placement).count();
        // O s with O s exceeds O s with H s, H s with O s and H s with H s; O s with O p exceeds H s with O p, and O p
        // with O s exceeds O p with H s: ss, sp, ps, pp on two centres and on one, the O shells with themselves.
        assert_eq!((count(Placement::Apart), count(Placement::OneCentre), count(Placement::SameShell)), (4, 4, 2));
        for (pair, placement) in &pairs {
            let (n_a, n_b) = (pair.get_n_prim_a(), pair.get_n_prim_b());
            let products = if *placement == Placement::SameShell { n_a * (n_a + 1) / 2 } else { n_a * n_b };
            assert_eq!(pair.get_products().len(), products, "every product kept");
            assert_eq!(pair.get_idx_a(), 0);
        }
        assert!(pairs.windows(2).all(|w| w[0].0.components() >= w[1].0.components()));
    }

    #[test]
    fn a_segmented_basis_probes_its_primitive_blocks_over_the_union_of_its_segments() {
        // An s contraction of three primitives and two single-primitive segments, one of them sharing an exponent.
        let general = [shell(0, &[10.0, 3.0, 1.0, 0.3], &[&[0.2, 0.5, 0.4, 0.0], &[0.0, 0.0, 1.0, 0.0], &[0.0, 0.0, 0.0, 1.0]], 0)];
        let segments = Segmented::reduce(&general);
        assert_eq!(segments.iter().map(|s| s.get_n_prim()).max(), Some(3));
        let blocks = Segmented::primitive_blocks(&segments);
        assert_eq!(blocks.iter().map(|b| b.get_n_prim()).collect::<Vec<_>>(), vec![4]);
        let pairs = probe_pairs::<Uncontracted>(&blocks, [0, 1], [0.0; 3]);
        assert_eq!(pairs.iter().map(|(pair, _)| pair.get_products().len()).max(), Some(16));
    }
}
