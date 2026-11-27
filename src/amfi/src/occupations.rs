// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The ground-state configurations of the neutral atoms that define the atomic mean field: per angular momentum, the
//! number of closed shells and the electrons in the open shell.

use crate::AmfiError;

/// The highest angular momentum an occupied atomic shell has.
pub const MAX_OCCUPIED_LQN: usize = 3;

/// The heaviest element with a configuration.
pub const MAX_ATOMIC_NUMBER: u8 = 118;

// The subshells (n, l) in filling order: by n + l, then by n.
const FILLING_ORDER: [(usize, usize); 19] = [
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2), (5, 1), (6, 0), (4, 3), (5, 2),
    (6, 1), (7, 0), (5, 3), (6, 2), (7, 1),
];

// The configurations that differ from the filling order: (Z, electrons moved, from (n, l), to (n, l)).
const EXCEPTIONS: [(u8, usize, (usize, usize), (usize, usize)); 10] = [
    (29, 1, (4, 0), (3, 2)), // Cu
    (47, 1, (5, 0), (4, 2)), // Ag
    (57, 1, (4, 3), (5, 2)), // La
    (78, 1, (6, 0), (5, 2)), // Pt
    (79, 1, (6, 0), (5, 2)), // Au
    (89, 1, (5, 3), (6, 2)), // Ac
    (90, 2, (5, 3), (6, 2)), // Th
    (91, 1, (5, 3), (6, 2)), // Pa
    (92, 1, (5, 3), (6, 2)), // U
    (93, 1, (5, 3), (6, 2)), // Np
];

/// One atom's mean-field configuration: per angular momentum `l <= MAX_OCCUPIED_LQN`, the closed shells (the first
/// `closed[l]` orbitals at `l`, two electrons in each spatial orbital) and the electrons of the open shell (the next
/// orbital at `l`, spread evenly over its `2l + 1` components).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicOccupations {
    closed: [usize; MAX_OCCUPIED_LQN + 1],
    open: [usize; MAX_OCCUPIED_LQN + 1],
}

impl AtomicOccupations {
    /// The ground-state configuration of the neutral atom `atomic_number`: the subshells filled in order of `n + l`,
    /// then `n`, with the known exceptions (Cu, Ag, La, Pt, Au, Ac, Th, Pa, U, Np). Hydrogen and a ghost (`0`) have
    /// no occupied orbitals: a one-electron atom has no two-electron mean field. An error above
    /// `MAX_ATOMIC_NUMBER`.
    pub fn for_element(atomic_number: u8) -> Result<Self, AmfiError> {
        if atomic_number > MAX_ATOMIC_NUMBER {
            return Err(AmfiError::UnsupportedElement { atomic_number });
        }
        let mut electrons = [0usize; FILLING_ORDER.len()];
        if atomic_number >= 2 {
            let mut left = atomic_number as usize;
            for (count, &(_, l)) in electrons.iter_mut().zip(&FILLING_ORDER) {
                *count = left.min(2 * (2 * l + 1));
                left -= *count;
            }
            if let Some(&(_, moved, from, to)) = EXCEPTIONS.iter().find(|e| e.0 == atomic_number) {
                let position = |subshell| FILLING_ORDER.iter().position(|&s| s == subshell).expect("a listed subshell");
                electrons[position(from)] -= moved;
                electrons[position(to)] += moved;
            }
        }
        let mut closed = [0; MAX_OCCUPIED_LQN + 1];
        let mut open = [0; MAX_OCCUPIED_LQN + 1];
        for (&count, &(_, l)) in electrons.iter().zip(&FILLING_ORDER) {
            if count == 2 * (2 * l + 1) {
                closed[l] += 1;
            } else if count > 0 {
                open[l] += count;
            }
        }
        Ok(Self { closed, open })
    }

    /// The closed shells at `l` (0 above `MAX_OCCUPIED_LQN`).
    pub fn get_closed(&self, l: usize) -> usize {
        self.closed.get(l).copied().unwrap_or(0)
    }

    /// The electrons of the open shell at `l` (0 above `MAX_OCCUPIED_LQN`).
    pub fn get_open(&self, l: usize) -> usize {
        self.open.get(l).copied().unwrap_or(0)
    }

    /// The occupied orbitals at `l`: the closed shells and, if there is one, the open shell.
    pub fn get_n_occupied(&self, l: usize) -> usize {
        self.get_closed(l) + (self.get_open(l) > 0) as usize
    }

    /// Per `l = 0..=MAX_OCCUPIED_LQN`, `get_n_occupied(l)`.
    pub fn n_occupied_per_lqn(&self) -> [usize; MAX_OCCUPIED_LQN + 1] {
        std::array::from_fn(|l| self.get_n_occupied(l))
    }

    /// The electrons in each spatial component of occupied orbital `ordinal` (0-based) at `l`: 2 for a closed shell,
    /// `open / (2l + 1)` for the open one, 0 past them.
    pub fn get_occupation(&self, l: usize, ordinal: usize) -> f64 {
        if ordinal < self.get_closed(l) {
            2.0
        } else if ordinal == self.get_closed(l) && self.get_open(l) > 0 {
            self.get_open(l) as f64 / (2 * l + 1) as f64
        } else {
            0.0
        }
    }

    /// Whether no orbital is occupied.
    pub fn is_empty(&self) -> bool {
        self.closed.iter().chain(&self.open).all(|&n| n == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configuration(z: u8) -> ([usize; 4], [usize; 4]) {
        let occupations = AtomicOccupations::for_element(z).unwrap();
        (occupations.closed, occupations.open)
    }

    #[test]
    fn hydrogen_and_ghosts_have_no_mean_field() {
        assert!(AtomicOccupations::for_element(0).unwrap().is_empty());
        assert!(AtomicOccupations::for_element(1).unwrap().is_empty());
    }

    #[test]
    fn filling_order_configurations() {
        assert_eq!(configuration(2), ([1, 0, 0, 0], [0, 0, 0, 0])); // 1s2
        assert_eq!(configuration(6), ([2, 0, 0, 0], [0, 2, 0, 0])); // [He]2s2 2p2
        assert_eq!(configuration(10), ([2, 1, 0, 0], [0, 0, 0, 0])); // [He]2s2 2p6
        assert_eq!(configuration(24), ([4, 2, 0, 0], [0, 0, 4, 0])); // [Ar]4s2 3d4
        assert_eq!(configuration(28), ([4, 2, 0, 0], [0, 0, 8, 0])); // [Ar]4s2 3d8
        assert_eq!(configuration(58), ([6, 4, 2, 0], [0, 0, 0, 2])); // [Xe]6s2 4f2
        assert_eq!(configuration(64), ([6, 4, 2, 0], [0, 0, 0, 8])); // [Xe]6s2 4f8
        assert_eq!(configuration(86), ([6, 5, 3, 1], [0, 0, 0, 0])); // Rn
    }

    #[test]
    fn exceptions_move_electrons_between_subshells() {
        assert_eq!(configuration(29), ([3, 2, 1, 0], [1, 0, 0, 0])); // [Ar]4s1 3d10
        assert_eq!(configuration(57), ([6, 4, 2, 0], [0, 0, 1, 0])); // [Xe]6s2 5d1
        assert_eq!(configuration(78), ([5, 4, 2, 1], [1, 0, 9, 0])); // [Xe]6s1 4f14 5d9
        assert_eq!(configuration(79), ([5, 4, 3, 1], [1, 0, 0, 0])); // [Xe]6s1 4f14 5d10
        assert_eq!(configuration(90), ([7, 5, 3, 1], [0, 0, 2, 0])); // [Rn]7s2 6d2
        assert_eq!(configuration(92), ([7, 5, 3, 1], [0, 0, 1, 3])); // [Rn]7s2 5f3 6d1
    }

    #[test]
    fn occupations_per_orbital() {
        let carbon = AtomicOccupations::for_element(6).unwrap();
        assert_eq!(carbon.n_occupied_per_lqn(), [2, 1, 0, 0]);
        assert_eq!(carbon.get_occupation(0, 1), 2.0);
        assert_eq!(carbon.get_occupation(1, 0), 2.0 / 3.0);
        assert_eq!(carbon.get_occupation(1, 1), 0.0);
        assert_eq!(carbon.get_occupation(4, 0), 0.0);
    }

    #[test]
    fn electron_count_is_the_atomic_number() {
        for z in 2..=MAX_ATOMIC_NUMBER {
            let occupations = AtomicOccupations::for_element(z).unwrap();
            let electrons: usize = (0..=MAX_OCCUPIED_LQN)
                .map(|l| 2 * (2 * l + 1) * occupations.get_closed(l) + occupations.get_open(l))
                .sum();
            assert_eq!(electrons, z as usize, "Z = {z}");
        }
        assert!(AtomicOccupations::for_element(MAX_ATOMIC_NUMBER + 1).is_err());
    }
}
