// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The classical charge carriers (sites, point charges, finite nuclei, AIMP screens, point multipoles,
//! polarizable sites) and the classical electrostatics between them: interaction energies, their geometric
//! gradients, and electric fields at sites.

mod charges;
mod energy;
mod energy_gradient;
pub mod field;
mod multipole;
mod nuclear;
mod sites;

pub use charges::*;
pub use energy::{
    compute_general_multipole_screened_interaction, compute_screened_coulomb_cross_interaction, compute_screened_coulomb_interaction,
    compute_screened_coulomb_self_interaction,
};
pub use energy_gradient::{
    general_multipole_screened_interaction_gradient, screened_coulomb_cross_interaction_gradient, screened_coulomb_interaction_gradient,
    screened_coulomb_self_interaction_gradient,
};
pub use multipole::*;
pub use nuclear::*;
pub use sites::*;

#[cfg(test)]
mod tests {
    use super::*;

    // Component k of the anisotropic tensor lands on the matrix entry (a, b) whose Cartesian quantum number
    // e_a + e_b is the k-th (lx, ly, lz) of cart2sph::get_cartesian_quantum_numbers(2).
    #[test]
    fn anisotropic_component_order_matches_the_cartesian_enumeration() {
        let components = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = Polarizability::Anisotropic(components).as_matrix();
        let quantum_numbers = cart2sph::get_cartesian_quantum_numbers(2);
        assert_eq!(quantum_numbers.len(), 6);
        for (k, (lx, ly, lz)) in quantum_numbers.into_iter().enumerate() {
            let mut axes = Vec::new();
            for (axis, count) in [lx, ly, lz].into_iter().enumerate() {
                axes.extend(std::iter::repeat(axis).take(count));
            }
            let [a, b] = axes[..] else { panic!("order-2 quantum numbers must sum to 2") };
            assert_eq!(matrix[a][b], components[k], "component {k} ({lx},{ly},{lz})");
            assert_eq!(matrix[b][a], components[k]);
        }
    }

    #[test]
    fn multipole_storage_follows_the_cartesian_enumeration() {
        // Orders are concatenated, each in cart2sph's enumeration.
        let mut expected_index = 0;
        for order in 0..=5 {
            for (lx, ly, lz) in cart2sph::get_cartesian_quantum_numbers(order) {
                assert_eq!(moment_index(lx, ly, lz), expected_index, "({lx},{ly},{lz})");
                expected_index += 1;
            }
            assert_eq!(expected_index, multipole_component_count(order));
        }
    }

    // The canonical Cartesian order: x descending, then y descending within each order.
    #[test]
    fn cartesian_enumeration_is_x_descending_then_y_descending() {
        let names = |order: usize| -> Vec<String> {
            cart2sph::get_cartesian_quantum_numbers(order)
                .into_iter()
                .map(|(lx, ly, lz)| format!("{}{}{}", "x".repeat(lx), "y".repeat(ly), "z".repeat(lz)))
                .collect()
        };
        assert_eq!(names(1), ["x", "y", "z"]);
        assert_eq!(names(2), ["xx", "xy", "xz", "yy", "yz", "zz"]);
        assert_eq!(names(3), ["xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz"]);
    }

    #[test]
    fn file_ordered_moments_land_on_the_right_components() {
        // Value = index in the file's list: order 0 -> 0, order 1 -> 1..3, order 2 -> 4..9, order 3 -> 10..19.
        let file_values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let m = PointMultipole::from_xfield_moments([0.0; 3], 3, &file_values);
        assert_eq!(m.get_max_order(), 3);
        assert_eq!(m.get_charge(), 0.0);
        assert_eq!((m.get_moment(1, 0, 0), m.get_moment(0, 1, 0), m.get_moment(0, 0, 1)), (1.0, 2.0, 3.0));
        // xx xy xz yy yz zz
        assert_eq!(m.get_moment(2, 0, 0), 4.0);
        assert_eq!(m.get_moment(1, 1, 0), 5.0);
        assert_eq!(m.get_moment(1, 0, 1), 6.0);
        assert_eq!(m.get_moment(0, 2, 0), 7.0);
        assert_eq!(m.get_moment(0, 1, 1), 8.0);
        assert_eq!(m.get_moment(0, 0, 2), 9.0);
        // xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
        assert_eq!(m.get_moment(3, 0, 0), 10.0);
        assert_eq!(m.get_moment(2, 1, 0), 11.0);
        assert_eq!(m.get_moment(2, 0, 1), 12.0);
        assert_eq!(m.get_moment(1, 2, 0), 13.0);
        assert_eq!(m.get_moment(1, 1, 1), 14.0);
        assert_eq!(m.get_moment(1, 0, 2), 15.0);
        assert_eq!(m.get_moment(0, 3, 0), 16.0);
        assert_eq!(m.get_moment(0, 2, 1), 17.0);
        assert_eq!(m.get_moment(0, 1, 2), 18.0);
        assert_eq!(m.get_moment(0, 0, 3), 19.0);
    }

    #[test]
    fn dipole_and_quadrupole_constructors_use_the_same_layout() {
        let q = PointMultipole::quadrupole([0.0; 3], 1.0, [2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(q.get_max_order(), 2);
        assert_eq!(q.get_moment(2, 0, 0), 5.0); // xx
        assert_eq!(q.get_moment(1, 0, 1), 7.0); // xz: third value of the file's xx xy xz yy yz zz
        assert_eq!(q.get_moment(0, 2, 0), 8.0); // yy
        let d = PointMultipole::dipole([0.0; 3], 1.0, [2.0, 3.0, 4.0]);
        assert_eq!(d.get_max_order(), 1);
    }

    #[test]
    fn fragments_are_stored_inline_and_bounded_to_two() {
        assert_eq!(Fragments::NONE.as_slice(), &[] as &[i32]);
        assert_eq!(Fragments::new(&[7]).as_slice(), &[7]);
        assert_eq!(Fragments::new(&[3, -1]).as_slice(), &[3, -1]);
        assert!(std::panic::catch_unwind(|| Fragments::new(&[1, 2, 3])).is_err());
        // Bare sites, nuclei-like types and AIMPs report no fragments.
        assert!(Site::new([0.0; 3]).get_fragments().is_empty());
        assert!(IndexedPointCharge::new(0, [0.0; 3], 1.0).get_fragments().is_empty());
    }

    #[test]
    fn fragment_exclusion_follows_the_molcas_rules() {
        let site = |fragments: &[i32]| PolarizableSite::new([0.0; 3], Polarizability::Isotropic(1.0), Fragments::new(fragments));
        let source = |fragments: &[i32]| PointCharge::new([1.0; 3], 1.0).with_fragments(Fragments::new(fragments));

        // One fragment number: excluded within the same fragment only.
        assert!(fragment_excludes(&site(&[4]), &source(&[4])));
        assert!(!fragment_excludes(&site(&[4]), &source(&[5])));
        // Two: a junction source [4, 6] is excluded from the sites of either fragment.
        assert!(fragment_excludes(&site(&[4]), &source(&[4, 6])));
        assert!(fragment_excludes(&site(&[6]), &source(&[4, 6])));
        assert!(!fragment_excludes(&site(&[5]), &source(&[4, 6])));
        // Only the site's OWN (first) fragment counts, not a second one it may also list.
        assert!(!fragment_excludes(&site(&[5, 4]), &source(&[4])));
        // Without fragments on either side nothing is excluded by this rule.
        assert!(!fragment_excludes(&site(&[]), &source(&[4])));
        assert!(!fragment_excludes(&site(&[4]), &source(&[])));
    }
}
