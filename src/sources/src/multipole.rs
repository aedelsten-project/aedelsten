// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use crate::charges::{MultipoleSourceTrait, PointCharge, PointChargeTrait};
use crate::sites::{Fragments, NO_INPUT_INDEX, SiteTrait, input_index_from_storage};

pub use math_utilities::cartesian::{moment_index, multipole_component_count};

// The contact-free (harmonic) part of the moments. The harmonic part of the order-n polynomial
// P(x) = Sum_k M_k x^k / k! is
//   H = Sum_j a_j r^(2j) Laplacian^j P,   a_j = (-1)^j (2n-2j-1)!! / ((2n-1)!! 2^j j!).

type Polynomial = std::collections::BTreeMap<(usize, usize, usize), f64>;

fn laplacian(p: &Polynomial) -> Polynomial {
    let mut result = Polynomial::new();
    for (&(a, b, c), &value) in p {
        if a >= 2 {
            *result.entry((a - 2, b, c)).or_insert(0.0) += value * (a * (a - 1)) as f64;
        }
        if b >= 2 {
            *result.entry((a, b - 2, c)).or_insert(0.0) += value * (b * (b - 1)) as f64;
        }
        if c >= 2 {
            *result.entry((a, b, c - 2)).or_insert(0.0) += value * (c * (c - 1)) as f64;
        }
    }
    result
}

fn times_r_squared(p: &Polynomial) -> Polynomial {
    let mut result = Polynomial::new();
    for (&(a, b, c), &value) in p {
        *result.entry((a + 2, b, c)).or_insert(0.0) += value;
        *result.entry((a, b + 2, c)).or_insert(0.0) += value;
        *result.entry((a, b, c + 2)).or_insert(0.0) += value;
    }
    result
}

// n!! for odd n >= -1 ((-1)!! = 1).
fn double_factorial(n: i64) -> f64 {
    (1..=n).rev().step_by(2).map(|k| k as f64).product()
}

fn factorial_f64(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// The harmonic part of the order-`order` moments `raw` (the components of that order alone, in the canonical
/// order), in the same layout. Orders 0 and 1 are their own harmonic part.
pub(crate) fn harmonic_part(order: usize, raw: &[f64]) -> Vec<f64> {
    if order < 2 {
        return raw.to_vec();
    }
    // P = Sum_k M_k x^k / k!
    let mut polynomial = Polynomial::new();
    let mut components = Vec::with_capacity(raw.len());
    math_utilities::cartesian::for_each_cartesian_component(order, |lx, ly, lz| components.push((lx, ly, lz)));
    for (&(lx, ly, lz), &moment) in components.iter().zip(raw) {
        polynomial.insert((lx, ly, lz), moment / (factorial_f64(lx) * factorial_f64(ly) * factorial_f64(lz)));
    }

    let mut harmonic = Polynomial::new();
    let mut laplacian_power = polynomial.clone();
    for j in 0..=order / 2 {
        let coefficient = (if j % 2 == 0 { 1.0 } else { -1.0 }) * double_factorial(2 * order as i64 - 2 * j as i64 - 1)
            / (double_factorial(2 * order as i64 - 1) * 2f64.powi(j as i32) * factorial_f64(j));
        let mut term = laplacian_power.clone();
        for _ in 0..j {
            term = times_r_squared(&term);
        }
        for (key, value) in term {
            *harmonic.entry(key).or_insert(0.0) += coefficient * value;
        }
        laplacian_power = laplacian(&laplacian_power);
    }
    components
        .iter()
        .map(|&(lx, ly, lz)| harmonic.get(&(lx, ly, lz)).copied().unwrap_or(0.0) * factorial_f64(lx) * factorial_f64(ly) * factorial_f64(lz))
        .collect()
}

// The whole storage layout (orders 0..=max_order concatenated) with every order replaced by its harmonic part.
fn contact_free_moments(max_order: usize, moments: &[f64]) -> Box<[f64]> {
    let mut result = Vec::with_capacity(moments.len());
    for order in 0..=max_order {
        let start = multipole_component_count(order) - (order + 1) * (order + 2) / 2;
        let end = multipole_component_count(order);
        result.extend(harmonic_part(order, &moments[start..end]));
    }
    result.into_boxed_slice()
}

/// Which interaction of an environment multipole with the electrons the operator represents. The classical
/// interaction with point charges away from the site is the same in both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MultipoleConvention {
    /// Without the contact terms: only the harmonic (traceless) part of each order's moments interacts with the
    /// electron density. The default.
    #[default]
    ContactFree,
    /// The exact interaction of the moments as a charge distribution, with true derivatives of 1/r: the trace
    /// parts couple to the electron density at the site (delta-function contact terms).
    Exact,
}

/// A point source with a charge and any number of higher Cartesian moments (dipole, quadrupole,
/// octupole, ...), up to a maximum order chosen per source.
///
/// The moments are bare Cartesian moment tensor components (see `MultipoleSourceTrait`), stored
/// concatenated over the orders 0..=max_order, each order in the canonical Cartesian order. Their
/// harmonic parts are stored alongside.
#[derive(Clone, Debug)]
pub struct PointMultipole {
    position: [f64; 3],
    max_order: usize,
    moments: Box<[f64]>,
    contact_free: Box<[f64]>,
    fragments: Fragments,
    input_index: usize,
}

impl PointMultipole {
    /// `moments` in the canonical storage order described above; its length must be
    /// `multipole_component_count(max_order)`.
    pub fn new(position: [f64; 3], max_order: usize, moments: impl Into<Box<[f64]>>) -> Self {
        let moments = moments.into();
        assert_eq!(
            moments.len(),
            multipole_component_count(max_order),
            "a point multipole of order {max_order} has {} moment components",
            multipole_component_count(max_order)
        );
        let contact_free = contact_free_moments(max_order, &moments);
        Self { position, max_order, moments, contact_free, fragments: Fragments::NONE, input_index: NO_INPUT_INDEX }
    }

    /// From an xfield record's moments of orders 0..=max_order, which are in the canonical order.
    /// Panics unless there are `multipole_component_count(max_order)` of them.
    pub fn from_xfield_moments(position: [f64; 3], max_order: usize, file_values: &[f64]) -> Self {
        assert_eq!(file_values.len(), multipole_component_count(max_order));
        let contact_free = contact_free_moments(max_order, file_values);
        Self { position, max_order, moments: file_values.into(), contact_free, fragments: Fragments::NONE, input_index: NO_INPUT_INDEX }
    }

    /// The fragments whose polarizing field this multipole must not contribute to.
    pub fn with_fragments(mut self, fragments: Fragments) -> Self {
        self.fragments = fragments;
        self
    }

    /// The record of the input file this multipole was read from (see `SiteTrait::get_input_index`).
    pub fn with_input_index(mut self, index: usize) -> Self {
        self.input_index = index;
        self
    }

    /// Every moment is exactly zero: such a source contributes nothing.
    pub fn is_zero(&self) -> bool {
        self.moments.iter().all(|&m| m == 0.0)
    }

    /// Charge and dipole.
    pub fn dipole(position: [f64; 3], charge: f64, dipole: [f64; 3]) -> Self {
        Self::from_xfield_moments(position, 1, &[charge, dipole[0], dipole[1], dipole[2]])
    }

    /// Charge, dipole and quadrupole, the latter as the xfield file's raw xx xy xz yy yz zz sextet.
    pub fn quadrupole(position: [f64; 3], charge: f64, dipole: [f64; 3], quadrupole: [f64; 6]) -> Self {
        let [xx, xy, xz, yy, yz, zz] = quadrupole;
        Self::from_xfield_moments(position, 2, &[charge, dipole[0], dipole[1], dipole[2], xx, xy, xz, yy, yz, zz])
    }

}

impl SiteTrait for PointMultipole {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }

    fn get_fragments(&self) -> &[i32] {
        self.fragments.as_slice()
    }

    fn get_input_index(&self) -> Option<usize> {
        input_index_from_storage(self.input_index)
    }
}

impl PointChargeTrait for PointMultipole {
    fn get_charge(&self) -> f64 {
        self.moments[0]
    }
}

impl MultipoleSourceTrait for PointMultipole {
    fn get_max_order(&self) -> usize {
        self.max_order
    }

    fn get_contact_free_moment(&self, lx: usize, ly: usize, lz: usize) -> f64 {
        self.contact_free[moment_index(lx, ly, lz)]
    }

    fn get_moment(&self, lx: usize, ly: usize, lz: usize) -> f64 {
        debug_assert!(lx + ly + lz <= self.max_order, "moment of order {} requested from a multipole of order {}", lx + ly + lz, self.max_order);
        self.moments[moment_index(lx, ly, lz)]
    }
}

/// A point dipole with no charge, held inline: a source of order 1 whose dipole can be changed in place.
#[derive(Clone, Copy, Debug, Default)]
pub struct PointDipole {
    position: [f64; 3],
    dipole: [f64; 3],
}

impl PointDipole {
    /// The dipole `dipole` at `position`.
    pub fn new(position: [f64; 3], dipole: [f64; 3]) -> Self {
        Self { position, dipole }
    }

    /// Replaces the dipole.
    pub fn set_dipole(&mut self, dipole: [f64; 3]) {
        self.dipole = dipole;
    }
}

impl SiteTrait for PointDipole {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }
}

impl PointChargeTrait for PointDipole {
    fn get_charge(&self) -> f64 {
        0.0
    }
}

impl MultipoleSourceTrait for PointDipole {
    fn get_max_order(&self) -> usize {
        1
    }

    fn get_moment(&self, lx: usize, ly: usize, lz: usize) -> f64 {
        match (lx, ly, lz) {
            (0, 0, 0) => 0.0,
            (1, 0, 0) => self.dipole[0],
            (0, 1, 0) => self.dipole[1],
            (0, 0, 1) => self.dipole[2],
            _ => panic!("moment ({lx}, {ly}, {lz}) requested from a dipole"),
        }
    }
}

/// The external multipoles of an xfield file, all of the file's order: point charges for order 0, point
/// multipoles otherwise.
#[derive(Clone, Debug)]
pub enum ExternalMultipoles {
    Charges(Box<[PointCharge]>),
    Multipoles(Box<[PointMultipole]>),
}

impl ExternalMultipoles {
    /// The memory held, in bytes, the moments' heap storage included.
    pub fn approx_memory_bytes(&self) -> usize {
        match self {
            ExternalMultipoles::Charges(charges) => std::mem::size_of_val(charges.as_ref()),
            ExternalMultipoles::Multipoles(multipoles) => multipoles
                .iter()
                .map(|m| std::mem::size_of::<PointMultipole>() + (m.moments.len() + m.contact_free.len()) * std::mem::size_of::<f64>())
                .sum(),
        }
    }
}

#[cfg(test)]
mod harmonic_tests {
    use super::*;
    use math_utilities::radial::{Coulomb, Derivatives};

    // A fixed, non-symmetric-looking set of raw moments for one order.
    fn raw_moments(order: usize) -> Vec<f64> {
        (0..(order + 1) * (order + 2) / 2).map(|i| 0.7 + 0.31 * ((i * 5 + order) % 7) as f64 - 0.9 * (i % 2) as f64).collect()
    }

    fn component(order: usize, lx: usize, ly: usize, lz: usize) -> usize {
        moment_index(lx, ly, lz) - (multipole_component_count(order) - (order + 1) * (order + 2) / 2)
    }

    #[test]
    fn quadrupole_harmonic_part_is_the_traceless_tensor() {
        let raw = raw_moments(2);
        let h = harmonic_part(2, &raw);
        let (xx, yy, zz) = (component(2, 2, 0, 0), component(2, 0, 2, 0), component(2, 0, 0, 2));
        let trace = raw[xx] + raw[yy] + raw[zz];
        for (index, (&r, &v)) in raw.iter().zip(&h).enumerate() {
            let expected = if [xx, yy, zz].contains(&index) { r - trace / 3.0 } else { r };
            assert!((v - expected).abs() < 1e-13, "component {index}: {v} vs {expected}");
        }
        assert!((h[xx] + h[yy] + h[zz]).abs() < 1e-13);
    }

    #[test]
    fn octupole_harmonic_part_removes_the_trace_vector() {
        let raw = raw_moments(3);
        let h = harmonic_part(3, &raw);
        let m = |lx, ly, lz| raw[component(3, lx, ly, lz)];
        let (vx, vy, vz) = (m(3, 0, 0) + m(1, 2, 0) + m(1, 0, 2), m(2, 1, 0) + m(0, 3, 0) + m(0, 1, 2), m(2, 0, 1) + m(0, 2, 1) + m(0, 0, 3));
        let expected = |lx, ly, lz| -> f64 {
            let removed = match (lx, ly, lz) {
                (3, 0, 0) => 3.0 * vx / 5.0,
                (2, 1, 0) => vy / 5.0,
                (2, 0, 1) => vz / 5.0,
                (1, 2, 0) => vx / 5.0,
                (1, 0, 2) => vx / 5.0,
                (0, 3, 0) => 3.0 * vy / 5.0,
                (0, 2, 1) => vz / 5.0,
                (0, 1, 2) => vy / 5.0,
                (0, 0, 3) => 3.0 * vz / 5.0,
                _ => 0.0, // xyz is trace-free
            };
            m(lx, ly, lz) - removed
        };
        math_utilities::cartesian::for_each_cartesian_component(3, |lx, ly, lz| {
            assert!((h[component(3, lx, ly, lz)] - expected(lx, ly, lz)).abs() < 1e-13, "({lx},{ly},{lz})");
        });
    }

    #[test]
    fn the_projection_is_idempotent_and_harmonic_at_every_order() {
        for order in 0..=6 {
            let raw = raw_moments(order);
            let h = harmonic_part(order, &raw);
            let again = harmonic_part(order, &h);
            for (a, b) in h.iter().zip(&again) {
                assert!((a - b).abs() < 1e-10 * (1.0 + a.abs()), "order {order}: not idempotent");
            }
            // Harmonic: the Laplacian of P(x) = Sum M_k x^k / k! vanishes, i.e. for every (a,b,c) of degree n-2,
            // M_(a+2,b,c) + M_(a,b+2,c) + M_(a,b,c+2) = 0 (the moments carry the k! already).
            if order >= 2 {
                math_utilities::cartesian::for_each_cartesian_component(order - 2, |lx, ly, lz| {
                    let trace = h[component(order, lx + 2, ly, lz)] + h[component(order, lx, ly + 2, lz)] + h[component(order, lx, ly, lz + 2)];
                    assert!(trace.abs() < 1e-10 * (1.0 + h.iter().fold(0.0_f64, |w, x| w.max(x.abs()))), "order {order} ({lx},{ly},{lz}): trace {trace}");
                });
            }
        }
    }

    // Away from the source the kernel is harmonic, so the raw and the harmonic moments have the same
    // interaction, at every order up to 5.
    #[test]
    fn raw_and_harmonic_moments_have_the_same_classical_interaction() {
        let d = [1.3, -0.8, 2.1];
        let mut derivatives = Derivatives::new(5);
        derivatives.compute(&Coulomb, d);
        for order in 2..=5 {
            let raw = raw_moments(order);
            let h = harmonic_part(order, &raw);
            let interaction = |moments: &[f64]| -> f64 {
                let mut sum = 0.0;
                let mut index = 0;
                math_utilities::cartesian::for_each_cartesian_component(order, |lx, ly, lz| {
                    let k_factorial = factorial_f64(lx) * factorial_f64(ly) * factorial_f64(lz);
                    sum += moments[index] / k_factorial * derivatives.get(lx, ly, lz);
                    index += 1;
                });
                sum
            };
            let (a, b) = (interaction(&raw), interaction(&h));
            assert!((a - b).abs() < 1e-11 * a.abs().max(1e-6), "order {order}: raw {a} vs harmonic {b}");
        }
    }

    #[test]
    fn point_multipoles_expose_both_conventions() {
        let source = PointMultipole::quadrupole([0.0; 3], 1.0, [0.1, 0.2, 0.3], [3.0, 0.5, 0.25, 6.0, 0.125, 9.0]);
        // Up to the dipole they coincide; the quadrupole loses its trace (18 / 3 = 6 per diagonal entry).
        assert_eq!(source.get_contact_free_moment(0, 0, 0), 1.0);
        assert_eq!(source.get_contact_free_moment(0, 1, 0), 0.2);
        assert!((source.get_contact_free_moment(2, 0, 0) - (3.0 - 6.0)).abs() < 1e-14);
        assert!((source.get_contact_free_moment(0, 2, 0) - 0.0).abs() < 1e-14);
        assert!((source.get_contact_free_moment(0, 0, 2) - 3.0).abs() < 1e-14);
        assert_eq!(source.get_contact_free_moment(1, 1, 0), source.get_moment(1, 1, 0));
        // The raw moments are untouched.
        assert_eq!(source.get_moment(2, 0, 0), 3.0);
    }
}
