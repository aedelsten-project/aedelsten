// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Damping of the dipole-dipole interaction at short range.
//
// Thole exponential damping (B. T. Thole, Chem. Phys. 59, 341 (1981); P. Th. van Duijnen and M. Swart, J. Phys.
// Chem. A 102, 2399 (1998); `citations.rs`: `THOLE_1981`, `VAN_DUIJNEN_SWART_1998`): with the reduced distance u = R /
// (alpha_i alpha_j)^(1/6) (alpha the isotropic mean of the tensor) and v = a u, a = 2.1304 by default, the
// dipole-dipole tensor T_ab = (3 R_a R_b / R^2 - delta_ab) / R^3 becomes (3 l5 R_a R_b / R^2 - l3 delta_ab) / R^3 with
//
//   l3 = 1 - (1 + v + v^2/2)       exp(-v)
//   l5 = 1 - (1 + v + v^2/2 + v^3/6) exp(-v)

/// The default Thole parameter.
pub const THOLE_DEFAULT_A: f64 = 2.1304;

/// The damping of the dipole-dipole coupling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Damping {
    /// Undamped.
    None,
    /// Thole exponential damping with the parameter `a`.
    Thole { a: f64 },
}

impl Damping {
    /// Thole damping with the default parameter.
    pub fn thole() -> Self {
        Damping::Thole { a: THOLE_DEFAULT_A }
    }

    /// The damping factors (l3, l5) of a pair at reduced distance `u`.
    pub fn factors(&self, u: f64) -> (f64, f64) {
        match *self {
            Damping::None => (1.0, 1.0),
            Damping::Thole { a } => {
                let v = a * u;
                if v < 2.0 {
                    // Cancellation-free: l_n = exp(-v) Sum_(k>=n) v^k/k!, i.e. 1 - exp(-v) Sum_(k<n) v^k/k!.
                    let exp_minus_v = (-v).exp();
                    (exp_minus_v * exponential_tail(3, v), exp_minus_v * exponential_tail(4, v))
                } else {
                    let exp_minus_v = (-v).exp();
                    let l3 = 1.0 - (1.0 + v + 0.5 * v * v) * exp_minus_v;
                    (l3, l3 - v * v * v / 6.0 * exp_minus_v)
                }
            }
        }
    }
}

// Sum_(k>=n) v^k/k!, by its own series (no subtraction of nearly equal numbers).
fn exponential_tail(n: usize, v: f64) -> f64 {
    let mut term = (1..=n).fold(1.0, |t, k| t * v / k as f64);
    let mut sum = term;
    let mut k = n;
    while term > 1e-17 * sum {
        k += 1;
        term *= v / k as f64;
        sum += term;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_damping_is_the_identity() {
        assert_eq!(Damping::None.factors(0.3), (1.0, 1.0));
    }

    #[test]
    fn thole_factors_match_the_closed_form_on_both_branches_and_are_continuous() {
        let direct = |v: f64| {
            let e = (-v).exp();
            (1.0 - (1.0 + v + 0.5 * v * v) * e, 1.0 - (1.0 + v + 0.5 * v * v + v * v * v / 6.0) * e)
        };
        let damping = Damping::Thole { a: 1.0 };
        for &v in &[0.3, 1.0, 1.999, 2.001, 3.5, 12.0, 60.0] {
            let (l3, l5) = damping.factors(v);
            let (d3, d5) = direct(v);
            assert!((l3 - d3).abs() < 1e-12, "l3({v}): {l3} vs {d3}");
            assert!((l5 - d5).abs() < 1e-12, "l5({v}): {l5} vs {d5}");
        }
    }

    #[test]
    fn thole_factors_vanish_at_short_range_and_tend_to_one_at_long_range() {
        let damping = Damping::thole();
        let u = 1e-4;
        let (l3, l5) = damping.factors(u);
        // l3 ~ v^3/6, l5 ~ v^4/24 for small v (v = a u), up to relative corrections of order v.
        let v: f64 = THOLE_DEFAULT_A * u;
        assert!((l3 - v.powi(3) / 6.0).abs() < 1e-3 * l3);
        assert!((l5 - v.powi(4) / 24.0).abs() < 1e-3 * l5);
        let (l3, l5) = damping.factors(100.0);
        assert!((l3 - 1.0).abs() < 1e-12 && (l5 - 1.0).abs() < 1e-12);
        // Monotonic in distance.
        let mut previous = (0.0, 0.0);
        for k in 1..50 {
            let current = damping.factors(0.1 * k as f64);
            assert!(current.0 >= previous.0 && current.1 >= previous.1);
            previous = current;
        }
    }
}
