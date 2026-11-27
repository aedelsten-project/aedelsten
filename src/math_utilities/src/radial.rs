// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Cartesian derivatives of radial kernels g(|d|), to any order.
//!
//! A radial kernel is fully described by B_m(r) = ((1/r) d/dr)^m g(r), because d/dx B_m = x B_{m+1}.
//! Differentiating a product then gives, for the derivative G(a,b,c; m) = D^(a,b,c) B_m at d = (x,y,z),
//!
//!   G(a+1,b,c; m) = x G(a,b,c; m+1) + a G(a-1,b,c; m+1)
//!
//! (and likewise in y and z), seeded by G(0,0,0; m) = B_m. Every derivative up to order N therefore
//! needs only B_0..B_N, and each kernel supplies its B_m.

use crate::cartesian::{for_each_cartesian_component, moment_index, multipole_component_count};

/// A radial kernel g(r), through `B_m(r) = ((1/r) d/dr)^m g(r)`.
pub trait RadialKernel {
    /// B_m(r).
    fn b(&self, m: usize, r: f64) -> f64;
}

/// g(r) = 1/r.
pub struct Coulomb;

impl RadialKernel for Coulomb {
    fn b(&self, m: usize, r: f64) -> f64 {
        // (-1)^m (2m-1)!! / r^(2m+1)
        let double_factorial: f64 = (1..=m).map(|k| (2 * k - 1) as f64).product();
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        sign * double_factorial / r.powi(2 * m as i32 + 1)
    }
}

/// g(r) = erf(kappa r)/r, the potential of a spherical Gaussian charge (the Visscher-Dyall
/// finite nucleus, kappa = sqrt(exponent)) felt by a point charge. Regular at r = 0.
pub struct ErfCoulomb {
    /// The square root of the Gaussian's exponent.
    pub kappa: f64,
}

impl RadialKernel for ErfCoulomb {
    fn b(&self, m: usize, r: f64) -> f64 {
        // erf(kappa r)/r = 2 kappa/sqrt(pi) F_0(kappa^2 r^2) and d/dt F_m = -F_(m+1), with
        // (1/r) d/dr = 2 kappa^2 d/dt, so B_m = 2 kappa/sqrt(pi) (-2 kappa^2)^m F_m(kappa^2 r^2).
        let kappa_squared = self.kappa * self.kappa;
        2.0 * self.kappa / std::f64::consts::PI.sqrt() * (-2.0 * kappa_squared).powi(m as i32) * crate::boys::compute_boys_function(m, kappa_squared * r * r)
    }
}

/// g(r) = exp(-c r^2)/r, one term of an AIMP's screening correction.
pub struct GaussianScreenedCoulomb {
    /// The Gaussian's exponent.
    pub c: f64,
}

impl RadialKernel for GaussianScreenedCoulomb {
    fn b(&self, m: usize, r: f64) -> f64 {
        // B_m = exp(-c r^2) Sum_j a_(m,j) r^-(2j+1). Applying (1/r) d/dr to a term a r^-p gives
        // -2c a r^-p - p a r^-(p+2), i.e. a_(m+1,j) = -2c a_(m,j) - (2j-1) a_(m,j-1).
        const MAX_M: usize = 16;
        assert!(m <= MAX_M, "radial kernel derivatives are only implemented up to B_{MAX_M}");
        let mut coefficients = [0.0; MAX_M + 1];
        coefficients[0] = 1.0;
        for step in 0..m {
            for j in (0..=step + 1).rev() {
                let lower = if j > 0 { (2 * j as i64 - 1) as f64 * coefficients[j - 1] } else { 0.0 };
                coefficients[j] = -2.0 * self.c * coefficients[j] - lower;
            }
        }
        let inverse_r_squared = 1.0 / (r * r);
        // Horner in 1/r^2, then the leading 1/r.
        let polynomial = coefficients[..=m].iter().rev().fold(0.0, |accumulator, &a| accumulator * inverse_r_squared + a);
        (-self.c * r * r).exp() / r * polynomial
    }
}

/// g(r) = (1 + Sum_i A_i exp(-c_i r^2))/r, the full potential of an AIMP per unit charge: the point
/// charge plus its screening correction.
pub struct ScreenedCoulomb<'a> {
    /// The exponents c_i.
    pub exponents: &'a [f64],
    /// The coefficients A_i.
    pub coefficients: &'a [f64],
}

impl RadialKernel for ScreenedCoulomb<'_> {
    fn b(&self, m: usize, r: f64) -> f64 {
        let mut b = Coulomb.b(m, r);
        for (&c, &coefficient) in self.exponents.iter().zip(self.coefficients) {
            b += coefficient * GaussianScreenedCoulomb { c }.b(m, r);
        }
        b
    }
}

/// Reusable storage for the Cartesian derivatives D^(a,b,c) g(|d|) with a+b+c <= max_order, so
/// that evaluating many (site, source) pairs allocates once.
pub struct Derivatives {
    max_order: usize,
    // tables[m][at(a,b,c)] = D^(a,b,c) B_m, flattened; filled for a+b+c <= max_order - m.
    tables: Vec<f64>,
    values: Vec<f64>,
}

impl Derivatives {
    /// Storage for the derivatives up to order `max_order`.
    pub fn new(max_order: usize) -> Self {
        let width = max_order + 1;
        Self { max_order, tables: vec![0.0; (max_order + 1) * width * width * width], values: vec![0.0; multipole_component_count(max_order)] }
    }

    /// The highest derivative order.
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// Evaluates every derivative up to `max_order` at `d`. `d` must not be the origin (the
    /// kernels are singular there, except erf).
    pub fn compute(&mut self, kernel: &impl RadialKernel, d: [f64; 3]) {
        let max_order = self.max_order;
        let width = max_order + 1;
        let cube = width * width * width;
        let at = |m: usize, a: usize, b: usize, c: usize| m * cube + (a * width + b) * width + c;
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();

        for m in 0..=max_order {
            self.tables[at(m, 0, 0, 0)] = kernel.b(m, r);
        }
        for order in 1..=max_order {
            for m in 0..=max_order - order {
                for a in 0..=order {
                    for b in 0..=order - a {
                        let c = order - a - b;
                        // Differentiate along the first nonzero direction of (a, b, c).
                        let (axis, position, reduced) = if a > 0 {
                            (0, d[0], [a - 1, b, c])
                        } else if b > 0 {
                            (1, d[1], [a, b - 1, c])
                        } else {
                            (2, d[2], [a, b, c - 1])
                        };
                        let previous = reduced[axis]; // the exponent along `axis` before this step
                        let mut value = position * self.tables[at(m + 1, reduced[0], reduced[1], reduced[2])];
                        if previous > 0 {
                            let mut twice_reduced = reduced;
                            twice_reduced[axis] -= 1;
                            value += previous as f64 * self.tables[at(m + 1, twice_reduced[0], twice_reduced[1], twice_reduced[2])];
                        }
                        self.tables[at(m, a, b, c)] = value;
                    }
                }
            }
        }

        for order in 0..=max_order {
            for_each_cartesian_component(order, |lx, ly, lz| {
                self.values[moment_index(lx, ly, lz)] = self.tables[at(0, lx, ly, lz)];
            });
        }
    }

    /// D^(lx,ly,lz) g at the last `compute`d point.
    pub fn get(&self, lx: usize, ly: usize, lz: usize) -> f64 {
        self.values[moment_index(lx, ly, lz)]
    }

    /// All values, indexed by `cartesian::moment_index`.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// One-shot convenience over `Derivatives`, into `out` (see `Derivatives::values` for the layout).
pub fn kernel_derivatives(kernel: &impl RadialKernel, d: [f64; 3], max_order: usize, out: &mut Vec<f64>) {
    let mut derivatives = Derivatives::new(max_order);
    derivatives.compute(kernel, d);
    out.clear();
    out.extend_from_slice(derivatives.values());
}

#[cfg(test)]
mod tests {
    use super::*;

    // Composite Simpson for F_m(t) = int_0^1 u^(2m) exp(-t u^2) du, as an independent reference.
    fn boys_by_quadrature(m: usize, t: f64) -> f64 {
        // Simpson's rule; u^(2m) peaks sharply at u = 1 for high m, so more points there.
        let n = if m > 10 { 200000 } else { 20000 };
        let h = 1.0 / n as f64;
        let f = |u: f64| u.powi(2 * m as i32) * (-t * u * u).exp();
        let mut sum = f(0.0) + f(1.0);
        for i in 1..n {
            sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    #[test]
    fn boys_function_matches_quadrature_including_every_branch() {
        // m = 40 exceeds the table, and with t = 36.0 and 60.0 takes the series rather than the upward recursion.
        for &m in &[0, 1, 3, 6, 10, 40] {
            for &t in &[0.0, 1e-6, 0.3, 4.0, 20.0, 34.9, 35.1, 35.99, 36.0, 60.0, 200.0] {
                let reference = boys_by_quadrature(m, t);
                let value = crate::boys::compute_boys_function(m, t);
                assert!((value - reference).abs() < 1e-12 * reference.max(1e-300) + 1e-15, "F_{m}({t}) = {value} vs {reference}");
            }
        }
    }

    fn kernel_value(kind: usize, r: f64) -> f64 {
        match kind {
            0 => 1.0 / r,
            1 => crate::special_functions::erf(0.83 * r) / r,
            _ => (-1.7 * r * r).exp() / r,
        }
    }

    fn kernel_b(kind: usize, m: usize, r: f64) -> f64 {
        match kind {
            0 => Coulomb.b(m, r),
            1 => ErfCoulomb { kappa: 0.83 }.b(m, r),
            _ => GaussianScreenedCoulomb { c: 1.7 }.b(m, r),
        }
    }

    #[test]
    fn b_zero_is_the_kernel_and_each_b_is_the_radial_derivative_of_the_previous() {
        for kind in 0..3 {
            for &r in &[0.4, 1.3, 2.9] {
                assert!((kernel_b(kind, 0, r) - kernel_value(kind, r)).abs() < 1e-13);
                for m in 0..5 {
                    let h = 1e-5 * r;
                    let numeric = (kernel_b(kind, m, r + h) - kernel_b(kind, m, r - h)) / (2.0 * h) / r;
                    let analytic = kernel_b(kind, m + 1, r);
                    assert!(
                        (numeric - analytic).abs() < 1e-6 * analytic.abs().max(1e-3),
                        "kernel {kind}, B_{}({r}): {analytic} vs finite difference {numeric}",
                        m + 1
                    );
                }
            }
        }
    }

    #[test]
    fn coulomb_derivatives_match_closed_forms() {
        let d: [f64; 3] = [0.7, -1.1, 0.4];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let r = r2.sqrt();
        let mut out = Vec::new();
        kernel_derivatives(&Coulomb, d, 3, &mut out);
        assert!((out[moment_index(0, 0, 0)] - 1.0 / r).abs() < 1e-13);
        assert!((out[moment_index(1, 0, 0)] + d[0] / r.powi(3)).abs() < 1e-13);
        assert!((out[moment_index(2, 0, 0)] - (3.0 * d[0] * d[0] - r2) / r.powi(5)).abs() < 1e-13);
        assert!((out[moment_index(1, 1, 0)] - 3.0 * d[0] * d[1] / r.powi(5)).abs() < 1e-13);
        assert!((out[moment_index(1, 1, 1)] + 15.0 * d[0] * d[1] * d[2] / r.powi(7)).abs() < 1e-13);
    }

    #[test]
    fn coulomb_derivatives_are_harmonic_at_every_order() {
        // Laplace: D^(a+2,b,c) + D^(a,b+2,c) + D^(a,b,c+2) = 0 for 1/r, at any (a,b,c).
        let d = [0.9, 0.35, -1.4];
        let mut out = Vec::new();
        kernel_derivatives(&Coulomb, d, 6, &mut out);
        for order in 0..=4 {
            for a in 0..=order {
                for b in 0..=order - a {
                    let c = order - a - b;
                    let laplacian = out[moment_index(a + 2, b, c)] + out[moment_index(a, b + 2, c)] + out[moment_index(a, b, c + 2)];
                    let scale = out[moment_index(a + 2, b, c)].abs().max(1e-3);
                    assert!(laplacian.abs() < 1e-10 * scale, "({a},{b},{c}): {laplacian}");
                }
            }
        }
    }

    #[test]
    fn derivatives_are_derivatives_of_the_previous_order_for_every_kernel() {
        // D^(k + e_axis) g must equal the central difference of D^k g along that axis.
        let d = [0.8, -0.6, 1.1];
        let max_order = 4;
        let h = 1e-4;
        for kind in 0..3 {
            let evaluate = |point: [f64; 3]| {
                let mut out = Vec::new();
                match kind {
                    0 => kernel_derivatives(&Coulomb, point, max_order, &mut out),
                    1 => kernel_derivatives(&ErfCoulomb { kappa: 0.83 }, point, max_order, &mut out),
                    _ => kernel_derivatives(&GaussianScreenedCoulomb { c: 1.7 }, point, max_order, &mut out),
                }
                out
            };
            let centre = evaluate(d);
            for order in 0..max_order {
                for_each_cartesian_component(order, |lx, ly, lz| {
                        for (axis, higher) in [(0, (lx + 1, ly, lz)), (1, (lx, ly + 1, lz)), (2, (lx, ly, lz + 1))] {
                            let mut plus = d;
                            let mut minus = d;
                            plus[axis] += h;
                            minus[axis] -= h;
                            let numeric = (evaluate(plus)[moment_index(lx, ly, lz)] - evaluate(minus)[moment_index(lx, ly, lz)]) / (2.0 * h);
                            let analytic = centre[moment_index(higher.0, higher.1, higher.2)];
                            assert!(
                                (numeric - analytic).abs() < 1e-5 * analytic.abs().max(1e-2),
                                "kernel {kind}, D^{higher:?}: {analytic} vs finite difference {numeric}"
                            );
                        }
                });
            }
        }
    }
}
