// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The Boys function `F_n(x) = int_0^1 t^(2n) exp(-x t^2) dt` for `n = 0..=n_max` at once. For `x < 36`, a 7th-order
//! Taylor expansion of every `F_n` around the nearest point of a grid with step 0.1 (`dF_n/dx = -F_(n+1)`); for
//! `x >= 36`, `F_0 = sqrt(pi/x) / 2` and the upward recursion `F_(n+1) = ((2n+1) F_n - exp(-x)) / (2x)` while
//! `2x > 2 n_max + 1`. An `n_max` above `BOYS_TABLE_N_MAX` takes the series for `F_(n_max)`, then the downward
//! recursion `F_n = (2x F_(n+1) + exp(-x)) / (2n+1)`.

use constants::{L_DERIVATIVE, L_MAX};
use std::sync::OnceLock;

/// The largest `n` the table holds: a quartet's highest total angular momentum, `4 L_MAX`, raised by derivatives up
/// to `L_DERIVATIVE`.
pub const BOYS_TABLE_N_MAX: usize = 4 * L_MAX + L_DERIVATIVE;

const TAYLOR_ORDER: usize = 7;
const COLUMNS: usize = BOYS_TABLE_N_MAX + TAYLOR_ORDER + 1;
const STEP: f64 = 0.1;
const INV_STEP: f64 = 10.0;
const X_SWITCH: f64 = 36.0;
const ROWS: usize = 361; // x_k = k * STEP for 0 <= x_k <= X_SWITCH

const fn build_inv_taylor() -> [f64; TAYLOR_ORDER] {
    let mut table = [0.0; TAYLOR_ORDER];
    let mut j = 0;
    while j < TAYLOR_ORDER {
        table[j] = 1.0 / (j + 1) as f64;
        j += 1;
    }
    table
}
// INV_TAYLOR[j] = 1 / (j + 1), for the Horner form of the Taylor expansion.
const INV_TAYLOR: [f64; TAYLOR_ORDER] = build_inv_taylor();

const fn build_inv_odd() -> [f64; BOYS_TABLE_N_MAX] {
    let mut table = [0.0; BOYS_TABLE_N_MAX];
    let mut n = 0;
    while n < BOYS_TABLE_N_MAX {
        table[n] = 1.0 / (2 * n + 1) as f64;
        n += 1;
    }
    table
}
// INV_ODD[n] = 1 / (2n + 1), for the downward recursion.
const INV_ODD: [f64; BOYS_TABLE_N_MAX] = build_inv_odd();

static TABLE: OnceLock<Vec<[f64; COLUMNS]>> = OnceLock::new();

// F_n(x_k) for every grid point, one row per x_k, built on first use: the top column from the series, the others
// from the downward recursion.
#[inline(always)]
fn table() -> &'static [[f64; COLUMNS]] {
    TABLE.get_or_init(|| {
        (0..ROWS)
            .map(|k| {
                let x = k as f64 * STEP;
                let expx = (-x).exp();
                let mut row = [0.0; COLUMNS];
                row[COLUMNS - 1] = boys_series(COLUMNS - 1, x);
                for n in (0..COLUMNS - 1).rev() {
                    row[n] = (2.0 * x * row[n + 1] + expx) / (2 * n + 1) as f64;
                }
                row
            })
            .collect()
    })
}

/// F_n(x) from the series F_n(x) = e^{-x} \sum_k (2x)^k / ((2n+1)(2n+3)...(2n+2k+1)), all terms
/// positive. Exact to rounding for any x; needs ~2x terms.
pub fn boys_series(n: usize, x: f64) -> f64 {
    let two_x = 2.0 * x;
    let mut denominator = (2 * n + 1) as f64;
    let mut term = 1.0 / denominator;
    let mut sum = term;
    loop {
        denominator += 2.0;
        term *= two_x / denominator;
        sum += term;
        if term <= sum * 1e-17 {
            break;
        }
    }
    (-x).exp() * sum
}

/// F_0(x) alone, from the table or, for x >= 36, sqrt(pi/x) / 2.
#[inline]
pub fn compute_boys_f0(x: f64) -> f64 {
    if x >= X_SWITCH {
        0.5 * (std::f64::consts::PI / x).sqrt()
    } else {
        taylor(table(), x, 0)
    }
}

/// F_n(x) for n = 0..values.len(), written to `values`. x = p R^2 >= 0.
pub fn compute_boys_functions(x: f64, values: &mut [f64]) {
    debug_assert!(x >= 0.0, "Boys function argument must be non-negative, got {x}");
    let Some(n_max) = values.len().checked_sub(1) else {
        return;
    };
    if x >= X_SWITCH && 2.0 * x > (2 * n_max + 1) as f64 {
        upward_recursion(x, values);
    } else {
        if n_max <= BOYS_TABLE_N_MAX && x < X_SWITCH {
            taylor_all(table(), x, values);
        } else {
            values[n_max] = boys_series(n_max, x);
            downward_recursion(x, values);
        }
    }
}

/// F_n(x) alone, by the same branches as `compute_boys_functions`: the upward recursion from F_0 for x >= 36 and
/// 2x > 2n + 1, the table's Taylor expansion for n <= `BOYS_TABLE_N_MAX` and x < 36, else the series.
pub fn compute_boys_function(n: usize, x: f64) -> f64 {
    debug_assert!(x >= 0.0, "Boys function argument must be non-negative, got {x}");
    if x >= X_SWITCH && 2.0 * x > (2 * n + 1) as f64 {
        let exp_minus_x = (-x).exp();
        let mut f = 0.5 * (std::f64::consts::PI / x).sqrt();
        for order in 0..n {
            f = ((2 * order + 1) as f64 * f - exp_minus_x) / (2.0 * x);
        }
        f
    } else if n <= BOYS_TABLE_N_MAX && x < X_SWITCH {
        taylor(table(), x, n)
    } else {
        boys_series(n, x)
    }
}

// F_n(x) by the Taylor expansion around the grid point nearest x (x < 36).
#[inline(always)]
fn taylor(table: &[[f64; COLUMNS]], x: f64, n: usize) -> f64 {
    let k = (x * INV_STEP + 0.5) as usize;
    let row = &table[k];
    let delta = x - k as f64 * STEP;
    // F_n(x_k + d) = \sum_j F_{n+j}(x_k) (-d)^j / j!, in Horner form.
    let mut f = row[n + TAYLOR_ORDER];
    for j in (0..TAYLOR_ORDER).rev() {
        f = row[n + j] - delta * INV_TAYLOR[j] * f;
    }
    f
}

// Every F_n(x), n = 0..values.len(), by its own Taylor expansion around the same grid point (x < 36): no exp(-x) and no
// recursion, the expansions independent of each other.
#[inline(always)]
fn taylor_all(table: &[[f64; COLUMNS]], x: f64, values: &mut [f64]) {
    let k = (x * INV_STEP + 0.5) as usize;
    let row = &table[k];
    let delta = x - k as f64 * STEP;
    for (n, value) in values.iter_mut().enumerate() {
        let mut f = row[n + TAYLOR_ORDER];
        for j in (0..TAYLOR_ORDER).rev() {
            f = row[n + j] - delta * INV_TAYLOR[j] * f;
        }
        *value = f;
    }
}

// values[n_max] is set; fills values[n_max - 1] down to values[0].
#[inline(always)]
fn downward_recursion(x: f64, values: &mut [f64]) {
    let n_max = values.len() - 1;
    if n_max == 0 {
        return;
    }
    let expx = (-x).exp();
    let two_x = 2.0 * x;
    for n in (0..n_max).rev() {
        let inv_odd = if n < BOYS_TABLE_N_MAX { INV_ODD[n] } else { 1.0 / (2 * n + 1) as f64 };
        values[n] = (two_x * values[n + 1] + expx) * inv_odd;
    }
}

// F_0(x) = sqrt(pi/x) / 2, then the upward recursion for values[1..].
#[inline(always)]
fn upward_recursion(x: f64, values: &mut [f64]) {
    values[0] = 0.5 * (std::f64::consts::PI / x).sqrt();
    if values.len() == 1 {
        return;
    }
    let expx = (-x).exp();
    let inv_two_x = 0.5 / x;
    let mut two_n_plus_one = 1.0;
    for n in 0..values.len() - 1 {
        values[n + 1] = (two_n_plus_one * values[n] - expx) * inv_two_x;
        two_n_plus_one += 2.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Largest relative error of compute_boys_functions against the series, per n, over the xs.
    fn max_relative_errors(n_max: usize, xs: impl Iterator<Item = f64>) -> Vec<(f64, f64)> {
        let mut worst = vec![(0.0, 0.0); n_max + 1];
        let mut values = vec![0.0; n_max + 1];
        for x in xs {
            compute_boys_functions(x, &mut values);
            for n in 0..=n_max {
                let reference = boys_series(n, x);
                let error = ((values[n] - reference) / reference).abs();
                if error > worst[n].0 {
                    worst[n] = (error, x);
                }
            }
        }
        worst
    }

    fn assert_below(worst: &[(f64, f64)], tolerance: impl Fn(usize) -> f64, label: &str) {
        for (n, &(error, x)) in worst.iter().enumerate() {
            assert!(error < tolerance(n), "{label}: n = {n}, relative error {error:e} at x = {x}");
        }
    }

    // Irregular steps so the points land everywhere between grid points, plus the grid points,
    // their midpoints, x = 0 and both sides of the switch to the upward recursion at x = 36.
    fn test_arguments() -> Vec<f64> {
        let mut xs: Vec<f64> = (0..5000).map(|i| i as f64 * 0.0123).collect();
        xs.extend((0..ROWS).map(|k| k as f64 * STEP));
        xs.extend((0..ROWS - 1).map(|k| (k as f64 + 0.5) * STEP));
        xs.extend([0.0, 1e-300, 1e-12, 35.95, 35.999999, X_SWITCH, 36.000001, 36.05, 40.0]);
        xs
    }

    #[test]
    fn table_range_matches_series() {
        let worst = max_relative_errors(BOYS_TABLE_N_MAX, test_arguments().into_iter());
        assert_below(&worst, |_| 1e-14, "table range");
    }

    // Every n_max, not only the largest: the Taylor step starts from a different column each time.
    #[test]
    fn every_n_max_matches_series() {
        let xs = [0.0, 0.03, 0.77, 3.14159, 9.96, 17.2, 29.95, 35.99, 36.0, 51.3];
        let mut values = vec![0.0; BOYS_TABLE_N_MAX + 1];
        for n_max in 0..=BOYS_TABLE_N_MAX {
            for &x in &xs {
                compute_boys_functions(x, &mut values[..=n_max]);
                for n in 0..=n_max {
                    let reference = boys_series(n, x);
                    let error = ((values[n] - reference) / reference).abs();
                    assert!(error < 1e-14, "n_max = {n_max}, n = {n}, x = {x}: relative error {error:e}");
                }
            }
        }
    }

    #[test]
    fn f0_matches_series() {
        for x in test_arguments() {
            let reference = boys_series(0, x);
            let error = ((compute_boys_f0(x) - reference) / reference).abs();
            assert!(error < 1e-14, "x = {x}: relative error {error:e}");
        }
    }

    // Beyond the table: the series for F_{n_max} below the switch, the upward recursion above it
    // only where 2x > 2 n_max + 1.
    #[test]
    fn beyond_table_matches_series() {
        let n_max = BOYS_TABLE_N_MAX + 12;
        let xs = (0..800).map(|i| i as f64 * 0.0731);
        let worst = max_relative_errors(n_max, xs);
        assert_below(&worst, |_| 1e-13, "beyond table");
    }

    // Large x, against the asymptotic form F_n(x) = Gamma(n + 1/2) / (2 x^{n + 1/2}), with Gamma(n + 1/2) built by
    // its recursion.
    #[test]
    fn large_x_matches_asymptotic_form() {
        let n_max = BOYS_TABLE_N_MAX;
        let mut values = vec![0.0; n_max + 1];
        for x in [150.0, 1e3, 1e5, 1e8] {
            compute_boys_functions(x, &mut values);
            let mut gamma = std::f64::consts::PI.sqrt(); // Gamma(n + 1/2), starting at n = 0
            for n in 0..=n_max {
                let a = n as f64 + 0.5;
                let reference = gamma / (2.0 * x.powf(a));
                gamma *= a;
                let error = ((values[n] - reference) / reference).abs();
                assert!(error < 1e-14, "n = {n}, x = {x}: relative error {error:e}");
            }
        }
    }

    // Prints the worst relative error per n; not a check.
    //   cargo test -p math_utilities --release -- --ignored --nocapture boys_error_report
    #[test]
    #[ignore]
    fn boys_error_report() {
        let worst = max_relative_errors(BOYS_TABLE_N_MAX, test_arguments().into_iter());
        for (n, (error, x)) in worst.iter().enumerate() {
            println!("n = {n:>2}: {error:.1e} at x = {x}");
        }
    }
}
