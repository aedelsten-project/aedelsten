// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The gamma and incomplete gamma functions, erf, factorials and binomial coefficients.

const MAX_ITERATION: i32 = 10000;
const TOLERANCE: f64 = 1.0e-15;
const FLOAT_MINIMUM: f64 = 1.0e-30;

// The Lanczos parameter g and the coefficients of `lanczos_gamma`.
const G: f64 = 8.0;
const NP: usize = 12;
const P: [f64; NP] = [
    0.9999999999999999298,
    1975.3739023578852322,
    -4397.3823927922428918,
    3462.6328459862717019,
    -1156.9851431631167820,
    154.53815050252775060,
    -6.2536716123689161798,
    0.034642762454736807441,
    -7.4776171974442977377e-7,
    6.3041253821852264261e-8,
    -2.7405717035683877489e-8,
    4.0486948817567609101e-9,
];

/// Gamma(x), x > 0, by the Lanczos approximation with g = 8 and 12 terms.
pub fn lanczos_gamma(x: f64) -> f64 {
    let xs = x - 1.0;
    let mut sum = P[0];
    for i in 1..NP {
        sum += P[i] / (xs + i as f64);
    }
    let t = xs + G + 0.5;
    let gamma = (2.0 * std::f64::consts::PI).sqrt() * t.powf(xs + 0.5) * (-t).exp() * sum;
    return gamma;
}

/// n!! (1 for n = 0 and 1).
pub fn double_factorial(n: u64) -> u64 {
    if n == 0 || n == 1 {
        return 1;
    }
    let mut result = 1;
    let mut current = n;
    while current > 1 {
        result *= current;
        current -= 2;
    }
    result
}

/// The error function, P(1/2, x^2) with the sign of x.
pub fn erf(x: f64) -> f64 {
    if x < 0.0 {
        return -compute_gamma_inc(0.5, x*x);
    } else {
        return compute_gamma_inc(0.5, x*x);
    }
}

/// The regularized lower incomplete gamma function P(a, x): its series for x < a + 1, else 1 - Q(a, x) by the
/// continued fraction.
pub fn compute_gamma_inc(a: f64, x: f64) -> f64 {
    let use_series: bool = x < a + 1.0;
    let gamma_inc: f64;
    if use_series {
        gamma_inc = compute_gamma_inc_as_series(a, x);
    } else {
        gamma_inc = 1.0 - compute_gamma_inc_as_continued_fraction(a, x);
    }
    return gamma_inc;
}

/// P(a, x) = exp(-x) x^a sum_n x^n / Gamma(a + 1 + n), the regularized lower incomplete gamma function by its series.
/// Panics if the series has not converged after `MAX_ITERATION` terms.
pub fn compute_gamma_inc_as_series(a: f64, x: f64) -> f64 {
    let mut a_tmp: f64 = a;
    let mut sum: f64 = 1.0 / a;
    let mut delta: f64 = sum;
    let mut converged: bool = false;
    'iterator: for _iteration in 0..MAX_ITERATION {
        a_tmp += 1.0;
        delta = delta * x / a_tmp;
        sum += delta;
        if delta.abs() < sum.abs() * TOLERANCE {
            converged = true;
            break 'iterator;
        }
    }
    if !converged {
        panic!(
            "ERROR: Computation in 'compute_gamma_inc_as_series' terminated before convergence.
                       Try increasing MAX_ITERATION.
                       Program is aborted.n"
        );
    }
    let gamma_inc: f64 = compute_gamma_inc_final(a, x, sum);
    return gamma_inc;
}

/// Q(a, x) = exp(-x) x^a / Gamma(a) (1/(x+1-a-) 1(1-a)/(x+3-a-) 2(2-a)/(x+5-a-) ...), the regularized upper
/// incomplete gamma function by its continued fraction, evaluated by the modified Lentz method. Panics if it has not
/// converged after `MAX_ITERATION` steps.
pub fn compute_gamma_inc_as_continued_fraction(a: f64, x: f64) -> f64 {
    let mut b: f64 = x + 1.0 - a;
    let mut c: f64 = 1.0 / FLOAT_MINIMUM;
    let mut d: f64 = 1.0 / b;
    let mut sum: f64 = d;
    let mut delta: f64;
    let mut converged: bool = false;
    'iterator: for i in 1..MAX_ITERATION + 1 {
        let i_float: f64 = i as f64;
        let an: f64 = -i_float * (i_float - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FLOAT_MINIMUM {
            d = FLOAT_MINIMUM;
        }
        c = b + an / c;
        if c.abs() < FLOAT_MINIMUM {
            c = FLOAT_MINIMUM;
        }
        d = 1.0 / d;
        delta = c * d;
        sum = sum * delta;
        if (delta - 1.0).abs() < TOLERANCE {
            converged = true;
            break 'iterator;
        }
    }
    if !converged {
        panic!("ERROR: Computation in 'compute_gamma_inc_as_continued_fraction' terminated before convergence.
                       Try increasing MAX_ITERATION.
                       Program is aborted.n");
    }
    let gamma_inc: f64 = compute_gamma_inc_final(a, x, sum);
    return gamma_inc;
}

/// `sum` times x^a exp(-x) / Gamma(a), the common factor of the series and the continued fraction.
#[inline(always)]
pub fn compute_gamma_inc_final(a: f64, x: f64, sum: f64) -> f64 {
    let factor = x.powf(a) * (-x).exp() / lanczos_gamma(a);
    sum * factor
}

/// n! (overflows `u64` above n = 20).
pub fn compute_factorial(n: u64) -> u64 {
    let mut n_factorial: u64 = 1;
    for nn in 2..=n {
        n_factorial *= nn;
    }
    return n_factorial;
}

// C(n, m) as the running product c_k = c_(k-1) (n - m + k) / k, k = 1..=m: c_k is C(n - m + k, k), so every
// division is exact. Panics if C(n, m) does not fit in u64.
fn binomial_recursion(n: u64, m: u64) -> u64 {
    let mut bin: u128 = 1;
    for k in 1..=m as u128 {
        bin = bin * (n as u128 - m as u128 + k) / k;
    }
    u64::try_from(bin).expect("binomial coefficient overflows u64")
}

/// The binomial coefficient C(n, m), exact; 0 for n < 0, m < 0 or m > n.
pub fn binomial(n: i64, m: i64) -> u64 {
    if n < 0 {
        return 0;
    } else if m < 0 {
        return 0;
    } else if n > m {
        let m_used: f64 = m as f64;
        let n_half: f64 = (n as f64) / 2.0;
        if n_half > m_used {
            return binomial_recursion(n as u64, m as u64);
        } else {
            let m_used: u64 = (n - m) as u64;
            return binomial_recursion(n as u64, m_used);
        }
    } else if n == m {
        return 1;
    } else {
        return 0;
    }
}

#[cfg(test)]
mod tests {
    use super::binomial;

    #[test]
    fn binomial_matches_pascals_triangle() {
        let mut row: Vec<u64> = vec![1];
        for n in 0..=62_i64 {
            for (m, &expected) in row.iter().enumerate() {
                assert_eq!(binomial(n, m as i64), expected, "C({n}, {m})");
            }
            assert_eq!(binomial(n, n + 1), 0);
            assert_eq!(binomial(n, -1), 0);
            let mut next = vec![1; row.len() + 1];
            for m in 1..row.len() {
                next[m] = row[m - 1] + row[m];
            }
            row = next;
        }
        assert_eq!(binomial(13, 6), 1716);
        assert_eq!(binomial(-1, 0), 0);
    }
}
