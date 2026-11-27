// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The Hermite Coulomb integrals `R^0_{tuv}` by the McMurchie-Davidson recursion, organized by shells: shell `l` holds
//! `R^n_{tuv}` with `t + u + v + n = l`, packed by `hermite_index(.., l)` (`n` implied). Shell `l` reads only itself
//! and shell `l - 1`, so two shells are kept, `O(n_max^3)` storage instead of the full tensor's `O(n_max^4)`. Each row
//! `(u, v)` of a shell is contiguous in `t`, and its last entry (`n = 0`) goes to the output.

use crate::hermite_expansion::hermite_index;

/// `R^0_{tuv}` for `t + u + v <= n_max` into `output[hermite_index(t, u, v, n_max)]`, from `boys[n] = (-2p)^n F_n(p
/// R_PC^2)` and `(pcx, pcy, pcz) = P - C`. `shell_a` and `shell_b` are scratch, each at least `hermite_count(n_max)`
/// long.
pub fn compute_r_tensor_rolling(
    output: &mut [f64],
    boys: &[f64],
    n_max: usize,
    pcx: f64,
    pcy: f64,
    pcz: f64,
    shell_a: &mut [f64],
    shell_b: &mut [f64],
) {
    output[0] = boys[0];
    shell_a[0] = boys[0];
    let mut prev: &mut [f64] = shell_a;
    let mut curr: &mut [f64] = shell_b;

    for l in 1..=n_max {
        // t axis (u = v = 0): sequential in t.
        curr[0] = boys[l];
        curr[1] = pcx * curr[0];
        for t in 2..=l {
            curr[t] = pcx * curr[t - 1] + (t - 1) as f64 * prev[t - 2];
        }
        output[hermite_index(l, 0, 0, n_max)] = curr[l];

        // The other rows (u, v) of shell l, each a whole-row step along y (plane v = 0, from row (u - 1, 0)) or z
        // (from row (u, v - 1)), plus at u (or v) >= 2 row (u - 2, 0) (or (u, v - 2)) of shell l - 1. The row offsets
        // run along each plane: in plane v of shell l row u is l - v - u + 1 long.
        let (mut dst, mut src, mut src2, mut out) = (l + 1, 0, 0, n_max + 1);
        for u in 1..=l {
            let len = l - u + 1;
            let p = (u >= 2).then(|| &prev[src2..src2 + len]);
            row_step(curr, dst, src, p, pcy, (u - 1) as f64, len);
            output[out + len - 1] = curr[dst + len - 1];
            if u >= 2 { src2 += len + 1; }
            src = dst;
            dst += len;
            out += n_max - u + 1;
        }
        for v in 1..=l {
            let m = l - v;
            let mut src = hermite_index(0, 0, v - 1, l);
            let mut src2 = if v >= 2 { hermite_index(0, 0, v - 2, l - 1) } else { 0 };
            let mut out = hermite_index(0, 0, v, n_max);
            for u in 0..=m {
                let len = m - u + 1;
                let p = (v >= 2).then(|| &prev[src2..src2 + len]);
                row_step(curr, dst, src, p, pcz, (v - 1) as f64, len);
                output[out + len - 1] = curr[dst + len - 1];
                dst += len;
                src += len + 1;
                src2 += len + 1;
                out += n_max - v - u + 1;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
}

/// `curr[dst..][..len] = a curr[src..][..len] + k p` (`src < dst`, `p` absent as zero), on slices so it vectorizes.
#[inline(always)]
fn row_step(curr: &mut [f64], dst: usize, src: usize, p: Option<&[f64]>, a: f64, k: f64, len: usize) {
    let (lo, hi) = curr.split_at_mut(dst);
    let (d, s) = (&mut hi[..len], &lo[src..src + len]);
    match p {
        Some(p) => d.iter_mut().zip(s).zip(p).for_each(|((d, s), p)| *d = a * s + k * p),
        None => d.iter_mut().zip(s).for_each(|(d, s)| *d = a * s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hermite_coulomb_integrals::compute_r_only_boys;
    use crate::hermite_expansion::hermite_count;

    /// `R_tuv` by the closed form that follows from `d/dX f_m = X f_(m+1)` (f_m the auxiliaries as functions of the
    /// separation): `d^t/dX^t f_m = Sum_i c(t, i) X^(t - 2i) f_(m + t - i)`, `c(t, i) = t! / (2^i i! (t - 2i)!)`, and
    /// likewise in y and z, so `R_tuv = Sum_ijk c(t, i) c(u, j) c(v, k) X^(t-2i) Y^(u-2j) Z^(v-2k)
    /// R^(t+u+v-i-j-k)_000`. Direct sums, no recursion.
    fn closed_form(n_max: usize, boys: &[f64], pcx: f64, pcy: f64, pcz: f64) -> Vec<f64> {
        let factorial = |n: usize| (1..=n).fold(1.0, |acc, k| acc * k as f64);
        let c = |t: usize, i: usize| factorial(t) / (2f64.powi(i as i32) * factorial(i) * factorial(t - 2 * i));
        let mut out = vec![0.0; hermite_count(n_max)];
        for v in 0..=n_max {
            for u in 0..=n_max - v {
                for t in 0..=n_max - v - u {
                    let mut sum = 0.0;
                    for i in 0..=t / 2 {
                        for j in 0..=u / 2 {
                            for k in 0..=v / 2 {
                                sum += c(t, i) * c(u, j) * c(v, k)
                                    * pcx.powi((t - 2 * i) as i32) * pcy.powi((u - 2 * j) as i32) * pcz.powi((v - 2 * k) as i32)
                                    * boys[t + u + v - i - j - k];
                            }
                        }
                    }
                    out[hermite_index(t, u, v, n_max)] = sum;
                }
            }
        }
        out
    }

    /// The rolling recursion against the closed form, to n_max = 24 ((ii|ii)), for exponents and separations on a
    /// centre, near and far. Both lose digits to cancellation at high order and large p R_PC^2: the closed form's terms
    /// alternate in sign, and at n_max = 24, p = 6, R_PC^2 = 10 they reach 7e16 for an R_24,0,0 of -1.4e10, which a
    /// 60-digit evaluation puts at -14159873782.19 (the closed form 1.1e-9 off, the recursion 2.2e-9). The bound,
    /// relative to the tensor's largest element, grows tenfold per four orders as the differences do (2e-13 at 12,
    /// 3e-12 at 16, 3e-11 at 20, 1e-9 at 24), with a margin of 10 to 50.
    #[test]
    fn rolling_matches_the_closed_form() {
        let mut largest: f64 = 0.0;
        let separations = [[0.0, 0.0, 0.0], [0.3, 0.7, -0.5], [1.1, -0.2, 0.4], [2.5, -1.7, 0.9]];
        let ps = [0.05, 0.4, 1.5, 6.0];
        for n_max in [0usize, 1, 2, 3, 6, 9, 12, 16, 20, 24] {
            let n = hermite_count(n_max);
            let (mut rolling, mut shell_a, mut shell_b) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            let mut boys = vec![0.0; n_max + 1];
            for &p in &ps {
                for pc in &separations {
                    let rpc_square = pc.iter().map(|x| x * x).sum();
                    compute_r_only_boys(&mut boys, n_max, p, rpc_square);
                    compute_r_tensor_rolling(&mut rolling, &boys, n_max, pc[0], pc[1], pc[2], &mut shell_a, &mut shell_b);
                    let expected = closed_form(n_max, &boys, pc[0], pc[1], pc[2]);
                    let scale = expected.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                    for (index, (x, y)) in rolling.iter().zip(&expected).enumerate() {
                        let relative = (x - y).abs() / scale.max(1e-300);
                        largest = largest.max(relative);
                        assert!(relative <= 1e-14 * 10f64.powf(n_max as f64 / 4.0), "n_max {n_max}, p {p}, pc {pc:?}, [{index}]: {x} against {y}");
                    }
                }
            }
        }
        eprintln!("largest difference relative to the tensor's largest element {largest:e}");
    }
}
