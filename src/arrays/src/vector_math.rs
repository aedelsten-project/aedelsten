// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Operations on vectors held as slices or 3-arrays. The two-vector operations check that the lengths agree.

use crate::blas_interface::{daxpy, ddot};

/// `a += b`.
#[inline(always)]
pub fn add_in_place(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "add_in_place: lengths {} and {}", a.len(), b.len());
    let n = a.len() as i32;
    let incx = 1;
    let incy = 1;
    let a = a.as_ptr();
    let b = b.as_ptr();
    let alpha = 1.0;
    daxpy(n, alpha, b, incx, a, incy)
}

/// `a += alpha b`.
#[inline(always)]
pub fn add_scaled_in_place(a: &mut [f64], b: &[f64], alpha: f64) {
    assert_eq!(a.len(), b.len(), "add_scaled_in_place: lengths {} and {}", a.len(), b.len());
    let n = a.len() as i32;
    let incx = 1;
    let incy = 1;
    let a = a.as_ptr();
    let b = b.as_ptr();
    daxpy(n, alpha, b, incx, a, incy)
}

/// `a -= b`.
#[inline(always)]
pub fn subtract_in_place(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "subtract_in_place: lengths {} and {}", a.len(), b.len());
    let n = a.len() as i32;
    let incx = 1;
    let incy = 1;
    let a = a.as_ptr();
    let b = b.as_ptr();
    let alpha = -1.0;
    daxpy(n, alpha, b, incx, a, incy)
}

/// `vec1 - vec2`.
#[inline(always)]
pub fn vector_subtract_3d(vec1: &[f64; 3], vec2: &[f64; 3]) -> [f64; 3] {
    let mut output: [f64; 3] = [0.0; 3];
    output[0] = vec1[0] - vec2[0];
    output[1] = vec1[1] - vec2[1];
    output[2] = vec1[2] - vec2[2];
    return output;
}

/// The Euclidean norm.
#[inline(always)]
pub fn norm(vec: &[f64]) -> f64 {
    let norm = vec.iter().map(|x| x * x).sum::<f64>();
    norm.sqrt()
}

/// |a - b|, without forming the difference.
#[inline(always)]
pub fn distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "distance: lengths {} and {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// The Euclidean norm.
#[inline(always)]
pub fn vector_norm_3d(vec: &[f64; 3]) -> f64 {
    let norm = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    norm.sqrt()
}

/// The squared Euclidean norm.
#[inline(always)]
pub fn vector_norm_square_3d(vec: &[f64; 3]) -> f64 {
    vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
}

/// `vec *= scale`.
#[inline(always)]
pub fn scale_in_place(vec: &mut [f64], scale: f64) {
    vec.iter_mut().for_each(|x| *x *= scale)
}

/// `a . b`.
#[inline(always)]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: lengths {} and {}", a.len(), b.len());
    let n = a.len() as i32;
    let incx = 1;
    let incy = 1;
    let a = a.as_ptr();
    let b = b.as_ptr();
    let res = ddot(n, a, incx, b, incy);
    res
}

