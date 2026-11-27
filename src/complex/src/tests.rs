// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use crate::Complex;
#[test]
fn test_complex_add() {
    let c1: Complex<f64> = Complex::<f64>::new(2.0, 3.0);
    let c2: Complex<f64> = Complex::<f64>::new(4.0, 7.0);
    let correct: Complex<f64> = Complex::<f64>::new(6.0, 10.0);
    let result: Complex<f64> = c1 + c2;
    assert_eq!(correct, result);
}
#[test]
fn test_complex_sub() {
    let c1: Complex<f64> = Complex::<f64>::new(2.0, 3.0);
    let c2: Complex<f64> = Complex::<f64>::new(4.0, 7.0);
    let correct: Complex<f64> = Complex::<f64>::new(-2.0, -4.0);
    let result: Complex<f64> = c1 - c2;
    assert_eq!(correct, result);
}
#[test]
fn test_complex_mul() {
    let c1: Complex<f64> = Complex::<f64>::new(2.0, 3.0);
    let c2: Complex<f64> = Complex::<f64>::new(4.0, 7.0);
    let correct: Complex<f64> = Complex::<f64>::new(-13.0, 26.0);
    let result: Complex<f64> = c1 * c2;
    assert_eq!(correct, result);
}
#[test]
fn test_complex_powf_1() {
    let c1: Complex<f64> = Complex::<f64>::new(2.0, 3.0);
    let correct: Complex<f64> = Complex::<f64>::new(1.8644405927881138, 1.432702373014988);
    let result: Complex<f64> = c1.powf(2.0 / 3.0);
    assert_eq!(correct, result);
}
#[test]
fn test_complex_powf_2() {
    let c1: Complex<f64> = Complex::<f64>::new(-1.0, 0.0);
    let sqrt: f64 = 0.5;
    let correct: Complex<f64> = Complex::<f64>::new(6.123233995736766e-17, 1.0);
    let result: Complex<f64> = c1.powf(sqrt);
    assert_eq!(correct, result);
}
#[test]
fn test_complex_mulf() {
    let c1: Complex<f64> = Complex::<f64>::new(2.0, 3.0);
    let f1: f64 = 2.0;
    let correct: Complex<f64> = Complex::<f64>::new(4.0, 6.0);
    let result: Complex<f64> = c1.mulf(f1);
    assert_eq!(correct, result);
}
