// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! A complex number over a real or integer field, with the arithmetic and the powers the real spherical
//! harmonics' transformation needs.

use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};

#[cfg(test)]
mod tests;

/// A complex number `re + i im`.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Complex<T: ComplexField> {
    re: T,
    im: T,
}

/// A field a `Complex` can be built over: the arithmetic, and the floating-point type its powers are taken in.
pub trait ComplexField: Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + MulAssign
    + Mul<Output = Self>
    + Copy
{
    /// The floating-point type of the powers and the polar form.
    type FloatType: ComplexField + Into<f64> + From<f64>;
    /// The value converted to `FloatType`.
    fn get_float_type(self) -> Self::FloatType;
}

impl ComplexField for f64 {
    type FloatType = f64;
    fn get_float_type(self) -> Self::FloatType {
        self as Self::FloatType
    }
}

impl ComplexField for i64 {
    type FloatType = f64;
    fn get_float_type(self) -> Self::FloatType {
        self as Self::FloatType
    }
}

impl ComplexField for u64 {
    type FloatType = f64;
    fn get_float_type(self) -> Self::FloatType {
        self as Self::FloatType
    }
}

impl ComplexField for usize {
    type FloatType = f64;
    fn get_float_type(self) -> Self::FloatType {
        self as Self::FloatType
    }
}

impl<T> Complex<T>
where
    T: ComplexField + Copy,
{
    /// The number `re + i im`.
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }

    /// The real part.
    pub fn re(&self) -> T {
        self.re
    }

    /// The imaginary part.
    pub fn im(&self) -> T {
        self.im
    }

    /// The number times the real scalar `other`.
    pub fn mulf(&self, other: T) -> Complex<T>
    where
        T: Mul<Output = T>,
    {
        Complex {
            re: self.re * other,
            im: self.im * other,
        }
    }

    /// The `n`-th power, by de Moivre's formula on the polar form (`r^n`, `n theta`).
    pub fn powi(&self, n: i32) -> Complex<T::FloatType>
    where
        T: Mul<Output = T>
    {
        let (rn, ntheta) = self.polar();
        let rn = (rn.get_float_type().into()).powi(n);
        let ntheta = (n as f64) * (ntheta.get_float_type().into());
        Complex {
            re: T::FloatType::from(rn * ntheta.cos()),
            im: T::FloatType::from(rn * ntheta.sin()),
        }
    }

    /// The real power `n`, by de Moivre's formula on the polar form; the principal branch (`theta` in `(-pi, pi]`).
    pub fn powf(&self, n: f64) -> Complex<T::FloatType>
    where
        T: Mul<Output = T>,
    {
        let (rn, ntheta) = self.polar();
        let rn = (rn.get_float_type().into()).powf(n);
        let ntheta = n * (ntheta.get_float_type().into());
        Complex {
            re: T::FloatType::from(rn * ntheta.cos()),
            im: T::FloatType::from(rn * ntheta.sin()),
        }
    }

    /// The modulus and the argument (`atan2(im, re)`).
    fn polar(&self) -> (T::FloatType, T::FloatType) {
        let re_f64 = self.re.get_float_type().into();
        let im_f64 = self.im.get_float_type().into();
        let r = (re_f64.powi(2) + im_f64.powi(2)).sqrt();
        let theta = im_f64.atan2(re_f64);
        (T::FloatType::from(r), T::FloatType::from(theta))
    }
}

impl<T: ComplexField> Add for Complex<T> {
    type Output = Complex<T>;
    fn add(self, other: Complex<T>) -> Complex<T> {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl<T: ComplexField> AddAssign for Complex<T> {
    fn add_assign(&mut self, other: Complex<T>) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl<T: ComplexField> Sub for Complex<T> {
    type Output = Complex<T>;
    fn sub(self, other: Complex<T>) -> Complex<T> {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl<T: ComplexField> Mul for Complex<T> {
    type Output = Complex<T>;
    fn mul(self, other: Complex<T>) -> Complex<T> {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl<T: ComplexField> MulAssign for Complex<T> {
    fn mul_assign(&mut self, other: Complex<T>) {
        let re_tmp = self.re;
        let im_tmp = self.im;
        self.re = re_tmp * other.re - im_tmp * other.im;
        self.im = re_tmp * other.im + im_tmp * other.re;
    }
}
