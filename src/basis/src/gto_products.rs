// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The product of two primitive Gaussians: their exponents and the product centre.

/// `(a pos_a + b pos_b) / p`: the centre of the product of two Gaussians of exponents `a` and `b`, `p = a + b`.
#[inline(always)]
pub fn gaussian_product_centre(
    pos_a: &[f64; 3],
    a: f64,
    pos_b: &[f64; 3],
    b: f64,
    p: f64,
) -> [f64; 3] {
    let mut pos_ab: [f64; 3] = [0.0, 0.0, 0.0];
    pos_ab[0] = (a * pos_a[0] + b * pos_b[0]) / p;
    pos_ab[1] = (a * pos_a[1] + b * pos_b[1]) / p;
    pos_ab[2] = (a * pos_a[2] + b * pos_b[2]) / p;
    pos_ab
}

/// A product of two primitive Gaussians with exponents `a` and `b`.
pub trait GTOProductTrait {
    /// The first exponent.
    fn get_a(&self) -> f64;
    /// The second exponent.
    fn get_b(&self) -> f64;
    /// The total exponent, `p = a + b`.
    fn get_p(&self) -> f64;
    /// The reduced exponent, `μ = a b / p`.
    fn get_μ(&self) -> f64;
    /// The product centre, `P = (a A + b B) / p`.
    fn get_product_centre(&self) -> [f64; 3];
}

/// A product of two primitives.
#[derive(Debug)]
pub struct GTOProduct {
    a: f64,
    b: f64,
    pos: [f64; 3],
}

impl GTOProduct {
    /// The product with exponents `a`, `b` and centre `pos`.
    pub fn new(a: f64, b: f64, pos: [f64; 3]) -> Self {
        Self { a, b, pos }
    }
}

impl GTOProductTrait for GTOProduct {
    fn get_a(&self) -> f64 {
        self.a
    }

    fn get_b(&self) -> f64 {
        self.b
    }

    fn get_p(&self) -> f64 {
        self.a + self.b
    }

    fn get_μ(&self) -> f64 {
        self.a * self.b / (self.a + self.b)
    }

    fn get_product_centre(&self) -> [f64; 3] {
        self.pos
    }
}

/// A product of two primitives with the product of their contraction coefficients.
pub trait ContractedGTOProductTrait: GTOProductTrait {
    /// The product of the contraction coefficients.
    fn get_cc(&self) -> f64;
}

/// A product of two contracted primitives.
#[derive(Debug)]
pub struct CGTOProduct {
    a: f64,
    b: f64,
    cc: f64,
    pos: [f64; 3],
}

impl CGTOProduct {
    /// The product with exponents `a`, `b`, coefficient product `cc` and centre `pos`.
    pub fn new(a: f64, b: f64, cc: f64, pos: [f64; 3]) -> Self {
        Self { a, b, cc, pos }
    }
}

impl GTOProductTrait for CGTOProduct {
    fn get_a(&self) -> f64 {
        self.a
    }

    fn get_b(&self) -> f64 {
        self.b
    }

    fn get_p(&self) -> f64 {
        self.a + self.b
    }

    fn get_μ(&self) -> f64 {
        self.a * self.b / (self.a + self.b)
    }

    fn get_product_centre(&self) -> [f64; 3] {
        self.pos
    }
}

impl ContractedGTOProductTrait for CGTOProduct {
    fn get_cc(&self) -> f64 {
        self.cc
    }
}
