// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Symmetric 3x3 blocks, stored packed as [xx, xy, xz, yy, yz, zz], the canonical Cartesian order.

use sources::Polarizability;

/// A packed symmetric 3x3 block.
pub type Packed = [f64; 6];

/// The polarizability tensor, packed.
pub fn from_polarizability(polarizability: &Polarizability) -> Packed {
    match *polarizability {
        Polarizability::Isotropic(alpha) => [alpha, 0.0, 0.0, alpha, 0.0, alpha],
        Polarizability::Anisotropic(components) => components,
    }
}

/// The block times the 3-vector `v`.
pub fn apply(t: &Packed, v: &[f64]) -> [f64; 3] {
    [
        t[0] * v[0] + t[1] * v[1] + t[2] * v[2],
        t[1] * v[0] + t[3] * v[1] + t[4] * v[2],
        t[2] * v[0] + t[4] * v[1] + t[5] * v[2],
    ]
}

/// The full 3x3 matrix.
pub fn to_matrix(t: &Packed) -> [[f64; 3]; 3] {
    [[t[0], t[1], t[2]], [t[1], t[3], t[4]], [t[2], t[4], t[5]]]
}

/// The inverse, or `None` if the block is singular (relative to its own scale).
pub fn invert(t: &Packed) -> Option<Packed> {
    let (a, b, c, d, e, f) = (t[0], t[1], t[2], t[3], t[4], t[5]);
    let determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    let scale = t.iter().fold(0.0_f64, |largest, x| largest.max(x.abs()));
    if !determinant.is_finite() || determinant.abs() <= 1e-12 * scale * scale * scale {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    Some([
        (d * f - e * e) * inverse_determinant,
        (c * e - b * f) * inverse_determinant,
        (b * e - c * d) * inverse_determinant,
        (a * f - c * c) * inverse_determinant,
        (b * c - a * e) * inverse_determinant,
        (a * d - b * b) * inverse_determinant,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_times_block_is_the_identity() {
        let t: Packed = [4.0, 1.0, 3.0, 0.5, 0.25, 2.0];
        let inverse = invert(&t).unwrap();
        for column in 0..3 {
            let mut unit = [0.0; 3];
            unit[column] = 1.0;
            let round_trip = apply(&t, &apply(&inverse, &unit));
            for row in 0..3 {
                assert!((round_trip[row] - unit[row]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn singular_blocks_have_no_inverse_and_isotropic_inverts_to_one_over_alpha() {
        assert!(invert(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).is_none());
        let inverse = invert(&from_polarizability(&Polarizability::Isotropic(4.0))).unwrap();
        assert_eq!(inverse, [0.25, 0.0, 0.0, 0.25, 0.0, 0.25]);
    }
}
