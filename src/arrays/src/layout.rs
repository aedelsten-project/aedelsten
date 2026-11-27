// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Column-major index arithmetic shared by the 3D-8D array types: `dims[0]` varies fastest.

use std::ops::Range;

/// The flat position of `index` (one entry per dimension, or fewer for the leading dimensions only).
#[inline(always)]
pub(crate) fn offset(dims: &[usize], index: &[usize]) -> usize {
    index.iter().zip(dims).rev().fold(0, |acc, (i, d)| acc * d + i)
}

/// The contiguous block left when the trailing `fixed.len()` indices are fixed: a sub-array of the leading dimensions.
#[inline(always)]
pub(crate) fn block(dims: &[usize], fixed: &[usize]) -> Range<usize> {
    let leading = dims.len() - fixed.len();
    debug_check_index!(fixed, &dims[leading..]);
    let len: usize = dims[..leading].iter().product();
    let start = len * offset(&dims[leading..], fixed);
    start..start + len
}

/// The sizes after merging adjacent dimensions: group `k` takes the next `groups[k]` dimensions.
#[inline(always)]
pub(crate) fn grouped<const M: usize>(dims: &[usize], groups: [usize; M]) -> [usize; M] {
    assert!(
        groups.iter().all(|&g| g > 0) && groups.iter().sum::<usize>() == dims.len(),
        "grouping {groups:?} does not cover the {} dimensions", dims.len()
    );
    let mut sizes = [0; M];
    let mut first = 0;
    for (size, g) in sizes.iter_mut().zip(groups) {
        *size = dims[first..first + g].iter().product();
        first += g;
    }
    sizes
}

/// Asserts that a reshape keeps the number of elements.
#[inline(always)]
pub(crate) fn check_reshape(from: &[usize], to: &[usize]) {
    let (n, m) = (from.iter().product::<usize>(), to.iter().product::<usize>());
    assert_eq!(n, m, "reshape from {from:?} ({n} elements) to {to:?} ({m} elements)");
}

/// `&data[k]`, unchecked with the `unchecked` feature.
#[inline(always)]
pub(crate) fn element<T>(data: &[T], k: usize) -> &T {
    #[cfg(not(feature = "unchecked"))]
    return &data[k];
    #[cfg(feature = "unchecked")]
    return unsafe { data.get_unchecked(k) };
}

/// `&mut data[k]`, unchecked with the `unchecked` feature.
#[inline(always)]
pub(crate) fn element_mut<T>(data: &mut [T], k: usize) -> &mut T {
    #[cfg(not(feature = "unchecked"))]
    return &mut data[k];
    #[cfg(feature = "unchecked")]
    return unsafe { data.get_unchecked_mut(k) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_blocks_and_groups() {
        let dims = [2, 3, 4];
        assert_eq!(offset(&dims, &[1, 2, 3]), 1 + 2 * (2 + 3 * 3));
        assert_eq!(block(&dims, &[3]), 18..24);
        assert_eq!(block(&dims, &[2, 3]), 22..24);
        assert_eq!(grouped(&dims, [1, 2]), [2, 12]);
        assert_eq!(grouped(&dims, [2, 1]), [6, 4]);
    }

    #[test]
    #[should_panic(expected = "does not cover")]
    fn a_grouping_must_cover_every_dimension() {
        grouped(&[2, 3, 4], [1, 1]);
    }
}
