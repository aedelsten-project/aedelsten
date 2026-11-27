// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The Cartesian component order of the library: within an order `lx` descending, then `ly` descending (xx, xy, xz,
//! yy, yz, zz for d); in "all orders up to n" storage (multipole moments, derivative tables) the orders 0..=n one after
//! the other.

/// Calls `f(lx, ly, lz)` for every Cartesian component of order `order`, in the library's order.
#[inline(always)]
pub fn for_each_cartesian_component(order: usize, mut f: impl FnMut(usize, usize, usize)) {
    for lx in (0..=order).rev() {
        for ly in (0..=(order - lx)).rev() {
            f(lx, ly, order - lx - ly);
        }
    }
}

/// Number of Cartesian components up to and including `max_order`:
/// 1 (order 0), 4 (up to order 1), 10, 20, 35, ...
pub const fn multipole_component_count(max_order: usize) -> usize {
    (max_order + 1) * (max_order + 2) * (max_order + 3) / 6
}

/// Position of the (lx, ly, lz) component in any "all orders up to n" Cartesian storage (a
/// multipole's moments, a derivative table, ...).
pub fn moment_index(lx: usize, ly: usize, lz: usize) -> usize {
    let order = lx + ly + lz;
    let orders_below = multipole_component_count(order) - (order + 1) * (order + 2) / 2;
    // Within the order: the blocks of every larger lx (sizes 1..=order-lx), then lz within this lx's block.
    let nx = order - lx;
    orders_below + nx * (nx + 1) / 2 + lz
}

#[cfg(test)]
mod cartesian_component_tests {
    use super::*;

    // moment_index reproduces for_each_cartesian_component's order, for every order up to 6.
    #[test]
    fn moment_index_matches_for_each_cartesian_component() {
        for max_order in 0..=6 {
            let mut expected_idx = 0;
            for order in 0..=max_order {
                for_each_cartesian_component(order, |lx, ly, lz| {
                    assert_eq!(
                        moment_index(lx, ly, lz), expected_idx,
                        "max_order={max_order} order={order} (lx,ly,lz)=({lx},{ly},{lz})"
                    );
                    expected_idx += 1;
                });
            }
            assert_eq!(expected_idx, multipole_component_count(max_order), "max_order={max_order}");
        }
    }
}
