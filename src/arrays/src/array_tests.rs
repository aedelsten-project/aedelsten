// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The views of the 3D-8D arrays against plain indexing (4D and 6D; every rank uses the same `layout` helpers).

use crate::matrix::MatrixTrait;
use crate::{Array3DTrait, Array4D, Array4DTrait, Array4DTraitMut};
use crate::array6d::{Array6D, Array6DTrait, Array6DTraitMut};

fn filled4(dims: [usize; 4]) -> Array4D<f64> {
    let n = dims.iter().product();
    Array4D::new_from_vec(dims[0], dims[1], dims[2], dims[3], (0..n).map(|x| x as f64).collect::<Vec<_>>().into_boxed_slice())
}

#[test]
fn indexing_is_column_major() {
    let a = filled4([2, 3, 4, 5]);
    assert_eq!(a[(1, 2, 3, 4)], (1 + 2 * (2 + 3 * (3 + 4 * 4))) as f64);
}

#[test]
fn sub_views_fibres_and_iteration_match_indexing() {
    let a = filled4([2, 3, 4, 5]);
    let m = a.view_sub_matrix((3, 4));
    let s = a.view_sub_array3d(4);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(m[(i, j)], a[(i, j, 3, 4)]);
            for k in 0..4 {
                assert_eq!(s[(i, j, k)], a[(i, j, k, 4)]);
            }
        }
    }
    assert_eq!(a.get_fibre((2, 3, 4)), &[a[(0, 2, 3, 4)], a[(1, 2, 3, 4)]]);
    for (l, sub) in a.iter().enumerate() {
        assert_eq!(sub[(1, 2, 3)], a[(1, 2, 3, l)]);
    }
    assert_eq!(a.iter().count(), 5);
}

#[test]
fn grouped_views_merge_adjacent_dimensions() {
    let a = filled4([2, 3, 4, 5]);
    let m = a.view_as_matrix((2, 2));
    assert_eq!((m.rows(), m.cols()), (6, 20));
    assert_eq!(m[(1 + 2 * 2, 3 + 4 * 4)], a[(1, 2, 3, 4)]);
    let b = a.view_as_array3d((1, 2, 1));
    assert_eq!((b.dim0(), b.dim1(), b.dim2()), (2, 12, 5));
    assert_eq!(b[(1, 2 + 3 * 3, 4)], a[(1, 2, 3, 4)]);
}

#[test]
fn mutable_views_write_through() {
    let mut a = Array6D::<f64>::new(2, 2, 2, 2, 2, 2);
    a.view_sub_array4d_mut((1, 0)).fill(1.0);
    a.get_fibre_mut((1, 1, 1, 1, 1))[0] = 7.0;
    a[(1, 0, 0, 0, 0, 1)] = 3.0;
    assert_eq!(a[(0, 1, 1, 1, 1, 0)], 1.0);
    assert_eq!(a[(0, 1, 1, 1, 1, 1)], 7.0);
    assert_eq!(a[(1, 0, 0, 0, 0, 1)], 3.0);
    assert_eq!(a.get_data().iter().filter(|&&x| x != 0.0).count(), 16 + 2);
    assert_eq!(a.view_as_matrix((3, 3)).get_data().len(), 64);
}

#[test]
#[should_panic(expected = "does not cover")]
fn a_grouping_must_cover_every_dimension() {
    filled4([2, 3, 4, 5]).view_as_matrix((1, 2));
}

#[test]
#[should_panic(expected = "reshape")]
fn a_reshape_must_keep_the_number_of_elements() {
    use crate::array4d::Array4DTrait as _;
    filled4([2, 3, 4, 5]).view_reshaped_array3d((2, 3, 4));
}

#[test]
fn a_7d_view_reshapes_into_8d_keeping_the_data() {
    use crate::array7d::Array7DViewMut;
    use crate::array8d::Array8DTrait;
    // The last dimension (6) split into 2 x 3: element (.., j) of the 7D view is (.., j % 2, j / 2) of the 8D one.
    let mut data: Vec<f64> = (0..2 * 6).map(|i| i as f64).collect();
    let pointer = data.as_ptr();
    let reshaped = Array7DViewMut::new(2, 1, 1, 1, 1, 1, 6, &mut data).into_reshaped_array8d((2, 1, 1, 1, 1, 1, 2, 3));
    assert_eq!((reshaped.dims(), reshaped.get_data().as_ptr()), ([2, 1, 1, 1, 1, 1, 2, 3], pointer));
    assert_eq!(reshaped[(1, 0, 0, 0, 0, 0, 1, 2)], (1 + 2 * 5) as f64);
}

#[test]
#[should_panic(expected = "reshape")]
fn a_7d_reshape_into_8d_must_keep_the_number_of_elements() {
    let mut data = vec![0.0; 6];
    crate::array7d::Array7DViewMut::new(1, 1, 1, 1, 1, 1, 6, &mut data).into_reshaped_array8d((1, 1, 1, 1, 1, 1, 2, 2));
}

#[test]
fn whole_views_share_the_data() {
    use crate::array4d::Array4DViewMut;
    let mut a = filled4([2, 3, 4, 5]);
    assert_eq!((a.view().dims(), a.view().get_data().as_ptr()), (a.dims(), a.get_data().as_ptr()));
    let mut view_mut: Array4DViewMut<'_, f64> = a.view_mut();
    // On a view, `view_mut` reborrows: the view stays usable after the reborrow ends.
    view_mut.view_mut()[(1, 2, 3, 4)] = -1.0;
    view_mut[(0, 0, 0, 0)] = -2.0;
    assert_eq!((a[(1, 2, 3, 4)], a[(0, 0, 0, 0)]), (-1.0, -2.0));
}

#[test]
fn distance_is_the_norm_of_the_difference() {
    let (a, b) = ([1.0, -2.0, 4.0], [4.0, 2.0, 4.0]);
    assert_eq!(crate::vector_math::distance(&a, &b), 5.0);
    assert_eq!(crate::vector_math::distance(&a, &a), 0.0);
}

#[test]
#[should_panic(expected = "distance: lengths")]
fn distance_rejects_different_lengths() {
    crate::vector_math::distance(&[1.0, 2.0], &[1.0]);
}
