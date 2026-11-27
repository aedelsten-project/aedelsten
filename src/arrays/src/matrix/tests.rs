// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use super::*;

#[cfg(test)]
mod whole_view_tests {
    use super::*;

    #[test]
    fn whole_views_share_the_data() {
        let mut m = Matrix::<f64>::new_from_vec(2, 3, (0..6).map(f64::from).collect::<Vec<_>>().into_boxed_slice());
        let view = m.view();
        assert_eq!((view.rows(), view.cols(), view.get_data().as_ptr()), (2, 3, m.get_data().as_ptr()));
        let mut view_mut = m.view_mut();
        // On a view, `view_mut` reborrows: the view stays usable after the reborrow ends.
        view_mut.view_mut()[(1, 2)] = -1.0;
        view_mut[(0, 0)] = -2.0;
        assert_eq!((m[(1, 2)], m[(0, 0)], m.view()[(1, 1)]), (-1.0, -2.0, 3.0));
    }
}

#[cfg(test)]
mod congruence_transform_tests {
    use super::*;

    // congruence_transform (other^T self other) with a rectangular `other`.
    #[test]
    fn congruence_transform_handles_rectangular_other() {
        let m = Matrix::<f64>::new_from_vec(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ].into_boxed_slice());
        let c = Matrix::<f64>::new_from_vec(3, 2, vec![
            1.0, 0.0, 2.0,
            0.5, 1.0, -1.0,
        ].into_boxed_slice());

        let via_matrix_transform = m.congruence_transform(&c);
        let manual = c.view_transposed().matmul(&m).matmul(&c);

        assert_eq!(via_matrix_transform.rows(), c.cols());
        assert_eq!(via_matrix_transform.cols(), c.cols());
        for i in 0..manual.rows() {
            for j in 0..manual.cols() {
                assert!(
                    (via_matrix_transform[(i, j)] - manual[(i, j)]).abs() < 1e-10,
                    "mismatch at ({i},{j}): congruence_transform={} manual={}",
                    via_matrix_transform[(i, j)], manual[(i, j)]
                );
            }
        }
    }

    #[test]
    fn congruence_transform_prealloc_matches_allocating_version() {
        let m = Matrix::<f64>::new_from_vec(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ].into_boxed_slice());
        let c = Matrix::<f64>::new_from_vec(3, 2, vec![
            1.0, 0.0, 2.0,
            0.5, 1.0, -1.0,
        ].into_boxed_slice());

        let expected = m.congruence_transform(&c);

        let mut product_scratch = Matrix::<f64>::new(m.rows(), c.cols());
        let mut output = Matrix::<f64>::new(c.cols(), c.cols());
        m.congruence_transform_prealloc(&c, &mut product_scratch, &mut output);

        for i in 0..expected.rows() {
            for j in 0..expected.cols() {
                assert!(
                    (output[(i, j)] - expected[(i, j)]).abs() < 1e-10,
                    "mismatch at ({i},{j}): prealloc={} allocating={}",
                    output[(i, j)], expected[(i, j)]
                );
            }
        }
    }

    // congruence_transform with a square `other`, against C^T M C formed by hand.
    #[test]
    fn congruence_transform_still_correct_for_square_other() {
        let m = Matrix::<f64>::new_from_vec(2, 2, vec![2.0, 1.0, 1.0, 3.0].into_boxed_slice());
        let u = Matrix::<f64>::new_from_vec(2, 2, vec![0.6, -0.8, 0.8, 0.6].into_boxed_slice());

        let via_matrix_transform = m.congruence_transform(&u);
        let manual = u.view_transposed().matmul(&m).matmul(&u);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (via_matrix_transform[(i, j)] - manual[(i, j)]).abs() < 1e-10,
                    "mismatch at ({i},{j}): congruence_transform={} manual={}",
                    via_matrix_transform[(i, j)], manual[(i, j)]
                );
            }
        }
    }
}

#[cfg(test)]
mod symmetric_pseudo_inverse_tests {
    use super::*;

    // A well-conditioned symmetric matrix: symmetric_pseudo_inverse with a
    // generous relative threshold (nothing gets dropped) should agree with
    // the ordinary matrix inverse.
    #[test]
    fn matches_ordinary_inverse_when_well_conditioned() {
        let m = Matrix::<f64>::new_from_vec(3, 3, vec![
            4.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ].into_boxed_slice());

        let via_pseudo_inverse = m.symmetric_pseudo_inverse(1e-10);
        let via_invert = m.invert();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (via_pseudo_inverse[(i, j)] - via_invert[(i, j)]).abs() < 1e-10,
                    "mismatch at ({i},{j}): pseudo_inverse={} invert={}",
                    via_pseudo_inverse[(i, j)], via_invert[(i, j)]
                );
            }
        }
    }

    // symmetric_pseudo_inverse is M M^T by construction: exactly symmetric for a severely ill-conditioned input.
    #[test]
    fn result_is_always_exactly_symmetric_even_when_ill_conditioned() {
        // A symmetric matrix engineered to be severely ill-conditioned:
        // two nearly-parallel rows/columns.
        let m = Matrix::<f64>::new_from_vec(3, 3, vec![
            1.0, 1.0, 0.0,
            1.0, 1.0 + 1e-10, 0.0,
            0.0, 0.0, 2.0,
        ].into_boxed_slice());

        let inv = m.symmetric_pseudo_inverse(1e-6);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(inv[(i, j)], inv[(j, i)], "asymmetric at ({i},{j})");
            }
        }
    }

    // A near-zero eigenvalue (linear dependence) must be dropped rather
    // than inverted into a huge, noise-dominated value.
    #[test]
    fn near_zero_eigenvalue_is_truncated_not_inverted() {
        let m = Matrix::<f64>::new_from_vec(2, 2, vec![
            1.0, 0.0,
            0.0, 1e-12,
        ].into_boxed_slice());

        let inv = m.symmetric_pseudo_inverse(1e-6);

        assert!((inv[(0, 0)] - 1.0).abs() < 1e-10, "the well-conditioned eigenvalue should invert normally");
        assert_eq!(inv[(1, 1)], 0.0, "the near-zero eigenvalue should be dropped, not inverted to ~1e12");
        assert_eq!(inv[(0, 1)], 0.0);
        assert_eq!(inv[(1, 0)], 0.0);
    }
}

#[cfg(test)]
mod symmetrize_tests {
    use super::*;

    // symmetrize on a general 4 x 4 matrix (not symmetric, every entry distinct), every column against X + X^T.
    #[test]
    fn symmetrize_in_place_matches_symmetrize_matrix() {
        let data: Vec<f64> = (0..16).map(|i| ((i * 5) % 7) as f64 - 3.0).collect();
        let m = Matrix::<f64>::new_from_vec(4, 4, data.into_boxed_slice());
        let expected = m.symmetrize(0.5);
        let mut got = m.clone();
        got.symmetrize_in_place(0.5);
        assert_eq!(got.get_data(), expected.get_data());
    }

    // The tiled symmetrize_in_place against an untiled loop, bitwise: sizes below, at and around the tile edge (64)
    // and several tiles with a partial last one; alpha 1 (X + X^T) and 0.5. A view of a larger buffer's columns too.
    #[test]
    fn tiled_symmetrize_in_place_is_bitwise_the_untiled_loop() {
        let untiled = |m: &Matrix<f64>, alpha: f64| {
            let mut out = m.clone();
            for j in 0..m.cols() {
                for i in j..m.rows() {
                    out[(i, j)] = alpha * (out[(i, j)] + out[(j, i)]);
                }
            }
            for j in 0..m.cols() {
                for i in 0..j {
                    out[(i, j)] = out[(j, i)];
                }
            }
            out
        };
        for n in [1, 2, 63, 64, 65, 130, 200] {
            let data: Vec<f64> = (0..n * n).map(|k| ((k * 37 % 101) as f64 - 50.0) / 7.0 + 1e-3 * k as f64).collect();
            let m = Matrix::<f64>::new_from_vec(n, n, data.into_boxed_slice());
            for alpha in [1.0, 0.5] {
                let expected = untiled(&m, alpha);
                let mut got = m.clone();
                got.symmetrize_in_place(alpha);
                assert!(got.get_data().iter().zip(expected.get_data()).all(|(a, b)| a.to_bits() == b.to_bits()), "n {n}, alpha {alpha}");
            }
        }
        // In parts, one after another as the processes sharing a matrix would at once: bitwise the whole.
        for (n, parts) in [(1, 3), (65, 2), (200, 3), (200, 7)] {
            let m = Matrix::<f64>::new_from_vec(n, n, (0..n * n).map(|k| ((k * 37 % 101) as f64 - 50.0) / 7.0).collect());
            let mut got = m.clone();
            for part in 0..parts {
                got.symmetrize_in_place_part(1.0, part, parts);
            }
            assert!(got.get_data().iter().zip(untiled(&m, 1.0).get_data()).all(|(a, b)| a.to_bits() == b.to_bits()), "n {n}, {parts} parts");
        }
        let mut wide = Matrix::<f64>::new_from_vec(70, 210, (0..70 * 210).map(|k| (k % 13) as f64 - 0.3 * (k % 7) as f64).collect());
        let block = Matrix::new_from_vec(70, 70, wide.view_columns(70..140).get_data().into());
        wide.view_columns_mut(70..140).symmetrize_in_place(1.0);
        assert_eq!(wide.view_columns(70..140).get_data(), untiled(&block, 1.0).get_data());
    }

    #[test]
    #[should_panic(expected = "not square")]
    fn symmetrize_in_place_refuses_a_non_square_matrix() {
        Matrix::<f64>::new(3, 4).symmetrize_in_place(1.0);
    }

    #[test]
    fn matches_hand_computed_average_with_transpose() {
        let m = Matrix::<f64>::new_from_vec(4, 4, vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ].into_boxed_slice());

        let symmetrized = m.symmetrize(0.5);

        for i in 0..4 {
            for j in 0..4 {
                let expected = 0.5 * (m[(i, j)] + m[(j, i)]);
                assert_eq!(
                    symmetrized[(i, j)], expected,
                    "mismatch at ({i},{j}): got {}, expected 0.5*(m[({i},{j})] + m[({j},{i})]) = {expected}",
                    symmetrized[(i, j)]
                );
            }
        }
    }

    #[test]
    fn result_is_exactly_symmetric() {
        let m = Matrix::<f64>::new_from_vec(4, 4, vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ].into_boxed_slice());

        let symmetrized = m.symmetrize(0.5);

        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(symmetrized[(i, j)], symmetrized[(j, i)], "asymmetric at ({i},{j})");
            }
        }
    }
}

#[cfg(test)]
mod invert_prealloc_tests {
    use super::*;

    #[test]
    fn matches_invert_and_reports_singular_matrices() {
        let a = Matrix::<f64>::new_from_vec(3, 3, vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0].into_boxed_slice());
        let reference = a.invert();
        let mut inverse = Matrix::<f64>::new(3, 3);
        let mut work = vec![0.0; 9];
        let mut pivot = vec![0; 3];
        assert!(a.invert_prealloc(&mut inverse, &mut work, &mut pivot));
        for (x, y) in inverse.get_data().iter().zip(reference.get_data()) {
            assert!((x - y).abs() < 1e-14);
        }
        // Reusing the same scratch for another matrix, including a view as the output.
        let mut storage = vec![0.0; 9];
        let mut view = MatrixViewMut::new(3, 3, &mut storage);
        assert!(a.invert_prealloc(&mut view, &mut work, &mut pivot));
        assert!((view[(0, 0)] - reference[(0, 0)]).abs() < 1e-14);

        let singular = Matrix::<f64>::new_from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0].into_boxed_slice());
        let mut output = Matrix::<f64>::new(2, 2);
        assert!(!singular.invert_prealloc(&mut output, &mut work, &mut pivot));
    }
}

#[cfg(test)]
mod eigenvectors_prealloc_tests {
    use super::*;

    #[test]
    fn matches_eigenvectors_into_views_and_reports_a_short_scratch() {
        let n = 5;
        let mut a = Matrix::<f64>::new(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = ((i * 7 + j * 3) % 11) as f64 - 5.0 + if i == j { 2.0 * i as f64 } else { 0.0 };
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
        }
        let (values, vectors) = a.eigenvectors();
        let mut work = vec![0.0; eigenvectors_work_len(n)];
        assert!(work.len() >= 3 * n - 1);
        // Into views over larger buffers, as a caller sizing everything once for the largest n would.
        let (mut value_storage, mut vector_storage) = (vec![0.0; 2 * n], vec![0.0; n * n + 7]);
        let mut view = MatrixViewMut::new(n, n, &mut vector_storage[..n * n]);
        assert!(a.eigenvectors_prealloc(&mut value_storage, &mut view, &mut work));
        for k in 0..n {
            assert!((value_storage[k] - values[k]).abs() < 1e-12);
            // Eigenvectors agree up to sign.
            let sign = if view[(0, k)] * vectors[(0, k)] < 0.0 { -1.0 } else { 1.0 };
            for i in 0..n {
                assert!((sign * view[(i, k)] - vectors[(i, k)]).abs() < 1e-10);
            }
        }
        // Below LAPACK's minimum the call fails instead of writing past the scratch.
        let mut short = vec![0.0; 3 * n - 2];
        let mut out = Matrix::<f64>::new(n, n);
        assert!(!a.eigenvectors_prealloc(&mut value_storage, &mut out, &mut short));
    }
}

#[cfg(test)]
mod matmul_scaled_tests {
    use super::*;

    // C = alpha A B + beta C: beta = 0 must overwrite (whatever C held), beta = 1 accumulate.
    #[test]
    fn matmul_scaled_prealloc_scales_the_product_and_the_existing_output() {
        let a = Matrix::<f64>::new_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice());
        let b = Matrix::<f64>::new_from_vec(3, 2, vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.5].into_boxed_slice());
        let product = a.matmul(&b);

        let mut c = Matrix::<f64>::new_from_vec(2, 2, vec![9.0, 9.0, 9.0, 9.0].into_boxed_slice());
        a.matmul_scaled_prealloc(&b, &mut c, -2.0, 0.0);
        for j in 0..2 {
            for i in 0..2 {
                assert!((c[(i, j)] + 2.0 * product[(i, j)]).abs() < 1e-14);
            }
        }
        a.matmul_scaled_prealloc(&b, &mut c, 3.0, 0.5);
        for j in 0..2 {
            for i in 0..2 {
                assert!((c[(i, j)] - (3.0 * product[(i, j)] - product[(i, j)])).abs() < 1e-14);
            }
        }
    }
}

#[cfg(test)]
mod sub_view_and_factor_tests {
    use super::*;

    fn filled(rows: usize, cols: usize, seed: usize) -> Matrix<f64> {
        Matrix::<f64>::new_from_vec(rows, cols, (0..rows * cols).map(|i| ((i * 7 + seed) % 13) as f64 - 6.0).collect::<Vec<_>>().into_boxed_slice())
    }

    /// `m[rows, cols]` copied into a matrix of its own.
    fn copied(m: &Matrix<f64>, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Matrix<f64> {
        let mut out = Matrix::<f64>::new(rows.len(), cols.len());
        for (j, col) in cols.enumerate() {
            for (i, row) in rows.clone().enumerate() {
                out[(i, j)] = m[(row, col)];
            }
        }
        out
    }

    fn assert_close(got: &Matrix<f64>, expected: &Matrix<f64>) {
        assert_eq!((got.rows(), got.cols()), (expected.rows(), expected.cols()));
        for (x, y) in got.get_data().iter().zip(expected.get_data()) {
            assert!((x - y).abs() < 1e-13, "{x} vs {y}");
        }
    }

    #[test]
    fn view_columns_reads_and_writes_the_columns_in_place() {
        let mut m = filled(4, 6, 0);
        let view = m.view_columns(2..5);
        assert_eq!((view.rows(), view.cols()), (4, 3));
        for j in 0..3 {
            assert_eq!(view.get_col(j), m.get_col(2 + j));
        }
        assert_eq!(m.view_columns(3..3).cols(), 0);
        m.view_columns_mut(1..2).fill(9.0);
        assert!(m.get_col(1).iter().all(|&x| x == 9.0));
        assert!(m.get_col(0).iter().chain(m.get_col(2)).all(|&x| x != 9.0));
    }

    #[test]
    fn get_two_cols_mut_gives_both_columns_in_the_order_asked() {
        let reference = filled(4, 6, 0);
        for (l0, l1) in [(1, 4), (4, 1), (0, 5)] {
            let mut m = reference.clone();
            let (first, second) = m.get_two_cols_mut(l0, l1);
            assert_eq!((&*first, &*second), (reference.get_col(l0), reference.get_col(l1)));
            first.fill(-1.0);
            second.fill(-2.0);
            assert!(m.get_col(l0).iter().all(|&x| x == -1.0) && m.get_col(l1).iter().all(|&x| x == -2.0));
        }
    }

    #[test]
    #[should_panic(expected = "get_two_cols_mut")]
    fn get_two_cols_mut_of_one_column_panics() {
        filled(4, 6, 0).get_two_cols_mut(2, 2);
    }

    #[test]
    #[should_panic(expected = "view_columns")]
    fn view_columns_out_of_range_panics() {
        filled(4, 6, 0).view_columns(4..7);
    }

    #[test]
    #[should_panic(expected = "view_rows")]
    fn view_rows_out_of_range_panics() {
        filled(4, 6, 0).view_rows(2..5);
    }

    #[test]
    fn view_rows_reads_like_a_copied_block() {
        let m = filled(7, 3, 1);
        let block = m.view_rows(2..5);
        let expected = copied(&m, 2..5, 0..3);
        assert_eq!(block.shape(), (3, 3));
        for j in 0..3 {
            assert_eq!(block.column(j), expected.get_col(j));
        }
        assert_eq!(m.view_rows(4..4).shape(), (0, 3));
    }

    #[test]
    fn view_rows_of_a_strided_block_reads_like_the_rows_of_the_matrix() {
        let m = filled(9, 4, 1);
        let block = m.view_rows(2..8).view_rows(1..4);
        let expected = copied(&m, 3..6, 0..4);
        assert_eq!(block.shape(), (3, 4));
        for j in 0..4 {
            assert_eq!(block.column(j), expected.get_col(j));
        }
        assert_eq!(m.view_rows(2..8).view_rows(6..6).shape(), (0, 4));
    }

    // Every combination of plain and strided factors, against the product of copied blocks.
    #[test]
    fn matmul_matches_the_dense_product_for_every_combination_of_factors() {
        let (a, b) = (filled(9, 4, 2), filled(6, 5, 3));
        let (a_rows, b_rows) = (3..8, 1..5);
        let (a_dense, b_dense) = (copied(&a, a_rows.clone(), 0..4), copied(&b, b_rows.clone(), 0..5));
        let expected = a_dense.matmul(&b_dense);
        let (a_strided, b_strided) = (a.view_rows(a_rows), b.view_rows(b_rows));

        let mut got = Matrix::<f64>::new(5, 5);
        a_dense.matmul_scaled_prealloc(&b_dense, &mut got, 1.0, 0.0);
        assert_close(&got, &expected);
        a_strided.matmul_scaled_prealloc(&b_dense, &mut got, 1.0, 0.0);
        assert_close(&got, &expected);
        a_dense.matmul_scaled_prealloc(&b_strided, &mut got, 1.0, 0.0);
        assert_close(&got, &expected);
        a_strided.matmul_scaled_prealloc(&b_strided, &mut got, 1.0, 0.0);
        assert_close(&got, &expected);

        // alpha and beta: got = 2 a b + 0.5 (a b) = 2.5 a b.
        a_strided.matmul_scaled_prealloc(&b_strided, &mut got, 2.0, 0.5);
        let mut scaled = expected.clone();
        scaled.get_data_mut().iter_mut().for_each(|x| *x *= 2.5);
        assert_close(&got, &scaled);
    }

    // A product into a block of rows of a larger matrix (`view_rows_mut`): the block takes the product, the rows
    // around it are untouched, and beta accumulates.
    #[test]
    fn matmul_into_a_strided_block_writes_only_its_rows() {
        let (a, b) = (filled(5, 4, 2), filled(4, 3, 3));
        let expected = a.matmul(&b);
        let mut m = filled(9, 3, 1);
        let original = m.clone();
        a.matmul_scaled_prealloc(&b, &mut m.view_rows_mut(2..7), 1.0, 0.0);
        assert_close(&copied(&m, 2..7, 0..3), &expected);
        for j in 0..3 {
            for i in (0..2).chain(7..9) {
                assert_eq!(m[(i, j)], original[(i, j)]);
            }
        }
        a.view_rows(0..5).matmul_scaled_prealloc(&b.view_rows(0..4), &mut m.view_rows_mut(2..7), 1.0, 1.0);
        let mut doubled = expected.clone();
        doubled.get_data_mut().iter_mut().for_each(|x| *x *= 2.0);
        assert_close(&copied(&m, 2..7, 0..3), &doubled);
        // The last rows of the last column: the block ends the buffer.
        let mut tail = Matrix::<f64>::new(9, 3);
        a.matmul_prealloc(&b, &mut tail.view_rows_mut(4..9));
        assert_close(&copied(&tail, 4..9, 0..3), &expected);
        assert_eq!(tail.view_rows_mut(9..9).shape(), (0, 3));
    }

    #[test]
    #[should_panic(expected = "view_rows_mut")]
    fn view_rows_mut_out_of_range_panics() {
        filled(4, 6, 0).view_rows_mut(2..5);
    }

    // Transposed views on either side, of plain and strided operands, against copied transposes.
    #[test]
    fn transposed_views_match_the_dense_product_on_either_side() {
        let (a, b) = (filled(9, 4, 2), filled(6, 5, 3));
        let (a_rows, b_rows) = (2..7, 1..6);
        let (a_dense, b_dense) = (copied(&a, a_rows.clone(), 0..4), copied(&b, b_rows.clone(), 0..5));
        let (a_strided, b_strided) = (a.view_rows(a_rows), b.view_rows(b_rows));

        // a is 5 x 4, b is 5 x 5.
        let expected = a_dense.view_transposed().matmul(&b_dense);
        assert_close(&a_dense.view_transposed().matmul(&b_dense), &expected);
        assert_close(&a_strided.view_transposed().matmul(&b_strided), &expected);

        let expected = a_dense.matmul(&a_dense.view_transposed());
        assert_close(&a_dense.matmul(&a_dense.view_transposed()), &expected);
        assert_close(&a_strided.matmul(&a_strided.view_transposed()), &expected);

        let expected = a_dense.view_transposed().matmul(&b_dense.view_transposed());
        assert_close(&a_dense.view_transposed().matmul(&b_dense.view_transposed()), &expected);
        assert_close(&a_strided.view_transposed().matmul(&b_strided.view_transposed()), &expected);
    }

    #[test]
    fn matvec_matches_the_dense_product_for_every_operand_kind() {
        let a = filled(7, 4, 5);
        let block = copied(&a, 1..6, 0..4);
        let (x4, x5): (Vec<f64>, Vec<f64>) = ((0..4).map(|i| i as f64 - 1.5).collect(), (0..5).map(|i| 0.5 * i as f64).collect());
        let dense = |m: &Matrix<f64>, x: &[f64]| -> Vec<f64> {
            (0..m.rows()).map(|i| (0..m.cols()).map(|j| m[(i, j)] * x[j]).sum()).collect()
        };
        let close = |got: &[f64], expected: &[f64]| {
            assert_eq!(got.len(), expected.len());
            assert!(got.iter().zip(expected).all(|(g, e)| (g - e).abs() < 1e-12), "{got:?} vs {expected:?}");
        };
        close(&block.matvec(&x4), &dense(&block, &x4));
        close(&a.view_rows(1..6).matvec(&x4), &dense(&block, &x4));
        close(&block.view_transposed().matvec(&x5), &dense(&block.transpose(), &x5));
        close(&a.view_rows(1..6).view_transposed().matvec(&x5), &dense(&block.transpose(), &x5));

        let mut y = vec![1.0; 5];
        block.matvec_scaled_prealloc(&x4, &mut y, 2.0, 0.5);
        let expected: Vec<f64> = dense(&block, &x4).iter().map(|v| 2.0 * v + 0.5).collect();
        close(&y, &expected);
    }

    #[test]
    fn the_arena_forms_match_the_heap_forms() {
        use crate::scratch::{Scratch, ByteUnit};
        let (a, b) = (filled(4, 3, 1), filled(3, 5, 2));
        let x = [1.0, -2.0, 0.5];
        let mut scratch = Scratch::new(1, ByteUnit::MiB).unwrap();
        let arena = scratch.get_arena();
        assert_eq!(a.matmul_arena(&b, &arena).unwrap().get_data(), a.matmul(&b).get_data());
        assert_eq!(&*a.matvec_arena(&x, &arena).unwrap(), &*a.matvec(&x));
    }

    #[test]
    fn matmul_with_an_empty_inner_dimension_scales_the_output() {
        let (a, b) = (filled(3, 0, 0), filled(0, 2, 0));
        let mut got = filled(3, 2, 4);
        let mut expected = got.clone();
        expected.get_data_mut().iter_mut().for_each(|x| *x *= 0.5);
        a.matmul_scaled_prealloc(&b, &mut got, 1.0, 0.5);
        assert_close(&got, &expected);
    }
}

// Small products as plain loops against dgemm: their agreement, and their timings (`crossover`, ignored;
// `cargo test -p arrays --release small_product_benchmark -- --ignored --nocapture`).
#[cfg(test)]
mod small_product_benchmark {
    use super::*;
    use std::time::Instant;

    /// `c = alpha op(a) b + beta c` as plain loops, `op(a)` `m x k` (`a` itself `k x m` when `trans_a`), `b` `k x n`,
    /// `c` `m x n` contiguous; `a` and `b` column-major with leading dimensions `lda`, `ldb`. `beta == 0` does not read
    /// `c`, as in BLAS.
    #[allow(clippy::too_many_arguments)]
    fn small_matrix_matrix_multiplication(
        a: &[f64],
        lda: usize,
        trans_a: CblasTranspose,
        b: &[f64],
        ldb: usize,
        (m, n, k): (usize, usize, usize),
        c: &mut [f64],
        alpha: f64,
        beta: f64,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        for (j, c_j) in c.chunks_exact_mut(m).take(n).enumerate() {
            if beta == 0.0 {
                c_j.fill(0.0);
            } else if beta != 1.0 {
                c_j.iter_mut().for_each(|x| *x *= beta);
            }
            if k == 0 {
                continue;
            }
            let b_j = &b[j * ldb..j * ldb + k];
            if matches!(trans_a, CblasNoTrans) {
                // c_j += alpha sum_p a_p b_pj: column updates, contiguous.
                for (p, &b_pj) in b_j.iter().enumerate() {
                    let w = alpha * b_pj;
                    for (c_ij, a_ip) in c_j.iter_mut().zip(&a[p * lda..p * lda + m]) {
                        *c_ij += w * a_ip;
                    }
                }
            } else {
                // c_ij += alpha a_i . b_j: column dot products, contiguous.
                for (i, c_ij) in c_j.iter_mut().enumerate() {
                    *c_ij += alpha * a[i * lda..i * lda + k].iter().zip(b_j).map(|(x, y)| x * y).sum::<f64>();
                }
            }
        }
    }

    fn filled(len: usize, seed: usize) -> Vec<f64> {
        (0..len).map(|i| ((i * 7 + seed) % 13) as f64 * 0.25 - 1.5).collect()
    }

    // The loops against dgemm, both transposes, strided inputs, alpha/beta, and beta = 0 ignoring NaN in c.
    #[test]
    fn small_products_match_dgemm() {
        for &(m, n, k) in &[(1, 1, 1), (3, 2, 5), (7, 4, 3), (5, 6, 0), (16, 3, 9)] {
            for trans in [CblasNoTrans, CblasTrans] {
                let (lda, ldb) = (if matches!(trans, CblasNoTrans) { m + 2 } else { k + 1 }, k + 3);
                let a_cols = if matches!(trans, CblasNoTrans) { k } else { m };
                let (a, b) = (filled(lda * a_cols.max(1), 1), filled(ldb * n, 2));
                for (alpha, beta) in [(1.0, 0.0), (0.5, 1.0), (-2.0, 0.25)] {
                    let start = if beta == 0.0 { vec![f64::NAN; m * n] } else { filled(m * n, 3) };
                    let mut loops = start.clone();
                    small_matrix_matrix_multiplication(&a, lda, trans, &b, ldb, (m, n, k), &mut loops, alpha, beta);
                    let blas = if beta == 0.0 { vec![0.0; m * n] } else { start.clone() };
                    dgemm(CblasColMajor, trans, CblasNoTrans, m as i32, n as i32, k as i32, alpha, a.as_ptr(), lda.max(1) as i32,
                          b.as_ptr(), ldb.max(1) as i32, beta, blas.as_ptr(), m as i32);
                    for (x, y) in loops.iter().zip(&blas) {
                        assert!((x - y).abs() < 1e-12, "{m} x {n} x {k}: {x} vs {y}");
                    }
                }
            }
        }
    }

    /// Nanoseconds per call of `work`, the best of a few runs.
    fn time(mut work: impl FnMut()) -> f64 {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let reps = 2000;
            let start = Instant::now();
            for _ in 0..reps {
                work();
            }
            best = best.min(start.elapsed().as_nanos() as f64 / reps as f64);
        }
        best
    }

    // Loops against dgemm over shapes of 1 to 136 rows, columns and inner dimension: prints m, n, k, their product,
    // the two times and their ratio.
    #[test]
    #[ignore]
    fn crossover() {
        let sizes = [1, 3, 5, 6, 10, 15, 25, 36, 60, 136];
        println!("{:>4} {:>4} {:>4} {:>9} {:>10} {:>10} {:>6}", "m", "n", "k", "mnk", "loops ns", "dgemm ns", "ratio");
        for &m in &sizes {
            for &n in &sizes {
                for &k in &sizes {
                    let mnk = m * n * k;
                    if mnk > 200_000 {
                        continue;
                    }
                    for trans in [CblasNoTrans, CblasTrans] {
                        let (a, b) = (vec![0.5; m * k], vec![0.25; k * n]);
                        let lda = if matches!(trans, CblasNoTrans) { m } else { k };
                        let mut c = vec![0.0; m * n];
                        let loops = time(|| small_matrix_matrix_multiplication(std::hint::black_box(&a), lda, trans, &b, k, (m, n, k), &mut c, 1.0, 1.0));
                        let blas = time(|| dgemm(CblasColMajor, trans, CblasNoTrans, m as i32, n as i32, k as i32, 1.0,
                                                 std::hint::black_box(&a).as_ptr(), lda as i32, b.as_ptr(), k as i32, 1.0, c.as_ptr(), m as i32));
                        let label = if matches!(trans, CblasNoTrans) { "N" } else { "T" };
                        println!("{label} {m:>4} {n:>4} {k:>4} {mnk:>9} {loops:>10.0} {blas:>10.0} {:>6.2}", loops / blas);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod add_tests {
    use super::*;
    use crate::scratch::{Scratch, ByteUnit};

    #[test]
    fn the_three_forms_of_add_agree() {
        let a = Matrix::<f64>::new_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice());
        let b = Matrix::<f64>::new_from_vec(2, 3, vec![0.5, -1.0, 2.0, 0.0, -3.0, 1.5].into_boxed_slice());
        let expected = [1.5, 1.0, 5.0, 4.0, 2.0, 7.5];
        assert_eq!(a.add(&b).get_data(), &expected);
        let mut out = Matrix::<f64>::new_from_vec(2, 3, vec![f64::NAN; 6].into_boxed_slice());
        a.add_prealloc(&b, &mut out);
        assert_eq!(out.get_data(), &expected);
        let mut scratch = Scratch::new(1, ByteUnit::MiB).unwrap();
        let arena = scratch.get_arena();
        assert_eq!(a.add_arena(&b, &arena).unwrap().get_data(), &expected);
    }

    #[test]
    #[should_panic(expected = "differ in shape")]
    fn add_checks_the_shapes() {
        let (a, b) = (Matrix::<f64>::new(2, 3), Matrix::<f64>::new(3, 2));
        let _ = a.add(&b);
    }
}

// Each operation's three forms agree (the heap and arena forms both wrap `_prealloc`), and the results are what
// they claim to be.
#[cfg(test)]
mod three_forms_tests {
    use super::*;
    use crate::scratch::{Scratch, ByteUnit};

    // A symmetric positive definite 4 x 4 matrix (`shift` on the diagonal of a Hilbert-like matrix).
    fn spd(shift: f64) -> Matrix<f64> {
        let mut m = Matrix::<f64>::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] = 1.0 / (1.0 + (i + j) as f64) + if i == j { shift } else { 0.0 };
            }
        }
        m
    }

    fn close<A: MatrixTrait<f64>, B: MatrixTrait<f64>>(a: &A, b: &B, tol: f64) {
        assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
        for (x, y) in a.get_data().iter().zip(b.get_data()) {
            assert!((x - y).abs() < tol, "{x} vs {y}");
        }
    }

    fn identity(n: usize) -> Matrix<f64> {
        let mut m = Matrix::<f64>::new(n, n);
        (0..n).for_each(|i| m[(i, i)] = 1.0);
        m
    }

    #[test]
    fn transforms_symmetrize_rank_k_update_and_invert() {
        let s = spd(1.0);
        let c = Matrix::<f64>::new_from_vec(4, 3, (0..12).map(|i| (i % 5) as f64 - 2.0).collect::<Vec<_>>().into_boxed_slice());
        let mut scratch = Scratch::new(1, ByteUnit::MiB).unwrap();
        let mut arena = scratch.get_arena();

        close(&s.congruence_transform(&c), &c.view_transposed().matmul(&s.matmul(&c)), 1e-12);
        close(&s.congruence_transform_arena(&c, &mut arena).unwrap(), &s.congruence_transform(&c), 1e-12);
        close(&s.symmetrize_arena(0.5, &arena).unwrap(), &s.symmetrize(0.5), 1e-14);
        close(&c.rank_k_update_arena(2.0, &arena).unwrap(), &c.rank_k_update(2.0), 1e-12);

        let inverse = s.invert();
        close(&s.matmul(&inverse), &identity(4), 1e-10);
        close(&s.invert_arena(&mut arena).unwrap(), &inverse, 1e-12);
    }

    #[test]
    fn eigen_based_forms_agree_and_reconstruct() {
        let s = spd(0.5);
        let mut scratch = Scratch::new(1, ByteUnit::MiB).unwrap();
        let mut arena = scratch.get_arena();

        let (values, vectors) = s.eigenvectors();
        // Eigenvalues alone ('N') and with vectors ('V') are different LAPACK paths: equal to rounding.
        let only = s.eigenvalues();
        assert!(only.iter().zip(values.iter()).all(|(a, b)| (a - b).abs() < 1e-13));
        assert_eq!(&*s.eigenvalues_arena(&mut arena).unwrap(), &*only);
        let (arena_values, arena_vectors) = s.eigenvectors_arena(&mut arena).unwrap();
        assert_eq!(&*arena_values, &*values);
        close(&arena_vectors, &vectors, 1e-14);

        let inverse_sqrt = s.symmetric_inverse_sqrt();
        close(&inverse_sqrt.matmul(&s.matmul(&inverse_sqrt)), &identity(4), 1e-10);
        close(&s.symmetric_inverse_sqrt_arena(&mut arena).unwrap(), &inverse_sqrt, 1e-12);

        let (sqrt, inverse_sqrt_2) = s.symmetric_sqrt_and_inverse_sqrt();
        close(&sqrt.matmul(&sqrt), &s, 1e-10);
        close(&inverse_sqrt_2, &inverse_sqrt, 1e-12);
        let (arena_sqrt, arena_inverse_sqrt) = s.symmetric_sqrt_and_inverse_sqrt_arena(&mut arena).unwrap();
        close(&arena_sqrt, &sqrt, 1e-12);
        close(&arena_inverse_sqrt, &inverse_sqrt, 1e-12);

        // Well conditioned: the pseudo-inverse is the inverse.
        close(&s.symmetric_pseudo_inverse(1e-12), &s.invert(), 1e-9);
        close(&s.symmetric_pseudo_inverse_arena(1e-12, &mut arena).unwrap(), &s.symmetric_pseudo_inverse(1e-12), 1e-12);
    }

    #[test]
    fn sygv_solves_the_generalized_problem() {
        let (a, b) = (spd(0.2), spd(1.0));
        let (values, vectors) = a.sygv(&b, 1);
        // A v = lambda B v, column by column.
        let (av, bv) = (a.matmul(&vectors), b.matmul(&vectors));
        for (k, lambda) in values.iter().enumerate() {
            for i in 0..4 {
                assert!((av[(i, k)] - lambda * bv[(i, k)]).abs() < 1e-10);
            }
        }
        let mut scratch = Scratch::new(1, ByteUnit::MiB).unwrap();
        let mut arena = scratch.get_arena();
        let (arena_values, arena_vectors) = a.sygv_arena(&b, 1, &mut arena).unwrap();
        assert_eq!(&*arena_values, &*values);
        close(&arena_vectors, &vectors, 1e-14);
    }

    #[test]
    fn invert_prealloc_reports_a_singular_matrix() {
        let singular = Matrix::<f64>::new(3, 3);
        let (mut out, mut work, mut pivot) = (Matrix::<f64>::new(3, 3), vec![0.0; 9], vec![0; 3]);
        assert!(!singular.invert_prealloc(&mut out, &mut work, &mut pivot));
    }
}

#[cfg(test)]
mod strided_vector_tests {
    use super::*;
    use crate::vector::{Vector, VectorMut};

    fn filled(rows: usize, cols: usize, seed: usize) -> Matrix<f64> {
        Matrix::<f64>::new_from_vec(rows, cols, (0..rows * cols).map(|i| ((i * 7 + seed) % 13) as f64 - 6.0).collect::<Vec<_>>().into_boxed_slice())
    }

    #[test]
    fn rows_and_diagonals_read_the_right_entries() {
        for (rows, cols) in [(4, 6), (6, 4), (5, 5), (1, 3), (3, 1)] {
            let a = filled(rows, cols, 2);
            for i in 0..rows {
                let row = a.view_row(i);
                assert_eq!(row.len(), cols);
                assert_eq!(row.to_vec(), (0..cols).map(|j| a[(i, j)]).collect::<Vec<_>>());
                assert!((0..cols).all(|j| row[j] == a[(i, j)]));
            }
            let diagonal = a.view_diagonal();
            assert_eq!(diagonal.to_vec(), (0..rows.min(cols)).map(|i| a[(i, i)]).collect::<Vec<_>>());
        }
    }

    #[test]
    fn empty_matrices_give_empty_views() {
        let a = Matrix::<f64>::new(0, 3);
        assert!(a.view_diagonal().is_empty());
        let b = Matrix::<f64>::new(3, 0);
        assert!(b.view_row(1).is_empty() && b.view_diagonal().is_empty());
    }

    #[test]
    fn writes_through_the_mutable_views_land_in_the_matrix() {
        let mut a = filled(4, 5, 1);
        let original = a.clone();
        a.view_diagonal_mut().fill_entries(1.0);
        a.view_row_mut(2).scale_in_place(-2.0);
        a.view_row_mut(3)[4] = 9.0;
        for j in 0..5 {
            for i in 0..4 {
                let mut expected = if i == j { 1.0 } else { original[(i, j)] };
                if i == 2 {
                    expected *= -2.0;
                }
                if (i, j) == (3, 4) {
                    expected = 9.0;
                }
                assert_eq!(a[(i, j)], expected, "({i}, {j})");
            }
        }
    }

    #[test]
    fn level_one_routines_match_the_dense_loops() {
        let a = filled(5, 7, 3);
        let b = filled(7, 7, 4);
        let (row, diagonal) = (a.view_row(1), b.view_diagonal());
        let x: Vec<f64> = row.to_vec();
        let expected: f64 = x.iter().zip(b.view_row(6).to_vec()).map(|(p, q)| p * q).sum();
        assert!((row.dot(&b.view_row(6)) - expected).abs() < 1e-12);
        assert!((row.dot(&x) - x.iter().map(|v| v * v).sum::<f64>()).abs() < 1e-12);
        assert!((diagonal.norm() - diagonal.to_vec().iter().map(|v| v * v).sum::<f64>().sqrt()).abs() < 1e-12);

        // y (a strided row) += alpha * x (a diagonal), and a contiguous target from a strided source.
        let mut c = filled(5, 5, 6);
        let d = filled(5, 5, 7);
        let before = c.view_row(2).to_vec();
        c.view_row_mut(2).add_scaled_in_place(&d.view_diagonal(), 0.5);
        let expected: Vec<f64> = before.iter().zip(d.view_diagonal().to_vec()).map(|(y, x)| y + 0.5 * x).collect();
        assert_eq!(c.view_row(2).to_vec(), expected);
        let mut contiguous = vec![1.0; 5];
        contiguous.add_scaled_in_place(&d.view_row(4), -1.0);
        assert_eq!(contiguous, d.view_row(4).to_vec().iter().map(|v| 1.0 - v).collect::<Vec<_>>());
        let mut copy = vec![0.0; 5];
        copy.copy_from(&d.view_diagonal());
        assert_eq!(copy, d.view_diagonal().to_vec());
    }

    #[test]
    fn matvec_takes_strided_vectors_on_both_sides() {
        let a = filled(4, 6, 1);
        let source = filled(6, 6, 2); // x: a row of it
        let dense = |m: &Matrix<f64>, x: &[f64]| -> Vec<f64> { (0..m.rows()).map(|i| (0..m.cols()).map(|j| m[(i, j)] * x[j]).sum()).collect() };
        let x = source.view_row(3);
        let expected = dense(&a, &x.to_vec());
        assert_eq!(&*a.matvec(&x), &expected[..]);

        // y: a row of a larger matrix, alpha/beta applied in place.
        let mut target = filled(4, 4, 5);
        let before = target.view_diagonal().to_vec();
        a.matvec_scaled_prealloc(&x, &mut target.view_diagonal_mut(), 2.0, -1.0);
        let want: Vec<f64> = expected.iter().zip(&before).map(|(e, b)| 2.0 * e - b).collect();
        assert!(target.view_diagonal().to_vec().iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-12));

        // The transposed side: y = A^T x with x a strided column-of-rows.
        let xt = source.view_diagonal().to_vec()[..4].to_vec();
        let at = a.transpose();
        assert_eq!(&*a.view_transposed().matvec(&xt), &dense(&at, &xt)[..]);
    }

    #[test]
    #[should_panic(expected = "view_row: row 3 of 3")]
    fn a_row_out_of_range_panics() {
        filled(3, 2, 0).view_row(3);
    }

    #[test]
    #[should_panic(expected = "dot: lengths")]
    fn dot_checks_the_lengths() {
        let a = filled(3, 4, 0);
        a.view_row(0).dot(&a.view_diagonal());
    }

    #[test]
    #[should_panic(expected = "add_in_place: lengths 3 and 2")]
    fn the_slice_helpers_check_the_lengths() {
        crate::vector_math::add_in_place(&mut [1.0, 2.0, 3.0], &[1.0, 2.0]);
    }
}

#[cfg(test)]
mod split_complex_tests {
    use super::*;

    type Dense = Vec<Vec<(f64, f64)>>; // [row][col] = (re, im)

    fn real(rows: usize, cols: usize, seed: usize) -> Matrix<f64> {
        Matrix::<f64>::new_from_vec(rows, cols, (0..rows * cols).map(|i| ((i * 7 + seed) % 13) as f64 * 0.25 - 1.5).collect::<Vec<_>>().into_boxed_slice())
    }

    fn complex(rows: usize, cols: usize, seed: usize) -> SplitComplexMatrix {
        SplitComplexMatrix::from_parts(real(rows, cols, seed), real(rows, cols, seed + 5))
    }

    fn dense_of(a: &SplitComplexMatrix) -> Dense {
        (0..a.rows()).map(|i| (0..a.cols()).map(|j| (a.re()[(i, j)], a.im()[(i, j)])).collect()).collect()
    }

    fn dense_real(a: &Matrix<f64>) -> Dense {
        (0..a.rows()).map(|i| (0..a.cols()).map(|j| (a[(i, j)], 0.0)).collect()).collect()
    }

    fn adjoint(a: &Dense) -> Dense {
        let (m, n) = (a.len(), a.first().map_or(0, |r| r.len()));
        (0..n).map(|j| (0..m).map(|i| (a[i][j].0, -a[i][j].1)).collect()).collect()
    }

    fn product(a: &Dense, b: &Dense) -> Dense {
        let (m, k, n) = (a.len(), b.len(), b.first().map_or(0, |r| r.len()));
        (0..m).map(|i| (0..n).map(|j| (0..k).fold((0.0, 0.0), |(re, im), l| {
            let ((ar, ai), (br, bi)) = (a[i][l], b[l][j]);
            (re + ar * br - ai * bi, im + ar * bi + ai * br)
        })).collect()).collect()
    }

    fn assert_matches<D: DataContainer<T = f64>>(got: &SplitComplexStruct<D>, expected: &Dense) {
        assert_eq!(got.rows(), expected.len());
        for (i, row) in expected.iter().enumerate() {
            assert_eq!(got.cols(), row.len());
            for (j, &(re, im)) in row.iter().enumerate() {
                assert!((got.re()[(i, j)] - re).abs() < 1e-12 && (got.im()[(i, j)] - im).abs() < 1e-12, "({i}, {j}): ({}, {}) vs ({re}, {im})", got.re()[(i, j)], got.im()[(i, j)]);
            }
        }
    }

    #[test]
    fn every_combination_of_complex_real_and_adjoint_matches_the_dense_product() {
        let (a, b) = (complex(4, 3, 1), complex(3, 5, 2));
        let (r_left, r_right) = (real(4, 3, 3), real(3, 5, 4));
        let (da, db) = (dense_of(&a), dense_of(&b));
        assert_matches(&a.matmul(&b), &product(&da, &db));
        assert_matches(&a.matmul(&r_right), &product(&da, &dense_real(&r_right)));
        assert_matches(&r_left.matmul_complex(&b), &product(&dense_real(&r_left), &db));

        // Adjoints on either side, and on both.
        let c = complex(4, 5, 6); // c^H is 5 x 4
        let dc = dense_of(&c);
        assert_matches(&c.view_adjoint().matmul(&a), &product(&adjoint(&dc), &da));
        let d = complex(5, 4, 7); // d^H is 4 x 5
        assert_matches(&a.view_adjoint().matmul(&d.view_adjoint()), &product(&adjoint(&da), &adjoint(&dense_of(&d))));
        assert_matches(&b.matmul(&c.view_adjoint()), &product(&db, &adjoint(&dc)));
        assert_matches(&a.view_adjoint().matmul(&a), &product(&adjoint(&da), &da));
        assert_matches(&r_left.view_transposed().matmul_complex(&a), &product(&adjoint(&dense_real(&r_left)), &da));
    }

    #[test]
    fn strided_and_transposed_real_factors_work_on_either_side() {
        let big = real(7, 3, 8);
        let a = complex(4, 5, 9);
        let block = real(5, 3, 0); // copied rows 1..6 of big, for the reference
        let mut block = block;
        for j in 0..3 {
            for i in 0..5 {
                block[(i, j)] = big[(i + 1, j)];
            }
        }
        assert_matches(&a.matmul(&big.view_rows(1..6)), &product(&dense_of(&a), &dense_real(&block)));
        assert_matches(&big.view_rows(1..6).view_transposed().matmul_complex(&a.view_adjoint()), &product(&adjoint(&dense_real(&block)), &adjoint(&dense_of(&a))));
    }

    #[test]
    fn alpha_and_beta_scale_both_parts() {
        let (a, b) = (complex(3, 4, 1), complex(4, 2, 2));
        let start = complex(3, 2, 3);
        for (alpha, beta) in [(2.0, 0.5), (-1.0, 0.0), (0.5, 1.0)] {
            let mut got = complex(3, 2, 3);
            a.matmul_scaled_prealloc(&b, &mut got, alpha, beta);
            let ab = product(&dense_of(&a), &dense_of(&b));
            let expected: Dense = ab.iter().zip(dense_of(&start)).map(|(p, s)| p.iter().zip(s).map(|(&(pr, pi), (sr, si))| (alpha * pr + beta * sr, alpha * pi + beta * si)).collect()).collect();
            assert_matches(&got, &expected);

            // A real left factor with a real right factor's imaginary part zero: the imaginary output is beta * old.
            let mut got = complex(3, 2, 3);
            real(3, 4, 7).matmul_complex_scaled_prealloc(&b, &mut got, alpha, beta);
            let rb = product(&dense_real(&real(3, 4, 7)), &dense_of(&b));
            let expected: Dense = rb.iter().zip(dense_of(&start)).map(|(p, s)| p.iter().zip(s).map(|(&(pr, pi), (sr, si))| (alpha * pr + beta * sr, alpha * pi + beta * si)).collect()).collect();
            assert_matches(&got, &expected);
        }
    }

    #[test]
    fn an_empty_inner_dimension_scales_the_output() {
        let (a, b) = (complex(3, 0, 1), complex(0, 2, 2));
        let mut got = complex(3, 2, 3);
        let start = dense_of(&got);
        a.matmul_scaled_prealloc(&b, &mut got, 1.0, 0.5);
        let expected: Dense = start.iter().map(|r| r.iter().map(|&(re, im)| (0.5 * re, 0.5 * im)).collect()).collect();
        assert_matches(&got, &expected);
        assert_eq!(complex(0, 4, 1).matmul(&complex(4, 3, 2)).rows(), 0);
    }

    #[test]
    fn the_arena_forms_match_the_heap_forms() {
        use crate::scratch::{Scratch, ByteUnit};
        let (a, b, r) = (complex(4, 3, 1), complex(3, 5, 2), real(4, 3, 3));
        let mut scratch = Scratch::new(1, ByteUnit::MiB).unwrap();
        let arena = scratch.get_arena();
        let heap = a.matmul(&b);
        let from_arena = a.matmul_arena(&b, &arena).unwrap();
        assert_eq!((from_arena.re().get_data(), from_arena.im().get_data()), (heap.re().get_data(), heap.im().get_data()));
        let heap = r.matmul_complex(&b);
        let from_arena = r.matmul_complex_arena(&b, &arena).unwrap();
        assert_eq!((from_arena.re().get_data(), from_arena.im().get_data()), (heap.re().get_data(), heap.im().get_data()));
        let (heap, from_arena) = (a.adjoint(), a.adjoint_arena(&arena).unwrap());
        assert_eq!((from_arena.re().get_data(), from_arena.im().get_data()), (heap.re().get_data(), heap.im().get_data()));
        assert_eq!(&*a.to_interleaved_arena(&arena).unwrap(), &*a.to_interleaved());
    }

    #[test]
    fn the_adjoint_copy_and_the_interleaved_layout() {
        let a = complex(3, 4, 2);
        assert_matches(&a.adjoint(), &adjoint(&dense_of(&a)));
        let interleaved = a.to_interleaved();
        assert_eq!(interleaved.len(), 24);
        // Column-major, (re, im) per element.
        assert_eq!((interleaved[2 * (1 + 3 * 2)], interleaved[2 * (1 + 3 * 2) + 1]), (a.re()[(1, 2)], a.im()[(1, 2)]));
        let back = SplitComplexMatrix::from_interleaved(3, 4, &interleaved);
        assert_eq!((back.re().get_data(), back.im().get_data()), (a.re().get_data(), a.im().get_data()));
    }

    #[test]
    fn mutable_parts_write_through() {
        let mut a = SplitComplexMatrix::new(2, 2);
        a.re_mut()[(0, 1)] = 3.0;
        let (re, im) = a.parts_mut();
        im[(1, 0)] = -2.0;
        re[(1, 1)] = 1.0;
        assert_eq!((a.re()[(0, 1)], a.im()[(1, 0)], a.re()[(1, 1)], a.im()[(0, 1)]), (3.0, -2.0, 1.0, 0.0));
        assert_eq!(a.view().re().get_data(), a.re().get_data());
    }

    #[test]
    #[should_panic(expected = "inner dimension mismatch")]
    fn mismatched_shapes_panic() {
        complex(3, 4, 0).matmul(&complex(3, 4, 1));
    }

    #[test]
    #[should_panic(expected = "from_parts")]
    fn parts_must_have_the_same_shape() {
        SplitComplexMatrix::from_parts(real(2, 3, 0), real(3, 2, 0));
    }

    /// A Hermitian matrix, (A + A^H) / 2 of an unstructured complex A.
    fn hermitian(n: usize, seed: usize) -> SplitComplexMatrix {
        let a = dense_of(&complex(n, n, seed));
        let (mut re, mut im) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
        for i in 0..n {
            for j in 0..n {
                re[(i, j)] = 0.5 * (a[i][j].0 + a[j][i].0);
                im[(i, j)] = 0.5 * (a[i][j].1 - a[j][i].1);
            }
        }
        SplitComplexMatrix::from_parts(re, im)
    }

    /// `V diag(values) V^H` equals `h`, and `V^H V = 1`, to `tolerance` relative to the largest eigenvalue.
    fn assert_decomposes(h: &SplitComplexMatrix, values: &[f64], vectors: &SplitComplexMatrix, tolerance: f64) {
        let n = h.rows();
        let scale = values.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        assert!(values.windows(2).all(|w| w[0] <= w[1]), "eigenvalues not ascending: {values:?}");
        let v = dense_of(vectors);
        let scaled: Dense = v.iter().map(|row| row.iter().zip(values).map(|(&(re, im), &l)| (l * re, l * im)).collect()).collect();
        let rebuilt = product(&scaled, &adjoint(&v));
        let overlap = product(&adjoint(&v), &v);
        let h = dense_of(h);
        for i in 0..n {
            for j in 0..n {
                let (re, im) = rebuilt[i][j];
                assert!((re - h[i][j].0).abs() < tolerance * scale && (im - h[i][j].1).abs() < tolerance * scale, "V L V^H ({i}, {j}): ({re}, {im}) vs {:?}", h[i][j]);
                let (re, im) = overlap[i][j];
                let one = if i == j { 1.0 } else { 0.0 };
                assert!((re - one).abs() < tolerance && im.abs() < tolerance, "V^H V ({i}, {j}): ({re}, {im})");
            }
        }
    }

    #[test]
    fn a_hermitian_matrix_is_v_diag_values_v_adjoint_with_orthonormal_v() {
        for (n, seed) in [(1, 3), (2, 4), (50, 5)] {
            let h = hermitian(n, seed);
            let (values, vectors) = h.eigenvectors();
            assert_eq!(values.len(), n);
            assert_decomposes(&h, &values, &vectors, 1e-12);
        }
    }

    #[test]
    fn a_real_symmetric_matrix_has_dsyev_s_eigenvalues() {
        let h = hermitian(7, 2);
        let real_part = Matrix::<f64>::new_from_vec(7, 7, h.re().get_data().to_vec().into_boxed_slice());
        let symmetric = SplitComplexMatrix::from_parts(real_part.clone(), Matrix::<f64>::new(7, 7));
        for (complex, real) in symmetric.eigenvalues().iter().zip(real_part.eigenvalues().iter()) {
            assert!((complex - real).abs() < 1e-12, "{complex} vs {real}");
        }
    }

    #[test]
    fn sigma_y_and_a_degenerate_pair() {
        // sigma_y = [[0, -i], [i, 0]]: eigenvalues -1 and +1.
        let mut sigma_y = SplitComplexMatrix::new(2, 2);
        sigma_y.im_mut()[(0, 1)] = -1.0;
        sigma_y.im_mut()[(1, 0)] = 1.0;
        let (values, vectors) = sigma_y.eigenvectors();
        assert!((values[0] + 1.0).abs() < 1e-14 && (values[1] - 1.0).abs() < 1e-14, "{values:?}");
        assert_decomposes(&sigma_y, &values, &vectors, 1e-14);

        // 2 + v v^H in three dimensions: 2 twice, then 2 + |v|^2.
        let v = [(1.0, 0.5), (-0.25, 1.0), (0.5, -0.75)];
        let mut h = SplitComplexMatrix::new(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let ((ar, ai), (br, bi)) = (v[i], v[j]);
                h.re_mut()[(i, j)] = ar * br + ai * bi + if i == j { 2.0 } else { 0.0 };
                h.im_mut()[(i, j)] = ai * br - ar * bi;
            }
        }
        let norm2: f64 = v.iter().map(|(r, i)| r * r + i * i).sum();
        let (values, vectors) = h.eigenvectors();
        assert!((values[0] - 2.0).abs() < 1e-13 && (values[1] - 2.0).abs() < 1e-13 && (values[2] - 2.0 - norm2).abs() < 1e-13, "{values:?}");
        assert_decomposes(&h, &values, &vectors, 1e-13);
    }

    #[test]
    fn the_eigenvalues_are_the_real_embedding_s_each_once() {
        // [[A, -B], [B, A]] for H = A + i B has every eigenvalue of H twice.
        let n = 6;
        let h = hermitian(n, 8);
        let mut embedding = Matrix::<f64>::new(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                let (a, b) = (h.re()[(i, j)], h.im()[(i, j)]);
                embedding[(i, j)] = a;
                embedding[(i + n, j + n)] = a;
                embedding[(i, j + n)] = -b;
                embedding[(i + n, j)] = b;
            }
        }
        let doubled = embedding.eigenvalues();
        for (k, value) in h.eigenvalues().iter().enumerate() {
            assert!((doubled[2 * k] - value).abs() < 1e-12 && (doubled[2 * k + 1] - value).abs() < 1e-12, "{k}: {value} vs {:?}", &doubled[2 * k..2 * k + 2]);
        }
    }

    #[test]
    fn the_three_forms_agree_and_a_short_scratch_is_reported() {
        use crate::scratch::{Scratch, ByteUnit};
        let n = 9;
        let h = hermitian(n, 11);
        let (values, vectors) = h.eigenvectors();
        // Eigenvalues alone take another LAPACK path than with the eigenvectors: equal to rounding, not to the bit.
        let values_only = h.eigenvalues();
        for (a, b) in values_only.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }

        // Arena: the same computations as the heap forms.
        let mut scratch = Scratch::new(1, ByteUnit::MiB).unwrap();
        let mut arena = scratch.get_arena();
        let (arena_values, arena_vectors) = h.eigenvectors_arena(&mut arena).unwrap();
        assert_eq!(&*arena_values, &*values);
        assert_eq!((arena_vectors.re().get_data(), arena_vectors.im().get_data()), (vectors.re().get_data(), vectors.im().get_data()));
        assert_eq!(&*h.eigenvalues_arena(&mut arena).unwrap(), &*values_only);

        // Prealloc, into views over larger buffers and with more work than needed (a caller sizing for the largest n).
        let mut work = vec![0.0; hermitian_eigenvectors_work_len(n) + 100];
        let mut value_storage = vec![0.0; n + 3];
        let (mut re_storage, mut im_storage) = (vec![0.0; n * n], vec![0.0; n * n]);
        let mut view = SplitComplexStruct::from_parts(MatrixViewMut::new(n, n, &mut re_storage), MatrixViewMut::new(n, n, &mut im_storage));
        assert!(h.eigenvectors_prealloc(&mut value_storage, &mut view, &mut work));
        let copied = SplitComplexMatrix::from_parts(
            Matrix::<f64>::new_from_vec(n, n, view.re().get_data().to_vec().into_boxed_slice()),
            Matrix::<f64>::new_from_vec(n, n, view.im().get_data().to_vec().into_boxed_slice()),
        );
        assert_decomposes(&h, &value_storage[..n], &copied, 1e-12);
        assert!(h.eigenvalues_prealloc(&mut value_storage, &mut work));
        for (a, b) in value_storage.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // Below the minimum (no room for the complex scratch zheev needs) the call fails instead of writing past
        // the end of `work`; so does a complex scratch below LAPACK's minimum 2n - 1.
        let mut out = SplitComplexMatrix::new(n, n);
        for short in [10, 2 * n * n + 3 * n - 2 + 2] {
            let mut work = vec![0.0; short];
            assert!(!h.eigenvectors_prealloc(&mut value_storage, &mut out, &mut work), "work of {short}");
            assert!(!h.eigenvalues_prealloc(&mut value_storage, &mut work), "work of {short}");
        }
        assert!(hermitian_eigenvectors_work_len(n) >= 2 * n * n + 3 * n - 2 + 2 * (2 * n - 1));

        // n = 0: nothing to do.
        let empty = SplitComplexMatrix::new(0, 0);
        assert!(empty.eigenvalues().is_empty());
    }
}

mod block_tests {
    use super::*;

    // A deterministic, unstructured matrix.
    fn filled(rows: usize, cols: usize, seed: usize) -> Matrix<f64> {
        let mut m = Matrix::<f64>::new(rows, cols);
        for (j, col) in m.iter_col_mut().enumerate() {
            for (i, x) in col.iter_mut().enumerate() {
                *x = (((i * 7 + j * 13 + seed * 3) % 17) as f64 - 8.0) / 5.0;
            }
        }
        m
    }

    // The block of `a` at `at`, `rows x cols`, as its own matrix.
    fn block(a: &Matrix<f64>, at: (usize, usize), rows: usize, cols: usize) -> Matrix<f64> {
        let mut b = Matrix::<f64>::new(rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                b[(i, j)] = a[(at.0 + i, at.1 + j)];
            }
        }
        b
    }

    #[test]
    fn block_sandwich_matches_dense_products() {
        let a = filled(9, 11, 1);
        let (left, right) = (filled(3, 4, 2), filled(5, 2, 3));
        let (a_at, out_at) = ((4, 2), (1, 3));
        let mut out = filled(6, 7, 4);
        let before = out.clone();
        let mut scratch = [0.0; 3];
        block_sandwich_prealloc(&left.view(), &a, a_at, &right.view(), &mut out, out_at, 0.5, 2.0, &mut scratch);
        let expected = left.view_transposed().matmul(&block(&a, a_at, 3, 5)).matmul(&right);
        for j in 0..7 {
            for i in 0..6 {
                let inside = (out_at.0..out_at.0 + 4).contains(&i) && (out_at.1..out_at.1 + 2).contains(&j);
                let want = if inside { 0.5 * expected[(i - out_at.0, j - out_at.1)] + 2.0 * before[(i, j)] } else { before[(i, j)] };
                assert!((out[(i, j)] - want).abs() < 1e-12, "({i}, {j}): {} vs {want}", out[(i, j)]);
            }
        }
    }

    #[test]
    fn block_sandwich_with_beta_zero_ignores_the_old_output() {
        let a = filled(4, 4, 5);
        let mut id = Matrix::<f64>::new(4, 4);
        (0..4).for_each(|i| id[(i, i)] = 1.0);
        let mut out = Matrix::<f64>::new(4, 4);
        out.get_data_mut().fill(f64::NAN);
        let mut scratch = [0.0; 4];
        block_sandwich_prealloc(&id.view(), &a, (0, 0), &id.view(), &mut out, (0, 0), 1.0, 0.0, &mut scratch);
        assert_eq!(out.get_data(), a.get_data());
    }

    #[test]
    fn signed_permutation_matches_the_dense_sandwich() {
        let a = filled(8, 8, 6);
        // Column j of the factor: its nonzero sign in row rows[j].
        let (l_rows, l_signs) = ([2usize, 0, 1], [1.0, -1.0, -1.0]);
        let (r_rows, r_signs) = ([1usize, 3, 0, 2], [-1.0, 1.0, 1.0, -1.0]);
        let dense = |rows: &[usize], signs: &[f64]| {
            let mut m = Matrix::<f64>::new(rows.len(), rows.len());
            for (j, (&r, &s)) in rows.iter().zip(signs).enumerate() {
                m[(r, j)] = s;
            }
            m
        };
        let (left, right) = (dense(&l_rows, &l_signs), dense(&r_rows, &r_signs));
        let mut fast = filled(5, 6, 7);
        let mut reference = fast.clone();
        let mut scratch = [0.0; 3];
        block_signed_permutation_prealloc(
            SignedPermutation { rows: &l_rows, signs: &l_signs },
            &a,
            (3, 2),
            SignedPermutation { rows: &r_rows, signs: &r_signs },
            &mut fast,
            (1, 1),
            -1.5,
            1.0,
        );
        block_sandwich_prealloc(&left.view(), &a, (3, 2), &right.view(), &mut reference, (1, 1), -1.5, 1.0, &mut scratch);
        for (x, y) in fast.get_data().iter().zip(reference.get_data()) {
            assert!((x - y).abs() < 1e-14);
        }
    }
}

#[cfg(test)]
mod singular_value_decomposition_tests {
    use super::*;

    fn matrix(rows: usize, cols: usize) -> Matrix<f64> {
        let mut m = Matrix::<f64>::new(rows, cols);
        for (j, column) in m.iter_mut().enumerate() {
            for (i, x) in column.iter_mut().enumerate() {
                *x = ((i * 5 + j * 3) as f64 * 0.71).sin() + if i == j { 0.5 } else { 0.0 };
            }
        }
        m
    }

    /// The thin SVD into caller storage, for tall, wide and square matrices: U diag(s) V^T is the matrix, U and V have
    /// orthonormal columns, the values descend and are those of the full (allocating) SVD.
    #[test]
    fn the_thin_svd_reconstructs_the_matrix_with_orthonormal_factors() {
        for (m, n) in [(7, 4), (4, 7), (5, 5), (1, 3)] {
            let a = matrix(m, n);
            let k = m.min(n);
            let (mut values, mut u, mut vt, mut work) = (vec![0.0; k], Matrix::<f64>::new(m, k), Matrix::<f64>::new(k, n), vec![0.0; svd_work_len(m, n)]);
            assert!(a.singular_value_decomposition_prealloc(&mut values, &mut u, &mut vt, &mut work));
            assert!(values.windows(2).all(|w| w[0] >= w[1]) && values.iter().all(|&s| s >= 0.0), "{m} x {n}: {values:?}");
            let mut scaled = u.clone();
            for (column, &s) in scaled.iter_mut().zip(&values) {
                column.iter_mut().for_each(|x| *x *= s);
            }
            let rebuilt = scaled.matmul(&vt);
            for (x, y) in rebuilt.get_data().iter().zip(a.get_data()) {
                assert!((x - y).abs() < 1e-12, "{m} x {n}: rebuilt {x} against {y}");
            }
            for (gram, what) in [(u.view_transposed().matmul(&u), "U^T U"), (vt.matmul(&vt.view_transposed()), "V^T V")] {
                for i in 0..k {
                    for j in 0..k {
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!((gram[(i, j)] - expected).abs() < 1e-12, "{m} x {n}: {what} ({i}, {j}) = {}", gram[(i, j)]);
                    }
                }
            }
            let full = a.singular_value_decomposition();
            for (x, y) in values.iter().zip(full.s.iter()) {
                assert!((x - y).abs() < 1e-12, "{m} x {n}: {x} against the full SVD's {y}");
            }
        }
    }

    /// A matrix with an empty dimension has nothing to decompose.
    #[test]
    fn an_empty_matrix_decomposes_to_nothing() {
        let a = Matrix::<f64>::new(0, 3);
        let (mut u, mut vt, mut work) = (Matrix::<f64>::new(0, 0), Matrix::<f64>::new(0, 3), vec![0.0; svd_work_len(0, 3)]);
        assert!(a.singular_value_decomposition_prealloc(&mut [], &mut u, &mut vt, &mut work));
    }
}

// `backslash`: the solution, `cols` values, by LAPACK's dgels; a matrix not of full rank is refused.
#[cfg(test)]
mod backslash_tests {
    use super::*;

    fn matrix(rows: usize, cols: usize, values: &[f64]) -> Matrix<f64> {
        Matrix::<f64>::new_from_vec(rows, cols, values.to_vec().into_boxed_slice())
    }

    #[test]
    fn a_square_system_is_solved_exactly() {
        let a = matrix(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, -1.0, 0.5, -1.0, 2.0]);
        let x = [0.3, -1.2, 2.5];
        let b = a.matvec(&x);
        let solved = a.backslash(&b);
        assert_eq!(solved.len(), 3);
        assert!(solved.iter().zip(&x).all(|(s, x)| (s - x).abs() < 1e-12), "{solved:?}");
    }

    #[test]
    fn more_rows_than_columns_give_the_least_squares_fit_of_cols_values() {
        // A straight line through four points: the normal equations' solution.
        let a = matrix(4, 2, &[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0]);
        let b = [1.0, 2.9, 5.1, 7.0];
        let fit = a.backslash(&b);
        assert_eq!(fit.len(), 2, "the solution, not dgels' padded output");
        let normal = a.view_transposed().matmul(&a);
        let rhs = a.view_transposed().matvec(&b);
        let direct = normal.backslash(&rhs);
        assert!(fit.iter().zip(direct.iter()).all(|(f, d)| (f - d).abs() < 1e-12), "{fit:?} vs {direct:?}");
    }

    #[test]
    fn fewer_rows_than_columns_give_the_minimum_norm_solution() {
        // x + y = 2: the minimum-norm solution is (1, 1).
        let a = matrix(1, 2, &[1.0, 1.0]);
        let x = a.backslash(&[2.0]);
        assert_eq!(x.len(), 2);
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12, "{x:?}");
    }

    #[test]
    #[should_panic(expected = "not of full rank")]
    fn a_rank_deficient_matrix_is_refused() {
        // A zero column: R has an exact zero on its diagonal, which dgels reports.
        let a = matrix(3, 2, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let _ = a.backslash(&[1.0, 2.0, 3.0]);
    }
}
