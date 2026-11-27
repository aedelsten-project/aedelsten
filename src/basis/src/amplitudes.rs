// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The values and gradients of the atomic orbitals at a set of points, given as a `(4, n_points)` matrix of x, y, z
//! and a weight (the weight is not read), point by point and shell by shell: the reference `ao_values` is tested
//! against.

use arrays::matrix::{MatrixTrait, MatrixTraitMut};

/// Every atomic orbital's value at every point of `grid` into `amplitude_matrix` (`n_ao x n_points`): column `j` the
/// orbitals at point `j`, shell by shell in the basis order (`n_cont * degeneracy` per shell).
pub fn compute_ao_amplitudes<S, Mmut, Mg>(
    basis_shells: &[crate::shell_types::BasisShell<S>],
    amplitude_matrix: &mut Mmut,
    grid: &Mg,
) where
    S: crate::contraction::ShellKind,
    Mmut: MatrixTraitMut<f64>,
    Mg: MatrixTrait<f64>,
{
    debug_assert_eq!(grid.rows(), 4);
    debug_assert_eq!(amplitude_matrix.cols(), grid.cols());
    for (amplitudes, point) in amplitude_matrix.iter_col_mut().zip(grid.iter_col()) {
        let x = point[0];
        let y = point[1];
        let z = point[2];
        let mut amp_start = 0;
        for shell in basis_shells.iter() {
            let shell_amplitudes = shell.compute_ao_amplitude(x, y, z);
            let amp_end = amp_start + shell_amplitudes.len();
            amplitudes[amp_start..amp_end].copy_from_slice(&shell_amplitudes);
            amp_start = amp_end;
        }
    }
}

/// Every atomic orbital's gradient at every point of `grid` into `gradient_x`, `gradient_y`, `gradient_z`, each laid
/// out as in `compute_ao_amplitudes`.
pub fn compute_ao_amplitude_gradients<S, Mmut, Mg>(
    basis_shells: &[crate::shell_types::BasisShell<S>],
    gradient_x: &mut Mmut,
    gradient_y: &mut Mmut,
    gradient_z: &mut Mmut,
    grid: &Mg,
) where
    S: crate::contraction::ShellKind,
    Mmut: MatrixTraitMut<f64>,
    Mg: MatrixTrait<f64>,
{
    debug_assert_eq!(grid.rows(), 4);
    debug_assert_eq!(gradient_x.cols(), grid.cols());
    debug_assert_eq!(gradient_y.cols(), grid.cols());
    debug_assert_eq!(gradient_z.cols(), grid.cols());
    for (((gx, gy), gz), point) in gradient_x.iter_col_mut()
        .zip(gradient_y.iter_col_mut())
        .zip(gradient_z.iter_col_mut())
        .zip(grid.iter_col())
    {
        let x = point[0];
        let y = point[1];
        let z = point[2];
        let mut amp_start = 0;
        for shell in basis_shells.iter() {
            let [shell_gx, shell_gy, shell_gz] = shell.compute_ao_amplitude_gradient(x, y, z);
            let amp_end = amp_start + shell_gx.len();
            gx[amp_start..amp_end].copy_from_slice(&shell_gx);
            gy[amp_start..amp_end].copy_from_slice(&shell_gy);
            gz[amp_start..amp_end].copy_from_slice(&shell_gz);
            amp_start = amp_end;
        }
    }
}
