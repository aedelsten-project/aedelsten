// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The spin-free exact two-component (X2C) one-electron Hamiltonian by the one-step construction (M. Iliaš and
//! T. Saue, J. Chem. Phys. 126, 064102 (2007); `citations.rs`: `ILIAS_SAUE_2007`), and the picture-change
//! transformation of other one-electron operators.

use arrays::matrix::*;

/// The X2C decoupling transforms that turn 4-component large/small-component
/// AO integrals into their 2-component (large-component-only) counterparts.
/// `u_large` transforms plain one-electron operators (S, T, V, ...);
/// `u_small` transforms their pXp picture-changed counterparts. Both are
/// needed together to fold any one-electron property operator into the X2C
/// picture, via `transform_one_electron_operator`.
#[derive(Debug)]
pub struct PictureChange {
    u_large: Matrix<f64>,
    u_small: Matrix<f64>,
    /// The speed of light the transforms were made with.
    c_light: f64,
    /// Supplied by the caller (`from_transforms`) rather than derived from the Hamiltonian.
    supplied: bool,
}

impl PictureChange {
    /// Transforms supplied by the caller (for an X2C Hamiltonian with the decoupling held at another geometry,
    /// `frozen_one_electron_hamiltonian`): `u_large` U_L and `u_small` U_S with its 1/2c, made with `c_light`.
    pub fn from_transforms(u_large: Matrix<f64>, u_small: Matrix<f64>, c_light: f64) -> Self {
        PictureChange { u_large, u_small, c_light, supplied: true }
    }

    /// Whether the transforms were supplied (`from_transforms`) rather than derived.
    pub fn is_supplied(&self) -> bool {
        self.supplied
    }

    /// The speed of light the transforms were made with.
    pub fn get_c_light(&self) -> f64 {
        self.c_light
    }

    /// The X2C one-electron Hamiltonian of `t`, `v` and `pvp` with these transforms held, U_S = 2c u_small:
    /// `U_L^T V U_L + U_L^T T U_S + U_S^T T U_L - U_S^T T U_S + U_S^T W U_S / 4c^2`, which is what
    /// `assemble_x2c_one_electron_hamiltonian` gives with the transforms it derives.
    pub fn frozen_one_electron_hamiltonian<M: MatrixMath<f64> + Factor, O: MatrixMathMut<f64>>(&self, oneham: &mut O, t: &M, v: &M, pvp: &M) {
        let n = self.u_large.rows();
        let two_c = 2.0 * self.c_light;
        let mut scratch = Matrix::<f64>::new(n, n);
        let mut term = Matrix::<f64>::new(n, n);
        v.congruence_transform_prealloc(&self.u_large, &mut scratch, &mut term);
        oneham.get_data_mut().copy_from_slice(term.get_data());
        pvp.congruence_transform_prealloc(&self.u_small, &mut scratch, &mut term);
        oneham.add_in_place(&term);
        t.congruence_transform_prealloc(&self.u_small, &mut scratch, &mut term);
        oneham.add_scaled_in_place(&term, -two_c * two_c);
        // U_L^T T u_small and its transpose, times 2c.
        let t_us = t.matmul(&self.u_small);
        let mixed = self.u_large.view_transposed().matmul(&t_us);
        for j in 0..n {
            for i in 0..n {
                oneham[(i, j)] += two_c * (mixed[(i, j)] + mixed[(j, i)]);
            }
        }
    }

    /// U_L, for the plain operators.
    pub fn get_u_large(&self) -> &Matrix<f64> {
        &self.u_large
    }

    /// U_S, for the pXp operators, the 1/2c kinetic-balance factor included.
    pub fn get_u_small(&self) -> &Matrix<f64> {
        &self.u_small
    }

    /// Both transforms, mutably: for cleaning them in place (point-group symmetry: from symmetric inputs they are
    /// symmetric up to the decoupling's rounding, which a caller may average away).
    pub fn get_transforms_mut(&mut self) -> (&mut Matrix<f64>, &mut Matrix<f64>) {
        (&mut self.u_large, &mut self.u_small)
    }

    /// Folds a one-electron property operator into the X2C picture:
    /// `out = u_large^T op_plain u_large + u_small^T op_pxp u_small`.
    /// `op_plain` is the operator's AO-basis matrix (e.g. a dipole moment integral); `op_pxp` is its pXp
    /// counterpart (the operator between momentum operators, as pVp is to V). For symmetric (scalar) operators only:
    /// the result is made exactly symmetric by mirroring its upper triangle; the spin planes of an operator take
    /// `transform_one_electron_spin_operator_prealloc`. Allocates its own buffers;
    /// `transform_one_electron_operator_prealloc` reuses the caller's.
    pub fn transform_one_electron_operator<T, TMut>(
        &self,
        op_plain: &T,
        op_pxp: &T,
        out: &mut TMut,
    )
    where
        T: MatrixMath<f64>,
        TMut: MatrixTraitMut<f64>,
    {
        let n = self.u_large.rows();
        let mut scratch = Matrix::<f64>::new(n, n);
        let mut transformed_plain = Matrix::<f64>::new(n, n);
        let mut transformed_pxp = Matrix::<f64>::new(n, n);
        self.transform_one_electron_operator_prealloc(
            op_plain, op_pxp, &mut scratch, &mut transformed_plain, &mut transformed_pxp, out,
        );
    }

    /// `transform_one_electron_operator` into caller-supplied buffers: `scratch`, `transformed_plain` and
    /// `transformed_pxp`, each (n, n) with n = get_u_large().rows(), reusable across calls.
    pub fn transform_one_electron_operator_prealloc<T, TMut, TScratch>(
        &self,
        op_plain: &T,
        op_pxp: &T,
        scratch: &mut TScratch,
        transformed_plain: &mut TScratch,
        transformed_pxp: &mut TScratch,
        out: &mut TMut,
    )
    where
        T: MatrixMath<f64>,
        TMut: MatrixTraitMut<f64>,
        TScratch: MatrixTraitMut<f64>,
    {
        // u^T op u, by the BLAS transpose flags.
        op_plain.congruence_transform_prealloc(&self.u_large, scratch, transformed_plain);
        op_pxp.congruence_transform_prealloc(&self.u_small, scratch, transformed_pxp);
        // The upper triangle, mirrored: the result is exactly symmetric.
        for j in 0..out.cols() {
            for i in 0..=j {
                let val = transformed_plain[(i, j)] + transformed_pxp[(i, j)];
                out[(j, i)] = val;
                out[(i, j)] = val;
            }
        }
    }
}

impl PictureChange {
    /// Folds a spin plane of a one-electron operator into the X2C picture: `out = u_small^T op_pxp u_small`, `op_pxp`
    /// a component `(p X x p)_k` of `(sigma.p) X (sigma.p) = p.Xp + i sigma.(p X x p)` (raw, the `1/2c` kinetic
    /// balance of `u_small` supplies the `1/4c^2`). No large-component term and no symmetrisation: the spin planes are
    /// antisymmetric. `scratch` (n, n).
    pub fn transform_one_electron_spin_operator_prealloc<T, TMut, TScratch>(&self, op_pxp: &T, scratch: &mut TScratch, out: &mut TMut)
    where
        T: MatrixMath<f64>,
        TMut: MatrixTraitMut<f64>,
        TScratch: MatrixTraitMut<f64>,
    {
        op_pxp.congruence_transform_prealloc(&self.u_small, scratch, out);
    }
}

/// Builds the spin-free X2C one-electron Hamiltonian from the AO-basis overlap, kinetic, potential and pVp
/// integrals by the one-step construction: the modified-Dirac generalized eigenvalue problem gives the
/// kinetic-balance X-matrix, which renormalizes S and T, from which the decoupling transforms U_L/U_S fold
/// everything into an n x n Hamiltonian. Returns the transforms (`PictureChange`) for other operators. `pvp`
/// is only read (unscaled; the 1/4c^2 is applied where it is used).
pub fn assemble_x2c_one_electron_hamiltonian<Mf64, Mf64Mut>(
    oneham: &mut Mf64Mut,
    s: &Mf64,
    t: &Mf64,
    v: &Mf64,
    pvp: &Mf64,
    c_light: f64,
) -> PictureChange
where
    Mf64: MatrixMath<f64> + Factor,
    Mf64Mut: MatrixMathMut<f64>,
{

    let n = oneham.rows();
    debug_assert_eq!(n, oneham.cols());
    debug_assert_eq!(n, s.rows());
    debug_assert_eq!(n, s.cols());
    debug_assert_eq!(n, t.rows());
    debug_assert_eq!(n, t.cols());
    debug_assert_eq!(n, v.rows());
    debug_assert_eq!(n, v.cols());
    debug_assert_eq!(n, pvp.rows());
    debug_assert_eq!(n, pvp.cols());

    let m = n + n;

    // Augmented 2n x 2n modified-Dirac generalized eigenvalue problem
    // (h_tilde_2c) C = (s_tilde_2c) C epsilon, in the large/small-component
    // block layout: rows/cols [0, n) are the large component, [n, 2n) the
    // small component.
    let mut s_tilde_2c = Matrix::<f64>::new(m, m);
    let mut h_tilde_2c = Matrix::<f64>::new(m, m);

    let inv_two_c = 1.0 / (2.0 * c_light);
    let inv_two_c_sq = 1.0 / (2.0 * c_light * c_light);
    let inv_four_c_sq = inv_two_c_sq / 2.0;

    construct_2c_matrices(&mut s_tilde_2c, &mut h_tilde_2c, s, t, v, pvp, inv_two_c_sq, inv_four_c_sq);

    let x_matrix = construct_x_matrix(h_tilde_2c, s_tilde_2c, n);
    let (s_renorm, t_renorm) = x_project_s_and_t(s, t, &x_matrix, inv_two_c_sq, n);

    // Final large-component renormalization:
    //   R = S^-1/2 [S^-1/2 s_renorm S^-1/2]^-1/2 S^1/2,
    // S^-1/2 and S^1/2 from one eigendecomposition of S.
    let (s_sqrt, s_inv_sqrt) = s.symmetric_sqrt_and_inverse_sqrt();
    let renorm_factor = s_renorm.congruence_transform(&s_inv_sqrt).symmetric_inverse_sqrt();
    let partial_renorm = s_inv_sqrt.matmul(&renorm_factor);

    let u_large = partial_renorm.matmul(&s_sqrt);
    let mut u_small = x_matrix.matmul(&u_large);

    // u^T op u, by the BLAS transpose flags.
    let t_transformed = t_renorm.congruence_transform(&u_large);
    let v_transformed = v.congruence_transform(&u_large);
    // U_S^T pVp U_S, then scaled by 1/4c^2.
    let mut pvp_transformed = pvp.congruence_transform(&u_small);
    for val in pvp_transformed.get_data_mut().iter_mut() {
        *val *= inv_four_c_sq;
    }

    {
        let oneham_data = oneham.get_data_mut();
        let t_transformed_data = t_transformed.get_data();
        oneham_data.copy_from_slice(t_transformed_data);
    }
    oneham.add_in_place(&v_transformed);
    oneham.add_in_place(&pvp_transformed);

    // The 1/2c kinetic-balance factor into U_S, for every pXp operator transformed with it.
    {
        let u_small_data = u_small.get_data_mut();
        for val in u_small_data.iter_mut() {
            *val *= inv_two_c;
        }
    }

    PictureChange { u_large, u_small, c_light, supplied: false }
}

/// Fills the augmented 2n x 2n modified-Dirac metric/Fock pencil
/// (s_tilde_2c, h_tilde_2c) from the AO-basis S/T/V/pVp blocks, in the
/// standard large/small-component block layout:
///   s_tilde_2c = [[S, 0], [0, T / 2c^2]]
///   h_tilde_2c = [[V, T], [T, pVp/4c^2 - T]]
#[inline(always)]
fn construct_2c_matrices<Mf64>(
    s_tilde_2c: &mut Matrix<f64>,
    h_tilde_2c: &mut Matrix<f64>,
    s: &Mf64,
    t: &Mf64,
    v: &Mf64,
    pvp: &Mf64,
    inv_two_c_sq: f64,
    inv_four_c_sq: f64,
)
where
    Mf64: MatrixMath<f64> + Factor,
{
    let n = s.cols();
    for j in 0..n {
        for i in 0..n {
            // Metric: large component is the plain overlap, small component
            // is the kinetically-balanced overlap.
            s_tilde_2c[(i, j)] = s[(i, j)];
            s_tilde_2c[(n + i, n + j)] = t[(i, j)] * inv_two_c_sq;
            // Modified-Dirac "Fock" matrix.
            h_tilde_2c[(i, j)] = v[(i, j)];
            h_tilde_2c[(n + i, j)] = t[(i, j)];
            h_tilde_2c[(i, n + j)] = t[(i, j)];
            h_tilde_2c[(n + i, n + j)] = pvp[(i, j)] * inv_four_c_sq - t[(i, j)];
        }
    }
}

/// Solves the augmented 2n x 2n modified-Dirac generalized eigenvalue
/// problem `h_tilde_2c C = s_tilde_2c C epsilon`, and extracts the
/// kinetic-balance X-matrix (X = small-component block * large-component
/// block^-1) from its positive-energy (electronic) eigenvectors.
///
/// The eigenvalues are in ascending order, so the positive-energy (electronic) solutions are columns
/// n..2n of the eigenvector matrix.
#[inline(always)]
fn construct_x_matrix<Mf64>(
    h_tilde_2c: Mf64,
    s_tilde_2c: Mf64,
    n: usize,
) -> Matrix<f64>
where
    Mf64: MatrixMath<f64> + Factor,
{
    let mut large_block = Matrix::<f64>::new(n, n);
    let mut small_block = Matrix::<f64>::new(n, n);

    let (_eigenvalues, eigenvectors) = h_tilde_2c.sygv(&s_tilde_2c, 1);
    {
        let mut k = 0;
        let large_block_data = large_block.get_data_mut();
        let small_block_data = small_block.get_data_mut();
        for j in n..2*n {
            for i in 0..n {
                large_block_data[k] = eigenvectors[(i, j)];
                small_block_data[k] = eigenvectors[(n + i, j)];
                k += 1;
            }
        }
    }
    let large_block_inv = large_block.invert();
    small_block.matmul(&large_block_inv)
}

/// Renormalizes S and T by folding in the X-matrix, returning
/// `(S + X^T T X / 2c^2, X^T T + T X - X^T T X)` -- the kinetically-balanced
/// overlap and kinetic-energy-like matrices used to build the final
/// large-component renormalization in
/// `assemble_x2c_one_electron_hamiltonian`.
#[inline(always)]
fn x_project_s_and_t<Mf64>(
    s: &Mf64,
    t: &Mf64,
    x_matrix: &Matrix<f64>,
    inv_two_c_sq: f64,
    n: usize,
) -> (Matrix<f64>, Matrix<f64>)
where
    Mf64: MatrixMath<f64> + Factor,
{
    let xt_t = x_matrix.view_transposed().matmul(t);
    let t_x = t.matmul(x_matrix);
    let xt_t_x = xt_t.matmul(x_matrix);

    let mut s_renorm = Matrix::<f64>::new(n, n);
    {
        // s_renorm = S + X^T T X / 2c^2
        let s_renorm_data = s_renorm.get_data_mut();
        let s_data = s.get_data();
        let xt_t_x_data = xt_t_x.get_data();
        for (out, (s_val, xt_t_x_val)) in s_renorm_data.iter_mut().zip(s_data.iter().zip(xt_t_x_data.iter())) {
            *out = *s_val + inv_two_c_sq * *xt_t_x_val;
        }
    }

    let mut t_renorm = Matrix::<f64>::new(n, n);
    {
        // t_renorm = X^T T + T X - X^T T X
        let t_renorm_data = t_renorm.get_data_mut();
        let xt_t_data = xt_t.get_data();
        let t_x_data = t_x.get_data();
        let xt_t_x_data = xt_t_x.get_data();
        for (out, (xt_t_val, (t_x_val, xt_t_x_val))) in
            t_renorm_data.iter_mut().zip(xt_t_data.iter().zip(t_x_data.iter().zip(xt_t_x_data.iter())))
        {
            *out = *xt_t_val + *t_x_val - *xt_t_x_val;
        }
    }

    (s_renorm, t_renorm)
}
