// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Basis shells of a contraction kind: `BasisShell<Contracted>` (a full `(n_prim, n_cont)` contraction matrix, several
//! AOs sharing one set of primitives), `BasisShell<Uncontracted>` (a block of exponents, each its own AO, implicit unit
//! coefficient) and `BasisShell<Segmented>` (one contracted AO per shell, primitives restricted to its own nonzero
//! coefficients). Every input basis is first parsed into contracted shells and `reduce`d to the other kinds.

use std::collections::HashMap;
use std::rc::Rc;

use arrays::matrix::*;
use arrays::vector::VectorMut;
use arrays::array3d::*;
use arrays::array4d::*;
use cart2sph;
use arrays::vector_math::vector_subtract_3d;
use crate::gto_products::gaussian_product_centre;

use crate::BasisInfo;
use crate::basis_parser::InputBasisShell;
use crate::gto_products;
use crate::contraction::{Contracted, Segmented, ShellKind, Uncontracted};
use crate::shell_pairs;

/// The angular components of a shell: Cartesian or real spherical harmonics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarmonicType {
    Cartesian,
    Spherical,
}

impl HarmonicType {
    /// Cartesian for `lqn <= 1`, spherical otherwise: the default, and the type s and p shells always get.
    pub fn default_for_lqn(lqn: usize) -> Self {
        if lqn <= 1 {
            HarmonicType::Cartesian
        } else {
            HarmonicType::Spherical
        }
    }
}

/// How the basis is contracted: generally, segmented, or not at all (every primitive its own function).
#[derive(Clone, Copy, Debug)]
pub enum ContractionType {
    General,
    Segmented,
    Primitive,
}

const AEDELSTEN_CONTRACTION_GENERAL: i32 = 0;
const AEDELSTEN_CONTRACTION_SEGMENTED: i32 = 1;
const AEDELSTEN_CONTRACTION_PRIMITIVE: i32 = 2;
impl ContractionType {
    /// The type of its C integer (0 general, 1 segmented, 2 primitive).
    pub fn from_c_int(value: i32) -> Result<Self, String> {
        match value {
            AEDELSTEN_CONTRACTION_GENERAL => Ok(ContractionType::General),
            AEDELSTEN_CONTRACTION_SEGMENTED => Ok(ContractionType::Segmented),
            AEDELSTEN_CONTRACTION_PRIMITIVE => Ok(ContractionType::Primitive),
            _ => Err(format!("Invalid contraction type: {}", value)),
        }
    }

    /// The type's C integer.
    pub fn get_c_int(&self) -> i32 {
        match self {
            ContractionType::General => AEDELSTEN_CONTRACTION_GENERAL,
            ContractionType::Segmented => AEDELSTEN_CONTRACTION_SEGMENTED,
            ContractionType::Primitive => AEDELSTEN_CONTRACTION_PRIMITIVE,
        }
    }

    /// The names the C interface's integer queries accept, with their values.
    pub const NAME_TABLE: &'static [(&'static str, i32)] = &[
        ("contraction_type_general", AEDELSTEN_CONTRACTION_GENERAL),
        ("contraction_type_segmented", AEDELSTEN_CONTRACTION_SEGMENTED),
        ("contraction_type_primitive", AEDELSTEN_CONTRACTION_PRIMITIVE),
    ];
}

/// One (centre, l) block of primitive exponents and what its contraction kind stores: the coefficient matrix
/// (`Contracted`) or nothing (`Uncontracted`, every exponent its own AO, in exponent order).
#[derive(Clone, Debug)]
pub struct BasisShell<S: ShellKind = Contracted> {
    lqn: usize,
    harmonic_type: HarmonicType,
    exp: Rc<[f64]>,
    coeffs: S::Coefficients,
    basis_index: usize, // The shell's first AO.
    centre_index: usize,
    position: [f64; 3],
}



// --- BasisShell<Contracted> ---

impl BasisShell<Contracted> {
    /// The shell of the parsed block `input_basis` on centre `centre_index` at `position`; its first AO is set later
    /// (`fix_basis_indices`).
    pub fn new(
        input_basis: &InputBasisShell,
        harmonic_type: HarmonicType,
        centre_index: usize,
        position: [f64; 3],
    ) -> Self {
        let lqn = input_basis.get_lqn();
        let exp = input_basis.get_exponents().to_vec().into_boxed_slice();
        let coeffs = input_basis.get_coefficients().clone();
        BasisShell::<Contracted> {
            lqn,
            harmonic_type,
            exp: Rc::from(exp),
            coeffs: Rc::new(coeffs),
            basis_index: 0,
            centre_index,
            position: position,
        }
    }

    /// `shell` as a general contraction: the same functions in the same AO order (its `(n_prim, n_cont)` coefficient
    /// matrix, the identity for an uncontracted shell), on the same centre with the same first AO.
    pub fn from_shell<S: crate::contraction::ShellKind>(shell: &BasisShell<S>) -> Self {
        let n_prim = shell.exp.len();
        // Copied: the general contraction owns its coefficient matrix, and the other kinds store none in that form.
        let coefficients = match S::amplitude_coefficients(&shell.coeffs, n_prim) {
            Some(matrix) => {
                let mut owned = Matrix::<f64>::new(matrix.rows(), matrix.cols());
                for (dst_col, src_col) in owned.iter_col_mut().zip(matrix.iter_col()) {
                    dst_col.copy_from_slice(src_col);
                }
                owned
            }
            None => {
                let mut identity = Matrix::<f64>::new(n_prim, n_prim);
                identity.view_diagonal_mut().copy_from(&vec![1.0; n_prim][..]);
                identity
            }
        };
        BasisShell::<Contracted> {
            lqn: shell.lqn,
            harmonic_type: shell.harmonic_type,
            exp: Rc::clone(&shell.exp),
            coeffs: Rc::new(coefficients),
            basis_index: shell.basis_index,
            centre_index: shell.centre_index,
            position: shell.position,
        }
    }

    /// The pair of this shell and `other` with its contraction matrix. `single_centre` (the shell with itself): the
    /// products `i <= j`, `j` outer. Otherwise every product, or with `overlap_threshold` those whose overlap times
    /// the largest coefficient product reaches it; `None` if none is left.
    pub(crate) fn form_pair(
        &self,
        other: &Self,
        single_centre: bool,
        overlap_threshold: Option<f64>
    ) -> Option<shell_pairs::ShellPair<crate::contraction::Contracted>> {
        let lqn_a = self.lqn;
        let lqn_b = other.lqn;
        let n_prim_a = self.exp.len();
        let n_prim_b = other.exp.len();
        let n_cont_a = self.coeffs.cols();
        let n_cont_b = other.coeffs.cols();
        let harmonic_a = self.harmonic_type.clone();
        let harmonic_b = other.harmonic_type.clone();
        let idx_a = self.basis_index;
        let idx_b = other.basis_index;

        let pos_a = self.get_position();
        let pos_b = other.get_position();

        let dist_ab = vector_subtract_3d(&other.get_position(), &self.get_position());
        let x = dist_ab[0];
        let y = dist_ab[1];
        let z = dist_ab[2];

        let mut products = Vec::<gto_products::GTOProduct>::with_capacity(n_prim_a * n_prim_b);

        let (products, contraction_matrix) = if single_centre {
            // The products i <= j, j outer (shell_pairs::contracted_pair_slot).
            for (j, &b) in other.get_exponents().iter().enumerate() {
                for &a in self.get_exponents().iter().take(j + 1) {
                    let p = a + b;
                    let pos_ab = gaussian_product_centre(&pos_a, a, &pos_b, b, p);
                    products.push(gto_products::GTOProduct::new(a, b, pos_ab));
                }
            }
            (products.into_boxed_slice(), self.pair_contraction_matrix_single_centre())
        } else {
            // With a threshold, the products that pass it and the contraction matrix's rows for them.
            if let Some(threshold) = overlap_threshold {
                let non_redundant_idx= self.compute_non_redundant_primitives(other, threshold);
                for &(i, j) in non_redundant_idx.iter() {
                    let a = self.get_exponents()[i];
                    let b = other.get_exponents()[j];
                    let p = a + b;
                    let pos_ab = gaussian_product_centre(&pos_a, a, &pos_b, b, p);
                    products.push(gto_products::GTOProduct::new(a, b, pos_ab));
                }
                (products.into_boxed_slice(), self.pair_contraction_matrix_screened(other, &non_redundant_idx))
            } else {
                for &b in other.get_exponents().iter() {
                    for &a in self.get_exponents().iter() {
                        let p = a + b;
                        let pos_ab = gaussian_product_centre(&pos_a, a, &pos_b, b, p);
                        products.push(gto_products::GTOProduct::new(a, b, pos_ab));
                    }
                }
                (products.into_boxed_slice(), self.pair_contraction_matrix(other) )
            }
        };

        if products.is_empty() {
            return None;
        }

        Some(shell_pairs::ShellPair::<crate::contraction::Contracted>::new(
            lqn_a,
            lqn_b,
            n_prim_a,
            n_prim_b,
            n_cont_a,
            n_cont_b,
            harmonic_a,
            harmonic_b,
            idx_a,
            idx_b,
            self.get_centre_index(),
            other.get_centre_index(),
            x,
            y,
            z,
            single_centre,
            contraction_matrix,
            products,
        ))
    }

    // The products (i, j) whose overlap exp(-mu r^2) times the largest coefficient product |c_ik c_jl| reaches
    // `threshold`.
    fn compute_non_redundant_primitives(
        &self,
        other: &Self,
        threshold: f64,
    ) -> Vec<(usize, usize)> {
        let n_prim_a = self.exp.len() as usize;
        let coeffs_a = self.get_coefficients();
        let n_cont_a = coeffs_a.cols();
        let n_prim_b = other.exp.len() as usize;
        let coeffs_b = other.get_coefficients();
        let n_cont_b = coeffs_b.cols();
        let dist_ab = vector_subtract_3d(&other.get_position(), &self.get_position());
        let r2 = dist_ab[0] * dist_ab[0] + dist_ab[1] * dist_ab[1] + dist_ab[2] * dist_ab[2];
        let mut non_redundant_idx = Vec::<(usize, usize)>::with_capacity(n_prim_a * n_prim_b);
        for (j, &b) in other.get_exponents().iter().enumerate() {
            for (i, &a) in self.get_exponents().iter().enumerate() {
                let mut max_cc_ab = 0.0;
                for l in 0..n_cont_a {
                    let c_a = coeffs_a[(i, l)];
                    for k in 0..n_cont_b {
                        let cc = coeffs_b[(j, k)] * c_a;
                        if cc.abs() > max_cc_ab {
                            max_cc_ab = cc.abs();
                        }
                    }
                }
                let ab = a * b / (a + b);
                let overlap = max_cc_ab * (-ab * r2).exp();
                if overlap.abs() >= threshold {
                    non_redundant_idx.push((i, j));
                }
            }
        }
        non_redundant_idx
    }

    // The contraction matrix of every product (`pair_contraction_matrix`).
    fn pair_contraction_matrix(&self, other: &Self) -> Matrix<f64> {
        pair_contraction_matrix(self.get_coefficients(), other.get_coefficients())
    }

    // The contraction matrix of the products `non_redundant_idx`: rows those products, columns the contracted pairs.
    fn pair_contraction_matrix_screened(
        &self,
        other: &Self,
        non_redundant_idx: &Vec<(usize, usize)>,
    ) -> Matrix<f64> {
        let coeffs_a = self.get_coefficients();
        let coeffs_b = other.get_coefficients();
        let n_cont_a = coeffs_a.cols();
        let n_cont_b = coeffs_b.cols();
        let non_redudant_prims = non_redundant_idx.len();
        if non_redudant_prims == 0 {
            return Matrix::<f64>::new(0, 0);
        }
        let mut large_contraction_matrix =
            Array3D::<f64>::new(non_redudant_prims, n_cont_a, n_cont_b);

        for (mut l, col_b) in large_contraction_matrix.iter_mut().zip(coeffs_b.iter_col()) {
            for (k, col_a) in l.iter_mut().zip(coeffs_a.iter_col()) {
                for (index, &(i, j)) in non_redundant_idx.iter().enumerate() {
                    let a = col_a[i];
                    let b = col_b[j];
                    k[index] = a * b;
                }
            }
        }

        large_contraction_matrix.into_matrix((1, 2))
    }

    // The contraction matrix of the shell with itself.
    fn pair_contraction_matrix_single_centre(&self) -> Matrix<f64> {
        let coeffs_a = self.get_coefficients();
        let n_exps_a = coeffs_a.rows();
        let n_cont_a = coeffs_a.cols();

        // Rows: the stored primitive products (i <= j, j outer); columns: the unique contracted pairs (k <= l,
        // l outer). A product with i != j stands for both orders, which give the same block.
        let mut large_contraction_matrix =
            Matrix::<f64>::new(n_exps_a * (n_exps_a + 1) / 2, n_cont_a * (n_cont_a + 1) / 2);

        let mut column = 0;
        for l in 0..n_cont_a {
            for k in 0..=l {
                let mut index = 0;
                for j in 0..n_exps_a {
                    for i in 0..=j {
                        let val1 = coeffs_a[(i, k)] * coeffs_a[(j, l)];
                        let val2 = coeffs_a[(j, k)] * coeffs_a[(i, l)];
                        large_contraction_matrix[(index, column)] = if i == j { val1 } else { val1 + val2 };
                        index += 1;
                    }
                }
                column += 1;
            }
        }

        large_contraction_matrix
    }
}

/// The `(n_prim_a n_prim_b, n_cont_a n_cont_b)` product of two shells' coefficient matrices (every primitive pair to
/// every contracted pair).
fn pair_contraction_matrix(coeffs_a: &Matrix<f64>, coeffs_b: &Matrix<f64>) -> Matrix<f64> {
    let n_exps_a = coeffs_a.rows();
    let n_cont_a = coeffs_a.cols();
    let n_exps_b = coeffs_b.rows();
    let n_cont_b = coeffs_b.cols();

    let mut large_contraction_matrix =
        Array4D::<f64>::new(n_exps_a, n_exps_b, n_cont_a, n_cont_b);

    for (mut l, col_b) in large_contraction_matrix.iter_mut().zip(coeffs_b.iter_col()) {
        for (mut k, col_a) in l.iter_mut().zip(coeffs_a.iter_col()) {
            for (j, b) in k.iter_col_mut().zip(col_b.iter()) {
                for (i, a) in j.iter_mut().zip(col_a.iter()) {
                    *i = a * b;
                }
            }
        }
    }

    large_contraction_matrix.into_matrix((2, 2))
}

// Every AO of a shell at (x, y, z), see BasisShell::compute_ao_amplitude: the normalized Cartesian primitives
// N(a, lx, ly, lz) (x-Ax)^lx (y-Ay)^ly (z-Az)^lz exp(-a r^2), contracted by `coefficients` (`None`: every primitive its
// own AO), transformed to spherical components if the shell is spherical.
fn ao_amplitudes<S: ShellKind>(shell: &BasisShell<S>, coefficients: Option<MatrixView<'_, f64>>, x: f64, y: f64, z: f64) -> Box<[f64]> {
    let gto_pos = shell.position;
    let px = x - gto_pos[0];
    let py = y - gto_pos[1];
    let pz = z - gto_pos[2];
    let r2 = px * px + py * py + pz * pz;
    // (n_prim, n_cart).
    let lqn = shell.lqn as usize;
    let cart_shell_size = cart2sph::compute_cartesian_degeneracy(lqn);
    let cart_comps = cart2sph::get_cartesian_quantum_numbers_m(lqn);
    let mut primitive_amplitudes = Matrix::<f64>::new(shell.exp.len(), cart_shell_size);
    for (amplitudes, comp) in primitive_amplitudes.iter_col_mut().zip(cart_comps.iter_col()) {
        let prefactor = (px.powi(comp[0] as i32)
            * py.powi(comp[1] as i32)
            * pz.powi(comp[2] as i32)) as f64;
        for (amp, &exp) in amplitudes.iter_mut().zip(shell.exp.iter()) {
            let norm_exp = cart2sph::compute_cart_gaussian_norm(
                exp,
                comp[0] as usize,
                comp[1] as usize,
                comp[2] as usize,
            );
            *amp = prefactor * norm_exp * (-exp * r2).exp();
        }
    }
    // (n_cont, n_cart).
    let cartesian_amplitudes = match coefficients {
        Some(coefficients) => coefficients.view_transposed().matmul(&primitive_amplitudes),
        None => primitive_amplitudes,
    };
    // (n_comp, n_cont).
    let final_amplitudes = match shell.harmonic_type {
        HarmonicType::Cartesian => {
            cartesian_amplitudes.transpose()
        }
        HarmonicType::Spherical => {
            let sph_transform = cart2sph::compute_cartesian_to_spherical_matrix(lqn);
            cartesian_amplitudes.matmul(&sph_transform).transpose()
        }
    };
    // One block of components per contracted AO.
    final_amplitudes.into_data()
}


// Same as ao_amplitudes, for the gradient with respect to the field point, see
// BasisShell::compute_ao_amplitude_gradient.
fn ao_amplitude_gradients<S: ShellKind>(shell: &BasisShell<S>, coefficients: Option<MatrixView<'_, f64>>, x: f64, y: f64, z: f64) -> [Box<[f64]>; 3] {
    let gto_pos = shell.position;
    let px = x - gto_pos[0];
    let py = y - gto_pos[1];
    let pz = z - gto_pos[2];

    let lqn = shell.lqn as usize;
    let cart_shell_size = cart2sph::compute_cartesian_degeneracy(lqn);
    let cart_comps = cart2sph::get_cartesian_quantum_numbers_m(lqn);
    let n_prim = shell.exp.len();

    // (n_prim, n_cart) per direction.
    let mut primitive_grad_x = Matrix::<f64>::new(n_prim, cart_shell_size);
    let mut primitive_grad_y = Matrix::<f64>::new(n_prim, cart_shell_size);
    let mut primitive_grad_z = Matrix::<f64>::new(n_prim, cart_shell_size);
    for c in 0..cart_shell_size {
        let comp = cart_comps.get_col(c);
        let (lx, ly, lz) = (comp[0], comp[1], comp[2]);
        for (p, &exp) in shell.exp.iter().enumerate() {
            let (_, dx, dy, dz) = cart2sph::compute_cart_gaussian_value_and_gradient(px, py, pz, lx, ly, lz, exp);
            let norm_exp = cart2sph::compute_cart_gaussian_norm(exp, lx, ly, lz);
            primitive_grad_x[(p, c)] = dx * norm_exp;
            primitive_grad_y[(p, c)] = dy * norm_exp;
            primitive_grad_z[(p, c)] = dz * norm_exp;
        }
    }

    let ct = coefficients.as_ref().map(|coefficients| coefficients.transpose());
    let sph_transform = match shell.harmonic_type {
        HarmonicType::Spherical => Some(cart2sph::compute_cartesian_to_spherical_matrix(lqn)),
        HarmonicType::Cartesian => None,
    };
    let contract = |primitive_grad: &Matrix<f64>| -> Box<[f64]> {
        let contracted;
        let cartesian = match &ct {
            Some(ct) => {
                contracted = ct.matmul(primitive_grad);
                &contracted
            }
            None => primitive_grad,
        };
        match &sph_transform {
            Some(t) => cartesian.matmul(t).transpose().into_data(),
            None => cartesian.transpose().into_data(),
        }
    };

    [contract(&primitive_grad_x), contract(&primitive_grad_y), contract(&primitive_grad_z)]
}

impl<S: ShellKind> BasisShell<S> {
    /// The angular momentum.
    pub fn get_lqn(&self) -> usize {
        self.lqn
    }
    /// Cartesian or spherical.
    pub fn get_harmonic_type(&self) -> HarmonicType {
        self.harmonic_type
    }
    /// Angular components, Cartesian or spherical per the harmonic type.
    pub fn get_degeneracy(&self) -> usize {
        match self.harmonic_type {
            HarmonicType::Cartesian => cart2sph::compute_cartesian_degeneracy(self.lqn),
            HarmonicType::Spherical => cart2sph::compute_spherical_degeneracy(self.lqn),
        }
    }
    /// The first AO of the shell.
    pub fn get_basis_index(&self) -> usize {
        self.basis_index
    }
    /// The centre the shell is on.
    pub fn get_centre_index(&self) -> usize {
        self.centre_index
    }
    /// The centre's position.
    pub fn get_position(&self) -> [f64; 3] {
        self.position
    }
    /// Sets the first AO.
    pub fn set_basis_index(&mut self, index: usize) {
        self.basis_index = index;
    }
    /// Sets the centre.
    pub fn set_centre_index(&mut self, index: usize) {
        self.centre_index = index;
    }
    /// Sets the position.
    pub fn set_position(&mut self, position: [f64; 3]) {
        self.position = position;
    }
    /// The number of primitives.
    pub fn get_n_prim(&self) -> usize {
        self.exp.len()
    }
    /// The primitive exponents.
    pub fn get_exponents(&self) -> &[f64] {
        &self.exp
    }
    /// The number of contracted functions.
    pub fn get_n_cont(&self) -> usize {
        S::n_cont(&self.coeffs, self.exp.len())
    }
    /// What the kind stores for the contraction.
    pub fn get_coefficients(&self) -> &S::Coefficients {
        &self.coeffs
    }
    /// (exponent, coefficient) of every primitive of contracted AO `cont`, the coefficient applying to an
    /// independently normalized primitive (the convention of MOLCAS's PRIMITIVES); zero coefficients are left out.
    pub fn contracted_primitives(&self, cont: usize) -> impl Iterator<Item = (f64, f64)> + '_ {
        S::contracted_primitives(&self.exp, &self.coeffs, cont)
    }
    /// The pair of this shell and `other` (`ShellKind::form_shell_pair`).
    pub fn form_shell_pair(&self, other: &Self, single_centre: bool, overlap_threshold: Option<f64>) -> Option<shell_pairs::ShellPair<S>> {
        S::form_shell_pair(self, other, single_centre, overlap_threshold)
    }
    /// The shells of this kind for a list of contracted (input) shells.
    pub fn reduce(general_shells: &[BasisShell<Contracted>]) -> Vec<Self> {
        S::reduce(general_shells)
    }
    /// Every AO of the shell at (x, y, z): one block of angular components (Cartesian or spherical) per
    /// contracted function, the AO order of the rest of the code.
    pub fn compute_ao_amplitude(&self, x: f64, y: f64, z: f64) -> Box<[f64]> {
        ao_amplitudes(self, S::amplitude_coefficients(&self.coeffs, self.exp.len()), x, y, z)
    }
    /// The gradient of `compute_ao_amplitude` with respect to the field point: d/dx, d/dy, d/dz blocks of the same
    /// layout.
    pub fn compute_ao_amplitude_gradient(&self, x: f64, y: f64, z: f64) -> [Box<[f64]>; 3] {
        ao_amplitude_gradients(self, S::amplitude_coefficients(&self.coeffs, self.exp.len()), x, y, z)
    }
    /// The `(n_prim, n_cont)` coefficient matrix, `None` when uncontracted.
    #[inline(always)]
    pub fn coefficient_matrix(&self) -> Option<&Matrix<f64>> {
        S::coefficient_matrix(&self.coeffs)
    }
    /// The per-primitive coefficients of a segmented shell, `None` otherwise.
    #[inline(always)]
    pub fn summed_coefficients(&self) -> Option<&[f64]> {
        S::summed_coefficients(&self.coeffs)
    }
}

/// A shell of `exponents` contracted by `coefficients` (`(n_prim, n_cont)`) on centre `centre_index` at `position`,
/// for tests.
#[cfg(test)]
impl BasisShell<Contracted> {
    pub(crate) fn for_test(lqn: usize, harmonic_type: HarmonicType, exponents: &[f64], coefficients: Matrix<f64>, centre_index: usize, position: [f64; 3]) -> Self {
        BasisShell::<Contracted> {
            lqn,
            harmonic_type,
            exp: Rc::from(exponents.to_vec().into_boxed_slice()),
            coeffs: Rc::new(coefficients),
            basis_index: 0,
            centre_index,
            position,
        }
    }
}

// --- BasisShell<Uncontracted> ---

impl BasisShell<Uncontracted> {
    /// The uncontracted (`Identity`) pair of this shell's and `other`'s primitives: every primitive product is its
    /// own pair, stored in the order of `shell_pairs::contracted_pair_slot` (a same-shell pair: i <= j, j outer).
    /// Products whose bare overlap is below `overlap_threshold` are left out (the criterion of true primitive
    /// pairs; a same-shell pair has none); `None` if none is left.
    pub(crate) fn form_pair(
        &self,
        other: &Self,
        single_centre: bool,
        overlap_threshold: Option<f64>,
    ) -> Option<shell_pairs::ShellPair<Uncontracted>> {
        let (n_prim_a, n_prim_b) = (self.exp.len(), other.exp.len());
        let pos_a = self.get_position();
        let pos_b = other.get_position();
        let dist_ab = vector_subtract_3d(&pos_b, &pos_a);
        let r2 = dist_ab[0] * dist_ab[0] + dist_ab[1] * dist_ab[1] + dist_ab[2] * dist_ab[2];
        let mut products = Vec::<gto_products::GTOProduct>::with_capacity(n_prim_a * n_prim_b);
        let mut placement = vec![u32::MAX; if single_centre { 0 } else { n_prim_a * n_prim_b }];
        let mut screened = false;
        for (j, &b) in other.get_exponents().iter().enumerate() {
            let a_end = if single_centre { j + 1 } else { n_prim_a };
            for (i, &a) in self.get_exponents().iter().take(a_end).enumerate() {
                if !single_centre {
                    if overlap_threshold.is_some_and(|threshold| (-(a * b / (a + b)) * r2).exp() < threshold) {
                        screened = true;
                        continue;
                    }
                    placement[i + n_prim_a * j] = products.len() as u32;
                }
                let p = a + b;
                products.push(gto_products::GTOProduct::new(a, b, gaussian_product_centre(&pos_a, a, &pos_b, b, p)));
            }
        }
        if products.is_empty() {
            return None;
        }
        let placement = crate::contraction::NoContraction { placement: screened.then(|| placement.into_boxed_slice()) };
        Some(shell_pairs::ShellPair::<Uncontracted>::new(
            self.lqn,
            other.lqn,
            n_prim_a,
            n_prim_b,
            n_prim_a,
            n_prim_b,
            self.harmonic_type,
            other.harmonic_type,
            self.basis_index,
            other.basis_index,
            self.get_centre_index(),
            other.get_centre_index(),
            dist_ab[0],
            dist_ab[1],
            dist_ab[2],
            single_centre,
            placement,
            products.into_boxed_slice(),
        ))
    }

}

impl BasisShell<Uncontracted> {
    /// Every (centre, l) block of exponents stays one shell; the contraction is dropped.
    pub(crate) fn blocks_of(general_shells: &[BasisShell<Contracted>]) -> Vec<Self> {
        general_shells.iter().map(|shell| BasisShell {
            lqn: shell.lqn,
            harmonic_type: shell.harmonic_type,
            exp: shell.exp.clone(),
            coeffs: (),
            basis_index: 0,
            centre_index: shell.centre_index,
            position: shell.position,
        }).collect()
    }
}

// --- BasisShell<Segmented> ---

impl BasisShell<Segmented> {
    /// One segmented shell per contracted function of `general_shell`, of the primitives whose coefficient is not
    /// exactly zero.
    pub(crate) fn segments_of(general_shell: &BasisShell<Contracted>) -> Box<[Self]> {
        let n_exp = general_shell.exp.len();
        let n_cont = general_shell.coeffs.cols();
        let mut segmented_shells = Vec::with_capacity(n_cont);
        for coeffs in general_shell.coeffs.iter_col() {
            let mut index_list = Vec::with_capacity(n_exp);
            for (i, coeff) in coeffs.iter().enumerate() {
                if *coeff != 0.0 {
                    index_list.push(i);
                }
            }
            let exp = index_list.iter().map(|&i| general_shell.exp[i]).collect::<Vec<f64>>().into_boxed_slice();
            let coeffs = index_list.iter().map(|&i| coeffs[i]).collect::<Vec<f64>>().into_boxed_slice();
            segmented_shells.push(BasisShell::<Segmented> {
                lqn: general_shell.lqn,
                harmonic_type: general_shell.harmonic_type.clone(),
                exp: Rc::from(exp),
                coeffs: Rc::from(coeffs),
                basis_index: 0,
                centre_index: general_shell.centre_index,
                position: general_shell.position,
            });
        }
        segmented_shells.into_boxed_slice()
    }

    /// The pair of this shell and `other`, each product carrying its coefficient product. `single_centre` (the shell
    /// with itself): the products `i >= j`, the off-diagonal ones with twice the coefficient. Otherwise every product
    /// whose overlap times coefficient product reaches `overlap_threshold`; `None` if none is left.
    pub(crate) fn form_pair(
        &self,
        other: &Self,
        single_centre: bool,
        overlap_threshold: Option<f64>,
    ) -> Option<shell_pairs::ShellPair<crate::contraction::Segmented>> {
        let lqn_a = self.lqn;
        let lqn_b = other.lqn;
        let n_prim_a = self.exp.len();
        let n_prim_b = other.exp.len();
        let harmonic_a = self.harmonic_type.clone();
        let harmonic_b = other.harmonic_type.clone();
        let idx_a = self.basis_index;
        let idx_b = other.basis_index;

        let pos_a = self.get_position();
        let pos_b = other.get_position();

        let dist_ab = vector_subtract_3d(&pos_b, &pos_a);
        let x = dist_ab[0];
        let y = dist_ab[1];
        let z = dist_ab[2];
        let r2 = x * x + y * y + z * z;

        let mut products: Vec<gto_products::CGTOProduct> = Vec::with_capacity(n_prim_a * n_prim_b);
        if !single_centre {
            for (&b, &cb) in other
                .get_exponents()
                .iter()
                .zip(other.get_coefficients().iter())
            {
                for (&a, &ca) in self
                    .get_exponents()
                    .iter()
                    .zip(self.get_coefficients().iter())
                {
                    let ab = a * b / (a + b);
                    let overlap = ca * cb * (-ab * r2).exp();
                    if let Some(threshold) = overlap_threshold {
                        if overlap.abs() < threshold {
                            continue;
                        }
                    }
                    let p = a + b;
                    let pos_ab = gaussian_product_centre(&pos_a, a, &pos_b, b, p);
                    let product = gto_products::CGTOProduct::new(a, b, ca * cb, pos_ab);
                    products.push(product);
                }
            }
        } else {
            for (j, (&b, &cb)) in other
                .get_exponents()
                .iter()
                .zip(other.get_coefficients().iter())
                .enumerate()
            {
                let p = 2.0 * b;
                let pos_ab = gaussian_product_centre(&pos_b, b, &pos_b, b, p);
                let product = gto_products::CGTOProduct::new(b, b, cb * cb, pos_ab);
                products.push(product);
                // Off-diagonal products twice, for both orders.
                for (&a, &ca) in self
                    .get_exponents()
                    .iter()
                    .zip(self.get_coefficients().iter())
                    .skip(j + 1)
                {
                    let p = a + b;
                    let pos_ab = gaussian_product_centre(&pos_a, a, &pos_b, b, p);
                    let product = gto_products::CGTOProduct::new(a, b, 2.0 * ca * cb, pos_ab);
                    products.push(product);
                }
            }
        }

        if products.is_empty() {
            return None;
        }

        Some(shell_pairs::ShellPair::<crate::contraction::Segmented>::new(
            lqn_a,
            lqn_b,
            n_prim_a,
            n_prim_b,
            1,
            1,
            harmonic_a,
            harmonic_b,
            idx_a,
            idx_b,
            self.get_centre_index(),
            other.get_centre_index(),
            x,
            y,
            z,
            single_centre,
            crate::contraction::SummedPair,
            products.into_boxed_slice(),
        ))
    }
}

/// A primitive count that does not match the basis while decontracting.
#[derive(Debug)]
pub struct BasisError;

impl std::fmt::Display for BasisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "primitive count mismatch while decontracting a basis")
    }
}

impl std::error::Error for BasisError {}

// The contraction matrix of the basis, rows the primitive AOs, columns the contracted AOs. For a shell with n_prim
// exponents, n_cont contracted functions and sph components:
//   primitive  (p, ml) -> row  prim_base + p * sph + ml
//   contracted (c, ml) -> col  cont_base + c * sph + ml
//   coefficient         = coeffs[(p, c)]   (the same for every ml)
// An error if the shells' primitive count is not `basis_info`'s.
fn build_contraction_matrix(shells: &[BasisShell<Contracted>], basis_info: &BasisInfo) -> Result<Matrix<f64>, BasisError> {
    let n_prim: usize = shells.iter().map(|gto| gto.get_n_prim() * gto.get_degeneracy()).sum();
    if n_prim != basis_info.get_n_prim() {
        return Err(BasisError)
    }

    let mut cm = Matrix::<f64>::new(basis_info.get_n_prim(), basis_info.get_n_cont());
    let mut prim_base = 0usize;
    let mut cont_base = 0usize;
    for gto in shells.iter()  {
        let sph    = gto.get_degeneracy();
        let coeffs = gto.get_coefficients();
        let n_prim = coeffs.rows();
        let n_cont = coeffs.cols();
        for c in 0..n_cont {
            for ml in 0..sph {
                let col = cont_base + c * sph + ml;
                for p in 0..n_prim {
                    let row = prim_base + p * sph + ml;
                    cm[(row, col)] = coeffs[(p, c)];
                }
            }
        }
        prim_base += n_prim * sph;
        cont_base += n_cont * sph;
    }
    Ok(cm)
}

/// The uncontracted (centre, l) blocks of a general basis, and the contraction matrix (rows: their primitive
/// AOs, columns: the contracted AOs).
pub fn decontract_general_shells(
    shells: &[BasisShell<Contracted>],
    basis_info: &BasisInfo
) -> Result<(Box<[BasisShell<Uncontracted>]>, Matrix<f64>), BasisError> {
    let cm = build_contraction_matrix(shells, basis_info)?;
    let mut blocks = Contracted::primitive_blocks(shells);
    crate::fix_basis_indices(&mut blocks);
    Ok((blocks.into_boxed_slice(), cm))
}

/// The uncontracted (centre, l) blocks of a segmented basis, and the contraction matrix (rows: their primitive AOs,
/// columns: the contracted AOs): each block's segments folded into one contracted shell over the union of their
/// exponents, then decontracted.
pub fn decontract_segmented_shells(
    shells: &[BasisShell<Segmented>],
    basis_info: &BasisInfo,
) -> Result<(Box<[BasisShell<Uncontracted>]>, Matrix<f64>), BasisError> {
    let (general_shells, general_basis_info) = fold_segmented_shells(shells, basis_info)?;
    decontract_general_shells(&general_shells, &general_basis_info)
}

/// The uncontracted (centre, l) blocks of a segmented basis (`decontract_segmented_shells`'s), without the
/// contraction matrix: each block over the union of its segments' exponents, a primitive several segments share once.
pub(crate) fn segmented_primitive_blocks(shells: &[BasisShell<Segmented>]) -> Vec<BasisShell<Uncontracted>> {
    Contracted::primitive_blocks(&make_general_contraction(shells))
}

// The segmented shells folded into one contracted shell per (centre, l) block, and their `BasisInfo`; an error if
// the folded primitive count is not `basis_info`'s unique count.
fn fold_segmented_shells(
    shells: &[BasisShell<Segmented>],
    basis_info: &BasisInfo,
) -> Result<(Vec<BasisShell<Contracted>>, BasisInfo), BasisError> {
    let mut general_shells = make_general_contraction(shells);
    let general_basis_info = crate::fix_basis_indices(&mut general_shells);
    if general_basis_info.get_n_prim() != basis_info.get_n_prim_unique() {
        return Err(BasisError);
    }
    Ok((general_shells, general_basis_info))
}

// Per (centre, l) block: its degeneracy and the union of its shells' exponents, compared exactly, sorted descending.
pub(crate) fn group_unique_exponents<S: ShellKind>(
    shells: &[BasisShell<S>],
) -> HashMap<(usize, usize), (usize, Vec<f64>)> {
    let mut groups: HashMap<(usize, usize), (usize, Vec<f64>)> = HashMap::new();
    for shell in shells.iter() {
        let key = (shell.get_centre_index(), shell.get_lqn());
        let (_, exponents) = groups
            .entry(key)
            .or_insert_with(|| (shell.get_degeneracy(), Vec::new()));
        for &exp in shell.get_exponents() {
            if !exponents.contains(&exp) {
                exponents.push(exp);
            }
        }
    }
    for (_, exponents) in groups.values_mut() {
        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap());
    }
    groups
}

// One contracted shell per (centre, l) block of segmented shells, over the union of their exponents, one column per
// segment; sorted by centre, then l.
fn make_general_contraction(shells: &[BasisShell<Segmented>]) -> Vec<BasisShell<Contracted>> {
    let mut shell_groups: HashMap<(usize, usize), Vec<&BasisShell<Segmented>>> = HashMap::new();
    for shell in shells.iter() {
        shell_groups
            .entry((shell.get_centre_index(), shell.get_lqn()))
            .or_default()
            .push(shell);
    }
    let unique_exponents = group_unique_exponents(shells);

    let mut general_shells: Vec<BasisShell<Contracted>> = shell_groups
        .into_iter()
        .map(|(key, group)| {
            let (_, exponents) = unique_exponents.get(&key).expect("every shell_groups key has a matching group_unique_exponents entry");

            // Each coefficient in the row of its exponent's value.
            let mut coeffs = Matrix::<f64>::new(exponents.len(), group.len());
            for (col, shell) in coeffs.iter_col_mut().zip(group.iter()) {
                for (&exp, &cc) in shell.get_exponents().iter().zip(shell.get_coefficients().iter()) {
                    let row = exponents.iter().position(|&e| e == exp).unwrap();
                    col[row] = cc;
                }
            }

            let first = group[0];
            BasisShell::<Contracted> {
                lqn: first.get_lqn(),
                harmonic_type: first.get_harmonic_type(),
                exp: Rc::from(exponents.clone().into_boxed_slice()),
                coeffs: Rc::new(coeffs),
                basis_index: 0,
                centre_index: first.get_centre_index(),
                position: first.get_position(),
            }
        })
        .collect();

    general_shells.sort_by(|a, b| {
        a.get_centre_index().cmp(&b.get_centre_index())
            .then_with(|| a.get_lqn().cmp(&b.get_lqn()))
    });
    general_shells
}

#[cfg(test)]
mod ao_amplitude_gradient_tests {
    // compute_ao_amplitude_gradient against a central difference of compute_ao_amplitude, for the three shell kinds
    // and both harmonic types (p Cartesian, d spherical).
    use super::*;

    fn general_shell(lqn: usize, exponents: &[f64], coeffs: &[f64], harmonic_type: HarmonicType, position: [f64; 3]) -> BasisShell::<Contracted> {
        let n_prim = exponents.len();
        let coeff_matrix = Matrix::<f64>::new_from_vec(n_prim, 1, coeffs.to_vec().into_boxed_slice());
        BasisShell::<Contracted> {
            lqn,
            harmonic_type,
            exp: Rc::from(exponents.to_vec().into_boxed_slice()),
            coeffs: Rc::new(coeff_matrix),
            basis_index: 0,
            centre_index: 0,
            position,
        }
    }

    fn test_points() -> [[f64; 3]; 4] {
        [
            [0.7, 0.5, 0.9],
            [-0.6, 1.1, -0.3],
            [0.0, 0.8, -0.5], // x = 0 relative to a shell at the origin
            [1.5, -1.2, 0.4],
        ]
    }

    fn check_gradient_matches_finite_difference<S: ShellKind>(shell: &BasisShell<S>, point: [f64; 3]) {
        let h = 1e-5;
        let [x, y, z] = point;
        let analytic = shell.compute_ao_amplitude_gradient(x, y, z);

        let central_difference = |dx: f64, dy: f64, dz: f64| -> Box<[f64]> {
            let plus = shell.compute_ao_amplitude(x + dx, y + dy, z + dz);
            let minus = shell.compute_ao_amplitude(x - dx, y - dy, z - dz);
            plus.iter().zip(minus.iter()).map(|(p, m)| (p - m) / (2.0 * h)).collect()
        };
        let expected = [
            central_difference(h, 0.0, 0.0),
            central_difference(0.0, h, 0.0),
            central_difference(0.0, 0.0, h),
        ];

        for (direction, (got, want)) in analytic.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.len(), want.len());
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < 1e-6,
                    "gradient mismatch at point {point:?}, direction {direction}, AO {i}: analytic {g} vs finite-difference {w}"
                );
            }
        }
    }

    #[test]
    fn general_shell_gradient_matches_finite_difference() {
        let p_shell = general_shell(1, &[1.2, 0.4], &[0.6, 0.5], HarmonicType::Cartesian, [0.1, -0.2, 0.3]);
        let d_shell = general_shell(2, &[0.9, 0.3], &[0.55, 0.45], HarmonicType::Spherical, [-0.15, 0.25, 0.05]);

        for point in test_points() {
            check_gradient_matches_finite_difference(&p_shell, point);
            check_gradient_matches_finite_difference(&d_shell, point);
        }
    }

    #[test]
    fn segmented_shell_gradient_matches_finite_difference() {
        let p_shell = general_shell(1, &[1.2, 0.4], &[0.6, 0.5], HarmonicType::Cartesian, [0.1, -0.2, 0.3]);
        let d_shell = general_shell(2, &[0.9, 0.3], &[0.55, 0.45], HarmonicType::Spherical, [-0.15, 0.25, 0.05]);

        for shell in BasisShell::<Segmented>::reduce(&[p_shell, d_shell]).iter() {
            for point in test_points() {
                check_gradient_matches_finite_difference(shell, point);
            }
        }
    }

    #[test]
    fn uncontracted_shell_gradient_matches_finite_difference() {
        let p_shell = general_shell(1, &[1.2, 0.4], &[0.6, 0.5], HarmonicType::Cartesian, [0.1, -0.2, 0.3]);
        let d_shell = general_shell(2, &[0.9, 0.3], &[0.55, 0.45], HarmonicType::Spherical, [-0.15, 0.25, 0.05]);

        for shell in BasisShell::<Uncontracted>::reduce(&[p_shell, d_shell]).iter() {
            for point in test_points() {
                check_gradient_matches_finite_difference(shell, point);
            }
        }
    }
}
