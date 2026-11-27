// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The relativistic Hamiltonian: the finite nuclear model, the picture-change integrals, the X2C one-electron
// Hamiltonian and its decoupling transforms, and one-electron operators transformed to the X2C picture.

use super::*;

impl AedelstenContext {
    /// The picture-change integrals of `operator` over the primitive basis (`n = get_n_prim_unique()`), into `output`,
    /// OVERWRITTEN: `p.Xp` (`n x n`, or `(n, n, K)` for an operator of `K` components) or, with `spin_orbit`, the `(p X
    /// x p)` planes x, y, z and the scalar `p.Xp` per component (`(n, n, 4K)`), raw (no prefactor, the spin planes
    /// antisymmetric). Needs `set_primitive_shell_pairs` and the scratch. AIMP-special operators are an error.
    pub fn compute_one_electron_picture_change_integrals(
        &self,
        output: &mut [f64],
        operator: crate::context::one_electron::OneElectronOperator<'_>,
        spin_orbit: bool,
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_one_electron_picture_change_integrals(output, operator, spin_orbit))
    }

    /// The spin-free X2C one-electron Hamiltonian from the primitive `s`, `t`, `v` and `pvp` (`p.Vp`), each `(n_prim,
    /// n_prim)`, into `oneham`, OVERWRITTEN, with the speed of light `c_light` (`None`: the default). The decoupling
    /// transforms are kept for the operators transformed later (`transform_one_electron_operator`); with transforms
    /// supplied by `set_picture_change_transforms`, those are used instead of new ones. With symmetry set, the inputs
    /// must be totally symmetric. An error if a picture change is already derived (`free_picture_change`).
    pub fn construct_x2c_one_electron_hamiltonian(
        &mut self,
        oneham: &mut [f64],
        s: &[f64],
        t: &[f64],
        v: &[f64],
        pvp: &[f64],
        c_light: Option<f64>,
    ) -> Result<(), AedelstenError> {
        dispatch!(self, construct_x2c_one_electron_hamiltonian(oneham, s, t, v, pvp, c_light)).inspect(|_| crate::citations::x2c())
    }

    /// Whether the X2C decoupling transforms are held.
    pub fn has_picture_change(&self) -> bool {
        dispatch!(self, has_picture_change())
    }

    /// Supplies the decoupling transforms instead of deriving them: `u_large` U_L and `u_small` U_S (its 1/2c
    /// included), each `(n_prim, n_prim)` as `get_picture_change_transforms` gives them, made with `c_light` (`None`:
    /// the default). The next `construct_x2c_one_electron_hamiltonian` builds the Hamiltonian with them held rather
    /// than deriving new ones: the X2C Hamiltonian with the decoupling of another geometry, whose derivative
    /// `compute_x2c_frozen_gradient` is exactly (its finite-difference test). Replaces a picture change already there.
    pub fn set_picture_change_transforms(&mut self, u_large: &[f64], u_small: &[f64], c_light: Option<f64>) -> Result<(), AedelstenError> {
        dispatch!(self, set_picture_change_transforms(u_large, u_small, c_light))
    }

    /// Holds the densities of the X2C gradient with the decoupling held (U_L and U_S fixed), from `density` (`n_cont x
    /// n_cont`, symmetric), for `compute_x2c_frozen_kinetic_gradient` and `compute_x2c_frozen_potential_gradient`:
    /// `D` taken back to the primitive basis, `D_p = C D C^T`, and
    ///
    ///   D_V = U_L D_p U_L^T,   D_W = U_S D_p U_S^T,   D_T = 2c (U_L D_p U_S^T + U_S D_p U_L^T) - 4c^2 D_W,
    ///
    /// U_S with its 1/2c, three `n_prim x n_prim` matrices (`memory_breakdown`'s `x2c_frozen_density`). Replaces
    /// densities already held. Needs the picture change and, for a contracted context, the contraction matrix
    /// (`set_primitive_shell_pairs`). A new picture change, `free_picture_change` or `set_centre_positions` drops them.
    pub fn set_x2c_frozen_density(&mut self, density: &[f64]) -> Result<(), AedelstenError> {
        dispatch!(self, set_x2c_frozen_density(density))
    }

    /// Frees the densities `set_x2c_frozen_density` holds.
    pub fn free_x2c_frozen_density(&mut self) {
        dispatch!(self, free_x2c_frozen_density())
    }

    /// The kinetic part of the X2C one-electron energy's gradient with the decoupling held, `tr(D_T T^X)` (D_T of
    /// `set_x2c_frozen_density`), into `gradient`, a `(3, get_n_gradient_centres())` column-major matrix,
    /// overwritten. With `compute_x2c_frozen_potential_gradient` for each potential that went into V and p.Vp, the
    /// sum is the derivative of `tr(D h_X2C)` with U_L and U_S fixed; at the decoupling's geometry the X2C
    /// one-electron energy's gradient but for the decoupling's own response (L. Cheng, J. Gauss, J. Chem. Phys. 135,
    /// 084114 (2011)). Needs the densities, the primitive shell pairs and the scratch. `active` is the frozen-centre
    /// mask of the other gradient calls.
    pub fn compute_x2c_frozen_kinetic_gradient(&self, gradient: &mut [f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        dispatch!(self, compute_x2c_frozen_kinetic_gradient(gradient, active)).inspect(|_| crate::citations::x2c_gradient())
    }

    /// The part of the X2C one-electron energy's gradient with the decoupling held of one potential `operator` that
    /// went into V and p.Vp, `tr(D_V V^X) + tr(D_W W^X)` (`W = p.Vp`; D_V and D_W of `set_x2c_frozen_density`), into
    /// `gradient` as in `compute_x2c_frozen_kinetic_gradient`, overwritten. The caller names each potential it put into
    /// the transformation (the nuclear attraction, the external multipoles, the embedding AIMPs' Coulomb part, ...)
    /// and sums the parts. The sources move as in `compute_one_electron_gradient`. A matrix operator; needs the
    /// densities, the primitive shell pairs and the scratch.
    pub fn compute_x2c_frozen_potential_gradient(
        &self,
        gradient: &mut [f64],
        operator: crate::context::one_electron::OneElectronOperator<'_>,
        active: Option<&[bool]>,
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_x2c_frozen_potential_gradient(gradient, operator, active)).inspect(|_| crate::citations::x2c_gradient())
    }

    /// Copies the X2C decoupling transforms (each `(n_prim, n_prim)`, column-major): `u_large` for plain one-electron
    /// operators, `u_small` for their pXp counterparts (see `transform_one_electron_operator`).
    pub fn get_picture_change_transforms(&self, u_large: &mut [f64], u_small: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, get_picture_change_transforms(u_large, u_small))
    }

    /// Frees the X2C decoupling transforms.
    pub fn free_picture_change(&mut self) {
        dispatch!(self, free_picture_change())
    }

    /// A one-electron operator in the X2C picture, `U_L^T X U_L + U_S^T (p.Xp) U_S` into `out`, from `op_plain` (`X`)
    /// and `op_pxp` (`p.Xp`), each `(n_prim, n_prim)` over the primitive basis. Needs the decoupling.
    pub fn transform_one_electron_operator(&self, op_plain: &[f64], op_pxp: &[f64], out: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, transform_one_electron_operator(op_plain, op_pxp, out))
    }

    /// `transform_one_electron_operator` for each of the `n_planes` planes, `(n_prim, n_prim, n_planes)` each.
    pub fn transform_one_electron_operator_array3d(&self, op_plain: &[f64], op_pxp: &[f64], n_planes: usize, out: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, transform_one_electron_operator_array3d(op_plain, op_pxp, n_planes, out))
    }

    /// The spin planes of a one-electron operator in the X2C picture, `U_S^T X_k U_S` per plane of `op_pxp` (`(n, n,
    /// n_planes)` over the primitive basis, `n` the decoupling's): the raw `(p X x p)_k` planes of
    /// `compute_one_electron_picture_change_integrals(.., spin_orbit: true)`, to which other raw spin planes may be
    /// added first (the AMFI planes of `compute_amfi_integrals`). Antisymmetric planes stay antisymmetric (no
    /// symmetrisation, unlike `transform_one_electron_operator_array3d`, which is for symmetric operators). Needs the
    /// decoupling.
    pub fn transform_one_electron_spin_operator(&self, op_pxp: &[f64], n_planes: usize, out: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, transform_one_electron_spin_operator(op_pxp, n_planes, out))
    }

    /// A one-electron operator `X` of `K` components (1 for a matrix operator) in the X2C picture,
    /// `X_2c = scalar (x) 1 + i sum_k sigma_k (x) spin_k`: `scalar` `(n, n, K)`, `U_L^T X U_L + U_S^T (p.Xp) U_S` per
    /// component, symmetric; `spin` `(n, n, 3K)`, plane `3c + k` the `k`-th of component `c`, `U_S^T (p X x p)_k U_S`,
    /// antisymmetric. Over the contracted basis (`contracted`) or the primitive one. Needs the decoupling and the
    /// primitive shell pairs (`set_primitive_shell_pairs`). AIMP-special operators are not supported.
    pub fn compute_picture_changed_one_electron_operator(
        &self,
        operator: crate::context::one_electron::OneElectronOperator<'_>,
        contracted: bool,
        scalar: &mut [f64],
        spin: &mut [f64],
    ) -> Result<(), AedelstenError> {
        dispatch!(self, compute_picture_changed_one_electron_operator(operator, contracted, scalar, spin))
    }

    /// The atomic mean-field two-electron spin-orbit integrals (B. A. Heß, C. M. Marian, U. Wahlgren, O. Gropen, Chem.
    /// Phys. Lett. 251, 365 (1996)) over the primitive basis, `(n_prim, n_prim, 6)`: the spin-same-orbit mean field `x,
    /// y, z`, then the spin-other-orbit exchange `x, y, z`. Raw, in the convention of the one-electron spin planes of
    /// `compute_one_electron_picture_change_integrals(.., spin_orbit: true)` (no prefactor, antisymmetric): Breit-Pauli
    /// is `C^T (SSO_k + SOO_k) C / 4c^2`, X2C the planes added to the one-electron ones before
    /// `transform_one_electron_spin_operator`. The density is each atom's ANO-RCC ground state; zero on hydrogen and
    /// ghost centres; core AIMP centres are an error. Needs the primitive shells (`set_primitive_shell_pairs`) and the
    /// scratch.
    pub fn compute_amfi_integrals(&self, output: &mut [f64]) -> Result<(), AedelstenError> {
        dispatch!(self, compute_amfi_integrals(output)).inspect(|_| crate::citations::amfi())
    }

    /// The nuclear charge distribution (point or Gaussian) of the nuclear attraction, the repulsion and the X2C
    /// Hamiltonian from now on. An error while a polarization state holds the static field (`free_polarization` first).
    pub fn set_finite_nuclear_model(&mut self, model: basis::charge_carriers::FiniteModel) -> Result<(), AedelstenError> {
        let cite = crate::citations::finite_nucleus(&model);
        dispatch!(self, set_finite_nuclear_model(model)).inspect(|_| cite())
    }
}

// The densities of the X2C gradient with the decoupling held, over the primitive basis (`set_x2c_frozen_density`).
#[derive(Debug)]
pub(crate) struct X2cFrozenDensities {
    // D_V, contracted with the potentials' derivative integrals.
    potential: arrays::matrix::Matrix<f64>,
    // D_W, with their p.Vp derivative integrals.
    pvp: arrays::matrix::Matrix<f64>,
    // D_T, with the kinetic energy's.
    kinetic: arrays::matrix::Matrix<f64>,
}

impl X2cFrozenDensities {
    // The bytes of the three matrices.
    pub(crate) fn bytes(&self) -> usize {
        use arrays::matrix::MatrixTrait;
        [&self.potential, &self.pvp, &self.kinetic].iter().map(|m| m.get_data().len()).sum::<usize>() * std::mem::size_of::<f64>()
    }
}

impl<S: basis::contraction::ShellKind> AedelstenContextImpl<S> {
    pub fn set_finite_nuclear_model(&mut self, model: basis::charge_carriers::FiniteModel) -> Result<(), AedelstenError> {
        if self.polarization_caches_static_field() {
            return Err(AedelstenError::CacheAlreadyDerived(
                "The polarization state caches the static field, which depends on the nuclear model. Call free_polarization() first.".to_string(),
            ));
        }
        let qm_region = self.qm_region.get_cache_mut().expect("input has been freed (free_input)");
        qm_region.input.set_finite_nuclear_model(model);
        Ok(())
    }

    pub fn construct_x2c_one_electron_hamiltonian(
        &mut self,
        oneham: &mut [f64],
        s: &[f64],
        t: &[f64],
        v: &[f64],
        pvp: &[f64],
        c_light: Option<f64>,
    ) -> Result<(), AedelstenError> {
        let supplied = matches!(&self.picture_change, AedelstenCacheState::Ready(pc) if pc.is_supplied());
        if self.picture_change.is_already_derived() && !supplied {
            return Err(AedelstenError::CacheAlreadyDerived(
                "Picture-change transform has already been derived. Call free_picture_change() before deriving a new one.".to_string()
            ));
        }

        let n = self.get_n_prim_unique();
        check_matrix_dimensions(n, oneham)?;
        check_matrix_dimensions(n, s)?;
        check_matrix_dimensions(n, t)?;
        check_matrix_dimensions(n, v)?;
        check_matrix_dimensions(n, pvp)?;

        // With symmetry set: from invariant inputs, the Hamiltonian and the decoupling transforms are invariant up to
        // rounding (X2C is covariant), so the inputs are checked here, before the nonlinear step. Averaging an
        // asymmetric environment after X2C would not be X2C of the averaged environment.
        const ADVICE: &str = "average the environment part before X2C (compute it with symmetry set, or apply \
                              symmetrise_matrix(.., Cleaning::Average) to your own term)";
        for (matrix, name) in [(s, "S"), (t, "T"), (v, "V"), (pvp, "pVp")] {
            self.require_symmetric_input(matrix, true, name, ADVICE)?;
        }

        let s_matrix = arrays::matrix::MatrixView::new(n, n, s);
        let t_matrix = arrays::matrix::MatrixView::new(n, n, t);
        let v_matrix = arrays::matrix::MatrixView::new(n, n, v);
        let pvp_matrix = arrays::matrix::MatrixView::new(n, n, pvp);
        let mut oneham_matrix = arrays::matrix::MatrixViewMut::new(n, n, oneham);

        let c_light = c_light.unwrap_or(constants::C_IN_AU);
        self.drop_x2c_frozen_density();
        if supplied {
            let pc = self.get_picture_change()?;
            if pc.get_u_large().rows() != n {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n, actual: pc.get_u_large().rows() });
            }
            if pc.get_c_light() != c_light {
                return Err(AedelstenError::StringError(format!("the supplied transforms were made with c = {}, not {c_light}", pc.get_c_light())));
            }
            pc.frozen_one_electron_hamiltonian(&mut oneham_matrix, &t_matrix, &v_matrix, &pvp_matrix);
            return Ok(());
        }
        let mut pc = x2c::assemble_x2c_one_electron_hamiltonian(&mut oneham_matrix, &s_matrix, &t_matrix, &v_matrix, &pvp_matrix, c_light);
        // From the checked inputs, the outputs are symmetric up to the decoupling's rounding, which grows with the
        // primitive basis's conditioning: averaged away (never refused), the amounts reported. The transforms too, so
        // that every operator transformed later keeps its symmetry to rounding.
        use crate::context::point_group::Cleaning::Average;
        use arrays::matrix::MatrixTraitMut;
        self.clean_library_output(oneham, true, Average, "x2c_one_electron_hamiltonian")?;
        let (u_large, u_small) = pc.get_transforms_mut();
        self.clean_library_output(u_large.get_data_mut(), true, Average, "x2c_u_large")?;
        self.clean_library_output(u_small.get_data_mut(), true, Average, "x2c_u_small")?;
        self.picture_change = AedelstenCacheState::Ready(pc);
        Ok(())
    }

    // Internal: `x2c::PictureChange` holds `arrays` matrices; the public getters copy into slices.
    pub(crate) fn get_picture_change(&self) -> Result<&x2c::PictureChange, AedelstenError> {
        self.picture_change.get_cache().map_err(|_| {
            AedelstenError::CacheNotReady("picture-change transform is not resident in memory; call construct_x2c_one_electron_hamiltonian() first".to_string())
        })
    }

    pub fn has_picture_change(&self) -> bool {
        self.picture_change.is_ready()
    }

    pub fn get_picture_change_transforms(&self, u_large: &mut [f64], u_small: &mut [f64]) -> Result<(), AedelstenError> {
        let pc = self.get_picture_change()?;
        for (out, source) in [(u_large, pc.get_u_large()), (u_small, pc.get_u_small())] {
            if out.len() != source.get_data().len() {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: source.get_data().len(), actual: out.len() });
            }
            // A getter: the transform stays in the cache, the caller gets its own copy.
            out.copy_from_slice(source.get_data());
        }
        Ok(())
    }

    pub fn free_picture_change(&mut self) {
        self.picture_change.debug_double_free_warning("picture-change transform");
        self.picture_change.free_cache();
        self.drop_x2c_frozen_density();
    }

    // The X2C frozen densities are of the decoupling held: a new one, or none, invalidates them.
    fn drop_x2c_frozen_density(&mut self) {
        if self.x2c_frozen_density.is_ready() {
            self.x2c_frozen_density.invalidate("the X2C decoupling changed; call set_x2c_frozen_density() again".to_string());
        }
    }

    pub fn set_picture_change_transforms(&mut self, u_large: &[f64], u_small: &[f64], c_light: Option<f64>) -> Result<(), AedelstenError> {
        let n = self.get_n_prim_unique();
        check_matrix_dimensions(n, u_large)?;
        check_matrix_dimensions(n, u_small)?;
        let copy = |data: &[f64]| arrays::matrix::Matrix::<f64>::new_from_vec(n, n, data.to_vec().into_boxed_slice());
        let c_light = c_light.unwrap_or(constants::C_IN_AU);
        self.drop_x2c_frozen_density();
        self.picture_change = AedelstenCacheState::Ready(x2c::PictureChange::from_transforms(copy(u_large), copy(u_small), c_light));
        Ok(())
    }

    pub fn set_x2c_frozen_density(&mut self, density: &[f64]) -> Result<(), AedelstenError> {
        use arrays::matrix::{ColumnFactor, Factor, Matrix, MatrixTrait, MatrixView};
        let n_cont = self.get_n_cont();
        check_matrix_dimensions(n_cont, density)?;
        let pc = self.get_picture_change()?;
        let (u_large, u_small) = (pc.get_u_large(), pc.get_u_small());
        let n = u_large.rows();
        // A X A^T.
        fn sandwich<X: Factor>(a: &Matrix<f64>, x: &X) -> Matrix<f64> {
            a.matmul(x).matmul(&a.view_transposed())
        }
        // D_p = C D C^T (the primitive basis is the contracted one for a primitive context).
        let d_p = if S::PRIMITIVE {
            Matrix::<f64>::new_from_vec(n, n, density.to_vec().into_boxed_slice())
        } else {
            let c = self.contraction_matrix.get_cache().map_err(|_| {
                AedelstenError::CacheNotReady("contraction matrix is not resident in memory; call set_primitive_shell_pairs() first".to_string())
            })?;
            sandwich(c, &MatrixView::new(n_cont, n_cont, density))
        };
        let potential = sandwich(u_large, &d_p);
        let pvp = sandwich(u_small, &d_p);
        // M = U_L D_p U_S^T.
        let m = u_large.matmul(&d_p).matmul(&u_small.view_transposed());
        let two_c = 2.0 * pc.get_c_light();
        let mut kinetic = Matrix::<f64>::new(n, n);
        for j in 0..n {
            for i in 0..n {
                kinetic[(i, j)] = two_c * (m[(i, j)] + m[(j, i)]) - two_c * two_c * pvp[(i, j)];
            }
        }
        self.x2c_frozen_density = AedelstenCacheState::Ready(X2cFrozenDensities { potential, pvp, kinetic });
        Ok(())
    }

    pub fn free_x2c_frozen_density(&mut self) {
        self.x2c_frozen_density.debug_double_free_warning("X2C frozen densities");
        self.x2c_frozen_density.free_cache();
    }

    // The densities, or the error naming their setter.
    fn x2c_frozen_densities(&self) -> Result<&X2cFrozenDensities, AedelstenError> {
        self.x2c_frozen_density.get_cache().map_err(|reason| {
            AedelstenError::CacheNotReady(format!("the X2C frozen densities are not held ({reason}); call set_x2c_frozen_density() first"))
        })
    }

    // An error unless `gradient` is `(3, n_centres)` and `active` has a flag per centre.
    fn check_gradient_output(&self, gradient: &[f64], active: Option<&[bool]>) -> Result<usize, AedelstenError> {
        let n_centres = self.get_n_gradient_centres();
        if gradient.len() != 3 * n_centres {
            return Err(AedelstenError::InvalidMatrixDimensions { expected: 3 * n_centres, actual: gradient.len() });
        }
        if let Some(mask) = active {
            if mask.len() != n_centres {
                return Err(AedelstenError::InvalidMatrixDimensions { expected: n_centres, actual: mask.len() });
            }
        }
        Ok(n_centres)
    }

    pub fn compute_x2c_frozen_kinetic_gradient(&self, gradient: &mut [f64], active: Option<&[bool]>) -> Result<(), AedelstenError> {
        let n_centres = self.check_gradient_output(gradient, active)?;
        let densities = self.x2c_frozen_densities()?;
        let pairs = self.primitive_pairs()?;
        let mut scratch = self.require_scratch()?;
        let mut total = arrays::matrix::MatrixViewMut::new(3, n_centres, gradient);
        crate::context::one_electron::compute_one_electron_gradient(
            pairs, crate::context::one_electron::OneElectronPlainOperator::Kinetic, &mut total, &densities.kinetic, active, &mut scratch, &self.comm,
        )
    }

    pub fn compute_x2c_frozen_potential_gradient(
        &self,
        gradient: &mut [f64],
        operator: crate::context::one_electron::OneElectronOperator<'_>,
        active: Option<&[bool]>,
    ) -> Result<(), AedelstenError> {
        use arrays::matrix::MatrixMathMut;
        let n_centres = self.check_gradient_output(gradient, active)?;
        let operator = match operator {
            crate::context::one_electron::OneElectronOperator::MatrixOutput(op) => op,
            _ => return Err(AedelstenError::StringError("a potential in the X2C transformation is a matrix operator".to_string())),
        };
        let densities = self.x2c_frozen_densities()?;
        let pairs = self.primitive_pairs()?;
        let mut scratch = self.require_scratch()?;
        let mut total = arrays::matrix::MatrixViewMut::new(3, n_centres, gradient);
        crate::context::one_electron::compute_one_electron_gradient(pairs, operator.clone(), &mut total, &densities.potential, active, &mut scratch, &self.comm)?;
        let mut pvp_part = arrays::matrix::Matrix::<f64>::new(3, n_centres);
        crate::context::one_electron::compute_one_electron_pvp_gradient(pairs, operator, &mut pvp_part, &densities.pvp, active, &mut scratch, &self.comm)?;
        total.add_in_place(&pvp_part);
        Ok(())
    }

    pub fn transform_one_electron_operator(&self, op_plain: &[f64], op_pxp: &[f64], out: &mut [f64]) -> Result<(), AedelstenError> {
        let pc = self.get_picture_change()?;
        let n = pc.get_u_large().rows();
        check_matrix_dimensions(n, op_plain)?;
        check_matrix_dimensions(n, op_pxp)?;
        check_matrix_dimensions(n, out)?;
        let op_plain_matrix = arrays::matrix::MatrixView::new(n, n, op_plain);
        let op_pxp_matrix = arrays::matrix::MatrixView::new(n, n, op_pxp);
        let mut out_matrix = arrays::matrix::MatrixViewMut::new(n, n, out);
        pc.transform_one_electron_operator(&op_plain_matrix, &op_pxp_matrix, &mut out_matrix);
        Ok(())
    }

    pub fn transform_one_electron_operator_array3d(&self, op_plain: &[f64], op_pxp: &[f64], n_planes: usize, out: &mut [f64]) -> Result<(), AedelstenError> {
        let pc = self.get_picture_change()?;
        let n = pc.get_u_large().rows();
        check_array3d_dimensions(n, n_planes, op_plain)?;
        check_array3d_dimensions(n, n_planes, op_pxp)?;
        check_array3d_dimensions(n, n_planes, out)?;
        let op_plain_view = arrays::array3d::Array3DView::new(n, n, n_planes, op_plain);
        let op_pxp_view = arrays::array3d::Array3DView::new(n, n, n_planes, op_pxp);
        let mut out_view = arrays::array3d::Array3DViewMut::new(n, n, n_planes, out);

        // scratch is reused between the op_plain and op_pxp sandwiches within
        // a single plane too (they run sequentially -- see
        // transform_one_electron_operator_prealloc's own doc comment).
        let mut scratch = arrays::matrix::Matrix::<f64>::new(n, n);
        let mut transformed_plain = arrays::matrix::Matrix::<f64>::new(n, n);
        let mut transformed_pxp = arrays::matrix::Matrix::<f64>::new(n, n);

        for k in 0..n_planes {
            let op_plain_plane = op_plain_view.view_sub_matrix(k);
            let op_pxp_plane = op_pxp_view.view_sub_matrix(k);
            let mut out_plane = out_view.view_sub_matrix_mut(k);
            pc.transform_one_electron_operator_prealloc(
                &op_plain_plane, &op_pxp_plane,
                &mut scratch, &mut transformed_plain, &mut transformed_pxp,
                &mut out_plane,
            );
        }
        Ok(())
    }

    pub fn transform_one_electron_spin_operator(&self, op_pxp: &[f64], n_planes: usize, out: &mut [f64]) -> Result<(), AedelstenError> {
        let pc = self.get_picture_change()?;
        let n = pc.get_u_small().rows();
        check_array3d_dimensions(n, n_planes, op_pxp)?;
        check_array3d_dimensions(n, n_planes, out)?;
        let mut scratch = arrays::matrix::Matrix::<f64>::new(n, n);
        for (plane, out_plane) in op_pxp.chunks_exact(n * n).zip(out.chunks_exact_mut(n * n)) {
            let plane = arrays::matrix::MatrixView::new(n, n, plane);
            let mut out_plane = arrays::matrix::MatrixViewMut::new(n, n, out_plane);
            pc.transform_one_electron_spin_operator_prealloc(&plane, &mut scratch, &mut out_plane);
        }
        Ok(())
    }

    pub fn compute_picture_changed_one_electron_operator(
        &self,
        operator: crate::context::one_electron::OneElectronOperator<'_>,
        contracted: bool,
        scalar: &mut [f64],
        spin: &mut [f64],
    ) -> Result<(), AedelstenError> {
        use crate::context::one_electron::OneElectronOperator;
        let n_components = match &operator {
            OneElectronOperator::MatrixOutput(_) => 1,
            OneElectronOperator::Array3DOutput(op) => op.get_expected_components(),
            OneElectronOperator::AimpSpecial(_) => {
                return Err(AedelstenError::StringError(
                    "picture-changed operators are not implemented for AIMP-special operators".to_string(),
                ));
            }
        };
        let pc = self.get_picture_change()?;
        let n = self.get_n_prim_unique();
        let n_out = if contracted { self.get_n_cont() } else { n };
        check_array3d_dimensions(n_out, n_components, scalar)?;
        check_array3d_dimensions(n_out, 3 * n_components, spin)?;

        // The plain integrals, then per component the spin planes x, y, z and the scalar p.Xp.
        let n2 = n * n;
        let mut plain = vec![0.0; n_components * n2];
        self.compute_one_electron_integrals(&mut plain, operator.clone(), true)?;
        let mut full = vec![0.0; 4 * n_components * n2];
        self.compute_one_electron_picture_change_integrals(&mut full, operator, true)?;

        let mut scratch = arrays::matrix::Matrix::<f64>::new(n, n);
        let mut transformed_plain = arrays::matrix::Matrix::<f64>::new(n, n);
        let mut transformed_pxp = arrays::matrix::Matrix::<f64>::new(n, n);
        let mut folded = arrays::matrix::Matrix::<f64>::new(n, n);
        let n_out2 = n_out * n_out;
        let place = |folded: &arrays::matrix::Matrix<f64>, target: &mut [f64]| -> Result<(), AedelstenError> {
            if contracted {
                self.contract_one_electron_matrix(folded.get_data(), target)
            } else {
                target.copy_from_slice(folded.get_data());
                Ok(())
            }
        };
        for c in 0..n_components {
            let plane = |k: usize| arrays::matrix::MatrixView::new(n, n, &full[(4 * c + k) * n2..(4 * c + k + 1) * n2]);
            let plain_c = arrays::matrix::MatrixView::new(n, n, &plain[c * n2..(c + 1) * n2]);
            pc.transform_one_electron_operator_prealloc(&plain_c, &plane(3), &mut scratch, &mut transformed_plain, &mut transformed_pxp, &mut folded);
            place(&folded, &mut scalar[c * n_out2..(c + 1) * n_out2])?;
            for k in 0..3 {
                pc.transform_one_electron_spin_operator_prealloc(&plane(k), &mut scratch, &mut folded);
                place(&folded, &mut spin[(3 * c + k) * n_out2..(3 * c + k + 1) * n_out2])?;
            }
        }
        Ok(())
    }

    pub fn compute_amfi_integrals(&self, output: &mut [f64]) -> Result<(), AedelstenError> {
        let shells = self.primitive_shells()?;
        let input = &self.qm_region.get_cache().expect("input has been freed (free_input)").input;
        let n = self.get_n_prim_unique();
        check_array3d_dimensions(n, amfi::N_AMFI_PLANES, output)?;
        {
            let mut scratch = self.require_scratch()?;
            let mut planes = arrays::array3d::Array3DViewMut::new(n, n, amfi::N_AMFI_PLANES, output);
            amfi::compute_amfi(input, shells, &mut planes, &mut scratch)?;
        }
        // With symmetry set: two axial vectors, the spin-same-orbit planes and the spin-other-orbit ones.
        use crate::context::point_group::Cleaning::Strict;
        let (same_orbit, other_orbit) = output.split_at_mut(3 * n * n);
        self.clean_library_tensor(same_orbit, true, 1, symmetry::TensorKind::Axial, None, Strict, "amfi_spin_same_orbit")?;
        self.clean_library_tensor(other_orbit, true, 1, symmetry::TensorKind::Axial, None, Strict, "amfi_spin_other_orbit")?;
        Ok(())
    }

    pub fn compute_one_electron_picture_change_integrals(
        &self,
        output: &mut [f64],
        operator: crate::context::one_electron::OneElectronOperator<'_>,
        spin_orbit: bool,
    ) -> Result<(), AedelstenError> {
        // Always over the primitive basis: the picture-change operator is contracted afterwards, never built in a
        // contracted basis.
        let expected_n = self.get_n_prim_unique();
        const SO_COMPS: usize = 4; // x, y, z, scalar
        // With symmetry set: the scalar integrals of a plain operator cleaned as the operator itself is; its spin-orbit
        // ones as an axial vector (x, y, z) and the scalar plane; the scalar picture-change moments as a tensor.
        let cleaning = crate::context::point_group::one_electron_cleaning(&operator);
        let moment = match &operator {
            crate::context::one_electron::OneElectronOperator::Array3DOutput(crate::context::one_electron::OneElectronComponentOperator::Moment(origin)) if !spin_orbit => {
                Some((*origin.get_origin(), origin.get_order()))
            }
            _ => None,
        };

        match operator {
            crate::context::one_electron::OneElectronOperator::MatrixOutput(op) => {
                let shell_pairs = self.primitive_pairs()?;
                let mut scratch = self.require_scratch()?;
                if !spin_orbit {
                    check_matrix_dimensions(expected_n, output)?;
                    let mut wrapped_output = arrays::matrix::MatrixViewMut::new(expected_n, expected_n, output);
                    crate::context::one_electron::compute_one_electron_scalar_picture_change_integrals_matrix(
                        shell_pairs, op, &mut scratch, &self.comm, &mut wrapped_output,
                    )?;
                } else {
                    check_array3d_dimensions(expected_n, SO_COMPS, output)?;
                    let mut wrapped_output = arrays::array3d::Array3DViewMut::new(expected_n, expected_n, SO_COMPS, output);
                    crate::context::one_electron::compute_one_electron_spin_orbit_picture_change_integrals_matrix(
                        shell_pairs, op, &mut scratch, &self.comm, &mut wrapped_output,
                    )?;
                }
            }
            crate::context::one_electron::OneElectronOperator::Array3DOutput(op) => {
                let shell_pairs = self.primitive_pairs()?;
                let mut scratch = self.require_scratch()?;
                let mut n_comps = op.get_expected_components();
                if spin_orbit {
                    n_comps *= SO_COMPS;
                }
                check_array3d_dimensions(expected_n, n_comps, output)?;
                let mut wrapped_output = arrays::array3d::Array3DViewMut::new(expected_n, expected_n, n_comps, output);
                if !spin_orbit {
                    crate::context::one_electron::compute_one_electron_scalar_picture_change_integrals_array3d(
                        shell_pairs, op, &mut scratch, &self.comm, &mut wrapped_output,
                    )?;
                } else {
                    crate::context::one_electron::compute_one_electron_spin_orbit_picture_change_integrals_array3d(
                        shell_pairs, op, &mut scratch, &self.comm, &mut wrapped_output,
                    )?;
                }
            }
            crate::context::one_electron::OneElectronOperator::AimpSpecial(_) => {
                return Err(AedelstenError::StringError(
                    "picture-change integrals are not implemented for AIMP-special operators".to_string(),
                ));
            }
        }
        if let Some(cleaning) = cleaning {
            if spin_orbit {
                // Planes x, y, z (an axial vector), then the scalar one.
                let (vector, scalar) = output.split_at_mut(3 * expected_n * expected_n);
                self.clean_library_tensor(vector, true, 1, symmetry::TensorKind::Axial, None, cleaning.cleaning, cleaning.spin_orbit)?;
                self.clean_library_output(scalar, true, cleaning.cleaning, cleaning.picture_change)?;
            } else {
                self.clean_library_output(output, true, cleaning.cleaning, cleaning.picture_change)?;
            }
        }
        if let Some((origin, order)) = moment {
            use crate::context::point_group::Cleaning::Strict;
            self.clean_library_tensor(output, true, order, symmetry::TensorKind::Polar, Some(origin), Strict, "picture_change_oneint_moment")?;
        }

        Ok(())
    }
}
