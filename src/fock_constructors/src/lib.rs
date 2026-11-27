// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Coulomb (J) and exchange (K) matrices from two-electron integrals: from a stored dense (S1) tensor
//! (`construct_*`), and the accumulators of the direct walks (`direct`, `active`,
//! `half_transform`). Every output is scaled by an explicit coefficient (its sign included), so that the caller's sums
//! of terms are plain additions: `J = coeff sum_kl D_kl (ij|kl)`, `K_ik = coeff sum_jl D_jl (ij|kl)`.

use arrays::matrix::MatrixTrait;
use arrays::array3d::Array3DTrait;

mod active;
mod digest;
mod direct;
mod half_transform;
pub use active::ActiveSpaceDirectFock;
pub use half_transform::HalfTransformedEri;
pub use direct::{DirectEriGradient, GeneralDirectFock, GeneralMultiDirectFock, InactiveActiveDirectFock, RestrictedDirectFock, TwoElectronTerm, TwoElectronTermKind, UnrestrictedDirectFock};

/// `j = j_coeff J[d]` and `k = k_coeff K[d]` from the dense tensor `g`, in one pass; both overwritten.
pub fn construct_j_and_k_stored<A4f64, Mf64, Mmutf64>(
    g: &A4f64,
    d: &Mf64,
    j: &mut Mmutf64,
    k: &mut Mmutf64,
    j_coeff: f64,
    k_coeff: f64,
)
where
    A4f64: arrays::array4d::Array4DTrait<f64>,
    Mf64: arrays::matrix::MatrixTrait<f64>,
    Mmutf64: arrays::matrix::MatrixTraitMut<f64>,
{
    j.fill(0.0);
    k.fill(0.0);
    for (g_l, d_l) in g.iter().zip(d.iter()) {
        for ((g_kl, d_kl), k_k) in g_l.iter().zip(d_l.iter()).zip(k.iter_mut()) {
            let d_kl_scaled = j_coeff * d_kl;
            for ((g_jkl, j_j), d_jl) in g_kl.iter().zip(j.iter_mut()).zip(d_l.iter()) {
                let d_jl_scaled = k_coeff * d_jl;
                for ((g_ijkl, j_ij), k_ik) in g_jkl.iter().zip(j_j.iter_mut()).zip(k_k.iter_mut()) {
                    *j_ij += d_kl_scaled * g_ijkl;
                    *k_ik += d_jl_scaled * g_ijkl;
                }
            }
        }
    }
}

/// `j = coeff J[d]` from the dense tensor `g`, overwritten.
pub fn construct_j_stored<A4f64, Mf64, Mmutf64>(
    g: &A4f64,
    d: &Mf64,
    j: &mut Mmutf64,
    coeff: f64,
)
where
    A4f64: arrays::array4d::Array4DTrait<f64>,
    Mf64: arrays::matrix::MatrixTrait<f64>,
    Mmutf64: arrays::matrix::MatrixTraitMut<f64>,
{
    j.fill(0.0);
    for (g_l, d_l) in g.iter().zip(d.iter()) {
        for (g_kl, d_kl) in g_l.iter().zip(d_l.iter()) {
            let d_kl_scaled = coeff * d_kl;
            for (g_jkl, j_j) in g_kl.iter().zip(j.iter_mut()) {
                for (g_ijkl, j_ij) in g_jkl.iter().zip(j_j.iter_mut()) {
                    *j_ij += d_kl_scaled * g_ijkl
                }
            }
        }
    }
}

/// `k = coeff K[d]`, `K[a,c] = sum_{b,d} D[b,d] (ab|cd)`, from the dense tensor `g`, overwritten: for a density of any
/// symmetry (`K[D]^T = K[D^T]`).
pub fn construct_k_stored<A4f64, Mf64, Mmutf64>(
    g: &A4f64,
    d: &Mf64,
    k: &mut Mmutf64,
    coeff: f64,
)
where
    A4f64: arrays::array4d::Array4DTrait<f64>,
    Mf64: arrays::matrix::MatrixTrait<f64>,
    Mmutf64: arrays::matrix::MatrixTraitMut<f64>,
{
    k.fill(0.0);
    for (g_l, d_l) in g.iter().zip(d.iter()) {
        for (g_kl, k_k) in g_l.iter().zip(k.iter_mut()) {
            for (g_jkl, d_jl) in g_kl.iter().zip(d_l.iter()) {
                let d_jl_scaled = coeff * d_jl;
                for (g_ijkl, k_ik) in g_jkl.iter().zip(k_k.iter_mut()) {
                    *k_ik += d_jl_scaled * g_ijkl
                }
            }
        }
    }
}

/// `k_alpha = coeff K[d_alpha]` and `k_beta = coeff K[d_beta]` from the dense tensor `g`, in one pass; both
/// overwritten.
pub fn construct_k_unrestricted_stored<A4f64, Mf64, Mmutf64>(
    g: &A4f64,
    d_alpha: &Mf64,
    d_beta: &Mf64,
    k_alpha: &mut Mmutf64,
    k_beta: &mut Mmutf64,
    coeff: f64,
)
where
    A4f64: arrays::array4d::Array4DTrait<f64>,
    Mf64: arrays::matrix::MatrixTrait<f64>,
    Mmutf64: arrays::matrix::MatrixTraitMut<f64>,
{
    k_alpha.fill(0.0);
    k_beta.fill(0.0);
    for ((g_l, d_alpha_l), d_beta_l) in g.iter().zip(d_alpha.iter()).zip(d_beta.iter()) {
        for ((g_kl, k_alpha_k), k_beta_k) in g_l.iter().zip(k_alpha.iter_mut()).zip(k_beta.iter_mut()) {
            for ((g_jkl, d_alpha_jl), d_beta_jl) in g_kl.iter().zip(d_alpha_l.iter()).zip(d_beta_l.iter()) {
                let d_alpha_scaled = coeff * d_alpha_jl;
                let d_beta_scaled = coeff * d_beta_jl;
                for ((g_ijkl, k_alpha_ik), k_beta_ik) in g_jkl.iter().zip(k_alpha_k.iter_mut()).zip(k_beta_k.iter_mut()) {
                    *k_alpha_ik += d_alpha_scaled * g_ijkl;
                    *k_beta_ik += d_beta_scaled * g_ijkl;
                }
            }
        }
    }
}

/// `j = j_coeff J[d_total]`, `k_alpha = k_coeff K[d_alpha]` and `k_beta = k_coeff K[d_beta]` from the dense tensor
/// `g`, in one pass; all three overwritten.
pub fn construct_j_and_k_unrestricted_stored<A4f64, Mf64, Mmutf64>(
    g: &A4f64,
    d_alpha: &Mf64,
    d_beta: &Mf64,
    d_total: &Mf64,
    j: &mut Mmutf64,
    k_alpha: &mut Mmutf64,
    k_beta: &mut Mmutf64,
    j_coeff: f64,
    k_coeff: f64,
)
where
    A4f64: arrays::array4d::Array4DTrait<f64>,
    Mf64: arrays::matrix::MatrixTrait<f64>,
    Mmutf64: arrays::matrix::MatrixTraitMut<f64>,
{
    j.fill(0.0);
    k_alpha.fill(0.0);
    k_beta.fill(0.0);
    for (((g_l, d_total_l), d_alpha_l), d_beta_l) in
        g.iter().zip(d_total.iter()).zip(d_alpha.iter()).zip(d_beta.iter())
    {
        for (((g_kl, d_total_kl), k_alpha_k), k_beta_k) in
            g_l.iter().zip(d_total_l.iter()).zip(k_alpha.iter_mut()).zip(k_beta.iter_mut())
        {
            let d_total_scaled = j_coeff * d_total_kl;
            for (((g_jkl, j_j), d_alpha_jl), d_beta_jl) in
                g_kl.iter().zip(j.iter_mut()).zip(d_alpha_l.iter()).zip(d_beta_l.iter())
            {
                let d_alpha_scaled = k_coeff * d_alpha_jl;
                let d_beta_scaled = k_coeff * d_beta_jl;
                for (((g_ijkl, j_ij), k_alpha_ik), k_beta_ik) in
                    g_jkl.iter().zip(j_j.iter_mut()).zip(k_alpha_k.iter_mut()).zip(k_beta_k.iter_mut())
                {
                    *j_ij += d_total_scaled * g_ijkl;
                    *k_alpha_ik += d_alpha_scaled * g_ijkl;
                    *k_beta_ik += d_beta_scaled * g_ijkl;
                }
            }
        }
    }
}

