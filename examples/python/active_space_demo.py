# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""The active-space walks from Python: the counterpart of examples/fortran/active_space_demo.f90 and
examples/julia/active_space_demo.jl. The Coulomb matrices of the active pair densities M from the direct walk, one
basis and two, against M from the stored S1 tensor, T from M (transform_pair_coulomb_one_index) against T from the
S1 tensor, J/K of several densities of no symmetry in one walk against the S1 tensor and one walk each, and the
half-transformed ERIs by batches of sigma (each shell alone), one basis and two, against the S1 tensor.

Run from the repository root, with the library built (`cargo build --release`):

    AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces python examples/python/active_space_demo.py
"""
import sys
import numpy as np
import aedelsten


def check(condition, label):
    print(("ok:   " if condition else "FAIL: ") + label)
    if not condition:
        sys.exit(1)


def close(x, y, label):
    scale = np.abs(y).max()
    difference = np.abs(x - y).max()
    check(scale > 1e-2 and difference < 1e-12 * scale, f"{label} (max {scale:.3e}, difference {difference:.3e})")


ctx = aedelsten.AedelstenContext("test_fixtures/mpi_eri_check.inp")
ctx.set_shell_pairs()
ctx.set_scratch(64, "mib")
ctx.set_eri_data()
n = ctx.get_n_cont()
g = ctx.compute_eri_stored("s1")  # g[mu, nu, lambda, sigma], symmetric in every pair
rows = np.array([[(min(mu, nu) * n - min(mu, nu) * (min(mu, nu) - 1) // 2 + abs(mu - nu)) for nu in range(n)] for mu in range(n)])

m_act = 3
c = np.array([[0.2 * ((mu * 5 + u * 3 + 1) % 7) - 0.55 for mu in range(n)] for u in range(m_act)])  # (n_active, n)
c_bra = np.array([[0.2 * ((mu * 3 + u * 5 + 2) % 7) - 0.6 for mu in range(n)] for u in range(m_act)])
density = np.array([[0.05 * ((min(i, j) * 7 + max(i, j) * 3) % 11) - 0.1 for j in range(n)] for i in range(n)])


def unpack(m):
    """M(mu, nu, pair) from the packed (n (n + 1)/2, pairs)."""
    return m[rows]


# One basis: the pairs v >= w at v (v + 1)/2 + w.
_, _, _, _, m = ctx.compute_fock_direct_active_space(density, density, c, 1.0, -0.5, 1.0, -0.5)
pairs = [(v, w) for v in range(m_act) for w in range(v + 1)]
p = np.stack([(np.outer(c[v], c[w]) + np.outer(c[w], c[v])) / 2 for v, w in pairs], axis=-1)
m_ref = np.einsum("mnls,lsx->mnx", g, p)
close(unpack(m), m_ref, "M, one basis, against the S1 tensor")
t = ctx.transform_pair_coulomb_one_index(m, c)  # t[mu, vw, u]
close(t, np.einsum("un,mnx->mxu", c, m_ref), "T from M, one basis")

# Two bases: every pair (v, w) at v + w n_active.
_, _, m_mixed = ctx.compute_fock_direct_mixed_active_space(density, c_bra, c, 1.0, -0.5)
p = np.zeros((n, n, m_act, m_act))
for v in range(m_act):
    for w in range(m_act):
        p[:, :, w, v] = (np.outer(c_bra[v], c[w]) + np.outer(c[w], c_bra[v])) / 2
m_ref = np.einsum("mnls,lswv->mnwv", g, p)
close(unpack(m_mixed), m_ref, "M, two bases, against the S1 tensor")
t = ctx.transform_pair_coulomb_one_index(m_mixed.reshape(-1, m_act * m_act), c)
close(t.reshape(n, m_act, m_act, m_act), np.einsum("un,mnwv->mwvu", c, m_ref), "T from M, two bases")

# J and K of two densities of no symmetry in one walk, against the S1 tensor and against one walk each.
densities = np.stack([density + np.array([[0.03 * s * ((i * 5 + j * 2) % 7 - 3) for j in range(n)] for i in range(n)]) for s in (1.0, -2.0)])
j_multi, k_multi = ctx.compute_fock_direct_general_multi(densities, 1.0, -0.5)
for index, d in enumerate(densities):
    close(j_multi[index], np.einsum("abcd,cd->ab", g, (d + d.T) / 2), f"J of density {index} of two, against the S1 tensor")
    close(k_multi[index], -0.5 * np.einsum("abcd,bd->ac", g, d), f"K of density {index} of two, against the S1 tensor")
    j_one, k_one = ctx.compute_fock_direct_general(np.ascontiguousarray(d), 1.0, -0.5)
    check(np.array_equal(j_one, j_multi[index]) and np.array_equal(k_one, k_multi[index]), f"J/K of density {index} of two, equal to its own walk")

# The half-transformed ERIs by batches of sigma, x[sigma, r, nu, p]: every shell alone, one basis (the same array on
# both sides) and two, against the S1 tensor; the scratch they take beyond the walk's own.
starts = sorted(ctx.get_shells()[2].tolist()) + [n]
for label, left, right in (("one basis", c, c), ("two bases", c_bra, c)):
    x_ref = np.einsum("pm,rl,mnls->srnp", left, right, g)
    batches = [ctx.compute_eri_half_transformed_sigma(left, right, s, e) for s, e in zip(starts[:-1], starts[1:])]
    close(np.concatenate(batches, axis=0), x_ref, f"X by batches of sigma, {label}, against the S1 tensor")
one, two = ctx.half_transform_scratch_bytes(m_act, m_act, True, n), ctx.half_transform_scratch_bytes(m_act, m_act, False, n)
check(0 < one < two, f"the half transform's scratch, one basis {one} and two {two} bytes")
print("PASS")
