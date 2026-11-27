# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""The X2C Hamiltonian with the decoupling held, and its gradient, from Python: the counterpart of
examples/fortran/x2c_frozen_demo.f90 and examples/julia/x2c_frozen_demo.jl. The transforms of a derived X2C Hamiltonian,
supplied back, give the same Hamiltonian and the same frozen gradient; the gradient of a free molecule sums to zero
over the centres.

Run from the repository root, with the library built (`cargo build --release`):

    AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces python examples/python/x2c_frozen_demo.py
"""
import sys
import numpy as np
import aedelsten


def frozen_gradient(ctx, density):
    """The frozen gradient of tr(D h): the kinetic part and the one potential in the transformation, the nuclei."""
    ctx.set_x2c_frozen_density(density)
    gradient = ctx.compute_x2c_frozen_kinetic_gradient() + ctx.compute_x2c_frozen_potential_gradient("nuclear_attraction")
    ctx.free_x2c_frozen_density()
    return gradient


def check(condition, label):
    print(("ok:   " if condition else "FAIL: ") + label)
    if not condition:
        sys.exit(1)


ctx = aedelsten.AedelstenContext("test_fixtures/mpi_oneint_check.inp")
ctx.set_shell_pairs()
ctx.set_scratch(64, "mib")
ctx.set_primitive_shell_pairs()
s, t, v = (ctx.compute_one_electron_integrals(op, primitive=True) for op in ("overlap", "kinetic", "nuclear_attraction"))
pvp = ctx.compute_one_electron_picture_change_integrals("nuclear_attraction")
h = ctx.construct_x2c_one_electron_hamiltonian(s, t, v, pvp)
density = 0.1 * np.eye(ctx.get_n_cont())
gradient = frozen_gradient(ctx, density)

u_large, u_small = ctx.get_picture_change_transforms()
ctx.set_picture_change_transforms(np.ascontiguousarray(u_large), np.ascontiguousarray(u_small))
h_held = ctx.construct_x2c_one_electron_hamiltonian(s, t, v, pvp)
check(np.abs(h_held - h).max() < 1e-12 * np.abs(h).max(), "the supplied transforms give the derived Hamiltonian")
check(np.array_equal(frozen_gradient(ctx, density), gradient), "and the same frozen gradient")
check(np.abs(gradient.sum(axis=0)).max() < 1e-10, f"the gradient sums to zero over the centres ({np.abs(gradient.sum(axis=0)).max():.1e})")

# The scalar transform of the raw planes (V, pVp) is the picture-changed nuclear attraction over the primitives; two
# planes at once are each plane's own transform.
v_2c = ctx.transform_one_electron_operator(v, pvp)
scalar, _ = ctx.compute_picture_changed_one_electron_operator("nuclear_attraction", contracted=False)
check(np.abs(v_2c - scalar[0]).max() < 1e-12 * np.abs(v_2c).max(), "transform_one_electron_operator(V, pVp) == the picture-changed V")
planes = ctx.transform_one_electron_operator(np.stack([v, s]), np.stack([pvp, t]))
check(np.abs(planes[0] - v_2c).max() < 1e-13 * np.abs(v_2c).max()
      and np.abs(planes[1] - ctx.transform_one_electron_operator(s, t)).max() < 1e-13 * np.abs(planes[1]).max(),
      "two planes at once: each its own transform")
print("PASS")
