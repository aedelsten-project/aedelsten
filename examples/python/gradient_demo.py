# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""The two-electron gradient from Python: the counterpart of examples/fortran/gradient_demo.f90 and
examples/julia/gradient_demo.jl. The restricted-HF terms of a model density on water, all centres at once and with
every other centre frozen (each the same row, the others zero), and the translational sum rule: every basis function
sits on a gradient centre, so the integral derivatives over all centres sum to zero.

Run from the repository root, with the library built (`cargo build --release`):

    AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces python examples/python/gradient_demo.py
"""
import sys
import numpy as np
import aedelsten


def check(condition, label):
    print(("ok:   " if condition else "FAIL: ") + label)
    if not condition:
        sys.exit(1)


ctx = aedelsten.AedelstenContext("test_fixtures/mpi_eri_check.inp")
ctx.set_shell_pairs()
ctx.set_scratch(64, "mib")
ctx.set_eri_data()
n = ctx.get_n_cont()
n_centres = ctx.get_n_gradient_centres()
i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
p = 0.05 * ((np.minimum(i, j) * 7 + np.maximum(i, j) * 3) % 11) - 0.1
densities = np.ascontiguousarray(p[np.newaxis])
terms = [("coulomb", 0, 0, 1.0), ("exchange", 0, 0, -0.5)]

gradient = ctx.compute_two_electron_gradient(densities, terms)
scale = np.abs(gradient).max()
print(f"n_cont = {n}, gradient centres = {n_centres}, largest entry = {scale:.6e}")
check(gradient.shape == (n_centres, 3) and scale > 1e-3, "the gradient, one row per centre")
check(np.abs(gradient.sum(axis=0)).max() < 1e-10 * scale, "translational sum rule: the rows sum to zero")
for centre in range(n_centres):
    one = ctx.compute_two_electron_gradient(densities, terms, active=[k == centre for k in range(n_centres)])
    others = np.delete(one, centre, axis=0)
    check(np.abs(one[centre] - gradient[centre]).max() < 1e-12 * scale and not others.any(),
          f"centre {centre} alone active: its row, the others zero")
try:
    ctx.compute_two_electron_gradient(densities, terms, active=[True] * (n_centres + 1))
    check(False, "a mask of the wrong length is refused")
except ValueError:
    check(True, "a mask of the wrong length is refused")
print("PASS")
