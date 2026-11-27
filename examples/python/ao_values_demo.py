# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""The AO values and their derivatives from Python: the counterpart of examples/fortran/ao_values_demo.f90 and
examples/julia/ao_values_demo.jl. compute_ao_hessians gives the values and gradients compute_ao_values gives, and second
derivatives that agree with central differences of the gradients.

Run from the repository root, with the library built (`cargo build --release`):

    AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces python examples/python/ao_values_demo.py
"""
import sys
import numpy as np
import aedelsten


def check(condition, label):
    print(("ok:   " if condition else "FAIL: ") + label)
    if not condition:
        sys.exit(1)


ctx = aedelsten.AedelstenContext("test_fixtures/mpi_oneint_check.inp")
shells = list(range(ctx.get_n_shells()))
t = np.arange(40, dtype=np.float64)
points = np.column_stack((np.sin(0.7 * t) * 1.8, np.cos(1.3 * t) * 1.5, 0.11 * t - 2.0))

values, gradients = ctx.compute_ao_values(shells, points, gradients=True)
h_values, h_gradients, hessians = ctx.compute_ao_hessians(shells, points)
check(np.array_equal(values, h_values) and np.array_equal(gradients, h_gradients),
      "compute_ao_hessians gives compute_ao_values's values and gradients")

step, worst = 1e-5, 0.0
for k, (a, b) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
    shift = np.zeros(3)
    shift[b] = step
    plus = ctx.compute_ao_values(shells, points + shift, gradients=True)[1][a]
    minus = ctx.compute_ao_values(shells, points - shift, gradients=True)[1][a]
    worst = max(worst, (np.abs(hessians[k] - (plus - minus) / (2 * step)) / (1 + np.abs(hessians[k]))).max())
check(worst < 1e-6, f"the second derivatives against central differences of the gradients (worst {worst:.1e})")
print("PASS")
