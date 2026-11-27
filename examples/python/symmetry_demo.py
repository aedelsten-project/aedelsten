# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""The point group's operations from Python: the counterpart of examples/fortran/symmetry_demo.f90 and
examples/julia/symmetry_demo.jl. Ammonia turned out of every axis (test_fixtures/symmetry_demo.inp), C3v: every
operation's matrix orthogonal, its images a permutation of equivalent centres, every vector between two centres
carried onto the vector between their images, one identity.
The irreps: the AO counts, the characters orthonormal, the irrep matrices orthogonal with the characters as
their traces, a projection onto a one-dimensional irrep carried to its character times itself.

Run from the repository root, with the library built (`cargo build --release`):

    AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces python examples/python/symmetry_demo.py
"""
import sys
import numpy as np
import aedelsten

# The fixture's positions (bohr) and elements.
POSITIONS = np.array([
    [0.4000000000000000, -1.1837482993408999, 2.3994294843469834],
    [2.0909455857523227, -0.2618653840665375, 2.1168787516457970],
    [-0.8984653640277757, 0.2580710345499226, 2.5548151556260290],
    [0.0075197782754526, -1.9820015685184955, 0.6680280306678168],
])
ELEMENTS = ["N", "H", "H", "H"]


def check(condition, label):
    print(("ok:   " if condition else "FAIL: ") + label)
    if not condition:
        sys.exit(1)


ctx = aedelsten.AedelstenContext("test_fixtures/symmetry_demo.inp")
ctx.set_shell_pairs()
ctx.set_scratch(64, "mib")
ctx.set_symmetry()
check(ctx.get_point_group_name() == "C3v", "the point group is C3v")
h = ctx.get_symmetry_operation_count()
check(h == 6, "six operations")
n = ctx.get_n_symmetry_centres()
check(n == len(POSITIONS), "four centres")

identities = 0
for op in range(h):
    m = ctx.get_symmetry_operation_matrix(op)
    images = ctx.get_symmetry_centre_images(op)
    check(np.abs(m.T @ m - np.eye(3)).max() < 1e-12, f"operation {op} is orthogonal")
    check(sorted(images) == list(range(n)) and all(ELEMENTS[a] == ELEMENTS[b] for a, b in enumerate(images)),
          f"operation {op} permutes equivalent centres")
    carried = max(np.abs(m @ (POSITIONS[a] - POSITIONS[b]) - (POSITIONS[images[a]] - POSITIONS[images[b]])).max()
                  for a in range(n) for b in range(n))
    check(carried < 1e-8, f"operation {op} carries every vector between centres onto its images'")
    identities += int(all(images == np.arange(n)) and np.abs(m - np.eye(3)).max() < 1e-12)
check(identities == 1, "one identity")

# The irreps: AO functions in each, together all; the characters orthonormal over the group; each aligned irrep
# matrix orthogonal with the character as its trace.
dimensions = [d for _, d in ctx.get_irreps()]
n_irreps = len(dimensions)
check(ctx.get_ao_irrep_counts().sum() == ctx.get_n_cont(), "every AO function in one irrep")
chi = np.array([[ctx.get_irrep_character(i, op) for op in range(h)] for i in range(n_irreps)])
check(np.abs(chi @ chi.T / h - np.eye(n_irreps)).max() < 1e-12, "the characters are orthonormal")
matrices = 0
for i in range(n_irreps):
    for op in range(h):
        g = ctx.get_irrep_matrix(i, op)
        if g is not None:
            check(np.abs(g.T @ g - np.eye(len(g))).max() < 1e-12 and abs(np.trace(g) - chi[i, op]) < 1e-12,
                  f"irrep {i}, operation {op}: Gamma orthogonal, its trace the character")
            matrices += int(len(g) == 2)
check(matrices == h, "E's matrices at every operation")

# The operations on orbitals: a model orbital projected onto a one-dimensional irrep with the operations and the
# characters, p = (1/h) sum_R chi(R) R c, goes to chi(R) p under every operation.
c = np.ascontiguousarray([[0.1 * ((k * 7) % 13) - 0.6 for k in range(ctx.get_n_cont())]])
images = [ctx.apply_symmetry_operation(op, c) for op in range(h)]
projected = 0
for i in range(n_irreps):
    if dimensions[i] != 1:
        continue
    p = np.ascontiguousarray(sum(chi[i, op] * images[op] for op in range(h)) / h)
    if np.abs(p).max() < 1e-8:
        continue
    for op in range(h):
        check(np.abs(ctx.apply_symmetry_operation(op, p) - chi[i, op] * p).max() < 1e-12, f"operation {op} on the irrep-{i} projection: chi times it")
    projected += 1
check(projected > 0, "a projection onto a one-dimensional irrep checked")
print("all checks passed")
