# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""The scratch need from Python: the counterpart of examples/fortran/scratch_need_demo.f90 and
examples/julia/scratch_need_demo.jl. Ammonia (test_fixtures/symmetry_demo.inp): the query from the input alone, its
breakdown per operation (the parts adding up, the largest the query), and the scratch sized from it holding the ERI
data and a restricted Fock build.

Run from the repository root, with the library built (`cargo build --release`):

    AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces python examples/python/scratch_need_demo.py
"""
import sys
import numpy as np
import aedelsten

OPERATIONS = ["eri_data", "restricted_fock", ("two_electron_gradient", 2), ("active_space_fock", 4),
              ("half_transform", 3, None, None, 1)]


def check(condition, label):
    print(("ok:   " if condition else "FAIL: ") + label)
    if not condition:
        sys.exit(1)


ctx = aedelsten.AedelstenContext("test_fixtures/symmetry_demo.inp")
# Before the shell pairs: the input alone.
query = ctx.scratch_bytes(OPERATIONS)
breakdown = ctx.scratch_breakdown(OPERATIONS)
check(len(breakdown) == len(OPERATIONS), "a need per operation")
for operation, need in zip(OPERATIONS, breakdown):
    print(f"      {operation}: {need['bytes']} bytes, {need['item']} on {need['shells']}")
    check(need["bytes"] == need["buffers"] + need["tables"] + need["row"] + need["item_bytes"],
          f"{operation}: the parts add up")
    check(need["bytes"] == ctx.scratch_bytes([operation]), f"{operation}: its need is its query")
    check(len(need["shells"]) == (4 if need["item"] == "quartet" else 2), f"{operation}: the item's shells")
check(max(need["bytes"] for need in breakdown) == query, "the query is the largest need")
check(ctx.scratch_bytes(["restricted_fock"], primitive=True) >= ctx.scratch_bytes(["restricted_fock"]),
      "the primitive blocks need at least the shells")

ctx.set_shell_pairs()
ctx.set_scratch(-(-query // 1024), "kib")
ctx.set_eri_data()
n = ctx.get_n_cont()
ctx.compute_fock_direct(np.eye(n), 1.0, -0.5)
peak = ctx.get_scratch_peak_bytes()
check(peak <= query, f"the ERI data and a Fock build fit in the query ({peak} of {query} bytes)")

# AMFI over the primitive shells: its query, measured on the atom kinds, is its peak.
amfi = ctx.scratch_bytes(["amfi"])
breakdown = ctx.scratch_breakdown(["amfi"])
check(breakdown[0]["item"] == "amfi_atom_kind" and breakdown[0]["bytes"] == amfi, "AMFI's need, set by an atom kind")
ctx.set_primitive_shell_pairs()
ctx.resize_scratch(-(-amfi // 1024), "kib")
ctx.compute_amfi_integrals()
peak = ctx.get_scratch_peak_bytes()
check(0 < peak <= amfi, f"the AMFI integrals fit in their query ({peak} of {amfi} bytes)")
print("all checks passed")
