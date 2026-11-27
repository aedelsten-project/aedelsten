# SPDX-FileCopyrightText: 2024 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""Demonstration of Aedelsten's induced-dipole (polarizable) embedding from Python; the counterpart of
examples/fortran/polarization_demo.f90 (same calls, same self-checks), eager where Fortran is not.

Run from the repository root, with the library built (`cargo build --release`):

    AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces python examples/python/polarization_demo.py
"""
import sys
import numpy as np
import aedelsten


def check(condition, label):
    print(("ok:   " if condition else "FAIL: ") + label)
    if not condition:
        sys.exit(1)


ctx = aedelsten.AedelstenContext("test_fixtures/mpi_oneint_check.inp")
ctx.set_shell_pairs()
ctx.set_scratch(64, "mib")
n = ctx.get_n_cont()

ctx.set_polarizable_sites("test_fixtures/polarization_demo.xfield")
ctx.set_polarization(threshold=1e-12, response_batch=2, state_slots=3)
print(f"n_cont = {n}, polarizable sites = {ctx.get_n_polarizable_sites()}")

i, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
d = 0.02 / (1.0 + i + j)
v, energy, iterations = ctx.compute_induction(d)
print(f"E_ind = {energy:14.6e}   ({iterations} solver iterations)")
assert ctx.get_induced_dipoles().shape == (3, 3)

x = np.stack([((3 + m) * np.minimum(i, j) + (5 + 2 * m) * np.maximum(i, j)) % 7 - 3.0 for m in (1, 2, 3)])
eps = 1e-2
v_plus, _, _ = ctx.compute_induction(np.ascontiguousarray(d + eps * x[0]))
v_minus, _, _ = ctx.compute_induction(np.ascontiguousarray(d - eps * x[0]))
finite_difference = (v_plus - v_minus) / (2 * eps)
v1 = ctx.compute_induction_response(np.ascontiguousarray(x[0]))
batch = ctx.compute_induction_response(x)
check(np.abs(v1 - finite_difference).max() < 1e-8 * np.abs(finite_difference).max() and np.abs(batch[0] - v1).max() < 1e-12,
      "response operator == finite difference of the induction operator; batch == single")
couplings = ctx.compute_induction_couplings(x)
contracted = np.array([[np.sum(x[m] * batch[k]) for k in range(len(x))] for m in range(len(x))])
check(np.abs(couplings - contracted).max() < 1e-10 * np.abs(contracted).max() and np.abs(couplings - couplings.T).max() < 1e-10 * np.abs(couplings).max(),
      "couplings == tr(D_m V1[D_n]) of the response operators; symmetric")

states = np.ascontiguousarray(d + 0.01 * x)
operators, energies = ctx.compute_induction_states(states)
v2, e2, _ = ctx.compute_induction(np.ascontiguousarray(states[1]))
check(abs(energies[1] - e2) < 1e-9 and np.abs(operators[1] - v2).max() < 1e-9,
      "state 2 == compute_induction of its own density")
_, energies_within = ctx.compute_induction_states(states, threshold=1e-12)
check(np.abs(energies_within - energies).max() < 1e-9, "compute_induction_states(threshold): the same energies")

# The effective external field: tr(d V^(i)) is minus the dipole the density's electrons induce at the sites.
eef = ctx.compute_effective_external_field()
_, _, _ = ctx.compute_induction(d)
with_d = ctx.get_induced_dipoles().sum(axis=0)
_, _, _ = ctx.compute_induction(np.zeros((n, n)))
induced = with_d - ctx.get_induced_dipoles().sum(axis=0)
traced = np.array([np.sum(d * eef[i]) for i in range(3)])
check(np.abs(traced + induced).max() < 1e-9 * np.abs(induced).max(), "EEF: tr(d V) == -(dipole induced by d)")

# The dipoles' start (another geometry's, in an optimization): from the solution, the next solve is shorter and the
# same.
_, e_solved, _ = ctx.compute_induction(d)
solution = ctx.get_induced_dipoles()
ctx.free_polarization()
ctx.set_polarization(threshold=1e-12)
_, _, from_static = ctx.compute_induction(d)
ctx.free_polarization()
ctx.set_polarization(threshold=1e-12)
ctx.set_induced_dipoles(solution)
_, e_started, from_given = ctx.compute_induction(d)
check(from_given < from_static and abs(e_started - e_solved) < 1e-10, "set_induced_dipoles: the solve starts from them")

# The field integrals kept between calls (a budget of 64 MB): the same induction.
ctx.free_polarization()
ctx.set_polarization(threshold=1e-12, resident_budget=64, resident_budget_unit=1)
_, e_kept, _ = ctx.compute_induction(d)
check(abs(e_kept - e_solved) < 1e-10, "resident field integrals: the same induction")
storage = ctx.get_polarization_storage()
check(storage["in_memory"] == storage["sites"] and 0 < storage["bytes_to_keep"] <= 64 << 20,
      "storage: all kept within the budget")

# Without a budget the integrals are evaluated in every pass (the same induction), and the storage says what keeping
# them would take.
ctx.free_polarization()
ctx.set_polarization(threshold=1e-12)
check(ctx.get_polarization_storage() == dict(storage, in_memory=0), "storage: none kept without a budget, the same need")
_, e_evaluated, _ = ctx.compute_induction(d)
check(abs(e_evaluated - e_solved) < 1e-10, "evaluated field integrals: the same induction")
# A budget set afterwards keeps what it holds: all of them here, the same induction.
ctx.set_resident_field_budget(64, "mib")
check(ctx.get_polarization_storage() == storage, "set_resident_field_budget: all kept, the same need")
_, e_later, _ = ctx.compute_induction(d)
check(abs(e_later - e_solved) < 1e-10, "field integrals kept after the polarization: the same induction")
ctx.free_polarization()
ctx.set_polarization(threshold=1e-12)

# A looser threshold for one call: fewer iterations, the energy off by about the square of the residual.
_, e_loose, loose = ctx.compute_induction(d, threshold=1e-4)
ctx.free_polarization()
ctx.set_polarization(threshold=1e-12)
_, e_tight, tight = ctx.compute_induction(d)
check(loose < tight and abs(e_loose - e_tight) < 1e-6 * abs(e_tight), "compute_induction(threshold): fewer iterations, the energy to the residual squared")

# The field at the sites: the electronic field is the field integrals contracted with the density, the same at custom
# points on the sites; the static field here is the nuclei's alone; the sites come from the file's records 0, 1, 2.
check(ctx.has_polarizable_sites() and ctx.has_polarization(), "has_polarizable_sites, has_polarization")
check(ctx.count_electric_field_sites("polarizable") == 3 and ctx.get_input_indices("polarizable") == [0, 1, 2],
      "the polarizable sites and their input records")
sites = np.array([[6.0, 0.0, 0.0], [-5.0, 4.0, 4.0], [0.0, -6.0, 3.0]])
field = ctx.compute_electronic_field(d, "polarizable", chunk_sites=2)
contracted = np.einsum("kxij,ij->kx", ctx.compute_electric_field_integrals("polarizable"), d)
at_points = ctx.compute_electronic_field(d, sites)
check(np.abs(field - contracted).max() < 1e-12 * np.abs(field).max() and np.abs(at_points - field).max() < 1e-12 * np.abs(field).max(),
      "electronic field == contracted field integrals == the field at custom points on the sites")
nuclear = ctx.compute_static_field("polarizable", core_aimps=False, embedding_aimps=False, external_multipoles=False)
check(np.array_equal(ctx.compute_static_field("polarizable"), nuclear) and np.abs(nuclear).max() > 0.0,
      "static field: only nuclei here")
check(ctx.get_input_indices(sites) == [None, None, None], "custom points come from no input record")
ctx.set_multipole_convention("exact")
check(ctx.get_multipole_convention() == "exact", "set_multipole_convention")
ctx.set_multipole_convention("contact_free")

ctx.free_polarization()
ctx.free_polarizable_sites()
check(not ctx.has_polarizable_sites() and not ctx.has_polarization(), "freed")

# The embedding AIMPs' sites: an Mg orthogonality block and two O, in input order.
aimps = aedelsten.AedelstenContext("test_fixtures/embedding_aimp_sites.inp")
z, xyz, q = aimps.get_embedding_aimp_sites()
check(list(z) == [12, 8, 8] and list(q) == [2.0, -2.0, -2.0] and np.array_equal(xyz[2], [0.5, 4.0, -2.5]),
      "embedding AIMP sites: elements, positions, charges")
check(aimps.count_electric_field_sites("embedding_aimps") == 3 and aimps.count_electric_field_sites("qm") == 3,
      "embedding AIMP and QM sites counted")
print("PASS")
