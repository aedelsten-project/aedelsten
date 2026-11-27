# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""One-electron integrals from Python: the counterpart of examples/fortran/oneint_demo.f90 and
examples/julia/oneint_demo.jl (the overlap's trace, the SAP screening of both fit sets).

Run from the repository root, with the library built (`cargo build --release`):

    AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces python examples/python/oneint_demo.py
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
print(f"n_cont = {n}")

overlap = ctx.compute_one_electron_integrals("overlap")
check(abs(np.trace(overlap) - n) < 1e-8, "trace(overlap) == n_cont")

sap_helfem = ctx.compute_one_electron_integrals("sap_helfem")
sap_grasp = ctx.compute_one_electron_integrals("oneint_sap_grasp")
for label, w in (("sap_helfem", sap_helfem), ("sap_grasp", sap_grasp)):
    check(np.abs(w - w.T).max() < 1e-12 and (np.diag(w) > 0.0).all(), f"{label} is symmetric with a positive diagonal")
check(np.abs(sap_helfem - sap_grasp).max() > 1e-6, "the two SAP fit sets differ")

# Every kind of centre (the AIMP from test_fixtures/basis: AEDELSTEN_BASIS_LIBRARY=test_fixtures/basis).
centres = aedelsten.AedelstenContext("test_fixtures/centres_demo.inp")
z, xyz, kinds = centres.get_centre_geometry()
check(centres.get_n_centres() == 3 and list(z) == [8, 1, 12] and kinds == ["real", "ghost", "orthogonality_ghost"], "the centres and their kinds")
check(abs(xyz[2, 2] - 5.0) < 1e-12, "the orthogonality ghost's position")
check(list(centres.get_nuclear_charges()) == [8.0, 0.0, 0.0], "only the real atom has a nucleus")
m = np.arange(2.0 * centres.get_n_cont()).reshape(2, -1) + 0.1
m0 = m.copy()
check(centres.rescale_cartesian_ao_rows("molcas", m) and centres.unrescale_cartesian_ao_rows("molcas", m) and np.allclose(m, m0, rtol=1e-12), "the Cartesian rescaling and its inverse")

# The overlap with another basis: with itself the overlap (unscreened, so to the screening threshold), with the
# centres' basis (only ctx has a scratch; it serves both) the transpose of the other order.
check(np.abs(ctx.compute_overlap_with(ctx) - overlap).max() < 1e-10, "compute_overlap_with itself == the overlap")
cross = ctx.compute_overlap_with(centres)
check(cross.shape == (centres.get_n_cont(), n) and np.abs(cross - centres.compute_overlap_with(ctx).T).max() < 1e-14,
      "compute_overlap_with another basis: the transpose of the other order")

# The frees of the primitive shells, the contraction matrix and the input: each empties its own field.
ctx.set_primitive_shell_pairs()
before = ctx.memory_breakdown()
check(all(before[k] > 0 for k in ("primitive_shells", "contraction_matrix", "qm_region")), "resident before the frees")
ctx.free_primitive_shells()
ctx.free_contraction_matrix()
ctx.free_input()
after = ctx.memory_breakdown()
check(all(after[k] == 0 for k in ("primitive_shells", "contraction_matrix", "qm_region"))
      and after["primitive_shell_pairs"] == before["primitive_shell_pairs"],
      "free_primitive_shells, free_contraction_matrix and free_input empty their fields only")

# The timings and the reference list of what ran above.
timings = aedelsten.aedelsten_timing_report()
check(any(label == "oneint::overlap" and calls >= 1 for label, _, calls in timings), "the timing report has the overlap")
check(aedelsten.aedelsten_profile_enabled() or aedelsten.aedelsten_profile_report() == [], "no fine timings without the profile feature")
tree = aedelsten.aedelsten_span_tree()
check(len(tree) > 0 and tree[0][1] == 0 and aedelsten.aedelsten_mpi_collective_seconds() >= 0.0, "the span tree")
references = aedelsten.aedelsten_citations()
bibtex = aedelsten.aedelsten_citations_bibtex()
check(len(references) > 0 and all(text and topics for text, topics in references) and bibtex.count("\n@") + 1 == len(references),
      f"the reference list ({len(references)} references) and its BibTeX")
aedelsten.aedelsten_clear_citations()
check(aedelsten.aedelsten_citations() == [] and aedelsten.aedelsten_citations_bibtex() == "", "cleared")
print("PASS")
