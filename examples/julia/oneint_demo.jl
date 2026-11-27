# SPDX-FileCopyrightText: 2024 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# User-test demonstration for Aedelsten's Julia interface (src/interfaces/Aedelsten.jl):
# creates a context from a small test input, sets up shell pairs + scratch, computes the
# overlap/kinetic/nuclear-attraction one-electron integral matrices plus a dipole (order-1
# multipole) tensor, and sanity-checks the results (overlap must be symmetric with a unit
# diagonal, so its trace must equal n_cont) -- something a user can run to confirm their own
# build of the library actually works, without needing to trust or re-derive the numbers by
# hand.
#
# Calls go through the ctx.f(...) dot-call sugar throughout (multiple dispatch's f(ctx, ...)
# form works identically -- see Aedelsten.jl -- this demo just picks one style). Every result
# is a freshly allocated, naturally shaped Julia Array ((n, n) for a plain matrix, (n, n, 3)
# for the dipole tensor): the interface is eager, unlike the Fortran one.
#
# Build the library first (from the repository root): `cargo build --release`. Then run this
# demo from the repository root, telling Julia where to find it (.so on Linux, .dll on
# Windows):
#
#   AEDELSTEN_LIB=$(pwd)/target/release/libaedelsten.dylib julia examples/julia/oneint_demo.jl
#
# (Or set DYLD_LIBRARY_PATH/LD_LIBRARY_PATH to its directory instead of AEDELSTEN_LIB, or drop
# the library at lib/libaedelsten.dylib relative to the repository root -- see
# Aedelsten.jl's _find_lib.)

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten
using Printf

function dump(label::AbstractString, buf::AbstractMatrix{Float64})::Nothing
    println("--- ", label, " ---")
    for r in axes(buf, 1)
        println(join((@sprintf("%10.5f", buf[r, c]) for c in axes(buf, 2))))
    end
    return nothing
end

function trace(buf::AbstractMatrix{Float64})::Float64
    return sum(buf[i, i] for i in axes(buf, 1))
end

function max_asymmetry(buf::AbstractMatrix{Float64})::Float64
    return maximum(abs(buf[r, c] - buf[c, r]) for r in axes(buf, 1), c in axes(buf, 2))
end

println("=== Aedelsten Julia interface demo ===")

ctx = AedelstenContext("test_fixtures/mpi_oneint_check.inp")
ctx.set_shell_pairs!()
ctx.set_scratch!(size=64, unit="mib")

n = ctx.get_n_cont()
println("n_cont = ", n)

overlap = ctx.compute_one_electron_integrals("overlap")
kinetic = ctx.compute_one_electron_integrals("kinetic")
nuclear_attraction = ctx.compute_one_electron_integrals("nuclear_attraction")
dipole = ctx.compute_one_electron_integrals("moment"; origin=(0.0, 0.0, 0.0), order=1)

dump("overlap", overlap)
dump("kinetic", kinetic)
dump("nuclear_attraction", nuclear_attraction)
dump("dipole (x component)", dipole[:, :, 1])

tr = trace(overlap)
println(@sprintf("trace(overlap)    = %12.6f", tr))
println("expected (n_cont) = ", n)

println(@sprintf("max asymmetry(overlap)            = %10.3e", max_asymmetry(overlap)))
println(@sprintf("max asymmetry(kinetic)            = %10.3e", max_asymmetry(kinetic)))
println(@sprintf("max asymmetry(nuclear_attraction) = %10.3e", max_asymmetry(nuclear_attraction)))
for i in 1:3
    println(@sprintf("max asymmetry(dipole component %d) = %10.3e", i, max_asymmetry(dipole[:, :, i])))
end

if isapprox(tr, Float64(n); atol=1.0e-8)
    println("PASS: trace(overlap) matches n_cont")
else
    println("FAIL: trace(overlap) does not match n_cont")
    exit(1)
end

# The SAP screening of both fit sets: symmetric and repulsive (positive diagonal), the two sets different.
sap_helfem = ctx.compute_one_electron_integrals("sap_helfem")
sap_grasp = ctx.compute_one_electron_integrals("oneint_sap_grasp")
for (label, w) in (("sap_helfem", sap_helfem), ("sap_grasp", sap_grasp))
    if max_asymmetry(w) < 1.0e-12 && all(w[i, i] > 0.0 for i in axes(w, 1))
        println("PASS: ", label, " is symmetric with a positive diagonal")
    else
        println("FAIL: ", label, " is not symmetric with a positive diagonal")
        exit(1)
    end
end
if maximum(abs.(sap_helfem .- sap_grasp)) > 1.0e-6
    println("PASS: the two SAP fit sets differ")
else
    println("FAIL: the two SAP fit sets agree")
    exit(1)
end

# The nuclei's interaction with an embedding AIMP and with external multipoles (test_fixtures/fock_builder_test.inp and
# its xfield; the AIMP from test_fixtures/basis, so AEDELSTEN_BASIS_LIBRARY=test_fixtures/basis), and the frees.
embedded = AedelstenContext("test_fixtures/fock_builder_test.inp")
embedded.set_external_multipoles!("test_fixtures/fock_builder_test.xfield")
aimp_energy = embedded.compute_nuclear_aimp_embedding_energy()
multipole_energy = embedded.compute_nuclear_multipole_interaction_energy()
println(@sprintf("nuclear-AIMP energy = %.10f, nuclear-multipole energy = %.10f", aimp_energy, multipole_energy))
if isfinite(aimp_energy) && aimp_energy != 0.0 && isfinite(multipole_energy) && multipole_energy != 0.0
    println("PASS: the nuclear embedding energies are finite and non-zero")
else
    println("FAIL: a nuclear embedding energy is zero or not finite")
    exit(1)
end
embedded.set_shell_pairs!()
embedded.set_scratch!(size=16, unit="mib")
embedded.compute_one_electron_integrals("overlap")
embedded.free_scratch!()
refused = try
    embedded.compute_one_electron_integrals("overlap"); false
catch
    true
end
embedded.set_scratch!(size=16, unit="mib")
again = embedded.compute_one_electron_integrals("overlap")
embedded.free_shell_pairs!()
embedded.free_primitive_shell_pairs!()
if refused && size(again, 1) == embedded.get_n_cont()
    println("PASS: free_scratch! leaves no scratch (a call is refused) and set_scratch! restores it")
else
    println("FAIL: free_scratch!/set_scratch! did not behave")
    exit(1)
end

# Every kind of centre, and the Cartesian rescaling and its inverse.
centres = AedelstenContext("test_fixtures/centres_demo.inp")
z, xyz, kinds = centres.get_centre_geometry()
m = reshape(collect(0.1:1.0:(2 * centres.get_n_cont())), centres.get_n_cont(), 2)
m0 = copy(m)
round_trip = centres.rescale_cartesian_ao_rows!("molcas", m) && centres.unrescale_cartesian_ao_rows!("molcas", m) && isapprox(m, m0; rtol=1e-12)
if centres.get_n_centres() == 3 && z == Int32[8, 1, 12] && kinds == [:real, :ghost, :orthogonality_ghost] &&
   xyz[3, 3] == 5.0 && centres.get_nuclear_charges() == [8.0, 0.0, 0.0] && round_trip
    println("PASS: the centres, their kinds and nuclear charges, and the Cartesian rescaling's inverse")
else
    println("FAIL: the centres or the Cartesian rescaling's inverse")
    exit(1)
end

# The overlap with another basis: with itself the overlap (unscreened, so to the screening threshold), with the
# centres' basis (only ctx has a scratch; it serves both) the transpose of the other order.
cross = ctx.compute_overlap_with(centres)
if maximum(abs.(ctx.compute_overlap_with(ctx) - overlap)) < 1e-10 && size(cross) == (n, centres.get_n_cont()) &&
   maximum(abs.(cross - transpose(centres.compute_overlap_with(ctx)))) < 1e-14
    println("PASS: compute_overlap_with itself is the overlap, with another basis the transpose of the other order")
else
    println("FAIL: compute_overlap_with")
    exit(1)
end

# The frees of the primitive shells, the contraction matrix and the input: each empties its own field.
ctx.set_primitive_shell_pairs!()
before = ctx.memory_breakdown()
ctx.free_primitive_shells!()
ctx.free_contraction_matrix!()
ctx.free_input!()
after = ctx.memory_breakdown()
freed = ("primitive_shells", "contraction_matrix", "qm_region")
if all(before[k] > 0 for k in freed) && all(after[k] == 0 for k in freed) &&
   after["primitive_shell_pairs"] == before["primitive_shell_pairs"]
    println("PASS: free_primitive_shells!, free_contraction_matrix! and free_input! empty their fields only")
else
    println("FAIL: free_primitive_shells!, free_contraction_matrix! or free_input!")
    exit(1)
end

# The timings and the reference list of what ran above.
timings = aedelsten_timing_report()
tree = aedelsten_span_tree()
references = aedelsten_citations()
bibtex = aedelsten_citations_bibtex()
if any(t.label == "oneint::overlap" && t.calls >= 1 for t in timings) &&
   (aedelsten_profile_enabled() || isempty(aedelsten_profile_report())) &&
   !isempty(tree) && tree[1].depth == 0 && aedelsten_mpi_collective_seconds() >= 0.0 &&
   !isempty(references) && all(!isempty(r.reference) && !isempty(r.topics) for r in references) &&
   count("\n@", bibtex) + 1 == length(references)
    println("PASS: the timings, the span tree and the reference list ($(length(references)) references) with its BibTeX")
else
    println("FAIL: the timings or the reference list")
    exit(1)
end
aedelsten_clear_citations()
if isempty(aedelsten_citations()) && isempty(aedelsten_citations_bibtex())
    println("PASS: aedelsten_clear_citations empties the list")
else
    println("FAIL: aedelsten_clear_citations")
    exit(1)
end

println("=== ctx freed automatically (GC finalizer) when it goes out of scope / at process exit ===")
