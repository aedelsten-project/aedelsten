# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# The scratch need from Julia: the counterpart of examples/python/scratch_need_demo.py and
# examples/fortran/scratch_need_demo.f90. Ammonia (test_fixtures/symmetry_demo.inp): the query from the input alone,
# its breakdown per operation (the parts adding up, the largest the query), and the scratch sized from it holding
# the ERI data and a restricted Fock build.
#
# Run from the repository root, with the library built (`cargo build --release`):
#
#   AEDELSTEN_LIB=$(pwd)/target/release/libaedelsten.dylib julia examples/julia/scratch_need_demo.jl

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten
using LinearAlgebra

const OPERATIONS = [:eri_data, :restricted_fock, (:two_electron_gradient, 2), (:active_space_fock, 4),
                    (:half_transform, 3, nothing, nothing, 1)]

function check(condition::Bool, label::String)
    println(condition ? "ok:   " : "FAIL: ", label)
    condition || exit(1)
end

ctx = AedelstenContext("test_fixtures/symmetry_demo.inp")
# Before the shell pairs: the input alone.
query = scratch_bytes(ctx, OPERATIONS)
breakdown = scratch_breakdown(ctx, OPERATIONS)
check(length(breakdown) == length(OPERATIONS), "a need per operation")
for (operation, need) in zip(OPERATIONS, breakdown)
    println("      $operation: $(need[:bytes]) bytes, $(need[:item]) on $(need[:shells])")
    check(need[:bytes] == need[:buffers] + need[:tables] + need[:row] + need[:item_bytes], "$operation: the parts add up")
    check(need[:bytes] == scratch_bytes(ctx, [operation]), "$operation: its need is its query")
    check(length(need[:shells]) == (need[:item] == :quartet ? 4 : 2), "$operation: the item's shells")
end
check(maximum(need[:bytes] for need in breakdown) == query, "the query is the largest need")
check(scratch_bytes(ctx, [:restricted_fock]; primitive=true) >= scratch_bytes(ctx, [:restricted_fock]),
      "the primitive blocks need at least the shells")

set_shell_pairs!(ctx)
set_scratch!(ctx; size=cld(query, 1024), unit="kib")
set_eri_data!(ctx)
n = get_n_cont(ctx)
compute_fock_direct(ctx, Matrix{Float64}(I, n, n), 1.0, -0.5)
peak = get_scratch_peak_bytes(ctx)
check(peak <= query, "the ERI data and a Fock build fit in the query ($peak of $query bytes)")

# AMFI over the primitive shells: its query, measured on the atom kinds, is its peak.
amfi = scratch_bytes(ctx, [:amfi])
amfi_need = scratch_breakdown(ctx, [:amfi])[1]
check(amfi_need[:item] == :amfi_atom_kind && amfi_need[:bytes] == amfi, "AMFI's need, set by an atom kind")
set_primitive_shell_pairs!(ctx)
resize_scratch!(ctx; size=cld(amfi, 1024), unit="kib")
compute_amfi_integrals(ctx)
peak = get_scratch_peak_bytes(ctx)
check(0 < peak <= amfi, "the AMFI integrals fit in their query ($peak of $amfi bytes)")
println("all checks passed")
