# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# The AO values and their derivatives from Julia: the counterpart of examples/python/ao_values_demo.py and
# examples/fortran/ao_values_demo.f90. compute_ao_hessians gives the values and gradients compute_ao_values gives, and
# second derivatives that agree with central differences of the gradients.
#
# Run from the repository root, with the library built (`cargo build --release`):
#
#   AEDELSTEN_LIB=$(pwd)/target/release/libaedelsten.dylib julia examples/julia/ao_values_demo.jl

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten
using Printf

function check(condition::Bool, label::AbstractString)
    println((condition ? "ok:   " : "FAIL: ") * label)
    condition || exit(1)
end

ctx = AedelstenContext("test_fixtures/mpi_oneint_check.inp")
shells = collect(0:get_n_shells(ctx) - 1)
t = collect(0.0:39.0)
points = permutedims(hcat(sin.(0.7 .* t) .* 1.8, cos.(1.3 .* t) .* 1.5, 0.11 .* t .- 2.0))

values, gradients = compute_ao_values(ctx, shells, points; gradients=true)
h_values, h_gradients, hessians = compute_ao_hessians(ctx, shells, points)
check(values == h_values && gradients == h_gradients, "compute_ao_hessians gives compute_ao_values's values and gradients")

step, worst = 1e-5, 0.0
for (k, (a, b)) in enumerate(((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)))
    shift = zeros(3, 1)
    shift[b] = step
    plus = compute_ao_values(ctx, shells, points .+ shift; gradients=true)[2][:, :, a]
    minus = compute_ao_values(ctx, shells, points .- shift; gradients=true)[2][:, :, a]
    h = hessians[:, :, k]
    global worst = max(worst, maximum(abs.(h .- (plus .- minus) ./ (2step)) ./ (1 .+ abs.(h))))
end
check(worst < 1e-6, @sprintf("the second derivatives against central differences of the gradients (worst %.1e)", worst))
println("PASS")
