# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# The X2C Hamiltonian with the decoupling held, and its gradient, from Julia: the counterpart of
# examples/python/x2c_frozen_demo.py and examples/fortran/x2c_frozen_demo.f90. The transforms of a derived X2C
# Hamiltonian, supplied back, give the same Hamiltonian and the same frozen gradient; the gradient of a free molecule
# sums to zero over the centres.
#
# Run from the repository root, with the library built (`cargo build --release`):
#
#   AEDELSTEN_LIB=$(pwd)/target/release/libaedelsten.dylib julia examples/julia/x2c_frozen_demo.jl

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten
using Printf

# The frozen gradient of tr(D h): the kinetic part and the one potential in the transformation, the nuclei.
function frozen_gradient(ctx, density)
    set_x2c_frozen_density!(ctx, density)
    gradient = compute_x2c_frozen_kinetic_gradient(ctx) .+ compute_x2c_frozen_potential_gradient(ctx, "nuclear_attraction")
    free_x2c_frozen_density!(ctx)
    return gradient
end

function check(condition::Bool, label::AbstractString)
    println((condition ? "ok:   " : "FAIL: ") * label)
    condition || exit(1)
end

ctx = AedelstenContext("test_fixtures/mpi_oneint_check.inp")
ctx.set_shell_pairs!()
ctx.set_scratch!(size=64, unit="mib")
ctx.set_primitive_shell_pairs!()
s, t, v = (compute_one_electron_integrals(ctx, op; primitive=true) for op in ("overlap", "kinetic", "nuclear_attraction"))
pvp = compute_one_electron_picture_change_integrals(ctx, "nuclear_attraction")
h = construct_x2c_one_electron_hamiltonian(ctx, s, t, v, pvp)
n = get_n_cont(ctx)
density = [i == j ? 0.1 : 0.0 for i in 1:n, j in 1:n]
gradient = frozen_gradient(ctx, density)

u_large, u_small = get_picture_change_transforms(ctx)
set_picture_change_transforms!(ctx, u_large, u_small)
h_held = construct_x2c_one_electron_hamiltonian(ctx, s, t, v, pvp)
check(maximum(abs.(h_held .- h)) < 1e-12 * maximum(abs.(h)), "the supplied transforms give the derived Hamiltonian")
check(frozen_gradient(ctx, density) == gradient, "and the same frozen gradient")
total = maximum(abs.(sum(gradient, dims=2)))
check(total < 1e-10, @sprintf("the gradient sums to zero over the centres (%.1e)", total))

# The scalar transform of the raw planes (V, pVp) is the picture-changed nuclear attraction over the primitives; two
# planes at once are each plane's own transform.
v_2c = transform_one_electron_operator(ctx, v, pvp)
scalar, _ = compute_picture_changed_one_electron_operator(ctx, "nuclear_attraction"; contracted=false)
check(maximum(abs.(v_2c .- scalar[:, :, 1])) < 1e-12 * maximum(abs.(v_2c)), "transform_one_electron_operator(V, pVp) == the picture-changed V")
planes = transform_one_electron_operator(ctx, cat(v, s; dims=3), cat(pvp, t; dims=3))
s_2c = transform_one_electron_operator(ctx, s, t)
check(maximum(abs.(planes[:, :, 1] .- v_2c)) < 1e-13 * maximum(abs.(v_2c)) &&
      maximum(abs.(planes[:, :, 2] .- s_2c)) < 1e-13 * maximum(abs.(s_2c)), "two planes at once: each its own transform")
println("PASS")
