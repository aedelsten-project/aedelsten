# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# The two-electron gradient from Julia: the counterpart of examples/python/gradient_demo.py and
# examples/fortran/gradient_demo.f90. The restricted-HF terms of a model density on water, all centres at once and with
# every other centre frozen (each the same column, the others zero), and the translational sum rule: every basis
# function sits on a gradient centre, so the integral derivatives over all centres sum to zero.
#
# Build the library first (from the repository root): `cargo build --release`. Then, from the repository root:
#
#   AEDELSTEN_LIB=$(pwd)/target/release/libaedelsten.dylib julia examples/julia/gradient_demo.jl

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten
using Printf

function check(condition, label)
    println((condition ? "ok:   " : "FAIL: ") * label)
    condition || exit(1)
end

ctx = AedelstenContext("test_fixtures/mpi_eri_check.inp")
ctx.set_shell_pairs!()
ctx.set_scratch!(size=64, unit="mib")
ctx.set_eri_data!()
n = ctx.get_n_cont()
n_centres = ctx.get_n_gradient_centres()
p = [0.05 * mod(min(i, j) * 7 + max(i, j) * 3, 11) - 0.1 for i in 0:n-1, j in 0:n-1]
densities = reshape(p, n, n, 1)
terms = [("coulomb", 1, 1, 1.0), ("exchange", 1, 1, -0.5)]

gradient = ctx.compute_two_electron_gradient(densities, terms)
scale = maximum(abs, gradient)
@printf("n_cont = %d, gradient centres = %d, largest entry = %.6e\n", n, n_centres, scale)
check(size(gradient) == (3, n_centres) && scale > 1e-3, "the gradient, one column per centre")
check(maximum(abs, sum(gradient; dims=2)) < 1e-10 * scale, "translational sum rule: the columns sum to zero")
for centre in 0:n_centres-1
    one = ctx.compute_two_electron_gradient(densities, terms; active=[k == centre for k in 0:n_centres-1])
    others = one[:, setdiff(1:n_centres, centre + 1)]
    check(maximum(abs.(one[:, centre + 1] - gradient[:, centre + 1])) < 1e-12 * scale && all(iszero, others),
          "centre $centre alone active: its column, the others zero")
end
refused = try
    ctx.compute_two_electron_gradient(densities, terms; active=fill(true, n_centres + 1)); false
catch
    true
end
check(refused, "a mask of the wrong length is refused")
println("PASS")
