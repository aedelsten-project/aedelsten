# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# The point group's operations from Julia: the counterpart of examples/python/symmetry_demo.py and
# examples/fortran/symmetry_demo.f90. Ammonia turned out of every axis (test_fixtures/symmetry_demo.inp), C3v: every
# operation's matrix orthogonal, its images a permutation of equivalent centres, every vector between two centres
# carried onto the vector between their images, one identity.
# The irreps: the AO counts, the characters orthonormal, the irrep matrices orthogonal with the characters as
# their traces, a projection onto a one-dimensional irrep carried to its character times itself.
# Indices from the library are 0-based.
#
# Run from the repository root, with the library built (`cargo build --release`):
#
#   AEDELSTEN_LIB=$(pwd)/target/release/libaedelsten.dylib julia examples/julia/symmetry_demo.jl

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten
using LinearAlgebra

# The fixture's positions (bohr, one column per centre) and elements.
const POSITIONS = [0.4000000000000000 2.0909455857523227 -0.8984653640277757 0.0075197782754526;
                   -1.1837482993408999 -0.2618653840665375 0.2580710345499226 -1.9820015685184955;
                   2.3994294843469834 2.1168787516457970 2.5548151556260290 0.6680280306678168]
const ELEMENTS = ["N", "H", "H", "H"]

function check(condition::Bool, label::String)
    println(condition ? "ok:   " : "FAIL: ", label)
    condition || exit(1)
end

ctx = AedelstenContext("test_fixtures/symmetry_demo.inp")
set_shell_pairs!(ctx)
set_scratch!(ctx; size=64, unit="mib")
set_symmetry!(ctx)
check(get_point_group_name(ctx) == "C3v", "the point group is C3v")
h = get_symmetry_operation_count(ctx)
check(h == 6, "six operations")
n = get_n_symmetry_centres(ctx)
check(n == size(POSITIONS, 2), "four centres")

identities = 0
for op in 0:h-1
    m = get_symmetry_operation_matrix(ctx, op)
    images = get_symmetry_centre_images(ctx, op) .+ 1
    check(maximum(abs.(m' * m - I)) < 1e-12, "operation $op is orthogonal")
    check(sort(images) == collect(1:n) && all(ELEMENTS[a] == ELEMENTS[images[a]] for a in 1:n), "operation $op permutes equivalent centres")
    carried = maximum(maximum(abs.(m * (POSITIONS[:, a] - POSITIONS[:, b]) - (POSITIONS[:, images[a]] - POSITIONS[:, images[b]]))) for a in 1:n, b in 1:n)
    check(carried < 1e-8, "operation $op carries every vector between centres onto its images'")
    global identities += (images == collect(1:n) && maximum(abs.(m - I)) < 1e-12) ? 1 : 0
end
check(identities == 1, "one identity")

# The irreps: AO functions in each, together all; the characters orthonormal over the group; each aligned irrep
# matrix orthogonal with the character as its trace.
dimensions = [d for (_, d) in get_irreps(ctx)]
n_irreps = length(dimensions)
check(sum(get_ao_irrep_counts(ctx)) == get_n_cont(ctx), "every AO function in one irrep")
chi = [get_irrep_character(ctx, i, op) for i in 0:n_irreps-1, op in 0:h-1]
check(maximum(abs.(chi * chi' / h - I)) < 1e-12, "the characters are orthonormal")
matrices = 0
for i in 0:n_irreps-1, op in 0:h-1
    g = get_irrep_matrix(ctx, i, op)
    g === nothing && continue
    check(maximum(abs.(g' * g - I)) < 1e-12 && abs(sum(g[k, k] for k in axes(g, 1)) - chi[i + 1, op + 1]) < 1e-12,
          "irrep $i, operation $op: Gamma orthogonal, its trace the character")
    global matrices += size(g, 1) == 2 ? 1 : 0
end
check(matrices == h, "E's matrices at every operation")

# The operations on orbitals: a model orbital projected onto a one-dimensional irrep with the operations and the
# characters, p = (1/h) sum_R chi(R) R c, goes to chi(R) p under every operation.
c = reshape([0.1 * mod((k - 1) * 7, 13) - 0.6 for k in 1:get_n_cont(ctx)], :, 1)
images = [apply_symmetry_operation(ctx, op, c) for op in 0:h-1]
projected = 0
for i in 1:n_irreps
    dimensions[i] == 1 || continue
    p = sum(chi[i, op] * images[op] for op in 1:h) / h
    maximum(abs, p) < 1e-8 && continue
    for op in 0:h-1
        check(maximum(abs.(apply_symmetry_operation(ctx, op, p) - chi[i, op + 1] * p)) < 1e-12, "operation $op on the irrep-$(i - 1) projection: chi times it")
    end
    global projected += 1
end
check(projected > 0, "a projection onto a one-dimensional irrep checked")
println("all checks passed")
