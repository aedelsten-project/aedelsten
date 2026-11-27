# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# The active-space walks from Julia: the counterpart of examples/python/active_space_demo.py and
# examples/fortran/active_space_demo.f90. The Coulomb matrices of the active pair densities M from the direct walk,
# one basis and two, against M from the stored S1 tensor, T from M (transform_pair_coulomb_one_index) against T
# from the S1 tensor, J/K of several densities of no symmetry in one walk against the S1 tensor and one walk each, and
# the half-transformed ERIs by batches of sigma (each shell alone), one basis and two, against the S1 tensor.
#
# Build the library first (from the repository root): `cargo build --release`. Then, from the repository root:
#
#   AEDELSTEN_LIB=$(pwd)/target/release/libaedelsten.dylib julia examples/julia/active_space_demo.jl

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten
using Printf

function close(x, y, label)
    scale = maximum(abs, y)
    difference = maximum(abs.(x .- y))
    ok = scale > 1e-2 && difference < 1e-12 * scale
    @printf("%s %s (max %.3e, difference %.3e)\n", ok ? "ok:  " : "FAIL:", label, scale, difference)
    ok || exit(1)
end

ctx = AedelstenContext("test_fixtures/mpi_eri_check.inp")
ctx.set_shell_pairs!()
ctx.set_scratch!(size=64, unit="mib")
ctx.set_eri_data!()
n = ctx.get_n_cont()
g = ctx.compute_eri_stored()  # g[mu, nu, lambda, sigma]
# The packed column of the function pair (mu, nu), 1-based.
column(mu, nu) = (q = min(mu, nu) - 1; q * n - q * (q - 1) ÷ 2 + abs(mu - nu) + 1)

m_act = 3
c = [0.2 * ((mu * 5 + u * 3 + 1) % 7) - 0.55 for mu in 0:n-1, u in 0:m_act-1]
c_bra = [0.2 * ((mu * 3 + u * 5 + 2) % 7) - 0.6 for mu in 0:n-1, u in 0:m_act-1]
density = [0.05 * ((min(i, j) * 7 + max(i, j) * 3) % 11) - 0.1 for i in 0:n-1, j in 0:n-1]

# M(pair, mu, nu) from the S1 tensor for pair densities p[lambda, sigma, pair].
function dense_m(p)
    q = size(p, 3)
    return [sum(g[mu, nu, l, s] * p[l, s, x] for l in 1:n, s in 1:n) for x in 1:q, mu in 1:n, nu in 1:n]
end
unpack(m) = [m[x, column(mu, nu)] for x in axes(m, 1), mu in 1:n, nu in 1:n]
# T[u, pair, mu] = sum_nu c[nu, u] M(pair, mu, nu).
dense_t(m_full) = [sum(c[nu, u] * m_full[x, mu, nu] for nu in 1:n) for u in 1:m_act, x in axes(m_full, 1), mu in 1:n]

# One basis: the pairs v >= w at v (v - 1)/2 + w (1-based).
_, _, _, _, m = ctx.compute_fock_direct_active_space(density, density, c, 1.0, -0.5, 1.0, -0.5)
pairs = [(v, w) for v in 1:m_act for w in 1:v]
p = [(c[l, v] * c[s, w] + c[s, v] * c[l, w]) / 2 for l in 1:n, s in 1:n, (v, w) in pairs]
m_ref = dense_m(p)
close(unpack(m), m_ref, "M, one basis, against the S1 tensor")
close(ctx.transform_pair_coulomb_one_index(m, c), dense_t(m_ref), "T from M, one basis")

# Two bases: every pair (v, w) at v + (w - 1) n_active.
_, _, m_mixed = ctx.compute_fock_direct_mixed_active_space(density, c_bra, c, 1.0, -0.5)
p = [(c_bra[l, v] * c[s, w] + c_bra[s, v] * c[l, w]) / 2 for l in 1:n, s in 1:n, v in 1:m_act, w in 1:m_act]
m_ref = dense_m(reshape(p, n, n, m_act * m_act))
close(unpack(m_mixed), m_ref, "M, two bases, against the S1 tensor")
close(ctx.transform_pair_coulomb_one_index(m_mixed, c), dense_t(m_ref), "T from M, two bases")

# J and K of two densities of no symmetry in one walk, against the S1 tensor and against one walk each.
densities = cat([density .+ [0.03 * s * ((i * 5 + j * 2) % 7 - 3) for i in 0:n-1, j in 0:n-1] for s in (1.0, -2.0)]...; dims=3)
j_multi, k_multi = ctx.compute_fock_direct_general_multi(densities, 1.0, -0.5)
for index in 1:2
    d = densities[:, :, index]
    close(j_multi[:, :, index], [sum(g[a, b, c, e] * (d[c, e] + d[e, c]) / 2 for c in 1:n, e in 1:n) for a in 1:n, b in 1:n], "J of density $index of two, against the S1 tensor")
    close(k_multi[:, :, index], [-0.5 * sum(g[a, b, c, e] * d[b, e] for b in 1:n, e in 1:n) for a in 1:n, c in 1:n], "K of density $index of two, against the S1 tensor")
    j_one, k_one = ctx.compute_fock_direct_general(d, 1.0, -0.5)
    ok = j_one == j_multi[:, :, index] && k_one == k_multi[:, :, index]
    println(ok ? "ok:   " : "FAIL: ", "J/K of density $index of two, equal to its own walk")
    ok || exit(1)
end
# The half-transformed ERIs by batches of sigma, x[p, nu, r, s]: every shell alone, one basis (the same matrix on
# both sides) and two, against the S1 tensor; the scratch they take beyond the walk's own.
starts = vcat(sort(Int.(ctx.get_shells()[:, 3])), n)
for (label, left, right) in (("one basis", c, c), ("two bases", c_bra, c))
    x_ref = [sum(left[mu, p] * right[l, r] * g[mu, nu, l, s] for mu in 1:n, l in 1:n) for p in 1:m_act, nu in 1:n, r in 1:m_act, s in 1:n]
    batches = [ctx.compute_eri_half_transformed_sigma(left, right, starts[k], starts[k + 1]) for k in 1:length(starts)-1]
    close(cat(batches...; dims=4), x_ref, "X by batches of sigma, $label, against the S1 tensor")
end
one, two = ctx.half_transform_scratch_bytes(m_act, m_act, true, n), ctx.half_transform_scratch_bytes(m_act, m_act, false, n)
0 < one < two || (println("FAIL: the half transform's scratch, one basis $one and two $two bytes"); exit(1))
println("ok:   the half transform's scratch, one basis $one and two $two bytes")
println("PASS")
