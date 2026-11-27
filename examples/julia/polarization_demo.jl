# SPDX-FileCopyrightText: 2024 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# Demonstration of Aedelsten's induced-dipole (polarizable) embedding from Julia; the counterpart of
# examples/fortran/polarization_demo.f90 (same calls, same self-checks), eager where Fortran is not.
#
#   AEDELSTEN_LIB=$(pwd)/target/release/libaedelsten.dylib julia examples/julia/polarization_demo.jl
#
# Run from the repository root.

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten
using Printf

function check(condition::Bool, label::AbstractString)
    println(condition ? "ok:   " : "FAIL: ", label)
    condition || exit(1)
end

ctx = AedelstenContext("test_fixtures/mpi_oneint_check.inp")
ctx.set_shell_pairs!()
ctx.set_scratch!(size=64, unit="mib")
n = ctx.get_n_cont()

ctx.set_polarizable_sites!("test_fixtures/polarization_demo.xfield")
ctx.set_polarization!(threshold=1e-12, response_batch=2, state_slots=3)   # or set_polarization!(ctx, PolarizationOptions(...))
@printf("n_cont = %d, polarizable sites = %d\n", n, ctx.get_n_polarizable_sites())

d = [0.02 / (1 + i + j) for i in 1:n, j in 1:n]
v, energy, iterations = ctx.compute_induction(d)
@printf("E_ind = %14.6e   (%d solver iterations)\n", energy, iterations)
@assert size(ctx.get_induced_dipoles()) == (3, 3)

x = Array{Float64}(undef, n, n, 3)
for m in 1:3, j in 1:n, i in 1:n
    x[i, j, m] = mod((3 + m) * min(i, j) + (5 + 2m) * max(i, j), 7) - 3
end
ε = 1e-2
v_plus, _ = ctx.compute_induction(d .+ ε .* x[:, :, 1])
v_minus, _ = ctx.compute_induction(d .- ε .* x[:, :, 1])
finite_difference = (v_plus .- v_minus) ./ (2ε)
v1 = ctx.compute_induction_response(x[:, :, 1])
batch = ctx.compute_induction_response(x)
check(maximum(abs.(v1 .- finite_difference)) < 1e-8 * maximum(abs.(finite_difference)) && maximum(abs.(batch[:, :, 1] .- v1)) < 1e-12,
      "response operator == finite difference of the induction operator; batch == single")
couplings = ctx.compute_induction_couplings(x)
contracted = [sum(x[:, :, m] .* batch[:, :, k]) for m in 1:size(x, 3), k in 1:size(x, 3)]
check(maximum(abs.(couplings .- contracted)) < 1e-10 * maximum(abs.(contracted)) && maximum(abs.(couplings .- couplings')) < 1e-10 * maximum(abs.(couplings)),
      "couplings == tr(D_m V1[D_n]) of the response operators; symmetric")

states = cat((d .+ 0.01 .* x[:, :, m] for m in 1:3)...; dims=3)
operators, energies = ctx.compute_induction_states(states)
v2, e2, _ = ctx.compute_induction(states[:, :, 2])
check(abs(energies[2] - e2) < 1e-9 && maximum(abs.(operators[:, :, 2] .- v2)) < 1e-9,
      "state 2 == compute_induction of its own density")
_, energies_within = ctx.compute_induction_states(states; threshold=1e-12)
check(maximum(abs.(energies_within .- energies)) < 1e-9, "compute_induction_states(threshold): the same energies")

# The effective external field: tr(d V^(i)) is minus the dipole the density's electrons induce at the sites.
eef = ctx.compute_effective_external_field()
ctx.compute_induction(d)
with_d = vec(sum(ctx.get_induced_dipoles(); dims=2))
ctx.compute_induction(zeros(n, n))
induced = with_d .- vec(sum(ctx.get_induced_dipoles(); dims=2))
traced = [sum(d .* eef[:, :, i]) for i in 1:3]
check(maximum(abs.(traced .+ induced)) < 1e-9 * maximum(abs.(induced)), "EEF: tr(d V) == -(dipole induced by d)")

# The dipoles' start (another geometry's, in an optimization): from the solution, the next solve is shorter and the
# same.
_, e_solved, _ = ctx.compute_induction(d)
solution = ctx.get_induced_dipoles()
ctx.free_polarization!()
ctx.set_polarization!(threshold=1e-12)
_, _, from_static = ctx.compute_induction(d)
ctx.free_polarization!()
ctx.set_polarization!(threshold=1e-12)
ctx.set_induced_dipoles!(solution)
_, e_started, from_given = ctx.compute_induction(d)
check(from_given < from_static && abs(e_started - e_solved) < 1e-10, "set_induced_dipoles: the solve starts from them")

# The field integrals kept between calls (a budget of 64 MB): the same induction.
ctx.free_polarization!()
ctx.set_polarization!(threshold=1e-12, resident_budget=64, resident_budget_unit=1)
_, e_kept, _ = ctx.compute_induction(d)
check(abs(e_kept - e_solved) < 1e-10, "resident field integrals: the same induction")
storage = ctx.get_polarization_storage()
check(storage.in_memory == storage.sites && 0 < storage.bytes_to_keep <= 64 << 20,
      "storage: all kept within the budget")

# Without a budget the integrals are evaluated in every pass (the same induction), and the storage says what keeping
# them would take.
ctx.free_polarization!()
ctx.set_polarization!(threshold=1e-12)
check(ctx.get_polarization_storage() == merge(storage, (in_memory=0,)), "storage: none kept without a budget, the same need")
_, e_evaluated, _ = ctx.compute_induction(d)
check(abs(e_evaluated - e_solved) < 1e-10, "evaluated field integrals: the same induction")
# A budget set afterwards keeps what it holds: all of them here, the same induction.
ctx.set_resident_field_budget!(budget=64, unit="mib")
check(ctx.get_polarization_storage() == storage, "set_resident_field_budget!: all kept, the same need")
_, e_later, _ = ctx.compute_induction(d)
check(abs(e_later - e_solved) < 1e-10, "field integrals kept after the polarization: the same induction")
ctx.free_polarization!()
ctx.set_polarization!(threshold=1e-12)

# A looser threshold for one call: fewer iterations, the energy off by about the square of the residual.
_, e_loose, loose = ctx.compute_induction(d; threshold=1e-4)
ctx.free_polarization!()
ctx.set_polarization!(threshold=1e-12)
_, e_tight, tight = ctx.compute_induction(d)
check(loose < tight && abs(e_loose - e_tight) < 1e-6 * abs(e_tight), "compute_induction(threshold): fewer iterations, the energy to the residual squared")

# The field at the sites: the electronic field is the field integrals contracted with the density, the same at custom
# points on the sites; the static field here is the nuclei's alone; the sites come from the file's records 0, 1, 2.
check(ctx.has_polarizable_sites() && ctx.has_polarization(), "has_polarizable_sites, has_polarization")
check(ctx.count_electric_field_sites(:polarizable) == 3 && ctx.get_input_indices(:polarizable) == [0, 1, 2],
      "the polarizable sites and their input records")
sites = [6.0 -5.0 0.0; 0.0 4.0 -6.0; 0.0 4.0 3.0]
field = ctx.compute_electronic_field(d, :polarizable; chunk_sites=2)
integrals = ctx.compute_electric_field_integrals(:polarizable)
contracted = [sum(integrals[:, :, x, k] .* d) for x in 1:3, k in 1:3]
at_points = ctx.compute_electronic_field(d, sites)
check(maximum(abs.(field - contracted)) < 1e-12 * maximum(abs.(field)) && maximum(abs.(at_points - field)) < 1e-12 * maximum(abs.(field)),
      "electronic field == contracted field integrals == the field at custom points on the sites")
nuclear = ctx.compute_static_field(:polarizable; core_aimps=false, embedding_aimps=false, external_multipoles=false)
check(ctx.compute_static_field(:polarizable) == nuclear && maximum(abs.(nuclear)) > 0.0, "static field: only nuclei here")
check(all(isnothing, ctx.get_input_indices(sites)), "custom points come from no input record")
ctx.set_multipole_convention!(:exact)
check(ctx.get_multipole_convention() == :exact, "set_multipole_convention!")
ctx.set_multipole_convention!(:contact_free)

ctx.free_polarization!()
ctx.free_polarizable_sites!()
check(!ctx.has_polarizable_sites() && !ctx.has_polarization(), "freed")

# The embedding AIMPs' sites: an Mg orthogonality block and two O, in input order.
aimps = AedelstenContext("test_fixtures/embedding_aimp_sites.inp")
z, xyz, q = aimps.get_embedding_aimp_sites()
check(z == Int32[12, 8, 8] && q == [2.0, -2.0, -2.0] && xyz[:, 3] == [0.5, 4.0, -2.5],
      "embedding AIMP sites: elements, positions, charges")
check(aimps.count_electric_field_sites(:embedding_aimps) == 3 && aimps.count_electric_field_sites(:qm) == 3,
      "embedding AIMP and QM sites counted")
println("PASS")
