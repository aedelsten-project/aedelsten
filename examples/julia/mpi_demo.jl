# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# MPI from Julia, both routes: the library initializes MPI (`aedelsten_mpi_init`, for a program without MPI of its
# own), or the program does (MPI.jl, configured with MPIPreferences to use the same system MPI library) and passes
# its communicators by their Fortran handle. The counterpart of examples/python/mpi_demo.py and
# examples/fortran/mpi_demo.f90.
#
# Run from the repository root, with the library built with MPI (`cargo build --release --features mpi`):
#
#   export AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib
#   mpirun -n 2 julia examples/julia/mpi_demo.jl library
#   mpirun -n 2 julia --project=<an environment with MPI.jl> examples/julia/mpi_demo.jl host

include(joinpath(@__DIR__, "..", "..", "src", "interfaces", "Aedelsten.jl"))
using .Aedelsten

function check(condition::Bool, label::AbstractString)
    println(condition ? "ok:   " : "FAIL: ", label)
    condition || exit(1)
end

route = isempty(ARGS) ? "library" : ARGS[1]

# The library's version and the commit it was built from (none from a tree that was no git checkout).
version, commit = aedelsten_get_version(), aedelsten_get_source_commit()
check(occursin(r"^\d+\.\d+\.\d+$", version), "version $version")
check(commit === nothing || occursin(r"^[0-9a-f]{12}$", commit), "source commit $(something(commit, "none"))")

if route == "library"
    check(!aedelsten_mpi_initialized(), "MPI not up before aedelsten_mpi_init")
    provided = aedelsten_mpi_init("funneled")
    check(provided >= 1, "thread level provided: $provided")
    check(aedelsten_mpi_init("funneled") == provided, "a second init returns the first level")
    rank, size = aedelsten_mpi_rank(), aedelsten_mpi_size()
    check(0 <= rank < size, "rank $rank of $size")
    # ... the library's computations ...
    aedelsten_mpi_finalize()
    check(!aedelsten_mpi_initialized(), "MPI finalized")
else
    using MPI
    MPI.Init(threadlevel=:funneled)
    world = MPI.COMM_WORLD
    fortran_handle(comm) = ccall((:MPI_Comm_c2f, MPI.libmpi), Cint, (MPI.MPI_Comm,), comm)
    check(aedelsten_mpi_initialized(), "MPI up, initialized by MPI.jl")
    check(aedelsten_mpi_rank(fortran_handle(world)) == MPI.Comm_rank(world), "the same rank")
    check(aedelsten_mpi_size() == MPI.Comm_size(world), "the world by default")
    halves = MPI.Comm_split(world, MPI.Comm_rank(world) % 2, MPI.Comm_rank(world))
    check(aedelsten_mpi_size(fortran_handle(halves)) == MPI.Comm_size(halves), "a sub-communicator by its handle")
    # A context's work split over a sub-communicator, then over the world: the overlap as on this process alone.
    alone = AedelstenContext("test_fixtures/mpi_oneint_check.inp")
    split = AedelstenContext("test_fixtures/mpi_oneint_check.inp"; comm=fortran_handle(halves))
    for ctx in (alone, split)
        set_shell_pairs!(ctx)
        set_scratch!(ctx; size=64, unit="mib")
    end
    overlap = compute_one_electron_integrals(alone, "overlap")
    check(compute_one_electron_integrals(split, "overlap") == overlap, "a context over a sub-communicator")
    set_communicator!(split, fortran_handle(world))
    check(compute_one_electron_integrals(split, "overlap") == overlap, "the context over the world")
    Aedelsten._free!(alone)
    Aedelsten._free!(split)
    # The half transform by the node's ranks into one X in their shared memory: the summed X to rounding.
    ctx = AedelstenContext("test_fixtures/mpi_eri_check.inp"; comm=fortran_handle(world))
    set_shell_pairs!(ctx)
    set_scratch!(ctx; size=64, unit="mib")
    set_eri_data!(ctx)
    n, m = get_n_cont(ctx), 3
    c = [0.1 * ((mu * 7 + u * 3 + 1) % 5) - 0.2 for mu in 0:n-1, u in 0:m-1]
    summed = compute_eri_half_transformed(ctx, c, c)
    shared = AedelstenSharedArray(m * n * m * n; comm=fortran_handle(world))
    on_node = compute_eri_half_transformed_on_node!(ctx, reshape(shared.array, m, n, m, n), c, c)
    check(maximum(abs.(on_node .- summed)) < 1e-12, "X in the node's shared memory")
    free_shared!(shared)
    Aedelsten._free!(ctx)
    aedelsten_mpi_finalize()
    check(aedelsten_mpi_initialized(), "the program's MPI left up by aedelsten_mpi_finalize")
    MPI.free(halves)
    MPI.Finalize()
end
