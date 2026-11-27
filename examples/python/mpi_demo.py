# SPDX-FileCopyrightText: 2026 Ernst D. Larsson
# SPDX-License-Identifier: MIT

"""MPI from Python, both routes: the library initializes MPI (`aedelsten_mpi_init`, for a program without MPI of its
own), or the program does (mpi4py, built against the same MPI library) and passes its communicators by their Fortran
handle. The counterpart of examples/julia/mpi_demo.jl and examples/fortran/mpi_demo.f90.

Run from the repository root, with the library built with MPI (`cargo build --release --features mpi`):

    export AEDELSTEN_LIB=$PWD/target/release/libaedelsten.dylib PYTHONPATH=src/interfaces
    mpirun -n 2 python examples/python/mpi_demo.py library
    mpirun -n 2 python examples/python/mpi_demo.py host
"""
import sys
import numpy as np
import aedelsten


def check(condition, label):
    print(("ok:   " if condition else "FAIL: ") + label)
    if not condition:
        sys.exit(1)


route = sys.argv[1] if len(sys.argv) > 1 else "library"
c = aedelsten.load_aedelsten_lib() and aedelsten._get_constants()

# The library's version and the commit it was built from (none from a tree that was no git checkout).
version, commit = aedelsten.aedelsten_get_version(), aedelsten.aedelsten_get_source_commit()
check(len(version.split(".")) == 3 and all(p.isdigit() for p in version.split(".")), f"version {version}")
check(commit is None or (len(commit) == 12 and all(ch in "0123456789abcdef" for ch in commit)), f"source commit {commit}")

if route == "library":
    check(not aedelsten.aedelsten_mpi_initialized(), "MPI not up before aedelsten_mpi_init")
    provided = aedelsten.aedelsten_mpi_init("funneled")
    check(provided >= c.mpi_thread_funneled, f"thread level provided: {provided}")
    check(aedelsten.aedelsten_mpi_init("funneled") == provided, "a second init returns the first level")
    rank, size = aedelsten.aedelsten_mpi_rank(), aedelsten.aedelsten_mpi_size()
    check(0 <= rank < size, f"rank {rank} of {size}")
    # ... the library's computations ...
    aedelsten.aedelsten_mpi_finalize()
    check(not aedelsten.aedelsten_mpi_initialized(), "MPI finalized")
else:
    from mpi4py import MPI  # initializes MPI, finalized at exit by mpi4py

    world = MPI.COMM_WORLD
    check(aedelsten.aedelsten_mpi_initialized(), "MPI up, initialized by mpi4py")
    check(aedelsten.aedelsten_mpi_thread_level() == MPI.Query_thread(), "the same thread level")
    check(aedelsten.aedelsten_mpi_rank(world.py2f()) == world.Get_rank(), "the same rank")
    check(aedelsten.aedelsten_mpi_size() == world.Get_size(), "the world by default")
    halves = world.Split(world.Get_rank() % 2, world.Get_rank())
    check(aedelsten.aedelsten_mpi_size(halves.py2f()) == halves.Get_size(), "a sub-communicator by its handle")
    # A context's work split over a sub-communicator, then over the world: the overlap as on this process alone.
    alone = aedelsten.AedelstenContext("test_fixtures/mpi_oneint_check.inp")
    split = aedelsten.AedelstenContext("test_fixtures/mpi_oneint_check.inp", comm=halves.py2f())
    for ctx in (alone, split):
        ctx.set_shell_pairs()
        ctx.set_scratch(64, "mib")
    overlap = alone.compute_one_electron_integrals("overlap")
    check(np.array_equal(split.compute_one_electron_integrals("overlap"), overlap), "a context over a sub-communicator")
    split.set_communicator(world.py2f())
    check(np.array_equal(split.compute_one_electron_integrals("overlap"), overlap), "the context over the world")
    del alone, split
    # The half transform by the node's ranks into one X in their shared memory: the summed X to rounding.
    ctx = aedelsten.AedelstenContext("test_fixtures/mpi_eri_check.inp", comm=world.py2f())
    ctx.set_shell_pairs()
    ctx.set_scratch(64, "mib")
    ctx.set_eri_data()
    n, m = ctx.get_n_cont(), 3
    c = np.array([[0.1 * ((mu * 7 + u * 3 + 1) % 5) - 0.2 for mu in range(n)] for u in range(m)])
    summed = ctx.compute_eri_half_transformed(c, c)
    shared = aedelsten.SharedArray(m * n * m * n, comm=world.py2f())
    on_node = ctx.compute_eri_half_transformed_on_node(c, c, shared.array.reshape((n, m, n, m)))
    check(np.max(np.abs(on_node - summed)) < 1e-12, "X in the node's shared memory")
    del on_node
    shared.free()
    del ctx
    try:
        aedelsten.aedelsten_mpi_init("single")
        check(False, "init refused when the host owns MPI")
    except RuntimeError:
        check(True, "init refused when the host owns MPI")
    aedelsten.aedelsten_mpi_finalize()
    check(aedelsten.aedelsten_mpi_initialized(), "the host's MPI left up by aedelsten_mpi_finalize")
    halves.Free()
