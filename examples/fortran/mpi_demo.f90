! SPDX-FileCopyrightText: 2026 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! MPI from Fortran, both routes: the library initializes MPI (aedelsten_mpi_init, for a program without MPI of its
! own), or the program does (`use mpi`, built with the same MPI library's mpif90) and passes its communicators as
! their Fortran handles. The counterpart of examples/python/mpi_demo.py and examples/julia/mpi_demo.jl.
!
! Build the library with MPI and the Fortran interface (from the repository root):
!   cargo build --release --features mpi
!   PREFIX=$INSTALL_PATH cargo install-aedelsten install-fortran --release
! The library route (plain gfortran):
!   gfortran -cpp -I$INSTALL_PATH/include examples/fortran/mpi_demo.f90 $INSTALL_PATH/lib/Aedelsten.o \
!       $INSTALL_PATH/lib/libaedelsten.dylib -o mpi_demo_library
! The host route (the MPI library's own compiler wrapper):
!   mpif90 -cpp -DHOST_MPI -I$INSTALL_PATH/include examples/fortran/mpi_demo.f90 $INSTALL_PATH/lib/Aedelsten.o \
!       $INSTALL_PATH/lib/libaedelsten.dylib -o mpi_demo_host
! Run either under mpirun, e.g. `mpirun -n 2 ./mpi_demo_library`.

program mpi_demo
#ifdef HOST_MPI
    use mpi
#endif
    use Aedelsten
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    integer :: provided, rank, size
#ifdef HOST_MPI
    integer :: ierr, mpi_rank, mpi_size, halves, halves_size
    type(AedelstenContext) :: alone, split, ctx
    type(AedelstenSharedArray) :: shared
    real(kind=real64), allocatable :: overlap(:, :), split_overlap(:, :), c(:, :), summed(:, :, :, :)
    real(kind=real64), pointer :: on_node(:, :, :, :)
    integer :: n, mu, u
    integer, parameter :: m = 3

    call check_version()
    call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
    call check(aedelsten_mpi_initialized(), "MPI up, initialized by the program")
    call check(aedelsten_mpi_thread_level() >= 1, "at least funneled")
    call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)
    call check(aedelsten_mpi_rank(MPI_COMM_WORLD) == mpi_rank, "the same rank")
    call check(aedelsten_mpi_size() == mpi_size, "the world by default")
    call MPI_Comm_split(MPI_COMM_WORLD, mod(mpi_rank, 2), mpi_rank, halves, ierr)
    call MPI_Comm_size(halves, halves_size, ierr)
    call check(aedelsten_mpi_size(halves) == halves_size, "a sub-communicator by its handle")
    ! A context's work split over a sub-communicator, then over the world: the overlap as on this process alone.
    call alone%set_aedelsten_context("test_fixtures/mpi_oneint_check.inp")
    call split%set_aedelsten_context("test_fixtures/mpi_oneint_check.inp", comm=halves)
    call alone%set_shell_pairs()
    call split%set_shell_pairs()
    call alone%set_scratch(64, "mib")
    call split%set_scratch(64, "mib")
    allocate(overlap(alone%get_n_cont(), alone%get_n_cont()), split_overlap(alone%get_n_cont(), alone%get_n_cont()))
    call alone%compute_one_electron_integrals(overlap, "overlap")
    call split%compute_one_electron_integrals(split_overlap, "overlap")
    call check(all(split_overlap == overlap), "a context over a sub-communicator")
    call split%set_communicator(MPI_COMM_WORLD)
    call split%compute_one_electron_integrals(split_overlap, "overlap")
    call check(all(split_overlap == overlap), "the context over the world")
    ! The half transform by the node's ranks into one X in their shared memory: the summed X to rounding.
    call ctx%set_aedelsten_context("test_fixtures/mpi_eri_check.inp", comm=MPI_COMM_WORLD)
    call ctx%set_shell_pairs()
    call ctx%set_scratch(64, "mib")
    call ctx%set_eri_data()
    n = ctx%get_n_cont()
    allocate(c(n, m), summed(m, n, m, n))
    do u = 1, m
        do mu = 1, n
            c(mu, u) = 0.1_real64 * mod((mu - 1) * 7 + (u - 1) * 3 + 1, 5) - 0.2_real64
        end do
    end do
    call ctx%compute_eri_half_transformed(summed, c, c)
    call aedelsten_shared_array(shared, int(m * n * m * n, kind=8), comm=MPI_COMM_WORLD)
    on_node(1:m, 1:n, 1:m, 1:n) => shared%data
    call ctx%compute_eri_half_transformed_on_node(on_node, c, c)
    call check(maxval(abs(on_node - summed)) < 1.0e-12_real64, "X in the node's shared memory")
    call aedelsten_shared_array_free(shared)
    call aedelsten_mpi_finalize()
    call check(aedelsten_mpi_initialized(), "the program's MPI left up by aedelsten_mpi_finalize")
    call MPI_Comm_free(halves, ierr)
    call MPI_Finalize(ierr)
#else
    call check_version()
    call check(.not. aedelsten_mpi_initialized(), "MPI not up before aedelsten_mpi_init")
    provided = aedelsten_mpi_init("funneled")
    call check(provided >= 1, "at least funneled")
    rank = aedelsten_mpi_rank()
    size = aedelsten_mpi_size()
    call check(rank >= 0 .and. rank < size, "a rank of the world")
    ! ... the library's computations ...
    call aedelsten_mpi_finalize()
    call check(.not. aedelsten_mpi_initialized(), "MPI finalized")
#endif

contains

    ! The library's version and the commit it was built from (empty from a tree that was no git checkout).
    subroutine check_version()
        character(len=:), allocatable :: version, commit
        version = aedelsten_get_version()
        commit = aedelsten_get_source_commit()
        call check(len(version) >= 5 .and. verify(version, "0123456789.") == 0, "version " // version)
        call check(len(commit) == 0 .or. (len(commit) == 12 .and. verify(commit, "0123456789abcdef") == 0), &
            "source commit " // commit)
    end subroutine check_version

    subroutine check(condition, label)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        if (condition) then
            print '(A)', "ok:   " // label
        else
            print '(A)', "FAIL: " // label
            error stop 1
        end if
    end subroutine check

end program mpi_demo
