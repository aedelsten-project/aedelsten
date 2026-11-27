! SPDX-FileCopyrightText: 2026 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! The scratch need from Fortran: the counterpart of examples/python/scratch_need_demo.py and
! examples/julia/scratch_need_demo.jl. Ammonia (test_fixtures/symmetry_demo.inp): the query from the input alone,
! its breakdown per operation (the parts adding up, the largest the query), and the scratch sized from it holding
! the ERI data and a restricted Fock build.
!
! Build as examples/fortran/oneint_demo.f90 (README.md), then run from the repository root:
!
!   gfortran -I$INSTALL_PATH/include examples/fortran/scratch_need_demo.f90 \
!       $INSTALL_PATH/lib/Aedelsten.o $INSTALL_PATH/lib/libaedelsten.dylib -o scratch_need_demo
!   ./scratch_need_demo
!
! With a module built with --int64, declare the integers from the library as integer(kind=int64).
program scratch_need_demo
    use iso_fortran_env, only: real64, int64
    use iso_c_binding, only: c_int
    use Aedelsten, only: AedelstenContext, scratch_eri_data, scratch_restricted_fock, &
        scratch_two_electron_gradient, scratch_active_space_fock, scratch_half_transform, scratch_amfi, scratch_none, &
        scratch_breakdown_field_count
    implicit none

    integer, parameter :: n_operations = 5
    ! A column per operation: the kind, then its parameters.
    integer(kind=c_int), parameter :: operations(5, n_operations) = reshape([ &
        scratch_eri_data, 0, 0, 0, 0, &
        scratch_restricted_fock, 0, 0, 0, 0, &
        scratch_two_electron_gradient, 2, 0, 0, 0, &
        scratch_active_space_fock, 4, 0, 0, 0, &
        scratch_half_transform, 3, scratch_none, scratch_none, 1], [5, n_operations])

    type(AedelstenContext)            :: ctx
    integer(kind=int64), allocatable  :: breakdown(:, :)
    integer(kind=int64)               :: query, peak
    real(kind=real64), allocatable    :: density(:, :), j(:, :), k(:, :), amfi_planes(:, :, :)
    integer(kind=c_int)               :: amfi_operation(5, 1)
    integer(kind=int64)               :: amfi
    integer                           :: i, n, shells

    call ctx%set_aedelsten_context("test_fixtures/symmetry_demo.inp")
    ! Before the shell pairs: the input alone.
    query = ctx%scratch_bytes(operations, .false.)
    allocate(breakdown(scratch_breakdown_field_count(), n_operations))
    call ctx%scratch_breakdown(operations, .false., breakdown)
    do i = 1, n_operations
        print '(a, i0, a, i0, a, i0)', "      operation ", i, ": ", breakdown(1, i), " bytes, item ", breakdown(6, i)
        call check(breakdown(1, i) == sum(breakdown(2:5, i)), "the parts add up")
        call check(breakdown(1, i) == ctx%scratch_bytes(operations(:, i:i), .false.), "its need is its query")
        shells = count(breakdown(7::3, i) >= 0)
        call check(shells == merge(4, 2, breakdown(6, i) == 4), "the item's shells")
    end do
    call check(maxval(breakdown(1, :)) == query, "the query is the largest need")
    call check(ctx%scratch_bytes(operations(:, 2:2), .true.) >= ctx%scratch_bytes(operations(:, 2:2), .false.), &
        "the primitive blocks need at least the shells")

    call ctx%set_shell_pairs()
    call ctx%set_scratch(int((query + 1023) / 1024), "kib")
    call ctx%set_eri_data()
    n = ctx%get_n_cont()
    allocate(density(n, n), j(n, n), k(n, n))
    density = 0.0_real64
    do i = 1, n
        density(i, i) = 1.0_real64
    end do
    call ctx%compute_fock_direct(j, k, density, 1.0_real64, -0.5_real64)
    peak = ctx%get_scratch_peak_bytes()
    call check(peak <= query, "the ERI data and a Fock build fit in the query")

    ! AMFI over the primitive shells: its query, measured on the atom kinds, is its peak.
    amfi_operation = reshape([scratch_amfi, 0, 0, 0, 0], [5, 1])
    amfi = ctx%scratch_bytes(amfi_operation, .false.)
    call ctx%set_primitive_shell_pairs()
    call ctx%resize_scratch(int((amfi + 1023) / 1024), "kib")
    allocate(amfi_planes(ctx%get_n_prim_unique(), ctx%get_n_prim_unique(), 6))
    call ctx%compute_amfi_integrals(amfi_planes)
    peak = ctx%get_scratch_peak_bytes()
    call check(peak > 0 .and. peak <= amfi, "the AMFI integrals fit in their query")
    print '(a)', "all checks passed"

contains

    subroutine check(condition, label)
        logical, intent(in)          :: condition
        character(len=*), intent(in) :: label
        if (condition) then
            print '(a)', "ok:   " // label
        else
            print '(a)', "FAIL: " // label
            error stop 1
        end if
    end subroutine check

end program scratch_need_demo
