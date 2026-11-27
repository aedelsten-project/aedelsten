! SPDX-FileCopyrightText: 2026 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! The AO values and their derivatives from Fortran: the counterpart of examples/python/ao_values_demo.py and
! examples/julia/ao_values_demo.jl. compute_ao_hessians gives the values and gradients compute_ao_values gives, and
! second derivatives that agree with central differences of the gradients.
!
! Build as examples/fortran/oneint_demo.f90 (after `cargo build --release` and
! `PREFIX=$INSTALL_PATH cargo install-aedelsten install-fortran --release`):
!
!   gfortran -I$INSTALL_PATH/include examples/fortran/ao_values_demo.f90 \
!       $INSTALL_PATH/lib/Aedelsten.o $INSTALL_PATH/lib/libaedelsten.dylib -o ao_values_demo
!
! Run from the repository root.
program ao_values_demo
    use iso_fortran_env, only: real64, int32
    use Aedelsten, only: AedelstenContext
    implicit none

    integer, parameter :: n_points = 40
    real(kind=real64), parameter :: step = 1.0e-5_real64
    integer, parameter :: pair_a(6) = [1, 1, 1, 2, 2, 3], pair_b(6) = [1, 2, 3, 2, 3, 3]
    type(AedelstenContext)         :: ctx
    integer(kind=int32), allocatable :: shells(:), counts(:,:)
    real(kind=real64) :: points(3, n_points), shifted(3, n_points), worst, t
    real(kind=real64), allocatable :: values(:,:), gradients(:,:,:), h_values(:,:), h_gradients(:,:,:), hessians(:,:,:)
    real(kind=real64), allocatable :: plus(:,:,:), minus(:,:,:), scratch(:,:)
    integer :: i, k, n_listed, n_shells

    call ctx%set_aedelsten_context("test_fixtures/mpi_oneint_check.inp")
    n_shells = ctx%get_n_shells()
    shells = [(int(i, int32), i = 0, n_shells - 1)]
    allocate(counts(n_shells, 4))
    call ctx%get_shells(counts)
    n_listed = sum(counts(:, 4))
    do i = 1, n_points
        t = real(i - 1, real64)
        points(:, i) = [sin(0.7_real64 * t) * 1.8_real64, cos(1.3_real64 * t) * 1.5_real64, 0.11_real64 * t - 2.0_real64]
    end do
    allocate(values(n_listed, n_points), gradients(n_listed, n_points, 3), h_values(n_listed, n_points), &
        h_gradients(n_listed, n_points, 3), hessians(n_listed, n_points, 6), plus(n_listed, n_points, 3), &
        minus(n_listed, n_points, 3), scratch(n_listed, n_points))

    call ctx%compute_ao_values(shells, points, values, gradients)
    call ctx%compute_ao_hessians(shells, points, h_values, h_gradients, hessians)
    call check(all(values == h_values) .and. all(gradients == h_gradients), &
        "compute_ao_hessians gives compute_ao_values's values and gradients")

    worst = 0.0_real64
    do k = 1, 6
        shifted = points
        shifted(pair_b(k), :) = shifted(pair_b(k), :) + step
        call ctx%compute_ao_values(shells, shifted, scratch, plus)
        shifted(pair_b(k), :) = shifted(pair_b(k), :) - 2.0_real64 * step
        call ctx%compute_ao_values(shells, shifted, scratch, minus)
        worst = max(worst, maxval(abs(hessians(:, :, k) - (plus(:, :, pair_a(k)) - minus(:, :, pair_a(k))) / (2.0_real64 * step)) &
            / (1.0_real64 + abs(hessians(:, :, k)))))
    end do
    call check(worst < 1.0e-6_real64, "the second derivatives against central differences of the gradients")
    print '(A)', "PASS"

contains

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
end program ao_values_demo
