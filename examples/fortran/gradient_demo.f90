! SPDX-FileCopyrightText: 2026 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! The two-electron gradient from Fortran: the counterpart of examples/python/gradient_demo.py and
! examples/julia/gradient_demo.jl. The restricted-HF terms of a model density on water, all centres at once and with
! every other centre frozen (each the same column, the others zero), and the translational sum rule: every basis
! function sits on a gradient centre, so the integral derivatives over all centres sum to zero.
!
! Build (after, from the repository root: `cargo build --release` then
! `PREFIX=$INSTALL_PATH cargo install-aedelsten install-fortran --release` -- see README.md):
!
!   gfortran -I$INSTALL_PATH/include examples/fortran/gradient_demo.f90 \
!       $INSTALL_PATH/lib/Aedelsten.o $INSTALL_PATH/lib/libaedelsten.dylib \
!       -o gradient_demo                                ! .so on Linux, .dll on Windows
!
! Run from the repository root (reads test_fixtures/mpi_eri_check.inp by a relative path):
!
!   ./gradient_demo
program gradient_demo
    use iso_fortran_env, only: real64
    use Aedelsten, only: AedelstenContext
    implicit none

    type(AedelstenContext)         :: ctx
    real(kind=real64), allocatable :: densities(:,:,:), gradient(:,:), one(:,:)
    character(len=8)               :: kinds(2) = ["coulomb ", "exchange"]
    real(kind=real64)              :: coefficients(2) = [1.0_real64, -0.5_real64], scale
    integer                        :: n, n_centres, i, j, centre
    character(len=64)              :: label

    call ctx%set_aedelsten_context("test_fixtures/mpi_eri_check.inp")
    call ctx%set_shell_pairs()
    call ctx%set_scratch(64, "mib")
    call ctx%set_eri_data()
    n = ctx%get_n_cont()
    n_centres = ctx%get_n_gradient_centres()
    allocate(densities(n, n, 1), gradient(3, n_centres), one(3, n_centres))
    do j = 1, n
        do i = 1, n
            densities(i, j, 1) = 0.05_real64 * mod((min(i, j) - 1) * 7 + (max(i, j) - 1) * 3, 11) - 0.1_real64
        end do
    end do

    call ctx%compute_two_electron_gradient(gradient, densities, kinds, [1, 1], [1, 1], coefficients)
    scale = maxval(abs(gradient))
    print '(A, I0, A, I0, A, ES14.6)', "n_cont = ", n, ", gradient centres = ", n_centres, ", largest entry = ", scale
    call check(scale > 1.0e-3_real64, "the gradient, one column per centre")
    call check(maxval(abs(sum(gradient, dim=2))) < 1.0e-10_real64 * scale, "translational sum rule: the columns sum to zero")
    do centre = 0, n_centres - 1
        call ctx%compute_two_electron_gradient(one, densities, kinds, [1, 1], [1, 1], coefficients, &
            active=[(i == centre, i = 0, n_centres - 1)])
        write(label, '(A, I0, A)') "centre ", centre, " alone active: its column, the others zero"
        call check(maxval(abs(one(:, centre + 1) - gradient(:, centre + 1))) < 1.0e-12_real64 * scale .and. &
                   all(one(:, 1:centre) == 0.0_real64) .and. all(one(:, centre + 2:) == 0.0_real64), trim(label))
    end do
    print '(A)', "PASS"

contains

    subroutine check(condition, label)
        logical,          intent(in) :: condition
        character(len=*), intent(in) :: label
        if (condition) then
            print '(A)', "ok:   " // label
        else
            print '(A)', "FAIL: " // label
            stop 1
        end if
    end subroutine check

end program gradient_demo
