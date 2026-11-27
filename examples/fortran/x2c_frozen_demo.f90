! SPDX-FileCopyrightText: 2026 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! The X2C Hamiltonian with the decoupling held, and its gradient, from Fortran: the counterpart of
! examples/python/x2c_frozen_demo.py and examples/julia/x2c_frozen_demo.jl. The transforms of a derived X2C
! Hamiltonian, supplied back, give the same Hamiltonian and the same frozen gradient; the gradient of a free molecule
! sums to zero over the centres.
!
! Build as examples/fortran/oneint_demo.f90 and run from the repository root.
program x2c_frozen_demo
    use iso_fortran_env, only: real64
    use Aedelsten, only: AedelstenContext
    implicit none

    type(AedelstenContext) :: ctx
    real(kind=real64), allocatable :: s(:,:), t(:,:), v(:,:), pvp(:,:), h(:,:), h_held(:,:), u_large(:,:), u_small(:,:)
    real(kind=real64), allocatable :: density(:,:), gradient(:,:), gradient_held(:,:), part(:,:), flat(:)
    real(kind=real64), allocatable :: v_2c(:,:), s_2c(:,:), scalar(:,:,:), spin(:,:,:), raw(:,:,:), raw_pxp(:,:,:), planes(:,:,:)
    integer :: n, m, n_centres, i

    call ctx%set_aedelsten_context("test_fixtures/mpi_oneint_check.inp")
    call ctx%set_shell_pairs()
    call ctx%set_scratch(64, "mib")
    call ctx%set_primitive_shell_pairs()
    n = ctx%get_n_prim_unique()
    m = ctx%get_n_cont()
    n_centres = ctx%get_n_gradient_centres()
    allocate(s(n, n), t(n, n), v(n, n), pvp(n, n), h(n, n), h_held(n, n), u_large(n, n), u_small(n, n), flat(n * n))
    allocate(density(m, m), gradient(3, n_centres), gradient_held(3, n_centres), part(3, n_centres))
    call ctx%compute_one_electron_integrals(s, "overlap", primitive=.true.)
    call ctx%compute_one_electron_integrals(t, "kinetic", primitive=.true.)
    call ctx%compute_one_electron_integrals(v, "nuclear_attraction", primitive=.true.)
    call ctx%compute_one_electron_picture_change_integrals(flat, n, "nuclear_attraction", .false.)
    pvp = reshape(flat, [n, n])
    call ctx%construct_x2c_one_electron_hamiltonian(h, s, t, v, pvp)
    density = 0.0_real64
    do i = 1, m
        density(i, i) = 0.1_real64
    end do
    call frozen_gradient(gradient)

    call ctx%get_picture_change_transforms(u_large, u_small)
    call ctx%set_picture_change_transforms(u_large, u_small)
    call ctx%construct_x2c_one_electron_hamiltonian(h_held, s, t, v, pvp)
    call check(maxval(abs(h_held - h)) < 1.0e-12_real64 * maxval(abs(h)), "the supplied transforms give the derived Hamiltonian")
    call frozen_gradient(gradient_held)
    call check(all(gradient_held == gradient), "and the same frozen gradient")
    call check(maxval(abs(sum(gradient, dim=2))) < 1.0e-10_real64, "the gradient sums to zero over the centres")

    ! The scalar transform of the raw planes (V, pVp) is the picture-changed nuclear attraction over the primitives;
    ! two planes at once are each plane's own transform.
    allocate(v_2c(n, n), s_2c(n, n), scalar(n, n, 1), spin(n, n, 3), raw(n, n, 2), raw_pxp(n, n, 2), planes(n, n, 2))
    call ctx%transform_one_electron_operator(v, pvp, v_2c)
    call ctx%compute_picture_changed_one_electron_operator(scalar, spin, "nuclear_attraction", .false.)
    call check(maxval(abs(v_2c - scalar(:, :, 1))) < 1.0e-12_real64 * maxval(abs(v_2c)), &
        "transform_one_electron_operator(V, pVp) == the picture-changed V")
    raw(:, :, 1) = v
    raw(:, :, 2) = s
    raw_pxp(:, :, 1) = pvp
    raw_pxp(:, :, 2) = t
    call ctx%transform_one_electron_operator(raw, raw_pxp, planes)
    call ctx%transform_one_electron_operator(s, t, s_2c)
    call check(maxval(abs(planes(:, :, 1) - v_2c)) < 1.0e-13_real64 * maxval(abs(v_2c)) .and. &
        maxval(abs(planes(:, :, 2) - s_2c)) < 1.0e-13_real64 * maxval(abs(s_2c)), "two planes at once: each its own transform")
    print '(A)', "PASS"

contains

    ! The frozen gradient of tr(D h): the kinetic part and the one potential in the transformation, the nuclei.
    subroutine frozen_gradient(total)
        real(kind=real64), intent(inout) :: total(:,:)
        call ctx%set_x2c_frozen_density(density)
        call ctx%compute_x2c_frozen_kinetic_gradient(total)
        call ctx%compute_x2c_frozen_potential_gradient(part, "nuclear_attraction")
        total = total + part
        call ctx%free_x2c_frozen_density()
    end subroutine frozen_gradient

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
end program x2c_frozen_demo
