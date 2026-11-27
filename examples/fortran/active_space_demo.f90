! SPDX-FileCopyrightText: 2026 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! The active-space walks from Fortran: the counterpart of examples/python/active_space_demo.py and
! examples/julia/active_space_demo.jl. The Coulomb matrices of the active pair densities M from the direct walk,
! one basis and two, against M from the stored S1 tensor, T from M (transform_pair_coulomb_one_index) against T
! from the S1 tensor, J/K of several densities of no symmetry in one walk against the S1 tensor and one walk each, and
! the half-transformed ERIs by batches of sigma (each shell alone), one basis and two, against the S1 tensor.
!
! Build (after, from the repository root: `cargo build --release` then
! `PREFIX=$INSTALL_PATH cargo install-aedelsten install-fortran --release` -- see README.md):
!
!   gfortran -I$INSTALL_PATH/include examples/fortran/active_space_demo.f90 \
!       $INSTALL_PATH/lib/Aedelsten.o $INSTALL_PATH/lib/libaedelsten.dylib \
!       -o active_space_demo                            ! .so on Linux, .dll on Windows
!
! Run from the repository root (reads test_fixtures/mpi_eri_check.inp by a relative path):
!
!   ./active_space_demo
program active_space_demo
    use iso_fortran_env, only: real64, int64
    use Aedelsten, only: AedelstenContext
    implicit none

    integer, parameter :: m_act = 3
    type(AedelstenContext)         :: ctx
    real(kind=real64), allocatable :: g(:,:,:,:), density(:,:), c(:,:), c_bra(:,:)
    real(kind=real64), allocatable :: j_i(:,:), k_i(:,:), j_a(:,:), k_a(:,:)
    real(kind=real64), allocatable :: m(:,:), m_mixed(:,:), p(:,:,:), m_ref(:,:), t(:,:,:), t_ref(:,:,:)
    real(kind=real64), allocatable :: densities(:,:,:), j_multi(:,:,:), k_multi(:,:,:), j_ref(:,:), k_ref(:,:)
    real(kind=real64), allocatable :: x_ref(:,:,:,:), x_all(:,:,:,:), batch(:,:,:,:)
    integer, allocatable :: shells(:,:), starts(:)
    integer :: n, mu, nu, u, v, w, x, i, j, s, ia, ib, ic, id, k, r, l, basis
    integer(kind=int64) :: one_basis_bytes, two_bases_bytes

    call ctx%set_aedelsten_context("test_fixtures/mpi_eri_check.inp")
    call ctx%set_shell_pairs()
    call ctx%set_scratch(64, "mib")
    call ctx%set_eri_data()
    n = ctx%get_n_cont()
    allocate(g(n,n,n,n))
    call ctx%compute_eri_stored(g)

    allocate(density(n,n), c(n,m_act), c_bra(n,m_act))
    do i = 1, n
        do j = 1, n
            density(i,j) = 0.05_real64 * mod((min(i,j) - 1) * 7 + (max(i,j) - 1) * 3, 11) - 0.1_real64
        end do
        do u = 1, m_act
            c(i,u) = 0.2_real64 * mod((i - 1) * 5 + (u - 1) * 3 + 1, 7) - 0.55_real64
            c_bra(i,u) = 0.2_real64 * mod((i - 1) * 3 + (u - 1) * 5 + 2, 7) - 0.6_real64
        end do
    end do
    allocate(j_i(n,n), k_i(n,n), j_a(n,n), k_a(n,n))

    ! One basis: the pairs v >= w at v (v - 1)/2 + w.
    allocate(m(m_act*(m_act+1)/2, n*(n+1)/2), p(n,n,m_act*(m_act+1)/2))
    call ctx%compute_fock_direct_active_space(j_i, k_i, j_a, k_a, m, density, density, c, &
        1.0_real64, -0.5_real64, 1.0_real64, -0.5_real64)
    do v = 1, m_act
        do w = 1, v
            p(:,:,v*(v-1)/2+w) = (spread(c(:,v), 2, n) * spread(c(:,w), 1, n) + spread(c(:,w), 2, n) * spread(c(:,v), 1, n)) / 2
        end do
    end do
    call dense(p, m_ref)
    call check_close(packed(m), m_ref, "M, one basis, against the S1 tensor")
    allocate(t(m_act, size(m,1), n))
    call ctx%transform_pair_coulomb_one_index(t, m, c)
    call transformed(m_ref, t_ref)
    call check_close(reshape(t, [size(t), 1]), reshape(t_ref, [size(t_ref), 1]), "T from M, one basis")
    deallocate(p, m_ref, t, t_ref)

    ! Two bases: every pair (v, w) at v + (w - 1) n_active.
    allocate(m_mixed(m_act*m_act, n*(n+1)/2), p(n,n,m_act*m_act))
    call ctx%compute_fock_direct_mixed_active_space(j_i, k_i, m_mixed, density, c_bra, c, 1.0_real64, -0.5_real64)
    do v = 1, m_act
        do w = 1, m_act
            p(:,:,v+(w-1)*m_act) = (spread(c_bra(:,v), 2, n) * spread(c(:,w), 1, n) &
                + spread(c(:,w), 2, n) * spread(c_bra(:,v), 1, n)) / 2
        end do
    end do
    call dense(p, m_ref)
    call check_close(packed(m_mixed), m_ref, "M, two bases, against the S1 tensor")
    allocate(t(m_act, size(m_mixed,1), n))
    call ctx%transform_pair_coulomb_one_index(t, m_mixed, c)
    call transformed(m_ref, t_ref)
    call check_close(reshape(t, [size(t), 1]), reshape(t_ref, [size(t_ref), 1]), "T from M, two bases")

    ! J and K of two densities of no symmetry in one walk, against the S1 tensor and against one walk each.
    allocate(densities(n,n,2), j_multi(n,n,2), k_multi(n,n,2), j_ref(n,n), k_ref(n,n))
    do s = 1, 2
        do j = 1, n
            do i = 1, n
                densities(i,j,s) = density(i,j) + merge(1.0_real64, -2.0_real64, s == 1) * 0.03_real64 &
                    * (mod((i - 1) * 5 + (j - 1) * 2, 7) - 3)
            end do
        end do
    end do
    call ctx%compute_fock_direct_general_multi(j_multi, k_multi, densities, 1.0_real64, -0.5_real64)
    do s = 1, 2
        do ib = 1, n
            do ia = 1, n
                j_ref(ia,ib) = sum(g(ia,ib,:,:) * (densities(:,:,s) + transpose(densities(:,:,s)))) / 2
            end do
        end do
        do ic = 1, n
            do ia = 1, n
                k_ref(ia,ic) = 0.0_real64
                do id = 1, n
                    do ib = 1, n
                        k_ref(ia,ic) = k_ref(ia,ic) - 0.5_real64 * g(ia,ib,ic,id) * densities(ib,id,s)
                    end do
                end do
            end do
        end do
        call check_close(j_multi(:,:,s), j_ref, "J of a density of two, against the S1 tensor")
        call check_close(k_multi(:,:,s), k_ref, "K of a density of two, against the S1 tensor")
        call ctx%compute_fock_direct_general(j_ref, k_ref, densities(:,:,s), 1.0_real64, -0.5_real64)
        if (any(j_ref /= j_multi(:,:,s)) .or. any(k_ref /= k_multi(:,:,s))) then
            print '(A)', "FAIL: J/K of a density of two, equal to its own walk"
            error stop 1
        end if
        print '(A)', "ok:   J/K of a density of two, equal to its own walk"
    end do

    ! The half-transformed ERIs by batches of sigma, x(p, nu, r, s): every shell alone, one basis (the same array on
    ! both sides) and two, against the S1 tensor; the scratch they take beyond the walk's own.
    allocate(shells(ctx%get_n_shells(), 4), starts(ctx%get_n_shells() + 1))
    call ctx%get_shells(shells)
    starts(1:size(shells, 1)) = shells(:, 3)
    starts(size(starts)) = n
    allocate(x_ref(m_act, n, m_act, n), x_all(m_act, n, m_act, n))
    do basis = 1, 2
        do s = 1, n
            do r = 1, m_act
                do nu = 1, n
                    do u = 1, m_act
                        x_ref(u, nu, r, s) = 0.0_real64
                        do l = 1, n
                            do mu = 1, n
                                if (basis == 1) then
                                    x_ref(u, nu, r, s) = x_ref(u, nu, r, s) + c(mu, u) * c(l, r) * g(mu, nu, l, s)
                                else
                                    x_ref(u, nu, r, s) = x_ref(u, nu, r, s) + c_bra(mu, u) * c(l, r) * g(mu, nu, l, s)
                                end if
                            end do
                        end do
                    end do
                end do
            end do
        end do
        do k = 1, size(starts) - 1
            allocate(batch(m_act, n, m_act, starts(k + 1) - starts(k)))
            if (basis == 1) then
                call ctx%compute_eri_half_transformed_sigma(batch, c, c, starts(k), starts(k + 1))
            else
                call ctx%compute_eri_half_transformed_sigma(batch, c_bra, c, starts(k), starts(k + 1))
            end if
            x_all(:, :, :, starts(k) + 1:starts(k + 1)) = batch
            deallocate(batch)
        end do
        if (basis == 1) then
            call check_close(reshape(x_all, [m_act * n, m_act * n]), reshape(x_ref, [m_act * n, m_act * n]), &
                             "X by batches of sigma, one basis, against the S1 tensor")
        else
            call check_close(reshape(x_all, [m_act * n, m_act * n]), reshape(x_ref, [m_act * n, m_act * n]), &
                             "X by batches of sigma, two bases, against the S1 tensor")
        end if
    end do
    one_basis_bytes = ctx%half_transform_scratch_bytes(m_act, m_act, .true., n, 1)
    two_bases_bytes = ctx%half_transform_scratch_bytes(m_act, m_act, .false., n, 1)
    if (.not. (0 < one_basis_bytes .and. one_basis_bytes < two_bases_bytes)) then
        print '(A)', "FAIL: the half transform's scratch, one basis less than two"
        error stop 1
    end if
    print '(A)', "ok:   the half transform's scratch, one basis less than two"
    print '(A)', "PASS"

contains

    ! The packed column of the function pair (mu, nu).
    integer function column(mu, nu)
        integer, intent(in) :: mu, nu
        integer :: q
        q = min(mu, nu) - 1
        column = q * n - q * (q - 1) / 2 + abs(mu - nu) + 1
    end function column

    ! M(pair, mu + (nu - 1) n) from the S1 tensor for the pair densities p(lambda, sigma, pair).
    subroutine dense(p, m_out)
        real(kind=real64), intent(in)               :: p(:,:,:)
        real(kind=real64), allocatable, intent(out) :: m_out(:,:)
        allocate(m_out(size(p,3), n*n))
        do nu = 1, n
            do mu = 1, n
                do x = 1, size(p,3)
                    m_out(x, mu+(nu-1)*n) = sum(g(mu,nu,:,:) * p(:,:,x))
                end do
            end do
        end do
    end subroutine dense

    ! The packed M unpacked to (pair, mu + (nu - 1) n).
    function packed(m_in) result(full)
        real(kind=real64), intent(in) :: m_in(:,:)
        real(kind=real64)             :: full(size(m_in,1), n*n)
        do nu = 1, n
            do mu = 1, n
                full(:, mu+(nu-1)*n) = m_in(:, column(mu, nu))
            end do
        end do
    end function packed

    ! T(u, pair, mu) = sum_nu c(nu, u) M(pair, mu nu).
    subroutine transformed(m_full, t_out)
        real(kind=real64), intent(in)               :: m_full(:,:)
        real(kind=real64), allocatable, intent(out) :: t_out(:,:,:)
        allocate(t_out(m_act, size(m_full,1), n))
        t_out = 0.0_real64
        do mu = 1, n
            do nu = 1, n
                do x = 1, size(m_full,1)
                    t_out(:, x, mu) = t_out(:, x, mu) + c(nu,:) * m_full(x, mu+(nu-1)*n)
                end do
            end do
        end do
    end subroutine transformed

    subroutine check_close(a, b, label)
        real(kind=real64), intent(in) :: a(:,:), b(:,:)
        character(len=*), intent(in)  :: label
        real(kind=real64)             :: scale, difference
        scale = maxval(abs(b))
        difference = maxval(abs(a - b))
        if (scale > 1.0e-2_real64 .and. difference < 1.0e-12_real64 * scale) then
            print '(A,A,A,ES10.3,A,ES10.3,A)', "ok:   ", label, " (max ", scale, ", difference ", difference, ")"
        else
            print '(A,A,A,ES10.3,A,ES10.3,A)', "FAIL: ", label, " (max ", scale, ", difference ", difference, ")"
            error stop 1
        end if
    end subroutine check_close
end program active_space_demo
