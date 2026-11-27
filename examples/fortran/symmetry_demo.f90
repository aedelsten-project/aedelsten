! SPDX-FileCopyrightText: 2026 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! The point group's operations from Fortran: the counterpart of examples/python/symmetry_demo.py and
! examples/julia/symmetry_demo.jl. Ammonia turned out of every axis (test_fixtures/symmetry_demo.inp), C3v: every
! operation's matrix orthogonal, its images a permutation of equivalent centres, every vector between two centres
! carried onto the vector between their images, one identity.
! The irreps: the AO counts, the characters orthonormal, the irrep matrices orthogonal with the characters as
! their traces, a projection onto a one-dimensional irrep carried to its character times itself.
! Indices from the library are 0-based.
!
! Build as examples/fortran/oneint_demo.f90 (README.md), then run from the repository root:
!
!   gfortran -I$INSTALL_PATH/include examples/fortran/symmetry_demo.f90 \
!       $INSTALL_PATH/lib/Aedelsten.o $INSTALL_PATH/lib/libaedelsten.dylib -o symmetry_demo
!   ./symmetry_demo
!
! With a module built with --int64, declare the integers from the library as integer(kind=int64).
program symmetry_demo
    use iso_fortran_env, only: real64
    use Aedelsten, only: AedelstenContext
    implicit none

    ! The fixture's positions (bohr, one column per centre) and elements.
    real(kind=real64), parameter :: positions(3, 4) = reshape([ &
        0.4000000000000000_real64, -1.1837482993408999_real64, 2.3994294843469834_real64, &
        2.0909455857523227_real64, -0.2618653840665375_real64, 2.1168787516457970_real64, &
        -0.8984653640277757_real64, 0.2580710345499226_real64, 2.5548151556260290_real64, &
        0.0075197782754526_real64, -1.9820015685184955_real64, 0.6680280306678168_real64], [3, 4])
    character(len=1), parameter :: elements(4) = ["N", "H", "H", "H"]

    type(AedelstenContext) :: ctx
    real(kind=real64)      :: m(3, 3), identity(3, 3), carried
    integer                :: h, n, op, images(4)
    integer                :: a, b, i, identities
    logical                :: permutes, seen(4), aligned
    integer                :: n_irreps, j, matrices, one_dimensional, d
    integer, allocatable   :: counts(:)
    real(kind=real64), allocatable :: chi(:,:), orbitals(:,:), images_of(:,:), projection(:,:)
    real(kind=real64)      :: g(2, 2)

    identity = 0.0_real64
    do i = 1, 3
        identity(i, i) = 1.0_real64
    end do

    call ctx%set_aedelsten_context("test_fixtures/symmetry_demo.inp")
    call ctx%set_shell_pairs()
    call ctx%set_scratch(64, "mib")
    call ctx%set_symmetry()
    call check(ctx%get_point_group_name() == "C3v", "the point group is C3v")
    h = ctx%get_symmetry_operation_count()
    call check(h == 6, "six operations")
    n = ctx%get_n_symmetry_centres()
    call check(n == 4, "four centres")

    identities = 0
    do op = 0, h - 1
        call ctx%get_symmetry_operation_matrix(op, m)
        call ctx%get_symmetry_centre_images(op, images)
        call check(maxval(abs(matmul(transpose(m), m) - identity)) < 1.0e-12_real64, "an operation is orthogonal")
        seen = .false.
        permutes = .true.
        do a = 1, 4
            permutes = permutes .and. images(a) >= 0 .and. images(a) < 4
            if (permutes) then
                permutes = permutes .and. .not. seen(images(a) + 1) .and. elements(a) == elements(images(a) + 1)
                seen(images(a) + 1) = .true.
            end if
        end do
        call check(permutes, "an operation permutes equivalent centres")
        carried = 0.0_real64
        do a = 1, 4
            do b = 1, 4
                carried = max(carried, maxval(abs(matmul(m, positions(:, a) - positions(:, b)) &
                    - (positions(:, images(a) + 1) - positions(:, images(b) + 1)))))
            end do
        end do
        call check(carried < 1.0e-8_real64, "an operation carries every vector between centres onto its images'")
        if (all(images == [0, 1, 2, 3]) .and. maxval(abs(m - identity)) < 1.0e-12_real64) &
            identities = identities + 1
    end do
    call check(identities == 1, "one identity")

    ! The irreps: AO functions in each, together all; the characters orthonormal over the group; each aligned irrep
    ! matrix orthogonal with the character as its trace.
    n_irreps = ctx%get_n_irreps()
    allocate(counts(n_irreps), chi(n_irreps, h))
    call ctx%get_ao_irrep_counts(counts)
    call check(sum(counts) == ctx%get_n_cont(), "every AO function in one irrep")
    do i = 1, n_irreps
        do op = 0, h - 1
            chi(i, op + 1) = ctx%get_irrep_character(i - 1, op)
        end do
    end do
    do i = 1, n_irreps
        do j = 1, n_irreps
            call check(abs(dot_product(chi(i, :), chi(j, :)) / h - merge(1.0_real64, 0.0_real64, i == j)) < 1.0e-12_real64, &
                       "the characters are orthonormal")
        end do
    end do
    matrices = 0
    do i = 1, n_irreps
        d = ctx%get_irrep_dimension(i - 1)
        do op = 0, h - 1
            call ctx%get_irrep_matrix(i - 1, op, g(1:d, 1:d), aligned)
            if (.not. aligned) cycle
            call check(maxval(abs(matmul(transpose(g(1:d, 1:d)), g(1:d, 1:d)) - identity(1:d, 1:d))) < 1.0e-12_real64 &
                       .and. abs(sum([(g(j, j), j = 1, d)]) - chi(i, op + 1)) < 1.0e-12_real64, &
                       "an irrep matrix is orthogonal, its trace the character")
            if (d == 2) matrices = matrices + 1
        end do
    end do
    call check(matrices == h, "E's matrices at every operation")

    ! The operations on orbitals: a model orbital projected onto a one-dimensional irrep with the operations and the
    ! characters, p = (1/h) sum_R chi(R) R c, goes to chi(R) p under every operation.
    allocate(orbitals(ctx%get_n_cont(), 1), images_of(ctx%get_n_cont(), 1), projection(ctx%get_n_cont(), 1))
    do i = 1, ctx%get_n_cont()
        orbitals(i, 1) = 0.1_real64 * mod((i - 1) * 7, 13) - 0.6_real64
    end do
    one_dimensional = 0
    do i = 1, n_irreps
        if (ctx%get_irrep_dimension(i - 1) /= 1) cycle
        projection = 0.0_real64
        do op = 0, h - 1
            call ctx%apply_symmetry_operation(op, orbitals, images_of)
            projection = projection + chi(i, op + 1) * images_of / h
        end do
        if (maxval(abs(projection)) < 1.0e-8_real64) cycle
        do op = 0, h - 1
            call ctx%apply_symmetry_operation(op, projection, images_of)
            call check(maxval(abs(images_of - chi(i, op + 1) * projection)) < 1.0e-12_real64, &
                       "an operation on a projection onto a one-dimensional irrep: chi times it")
        end do
        one_dimensional = one_dimensional + 1
    end do
    call check(one_dimensional > 0, "a projection onto a one-dimensional irrep checked")
    print '(A)', "all checks passed"

contains

    subroutine check(condition, label)
        logical,          intent(in) :: condition
        character(len=*), intent(in) :: label
        if (condition) then
            print '(2A)', "ok:   ", label
        else
            print '(2A)', "FAIL: ", label
            error stop 1
        end if
    end subroutine check
end program symmetry_demo
