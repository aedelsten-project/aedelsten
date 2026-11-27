! SPDX-FileCopyrightText: 2024 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! User-test demonstration for Aedelsten's Fortran interface (src/interfaces/aedelsten.f90):
! creates a context from a small test input, sets up shell pairs + scratch, computes the
! overlap/kinetic/nuclear-attraction one-electron integral matrices plus a dipole (order-1
! multipole) tensor, and sanity-checks the results (overlap must be symmetric with a unit
! diagonal, so its trace must equal n_cont) -- something a user can build and run to confirm
! their own build of the module/library pair actually works, without needing to trust or
! re-derive the numbers by hand.
!
! Output arrays are declared with their natural Fortran shape -- (n, n) for a plain matrix,
! (n, n, n_planes) for the multipole tensor -- not flattened by hand. compute_one_electron_
! integrals is GENERIC on output's rank and derives n/n_planes from that shape itself, so
! there's no separate size argument to keep in sync (the raw flat buffer the C ABI actually
! wants is an implementation detail handled inside the module, not something callers see).
!
! Build (after, from the repository root: `cargo build --release` then
! `PREFIX=$INSTALL_PATH cargo install-aedelsten install-fortran --release` -- see README.md):
!
!   gfortran -I$INSTALL_PATH/include examples/fortran/oneint_demo.f90 \
!       $INSTALL_PATH/lib/Aedelsten.o $INSTALL_PATH/lib/libaedelsten.dylib \
!       -o oneint_demo                                  ! .so on Linux, .dll on Windows
!
! Run from the repository root (reads test_fixtures/mpi_oneint_check.inp by a relative path):
!
!   ./oneint_demo
!
! If the module/library pair was built with --int64 (external_int_size = int64), declare
! `order` below as `integer(kind=int64)` and pass the scratch size as `64_int64` instead
! (use iso_fortran_env, only: int64) -- everything else is unchanged (n/n_planes always come from output's own shape).
program oneint_demo
    use iso_fortran_env, only: real64
    use Aedelsten, only: AedelstenContext, aedelsten_constant, memory_breakdown_field_count, aedelsten_timing_count, &
        aedelsten_timing_span, aedelsten_span_tree_count, aedelsten_span_tree_node, aedelsten_citation_count, &
        aedelsten_citation, aedelsten_citations_bibtex, aedelsten_clear_citations, aedelsten_mpi_collective_seconds
    use iso_c_binding, only: c_int32_t, c_int64_t
    implicit none

    type(AedelstenContext)         :: ctx, centres
    integer(kind=c_int32_t)        :: z(3), kinds(3)
    real(kind=real64)              :: xyz(3, 3), charges(3)
    real(kind=real64), allocatable :: m(:,:), m0(:,:)
    logical                        :: forward, back
    real(kind=real64), allocatable :: overlap(:,:), kinetic(:,:), nuclear_attraction(:,:)
    real(kind=real64), allocatable :: sap_helfem(:,:), sap_grasp(:,:)
    real(kind=real64), allocatable :: dipole(:,:,:), self_overlap(:,:), cross(:,:), cross_back(:,:)
    integer(kind=c_int64_t), allocatable :: before(:), after(:)
    character(len=:), allocatable :: label, bibtex
    real(kind=real64) :: seconds
    integer(kind=c_int64_t) :: calls
    logical :: found, fine
    integer :: k, depth, n_references
    integer :: n, i

    print '(A)', "=== Aedelsten Fortran interface demo ==="

    call ctx%set_aedelsten_context("test_fixtures/mpi_oneint_check.inp")
    call ctx%set_shell_pairs()
    call ctx%set_scratch(64, "mib")

    n = ctx%get_n_cont()
    print '(A,I0)', "n_cont = ", n

    allocate(overlap(n,n), kinetic(n,n), nuclear_attraction(n,n), sap_helfem(n,n), sap_grasp(n,n))
    allocate(dipole(n,n,3)) ! order 1 (dipole): 3 Cartesian components (x, y, z)

    call ctx%compute_one_electron_integrals(overlap, "overlap")
    call ctx%compute_one_electron_integrals(kinetic, "kinetic")
    call ctx%compute_one_electron_integrals(nuclear_attraction, "nuclear_attraction")
    call ctx%compute_one_electron_integrals(dipole, "moment", origin=[0.0_real64, 0.0_real64, 0.0_real64], order=1)

    call dump("overlap", overlap)
    call dump("kinetic", kinetic)
    call dump("nuclear_attraction", nuclear_attraction)
    call dump("dipole (x component)", dipole(:,:,1))

    print '(A,F12.6)', "trace(overlap)    = ", trace(overlap)
    print '(A,I0)',    "expected (n_cont) = ", n

    print '(A,ES10.3)', "max asymmetry(overlap)            = ", max_asymmetry(overlap)
    print '(A,ES10.3)', "max asymmetry(kinetic)            = ", max_asymmetry(kinetic)
    print '(A,ES10.3)', "max asymmetry(nuclear_attraction) = ", max_asymmetry(nuclear_attraction)
    do i = 1, 3
        print '(A,I0,A,ES10.3)', "max asymmetry(dipole component ", i, ") = ", max_asymmetry(dipole(:,:,i))
    end do

    if (abs(trace(overlap) - real(n, kind=real64)) < 1.0e-8_real64) then
        print '(A)', "PASS: trace(overlap) matches n_cont"
    else
        print '(A)', "FAIL: trace(overlap) does not match n_cont"
        error stop 1
    end if

    ! The SAP screening of both fit sets: symmetric and repulsive (positive diagonal), the two sets different.
    call ctx%compute_one_electron_integrals(sap_helfem, "sap_helfem")
    call ctx%compute_one_electron_integrals(sap_grasp, "oneint_sap_grasp")
    if (max_asymmetry(sap_helfem) < 1.0e-12_real64 .and. max_asymmetry(sap_grasp) < 1.0e-12_real64 &
        .and. all([(sap_helfem(i,i) > 0.0_real64 .and. sap_grasp(i,i) > 0.0_real64, i = 1, n)])) then
        print '(A)', "PASS: the SAP screenings are symmetric with a positive diagonal"
    else
        print '(A)', "FAIL: the SAP screenings are not symmetric with a positive diagonal"
        error stop 1
    end if
    if (maxval(abs(sap_helfem - sap_grasp)) > 1.0e-6_real64) then
        print '(A)', "PASS: the two SAP fit sets differ"
    else
        print '(A)', "FAIL: the two SAP fit sets agree"
        error stop 1
    end if

    print '(A)', "=== ctx freed automatically (FINAL) when it goes out of scope below ==="

    ! Every kind of centre (the AIMP from test_fixtures/basis: AEDELSTEN_BASIS_LIBRARY=test_fixtures/basis), and the
    ! Cartesian rescaling and its inverse.
    call centres%set_aedelsten_context("test_fixtures/centres_demo.inp")
    call centres%get_centre_geometry(z, xyz, kinds)
    call centres%get_nuclear_charges(charges)
    if (centres%get_n_centres() /= 3 .or. any(z /= [8, 1, 12]) .or. kinds(1) /= aedelsten_constant("atom_kind_real") &
        .or. kinds(2) /= aedelsten_constant("atom_kind_ghost") .or. kinds(3) /= aedelsten_constant("atom_kind_orthogonality_ghost") &
        .or. xyz(3, 3) /= 5.0_real64 .or. any(charges /= [8.0_real64, 0.0_real64, 0.0_real64])) then
        print '(A)', "FAIL: the centres, their kinds or nuclear charges"
        stop 1
    end if
    allocate(m(centres%get_n_cont(), 2))
    do i = 1, size(m)
        m(mod(i - 1, size(m, 1)) + 1, (i - 1) / size(m, 1) + 1) = 0.1_real64 + i
    end do
    m0 = m
    call centres%rescale_cartesian_ao_rows("molcas", 2, m, forward)
    call centres%unrescale_cartesian_ao_rows("molcas", 2, m, back)
    if (.not. (forward .and. back) .or. maxval(abs(m - m0) / abs(m0)) > 1.0e-12_real64) then
        print '(A)', "FAIL: the Cartesian rescaling's inverse"
        stop 1
    end if
    print '(A)', "PASS: the centres, their kinds and nuclear charges, and the Cartesian rescaling's inverse"

    ! The overlap with another basis: with itself the overlap (unscreened, so to the screening threshold), with the
    ! centres' basis (only ctx has a scratch; it serves both) the transpose of the other order.
    allocate(self_overlap(n, n), cross(n, centres%get_n_cont()), cross_back(centres%get_n_cont(), n))
    call ctx%compute_overlap_with(ctx, self_overlap)
    call ctx%compute_overlap_with(centres, cross)
    call centres%compute_overlap_with(ctx, cross_back)
    if (maxval(abs(self_overlap - overlap)) > 1.0e-10_real64 .or. maxval(abs(cross - transpose(cross_back))) > 1.0e-14_real64) then
        print '(A)', "FAIL: compute_overlap_with"
        stop 1
    end if
    print '(A)', "PASS: compute_overlap_with itself is the overlap, with another basis the transpose of the other order"

    ! The frees of the primitive shells, the contraction matrix and the input: each empties its own field of the
    ! memory breakdown (1 qm_region, 8 primitive_shells, 9 primitive_shell_pairs, 10 contraction_matrix).
    call ctx%set_primitive_shell_pairs()
    allocate(before(memory_breakdown_field_count()), after(memory_breakdown_field_count()))
    call ctx%memory_breakdown(before)
    call ctx%free_primitive_shells()
    call ctx%free_contraction_matrix()
    call ctx%free_input()
    call ctx%memory_breakdown(after)
    if (any(before([1, 8, 10]) == 0) .or. any(after([1, 8, 10]) /= 0) .or. after(9) /= before(9)) then
        print '(A)', "FAIL: free_primitive_shells, free_contraction_matrix and free_input"
        stop 1
    end if
    print '(A)', "PASS: free_primitive_shells, free_contraction_matrix and free_input empty their fields only"

    ! The timings and the reference list of what ran above.
    found = .false.
    do k = 0, aedelsten_timing_count(.false.) - 1
        call aedelsten_timing_span(.false., k, label, seconds, calls)
        if (label == "oneint::overlap" .and. calls >= 1) found = .true.
    end do
    call aedelsten_span_tree_node(0, label, depth, fine, seconds, calls)
    n_references = aedelsten_citation_count()
    found = found .and. aedelsten_span_tree_count() > 0 .and. depth == 0 .and. aedelsten_mpi_collective_seconds() >= 0.0d0
    do k = 0, n_references - 1
        found = found .and. len(aedelsten_citation(k)) > 0 .and. len(aedelsten_citation(k, topics=.true.)) > 0
    end do
    bibtex = aedelsten_citations_bibtex()
    if (.not. found .or. n_references == 0 .or. bibtex(1:1) /= "@") then
        print '(A)', "FAIL: the timings or the reference list"
        stop 1
    end if
    call aedelsten_clear_citations()
    if (aedelsten_citation_count() /= 0 .or. len(aedelsten_citations_bibtex()) /= 0) then
        print '(A)', "FAIL: aedelsten_clear_citations"
        stop 1
    end if
    print '(A,I0,A)', "PASS: the timings, the span tree and the reference list (", n_references, " references), cleared"

contains

    subroutine dump(label, buf)
        character(len=*),  intent(in) :: label
        real(kind=real64), intent(in) :: buf(:,:)
        integer :: r, c
        print '(A)', "--- " // trim(label) // " ---"
        do r = 1, size(buf, 1)
            do c = 1, size(buf, 2)
                write(*, '(F10.5)', advance="no") buf(r, c)
            end do
            print *
        end do
    end subroutine dump

    function trace(buf) result(tr)
        real(kind=real64), intent(in) :: buf(:,:)
        real(kind=real64) :: tr
        integer :: i
        tr = 0.0_real64
        do i = 1, size(buf, 1)
            tr = tr + buf(i, i)
        end do
    end function trace

    function max_asymmetry(buf) result(worst)
        real(kind=real64), intent(in) :: buf(:,:)
        real(kind=real64) :: worst
        integer :: r, c
        worst = 0.0_real64
        do r = 1, size(buf, 1)
            do c = 1, size(buf, 2)
                worst = max(worst, abs(buf(r, c) - buf(c, r)))
            end do
        end do
    end function max_asymmetry

end program oneint_demo

