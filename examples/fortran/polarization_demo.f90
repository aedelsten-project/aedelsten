! SPDX-FileCopyrightText: 2024 Ernst D. Larsson
! SPDX-License-Identifier: MIT

! Demonstration of Aedelsten's induced-dipole (polarizable) embedding from Fortran: the calls a
! SCF/CI/CC/response program makes, with self-checks that need no reference data.
!
!   set_polarizable_sites -> set_polarization(options) -> compute_induction            (per Fock build)
!                                                      -> compute_induction_response   (per trial-vector batch)
!                                                      -> compute_induction_states     (per-state densities)
!
! Every matrix is the caller's own (n,n) [or (n,n,k)] array; sizes come from the shapes and nothing is
! copied. Build as in oneint_demo.f90 (same module/library pair), run from the repository root.
program polarization_demo
    use iso_fortran_env, only: real64, int64
    use iso_c_binding,   only: c_int, c_int32_t, c_double
    use Aedelsten,       only: AedelstenContext, AedelstenPolarizationOptions
    implicit none

    type(AedelstenContext)             :: ctx, aimps
    type(AedelstenPolarizationOptions) :: options      ! defaults; edit what you care about
    real(kind=real64), allocatable     :: d(:,:), v(:,:), d3(:,:,:), v3(:,:,:), vs(:,:,:), mu(:,:), energies(:)
    real(kind=real64)                  :: energy, energy_plus, energy_minus, epsilon, analytic, numeric, induced(3)
    real(kind=real64)                  :: couplings(3,3), contracted(3,3)
    integer                            :: n, i, j, k, m, iterations, from_static, from_given
    real(kind=real64)                  :: e_solved, e_started, solution(3, 3), xyz(3, 3), charges(3)
    integer(kind=c_int32_t)            :: atomic_numbers(3)
    logical                            :: ok
    real(kind=real64)                  :: sites(3, 3), field(3, 3), at_points(3, 3), nuclear(3, 3)
    real(kind=real64), allocatable     :: integrals(:,:,:)
    integer(kind=int64)                :: indices(3)
    integer(kind=int64)                :: storage(4), unkept(4)
    integer                            :: loose, tight
    real(kind=real64)                  :: e_loose, e_tight, energies_within(3)

    call ctx%set_aedelsten_context("test_fixtures/mpi_oneint_check.inp")
    call ctx%set_shell_pairs()
    call ctx%set_scratch(64, "mib")
    n = ctx%get_n_cont()

    call ctx%set_polarizable_sites("test_fixtures/polarization_demo.xfield")
    options%threshold      = 1.0e-12_c_double  ! tight solver
    options%response_batch = 2_c_int           ! enables compute_induction_response (2 densities per pass)
    options%state_slots    = 3_c_int           ! enables compute_induction_states (up to 3 states)
    call ctx%set_polarization(options)
    print '(A,I0,A,I0)', "n_cont = ", n, ", polarizable sites = ", ctx%get_n_polarizable_sites()

    allocate(d(n,n), v(n,n), d3(n,n,3), v3(n,n,3), vs(n,n,3), energies(3), mu(3, ctx%get_n_polarizable_sites()))
    do j = 1, n
        do i = 1, n
            d(i,j) = 0.02_real64 / (1.0_real64 + i + j)
        end do
    end do

    ! Ground state: one call per Fock build; v is overwritten, energy goes into the total energy.
    call ctx%compute_induction(d, v, energy, iterations)
    call ctx%get_induced_dipoles(mu)
    print '(A,ES14.6,A,I0,A)', "E_ind = ", energy, "   (", iterations, " solver iterations)"

    ! Linear response of the induction along a symmetric direction x = d3(:,:,1), checked against a
    ! finite difference of the ground-state operator (the operator is affine in the density).
    do m = 1, 3
        do j = 1, n
            do i = 1, n
                d3(i,j,m) = real(mod((3 + m) * min(i,j) + (5 + 2*m) * max(i,j), 7) - 3, real64)
            end do
        end do
    end do
    epsilon = 1.0e-2_real64
    call ctx%compute_induction(d + epsilon * d3(:,:,1), v3(:,:,1), energy_plus)
    call ctx%compute_induction(d - epsilon * d3(:,:,1), v3(:,:,2), energy_minus)
    numeric = maxval(abs((v3(:,:,1) - v3(:,:,2)) / (2 * epsilon)))
    call ctx%compute_induction_response(d3(:,:,1), v)               ! one density
    call ctx%compute_induction_response(d3, vs)                     ! a batch of three
    ok = maxval(abs(v - (v3(:,:,1) - v3(:,:,2)) / (2 * epsilon))) < 1.0e-8_real64 * numeric &
         .and. maxval(abs(vs(:,:,1) - v)) < 1.0e-12_real64
    call check(ok, "response operator == finite difference of the induction operator; batch == single")
    ! The couplings between the three densities: tr(D_m V1[D_n]) without forming the operators.
    call ctx%compute_induction_couplings(d3, couplings)
    do k = 1, 3
        do m = 1, 3
            contracted(m, k) = sum(d3(:,:,m) * vs(:,:,k))
        end do
    end do
    ok = maxval(abs(couplings - contracted)) < 1.0e-10_real64 * maxval(abs(contracted)) &
         .and. maxval(abs(couplings - transpose(couplings))) < 1.0e-10_real64 * maxval(abs(couplings))
    call check(ok, "couplings == tr(D_m V1[D_n]) of the response operators; symmetric")

    ! States with their own dipoles: operators(:,:,m) and energies(m) are those of densities(:,:,m) alone.
    do m = 1, 3
        d3(:,:,m) = d + 0.01_real64 * d3(:,:,m)
    end do
    call ctx%compute_induction_states(d3, v3, energies)
    call ctx%compute_induction(d3(:,:,2), v, energy)
    call check(abs(energies(2) - energy) < 1.0e-9_real64 .and. maxval(abs(v3(:,:,2) - v)) < 1.0e-9_real64, &
               "state 2 == compute_induction of its own density")
    energies_within = energies
    call ctx%compute_induction_states(d3, v3, energies, threshold=1.0e-12_real64)
    call check(maxval(abs(energies - energies_within)) < 1.0e-9_real64, "compute_induction_states(threshold): the same energies")

    ! The effective external field: tr(d V^(i)) is minus the dipole the density's electrons induce at the sites.
    call ctx%compute_effective_external_field(vs)
    call ctx%compute_induction(d, v, energy)
    call ctx%get_induced_dipoles(mu)
    induced = sum(mu, dim=2)
    call ctx%compute_induction(0.0_real64 * d, v, energy)
    call ctx%get_induced_dipoles(mu)
    induced = induced - sum(mu, dim=2)
    ok = .true.
    do m = 1, 3
        ok = ok .and. abs(sum(d * vs(:,:,m)) + induced(m)) < 1.0e-9_real64 * maxval(abs(induced))
    end do
    call check(ok, "EEF: tr(d V) == -(dipole induced by d)")

    ! The dipoles' start (another geometry's, in an optimization): from the solution, the next solve is shorter and the
    ! same.
    call ctx%compute_induction(d, v, e_solved)
    call ctx%get_induced_dipoles(solution)
    call ctx%free_polarization()
    call ctx%set_polarization(options)
    call ctx%compute_induction(d, v, energy, from_static)
    call ctx%free_polarization()
    call ctx%set_polarization(options)
    call ctx%set_induced_dipoles(solution)
    call ctx%compute_induction(d, v, e_started, from_given)
    call check(from_given < from_static .and. abs(e_started - e_solved) < 1.0e-10_real64, &
               "set_induced_dipoles: the solve starts from them")

    ! The field integrals kept between calls (a budget of 64 MB): the same induction.
    call ctx%free_polarization()
    options%resident_budget = 64_c_int
    options%resident_budget_unit = 1_c_int
    call ctx%set_polarization(options)
    call ctx%compute_induction(d, v, e_started)
    call check(abs(e_started - e_solved) < 1.0e-10_real64, "resident field integrals: the same induction")
    call ctx%get_polarization_storage(storage)
    call check(storage(2) == storage(1) .and. storage(4) > 0_int64 .and. storage(4) <= 64_int64 * 1024_int64**2, &
               "storage: all kept within the budget")

    ! Without a budget the integrals are evaluated in every pass (the same induction), and the storage says what
    ! keeping them would take.
    call ctx%free_polarization()
    options%resident_budget = 0_c_int
    call ctx%set_polarization(options)
    call ctx%get_polarization_storage(unkept)
    call check(unkept(2) == 0_int64 .and. unkept(4) == storage(4), "storage: none kept without a budget, the same need")
    call ctx%compute_induction(d, v, e_loose)
    call check(abs(e_loose - e_solved) < 1.0e-10_real64, "evaluated field integrals: the same induction")
    ! A budget set afterwards keeps what it holds: all of them here, the same induction.
    call ctx%set_resident_field_budget(64, "mib")
    call ctx%get_polarization_storage(unkept)
    call check(all(unkept == storage), "set_resident_field_budget: all kept, the same need")
    call ctx%compute_induction(d, v, e_loose)
    call check(abs(e_loose - e_solved) < 1.0e-10_real64, "field integrals kept after the polarization: the same induction")
    call ctx%free_polarization()
    call ctx%set_polarization(options)

    ! A looser threshold for one call: fewer iterations, the energy off by about the square of the residual.
    call ctx%compute_induction(d, v, e_loose, loose, threshold=1.0e-4_real64)
    call ctx%free_polarization()
    call ctx%set_polarization(options)
    call ctx%compute_induction(d, v, e_tight, tight)
    call check(loose < tight .and. abs(e_loose - e_tight) < 1.0e-6_real64 * abs(e_tight), &
               "compute_induction(threshold): fewer iterations, the energy to the residual squared")

    ! The field at the sites: the electronic field is the field integrals contracted with the density, the same at
    ! custom points on the sites; the static field here is the nuclei's alone; the sites from the records 0, 1, 2.
    call check(ctx%has_polarizable_sites() .and. ctx%has_polarization(), "has_polarizable_sites, has_polarization")
    call ctx%get_input_indices("polarizable", indices)
    call check(ctx%count_electric_field_sites("polarizable") == 3 .and. all(indices == [0_int64, 1_int64, 2_int64]), &
               "the polarizable sites and their input records")
    sites = reshape([6.0_real64, 0.0_real64, 0.0_real64, -5.0_real64, 4.0_real64, 4.0_real64, &
                     0.0_real64, -6.0_real64, 3.0_real64], [3, 3])
    call ctx%compute_electronic_field(d, "polarizable", field, chunk_sites=2_int64)
    allocate(integrals(n, n, 9))
    call ctx%compute_electric_field_integrals("polarizable", 0_int64, integrals)
    do k = 1, 3
        do m = 1, 3
            contracted(m, k) = sum(integrals(:, :, 3 * (k - 1) + m) * d)
        end do
    end do
    call ctx%compute_electronic_field(d, "custom", at_points, points=sites)
    call check(maxval(abs(field - contracted)) < 1.0e-12_real64 * maxval(abs(field)) .and. &
               maxval(abs(at_points - field)) < 1.0e-12_real64 * maxval(abs(field)), &
               "electronic field == contracted field integrals == the field at custom points on the sites")
    call ctx%compute_static_field("polarizable", nuclear, core_aimps=.false., embedding_aimps=.false., external_multipoles=.false.)
    call ctx%compute_static_field("polarizable", field)
    call check(all(field == nuclear) .and. maxval(abs(nuclear)) > 0.0_real64, "static field: only nuclei here")
    call ctx%get_input_indices("custom", indices, points=sites)
    call check(all(indices == -1_int64), "custom points come from no input record")
    call ctx%set_multipole_convention("exact")
    call check(ctx%get_multipole_convention() == "exact", "set_multipole_convention")
    call ctx%set_multipole_convention("contact_free")

    call ctx%free_polarization()
    call ctx%free_polarizable_sites()
    call check(.not. (ctx%has_polarizable_sites() .or. ctx%has_polarization()), "freed")

    ! The embedding AIMPs' sites: an Mg orthogonality block and two O, in input order.
    call aimps%set_aedelsten_context("test_fixtures/embedding_aimp_sites.inp")
    call aimps%get_embedding_aimp_sites(atomic_numbers, xyz, charges)
    call check(all(atomic_numbers == [12, 8, 8]) .and. all(charges == [2.0_real64, -2.0_real64, -2.0_real64]) &
               .and. all(xyz(:, 3) == [0.5_real64, 4.0_real64, -2.5_real64]), "embedding AIMP sites: elements, positions, charges")
    call check(aimps%count_electric_field_sites("embedding_aimps") == 3 .and. aimps%count_electric_field_sites("qm") == 3, &
               "embedding AIMP and QM sites counted")
    print '(A)', "PASS"

contains

    subroutine check(condition, label)
        logical,          intent(in) :: condition
        character(len=*), intent(in) :: label
        if (condition) then
            print '(A)', "ok:   " // label
        else
            print '(A)', "FAIL: " // label
            error stop 1
        end if
    end subroutine check

end program polarization_demo
