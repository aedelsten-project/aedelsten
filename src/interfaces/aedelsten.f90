! SPDX-FileCopyrightText: 2024 Ernst D. Larsson
! SPDX-License-Identifier: MIT

module Aedelsten
    use iso_c_binding, only: c_int32_t, c_int64_t, c_ptr, c_int, c_double, c_char, c_null_char, c_null_ptr, c_associated, c_loc, c_f_pointer
    use iso_fortran_env, only: int32, int64, error_unit
    use ieee_arithmetic, only: ieee_is_nan
    implicit none
    private

    ! ── Public API ──────────────────────────────────────────────────────────
    public :: AedelstenContext
    ! A named constant of the library by its key ("atom_kind_real", "cartesian_convention_molcas", ...).
    public :: aedelsten_constant
    public :: AedelstenPolarizationOptions
    ! Free functions (not type-bound): these operate purely on the arrays a
    ! caller passes in (a dense ERI tensor + density matrices), needing no
    ! context state, so unlike everything else here they aren't methods on
    ! AedelstenContext.
    public :: compute_j_and_k_matrix_stored
    public :: compute_j_and_k_unrestricted_matrix_stored
    public :: compute_j_matrix_stored
    public :: compute_k_matrix_stored
    public :: compute_k_matrix_unrestricted_stored
    public :: memory_breakdown_field_count
    ! The scratch query's operations (scratch_bytes): a column of operations(5, n) per operation, the kind and up
    ! to four parameters, scratch_none for a parameter's None.
    public :: scratch_eri_data, scratch_restricted_fock, scratch_unrestricted_fock
    public :: scratch_two_electron_gradient, scratch_general_fock, scratch_general_multi_fock
    public :: scratch_active_space_fock, scratch_mixed_active_space_fock, scratch_half_transform, scratch_amfi
    public :: scratch_none
    public :: scratch_breakdown_field_count
    ! MPI for a program without MPI of its own. A program that has MPI initializes it itself and passes its
    ! communicators as their Fortran handles (`comm` with `use mpi`, `comm%MPI_VAL` with `use mpi_f08`).
    public :: aedelsten_mpi_init, aedelsten_mpi_finalize, aedelsten_mpi_initialized, aedelsten_mpi_thread_level
    public :: aedelsten_mpi_rank, aedelsten_mpi_size
    public :: AedelstenSharedArray, aedelsten_shared_array, aedelsten_shared_array_sync, aedelsten_shared_array_free
    ! The BLAS's thread count, the process's, set from the main thread only, outside any threaded region.
    public :: aedelsten_blas_set_threads, aedelsten_blas_threads, aedelsten_blas_info, aedelsten_blas_available_cores
    ! The library's version and the git commit it was built from.
    public :: aedelsten_get_version, aedelsten_get_source_commit
    public :: aedelsten_mpi_collective_seconds, aedelsten_profile_enabled, aedelsten_timing_count, aedelsten_timing_span
    public :: aedelsten_span_tree_count, aedelsten_span_tree_node, aedelsten_citation_count, aedelsten_citation
    public :: aedelsten_citations_bibtex, aedelsten_clear_citations

    ! Working kinds
    integer(kind=c_int), parameter :: aedelsten_int_size = c_int
!   A caller passes its reals as c_double (real64, real*8).
    integer(kind=aedelsten_int_size), parameter :: aedelsten_float_size = c_double
#ifdef AEDELSTEN_FORTRAN_INT64
    integer(kind=aedelsten_int_size), parameter :: external_int_size = int64
#else
    integer(kind=aedelsten_int_size), parameter :: external_int_size = int32
#endif

    ! Wire-key prefix Aedelsten's one-electron operator constants are queried
    ! under on the Rust side (crate::context::one_electron::OneElectronOperatorKind's
    ! NAME_TABLE, e.g. "oneint_overlap") -- callers may pass either the bare
    ! suffix ("overlap") or the fully-qualified key ("oneint_overlap").
    character(len=*), parameter :: oneint_label = "oneint_"

    ! Wire-key prefix for ERI AO-symmetry constants (the library's context::two_electron::
    ! ERIAOSymmetry::NAME_TABLE, e.g. "eri_ao_symmetry_s1") -- same
    ! bare-suffix-or-full-key convention as oneint_label above.
    character(len=*), parameter :: eri_symmetry_label = "eri_ao_symmetry_"
    character(len=*), parameter :: eri_range_label = "eri_"

    ! The scratch query's operation kinds (the C codes) and their parameters (in rows 2 to 5 of a column):
    ! two_electron_gradient (terms), general_multi_fock (densities), active_space_fock and mixed_active_space_fock
    ! (active), half_transform (left, right, sigma, node_ranks; right scratch_none for one basis, sigma for all);
    ! amfi none (over the primitive shells whatever primitive says).
    integer(kind=aedelsten_int_size), parameter :: scratch_eri_data = 1, scratch_restricted_fock = 2, &
        scratch_unrestricted_fock = 3, scratch_two_electron_gradient = 4, scratch_general_fock = 5, &
        scratch_general_multi_fock = 6, scratch_active_space_fock = 7, scratch_mixed_active_space_fock = 8, &
        scratch_half_transform = 9, scratch_amfi = 10, scratch_none = -1


    ! Options of set_polarization: the same fields, in the same order, as the C struct
    ! AedelstenPolarizationOptions (include/aedelsten.h). The default initialisers below ARE the library's
    ! defaults (a negative number means "the library's default"), so `type(AedelstenPolarizationOptions) :: o`
    ! needs editing only where you care: o%threshold = 1.0e-10_c_double, o%response_batch = 4_c_int, ...
    type, bind(C), public :: AedelstenPolarizationOptions
        integer(kind=c_int) :: damping        = 0      ! 0: none, 1: Thole
        real(kind=c_double) :: thole_a        = -1.0d0 ! Thole parameter a; negative: 2.1304
        real(kind=c_double) :: threshold      = -1.0d0 ! dipole-solver convergence; negative: 1e-8
        integer(kind=c_int) :: max_iterations = -1     ! negative: library default
        integer(kind=c_int) :: static_field   = 0      ! 0: keep the static field in memory, 1: recompute it
        integer(kind=c_int) :: chunk_sites    = -1     ! sites per integral chunk; not positive: use the budget
        integer(kind=c_int) :: chunk_budget    = -1     ! integral-chunk memory budget (in chunk_budget_unit); negative: default
        integer(kind=c_int) :: chunk_budget_unit    = 1      ! byte unit: 0 GiB, 1 MiB, 2 KiB
        integer(kind=c_int) :: response_batch = 0      ! densities per compute_induction_response batch; 0: none
        integer(kind=c_int) :: state_slots    = 0      ! states held by compute_induction_states; 0: none
        integer(kind=c_int) :: resident_budget  = 0      ! budget for keeping the field integrals between calls; 0: never
        integer(kind=c_int) :: resident_budget_unit  = 1      ! byte unit of resident_budget: 0 GiB, 1 MiB, 2 KiB
    end type AedelstenPolarizationOptions

    ! `data` doubles in the shared memory of a communicator's ranks on one node (aedelsten_shared_array): the same
    ! memory on every rank. Freed by aedelsten_shared_array_free, collectively.
    type :: AedelstenSharedArray
        type(c_ptr) :: handle = c_null_ptr
        real(kind=aedelsten_float_size), pointer :: data(:) => null()
    end type AedelstenSharedArray

    ! Opaque context — wraps the raw C pointer so user never touches c_ptr directly
    type :: AedelstenContext
        private
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: set_aedelsten_context
        procedure :: set_shell_pairs
        procedure :: free_shell_pairs
        procedure :: free_input
        procedure :: free_primitive_shells
        procedure :: free_contraction_matrix
        procedure :: set_scratch
        procedure :: set_communicator
        procedure :: resize_scratch
        procedure :: get_scratch_peak_bytes
        procedure :: free_scratch
        procedure :: get_n_cont
        procedure :: get_n_prim
        procedure :: get_n_prim_unique
        procedure :: get_n_primitive_rows
        ! Per-field breakdown of how much memory this context is actually holding right now.
        ! `bytes` must be pre-allocated to memory_breakdown_field_count() -- see that free
        ! function's own comment for the fixed field order.
        procedure :: memory_breakdown
        ! The basis's description for an orbital file (as Molcas's BASIS_FUNCTION_IDS, PRIMITIVE_IDS, PRIMITIVES and
        ! CENTER_LABELS) and the Cartesian rescaling a reader of another normalization needs (BasisFunctionId,
        ! PrimitiveId, Primitive, CentreLabel and SharedCartesianReference in the library's documentation).
        procedure :: get_basis_function_ids
        procedure :: get_primitive_ids
        procedure :: get_primitives
        procedure :: get_centre_label
        procedure :: rescale_cartesian_ao_rows
        procedure :: unrescale_cartesian_ao_rows
        procedure :: get_n_centres
        procedure :: get_centre_geometry
        procedure :: get_nuclear_charges
        procedure :: rescale_cartesian_ao_columns
        ! The model basis: the free-atom reference orbitals (ANO-RCC) of the model Fockian, for an
        ! atomic frozen core -- their descriptions, their overlaps, and the model Fockian itself.
        procedure :: get_n_model_functions
        procedure :: get_model_functions
        procedure :: compute_model_basis_overlaps
        procedure :: compute_model_fockian
        ! The overlap with another context's basis (any basis, geometry, harmonics, contraction), unscreened.
        procedure :: compute_overlap_with
        ! Molecular properties, and the finite (Visscher-Dyall) nuclear
        ! charge model -- point charges are the default; call this once,
        ! before any integral involving nuclear attraction, to switch.
        procedure :: set_finite_nuclear_model
        procedure :: compute_centre_of_mass
        procedure :: compute_nuclear_repulsion_energy
        procedure :: compute_nuclear_charge
        procedure :: compute_nuclear_multipole_interaction_energy
        procedure :: compute_nuclear_aimp_embedding_energy
        ! Primitive (decontracted) shell pairs -- required before
        ! compute_one_electron_integrals(..., primitive=.true.) or any
        ! picture-change/contract-back call below.
        procedure :: set_primitive_shell_pairs
        procedure :: free_primitive_shell_pairs
        procedure :: has_primitive
        ! External point multipoles (charges, dipoles, quadrupoles, ...) -- once
        ! loaded, accessed through the ordinary compute_one_electron_integrals
        ! with operator "external_multipole" (every order that was loaded here).
        procedure :: set_external_multipoles
        procedure :: free_external_multipoles
        procedure :: has_external_multipoles
        ! Induced-dipole (polarizable) embedding. The intended order is: set_polarizable_sites (xfield file),
        ! optionally set_external_multipoles, then set_polarization; then one compute_induction per Fock
        ! build (density -> operator, energy), plus the batched response / per-state variants below for
        ! CI, CC and response methods. Every matrix is a caller-owned (n,n) array (batches: (n,n,k));
        ! sizes come from the array shapes, nothing is copied or retained.
        procedure :: set_polarizable_sites
        procedure :: free_polarizable_sites
        procedure :: get_n_polarizable_sites
        procedure :: has_polarizable_sites
        procedure :: has_polarization
        ! The electric field at sites. A selection is "qm" (the nuclei and core AIMPs), "embedding_aimps",
        ! "external_multipoles", "polarizable" or "custom" (then `points`, (3, n_points), bohr). Fields are
        ! (3, n_sites); first_site and the input indices are 0-based, an index -1 for a site from no input record.
        procedure :: count_electric_field_sites
        procedure :: get_input_indices
        procedure :: compute_electric_field_integrals
        procedure :: compute_static_field
        procedure :: compute_electronic_field
        ! How the external multipoles meet the electrons: "contact_free" (the default) or "exact".
        procedure :: set_multipole_convention
        procedure :: get_multipole_convention
        procedure :: set_polarization
        procedure :: free_polarization
        procedure :: compute_induction
        ! How many of its sites' field integrals this rank keeps in memory, and the bytes keeping them needs.
        procedure :: get_polarization_storage
        ! The memory budget of this rank's kept field integrals, in place of set_polarization's.
        procedure :: set_resident_field_budget
        ! dE_ind/dX at fixed density, (3, n_centres), from the dipoles of the last compute_induction.
        procedure :: compute_induction_gradient
        procedure :: compute_induction_states
        procedure :: get_induced_dipoles
        ! The start of the next compute_induction's solve (another geometry's dipoles), as get_induced_dipoles gives
        ! them.
        procedure :: set_induced_dipoles
        procedure :: compute_effective_external_field
        procedure, private :: compute_induction_response_matrix
        procedure, private :: compute_induction_response_tensor
        generic             :: compute_induction_response => &
            compute_induction_response_matrix, compute_induction_response_tensor
        ! The induction's couplings K(m,n) between (n,n,k) densities, without forming operators.
        procedure :: compute_induction_couplings
        ! AIMP (ab initio model potential) embedding. Core AIMP regions (if
        ! any) are already resident once the context exists -- has_core_aimp
        ! just reports whether the input actually had any; their integrals
        ! are reached the same way as everything else, via
        ! compute_one_electron_integrals with e.g. "core_aimp_coulomb" /
        ! "core_aimp_exchange". Embedding AIMP regions need
        ! set_embedding_aimp_operators() called first (materializes their
        ! projection operators), then "embedding_aimp_coulomb" /
        ! "embedding_aimp_exchange" the same way.
        procedure :: has_core_aimp
        procedure :: has_embedding_aimp
        procedure :: get_n_embedding_aimps
        ! Every embedding AIMP's atomic number, position (bohr) and charge, in input order.
        procedure :: get_embedding_aimp_sites
        procedure :: set_embedding_aimp_operators
        procedure :: free_embedding_aimp_operators
        ! compute_one_electron_integrals is generic on output's rank: a
        ! square (n,n) matrix for plain operators (overlap, kinetic, ...), a
        ! square-in-(1,2) (n,n,n_planes) tensor for the moment operator --
        ! n/n_planes are derived from output's real shape in both cases, not
        ! taken as separate scalar arguments a caller's array could silently
        ! disagree with -- or a flat rank-1 buffer with n/n_planes passed
        ! explicitly (legitimate for callers managing their own linear
        ! storage), still checked against output's real size rather than
        ! trusted outright. See each concrete procedure's own comment.
        procedure, private :: compute_one_electron_integrals_matrix
        procedure, private :: compute_one_electron_integrals_tensor
        procedure, private :: compute_one_electron_integrals_flat
        generic             :: compute_one_electron_integrals => &
            compute_one_electron_integrals_matrix, compute_one_electron_integrals_tensor, &
            compute_one_electron_integrals_flat
        ! The gradient of one operator contracted with a density, for ALL centres at once: fills a
        ! (3, get_n_gradient_centres()) matrix. One operator per call; the caller sums the operators it needs.
        procedure :: get_n_gradient_centres
        procedure :: compute_one_electron_gradient
        ! A geometry step: move the gradient centres of this context (same (3, n_centres) layout as the gradients).
        procedure :: set_centre_positions
        ! Gradients of the classical energies (compute_nuclear_*_energy), one call each, (3, n_centres).
        procedure :: compute_nuclear_repulsion_gradient
        procedure :: compute_nuclear_aimp_embedding_gradient
        procedure :: compute_nuclear_multipole_interaction_gradient
        ! The shells, their extents, and the AO values (and gradients) of a list of shells.
        procedure :: get_n_shells
        procedure :: get_shells
        procedure :: compute_shell_extents
        procedure :: compute_ao_values
        procedure :: compute_ao_hessians
        ! Two-electron repulsion integrals (ERIs). AO index order for the
        ! S4/S8 symmetry-packed forms is always column-major internally --
        ! Fortran has no other array order, so unlike the C/Python
        ! interfaces there is no row/column-major choice exposed here at
        ! all. Rank-4 output is always S1 (dense, shape self-describes, no
        ! ao_symmetry argument needed); rank-1 output needs an explicit
        ! ao_symmetry ("s1"/"s4"/"s8") since S4 and S8 are both 1-D packed
        ! and can't be told apart by shape. range/omega are bundled like
        ! origin/order elsewhere: omit both for the (default) full-range
        ! Coulomb operator, or pass both together for range-separated.
        procedure, private :: compute_eri_stored_tensor
        procedure, private :: compute_eri_stored_flat
        generic             :: compute_eri_stored => &
            compute_eri_stored_tensor, compute_eri_stored_flat
        procedure :: compute_s4_size
        procedure :: compute_s8_size
        ! Direct Fock construction: no stored ERI tensor -- shell-quartet
        ! integrals are evaluated and immediately folded into J/K, one
        ! shell quartet at a time. Requires set_eri_data() first. Unlike
        ! compute_j_and_k_matrix_stored and friends above, these are
        ! type-bound, not free functions: they genuinely need context
        ! state (the shell pairs and cached eri_data), whereas those
        ! operate purely on a caller-supplied dense tensor.
        procedure :: set_eri_data
        procedure :: free_eri_data
        procedure :: compute_fock_direct
        procedure :: compute_fock_direct_unrestricted
        procedure :: compute_fock_direct_general
        procedure :: compute_fock_direct_general_multi
        procedure :: compute_fock_direct_mixed_active_space
        procedure :: compute_fock_direct_active_space
        ! T(u, vw, mu) from the active walks' M, one index transformed.
        procedure :: transform_pair_coulomb_one_index
        ! The half-transformed ERIs X(p,nu,r,sigma) from one direct walk (MRPT2, integral-direct MP2).
        procedure :: compute_eri_half_transformed
        procedure :: compute_eri_half_transformed_on_node
        procedure :: compute_eri_half_transformed_sigma
        procedure :: compute_eri_half_transformed_sigma_on_node
        procedure :: half_transform_scratch_bytes
        ! The scratch every operation fits in, from the input alone, and per operation what sets it.
        procedure :: scratch_bytes
        procedure :: scratch_breakdown
        ! The contracted twin of the direct J/K builds: the two-electron energy gradient for every gradient centre,
        ! into a (3, get_n_gradient_centres()) matrix, from a list of density-pair terms (RHF, UHF, two densities).
        procedure :: compute_two_electron_gradient
        ! Relativistic picture-change integrals -- always over the
        ! decontracted (primitive) basis (requires set_primitive_shell_pairs()
        ! first; no separate primitive= toggle, unlike
        ! compute_one_electron_integrals). Flat/1-D output only: unlike the
        ! plain integrals above, spin_orbit changes output's *rank* on top of
        ! whatever the operator's own shape already is (a plain matrix
        ! operator becomes a 3-D (x,y,z,scalar) tensor under spin_orbit; an
        ! already-3-D operator's plane count multiplies by 4 instead) --
        ! rank alone can no longer tell apart "spin-orbit of a matrix
        ! operator" from "scalar picture-change of a tensor operator" (both
        ! are rank 3), so this doesn't get the same rank-dispatched
        ! (n,n)/(n,n,n_planes) convenience the plain integrals do. See its
        ! own comment.
        procedure :: compute_one_electron_picture_change_integrals
        ! Builds the spin-free X2C one-electron Hamiltonian from primitive-
        ! basis S/T/V/pVp (compute_one_electron_integrals(..., primitive=
        ! .true.) for S/T/V, compute_one_electron_picture_change_integrals
        ! for pVp) and caches the resulting picture-change transform.
        ! Errors on a second call without an intervening free_picture_change.
        ! With symmetry set, S/T/V/pVp must be symmetric (an unaveraged
        ! environment in V is refused) and the outputs come back averaged.
        procedure :: construct_x2c_one_electron_hamiltonian
        procedure :: has_picture_change
        procedure :: get_picture_change_transforms
        procedure :: set_picture_change_transforms
        procedure :: set_x2c_frozen_density
        procedure :: free_x2c_frozen_density
        procedure :: compute_x2c_frozen_kinetic_gradient
        procedure :: compute_x2c_frozen_potential_gradient
        procedure :: free_picture_change
        ! The X2C picture with the spin part: the spin planes folded (u_small^T X u_small, antisymmetric, no
        ! symmetrisation), and an operator's scalar and spin parts together; the AMFI planes (raw, primitive basis).
        procedure, private :: transform_one_electron_operator_matrix, transform_one_electron_operator_planes
        ! The scalar part of a one-electron operator in the X2C picture, from its raw plain and p.X p planes.
        generic :: transform_one_electron_operator => transform_one_electron_operator_matrix, &
            transform_one_electron_operator_planes
        procedure :: transform_one_electron_spin_operator
        procedure :: compute_picture_changed_one_electron_operator
        procedure :: compute_amfi_integrals
        ! Point-group symmetry. With symmetry set, the
        ! library cleans the integrals it computes (strictly for the QM
        ! region, averaged for the environment); the caller cleans what it
        ! assembles (the Fock matrix: symmetrise_matrix) and its orbitals.
        ! Irrep indices are 0-based, as the library's own.
        procedure :: set_symmetry
        procedure :: free_symmetry
        procedure :: get_symmetry_options
        procedure :: get_symmetrisation_report
        procedure :: get_point_group_name
        procedure :: get_n_irreps
        procedure :: get_irrep_name
        procedure :: get_irrep_dimension
        procedure :: get_irrep_index
        procedure :: get_totally_symmetric_irrep
        procedure :: get_cartesian_irrep
        procedure :: get_symmetry_standard_axes
        procedure :: get_symmetry_operation_count
        procedure :: get_symmetry_operation_matrix
        procedure :: get_n_symmetry_centres
        procedure :: get_symmetry_centre_images
        procedure :: symmetrise_matrix
        procedure :: analyse_matrix
        procedure :: irrep_norms
        procedure :: symmetrise_orbitals
        ! The group's action: AO functions per irrep, an operation on MO coefficients, characters, irrep matrices.
        procedure :: get_ao_irrep_counts
        procedure :: apply_symmetry_operation
        procedure :: get_irrep_character
        procedure :: get_irrep_matrix
        ! Contracts a primitive-basis result (e.g. from
        ! compute_one_electron_integrals(..., primitive=.true.) or
        ! compute_one_electron_picture_change_integrals) into the contracted
        ! basis -- generic on rank exactly like compute_one_electron_integrals
        ! above (no spin_orbit-like extra axis here, so the same trick
        ! applies cleanly).
        procedure, private :: contract_one_electron_integrals_matrix
        procedure, private :: contract_one_electron_integrals_tensor
        generic             :: contract_one_electron_integrals => &
            contract_one_electron_integrals_matrix, contract_one_electron_integrals_tensor
        final     :: free_aedelsten_context
    end type AedelstenContext

    ! ── C bindings (private — user goes through the type above) ─────────────
    interface
        function set_aedelsten_context_c(filename, contraction_type, comm) &
                bind(C, name="set_aedelsten_context_c") result(handle)
            import :: c_ptr, aedelsten_int_size, c_char
            character(kind=c_char)          , intent(in)        :: filename(*)
            integer(kind=aedelsten_int_size), intent(in), value :: contraction_type
            integer(kind=aedelsten_int_size), intent(in), value :: comm
            type(c_ptr)                                         :: handle
        end function set_aedelsten_context_c

        function aedelsten_query_i32_c(char_query) &
                bind(C, name="aedelsten_query_i32_c") result(i32_answer)
            import :: c_int, c_char
            character(kind=c_char), intent(in)  :: char_query(*)
            integer(c_int) :: i32_answer
        end function aedelsten_query_i32_c

        function aedelsten_query_f64_c(char_query) &
                bind(C, name="aedelsten_query_f64_c") result(f64_answer)
            import :: c_char, aedelsten_float_size
            character(kind=c_char), intent(in) :: char_query(*)
            real(kind=aedelsten_float_size)    :: f64_answer
        end function aedelsten_query_f64_c

        function aedelsten_mpi_init_c(required_level) bind(C, name="aedelsten_mpi_init_c") result(provided)
            import :: c_int
            integer(c_int), intent(in), value :: required_level
            integer(c_int) :: provided
        end function aedelsten_mpi_init_c

        function aedelsten_mpi_finalize_c() bind(C, name="aedelsten_mpi_finalize_c") result(rc)
            import :: c_int
            integer(c_int) :: rc
        end function aedelsten_mpi_finalize_c

        function aedelsten_mpi_initialized_c() bind(C, name="aedelsten_mpi_initialized_c") result(flag)
            import :: c_int
            integer(c_int) :: flag
        end function aedelsten_mpi_initialized_c

        function aedelsten_mpi_thread_level_c() bind(C, name="aedelsten_mpi_thread_level_c") result(level)
            import :: c_int
            integer(c_int) :: level
        end function aedelsten_mpi_thread_level_c

        function aedelsten_mpi_rank_c(comm, rank) bind(C, name="aedelsten_mpi_rank_c") result(rc)
            import :: c_int
            integer(c_int), intent(in), value :: comm
            integer(c_int), intent(out) :: rank
            integer(c_int) :: rc
        end function aedelsten_mpi_rank_c

        function aedelsten_mpi_shared_array_c(comm, length, data) bind(C, name="aedelsten_mpi_shared_array_c") result(handle)
            import :: c_int, c_int64_t, c_ptr
            integer(c_int), intent(in), value :: comm
            integer(c_int64_t), intent(in), value :: length
            type(c_ptr), intent(out) :: data
            type(c_ptr) :: handle
        end function aedelsten_mpi_shared_array_c

        function aedelsten_mpi_shared_array_sync_c(handle) bind(C, name="aedelsten_mpi_shared_array_sync_c") result(rc)
            import :: c_int, c_ptr
            type(c_ptr), intent(in), value :: handle
            integer(c_int) :: rc
        end function aedelsten_mpi_shared_array_sync_c

        subroutine aedelsten_mpi_shared_array_free_c(handle) bind(C, name="aedelsten_mpi_shared_array_free_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: handle
        end subroutine aedelsten_mpi_shared_array_free_c

        function compute_eri_half_transformed_on_node_c(ctx, x_out, c_left, m_left, c_right, m_right, n, threshold, &
                pairwise_threshold) bind(C, name="compute_eri_half_transformed_on_node_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: x_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: c_left(*)
            integer(kind=aedelsten_int_size), intent(in), value :: m_left
            real(kind=aedelsten_float_size), intent(in)         :: c_right(*)
            integer(kind=aedelsten_int_size), intent(in), value :: m_right
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: threshold
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_eri_half_transformed_on_node_c

        function compute_eri_half_transformed_sigma_c(ctx, x_out, c_left, m_left, c_right, m_right, n, sigma_start, &
                sigma_end, threshold, pairwise_threshold, partition_over_ranks) &
                bind(C, name="compute_eri_half_transformed_sigma_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: x_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: c_left(*)
            integer(kind=aedelsten_int_size), intent(in), value :: m_left
            real(kind=aedelsten_float_size), intent(in)         :: c_right(*)
            integer(kind=aedelsten_int_size), intent(in), value :: m_right
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: sigma_start
            integer(kind=aedelsten_int_size), intent(in), value :: sigma_end
            real(kind=aedelsten_float_size), intent(in), value  :: threshold
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size), intent(in), value :: partition_over_ranks
            integer(kind=aedelsten_int_size) :: rc
        end function compute_eri_half_transformed_sigma_c

        function compute_eri_half_transformed_sigma_on_node_c(ctx, x_out, c_left, m_left, c_right, m_right, n, &
                sigma_start, sigma_end, threshold, pairwise_threshold) &
                bind(C, name="compute_eri_half_transformed_sigma_on_node_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: x_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: c_left(*)
            integer(kind=aedelsten_int_size), intent(in), value :: m_left
            real(kind=aedelsten_float_size), intent(in)         :: c_right(*)
            integer(kind=aedelsten_int_size), intent(in), value :: m_right
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: sigma_start
            integer(kind=aedelsten_int_size), intent(in), value :: sigma_end
            real(kind=aedelsten_float_size), intent(in), value  :: threshold
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_eri_half_transformed_sigma_on_node_c

        function half_transform_scratch_bytes_c(ctx, m_left, m_right, one_basis, n_sigma, node_ranks) &
                bind(C, name="half_transform_scratch_bytes_c") result(bytes)
            import :: c_ptr, c_int64_t, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: m_left
            integer(kind=aedelsten_int_size), intent(in), value :: m_right
            integer(kind=aedelsten_int_size), intent(in), value :: one_basis
            integer(kind=aedelsten_int_size), intent(in), value :: n_sigma
            integer(kind=aedelsten_int_size), intent(in), value :: node_ranks
            integer(kind=c_int64_t) :: bytes
        end function half_transform_scratch_bytes_c

        function scratch_bytes_c(ctx, operations, n_operations, primitive) bind(C, name="scratch_bytes_c") result(bytes)
            import :: c_ptr, c_int64_t, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in) :: operations(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_operations
            integer(kind=aedelsten_int_size), intent(in), value :: primitive
            integer(kind=c_int64_t) :: bytes
        end function scratch_bytes_c

        function scratch_breakdown_field_count_c() bind(C, name="scratch_breakdown_field_count_c") result(n)
            import :: aedelsten_int_size
            integer(kind=aedelsten_int_size) :: n
        end function scratch_breakdown_field_count_c

        function scratch_breakdown_c(ctx, operations, n_operations, primitive, output, length) &
                bind(C, name="scratch_breakdown_c") result(rc)
            import :: c_ptr, c_int64_t, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in) :: operations(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_operations
            integer(kind=aedelsten_int_size), intent(in), value :: primitive
            integer(kind=c_int64_t), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function scratch_breakdown_c

        function aedelsten_mpi_size_c(comm, size) bind(C, name="aedelsten_mpi_size_c") result(rc)
            import :: c_int
            integer(c_int), intent(in), value :: comm
            integer(c_int), intent(out) :: size
            integer(c_int) :: rc
        end function aedelsten_mpi_size_c

        function aedelsten_blas_set_threads_c(k) bind(C, name="aedelsten_blas_set_threads_c") result(previous)
            import :: c_int
            integer(c_int), intent(in), value :: k
            integer(c_int) :: previous
        end function aedelsten_blas_set_threads_c

        function aedelsten_mpi_collective_seconds_c() bind(C, name="aedelsten_mpi_collective_seconds_c") result(s)
            import :: c_double
            real(c_double) :: s
        end function aedelsten_mpi_collective_seconds_c

        function aedelsten_profile_enabled_c() bind(C, name="aedelsten_profile_enabled_c") result(flag)
            import :: c_int
            integer(c_int) :: flag
        end function aedelsten_profile_enabled_c

        function aedelsten_timing_count_c(fine) bind(C, name="aedelsten_timing_count_c") result(n)
            import :: c_int
            integer(c_int), intent(in), value :: fine
            integer(c_int) :: n
        end function aedelsten_timing_count_c

        function aedelsten_timing_span_c(fine, index, label, label_len, seconds, calls) &
                bind(C, name="aedelsten_timing_span_c") result(rc)
            import :: c_int, c_char, c_double, c_int64_t
            integer(c_int), intent(in), value :: fine, index, label_len
            character(kind=c_char), intent(out) :: label(*)
            real(c_double), intent(out) :: seconds
            integer(c_int64_t), intent(out) :: calls
            integer(c_int) :: rc
        end function aedelsten_timing_span_c

        function aedelsten_span_tree_count_c() bind(C, name="aedelsten_span_tree_count_c") result(n)
            import :: c_int
            integer(c_int) :: n
        end function aedelsten_span_tree_count_c

        function aedelsten_span_tree_node_c(index, label, label_len, depth, fine, seconds, calls) &
                bind(C, name="aedelsten_span_tree_node_c") result(rc)
            import :: c_int, c_char, c_double, c_int64_t
            integer(c_int), intent(in), value :: index, label_len
            character(kind=c_char), intent(out) :: label(*)
            integer(c_int), intent(out) :: depth, fine
            real(c_double), intent(out) :: seconds
            integer(c_int64_t), intent(out) :: calls
            integer(c_int) :: rc
        end function aedelsten_span_tree_node_c

        function aedelsten_citation_count_c() bind(C, name="aedelsten_citation_count_c") result(n)
            import :: c_int
            integer(c_int) :: n
        end function aedelsten_citation_count_c

        function aedelsten_citation_length_c(index, topics) bind(C, name="aedelsten_citation_length_c") result(n)
            import :: c_int, c_int64_t
            integer(c_int), intent(in), value :: index, topics
            integer(c_int64_t) :: n
        end function aedelsten_citation_length_c

        function aedelsten_citation_c(index, topics, buf, buf_len) bind(C, name="aedelsten_citation_c") result(rc)
            import :: c_int, c_char
            integer(c_int), intent(in), value :: index, topics, buf_len
            character(kind=c_char), intent(out) :: buf(*)
            integer(c_int) :: rc
        end function aedelsten_citation_c

        function aedelsten_citations_bibtex_length_c() bind(C, name="aedelsten_citations_bibtex_length_c") result(n)
            import :: c_int64_t
            integer(c_int64_t) :: n
        end function aedelsten_citations_bibtex_length_c

        function aedelsten_citations_bibtex_c(buf, buf_len) bind(C, name="aedelsten_citations_bibtex_c") result(rc)
            import :: c_int, c_char
            character(kind=c_char), intent(out) :: buf(*)
            integer(c_int), intent(in), value :: buf_len
            integer(c_int) :: rc
        end function aedelsten_citations_bibtex_c

        subroutine aedelsten_clear_citations_c() bind(C, name="aedelsten_clear_citations_c")
        end subroutine aedelsten_clear_citations_c

        function aedelsten_blas_threads_c() bind(C, name="aedelsten_blas_threads_c") result(k)
            import :: c_int
            integer(c_int) :: k
        end function aedelsten_blas_threads_c

        function aedelsten_get_version_c(buf, buf_len) bind(C, name="aedelsten_get_version_c") result(rc)
            import :: c_int, c_char
            character(kind=c_char), intent(out) :: buf(*)
            integer(c_int), intent(in), value :: buf_len
            integer(c_int) :: rc
        end function aedelsten_get_version_c

        function aedelsten_get_source_commit_c(buf, buf_len) bind(C, name="aedelsten_get_source_commit_c") result(rc)
            import :: c_int, c_char
            character(kind=c_char), intent(out) :: buf(*)
            integer(c_int), intent(in), value :: buf_len
            integer(c_int) :: rc
        end function aedelsten_get_source_commit_c

        function aedelsten_blas_info_c(buf, buf_len) bind(C, name="aedelsten_blas_info_c") result(rc)
            import :: c_int, c_char
            character(kind=c_char), intent(out) :: buf(*)
            integer(c_int), intent(in), value :: buf_len
            integer(c_int) :: rc
        end function aedelsten_blas_info_c

        function aedelsten_blas_available_cores_c(count, bound) bind(C, name="aedelsten_blas_available_cores_c") result(rc)
            import :: c_int
            integer(c_int), intent(out) :: count, bound
            integer(c_int) :: rc
        end function aedelsten_blas_available_cores_c

        function aedelsten_blas_available_cores_info_c(buf, buf_len) &
                bind(C, name="aedelsten_blas_available_cores_info_c") result(rc)
            import :: c_int, c_char
            character(kind=c_char), intent(out) :: buf(*)
            integer(c_int), intent(in), value :: buf_len
            integer(c_int) :: rc
        end function aedelsten_blas_available_cores_info_c

        subroutine free_aedelsten_context_c(ctx) bind(C, name="free_aedelsten_context_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_aedelsten_context_c

        ! Shell pairs.
        function set_shell_pairs_c(ctx, overlap_threshold) &
                bind(C, name="set_shell_pairs_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in), value :: overlap_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function set_shell_pairs_c

        subroutine free_shell_pairs_c(ctx) bind(C, name="free_shell_pairs_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_shell_pairs_c

        ! Scratch.
        function resize_scratch_c(ctx, size, unit) bind(C, name="resize_scratch_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: size
            integer(kind=aedelsten_int_size), intent(in), value :: unit
            integer(kind=aedelsten_int_size) :: rc
        end function resize_scratch_c

        function get_scratch_peak_bytes_c(ctx) bind(C, name="get_scratch_peak_bytes_c") result(bytes)
            import :: c_ptr, c_int64_t
            type(c_ptr), intent(in), value :: ctx
            integer(kind=c_int64_t) :: bytes
        end function get_scratch_peak_bytes_c

        function set_communicator_c(ctx, comm) &
                bind(C, name="set_communicator_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: comm
            integer(kind=aedelsten_int_size) :: rc
        end function set_communicator_c

        function set_scratch_c(ctx, size, unit) &
                bind(C, name="set_scratch_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: size
            integer(kind=aedelsten_int_size), intent(in), value :: unit
            integer(kind=aedelsten_int_size) :: rc
        end function set_scratch_c

        subroutine free_scratch_c(ctx) bind(C, name="free_scratch_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_scratch_c

        ! Basis-size getters.
        function get_n_cont_c(ctx) bind(C, name="get_n_cont_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_cont_c

        function get_n_prim_c(ctx) bind(C, name="get_n_prim_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_prim_c

        function get_n_prim_unique_c(ctx) bind(C, name="get_n_prim_unique_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_prim_unique_c

        ! Molecular properties and the finite nuclear charge model.
        function set_finite_nuclear_model_c(ctx) bind(C, name="set_finite_nuclear_model_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: rc
        end function set_finite_nuclear_model_c

        subroutine compute_centre_of_mass_c(ctx, x, y, z) bind(C, name="compute_centre_of_mass_c")
            import :: c_ptr, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(out) :: x, y, z
        end subroutine compute_centre_of_mass_c

        function compute_nuclear_repulsion_energy_c(ctx) &
                bind(C, name="compute_nuclear_repulsion_energy_c") result(e)
            import :: c_ptr, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size) :: e
        end function compute_nuclear_repulsion_energy_c

        function compute_nuclear_charge_c(ctx) bind(C, name="compute_nuclear_charge_c") result(q)
            import :: c_ptr, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size) :: q
        end function compute_nuclear_charge_c

        function compute_nuclear_multipole_interaction_energy_c(ctx) &
                bind(C, name="compute_nuclear_multipole_interaction_energy_c") result(e)
            import :: c_ptr, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size) :: e
        end function compute_nuclear_multipole_interaction_energy_c

        function compute_nuclear_aimp_embedding_energy_c(ctx) &
                bind(C, name="compute_nuclear_aimp_embedding_energy_c") result(e)
            import :: c_ptr, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size) :: e
        end function compute_nuclear_aimp_embedding_energy_c

        ! Primitive (decontracted) shell pairs.
        function set_primitive_shell_pairs_c(ctx, overlap_threshold) &
                bind(C, name="set_primitive_shell_pairs_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in), value :: overlap_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function set_primitive_shell_pairs_c

        subroutine free_primitive_shell_pairs_c(ctx) bind(C, name="free_primitive_shell_pairs_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_primitive_shell_pairs_c

        subroutine free_input_c(ctx) bind(C, name="free_input_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_input_c

        subroutine free_primitive_shells_c(ctx) bind(C, name="free_primitive_shells_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_primitive_shells_c

        subroutine free_contraction_matrix_c(ctx) bind(C, name="free_contraction_matrix_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_contraction_matrix_c

        function has_primitive_c(ctx) bind(C, name="has_primitive_c") result(flag)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: flag
        end function has_primitive_c

        ! External multipole field.
        function set_external_multipoles_c(ctx, filename) &
                bind(C, name="set_external_multipoles_c") result(rc)
            import :: c_ptr, aedelsten_int_size, c_char
            type(c_ptr), intent(in), value :: ctx
            character(kind=c_char), intent(in) :: filename(*)
            integer(kind=aedelsten_int_size) :: rc
        end function set_external_multipoles_c

        subroutine free_external_multipoles_c(ctx) bind(C, name="free_external_multipoles_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_external_multipoles_c

        function has_external_multipoles_c(ctx) bind(C, name="has_external_multipoles_c") result(flag)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: flag
        end function has_external_multipoles_c

        ! Induced-dipole embedding.
        function set_polarizable_sites_c(ctx, filename) bind(C, name="set_polarizable_sites_c") result(rc)
            import :: c_ptr, aedelsten_int_size, c_char
            type(c_ptr), intent(in), value :: ctx
            character(kind=c_char), intent(in) :: filename(*)
            integer(kind=aedelsten_int_size) :: rc
        end function set_polarizable_sites_c

        subroutine free_polarizable_sites_c(ctx) bind(C, name="free_polarizable_sites_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_polarizable_sites_c

        function get_n_polarizable_sites_c(ctx) bind(C, name="get_n_polarizable_sites_c") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), intent(in), value :: ctx
            integer(kind=c_int64_t) :: n
        end function get_n_polarizable_sites_c

        function has_polarizable_sites_c(ctx) bind(C, name="has_polarizable_sites_c") result(flag)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: flag
        end function has_polarizable_sites_c

        function has_polarization_c(ctx) bind(C, name="has_polarization_c") result(flag)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: flag
        end function has_polarization_c

        function count_electric_field_sites_c(ctx, selection, points, n_points) &
                bind(C, name="count_electric_field_sites_c") result(n)
            import :: c_ptr, c_int64_t, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx, points
            integer(kind=aedelsten_int_size), intent(in), value :: selection
            integer(kind=c_int64_t), intent(in), value :: n_points
            integer(kind=c_int64_t) :: n
        end function count_electric_field_sites_c

        function get_input_indices_c(ctx, selection, points, n_points, indices, n) &
                bind(C, name="get_input_indices_c") result(rc)
            import :: c_ptr, c_int64_t, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx, points
            integer(kind=aedelsten_int_size), intent(in), value :: selection
            integer(kind=c_int64_t), intent(in), value :: n_points, n
            integer(kind=c_int64_t), intent(inout) :: indices(*)
            integer(kind=aedelsten_int_size) :: rc
        end function get_input_indices_c

        function compute_electric_field_integrals_c(ctx, selection, points, n_points, first_site, n_sites, output, length) &
                bind(C, name="compute_electric_field_integrals_c") result(rc)
            import :: c_ptr, c_int64_t, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx, points
            integer(kind=aedelsten_int_size), intent(in), value :: selection
            integer(kind=c_int64_t), intent(in), value :: n_points, first_site, n_sites, length
            real(kind=aedelsten_float_size), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size) :: rc
        end function compute_electric_field_integrals_c

        function compute_static_field_c(ctx, selection, points, n_points, nuclei, core_aimps, embedding_aimps, &
                external_multipoles, field, length) bind(C, name="compute_static_field_c") result(rc)
            import :: c_ptr, c_int64_t, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx, points
            integer(kind=aedelsten_int_size), intent(in), value :: selection, nuclei, core_aimps, embedding_aimps, &
                external_multipoles
            integer(kind=c_int64_t), intent(in), value :: n_points, length
            real(kind=aedelsten_float_size), intent(inout) :: field(*)
            integer(kind=aedelsten_int_size) :: rc
        end function compute_static_field_c

        function compute_electronic_field_c(ctx, density, selection, points, n_points, chunk_sites, chunk_budget, &
                chunk_budget_unit, field, length) bind(C, name="compute_electronic_field_c") result(rc)
            import :: c_ptr, c_int64_t, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx, points
            real(kind=aedelsten_float_size), intent(in) :: density(*)
            integer(kind=aedelsten_int_size), intent(in), value :: selection, chunk_budget_unit
            integer(kind=c_int64_t), intent(in), value :: n_points, chunk_sites, chunk_budget, length
            real(kind=aedelsten_float_size), intent(inout) :: field(*)
            integer(kind=aedelsten_int_size) :: rc
        end function compute_electronic_field_c

        function set_multipole_convention_c(ctx, convention) bind(C, name="set_multipole_convention_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: convention
            integer(kind=aedelsten_int_size) :: rc
        end function set_multipole_convention_c

        function get_multipole_convention_c(ctx) bind(C, name="get_multipole_convention_c") result(convention)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: convention
        end function get_multipole_convention_c

        function set_polarization_c(ctx, options) bind(C, name="set_polarization_c") result(rc)
            import :: c_ptr, aedelsten_int_size, AedelstenPolarizationOptions
            type(c_ptr), intent(in), value :: ctx
            type(AedelstenPolarizationOptions), intent(in) :: options
            integer(kind=aedelsten_int_size) :: rc
        end function set_polarization_c

        subroutine free_polarization_c(ctx) bind(C, name="free_polarization_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_polarization_c

        function compute_induction_gradient_c(ctx, gradient, n_centres, density, active) &
                bind(C, name="compute_induction_gradient_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: gradient(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            type(c_ptr), intent(in), value :: active
            integer(kind=aedelsten_int_size) :: rc
        end function compute_induction_gradient_c

        function compute_induction_c(ctx, density, v_ind, energy, iterations) &
                bind(C, name="compute_induction_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)    :: density(*)
            real(kind=aedelsten_float_size), intent(inout) :: v_ind(*)
            real(kind=aedelsten_float_size), intent(out)   :: energy
            integer(kind=aedelsten_int_size), intent(out)  :: iterations
            integer(kind=aedelsten_int_size) :: rc
        end function compute_induction_c

        function compute_induction_within_c(ctx, density, v_ind, threshold, energy, iterations) &
                bind(C, name="compute_induction_within_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)    :: density(*)
            real(kind=aedelsten_float_size), intent(inout) :: v_ind(*)
            real(kind=aedelsten_float_size), intent(in), value :: threshold
            real(kind=aedelsten_float_size), intent(out)   :: energy
            integer(kind=aedelsten_int_size), intent(out)  :: iterations
            integer(kind=aedelsten_int_size) :: rc
        end function compute_induction_within_c

        function get_polarization_storage_c(ctx, storage) bind(C, name="get_polarization_storage_c") result(rc)
            import :: c_ptr, c_int64_t, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=c_int64_t), intent(out) :: storage(4)
            integer(kind=aedelsten_int_size) :: rc
        end function get_polarization_storage_c

        function set_resident_field_budget_c(ctx, budget, budget_unit) bind(C, name="set_resident_field_budget_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: budget, budget_unit
            integer(kind=aedelsten_int_size) :: rc
        end function set_resident_field_budget_c


        function compute_induction_response_c(ctx, densities, n_mat, operators) &
                bind(C, name="compute_induction_response_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)    :: densities(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_mat
            real(kind=aedelsten_float_size), intent(inout) :: operators(*)
            integer(kind=aedelsten_int_size) :: rc
        end function compute_induction_response_c

        function compute_induction_couplings_c(ctx, densities, n_mat, couplings) &
                bind(C, name="compute_induction_couplings_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)    :: densities(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_mat
            real(kind=aedelsten_float_size), intent(inout) :: couplings(*)
            integer(kind=aedelsten_int_size) :: rc
        end function compute_induction_couplings_c

        function compute_effective_external_field_c(ctx, operators) &
                bind(C, name="compute_effective_external_field_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: operators(*)
            integer(kind=aedelsten_int_size) :: rc
        end function compute_effective_external_field_c

        function compute_induction_states_c(ctx, densities, n_states, operators, energies) &
                bind(C, name="compute_induction_states_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)    :: densities(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_states
            real(kind=aedelsten_float_size), intent(inout) :: operators(*)
            real(kind=aedelsten_float_size), intent(inout) :: energies(*)
            integer(kind=aedelsten_int_size) :: rc
        end function compute_induction_states_c

        function compute_induction_states_within_c(ctx, densities, n_states, operators, energies, threshold) &
                bind(C, name="compute_induction_states_within_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)    :: densities(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_states
            real(kind=aedelsten_float_size), intent(inout) :: operators(*)
            real(kind=aedelsten_float_size), intent(inout) :: energies(*)
            real(kind=aedelsten_float_size), intent(in), value :: threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_induction_states_within_c

        function get_induced_dipoles_c(ctx, dipoles, length) bind(C, name="get_induced_dipoles_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size, c_int64_t
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: dipoles(*)
            integer(kind=c_int64_t), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function get_induced_dipoles_c

        ! AIMP.
        function has_core_aimp_c(ctx) bind(C, name="has_core_aimp_c") result(flag)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: flag
        end function has_core_aimp_c

        function set_induced_dipoles_c(ctx, dipoles, length) bind(C, name="set_induced_dipoles_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size, c_int64_t
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in) :: dipoles(*)
            integer(kind=c_int64_t), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function set_induced_dipoles_c

        function get_n_embedding_aimps_c(ctx) bind(C, name="get_n_embedding_aimps_c") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), intent(in), value :: ctx
            integer(kind=c_int64_t) :: n
        end function get_n_embedding_aimps_c

        function get_embedding_aimp_sites_c(ctx, atomic_numbers, positions, charges, n) &
                bind(C, name="get_embedding_aimp_sites_c") result(rc)
            import :: c_ptr, c_int32_t, c_int64_t, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=c_int32_t), intent(inout) :: atomic_numbers(*)
            real(kind=aedelsten_float_size), intent(inout) :: positions(*), charges(*)
            integer(kind=c_int64_t), intent(in), value :: n
            integer(kind=aedelsten_int_size) :: rc
        end function get_embedding_aimp_sites_c

        function has_embedding_aimp_c(ctx) bind(C, name="has_embedding_aimp_c") result(flag)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: flag
        end function has_embedding_aimp_c

        ! filename is a raw c_ptr so that c_null_ptr can be passed: the embedding AIMPs of the input (a non-null
        ! filename is refused).
        function set_embedding_aimp_operators_c(ctx, filename) &
                bind(C, name="set_embedding_aimp_operators_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            type(c_ptr), intent(in), value :: filename
            integer(kind=aedelsten_int_size) :: rc
        end function set_embedding_aimp_operators_c

        subroutine free_embedding_aimp_operators_c(ctx) bind(C, name="free_embedding_aimp_operators_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_embedding_aimp_operators_c

        ! Integral routines.
        function compute_one_electron_integrals_c(&
            ctx, &
            output, &
            n, &
            operator, &
            n_planes, &
            origin_x, &
            origin_y, &
            origin_z, &
            order, &
            primitive &
            ) bind(C, name="compute_one_electron_integrals_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size) , intent(inout)     :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: operator
            integer(kind=aedelsten_int_size), intent(in), value :: n_planes
            real(kind=aedelsten_float_size) , intent(in), value :: origin_x
            real(kind=aedelsten_float_size) , intent(in), value :: origin_y
            real(kind=aedelsten_float_size) , intent(in), value :: origin_z
            integer(kind=aedelsten_int_size), intent(in), value :: order
            integer(kind=aedelsten_int_size), intent(in), value :: primitive
            integer(kind=aedelsten_int_size) :: rc
        end function compute_one_electron_integrals_c

        function get_n_gradient_centres_c(ctx) bind(C, name="get_n_gradient_centres_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_gradient_centres_c

        function compute_nuclear_repulsion_gradient_c(ctx, gradient, n_centres, active) &
                bind(C, name="compute_nuclear_repulsion_gradient_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: gradient(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            type(c_ptr), intent(in), value :: active
            integer(kind=aedelsten_int_size) :: rc
        end function compute_nuclear_repulsion_gradient_c

        function compute_nuclear_aimp_embedding_gradient_c(ctx, gradient, n_centres, active) &
                bind(C, name="compute_nuclear_aimp_embedding_gradient_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: gradient(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            type(c_ptr), intent(in), value :: active
            integer(kind=aedelsten_int_size) :: rc
        end function compute_nuclear_aimp_embedding_gradient_c

        function compute_nuclear_multipole_interaction_gradient_c(ctx, gradient, n_centres, active) &
                bind(C, name="compute_nuclear_multipole_interaction_gradient_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: gradient(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            type(c_ptr), intent(in), value :: active
            integer(kind=aedelsten_int_size) :: rc
        end function compute_nuclear_multipole_interaction_gradient_c

        function set_centre_positions_c(ctx, positions, n_centres) bind(C, name="set_centre_positions_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)         :: positions(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            integer(kind=aedelsten_int_size) :: rc
        end function set_centre_positions_c

        function compute_one_electron_gradient_c(ctx, gradient, n_centres, density, n, operator, active) &
                bind(C, name="compute_one_electron_gradient_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size) , intent(inout)     :: gradient(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            real(kind=aedelsten_float_size) , intent(in)        :: density(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: operator
            type(c_ptr), intent(in), value :: active
            integer(kind=aedelsten_int_size) :: rc
        end function compute_one_electron_gradient_c

        function get_n_shells_c(ctx) bind(C, name="get_n_shells_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_shells_c

        ! output: all centres (0-based), all l (negated for Cartesian), all first AOs (0-based), all AO counts.
        function get_shells_c(ctx, output, length) bind(C, name="get_shells_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function get_shells_c

        function compute_shell_extents_c(ctx, threshold, output, n_shells) bind(C, name="compute_shell_extents_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in), value  :: threshold
            real(kind=aedelsten_float_size), intent(inout)      :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_shells
            integer(kind=aedelsten_int_size) :: rc
        end function compute_shell_extents_c

        ! shells 0-based; points (point_rows, n_points); values (n_ao_listed, n_points); gradients c_null_ptr or
        ! three such blocks (x, y, z).
        function compute_ao_values_c(ctx, shells, n_shells, points, point_rows, n_points, values, gradients, n_ao_listed) &
                bind(C, name="compute_ao_values_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in)        :: shells(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_shells
            real(kind=aedelsten_float_size), intent(in)         :: points(*)
            integer(kind=aedelsten_int_size), intent(in), value :: point_rows
            integer(kind=aedelsten_int_size), intent(in), value :: n_points
            real(kind=aedelsten_float_size), intent(inout)      :: values(*)
            type(c_ptr), intent(in), value :: gradients
            integer(kind=aedelsten_int_size), intent(in), value :: n_ao_listed
            integer(kind=aedelsten_int_size) :: rc
        end function compute_ao_values_c

        ! As compute_ao_values_c, gradients three blocks (x, y, z) and hessians six (xx, xy, xz, yy, yz, zz), neither
        ! null.
        function compute_ao_hessians_c(ctx, shells, n_shells, points, point_rows, n_points, values, gradients, hessians, &
                n_ao_listed) bind(C, name="compute_ao_hessians_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in)        :: shells(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_shells
            real(kind=aedelsten_float_size), intent(in)         :: points(*)
            integer(kind=aedelsten_int_size), intent(in), value :: point_rows
            integer(kind=aedelsten_int_size), intent(in), value :: n_points
            real(kind=aedelsten_float_size), intent(inout)      :: values(*)
            real(kind=aedelsten_float_size), intent(inout)      :: gradients(*)
            real(kind=aedelsten_float_size), intent(inout)      :: hessians(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_ao_listed
            integer(kind=aedelsten_int_size) :: rc
        end function compute_ao_hessians_c

        ! Picture-change counterpart of compute_one_electron_integrals_c --
        ! same shape, plus spin_orbit; always decontracted (no `primitive`
        ! parameter here, see the type-bound procedure's own comment).
        function compute_one_electron_picture_change_integrals_c(&
            ctx, &
            output, &
            n, &
            operator, &
            n_planes, &
            origin_x, &
            origin_y, &
            origin_z, &
            order, &
            spin_orbit &
            ) bind(C, name="compute_one_electron_picture_change_integrals_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size) , intent(inout)     :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: operator
            integer(kind=aedelsten_int_size), intent(in), value :: n_planes
            real(kind=aedelsten_float_size) , intent(in), value :: origin_x
            real(kind=aedelsten_float_size) , intent(in), value :: origin_y
            real(kind=aedelsten_float_size) , intent(in), value :: origin_z
            integer(kind=aedelsten_int_size), intent(in), value :: order
            integer(kind=aedelsten_int_size), intent(in), value :: spin_orbit
            integer(kind=aedelsten_int_size) :: rc
        end function compute_one_electron_picture_change_integrals_c

        ! c_light <= 0.0 means "use the library default".
        function construct_x2c_one_electron_hamiltonian_c(ctx, oneham, s, t, v, pvp, n, c_light) &
                bind(C, name="construct_x2c_one_electron_hamiltonian_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: oneham(*)
            real(kind=aedelsten_float_size), intent(in)         :: s(*)
            real(kind=aedelsten_float_size), intent(in)         :: t(*)
            real(kind=aedelsten_float_size), intent(in)         :: v(*)
            real(kind=aedelsten_float_size), intent(in)         :: pvp(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: c_light
            integer(kind=aedelsten_int_size) :: rc
        end function construct_x2c_one_electron_hamiltonian_c

        ! Contract a primitive-basis result into the contracted basis.
        function contract_one_electron_matrix_c(ctx, primitive, n_prim, output, n_cont) &
                bind(C, name="contract_one_electron_matrix_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)          :: primitive(*)
            integer(kind=aedelsten_int_size), intent(in), value  :: n_prim
            real(kind=aedelsten_float_size), intent(inout)       :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value  :: n_cont
            integer(kind=aedelsten_int_size) :: rc
        end function contract_one_electron_matrix_c

        function contract_one_electron_array3d_c(ctx, primitive, n_prim, n_planes, output, n_cont) &
                bind(C, name="contract_one_electron_array3d_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)          :: primitive(*)
            integer(kind=aedelsten_int_size), intent(in), value  :: n_prim
            integer(kind=aedelsten_int_size), intent(in), value  :: n_planes
            real(kind=aedelsten_float_size), intent(inout)       :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value  :: n_cont
            integer(kind=aedelsten_int_size) :: rc
        end function contract_one_electron_array3d_c

        ! Number of Cartesian components a moment integral of the
        ! given order produces (order 0 -> 1, order 1 (dipole) -> 3, order 2
        ! (quadrupole) -> 6, ...) -- pure function of order, no context
        ! needed. Used to validate a caller's rank-3 output array against
        ! the order it asked for, before ever reaching Rust.
        function compute_moment_component_count_c(order) &
                bind(C, name="compute_moment_component_count_c") result(n_components)
            import :: aedelsten_int_size
            integer(kind=aedelsten_int_size), intent(in), value :: order
            integer(kind=aedelsten_int_size) :: n_components
        end function compute_moment_component_count_c

        function compute_one_electron_component_count_c(operator, order) &
                bind(C, name="compute_one_electron_component_count_c") result(n_components)
            import :: aedelsten_int_size
            integer(kind=aedelsten_int_size), intent(in), value :: operator
            integer(kind=aedelsten_int_size), intent(in), value :: order
            integer(kind=aedelsten_int_size) :: n_components
        end function compute_one_electron_component_count_c

        ! Two-electron repulsion integrals. ao_index_order is always
        ! aedelsten_int_size 1 (column-major) here -- see the type-bound
        ! procedure's own comment.
        function compute_eri_stored_c(ctx, output, n, ao_symmetry, ao_index_order, range_operator, omega) &
                bind(C, name="compute_eri_stored_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: ao_symmetry
            integer(kind=aedelsten_int_size), intent(in), value :: ao_index_order
            integer(kind=aedelsten_int_size), intent(in), value :: range_operator
            real(kind=aedelsten_float_size), intent(in), value  :: omega
            integer(kind=aedelsten_int_size) :: rc
        end function compute_eri_stored_c

        function compute_s4_size_c(ctx) bind(C, name="compute_s4_size_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function compute_s4_size_c

        function compute_s8_size_c(ctx) bind(C, name="compute_s8_size_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function compute_s8_size_c

        ! Direct Fock construction (no stored ERI tensor).
        function set_eri_data_c(ctx, schwarz_threshold) bind(C, name="set_eri_data_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in), value :: schwarz_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function set_eri_data_c

        subroutine free_eri_data_c(ctx) bind(C, name="free_eri_data_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_eri_data_c

        function compute_fock_direct_restricted_c(&
            ctx, j_out, k_out, density, n, j_coeff, k_coeff, range_operator, omega, pairwise_threshold &
            ) bind(C, name="compute_fock_direct_restricted_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: j_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: j_coeff
            real(kind=aedelsten_float_size), intent(in), value  :: k_coeff
            integer(kind=aedelsten_int_size), intent(in), value :: range_operator
            real(kind=aedelsten_float_size), intent(in), value  :: omega
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_fock_direct_restricted_c

        function compute_fock_direct_general_c(&
            ctx, j_out, k_out, density, n, j_coeff, k_coeff, range_operator, omega, pairwise_threshold &
            ) bind(C, name="compute_fock_direct_general_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: j_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: j_coeff
            real(kind=aedelsten_float_size), intent(in), value  :: k_coeff
            integer(kind=aedelsten_int_size), intent(in), value :: range_operator
            real(kind=aedelsten_float_size), intent(in), value  :: omega
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_fock_direct_general_c

        function compute_fock_direct_general_multi_c(&
            ctx, j_out, k_out, density, n, n_mat, j_coeff, k_coeff, range_operator, omega, pairwise_threshold &
            ) bind(C, name="compute_fock_direct_general_multi_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: j_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: n_mat
            real(kind=aedelsten_float_size), intent(in), value  :: j_coeff
            real(kind=aedelsten_float_size), intent(in), value  :: k_coeff
            integer(kind=aedelsten_int_size), intent(in), value :: range_operator
            real(kind=aedelsten_float_size), intent(in), value  :: omega
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_fock_direct_general_multi_c

        function compute_fock_direct_mixed_active_space_c(&
            ctx, j_out, k_out, m_out, density, c_bra, c_ket, n, n_active, j_coeff, k_coeff, &
            active_threshold, pairwise_threshold &
            ) bind(C, name="compute_fock_direct_mixed_active_space_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: j_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: m_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            real(kind=aedelsten_float_size), intent(in)         :: c_bra(*)
            real(kind=aedelsten_float_size), intent(in)         :: c_ket(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: n_active
            real(kind=aedelsten_float_size), intent(in), value  :: j_coeff
            real(kind=aedelsten_float_size), intent(in), value  :: k_coeff
            real(kind=aedelsten_float_size), intent(in), value  :: active_threshold
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_fock_direct_mixed_active_space_c

        function compute_two_electron_gradient_c(&
            ctx, gradient, n_centres, densities, n, n_densities, n_terms, term_kind, term_left, term_right, &
            term_coeff, range_operator, omega, pairwise_threshold, active &
            ) bind(C, name="compute_two_electron_gradient_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: gradient(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            real(kind=aedelsten_float_size), intent(in)         :: densities(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: n_densities
            integer(kind=aedelsten_int_size), intent(in), value :: n_terms
            integer(kind=aedelsten_int_size), intent(in)        :: term_kind(*)
            integer(kind=aedelsten_int_size), intent(in)        :: term_left(*)
            integer(kind=aedelsten_int_size), intent(in)        :: term_right(*)
            real(kind=aedelsten_float_size), intent(in)         :: term_coeff(*)
            integer(kind=aedelsten_int_size), intent(in), value :: range_operator
            real(kind=aedelsten_float_size), intent(in), value  :: omega
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            type(c_ptr), intent(in), value :: active
            integer(kind=aedelsten_int_size) :: rc
        end function compute_two_electron_gradient_c

        function compute_fock_direct_unrestricted_c(&
            ctx, j_out, k_alpha_out, k_beta_out, density_alpha, density_beta, density_total, n, &
            j_coeff, k_coeff, range_operator, omega, pairwise_threshold &
            ) bind(C, name="compute_fock_direct_unrestricted_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: j_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_alpha_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_beta_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_alpha(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_beta(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_total(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: j_coeff
            real(kind=aedelsten_float_size), intent(in), value  :: k_coeff
            integer(kind=aedelsten_int_size), intent(in), value :: range_operator
            real(kind=aedelsten_float_size), intent(in), value  :: omega
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_fock_direct_unrestricted_c

        function compute_fock_direct_active_space_c(&
            ctx, j_inactive_out, k_inactive_out, j_active_out, k_active_out, m_out, density_inactive, &
            density_active, c_active, n, n_active, coeff_j_inactive, coeff_k_inactive, coeff_j_active, &
            coeff_k_active, active_threshold, pairwise_threshold &
            ) bind(C, name="compute_fock_direct_active_space_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: j_inactive_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_inactive_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: j_active_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_active_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: m_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_inactive(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_active(*)
            real(kind=aedelsten_float_size), intent(in)         :: c_active(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: n_active
            real(kind=aedelsten_float_size), intent(in), value  :: coeff_j_inactive
            real(kind=aedelsten_float_size), intent(in), value  :: coeff_k_inactive
            real(kind=aedelsten_float_size), intent(in), value  :: coeff_j_active
            real(kind=aedelsten_float_size), intent(in), value  :: coeff_k_active
            real(kind=aedelsten_float_size), intent(in), value  :: active_threshold
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size) :: rc
        end function compute_fock_direct_active_space_c

        function transform_pair_coulomb_one_index_c(ctx, t_out, m, c, n, n_pairs, n_columns) &
            bind(C, name="transform_pair_coulomb_one_index_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: t_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: m(*)
            real(kind=aedelsten_float_size), intent(in)         :: c(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size), intent(in), value :: n_pairs
            integer(kind=aedelsten_int_size), intent(in), value :: n_columns
            integer(kind=aedelsten_int_size) :: rc
        end function transform_pair_coulomb_one_index_c

        function compute_eri_half_transformed_c(ctx, x_out, c_left, m_left, c_right, m_right, n, threshold, &
                pairwise_threshold, partition_over_ranks) bind(C, name="compute_eri_half_transformed_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: x_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: c_left(*)
            integer(kind=aedelsten_int_size), intent(in), value :: m_left
            real(kind=aedelsten_float_size), intent(in)         :: c_right(*)
            integer(kind=aedelsten_int_size), intent(in), value :: m_right
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: threshold
            real(kind=aedelsten_float_size), intent(in), value  :: pairwise_threshold
            integer(kind=aedelsten_int_size), intent(in), value :: partition_over_ranks
            integer(kind=aedelsten_int_size) :: rc
        end function compute_eri_half_transformed_c

        ! J/K from an in-memory dense (S1) ERI tensor -- pure functions of
        ! their own arguments, no ctx: see the module-level procedures'
        ! own comments for why these stay free functions.
        function compute_j_and_k_matrix_stored_c(j_out, k_out, density, eri_dense, n, j_coeff, k_coeff) &
                bind(C, name="compute_j_and_k_matrix_stored_c") result(rc)
            import :: aedelsten_int_size, aedelsten_float_size
            real(kind=aedelsten_float_size), intent(inout)      :: j_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            real(kind=aedelsten_float_size), intent(in)         :: eri_dense(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: j_coeff
            real(kind=aedelsten_float_size), intent(in), value  :: k_coeff
            integer(kind=aedelsten_int_size) :: rc
        end function compute_j_and_k_matrix_stored_c

        function compute_j_and_k_unrestricted_matrix_stored_c(&
            j_out, k_alpha_out, k_beta_out, density_alpha, density_beta, density_total, eri_dense, n, j_coeff, k_coeff &
            ) bind(C, name="compute_j_and_k_unrestricted_matrix_stored_c") result(rc)
            import :: aedelsten_int_size, aedelsten_float_size
            real(kind=aedelsten_float_size), intent(inout)      :: j_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_alpha_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_beta_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_alpha(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_beta(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_total(*)
            real(kind=aedelsten_float_size), intent(in)         :: eri_dense(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: j_coeff
            real(kind=aedelsten_float_size), intent(in), value  :: k_coeff
            integer(kind=aedelsten_int_size) :: rc
        end function compute_j_and_k_unrestricted_matrix_stored_c

        function compute_j_matrix_stored_c(j_out, density, eri_dense, n, coeff) &
                bind(C, name="compute_j_matrix_stored_c") result(rc)
            import :: aedelsten_int_size, aedelsten_float_size
            real(kind=aedelsten_float_size), intent(inout)      :: j_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            real(kind=aedelsten_float_size), intent(in)         :: eri_dense(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: coeff
            integer(kind=aedelsten_int_size) :: rc
        end function compute_j_matrix_stored_c

        function compute_k_matrix_stored_c(k_out, density, eri_dense, n, coeff) &
                bind(C, name="compute_k_matrix_stored_c") result(rc)
            import :: aedelsten_int_size, aedelsten_float_size
            real(kind=aedelsten_float_size), intent(inout)      :: k_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            real(kind=aedelsten_float_size), intent(in)         :: eri_dense(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: coeff
            integer(kind=aedelsten_int_size) :: rc
        end function compute_k_matrix_stored_c

        function compute_k_matrix_unrestricted_stored_c(&
            k_alpha_out, k_beta_out, density_alpha, density_beta, eri_dense, n, coeff &
            ) bind(C, name="compute_k_matrix_unrestricted_stored_c") result(rc)
            import :: aedelsten_int_size, aedelsten_float_size
            real(kind=aedelsten_float_size), intent(inout)      :: k_alpha_out(*)
            real(kind=aedelsten_float_size), intent(inout)      :: k_beta_out(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_alpha(*)
            real(kind=aedelsten_float_size), intent(in)         :: density_beta(*)
            real(kind=aedelsten_float_size), intent(in)         :: eri_dense(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            real(kind=aedelsten_float_size), intent(in), value  :: coeff
            integer(kind=aedelsten_int_size) :: rc
        end function compute_k_matrix_unrestricted_stored_c

        ! ── Memory diagnostics, Molcas-export identity data, Cartesian rescaling ──────────

        function get_n_primitive_rows_c(ctx) bind(C, name="get_n_primitive_rows_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_primitive_rows_c

        function memory_breakdown_field_count_c() bind(C, name="memory_breakdown_field_count_c") result(n)
            import :: aedelsten_int_size
            integer(kind=aedelsten_int_size) :: n
        end function memory_breakdown_field_count_c

        function memory_breakdown_c(ctx, output, length) bind(C, name="memory_breakdown_c") result(rc)
            import :: c_ptr, aedelsten_int_size, c_int64_t
            type(c_ptr), intent(in), value :: ctx
            integer(kind=c_int64_t), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function memory_breakdown_c

        function get_basis_function_ids_c(ctx, output, length) bind(C, name="get_basis_function_ids_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function get_basis_function_ids_c

        function get_n_model_functions_c(ctx) bind(C, name="get_n_model_functions_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_model_functions_c

        function get_model_functions_c(ctx, ids, ids_length, energies, energies_length) bind(C, name="get_model_functions_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(inout) :: ids(*)
            integer(kind=aedelsten_int_size), intent(in), value :: ids_length
            real(kind=aedelsten_float_size), intent(inout) :: energies(*)
            integer(kind=aedelsten_int_size), intent(in), value :: energies_length
            integer(kind=aedelsten_int_size) :: rc
        end function get_model_functions_c

        function compute_model_basis_overlaps_c(ctx, s_model_working, s_mw_length, s_model_model, s_mm_length) &
                bind(C, name="compute_model_basis_overlaps_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: s_model_working(*)
            integer(kind=aedelsten_int_size), intent(in), value :: s_mw_length
            type(c_ptr), intent(in), value :: s_model_model
            integer(kind=aedelsten_int_size), intent(in), value :: s_mm_length
            integer(kind=aedelsten_int_size) :: rc
        end function compute_model_basis_overlaps_c

        function compute_model_fockian_c(ctx, output, length) bind(C, name="compute_model_fockian_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function compute_model_fockian_c

        function compute_overlap_with_c(ctx, other, output, length) bind(C, name="compute_overlap_with_c") result(rc)
            import :: c_ptr, c_int64_t, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx, other
            real(kind=aedelsten_float_size), intent(inout) :: output(*)
            integer(kind=c_int64_t), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function compute_overlap_with_c

        function get_primitive_ids_c(ctx, output, length) bind(C, name="get_primitive_ids_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function get_primitive_ids_c

        function get_primitives_c(ctx, output, length) bind(C, name="get_primitives_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function get_primitives_c

        function get_centre_label_c(ctx, centre_idx, buf, buf_len) bind(C, name="get_centre_label_c") result(rc)
            import :: c_ptr, aedelsten_int_size, c_char
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: centre_idx
            character(kind=c_char), intent(inout) :: buf(*)
            integer(kind=aedelsten_int_size), intent(in), value :: buf_len
            integer(kind=aedelsten_int_size) :: rc
        end function get_centre_label_c

        function rescale_cartesian_ao_rows_c(ctx, convention, n_cols, matrix, length, rescaled) &
                bind(C, name="rescale_cartesian_ao_rows_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: convention
            integer(kind=aedelsten_int_size), intent(in), value :: n_cols
            real(kind=aedelsten_float_size), intent(inout) :: matrix(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size), intent(out) :: rescaled
            integer(kind=aedelsten_int_size) :: rc
        end function rescale_cartesian_ao_rows_c

        function unrescale_cartesian_ao_rows_c(ctx, convention, n_cols, matrix, length, rescaled) &
                bind(C, name="unrescale_cartesian_ao_rows_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: convention
            integer(kind=aedelsten_int_size), intent(in), value :: n_cols
            real(kind=aedelsten_float_size), intent(inout) :: matrix(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size), intent(out) :: rescaled
            integer(kind=aedelsten_int_size) :: rc
        end function unrescale_cartesian_ao_rows_c

        function get_n_centres_c(ctx) bind(C, name="get_n_centres_c") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), intent(in), value :: ctx
            integer(kind=c_int64_t) :: n
        end function get_n_centres_c

        function get_centre_geometry_c(ctx, atomic_numbers, positions, kinds, n) &
                bind(C, name="get_centre_geometry_c") result(rc)
            import :: c_ptr, c_int32_t, c_int64_t, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=c_int32_t), intent(inout) :: atomic_numbers(*), kinds(*)
            real(kind=aedelsten_float_size), intent(inout) :: positions(*)
            integer(kind=c_int64_t), intent(in), value :: n
            integer(kind=aedelsten_int_size) :: rc
        end function get_centre_geometry_c

        function get_nuclear_charges_c(ctx, charges, n) bind(C, name="get_nuclear_charges_c") result(rc)
            import :: c_ptr, c_int64_t, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: charges(*)
            integer(kind=c_int64_t), intent(in), value :: n
            integer(kind=aedelsten_int_size) :: rc
        end function get_nuclear_charges_c

        function rescale_cartesian_ao_columns_c(ctx, convention, n_rows, matrix, length, rescaled) &
                bind(C, name="rescale_cartesian_ao_columns_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: convention
            integer(kind=aedelsten_int_size), intent(in), value :: n_rows
            real(kind=aedelsten_float_size), intent(inout) :: matrix(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size), intent(out) :: rescaled
            integer(kind=aedelsten_int_size) :: rc
        end function rescale_cartesian_ao_columns_c

        subroutine free_picture_change_c(ctx) bind(C, name="free_picture_change_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_picture_change_c

        function has_picture_change_c(ctx) bind(C, name="has_picture_change_c") result(flag)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: flag
        end function has_picture_change_c

        function get_picture_change_transforms_c(ctx, u_large, u_small, length) &
                bind(C, name="get_picture_change_transforms_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: u_large(*), u_small(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            integer(kind=aedelsten_int_size) :: rc
        end function get_picture_change_transforms_c

        function set_picture_change_transforms_c(ctx, u_large, u_small, length, c_light) &
                bind(C, name="set_picture_change_transforms_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in) :: u_large(*), u_small(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            real(kind=aedelsten_float_size), intent(in), value  :: c_light
            integer(kind=aedelsten_int_size) :: rc
        end function set_picture_change_transforms_c

        function set_x2c_frozen_density_c(ctx, density, n) bind(C, name="set_x2c_frozen_density_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)         :: density(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size) :: rc
        end function set_x2c_frozen_density_c

        subroutine free_x2c_frozen_density_c(ctx) bind(C, name="free_x2c_frozen_density_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_x2c_frozen_density_c

        function compute_x2c_frozen_kinetic_gradient_c(ctx, gradient, n_centres, active) &
                bind(C, name="compute_x2c_frozen_kinetic_gradient_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: gradient(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            type(c_ptr), intent(in), value :: active
            integer(kind=aedelsten_int_size) :: rc
        end function compute_x2c_frozen_kinetic_gradient_c

        function compute_x2c_frozen_potential_gradient_c(ctx, gradient, n_centres, operator, active) &
                bind(C, name="compute_x2c_frozen_potential_gradient_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout)      :: gradient(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_centres
            integer(kind=aedelsten_int_size), intent(in), value :: operator
            type(c_ptr), intent(in), value :: active
            integer(kind=aedelsten_int_size) :: rc
        end function compute_x2c_frozen_potential_gradient_c

        function transform_one_electron_operator_c(ctx, op_plain, op_pxp, out, n, n_planes) &
                bind(C, name="transform_one_electron_operator_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)    :: op_plain(*), op_pxp(*)
            real(kind=aedelsten_float_size), intent(inout) :: out(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n, n_planes
            integer(kind=aedelsten_int_size) :: rc
        end function transform_one_electron_operator_c

        function transform_one_electron_spin_operator_c(ctx, op_pxp, out, n, n_planes) &
                bind(C, name="transform_one_electron_spin_operator_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in)    :: op_pxp(*)
            real(kind=aedelsten_float_size), intent(inout) :: out(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n, n_planes
            integer(kind=aedelsten_int_size) :: rc
        end function transform_one_electron_spin_operator_c

        function compute_picture_changed_one_electron_operator_c(ctx, scalar, spin, n, operator, n_components, &
                origin_x, origin_y, origin_z, order, contracted) &
                bind(C, name="compute_picture_changed_one_electron_operator_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: scalar(*), spin(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n, operator, n_components
            real(kind=aedelsten_float_size), intent(in), value :: origin_x, origin_y, origin_z
            integer(kind=aedelsten_int_size), intent(in), value :: order, contracted
            integer(kind=aedelsten_int_size) :: rc
        end function compute_picture_changed_one_electron_operator_c

        function compute_amfi_integrals_c(ctx, output, n) bind(C, name="compute_amfi_integrals_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size) :: rc
        end function compute_amfi_integrals_c

        function set_symmetry_c(ctx, point_group, principal_axis, geometry_tolerance, strict_tolerance, average_warning) &
                bind(C, name="set_symmetry_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            type(c_ptr), intent(in), value :: point_group      ! a C string, or null: the full group
            type(c_ptr), intent(in), value :: principal_axis   ! double[3], or null
            real(kind=aedelsten_float_size), intent(in), value :: geometry_tolerance, strict_tolerance, average_warning
            integer(kind=aedelsten_int_size) :: rc
        end function set_symmetry_c

        subroutine free_symmetry_c(ctx) bind(C, name="free_symmetry_c")
            import :: c_ptr
            type(c_ptr), intent(in), value :: ctx
        end subroutine free_symmetry_c

        function get_symmetry_options_c(ctx, options) bind(C, name="get_symmetry_options_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: options(3)
            integer(kind=aedelsten_int_size) :: rc
        end function get_symmetry_options_c

        function get_symmetrisation_report_c(ctx, name, primitive, found, report) &
                bind(C, name="get_symmetrisation_report_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size, c_char
            type(c_ptr), intent(in), value :: ctx
            character(kind=c_char), intent(in) :: name(*)
            integer(kind=aedelsten_int_size), intent(in), value :: primitive
            integer(kind=aedelsten_int_size), intent(out) :: found
            real(kind=aedelsten_float_size), intent(inout) :: report(4)
            integer(kind=aedelsten_int_size) :: rc
        end function get_symmetrisation_report_c

        function get_point_group_name_c(ctx, buf, buf_len) bind(C, name="get_point_group_name_c") result(rc)
            import :: c_ptr, aedelsten_int_size, c_char
            type(c_ptr), intent(in), value :: ctx
            character(kind=c_char), intent(inout) :: buf(*)
            integer(kind=aedelsten_int_size), intent(in), value :: buf_len
            integer(kind=aedelsten_int_size) :: rc
        end function get_point_group_name_c

        function get_n_irreps_c(ctx) bind(C, name="get_n_irreps_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_irreps_c

        function get_irrep_name_c(ctx, irrep, buf, buf_len) bind(C, name="get_irrep_name_c") result(rc)
            import :: c_ptr, aedelsten_int_size, c_char
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: irrep
            character(kind=c_char), intent(inout) :: buf(*)
            integer(kind=aedelsten_int_size), intent(in), value :: buf_len
            integer(kind=aedelsten_int_size) :: rc
        end function get_irrep_name_c

        function get_irrep_dimension_c(ctx, irrep) bind(C, name="get_irrep_dimension_c") result(d)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: irrep
            integer(kind=aedelsten_int_size) :: d
        end function get_irrep_dimension_c

        function get_irrep_index_c(ctx, name) bind(C, name="get_irrep_index_c") result(k)
            import :: c_ptr, aedelsten_int_size, c_char
            type(c_ptr), intent(in), value :: ctx
            character(kind=c_char), intent(in) :: name(*)
            integer(kind=aedelsten_int_size) :: k
        end function get_irrep_index_c

        function get_totally_symmetric_irrep_c(ctx) bind(C, name="get_totally_symmetric_irrep_c") result(k)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: k
        end function get_totally_symmetric_irrep_c

        function get_cartesian_irrep_c(ctx, lx, ly, lz) bind(C, name="get_cartesian_irrep_c") result(k)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: lx, ly, lz
            integer(kind=aedelsten_int_size) :: k
        end function get_cartesian_irrep_c

        function get_symmetry_standard_axes_c(ctx, axes) bind(C, name="get_symmetry_standard_axes_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: axes(3, 3)
            integer(kind=aedelsten_int_size) :: rc
        end function get_symmetry_standard_axes_c

        function get_symmetry_operation_count_c(ctx) bind(C, name="get_symmetry_operation_count_c") result(h)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: h
        end function get_symmetry_operation_count_c

        function get_symmetry_operation_matrix_c(ctx, op, matrix) bind(C, name="get_symmetry_operation_matrix_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: op
            real(kind=aedelsten_float_size), intent(inout) :: matrix(3, 3)
            integer(kind=aedelsten_int_size) :: rc
        end function get_symmetry_operation_matrix_c

        function get_n_symmetry_centres_c(ctx) bind(C, name="get_n_symmetry_centres_c") result(n)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size) :: n
        end function get_n_symmetry_centres_c

        function get_symmetry_centre_images_c(ctx, op, images, n) bind(C, name="get_symmetry_centre_images_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: op, n
            integer(kind=aedelsten_int_size), intent(inout) :: images(*)
            integer(kind=aedelsten_int_size) :: rc
        end function get_symmetry_centre_images_c

        function get_ao_irrep_counts_c(ctx, counts, n) bind(C, name="get_ao_irrep_counts_c") result(rc)
            import :: c_ptr, aedelsten_int_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(inout) :: counts(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n
            integer(kind=aedelsten_int_size) :: rc
        end function get_ao_irrep_counts_c

        function apply_symmetry_operation_c(ctx, op, coefficients, output, length, n_mo) &
                bind(C, name="apply_symmetry_operation_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: op, length, n_mo
            real(kind=aedelsten_float_size), intent(in) :: coefficients(*)
            real(kind=aedelsten_float_size), intent(inout) :: output(*)
            integer(kind=aedelsten_int_size) :: rc
        end function apply_symmetry_operation_c

        function get_irrep_character_c(ctx, irrep, op, character) bind(C, name="get_irrep_character_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: irrep, op
            real(kind=aedelsten_float_size), intent(out) :: character
            integer(kind=aedelsten_int_size) :: rc
        end function get_irrep_character_c

        function get_irrep_matrix_c(ctx, irrep, op, matrix, length, aligned) bind(C, name="get_irrep_matrix_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            integer(kind=aedelsten_int_size), intent(in), value :: irrep, op, length
            real(kind=aedelsten_float_size), intent(inout) :: matrix(*)
            integer(kind=aedelsten_int_size), intent(out) :: aligned
            integer(kind=aedelsten_int_size) :: rc
        end function get_irrep_matrix_c

        function symmetrise_matrix_c(ctx, matrix, length, average, report) bind(C, name="symmetrise_matrix_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: matrix(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length, average
            real(kind=aedelsten_float_size), intent(inout) :: report(4)
            integer(kind=aedelsten_int_size) :: rc
        end function symmetrise_matrix_c

        function analyse_matrix_c(ctx, matrix, length, irrep, report) bind(C, name="analyse_matrix_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in) :: matrix(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length, irrep
            real(kind=aedelsten_float_size), intent(inout) :: report(4)
            integer(kind=aedelsten_int_size) :: rc
        end function analyse_matrix_c

        function get_irrep_norms_c(ctx, matrix, length, norms, n_norms) bind(C, name="get_irrep_norms_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(in) :: matrix(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length
            real(kind=aedelsten_float_size), intent(inout) :: norms(*)
            integer(kind=aedelsten_int_size), intent(in), value :: n_norms
            integer(kind=aedelsten_int_size) :: rc
        end function get_irrep_norms_c

        function symmetrise_orbitals_c(ctx, orbitals, length, n_mo, irreps, components, dropped) &
                bind(C, name="symmetrise_orbitals_c") result(rc)
            import :: c_ptr, aedelsten_int_size, aedelsten_float_size
            type(c_ptr), intent(in), value :: ctx
            real(kind=aedelsten_float_size), intent(inout) :: orbitals(*)
            integer(kind=aedelsten_int_size), intent(in), value :: length, n_mo
            integer(kind=aedelsten_int_size), intent(inout) :: irreps(*), components(*)
            real(kind=aedelsten_float_size), intent(out) :: dropped
            integer(kind=aedelsten_int_size) :: rc
        end function symmetrise_orbitals_c
    end interface

contains

    ! ── Internal helpers ─────────────────────────────────────────────────

    ! Fortran strings are not null-terminated -- every string handed to a
    ! bind(C) routine that reads it as a C string needs one appended first.
    pure function to_c_string(str) result(out)
        character(len=*), intent(in)                    :: str
        character(len=len_trim(str)+1, kind=c_char)      :: out
        out = trim(str) // c_null_char
    end function to_c_string

    ! The reverse of to_c_string: a NUL-terminated C-char buffer (as filled by e.g.
    ! get_centre_label_c) trimmed at the first NUL into a plain Fortran string.
    pure function from_c_string_buf(buf) result(out)
        character(kind=c_char), intent(in) :: buf(:)
        character(len=:), allocatable      :: out
        integer :: i, n
        n = 0
        do i = 1, size(buf)
            if (buf(i) == c_null_char) exit
            n = n + 1
        end do
        allocate(character(len=n) :: out)
        do i = 1, n
            out(i:i) = buf(i)
        end do
    end function from_c_string_buf

    ! `str` in lower case (ASCII).
    pure function lowercase(str) result(out)
        character(len=*), intent(in) :: str
        character(len=len(str))      :: out
        integer :: i, code
        do i = 1, len(str)
            code = iachar(str(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) then
                out(i:i) = achar(code + 32)
            else
                out(i:i) = str(i:i)
            end if
        end do
    end function lowercase

    ! Resolves a named i32 constant (contraction types, one-electron operator
    ! kinds, byte units, ...) via aedelsten_query_i32_c -- every wire-key
    ! NAME_TABLE on the Rust side is queried through this single entry point,
    ! so the lowercasing/null-termination only has to be done right once.
    ! A named constant of the library by its key ("atom_kind_real", ...); stops on an unknown key.
    function aedelsten_constant(key) result(val)
        character(len=*), intent(in)     :: key
        integer(kind=aedelsten_int_size) :: val
        val = query_i32_checked(key)
    end function aedelsten_constant

    ! The integer constant named `key` (aedelsten_query_i32_c); stops on an unknown key.
    function query_i32_checked(key) result(val)
        character(len=*), intent(in)     :: key
        integer(kind=aedelsten_int_size) :: val
        val = aedelsten_query_i32_c(to_c_string(lowercase(key)))
        if (val == -1) then
            write(error_unit, '(A)') "Aedelsten: unknown constant key '" // trim(key) // "'"
            error stop "Aedelsten: unknown constant key"
        end if
    end function query_i32_checked

    ! The floating-point constant named `key` (aedelsten_query_f64_c); stops on an unknown key.
    function query_f64_checked(key) result(val)
        character(len=*), intent(in)    :: key
        real(kind=aedelsten_float_size) :: val
        val = aedelsten_query_f64_c(to_c_string(lowercase(key)))
        if (ieee_is_nan(val)) then ! NaN sentinel for an unknown key.
            write(error_unit, '(A)') "Aedelsten: unknown constant key '" // trim(key) // "'"
            error stop "Aedelsten: unknown constant key"
        end if
    end function query_f64_checked

    ! Narrows a caller-supplied external_int_size value to the fixed-i32 wire
    ! ABI, guarded rather than left to plain int()'s silent wraparound: with
    ! --int64 (external_int_size = int64), a value outside INTEGER(4)'s range
    ! would otherwise cross into Rust corrupted (e.g. wrapped negative), which
    ! can drive Rust's own array-size arithmetic out of bounds rather than
    ! just producing a wrong answer. A no-op check when external_int_size is
    ! already int32 (the default), since nothing of that kind can ever be
    ! out of i32 range.
    function to_wire_i32(value, label) result(wire)
        integer(kind=external_int_size), intent(in) :: value
        character(len=*),                intent(in) :: label
        integer(kind=aedelsten_int_size)             :: wire
        integer(kind=external_int_size)              :: max_wire, min_wire

        max_wire = int(huge(wire), kind=external_int_size)
        min_wire = -max_wire - 1_external_int_size
        if (value > max_wire .or. value < min_wire) then
            write(error_unit, '(A)') "Aedelsten: '" // trim(label) // &
                "' does not fit in Aedelsten's 32-bit wire integer type"
            error stop "Aedelsten: integer argument out of range for the C ABI"
        end if
        wire = int(value, kind=aedelsten_int_size)
    end function to_wire_i32

    ! ── Context lifecycle ────────────────────────────────────────────────

    ! The context of the input file `filename`; its parallel computations are split over the ranks of communicator
    ! `comm` (its Fortran handle, MPI_Comm%MPI_VAL; absent: this process alone, without MPI).
    subroutine set_aedelsten_context(self, filename, contraction_type, comm)
        class(AedelstenContext), intent(out)          :: self
        character(len=*),        intent(in)           :: filename
        character(len=*),        intent(in), optional :: contraction_type
        integer(kind=aedelsten_int_size), intent(in), optional :: comm

        integer(kind=aedelsten_int_size) :: contraction_int

        ! Default to the library's own default contraction if nothing special was requested.
        if (present(contraction_type)) then
            contraction_int = query_i32_checked("contraction_type_" // trim(lowercase(contraction_type)))
        else
            contraction_int = query_i32_checked("default_contraction_type")
        end if

        self%handle = set_aedelsten_context_c(to_c_string(filename), contraction_int, context_communicator_handle(comm))

        if (.not. c_associated(self%handle)) then
            error stop "Aedelsten: failed to create context"
        end if
    end subroutine set_aedelsten_context

    ! Called automatically when AedelstenContext goes out of scope
    subroutine free_aedelsten_context(self)
        type(AedelstenContext), intent(inout) :: self
        if (c_associated(self%handle)) then
            call free_aedelsten_context_c(self%handle)
            self%handle = c_null_ptr
        end if
    end subroutine free_aedelsten_context

    ! ── Shell pairs ──────────────────────────────────────────────────────

    ! Required before compute_one_electron_integrals. Primitive pairs whose
    ! overlap is below overlap_threshold are dropped; absent, none are.
    subroutine set_shell_pairs(self, overlap_threshold)
        class(AedelstenContext), intent(inout)                :: self
        real(kind=aedelsten_float_size), intent(in), optional :: overlap_threshold

        real(kind=aedelsten_float_size)  :: threshold_local
        integer(kind=aedelsten_int_size) :: rc

        ! Absent: no screening (the C entry point's negative sentinel).
        threshold_local = -1.0_aedelsten_float_size
        if (present(overlap_threshold)) threshold_local = overlap_threshold

        rc = set_shell_pairs_c(self%handle, threshold_local)
        if (rc /= 0) error stop "Aedelsten: failed to set shell pairs"
    end subroutine set_shell_pairs

    ! Frees the shell pairs, and with them the ERI data.
    subroutine free_shell_pairs(self)
        class(AedelstenContext), intent(inout) :: self
        call free_shell_pairs_c(self%handle)
    end subroutine free_shell_pairs

    ! ── Scratch ────────────────────────────────────────────────────────

    ! Required before any integral computation. The caller chooses the size;
    ! the library makes no judgment about it. unit is one of "kib", "mib", "gib"
    ! (case-insensitive); scratch_bytes gives what the chosen operations need, from the input alone. That depends
    ! only on the basis's largest shells, not on the molecule; two-electron integral gradients need the most,
    ! one-electron integrals (X2C included) little. The buffer is committed lazily. It never grows: a computation
    ! that runs out of it fails, naming the size needed.
    subroutine set_scratch(self, size, unit)
        class(AedelstenContext), intent(inout)                 :: self
        integer(kind=external_int_size), intent(in)           :: size
        character(len=*),                intent(in)           :: unit

        integer(kind=aedelsten_int_size) :: size_local, unit_local, rc

        size_local = to_wire_i32(size, "size")

        unit_local = query_i32_checked("byte_unit_" // trim(lowercase(unit)))

        rc = set_scratch_c(self%handle, size_local, unit_local)
        if (rc /= 0) error stop "Aedelsten: failed to set scratch"
    end subroutine set_scratch

    ! Frees the scratch buffer.
    subroutine free_scratch(self)
        class(AedelstenContext), intent(inout) :: self
        call free_scratch_c(self%handle)
    end subroutine free_scratch

    ! Splits the context's parallel computations over communicator `comm` from now on (as in set_aedelsten_context).
    subroutine set_communicator(self, comm)
        class(AedelstenContext), intent(inout)                 :: self
        integer(kind=aedelsten_int_size), intent(in), optional :: comm
        integer(kind=aedelsten_int_size) :: rc
        rc = set_communicator_c(self%handle, context_communicator_handle(comm))
        if (rc /= 0) error stop "Aedelsten: failed to set the communicator"
    end subroutine set_communicator

    ! Replaces the scratch buffer (arguments as set_scratch), e.g. to shrink it to get_scratch_peak_bytes() plus a
    ! margin, or to grow it after an out-of-space error. The new buffer is committed lazily; the high-water mark starts
    ! over.
    subroutine resize_scratch(self, size, unit)
        class(AedelstenContext), intent(inout)                 :: self
        integer(kind=external_int_size), intent(in)           :: size
        character(len=*),                intent(in)           :: unit
        integer(kind=aedelsten_int_size) :: size_local, unit_local, rc
        size_local = to_wire_i32(size, "size")
        unit_local = query_i32_checked("byte_unit_" // trim(lowercase(unit)))
        rc = resize_scratch_c(self%handle, size_local, unit_local)
        if (rc /= 0) error stop "Aedelsten: failed to resize scratch"
    end subroutine resize_scratch

    ! The most bytes of the scratch any computation has used so far (its high-water mark).
    function get_scratch_peak_bytes(self) result(bytes)
        class(AedelstenContext), intent(in) :: self
        integer(kind=c_int64_t)              :: bytes
        bytes = get_scratch_peak_bytes_c(self%handle)
        if (bytes < 0) error stop "Aedelsten: no scratch (set_scratch first)"
    end function get_scratch_peak_bytes

    ! ── Basis-size getters ───────────────────────────────────────────────

    function get_n_cont(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        n = int(get_n_cont_c(self%handle), kind=external_int_size)
    end function get_n_cont

    ! The sum of the shells' primitive AOs (a primitive shared by sibling segments once per sibling).
    function get_n_prim(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        n = int(get_n_prim_c(self%handle), kind=external_int_size)
    end function get_n_prim

    ! The number of distinct primitive AOs: the dimension of the primitive basis.
    function get_n_prim_unique(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        n = int(get_n_prim_unique_c(self%handle), kind=external_int_size)
    end function get_n_prim_unique

    ! ── Molecular properties ─────────────────────────────────────────────

    subroutine set_finite_nuclear_model(self)
        class(AedelstenContext), intent(inout) :: self
        integer(kind=aedelsten_int_size) :: rc
        rc = set_finite_nuclear_model_c(self%handle)
        if (rc /= 0) error stop "Aedelsten: failed to set finite nuclear model"
    end subroutine set_finite_nuclear_model

    ! The centre of mass of the real atoms (bohr).
    function compute_centre_of_mass(self) result(com)
        class(AedelstenContext), intent(in) :: self
        real(kind=aedelsten_float_size) :: com(3)
        call compute_centre_of_mass_c(self%handle, com(1), com(2), com(3))
    end function compute_centre_of_mass

    ! The repulsion energy of the QM region's nuclei and core AIMPs.
    function compute_nuclear_repulsion_energy(self) result(e)
        class(AedelstenContext), intent(in) :: self
        real(kind=aedelsten_float_size) :: e
        e = compute_nuclear_repulsion_energy_c(self%handle)
    end function compute_nuclear_repulsion_energy

    ! The total charge of the QM region's nuclei and core AIMPs.
    function compute_nuclear_charge(self) result(q)
        class(AedelstenContext), intent(in) :: self
        real(kind=aedelsten_float_size) :: q
        q = compute_nuclear_charge_c(self%handle)
    end function compute_nuclear_charge

    ! The energy of the QM region's nuclei and core AIMPs in the external multipoles' potential; NaN without them.
    function compute_nuclear_multipole_interaction_energy(self) result(e)
        class(AedelstenContext), intent(in) :: self
        real(kind=aedelsten_float_size) :: e
        e = compute_nuclear_multipole_interaction_energy_c(self%handle)
    end function compute_nuclear_multipole_interaction_energy

    ! The energy of the QM region's nuclei and core AIMPs in the embedding AIMPs' screened potentials; 0 without them.
    function compute_nuclear_aimp_embedding_energy(self) result(e)
        class(AedelstenContext), intent(in) :: self
        real(kind=aedelsten_float_size) :: e
        e = compute_nuclear_aimp_embedding_energy_c(self%handle)
    end function compute_nuclear_aimp_embedding_energy

    ! ── Primitive (decontracted) shell pairs ────────────────────────────

    ! Required before compute_one_electron_integrals(..., primitive=.true.),
    ! compute_one_electron_picture_change_integrals, or
    ! contract_one_electron_integrals. Pairs whose overlap is below
    ! overlap_threshold are dropped; absent, none are.
    subroutine set_primitive_shell_pairs(self, overlap_threshold)
        class(AedelstenContext), intent(inout)                :: self
        real(kind=aedelsten_float_size), intent(in), optional :: overlap_threshold

        real(kind=aedelsten_float_size)  :: threshold_local
        integer(kind=aedelsten_int_size) :: rc

        ! Absent: no screening (the C entry point's negative sentinel).
        threshold_local = -1.0_aedelsten_float_size
        if (present(overlap_threshold)) threshold_local = overlap_threshold

        rc = set_primitive_shell_pairs_c(self%handle, threshold_local)
        if (rc /= 0) error stop "Aedelsten: failed to set primitive shell pairs"
    end subroutine set_primitive_shell_pairs

    ! Frees the primitive shell pairs.
    subroutine free_primitive_shell_pairs(self)
        class(AedelstenContext), intent(inout) :: self
        call free_primitive_shell_pairs_c(self%handle)
    end subroutine free_primitive_shell_pairs

    ! Frees the parsed input (basis, nuclei, embedding). Only after the last call that reads it (the operators, the
    ! shell pairs, the primitive shell pairs, the embedding): such a call afterwards stops the program.
    subroutine free_input(self)
        class(AedelstenContext), intent(inout) :: self
        call free_input_c(self%handle)
    end subroutine free_input

    ! Frees the primitive shells made by set_primitive_shell_pairs; the primitive shell pairs stay.
    subroutine free_primitive_shells(self)
        class(AedelstenContext), intent(inout) :: self
        call free_primitive_shells_c(self%handle)
    end subroutine free_primitive_shells

    ! Frees the contraction matrix made by set_primitive_shell_pairs.
    subroutine free_contraction_matrix(self)
        class(AedelstenContext), intent(inout) :: self
        call free_contraction_matrix_c(self%handle)
    end subroutine free_contraction_matrix

    ! Whether the primitive basis is set (always for a primitive context).
    function has_primitive(self) result(flag)
        class(AedelstenContext), intent(in) :: self
        logical :: flag
        flag = has_primitive_c(self%handle) /= 0
    end function has_primitive

    ! ── External multipole field ────────────────────────────────────────

    subroutine set_external_multipoles(self, filename)
        class(AedelstenContext), intent(inout) :: self
        character(len=*),        intent(in)    :: filename
        integer(kind=aedelsten_int_size) :: rc
        rc = set_external_multipoles_c(self%handle, to_c_string(filename))
        if (rc /= 0) error stop "Aedelsten: failed to set external multipoles"
    end subroutine set_external_multipoles

    ! Frees the external multipoles.
    subroutine free_external_multipoles(self)
        class(AedelstenContext), intent(inout) :: self
        call free_external_multipoles_c(self%handle)
    end subroutine free_external_multipoles

    ! Whether external multipoles are set.
    function has_external_multipoles(self) result(flag)
        class(AedelstenContext), intent(in) :: self
        logical :: flag
        flag = has_external_multipoles_c(self%handle) /= 0
    end function has_external_multipoles

    ! ── AIMP ─────────────────────────────────────────────────────────────

    function has_core_aimp(self) result(flag)
        class(AedelstenContext), intent(in) :: self
        logical :: flag
        flag = has_core_aimp_c(self%handle) /= 0
    end function has_core_aimp

    ! Whether the input has embedding AIMPs.
    function has_embedding_aimp(self) result(flag)
        class(AedelstenContext), intent(in) :: self
        logical :: flag
        flag = has_embedding_aimp_c(self%handle) /= 0
    end function has_embedding_aimp

    ! The number of embedding AIMPs.
    function get_n_embedding_aimps(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=int64) :: n
        n = get_n_embedding_aimps_c(self%handle)
        if (n < 0) error stop "Aedelsten: could not count the embedding AIMPs"
    end function get_n_embedding_aimps

    ! `atomic_numbers(s)` (0 when the block's name is not an element symbol), `positions(:, s)` (bohr) and
    ! `charges(s)` for embedding AIMP s, in input order; each sized by get_n_embedding_aimps.
    subroutine get_embedding_aimp_sites(self, atomic_numbers, positions, charges)
        class(AedelstenContext),         intent(in)    :: self
        integer(kind=c_int32_t),         intent(inout) :: atomic_numbers(:)
        real(kind=aedelsten_float_size), intent(inout) :: positions(:,:), charges(:)
        integer(kind=int64) :: n
        integer(kind=aedelsten_int_size) :: rc
        n = self%get_n_embedding_aimps()
        if (.not. (is_contiguous(atomic_numbers) .and. is_contiguous(positions) .and. is_contiguous(charges))) &
            error stop "Aedelsten: get_embedding_aimp_sites: the outputs must be contiguous"
        if (size(atomic_numbers, kind=int64) /= n .or. size(positions, 1) /= 3 .or. size(positions, 2, kind=int64) /= n &
            .or. size(charges, kind=int64) /= n) &
            error stop "Aedelsten: get_embedding_aimp_sites: the outputs must be (n), (3, n) and (n), n embedding AIMPs"
        rc = get_embedding_aimp_sites_c(self%handle, atomic_numbers, positions, charges, int(n, c_int64_t))
        if (rc /= 0) error stop "Aedelsten: get_embedding_aimp_sites failed"
    end subroutine get_embedding_aimp_sites

    ! Builds the embedding AIMPs' projection operators from the input's embedding AIMPs. A filename is refused.
    subroutine set_embedding_aimp_operators(self, filename)
        class(AedelstenContext), intent(inout)      :: self
        character(len=*), intent(in), optional      :: filename
        character(len=:), allocatable, target :: filename_c
        integer(kind=aedelsten_int_size)      :: rc

        if (present(filename)) then
            filename_c = to_c_string(filename)
            rc = set_embedding_aimp_operators_c(self%handle, c_loc(filename_c))
        else
            rc = set_embedding_aimp_operators_c(self%handle, c_null_ptr)
        end if
        if (rc /= 0) error stop "Aedelsten: failed to set embedding AIMP operators"
    end subroutine set_embedding_aimp_operators

    ! Frees the embedding AIMPs' projection operators.
    subroutine free_embedding_aimp_operators(self)
        class(AedelstenContext), intent(inout) :: self
        call free_embedding_aimp_operators_c(self%handle)
    end subroutine free_embedding_aimp_operators

    ! ── Integral routines ────────────────────────────────────────────────

    ! Resolves an operator name to its wire-key integer (Aedelsten requires
    ! the fully-qualified "oneint_XX" wire key, but a Fortran caller may pass
    ! only "XX", so the prefix is prepended here when not already present,
    ! and lowercased throughout), and checks that whether it names a
    ! component operator (moment, angular_momentum, nabla: the rank-3
    ! outputs; everything else is rank-2) matches expect_components, so this
    ! catches e.g. ctx%compute_one_electron_integrals(matrix_2d, "moment")
    ! -- or the reverse -- before ever reaching Rust. Among the rank-2 ones:
    ! "sap_helfem" and "sap_grasp", the SAP screening (each atom's electrons
    ! as Gaussian charges fitted to nonrelativistic or four-component atoms).
    function resolve_operator(operator, expect_components) result(operator_int)
        character(len=*), intent(in)     :: operator
        logical,           intent(in)    :: expect_components
        integer(kind=aedelsten_int_size) :: operator_int
        character(len=:), allocatable    :: operator_lower, operator_key
        logical                          :: is_component

        operator_lower = lowercase(trim(operator))
        if (len(operator_lower) >= len(oneint_label)) then
            if (operator_lower(1:len(oneint_label)) == oneint_label) then
                operator_key = operator_lower
            else
                operator_key = oneint_label // operator_lower
            end if
        else
            operator_key = oneint_label // operator_lower
        end if

        operator_int = query_i32_checked(operator_key)
        is_component = compute_one_electron_component_count_c(operator_int, 0_aedelsten_int_size) > 0
        if (is_component .neqv. expect_components) then
            if (expect_components) then
                write(error_unit, '(A)') "Aedelsten: a rank-3 output array requires a component operator " // &
                    "('moment', 'angular_momentum', 'nabla'), got '" // trim(operator) // "'"
            else
                write(error_unit, '(A)') "Aedelsten: '" // trim(operator) // "' needs a rank-3 output array " // &
                    "(n, n, n_planes) -- call compute_one_electron_integrals with a 3-D array instead"
            end if
            error stop "Aedelsten: operator/output-rank mismatch"
        end if
    end function resolve_operator

    ! Every non-moment one-electron integral (overlap, kinetic,
    ! nuclear_attraction, the AIMP terms, ...): output is a plain (n, n)
    ! matrix, with n taken from output's own shape rather than a separate
    ! scalar argument a caller's array could silently disagree with.
    subroutine compute_one_electron_integrals_matrix(self, output, operator, primitive)
        class(AedelstenContext), intent(inout) :: self
        real(kind=aedelsten_float_size), intent(inout) :: output(:,:)
        character(len=*), intent(in)           :: operator
        logical, intent(in), optional          :: primitive

        integer(kind=aedelsten_int_size) :: n, operator_int, primitive_int, rc

        if (.not. is_contiguous(output)) then
            write(error_unit, '(A)') "Aedelsten: output array must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: output array is not contiguous"
        end if
        if (size(output, 1) /= size(output, 2)) then
            write(error_unit, '(A)') "Aedelsten: one-electron integral output matrix must be square"
            error stop "Aedelsten: one-electron integral output matrix must be square"
        end if
        n = size(output, 1, kind=aedelsten_int_size)
        operator_int = resolve_operator(operator, expect_components=.false.)

        primitive_int = 0_aedelsten_int_size
        if (present(primitive)) then
            if (primitive) primitive_int = 1_aedelsten_int_size
        end if

        rc = compute_one_electron_integrals_c( &
            self%handle, output, n, operator_int, 1_aedelsten_int_size, &
            0.0_aedelsten_float_size, 0.0_aedelsten_float_size, 0.0_aedelsten_float_size, &
            0_aedelsten_int_size, primitive_int)

        ! Rust should already have killed the run before this.
        if (rc /= 0) error stop "Problem when evaluating one-electron integrals. Stopping program."
    end subroutine compute_one_electron_integrals_matrix

    ! The component operators: output is an (n, n, n_planes) tensor, one
    ! Cartesian plane per component -- the moment (order 0 -> 1 plane,
    ! order 1 (dipole) -> 3, order 2 (quadrupole) -> 6, ...; `order`
    ! required, about `origin`), the angular momentum about the gauge origin
    ! `origin` (3 planes, x y z, real antisymmetric: L = -i A), nabla (3
    ! planes, real antisymmetric: p = -i nabla; no origin). `origin` defaults
    ! to (0, 0, 0). n and n_planes are both taken from output's own shape,
    ! and n_planes is cross-checked against what the operator produces
    ! (queried from Rust, not re-derived by hand here) before the call,
    ! rather than trusting the caller's array to be the right size.
    subroutine compute_one_electron_integrals_tensor(self, output, operator, origin, order, primitive)
        class(AedelstenContext), intent(inout)        :: self
        real(kind=aedelsten_float_size), intent(inout) :: output(:,:,:)
        character(len=*), intent(in)                   :: operator
        real(kind=aedelsten_float_size), intent(in), optional :: origin(3)
        integer(kind=external_int_size), intent(in), optional :: order
        logical, intent(in), optional                  :: primitive

        integer(kind=aedelsten_int_size) :: n, n_planes, order_wire, operator_int, expected_planes, primitive_int, rc
        real(kind=aedelsten_float_size)  :: origin_wire(3)

        if (.not. is_contiguous(output)) then
            write(error_unit, '(A)') "Aedelsten: output array must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: output array is not contiguous"
        end if
        if (size(output, 1) /= size(output, 2)) then
            write(error_unit, '(A)') "Aedelsten: one-electron integral output tensor must be square in its first two dimensions"
            error stop "Aedelsten: one-electron integral output tensor must be square in its first two dimensions"
        end if
        n = size(output, 1, kind=aedelsten_int_size)
        n_planes = size(output, 3, kind=aedelsten_int_size)
        operator_int = resolve_operator(operator, expect_components=.true.)
        if (operator_int == query_i32_checked(oneint_label // "moment") .and. .not. present(order)) then
            error stop "Aedelsten: the moment operator needs an order"
        end if
        order_wire = 0
        if (present(order)) order_wire = to_wire_i32(order, "order")
        origin_wire = 0.0_aedelsten_float_size
        if (present(origin)) origin_wire = origin

        expected_planes = compute_one_electron_component_count_c(operator_int, order_wire)
        if (n_planes /= expected_planes) then
            write(error_unit, '(A)') "Aedelsten: output array's third dimension does not match the number " // &
                "of components this operator produces"
            error stop "Aedelsten: component output array has the wrong number of components"
        end if

        primitive_int = 0_aedelsten_int_size
        if (present(primitive)) then
            if (primitive) primitive_int = 1_aedelsten_int_size
        end if

        rc = compute_one_electron_integrals_c( &
            self%handle, output, n, operator_int, n_planes, &
            origin_wire(1), origin_wire(2), origin_wire(3), order_wire, primitive_int)

        ! Rust should already have killed the run before this.
        if (rc /= 0) error stop "Problem when evaluating one-electron integrals. Stopping program."
    end subroutine compute_one_electron_integrals_tensor

    ! Escape hatch for callers managing their own flat/linearized storage
    ! (e.g. a big reused scratch buffer, or code that otherwise wants
    ! manual 1-D indexing) rather than a shaped (n,n) or (n,n,n_planes)
    ! array: n/n_planes/origin/order are explicit, matching a plain C-style
    ! call -- but unlike the raw C ABI itself, output is assumed-shape
    ! rank-1 rather than assumed-size, so its real size is still checked
    ! against n*n*n_planes (and, when order/origin are given, n_planes is
    ! additionally cross-checked against what that moment order actually
    ! produces) before the call, rather than trusting the caller outright.
    subroutine compute_one_electron_integrals_flat(self, output, n, operator, n_planes, origin, order, primitive)
        class(AedelstenContext), intent(inout)             :: self
        real(kind=aedelsten_float_size), intent(inout)     :: output(:)
        integer(kind=external_int_size), intent(in)        :: n
        character(len=*), intent(in)                       :: operator
        integer(kind=external_int_size), intent(in), optional :: n_planes
        real(kind=aedelsten_float_size), intent(in), optional :: origin(3)
        integer(kind=external_int_size), intent(in), optional :: order
        logical, intent(in), optional                         :: primitive

        integer(kind=aedelsten_int_size) :: n_wire, n_planes_wire, order_wire, operator_int, primitive_int, &
            expected_planes, rc
        real(kind=aedelsten_float_size)  :: origin_x, origin_y, origin_z
        logical                          :: expect_components

        if (.not. is_contiguous(output)) then
            write(error_unit, '(A)') "Aedelsten: output array must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: output array is not contiguous"
        end if

        ! order and origin are bundled arguments (a component operator: the moment's order and origin, the
        ! angular momentum's gauge origin with any order), so force both to be present at the same time, or both absent.
        if (present(order) .neqv. present(origin)) then
            error stop "Aedelsten: order and origin must both be present, or both absent, for one-electron integrals"
        end if
        expect_components = present(order)

        n_wire = to_wire_i32(n, "n")
        if (present(n_planes)) then
            n_planes_wire = to_wire_i32(n_planes, "n_planes")
        else
            n_planes_wire = 1
        end if

        if (size(output, kind=aedelsten_int_size) /= n_wire * n_wire * n_planes_wire) then
            write(error_unit, '(A)') "Aedelsten: output buffer's size does not match n*n*n_planes"
            error stop "Aedelsten: output buffer size mismatch"
        end if

        operator_int = resolve_operator(operator, expect_components=expect_components)
        if (expect_components) then
            order_wire = to_wire_i32(order, "order")
            expected_planes = compute_one_electron_component_count_c(operator_int, order_wire)
            if (n_planes_wire /= expected_planes) then
                write(error_unit, '(A)') "Aedelsten: n_planes does not match the number of components " // &
                    "this operator produces"
                error stop "Aedelsten: component output buffer has the wrong number of components"
            end if
            origin_x = origin(1)
            origin_y = origin(2)
            origin_z = origin(3)
        else
            order_wire = 0
            origin_x = 0.0_aedelsten_float_size
            origin_y = 0.0_aedelsten_float_size
            origin_z = 0.0_aedelsten_float_size
        end if

        primitive_int = 0_aedelsten_int_size
        if (present(primitive)) then
            if (primitive) primitive_int = 1_aedelsten_int_size
        end if

        rc = compute_one_electron_integrals_c( &
            self%handle, output, n_wire, operator_int, n_planes_wire, &
            origin_x, origin_y, origin_z, order_wire, primitive_int)

        ! Rust should already have killed the run before this.
        if (rc /= 0) error stop "Problem when evaluating one-electron integrals. Stopping program."
    end subroutine compute_one_electron_integrals_flat

    ! ── Integral gradients ───────────────────────────────────────────────

    ! The number of centres `compute_one_electron_gradient`'s matrix has columns for: real atoms and pseudocharge
    ! ghosts. Orthogonality-function ghosts (last in the centre list) are not degrees of freedom and not included.
    function get_n_gradient_centres(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        n = int(get_n_gradient_centres_c(self%handle), kind=external_int_size)
    end function get_n_gradient_centres

    ! Checks a (3, n_centres) gradient and an optional active(n_centres) mask; returns the mask as C flags.
    subroutine check_classical_gradient(self, gradient, active, n_centres, active_flags, active_ptr)
        class(AedelstenContext), intent(in)            :: self
        real(kind=aedelsten_float_size), intent(in)    :: gradient(:,:)
        logical, intent(in), optional                  :: active(:)
        integer(kind=aedelsten_int_size), intent(out)  :: n_centres
        integer(kind=c_int), allocatable, target, intent(inout) :: active_flags(:)
        type(c_ptr), intent(out)                       :: active_ptr

        if (.not. is_contiguous(gradient)) error stop "Aedelsten: gradient array is not contiguous"
        if (size(gradient, 1) /= 3) error stop "Aedelsten: the gradient's first dimension must be exactly 3 (x, y, z)"
        if (size(gradient, 2, kind=aedelsten_int_size) /= get_n_gradient_centres_c(self%handle)) then
            error stop "Aedelsten: the gradient's second dimension must be get_n_gradient_centres()"
        end if
        n_centres = size(gradient, 2, kind=aedelsten_int_size)
        active_ptr = c_null_ptr
        if (present(active)) then
            if (size(active, kind=aedelsten_int_size) /= n_centres) then
                error stop "Aedelsten: active must have one flag per gradient centre"
            end if
            allocate(active_flags(n_centres))
            active_flags = merge(1_c_int, 0_c_int, active)
            active_ptr = c_loc(active_flags)
        end if
    end subroutine check_classical_gradient

    ! gradient(3, n_centres) is OVERWRITTEN with the gradient of compute_nuclear_repulsion_energy (nuclei and core
    ! AIMPs, which move with their centres). `active` as in compute_one_electron_gradient.
    subroutine compute_nuclear_repulsion_gradient(self, gradient, active)
        class(AedelstenContext), intent(inout)         :: self
        real(kind=aedelsten_float_size), intent(inout) :: gradient(:,:)
        logical, intent(in), optional                  :: active(:)
        integer(kind=aedelsten_int_size) :: n_centres, rc
        integer(kind=c_int), allocatable, target :: active_flags(:)
        type(c_ptr) :: active_ptr
        call check_classical_gradient(self, gradient, active, n_centres, active_flags, active_ptr)
        rc = compute_nuclear_repulsion_gradient_c(self%handle, gradient, n_centres, active_ptr)
        if (rc /= 0) error stop "Problem when evaluating the nuclear repulsion gradient. Stopping program."
    end subroutine compute_nuclear_repulsion_gradient

    ! The gradient of compute_nuclear_aimp_embedding_energy (the QM nuclei and core AIMPs in the fixed embedding AIMPs).
    subroutine compute_nuclear_aimp_embedding_gradient(self, gradient, active)
        class(AedelstenContext), intent(inout)         :: self
        real(kind=aedelsten_float_size), intent(inout) :: gradient(:,:)
        logical, intent(in), optional                  :: active(:)
        integer(kind=aedelsten_int_size) :: n_centres, rc
        integer(kind=c_int), allocatable, target :: active_flags(:)
        type(c_ptr) :: active_ptr
        call check_classical_gradient(self, gradient, active, n_centres, active_flags, active_ptr)
        rc = compute_nuclear_aimp_embedding_gradient_c(self%handle, gradient, n_centres, active_ptr)
        if (rc /= 0) error stop "Problem when evaluating the nuclear-AIMP embedding gradient. Stopping program."
    end subroutine compute_nuclear_aimp_embedding_gradient

    ! The gradient of compute_nuclear_multipole_interaction_energy (the QM nuclei and core AIMPs in the fixed external
    ! multipoles); needs set_external_multipoles.
    subroutine compute_nuclear_multipole_interaction_gradient(self, gradient, active)
        class(AedelstenContext), intent(inout)         :: self
        real(kind=aedelsten_float_size), intent(inout) :: gradient(:,:)
        logical, intent(in), optional                  :: active(:)
        integer(kind=aedelsten_int_size) :: n_centres, rc
        integer(kind=c_int), allocatable, target :: active_flags(:)
        type(c_ptr) :: active_ptr
        call check_classical_gradient(self, gradient, active, n_centres, active_flags, active_ptr)
        rc = compute_nuclear_multipole_interaction_gradient_c(self%handle, gradient, n_centres, active_ptr)
        if (rc /= 0) error stop "Problem when evaluating the nuclear-multipole gradient. Stopping program."
    end subroutine compute_nuclear_multipole_interaction_gradient

    ! Moves the gradient centres to positions(3, get_n_gradient_centres()) (Bohr), for the next geometry step: each
    ! centre carries its basis shells, nucleus and core AIMP; orthogonality ghosts and embedding AIMPs stay fixed.
    ! The basis, scratch and environment are kept; the geometry-derived data is invalidated, so call
    ! set_shell_pairs and set_eri_data (and set_primitive_shell_pairs / the X2C construction, if used) again.
    subroutine set_centre_positions(self, positions)
        class(AedelstenContext), intent(inout)      :: self
        real(kind=aedelsten_float_size), intent(in) :: positions(:,:)

        integer(kind=aedelsten_int_size) :: rc

        if (.not. is_contiguous(positions)) then
            error stop "Aedelsten: positions array is not contiguous"
        end if
        if (size(positions, 1) /= 3) then
            error stop "Aedelsten: the positions' first dimension must be exactly 3 (x, y, z)"
        end if
        if (size(positions, 2, kind=aedelsten_int_size) /= get_n_gradient_centres_c(self%handle)) then
            error stop "Aedelsten: the positions' second dimension must be get_n_gradient_centres()"
        end if
        rc = set_centre_positions_c(self%handle, positions, size(positions, 2, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Problem when moving the centres. Stopping program."
    end subroutine set_centre_positions

    ! gradient(3, n_centres) is OVERWRITTEN with sum_mn density(m,n) dO_mn/dX_c for the operator (same names as
    ! compute_one_electron_integrals: overlap -- pass the energy-weighted density --, kinetic, nuclear_attraction,
    ! external_multipole, the AIMP operators; not moment), where X runs over the get_n_gradient_centres()
    ! centres and c over x, y, z. Sources that are not centres (external charges/multipoles, embedding AIMPs)
    ! do not move and give no forces here.
    !
    ! `active(n_centres)` (optional; all centres if absent) freezes centres: a frozen centre's column is returned
    ! as zero and is not computed, and work that only feeds frozen columns is skipped (pairs of two frozen
    ! centres, the source terms of frozen nuclei and core AIMPs).
    subroutine compute_one_electron_gradient(self, gradient, density, operator, active)
        class(AedelstenContext), intent(inout)         :: self
        real(kind=aedelsten_float_size), intent(inout) :: gradient(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density(:,:)
        character(len=*), intent(in)                   :: operator
        logical, intent(in), optional                  :: active(:)

        integer(kind=aedelsten_int_size) :: n, n_centres, operator_int, rc
        integer(kind=c_int), allocatable, target :: active_flags(:)
        type(c_ptr) :: active_ptr

        if (.not. is_contiguous(gradient)) then
            error stop "Aedelsten: gradient array is not contiguous"
        end if
        if (size(gradient, 1) /= 3) then
            write(error_unit, '(A)') "Aedelsten: the gradient's first dimension must be exactly 3 (x, y, z)"
            error stop "Aedelsten: gradient has the wrong first dimension"
        end if
        if (size(gradient, 2, kind=aedelsten_int_size) /= get_n_gradient_centres_c(self%handle)) then
            write(error_unit, '(A)') "Aedelsten: the gradient's second dimension must be get_n_gradient_centres()"
            error stop "Aedelsten: gradient has the wrong number of centres"
        end if
        if (size(density, 1) /= size(density, 2)) then
            error stop "Aedelsten: the density must be square"
        end if
        n = size(density, 1, kind=aedelsten_int_size)
        n_centres = size(gradient, 2, kind=aedelsten_int_size)
        operator_int = resolve_operator(operator, expect_components=.false.)

        active_ptr = c_null_ptr
        if (present(active)) then
            if (size(active, kind=aedelsten_int_size) /= n_centres) then
                error stop "Aedelsten: active must have one flag per gradient centre"
            end if
            allocate(active_flags(n_centres))
            active_flags = merge(1_c_int, 0_c_int, active)
            active_ptr = c_loc(active_flags)
        end if

        rc = compute_one_electron_gradient_c(self%handle, gradient, n_centres, density, n, operator_int, active_ptr)
        if (rc /= 0) error stop "Problem when evaluating a one-electron gradient. Stopping program."
    end subroutine compute_one_electron_gradient

    ! The number of shells.
    function get_n_shells(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        n = int(get_n_shells_c(self%handle), kind=external_int_size)
    end function get_n_shells

    ! shells(:,1)=centre (0-based), shells(:,2)=l (negated for a Cartesian shell), shells(:,3)=first AO (0-based),
    ! shells(:,4)=AO count, one row per shell in AO order. `shells` must be pre-allocated to (get_n_shells(), 4).
    subroutine get_shells(self, shells)
        class(AedelstenContext),          intent(in)    :: self
        integer(kind=aedelsten_int_size), intent(inout) :: shells(:,:)
        integer(kind=aedelsten_int_size)                :: rc
        if (size(shells, 1, kind=external_int_size) /= self%get_n_shells() .or. size(shells, 2) /= 4) &
            error stop "Aedelsten: get_shells: shells must be (get_n_shells(), 4)"
        rc = get_shells_c(self%handle, shells, size(shells, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: get_shells failed"
    end subroutine get_shells

    ! extents(s): the distance from shell s's centre beyond which its AOs and every component of their gradients are
    ! below threshold in magnitude. `extents` must have get_n_shells() entries.
    subroutine compute_shell_extents(self, threshold, extents)
        class(AedelstenContext), intent(in)            :: self
        real(kind=aedelsten_float_size), intent(in)    :: threshold
        real(kind=aedelsten_float_size), intent(inout) :: extents(:)
        integer(kind=aedelsten_int_size)               :: rc
        if (size(extents, kind=external_int_size) /= self%get_n_shells()) &
            error stop "Aedelsten: compute_shell_extents: extents must have get_n_shells() entries"
        rc = compute_shell_extents_c(self%handle, threshold, extents, size(extents, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: compute_shell_extents failed"
    end subroutine compute_shell_extents

    ! The AOs of the shells `shells` (0-based, any order and subset) at points (3, n_points) or (4, n_points) (x, y, z
    ! and a weight that is not read): values (n_listed, n_points), the AOs of the listed shells in the order of the
    ! list, and optionally gradients (n_listed, n_points, 3) (x, y, z).
    subroutine compute_ao_values(self, shells, points, values, gradients)
        class(AedelstenContext), intent(in)                     :: self
        integer(kind=aedelsten_int_size), intent(in)            :: shells(:)
        real(kind=aedelsten_float_size), intent(in)             :: points(:,:)
        real(kind=aedelsten_float_size), intent(inout)          :: values(:,:)
        real(kind=aedelsten_float_size), intent(inout), optional, target, contiguous :: gradients(:,:,:)
        type(c_ptr)                                             :: gradients_ptr
        integer(kind=aedelsten_int_size)                        :: rc

        if (.not. is_contiguous(shells) .or. .not. is_contiguous(points) .or. .not. is_contiguous(values)) then
            write(error_unit, '(A)') "Aedelsten: shells/points/values must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (size(points, 1) /= 3 .and. size(points, 1) /= 4) &
            error stop "Aedelsten: compute_ao_values: points must be (3, n_points) or (4, n_points)"
        if (size(values, 2) /= size(points, 2)) &
            error stop "Aedelsten: compute_ao_values: values and points disagree on n_points"
        gradients_ptr = c_null_ptr
        if (present(gradients)) then
            if (size(gradients, 1) /= size(values, 1) .or. size(gradients, 2) /= size(values, 2) .or. size(gradients, 3) /= 3) &
                error stop "Aedelsten: compute_ao_values: gradients must be (n_listed, n_points, 3)"
            gradients_ptr = c_loc(gradients)
        end if
        rc = compute_ao_values_c(self%handle, shells, size(shells, kind=aedelsten_int_size), points, &
            size(points, 1, kind=aedelsten_int_size), size(points, 2, kind=aedelsten_int_size), values, gradients_ptr, &
            size(values, 1, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: compute_ao_values failed"
    end subroutine compute_ao_values

    ! As compute_ao_values, with the gradients (n_listed, n_points, 3) (x, y, z) and the second derivatives
    ! (n_listed, n_points, 6) (xx, xy, xz, yy, yz, zz) both returned.
    subroutine compute_ao_hessians(self, shells, points, values, gradients, hessians)
        class(AedelstenContext), intent(in)              :: self
        integer(kind=aedelsten_int_size), intent(in)     :: shells(:)
        real(kind=aedelsten_float_size), intent(in)      :: points(:,:)
        real(kind=aedelsten_float_size), intent(inout)   :: values(:,:)
        real(kind=aedelsten_float_size), intent(inout)   :: gradients(:,:,:)
        real(kind=aedelsten_float_size), intent(inout)   :: hessians(:,:,:)
        integer(kind=aedelsten_int_size)                 :: rc

        if (.not. is_contiguous(shells) .or. .not. is_contiguous(points) .or. .not. is_contiguous(values) .or. &
            .not. is_contiguous(gradients) .or. .not. is_contiguous(hessians)) then
            write(error_unit, '(A)') "Aedelsten: shells/points/values/gradients/hessians must be contiguous (got a " // &
                "strided/non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (size(points, 1) /= 3 .and. size(points, 1) /= 4) &
            error stop "Aedelsten: compute_ao_hessians: points must be (3, n_points) or (4, n_points)"
        if (size(values, 2) /= size(points, 2)) &
            error stop "Aedelsten: compute_ao_hessians: values and points disagree on n_points"
        if (size(gradients, 1) /= size(values, 1) .or. size(gradients, 2) /= size(values, 2) .or. size(gradients, 3) /= 3) &
            error stop "Aedelsten: compute_ao_hessians: gradients must be (n_listed, n_points, 3)"
        if (size(hessians, 1) /= size(values, 1) .or. size(hessians, 2) /= size(values, 2) .or. size(hessians, 3) /= 6) &
            error stop "Aedelsten: compute_ao_hessians: hessians must be (n_listed, n_points, 6)"
        rc = compute_ao_hessians_c(self%handle, shells, size(shells, kind=aedelsten_int_size), points, &
            size(points, 1, kind=aedelsten_int_size), size(points, 2, kind=aedelsten_int_size), values, gradients, hessians, &
            size(values, 1, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: compute_ao_hessians failed"
    end subroutine compute_ao_hessians

    ! ── Two-electron repulsion integrals (ERIs) ──────────────────────────

    ! Bundled range/omega, matching the origin/order convention used
    ! elsewhere: both present or both absent. Absent means full-range
    ! Coulomb (also what the Rust side does regardless of range_wire
    ! whenever omega_wire is exactly 0.0).
    subroutine resolve_eri_range(range, omega, range_wire, omega_wire)
        character(len=*), intent(in), optional                 :: range
        real(kind=aedelsten_float_size), intent(in), optional  :: omega
        integer(kind=aedelsten_int_size), intent(out)          :: range_wire
        real(kind=aedelsten_float_size), intent(out)           :: omega_wire

        if (present(range) .neqv. present(omega)) then
            error stop "Aedelsten: range and omega must both be present, or both absent (absent = full range)"
        end if
        if (present(range)) then
            range_wire = query_i32_checked(eri_range_label // trim(lowercase(range)))
            omega_wire = omega
        else
            range_wire = 0_aedelsten_int_size
            omega_wire = 0.0_aedelsten_float_size
        end if
    end subroutine resolve_eri_range

    ! Rank-4: dense (S1) storage, output(n,n,n,n) self-describes n -- no
    ! ao_symmetry argument needed, same shape-self-describes philosophy as
    ! compute_one_electron_integrals.
    ! Under MPI, both forms are collective: every rank must call them; each
    ! computes its share of the integrals and every rank gets the whole tensor.
    subroutine compute_eri_stored_tensor(self, output, range, omega)
        class(AedelstenContext), intent(inout)                :: self
        real(kind=aedelsten_float_size), intent(inout)        :: output(:,:,:,:)
        character(len=*), intent(in), optional                :: range
        real(kind=aedelsten_float_size), intent(in), optional :: omega

        integer(kind=aedelsten_int_size) :: n, symmetry_int, index_order_int, range_wire, rc
        real(kind=aedelsten_float_size)  :: omega_wire

        if (.not. is_contiguous(output)) then
            write(error_unit, '(A)') "Aedelsten: output array must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: output array is not contiguous"
        end if
        if (size(output,1) /= size(output,2) .or. size(output,1) /= size(output,3) &
                .or. size(output,1) /= size(output,4)) then
            write(error_unit, '(A)') "Aedelsten: ERI output tensor must be square in all four dimensions"
            error stop "Aedelsten: ERI output tensor must be square in all four dimensions"
        end if
        n = size(output, 1, kind=aedelsten_int_size)
        symmetry_int = query_i32_checked(eri_symmetry_label // "s1")
        index_order_int = query_i32_checked("eri_index_order_column_major")
        call resolve_eri_range(range, omega, range_wire, omega_wire)

        rc = compute_eri_stored_c(self%handle, output, n, symmetry_int, index_order_int, range_wire, omega_wire)
        if (rc /= 0) error stop "Problem when evaluating ERIs. Stopping program."
    end subroutine compute_eri_stored_tensor

    ! Rank-1: S4/S8 packed storage (always column-major -- see this
    ! procedure group's own comment in the type definition), or a flat S1
    ! escape hatch. ao_symmetry is required: S4 and S8 sizes generally
    ! can't be told apart from a flat buffer's length alone, and neither
    ! can S1 in general.
    subroutine compute_eri_stored_flat(self, output, ao_symmetry, range, omega)
        class(AedelstenContext), intent(inout)                :: self
        real(kind=aedelsten_float_size), intent(inout)        :: output(:)
        character(len=*), intent(in)                          :: ao_symmetry
        character(len=*), intent(in), optional                :: range
        real(kind=aedelsten_float_size), intent(in), optional :: omega

        integer(kind=aedelsten_int_size) :: n, symmetry_int, index_order_int, range_wire, rc
        integer(kind=int64)              :: expected_size ! n^4 for S1 can exceed 32-bit range.
        real(kind=aedelsten_float_size)  :: omega_wire
        character(len=:), allocatable    :: symmetry_key

        if (.not. is_contiguous(output)) then
            write(error_unit, '(A)') "Aedelsten: output array must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: output array is not contiguous"
        end if

        n = to_wire_i32(self%get_n_cont(), "n")
        symmetry_key = lowercase(trim(ao_symmetry))
        if (len(symmetry_key) < len(eri_symmetry_label)) then
            symmetry_key = eri_symmetry_label // symmetry_key
        else if (symmetry_key(1:len(eri_symmetry_label)) /= eri_symmetry_label) then
            symmetry_key = eri_symmetry_label // symmetry_key
        end if
        symmetry_int = query_i32_checked(symmetry_key)

        select case (symmetry_key)
        case ("eri_ao_symmetry_s1")
            expected_size = int(n, kind=int64)**4
        case ("eri_ao_symmetry_s4")
            expected_size = int(compute_s4_size_c(self%handle), kind=int64)
        case ("eri_ao_symmetry_s8")
            expected_size = int(compute_s8_size_c(self%handle), kind=int64)
        case default
            error stop "Aedelsten: unknown ERI AO symmetry"
        end select
        if (int(size(output), kind=int64) /= expected_size) then
            write(error_unit, '(A)') "Aedelsten: output buffer's size does not match the expected size for this AO symmetry"
            error stop "Aedelsten: output buffer size mismatch"
        end if

        index_order_int = query_i32_checked("eri_index_order_column_major")
        call resolve_eri_range(range, omega, range_wire, omega_wire)

        rc = compute_eri_stored_c(self%handle, output, n, symmetry_int, index_order_int, range_wire, omega_wire)
        if (rc /= 0) error stop "Problem when evaluating ERIs. Stopping program."
    end subroutine compute_eri_stored_flat

    ! The number of values of the S4 ERI tensor of this basis.
    function compute_s4_size(self) result(sz)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: sz
        sz = int(compute_s4_size_c(self%handle), kind=external_int_size)
    end function compute_s4_size

    ! The number of values of the S8 ERI tensor of this basis.
    function compute_s8_size(self) result(sz)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: sz
        sz = int(compute_s8_size_c(self%handle), kind=external_int_size)
    end function compute_s8_size

    ! ── Direct Fock construction (no stored ERI tensor) ──────────────────

    subroutine set_eri_data(self, schwarz_threshold)
        class(AedelstenContext), intent(inout)                :: self
        real(kind=aedelsten_float_size), intent(in), optional :: schwarz_threshold

        integer(kind=aedelsten_int_size) :: rc
        real(kind=aedelsten_float_size)  :: threshold_wire

        if (present(schwarz_threshold)) then
            threshold_wire = schwarz_threshold
        else
            threshold_wire = -1.0_aedelsten_float_size ! Sentinel: tells Rust to use its own default.
        end if

        rc = set_eri_data_c(self%handle, threshold_wire)
        if (rc /= 0) error stop "Aedelsten: failed to set ERI data"
    end subroutine set_eri_data

    ! Frees the ERI data.
    subroutine free_eri_data(self)
        class(AedelstenContext), intent(inout) :: self
        call free_eri_data_c(self%handle)
    end subroutine free_eri_data

    ! j_out/k_out are the restricted-HF J and K matrices; density is the
    ! total (both-spin) density -- see fock_constructors' own doc comment
    ! for the coeff_j/coeff_k convention (coeff_j always +1, coeff_k -0.5
    ! for HF). range/omega bundled like compute_eri_stored above;
    ! pairwise_threshold is a Schwarz-bound cutoff applied during the
    ! sweep itself, separate from set_eri_data's own screening.
    subroutine compute_fock_direct(self, j_out, k_out, density, j_coeff, k_coeff, range, omega, pairwise_threshold)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: j_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: k_out(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: j_coeff
        real(kind=aedelsten_float_size), intent(in)            :: k_coeff
        character(len=*), intent(in), optional                 :: range
        real(kind=aedelsten_float_size), intent(in), optional  :: omega
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold

        integer(kind=aedelsten_int_size) :: n, range_wire, rc
        real(kind=aedelsten_float_size)  :: omega_wire, pairwise_threshold_wire

        if (.not. is_contiguous(j_out) .or. .not. is_contiguous(k_out) .or. .not. is_contiguous(density)) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a " // &
                "plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (any(shape(j_out) /= shape(k_out)) .or. any(shape(k_out) /= shape(density)) &
                .or. size(j_out,1) /= size(j_out,2)) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density must all be square and the same shape"
            error stop "Aedelsten: array has the wrong shape"
        end if
        n = size(j_out, 1, kind=aedelsten_int_size)

        call resolve_eri_range(range, omega, range_wire, omega_wire)
        if (present(pairwise_threshold)) then
            pairwise_threshold_wire = pairwise_threshold
        else
            pairwise_threshold_wire = -1.0_aedelsten_float_size
        end if

        rc = compute_fock_direct_restricted_c(self%handle, j_out, k_out, density, n, j_coeff, k_coeff, &
            range_wire, omega_wire, pairwise_threshold_wire)
        if (rc /= 0) error stop "Problem when computing the direct Fock matrix. Stopping program."
    end subroutine compute_fock_direct

    ! compute_fock_direct for a density of any symmetry (a transition density, the spin-flip density of a stability
    ! analysis): j_out = j_coeff J(density), from the density's symmetric part, and k_out = k_coeff K(density) with
    ! K(a,c) = sum_bd density(b,d) (ab|cd), which is not symmetric for a non-symmetric density and is not symmetrized.
    ! A symmetric density gives compute_fock_direct's J and K. Arguments as for compute_fock_direct.
    subroutine compute_fock_direct_general(self, j_out, k_out, density, j_coeff, k_coeff, range, omega, pairwise_threshold)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: j_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: k_out(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: j_coeff
        real(kind=aedelsten_float_size), intent(in)            :: k_coeff
        character(len=*), intent(in), optional                 :: range
        real(kind=aedelsten_float_size), intent(in), optional  :: omega
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold

        integer(kind=aedelsten_int_size) :: n, range_wire, rc
        real(kind=aedelsten_float_size)  :: omega_wire, pairwise_threshold_wire

        if (.not. is_contiguous(j_out) .or. .not. is_contiguous(k_out) .or. .not. is_contiguous(density)) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a " // &
                "plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (any(shape(j_out) /= shape(k_out)) .or. any(shape(k_out) /= shape(density)) &
                .or. size(j_out,1) /= size(j_out,2)) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density must all be square and the same shape"
            error stop "Aedelsten: array has the wrong shape"
        end if
        n = size(j_out, 1, kind=aedelsten_int_size)

        call resolve_eri_range(range, omega, range_wire, omega_wire)
        if (present(pairwise_threshold)) then
            pairwise_threshold_wire = pairwise_threshold
        else
            pairwise_threshold_wire = -1.0_aedelsten_float_size
        end if

        rc = compute_fock_direct_general_c(self%handle, j_out, k_out, density, n, j_coeff, k_coeff, &
            range_wire, omega_wire, pairwise_threshold_wire)
        if (rc /= 0) error stop "Problem when computing the direct Fock matrices of a general density. Stopping program."
    end subroutine compute_fock_direct_general

    ! compute_fock_direct_general for several densities in one direct walk: density(:,:,m) the m-th density, and
    ! j_out(:,:,m), k_out(:,:,m) its J and K as compute_fock_direct_general gives them (to within the screening, which
    ! keeps an integral batch any density needs). Arguments otherwise as for compute_fock_direct_general.
    subroutine compute_fock_direct_general_multi(self, j_out, k_out, density, j_coeff, k_coeff, range, omega, &
            pairwise_threshold)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: j_out(:,:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: k_out(:,:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density(:,:,:)
        real(kind=aedelsten_float_size), intent(in)            :: j_coeff
        real(kind=aedelsten_float_size), intent(in)            :: k_coeff
        character(len=*), intent(in), optional                 :: range
        real(kind=aedelsten_float_size), intent(in), optional  :: omega
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold

        integer(kind=aedelsten_int_size) :: n, n_mat, range_wire, rc
        real(kind=aedelsten_float_size)  :: omega_wire, pairwise_threshold_wire

        if (.not. is_contiguous(j_out) .or. .not. is_contiguous(k_out) .or. .not. is_contiguous(density)) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a " // &
                "plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (any(shape(j_out) /= shape(k_out)) .or. any(shape(k_out) /= shape(density)) &
                .or. size(j_out,1) /= size(j_out,2) .or. size(j_out,3) < 1) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density must all be n x n x n_mat, n_mat >= 1"
            error stop "Aedelsten: array has the wrong shape"
        end if
        n = size(j_out, 1, kind=aedelsten_int_size)
        n_mat = size(j_out, 3, kind=aedelsten_int_size)

        call resolve_eri_range(range, omega, range_wire, omega_wire)
        if (present(pairwise_threshold)) then
            pairwise_threshold_wire = pairwise_threshold
        else
            pairwise_threshold_wire = -1.0_aedelsten_float_size
        end if

        rc = compute_fock_direct_general_multi_c(self%handle, j_out, k_out, density, n, n_mat, j_coeff, k_coeff, &
            range_wire, omega_wire, pairwise_threshold_wire)
        if (rc /= 0) error stop "Problem when computing the direct Fock matrices of general densities. Stopping program."
    end subroutine compute_fock_direct_general_multi

    ! The integrals of a state interaction between two runs with different orbitals, in one direct walk (full range):
    ! j_out, k_out as in compute_fock_direct_general, and m_out(v + (w-1) n_active, mu nu) = M(vw, mu nu) =
    ! sum_{lambda sigma} P(lambda sigma, vw) (mu nu|lambda sigma), P(lambda sigma, vw) =
    ! (c_bra(lambda,v) c_ket(sigma,w) + c_bra(sigma,v) c_ket(lambda,w))/2, for every pair (v, w), packed over the
    ! function pairs mu >= nu (column (nu-1) n - (nu-1)(nu-2)/2 + mu - nu + 1 in 1-based indices). The caller makes
    ! T(u, vw, mu) = sum_nu c_ket(nu,u) M(vw, mu nu) with transform_pair_coulomb_one_index and finishes
    ! (tu|vw) = sum_mu c_bra(mu,t) T(u vw, mu) with one product (t, v the bra's orbitals, u, w the ket's).
    ! c_bra, c_ket are (n, n_active); m_out is (n_active**2, n (n + 1)/2). Every output OVERWRITTEN. active_threshold:
    ! a batch whose Schwarz bound times its largest active coefficients is below it adds nothing to M; absent = every
    ! batch. pairwise_threshold as for compute_fock_direct.
    subroutine compute_fock_direct_mixed_active_space(self, j_out, k_out, m_out, density, c_bra, c_ket, j_coeff, &
            k_coeff, active_threshold, pairwise_threshold)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: j_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: k_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: m_out(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_bra(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_ket(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: j_coeff
        real(kind=aedelsten_float_size), intent(in)            :: k_coeff
        real(kind=aedelsten_float_size), intent(in), optional  :: active_threshold
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold

        integer(kind=aedelsten_int_size) :: n, n_active, rc
        real(kind=aedelsten_float_size)  :: active_threshold_wire, pairwise_threshold_wire

        if (.not. is_contiguous(j_out) .or. .not. is_contiguous(k_out) .or. .not. is_contiguous(m_out) &
                .or. .not. is_contiguous(density) .or. .not. is_contiguous(c_bra) .or. .not. is_contiguous(c_ket)) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/m_out/density/c_bra/c_ket must be contiguous (got a " // &
                "strided/non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a " // &
                "plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        n = size(j_out, 1, kind=aedelsten_int_size)
        n_active = size(c_ket, 2, kind=aedelsten_int_size)
        if (any(shape(j_out) /= shape(k_out)) .or. any(shape(k_out) /= shape(density)) .or. size(j_out,2) /= n &
                .or. any(shape(c_bra) /= shape(c_ket)) .or. size(c_ket,1) /= n &
                .or. any(shape(m_out) /= [n_active*n_active, n*(n+1)/2])) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density must be (n,n), c_bra/c_ket (n,n_active) and " // &
                "m_out (n_active**2,n*(n+1)/2)"
            error stop "Aedelsten: array has the wrong shape"
        end if

        active_threshold_wire = -1.0_aedelsten_float_size
        if (present(active_threshold)) active_threshold_wire = active_threshold
        pairwise_threshold_wire = -1.0_aedelsten_float_size
        if (present(pairwise_threshold)) pairwise_threshold_wire = pairwise_threshold

        rc = compute_fock_direct_mixed_active_space_c(self%handle, j_out, k_out, m_out, density, c_bra, c_ket, n, &
            n_active, j_coeff, k_coeff, active_threshold_wire, pairwise_threshold_wire)
        if (rc /= 0) error stop "Problem when computing the mixed-basis active-space integrals. Stopping program."
    end subroutine compute_fock_direct_mixed_active_space

    ! gradient(3, n_centres) is OVERWRITTEN with dE_2e/dX_c for every gradient centre X (get_n_gradient_centres()),
    ! the energy being E_2e = 1/2 sum_t term_coeff(t) sum_abcd W_t (ab|cd) with, for term t pairing
    ! L = densities(:,:,term_left(t)) and R = densities(:,:,term_right(t)) (1-based indices):
    !   term_kind(t) = "coulomb":  W = L(a,b) R(c,d)        term_kind(t) = "exchange":  W = L(a,c) R(b,d)
    ! Restricted HF: densities(:,:,1) = P, terms ("coulomb", 1, 1, 1.0) and ("exchange", 1, 1, -0.5). Unrestricted:
    ! densities = [P, P_a, P_b], ("coulomb", 1, 1, 1.0), ("exchange", 2, 2, -1.0), ("exchange", 3, 3, -1.0). Only the
    ! integral derivatives (the Pulay term comes with the overlap gradient); one range operator per call; `active` as in
    ! compute_one_electron_gradient. Needs set_eri_data first.
    subroutine compute_two_electron_gradient(self, gradient, densities, term_kind, term_left, term_right, term_coeff, &
            range, omega, pairwise_threshold, active)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: gradient(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: densities(:,:,:)
        character(len=*), intent(in)                           :: term_kind(:)
        integer(kind=external_int_size), intent(in)            :: term_left(:)
        integer(kind=external_int_size), intent(in)            :: term_right(:)
        real(kind=aedelsten_float_size), intent(in)            :: term_coeff(:)
        character(len=*), intent(in), optional                 :: range
        real(kind=aedelsten_float_size), intent(in), optional  :: omega
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold
        logical, intent(in), optional                          :: active(:)

        integer(kind=aedelsten_int_size) :: n, n_centres, n_densities, n_terms, range_wire, rc
        real(kind=aedelsten_float_size)  :: omega_wire, pairwise_threshold_wire
        integer(kind=aedelsten_int_size), allocatable :: kind_wire(:), left_wire(:), right_wire(:)
        integer(kind=c_int), allocatable, target :: active_flags(:)
        type(c_ptr) :: active_ptr

        call two_electron_gradient_wires(self, gradient, densities, term_kind, term_left, term_right, term_coeff, &
            n, n_centres, n_densities, n_terms, kind_wire, left_wire, right_wire)

        call resolve_eri_range(range, omega, range_wire, omega_wire)
        if (present(pairwise_threshold)) then
            pairwise_threshold_wire = pairwise_threshold
        else
            pairwise_threshold_wire = -1.0_aedelsten_float_size
        end if
        active_ptr = c_null_ptr
        if (present(active)) then
            if (size(active, kind=aedelsten_int_size) /= n_centres) then
                error stop "Aedelsten: active must have one flag per gradient centre"
            end if
            allocate(active_flags(n_centres))
            active_flags = merge(1_c_int, 0_c_int, active)
            active_ptr = c_loc(active_flags)
        end if

        rc = compute_two_electron_gradient_c(self%handle, gradient, n_centres, densities, n, n_densities, n_terms, &
            kind_wire, left_wire, right_wire, term_coeff, range_wire, omega_wire, pairwise_threshold_wire, active_ptr)
        if (rc /= 0) error stop "Problem when evaluating the two-electron gradient. Stopping program."
    end subroutine compute_two_electron_gradient

    ! The two-electron gradients' arrays checked and their terms as the C wire arrays (0-based density indices).
    subroutine two_electron_gradient_wires(self, gradient, densities, term_kind, term_left, term_right, term_coeff, &
            n, n_centres, n_densities, n_terms, kind_wire, left_wire, right_wire)
        class(AedelstenContext), intent(in)                    :: self
        real(kind=aedelsten_float_size), intent(in)            :: gradient(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: densities(:,:,:)
        character(len=*), intent(in)                           :: term_kind(:)
        integer(kind=external_int_size), intent(in)            :: term_left(:)
        integer(kind=external_int_size), intent(in)            :: term_right(:)
        real(kind=aedelsten_float_size), intent(in)            :: term_coeff(:)
        integer(kind=aedelsten_int_size), intent(out)          :: n, n_centres, n_densities, n_terms
        integer(kind=aedelsten_int_size), allocatable, intent(out) :: kind_wire(:), left_wire(:), right_wire(:)
        integer(kind=aedelsten_int_size) :: t

        if (.not. is_contiguous(gradient) .or. .not. is_contiguous(densities)) then
            write(error_unit, '(A)') "Aedelsten: gradient/densities must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a " // &
                "plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (size(gradient, 1) /= 3) then
            error stop "Aedelsten: the gradient's first dimension must be exactly 3 (x, y, z)"
        end if
        if (size(gradient, 2, kind=aedelsten_int_size) /= get_n_gradient_centres_c(self%handle)) then
            error stop "Aedelsten: the gradient's second dimension must be get_n_gradient_centres()"
        end if
        if (size(densities, 1) /= size(densities, 2)) then
            error stop "Aedelsten: every density must be square"
        end if
        n_terms = size(term_kind, kind=aedelsten_int_size)
        if (size(term_left, kind=aedelsten_int_size) /= n_terms .or. size(term_right, kind=aedelsten_int_size) /= n_terms &
                .or. size(term_coeff, kind=aedelsten_int_size) /= n_terms) then
            error stop "Aedelsten: term_kind, term_left, term_right and term_coeff must have the same length"
        end if
        n = size(densities, 1, kind=aedelsten_int_size)
        n_densities = size(densities, 3, kind=aedelsten_int_size)
        n_centres = size(gradient, 2, kind=aedelsten_int_size)

        allocate(kind_wire(n_terms), left_wire(n_terms), right_wire(n_terms))
        do t = 1, n_terms
            select case (trim(lowercase(term_kind(t))))
            case ("coulomb")
                kind_wire(t) = 0_aedelsten_int_size
            case ("exchange")
                kind_wire(t) = 1_aedelsten_int_size
            case default
                write(error_unit, '(A)') "Aedelsten: unknown two-electron term kind '" // trim(term_kind(t)) // &
                    "' (coulomb or exchange)"
                error stop "Aedelsten: unknown two-electron term kind"
            end select
            if (term_left(t) < 1 .or. term_left(t) > n_densities .or. term_right(t) < 1 .or. term_right(t) > n_densities) then
                error stop "Aedelsten: a term's density index is outside 1..size(densities, 3)"
            end if
            left_wire(t) = int(term_left(t) - 1, kind=aedelsten_int_size)
            right_wire(t) = int(term_right(t) - 1, kind=aedelsten_int_size)
        end do
    end subroutine two_electron_gradient_wires

    ! Unrestricted direct J and K: j_out = j_coeff J(density_total), k_alpha_out and k_beta_out = k_coeff K of each
    ! spin's density; density_total is the caller's sum of the two. Other arguments as for compute_fock_direct.
    subroutine compute_fock_direct_unrestricted(&
        self, j_out, k_alpha_out, k_beta_out, density_alpha, density_beta, density_total, &
        j_coeff, k_coeff, range, omega, pairwise_threshold &
    )
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: j_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: k_alpha_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: k_beta_out(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density_alpha(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density_beta(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density_total(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: j_coeff
        real(kind=aedelsten_float_size), intent(in)            :: k_coeff
        character(len=*), intent(in), optional                 :: range
        real(kind=aedelsten_float_size), intent(in), optional  :: omega
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold

        integer(kind=aedelsten_int_size) :: n, range_wire, rc
        real(kind=aedelsten_float_size)  :: omega_wire, pairwise_threshold_wire

        if (.not. is_contiguous(j_out) .or. .not. is_contiguous(k_alpha_out) .or. &
            .not. is_contiguous(k_beta_out) .or. .not. is_contiguous(density_alpha) .or. &
            .not. is_contiguous(density_beta) .or. .not. is_contiguous(density_total)) then
            write(error_unit, '(A)') "Aedelsten: all arrays must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (any(shape(k_alpha_out) /= shape(k_beta_out)) .or. any(shape(k_alpha_out) /= shape(j_out)) &
                .or. any(shape(density_alpha) /= shape(density_beta)) .or. any(shape(density_alpha) /= shape(density_total)) &
                .or. size(j_out,1) /= size(j_out,2)) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_alpha_out/k_beta_out and density_alpha/density_beta/" // &
                "density_total must each be square and pairwise the same shape"
            error stop "Aedelsten: array has the wrong shape"
        end if
        if (size(j_out,1) /= size(density_alpha,1)) then
            write(error_unit, '(A)') "Aedelsten: j_out and density_alpha must share the same n"
            error stop "Aedelsten: arrays disagree on n"
        end if
        n = size(j_out, 1, kind=aedelsten_int_size)

        call resolve_eri_range(range, omega, range_wire, omega_wire)
        if (present(pairwise_threshold)) then
            pairwise_threshold_wire = pairwise_threshold
        else
            pairwise_threshold_wire = -1.0_aedelsten_float_size
        end if

        rc = compute_fock_direct_unrestricted_c(&
            self%handle, j_out, k_alpha_out, k_beta_out, density_alpha, density_beta, density_total, n, &
            j_coeff, k_coeff, range_wire, omega_wire, pairwise_threshold_wire)
        if (rc /= 0) error stop "Problem when computing the unrestricted direct Fock matrices. Stopping program."
    end subroutine compute_fock_direct_unrestricted

    ! The active-space integrals of a CASSCF macro-iteration in one direct walk (full range): the inactive and active
    ! J/K, each against its own coefficients as in compute_fock_direct, and m_out(vw, mu nu) = M(vw, mu nu) =
    ! sum_{lambda sigma} P(lambda sigma, vw) (mu nu|lambda sigma), P(lambda sigma, vw) =
    ! (c_active(lambda,v) c_active(sigma,w) + c_active(sigma,v) c_active(lambda,w))/2, for the pairs v >= w at
    ! vw = v (v - 1)/2 + w in 1-based indices (v (v + 1)/2 + w + 1 in 0-based ones), packed over the function pairs
    ! mu >= nu (column (nu-1) n - (nu-1)(nu-2)/2 + mu - nu + 1 in 1-based indices). The caller makes
    ! T(u, vw, mu) = sum_nu c_active(nu,u) M(vw, mu nu) with transform_pair_coulomb_one_index and finishes
    ! (pu|vw) = sum_mu T(u vw, mu) C(mu, p) with one product. c_active is (n, n_active); m_out is
    ! (n_active (n_active + 1)/2, n (n + 1)/2). Every output OVERWRITTEN. active_threshold: a batch whose Schwarz bound
    ! times its largest active coefficients is below it adds nothing to M; absent = every batch.
    subroutine compute_fock_direct_active_space(self, j_inactive_out, k_inactive_out, j_active_out, k_active_out, &
            m_out, density_inactive, density_active, c_active, coeff_j_inactive, coeff_k_inactive, coeff_j_active, &
            coeff_k_active, active_threshold, pairwise_threshold)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: j_inactive_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: k_inactive_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: j_active_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: k_active_out(:,:)
        real(kind=aedelsten_float_size), intent(inout)         :: m_out(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density_inactive(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: density_active(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_active(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: coeff_j_inactive
        real(kind=aedelsten_float_size), intent(in)            :: coeff_k_inactive
        real(kind=aedelsten_float_size), intent(in)            :: coeff_j_active
        real(kind=aedelsten_float_size), intent(in)            :: coeff_k_active
        real(kind=aedelsten_float_size), intent(in), optional  :: active_threshold
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold

        integer(kind=aedelsten_int_size) :: n, n_active, rc
        real(kind=aedelsten_float_size)  :: active_threshold_wire, pairwise_threshold_wire

        if (.not. is_contiguous(j_inactive_out) .or. .not. is_contiguous(k_inactive_out) .or. &
            .not. is_contiguous(j_active_out) .or. .not. is_contiguous(k_active_out) .or. .not. is_contiguous(m_out) &
            .or. .not. is_contiguous(density_inactive) .or. .not. is_contiguous(density_active) &
            .or. .not. is_contiguous(c_active)) then
            write(error_unit, '(A)') "Aedelsten: all arrays must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        n = size(j_inactive_out, 1, kind=aedelsten_int_size)
        n_active = size(c_active, 2, kind=aedelsten_int_size)
        if (size(j_inactive_out, 2) /= n .or. any(shape(k_inactive_out) /= shape(j_inactive_out)) &
                .or. any(shape(j_active_out) /= shape(j_inactive_out)) .or. any(shape(k_active_out) /= shape(j_inactive_out)) &
                .or. any(shape(density_inactive) /= shape(j_inactive_out)) &
                .or. any(shape(density_active) /= shape(j_inactive_out)) .or. size(c_active, 1) /= n &
                .or. any(shape(m_out) /= [n_active*(n_active+1)/2, n*(n+1)/2])) then
            write(error_unit, '(A)') "Aedelsten: J/K and densities must be (n,n), c_active (n,n_active) and m_out " // &
                "(n_active*(n_active+1)/2,n*(n+1)/2)"
            error stop "Aedelsten: array has the wrong shape"
        end if

        active_threshold_wire = -1.0_aedelsten_float_size
        if (present(active_threshold)) active_threshold_wire = active_threshold
        pairwise_threshold_wire = -1.0_aedelsten_float_size
        if (present(pairwise_threshold)) pairwise_threshold_wire = pairwise_threshold

        rc = compute_fock_direct_active_space_c(self%handle, j_inactive_out, k_inactive_out, j_active_out, &
            k_active_out, m_out, density_inactive, density_active, c_active, n, n_active, coeff_j_inactive, &
            coeff_k_inactive, coeff_j_active, coeff_k_active, active_threshold_wire, pairwise_threshold_wire)
        if (rc /= 0) error stop "Problem when computing the active-space integrals. Stopping program."
    end subroutine compute_fock_direct_active_space

    ! One index of the active walks' M transformed: t_out(u, vw, mu) = T(u, vw, mu) = sum_nu c(nu,u) M(vw, mu nu).
    ! m is (n_pairs, n (n + 1)/2) as compute_fock_direct_active_space (or the mixed walk) writes it; c is
    ! (n, n_columns); t_out is (n_columns, n_pairs, n), OVERWRITTEN. Rank-local.
    subroutine transform_pair_coulomb_one_index(self, t_out, m, c)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: t_out(:,:,:)
        real(kind=aedelsten_float_size), intent(in)            :: m(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c(:,:)

        integer(kind=aedelsten_int_size) :: n, n_pairs, n_columns, rc

        if (.not. is_contiguous(t_out) .or. .not. is_contiguous(m) .or. .not. is_contiguous(c)) then
            write(error_unit, '(A)') "Aedelsten: t_out/m/c must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        n = size(c, 1, kind=aedelsten_int_size)
        n_columns = size(c, 2, kind=aedelsten_int_size)
        n_pairs = size(m, 1, kind=aedelsten_int_size)
        if (size(m, 2) /= n*(n+1)/2 .or. any(shape(t_out) /= [n_columns, n_pairs, n])) then
            write(error_unit, '(A)') "Aedelsten: c must be (n,n_columns), m (n_pairs,n*(n+1)/2) and t_out " // &
                "(n_columns,n_pairs,n)"
            error stop "Aedelsten: array has the wrong shape"
        end if

        rc = transform_pair_coulomb_one_index_c(self%handle, t_out, m, c, n, n_pairs, n_columns)
        if (rc /= 0) error stop "Problem when transforming the pair densities' Coulomb matrices. Stopping program."
    end subroutine transform_pair_coulomb_one_index

    ! The half-transformed ERIs from one direct walk (full range, no J/K): x_out(p,nu,r,sigma) =
    ! sum_{mu lambda} c_left(mu,p) c_right(lambda,r) (mu nu|lambda sigma), x_out (m_left,n,m_right,n),
    ! OVERWRITTEN; c_left (n,m_left), c_right (n,m_right), e.g. column blocks of the orbital matrix.
    ! The same array for both takes the symmetric path, x_out(p,nu,r,sigma) = x_out(r,sigma,p,nu), at
    ! half the work. threshold: a batch whose Schwarz bound times the largest |c_left| on one of its
    ! shell pairs times the largest |c_right| on the other is below it is skipped (absent = every
    ! batch). partition_over_ranks (default .true.): under MPI the call is collective and every rank
    ! gets the whole x_out; .false. computes it on this rank alone. The scratch holds the walk's row
    ! buffers besides its usual scratch. Requires set_eri_data first.
    subroutine compute_eri_half_transformed(self, x_out, c_left, c_right, threshold, pairwise_threshold, partition_over_ranks)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: x_out(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_left(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_right(:,:)
        real(kind=aedelsten_float_size), intent(in), optional  :: threshold
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold
        logical, intent(in), optional                          :: partition_over_ranks

        integer(kind=aedelsten_int_size) :: n, m_left, m_right, partition_wire, rc
        real(kind=aedelsten_float_size)  :: threshold_wire, pairwise_threshold_wire

        if (.not. is_contiguous(x_out) .or. .not. is_contiguous(c_left) .or. .not. is_contiguous(c_right)) then
            write(error_unit, '(A)') "Aedelsten: all arrays must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        n = get_n_cont_c(self%handle)
        m_left = size(c_left, 2, kind=aedelsten_int_size)
        m_right = size(c_right, 2, kind=aedelsten_int_size)
        if (size(c_left, 1) /= n .or. size(c_right, 1) /= n .or. any(shape(x_out) /= [m_left, n, m_right, n])) then
            write(error_unit, '(A)') "Aedelsten: c_left must be (n,m_left), c_right (n,m_right) and x_out (m_left,n,m_right,n)"
            error stop "Aedelsten: array has the wrong shape"
        end if

        threshold_wire = -1.0_aedelsten_float_size
        if (present(threshold)) threshold_wire = threshold
        pairwise_threshold_wire = -1.0_aedelsten_float_size
        if (present(pairwise_threshold)) pairwise_threshold_wire = pairwise_threshold
        partition_wire = 1
        if (present(partition_over_ranks)) then
            if (.not. partition_over_ranks) partition_wire = 0
        end if

        rc = compute_eri_half_transformed_c(self%handle, x_out, c_left, m_left, c_right, m_right, n, threshold_wire, &
                                            pairwise_threshold_wire, partition_wire)
        if (rc /= 0) error stop "Aedelsten: failed to compute the half-transformed integrals"
    end subroutine compute_eri_half_transformed

    ! compute_eri_half_transformed by the ranks of the context's communicator together, into one x_out in their node's
    ! shared memory: x_out points into an AedelstenSharedArray of the same communicator (x_out(1:m_left, 1:n, 1:m_right,
    ! 1:n) => shared%data), the ranks share one node, and every rank has the whole X when it returns. Collective.
    subroutine compute_eri_half_transformed_on_node(self, x_out, c_left, c_right, threshold, pairwise_threshold)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: x_out(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_left(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_right(:,:)
        real(kind=aedelsten_float_size), intent(in), optional  :: threshold
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold

        integer(kind=aedelsten_int_size) :: n, m_left, m_right, rc
        real(kind=aedelsten_float_size)  :: threshold_wire, pairwise_threshold_wire

        if (.not. is_contiguous(x_out) .or. .not. is_contiguous(c_left) .or. .not. is_contiguous(c_right)) then
            write(error_unit, '(A)') "Aedelsten: all arrays must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        n = get_n_cont_c(self%handle)
        m_left = size(c_left, 2, kind=aedelsten_int_size)
        m_right = size(c_right, 2, kind=aedelsten_int_size)
        if (size(c_left, 1) /= n .or. size(c_right, 1) /= n .or. any(shape(x_out) /= [m_left, n, m_right, n])) then
            write(error_unit, '(A)') "Aedelsten: c_left must be (n,m_left), c_right (n,m_right) and x_out (m_left,n,m_right,n)"
            error stop "Aedelsten: array has the wrong shape"
        end if
        threshold_wire = -1.0_aedelsten_float_size
        if (present(threshold)) threshold_wire = threshold
        pairwise_threshold_wire = -1.0_aedelsten_float_size
        if (present(pairwise_threshold)) pairwise_threshold_wire = pairwise_threshold
        rc = compute_eri_half_transformed_on_node_c(self%handle, x_out, c_left, m_left, c_right, m_right, n, &
                                                    threshold_wire, pairwise_threshold_wire)
        if (rc /= 0) error stop "Aedelsten: failed to compute the half-transformed integrals on the node"
    end subroutine compute_eri_half_transformed_on_node

    ! compute_eri_half_transformed for a batch of sigma: x_out(p, nu, r, s) = X(p, nu, r, sigma_start + s - 1) for the
    ! sigma in sigma_start .. sigma_end - 1 only (0-based AOs as get_shells gives them, a run of whole shells: each end
    ! a shell's first AO or n; not empty), x_out (m_left, n, m_right, sigma_end - sigma_start). The walk computes only
    ! the quartets with a shell in the batch, so batches that cover the basis compute each quartet at most four times in
    ! all. The other arguments as compute_eri_half_transformed; the scratch beyond the walk's own:
    ! half_transform_scratch_bytes.
    subroutine compute_eri_half_transformed_sigma(self, x_out, c_left, c_right, sigma_start, sigma_end, threshold, &
            pairwise_threshold, partition_over_ranks)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: x_out(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_left(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_right(:,:)
        integer(kind=aedelsten_int_size), intent(in)           :: sigma_start
        integer(kind=aedelsten_int_size), intent(in)           :: sigma_end
        real(kind=aedelsten_float_size), intent(in), optional  :: threshold
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold
        logical, intent(in), optional                          :: partition_over_ranks

        integer(kind=aedelsten_int_size) :: n, m_left, m_right, partition_wire, rc
        real(kind=aedelsten_float_size)  :: threshold_wire, pairwise_threshold_wire

        call check_sigma_operands(self, x_out, c_left, c_right, sigma_start, sigma_end, n, m_left, m_right)
        threshold_wire = -1.0_aedelsten_float_size
        if (present(threshold)) threshold_wire = threshold
        pairwise_threshold_wire = -1.0_aedelsten_float_size
        if (present(pairwise_threshold)) pairwise_threshold_wire = pairwise_threshold
        partition_wire = 1
        if (present(partition_over_ranks)) then
            if (.not. partition_over_ranks) partition_wire = 0
        end if
        rc = compute_eri_half_transformed_sigma_c(self%handle, x_out, c_left, m_left, c_right, m_right, n, sigma_start, &
                                                  sigma_end, threshold_wire, pairwise_threshold_wire, partition_wire)
        if (rc /= 0) error stop "Aedelsten: failed to compute the half-transformed integrals of a batch of sigma"
    end subroutine compute_eri_half_transformed_sigma

    ! compute_eri_half_transformed_sigma by the ranks of the context's communicator together, into one x_out in their
    ! node's shared memory, as compute_eri_half_transformed_on_node: x_out (m_left, n, m_right, sigma_end - sigma_start)
    ! points into an AedelstenSharedArray of the same communicator. Collective.
    subroutine compute_eri_half_transformed_sigma_on_node(self, x_out, c_left, c_right, sigma_start, sigma_end, &
            threshold, pairwise_threshold)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: x_out(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_left(:,:)
        real(kind=aedelsten_float_size), intent(in)            :: c_right(:,:)
        integer(kind=aedelsten_int_size), intent(in)           :: sigma_start
        integer(kind=aedelsten_int_size), intent(in)           :: sigma_end
        real(kind=aedelsten_float_size), intent(in), optional  :: threshold
        real(kind=aedelsten_float_size), intent(in), optional  :: pairwise_threshold

        integer(kind=aedelsten_int_size) :: n, m_left, m_right, rc
        real(kind=aedelsten_float_size)  :: threshold_wire, pairwise_threshold_wire

        call check_sigma_operands(self, x_out, c_left, c_right, sigma_start, sigma_end, n, m_left, m_right)
        threshold_wire = -1.0_aedelsten_float_size
        if (present(threshold)) threshold_wire = threshold
        pairwise_threshold_wire = -1.0_aedelsten_float_size
        if (present(pairwise_threshold)) pairwise_threshold_wire = pairwise_threshold
        rc = compute_eri_half_transformed_sigma_on_node_c(self%handle, x_out, c_left, m_left, c_right, m_right, n, &
                                                          sigma_start, sigma_end, threshold_wire, pairwise_threshold_wire)
        if (rc /= 0) error stop "Aedelsten: failed to compute the half-transformed integrals of a batch of sigma on the node"
    end subroutine compute_eri_half_transformed_sigma_on_node

    ! The checks of a batch of sigma's arrays (contiguous, shaped), and n, m_left, m_right from them.
    subroutine check_sigma_operands(self, x_out, c_left, c_right, sigma_start, sigma_end, n, m_left, m_right)
        class(AedelstenContext), intent(in)             :: self
        real(kind=aedelsten_float_size), intent(in)     :: x_out(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)     :: c_left(:,:)
        real(kind=aedelsten_float_size), intent(in)     :: c_right(:,:)
        integer(kind=aedelsten_int_size), intent(in)    :: sigma_start
        integer(kind=aedelsten_int_size), intent(in)    :: sigma_end
        integer(kind=aedelsten_int_size), intent(out)   :: n, m_left, m_right

        if (.not. is_contiguous(x_out) .or. .not. is_contiguous(c_left) .or. .not. is_contiguous(c_right)) then
            write(error_unit, '(A)') "Aedelsten: all arrays must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        n = get_n_cont_c(self%handle)
        m_left = size(c_left, 2, kind=aedelsten_int_size)
        m_right = size(c_right, 2, kind=aedelsten_int_size)
        if (sigma_start < 0 .or. sigma_end <= sigma_start .or. sigma_end > n) then
            write(error_unit, '(A)') "Aedelsten: sigma_start .. sigma_end - 1 must be a run of the n AOs (0-based)"
            error stop "Aedelsten: sigma out of range"
        end if
        if (size(c_left, 1) /= n .or. size(c_right, 1) /= n .or. &
                any(shape(x_out) /= [m_left, n, m_right, sigma_end - sigma_start])) then
            write(error_unit, '(A)') "Aedelsten: c_left must be (n,m_left), c_right (n,m_right) and x_out " // &
                "(m_left,n,m_right,sigma_end-sigma_start)"
            error stop "Aedelsten: array has the wrong shape"
        end if
    end subroutine check_sigma_operands

    ! The scratch a half transform takes besides the walk's own, in bytes (set_scratch holds both): m_left
    ! and m_right orbitals, one_basis (the same array on both sides) or two, n_sigma AOs of sigma (n for
    ! compute_eri_half_transformed), node_ranks ranks sharing X (the _on_node forms; 1 for the others).
    function half_transform_scratch_bytes(self, m_left, m_right, one_basis, n_sigma, node_ranks) result(bytes)
        class(AedelstenContext), intent(in)          :: self
        integer(kind=aedelsten_int_size), intent(in) :: m_left, m_right, n_sigma, node_ranks
        logical, intent(in)                          :: one_basis
        integer(kind=c_int64_t)                      :: bytes
        integer(kind=aedelsten_int_size)             :: one_basis_wire

        one_basis_wire = 0
        if (one_basis) one_basis_wire = 1
        bytes = half_transform_scratch_bytes_c(self%handle, m_left, m_right, one_basis_wire, n_sigma, node_ranks)
        if (bytes < 0) error stop "Aedelsten: half_transform_scratch_bytes: bad arguments"
    end function half_transform_scratch_bytes

    ! The scratch in bytes that every operation fits in, from the input alone (before set_shell_pairs; size
    ! set_scratch with it): operations(5, n), a column per operation (scratch_eri_data, ...; see their
    ! declaration). primitive: on the primitive blocks.
    function scratch_bytes(self, operations, primitive) result(bytes)
        class(AedelstenContext), intent(in)          :: self
        integer(kind=aedelsten_int_size), intent(in) :: operations(:, :)
        logical, intent(in)                          :: primitive
        integer(kind=c_int64_t)                      :: bytes

        if (size(operations, 1) /= 5) error stop "Aedelsten: scratch_bytes: operations must be (5, n)"
        bytes = scratch_bytes_c(self%handle, operations, size(operations, 2, kind=aedelsten_int_size), &
            merge(1_aedelsten_int_size, 0_aedelsten_int_size, primitive))
        if (bytes < 0) error stop "Aedelsten: scratch_bytes: bad operations"
    end function scratch_bytes

    ! Fields per operation of scratch_breakdown: allocate breakdown(scratch_breakdown_field_count(), n).
    function scratch_breakdown_field_count() result(n)
        integer(kind=external_int_size) :: n
        n = int(scratch_breakdown_field_count_c(), kind=external_int_size)
    end function scratch_breakdown_field_count

    ! scratch_bytes per operation, a column of breakdown(18, n) per column of operations(5, n): bytes, buffers, tables,
    ! row, item_bytes (bytes their sum), item (1 ERI data entry, 2 gradient pair data, 3 derivative bound, 4 quartet, 5
    ! AMFI atom kind), then l, primitives and contracted of the item's four shells (-1 for a pair's last two).
    subroutine scratch_breakdown(self, operations, primitive, breakdown)
        class(AedelstenContext), intent(in)          :: self
        integer(kind=aedelsten_int_size), intent(in) :: operations(:, :)
        logical, intent(in)                          :: primitive
        integer(kind=c_int64_t), intent(inout)       :: breakdown(:, :)
        integer(kind=aedelsten_int_size)             :: rc

        if (size(operations, 1) /= 5 .or. size(breakdown, 1, kind=external_int_size) /= scratch_breakdown_field_count() &
                .or. size(breakdown, 2) /= size(operations, 2)) &
            error stop "Aedelsten: scratch_breakdown: operations must be (5, n), breakdown (scratch_breakdown_field_count(), n)"
        rc = scratch_breakdown_c(self%handle, operations, size(operations, 2, kind=aedelsten_int_size), &
            merge(1_aedelsten_int_size, 0_aedelsten_int_size, primitive), breakdown, size(breakdown, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: scratch_breakdown failed"
    end subroutine scratch_breakdown

    ! ── J/K from an in-memory dense (S1) ERI tensor ──────────────────────
    ! Free functions, not type-bound: pure functions of their own
    ! arguments (a caller-supplied ERI tensor + density matrices), no
    ! context state needed at all.

    ! Single pass over eri_dense for both J and K -- prefer this over
    ! separate compute_j_matrix_stored/compute_k_matrix_stored calls
    ! when both are wanted, which would otherwise sweep eri_dense twice.
    subroutine compute_j_and_k_matrix_stored(j_out, k_out, density, eri_dense, j_coeff, k_coeff)
        real(kind=aedelsten_float_size), intent(inout) :: j_out(:,:)
        real(kind=aedelsten_float_size), intent(inout) :: k_out(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: eri_dense(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)    :: j_coeff
        real(kind=aedelsten_float_size), intent(in)    :: k_coeff

        integer(kind=aedelsten_int_size) :: n, rc

        if (.not. is_contiguous(j_out) .or. .not. is_contiguous(k_out) .or. &
            .not. is_contiguous(density) .or. .not. is_contiguous(eri_dense)) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density/eri_dense must be contiguous (got a " // &
                "strided/non-contiguous section) -- Aedelsten never silently copies; pass a contiguous " // &
                "slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (any(shape(j_out) /= shape(k_out)) .or. size(j_out,1) /= size(j_out,2) &
                .or. size(density,1) /= size(density,2) .or. any(shape(eri_dense) /= size(eri_dense,1))) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_out/density must be square (and j_out, k_out the " // &
                "same shape), eri_dense must be square in all four dimensions"
            error stop "Aedelsten: array has the wrong shape"
        end if
        if (size(j_out,1) /= size(density,1) .or. size(j_out,1) /= size(eri_dense,1)) then
            write(error_unit, '(A)') "Aedelsten: j_out, density and eri_dense must all share the same n"
            error stop "Aedelsten: arrays disagree on n"
        end if
        n = size(j_out, 1, kind=aedelsten_int_size)

        rc = compute_j_and_k_matrix_stored_c(j_out, k_out, density, eri_dense, n, j_coeff, k_coeff)
        if (rc /= 0) error stop "Problem when constructing the J and K matrices. Stopping program."
    end subroutine compute_j_and_k_matrix_stored

    ! Single pass over eri_dense for J, K_alpha and K_beta together --
    ! prefer this over separate compute_j_matrix_stored +
    ! compute_k_matrix_unrestricted_stored calls when all three are
    ! wanted. density_total is J's own density (alpha+beta, computed by
    ! the caller, same requirement as compute_k_matrix_unrestricted_stored's
    ! density_alpha/density_beta split).
    subroutine compute_j_and_k_unrestricted_matrix_stored(&
        j_out, k_alpha_out, k_beta_out, density_alpha, density_beta, density_total, eri_dense, j_coeff, k_coeff &
    )
        real(kind=aedelsten_float_size), intent(inout) :: j_out(:,:)
        real(kind=aedelsten_float_size), intent(inout) :: k_alpha_out(:,:)
        real(kind=aedelsten_float_size), intent(inout) :: k_beta_out(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density_alpha(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density_beta(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density_total(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: eri_dense(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)    :: j_coeff
        real(kind=aedelsten_float_size), intent(in)    :: k_coeff

        integer(kind=aedelsten_int_size) :: n, rc

        if (.not. is_contiguous(j_out) .or. .not. is_contiguous(k_alpha_out) .or. &
            .not. is_contiguous(k_beta_out) .or. .not. is_contiguous(density_alpha) .or. &
            .not. is_contiguous(density_beta) .or. .not. is_contiguous(density_total) .or. &
            .not. is_contiguous(eri_dense)) then
            write(error_unit, '(A)') "Aedelsten: all arrays must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (any(shape(k_alpha_out) /= shape(k_beta_out)) .or. any(shape(k_alpha_out) /= shape(j_out)) &
                .or. any(shape(density_alpha) /= shape(density_beta)) .or. any(shape(density_alpha) /= shape(density_total)) &
                .or. size(j_out,1) /= size(j_out,2) .or. size(density_alpha,1) /= size(density_alpha,2) &
                .or. any(shape(eri_dense) /= size(eri_dense,1))) then
            write(error_unit, '(A)') "Aedelsten: j_out/k_alpha_out/k_beta_out and density_alpha/density_beta/" // &
                "density_total must each be square and pairwise the same shape, eri_dense must be square " // &
                "in all four dimensions"
            error stop "Aedelsten: array has the wrong shape"
        end if
        if (size(j_out,1) /= size(density_alpha,1) .or. size(j_out,1) /= size(eri_dense,1)) then
            write(error_unit, '(A)') "Aedelsten: j_out, density_alpha and eri_dense must all share the same n"
            error stop "Aedelsten: arrays disagree on n"
        end if
        n = size(j_out, 1, kind=aedelsten_int_size)

        rc = compute_j_and_k_unrestricted_matrix_stored_c(&
            j_out, k_alpha_out, k_beta_out, density_alpha, density_beta, density_total, eri_dense, n, j_coeff, k_coeff)
        if (rc /= 0) error stop "Problem when constructing the unrestricted J and K matrices. Stopping program."
    end subroutine compute_j_and_k_unrestricted_matrix_stored

    ! j_out = coeff J(density) from the dense (S1) ERI tensor eri_dense.
    subroutine compute_j_matrix_stored(j_out, density, eri_dense, coeff)
        real(kind=aedelsten_float_size), intent(inout) :: j_out(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: eri_dense(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)    :: coeff

        integer(kind=aedelsten_int_size) :: n, rc

        if (.not. is_contiguous(j_out) .or. .not. is_contiguous(density) .or. .not. is_contiguous(eri_dense)) then
            write(error_unit, '(A)') "Aedelsten: j_out/density/eri_dense must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (size(j_out,1) /= size(j_out,2) .or. size(density,1) /= size(density,2) &
                .or. any(shape(eri_dense) /= size(eri_dense,1))) then
            write(error_unit, '(A)') "Aedelsten: j_out/density must be square, eri_dense must be square in all four dimensions"
            error stop "Aedelsten: array has the wrong shape"
        end if
        if (size(j_out,1) /= size(density,1) .or. size(j_out,1) /= size(eri_dense,1)) then
            write(error_unit, '(A)') "Aedelsten: j_out, density and eri_dense must all share the same n"
            error stop "Aedelsten: arrays disagree on n"
        end if
        n = size(j_out, 1, kind=aedelsten_int_size)

        rc = compute_j_matrix_stored_c(j_out, density, eri_dense, n, coeff)
        if (rc /= 0) error stop "Problem when constructing the J matrix. Stopping program."
    end subroutine compute_j_matrix_stored

    ! k_out = coeff K(density) from the dense (S1) ERI tensor eri_dense.
    subroutine compute_k_matrix_stored(k_out, density, eri_dense, coeff)
        real(kind=aedelsten_float_size), intent(inout) :: k_out(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: eri_dense(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)    :: coeff

        integer(kind=aedelsten_int_size) :: n, rc

        if (.not. is_contiguous(k_out) .or. .not. is_contiguous(density) .or. .not. is_contiguous(eri_dense)) then
            write(error_unit, '(A)') "Aedelsten: k_out/density/eri_dense must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (size(k_out,1) /= size(k_out,2) .or. size(density,1) /= size(density,2) &
                .or. any(shape(eri_dense) /= size(eri_dense,1))) then
            write(error_unit, '(A)') "Aedelsten: k_out/density must be square, eri_dense must be square in all four dimensions"
            error stop "Aedelsten: array has the wrong shape"
        end if
        if (size(k_out,1) /= size(density,1) .or. size(k_out,1) /= size(eri_dense,1)) then
            write(error_unit, '(A)') "Aedelsten: k_out, density and eri_dense must all share the same n"
            error stop "Aedelsten: arrays disagree on n"
        end if
        n = size(k_out, 1, kind=aedelsten_int_size)

        rc = compute_k_matrix_stored_c(k_out, density, eri_dense, n, coeff)
        if (rc /= 0) error stop "Problem when constructing the K matrix. Stopping program."
    end subroutine compute_k_matrix_stored

    ! k_alpha_out and k_beta_out = coeff K of each spin's density from the dense (S1) ERI tensor eri_dense, in one pass.
    subroutine compute_k_matrix_unrestricted_stored(k_alpha_out, k_beta_out, density_alpha, density_beta, eri_dense, coeff)
        real(kind=aedelsten_float_size), intent(inout) :: k_alpha_out(:,:)
        real(kind=aedelsten_float_size), intent(inout) :: k_beta_out(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density_alpha(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density_beta(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: eri_dense(:,:,:,:)
        real(kind=aedelsten_float_size), intent(in)    :: coeff

        integer(kind=aedelsten_int_size) :: n, rc

        if (.not. is_contiguous(k_alpha_out) .or. .not. is_contiguous(k_beta_out) .or. &
            .not. is_contiguous(density_alpha) .or. .not. is_contiguous(density_beta) .or. &
            .not. is_contiguous(eri_dense)) then
            write(error_unit, '(A)') "Aedelsten: all arrays must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (any(shape(k_alpha_out) /= shape(k_beta_out)) .or. any(shape(density_alpha) /= shape(density_beta)) &
                .or. size(k_alpha_out,1) /= size(k_alpha_out,2) .or. size(density_alpha,1) /= size(density_alpha,2) &
                .or. any(shape(eri_dense) /= size(eri_dense,1))) then
            write(error_unit, '(A)') "Aedelsten: k_alpha_out/k_beta_out/density_alpha/density_beta must be " // &
                "square and pairwise the same shape, eri_dense must be square in all four dimensions"
            error stop "Aedelsten: array has the wrong shape"
        end if
        if (size(k_alpha_out,1) /= size(density_alpha,1) .or. size(k_alpha_out,1) /= size(eri_dense,1)) then
            write(error_unit, '(A)') "Aedelsten: k_alpha_out, density_alpha and eri_dense must all share the same n"
            error stop "Aedelsten: arrays disagree on n"
        end if
        n = size(k_alpha_out, 1, kind=aedelsten_int_size)

        rc = compute_k_matrix_unrestricted_stored_c(k_alpha_out, k_beta_out, density_alpha, density_beta, eri_dense, n, coeff)
        if (rc /= 0) error stop "Problem when constructing the unrestricted K matrices. Stopping program."
    end subroutine compute_k_matrix_unrestricted_stored

    ! ── Picture-change integrals ─────────────────────────────────────────

    ! Relativistic picture-change integrals, always over the decontracted
    ! (primitive) basis -- requires set_primitive_shell_pairs() first, no
    ! separate primitive= toggle (unlike compute_one_electron_integrals).
    ! Flat/1-D output only -- see this procedure's declaration in the type
    ! definition for why it doesn't get the rank-dispatched (n,n)/
    ! (n,n,n_planes) convenience compute_one_electron_integrals does.
    !
    ! spin_orbit=.false.: output has whatever shape the operator would
    ! normally produce (n_planes=1 for a plain operator, or the operator's
    ! own component count for a moment). spin_orbit=.true.: output gains
    ! an extra factor of 4 (x, y, z, scalar) on top of that.
    subroutine compute_one_electron_picture_change_integrals(self, output, n, operator, spin_orbit, n_planes, origin, order)
        class(AedelstenContext), intent(inout)                 :: self
        real(kind=aedelsten_float_size), intent(inout)         :: output(:)
        integer(kind=external_int_size), intent(in)            :: n
        character(len=*), intent(in)                           :: operator
        logical, intent(in)                                    :: spin_orbit
        integer(kind=external_int_size), intent(in), optional  :: n_planes
        real(kind=aedelsten_float_size), intent(in), optional  :: origin(3)
        integer(kind=external_int_size), intent(in), optional  :: order

        integer(kind=aedelsten_int_size) :: n_wire, n_planes_wire, order_wire, operator_int, &
            spin_orbit_int, expected_planes, rc
        real(kind=aedelsten_float_size)  :: origin_x, origin_y, origin_z
        logical                          :: expect_components

        if (.not. is_contiguous(output)) then
            write(error_unit, '(A)') "Aedelsten: output array must be contiguous (got a strided/non-contiguous " // &
                "section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: output array is not contiguous"
        end if

        ! order and origin are bundled arguments (moment evaluation), so
        ! force both to be present at the same time, or both absent.
        if (present(order) .neqv. present(origin)) then
            error stop "Aedelsten: order and origin must both be present, or both absent, for one-electron integrals"
        end if
        expect_components = present(order)

        n_wire = to_wire_i32(n, "n")
        if (present(n_planes)) then
            n_planes_wire = to_wire_i32(n_planes, "n_planes")
        else
            n_planes_wire = 1
        end if

        if (size(output, kind=aedelsten_int_size) /= n_wire * n_wire * n_planes_wire) then
            write(error_unit, '(A)') "Aedelsten: output buffer's size does not match n*n*n_planes"
            error stop "Aedelsten: output buffer size mismatch"
        end if

        if (expect_components) then
            order_wire = to_wire_i32(order, "order")
            expected_planes = compute_moment_component_count_c(order_wire)
            if (spin_orbit) expected_planes = expected_planes * 4_aedelsten_int_size
            origin_x = origin(1)
            origin_y = origin(2)
            origin_z = origin(3)
        else
            order_wire = 0
            origin_x = 0.0_aedelsten_float_size
            origin_y = 0.0_aedelsten_float_size
            origin_z = 0.0_aedelsten_float_size
            expected_planes = 1_aedelsten_int_size
            if (spin_orbit) expected_planes = 4_aedelsten_int_size
        end if
        if (n_planes_wire /= expected_planes) then
            write(error_unit, '(A)') "Aedelsten: n_planes does not match the number of components this " // &
                "operator/spin_orbit combination produces"
            error stop "Aedelsten: picture-change output buffer has the wrong number of components"
        end if

        operator_int = resolve_operator(operator, expect_components=expect_components)
        spin_orbit_int = 0_aedelsten_int_size
        if (spin_orbit) spin_orbit_int = 1_aedelsten_int_size

        rc = compute_one_electron_picture_change_integrals_c( &
            self%handle, output, n_wire, operator_int, n_planes_wire, &
            origin_x, origin_y, origin_z, order_wire, spin_orbit_int)

        ! Rust should already have killed the run before this.
        if (rc /= 0) error stop "Problem when evaluating one-electron picture-change integrals. Stopping program."
    end subroutine compute_one_electron_picture_change_integrals

    ! ── X2C one-electron Hamiltonian ──────────────────────────────────

    ! oneham/s/t/v/pvp are all primitive-basis (n_prim_unique, n_prim_unique)
    ! -- s/t/v from compute_one_electron_integrals(..., primitive=.true.)
    ! for overlap/kinetic/nuclear_attraction, pvp from
    ! compute_one_electron_picture_change_integrals(..., spin_orbit=.false.)
    ! on the same operator. c_light absent uses the library's own default.
    subroutine construct_x2c_one_electron_hamiltonian(self, oneham, s, t, v, pvp, c_light)
        class(AedelstenContext), intent(inout)                :: self
        real(kind=aedelsten_float_size), intent(inout)        :: oneham(:,:)
        real(kind=aedelsten_float_size), intent(in)           :: s(:,:)
        real(kind=aedelsten_float_size), intent(in)           :: t(:,:)
        real(kind=aedelsten_float_size), intent(in)           :: v(:,:)
        real(kind=aedelsten_float_size), intent(in)           :: pvp(:,:)
        real(kind=aedelsten_float_size), intent(in), optional :: c_light

        integer(kind=aedelsten_int_size) :: n, rc
        real(kind=aedelsten_float_size)  :: c_light_wire

        if (.not. is_contiguous(oneham) .or. .not. is_contiguous(s) .or. .not. is_contiguous(t) .or. &
            .not. is_contiguous(v) .or. .not. is_contiguous(pvp)) then
            write(error_unit, '(A)') "Aedelsten: oneham/s/t/v/pvp must be contiguous (got a strided/non-" // &
                "contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a " // &
                "plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (any(shape(oneham) /= shape(s)) .or. any(shape(s) /= shape(t)) .or. any(shape(t) /= shape(v)) &
                .or. any(shape(v) /= shape(pvp)) .or. size(oneham,1) /= size(oneham,2)) then
            write(error_unit, '(A)') "Aedelsten: oneham/s/t/v/pvp must all be square and the same shape"
            error stop "Aedelsten: array has the wrong shape"
        end if
        n = size(oneham, 1, kind=aedelsten_int_size)

        if (present(c_light)) then
            c_light_wire = c_light
        else
            c_light_wire = -1.0_aedelsten_float_size ! Sentinel: tells Rust to use its own default.
        end if

        rc = construct_x2c_one_electron_hamiltonian_c(self%handle, oneham, s, t, v, pvp, n, c_light_wire)
        if (rc /= 0) error stop "Problem when constructing the X2C one-electron Hamiltonian. Stopping program."
    end subroutine construct_x2c_one_electron_hamiltonian

    ! ── Contract primitive-basis results back to the contracted basis ───

    ! e.g. the output of compute_one_electron_integrals(..., primitive=.true.)
    ! or compute_one_electron_picture_change_integrals. Requires
    ! set_primitive_shell_pairs() first (that's where the contraction
    ! matrix this uses comes from). n_prim/n_cont are taken from primitive's
    ! and output's own shapes respectively, not passed separately.
    subroutine contract_one_electron_integrals_matrix(self, primitive, output)
        class(AedelstenContext), intent(inout)         :: self
        real(kind=aedelsten_float_size), intent(in)    :: primitive(:,:)
        real(kind=aedelsten_float_size), intent(inout) :: output(:,:)

        integer(kind=aedelsten_int_size) :: n_prim, n_cont, rc

        if (.not. is_contiguous(primitive) .or. .not. is_contiguous(output)) then
            write(error_unit, '(A)') "Aedelsten: primitive/output arrays must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (size(primitive, 1) /= size(primitive, 2)) then
            write(error_unit, '(A)') "Aedelsten: primitive input matrix must be square"
            error stop "Aedelsten: primitive input matrix must be square"
        end if
        if (size(output, 1) /= size(output, 2)) then
            write(error_unit, '(A)') "Aedelsten: output matrix must be square"
            error stop "Aedelsten: output matrix must be square"
        end if
        n_prim = size(primitive, 1, kind=aedelsten_int_size)
        n_cont = size(output, 1, kind=aedelsten_int_size)

        rc = contract_one_electron_matrix_c(self%handle, primitive, n_prim, output, n_cont)
        if (rc /= 0) error stop "Problem when contracting one-electron integrals. Stopping program."
    end subroutine contract_one_electron_integrals_matrix

    ! contract_one_electron_integrals plane by plane: primitive (n_prim, n_prim, n_planes) into output (n_cont, n_cont,
    ! n_planes).
    subroutine contract_one_electron_integrals_tensor(self, primitive, output)
        class(AedelstenContext), intent(inout)         :: self
        real(kind=aedelsten_float_size), intent(in)    :: primitive(:,:,:)
        real(kind=aedelsten_float_size), intent(inout) :: output(:,:,:)

        integer(kind=aedelsten_int_size) :: n_prim, n_cont, n_planes, rc

        if (.not. is_contiguous(primitive) .or. .not. is_contiguous(output)) then
            write(error_unit, '(A)') "Aedelsten: primitive/output arrays must be contiguous (got a strided/" // &
                "non-contiguous section) -- Aedelsten never silently copies; pass a contiguous slice or a plain array instead"
            error stop "Aedelsten: array is not contiguous"
        end if
        if (size(primitive, 1) /= size(primitive, 2)) then
            write(error_unit, '(A)') "Aedelsten: primitive input tensor must be square in its first two dimensions"
            error stop "Aedelsten: primitive input tensor must be square in its first two dimensions"
        end if
        if (size(output, 1) /= size(output, 2)) then
            write(error_unit, '(A)') "Aedelsten: output tensor must be square in its first two dimensions"
            error stop "Aedelsten: output tensor must be square in its first two dimensions"
        end if
        if (size(primitive, 3) /= size(output, 3)) then
            write(error_unit, '(A)') "Aedelsten: primitive and output tensors must have the same number of planes"
            error stop "Aedelsten: primitive and output tensors must have the same number of planes"
        end if
        n_prim = size(primitive, 1, kind=aedelsten_int_size)
        n_cont = size(output, 1, kind=aedelsten_int_size)
        n_planes = size(primitive, 3, kind=aedelsten_int_size)

        rc = contract_one_electron_array3d_c(self%handle, primitive, n_prim, n_planes, output, n_cont)
        if (rc /= 0) error stop "Problem when contracting one-electron integrals. Stopping program."
    end subroutine contract_one_electron_integrals_tensor

    ! ── Induced-dipole embedding ─────────────────────────────────────────

    subroutine set_polarizable_sites(self, filename)
        class(AedelstenContext), intent(inout) :: self
        character(len=*),        intent(in)    :: filename
        integer(kind=aedelsten_int_size) :: rc
        rc = set_polarizable_sites_c(self%handle, to_c_string(filename))
        if (rc /= 0) error stop "Aedelsten: failed to set polarizable sites"
    end subroutine set_polarizable_sites

    ! Frees the polarizable sites and the polarization set up on them.
    subroutine free_polarizable_sites(self)
        class(AedelstenContext), intent(inout) :: self
        call free_polarizable_sites_c(self%handle)
    end subroutine free_polarizable_sites

    ! The number of polarizable sites; the dipoles are a (3, n) array.
    function get_n_polarizable_sites(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=int64) :: n
        n = get_n_polarizable_sites_c(self%handle)
        if (n < 0) error stop "Aedelsten: could not count the polarizable sites (are they set?)"
    end function get_n_polarizable_sites

    ! Whether polarizable sites are set.
    function has_polarizable_sites(self) result(flag)
        class(AedelstenContext), intent(in) :: self
        logical :: flag
        flag = has_polarizable_sites_c(self%handle) == 1
    end function has_polarizable_sites

    ! Whether set_polarization has set the polarization up.
    function has_polarization(self) result(flag)
        class(AedelstenContext), intent(in) :: self
        logical :: flag
        flag = has_polarization_c(self%handle) == 1
    end function has_polarization

    ! A selection's constant and its custom points (null and 0 without them).
    subroutine site_selection(selection, points, s, p, n_points)
        character(len=*),                intent(in)                                 :: selection
        real(kind=aedelsten_float_size), intent(in), optional, target, contiguous   :: points(:,:)
        integer(kind=aedelsten_int_size), intent(out)                               :: s
        type(c_ptr),                     intent(out)                                :: p
        integer(kind=c_int64_t),         intent(out)                                :: n_points
        s = query_i32_checked("site_selection_" // trim(lowercase(selection)))
        p = c_null_ptr
        n_points = 0
        if (present(points)) then
            if (size(points, 1) /= 3) error stop "Aedelsten: the custom points must be (3, n_points)"
            p = c_loc(points)
            n_points = size(points, 2, kind=c_int64_t)
        end if
    end subroutine site_selection

    ! The number of sites of selection.
    function count_electric_field_sites(self, selection, points) result(n)
        class(AedelstenContext),         intent(in)                               :: self
        character(len=*),                intent(in)                               :: selection
        real(kind=aedelsten_float_size), intent(in), optional, target, contiguous :: points(:,:)
        integer(kind=int64) :: n
        integer(kind=aedelsten_int_size) :: s
        type(c_ptr) :: p
        integer(kind=c_int64_t) :: n_points
        call site_selection(selection, points, s, p, n_points)
        n = count_electric_field_sites_c(self%handle, s, p, n_points)
        if (n < 0) error stop "Aedelsten: count_electric_field_sites failed"
    end function count_electric_field_sites

    ! Each site's record in its input file (0-based), -1 for a site from none (nuclei, AIMPs, custom points).
    subroutine get_input_indices(self, selection, indices, points)
        class(AedelstenContext),         intent(in)                               :: self
        character(len=*),                intent(in)                               :: selection
        integer(kind=int64),             intent(inout), contiguous                :: indices(:)
        real(kind=aedelsten_float_size), intent(in), optional, target, contiguous :: points(:,:)
        integer(kind=aedelsten_int_size) :: s, rc
        type(c_ptr) :: p
        integer(kind=c_int64_t) :: n_points
        call site_selection(selection, points, s, p, n_points)
        rc = get_input_indices_c(self%handle, s, p, n_points, indices, size(indices, kind=c_int64_t))
        if (rc /= 0) error stop "Aedelsten: get_input_indices failed (one entry a site?)"
    end subroutine get_input_indices

    ! The electric-field integrals of size(output, 3) / 3 sites from first_site (0-based), (n, n, 3 count): per site
    ! its x, y and z planes.
    subroutine compute_electric_field_integrals(self, selection, first_site, output, points)
        class(AedelstenContext),         intent(in)                               :: self
        character(len=*),                intent(in)                               :: selection
        integer(kind=int64),             intent(in)                               :: first_site
        real(kind=aedelsten_float_size), intent(inout), contiguous                :: output(:,:,:)
        real(kind=aedelsten_float_size), intent(in), optional, target, contiguous :: points(:,:)
        integer(kind=aedelsten_int_size) :: s, rc
        type(c_ptr) :: p
        integer(kind=c_int64_t) :: n_points
        if (mod(size(output, 3), 3) /= 0) error stop "Aedelsten: compute_electric_field_integrals: output must be (n, n, 3 count)"
        call site_selection(selection, points, s, p, n_points)
        rc = compute_electric_field_integrals_c(self%handle, s, p, n_points, int(first_site, c_int64_t), &
            int(size(output, 3) / 3, c_int64_t), output, size(output, kind=c_int64_t))
        if (rc /= 0) error stop "Aedelsten: compute_electric_field_integrals failed"
    end subroutine compute_electric_field_integrals

    ! The classical field at the sites of `selection` from the sources switched on (all by default), (3, n_sites).
    subroutine compute_static_field(self, selection, field, nuclei, core_aimps, embedding_aimps, external_multipoles, points)
        class(AedelstenContext),         intent(in)                               :: self
        character(len=*),                intent(in)                               :: selection
        real(kind=aedelsten_float_size), intent(inout), contiguous                :: field(:,:)
        logical,                         intent(in), optional                     :: nuclei, core_aimps, embedding_aimps, &
                                                                                     external_multipoles
        real(kind=aedelsten_float_size), intent(in), optional, target, contiguous :: points(:,:)
        integer(kind=aedelsten_int_size) :: s, rc
        type(c_ptr) :: p
        integer(kind=c_int64_t) :: n_points
        call site_selection(selection, points, s, p, n_points)
        rc = compute_static_field_c(self%handle, s, p, n_points, switch(nuclei), switch(core_aimps), &
            switch(embedding_aimps), switch(external_multipoles), field, size(field, kind=c_int64_t))
        if (rc /= 0) error stop "Aedelsten: compute_static_field failed"
    contains
        ! 1 unless flag is present and false.
        integer(kind=aedelsten_int_size) function switch(flag)
            logical, intent(in), optional :: flag
            switch = 1
            if (present(flag)) then
                if (.not. flag) switch = 0
            end if
        end function switch
    end subroutine compute_static_field

    ! The field of the plain density at the sites of `selection`, (3, n_sites); chunked by chunk_sites sites when
    ! given and positive, else by chunk_budget in chunk_budget_unit (the default without one).
    subroutine compute_electronic_field(self, density, selection, field, chunk_sites, chunk_budget, chunk_budget_unit, points)
        class(AedelstenContext),         intent(in)                               :: self
        real(kind=aedelsten_float_size), intent(in), contiguous                   :: density(:,:)
        character(len=*),                intent(in)                               :: selection
        real(kind=aedelsten_float_size), intent(inout), contiguous                :: field(:,:)
        integer(kind=int64),             intent(in), optional                     :: chunk_sites, chunk_budget
        character(len=*),                intent(in), optional                     :: chunk_budget_unit
        real(kind=aedelsten_float_size), intent(in), optional, target, contiguous :: points(:,:)
        integer(kind=aedelsten_int_size) :: s, rc, unit
        type(c_ptr) :: p
        integer(kind=c_int64_t) :: n_points, sites, budget
        if (size(density, 1, kind=int64) /= self%get_n_cont() .or. size(density, 2, kind=int64) /= self%get_n_cont()) &
            error stop "Aedelsten: compute_electronic_field: density must be (n, n)"
        sites = -1
        budget = -1
        unit = query_i32_checked("byte_unit_mib")
        if (present(chunk_sites)) sites = int(chunk_sites, c_int64_t)
        if (present(chunk_budget)) budget = int(chunk_budget, c_int64_t)
        if (present(chunk_budget_unit)) unit = query_i32_checked("byte_unit_" // trim(lowercase(chunk_budget_unit)))
        call site_selection(selection, points, s, p, n_points)
        rc = compute_electronic_field_c(self%handle, density, s, p, n_points, sites, budget, unit, field, &
            size(field, kind=c_int64_t))
        if (rc /= 0) error stop "Aedelsten: compute_electronic_field failed"
    end subroutine compute_electronic_field

    ! How the external multipoles meet the electrons: "contact_free" (the default) or "exact".
    subroutine set_multipole_convention(self, convention)
        class(AedelstenContext), intent(inout) :: self
        character(len=*),        intent(in)    :: convention
        integer(kind=aedelsten_int_size) :: rc
        rc = set_multipole_convention_c(self%handle, query_i32_checked("multipole_convention_" // trim(lowercase(convention))))
        if (rc /= 0) error stop "Aedelsten: set_multipole_convention failed"
    end subroutine set_multipole_convention

    ! The convention set_multipole_convention set: "contact_free" or "exact".
    function get_multipole_convention(self) result(convention)
        class(AedelstenContext), intent(in) :: self
        character(len=:), allocatable :: convention
        integer(kind=aedelsten_int_size) :: c
        c = get_multipole_convention_c(self%handle)
        if (c == query_i32_checked("multipole_convention_exact")) then
            convention = "exact"
        else if (c == query_i32_checked("multipole_convention_contact_free")) then
            convention = "contact_free"
        else
            error stop "Aedelsten: get_multipole_convention failed"
        end if
    end function get_multipole_convention

    ! Once per geometry, after set_polarizable_sites (and after set_external_multipoles, if any): allocates
    ! every buffer the calls below reuse. `options` defaults to the library's defaults.
    subroutine set_polarization(self, options)
        class(AedelstenContext),                      intent(inout) :: self
        type(AedelstenPolarizationOptions), optional, intent(in)    :: options
        type(AedelstenPolarizationOptions) :: local
        integer(kind=aedelsten_int_size) :: rc
        if (present(options)) local = options
        rc = set_polarization_c(self%handle, local)
        if (rc /= 0) error stop "Aedelsten: failed to set up the polarization"
    end subroutine set_polarization

    ! Frees the polarization state.
    subroutine free_polarization(self)
        class(AedelstenContext), intent(inout) :: self
        call free_polarization_c(self%handle)
    end subroutine free_polarization

    ! One Fock build: from the 1-RDM `density` (n,n) the induced dipoles are solved (warm-started from the
    ! previous call), `v_ind` (n,n) is OVERWRITTEN with the operator to add to the one-electron Hamiltonian,
    ! and `energy` is E_ind, to add to the total energy. With `threshold`, the dipoles are solved to it for this
    ! call in place of the setup's.
    subroutine compute_induction(self, density, v_ind, energy, iterations, threshold)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(in)    :: density(:,:)
        real(kind=aedelsten_float_size), intent(inout) :: v_ind(:,:)
        real(kind=aedelsten_float_size), intent(out)   :: energy
        integer(kind=external_int_size), intent(out), optional :: iterations
        real(kind=aedelsten_float_size), intent(in), optional :: threshold
        integer(kind=aedelsten_int_size) :: rc, iter
        call require_square_pair(density, v_ind, "compute_induction")
        if (present(threshold)) then
            rc = compute_induction_within_c(self%handle, density, v_ind, threshold, energy, iter)
        else
            rc = compute_induction_c(self%handle, density, v_ind, energy, iter)
        end if
        if (rc /= 0) error stop "Aedelsten: compute_induction failed"
        if (present(iterations)) iterations = iter
    end subroutine compute_induction

    ! How this rank holds the field integrals of its share of the polarizable sites: storage(1) the sites of the
    ! share, (2) the first kept in memory (the rest evaluated in every pass), (3) the bytes of a site's kept
    ! integrals, (4) the resident_budget in bytes that keeps the whole share in memory.
    subroutine get_polarization_storage(self, storage)
        class(AedelstenContext), intent(in)  :: self
        integer(kind=int64),     intent(out) :: storage(4)
        integer(kind=aedelsten_int_size) :: rc
        integer(kind=c_int64_t) :: values(4)
        rc = get_polarization_storage_c(self%handle, values)
        if (rc /= 0) error stop "Aedelsten: get_polarization_storage failed (has set_polarization run?)"
        storage = values
    end subroutine get_polarization_storage

    ! The memory budget of this rank's kept field integrals, `budget` in `unit` ("gib", "mib" or "kib"; 0: none kept),
    ! in place of set_polarization's resident_budget: those of the first sites it holds are computed at the next pass
    ! and kept, the rest evaluated in every pass. Collective.
    subroutine set_resident_field_budget(self, budget, unit)
        class(AedelstenContext),         intent(inout) :: self
        integer(kind=external_int_size), intent(in)    :: budget
        character(len=*),                intent(in)    :: unit
        integer(kind=aedelsten_int_size) :: rc
        rc = set_resident_field_budget_c(self%handle, to_wire_i32(budget, "budget"), &
            query_i32_checked("byte_unit_" // trim(lowercase(unit))))
        if (rc /= 0) error stop "Aedelsten: set_resident_field_budget failed (has set_polarization run?)"
    end subroutine set_resident_field_budget

    ! gradient(3, n_centres) is OVERWRITTEN with dE_ind/dX at fixed density, from the induced dipoles of the last
    ! compute_induction (call it with the same density first; in an SCF, the converged one). `active` as in
    ! compute_one_electron_gradient.
    subroutine compute_induction_gradient(self, gradient, density, active)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: gradient(:,:)
        real(kind=aedelsten_float_size), intent(in)    :: density(:,:)
        logical, intent(in), optional                  :: active(:)
        integer(kind=aedelsten_int_size) :: n_centres, rc
        integer(kind=c_int), allocatable, target :: active_flags(:)
        type(c_ptr) :: active_ptr
        if (.not. is_contiguous(density)) error stop "Aedelsten: density array is not contiguous"
        if (size(density, 1) /= size(density, 2) .or. size(density, 1, kind=aedelsten_int_size) /= get_n_cont_c(self%handle)) then
            error stop "Aedelsten: density must be n_cont x n_cont"
        end if
        call check_classical_gradient(self, gradient, active, n_centres, active_flags, active_ptr)
        rc = compute_induction_gradient_c(self%handle, gradient, n_centres, density, active_ptr)
        if (rc /= 0) error stop "Aedelsten: compute_induction_gradient failed"
    end subroutine compute_induction_gradient

    ! Linear response of the induction to one density (no static field; the ground-state dipoles are not
    ! touched): v1 = V1[d1]. Needs response_batch >= 1 in the options.
    subroutine compute_induction_response_matrix(self, density, v1)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(in)    :: density(:,:)
        real(kind=aedelsten_float_size), intent(inout) :: v1(:,:)
        integer(kind=aedelsten_int_size) :: rc
        call require_square_pair(density, v1, "compute_induction_response")
        rc = compute_induction_response_c(self%handle, density, 1_aedelsten_int_size, v1)
        if (rc /= 0) error stop "Aedelsten: compute_induction_response failed"
    end subroutine compute_induction_response_matrix

    ! The same for a batch: (n,n,k) densities -> (n,n,k) operators; the batch shares the integral
    ! evaluations (k may exceed response_batch, the library then works in sub-batches).
    subroutine compute_induction_response_tensor(self, densities, v1)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(in)    :: densities(:,:,:)
        real(kind=aedelsten_float_size), intent(inout) :: v1(:,:,:)
        integer(kind=aedelsten_int_size) :: rc
        call require_batch_pair(densities, v1, "compute_induction_response")
        rc = compute_induction_response_c(self%handle, densities, &
            size(densities, 3, kind=aedelsten_int_size), v1)
        if (rc /= 0) error stop "Aedelsten: compute_induction_response failed"
    end subroutine compute_induction_response_tensor

    ! The induction's couplings between (n,n,k) densities: couplings(m,n) = -f_m . (alpha^-1 - T)^-1 f_n, f the
    ! electrons' field at the sites (no static field), equal to tr(D_m V1[D_n]) without forming any operator.
    ! couplings is (k,k), overwritten. Needs response_batch >= 1 in the options.
    subroutine compute_induction_couplings(self, densities, couplings)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(in)    :: densities(:,:,:)
        real(kind=aedelsten_float_size), intent(inout) :: couplings(:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (.not. (is_contiguous(densities) .and. is_contiguous(couplings))) &
            error stop "Aedelsten: compute_induction_couplings: arrays must be contiguous (Aedelsten never silently copies)"
        if (size(densities, 1) /= size(densities, 2)) &
            error stop "Aedelsten: compute_induction_couplings: densities must be (n,n,k)"
        if (size(couplings, 1) /= size(densities, 3) .or. size(couplings, 2) /= size(densities, 3)) &
            error stop "Aedelsten: compute_induction_couplings: couplings must be (k,k) for (n,n,k) densities"
        rc = compute_induction_couplings_c(self%handle, densities, &
            size(densities, 3, kind=aedelsten_int_size), couplings)
        if (rc /= 0) error stop "Aedelsten: compute_induction_couplings failed"
    end subroutine compute_induction_couplings

    ! compute_induction for k states at once, each with its OWN induced dipoles (kept and warm-started
    ! between calls, so keep the state order): operators(:,:,m) and energies(m) belong to densities(:,:,m).
    ! Weight them yourself for a state average. Needs state_slots >= k in the options. With `threshold`, the
    ! dipoles are solved to it for this call in place of the setup's.
    subroutine compute_induction_states(self, densities, operators, energies, threshold)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(in)    :: densities(:,:,:)
        real(kind=aedelsten_float_size), intent(inout) :: operators(:,:,:)
        real(kind=aedelsten_float_size), intent(inout) :: energies(:)
        real(kind=aedelsten_float_size), intent(in), optional :: threshold
        integer(kind=aedelsten_int_size) :: rc
        call require_batch_pair(densities, operators, "compute_induction_states")
        if (size(energies) /= size(densities, 3)) error stop "Aedelsten: compute_induction_states: energies must have one entry per state"
        if (present(threshold)) then
            rc = compute_induction_states_within_c(self%handle, densities, &
                size(densities, 3, kind=aedelsten_int_size), operators, energies, threshold)
        else
            rc = compute_induction_states_c(self%handle, densities, &
                size(densities, 3, kind=aedelsten_int_size), operators, energies)
        end if
        if (rc /= 0) error stop "Aedelsten: compute_induction_states failed"
    end subroutine compute_induction_states

    ! The effective external field (EEF) operators, v(:,:,i) for i = x, y, z: the induced dipoles of a uniform unit
    ! external field coupled to the electrons, V^(i) = -sum_s mu_s^(i) . t_s; added to the dipole integrals r_i.
    subroutine compute_effective_external_field(self, v)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: v(:,:,:)
        integer(kind=aedelsten_int_size) :: rc, n
        n = self%get_n_cont()
        if (size(v, 1) /= n .or. size(v, 2) /= n .or. size(v, 3) /= 3) &
            error stop "Aedelsten: compute_effective_external_field: v must be (n_cont, n_cont, 3)"
        rc = compute_effective_external_field_c(self%handle, v)
        if (rc /= 0) error stop "Aedelsten: compute_effective_external_field failed"
    end subroutine compute_effective_external_field

    ! The induced dipoles of the last compute_induction, `dipoles(:, s)` for polarizable site s (input order).
    subroutine get_induced_dipoles(self, dipoles)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: dipoles(:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (.not. is_contiguous(dipoles)) error stop "Aedelsten: dipoles must be contiguous"
        if (size(dipoles, 1) /= 3 .or. size(dipoles, 2, kind=int64) /= self%get_n_polarizable_sites()) &
            error stop "Aedelsten: get_induced_dipoles: dipoles must be (3, n_polarizable_sites)"
        rc = get_induced_dipoles_c(self%handle, dipoles, size(dipoles, kind=c_int64_t))
        if (rc /= 0) error stop "Aedelsten: get_induced_dipoles failed (has compute_induction run?)"
    end subroutine get_induced_dipoles

    ! The start of the next compute_induction's solve, `dipoles(:, s)` for polarizable site s: not a solution until it.
    subroutine set_induced_dipoles(self, dipoles)
        class(AedelstenContext),         intent(inout) :: self
        real(kind=aedelsten_float_size), intent(in)    :: dipoles(:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (.not. is_contiguous(dipoles)) error stop "Aedelsten: dipoles must be contiguous"
        if (size(dipoles, 1) /= 3 .or. size(dipoles, 2, kind=int64) /= self%get_n_polarizable_sites()) &
            error stop "Aedelsten: set_induced_dipoles: dipoles must be (3, n_polarizable_sites)"
        rc = set_induced_dipoles_c(self%handle, dipoles, size(dipoles, kind=c_int64_t))
        if (rc /= 0) error stop "Aedelsten: set_induced_dipoles failed (is the polarization set up?)"
    end subroutine set_induced_dipoles

    ! Stops unless a and b are contiguous square matrices of one size.
    subroutine require_square_pair(a, b, caller)
        real(kind=aedelsten_float_size), intent(in) :: a(:,:), b(:,:)
        character(len=*), intent(in) :: caller
        if (.not. (is_contiguous(a) .and. is_contiguous(b))) &
            error stop "Aedelsten: " // caller // ": arrays must be contiguous (Aedelsten never silently copies)"
        if (size(a, 1) /= size(a, 2) .or. any(shape(a) /= shape(b))) &
            error stop "Aedelsten: " // caller // ": density and operator must be square and of the same shape"
    end subroutine require_square_pair

    ! Stops unless a and b are contiguous batches of square matrices of one shape.
    subroutine require_batch_pair(a, b, caller)
        real(kind=aedelsten_float_size), intent(in) :: a(:,:,:), b(:,:,:)
        character(len=*), intent(in) :: caller
        if (.not. (is_contiguous(a) .and. is_contiguous(b))) &
            error stop "Aedelsten: " // caller // ": arrays must be contiguous (Aedelsten never silently copies)"
        if (size(a, 1) /= size(a, 2) .or. any(shape(a) /= shape(b))) &
            error stop "Aedelsten: " // caller // ": densities and operators must be (n,n,k) of the same shape"
    end subroutine require_batch_pair

    ! ── Memory diagnostics, Molcas-export identity data, Cartesian rescaling ──────

    ! Number of entries memory_breakdown fills -- allocate bytes(:) to this length before calling.
    function memory_breakdown_field_count() result(n)
        integer(kind=external_int_size) :: n
        n = int(memory_breakdown_field_count_c(), kind=external_int_size)
    end function memory_breakdown_field_count

    ! The number of rows of get_primitive_ids and get_primitives.
    function get_n_primitive_rows(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        n = int(get_n_primitive_rows_c(self%handle), kind=external_int_size)
    end function get_n_primitive_rows

    ! Per-field breakdown of how much memory this context is actually holding right now, in
    ! bytes. `bytes` must be pre-allocated to memory_breakdown_field_count(); fixed field order:
    ! qm_region, shell_pairs, scratch, external_multipoles, polarizable_sites, polarization,
    ! embedding_aimp_operators, primitive_shells, primitive_shell_pairs, contraction_matrix,
    ! eri_data, picture_change, symmetry (see AedelstenContext::memory_breakdown's Rust-side doc comment).
    subroutine memory_breakdown(self, bytes)
        class(AedelstenContext), intent(in)    :: self
        integer(kind=c_int64_t), intent(inout) :: bytes(:)
        integer(kind=aedelsten_int_size)       :: rc
        if (size(bytes, kind=external_int_size) /= memory_breakdown_field_count()) &
            error stop "Aedelsten: memory_breakdown: bytes must have length memory_breakdown_field_count()"
        rc = memory_breakdown_c(self%handle, bytes, size(bytes, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: memory_breakdown failed"
    end subroutine memory_breakdown

    ! BASIS_FUNCTION_IDS-equivalent: ids(:,1)=centre, ids(:,2)=shell, ids(:,3)=l, ids(:,4)=m, one
    ! row per AO in AO order. `ids` must be pre-allocated to (get_n_cont(), 4).
    subroutine get_basis_function_ids(self, ids)
        class(AedelstenContext),          intent(in)    :: self
        integer(kind=aedelsten_int_size), intent(inout) :: ids(:,:)
        integer(kind=aedelsten_int_size)                :: rc
        if (size(ids, 1, kind=external_int_size) /= self%get_n_cont() .or. size(ids, 2) /= 4) &
            error stop "Aedelsten: get_basis_function_ids: ids must be (get_n_cont(), 4)"
        rc = get_basis_function_ids_c(self%handle, ids, size(ids, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: get_basis_function_ids failed"
    end subroutine get_basis_function_ids

    ! The number of model functions: the free-atom reference orbitals of the model basis (ANO-RCC,
    ! the model Fockian's), one per component, on the QM centres that are atoms; a core AIMP's share
    ! is left out.
    function get_n_model_functions(self) result(n_model)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n_model
        integer(kind=aedelsten_int_size)    :: n
        n = get_n_model_functions_c(self%handle)
        if (n < 0) error stop "Aedelsten: get_n_model_functions failed"
        n_model = int(n, kind=external_int_size)
    end function get_n_model_functions

    ! The model functions, one row per function in the model basis's AO order: ids(:,1)=centre
    ! (0-based, as get_centre_label takes it), ids(:,2)=l, ids(:,3)=m (as get_basis_function_ids
    ! gives it for a shell of the same type: s and p Cartesian, p as x, y, z = -1, 0, 1; d and up
    ! spherical, -l..l), ids(:,4)=the orbital's ordinal within its l in the reference library
    ! (0-based), counting what a core AIMP covers (a centre whose AIMP projection basis has k
    ! columns at l starts at k); energies(:) the reference orbital energies. `ids` must be
    ! pre-allocated to (get_n_model_functions(), 4), `energies` to get_n_model_functions().
    subroutine get_model_functions(self, ids, energies)
        class(AedelstenContext),          intent(in)    :: self
        integer(kind=aedelsten_int_size), intent(inout) :: ids(:,:)
        real(kind=aedelsten_float_size),  intent(inout) :: energies(:)
        integer(kind=external_int_size)                 :: n_model
        integer(kind=aedelsten_int_size)                :: rc
        n_model = self%get_n_model_functions()
        if (size(ids, 1, kind=external_int_size) /= n_model .or. size(ids, 2) /= 4 &
                .or. size(energies, kind=external_int_size) /= n_model) &
            error stop "Aedelsten: get_model_functions: ids must be (get_n_model_functions(), 4), energies get_n_model_functions()"
        rc = get_model_functions_c(self%handle, ids, size(ids, kind=aedelsten_int_size), energies, &
                                   size(energies, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: get_model_functions failed"
    end subroutine get_model_functions

    ! The overlaps of the model basis: s_model_working(i, nu) = <i|nu>, (get_n_model_functions(),
    ! get_n_cont()), model functions by working basis functions, and, if present, s_model_model
    ! (get_n_model_functions(), get_n_model_functions()), symmetric. Both OVERWRITTEN; they must
    ! not overlap.
    subroutine compute_model_basis_overlaps(self, s_model_working, s_model_model)
        class(AedelstenContext),                                    intent(in)    :: self
        real(kind=aedelsten_float_size),                            intent(inout) :: s_model_working(:,:)
        real(kind=aedelsten_float_size), optional, target, contiguous, intent(inout) :: s_model_model(:,:)
        integer(kind=external_int_size)                                           :: n_model
        integer(kind=aedelsten_int_size)                                          :: rc, s_mm_length
        type(c_ptr)                                                               :: s_mm_ptr
        n_model = self%get_n_model_functions()
        if (size(s_model_working, 1, kind=external_int_size) /= n_model &
                .or. size(s_model_working, 2, kind=external_int_size) /= self%get_n_cont()) &
            error stop "Aedelsten: compute_model_basis_overlaps: s_model_working must be (get_n_model_functions(), get_n_cont())"
        s_mm_ptr = c_null_ptr
        s_mm_length = 0
        if (present(s_model_model)) then
            if (size(s_model_model, 1, kind=external_int_size) /= n_model &
                    .or. size(s_model_model, 2, kind=external_int_size) /= n_model) &
                error stop "Aedelsten: compute_model_basis_overlaps: s_model_model must be (get_n_model_functions(), get_n_model_functions())"
            s_mm_ptr = c_loc(s_model_model)
            s_mm_length = size(s_model_model, kind=aedelsten_int_size)
        end if
        rc = compute_model_basis_overlaps_c(self%handle, s_model_working, size(s_model_working, kind=aedelsten_int_size), &
                                            s_mm_ptr, s_mm_length)
        if (rc /= 0) error stop "Aedelsten: compute_model_basis_overlaps failed"
    end subroutine compute_model_basis_overlaps

    ! The model-Fock start guess S_mw^T diag(E) S_mw, (get_n_cont(), get_n_cont()), symmetric,
    ! OVERWRITTEN.
    subroutine compute_model_fockian(self, fockian)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: fockian(:,:)
        integer(kind=aedelsten_int_size)               :: rc
        if (size(fockian, 1, kind=external_int_size) /= self%get_n_cont() .or. size(fockian, 1) /= size(fockian, 2)) &
            error stop "Aedelsten: compute_model_fockian: fockian must be (get_n_cont(), get_n_cont())"
        rc = compute_model_fockian_c(self%handle, fockian, size(fockian, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: compute_model_fockian failed"
    end subroutine compute_model_fockian

    ! The overlap between this context's basis and other's: s(i, j) = <self_i|other_j>, (get_n_cont(),
    ! other%get_n_cont()), OVERWRITTEN; unscreened.
    subroutine compute_overlap_with(self, other, s)
        class(AedelstenContext),         intent(in)    :: self
        type(AedelstenContext),          intent(in)    :: other
        real(kind=aedelsten_float_size), intent(inout), contiguous :: s(:,:)
        integer(kind=aedelsten_int_size)               :: rc
        if (size(s, 1, kind=external_int_size) /= self%get_n_cont() .or. size(s, 2, kind=external_int_size) /= other%get_n_cont()) &
            error stop "Aedelsten: compute_overlap_with: s must be (get_n_cont(), other%get_n_cont())"
        rc = compute_overlap_with_c(self%handle, other%handle, s, size(s, kind=c_int64_t))
        if (rc /= 0) error stop "Aedelsten: compute_overlap_with failed"
    end subroutine compute_overlap_with

    ! PRIMITIVE_IDS-equivalent: ids(:,1)=centre, ids(:,2)=l, ids(:,3)=shell, parallel to
    ! get_primitives (row i of each belongs together). `ids` must be pre-allocated to
    ! (get_n_primitive_rows(), 3).
    subroutine get_primitive_ids(self, ids)
        class(AedelstenContext),          intent(in)    :: self
        integer(kind=aedelsten_int_size), intent(inout) :: ids(:,:)
        integer(kind=aedelsten_int_size)                :: rc
        if (size(ids, 1, kind=external_int_size) /= self%get_n_primitive_rows() .or. size(ids, 2) /= 3) &
            error stop "Aedelsten: get_primitive_ids: ids must be (get_n_primitive_rows(), 3)"
        rc = get_primitive_ids_c(self%handle, ids, size(ids, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: get_primitive_ids failed"
    end subroutine get_primitive_ids

    ! PRIMITIVES-equivalent: primitives(:,1)=exponent, primitives(:,2)=coefficient, parallel to
    ! get_primitive_ids. `primitives` must be pre-allocated to (get_n_primitive_rows(), 2).
    subroutine get_primitives(self, primitives)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: primitives(:,:)
        integer(kind=aedelsten_int_size)                :: rc
        if (size(primitives, 1, kind=external_int_size) /= self%get_n_primitive_rows() .or. size(primitives, 2) /= 2) &
            error stop "Aedelsten: get_primitives: primitives must be (get_n_primitive_rows(), 2)"
        rc = get_primitives_c(self%handle, primitives, size(primitives, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: get_primitives failed"
    end subroutine get_primitives

    ! centre_idx is 0-based (BasisFunctionId/PrimitiveId's own `centre` field is 1-based -- pass
    ! centre - 1). Returns .false. (and an empty label), not an error stop, when centre_idx has
    ! no label (out of range, or an orthogonality ghost centre) -- a legitimate query outcome,
    ! unlike everywhere else in this file.
    function get_centre_label(self, centre_idx, label) result(found)
        class(AedelstenContext),        intent(in)  :: self
        integer(kind=external_int_size), intent(in)  :: centre_idx
        character(len=:), allocatable,  intent(out) :: label
        logical                          :: found
        character(kind=c_char)           :: buf(64)
        integer(kind=aedelsten_int_size) :: rc
        rc = get_centre_label_c(self%handle, int(centre_idx, kind=aedelsten_int_size), buf, size(buf, kind=aedelsten_int_size))
        found = (rc == 0)
        if (found) then
            label = from_c_string_buf(buf)
        else
            label = ""
        end if
    end function get_centre_label

    ! convention: "molcas", "dalton", or "dirac" (case-insensitive) -- see
    ! cart2sph::SharedCartesianReference's own doc comment for what these mean. Rescales `matrix`
    ! (get_n_cont() x n_cols, AO axis = rows) in place; rescaled (if present) reports whether
    ! anything actually changed (.false. when every factor was exactly 1.0 -- e.g. no Cartesian
    ! d/f/g/... shells in this basis).
    subroutine rescale_cartesian_ao_rows(self, convention, n_cols, matrix, rescaled)
        class(AedelstenContext),         intent(in)    :: self
        character(len=*),                intent(in)    :: convention
        integer(kind=external_int_size), intent(in)    :: n_cols
        real(kind=aedelsten_float_size), intent(inout) :: matrix(:,:)
        logical,                         intent(out), optional :: rescaled
        integer(kind=aedelsten_int_size) :: rc, rescaled_int
        if (size(matrix, 1, kind=external_int_size) /= self%get_n_cont() .or. &
            size(matrix, 2, kind=external_int_size) /= n_cols) &
            error stop "Aedelsten: rescale_cartesian_ao_rows: matrix must be (get_n_cont(), n_cols)"
        rc = rescale_cartesian_ao_rows_c(self%handle, &
            query_i32_checked("cartesian_convention_" // trim(lowercase(convention))), &
            int(n_cols, kind=aedelsten_int_size), matrix, size(matrix, kind=aedelsten_int_size), rescaled_int)
        if (rc /= 0) error stop "Aedelsten: rescale_cartesian_ao_rows failed"
        if (present(rescaled)) rescaled = (rescaled_int /= 0)
    end subroutine rescale_cartesian_ao_rows

    ! The inverse of rescale_cartesian_ao_rows: matrix (get_n_cont() x n_cols) from convention's Cartesian
    ! normalization to this library's, in place; rescaled (if present) whether anything changed.
    subroutine unrescale_cartesian_ao_rows(self, convention, n_cols, matrix, rescaled)
        class(AedelstenContext),         intent(in)    :: self
        character(len=*),                intent(in)    :: convention
        integer(kind=external_int_size), intent(in)    :: n_cols
        real(kind=aedelsten_float_size), intent(inout) :: matrix(:,:)
        logical,                         intent(out), optional :: rescaled
        integer(kind=aedelsten_int_size) :: rc, rescaled_int
        if (size(matrix, 1, kind=external_int_size) /= self%get_n_cont() .or. &
            size(matrix, 2, kind=external_int_size) /= n_cols) &
            error stop "Aedelsten: unrescale_cartesian_ao_rows: matrix must be (get_n_cont(), n_cols)"
        rc = unrescale_cartesian_ao_rows_c(self%handle, &
            query_i32_checked("cartesian_convention_" // trim(lowercase(convention))), &
            int(n_cols, kind=aedelsten_int_size), matrix, size(matrix, kind=aedelsten_int_size), rescaled_int)
        if (rc /= 0) error stop "Aedelsten: unrescale_cartesian_ao_rows failed"
        if (present(rescaled)) rescaled = (rescaled_int /= 0)
    end subroutine unrescale_cartesian_ao_rows

    ! The number of centres: real atoms, ghost centres and orthogonality ghosts.
    function get_n_centres(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        integer(kind=c_int64_t) :: count
        count = get_n_centres_c(self%handle)
        if (count < 0) error stop "Aedelsten: get_n_centres failed"
        n = int(count, kind=external_int_size)
    end function get_n_centres

    ! Every centre in centre order: atomic numbers (n), positions (3, n; bohr) and kinds (n; compare with
    ! aedelsten_constant("atom_kind_real"), "atom_kind_ghost" or "atom_kind_orthogonality_ghost").
    subroutine get_centre_geometry(self, atomic_numbers, positions, kinds)
        class(AedelstenContext),         intent(in)    :: self
        integer(kind=c_int32_t),         intent(inout) :: atomic_numbers(:), kinds(:)
        real(kind=aedelsten_float_size), intent(inout) :: positions(:,:)
        integer(kind=int64) :: n
        integer(kind=aedelsten_int_size) :: rc
        n = int(self%get_n_centres(), kind=int64)
        if (.not. (is_contiguous(atomic_numbers) .and. is_contiguous(positions) .and. is_contiguous(kinds))) &
            error stop "Aedelsten: get_centre_geometry: the outputs must be contiguous"
        if (size(atomic_numbers, kind=int64) /= n .or. size(positions, 1) /= 3 .or. size(positions, 2, kind=int64) /= n &
            .or. size(kinds, kind=int64) /= n) &
            error stop "Aedelsten: get_centre_geometry: the outputs must be (n), (3, n) and (n), n centres"
        rc = get_centre_geometry_c(self%handle, atomic_numbers, positions, kinds, int(n, c_int64_t))
        if (rc /= 0) error stop "Aedelsten: get_centre_geometry failed"
    end subroutine get_centre_geometry

    ! Every centre's nuclear charge (its nucleus's or its core AIMP's; 0 for a ghost centre or an orthogonality ghost).
    subroutine get_nuclear_charges(self, charges)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: charges(:)
        integer(kind=aedelsten_int_size) :: rc
        if (size(charges, kind=external_int_size) /= self%get_n_centres()) &
            error stop "Aedelsten: get_nuclear_charges: charges must have get_n_centres() entries"
        rc = get_nuclear_charges_c(self%handle, charges, int(size(charges), c_int64_t))
        if (rc /= 0) error stop "Aedelsten: get_nuclear_charges failed"
    end subroutine get_nuclear_charges

    ! Same as rescale_cartesian_ao_rows, but for matrix (n_rows x get_n_cont(), AO axis = columns).
    subroutine rescale_cartesian_ao_columns(self, convention, n_rows, matrix, rescaled)
        class(AedelstenContext),         intent(in)    :: self
        character(len=*),                intent(in)    :: convention
        integer(kind=external_int_size), intent(in)    :: n_rows
        real(kind=aedelsten_float_size), intent(inout) :: matrix(:,:)
        logical,                         intent(out), optional :: rescaled
        integer(kind=aedelsten_int_size) :: rc, rescaled_int
        if (size(matrix, 1, kind=external_int_size) /= n_rows .or. &
            size(matrix, 2, kind=external_int_size) /= self%get_n_cont()) &
            error stop "Aedelsten: rescale_cartesian_ao_columns: matrix must be (n_rows, get_n_cont())"
        rc = rescale_cartesian_ao_columns_c(self%handle, &
            query_i32_checked("cartesian_convention_" // trim(lowercase(convention))), &
            int(n_rows, kind=aedelsten_int_size), matrix, size(matrix, kind=aedelsten_int_size), rescaled_int)
        if (rc /= 0) error stop "Aedelsten: rescale_cartesian_ao_columns failed"
        if (present(rescaled)) rescaled = (rescaled_int /= 0)
    end subroutine rescale_cartesian_ao_columns

    ! ── The X2C picture-change cache ─────────────────────────────────

    function has_picture_change(self) result(flag)
        class(AedelstenContext), intent(in) :: self
        logical                             :: flag
        flag = has_picture_change_c(self%handle) == 1
    end function has_picture_change

    ! The cached X2C decoupling transforms, each (n_prim_unique, n_prim_unique): u_large for plain operators, u_small
    ! for their pXp counterparts.
    subroutine get_picture_change_transforms(self, u_large, u_small)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: u_large(:,:), u_small(:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (.not. is_contiguous(u_large) .or. .not. is_contiguous(u_small)) error stop "Aedelsten: array is not contiguous"
        if (any(shape(u_large) /= shape(u_small))) error stop "Aedelsten: u_large and u_small must have the same shape"
        rc = get_picture_change_transforms_c(self%handle, u_large, u_small, size(u_large, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: get_picture_change_transforms failed"
    end subroutine get_picture_change_transforms

    ! Supplies the decoupling transforms, as get_picture_change_transforms gives them, instead of deriving them: the
    ! next construct_x2c_one_electron_hamiltonian holds them (the X2C Hamiltonian with another geometry's decoupling).
    ! c_light as they were made with (absent: the library default).
    subroutine set_picture_change_transforms(self, u_large, u_small, c_light)
        class(AedelstenContext),         intent(in) :: self
        real(kind=aedelsten_float_size), intent(in) :: u_large(:,:), u_small(:,:)
        real(kind=aedelsten_float_size), intent(in), optional :: c_light
        real(kind=aedelsten_float_size) :: c_light_wire
        integer(kind=aedelsten_int_size) :: rc
        if (.not. is_contiguous(u_large) .or. .not. is_contiguous(u_small)) error stop "Aedelsten: array is not contiguous"
        if (any(shape(u_large) /= shape(u_small))) error stop "Aedelsten: u_large and u_small must have the same shape"
        c_light_wire = -1.0_aedelsten_float_size
        if (present(c_light)) c_light_wire = c_light
        rc = set_picture_change_transforms_c(self%handle, u_large, u_small, size(u_large, kind=aedelsten_int_size), c_light_wire)
        if (rc /= 0) error stop "Aedelsten: set_picture_change_transforms failed"
    end subroutine set_picture_change_transforms

    ! Holds the densities of the X2C gradient with the decoupling held, from the symmetric density (n_cont, n_cont),
    ! for compute_x2c_frozen_kinetic_gradient and compute_x2c_frozen_potential_gradient; replaces densities already
    ! held. A new decoupling, free_picture_change or set_centre_positions drops them.
    subroutine set_x2c_frozen_density(self, density)
        class(AedelstenContext),         intent(in) :: self
        real(kind=aedelsten_float_size), intent(in) :: density(:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (.not. is_contiguous(density)) error stop "Aedelsten: density array is not contiguous"
        if (size(density, 1) /= size(density, 2) .or. size(density, 1, kind=aedelsten_int_size) /= get_n_cont_c(self%handle)) then
            error stop "Aedelsten: density must be n_cont x n_cont"
        end if
        rc = set_x2c_frozen_density_c(self%handle, density, size(density, 1, kind=aedelsten_int_size))
        if (rc /= 0) error stop "Aedelsten: set_x2c_frozen_density failed"
    end subroutine set_x2c_frozen_density

    ! Frees the densities set_x2c_frozen_density holds.
    subroutine free_x2c_frozen_density(self)
        class(AedelstenContext), intent(in) :: self
        call free_x2c_frozen_density_c(self%handle)
    end subroutine free_x2c_frozen_density

    ! The kinetic part of the X2C one-electron energy tr(D h)'s gradient with the decoupling held, gradient (3,
    ! n_centres), from the densities of set_x2c_frozen_density. With compute_x2c_frozen_potential_gradient for each
    ! potential in the transformation, the sum is the whole gradient. active as in the other gradients.
    subroutine compute_x2c_frozen_kinetic_gradient(self, gradient, active)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: gradient(:,:)
        logical, intent(in), optional                  :: active(:)
        integer(kind=aedelsten_int_size) :: n_centres, rc
        integer(kind=c_int), allocatable, target :: active_flags(:)
        type(c_ptr) :: active_ptr
        call check_classical_gradient(self, gradient, active, n_centres, active_flags, active_ptr)
        rc = compute_x2c_frozen_kinetic_gradient_c(self%handle, gradient, n_centres, active_ptr)
        if (rc /= 0) error stop "Aedelsten: compute_x2c_frozen_kinetic_gradient failed"
    end subroutine compute_x2c_frozen_kinetic_gradient

    ! The part of the X2C one-electron energy's gradient with the decoupling held of one potential operator that went
    ! into V and p.Vp ("nuclear_attraction", "external_multipole", "embedding_aimp_coulomb", ...), gradient (3,
    ! n_centres); the caller sums the parts. active as in the other gradients.
    subroutine compute_x2c_frozen_potential_gradient(self, gradient, operator, active)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: gradient(:,:)
        character(len=*), intent(in)                   :: operator
        logical, intent(in), optional                  :: active(:)
        integer(kind=aedelsten_int_size) :: n_centres, rc, operator_int
        integer(kind=c_int), allocatable, target :: active_flags(:)
        type(c_ptr) :: active_ptr
        call check_classical_gradient(self, gradient, active, n_centres, active_flags, active_ptr)
        operator_int = resolve_operator(operator, expect_components=.false.)
        rc = compute_x2c_frozen_potential_gradient_c(self%handle, gradient, n_centres, operator_int, active_ptr)
        if (rc /= 0) error stop "Aedelsten: compute_x2c_frozen_potential_gradient failed"
    end subroutine compute_x2c_frozen_potential_gradient

    ! A one-electron operator in the X2C picture, the scalar part: out = u_large^T op_plain u_large + u_small^T op_pxp
    ! u_small (op_plain X, op_pxp p.X p), (n_prim_unique, n_prim_unique), the raw planes of
    ! compute_one_electron_picture_change_integrals; symmetrised. Needs the X2C decoupling.
    subroutine transform_one_electron_operator_matrix(self, op_plain, op_pxp, out)
        class(AedelstenContext),         intent(in)                :: self
        real(kind=aedelsten_float_size), intent(in), contiguous    :: op_plain(:,:), op_pxp(:,:)
        real(kind=aedelsten_float_size), intent(inout), contiguous :: out(:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (any(shape(op_plain) /= shape(out)) .or. any(shape(op_pxp) /= shape(out)) .or. size(out, 1) /= size(out, 2)) &
            error stop "Aedelsten: transform_one_electron_operator: op_plain, op_pxp and out must be the same (n, n)"
        rc = transform_one_electron_operator_c(self%handle, op_plain, op_pxp, out, &
            to_wire_i32(size(out, 1, kind=external_int_size), "n"), 1_aedelsten_int_size)
        if (rc /= 0) error stop "Aedelsten: transform_one_electron_operator failed"
    end subroutine transform_one_electron_operator_matrix

    ! The same per plane, (n_prim_unique, n_prim_unique, n_planes).
    subroutine transform_one_electron_operator_planes(self, op_plain, op_pxp, out)
        class(AedelstenContext),         intent(in)                :: self
        real(kind=aedelsten_float_size), intent(in), contiguous    :: op_plain(:,:,:), op_pxp(:,:,:)
        real(kind=aedelsten_float_size), intent(inout), contiguous :: out(:,:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (any(shape(op_plain) /= shape(out)) .or. any(shape(op_pxp) /= shape(out)) .or. size(out, 1) /= size(out, 2)) &
            error stop "Aedelsten: transform_one_electron_operator: op_plain, op_pxp and out must be the same (n, n, n_planes)"
        rc = transform_one_electron_operator_c(self%handle, op_plain, op_pxp, out, &
            to_wire_i32(size(out, 1, kind=external_int_size), "n"), to_wire_i32(size(out, 3, kind=external_int_size), "n_planes"))
        if (rc /= 0) error stop "Aedelsten: transform_one_electron_operator failed"
    end subroutine transform_one_electron_operator_planes

    ! The spin planes of a one-electron operator in the X2C picture: out(:,:,k) = u_small^T op_pxp(:,:,k) u_small,
    ! (n_prim_unique, n_prim_unique, n_planes), the raw (p X x p)_k planes
    ! (compute_one_electron_picture_change_integrals with spin_orbit), other raw spin planes such as AMFI's added first.
    ! Antisymmetric, no symmetrisation. Needs the X2C decoupling.
    subroutine transform_one_electron_spin_operator(self, op_pxp, out)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(in)    :: op_pxp(:,:,:)
        real(kind=aedelsten_float_size), intent(inout) :: out(:,:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (.not. is_contiguous(op_pxp) .or. .not. is_contiguous(out)) error stop "Aedelsten: array is not contiguous"
        if (any(shape(op_pxp) /= shape(out))) error stop "Aedelsten: op_pxp and out must have the same shape"
        if (size(op_pxp, 1) /= size(op_pxp, 2)) error stop "Aedelsten: op_pxp planes must be square"
        rc = transform_one_electron_spin_operator_c(self%handle, op_pxp, out, &
            to_wire_i32(size(op_pxp, 1, kind=external_int_size), "n"), to_wire_i32(size(op_pxp, 3, kind=external_int_size), "n_planes"))
        if (rc /= 0) error stop "Aedelsten: transform_one_electron_spin_operator failed"
    end subroutine transform_one_electron_spin_operator

    ! A one-electron operator X in the X2C picture, X_2c = scalar (x) 1 + i sum_k sigma_k (x) spin_k: scalar(:,:,c) =
    ! u_large^T X_c u_large + u_small^T (p.X_c p) u_small, spin(:,:,3(c-1)+k) = u_small^T (p X_c x p)_k u_small, over
    ! the contracted basis (contracted, n = n_cont) or the primitive one (n = n_prim_unique). scalar (n, n, K), spin
    ! (n, n, 3K), K = 1 for a plain operator, the moment's component count for one with origin and order. Needs the
    ! X2C decoupling and the primitive shell pairs.
    subroutine compute_picture_changed_one_electron_operator(self, scalar, spin, operator, contracted, origin, order)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: scalar(:,:,:), spin(:,:,:)
        character(len=*), intent(in)                   :: operator
        logical, intent(in)                            :: contracted
        real(kind=aedelsten_float_size), intent(in), optional  :: origin(3)
        integer(kind=external_int_size), intent(in), optional  :: order
        integer(kind=aedelsten_int_size) :: n_wire, n_components, order_wire, operator_int, contracted_int, rc
        real(kind=aedelsten_float_size)  :: origin_x, origin_y, origin_z

        if (.not. is_contiguous(scalar) .or. .not. is_contiguous(spin)) error stop "Aedelsten: array is not contiguous"
        if (present(order) .neqv. present(origin)) then
            error stop "Aedelsten: order and origin must both be present, or both absent"
        end if
        if (size(scalar, 1) /= size(scalar, 2) .or. size(spin, 1) /= size(scalar, 1) .or. size(spin, 2) /= size(scalar, 2)) then
            error stop "Aedelsten: scalar and spin must be (n, n, ...) with the same n"
        end if
        n_wire = to_wire_i32(size(scalar, 1, kind=external_int_size), "n")
        if (present(order)) then
            order_wire = to_wire_i32(order, "order")
            n_components = compute_moment_component_count_c(order_wire)
            origin_x = origin(1)
            origin_y = origin(2)
            origin_z = origin(3)
        else
            order_wire = 0
            n_components = 1
            origin_x = 0.0_aedelsten_float_size
            origin_y = 0.0_aedelsten_float_size
            origin_z = 0.0_aedelsten_float_size
        end if
        if (size(scalar, 3) /= n_components .or. size(spin, 3) /= 3 * n_components) then
            error stop "Aedelsten: scalar needs one plane per operator component, spin three"
        end if
        operator_int = resolve_operator(operator, expect_components=present(order))
        contracted_int = 0_aedelsten_int_size
        if (contracted) contracted_int = 1_aedelsten_int_size
        rc = compute_picture_changed_one_electron_operator_c(self%handle, scalar, spin, n_wire, operator_int, n_components, &
            origin_x, origin_y, origin_z, order_wire, contracted_int)
        if (rc /= 0) error stop "Aedelsten: compute_picture_changed_one_electron_operator failed"
    end subroutine compute_picture_changed_one_electron_operator

    ! The atomic mean-field two-electron spin-orbit integrals (AMFI) over the primitive basis, output (n_prim_unique,
    ! n_prim_unique, 6): spin-same-orbit x, y, z, then spin-other-orbit exchange x, y, z, raw (no prefactor,
    ! antisymmetric; Breit-Pauli is C^T (SSO + SOO) C / 4c^2). Needs the primitive shell pairs and the scratch.
    subroutine compute_amfi_integrals(self, output)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: output(:,:,:)
        integer(kind=aedelsten_int_size) :: rc
        if (.not. is_contiguous(output)) error stop "Aedelsten: array is not contiguous"
        if (size(output, 1) /= size(output, 2) .or. size(output, 3) /= 6) error stop "Aedelsten: AMFI output must be (n, n, 6)"
        rc = compute_amfi_integrals_c(self%handle, output, to_wire_i32(size(output, 1, kind=external_int_size), "n"))
        if (rc /= 0) error stop "Aedelsten: compute_amfi_integrals failed"
    end subroutine compute_amfi_integrals

    ! Frees the X2C decoupling transforms.
    subroutine free_picture_change(self)
        class(AedelstenContext), intent(inout) :: self
        call free_picture_change_c(self%handle)
    end subroutine free_picture_change

    ! ── Point-group symmetry ──────────────────────────────────────────

    ! Finds the point group of the QM centres (point_group absent) or uses the named one (e.g. "C2v": the full group or
    ! a subgroup; principal_axis picks a subgroup's orientation). Tolerances absent: the defaults (1e-8 Bohr, 1e-10
    ! relative, no warning). Stops if the geometry is not symmetric to the geometry tolerance (the reason on stderr).
    subroutine set_symmetry(self, point_group, principal_axis, geometry_tolerance, strict_tolerance, average_warning)
        class(AedelstenContext),         intent(inout)                  :: self
        character(len=*),                intent(in), optional           :: point_group
        real(kind=aedelsten_float_size), intent(in), optional, target   :: principal_axis(3)
        real(kind=aedelsten_float_size), intent(in), optional           :: geometry_tolerance, strict_tolerance, average_warning
        character(len=:, kind=c_char), allocatable, target :: name
        real(kind=aedelsten_float_size), target :: axis(3)
        type(c_ptr) :: name_ptr, axis_ptr
        integer(kind=aedelsten_int_size) :: rc
        name_ptr = c_null_ptr
        axis_ptr = c_null_ptr
        if (present(point_group)) then
            name = to_c_string(point_group)
            name_ptr = c_loc(name)
        end if
        if (present(principal_axis)) then
            axis = principal_axis
            axis_ptr = c_loc(axis)
        end if
        rc = set_symmetry_c(self%handle, name_ptr, axis_ptr, optional_or(geometry_tolerance), optional_or(strict_tolerance), &
            optional_or(average_warning))
        if (rc /= 0) error stop "Aedelsten: set_symmetry failed (the reason is on stderr)"
    contains
        ! x, or the sentinel of the library's default when absent.
        pure function optional_or(x) result(wire)
            real(kind=aedelsten_float_size), intent(in), optional :: x
            real(kind=aedelsten_float_size) :: wire
            wire = -1.0_aedelsten_float_size ! Sentinel: the library's default.
            if (present(x)) wire = x
        end function optional_or
    end subroutine set_symmetry

    ! Frees the point group and its AO representation.
    subroutine free_symmetry(self)
        class(AedelstenContext), intent(inout) :: self
        call free_symmetry_c(self%handle)
    end subroutine free_symmetry

    ! average_warning is 0 when there is none.
    subroutine get_symmetry_options(self, geometry_tolerance, strict_tolerance, average_warning)
        class(AedelstenContext),         intent(in)  :: self
        real(kind=aedelsten_float_size), intent(out) :: geometry_tolerance, strict_tolerance, average_warning
        real(kind=aedelsten_float_size) :: options(3)
        if (get_symmetry_options_c(self%handle, options) /= 0) error stop "Aedelsten: get_symmetry_options: no symmetry set"
        geometry_tolerance = options(1)
        strict_tolerance = options(2)
        average_warning = options(3)
    end subroutine get_symmetry_options

    ! The report of the last cleaning of a library output (e.g. "oneint_overlap", "oneint_external_multipole",
    ! "x2c_one_electron_hamiltonian") in the primitive or the contracted basis: report = (irrep, largest change, norm
    ! removed, norm before). .false. if it was not cleaned.
    function get_symmetrisation_report(self, name, primitive, report) result(found)
        class(AedelstenContext),         intent(in)    :: self
        character(len=*),                intent(in)    :: name
        logical,                         intent(in)    :: primitive
        real(kind=aedelsten_float_size), intent(inout) :: report(4)
        logical                          :: found
        integer(kind=aedelsten_int_size) :: rc, found_wire, primitive_wire
        primitive_wire = merge(1_aedelsten_int_size, 0_aedelsten_int_size, primitive)
        rc = get_symmetrisation_report_c(self%handle, to_c_string(name), primitive_wire, found_wire, report)
        if (rc /= 0) error stop "Aedelsten: get_symmetrisation_report failed"
        found = found_wire == 1
    end function get_symmetrisation_report

    ! The point group in use, e.g. "C2v".
    function get_point_group_name(self) result(group)
        class(AedelstenContext), intent(in) :: self
        character(len=:), allocatable       :: group
        character(kind=c_char)              :: buf(16)
        if (get_point_group_name_c(self%handle, buf, size(buf, kind=aedelsten_int_size)) /= 0) &
            error stop "Aedelsten: get_point_group_name: no symmetry set"
        group = from_c_string_buf(buf)
    end function get_point_group_name

    ! -1 if no symmetry is set.
    function get_n_irreps(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        n = int(get_n_irreps_c(self%handle), kind=external_int_size)
    end function get_n_irreps

    ! Irrep irrep's name (0-based, character-table order).
    function get_irrep_name(self, irrep) result(name)
        class(AedelstenContext),         intent(in) :: self
        integer(kind=external_int_size), intent(in) :: irrep
        character(len=:), allocatable               :: name
        character(kind=c_char)                      :: buf(16)
        if (get_irrep_name_c(self%handle, to_wire_i32(irrep, "irrep"), buf, size(buf, kind=aedelsten_int_size)) /= 0) &
            error stop "Aedelsten: get_irrep_name: no such irrep"
        name = from_c_string_buf(buf)
    end function get_irrep_name

    ! Irrep irrep's dimension (0-based).
    function get_irrep_dimension(self, irrep) result(d)
        class(AedelstenContext),         intent(in) :: self
        integer(kind=external_int_size), intent(in) :: irrep
        integer(kind=external_int_size)             :: d
        d = int(get_irrep_dimension_c(self%handle, to_wire_i32(irrep, "irrep")), kind=external_int_size)
    end function get_irrep_dimension

    ! -1 if the group has no irrep of that name (case-insensitive).
    function get_irrep_index(self, name) result(k)
        class(AedelstenContext), intent(in) :: self
        character(len=*),        intent(in) :: name
        integer(kind=external_int_size)     :: k
        k = int(get_irrep_index_c(self%handle, to_c_string(name)), kind=external_int_size)
    end function get_irrep_index

    ! The index (0-based) of the totally symmetric irrep.
    function get_totally_symmetric_irrep(self) result(k)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: k
        k = int(get_totally_symmetric_irrep_c(self%handle), kind=external_int_size)
    end function get_totally_symmetric_irrep

    ! The irrep of x^lx y^ly z^lz about the centre of mass; -1 if it is not a single one-dimensional irrep.
    function get_cartesian_irrep(self, lx, ly, lz) result(k)
        class(AedelstenContext),         intent(in) :: self
        integer(kind=external_int_size), intent(in) :: lx, ly, lz
        integer(kind=external_int_size)             :: k
        k = int(get_cartesian_irrep_c(self%handle, to_wire_i32(lx, "lx"), to_wire_i32(ly, "ly"), to_wire_i32(lz, "lz")), &
            kind=external_int_size)
    end function get_cartesian_irrep

    ! axes(:, k) is the group's standard axis k in the input's coordinates (what the irrep names refer to).
    subroutine get_symmetry_standard_axes(self, axes)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(inout) :: axes(3, 3)
        if (get_symmetry_standard_axes_c(self%handle, axes) /= 0) &
            error stop "Aedelsten: get_symmetry_standard_axes: no symmetry set"
    end subroutine get_symmetry_standard_axes

    ! The group's order h: operations are numbered 0..h-1 in a fixed order; -1 if no symmetry is set.
    function get_symmetry_operation_count(self) result(h)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: h
        h = int(get_symmetry_operation_count_c(self%handle), kind=external_int_size)
    end function get_symmetry_operation_count

    ! Operation op (0-based) in the input's axes: a displacement or gradient v of a centre goes to matmul(matrix, v)
    ! at its image (get_symmetry_centre_images). Orthogonal.
    subroutine get_symmetry_operation_matrix(self, op, matrix)
        class(AedelstenContext),         intent(in)    :: self
        integer(kind=external_int_size), intent(in)    :: op
        real(kind=aedelsten_float_size), intent(inout) :: matrix(3, 3)
        if (get_symmetry_operation_matrix_c(self%handle, to_wire_i32(op, "op"), matrix) /= 0) &
            error stop "Aedelsten: get_symmetry_operation_matrix: no symmetry set, or no such operation"
    end subroutine get_symmetry_operation_matrix

    ! The QM centres the group acts on (ghosts included; the gradient centres first); -1 if no symmetry is set.
    function get_n_symmetry_centres(self) result(n)
        class(AedelstenContext), intent(in) :: self
        integer(kind=external_int_size)     :: n
        n = int(get_n_symmetry_centres_c(self%handle), kind=external_int_size)
    end function get_n_symmetry_centres

    ! images(a + 1): the centre (0-based, as every index the library returns) operation op takes centre a to;
    ! size(images) = get_n_symmetry_centres().
    subroutine get_symmetry_centre_images(self, op, images)
        class(AedelstenContext),         intent(in)    :: self
        integer(kind=external_int_size), intent(in)    :: op
        integer(kind=external_int_size), intent(inout) :: images(:)
        integer(kind=aedelsten_int_size), allocatable  :: wire(:)
        allocate(wire(size(images)))
        if (get_symmetry_centre_images_c(self%handle, to_wire_i32(op, "op"), wire, &
                to_wire_i32(int(size(images), kind=external_int_size), "size(images)")) /= 0) &
            error stop "Aedelsten: get_symmetry_centre_images: no such operation, or images of the wrong size"
        images = int(wire, kind=external_int_size)
    end subroutine get_symmetry_centre_images

    ! How many AO functions span each irrep; size(counts) = the number of irreps.
    subroutine get_ao_irrep_counts(self, counts)
        class(AedelstenContext),         intent(in)    :: self
        integer(kind=external_int_size), intent(inout) :: counts(:)
        integer(kind=aedelsten_int_size), allocatable  :: wire(:)
        allocate(wire(size(counts)))
        if (get_ao_irrep_counts_c(self%handle, wire, to_wire_i32(int(size(counts), kind=external_int_size), "size(counts)")) /= 0) &
            error stop "Aedelsten: get_ao_irrep_counts: no symmetry set, or counts of the wrong size"
        counts = int(wire, kind=external_int_size)
    end subroutine get_ao_irrep_counts

    ! Operation op (0-based) applied to MO coefficients (n_cont, n_mo): output the coefficients of each R phi.
    subroutine apply_symmetry_operation(self, op, coefficients, output)
        class(AedelstenContext),         intent(in)                :: self
        integer(kind=external_int_size), intent(in)                :: op
        real(kind=aedelsten_float_size), intent(in), contiguous    :: coefficients(:,:)
        real(kind=aedelsten_float_size), intent(inout), contiguous :: output(:,:)
        if (size(coefficients, 1, kind=external_int_size) /= self%get_n_cont() .or. any(shape(output) /= shape(coefficients))) &
            error stop "Aedelsten: apply_symmetry_operation: coefficients and output must be (get_n_cont(), n_mo)"
        if (apply_symmetry_operation_c(self%handle, to_wire_i32(op, "op"), coefficients, output, &
                to_wire_i32(int(size(coefficients), kind=external_int_size), "size(coefficients)"), &
                to_wire_i32(int(size(coefficients, 2), kind=external_int_size), "n_mo")) /= 0) &
            error stop "Aedelsten: apply_symmetry_operation: no symmetry set, or no such operation"
    end subroutine apply_symmetry_operation

    ! The character of irrep irrep at operation op (both 0-based).
    function get_irrep_character(self, irrep, op) result(character)
        class(AedelstenContext),         intent(in) :: self
        integer(kind=external_int_size), intent(in) :: irrep, op
        real(kind=aedelsten_float_size)             :: character
        if (get_irrep_character_c(self%handle, to_wire_i32(irrep, "irrep"), to_wire_i32(op, "op"), character) /= 0) &
            error stop "Aedelsten: get_irrep_character: no symmetry set, or no such irrep or operation"
    end function get_irrep_character

    ! Gamma(R) of irrep irrep at operation op (both 0-based), (d, d): R f_k = sum_j matrix(j, k) f_j, for the partner
    ! components symmetrise_orbitals aligns; aligned .false. (matrix untouched) when the irrep has none.
    subroutine get_irrep_matrix(self, irrep, op, matrix, aligned)
        class(AedelstenContext),         intent(in)                :: self
        integer(kind=external_int_size), intent(in)                :: irrep, op
        real(kind=aedelsten_float_size), intent(inout), contiguous :: matrix(:,:)
        logical,                         intent(out)               :: aligned
        integer(kind=aedelsten_int_size) :: flag
        if (get_irrep_matrix_c(self%handle, to_wire_i32(irrep, "irrep"), to_wire_i32(op, "op"), matrix, &
                to_wire_i32(int(size(matrix), kind=external_int_size), "size(matrix)"), flag) /= 0) &
            error stop "Aedelsten: get_irrep_matrix: no symmetry set, no such irrep or operation, or matrix not (d, d)"
        aligned = flag == 1
    end subroutine get_irrep_matrix

    ! Cleans the totally symmetric (n_cont, n_cont) matrix in place (a Fock matrix, a density). average absent or
    ! .false.: strict, refused (matrix untouched) above the strict tolerance; .true.: averaged over the group,
    ! whatever it removes. report: (irrep, largest change, norm removed, norm before). A refusal stops the program
    ! unless `accepted` is present, which then reports it.
    subroutine symmetrise_matrix(self, matrix, average, report, accepted)
        class(AedelstenContext),         intent(in)              :: self
        real(kind=aedelsten_float_size), intent(inout)           :: matrix(:,:)
        logical,                         intent(in),  optional   :: average
        real(kind=aedelsten_float_size), intent(inout), optional :: report(4)
        logical,                         intent(out), optional   :: accepted
        real(kind=aedelsten_float_size)  :: wire_report(4)
        integer(kind=aedelsten_int_size) :: rc, average_wire
        if (.not. is_contiguous(matrix)) error stop "Aedelsten: matrix is not contiguous"
        average_wire = 0
        if (present(average)) average_wire = merge(1_aedelsten_int_size, 0_aedelsten_int_size, average)
        rc = symmetrise_matrix_c(self%handle, matrix, size(matrix, kind=aedelsten_int_size), average_wire, wire_report)
        if (present(accepted)) then
            accepted = rc == 0
        else if (rc /= 0) then
            error stop "Aedelsten: symmetrise_matrix refused the matrix (the reason is on stderr)"
        end if
        if (present(report) .and. rc == 0) report = wire_report
    end subroutine symmetrise_matrix

    ! What cleaning `matrix` as `irrep` (0-based; absent: whichever irrep carries most of it) would remove; nothing
    ! changed. report: (irrep, largest change, norm removed, norm before).
    subroutine analyse_matrix(self, matrix, report, irrep)
        class(AedelstenContext),         intent(in)           :: self
        real(kind=aedelsten_float_size), intent(in)           :: matrix(:,:)
        real(kind=aedelsten_float_size), intent(inout)        :: report(4)
        integer(kind=external_int_size), intent(in), optional :: irrep
        integer(kind=aedelsten_int_size) :: irrep_wire
        if (.not. is_contiguous(matrix)) error stop "Aedelsten: matrix is not contiguous"
        irrep_wire = -1
        if (present(irrep)) irrep_wire = to_wire_i32(irrep, "irrep")
        if (analyse_matrix_c(self%handle, matrix, size(matrix, kind=aedelsten_int_size), irrep_wire, report) /= 0) &
            error stop "Aedelsten: analyse_matrix failed"
    end subroutine analyse_matrix

    ! The norm of each irrep's part of the (n_cont, n_cont) matrix: norms must hold get_n_irreps() values.
    subroutine irrep_norms(self, matrix, norms)
        class(AedelstenContext),         intent(in)    :: self
        real(kind=aedelsten_float_size), intent(in)    :: matrix(:,:)
        real(kind=aedelsten_float_size), intent(inout) :: norms(:)
        if (.not. is_contiguous(matrix)) error stop "Aedelsten: matrix is not contiguous"
        if (get_irrep_norms_c(self%handle, matrix, size(matrix, kind=aedelsten_int_size), norms, &
                size(norms, kind=aedelsten_int_size)) /= 0) error stop "Aedelsten: irrep_norms failed"
    end subroutine irrep_norms

    ! Symmetrizes MO coefficients in place: orbitals (n_cont, n_mo), one MO per column (any n_mo); irreps(n_mo) gets
    ! each MO's irrep (0-based), components(n_mo) its partner component, dropped the largest coefficient norm
    ! projected away. Not re-orthonormalized.
    subroutine symmetrise_orbitals(self, orbitals, irreps, components, dropped)
        class(AedelstenContext),          intent(in)              :: self
        real(kind=aedelsten_float_size),  intent(inout)           :: orbitals(:,:)
        integer(kind=aedelsten_int_size), intent(inout)           :: irreps(:)
        integer(kind=aedelsten_int_size), intent(inout), optional :: components(:)
        real(kind=aedelsten_float_size),  intent(out),   optional :: dropped
        integer(kind=aedelsten_int_size), allocatable :: component_buffer(:)
        real(kind=aedelsten_float_size)  :: dropped_wire
        integer(kind=aedelsten_int_size) :: rc, n_mo, length
        if (.not. is_contiguous(orbitals)) error stop "Aedelsten: orbitals is not contiguous"
        n_mo = size(orbitals, 2, kind=aedelsten_int_size)
        length = size(orbitals, kind=aedelsten_int_size)
        if (size(irreps, kind=aedelsten_int_size) /= n_mo) error stop "Aedelsten: symmetrise_orbitals: irreps must hold n_mo values"
        if (present(components)) then
            if (size(components, kind=aedelsten_int_size) /= n_mo) &
                error stop "Aedelsten: symmetrise_orbitals: components must hold n_mo values"
            rc = symmetrise_orbitals_c(self%handle, orbitals, length, n_mo, irreps, components, dropped_wire)
        else
            allocate(component_buffer(n_mo))
            rc = symmetrise_orbitals_c(self%handle, orbitals, length, n_mo, irreps, component_buffer, dropped_wire)
        end if
        if (rc /= 0) error stop "Aedelsten: symmetrise_orbitals failed"
        if (present(dropped)) dropped = dropped_wire
    end subroutine symmetrise_orbitals

    ! Initializes MPI at thread level required_level ("single", "funneled", "serialized", "multiple") and returns the
    ! level provided (0-3). Stops when MPI is initialized already by someone else, or provides less.
    function aedelsten_mpi_init(required_level) result(provided)
        character(len=*), intent(in)     :: required_level
        integer(kind=aedelsten_int_size) :: provided
        provided = aedelsten_mpi_init_c(query_i32_checked("mpi_thread_" // trim(lowercase(required_level))))
        if (provided < 0) error stop "Aedelsten: MPI initialization failed"
    end function aedelsten_mpi_init

    ! Finalizes MPI if aedelsten_mpi_init initialized it; MPI initialized by the host is left alone.
    subroutine aedelsten_mpi_finalize()
        integer(kind=aedelsten_int_size) :: rc
        rc = aedelsten_mpi_finalize_c()
    end subroutine aedelsten_mpi_finalize

    ! Whether MPI is initialized and not finalized (always false without the library's mpi build option).
    function aedelsten_mpi_initialized() result(flag)
        logical :: flag
        flag = aedelsten_mpi_initialized_c() == 1
    end function aedelsten_mpi_initialized

    ! The thread level MPI runs at (0-3), or -1 while it is not up.
    function aedelsten_mpi_thread_level() result(level)
        integer(kind=aedelsten_int_size) :: level
        level = aedelsten_mpi_thread_level_c()
    end function aedelsten_mpi_thread_level

    ! This rank's index in communicator comm (its Fortran handle; absent: MPI_COMM_WORLD).
    function aedelsten_mpi_rank(comm) result(rank)
        integer(kind=aedelsten_int_size), intent(in), optional :: comm
        integer(kind=aedelsten_int_size) :: rank
        if (aedelsten_mpi_rank_c(communicator_handle(comm), rank) /= 0) error stop "Aedelsten: aedelsten_mpi_rank failed"
    end function aedelsten_mpi_rank

    ! `length` doubles in the shared memory of the ranks of communicator comm (its Fortran handle; absent: this process
    ! alone), which must share one node: array%data, the same memory on every rank. Collective.
    subroutine aedelsten_shared_array(array, length, comm)
        type(AedelstenSharedArray), intent(out)                :: array
        integer(kind=c_int64_t), intent(in)                    :: length
        integer(kind=aedelsten_int_size), intent(in), optional :: comm
        type(c_ptr) :: data
        array%handle = aedelsten_mpi_shared_array_c(context_communicator_handle(comm), length, data)
        if (.not. c_associated(array%handle)) error stop "Aedelsten: failed to make a shared array"
        if (length > 0) call c_f_pointer(data, array%data, [length])
    end subroutine aedelsten_shared_array

    ! Every rank's writes to the shared array made visible to all, and every rank waited for. Collective.
    subroutine aedelsten_shared_array_sync(array)
        type(AedelstenSharedArray), intent(in) :: array
        if (aedelsten_mpi_shared_array_sync_c(array%handle) /= 0) error stop "Aedelsten: failed to synchronize a shared array"
    end subroutine aedelsten_shared_array_sync

    ! Frees the shared array. Collective.
    subroutine aedelsten_shared_array_free(array)
        type(AedelstenSharedArray), intent(inout) :: array
        call aedelsten_mpi_shared_array_free_c(array%handle)
        array%handle = c_null_ptr
        nullify(array%data)
    end subroutine aedelsten_shared_array_free

    ! The number of ranks in communicator comm (its Fortran handle; absent: MPI_COMM_WORLD).
    function aedelsten_mpi_size(comm) result(size)
        integer(kind=aedelsten_int_size), intent(in), optional :: comm
        integer(kind=aedelsten_int_size) :: size
        if (aedelsten_mpi_size_c(communicator_handle(comm), size) /= 0) error stop "Aedelsten: aedelsten_mpi_size failed"
    end function aedelsten_mpi_size

    ! Sets the BLAS's thread count to k and returns the previous count. Stops when the BLAS cannot run k threads.
    function aedelsten_blas_set_threads(k) result(previous)
        integer(kind=aedelsten_int_size), intent(in) :: k
        integer(kind=aedelsten_int_size) :: previous
        previous = aedelsten_blas_set_threads_c(k)
        if (previous < 0) error stop "Aedelsten: the BLAS cannot run that many threads"
    end function aedelsten_blas_set_threads

    ! The BLAS's thread count now.
    function aedelsten_blas_threads() result(k)
        integer(kind=aedelsten_int_size) :: k
        k = aedelsten_blas_threads_c()
    end function aedelsten_blas_threads

    ! The library's version, "<major>.<minor>.<patch>" (a release is tagged v<version>).
    function aedelsten_get_version() result(version)
        character(len=:), allocatable :: version
        character(kind=c_char) :: buf(64)
        if (aedelsten_get_version_c(buf, size(buf, kind=c_int)) /= 0) error stop "Aedelsten: aedelsten_get_version failed"
        version = from_c_string_buf(buf)
    end function aedelsten_get_version

    ! The git commit the library was built from (12 hex digits), empty when its source tree was no git checkout.
    function aedelsten_get_source_commit() result(commit)
        character(len=:), allocatable :: commit
        character(kind=c_char) :: buf(64)
        if (aedelsten_get_source_commit_c(buf, size(buf, kind=c_int)) /= 0) then
            error stop "Aedelsten: aedelsten_get_source_commit failed"
        end if
        commit = from_c_string_buf(buf)
    end function aedelsten_get_source_commit

    ! ── Timings and the reference list ──
    ! Read anew at each call: the coarse timings are always recorded, the fine ones only in a `profile` build; both
    ! are this thread's. The references are the process's: everything cited so far, in the order first cited.
    ! Indices count from 0, as everywhere in this module.

    ! Seconds this process has spent in MPI collectives since it started (0 without MPI or on one rank).
    function aedelsten_mpi_collective_seconds() result(s)
        real(c_double) :: s
        s = aedelsten_mpi_collective_seconds_c()
    end function aedelsten_mpi_collective_seconds

    ! Whether the library was built with the `profile` feature (the fine timings are recorded).
    function aedelsten_profile_enabled() result(flag)
        logical :: flag
        flag = aedelsten_profile_enabled_c() /= 0
    end function aedelsten_profile_enabled

    ! The number of labels in the coarse timing report (fine false) or the fine one (fine true).
    function aedelsten_timing_count(fine) result(n)
        logical, intent(in) :: fine
        integer(kind=external_int_size) :: n
        n = int(aedelsten_timing_count_c(merge(1_c_int, 0_c_int, fine)), kind=external_int_size)
    end function aedelsten_timing_count

    ! Entry index (from 0) of the report fine chooses: its label, seconds and calls.
    subroutine aedelsten_timing_span(fine, index, label, seconds, calls)
        logical, intent(in) :: fine
        integer(kind=external_int_size), intent(in) :: index
        character(len=:), allocatable, intent(out) :: label
        real(c_double), intent(out) :: seconds
        integer(c_int64_t), intent(out) :: calls
        character(kind=c_char) :: buf(256)
        if (aedelsten_timing_span_c(merge(1_c_int, 0_c_int, fine), to_wire_i32(index, "index"), buf, &
                size(buf, kind=c_int), seconds, calls) /= 0) error stop "Aedelsten: aedelsten_timing_span: no such entry"
        label = from_c_string_buf(buf)
    end subroutine aedelsten_timing_span

    ! The number of nodes in the span tree (coarse and fine spans, depth first).
    function aedelsten_span_tree_count() result(n)
        integer(kind=external_int_size) :: n
        n = int(aedelsten_span_tree_count_c(), kind=external_int_size)
    end function aedelsten_span_tree_count

    ! Node index (from 0) of the span tree, depth first (each node followed by its children): its label, depth (0 at
    ! the top), whether it is fine, its seconds (children included) and its calls.
    subroutine aedelsten_span_tree_node(index, label, depth, fine, seconds, calls)
        integer(kind=external_int_size), intent(in) :: index
        character(len=:), allocatable, intent(out) :: label
        integer(kind=external_int_size), intent(out) :: depth
        logical, intent(out) :: fine
        real(c_double), intent(out) :: seconds
        integer(c_int64_t), intent(out) :: calls
        character(kind=c_char) :: buf(256)
        integer(c_int) :: depth_c, fine_c
        if (aedelsten_span_tree_node_c(to_wire_i32(index, "index"), buf, size(buf, kind=c_int), depth_c, fine_c, &
                seconds, calls) /= 0) error stop "Aedelsten: aedelsten_span_tree_node: no such node"
        label = from_c_string_buf(buf)
        depth = int(depth_c, kind=external_int_size)
        fine = fine_c /= 0
    end subroutine aedelsten_span_tree_node

    ! The number of references cited so far.
    function aedelsten_citation_count() result(n)
        integer(kind=external_int_size) :: n
        n = int(aedelsten_citation_count_c(), kind=external_int_size)
    end function aedelsten_citation_count

    ! Reference index (from 0): the formatted reference (authors, title, journal, volume, pages, year, DOI), or with
    ! topics true the topics it was cited for, joined by ", ".
    function aedelsten_citation(index, topics) result(text)
        integer(kind=external_int_size), intent(in) :: index
        logical, intent(in), optional :: topics
        character(len=:), allocatable :: text
        character(kind=c_char), allocatable :: buf(:)
        integer(c_int) :: index_c, part
        integer(c_int64_t) :: n
        index_c = to_wire_i32(index, "index")
        part = 0_c_int
        if (present(topics)) part = merge(1_c_int, 0_c_int, topics)
        n = aedelsten_citation_length_c(index_c, part)
        if (n < 0) error stop "Aedelsten: aedelsten_citation: no such reference"
        allocate(buf(n + 1))
        if (aedelsten_citation_c(index_c, part, buf, size(buf, kind=c_int)) /= 0) error stop "Aedelsten: aedelsten_citation failed"
        text = from_c_string_buf(buf)
    end function aedelsten_citation

    ! The references cited so far as a BibTeX file (one entry each, the topics as keywords).
    function aedelsten_citations_bibtex() result(text)
        character(len=:), allocatable :: text
        character(kind=c_char), allocatable :: buf(:)
        allocate(buf(aedelsten_citations_bibtex_length_c() + 1))
        if (aedelsten_citations_bibtex_c(buf, size(buf, kind=c_int)) /= 0) error stop "Aedelsten: aedelsten_citations_bibtex failed"
        text = from_c_string_buf(buf)
    end function aedelsten_citations_bibtex

    ! Forgets every reference cited so far (a host running several independent jobs in one process).
    subroutine aedelsten_clear_citations()
        call aedelsten_clear_citations_c()
    end subroutine aedelsten_clear_citations

    ! The BLAS, its threading and its thread count, as one line (e.g. "OpenBLAS (OpenMP build), 1 thread").
    function aedelsten_blas_info() result(info)
        character(len=:), allocatable :: info
        character(kind=c_char) :: buf(256)
        if (aedelsten_blas_info_c(buf, size(buf, kind=c_int)) /= 0) error stop "Aedelsten: aedelsten_blas_info failed"
        info = from_c_string_buf(buf)
    end function aedelsten_blas_info

    ! The cores available to this process: their count, whether a launcher bound the process to them (else they are the
    ! node's, shared by its ranks), and optionally a line for a run's header (e.g. "4 performance cores, node wide").
    subroutine aedelsten_blas_available_cores(count, bound, description)
        integer(kind=aedelsten_int_size), intent(out) :: count
        logical, intent(out) :: bound
        character(len=:), allocatable, intent(out), optional :: description
        integer(c_int) :: bound_c
        character(kind=c_char) :: buf(256)
        if (aedelsten_blas_available_cores_c(count, bound_c) /= 0) error stop "Aedelsten: aedelsten_blas_available_cores failed"
        bound = bound_c /= 0
        if (present(description)) then
            if (aedelsten_blas_available_cores_info_c(buf, size(buf, kind=c_int)) /= 0) then
                error stop "Aedelsten: aedelsten_blas_available_cores failed"
            end if
            description = from_c_string_buf(buf)
        end if
    end subroutine aedelsten_blas_available_cores

    ! The handle a context takes for comm: this process alone when absent.
    function communicator_handle(comm) result(handle)
        integer(kind=aedelsten_int_size), intent(in), optional :: comm
        integer(kind=aedelsten_int_size) :: handle
        if (present(comm)) then
            handle = comm
        else
            handle = query_i32_checked("mpi_comm_world")
        end if
    end function communicator_handle

    ! The handle a context takes for `comm`: this process alone when absent.
    function context_communicator_handle(comm) result(handle)
        integer(kind=aedelsten_int_size), intent(in), optional :: comm
        integer(kind=aedelsten_int_size) :: handle
        if (present(comm)) then
            handle = comm
        else
            handle = query_i32_checked("mpi_comm_single")
        end if
    end function context_communicator_handle

end module Aedelsten
