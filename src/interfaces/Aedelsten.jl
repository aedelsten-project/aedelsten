# SPDX-FileCopyrightText: 2024 Ernst D. Larsson
# SPDX-License-Identifier: MIT

module Aedelsten

# The Julia interface to the library, over its C ABI: an opaque context and the operations on it, each a function
# f(ctx, ...) that `ctx.f(...)` also reaches (for every name in `_METHODS`). Every computation returns a new Array,
# but for the `!` functions that write into the caller's. Julia's arrays are column-major as the library's, so every
# shape is the library's own, nothing transposed.

export AedelstenContext
export set_shell_pairs!, set_primitive_shell_pairs!, set_scratch!, set_finite_nuclear_model!
export free_shell_pairs!, free_primitive_shell_pairs!, free_scratch!
export free_input!, free_primitive_shells!, free_contraction_matrix!
export resize_scratch!, get_scratch_peak_bytes, set_communicator!
export get_n_cont, get_n_prim, get_n_prim_unique, has_primitive
export compute_centre_of_mass, compute_nuclear_repulsion_energy, compute_nuclear_charge
export compute_nuclear_aimp_embedding_energy, compute_nuclear_multipole_interaction_energy
export get_n_centres, get_centre_geometry, get_nuclear_charges, unrescale_cartesian_ao_rows!
export set_external_multipoles!, free_external_multipoles!, has_external_multipoles
export PolarizationOptions, set_polarizable_sites!, free_polarizable_sites!, get_n_polarizable_sites
export set_polarization!, free_polarization!, compute_induction, compute_induction!, compute_induction_gradient
export get_polarization_storage, set_resident_field_budget!
export compute_induction_response, compute_induction_couplings, compute_induction_states, get_induced_dipoles, compute_effective_external_field
export set_induced_dipoles!
export has_polarizable_sites, has_polarization, count_electric_field_sites, get_input_indices, compute_electric_field_integrals, compute_static_field, compute_electronic_field, set_multipole_convention!, get_multipole_convention
export has_core_aimp, has_embedding_aimp, set_embedding_aimp_operators!, free_embedding_aimp_operators!
export get_n_embedding_aimps, get_embedding_aimp_sites
export compute_one_electron_integrals
export get_n_gradient_centres, compute_one_electron_gradient, set_centre_positions!
export compute_nuclear_repulsion_gradient, compute_nuclear_aimp_embedding_gradient, compute_nuclear_multipole_interaction_gradient
export get_n_shells, get_shells, compute_shell_extents, compute_ao_values, compute_ao_hessians
export compute_one_electron_picture_change_integrals, construct_x2c_one_electron_hamiltonian
export contract_one_electron_integrals
export compute_eri_stored, compute_two_electron_gradient
export compute_j_and_k_matrix, compute_j_and_k_unrestricted_matrix
export compute_j_matrix, compute_k_matrix, compute_k_matrix_unrestricted
export set_eri_data!, free_eri_data!, compute_fock_direct, compute_fock_direct_unrestricted, compute_fock_direct_general
export compute_fock_direct_general_multi
export compute_fock_direct_mixed_active_space, compute_fock_direct_active_space
export transform_pair_coulomb_one_index
export compute_eri_half_transformed, compute_eri_half_transformed_on_node!, compute_eri_half_transformed_sigma, compute_eri_half_transformed_sigma_on_node!, half_transform_scratch_bytes
export scratch_bytes, scratch_breakdown
export AedelstenSharedArray, sync_shared!, free_shared!
export get_n_primitive_rows, memory_breakdown_field_count, memory_breakdown
export get_basis_function_ids, get_primitive_ids, get_primitives, get_centre_label
export get_n_model_functions, get_model_functions, compute_model_basis_overlaps, compute_model_fockian, compute_overlap_with
export rescale_cartesian_ao_rows!, rescale_cartesian_ao_columns!
export has_picture_change, get_picture_change_transforms, free_picture_change!, set_picture_change_transforms!
export set_x2c_frozen_density!, free_x2c_frozen_density!, compute_x2c_frozen_kinetic_gradient, compute_x2c_frozen_potential_gradient
export transform_one_electron_operator, transform_one_electron_spin_operator, compute_picture_changed_one_electron_operator, compute_amfi_integrals
export set_symmetry!, free_symmetry!, get_symmetry_options, get_symmetrisation_report
export get_point_group_name, get_irreps, get_irrep_index, get_totally_symmetric_irrep, get_cartesian_irrep
export get_ao_irrep_counts, apply_symmetry_operation, get_irrep_character, get_irrep_matrix
export get_symmetry_standard_axes, get_symmetry_operation_count, get_symmetry_operation_matrix, get_n_symmetry_centres, get_symmetry_centre_images, symmetrise_matrix!, analyse_matrix, irrep_norms, symmetrise_orbitals!
export aedelsten_mpi_init, aedelsten_mpi_finalize, aedelsten_mpi_initialized, aedelsten_mpi_thread_level
export aedelsten_mpi_rank, aedelsten_mpi_size
export aedelsten_blas_set_threads, aedelsten_blas_threads, aedelsten_blas_info, aedelsten_blas_available_cores
export aedelsten_get_version, aedelsten_get_source_commit
export aedelsten_mpi_collective_seconds, aedelsten_profile_enabled, aedelsten_timing_report, aedelsten_profile_report
export aedelsten_span_tree, aedelsten_citations, aedelsten_citations_bibtex, aedelsten_clear_citations

using Libdl

# --- Library loading ---

const _lib_ref = Ref{Ptr{Cvoid}}(C_NULL)

function _lib_name()::String
    Sys.iswindows() && return "aedelsten.dll"
    Sys.isapple()   && return "libaedelsten.dylib"
    return "libaedelsten.so"
end

function _find_lib()::String
    name = _lib_name()
    haskey(ENV, "AEDELSTEN_LIB") && return ENV["AEDELSTEN_LIB"]
    search_var = Sys.isapple() ? "DYLD_LIBRARY_PATH" : "LD_LIBRARY_PATH"
    for dir in split(get(ENV, search_var, ""), ':')
        isempty(dir) && continue
        candidate = joinpath(dir, name)
        isfile(candidate) && return candidate
    end
    bundled = normpath(joinpath(@__DIR__, "..", "..", "lib", name))
    isfile(bundled) && return bundled
    error("Could not find $name. Set AEDELSTEN_LIB to its full path, or add its directory to $search_var.")
end

function _load_lib()::Ptr{Cvoid}
    _lib_ref[] != C_NULL && return _lib_ref[]
    lib = Libdl.dlopen(_find_lib())
    _lib_ref[] = lib
    return lib
end

const _sym_cache = Dict{Symbol,Ptr{Cvoid}}()
_sym(name::Symbol)::Ptr{Cvoid} = get!(() -> Libdl.dlsym(_load_lib(), name), _sym_cache, name)

# --- Wire-key constant resolution ---
# Every named constant (contraction types, one-electron operator kinds,
# byte units, ...) is queried through these two, exactly like
# aedelsten.py's _Constants class and Fortran's query_i32_checked/
# query_f64_checked -- never hardcoded here, so a new constant added on
# the Rust side needs no matching edit in this file.

const _i32_cache = Dict{String,Int32}()
function _query_i32(key::AbstractString)::Int32
    get!(_i32_cache, lowercase(key)) do
        val = ccall(_sym(:aedelsten_query_i32_c), Cint, (Cstring,), lowercase(key))
        val == -1 && error("Aedelsten: unknown constant key '$key'")
        Int32(val)
    end
end

function _query_f64(key::AbstractString)::Float64
    val = ccall(_sym(:aedelsten_query_f64_c), Cdouble, (Cstring,), lowercase(key))
    isnan(val) && error("Aedelsten: unknown constant key '$key'")
    return val
end

function _resolve_contraction_type(value::Union{AbstractString,Nothing})::Int32
    value === nothing && return _query_i32("default_contraction_type")
    return _query_i32("contraction_type_" * lowercase(value))
end

# Same bare-suffix-or-full-key convention as the Fortran interface's
# resolve_operator: "overlap" and "oneint_overlap" both work.
function _resolve_operator(operator::AbstractString)::Int32
    key = lowercase(operator)
    startswith(key, "oneint_") || (key = "oneint_" * key)
    return _query_i32(key)
end

# Bundled range/omega, matching Fortran's resolve_eri_range: both present
# or both absent (absent = full-range Coulomb).
function _resolve_eri_range(range::Union{AbstractString,Nothing}, omega::Union{Real,Nothing})::Tuple{Int32,Float64}
    (range === nothing) == (omega === nothing) ||
        error("Aedelsten: range and omega must both be given, or both omitted (omitted = full range)")
    range === nothing && return (Int32(0), 0.0)
    return (_query_i32("eri_" * lowercase(range)), Float64(omega))
end

# --- Context ---

struct AedelstenContextHandle end  # opaque

mutable struct AedelstenContext
    handle::Ptr{AedelstenContextHandle}

    # Its parallel computations are split over the ranks of communicator `comm` (its Fortran handle, e.g. MPI.jl's
    # `comm.val`, or `_query_i32("mpi_comm_world")`; `nothing`: this process alone, without MPI).
    function AedelstenContext(filename::AbstractString; contraction_type::Union{AbstractString,Nothing}=nothing, comm::Union{Integer,Nothing}=nothing)
        ct = _resolve_contraction_type(contraction_type)
        handle = ccall(
            _sym(:set_aedelsten_context_c), Ptr{AedelstenContextHandle},
            (Cstring, Cint, Cint), filename, ct, _communicator_handle(comm),
        )
        handle == C_NULL && error("Aedelsten: failed to create context for '$filename'")
        ctx = new(handle)
        finalizer(_free!, ctx)  # GC-triggered, like Fortran's FINAL / Rust's Drop
        return ctx
    end
end

function _free!(ctx::AedelstenContext)::Nothing
    if getfield(ctx, :handle) != C_NULL
        ccall(_sym(:free_aedelsten_context_c), Cvoid, (Ptr{AedelstenContextHandle},), getfield(ctx, :handle))
        setfield!(ctx, :handle, Ptr{AedelstenContextHandle}(C_NULL))
    end
    return nothing
end

# --- ctx.f(...) dot-call sugar ---
# Julia has no obj.method(...) syntax of its own; every operation below is
# a plain function f(ctx, ...) taking the context as its first argument
# (multiple dispatch on that argument's type is as close to "a method on a
# type" as the language gets natively). This block adds the syntactic
# sugar on top: accessing ctx.name for a name in _METHODS hands back a
# closure over ctx, so ctx.compute_one_electron_integrals("overlap") and
# compute_one_electron_integrals(ctx, "overlap") reach the same function.
# `handle` stays a real, gettable/settable field throughout.

const _METHODS = (
    :set_shell_pairs!, :set_primitive_shell_pairs!, :set_scratch!, :set_finite_nuclear_model!, :set_communicator!,
    :free_shell_pairs!, :free_primitive_shell_pairs!, :free_scratch!,
    :free_input!, :free_primitive_shells!, :free_contraction_matrix!,
    :resize_scratch!, :get_scratch_peak_bytes,
    :get_n_cont, :get_n_prim, :get_n_prim_unique, :has_primitive,
    :compute_centre_of_mass, :compute_nuclear_repulsion_energy, :compute_nuclear_charge,
    :compute_nuclear_aimp_embedding_energy, :compute_nuclear_multipole_interaction_energy,
    :get_n_centres, :get_centre_geometry, :get_nuclear_charges, :unrescale_cartesian_ao_rows!,
    :set_external_multipoles!, :free_external_multipoles!, :has_external_multipoles,
    :set_polarizable_sites!, :free_polarizable_sites!, :get_n_polarizable_sites,
    :set_polarization!, :free_polarization!, :compute_induction, :compute_induction!, :compute_induction_gradient,
    :get_polarization_storage, :set_resident_field_budget!,
    :compute_induction_response, :compute_induction_couplings, :compute_induction_states, :get_induced_dipoles, :compute_effective_external_field,
    :set_induced_dipoles!,
    :has_polarizable_sites, :has_polarization, :count_electric_field_sites, :get_input_indices, :compute_electric_field_integrals, :compute_static_field, :compute_electronic_field, :set_multipole_convention!, :get_multipole_convention,
    :has_core_aimp, :has_embedding_aimp, :set_embedding_aimp_operators!, :free_embedding_aimp_operators!,
    :get_n_embedding_aimps, :get_embedding_aimp_sites,
    :compute_one_electron_integrals,
    :get_n_gradient_centres, :compute_one_electron_gradient, :set_centre_positions!,
    :compute_nuclear_repulsion_gradient, :compute_nuclear_aimp_embedding_gradient, :compute_nuclear_multipole_interaction_gradient,
    :get_n_shells, :get_shells, :compute_shell_extents, :compute_ao_values, :compute_ao_hessians,
    :compute_one_electron_picture_change_integrals, :construct_x2c_one_electron_hamiltonian,
    :contract_one_electron_integrals,
    :compute_eri_stored, :set_eri_data!, :free_eri_data!,
    :compute_fock_direct, :compute_fock_direct_unrestricted, :compute_fock_direct_general, :compute_fock_direct_general_multi, :compute_fock_direct_mixed_active_space, :compute_fock_direct_active_space, :transform_pair_coulomb_one_index, :compute_eri_half_transformed, :compute_eri_half_transformed_on_node!, :compute_eri_half_transformed_sigma, :compute_eri_half_transformed_sigma_on_node!, :half_transform_scratch_bytes, :scratch_bytes, :scratch_breakdown, :compute_two_electron_gradient,
    :get_n_primitive_rows, :memory_breakdown,
    :get_basis_function_ids, :get_primitive_ids, :get_primitives, :get_centre_label,
    :get_n_model_functions, :get_model_functions, :compute_model_basis_overlaps, :compute_model_fockian, :compute_overlap_with,
    :rescale_cartesian_ao_rows!, :rescale_cartesian_ao_columns!,
    :has_picture_change, :get_picture_change_transforms, :free_picture_change!, :set_picture_change_transforms!,
    :set_x2c_frozen_density!, :free_x2c_frozen_density!, :compute_x2c_frozen_kinetic_gradient, :compute_x2c_frozen_potential_gradient,
    :transform_one_electron_operator, :transform_one_electron_spin_operator, :compute_picture_changed_one_electron_operator, :compute_amfi_integrals,
    :set_symmetry!, :free_symmetry!, :get_symmetry_options, :get_symmetrisation_report,
    :get_point_group_name, :get_irreps, :get_irrep_index, :get_totally_symmetric_irrep, :get_cartesian_irrep,
    :get_ao_irrep_counts, :apply_symmetry_operation, :get_irrep_character, :get_irrep_matrix,
    :get_symmetry_standard_axes, :get_symmetry_operation_count, :get_symmetry_operation_matrix, :get_n_symmetry_centres, :get_symmetry_centre_images, :symmetrise_matrix!, :analyse_matrix, :irrep_norms, :symmetrise_orbitals!,
)
# The functions that take no context (memory_breakdown_field_count, the dense J/K builders) are not in _METHODS.

function Base.getproperty(ctx::AedelstenContext, name::Symbol)
    name === :handle && return getfield(ctx, :handle)
    name in _METHODS && return (args...; kwargs...) -> getfield(Aedelsten, name)(ctx, args...; kwargs...)
    error("type AedelstenContext has no field or method `$name`")
end

Base.propertynames(::AedelstenContext) = (:handle, _METHODS...)

# --- Shell pairs / scratch / finite nuclear model ---
# `!`-suffixed: Julia's own convention for a function that mutates its
# (first) argument, the idiomatic equivalent of Fortran's set_* naming.

function set_shell_pairs!(ctx::AedelstenContext; overlap_threshold::Union{Real,Nothing}=nothing)::AedelstenContext
    # `nothing` (the default): no screening (the C entry point's negative sentinel).
    threshold::Float64 = overlap_threshold === nothing ? -1.0 : Float64(overlap_threshold)
    ret = ccall(_sym(:set_shell_pairs_c), Cint, (Ptr{AedelstenContextHandle}, Cdouble), ctx.handle, threshold)
    ret == 0 || error("Aedelsten: failed to set shell pairs (code $ret)")
    return ctx
end

"""Decontracts the basis into its primitive shells, their pairs (primitive pairs whose overlap is below
`overlap_threshold` dropped; `nothing`: none) and the contraction matrix: the primitive basis of X2C, AMFI and the
integrals with `primitive=true`."""
function set_primitive_shell_pairs!(ctx::AedelstenContext; overlap_threshold::Union{Real,Nothing}=nothing)::AedelstenContext
    # `nothing` (the default): no screening (the C entry point's negative sentinel).
    threshold::Float64 = overlap_threshold === nothing ? -1.0 : Float64(overlap_threshold)
    ret = ccall(
        _sym(:set_primitive_shell_pairs_c), Cint,
        (Ptr{AedelstenContextHandle}, Cdouble), ctx.handle, threshold,
    )
    ret == 0 || error("Aedelsten: failed to set primitive shell pairs (code $ret)")
    return ctx
end

"""
    set_scratch!(ctx; size, unit)

The buffer every integral computation takes its scratch from; the caller chooses its size (required), the library
makes no judgment about it (`unit`: "kib", "mib" or "gib"). What a computation needs
depends only on the basis's largest shells, not on the molecule; two-electron integral gradients need the most (about
ten times the two-electron integrals); one-electron integrals, X2C included, need little; deep, generally contracted or
Cartesian shells raise the need steeply. Measure it on a dimer of the element with the deepest basis
(`get_scratch_peak_bytes(ctx)` after the call). The buffer is committed lazily. It never grows: a computation that
runs out of it fails with an error naming the size needed.
"""
function set_scratch!(
    ctx::AedelstenContext; size::Integer, unit::AbstractString
)::AedelstenContext
    size_c::Int32 = Int32(size)
    unit_c::Int32 = _query_i32("byte_unit_" * lowercase(unit))
    ret = ccall(_sym(:set_scratch_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Cint), ctx.handle, size_c, unit_c)
    ret == 0 || error("Aedelsten: failed to set scratch (code $ret)")
    return ctx
end

"""
    set_communicator!(ctx, comm)

Splits the context's parallel computations over communicator `comm` from now on (its Fortran handle; `nothing`: this
process alone).
"""
function set_communicator!(ctx::AedelstenContext, comm::Union{Integer,Nothing})::AedelstenContext
    ret = ccall(_sym(:set_communicator_c), Cint, (Ptr{AedelstenContextHandle}, Cint), ctx.handle, _communicator_handle(comm))
    ret == 0 || error("Aedelsten: failed to set communicator $comm")
    return ctx
end

"""
    resize_scratch!(ctx; size, unit)

Replaces the scratch buffer (arguments as `set_scratch!`), e.g. to shrink it to `get_scratch_peak_bytes(ctx)`
plus a margin, or to grow it after an out-of-space error. The new buffer is committed lazily; the high-water mark
starts over.
"""
function resize_scratch!(
    ctx::AedelstenContext; size::Integer, unit::AbstractString
)::AedelstenContext
    size_c::Int32 = Int32(size)
    unit_c::Int32 = _query_i32("byte_unit_" * lowercase(unit))
    ret = ccall(_sym(:resize_scratch_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Cint), ctx.handle, size_c, unit_c)
    ret == 0 || error("Aedelsten: failed to resize scratch (code $ret)")
    return ctx
end

"""The most bytes of the scratch any computation has used so far (its high-water mark)."""
function get_scratch_peak_bytes(ctx::AedelstenContext)::Int
    bytes = ccall(_sym(:get_scratch_peak_bytes_c), Int64, (Ptr{AedelstenContextHandle},), ctx.handle)
    bytes >= 0 || error("Aedelsten: no scratch (set_scratch! first)")
    return Int(bytes)
end

"""Switches the nuclei from point charges to Gaussian charge distributions (Visscher-Dyall)."""
function set_finite_nuclear_model!(ctx::AedelstenContext)::AedelstenContext
    ret = ccall(_sym(:set_finite_nuclear_model_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle)
    ret == 0 || error("Aedelsten: failed to set finite nuclear model (code $ret)")
    return ctx
end

# --- Basis-size / molecular-property getters ---

get_n_cont(ctx::AedelstenContext)::Int = Int(ccall(_sym(:get_n_cont_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle))
"""The sum of the shells' primitive AOs (a primitive shared by sibling segments once per sibling)."""
get_n_prim(ctx::AedelstenContext)::Int = Int(ccall(_sym(:get_n_prim_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle))
"""The number of distinct primitive AOs: the dimension of the primitive basis."""
get_n_prim_unique(ctx::AedelstenContext)::Int =
    Int(ccall(_sym(:get_n_prim_unique_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle))
"""Whether the primitive basis is set (always for a primitive context)."""
has_primitive(ctx::AedelstenContext)::Bool =
    ccall(_sym(:has_primitive_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle) != 0

"""The centre of mass of the real atoms (bohr), a 3-vector."""
function compute_centre_of_mass(ctx::AedelstenContext)::Vector{Float64}
    x = Ref{Cdouble}(0.0); y = Ref{Cdouble}(0.0); z = Ref{Cdouble}(0.0)
    ccall(
        _sym(:compute_centre_of_mass_c), Cvoid,
        (Ptr{AedelstenContextHandle}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
        ctx.handle, x, y, z,
    )
    return [x[], y[], z[]]
end

"""The repulsion energy of the QM region's nuclei and core AIMPs, Eh."""
compute_nuclear_repulsion_energy(ctx::AedelstenContext)::Float64 =
    ccall(_sym(:compute_nuclear_repulsion_energy_c), Cdouble, (Ptr{AedelstenContextHandle},), ctx.handle)
"""The total charge of the QM region's nuclei and core AIMPs."""
compute_nuclear_charge(ctx::AedelstenContext)::Float64 =
    ccall(_sym(:compute_nuclear_charge_c), Cdouble, (Ptr{AedelstenContextHandle},), ctx.handle)

"""
    compute_nuclear_aimp_embedding_energy(ctx) -> Float64

The interaction of the nuclei (and core AIMPs) with the embedding AIMPs, Eh; 0 without embedding AIMPs.
"""
compute_nuclear_aimp_embedding_energy(ctx::AedelstenContext)::Float64 =
    ccall(_sym(:compute_nuclear_aimp_embedding_energy_c), Cdouble, (Ptr{AedelstenContextHandle},), ctx.handle)

"""
    compute_nuclear_multipole_interaction_energy(ctx) -> Float64

The interaction of the nuclei (and core AIMPs) with the external multipoles, Eh; NaN without them.
"""
compute_nuclear_multipole_interaction_energy(ctx::AedelstenContext)::Float64 =
    ccall(_sym(:compute_nuclear_multipole_interaction_energy_c), Cdouble, (Ptr{AedelstenContextHandle},), ctx.handle)

"""
    free_shell_pairs!(ctx), free_primitive_shell_pairs!(ctx), free_scratch!(ctx)

Release the contracted shell pairs, the primitive shell pairs, or the scratch buffer (each set again by its `set_`
call before it is needed).
"""
function free_shell_pairs!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_shell_pairs_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

"""Frees the primitive shell pairs."""
function free_primitive_shell_pairs!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_primitive_shell_pairs_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

"""Frees the scratch buffer."""
function free_scratch!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_scratch_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

"""
    free_input!(ctx), free_primitive_shells!(ctx), free_contraction_matrix!(ctx)

Release the parsed input (basis, nuclei, embedding), or the primitive shells or the contraction matrix made by
`set_primitive_shell_pairs!` (the primitive shell pairs stay). `free_input!` only after the last call that reads the
input (the operators, the shell pairs, the primitive shell pairs, the embedding): such a call afterwards stops the
program.
"""
function free_input!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_input_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

"""Frees the primitive shells made by `set_primitive_shell_pairs!`; the primitive shell pairs stay."""
function free_primitive_shells!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_primitive_shells_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

"""Frees the contraction matrix made by `set_primitive_shell_pairs!`."""
function free_contraction_matrix!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_contraction_matrix_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

# --- External multipoles / AIMP ---
# Both just materialize operators reached afterwards the same way as any
# other one-electron operator, via compute_one_electron_integrals with the
# matching wire-key suffix ("external_multipole", "core_aimp_coulomb",
# "core_aimp_exchange", "embedding_aimp_coulomb", "embedding_aimp_exchange")
# -- core AIMP regions need no setup call, they're already resident once
# the context exists; has_core_aimp just reports whether the input had any.

function set_external_multipoles!(ctx::AedelstenContext, filename::AbstractString)::AedelstenContext
    ret = ccall(
        _sym(:set_external_multipoles_c), Cint, (Ptr{AedelstenContextHandle}, Cstring), ctx.handle, filename,
    )
    ret == 0 || error("Aedelsten: failed to set external multipoles (code $ret)")
    return ctx
end

"""Frees the external multipoles."""
function free_external_multipoles!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_external_multipoles_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

"""Whether external multipoles are set."""
has_external_multipoles(ctx::AedelstenContext)::Bool =
    ccall(_sym(:has_external_multipoles_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle) != 0

# --- Induced-dipole (polarizable) embedding ---
# Mirrors the Fortran interface (same names, same option fields, same array layouts: (n, n) matrices,
# (n, n, k) batches, (3, n_sites) dipoles), but eager: the non-bang functions allocate and return their
# results; `compute_induction!` writes into a caller-owned matrix like Fortran does. The intended order is
# set_polarizable_sites! -> (set_external_multipoles!) -> set_polarization! -> compute_induction per Fock
# build; compute_induction_response for CI/CC/response trial-vector batches (needs response_batch >= 1);
# compute_induction_states for per-state dipoles (needs state_slots >= k).

"""Options of `set_polarization!`: the fields of the C struct `AedelstenPolarizationOptions`, in order. A
negative number means "the library's default"; construct with keywords, e.g.
`PolarizationOptions(threshold=1e-10, response_batch=4)`."""
Base.@kwdef struct PolarizationOptions
    damping::Cint = 0             # 0: none, 1: Thole
    thole_a::Cdouble = -1.0       # Thole parameter a; negative: 2.1304
    threshold::Cdouble = -1.0     # dipole-solver convergence; negative: 1e-8
    max_iterations::Cint = -1
    static_field::Cint = 0        # 0: keep the static field in memory, 1: recompute it
    chunk_sites::Cint = -1        # sites per integral chunk; not positive: use the budget
    chunk_budget::Cint = -1        # integral-chunk memory budget (in chunk_budget_unit); negative: default
    chunk_budget_unit::Cint = 1         # byte unit: 0 GiB, 1 MiB, 2 KiB
    response_batch::Cint = 0      # densities per compute_induction_response batch; 0: none
    state_slots::Cint = 0         # states held by compute_induction_states; 0: none
    resident_budget::Cint = 0       # budget for keeping the field integrals between calls; 0: never
    resident_budget_unit::Cint = 1       # byte unit of resident_budget: 0 GiB, 1 MiB, 2 KiB
end

"""Reads the polarizable sites of the xfield file `filename`."""
function set_polarizable_sites!(ctx::AedelstenContext, filename::AbstractString)::AedelstenContext
    ret = ccall(_sym(:set_polarizable_sites_c), Cint, (Ptr{AedelstenContextHandle}, Cstring), ctx.handle, filename)
    ret == 0 || error("Aedelsten: failed to set polarizable sites (code $ret)")
    return ctx
end

"""Frees the polarizable sites and the polarization set up on them."""
function free_polarizable_sites!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_polarizable_sites_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

"""The number of polarizable sites."""
function get_n_polarizable_sites(ctx::AedelstenContext)::Int
    n = ccall(_sym(:get_n_polarizable_sites_c), Int64, (Ptr{AedelstenContextHandle},), ctx.handle)
    n >= 0 || error("Aedelsten: could not count the polarizable sites (are they set?)")
    return Int(n)
end

"""Whether polarizable sites are set."""
has_polarizable_sites(ctx::AedelstenContext)::Bool =
    ccall(_sym(:has_polarizable_sites_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle) == 1

"""Whether `set_polarization!` has set the polarization up."""
has_polarization(ctx::AedelstenContext)::Bool =
    ccall(_sym(:has_polarization_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle) == 1

# --- The electric field at sites ---
# A selection is a name (:qm, the nuclei and core AIMPs; :embedding_aimps, :external_multipoles, :polarizable) or a
# (3, n_points) matrix of custom points (bohr). Fields are (3, n_sites) matrices.

_selection(selection::Union{Symbol, AbstractString}) =
    (_query_i32("site_selection_" * lowercase(String(selection))), Float64[], 0)
function _selection(points::AbstractMatrix{<:Real})
    size(points, 1) == 3 || error("Aedelsten: custom sites must be a (3, n_points) matrix")
    return (_query_i32("site_selection_custom"), Matrix{Float64}(points), size(points, 2))
end

"""The number of sites of `selection`."""
function count_electric_field_sites(ctx::AedelstenContext, selection)::Int
    s, points, n_points = _selection(selection)
    n = ccall(_sym(:count_electric_field_sites_c), Int64, (Ptr{AedelstenContextHandle}, Cint, Ptr{Cdouble}, Int64),
              ctx.handle, s, points, Int64(n_points))
    n >= 0 || error("Aedelsten: count_electric_field_sites failed for $(repr(selection))")
    return Int(n)
end

"""Each site's record in its input file (0-based), `nothing` for a site from none (nuclei, AIMPs, custom points)."""
function get_input_indices(ctx::AedelstenContext, selection)::Vector{Union{Nothing, Int}}
    s, points, n_points = _selection(selection)
    n = count_electric_field_sites(ctx, selection)
    indices = zeros(Int64, n)
    ret = ccall(_sym(:get_input_indices_c), Cint,
                (Ptr{AedelstenContextHandle}, Cint, Ptr{Cdouble}, Int64, Ptr{Int64}, Int64),
                ctx.handle, s, points, Int64(n_points), indices, Int64(n))
    ret == 0 || error("Aedelsten: get_input_indices failed (code $ret)")
    return [i < 0 ? nothing : Int(i) for i in indices]
end

"""The electric-field integrals of the sites `sites` (0-based, a unit range; all when `nothing`), an
(n, n, 3, count) array: per site its x, y and z planes."""
function compute_electric_field_integrals(ctx::AedelstenContext, selection, sites::Union{Nothing, AbstractUnitRange{<:Integer}} = nothing)::Array{Float64, 4}
    s, points, n_points = _selection(selection)
    sites = sites === nothing ? (0:count_electric_field_sites(ctx, selection) - 1) : sites
    n = get_n_cont(ctx)
    out = zeros(Float64, n, n, 3, length(sites))
    ret = ccall(_sym(:compute_electric_field_integrals_c), Cint,
                (Ptr{AedelstenContextHandle}, Cint, Ptr{Cdouble}, Int64, Int64, Int64, Ptr{Cdouble}, Int64),
                ctx.handle, s, points, Int64(n_points), Int64(first(sites)), Int64(length(sites)), out, Int64(length(out)))
    ret == 0 || error("Aedelsten: compute_electric_field_integrals failed (code $ret)")
    return out
end

"""The classical field of the sources switched on at the sites of `selection`, (3, n_sites)."""
function compute_static_field(ctx::AedelstenContext, selection; nuclei::Bool = true, core_aimps::Bool = true,
                              embedding_aimps::Bool = true, external_multipoles::Bool = true)::Matrix{Float64}
    s, points, n_points = _selection(selection)
    field = zeros(Float64, 3, count_electric_field_sites(ctx, selection))
    ret = ccall(_sym(:compute_static_field_c), Cint,
                (Ptr{AedelstenContextHandle}, Cint, Ptr{Cdouble}, Int64, Cint, Cint, Cint, Cint, Ptr{Cdouble}, Int64),
                ctx.handle, s, points, Int64(n_points), Cint(nuclei), Cint(core_aimps), Cint(embedding_aimps),
                Cint(external_multipoles), field, Int64(length(field)))
    ret == 0 || error("Aedelsten: compute_static_field failed (code $ret)")
    return field
end

"""The field of the plain density `density` at the sites of `selection`, (3, n_sites); chunked by `chunk_sites`
sites when positive, else by `chunk_budget` in `chunk_budget_unit` (negative: the default)."""
function compute_electronic_field(ctx::AedelstenContext, density::AbstractMatrix{<:Real}, selection;
                                  chunk_sites::Integer = -1, chunk_budget::Integer = -1,
                                  chunk_budget_unit::AbstractString = "MiB")::Matrix{Float64}
    _check_matrix(ctx, density, "density")
    d = _dense(density)
    s, points, n_points = _selection(selection)
    field = zeros(Float64, 3, count_electric_field_sites(ctx, selection))
    ret = ccall(_sym(:compute_electronic_field_c), Cint,
                (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Int64, Int64, Int64, Cint, Ptr{Cdouble}, Int64),
                ctx.handle, d, s, points, Int64(n_points), Int64(chunk_sites), Int64(chunk_budget),
                _query_i32("byte_unit_" * lowercase(chunk_budget_unit)), field, Int64(length(field)))
    ret == 0 || error("Aedelsten: compute_electronic_field failed (code $ret)")
    return field
end

"""How the external multipoles meet the electrons: :contact_free (the default) or :exact."""
function set_multipole_convention!(ctx::AedelstenContext, convention::Union{Symbol, AbstractString})::AedelstenContext
    c = _query_i32("multipole_convention_" * lowercase(String(convention)))
    ret = ccall(_sym(:set_multipole_convention_c), Cint, (Ptr{AedelstenContextHandle}, Cint), ctx.handle, c)
    ret == 0 || error("Aedelsten: set_multipole_convention! failed for $(repr(convention))")
    return ctx
end

"""How the external multipoles meet the electrons: `:contact_free` or `:exact`."""
function get_multipole_convention(ctx::AedelstenContext)::Symbol
    c = ccall(_sym(:get_multipole_convention_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle)
    c == _query_i32("multipole_convention_exact") && return :exact
    c == _query_i32("multipole_convention_contact_free") && return :contact_free
    error("Aedelsten: get_multipole_convention failed")
end

"""Sets up the induced-dipole polarization with `options`, once per geometry, after `set_polarizable_sites!`.
Collective under MPI."""
function set_polarization!(ctx::AedelstenContext, options::PolarizationOptions)::AedelstenContext
    ret = ccall(
        _sym(:set_polarization_c), Cint, (Ptr{AedelstenContextHandle}, Ref{PolarizationOptions}), ctx.handle, options,
    )
    ret == 0 || error("Aedelsten: failed to set up the polarization (code $ret)")
    return ctx
end
set_polarization!(ctx::AedelstenContext; kwargs...)::AedelstenContext = set_polarization!(ctx, PolarizationOptions(; kwargs...))

"""Frees the polarization state."""
function free_polarization!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_polarization_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

_dense(x::AbstractArray)::Array{Float64} = convert(Array{Float64}, x)

function _check_matrix(ctx::AedelstenContext, x::AbstractArray, what::AbstractString)
    n = get_n_cont(ctx)
    size(x)[1:2] == (n, n) || error("Aedelsten: $what must be ($n, $n[, k]), got $(size(x))")
    return n
end

"""`compute_induction!(ctx, v_ind, density; threshold=nothing) -> (energy, iterations)`: one Fock build. `v_ind`
(n, n, a Float64 `Array`) is overwritten with the operator to add to the one-electron Hamiltonian; `energy` is E_ind.
With `threshold`, the dipoles are solved to it for this call in place of the setup's (`compute_induction_within_c`)."""
function compute_induction!(ctx::AedelstenContext, v_ind::Matrix{Float64}, density::AbstractMatrix;
                            threshold::Union{Nothing,Real}=nothing)
    _check_matrix(ctx, v_ind, "v_ind"); _check_matrix(ctx, density, "density")
    d = _dense(density)
    energy = Ref{Cdouble}(0.0); iterations = Ref{Cint}(0)
    ret = if threshold === nothing
        ccall(
            _sym(:compute_induction_c), Cint,
            (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cint}),
            ctx.handle, d, v_ind, energy, iterations,
        )
    else
        ccall(
            _sym(:compute_induction_within_c), Cint,
            (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ref{Cdouble}, Ref{Cint}),
            ctx.handle, d, v_ind, Cdouble(threshold), energy, iterations,
        )
    end
    ret == 0 || error("Aedelsten: compute_induction failed (code $ret)")
    return (energy[], Int(iterations[]))
end

"""`compute_induction(ctx, density; threshold=nothing) -> (v_ind, energy, iterations)`, allocating `v_ind`."""
function compute_induction(ctx::AedelstenContext, density::AbstractMatrix; threshold::Union{Nothing,Real}=nothing)
    v = zeros(Float64, get_n_cont(ctx), get_n_cont(ctx))
    energy, iterations = compute_induction!(ctx, v, density; threshold=threshold)
    return (v, energy, iterations)
end

"""`get_polarization_storage(ctx)`: how this rank holds the field integrals of its share of the polarizable sites, a
named tuple `(sites, in_memory, bytes_per_site, bytes_to_keep)`: the sites of the share, the first kept in memory (the
rest evaluated in every pass), the bytes of a site's kept integrals, and the `resident_budget` that keeps the whole
share in memory."""
function get_polarization_storage(ctx::AedelstenContext)
    out = zeros(Int64, 4)
    ret = ccall(_sym(:get_polarization_storage_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Int64}), ctx.handle, out)
    ret == 0 || error("Aedelsten: get_polarization_storage failed (code $ret)")
    return (sites=Int(out[1]), in_memory=Int(out[2]), bytes_per_site=Int(out[3]), bytes_to_keep=Int(out[4]))
end

"""`set_resident_field_budget!(ctx; budget, unit)`: the memory budget of this rank's kept field integrals, `budget` in
`unit` ("gib", "mib" or "kib"; 0: none kept), in place of `set_polarization!`'s `resident_budget`: those of the first
sites it holds are computed at the next pass and kept, the rest evaluated in every pass. Collective."""
function set_resident_field_budget!(ctx::AedelstenContext; budget::Integer, unit::AbstractString)
    unit_c::Int32 = _query_i32("byte_unit_" * lowercase(unit))
    ret = ccall(_sym(:set_resident_field_budget_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Cint), ctx.handle, Int32(budget), unit_c)
    ret == 0 || error("Aedelsten: set_resident_field_budget! failed (code $ret)")
    return nothing
end

"""
    compute_induction_gradient(ctx, density; active=nothing) -> Matrix{Float64}

dE_ind/dX at fixed `density`, a freshly allocated `(3, get_n_gradient_centres(ctx))` matrix, from the induced dipoles
of the last `compute_induction` (call it with the same density first). `active` as in `compute_one_electron_gradient`.
"""
function compute_induction_gradient(ctx::AedelstenContext, density::AbstractMatrix; active::Union{Nothing,AbstractVector{Bool}}=nothing)::Matrix{Float64}
    n::Int = get_n_cont(ctx)
    size(density) == (n, n) || error("Aedelsten: density must be $n x $n, got $(size(density))")
    n_centres::Int = get_n_gradient_centres(ctx)
    mask = active === nothing ? nothing : Cint.(active)
    mask === nothing || length(mask) == n_centres || error("Aedelsten: active must have $n_centres flags")
    d = Matrix{Float64}(density)
    gradient = zeros(Float64, 3, n_centres)
    ret = ccall(_sym(:compute_induction_gradient_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cint}),
        ctx.handle, gradient, Int32(n_centres), d, mask === nothing ? C_NULL : mask)
    ret == 0 || error("Aedelsten: failed to compute the induction gradient (code $ret)")
    return gradient
end

"""Linear response of the induction (no static field; the ground-state dipoles are untouched): `V1[D1]` for
one density (n, n) or a batch (n, n, k) -> an array of the same shape. Needs `response_batch >= 1`."""
function compute_induction_response(ctx::AedelstenContext, densities::AbstractArray{<:Real})::Array{Float64}
    ndims(densities) in (2, 3) || error("Aedelsten: densities must be (n, n) or (n, n, k)")
    _check_matrix(ctx, densities, "densities")
    d = _dense(densities)
    out = zeros(Float64, size(d))
    ret = ccall(
        _sym(:compute_induction_response_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}),
        ctx.handle, d, Cint(size(d, 3)), out,
    )
    ret == 0 || error("Aedelsten: compute_induction_response failed (code $ret)")
    return out
end

"""The induction's couplings between (n, n, k) densities: `K[m, n] = -f_m . (alpha^-1 - T)^-1 f_n`, f the electrons'
field at the sites (no static field), equal to `tr(D_m V1[D_n])` without forming any operator; a (k, k) matrix.
Needs `response_batch >= 1`."""
function compute_induction_couplings(ctx::AedelstenContext, densities::AbstractArray{<:Real, 3})::Matrix{Float64}
    _check_matrix(ctx, densities, "densities")
    d = _dense(densities)
    k = size(d, 3)
    out = zeros(Float64, k, k)
    ret = ccall(
        _sym(:compute_induction_couplings_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}),
        ctx.handle, d, Cint(k), out,
    )
    ret == 0 || error("Aedelsten: compute_induction_couplings failed (code $ret)")
    return out
end

"""`compute_induction_states(ctx, densities; threshold=nothing) -> (operators, energies)` for (n, n, k) densities:
each state has its own induced dipoles (kept between calls: keep the state order); weight the results yourself for a
state average. Needs `state_slots >= k`. With `threshold`, the dipoles are solved to it for this call in place of the
setup's."""
function compute_induction_states(ctx::AedelstenContext, densities::AbstractArray{<:Real,3}; threshold::Union{Nothing,Real}=nothing)
    _check_matrix(ctx, densities, "densities")
    d = _dense(densities)
    k = size(d, 3)
    operators = zeros(Float64, size(d)); energies = zeros(Float64, k)
    ret = if threshold === nothing
        ccall(
            _sym(:compute_induction_states_c), Cint,
            (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
            ctx.handle, d, Cint(k), operators, energies,
        )
    else
        ccall(
            _sym(:compute_induction_states_within_c), Cint,
            (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble),
            ctx.handle, d, Cint(k), operators, energies, Cdouble(threshold),
        )
    end
    ret == 0 || error("Aedelsten: compute_induction_states failed (code $ret)")
    return (operators, energies)
end

"""`compute_effective_external_field(ctx) -> (n, n, 3)`: the effective external field (EEF) operators for x, y, z,
the induced dipoles of a uniform unit external field coupled to the electrons, V^(i) = -sum_s mu_s^(i) . t_s; added
to the dipole integrals r_i."""
function compute_effective_external_field(ctx::AedelstenContext)::Array{Float64,3}
    n = ctx.get_n_cont()
    out = zeros(Float64, n, n, 3)
    ret = ccall(_sym(:compute_effective_external_field_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}), ctx.handle, out)
    ret == 0 || error("Aedelsten: compute_effective_external_field failed (code $ret)")
    return out
end

"""The start of the next `compute_induction`'s solve, a (3, n_sites) matrix as `get_induced_dipoles` gives it (another
geometry's dipoles); not a solution until that solve."""
function set_induced_dipoles!(ctx::AedelstenContext, dipoles::AbstractMatrix{<:Real})::AedelstenContext
    n = get_n_polarizable_sites(ctx)
    size(dipoles) == (3, n) || throw(DimensionMismatch("set_induced_dipoles!: dipoles must be (3, $n), got $(size(dipoles))"))
    mu = Matrix{Float64}(dipoles)
    ret = ccall(
        _sym(:set_induced_dipoles_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Int64), ctx.handle, mu, Int64(3n),
    )
    ret == 0 || error("Aedelsten: set_induced_dipoles failed (is the polarization set up?)")
    return ctx
end

"""The number of embedding AIMPs (0 without any)."""
function get_n_embedding_aimps(ctx::AedelstenContext)::Int
    n = ccall(_sym(:get_n_embedding_aimps_c), Int64, (Ptr{AedelstenContextHandle},), ctx.handle)
    n >= 0 || error("Aedelsten: could not count the embedding AIMPs")
    return Int(n)
end

"""The number of centres: real atoms, ghost centres and orthogonality ghosts."""
function get_n_centres(ctx::AedelstenContext)::Int
    n = ccall(_sym(:get_n_centres_c), Int64, (Ptr{AedelstenContextHandle},), ctx.handle)
    n >= 0 || error("Aedelsten: could not count the centres")
    return Int(n)
end

"""Every centre in centre order: atomic numbers, positions (a (3, n) matrix, bohr) and kinds (`:real`, `:ghost` or
`:orthogonality_ghost`)."""
function get_centre_geometry(ctx::AedelstenContext)::Tuple{Vector{Int32}, Matrix{Float64}, Vector{Symbol}}
    n = get_n_centres(ctx)
    atomic_numbers = zeros(Int32, n)
    positions = zeros(Float64, 3, n)
    kinds = zeros(Int32, n)
    ret = ccall(
        _sym(:get_centre_geometry_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Int64),
        ctx.handle, atomic_numbers, positions, kinds, Int64(n),
    )
    ret == 0 || error("Aedelsten: get_centre_geometry failed (code $ret)")
    names = Dict(_query_i32("atom_kind_real") => :real, _query_i32("atom_kind_ghost") => :ghost,
                 _query_i32("atom_kind_orthogonality_ghost") => :orthogonality_ghost)
    return atomic_numbers, positions, [names[k] for k in kinds]
end

"""Every centre's nuclear charge (its nucleus's or its core AIMP's; 0 for a ghost centre or an orthogonality ghost)."""
function get_nuclear_charges(ctx::AedelstenContext)::Vector{Float64}
    charges = zeros(Float64, get_n_centres(ctx))
    ret = ccall(_sym(:get_nuclear_charges_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Int64), ctx.handle, charges, Int64(length(charges)))
    ret == 0 || error("Aedelsten: get_nuclear_charges failed (code $ret)")
    return charges
end

"""Every embedding AIMP's site in input order: atomic numbers (0 when its block's name is not an element symbol),
positions (a (3, n) matrix, bohr) and charges."""
function get_embedding_aimp_sites(ctx::AedelstenContext)::Tuple{Vector{Int32}, Matrix{Float64}, Vector{Float64}}
    n = get_n_embedding_aimps(ctx)
    atomic_numbers = zeros(Int32, n)
    positions = zeros(Float64, 3, n)
    charges = zeros(Float64, n)
    ret = ccall(
        _sym(:get_embedding_aimp_sites_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Int64),
        ctx.handle, atomic_numbers, positions, charges, Int64(n),
    )
    ret == 0 || error("Aedelsten: get_embedding_aimp_sites failed (code $ret)")
    return atomic_numbers, positions, charges
end

"""The induced dipoles of the last `compute_induction`, a (3, n_sites) matrix in input order."""
function get_induced_dipoles(ctx::AedelstenContext)::Matrix{Float64}
    n = get_n_polarizable_sites(ctx)
    mu = zeros(Float64, 3, n)
    ret = ccall(
        _sym(:get_induced_dipoles_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Int64), ctx.handle, mu, Int64(3n),
    )
    ret == 0 || error("Aedelsten: get_induced_dipoles failed (has compute_induction run?)")
    return mu
end

"""Whether the QM region has core AIMPs."""
has_core_aimp(ctx::AedelstenContext)::Bool =
    ccall(_sym(:has_core_aimp_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle) != 0
"""Whether the input has embedding AIMPs."""
has_embedding_aimp(ctx::AedelstenContext)::Bool =
    ccall(_sym(:has_embedding_aimp_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle) != 0

"""Builds the embedding AIMPs' projection operators, needed by their projection and exchange operators; `filename`
must be `nothing` (the AIMPs come from the input)."""
function set_embedding_aimp_operators!(ctx::AedelstenContext; filename::Union{AbstractString,Nothing}=nothing)::AedelstenContext
    ret = if filename === nothing
        ccall(
            _sym(:set_embedding_aimp_operators_c), Cint,
            (Ptr{AedelstenContextHandle}, Ptr{Cvoid}), ctx.handle, C_NULL,
        )
    else
        ccall(
            _sym(:set_embedding_aimp_operators_c), Cint,
            (Ptr{AedelstenContextHandle}, Cstring), ctx.handle, filename,
        )
    end
    ret == 0 || error("Aedelsten: failed to set embedding AIMP operators (code $ret)")
    return ctx
end

"""Frees the embedding AIMPs' projection operators."""
function free_embedding_aimp_operators!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_embedding_aimp_operators_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

# --- One-electron integrals (eager) ---

"""
    compute_one_electron_integrals(ctx, operator; primitive=false, origin=nothing, order=nothing)

Returns a freshly allocated `(n, n)` `Matrix{Float64}`, or for a component
operator an `(n, n, n_planes)` `Array{Float64,3}`. `n` is
`get_n_prim_unique(ctx)` when `primitive=true` (requires
`set_primitive_shell_pairs!` first), else `get_n_cont(ctx)`. `operator` is any
wire-key suffix ("overlap", "kinetic", "nuclear_attraction",
"external_multipole", "moment", "angular_momentum", "nabla", "sap_helfem",
"sap_grasp", ...) or the fully-qualified key ("oneint_overlap"). "sap_helfem"
and "sap_grasp" are the SAP screening, each atom's electrons as Gaussian
charges fitted to nonrelativistic or four-component atoms.

The component operators:
- "moment": the property moments `<mu| (x-O_x)^a (y-O_y)^b (z-O_z)^c |nu>`
  about `origin` for every Cartesian component of `order` (required; order 0
  -> 1 plane, 1 -> 3, 2 -> 6, ...);
- "angular_momentum" (about `origin`) and "nabla": `(n, n, 3)`, planes x, y, z,
  of the real ANTISYMMETRIC `A[mu, nu, k] = <mu| ((r - O) x nabla)_k |nu>`
  (L = -i A) and `nabla[mu, nu, k] = <mu| d/dk |nu>` (p = -i nabla).
`origin` defaults to (0, 0, 0).
"""
function compute_one_electron_integrals(
    ctx::AedelstenContext,
    operator::AbstractString;
    primitive::Bool=false,
    origin::Union{NTuple{3,<:Real},Nothing}=nothing,
    order::Union{Integer,Nothing}=nothing,
)::Union{Matrix{Float64},Array{Float64,3}}
    n::Int = primitive ? get_n_prim_unique(ctx) : get_n_cont(ctx)
    operator_int::Int32 = _resolve_operator(operator)
    (operator_int == _resolve_operator("moment") && order === nothing) && error("Aedelsten: the moment operator needs an order")
    order_c::Int32 = order === nothing ? Int32(0) : Int32(order)
    n_planes::Int32 = ccall(_sym(:compute_one_electron_component_count_c), Cint, (Cint, Cint), operator_int, order_c)
    ox, oy, oz = origin === nothing ? (0.0, 0.0, 0.0) : origin

    # zeros, not undef: shell pairs screened out as negligible are simply
    # never visited by the kernel, so an unwritten entry must already be
    # zero -- true everywhere in this file the output is a screened
    # one-electron quantity, not just here.
    out::Union{Matrix{Float64},Array{Float64,3}} = n_planes > 0 ? zeros(Float64, n, n, n_planes) : zeros(Float64, n, n)

    ret = ccall(
        _sym(:compute_one_electron_integrals_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Cint, Cint, Cdouble, Cdouble, Cdouble, Cint, Cint),
        ctx.handle, out, Int32(n), operator_int, max(n_planes, Int32(1)),
        Float64(ox), Float64(oy), Float64(oz), order_c, Int32(primitive),
    )
    ret == 0 || error("Aedelsten: failed to compute one-electron integrals for '$operator' (code $ret)")
    return out
end

"""
    get_n_gradient_centres(ctx)

The number of centres a gradient has columns for: real atoms and pseudocharge ghosts (orthogonality-function
ghosts, last in the centre list, are not degrees of freedom and not included).
"""
get_n_gradient_centres(ctx::AedelstenContext)::Int =
    Int(ccall(_sym(:get_n_gradient_centres_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle))

function _classical_gradient(ctx::AedelstenContext, symbol::Symbol, label::String, active)::Matrix{Float64}
    n_centres::Int = get_n_gradient_centres(ctx)
    mask = active === nothing ? nothing : Cint.(active)
    mask === nothing || length(mask) == n_centres || error("Aedelsten: active must have $n_centres flags")
    gradient = zeros(Float64, 3, n_centres)
    ret = ccall(_sym(symbol), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cint}),
        ctx.handle, gradient, Int32(n_centres), mask === nothing ? C_NULL : mask)
    ret == 0 || error("Aedelsten: failed to compute the $label gradient (code $ret)")
    return gradient
end

"""
    compute_nuclear_repulsion_gradient(ctx; active=nothing) -> Matrix{Float64}

The gradient of `compute_nuclear_repulsion_energy` (nuclei and core AIMPs, moving with their centres), a freshly
allocated `(3, get_n_gradient_centres(ctx))` matrix. `active` as in `compute_one_electron_gradient`.
"""
compute_nuclear_repulsion_gradient(ctx::AedelstenContext; active::Union{Nothing,AbstractVector{Bool}}=nothing) =
    _classical_gradient(ctx, :compute_nuclear_repulsion_gradient_c, "nuclear repulsion", active)

"""
    compute_nuclear_aimp_embedding_gradient(ctx; active=nothing) -> Matrix{Float64}

The gradient of `compute_nuclear_aimp_embedding_energy` (the QM nuclei and core AIMPs in the fixed embedding AIMPs).
"""
compute_nuclear_aimp_embedding_gradient(ctx::AedelstenContext; active::Union{Nothing,AbstractVector{Bool}}=nothing) =
    _classical_gradient(ctx, :compute_nuclear_aimp_embedding_gradient_c, "nuclear-AIMP embedding", active)

"""
    compute_nuclear_multipole_interaction_gradient(ctx; active=nothing) -> Matrix{Float64}

The gradient of `compute_nuclear_multipole_interaction_energy` (the QM nuclei and core AIMPs in the fixed external
multipoles); needs `set_external_multipoles!`.
"""
compute_nuclear_multipole_interaction_gradient(ctx::AedelstenContext; active::Union{Nothing,AbstractVector{Bool}}=nothing) =
    _classical_gradient(ctx, :compute_nuclear_multipole_interaction_gradient_c, "nuclear-multipole", active)

"""
    set_centre_positions!(ctx, positions)

Moves the gradient centres to `positions`, a `(3, get_n_gradient_centres(ctx))` matrix in Bohr (the layout of the
gradients), for the next geometry step. Each centre carries its basis shells, nucleus and core AIMP; orthogonality
ghosts and embedding AIMPs stay fixed. The geometry-derived data is invalidated: call `set_shell_pairs!` and
`set_eri_data!` (and the primitive shell pairs / X2C construction, if used) again.
"""
function set_centre_positions!(ctx::AedelstenContext, positions::AbstractMatrix{Float64})
    n_centres::Int = get_n_gradient_centres(ctx)
    size(positions) == (3, n_centres) || error("Aedelsten: positions must be 3 x $n_centres, got $(size(positions))")
    p = Matrix{Float64}(positions)
    ret = ccall(_sym(:set_centre_positions_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint), ctx.handle, p, Int32(n_centres))
    ret == 0 || error("Aedelsten: failed to move the centres (code $ret)")
    return nothing
end

"""
    compute_one_electron_gradient(ctx, density, operator)

`sum_mn D_mn dO_mn/dX_c` for one operator, all centres at once: a freshly allocated `(3, n_centres)` matrix,
column X = the x, y, z gradient of centre X. Only the symmetric part of `density` matters; use the energy-weighted
density for "overlap". `active` (one `Bool` per centre; default all) freezes centres: a frozen centre's column is
zero and is not computed, and work that only feeds frozen columns is skipped. Sources that are not centres (external charges/multipoles, embedding AIMPs) do not move and
give no forces here. The moment (property) operator has no gradient.
"""
function compute_one_electron_gradient(
    ctx::AedelstenContext, density::AbstractMatrix{Float64}, operator::AbstractString;
    active::Union{Nothing,AbstractVector{Bool}}=nothing,
)::Matrix{Float64}
    n::Int = get_n_cont(ctx)
    size(density) == (n, n) || error("Aedelsten: density must be $n x $n, got $(size(density))")
    n_centres::Int = get_n_gradient_centres(ctx)
    operator_int::Int32 = _resolve_operator(operator)
    gradient = zeros(Float64, 3, n_centres)
    d = Matrix{Float64}(density)
    mask = active === nothing ? nothing : Cint.(active)
    mask === nothing || length(mask) == n_centres || error("Aedelsten: active must have $n_centres flags")
    ret = ccall(
        _sym(:compute_one_electron_gradient_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Ptr{Cint}),
        ctx.handle, gradient, Int32(n_centres), d, Int32(n), operator_int, mask === nothing ? C_NULL : mask,
    )
    ret == 0 || error("Aedelsten: failed to compute the one-electron gradient for '$operator' (code $ret)")
    return gradient
end

"""The number of shells."""
get_n_shells(ctx::AedelstenContext)::Int = Int(ccall(_sym(:get_n_shells_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle))

"""
    get_shells(ctx) -> Matrix{Int32}

Every shell in AO order, an `(n_shells, 4)` matrix: column 1 the centre (0-based), 2 the angular momentum (negated
for a Cartesian shell), 3 the first AO (0-based), 4 the AO count.
"""
function get_shells(ctx::AedelstenContext)::Matrix{Int32}
    n = get_n_shells(ctx)
    shells = zeros(Int32, n, 4)
    ret = ccall(
        _sym(:get_shells_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint), ctx.handle, shells, Cint(4n),
    )
    ret == 0 || error("Aedelsten: get_shells failed (code $ret)")
    return shells
end

"""
    compute_shell_extents(ctx, threshold) -> Vector{Float64}

Per shell, the distance from its centre beyond which its AOs and every component of their gradients are below
`threshold` in magnitude.
"""
function compute_shell_extents(ctx::AedelstenContext, threshold::Real)::Vector{Float64}
    n = get_n_shells(ctx)
    extents = zeros(Float64, n)
    ret = ccall(
        _sym(:compute_shell_extents_c), Cint,
        (Ptr{AedelstenContextHandle}, Cdouble, Ptr{Cdouble}, Cint), ctx.handle, Float64(threshold), extents, Cint(n),
    )
    ret == 0 || error("Aedelsten: compute_shell_extents failed (code $ret)")
    return extents
end

"""
    compute_ao_values(ctx, shells, points; gradients=false)

The AOs of the shells `shells` (0-based, any order and subset) at `points`, `(3, n_points)` or `(4, n_points)` (x, y,
z and a weight that is not read): the values, `(n_listed, n_points)`, the AOs of the listed shells in the order of
the list; with `gradients`, the tuple of the values and an `(n_listed, n_points, 3)` array (x, y, z).
"""
function compute_ao_values(ctx::AedelstenContext, shells::AbstractVector{<:Integer}, points::Matrix{Float64}; gradients::Bool=false)
    rows = size(points, 1)
    rows in (3, 4) || error("Aedelsten: points must be shaped (3, n_points) or (4, n_points)")
    n_points = size(points, 2)
    shell_list = Int32.(collect(shells))
    counts = get_shells(ctx)[:, 4]
    n_listed = isempty(shell_list) ? 0 : Int(sum(counts[s + 1] for s in shell_list))
    values = zeros(Float64, n_listed, n_points)
    grads = gradients ? zeros(Float64, n_listed, n_points, 3) : nothing
    ret = ccall(
        _sym(:compute_ao_values_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint, Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint),
        ctx.handle, shell_list, Cint(length(shell_list)), points, Cint(rows), Cint(n_points),
        values, gradients ? grads : Ptr{Cdouble}(C_NULL), Cint(n_listed),
    )
    ret == 0 || error("Aedelsten: compute_ao_values failed (code $ret)")
    return gradients ? (values, grads) : values
end

"""
    compute_ao_hessians(ctx, shells, points)

As `compute_ao_values` with gradients, and the second derivatives: the tuple of the values `(n_listed, n_points)`, the
gradients `(n_listed, n_points, 3)` (x, y, z) and the second derivatives `(n_listed, n_points, 6)` (xx, xy, xz, yy,
yz, zz).
"""
function compute_ao_hessians(ctx::AedelstenContext, shells::AbstractVector{<:Integer}, points::Matrix{Float64})
    rows = size(points, 1)
    rows in (3, 4) || error("Aedelsten: points must be shaped (3, n_points) or (4, n_points)")
    n_points = size(points, 2)
    shell_list = Int32.(collect(shells))
    counts = get_shells(ctx)[:, 4]
    n_listed = isempty(shell_list) ? 0 : Int(sum(counts[s + 1] for s in shell_list))
    values = zeros(Float64, n_listed, n_points)
    grads = zeros(Float64, n_listed, n_points, 3)
    hessians = zeros(Float64, n_listed, n_points, 6)
    ret = ccall(
        _sym(:compute_ao_hessians_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint, Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint),
        ctx.handle, shell_list, Cint(length(shell_list)), points, Cint(rows), Cint(n_points),
        values, grads, hessians, Cint(n_listed),
    )
    ret == 0 || error("Aedelsten: compute_ao_hessians failed (code $ret)")
    return (values, grads, hessians)
end

# --- X2C: picture-change integrals, Hamiltonian construction, contraction ---

"""
    compute_one_electron_picture_change_integrals(ctx, operator; spin_orbit=false,
        origin=nothing, order=nothing)

Relativistic picture-change integrals, always over the decontracted
(primitive) basis (`n = get_n_prim_unique(ctx)`, requires
`set_primitive_shell_pairs!` first -- unlike `compute_one_electron_integrals`
there's no separate `primitive` toggle). Returns a freshly allocated
`(n, n)` `Matrix{Float64}`, or an `(n, n, n_planes)` `Array{Float64,3}`
whenever `spin_orbit` and/or the moment's `order` push the plane count above
1 (`spin_orbit` alone: 4 planes (x, y, z, scalar); combined with a
moment order: 4x the order's own component count). `origin`/`order`: the
"moment" operator's.
"""
function compute_one_electron_picture_change_integrals(
    ctx::AedelstenContext,
    operator::AbstractString;
    spin_orbit::Bool=false,
    origin::Union{NTuple{3,<:Real},Nothing}=nothing,
    order::Union{Integer,Nothing}=nothing,
)::Union{Matrix{Float64},Array{Float64,3}}
    n::Int = get_n_prim_unique(ctx)
    operator_int::Int32 = _resolve_operator(operator)

    n_planes_component::Int32 = if order !== nothing
        Int32(ccall(_sym(:compute_moment_component_count_c), Cint, (Cint,), Int32(order)))
    else
        Int32(1)
    end
    n_planes::Int32 = spin_orbit ? n_planes_component * Int32(4) : n_planes_component
    order_c::Int32 = order === nothing ? Int32(0) : Int32(order)
    ox, oy, oz = origin === nothing ? (0.0, 0.0, 0.0) : origin

    out::Union{Matrix{Float64},Array{Float64,3}} = n_planes == 1 ?
        zeros(Float64, n, n) : zeros(Float64, n, n, n_planes)

    ret = ccall(
        _sym(:compute_one_electron_picture_change_integrals_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Cint, Cint, Cdouble, Cdouble, Cdouble, Cint, Cint),
        ctx.handle, out, Int32(n), operator_int, n_planes,
        Float64(ox), Float64(oy), Float64(oz), order_c, Int32(spin_orbit),
    )
    ret == 0 || error("Aedelsten: failed to compute one-electron picture-change integrals for '$operator' (code $ret)")
    return out
end

"""
    construct_x2c_one_electron_hamiltonian(ctx, s, t, v, pvp; c_light=nothing)

Builds and returns the spin-free X2C one-electron Hamiltonian (an
`(n, n)` `Matrix{Float64}`, `n = get_n_prim_unique(ctx)`) from
primitive-basis overlap/kinetic/nuclear-attraction/pVp matrices (`s`,
`t`, `v` from `compute_one_electron_integrals(ctx, ...; primitive=true)`,
`pvp` from `compute_one_electron_picture_change_integrals(ctx, ...;
spin_orbit=false)` on the same operator), and caches the resulting
picture-change transform for later use. `c_light` defaults to the
library's own value. Errors if a transform is already cached:
`free_picture_change!` first. With symmetry set, `s`/`t`/`v`/`pvp` must be
symmetric (an unaveraged environment in `v` is refused) and the outputs come
back averaged.
"""
function construct_x2c_one_electron_hamiltonian(
    ctx::AedelstenContext,
    s::Matrix{Float64}, t::Matrix{Float64}, v::Matrix{Float64}, pvp::Matrix{Float64};
    c_light::Union{Real,Nothing}=nothing,
)::Matrix{Float64}
    n::Int = size(s, 1)
    (size(s) == size(t) == size(v) == size(pvp) && size(s, 1) == size(s, 2)) ||
        error("Aedelsten: s/t/v/pvp must all be square and the same shape")
    oneham = zeros(Float64, n, n)
    c_light_wire::Float64 = c_light === nothing ? -1.0 : Float64(c_light)  # sentinel: use the library default
    ret = ccall(
        _sym(:construct_x2c_one_electron_hamiltonian_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble),
        ctx.handle, oneham, s, t, v, pvp, Int32(n), c_light_wire,
    )
    ret == 0 || error("Aedelsten: failed to construct the X2C one-electron Hamiltonian (code $ret)")
    return oneham
end

# Contracts a primitive-basis result (e.g. from compute_one_electron_integrals
# (...; primitive=true) or compute_one_electron_picture_change_integrals)
# into the contracted basis. n_prim/n_cont come from primitive's own shape
# and get_n_cont(ctx) respectively; requires set_primitive_shell_pairs!
# first. Unlike Fortran's GENERIC dispatch on rank, this is just ordinary
# multiple dispatch on primitive's own type.

function contract_one_electron_integrals(ctx::AedelstenContext, primitive::Matrix{Float64})::Matrix{Float64}
    size(primitive, 1) == size(primitive, 2) || error("Aedelsten: primitive input matrix must be square")
    n_prim::Int = size(primitive, 1)
    n_cont::Int = get_n_cont(ctx)
    output = zeros(Float64, n_cont, n_cont)
    ret = ccall(
        _sym(:contract_one_electron_matrix_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint),
        ctx.handle, primitive, Int32(n_prim), output, Int32(n_cont),
    )
    ret == 0 || error("Aedelsten: failed to contract one-electron integrals (code $ret)")
    return output
end

function contract_one_electron_integrals(ctx::AedelstenContext, primitive::Array{Float64,3})::Array{Float64,3}
    size(primitive, 1) == size(primitive, 2) ||
        error("Aedelsten: primitive input tensor must be square in its first two dimensions")
    n_prim::Int = size(primitive, 1)
    n_planes::Int = size(primitive, 3)
    n_cont::Int = get_n_cont(ctx)
    output = zeros(Float64, n_cont, n_cont, n_planes)
    ret = ccall(
        _sym(:contract_one_electron_array3d_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}, Cint),
        ctx.handle, primitive, Int32(n_prim), Int32(n_planes), output, Int32(n_cont),
    )
    ret == 0 || error("Aedelsten: failed to contract one-electron integrals (code $ret)")
    return output
end

# --- Two-electron repulsion integrals (ERIs) ---

"""
    compute_eri_stored(ctx; range=nothing, omega=nothing)

Returns a freshly allocated dense `(n, n, n, n)` `Array{Float64,4}` of
two-electron repulsion integrals (S1, no packing; for a basis whose tensor
does not fit in memory, the direct builds `compute_fock_direct` and
`compute_fock_direct_unrestricted` form none). `range`/`omega` select a range-separated
operator ("short_range"/"long_range" with the given `omega`); both or
neither must be given (neither = full-range Coulomb).

Under MPI the call is collective: every rank must make it; each computes its
share of the integrals and every rank gets the whole tensor.
"""
function compute_eri_stored(
    ctx::AedelstenContext; range::Union{AbstractString,Nothing}=nothing, omega::Union{Real,Nothing}=nothing
)::Array{Float64,4}
    n::Int = get_n_cont(ctx)
    symmetry_int::Int32 = _query_i32("eri_ao_symmetry_s1")
    index_order_int::Int32 = _query_i32("eri_index_order_column_major")
    range_wire, omega_wire = _resolve_eri_range(range, omega)
    out = zeros(Float64, n, n, n, n)
    ret = ccall(
        _sym(:compute_eri_stored_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Cint, Cint, Cint, Cdouble),
        ctx.handle, out, Int32(n), symmetry_int, index_order_int, range_wire, omega_wire,
    )
    ret == 0 || error("Aedelsten: failed to compute ERIs (code $ret)")
    return out
end

# --- J/K from a dense in-memory ERI tensor (free functions) ---
# None of these take ctx: each is purely a function of its own arguments
# (density and eri_dense just need to agree on shape), exactly like the
# Fortran interface's free-function design for these -- no dot-call
# sugar, no context needed.

_check_square_and_shares_n(density::Matrix{Float64}, eri_dense::Array{Float64,4})::Int = begin
    size(density, 1) == size(density, 2) || error("Aedelsten: density must be square")
    n = size(density, 1)
    all(size(eri_dense, d) == n for d in 1:4) ||
        error("Aedelsten: eri_dense must be square in all four dimensions and share n with density")
    n
end

"""
    compute_j_and_k_matrix(density, eri_dense, j_coeff, k_coeff) -> (j, k)

Builds the Coulomb (J) and exchange (K) matrices from a dense in-memory
ERI tensor: `j = j_coeff * J(density)`, `k = k_coeff * K(density)`.
"""
function compute_j_and_k_matrix(
    density::Matrix{Float64}, eri_dense::Array{Float64,4}, j_coeff::Real, k_coeff::Real
)::Tuple{Matrix{Float64},Matrix{Float64}}
    n = _check_square_and_shares_n(density, eri_dense)
    j = zeros(Float64, n, n)
    k = zeros(Float64, n, n)
    ret = ccall(
        _sym(:compute_j_and_k_matrix_stored_c), Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble, Cdouble),
        j, k, density, eri_dense, Int32(n), Float64(j_coeff), Float64(k_coeff),
    )
    ret == 0 || error("Aedelsten: failed to construct the J and K matrices (code $ret)")
    return (j, k)
end

"""
    compute_j_and_k_unrestricted_matrix(density_alpha, density_beta, density_total,
        eri_dense, j_coeff, k_coeff) -> (j, k_alpha, k_beta)

Single pass over `eri_dense` for J, K_alpha and K_beta together -- prefer
this over separate `compute_j_matrix` + `compute_k_matrix_unrestricted`
calls when all three are wanted. `density_total` is J's own density
(alpha + beta, the caller's to compute), same requirement as
`compute_k_matrix_unrestricted`'s `density_alpha`/`density_beta` split.
"""
function compute_j_and_k_unrestricted_matrix(
    density_alpha::Matrix{Float64}, density_beta::Matrix{Float64}, density_total::Matrix{Float64},
    eri_dense::Array{Float64,4}, j_coeff::Real, k_coeff::Real,
)::Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    (size(density_alpha) == size(density_beta) == size(density_total)) ||
        error("Aedelsten: density_alpha/density_beta/density_total must all be the same shape")
    n = _check_square_and_shares_n(density_alpha, eri_dense)
    j = zeros(Float64, n, n)
    k_alpha = zeros(Float64, n, n)
    k_beta = zeros(Float64, n, n)
    ret = ccall(
        _sym(:compute_j_and_k_unrestricted_matrix_stored_c), Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
         Cint, Cdouble, Cdouble),
        j, k_alpha, k_beta, density_alpha, density_beta, density_total, eri_dense, Int32(n),
        Float64(j_coeff), Float64(k_coeff),
    )
    ret == 0 || error("Aedelsten: failed to construct the unrestricted J and K matrices (code $ret)")
    return (j, k_alpha, k_beta)
end

"""`coeff * J(density)` from the dense `(n, n, n, n)` ERI tensor `eri_dense`."""
function compute_j_matrix(density::Matrix{Float64}, eri_dense::Array{Float64,4}, coeff::Real)::Matrix{Float64}
    n = _check_square_and_shares_n(density, eri_dense)
    j = zeros(Float64, n, n)
    ret = ccall(
        _sym(:compute_j_matrix_stored_c), Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble),
        j, density, eri_dense, Int32(n), Float64(coeff),
    )
    ret == 0 || error("Aedelsten: failed to construct the J matrix (code $ret)")
    return j
end

"""`coeff * K(density)` from the dense `(n, n, n, n)` ERI tensor `eri_dense`."""
function compute_k_matrix(density::Matrix{Float64}, eri_dense::Array{Float64,4}, coeff::Real)::Matrix{Float64}
    n = _check_square_and_shares_n(density, eri_dense)
    k = zeros(Float64, n, n)
    ret = ccall(
        _sym(:compute_k_matrix_stored_c), Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble),
        k, density, eri_dense, Int32(n), Float64(coeff),
    )
    ret == 0 || error("Aedelsten: failed to construct the K matrix (code $ret)")
    return k
end

function compute_k_matrix_unrestricted(
    density_alpha::Matrix{Float64}, density_beta::Matrix{Float64}, eri_dense::Array{Float64,4}, coeff::Real
)::Tuple{Matrix{Float64},Matrix{Float64}}
    size(density_alpha) == size(density_beta) || error("Aedelsten: density_alpha/density_beta must be the same shape")
    n = _check_square_and_shares_n(density_alpha, eri_dense)
    k_alpha = zeros(Float64, n, n)
    k_beta = zeros(Float64, n, n)
    ret = ccall(
        _sym(:compute_k_matrix_unrestricted_stored_c), Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble),
        k_alpha, k_beta, density_alpha, density_beta, eri_dense, Int32(n), Float64(coeff),
    )
    ret == 0 || error("Aedelsten: failed to construct the unrestricted K matrices (code $ret)")
    return (k_alpha, k_beta)
end

# --- Direct Fock construction (no stored ERI tensor) ---
# Unlike the free J/K builders above, these are context-bound: they
# genuinely need context state (the shell pairs and cached ERI data),
# integrals are evaluated and immediately folded into J/K one shell
# quartet at a time. Requires set_eri_data! first.

function set_eri_data!(ctx::AedelstenContext; schwarz_threshold::Union{Real,Nothing}=nothing)::AedelstenContext
    threshold_wire::Float64 = schwarz_threshold === nothing ? -1.0 : Float64(schwarz_threshold)  # sentinel: library default
    ret = ccall(_sym(:set_eri_data_c), Cint, (Ptr{AedelstenContextHandle}, Cdouble), ctx.handle, threshold_wire)
    ret == 0 || error("Aedelsten: failed to set ERI data (code $ret)")
    return ctx
end

"""Frees the ERI data."""
function free_eri_data!(ctx::AedelstenContext)::AedelstenContext
    ccall(_sym(:free_eri_data_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)
    return ctx
end

"""
    compute_fock_direct(ctx, density, j_coeff, k_coeff; range=nothing,
        omega=nothing, pairwise_threshold=nothing) -> (j, k)

Restricted direct Fock construction: `j = j_coeff * J(density)`,
`k = k_coeff * K(density)`, with no dense ERI tensor ever materialized.
Requires `set_eri_data!` first.
"""
function compute_fock_direct(
    ctx::AedelstenContext, density::Matrix{Float64}, j_coeff::Real, k_coeff::Real;
    range::Union{AbstractString,Nothing}=nothing, omega::Union{Real,Nothing}=nothing,
    pairwise_threshold::Union{Real,Nothing}=nothing,
)::Tuple{Matrix{Float64},Matrix{Float64}}
    size(density, 1) == size(density, 2) || error("Aedelsten: density must be square")
    n::Int = size(density, 1)
    range_wire, omega_wire = _resolve_eri_range(range, omega)
    pairwise_threshold_wire::Float64 = pairwise_threshold === nothing ? -1.0 : Float64(pairwise_threshold)
    j = zeros(Float64, n, n)
    k = zeros(Float64, n, n)
    ret = ccall(
        _sym(:compute_fock_direct_restricted_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble, Cdouble,
         Cint, Cdouble, Cdouble),
        ctx.handle, j, k, density, Int32(n), Float64(j_coeff), Float64(k_coeff),
        range_wire, omega_wire, pairwise_threshold_wire,
    )
    ret == 0 || error("Aedelsten: failed to compute the direct Fock matrix (code $ret)")
    return (j, k)
end

"""
    compute_fock_direct_general(ctx, density, j_coeff, k_coeff; range=nothing,
        omega=nothing, pairwise_threshold=nothing) -> (j, k)

`compute_fock_direct` for a density of any symmetry (a transition density, the
spin-flip density of a stability analysis): `j = j_coeff * J(density)`, from the
density's symmetric part, and `k = k_coeff * K(density)` with
`K[a, c] = sum_bd density[b, d] (ab|cd)`, which is not symmetric for a
non-symmetric density and is not symmetrized. A symmetric density gives
`compute_fock_direct`'s J and K. Requires `set_eri_data!` first.
"""
function compute_fock_direct_general(
    ctx::AedelstenContext, density::Matrix{Float64}, j_coeff::Real, k_coeff::Real;
    range::Union{AbstractString,Nothing}=nothing, omega::Union{Real,Nothing}=nothing,
    pairwise_threshold::Union{Real,Nothing}=nothing,
)::Tuple{Matrix{Float64},Matrix{Float64}}
    size(density, 1) == size(density, 2) || error("Aedelsten: density must be square")
    n::Int = size(density, 1)
    range_wire, omega_wire = _resolve_eri_range(range, omega)
    pairwise_threshold_wire::Float64 = pairwise_threshold === nothing ? -1.0 : Float64(pairwise_threshold)
    j = zeros(Float64, n, n)
    k = zeros(Float64, n, n)
    ret = ccall(
        _sym(:compute_fock_direct_general_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble, Cdouble,
         Cint, Cdouble, Cdouble),
        ctx.handle, j, k, density, Int32(n), Float64(j_coeff), Float64(k_coeff),
        range_wire, omega_wire, pairwise_threshold_wire,
    )
    ret == 0 || error("Aedelsten: failed to compute the direct Fock matrices of a general density (code $ret)")
    return (j, k)
end

"""
    compute_fock_direct_general_multi(ctx, densities, j_coeff, k_coeff; range=nothing,
        omega=nothing, pairwise_threshold=nothing) -> (j, k)

`compute_fock_direct_general` for several densities in one direct walk:
`densities` is `n x n x n_mat`, and `j[:, :, m]`, `k[:, :, m]` are the J and K of
`densities[:, :, m]` as `compute_fock_direct_general` gives them (to within the
screening, which keeps an integral batch any density needs). Requires
`set_eri_data!` first.
"""
function compute_fock_direct_general_multi(
    ctx::AedelstenContext, densities::Array{Float64,3}, j_coeff::Real, k_coeff::Real;
    range::Union{AbstractString,Nothing}=nothing, omega::Union{Real,Nothing}=nothing,
    pairwise_threshold::Union{Real,Nothing}=nothing,
)::Tuple{Array{Float64,3},Array{Float64,3}}
    size(densities, 1) == size(densities, 2) || error("Aedelsten: densities must be n x n x n_mat")
    n::Int, n_mat::Int = size(densities, 1), size(densities, 3)
    n_mat >= 1 || error("Aedelsten: at least one density")
    range_wire, omega_wire = _resolve_eri_range(range, omega)
    pairwise_threshold_wire::Float64 = pairwise_threshold === nothing ? -1.0 : Float64(pairwise_threshold)
    j = zeros(Float64, n, n, n_mat)
    k = zeros(Float64, n, n, n_mat)
    ret = ccall(
        _sym(:compute_fock_direct_general_multi_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble,
         Cint, Cdouble, Cdouble),
        ctx.handle, j, k, densities, Int32(n), Int32(n_mat), Float64(j_coeff), Float64(k_coeff),
        range_wire, omega_wire, pairwise_threshold_wire,
    )
    ret == 0 || error("Aedelsten: failed to compute the direct Fock matrices of general densities (code $ret)")
    return (j, k)
end

"""
    compute_fock_direct_mixed_active_space(ctx, density, c_bra, c_ket, j_coeff, k_coeff;
        active_threshold=nothing, pairwise_threshold=nothing) -> (j, k, m)

The integrals of a state interaction between two runs with different orbitals,
in one direct walk (full range): `j`, `k` of `density` as in
`compute_fock_direct_general`, and the Coulomb matrices of the active pair
densities `m[v + (w-1) n_active, mu nu] = M(vw, mu nu) = sum_{lambda sigma}
P[lambda sigma, vw] (mu nu|lambda sigma)`, `P[lambda sigma, vw] =
(c_bra[lambda, v] c_ket[sigma, w] + c_bra[sigma, v] c_ket[lambda, w]) / 2`, for
every pair `(v, w)`, packed over the function pairs `mu >= nu` (column
`(nu-1) n - (nu-1)(nu-2)/2 + mu - nu + 1`). `transform_pair_coulomb_one_index(ctx, m, c_ket)`
gives `t[u, vw, mu]`, and the caller finishes `(tu|vw) = sum_mu c_bra[mu, t] t[u, vw, mu]`
with one product (`t`, `v` the bra's orbitals, `u`, `w` the ket's). `c_bra`,
`c_ket` are `n x n_active`; `m` is `n_active^2 x n (n + 1)/2`. `active_threshold`:
a batch whose Schwarz bound times its largest active coefficients is below it adds
nothing to `M`; `nothing` = every batch. Requires `set_eri_data!` first.
"""
function compute_fock_direct_mixed_active_space(
    ctx::AedelstenContext, density::Matrix{Float64}, c_bra::Matrix{Float64}, c_ket::Matrix{Float64},
    j_coeff::Real, k_coeff::Real;
    active_threshold::Union{Real,Nothing}=nothing, pairwise_threshold::Union{Real,Nothing}=nothing,
)::Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    size(density, 1) == size(density, 2) || error("Aedelsten: density must be square")
    n::Int = size(density, 1)
    (size(c_bra) == size(c_ket) && size(c_ket, 1) == n) || error("Aedelsten: c_bra and c_ket must both be n x n_active")
    n_active::Int = size(c_ket, 2)
    wire(x) = x === nothing ? -1.0 : Float64(x)
    j = zeros(Float64, n, n)
    k = zeros(Float64, n, n)
    m = zeros(Float64, n_active * n_active, n * (n + 1) ÷ 2)
    ret = ccall(
        _sym(:compute_fock_direct_mixed_active_space_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
         Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble, Cdouble, Cdouble),
        ctx.handle, j, k, m, density, c_bra, c_ket, Int32(n), Int32(n_active), Float64(j_coeff), Float64(k_coeff),
        wire(active_threshold), wire(pairwise_threshold),
    )
    ret == 0 || error("Aedelsten: failed to compute the mixed-basis active-space integrals (code $ret)")
    return (j, k, m)
end

"""
    compute_fock_direct_unrestricted(ctx, density_alpha, density_beta, density_total,
        j_coeff, k_coeff; range=nothing, omega=nothing, pairwise_threshold=nothing)
        -> (j, k_alpha, k_beta)

Unrestricted direct Fock construction. `density_total` is J's own
density -- the caller's to supply (same requirement as
`compute_j_and_k_unrestricted_matrix`'s `density_total`), not derived
here. Requires `set_eri_data!` first.
"""
function compute_fock_direct_unrestricted(
    ctx::AedelstenContext, density_alpha::Matrix{Float64}, density_beta::Matrix{Float64}, density_total::Matrix{Float64},
    j_coeff::Real, k_coeff::Real;
    range::Union{AbstractString,Nothing}=nothing, omega::Union{Real,Nothing}=nothing,
    pairwise_threshold::Union{Real,Nothing}=nothing,
)::Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    (size(density_alpha) == size(density_beta) == size(density_total)) ||
        error("Aedelsten: density_alpha/density_beta/density_total must all be the same shape")
    size(density_alpha, 1) == size(density_alpha, 2) || error("Aedelsten: density_alpha must be square")
    n::Int = size(density_alpha, 1)
    range_wire, omega_wire = _resolve_eri_range(range, omega)
    pairwise_threshold_wire::Float64 = pairwise_threshold === nothing ? -1.0 : Float64(pairwise_threshold)
    j = zeros(Float64, n, n)
    k_alpha = zeros(Float64, n, n)
    k_beta = zeros(Float64, n, n)
    ret = ccall(
        _sym(:compute_fock_direct_unrestricted_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
         Cint, Cdouble, Cdouble, Cint, Cdouble, Cdouble),
        ctx.handle, j, k_alpha, k_beta, density_alpha, density_beta, density_total, Int32(n),
        Float64(j_coeff), Float64(k_coeff), range_wire, omega_wire, pairwise_threshold_wire,
    )
    ret == 0 || error("Aedelsten: failed to compute the unrestricted direct Fock matrices (code $ret)")
    return (j, k_alpha, k_beta)
end

"""
    compute_fock_direct_active_space(ctx, density_inactive, density_active, c_active, coeff_j_inactive,
        coeff_k_inactive, coeff_j_active, coeff_k_active; active_threshold=nothing, pairwise_threshold=nothing)
        -> (j_inactive, k_inactive, j_active, k_active, m)

The active-space integrals of a CASSCF macro-iteration in one direct walk (full
range): the inactive and active J/K, each against its own coefficients as in
`compute_fock_direct`, and the Coulomb matrices of the active pair densities
`m[vw, mu nu] = M(vw, mu nu) = sum_{lambda sigma} P[lambda sigma, vw] (mu nu|lambda sigma)`,
`P[lambda sigma, vw] = (c_active[lambda, v] c_active[sigma, w] + c_active[sigma, v] c_active[lambda, w]) / 2`,
for the pairs `v >= w` at `vw = v (v - 1)/2 + w` (1-based), packed over the
function pairs `mu >= nu` (column `(nu-1) n - (nu-1)(nu-2)/2 + mu - nu + 1`).
`transform_pair_coulomb_one_index(ctx, m, c_active)` gives `t[u, vw, mu]`, and the
caller finishes `(pu|vw) = sum_mu C[mu, p] t[u, vw, mu]` with one product.
`c_active` is `n x n_active`; `m` is `n_active (n_active + 1)/2 x n (n + 1)/2`.
`active_threshold`: a batch whose Schwarz bound times its largest active coefficients
is below it adds nothing to `M`; `nothing` = every batch. Requires
`set_eri_data!` first.
"""
function compute_fock_direct_active_space(
    ctx::AedelstenContext, density_inactive::Matrix{Float64}, density_active::Matrix{Float64}, c_active::Matrix{Float64},
    coeff_j_inactive::Real, coeff_k_inactive::Real, coeff_j_active::Real, coeff_k_active::Real;
    active_threshold::Union{Real,Nothing}=nothing, pairwise_threshold::Union{Real,Nothing}=nothing,
)
    size(density_inactive, 1) == size(density_inactive, 2) || error("Aedelsten: the densities must be square")
    n::Int = size(density_inactive, 1)
    (size(density_active) == size(density_inactive) && size(c_active, 1) == n) ||
        error("Aedelsten: the densities must be n x n and c_active n x n_active")
    n_active::Int = size(c_active, 2)
    wire(x) = x === nothing ? -1.0 : Float64(x)
    j_inactive, k_inactive, j_active, k_active = (zeros(Float64, n, n) for _ in 1:4)
    m = zeros(Float64, n_active * (n_active + 1) ÷ 2, n * (n + 1) ÷ 2)
    ret = ccall(
        _sym(:compute_fock_direct_active_space_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
         Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble),
        ctx.handle, j_inactive, k_inactive, j_active, k_active, m, density_inactive, density_active, c_active,
        Int32(n), Int32(n_active), Float64(coeff_j_inactive), Float64(coeff_k_inactive), Float64(coeff_j_active),
        Float64(coeff_k_active), wire(active_threshold), wire(pairwise_threshold),
    )
    ret == 0 || error("Aedelsten: failed to compute the active-space integrals (code $ret)")
    return (j_inactive, k_inactive, j_active, k_active, m)
end

"""
    transform_pair_coulomb_one_index(ctx, m, c) -> Array{Float64,3}

One index of the active walks' `M` transformed: `t[u, vw, mu] = sum_nu c[nu, u] M(vw, mu nu)`.
`m` is `n_pairs x n (n + 1)/2` as `compute_fock_direct_active_space` (or the mixed
walk) returns it; `c` is `n x n_columns`; `t` is `n_columns x n_pairs x n`. Rank-local.
"""
function transform_pair_coulomb_one_index(ctx::AedelstenContext, m::Matrix{Float64}, c::Matrix{Float64})::Array{Float64,3}
    n::Int = size(c, 1)
    size(m, 2) == n * (n + 1) ÷ 2 || error("Aedelsten: m must be n_pairs x n (n + 1)/2 for c's n rows")
    n_pairs::Int, n_columns::Int = size(m, 1), size(c, 2)
    t = zeros(Float64, n_columns, n_pairs, n)
    ret = ccall(
        _sym(:transform_pair_coulomb_one_index_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cint),
        ctx.handle, t, m, c, Int32(n), Int32(n_pairs), Int32(n_columns),
    )
    ret == 0 || error("Aedelsten: failed to transform the pair densities' Coulomb matrices (code $ret)")
    return t
end

"""
    compute_eri_half_transformed(ctx, c_left, c_right; threshold=nothing, pairwise_threshold=nothing,
        partition_over_ranks=true) -> Array{Float64,4}

The half-transformed ERIs from one direct walk (full range, no J/K): `x[p, nu, r, sigma] = sum_{mu lambda}
c_left[mu, p] c_right[lambda, r] (mu nu|lambda sigma)`, `(m_left, n, m_right, n)`. `c_left` is `n x m_left`, `c_right`
`n x m_right` (e.g. column blocks of the orbital matrix). The same matrix for both (`c_left === c_right`) takes the
symmetric path, `x[p, nu, r, sigma] = x[r, sigma, p, nu]`, at half the work. `threshold`: a batch whose Schwarz bound
times the largest |c_left| on one of its shell pairs times the largest |c_right| on the other is below it is skipped;
`nothing` = every batch. `partition_over_ranks`: under MPI, `true` makes the call collective (every rank gets the whole
x), `false` computes x on this rank alone. The scratch holds the walk's row buffers besides its usual scratch.
Requires `set_eri_data!` first.
"""
function compute_eri_half_transformed(
    ctx::AedelstenContext, c_left::Matrix{Float64}, c_right::Matrix{Float64};
    threshold::Union{Real,Nothing}=nothing, pairwise_threshold::Union{Real,Nothing}=nothing, partition_over_ranks::Bool=true,
)
    n = get_n_cont(ctx)
    (size(c_left, 1) == n && size(c_right, 1) == n) || error("Aedelsten: c_left and c_right must be n x m")
    m_left, m_right = size(c_left, 2), size(c_right, 2)
    wire(x) = x === nothing ? -1.0 : Float64(x)
    x = zeros(Float64, m_left, n, m_right, n)
    ret = ccall(
        _sym(:compute_eri_half_transformed_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble, Cint),
        ctx.handle, x, c_left, Int32(m_left), c_right, Int32(m_right), Int32(n), wire(threshold), wire(pairwise_threshold),
        Int32(partition_over_ranks ? 1 : 0),
    )
    ret == 0 || error("Aedelsten: failed to compute the half-transformed integrals (code $ret)")
    return x
end

"""
    compute_eri_half_transformed_on_node!(ctx, x, c_left, c_right; threshold=nothing, pairwise_threshold=nothing) -> x

`compute_eri_half_transformed` by the ranks of the context's communicator together, into one `x` in their node's
shared memory: `x` is an `(m_left, n, m_right, n)` reshape of an `AedelstenSharedArray`'s `array` of the same
communicator, the ranks share one node, and every rank has the whole X when it returns. Collective.
"""
function compute_eri_half_transformed_on_node!(
    ctx::AedelstenContext, x::Array{Float64,4}, c_left::Matrix{Float64}, c_right::Matrix{Float64};
    threshold::Union{Real,Nothing}=nothing, pairwise_threshold::Union{Real,Nothing}=nothing,
)
    n = get_n_cont(ctx)
    (size(c_left, 1) == n && size(c_right, 1) == n) || error("Aedelsten: c_left and c_right must be n x m")
    m_left, m_right = size(c_left, 2), size(c_right, 2)
    size(x) == (m_left, n, m_right, n) || error("Aedelsten: x must be (m_left, n, m_right, n)")
    wire(v) = v === nothing ? -1.0 : Float64(v)
    ret = ccall(
        _sym(:compute_eri_half_transformed_on_node_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble),
        ctx.handle, x, c_left, Int32(m_left), c_right, Int32(m_right), Int32(n), wire(threshold), wire(pairwise_threshold),
    )
    ret == 0 || error("Aedelsten: failed to compute the half-transformed integrals on the node (code $ret)")
    return x
end

"""
    compute_eri_half_transformed_sigma(ctx, c_left, c_right, sigma_start, sigma_end; threshold=nothing,
        pairwise_threshold=nothing, partition_over_ranks=true) -> Array{Float64,4}

`compute_eri_half_transformed` for a batch of sigma: `x[p, nu, r, s] = X(p, nu, r, sigma_start + s - 1)` for the sigma
in `sigma_start:sigma_end-1` only (0-based AOs as `get_shells` gives them, a run of whole shells: each end a shell's
first AO or n; not empty), `(m_left, n, m_right, sigma_end - sigma_start)`. The walk computes only the quartets with a
shell in the batch, so batches that cover the basis compute each quartet at most four times in all. The other
arguments as `compute_eri_half_transformed`; the scratch beyond the walk's own: `half_transform_scratch_bytes`.
"""
function compute_eri_half_transformed_sigma(
    ctx::AedelstenContext, c_left::Matrix{Float64}, c_right::Matrix{Float64}, sigma_start::Integer, sigma_end::Integer;
    threshold::Union{Real,Nothing}=nothing, pairwise_threshold::Union{Real,Nothing}=nothing, partition_over_ranks::Bool=true,
)
    n = get_n_cont(ctx)
    (size(c_left, 1) == n && size(c_right, 1) == n) || error("Aedelsten: c_left and c_right must be n x m")
    0 <= sigma_start < sigma_end <= n || error("Aedelsten: sigma $sigma_start:$sigma_end is not a run of the $n AOs")
    m_left, m_right = size(c_left, 2), size(c_right, 2)
    wire(v) = v === nothing ? -1.0 : Float64(v)
    x = zeros(Float64, m_left, n, m_right, sigma_end - sigma_start)
    ret = ccall(
        _sym(:compute_eri_half_transformed_sigma_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Cint, Cint, Cdouble, Cdouble, Cint),
        ctx.handle, x, c_left, Int32(m_left), c_right, Int32(m_right), Int32(n), Int32(sigma_start), Int32(sigma_end),
        wire(threshold), wire(pairwise_threshold), Int32(partition_over_ranks ? 1 : 0),
    )
    ret == 0 || error("Aedelsten: failed to compute the half-transformed integrals of sigma $sigma_start:$sigma_end (code $ret)")
    return x
end

"""
    compute_eri_half_transformed_sigma_on_node!(ctx, x, c_left, c_right, sigma_start, sigma_end; threshold=nothing,
        pairwise_threshold=nothing) -> x

`compute_eri_half_transformed_sigma` by the ranks of the context's communicator together, into one `x` in their node's
shared memory, as `compute_eri_half_transformed_on_node!`: `x` an `(m_left, n, m_right, sigma_end - sigma_start)`
reshape of an `AedelstenSharedArray`'s `array` of the same communicator. Collective.
"""
function compute_eri_half_transformed_sigma_on_node!(
    ctx::AedelstenContext, x::Array{Float64,4}, c_left::Matrix{Float64}, c_right::Matrix{Float64}, sigma_start::Integer, sigma_end::Integer;
    threshold::Union{Real,Nothing}=nothing, pairwise_threshold::Union{Real,Nothing}=nothing,
)
    n = get_n_cont(ctx)
    (size(c_left, 1) == n && size(c_right, 1) == n) || error("Aedelsten: c_left and c_right must be n x m")
    0 <= sigma_start < sigma_end <= n || error("Aedelsten: sigma $sigma_start:$sigma_end is not a run of the $n AOs")
    m_left, m_right = size(c_left, 2), size(c_right, 2)
    size(x) == (m_left, n, m_right, sigma_end - sigma_start) || error("Aedelsten: x must be (m_left, n, m_right, n_sigma)")
    wire(v) = v === nothing ? -1.0 : Float64(v)
    ret = ccall(
        _sym(:compute_eri_half_transformed_sigma_on_node_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Cint, Cint, Cdouble, Cdouble),
        ctx.handle, x, c_left, Int32(m_left), c_right, Int32(m_right), Int32(n), Int32(sigma_start), Int32(sigma_end),
        wire(threshold), wire(pairwise_threshold),
    )
    ret == 0 || error("Aedelsten: failed to compute the half-transformed integrals of sigma $sigma_start:$sigma_end on the node (code $ret)")
    return x
end

"""
    half_transform_scratch_bytes(ctx, m_left, m_right, one_basis, n_sigma; node_ranks=1) -> Int

The scratch a half transform takes besides the walk's own, in bytes (`set_scratch!` holds both): `m_left`
and `m_right` orbitals, `one_basis` (the same matrix on both sides) or two, `n_sigma` AOs of sigma (n for
`compute_eri_half_transformed`), `node_ranks` ranks sharing X (the `_on_node!` forms; 1 for the others).
"""
function half_transform_scratch_bytes(ctx::AedelstenContext, m_left::Integer, m_right::Integer, one_basis::Bool, n_sigma::Integer; node_ranks::Integer=1)::Int
    bytes = ccall(
        _sym(:half_transform_scratch_bytes_c), Int64,
        (Ptr{AedelstenContextHandle}, Cint, Cint, Cint, Cint, Cint),
        ctx.handle, Int32(m_left), Int32(m_right), Int32(one_basis ? 1 : 0), Int32(n_sigma), Int32(node_ranks),
    )
    bytes >= 0 || error("Aedelsten: half_transform_scratch_bytes: bad arguments")
    return Int(bytes)
end

# The scratch query's operations by name: their C kind codes and parameter counts (`scratch_bytes`).
const _SCRATCH_OPERATIONS = Dict(
    :eri_data => (1, 0), :restricted_fock => (2, 0), :unrestricted_fock => (3, 0), :two_electron_gradient => (4, 1),
    :general_fock => (5, 0), :general_multi_fock => (6, 1), :active_space_fock => (7, 1),
    :mixed_active_space_fock => (8, 1), :half_transform => (9, 4), :amfi => (10, 0),
)
const _SCRATCH_ITEMS = Dict(1 => :eri_data_entry, 2 => :gradient_pair_data, 3 => :derivative_bound, 4 => :quartet,
                            5 => :amfi_atom_kind)

# The C layout of operations given as symbols or tuples (name, parameters...), `nothing` for a parameter's None: a
# (5, n) Int32 matrix, a column per operation.
function _scratch_operation_rows(operations)::Matrix{Int32}
    rows = zeros(Int32, 5, max(length(operations), 1))
    for (i, operation) in enumerate(operations)
        name, parameters = operation isa Symbol ? (operation, ()) : (operation[1], operation[2:end])
        haskey(_SCRATCH_OPERATIONS, name) || error("Aedelsten: unknown scratch operation $name")
        code, count = _SCRATCH_OPERATIONS[name]
        length(parameters) == count || error("Aedelsten: scratch operation $name takes $count parameters, got $(length(parameters))")
        rows[1, i] = code
        for (k, p) in enumerate(parameters)
            rows[1 + k, i] = p === nothing ? -1 : Int32(p)
        end
    end
    return rows
end

"""
    scratch_bytes(ctx, operations; primitive=false) -> Int

The scratch in bytes that every operation fits in, from the input alone (before `set_shell_pairs!`; size
`set_scratch!` with it). Operations as symbols or tuples: `:eri_data`, `:restricted_fock`, `:unrestricted_fock`,
`(:two_electron_gradient, terms)`, `:general_fock`, `(:general_multi_fock, densities)`, `(:active_space_fock, active)`,
`(:mixed_active_space_fock, active)`, `(:half_transform, left, right, sigma, node_ranks)` with `right` and `sigma`
`nothing` for one basis and all functions, `:amfi` (over the primitive shells whatever `primitive` says).
`primitive`: on the primitive blocks.
"""
function scratch_bytes(ctx::AedelstenContext, operations; primitive::Bool=false)::Int
    rows = _scratch_operation_rows(operations)
    bytes = ccall(
        _sym(:scratch_bytes_c), Int64,
        (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint, Cint),
        ctx.handle, rows, Cint(length(operations)), Cint(primitive ? 1 : 0),
    )
    bytes >= 0 || error("Aedelsten: scratch_bytes failed for $operations")
    return Int(bytes)
end

"""
    scratch_breakdown(ctx, operations; primitive=false) -> Vector{Dict{Symbol,Any}}

`scratch_bytes` per operation, in order, with what sets it: `:bytes` = `:buffers` + `:tables` + `:row` +
`:item_bytes`, `:item` (`:eri_data_entry`, `:gradient_pair_data`, `:derivative_bound`, `:quartet` or
`:amfi_atom_kind`) and `:shells`, the item's `(l, primitives, contracted)`: two for a pair's item, four for a quartet,
none for an atom kind.
"""
function scratch_breakdown(ctx::AedelstenContext, operations; primitive::Bool=false)::Vector{Dict{Symbol,Any}}
    fields = Int(ccall(_sym(:scratch_breakdown_field_count_c), Cint, ()))
    names = [Symbol(unsafe_string(ccall(_sym(:scratch_breakdown_field_name_c), Cstring, (Cint,), Int32(i)))) for i in 0:fields-1]
    rows = _scratch_operation_rows(operations)
    output = zeros(Int64, fields, max(length(operations), 1))
    ret = ccall(
        _sym(:scratch_breakdown_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint, Cint, Ptr{Int64}, Cint),
        ctx.handle, rows, Cint(length(operations)), Cint(primitive ? 1 : 0), output, Cint(fields * length(operations)),
    )
    ret == 0 || error("Aedelsten: scratch_breakdown failed (code $ret)")
    return [begin
        need = Dict{Symbol,Any}(names[k] => Int(output[k, i]) for k in 1:6)
        need[:item] = _SCRATCH_ITEMS[need[:item]]
        need[:shells] = [Tuple(Int.(output[k:k+2, i])) for k in 7:3:fields if output[k, i] >= 0]
        need
    end for i in 1:length(operations)]
end

"""
    compute_two_electron_gradient(ctx, densities, terms; range=nothing, omega=nothing,
        pairwise_threshold=nothing, active=nothing) -> Matrix{Float64}

The two-electron energy gradient `dE_2e/dX_c` for every gradient centre, a freshly allocated
`(3, get_n_gradient_centres(ctx))` matrix: the contracted twin of the direct J/K builds. `densities` is
`(n, n, n_densities)`; each term is a tuple `(kind, left, right, coeff)` with `kind` "coulomb"
(`W = L[a,b] R[c,d]`) or "exchange" (`W = L[a,c] R[b,d]`) and `left`, `right` 1-based indices into the
third dimension of `densities`; `E_2e = 1/2 sum_terms coeff sum_abcd W (ab|cd)`. Restricted HF:
`densities = P`, terms `[("coulomb", 1, 1, 1.0), ("exchange", 1, 1, -0.5)]`; unrestricted `[P, P_a, P_b]`,
`[("coulomb", 1, 1, 1.0), ("exchange", 2, 2, -1.0), ("exchange", 3, 3, -1.0)]`. Only the integral
derivatives (the Pulay term comes with the overlap gradient); one range operator per call; `active` as in
`compute_one_electron_gradient`. Requires `set_eri_data!` first.
"""
function compute_two_electron_gradient(
    ctx::AedelstenContext, densities::AbstractArray{Float64,3}, terms::AbstractVector;
    range::Union{AbstractString,Nothing}=nothing, omega::Union{Real,Nothing}=nothing,
    pairwise_threshold::Union{Real,Nothing}=nothing, active::Union{Nothing,AbstractVector{Bool}}=nothing,
)::Matrix{Float64}
    n::Int = get_n_cont(ctx)
    n_densities, kinds, lefts, rights, coeffs = _two_electron_terms(ctx, densities, terms)
    n_centres::Int = get_n_gradient_centres(ctx)
    range_wire, omega_wire = _resolve_eri_range(range, omega)
    pairwise_threshold_wire::Float64 = pairwise_threshold === nothing ? -1.0 : Float64(pairwise_threshold)
    mask = active === nothing ? nothing : Cint.(active)
    mask === nothing || length(mask) == n_centres || error("Aedelsten: active must have $n_centres flags")
    d = Array{Float64,3}(densities)
    gradient = zeros(Float64, 3, n_centres)
    ret = ccall(
        _sym(:compute_two_electron_gradient_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Cint,
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Cdouble, Cdouble, Ptr{Cint}),
        ctx.handle, gradient, Int32(n_centres), d, Int32(n), Int32(n_densities), Int32(length(terms)),
        kinds, lefts, rights, coeffs, range_wire, omega_wire, pairwise_threshold_wire,
        mask === nothing ? C_NULL : mask,
    )
    ret == 0 || error("Aedelsten: failed to compute the two-electron gradient (code $ret)")
    return gradient
end

# The densities checked and the terms as the C arrays (0-based density indices).
function _two_electron_terms(ctx::AedelstenContext, densities::AbstractArray{Float64,3}, terms::AbstractVector)
    n::Int = get_n_cont(ctx)
    (size(densities, 1) == n && size(densities, 2) == n) ||
        error("Aedelsten: densities must be $n x $n x n_densities, got $(size(densities))")
    n_densities::Int = size(densities, 3)
    kinds = Vector{Cint}(undef, length(terms))
    lefts = Vector{Cint}(undef, length(terms))
    rights = Vector{Cint}(undef, length(terms))
    coeffs = Vector{Cdouble}(undef, length(terms))
    for (t, (kind, left, right, coeff)) in enumerate(terms)
        kinds[t] = lowercase(String(kind)) == "coulomb" ? Cint(0) :
                   lowercase(String(kind)) == "exchange" ? Cint(1) :
                   error("Aedelsten: unknown two-electron term kind '$kind' (coulomb or exchange)")
        (1 <= left <= n_densities && 1 <= right <= n_densities) ||
            error("Aedelsten: term $t's density index is outside 1:$n_densities")
        lefts[t], rights[t], coeffs[t] = Cint(left - 1), Cint(right - 1), Float64(coeff)
    end
    return n_densities, kinds, lefts, rights, coeffs
end

# --- Memory diagnostics, Molcas-export identity data, Cartesian rescaling ---

get_n_primitive_rows(ctx::AedelstenContext)::Int =
    Int(ccall(_sym(:get_n_primitive_rows_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle))

"""The number of fields of `memory_breakdown`."""
memory_breakdown_field_count()::Int = Int(ccall(_sym(:memory_breakdown_field_count_c), Cint, ()))

_memory_breakdown_field_name(index::Integer)::String =
    unsafe_string(ccall(_sym(:memory_breakdown_field_name_c), Cstring, (Cint,), Int32(index)))

"""The memory this context holds now, a Dict of bytes by field name (0 for a field that is empty or invalidated)."""
function memory_breakdown(ctx::AedelstenContext)::Dict{String,Int}
    n = memory_breakdown_field_count()
    bytes = zeros(UInt64, n)
    ret = ccall(
        _sym(:memory_breakdown_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{UInt64}, Cint), ctx.handle, bytes, Cint(n),
    )
    ret == 0 || error("Aedelsten: memory_breakdown failed (code $ret)")
    return Dict(_memory_breakdown_field_name(i - 1) => Int(bytes[i]) for i in 1:n)
end

"""BASIS_FUNCTION_IDS-equivalent: ids[:,1]=centre, ids[:,2]=shell, ids[:,3]=l, ids[:,4]=m, one row
per AO in AO order. See BasisFunctionId's own doc comment on the Rust side for exactly what l/m
mean for Cartesian vs Spherical shells."""
function get_basis_function_ids(ctx::AedelstenContext)::Matrix{Int32}
    n = get_n_cont(ctx)
    ids = zeros(Int32, n, 4)
    ret = ccall(
        _sym(:get_basis_function_ids_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint), ctx.handle, ids, Cint(4n),
    )
    ret == 0 || error("Aedelsten: get_basis_function_ids failed (code $ret)")
    return ids
end

# --- The model basis: free-atom reference orbitals (ANO-RCC), for an atomic frozen core ---

"""The number of model functions: the free-atom reference orbitals of the model basis (ANO-RCC, the model
Fockian's), one per component, on the QM centres that are atoms; a core AIMP's share is left out."""
function get_n_model_functions(ctx::AedelstenContext)::Int
    n_model = ccall(_sym(:get_n_model_functions_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle)
    n_model >= 0 || error("Aedelsten: get_n_model_functions failed")
    return Int(n_model)
end

"""
    get_model_functions(ctx) -> (ids::Matrix{Int32}, energies::Vector{Float64})

The model functions, one row per function in the model basis's AO order: ids[:,1]=centre (0-based, as
`get_centre_label` takes it), ids[:,2]=l, ids[:,3]=m (as `get_basis_function_ids` gives it for a shell of the same
type: s and p Cartesian, p as x, y, z = -1, 0, 1; d and up spherical, -l..l), ids[:,4]=the orbital's ordinal within
its l in the reference library (0-based), counting what a core AIMP covers (a centre whose AIMP projection basis has
k columns at l starts at k); `energies` the reference orbital energies.
"""
function get_model_functions(ctx::AedelstenContext)
    n_model = get_n_model_functions(ctx)
    ids = zeros(Int32, n_model, 4)
    energies = zeros(Float64, n_model)
    ret = ccall(
        _sym(:get_model_functions_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint, Ptr{Cdouble}, Cint), ctx.handle, ids, Cint(4n_model), energies, Cint(n_model),
    )
    ret == 0 || error("Aedelsten: get_model_functions failed (code $ret)")
    return (ids, energies)
end

"""
    compute_model_basis_overlaps(ctx; model_model=true) -> (s_model_working, s_model_model)

The overlaps of the model basis: `s_model_working[i, nu] = <i|nu>`, `(n_model, get_n_cont(ctx))`, model functions by
working basis functions, and `s_model_model` (`(n_model, n_model)`, symmetric), `nothing` unless `model_model`.
"""
function compute_model_basis_overlaps(ctx::AedelstenContext; model_model::Bool=true)
    n, n_model = get_n_cont(ctx), get_n_model_functions(ctx)
    s_mw = zeros(Float64, n_model, n)
    s_mm = model_model ? zeros(Float64, n_model, n_model) : nothing
    ret = ccall(
        _sym(:compute_model_basis_overlaps_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint),
        ctx.handle, s_mw, Cint(n_model * n), s_mm === nothing ? C_NULL : s_mm, Cint(s_mm === nothing ? 0 : n_model * n_model),
    )
    ret == 0 || error("Aedelsten: compute_model_basis_overlaps failed (code $ret)")
    return (s_mw, s_mm)
end

"""The model-Fock start guess `S_mw' diag(E) S_mw`, `(get_n_cont(ctx), get_n_cont(ctx))`, symmetric."""
function compute_model_fockian(ctx::AedelstenContext)::Matrix{Float64}
    n = get_n_cont(ctx)
    fockian = zeros(Float64, n, n)
    ret = ccall(_sym(:compute_model_fockian_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint), ctx.handle, fockian, Cint(n * n))
    ret == 0 || error("Aedelsten: compute_model_fockian failed (code $ret)")
    return fockian
end

"""The overlap between this context's basis and `other`'s, which may differ in basis set, geometry, harmonics and
contraction type: `s[i, j] = <self_i|other_j>`, `(get_n_cont(ctx), get_n_cont(other))`; unscreened."""
function compute_overlap_with(ctx::AedelstenContext, other::AedelstenContext)::Matrix{Float64}
    s = zeros(Float64, get_n_cont(ctx), get_n_cont(other))
    ret = ccall(_sym(:compute_overlap_with_c), Cint,
                (Ptr{AedelstenContextHandle}, Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Int64),
                ctx.handle, other.handle, s, Int64(length(s)))
    ret == 0 || error("Aedelsten: compute_overlap_with failed (code $ret)")
    return s
end

"""PRIMITIVE_IDS-equivalent: ids[:,1]=centre, ids[:,2]=l, ids[:,3]=shell, parallel to
get_primitives (row i of each belongs together)."""
function get_primitive_ids(ctx::AedelstenContext)::Matrix{Int32}
    n = get_n_primitive_rows(ctx)
    ids = zeros(Int32, n, 3)
    ret = ccall(
        _sym(:get_primitive_ids_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint), ctx.handle, ids, Cint(3n),
    )
    ret == 0 || error("Aedelsten: get_primitive_ids failed (code $ret)")
    return ids
end

"""PRIMITIVES-equivalent: primitives[:,1]=exponent, primitives[:,2]=coefficient, parallel to
get_primitive_ids."""
function get_primitives(ctx::AedelstenContext)::Matrix{Float64}
    n = get_n_primitive_rows(ctx)
    primitives = zeros(Float64, n, 2)
    ret = ccall(
        _sym(:get_primitives_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint), ctx.handle, primitives, Cint(2n),
    )
    ret == 0 || error("Aedelsten: get_primitives failed (code $ret)")
    return primitives
end

"""centre_idx is 0-based (get_basis_function_ids/get_primitive_ids's own `centre` column is
1-based -- pass centre - 1). `nothing` if centre_idx has no label (out of range, or an
orthogonality ghost centre) -- not an error, unlike everywhere else in this file."""
function get_centre_label(ctx::AedelstenContext, centre_idx::Integer)::Union{String,Nothing}
    buf = zeros(UInt8, 64)
    ret = ccall(
        _sym(:get_centre_label_c), Cint,
        (Ptr{AedelstenContextHandle}, Cint, Ptr{UInt8}, Cint), ctx.handle, Cint(centre_idx), buf, Cint(length(buf)),
    )
    return ret == 0 ? unsafe_string(pointer(buf)) : nothing
end

"""
    rescale_cartesian_ao_rows!(ctx, convention, matrix) -> Bool

Rescales `matrix` (get_n_cont(), n_cols; AO axis = rows) in place, from this library's own
Cartesian normalization to `convention`'s ("molcas", "dalton", or "dirac", case-insensitive) --
e.g. an AO x MO coefficient matrix being written out for a reader that expects `convention`.
Returns whether anything actually changed (`false` when every factor was exactly 1.0, e.g. no
Cartesian d/f/g/... shells in this basis).
"""
function rescale_cartesian_ao_rows!(ctx::AedelstenContext, convention::AbstractString, matrix::Matrix{Float64})::Bool
    n = get_n_cont(ctx)
    size(matrix, 1) == n || error("Aedelsten: rescale_cartesian_ao_rows!: matrix must be (get_n_cont(), n_cols)")
    convention_int = _query_i32("cartesian_convention_" * lowercase(convention))
    rescaled = Ref{Cint}(0)
    ret = ccall(
        _sym(:rescale_cartesian_ao_rows_c), Cint,
        (Ptr{AedelstenContextHandle}, Cint, Cint, Ptr{Cdouble}, Cint, Ref{Cint}),
        ctx.handle, convention_int, Cint(size(matrix, 2)), matrix, Cint(length(matrix)), rescaled,
    )
    ret == 0 || error("Aedelsten: rescale_cartesian_ao_rows! failed (code $ret)")
    return rescaled[] != 0
end

"""
    unrescale_cartesian_ao_rows!(ctx, convention, matrix) -> Bool

The inverse of `rescale_cartesian_ao_rows!`: `matrix` (get_n_cont(), n_cols) from `convention`'s Cartesian
normalization to this library's, in place. Returns whether anything changed.
"""
function unrescale_cartesian_ao_rows!(ctx::AedelstenContext, convention::AbstractString, matrix::Matrix{Float64})::Bool
    n = get_n_cont(ctx)
    size(matrix, 1) == n || error("Aedelsten: unrescale_cartesian_ao_rows!: matrix must be (get_n_cont(), n_cols)")
    convention_int = _query_i32("cartesian_convention_" * lowercase(convention))
    rescaled = Ref{Cint}(0)
    ret = ccall(
        _sym(:unrescale_cartesian_ao_rows_c), Cint,
        (Ptr{AedelstenContextHandle}, Cint, Cint, Ptr{Cdouble}, Cint, Ref{Cint}),
        ctx.handle, convention_int, Cint(size(matrix, 2)), matrix, Cint(length(matrix)), rescaled,
    )
    ret == 0 || error("Aedelsten: unrescale_cartesian_ao_rows! failed (code $ret)")
    return rescaled[] != 0
end

"""Same as `rescale_cartesian_ao_rows!`, but for `matrix` (n_rows, get_n_cont(); AO axis =
columns)."""
function rescale_cartesian_ao_columns!(ctx::AedelstenContext, convention::AbstractString, matrix::Matrix{Float64})::Bool
    n = get_n_cont(ctx)
    size(matrix, 2) == n || error("Aedelsten: rescale_cartesian_ao_columns!: matrix must be (n_rows, get_n_cont())")
    convention_int = _query_i32("cartesian_convention_" * lowercase(convention))
    rescaled = Ref{Cint}(0)
    ret = ccall(
        _sym(:rescale_cartesian_ao_columns_c), Cint,
        (Ptr{AedelstenContextHandle}, Cint, Cint, Ptr{Cdouble}, Cint, Ref{Cint}),
        ctx.handle, convention_int, Cint(size(matrix, 1)), matrix, Cint(length(matrix)), rescaled,
    )
    ret == 0 || error("Aedelsten: rescale_cartesian_ao_columns! failed (code $ret)")
    return rescaled[] != 0
end

# --- The X2C picture-change cache ---

has_picture_change(ctx::AedelstenContext)::Bool =
    ccall(_sym(:has_picture_change_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle) == 1

"""The cached X2C decoupling transforms `(u_large, u_small)`, each `(n_prim, n_prim)`: `u_large` for plain
operators, `u_small` for their pXp counterparts."""
function get_picture_change_transforms(ctx::AedelstenContext)::Tuple{Matrix{Float64},Matrix{Float64}}
    n = get_n_prim_unique(ctx)
    u_large, u_small = zeros(Float64, n, n), zeros(Float64, n, n)
    ret = ccall(
        _sym(:get_picture_change_transforms_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cint), ctx.handle, u_large, u_small, Cint(n * n),
    )
    ret == 0 || error("Aedelsten: get_picture_change_transforms failed (code $ret)")
    return u_large, u_small
end

"""Frees the X2C decoupling transforms."""
free_picture_change!(ctx::AedelstenContext) = ccall(_sym(:free_picture_change_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)

"""
    set_picture_change_transforms!(ctx, u_large, u_small; c_light=nothing)

Supplies the decoupling transforms, as `get_picture_change_transforms` returns them, instead of deriving them: the next
`construct_x2c_one_electron_hamiltonian` holds them (the X2C Hamiltonian with another geometry's decoupling).
`c_light` as they were made with (`nothing`: the library default).
"""
function set_picture_change_transforms!(ctx::AedelstenContext, u_large::Matrix{Float64}, u_small::Matrix{Float64}; c_light::Union{Real,Nothing}=nothing)
    n = get_n_prim_unique(ctx)
    size(u_large) == (n, n) && size(u_small) == (n, n) || error("Aedelsten: the transforms must be $n x $n")
    c_light_wire::Float64 = c_light === nothing ? -1.0 : Float64(c_light)
    ret = ccall(
        _sym(:set_picture_change_transforms_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble), ctx.handle, u_large, u_small, Cint(n * n), c_light_wire,
    )
    ret == 0 || error("Aedelsten: set_picture_change_transforms failed (code $ret)")
    return nothing
end

"""
    set_x2c_frozen_density!(ctx, density)

Holds the densities of the X2C gradient with the decoupling held, from the symmetric `density` `(n_cont, n_cont)`, for
`compute_x2c_frozen_kinetic_gradient` and `compute_x2c_frozen_potential_gradient`; replaces densities already held. A
new decoupling, `free_picture_change!` or `set_centre_positions!` drops them.
"""
function set_x2c_frozen_density!(ctx::AedelstenContext, density::AbstractMatrix{Float64})
    n::Int = get_n_cont(ctx)
    size(density) == (n, n) || error("Aedelsten: density must be $n x $n, got $(size(density))")
    d = Matrix{Float64}(density)
    ret = ccall(_sym(:set_x2c_frozen_density_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint), ctx.handle, d, Int32(n))
    ret == 0 || error("Aedelsten: set_x2c_frozen_density failed (code $ret)")
    return nothing
end

"""Frees the densities `set_x2c_frozen_density!` holds."""
free_x2c_frozen_density!(ctx::AedelstenContext) = ccall(_sym(:free_x2c_frozen_density_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)

function _x2c_frozen_gradient_arguments(ctx::AedelstenContext, active)
    n_centres::Int = get_n_gradient_centres(ctx)
    mask = active === nothing ? nothing : Cint.(active)
    mask === nothing || length(mask) == n_centres || error("Aedelsten: active must have $n_centres flags")
    return n_centres, mask, zeros(Float64, 3, n_centres)
end

"""
    compute_x2c_frozen_kinetic_gradient(ctx; active=nothing)

The kinetic part of the X2C one-electron energy tr(D h)'s gradient with the decoupling held, `(3, n_centres)`, from the
densities of `set_x2c_frozen_density!`. With `compute_x2c_frozen_potential_gradient` for each potential in the
transformation, the sum is the whole gradient. `active` as in `compute_one_electron_gradient`.
"""
function compute_x2c_frozen_kinetic_gradient(ctx::AedelstenContext; active::Union{Nothing,AbstractVector{Bool}}=nothing)::Matrix{Float64}
    n_centres, mask, gradient = _x2c_frozen_gradient_arguments(ctx, active)
    ret = ccall(
        _sym(:compute_x2c_frozen_kinetic_gradient_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cint}),
        ctx.handle, gradient, Int32(n_centres), mask === nothing ? C_NULL : mask,
    )
    ret == 0 || error("Aedelsten: compute_x2c_frozen_kinetic_gradient failed (code $ret)")
    return gradient
end

"""
    compute_x2c_frozen_potential_gradient(ctx, operator; active=nothing)

The part of the X2C one-electron energy's gradient with the decoupling held of one potential `operator` that went into
V and p.Vp (`"nuclear_attraction"`, `"external_multipole"`, `"embedding_aimp_coulomb"`, ...), `(3, n_centres)`; the
caller sums the parts. `active` as in `compute_one_electron_gradient`.
"""
function compute_x2c_frozen_potential_gradient(
    ctx::AedelstenContext, operator::AbstractString; active::Union{Nothing,AbstractVector{Bool}}=nothing,
)::Matrix{Float64}
    n_centres, mask, gradient = _x2c_frozen_gradient_arguments(ctx, active)
    operator_int::Int32 = _resolve_operator(operator)
    ret = ccall(
        _sym(:compute_x2c_frozen_potential_gradient_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Cint, Ptr{Cint}),
        ctx.handle, gradient, Int32(n_centres), operator_int, mask === nothing ? C_NULL : mask,
    )
    ret == 0 || error("Aedelsten: compute_x2c_frozen_potential_gradient failed for '$operator' (code $ret)")
    return gradient
end

"""
    transform_one_electron_operator(ctx, op_plain, op_pxp)

A one-electron operator in the X2C picture, the scalar part: `u_large' * X * u_large + u_small' * (p.X p) * u_small`
per plane, from `op_plain` (X) and `op_pxp` (p.X p) over the primitive basis, both `(n_prim, n_prim)` or
`(n_prim, n_prim, n_planes)` (the raw planes of `compute_one_electron_picture_change_integrals`); symmetrised. Needs
the X2C decoupling.
"""
function transform_one_electron_operator(ctx::AedelstenContext, op_plain::Array{Float64,N}, op_pxp::Array{Float64,N})::Array{Float64,N} where {N}
    (N in (2, 3) && size(op_plain) == size(op_pxp) && size(op_plain, 1) == size(op_plain, 2)) ||
        error("Aedelsten: op_plain and op_pxp must both be (n, n) or (n, n, n_planes)")
    n, n_planes = size(op_plain, 1), size(op_plain, 3)
    out = zeros(Float64, size(op_plain))
    ret = ccall(
        _sym(:transform_one_electron_operator_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
        ctx.handle, op_plain, op_pxp, out, Cint(n), Cint(n_planes),
    )
    ret == 0 || error("Aedelsten: transform_one_electron_operator failed (code $ret)")
    return out
end

"""
    transform_one_electron_spin_operator(ctx, op_pxp)

The spin planes of a one-electron operator in the X2C picture, `u_small' * op_pxp[:, :, k] * u_small` per plane of
`op_pxp` `(n_prim, n_prim, n_planes)`: the raw `(p X x p)_k` planes of
`compute_one_electron_picture_change_integrals(...; spin_orbit=true)`, other raw spin planes such as
`compute_amfi_integrals`' added first. No symmetrisation (the planes are antisymmetric). Needs the X2C decoupling.
"""
function transform_one_electron_spin_operator(ctx::AedelstenContext, op_pxp::Array{Float64,3})::Array{Float64,3}
    n, n2, n_planes = size(op_pxp)
    n == n2 || error("Aedelsten: op_pxp planes must be square")
    out = zeros(Float64, n, n, n_planes)
    ret = ccall(
        _sym(:transform_one_electron_spin_operator_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint), ctx.handle, op_pxp, out, Cint(n), Cint(n_planes),
    )
    ret == 0 || error("Aedelsten: transform_one_electron_spin_operator failed (code $ret)")
    return out
end

"""
    compute_picture_changed_one_electron_operator(ctx, operator; contracted=true, origin=nothing, order=nothing)

A one-electron operator X in the X2C picture, `X_2c = scalar ⊗ 1 + i Σ_k σ_k ⊗ spin_k`: `(scalar, spin)`, `scalar`
`(n, n, K)` with `u_large' X_c u_large + u_small' (p.X_c p) u_small`, `spin` `(n, n, 3K)` with plane `3(c-1) + k`
from `u_small' (p X_c × p)_k u_small` (antisymmetric), over the contracted basis (`contracted`) or the primitive one.
`K = 1` for a plain operator, the moment's component count with `order`. Needs the X2C decoupling and
`set_primitive_shell_pairs!`.
"""
function compute_picture_changed_one_electron_operator(
    ctx::AedelstenContext,
    operator::AbstractString;
    contracted::Bool=true,
    origin::Union{NTuple{3,<:Real},Nothing}=nothing,
    order::Union{Integer,Nothing}=nothing,
)::Tuple{Array{Float64,3},Array{Float64,3}}
    n::Int = contracted ? get_n_cont(ctx) : get_n_prim_unique(ctx)
    operator_int::Int32 = _resolve_operator(operator)
    n_components::Int32 = order === nothing ? Int32(1) :
        Int32(ccall(_sym(:compute_moment_component_count_c), Cint, (Cint,), Int32(order)))
    ox, oy, oz = origin === nothing ? (0.0, 0.0, 0.0) : origin
    scalar = zeros(Float64, n, n, n_components)
    spin = zeros(Float64, n, n, 3 * n_components)
    ret = ccall(
        _sym(:compute_picture_changed_one_electron_operator_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cint, Cdouble, Cdouble, Cdouble, Cint, Cint),
        ctx.handle, scalar, spin, Int32(n), operator_int, n_components,
        Float64(ox), Float64(oy), Float64(oz), order === nothing ? Int32(0) : Int32(order), Int32(contracted),
    )
    ret == 0 || error("Aedelsten: compute_picture_changed_one_electron_operator failed for '$operator' (code $ret)")
    return scalar, spin
end

"""
    compute_amfi_integrals(ctx)

The atomic mean-field two-electron spin-orbit integrals over the primitive basis, `(n_prim, n_prim, 6)`:
spin-same-orbit x, y, z, then spin-other-orbit exchange x, y, z, raw (no prefactor, antisymmetric; Breit-Pauli is
`C' (SSO + SOO) C / 4c^2`). Needs `set_primitive_shell_pairs!` and the scratch.
"""
function compute_amfi_integrals(ctx::AedelstenContext)::Array{Float64,3}
    n = get_n_prim_unique(ctx)
    out = zeros(Float64, n, n, 6)
    ret = ccall(_sym(:compute_amfi_integrals_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint), ctx.handle, out, Cint(n))
    ret == 0 || error("Aedelsten: compute_amfi_integrals failed (code $ret)")
    return out
end

# --- Point-group symmetry ---
# Irrep indices are 0-based, as the library's (and the reports') own.

"""
    set_symmetry!(ctx; point_group=nothing, principal_axis=nothing, geometry_tolerance=nothing,
                  strict_tolerance=nothing, average_warning=nothing)

Finds the point group of the QM centres (`point_group` `nothing`) or uses the named one (e.g. `"C2v"`: the full group
or a subgroup; `principal_axis`, a 3-vector, picks a subgroup's orientation). Tolerances `nothing`: the defaults
(1e-8 Bohr, 1e-10 relative, no warning). With symmetry set the library cleans the integrals it computes (strictly for
the QM region, averaged for the environment; see `get_symmetrisation_report`).
"""
function set_symmetry!(
    ctx::AedelstenContext; point_group::Union{AbstractString,Nothing}=nothing, principal_axis=nothing,
    geometry_tolerance::Union{Real,Nothing}=nothing, strict_tolerance::Union{Real,Nothing}=nothing,
    average_warning::Union{Real,Nothing}=nothing,
)
    wire(x) = x === nothing ? -1.0 : Float64(x)
    axis = principal_axis === nothing ? C_NULL : Vector{Float64}(principal_axis)
    axis === C_NULL || length(axis) == 3 || error("Aedelsten: set_symmetry!: principal_axis must have 3 components")
    ret = ccall(
        _sym(:set_symmetry_c), Cint,
        (Ptr{AedelstenContextHandle}, Cstring, Ptr{Cdouble}, Cdouble, Cdouble, Cdouble),
        ctx.handle, point_group === nothing ? C_NULL : point_group, axis,
        wire(geometry_tolerance), wire(strict_tolerance), wire(average_warning),
    )
    ret == 0 || error("Aedelsten: set_symmetry! failed (code $ret; the reason is on stderr)")
    return nothing
end

"""Frees the point group and its AO representation."""
free_symmetry!(ctx::AedelstenContext) = ccall(_sym(:free_symmetry_c), Cvoid, (Ptr{AedelstenContextHandle},), ctx.handle)

"""The tolerances symmetry was set with: `geometry_tolerance`, `strict_tolerance`, `average_warning` (`nothing`: no
warning)."""
function get_symmetry_options(ctx::AedelstenContext)
    options = zeros(Float64, 3)
    ret = ccall(_sym(:get_symmetry_options_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}), ctx.handle, options)
    ret == 0 || error("Aedelsten: get_symmetry_options failed: no symmetry set")
    return (geometry_tolerance=options[1], strict_tolerance=options[2], average_warning=options[3] == 0.0 ? nothing : options[3])
end

_symmetrisation_report(r::Vector{Float64}) = (irrep=Int(r[1]), max_abs_removed=r[2], removed_norm=r[3], norm=r[4])

"""The report of the last cleaning of a library output (e.g. `"oneint_overlap"`, `"oneint_external_multipole"`,
`"x2c_one_electron_hamiltonian"`) in the primitive or the contracted basis; `nothing` if it was not cleaned."""
function get_symmetrisation_report(ctx::AedelstenContext, name::AbstractString; primitive::Bool=false)
    found = Ref{Cint}(0)
    report = zeros(Float64, 4)
    ret = ccall(
        _sym(:get_symmetrisation_report_c), Cint,
        (Ptr{AedelstenContextHandle}, Cstring, Cint, Ref{Cint}, Ptr{Cdouble}), ctx.handle, name, Cint(primitive), found, report,
    )
    ret == 0 || error("Aedelsten: get_symmetrisation_report failed (code $ret)")
    return found[] == 1 ? _symmetrisation_report(report) : nothing
end

"""The point group in use, e.g. "C2v"."""
function get_point_group_name(ctx::AedelstenContext)::String
    buf = zeros(UInt8, 16)
    ret = ccall(_sym(:get_point_group_name_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{UInt8}, Cint), ctx.handle, buf, Cint(length(buf)))
    ret == 0 || error("Aedelsten: get_point_group_name failed: no symmetry set")
    return unsafe_string(pointer(buf))
end

"""The irreps, `(name, dimension)`, in character-table order (what every 0-based irrep index refers to)."""
function get_irreps(ctx::AedelstenContext)::Vector{Tuple{String,Int}}
    n = ccall(_sym(:get_n_irreps_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle)
    n >= 0 || error("Aedelsten: get_irreps failed: no symmetry set")
    map(0:n-1) do k
        buf = zeros(UInt8, 16)
        ccall(_sym(:get_irrep_name_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Ptr{UInt8}, Cint), ctx.handle, Cint(k), buf, Cint(length(buf)))
        dimension = ccall(_sym(:get_irrep_dimension_c), Cint, (Ptr{AedelstenContextHandle}, Cint), ctx.handle, Cint(k))
        (unsafe_string(pointer(buf)), Int(dimension))
    end
end

"""The index (0-based) of the irrep called `name` (case-insensitive), `nothing` if the group has none."""
function get_irrep_index(ctx::AedelstenContext, name::AbstractString)::Union{Int,Nothing}
    k = ccall(_sym(:get_irrep_index_c), Cint, (Ptr{AedelstenContextHandle}, Cstring), ctx.handle, name)
    return k < 0 ? nothing : Int(k)
end

"""The index (0-based) of the totally symmetric irrep."""
get_totally_symmetric_irrep(ctx::AedelstenContext)::Int =
    Int(ccall(_sym(:get_totally_symmetric_irrep_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle))

"""The irrep of x^lx y^ly z^lz about the centre of mass; `nothing` if it is not a single one-dimensional irrep."""
function get_cartesian_irrep(ctx::AedelstenContext, lx::Integer, ly::Integer, lz::Integer)::Union{Int,Nothing}
    k = ccall(_sym(:get_cartesian_irrep_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Cint, Cint), ctx.handle, Cint(lx), Cint(ly), Cint(lz))
    return k < 0 ? nothing : Int(k)
end

"""`(3, 3)`: column k is the group's standard axis k in the input's coordinates (what the irrep names refer to)."""
function get_symmetry_standard_axes(ctx::AedelstenContext)::Matrix{Float64}
    axes = zeros(Float64, 3, 3)
    ret = ccall(_sym(:get_symmetry_standard_axes_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Cdouble}), ctx.handle, axes)
    ret == 0 || error("Aedelsten: get_symmetry_standard_axes failed: no symmetry set")
    return axes
end

"""The group's order h: operations are numbered 0..h-1 in a fixed order."""
function get_symmetry_operation_count(ctx::AedelstenContext)::Int
    h = ccall(_sym(:get_symmetry_operation_count_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle)
    h >= 0 || error("Aedelsten: get_symmetry_operation_count failed: no symmetry set")
    return Int(h)
end

"""`(3, 3)`: operation `op` (0-based) in the input's axes; a displacement or gradient `v` of a centre goes to `m * v`
at its image (`get_symmetry_centre_images`). Orthogonal."""
function get_symmetry_operation_matrix(ctx::AedelstenContext, op::Integer)::Matrix{Float64}
    m = zeros(Float64, 3, 3)
    ret = ccall(_sym(:get_symmetry_operation_matrix_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Ptr{Cdouble}), ctx.handle, Cint(op), m)
    ret == 0 || error("Aedelsten: get_symmetry_operation_matrix failed: no symmetry set, or no operation $op")
    return m
end

"""The QM centres the group acts on (ghosts included; the gradient centres first)."""
function get_n_symmetry_centres(ctx::AedelstenContext)::Int
    n = ccall(_sym(:get_n_symmetry_centres_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle)
    n >= 0 || error("Aedelsten: get_n_symmetry_centres failed: no symmetry set")
    return Int(n)
end

"""`images[a + 1]`: the centre (0-based, as every index the library returns) operation `op` takes centre `a` to."""
function get_symmetry_centre_images(ctx::AedelstenContext, op::Integer)::Vector{Int}
    n = get_n_symmetry_centres(ctx)
    images = zeros(Int32, n)
    ret = ccall(_sym(:get_symmetry_centre_images_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Ptr{Int32}, Cint), ctx.handle, Cint(op), images, Cint(n))
    ret == 0 || error("Aedelsten: get_symmetry_centre_images failed: no operation $op")
    return Int.(images)
end

"""How many AO functions span each irrep (irrep order)."""
function get_ao_irrep_counts(ctx::AedelstenContext)::Vector{Int}
    n_irreps = ccall(_sym(:get_n_irreps_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle)
    n_irreps >= 0 || error("Aedelsten: get_ao_irrep_counts failed: no symmetry set")
    counts = zeros(Int32, n_irreps)
    ret = ccall(_sym(:get_ao_irrep_counts_c), Cint, (Ptr{AedelstenContextHandle}, Ptr{Int32}, Cint), ctx.handle, counts, n_irreps)
    ret == 0 || error("Aedelsten: get_ao_irrep_counts failed")
    return Int.(counts)
end

"""Operation `op` (0-based) applied to MO coefficients `(n_cont, n_mo)`: the coefficients of each R phi."""
function apply_symmetry_operation(ctx::AedelstenContext, op::Integer, coefficients::AbstractMatrix{<:Real})::Matrix{Float64}
    size(coefficients, 1) == get_n_cont(ctx) || error("Aedelsten: apply_symmetry_operation: coefficients must be (get_n_cont(), n_mo)")
    c = Matrix{Float64}(coefficients)
    out = similar(c)
    ret = ccall(_sym(:apply_symmetry_operation_c), Cint,
                (Ptr{AedelstenContextHandle}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
                ctx.handle, Cint(op), c, out, Cint(length(c)), Cint(size(c, 2)))
    ret == 0 || error("Aedelsten: apply_symmetry_operation failed: no symmetry set, or no operation $op")
    return out
end

"""The character of irrep `irrep` at operation `op` (both 0-based)."""
function get_irrep_character(ctx::AedelstenContext, irrep::Integer, op::Integer)::Float64
    character = Ref{Cdouble}(0.0)
    ret = ccall(_sym(:get_irrep_character_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Cint, Ref{Cdouble}),
                ctx.handle, Cint(irrep), Cint(op), character)
    ret == 0 || error("Aedelsten: get_irrep_character failed: no symmetry set, or no irrep $irrep or operation $op")
    return character[]
end

"""Gamma(R) of irrep `irrep` at operation `op` (both 0-based), `(d, d)`, `R f_k = sum_j Gamma[j, k] f_j`, for the
partner components `symmetrise_orbitals!` aligns; `nothing` when the irrep has none (complex pairs; T, Th)."""
function get_irrep_matrix(ctx::AedelstenContext, irrep::Integer, op::Integer)::Union{Nothing, Matrix{Float64}}
    d = ccall(_sym(:get_irrep_dimension_c), Cint, (Ptr{AedelstenContextHandle}, Cint), ctx.handle, Cint(irrep))
    d >= 0 || error("Aedelsten: get_irrep_matrix failed: no symmetry set, or no irrep $irrep")
    matrix = zeros(Float64, d, d)
    aligned = Ref{Cint}(0)
    ret = ccall(_sym(:get_irrep_matrix_c), Cint, (Ptr{AedelstenContextHandle}, Cint, Cint, Ptr{Cdouble}, Cint, Ref{Cint}),
                ctx.handle, Cint(irrep), Cint(op), matrix, Cint(d * d), aligned)
    ret == 0 || error("Aedelsten: get_irrep_matrix failed: no operation $op")
    return aligned[] == 1 ? matrix : nothing
end

"""
    symmetrise_matrix!(ctx, matrix; average=false) -> report

Cleans the totally symmetric `(n_cont, n_cont)` `matrix` in place (a Fock matrix, a density) and returns the report.
`average` false: refused (an error, `matrix` untouched) above the strict tolerance; true: averaged over the group,
whatever it removes.
"""
function symmetrise_matrix!(ctx::AedelstenContext, matrix::Matrix{Float64}; average::Bool=false)
    size(matrix, 1) == size(matrix, 2) || error("Aedelsten: symmetrise_matrix!: matrix must be square")
    report = zeros(Float64, 4)
    ret = ccall(
        _sym(:symmetrise_matrix_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}), ctx.handle, matrix, Cint(length(matrix)), Cint(average), report,
    )
    ret == 0 || error("Aedelsten: symmetrise_matrix! failed (code $ret; the reason is on stderr)")
    return _symmetrisation_report(report)
end

"""What cleaning `matrix` as `irrep` (0-based; `nothing`: whichever irrep carries most of it) would remove; nothing
changed."""
function analyse_matrix(ctx::AedelstenContext, matrix::Matrix{Float64}; irrep::Union{Integer,Nothing}=nothing)
    report = zeros(Float64, 4)
    ret = ccall(
        _sym(:analyse_matrix_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}),
        ctx.handle, matrix, Cint(length(matrix)), Cint(irrep === nothing ? -1 : irrep), report,
    )
    ret == 0 || error("Aedelsten: analyse_matrix failed (code $ret)")
    return _symmetrisation_report(report)
end

"""The norm of each irrep's part of the `(n_cont, n_cont)` `matrix`, one per irrep."""
function irrep_norms(ctx::AedelstenContext, matrix::Matrix{Float64})::Vector{Float64}
    n_irreps = ccall(_sym(:get_n_irreps_c), Cint, (Ptr{AedelstenContextHandle},), ctx.handle)
    norms = zeros(Float64, max(n_irreps, 0))
    ret = ccall(
        _sym(:get_irrep_norms_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), ctx.handle, matrix, Cint(length(matrix)), norms, n_irreps,
    )
    ret == 0 || error("Aedelsten: irrep_norms failed (code $ret)")
    return norms
end

"""
    symmetrise_orbitals!(ctx, orbitals) -> (irreps, components, dropped)

Symmetrizes MO coefficients in place: `orbitals` `(n_cont, n_mo)`, one MO per column (any `n_mo`). Returns each MO's
irrep (0-based) and partner component, and the largest coefficient norm projected away. Not re-orthonormalized.
"""
function symmetrise_orbitals!(ctx::AedelstenContext, orbitals::Matrix{Float64})
    size(orbitals, 1) == get_n_cont(ctx) || error("Aedelsten: symmetrise_orbitals!: orbitals must be (get_n_cont(), n_mo)")
    n_mo = size(orbitals, 2)
    irreps, components = zeros(Cint, n_mo), zeros(Cint, n_mo)
    dropped = Ref{Cdouble}(0.0)
    ret = ccall(
        _sym(:symmetrise_orbitals_c), Cint,
        (Ptr{AedelstenContextHandle}, Ptr{Cdouble}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ref{Cdouble}),
        ctx.handle, orbitals, Cint(length(orbitals)), Cint(n_mo), irreps, components, dropped,
    )
    ret == 0 || error("Aedelsten: symmetrise_orbitals! failed (code $ret)")
    return Int.(irreps), Int.(components), dropped[]
end

# --- MPI ---
# For a program without MPI of its own. A program that has MPI (MPI.jl, built against the same MPI library)
# initializes it itself and passes its communicators by their Fortran handle (`MPI.Comm_c2f`-style, an integer);
# `nothing` means MPI_COMM_WORLD.

"""
    aedelsten_mpi_init(required_level="funneled") -> Int

Initializes MPI at thread level `required_level` ("single", "funneled", "serialized", "multiple") and returns the
level provided (0-3). Refused when MPI is initialized already by someone else, or provides less.
"""
function aedelsten_mpi_init(required_level::AbstractString="funneled")::Int
    provided = ccall(_sym(:aedelsten_mpi_init_c), Cint, (Cint,), _query_i32("mpi_thread_" * lowercase(required_level)))
    provided < 0 && error("Aedelsten: MPI initialization at thread level '$required_level' failed")
    return Int(provided)
end

"""
    aedelsten_mpi_finalize()

Finalizes MPI if `aedelsten_mpi_init` initialized it; MPI initialized by the host is left alone.
"""
function aedelsten_mpi_finalize()::Nothing
    ccall(_sym(:aedelsten_mpi_finalize_c), Cint, ())
    return nothing
end

"""
    aedelsten_mpi_initialized() -> Bool

Whether MPI is initialized and not finalized (always `false` without the library's `mpi` build option).
"""
aedelsten_mpi_initialized()::Bool = ccall(_sym(:aedelsten_mpi_initialized_c), Cint, ()) == 1

"""
    aedelsten_mpi_thread_level() -> Int

The thread level MPI runs at (0-3), or -1 while it is not up.
"""
aedelsten_mpi_thread_level()::Int = Int(ccall(_sym(:aedelsten_mpi_thread_level_c), Cint, ()))

"""
    AedelstenSharedArray(length; comm=nothing)

`length` doubles in the shared memory of the ranks of communicator `comm` (its Fortran handle; `nothing`: this process
alone), which must share one node: `array` is the same memory on every rank. Made collectively; `sync_shared!` makes
every rank's writes visible to all; freed by `free_shared!`, collectively (not by a finalizer).
"""
mutable struct AedelstenSharedArray
    handle::Ptr{Cvoid}
    array::Vector{Float64}
end

function AedelstenSharedArray(length::Integer; comm::Union{Integer,Nothing}=nothing)
    data = Ref{Ptr{Cdouble}}(C_NULL)
    handle = ccall(_sym(:aedelsten_mpi_shared_array_c), Ptr{Cvoid}, (Cint, Int64, Ref{Ptr{Cdouble}}), _communicator_handle(comm), Int64(length), data)
    handle == C_NULL && error("Aedelsten: failed to make a shared array of $length doubles")
    array = length > 0 ? unsafe_wrap(Array, data[], length; own=false) : Float64[]
    return AedelstenSharedArray(handle, array)
end

"""Makes every rank's writes to `shared` visible to all, every rank waiting for the others. Collective."""
function sync_shared!(shared::AedelstenSharedArray)
    ccall(_sym(:aedelsten_mpi_shared_array_sync_c), Cint, (Ptr{Cvoid},), shared.handle) == 0 || error("Aedelsten: failed to synchronize a shared array")
    return shared
end

"""Frees `shared`'s memory. Collective."""
function free_shared!(shared::AedelstenSharedArray)
    if shared.handle != C_NULL
        shared.array = Float64[]
        ccall(_sym(:aedelsten_mpi_shared_array_free_c), Cvoid, (Ptr{Cvoid},), shared.handle)
        shared.handle = C_NULL
    end
    return nothing
end

# The handle a context takes for `comm`: this process alone for `nothing`.
_communicator_handle(comm::Union{Integer,Nothing})::Cint = comm === nothing ? _query_i32("mpi_comm_single") : Cint(comm)

function _communicator_query(name::Symbol, comm::Union{Integer,Nothing}, what::AbstractString)::Int
    handle = comm === nothing ? _query_i32("mpi_comm_world") : Cint(comm)
    out = Ref{Cint}(0)
    ret = ccall(_sym(name), Cint, (Cint, Ref{Cint}), handle, out)
    ret == 0 || error("Aedelsten: failed to query the $what of communicator $handle")
    return Int(out[])
end

"""
    aedelsten_mpi_rank(comm=nothing) -> Int

This rank's index in communicator `comm` (its Fortran handle; `nothing`: the world).
"""
aedelsten_mpi_rank(comm::Union{Integer,Nothing}=nothing)::Int = _communicator_query(:aedelsten_mpi_rank_c, comm, "rank")

"""
    aedelsten_mpi_size(comm=nothing) -> Int

The number of ranks in communicator `comm` (as in `aedelsten_mpi_rank`).
"""
aedelsten_mpi_size(comm::Union{Integer,Nothing}=nothing)::Int = _communicator_query(:aedelsten_mpi_size_c, comm, "size")

# --- BLAS threads ---
# The process's count, set from the main thread only, outside any threaded region.

"""
    aedelsten_blas_set_threads(k) -> Int

Sets the BLAS's thread count to `k` and returns the previous count. Refused (an error, count unchanged) when the BLAS
cannot run `k` threads.
"""
function aedelsten_blas_set_threads(k::Integer)::Int
    previous = ccall(_sym(:aedelsten_blas_set_threads_c), Cint, (Cint,), k)
    previous < 0 && error("Aedelsten: the BLAS cannot run $k threads ($(aedelsten_blas_info()))")
    return Int(previous)
end

"""
    aedelsten_get_version() -> String

The library's version, "<major>.<minor>.<patch>" (a release is tagged v<version>).
"""
function aedelsten_get_version()::String
    buf = zeros(UInt8, 64)
    ret = ccall(_sym(:aedelsten_get_version_c), Cint, (Ptr{UInt8}, Cint), buf, Cint(length(buf)))
    ret == 0 || error("Aedelsten: failed to read the library's version")
    return unsafe_string(pointer(buf))
end

"""
    aedelsten_get_source_commit() -> Union{String,Nothing}

The git commit the library was built from (12 hex digits), `nothing` when its source tree was no git checkout.
"""
function aedelsten_get_source_commit()::Union{String,Nothing}
    buf = zeros(UInt8, 64)
    ret = ccall(_sym(:aedelsten_get_source_commit_c), Cint, (Ptr{UInt8}, Cint), buf, Cint(length(buf)))
    ret == 0 || error("Aedelsten: failed to read the library's source commit")
    commit = unsafe_string(pointer(buf))
    return isempty(commit) ? nothing : commit
end

"""
    aedelsten_blas_threads() -> Int

The BLAS's thread count now.
"""
aedelsten_blas_threads()::Int = Int(ccall(_sym(:aedelsten_blas_threads_c), Cint, ()))

"""
    aedelsten_blas_info() -> String

The BLAS, its threading and its thread count, as one line (e.g. "OpenBLAS (OpenMP build), 1 thread").
"""
function aedelsten_blas_info()::String
    buf = zeros(UInt8, 256)
    ret = ccall(_sym(:aedelsten_blas_info_c), Cint, (Ptr{UInt8}, Cint), buf, Cint(length(buf)))
    ret == 0 || error("Aedelsten: failed to describe the BLAS")
    return unsafe_string(pointer(buf))
end

"""
    aedelsten_blas_available_cores() -> (count, bound, description)

The cores available to this process: their count, whether a launcher bound the process to them (else they are the
node's, shared by its ranks), and a line for a run's header (e.g. "4 performance cores, node wide").
"""
function aedelsten_blas_available_cores()::NamedTuple{(:count, :bound, :description),Tuple{Int,Bool,String}}
    count, bound = Ref{Cint}(0), Ref{Cint}(0)
    ret = ccall(_sym(:aedelsten_blas_available_cores_c), Cint, (Ref{Cint}, Ref{Cint}), count, bound)
    ret == 0 || error("Aedelsten: failed to count the available cores")
    buf = zeros(UInt8, 256)
    ret = ccall(_sym(:aedelsten_blas_available_cores_info_c), Cint, (Ptr{UInt8}, Cint), buf, Cint(length(buf)))
    ret == 0 || error("Aedelsten: failed to describe the available cores")
    return (count=Int(count[]), bound=bound[] != 0, description=unsafe_string(pointer(buf)))
end

# --- Timings and the reference list ---
# Read anew at each call: the coarse timings are always recorded, the fine ones only in a `profile` build; both are
# this thread's. The references are the process's: everything cited so far, in the order first cited.

"""
    aedelsten_mpi_collective_seconds() -> Float64

Seconds this process has spent in MPI collectives since it started (0 without MPI or on one rank).
"""
aedelsten_mpi_collective_seconds()::Float64 = ccall(_sym(:aedelsten_mpi_collective_seconds_c), Cdouble, ())

"""
    aedelsten_profile_enabled() -> Bool

Whether the library was built with the `profile` feature (`aedelsten_profile_report` can return anything).
"""
aedelsten_profile_enabled()::Bool = ccall(_sym(:aedelsten_profile_enabled_c), Cint, ()) != 0

function _timing_spans(fine::Integer)
    out = NamedTuple{(:label, :seconds, :calls),Tuple{String,Float64,Int}}[]
    for i in 0:(Int(ccall(_sym(:aedelsten_timing_count_c), Cint, (Cint,), fine)) - 1)
        label, seconds, calls = zeros(UInt8, 256), Ref{Cdouble}(0.0), Ref{Int64}(0)
        ret = ccall(_sym(:aedelsten_timing_span_c), Cint, (Cint, Cint, Ptr{UInt8}, Cint, Ref{Cdouble}, Ref{Int64}),
                    fine, i, label, Cint(length(label)), seconds, calls)
        ret == 0 || error("Aedelsten: failed to read timing entry $i")
        push!(out, (label=unsafe_string(pointer(label)), seconds=seconds[], calls=Int(calls[])))
    end
    return out
end

"""
    aedelsten_timing_report() -> Vector{(label, seconds, calls)}

The coarse timings so far, one entry per label, in the order the labels were first seen.
"""
aedelsten_timing_report() = _timing_spans(0)

"""
    aedelsten_profile_report() -> Vector{(label, seconds, calls)}

The fine timings so far, as `aedelsten_timing_report`; empty without the `profile` feature.
"""
aedelsten_profile_report() = _timing_spans(1)

"""
    aedelsten_span_tree() -> Vector{(label, depth, fine, seconds, calls)}

The spans as a tree, depth first (each node followed by its children): depth from 0, `fine` for a fine span, seconds
with the children included.
"""
function aedelsten_span_tree()
    out = NamedTuple{(:label, :depth, :fine, :seconds, :calls),Tuple{String,Int,Bool,Float64,Int}}[]
    for i in 0:(Int(ccall(_sym(:aedelsten_span_tree_count_c), Cint, ())) - 1)
        label, depth, fine, seconds, calls = zeros(UInt8, 256), Ref{Cint}(0), Ref{Cint}(0), Ref{Cdouble}(0.0), Ref{Int64}(0)
        ret = ccall(_sym(:aedelsten_span_tree_node_c), Cint,
                    (Cint, Ptr{UInt8}, Cint, Ref{Cint}, Ref{Cint}, Ref{Cdouble}, Ref{Int64}),
                    i, label, Cint(length(label)), depth, fine, seconds, calls)
        ret == 0 || error("Aedelsten: failed to read span tree node $i")
        push!(out, (label=unsafe_string(pointer(label)), depth=Int(depth[]), fine=fine[] != 0, seconds=seconds[], calls=Int(calls[])))
    end
    return out
end

function _citation_part(index::Integer, topics::Integer)::String
    n = ccall(_sym(:aedelsten_citation_length_c), Int64, (Cint, Cint), index, topics)
    n >= 0 || error("Aedelsten: no reference $index")
    buf = zeros(UInt8, n + 1)
    ret = ccall(_sym(:aedelsten_citation_c), Cint, (Cint, Cint, Ptr{UInt8}, Cint), index, topics, buf, Cint(length(buf)))
    ret == 0 || error("Aedelsten: failed to read reference $index")
    return String(buf[1:n])
end

"""
    aedelsten_citations() -> Vector{(reference, topics)}

The references cited so far: the formatted reference (authors, title, journal, volume, pages, year, DOI) and the
topics it was cited for, joined by ", ".
"""
aedelsten_citations() = [(reference=_citation_part(i, 0), topics=_citation_part(i, 1))
                         for i in 0:(Int(ccall(_sym(:aedelsten_citation_count_c), Cint, ())) - 1)]

"""
    aedelsten_citations_bibtex() -> String

The references cited so far as a BibTeX file (one entry each, the topics as keywords).
"""
function aedelsten_citations_bibtex()::String
    n = ccall(_sym(:aedelsten_citations_bibtex_length_c), Int64, ())
    buf = zeros(UInt8, n + 1)
    ret = ccall(_sym(:aedelsten_citations_bibtex_c), Cint, (Ptr{UInt8}, Cint), buf, Cint(length(buf)))
    ret == 0 || error("Aedelsten: failed to write the references as BibTeX")
    return String(buf[1:n])
end

"""
    aedelsten_clear_citations()

Forgets every reference cited so far (a host running several independent jobs in one process).
"""
aedelsten_clear_citations()::Nothing = ccall(_sym(:aedelsten_clear_citations_c), Cvoid, ())

end # module Aedelsten
