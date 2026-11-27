# SPDX-FileCopyrightText: 2024 Ernst D. Larsson
# SPDX-License-Identifier: MIT

# The Python interface to the library, over its C ABI: the context's methods under the same names as in the Fortran
# and Julia interfaces, operators by wire key ("overlap" or "oneint_overlap"), the dense J/K builders as free
# functions. Every computation returns a new numpy array.
#
# Every array is C-contiguous with its axes in the reverse of the library's column-major shape (its
# (n_cont, n_cont, n_planes) moment tensor is (n_planes, n_cont, n_cont) here): the same bytes, read row-major, not a
# copy. Every matrix therefore comes back transposed, which a symmetric one does not show (the one-electron integrals,
# J, K, the X2C Hamiltonian, the S1 ERI tensor). It matters for a density that is not symmetric (pass the transpose)
# and for the S4 and S8 pair indices (`ao_index_order`), both said where they occur.
#
# Arrays are checked, never copied: a C-contiguous float64 array of the right shape, or an error.

import ctypes
import sys
import os
from pathlib import Path
import numpy as np

# --- Opaque handle ---

class _AedelstenContextHandle(ctypes.Structure):
    pass

AedelstenContextHandle = ctypes.POINTER(_AedelstenContextHandle)

# --- Polarization options (the C struct AedelstenPolarizationOptions, field for field) ---

class PolarizationOptions(ctypes.Structure):
    """Options of AedelstenContext.set_polarization -- same fields, order and defaults as the Fortran type and
    the Julia struct. A negative number means "the library's default". Construct with keywords, e.g.
    PolarizationOptions(threshold=1e-10, response_batch=4)."""
    _fields_ = [
        ("damping",        ctypes.c_int),     # 0: none, 1: Thole
        ("thole_a",        ctypes.c_double),  # Thole parameter a; negative: 2.1304
        ("threshold",      ctypes.c_double),  # dipole-solver convergence; negative: 1e-8
        ("max_iterations", ctypes.c_int),
        ("static_field",   ctypes.c_int),     # 0: keep the static field in memory, 1: recompute it
        ("chunk_sites",    ctypes.c_int),     # sites per integral chunk; not positive: use the budget
        ("chunk_budget",    ctypes.c_int),     # integral-chunk memory budget (in chunk_budget_unit); negative: default
        ("chunk_budget_unit",    ctypes.c_int),     # byte unit: 0 GiB, 1 MiB, 2 KiB
        ("response_batch", ctypes.c_int),     # densities per compute_induction_response batch; 0: none
        ("state_slots",    ctypes.c_int),     # states held by compute_induction_states; 0: none
        ("resident_budget",  ctypes.c_int),     # budget for keeping the field integrals between calls; 0: never
        ("resident_budget_unit",  ctypes.c_int),     # byte unit of resident_budget: 0 GiB, 1 MiB, 2 KiB
    ]
    _defaults = dict(damping=0, thole_a=-1.0, threshold=-1.0, max_iterations=-1, static_field=0,
                     chunk_sites=-1, chunk_budget=-1, chunk_budget_unit=1, response_batch=0, state_slots=0,
                     resident_budget=0, resident_budget_unit=1)

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(self._defaults)
        if unknown:
            raise TypeError(f"unknown polarization option(s): {sorted(unknown)}")
        super().__init__(**{**self._defaults, **kwargs})

# --- Library loading ---

_aedelsten_lib = None

_SIGNATURES = [
    ("set_aedelsten_context_c",  [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],       AedelstenContextHandle),
    ("free_aedelsten_context_c", [AedelstenContextHandle],                            None),
    ("set_shell_pairs_c",        [AedelstenContextHandle, ctypes.c_double],           ctypes.c_int),
    ("free_shell_pairs_c",       [AedelstenContextHandle],                            None),
    ("set_primitive_shell_pairs_c", [AedelstenContextHandle, ctypes.c_double], ctypes.c_int),
    ("free_primitive_shell_pairs_c", [AedelstenContextHandle],                        None),
    ("free_input_c",             [AedelstenContextHandle],                            None),
    ("free_primitive_shells_c",  [AedelstenContextHandle],                            None),
    ("free_contraction_matrix_c", [AedelstenContextHandle],                           None),
    ("has_primitive_c",          [AedelstenContextHandle],                            ctypes.c_int),
    ("set_scratch_c",          [AedelstenContextHandle, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("set_communicator_c",       [AedelstenContextHandle, ctypes.c_int],              ctypes.c_int),
    ("resize_scratch_c",       [AedelstenContextHandle, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("get_scratch_peak_bytes_c", [AedelstenContextHandle],                        ctypes.c_int64),
    ("free_scratch_c",         [AedelstenContextHandle],                            None),
    ("set_finite_nuclear_model_c", [AedelstenContextHandle],                          ctypes.c_int),
    ("compute_centre_of_mass_c", [AedelstenContextHandle, ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)], None),
    ("compute_nuclear_repulsion_energy_c", [AedelstenContextHandle],                  ctypes.c_double),
    ("compute_nuclear_charge_c", [AedelstenContextHandle],                            ctypes.c_double),
    ("compute_nuclear_multipole_interaction_energy_c", [AedelstenContextHandle],      ctypes.c_double),
    ("compute_nuclear_aimp_embedding_energy_c", [AedelstenContextHandle],             ctypes.c_double),
    ("get_n_cont_c",             [AedelstenContextHandle],                            ctypes.c_int),
    ("get_n_centres_c",          [AedelstenContextHandle],                            ctypes.c_int64),
    ("get_centre_geometry_c",    [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("get_nuclear_charges_c",    [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("unrescale_cartesian_ao_rows_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("get_n_prim_c",             [AedelstenContextHandle],                            ctypes.c_int),
    ("get_n_prim_unique_c",      [AedelstenContextHandle],                            ctypes.c_int),
    ("set_external_multipoles_c",  [AedelstenContextHandle, ctypes.c_char_p],         ctypes.c_int),
    ("free_external_multipoles_c", [AedelstenContextHandle],                          None),
    ("has_external_multipoles_c", [AedelstenContextHandle],                           ctypes.c_int),
    ("set_polarizable_sites_c",  [AedelstenContextHandle, ctypes.c_char_p],         ctypes.c_int),
    ("free_polarizable_sites_c", [AedelstenContextHandle],                          None),
    ("get_n_polarizable_sites_c", [AedelstenContextHandle],                          ctypes.c_int64),
    ("has_polarizable_sites_c",  [AedelstenContextHandle],                          ctypes.c_int),
    ("has_polarization_c",       [AedelstenContextHandle],                          ctypes.c_int),
    ("count_electric_field_sites_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int64),
    ("get_input_indices_c",      [AedelstenContextHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("compute_electric_field_integrals_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("compute_static_field_c",   [AedelstenContextHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("compute_electronic_field_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("set_multipole_convention_c", [AedelstenContextHandle, ctypes.c_int],          ctypes.c_int),
    ("get_multipole_convention_c", [AedelstenContextHandle],                        ctypes.c_int),
    ("set_polarization_c",       [AedelstenContextHandle, ctypes.POINTER(PolarizationOptions)], ctypes.c_int),
    ("free_polarization_c",      [AedelstenContextHandle],                          None),
    ("compute_induction_gradient_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p],
     ctypes.c_int),
    ("compute_induction_c",      [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("compute_induction_within_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("get_polarization_storage_c", [AedelstenContextHandle, ctypes.c_void_p], ctypes.c_int),
    ("set_resident_field_budget_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("compute_induction_response_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
    ("compute_induction_couplings_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
    ("compute_effective_external_field_c", [AedelstenContextHandle, ctypes.c_void_p], ctypes.c_int),
    ("compute_induction_states_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                     ctypes.c_void_p],                                ctypes.c_int),
    ("compute_induction_states_within_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_void_p, ctypes.c_double],        ctypes.c_int),
    ("get_induced_dipoles_c",    [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("set_induced_dipoles_c",    [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("get_n_embedding_aimps_c",  [AedelstenContextHandle],                            ctypes.c_int64),
    ("get_embedding_aimp_sites_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_int64],                                 ctypes.c_int),
    ("has_core_aimp_c",          [AedelstenContextHandle],                            ctypes.c_int),
    ("has_embedding_aimp_c",     [AedelstenContextHandle],                            ctypes.c_int),
    ("set_embedding_aimp_operators_c", [AedelstenContextHandle, ctypes.c_char_p],     ctypes.c_int),
    ("free_embedding_aimp_operators_c", [AedelstenContextHandle],                     None),
    ("compute_one_electron_integrals_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                           ctypes.c_int, ctypes.c_int],                ctypes.c_int),
    ("get_n_gradient_centres_c", [AedelstenContextHandle],                          ctypes.c_int),
    ("set_centre_positions_c",   [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("compute_nuclear_repulsion_gradient_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
    ("compute_nuclear_aimp_embedding_gradient_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
    ("compute_nuclear_multipole_interaction_gradient_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p],
     ctypes.c_int),
    ("compute_one_electron_gradient_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
    ("compute_moment_component_count_c", [ctypes.c_int],                          ctypes.c_int),
    ("compute_one_electron_component_count_c", [ctypes.c_int, ctypes.c_int],         ctypes.c_int),
    ("compute_one_electron_picture_change_integrals_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int,
                                                           ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                                           ctypes.c_double, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("construct_x2c_one_electron_hamiltonian_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p,
                                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                    ctypes.c_double],                   ctypes.c_int),
    ("contract_one_electron_matrix_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_int],                                ctypes.c_int),
    ("contract_one_electron_array3d_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_int],              ctypes.c_int),
    ("compute_eri_stored_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_double],                      ctypes.c_int),
    ("compute_s4_size_c",        [AedelstenContextHandle],                            ctypes.c_int),
    ("compute_s8_size_c",        [AedelstenContextHandle],                            ctypes.c_int),
    ("set_eri_data_c",           [AedelstenContextHandle, ctypes.c_double],           ctypes.c_int),
    ("free_eri_data_c",          [AedelstenContextHandle],                            None),
    ("compute_fock_direct_restricted_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                           ctypes.c_double, ctypes.c_double],          ctypes.c_int),
    ("compute_fock_direct_general_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double],             ctypes.c_int),
    ("compute_fock_direct_general_multi_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                              ctypes.c_int, ctypes.c_double, ctypes.c_double],  ctypes.c_int),
    ("compute_fock_direct_mixed_active_space_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                                   ctypes.c_double, ctypes.c_double, ctypes.c_double], ctypes.c_int),
    ("compute_fock_direct_active_space_c", [AedelstenContextHandle] + [ctypes.c_void_p] * 8
                                           + [ctypes.c_int, ctypes.c_int] + [ctypes.c_double] * 6, ctypes.c_int),
    ("transform_pair_coulomb_one_index_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int],                  ctypes.c_int),
    ("compute_eri_half_transformed_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_int],                                            ctypes.c_int),
    ("compute_eri_half_transformed_on_node_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                                 ctypes.c_double],                                 ctypes.c_int),
    ("compute_eri_half_transformed_sigma_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                               ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_double, ctypes.c_double, ctypes.c_int],    ctypes.c_int),
    ("compute_eri_half_transformed_sigma_on_node_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p,
                                                       ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                       ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                                       ctypes.c_double],                           ctypes.c_int),
    ("half_transform_scratch_bytes_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int],                            ctypes.c_int64),
    ("scratch_bytes_c",        [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int64),
    ("scratch_breakdown_field_count_c", [],                                         ctypes.c_int),
    ("scratch_breakdown_field_name_c", [ctypes.c_int],                              ctypes.c_char_p),
    ("scratch_breakdown_c",    [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_int],                                      ctypes.c_int),
    ("compute_fock_direct_unrestricted_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                             ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
                                             ctypes.c_double],                         ctypes.c_int),
    ("compute_two_electron_gradient_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
                                          ctypes.c_double, ctypes.c_void_p],        ctypes.c_int),
    # Free J/K builders (no ctx: purely a function of their own arguments,
    # exactly like the Fortran/Julia interfaces' free-function design).
    ("compute_j_and_k_matrix_stored_c", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_int, ctypes.c_double, ctypes.c_double], ctypes.c_int),
    ("compute_j_and_k_unrestricted_matrix_stored_c", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                           ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
                                                           ctypes.c_double],           ctypes.c_int),
    ("compute_j_matrix_stored_c", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_double],                              ctypes.c_int),
    ("compute_k_matrix_stored_c", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_double],                              ctypes.c_int),
    ("compute_k_matrix_unrestricted_stored_c", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                     ctypes.c_double],                 ctypes.c_int),
    # Memory diagnostics, Molcas-export identity data, Cartesian rescaling.
    ("memory_breakdown_field_count_c", [],                                            ctypes.c_int),
    ("memory_breakdown_field_name_c", [ctypes.c_int],                                 ctypes.c_char_p),
    ("memory_breakdown_c",       [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("get_n_primitive_rows_c",   [AedelstenContextHandle],                            ctypes.c_int),
    ("get_basis_function_ids_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    # Shells and the AO values of a list of them.
    ("get_n_shells_c",           [AedelstenContextHandle],                            ctypes.c_int),
    ("get_shells_c",             [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("compute_shell_extents_c",  [AedelstenContextHandle, ctypes.c_double, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("compute_ao_values_c",      [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("compute_ao_hessians_c",    [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("get_n_model_functions_c", [AedelstenContextHandle], ctypes.c_int),
    ("get_model_functions_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("compute_model_basis_overlaps_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("compute_model_fockian_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("compute_overlap_with_c", [AedelstenContextHandle, AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int),
    ("get_primitive_ids_c",      [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("get_primitives_c",         [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("get_centre_label_c",       [AedelstenContextHandle, ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
    ("rescale_cartesian_ao_rows_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_int, ctypes.POINTER(ctypes.c_int)],     ctypes.c_int),
    ("rescale_cartesian_ao_columns_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.POINTER(ctypes.c_int)],  ctypes.c_int),
    # The X2C picture-change cache.
    ("free_picture_change_c",    [AedelstenContextHandle],                            None),
    ("has_picture_change_c",     [AedelstenContextHandle],                            ctypes.c_int),
    ("set_picture_change_transforms_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_double],
     ctypes.c_int),
    ("set_x2c_frozen_density_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("free_x2c_frozen_density_c", [AedelstenContextHandle],                          None),
    ("compute_x2c_frozen_kinetic_gradient_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p],
     ctypes.c_int),
    ("compute_x2c_frozen_potential_gradient_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_void_p], ctypes.c_int),
    ("get_picture_change_transforms_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                                                                                      ctypes.c_int),
    ("transform_one_electron_operator_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("transform_one_electron_spin_operator_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_int],                       ctypes.c_int),
    ("compute_picture_changed_one_electron_operator_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_void_p,
                                                          ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                                          ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int],
                                                                                      ctypes.c_int),
    ("compute_amfi_integrals_c", [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    # Point-group symmetry.
    ("set_symmetry_c",           [AedelstenContextHandle, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double],                  ctypes.c_int),
    ("free_symmetry_c",          [AedelstenContextHandle],                            None),
    ("get_symmetry_options_c",   [AedelstenContextHandle, ctypes.c_void_p],           ctypes.c_int),
    ("get_symmetrisation_report_c", [AedelstenContextHandle, ctypes.c_char_p, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_void_p],  ctypes.c_int),
    ("get_point_group_name_c",   [AedelstenContextHandle, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
    ("get_n_irreps_c",           [AedelstenContextHandle],                            ctypes.c_int),
    ("get_irrep_name_c",         [AedelstenContextHandle, ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
    ("get_irrep_dimension_c",    [AedelstenContextHandle, ctypes.c_int],              ctypes.c_int),
    ("get_irrep_index_c",        [AedelstenContextHandle, ctypes.c_char_p],           ctypes.c_int),
    ("get_totally_symmetric_irrep_c", [AedelstenContextHandle],                       ctypes.c_int),
    ("get_cartesian_irrep_c",    [AedelstenContextHandle, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("get_symmetry_standard_axes_c", [AedelstenContextHandle, ctypes.c_void_p],       ctypes.c_int),
    ("get_symmetry_operation_count_c", [AedelstenContextHandle],                      ctypes.c_int),
    ("get_symmetry_operation_matrix_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
    ("get_n_symmetry_centres_c", [AedelstenContextHandle],                            ctypes.c_int),
    ("get_symmetry_centre_images_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int],
                                                                                      ctypes.c_int),
    ("symmetrise_matrix_c",      [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p],
                                                                                      ctypes.c_int),
    ("analyse_matrix_c",         [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p],
                                                                                      ctypes.c_int),
    ("get_irrep_norms_c",        [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int],
                                                                                      ctypes.c_int),
    ("symmetrise_orbitals_c",    [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)],  ctypes.c_int),
    ("get_ao_irrep_counts_c",    [AedelstenContextHandle, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("apply_symmetry_operation_c", [AedelstenContextHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_int],                                      ctypes.c_int),
    ("get_irrep_character_c",    [AedelstenContextHandle, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)],
                                                                                      ctypes.c_int),
    ("get_irrep_matrix_c",       [AedelstenContextHandle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int)],                      ctypes.c_int),
]

_QUERY_SIGNATURES = [
    ("aedelsten_query_i32_c", [ctypes.c_char_p], ctypes.c_int),
    ("aedelsten_query_f64_c", [ctypes.c_char_p], ctypes.c_double),
]

_RUNTIME_SIGNATURES = [
    ("aedelsten_mpi_init_c",         [ctypes.c_int],                              ctypes.c_int),
    ("aedelsten_mpi_finalize_c",     [],                                          ctypes.c_int),
    ("aedelsten_mpi_initialized_c",  [],                                          ctypes.c_int),
    ("aedelsten_mpi_thread_level_c", [],                                          ctypes.c_int),
    ("aedelsten_mpi_rank_c",         [ctypes.c_int, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("aedelsten_mpi_size_c",         [ctypes.c_int, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("aedelsten_get_version_c",      [ctypes.c_char_p, ctypes.c_int],             ctypes.c_int),
    ("aedelsten_get_source_commit_c", [ctypes.c_char_p, ctypes.c_int],            ctypes.c_int),
    ("aedelsten_blas_set_threads_c", [ctypes.c_int],                              ctypes.c_int),
    ("aedelsten_blas_threads_c",     [],                                          ctypes.c_int),
    ("aedelsten_blas_info_c",        [ctypes.c_char_p, ctypes.c_int],             ctypes.c_int),
    ("aedelsten_blas_available_cores_c", [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("aedelsten_blas_available_cores_info_c", [ctypes.c_char_p, ctypes.c_int],    ctypes.c_int),
    ("aedelsten_mpi_shared_array_c", [ctypes.c_int, ctypes.c_int64, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))], ctypes.c_void_p),
    ("aedelsten_mpi_shared_array_sync_c", [ctypes.c_void_p],                       ctypes.c_int),
    ("aedelsten_mpi_shared_array_free_c", [ctypes.c_void_p],                       None),
    ("aedelsten_mpi_collective_seconds_c", [],                                    ctypes.c_double),
    ("aedelsten_profile_enabled_c",  [],                                          ctypes.c_int),
    ("aedelsten_timing_count_c",     [ctypes.c_int],                              ctypes.c_int),
    ("aedelsten_timing_span_c",      [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64)], ctypes.c_int),
    ("aedelsten_span_tree_count_c",  [],                                          ctypes.c_int),
    ("aedelsten_span_tree_node_c",   [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
                                      ctypes.POINTER(ctypes.c_int64)],             ctypes.c_int),
    ("aedelsten_citation_count_c",   [],                                          ctypes.c_int),
    ("aedelsten_citation_length_c",  [ctypes.c_int, ctypes.c_int],                ctypes.c_int64),
    ("aedelsten_citation_c",         [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
    ("aedelsten_citations_bibtex_length_c", [],                                   ctypes.c_int64),
    ("aedelsten_citations_bibtex_c", [ctypes.c_char_p, ctypes.c_int],             ctypes.c_int),
    ("aedelsten_clear_citations_c",  [],                                          None),
]

class _Constants:
    """
    Integer constants resolve lazily from the Rust library on first access,
    via aedelsten_query_i32_c(name) -- the attribute name IS the query key,
    verbatim (see each wire enum's NAME_TABLE on the Rust side), so a new
    constant added there needs no matching edit here. Resolved values are
    cached directly into the instance dict, so repeat lookups never re-cross
    the FFI boundary.
    """

    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name: str) -> int:
        if name.startswith('_'):
            raise AttributeError(name)
        value = self._lib.aedelsten_query_i32_c(name.encode())
        if value == -1:
            raise AttributeError(f"'{name}' is not a known aedelsten constant")
        self.__dict__[name] = value
        return value


_CONSTANTS: "_Constants | None" = None

def _lib_name() -> str:
    if sys.platform == "win32":  return "aedelsten.dll"
    if sys.platform == "darwin": return "libaedelsten.dylib"
    return "libaedelsten.so"

def _find_lib() -> Path:
    name = _lib_name()
    if env := os.environ.get("AEDELSTEN_LIB"):
        return Path(env)
    search_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    for directory in os.environ.get(search_var, "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists():
            return candidate
    bundled = Path(__file__).parent.parent.parent / "lib" / name
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        f"Could not find {name}. Set AEDELSTEN_LIB to its full path, or add its directory to "
        f"{'DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'}."
    )

def load_aedelsten_lib() -> ctypes.CDLL:
    """The library, loaded on first use (from AEDELSTEN_LIB, else the library search path, else PREFIX/lib beside
    this module) with every function's signature declared."""
    global _aedelsten_lib, _CONSTANTS
    if _aedelsten_lib is not None:
        return _aedelsten_lib
    lib = ctypes.CDLL(str(_find_lib()))
    for name, argtypes, restype in _SIGNATURES + _QUERY_SIGNATURES + _RUNTIME_SIGNATURES:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    _CONSTANTS = _Constants(lib)
    _aedelsten_lib = lib
    return lib

def _get_constants() -> _Constants:
    if _CONSTANTS is None:
        load_aedelsten_lib()
    return _CONSTANTS

# --- Wire-key resolution ---
# Every named constant is queried by name through _Constants.__getattr__
# above -- never hardcoded as a literal int here, so a new constant added
# on the Rust side needs no matching edit in this file.

def _resolve_contraction_type(value: str | None) -> int:
    if value is None:
        return _get_constants().default_contraction_type
    return getattr(_get_constants(), "contraction_type_" + value.lower())

def _resolve_byte_unit(value: str) -> int:
    return getattr(_get_constants(), "byte_unit_" + value.lower())

def _resolve_cartesian_convention(value: str) -> int:
    return getattr(_get_constants(), "cartesian_convention_" + value.lower())

# Bare suffix ("overlap") or fully-qualified key ("oneint_overlap") --
# same convention as the Fortran/Julia interfaces' resolve_operator.
def _resolve_operator(operator: str) -> int:
    key = operator.lower()
    if not key.startswith("oneint_"):
        key = "oneint_" + key
    return getattr(_get_constants(), key)

def _resolve_eri_symmetry(ao_symmetry: str) -> int:
    key = ao_symmetry.lower()
    if not key.startswith("eri_ao_symmetry_"):
        key = "eri_ao_symmetry_" + key
    return getattr(_get_constants(), key)

def _resolve_eri_index_order(ao_index_order: str) -> int:
    key = ao_index_order.lower()
    if not key.startswith("eri_index_order_"):
        key = "eri_index_order_" + key
    return getattr(_get_constants(), key)

# Bundled range/omega, matching Fortran/Julia's resolve_eri_range: both
# present or both absent (absent = full-range Coulomb).
def _resolve_eri_range(eri_range: str | None, omega: float | None) -> tuple[int, float]:
    if (eri_range is None) != (omega is None):
        raise ValueError("eri_range and omega must both be given, or both omitted (omitted = full range)")
    if eri_range is None:
        return 0, 0.0
    key = eri_range.lower()
    if not key.startswith("eri_"):
        key = "eri_" + key
    return getattr(_get_constants(), key), float(omega)

# --- Array validation ---
# Never silently copy a caller's array (e.g. via np.ascontiguousarray) --
# the same principle the Fortran interface enforces with is_contiguous():
# a wrong-layout array is a caller error to fix, not something to paper
# over with a hidden copy.

# The scratch query's operations by name, their C kind codes and parameters (`scratch_bytes`).
_SCRATCH_OPERATIONS = {
    "eri_data": (1, ()),
    "restricted_fock": (2, ()),
    "unrestricted_fock": (3, ()),
    "two_electron_gradient": (4, ("terms",)),
    "general_fock": (5, ()),
    "general_multi_fock": (6, ("densities",)),
    "active_space_fock": (7, ("active",)),
    "mixed_active_space_fock": (8, ("active",)),
    "half_transform": (9, ("left", "right", "sigma", "node_ranks")),
    "amfi": (10, ()),
}
_SCRATCH_ITEMS = {1: "eri_data_entry", 2: "gradient_pair_data", 3: "derivative_bound", 4: "quartet", 5: "amfi_atom_kind"}


def _scratch_operation_rows(operations) -> np.ndarray:
    """int32 rows of five (the C layout) of operations given as (name, *parameters), None for a parameter's None."""
    rows = np.zeros((max(len(operations), 1), 5), dtype=np.int32)
    for i, operation in enumerate(operations):
        name, *parameters = (operation,) if isinstance(operation, str) else operation
        if name not in _SCRATCH_OPERATIONS:
            raise ValueError(f"unknown scratch operation {name!r}: one of {', '.join(_SCRATCH_OPERATIONS)}")
        code, wanted = _SCRATCH_OPERATIONS[name]
        if len(parameters) != len(wanted):
            raise ValueError(f"scratch operation {name!r} takes ({', '.join(wanted)}), got {tuple(parameters)}")
        rows[i, 0] = code
        rows[i, 1:1 + len(parameters)] = [-1 if p is None else int(p) for p in parameters]
    return rows


def _check_f64_c_contiguous(arr: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.dtype != np.float64:
        raise TypeError(f"{name} must be a numpy float64 array, got {type(arr).__name__} "
                         f"dtype={getattr(arr, 'dtype', None)}")
    if not arr.flags['C_CONTIGUOUS']:
        raise ValueError(
            f"{name} must be C-contiguous (got a strided/non-contiguous array) -- Aedelsten never "
            "silently copies; pass a C-contiguous array instead"
        )
    return arr

def _check_square(arr: np.ndarray, name: str) -> int:
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square (n, n), got shape {arr.shape}")
    return arr.shape[0]

# --- Context ---

class AedelstenContext:
    """The context of one input file: its basis, geometry and embedding, and every computation over them."""
    _ctx: AedelstenContextHandle

    def __init__(self, filename: str, contraction_type: str | None = None, comm: "int | None" = None):
        """The context of the input file `filename`; its parallel computations are split over the ranks of
        communicator `comm` (its Fortran handle, e.g. mpi4py's `comm.py2f()`, or `_get_constants().mpi_comm_world`;
        None: this process alone, without MPI). Every collective method must then be called by every rank of
        `comm`."""
        load_aedelsten_lib()
        self._ctx = _aedelsten_lib.set_aedelsten_context_c(
            filename.encode('utf-8'), ctypes.c_int(_resolve_contraction_type(contraction_type)), ctypes.c_int(_communicator_handle(comm)),
        )
        if not self._ctx:
            raise RuntimeError(f"Failed to create context for '{filename}'")

    def set_communicator(self, comm: "int | None"):
        """Splits the context's parallel computations over communicator `comm` from now on (as in the constructor)."""
        if _aedelsten_lib.set_communicator_c(self._ctx, ctypes.c_int(_communicator_handle(comm))) != 0:
            raise RuntimeError(f"Failed to set communicator {comm}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._free_ctx()

    def __del__(self):
        self._free_ctx()

    def _free_ctx(self):
        if hasattr(self, '_ctx') and self._ctx:
            _aedelsten_lib.free_aedelsten_context_c(self._ctx)
            self._ctx = None

    # --- Shell pairs / scratch / finite nuclear model ---

    def set_shell_pairs(self, overlap_threshold: float | None = None):
        """Derive the shell pairs, dropping primitive pairs whose overlap is below `overlap_threshold`; None (the
        default) drops none."""
        threshold = -1.0 if overlap_threshold is None else float(overlap_threshold)  # -1: no screening
        result = _aedelsten_lib.set_shell_pairs_c(self._ctx, ctypes.c_double(threshold))
        if result != 0:
            raise RuntimeError(f"Failed to set shell pairs (threshold={threshold}, code={result})")

    def free_shell_pairs(self):
        """Frees the shell pairs, and with them the ERI data."""
        _aedelsten_lib.free_shell_pairs_c(self._ctx)

    def has_primitive(self) -> bool:
        """Whether the primitive basis is set (always for a primitive context)."""
        return bool(_aedelsten_lib.has_primitive_c(self._ctx))

    def set_primitive_shell_pairs(self, overlap_threshold: float | None = None):
        """Derive the decontracted (primitive) shell pairs, needed by
        compute_one_electron_integrals(..., primitive=True), dropping pairs whose
        overlap is below `overlap_threshold`; None (the default) drops none.
        Requires set_shell_pairs() first.
        """
        threshold = -1.0 if overlap_threshold is None else float(overlap_threshold)  # -1: no screening
        result = _aedelsten_lib.set_primitive_shell_pairs_c(self._ctx, ctypes.c_double(threshold))
        if result != 0:
            raise RuntimeError(f"Failed to set primitive shell pairs (threshold={threshold}, code={result})")

    def free_primitive_shell_pairs(self):
        """Frees the primitive shell pairs."""
        _aedelsten_lib.free_primitive_shell_pairs_c(self._ctx)

    def free_input(self):
        """Frees the parsed input (basis, nuclei, embedding). Only after the last call that reads it (the operators,
        the shell pairs, the primitive shell pairs, the embedding): such a call afterwards stops the program."""
        _aedelsten_lib.free_input_c(self._ctx)

    def free_primitive_shells(self):
        """Frees the primitive shells made by set_primitive_shell_pairs; the primitive shell pairs stay."""
        _aedelsten_lib.free_primitive_shells_c(self._ctx)

    def free_contraction_matrix(self):
        """Frees the contraction matrix made by set_primitive_shell_pairs."""
        _aedelsten_lib.free_contraction_matrix_c(self._ctx)

    def set_scratch(self, size: int, unit: str):
        """The buffer every integral computation takes its scratch from; the caller chooses its size, the library
        makes no judgment about it (unit: "kib", "mib" or "gib"); scratch_bytes gives what the chosen operations
        need, from the input alone. That depends only on the basis's largest shells, not on the molecule; two-electron
        integral gradients need the most, one-electron integrals (X2C included) little. The buffer is committed
        lazily. It never grows: a computation that runs out of it fails with an error naming the size needed."""
        result = _aedelsten_lib.set_scratch_c(self._ctx, ctypes.c_int(size), ctypes.c_int(_resolve_byte_unit(unit)))
        if result != 0:
            raise RuntimeError(f"Failed to set scratch (size={size}, unit={unit}, code={result})")

    def free_scratch(self):
        """Frees the scratch buffer."""
        _aedelsten_lib.free_scratch_c(self._ctx)

    def resize_scratch(self, size: int, unit: str):
        """Replaces the scratch buffer (arguments as set_scratch), e.g. to shrink it to get_scratch_peak_bytes()
        plus a margin, or to grow it after an out-of-space error. The new buffer is committed lazily; the high-water
        mark starts over."""
        result = _aedelsten_lib.resize_scratch_c(self._ctx, ctypes.c_int(size), ctypes.c_int(_resolve_byte_unit(unit)))
        if result != 0:
            raise RuntimeError(f"Failed to resize scratch (size={size}, unit={unit}, code={result})")

    def get_scratch_peak_bytes(self) -> int:
        """The most bytes of the scratch any computation has used so far (its high-water mark)."""
        result = _aedelsten_lib.get_scratch_peak_bytes_c(self._ctx)
        if result < 0:
            raise RuntimeError("No scratch (set_scratch first)")
        return result

    def set_finite_nuclear_model(self):
        """Switch the nuclear charge distribution from point charges to the Visscher-Dyall finite-nucleus model."""
        result = _aedelsten_lib.set_finite_nuclear_model_c(self._ctx)
        if result != 0:
            raise RuntimeError(f"Failed to set finite nuclear model (code={result})")

    # --- Basis-size / molecular-property getters ---

    def get_n_cont(self) -> int:
        """The number of contracted basis functions (AOs)."""
        return _aedelsten_lib.get_n_cont_c(self._ctx)

    def get_n_prim(self) -> int:
        """The sum of the shells' primitive AOs (a primitive shared by sibling segments once per sibling)."""
        return _aedelsten_lib.get_n_prim_c(self._ctx)

    def get_n_gradient_centres(self) -> int:
        """The centres a gradient has columns for: every centre but the orthogonality functions of embedding AIMPs."""
        return int(_aedelsten_lib.get_n_gradient_centres_c(self._ctx))

    def get_n_prim_unique(self) -> int:
        """The number of distinct primitive AOs: the dimension of the primitive basis."""
        return _aedelsten_lib.get_n_prim_unique_c(self._ctx)

    def get_n_primitive_rows(self) -> int:
        """The number of rows of get_primitive_ids and get_primitives."""
        return _aedelsten_lib.get_n_primitive_rows_c(self._ctx)

    # --- Memory diagnostics, Molcas-export identity data, Cartesian rescaling ---

    def memory_breakdown(self) -> dict[str, int]:
        """The memory this context holds now, a dict of bytes by field name (0 for a field that is empty or
        invalidated)."""
        n = _aedelsten_lib.memory_breakdown_field_count_c()
        names = [_aedelsten_lib.memory_breakdown_field_name_c(i).decode() for i in range(n)]
        bytes_ = np.zeros(n, dtype=np.uint64)
        result = _aedelsten_lib.memory_breakdown_c(self._ctx, bytes_.ctypes.data, n)
        if result != 0:
            raise RuntimeError(f"memory_breakdown failed (code={result})")
        return dict(zip(names, (int(b) for b in bytes_)))

    def get_basis_function_ids(self) -> np.ndarray:
        """Molcas-style BASIS_FUNCTION_IDS: an (4, n_cont) array, ids[0]=centre, ids[1]=shell,
        ids[2]=l, ids[3]=m, one column per AO in AO order. See BasisFunctionId's own doc comment
        on the Rust side for exactly what l/m mean for Cartesian vs Spherical shells."""
        n = self.get_n_cont()
        ids = np.zeros((4, n), dtype=np.int32)
        result = _aedelsten_lib.get_basis_function_ids_c(self._ctx, ids.ctypes.data, 4 * n)
        if result != 0:
            raise RuntimeError(f"get_basis_function_ids failed (code={result})")
        return ids

    # --- The model basis: free-atom reference orbitals (ANO-RCC), for an atomic frozen core ---

    def get_n_model_functions(self) -> int:
        """The number of model functions: the free-atom reference orbitals of the model basis (ANO-RCC, the model
        Fockian's), one per component, on the QM centres that are atoms; a core AIMP's share is left out."""
        n_model = _aedelsten_lib.get_n_model_functions_c(self._ctx)
        if n_model < 0:
            raise RuntimeError("get_n_model_functions failed")
        return n_model

    def get_model_functions(self) -> tuple[np.ndarray, np.ndarray]:
        """The model functions, in the model basis's AO order: `(ids, energies)`, `ids` a (4, n_model) int32 array,
        ids[0]=centre (0-based, get_centre_label's index), ids[1]=l, ids[2]=m (as get_basis_function_ids gives it
        for a shell of the same type: s and p Cartesian, p as x, y, z = -1, 0, 1; d and up spherical, -l..l),
        ids[3]=the orbital's ordinal within its l in the reference library (0-based), counting what a core AIMP
        covers (a centre whose AIMP projection basis has k columns at l starts at k); `energies` the (n_model,)
        reference orbital energies."""
        n_model = self.get_n_model_functions()
        ids = np.zeros((4, n_model), dtype=np.int32)
        energies = np.zeros(n_model, dtype=np.float64)
        result = _aedelsten_lib.get_model_functions_c(self._ctx, ids.ctypes.data, 4 * n_model, energies.ctypes.data, n_model)
        if result != 0:
            raise RuntimeError(f"get_model_functions failed (code={result})")
        return ids, energies

    def compute_model_basis_overlaps(self, with_model_model: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """The overlaps of the model basis: `(s_working_model, s_model_model)`. As everywhere in this interface the
        arrays are the library's column-major matrices read row-major, so with i a model function and nu a working
        basis function, `s_working_model[nu, i] = <i|nu>` ((get_n_cont(), n_model)), and `s_model_model[j, i] =
        <i|j>` ((n_model, n_model), symmetric). `s_model_model` is None unless `with_model_model`. For orbitals `c`
        as rows (n_mo, get_n_cont()), the overlaps with the model functions are `c @ s_working_model`."""
        n, n_model = self.get_n_cont(), self.get_n_model_functions()
        s_wm = np.zeros((n, n_model), dtype=np.float64)
        s_mm = np.zeros((n_model, n_model), dtype=np.float64) if with_model_model else None
        result = _aedelsten_lib.compute_model_basis_overlaps_c(
            self._ctx, s_wm.ctypes.data, n * n_model, None if s_mm is None else s_mm.ctypes.data, 0 if s_mm is None else n_model * n_model,
        )
        if result != 0:
            raise RuntimeError(f"compute_model_basis_overlaps failed (code={result})")
        return s_wm, s_mm

    def compute_model_fockian(self) -> np.ndarray:
        """The model-Fock start guess `S_mw^T diag(E) S_mw` (symmetric, (get_n_cont(), get_n_cont())), from the
        model basis's overlaps and reference energies."""
        n = self.get_n_cont()
        fockian = np.zeros((n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_model_fockian_c(self._ctx, fockian.ctypes.data, n * n)
        if result != 0:
            raise RuntimeError(f"compute_model_fockian failed (code={result})")
        return fockian

    def compute_overlap_with(self, other: "AedelstenContext") -> np.ndarray:
        """The overlap between this context's basis and `other`'s, which may differ in basis set, geometry, harmonics
        and contraction type; unscreened. As everywhere in this interface the library's column-major matrix read
        row-major: `s[j, i] = <self_i|other_j>`, (other.get_n_cont(), get_n_cont()). For orbitals `c` of this
        basis as rows (n_mo, get_n_cont()), their overlaps with other's functions are `c @ s.T`."""
        s = np.zeros((other.get_n_cont(), self.get_n_cont()), dtype=np.float64)
        result = _aedelsten_lib.compute_overlap_with_c(self._ctx, other._ctx, s.ctypes.data, s.size)
        if result != 0:
            raise RuntimeError(f"compute_overlap_with failed (code={result})")
        return s

    def get_primitive_ids(self) -> np.ndarray:
        """Molcas-style PRIMITIVE_IDS: a (3, get_n_primitive_rows()) array, ids[0]=centre,
        ids[1]=l, ids[2]=shell, parallel to get_primitives (column i of each belongs together)."""
        n = self.get_n_primitive_rows()
        ids = np.zeros((3, n), dtype=np.int32)
        result = _aedelsten_lib.get_primitive_ids_c(self._ctx, ids.ctypes.data, 3 * n)
        if result != 0:
            raise RuntimeError(f"get_primitive_ids failed (code={result})")
        return ids

    def get_primitives(self) -> np.ndarray:
        """Molcas-style PRIMITIVES: a (2, get_n_primitive_rows()) array, primitives[0]=exponent,
        primitives[1]=coefficient, parallel to get_primitive_ids."""
        n = self.get_n_primitive_rows()
        primitives = np.zeros((2, n), dtype=np.float64)
        result = _aedelsten_lib.get_primitives_c(self._ctx, primitives.ctypes.data, 2 * n)
        if result != 0:
            raise RuntimeError(f"get_primitives failed (code={result})")
        return primitives

    def get_n_shells(self) -> int:
        """The number of shells."""
        return _aedelsten_lib.get_n_shells_c(self._ctx)

    def get_shells(self) -> np.ndarray:
        """Every shell in AO order: a (4, get_n_shells()) array, shells[0]=centre (0-based), shells[1]=l (negated
        for a Cartesian shell), shells[2]=first AO (0-based), shells[3]=AO count."""
        n = self.get_n_shells()
        shells = np.zeros((4, n), dtype=np.int32)
        result = _aedelsten_lib.get_shells_c(self._ctx, shells.ctypes.data, 4 * n)
        if result != 0:
            raise RuntimeError(f"get_shells failed (code={result})")
        return shells

    def compute_shell_extents(self, threshold: float) -> np.ndarray:
        """Per shell, the distance from its centre beyond which its AOs and every component of their gradients are
        below `threshold` in magnitude."""
        n = self.get_n_shells()
        extents = np.zeros(n, dtype=np.float64)
        result = _aedelsten_lib.compute_shell_extents_c(self._ctx, threshold, extents.ctypes.data, n)
        if result != 0:
            raise RuntimeError(f"compute_shell_extents failed (code={result})")
        return extents

    def compute_ao_values(self, shells, points: np.ndarray, gradients: bool = False):
        """The AOs of the shells `shells` (0-based, any order and subset) at `points`, (n_points, 3) or
        (n_points, 4) (x, y, z and a weight that is not read): values (n_points, n_listed), the AOs of the listed
        shells in the order of the list; with `gradients`, also (3, n_points, n_listed) (x, y, z)."""
        shell_list = np.ascontiguousarray(shells, dtype=np.int32)
        points = _check_f64_c_contiguous(points, "points")
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError(f"points must be (n_points, 3) or (n_points, 4), got shape {points.shape}")
        n_points, rows = points.shape
        counts = self.get_shells()[3]
        if shell_list.ndim != 1 or np.any(shell_list < 0) or np.any(shell_list >= counts.size):
            raise ValueError(f"shells must be a list of shell indices below {counts.size}")
        n_listed = int(counts[shell_list].sum())
        values = np.zeros((n_points, n_listed), dtype=np.float64)
        grads = np.zeros((3, n_points, n_listed), dtype=np.float64) if gradients else None
        result = _aedelsten_lib.compute_ao_values_c(
            self._ctx, shell_list.ctypes.data, shell_list.size, points.ctypes.data, rows, n_points,
            values.ctypes.data, grads.ctypes.data if gradients else None, n_listed)
        if result != 0:
            raise RuntimeError(f"compute_ao_values failed (code={result})")
        return (values, grads) if gradients else values

    def compute_ao_hessians(self, shells, points: np.ndarray):
        """As `compute_ao_values` with gradients, and the second derivatives: the tuple of the values
        (n_points, n_listed), the gradients (3, n_points, n_listed) (x, y, z) and the second derivatives
        (6, n_points, n_listed) (xx, xy, xz, yy, yz, zz)."""
        shell_list = np.ascontiguousarray(shells, dtype=np.int32)
        points = _check_f64_c_contiguous(points, "points")
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError(f"points must be (n_points, 3) or (n_points, 4), got shape {points.shape}")
        n_points, rows = points.shape
        counts = self.get_shells()[3]
        if shell_list.ndim != 1 or np.any(shell_list < 0) or np.any(shell_list >= counts.size):
            raise ValueError(f"shells must be a list of shell indices below {counts.size}")
        n_listed = int(counts[shell_list].sum())
        values = np.zeros((n_points, n_listed), dtype=np.float64)
        grads = np.zeros((3, n_points, n_listed), dtype=np.float64)
        hessians = np.zeros((6, n_points, n_listed), dtype=np.float64)
        result = _aedelsten_lib.compute_ao_hessians_c(
            self._ctx, shell_list.ctypes.data, shell_list.size, points.ctypes.data, rows, n_points,
            values.ctypes.data, grads.ctypes.data, hessians.ctypes.data, n_listed)
        if result != 0:
            raise RuntimeError(f"compute_ao_hessians failed (code={result})")
        return values, grads, hessians

    def get_centre_label(self, centre_idx: int) -> str | None:
        """centre_idx is 0-based (get_basis_function_ids/get_primitive_ids's own `centre` row is
        1-based -- pass centre - 1). None if centre_idx has no label (out of range, or an
        orthogonality ghost centre) -- not an error, unlike everything else in this class."""
        buf = ctypes.create_string_buffer(64)
        result = _aedelsten_lib.get_centre_label_c(self._ctx, centre_idx, buf, len(buf))
        return buf.value.decode() if result == 0 else None

    def rescale_cartesian_ao_rows(self, convention: str, matrix: np.ndarray) -> bool:
        """Rescales `matrix` (n_cols, get_n_cont()) in place, from this library's own Cartesian
        normalization to `convention`'s ("molcas", "dalton", or "dirac", case-insensitive) -- e.g.
        an MO x AO coefficient matrix being written out for a reader that expects `convention`.
        Returns whether anything actually changed (False when every factor was exactly 1.0)."""
        n = self.get_n_cont()
        n_cols = matrix.shape[0]
        if matrix.shape != (n_cols, n) or not matrix.flags["C_CONTIGUOUS"] or matrix.dtype != np.float64:
            raise ValueError(f"rescale_cartesian_ao_rows: matrix must be a C-contiguous float64 (n_cols, {n}) array")
        convention_int = _resolve_cartesian_convention(convention)
        rescaled = ctypes.c_int(0)
        result = _aedelsten_lib.rescale_cartesian_ao_rows_c(
            self._ctx, convention_int, n_cols, matrix.ctypes.data, matrix.size, ctypes.byref(rescaled)
        )
        if result != 0:
            raise RuntimeError(f"rescale_cartesian_ao_rows failed (code={result})")
        return bool(rescaled.value)

    def unrescale_cartesian_ao_rows(self, convention: str, matrix: np.ndarray) -> bool:
        """The inverse of rescale_cartesian_ao_rows: `matrix` (n_cols, get_n_cont()) from `convention`'s Cartesian
        normalization to this library's, in place (e.g. coefficients read back from a file written for `convention`).
        Returns whether anything changed."""
        n = self.get_n_cont()
        n_cols = matrix.shape[0]
        if matrix.shape != (n_cols, n) or not matrix.flags["C_CONTIGUOUS"] or matrix.dtype != np.float64:
            raise ValueError(f"unrescale_cartesian_ao_rows: matrix must be a C-contiguous float64 (n_cols, {n}) array")
        convention_int = _resolve_cartesian_convention(convention)
        rescaled = ctypes.c_int(0)
        result = _aedelsten_lib.unrescale_cartesian_ao_rows_c(
            self._ctx, convention_int, n_cols, matrix.ctypes.data, matrix.size, ctypes.byref(rescaled)
        )
        if result != 0:
            raise RuntimeError(f"unrescale_cartesian_ao_rows failed (code={result})")
        return bool(rescaled.value)

    def rescale_cartesian_ao_columns(self, convention: str, matrix: np.ndarray) -> bool:
        """Same as rescale_cartesian_ao_rows, but for matrix (get_n_cont(), n_rows) -- e.g. a
        symmetric AO x AO matrix needs both."""
        n = self.get_n_cont()
        n_rows = matrix.shape[1]
        if matrix.shape != (n, n_rows) or not matrix.flags["C_CONTIGUOUS"] or matrix.dtype != np.float64:
            raise ValueError(f"rescale_cartesian_ao_columns: matrix must be a C-contiguous float64 ({n}, n_rows) array")
        convention_int = _resolve_cartesian_convention(convention)
        rescaled = ctypes.c_int(0)
        result = _aedelsten_lib.rescale_cartesian_ao_columns_c(
            self._ctx, convention_int, n_rows, matrix.ctypes.data, matrix.size, ctypes.byref(rescaled)
        )
        if result != 0:
            raise RuntimeError(f"rescale_cartesian_ao_columns failed (code={result})")
        return bool(rescaled.value)

    # --- Point-group symmetry ---

    def set_symmetry(self, point_group: str | None = None, principal_axis=None, geometry_tolerance: float | None = None,
                     strict_tolerance: float | None = None, average_warning: float | None = None) -> None:
        """Finds the point group of the QM centres (point_group None) or uses the named one (e.g. "C2v": the full group
        or a subgroup; principal_axis, a 3-vector, picks a subgroup's orientation). Tolerances None: the defaults
        (1e-8 Bohr, 1e-10 relative, no warning). With symmetry set the library cleans the integrals it computes
        (strictly for the QM region, averaged for the environment; see get_symmetrisation_report)."""
        name = None if point_group is None else point_group.encode()
        axis = None
        if principal_axis is not None:
            axis = np.ascontiguousarray(principal_axis, dtype=np.float64)
            if axis.shape != (3,):
                raise ValueError("principal_axis must have 3 components")
        result = _aedelsten_lib.set_symmetry_c(
            self._ctx, name, None if axis is None else axis.ctypes.data,
            -1.0 if geometry_tolerance is None else float(geometry_tolerance),
            -1.0 if strict_tolerance is None else float(strict_tolerance),
            -1.0 if average_warning is None else float(average_warning),
        )
        if result != 0:
            raise RuntimeError(f"set_symmetry failed (code={result}; the reason is on stderr)")

    def free_symmetry(self) -> None:
        """Frees the point group and its AO representation."""
        _aedelsten_lib.free_symmetry_c(self._ctx)

    def get_symmetry_options(self) -> dict:
        """The tolerances symmetry was set with: "geometry_tolerance", "strict_tolerance", "average_warning" (None: no
        warning)."""
        options = np.zeros(3, dtype=np.float64)
        if _aedelsten_lib.get_symmetry_options_c(self._ctx, options.ctypes.data) != 0:
            raise RuntimeError("get_symmetry_options failed: no symmetry set")
        return {"geometry_tolerance": options[0], "strict_tolerance": options[1],
                "average_warning": None if options[2] == 0.0 else options[2]}

    @staticmethod
    def _report(report: np.ndarray) -> dict:
        return {"irrep": int(report[0]), "max_abs_removed": report[1], "removed_norm": report[2], "norm": report[3]}

    def get_symmetrisation_report(self, name: str, primitive: bool = False) -> dict | None:
        """The report of the last cleaning of a library output (e.g. "oneint_overlap", "oneint_external_multipole",
        "x2c_one_electron_hamiltonian"), in the primitive or the contracted basis; None if it was not cleaned."""
        found = ctypes.c_int(0)
        report = np.zeros(4, dtype=np.float64)
        result = _aedelsten_lib.get_symmetrisation_report_c(self._ctx, name.encode(), int(primitive), ctypes.byref(found), report.ctypes.data)
        if result != 0:
            raise RuntimeError(f"get_symmetrisation_report failed (code={result})")
        return self._report(report) if found.value == 1 else None

    def get_point_group_name(self) -> str:
        """The point group in use, e.g. "C2v"."""
        buf = ctypes.create_string_buffer(16)
        if _aedelsten_lib.get_point_group_name_c(self._ctx, buf, len(buf)) != 0:
            raise RuntimeError("get_point_group_name failed: no symmetry set")
        return buf.value.decode()

    def get_irreps(self) -> list[tuple[str, int]]:
        """The irreps, (name, dimension), in character-table order (what every irrep index refers to)."""
        n = _aedelsten_lib.get_n_irreps_c(self._ctx)
        if n < 0:
            raise RuntimeError("get_irreps failed: no symmetry set")
        irreps = []
        for k in range(n):
            buf = ctypes.create_string_buffer(16)
            _aedelsten_lib.get_irrep_name_c(self._ctx, k, buf, len(buf))
            irreps.append((buf.value.decode(), _aedelsten_lib.get_irrep_dimension_c(self._ctx, k)))
        return irreps

    def get_irrep_index(self, name: str) -> int | None:
        """The index of the irrep called `name` (case-insensitive), None if the group has none."""
        k = _aedelsten_lib.get_irrep_index_c(self._ctx, name.encode())
        return None if k < 0 else k

    def get_totally_symmetric_irrep(self) -> int:
        """The index of the totally symmetric irrep."""
        return _aedelsten_lib.get_totally_symmetric_irrep_c(self._ctx)

    def get_cartesian_irrep(self, lx: int, ly: int, lz: int) -> int | None:
        """The irrep of x^lx y^ly z^lz about the centre of mass; None if it is not a single one-dimensional irrep."""
        k = _aedelsten_lib.get_cartesian_irrep_c(self._ctx, lx, ly, lz)
        return None if k < 0 else k

    def get_symmetry_standard_axes(self) -> np.ndarray:
        """(3, 3): row k is the group's standard axis k in the input's coordinates (what the irrep names refer to)."""
        axes = np.zeros((3, 3), dtype=np.float64)
        if _aedelsten_lib.get_symmetry_standard_axes_c(self._ctx, axes.ctypes.data) != 0:
            raise RuntimeError("get_symmetry_standard_axes failed: no symmetry set")
        return axes

    def get_symmetry_operation_count(self) -> int:
        """The group's order h: operations are numbered 0..h in a fixed order. RuntimeError if no symmetry is set."""
        h = _aedelsten_lib.get_symmetry_operation_count_c(self._ctx)
        if h < 0:
            raise RuntimeError("get_symmetry_operation_count failed: no symmetry set")
        return h

    def get_symmetry_operation_matrix(self, op: int) -> np.ndarray:
        """(3, 3): operation op in the input's axes; a displacement or gradient v of a centre goes to m @ v at its image
        (get_symmetry_centre_images). Orthogonal."""
        matrix = np.zeros((3, 3), dtype=np.float64, order="F")
        if _aedelsten_lib.get_symmetry_operation_matrix_c(self._ctx, op, matrix.ctypes.data) != 0:
            raise RuntimeError(f"get_symmetry_operation_matrix failed: no symmetry set, or no operation {op}")
        return matrix

    def get_n_symmetry_centres(self) -> int:
        """The QM centres the group acts on (ghosts included; the gradient centres first)."""
        n = _aedelsten_lib.get_n_symmetry_centres_c(self._ctx)
        if n < 0:
            raise RuntimeError("get_n_symmetry_centres failed: no symmetry set")
        return n

    def get_symmetry_centre_images(self, op: int) -> np.ndarray:
        """images[a]: the centre (0-based) operation op takes centre a to."""
        n = self.get_n_symmetry_centres()
        images = np.zeros(n, dtype=np.int32)
        if _aedelsten_lib.get_symmetry_centre_images_c(self._ctx, op, images.ctypes.data, n) != 0:
            raise RuntimeError(f"get_symmetry_centre_images failed: no operation {op}")
        return images.astype(np.int64)

    def symmetrise_matrix(self, matrix: np.ndarray, average: bool = False) -> dict:
        """Cleans the totally symmetric (n_cont, n_cont) matrix in place (a Fock matrix, a density) and returns the
        report. average False: refused (RuntimeError, matrix untouched) above the strict tolerance; True: averaged
        over the group, whatever it removes. Any matrix, symmetric or not (the axis reversal of this interface commutes
        with the averaging)."""
        _check_f64_c_contiguous(matrix, "matrix")
        _check_square(matrix, "matrix")
        report = np.zeros(4, dtype=np.float64)
        result = _aedelsten_lib.symmetrise_matrix_c(self._ctx, matrix.ctypes.data, matrix.size, int(average), report.ctypes.data)
        if result != 0:
            raise RuntimeError(f"symmetrise_matrix failed (code={result}; the reason is on stderr)")
        return self._report(report)

    def analyse_matrix(self, matrix: np.ndarray, irrep: int | None = None) -> dict:
        """What cleaning matrix as irrep (None: whichever irrep carries most of it) would remove; nothing changed."""
        _check_f64_c_contiguous(matrix, "matrix")
        _check_square(matrix, "matrix")
        report = np.zeros(4, dtype=np.float64)
        result = _aedelsten_lib.analyse_matrix_c(self._ctx, matrix.ctypes.data, matrix.size, -1 if irrep is None else irrep, report.ctypes.data)
        if result != 0:
            raise RuntimeError(f"analyse_matrix failed (code={result})")
        return self._report(report)

    def irrep_norms(self, matrix: np.ndarray) -> np.ndarray:
        """The norm of each irrep's part of the (n_cont, n_cont) matrix, one per irrep."""
        _check_f64_c_contiguous(matrix, "matrix")
        _check_square(matrix, "matrix")
        n_irreps = _aedelsten_lib.get_n_irreps_c(self._ctx)
        norms = np.zeros(max(n_irreps, 0), dtype=np.float64)
        result = _aedelsten_lib.get_irrep_norms_c(self._ctx, matrix.ctypes.data, matrix.size, norms.ctypes.data, n_irreps)
        if result != 0:
            raise RuntimeError(f"irrep_norms failed (code={result})")
        return norms

    def symmetrise_orbitals(self, orbitals: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Symmetrizes MO coefficients in place: orbitals (n_mo, n_cont), one MO per row (any n_mo). Returns
        (irreps, components, dropped): each MO's irrep and partner component, and the largest coefficient norm
        projected away. Not re-orthonormalized."""
        _check_f64_c_contiguous(orbitals, "orbitals")
        if orbitals.ndim != 2 or orbitals.shape[1] != self.get_n_cont():
            raise ValueError(f"orbitals must be (n_mo, {self.get_n_cont()}), got {orbitals.shape}")
        n_mo = orbitals.shape[0]
        irreps, components = np.zeros(n_mo, dtype=np.int32), np.zeros(n_mo, dtype=np.int32)
        dropped = ctypes.c_double(0.0)
        result = _aedelsten_lib.symmetrise_orbitals_c(self._ctx, orbitals.ctypes.data, orbitals.size, n_mo,
                                                      irreps.ctypes.data, components.ctypes.data, ctypes.byref(dropped))
        if result != 0:
            raise RuntimeError(f"symmetrise_orbitals failed (code={result})")
        return irreps, components, dropped.value

    def get_ao_irrep_counts(self) -> np.ndarray:
        """How many AO functions span each irrep (irrep order)."""
        n_irreps = _aedelsten_lib.get_n_irreps_c(self._ctx)
        if n_irreps < 0:
            raise RuntimeError("get_ao_irrep_counts failed: no symmetry set")
        counts = np.zeros(n_irreps, dtype=np.int32)
        if _aedelsten_lib.get_ao_irrep_counts_c(self._ctx, counts.ctypes.data, n_irreps) != 0:
            raise RuntimeError("get_ao_irrep_counts failed")
        return counts.astype(np.int64)

    def apply_symmetry_operation(self, op: int, orbitals: np.ndarray) -> np.ndarray:
        """Operation op applied to MO coefficients, orbitals (n_mo, n_cont) one MO per row as in symmetrise_orbitals:
        the coefficients of each R phi, a new array of the same shape."""
        _check_f64_c_contiguous(orbitals, "orbitals")
        if orbitals.ndim != 2 or orbitals.shape[1] != self.get_n_cont():
            raise ValueError(f"orbitals must be (n_mo, {self.get_n_cont()}), got {orbitals.shape}")
        out = np.zeros_like(orbitals)
        result = _aedelsten_lib.apply_symmetry_operation_c(self._ctx, op, orbitals.ctypes.data, out.ctypes.data,
                                                           orbitals.size, orbitals.shape[0])
        if result != 0:
            raise RuntimeError(f"apply_symmetry_operation failed: no symmetry set, or no operation {op}")
        return out

    def get_irrep_character(self, irrep: int, op: int) -> float:
        """The character of irrep `irrep` at operation `op`."""
        character = ctypes.c_double(0.0)
        if _aedelsten_lib.get_irrep_character_c(self._ctx, irrep, op, ctypes.byref(character)) != 0:
            raise RuntimeError(f"get_irrep_character failed: no symmetry set, or no irrep {irrep} or operation {op}")
        return character.value

    def get_irrep_matrix(self, irrep: int, op: int) -> np.ndarray | None:
        """Gamma(R) of irrep `irrep` at operation `op`, (d, d) with g[j, k] = Gamma_jk (R f_k = sum_j Gamma_jk f_j),
        for the partner components symmetrise_orbitals aligns; None when the irrep has none (complex pairs; T, Th)."""
        d = _aedelsten_lib.get_irrep_dimension_c(self._ctx, irrep)
        if d < 0:
            raise RuntimeError(f"get_irrep_matrix failed: no symmetry set, or no irrep {irrep}")
        matrix = np.zeros((d, d), dtype=np.float64, order="F")
        aligned = ctypes.c_int(0)
        if _aedelsten_lib.get_irrep_matrix_c(self._ctx, irrep, op, matrix.ctypes.data, d * d, ctypes.byref(aligned)) != 0:
            raise RuntimeError(f"get_irrep_matrix failed: no operation {op}")
        return matrix if aligned.value else None

    def compute_centre_of_mass(self) -> np.ndarray:
        """The centre of mass of the real atoms (bohr), a 3-vector."""
        com = np.zeros(3, dtype=np.float64)
        _aedelsten_lib.compute_centre_of_mass_c(
            self._ctx,
            com[0:1].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            com[1:2].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            com[2:3].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return com

    def compute_nuclear_repulsion_energy(self) -> float:
        """The repulsion energy of the QM region's nuclei and core AIMPs."""
        return _aedelsten_lib.compute_nuclear_repulsion_energy_c(self._ctx)

    def compute_nuclear_charge(self) -> float:
        """The total charge of the QM region's nuclei and core AIMPs."""
        return _aedelsten_lib.compute_nuclear_charge_c(self._ctx)

    def compute_nuclear_multipole_interaction_energy(self) -> float:
        """Interaction energy between the nuclear potential and any loaded external multipoles."""
        return _aedelsten_lib.compute_nuclear_multipole_interaction_energy_c(self._ctx)

    def compute_nuclear_aimp_embedding_energy(self) -> float:
        """The energy of the QM region's nuclei and core AIMPs in the embedding AIMPs' screened potentials; 0 without
        embedding AIMPs."""
        return _aedelsten_lib.compute_nuclear_aimp_embedding_energy_c(self._ctx)

    # --- External multipoles / AIMP ---
    # Both just materialize operators reached afterwards the same way as
    # any other one-electron operator, via compute_one_electron_integrals
    # with the matching wire-key suffix ("external_multipole",
    # "core_aimp_coulomb", "core_aimp_exchange", "embedding_aimp_coulomb",
    # "embedding_aimp_exchange") -- core AIMP regions need no setup call,
    # they're already resident once the context exists.

    def set_external_multipoles(self, filename: str):
        """Reads the external point charges and multipoles of the xfield file `filename`."""
        result = _aedelsten_lib.set_external_multipoles_c(self._ctx, filename.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to set external multipoles (code={result})")

    def free_external_multipoles(self):
        """Frees the external multipoles."""
        _aedelsten_lib.free_external_multipoles_c(self._ctx)

    def has_external_multipoles(self) -> bool:
        """Whether external multipoles are set."""
        return bool(_aedelsten_lib.has_external_multipoles_c(self._ctx))

    # --- Induced-dipole (polarizable) embedding ---
    # Mirrors the Fortran interface (same names and options); eager where Fortran is not. Order:
    # set_polarizable_sites -> (set_external_multipoles) -> set_polarization -> compute_induction per Fock
    # build; compute_induction_response for CI/CC/response trial-vector batches (needs response_batch >= 1);
    # compute_induction_states for per-state dipoles (needs state_slots >= k). Layouts follow this module's
    # reversed-axes rule: (n, n) matrices, (k, n, n) batches, (n_sites, 3) dipoles. The induction operator
    # depends on a density only through its symmetric part and is itself symmetric, so the axis reversal is
    # invisible here.

    def set_polarizable_sites(self, filename: str):
        """Reads the polarizable sites of the xfield file `filename`."""
        result = _aedelsten_lib.set_polarizable_sites_c(self._ctx, filename.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to set polarizable sites (code={result})")

    def free_polarizable_sites(self):
        """Frees the polarizable sites and the polarization set up on them."""
        _aedelsten_lib.free_polarizable_sites_c(self._ctx)

    def get_n_polarizable_sites(self) -> int:
        """The number of polarizable sites."""
        n = _aedelsten_lib.get_n_polarizable_sites_c(self._ctx)
        if n < 0:
            raise RuntimeError("Could not count the polarizable sites (are they set?)")
        return int(n)

    def has_polarizable_sites(self) -> bool:
        """Whether polarizable sites are set."""
        return _aedelsten_lib.has_polarizable_sites_c(self._ctx) == 1

    def has_polarization(self) -> bool:
        """Whether set_polarization has set the polarization up."""
        return _aedelsten_lib.has_polarization_c(self._ctx) == 1

    # --- The electric field at sites ---
    # A selection is a name ("qm": the nuclei and core AIMPs, "embedding_aimps", "external_multipoles",
    # "polarizable") or an (n_points, 3) array of custom points (bohr). Fields are (n_sites, 3) arrays.

    @staticmethod
    def _selection(selection):
        if isinstance(selection, str):
            return getattr(_get_constants(), "site_selection_" + selection.lower()), None, None, 0
        points = np.ascontiguousarray(selection, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"custom sites must be an (n_points, 3) array, got {points.shape}")
        return _get_constants().site_selection_custom, points, points.ctypes.data, points.shape[0]

    def count_electric_field_sites(self, selection) -> int:
        """The number of sites of `selection`."""
        s, _keep, points, n_points = self._selection(selection)
        n = _aedelsten_lib.count_electric_field_sites_c(self._ctx, s, points, n_points)
        if n < 0:
            raise RuntimeError(f"count_electric_field_sites failed for {selection!r}")
        return int(n)

    def get_input_indices(self, selection) -> list[int | None]:
        """Each site's record in its input file (0-based), None for a site from none (nuclei, AIMPs, custom)."""
        s, _keep, points, n_points = self._selection(selection)
        n = self.count_electric_field_sites(selection)
        indices = np.zeros(n, dtype=np.int64)
        result = _aedelsten_lib.get_input_indices_c(self._ctx, s, points, n_points, indices.ctypes.data, n)
        if result != 0:
            raise RuntimeError(f"get_input_indices failed (code={result})")
        return [None if i < 0 else int(i) for i in indices]

    def compute_electric_field_integrals(self, selection, sites: range | None = None) -> np.ndarray:
        """The electric-field integrals of the sites `sites` (all when None), a (count, 3, n, n) array: per site its
        x, y and z planes."""
        s, _keep, points, n_points = self._selection(selection)
        if sites is None:
            sites = range(self.count_electric_field_sites(selection))
        if sites.step != 1:
            raise ValueError("compute_electric_field_integrals: sites must be a contiguous range")
        n = self.get_n_cont()
        output = np.zeros((len(sites), 3, n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_electric_field_integrals_c(
            self._ctx, s, points, n_points, sites.start, len(sites), output.ctypes.data, output.size)
        if result != 0:
            raise RuntimeError(f"compute_electric_field_integrals failed (code={result})")
        return output

    def compute_static_field(self, selection, nuclei: bool = True, core_aimps: bool = True,
                             embedding_aimps: bool = True, external_multipoles: bool = True) -> np.ndarray:
        """The classical field of the sources switched on at the sites of `selection`."""
        s, _keep, points, n_points = self._selection(selection)
        field = np.zeros((self.count_electric_field_sites(selection), 3), dtype=np.float64)
        result = _aedelsten_lib.compute_static_field_c(
            self._ctx, s, points, n_points, int(nuclei), int(core_aimps), int(embedding_aimps),
            int(external_multipoles), field.ctypes.data, field.size)
        if result != 0:
            raise RuntimeError(f"compute_static_field failed (code={result})")
        return field

    def compute_electronic_field(self, density: np.ndarray, selection, chunk_sites: int = -1,
                                 chunk_budget: int = -1, chunk_budget_unit: str = "MiB") -> np.ndarray:
        """The field of the plain density `density` at the sites of `selection`; chunked by `chunk_sites` sites when
        positive, else by `chunk_budget` (negative: the default)."""
        n = self.get_n_cont()
        density = np.ascontiguousarray(density, dtype=np.float64)
        if density.shape != (n, n):
            raise ValueError(f"compute_electronic_field: density must be ({n}, {n}), got {density.shape}")
        s, _keep, points, n_points = self._selection(selection)
        field = np.zeros((self.count_electric_field_sites(selection), 3), dtype=np.float64)
        result = _aedelsten_lib.compute_electronic_field_c(
            self._ctx, density.ctypes.data, s, points, n_points, chunk_sites, chunk_budget,
            _resolve_byte_unit(chunk_budget_unit), field.ctypes.data, field.size)
        if result != 0:
            raise RuntimeError(f"compute_electronic_field failed (code={result})")
        return field

    def set_multipole_convention(self, convention: str):
        """How the external multipoles meet the electrons: "contact_free" (the default) or "exact"."""
        key = getattr(_get_constants(), "multipole_convention_" + convention.lower())
        if _aedelsten_lib.set_multipole_convention_c(self._ctx, key) != 0:
            raise RuntimeError(f"set_multipole_convention failed for {convention!r}")

    def get_multipole_convention(self) -> str:
        """How the external multipoles meet the electrons: "contact_free" or "exact"."""
        c = _get_constants()
        names = {c.multipole_convention_contact_free: "contact_free", c.multipole_convention_exact: "exact"}
        return names[_aedelsten_lib.get_multipole_convention_c(self._ctx)]

    def set_polarization(self, options: PolarizationOptions | None = None, **kwargs):
        """Pass a PolarizationOptions, or its fields as keywords (or nothing, for the defaults)."""
        if options is not None and kwargs:
            raise TypeError("pass either a PolarizationOptions or keyword options, not both")
        options = options if options is not None else PolarizationOptions(**kwargs)
        result = _aedelsten_lib.set_polarization_c(self._ctx, ctypes.byref(options))
        if result != 0:
            raise RuntimeError(f"Failed to set up the polarization (code={result})")

    def free_polarization(self):
        """Frees the polarization state."""
        _aedelsten_lib.free_polarization_c(self._ctx)

    def compute_induction(self, density: np.ndarray, out: np.ndarray | None = None,
                          threshold: float | None = None) -> tuple[np.ndarray, float, int]:
        """One Fock build: returns (v_ind, energy, iterations). `v_ind` (n, n) is the operator to add to the
        one-electron Hamiltonian (written into `out` if given, else freshly allocated), `energy` is E_ind. With
        `threshold`, the dipoles are solved to it for this call in place of the setup's (compute_induction_within_c)."""
        n = self.get_n_cont()
        _check_f64_c_contiguous(density, "density")
        if density.shape != (n, n):
            raise ValueError(f"density must be ({n}, {n}), got {density.shape}")
        if out is None:
            out = np.zeros((n, n), dtype=np.float64)
        _check_f64_c_contiguous(out, "out")
        if out.shape != (n, n):
            raise ValueError(f"out must be ({n}, {n}), got {out.shape}")
        energy = ctypes.c_double(0.0)
        iterations = ctypes.c_int(0)
        if threshold is None:
            result = _aedelsten_lib.compute_induction_c(
                self._ctx, density.ctypes.data, out.ctypes.data, ctypes.byref(energy), ctypes.byref(iterations))
        else:
            result = _aedelsten_lib.compute_induction_within_c(
                self._ctx, density.ctypes.data, out.ctypes.data, ctypes.c_double(threshold), ctypes.byref(energy),
                ctypes.byref(iterations))
        if result != 0:
            raise RuntimeError(f"compute_induction failed (code={result})")
        return out, energy.value, iterations.value

    def get_polarization_storage(self) -> dict:
        """How this rank holds the field integrals of its share of the polarizable sites: a dict with "sites" (of the
        share), "in_memory" (the first sites kept there; the rest evaluated in every pass), "bytes_per_site" and
        "bytes_to_keep" (the resident_budget that keeps the whole share in memory)."""
        out = np.zeros(4, dtype=np.int64)
        result = _aedelsten_lib.get_polarization_storage_c(self._ctx, out.ctypes.data)
        if result != 0:
            raise RuntimeError(f"get_polarization_storage failed (code={result}); has set_polarization run?")
        return dict(zip(("sites", "in_memory", "bytes_per_site", "bytes_to_keep"), (int(v) for v in out)))

    def set_resident_field_budget(self, budget: int, unit: str):
        """The memory budget of this rank's kept field integrals, `budget` in `unit` ("gib", "mib" or "kib"; 0: none
        kept), in place of set_polarization's resident_budget: those of the first sites it holds are computed at the
        next pass and kept, the rest evaluated in every pass. Collective."""
        result = _aedelsten_lib.set_resident_field_budget_c(self._ctx, ctypes.c_int(budget), ctypes.c_int(_resolve_byte_unit(unit)))
        if result != 0:
            raise RuntimeError(f"set_resident_field_budget failed (code={result}); has set_polarization run?")

    def compute_induction_gradient(self, density: np.ndarray, active=None) -> np.ndarray:
        """dE_ind/dX at fixed `density` (n, n), a freshly allocated `(n_centres, 3)` array (row X = centre X), from the
        induced dipoles of the last compute_induction (call it with the same density first). `active` as in
        compute_one_electron_gradient."""
        n = self.get_n_cont()
        _check_f64_c_contiguous(density, "density")
        if density.shape != (n, n):
            raise ValueError(f"density must be ({n}, {n}), got {density.shape}")
        n_centres = self.get_n_gradient_centres()
        mask = None
        if active is not None:
            mask = np.ascontiguousarray(active, dtype=np.int32)
            if mask.shape != (n_centres,):
                raise ValueError(f"active must have {n_centres} flags, got shape {mask.shape}")
        gradient = np.zeros((n_centres, 3), dtype=np.float64)
        result = _aedelsten_lib.compute_induction_gradient_c(self._ctx, gradient.ctypes.data, ctypes.c_int(n_centres),
                                                             density.ctypes.data, None if mask is None else ctypes.c_void_p(mask.ctypes.data))
        if result != 0:
            raise RuntimeError(f"compute_induction_gradient failed (code={result})")
        return gradient

    def compute_induction_response(self, densities: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Linear response of the induction (no static field; ground-state dipoles untouched): V1[D1] for one
        density (n, n) or a batch (k, n, n) -> an array of the same shape."""
        n = self.get_n_cont()
        _check_f64_c_contiguous(densities, "densities")
        if densities.ndim not in (2, 3) or densities.shape[-2:] != (n, n):
            raise ValueError(f"densities must be ({n}, {n}) or (k, {n}, {n}), got {densities.shape}")
        if out is None:
            out = np.zeros_like(densities)
        _check_f64_c_contiguous(out, "out")
        if out.shape != densities.shape:
            raise ValueError(f"out must have the shape of densities {densities.shape}, got {out.shape}")
        k = densities.shape[0] if densities.ndim == 3 else 1
        result = _aedelsten_lib.compute_induction_response_c(self._ctx, densities.ctypes.data, k, out.ctypes.data)
        if result != 0:
            raise RuntimeError(f"compute_induction_response failed (code={result})")
        return out

    def compute_induction_couplings(self, densities: np.ndarray) -> np.ndarray:
        """The induction's couplings between (k, n, n) densities: K[m, n] = -f_m . (alpha^-1 - T)^-1 f_n, f the
        electrons' field at the sites (no static field), equal to tr(D_m V1[D_n]) without forming any operator; a
        (k, k) array (Fortran order, as the library writes it). Needs response_batch >= 1."""
        n = self.get_n_cont()
        _check_f64_c_contiguous(densities, "densities")
        if densities.ndim != 3 or densities.shape[1:] != (n, n):
            raise ValueError(f"densities must be (k, {n}, {n}), got {densities.shape}")
        k = densities.shape[0]
        out = np.zeros((k, k), dtype=np.float64, order="F")
        result = _aedelsten_lib.compute_induction_couplings_c(self._ctx, densities.ctypes.data, k, out.ctypes.data)
        if result != 0:
            raise RuntimeError(f"compute_induction_couplings failed (code={result})")
        return out

    def compute_effective_external_field(self) -> np.ndarray:
        """The effective external field (EEF) operators, (3, n, n) for x, y, z: the induced dipoles of a uniform unit
        external field coupled to the electrons, V^(i) = -sum_s mu_s^(i) . t_s; added to the dipole integrals r_i."""
        n = self.get_n_cont()
        out = np.zeros((3, n, n))
        result = _aedelsten_lib.compute_effective_external_field_c(self._ctx, out.ctypes.data)
        if result != 0:
            raise RuntimeError(f"compute_effective_external_field failed (code={result})")
        return out

    def compute_induction_states(self, densities: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(k, n, n) densities -> (operators (k, n, n), energies (k,)); each state has its own induced dipoles
        (kept between calls: keep the state order); weight the results yourself for a state average. With
        `threshold`, the dipoles are solved to it for this call in place of the setup's."""
        n = self.get_n_cont()
        _check_f64_c_contiguous(densities, "densities")
        if densities.ndim != 3 or densities.shape[1:] != (n, n):
            raise ValueError(f"densities must be (k, {n}, {n}), got {densities.shape}")
        k = densities.shape[0]
        operators = np.zeros_like(densities)
        energies = np.zeros(k, dtype=np.float64)
        if threshold is None:
            result = _aedelsten_lib.compute_induction_states_c(
                self._ctx, densities.ctypes.data, k, operators.ctypes.data, energies.ctypes.data)
        else:
            result = _aedelsten_lib.compute_induction_states_within_c(
                self._ctx, densities.ctypes.data, k, operators.ctypes.data, energies.ctypes.data,
                ctypes.c_double(threshold))
        if result != 0:
            raise RuntimeError(f"compute_induction_states failed (code={result})")
        return operators, energies

    def get_induced_dipoles(self) -> np.ndarray:
        """The induced dipoles of the last compute_induction, an (n_sites, 3) array in input order."""
        n_sites = self.get_n_polarizable_sites()
        mu = np.zeros((n_sites, 3), dtype=np.float64)
        result = _aedelsten_lib.get_induced_dipoles_c(self._ctx, mu.ctypes.data, 3 * n_sites)
        if result != 0:
            raise RuntimeError(f"get_induced_dipoles failed (code={result}); has compute_induction run?")
        return mu

    def has_core_aimp(self) -> bool:
        """Whether the QM region has core AIMPs."""
        return bool(_aedelsten_lib.has_core_aimp_c(self._ctx))

    def set_induced_dipoles(self, dipoles: np.ndarray):
        """The start of the next compute_induction's solve, an (n_sites, 3) array as get_induced_dipoles gives it
        (another geometry's dipoles). Not a solution until that solve."""
        n_sites = self.get_n_polarizable_sites()
        mu = np.ascontiguousarray(dipoles, dtype=np.float64)
        if mu.shape != (n_sites, 3):
            raise ValueError(f"set_induced_dipoles: dipoles must be ({n_sites}, 3), got {mu.shape}")
        result = _aedelsten_lib.set_induced_dipoles_c(self._ctx, mu.ctypes.data, 3 * n_sites)
        if result != 0:
            raise RuntimeError(f"set_induced_dipoles failed (code={result}); is the polarization set up?")

    def has_embedding_aimp(self) -> bool:
        """Whether the input has embedding AIMPs."""
        return bool(_aedelsten_lib.has_embedding_aimp_c(self._ctx))

    def get_n_embedding_aimps(self) -> int:
        """The number of embedding AIMPs."""
        n = _aedelsten_lib.get_n_embedding_aimps_c(self._ctx)
        if n < 0:
            raise RuntimeError("Could not count the embedding AIMPs")
        return int(n)

    def get_n_centres(self) -> int:
        """The number of centres: real atoms, ghost centres and orthogonality ghosts."""
        n = _aedelsten_lib.get_n_centres_c(self._ctx)
        if n < 0:
            raise RuntimeError("get_n_centres failed")
        return n

    def get_centre_geometry(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Every centre in centre order: atomic numbers, positions (an (n, 3) array, bohr) and kinds ("real",
        "ghost" or "orthogonality_ghost")."""
        n = self.get_n_centres()
        atomic_numbers = np.zeros(n, dtype=np.int32)
        positions = np.zeros((n, 3), dtype=np.float64)
        kinds = np.zeros(n, dtype=np.int32)
        result = _aedelsten_lib.get_centre_geometry_c(self._ctx, atomic_numbers.ctypes.data, positions.ctypes.data, kinds.ctypes.data, n)
        if result != 0:
            raise RuntimeError(f"get_centre_geometry failed (code={result})")
        c = _get_constants()
        names = {c.atom_kind_real: "real", c.atom_kind_ghost: "ghost", c.atom_kind_orthogonality_ghost: "orthogonality_ghost"}
        return atomic_numbers, positions, [names[int(k)] for k in kinds]

    def get_nuclear_charges(self) -> np.ndarray:
        """Every centre's nuclear charge (its nucleus's or its core AIMP's; 0 for a ghost centre or an orthogonality
        ghost)."""
        n = self.get_n_centres()
        charges = np.zeros(n, dtype=np.float64)
        result = _aedelsten_lib.get_nuclear_charges_c(self._ctx, charges.ctypes.data, n)
        if result != 0:
            raise RuntimeError(f"get_nuclear_charges failed (code={result})")
        return charges

    def get_embedding_aimp_sites(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every embedding AIMP's site in input order: atomic numbers (0 when its block's name is not an element
        symbol), positions (an (n, 3) array, bohr) and charges."""
        n = self.get_n_embedding_aimps()
        atomic_numbers = np.zeros(n, dtype=np.int32)
        positions = np.zeros((n, 3), dtype=np.float64)
        charges = np.zeros(n, dtype=np.float64)
        result = _aedelsten_lib.get_embedding_aimp_sites_c(
            self._ctx, atomic_numbers.ctypes.data, positions.ctypes.data, charges.ctypes.data, n)
        if result != 0:
            raise RuntimeError(f"get_embedding_aimp_sites failed (code={result})")
        return atomic_numbers, positions, charges

    def set_embedding_aimp_operators(self, filename: str | None = None):
        """Builds the embedding AIMPs' projection operators, needed by their projection and exchange operators;
        `filename` must be None (the AIMPs come from the input)."""
        filename_c = filename.encode('utf-8') if filename is not None else None
        result = _aedelsten_lib.set_embedding_aimp_operators_c(self._ctx, filename_c)
        if result != 0:
            raise RuntimeError(f"Failed to set embedding AIMP operators (filename={filename!r}, code={result})")

    def free_embedding_aimp_operators(self):
        """Frees the embedding AIMPs' projection operators."""
        _aedelsten_lib.free_embedding_aimp_operators_c(self._ctx)

    # --- One-electron integrals (eager) ---

    def compute_one_electron_integrals(
        self,
        operator: str,
        primitive: bool = False,
        origin: tuple[float, float, float] | None = None,
        order: int | None = None,
    ) -> np.ndarray:
        """Returns a freshly allocated `(n, n)` array, or for a component
        operator an `(n_planes, n, n)` array (n_planes leading). `n` is
        get_n_prim_unique() when primitive=True (requires
        set_primitive_shell_pairs() first), else get_n_cont(). `operator` is
        any wire-key suffix ("overlap", "kinetic", "nuclear_attraction",
        "core_aimp_coulomb", "embedding_aimp_exchange", "external_multipole",
        "moment", "angular_momentum", "nabla", "sap_helfem", "sap_grasp", ...)
        or the fully-qualified key ("oneint_overlap"). "sap_helfem" and
        "sap_grasp" are the SAP screening, each atom's electrons as Gaussian
        charges fitted to nonrelativistic or four-component atoms.

        The component operators:
        - "moment": the property moments <mu| (x-O_x)^a (y-O_y)^b (z-O_z)^c |nu>
          about `origin` for every Cartesian component of `order` (required;
          order 0 -> 1 plane, 1 -> 3, 2 -> 6, ...);
        - "angular_momentum" (about `origin`) and "nabla": `(3, n, n)`, planes
          x, y, z, of the real ANTISYMMETRIC
          `A_k(mu, nu) = <mu| ((r - O) x nabla)_k |nu>` (L = -i A) and
          `nabla_k(mu, nu) = <mu| d/dk |nu>` (p = -i nabla). As every plane here
          (see the top of this file), each comes back transposed, which for
          these is a sign: `out[k, mu, nu] = A_k(nu, mu) = -A_k(mu, nu)`.
        `origin` defaults to (0, 0, 0).
        """
        n = self.get_n_prim_unique() if primitive else self.get_n_cont()
        operator_int = _resolve_operator(operator)
        if operator_int == _resolve_operator("moment") and order is None:
            raise ValueError("the moment operator needs an order")
        order_c = order if order is not None else 0
        n_planes = _aedelsten_lib.compute_one_electron_component_count_c(ctypes.c_int(operator_int), ctypes.c_int(order_c))
        ox, oy, oz = origin if origin is not None else (0.0, 0.0, 0.0)
        shape = (n_planes, n, n) if n_planes > 0 else (n, n)
        output = np.zeros(shape, dtype=np.float64)

        result = _aedelsten_lib.compute_one_electron_integrals_c(
            self._ctx, output.ctypes.data, ctypes.c_int(n), ctypes.c_int(operator_int), ctypes.c_int(max(n_planes, 1)),
            ctypes.c_double(ox), ctypes.c_double(oy), ctypes.c_double(oz), ctypes.c_int(order_c), ctypes.c_int(int(primitive)),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute one-electron integrals (operator={operator!r}, code={result})")
        return output

    # --- X2C: picture-change integrals, Hamiltonian construction, contraction ---

    def compute_one_electron_picture_change_integrals(
        self,
        operator: str,
        spin_orbit: bool = False,
        origin: tuple[float, float, float] | None = None,
        order: int | None = None,
    ) -> np.ndarray:
        """Relativistic picture-change integrals, always over the
        decontracted (primitive) basis (n = get_n_prim_unique(), requires
        set_primitive_shell_pairs() first -- no separate primitive= toggle).
        Returns a freshly allocated `(n, n)` array, or an `(n_planes, n, n)`
        array whenever spin_orbit and/or the moment's `order` push the plane
        count above 1 (spin_orbit alone: 4 planes (x, y, z, scalar);
        combined with a moment order: 4x the order's own component count).
        `origin`/`order`: the "moment" operator's (as in
        compute_one_electron_integrals).
        """
        n = self.get_n_prim_unique()
        operator_int = _resolve_operator(operator)

        n_planes_component = (
            _aedelsten_lib.compute_moment_component_count_c(ctypes.c_int(order))
            if order is not None else 1
        )
        n_planes = n_planes_component * 4 if spin_orbit else n_planes_component
        order_c = order if order is not None else 0
        ox, oy, oz = origin if origin is not None else (0.0, 0.0, 0.0)

        output = np.zeros((n, n), dtype=np.float64) if n_planes == 1 else np.zeros((n_planes, n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_one_electron_picture_change_integrals_c(
            self._ctx, output.ctypes.data, ctypes.c_int(n), ctypes.c_int(operator_int), ctypes.c_int(n_planes),
            ctypes.c_double(ox), ctypes.c_double(oy), ctypes.c_double(oz), ctypes.c_int(order_c), ctypes.c_int(int(spin_orbit)),
        )
        if result != 0:
            raise RuntimeError(
                f"Failed to compute one-electron picture-change integrals (operator={operator!r}, code={result})"
            )
        return output

    def construct_x2c_one_electron_hamiltonian(
        self, s: np.ndarray, t: np.ndarray, v: np.ndarray, pvp: np.ndarray, c_light: float | None = None,
    ) -> np.ndarray:
        """Builds and returns the spin-free X2C one-electron Hamiltonian
        (an `(n, n)` array, n = get_n_prim_unique()) from primitive-basis
        overlap/kinetic/nuclear-attraction/pVp matrices (s, t, v from
        compute_one_electron_integrals(..., primitive=True), pvp from
        compute_one_electron_picture_change_integrals(..., spin_orbit=False)
        on the same operator), and caches the resulting picture-change
        transform for later use. c_light defaults to the library's own
        value. Errors if a transform is already cached: free_picture_change()
        first. With symmetry set, s/t/v/pvp must be symmetric (an unaveraged
        environment in v is refused) and the outputs come back averaged.
        """
        for arr, name in ((s, "s"), (t, "t"), (v, "v"), (pvp, "pvp")):
            _check_f64_c_contiguous(arr, name)
        n = _check_square(s, "s")
        if not (s.shape == t.shape == v.shape == pvp.shape):
            raise ValueError("s/t/v/pvp must all be square and the same shape")
        oneham = np.zeros((n, n), dtype=np.float64)
        c_light_wire = -1.0 if c_light is None else float(c_light)  # sentinel: use the library default
        result = _aedelsten_lib.construct_x2c_one_electron_hamiltonian_c(
            self._ctx, oneham.ctypes.data, s.ctypes.data, t.ctypes.data, v.ctypes.data, pvp.ctypes.data,
            ctypes.c_int(n), ctypes.c_double(c_light_wire),
        )
        if result != 0:
            raise RuntimeError(f"Failed to construct the X2C one-electron Hamiltonian (code={result})")
        return oneham

    def has_picture_change(self) -> bool:
        """Whether the X2C decoupling transforms are held."""
        return _aedelsten_lib.has_picture_change_c(self._ctx) == 1

    def get_picture_change_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """The cached X2C decoupling transforms (u_large, u_small), each (n_prim, n_prim) (the transposes of the
        column-major ones, see the file header): u_large for plain operators, u_small for their pXp counterparts."""
        n = self.get_n_prim_unique()
        u_large, u_small = np.zeros((n, n), dtype=np.float64), np.zeros((n, n), dtype=np.float64)
        result = _aedelsten_lib.get_picture_change_transforms_c(self._ctx, u_large.ctypes.data, u_small.ctypes.data, n * n)
        if result != 0:
            raise RuntimeError(f"get_picture_change_transforms failed (code={result})")
        return u_large, u_small

    def free_picture_change(self) -> None:
        """Frees the X2C decoupling transforms."""
        _aedelsten_lib.free_picture_change_c(self._ctx)

    def set_picture_change_transforms(self, u_large: np.ndarray, u_small: np.ndarray, c_light: float | None = None) -> None:
        """Supplies the decoupling transforms, as get_picture_change_transforms returns them, instead of deriving them:
        the next construct_x2c_one_electron_hamiltonian holds them (the X2C Hamiltonian with another geometry's
        decoupling). c_light as they were made with (None: the library default)."""
        n = self.get_n_prim_unique()
        for name, u in (("u_large", u_large), ("u_small", u_small)):
            _check_f64_c_contiguous(u, name)
            if u.shape != (n, n):
                raise ValueError(f"{name} must be ({n}, {n}), got {u.shape}")
        c_light_wire = -1.0 if c_light is None else float(c_light)
        result = _aedelsten_lib.set_picture_change_transforms_c(self._ctx, u_large.ctypes.data, u_small.ctypes.data, n * n,
                                                                ctypes.c_double(c_light_wire))
        if result != 0:
            raise RuntimeError(f"set_picture_change_transforms failed (code={result})")

    def set_x2c_frozen_density(self, density: np.ndarray) -> None:
        """Holds the densities of the X2C gradient with the decoupling held, from the symmetric `density` (n_cont,
        n_cont), for compute_x2c_frozen_kinetic_gradient and compute_x2c_frozen_potential_gradient; replaces densities
        already held. A new decoupling, free_picture_change or set_centre_positions drops them."""
        n = self.get_n_cont()
        _check_f64_c_contiguous(density, "density")
        if density.shape != (n, n):
            raise ValueError(f"density must be ({n}, {n}), got {density.shape}")
        result = _aedelsten_lib.set_x2c_frozen_density_c(self._ctx, density.ctypes.data, ctypes.c_int(n))
        if result != 0:
            raise RuntimeError(f"set_x2c_frozen_density failed (code={result})")

    def free_x2c_frozen_density(self) -> None:
        """Frees the densities set_x2c_frozen_density holds."""
        _aedelsten_lib.free_x2c_frozen_density_c(self._ctx)

    def _x2c_frozen_gradient_arguments(self, active):
        n_centres = self.get_n_gradient_centres()
        mask = None
        if active is not None:
            mask = np.ascontiguousarray(active, dtype=np.int32)
            if mask.shape != (n_centres,):
                raise ValueError(f"active must have {n_centres} flags, got shape {mask.shape}")
        return n_centres, mask, np.zeros((n_centres, 3), dtype=np.float64)

    def compute_x2c_frozen_kinetic_gradient(self, active=None) -> np.ndarray:
        """The kinetic part of the X2C one-electron energy tr(D h)'s gradient with the decoupling held, a freshly
        allocated `(n_centres, 3)` array (row X = centre X), from the densities of set_x2c_frozen_density. With
        compute_x2c_frozen_potential_gradient for each potential in the transformation, the sum is the whole gradient.
        `active` as in compute_one_electron_gradient."""
        n_centres, mask, gradient = self._x2c_frozen_gradient_arguments(active)
        result = _aedelsten_lib.compute_x2c_frozen_kinetic_gradient_c(
            self._ctx, gradient.ctypes.data, ctypes.c_int(n_centres), None if mask is None else ctypes.c_void_p(mask.ctypes.data))
        if result != 0:
            raise RuntimeError(f"compute_x2c_frozen_kinetic_gradient failed (code={result})")
        return gradient

    def compute_x2c_frozen_potential_gradient(self, operator: str, active=None) -> np.ndarray:
        """The part of the X2C one-electron energy's gradient with the decoupling held of one potential `operator`
        that went into V and p.Vp ("nuclear_attraction", "external_multipole", "embedding_aimp_coulomb", ...), laid out
        as compute_x2c_frozen_kinetic_gradient's; the caller sums the parts. `active` as in
        compute_one_electron_gradient."""
        n_centres, mask, gradient = self._x2c_frozen_gradient_arguments(active)
        result = _aedelsten_lib.compute_x2c_frozen_potential_gradient_c(
            self._ctx, gradient.ctypes.data, ctypes.c_int(n_centres), ctypes.c_int(_resolve_operator(operator)),
            None if mask is None else ctypes.c_void_p(mask.ctypes.data))
        if result != 0:
            raise RuntimeError(f"compute_x2c_frozen_potential_gradient failed (operator={operator!r}, code={result})")
        return gradient

    def transform_one_electron_operator(self, op_plain: np.ndarray, op_pxp: np.ndarray) -> np.ndarray:
        """A one-electron operator in the X2C picture, the scalar part: `u_large^T X u_large + u_small^T (p.X p)
        u_small` per plane, from `op_plain` (X) and `op_pxp` (p.X p) over the primitive basis, both `(n_prim, n_prim)`
        or `(n_planes, n_prim, n_prim)` (the raw planes of compute_one_electron_picture_change_integrals);
        symmetrised. Needs the X2C decoupling."""
        _check_f64_c_contiguous(op_plain, "op_plain")
        _check_f64_c_contiguous(op_pxp, "op_pxp")
        if op_plain.shape != op_pxp.shape or op_plain.ndim not in (2, 3) or op_plain.shape[-1] != op_plain.shape[-2]:
            raise ValueError(f"op_plain and op_pxp must both be (n, n) or (n_planes, n, n), got {op_plain.shape}, {op_pxp.shape}")
        n = op_plain.shape[-1]
        n_planes = 1 if op_plain.ndim == 2 else op_plain.shape[0]
        out = np.zeros_like(op_plain)
        result = _aedelsten_lib.transform_one_electron_operator_c(
            self._ctx, op_plain.ctypes.data, op_pxp.ctypes.data, out.ctypes.data, ctypes.c_int(n), ctypes.c_int(n_planes))
        if result != 0:
            raise RuntimeError(f"transform_one_electron_operator failed (code={result})")
        return out

    def transform_one_electron_spin_operator(self, op_pxp: np.ndarray) -> np.ndarray:
        """The spin planes of a one-electron operator in the X2C picture, `u_small^T X_k u_small` per plane of
        `op_pxp` `(n_planes, n_prim, n_prim)`: the raw `(p X x p)_k` planes of
        compute_one_electron_picture_change_integrals(..., spin_orbit=True), other raw spin planes such as
        compute_amfi_integrals' added first. No symmetrisation (the planes are antisymmetric). As every plane here (see
        the top of this file), input and output are transposed alike, so the sign convention carries through. Needs the
        X2C decoupling."""
        _check_f64_c_contiguous(op_pxp, "op_pxp")
        if op_pxp.ndim != 3 or op_pxp.shape[1] != op_pxp.shape[2]:
            raise ValueError("op_pxp must be (n_planes, n, n)")
        n_planes, n = op_pxp.shape[0], op_pxp.shape[1]
        out = np.zeros_like(op_pxp)
        result = _aedelsten_lib.transform_one_electron_spin_operator_c(
            self._ctx, op_pxp.ctypes.data, out.ctypes.data, ctypes.c_int(n), ctypes.c_int(n_planes))
        if result != 0:
            raise RuntimeError(f"transform_one_electron_spin_operator failed (code={result})")
        return out

    def compute_picture_changed_one_electron_operator(
        self,
        operator: str,
        contracted: bool = True,
        origin: tuple[float, float, float] | None = None,
        order: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """A one-electron operator X in the X2C picture, `X_2c = scalar (x) 1 + i sum_k sigma_k (x) spin_k`:
        `scalar` `(K, n, n)`, `u_large^T X_c u_large + u_small^T (p.X_c p) u_small`, and `spin` `(3K, n, n)`, plane
        `3c + k` from `u_small^T (p X_c x p)_k u_small`, over the contracted basis (`contracted`) or the primitive one.
        K = 1 for a plain operator, the moment's component count with `order`. The spin planes are antisymmetric and,
        as every plane here (see the top of this file), come back transposed, which for them is a sign:
        `spin[p, mu, nu] = -spin_p(mu, nu)`. Needs the X2C decoupling and set_primitive_shell_pairs()."""
        n = self.get_n_cont() if contracted else self.get_n_prim_unique()
        operator_int = _resolve_operator(operator)
        n_components = (_aedelsten_lib.compute_moment_component_count_c(ctypes.c_int(order)) if order is not None else 1)
        ox, oy, oz = origin if origin is not None else (0.0, 0.0, 0.0)
        scalar = np.zeros((n_components, n, n), dtype=np.float64)
        spin = np.zeros((3 * n_components, n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_picture_changed_one_electron_operator_c(
            self._ctx, scalar.ctypes.data, spin.ctypes.data, ctypes.c_int(n), ctypes.c_int(operator_int),
            ctypes.c_int(n_components), ctypes.c_double(ox), ctypes.c_double(oy), ctypes.c_double(oz),
            ctypes.c_int(order if order is not None else 0), ctypes.c_int(int(contracted)))
        if result != 0:
            raise RuntimeError(f"compute_picture_changed_one_electron_operator failed (operator={operator!r}, code={result})")
        return scalar, spin

    def compute_amfi_integrals(self) -> np.ndarray:
        """The atomic mean-field two-electron spin-orbit integrals over the primitive basis, `(6, n_prim, n_prim)`:
        spin-same-orbit x, y, z, then spin-other-orbit exchange x, y, z, raw (no prefactor; Breit-Pauli is
        `C^T (SSO + SOO) C / 4c^2`). Antisymmetric and, as every plane here (see the top of this file), transposed,
        which for them is a sign: `out[k, mu, nu] = -W_k(mu, nu)`. Needs set_primitive_shell_pairs() and the
        scratch."""
        n = self.get_n_prim_unique()
        out = np.zeros((6, n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_amfi_integrals_c(self._ctx, out.ctypes.data, ctypes.c_int(n))
        if result != 0:
            raise RuntimeError(f"compute_amfi_integrals failed (code={result})")
        return out

    def contract_one_electron_integrals(self, primitive: np.ndarray) -> np.ndarray:
        """Contracts a primitive-basis result (e.g. from
        compute_one_electron_integrals(..., primitive=True) or
        compute_one_electron_picture_change_integrals) into the contracted
        basis. Dispatches on primitive.ndim: 2 -> plain matrix, 3 ->
        leading-plane tensor. n_prim/n_cont come from primitive's own
        shape and get_n_cont() respectively; requires
        set_primitive_shell_pairs() first.
        """
        _check_f64_c_contiguous(primitive, "primitive")
        n_cont = self.get_n_cont()
        if primitive.ndim == 2:
            n_prim = _check_square(primitive, "primitive")
            output = np.zeros((n_cont, n_cont), dtype=np.float64)
            result = _aedelsten_lib.contract_one_electron_matrix_c(
                self._ctx, primitive.ctypes.data, ctypes.c_int(n_prim), output.ctypes.data, ctypes.c_int(n_cont),
            )
        elif primitive.ndim == 3:
            n_planes, n_prim, n_prim2 = primitive.shape
            if n_prim != n_prim2:
                raise ValueError(f"primitive must be square in its last two dimensions, got shape {primitive.shape}")
            output = np.zeros((n_planes, n_cont, n_cont), dtype=np.float64)
            result = _aedelsten_lib.contract_one_electron_array3d_c(
                self._ctx, primitive.ctypes.data, ctypes.c_int(n_prim), ctypes.c_int(n_planes),
                output.ctypes.data, ctypes.c_int(n_cont),
            )
        else:
            raise ValueError(f"primitive must be 2-D or 3-D, got ndim={primitive.ndim}")
        if result != 0:
            raise RuntimeError(f"Failed to contract one-electron integrals (code={result})")
        return output

    # --- Two-electron repulsion integrals (ERIs) ---

    def compute_s4_size(self) -> int:
        """The number of values of the S4 ERI tensor of this basis."""
        return _aedelsten_lib.compute_s4_size_c(self._ctx)

    def compute_s8_size(self) -> int:
        """The number of values of the S8 ERI tensor of this basis."""
        return _aedelsten_lib.compute_s8_size_c(self._ctx)

    def compute_eri_stored(
        self,
        ao_symmetry: str = "s1",
        ao_index_order: str = "column_major",
        eri_range: str | None = None,
        omega: float | None = None,
    ) -> np.ndarray:
        """Returns a freshly allocated dense (`ao_symmetry="s1"`, shape
        `(n, n, n, n)`) or symmetry-packed (`"s4"`, shape `(n_pairs, n_pairs)`;
        `"s8"`, shape `(size,)` flat triangular) array of two-electron
        repulsion integrals. `eri_range`/`omega` select a range-separated
        operator ("short_range"/"long_range" with the given omega); both
        or neither must be given (neither = full-range Coulomb).

        `ao_index_order` ("row_major"/"column_major") picks how a pair of
        AO indices packs into S4/S8's linear pair index -- it has no
        effect on S1 (dense, no packing). It is NOT a numpy memory-layout
        choice (every array here is always C-contiguous); it is an
        algorithmic choice about what a given linear index *means*, and
        the only way to correctly unpack an S4/S8 buffer's individual
        entries back into (i, j, k, l) later. "row_major" matches PySCF's
        aosym="s4"/"s8" convention.

        Under MPI the call is collective: every rank must make it; each
        computes its share of the integrals and every rank gets the whole
        tensor.
        """
        n_cont = self.get_n_cont()
        symmetry_int = _resolve_eri_symmetry(ao_symmetry)
        index_order_int = _resolve_eri_index_order(ao_index_order)
        range_wire, omega_wire = _resolve_eri_range(eri_range, omega)

        key = ao_symmetry.lower()
        if key in ("s1", "eri_ao_symmetry_s1"):
            output = np.zeros((n_cont, n_cont, n_cont, n_cont), dtype=np.float64)
        elif key in ("s4", "eri_ao_symmetry_s4"):
            # S4 keeps the bra pair (ij) and ket pair (kl) distinct, so
            # storage is the full (n_pairs, n_pairs) square, and (like S1)
            # is symmetric under transpose (<ij|kl> = <kl|ij> always) --
            # no ao_index_order caveat for reading it as a whole, only for
            # interpreting a given pair index back into (i, j).
            n_pairs = n_cont * (n_cont + 1) // 2
            output = np.zeros((n_pairs, n_pairs), dtype=np.float64)
        elif key in ("s8", "eri_ao_symmetry_s8"):
            output = np.zeros(self.compute_s8_size(), dtype=np.float64)
        else:
            raise ValueError(f"Unknown ao_symmetry '{ao_symmetry}'. Must be one of 's1', 's4', 's8'.")

        result = _aedelsten_lib.compute_eri_stored_c(
            self._ctx, output.ctypes.data, ctypes.c_int(n_cont), ctypes.c_int(symmetry_int),
            ctypes.c_int(index_order_int), ctypes.c_int(range_wire), ctypes.c_double(omega_wire),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute ERIs (ao_symmetry={ao_symmetry!r}, code={result})")
        return output

    # --- Direct Fock construction (no stored ERI tensor) ---
    # Context-bound (need the shell pairs and cached ERI data), unlike the
    # free J/K builders below: integrals are evaluated and immediately
    # folded into J/K one shell quartet at a time. Requires set_eri_data()
    # first. density here (and for every free J/K builder below) is
    # assumed to already be in this interface's own axis convention --
    # this interface does not transpose it for you, and (for a
    # genuinely symmetric density) there is nothing to do; if yours isn't
    # symmetric, transpose it yourself first.

    def set_eri_data(self, schwarz_threshold: float | None = None):
        """Derives the ERI data of the direct walks from the shell pairs, dropping the pairs that cannot reach
        `schwarz_threshold` (None: none dropped). Needs set_shell_pairs and the scratch."""
        threshold_wire = -1.0 if schwarz_threshold is None else float(schwarz_threshold)  # sentinel: library default
        result = _aedelsten_lib.set_eri_data_c(self._ctx, ctypes.c_double(threshold_wire))
        if result != 0:
            raise RuntimeError(f"Failed to set ERI data (code={result})")

    def free_eri_data(self):
        """Frees the ERI data."""
        _aedelsten_lib.free_eri_data_c(self._ctx)

    def compute_fock_direct(
        self,
        density: np.ndarray,
        j_coeff: float,
        k_coeff: float,
        eri_range: str | None = None,
        omega: float | None = None,
        pairwise_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Restricted direct Fock construction: `j = j_coeff * J(density)`,
        `k = k_coeff * K(density)`, with no dense ERI tensor ever
        materialized. Requires set_eri_data() first.
        """
        _check_f64_c_contiguous(density, "density")
        n = _check_square(density, "density")
        range_wire, omega_wire = _resolve_eri_range(eri_range, omega)
        pairwise_threshold_wire = -1.0 if pairwise_threshold is None else float(pairwise_threshold)
        j = np.zeros((n, n), dtype=np.float64)
        k = np.zeros((n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_fock_direct_restricted_c(
            self._ctx, j.ctypes.data, k.ctypes.data, density.ctypes.data, ctypes.c_int(n),
            ctypes.c_double(j_coeff), ctypes.c_double(k_coeff), ctypes.c_int(range_wire),
            ctypes.c_double(omega_wire), ctypes.c_double(pairwise_threshold_wire),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the direct Fock matrix (code={result})")
        return j, k

    def compute_fock_direct_general(
        self,
        density: np.ndarray,
        j_coeff: float,
        k_coeff: float,
        eri_range: str | None = None,
        omega: float | None = None,
        pairwise_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """compute_fock_direct for a density of any symmetry (a transition density, the spin-flip density of a
        stability analysis): `j = j_coeff * J(density)`, from the density's symmetric part, and
        `k = k_coeff * K(density)` with `K[a, c] = sum_bd density[b, d] (ab|cd)`, which is not symmetric for a
        non-symmetric density and is not symmetrized. Unlike compute_fock_direct, a non-symmetric density needs no
        transposing here: the library sees `density.T` (row-major against its column-major), computes
        `K[density.T] = K[density].T`, and that comes back transposed again, so `k` is `K[density]` in this
        interface's own index order. Requires set_eri_data() first.
        """
        _check_f64_c_contiguous(density, "density")
        n = _check_square(density, "density")
        range_wire, omega_wire = _resolve_eri_range(eri_range, omega)
        pairwise_threshold_wire = -1.0 if pairwise_threshold is None else float(pairwise_threshold)
        j = np.zeros((n, n), dtype=np.float64)
        k = np.zeros((n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_fock_direct_general_c(
            self._ctx, j.ctypes.data, k.ctypes.data, density.ctypes.data, ctypes.c_int(n),
            ctypes.c_double(j_coeff), ctypes.c_double(k_coeff), ctypes.c_int(range_wire),
            ctypes.c_double(omega_wire), ctypes.c_double(pairwise_threshold_wire),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the direct Fock matrices of a general density (code={result})")
        return j, k

    def compute_fock_direct_general_multi(
        self,
        densities: np.ndarray,
        j_coeff: float,
        k_coeff: float,
        eri_range: str | None = None,
        omega: float | None = None,
        pairwise_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """compute_fock_direct_general for several densities in one direct walk: `densities` is `(n_mat, n, n)`,
        and `j[m]`, `k[m]` are the J and K of `densities[m]` as compute_fock_direct_general gives them (to within the
        screening, which keeps an integral batch any density needs). Requires set_eri_data() first.
        """
        _check_f64_c_contiguous(densities, "densities")
        if densities.ndim != 3 or densities.shape[1] != densities.shape[2] or densities.shape[0] < 1:
            raise ValueError(f"densities must be (n_mat, n, n) with n_mat >= 1, got {densities.shape}")
        n_mat, n = densities.shape[0], densities.shape[1]
        range_wire, omega_wire = _resolve_eri_range(eri_range, omega)
        pairwise_threshold_wire = -1.0 if pairwise_threshold is None else float(pairwise_threshold)
        j = np.zeros((n_mat, n, n), dtype=np.float64)
        k = np.zeros((n_mat, n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_fock_direct_general_multi_c(
            self._ctx, j.ctypes.data, k.ctypes.data, densities.ctypes.data, ctypes.c_int(n), ctypes.c_int(n_mat),
            ctypes.c_double(j_coeff), ctypes.c_double(k_coeff), ctypes.c_int(range_wire),
            ctypes.c_double(omega_wire), ctypes.c_double(pairwise_threshold_wire),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the direct Fock matrices of general densities (code={result})")
        return j, k

    def compute_fock_direct_mixed_active_space(
        self,
        density: np.ndarray,
        c_bra: np.ndarray,
        c_ket: np.ndarray,
        j_coeff: float,
        k_coeff: float,
        active_threshold: float | None = None,
        pairwise_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The integrals of a state interaction between two runs with different orbitals, in one direct walk (full
        range): `j`, `k` of `density` as in compute_fock_direct_general, and the Coulomb matrices of the active pair
        densities `M(vw, mu nu) = sum_{lambda sigma} P[lambda sigma, vw] (mu nu|lambda sigma)`,
        `P[lambda sigma, vw] = (c_bra[v, lambda] c_ket[w, sigma] + c_bra[v, sigma] c_ket[w, lambda]) / 2`, for every
        pair `(v, w)`.

        `c_bra` and `c_ket` are `(n_active, n)`, one active orbital per row (so no copy: that is the library's
        column-major `n x n_active`). `m` comes back `(n (n + 1)/2, n_active, n_active)`, packed over the function
        pairs `mu >= nu` (row `nu n - nu (nu - 1)/2 + mu - nu`), with `m[row, w, v] = M(vw, mu nu)`.
        transform_pair_coulomb_one_index(m.reshape(-1, n_active**2), c_ket) gives `t[mu, w, v, u]` after a reshape to
        `(n, n_active, n_active, n_active)`, and the caller finishes `(tu|vw) = np.einsum("tm,mwvu->tuvw", c_bra, t)`
        (`t`, `v` the bra's orbitals, `u`, `w` the ket's). `active_threshold`: a batch whose Schwarz bound times its
        largest active coefficients is below it adds nothing to `M`; None = every batch. Requires set_eri_data() first.
        """
        for arr, name in ((density, "density"), (c_bra, "c_bra"), (c_ket, "c_ket")):
            _check_f64_c_contiguous(arr, name)
        n = _check_square(density, "density")
        if c_bra.ndim != 2 or c_bra.shape != c_ket.shape or c_ket.shape[1] != n:
            raise ValueError(f"c_bra and c_ket must both be (n_active, {n}), got {c_bra.shape} and {c_ket.shape}")
        n_active = c_ket.shape[0]
        threshold = lambda x: -1.0 if x is None else float(x)
        j = np.zeros((n, n), dtype=np.float64)
        k = np.zeros((n, n), dtype=np.float64)
        m = np.zeros((n * (n + 1) // 2, n_active, n_active), dtype=np.float64)
        result = _aedelsten_lib.compute_fock_direct_mixed_active_space_c(
            self._ctx, j.ctypes.data, k.ctypes.data, m.ctypes.data, density.ctypes.data, c_bra.ctypes.data,
            c_ket.ctypes.data, ctypes.c_int(n), ctypes.c_int(n_active), ctypes.c_double(j_coeff),
            ctypes.c_double(k_coeff), ctypes.c_double(threshold(active_threshold)),
            ctypes.c_double(threshold(pairwise_threshold)),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the mixed-basis active-space integrals (code={result})")
        return j, k, m

    def compute_fock_direct_unrestricted(
        self,
        density_alpha: np.ndarray,
        density_beta: np.ndarray,
        density_total: np.ndarray,
        j_coeff: float,
        k_coeff: float,
        eri_range: str | None = None,
        omega: float | None = None,
        pairwise_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unrestricted direct Fock construction. `density_total` is J's
        own density -- the caller's to supply (same requirement as
        compute_j_and_k_unrestricted_matrix's density_total below), not
        derived here. Requires set_eri_data() first.
        """
        for arr, name in ((density_alpha, "density_alpha"), (density_beta, "density_beta"), (density_total, "density_total")):
            _check_f64_c_contiguous(arr, name)
        n = _check_square(density_alpha, "density_alpha")
        if not (density_alpha.shape == density_beta.shape == density_total.shape):
            raise ValueError("density_alpha/density_beta/density_total must all be the same shape")
        range_wire, omega_wire = _resolve_eri_range(eri_range, omega)
        pairwise_threshold_wire = -1.0 if pairwise_threshold is None else float(pairwise_threshold)
        j = np.zeros((n, n), dtype=np.float64)
        k_alpha = np.zeros((n, n), dtype=np.float64)
        k_beta = np.zeros((n, n), dtype=np.float64)
        result = _aedelsten_lib.compute_fock_direct_unrestricted_c(
            self._ctx, j.ctypes.data, k_alpha.ctypes.data, k_beta.ctypes.data, density_alpha.ctypes.data,
            density_beta.ctypes.data, density_total.ctypes.data, ctypes.c_int(n), ctypes.c_double(j_coeff),
            ctypes.c_double(k_coeff), ctypes.c_int(range_wire), ctypes.c_double(omega_wire),
            ctypes.c_double(pairwise_threshold_wire),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the unrestricted direct Fock matrices (code={result})")
        return j, k_alpha, k_beta

    def compute_fock_direct_active_space(
        self,
        density_inactive: np.ndarray,
        density_active: np.ndarray,
        c_active: np.ndarray,
        coeff_j_inactive: float,
        coeff_k_inactive: float,
        coeff_j_active: float,
        coeff_k_active: float,
        active_threshold: float | None = None,
        pairwise_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The active-space integrals of a CASSCF macro-iteration in one direct walk (full range): the inactive and
        active J/K, each against its own coefficients as in compute_fock_direct, and the Coulomb matrices of the active
        pair densities `M(vw, mu nu) = sum_{lambda sigma} P[lambda sigma, vw] (mu nu|lambda sigma)`,
        `P[lambda sigma, vw] = (c_active[v, lambda] c_active[w, sigma] + c_active[v, sigma] c_active[w, lambda]) / 2`,
        for the pairs `v >= w` at `vw = v (v + 1)/2 + w`.

        `c_active` is `(n_active, n)`, one active orbital per row (so no copy: that is the library's column-major
        `n x n_active`). `m` comes back `(n (n + 1)/2, n_active (n_active + 1)/2)`, packed over the function pairs
        `mu >= nu` (row `nu n - nu (nu - 1)/2 + mu - nu`), with `m[row, vw] = M(vw, mu nu)`.
        transform_pair_coulomb_one_index(m, c_active) gives `t[mu, vw, u] = T(u, vw, mu)`, and the caller finishes
        `(pu|vw) = np.einsum("pm,mxu->pux", c, t)` for orbitals `c` (rows). The densities are symmetric.
        `active_threshold`: a batch whose Schwarz bound times its largest active coefficients is below it adds nothing to
        `M`; None = every batch. Returns `(j_inactive, k_inactive, j_active, k_active, m)`. Requires set_eri_data()
        first.
        """
        for arr, name in ((density_inactive, "density_inactive"), (density_active, "density_active"), (c_active, "c_active")):
            _check_f64_c_contiguous(arr, name)
        n = _check_square(density_inactive, "density_inactive")
        if density_active.shape != density_inactive.shape or c_active.ndim != 2 or c_active.shape[1] != n:
            raise ValueError(f"the densities must be ({n}, {n}) and c_active (n_active, {n}), got {density_active.shape}, {c_active.shape}")
        n_active = c_active.shape[0]
        threshold = lambda x: -1.0 if x is None else float(x)
        j_inactive, k_inactive, j_active, k_active = (np.zeros((n, n), dtype=np.float64) for _ in range(4))
        m = np.zeros((n * (n + 1) // 2, n_active * (n_active + 1) // 2), dtype=np.float64)
        result = _aedelsten_lib.compute_fock_direct_active_space_c(
            self._ctx, j_inactive.ctypes.data, k_inactive.ctypes.data, j_active.ctypes.data, k_active.ctypes.data,
            m.ctypes.data, density_inactive.ctypes.data, density_active.ctypes.data, c_active.ctypes.data,
            ctypes.c_int(n), ctypes.c_int(n_active), ctypes.c_double(coeff_j_inactive), ctypes.c_double(coeff_k_inactive),
            ctypes.c_double(coeff_j_active), ctypes.c_double(coeff_k_active), ctypes.c_double(threshold(active_threshold)),
            ctypes.c_double(threshold(pairwise_threshold)),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the active-space integrals (code={result})")
        return j_inactive, k_inactive, j_active, k_active, m

    def transform_pair_coulomb_one_index(self, m: np.ndarray, c: np.ndarray) -> np.ndarray:
        """One index of the active walks' `M` transformed: `T(u, vw, mu) = sum_nu c[u, nu] M(vw, mu nu)`. `m` is
        `(n (n + 1)/2, n_pairs)` as compute_fock_direct_active_space returns it (or the mixed walk's, reshaped to two
        dimensions); `c` is `(n_columns, n)`, one orbital per row. `t` comes back `(n, n_pairs, n_columns)` with
        `t[mu, vw, u] = T(u, vw, mu)`. Rank-local.
        """
        for arr, name in ((m, "m"), (c, "c")):
            _check_f64_c_contiguous(arr, name)
        if m.ndim != 2 or c.ndim != 2:
            raise ValueError(f"m must be (n (n + 1)/2, n_pairs) and c (n_columns, n), got {m.shape} and {c.shape}")
        n = c.shape[1]
        if m.shape[0] != n * (n + 1) // 2:
            raise ValueError(f"m must have {n * (n + 1) // 2} rows for {n} functions, got {m.shape[0]}")
        n_pairs, n_columns = m.shape[1], c.shape[0]
        t = np.zeros((n, n_pairs, n_columns), dtype=np.float64)
        result = _aedelsten_lib.transform_pair_coulomb_one_index_c(
            self._ctx, t.ctypes.data, m.ctypes.data, c.ctypes.data, ctypes.c_int(n), ctypes.c_int(n_pairs),
            ctypes.c_int(n_columns),
        )
        if result != 0:
            raise RuntimeError(f"Failed to transform the pair densities' Coulomb matrices (code={result})")
        return t

    def compute_eri_half_transformed(
        self,
        c_left: np.ndarray,
        c_right: np.ndarray,
        threshold: float | None = None,
        pairwise_threshold: float | None = None,
        partition_over_ranks: bool = True,
    ) -> np.ndarray:
        """The half-transformed ERIs from one direct walk (full range, no J/K): `X(p, nu, r, sigma) = sum_{mu lambda}
        c_left[p, mu] c_right[r, lambda] (mu nu|lambda sigma)`, two indices in the orbitals, two in the AO basis.

        `c_left` is `(m_left, n)` and `c_right` `(m_right, n)`, one orbital per row (so no copy: the library's
        column-major `n x m`). Passing the same array for both takes the symmetric path, `X(p, nu, r, sigma) =
        X(r, sigma, p, nu)`, at half the work (equal but distinct arrays take the general path). `x` comes back
        `(n, m_right, n, m_left)` with `x[sigma, r, nu, p] = X(p, nu, r, sigma)`. `threshold`: a batch whose Schwarz bound
        times the largest |c_left| on one of its shell pairs times the largest |c_right| on the other is below it is
        skipped; None = every batch. `partition_over_ranks`: under MPI, True makes the call collective (every rank gets
        the whole X), False computes X on this rank alone. The scratch holds the walk's row buffers besides its
        usual scratch. Requires set_eri_data() first.
        """
        for arr, name in ((c_left, "c_left"), (c_right, "c_right")):
            _check_f64_c_contiguous(arr, name)
        n = self.get_n_cont()
        if c_left.ndim != 2 or c_right.ndim != 2 or c_left.shape[1] != n or c_right.shape[1] != n:
            raise ValueError(f"c_left and c_right must be (m, {n}), got {c_left.shape} and {c_right.shape}")
        m_left, m_right = c_left.shape[0], c_right.shape[0]
        wire = lambda x: -1.0 if x is None else float(x)
        x = np.zeros((n, m_right, n, m_left), dtype=np.float64)
        result = _aedelsten_lib.compute_eri_half_transformed_c(
            self._ctx, x.ctypes.data, c_left.ctypes.data, ctypes.c_int(m_left), c_right.ctypes.data,
            ctypes.c_int(m_right), ctypes.c_int(n), ctypes.c_double(wire(threshold)),
            ctypes.c_double(wire(pairwise_threshold)), ctypes.c_int(1 if partition_over_ranks else 0),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the half-transformed integrals (code={result})")
        return x

    def compute_eri_half_transformed_on_node(
        self,
        c_left: np.ndarray,
        c_right: np.ndarray,
        x: np.ndarray,
        threshold: float | None = None,
        pairwise_threshold: float | None = None,
    ) -> np.ndarray:
        """`compute_eri_half_transformed` by the ranks of the context's communicator together, into one X in their
        node's shared memory: `x` is an `(n, m_right, n, m_left)` view of a `SharedArray` of the same communicator
        (`aedelsten_mpi_shared_array`), the ranks must share one node, and every rank has the whole X when it returns.
        Collective. Returns `x`."""
        for arr, name in ((c_left, "c_left"), (c_right, "c_right"), (x, "x")):
            _check_f64_c_contiguous(arr, name)
        n = self.get_n_cont()
        if c_left.ndim != 2 or c_right.ndim != 2 or c_left.shape[1] != n or c_right.shape[1] != n:
            raise ValueError(f"c_left and c_right must be (m, {n}), got {c_left.shape} and {c_right.shape}")
        m_left, m_right = c_left.shape[0], c_right.shape[0]
        if x.shape != (n, m_right, n, m_left):
            raise ValueError(f"x must be {(n, m_right, n, m_left)}, got {x.shape}")
        wire = lambda v: -1.0 if v is None else float(v)
        result = _aedelsten_lib.compute_eri_half_transformed_on_node_c(
            self._ctx, x.ctypes.data, c_left.ctypes.data, ctypes.c_int(m_left), c_right.ctypes.data,
            ctypes.c_int(m_right), ctypes.c_int(n), ctypes.c_double(wire(threshold)), ctypes.c_double(wire(pairwise_threshold)),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the half-transformed integrals on the node (code={result})")
        return x

    def compute_eri_half_transformed_sigma(
        self,
        c_left: np.ndarray,
        c_right: np.ndarray,
        sigma_start: int,
        sigma_end: int,
        threshold: float | None = None,
        pairwise_threshold: float | None = None,
        partition_over_ranks: bool = True,
    ) -> np.ndarray:
        """`compute_eri_half_transformed` for a batch of sigma: X(p, nu, r, sigma) for the sigma in
        `sigma_start:sigma_end` only (0-based AOs, a run of whole shells, from one shell's first AO, `get_shells`, to
        another's or n; not empty), `(n_sigma, m_right, n, m_left)` with `x[sigma - sigma_start, r, nu, p]`. The walk
        computes only the quartets with a shell in the batch, so batches that cover the basis compute each quartet at
        most four times in all. The other arguments as `compute_eri_half_transformed`; the scratch beyond the walk's
        own: `half_transform_scratch_bytes`."""
        for arr, name in ((c_left, "c_left"), (c_right, "c_right")):
            _check_f64_c_contiguous(arr, name)
        n = self.get_n_cont()
        if c_left.ndim != 2 or c_right.ndim != 2 or c_left.shape[1] != n or c_right.shape[1] != n:
            raise ValueError(f"c_left and c_right must be (m, {n}), got {c_left.shape} and {c_right.shape}")
        if not 0 <= sigma_start < sigma_end <= n:
            raise ValueError(f"sigma {sigma_start}:{sigma_end} is not a run of the {n} AOs")
        m_left, m_right = c_left.shape[0], c_right.shape[0]
        wire = lambda v: -1.0 if v is None else float(v)
        x = np.zeros((sigma_end - sigma_start, m_right, n, m_left), dtype=np.float64)
        result = _aedelsten_lib.compute_eri_half_transformed_sigma_c(
            self._ctx, x.ctypes.data, c_left.ctypes.data, ctypes.c_int(m_left), c_right.ctypes.data,
            ctypes.c_int(m_right), ctypes.c_int(n), ctypes.c_int(sigma_start), ctypes.c_int(sigma_end),
            ctypes.c_double(wire(threshold)), ctypes.c_double(wire(pairwise_threshold)),
            ctypes.c_int(1 if partition_over_ranks else 0),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the half-transformed integrals of sigma {sigma_start}:{sigma_end} (code={result})")
        return x

    def compute_eri_half_transformed_sigma_on_node(
        self,
        c_left: np.ndarray,
        c_right: np.ndarray,
        sigma_start: int,
        sigma_end: int,
        x: np.ndarray,
        threshold: float | None = None,
        pairwise_threshold: float | None = None,
    ) -> np.ndarray:
        """`compute_eri_half_transformed_sigma` by the ranks of the context's communicator together, into one X in
        their node's shared memory, as `compute_eri_half_transformed_on_node`: `x` an `(n_sigma, m_right, n, m_left)`
        view of a `SharedArray` of the same communicator. Collective. Returns `x`."""
        for arr, name in ((c_left, "c_left"), (c_right, "c_right"), (x, "x")):
            _check_f64_c_contiguous(arr, name)
        n = self.get_n_cont()
        if c_left.ndim != 2 or c_right.ndim != 2 or c_left.shape[1] != n or c_right.shape[1] != n:
            raise ValueError(f"c_left and c_right must be (m, {n}), got {c_left.shape} and {c_right.shape}")
        if not 0 <= sigma_start < sigma_end <= n:
            raise ValueError(f"sigma {sigma_start}:{sigma_end} is not a run of the {n} AOs")
        m_left, m_right = c_left.shape[0], c_right.shape[0]
        if x.shape != (sigma_end - sigma_start, m_right, n, m_left):
            raise ValueError(f"x must be {(sigma_end - sigma_start, m_right, n, m_left)}, got {x.shape}")
        wire = lambda v: -1.0 if v is None else float(v)
        result = _aedelsten_lib.compute_eri_half_transformed_sigma_on_node_c(
            self._ctx, x.ctypes.data, c_left.ctypes.data, ctypes.c_int(m_left), c_right.ctypes.data,
            ctypes.c_int(m_right), ctypes.c_int(n), ctypes.c_int(sigma_start), ctypes.c_int(sigma_end),
            ctypes.c_double(wire(threshold)), ctypes.c_double(wire(pairwise_threshold)),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the half-transformed integrals of sigma {sigma_start}:{sigma_end} on the node (code={result})")
        return x

    def half_transform_scratch_bytes(self, m_left: int, m_right: int, one_basis: bool, n_sigma: int, node_ranks: int = 1) -> int:
        """The scratch a half transform takes besides the walk's own, in bytes (set_scratch holds both):
        `m_left` and `m_right` orbitals, `one_basis` (the same array on both sides) or two, `n_sigma` AOs of sigma (n
        for `compute_eri_half_transformed`), `node_ranks` ranks sharing X (the `_on_node` forms; 1 for the others)."""
        result = _aedelsten_lib.half_transform_scratch_bytes_c(
            self._ctx, ctypes.c_int(m_left), ctypes.c_int(m_right), ctypes.c_int(1 if one_basis else 0),
            ctypes.c_int(n_sigma), ctypes.c_int(node_ranks),
        )
        if result < 0:
            raise ValueError(f"half_transform_scratch_bytes: bad arguments ({m_left}, {m_right}, {n_sigma}, {node_ranks})")
        return int(result)

    def scratch_bytes(self, operations, primitive: bool = False) -> int:
        """The scratch in bytes that every operation fits in, from the input alone (before set_shell_pairs; size
        set_scratch with it). Operations as (name, *parameters), a bare name for none: "eri_data", "restricted_fock",
        "unrestricted_fock", ("two_electron_gradient", terms), "general_fock", ("general_multi_fock", densities),
        ("active_space_fock", active), ("mixed_active_space_fock", active), ("half_transform", left, right, sigma,
        node_ranks) with right and sigma None for one basis and all functions, "amfi" (over the primitive shells
        whatever `primitive` says). `primitive`: on the primitive blocks."""
        rows = _scratch_operation_rows(operations)
        result = _aedelsten_lib.scratch_bytes_c(self._ctx, rows.ctypes.data, len(operations), int(primitive))
        if result < 0:
            raise RuntimeError(f"scratch_bytes failed for {operations}")
        return int(result)

    def scratch_breakdown(self, operations, primitive: bool = False) -> list[dict]:
        """scratch_bytes per operation, in order, with what sets it: a dict of "bytes" = "buffers" + "tables" +
        "row" + "item_bytes", "item" ("eri_data_entry", "gradient_pair_data", "derivative_bound", "quartet" or
        "amfi_atom_kind") and "shells", the item's (l, primitives, contracted): two for a pair's item, four for a
        quartet, none for an atom kind."""
        fields = _aedelsten_lib.scratch_breakdown_field_count_c()
        names = [_aedelsten_lib.scratch_breakdown_field_name_c(i).decode() for i in range(fields)]
        rows = _scratch_operation_rows(operations)
        output = np.zeros((max(len(operations), 1), fields), dtype=np.int64)
        result = _aedelsten_lib.scratch_breakdown_c(
            self._ctx, rows.ctypes.data, len(operations), int(primitive), output.ctypes.data, fields * len(operations),
        )
        if result != 0:
            raise RuntimeError(f"scratch_breakdown failed for {operations} (code={result})")
        breakdown = []
        for row in output[:len(operations)]:
            need = {name: int(value) for name, value in zip(names[:6], row[:6])}
            need["item"] = _SCRATCH_ITEMS[need["item"]]
            need["shells"] = [tuple(int(v) for v in row[k:k + 3]) for k in range(6, fields, 3) if row[k] >= 0]
            breakdown.append(need)
        return breakdown

    def compute_two_electron_gradient(
        self,
        densities: np.ndarray,
        terms,
        eri_range: str | None = None,
        omega: float | None = None,
        pairwise_threshold: float | None = None,
        active=None,
    ) -> np.ndarray:
        """The two-electron energy gradient `dE_2e/dX_c` for every gradient centre, a freshly allocated
        `(n_centres, 3)` array, row X = the x, y, z gradient of centre X (the (3, n_centres) column-major matrix of
        the C interface): the contracted twin of the direct J/K builds. `densities` is `(n_densities, n, n)`
        (C-contiguous float64); each term is a tuple `(kind, left, right, coeff)` with `kind` "coulomb"
        (`W = L[a,b] R[c,d]`) or "exchange" (`W = L[a,c] R[b,d]`) and `left`, `right` 0-based indices into the
        first axis of `densities`; `E_2e = 1/2 sum_terms coeff sum_abcd W (ab|cd)`. Restricted HF: `[P]`,
        `[("coulomb", 0, 0, 1.0), ("exchange", 0, 0, -0.5)]`; unrestricted `[P, P_a, P_b]`,
        `[("coulomb", 0, 0, 1.0), ("exchange", 1, 1, -1.0), ("exchange", 2, 2, -1.0)]`. (The C side reads
        each row-major density as its transpose; transposing every density leaves every term unchanged by the
        permutational symmetry of the integrals, so non-symmetric densities are fine.) Only the integral
        derivatives (the Pulay term comes with the overlap gradient); one range operator per call; `active` as in
        `compute_one_electron_gradient`. Requires set_eri_data() first."""
        n = self.get_n_cont()
        n_centres = self.get_n_gradient_centres()
        n_densities, kinds, lefts, rights, coeffs = self._two_electron_terms(densities, terms)
        mask = None
        if active is not None:
            mask = np.ascontiguousarray(active, dtype=np.int32)
            if mask.shape != (n_centres,):
                raise ValueError(f"active must have {n_centres} flags, got shape {mask.shape}")
        range_wire, omega_wire = _resolve_eri_range(eri_range, omega)
        threshold_wire = -1.0 if pairwise_threshold is None else float(pairwise_threshold)
        gradient = np.zeros((n_centres, 3), dtype=np.float64)
        result = _aedelsten_lib.compute_two_electron_gradient_c(
            self._ctx, gradient.ctypes.data, ctypes.c_int(n_centres), densities.ctypes.data, ctypes.c_int(n),
            ctypes.c_int(n_densities), ctypes.c_int(len(terms)), kinds.ctypes.data, lefts.ctypes.data,
            rights.ctypes.data, coeffs.ctypes.data, ctypes.c_int(range_wire), ctypes.c_double(omega_wire),
            ctypes.c_double(threshold_wire), None if mask is None else ctypes.c_void_p(mask.ctypes.data),
        )
        if result != 0:
            raise RuntimeError(f"Failed to compute the two-electron gradient (code={result})")
        return gradient

    def _two_electron_terms(self, densities: np.ndarray, terms):
        """The densities checked and the terms as the C arrays: (n_densities, kinds, lefts, rights, coeffs)."""
        n = self.get_n_cont()
        _check_f64_c_contiguous(densities, "densities")
        if densities.ndim != 3 or densities.shape[1:] != (n, n):
            raise ValueError(f"densities must be (n_densities, {n}, {n}), got {densities.shape}")
        n_densities = densities.shape[0]
        kinds = np.empty(len(terms), dtype=np.int32)
        lefts = np.empty(len(terms), dtype=np.int32)
        rights = np.empty(len(terms), dtype=np.int32)
        coeffs = np.empty(len(terms), dtype=np.float64)
        for t, (kind, left, right, coeff) in enumerate(terms):
            kind_code = {"coulomb": 0, "exchange": 1}.get(str(kind).lower())
            if kind_code is None:
                raise ValueError(f"unknown two-electron term kind {kind!r} (coulomb or exchange)")
            if not (0 <= left < n_densities and 0 <= right < n_densities):
                raise ValueError(f"term {t}: density index outside 0..{n_densities - 1}")
            kinds[t], lefts[t], rights[t], coeffs[t] = kind_code, left, right, coeff
        return n_densities, kinds, lefts, rights, coeffs


# --- J/K from a dense in-memory ERI tensor (free functions) ---
# None of these take a context: each is purely a function of its own
# arguments (density and eri_dense just need to agree on shape), exactly
# like the Fortran/Julia interfaces' free-function design for these.
# eri_dense must be dense S1 (n, n, n, n) -- there is no S4/S8-aware
# variant of these at the Rust level. If you built eri_dense yourself by
# unpacking an S4/S8 buffer from compute_eri_stored, using the wrong
# ao_index_order there will silently produce a wrong dense tensor here --
# this interface has no way to detect that from the tensor alone.

def _check_density_and_eri(density: np.ndarray, eri_dense: np.ndarray) -> int:
    _check_f64_c_contiguous(density, "density")
    _check_f64_c_contiguous(eri_dense, "eri_dense")
    n = _check_square(density, "density")
    if eri_dense.shape != (n, n, n, n):
        raise ValueError(f"eri_dense must have shape {(n, n, n, n)} (matching density's n={n}), got {eri_dense.shape}")
    return n

def compute_j_and_k_matrix(density: np.ndarray, eri_dense: np.ndarray, j_coeff: float, k_coeff: float) -> tuple[np.ndarray, np.ndarray]:
    """Builds the Coulomb (J) and exchange (K) matrices from a dense
    in-memory ERI tensor: `j = j_coeff * J(density)`, `k = k_coeff * K(density)`.
    """
    n = _check_density_and_eri(density, eri_dense)
    j = np.zeros((n, n), dtype=np.float64)
    k = np.zeros((n, n), dtype=np.float64)
    result = _aedelsten_lib.compute_j_and_k_matrix_stored_c(
        j.ctypes.data, k.ctypes.data, density.ctypes.data, eri_dense.ctypes.data, ctypes.c_int(n),
        ctypes.c_double(j_coeff), ctypes.c_double(k_coeff),
    )
    if result != 0:
        raise RuntimeError(f"Failed to construct the J and K matrices (code={result})")
    return j, k

def compute_j_and_k_unrestricted_matrix(
    density_alpha: np.ndarray, density_beta: np.ndarray, density_total: np.ndarray, eri_dense: np.ndarray,
    j_coeff: float, k_coeff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single pass over eri_dense for J, K_alpha and K_beta together --
    prefer this over separate compute_j_matrix + compute_k_matrix_unrestricted
    calls when all three are wanted. density_total is J's own density
    (alpha + beta, the caller's to compute), same requirement as
    compute_k_matrix_unrestricted's density_alpha/density_beta split.
    """
    for arr, name in ((density_alpha, "density_alpha"), (density_beta, "density_beta"), (density_total, "density_total")):
        _check_f64_c_contiguous(arr, name)
    n = _check_density_and_eri(density_alpha, eri_dense)
    if not (density_alpha.shape == density_beta.shape == density_total.shape):
        raise ValueError("density_alpha/density_beta/density_total must all be the same shape")
    j = np.zeros((n, n), dtype=np.float64)
    k_alpha = np.zeros((n, n), dtype=np.float64)
    k_beta = np.zeros((n, n), dtype=np.float64)
    result = _aedelsten_lib.compute_j_and_k_unrestricted_matrix_stored_c(
        j.ctypes.data, k_alpha.ctypes.data, k_beta.ctypes.data, density_alpha.ctypes.data, density_beta.ctypes.data,
        density_total.ctypes.data, eri_dense.ctypes.data, ctypes.c_int(n), ctypes.c_double(j_coeff), ctypes.c_double(k_coeff),
    )
    if result != 0:
        raise RuntimeError(f"Failed to construct the unrestricted J and K matrices (code={result})")
    return j, k_alpha, k_beta

def compute_j_matrix(density: np.ndarray, eri_dense: np.ndarray, coeff: float) -> np.ndarray:
    """`coeff * J(density)` from the dense (n, n, n, n) ERI tensor `eri_dense`."""
    n = _check_density_and_eri(density, eri_dense)
    j = np.zeros((n, n), dtype=np.float64)
    result = _aedelsten_lib.compute_j_matrix_stored_c(
        j.ctypes.data, density.ctypes.data, eri_dense.ctypes.data, ctypes.c_int(n), ctypes.c_double(coeff),
    )
    if result != 0:
        raise RuntimeError(f"Failed to construct the J matrix (code={result})")
    return j

def compute_k_matrix(density: np.ndarray, eri_dense: np.ndarray, coeff: float) -> np.ndarray:
    """`coeff * K(density)` from the dense (n, n, n, n) ERI tensor `eri_dense`."""
    n = _check_density_and_eri(density, eri_dense)
    k = np.zeros((n, n), dtype=np.float64)
    result = _aedelsten_lib.compute_k_matrix_stored_c(
        k.ctypes.data, density.ctypes.data, eri_dense.ctypes.data, ctypes.c_int(n), ctypes.c_double(coeff),
    )
    if result != 0:
        raise RuntimeError(f"Failed to construct the K matrix (code={result})")
    return k

def compute_k_matrix_unrestricted(
    density_alpha: np.ndarray, density_beta: np.ndarray, eri_dense: np.ndarray, coeff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`(coeff * K(density_alpha), coeff * K(density_beta))` from the dense (n, n, n, n) ERI tensor `eri_dense`, in one
    pass."""
    _check_f64_c_contiguous(density_beta, "density_beta")
    n = _check_density_and_eri(density_alpha, eri_dense)
    if density_alpha.shape != density_beta.shape:
        raise ValueError("density_alpha/density_beta must be the same shape")
    k_alpha = np.zeros((n, n), dtype=np.float64)
    k_beta = np.zeros((n, n), dtype=np.float64)
    result = _aedelsten_lib.compute_k_matrix_unrestricted_stored_c(
        k_alpha.ctypes.data, k_beta.ctypes.data, density_alpha.ctypes.data, density_beta.ctypes.data,
        eri_dense.ctypes.data, ctypes.c_int(n), ctypes.c_double(coeff),
    )
    if result != 0:
        raise RuntimeError(f"Failed to construct the unrestricted K matrices (code={result})")
    return k_alpha, k_beta


# --- MPI ---
# For a program without MPI of its own. A program that has MPI (mpi4py) initializes it itself and passes its
# communicators by their Fortran handle, `comm.py2f()`; None means MPI_COMM_WORLD.

def aedelsten_mpi_init(required_level: str = "funneled") -> int:
    """Initializes MPI at thread level `required_level` ("single", "funneled", "serialized", "multiple") and returns
    the level provided (0-3). Refused when MPI is initialized already by someone else, or provides less."""
    lib = load_aedelsten_lib()
    provided = lib.aedelsten_mpi_init_c(getattr(_get_constants(), "mpi_thread_" + required_level.lower()))
    if provided < 0:
        raise RuntimeError(f"MPI initialization at thread level '{required_level}' failed")
    return provided

def aedelsten_mpi_finalize() -> None:
    """Finalizes MPI if aedelsten_mpi_init initialized it; MPI initialized by the host is left alone."""
    load_aedelsten_lib().aedelsten_mpi_finalize_c()

def aedelsten_mpi_initialized() -> bool:
    """Whether MPI is initialized and not finalized (always False without the library's `mpi` build option)."""
    return load_aedelsten_lib().aedelsten_mpi_initialized_c() == 1

def aedelsten_mpi_thread_level() -> int:
    """The thread level MPI runs at (0-3), or -1 while it is not up."""
    return load_aedelsten_lib().aedelsten_mpi_thread_level_c()

def _communicator_handle(comm: "int | None") -> int:
    """The handle a context takes for `comm`: this process alone for None."""
    return _get_constants().mpi_comm_single if comm is None else comm

class SharedArray:
    """`len` doubles in the shared memory of the ranks of communicator `comm` (its Fortran handle; None: this process
    alone), which must share one node: `array` is a numpy view of them, the same memory on every rank. Made
    collectively; `sync()` makes every rank's writes visible to all; freed by `free()`, collectively (every rank at
    the same point: not left to the garbage collector)."""

    def __init__(self, length: int, comm: "int | None" = None):
        lib = load_aedelsten_lib()
        data = ctypes.POINTER(ctypes.c_double)()
        self._handle = lib.aedelsten_mpi_shared_array_c(ctypes.c_int(_communicator_handle(comm)), ctypes.c_int64(length), ctypes.byref(data))
        if not self._handle:
            raise RuntimeError(f"Failed to make a shared array of {length} doubles")
        self.array = np.ctypeslib.as_array(data, shape=(length,)) if length > 0 else np.zeros(0)

    def sync(self):
        """Makes every rank's writes visible to all, every rank waiting for the others. Collective."""
        if load_aedelsten_lib().aedelsten_mpi_shared_array_sync_c(self._handle) != 0:
            raise RuntimeError("Failed to synchronize a shared array")

    def free(self):
        """Frees the shared memory. Collective."""
        if getattr(self, "_handle", None):
            self.array = None
            load_aedelsten_lib().aedelsten_mpi_shared_array_free_c(self._handle)
            self._handle = None

def _communicator_query(function, comm: "int | None", what: str) -> int:
    handle = _get_constants().mpi_comm_world if comm is None else comm
    out = ctypes.c_int(0)
    if function(ctypes.c_int(handle), ctypes.byref(out)) != 0:
        raise RuntimeError(f"Failed to query the {what} of communicator {handle}")
    return out.value

def aedelsten_mpi_rank(comm: "int | None" = None) -> int:
    """This rank's index in communicator `comm` (its Fortran handle, e.g. mpi4py's `comm.py2f()`; None: the world)."""
    return _communicator_query(load_aedelsten_lib().aedelsten_mpi_rank_c, comm, "rank")

def aedelsten_mpi_size(comm: "int | None" = None) -> int:
    """The number of ranks in communicator `comm` (as in aedelsten_mpi_rank)."""
    return _communicator_query(load_aedelsten_lib().aedelsten_mpi_size_c, comm, "size")


# --- BLAS threads ---
# The process's count, set from the main thread only, outside any threaded region.

def aedelsten_blas_set_threads(k: int) -> int:
    """Sets the BLAS's thread count to k and returns the previous count. Refused (RuntimeError, count unchanged) when
    the BLAS cannot run k threads."""
    previous = load_aedelsten_lib().aedelsten_blas_set_threads_c(k)
    if previous < 0:
        raise RuntimeError(f"The BLAS cannot run {k} threads ({aedelsten_blas_info()})")
    return previous

def aedelsten_get_version() -> str:
    """The library's version, "<major>.<minor>.<patch>" (a release is tagged v<version>)."""
    buf = ctypes.create_string_buffer(64)
    if load_aedelsten_lib().aedelsten_get_version_c(buf, len(buf)) != 0:
        raise RuntimeError("Failed to read the library's version")
    return buf.value.decode()

def aedelsten_get_source_commit() -> "str | None":
    """The git commit the library was built from (12 hex digits), None when its source tree was no git checkout."""
    buf = ctypes.create_string_buffer(64)
    if load_aedelsten_lib().aedelsten_get_source_commit_c(buf, len(buf)) != 0:
        raise RuntimeError("Failed to read the library's source commit")
    return buf.value.decode() or None

def aedelsten_blas_threads() -> int:
    """The BLAS's thread count now."""
    return load_aedelsten_lib().aedelsten_blas_threads_c()

def aedelsten_blas_info() -> str:
    """The BLAS, its threading and its thread count, as one line (e.g. "OpenBLAS (OpenMP build), 1 thread")."""
    buf = ctypes.create_string_buffer(256)
    if load_aedelsten_lib().aedelsten_blas_info_c(buf, len(buf)) != 0:
        raise RuntimeError("Failed to describe the BLAS")
    return buf.value.decode()

# --- Timings and the reference list ---
# Read anew at each call: the coarse timings are always recorded, the fine ones only in a `profile` build; both are
# this thread's. The references are the process's: everything cited so far, in the order first cited.

def aedelsten_mpi_collective_seconds() -> float:
    """Seconds this process has spent in MPI collectives since it started (0 without MPI or on one rank)."""
    return load_aedelsten_lib().aedelsten_mpi_collective_seconds_c()

def aedelsten_profile_enabled() -> bool:
    """Whether the library was built with the `profile` feature (aedelsten_profile_report can return anything)."""
    return bool(load_aedelsten_lib().aedelsten_profile_enabled_c())

def _timing_spans(fine: int) -> "list[tuple[str, float, int]]":
    lib = load_aedelsten_lib()
    out = []
    for i in range(lib.aedelsten_timing_count_c(fine)):
        label, seconds, calls = ctypes.create_string_buffer(256), ctypes.c_double(0.0), ctypes.c_int64(0)
        if lib.aedelsten_timing_span_c(fine, i, label, len(label), ctypes.byref(seconds), ctypes.byref(calls)) != 0:
            raise RuntimeError(f"Failed to read timing entry {i}")
        out.append((label.value.decode(), seconds.value, calls.value))
    return out

def aedelsten_timing_report() -> "list[tuple[str, float, int]]":
    """The coarse timings so far, (label, seconds, calls) per label, in the order the labels were first seen."""
    return _timing_spans(0)

def aedelsten_profile_report() -> "list[tuple[str, float, int]]":
    """The fine timings so far, as aedelsten_timing_report; empty without the `profile` feature."""
    return _timing_spans(1)

def aedelsten_span_tree() -> "list[tuple[str, int, bool, float, int]]":
    """The spans as a tree, depth first (each node followed by its children): (label, depth from 0, fine, seconds with
    the children included, calls) per node."""
    lib = load_aedelsten_lib()
    out = []
    for i in range(lib.aedelsten_span_tree_count_c()):
        label, depth, fine = ctypes.create_string_buffer(256), ctypes.c_int(0), ctypes.c_int(0)
        seconds, calls = ctypes.c_double(0.0), ctypes.c_int64(0)
        if lib.aedelsten_span_tree_node_c(i, label, len(label), ctypes.byref(depth), ctypes.byref(fine),
                                          ctypes.byref(seconds), ctypes.byref(calls)) != 0:
            raise RuntimeError(f"Failed to read span tree node {i}")
        out.append((label.value.decode(), depth.value, bool(fine.value), seconds.value, calls.value))
    return out

def _citation_part(index: int, topics: int) -> str:
    lib = load_aedelsten_lib()
    n = lib.aedelsten_citation_length_c(index, topics)
    buf = ctypes.create_string_buffer(max(n, 0) + 1)
    if n < 0 or lib.aedelsten_citation_c(index, topics, buf, len(buf)) != 0:
        raise RuntimeError(f"Failed to read reference {index}")
    return buf.raw[:n].decode()

def aedelsten_citations() -> "list[tuple[str, str]]":
    """The references cited so far, (reference, topics) each: the formatted reference (authors, title, journal, volume,
    pages, year, DOI) and the topics it was cited for, joined by ", "."""
    return [(_citation_part(i, 0), _citation_part(i, 1)) for i in range(load_aedelsten_lib().aedelsten_citation_count_c())]

def aedelsten_citations_bibtex() -> str:
    """The references cited so far as a BibTeX file (one entry each, the topics as keywords)."""
    lib = load_aedelsten_lib()
    n = lib.aedelsten_citations_bibtex_length_c()
    buf = ctypes.create_string_buffer(n + 1)
    if lib.aedelsten_citations_bibtex_c(buf, len(buf)) != 0:
        raise RuntimeError("Failed to write the references as BibTeX")
    return buf.raw[:n].decode()

def aedelsten_clear_citations():
    """Forgets every reference cited so far (a host running several independent jobs in one process)."""
    load_aedelsten_lib().aedelsten_clear_citations_c()

def aedelsten_blas_available_cores() -> "tuple[int, bool, str]":
    """The cores available to this process: their count, whether a launcher bound the process to them (else they are
    the node's, shared by its ranks), and a line for a run's header (e.g. "4 performance cores, node wide")."""
    lib = load_aedelsten_lib()
    count, bound = ctypes.c_int(0), ctypes.c_int(0)
    buf = ctypes.create_string_buffer(256)
    if lib.aedelsten_blas_available_cores_c(ctypes.byref(count), ctypes.byref(bound)) != 0 \
            or lib.aedelsten_blas_available_cores_info_c(buf, len(buf)) != 0:
        raise RuntimeError("Failed to count the available cores")
    return count.value, bool(bound.value), buf.value.decode()
