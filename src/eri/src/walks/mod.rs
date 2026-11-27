// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The walks over shell-pair quartets: the stored tensor's (`in_memory`) and the direct Fock build's (`direct`) over
//! the data of `eri_data`.

mod direct;
mod eri_data;
mod in_memory;

pub(crate) use direct::{compute_fock_direct_from_data, direct_walk_need};
pub use eri_data::{EriData, EriDataEntry};
pub(crate) use eri_data::{derive_eri_data, pair_entry_buffers, walk_pieces, Handout, RowPiece};
pub(crate) use in_memory::compute_eri_stored;
