// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The electric-field integrals of polarizable sites kept in memory between calls and their two
//! contractions: the electronic field of densities at the kept sites, and its transpose, the one-electron operator of
//! dipoles there. The size of the pieces of sites a field pass hands out (`SiteChunking`).

mod chunker;
mod site_chunking;

pub use chunker::FieldChunker;
pub use site_chunking::{
    DEFAULT_CHUNK_BUDGET_BYTES, SiteChunking, SiteChunkingError, field_integral_bytes_per_site,
    resolve_field_chunk_size,
};

/// An error of the kept integrals; `E` is the error of the caller's integral evaluation.
#[derive(Debug)]
pub enum EfieldError<E> {
    /// A buffer of `bytes` could not be allocated.
    Allocation { bytes: usize },
    /// The error of the caller-supplied integral evaluation.
    Integrals(E),
}

impl<E: std::fmt::Display> std::fmt::Display for EfieldError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation { bytes } => write!(f, "could not allocate {bytes} bytes for the electric-field integrals"),
            Self::Integrals(e) => write!(f, "{e}"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for EfieldError<E> {}
