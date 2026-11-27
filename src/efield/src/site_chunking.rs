// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use arrays::scratch::ByteUnit;

/// Budget used when the caller asks for a memory budget but gives no size
/// (the `None` of `SiteChunking::Budget`).
pub const DEFAULT_CHUNK_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// How many sites a field pass hands out as one piece (with a run of shell pairs): a number, or as many as a buffer of
/// their dense electric-field integrals (`n_ao * n_ao * 3` doubles a site) of a size would hold.
pub enum SiteChunking {
    /// A fixed number of sites per piece.
    Sites(usize),
    /// The size of that buffer; the number of sites per piece is derived from it.
    /// `None` uses `DEFAULT_CHUNK_BUDGET_BYTES`.
    Budget(Option<(usize, ByteUnit)>),
}

/// Why a `SiteChunking` could not be resolved.
#[derive(Debug, PartialEq)]
pub enum SiteChunkingError {
    /// A fixed chunk of zero sites.
    ZeroSites,
    /// The budget in bytes overflows `usize`.
    BudgetOverflow,
    /// Even a single site does not fit in the requested budget.
    BudgetTooSmall { required_bytes: usize, budget_bytes: usize },
}

impl std::fmt::Display for SiteChunkingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSites => write!(f, "number of sites per chunk must be at least 1"),
            Self::BudgetOverflow => write!(f, "memory budget does not fit in a usize"),
            Self::BudgetTooSmall { required_bytes, budget_bytes } => write!(
                f,
                "memory budget of {budget_bytes} bytes cannot hold the integrals of a single site ({required_bytes} bytes needed)"
            ),
        }
    }
}

impl std::error::Error for SiteChunkingError {}

/// Bytes of integral buffer needed per site: three planes of `n_ao * n_ao` doubles.
pub fn field_integral_bytes_per_site(n_ao: usize) -> usize {
    3 * n_ao * n_ao * std::mem::size_of::<f64>()
}

/// Resolve a `SiteChunking` to a concrete number of sites per chunk,
/// clamped to `n_sites` (and at least 1 for a non-empty list).
pub fn resolve_field_chunk_size(chunking: &SiteChunking, n_ao: usize, n_sites: usize) -> Result<usize, SiteChunkingError> {
    let per_site = field_integral_bytes_per_site(n_ao).max(1);
    let chunk = match chunking {
        SiteChunking::Sites(0) => return Err(SiteChunkingError::ZeroSites),
        SiteChunking::Sites(n) => *n,
        SiteChunking::Budget(budget) => {
            let budget_bytes = match budget {
                Some((size, unit)) => size.checked_mul(unit.bytes_per_unit()).ok_or(SiteChunkingError::BudgetOverflow)?,
                None => DEFAULT_CHUNK_BUDGET_BYTES,
            };
            if budget_bytes < per_site {
                return Err(SiteChunkingError::BudgetTooSmall { required_bytes: per_site, budget_bytes });
            }
            budget_bytes / per_site
        }
    };
    Ok(chunk.min(n_sites.max(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_sites_are_clamped_to_the_site_count() {
        assert_eq!(resolve_field_chunk_size(&SiteChunking::Sites(50), 10, 8), Ok(8));
        assert_eq!(resolve_field_chunk_size(&SiteChunking::Sites(0), 10, 8), Err(SiteChunkingError::ZeroSites));
    }

    #[test]
    fn budget_is_reverse_engineered_into_sites() {
        // n_ao = 10 -> 3 * 100 * 8 = 2400 bytes per site; 1 KB * 12 = 12288 bytes -> 5 sites.
        let chunking = SiteChunking::Budget(Some((12, ByteUnit::KiB)));
        assert_eq!(resolve_field_chunk_size(&chunking, 10, 1000), Ok(5));
    }

    #[test]
    fn budget_too_small_for_one_site_is_an_error() {
        let chunking = SiteChunking::Budget(Some((1, ByteUnit::KiB)));
        assert_eq!(
            resolve_field_chunk_size(&chunking, 10, 1000),
            Err(SiteChunkingError::BudgetTooSmall { required_bytes: 2400, budget_bytes: 1024 })
        );
    }
}
