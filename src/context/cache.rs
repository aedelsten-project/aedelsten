// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

/// A derived piece of the context's state: not made (or freed), made and then invalidated (with the reason, naming the
/// setter that rebuilds it), or ready.
#[derive(Debug)]
pub(crate) enum AedelstenCacheState<T> {
    Empty,
    Invalid(InvalidationReason),
    Ready(T),
}

impl<T> AedelstenCacheState<T> {
    /// Whether it is ready.
    pub(crate) fn is_ready(&self) -> bool {
        matches!(self, AedelstenCacheState::Ready(_))
    }

    /// Whether it is ready, so that setting it again is refused.
    pub(crate) fn is_already_derived(&self) -> bool {
        self.is_ready()
    }

    /// Whether it is empty or invalidated.
    pub(crate) fn is_freed(&self) -> bool {
        matches!(self, AedelstenCacheState::Empty | AedelstenCacheState::Invalid(_))
    }

    /// In debug builds, a warning on stderr when `cache_name` is freed twice.
    pub(crate) fn debug_double_free_warning(&self, cache_name: &str) {
        if cfg!(debug_assertions) && self.is_freed() {
            eprintln!(
                "Warning: Attempting to free {} that is already freed. \
                 This is a no-op, but may indicate a double-free in the calling code.",
                cache_name
            );
        }
    }

    /// Invalidates it, with `msg` as the reason a later use reports.
    pub(crate) fn invalidate(&mut self, msg: String) {
        *self = AedelstenCacheState::Invalid(InvalidationReason { msg });
    }

    /// The value, or why there is none.
    pub(crate) fn get_cache(&self) -> Result<&T, String> {
        match self {
            AedelstenCacheState::Ready(v) => Ok(v),
            AedelstenCacheState::Invalid(reason) => Err(format!("Cache is invalid: {}", reason.msg)),
            _ => Err("Cache is not ready.".to_string()),
        }
    }

    /// `get_cache`, mutable.
    pub(crate) fn get_cache_mut(&mut self) -> Result<&mut T, String> {
        match self {
            AedelstenCacheState::Ready(v) => Ok(v),
            AedelstenCacheState::Invalid(reason) => Err(format!("Cache is invalid: {}", reason.msg)),
            _ => Err("Cache is not ready.".to_string()),
        }
    }

    /// Drops the value.
    pub(crate) fn free_cache(&mut self) {
        *self = AedelstenCacheState::Empty;
    }
}

/// Why a cache was invalidated.
#[derive(Debug)]
pub(crate) struct InvalidationReason {
    msg: String,
}
