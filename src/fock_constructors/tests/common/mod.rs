// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// What the test files share.

use std::sync::atomic::{AtomicUsize, Ordering};

/// A path in the temporary directory that no other call in this process returns: the tests run in parallel, and
/// each removes its own files.
pub fn temp_path(name: &str) -> std::path::PathBuf {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("aedelsten_{}_{call}_{name}", std::process::id()))
}
