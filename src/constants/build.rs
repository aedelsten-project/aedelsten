// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Writes `lmax.rs` with `L_MAX`, the build-time environment variable `L_MAX` (6 if unset: i functions, the highest
// in the bundled and tested basis sets).

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let l_max = env::var("L_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(6);

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("lmax.rs");

    fs::write(dest_path, format!("pub const L_MAX: usize = {};", l_max)).unwrap();
}
