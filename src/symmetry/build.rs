// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let libmsym_dir = manifest_dir.join("src/libmsym");
    let src_dir = libmsym_dir.join("src");

    println!("cargo:rerun-if-changed={}", src_dir.display());
    println!("cargo:rerun-if-changed={}", libmsym_dir.join("src/msym.h").display());
    println!("cargo:rerun-if-changed={}", libmsym_dir.join("src/msym_error.h").display());
    println!("cargo:rerun-if-changed=build.rs");

    // CMake normally generates this header. For a static build, a minimal version is enough.
    let exports_header = out_dir.join("msym_EXPORTS.h");
    write_msym_exports_header(&exports_header);

    let mut build = cc::Build::new();

    build
        .include(&libmsym_dir)   // matches CMake's include_directories("${PROJECT_SOURCE_DIR}")
        .include(&out_dir)       // matches CMake's include_directories("${PROJECT_BINARY_DIR}")
        .define("MSYM_EXPORTS_BUILT_AS_STATIC", None)
        .flag_if_supported("-std=c99")
        .pic(true)
        // Vendored third-party code: its warnings are not ours to fix.
        .warnings(false);

    for file in c_sources(&src_dir) {
        println!("cargo:rerun-if-changed={}", file.display());
        build.file(file);
    }

    build.compile("msym");

    #[cfg(unix)]
    println!("cargo:rustc-link-lib=m");
}

fn c_sources(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    for entry in fs::read_dir(dir).expect("Failed to read libmsym source directory") {
        let path = entry.expect("Bad directory entry").path();
        if path.extension().and_then(|s| s.to_str()) == Some("c") {
            files.push(path);
        }
    }

    files.sort();
    files
}

fn write_msym_exports_header(path: &Path) {
    let contents = r#"
#ifndef MSYM_EXPORTS_H
#define MSYM_EXPORTS_H

#define MSYM_EXPORT
#define MSYM_NO_EXPORT

#endif
"#;

    fs::write(path, contents).expect("Failed to write msym_EXPORTS.h");
}