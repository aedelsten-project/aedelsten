// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Links the BLAS and LAPACK the build chose: Intel MKL (`mkl`), OpenBLAS (`openblas`), the system BLAS
// (`system-blas`), or by default Accelerate on macOS and the system BLAS elsewhere.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

const OPENBLAS_ENV: &str = "OPENBLASROOT";
const MKL_ENV: &str = "MKLROOT";
const OPENBLAS_STATIC_ENV: &str = "OPENBLAS_STATIC";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={}", OPENBLAS_ENV);
    println!("cargo:rerun-if-env-changed={}", MKL_ENV);
    println!("cargo:rerun-if-env-changed={}", OPENBLAS_STATIC_ENV);

    // Prevent ambiguous configurations
    let chosen = [
        cfg!(feature = "mkl"),
        cfg!(feature = "openblas"),
        cfg!(feature = "system-blas"),
    ]
    .into_iter()
    .filter(|&x| x)
    .count();

    if chosen > 1 {
        panic!("Enable only one of: features = [\"mkl\", \"openblas\", \"system-blas\"]");
    }

    #[cfg(target_os = "macos")]
    mac_setup();

    #[cfg(not(target_os = "macos"))]
    linux_setup();
}

#[cfg(target_os = "macos")]
fn mac_setup() {
    if cfg!(feature = "mkl") {
        mkl_setup();
        return;
    }

    if cfg!(feature = "openblas") {
        openblas_setup_from_env_or_pkgconfig(true);
        return;
    }

    // Default on macOS: Accelerate
    println!("cargo:rustc-link-lib=framework=Accelerate");
}

#[cfg(not(target_os = "macos"))]
fn linux_setup() {
    if cfg!(feature = "mkl") {
        mkl_setup();
        return;
    }

    if cfg!(feature = "openblas") {
        openblas_setup_from_env_or_pkgconfig(false);
        return;
    }

    // Default on Linux: try system BLAS first, then fall back to OPENBLASROOT.
    if cfg!(feature = "system-blas") || !cfg!(feature = "openblas") {
        if try_system_blas_pkg_config() {
            return;
        }
    }

    // Fallback: OPENBLASROOT
    openblas_setup_from_env_or_pkgconfig(false);
}

/// Links the system BLAS through pkg-config, OpenBLAS first, then the generic `blas`; false if neither is found.
#[allow(dead_code)] // Unused on macOS.
fn try_system_blas_pkg_config() -> bool {
    // Prefer OpenBLAS if installed (common on Linux)
    if try_pkg_config_lib("openblas") {
        return true;
    }
    // Generic BLAS (provided by system alternatives, ATLAS, OpenBLAS, BLIS, MKL packages, etc.)
    if try_pkg_config_lib("blas") {
        return true;
    }
    false
}

fn openblas_setup_from_env_or_pkgconfig(is_macos: bool) {
    // OPENBLASROOT first, when set.
    if let Ok(root) = env::var(OPENBLAS_ENV) {
        let (libdir, kind) = find_openblas_libdir_and_kind(&root, is_macos);
        println!("cargo:rustc-link-search=native={}", libdir.display());
        println!("cargo:rustc-link-lib={}={}", kind, "openblas");
        return;
    }

    // Otherwise try pkg-config (system BLAS/OpenBLAS)
    if try_pkg_config_lib("openblas") || try_pkg_config_lib("blas") {
        return;
    }

    panic!(
        "Could not find OpenBLAS.\n\
         - Set {env} to your OpenBLAS install prefix (or its lib dir), or\n\
         - Install a system BLAS/OpenBLAS and ensure pkg-config can find it.\n\
         On Debian/Ubuntu, packages are often: libopenblas-dev and pkg-config.",
        env = OPENBLAS_ENV
    );
}

fn find_openblas_libdir_and_kind(openblas_root: &str, is_macos: bool) -> (PathBuf, &'static str) {
    let mut root = openblas_root.to_owned();

    // The prefix's lib directory.
    if root.ends_with("/lib") || root.ends_with("/lib/") {
    } else if root.ends_with('/') {
        root.push_str("lib");
    } else {
        root.push_str("/lib");
    }

    let libdir = PathBuf::from(&root);
    if !libdir.exists() {
        panic!("OpenBLAS lib dir does not exist: {}", libdir.display());
    }

    // Decide static vs dynamic:
    // - if OPENBLAS_STATIC=1 -> static
    // - else prefer dynamic if present
    let force_static = matches!(
        env::var(OPENBLAS_STATIC_ENV).as_deref(),
        Ok("1") | Ok("true") | Ok("yes") | Ok("on")
    );

    let (shared_name, static_name) = if is_macos {
        ("libopenblas.dylib", "libopenblas.a")
    } else {
        ("libopenblas.so", "libopenblas.a")
    };

    let shared = libdir.join(shared_name);
    let static_ = libdir.join(static_name);

    if force_static {
        if !static_.exists() {
            panic!(
                "OPENBLAS_STATIC requested, but not found: {}",
                static_.display()
            );
        }
        return (libdir, "static");
    }

    if shared.exists() {
        return (libdir, "dylib");
    }
    if static_.exists() {
        return (libdir, "static");
    }

    // Some distros only ship versioned sonames (libopenblas.so.0)
    if !is_macos {
        if let Some(p) = find_first_matching(&libdir, "libopenblas.so.") {
            // Linker still uses -lopenblas, but we confirm *some* shared object exists
            let _ = p;
            return (libdir, "dylib");
        }
    }

    panic!(
        "Found OpenBLAS libdir {}, but no OpenBLAS library found (expected {} or {}).",
        libdir.display(),
        shared_name,
        static_name
    );
}

fn mkl_setup() {
    let mklroot = env::var(MKL_ENV).unwrap_or_else(|_| {
        panic!(
            "{env} not set. Source Intel oneAPI setvars.sh (or equivalent) so MKLROOT is defined.",
            env = MKL_ENV
        )
    });

    let libdir = find_mkl_libdir(&mklroot);
    println!("cargo:rustc-link-search=native={}", libdir.display());

    // Simplest MKL link: single dynamic dispatcher
    println!("cargo:rustc-link-lib=dylib=mkl_rt");

    // Common MKL deps (especially on Linux)
    #[cfg(not(target_os = "macos"))]
    {
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
    }

    // If Intel OpenMP runtime is present, link it (often needed depending on MKL threading)
    if let Some(iomp_kind) = detect_iomp(&libdir) {
        println!("cargo:rustc-link-lib={}={}", iomp_kind, "iomp5");
    }
}

fn find_mkl_libdir(mklroot: &str) -> PathBuf {
    let root = Path::new(mklroot);

    // Common layouts:
    // oneAPI Linux: $MKLROOT/lib/intel64
    // some installs: $MKLROOT/lib
    let candidates = [
        root.join("lib/intel64"),
        root.join("lib/intel64_lin"),
        root.join("lib"),
    ];

    for c in candidates {
        if c.exists() {
            // Verify mkl_rt exists (shared)
            #[cfg(target_os = "macos")]
            let ok = c.join("libmkl_rt.dylib").exists();
            #[cfg(not(target_os = "macos"))]
            let ok = c.join("libmkl_rt.so").exists() || find_first_matching(&c, "libmkl_rt.so.").is_some();

            if ok {
                return c;
            }
        }
    }

    panic!(
        "Could not locate MKL libraries under {} (looked for libmkl_rt in lib/intel64, lib/intel64_lin, lib).",
        mklroot
    );
}

fn detect_iomp(libdir: &Path) -> Option<&'static str> {
    #[cfg(target_os = "macos")]
    {
        let dylib = libdir.join("libiomp5.dylib");
        if dylib.exists() {
            return Some("dylib");
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let so = libdir.join("libiomp5.so");
        if so.exists() || find_first_matching(libdir, "libiomp5.so.").is_some() {
            return Some("dylib");
        }
    }
    None
}

/// Minimal pkg-config parser: reads `pkg-config --libs <name>` and emits cargo link directives.
/// Returns true if pkg-config succeeded and we printed at least one -l.
fn try_pkg_config_lib(name: &str) -> bool {
    let output = Command::new("pkg-config")
        .arg("--libs")
        .arg(name)
        .output();

    let Ok(out) = output else { return false };
    if !out.status.success() {
        return false;
    }

    let s = String::from_utf8_lossy(&out.stdout);
    let mut saw_lib = false;

    for tok in s.split_whitespace() {
        if let Some(path) = tok.strip_prefix("-L") {
            if !path.is_empty() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        } else if let Some(lib) = tok.strip_prefix("-l") {
            if !lib.is_empty() {
                // default kind is dylib (good for system libs)
                println!("cargo:rustc-link-lib={}", lib);
                saw_lib = true;
            }
        } else if tok == "-pthread" {
            println!("cargo:rustc-link-lib=pthread");
            saw_lib = true;
        } else {
            // Ignore other flags: -Wl,..., -rpath, etc.
        }
    }

    saw_lib
}

fn find_first_matching(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(prefix) {
            return Some(e.path());
        }
    }
    None
}
