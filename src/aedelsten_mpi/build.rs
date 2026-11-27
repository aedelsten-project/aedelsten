// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

fn main() {
    println!("cargo:rerun-if-changed=src/shim.c");
    println!("cargo:rerun-if-env-changed=MPICC");

    #[cfg(feature = "mpi")]
    build_shim();
}

// Compiles `src/shim.c` with the MPI implementation's own compiler wrapper (`MPICC`, else `mpicc`), and links the MPI
// runtime.
#[cfg(feature = "mpi")]
fn build_shim() {
    let mpicc = std::env::var("MPICC").unwrap_or_else(|_| "mpicc".to_string());

    let found = std::process::Command::new(&mpicc)
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);

    if !found {
        panic!(
            "The `mpi` feature is enabled, but no working MPI compiler wrapper was found \
             (tried running `{mpicc} --version`).\n\
             - Install an MPI implementation, e.g. `brew install open-mpi` on macOS or \
               `apt install libopenmpi-dev` on Debian/Ubuntu, or\n\
             - Set MPICC to the full path of your MPI compiler wrapper if it isn't on PATH.",
            mpicc = mpicc
        );
    }

    cc::Build::new()
        .compiler(&mpicc)
        .file("src/shim.c")
        .warnings(true)
        .compile("aedelsten_mpi_shim");

    link_mpi_runtime(&mpicc);
}

// The MPI runtime's library paths and libraries, read from the `-L` and `-l` flags `mpicc -show` prints, passed to
// the link.
#[cfg(feature = "mpi")]
fn link_mpi_runtime(mpicc: &str) {
    let output = std::process::Command::new(mpicc)
        .arg("-show")
        .output()
        .unwrap_or_else(|e| panic!("Failed to run `{mpicc} -show` to determine MPI link flags: {e}"));
    if !output.status.success() {
        panic!(
            "`{mpicc} -show` exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let flags = String::from_utf8_lossy(&output.stdout);

    let mut saw_lib = false;
    for tok in flags.split_whitespace() {
        if let Some(path) = tok.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = tok.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={lib}");
            saw_lib = true;
        }
    }

    if !saw_lib {
        panic!(
            "`{mpicc} -show` produced no -l flags, so the MPI runtime library to link \
             against couldn't be determined. Full output: {flags}"
        );
    }
}
