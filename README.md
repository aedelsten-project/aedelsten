# Ädelsten

Ädelsten is a library of molecular integrals over Gaussian basis functions, written in Rust and callable from Rust, C,
Fortran, Python and Julia. It computes the integrals and the single operations built directly on them; a host program
(an SCF, a multiconfigurational or perturbation-theory code) assembles its Fock matrices, Hamiltonians and energies
from them.

- **One-electron integrals:** overlap, kinetic energy, the potentials of point and finite (Gaussian) nuclei, moments,
  linear and angular momentum, electric fields at sites; over contracted (general or segmented) or primitive bases,
  Cartesian or spherical.
- **Two-electron integrals:** the stored tensor, and integral-direct J and K builds (restricted, unrestricted, for
  densities of any symmetry, for several densities in one walk, the active-space integrals of a CASSCF, the
  half-transformed integrals of a perturbation theory), full-range and range-separated (erf and erfc), with Schwarz
  screening.
- **Geometric first derivatives** of the one- and two-electron integrals, contracted with densities, with frozen
  centres.
- **Relativity:** the spin-free exact two-component (X2C) one-electron Hamiltonian and the picture change of other
  operators; spin-orbit integrals, one-electron and the atomic mean-field (AMFI) two-electron term.
- **Embedding:** external point multipoles of any order, ab initio model potentials (AIMPs) for cores and for an
  environment, and polarizable sites (induced dipoles, with Thole damping).
- **Point-group symmetry** of the molecule and its operations on matrices and orbitals.
- **MPI:** every computation can split its work over the ranks of a communicator the host gives it.
- **Reporting:** the memory each part of a context holds, timings of the library's own work, and the list of
  references for the methods a run used.

## Building

Ädelsten needs **Rust 1.88 or newer** and a **BLAS and LAPACK with 32-bit (LP64) integers**:

```
cargo build --release
```

The BLAS is a build-time choice: Accelerate on macOS and the system BLAS elsewhere by default, or one of the features
`openblas` (found through `OPENBLASROOT` or pkg-config), `mkl` (through `MKLROOT`) or `system-blas`. Further features:

| Feature | Effect |
|---|---|
| `mpi` | The work of each computation split over the ranks of a communicator (needs an MPI installation). |
| `unchecked-arrays` | Array accesses without bounds checks: faster, for production builds once a build has run checked. |
| `profile` | The fine-grained timings (per batch of integrals) as well as the coarse ones. |

The build generates code for the CPU it runs on (`.cargo/config.toml`), so a binary runs on that type of CPU only:
build on the machine, or a node of the partition, that runs it. A `RUSTFLAGS` in the environment replaces this.

## Installing for C, Fortran, Python and Julia

After `cargo build --release`, choose where the files go and install them with `--release`, so that the release build
is the one copied:

```
export PREFIX=/path/to/install
cargo install-aedelsten install --release            # PREFIX/lib/libaedelsten.{dylib,so}, PREFIX/include/aedelsten.h,
                                                     # PREFIX/share/aedelsten/basis
cargo install-aedelsten install-python               # PREFIX/lib/python/aedelsten.py
cargo install-aedelsten install-julia                # the first JULIA_DEPOT_PATH entry as a package, else PREFIX/lib/julia
cargo install-aedelsten install-fortran --release    # PREFIX/lib/Aedelsten.o, PREFIX/include/aedelsten.mod
```

- **C:** include `aedelsten.h` and link `libaedelsten`. Every function is documented in the header.
- **Python** (3.10 or newer, with numpy) and **Julia** load the library at run time: set `AEDELSTEN_LIB` to its full
  path, or put its directory on `DYLD_LIBRARY_PATH` (macOS) or `LD_LIBRARY_PATH` (Linux). For Python, put
  `PREFIX/lib/python` on `PYTHONPATH`.
- **Fortran:** the module is compiled by `FC` (else the first of gfortran, ifx, ifort, nvfortran and flang that runs)
  with gfortran's preprocessor flag `-cpp`; another compiler may need its own spelling. A module file can be read only
  by the compiler that wrote it, so compile the module with the compiler of your program. With `--int64` the
  module's default integer is 64-bit, for programs compiled with 64-bit default integers. Link a program as
  ```
  gfortran -I$PREFIX/include program.f90 $PREFIX/lib/Aedelsten.o $PREFIX/lib/libaedelsten.dylib
  ```
  (on macOS, gfortran may need `SDKROOT=$(xcrun --show-sdk-path)`).

`examples/python`, `examples/julia` and `examples/fortran` hold a demonstration program of each area in each language;
each says at its top how to run it.

## Using the library

A computation goes through a **context**, made from an input file that gives the basis sets and the centres. The
context derives what the integrals need when asked to (`set_shell_pairs`, `set_eri_data`, ...) and keeps it until it
is freed (`free_*`); `memory_breakdown` reports what it holds. Every computation takes its temporary memory from a
**scratch** buffer that the caller sizes once (`set_scratch`); `scratch_bytes` gives the size the chosen operations
need, from the input alone. Results go into the caller's arrays, column-major.

**Basis sets** are read in the Molcas basis-library format. Take them from the Basis Set Exchange in that format, so
that a calculation can be reproduced: a file is looked for in the working directory, then in the directories of
`AEDELSTEN_BASIS_LIBRARY` (separated as in `PATH`), then in the bundled library. The bundled files (`src/basis/data`,
installed to `PREFIX/share/aedelsten/basis`, which an installed library finds from its own location; the source tree's
copy after it) serve the library's tests and its own uses (the reference orbitals of the model Fockian and of AMFI).

**Under MPI** the host passes its communicator when it makes the context, as a Fortran handle from C, Fortran, Python
and Julia; a host without MPI of its own can initialize it through the library (`aedelsten_mpi_init_c`).

## Versions, and depending on Ädelsten from Rust

Every crate of the workspace (`aedelsten` and its foundation crates `arrays`, `aedelsten_mpi`, `instrumentation`,
`constants`) carries the one version of the workspace, and a release is tagged `v<version>`. A Rust program depends
on a release through its tag, naming each crate it uses:

```toml
[dependencies]
aedelsten = { git = "<Ädelsten's release repository>", tag = "v0.5.0", version = "0.5" }
arrays = { git = "<Ädelsten's release repository>", tag = "v0.5.0", version = "0.5" }
```

The tag fixes the release built against (and `Cargo.lock` its commit); the version requirement makes Cargo refuse a
source whose version does not match, which also holds when the dependency is a `path` to a local checkout. At run time
`aedelsten::get_version()` gives the version and `aedelsten::get_source_commit()` the git commit the library was built
from (`None` for a source tree that was no git checkout); from C `aedelsten_get_version_c` and
`aedelsten_get_source_commit_c`, from Python, Julia and Fortran `aedelsten_get_version` and
`aedelsten_get_source_commit`.

### What a version number promises

Ädelsten follows [Semantic Versioning 2.0.0](https://semver.org). The interfaces it versions strictly are the Rust
API (what the `aedelsten` crate exports), the C interface (its entry points and `include/aedelsten.h`) and the file
formats it reads (basis and AIMP libraries, inputs, xfields):

- a **major** release for any change an existing program can notice as a break: a function or C entry point removed
  or renamed, a signature, an argument's order, type or meaning changed, a default changed so that results change, a
  file accepted before now refused;
- a **minor** release for additions that leave every existing call compiling and behaving as before;
- a **patch** release for fixes that change no public signature or meaning.

Every release is checked against the previous one before it is published (`cargo semver-checks` for the Rust API, a
diff of the header for the C interface), and its number set by the largest change found.

**Before 1.0.0** (versions 0.y.z, as this one) the interfaces are still settling: a minor release may break them, each
break listed in the release's notes.

**The Fortran, Python and Julia interfaces are exempt from the strict promise.** They are convenience layers over the
C interface, written so that those languages need no glue of their own; anyone can write a different one over the C
interface, which is the stable contract. They carry the library's version and gain each new function with the C
interface, but their own conventions (names, keyword arguments, return types, error handling) may change in a minor
release, so that they can follow what is idiomatic in each language. Every such change is listed in the release's
notes, and none comes in a patch release. A change underneath them, in the C interface, is a major release anyway.

## Releases, bug reports and contributions

This repository holds released versions of Ädelsten only; it lives on GitLab and is mirrored to GitHub. Development is
internal: each release is an export of the development tree at its tag, without the history between releases.

Ädelsten is provided as is, without warranty of any kind, including as to the accuracy or correctness of its results
(see [LICENSE](LICENSE)). Check results that matter against independent references.

Bug reports are welcome, as issues on GitLab, at <https://gitlab.com/aedelsten/aedelsten/-/issues> (also for the GitHub
mirror): the version and source commit (above), the input, and what went wrong. Issues are the only channel for reports
and questions.

The project is not looking for collaborators. If you want to add code to it, open an issue first, with a clear plan of
what you want to add and why it belongs in Ädelsten. Otherwise, fork a release and develop it on your own, under the
terms of its licence.

What changed in each release is in [CHANGELOG.md](CHANGELOG.md).

## Citing Ädelsten

If you use results computed with Ädelsten, please cite the software: [CITATION.bib](CITATION.bib) has the entry.
Each release is archived on Zenodo, with a DOI of its own and a concept DOI that always points to the latest release.
The library also gives the host program the references for the methods a run used, to list with its output.

## Credits

Ernst D. Larsson wrote the library, its design, implementation and validation, from summer 2024 to summer 2026.
Since then he has directed the work, with Claude (Anthropic) writing most of the code.

The point-group symmetry uses libmsym (Marcus Johansson), vendored in `src/symmetry/src/libmsym`.

## Licence

MIT, see [LICENSE](LICENSE). Every file's licence is declared in REUSE format (SPDX headers and
[REUSE.toml](REUSE.toml)); check with `reuse lint`.

Some files in the repository are not this library's own and keep their own licences:

- libmsym (`src/symmetry/src/libmsym`): MIT, Copyright (c) 2015 Marcus Johansson.
- The STO-3G, cc-pVDZ, x2c-SVPall and x2c-QZVPPall basis sets in `src/basis/data`: from the Basis Set Exchange, BSD 3-Clause.
- The ANO-RCC basis library and its shorthand contractions (`ANO-RCC.aliases`, from `basis.tbl`) in `src/basis/data`:
  from OpenMolcas, LGPL 2.1. No embedding AIMPs are bundled; the tests' (`test_fixtures/basis/TEST-EMB-AIMP`) are
  MgO's from a self-consistent embedded ion (SCEI) calculation by E. D. Larsson, MIT.
- `test_fixtures/pehf_1pa`: converted from Dalton's `pehf_1pa` test, LGPL 2.1.

The licence texts are in [LICENSES](LICENSES). The libraries this library is built against keep their own licences:
the BLAS and LAPACK chosen at build time (Accelerate, OpenBLAS or Intel MKL) and, with the `mpi` feature, the MPI
implementation. cbindgen and cc are used at build time only.
