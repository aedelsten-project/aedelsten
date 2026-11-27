// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The build tasks (`cargo install-aedelsten ...`): installing the library, the C header and the Python, Julia and
//! Fortran interfaces under `PREFIX`, regenerating or checking the committed C header, and exporting a release.

use std::path::{Path, PathBuf};
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let release = args.contains(&"--release".to_string());
    let int64 = args.contains(&"--int64".to_string());
    match args.get(1).map(String::as_str) {
        Some("install") => install(release),
        Some("install-julia") => install_julia(),
        Some("install-python") => install_python(),
        Some("install-fortran") => compile_fortran(release, int64),
        Some("header") => header(args.contains(&"--check".to_string())),
        Some("export") => export(&args[2..]),
        _ => eprintln!(
            "Usage: cargo install-aedelsten [install|install-julia|install-python|install-fortran|header] [--release] [--int64] [--check]\n       \
             cargo install-aedelsten export <dir> [--tag <tag>] [--test]"
        ),
    }
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}

// The C header is committed (include/aedelsten.h), so building or installing the library needs no cbindgen. After a
// change to the C ABI, `header` regenerates it; `header --check` and the test below fail while it is out of date.
fn generate_header() -> Vec<u8> {
    let root = project_root();
    let mut header = Vec::new();
    cbindgen::Builder::new()
        .with_crate(&root)
        .with_config(cbindgen::Config::from_file(root.join("cbindgen.toml")).unwrap())
        .generate()
        .expect("unable to generate C bindings")
        .write(&mut header);
    header
}

fn header(check: bool) {
    let path = project_root().join("include").join("aedelsten.h");
    let generated = generate_header();
    if check {
        if fs::read(&path).ok().as_deref() != Some(generated.as_slice()) {
            eprintln!("{} is out of date: run `cargo install-aedelsten header`", path.display());
            std::process::exit(1);
        }
        println!("{} is up to date", path.display());
    } else {
        fs::write(&path, generated).unwrap();
        println!("Wrote {}", path.display());
    }
}

// Copies the built library, the C header and the bundled basis library to PREFIX/lib, PREFIX/include and
// PREFIX/share/aedelsten/basis (where the installed library looks for it).
fn install(release: bool) {
    let profile = if release { "release" } else { "debug" };
    let root = project_root();

    let lib_src = root.join("target").join(profile).join(lib_name());
    let lib_dst = install_prefix().join("lib").join(lib_name());
    let header_src = root.join("include").join("aedelsten.h");
    let header_dst = install_prefix().join("include").join("aedelsten.h");

    // Print all paths before attempting anything
    println!("project_root : {}", root.display());
    println!("lib_src      : {}", lib_src.display());
    println!("lib_dst      : {}", lib_dst.display());
    println!("header_src   : {}", header_src.display());
    println!("header_dst   : {}", header_dst.display());

    // Verify sources exist before trying to copy
    for src in [&lib_src, &header_src] {
        if !src.exists() {
            eprintln!("ERROR: source file not found: {}", src.display());
            std::process::exit(1);
        }
    }

    fs::create_dir_all(lib_dst.parent().unwrap()).unwrap();
    fs::copy(&lib_src, &lib_dst).unwrap();
    println!("Installed library → {}", lib_dst.display());

    fs::create_dir_all(header_dst.parent().unwrap()).unwrap();
    fs::copy(&header_src, &header_dst).unwrap();
    println!("Installed header  → {}", header_dst.display());

    let basis_src = root.join("src").join("basis").join("data");
    let basis_dst = install_prefix().join("share").join("aedelsten").join("basis");
    fs::create_dir_all(&basis_dst).unwrap();
    for entry in fs::read_dir(&basis_src).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            fs::copy(&path, basis_dst.join(path.file_name().unwrap())).unwrap();
        }
    }
    println!("Installed basis   → {}", basis_dst.display());
}

// Copies the Python interface to PREFIX/lib/python.
fn install_python() {
    let src = interfaces_dir().join("aedelsten.py");
    let dst = install_prefix().join("lib").join("python").join("aedelsten.py");

    if !src.exists() {
        eprintln!("ERROR: source file not found: {}", src.display());
        std::process::exit(1);
    }

    fs::create_dir_all(dst.parent().unwrap()).unwrap();
    fs::copy(&src, &dst).unwrap();
    println!("Installed Python interface → {}", dst.display());
}

// Copies the Julia interface into the first JULIA_DEPOT_PATH entry as a package, or to PREFIX/lib/julia.
fn install_julia() {
    let src = interfaces_dir().join("Aedelsten.jl");

    let dst = if let Ok(depot) = std::env::var("JULIA_DEPOT_PATH") {
        // Install as a proper Julia package into the depot
        PathBuf::from(depot.split(':').next().unwrap())
            .join("packages")
            .join("Aedelsten")
            .join("src")
            .join("Aedelsten.jl")
    } else {
        // Fall back to PREFIX/lib/julia/
        install_prefix().join("lib").join("julia").join("Aedelsten.jl")
    };

    println!("src: {}", src.display());
    println!("dst: {}", dst.display());

    if !src.exists() {
        eprintln!("ERROR: source file not found: {}", src.display());
        std::process::exit(1);
    }

    fs::create_dir_all(dst.parent().unwrap()).unwrap();
    fs::copy(&src, &dst).unwrap();
    println!("Installed Julia interface → {}", dst.display());
}

// Compiles the Fortran interface to PREFIX/lib/Aedelsten.o, its module file into PREFIX/include; `int64` selects
// 64-bit default integers on the Fortran side.
fn compile_fortran(release: bool, int64: bool) {
    let root = project_root();
    let prefix = install_prefix();
    let profile = if release { "release" } else { "debug" };

    let fc = std::env::var("FC")
        .unwrap_or_else(|_| find_fortran_compiler().to_string());

    let src  = interfaces_dir().join("aedelsten.f90");
    let lib  = root.join("target").join(profile).join(lib_name());
    let out  = prefix.join("lib").join("Aedelsten.o");
    let moddir = prefix.join("include");

    for path in [&src, &lib] {
        if !path.exists() {
            eprintln!("ERROR: source file not found: {}", path.display());
            std::process::exit(1);
        }
    }

    fs::create_dir_all(&moddir).unwrap();
    fs::create_dir_all(out.parent().unwrap()).unwrap();

    // -cpp runs the preprocessor (gfortran's spelling): AEDELSTEN_FORTRAN_INT64 makes the interface's default
    // integer kind int64 (for callers compiled with -fdefault-integer-8, ...); the wire ABI to Rust stays i32.
    let mut cmd = std::process::Command::new(&fc);
    cmd.arg("-cpp");
    if int64 {
        cmd.arg("-DAEDELSTEN_FORTRAN_INT64");
    }
    cmd.args([
        "-c",
        src.to_str().unwrap(),
        "-o", out.to_str().unwrap(),
        &format!("-J{}", moddir.display()),  // .mod output dir (gfortran/ifort)
        &format!("-I{}", prefix.join("include").display()),
    ]);
    let status = cmd.status().expect("Failed to run Fortran compiler");

    if !status.success() {
        eprintln!("ERROR: Fortran compilation failed");
        std::process::exit(1);
    }

    println!("Compiled Fortran interface (external_int_size = {})", if int64 { "int64" } else { "int32" });
    println!("  object file : {}", out.display());
    println!("  module file : {}", moddir.join("aedelsten.mod").display());
    println!();
    println!("To link in your program:");
    println!("  {} your_program.f90 {} {} -o your_program",
        fc,
        out.display(),
        lib.display()
    );
}

// The first of the known Fortran compilers that runs; exits if none does.
fn find_fortran_compiler() -> &'static str {
    for compiler in &["gfortran", "ifx", "ifort", "nvfortran", "flang"] {
        if std::process::Command::new(compiler)
            .arg("--version")
            .output()
            .is_ok()
        {
            return compiler;
        }
    }
    eprintln!("ERROR: no Fortran compiler found. Install gfortran or set FC.");
    std::process::exit(1);
}

fn interfaces_dir() -> PathBuf {
    project_root().join("src").join("interfaces")
}

fn lib_name() -> &'static str {
    if cfg!(target_os = "windows") { "aedelsten.dll" }
    else if cfg!(target_os = "macos") { "libaedelsten.dylib" }
    else { "libaedelsten.so" }
}

// PREFIX; exits with usage advice if it is not set.
fn install_prefix() -> PathBuf {
    std::env::var("PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            eprintln!("ERROR: PREFIX is not set.");
            eprintln!("Please specify an installation directory, e.g.:");
            eprintln!("  PREFIX=~/.local cargo install-aedelsten install");
            eprintln!("  PREFIX=/opt/aedelsten cargo install-aedelsten install");
            std::process::exit(1);
        })
}

// The message of a release tag: what the release repository is, the warranty, bug reports and contributions (as the
// README's section on them).
fn release_tag_message(version: &str) -> String {
    format!(
        "Ädelsten {version}, a release version.\n\n\
         This repository holds released versions only; development is internal. Provided as is, without warranty of\n\
         any kind, including as to the accuracy or correctness of its results (see LICENSE). Bug reports are welcome,\n\
         as issues at gitlab.com/aedelsten/aedelsten. The project is not looking for collaborators: to add code, open an\n\
         issue first with a clear plan; otherwise fork the release under the terms of its licence. The changes are\n\
         in CHANGELOG.md; see README.md.\n"
    )
}

// What a release leaves out of the tree: the development documents, everything written for Claude, and the CI that
// publishes releases.
const NOT_SHIPPED: [&str; 3] = ["docs", "CLAUDE.md", ".gitlab-ci.yml"];

// Exits with `message`.
fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ERROR: {message}");
    std::process::exit(1);
}

// Runs `command`, exiting with `what` if it cannot start or fails; its standard output.
fn run(command: &mut std::process::Command, what: &str) -> Vec<u8> {
    let output = command.stderr(std::process::Stdio::inherit()).output().unwrap_or_else(|e| fail(format!("{what}: {e}")));
    if !output.status.success() {
        fail(format!("{what} failed ({})", output.status));
    }
    output.stdout
}

// The `version` of the `[workspace.package]` table of a root `Cargo.toml`.
fn workspace_version(manifest: &str) -> Option<String> {
    let table = manifest.split("[workspace.package]").nth(1)?;
    let table = table.split("\n[").next()?;
    table.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim() == "version").then(|| value.trim().trim_matches('"').to_string())
    })
}

// `REUSE.toml` without the entries of files that do not ship.
fn shipped_reuse_toml(reuse: &str) -> String {
    let unshipped = |line: &str| {
        let entry = line.trim().trim_end_matches(',').trim_matches('"');
        NOT_SHIPPED.iter().any(|path| entry == *path || entry.starts_with(&format!("{path}/")))
    };
    reuse.lines().filter(|line| !unshipped(line)).map(|line| format!("{line}\n")).collect()
}

// Writes the release of `tag` (default `v<workspace version>`) into `dir`: the tag's tracked files without history,
// without `NOT_SHIPPED`, `REUSE.toml` listing only what ships. `dir` is empty or new, or a checkout of the release
// repository (everything but its `.git` is replaced, so the release can be committed and tagged there). Refuses a tag
// that does not exist or whose version is not the tag's. The copy is then built (in a temporary target directory)
// and its header checked; with `--test` its suite run too.
fn export(args: &[String]) {
    let mut dir = None;
    let mut tag = None;
    let mut test = false;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--tag" => tag = Some(rest.next().unwrap_or_else(|| fail("--tag needs a tag")).clone()),
            "--test" => test = true,
            _ if dir.is_none() && !arg.starts_with("--") => dir = Some(PathBuf::from(arg)),
            _ => fail(format!("unknown argument {arg}")),
        }
    }
    let dir = dir.unwrap_or_else(|| fail("usage: cargo install-aedelsten export <dir> [--tag <tag>] [--test]"));
    let root = project_root();
    let git = |args: &[&str]| {
        let mut command = std::process::Command::new("git");
        command.current_dir(&root).args(args);
        command
    };

    let manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap_or_else(|e| fail(format!("Cargo.toml: {e}")));
    let tag = tag.unwrap_or_else(|| format!("v{}", workspace_version(&manifest).unwrap_or_else(|| fail("no [workspace.package] version"))));
    if !git(&["rev-parse", "--verify", "--quiet", &format!("refs/tags/{tag}")]).status().is_ok_and(|s| s.success()) {
        fail(format!("no tag {tag}"));
    }
    let tagged_manifest = String::from_utf8(run(&mut git(&["show", &format!("{tag}:Cargo.toml")]), "reading the tag's Cargo.toml")).unwrap();
    let version = workspace_version(&tagged_manifest).unwrap_or_else(|| fail(format!("{tag}: no [workspace.package] version")));
    if tag.trim_start_matches('v') != version {
        fail(format!("the tag {tag} carries version {version}"));
    }

    fs::create_dir_all(&dir).unwrap_or_else(|e| fail(format!("{}: {e}", dir.display())));
    let entries: Vec<PathBuf> = fs::read_dir(&dir).unwrap().map(|entry| entry.unwrap().path()).collect();
    let checkout = dir.join(".git").exists();
    if !entries.is_empty() && !checkout {
        fail(format!("{} is neither empty nor a git checkout", dir.display()));
    }
    for path in entries.iter().filter(|path| path.file_name() != Some(std::ffi::OsStr::new(".git"))) {
        if path.is_dir() { fs::remove_dir_all(path) } else { fs::remove_file(path) }.unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    }

    let mut archive = vec!["archive".to_string(), "--format=tar".to_string(), tag.clone(), "--".to_string(), ".".to_string()];
    archive.extend(NOT_SHIPPED.iter().map(|path| format!(":(exclude){path}")));
    let tar = run(&mut git(&archive.iter().map(String::as_str).collect::<Vec<_>>()), "git archive");
    let mut extract = std::process::Command::new("tar");
    extract.current_dir(&dir).args(["-x", "-f", "-"]).stdin(std::process::Stdio::piped());
    let mut child = extract.spawn().unwrap_or_else(|e| fail(format!("tar: {e}")));
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), &tar).unwrap();
    if !child.wait().unwrap().success() {
        fail("tar failed");
    }
    let reuse = dir.join("REUSE.toml");
    let shipped = shipped_reuse_toml(&fs::read_to_string(&reuse).unwrap_or_else(|e| fail(format!("REUSE.toml: {e}"))));
    fs::write(&reuse, shipped).unwrap();
    println!("Exported {tag} (version {version}) → {}", dir.display());

    let target = std::env::temp_dir().join(format!("aedelsten-export-{version}"));
    let cargo = |args: &[&str], what: &str| {
        let mut command = std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()));
        command.current_dir(&dir).env("CARGO_TARGET_DIR", &target).args(args);
        let status = command.status().unwrap_or_else(|e| fail(format!("{what}: {e}")));
        if !status.success() {
            fail(format!("{what} of the export failed"));
        }
    };
    cargo(&["build", "--release"], "the build");
    cargo(&["run", "--package", "xtask", "--", "header", "--check"], "the header check");
    if test {
        cargo(&["test", "--workspace", "--no-fail-fast"], "the suite");
    }
    println!("The export builds{} (target directory {})", if test { " and its suite passes" } else { "" }, target.display());
    let message = dir.join(".git").join("RELEASE_TAG_MESSAGE");
    if checkout {
        fs::write(&message, release_tag_message(&version)).unwrap();
        println!("To release it, in {}:", dir.display());
        println!("  git add -A && git commit -m \"Ädelsten {version}\" && git tag -a {tag} -F .git/RELEASE_TAG_MESSAGE");
    } else {
        println!("The release tag's message:\n\n{}", release_tag_message(&version));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn committed_header_is_current() {
        let path = super::project_root().join("include").join("aedelsten.h");
        let committed = std::fs::read(&path).expect("include/aedelsten.h is missing");
        assert!(committed == super::generate_header(), "include/aedelsten.h is out of date: run `cargo install-aedelsten header`");
    }

    #[test]
    fn the_release_tag_says_what_a_release_is() {
        let message = super::release_tag_message("0.5.0");
        assert!(message.starts_with("Ädelsten 0.5.0, a release version."));
        for statement in ["development is internal", "without warranty", "as issues", "not looking for collaborators", "fork the release", "CHANGELOG.md"] {
            assert!(message.contains(statement), "{statement}");
        }
    }

    #[test]
    fn the_workspace_version_is_read_from_its_table() {
        let manifest = "[workspace]\nmembers = []\n\n[workspace.package]\nversion = \"0.5.0\"\nauthors = []\n\n[package]\nversion = \"9.9.9\"\n";
        assert_eq!(super::workspace_version(manifest).as_deref(), Some("0.5.0"));
        assert_eq!(super::workspace_version("[package]\nversion = \"1.0.0\"\n"), None);
    }

    #[test]
    fn the_shipped_reuse_toml_lists_only_what_ships() {
        let reuse = "path = [\n    \"README.md\",\n    \"CLAUDE.md\",\n    \"docs/**\",\n    \"docs-like.txt\",\n]\n";
        assert_eq!(super::shipped_reuse_toml(reuse), "path = [\n    \"README.md\",\n    \"docs-like.txt\",\n]\n");
    }

    #[test]
    fn the_shipped_reuse_toml_of_this_tree_names_nothing_left_out() {
        let reuse = std::fs::read_to_string(super::project_root().join("REUSE.toml")).unwrap();
        let shipped = super::shipped_reuse_toml(&reuse);
        assert!(!shipped.contains("CLAUDE.md") && !shipped.contains("\"docs/"), "{shipped}");
        assert_eq!(reuse.lines().count() - shipped.lines().count(), 2);
    }
}
