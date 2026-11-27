// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The source commit the library is built from, for `get_source_commit`: the git commit of the source tree when it is
//! a checkout (a development tree, or the checkout a consumer's build makes of a release tag), none otherwise (an
//! exported tree, or no git on the machine).

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=build.rs");
    // A new commit moves HEAD or the branch it names; rebuild on either.
    if let Some(git_dir) = git_dir(&root) {
        let head = git_dir.join("HEAD");
        println!("cargo:rerun-if-changed={}", head.display());
        if let Some(reference) = std::fs::read_to_string(&head).ok().and_then(|h| h.strip_prefix("ref: ").map(|r| r.trim().to_string())) {
            println!("cargo:rerun-if-changed={}", git_dir.join(reference).display());
        }
    }
    let commit = Command::new("git")
        .args(["-C", &root.to_string_lossy(), "rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|commit| !commit.is_empty());
    if let Some(commit) = commit {
        println!("cargo:rustc-env=AEDELSTEN_SOURCE_COMMIT={commit}");
    }
}

/// The repository's git directory: `.git` itself, or the directory a `.git` file names (a worktree).
fn git_dir(root: &Path) -> Option<PathBuf> {
    let dot_git = root.join(".git");
    if dot_git.is_dir() {
        return Some(dot_git);
    }
    let pointer = std::fs::read_to_string(&dot_git).ok()?;
    let dir = PathBuf::from(pointer.strip_prefix("gitdir:")?.trim());
    Some(if dir.is_absolute() { dir } else { root.join(dir) })
}
