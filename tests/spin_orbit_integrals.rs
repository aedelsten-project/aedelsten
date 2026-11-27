// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Spin-orbit picture-change integrals: the four planes x, y, z, scalar of PySCF's int1e_spnucsp,
// (sigma.p) V (sigma.p) = p.Vp + i sigma.(p x Vp), for the nuclear potential.
//
// The vector planes (x, y, z) are antisymmetric matrices, and the scalar plane that a spin-free X2C calculation uses
// does not see them: they are checked here on their own.
//
// Element-by-element comparison with PySCF is not possible -- the two codes order and phase the
// spherical AOs differently -- so the comparison is of quantities invariant under any orthogonal
// change of the contracted basis: with S the overlap, W_c the matrix of plane c and
// M_c = S^{-1/2} W_c S^{-1/2},
//   * the largest eigenvalues of M_c M_c^T for each plane,
//   * Tr(M_x M_y M_z), which is sensitive to the relative sign and handedness of the three planes,
//   * Tr(M_scalar).
// Reference numbers: PySCF 2.11 with this library's x2c-SVPall entries for F and H (src/basis/data/X2C-SVPall, read
// into PySCF's basis format), spherical, Bohr (F at the origin, H at (0,0,1.733) and at 1.733 * (0.3, 0.4,
// 0.866025403784)):
//   from pyscf import gto; mol = gto.M(atom=..., basis={'F': ..., 'H': ...}, unit='Bohr')
//   S = mol.intor('int1e_ovlp'); W = mol.intor('int1e_spnucsp')   # (4, n, n)
//   M_c = S^{-1/2} W_c S^{-1/2}; eigvalsh(M_c M_c^T)[::-1][:2]; trace(M_x M_y M_z); trace(M_3)
// A single-primitive p shell on a bare Z = 9 nucleus has the analytic value Z <r^-3> = 19.14923 per
// L_c matrix element, i.e. an M M^T eigenvalue of 366.6929888837, which PySCF also gives.
//
// The single-primitive check changes the process's current directory (the basis lookup tries it first), so
// everything is in ONE test function.

mod common;

use aedelsten::arrays::matrix::{Matrix, MatrixMath, MatrixTrait, MatrixTraitMut};

use aedelsten::arrays::matrix::ColumnFactor;
use aedelsten::arrays::matrix::Factor;
use aedelsten::{AedelstenContext, ContractionType, MomentOrigin, OneElectronOperator};

struct Planes {
    s: Matrix<f64>,
    w: Vec<Matrix<f64>>,
}

/// Which spin-orbit integrals to compute.
#[derive(Clone, Copy)]
enum Kind {
    /// (sigma.p) V_nuc (sigma.p): 4 planes (x, y, z, scalar).
    Nuclear,
    /// (sigma.p) r (sigma.p) around the given origin: 12 planes, plane 4 * component + piece.
    Dipole([f64; 3]),
}

/// Builds the contracted planes for `input`, with `files` (name, content) written to a scratch
/// directory that is the current directory while the context is built.
fn planes(kind: Kind, input: &str, files: &[(&str, String)], tag: &str) -> Planes {
    let dir = common::temp_path(&format!("so_{}", tag));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, content) in files {
        std::fs::write(dir.join(name), content).unwrap();
    }
    let path = dir.join("m.inp");
    std::fs::write(&path, input).unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let context = AedelstenContext::new(path.to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single());
    std::env::set_current_dir(previous).unwrap();
    let mut ctx = context.unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_primitive_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let (n_prim, n) = (ctx.get_n_prim_unique(), ctx.get_n_cont());

    let n_planes = match kind {
        Kind::Nuclear => 4,
        Kind::Dipole(_) => 12,
    };
    let mut so = vec![0.0; n_prim * n_prim * n_planes];
    match kind {
        Kind::Nuclear => ctx.compute_one_electron_picture_change_integrals(&mut so, ctx.get_nuclear_attraction_operator(), true),
        Kind::Dipole(origin) => ctx.compute_one_electron_picture_change_integrals(&mut so, OneElectronOperator::moment(MomentOrigin::new(origin, 1)), true),
    }
    .unwrap();
    let mut contracted = vec![0.0; n * n * n_planes];
    ctx.contract_one_electron_array3d(&so, n_planes, &mut contracted).unwrap();
    let mut s = Matrix::<f64>::new(n, n);
    ctx.compute_one_electron_integrals(s.get_data_mut(), OneElectronOperator::overlap(), false).unwrap();
    let plane = |c: usize| {
        let mut m = Matrix::<f64>::new(n, n);
        for j in 0..n {
            for i in 0..n {
                m[(i, j)] = contracted[i + n * (j + n * c)];
            }
        }
        m
    };
    let _ = std::fs::remove_dir_all(&dir);
    Planes { s, w: (0..n_planes).map(plane).collect() }
}

fn inverse_sqrt(s: &Matrix<f64>) -> Matrix<f64> {
    let (values, vectors) = s.eigenvectors();
    let mut scaled = vectors.clone();
    for (j, &v) in values.iter().enumerate() {
        scaled.get_col_mut(j).iter_mut().for_each(|x| *x /= v.sqrt());
    }
    scaled.matmul(&vectors.view_transposed())
}

fn trace(m: &Matrix<f64>) -> f64 {
    (0..m.rows()).map(|i| m[(i, i)]).sum()
}

struct Invariants {
    top: [f64; 4], // the largest eigenvalue of M M^T of each plane
    triple: f64,
    scalar_trace: f64,
    antisymmetry: f64, // the largest |W_c + W_c^T| over the vector planes, relative to their size
}

fn invariants(p: &Planes) -> Invariants {
    let m = scaled_planes(p);
    let top = std::array::from_fn(|c| {
        let (values, _) = m[c].matmul(&m[c].view_transposed()).eigenvectors();
        values[values.len() - 1]
    });
    let mut worst = 0.0_f64;
    let mut size = 0.0_f64;
    for w in &p.w[..3] {
        for j in 0..w.cols() {
            for i in 0..w.rows() {
                worst = worst.max((w[(i, j)] + w[(j, i)]).abs());
                size = size.max(w[(i, j)].abs());
            }
        }
    }
    Invariants { top, triple: trace(&m[0].matmul(&m[1]).matmul(&m[2])), scalar_trace: trace(&m[3]), antisymmetry: worst / size }
}

/// The M_k = S^{-1/2} W_k S^{-1/2} of every plane.
fn scaled_planes(p: &Planes) -> Vec<Matrix<f64>> {
    let x = inverse_sqrt(&p.s);
    p.w.iter().map(|w| x.matmul(w).matmul(&x)).collect()
}

fn triple(m: &[Matrix<f64>], a: usize, b: usize, c: usize) -> f64 {
    trace(&m[a].matmul(&m[b]).matmul(&m[c]))
}

fn close(label: &str, got: f64, expected: f64, relative: f64) {
    assert!((got - expected).abs() <= relative * expected.abs(), "{label}: {got} vs PySCF {expected}");
}

#[test]
fn the_spin_orbit_planes_match_pyscf_and_the_analytic_single_primitive_value() {
    let hf = |h: &str| format!("Basis set\n    F.x2c-SVPall\n    F 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H {h}\nEnd of Basis\n");

    // HF along z: x and y must be equivalent (rotation about the bond), and everything matches PySCF.
    let along_z = invariants(&planes(Kind::Nuclear, &hf("0.0 0.0 1.733"), &[], "z"));
    close("z: plane x", along_z.top[0], 21294.155350297406, 1e-9);
    close("z: plane y", along_z.top[1], 21294.15535029739, 1e-9);
    close("z: plane z", along_z.top[2], 19360.306176963782, 1e-9);
    close("z: scalar plane", along_z.top[3], 50984000.70942227, 1e-9);
    close("z: triple product", along_z.triple, -2974275.9016006226, 1e-9);
    close("z: scalar trace", along_z.scalar_trace, -9072.87453436647, 1e-9);
    assert!(along_z.antisymmetry < 1e-12, "the vector planes must be antisymmetric ({})", along_z.antisymmetry);

    // A tilted bond puts weight on all three axes.
    let b = 1.733;
    let tilted = format!("{:.10} {:.10} {:.10}", 0.3 * b, 0.4 * b, 0.866025403784 * b);
    let tilted = invariants(&planes(Kind::Nuclear, &hf(&tilted), &[], "tilted"));
    close("tilted: plane x", tilted.top[0], 21120.080786913462, 1e-9);
    close("tilted: plane y", tilted.top[1], 20984.693208520493, 1e-9);
    close("tilted: plane z", tilted.top[2], 19843.702834462158, 1e-9);
    close("tilted: triple product", tilted.triple, -2974275.901601258, 1e-9);
    assert!(tilted.antisymmetry < 1e-12);

    // An atom is rotationally invariant: the three planes have the same spectrum.
    let atom = invariants(&planes(Kind::Nuclear, "Basis set\n    F.x2c-SVPall\n    F 0.0 0.0 0.0\nEnd of Basis\n", &[], "atom"));
    close("atom: y equals x", atom.top[1], atom.top[0], 1e-10);
    close("atom: z equals x", atom.top[2], atom.top[0], 1e-10);

    // (sigma.p) r (sigma.p) around the origin (PySCF int1e_sprsp with the common origin at 0): 12 planes,
    // plane 4 * component + piece. Planes 0, 5 and 10 vanish identically ((p x r_x p)_x = 0, ...), the
    // vector planes are antisymmetric, and the scalar planes 3, 7, 11 are symmetric. Reference values:
    // PySCF for F at (0.2, -0.3, 0.5), H at (0.7, 0.52, 2.1) Bohr, x2c-SVPall. Without the antisymmetric
    // placement 63 of the 64 non-zero triple products of these planes disagree with PySCF.
    let geometry = "Basis set\n    F.x2c-SVPall\n    F 0.2 -0.3 0.5\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H 0.7 0.52 2.1\nEnd of Basis\n";
    let dipole = planes(Kind::Dipole([0.0, 0.0, 0.0]), geometry, &[], "dipole");
    let m = scaled_planes(&dipole);
    let top = |k: usize| {
        let (values, _) = m[k].matmul(&m[k].view_transposed()).eigenvectors();
        values[values.len() - 1]
    };
    for (k, expected) in [(1, 6.776916645203631), (2, 6.429966817928909), (3, 314.3746613358478), (4, 6.776916645203634), (6, 6.433116336264591), (7, 703.9382195773136), (8, 6.429966817928905), (9, 6.433116336264593), (11, 1956.6010794678223)] {
        close(&format!("sprsp plane {k}"), top(k), expected, 1e-9);
    }
    for k in [0, 5, 10] {
        assert!(top(k).abs() < 1e-18, "plane {k} must vanish ({})", top(k));
    }
    for (k, expected) in [(3, 52.42268915910168), (7, -48.12039968067334), (11, 137.85912539475592)] {
        close(&format!("sprsp trace {k}"), trace(&m[k]), expected, 1e-9);
    }
    for k in [1, 2, 4, 6, 8, 9] {
        let defect = (0..m[k].rows()).flat_map(|i| (0..m[k].cols()).map(move |j| (i, j))).map(|(i, j)| (m[k][(i, j)] + m[k][(j, i)]).abs()).fold(0.0_f64, f64::max);
        assert!(defect < 1e-11, "vector plane {k} must be antisymmetric ({defect})");
    }
    for ((a, b, c), expected) in [
        ((3, 7, 11), -20791.27022227956),
        ((1, 4, 11), 287.15290027291474),
        ((2, 8, 11), 250.79063326859912),
        ((6, 9, 11), 243.11471533237167),
        ((1, 4, 7), -137.345586465172),
        ((2, 7, 8), -132.0325897181871),
        ((6, 7, 9), -132.13598818343874),
        ((1, 3, 4), 111.38562022252086),
    ] {
        close(&format!("sprsp triple ({a},{b},{c})"), triple(&m, a, b, c), expected, 1e-8);
    }

    // One p primitive (exponent 1) on Z = 9: M M^T eigenvalue (Z <r^-3>)^2 with <r^-3> = 2.12769216.
    let single = "/F.tst.Test.1s1p.1s1p.\nsingle primitives\nFLUORINE\n     9.0   1\n* s-type functions\n    1    1\n      1.00000000\n  1.0000000000\n* p-type functions\n    1    1\n      1.00000000\n  1.0000000000\n";
    let one = invariants(&planes(Kind::Nuclear, "Basis set\n    F.tst\n    F 0.0 0.0 0.0\nEnd of Basis\n", &[("tst", single.to_string())], "single"));
    for c in 0..3 {
        close(&format!("single primitive plane {c}"), one.top[c], 366.6929888837, 1e-9);
    }
}
