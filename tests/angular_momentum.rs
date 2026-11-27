// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The orbital angular momentum and linear momentum integrals:
// `A_k = <mu| ((r - O) x nabla)_k |nu>` (L = -i A) and `nabla_k = <mu| d/dk |nu>` (p = -i nabla), through
// `compute_one_electron_integrals` like every other operator.
//
// - antisymmetry, every component, contracted (general and segmented) and primitive;
// - exact angular momentum on one centre: with O at the nucleus, the eigenvalues of L_z are m = -l..l per shell and
//   L^2 is l(l + 1) on each shell, for l up to 3;
// - the gauge origin: A(O') = A(O) + (O - O') x nabla, which ties the two operators together;
// - nabla against finite differences of the overlap (moving one atom), which fixes its sign and value.

mod common;

use aedelsten::arrays::matrix::{ColumnFactor, Factor, Matrix, MatrixMath, MatrixTrait, MatrixTraitMut, SplitComplexMatrix};
use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator};

const WATER_ANO: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";
const WATER_SVP: &str = "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n";

fn context(name: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let dir = common::temp_path(&format!("angular_momentum_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx
}

/// The three planes (x, y, z) of a component operator, each `n x n`.
fn planes(ctx: &AedelstenContext, operator: OneElectronOperator<'_>, primitive: bool) -> [Matrix<f64>; 3] {
    let n = if primitive { ctx.get_n_prim_unique() } else { ctx.get_n_cont() };
    let mut out = vec![f64::NAN; 3 * n * n];
    ctx.compute_one_electron_integrals(&mut out, operator, primitive).unwrap();
    std::array::from_fn(|k| {
        let mut m = Matrix::<f64>::new(n, n);
        m.get_data_mut().copy_from_slice(&out[k * n * n..(k + 1) * n * n]);
        m
    })
}

fn overlap(ctx: &AedelstenContext) -> Matrix<f64> {
    let n = ctx.get_n_cont();
    let mut s = Matrix::<f64>::new(n, n);
    ctx.compute_one_electron_integrals(s.get_data_mut(), OneElectronOperator::overlap(), false).unwrap();
    s
}

fn largest(m: &Matrix<f64>) -> f64 {
    m.get_data().iter().fold(0.0_f64, |x, v| x.max(v.abs()))
}

#[test]
fn both_operators_are_antisymmetric_contracted_and_primitive() {
    for (name, input, contraction) in [("ano", WATER_ANO, ContractionType::General), ("svp", WATER_SVP, ContractionType::Segmented)] {
        let mut ctx = context(name, input, contraction);
        ctx.set_primitive_shell_pairs(None).unwrap();
        for primitive in [false, true] {
            for (label, operator) in [("A", OneElectronOperator::angular_momentum([0.3, -0.2, 0.5])), ("nabla", OneElectronOperator::nabla())] {
                for (k, m) in planes(&ctx, operator, primitive).iter().enumerate() {
                    let n = m.rows();
                    let asymmetry = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (m[(i, j)] + m[(j, i)]).abs()).fold(0.0, f64::max);
                    eprintln!("{name}, primitive {primitive}: {label}_{k}: max {:.3e}, |M + M^T| {asymmetry:.3e}", largest(m));
                    assert!(largest(m) > 1e-2, "{name} {label}_{k}: vacuous");
                    assert!(asymmetry < 1e-12 * largest(m), "{name}, primitive {primitive}: {label}_{k} not antisymmetric: {asymmetry}");
                }
            }
        }
    }
}

#[test]
fn the_angular_momentum_is_exact_on_one_centre() {
    // One atom with s, p, d and f shells (Cu x2c-SVPall, general contraction, spherical), its nucleus the gauge origin:
    // the AO basis of each shell is closed under rotations, so S^-1 A represents L exactly.
    let nucleus = [0.2, -0.1, 0.3];
    let input = format!("Basis set\n    Cu.x2c-SVPall\n    Cu1 {} {} {}\nEnd of Basis\n", nucleus[0], nucleus[1], nucleus[2]);
    let ctx = context("atom", &input, ContractionType::General);
    let n = ctx.get_n_cont();
    // Shells: 5 s, 3 p, 3 d, 1 f contractions.
    let shells: [(usize, usize); 4] = [(0, 5), (1, 3), (2, 3), (3, 1)];
    assert_eq!(n, shells.iter().map(|&(l, c)| c * (2 * l + 1)).sum::<usize>(), "spherical functions expected");
    let a = planes(&ctx, OneElectronOperator::angular_momentum(nucleus), false);

    // Loewdin: X = S^-1/2, K_k = X A_k X real antisymmetric, L_k = -i K_k Hermitian.
    let (values, vectors) = overlap(&ctx).eigenvectors();
    let mut scaled = vectors.clone();
    for (column, s) in scaled.iter_mut().zip(values.iter()) {
        column.iter_mut().for_each(|v| *v /= s.sqrt());
    }
    let x = scaled.matmul(&vectors.view_transposed());
    let k: [Matrix<f64>; 3] = std::array::from_fn(|c| x.matmul(&a[c]).matmul(&x));

    // L_z = -i K_z: eigenvalues m, (2l + 1) of them per contraction of each shell.
    let mut minus_k = k[2].clone();
    minus_k.get_data_mut().iter_mut().for_each(|v| *v = -*v);
    let l_z = SplitComplexMatrix::from_parts(Matrix::<f64>::new(n, n), minus_k);
    let eigenvalues = l_z.eigenvalues();
    let mut expected: Vec<f64> = shells.iter().flat_map(|&(l, c)| (0..c).flat_map(move |_| (-(l as i64)..=l as i64).map(|m| m as f64))).collect();
    expected.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let worst = eigenvalues.iter().zip(&expected).map(|(e, m)| (e - m).abs()).fold(0.0, f64::max);
    eprintln!("L_z eigenvalues {:?}, worst deviation {worst:.3e}", eigenvalues);
    assert!(worst < 1e-10, "L_z eigenvalues off by {worst}");

    // L^2 = -(K_x^2 + K_y^2 + K_z^2): l(l + 1) on each shell, as eigenvalues of the symmetric matrix.
    let mut l2 = Matrix::<f64>::new(n, n);
    for kc in &k {
        let square = kc.matmul(kc);
        l2.get_data_mut().iter_mut().zip(square.get_data()).for_each(|(o, v)| *o -= v);
    }
    let l2_values = l2.eigenvalues();
    let mut expected: Vec<f64> = shells.iter().flat_map(|&(l, c)| std::iter::repeat_n((l * (l + 1)) as f64, c * (2 * l + 1))).collect();
    expected.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let worst = l2_values.iter().zip(&expected).map(|(e, v)| (e - v).abs()).fold(0.0, f64::max);
    eprintln!("L^2 worst deviation {worst:.3e}");
    assert!(worst < 1e-10, "L^2 eigenvalues off by {worst}");
}

#[test]
fn moving_the_gauge_origin_adds_a_cross_product_with_nabla() {
    for (name, input, contraction) in [("ano", WATER_ANO, ContractionType::General), ("svp", WATER_SVP, ContractionType::Segmented)] {
        let ctx = context(name, input, contraction);
        let (o, o_prime) = ([0.1, -0.2, 0.3], [1.0, 0.5, -0.7]);
        let (a, a_prime, nabla) = (planes(&ctx, OneElectronOperator::angular_momentum(o), false), planes(&ctx, OneElectronOperator::angular_momentum(o_prime), false), planes(&ctx, OneElectronOperator::nabla(), false));
        let d: [f64; 3] = std::array::from_fn(|k| o[k] - o_prime[k]);
        for k in 0..3 {
            let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
            // A_k(O') = A_k(O) + d_k1 nabla_k2 - d_k2 nabla_k1.
            let worst = (0..a[k].get_data().len())
                .map(|i| (a_prime[k].get_data()[i] - a[k].get_data()[i] - d[k1] * nabla[k2].get_data()[i] + d[k2] * nabla[k1].get_data()[i]).abs())
                .fold(0.0, f64::max);
            let change = (0..a[k].get_data().len()).map(|i| (a_prime[k].get_data()[i] - a[k].get_data()[i]).abs()).fold(0.0, f64::max);
            eprintln!("{name}: A_{k}: change {change:.3e}, deviation {worst:.3e}");
            assert!(change > 1e-2, "{name}: the origin change does nothing");
            assert!(worst < 1e-12 * change, "{name}: A_{k}(O') - A_{k}(O) off by {worst}");
        }
    }
}

#[test]
fn nabla_is_minus_the_derivative_of_the_overlap_by_the_ket_centre() {
    // <mu| d/dk |nu> = -dS_mu,nu / dB_k for nu on atom B and mu elsewhere: move H1 by +-h along each axis. Water
    // ANO-RCC: O carries AOs 0..14, H1 14..19, H2 19..24.
    let h1 = [0.0, 1.4, -0.9];
    let water = |shift: [f64; 3]| {
        format!(
            "Basis set\n    O.x2c-SVPall\n    O1 0.0 0.0 0.1\nEnd of Basis\nBasis set\n    H.x2c-SVPall\n    H1 {} {} {}\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n",
            h1[0] + shift[0], h1[1] + shift[1], h1[2] + shift[2],
        )
    };
    let ctx = context("fd", &water([0.0; 3]), ContractionType::General);
    let n = ctx.get_n_cont();
    assert_eq!(n, 24);
    let nabla = planes(&ctx, OneElectronOperator::nabla(), false);
    let step = 1e-4;
    let (on_h1, elsewhere): (Vec<usize>, Vec<usize>) = ((14..19).collect(), (0..14).chain(19..24).collect());
    for k in 0..3 {
        let mut shift = [0.0; 3];
        shift[k] = step;
        let plus = overlap(&context("fd_plus", &water(shift), ContractionType::General));
        shift[k] = -step;
        let minus = overlap(&context("fd_minus", &water(shift), ContractionType::General));
        let (mut worst, mut scale) = (0.0_f64, 0.0_f64);
        for &nu in &on_h1 {
            for &mu in &elsewhere {
                let finite = -(plus[(mu, nu)] - minus[(mu, nu)]) / (2.0 * step);
                worst = worst.max((nabla[k][(mu, nu)] - finite).abs());
                scale = scale.max(finite.abs());
            }
        }
        eprintln!("nabla_{k}: max {scale:.3e}, finite-difference deviation {worst:.3e}");
        assert!(scale > 1e-2, "nabla_{k}: vacuous");
        assert!(worst < 1e-7, "nabla_{k} off the finite differences by {worst}");
    }
}

unsafe extern "C" {
    fn set_aedelsten_context_c(filename: *const std::ffi::c_char, contraction_type: i32, comm: i32) -> *mut std::ffi::c_void;
    fn free_aedelsten_context_c(ctx: *mut std::ffi::c_void);
    fn set_shell_pairs_c(ctx: *mut std::ffi::c_void, threshold: f64) -> i32;
    fn set_scratch_c(ctx: *mut std::ffi::c_void, size: i32, unit: i32) -> i32;
    fn aedelsten_query_i32_c(key: *const std::ffi::c_char) -> i32;
    fn compute_one_electron_component_count_c(operator: i32, order: i32) -> i32;
    fn compute_one_electron_integrals_c(
        ctx: *mut std::ffi::c_void,
        output: *mut f64,
        n: i32,
        operator: i32,
        n_planes: i32,
        origin_x: f64,
        origin_y: f64,
        origin_z: f64,
        order: i32,
        primitive: i32,
    ) -> i32;
}

#[test]
fn the_c_entry_points_count_the_components_and_reproduce_the_rust_integrals() {
    let path = common::temp_path("angular_momentum_ffi.inp");
    std::fs::write(&path, WATER_ANO).unwrap();
    let reference = context("ffi_reference", WATER_ANO, ContractionType::General);
    let n = reference.get_n_cont();
    let origin = [0.3, -0.2, 0.5];
    let key = |name: &str| unsafe { aedelsten_query_i32_c(std::ffi::CString::new(name).unwrap().as_ptr()) };
    let (angular_momentum, nabla, multipole, overlap) = (key("oneint_angular_momentum"), key("oneint_nabla"), key("oneint_moment"), key("oneint_overlap"));
    unsafe {
        assert_eq!(compute_one_electron_component_count_c(angular_momentum, 0), 3);
        assert_eq!(compute_one_electron_component_count_c(nabla, 5), 3);
        assert_eq!(compute_one_electron_component_count_c(multipole, 2), 6);
        assert_eq!(compute_one_electron_component_count_c(overlap, 0), 0);
        assert_eq!(compute_one_electron_component_count_c(-7, 0), -1);

        let inp = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        let ctx = set_aedelsten_context_c(inp.as_ptr(), ContractionType::General.get_c_int(), aedelsten_mpi::COMM_SINGLE_SENTINEL);
        assert!(!ctx.is_null());
        assert_eq!(set_shell_pairs_c(ctx, -1.0), 0);
        assert_eq!(set_scratch_c(ctx, 64, 1), 0);
        for (operator, rust) in [(angular_momentum, OneElectronOperator::angular_momentum(origin)), (nabla, OneElectronOperator::nabla())] {
            let mut expected = vec![0.0; 3 * n * n];
            reference.compute_one_electron_integrals(&mut expected, rust, false).unwrap();
            let mut out = vec![f64::NAN; 3 * n * n];
            let status = compute_one_electron_integrals_c(ctx, out.as_mut_ptr(), n as i32, operator, 3, origin[0], origin[1], origin[2], 0, 0);
            assert_eq!(status, 0);
            assert_eq!(out, expected);
            // One plane where three are made: refused, not overrun.
            let mut short = vec![0.0; n * n];
            assert_ne!(compute_one_electron_integrals_c(ctx, short.as_mut_ptr(), n as i32, operator, 1, origin[0], origin[1], origin[2], 0, 0), 0);
        }
        free_aedelsten_context_c(ctx);
    }
    let _ = std::fs::remove_file(&path);
}
