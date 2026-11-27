// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Correctness check for the MPI-partitioned stored ERI tensor (eri drivers.rs,
// compute_eri_stored_parallel) and the direct J/K build with block digestion
// (compute_fock_direct_from_data_parallel, fock_constructors' tile kernels and
// their `finish` before the all-reduce), unscreened and density-screened, the general-density build (one density and
// several in one walk), the active-space accumulator's T in one basis and two,
// the half-transformed X in one basis and two, partitioned and rank-local, and the two-electron gradient
// (compute_eri_gradient_from_data, DirectEriGradient), unscreened and with the pairwise threshold.
// Same procedure as mpi_oneint_check.rs:
//
//   cargo build --release --example mpi_eri_check                        # serial reference build
//   ./target/release/examples/mpi_eri_check > /tmp/reference.txt
//   cargo build --release --example mpi_eri_check --features mpi         # MPI build
//   mpirun -n 1 ./target/release/examples/mpi_eri_check > /tmp/n1.txt
//   mpirun -n 3 ./target/release/examples/mpi_eri_check > /tmp/n3.txt
//   diff /tmp/reference.txt /tmp/n1.txt   # sanity: -n 1 under the mpi feature still matches plain serial
//   diff /tmp/n1.txt /tmp/n3.txt          # the partitioned computation
//
// The tensors are compared exactly: each position is written by one rank and summed with zeros. The direct J/K
// matrices are sums of per-rank partial sums, so their last digits may differ between rank counts; they are
// printed to 10 significant digits.

use aedelsten::arrays::array3d::{Array3D, Array3DTrait, Array3DTraitMut};
use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, ERIAOSymmetry, ERIIndexOrder, ERIRangeOperator, TwoElectronTerm, TwoElectronTermKind};

fn dump(label: &str, buf: &[f64], digits: usize) {
    for (i, v) in buf.iter().enumerate() {
        println!("{label}[{i}] = {v:.digits$e}");
    }
}

fn main() {
    let mpi = aedelsten_mpi::Session::init(aedelsten_mpi::ThreadLevel::Funneled).expect("MPI init failed");
    let world = mpi.world().expect("the world communicator");
    let rank = aedelsten_mpi::rank();
    eprintln!("rank {rank} of {}", aedelsten_mpi::size());

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/mpi_eri_check.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &world).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();

    // Every rank makes every call (they are collective); rank 0 prints.
    let omega = 0.4;
    let cases: [(&str, ERIAOSymmetry, ERIRangeOperator); 7] = [
        ("s1", ERIAOSymmetry::S1, ERIRangeOperator::FullRange),
        ("s4_row", ERIAOSymmetry::S4 { order: ERIIndexOrder::RowMajor }, ERIRangeOperator::FullRange),
        ("s4_col", ERIAOSymmetry::S4 { order: ERIIndexOrder::ColumnMajor }, ERIRangeOperator::FullRange),
        ("s8_row", ERIAOSymmetry::S8 { order: ERIIndexOrder::RowMajor }, ERIRangeOperator::FullRange),
        ("s8_col", ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor }, ERIRangeOperator::FullRange),
        ("s8_col_sr", ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor }, ERIRangeOperator::ShortRange { omega }),
        ("s8_col_lr", ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor }, ERIRangeOperator::LongRange { omega }),
    ];
    if rank == 0 {
        println!("n = {n}");
    }
    for (label, symmetry, range) in cases {
        let len = match symmetry {
            ERIAOSymmetry::S1 => n * n * n * n,
            ERIAOSymmetry::S4 { .. } => aedelsten::compute_s4_size(n),
            ERIAOSymmetry::S8 { .. } => aedelsten::compute_s8_size(n),
        };
        // Garbage in the buffer: the tensors zero themselves.
        let mut eri = vec![f64::NAN; len];
        ctx.compute_eri_stored(&mut eri, symmetry, range).unwrap();
        if rank == 0 {
            dump(label, &eri, 14);
        }
    }

    ctx.set_eri_data(None).unwrap();
    let mut p = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            p[(i, j)] = 0.1 + 0.05 * (((i.min(j) * 7 + i.max(j) * 3) % 11) as f64);
        }
    }
    let (mut j, mut k) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    ctx.compute_fock_direct_restricted(p.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap();
    if rank == 0 {
        dump("direct_j", j.get_data(), 10);
        dump("direct_k", k.get_data(), 10);
    }

    // The sorted walk with density-weighted screening (and its partition by the quartets each row keeps), for a
    // density that decays with distance (the overlap matrix) and one 1e-5 times smaller (a difference density).
    let mut s = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut s, aedelsten::OneElectronOperator::overlap(), false).unwrap();
    for (label, scale) in [("screened", 1.0), ("screened_difference", 1e-5)] {
        let d: Vec<f64> = s.iter().map(|x| scale * x).collect();
        ctx.compute_fock_direct_restricted(&d, j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, ERIRangeOperator::FullRange, Some(1e-10)).unwrap();
        if rank == 0 {
            dump(&format!("{label}_j"), j.get_data(), 10);
            dump(&format!("{label}_k"), k.get_data(), 10);
        }
    }

    // J and K of a non-symmetric density (compute_fock_direct_general): J from its symmetric part, K in full.
    let mut q = Matrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            q[(i, j)] = 0.3 * (((i * 7 + j * 3 + 2) % 11) as f64) - 1.1 + 0.07 * (i as f64) - 0.13 * (j as f64);
        }
    }
    ctx.compute_fock_direct_general(q.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap();
    if rank == 0 {
        dump("general_j", j.get_data(), 10);
        dump("general_k", k.get_data(), 10);
    }

    // Two non-symmetric densities in one walk (compute_fock_direct_general_multi): q and another.
    let mut stacked = q.get_data().to_vec();
    stacked.extend((0..n * n).map(|x| 0.2 * (((x * 5 + 1) % 13) as f64) - 1.3 + 0.01 * ((x % n) as f64)));
    let (mut j_multi, mut k_multi) = (vec![0.0; 2 * n * n], vec![0.0; 2 * n * n]);
    ctx.compute_fock_direct_general_multi(&stacked, &mut j_multi, &mut k_multi, 2, 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap();
    if rank == 0 {
        dump("general_multi_j", &j_multi, 10);
        dump("general_multi_k", &k_multi, 10);
    }

    // The active-space accumulator: inactive/active J/K and M from one walk, and T from M.
    let n_act = 3;
    let mut c = Matrix::<f64>::new(n, n_act);
    for mu in 0..n {
        for u in 0..n_act {
            c[(mu, u)] = 0.2 * (((mu * 5 + u * 3 + 1) % 7) as f64) - 0.55;
        }
    }
    let new = || Matrix::<f64>::new(n, n);
    let (mut j_i, mut k_i, mut j_a, mut k_a) = (new(), new(), new(), new());
    let pairs = n_act * (n_act + 1) / 2;
    let mut coulomb = vec![0.0; n * (n + 1) / 2 * pairs];
    ctx.compute_fock_direct_active_space(
        p.get_data(), p.get_data(), j_i.get_data_mut(), k_i.get_data_mut(), j_a.get_data_mut(), k_a.get_data_mut(),
        1.0, -0.5, 1.0, -0.5, c.get_data(), n_act, &mut coulomb, None, None,
    )
    .unwrap();
    let mut t = Array3D::<f64>::new(n_act, pairs, n);
    ctx.transform_pair_coulomb_one_index(&coulomb, pairs, c.get_data(), n_act, t.get_data_mut()).unwrap();
    if rank == 0 {
        dump("active_k", k_a.get_data(), 10);
        dump("active_m", &coulomb, 10);
        dump("active_t", t.get_data(), 10);
    }

    // The mixed-basis active-space accumulator: J/K of the non-symmetric density and M in two bases from one walk.
    let mut c_bra = Matrix::<f64>::new(n, n_act);
    for mu in 0..n {
        for u in 0..n_act {
            c_bra[(mu, u)] = 0.2 * (((mu * 3 + u * 5 + 2) % 7) as f64) - 0.6;
        }
    }
    let mut m_mixed = vec![0.0; n * (n + 1) / 2 * n_act * n_act];
    ctx.compute_fock_direct_mixed_active_space(q.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, c_bra.get_data(), c.get_data(), n_act, &mut m_mixed, None, None).unwrap();
    if rank == 0 {
        dump("mixed_k", k.get_data(), 10);
        dump("mixed_m", &m_mixed, 10);
    }

    // The half-transformed integrals: one basis (its symmetric fill in `finish`, before the all-reduce) and two, the
    // second also rank-local (every rank the whole X, no communication; rank 0's printed).
    let mut x = vec![0.0; n_act * n * n_act * n];
    ctx.compute_eri_half_transformed(c.get_data(), n_act, c.get_data(), n_act, &mut x, None, None, true).unwrap();
    if rank == 0 {
        dump("half_x", &x, 10);
    }
    let mut x_mixed = vec![0.0; 2 * n * n_act * n];
    ctx.compute_eri_half_transformed(&c_bra.get_data()[..2 * n], 2, c.get_data(), n_act, &mut x_mixed, None, None, true).unwrap();
    if rank == 0 {
        dump("half_x_mixed", &x_mixed, 10);
    }
    ctx.compute_eri_half_transformed(&c_bra.get_data()[..2 * n], 2, c.get_data(), n_act, &mut x_mixed, None, None, false).unwrap();
    if rank == 0 {
        dump("half_x_mixed", &x_mixed, 10);
    }

    // The two-electron gradient of an RHF-like energy (J of p, K of the non-symmetric q), every centre, and one with
    // the others frozen.
    let terms = [
        TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: p.get_data(), right: p.get_data(), coeff: 1.0 },
        TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: q.get_data(), right: q.get_data(), coeff: -0.5 },
    ];
    let mut gradient = vec![0.0; 3 * ctx.get_n_gradient_centres()];
    for (label, threshold) in [("gradient", None), ("screened_gradient", Some(1e-10))] {
        ctx.compute_two_electron_gradient(&mut gradient, &terms, ERIRangeOperator::FullRange, threshold, None).unwrap();
        if rank == 0 {
            dump(label, &gradient, 10);
        }
    }
    let only_centre_1: Vec<bool> = (0..ctx.get_n_gradient_centres()).map(|centre| centre == 1).collect();
    ctx.compute_two_electron_gradient(&mut gradient, &terms, ERIRangeOperator::FullRange, None, Some(&only_centre_1)).unwrap();
    if rank == 0 {
        dump("gradient_centre_1_active", &gradient, 10);
    }

    // The context's work over two groups at once (the even and the odd ranks), then over the world again: each group's
    // stored tensor equals the world's exactly, its direct J/K to rounding, on every rank of either group.
    let mut whole = vec![0.0; n * n * n * n];
    ctx.compute_eri_stored(&mut whole, ERIAOSymmetry::S1, ERIRangeOperator::FullRange).unwrap();
    let (mut j_whole, mut k_whole) = (Matrix::<f64>::new(n, n), Matrix::<f64>::new(n, n));
    ctx.compute_fock_direct_restricted(p.get_data(), j_whole.get_data_mut(), k_whole.get_data_mut(), 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap();
    // The half transform by each node's ranks into one X in their shared memory (one basis, two bases, a screened
    // walk): as the summed X to rounding.
    let node = world.node().unwrap();
    let previous = ctx.set_communicator(&node);
    let mut node_failed = [0.0];
    for (left, m_left, threshold) in [(c.get_data(), n_act, None), (&c_bra.get_data()[..2 * n], 2, None), (c.get_data(), n_act, Some(1e-10))] {
        let len = m_left * n * n_act * n;
        let mut expected = vec![0.0; len];
        ctx.compute_eri_half_transformed(left, m_left, c.get_data(), n_act, &mut expected, threshold, None, false).unwrap();
        let mut shared = node.shared_array(len).unwrap();
        ctx.compute_eri_half_transformed_on_node(left, m_left, c.get_data(), n_act, shared.as_mut_slice(), threshold, None).unwrap();
        let error = shared.as_slice().iter().zip(&expected).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        node_failed[0] += (error > 1e-12) as i32 as f64;
    }
    ctx.set_communicator(&previous);
    world.sum(&mut node_failed).unwrap();
    if rank == 0 {
        println!("ranks whose X on the node differs: {}", node_failed[0]);
    }

    let halves = world.split(Some(world.rank() % 2), world.rank()).unwrap().expect("every rank joins a group");
    let previous = ctx.set_communicator(&halves);
    let mut split = vec![f64::NAN; n * n * n * n];
    ctx.compute_eri_stored(&mut split, ERIAOSymmetry::S1, ERIRangeOperator::FullRange).unwrap();
    ctx.compute_fock_direct_restricted(p.get_data(), j.get_data_mut(), k.get_data_mut(), 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap();
    let largest = |a: &[f64], b: &[f64]| a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    let jk_error = largest(j.get_data(), j_whole.get_data()).max(largest(k.get_data(), k_whole.get_data()));
    let mut failed = [((split != whole) || jk_error > 1e-12) as i32 as f64];
    ctx.set_communicator(&previous);
    world.sum(&mut failed).unwrap();
    if rank == 0 {
        println!("ranks whose group differs from the world: {}", failed[0]);
    }

    aedelsten_mpi::barrier();
}
