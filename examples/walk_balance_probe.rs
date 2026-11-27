// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Timing probe for the direct walks' split over the ranks. For an input file: the restricted J/K
// walk, the active-space walk (J/K of two densities and M) and the half transform, each timed per rank (the
// `eri::direct::rows` span: the rank's own walk) with its wall time and the max/mean over the ranks.
//
//   cargo build --release --example walk_balance_probe --features mpi
//   mpirun -n 8 target/release/examples/walk_balance_probe <input> <n_active> [repeats]   (general contraction)

use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, ERIRangeOperator};
use std::time::Instant;

fn main() {
    let mpi = aedelsten_mpi::Session::init(aedelsten_mpi::ThreadLevel::Funneled).expect("MPI init failed");
    let world = mpi.world().expect("the world communicator");
    let (rank, size) = (aedelsten_mpi::rank(), aedelsten_mpi::size());
    let args: Vec<String> = std::env::args().collect();
    let n_active: usize = args[2].parse().unwrap();
    let repeats: usize = args.get(3).map_or(1, |r| r.parse().unwrap());

    let mut ctx = AedelstenContext::new(&args[1], ContractionType::General, &world).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((1024, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(None).unwrap();
    let n = ctx.get_n_cont();
    if rank == 0 {
        println!("{}: n = {n}, {} ranks, {n_active} active", args[1], size);
    }

    let rows_seconds = || aedelsten::timing_report().iter().find(|s| s.get_label() == "eri::direct::rows").map_or(0.0, |s| s.get_seconds());
    let time = |label: &str, f: &mut dyn FnMut()| {
        for _ in 0..repeats {
            aedelsten_mpi::barrier();
            let (before, start) = (rows_seconds(), Instant::now());
            f();
            let mut own = vec![0.0; size];
            own[rank] = rows_seconds() - before;
            aedelsten_mpi::barrier();
            let wall = start.elapsed().as_secs_f64();
            aedelsten_mpi::allreduce_sum_f64(&mut own).unwrap();
            if rank == 0 {
                let (max, mean) = (own.iter().fold(0.0_f64, |m, &t| m.max(t)), own.iter().sum::<f64>() / size as f64);
                let ranks: Vec<String> = own.iter().map(|t| format!("{t:.2}")).collect();
                println!("{label}: wall {wall:.2} s, walk per rank [{}] s, sum {:.2} s, max/mean {:.3}", ranks.join(" "), mean * size as f64, max / mean);
            }
        }
    };

    let mut p = Matrix::<f64>::new(n, n);
    for (j, column) in p.iter_mut().enumerate() {
        for (i, p_ij) in column.iter_mut().enumerate() {
            *p_ij = 0.01 * (((i * 7 + j * 7 + 3) % 11) as f64) * (-0.001 * (i as f64 - j as f64).abs()).exp();
        }
    }
    let mut c = Matrix::<f64>::new(n, n_active);
    for (u, column) in c.iter_mut().enumerate() {
        for (mu, c_mu) in column.iter_mut().enumerate() {
            *c_mu = 0.2 * (((mu * 5 + u * 3 + 1) % 7) as f64) - 0.55;
        }
    }
    let new = || vec![0.0; n * n];
    let (mut j, mut k, mut j_a, mut k_a) = (new(), new(), new(), new());

    time("restricted J/K", &mut || ctx.compute_fock_direct_restricted(p.get_data(), &mut j, &mut k, 1.0, -0.5, ERIRangeOperator::FullRange, None).unwrap());

    let pairs = n_active * (n_active + 1) / 2;
    let mut m = vec![0.0; n * (n + 1) / 2 * pairs];
    time("active space", &mut || {
        ctx.compute_fock_direct_active_space(p.get_data(), p.get_data(), &mut j, &mut k, &mut j_a, &mut k_a, 1.0, -0.5, 1.0, -0.5, c.get_data(), n_active, &mut m, None, None)
            .unwrap()
    });

    let mut x = vec![0.0; n_active * n * n_active * n];
    time("half transform", &mut || ctx.compute_eri_half_transformed(c.get_data(), n_active, c.get_data(), n_active, &mut x, None, None, true).unwrap());
}
