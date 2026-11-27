// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Timing probe for compute_eri_half_transformed. A cluster of water
// molecules in ANO-RCC [3s2p1d/2s1p] (cc-pVDZ-sized, d shells) or of H2 molecules in cc-pVDZ, segmented in x2c-SVPall
// (a segmented set, the same sizes: O [3s2p1d], H [2s1p]); the direct restricted
// J/K walk against the half transform, and the screen with orbitals localized on the first quarter of the basis.
//
//   cargo run --release --example half_transform_probe -- <water|h2> <molecules> <general|segmented> <m,...> [localized]
//   mpirun -n 4 target/release/examples/half_transform_probe ...   (built with --features mpi)

use aedelsten::arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, ERIRangeOperator};
use std::time::Instant;

// The peak resident set of this process (getrusage; macOS reports ru_maxrss in bytes, at byte 32 of the struct).
fn peak_bytes() -> i64 {
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut i64) -> i32;
    }
    let mut usage = [0i64; 18];
    unsafe { getrusage(0, usage.as_mut_ptr()) };
    usage[4]
}

fn main() {
    let mpi = aedelsten_mpi::Session::init(aedelsten_mpi::ThreadLevel::Funneled).expect("MPI init failed");
    let world = mpi.world().expect("the world communicator");
    let rank = aedelsten_mpi::rank();
    let args: Vec<String> = std::env::args().collect();
    let water = args[1] == "water";
    let waters: usize = args[2].parse().unwrap();
    let contraction = if args[3] == "segmented" { ContractionType::Segmented } else { ContractionType::General };
    let ms: Vec<usize> = args[4].split(',').map(|m| m.parse().unwrap()).collect();
    let localized = args.get(5).is_some_and(|a| a == "localized");

    // Molecules on a cubic grid 5.6 bohr apart.
    let side = (waters as f64).cbrt().ceil() as usize;
    let (mut o, mut h) = (String::new(), String::new());
    for w in 0..waters {
        let (x, y, z) = ((w % side) as f64 * 5.6, ((w / side) % side) as f64 * 5.6, (w / side / side) as f64 * 5.6);
        if water {
            o.push_str(&format!("    O{} {x:.4} {y:.4} {:.4}\n", w + 1, z + 0.12));
            h.push_str(&format!("    H{} {x:.4} {:.4} {:.4}\n", 2 * w + 1, y + 1.43, z - 0.96));
            h.push_str(&format!("    H{} {x:.4} {:.4} {:.4}\n", 2 * w + 2, y - 1.43, z - 0.96));
        } else {
            h.push_str(&format!("    H{} {x:.4} {y:.4} {:.4}\n", 2 * w + 1, z + 0.7));
            h.push_str(&format!("    H{} {x:.4} {y:.4} {:.4}\n", 2 * w + 2, z - 0.7));
        }
    }
    let segmented = matches!(contraction, ContractionType::Segmented);
    let input = match (water, segmented) {
        (true, false) => format!("Basis set\n    O.ANO-RCC...3s2p1d\n{o}End of Basis\nBasis set\n    H.ANO-RCC...2s1p\n{h}End of Basis\n"),
        (true, true) => format!("Basis set\n    O.x2c-SVPall\n{o}End of Basis\nBasis set\n    H.x2c-SVPall\n{h}End of Basis\n"),
        (false, false) => format!("Basis set\n    H.cc-pVDZ\n{h}End of Basis\n"),
        (false, true) => format!("Basis set\n    H.x2c-SVPall\n{h}End of Basis\n"),
    };
    let path = std::env::temp_dir().join(format!("half_transform_probe_{}_{rank}.inp", std::process::id()));
    std::fs::write(&path, input).unwrap();
    let mut ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &world).unwrap();
    std::fs::remove_file(&path).ok();
    ctx.set_shell_pairs(Some(1e-13)).unwrap();
    ctx.set_scratch((256, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(Some(1e-12)).unwrap();
    let n = ctx.get_n_cont();
    let report = |label: &str, seconds: f64| {
        if rank == 0 {
            println!("{waters} {}, n = {n}, {}, {} ranks: {label}: {seconds:.2} s", args[1], args[3], aedelsten_mpi::size());
        }
    };
    // Per rank: the call's walk, finish and all-reduce (the coarse spans' growth over the call) and the peak memory.
    let spans = || -> Vec<(&'static str, f64)> { aedelsten::timing_report().iter().map(|s| (s.get_label(), s.get_seconds())).collect() };
    let split = |before: &[(&'static str, f64)], label: &str| {
        let after = spans();
        let get = |list: &[(&'static str, f64)], key: &str| list.iter().find(|(l, _)| *l == key).map_or(0.0, |(_, t)| *t);
        let parts: Vec<String> = ["eri::direct::rows", "eri::direct::finish", "eri::direct::allreduce"]
            .iter()
            .map(|key| format!("{} {:.2} s", key.trim_start_matches("eri::direct::"), get(&after, key) - get(before, key)))
            .collect();
        eprintln!("  rank {rank}: {label}: {}, peak {:.2} GB", parts.join(", "), peak_bytes() as f64 / 1e9);
    };
    let time = |f: &mut dyn FnMut()| {
        aedelsten_mpi::barrier();
        let before = spans();
        let start = Instant::now();
        f();
        let own = start.elapsed().as_secs_f64();
        split(&before, &format!("{own:.2} s"));
        aedelsten_mpi::barrier();
        start.elapsed().as_secs_f64()
    };

    let mut p = Matrix::<f64>::new(n, n);
    for (j, column) in p.iter_mut().enumerate() {
        for (i, p_ij) in column.iter_mut().enumerate() {
            *p_ij = 0.01 * (((i * 7 + j * 7 + 3) % 11) as f64) * (-0.001 * (i as f64 - j as f64).abs()).exp();
        }
    }
    let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    let t = time(&mut || ctx.compute_fock_direct_restricted(p.get_data(), &mut j, &mut k, 1.0, -0.5, ERIRangeOperator::FullRange, Some(1e-12)).unwrap());
    report("restricted J/K", t);

    for &m in &ms {
        let mut c = Matrix::<f64>::new(n, m);
        for (u, column) in c.iter_mut().enumerate() {
            for (mu, c_mu) in column.iter_mut().enumerate() {
                *c_mu = 0.2 * (((mu * 5 + u * 3 + 1) % 7) as f64) - 0.55;
            }
        }
        if localized {
            // Weight on the first quarter of the basis only (the first quarter of the waters).
            for column in c.iter_mut() {
                column[n / 4..].fill(0.0);
            }
        }
        let mut x = vec![0.0; m * n * m * n];
        let thresholds: &[Option<f64>] = if localized { &[None, Some(1e-10)] } else { &[None] };
        for &threshold in thresholds {
            let t = time(&mut || ctx.compute_eri_half_transformed(c.get_data(), m, c.get_data(), m, &mut x, threshold, Some(1e-12), true).unwrap());
            report(&format!("half transform, one basis, m = {m}, threshold {threshold:?}"), t);
        }
        if !localized {
            // Each node's ranks into one X in their shared memory, one basis and a quarter of the orbitals on the left.
            let node = world.node().unwrap();
            let previous = ctx.set_communicator(&node);
            let mut shared = node.shared_array(m * n * m * n).unwrap();
            let t = time(&mut || ctx.compute_eri_half_transformed_on_node(c.get_data(), m, c.get_data(), m, shared.as_mut_slice(), None, Some(1e-12)).unwrap());
            report(&format!("half transform on the node, one basis, m = {m}"), t);
            let quarter = (m / 4).max(1);
            let mut batch = vec![0.0; quarter * n * m * n];
            ctx.set_communicator(&previous);
            let t = time(&mut || ctx.compute_eri_half_transformed(&c.get_data()[..quarter * n], quarter, c.get_data(), m, &mut batch, None, Some(1e-12), true).unwrap());
            report(&format!("half transform, two bases, {quarter} of m = {m}"), t);
            ctx.set_communicator(&node);
            let t = time(&mut || ctx.compute_eri_half_transformed_on_node(&c.get_data()[..quarter * n], quarter, c.get_data(), m, &mut shared.as_mut_slice()[..quarter * n * m * n], None, Some(1e-12)).unwrap());
            report(&format!("half transform on the node, two bases, {quarter} of m = {m}"), t);
            ctx.set_communicator(&previous);
        }
        if !localized {
            // Batches of sigma that cover the basis, one basis and (a quarter of m on the left) two: X's columns in
            // `batches` walks, each batch of whole shells of about n / batches functions; checked against the whole X.
            let mut shells = vec![aedelsten::ShellInfo::default(); ctx.get_n_shells()];
            ctx.get_shells(&mut shells).unwrap();
            let mut starts: Vec<usize> = shells.iter().map(|s| s.first_ao).chain([n]).collect();
            starts.sort_unstable();
            for batches in [1, 4, 16] {
                let mut bounds: Vec<usize> = (0..batches).map(|k| *starts.iter().find(|&&f| f * batches >= k * n).unwrap()).chain([n]).collect();
                bounds.dedup();
                let widest = bounds.windows(2).map(|w| w[1] - w[0]).max().unwrap();
                let mut x_sigma = vec![0.0; m * n * m * widest];
                let mut difference = 0.0_f64;
                let t = time(&mut || {
                    for w in bounds.windows(2) {
                        let part = &mut x_sigma[..m * n * m * (w[1] - w[0])];
                        ctx.compute_eri_half_transformed_sigma(c.get_data(), m, c.get_data(), m, w[0], w[1], part, None, Some(1e-12), true).unwrap();
                        let whole = &x[m * n * m * w[0]..m * n * m * w[1]];
                        difference = part.iter().zip(whole).fold(difference, |d, (a, b)| d.max((a - b).abs()));
                    }
                });
                report(&format!("half transform by sigma, one basis, m = {m}, {} batches (difference from the whole X {difference:.1e})", bounds.len() - 1), t);
                // On the node: the node's ranks into one shared batch.
                let node = world.node().unwrap();
                let previous = ctx.set_communicator(&node);
                let mut shared = node.shared_array(m * n * m * widest).unwrap();
                let mut node_difference = 0.0_f64;
                let t = time(&mut || {
                    for w in bounds.windows(2) {
                        let part = &mut shared.as_mut_slice()[..m * n * m * (w[1] - w[0])];
                        ctx.compute_eri_half_transformed_sigma_on_node(c.get_data(), m, c.get_data(), m, w[0], w[1], part, None, Some(1e-12)).unwrap();
                        let whole = &x[m * n * m * w[0]..m * n * m * w[1]];
                        node_difference = part.iter().zip(whole).fold(node_difference, |d, (a, b)| d.max((a - b).abs()));
                    }
                });
                ctx.set_communicator(&previous);
                report(&format!("half transform by sigma on the node, one basis, m = {m}, {} batches (difference from the whole X {node_difference:.1e})", bounds.len() - 1), t);
                let quarter = (m / 4).max(1);
                let mut x_two = vec![0.0; quarter * n * m * widest];
                let t = time(&mut || {
                    for w in bounds.windows(2) {
                        let part = &mut x_two[..quarter * n * m * (w[1] - w[0])];
                        ctx.compute_eri_half_transformed_sigma(&c.get_data()[..quarter * n], quarter, c.get_data(), m, w[0], w[1], part, None, Some(1e-12), true).unwrap();
                    }
                });
                report(&format!("half transform by sigma, two bases, {quarter} of m = {m}, {} batches", bounds.len() - 1), t);
            }
        }
        if !localized && m <= 10 {
            // The general path, for comparison: the same coefficients in a second buffer.
            let copy = c.clone();
            let t = time(&mut || ctx.compute_eri_half_transformed(c.get_data(), m, copy.get_data(), m, &mut x, None, Some(1e-12), true).unwrap());
            report(&format!("half transform, two bases, m = {m}"), t);
        }
    }
}
