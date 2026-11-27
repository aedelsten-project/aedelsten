// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Timing probe for an environment over the ranks. A QM input file (its basis blocks), around which are placed
// embedding AIMPs (Na, on shells 10, 12, ... bohr out), point multipoles to second order and polarizable sites
// (each on further shells, an even spiral on each, about 4 bohr between neighbours). Times, with rank 0's time in
// MPI: the multipole operator, the AIMP Coulomb and exchange operators, the density-contracted gradient of each, and
// the induction (cold, then warm-started).
//
//   cargo build --release --example embedding_probe --features mpi
//   mpirun -n 8 target/release/examples/embedding_probe <input> <aimps> <multipoles> <polarizable>   (general contraction)

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator, PolarizationOptions, SolverOptions};
use std::time::Instant;

/// `count` points on shells from radius `r0` out, 2 bohr apart, an even spiral on each.
fn shells(count: usize, r0: f64) -> (Vec<[f64; 3]>, f64) {
    let (mut points, mut r) = (Vec::with_capacity(count), r0);
    while points.len() < count {
        let on_shell = ((4.0 * std::f64::consts::PI * r * r / 16.0) as usize).min(count - points.len());
        for i in 0..on_shell {
            let z = 1.0 - (2.0 * i as f64 + 1.0) / on_shell as f64;
            let (rho, phi) = ((1.0 - z * z).sqrt(), 2.399_963 * i as f64 + r);
            points.push([r * rho * phi.cos(), r * rho * phi.sin(), r * z]);
        }
        r += 2.0;
    }
    (points, r)
}

fn main() {
    let mpi = aedelsten_mpi::Session::init(aedelsten_mpi::ThreadLevel::Funneled).expect("MPI init failed");
    let world = mpi.world().expect("the world communicator");
    let (rank, size) = (aedelsten_mpi::rank(), aedelsten_mpi::size());
    let args: Vec<String> = std::env::args().collect();
    let count = |k: usize| -> usize { args[k].parse().unwrap() };
    let (n_aimps, n_multipoles, n_polarizable) = (count(2), count(3), count(4));

    let (aimps, r) = shells(n_aimps, 10.0);
    let (multipoles, r) = shells(n_multipoles, r);
    let (polarizable, _) = shells(n_polarizable, r);
    let mut input = std::fs::read_to_string(&args[1]).unwrap();
    if n_aimps > 0 {
        input.push_str("Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n");
        for (i, [x, y, z]) in aimps.iter().enumerate() {
            input.push_str(&format!("    Na{} {x} {y} {z}\n", i + 1));
        }
        input.push_str("End of Basis\n");
    }
    let mut xfield = format!("{n_multipoles} Bohr 2\n");
    for (i, [x, y, z]) in multipoles.iter().enumerate() {
        let q = if i % 2 == 0 { 0.4 } else { -0.4 };
        xfield.push_str(&format!("{x} {y} {z} {q} 0.05 -0.02 0.03 0.01 0.0 -0.01 0.02 0.0 -0.03\n"));
    }
    let mut pol = format!("{n_polarizable} Bohr -1 2 1\n");
    for (i, [x, y, z]) in polarizable.iter().enumerate() {
        let a = 1.0 + 0.1 * (i % 7) as f64;
        pol.push_str(&format!("{} {x} {y} {z} {a} 0.0 0.0 {a} 0.0 {a}\n", i + 1));
    }
    let stem = std::env::temp_dir().join(format!("embedding_probe_{}_{rank}", std::process::id()));
    let files = [(stem.with_extension("inp"), input), (stem.with_extension("mp.xfield"), xfield), (stem.with_extension("pol.xfield"), pol)];
    for (path, contents) in &files {
        std::fs::write(path, contents).unwrap();
    }
    let mut ctx = AedelstenContext::new(files[0].0.to_str().unwrap(), ContractionType::General, &world).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((256, arrays::scratch::ByteUnit::MiB)).unwrap();
    if n_multipoles > 0 {
        ctx.set_external_multipoles(files[1].0.to_str().unwrap()).unwrap();
    }
    if n_aimps > 0 {
        ctx.set_embedding_aimp_operators().unwrap();
    }
    if n_polarizable > 0 {
        ctx.set_polarizable_sites(files[2].0.to_str().unwrap()).unwrap();
        ctx.set_polarization(PolarizationOptions { solver: SolverOptions { threshold: 1e-8, ..SolverOptions::default() }, ..PolarizationOptions::default() }).unwrap();
    }
    for (path, _) in &files {
        std::fs::remove_file(path).ok();
    }
    let n = ctx.get_n_cont();
    if rank == 0 {
        println!("{}: n = {n}, {n_aimps} AIMPs, {n_multipoles} multipoles, {n_polarizable} polarizable sites, {size} ranks", args[1]);
    }

    let time = |label: &str, f: &mut dyn FnMut()| {
        aedelsten_mpi::barrier();
        let (start, mpi) = (Instant::now(), aedelsten_mpi::collective_seconds());
        f();
        aedelsten_mpi::barrier();
        if rank == 0 {
            println!("  {label:28} wall {:7.3} s, in MPI {:6.3} s", start.elapsed().as_secs_f64(), aedelsten_mpi::collective_seconds() - mpi);
        }
    };
    let density: Vec<f64> = (0..n * n).map(|idx| 0.05 / (1.0 + (idx % n) as f64 + (idx / n) as f64)).collect();
    let mut operators: Vec<(&str, OneElectronOperator<'_>)> = Vec::new();
    if n_multipoles > 0 {
        operators.push(("multipoles", ctx.get_external_multipole_operator().unwrap()));
    }
    if n_aimps > 0 {
        operators.push(("AIMP Coulomb", ctx.get_embedding_aimp_coulomb_operator().unwrap()));
        operators.push(("AIMP exchange", ctx.get_embedding_aimp_exchange_operator().unwrap()));
    }
    let n_centres = ctx.get_n_gradient_centres();
    for (label, operator) in operators {
        let mut matrix = vec![0.0; n * n];
        time(&format!("{label} operator"), &mut || ctx.compute_one_electron_integrals(&mut matrix, operator.clone(), false).unwrap());
        let mut gradient = vec![0.0; 3 * n_centres];
        time(&format!("{label} gradient"), &mut || ctx.compute_one_electron_gradient(&mut gradient, &density, operator.clone(), None).unwrap());
    }
    if n_polarizable > 0 {
        let mut v_ind = vec![0.0; n * n];
        time("induction, cold", &mut || {
            ctx.compute_induction(&density, &mut v_ind).unwrap();
        });
        time("induction, warm", &mut || {
            ctx.compute_induction(&density, &mut v_ind).unwrap();
        });
    }
}
