// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Correctness check for the induced-dipole embedding under MPI (the field passes over runs of sites handed out over the
// ranks, each summed): compute_induction (energy, dipoles, operator), compute_induction_response,
// compute_induction_states, compute_effective_external_field and compute_electronic_field, for water in ANO-RCC and 60
// polarizable sites around it; and an environment of 40 embedding AIMPs and 300 point multipoles to second order around
// water (one-electron pieces of shell pairs and sites handed out over the ranks): their operators and
// density-contracted gradients. Same procedure as mpi_eri_check.rs:
//
//   cargo build --release --example mpi_embedding_check                  # serial reference build
//   ./target/release/examples/mpi_embedding_check > /tmp/reference.txt
//   cargo build --release --example mpi_embedding_check --features mpi   # MPI build
//   mpirun -n 3 ./target/release/examples/mpi_embedding_check > /tmp/n3.txt
//   diff /tmp/reference.txt /tmp/n3.txt
//   mpirun -n 3 ./target/release/examples/mpi_embedding_check resident > /tmp/n3r.txt   # the field integrals kept,
//   diff /tmp/reference.txt /tmp/n3r.txt                                                # each rank its share
//   mpirun -n 3 ./target/release/examples/mpi_embedding_check partial > /tmp/n3p.txt    # some kept, different counts
//   diff /tmp/reference.txt /tmp/n3p.txt                                                # on each rank, the rest
//                                                                                       # evaluated
//
// The results are sums of per-rank partial sums, so their last digits may differ between rank counts; they are
// printed to 10 significant digits.

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator, PolarizationOptions, SiteChunking, SiteSelection, SolverOptions};

fn dump(label: &str, buf: &[f64]) {
    for (i, v) in buf.iter().enumerate() {
        println!("{label}[{i}] = {v:.10e}");
    }
}

fn main() {
    let mpi = aedelsten_mpi::Session::init(aedelsten_mpi::ThreadLevel::Funneled).expect("MPI init failed");
    let world = mpi.world().expect("the world communicator");
    let rank = aedelsten_mpi::rank();

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/mpi_eri_check.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &world).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n = ctx.get_n_cont();

    // 60 sites, 30 on each of two shells around the molecule (an even spiral), one fragment each, anisotropic.
    let n_sites = 60;
    let mut sites = format!("{n_sites} Bohr -1 2 1\n");
    for s in 0..n_sites {
        let (shell, i) = (s % 2, (s / 2) as f64);
        let z = 1.0 - (2.0 * i + 1.0) / 30.0;
        let (rho, phi) = ((1.0 - z * z).sqrt(), 2.399_963 * i + shell as f64);
        let r = if shell == 0 { 9.0 } else { 12.0 };
        let (x, y, z) = (r * rho * phi.cos(), r * rho * phi.sin(), r * z);
        let a = 0.8 + 0.01 * s as f64;
        sites.push_str(&format!("{} {x} {y} {z} {a} 0.1 0.0 {} 0.05 {}\n", s + 1, a + 0.3, a - 0.2));
    }
    let file = std::env::temp_dir().join(format!("mpi_embedding_check_{}_{rank}.xfield", std::process::id()));
    std::fs::write(&file, sites).unwrap();
    ctx.set_polarizable_sites(file.to_str().unwrap()).unwrap();
    std::fs::remove_file(&file).ok();
    let options = |resident_field_budget| PolarizationOptions {
        solver: SolverOptions { threshold: 1e-12, ..SolverOptions::default() },
        chunking: SiteChunking::Sites(7),
        response_batch: 2,
        state_slots: 2,
        resident_field_budget,
        ..PolarizationOptions::default()
    };
    ctx.set_polarization(options(std::env::args().any(|a| a == "resident").then_some((1, arrays::scratch::ByteUnit::GiB)))).unwrap();
    if std::env::args().any(|a| a == "partial") {
        // Rank r keeps r + 2 sites (or fewer, as the KiB round), the rest of its share evaluated.
        let storage = ctx.get_polarization_storage().unwrap();
        let scratch = storage.bytes_to_keep - storage.sites * storage.bytes_per_site;
        ctx.free_polarization();
        let memory = ((rank + 2) * storage.bytes_per_site + scratch) / 1024;
        ctx.set_polarization(options(Some((memory, arrays::scratch::ByteUnit::KiB)))).unwrap();
        let kept = ctx.get_polarization_storage().unwrap();
        eprintln!("rank {rank}: {} of {} sites kept", kept.sites_in_memory, kept.sites);
    }

    let density = |shift: f64| -> Vec<f64> { (0..n * n).map(|idx| 0.05 / (1.0 + shift + (idx % n) as f64 + (idx / n) as f64)).collect() };
    let (d0, d1) = (density(0.0), density(0.7));

    let mut v_ind = vec![0.0; n * n];
    let report = ctx.compute_induction(&d0, &mut v_ind).unwrap();
    let mut dipoles = vec![0.0; 3 * n_sites];
    ctx.get_induced_dipoles(&mut dipoles).unwrap();
    let mut field = vec![0.0; 3 * n_sites];
    ctx.compute_electronic_field(&d0, SiteSelection::Polarizable, &SiteChunking::Sites(5), &mut field).unwrap();

    let both: Vec<f64> = d0.iter().chain(&d1).copied().collect();
    let mut response = vec![0.0; 2 * n * n];
    ctx.compute_induction_response(&both, 2, &mut response).unwrap();
    let (mut states, mut energies) = (vec![0.0; 2 * n * n], vec![0.0; 2]);
    ctx.compute_induction_states(&both, 2, &mut states, &mut energies).unwrap();
    let mut eef = vec![0.0; 3 * n * n];
    ctx.compute_effective_external_field(&mut eef).unwrap();

    if rank == 0 {
        dump("energy", &[report.energy]);
        dump("dipoles", &dipoles);
        dump("v_ind", &v_ind);
        dump("field", &field);
        dump("response", &response);
        dump("state_energies", &energies);
        dump("states", &states);
        dump("eef", &eef);
    }
    environment(&world, &d0);
    aedelsten_mpi::barrier();
}

/// `count` points on a sphere of radius `r` about the origin, an even spiral.
fn spiral(count: usize, r: f64) -> Vec<[f64; 3]> {
    (0..count)
        .map(|i| {
            let z = 1.0 - (2.0 * i as f64 + 1.0) / count as f64;
            let (rho, phi) = ((1.0 - z * z).sqrt(), 2.399_963 * i as f64);
            [r * rho * phi.cos(), r * rho * phi.sin(), r * z]
        })
        .collect()
}

/// Water with 40 embedding AIMPs and 300 point multipoles: the multipole, AIMP Coulomb and AIMP exchange operators
/// and the gradients of `density` with them.
fn environment(world: &aedelsten_mpi::Communicator, density: &[f64]) {
    let rank = world.rank();
    let mut input = String::from("Basis set\n    O.ANO-RCC...3s2p1d\n    O1 0.0 0.0 0.1\nEnd of Basis\n");
    input.push_str("Basis set\n    H.ANO-RCC...2s1p\n    H1 0.0 1.4 -0.9\n    H2 0.0 -1.4 -0.9\nEnd of Basis\n");
    input.push_str("Basis set\n    Mg.TEST-EMB-AIMP.Larsson.0s.0s.ECP.MgO\n");
    for (i, [x, y, z]) in spiral(40, 10.0).into_iter().enumerate() {
        input.push_str(&format!("    Na{} {x} {y} {z}\n", i + 1));
    }
    input.push_str("End of Basis\n");
    let mut multipoles = String::from("300 Bohr 2\n");
    for (i, [x, y, z]) in spiral(150, 14.0).into_iter().chain(spiral(150, 17.0)).enumerate() {
        let q = if i % 2 == 0 { 0.4 } else { -0.4 };
        multipoles.push_str(&format!("{x} {y} {z} {q} 0.05 -0.02 0.03 0.01 0.0 -0.01 0.02 0.0 -0.03\n"));
    }
    let stem = std::env::temp_dir().join(format!("mpi_embedding_check_env_{}_{rank}", std::process::id()));
    let (inp, xfield) = (stem.with_extension("inp"), stem.with_extension("xfield"));
    std::fs::write(&inp, input).unwrap();
    std::fs::write(&xfield, multipoles).unwrap();
    let mut ctx = AedelstenContext::new(inp.to_str().unwrap(), ContractionType::General, &world).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_external_multipoles(xfield.to_str().unwrap()).unwrap();
    ctx.set_embedding_aimp_operators().unwrap();
    std::fs::remove_file(&inp).ok();
    std::fs::remove_file(&xfield).ok();
    let n = ctx.get_n_cont();
    assert_eq!(n * n, density.len(), "the environment's QM region is the same water");
    let n_centres = ctx.get_n_gradient_centres();

    let operators: [(&str, OneElectronOperator<'_>); 3] = [
        ("multipoles", ctx.get_external_multipole_operator().unwrap()),
        ("aimp_coulomb", ctx.get_embedding_aimp_coulomb_operator().unwrap()),
        ("aimp_exchange", ctx.get_embedding_aimp_exchange_operator().unwrap()),
    ];
    for (label, operator) in operators {
        let mut matrix = vec![0.0; n * n];
        ctx.compute_one_electron_integrals(&mut matrix, operator.clone(), false).unwrap();
        let mut gradient = vec![0.0; 3 * n_centres];
        ctx.compute_one_electron_gradient(&mut gradient, density, operator, None).unwrap();
        if rank == 0 {
            dump(label, &matrix);
            dump(&format!("{label}_gradient"), &gradient);
        }
    }
}
