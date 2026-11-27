// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Correctness check for oneint's MPI-partitioned one-electron integral loops
// (src/oneint/src/lib.rs's one_electron_integrals_parallel_loop and friends).
//
// Run under `mpirun -n 1` (or plainly, without mpirun at all -- a single MPI
// process is a valid, trivial MPI job, and without the `mpi` feature this program
// never touches MPI at all) to get the reference result: aedelsten_mpi::size() is 1,
// so oneint's loops take the serial path.
//
// Run the same binary under `mpirun -n 2`, `-n 3`, etc. (built with --features mpi)
// to exercise the parallel path (pieces of shell pairs and of the operator's sites
// handed out over the ranks, zero output, Allreduce-sum). Every rank prints the same
// format, so comparing an -n N run's output against the -n 1 reference (e.g. via
// `diff`) checks that the pieces + Allreduce reproduce the serial result. An operator
// summed over its sites is summed by runs of sites on several ranks, so its values may
// differ between rank counts by rounding (at most 1e-13 absolute here): in the last of the 13
// printed digits, or more digits of an element that cancels to a small value.
//
//   cargo build --release --example mpi_oneint_check                        # serial reference build
//   ./target/release/examples/mpi_oneint_check > /tmp/reference.txt
//   cargo build --release --example mpi_oneint_check --features mpi         # MPI build
//   mpirun -n 1 ./target/release/examples/mpi_oneint_check > /tmp/n1.txt
//   mpirun -n 3 ./target/release/examples/mpi_oneint_check > /tmp/n3.txt
//   diff /tmp/reference.txt /tmp/n1.txt   # sanity: -n 1 under the mpi feature still matches plain serial
//   diff /tmp/n1.txt /tmp/n3.txt          # the actual thing being tested
//
// On test_fixtures/screened_pairs_test.inp (atoms 100 bohr apart, pairs screened out) each entry point is also called
// into a buffer filled with a non-zero value, and must give the same result as into a zeroed one on every rank.

use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator, Site, SiteSelection};

// `v + 0.0` prints an exact zero as 0 whatever its sign: the allreduce sums a rank's -0.0 with the others' +0.0, and
// an antisymmetric plane's fold negates that, so the sign of a zero depends on the number of ranks.
fn dump(label: &str, buf: &[f64]) {
    for (i, v) in buf.iter().enumerate() {
        println!("{label}[{i}] = {:.12e}", v + 0.0);
    }
}

// `compute` into a zeroed buffer and into a filled one: the same values, which are returned.
fn overwritten(label: &str, len: usize, compute: impl Fn(&mut [f64])) -> Vec<f64> {
    let mut clean = vec![0.0; len];
    compute(&mut clean);
    let mut dirty = vec![0.75; len];
    compute(&mut dirty);
    let differing = clean.iter().zip(&dirty).filter(|(a, b)| (**a + 0.0).to_bits() != (**b + 0.0).to_bits()).count();
    assert_eq!(differing, 0, "{label}: {differing} of {len} elements depend on what the buffer held");
    clean
}

fn main() {
    let mpi = aedelsten_mpi::Session::init(aedelsten_mpi::ThreadLevel::Funneled).expect("MPI init failed");
    let world = mpi.world().expect("the world communicator");
    let rank = aedelsten_mpi::rank();
    let size = aedelsten_mpi::size();
    eprintln!("rank {rank} of {size}");

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/mpi_oneint_check.inp");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &world).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();

    let n = ctx.get_n_cont();

    // Every rank must run the same computations (Allreduce is a collective -- all
    // ranks call it the same number of times, or the job deadlocks/misbehaves), but
    // after that every rank holds an identical result, so only rank 0 needs to print
    // it. Multiple ranks writing to the same shared stdout would interleave lines
    // non-deterministically and make the output unusable for diffing anyway.
    let mut overlap = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut overlap, OneElectronOperator::overlap(), false).unwrap();

    let mut kinetic = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut kinetic, OneElectronOperator::kinetic(), false).unwrap();

    let mut nuclear = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut nuclear, OneElectronOperator::nuclear_attraction(None), false).unwrap();

    // Component operators with antisymmetric planes (the array loop's fold and the negation of swapped pairs), on
    // water ANO-RCC (s, p and d shells, general contraction), so that pairs of different angular momenta occur.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/mpi_eri_check.inp");
    let mut water = AedelstenContext::new(path, ContractionType::General, &world).unwrap();
    water.set_shell_pairs(None).unwrap();
    water.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let n_water = water.get_n_cont();
    let mut angular_momentum = vec![0.0; 3 * n_water * n_water];
    water.compute_one_electron_integrals(&mut angular_momentum, OneElectronOperator::angular_momentum([0.3, -0.2, 0.5]), false).unwrap();
    let mut nabla = vec![0.0; 3 * n_water * n_water];
    water.compute_one_electron_integrals(&mut nabla, OneElectronOperator::nabla(), false).unwrap();

    // Pairs screened out, an environment (point charges, an embedding AIMP) as in a host's one-electron Hamiltonian.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/screened_pairs_test.inp");
    let mut screened = AedelstenContext::new(path, ContractionType::General, &world).unwrap();
    screened.set_shell_pairs(Some(1e-12)).unwrap();
    screened.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    let xfield = std::env::temp_dir().join(format!("mpi_oneint_check_{}_{rank}.xfield", std::process::id()));
    std::fs::write(&xfield, "2 Bohr 0\n 0.0 -2.0 25.0  0.6\n 0.5  1.0 75.0 -0.9\n").unwrap();
    screened.set_external_multipoles(xfield.to_str().unwrap()).unwrap();
    std::fs::remove_file(&xfield).ok();
    screened.set_embedding_aimp_operators().unwrap();
    let n_screened = screened.get_n_cont();
    let nn = n_screened * n_screened;
    let mut screened_results = Vec::new();
    let matrix_operators = [
        ("screened_overlap", OneElectronOperator::overlap()),
        ("screened_nuclear_attraction", screened.get_nuclear_attraction_operator()),
        ("screened_external_multipoles", screened.get_external_multipole_operator().unwrap()),
        ("screened_embedding_aimp_coulomb", screened.get_embedding_aimp_coulomb_operator().unwrap()),
        ("screened_embedding_aimp_exchange", screened.get_embedding_aimp_exchange_operator().unwrap()),
        ("screened_sap_helfem", screened.get_sap_screening_operator(aedelsten::SapFit::HelFem).unwrap()),
        ("screened_sap_grasp", screened.get_sap_screening_operator(aedelsten::SapFit::Grasp).unwrap()),
    ];
    for (label, operator) in matrix_operators {
        let values = overwritten(label, nn, |out| screened.compute_one_electron_integrals(out, operator.clone(), false).unwrap());
        screened_results.push((label, values));
    }
    let values = overwritten("screened_nabla", 3 * nn, |out| {
        screened.compute_one_electron_integrals(out, OneElectronOperator::nabla(), false).unwrap()
    });
    screened_results.push(("screened_nabla", values));
    let sites: Vec<Site> = [[0.0, 1.0, 5.0], [1.0, -1.0, 95.0]].iter().map(|&p| Site::new(p)).collect();
    let values = overwritten("screened_electric_field", 3 * sites.len() * nn, |out| {
        screened.compute_electric_field_integrals(SiteSelection::Custom(&sites), 0..sites.len(), out).unwrap()
    });
    screened_results.push(("screened_electric_field", values));
    let unit_density: Vec<f64> = (0..nn).map(|k| if k % (n_screened + 1) == 0 { 1.0 } else { 0.0 }).collect();
    let values = overwritten("screened_nuclear_attraction_gradient", 3 * screened.get_n_gradient_centres(), |out| {
        screened.compute_one_electron_gradient(out, &unit_density, screened.get_nuclear_attraction_operator(), None).unwrap()
    });
    screened_results.push(("screened_nuclear_attraction_gradient", values));

    if rank == 0 {
        println!("n = {n}");
        dump("overlap", &overlap);
        dump("kinetic", &kinetic);
        dump("nuclear_attraction", &nuclear);
        dump("angular_momentum", &angular_momentum);
        dump("nabla", &nabla);
        for (label, values) in &screened_results {
            dump(label, values);
        }
    }

    aedelsten_mpi::barrier();
}
