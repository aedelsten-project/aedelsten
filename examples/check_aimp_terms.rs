// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

use aedelsten::{AedelstenContext, ContractionType};

fn frob_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn max_asymmetry(v: &[f64], n: usize) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let d = (v[i + j * n] - v[j + i * n]).abs();
            worst = worst.max(d);
        }
    }
    worst
}

fn dump(label: &str, buf: &[f64], n: usize) {
    println!("--- {label} (full {n}x{n}, column-major) ---");
    for i in 0..n {
        for j in 0..n {
            print!("{:>10.5} ", buf[i + j * n]);
        }
        println!();
    }
}

fn report(label: &str, buf: &[f64], n: usize) {
    println!(
        "{label}: frob={:.6} max={:.6} max_asymmetry={:.3e}",
        frob_norm(buf),
        buf.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs())),
        max_asymmetry(buf, n)
    );
}

fn run(path: &str, label: &str) {
    println!("=== {label} ({path}) ===");
    let mut ctx = AedelstenContext::new(path, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    if ctx.has_embedding_aimp() {
        ctx.set_embedding_aimp_operators().unwrap();
    }
    let n = ctx.get_n_cont();
    println!("n_cont = {n}");
    println!("nuclear_charge = {}", ctx.compute_nuclear_charge());
    println!("nuclear_repulsion_energy = {}", ctx.compute_nuclear_repulsion_energy());
    println!("nuclear_aimp_embedding_energy = {}", ctx.compute_nuclear_aimp_embedding_energy());

    let mut buf = vec![0.0; n * n];
    if ctx.has_core_aimp() {
        ctx.compute_one_electron_integrals(&mut buf, ctx.get_core_aimp_point_charge_operator().unwrap(), false).unwrap();
        report("core_aimp_point_charge", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_core_aimp_screened_coulomb_operator().unwrap(), false).unwrap();
        report("core_aimp_screened_coulomb", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_core_aimp_coulomb_operator().unwrap(), false).unwrap();
        report("core_aimp_coulomb (pc+screened)", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_core_aimp_projection_operator().unwrap(), false).unwrap();
        report("core_aimp_projection (W)", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_core_aimp_spectrally_resolved_exchange_operator().unwrap(), false).unwrap();
        report("core_aimp_spectrally_resolved_exchange (K_sro)", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_core_aimp_exchange_operator().unwrap(), false).unwrap();
        report("core_aimp_exchange (W+K_sro)", &buf, n);
    }

    if ctx.has_embedding_aimp() {
        ctx.compute_one_electron_integrals(&mut buf, ctx.get_embedding_aimp_point_charge_operator().unwrap(), false).unwrap();
        report("embedding_aimp_point_charge", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_embedding_aimp_screened_coulomb_operator().unwrap(), false).unwrap();
        report("embedding_aimp_screened_coulomb", &buf, n);
        dump("embedding_aimp_screened_coulomb", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_embedding_aimp_coulomb_operator().unwrap(), false).unwrap();
        report("embedding_aimp_coulomb (pc+screened)", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_embedding_aimp_projection_operator().unwrap(), false).unwrap();
        report("embedding_aimp_projection (W)", &buf, n);
        dump("embedding_aimp_projection", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_embedding_aimp_spectrally_resolved_exchange_operator().unwrap(), false).unwrap();
        report("embedding_aimp_spectrally_resolved_exchange (K_sro)", &buf, n);
        dump("embedding_aimp_spectrally_resolved_exchange", &buf, n);

        ctx.compute_one_electron_integrals(&mut buf, ctx.get_embedding_aimp_exchange_operator().unwrap(), false).unwrap();
        report("embedding_aimp_exchange (W+K_sro)", &buf, n);
    }

    ctx.compute_one_electron_integrals(&mut buf, ctx.get_nuclear_attraction_operator(), false).unwrap();
    report("nuclear_attraction", &buf, n);

    ctx.compute_one_electron_integrals(&mut buf, aedelsten::OneElectronOperator::kinetic(), false).unwrap();
    report("kinetic", &buf, n);

    ctx.compute_one_electron_integrals(&mut buf, aedelsten::OneElectronOperator::overlap(), false).unwrap();
    report("overlap", &buf, n);
    dump("overlap", &buf, n);
    println!();
}

fn main() {
    let fe_dir = "/Users/edl/Source/aedelsten_cleanup/aedelsten_scf/tests/only_emb";
    run(&format!("{fe_dir}/wierd_aimp_test.inp"), "F + Fe embedding AIMP, partial d-shell occupation (Fe, d4)");
}
