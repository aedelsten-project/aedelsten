// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The gradient quartet's time and arena peak at the compiled `BRA_PRODUCT_BLOCK`: CuCl2 in full ANO-RCC, generally
//! contracted, every quartet of the Cu-Cu and Cu-Cl shell pairs. A benchmark (`--ignored`), run once per block size.

use arrays::scratch::{Scratch, ByteUnit};
use basis::contraction::Contracted;
use basis::shell_pairs::derive_shell_pairs;
use mmd::hermite_coulomb_integrals::FullRange;

use super::{compute_quartet, pair_data, BraRows, GradientCentres, GradientSink, IntegralDerivatives, BRA_PRODUCT_BLOCK};

struct Discard;

impl GradientSink for Discard {
    fn select_atoms(&mut self, _atoms: &[usize]) {}
    fn distribute(&mut self, _i: usize, _j: usize, _k: usize, _l: usize, _derivatives: IntegralDerivatives<'_>) {}
}

#[test]
#[ignore]
fn gradient_quartets_at_the_block_size() {
    let dir = std::env::temp_dir().join(format!("aedelsten_block_benchmark_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cucl2.inp");
    std::fs::write(&path, "Basis set\n    Cu.ANO-RCC\n    Cu1 0.0 0.0 0.0\nEnd of Basis\nBasis set\n    Cl.ANO-RCC\n    Cl1 0.0 0.0 4.0\n    Cl2 0.0 0.0 -4.0\nEnd of Basis\n").unwrap();
    let system = basis::setup_basis_system::<Contracted>(path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    // The Cu-Cu and Cu-Cl1 pairs (Cu is centre 0, Cl1 centre 1).
    let pairs: Vec<_> = derive_shell_pairs(system.get_basis_shells(), None)
        .unwrap()
        .into_iter()
        .filter(|pair| pair.get_centre_idx_a() == 0 && pair.get_centre_idx_b() <= 1 || pair.get_centre_idx_b() == 0 && pair.get_centre_idx_a() <= 1)
        .collect();
    let mut scratch = Scratch::new(4, ByteUnit::GiB).unwrap();
    let mut arena = scratch.get_arena();
    let data: Vec<_> = pairs.iter().map(|pair| pair_data(pair, &mut arena).unwrap()).collect();
    let mut sink = Discard;
    let (mut total, mut slowest) = (0.0_f64, 0.0_f64);
    let mut quartets = 0;
    for (i, pair_ab) in pairs.iter().enumerate() {
        for (j, pair_cd) in pairs.iter().enumerate().skip(i) {
            let started = std::time::Instant::now();
            arena
                .scope(|scratch| {
                    let bra_rows = BraRows::new_in(pair_ab, scratch)?;
                    compute_quartet::<_, FullRange, Contracted, Contracted>(&mut sink, pair_ab, &data[i], &bra_rows, pair_cd, &data[j], GradientCentres::all(3), scratch, 0.0)
                })
                .unwrap();
            let seconds = started.elapsed().as_secs_f64();
            total += seconds;
            slowest = slowest.max(seconds);
            quartets += 1;
        }
    }
    drop(arena);
    eprintln!(
        "BLOCK {BRA_PRODUCT_BLOCK}: {} pairs, {quartets} quartets in {total:.2} s, the slowest {slowest:.3} s, the largest arena peak {} MiB",
        pairs.len(),
        scratch.peak_bytes() >> 20
    );
}
