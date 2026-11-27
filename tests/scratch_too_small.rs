// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The scratch never grows: a computation that runs out of it fails with an arena error naming the size it needed
// at that point, and a scratch buffer of exactly the measured high-water mark is enough for the same work. Covered: the
// one-electron matrices and contracted gradients, the ERI tensor, set_eri_data's Schwarz screening, a direct Fock
// build, the half transform (one basis and two) and the two-electron gradient.

mod common;

use aedelsten::arrays::scratch::{ArenaError, ByteUnit};
use aedelsten::{TwoElectronTerm, TwoElectronTermKind};
use aedelsten::{AedelstenContext, AedelstenError, ContractionType, ERIAOSymmetry, ERIIndexOrder, ERIRangeOperator};

const CC_PVDZ: &str = include_str!("../src/basis/data/cc-pVDZ");

static CURRENT_DIRECTORY: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn context(tag: &str, basis: &str, scratch: (usize, ByteUnit)) -> AedelstenContext {
    let dir = common::temp_path(&format!("wssmall_{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("cc-pVDZ"), CC_PVDZ).unwrap();
    let (heavy, light) = if basis == "cc-pVDZ" { ("F", "H") } else { ("O", "H") };
    let input = format!(
        "Basis set\n    {heavy}.{basis}\n    {heavy}1 0.0 0.1 0.2\nEnd of Basis\nBasis set\n    {light}.{basis}\n    {light}1 0.3 1.5 -0.9\n    {light}2 -0.2 -1.4 -1.1\nEnd of Basis\n"
    );
    std::fs::write(dir.join("m.inp"), input).unwrap();
    let ctx = {
        let _guard = CURRENT_DIRECTORY.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let ctx = AedelstenContext::new(dir.join("m.inp").to_str().unwrap(), ContractionType::General, &aedelsten_mpi::Communicator::single());
        std::env::set_current_dir(previous).unwrap();
        ctx
    };
    let _ = std::fs::remove_dir_all(&dir);
    let mut ctx = ctx.unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch(scratch).unwrap();
    ctx
}

fn density(n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[i + n * j] = 0.05 * (((i.min(j) * 7 + i.max(j) * 3 + 1) % 11) as f64) - 0.1;
        }
    }
    d
}

/// Everything that runs through the scratch, flattened.
fn outputs(ctx: &mut AedelstenContext) -> Result<Vec<(String, Vec<f64>)>, AedelstenError> {
    let n = ctx.get_n_cont();
    let n_centres = ctx.get_n_gradient_centres();
    let d = density(n);
    let mut out = Vec::new();
    for (name, operator) in [("overlap", ctx.get_overlap_operator()), ("kinetic", ctx.get_kinetic_operator()), ("nuclear", ctx.get_nuclear_attraction_operator())] {
        let mut m = vec![0.0; n * n];
        ctx.compute_one_electron_integrals(&mut m, operator, false)?;
        out.push((name.to_string(), m));
    }
    let mut g = vec![0.0; 3 * n_centres];
    ctx.compute_one_electron_gradient(&mut g, &d, ctx.get_nuclear_attraction_operator(), None)?;
    out.push(("nuclear gradient".to_string(), g));
    let mut eri = vec![0.0; aedelsten::compute_s8_size(n)];
    ctx.compute_eri_stored(&mut eri, ERIAOSymmetry::S8 { order: ERIIndexOrder::ColumnMajor }, ERIRangeOperator::FullRange)?;
    out.push(("ERI".to_string(), eri));
    ctx.set_eri_data(Some(1e-12))?;
    let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    ctx.compute_fock_direct_restricted(&d, &mut j, &mut k, 1.0, -0.5, ERIRangeOperator::FullRange, None)?;
    out.push(("J".to_string(), j));
    out.push(("K".to_string(), k));
    // The half transform: its row buffers come off the front of the scratch, the walk runs in the rest.
    let m = 3;
    let c: Vec<f64> = (0..n * m).map(|i| 0.1 * (i % 5) as f64 - 0.2).collect();
    let mut x = vec![0.0; m * n * m * n];
    ctx.compute_eri_half_transformed(&c, m, &c, m, &mut x, None, None, true)?;
    out.push(("X, one basis".to_string(), x));
    let mut x = vec![0.0; n * m * n];
    ctx.compute_eri_half_transformed(&c[n..2 * n], 1, &c, m, &mut x, None, None, true)?;
    out.push(("X, two bases".to_string(), x));
    let mut shells = vec![aedelsten::ShellInfo::default(); ctx.get_n_shells()];
    ctx.get_shells(&mut shells)?;
    let sigma = shells[1].first_ao..n;
    let mut x = vec![0.0; m * n * m * sigma.len()];
    ctx.compute_eri_half_transformed_sigma(&c, m, &c, m, sigma.start, sigma.end, &mut x, None, None, true)?;
    out.push(("X, a batch of sigma, one basis".to_string(), x));
    let mut x = vec![0.0; n * m * sigma.len()];
    ctx.compute_eri_half_transformed_sigma(&c[n..2 * n], 1, &c, m, sigma.start, sigma.end, &mut x, None, None, true)?;
    out.push(("X, a batch of sigma, two bases".to_string(), x));
    let view = &d[..];
    let terms = [
        TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: view, right: view, coeff: 1.0 },
        TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: view, right: view, coeff: -0.5 },
    ];
    let mut g = vec![0.0; 3 * n_centres];
    ctx.compute_two_electron_gradient(&mut g, &terms, ERIRangeOperator::FullRange, None, None)?;
    out.push(("two-electron gradient".to_string(), g));
    Ok(out)
}

fn check(basis: &str) {
    let mut large = context(&format!("{basis}_large"), basis, (1, ByteUnit::GiB));
    let reference = outputs(&mut large).unwrap();
    let peak = large.get_scratch_peak_bytes().unwrap();

    // Too small: an error, not a regrowth, with a lower bound on the size needed.
    let mut small = context(&format!("{basis}_small"), basis, (1, ByteUnit::KiB));
    match outputs(&mut small) {
        Err(AedelstenError::Arena(error @ ArenaError::OutOfMemory { needed, available, used })) => {
            assert_eq!(used + available, 1024, "{basis}: the error reports the scratch's size");
            assert!(used + needed > 1024 && used + needed <= peak, "{basis}: needs {} of a peak {peak}", used + needed);
            assert!(error.to_string().contains("too small"), "{error}");
        }
        other => panic!("{basis}: expected an out-of-space error, got {:?}", other.map(|_| ())),
    }

    // Exactly the high-water mark (rounded up to the unit) is enough, and gives the same results.
    let mut exact = context(&format!("{basis}_exact"), basis, (peak.div_ceil(1024), ByteUnit::KiB));
    for ((name, a), (_, b)) in outputs(&mut exact).unwrap().iter().zip(&reference) {
        assert_eq!(a, b, "{basis} {name}: the scratch size must not change the results");
    }
    assert_eq!(peak, exact.get_scratch_peak_bytes().unwrap(), "{basis}: same work, same high-water mark");

    // After an error, a resize is all it takes.
    small.resize_scratch((peak.div_ceil(1024), ByteUnit::KiB)).unwrap();
    for ((name, a), (_, b)) in outputs(&mut small).unwrap().iter().zip(&reference) {
        assert_eq!(a, b, "{basis} {name}: after resize_scratch");
    }
}

#[test]
fn a_too_small_scratch_is_an_error_and_the_peak_is_enough() {
    check("STO-3G");
    check("cc-pVDZ");
}

#[test]
fn the_scratch_is_the_size_given() {
    let ctx = context("given", "STO-3G", (3, ByteUnit::MiB));
    let mut breakdown = vec![0usize; aedelsten::MEMORY_BREAKDOWN_FIELDS.len()];
    ctx.memory_breakdown(&mut breakdown).unwrap();
    assert_eq!(breakdown[2], 3 * 1024 * 1024);
}
