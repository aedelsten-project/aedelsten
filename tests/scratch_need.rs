// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The scratch query (`scratch_bytes`, from the input alone) against the peak each operation actually reaches on
// the same context: never below it, and close to it. Contexts of each kind (general, segmented, primitive) on
// x2c-SVPall.

mod common;

use aedelsten::arrays::scratch::ByteUnit;
use aedelsten::{AedelstenContext, ContractionType, ERIRangeOperator, MomentOrigin, OneElectronOperator, TwoElectronTerm, TwoElectronTermKind, ScratchItem, ScratchOperation};

// x2c-SVPall for every kind (the one exception to the test sets' ANO-RCC for the general and primitive kinds: this
// file checks the scratch estimates, not the integrals, and ANO-RCC's deep primitives made it 58 s against 3 s).
// Two O 2.6 bohr apart and one H: the query's largest batches are two-centre O-O ones, so the close contact is what
// makes the peaks reach them (with the O 6 bohr apart the overlap screening leaves out more of the pairs between them,
// and the query, which cannot know the geometry, is up to 1.74 times the peak; with one O, up to 4 times). More atoms
// add quartets, not larger batches (two waters and three more H gave the same ratios in twice the time).
// The orbitals of the active-space walks and the half transforms.
const ACTIVE: usize = 4;
const OCCUPIED: usize = 3;

fn input(oxygen: &str, hydrogen: &str) -> String {
    format!(
        "Basis set\n    O.{oxygen}\n    O1 0.0 0.0 0.1\n    O2 2.6 0.3 0.0\nEnd of Basis\nBasis set\n    H.{hydrogen}\n    H1 0.0 1.4 -0.9\nEnd of Basis\n"
    )
}

fn cases() -> [(&'static str, String, ContractionType); 3] {
    [
        ("general", input("x2c-SVPall", "x2c-SVPall"), ContractionType::General),
        ("segmented", input("x2c-SVPall", "x2c-SVPall"), ContractionType::Segmented),
        ("primitive", input("x2c-SVPall", "x2c-SVPall"), ContractionType::Primitive),
    ]
}

fn context(name: &str, input: &str, contraction: ContractionType) -> AedelstenContext {
    let path = common::temp_path(&format!("scratch_need_{name}.inp"));
    std::fs::write(&path, input).unwrap();
    let ctx = AedelstenContext::new(path.to_str().unwrap(), contraction, &aedelsten_mpi::Communicator::single()).unwrap();
    let _ = std::fs::remove_file(&path);
    ctx
}

// A fresh scratch, so the peak is the next operation's alone.
fn fresh_scratch(ctx: &mut AedelstenContext) {
    ctx.free_scratch();
    ctx.set_scratch((512, ByteUnit::MiB)).unwrap();
}

// The overlap matrix, a density that decays as a converged one does.
fn density(ctx: &AedelstenContext) -> Vec<f64> {
    let n = ctx.get_n_cont();
    let mut s = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut s, OneElectronOperator::overlap(), false).unwrap();
    s
}

#[test]
fn the_query_is_never_below_the_peak_of_an_operation_and_close_to_it() {
    let range = ERIRangeOperator::FullRange;
    for (name, input, contraction) in cases() {
        let mut ctx = context(name, &input, contraction);
        // Before the shell pairs: the input alone.
        let query = |ctx: &AedelstenContext, operation| ctx.scratch_bytes(&[operation], false).unwrap();
        let started = std::time::Instant::now();
        let eri_data = query(&ctx, ScratchOperation::EriData);
        let restricted = query(&ctx, ScratchOperation::RestrictedFock);
        let unrestricted = query(&ctx, ScratchOperation::UnrestrictedFock);
        let gradient = query(&ctx, ScratchOperation::TwoElectronGradient { terms: 2 });
        let general = query(&ctx, ScratchOperation::GeneralFock);
        let multi = query(&ctx, ScratchOperation::GeneralMultiFock { densities: 3 });
        let active = query(&ctx, ScratchOperation::ActiveSpaceFock { active: ACTIVE });
        let mixed = query(&ctx, ScratchOperation::MixedActiveSpaceFock { active: ACTIVE });
        let half_one = query(&ctx, ScratchOperation::HalfTransform { left: OCCUPIED, right: None, sigma: None, node_ranks: 1 });
        let half_two = query(&ctx, ScratchOperation::HalfTransform { left: OCCUPIED, right: Some(ACTIVE), sigma: None, node_ranks: 1 });
        let all = ctx.scratch_bytes(&[ScratchOperation::EriData, ScratchOperation::RestrictedFock, ScratchOperation::UnrestrictedFock], false).unwrap();
        eprintln!("{name}: the queries took {:.2} s", started.elapsed().as_secs_f64());
        assert_eq!(all, eri_data.max(restricted).max(unrestricted));

        ctx.set_shell_pairs(Some(1e-14)).unwrap();
        assert_eq!(query(&ctx, ScratchOperation::RestrictedFock), restricted, "{name}: the query depends on the input alone");
        fresh_scratch(&mut ctx);
        ctx.set_eri_data(Some(1e-12)).unwrap();
        let mut peaks = vec![("ERI data", ctx.get_scratch_peak_bytes().unwrap(), eri_data)];

        let n = ctx.get_n_cont();
        let d = density(&ctx);
        let (mut j, mut k, mut k_beta) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
        fresh_scratch(&mut ctx);
        ctx.compute_fock_direct_restricted(&d, &mut j, &mut k, 1.0, -0.5, range, Some(1e-12)).unwrap();
        peaks.push(("restricted Fock", ctx.get_scratch_peak_bytes().unwrap(), restricted));
        let half: Vec<f64> = d.iter().map(|x| 0.5 * x).collect();
        fresh_scratch(&mut ctx);
        ctx.compute_fock_direct_unrestricted(&half, &half, &d, &mut j, &mut k, &mut k_beta, 1.0, -1.0, range, Some(1e-12)).unwrap();
        peaks.push(("unrestricted Fock", ctx.get_scratch_peak_bytes().unwrap(), unrestricted));
        // The restricted gradient's `1/2 (J - 1/2 K)`.
        let terms = [
            TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: &d, right: &d, coeff: 0.5 },
            TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: &d, right: &d, coeff: -0.25 },
        ];
        let mut flat = vec![0.0; 3 * ctx.get_n_gradient_centres()];
        fresh_scratch(&mut ctx);
        ctx.compute_two_electron_gradient(&mut flat, &terms, range, Some(1e-12), None).unwrap();
        peaks.push(("two-electron gradient", ctx.get_scratch_peak_bytes().unwrap(), gradient));

        fresh_scratch(&mut ctx);
        ctx.compute_fock_direct_general(&d, &mut j, &mut k, 1.0, -0.5, range, Some(1e-12)).unwrap();
        peaks.push(("general Fock", ctx.get_scratch_peak_bytes().unwrap(), general));
        let three: Vec<f64> = (0..3).flat_map(|_| d.iter().copied()).collect();
        let (mut j3, mut k3) = (vec![0.0; 3 * n * n], vec![0.0; 3 * n * n]);
        fresh_scratch(&mut ctx);
        ctx.compute_fock_direct_general_multi(&three, &mut j3, &mut k3, 3, 1.0, -0.5, range, Some(1e-12)).unwrap();
        peaks.push(("general Fock of three densities", ctx.get_scratch_peak_bytes().unwrap(), multi));

        // Orbitals: the first columns of the identity, as many as asked.
        let columns = |m: usize| -> Vec<f64> { (0..n * m).map(|i| if i % n == i / n { 1.0 } else { 0.0 }).collect() };
        let c_active = columns(ACTIVE);
        let pairs = ACTIVE * (ACTIVE + 1) / 2;
        let mut m = vec![0.0; n * (n + 1) / 2 * pairs];
        let (mut j2, mut k2) = (vec![0.0; n * n], vec![0.0; n * n]);
        fresh_scratch(&mut ctx);
        ctx.compute_fock_direct_active_space(&half, &half, &mut j, &mut k, &mut j2, &mut k2, 1.0, -0.5, 1.0, -0.5, &c_active, ACTIVE, &mut m, None, Some(1e-12)).unwrap();
        peaks.push(("active-space Fock", ctx.get_scratch_peak_bytes().unwrap(), active));
        let mut m = vec![0.0; n * (n + 1) / 2 * ACTIVE * ACTIVE];
        fresh_scratch(&mut ctx);
        ctx.compute_fock_direct_mixed_active_space(&d, &mut j, &mut k, 1.0, -0.5, &c_active, &c_active, ACTIVE, &mut m, None, Some(1e-12)).unwrap();
        peaks.push(("mixed active-space Fock", ctx.get_scratch_peak_bytes().unwrap(), mixed));

        let c_occupied = columns(OCCUPIED);
        let mut x = vec![0.0; OCCUPIED * OCCUPIED * n * n];
        fresh_scratch(&mut ctx);
        ctx.compute_eri_half_transformed(&c_occupied, OCCUPIED, &c_occupied, OCCUPIED, &mut x, None, Some(1e-12), false).unwrap();
        peaks.push(("half transform, one basis", ctx.get_scratch_peak_bytes().unwrap(), half_one));
        let mut x = vec![0.0; OCCUPIED * ACTIVE * n * n];
        fresh_scratch(&mut ctx);
        ctx.compute_eri_half_transformed(&c_occupied, OCCUPIED, &c_active, ACTIVE, &mut x, None, Some(1e-12), false).unwrap();
        peaks.push(("half transform, two bases", ctx.get_scratch_peak_bytes().unwrap(), half_two));

        for (operation, peak, query) in peaks {
            eprintln!("{name}, {operation}: peak {peak} bytes, query {query} ({:.3})", query as f64 / peak as f64);
            assert!(query >= peak, "{name}, {operation}: the query {query} below the peak {peak}");
            assert!((query as f64) <= 1.5 * peak as f64, "{name}, {operation}: the query {query} far above the peak {peak}");
        }
    }
}

// The breakdown (`scratch_breakdown`): per operation, in the order asked, parts that add up to its need, on the shells
// of the item that sets it; the query is its largest need.
#[test]
fn the_breakdown_adds_up_to_the_query() {
    let operations = [
        ScratchOperation::EriData,
        ScratchOperation::RestrictedFock,
        ScratchOperation::TwoElectronGradient { terms: 2 },
        ScratchOperation::GeneralMultiFock { densities: 3 },
        ScratchOperation::ActiveSpaceFock { active: ACTIVE },
        ScratchOperation::HalfTransform { left: OCCUPIED, right: Some(ACTIVE), sigma: None, node_ranks: 2 },
    ];
    for (name, input, contraction) in cases() {
        let ctx = context(&format!("breakdown_{name}"), &input, contraction);
        for primitive in [false, true] {
            let breakdown = ctx.scratch_breakdown(&operations, primitive).unwrap();
            assert_eq!(breakdown.len(), operations.len());
            for (need, operation) in breakdown.iter().zip(&operations) {
                eprintln!(
                    "{name}{}, {:?}: {} = buffers {} + tables {} + row {} + {:?} {} on {:?}",
                    if primitive { " (primitive)" } else { "" },
                    need.operation, need.bytes, need.buffers, need.tables, need.row, need.item, need.item_bytes, need.shells,
                );
                assert_eq!(format!("{:?}", need.operation), format!("{operation:?}"));
                assert_eq!(need.bytes, need.buffers + need.tables + need.row + need.item_bytes, "{name}, {operation:?}: the parts add up");
                assert!(need.item_bytes > 0);
                let expected_shells = if need.item == ScratchItem::Quartet { 4 } else { 2 };
                assert_eq!(need.shells.len(), expected_shells);
                if need.item != ScratchItem::Quartet {
                    assert_eq!((need.tables, need.row), (0, 0), "a pair's item has no row");
                }
                assert_eq!(need.bytes, ctx.scratch_bytes(&[*operation], primitive).unwrap());
            }
            let largest = breakdown.iter().map(|need| need.bytes).max().unwrap();
            assert_eq!(ctx.scratch_bytes(&operations, primitive).unwrap(), largest);
        }
        assert!(matches!(ctx.scratch_breakdown(&[ScratchOperation::EriData], false).unwrap()[0].item, ScratchItem::EriDataEntry));
    }
}

// The primitive query (`primitive = true`) of a contracted context: the same operations on the primitive blocks, never
// below the contracted query, and never below the peaks of the same input run as the primitive kind.
#[test]
fn the_primitive_query_covers_the_primitive_kind() {
    let range = ERIRangeOperator::FullRange;
    let input = input("x2c-SVPall", "x2c-SVPall");
    let general = context("primitive_query_general", &input, ContractionType::General);
    let operations = [
        ScratchOperation::EriData,
        ScratchOperation::RestrictedFock,
        ScratchOperation::TwoElectronGradient { terms: 2 },
    ];
    let primitive_query = general.scratch_bytes(&operations, true).unwrap();
    assert!(primitive_query >= general.scratch_bytes(&operations, false).unwrap());

    let mut ctx = context("primitive_query_primitive", &input, ContractionType::Primitive);
    ctx.set_shell_pairs(Some(1e-14)).unwrap();
    fresh_scratch(&mut ctx);
    ctx.set_eri_data(Some(1e-12)).unwrap();
    let mut peak = ctx.get_scratch_peak_bytes().unwrap();
    let n = ctx.get_n_cont();
    let d = density(&ctx);
    let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    fresh_scratch(&mut ctx);
    ctx.compute_fock_direct_restricted(&d, &mut j, &mut k, 1.0, -0.5, range, Some(1e-12)).unwrap();
    peak = peak.max(ctx.get_scratch_peak_bytes().unwrap());
    let terms = [
        TwoElectronTerm { kind: TwoElectronTermKind::Coulomb, left: &d, right: &d, coeff: 0.5 },
        TwoElectronTerm { kind: TwoElectronTermKind::Exchange, left: &d, right: &d, coeff: -0.25 },
    ];
    let mut flat = vec![0.0; 3 * ctx.get_n_gradient_centres()];
    fresh_scratch(&mut ctx);
    ctx.compute_two_electron_gradient(&mut flat, &terms, range, Some(1e-12), None).unwrap();
    peak = peak.max(ctx.get_scratch_peak_bytes().unwrap());
    eprintln!("primitive query {primitive_query} bytes, the primitive kind's peak {peak} ({:.3})", primitive_query as f64 / peak as f64);
    assert!(primitive_query >= peak, "the primitive query {primitive_query} below the primitive kind's peak {peak}");
    assert!((primitive_query as f64) <= 1.5 * peak as f64, "the primitive query {primitive_query} far above the peak {peak}");
}

// The one-electron operations are not measured by the query: the ERI items cover them (an item of a shell pair needs
// less than the ERI data entry of the same pair, whose diagonal batch (ab|ab) holds the pair's products squared and
// Hermite buffers to twice its order). Checked here on each kind of context: the integrals (overlap, kinetic, nuclear
// attraction, moments to order 2, nabla, angular momentum) against the ERI data query; the primitive-basis operations
// (X2C's pVp, scalar and spin-orbit) against the primitive one, AMFI against its own (measured on the atom kinds); the
// one-electron gradients against the two-electron gradient's; the stored tensor's walk (the direct walk's batches)
// against the direct Fock's.
#[test]
fn the_one_electron_operations_are_covered_by_the_eri_queries() {
    for (name, input, contraction) in cases() {
        let mut ctx = context(name, &input, contraction);
        let eri_data = ctx.scratch_bytes(&[ScratchOperation::EriData], false).unwrap();
        let primitive = ctx.scratch_bytes(&[ScratchOperation::EriData], true).unwrap();
        let gradient = ctx.scratch_bytes(&[ScratchOperation::TwoElectronGradient { terms: 2 }], false).unwrap();
        let restricted = ctx.scratch_bytes(&[ScratchOperation::RestrictedFock], false).unwrap();
        let amfi = ctx.scratch_bytes(&[ScratchOperation::Amfi], false).unwrap();
        ctx.set_shell_pairs(Some(1e-14)).unwrap();
        ctx.set_primitive_shell_pairs(None).unwrap();
        let n = ctx.get_n_cont();
        ctx.set_scratch((64, aedelsten::arrays::scratch::ByteUnit::MiB)).unwrap();
        let d = {
            let mut s = vec![0.0; n * n];
            ctx.compute_one_electron_integrals(&mut s, OneElectronOperator::overlap(), false).unwrap();
            s
        };
        let mut flat = vec![0.0; 3 * ctx.get_n_gradient_centres()];
        let n_prim = ctx.get_n_prim_unique();
        let mut covered = |what: &str, query: usize, run: &mut dyn FnMut(&mut AedelstenContext)| {
            fresh_scratch(&mut ctx);
            run(&mut ctx);
            let peak = ctx.get_scratch_peak_bytes().unwrap();
            eprintln!("{name}, {what}: peak {peak} bytes against the query {query} ({:.3})", query as f64 / peak.max(1) as f64);
            assert!(peak <= query, "{name}, {what}: the peak {peak} above the query {query}");
        };
        let mut matrix = vec![0.0; n * n];
        let mut planes = vec![0.0; n * n * 6];
        let (mut prim_matrix, mut prim_planes) = (vec![0.0; n_prim * n_prim], vec![0.0; 6 * n_prim * n_prim]);
        covered("overlap", eri_data, &mut |ctx| ctx.compute_one_electron_integrals(&mut matrix, OneElectronOperator::overlap(), false).unwrap());
        covered("kinetic", eri_data, &mut |ctx| ctx.compute_one_electron_integrals(&mut matrix, OneElectronOperator::kinetic(), false).unwrap());
        covered("nuclear attraction", eri_data, &mut |ctx| {
            let operator = ctx.get_nuclear_attraction_operator();
            ctx.compute_one_electron_integrals(&mut matrix, operator, false).unwrap()
        });
        covered("quadrupole moments", eri_data, &mut |ctx| ctx.compute_one_electron_integrals(&mut planes, OneElectronOperator::moment(MomentOrigin::new([0.1, -0.2, 0.3], 2)), false).unwrap());
        covered("nabla", eri_data, &mut |ctx| ctx.compute_one_electron_integrals(&mut planes[..3 * n * n], OneElectronOperator::nabla(), false).unwrap());
        covered("angular momentum", eri_data, &mut |ctx| ctx.compute_one_electron_integrals(&mut planes[..3 * n * n], OneElectronOperator::angular_momentum([0.1, 0.2, -0.3]), false).unwrap());
        covered("pVp, scalar", primitive, &mut |ctx| {
            let operator = ctx.get_nuclear_attraction_operator();
            ctx.compute_one_electron_picture_change_integrals(&mut prim_matrix, operator, false).unwrap()
        });
        covered("pVp, spin-orbit", primitive, &mut |ctx| {
            let operator = ctx.get_nuclear_attraction_operator();
            ctx.compute_one_electron_picture_change_integrals(&mut prim_planes[..4 * n_prim * n_prim], operator, true).unwrap()
        });
        let mut amfi_planes = vec![0.0; 6 * n_prim * n_prim];
        covered("AMFI", amfi, &mut |ctx| ctx.compute_amfi_integrals(&mut amfi_planes).unwrap());
        // The stored tensor's walk computes the direct walk's batches without its tiles (the tensor itself is on the
        // heap): covered by the direct Fock query. On the smallest basis only (n^4 values).
        if name == "general" {
            let mut dense = vec![0.0; n * n * n * n];
            covered("stored tensor", restricted, &mut |ctx| ctx.compute_eri_stored(&mut dense, aedelsten::ERIAOSymmetry::S1, ERIRangeOperator::FullRange).unwrap());
        }
        covered("kinetic gradient", gradient, &mut |ctx| ctx.compute_one_electron_gradient(&mut flat, &d, OneElectronOperator::kinetic(), None).unwrap());
        covered("nuclear attraction gradient", gradient, &mut |ctx| {
            let operator = ctx.get_nuclear_attraction_operator();
            ctx.compute_one_electron_gradient(&mut flat, &d, operator, None).unwrap()
        });
    }
}
