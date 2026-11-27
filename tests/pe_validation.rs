// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// Validation of the polarizable-embedding pipeline against Dalton/PElib.
//
// NOTE ON THE CONVENTION THIS TEST PINS DOWN. The interaction of the electrons with the environment's quadrupoles and
// octupoles is defined WITHOUT the contact terms (the delta functions in the derivatives of 1/r), i.e. with the
// harmonic (traceless) part of each order's moments; see sources::multipole. Computing it with the raw moments and true
// derivatives instead gives an electronic-multipole energy that is 4.2e-4 Eh too negative here, which is what this test
// found and what PElib/Dalton do not have.
//
// The reference is Dalton's `pehf_1pa` test (acrolein, STO-3G, HF, embedded in two polarizable water molecules
// given by a PElib potential with multipoles up to octupoles and anisotropic dipole-dipole polarizabilities,
// iterative induced-dipole solver, no damping). The potential was converted to one xfield file with a throwaway
// script (`test_fixtures/pehf_1pa/environment.xfield`: nOrd = 3, p = 2, one fragment per water, in Bohr with
// PElib's own conversion factor), and the acrolein geometry likewise (`acrolein.inp`).
// STO-3G here is the exception to the test sets (x2c-SVPall): the basis of Dalton's own published case,
// so the reference stays a number Dalton itself tests.
//
// A small restricted-HF loop with DIIS drives the library exactly the way an SCF driver would: the external
// multipole operator in the core Hamiltonian, `compute_induction` in every iteration, and the energy assembled
//
//   E = Tr(P h) + 1/2 Tr(P G) + E_nn + E_nuc,mult + E_ind,     F = h + G + V_ind,
//
// with h = T + V_nuc + V_multipoles. The converged energy and its parts are compared with the numbers in
// Dalton's output (DALTON/test/pehf_1pa/result/pehf_1pa.out). Dalton's "Total PE energy" is
// Ees(electronic) + Ees(nuclear) + Epol(electronic + nuclear + multipole); its "Ees(multipole)" line is the
// environment's own multipole-multipole energy, a constant that is deliberately not part of it.

use aedelsten::arrays::array4d::Array4DView;

use aedelsten::arrays::matrix::ColumnFactor;
use aedelsten::arrays::matrix::Factor;
use aedelsten::arrays::matrix::{Matrix, MatrixMath, MatrixTrait, MatrixTraitMut};
use aedelsten::{AedelstenContext, ContractionType, OneElectronOperator, PolarizationOptions, SiteChunking, SiteSelection, SolverOptions, StaticFieldSources};

const INP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/pehf_1pa/acrolein.inp");
const XFIELD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_fixtures/pehf_1pa/environment.xfield");

// Dalton's numbers (hartree), from result/pehf_1pa.out.
const DALTON_TOTAL_ENERGY: f64 = -188.317391916096;
const DALTON_NUCLEAR_REPULSION: f64 = 102.664846860515;
const DALTON_ELECTRONIC_ENERGY: f64 = -290.963054314876; // total - nuclear repulsion - embedding energy
const DALTON_EMBEDDING_ENERGY: f64 = -0.019184461735;
const DALTON_ES_ELECTRONIC: f64 = -0.467226876813;
const DALTON_ES_NUCLEAR: f64 = 0.448807800596;
const DALTON_POL_ELECTRONIC: f64 = -0.043153863434;
const DALTON_POL_NUCLEAR: f64 = 0.042259438171;
const DALTON_POL_MULTIPOLE: f64 = 0.000129039745;
const DALTON_POL_TOTAL: f64 = -0.000765385518;

const N_ELECTRONS: usize = 30;

fn setup() -> (AedelstenContext, usize) {
    let mut ctx = AedelstenContext::new(INP, ContractionType::General, &aedelsten_mpi::Communicator::single()).unwrap();
    ctx.set_shell_pairs(None).unwrap();
    ctx.set_scratch((64, arrays::scratch::ByteUnit::MiB)).unwrap();
    ctx.set_eri_data(None).unwrap();
    // The same file feeds both lists: the multipole columns become the external multipoles, the polarizability
    // columns the polarizable sites.
    ctx.set_external_multipoles(XFIELD).unwrap();
    ctx.set_polarizable_sites(XFIELD).unwrap();
    let n = ctx.get_n_cont();
    (ctx, n)
}

fn operator(ctx: &AedelstenContext, n: usize, which: OneElectronOperator<'_>) -> Matrix<f64> {
    let mut data = vec![0.0; n * n];
    ctx.compute_one_electron_integrals(&mut data, which, false).unwrap();
    Matrix::<f64>::new_from_vec(n, n, data.into_boxed_slice())
}

fn add(a: &mut Matrix<f64>, b: &Matrix<f64>) {
    for (x, y) in a.get_data_mut().iter_mut().zip(b.get_data()) {
        *x += y;
    }
}

fn trace_product(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
    a.get_data().iter().zip(b.get_data()).map(|(x, y)| x * y).sum()
}

thread_local! {
    static ERI: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct Converged {
    energy: f64,
    density: Matrix<f64>,
    electronic_energy: f64, // Tr(P (T + V_nuc)) + 1/2 Tr(P G): the QM energy without any embedding operator
    iterations: usize,
}

// Restricted HF with DIIS. `induction`: whether to call compute_induction (needs set_polarization).
fn scf(ctx: &mut AedelstenContext, n: usize, induction: bool) -> Converged {
    let ctx = &*{
        // The dense ERI tensor needs the mutable context; everything after is read-only.
        let mut eri_holder = vec![0.0; n * n * n * n];
        ctx.compute_eri_stored(&mut eri_holder, aedelsten::ERIAOSymmetry::S1, aedelsten::ERIRangeOperator::FullRange).unwrap();
        ERI.with(|cell| *cell.borrow_mut() = eri_holder);
        ctx
    };
    let s = operator(ctx, n, OneElectronOperator::overlap());
    let t = operator(ctx, n, OneElectronOperator::kinetic());
    let v_nuclear = operator(ctx, n, ctx.get_nuclear_attraction_operator());
    let v_multipoles = operator(ctx, n, ctx.get_external_multipole_operator().unwrap());
    let mut bare = t.matmul(&Matrix::<f64>::new(n, n)); // zero matrix of the right type
    bare.get_data_mut().fill(0.0);
    add(&mut bare, &t);
    add(&mut bare, &v_nuclear); // T + V_nuc
    let mut h = Matrix::<f64>::new(n, n);
    add(&mut h, &bare);
    add(&mut h, &v_multipoles);

    let eri = ERI.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    let g_tensor = Array4DView::new(n, n, n, n, &eri);

    let classical = ctx.compute_nuclear_repulsion_energy() + ctx.compute_nuclear_multipole_interaction_energy();
    let x = s.symmetric_inverse_sqrt();
    let n_occupied = N_ELECTRONS / 2;

    let density_from = |fock: &Matrix<f64>| -> Matrix<f64> {
        let transformed = x.view_transposed().matmul(fock).matmul(&x);
        let (_, coefficients_orthogonal) = transformed.eigenvectors();
        let coefficients = x.matmul(&coefficients_orthogonal);
        let mut occupied = Matrix::<f64>::new(n, n_occupied);
        for j in 0..n_occupied {
            occupied.get_col_mut(j).copy_from_slice(coefficients.get_col(j));
        }
        let mut p = occupied.matmul(&occupied.view_transposed());
        for value in p.get_data_mut() {
            *value *= 2.0;
        }
        p
    };

    let mut density = density_from(&h); // core guess
    let mut fock_history: Vec<Matrix<f64>> = Vec::new();
    let mut error_history: Vec<Matrix<f64>> = Vec::new();
    let mut previous_energy = 0.0;
    for iteration in 1..=200 {
        let mut j = Matrix::<f64>::new(n, n);
        let mut k = Matrix::<f64>::new(n, n);
        aedelsten::fock_constructors::construct_j_and_k_stored(&g_tensor, &density, &mut j, &mut k, 1.0, -0.5);
        let mut g = Matrix::<f64>::new(n, n);
        add(&mut g, &j);
        add(&mut g, &k);

        let mut fock = Matrix::<f64>::new(n, n);
        add(&mut fock, &h);
        add(&mut fock, &g);
        let mut induction_energy = 0.0;
        if induction {
            let mut v_ind = vec![0.0; n * n];
            let report = ctx.compute_induction(density.get_data(), &mut v_ind).unwrap();
            induction_energy = report.energy;
            add(&mut fock, &Matrix::<f64>::new_from_vec(n, n, v_ind.into_boxed_slice()));
        }
        let electronic = trace_product(&density, &bare) + 0.5 * trace_product(&density, &g);
        let energy = electronic + trace_product(&density, &v_multipoles) + classical + induction_energy;

        // DIIS on the orthogonalised commutator FPS - SPF.
        let commutator = {
            let fps = fock.matmul(&density).matmul(&s);
            let mut c = Matrix::<f64>::new(n, n);
            add(&mut c, &fps);
            add(&mut c, &fps.transpose().map_scaled(-1.0));
            x.view_transposed().matmul(&c).matmul(&x)
        };
        let error_norm = commutator.get_data().iter().fold(0.0_f64, |w, v| w.max(v.abs()));
        if (energy - previous_energy).abs() < 1e-11 && error_norm < 1e-8 {
            return Converged { energy, density, electronic_energy: electronic, iterations: iteration };
        }
        previous_energy = energy;

        fock_history.push(fock);
        error_history.push(commutator);
        if fock_history.len() > 8 {
            fock_history.remove(0);
            error_history.remove(0);
        }
        let m = fock_history.len();
        let extrapolated = if m >= 2 {
            let mut b = Matrix::<f64>::new(m + 1, m + 1);
            for a in 0..m {
                for c in 0..m {
                    b[(a, c)] = trace_product(&error_history[a], &error_history[c]);
                }
                b[(a, m)] = -1.0;
                b[(m, a)] = -1.0;
            }
            let mut rhs = vec![0.0; m + 1];
            rhs[m] = -1.0;
            let weights = b.backslash(&rhs);
            let mut f = Matrix::<f64>::new(n, n);
            for a in 0..m {
                for (dst, src) in f.get_data_mut().iter_mut().zip(fock_history[a].get_data()) {
                    *dst += weights[a] * src;
                }
            }
            f
        } else {
            let mut f = Matrix::<f64>::new(n, n);
            add(&mut f, &fock_history[0]);
            f
        };
        density = density_from(&extrapolated);
    }
    panic!("SCF did not converge");
}

trait Scaled {
    fn map_scaled(&self, factor: f64) -> Matrix<f64>;
}
impl Scaled for Matrix<f64> {
    fn map_scaled(&self, factor: f64) -> Matrix<f64> {
        let mut m = Matrix::<f64>::new(self.rows(), self.cols());
        for (dst, src) in m.get_data_mut().iter_mut().zip(self.get_data()) {
            *dst = factor * src;
        }
        m
    }
}

fn report(label: &str, ours: f64, reference: f64) -> f64 {
    let difference = ours - reference;
    println!("{label:<34} ours {ours:>22.12}   Dalton {reference:>22.12}   diff {difference:>10.3e}");
    difference
}

#[test]
fn geometry_and_classical_energies_agree_with_dalton() {
    let (ctx, _n) = setup();
    // Agreement to 2e-8 Eh (2e-10 relative): what remains is Dalton's own rounding of the geometry conversion.
    let d = report("nuclear repulsion", ctx.compute_nuclear_repulsion_energy(), DALTON_NUCLEAR_REPULSION);
    assert!(d.abs() < 1e-7, "the geometry (units, nuclei) is not the same as Dalton's");
}

#[test]
fn pe_hf_energy_and_its_components_agree_with_dalton() {
    let (mut ctx, n) = setup();
    ctx.set_polarization(PolarizationOptions {
        solver: SolverOptions { threshold: 1e-12, ..SolverOptions::default() },
        chunking: SiteChunking::Sites(4),
        ..PolarizationOptions::default()
    })
    .unwrap();
    let result = scf(&mut ctx, n, true);
    println!("converged in {} iterations", result.iterations);

    // Everything that follows is evaluated at the converged density.
    let p = &result.density;
    let v_multipoles = operator(&ctx, n, ctx.get_external_multipole_operator().unwrap());
    let es_electronic = trace_product(p, &v_multipoles);
    let es_nuclear = ctx.compute_nuclear_multipole_interaction_energy();

    let mut dipoles = vec![0.0; 30];
    let mut operator_unused = vec![0.0; n * n];
    let last = ctx.compute_induction(p.get_data(), &mut operator_unused).unwrap();
    ctx.get_induced_dipoles(&mut dipoles).unwrap();
    let field_of = |sources: StaticFieldSources| {
        let mut field = vec![0.0; 30];
        ctx.compute_static_field(SiteSelection::Polarizable, sources, &mut field).unwrap();
        field
    };
    let nuclei_only = StaticFieldSources { nuclei: true, core_aimps: false, embedding_aimps: false, external_multipoles: false };
    let multipoles_only = StaticFieldSources { nuclei: false, core_aimps: false, embedding_aimps: false, external_multipoles: true };
    let mut electronic_field = vec![0.0; 30];
    ctx.compute_electronic_field(p.get_data(), SiteSelection::Polarizable, &SiteChunking::Sites(4), &mut electronic_field).unwrap();
    let half_mu_dot = |field: &[f64]| -> f64 { -0.5 * dipoles.iter().zip(field).map(|(m, f)| m * f).sum::<f64>() };
    let pol_electronic = half_mu_dot(&electronic_field);
    let pol_nuclear = half_mu_dot(&field_of(nuclei_only));
    let pol_multipole = half_mu_dot(&field_of(multipoles_only));

    let embedding = es_electronic + es_nuclear + last.energy;
    println!();
    let diffs = [
        report("total energy", result.energy, DALTON_TOTAL_ENERGY),
        report("QM electronic energy", result.electronic_energy, DALTON_ELECTRONIC_ENERGY),
        report("embedding energy", embedding, DALTON_EMBEDDING_ENERGY),
        report("electrostatic, electronic", es_electronic, DALTON_ES_ELECTRONIC),
        report("electrostatic, nuclear", es_nuclear, DALTON_ES_NUCLEAR),
        report("polarization, electronic", pol_electronic, DALTON_POL_ELECTRONIC),
        report("polarization, nuclear", pol_nuclear, DALTON_POL_NUCLEAR),
        report("polarization, multipole", pol_multipole, DALTON_POL_MULTIPOLE),
        report("polarization, total", last.energy, DALTON_POL_TOTAL),
    ];
    // The parts add up to the induction energy by construction.
    assert!((pol_electronic + pol_nuclear + pol_multipole - last.energy).abs() < 1e-12);
    // Tolerances. The total and its electrostatic parts agree to a few 1e-8 Eh. The electronic and nuclear parts of the
    // polarization energy are first order in the density and in the dipoles, and Dalton stopped its SCF at a gradient
    // of 9e-6 (ours is converged to 1e-8), so they only agree to about 2e-7 -- with opposite signs, which is why the
    // total polarization energy (4.5e-9) and the total energy (1.9e-8) are so much better.
    let tolerances = [
        1e-7, // total energy
        2e-7, // QM electronic energy
        1e-7, // embedding energy
        1e-7, // electrostatic, electronic
        1e-9, // electrostatic, nuclear: classical, any order
        5e-7, // polarization, electronic
        5e-7, // polarization, nuclear
        1e-8, // polarization, multipole
        5e-8, // polarization, total
    ];
    for (difference, tolerance) in diffs.iter().zip(tolerances) {
        assert!(difference.abs() < tolerance, "difference {difference:e} exceeds {tolerance:e}");
    }
}

// The two conventions at one and the same (contact-free-converged) density: the exact one is 4.2e-4 Eh too negative
// in the electronic-multipole energy -- the contact-free one reproduces Dalton, the exact one does not.
#[test]
fn the_exact_convention_differs_from_dalton_by_the_contact_energy() {
    let (mut ctx, n) = setup();
    ctx.set_polarization(PolarizationOptions { solver: SolverOptions { threshold: 1e-12, ..SolverOptions::default() }, ..PolarizationOptions::default() }).unwrap();
    let density = scf(&mut ctx, n, true).density;
    ctx.free_polarization();
    let energy_of = |convention: aedelsten::MultipoleConvention, ctx: &mut AedelstenContext| {
        ctx.set_multipole_convention(convention);
        trace_product(&density, &operator(ctx, n, ctx.get_external_multipole_operator().unwrap()))
    };
    let contact_free = energy_of(aedelsten::MultipoleConvention::ContactFree, &mut ctx);
    let exact = energy_of(aedelsten::MultipoleConvention::Exact, &mut ctx);
    report("electrostatic, electronic (contact-free)", contact_free, DALTON_ES_ELECTRONIC);
    report("electrostatic, electronic (exact)", exact, DALTON_ES_ELECTRONIC);
    assert!((contact_free - DALTON_ES_ELECTRONIC).abs() < 1e-7);
    let contact_energy = exact - contact_free;
    assert!(contact_energy < -4.0e-4 && contact_energy > -4.3e-4, "contact energy {contact_energy:e}");
}
