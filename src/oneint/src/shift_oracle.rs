// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! An oracle for the derivatives of the one-electron integrals with respect to the basis centres, from the value
//! kernels alone: along one axis, d/dA of `G_i = (x - A)^i exp(-a (x - A)^2)` is `2a G_(i+1) - i G_(i-1)`, so any
//! derivative of a primitive integral is a short sum of the unnormalized value integrals at shifted Cartesian
//! exponents (the second, `4a^2 G_(i+2) - 2a (2i + 1) G_i + i (i - 1) G_(i-2)`, is the first applied twice). No
//! derivative code enters it. A single source's derivatives follow from translational invariance,
//! `d/dC = -(d/dA + d/dB)` on that source's integral.
//!
//! The oracle is checked here against the first-derivative kernels directly, and its second derivatives against
//! central differences of those kernels, for the overlap, the kinetic energy, point charges, Gaussian charges,
//! screened charges, point multipoles and p.Vp, every pair of l up to g.

use std::collections::HashMap;

use arrays::array3d::Array3DViewMut;
use arrays::matrix::{MatrixTrait, MatrixViewMut};
use arrays::scratch::{Scratch, ByteUnit};
use basis::charge_carriers::{
    IndexedGaussianCharge, PointCharge, PointChargeTrait, PointMultipole, ScreenedChargeTrait, ScreenedCoulombPotential, SiteTrait,
};

use crate::additional_arguments::EmptyAdditionalArguments;
use crate::gradient_kernels::{CoulombGradientKernel, GradientArguments, KineticGradient, OneElectronIntegralGradientKernel, OverlapGradient};
use crate::kernels::{CoulombKernel, ElectricFieldIntegrals, FiniteMode, KineticKernel, MultipoleSourceMode, OneElectronIntegralKernel, OverlapKernel, PointMode, ScreenMode};
use crate::picture_change_kernels::{PvpGradientKernel, PvpKernel};

/// The highest angular momentum tested (g).
const L_MAX: usize = 4;

/// The highest angular momentum the oracle reads: two above the tested, for the second derivatives.
const L_READ: usize = L_MAX + 2;

/// One primitive pair and one source: exponents a at A, b at B, the source at C.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    a: f64,
    b: f64,
    pos_a: [f64; 3],
    pos_b: [f64; 3],
    pos_c: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
enum Centre {
    A,
    B,
    C,
}

impl Geometry {
    /// The geometry with `centre` moved by `step` along `axis`.
    fn displaced(&self, centre: Centre, axis: usize, step: f64) -> Geometry {
        let mut moved = *self;
        let position = match centre {
            Centre::A => &mut moved.pos_a,
            Centre::B => &mut moved.pos_b,
            Centre::C => &mut moved.pos_c,
        };
        position[axis] += step;
        moved
    }

    /// `p`, `ab = a b / p`, `B - A` and `P`, as the kernels take them.
    fn pair(&self) -> (f64, f64, [f64; 3], [f64; 3]) {
        let p = self.a + self.b;
        let separation = std::array::from_fn(|k| self.pos_b[k] - self.pos_a[k]);
        let centre = std::array::from_fn(|k| (self.a * self.pos_a[k] + self.b * self.pos_b[k]) / p);
        (p, self.a * self.b / p, separation, centre)
    }
}

/// (A, B, C): the source off the pair's axis, close to the product centre, far away, and a one-centre pair.
const POSITIONS: [([f64; 3], [f64; 3], [f64; 3]); 4] = [
    ([0.0, 0.0, 0.0], [0.3, -1.1, 0.8], [1.4, 0.6, -0.9]),
    ([0.2, 0.1, -0.3], [-0.4, 0.9, 0.5], [0.05, 0.42, 0.11]),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 1.7], [6.0, -7.5, 3.2]),
    ([0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.7, -0.2, 0.4]),
];

/// The pair's exponents (a, b): balanced, diffuse with tight, tight with diffuse.
const EXPONENTS: [(f64, f64); 3] = [(1.3, 0.7), (0.08, 12.0), (35.0, 0.25)];

fn geometries() -> impl Iterator<Item = Geometry> {
    POSITIONS
        .iter()
        .flat_map(|&(pos_a, pos_b, pos_c)| EXPONENTS.iter().map(move |&(a, b)| Geometry { a, b, pos_a, pos_b, pos_c }))
}

/// The Cartesian exponents of a shell of angular momentum l, in the kernels' order.
fn components(l: usize) -> Vec<[usize; 3]> {
    cart2sph::get_cartesian_quantum_numbers_m(l).iter_col().map(|lmn| [lmn[0], lmn[1], lmn[2]]).collect()
}

/// A one-electron operator under test: its value kernel and its first-derivative kernel, each for the shells of
/// angular momenta (la, lb) of one primitive pair, normalized Cartesian, column-major `na x nb` (three such planes,
/// x, y, z, for a derivative), from the given scratch. `has_source` says whether the operator has the single
/// source at C.
struct Operator<V, G>
where
    V: Fn(&mut Scratch, &Geometry, usize, usize) -> Vec<f64>,
    G: Fn(&mut Scratch, &Geometry, usize, usize, GradientArguments) -> Vec<f64>,
{
    name: &'static str,
    value: V,
    gradient: G,
    has_source: bool,
}

/// The value kernel `K` on one primitive pair, weight 1.
fn value_of<'c, K>(scratch: &mut Scratch, arguments: &K::AdditionalArguments<'c>, geometry: &Geometry, la: usize, lb: usize) -> Vec<f64>
where
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = MatrixViewMut<'b, f64>> + 'c,
{
    let (cart_a, cart_b) = (cart2sph::get_cartesian_quantum_numbers_m(la), cart2sph::get_cartesian_quantum_numbers_m(lb));
    let (na, nb) = (cart_a.cols(), cart_b.cols());
    let arena = scratch.get_arena();
    let mut buffer = K::kernel_buffer_allocation(la, lb, &arena, arguments).unwrap();
    let (p, ab, separation, centre) = geometry.pair();
    let mut out = vec![0.0; na * nb];
    K::compute(&mut MatrixViewMut::new(na, nb, &mut out[..]), &mut buffer, geometry.a, geometry.b, p, ab, &cart_a, &cart_b, &separation, &centre, arguments, 1.0);
    out
}

/// The first-derivative kernel `K` on one primitive pair, weight 1: the x, y, z planes of the selected terms.
fn gradient_of<'c, K>(scratch: &mut Scratch, arguments: &K::AdditionalArguments<'c>, geometry: &Geometry, la: usize, lb: usize, terms: GradientArguments) -> Vec<f64>
where
    K: for<'b> OneElectronIntegralGradientKernel<KernelBatch<'b> = Array3DViewMut<'b, f64>> + 'c,
{
    let (cart_a, cart_b) = (cart2sph::get_cartesian_quantum_numbers_m(la), cart2sph::get_cartesian_quantum_numbers_m(lb));
    let (na, nb) = (cart_a.cols(), cart_b.cols());
    let arena = scratch.get_arena();
    let mut buffer = K::kernel_buffer_allocation(la, lb, &arena, arguments).unwrap();
    let (p, ab, separation, centre) = geometry.pair();
    let mut out = vec![0.0; 3 * na * nb];
    K::compute(&mut Array3DViewMut::new(na, nb, 3, &mut out[..]), &mut buffer, geometry.a, geometry.b, p, ab, &cart_a, &cart_b, &separation, &centre, arguments, &terms, 1.0);
    out
}

/// The unnormalized primitive integrals of one operator at one geometry, for every pair of Cartesian components up
/// to `L_READ`, from its value kernel.
struct Unnormalized {
    a: f64,
    b: f64,
    integrals: HashMap<([usize; 3], [usize; 3]), f64>,
}

impl Unnormalized {
    fn new(value: impl Fn(&mut Scratch, &Geometry, usize, usize) -> Vec<f64>, scratch: &mut Scratch, geometry: &Geometry) -> Self {
        let mut integrals = HashMap::new();
        for la in 0..=L_READ {
            for lb in 0..=L_READ {
                let (cart_a, cart_b) = (components(la), components(lb));
                let batch = value(scratch, geometry, la, lb);
                for (jb, lmn_b) in cart_b.iter().enumerate() {
                    for (ia, lmn_a) in cart_a.iter().enumerate() {
                        let norm = cart2sph::compute_cart_gaussian_norm(geometry.a, lmn_a[0], lmn_a[1], lmn_a[2])
                            * cart2sph::compute_cart_gaussian_norm(geometry.b, lmn_b[0], lmn_b[1], lmn_b[2]);
                        integrals.insert((*lmn_a, *lmn_b), batch[ia + cart_a.len() * jb] / norm);
                    }
                }
            }
        }
        Self { a: geometry.a, b: geometry.b, integrals }
    }

    /// The derivative of the normalized shells (la, lb) with respect to A along each of `axes_a` and to B along each
    /// of `axes_b`, column-major `na x nb`.
    fn derivative(&self, la: usize, lb: usize, axes_a: &[usize], axes_b: &[usize]) -> Vec<f64> {
        let (cart_a, cart_b) = (components(la), components(lb));
        let mut out = vec![0.0; cart_a.len() * cart_b.len()];
        for (jb, lmn_b) in cart_b.iter().enumerate() {
            for (ia, lmn_a) in cart_a.iter().enumerate() {
                let norm = cart2sph::compute_cart_gaussian_norm(self.a, lmn_a[0], lmn_a[1], lmn_a[2])
                    * cart2sph::compute_cart_gaussian_norm(self.b, lmn_b[0], lmn_b[1], lmn_b[2]);
                let terms_a = shifted(self.a, *lmn_a, axes_a);
                let terms_b = shifted(self.b, *lmn_b, axes_b);
                let sum: f64 = terms_a
                    .iter()
                    .flat_map(|(shift_a, c_a)| terms_b.iter().map(move |(shift_b, c_b)| c_a * c_b * self.integrals[&(*shift_a, *shift_b)]))
                    .sum();
                out[ia + cart_a.len() * jb] = norm * sum;
            }
        }
        out
    }
}

/// The derivative of the unnormalized Cartesian Gaussian with exponents `lmn` and exponent `exponent` with respect
/// to its centre along each of `axes` in turn, as (exponents, coefficient) terms.
fn shifted(exponent: f64, lmn: [usize; 3], axes: &[usize]) -> Vec<([usize; 3], f64)> {
    let mut terms = vec![(lmn, 1.0)];
    for &k in axes {
        terms = terms
            .into_iter()
            .flat_map(|(l, c)| {
                let mut up = l;
                up[k] += 1;
                let lowered = (l[k] > 0).then(|| {
                    let mut down = l;
                    down[k] -= 1;
                    (down, -(l[k] as f64) * c)
                });
                std::iter::once((up, 2.0 * exponent * c)).chain(lowered)
            })
            .collect();
    }
    terms
}

/// `x + y`, element by element.
fn sum(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(u, v)| u + v).collect()
}

/// `-x`.
fn negated(x: &[f64]) -> Vec<f64> {
    x.iter().map(|u| -u).collect()
}

/// Plane `axis` of a stack of three `n`-element planes.
fn plane(planes: &[f64], axis: usize) -> &[f64] {
    let n = planes.len() / 3;
    &planes[axis * n..(axis + 1) * n]
}

/// The largest absolute value.
fn largest(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Compares each (what, expected, got) of one case, the differences relative to the largest expected value of the
/// whole case (a component that vanishes by symmetry has no scale of its own), and at least `floor` (the case's
/// value integrals: a one-centre pair's first derivatives all vanish); absolute where the whole case vanishes by
/// parity (a one-centre pair of odd l_a + l_b at even order). Returns the largest.
fn compare(case: &str, comparisons: &[(String, Vec<f64>, Vec<f64>)], floor: f64, tolerance: f64) -> f64 {
    let scale = comparisons.iter().fold(floor, |m, (_, expected, _)| m.max(largest(expected)));
    let scale = if scale > 0.0 { scale } else { 1.0 };
    comparisons.iter().fold(0.0_f64, |m, (what, expected, got)| {
        let difference = expected.iter().zip(got).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs())) / scale;
        assert!(difference < tolerance, "{what}, {case}: {difference:e}");
        m.max(difference)
    })
}

/// The derivative of `f` with respect to `centre` along `axis` by the five-point central difference, the step
/// shrinking with the tighter exponent.
fn central_difference(
    f: impl Fn(&mut Scratch, &Geometry) -> Vec<f64>,
    scratch: &mut Scratch,
    geometry: &Geometry,
    centre: Centre,
    axis: usize,
) -> Vec<f64> {
    let h = 2e-3 / geometry.a.max(geometry.b).max(1.0).sqrt();
    let mut at = |steps: f64| f(scratch, &geometry.displaced(centre, axis, steps * h));
    let (m2, m1, p1, p2) = (at(-2.0), at(-1.0), at(1.0), at(2.0));
    m2.iter()
        .zip(&m1)
        .zip(p1.iter().zip(&p2))
        .map(|((m2, m1), (p1, p2))| (m2 - 8.0 * m1 + 8.0 * p1 - p2) / (12.0 * h))
        .collect()
}

/// A scratch buffer of the size the largest pair of the check needs, measured: the value kernel at (`L_READ`, `L_READ`)
/// and the derivative kernel at (`L_MAX`, `L_MAX`) with each of `terms`, every call from the start of the buffer;
/// a smaller pair needs less.
fn sized_for_largest_pair<V, G>(operator: &Operator<V, G>, terms: &[GradientArguments]) -> Scratch
where
    V: Fn(&mut Scratch, &Geometry, usize, usize) -> Vec<f64>,
    G: Fn(&mut Scratch, &Geometry, usize, usize, GradientArguments) -> Vec<f64>,
{
    let mut probe = Scratch::new(64, ByteUnit::MiB).unwrap();
    let geometry = geometries().next().unwrap();
    (operator.value)(&mut probe, &geometry, L_READ, L_READ);
    for &selected in terms {
        (operator.gradient)(&mut probe, &geometry, L_MAX, L_MAX, selected);
    }
    Scratch::new(probe.peak_bytes().div_ceil(1024), ByteUnit::KiB).unwrap()
}

/// The oracle's first derivatives against the operator's derivative kernel, and its second derivatives against
/// central differences of that kernel, for every pair of l up to `L_MAX` at every geometry. Returns the largest
/// relative differences (first, second).
fn check<V, G>(operator: &Operator<V, G>) -> (f64, f64)
where
    V: Fn(&mut Scratch, &Geometry, usize, usize) -> Vec<f64>,
    G: Fn(&mut Scratch, &Geometry, usize, usize, GradientArguments) -> Vec<f64>,
{
    let (mut largest_first, mut largest_second): (f64, f64) = (0.0, 0.0);
    let on_a = GradientArguments::new(true, false, None);
    let on_b = GradientArguments::new(false, true, None);
    let on_c = GradientArguments::new(false, false, Some(0));
    let mut terms = vec![on_a, on_b];
    if operator.has_source {
        terms.push(on_c);
    }
    let scratch = &mut sized_for_largest_pair(operator, &terms);
    for geometry in geometries() {
        let oracle = Unnormalized::new(&operator.value, scratch, &geometry);
        for la in 0..=L_MAX {
            for lb in 0..=L_MAX {
                let case = format!("{}, l {la} {lb}, {geometry:?}", operator.name);
                let floor = largest(&oracle.derivative(la, lb, &[], &[]));
                let kernel_a = (operator.gradient)(scratch, &geometry, la, lb, on_a);
                let kernel_b = (operator.gradient)(scratch, &geometry, la, lb, on_b);
                let kernel_c = operator.has_source.then(|| (operator.gradient)(scratch, &geometry, la, lb, on_c));
                let mut first = Vec::new();
                for x in 0..3 {
                    let d_a = oracle.derivative(la, lb, &[x], &[]);
                    let d_b = oracle.derivative(la, lb, &[], &[x]);
                    if let Some(kernel_c) = &kernel_c {
                        first.push((format!("d/dC_{x}"), negated(&sum(&d_a, &d_b)), plane(kernel_c, x).to_vec()));
                    }
                    first.push((format!("d/dA_{x}"), d_a, plane(&kernel_a, x).to_vec()));
                    first.push((format!("d/dB_{x}"), d_b, plane(&kernel_b, x).to_vec()));
                }
                largest_first = largest_first.max(compare(&case, &first, floor, 1e-10));

                let differentiated = |scratch: &mut Scratch, terms: GradientArguments, centre: Centre, y: usize| {
                    central_difference(|scratch, moved| (operator.gradient)(scratch, moved, la, lb, terms), scratch, &geometry, centre, y)
                };
                let mut second = Vec::new();
                for y in 0..3 {
                    let by_a_y = differentiated(scratch, on_a, Centre::A, y);
                    let a_by_b_y = differentiated(scratch, on_a, Centre::B, y);
                    let b_by_b_y = differentiated(scratch, on_b, Centre::B, y);
                    let by_c_y = operator.has_source.then(|| {
                        (
                            differentiated(scratch, on_a, Centre::C, y),
                            differentiated(scratch, on_b, Centre::C, y),
                            differentiated(scratch, on_c, Centre::C, y),
                        )
                    });
                    for x in 0..3 {
                        let aa = oracle.derivative(la, lb, &[y, x], &[]);
                        let ab = oracle.derivative(la, lb, &[x], &[y]);
                        let ba = oracle.derivative(la, lb, &[y], &[x]);
                        let bb = oracle.derivative(la, lb, &[], &[y, x]);
                        if let Some((c_a, c_b, c_c)) = &by_c_y {
                            second.push((format!("[C_{y} A_{x}]"), negated(&sum(&aa, &ab)), plane(c_a, x).to_vec()));
                            second.push((format!("[C_{y} B_{x}]"), negated(&sum(&ba, &bb)), plane(c_b, x).to_vec()));
                            second.push((format!("[C_{y} C_{x}]"), sum(&sum(&aa, &ab), &sum(&ba, &bb)), plane(c_c, x).to_vec()));
                        }
                        second.push((format!("[A_{y} A_{x}]"), aa, plane(&by_a_y, x).to_vec()));
                        second.push((format!("[B_{y} A_{x}]"), ab, plane(&a_by_b_y, x).to_vec()));
                        second.push((format!("[B_{y} B_{x}]"), bb, plane(&b_by_b_y, x).to_vec()));
                    }
                }
                largest_second = largest_second.max(compare(&case, &second, floor, 1e-8));
            }
        }
    }
    eprintln!("{}: largest relative differences, first derivatives {largest_first:e}, second {largest_second:e}", operator.name);
    (largest_first, largest_second)
}

#[test]
fn the_shift_oracle_gives_the_overlap_derivatives() {
    check(&Operator {
        name: "overlap",
        value: |w: &mut Scratch, g: &Geometry, la, lb| value_of::<OverlapKernel>(w, &EmptyAdditionalArguments, g, la, lb),
        gradient: |w: &mut Scratch, g: &Geometry, la, lb, terms| gradient_of::<OverlapGradient>(w, &EmptyAdditionalArguments, g, la, lb, terms),
        has_source: false,
    });
}

#[test]
fn the_shift_oracle_gives_the_kinetic_derivatives() {
    check(&Operator {
        name: "kinetic",
        value: |w: &mut Scratch, g: &Geometry, la, lb| value_of::<KineticKernel>(w, &EmptyAdditionalArguments, g, la, lb),
        gradient: |w: &mut Scratch, g: &Geometry, la, lb, terms| gradient_of::<KineticGradient>(w, &EmptyAdditionalArguments, g, la, lb, terms),
        has_source: false,
    });
}

#[test]
fn the_shift_oracle_gives_the_point_charge_derivatives() {
    let sources = |g: &Geometry| [PointCharge::new(g.pos_c, 1.7)];
    check(&Operator {
        name: "point charge",
        value: |w: &mut Scratch, g: &Geometry, la, lb| value_of::<CoulombKernel<PointCharge, PointMode>>(w, &&sources(g)[..], g, la, lb),
        gradient: |w: &mut Scratch, g: &Geometry, la, lb, terms| gradient_of::<CoulombGradientKernel<PointCharge, PointMode>>(w, &&sources(g)[..], g, la, lb, terms),
        has_source: true,
    });
}

#[test]
fn the_shift_oracle_gives_the_gaussian_charge_derivatives() {
    let sources = |g: &Geometry| [IndexedGaussianCharge::new(0, g.pos_c, 1.7, 2.3)];
    check(&Operator {
        name: "Gaussian charge",
        value: |w: &mut Scratch, g: &Geometry, la, lb| value_of::<CoulombKernel<IndexedGaussianCharge, FiniteMode>>(w, &&sources(g)[..], g, la, lb),
        gradient: |w: &mut Scratch, g: &Geometry, la, lb, terms| gradient_of::<CoulombGradientKernel<IndexedGaussianCharge, FiniteMode>>(w, &&sources(g)[..], g, la, lb, terms),
        has_source: true,
    });
}

/// A charge with a screen of four Gaussians from diffuse to tight.
struct Screen {
    position: [f64; 3],
    screen: ScreenedCoulombPotential,
}
impl SiteTrait for Screen {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }
}
impl PointChargeTrait for Screen {
    fn get_charge(&self) -> f64 {
        2.0
    }
}
impl ScreenedChargeTrait for Screen {
    fn get_screened_coulomb_potential(&self) -> &ScreenedCoulombPotential {
        &self.screen
    }
}

#[test]
fn the_shift_oracle_gives_the_screen_derivatives() {
    let sources = |g: &Geometry| {
        [Screen { position: g.pos_c, screen: ScreenedCoulombPotential::new(vec![0.05, 0.9, 7.5, 60.0], vec![-1.3, 0.4, -0.8, 0.25]) }]
    };
    check(&Operator {
        name: "screen",
        value: |w: &mut Scratch, g: &Geometry, la, lb| value_of::<CoulombKernel<Screen, ScreenMode>>(w, &&sources(g)[..], g, la, lb),
        gradient: |w: &mut Scratch, g: &Geometry, la, lb, terms| gradient_of::<CoulombGradientKernel<Screen, ScreenMode>>(w, &&sources(g)[..], g, la, lb, terms),
        has_source: true,
    });
}

#[test]
fn the_shift_oracle_gives_the_multipole_derivatives() {
    // A charge, a dipole and a quadrupole (xx, xy, xz, yy, yz, zz).
    let moments = [0.6, -0.3, 0.8, 0.45, 1.1, -0.2, 0.35, -0.7, 0.15, -0.4];
    let sources = move |g: &Geometry| [PointMultipole::new(g.pos_c, 2, moments)];
    check(&Operator {
        name: "multipole",
        value: |w: &mut Scratch, g: &Geometry, la, lb| value_of::<CoulombKernel<PointMultipole, MultipoleSourceMode>>(w, &&sources(g)[..], g, la, lb),
        gradient: |w: &mut Scratch, g: &Geometry, la, lb, terms| gradient_of::<CoulombGradientKernel<PointMultipole, MultipoleSourceMode>>(w, &&sources(g)[..], g, la, lb, terms),
        has_source: true,
    });
}

#[test]
fn the_shift_oracle_gives_the_pvp_derivatives() {
    let sources = |g: &Geometry| [PointCharge::new(g.pos_c, 1.7)];
    check(&Operator {
        name: "p.Vp",
        value: |w: &mut Scratch, g: &Geometry, la, lb| value_of::<PvpKernel<PointCharge, PointMode>>(w, &&sources(g)[..], g, la, lb),
        gradient: |w: &mut Scratch, g: &Geometry, la, lb, terms| gradient_of::<PvpGradientKernel<PointCharge, PointMode>>(w, &&sources(g)[..], g, la, lb, terms),
        has_source: true,
    });
}

/// The planes of an array-batch value kernel `K` on one primitive pair, weight 1: `n_planes` planes of `na x nb`.
fn planes_of<'c, K>(scratch: &mut Scratch, arguments: &K::AdditionalArguments<'c>, geometry: &Geometry, la: usize, lb: usize, n_planes: usize) -> Vec<f64>
where
    K: for<'b> OneElectronIntegralKernel<KernelBatch<'b> = Array3DViewMut<'b, f64>> + 'c,
{
    let (cart_a, cart_b) = (cart2sph::get_cartesian_quantum_numbers_m(la), cart2sph::get_cartesian_quantum_numbers_m(lb));
    let (na, nb) = (cart_a.cols(), cart_b.cols());
    let arena = scratch.get_arena();
    let mut buffer = K::kernel_buffer_allocation(la, lb, &arena, arguments).unwrap();
    let (p, ab, separation, centre) = geometry.pair();
    let mut out = vec![0.0; n_planes * na * nb];
    K::compute(&mut Array3DViewMut::new(na, nb, n_planes, &mut out[..]), &mut buffer, geometry.a, geometry.b, p, ab, &cart_a, &cart_b, &separation, &centre, arguments, 1.0);
    out
}

/// The electric-field integrals `<A| (r - C) / |r - C|^3 |B>` of one site are `d/dC <A| 1/r_C |B>`, and the point
/// charge's kernel gives `V = -q <A| 1/r_C |B>`, so the field is `(d/dA + d/dB) V` at q = 1 and its derivatives with
/// respect to A, B and C are V's second-order blocks: the field kernel against the oracle of the point charge's
/// value kernel, and central differences of the field kernel against those blocks.
#[test]
fn the_shift_oracle_gives_the_electric_field_integrals_and_their_derivatives() {
    let site = |g: &Geometry| [PointCharge::new(g.pos_c, 1.0)];
    let value = |w: &mut Scratch, g: &Geometry, la, lb| value_of::<CoulombKernel<PointCharge, PointMode>>(w, &&site(g)[..], g, la, lb);
    let field = |w: &mut Scratch, g: &Geometry, la, lb| planes_of::<ElectricFieldIntegrals<PointCharge>>(w, &&site(g)[..], g, la, lb, 3);
    // Sized for the largest pair, as `sized_for_largest_pair`.
    let mut probe = Scratch::new(64, ByteUnit::MiB).unwrap();
    let first = geometries().next().unwrap();
    value(&mut probe, &first, L_READ, L_READ);
    field(&mut probe, &first, L_MAX, L_MAX);
    let scratch = &mut Scratch::new(probe.peak_bytes().div_ceil(1024), ByteUnit::KiB).unwrap();
    let (mut largest_value, mut largest_derivative): (f64, f64) = (0.0, 0.0);
    for geometry in geometries() {
        let oracle = Unnormalized::new(value, scratch, &geometry);
        for la in 0..=L_MAX {
            for lb in 0..=L_MAX {
                let case = format!("electric field, l {la} {lb}, {geometry:?}");
                let floor = largest(&oracle.derivative(la, lb, &[], &[]));
                let got = field(scratch, &geometry, la, lb);
                let values: Vec<_> = (0..3)
                    .map(|x| (format!("F_{x}"), sum(&oracle.derivative(la, lb, &[x], &[]), &oracle.derivative(la, lb, &[], &[x])), plane(&got, x).to_vec()))
                    .collect();
                largest_value = largest_value.max(compare(&case, &values, floor, 1e-10));

                let mut derivatives = Vec::new();
                for y in 0..3 {
                    let by = |scratch: &mut Scratch, centre: Centre| {
                        central_difference(|scratch, moved| field(scratch, moved, la, lb), scratch, &geometry, centre, y)
                    };
                    let (by_a, by_b, by_c) = (by(scratch, Centre::A), by(scratch, Centre::B), by(scratch, Centre::C));
                    for x in 0..3 {
                        let a_y = sum(&oracle.derivative(la, lb, &[y, x], &[]), &oracle.derivative(la, lb, &[y], &[x]));
                        let b_y = sum(&oracle.derivative(la, lb, &[x], &[y]), &oracle.derivative(la, lb, &[], &[y, x]));
                        derivatives.push((format!("dF_{x}/dC_{y}"), negated(&sum(&a_y, &b_y)), plane(&by_c, x).to_vec()));
                        derivatives.push((format!("dF_{x}/dA_{y}"), a_y, plane(&by_a, x).to_vec()));
                        derivatives.push((format!("dF_{x}/dB_{y}"), b_y, plane(&by_b, x).to_vec()));
                    }
                }
                largest_derivative = largest_derivative.max(compare(&case, &derivatives, floor, 1e-8));
            }
        }
    }
    eprintln!("electric field: largest relative differences, values {largest_value:e}, derivatives {largest_derivative:e}");
}
