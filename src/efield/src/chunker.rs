// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The electric-field integrals of the first sites of a share, kept between calls, and the two contractions of them:
//   - with a density D, giving the electronic field at each kept site:  F_(s,a) = D . plane_(s,a)
//   - with induced dipoles mu, giving an operator:                       V += scale * Sum_(s,a) mu_(s,a) plane_(s,a)
// (the second is the transpose of the first), plane_(s,a) the n_ao x n_ao integrals of component a at site s.
//
// Each plane's lower triangle is kept in memory, packed column by column (n_pack = n_ao (n_ao + 1) / 2 values), for as
// many sites as a memory budget holds. They are filled once (`fill_resident`, through a buffer of dense planes of at
// most `FILL_CHUNK_BYTES`) and kept until `invalidate`. A call contracts the kept planes, the fields as one product
// with the packed densities (their off-diagonal elements summed with their transposes) and the operators as one
// product with the dipoles, unpacked symmetrically; the sites not kept are the caller's. Nothing is allocated after
// construction.

use crate::site_chunking::field_integral_bytes_per_site;
use crate::EfieldError;
use arrays::matrix::{ColumnFactor, Factor, MatrixView};
use std::ops::Range;

/// The sites a resident chunker's fill evaluates per chunk at most, in bytes of their dense planes.
const FILL_CHUNK_BYTES: usize = 16 << 20;

/// The kept electric-field integrals of the first sites of a share (`Resident`), or none, with the buffer their fill
/// evaluates them in.
#[derive(Debug)]
pub struct FieldChunker {
    n_ao: usize,
    fill_sites: usize,
    buffer: Vec<f64>,
    resident: Option<Resident>,
}

/// The kept planes of the sites `kept` (the first of a share), `n_pack x 3 kept.len()`, and a scratch of
/// `n_pack x batch` for the packed densities or operators of a call.
#[derive(Debug)]
struct Resident {
    kept: Range<usize>,
    packed: Vec<f64>,
    scratch: Vec<f64>,
    filled: bool,
}

/// `n (n + 1) / 2`.
fn n_pack(n: usize) -> usize {
    n * (n + 1) / 2
}

/// The lower triangle of the column-major `n x n` `plane`, packed column by column into `packed`.
fn pack_lower(n: usize, plane: &[f64], packed: &mut [f64]) {
    let mut offset = 0;
    for (j, column) in plane.chunks_exact(n).enumerate() {
        packed[offset..offset + n - j].copy_from_slice(&column[j..]);
        offset += n - j;
    }
}

/// The packed weights of a contraction with the (not necessarily symmetric) `n x n` `density` against symmetric
/// planes: `D_jj` on the diagonal, `D_ij + D_ji` below it.
fn pack_symmetrised(n: usize, density: &[f64], packed: &mut [f64]) {
    let mut offset = 0;
    for (j, column) in density.chunks_exact(n).enumerate() {
        for (i, (value, &d_ij)) in packed[offset..offset + n - j].iter_mut().zip(&column[j..]).enumerate() {
            let i = i + j;
            // The transposed partner, D_ji.
            *value = if i == j { d_ij } else { d_ij + density[j + n * i] };
        }
        offset += n - j;
    }
}

/// `operator += packed` as a symmetric `n x n` matrix whose lower triangle `packed` holds.
fn add_unpacked(n: usize, packed: &[f64], operator: &mut [f64]) {
    let mut offset = 0;
    for j in 0..n {
        for (i, &value) in (j..n).zip(&packed[offset..offset + n - j]) {
            operator[i + n * j] += value;
            if i != j {
                operator[j + n * i] += value;
            }
        }
        offset += n - j;
    }
}

// A zeroed `Vec` of `length` doubles, or the allocation error.
fn zeroed(length: usize) -> Result<Vec<f64>, EfieldError<std::convert::Infallible>> {
    let mut values: Vec<f64> = Vec::new();
    values.try_reserve_exact(length).map_err(|_| EfieldError::Allocation { bytes: length * std::mem::size_of::<f64>() })?;
    values.resize(length, 0.0);
    Ok(values)
}

impl FieldChunker {
    /// A chunker for `n_ao` basis functions that keeps nothing: it holds no memory, and its passes add nothing.
    pub fn new(n_ao: usize) -> Self {
        Self { n_ao, fill_sites: 0, buffer: Vec::new(), resident: None }
    }

    /// The bytes of one site's kept planes: `3 n_ao (n_ao + 1) / 2` doubles.
    pub fn bytes_per_site(n_ao: usize) -> usize {
        3 * n_pack(n_ao) * std::mem::size_of::<f64>()
    }

    /// The bytes of the scratch of `batch` packed matrices that kept planes need beside them.
    pub fn scratch_bytes(n_ao: usize, batch: usize) -> usize {
        n_pack(n_ao) * batch.max(1) * std::mem::size_of::<f64>()
    }

    /// The bytes that keep the integrals of `share_len` sites, with the scratch of `batch` packed matrices.
    pub fn resident_bytes(n_ao: usize, share_len: usize, batch: usize) -> usize {
        Self::bytes_per_site(n_ao).saturating_mul(share_len).saturating_add(Self::scratch_bytes(n_ao, batch))
    }

    /// The sites of a share of `share_len` that `memory_bytes` keeps with the scratch of `batch` packed matrices.
    pub fn kept_sites(n_ao: usize, share_len: usize, batch: usize, memory_bytes: usize) -> usize {
        (memory_bytes.saturating_sub(Self::scratch_bytes(n_ao, batch)) / Self::bytes_per_site(n_ao).max(1)).min(share_len)
    }

    /// A chunker that keeps the integrals of the first sites of `share` (of `n_sites`) between calls, as many as
    /// `memory_bytes` holds (`kept_sites`), with the buffer the fill evaluates them in (`FILL_CHUNK_BYTES` of dense
    /// planes at most). Without any kept, `new`.
    pub fn with_kept(n_ao: usize, n_sites: usize, share: Range<usize>, batch: usize, memory_bytes: usize) -> Result<Self, EfieldError<std::convert::Infallible>> {
        Self::with_kept_filling(n_ao, n_sites, share, batch, memory_bytes, FILL_CHUNK_BYTES)
    }

    // `with_kept`, filling in chunks of `fill_bytes` of dense planes.
    fn with_kept_filling(n_ao: usize, n_sites: usize, share: Range<usize>, batch: usize, memory_bytes: usize, fill_bytes: usize) -> Result<Self, EfieldError<std::convert::Infallible>> {
        let kept = Self::kept_sites(n_ao, share.len(), batch, memory_bytes);
        if share.end > n_sites || kept == 0 {
            return Ok(Self::new(n_ao));
        }
        let pack = n_pack(n_ao);
        let fill_sites = (fill_bytes / field_integral_bytes_per_site(n_ao).max(1)).clamp(1, kept);
        let buffer = zeroed(3 * n_ao * n_ao * fill_sites)?;
        let resident = Resident { kept: share.start..share.start + kept, packed: zeroed(3 * kept * pack)?, scratch: zeroed(pack * batch.max(1))?, filled: false };
        Ok(Self { n_ao, fill_sites, buffer, resident: Some(resident) })
    }

    /// The sites kept, empty without any.
    pub fn kept(&self) -> Range<usize> {
        self.resident.as_ref().map_or(0..0, |r| r.kept.clone())
    }

    /// Whether the integrals of some sites are kept between calls.
    pub fn is_resident(&self) -> bool {
        self.resident.is_some()
    }

    /// Whether the kept integrals are filled (always false without them).
    pub fn is_filled(&self) -> bool {
        self.resident.as_ref().is_some_and(|r| r.filled)
    }

    /// The kept integrals no longer hold (the basis' shell pairs changed): the next `fill_resident` computes them.
    pub fn invalidate(&mut self) {
        if let Some(resident) = self.resident.as_mut() {
            resident.filled = false;
        }
    }

    /// The bytes the chunker holds: its fill buffer and the kept integrals with their scratch.
    pub fn bytes(&self) -> usize {
        let resident = self.resident.as_ref().map_or(0, |r| r.packed.len() + r.scratch.len());
        (self.buffer.len() + resident) * std::mem::size_of::<f64>()
    }

    /// Fills the kept integrals of `sites` if they are not, by `compute_chunk` per chunk (the full integrals of the
    /// sites it is given). Nothing without kept integrals.
    pub fn fill_resident<S, E, F>(&mut self, sites: &[S], mut compute_chunk: F) -> Result<(), EfieldError<E>>
    where
        F: FnMut(&mut arrays::array3d::Array3DViewMut<'_, f64>, &[S]) -> Result<(), E>,
    {
        let Some(mut resident) = self.resident.take_if(|r| !r.filled) else {
            return Ok(());
        };
        let (n, pack, size) = (self.n_ao, n_pack(self.n_ao), self.fill_sites);
        let mut result = Ok(());
        for start in resident.kept.clone().step_by(size) {
            let run = start..(start + size).min(resident.kept.end);
            let used = match self.integrals(sites, &run, &mut compute_chunk) {
                Ok(used) => used,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            };
            let first = 3 * (run.start - resident.kept.start) * pack;
            let target = &mut resident.packed[first..first + 3 * run.len() * pack];
            for (plane, packed) in used.chunks_exact(n * n).zip(target.chunks_exact_mut(pack)) {
                pack_lower(n, plane, packed);
            }
        }
        resident.filled = result.is_ok();
        self.resident = Some(resident);
        result
    }

    /// The kept planes and the packed length, when filled.
    fn filled(&mut self) -> Option<(&mut Resident, usize)> {
        let pack = n_pack(self.n_ao);
        self.resident.as_mut().filter(|r| r.filled).map(|r| (r, pack))
    }

    /// From the kept planes, when filled: the fields `F_(s,a) = Sum_ij D_ij plane_(s,a),ij` of the densities in
    /// `densities` (consecutive `n_ao x n_ao` matrices, not necessarily symmetric) at the kept sites, added to
    /// `fields`, density `m`'s to its `column` values from `m column` (3 per site, every site of the sites' list); in
    /// groups of as many densities as the scratch holds, one product with the planes a group. Nothing without filled
    /// planes.
    pub fn add_kept_fields(&mut self, densities: &[f64], fields: &mut [f64], column: usize) {
        let n = self.n_ao;
        let n2 = n * n;
        assert!(densities.len() % n2.max(1) == 0 && fields.len() == densities.len() / n2.max(1) * column, "a column of fields per density");
        let Some((resident, pack)) = self.filled() else { return };
        let Resident { kept, packed, scratch, .. } = resident;
        let planes = MatrixView::new(pack, 3 * kept.len(), packed);
        let width = (scratch.len() / pack).max(1);
        let k = densities.len() / n2;
        for first in (0..k).step_by(width) {
            let group = first..(first + width).min(k);
            for (m, weights) in group.clone().zip(scratch.chunks_exact_mut(pack)) {
                pack_symmetrised(n, &densities[m * n2..(m + 1) * n2], weights);
            }
            for (m, w) in group.zip(scratch.chunks_exact(pack)) {
                let out = &mut fields[m * column + 3 * kept.start..m * column + 3 * kept.end];
                planes.view_transposed().matvec_scaled_prealloc(w, out, 1.0, 1.0);
            }
        }
    }

    /// From the kept planes, when filled: `operator_m += scale * Sum_(s,a) mu_(m,s,a) plane_(s,a)` over the kept sites
    /// into `operators` (consecutive `n_ao x n_ao` matrices), the dipoles `mu_m` the `column` values of `dipoles` from
    /// `m column` (3 per site); the transpose of `add_kept_fields`. Nothing without filled planes.
    pub fn add_kept_operators(&mut self, dipoles: &[f64], column: usize, scale: f64, operators: &mut [f64]) {
        let n = self.n_ao;
        let n2 = n * n;
        assert!(operators.len() % n2.max(1) == 0 && dipoles.len() == operators.len() / n2.max(1) * column, "a column of dipoles per operator");
        let Some((resident, pack)) = self.filled() else { return };
        let Resident { kept, packed, scratch, .. } = resident;
        let planes = MatrixView::new(pack, 3 * kept.len(), packed);
        let w = &mut scratch[..pack];
        for (operator, mu) in operators.chunks_exact_mut(n2).zip(dipoles.chunks_exact(column)) {
            planes.matvec_scaled_prealloc(&mu[3 * kept.start..3 * kept.end], w, scale, 0.0);
            add_unpacked(n, w, operator);
        }
    }

    /// The number of basis functions.
    pub fn n_ao(&self) -> usize {
        self.n_ao
    }

    /// The integrals of the sites `run` into the buffer, as an `n_ao x n_ao x 3*run.len()` array, by `compute_chunk`.
    fn integrals<S, E, F>(&mut self, sites: &[S], run: &Range<usize>, compute_chunk: &mut F) -> Result<&[f64], EfieldError<E>>
    where
        F: FnMut(&mut arrays::array3d::Array3DViewMut<'_, f64>, &[S]) -> Result<(), E>,
    {
        assert!(run.len() <= self.fill_sites && run.end <= sites.len(), "a run of sites {run:?} beyond a chunk or the sites");
        let n_ao = self.n_ao;
        let used = &mut self.buffer[..3 * run.len() * n_ao * n_ao];
        {
            let mut view = arrays::array3d::Array3DViewMut::new(n_ao, n_ao, 3 * run.len(), &mut *used);
            compute_chunk(&mut view, &sites[run.clone()]).map_err(EfieldError::Integrals)?;
        }
        Ok(used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrays::matrix::{Matrix, MatrixTrait, MatrixTraitMut};

    const N: usize = 5;

    // Symmetric fake planes: element (i, j) of plane k of site s is a function of s, k and the unordered pair.
    fn plane_value(s: usize, k: usize, i: usize, j: usize) -> f64 {
        let (lo, hi) = (i.min(j), i.max(j));
        ((7 * s + 3 * k + 2 * lo + 5 * hi) % 11) as f64 - 5.0 + 0.1 * (s + lo) as f64
    }

    fn fake_chunk(view: &mut arrays::array3d::Array3DViewMut<'_, f64>, sites: &[usize]) -> Result<(), String> {
        use arrays::array3d::Array3DTraitMut;
        for (i_site, &s) in sites.iter().enumerate() {
            for k in 0..3 {
                let mut plane = view.view_sub_matrix_mut(3 * i_site + k);
                for j in 0..N {
                    for i in 0..N {
                        plane[(i, j)] = plane_value(s, k, i, j);
                    }
                }
            }
        }
        Ok(())
    }

    /// The kept passes of `kept` (filled) against the dense contractions over its kept sites, with densities that are
    /// not symmetric, three at a time (more than the scratch of two packed matrices holds): equal to rounding, the
    /// sites not kept untouched.
    fn passes_agree(kept: &mut FieldChunker, n_sites: usize, round: usize) {
        let k = 3;
        let column = 3 * n_sites;
        let densities: Vec<f64> = (0..k * N * N).map(|x| ((x * 13 + round) % 7) as f64 * 0.3 - 0.8).collect();
        let dipoles: Vec<f64> = (0..k * column).map(|x| ((x * 5 + round) % 9) as f64 * 0.2 - 0.7).collect();
        let (mut expected_fields, mut expected_operators) = (vec![0.0; k * column], vec![0.0; k * N * N]);
        let scale = -0.5;
        for m in 0..k {
            for s in kept.kept() {
                for c in 0..3 {
                    for j in 0..N {
                        for i in 0..N {
                            let value = plane_value(s, c, i, j);
                            expected_fields[m * column + 3 * s + c] += densities[m * N * N + i + N * j] * value;
                            expected_operators[m * N * N + i + N * j] += scale * dipoles[m * column + 3 * s + c] * value;
                        }
                    }
                }
            }
        }
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12 * y.abs().max(1.0));
        let (mut fields, mut operators) = (vec![0.0; k * column], vec![0.0; k * N * N]);
        kept.add_kept_fields(&densities, &mut fields, column);
        kept.add_kept_operators(&dipoles, column, scale, &mut operators);
        assert!(close(&fields, &expected_fields), "fields {fields:?} against {expected_fields:?}");
        assert!(close(&operators, &expected_operators), "operators {operators:?} against {expected_operators:?}");
    }

    /// The whole share kept, filled in chunks of 2 sites; invalidated, the kept planes take no part until the next
    /// fill.
    #[test]
    fn the_kept_integrals_give_the_dense_contractions() {
        let sites: Vec<usize> = (0..9).collect();
        let mut kept = FieldChunker::with_kept_filling(N, sites.len(), 2..7, 2, usize::MAX, 2 * field_integral_bytes_per_site(N)).unwrap();
        assert!(kept.is_resident() && !kept.is_filled());
        assert_eq!(kept.kept(), 2..7);
        for round in 0..2 {
            kept.fill_resident(&sites, fake_chunk).unwrap();
            assert!(kept.is_filled());
            passes_agree(&mut kept, sites.len(), round);
            kept.invalidate();
        }
        assert!(!kept.is_filled());
        let mut fields = vec![0.0; 3 * sites.len()];
        kept.add_kept_fields(&vec![0.1; N * N], &mut fields, 3 * sites.len());
        assert!(fields.iter().all(|&f| f == 0.0));
    }

    /// What fits: 2 sites (with the scratch of 2 packed matrices) of a share of 7; the other 5 are the caller's.
    #[test]
    fn what_fits_is_kept() {
        let sites: Vec<usize> = (0..9).collect();
        let per_site = FieldChunker::bytes_per_site(N);
        let budget = 2 * per_site + FieldChunker::scratch_bytes(N, 2) + per_site / 2;
        assert_eq!(FieldChunker::kept_sites(N, 7, 2, budget), 2);
        let mut partial = FieldChunker::with_kept(N, sites.len(), 1..8, 2, budget).unwrap();
        assert_eq!(partial.kept(), 1..3);
        partial.fill_resident(&sites, fake_chunk).unwrap();
        passes_agree(&mut partial, sites.len(), 0);
    }

    /// A budget too small for one site: nothing kept, no memory held, and the passes add nothing.
    #[test]
    fn a_budget_below_one_site_keeps_nothing() {
        let mut chunker = FieldChunker::with_kept(N, 9, 0..9, 1, 100).unwrap();
        assert!(!chunker.is_resident());
        assert_eq!(chunker.bytes(), 0);
        let mut operators = Matrix::<f64>::new(N, N);
        chunker.add_kept_operators(&vec![1.0; 27], 27, 1.0, operators.get_data_mut());
        assert!(operators.get_data().iter().all(|&v| v == 0.0));
    }
}
