// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

/// A point in space carrying classical data.
pub trait SiteTrait {
    /// The position, in bohr.
    fn get_position(&self) -> &[f64; 3];

    /// Fragment numbers, as in the xfield file's leading integers: for a polarizable site the first
    /// is the fragment its polarizability belongs to; for a source they are the fragments whose
    /// polarizing field it must not contribute to. Empty for everything that has no fragments
    /// (nuclei, AIMPs -- an AIMP is its own fragment -- and bare sites). The number 0 is not a
    /// fragment but an unused slot (a polarizable AIMP centre, for instance, is written with
    /// fragment 0), and never excludes or is excluded by anything.
    fn get_fragments(&self) -> &[i32] {
        &[]
    }

    /// Position of the record this site was read from among the records of its input file
    /// (0-based), for mapping results back to the input. `None` for sites not read from a file.
    /// Dropped records (all-zero moments, zero polarizability) still count: the index is the record's
    /// position in the file, not in the retained list.
    fn get_input_index(&self) -> Option<usize> {
        None
    }
}

/// The stored input index of a site not read from a file.
pub const NO_INPUT_INDEX: usize = usize::MAX;

/// The stored input index as `get_input_index` returns it.
pub(crate) fn input_index_from_storage(stored: usize) -> Option<usize> {
    (stored != NO_INPUT_INDEX).then_some(stored)
}

/// Up to two fragment numbers (the most the xfield format allows), stored inline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fragments {
    ids: [i32; 2],
    len: u8,
}

impl Fragments {
    /// No fragment numbers.
    pub const NONE: Fragments = Fragments { ids: [0, 0], len: 0 };

    /// From a list of at most two fragment numbers; panics for more.
    pub fn new(ids: &[i32]) -> Self {
        assert!(ids.len() <= 2, "at most two fragment numbers are supported, got {}", ids.len());
        let mut stored = [0; 2];
        stored[..ids.len()].copy_from_slice(ids);
        Self { ids: stored, len: ids.len() as u8 }
    }

    /// The fragment numbers, in input order.
    pub fn as_slice(&self) -> &[i32] {
        &self.ids[..self.len as usize]
    }

}

/// Whether `source` is excluded from the polarizing field at `site` by fragments: the site's own
/// fragment (the first of its list) appears among the source's. With one fragment number per
/// source this is "same fragment"; with two, a junction source is excluded from the sites of
/// either neighbouring fragment. Fragment 0 means "no fragment": a site whose own fragment is 0
/// (e.g. a polarizable AIMP centre) is never excluded from anything by this rule, and an unused 0
/// slot in a source's list excludes nothing. A site or source without fragments is never excluded
/// here (a site's exclusion from its *own* multipole, without fragment numbers, is a matter of
/// position).
pub fn fragment_excludes(site: &impl SiteTrait, source: &impl SiteTrait) -> bool {
    match site.get_fragments().first() {
        Some(&own_fragment) if own_fragment != 0 => source.get_fragments().contains(&own_fragment),
        _ => false,
    }
}

/// A bare point in space at which a property (e.g. the electric field) is evaluated,
/// carrying no charge or other source data.
#[derive(Debug, Clone)]
pub struct Site {
    position: [f64; 3],
}

impl Site {
    /// A site at `position` (bohr).
    pub fn new(position: [f64; 3]) -> Self {
        Self { position }
    }
}

impl SiteTrait for Site {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }
}

/// The dipole-dipole polarizability of a site, in the global frame, in atomic units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarizability {
    Isotropic(f64),
    /// The symmetric tensor's unique components in the canonical Cartesian order: xx, xy, xz, yy, yz, zz.
    Anisotropic([f64; 6]),
}

impl Polarizability {
    /// Every component is exactly zero: not a polarizable site.
    pub fn is_zero(&self) -> bool {
        match self {
            Polarizability::Isotropic(alpha) => *alpha == 0.0,
            Polarizability::Anisotropic(components) => components.iter().all(|&c| c == 0.0),
        }
    }

    /// Every component is finite.
    pub fn is_finite(&self) -> bool {
        match self {
            Polarizability::Isotropic(alpha) => alpha.is_finite(),
            Polarizability::Anisotropic(components) => components.iter().all(|c| c.is_finite()),
        }
    }

    /// Whether the tensor is finite and positive semi-definite (isotropic: alpha >= 0): all principal minors
    /// of the symmetric 3x3 non-negative, to a tolerance of 1e-10 relative to the tensor's largest component.
    pub fn is_positive_semidefinite(&self) -> bool {
        if !self.is_finite() {
            return false;
        }
        let m = self.as_matrix();
        let scale = m.iter().flatten().fold(0.0_f64, |largest, x| largest.max(x.abs()));
        if scale == 0.0 {
            return true;
        }
        let tolerance = |order: i32| 1e-10 * scale.powi(order);
        let minor2 = |i: usize, j: usize| m[i][i] * m[j][j] - m[i][j] * m[j][i];
        let determinant = m[0][0] * minor2(1, 2) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        (0..3).all(|i| m[i][i] >= -tolerance(1))
            && [(0, 1), (0, 2), (1, 2)].iter().all(|&(i, j)| minor2(i, j) >= -tolerance(2))
            && determinant >= -tolerance(3)
    }

    /// The full symmetric 3x3 tensor.
    pub fn as_matrix(&self) -> [[f64; 3]; 3] {
        match *self {
            Polarizability::Isotropic(alpha) => [[alpha, 0.0, 0.0], [0.0, alpha, 0.0], [0.0, 0.0, alpha]],
            Polarizability::Anisotropic([xx, xy, xz, yy, yz, zz]) => [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
        }
    }
}

/// A site carrying a polarizability, for induced-dipole embedding. A list of these is independent of the
/// permanent multipoles and the AIMPs (a site may coincide with either) and is in input order, the site order
/// of the 3S-dimensional induced-dipole and field vectors.
#[derive(Debug, Clone)]
pub struct PolarizableSite {
    position: [f64; 3],
    polarizability: Polarizability,
    fragments: Fragments,
    input_index: usize,
}

impl PolarizableSite {
    /// A site at `position` (bohr); the first of `fragments` is the fragment its polarizability belongs to.
    pub fn new(position: [f64; 3], polarizability: Polarizability, fragments: Fragments) -> Self {
        Self { position, polarizability, fragments, input_index: NO_INPUT_INDEX }
    }

    /// The record of the input file this site was read from (see `SiteTrait::get_input_index`).
    pub fn with_input_index(mut self, index: usize) -> Self {
        self.input_index = index;
        self
    }

    /// The polarizability, in atomic units.
    pub fn get_polarizability(&self) -> &Polarizability {
        &self.polarizability
    }
}

impl SiteTrait for PolarizableSite {
    fn get_position(&self) -> &[f64; 3] {
        &self.position
    }

    fn get_fragments(&self) -> &[i32] {
        self.fragments.as_slice()
    }

    fn get_input_index(&self) -> Option<usize> {
        input_index_from_storage(self.input_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PointCharge, Polarizability};

    #[test]
    fn fragment_zero_means_no_fragment() {
        let site = |fragments: &[i32]| PolarizableSite::new([0.0; 3], Polarizability::Isotropic(1.0), Fragments::new(fragments));
        let source = |fragments: &[i32]| PointCharge::new([1.0; 3], 1.0).with_fragments(Fragments::new(fragments));
        // A site in fragment 0 (a polarizable AIMP centre) is excluded from nothing, even from sources that list a 0.
        assert!(!fragment_excludes(&site(&[0]), &source(&[0])));
        assert!(!fragment_excludes(&site(&[0]), &source(&[1, 0])));
        assert!(!fragment_excludes(&site(&[0]), &source(&[3])));
        // An unused 0 slot in a junction source's list excludes nothing beyond its real fragment.
        assert!(fragment_excludes(&site(&[1]), &source(&[1, 0])));
        assert!(!fragment_excludes(&site(&[2]), &source(&[1, 0])));
    }
}
