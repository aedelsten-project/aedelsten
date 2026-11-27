// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Reading MOLCAS xfield files (the XFIELD keyword) into the classical sources of the `sources` crate.

use std::path::Path;

use crate::basis_parser::{InputFileParseError, InputParser};
use sources::{ExternalMultipoles, Fragments, PointCharge, PointMultipole, PolarizableSite, Polarizability, multipole_component_count};

/// The permanent multipoles of an xfield file, in file order; records whose moments are all zero are left out but
/// keep their place in the input indices. The file is a MOLCAS input with an `XFIELD` keyword (the data inline, or a
/// file name after it or after `=`, with `$CurrDir`, `$Project`, `$WorkDir`), or the xfield data alone:
///
/// ```text
///      nXF [Angstrom|Bohr] [nOrd [p [nFrag [nRead]]]]
///      [nFrag+nRead integers] x y z [multipole entries] [polarizability entries]
/// ```
///
/// with nXF centre lines after the header, where
/// - `nOrd`: the highest multipole order, default 1: -1 none, else the moments of orders 0..=nOrd,
///   (nOrd+1)(nOrd+2)(nOrd+3)/6 entries, each order x descending, then y descending (x y z; xx xy xz yy yz zz; ...).
///   MOLCAS reads up to quadrupoles; higher orders extend its grammar.
/// - `p`: the polarizability, 0 none, 1 isotropic (1 entry), 2 anisotropic (6, xx xy xz yy yz zz).
/// - `nFrag`: the number of leading fragment integers (0-2); the first is the fragment of the site's polarizability.
/// - `nRead`: 0 or 1, whether an element number follows the fragment integers (read and not used).
///
/// The unit applies to the coordinates only; everything else is in atomic units. Extra trailing columns are ignored.
pub fn read_xfield(path: &str) -> Result<ExternalMultipoles, InputFileParseError> {
    read_xfield_with(path, parse_xfield_multipoles)
}

/// The polarizable sites of an xfield file (`p` = 1 or 2, see `read_xfield`), in file order; records with a zero
/// polarizability are left out but keep their place in the input indices, and a polarizability that is not positive
/// semi-definite is an error. Permanent multipole columns are skipped.
pub fn read_xfield_polarizable_sites(path: &str) -> Result<Box<[PolarizableSite]>, InputFileParseError> {
    read_xfield_with(path, parse_xfield_polarizable_sites)
}

// The data of the xfield file `path`, read by `parse` from where it is (see `read_xfield`).
fn read_xfield_with<R>(
    path: &str,
    parse: fn(&mut InputParser) -> Result<R, InputFileParseError>,
) -> Result<R, InputFileParseError> {
    let mut parser = InputParser::new(path)?;

    // An XFIELD keyword (in a &GATEWAY or &SEWARD block)?
    let xfield_line = parser.find_line("xfield");

    // None: the file is the xfield data, read from the top again.
    let Some(xfield_line) = xfield_line else {
        let mut parser = InputParser::new(path)?;
        return parse(&mut parser);
    };

    if let Some((_, name)) = xfield_line.split_once('=') {
        // `XFIELD = <file>`.
        let filename = resolve_molcas_variables(name.trim(), path)?;
        let mut file_parser = InputParser::new(&filename)?;
        parse(&mut file_parser)
    } else {
        // The next line is a file name, or the header of the data inline.
        let next = parser.expect_peek_line("expected a filename or 'nr_of_charges [distance_unit]' after 'xfield'")?;
        let is_inline_header = next
            .trim()
            .split_whitespace()
            .next()
            .is_some_and(|token| token.parse::<usize>().is_ok());

        if is_inline_header {
            parse(&mut parser)
        } else {
            parser.clear_peek();
            let filename = resolve_molcas_variables(next.trim(), path)?;
            let mut file_parser = InputParser::new(&filename)?;
            parse(&mut file_parser)
        }
    }
}

// `name` with `$CurrDir` (the working directory), `$Project` (the main input's file name without .input or .inp) and
// `$WorkDir` (the environment variable `WorkDir`) replaced.
fn resolve_molcas_variables(name: &str, main_input_path: &str) -> Result<String, InputFileParseError> {
    let mut resolved = name.to_string();

    if resolved.contains("$CurrDir") {
        let cwd = std::env::current_dir().map_err(|_| {
            InputFileParseError::InvalidFormat(
                "Could not resolve '$CurrDir': failed to determine the current working directory".to_string(),
            )
        })?;
        resolved = resolved.replace("$CurrDir", &cwd.to_string_lossy());
    }

    if resolved.contains("$Project") {
        let project = Path::new(main_input_path)
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.strip_suffix(".input").or_else(|| name.strip_suffix(".inp")).unwrap_or(name))
            .ok_or_else(|| {
                InputFileParseError::InvalidFormat(format!(
                    "Could not resolve '$Project' from input file path '{}'",
                    main_input_path
                ))
            })?;
        resolved = resolved.replace("$Project", project);
    }

    if resolved.contains("$WorkDir") {
        let work_dir = std::env::var("WorkDir").map_err(|_| {
            InputFileParseError::InvalidFormat(
                "Could not resolve '$WorkDir': environment variable 'WorkDir' is not set".to_string(),
            )
        })?;
        resolved = resolved.replace("$WorkDir", &work_dir);
    }

    Ok(resolved)
}

// The header of xfield data: the number of centres, the coordinates' factor to bohr, nOrd, p, nFrag, nRead.
struct XfieldHeader {
    nr_of_centres: usize,
    to_bohr: f64,
    n_ord: i32,
    n_pol: usize,
    n_frag: usize,
    n_read: usize,
}

impl XfieldHeader {
    // The multipole entries per line.
    fn n_multipole_entries(&self) -> usize {
        if self.n_ord < 0 { 0 } else { multipole_component_count(self.n_ord as usize) }
    }

    // The polarizability entries per line.
    fn n_polarizability_entries(&self) -> usize {
        match self.n_pol {
            1 => 1,
            2 => 6,
            _ => 0,
        }
    }
}

// One centre line, split into its blocks.
struct XfieldCentre {
    fragments: Vec<i32>,
    position: [f64; 3],
    multipoles: Vec<f64>,
    polarizability: Vec<f64>,
}

// The permanent multipoles of xfield data (`read_xfield`).
fn parse_xfield_multipoles(parser: &mut InputParser) -> Result<ExternalMultipoles, InputFileParseError> {
    let header = read_xfield_header(parser)?;

    if header.n_ord == -1 {
        return Err(InputFileParseError::UnsupportedOption(
            "xfield data with n_ord = -1 has no permanent multipoles; polarizable-only data is read with read_xfield_polarizable_sites"
                .to_string(),
        ));
    }

    // Records whose moments are all exactly zero are dropped; a source's input index is its record's position.
    let centres = read_xfield_centres(parser, &header)?;
    let retained = centres.iter().enumerate().filter(|(_, c)| c.multipoles.iter().any(|&m| m != 0.0));
    if header.n_ord == 0 {
        return Ok(ExternalMultipoles::Charges(
            retained
                .map(|(index, c)| {
                    PointCharge::new(c.position, c.multipoles[0]).with_fragments(Fragments::new(&c.fragments)).with_input_index(index)
                })
                .collect(),
        ));
    }
    let max_order = header.n_ord as usize;
    Ok(ExternalMultipoles::Multipoles(
        retained
            .map(|(index, c)| {
                PointMultipole::from_xfield_moments(c.position, max_order, &c.multipoles)
                    .with_fragments(Fragments::new(&c.fragments))
                    .with_input_index(index)
            })
            .collect(),
    ))
}

// The polarizable sites of xfield data (`read_xfield_polarizable_sites`).
fn parse_xfield_polarizable_sites(parser: &mut InputParser) -> Result<Box<[PolarizableSite]>, InputFileParseError> {
    let header = read_xfield_header(parser)?;

    if header.n_pol == 0 {
        return Err(InputFileParseError::UnsupportedOption(
            "xfield data has no polarizabilities (p = 0): expected p = 1 (isotropic) or 2 (anisotropic)".to_string(),
        ));
    }

    let centres = read_xfield_centres(parser, &header)?;
    let mut sites = Vec::new();
    for (index, c) in centres.into_iter().enumerate() {
        let polarizability = match c.polarizability.as_slice() {
            [alpha] => Polarizability::Isotropic(*alpha),
            &[xx, xy, xz, yy, yz, zz] => Polarizability::Anisotropic([xx, xy, xz, yy, yz, zz]),
            _ => unreachable!("read_xfield_centres returns exactly n_polarizability_entries values"),
        };
        if polarizability.is_zero() {
            continue;
        }
        if !polarizability.is_positive_semidefinite() {
            return Err(InputFileParseError::InvalidFormat(format!(
                "the polarizability of xfield record {} ({:?}) is not positive semi-definite",
                index + 1,
                polarizability
            )));
        }
        sites.push(PolarizableSite::new(c.position, polarizability, Fragments::new(&c.fragments)).with_input_index(index));
    }
    Ok(sites.into_boxed_slice())
}

// Reads and parses the header line.
fn read_xfield_header(parser: &mut InputParser) -> Result<XfieldHeader, InputFileParseError> {
    parser.skip_blank_lines();
    let header = parser.expect_next_line("expected 'nXF [distance_unit] [nOrd [p [nFrag [nRead]]]]' line in xfield data")?;
    parse_xfield_header(&header)
}

// Reads the header's number of centre lines.
fn read_xfield_centres(parser: &mut InputParser, header: &XfieldHeader) -> Result<Vec<XfieldCentre>, InputFileParseError> {
    (0..header.nr_of_centres).map(|_| read_xfield_centre(parser, header)).collect()
}

// The header of `header`; an error for a value out of range.
fn parse_xfield_header(header: &str) -> Result<XfieldHeader, InputFileParseError> {
    let mut tokens = header.trim().split_whitespace();

    let nr_of_centres = tokens
        .next()
        .ok_or_else(|| InputFileParseError::InvalidFormat("Expected 'nXF' in xfield header, found an empty line".to_string()))?
        .parse::<usize>()
        .map_err(|_| InputFileParseError::InvalidFormat(format!("Invalid number of centres in xfield header, found '{}'", header.trim())))?;

    let mut next = tokens.next();
    let to_bohr = match next {
        Some(unit) if unit.eq_ignore_ascii_case("BOHR") || unit.eq_ignore_ascii_case("AU") => {
            next = tokens.next();
            1.0
        }
        Some(unit) if unit.eq_ignore_ascii_case("ANGSTROM") => {
            next = tokens.next();
            constants::ANGSTROM2AU
        }
        // No unit: `next` is nOrd.
        _ => 1.0,
    };

    // nOrd, p, nFrag, nRead: all optional integers, in that order.
    let mut integers = next.into_iter().chain(tokens);
    let mut next_integer = |name: &str, default: i32| -> Result<i32, InputFileParseError> {
        match integers.next() {
            Some(token) => token.parse::<i32>().map_err(|_| {
                InputFileParseError::InvalidFormat(format!("Invalid {name} value in xfield header: expected an integer, found '{token}'"))
            }),
            None => Ok(default),
        }
    };
    let n_ord = next_integer("nOrd", 1)?;
    let n_pol = next_integer("p", 0)?;
    let n_frag = next_integer("nFrag", 0)?;
    let n_read = next_integer("nRead", 0)?;

    let in_range = |name: &str, value: i32, min: i32, max: i32| {
        if (min..=max).contains(&value) {
            Ok(())
        } else {
            Err(InputFileParseError::UnsupportedOption(format!(
                "Unsupported {name} value {value} in xfield data: expected {min} to {max}"
            )))
        }
    };
    if n_ord < -1 {
        return Err(InputFileParseError::UnsupportedOption(format!("Unsupported nOrd value {n_ord} in xfield data: expected -1 or more")));
    }
    in_range("p", n_pol, 0, 2)?;
    in_range("nFrag", n_frag, 0, 2)?;
    in_range("nRead", n_read, 0, 1)?;

    Ok(XfieldHeader { nr_of_centres, to_bohr, n_ord, n_pol: n_pol as usize, n_frag: n_frag as usize, n_read: n_read as usize })
}

// Reads one "[nFrag+nRead integers] x y z <multipoles> <polarizabilities>" line, the position in bohr. Further
// columns are ignored.
fn read_xfield_centre(parser: &mut InputParser, header: &XfieldHeader) -> Result<XfieldCentre, InputFileParseError> {
    parser.skip_blank_lines();
    let line = parser.expect_next_line("expected a centre line in xfield data")?;
    let parts: Vec<&str> = line.trim().split_whitespace().collect();

    let n_leading = header.n_frag + header.n_read;
    let n_multipoles = header.n_multipole_entries();
    let n_polarizabilities = header.n_polarizability_entries();
    let n_required = n_leading + 3 + n_multipoles + n_polarizabilities;
    if parts.len() < n_required {
        return Err(InputFileParseError::InvalidFormat(format!(
            "Invalid centre line in xfield data: expected {n_leading} leading integer(s), 'x y z', {n_multipoles} multipole and \
             {n_polarizabilities} polarizability value(s), found '{}'",
            line.trim()
        )));
    }

    let mut leading = Vec::with_capacity(n_leading);
    for part in &parts[..n_leading] {
        leading.push(part.parse::<i32>().map_err(|_| {
            InputFileParseError::InvalidFormat(format!("Invalid fragment/element integer in xfield centre line: found '{part}'"))
        })?);
    }
    // The element number (nRead) trails the fragment numbers and is not used.
    leading.truncate(header.n_frag);

    let mut values = Vec::with_capacity(n_required - n_leading);
    for part in &parts[n_leading..n_required] {
        let value = part.parse::<f64>().map_err(|_| {
            InputFileParseError::InvalidFormat(format!("Invalid numeric value in xfield centre line: found '{part}'"))
        })?;
        if !value.is_finite() {
            return Err(InputFileParseError::InvalidFormat(format!("Non-finite value in xfield centre line: found '{part}'")));
        }
        values.push(value);
    }

    let position = [values[0] * header.to_bohr, values[1] * header.to_bohr, values[2] * header.to_bohr];
    let polarizability = values.split_off(3 + n_multipoles);
    let multipoles = values.split_off(3);
    Ok(XfieldCentre { fragments: leading, position, multipoles, polarizability })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sources::{MultipoleSourceTrait, PointChargeTrait, SiteTrait};

    fn write_xfield(tag: &str, contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("aedelsten_xfield_test_{}_{}.xfield", std::process::id(), tag));
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn read_multipoles(tag: &str, contents: &str) -> Result<ExternalMultipoles, InputFileParseError> {
        let path = write_xfield(tag, contents);
        let result = read_xfield(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        result
    }

    fn read_polarizable(tag: &str, contents: &str) -> Result<Box<[PolarizableSite]>, InputFileParseError> {
        let path = write_xfield(tag, contents);
        let result = read_xfield_polarizable_sites(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        result
    }

    #[test]
    fn charge_dipole_and_isotropic_polarizability_share_a_file() {
        // nOrd = 1, p = 1: x y z q dx dy dz alpha
        let contents = "2 Bohr 1 1\n1.0 2.0 3.0 0.5 0.1 0.2 0.3 7.5\n-1.0 0.0 4.0 -0.5 0.0 0.0 0.4 9.0\n";

        let ExternalMultipoles::Multipoles(dipoles) = read_multipoles("mp_iso", contents).unwrap() else {
            panic!("expected multipoles");
        };
        assert_eq!(dipoles.len(), 2);
        assert_eq!(dipoles[0].get_position(), &[1.0, 2.0, 3.0]);

        let sites = read_polarizable("pol_iso", contents).unwrap();
        assert_eq!(sites.len(), 2);
        assert_eq!(sites[0].get_position(), &[1.0, 2.0, 3.0]);
        assert_eq!(sites[0].get_polarizability(), &Polarizability::Isotropic(7.5));
        assert_eq!(sites[1].get_polarizability(), &Polarizability::Isotropic(9.0));
    }

    #[test]
    fn polarizable_only_anisotropic_file_with_angstrom() {
        // nOrd = -1, p = 2, positions in Angstrom, tensor in the file's xx xy xz yy yz zz order
        // (a positive-definite one: xx=4, xy=1, xz=0.5, yy=3, yz=0.25, zz=2).
        let contents = "1 Angstrom -1 2\n1.0 0.0 0.0 4.0 1.0 0.5 3.0 0.25 2.0\n";
        let sites = read_polarizable("pol_aniso", contents).unwrap();
        assert_eq!(sites.len(), 1);
        assert!((sites[0].get_position()[0] - constants::ANGSTROM2AU).abs() < 1e-12);
        let Polarizability::Anisotropic(components) = *sites[0].get_polarizability() else {
            panic!("expected an anisotropic tensor");
        };
        // File order is the internal order: xx, xy, xz, yy, yz, zz.
        assert_eq!(components, [4.0, 1.0, 0.5, 3.0, 0.25, 2.0]);
        assert_eq!(sites[0].get_polarizability().as_matrix(), [[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]);

        // No permanent multipoles in such a file.
        assert!(read_multipoles("mp_none", contents).is_err());
    }

    #[test]
    fn arbitrary_multipole_orders_are_read_from_xfield() {
        // nOrd = 3: x y z + 20 entries.
        let values: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        let contents = format!("1 Bohr 3\n1.0 2.0 3.0 {}\n", values.join(" "));
        let ExternalMultipoles::Multipoles(multipoles) = read_multipoles("octupole", &contents).unwrap() else {
            panic!("expected multipoles");
        };
        assert_eq!(multipoles[0].get_max_order(), 3);
        assert_eq!(multipoles[0].get_moment(1, 1, 1), 14.0);
        assert_eq!(multipoles[0].get_position(), &[1.0, 2.0, 3.0]);

        // One entry short is an error, as is a nonsensical order.
        let short = format!("1 Bohr 3\n1.0 2.0 3.0 {}\n", values[..19].join(" "));
        assert!(read_multipoles("octupole_short", &short).is_err());
        assert!(read_multipoles("order_minus_two", "1 Bohr -2\n1.0 2.0 3.0\n").is_err());
        // nOrd = 0 keeps the dedicated charge path.
        assert!(matches!(read_multipoles("charges", "1 Bohr 0\n1.0 2.0 3.0 0.5\n").unwrap(), ExternalMultipoles::Charges(_)));
    }

    #[test]
    fn the_reader_stores_fragments_on_charges_and_multipoles() {
        // nFrag = 2: each line starts with two fragment integers.
        let ExternalMultipoles::Charges(charges) = read_multipoles("frag_charges", "2 Bohr 0 0 2\n1 2 0.0 0.0 0.0 0.5\n3 0 1.0 1.0 1.0 -0.5\n").unwrap() else {
            panic!("expected charges");
        };
        assert_eq!(charges[0].get_fragments(), &[1, 2]);
        assert_eq!(charges[1].get_fragments(), &[3, 0]);
        assert_eq!(charges[1].get_position(), &[1.0, 1.0, 1.0]);

        let ExternalMultipoles::Multipoles(dipoles) = read_multipoles("frag_dipoles", "1 Bohr 1 0 1\n9 0.0 0.0 0.0 1.0 0.1 0.2 0.3\n").unwrap() else {
            panic!("expected multipoles");
        };
        assert_eq!(dipoles[0].get_fragments(), &[9]);
        // No fragment columns: none stored.
        let ExternalMultipoles::Charges(plain) = read_multipoles("no_frag", "1 Bohr 0\n0.0 0.0 0.0 0.5\n").unwrap() else {
            panic!("expected charges");
        };
        assert!(plain[0].get_fragments().is_empty());
    }

    // ---- One file for both lists: records that carry nothing for a list are dropped from it ----

    // Record 0: an AIMP-like site (no multipoles, polarizable); 1: environment site with both;
    // 2: multipole-only (alpha = 0); 3: polarizable-only environment site; 4: an all-zero padding record.
    const ONE_FILE: &str = "5 Bohr 1 1 1\n\
        0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0\n\
        1 5.0 5.0 5.0 1.0 0.1 0.2 0.3 1.5\n\
        1 6.0 6.0 6.0 -0.5 0.0 0.0 0.0 0.0\n\
        2 7.0 7.0 7.0 0.0 0.0 0.0 0.0 3.0\n\
        2 8.0 8.0 8.0 0.0 0.0 0.0 0.0 0.0\n";

    #[test]
    fn records_without_moments_or_polarizability_are_dropped_but_keep_their_input_index() {
        let ExternalMultipoles::Multipoles(sources) = read_multipoles("one_file_mp", ONE_FILE).unwrap() else {
            panic!("expected multipoles");
        };
        assert_eq!(sources.len(), 2, "records 1 and 2 have moments");
        assert_eq!(sources.iter().map(|s| s.get_input_index()).collect::<Vec<_>>(), [Some(1), Some(2)]);
        assert_eq!(sources[0].get_position(), &[5.0, 5.0, 5.0]);
        assert!(sources.iter().all(|s| !s.is_zero()));

        let sites = read_polarizable("one_file_pol", ONE_FILE).unwrap();
        assert_eq!(sites.len(), 3, "records 0, 1 and 3 are polarizable");
        assert_eq!(sites.iter().map(|s| s.get_input_index()).collect::<Vec<_>>(), [Some(0), Some(1), Some(3)]);
        assert_eq!(sites[0].get_fragments(), &[0]);
        assert_eq!(sites[2].get_position(), &[7.0, 7.0, 7.0]);
    }

    #[test]
    fn charge_only_files_drop_zero_charges_too() {
        let ExternalMultipoles::Charges(charges) = read_multipoles("zero_charges", "3 Bohr 0\n0 0 0 0.0\n1 1 1 0.5\n2 2 2 0.0\n").unwrap() else {
            panic!("expected charges");
        };
        assert_eq!(charges.len(), 1);
        assert_eq!(charges[0].get_input_index(), Some(1));
    }

    #[test]
    fn a_partially_zero_tensor_is_kept_and_an_all_zero_one_dropped() {
        // Record 0: rank-one (semi-definite) tensor, kept; record 1: all zeros, dropped.
        let contents = "2 Bohr -1 2\n0 0 0 1.0 0.0 0.0 0.0 0.0 0.0\n1 1 1 0.0 0.0 0.0 0.0 0.0 0.0\n";
        let sites = read_polarizable("partial_zero", contents).unwrap();
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].get_input_index(), Some(0));
    }

    #[test]
    fn unphysical_polarizabilities_and_non_finite_values_are_errors() {
        // Negative isotropic polarizability.
        assert!(read_polarizable("neg_iso", "1 Bohr -1 1\n0 0 0 -1.0\n").is_err());
        // Symmetric but indefinite tensor (positive diagonal, off-diagonal too large: det < 0).
        assert!(read_polarizable("indefinite", "1 Bohr -1 2\n0 0 0 1.0 2.0 0.0 1.0 0.0 1.0\n").is_err());
        // Negative diagonal component.
        assert!(read_polarizable("neg_diag", "1 Bohr -1 2\n0 0 0 1.0 0.0 0.0 -0.1 0.0 1.0\n").is_err());
        // NaN / infinity anywhere in a record.
        assert!(read_polarizable("nan", "1 Bohr -1 1\n0 0 0 NaN\n").is_err());
        assert!(read_multipoles("inf", "1 Bohr 0\n0 0 inf 1.0\n").is_err());
        // The multipole reader does not validate polarizabilities it does not use, but the sign
        // check for them happens only where they are read: a bad tensor in a file read for its
        // multipoles is still parsed as a number and accepted here.
        assert!(read_multipoles("mp_ignores_pol", "1 Bohr 0 1\n0 0 0 1.0 -5.0\n").is_ok());
    }

    #[test]
    fn positive_semidefiniteness_is_checked_on_all_principal_minors() {
        let tensor = |c: [f64; 6]| Polarizability::Anisotropic(c); // internal order xx xy xz yy yz zz
        assert!(tensor([4.0, 1.0, 0.5, 3.0, 0.25, 2.0]).is_positive_semidefinite());
        assert!(tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0]).is_positive_semidefinite()); // singular but PSD
        assert!(!tensor([1.0, 2.0, 0.0, 1.0, 0.0, 1.0]).is_positive_semidefinite());
        // Leading minors alone would miss this one: leading minors 0, 0, 0 but the tensor is indefinite.
        assert!(!tensor([0.0, 0.0, 0.0, 1.0, 0.0, -1.0]).is_positive_semidefinite());
        assert!(Polarizability::Isotropic(0.0).is_positive_semidefinite());
        assert!(!Polarizability::Isotropic(-1e-3).is_positive_semidefinite());
    }

    #[test]
    fn header_defaults_follow_the_manual() {
        // Only the number of centres: nOrd defaults to 1, i.e. x y z q dx dy dz per line.
        let ExternalMultipoles::Multipoles(dipoles) = read_multipoles("default_ord", "1\n1.0 2.0 3.0 0.5 0.1 0.2 0.3\n").unwrap() else {
            panic!("expected dipoles from the default nOrd = 1");
        };
        assert_eq!(dipoles[0].get_max_order(), 1);
        assert_eq!(dipoles[0].get_position(), &[1.0, 2.0, 3.0]);
        // ... so a charges-only line is too short.
        assert!(read_multipoles("default_ord_short", "1\n1.0 2.0 3.0 0.5\n").is_err());
    }

    #[test]
    fn leading_fragment_and_element_integers_are_skipped() {
        // nOrd = 0, p = 1, nFrag = 1, nRead = 1: fragment element x y z q alpha
        let contents = "1 Bohr 0 1 1 1\n3 8 1.0 2.0 3.0 -0.8 1.4\n";
        let ExternalMultipoles::Charges(charges) = read_multipoles("mp_frag", contents).unwrap() else {
            panic!("expected charges");
        };
        assert_eq!(charges[0].get_position(), &[1.0, 2.0, 3.0]);
        assert_eq!(charges[0].get_charge(), -0.8);

        let sites = read_polarizable("pol_frag", contents).unwrap();
        assert_eq!(sites[0].get_fragments(), &[3]);
        assert_eq!(sites[0].get_position(), &[1.0, 2.0, 3.0]);
        assert_eq!(sites[0].get_polarizability(), &Polarizability::Isotropic(1.4));
    }

    #[test]
    fn two_fragment_numbers_are_kept_in_order() {
        let sites = read_polarizable("pol_two_frag", "1 Bohr -1 1 2\n4 5 0.0 0.0 0.0 2.0\n").unwrap();
        assert_eq!(sites[0].get_fragments(), &[4, 5]);
    }

    #[test]
    fn polarizable_reader_rejects_files_without_polarizabilities() {
        assert!(read_polarizable("pol_none", "1 Bohr 0\n0.0 0.0 0.0 1.0\n").is_err());
    }

    #[test]
    fn malformed_files_are_errors() {
        // Too few columns for p = 2.
        assert!(read_polarizable("short", "1 Bohr -1 2\n0.0 0.0 0.0 1.0 2.0\n").is_err());
        // Out-of-range header values.
        assert!(read_polarizable("bad_p", "1 Bohr -1 3\n0.0 0.0 0.0 1.0\n").is_err());
        assert!(read_multipoles("bad_ord", "1 Bohr 3\n0.0 0.0 0.0 1.0\n").is_err());
        // Missing centre lines.
        assert!(read_polarizable("missing", "2 Bohr -1 1\n0.0 0.0 0.0 1.0\n").is_err());
    }
}
