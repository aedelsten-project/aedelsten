// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The parsers of the input file's basis blocks and of the basis and AIMP library files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use arrays::matrix::*;
use crate::shell_types::HarmonicType;

// One library entry: its charge, its valence shells (none for an embedding AIMP) and its AIMP core, if any.
#[derive(Clone, Debug)]
pub(crate) struct ParsedBasisSet {
    charge: f64,
    valence: Option<Vec<InputBasisShell>>,
    core: Option<InputAimp>,
}

impl ParsedBasisSet {
    // The valence shells, `None` for an embedding AIMP.
    pub fn get_valence(&self) -> Option<&Vec<InputBasisShell>> {
        self.valence.as_ref()
    }

    // The AIMP core.
    pub fn get_core(&self) -> Option<&InputAimp> {
        self.core.as_ref()
    }

    // The entry's charge.
    pub fn get_charge(&self) -> f64 {
        self.charge
    }
}

/// One parsed shell: its exponents, its normalized `(n_prim, n_cont)` coefficients, and the orbital energies and
/// occupations its entry gives.
#[derive(Clone, Debug)]
pub struct InputBasisShell {
    lqn: usize,
    exponents: Box<[f64]>,
    coefficients: Matrix<f64>,
    energy: Option<Box<[f64]>>,
    // Electrons per contracted function: projection (PROJOP) shells only.
    occupation: Option<Box<[f64]>>,
}

impl InputBasisShell {
    /// The shell of the given data, as given: angular momentum, exponents, the `(n_prim, n_cont)` coefficients, and the
    /// orbital energies and occupations an AIMP resolution shell carries.
    pub fn new(
        lqn: usize,
        exponents: Box<[f64]>,
        coefficients: Matrix<f64>,
        energy: Option<Box<[f64]>>,
        occupation: Option<Box<[f64]>>,
    ) -> Self {
        Self { lqn, exponents, coefficients, energy, occupation }
    }

    /// The angular momentum.
    pub fn get_lqn(&self) -> usize {
        self.lqn
    }

    /// The exponents.
    pub fn get_exponents(&self) -> &[f64] {
        &self.exponents
    }

    /// The `(n_prim, n_cont)` coefficients.
    pub fn get_coefficients(&self) -> &Matrix<f64> {
        &self.coefficients
    }

    /// The orbital energies of the first columns, or a projection shell's projection constants.
    pub fn get_energies(&self) -> Option<&Box<[f64]>> {
        self.energy.as_ref()
    }

    /// A projection shell's electrons per contracted function.
    pub fn get_occupations(&self) -> Option<&Box<[f64]>> {
        self.occupation.as_ref()
    }

    // Scales each contracted function to unit norm.
    fn normalise_shell(&mut self) {
        let lqn = self.lqn as f64;
        let power = lqn + 1.5;
        for col in self.coefficients.iter_col_mut() {
            let mut norm_constant: f64 = 0.0;
            for (coeff1, exp1) in col.iter().zip(self.exponents.iter()) {
                for (coeff2, exp2) in col.iter().zip(self.exponents.iter()) {
                    let p = 2.0 * (exp1 * exp2).sqrt();
                    let q = exp1 + exp2;
                    norm_constant += coeff1 * coeff2 * (p / q).powf(power);
                }
            }
            norm_constant = 1.0 / norm_constant.sqrt();
            for coeff in col.iter_mut() {
                *coeff *= norm_constant;
            }
        }
    }
}

// An AIMP core: its screened Coulomb potential and its projection (PROJOP) shells.
#[derive(Clone, Debug)]
pub(crate) struct InputAimp {
    potential: crate::charge_carriers::ScreenedCoulombPotential,
    projection_shells: Vec<InputBasisShell>,
}

impl InputAimp {
    // The screened Coulomb potential (M1).
    pub fn get_potential(&self) -> &crate::charge_carriers::ScreenedCoulombPotential {
        &self.potential
    }

    // The projection shells.
    pub fn get_shells(&self) -> &[InputBasisShell] {
        &self.projection_shells
    }

    // The orthogonality functions: per projection shell, its frozen orbital of smallest projection constant.
    pub fn get_orthogonality_shells(&self) -> Vec<InputBasisShell> {
        self.projection_shells.iter().map(|shell| {
            let energies = shell.get_energies().expect("PROJOP shells always carry projection-constant energies");
            let outermost = energies.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("projection constants are never NaN"))
                .map(|(idx, _)| idx)
                .expect("PROJOP shell always has at least one contracted function");

            let source_column = shell.get_coefficients().iter_col().nth(outermost)
                .expect("outermost index is within bounds by construction");
            let mut coefficients = Matrix::<f64>::new(shell.get_coefficients().rows(), 1);
            coefficients.iter_col_mut().next().unwrap().copy_from_slice(source_column);

            InputBasisShell {
                lqn: shell.lqn,
                exponents: shell.exponents.clone(),
                coefficients,
                energy: None,
                occupation: None,
            }
        }).collect()
    }
}

// Per angular momentum, one shell of the projection and then the valence primitives, the projection orbitals'
// coefficients on the projection primitives and zero on the valence ones (`AimpTrait::get_resolution_basis`).
pub(crate) fn merge_resolution_shells(core_shells: &[InputBasisShell], valence_shells: &[InputBasisShell]) -> Vec<InputBasisShell> {
    let Some(max_lqn) = core_shells.iter().chain(valence_shells.iter()).map(|s| s.lqn).max() else {
        return Vec::new();
    };

    let mut merged = Vec::new();
    for lqn in 0..=max_lqn {
        let core_at_l: Vec<&InputBasisShell> = core_shells.iter().filter(|s| s.lqn == lqn).collect();
        let valence_at_l: Vec<&InputBasisShell> = valence_shells.iter().filter(|s| s.lqn == lqn).collect();
        if core_at_l.is_empty() && valence_at_l.is_empty() {
            continue;
        }

        let core_prim: usize = core_at_l.iter().map(|s| s.exponents.len()).sum();
        let valence_prim: usize = valence_at_l.iter().map(|s| s.exponents.len()).sum();
        let num_contr: usize = core_at_l.iter().map(|s| s.coefficients.cols()).sum();

        let mut exponents = Vec::with_capacity(core_prim + valence_prim);
        for s in &core_at_l {
            exponents.extend_from_slice(&s.exponents);
        }
        for s in &valence_at_l {
            exponents.extend_from_slice(&s.exponents);
        }

        // The valence rows stay zero.
        let mut coefficients = Matrix::<f64>::new(core_prim + valence_prim, num_contr);
        let mut row_offset = 0;
        let mut col_offset = 0;
        for s in &core_at_l {
            for (j, col) in s.coefficients.iter_col().enumerate() {
                for (i, value) in col.iter().enumerate() {
                    coefficients[(row_offset + i, col_offset + j)] = *value;
                }
            }
            row_offset += s.exponents.len();
            col_offset += s.coefficients.cols();
        }

        let energy: Box<[f64]> = core_at_l.iter()
            .flat_map(|s| s.energy.as_ref().expect("PROJOP shells always carry projection-constant energies").iter().copied())
            .collect();
        let occupation: Box<[f64]> = core_at_l.iter()
            .flat_map(|s| s.occupation.as_ref().expect("PROJOP shells always carry occupation numbers").iter().copied())
            .collect();

        merged.push(InputBasisShell {
            lqn,
            exponents: exponents.into_boxed_slice(),
            coefficients,
            energy: Some(energy),
            occupation: Some(occupation),
        });
    }
    merged
}

// The input file's basis blocks.
#[derive(Debug)]
pub(crate) struct ParsedInput {
    basis_blocks: Vec<InputBasisBlock>,
}

impl ParsedInput {
    // The blocks, in input order.
    pub fn get_basis_blocks(&self) -> &Vec<InputBasisBlock> {
        &self.basis_blocks
    }
}

// One `Basis set ... End` block of the input: the basis set spec, its options and its centres.
#[derive(Debug)]
pub(crate) struct InputBasisBlock {
    basis_set: String,
    charge: Option<f64>,
    pseudocharge: bool,
    harmonic_overrides: HarmonicOverrides,
    // The ORTHOGONALITY option: orthogonality functions from the block's embedding AIMP.
    orthogonality: bool,
    centres: Vec<InputCoordinates>,
}

impl InputBasisBlock {
    // The spec's element, before the first dot ("H" of "H.ANO-RCC").
    pub fn get_element(&self) -> &str {
        self.basis_set.split('.').next().unwrap_or("")
    }

    // The spec's library file, after the first dot ("ANO-RCC" of "H.ANO-RCC").
    pub fn get_basis_file(&self) -> &str {
        self.basis_set.split('.').nth(1).unwrap_or("")
    }

    // The whole spec as written.
    pub fn get_basis_set_spec(&self) -> &str {
        &self.basis_set
    }

    // The centres.
    pub fn get_centres(&self) -> &Vec<InputCoordinates> {
        &self.centres
    }

    // GHOST or PSEUDOCHARGE given (or a zero charge).
    pub fn is_pseudocharge(&self) -> bool {
        self.pseudocharge
    }

    // ORTHOGONALITY given.
    pub fn is_orthogonality(&self) -> bool {
        self.orthogonality
    }

    // The harmonic type of a shell of angular momentum `lqn` (`HarmonicOverrides::resolve`).
    pub fn resolve_harmonic_type(&self, lqn: usize) -> HarmonicType {
        self.harmonic_overrides.resolve(lqn)
    }

    // The charge given in the input.
    pub fn get_charge(&self) -> Option<f64> {
        self.charge
    }
}

// A block's Spherical/Cartesian cards: one for every shell ("all"), and per angular momentum.
#[derive(Clone, Debug, Default)]
struct HarmonicOverrides {
    all: Option<HarmonicType>,
    per_lqn: HashMap<usize, HarmonicType>,
}

impl HarmonicOverrides {
    // Records one card.
    fn set(&mut self, target: HarmonicTarget, harmonic: HarmonicType) {
        match target {
            HarmonicTarget::All => self.all = Some(harmonic),
            HarmonicTarget::Shells(lqns) => {
                for lqn in lqns {
                    self.per_lqn.insert(lqn, harmonic);
                }
            }
        }
    }

    // Cartesian for s and p whatever the cards; otherwise the card for `lqn`, then the "all" card, then the default.
    fn resolve(&self, lqn: usize) -> HarmonicType {
        if lqn <= 1 {
            return HarmonicType::Cartesian;
        }
        self.per_lqn.get(&lqn).copied()
            .or(self.all)
            .unwrap_or_else(|| HarmonicType::default_for_lqn(lqn))
    }
}

// What a single Spherical/Cartesian card line applies to.
enum HarmonicTarget {
    All,
    Shells(Vec<usize>),
}

// The shell letters in angular-momentum order, `j` skipped.
const SHELL_CHARACTERS: &[u8] = b"spdfghiklmnoqrtuvwxyz";

// The angular momentum of a shell designator: a letter (s, p, d, ...) or an integer.
fn parse_shell_designator(token: &str, path: &str) -> Result<usize, InputFileParseError> {
    if let Ok(lqn) = token.parse::<usize>() {
        return Ok(lqn);
    }
    if token.len() == 1 {
        let ch = token.as_bytes()[0].to_ascii_lowercase();
        if let Some(lqn) = SHELL_CHARACTERS.iter().position(|&c| c == ch) {
            return Ok(lqn);
        }
    }
    Err(InputFileParseError::InvalidFormat(format!(
        "Invalid shell designator '{}' in input file '{}': expected a shell character (s, p, d, f, ...) or an angular-momentum integer",
        token, path
    )))
}

// The column count per angular momentum of a shell-count string ("4s3p2d"); `None` if `s` is not one.
fn parse_shell_count_string(s: &str) -> Option<HashMap<usize, usize>> {
    let mut map = HashMap::new();
    let mut digits = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            continue;
        }
        if digits.is_empty() || !ch.is_ascii_alphabetic() {
            return None;
        }
        let count: usize = digits.parse().ok()?;
        digits.clear();
        let lqn = SHELL_CHARACTERS.iter().position(|&c| c == ch.to_ascii_lowercase() as u8)?;
        if map.insert(lqn, count).is_some() {
            return None;
        }
    }
    if !digits.is_empty() || map.is_empty() {
        return None;
    }
    Some(map)
}

// The target of a Spherical/Cartesian card: "all" or a list of shell designators, on the keyword's line or the next.
fn parse_harmonic_target(
    parser: &mut InputParser,
    same_line_tokens: std::str::SplitWhitespace,
    keyword: &str,
    path: &str,
) -> Result<HarmonicTarget, InputFileParseError> {
    let same_line: Vec<String> = same_line_tokens.map(str::to_string).collect();
    let tokens = if !same_line.is_empty() {
        same_line
    } else {
        let next = parser.expect_next_line(&format!("expected a shell list after '{}'", keyword))?;
        next.trim().split_whitespace().map(str::to_string).collect()
    };
    if tokens.is_empty() {
        return Err(InputFileParseError::InvalidFormat(format!(
            "Expected 'all' or a shell list after '{}' in input file '{}'", keyword, path
        )));
    }
    if tokens.len() == 1 && tokens[0].eq_ignore_ascii_case("all") {
        return Ok(HarmonicTarget::All);
    }
    let lqns = tokens.iter()
        .map(|token| parse_shell_designator(token, path))
        .collect::<Result<Vec<usize>, _>>()?;
    Ok(HarmonicTarget::Shells(lqns))
}

// A centre's position (bohr).
#[derive(Debug)]
pub(crate) struct InputCoordinates {
    x: f64,
    y: f64,
    z: f64,
}

impl InputCoordinates {
    // The position.
    pub fn get_position(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

// The line reader of input and library files: a '*' starts a comment to the end of the line; a line can be peeked
// without being consumed.
pub(crate) struct InputParser {
    reader: BufReader<File>,
    peeked: Option<String>,
}

const COMMENT_MARKER: &str = "*";

/// An error reading an input or library file.
#[derive(Debug)]
pub enum InputFileParseError {
    FileNotFound(String),
    InvalidFormat(String),
    UnsupportedOption(String),
}

impl std::fmt::Display for InputFileParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputFileParseError::FileNotFound(path) => write!(f, "file not found: '{}'", path),
            InputFileParseError::InvalidFormat(msg) => write!(f, "invalid format: {}", msg),
            InputFileParseError::UnsupportedOption(msg) => write!(f, "unsupported option: {}", msg),
        }
    }
}

impl std::error::Error for InputFileParseError {}

impl InputParser {
    // A reader of the file `path`.
    pub(crate) fn new(path: &str) -> Result<InputParser, InputFileParseError> {
        let file = std::fs::File::open(path)
            .map_err(|_| InputFileParseError::FileNotFound(path.to_string()))?;
        let reader = BufReader::new(file);
        Ok(InputParser { reader, peeked: None })
    }

    // The next line, without its comment (a comment-only line is empty); `None` at the end.
    fn next_line(&mut self) -> Option<String> {
        if let Some(line) = self.peeked.take() {
            return Some(line);
        }
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => {
                if let Some(idx) = line.find(COMMENT_MARKER) {
                    line.truncate(idx);
                }
                Some(line)
            }
            Err(_) => None,
        }
    }

    // The next line, not consumed.
    fn peek_line(&mut self) -> Option<&str> {
        if self.peeked.is_none() {
            self.peeked = self.next_line();
        }
        self.peeked.as_deref()
    }

    // Consumes the peeked line.
    pub(crate) fn clear_peek(&mut self) {
        self.peeked = None;
    }

    // Whether the peeked line, trimmed, satisfies `predicate`.
    fn current_line_matches(&mut self, predicate: impl Fn(&str) -> bool) -> bool {
        self.peek_line()
            .map(|line| predicate(line.trim()))
            .unwrap_or(false)
    }

    // `next_line`, the end of the file an error naming `context`.
    pub(crate) fn expect_next_line(&mut self, context: &str) -> Result<String, InputFileParseError> {
        self.next_line()
            .ok_or_else(|| InputFileParseError::InvalidFormat(format!("Unexpected end of file: {}", context)))
    }

    // `peek_line`, the end of the file an error naming `context`.
    pub(crate) fn expect_peek_line(&mut self, context: &str) -> Result<String, InputFileParseError> {
        self.peek_line()
            .map(str::to_string)
            .ok_or_else(|| InputFileParseError::InvalidFormat(format!("Unexpected end of file: {}", context)))
    }

    // The next line containing `target` (case-insensitively), trimmed; the lines before it are consumed.
    pub(crate) fn find_line(&mut self, target: &str) -> Option<String> {
        let uppercase_target = target.to_uppercase();
        loop {
            let line = self.next_line()?;
            if line.to_uppercase().contains(&uppercase_target) {
                return Some(line.trim().to_string());
            }
        }
    }

    // Consumes up to `n` lines; returns how many there were.
    fn skip_lines(&mut self, n: usize) -> usize {
        let mut advanced = 0;
        for _ in 0..n {
            if self.next_line().is_none() {
                break;
            }
            advanced += 1;
        }
        advanced
    }

    // Consumes blank and comment-only lines.
    pub(crate) fn skip_blank_lines(&mut self) {
        while self.current_line_matches(|line| line.is_empty()) {
            self.clear_peek();
        }
    }

    // `n` numbers, over as many lines as they take.
    fn read_n_f64(&mut self, n: usize) -> Result<Vec<f64>, InputFileParseError> {
        let mut floats = Vec::with_capacity(n);
        while floats.len() < n {
            let line = self.expect_next_line("reading floats")?;
            for part in line.trim().split_whitespace() {
                if floats.len() >= n {
                    break;
                }
                let value = part.parse::<f64>().map_err(|_| {
                    InputFileParseError::InvalidFormat(format!("Invalid float value in basis file: '{}'", part))
                })?;
                floats.push(value);
            }
        }
        Ok(floats)
    }

    // `num_prim` lines of `num_contr` numbers.
    fn read_coefficient_matrix(&mut self, num_prim: usize, num_contr: usize) -> Result<Matrix<f64>, InputFileParseError> {
        let mut matrix = Matrix::<f64>::new(num_prim, num_contr);
        for i in 0..num_prim {
            let line = self.expect_next_line("reading coefficient matrix")?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() != num_contr {
                return Err(InputFileParseError::InvalidFormat(format!(
                    "Expected {} coefficients per line in coefficient matrix, found {}",
                    num_contr,
                    parts.len()
                )));
            }
            for (j, part) in parts.iter().enumerate() {
                let value = part.parse::<f64>().map_err(|_| {
                    InputFileParseError::InvalidFormat(format!("Invalid float value in coefficient matrix: '{}'", part))
                })?;
                matrix[(i, j)] = value;
            }
        }
        Ok(matrix)
    }
}

// The dot-separated components of a library header ("/element.file.author.primitive_set.contracted_set.other.") or a
// spec of the same shape, without the leading '/' and one trailing '.'.
fn split_spec_components(spec: &str) -> Vec<String> {
    spec.trim().trim_start_matches('/').trim_end_matches('.').split('.').map(str::to_string).collect()
}

// Whether a spec component matches a header's: absent or empty matches anything, otherwise case-insensitive equality.
fn component_matches(query: Option<&str>, header: &str) -> bool {
    match query {
        None => true,
        Some(q) if q.is_empty() => true,
        Some(q) => q.eq_ignore_ascii_case(header),
    }
}

// The one library header of the file `path` that `basis_set_spec` matches, as a reader positioned after it, and the
// column counts to keep if the spec's contracted set is a smaller shell-count string. Element and file must match;
// author, primitive set and the tail match exactly or are skipped when empty; the contracted set matches exactly or as
// a shell-count string within the header's. No match, or more than one, is an error.
fn resolve_basis_header(basis_set_spec: &str, path: &str) -> Result<(InputParser, Option<HashMap<usize, usize>>), InputFileParseError> {
    let query = split_spec_components(basis_set_spec);
    let element = query.first().map(String::as_str).unwrap_or("");
    let file = query.get(1).map(String::as_str).unwrap_or("");
    let query_at = |i: usize| query.get(i).map(String::as_str);
    let query_other = if query.len() > 5 { Some(query[5..].join(".")) } else { None };

    let mut scan = InputParser::new(path)?;
    let mut candidates: Vec<(String, Vec<String>)> = Vec::new();
    while let Some(line) = scan.next_line() {
        let trimmed = line.trim();
        if !trimmed.starts_with('/') {
            continue;
        }
        let components = split_spec_components(trimmed);
        if components.len() < 2
            || !components[0].eq_ignore_ascii_case(element)
            || !components[1].eq_ignore_ascii_case(file)
        {
            continue;
        }
        let header_at = |i: usize| components.get(i).map(String::as_str).unwrap_or("");
        if !component_matches(query_at(2), header_at(2)) || !component_matches(query_at(3), header_at(3)) {
            continue;
        }
        let header_other = if components.len() > 5 { components[5..].join(".") } else { String::new() };
        if !component_matches(query_other.as_deref(), &header_other) {
            continue;
        }
        let contracted_ok = match query_at(4) {
            None => true,
            Some(q) if q.is_empty() => true,
            Some(q) if q.eq_ignore_ascii_case(header_at(4)) => true,
            Some(q) => match (parse_shell_count_string(q), parse_shell_count_string(header_at(4))) {
                (Some(requested), Some(available)) => requested.iter()
                    .all(|(lqn, &count)| available.get(lqn).is_some_and(|&avail| count <= avail)),
                _ => false,
            },
        };
        if !contracted_ok {
            continue;
        }
        candidates.push((trimmed.to_string(), components));
    }

    let (winning_line, winning_components) = match candidates.len() {
        0 => return Err(InputFileParseError::InvalidFormat(format!(
            "No basis set entry in file '{}' matches spec '{}' -- check the element, author, primitive/contracted set and any trailing qualifier, and that any requested contraction ('4s3p2d'-style) doesn't ask for more functions than the entry provides",
            path, basis_set_spec
        ))),
        1 => candidates.into_iter().next().expect("length checked above"),
        n => {
            let listing: Vec<String> = candidates.into_iter().map(|(line, _)| line).collect();
            return Err(InputFileParseError::InvalidFormat(format!(
                "Basis set spec '{}' is ambiguous in file '{}' -- it matches {} entries:\n{}\nGive a fuller qualifier (author, primitive/contracted set, or trailing tag) to pick one",
                basis_set_spec, path, n, listing.join("\n")
            )));
        }
    };

    // A truncation only if the contracted set is a smaller shell-count string.
    let winning_contracted = winning_components.get(4).map(String::as_str).unwrap_or("");
    let truncation = match query_at(4) {
        None => None,
        Some(q) if q.is_empty() => None,
        Some(q) if q.eq_ignore_ascii_case(winning_contracted) => None,
        Some(q) => Some(parse_shell_count_string(q).expect("validated as a shell-count string during candidate filtering")),
    };

    let mut parser = InputParser::new(path)?;
    parser.find_line(&winning_line).ok_or_else(|| {
        InputFileParseError::InvalidFormat(format!(
            "Internal error: previously found header line '{}' vanished on re-read of '{}'", winning_line, path
        ))
    })?;

    Ok((parser, truncation))
}

// The shells cut to the first `truncation[l]` columns (energies and occupations too), those of an `l` not in
// `truncation` dropped; an error if a shell has fewer columns than asked for.
fn apply_contraction_truncation(
    shells: Vec<InputBasisShell>,
    truncation: &HashMap<usize, usize>,
    path: &str,
) -> Result<Vec<InputBasisShell>, InputFileParseError> {
    let mut result = Vec::with_capacity(truncation.len());
    for mut shell in shells {
        let Some(&keep) = truncation.get(&shell.lqn) else {
            continue;
        };
        let available = shell.coefficients.cols();
        if keep > available {
            return Err(InputFileParseError::InvalidFormat(format!(
                "Requested {} contracted functions for lqn {} in basis file '{}', but only {} are available",
                keep, shell.lqn, path, available
            )));
        }
        let mut truncated = Matrix::<f64>::new(shell.coefficients.rows(), keep);
        for (col_out, col_in) in truncated.iter_col_mut().zip(shell.coefficients.iter_col()) {
            col_out.copy_from_slice(col_in);
        }
        shell.coefficients = truncated;
        shell.energy = shell.energy.map(|e| {
            let n = keep.min(e.len());
            e[..n].to_vec().into_boxed_slice()
        });
        shell.occupation = shell.occupation.map(|o| {
            let n = keep.min(o.len());
            o[..n].to_vec().into_boxed_slice()
        });
        result.push(shell);
    }
    Ok(result)
}

/// The library entry of `basis_set_spec` in the file `path`, its reference cited. An entry with orthogonality
/// functions written in as basis functions is an error.
pub(crate) fn parse_basis_file(basis_set_spec: &str, path: &str) -> Result<ParsedBasisSet, InputFileParseError> {
    let parsed = parse_basis_file_unchecked(basis_set_spec, path)?;
    reject_orthogonality_functions(&parsed, basis_set_spec, path)?;
    Ok(parsed)
}

// An error if an AIMP entry's basis functions are its projection operator's orbitals (orthogonality functions written
// into the entry; the ORTHOGONALITY card derives them).
fn reject_orthogonality_functions(parsed: &ParsedBasisSet, basis_set_spec: &str, path: &str) -> Result<(), InputFileParseError> {
    let (Some(valence), Some(core)) = (&parsed.valence, &parsed.core) else { return Ok(()) };
    for shell in valence {
        for projection in core.projection_shells.iter().filter(|p| p.lqn == shell.lqn) {
            for (j, function) in shell.coefficients.iter_col().enumerate() {
                let outside = 1.0 - fraction_in_span(shell.lqn, &shell.exponents, function, &projection.exponents, &projection.coefficients);
                if outside < ORTHOGONALITY_FUNCTION_TOLERANCE {
                    return Err(InputFileParseError::InvalidFormat(format!(
                        "Basis set '{}' in basis file '{}': basis function {} of l = {} lies in the span of the AIMP \
                         projection operator's orbitals (all but {:.1e} of its norm), i.e. orthogonality functions \
                         written into the AIMP entry. Use the entry without basis functions and the ORTHOGONALITY card \
                         instead.",
                        basis_set_spec, path, j + 1, shell.lqn, outside.max(0.0)
                    )));
                }
            }
        }
    }
    Ok(())
}

// A basis function with less than this fraction of its norm outside the projection operator's orbitals is one of
// them.
const ORTHOGONALITY_FUNCTION_TOLERANCE: f64 = 1e-6;

// The citation topic of a library entry's own reference (basis sets and model potentials alike).
const LIBRARY_DATA_TOPIC: &str = "Basis sets and model potentials";

// The library entry of `basis_set_spec`: header, reference, options, charge, shells, and the AIMP core (M1, PROJOP).
fn parse_basis_file_unchecked(basis_set_spec: &str, path: &str) -> Result<ParsedBasisSet, InputFileParseError> {
    let element = basis_set_spec.split('.').next().unwrap_or("");
    let (mut parser, truncation) = resolve_basis_header(basis_set_spec, path)?;

    // The two lines after the label: the entry's reference, cited for the run, and its description. Then any blank
    // lines.
    if let Some(reference) = parser.next_line() {
        instrumentation::cite_data(LIBRARY_DATA_TOPIC, &reference);
    }
    parser.skip_lines(1);
    parser.skip_blank_lines();

    let line = parser.expect_peek_line("expected an Options section or a charge/max_lqn line")?;
    let included_energies = if line.to_uppercase().contains("OPTIONS") {
        // OrbitalEnergies is the one option.
        parser.clear_peek();
        let orbital_energies = parser.expect_next_line("expected an OrbitalEnergies option")?;
        if !orbital_energies.to_uppercase().contains("ORBITALENERGIES") {
            return Err(InputFileParseError::UnsupportedOption(format!(
                "Unsupported options found in basis file '{}': expected 'OrbitalEnergies', found '{}'",
                path,
                orbital_energies.trim()
            )));
        }
        let end_options_line = parser.expect_next_line("expected 'EndOptions'")?;
        if !end_options_line.to_uppercase().contains("ENDOPTIONS") {
            return Err(InputFileParseError::InvalidFormat(format!(
                "Expected 'EndOptions' after options section in basis file '{}', found '{}'",
                path,
                end_options_line.trim()
            )));
        }
        true
    } else {
        false
    };

    parser.skip_blank_lines();
    let line = parser.expect_peek_line("expected a 'charge max_lqn' line")?;
    let parts: Vec<&str> = line.trim().split_whitespace().collect();
    if parts.len() != 2 {
        return Err(InputFileParseError::InvalidFormat(format!(
            "Expected 'charge max_lqn' line in basis file '{}', found '{}'",
            path,
            line.trim()
        )));
    }
    let charge = parts[0].parse::<f64>().map_err(|_| {
        InputFileParseError::InvalidFormat(format!(
            "Invalid charge value in basis file '{}': expected a number, found '{}'",
            path, parts[0]
        ))
    })?;
    let max_lqn = parts[1].parse::<usize>().map_err(|_| {
        InputFileParseError::InvalidFormat(format!(
            "Invalid max_lqn value in basis file '{}': expected a non-negative integer, found '{}'",
            path, parts[1]
        ))
    })?;
    parser.clear_peek();

    let mut shells = Vec::with_capacity(max_lqn + 1);
    for lqn in 0..=max_lqn {
        parser.skip_blank_lines();

        // "num_prim num_contr".
        let line = parser.expect_peek_line(&format!("expected a 'num_prim num_contr' line for lqn {}", lqn))?;
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.len() != 2 {
            return Err(InputFileParseError::InvalidFormat(format!(
                "Expected 'num_prim num_contr' line for lqn {} in basis file '{}', found '{}'",
                lqn,
                path,
                line.trim()
            )));
        }
        let num_prim = parts[0].parse::<usize>().map_err(|_| {
            InputFileParseError::InvalidFormat(format!(
                "Invalid num_prim value for lqn {} in basis file '{}': expected an integer, found '{}'",
                lqn, path, parts[0]
            ))
        })?;

        let num_contr = parts[1].parse::<usize>().map_err(|_| {
            InputFileParseError::InvalidFormat(format!(
                "Invalid num_contr value for lqn {} in basis file '{}': expected an integer, found '{}'",
                lqn, path, parts[1]
            ))
        })?;

        if num_contr > num_prim {
            return Err(InputFileParseError::InvalidFormat(format!(
                "Invalid basis set for lqn {} in basis file '{}': num_contr ({}) cannot be greater than num_prim ({})",
                lqn, path, num_contr, num_prim
            )));
        }

        parser.clear_peek();

        // num_prim exponents, over any number of lines; num_prim lines of num_contr coefficients.
        let exponents = parser.read_n_f64(num_prim)?;
        let coefficients = parser.read_coefficient_matrix(num_prim, num_contr)?;
        // With OrbitalEnergies: a count (at most num_contr), then the energies.
        let energy = if included_energies {
            let energy_line = parser.expect_peek_line(&format!("expected an orbital energies count for lqn {}", lqn))?;
            let parts: Vec<&str> = energy_line.trim().split_whitespace().collect();
            if parts.len() != 1 {
                return Err(InputFileParseError::InvalidFormat(format!(
                    "Expected orbital energies line for lqn {} in basis file '{}', found '{}'",
                    lqn,
                    path,
                    energy_line.trim()
                )));
            }
            let num_energies = parts[0].parse::<usize>().map_err(|_| {
                InputFileParseError::InvalidFormat(format!(
                    "Invalid number of orbital energies for lqn {} in basis file '{}': expected an integer, found '{}'",
                    lqn, path, parts[0]
                ))
            })?;
            if num_energies > num_contr {
                return Err(InputFileParseError::InvalidFormat(format!(
                    "Invalid number of orbital energies {} for lqn {} in basis file '{}': cannot be greater than num_contr ({})",
                    num_energies, lqn, path, num_contr
                )));
            }
            parser.clear_peek();
            if num_energies > 0 {
                Some(parser.read_n_f64(num_energies)?.into_boxed_slice())
            } else {
                None
            }
        } else {
            None
        };

        let mut new_shell = InputBasisShell {
            lqn,
            exponents: exponents.into_boxed_slice(),
            coefficients,
            energy,
            occupation: None,
        };

        if num_prim > 0 {
            new_shell.normalise_shell()
        }

        shells.push(new_shell);

    }

    let shells = match &truncation {
        Some(map) => apply_contraction_truncation(shells, map, path)?,
        None => shells,
    };

    // One shell without primitives: no valence (a core or embedding entry).
    let valence = if shells.len() == 1 {
        if shells[0].exponents.is_empty() {
            None
        } else {
            Some(shells)
        }
    } else {
        Some(shells)
    };

    // Then the end of the file or the next entry (no core), or M1 (an AIMP core).
    parser.skip_blank_lines();
    let peeked = parser.peek_line().map(str::to_string);
    let core = match peeked {
        None => None,
        Some(ref line) if line.trim().starts_with('/') => None,
        Some(line) => {
            if !line.to_uppercase().contains("M1") {
                return Err(InputFileParseError::UnsupportedOption(format!(
                    "Unsupported core section found in basis file '{}': expected 'M1' for AIMP, found '{}'",
                    path,
                    line.trim()
                )));
            }
            // M1: the number of terms, the exponents, the coefficients.
            parser.clear_peek();
            let n_potential = parser.expect_peek_line("expected number of terms in M1 potential")?;
            let n_potential = n_potential.trim().parse::<usize>().map_err(|_| {
                InputFileParseError::InvalidFormat(format!(
                    "Invalid number of terms in M1 potential in basis file '{}': expected an integer, found '{}'",
                    path, n_potential.trim()
                ))
            })?;
            parser.clear_peek();
            let exponents = parser.read_n_f64(n_potential)?;
            let coefficients = parser.read_n_f64(n_potential)?;
            let potential = crate::charge_carriers::ScreenedCoulombPotential::new(exponents, coefficients);

            // M2 and COREREP are not used.
            parser.find_line("PROJOP").ok_or_else(|| {
                InputFileParseError::InvalidFormat(format!(
                    "Expected 'PROJOP' section for AIMP core in basis file '{}', but it was not found",
                    path
                ))
            })?;

            let max_proj_lqn = parser.expect_peek_line("expected max lqn for AIMP projection")?;
            let max_proj_lqn = max_proj_lqn.trim().parse::<usize>().map_err(|_| {
                InputFileParseError::InvalidFormat(format!(
                    "Invalid max lqn for AIMP projection in basis file '{}': expected an integer, found '{}'",
                    path, max_proj_lqn.trim()
                ))
            })?;
            parser.clear_peek();

            let mut projection_shells = Vec::with_capacity(max_proj_lqn + 1);
            for lqn in 0..=max_proj_lqn {
                parser.skip_blank_lines();
                // "num_prim num_contr [num_contr occupations]".
                let line = parser.expect_peek_line(&format!("expected a 'num_prim num_contr' line for AIMP projection lqn {}", lqn))?;
                let parts: Vec<&str> = line.trim().split_whitespace().collect();
                let num_prim = parts.first().and_then(|s| s.parse::<usize>().ok()).ok_or_else(|| {
                    InputFileParseError::InvalidFormat(format!(
                        "Invalid num_prim value for AIMP projection lqn {} in basis file '{}', found '{}'",
                        lqn, path, line.trim()
                    ))
                })?;
                let num_contr = parts.get(1).and_then(|s| s.parse::<usize>().ok()).ok_or_else(|| {
                    InputFileParseError::InvalidFormat(format!(
                        "Invalid num_contr value for AIMP projection lqn {} in basis file '{}', found '{}'",
                        lqn, path, line.trim()
                    ))
                })?;
                if num_contr > num_prim {
                    return Err(InputFileParseError::InvalidFormat(format!(
                        "Invalid AIMP projection shell for lqn {} in basis file '{}': num_contr ({}) cannot be greater than num_prim ({})",
                        lqn, path, num_contr, num_prim
                    )));
                }
                // A projection shell projects out at least one frozen orbital.
                if num_contr == 0 {
                    return Err(InputFileParseError::InvalidFormat(format!(
                        "Empty AIMP projection shell for lqn {} of '{}' in basis file '{}': PROJOP gives every l up to its \
                         maximum at least one function (num_prim {}, num_contr 0); lower the maximum l if no orbital of this l \
                         is frozen",
                        lqn, basis_set_spec, path, num_prim
                    )));
                }
                let occupation = match parts.len() {
                    2 => {
                        // Full: 2(2l+1) electrons each.
                        vec![2.0 * (2 * lqn + 1) as f64; num_contr].into_boxed_slice()
                    }
                    n if n == 2 + num_contr => {
                        let mut occupation = Vec::with_capacity(num_contr);
                        for part in &parts[2..] {
                            let value = part.parse::<f64>().map_err(|_| {
                                InputFileParseError::InvalidFormat(format!(
                                    "Invalid occupation number for AIMP projection lqn {} in basis file '{}': expected a number, found '{}'",
                                    lqn, path, part
                                ))
                            })?;
                            occupation.push(value);
                        }
                        occupation.into_boxed_slice()
                    }
                    _ => {
                        return Err(InputFileParseError::InvalidFormat(format!(
                            "Expected 'num_prim num_contr' or 'num_prim num_contr' followed by {} occupation numbers for AIMP projection lqn {} in basis file '{}', found '{}'",
                            num_contr, lqn, path, line.trim()
                        )));
                    }
                };
                parser.clear_peek();

                // A projection constant per contracted function, the exponents, the coefficients.
                let energies = parser.read_n_f64(num_contr)?.into_boxed_slice();
                let exponents = parser.read_n_f64(num_prim)?;
                let coefficients = parser.read_coefficient_matrix(num_prim, num_contr)?;

                projection_shells.push(InputBasisShell {
                    lqn,
                    exponents: exponents.into_boxed_slice(),
                    coefficients,
                    energy: Some(energies),
                    occupation: Some(occupation),
                });
            }
            Some(InputAimp {
                potential,
                projection_shells,
            })
        }
    };

    if valence.is_none() && core.is_none() {
        return Err(InputFileParseError::InvalidFormat(format!(
            "Invalid basis set for element '{}' in basis file '{}': no valence shells and no core/embedding provided",
            element, path
        )));
    }

    Ok(ParsedBasisSet {
        charge,
        valence,
        core,
    })
}

// The fraction of a contracted function's norm that lies in the span of `orbitals` (all on one centre, same l,
// coefficients on normalized primitives): <f|P|f> / <f|f>, P the projector onto the orbitals' span.
fn fraction_in_span(lqn: usize, exponents: &[f64], function: &[f64], orbital_exponents: &[f64], orbitals: &Matrix<f64>) -> f64 {
    let s = |a: f64, b: f64| (2.0 * (a * b).sqrt() / (a + b)).powf(lqn as f64 + 1.5);
    let (n, m) = (orbital_exponents.len(), orbitals.cols());
    // <f|f>, <orbital_j|f>, <orbital_i|orbital_j>
    let ff: f64 = exponents.iter().zip(function).flat_map(|(&a, &ca)| exponents.iter().zip(function).map(move |(&b, &cb)| ca * cb * s(a, b))).sum();
    let of: Vec<f64> = (0..m).map(|j| (0..n).map(|p| orbitals[(p, j)] * exponents.iter().zip(function).map(|(&b, &cb)| cb * s(orbital_exponents[p], b)).sum::<f64>()).sum()).collect();
    let mut oo = Matrix::<f64>::new(m, m);
    for i in 0..m {
        for j in 0..m {
            oo[(i, j)] = (0..n).flat_map(|p| (0..n).map(move |q| (p, q))).map(|(p, q)| orbitals[(p, i)] * orbitals[(q, j)] * s(orbital_exponents[p], orbital_exponents[q])).sum();
        }
    }
    let inverse = oo.symmetric_pseudo_inverse(1e-12);
    let inside: f64 = (0..m).flat_map(|i| (0..m).map(move |j| (i, j))).map(|(i, j)| of[i] * inverse[(i, j)] * of[j]).sum();
    inside / ff
}

// Whether `s` has the form of an element symbol: an uppercase letter, then possibly a lowercase one.
fn is_valid_element_symbol(s: &str) -> bool {
    let mut chars = s.chars();
    match (chars.next(), chars.next(), chars.next()) {
        (Some(first), None, _) => first.is_ascii_uppercase(),
        (Some(first), Some(second), None) => first.is_ascii_uppercase() && second.is_ascii_lowercase(),
        _ => false,
    }
}

// The basis blocks of the input file `path` (at least one).
pub(crate) fn parse_input_file(path: &str) -> Result<ParsedInput, InputFileParseError> {
    let mut parser = InputParser::new(path)?;

    parser.find_line("basis").ok_or_else(|| {
        InputFileParseError::InvalidFormat(format!("Expected 'basis' section in input file '{}', but it was not found", path))
    })?;
    let mut basis_blocks = vec![parse_basis_block(&mut parser, path)?];

    while parser.find_line("basis").is_some() {
        basis_blocks.push(parse_basis_block(&mut parser, path)?);
    }

    Ok(ParsedInput { basis_blocks })
}

// One "Basis set ... End" block after its "basis" line: the spec line, the options, the centres.
// The ANO-RCC shorthand contractions (`O.ANO-RCC-VDZP` for `O.ANO-RCC...3s2p1d.`), as in the Molcas basis library.
const ANO_RCC_ALIASES: &str = include_str!("../data/ANO-RCC.aliases");

// The spec an alias names (matched case-insensitively), or the spec itself.
pub(crate) fn resolve_alias(spec: &str) -> String {
    ANO_RCC_ALIASES
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once(char::is_whitespace).map(|(alias, target)| (alias, target.trim())))
        .find(|(alias, _)| alias.eq_ignore_ascii_case(spec))
        .map_or_else(|| spec.to_string(), |(_, target)| target.to_string())
}

fn parse_basis_block(parser: &mut InputParser, path: &str) -> Result<InputBasisBlock, InputFileParseError> {
    // The spec, element.file[.author.primitive_set.contracted_set.other...]; only the element is checked here. An
    // alias (ANO-RCC-VDZP) is replaced by the spec it names.
    parser.skip_blank_lines();
    let line = parser.expect_peek_line("expected a basis set specification line after 'basis'")?;
    let basis_set = resolve_alias(line.trim());
    let parts: Vec<&str> = basis_set.split('.').collect();
    if parts.len() < 2 {
        return Err(InputFileParseError::InvalidFormat(format!(
            "Invalid basis set specification line in input file '{}': expected at least 'element.file', found '{}'",
            path, basis_set
        )));
    }
    let element = parts[0];
    if !is_valid_element_symbol(element) {
        return Err(InputFileParseError::InvalidFormat(format!(
            "Invalid element symbol in basis set specification line in input file '{}': expected 1-2 letters with first uppercase and second lowercase if present, found '{}'",
            path, element
        )));
    }
    parser.clear_peek();

    // The options, in any order, up to the first line that is not one.
    let mut pseudocharge = false;
    let mut charge: Option<f64> = None;
    let mut harmonic_overrides = HarmonicOverrides::default();
    let mut orthogonality = false;
    loop {
        parser.skip_blank_lines();
        let Some(option_line) = parser.peek_line().map(str::to_string) else {
            break;
        };
        let mut tokens = option_line.trim().split_whitespace();
        let Some(keyword) = tokens.next() else {
            break;
        };
        match keyword.to_uppercase().as_str() {
            "PSEUDOCHARGE" | "GHOST" => {
                pseudocharge = true;
                parser.clear_peek();
            }
            "CARTESIAN" | "SPHERICAL" => {
                let harmonic = if keyword.eq_ignore_ascii_case("CARTESIAN") {
                    HarmonicType::Cartesian
                } else {
                    HarmonicType::Spherical
                };
                parser.clear_peek();
                let target = parse_harmonic_target(parser, tokens, keyword, path)?;
                harmonic_overrides.set(target, harmonic);
            }
            "ORTHOGONALITY" => {
                orthogonality = true;
                parser.clear_peek();
            }
            "CHARGE" => {
                let same_line_value = tokens.next().map(str::to_string);
                parser.clear_peek();
                let value_str = match same_line_value {
                    Some(value) => value,
                    None => {
                        let next = parser.expect_peek_line("expected a charge value after 'Charge'")?;
                        parser.clear_peek();
                        next
                    }
                };
                let value = value_str.trim().parse::<f64>().map_err(|_| {
                    InputFileParseError::InvalidFormat(format!(
                        "Invalid charge value in input file '{}': expected a number, found '{}'",
                        path,
                        value_str.trim()
                    ))
                })?;
                // A zero charge makes a ghost centre.
                if value == 0.0 {
                    pseudocharge = true;
                } else {
                    charge = Some(value);
                }
            }
            _ => break,
        }
    }

    // "label x y z [Bohr|Angstrom]" lines up to "End", bohr the default.
    let mut centres = Vec::new();
    loop {
        parser.skip_blank_lines();
        let line = parser.expect_peek_line("expected a coordinate line or 'End'")?;
        if line.trim().to_uppercase().contains("END") {
            parser.clear_peek();
            break;
        }

        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.len() < 4 || parts.len() > 5 {
            return Err(InputFileParseError::InvalidFormat(format!(
                "Invalid coordinate line in input file '{}': expected 'label x y z' with an optional unit, found '{}'",
                path,
                line.trim()
            )));
        }
        let label = parts[0].to_string();
        let mut x = parse_coordinate(parts[1], "x", &label, path)?;
        let mut y = parse_coordinate(parts[2], "y", &label, path)?;
        let mut z = parse_coordinate(parts[3], "z", &label, path)?;
        if let Some(unit) = parts.get(4) {
            match unit.to_uppercase().as_str() {
                "BOHR" => {}
                "ANGSTROM" => {
                    x *= constants::ANGSTROM2AU;
                    y *= constants::ANGSTROM2AU;
                    z *= constants::ANGSTROM2AU;
                }
                other => {
                    return Err(InputFileParseError::InvalidFormat(format!(
                        "Invalid unit for atom '{}' in input file '{}': expected 'Bohr' or 'Angstrom', found '{}'",
                        label, path, other
                    )));
                }
            }
        }
        parser.clear_peek();

        centres.push(InputCoordinates { x, y, z });
    }

    Ok(InputBasisBlock {
        basis_set,
        charge,
        pseudocharge,
        harmonic_overrides,
        orthogonality,
        centres,
    })
}

// One coordinate as a number.
fn parse_coordinate(value: &str, axis: &str, label: &str, path: &str) -> Result<f64, InputFileParseError> {
    value.parse::<f64>().map_err(|_| {
        InputFileParseError::InvalidFormat(format!(
            "Invalid {} coordinate for atom '{}' in input file '{}': expected a number, found '{}'",
            axis, label, path, value
        ))
    })
}

#[cfg(test)]
mod harmonic_option_tests {
    use super::*;

    fn write_and_parse(tag: &str, body: &str) -> ParsedInput {
        let path = std::env::temp_dir().join(format!("aedelsten_harmonic_option_test_{}_{}.inp", std::process::id(), tag));
        std::fs::write(&path, body).unwrap();
        let result = parse_input_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        result
    }

    #[test]
    fn defaults_to_molcas_rule_with_no_cards() {
        let parsed = write_and_parse("defaults", "Basis set\nH.STO-3G\nH1 0.0 0.0 0.0\nEnd\n");
        let block = &parsed.get_basis_blocks()[0];
        assert_eq!(block.resolve_harmonic_type(0), HarmonicType::Cartesian);
        assert_eq!(block.resolve_harmonic_type(1), HarmonicType::Cartesian);
        assert_eq!(block.resolve_harmonic_type(2), HarmonicType::Spherical);
        assert_eq!(block.resolve_harmonic_type(4), HarmonicType::Spherical);
    }

    #[test]
    fn blanket_all_card_is_overridden_by_specific_shell_cards() {
        let parsed = write_and_parse(
            "blanket_and_specific",
            "Basis set\nH.STO-3G\nSpherical all\nCartesian d f\nH1 0.0 0.0 0.0\nEnd\n",
        );
        let block = &parsed.get_basis_blocks()[0];
        // s/p forced Cartesian regardless of the blanket Spherical card.
        assert_eq!(block.resolve_harmonic_type(0), HarmonicType::Cartesian);
        assert_eq!(block.resolve_harmonic_type(1), HarmonicType::Cartesian);
        // d/f explicitly overridden back to Cartesian.
        assert_eq!(block.resolve_harmonic_type(2), HarmonicType::Cartesian);
        assert_eq!(block.resolve_harmonic_type(3), HarmonicType::Cartesian);
        // g falls through to the blanket Spherical card.
        assert_eq!(block.resolve_harmonic_type(4), HarmonicType::Spherical);
    }

    #[test]
    fn integer_and_character_shell_designators_are_equivalent() {
        let parsed = write_and_parse(
            "integer_syntax",
            "Basis set\nH.STO-3G\nCartesian 2\nSpherical 3\nH1 0.0 0.0 0.0\nEnd\n",
        );
        let block = &parsed.get_basis_blocks()[0];
        assert_eq!(block.resolve_harmonic_type(2), HarmonicType::Cartesian); // "2" == 'd'
        assert_eq!(block.resolve_harmonic_type(3), HarmonicType::Spherical); // "3" == 'f'
    }

    #[test]
    fn shell_list_may_fall_on_the_following_line() {
        let parsed = write_and_parse(
            "next_line_list",
            "Basis set\nH.STO-3G\nCartesian\nd f\nH1 0.0 0.0 0.0\nEnd\n",
        );
        let block = &parsed.get_basis_blocks()[0];
        assert_eq!(block.resolve_harmonic_type(2), HarmonicType::Cartesian);
        assert_eq!(block.resolve_harmonic_type(3), HarmonicType::Cartesian);
        assert_eq!(block.resolve_harmonic_type(4), HarmonicType::Spherical); // untouched, default rule
    }

    #[test]
    fn rejects_unknown_shell_designator() {
        let path = std::env::temp_dir().join(format!("aedelsten_harmonic_option_test_{}_bad_designator.inp", std::process::id()));
        std::fs::write(&path, "Basis set\nH.STO-3G\nCartesian j\nH1 0.0 0.0 0.0\nEnd\n").unwrap();
        let result = parse_input_file(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(result, Err(InputFileParseError::InvalidFormat(_))));
    }
}

#[cfg(test)]
mod basis_header_resolution_tests {
    use super::*;

    // A library file where a run finds it (the bundled library, or the tests' fixtures through AEDELSTEN_BASIS_LIBRARY).
    fn library_path(file: &str) -> String {
        crate::find_library_file(file).unwrap().unwrap_or_else(|| panic!("{file} not found")).to_string_lossy().into_owned()
    }

    // The header is "/C.ANO-rcc.Roos.14s9p4d3f2g.8s8p4d3f2g.": the file component in lower case.
    #[test]
    fn contraction_truncation_keeps_requested_columns_and_drops_unmentioned_shells() {
        let path = library_path("ANO-RCC");
        let parsed = parse_basis_file("C.ANO-RCC...4s3p2d", &path).unwrap();
        let valence = parsed.get_valence().expect("C.ANO-RCC has a valence basis");

        // f and g are dropped.
        assert_eq!(valence.iter().map(|s| s.get_lqn()).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(valence[0].get_coefficients().cols(), 4); // s: 4 of 8
        assert_eq!(valence[1].get_coefficients().cols(), 3); // p: 3 of 8
        assert_eq!(valence[2].get_coefficients().cols(), 2); // d: 2 of 4
    }

    #[test]
    fn contraction_truncation_matching_the_real_contracted_set_is_a_no_op() {
        let path = library_path("ANO-RCC");
        let parsed = parse_basis_file("C.ANO-RCC...8s8p4d3f2g", &path).unwrap();
        let valence = parsed.get_valence().unwrap();
        assert_eq!(valence.iter().map(|s| s.get_lqn()).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        assert_eq!(valence[0].get_coefficients().cols(), 8);
    }

    #[test]
    fn requesting_more_columns_than_available_is_an_error() {
        let path = library_path("ANO-RCC");
        let result = parse_basis_file("C.ANO-RCC...20s", &path);
        assert!(matches!(result, Err(InputFileParseError::InvalidFormat(_))), "{:?}", result);
    }

    // TEST-EMB-AIMP has two Mg entries by the same author.
    #[test]
    fn ambiguous_element_file_query_is_an_error() {
        let path = library_path("TEST-EMB-AIMP");
        let result = parse_basis_file("Mg.TEST-EMB-AIMP", &path);
        assert!(matches!(result, Err(InputFileParseError::InvalidFormat(_))), "{:?}", result);
    }

    #[test]
    fn primitive_set_alone_disambiguates_when_it_is_unique() {
        // Two entries of one element, file and author, with primitive sets 2s and 3s.
        let entry = |primitives: &str, exponents: &[f64]| {
            let mut entry = format!("/H.TWO.Author.{primitives}.1s.\nreference\ndescription\n  1.0 0\n  {} 1\n", exponents.len());
            exponents.iter().for_each(|e| entry += &format!("  {e}\n"));
            exponents.iter().for_each(|_| entry += "  0.5\n");
            entry
        };
        let path = std::env::temp_dir().join(format!("aedelsten_two_entries_{}", std::process::id()));
        std::fs::write(&path, entry("2s", &[1.0, 0.2]) + &entry("3s", &[3.0, 1.0, 0.2])).unwrap();
        let three = parse_basis_file("H.TWO.Author.3s", path.to_str().unwrap());
        let both = parse_basis_file("H.TWO.Author", path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        assert_eq!(three.unwrap().get_valence().unwrap()[0].get_exponents().len(), 3);
        assert!(matches!(both, Err(InputFileParseError::InvalidFormat(_))), "{:?}", both);
    }
}

#[cfg(test)]
mod orthogonality_function_aimp_tests {
    use super::*;

    // A library file where a run finds it (the bundled library, or the tests' fixtures through AEDELSTEN_BASIS_LIBRARY).
    fn library_path(file: &str) -> String {
        crate::find_library_file(file).unwrap().unwrap_or_else(|| panic!("{file} not found")).to_string_lossy().into_owned()
    }

    // Every entry of a library file: the header without its leading '/' and trailing '.'.
    fn entries(file: &str) -> Vec<String> {
        std::fs::read_to_string(library_path(file)).unwrap().lines()
            .filter(|l| l.starts_with('/'))
            .map(|l| l.trim_start_matches('/').trim_end().trim_end_matches('.').to_string())
            .collect()
    }

    fn is_rejected(spec: &str, file: &str) -> bool {
        matches!(parse_basis_file(spec, &library_path(file)), Err(InputFileParseError::InvalidFormat(m)) if m.contains("orthogonality functions"))
    }

    // The orthogonality-function entry of TEST-EMB-AIMP (the one with basis functions, label not 0s.0s) is rejected,
    // the others are not; the core AIMP of TEST-CORE-AIMP is not.
    #[test]
    fn exactly_the_orthogonality_function_entries_are_rejected() {
        for spec in entries("TEST-EMB-AIMP") {
            let has_basis_functions = !spec.contains(".0s.0s.");
            assert_eq!(is_rejected(&spec, "TEST-EMB-AIMP"), has_basis_functions, "{spec}");
        }
        for spec in entries("TEST-CORE-AIMP") {
            assert!(parse_basis_file(&spec, &library_path("TEST-CORE-AIMP")).is_ok(), "{spec}");
        }
    }

    // The ANO-RCC shorthands resolve to their contractions, in any case; anything else is left as it is; and a
    // resolved spec parses (every alias of every element).
    #[test]
    fn the_ano_rcc_aliases_name_their_contractions() {
        assert_eq!(resolve_alias("O.ANO-RCC-VDZP"), "O.ANO-rcc...3s2p1d.");
        assert_eq!(resolve_alias("h.ano-rcc-vdzp"), "H.ANO-rcc...2s1p.");
        assert_eq!(resolve_alias("Fe.ANO-RCC-VQZP"), "Fe.ANO-rcc...7s6p4d3f2g1h.");
        assert_eq!(resolve_alias("O.ANO-RCC...3s2p1d"), "O.ANO-RCC...3s2p1d");
        assert_eq!(resolve_alias("O.x2c-SVPall"), "O.x2c-SVPall");
        let aliases: Vec<&str> = ANO_RCC_ALIASES.lines().filter(|l| !l.starts_with('#')).filter_map(|l| l.split_whitespace().next()).collect();
        assert!(aliases.len() > 500, "{}", aliases.len());
        for alias in aliases {
            let spec = resolve_alias(alias);
            assert!(parse_basis_file(spec.trim_end_matches('.'), &library_path("ANO-RCC")).is_ok(), "{alias} -> {spec}");
        }
    }

    // A PROJOP shell without functions (here an l = 2 shell "0 0" appended to TEST-CORE-AIMP's) is an error naming the
    // entry and the l, not a panic when the orthogonality shells are taken.
    #[test]
    fn an_empty_projection_shell_is_an_error_naming_it() {
        let library = std::fs::read_to_string(library_path("TEST-CORE-AIMP")).unwrap();
        assert_eq!(library.matches("PROJOP\n    1\n").count(), 1);
        let empty_d = library.replace("PROJOP\n    1\n", "PROJOP\n    2\n").trim_end().to_string() + "\n    0    0\n";
        let path = std::env::temp_dir().join(format!("basis_parser_empty_projop_{}", std::process::id()));
        std::fs::write(&path, empty_d).unwrap();
        let result = parse_basis_file("Sc.TEST-CORE-AIMP", path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        let Err(InputFileParseError::InvalidFormat(message)) = result else {
            panic!("an empty projection shell must be rejected");
        };
        assert!(message.contains("Empty AIMP projection shell for lqn 2") && message.contains("Sc.TEST-CORE-AIMP"), "{message}");
    }

    // The bundled segmented set (x2c-SVPall, H-Rn, all-electron): every entry parses, its reference cited.
    #[test]
    fn every_x2c_svpall_entry_parses() {
        let specs = entries("X2C-SVPall");
        assert_eq!(specs.len(), 86);
        for spec in specs {
            let parsed = parse_basis_file(&spec, &library_path("X2C-SVPall")).unwrap_or_else(|e| panic!("{spec}: {e:?}"));
            assert!(parsed.valence.is_some(), "{spec}");
        }
    }

    #[test]
    fn the_error_names_the_entry_and_the_way_out() {
        let spec = "Mg.TEST-EMB-AIMP.Larsson.14s.1s.ECP.MgO-orthogonality";
        let Err(InputFileParseError::InvalidFormat(message)) = parse_basis_file(spec, &library_path("TEST-EMB-AIMP")) else {
            panic!("{spec} must be rejected");
        };
        assert!(message.contains(spec) && message.contains("ORTHOGONALITY"), "{message}");
    }
}
