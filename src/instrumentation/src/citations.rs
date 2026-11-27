// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The citation registry: what a run used, cited where it runs, read once at the end.
//!
//! - `cite(topic, &ENTRY)` registers a compiled-in entry. Each code keeps its own entries as `static Citation`s next to
//!   the code that cites them; the registry holds references, so citing an entry a second time costs a lock and a
//!   short scan, and allocates nothing (the paths an SCF iterates may cite).
//! - `cite_data(topic, text)` registers a reference that comes with data read at run time: the reference line of a
//!   basis-set or model-potential library entry. Its DOI is taken from the text (`doi:...`).
//!
//! One registry per process, shared by every code linked into it (the library and the program that drives it), so one
//! References section lists both. Entries are deduplicated by DOI, or by text when there is none; an entry cited
//! under several topics keeps them all. Under MPI every rank registers what it ran.
//!
//! `bibtex(&records)` writes them as a BibTeX file: one entry per reference, its key from the reference alone (the
//! first author's surname, the year, the title's first word), so it is the same in every run that cites it.

use std::borrow::Cow;
use std::fmt;
use std::sync::Mutex;

/// A compiled-in reference: a journal article (a literal of all the fields), a book (`Citation::book`), or unpublished
/// work (`Citation::unpublished`). An article or a book may have no DOI.
#[derive(Debug)]
pub struct Citation {
    /// The authors, separated by ", " (`B. A. Heß, C. M. Marian`).
    pub authors: &'static str,
    pub title: &'static str,
    /// The journal's abbreviated name; a book's publisher and place (`Wiley, Chichester`).
    pub journal: &'static str,
    /// Empty for a book.
    pub volume: &'static str,
    /// First and last page, separated by "-"; empty for a book.
    pub pages: &'static str,
    pub year: u16,
    /// The DOI without a prefix (`10.1063/1.456153`); empty when there is none.
    pub doi: &'static str,
}

impl Citation {
    /// Unpublished work: `title` says what the method is, `authors` who developed it, `year` when.
    pub const fn unpublished(authors: &'static str, title: &'static str, year: u16) -> Self {
        Citation { authors, title, journal: UNPUBLISHED, volume: "", pages: "", year, doi: "" }
    }

    /// A book: `publisher` its publisher and place (`Wiley, Chichester`), `doi` empty when it has none.
    pub const fn book(authors: &'static str, title: &'static str, publisher: &'static str, year: u16, doi: &'static str) -> Self {
        Citation { authors, title, journal: publisher, volume: "", pages: "", year, doi }
    }

    /// Whether the entry is unpublished work.
    pub fn is_unpublished(&self) -> bool {
        self.journal == UNPUBLISHED
    }

    /// Whether the entry is a book: published, with neither volume nor pages.
    pub fn is_book(&self) -> bool {
        !self.is_unpublished() && self.volume.is_empty() && self.pages.is_empty()
    }

    /// The DOI, none when the entry has none.
    pub fn get_doi(&self) -> Option<&'static str> {
        (!self.doi.is_empty()).then_some(self.doi)
    }
}

const UNPUBLISHED: &str = "unpublished work";

/// A registered reference: compiled in, or read with data.
#[derive(Clone, Debug)]
pub enum Reference {
    /// A compiled-in entry ([`cite`]).
    Entry(&'static Citation),
    /// A reference line read with data ([`cite_data`]), and the DOI found in it.
    Data { text: String, doi: Option<String> },
}

impl Reference {
    /// The DOI; none for an entry or a reference line without one.
    pub fn doi(&self) -> Option<&str> {
        match self {
            Reference::Entry(c) => c.get_doi(),
            Reference::Data { doi, .. } => doi.as_deref(),
        }
    }
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reference::Entry(c) => {
                if c.is_unpublished() || c.is_book() {
                    write!(f, "{}, \"{}\", {} ({})", c.authors, c.title, c.journal, c.year)?;
                } else {
                    write!(f, "{}, \"{}\", {} {}, {} ({})", c.authors, c.title, c.journal, c.volume, c.pages, c.year)?;
                }
                match c.get_doi() {
                    Some(doi) => write!(f, ". doi:{doi}"),
                    None => Ok(()),
                }
            }
            Reference::Data { text, .. } => f.write_str(text),
        }
    }
}

/// One line of the report: a reference and the topics it was cited for, in the order first cited.
#[derive(Clone, Debug)]
pub struct Record {
    /// The topics it was cited under, in the order first cited.
    pub topics: Vec<Cow<'static, str>>,
    /// The reference.
    pub reference: Reference,
}

static REGISTRY: Mutex<Vec<Record>> = Mutex::new(Vec::new());

fn same_doi(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn register(topic: Cow<'static, str>, found: impl Fn(&Reference) -> bool, make: impl FnOnce() -> Reference) {
    let mut registry = REGISTRY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match registry.iter_mut().find(|record| found(&record.reference)) {
        Some(record) => {
            if !record.topics.iter().any(|t| *t == topic) {
                record.topics.push(topic);
            }
        }
        None => registry.push(Record { topics: vec![topic], reference: make() }),
    }
}

/// Registers a compiled-in entry under `topic`.
pub fn cite(topic: &'static str, entry: &'static Citation) {
    register(
        Cow::Borrowed(topic),
        |r| match r {
            Reference::Entry(c) => match (c.get_doi(), entry.get_doi()) {
                (Some(a), Some(b)) => same_doi(a, b),
                _ => std::ptr::eq(*c, entry),
            },
            Reference::Data { doi, .. } => entry.get_doi().is_some_and(|e| doi.as_deref().is_some_and(|d| same_doi(d, e))),
        },
        || Reference::Entry(entry),
    );
}

/// Registers a reference read with data (e.g. `T.H. Dunning. J. Chem. Phys. 90 (1989) 1007-1023.
/// doi:10.1063/1.456153`) under `topic`. Blank text registers nothing.
pub fn cite_data(topic: impl Into<Cow<'static, str>>, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let doi = doi_in(text);
    register(
        topic.into(),
        |r| match (r.doi(), doi) {
            (Some(a), Some(b)) => same_doi(a, b),
            (None, None) => matches!(r, Reference::Data { text: t, .. } if t == text),
            _ => false,
        },
        || Reference::Data { text: text.to_string(), doi: doi.map(str::to_string) },
    );
}

// The DOI in a reference line: what follows "doi:" (any case) or "doi.org/", up to whitespace, without a final period.
fn doi_in(text: &str) -> Option<&str> {
    let lower = text.to_ascii_lowercase();
    let start = ["doi.org/", "doi:"].iter().find_map(|m| lower.find(m).map(|i| i + m.len()))?;
    let rest = text[start..].trim_start();
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let doi = rest[..end].trim_end_matches('.');
    (!doi.is_empty()).then_some(doi)
}

impl Reference {
    /// The BibTeX key: the first author's surname, the year and, for a compiled-in entry, the title's first word of
    /// four or more letters, in lower-case ASCII (`mcmurchie1978electron`). A reference read with data has no
    /// structure: its first word of three or more letters stands for the surname, and its first year in it for the
    /// year (`dunning1989`). [`bibtex`] makes keys that still coincide unique.
    pub fn bibtex_key(&self) -> String {
        match self {
            Reference::Entry(c) => {
                let first = c.authors.split(',').next().unwrap_or("");
                let surname = first.split_whitespace().last().unwrap_or("anonymous");
                let word = words(c.title).find(|w| w.chars().count() >= 4).unwrap_or_default();
                format!("{}{}{}", ascii_lower(surname), c.year, ascii_lower(&word))
            }
            Reference::Data { text, .. } => {
                let surname = words(text).find(|w| w.chars().count() >= 3).unwrap_or_else(|| "data".to_string());
                let year = year_in(text).map_or_else(|| "nd".to_string(), |y| y.to_string());
                format!("{}{year}", ascii_lower(&surname))
            }
        }
    }

    /// This reference as a BibTeX entry under `key`, with `topics` as its keywords: `@article` for a journal article,
    /// `@book` for a book, `@unpublished` for unpublished work, `@misc` with the text as its note for a reference read
    /// with data.
    pub fn bibtex(&self, key: &str, topics: &[Cow<'static, str>]) -> String {
        let mut fields: Vec<(&str, String)> = Vec::new();
        let kind = match self {
            Reference::Entry(c) => {
                fields.push(("author", c.authors.split(", ").collect::<Vec<_>>().join(" and ")));
                fields.push(("title", format!("{{{}}}", c.title)));
                if c.is_unpublished() {
                    fields.push(("note", UNPUBLISHED.to_string()));
                } else if c.is_book() {
                    fields.push(("publisher", c.journal.to_string()));
                } else {
                    fields.push(("journal", c.journal.to_string()));
                    fields.push(("volume", c.volume.to_string()));
                    fields.push(("pages", c.pages.replace('-', "--")));
                }
                fields.push(("year", c.year.to_string()));
                fields.push(("doi", c.doi.to_string()));
                if c.is_unpublished() {
                    "unpublished"
                } else if c.is_book() {
                    "book"
                } else {
                    "article"
                }
            }
            Reference::Data { text, doi } => {
                fields.push(("note", text.clone()));
                if let Some(year) = year_in(text) {
                    fields.push(("year", year.to_string()));
                }
                if let Some(doi) = doi {
                    fields.push(("doi", doi.clone()));
                }
                "misc"
            }
        };
        if !topics.is_empty() {
            fields.push(("keywords", topics.join(", ")));
        }
        let mut out = format!("@{kind}{{{key},\n");
        for (name, value) in fields.iter().filter(|(_, v)| !v.is_empty()) {
            out.push_str(&format!("  {name} = {{{}}},\n", escape_bibtex(value)));
        }
        out.push_str("}\n");
        out
    }
}

/// `records` as a BibTeX file, in their order: one entry each, keyed by [`Reference::bibtex_key`], a key that occurs
/// again given a letter (`roos1980superb`), its topics as keywords.
pub fn bibtex(records: &[Record]) -> String {
    let mut keys: Vec<String> = Vec::with_capacity(records.len());
    let mut out = String::new();
    for record in records {
        let base = record.reference.bibtex_key();
        let key = std::iter::once(base.clone())
            .chain((b'b'..=b'z').map(|c| format!("{base}{}", c as char)))
            .chain((2..).map(|n| format!("{base}_{n}")))
            .find(|k| !keys.contains(k))
            .unwrap_or(base);
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&record.reference.bibtex(&key, &record.topics));
        keys.push(key);
    }
    out
}

// The alphanumeric runs of `text`.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty() && w.chars().all(char::is_alphabetic)).map(str::to_string)
}

// The first year (1800-2099) standing alone in `text`.
fn year_in(text: &str) -> Option<u16> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter(|d| d.len() == 4)
        .filter_map(|d| d.parse::<u16>().ok())
        .find(|y| (1800..2100).contains(y))
}

// Lower-case ASCII letters and digits: accented Latin letters lose their accent, other characters are dropped.
fn ascii_lower(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars().flat_map(char::to_lowercase) {
        match c {
            'a'..='z' | '0'..='9' => out.push(c),
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' => out.push('a'),
            'æ' => out.push_str("ae"),
            'ç' | 'č' | 'ć' => out.push('c'),
            'è' | 'é' | 'ê' | 'ë' | 'ě' => out.push('e'),
            'ì' | 'í' | 'î' | 'ï' | 'ı' => out.push('i'),
            'ñ' | 'ń' | 'ň' => out.push('n'),
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ő' => out.push('o'),
            'ř' => out.push('r'),
            'š' | 'ś' => out.push('s'),
            'ß' => out.push_str("ss"),
            'ù' | 'ú' | 'û' | 'ü' | 'ů' | 'ű' => out.push('u'),
            'ý' | 'ÿ' => out.push('y'),
            'ž' | 'ź' | 'ż' => out.push('z'),
            _ => {}
        }
    }
    out
}

// The characters BibTeX treats as commands or comments, escaped; braces are kept (a title's protecting braces).
fn escape_bibtex(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if matches!(c, '&' | '%' | '$' | '#' | '_') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Everything cited so far, in the order first cited.
pub fn citations() -> Vec<Record> {
    REGISTRY.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
}

/// Forgets everything cited (a host that runs several independent jobs in one process).
pub fn clear_citations() {
    REGISTRY.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    static A: Citation = Citation {
        authors: "A. Author",
        title: "A title",
        journal: "J. Test",
        volume: "1",
        pages: "2-3",
        year: 2000,
        doi: "10.1000/a",
    };

    static NEW_1: Citation = Citation::unpublished("E. D. Larsson", "A first method of this code", 2026);
    static WITHOUT_DOI: Citation = Citation {
        authors: "V. I. Lebedev, D. N. Laikov",
        title: "A quadrature formula for the sphere of the 131st algebraic order of accuracy",
        journal: "Dokl. Math.",
        volume: "59",
        pages: "477-481",
        year: 1999,
        doi: "",
    };
    static OTHER_WITHOUT_DOI: Citation = Citation { title: "Another paper without a DOI", ..WITHOUT_DOI };
    static BOOK: Citation =
        Citation::book("T. Helgaker, P. Jørgensen, J. Olsen", "Molecular Electronic-Structure Theory", "Wiley, Chichester", 2000, "10.1002/9781119019572");
    static NEW_2: Citation = Citation::unpublished("E. D. Larsson", "A second method of this code", 2026);

    // The registry is process-wide and the tests of this crate run in parallel: one test exercises it in sequence.
    #[test]
    fn registry_deduplicates_and_keeps_topics() {
        clear_citations();
        cite("Integrals", &A);
        cite("Integrals", &A);
        cite("Gradients", &A);
        cite_data("Basis sets", "T.H. Dunning. J. Chem. Phys. 90 (1989) 1007-1023. doi:10.1063/1.456153");
        cite_data("Basis sets", "T.H. Dunning. J. Chem. Phys. 90 (1989) 1007-1023. DOI:10.1063/1.456153.");
        cite_data("Basis sets", "A reference without one");
        cite_data("Basis sets", "A reference without one");
        cite_data("Basis sets", "   ");
        cite_data("Model potentials", "Same paper. doi:10.1000/A");
        cite("Integrals", &NEW_1);
        cite("Integrals", &NEW_2);
        cite("Gradients", &NEW_1);
        cite("Grids", &WITHOUT_DOI);
        cite("Grids", &WITHOUT_DOI);
        cite("Grids", &OTHER_WITHOUT_DOI);
        let records = citations();
        assert_eq!(records.len(), 7);
        assert_eq!(records[0].topics, ["Integrals", "Gradients", "Model potentials"]);
        assert_eq!(records[0].reference.to_string(), "A. Author, \"A title\", J. Test 1, 2-3 (2000). doi:10.1000/a");
        assert_eq!(records[1].reference.doi(), Some("10.1063/1.456153"));
        assert_eq!(records[2].reference.doi(), None);
        // Unpublished entries have no DOI, and two of them are never taken for one.
        assert_eq!(records[3].topics, ["Integrals", "Gradients"]);
        assert_eq!(records[3].reference.doi(), None);
        assert_eq!(records[3].reference.to_string(), "E. D. Larsson, \"A first method of this code\", unpublished work (2026)");
        assert_eq!(records[4].reference.to_string(), "E. D. Larsson, \"A second method of this code\", unpublished work (2026)");
        // Articles without a DOI likewise: the same entry once, two of them two.
        assert!(matches!(records[5].reference, Reference::Entry(c) if std::ptr::eq(c, &WITHOUT_DOI)));
        assert!(matches!(records[6].reference, Reference::Entry(c) if std::ptr::eq(c, &OTHER_WITHOUT_DOI)));
        clear_citations();
        assert!(citations().is_empty());
    }

    #[test]
    fn bibtex_entries_have_stable_keys() {
        static HESS: Citation = Citation {
            authors: "B. A. Heß, C. M. Marian, U. Wahlgren, O. Gropen",
            title: "A mean-field spin-orbit method applicable to correlated wavefunctions",
            journal: "Chem. Phys. Lett.",
            volume: "251",
            pages: "365-371",
            year: 1996,
            doi: "10.1016/0009-2614(96)00119-4",
        };
        static HESS_AGAIN: Citation = Citation { doi: "10.1000/other", ..HESS };
        let record = |topics: &[&'static str], reference| Record { topics: topics.iter().map(|&t| Cow::Borrowed(t)).collect(), reference };
        let records = [
            record(&["Spin-orbit", "AMFI"], Reference::Entry(&HESS)),
            record(&["Spin-orbit"], Reference::Entry(&HESS_AGAIN)),
            record(&["Integrals"], Reference::Entry(&NEW_1)),
            record(&["Basis sets"], Reference::Data {
                text: "T.H. Dunning. J. Chem. Phys. 90 (1989) 1007-1023. doi:10.1063/1.456153".to_string(),
                doi: Some("10.1063/1.456153".to_string()),
            }),
        ];
        assert_eq!(records[0].reference.bibtex_key(), "hess1996mean");
        assert_eq!(records[2].reference.bibtex_key(), "larsson2026first");
        assert_eq!(records[3].reference.bibtex_key(), "dunning1989");
        let file = bibtex(&records);
        assert_eq!(
            file.split("\n\n").next().unwrap(),
            "@article{hess1996mean,\n  author = {B. A. Heß and C. M. Marian and U. Wahlgren and O. Gropen},\n  \
             title = {{A mean-field spin-orbit method applicable to correlated wavefunctions}},\n  \
             journal = {Chem. Phys. Lett.},\n  volume = {251},\n  pages = {365--371},\n  year = {1996},\n  \
             doi = {10.1016/0009-2614(96)00119-4},\n  keywords = {Spin-orbit, AMFI},\n}"
        );
        assert!(file.contains("@article{hess1996meanb,"), "a repeated key gets a letter:\n{file}");
        assert!(file.contains("@unpublished{larsson2026first,\n  author = {E. D. Larsson},"), "{file}");
        assert!(file.contains("  note = {unpublished work},\n  year = {2026},\n  keywords = {Integrals},\n}"), "{file}");
        assert!(file.contains("@misc{dunning1989,\n  note = {T.H. Dunning."), "{file}");
        assert!(file.contains("  year = {1989},\n  doi = {10.1063/1.456153},\n"), "{file}");
        assert_eq!(escape_bibtex("A & B_1 50%"), "A \\& B\\_1 50\\%");
    }

    /// An article without a DOI is an article, a book a book; neither is unpublished work.
    #[test]
    fn articles_without_a_doi_and_books_print_as_what_they_are() {
        assert!(!WITHOUT_DOI.is_unpublished() && !WITHOUT_DOI.is_book() && WITHOUT_DOI.get_doi().is_none());
        assert!(BOOK.is_book() && !BOOK.is_unpublished() && !NEW_1.is_book());
        assert_eq!(
            Reference::Entry(&WITHOUT_DOI).to_string(),
            "V. I. Lebedev, D. N. Laikov, \"A quadrature formula for the sphere of the 131st algebraic order of accuracy\", Dokl. Math. 59, 477-481 (1999)"
        );
        assert_eq!(
            Reference::Entry(&BOOK).to_string(),
            "T. Helgaker, P. Jørgensen, J. Olsen, \"Molecular Electronic-Structure Theory\", Wiley, Chichester (2000). doi:10.1002/9781119019572"
        );
        let article = Reference::Entry(&WITHOUT_DOI).bibtex("lebedev1999quadrature", &[]);
        assert!(article.starts_with("@article{lebedev1999quadrature,") && article.contains("  pages = {477--481},\n  year = {1999},\n}"), "{article}");
        assert!(!article.contains("doi") && !article.contains("unpublished"), "{article}");
        let book = Reference::Entry(&BOOK).bibtex("helgaker2000molecular", &[Cow::Borrowed("CASSCF")]);
        assert_eq!(
            book,
            "@book{helgaker2000molecular,\n  author = {T. Helgaker and P. Jørgensen and J. Olsen},\n  \
             title = {{Molecular Electronic-Structure Theory}},\n  publisher = {Wiley, Chichester},\n  year = {2000},\n  \
             doi = {10.1002/9781119019572},\n  keywords = {CASSCF},\n}\n"
        );
    }

    #[test]
    fn doi_is_found_in_a_reference_line() {
        assert_eq!(doi_in("X. J. Chem. Phys. 98 (1993) 1358. doi:10.1063/1.464303"), Some("10.1063/1.464303"));
        assert_eq!(doi_in("see https://doi.org/10.1021/jp031064+ for it"), Some("10.1021/jp031064+"));
        assert_eq!(doi_in("no identifier"), None);
    }
}
