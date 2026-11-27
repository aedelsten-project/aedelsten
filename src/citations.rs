// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The library's compiled-in references, cited (`instrumentation::cite`) where the context runs the method. The
// references of basis-set and model-potential data are cited by the basis parser from the library files themselves.

use instrumentation::{cite, Citation};

// The reference list's topics.
pub(crate) const INTEGRALS: &str = "Integrals";
pub(crate) const GRADIENTS: &str = "Integral derivatives";
pub(crate) const MODEL_POTENTIALS: &str = "Ab initio model potentials";
pub(crate) const POLARIZABLE_EMBEDDING: &str = "Polarizable embedding";
pub(crate) const RELATIVITY: &str = "Relativistic Hamiltonian";
pub(crate) const SYMMETRY: &str = "Point-group symmetry";
pub(crate) const START_GUESS: &str = "Start guess";

/// The integrals: every context.
pub(crate) static MCMURCHIE_DAVIDSON_1978: Citation = Citation {
    authors: "L. E. McMurchie, E. R. Davidson",
    title: "One- and two-electron integrals over Cartesian Gaussian functions",
    journal: "J. Comput. Phys.",
    volume: "26",
    pages: "218-231",
    year: 1978,
    doi: "10.1016/0021-9991(78)90092-X",
};

/// The spherical transformation: a spherical shell with l >= 2.
pub(crate) static SCHLEGEL_FRISCH_1995: Citation = Citation {
    authors: "H. B. Schlegel, M. J. Frisch",
    title: "Transformation between Cartesian and pure spherical harmonic Gaussians",
    journal: "Int. J. Quantum Chem.",
    volume: "54",
    pages: "83-87",
    year: 1995,
    doi: "10.1002/qua.560540202",
};

// The ERI evaluation follows its formulation: McMurchie-Davidson as BLAS level-3 products.
pub(crate) static NEESE_SHARK_2023: Citation = Citation {
    authors: "F. Neese",
    title: "The SHARK integral generation and digestion system",
    journal: "J. Comput. Chem.",
    volume: "44",
    pages: "381-396",
    year: 2023,
    doi: "10.1002/jcc.26942",
};

/// Schwarz screening.
pub(crate) static HASER_AHLRICHS_1989: Citation = Citation {
    authors: "M. Häser, R. Ahlrichs",
    title: "Improvements on the direct SCF method",
    journal: "J. Comput. Chem.",
    volume: "10",
    pages: "104-111",
    year: 1989,
    doi: "10.1002/jcc.540100111",
};

/// Integral derivatives.
pub(crate) static HELGAKER_TAYLOR_1992: Citation = Citation {
    authors: "T. Helgaker, P. R. Taylor",
    title: "On the evaluation of derivatives of Gaussian integrals",
    journal: "Theor. Chim. Acta",
    volume: "83",
    pages: "177-183",
    year: 1992,
    doi: "10.1007/BF01132826",
};

/// Polarizable embedding: the induced dipoles.
pub(crate) static OLSEN_AIDAS_KONGSTED_2010: Citation = Citation {
    authors: "J. M. Olsen, K. Aidas, J. Kongsted",
    title: "Excited states in solution through polarizable embedding",
    journal: "J. Chem. Theory Comput.",
    volume: "6",
    pages: "3721-3734",
    year: 2010,
    doi: "10.1021/ct1003803",
};

// The conjugate gradients of the induced-dipole solver.
pub(crate) static HESTENES_STIEFEL_1952: Citation = Citation {
    authors: "M. R. Hestenes, E. Stiefel",
    title: "Methods of conjugate gradients for solving linear systems",
    journal: "J. Res. Natl. Bur. Stand.",
    volume: "49",
    pages: "409-436",
    year: 1952,
    doi: "10.6028/jres.049.044",
};

/// Thole damping.
pub(crate) static THOLE_1981: Citation = Citation {
    authors: "B. T. Thole",
    title: "Molecular polarizabilities calculated with a modified dipole interaction",
    journal: "Chem. Phys.",
    volume: "59",
    pages: "341-350",
    year: 1981,
    doi: "10.1016/0301-0104(81)85176-2",
};

// The effective external field of polarizable embedding.
pub(crate) static LIST_JENSEN_KONGSTED_2016: Citation = Citation {
    authors: "N. H. List, H. J. Aa. Jensen, J. Kongsted",
    title: "Local electric fields and molecular properties in heterogeneous environments through polarizable embedding",
    journal: "Phys. Chem. Chem. Phys.",
    volume: "18",
    pages: "10070-10080",
    year: 2016,
    doi: "10.1039/c6cp00669h",
};

// The default Thole parameter, a = 2.1304.
pub(crate) static VAN_DUIJNEN_SWART_1998: Citation = Citation {
    authors: "P. Th. van Duijnen, M. Swart",
    title: "Molecular and atomic polarizabilities: Thole's model revisited",
    journal: "J. Phys. Chem. A",
    volume: "102",
    pages: "2399-2407",
    year: 1998,
    doi: "10.1021/jp980221f",
};

/// Any AIMP.
pub(crate) static HUZINAGA_AIMP_1987: Citation = Citation {
    authors: "S. Huzinaga, L. Seijo, Z. Barandiarán, M. Klobukowski",
    title: "The ab initio model potential method. Main group elements",
    journal: "J. Chem. Phys.",
    volume: "86",
    pages: "2132-2145",
    year: 1987,
    doi: "10.1063/1.452111",
};

/// An embedding AIMP.
pub(crate) static BARANDIARAN_SEIJO_1988: Citation = Citation {
    authors: "Z. Barandiarán, L. Seijo",
    title: "The ab initio model potential representation of the crystalline environment. Theoretical study of the local distortion on NaCl:Cu+",
    journal: "J. Chem. Phys.",
    volume: "89",
    pages: "5739-5746",
    year: 1988,
    doi: "10.1063/1.455549",
};

/// X2C.
pub(crate) static ILIAS_SAUE_2007: Citation = Citation {
    authors: "M. Iliaš, T. Saue",
    title: "An infinite-order two-component relativistic Hamiltonian by a simple one-step transformation",
    journal: "J. Chem. Phys.",
    volume: "126",
    pages: "064102",
    year: 2007,
    doi: "10.1063/1.2436882",
};

/// X2C.
pub(crate) static KUTZELNIGG_LIU_2005: Citation = Citation {
    authors: "W. Kutzelnigg, W. Liu",
    title: "Quasirelativistic theory equivalent to fully relativistic theory",
    journal: "J. Chem. Phys.",
    volume: "123",
    pages: "241102",
    year: 2005,
    doi: "10.1063/1.2137315",
};

/// The Gaussian nuclear charge distribution.
pub(crate) static VISSCHER_DYALL_1997: Citation = Citation {
    authors: "L. Visscher, K. G. Dyall",
    title: "Dirac-Fock atomic electronic structure calculations using different nuclear charge distributions",
    journal: "At. Data Nucl. Data Tables",
    volume: "67",
    pages: "207-224",
    year: 1997,
    doi: "10.1006/adnd.1997.0751",
};

/// Point-group symmetry.
pub(crate) static JOHANSSON_VERYAZOV_2017: Citation = Citation {
    authors: "M. Johansson, V. Veryazov",
    title: "Automatic procedure for generating symmetry adapted wavefunctions",
    journal: "J. Cheminform.",
    volume: "9",
    pages: "8",
    year: 2017,
    doi: "10.1186/s13321-017-0193-3",
};

/// The SAP guess.
pub(crate) static LEHTOLA_2019: Citation = Citation {
    authors: "S. Lehtola",
    title: "Assessment of initial guesses for self-consistent field calculations. Superposition of atomic potentials: simple yet efficient",
    journal: "J. Chem. Theory Comput.",
    volume: "15",
    pages: "1593-1604",
    year: 2019,
    doi: "10.1021/acs.jctc.8b01089",
};

/// The SAP guess's Gaussian-charge fits.
pub(crate) static LEHTOLA_VISSCHER_ENGEL_2020: Citation = Citation {
    authors: "S. Lehtola, L. Visscher, E. Engel",
    title: "Efficient implementation of the superposition of atomic potentials initial guess for electronic structure calculations in Gaussian basis sets",
    journal: "J. Chem. Phys.",
    volume: "152",
    pages: "144105",
    year: 2020,
    doi: "10.1063/5.0004046",
};

// What each entry point cites once it has run.

/// A context: the integrals, the spherical transformation (a spherical shell with l >= 2), the AIMPs.
pub(crate) fn context(spherical_d_or_higher: bool, core_aimp: bool, embedding_aimp: bool) {
    cite(INTEGRALS, &MCMURCHIE_DAVIDSON_1978);
    if spherical_d_or_higher {
        cite(INTEGRALS, &SCHLEGEL_FRISCH_1995);
    }
    if core_aimp || embedding_aimp {
        cite(MODEL_POTENTIALS, &HUZINAGA_AIMP_1987);
    }
    if embedding_aimp {
        cite(MODEL_POTENTIALS, &BARANDIARAN_SEIJO_1988);
    }
}

/// An ERI walk (`set_eri_data` computes the diagonal ERIs for the Schwarz bounds; the stored tensor).
pub(crate) fn eri(schwarz_screening: bool) {
    cite(INTEGRALS, &NEESE_SHARK_2023);
    if schwarz_screening {
        cite(INTEGRALS, &HASER_AHLRICHS_1989);
    }
}

/// Integral derivatives, one- or two-electron.
pub(crate) fn gradients() {
    cite(GRADIENTS, &HELGAKER_TAYLOR_1992);
}

/// The induced-dipole polarization and its solver; Thole damping, with its default parameter's source when it is used.
pub(crate) fn polarization(damping: &polarization::Damping) {
    cite(POLARIZABLE_EMBEDDING, &OLSEN_AIDAS_KONGSTED_2010);
    cite(POLARIZABLE_EMBEDDING, &HESTENES_STIEFEL_1952);
    if let polarization::Damping::Thole { a } = *damping {
        cite(POLARIZABLE_EMBEDDING, &THOLE_1981);
        if a == polarization::THOLE_DEFAULT_A {
            cite(POLARIZABLE_EMBEDDING, &VAN_DUIJNEN_SWART_1998);
        }
    }
}

/// The effective external field.
pub(crate) fn effective_external_field() {
    cite(POLARIZABLE_EMBEDDING, &LIST_JENSEN_KONGSTED_2016);
}

// The atomic mean-field two-electron spin-orbit integrals (AMFI).
pub(crate) static HESS_MARIAN_WAHLGREN_GROPEN_1996: Citation = Citation {
    authors: "B. A. Heß, C. M. Marian, U. Wahlgren, O. Gropen",
    title: "A mean-field spin-orbit method applicable to correlated wavefunctions",
    journal: "Chem. Phys. Lett.",
    volume: "251",
    pages: "365-371",
    year: 1996,
    doi: "10.1016/0009-2614(96)00119-4",
};

/// AMFI.
pub(crate) fn amfi() {
    cite(RELATIVITY, &HESS_MARIAN_WAHLGREN_GROPEN_1996);
}

/// The X2C Hamiltonian.
pub(crate) fn x2c() {
    cite(RELATIVITY, &ILIAS_SAUE_2007);
    cite(RELATIVITY, &KUTZELNIGG_LIU_2005);
}

/// The X2C derivatives.
pub(crate) static CHENG_GAUSS_2011: Citation = Citation {
    authors: "L. Cheng, J. Gauss",
    title: "Analytic energy gradients for the spin-free exact two-component theory using an exact block diagonalization for the one-electron Dirac Hamiltonian",
    journal: "J. Chem. Phys.",
    volume: "135",
    pages: "084114",
    year: 2011,
    doi: "10.1063/1.3624397",
};

/// The X2C Hamiltonian's derivatives with the decoupling held.
pub(crate) fn x2c_gradient() {
    cite(GRADIENTS, &CHENG_GAUSS_2011);
    cite(GRADIENTS, &HELGAKER_TAYLOR_1992);
}

/// The model is chosen before the call that takes it; the citation, once that call has succeeded.
pub(crate) fn finite_nucleus(model: &basis::charge_carriers::FiniteModel) -> impl FnOnce() + use<> {
    let entry = match model {
        basis::charge_carriers::FiniteModel::VisscherDyall => &VISSCHER_DYALL_1997,
    };
    move || cite(RELATIVITY, entry)
}

/// The SAP screening's integrals: the guess and its fits.
pub(crate) fn sap() {
    cite(START_GUESS, &LEHTOLA_2019);
    cite(START_GUESS, &LEHTOLA_VISSCHER_ENGEL_2020);
}

/// Point-group symmetry.
pub(crate) fn symmetry() {
    cite(SYMMETRY, &JOHANSSON_VERYAZOV_2017);
}
