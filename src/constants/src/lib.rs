// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Physical constants, unit conversions and per-element data (masses, mass numbers, symbols, radii), and the highest
//! angular momentum `L_MAX`.

use std::f64::consts::PI;

// CODATA 2022 recommended values, SI units.
const CONST_AMU_IN_SI_: f64               =  1.66053906892e-27;
const CONST_AU_TIME_IN_SI_: f64           =  2.4188843265864e-17;
const CONST_AU_VELOCITY_IN_SI_: f64       =  2.18769126216e6;
const CONST_AVOGADRO_: f64                =  6.02214076e23;
const CONST_BOHR_RADIUS_IN_SI_: f64       =  5.29177210544e-11;
const CONST_ELECTRON_MASS_IN_SI_: f64     =  9.1093837139e-31;
const CONST_MUON_MASS_IN_SI_: f64         =  1.883531627e-28;
const CONST_BOLTZMANN_: f64               =  1.380649e-23;
const CONST_ELEMENTARY_CHARGE_IN_SI_: f64 =  1.602176634e-19;
const CONST_PLANCK_: f64                  =  6.62607015e-34;
const CONST_DIELECTRIC_IN_SI_: f64        =  8.8541878188e-12;
const CONST_ELECTRON_G_FACTOR_: f64       = -2.00231930436092;
const CONST_BOHR_MAGNETON_IN_SI_: f64     =  9.2740100657e-24;
const CONV_AU_TO_EV_: f64                 = 27.211386245981;
const CONV_AU_TO_CM1_: f64                =  2.1947463136314e5;
const CONV_AU_TO_KJ_: f64                 =  4.3597447222060e-21;
const CONV_AU_TO_HZ_: f64                 =  6.5796839204999e15;
const CONV_AU_TO_T_: f64                  =  2.35051757077e5;
const CONV_CAL_TO_J_: f64                 = 4.184;
const CONST_C_IN_SI_: f64                 = 2.99792458e8;
const CONST_ATM_IN_SI_: f64               = 101325.0;

// Derived from the values above.
const CONST_DIPOLE_IN_SI_: f64            = CONST_ELEMENTARY_CHARGE_IN_SI_ * CONST_BOHR_RADIUS_IN_SI_;
const CONST_MOLAR_GAS_: f64               = CONST_AVOGADRO_ * CONST_BOLTZMANN_;
const CONV_AU_TO_DEBYE_: f64              = CONST_DIPOLE_IN_SI_ * CONST_C_IN_SI_ * 1.0e21;
const CONV_AMU_TO_AU_: f64                = CONST_AMU_IN_SI_ / CONST_ELECTRON_MASS_IN_SI_;
const CONV_AU_TO_KJ_PER_MOLE_: f64        = CONST_AVOGADRO_ * CONV_AU_TO_KJ_;
const CONV_ATM_TO_AU_: f64                = CONST_ATM_IN_SI_ * CONST_BOHR_RADIUS_IN_SI_*CONST_BOHR_RADIUS_IN_SI_*CONST_BOHR_RADIUS_IN_SI_ / CONV_AU_TO_KJ_*1.0e-3;

/// The bohr in angstrom.
pub const AU2ANGSTROM: f64 = CONST_BOHR_RADIUS_IN_SI_*1.0e10;
/// The angstrom in bohr.
pub const ANGSTROM2AU: f64 = 1.0 / AU2ANGSTROM;
/// The standard atmosphere in atomic units of pressure (hartree / bohr^3).
pub const ATM2AU: f64      = CONV_ATM_TO_AU_;
/// The standard atmosphere in Pa.
pub const ATM2PA: f64      = CONST_ATM_IN_SI_;
/// The unified atomic mass unit (dalton) in kg.
pub const AU2KG: f64       = CONST_AMU_IN_SI_;
/// The hartree in cm^-1.
pub const AU2CM: f64       = CONV_AU_TO_CM1_;
/// The hartree in eV.
pub const AU2EV: f64       = CONV_AU_TO_EV_;
/// The atomic unit of time in fs.
pub const AU2FS: f64       = CONST_AU_TIME_IN_SI_*1.0e15;
/// The hartree in Hz.
pub const AU2HZ: f64       = CONV_AU_TO_HZ_;
/// The hartree in kcal/mol.
pub const AU2KCALMOL: f64  = CONV_AU_TO_KJ_PER_MOLE_ / CONV_CAL_TO_J_;
/// The hartree in kJ.
pub const AU2KJ: f64       = CONV_AU_TO_KJ_;
/// The hartree in kJ/mol.
pub const AU2KJMOL: f64    = CONV_AU_TO_KJ_PER_MOLE_;
/// The atomic unit of force (hartree / bohr) in kJ/(mol nm).
pub const AU2KJMOLNM: f64  = 1.0e-9*CONV_AU_TO_KJ_PER_MOLE_ / CONST_BOHR_RADIUS_IN_SI_;
/// The atomic unit of force (hartree / bohr) in N.
pub const AU2N: f64        = CONV_AU_TO_KJ_ / CONST_BOHR_RADIUS_IN_SI_ * 1.0e3;
/// The atomic unit of magnetic flux density in T.
pub const AU2T: f64        = CONV_AU_TO_T_;
/// The speed of light in atomic units.
pub const C_IN_AU: f64     = CONST_C_IN_SI_ / CONST_AU_VELOCITY_IN_SI_;
/// The thermochemical calorie in J.
pub const CAL_TO_J: f64    = CONV_CAL_TO_J_;
/// The speed of light in m/s.
pub const C_LIGHT: f64     = CONST_C_IN_SI_;
/// The speed of light in cm/s.
pub const CM_S: f64        = CONST_C_IN_SI_*1.0e2;
/// The atomic unit of electric dipole moment in debye.
pub const DEBYE: f64       = CONV_AU_TO_DEBYE_;
/// The degree in radians.
pub const DEG2RAD: f64     = PI/180.0;
/// The vacuum electric permittivity in F/m.
pub const DIEL: f64        = CONST_DIELECTRIC_IN_SI_;
/// The elementary charge in C.
pub const ELCHARGE: f64    = CONST_ELEMENTARY_CHARGE_IN_SI_;
/// The electron mass in kg.
pub const ELMASS: f64      = CONST_ELECTRON_MASS_IN_SI_;
/// The electron g-factor.
pub const G_ELECTRON: f64  = CONST_ELECTRON_G_FACTOR_;
/// The Planck constant in J s.
pub const H_PLANCK: f64    = CONST_PLANCK_;
/// The Boltzmann constant in J/K.
pub const K_BOLTZMANN: f64 = CONST_BOLTZMANN_;
/// The Bohr magneton in J/T.
pub const M_BOHR: f64      = CONST_BOHR_MAGNETON_IN_SI_;
/// The muon mass in electron masses.
pub const MU2ELMASS: f64   = CONST_MUON_MASS_IN_SI_ / CONST_ELECTRON_MASS_IN_SI_;
/// The Bohr radius in m.
pub const R_BOHR: f64      = CONST_BOHR_RADIUS_IN_SI_;
/// The molar gas constant in J/(mol K).
pub const R_GAS: f64       = CONST_MOLAR_GAS_;
/// The Avogadro constant in mol^-1.
pub const R_NAVO: f64      = CONST_AVOGADRO_;
/// The unified atomic mass unit in electron masses.
pub const U_TO_AU: f64     = CONV_AMU_TO_AU_;



// `L_MAX`, the highest angular momentum of a shell: the build-time environment variable `L_MAX`, 6 if unset
// (`build.rs`). A basis with a higher shell is refused at setup.
include!(concat!(env!("OUT_DIR"), "/lmax.rs"));

/// The highest order of nuclear derivatives the integral code is sized for (2: Hessians). A derivative raises the
/// angular momentum a quartet's Hermite expansion reaches by one, never a shell's own, so only what runs over a
/// quartet's total angular momentum (the Boys function's orders) is sized beyond `4 L_MAX`.
pub const L_DERIVATIVE: usize = 2;

/// The standard atomic weight of each element in u (IUPAC), for elements without one the mass number of its most
/// stable isotope; indexed by atomic number, index 0 unused.
pub const AVERAGE_MASSES: [f64; 119] = [
    0.0, // dummy so index = atomic number
    1.008,      // 1  H
    4.002602,   // 2  He
    6.94,       // 3  Li
    9.0121831,  // 4  Be
    10.81,      // 5  B
    12.011,     // 6  C
    14.007,     // 7  N
    15.999,     // 8  O
    18.998403163,// 9  F
    20.1797,    // 10 Ne
    22.98976928,// 11 Na
    24.305,     // 12 Mg
    26.9815385, // 13 Al
    28.085,     // 14 Si
    30.973761998,//15 P
    32.06,      // 16 S
    35.45,      // 17 Cl
    39.948,     // 18 Ar
    39.0983,    // 19 K
    40.078,     // 20 Ca
    44.955908,  // 21 Sc
    47.867,     // 22 Ti
    50.9415,    // 23 V
    51.9961,    // 24 Cr
    54.938044,  // 25 Mn
    55.845,     // 26 Fe
    58.933194,  // 27 Co
    58.6934,    // 28 Ni
    63.546,     // 29 Cu
    65.38,      // 30 Zn
    69.723,     // 31 Ga
    72.630,     // 32 Ge
    74.921595,  // 33 As
    78.971,     // 34 Se
    79.904,     // 35 Br
    83.798,     // 36 Kr
    85.4678,    // 37 Rb
    87.62,      // 38 Sr
    88.90584,   // 39 Y
    91.224,     // 40 Zr
    92.90637,   // 41 Nb
    95.95,      // 42 Mo
    98.0,       // 43 Tc
    101.07,     // 44 Ru
    102.90550,  // 45 Rh
    106.42,     // 46 Pd
    107.8682,   // 47 Ag
    112.414,    // 48 Cd
    114.818,    // 49 In
    118.710,    // 50 Sn
    121.760,    // 51 Sb
    127.60,     // 52 Te
    126.90447,  // 53 I
    131.293,    // 54 Xe
    132.90545196,//55 Cs
    137.327,    // 56 Ba
    138.90547,  // 57 La
    140.116,    // 58 Ce
    140.90766,  // 59 Pr
    144.242,    // 60 Nd
    145.0,      // 61 Pm
    150.36,     // 62 Sm
    151.964,    // 63 Eu
    157.25,     // 64 Gd
    158.92535,  // 65 Tb
    162.500,    // 66 Dy
    164.93033,  // 67 Ho
    167.259,    // 68 Er
    168.93422,  // 69 Tm
    173.045,    // 70 Yb
    174.9668,   // 71 Lu
    178.49,     // 72 Hf
    180.94788,  // 73 Ta
    183.84,     // 74 W
    186.207,    // 75 Re
    190.23,     // 76 Os
    192.217,    // 77 Ir
    195.084,    // 78 Pt
    196.966569, // 79 Au
    200.592,    // 80 Hg
    204.38,     // 81 Tl
    207.2,      // 82 Pb
    208.98040,  // 83 Bi
    209.0,      // 84 Po
    210.0,      // 85 At
    222.0,      // 86 Rn
    223.0,      // 87 Fr
    226.0,      // 88 Ra
    227.0,      // 89 Ac
    232.0377,   // 90 Th
    231.03588,  // 91 Pa
    238.02891,  // 92 U
    237.0,      // 93 Np
    244.0,      // 94 Pu
    243.0,      // 95 Am
    247.0,      // 96 Cm
    247.0,      // 97 Bk
    251.0,      // 98 Cf
    252.0,      // 99 Es
    257.0,      //100 Fm
    258.0,      //101 Md
    259.0,      //102 No
    266.0,      //103 Lr
    267.0,      //104 Rf
    270.0,      //105 Db
    271.0,      //106 Sg
    270.0,      //107 Bh
    277.0,      //108 Hs
    278.0,      //109 Mt
    281.0,      //110 Ds
    282.0,      //111 Rg
    285.0,      //112 Cn
    286.0,      //113 Nh
    289.0,      //114 Fl
    290.0,      //115 Mc
    293.0,      //116 Lv
    294.0,      //117 Ts
    294.0,      //118 Og
];

/// The mass number of each element's most abundant isotope (the lighter one on a tie), for elements 95-118 of its
/// most stable one; indexed by atomic number, index 0 unused. The `A` of the finite-nucleus models (L. Visscher and
/// K. G. Dyall, At. Data Nucl. Data Tables 67, 207 (1997)). Abundances from J. S. Coursey et al., "Atomic Weights and
/// Isotopic Compositions with Relative Atomic Masses", NIST, version 4.1.
pub const MASS_NUMBERS: [u16; 119] = [
      0, // dummy so index = atomic number
      1,   4,   7,   9,  11,  12,  14,  16,  19,  20, // 1-10
     23,  24,  27,  28,  31,  32,  35,  40,  39,  40, // 11-20
     45,  48,  51,  52,  55,  56,  59,  58,  63,  64, // 21-30
     69,  74,  75,  80,  79,  84,  85,  88,  89,  90, // 31-40
     93,  98,  98, 102, 103, 106, 107, 114, 115, 120, // 41-50
    121, 130, 127, 132, 133, 138, 139, 140, 141, 142, // 51-60
    145, 152, 153, 158, 159, 164, 165, 166, 169, 174, // 61-70
    175, 180, 181, 184, 187, 192, 193, 195, 197, 202, // 71-80
    205, 208, 209, 209, 210, 222, 223, 226, 227, 232, // 81-90
    231, 238, 237, 244, 243, 247, 247, 251, 252, 257, // 91-100
    258, 259, 266, 267, 268, 269, 270, 269, 278, 281, // 101-110
    283, 285, 286, 289, 290, 293, 294, 294, // 111-118
];

/// The element symbols, indexed by atomic number, index 0 empty.
pub const ELEMENT_SYMBOLS: [&str; 119] = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
];

/// The Bragg-Slater atomic radii in bohr, indexed by atomic number, index 0 a placeholder. J. C. Slater, J. Chem.
/// Phys. 41, 3199 (1964), completed for the noble gases and the rare earths after A. D. Becke, J. Chem. Phys. 88, 2547
/// (1988); the table of PySCF 2.11.0 (`pyscf.dft.radi.BRAGG_RADII`).
pub const BRAGG_RADII: [f64; 119] = [
    3.779450359403999,      //0 dummy
    0.6614041435977716,      //1 H
    2.645616574391086,      //2 He
    2.7401028806193395,      //3 Li
    1.984212430793315,      //4 Be
    1.6062672058803025,      //5 B
    1.322808287195543,      //6 C
    1.2283219809672903,      //7 N
    1.133835674739037,      //8 O
    0.9448630622825309,      //9 F
    2.8345891868475928,      //10 Ne
    3.4015070242171115,      //11 Na
    2.8345891868475928,      //12 Mg
    2.3621576557063273,      //13 Al
    2.0786987370215684,      //14 Si
    1.8897261245650618,      //15 P
    1.8897261245650618,      //16 S
    1.8897261245650618,      //17 Cl
    3.4015070242171115,      //18 Ar
    4.157397474043137,      //19 K
    3.4015070242171115,      //20 Ca
    3.0235617993040993,      //21 Sc
    2.645616574391086,      //22 Ti
    2.551130268162834,      //23 V
    2.645616574391086,      //24 Cr
    2.645616574391086,      //25 Mn
    2.645616574391086,      //26 Fe
    2.551130268162834,      //27 Co
    2.551130268162834,      //28 Ni
    2.551130268162834,      //29 Cu
    2.551130268162834,      //30 Zn
    2.4566439619345806,      //31 Ga
    2.3621576557063273,      //32 Ge
    2.1731850432498208,      //33 As
    2.1731850432498208,      //34 Se
    2.1731850432498208,      //35 Br
    3.590479636673617,      //36 Kr
    4.440856392727896,      //37 Rb
    3.7794522491301237,      //38 Sr
    3.4015070242171115,      //39 Y
    2.929075493075846,      //40 Zr
    2.7401028806193395,      //41 Nb
    2.7401028806193395,      //42 Mo
    2.551130268162834,      //43 Tc
    2.4566439619345806,      //44 Ru
    2.551130268162834,      //45 Rh
    2.645616574391086,      //46 Pd
    3.0235617993040993,      //47 Ag
    2.929075493075846,      //48 Cd
    2.929075493075846,      //49 In
    2.7401028806193395,      //50 Sn
    2.7401028806193395,      //51 Sb
    2.645616574391086,      //52 Te
    2.645616574391086,      //53 I
    3.96842486158663,      //54 Xe
    4.913287923869161,      //55 Cs
    4.062911167814883,      //56 Ba
    3.6849659429018704,      //57 La
    3.4959933304453648,      //58 Ce
    3.4959933304453648,      //59 Pr
    3.4959933304453648,      //60 Nd
    3.4959933304453648,      //61 Pm
    3.4959933304453648,      //62 Sm
    3.4959933304453648,      //63 Eu
    3.4015070242171115,      //64 Gd
    3.307020717988858,      //65 Tb
    3.307020717988858,      //66 Dy
    3.307020717988858,      //67 Ho
    3.307020717988858,      //68 Er
    3.307020717988858,      //69 Tm
    3.307020717988858,      //70 Yb
    3.307020717988858,      //71 Lu
    2.929075493075846,      //72 Hf
    2.7401028806193395,      //73 Ta
    2.551130268162834,      //74 W
    2.551130268162834,      //75 Re
    2.4566439619345806,      //76 Os
    2.551130268162834,      //77 Ir
    2.551130268162834,      //78 Pt
    2.551130268162834,      //79 Au
    2.8345891868475928,      //80 Hg
    3.590479636673617,      //81 Tl
    3.4015070242171115,      //82 Pb
    3.0235617993040993,      //83 Bi
    3.590479636673617,      //84 Po
    2.7401028806193395,      //85 At
    3.96842486158663,      //86 Rn
    3.4015070242171115,      //87 Fr
    4.062911167814883,      //88 Ra
    3.6849659429018704,      //89 Ac
    3.4015070242171115,      //90 Th
    3.4015070242171115,      //91 Pa
    3.307020717988858,      //92 U
    3.307020717988858,      //93 Np
    3.307020717988858,      //94 Pu
    3.307020717988858,      //95 Am
    3.307020717988858,      //96 Cm
    3.307020717988858,      //97 Bk
    3.307020717988858,      //98 Cf
    3.307020717988858,      //99 Es
    3.307020717988858,      //100 Fm
    3.307020717988858,      //101 Md
    3.307020717988858,      //102 No
    3.307020717988858,      //103 Lr
    3.307020717988858,      //104 Rf
    3.307020717988858,      //105 Db
    3.307020717988858,      //106 Sg
    3.307020717988858,      //107 Bh
    3.307020717988858,      //108 Hs
    3.307020717988858,      //109 Mt
    3.307020717988858,      //110 Ds
    3.307020717988858,      //111 Rg
    3.307020717988858,      //112 Cn
    3.307020717988858,      //113 Nh
    3.307020717988858,      //114 Fl
    3.307020717988858,      //115 Mc
    3.307020717988858,      //116 Lv
    3.307020717988858,      //117 Ts
    3.307020717988858,      //118 Og
];

