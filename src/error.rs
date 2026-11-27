// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

/// An error of the library: a cache not set or set twice, a wrong code, size or input, or an error of the crate below.
#[derive(Debug)]
pub enum AedelstenError {
    CacheNotReady(String),
    CacheAlreadyDerived(String),
    InvalidIntegralType(i32),
    ExternalFieldMomentUnavailable(String),
    InvalidMatrixDimensions { expected: usize, actual: usize },
    SetupError(basis::SetupError),
    Basis(basis::shell_types::BasisError),
    StringError(String),
    Arena(arrays::scratch::ArenaError),
    Scratch(arrays::scratch::ScratchError),
    InputFileParseError(basis::basis_parser::InputFileParseError),
    Aimp(aimp::AimpParsingError),
    Symmetry(symmetry::SymmetryError),
    AoValues(basis::ao_values::AoValuesError),
}

impl From<basis::SetupError> for AedelstenError {
    fn from(error: basis::SetupError) -> Self {
        AedelstenError::SetupError(error)
    }
}

impl From<basis::shell_types::BasisError> for AedelstenError {
    fn from(error: basis::shell_types::BasisError) -> Self {
        AedelstenError::Basis(error)
    }
}

impl From<String> for AedelstenError {
    fn from(error: String) -> Self {
        AedelstenError::StringError(error)
    }
}

impl From<arrays::scratch::ArenaError> for AedelstenError {
    fn from(error: arrays::scratch::ArenaError) -> Self {
        AedelstenError::Arena(error)
    }
}

impl From<arrays::scratch::ScratchError> for AedelstenError {
    fn from(error: arrays::scratch::ScratchError) -> Self {
        AedelstenError::Scratch(error)
    }
}

impl From<basis::basis_parser::InputFileParseError> for AedelstenError {
    fn from(error: basis::basis_parser::InputFileParseError) -> Self {
        AedelstenError::InputFileParseError(error)
    }
}

impl From<aimp::AimpParsingError> for AedelstenError {
    fn from(error: aimp::AimpParsingError) -> Self {
        AedelstenError::Aimp(error)
    }
}

impl From<basis::ao_values::AoValuesError> for AedelstenError {
    fn from(error: basis::ao_values::AoValuesError) -> Self {
        AedelstenError::AoValues(error)
    }
}

impl From<symmetry::SymmetryError> for AedelstenError {
    fn from(error: symmetry::SymmetryError) -> Self {
        AedelstenError::Symmetry(error)
    }
}

impl std::fmt::Display for AedelstenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AedelstenError::CacheNotReady(msg) => write!(f, "cache not ready: {}", msg),
            AedelstenError::CacheAlreadyDerived(msg) => write!(f, "cache already derived: {}", msg),
            AedelstenError::InvalidIntegralType(t) => write!(f, "invalid integral type: {}", t),
            AedelstenError::ExternalFieldMomentUnavailable(msg) => write!(f, "external field moment unavailable: {}", msg),
            AedelstenError::InvalidMatrixDimensions { expected, actual } => {
                write!(f, "invalid matrix dimensions: expected {}, found {}", expected, actual)
            },
            AedelstenError::SetupError(source) => write!(f, "{}", source),
            AedelstenError::Basis(source) => write!(f, "{}", source),
            AedelstenError::StringError(msg) => write!(f, "{}", msg),
            AedelstenError::Arena(source) => write!(f, "{}", source),
            AedelstenError::Scratch(source) => write!(f, "{}", source),
            AedelstenError::InputFileParseError(source) => write!(f, "{}", source),
            AedelstenError::Aimp(source) => write!(f, "{}", source),
            AedelstenError::Symmetry(source) => write!(f, "{}", source),
            AedelstenError::AoValues(source) => write!(f, "{}", source),
        }
    }
}

impl std::error::Error for AedelstenError {}

impl From<amfi::AmfiError> for AedelstenError {
    fn from(error: amfi::AmfiError) -> Self {
        AedelstenError::StringError(error.to_string())
    }
}

impl From<polarization::PolarizationError> for AedelstenError {
    fn from(error: polarization::PolarizationError) -> Self {
        AedelstenError::StringError(error.to_string())
    }
}
