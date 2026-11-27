// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// A wire enum: a fieldless enum that crosses the C boundary as a stable i32 code. From one list of variants, codes
// and wire names the macro makes the enum, `get_c_int`, `from_c_int` and the `NAME_TABLE` the C interface's
// named-constant query reads, so a new variant is queryable from C and the bindings with no other change.
macro_rules! c_int_enum {
    (
        pub enum $name:ident {
            $( $variant:ident = $disc:literal => $wire:literal ),+ $(,)?
        }
        invalid_error = $invalid:expr;
    ) => {
        #[derive(Clone, Copy, Debug)]
        #[repr(i32)]
        pub enum $name {
            $( $variant = $disc ),+
        }

        impl $name {
            /// The C code.
            #[allow(dead_code)] // made for every wire enum; not every one is passed back to C
            pub fn get_c_int(&self) -> i32 {
                *self as i32
            }

            /// The value of a C code; an error for an unknown code.
            pub fn from_c_int(value: i32) -> Result<Self, crate::error::AedelstenError> {
                match value {
                    $( $disc => Ok($name::$variant), )+
                    _ => Err($invalid(value)),
                }
            }

            /// The C codes by wire name, for the named-constant query of the C interface.
            pub const NAME_TABLE: &'static [(&'static str, i32)] = &[
                $( ($wire, $disc) ),+
            ];
        }
    };
}
pub(crate) use c_int_enum;
