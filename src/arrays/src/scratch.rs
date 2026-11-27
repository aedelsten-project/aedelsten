// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The scratch: one buffer, allocated once at the size the caller chooses, from which computations take their
//! temporary memory through bump-allocating arenas (`Arena`), and its high-water mark.

use std::cell::Cell;
use std::mem::{align_of, size_of};
use std::slice;

/// The scratch buffer, zeroed and committed lazily: a page becomes resident when an arena first hands it out, so the
/// resident size is about the high-water mark, not the buffer's size.
#[derive(Debug)]
pub struct Scratch {
    // Words, so that the buffer starts 8-byte aligned: an arena of f64s never pads.
    memory: Box<[u64]>,
    // High-water mark: the largest number of bytes (from the start of `memory`) any arena has handed out.
    peak: Cell<usize>,
}

/// The unit of a size in bytes.
pub enum ByteUnit {
    /// 1024 bytes.
    KiB,
    /// 1024^2 bytes.
    MiB,
    /// 1024^3 bytes.
    GiB,
}

const AEDELSTEN_BYTE_UNIT_GIB: i32 = 0;
const AEDELSTEN_BYTE_UNIT_MIB: i32 = 1;
const AEDELSTEN_BYTE_UNIT_KIB: i32 = 2;
impl ByteUnit {
    /// The unit from its value in the C ABI.
    pub fn from_c_int(value: i32) -> Result<Self, String> {
        match value {
            AEDELSTEN_BYTE_UNIT_GIB => Ok(ByteUnit::GiB),
            AEDELSTEN_BYTE_UNIT_MIB => Ok(ByteUnit::MiB),
            AEDELSTEN_BYTE_UNIT_KIB => Ok(ByteUnit::KiB),
            _ => Err(format!("Invalid byte unit: {}", value)),
        }
    }

    /// The bytes in one unit.
    pub fn bytes_per_unit(&self) -> usize {
        match self {
            ByteUnit::KiB => 1024,
            ByteUnit::MiB => 1024 * 1024,
            ByteUnit::GiB => 1024 * 1024 * 1024,
        }
    }

    /// The unit's value in the C ABI.
    pub fn get_c_int(&self) -> i32 {
        match self {
            ByteUnit::GiB => AEDELSTEN_BYTE_UNIT_GIB,
            ByteUnit::MiB => AEDELSTEN_BYTE_UNIT_MIB,
            ByteUnit::KiB => AEDELSTEN_BYTE_UNIT_KIB,
        }
    }

    /// Each unit's name and value for the C ABI's named-constant query.
    pub const NAME_TABLE: &'static [(&'static str, i32)] = &[
        ("byte_unit_kib", AEDELSTEN_BYTE_UNIT_KIB),
        ("byte_unit_mib", AEDELSTEN_BYTE_UNIT_MIB),
        ("byte_unit_gib", AEDELSTEN_BYTE_UNIT_GIB),
    ];
}

/// Why a scratch buffer could not be built (an exhausted one is an `ArenaError`).
#[derive(Debug)]
pub enum ScratchError {
    /// The size in bytes overflows `usize`.
    SizeOverflow,
    /// The allocator could not provide `bytes`.
    AllocationFailed { bytes: usize },
}

impl std::fmt::Display for ScratchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScratchError::SizeOverflow => write!(f, "requested scratch size overflows usize"),
            ScratchError::AllocationFailed { bytes } => write!(f, "failed to allocate a {} byte scratch", bytes),
        }
    }
}

impl std::error::Error for ScratchError {}

impl Scratch {
    /// A scratch of `size` `unit`s.
    pub fn new(size: usize, unit: ByteUnit) -> Result<Self, ScratchError> {
        let size_bytes = size.checked_mul(unit.bytes_per_unit()).ok_or(ScratchError::SizeOverflow)?;
        Self::from_bytes(size_bytes)
    }

    /// The buffer's size in bytes.
    pub fn byte_len(&self) -> usize {
        self.memory.len() * size_of::<u64>()
    }

    /// The bytes `work` takes from an arena of this scratch when the space left never limits it: run until it fits
    /// with at most half the buffer taken, the buffer replaced by one twice as large (at least the failed allocation's
    /// need, `used + needed`) after each run that fails or takes more. Work that takes more of a larger scratch, up to
    /// a cap, and nearly all of the space left below it, has reached its cap in a run that took at most half; in a
    /// scratch of its high-water mark it then runs as in any larger one. The scratch keeps its last size, so measuring
    /// several computations in turn grows it once, to twice the largest.
    pub fn measure(&mut self, mut work: impl FnMut(&mut Arena<'_>) -> Result<(), ArenaError>) -> Result<usize, ScratchError> {
        loop {
            self.reset_peak();
            let bytes = match work(&mut self.get_arena()) {
                Ok(()) if 2 * self.peak_bytes() <= self.byte_len() => return Ok(self.peak_bytes()),
                Ok(()) => 2 * self.byte_len(),
                Err(ArenaError::OutOfMemory { needed, used, .. }) => {
                    used.checked_add(needed).ok_or(ScratchError::SizeOverflow)?.max(2 * self.byte_len())
                }
                Err(ArenaError::SizeOverflow) => return Err(ScratchError::SizeOverflow),
            };
            *self = Self::from_bytes(bytes)?;
        }
    }

    /// Replaces the buffer by a new one of `size` `unit`s (zeroed, committed lazily) and resets the high-water
    /// mark. On failure the old buffer is kept. Growing or shrinking alike; nothing may hold an arena meanwhile, which
    /// `&mut self` guarantees.
    pub fn resize(&mut self, size: usize, unit: ByteUnit) -> Result<(), ScratchError> {
        let size_bytes = size.checked_mul(unit.bytes_per_unit()).ok_or(ScratchError::SizeOverflow)?;
        *self = Self::from_bytes(size_bytes)?;
        Ok(())
    }

    fn from_bytes(bytes: usize) -> Result<Self, ScratchError> {
        // A zeroed allocation, which takes fresh zero pages without writing them (committed when first handed out),
        // through the raw allocator, which returns null on failure rather than aborting.
        let words = bytes.div_ceil(size_of::<u64>());
        if words == 0 {
            return Ok(Self { memory: Box::new([]), peak: Cell::new(0) });
        }
        let layout = std::alloc::Layout::array::<u64>(words).map_err(|_| ScratchError::SizeOverflow)?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut u64;
        if ptr.is_null() {
            return Err(ScratchError::AllocationFailed { bytes });
        }
        // The layout is exactly the one `Box<[u64]>` deallocates with (`Layout::array::<u64>(words)`).
        let memory = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, words)) };
        Ok(Self { memory, peak: Cell::new(0) })
    }

    /// The most bytes any arena of this scratch has handed out since it was built (or since `reset_peak`),
    /// counted from the start of the buffer, alignment padding included: what the computations so far actually
    /// needed, and (with the lazily committed buffer) about what is resident.
    pub fn peak_bytes(&self) -> usize {
        self.peak.get()
    }

    /// Sets the high-water mark to 0.
    pub fn reset_peak(&self) {
        self.peak.set(0);
    }

    /// An arena over the whole buffer.
    pub fn get_arena(&mut self) -> Arena<'_> {
        let origin = self.memory.as_ptr() as usize;
        let bytes = self.byte_len();
        let Scratch { memory, peak, .. } = self;
        // The words as bytes: u8 has no alignment or validity requirement, and the borrow of `memory` is held for
        // the arena's lifetime.
        let memory = unsafe { slice::from_raw_parts_mut(memory.as_mut_ptr() as *mut u8, bytes) };
        Arena { rem: Cell::new(memory), origin, peak }
    }
}

/// What a computation takes its arena from: the whole `Scratch`, or a `SubScratch`, the rest of one after a
/// caller has taken buffers off its front (`Arena::into_sub_scratch`), so that the caller's buffers stay valid while
/// the computation runs in the remainder.
pub trait ScratchSource {
    /// An arena over the whole of it.
    fn get_arena(&mut self) -> Arena<'_>;
}

impl ScratchSource for Scratch {
    fn get_arena(&mut self) -> Arena<'_> {
        Scratch::get_arena(self)
    }
}

impl ScratchSource for SubScratch<'_> {
    fn get_arena(&mut self) -> Arena<'_> {
        SubScratch::get_arena(self)
    }
}

/// The rest of a scratch after the buffers taken off its front, its high-water mark the scratch's.
pub struct SubScratch<'a> {
    rem: &'a mut [u8],
    origin: usize,
    peak: &'a Cell<usize>,
}

impl<'a> SubScratch<'a> {
    /// An arena over the whole of the rest.
    pub fn get_arena(&mut self) -> Arena<'_> {
        Arena { rem: Cell::new(&mut *self.rem), origin: self.origin, peak: self.peak }
    }
}

/// Why an arena could not hand out an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// `used` bytes were already handed out (from the start of the buffer) and `available` were left when an
    /// allocation of `needed` bytes failed: the buffer holds `used + available`, and the computation needs at least
    /// `used + needed`.
    OutOfMemory { needed: usize, available: usize, used: usize },
    /// The size in bytes overflows `usize`.
    SizeOverflow,
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArenaError::OutOfMemory { needed, available, used } => {
                let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
                write!(
                    f,
                    "scratch of {:.2} MiB too small: needs at least {:.2} MiB ({} bytes); pass a larger size to set_scratch",
                    mib(used + available),
                    mib(used + needed),
                    used + needed
                )
            }
            ArenaError::SizeOverflow => write!(f, "requested allocation size overflows usize"),
        }
    }
}

impl std::error::Error for ArenaError {}

/// The product of `dims`, or `SizeOverflow` when it overflows `usize`: the element count of an allocation.
pub fn checked_product(dims: &[usize]) -> Result<usize, ArenaError> {
    dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or(ArenaError::SizeOverflow)
}

/// A bump allocator over a scratch buffer: each allocation is the next disjoint region of it, valid for the arena's
/// lifetime. Nothing is freed one by one: a fresh arena starts over, and a scope (`Arena::scope`) frees what was
/// allocated in it. Allocation takes `&self`, so one `&Arena` serves any number of allocations.
pub struct Arena<'a> {
    rem: Cell<&'a mut [u8]>,
    // Start of the scratch's buffer, and its high-water mark (see `Scratch::peak_bytes`).
    origin: usize,
    peak: &'a Cell<usize>,
}

impl<'a> Arena<'a> {
    /// `n` values of `T`, aligned, uninitialised (whatever the buffer held), or `OutOfMemory` with what was needed.
    pub fn alloc<T>(&self, n: usize) -> Result<&'a mut [T], ArenaError> {
        // ZST: nothing to take from the arena, and nothing to update either.
        if size_of::<T>() == 0 {
            return Ok(unsafe {
                slice::from_raw_parts_mut(std::ptr::NonNull::<T>::dangling().as_ptr(), n)
            });
        }

        let rem = self.rem.take();

        let align = align_of::<T>();
        let bytes = match n.checked_mul(size_of::<T>()) {
            Some(bytes) => bytes,
            None => {
                // A failed allocation leaves the arena as it was.
                self.rem.set(rem);
                return Err(ArenaError::SizeOverflow);
            }
        };

        // The padding that aligns the start for T.
        let addr = rem.as_ptr() as usize;
        let pad = (align - (addr % align)) % align;

        if pad + bytes > rem.len() {
            let needed = pad + bytes;
            let available = rem.len();
            let used = addr - self.origin;
            self.rem.set(rem);
            return Err(ArenaError::OutOfMemory { needed, available, used });
        }

        let (_padding, after_pad) = rem.split_at_mut(pad);
        let (chunk, rest) = after_pad.split_at_mut(bytes);

        debug_assert_eq!(chunk.as_ptr() as usize % align, 0);
        debug_assert_eq!(chunk.len() % size_of::<T>(), 0);

        let ptr = chunk.as_mut_ptr() as *mut T;
        let out = unsafe { slice::from_raw_parts_mut(ptr, n) };

        let used = rest.as_ptr() as usize - self.origin;
        if used > self.peak.get() {
            self.peak.set(used);
        }
        self.rem.set(rest);
        Ok(out)
    }

    /// `n` values `init(0)`, `init(1)`, ..., built in place: for element types that are not plain numbers (views,
    /// tuples of them), which `alloc` would hand out uninitialised. Nothing is dropped when the arena is reset, so
    /// `T` should own nothing (a view, not a `Vec`).
    pub fn alloc_with<T>(&self, n: usize, mut init: impl FnMut(usize) -> T) -> Result<&'a mut [T], ArenaError> {
        let slots = self.alloc::<std::mem::MaybeUninit<T>>(n)?;
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.write(init(i));
        }
        // Every slot was written above; MaybeUninit<T> has T's layout.
        Ok(unsafe { &mut *(slots as *mut [std::mem::MaybeUninit<T>] as *mut [T]) })
    }

    /// Bytes handed out so far, counted from the start of the scratch's buffer (alignment padding included).
    pub fn used(&self) -> usize {
        let rem = self.rem.take();
        let used = rem.as_ptr() as usize - self.origin;
        self.rem.set(rem);
        used
    }

    /// Bytes still free: what an allocation can take (less any alignment padding).
    pub fn available(&self) -> usize {
        let rem = self.rem.take();
        let available = rem.len();
        self.rem.set(rem);
        available
    }

    /// `n` values `T::default()`.
    pub fn alloc_zeroed<T>(&self, n: usize) -> Result<&'a mut [T], ArenaError>
    where
        T: Default + Clone,
    {
        let slice = self.alloc::<T>(n)?;
        slice.fill(T::default());
        Ok(slice)
    }

    /// Runs `work` with an arena over the space this one has left; everything `work` allocates is free again when it
    /// returns, while what was allocated here before stays valid (inside `work` too). Scopes nest. The lifetimes keep
    /// the scope's allocations from escaping `work`, and `&mut self` keeps this arena from allocating while `work` runs
    /// (the two would overlap). The high-water mark counts the scope's allocations as any others.
    ///
    /// A scope's allocation cannot escape it:
    /// ```compile_fail
    /// use arrays::scratch::{Scratch, ByteUnit};
    /// let mut ws = Scratch::new(1, ByteUnit::KiB).unwrap();
    /// let mut arena = ws.get_arena();
    /// let escaped = arena.scope(|scratch| scratch.alloc::<f64>(1).unwrap());
    /// ```
    /// and the enclosing arena cannot allocate inside it:
    /// ```compile_fail
    /// use arrays::scratch::{Scratch, ByteUnit};
    /// let mut ws = Scratch::new(1, ByteUnit::KiB).unwrap();
    /// let mut arena = ws.get_arena();
    /// arena.scope(|scratch| { let _a = scratch.alloc::<f64>(1); let _b = arena.alloc::<f64>(1); });
    /// ```
    pub fn scope<R>(&mut self, work: impl for<'s> FnOnce(&mut Arena<'s>) -> R) -> R {
        let mut inner = Arena { rem: Cell::new(&mut **self.rem.get_mut()), origin: self.origin, peak: self.peak };
        work(&mut inner)
    }

    /// `scope`, and the bytes it reached beyond what was taken before it (its own high-water mark, nested scopes and
    /// alignment padding included): for measuring which piece of work sets a peak. The scratch's high-water mark is
    /// kept as `scope` keeps it.
    pub fn scope_extent<R>(&mut self, work: impl for<'s> FnOnce(&mut Arena<'s>) -> R) -> (R, usize) {
        let start = self.used();
        let outer = self.peak.get();
        self.peak.set(start);
        let result = self.scope(work);
        let reached = self.peak.get();
        self.peak.set(outer.max(reached));
        (result, reached - start)
    }

    /// The space this arena has left, as a scratch of its own; what it handed out stays valid.
    pub fn into_sub_scratch(self) -> SubScratch<'a> {
        SubScratch { rem: self.rem.into_inner(), origin: self.origin, peak: self.peak }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_measurement_grows_the_scratch_until_the_work_fits_and_returns_its_peak() {
        let mut ws = Scratch::new(1, ByteUnit::KiB).unwrap();
        // 10 kB, then 30 kB in a scope after 4 kB more, the scope's taken back: a peak of 44 kB.
        let work = |arena: &mut Arena<'_>| {
            arena.alloc::<u8>(10_000)?;
            arena.scope(|inner| inner.alloc::<u8>(4_000).map(|_| ()))?;
            arena.alloc::<u8>(4_000)?;
            arena.scope(|inner| inner.alloc::<u8>(30_000).map(|_| ()))
        };
        assert_eq!(ws.measure(work).unwrap(), 44_000);
        assert!(ws.byte_len() >= 88_000);
        // Large enough already: the same peak without growing.
        let size = ws.byte_len();
        assert_eq!(ws.measure(work).unwrap(), 44_000);
        assert_eq!(ws.byte_len(), size);
        assert!(matches!(ws.measure(|arena| arena.alloc::<f64>(usize::MAX).map(|_| ())), Err(ScratchError::SizeOverflow)));

        // A kernel that takes all the space left, up to 300 kB: measured at its cap.
        let mut ws = Scratch::new(1, ByteUnit::KiB).unwrap();
        assert_eq!(ws.measure(|arena| arena.alloc::<u8>(arena.available().min(300_000)).map(|_| ())).unwrap(), 300_000);
    }

    #[test]
    fn a_scope_extent_is_what_the_scope_reached_beyond_its_start() {
        let mut ws = Scratch::new(64, ByteUnit::KiB).unwrap();
        let mut arena = ws.get_arena();
        arena.alloc::<u8>(5_000).unwrap();
        arena.scope(|inner| inner.alloc::<u8>(20_000).map(|_| ())).unwrap();
        // Nested scopes count; a smaller scope after a larger one is measured on its own.
        let (_, extent) = arena.scope_extent(|inner| {
            inner.alloc::<u8>(1_000).unwrap();
            inner.scope(|nested| nested.alloc::<u8>(3_000).map(|_| ())).unwrap();
        });
        assert_eq!(extent, 4_000);
        drop(arena);
        assert_eq!(ws.peak_bytes(), 25_000, "the scratch's high-water mark is kept");
    }

    #[test]
    fn the_peak_is_the_largest_extent_any_arena_reached() {
        let mut ws = Scratch::new(64, ByteUnit::KiB).unwrap();
        assert!(ws.memory.iter().all(|&w| w == 0), "a new scratch is zeroed");
        {
            let arena = ws.get_arena();
            arena.alloc::<f64>(100).unwrap();
            arena.alloc::<u8>(3).unwrap();
        }
        assert_eq!(ws.peak_bytes(), 803);
        {
            // A later, smaller use leaves the peak; alignment padding counts.
            let arena = ws.get_arena();
            arena.alloc::<u8>(1).unwrap();
            arena.alloc::<f64>(1).unwrap();
        }
        assert_eq!(ws.peak_bytes(), 803);
        {
            let arena = ws.get_arena();
            arena.alloc::<f64>(200).unwrap();
            let mut daughter = arena.into_sub_scratch();
            daughter.get_arena().alloc::<f64>(50).unwrap();
        }
        assert_eq!(ws.peak_bytes(), 2000);
        assert!(ws.get_arena().alloc::<u8>(64 * 1024 + 1).is_err());
        assert_eq!(ws.peak_bytes(), 2000, "a failed allocation does not move the peak");
        ws.reset_peak();
        assert_eq!(ws.peak_bytes(), 0);

        // Resizing replaces the buffer: the new size holds, and a request beyond it fails.
        ws.resize(2, ByteUnit::KiB).unwrap();
        assert_eq!(ws.byte_len(), 2048);
        assert!(ws.get_arena().alloc::<u8>(2049).is_err());
        ws.get_arena().alloc::<f64>(256).unwrap();
        assert_eq!(ws.peak_bytes(), 2048);
    }

    #[test]
    fn running_out_of_space_names_the_size_needed() {
        let mut ws = Scratch::new(1, ByteUnit::KiB).unwrap();
        let arena = ws.get_arena();
        arena.alloc::<f64>(100).unwrap();
        let error = arena.alloc::<f64>(100).unwrap_err();
        assert_eq!(error, ArenaError::OutOfMemory { needed: 800, available: 224, used: 800 });
        assert!(error.to_string().contains("needs at least 0.00 MiB (1600 bytes)"), "{error}");
        // The failed allocation took nothing.
        assert_eq!(arena.used(), 800);
        assert_eq!(arena.available(), 224);
        arena.alloc::<f64>(28).unwrap();
        assert_eq!(arena.available(), 0);
    }

    #[test]
    fn available_is_what_is_left_and_a_scope_sees_the_same() {
        let mut ws = Scratch::new(1, ByteUnit::KiB).unwrap();
        let mut arena = ws.get_arena();
        assert_eq!(arena.available(), 1024);
        arena.alloc::<f64>(16).unwrap();
        assert_eq!(arena.used() + arena.available(), 1024);
        let before = arena.available();
        arena.scope(|scratch| {
            assert_eq!(scratch.available(), before);
            scratch.alloc::<f64>(10).unwrap();
            assert_eq!(scratch.available(), before - 80);
        });
        assert_eq!(arena.available(), before);
    }

    #[test]
    fn a_scope_frees_its_allocations_and_keeps_the_ones_before_it() {
        let mut ws = Scratch::new(64, ByteUnit::KiB).unwrap();
        let mut arena = ws.get_arena();
        let row = arena.alloc::<f64>(10).unwrap();
        row.fill(1.0);
        for quartet in 0..3 {
            let start = arena.used();
            let sum = arena.scope(|scratch| {
                let a = scratch.alloc::<f64>(100 * (quartet + 1)).unwrap();
                a.fill(2.0);
                // Nested: freed on return, the enclosing scope's allocation untouched.
                let inner_start = scratch.used();
                scratch.scope(|inner| inner.alloc::<f64>(1000).unwrap().fill(3.0));
                assert_eq!(scratch.used(), inner_start);
                // Allocations from before the scope are usable inside it.
                row.iter().sum::<f64>() + a.iter().sum::<f64>()
            });
            assert_eq!(sum, 10.0 + 200.0 * (quartet + 1) as f64);
            assert_eq!(arena.used(), start, "the scope's allocations are free again");
        }
        assert!(row.iter().all(|&x| x == 1.0), "nothing in a scope overlaps what was allocated before it");
        // The next allocation reuses the scopes' space.
        let next = arena.alloc::<f64>(1).unwrap();
        assert_eq!(next.as_ptr() as usize, row.as_ptr() as usize + 80);
        drop(arena);
        // The peak: the row, the last quartet's 300 and its nested 1000.
        assert_eq!(ws.peak_bytes(), 8 * (10 + 300 + 1000));
    }

    #[test]
    fn running_out_inside_a_scope_counts_what_was_allocated_before_it() {
        let mut ws = Scratch::new(1, ByteUnit::KiB).unwrap();
        let mut arena = ws.get_arena();
        arena.alloc::<f64>(64).unwrap();
        let error = arena.scope(|scratch| {
            scratch.alloc::<f64>(32).unwrap();
            scratch.alloc::<f64>(64).unwrap_err()
        });
        assert_eq!(error, ArenaError::OutOfMemory { needed: 512, available: 256, used: 768 });
        // The failed scope took nothing either.
        assert_eq!(arena.used(), 512);
    }

    #[test]
    fn alloc_with_builds_each_element_in_place() {
        let mut ws = Scratch::new(1, ByteUnit::KiB).unwrap();
        let arena = ws.get_arena();
        let data = arena.alloc::<f64>(6).unwrap();
        data.iter_mut().enumerate().for_each(|(i, x)| *x = i as f64);
        let mut rest = Some(&mut data[..]);
        // Three views of two elements each, as the Hermite generations are carved.
        let views = arena.alloc_with(3, |_| {
            let (chunk, tail) = rest.take().unwrap().split_at_mut(2);
            rest = Some(tail);
            chunk
        }).unwrap();
        assert_eq!(views.iter().map(|v| v[1]).collect::<Vec<_>>(), [1.0, 3.0, 5.0]);
        // The views are the arena's numbers, not copies.
        views[2][0] = 40.0;
        assert_eq!(views[2], [40.0, 5.0]);
    }
}
