// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! An array and locks in the memory of one node, used by every rank of a communicator whose ranks share that node
//! (MPI-3 shared memory; plain local memory on one rank or without the `mpi` feature).

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Communicator, MpiError, SharedWindow};

/// One contiguous array of doubles in a node's memory, mapped into the process of every rank of the communicator
/// that made it ([`Communicator::shared_array`]): every rank reads and writes all of it. Writes by different ranks to
/// the same elements must be ordered by the caller (under a lock of [`SharedLocks`], or in phases separated by
/// [`SharedArray::sync`]). The contents are unspecified until written. Freed when dropped, which is collective.
pub struct SharedArray {
    storage: ArrayStorage,
}

enum ArrayStorage {
    Local(Box<[f64]>),
    Node { window: SharedWindow, base: *mut f64, len: usize },
}

impl SharedArray {
    /// See [`Communicator::shared_array`].
    pub(crate) fn new(comm: &Communicator, len: usize) -> Result<SharedArray, MpiError> {
        match node_window(comm, len, "a shared array")? {
            None => Ok(SharedArray { storage: ArrayStorage::Local(vec![0.0; len].into_boxed_slice()) }),
            Some(window) => {
                // The segment may be longer than asked for (whole pages): the array is its first `len` doubles.
                let (base, available) = window.segment(0);
                assert!(available >= len, "a shared segment of {available} doubles for {len}");
                Ok(SharedArray { storage: ArrayStorage::Node { window, base, len } })
            }
        }
    }

    /// The number of doubles.
    pub fn len(&self) -> usize {
        match &self.storage {
            ArrayStorage::Local(data) => data.len(),
            ArrayStorage::Node { len, .. } => *len,
        }
    }

    /// Whether it holds no doubles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The whole array as this process sees it.
    pub fn as_slice(&self) -> &[f64] {
        match &self.storage {
            ArrayStorage::Local(data) => data,
            // Mapped for the window's lifetime; the other ranks' writes become visible at `sync`.
            ArrayStorage::Node { base, len, .. } => unsafe { std::slice::from_raw_parts(*base, *len) },
        }
    }

    /// The whole array as this process sees it, to write.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        match &mut self.storage {
            ArrayStorage::Local(data) => data,
            // Mapped for the window's lifetime; this process holds the only reference to it here.
            ArrayStorage::Node { base, len, .. } => unsafe { std::slice::from_raw_parts_mut(*base, *len) },
        }
    }

    /// Every rank's writes made visible to all, and every rank waited for. Collective; counted as time in
    /// collectives.
    pub fn sync(&self) -> Result<(), MpiError> {
        match &self.storage {
            ArrayStorage::Local(_) => Ok(()),
            ArrayStorage::Node { window, .. } => window.sync(),
        }
    }
}

/// `n` locks shared by the ranks of the communicator that made them ([`Communicator::shared_locks`]): spin locks in
/// the node's shared memory, each held by at most one rank at a time; on one rank they never wait. Taking a lock
/// orders the holder's reads and writes after the last holder's. Freed when dropped, which is collective.
pub struct SharedLocks {
    storage: LockStorage,
    waits: std::cell::Cell<u64>,
}

enum LockStorage {
    Local(usize),
    Node { _window: SharedWindow, base: *mut u64, n: usize },
}

impl SharedLocks {
    /// See [`Communicator::shared_locks`].
    pub(crate) fn new(comm: &Communicator, n: usize) -> Result<SharedLocks, MpiError> {
        let storage = match node_window(comm, n, "shared locks")? {
            None => LockStorage::Local(n),
            Some(window) => {
                let (base, available) = window.segment(0);
                assert!(available >= n, "a shared segment of {available} doubles for {n} locks");
                let base = base.cast::<u64>();
                if comm.is_root() {
                    for i in 0..n {
                        // Every lock free before any rank takes one (the sync below).
                        unsafe { AtomicU64::from_ptr(base.add(i)) }.store(0, Ordering::Relaxed);
                    }
                }
                window.sync()?;
                LockStorage::Node { _window: window, base, n }
            }
        };
        Ok(SharedLocks { storage, waits: std::cell::Cell::new(0) })
    }

    /// The number of locks.
    pub fn len(&self) -> usize {
        match &self.storage {
            LockStorage::Local(n) | LockStorage::Node { n, .. } => *n,
        }
    }

    /// Whether there are no locks.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lock `i`, waited for while another rank holds it; released when the guard is dropped.
    pub fn lock(&self, i: usize) -> SharedLockGuard<'_> {
        assert!(i < self.len(), "lock {i} of {}", self.len());
        let flag = match &self.storage {
            LockStorage::Local(_) => None,
            LockStorage::Node { base, .. } => {
                // A u64 of the node's shared memory, accessed only atomically, by every rank, for the window's
                // lifetime.
                let flag = unsafe { AtomicU64::from_ptr(base.add(i)) };
                let mut spins = 0u32;
                while flag.compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
                    spins += 1;
                    if spins < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
                if spins > 0 {
                    self.waits.set(self.waits.get() + 1);
                }
                Some(flag)
            }
        };
        SharedLockGuard { flag }
    }

    /// How many times this rank found a lock held by another and waited for it.
    pub fn waits(&self) -> u64 {
        self.waits.get()
    }
}

/// A lock of [`SharedLocks`] held by this rank; released when dropped.
pub struct SharedLockGuard<'l> {
    flag: Option<&'l AtomicU64>,
}

impl Drop for SharedLockGuard<'_> {
    fn drop(&mut self) {
        if let Some(flag) = self.flag {
            flag.store(0, Ordering::Release);
        }
    }
}

/// A window of `n` doubles on the root of `comm`, none on the other ranks, when `comm` has several ranks on one node
/// and `n > 0`; `None` (local memory) on one rank or without MPI; an error for ranks on several nodes. Collective.
fn node_window(comm: &Communicator, n: usize, what: &str) -> Result<Option<SharedWindow>, MpiError> {
    if comm.size() == 1 || n == 0 || !cfg!(feature = "mpi") {
        return Ok(None);
    }
    match comm.shared_window(if comm.is_root() { n } else { 0 })? {
        Some(window) => Ok(Some(window)),
        None => Err(MpiError::Argument(format!("{what} needs every rank of its communicator on one node"))),
    }
}

impl Communicator {
    /// Every rank's writes to memory it shares with the others (a [`SharedArray`], or a node's memory shared by other
    /// means) made visible to all, and every rank waited for: a full fence, a barrier, a full fence. Collective.
    pub fn shared_memory_barrier(&self) -> Result<(), MpiError> {
        std::sync::atomic::fence(Ordering::SeqCst);
        self.barrier()?;
        std::sync::atomic::fence(Ordering::SeqCst);
        Ok(())
    }

    /// A [`SharedArray`] of `len` doubles over the ranks of this group, which must share one node's memory (an error
    /// otherwise; [`Communicator::node`] gives such a group). Collective.
    pub fn shared_array(&self, len: usize) -> Result<SharedArray, MpiError> {
        SharedArray::new(self, len)
    }

    /// `n` [`SharedLocks`] over the ranks of this group, which must share one node's memory, all free. Collective.
    pub fn shared_locks(&self, n: usize) -> Result<SharedLocks, MpiError> {
        SharedLocks::new(self, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_one_rank_the_array_is_local_and_the_locks_never_wait() {
        let comm = Communicator::single();
        let mut array = comm.shared_array(5).unwrap();
        assert_eq!(array.len(), 5);
        array.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        array.sync().unwrap();
        assert_eq!(array.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let locks = comm.shared_locks(3).unwrap();
        {
            let _first = locks.lock(0);
            let _second = locks.lock(0);
        }
        assert_eq!((locks.len(), locks.waits()), (3, 0));
    }
}
