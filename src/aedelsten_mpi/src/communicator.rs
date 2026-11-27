// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Communicators: a group of ranks and the operations among them, over `f64` buffers.

use std::sync::Arc;

use crate::{greedy_partition, MpiError};

#[cfg(feature = "mpi")]
use crate::{check, mpi_count, raw, timed_collective};

/// A group of ranks: this process alone ([`Communicator::single`]), or an MPI communicator, had from a
/// [`crate::Session`] (its world, a Fortran handle) or from [`Communicator::split`]. Every operation that names it
/// collective must be called by every rank of the group at the same point. A clone shares the group (cheap: the
/// handle is shared, not duplicated). A communicator made by `split` is freed when its last clone is dropped, which is
/// collective too: every rank drops its last clone at the same point.
#[derive(Debug, Clone)]
pub struct Communicator {
    kind: Kind,
    rank: usize,
    size: usize,
}

#[derive(Debug, Clone)]
enum Kind {
    Single,
    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    Mpi(Arc<Handle>),
}

/// An MPI communicator's Fortran handle; one made by `split` (`owned`) is freed when dropped.
#[derive(Debug)]
struct Handle {
    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    handle: i32,
    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    owned: bool,
}

impl Communicator {
    /// This process alone: rank 0 of 1. Needs no MPI.
    pub fn single() -> Communicator {
        Communicator { kind: Kind::Single, rank: 0, size: 1 }
    }

    /// The communicator with Fortran handle `handle`, not freed when dropped. MPI must be up.
    #[cfg(feature = "mpi")]
    pub(crate) fn from_handle(handle: i32, owned: bool) -> Result<Communicator, MpiError> {
        let (mut rank, mut size) = (0, 0);
        check("MPI_Comm_rank", unsafe { raw::aedelsten_mpi_comm_rank(handle, &mut rank) })?;
        check("MPI_Comm_size", unsafe { raw::aedelsten_mpi_comm_size(handle, &mut size) })?;
        Ok(Communicator { kind: Kind::Mpi(Arc::new(Handle { handle, owned })), rank: rank as usize, size: size as usize })
    }

    /// This rank's index in the group.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The number of ranks in the group.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Whether this is rank 0.
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// The MPI Fortran handle (`MPI_Fint`) of the communicator, for passing it to another MPI user; `None` for
    /// [`Communicator::single`].
    pub fn fortran_handle(&self) -> Option<i32> {
        match &self.kind {
            Kind::Single => None,
            Kind::Mpi(inner) => Some(inner.handle),
        }
    }

    #[cfg(feature = "mpi")]
    fn handle(&self) -> Option<i32> {
        match &self.kind {
            Kind::Mpi(inner) if self.size > 1 => Some(inner.handle),
            _ => None,
        }
    }

    /// Returns once every rank has called it. Collective.
    pub fn barrier(&self) -> Result<(), MpiError> {
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            return check("MPI_Barrier", timed_collective(|| unsafe { raw::aedelsten_mpi_barrier(comm) }));
        }
        Ok(())
    }

    /// `buf` from rank `root` on every rank, in place (every rank passes a buffer of the same length). Collective.
    pub fn broadcast(&self, buf: &mut [f64], root: usize) -> Result<(), MpiError> {
        self.check_rank(root)?;
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            let n = mpi_count(buf.len())?;
            return check(
                "MPI_Bcast",
                timed_collective(|| unsafe { raw::aedelsten_mpi_bcast_f64(comm, buf.as_mut_ptr(), n, root as i32) }),
            );
        }
        let _ = buf;
        Ok(())
    }

    /// `buf` summed over the ranks, in place, on every rank. Collective. Sums at most 2^27 doubles (1 GB) per MPI
    /// call.
    pub fn sum(&self, buf: &mut [f64]) -> Result<(), MpiError> {
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            return timed_collective(|| {
                for chunk in buf.chunks_mut(crate::ALLREDUCE_CHUNK) {
                    check("MPI_Allreduce", unsafe {
                        raw::aedelsten_mpi_allreduce_sum_f64(comm, chunk.as_mut_ptr(), chunk.len() as i32)
                    })?;
                }
                Ok(())
            });
        }
        let _ = buf;
        Ok(())
    }

    /// Every rank's `local` (the same length on all) on rank `root`, concatenated in rank order into `out` (room
    /// for `size() * local.len()`; not read on the other ranks, where it may be empty). Collective.
    pub fn gather(&self, local: &[f64], out: &mut [f64], root: usize) -> Result<(), MpiError> {
        self.check_rank(root)?;
        if self.rank == root {
            check_room(out.len(), local.len() * self.size)?;
        }
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            let n = mpi_count(local.len())?;
            let recv = if self.rank == root { out.as_mut_ptr() } else { std::ptr::null_mut() };
            return check(
                "MPI_Gather",
                timed_collective(|| unsafe { raw::aedelsten_mpi_gather_f64(comm, local.as_ptr(), n, recv, root as i32) }),
            );
        }
        out[..local.len()].copy_from_slice(local);
        Ok(())
    }

    /// Every rank's `local` (the same length on all), concatenated in rank order into `out` (room for
    /// `size() * local.len()`) on every rank. Collective.
    pub fn all_gather(&self, local: &[f64], out: &mut [f64]) -> Result<(), MpiError> {
        check_room(out.len(), local.len() * self.size)?;
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            let n = mpi_count(local.len())?;
            return check(
                "MPI_Allgather",
                timed_collective(|| unsafe { raw::aedelsten_mpi_allgather_f64(comm, local.as_ptr(), n, out.as_mut_ptr()) }),
            );
        }
        out[..local.len()].copy_from_slice(local);
        Ok(())
    }

    /// Personalized exchange: this rank sends `send[send_displs[k]..][..send_counts[k]]` to rank `k` and receives
    /// rank `k`'s part into `recv[recv_displs[k]..][..recv_counts[k]]`. Every rank passes `size()` counts and
    /// displacements each way, consistent with what the others send. Allocates nothing. Collective.
    pub fn all_to_all(
        &self,
        send: &[f64],
        send_counts: &[i32],
        send_displs: &[i32],
        recv: &mut [f64],
        recv_counts: &[i32],
        recv_displs: &[i32],
    ) -> Result<(), MpiError> {
        let ranks = self.size;
        if [send_counts.len(), send_displs.len(), recv_counts.len(), recv_displs.len()] != [ranks; 4] {
            return Err(MpiError::Argument(format!(
                "an all-to-all over {ranks} ranks needs {ranks} counts and displacements each way"
            )));
        }
        let fits = |data: usize, counts: &[i32], displs: &[i32]| {
            counts.iter().zip(displs).all(|(&c, &d)| c >= 0 && d >= 0 && c as usize + d as usize <= data)
        };
        if !fits(send.len(), send_counts, send_displs) || !fits(recv.len(), recv_counts, recv_displs) {
            return Err(MpiError::Argument("all-to-all counts and displacements outside the buffers".into()));
        }
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            return check(
                "MPI_Alltoallv",
                timed_collective(|| unsafe {
                    raw::aedelsten_mpi_alltoallv_f64(
                        comm,
                        send.as_ptr(),
                        send_counts.as_ptr(),
                        send_displs.as_ptr(),
                        recv.as_mut_ptr(),
                        recv_counts.as_ptr(),
                        recv_displs.as_ptr(),
                    )
                }),
            );
        }
        let (n, from, to) = (send_counts[0] as usize, send_displs[0] as usize, recv_displs[0] as usize);
        if recv_counts[0] as usize != n {
            return Err(MpiError::Argument(format!(
                "an all-to-all on one rank sends {n} values and receives {}",
                recv_counts[0]
            )));
        }
        recv[to..to + n].copy_from_slice(&send[from..from + n]);
        Ok(())
    }

    /// Sends `send` to rank `to` and receives `recv` (as long as rank `from` sends) from rank `from`: one pairwise
    /// exchange, in which only the partners wait for each other. Its time counts as time in collectives. On one
    /// rank a copy.
    pub fn send_receive(&self, send: &[f64], to: usize, recv: &mut [f64], from: usize) -> Result<(), MpiError> {
        self.check_rank(to)?;
        self.check_rank(from)?;
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            let (n_send, n_recv) = (mpi_count(send.len())?, mpi_count(recv.len())?);
            return check(
                "MPI_Sendrecv",
                timed_collective(|| unsafe {
                    raw::aedelsten_mpi_sendrecv_f64(comm, send.as_ptr(), n_send, to as i32, recv.as_mut_ptr(), n_recv, from as i32)
                }),
            );
        }
        if send.len() != recv.len() {
            return Err(MpiError::Argument(format!(
                "a send-receive on one rank sends {} values and receives {}",
                send.len(),
                recv.len()
            )));
        }
        recv.copy_from_slice(send);
        Ok(())
    }

    /// The ranks with the same `color` as a new group, ordered by `key` (then by rank here); `None` on a rank whose
    /// `color` is `None`, which joins no group. Collective. The new communicator is freed when dropped.
    pub fn split(&self, color: Option<usize>, key: usize) -> Result<Option<Communicator>, MpiError> {
        match &self.kind {
            Kind::Single => {
                let _ = key;
                Ok(color.map(|_| Communicator::single()))
            }
            #[cfg_attr(not(feature = "mpi"), allow(unused_variables))]
            Kind::Mpi(inner) => {
                #[cfg(feature = "mpi")]
                {
                    let color = match color {
                        Some(c) => i32::try_from(c).map_err(|_| MpiError::Argument(format!("color {c} too large")))?,
                        None => -1,
                    };
                    let key = i32::try_from(key).map_err(|_| MpiError::Argument(format!("key {key} too large")))?;
                    let mut split = 0;
                    check(
                        "MPI_Comm_split",
                        timed_collective(|| unsafe { raw::aedelsten_mpi_comm_split(inner.handle, color, key, &mut split) }),
                    )?;
                    if split == unsafe { raw::aedelsten_mpi_comm_null() } {
                        return Ok(None);
                    }
                    Communicator::from_handle(split, true).map(Some)
                }
                #[cfg(not(feature = "mpi"))]
                unreachable!("MPI communicators exist only with the mpi feature")
            }
        }
    }

    /// The ranks of the group that share this rank's node (its memory: `MPI_COMM_TYPE_SHARED`) as a group of their
    /// own, in their order here; [`Communicator::single`]'s is a single one. Collective. Freed when its last clone is
    /// dropped, like a [`Communicator::split`] one.
    pub fn node(&self) -> Result<Communicator, MpiError> {
        match &self.kind {
            Kind::Single => Ok(Communicator::single()),
            #[cfg_attr(not(feature = "mpi"), allow(unused_variables))]
            Kind::Mpi(inner) => {
                #[cfg(feature = "mpi")]
                {
                    let key = i32::try_from(self.rank).map_err(|_| MpiError::Argument(format!("rank {} too large", self.rank)))?;
                    let mut node = 0;
                    check(
                        "MPI_Comm_split_type",
                        timed_collective(|| unsafe { raw::aedelsten_mpi_comm_split_node(inner.handle, key, &mut node) }),
                    )?;
                    Communicator::from_handle(node, true)
                }
                #[cfg(not(feature = "mpi"))]
                unreachable!("MPI communicators exist only with the mpi feature")
            }
        }
    }

    /// Whether every rank of the group shares one node's memory (see [`Communicator::node`]); `true` on one rank.
    /// Collective.
    pub fn is_one_node(&self) -> Result<bool, MpiError> {
        if self.size == 1 {
            return Ok(true);
        }
        Ok(self.node()?.size() == self.size)
    }

    /// A shared-memory window with a segment of `n` doubles for this rank, when there are several ranks and all of
    /// them share one node's memory; `None` otherwise (one rank, ranks on several nodes, no `mpi` feature).
    /// Collective.
    pub fn shared_window(&self, n: usize) -> Result<Option<SharedWindow>, MpiError> {
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            if !self.is_one_node()? {
                return Ok(None);
            }
            let n_long = std::ffi::c_long::try_from(n)
                .map_err(|_| MpiError::Argument(format!("a shared window of {n} doubles is too large")))?;
            let (mut mine, mut handle) = (std::ptr::null_mut(), std::ptr::null_mut());
            check(
                "MPI_Win_allocate_shared",
                timed_collective(|| unsafe { raw::aedelsten_mpi_shared_window_allocate(comm, n_long, &mut mine, &mut handle) }),
            )?;
            let mut segments = Vec::with_capacity(self.size);
            for rank in 0..self.size {
                let (mut base, mut len) = (std::ptr::null_mut(), 0);
                check("MPI_Win_shared_query", unsafe {
                    raw::aedelsten_mpi_shared_window_query(handle, rank as i32, &mut base, &mut len)
                })?;
                segments.push((base, len as usize));
            }
            let window = SharedWindow { handle, segments, me: self.rank };
            crate::shared_mapped(window.mapped_bytes());
            return Ok(Some(window));
        }
        let _ = n;
        Ok(None)
    }

    /// A counter shared by the group, starting at 0: each [`Counter::next`] on any rank takes the next value, so
    /// that the ranks hand out items among themselves as they become free. Collective.
    pub fn counter(&self) -> Result<Counter, MpiError> {
        #[cfg(feature = "mpi")]
        if let Some(comm) = self.handle() {
            let mut handle = std::ptr::null_mut();
            check("MPI_Win_allocate", timed_collective(|| unsafe { raw::aedelsten_mpi_counter_allocate(comm, &mut handle) }))?;
            return Ok(Counter { kind: CounterKind::Mpi(handle) });
        }
        Ok(Counter { kind: CounterKind::Local(std::cell::Cell::new(0)) })
    }

    /// This rank's share of items `0..costs.len()` (ascending), balanced by cost over the group by the greedy
    /// (longest-processing-time) heuristic. Every rank must pass the same `costs` in the same order; no
    /// communication.
    pub fn partition(&self, costs: &[usize]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..costs.len()).collect();
        order.sort_unstable_by_key(|&i| std::cmp::Reverse(costs[i]));
        let sorted_costs: Vec<usize> = order.iter().map(|&i| costs[i]).collect();
        let buckets = greedy_partition(&sorted_costs, self.size);
        let mut owned: Vec<usize> = buckets[self.rank].iter().map(|&pos| order[pos]).collect();
        owned.sort_unstable();
        owned
    }

    fn check_rank(&self, rank: usize) -> Result<(), MpiError> {
        if rank < self.size {
            Ok(())
        } else {
            Err(MpiError::Argument(format!("rank {rank} in a group of {}", self.size)))
        }
    }
}

fn check_room(room: usize, needed: usize) -> Result<(), MpiError> {
    if room < needed {
        Err(MpiError::Argument(format!("room for {room} values, {needed} needed")))
    } else {
        Ok(())
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        #[cfg(feature = "mpi")]
        if self.owned && unsafe { raw::aedelsten_mpi_is_up() } == 1 {
            unsafe { raw::aedelsten_mpi_comm_free(self.handle) };
        }
    }
}

/// A shared-memory window over the ranks of a communicator on one node (MPI-3): each rank's segment, mapped into
/// every rank's process, so that a rank reads another's segment directly. A segment is written by its owner only;
/// [`SharedWindow::sync`] goes between the writes and the reads, and between the reads and the next writes. Freed
/// when dropped, which is collective.
pub struct SharedWindow {
    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    handle: *mut std::ffi::c_void,
    segments: Vec<(*mut f64, usize)>,
    me: usize,
}

impl SharedWindow {
    /// Where rank `rank`'s segment is mapped in this process, and its length in doubles.
    pub(crate) fn segment(&self, rank: usize) -> (*mut f64, usize) {
        self.segments[rank]
    }

    /// Doubles in rank `rank`'s segment.
    pub fn len_of(&self, rank: usize) -> usize {
        self.segments[rank].1
    }

    /// Bytes of this rank's segment.
    pub fn bytes(&self) -> usize {
        self.len_of(self.me) * size_of::<f64>()
    }

    /// Bytes of all ranks' segments, each mapped into this process.
    fn mapped_bytes(&self) -> usize {
        self.segments.iter().map(|&(_, len)| len).sum::<usize>() * size_of::<f64>()
    }

    /// Copies `data` to the start of this rank's segment (at most its length).
    pub fn write(&self, data: &[f64]) -> Result<(), MpiError> {
        let (base, len) = self.segments[self.me];
        if data.len() > len {
            return Err(MpiError::Argument(format!("{} doubles for a shared segment of {len}", data.len())));
        }
        // The segment is this rank's, and no other rank reads it between the two syncs around this write.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), base, data.len()) };
        Ok(())
    }

    /// The first `len` doubles of rank `rank`'s segment, as its owner wrote them before the last sync.
    pub fn read(&self, rank: usize, len: usize) -> &[f64] {
        let (base, available) = self.segments[rank];
        assert!(len <= available, "{len} doubles read from a shared segment of {available}");
        // Mapped for the window's lifetime; its owner writes it only after the next sync.
        unsafe { std::slice::from_raw_parts(base, len) }
    }

    /// Every rank's writes made visible to all, and every rank waited for. Collective; counted as time in
    /// collectives.
    pub fn sync(&self) -> Result<(), MpiError> {
        #[cfg(feature = "mpi")]
        check("MPI_Win_sync", timed_collective(|| unsafe { raw::aedelsten_mpi_shared_window_sync(self.handle) }))?;
        Ok(())
    }
}

impl Drop for SharedWindow {
    fn drop(&mut self) {
        crate::shared_unmapped(self.mapped_bytes());
        #[cfg(feature = "mpi")]
        if unsafe { raw::aedelsten_mpi_is_up() } == 1 {
            unsafe { raw::aedelsten_mpi_shared_window_free(self.handle) };
        }
    }
}

/// A counter shared by the ranks of a communicator ([`Communicator::counter`]): one value on rank 0, taken and
/// incremented atomically by any rank (MPI-3 `MPI_Fetch_and_op` in a passive epoch), or a local count on one rank.
/// Freed when dropped, which is collective.
pub struct Counter {
    kind: CounterKind,
}

enum CounterKind {
    Local(std::cell::Cell<usize>),
    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    Mpi(*mut std::ffi::c_void),
}

impl Counter {
    /// The counter's value, which this call increments by one for every rank: no two calls, on any ranks, get the
    /// same value. Counted as time in collectives.
    pub fn next(&self) -> Result<usize, MpiError> {
        match &self.kind {
            CounterKind::Local(count) => {
                let value = count.get();
                count.set(value + 1);
                Ok(value)
            }
            #[cfg(feature = "mpi")]
            CounterKind::Mpi(handle) => {
                let mut value: std::ffi::c_long = 0;
                check("MPI_Fetch_and_op", timed_collective(|| unsafe { raw::aedelsten_mpi_counter_next(*handle, &mut value) }))?;
                Ok(value as usize)
            }
            #[cfg(not(feature = "mpi"))]
            CounterKind::Mpi(_) => unreachable!("an MPI counter without the mpi feature"),
        }
    }
}

impl Drop for Counter {
    fn drop(&mut self) {
        #[cfg(feature = "mpi")]
        if let CounterKind::Mpi(handle) = self.kind {
            if unsafe { raw::aedelsten_mpi_is_up() } == 1 {
                unsafe { raw::aedelsten_mpi_counter_free(handle) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_communicator_is_rank_zero_of_one_and_copies_locally() {
        let comm = Communicator::single();
        assert_eq!((comm.rank(), comm.size(), comm.is_root()), (0, 1, true));
        assert_eq!(comm.fortran_handle(), None);

        let mut buf = [1.0, 2.0, 3.0];
        comm.sum(&mut buf).unwrap();
        comm.broadcast(&mut buf, 0).unwrap();
        assert_eq!(buf, [1.0, 2.0, 3.0]);
        assert!(comm.broadcast(&mut buf, 1).is_err(), "rank 1 does not exist");

        let mut out = [0.0; 3];
        comm.all_gather(&buf, &mut out).unwrap();
        assert_eq!(out, buf);
        let mut out = [0.0; 3];
        comm.gather(&buf, &mut out, 0).unwrap();
        assert_eq!(out, buf);
        assert!(comm.all_gather(&buf, &mut [0.0; 2]).is_err(), "too little room");

        let mut recv = [0.0; 3];
        comm.send_receive(&buf, 0, &mut recv, 0).unwrap();
        assert_eq!(recv, buf);

        let mut recv = [0.0; 4];
        comm.all_to_all(&buf, &[2], &[1], &mut recv, &[2], &[2]).unwrap();
        assert_eq!(recv, [0.0, 0.0, 2.0, 3.0]);
        assert!(comm.all_to_all(&buf, &[3], &[1], &mut recv, &[3], &[0]).is_err(), "outside the send buffer");

        comm.barrier().unwrap();
        assert!(comm.is_one_node().unwrap());
        assert!(comm.shared_window(10).unwrap().is_none());
    }

    #[test]
    fn a_single_communicator_counts_from_zero() {
        let counter = Communicator::single().counter().unwrap();
        assert_eq!((0..4).map(|_| counter.next().unwrap()).collect::<Vec<_>>(), [0, 1, 2, 3]);
    }

    #[test]
    fn a_clone_of_a_single_communicator_is_one_too() {
        let comm = Communicator::single();
        let clone = comm.clone();
        drop(comm);
        let mut buf = [1.0, 2.0];
        clone.sum(&mut buf).unwrap();
        assert_eq!((clone.rank(), clone.size(), buf), (0, 1, [1.0, 2.0]));
    }

    #[test]
    fn splitting_a_single_communicator_gives_a_single_one_or_none() {
        let comm = Communicator::single();
        assert_eq!(comm.split(Some(3), 0).unwrap().map(|c| c.size()), Some(1));
        assert!(comm.split(None, 0).unwrap().is_none());
    }

    #[test]
    fn a_single_communicators_node_is_a_single_one() {
        let node = Communicator::single().node().unwrap();
        assert_eq!((node.rank(), node.size(), node.fortran_handle()), (0, 1, None));
    }

    #[test]
    fn one_rank_owns_every_item_in_the_original_order() {
        let costs = [3, 1, 4, 1, 5, 9, 2, 6];
        assert_eq!(Communicator::single().partition(&costs), (0..costs.len()).collect::<Vec<_>>());
    }
}
