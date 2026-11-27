// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Every communicator operation on the ranks it runs on, checked against the values each rank can compute alone.
//! Run under MPI on any number of ranks:
//!
//! `cargo build -p aedelsten_mpi --features mpi --example communicator_check`, then
//! `mpirun -n 4 target/debug/examples/communicator_check`.

use aedelsten_mpi::{collective_seconds, Communicator, MpiError, Session, ThreadLevel};

fn check_group(comm: &Communicator, label: &str) -> Result<(), MpiError> {
    let (rank, size) = (comm.rank(), comm.size());
    let value = |r: usize, k: usize| (100 * r + k) as f64;

    let mut buf: Vec<f64> = (0..3).map(|k| value(rank, k)).collect();
    comm.sum(&mut buf)?;
    let expected: Vec<f64> = (0..3).map(|k| (0..size).map(|r| value(r, k)).sum()).collect();
    assert_eq!(buf, expected, "{label}: sum");

    let mut buf: Vec<f64> = (0..3).map(|k| value(rank, k)).collect();
    comm.broadcast(&mut buf, size - 1)?;
    assert_eq!(buf, (0..3).map(|k| value(size - 1, k)).collect::<Vec<_>>(), "{label}: broadcast");

    let local: Vec<f64> = (0..2).map(|k| value(rank, k)).collect();
    let everyone: Vec<f64> = (0..size).flat_map(|r| (0..2).map(move |k| value(r, k))).collect();
    let mut out = vec![0.0; 2 * size];
    comm.all_gather(&local, &mut out)?;
    assert_eq!(out, everyone, "{label}: all_gather");
    let mut out = if comm.is_root() { vec![0.0; 2 * size] } else { Vec::new() };
    comm.gather(&local, &mut out, 0)?;
    if comm.is_root() {
        assert_eq!(out, everyone, "{label}: gather");
    }

    // Rank r sends k + 1 values to rank k: value(r, k) repeated.
    let send_counts: Vec<i32> = (0..size as i32).map(|k| k + 1).collect();
    let send_displs: Vec<i32> = (0..size as i32).map(|k| k * (k + 1) / 2).collect();
    let send: Vec<f64> = (0..size).flat_map(|k| std::iter::repeat_n(value(rank, k), k + 1)).collect();
    let recv_counts = vec![rank as i32 + 1; size];
    let recv_displs: Vec<i32> = (0..size as i32).map(|r| r * (rank as i32 + 1)).collect();
    let mut recv = vec![0.0; size * (rank + 1)];
    comm.all_to_all(&send, &send_counts, &send_displs, &mut recv, &recv_counts, &recv_displs)?;
    let expected: Vec<f64> = (0..size).flat_map(|r| std::iter::repeat_n(value(r, rank), rank + 1)).collect();
    assert_eq!(recv, expected, "{label}: all_to_all");

    // A ring: each rank sends to the next, receives from the previous.
    let (next, previous) = ((rank + 1) % size, (rank + size - 1) % size);
    let mut recv = [0.0; 2];
    comm.send_receive(&[value(rank, 0), value(rank, 1)], next, &mut recv, previous)?;
    assert_eq!(recv, [value(previous, 0), value(previous, 1)], "{label}: send_receive");

    comm.barrier()?;

    if let Some(window) = comm.shared_window(4)? {
        window.write(&[value(rank, 0), value(rank, 1), value(rank, 2), value(rank, 3)])?;
        window.sync()?;
        for r in 0..size {
            assert_eq!(window.read(r, 4), [value(r, 0), value(r, 1), value(r, 2), value(r, 3)], "{label}: window");
        }
        window.sync()?;
    } else {
        assert!(size == 1 || !comm.is_one_node()?, "{label}: a window on one node");
    }

    // A node's shared array: every rank writes its own elements and reads all; then every rank adds into the same
    // elements under shared locks, so that nothing is lost.
    let node = comm.node()?;
    let mut array = node.shared_array(2 * node.size())?;
    array.as_mut_slice()[node.rank()] = value(node.rank(), 0);
    array.as_mut_slice()[node.size() + node.rank()] = 0.0;
    array.sync()?;
    assert!((0..node.size()).all(|r| array.as_slice()[r] == value(r, 0)), "{label}: a shared array");
    let locks = node.shared_locks(node.size())?;
    for round in 0..200 {
        let k = node.size() + (round + node.rank()) % node.size();
        let _held = locks.lock(k - node.size());
        array.as_mut_slice()[k] += 1.0;
    }
    array.sync()?;
    let added: f64 = array.as_slice()[node.size()..].iter().sum();
    assert_eq!(added, 200.0 * node.size() as f64, "{label}: additions under shared locks");
    array.sync()?;

    let costs: Vec<usize> = (0..17).map(|i| (i * 7) % 5 + 1).collect();
    let mine: Vec<f64> = comm.partition(&costs).iter().map(|&i| i as f64).collect();
    let mut counts = vec![0.0; costs.len()];
    for &i in &mine {
        counts[i as usize] += 1.0;
    }
    comm.sum(&mut counts)?;
    assert!(counts.iter().all(|&c| c == 1.0), "{label}: every item on exactly one rank");

    // The ranks take values from a shared counter until 50 are gone, rank 0 slowed down: every value once.
    let counter = comm.counter()?;
    let mut taken = vec![0.0; 50];
    loop {
        let value = counter.next()?;
        if value >= taken.len() {
            break;
        }
        taken[value] += 1.0;
        if rank == 0 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    comm.sum(&mut taken)?;
    assert!(taken.iter().all(|&c| c == 1.0), "{label}: every counter value taken once");
    drop(counter);
    Ok(())
}

fn main() -> Result<(), MpiError> {
    let session = Session::init(ThreadLevel::Funneled)?;
    assert!(session.is_owner());
    assert!(session.thread_level() >= ThreadLevel::Funneled);
    assert_eq!(Session::init(ThreadLevel::Single).unwrap_err(), MpiError::AlreadyInitialized);
    let attached = Session::attach(ThreadLevel::Single)?;
    assert!(!attached.is_owner());
    drop(attached);

    let world = session.world()?;
    check_group(&world, "world")?;
    check_group(&Communicator::single(), "single")?;

    let by_handle = session.communicator(world.fortran_handle().unwrap())?;
    assert_eq!((by_handle.rank(), by_handle.size()), (world.rank(), world.size()));
    let by_sentinel = session.communicator(aedelsten_mpi::COMM_WORLD_SENTINEL)?;
    assert_eq!(by_sentinel.size(), world.size());

    // Two groups, even and odd ranks, ordered in reverse; and a group rank 0 stays out of.
    let parity = world.split(Some(world.rank() % 2), world.size() - world.rank())?.expect("every rank has a color");
    assert_eq!(parity.size(), (world.size() + 1 - world.rank() % 2) / 2);
    check_group(&parity, "parity")?;
    let without_root = world.split((world.rank() > 0).then_some(0), world.rank())?;
    assert_eq!(without_root.is_none(), world.rank() == 0);
    if let Some(comm) = &without_root {
        assert_eq!((comm.rank(), comm.size()), (world.rank() - 1, world.size() - 1));
        check_group(comm, "without root")?;
    }
    drop(without_root);

    // Clones share the split group: usable after the original is dropped, freed once, with the last clone.
    let clone = parity.clone();
    let second = clone.clone();
    assert_eq!(clone.fortran_handle(), parity.fortran_handle());
    drop(parity);
    check_group(&clone, "parity, a clone")?;
    drop(clone);
    let mut ones = [1.0];
    second.sum(&mut ones)?;
    assert_eq!(ones[0], second.size() as f64, "parity, the last clone: sum");
    drop(second);
    // Every rank here is on this machine's node: the node of the world is the world, in its order; a split group's
    // node holds that group's ranks only.
    let node = world.node()?;
    assert_eq!((node.rank(), node.size()), (world.rank(), world.size()), "world: node");
    check_group(&node, "world's node")?;
    drop(node);
    let halves = world.split(Some(world.rank() % 2), world.rank())?.expect("every rank has a color");
    let halves_node = halves.node()?;
    assert_eq!((halves_node.rank(), halves_node.size()), (halves.rank(), halves.size()), "a split group: node");
    drop(halves_node);
    drop(halves);
    assert!(world.is_one_node()?);
    assert_eq!(Communicator::single().node()?.size(), 1);

    let world_clone = world.clone();
    check_group(&world_clone, "world, a clone")?;
    drop(world_clone);

    assert_eq!(aedelsten_mpi::rank(), world.rank());
    assert_eq!(aedelsten_mpi::size(), world.size());
    if world.size() > 1 {
        assert!(collective_seconds() > 0.0);
    }
    // A collective's time is a span under the span open on the thread; on one rank no MPI call is made.
    instrumentation::spans::reset();
    {
        let _caller = instrumentation::span("caller");
        world.sum(&mut [1.0])?;
    }
    let tree = instrumentation::tree();
    let labels: Vec<(&str, usize)> = tree.nodes().iter().map(|n| (n.label, n.depth)).collect();
    let expected: &[(&str, usize)] = if world.size() > 1 { &[("caller", 0), (aedelsten_mpi::COLLECTIVE_SPAN, 1)] } else { &[("caller", 0)] };
    assert_eq!(labels, expected, "the collective's span");
    if world.is_root() {
        println!("communicator_check: all passed on {} ranks ({})", world.size(), session.thread_level());
    }
    Ok(())
}
