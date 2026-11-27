// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

// The MPI calls of aedelsten_mpi, over plain C types only: communicators cross as their Fortran handles (MPI_Fint,
// an int), thread levels as 0-3 (single, funneled, serialized, multiple), counts as int. Every function returns 0 on
// success and -1 on failure. Compiled by the MPI implementation's own mpicc (build.rs).

#include <mpi.h>
#include <stddef.h>
#include <stdlib.h>

static MPI_Comm comm_of(int handle) {
    return MPI_Comm_f2c((MPI_Fint)handle);
}

static int ok(int code) {
    return code == MPI_SUCCESS ? 0 : -1;
}

static int level_to_mpi(int level) {
    switch (level) {
    case 0: return MPI_THREAD_SINGLE;
    case 1: return MPI_THREAD_FUNNELED;
    case 2: return MPI_THREAD_SERIALIZED;
    default: return MPI_THREAD_MULTIPLE;
    }
}

static int level_from_mpi(int level) {
    if (level == MPI_THREAD_SINGLE) return 0;
    if (level == MPI_THREAD_FUNNELED) return 1;
    if (level == MPI_THREAD_SERIALIZED) return 2;
    return 3;
}

// Whether MPI is initialized and not yet finalized: 1 or 0. Callable at any time.
int aedelsten_mpi_is_up(void) {
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    return initialized && !finalized;
}

// Whether MPI_Init has been called, finalized or not: 1 or 0. Callable at any time.
int aedelsten_mpi_was_initialized(void) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    return initialized;
}

// MPI_Init_thread at `required`; the level provided into `provided`.
int aedelsten_mpi_init_thread(int required, int *provided) {
    int mpi_provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(NULL, NULL, level_to_mpi(required), &mpi_provided) != MPI_SUCCESS) {
        return -1;
    }
    *provided = level_from_mpi(mpi_provided);
    return 0;
}

// The thread level MPI was initialized with, into `provided`.
int aedelsten_mpi_query_thread(int *provided) {
    int mpi_provided = MPI_THREAD_SINGLE;
    if (MPI_Query_thread(&mpi_provided) != MPI_SUCCESS) {
        return -1;
    }
    *provided = level_from_mpi(mpi_provided);
    return 0;
}

// MPI_Init unless MPI is initialized already.
int aedelsten_mpi_init(void) {
    if (aedelsten_mpi_was_initialized()) {
        return 0;
    }
    return ok(MPI_Init(NULL, NULL));
}

// MPI_Finalize unless MPI is finalized already.
int aedelsten_mpi_finalize(void) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
        return 0;
    }
    return ok(MPI_Finalize());
}

// The Fortran handle of MPI_COMM_WORLD.
int aedelsten_mpi_comm_world(void) {
    return (int)MPI_Comm_c2f(MPI_COMM_WORLD);
}

int aedelsten_mpi_comm_rank(int comm, int *rank) {
    return ok(MPI_Comm_rank(comm_of(comm), rank));
}

int aedelsten_mpi_comm_size(int comm, int *size) {
    return ok(MPI_Comm_size(comm_of(comm), size));
}

// MPI_Comm_split; the new communicator's Fortran handle into `split`. A negative color is MPI_UNDEFINED: this rank
// joins no group, and `split` is the handle of MPI_COMM_NULL.
int aedelsten_mpi_comm_split(int comm, int color, int key, int *split) {
    MPI_Comm out;
    if (MPI_Comm_split(comm_of(comm), color < 0 ? MPI_UNDEFINED : color, key, &out) != MPI_SUCCESS) {
        return -1;
    }
    *split = (int)MPI_Comm_c2f(out);
    return 0;
}

// The Fortran handle of MPI_COMM_NULL.
int aedelsten_mpi_comm_null(void) {
    return (int)MPI_Comm_c2f(MPI_COMM_NULL);
}

int aedelsten_mpi_comm_free(int comm) {
    MPI_Comm c = comm_of(comm);
    return ok(MPI_Comm_free(&c));
}

int aedelsten_mpi_barrier(int comm) {
    return ok(MPI_Barrier(comm_of(comm)));
}

int aedelsten_mpi_bcast_f64(int comm, double *buf, int n, int root) {
    return ok(MPI_Bcast(buf, n, MPI_DOUBLE, root, comm_of(comm)));
}

int aedelsten_mpi_allreduce_sum_f64(int comm, double *buf, int n) {
    return ok(MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_SUM, comm_of(comm)));
}

// `n` doubles from every rank onto `root`, rank-major into `recv` (read on `root` only; may be NULL elsewhere).
int aedelsten_mpi_gather_f64(int comm, const double *send, int n, double *recv, int root) {
    return ok(MPI_Gather(send, n, MPI_DOUBLE, recv, n, MPI_DOUBLE, root, comm_of(comm)));
}

// `n` doubles from every rank onto every rank, rank-major into `recv`.
int aedelsten_mpi_allgather_f64(int comm, const double *send, int n, double *recv) {
    return ok(MPI_Allgather(send, n, MPI_DOUBLE, recv, n, MPI_DOUBLE, comm_of(comm)));
}

// Each rank sends send[send_displs[k] ..][.. send_counts[k]] to rank k and receives rank k's part into
// recv[recv_displs[k] ..][.. recv_counts[k]] (counts and displacements in doubles).
int aedelsten_mpi_alltoallv_f64(int comm, const double *send, const int *send_counts, const int *send_displs,
                                double *recv, const int *recv_counts, const int *recv_displs) {
    return ok(MPI_Alltoallv(send, send_counts, send_displs, MPI_DOUBLE, recv, recv_counts, recv_displs, MPI_DOUBLE,
                            comm_of(comm)));
}

// `n_send` doubles to rank `to`, `n_recv` doubles from rank `from`, in one pairwise exchange.
int aedelsten_mpi_sendrecv_f64(int comm, const double *send, int n_send, int to, double *recv, int n_recv, int from) {
    return ok(MPI_Sendrecv(send, n_send, MPI_DOUBLE, to, 0, recv, n_recv, MPI_DOUBLE, from, 0, comm_of(comm),
                           MPI_STATUS_IGNORE));
}

// MPI_Comm_split_type by MPI_COMM_TYPE_SHARED: the ranks of `comm` on this rank's node, ordered by `key`; the new
// communicator's Fortran handle into `node`.
int aedelsten_mpi_comm_split_node(int comm, int key, int *node) {
    MPI_Comm out;
    if (MPI_Comm_split_type(comm_of(comm), MPI_COMM_TYPE_SHARED, key, MPI_INFO_NULL, &out) != MPI_SUCCESS) {
        return -1;
    }
    *node = (int)MPI_Comm_c2f(out);
    return 0;
}

typedef struct {
    MPI_Win win;
    MPI_Comm comm;
} aedelsten_shared_window;

// A shared-memory window over `comm` (every rank on one node): this rank's segment of `n` doubles into `mine`,
// placed apart from the others (alloc_shared_noncontig), in one passive epoch for the window's lifetime (lock_all).
// The handle is freed by aedelsten_mpi_shared_window_free.
int aedelsten_mpi_shared_window_allocate(int comm, long n, double **mine, void **handle) {
    aedelsten_shared_window *w = malloc(sizeof(aedelsten_shared_window));
    if (w == NULL) {
        return -1;
    }
    w->comm = comm_of(comm);
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "alloc_shared_noncontig", "true");
    int code = MPI_Win_allocate_shared((MPI_Aint)n * (MPI_Aint)sizeof(double), sizeof(double), info, w->comm, mine,
                                       &w->win);
    MPI_Info_free(&info);
    if (code != MPI_SUCCESS || MPI_Win_lock_all(MPI_MODE_NOCHECK, w->win) != MPI_SUCCESS) {
        free(w);
        return -1;
    }
    *handle = w;
    return 0;
}

// Where rank `rank`'s segment is mapped in this process, and its length in doubles.
int aedelsten_mpi_shared_window_query(void *handle, int rank, double **base, long *n) {
    aedelsten_shared_window *w = handle;
    MPI_Aint size;
    int unit;
    if (MPI_Win_shared_query(w->win, rank, &size, &unit, base) != MPI_SUCCESS) {
        return -1;
    }
    *n = (long)(size / (MPI_Aint)sizeof(double));
    return 0;
}

// Every rank's writes to its segment made visible to all, and every rank waited for: sync, barrier, sync.
int aedelsten_mpi_shared_window_sync(void *handle) {
    aedelsten_shared_window *w = handle;
    if (MPI_Win_sync(w->win) != MPI_SUCCESS || MPI_Barrier(w->comm) != MPI_SUCCESS) {
        return -1;
    }
    return ok(MPI_Win_sync(w->win));
}

// Ends the epoch and frees the window and the handle.
int aedelsten_mpi_shared_window_free(void *handle) {
    aedelsten_shared_window *w = handle;
    int code = MPI_Win_unlock_all(w->win) == MPI_SUCCESS && MPI_Win_free(&w->win) == MPI_SUCCESS ? 0 : -1;
    free(w);
    return code;
}

typedef struct {
    MPI_Win win;
    long *base;
} aedelsten_counter;

// A shared counter over `comm`: one long on rank 0, set to 0, in one passive epoch for the counter's lifetime
// (lock_all). The handle is freed by aedelsten_mpi_counter_free.
int aedelsten_mpi_counter_allocate(int comm, void **handle) {
    MPI_Comm c = comm_of(comm);
    int rank;
    if (MPI_Comm_rank(c, &rank) != MPI_SUCCESS) {
        return -1;
    }
    aedelsten_counter *w = malloc(sizeof(aedelsten_counter));
    if (w == NULL) {
        return -1;
    }
    MPI_Aint size = rank == 0 ? (MPI_Aint)sizeof(long) : 0;
    if (MPI_Win_allocate(size, sizeof(long), MPI_INFO_NULL, c, &w->base, &w->win) != MPI_SUCCESS) {
        free(w);
        return -1;
    }
    int code = MPI_SUCCESS;
    if (rank == 0) {
        code = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, w->win);
        *w->base = 0;
        if (code == MPI_SUCCESS) {
            code = MPI_Win_unlock(0, w->win);
        }
    }
    if (code != MPI_SUCCESS || MPI_Barrier(c) != MPI_SUCCESS || MPI_Win_lock_all(0, w->win) != MPI_SUCCESS) {
        MPI_Win_free(&w->win);
        free(w);
        return -1;
    }
    *handle = w;
    return 0;
}

// The counter's value into `value`, and the counter incremented by one, atomically.
int aedelsten_mpi_counter_next(void *handle, long *value) {
    aedelsten_counter *w = handle;
    long one = 1;
    if (MPI_Fetch_and_op(&one, value, MPI_LONG, 0, 0, MPI_SUM, w->win) != MPI_SUCCESS) {
        return -1;
    }
    return ok(MPI_Win_flush(0, w->win));
}

// Ends the epoch and frees the window and the handle. Collective.
int aedelsten_mpi_counter_free(void *handle) {
    aedelsten_counter *w = handle;
    int code = MPI_Win_unlock_all(w->win) == MPI_SUCCESS && MPI_Win_free(&w->win) == MPI_SUCCESS ? 0 : -1;
    free(w);
    return code;
}
