// SPDX-FileCopyrightText: 2024 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! The BLAS and LAPACK routines the crate calls: the C declarations and thin wrappers that pass the arguments
//! through (CBLAS for BLAS, the Fortran calling convention for LAPACK). The only place BLAS and LAPACK are called.

use std::ffi::c_char;

/// CBLAS's storage order.
#[repr(C)]
#[allow(dead_code)]
pub(crate) enum CblasOrder {
    CblasRowMajor = 101,
    CblasColMajor = 102,
}
/// CBLAS's transpose flag.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub(crate) enum CblasTranspose {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
}
/// CBLAS's triangle: upper or lower.
#[repr(C)]
pub(crate) enum CblasUplo {
    CblasUpper = 121,
    CblasLower = 122,
}
/// CBLAS's unit-diagonal flag.
#[repr(C)]
#[allow(dead_code)]
enum CblasDiag {
    CblasNonUnit = 131,
    CblasUnit = 132,
}
#[repr(C)]
#[allow(dead_code)]
enum CblasSide {
    CblasLeft = 141,
    CblasRight = 142,
}

extern "C" {
    fn cblas_dgemm(
        cblas_order: CblasOrder,
        cblas_trans_a: CblasTranspose,
        cblas_trans_b: CblasTranspose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *const f64,
        ldc: i32,
    );
    fn cblas_dgemv(
        cblas_order: CblasOrder,
        cblas_trans: CblasTranspose,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *const f64,
        incy: i32,
    );

    fn dgetrf_(m: &i32, n: &i32, a: *const f64, lda: &i32, ipiv: *const i32, info: &mut i32);
    fn dgetri_(
        n: &i32,
        a: *const f64,
        lda: &i32,
        ipiv: *const i32,
        work: *const f64,
        lwork: &i32,
        info: &mut i32,
    );
    fn dgels_(
        trans: &c_char,
        m: &i32,
        n: &i32,
        nrhs: &i32,
        a: *mut f64,
        lda: &i32,
        b: *mut f64,
        ldb: &i32,
        work: *mut f64,
        lwork: &i32,
        info: &mut i32,
    );
    fn dgesvd_(
        jobu: &c_char,
        jobvt: &c_char,
        m: &i32,
        n: &i32,
        a: *mut f64,
        lda: &i32,
        s: *mut f64,
        u: *mut f64,
        ldu: &i32,
        vt: *mut f64,
        ldvt: &i32,
        work: *mut f64,
        lwork: &i32,
        info: *mut i32,
    );
    fn dsyev_(
        jobz: &c_char,
        uplo: &c_char,
        n: &i32,
        a: *const f64,
        lda: &i32,
        w: *const f64,
        work: *const f64,
        lwork: &i32,
        info: *mut i32,
    );
    // Complex Hermitian eigenproblem: `a` and `work` are complex*16 (interleaved re, im), `lwork` counts complex
    // elements, `rwork` is real (3n - 2).
    fn zheev_(
        jobz: &c_char,
        uplo: &c_char,
        n: &i32,
        a: *mut f64,
        lda: &i32,
        w: *mut f64,
        work: *mut f64,
        lwork: &i32,
        rwork: *mut f64,
        info: *mut i32,
    );
    fn dsygv_(
        itype: &i32,
        jobz: &c_char,
        uplo: &c_char,
        n: &i32,
        a: *const f64,
        lda: &i32,
        b: *const f64,
        ldb: &i32,
        w: *const f64,
        work: *const f64,
        lwork: &i32,
        info: *mut i32,
    );
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32);
}

/// BLAS's dgemm: `c = alpha op(a) op(b) + beta c`.
#[inline(always)]
pub(crate) fn dgemm(
    cblas_order: CblasOrder,
    cblas_trans_a: CblasTranspose,
    cblas_trans_b: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *const f64,
    ldc: i32,
) {
    unsafe {
        cblas_dgemm(
            cblas_order,
            cblas_trans_a,
            cblas_trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
}

/// BLAS's dgemv: `y = alpha op(a) x + beta y`.
#[inline(always)]
pub(crate) fn dgemv(
    cblas_order: CblasOrder,
    cblas_trans: CblasTranspose,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *const f64,
    incy: i32,
) {
    unsafe {
        cblas_dgemv(
            cblas_order,
            cblas_trans,
            m,
            n,
            alpha,
            a,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        )
    }
}

/// LAPACK's dgetrf: the LU factorization of `a`, in place.
#[inline(always)]
pub(crate) fn dgetrf(m: &i32, n: &i32, a: *const f64, lda: &i32, ipiv: *const i32, info: &mut i32) {
    unsafe { dgetrf_(&m, &n, a, &lda, ipiv, info) }
}

/// LAPACK's dgetri: the inverse from dgetrf's factorization, in place.
#[inline(always)]
pub(crate) fn dgetri(
    n: &i32,
    a: *const f64,
    lda: &i32,
    ipiv: *const i32,
    work: *const f64,
    lwork: &i32,
    info: &mut i32,
) {
    unsafe { dgetri_(&n, a, &lda, ipiv, work, &lwork, info) }
}

/// BLAS's ddot: `x . y`.
#[inline(always)]
pub(crate) fn ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64 {
    let res = unsafe { cblas_ddot(n, x, incx, y, incy) };
    res
}

/// BLAS's daxpy: `y += alpha x`.
#[inline(always)]
pub(crate) fn daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32) {
    unsafe { cblas_daxpy(n, alpha, x, incx, y, incy) }
}

/// LAPACK's dgels: the least-squares solution of `op(a) x = b`, over `b` (`a` overwritten); `info` its status.
#[inline(always)]
pub(crate) fn dgels(
    cblas_trans: CblasTranspose,
    m: &i32,
    n: &i32,
    nrhs: &i32,
    a: *mut f64,
    lda: &i32,
    b: *mut f64,
    ldb: &i32,
    work: *mut f64,
    lwork: &i32,
    info: &mut i32,
) {
    let trans: c_char = match cblas_trans {
        CblasTranspose::CblasNoTrans => 'N' as c_char,
        CblasTranspose::CblasTrans => 'T' as c_char,
        CblasTranspose::CblasConjTrans => 'C' as c_char,
    };
    unsafe { dgels_(&trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info) }
}

/// LAPACK's dgesvd: the singular value decomposition of `a` (overwritten).
#[inline(always)]
pub(crate) fn dgesvd(
    jobu: char,
    jobvt: char,
    m: &i32,
    n: &i32,
    a: *mut f64,
    lda: &i32,
    s: *mut f64,
    u: *mut f64,
    ldu: &i32,
    vt: *mut f64,
    ldvt: &i32,
    work: *mut f64,
    lwork: &i32,
    info: &mut i32,
) {
    // Every output (`a` overwritten, `s`, `u`, `vt`, `work`, `info`) is written by LAPACK, so each is a mutable
    // pointer.
    let jobu = jobu as c_char;
    let jobvt = jobvt as c_char;
    unsafe {
        dgesvd_(
            &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info,
        )
    }
}

/// LAPACK's dsyev: the eigenvalues and, with `jobz = 'V'`, eigenvectors of the symmetric `a`.
#[inline(always)]
pub(crate) fn dsyev(
    jobz: char,
    uplo: &CblasUplo,
    n: &i32,
    a: *const f64,
    lda: &i32,
    w: *const f64,
    work: *const f64,
    lwork: &i32,
    info: &mut i32,
) {
    // `info` is written by LAPACK, so it is passed as a mutable pointer (a shared reference would
    // let the compiler assume it still holds what it held before the call).
    let jobz = jobz as c_char;
    let uplo = match uplo {
        CblasUplo::CblasUpper => 'U' as c_char,
        CblasUplo::CblasLower => 'L' as c_char,
    };
    unsafe { dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info) }
}

/// LAPACK's zheev: eigenvalues (ascending, into `w`) and, with `jobz = 'V'`, eigenvectors (over `a`) of the Hermitian
/// matrix in `a`, interleaved complex (`re, im` per element, column-major, `lda` complex elements per column). `work`
/// holds `lwork` complex elements (`2 lwork` doubles), `rwork` at least `max(1, 3n - 2)` doubles. `lwork = -1` is a
/// scratch query: the optimal `lwork` comes back in `work[0]`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn zheev(
    jobz: char,
    uplo: &CblasUplo,
    n: &i32,
    a: *mut f64,
    lda: &i32,
    w: *mut f64,
    work: *mut f64,
    lwork: &i32,
    rwork: *mut f64,
    info: &mut i32,
) {
    let jobz = jobz as c_char;
    let uplo = match uplo {
        CblasUplo::CblasUpper => 'U' as c_char,
        CblasUplo::CblasLower => 'L' as c_char,
    };
    unsafe { zheev_(&jobz, &uplo, n, a, lda, w, work, lwork, rwork, info) }
}

/// LAPACK's dsygv: the generalized symmetric eigenproblem of `a` and `b`.
#[inline(always)]
pub(crate) fn dsygv(
    itype: &i32,
    jobz: char,
    uplo: &CblasUplo,
    n: &i32,
    a: *const f64,
    lda: &i32,
    b: *const f64,
    ldb: &i32,
    w: *const f64,
    work: *const f64,
    lwork: &i32,
    info: &mut i32,
) {
    // `info` is written by LAPACK: a mutable pointer, as for dsyev.
    let jobz = jobz as c_char;
    let uplo = match uplo {
        CblasUplo::CblasUpper => 'U' as c_char,
        CblasUplo::CblasLower => 'L' as c_char,
    };
    unsafe {
        dsygv_(
            &itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, info,
        )
    }
}
