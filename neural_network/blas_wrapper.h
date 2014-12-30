#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

#include <cblas.h>

namespace blas {


inline void gemm(double* Y, const double* A, bool transposeA, int rowsA, int colsA, const double* B, bool transposeB, int rowsB, int colsB, double alpha = 1.0, double beta = 0.0)
{
	cblas_dgemm(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, transposeB ? CblasTrans : CblasNoTrans, transposeA ? colsA : rowsA, transposeB ? rowsB : colsB, transposeB ? colsB : rowsB, alpha, A, colsA, B, colsB, beta, Y, transposeB ? rowsB : colsB);
}

inline void gemm(float* Y, const float* A, bool transposeA, int rowsA, int colsA, const float* B, bool transposeB, int rowsB, int colsB, float alpha = 1.0f, float beta = 0.0f)
{
	cblas_sgemm(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, transposeB ? CblasTrans : CblasNoTrans, transposeA ? colsA : rowsA, transposeB ? rowsB : colsB, transposeB ? colsB : rowsB, alpha, A, colsA, B, colsB, beta, Y, transposeB ? rowsB : colsB);
}

inline void gemv(double* Y, const double* A, bool transposeA, int rowsA, int colsA, const double* x, double alpha = 1.0, double beta = 0.0)
{
	cblas_dgemv(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, rowsA, colsA, alpha, A, colsA, x, 1, beta, Y, 1);
}

inline void gemv(float* Y, const float* A, bool transposeA, int rowsA, int colsA, const float* x, float alpha = 1.0, float beta = 0.0)
{
	cblas_sgemv(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, rowsA, colsA, alpha, A, colsA, x, 1, beta, Y, 1);
}

inline void ger(double* Y, const double * A, int rowsA, const double* X, int rowsX, double alpha = 1.0)
{
	cblas_dger(CblasRowMajor, rowsA, rowsX, alpha, A, 1, X, 1, Y, rowsX);
}

inline void ger(float* Y, const float * A, int rowsA, const float* X, int rowsX, float alpha = 1.0)
{
	cblas_sger(CblasRowMajor, rowsA, rowsX, alpha, A, 1, X, 1, Y, rowsX);
}

inline double norm2(double* X, int n, int stride = 1)
{
	return cblas_dnrm2(n, X, stride);
}

inline float norm2(float* X, int n, int stride = 1)
{
	return cblas_snrm2(n, X, stride);
}

inline void scale(float alpha, float* X, int n, int stride = 1)
{
	return cblas_sscal(n, alpha, X, stride);
}

inline void scale(double alpha, double* X, int n, int stride = 1)
{
	return cblas_dscal(n, alpha, X, stride);
}
}

#endif
