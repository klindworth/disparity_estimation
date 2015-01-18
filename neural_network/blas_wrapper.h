/*
Copyright (c) 2014, Kai Klindworth
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
