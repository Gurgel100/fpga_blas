//
// Created by stpascal on 26.06.18.
//

#ifndef BLAS_HLS_GEMV_H
#define BLAS_HLS_GEMV_H

#include "Matrix.h"
#include "Config.h"

extern "C" {
	void blas_gemv(const FBLAS::MatrixVectorMultiplication<Data_t>::Col_t memoryIn_A[], const FBLAS::MatrixVectorMultiplication<Data_t>::Col_t memoryIn_X[],
			Data_t memoryOut_Y[], size_t N, size_t M);

	void blas_gemv_transposed(const FBLAS::MatrixVectorMultiplicationTransposed<Data_t>::Col_t memoryIn_A[], const Data_t memoryIn_X[],
			FBLAS::MatrixVectorMultiplicationTransposed<Data_t>::Col_t memoryOut[], size_t N, size_t M);
};

#endif //BLAS_HLS_GEMV_H
