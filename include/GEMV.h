//
// Created by stpascal on 26.06.18.
//

#ifndef BLAS_HLS_GEMV_H
#define BLAS_HLS_GEMV_H

#include "Matrix.h"
#include "Config.h"

#ifndef GEMV_SIZE_ROWCHUNK
#define GEMV_SIZE_ROWCHUNK   16
#endif

#ifndef GEMV_SIZE_COLCHUNK
#define GEMV_SIZE_COLCHUNK   16
#endif

#ifndef GEMV_SIZE_COLUMN
#define GEMV_SIZE_COLUMN     GEMV_SIZE_COLCHUNK
#endif

typedef FBLAS::MatrixVectorMultiplication<Data_t, GEMV_SIZE_ROWCHUNK, GEMV_SIZE_COLCHUNK, GEMV_SIZE_COLUMN> MatrixVectorMultiplication_t;
typedef FBLAS::MatrixVectorMultiplicationTransposed<Data_t, GEMV_SIZE_ROWCHUNK, GEMV_SIZE_COLCHUNK, GEMV_SIZE_COLUMN> MatrixVectorMultiplicationTransposed_t ;

extern "C" {
	void blas_gemv(const MatrixVectorMultiplication_t::Col_t memoryIn_A[], const MatrixVectorMultiplication_t::Col_t memoryIn_X[],
			Data_t memoryOut_Y[], size_t N, size_t M);

	void blas_gemv_transposed(const MatrixVectorMultiplicationTransposed_t::Col_t memoryIn_A[], const Data_t memoryIn_X[],
	                          MatrixVectorMultiplicationTransposed_t::Col_t memoryOut[], size_t N, size_t M);
};

#endif //BLAS_HLS_GEMV_H
