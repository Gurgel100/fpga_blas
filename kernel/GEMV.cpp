#include <hlslib/Stream.h>
#include <hlslib/Simulation.h>
#include <hlslib/TreeReduce.h>
#include <hlslib/Operators.h>
#include <hlslib/DataPack.h>
#include "Matrix.h"
#include "Memory.h"
#include "GEMV.h"

using hlslib::Stream;

/*
 * +-----m-----+ +-----+   +-----+
 * |           | |     |   |     |
 * |           | |     |   |     |
 * n     A     | m  x  | = n  y  |
 * |           | |     |   |     |
 * |           | |     |   |     |
 * +-----------+ +-----+   +-----+
 */

void blas_gemv(const MatrixVectorMultiplication_t::Col_t memoryIn_A[], const MatrixVectorMultiplication_t::Col_t memoryIn_X[],
		Data_t memoryOut_Y[], const size_t N, const size_t M) {
	#pragma HLS INTERFACE m_axi port=memoryIn_A offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut_Y offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_A bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=M bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	Stream<MatrixVectorMultiplication_t::Col_t> inA("inA"), inX("inX");
	Stream<Data_t> out("out");

	HLSLIB_DATAFLOW_INIT();

	MatrixVectorMultiplication_t matrixVectorMultiplication(N, M, inA, inX, out);
	matrixVectorMultiplication.getReaderX().readFromMemory<true>(memoryIn_X);
	matrixVectorMultiplication.getReaderA().readFromMemory<true>(memoryIn_A);
	matrixVectorMultiplication.calc<true>();
	matrixVectorMultiplication.getWriter().writeToMemory<true>(memoryOut_Y);

	HLSLIB_DATAFLOW_FINALIZE();
}

void blas_gemv_transposed(const MatrixVectorMultiplicationTransposed_t::Col_t memoryIn_A[], const Data_t memoryIn_X[],
                          MatrixVectorMultiplicationTransposed_t::Col_t memoryOut[], const size_t N, const size_t M) {
	#pragma HLS INTERFACE m_axi port=memoryIn_A offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_A bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=M bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	Stream<MatrixVectorMultiplicationTransposed_t::Col_t> inA("inA"), out("out");
	Stream<Data_t> inX("inX");

	HLSLIB_DATAFLOW_INIT();

	MatrixVectorMultiplicationTransposed_t matrixVectorMultiplication(N, M, inA, inX, out);
	matrixVectorMultiplication.getReaderX().readFromMemory<true>(memoryIn_X);
	matrixVectorMultiplication.getReaderA().readFromMemory<true>(memoryIn_A);
	matrixVectorMultiplication.calc<true>();
	matrixVectorMultiplication.getWriter().writeToMemory<true>(memoryOut);

	HLSLIB_DATAFLOW_FINALIZE();
}
