#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "hlslib/TreeReduce.h"
#include "hlslib/Operators.h"
#include "DotProduct.h"
#include "dot.h"

using hlslib::Stream;

void blas_dot(const size_t N, Data_t const memoryIn_X[], const int incX, Data_t const memoryIn_Y[], const int incY, Data_t memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=incY bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	HLSLIB_DATAFLOW_INIT();

	FBLAS::DotProduct<Data_t> dotProduct(N, incX, incY);
	dotProduct.getReaderX().readFromMemory<true>(memoryIn_X);
	dotProduct.getReaderY().readFromMemory<true>(memoryIn_Y);
	dotProduct.calc<true>();
	dotProduct.getWriter().writeToMemory<true>(memoryOut);

	HLSLIB_DATAFLOW_FINALIZE();
}

void blas_dot_multiple(const size_t N, const size_t n_prod, Data_t const memoryIn_X[], Data_t const memoryIn_Y[], Data_t memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=n_prod bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	HLSLIB_DATAFLOW_INIT();

	FBLAS::DotProductInterleaved<Data_t> dotProduct(N, n_prod);
	dotProduct.getReaderX().readFromMemory<true>(memoryIn_X);
	dotProduct.getReaderY().readFromMemory<true>(memoryIn_Y);
	dotProduct.calc<true>();
	dotProduct.getWriter().writeToMemory<true>(memoryOut);

	HLSLIB_DATAFLOW_FINALIZE();
}
