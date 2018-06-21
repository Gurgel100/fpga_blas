#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "hlslib/TreeReduce.h"
#include "hlslib/Operators.h"
#include "DotProduct.h"
#include "Memory.h"
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

	Stream<Data_t> pipeIn_X("pipeIn_X"), pipeIn_Y("pipeIn_Y"), pipeOut("pipeOut");
	DotProduct<Data_t> dotProduct(pipeOut);
	HLSLIB_DATAFLOW_INIT();
	HLSLIB_DATAFLOW_FUNCTION(Memory::ReadMemory<Data_t>, memoryIn_X, pipeIn_X, N, incX);
	HLSLIB_DATAFLOW_FUNCTION(Memory::ReadMemory<Data_t>, memoryIn_Y, pipeIn_Y, N, incY);
	dotProduct.calc(N, pipeIn_X, pipeIn_Y);
	HLSLIB_DATAFLOW_FUNCTION(Memory::WriteMemory<Data_t>, pipeOut, memoryOut, 1, 1);
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

	Stream<Data_t> pipeIn_X("pipeIn_X"), pipeIn_Y("pipeIn_Y"), pipeOut("pipeOut");
	DotProductInterleaved<Data_t> dotProduct(pipeOut);
	HLSLIB_DATAFLOW_INIT();
	// FIXME: hack to overcome the problem of HLSLIB_DATAFLOW_FUNCTION with templated functions
	#ifdef HLSLIB_SYNTHESIS
		Memory::ReadMemoryInterleaved<Data_t, DotProductInterleaved<Data_t>::partialSums>(memoryIn_X, pipeIn_X, N, n_prod);
		Memory::ReadMemoryInterleaved<Data_t, DotProductInterleaved<Data_t>::partialSums>(memoryIn_Y, pipeIn_Y, N, n_prod);
	#else
		HLSLIB_DATAFLOW_FUNCTION(Memory::ReadMemoryInterleaved<Data_t, DotProductInterleaved<Data_t>::partialSums>, memoryIn_X, pipeIn_X, N, n_prod);
		HLSLIB_DATAFLOW_FUNCTION(Memory::ReadMemoryInterleaved<Data_t, DotProductInterleaved<Data_t>::partialSums>, memoryIn_Y, pipeIn_Y, N, n_prod);
	#endif
	dotProduct.calc(N, n_prod, pipeIn_X, pipeIn_Y);
	HLSLIB_DATAFLOW_FUNCTION(Memory::WriteMemory<Data_t>, pipeOut, memoryOut, n_prod, 1);
	HLSLIB_DATAFLOW_FINALIZE();
}
