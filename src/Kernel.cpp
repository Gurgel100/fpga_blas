#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "hlslib/TreeReduce.h"
#include "hlslib/Operators.h"
#include "Kernel.h"
#include "dot.h"
using hlslib::Stream;

void ReadMemory_multiple(Data_t const memory[], Stream<Data_t> pipe[], const size_t N, const size_t n_vectors) {
	for (size_t i = 0; i < N; ++i) {
		#pragma HLS PIPELINE II=1
		for (size_t j = 0; j < n_vectors; ++j) {
			pipe[j].Push(memory[j * N + i]);
		}
	}
}

void ReadMemory(Data_t const memory[], Stream<Data_t> &pipe, const size_t N, const int increment) {
	size_t currentLoc = 0;
	for (size_t i = 0; i < N; ++i) {
		#pragma HLS PIPELINE II=1
		pipe.Push(memory[currentLoc]);
		currentLoc += increment;
	}
}

void WriteMemory(Stream<Data_t> &pipe, Data_t memory[], const size_t N, const int increment) {
	size_t currentLoc = 0;
	for (size_t i = 0; i < N; ++i) {
		#pragma HLS PIPELINE II=1
		memory[currentLoc] = pipe.Pop();
		currentLoc += increment;
	}
}

void blas_dot(Data_t const memoryIn_X[], const int incX, Data_t const memoryIn_Y[], const int incY, Data_t memoryOut[], const size_t N) {
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
	HLSLIB_DATAFLOW_FUNCTION(ReadMemory, memoryIn_X, pipeIn_X, N, incX);
	HLSLIB_DATAFLOW_FUNCTION(ReadMemory, memoryIn_Y, pipeIn_Y, N, incY);
	dotProduct.calc(N, pipeIn_X, pipeIn_Y);
	HLSLIB_DATAFLOW_FUNCTION(WriteMemory, pipeOut, memoryOut, 1, 1);
	HLSLIB_DATAFLOW_FINALIZE();
}

void blas_dot_multiple(Data_t const memoryIn_X[], Data_t const memoryIn_Y[], Data_t memoryOut[], const size_t N, const size_t n_prod) {
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

	Stream<Data_t> pipeIn_X[n_prod], pipeIn_Y[n_prod], pipeOut;
	DotProductInterleaved<Data_t> dotProduct(pipeOut);
	HLSLIB_DATAFLOW_INIT();
	HLSLIB_DATAFLOW_FUNCTION(ReadMemory_multiple, memoryIn_X, pipeIn_X, N, n_prod);
	HLSLIB_DATAFLOW_FUNCTION(ReadMemory_multiple, memoryIn_Y, pipeIn_Y, N, n_prod);
	dotProduct.calc(N, n_prod, pipeIn_X, pipeIn_Y);
	HLSLIB_DATAFLOW_FUNCTION(WriteMemory, pipeOut, memoryOut, n_prod, 1);
	HLSLIB_DATAFLOW_FINALIZE();
}
