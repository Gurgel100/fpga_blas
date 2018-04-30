#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "hlslib/TreeReduce.h"
#include "hlslib/Operators.h"
#include "Kernel.h"
using hlslib::Stream;

#define NUM_PART_SUMS	11

void ReadMemory(Data_t const memory[], Stream<Data_t> &pipe, const int N) {
  for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
    pipe.Push(memory[i]);
  }
}

void macc(const size_t N, Stream<Data_t> &in_X, Stream<Data_t> &in_Y, Stream<Data_t> &out) {
	Data_t part_sums[NUM_PART_SUMS];

	for (size_t i = 0; i < NUM_PART_SUMS; ++i) {
		#pragma HLS PIPELINE II=1
		part_sums[i] = 0;
	}
	size_t current_part = 0;
	macc_Calc: for (size_t i = 0; i < N; ++i) {
		#pragma HLS PIPELINE II=1
		/*if (i < NUM_PART_SUMS)
			part_sums[i] = 0;*/
		Data_t x = in_X.Pop();
		Data_t y = in_Y.Pop();
		part_sums[current_part] += x * y;
		#pragma HLS DEPENDENCE variable=part_sums inter false
		if (++current_part >= NUM_PART_SUMS)
			current_part = 0;
	}
	macc_Write: for (size_t i = 0; i < NUM_PART_SUMS; ++i) {
		#pragma HLS PIPELINE II=1
		out.Push(part_sums[i]);
	}
}

void accumulate(Stream<Data_t> &in, Stream<Data_t> &out) {
	//Data_t result = 0;
	Data_t arr[NUM_PART_SUMS];
	Accumulate_load: for (size_t i = 0; i < NUM_PART_SUMS; ++i) {
		#pragma HLS PIPELINE II=1
		arr[i] = in.Pop();
	}
	Data_t result = hlslib::TreeReduce<Data_t, hlslib::op::Add<Data_t>, NUM_PART_SUMS>(arr);
	/*for (size_t i = 0; i < NUM_PART_SUMS; ++i) {
		result += in.Pop();
	}*/
	out.Push(result);
}

void WriteMemory(Stream<Data_t> &pipe, Data_t memory[], const int N) {
  for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
    memory[i] = pipe.Pop();
  }
}

void blas_xdot(Data_t const memoryIn_X[], Data_t const memoryIn_Y[], Data_t memoryOut[], const size_t N) {
  #pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem1
  #pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
  #pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
  #pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
  #pragma HLS INTERFACE s_axilite port=N bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  Stream<Data_t> pipeIn_X, pipeIn_Y, pipeOut, pipeIntermediate;
  HLSLIB_DATAFLOW_INIT();
  HLSLIB_DATAFLOW_FUNCTION(ReadMemory, memoryIn_X, pipeIn_X, N);
  HLSLIB_DATAFLOW_FUNCTION(ReadMemory, memoryIn_Y, pipeIn_Y, N);
  HLSLIB_DATAFLOW_FUNCTION(macc, N, pipeIn_X, pipeIn_Y, pipeIntermediate);
  HLSLIB_DATAFLOW_FUNCTION(accumulate, pipeIntermediate, pipeOut);
  HLSLIB_DATAFLOW_FUNCTION(WriteMemory, pipeOut, memoryOut, 1);
  HLSLIB_DATAFLOW_FINALIZE();
}
