#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "hlslib/TreeReduce.h"
#include "hlslib/Operators.h"
#include "Kernel.h"
#include "../include/dot.h"
using hlslib::Stream;

#define NUM_PART_SUMS	11

void ReadMemory(Data_t const memory[], Stream<Data_t> &pipe, const int N) {
  for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
    pipe.Push(memory[i]);
  }
}

void WriteMemory(Stream<Data_t> &pipe, Data_t memory[], const int N) {
  for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
    memory[i] = pipe.Pop();
  }
}

void blas_xdot(Data_t const memoryIn_X[], Data_t const memoryIn_Y[], Data_t memoryOut[], const size_t N) {
  #pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
  #pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
  #pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
  #pragma HLS INTERFACE s_axilite port=N bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  #pragma HLS DATAFLOW

  Stream<Data_t> pipeIn_X, pipeIn_Y, pipeOut;
  HLSLIB_DATAFLOW_INIT();
  HLSLIB_DATAFLOW_FUNCTION(ReadMemory, memoryIn_X, pipeIn_X, N);
  HLSLIB_DATAFLOW_FUNCTION(ReadMemory, memoryIn_Y, pipeIn_Y, N);
  HLSLIB_DATAFLOW_FUNCTION(dot<NUM_PART_SUMS>, N, pipeIn_X, pipeIn_Y, pipeOut);
  HLSLIB_DATAFLOW_FUNCTION(WriteMemory, pipeOut, memoryOut, 1);
  HLSLIB_DATAFLOW_FINALIZE();
}
