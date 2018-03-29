#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "Kernel.h"
using hlslib::Stream;

void ReadMemory(Data_t const memory[], Stream<Data_t> &pipe, const int N) {
  for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
    pipe.Push(memory[i]);
  }
}

void AddOneCompute(Stream<Data_t> &in, Stream<Data_t> &out, const int N) {
  for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
    out.Push(in.Pop() + 1);
  }
}

void WriteMemory(Stream<Data_t> &pipe, Data_t memory[], const int N) {
  for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
    memory[i] = pipe.Pop();
  }
}

void AddOne(Data_t const memoryIn[], Data_t memoryOut[], const int N) {
  #pragma HLS INTERFACE m_axi port=memoryIn offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem1
  #pragma HLS INTERFACE s_axilite port=memoryIn bundle=control
  #pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
  #pragma HLS INTERFACE s_axilite port=N bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  Stream<Data_t> pipeIn, pipeOut;
  HLSLIB_DATAFLOW_INIT();
  HLSLIB_DATAFLOW_FUNCTION(ReadMemory, memoryIn, pipeIn, N);
  HLSLIB_DATAFLOW_FUNCTION(AddOneCompute, pipeIn, pipeOut, N);
  HLSLIB_DATAFLOW_FUNCTION(WriteMemory, pipeOut, memoryOut, N);
  HLSLIB_DATAFLOW_FINALIZE();
}
