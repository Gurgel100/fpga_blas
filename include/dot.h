//
// Created by Pascal on 09.05.2018.
//

#ifndef BLAS_HLS_MACC_H
#define BLAS_HLS_MACC_H

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"

using hlslib::Stream;

template <size_t NUM_PART_SUMS>
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

template <size_t NUM_PART_SUMS>
void accumulate(Stream<Data_t> &in, Stream<Data_t> &out) {
	Data_t result = 0;
	for (size_t i = 0; i < NUM_PART_SUMS; ++i) {
		result += in.Pop();
	}
	out.Push(result);
}

template <size_t n_prod, size_t my_offset>
void read(const size_t N, Stream<Data_t> &in, Stream<Data_t> &out) {
	#pragma HLS INLINE
	for (int n = 0; n < N; ++n)
	{
		for (size_t i = 0; i < n_prod; ++i)
		{
			#pragma HLS UNROLL
			if (n == my_offset) {
				out.Push(in.Pop());
			}
		}
	}
}

template <size_t NUM_PART_SUMS, size_t n_prod = 1, size_t my_offset = 0>
void dot(const size_t N, Stream<Data_t> &in_X, Stream<Data_t> &in_Y, Stream<Data_t> &out) {
	#pragma HLS INLINE
	for (size_t i = 0; i < n_prod; ++i) {
		#pragma HLS UNROLL
		Stream<Data_t> pipe_x, pipe_y, pipeIntermediate;
		read<n_prod, my_offset>(N, in_X, pipe_x);
		read<n_prod, my_offset>(N, in_Y, pipe_y);
		macc<NUM_PART_SUMS>(N, pipe_x, pipe_y, pipeIntermediate);
		accumulate<NUM_PART_SUMS>(pipeIntermediate, out);
	}
}

#endif //BLAS_HLS_MACC_H
