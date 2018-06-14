//
// Created by Pascal on 09.05.2018.
//

#ifndef BLAS_HLS_MACC_H
#define BLAS_HLS_MACC_H

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "Core.h"
#include <assert.h>

#define NUM_PART_SUMS	11

using hlslib::Stream;

void dot(const size_t N, Stream<Data_t> &X, Stream<Data_t> &Y, Stream<Data_t> &out) {
	#pragma HLS INLINE
	Stream<Data_t> pipeIntermediate("dot.intermediatePipe", NUM_PART_SUMS);
	HLSLIB_DATAFLOW_FUNCTION(Core::macc<NUM_PART_SUMS>, N, X, Y, pipeIntermediate);
	HLSLIB_DATAFLOW_FUNCTION(Core::accumulate<NUM_PART_SUMS>, pipeIntermediate, out);
}

void dot_interleaved(const size_t N, const size_t num_prods, Stream<Data_t> X[], Stream<Data_t> Y[], Stream<Data_t> &out) {
	#pragma HLS INLINE
	assert(num_prods % NUM_PART_SUMS == 0);
	const size_t num_partials = num_prods / NUM_PART_SUMS;
	Data_t sums[NUM_PART_SUMS];
	loop_dot_interleaved_part: for (size_t part = 0; part < num_partials; ++part) {
		loop_dot_interleaved_N: for (size_t i = 0; i < N; ++i) {
			loop_dot_interleaved_s: for (int s = 0; s < NUM_PART_SUMS; ++s) {
				#pragma HLS PIPELINE II=1
				#pragma HLS LOOP_FLATTEN
				const size_t current_sum = part * NUM_PART_SUMS + s;
				Core::macc_step<NUM_PART_SUMS>(X[current_sum], Y[current_sum], sums, s, i, N);

				if (i == N - 1) {
					//In the last round we push to the output stream
					out.Push(sums[s]);
				}
			}
		}
	}
}

#endif //BLAS_HLS_MACC_H
