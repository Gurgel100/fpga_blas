//
// Created by Pascal on 09.05.2018.
//

#ifndef BLAS_HLS_MACC_H
#define BLAS_HLS_MACC_H

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "Core.h"
#include <assert.h>

#define NUM_PART_SUMS	16

using hlslib::Stream;

template <class Data_t>
class DotProductBase {
public:
	DotProductBase(Stream<Data_t> &out) : out(out) {
		#pragma HLS INLINE
	}

protected:
	Stream<Data_t> &out;
};

template <class Data_t>
class DotProduct : public DotProductBase<Data_t> {
public:
	DotProduct(Stream<Data_t> &out) : DotProductBase<Data_t>(out), pipeIntermediate("dot.intermediatePipe", NUM_PART_SUMS) {
		#pragma HLS INLINE
	}

	void calc(const size_t N, Stream<Data_t> &X, Stream<Data_t> &Y) {
		#pragma HLS INLINE
		HLSLIB_DATAFLOW_FUNCTION(Core::macc<NUM_PART_SUMS>, N, X, Y, pipeIntermediate);
		HLSLIB_DATAFLOW_FUNCTION(Core::accumulate<NUM_PART_SUMS>, pipeIntermediate, this->out);
	}

private:
	Stream<Data_t> pipeIntermediate;
};

template <class Data_t>
class DotProductInterleaved : public DotProductBase<Data_t> {
public:
	DotProductInterleaved(Stream<Data_t> &out) : DotProductBase<Data_t>(out) {
		#pragma HLS INLINE
	}

	void calc(const size_t N, const size_t num_prods, Stream<Data_t> X[], Stream<Data_t> Y[]) {
		#pragma HLS INLINE
		HLSLIB_DATAFLOW_FUNCTION(calc_internal, N, num_prods, X, Y, this->out);
	}

private:
	static void calc_internal(const size_t N, const size_t num_prods, Stream<Data_t> X[], Stream<Data_t> Y[], Stream<Data_t> &out) {
		#pragma HLS INLINE
		assert(num_prods % NUM_PART_SUMS == 0);
		const size_t num_partials = num_prods / NUM_PART_SUMS;
		Data_t sums[NUM_PART_SUMS];
		loop_dot_interleaved_part:
		for (size_t part = 0; part < num_partials; ++part) {
			loop_dot_interleaved_N:
			for (size_t i = 0; i < N; ++i) {
				loop_dot_interleaved_s:
				for (int s = 0; s < NUM_PART_SUMS; ++s) {
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
};

#endif //BLAS_HLS_MACC_H
