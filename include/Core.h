//
// Created by Pascal on 20.05.2018.
//

#ifndef BLAS_HLS_CORE_H
#define BLAS_HLS_CORE_H

#include "hlslib/Stream.h"

using hlslib::Stream;

class Core {
public:
	Core() = delete;
	Core(Core &&) = delete;
	Core(const Core &) = delete;
	Core &operator=(Core const &) = delete;
	Core &operator=(Core &&) = delete;

	template <size_t num_partial_sums>
	static void macc_step(Stream<Data_t> &in_X, Stream<Data_t> &in_Y, Data_t part_sums[num_partial_sums], const size_t index, const size_t round) {
		#pragma HLS INLINE
		#pragma HLS PIPELINE II=1
		Data_t sum = (round == 0) ? 0 : part_sums[index];
		Data_t x = in_X.Pop();
		Data_t y = in_Y.Pop();
		part_sums[index] = sum + x * y;
		#pragma HLS DEPENDENCE variable=part_sums inter false
	}

	template <size_t num_partial_sums>
	static void macc(const size_t N, Stream<Data_t> &in_X, Stream<Data_t> &in_Y, Stream<Data_t> &out) {
		#pragma HLS INLINE
		Data_t part_sums[num_partial_sums];

		const size_t rounds = N / num_partial_sums;
		macc_loop: for (size_t round = 0; round < rounds; ++round) {
			#pragma HLS PIPELINE II=1
			for (size_t i = 0; i < num_partial_sums; ++i) {
				macc_step<num_partial_sums>(in_X, in_Y, part_sums, i, round);
			}
		}
		for (size_t i = 0; i < num_partial_sums; ++i) {
			Data_t sum = 0;
			if (i < N) {
				sum = part_sums[i];
			}
			out.Push(sum);
		}
	}

	template <size_t num_partial_sums>
	static void accumulate(Stream<Data_t> &in, Stream<Data_t> &out) {
		#pragma HLS INLINE
		Data_t result = 0;
		for (size_t i = 0; i < num_partial_sums; ++i) {
			#pragma HLS PIPELINE II=1
			result += in.Pop();
		}
		out.Push(result);
	}
};

#endif //BLAS_HLS_CORE_H