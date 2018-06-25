//
// Created by Pascal on 20.05.2018.
//

#ifndef BLAS_HLS_CORE_H
#define BLAS_HLS_CORE_H

#include <hlslib/Stream.h>

namespace Core {

	using hlslib::Stream;

	template <class T, size_t num_partial_sums>
	static void
	macc_step(Stream<T> &in_X, Stream<T> &in_Y, T part_sums[num_partial_sums], const size_t index,
	          const size_t round, const size_t currentN, const size_t N) {
		#pragma HLS INLINE
		T sum = (round == 0) ? 0 : part_sums[index];
		T x = currentN < N ? in_X.Pop() : 0;
		T y = currentN < N ? in_Y.Pop() : 0;
		part_sums[index] = sum + x * y;
		#pragma HLS DEPENDENCE variable=part_sums inter false
	}

	template <class T, size_t num_partial_sums>
	static void macc(const size_t N, Stream<T> &in_X, Stream<T> &in_Y, Stream<T> &out) {
		#pragma HLS INLINE
		T part_sums[num_partial_sums];

		const size_t rounds = N / num_partial_sums;
		macc_round_loop:
		for (size_t round = 0; round < rounds + 1; ++round) {
			macc_partials_loop:
			for (size_t i = 0; i < num_partial_sums; ++i) {
				#pragma HLS PIPELINE II=1
				#pragma HLS LOOP_FLATTEN
				macc_step<T, num_partial_sums>(in_X, in_Y, part_sums, i, round, round * num_partial_sums + i, N);

				if (round == rounds) {
					const T sum = (i < N) ? part_sums[i] : 0;
					out.Push(sum);
				}
			}
		}
	}

	template <class T, size_t num_partial_sums>
	static void accumulate(Stream<T> &in, Stream<T> &out) {
		#pragma HLS INLINE
		T result = 0;
		accumulate_loop:
		for (size_t i = 0; i < num_partial_sums; ++i) {
			#pragma HLS PIPELINE II=10
			result += in.Pop();
		}
		out.Push(result);
	}
}

#endif //BLAS_HLS_CORE_H
