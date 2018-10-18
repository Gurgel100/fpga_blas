//
// Created by Pascal on 20.05.2018.
//

#ifndef BLAS_HLS_CORE_H
#define BLAS_HLS_CORE_H

#include <hlslib/Stream.h>
#include <hlslib/DataPack.h>
#include <hlslib/TreeReduce.h>
#include <array>

namespace FBLAS {
	namespace Core {

		using hlslib::Stream;

		template <class T, size_t num_partial_sums, class U = T>
		static void
		macc_step(Stream<T> &in_X, Stream<T> &in_Y, T part_sums[num_partial_sums], const size_t index,
		          const size_t round, const size_t currentN, const size_t N) {
			#pragma HLS INLINE
			const T zero = static_cast<U>(0);
			T sum = (round == 0) ? zero : part_sums[index];
			T x = currentN < N ? in_X.Pop() : zero;
			T y = currentN < N ? in_Y.Pop() : zero;
			part_sums[index] = sum + x * y;
			#pragma HLS DEPENDENCE variable=part_sums inter false
		}

		template <class T, size_t num_partial_sums, class U = T>
		static void macc(const size_t N, Stream<T> &in_X, Stream<T> &in_Y, Stream<T> &out) {
			#pragma HLS INLINE
			T part_sums[num_partial_sums];
			const T zero = static_cast<U>(0);

			const size_t rounds = N / num_partial_sums;
			macc_round_loop:
			for (size_t round = 0; round < rounds + 1; ++round) {
				macc_partials_loop:
				for (size_t i = 0; i < num_partial_sums; ++i) {
					#pragma HLS PIPELINE II=1
					#pragma HLS LOOP_FLATTEN
					macc_step<T, num_partial_sums, U>(in_X, in_Y, part_sums, i, round, round * num_partial_sums + i, N);

					if (round == rounds) {
						const T sum = (i < N) ? part_sums[i] : zero;
						out.Push(sum);
					}
				}
			}
		}

		template <class T, size_t num_partial_sums, class U = T, size_t num_parallel_sums = 1>
		static void accumulate(Stream<T> &in, Stream<T> &out, U init = 0) {
			#pragma HLS INLINE
			std::array<T, num_parallel_sums> result;

			accumulate_loop:
			for (size_t i = 0; i < num_partial_sums; ++i) {
				for (size_t j = 0; j < num_parallel_sums; ++j) {
					#pragma HLS PIPELINE II=1
					#pragma HLS LOOP_FLATTEN
					T prev = i == 0 ? T(init) : result[j];
					result[j] = prev + in.Pop();

					if(i == num_partial_sums - 1) {
						out.Push(result[j]);
					}
				}
			}
		}

		template <class T, int width, class Operator>
		static void reduceDataPack(Stream<hlslib::DataPack<T, width>> &in, Stream<T> &out) {
			#pragma HLS INLINE
			T tmp[width];
			auto r = in.Pop();
			r.Unpack(tmp);
			T result = hlslib::TreeReduce<T, Operator, width>(tmp);
			out.Push(result);
		}
	}
}

#endif //BLAS_HLS_CORE_H
