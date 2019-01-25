//
// Created by stpascal on 21.06.18.
//

#ifndef BLAS_HLS_SCALAR_H
#define BLAS_HLS_SCALAR_H

#include <hlslib/Stream.h>
#include <hlslib/Simulation.h>
#include <hlslib/DataPack.h>
#include "Memory.h"

namespace FBLAS {

	using hlslib::Stream;

	template<class T, size_t width = 1>
	class Scalar {
	public:
		using Data_t = typename std::conditional<width == 1, T, hlslib::DataPack<T, width>>::type;

		Scalar(const size_t N, Stream<Data_t> &inX, Stream<Data_t> &out) : N(N), inX(inX), out(out) {
			#pragma HLS INLINE
		}

		Memory::MemoryReader<Data_t> getReaderX() {
			#pragma HLS INLINE
			return Memory::MemoryReader<Data_t>(inX, N / width);
		}

		Memory::MemoryWriter<Data_t> getWriterX() {
			#pragma HLS INLINE
			return Memory::MemoryWriter<Data_t>(out, N / width);
		}

		template <bool dataflow = false>
		void calc(Stream<T> &factor) {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(calcStream, N, factor, inX, out);
			} else {
				calcStream(N, factor, inX, out);
			}
		}

		template <bool dataflow = false>
		void calc(const T factor) {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(calc, N, factor, inX, out);
			} else {
				calc(N, factor, inX, out);
			}
		}

	private:
		size_t N;
		Stream<Data_t> &inX, &out;

		static void calc(const size_t N, const T factor, Stream<Data_t> &inX, Stream<Data_t> &out) {
			#pragma HLS INLINE
			Scalar_calc_loop:
			for (size_t i = 0; i < N / width; ++i) {
				#pragma HLS PIPELINE II=1
				out.Push(inX.Pop() * factor);
			}
		}

		static void calcStream(const size_t N, Stream<T> &factor, Stream<Data_t> &inX, Stream<Data_t> &out) {
			#pragma HLS INLINE
			Scalar_calc_loop:
			T factor_val;
			for (size_t i = 0; i < N / width; ++i) {
				#pragma HLS PIPELINE II=1
				if (i == 0) {
					factor_val = factor.Pop();
				}
				out.Push(inX.Pop() * factor_val);
			}
		}
	};
}


#endif //BLAS_HLS_SCALAR_H
