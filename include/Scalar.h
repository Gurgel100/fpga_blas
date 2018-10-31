//
// Created by stpascal on 21.06.18.
//

#ifndef BLAS_HLS_SCALAR_H
#define BLAS_HLS_SCALAR_H

#include <hlslib/Stream.h>
#include <hlslib/Simulation.h>
#include "Memory.h"

namespace FBLAS {

	using hlslib::Stream;

	template<class T>
	class Scalar {
	public:
		Scalar(const size_t N, Stream<T> &inX, Stream<T> &out) : N(N), inX(inX), out(out) {
			#pragma HLS INLINE
		}

		Memory::MemoryReader<T> getReaderX() {
			#pragma HLS INLINE
			return Memory::MemoryReader<T>(inX, N);
		}

		Memory::MemoryWriter<T> getWriterX() {
			#pragma HLS INLINE
			return Memory::MemoryWriter<T>(out, N);
		}

		template <bool dataflow = false>
		void calc(Stream<T> &factor) {
			#pragma HLS INLINE
			calc<dataflow>(N, factor.Pop(), out);
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
		Stream<T> &inX, &out;

		static void calc(const size_t N, const T factor, Stream<T> &inX, Stream<T> &out) {
			#pragma HLS INLINE
			Scalar_calc_loop:
			for (size_t i = 0; i < N; ++i) {
				#pragma HLS PIPELINE II=1
				out.Push(inX.Pop() * factor);
			}
		}
	};
}


#endif //BLAS_HLS_SCALAR_H
