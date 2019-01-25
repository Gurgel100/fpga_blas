//
// Created by stpascal on 17.07.18.
//

#ifndef BLAS_HLS_VECTORPRODUCT_H
#define BLAS_HLS_VECTORPRODUCT_H

#include <hlslib/Stream.h>
#include <hlslib/Simulation.h>
#include "Memory.h"

namespace FBLAS {

	using hlslib::Stream;

	template <class T, class Op>
	class VectorElementOperation {
	public:
		VectorElementOperation(const size_t N, Stream<T> &inX, Stream<T> &inY, Stream<T> &out)
				: N(N), inX(inX), inY(inY), out(out) {
			#pragma HLS INLINE
		}

		Memory::MemoryReader<T> getReaderX() {
			#pragma HLS INLINE
			return Memory::MemoryReader<T>(inX, N);
		}

		Memory::MemoryReader<T> getReaderY() {
			#pragma HLS INLINE
			return Memory::MemoryReader<T>(inY, N);
		}

		Memory::MemoryWriter<T> getWriter() {
			#pragma HLS INLINE
			return Memory::MemoryWriter<T>(out, N);
		}

		template <bool dataflow = false>
		void calc() {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(calc, N, inX, inY, out);
			} else {
				calc(N, inX, inY, out);
			}
		}

	private:
		const size_t N;
		Stream<T> &inX, &inY, &out;

		static void calc(const size_t N, Stream<T> &X, Stream<T> &Y, Stream<T> &out) {
			#pragma HLS INLINE
			for (size_t i = 0; i < N; ++i) {
				out.Push(Op::Apply(X.Pop(), Y.Pop()));
			}
		}
	};
}


#endif //BLAS_HLS_VECTORPRODUCT_H
