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
	namespace {
		using hlslib::Stream;

		template<class T, size_t width, class PackedType>
		class InternalScalar {
		public:
			InternalScalar(const size_t N, Stream<PackedType> &inX, Stream<PackedType> &out) : N(N), inX(inX), out(out) {
				#pragma HLS INLINE
			}

			Memory::MemoryReader<PackedType> getReaderX() {
				#pragma HLS INLINE
				return Memory::MemoryReader<PackedType>(inX, N / width);
			}

			Memory::MemoryWriter<PackedType> getWriterX() {
				#pragma HLS INLINE
				return Memory::MemoryWriter<PackedType>(out, N / width);
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
			Stream<PackedType> &inX, &out;

			static void calc(const size_t N, const T factor, Stream<PackedType> &inX, Stream<PackedType> &out) {
				#pragma HLS INLINE
				Scalar_calc_loop:
				for (size_t i = 0; i < N / width; ++i) {
					#pragma HLS PIPELINE II=1
					out.Push(inX.Pop() * factor);
				}
			}

			static void calcStream(const size_t N, Stream<T> &factor, Stream<PackedType> &inX, Stream<PackedType> &out) {
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

	template <class T, size_t width = 1>
	using PackedScalar = InternalScalar<T, width, hlslib::DataPack<T, width>>;

	template <class T>
	using Scalar = InternalScalar<T, 1, T>;
}


#endif //BLAS_HLS_SCALAR_H
