//
// Created by stpascal on 01.07.18.
//

#ifndef BLAS_HLS_NORM_H
#define BLAS_HLS_NORM_H

#include <hlslib/Stream.h>
#include <hlslib/Simulation.h>
#include <hlslib/DataPack.h>
#include <cmath>
#include "Memory.h"
#include "dot.h"

namespace FBLAS {

	using namespace Memory;
	using hlslib::Stream;

	template <class T, size_t width = 1>
	class Norm {
	public:
		using Type = typename std::conditional<width == 1, T, hlslib::DataPack<T, width>>::type;

		Norm(const size_t N, Stream<Type> &in, Stream<T> &out)
			: N(N), in(in), inDotA("Norm_inDotA"), inDotB("Norm_inDotB"), out(out), outDot("Norm_outDot"),
			duplicator(N / width), dotProduct(N, inDotA, inDotB, outDot) {
			#pragma HLS INLINE
		}

		MemoryReader<Type> getReader() {
			#pragma HLS INLINE
			return MemoryReader<Type>(in, N / width);
		}

		MemoryWriter<T> getWriter() {
			#pragma HLS INLINE
			return MemoryWriter<T>(out, 1);
		}

		template <bool dataflow = false>
		void calc() {
			#pragma HLS INLINE
			if (dataflow) {
				duplicator.template duplicate<true>(in, inDotA, inDotB);
				dotProduct.template calc<true>();
				HLSLIB_DATAFLOW_FUNCTION(calc, N, outDot, out);
			} else {
				duplicator.duplicate(in, inDotA, inDotB);
				dotProduct.calc();
				calc(N, outDot, out);
			}
		}

	private:
		const size_t N;
		Stream<Type> &in, inDotA, inDotB;
		Stream<T> &out, outDot;
		StreamDuplicator<Type> duplicator;
		DotProduct<T, width> dotProduct;

		static void calc(const size_t N, Stream<T> &in, Stream<T> &out) {
			#pragma HLS PIPELINE II=1
			auto res = sqrt(in.Pop());
			out.Push(res);
		}
	};
}

#endif //BLAS_HLS_NORM_H
