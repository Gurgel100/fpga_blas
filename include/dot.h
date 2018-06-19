//
// Created by Pascal on 09.05.2018.
//

#ifndef BLAS_HLS_MACC_H
#define BLAS_HLS_MACC_H

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "Core.h"

using hlslib::Stream;

template <class Data_t>
class DotProductBase {
public:
	static const size_t partialSums = 16;

	DotProductBase(Stream<Data_t> &out) : out(out) {
		#pragma HLS INLINE
	}

protected:
	Stream<Data_t> &out;
};

template <class Data_t>
class DotProduct : public DotProductBase<Data_t> {
public:
	DotProduct(Stream<Data_t> &out) : DotProductBase<Data_t>(out), pipeIntermediate("dot.intermediatePipe", Parent::partialSums) {
		#pragma HLS INLINE
	}

	void calc(const size_t N, Stream<Data_t> &X, Stream<Data_t> &Y) {
		#pragma HLS INLINE
		HLSLIB_DATAFLOW_FUNCTION(Core::macc<Parent::partialSums>, N, X, Y, pipeIntermediate);
		HLSLIB_DATAFLOW_FUNCTION(Core::accumulate<Parent::partialSums>, pipeIntermediate, this->out);
	}

private:
	using Parent = DotProductBase<Data_t>;

	Stream<Data_t> pipeIntermediate;
};

template <class Data_t>
class DotProductInterleaved : public DotProductBase<Data_t> {
public:
	DotProductInterleaved(Stream<Data_t> &out) : DotProductBase<Data_t>(out) {
		#pragma HLS INLINE
	}

	void calc(const size_t N, const size_t num_prods, Stream<Data_t> &X, Stream<Data_t> &Y) {
		#pragma HLS INLINE
		HLSLIB_DATAFLOW_FUNCTION(calc_internal, N, num_prods, X, Y, this->out);
	}

private:
	using Parent = DotProductBase<Data_t>;

	static void calc_internal(const size_t N, const size_t num_prods, Stream<Data_t> &X, Stream<Data_t> &Y, Stream<Data_t> &out) {
		#pragma HLS INLINE
		Data_t sums[Parent::partialSums];
		const size_t num_partials = num_prods / Parent::partialSums;
		const size_t num_remaining = num_prods % Parent::partialSums;

		loop_dot_interleaved_part:
		for (size_t part = 0; part < num_partials + (num_remaining ? 1 : 0); ++part) {
			loop_dot_interleaved_N:
			for (size_t i = 0; i < N; ++i) {
				loop_dot_interleaved_s:
				for (size_t s = 0; s < Parent::partialSums; ++s) {
					#pragma HLS PIPELINE II=1
					#pragma HLS LOOP_FLATTEN
					if (part < num_partials || s < num_remaining) {
						Core::macc_step<Parent::partialSums>(X, Y, sums, s, i, N);

						if (i == N - 1) {
							//In the last round we push to the output stream
							out.Push(sums[s]);
						}
					}
				}
			}
		}
	}
};

#endif //BLAS_HLS_MACC_H
