//
// Created by Pascal on 09.05.2018.
//

#ifndef BLAS_HLS_MACC_H
#define BLAS_HLS_MACC_H

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include <hlslib/DataPack.h>
#include <hlslib/TreeReduce.h>
#include <hlslib/Operators.h>
#include "Core.h"
#include "Memory.h"

namespace FBLAS {

	using namespace Memory;
	using hlslib::Stream;

	template <class T, size_t width = 1>
	class DotProductBase {
	public:
		using Chunk = hlslib::DataPack<T, width>;

		static const size_t partialSums = 16;

		DotProductBase(const size_t N, Stream<Chunk> &inX, Stream<Chunk> &inY, Stream<T> &out)
				: N(N), inX(inX), inY(inY), out(out) {
			#pragma HLS INLINE
		}

		MemoryReader<Chunk> getReaderX(void) {
			#pragma HLS INLINE
			return MemoryReader<Chunk>(inX, N / width);
		}

		MemoryReader<Chunk> getReaderY(void) {
			#pragma HLS INLINE
			return MemoryReader<Chunk>(inY, N / width);
		}

		MemoryWriter<T> getWriter(void) {
			#pragma HLS INLINE
			return MemoryWriter<T>(out, 1);
		}

	protected:
		const size_t N;
		Stream<Chunk> &inX, &inY;
		Stream<T> &out;
	};

	template <class T, size_t width = 1>
	class DotProduct : public DotProductBase<T, width> {
	public:
		DotProduct(const size_t N, Stream<hlslib::DataPack<T, width>> &inX, Stream<hlslib::DataPack<T, width>> &inY, Stream<T> &out)
				: Parent(N, inX, inY, out), pipeIntermediate("DotProduct_pipeIntermediate", Parent::partialSums) {
			#pragma HLS INLINE
		}

		template <bool dataflow = false>
		void calc(void) {
			#pragma HLS INLINE
			if (dataflow) {
				//HLSLIB_DATAFLOW_FUNCTION(Core::macc<Data_t, Parent::partialSums>, this->N, this->inX, this->inY, pipeIntermediate);
				//HLSLIB_DATAFLOW_FUNCTION(Core::accumulate<Data_t, Parent::partialSums>, pipeIntermediate, this->out);
				// we can't use the HLSLIB_DATAFLOW_FUNCTION macro because of a bug in vivado hls
				#ifndef HLSLIB_SYNTHESIS
					hlslib::_Dataflow::Get().AddFunction(Core::macc<typename Parent::Chunk, Parent::partialSums, T>, this->N / width, this->inX, this->inY, pipeIntermediate);
					hlslib::_Dataflow::Get().AddFunction(Core::accumulate<typename Parent::Chunk, Parent::partialSums, T>, pipeIntermediate, pipeReduce, 0);
					hlslib::_Dataflow::Get().AddFunction(Core::reduceDataPack<T, width, hlslib::op::Add<T>>, pipeReduce, this->out);
				#else
					Core::macc<typename Parent::Chunk, Parent::partialSums, T>(this->N / width, this->inX, this->inY, pipeIntermediate);
					Core::accumulate<typename Parent::Chunk, Parent::partialSums, T>(pipeIntermediate, pipeReduce);
					Core::reduceDataPack<T, width, hlslib::op::Add<T>>(pipeReduce, this->out);
				#endif
			} else {
				Core::macc<typename Parent::Chunk, Parent::partialSums, T>(this->N / width, this->inX, this->inY, pipeIntermediate);
				Core::accumulate<typename Parent::Chunk, Parent::partialSums, T>(pipeIntermediate, pipeReduce);
				Core::reduceDataPack<T, width, hlslib::op::Add<T>>(pipeReduce, this->out);
			}
		}

	private:
		using Parent = DotProductBase<T, width>;

		Stream<typename Parent::Chunk> pipeIntermediate, pipeReduce;
	};

	namespace Memory {
		template <class T, size_t partition_size>
		class MemoryReaderInterleaved : MemoryReader<T> {
		public:
			MemoryReaderInterleaved(Stream<T> &pipe, const size_t N, const size_t numVectors)
					: MemoryReader<T>(pipe, N), num_partitions(numVectors / partition_size), num_remaining(numVectors % partition_size) {
				#pragma HLS INLINE
			}

			template <bool dataflow = false>
			void readFromMemory(const T memory[]) {
				#pragma HLS INLINE
				if (dataflow) {
					HLSLIB_DATAFLOW_FUNCTION(readFromMemory, this->N, num_partitions, num_remaining, memory, this->pipe);
				} else {
					readFromMemory(this->N, num_partitions, num_remaining, memory, this->pipe);
				}
			}

		private:
			const size_t num_partitions, num_remaining;

			static void readFromMemory(const size_t N, const size_t num_partitions, const size_t num_remaining, const T memory[], Stream<T> &pipe) {
				#pragma HLS INLINE

				MemoryReaderInterleaved_reader_loop:
				for (size_t part = 0; part < num_partitions + (num_remaining ? 1 : 0); ++part) {
					for (size_t i = 0; i < N; ++i) {
						for (size_t j = 0; j < partition_size; ++j) {
							#pragma HLS PIPELINE II=1
							#pragma HLS LOOP_FLATTEN
							if (part < num_partitions || j < num_remaining) {
								const size_t index = (part * partition_size + j) * N + i;
								pipe.Push(memory[index]);
							}
						}
					}
				}
			}
		};
	}

	template <class T, size_t width>
	class DotProductInterleaved : public DotProductBase<T, width> {
	public:
		DotProductInterleaved(const size_t N, const size_t numVectors, Stream<hlslib::DataPack<T, width>> &inX, Stream<hlslib::DataPack<T, width>> &inY,
				Stream<T> &out)
				: DotProductBase<T, width>(N, inX, inY, out), numVectors(numVectors), num_partitions(numVectors / Parent::partialSums),
				  num_remaining(numVectors % Parent::partialSums) {
			#pragma HLS INLINE
		}

		template <bool dataflow = false>
		void calc(void) {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(calc_internal, this->N, num_partitions, num_remaining, this->inX, this->inY, this->out);
			} else {
				calc_internal(this->N, num_partitions, num_remaining, this->inX, this->inY, this->out);
			}
		}

		MemoryReaderInterleaved<typename DotProductInterleaved::Chunk, DotProductInterleaved::partialSums> getReaderX(void) {
			#pragma HLS INLINE
			return MemoryReaderInterleaved<typename Parent::Chunk, Parent::partialSums>(this->inX, this->N, numVectors);
		}

		MemoryReaderInterleaved<typename DotProductInterleaved::Chunk, DotProductInterleaved::partialSums> getReaderY(void) {
			#pragma HLS INLINE
			return MemoryReaderInterleaved<typename Parent::Chunk, Parent::partialSums>(this->inY, this->N, numVectors);
		}

		MemoryWriter<T> getWriter(void) {
			#pragma HLS INLINE
			return MemoryWriter<T>(this->out, numVectors);
		}

	private:
		using Parent = DotProductBase<T, width>;

		const size_t numVectors, num_partitions, num_remaining;

		static void calc_internal(const size_t N, const size_t num_partials, const size_t num_remaining, Stream<typename Parent::Chunk> &X,
				Stream<typename Parent::Chunk> &Y, Stream<T> &out) {
			#pragma HLS INLINE
			typename Parent::Chunk sums[Parent::partialSums];

			loop_dot_interleaved_part:
			for (size_t part = 0; part < num_partials + (num_remaining ? 1 : 0); ++part) {
				loop_dot_interleaved_N:
				for (size_t i = 0; i < N; ++i) {
					loop_dot_interleaved_s:
					for (size_t s = 0; s < Parent::partialSums; ++s) {
						#pragma HLS PIPELINE II=1
						#pragma HLS LOOP_FLATTEN
						if (part < num_partials || s < num_remaining) {
							Core::macc_step<typename Parent::Chunk, Parent::partialSums, T>(X, Y, sums, s, i, i, N);

							if (i == N - 1) {
								//In the last round we push to the output stream
								auto res = hlslib::TreeReduce<T, hlslib::op::Add<T>, width>(sums[s]);
								out.Push(res);
							}
						}
					}
				}
			}
		}
	};
}

#endif //BLAS_HLS_MACC_H
