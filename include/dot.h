//
// Created by Pascal on 09.05.2018.
//

#ifndef BLAS_HLS_MACC_H
#define BLAS_HLS_MACC_H

#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "Core.h"
#include "Memory.h"

namespace FBLAS {

	using namespace Memory;
	using hlslib::Stream;

	template <class T>
	class DotProductBase {
	public:
		static const size_t partialSums = 16;

		DotProductBase(const size_t N, const size_t incX, const size_t incY)
				: inX("DotProductBase_inX"), inY("DotProductBase_inY"), out("DotProductBase_out"), N(N),
				  memoryReaderX(inX, N, incX), memoryReaderY(inY, N, incY), memoryWriter(out, 1, 1) {
			#pragma HLS INLINE
		}

		MemoryReader<T> &getReaderX(void) {
			#pragma HLS INLINE
			return memoryReaderX;
		}

		MemoryReader<T> &getReaderY(void) {
			#pragma HLS INLINE
			return memoryReaderY;
		}

		MemoryWriter<T> &getWriter(void) {
			#pragma HLS INLINE
			return memoryWriter;
		}

	protected:
		Stream<T> inX, inY, out;
		const size_t N;

	private:
		MemoryReader<T> memoryReaderX, memoryReaderY;
		MemoryWriter<T> memoryWriter;
	};

	template <class Data_t>
	class DotProduct : public DotProductBase<Data_t> {
	public:
		DotProduct(const size_t N, const size_t incX, const size_t incY)
				: DotProductBase<Data_t>(N, incX, incY), pipeIntermediate("DotProduct_pipeIntermediate", Parent::partialSums) {
			#pragma HLS INLINE
		}

		template <bool dataflow = false>
		void calc(void) {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(Core::macc<Parent::partialSums>, this->N, this->inX, this->inY, pipeIntermediate);
				HLSLIB_DATAFLOW_FUNCTION(Core::accumulate<Parent::partialSums>, pipeIntermediate, this->out);
			} else {
				Core::macc<Parent::partialSums>(this->N, this->inX, this->inY, pipeIntermediate);
				Core::accumulate<Parent::partialSums>(pipeIntermediate, this->out);
			}
		}

	private:
		using Parent = DotProductBase<Data_t>;

		Stream<Data_t> pipeIntermediate;
	};

	namespace Memory {
		template <class T, size_t partition_size>
		class MemoryReaderInterleaved : MemoryReader<T> {
		public:
			MemoryReaderInterleaved(Stream<T> &pipe, const size_t N, const size_t numVectors)
					: MemoryReader<T>(pipe, N, 1), num_partitions(numVectors / partition_size), num_remaining(numVectors % partition_size) {
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

	template <class Data_t>
	class DotProductInterleaved : public DotProductBase<Data_t> {
	public:
		DotProductInterleaved(const size_t N, const size_t numVectors)
				: DotProductBase<Data_t>(N, 1, 1), num_partitions(numVectors / Parent::partialSums),
				  num_remaining(numVectors % Parent::partialSums), memoryReaderX(this->inX, N, numVectors),
				  memoryReaderY(this->inY, N, numVectors), memoryWriter(this->out, numVectors, 1) {
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

		MemoryReaderInterleaved<Data_t, DotProductInterleaved::partialSums> &getReaderX(void) {
			#pragma HLS INLINE
			return memoryReaderX;
		}

		MemoryReaderInterleaved<Data_t, DotProductInterleaved::partialSums> &getReaderY(void) {
			#pragma HLS INLINE
			return memoryReaderY;
		}

		MemoryWriter<Data_t> &getWriter(void) {
			#pragma HLS INLINE
			return memoryWriter;
		}

	private:
		using Parent = DotProductBase<Data_t>;

		const size_t num_partitions, num_remaining;
		MemoryReaderInterleaved<Data_t, Parent::partialSums> memoryReaderX, memoryReaderY;
		MemoryWriter<Data_t> memoryWriter;

		static void calc_internal(const size_t N, const size_t num_partials, const size_t num_remaining, Stream<Data_t> &X, Stream<Data_t> &Y,
		                          Stream<Data_t> &out) {
			#pragma HLS INLINE
			Data_t sums[Parent::partialSums];

			loop_dot_interleaved_part:
			for (size_t part = 0; part < num_partials + (num_remaining ? 1 : 0); ++part) {
				loop_dot_interleaved_N:
				for (size_t i = 0; i < N; ++i) {
					loop_dot_interleaved_s:
					for (size_t s = 0; s < Parent::partialSums; ++s) {
						#pragma HLS PIPELINE II=1
						#pragma HLS LOOP_FLATTEN
						if (part < num_partials || s < num_remaining) {
							Core::macc_step<Parent::partialSums>(X, Y, sums, s, i, i, N);

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
}

#endif //BLAS_HLS_MACC_H
