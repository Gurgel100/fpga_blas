//
// Created by stpascal on 16.06.18.
//

#ifndef BLAS_HLS_MEMORY_H
#define BLAS_HLS_MEMORY_H

#include <hlslib/Stream.h>
#include <hlslib/Simulation.h>

namespace FBLAS {
	namespace Memory {

		using hlslib::Stream;

		template <class T>
		class MemoryReader {
		public:
			MemoryReader(Stream<T> &pipe, const size_t N, const size_t increment) : pipe(pipe), N(N), increment(increment) {
				#pragma HLS INLINE
			}

			template <bool dataflow = false>
			void readFromMemory(const T memory[]) {
				#pragma HLS INLINE
				if (dataflow) {
					HLSLIB_DATAFLOW_FUNCTION(readFromMemory, N, increment, memory, pipe);
				} else {
					readFromMemory(N, increment, memory, pipe);
				}
			}

		protected:
			Stream<T> &pipe;
			const size_t N, increment;

		private:
			static void readFromMemory(const size_t N, const size_t increment, const T memory[], Stream<T> &pipe) {
				#pragma HLS INLINE
				size_t currentLoc = 0;
				MemoryReader_read_loop:
				for (size_t i = 0; i < N; ++i) {
					#pragma HLS PIPELINE II=1
					pipe.Push(memory[currentLoc]);
					currentLoc += increment;
				}
			}
		};

		template <class T>
		class MemoryWriter {
		public:
			MemoryWriter(Stream<T> &pipe, const size_t N, const size_t increment) : pipe(pipe), N(N), increment(increment) {
				#pragma HLS INLINE
			}

			template <bool dataflow = false>
			void writeToMemory(T memory[]) {
				#pragma HLS INLINE
				if (dataflow) {
					HLSLIB_DATAFLOW_FUNCTION(writeToMemory, N, increment, memory, pipe);
				} else {
					writeToMemory(N, increment, memory, pipe);
				}
			}

		protected:
			Stream<T> &pipe;
			const size_t N, increment;

		private:
			static void writeToMemory(const size_t N, const size_t increment, T memory[], Stream<T> &pipe) {
				#pragma HLS INLINE
				size_t currentLoc = 0;
				MemoryWriter_write_loop:
				for (size_t i = 0; i < N; ++i) {
					#pragma HLS PIPELINE II=1
					memory[currentLoc] = pipe.Pop();
					currentLoc += increment;
				}
			}
		};

		template<class T>
		void ReadMemory(const T memory[], Stream<T> &pipe, const size_t N, const int increment) {
			size_t currentLoc = 0;
			for (size_t i = 0; i < N; ++i) {
				#pragma HLS PIPELINE II=1
				pipe.Push(memory[currentLoc]);
				currentLoc += increment;
			}
		}

		template<class T>
		void WriteMemory(Stream<T> &pipe, T memory[], const size_t N, const int increment) {
			size_t currentLoc = 0;
			for (size_t i = 0; i < N; ++i) {
				#pragma HLS PIPELINE II=1
				memory[currentLoc] = pipe.Pop();
				currentLoc += increment;
			}
		}

		template <class T, size_t partition_size>
		void ReadMemoryInterleaved(const T memory[], Stream<T> &pipe, const size_t N, const size_t num_vectors) {
			const size_t num_partitions = num_vectors / partition_size;
			const size_t num_remaining = num_vectors % partition_size;

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
	}
}

#endif //BLAS_HLS_MEMORY_H
