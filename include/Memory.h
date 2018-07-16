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
	}
}

#endif //BLAS_HLS_MEMORY_H
