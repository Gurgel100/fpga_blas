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
			MemoryReader(Stream<T> &pipe, const size_t N) : pipe(pipe), N(N) {
				#pragma HLS INLINE
			}

			template <bool dataflow = false>
			void readFromMemory(const T memory[], const size_t increment = 1) {
				#pragma HLS INLINE
				if (dataflow) {
					HLSLIB_DATAFLOW_FUNCTION(readFromMemory, N, increment, memory, pipe);
				} else {
					readFromMemory(N, increment, memory, pipe);
				}
			}

		protected:
			Stream<T> &pipe;
			const size_t N;

		private:
			static void readFromMemory(const size_t N, const size_t increment, const T memory[], Stream<T> &pipe) {
				#pragma HLS INLINE
				if (increment == 1) {
					MemoryReader_read_loop:
					for (size_t i = 0; i < N; ++i) {
						#pragma HLS PIPELINE II=1
						pipe.Push(memory[i]);
					}
				} else {
					size_t currentLoc = 0;
					MemoryReader_read_increment_loop:
					for (size_t i = 0; i < N; ++i) {
						#pragma HLS PIPELINE II=1
						pipe.Push(memory[currentLoc]);
						currentLoc += increment;
					}
				}
			}
		};

		template <class T>
		class MemoryWriter {
		public:
			MemoryWriter(Stream<T> &pipe, const size_t N) : pipe(pipe), N(N) {
				#pragma HLS INLINE
			}

			template <bool dataflow = false>
			void writeToMemory(T memory[], const size_t increment = 1) {
				#pragma HLS INLINE
				if (dataflow) {
					HLSLIB_DATAFLOW_FUNCTION(writeToMemory, N, increment, memory, pipe);
				} else {
					writeToMemory(N, increment, memory, pipe);
				}
			}

		protected:
			Stream<T> &pipe;
			const size_t N;

		private:
			static void writeToMemory(const size_t N, const size_t increment, T memory[], Stream<T> &pipe) {
				#pragma HLS INLINE
				if (increment == 1) {
					MemoryWriter_write_loop:
					for (size_t i = 0; i < N; ++i) {
						#pragma HLS PIPELINE II=1
						memory[i] = pipe.Pop();
					}
				} else {
					size_t currentLoc = 0;
					MemoryWriter_write_increment_loop:
					for (size_t i = 0; i < N; ++i) {
						#pragma HLS PIPELINE II=1
						memory[currentLoc] = pipe.Pop();
						currentLoc += increment;
					}
				}
			}
		};

		namespace {
			template <class T>
			struct DuplicateAll {
				template <class I>
				inline bool operator()(const I &index, const T &value) {
					return true;
				}
			};
		}

		/**
		 * @brief Duplicates a stream if the condition of @p Functor is fullfilled
		 *
		 * @tparam T The type of the elements in the stream
		 * @tparam Functor Returns true if it should be duplicated else it only copies values from in to outA
		 */
		template <class T, class Functor = DuplicateAll<T>>
		class StreamDuplicator {
		public:
			StreamDuplicator(const size_t N) : N(N) {
				#pragma HLS INLINE
			}

			template <bool dataflow = false>
			void duplicate(Stream<T> &in, Stream<T> &outA, Stream<T> &outB) {
				#pragma HLS INLINE
				if (dataflow) {
					HLSLIB_DATAFLOW_FUNCTION(duplicate, N, in, outA, outB);
				} else {
					duplicate(N, in, outA, outB);
				}
			}

		protected:
			const size_t N;

		private:
			static void duplicate(const size_t N, Stream<T> &in, Stream<T> &outA, Stream<T> &outB) {
				#pragma HLS INLINE
				Functor check;
				StreamDuplicator_duplicate_loop:
				for (size_t i = 0; i < N; ++i) {
					#pragma HLS PIPELINE II=1
					auto val = in.Pop();
					outA.Push(val);
					if (check(i, val)) {
						outB.Push(val);
					}
				}
			}
		};
	}
}

#endif //BLAS_HLS_MEMORY_H
