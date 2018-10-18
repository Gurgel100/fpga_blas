//
// Created by stpascal on 21.06.18.
//

#ifndef BLAS_HLS_MATRIXVECTORMULTIPLICATION_H
#define BLAS_HLS_MATRIXVECTORMULTIPLICATION_H

#include <hlslib/Stream.h>
#include <hlslib/Simulation.h>
#include <hlslib/DataPack.h>
#include <hlslib/TreeReduce.h>
#include <hlslib/Operators.h>
#include <hlslib/Utility.h>
#include <assert.h>
#include "Memory.h"
#include "Core.h"

// Note: clang <3.3 doesn't support constructor inheritance

/*
 * Format:
 * +-----N-----+ +-----+   +-----+
 * |           | |     |   |     |
 * |           | |     |   |     |
 * M     A     | N  x  | = M  y  |
 * |           | |     |   |     |
 * |           | |     |   |     |
 * +-----------+ +-----+   +-----+
 */

/*
 * TODO: Generalize multiplication so that N must not be a multiple of size_colchunk
 */

namespace FBLAS {

	using namespace Memory;
	using hlslib::Stream;

	namespace Memory {
		namespace {
			template <class T, size_t typewidth, size_t chunksize>
			class MemoryReaderVectorImplementation : public MemoryReader<T> {
			public:
				static_assert(typewidth > 0, "typewidth must be greater than 0");
				static_assert(chunksize >= typewidth, "chunksize must be bigger or equal to typewidth");
				static_assert(chunksize % typewidth == 0, "chunksize must be divisible by typewidth");

				MemoryReaderVectorImplementation(Stream<T> &pipe, const size_t N, const size_t times)
						: MemoryReader<T>(pipe, N), times(times) {
					#pragma HLS INLINE
				}

				template <bool dataflow = false>
				void readFromMemory(const T memory[], const size_t increment = 1) {
					#pragma HLS INLINE
					if (dataflow) {
						HLSLIB_DATAFLOW_FUNCTION(readFromMemory, this->N, times, increment, memory, this->pipe);
					} else {
						readFromMemory(this->N, times, increment, memory, this->pipe);
					}
				}

			private:
				const size_t times;

				//TODO: handle case where N % typewidth != 0
				static void readFromMemory(const size_t N, const size_t times, const size_t increment, const T memory[], Stream<T> &pipe) {
					#pragma HLS INLINE

					constexpr size_t num_cols = chunksize / typewidth;
					const size_t num_chunks = N / chunksize;
					const size_t num_cols_last = (N - num_chunks * chunksize) / typewidth ?: num_cols;

					MemoryReaderVector_times:
					for (size_t i = 0; i < times; ++i) {
						MemoryReaderVector_chunk:
						for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
							for (size_t col = 0; col < num_cols; ++col) {
								#pragma HLS PIPELINE II=1
								#pragma HLS LOOP_FLATTEN

								if (chunk < num_chunks - 1 || col < num_cols_last) {
									pipe.Push(memory[chunk * num_cols + col]);
								}
							}
						}
					}
				}
			};

			template <class T, size_t typewidth, size_t size_rowchunk, size_t size_colchunk>
			class MemoryReaderMatrixImplementation : public MemoryReader<T> {
			public:
				static constexpr size_t num_columns_per_colchunk = size_colchunk / typewidth;

				MemoryReaderMatrixImplementation(Stream<T> &pipe, const size_t N, const size_t M) : MemoryReader<T>(pipe, N), M(M) {
					#pragma HLS INLINE
				}

				template <bool dataflow = false>
				void readFromMemory(const T memory[]) {
					#pragma HLS INLINE
					if (dataflow) {
						HLSLIB_DATAFLOW_FUNCTION(readFromMemory, this->N, M, memory, this->pipe);
					} else {
						readFromMemory(this->N, M, memory, this->pipe);
					}
				}

			protected:
				const size_t M;

			private:
				static_assert(typewidth > 0, "typewidth must be greater than 0");
				static_assert(typewidth <= size_colchunk, "typewidth can't be bigger than size_colchunk");
				static_assert(size_colchunk % typewidth == 0, "size_colchunk must be dividable by typewidth");

				//TODO: handle N < size_colchunk
				static void readFromMemory(const size_t N, const size_t M, const T memory[], Stream<T> &pipe) {
					#pragma HLS INLINE

					const size_t num_rowchunks = M / size_rowchunk + (M % size_rowchunk ? 1 : 0);
					const size_t num_colchunks = N / size_colchunk + (N % size_colchunk ? 1 : 0);

					MemoryReaderMatrix_y_chunk:
					for (size_t rc = 0; rc < num_rowchunks; ++rc) {
						MemoryReaderMatrix_x_chunk:
						for (size_t cc = 0; cc < num_colchunks; ++cc) {
							#pragma HLS LOOP_FLATTEN
							const size_t num_subrows = (rc == num_rowchunks - 1) ? M - rc * size_rowchunk : size_rowchunk;
							const size_t current_row = rc * size_rowchunk;

							MemoryReaderMatrix_row:
							for (size_t rowi = 0; rowi < num_subrows; ++rowi) {
								MemoryReaderMatrix_col:
								for (size_t coli = 0; coli < num_columns_per_colchunk; ++coli) {
									#pragma HLS LOOP_FLATTEN
									#pragma HLS PIPELINE II=1

									pipe.Push(memory[((current_row + rowi) * num_colchunks + cc) * num_columns_per_colchunk + coli]);
								}
							}
						}
					}
				}
			};

			template <class T, size_t typewidth, size_t size_rowchunk, size_t size_colchunk>
			class MemoryReaderTransposedMatrixImplementation : public MemoryReaderMatrixImplementation<T, typewidth, size_rowchunk, size_colchunk> {
			public:
				MemoryReaderTransposedMatrixImplementation(Stream<T> &pipe, const size_t N, const size_t M)
					: MemoryReaderMatrixImplementation<T, typewidth, size_rowchunk, size_colchunk>(pipe, N, M) {
					#pragma HLS INLINE
				};

				template <bool dataflow = false>
				void readFromMemory(const T memory[]) {
					#pragma HLS INLINE
					if (dataflow) {
						HLSLIB_DATAFLOW_FUNCTION(readFromMemory, this->N, this->M, memory, this->pipe);
					} else {
						readFromMemory(this->N, this->M, memory, this->pipe);
					}
				}

			private:
				using Parent = MemoryReaderMatrixImplementation<T, typewidth, size_rowchunk, size_colchunk>;

				static void readFromMemory(const size_t N, const size_t M, const T memory[], Stream<T> &pipe) {
					#pragma HLS INLINE

					const size_t num_rowchunks = M / size_rowchunk + (M % size_rowchunk ? 1 : 0);
					const size_t num_colchunks = N / size_colchunk + (N % size_colchunk ? 1 : 0);

					MemoryReaderMatrix_x_chunk:
					for (size_t cc = 0; cc < num_colchunks; ++cc) {
						MemoryReaderMatrix_y_chunk:
						for (size_t rc = 0; rc < num_rowchunks; ++rc) {
							#pragma HLS LOOP_FLATTEN
							const size_t num_subrows = (rc == num_rowchunks - 1) ? M - rc * size_rowchunk : size_rowchunk;
							const size_t current_row = rc * size_rowchunk;

							MemoryReaderMatrix_col:
							for (size_t coli = 0; coli < Parent::num_columns_per_colchunk; ++coli) {
								MemoryReaderMatrix_row:
								for (size_t rowi = 0; rowi < num_subrows; ++rowi) {
									#pragma HLS LOOP_FLATTEN
									#pragma HLS PIPELINE II=1

									pipe.Push(memory[((current_row + rowi) * num_colchunks + cc) * Parent::num_columns_per_colchunk + coli]);
								}
							}
						}
					}
				}
			};
		}

		template <class T, size_t chunksize>
		class MemoryReaderVector : public MemoryReaderVectorImplementation<T, 1, chunksize> {
		public:
			MemoryReaderVector(Stream<T> &pipe, const size_t N, const size_t times)
				: MemoryReaderVectorImplementation<T, 1, chunksize>(pipe, N, times) {
				#pragma HLS INLINE
			}
		};

		template <class T, int width, size_t chunksize>
		class MemoryReaderVector<hlslib::DataPack<T, width>, chunksize> : public MemoryReaderVectorImplementation<hlslib::DataPack<T, width>, width, chunksize> {
		public:
			MemoryReaderVector(Stream<hlslib::DataPack<T, width>> &pipe, const size_t N, const size_t times)
				: MemoryReaderVectorImplementation<hlslib::DataPack<T, width>, width, chunksize>(pipe, N, times) {
				#pragma HLS INLINE
			}
		};

		template <class T, size_t size_rowchunk, size_t size_colchunk>
		class MemoryReaderMatrix : public MemoryReaderMatrixImplementation<T, 1, size_rowchunk, size_colchunk> {
		public:
			MemoryReaderMatrix(Stream<T> &pipe, const size_t N, const size_t M)
				: MemoryReaderMatrixImplementation<T, 1, size_rowchunk, size_colchunk>(pipe, N, M) {
				#pragma HLS INLINE
			}
		};

		template <class T, int width, size_t size_rowchunk, size_t size_colchunk>
		class MemoryReaderMatrix<hlslib::DataPack<T, width>, size_rowchunk, size_colchunk> : public MemoryReaderMatrixImplementation<hlslib::DataPack<T, width>, width, size_rowchunk, size_colchunk> {
		public:
			MemoryReaderMatrix(Stream<hlslib::DataPack<T, width>> &pipe, const size_t N, const size_t M)
					: MemoryReaderMatrixImplementation<hlslib::DataPack<T, width>, width, size_rowchunk, size_colchunk>(pipe, N, M) {
				#pragma HLS INLINE
			}
		};

		template <class T, size_t size_rowchunk, size_t size_colchunk>
		class MemoryReaderTransposedMatrix : public MemoryReaderTransposedMatrixImplementation<T, 1, size_rowchunk, size_colchunk> {
		public:
			MemoryReaderTransposedMatrix(Stream<T> &pipe, const size_t N, const size_t M)
					: MemoryReaderTransposedMatrixImplementation<T, 1, size_rowchunk, size_colchunk>(pipe, N, M) {
				#pragma HLS INLINE
			}
		};

		template <class T, int width, size_t size_rowchunk, size_t size_colchunk>
		class MemoryReaderTransposedMatrix<hlslib::DataPack<T, width>, size_rowchunk, size_colchunk> : public MemoryReaderTransposedMatrixImplementation<hlslib::DataPack<T, width>, width, size_rowchunk, size_colchunk> {
		public:
			MemoryReaderTransposedMatrix(Stream<hlslib::DataPack<T, width>> &pipe, const size_t N, const size_t M)
			: MemoryReaderTransposedMatrixImplementation<hlslib::DataPack<T, width>, width, size_rowchunk, size_colchunk>(pipe, N, M) {
				#pragma HLS INLINE
			}
		};
	}

	template <class T, size_t size_rowchunk = 16, size_t size_colchunk = 16, size_t size_column = size_colchunk>
	class MatrixVectorMultiplication {
	public:
		static_assert(size_column <= size_colchunk, "size_column must not be bigger than size_colchunk");

		using Col_t = hlslib::DataPack<T, size_column>;
		using MemoryReaderVectorType = MemoryReaderVector<Col_t, size_colchunk>;
		using MemoryReaderMatrixType = MemoryReaderMatrix<Col_t, size_rowchunk, size_colchunk>;

		static constexpr size_t num_columns_per_colchunk = size_colchunk / size_column;

		MatrixVectorMultiplication(const size_t N, const size_t M, Stream<Col_t> &inA, Stream<Col_t> &inX, Stream<T> &out)
				: N(N), M(M), inA(inA), inX(inX), out(out) {
			#pragma HLS INLINE
		}

		template <bool dataflow = false>
		void calc() {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(calc, inA, inX, out, N, M);
			} else {
				calc(inA, inX, out, N, M);
			}
		}

		MemoryReaderVectorType getReaderX() {
			#pragma HLS INLINE
			return MemoryReaderVectorType(inX, N, hlslib::CeilDivide(M, size_rowchunk));
		}

		MemoryReaderMatrixType getReaderA() {
			#pragma HLS INLINE
			return MemoryReaderMatrixType(inA, N, M);
		}

		MemoryWriter<T> getWriter() {
			#pragma HLS INLINE
			return MemoryWriter<T>(out, M);
		}

	private:
		size_t N, M;
		Stream<Col_t> &inA, &inX;
		Stream<T> &out;

		static void calc(Stream<Col_t> &inA, Stream<Col_t> &inX, Stream<T> &outY, const size_t N, const size_t M)
		{
			#pragma HLS INLINE

			const size_t num_rowchunks = M / size_rowchunk + (M % size_rowchunk ? 1 : 0);
			const size_t num_colchunks = N / size_colchunk + (N % size_colchunk ? 1 : 0);

			y_chunk:
			for (size_t rc = 0; rc < num_rowchunks; ++rc) {
				#pragma HLS LOOP_TRIPCOUNT min=1
				T chunkres[size_rowchunk];
				x_chunk:
				for (size_t cc = 0; cc < num_colchunks; ++cc) {
					#pragma HLS LOOP_FLATTEN
					#pragma HLS LOOP_TRIPCOUNT min=1
					const size_t num_subrows = (rc == num_rowchunks - 1) ? M - rc * size_rowchunk : size_rowchunk;
					Col_t xChunk[num_columns_per_colchunk];
					row:
					for (size_t rowi = 0; rowi < num_subrows; ++rowi) {
						#pragma HLS LOOP_TRIPCOUNT min=1 max=size_colchunk
						T colres[num_columns_per_colchunk];
						col:
						for (size_t coli = 0; coli < num_columns_per_colchunk; ++coli) {
							#pragma HLS PIPELINE II=1
							#pragma HLS LOOP_FLATTEN
							T tmp_row[size_colchunk];

							if (rowi == 0) {
								xChunk[coli] = inX.Pop();
							}

							auto row = inA.Pop() * xChunk[coli];
							row.Unpack(tmp_row);
							colres[coli] = hlslib::TreeReduce<T, hlslib::op::Add<T>, size_colchunk>(tmp_row);

							if (coli == num_columns_per_colchunk - 1) {
								auto prev = cc == 0 ? 0 : chunkres[rowi];
								chunkres[rowi] = prev + hlslib::TreeReduce<T, hlslib::op::Add<T>, num_columns_per_colchunk>(colres);

								if (cc == num_colchunks - 1) {
									outY.Push(chunkres[rowi]);
								}
							}
						}
					}
				}
			}
		}
	};

	template <class T, size_t size_rowchunk = 16, size_t size_colchunk = 16, size_t size_column = size_colchunk>
	class MatrixVectorMultiplicationTransposed {
	public:
		static_assert(size_column <= size_colchunk, "size_column must not be bigger than size_colchunk");

		using Col_t = hlslib::DataPack<T, size_colchunk>;
		using MemoryReaderVectorType = MemoryReaderVector<T, 1>;
		using MemoryReaderMatrixType = MemoryReaderTransposedMatrix<Col_t, size_rowchunk, size_colchunk>;

		static constexpr size_t num_columns_per_colchunk = size_colchunk / size_column;

		MatrixVectorMultiplicationTransposed(const size_t N, const size_t M, Stream<Col_t> &inA, Stream<T> &inX, Stream<Col_t> &out)
				: N(N), M(M), inA(inA), inX(inX), out(out), intermediate("MatrixVectorMultiplicationTransposed_intermediate") {
			#pragma HLS INLINE
			assert(N % size_colchunk == 0);
			assert(M % size_rowchunk == 0);
		}

		template <bool dataflow = false>
		void calc() {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(calc, inA, inX, intermediate, N, M);
				HLSLIB_DATAFLOW_FUNCTION(accumulate_colchunk, intermediate, out, N);
			} else {
				calc(inA, inX, intermediate, N, M);
				accumulate_colchunk(intermediate, out, N);
			}
		}

		MemoryReaderVectorType getReaderX() {
			#pragma HLS INLINE
			return MemoryReaderVectorType(inX, M, hlslib::CeilDivide(N, size_colchunk));
		}

		MemoryReaderMatrixType getReaderA() {
			#pragma HLS INLINE
			return MemoryReaderMatrixType(inA, N, M);
		}

		MemoryWriter<Col_t> getWriter() {
			#pragma HLS INLINE
			return MemoryWriter<Col_t>(out, N / size_colchunk);
		}

	private:
		size_t N, M;
		Stream<Col_t> &inA;
		Stream<T> &inX;
		Stream<Col_t> &out;
		Stream<Col_t> intermediate;

		static void calc(Stream<Col_t> &inA, Stream<T> &inX, Stream<Col_t> &out, const size_t N, const size_t M)
		{
			#pragma HLS INLINE

			const size_t num_rowchunks = M / size_rowchunk + (M % size_rowchunk ? 1 : 0);
			const size_t num_colchunks = N / size_colchunk + (N % size_colchunk ? 1 : 0);

			col_chunk:
			for (size_t cc = 0; cc < num_colchunks; ++cc) {
				#pragma HLS LOOP_TRIPCOUNT min=1
				Col_t rowRes[size_rowchunk * num_columns_per_colchunk];

				row_chunk:
				for (size_t rc = 0; rc < num_rowchunks; ++rc) {
					#pragma HLS LOOP_TRIPCOUNT min=1
					row:
					for (size_t r = 0; r < size_rowchunk; ++r) {
						Col_t xChunk;
						col:
						for (size_t coli = 0; coli < num_columns_per_colchunk; ++coli) {
							#pragma HLS LOOP_FLATTEN
							#pragma HLS PIPELINE II=1

							if (coli == 0) {
								xChunk = inX.Pop();
							}

							auto row = inA.Pop() * xChunk;
							size_t rowResIndex = r * num_columns_per_colchunk + coli;
							auto prev = rc == 0 ? Col_t(T(0)) : rowRes[rowResIndex];
							rowRes[rowResIndex] = prev + row;

							if (rc == num_rowchunks - 1) {
								out.Push(rowRes[rowResIndex]);
							}
						}
					}
				}
			}
		}

		static void accumulate_colchunk(Stream<Col_t> &in, Stream<Col_t> &out, const size_t N) {
			#pragma HLS INLINE
			const size_t num_colchunks = N / size_colchunk + (N % size_colchunk ? 1 : 0);

			for (size_t colchunk = 0; colchunk < num_colchunks; ++colchunk) {
				Core::accumulate<Col_t, size_rowchunk, T, num_columns_per_colchunk>(in, out);
			}
		}
	};
}


#endif //BLAS_HLS_MATRIXVECTORMULTIPLICATION_H
