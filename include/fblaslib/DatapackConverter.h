//
// Created by stpascal on 19.12.18.
//

#ifndef BLAS_HLS_DATAPACKCONVERTER_H
#define BLAS_HLS_DATAPACKCONVERTER_H

#include <hlslib/Stream.h>
#include <hlslib/DataPack.h>
#include <hlslib/Simulation.h>

namespace FBLAS {

	using namespace hlslib;

	template <class T>
	class DatapackConverter {
	public:
		DatapackConverter(size_t N) : N(N) {
			#pragma HLS INLINE
		}

		template <bool dataflow = false, int width_from, int width_to>
		void convert(Stream<DataPack<T, width_from>> &in, Stream<DataPack<T, width_to>> &out) {
			#pragma HLS INLINE
			if (dataflow) {
				auto func = convertManyToMany<width_from, width_to>;
				HLSLIB_DATAFLOW_FUNCTION(func, N, in, out);
			} else {
				convertManyToMany(N, in, out);
			}
		}

		template <bool dataflow = false, int width_from>
		void convert(Stream<DataPack<T, width_from>> &in, Stream<T> &out) {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(convertManyToSimple<width_from>, N, in, out);
			} else {
				convertManyToSimple(N, in, out);
			}
		}

		template <bool dataflow = false, int width_to>
		void convert(Stream<T> &in, Stream<DataPack<T, width_to>> &out) {
			#pragma HLS INLINE
			if (dataflow) {
				HLSLIB_DATAFLOW_FUNCTION(convertSimpleToMany<width_to>, N, in, out);
			} else {
				convertSimpleToMany(N, in, out);
			}
		}

	private:
		size_t N;

		template <int width_from, int width_to>
		static void convertManyToMany(size_t N, Stream<DataPack<T, width_from>> &in, Stream<DataPack<T, width_to>> &out) {
			#pragma HLS INLINE
			//static_assert((width_from >= width_to && width_from % width_to == 0) || width_to % width_from == 0);

			size_t current_out = 0;
			DataPack<T, width_to> tmp = T();
			DatapackConverter_convert_Datapack_Datapack_loop:
			for (size_t i = 0; i < N; ++i) {
				#pragma HLS PIPELINE II=1
				auto val = in.Pop();
				for (int j = 0; j < width_from; ++j) {
					tmp[current_out++] = val[j];
					if (current_out >= width_to) {
						out.Push(tmp);
						current_out = 0;
					}
				}
				if (i == N - 1 && current_out < width_to) {
					out.Push(tmp);
				}
			}
		}

		template <int width_from>
		static void convertManyToSimple(size_t N, Stream<DataPack<T, width_from>> &in, Stream<T> &out) {
			#pragma HLS INLINE

			DatapackConverter_convert_Datapack_Simple_loop:
			for (size_t i = 0; i < N; ++i) {
				auto val = in.Pop();
				for (int j = 0; j < width_from; ++j) {
					#pragma HLS PIPELINE II=1
					out.Push(val[j]);
				}
			}
		}

		template <int width_to>
		static void convertSimpleToMany(size_t N, Stream<T> &in, Stream<DataPack<T, width_to>> &out) {
			#pragma HLS INLINE

			size_t current_out = 0;
			DataPack<T, width_to> tmp = T();
			DatapackConverter_convert_Simple_Datapack_loop:
			for (size_t i = 0; i < N; ++i) {
				#pragma HLS PIPELINE II=1
				auto val = in.Pop();
				tmp[current_out++] = val;
				if (current_out >= width_to || i == N - 1) {
					out.Push(tmp);
					current_out = 0;
					tmp = T();
				}
			}
		}
	};
}

#endif //BLAS_HLS_DATAPACKCONVERTER_H
