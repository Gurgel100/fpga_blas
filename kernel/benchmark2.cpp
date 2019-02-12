//
// Created by stpascal on 19.12.18.
//

#include <dot.h>
#include <Matrix.h>
#include <VectorElementOperation.h>
#include <Scalar.h>
#include <DatapackConverter.h>
#include <Norm.h>

//TMP
#define WIDTH   32
#define SIZE_ROWCHUNK   64
#define SIZE_COLCHUNK   256

#ifndef DATATYPE
#define DATATYPE    float
#endif

#ifndef WIDTH
#define WIDTH   16
#endif

#ifndef SIZE_ROWCHUNK
#define SIZE_ROWCHUNK    16
#endif

#ifndef SIZE_COLCHUNK
#define SIZE_COLCHUNK   16
#endif

static_assert(SIZE_COLCHUNK % WIDTH == 0, "SIZE_COLCHUNK must be divisible by WIDTH");

constexpr auto width = WIDTH;

template <class T>
using Type = hlslib::DataPack<T, width>;

// Ax . y

extern "C"
void benchmark2(const size_t N, const size_t M, const Type<DATATYPE> memoryIn_A[], const DATATYPE memoryIn_x[], const Type<DATATYPE> memoryIn_y[], DATATYPE memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_A offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_x offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryIn_y offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem3
	#pragma HLS INTERFACE s_axilite port=memoryIn_A bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_x bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=M bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	using MatrixVectorMultiplication_t = FBLAS::MatrixVectorMultiplicationTransposed<DATATYPE, SIZE_ROWCHUNK, SIZE_COLCHUNK, width>;

	hlslib::Stream<Type<DATATYPE>> inA("inA"), inY("inY"), Ax("Ax");
	hlslib::Stream<DATATYPE> inX("inX"), out("out");

	HLSLIB_DATAFLOW_INIT();

	MatrixVectorMultiplication_t matrixVectorMultiplicationAx(N, M, inA, inX, Ax);

	FBLAS::DotProduct<DATATYPE, width> dotProduct(N, Ax, inY, out);

	matrixVectorMultiplicationAx.getReaderA().readFromMemory<true>(memoryIn_A);
	matrixVectorMultiplicationAx.getReaderX().readFromMemory<true>(memoryIn_x);

	matrixVectorMultiplicationAx.calc<true>();

	dotProduct.getReaderY().readFromMemory<true>(memoryIn_y);

	dotProduct.calc<true>();

	dotProduct.getWriter().writeToMemory<true>(memoryOut);

	HLSLIB_DATAFLOW_FINALIZE();
}
