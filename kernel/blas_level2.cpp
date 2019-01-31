//
// Created by stpascal on 23.08.18.
//

#include <hlslib/Simulation.h>
#include <Matrix.h>
#include <Scalar.h>
#include <VectorElementOperation.h>
#include <fblas.h>

#ifndef GEMV_SIZE_ROWCHUNK
#define GEMV_SIZE_ROWCHUNK  16
#endif

#ifndef GEMV_SIZE_COLCHUNK
#define GEMV_SIZE_COLCHUNK  16
#endif

using namespace FBLAS;
using namespace hlslib;

template <class T>
using Type = DataPack<T, WIDTH>;

/*
 * GEMV
 */

template <class T>
static void _gemv(
		size_t M,
		size_t N,
		T a,
		const Type<T> memoryIn_A[],
		const Type<T> memoryIn_X[],
		size_t incX,
		T b,
		const T memoryIn_Y[],
		size_t incY,
		T memoryOut[],
		size_t incOut) {
	#pragma HLS INLINE

	Stream<Type<T>> inA("gemv_inA"), inX("gemv_inX");
	Stream<T> Ax("gemv_Ax"), scaledAx("gemv_scaledAx"), scaledY("gemv_scaledY"), inY("gemv_inY"), out("gemv_out");
	Scalar<T> scalarAx(M, Ax, scaledAx), scalarY(M, inY, scaledY);
	VectorElementOperation<T, hlslib::op::Add<T>> vectorAddition(M, scaledAx, scaledY, out);
	MatrixVectorMultiplication<T, GEMV_SIZE_ROWCHUNK, GEMV_SIZE_COLCHUNK, WIDTH> matrixVectorMultiplicationAx(N, M, inA, inX, Ax);

	HLSLIB_DATAFLOW_INIT();
	matrixVectorMultiplicationAx.getReaderA().template readFromMemory<true>(memoryIn_A);
	matrixVectorMultiplicationAx.getReaderX().template readFromMemory<true>(memoryIn_X, incX);
	matrixVectorMultiplicationAx.template calc<true>();
	scalarAx.template calc<true>(a);

	scalarY.getReaderX().template readFromMemory<true>(memoryIn_Y, incY);
	scalarY.template calc<true>(b);

	vectorAddition.template calc<true>();
	vectorAddition.getWriter().template writeToMemory<true>(memoryOut, incOut);

	HLSLIB_DATAFLOW_FINALIZE();
}

extern "C" void sgemv(
		const int M,
		const int N,
		const float a,
		const Type<float> memoryIn_A[],
		const Type<float> memoryIn_X[],
		int incX,
		const float b,
		const float memoryIn_Y[],
		int incY,
		float memoryOut[],
		int incOut) {
	#pragma HLS INTERFACE m_axi port=memoryIn_A offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem3
	#pragma HLS INTERFACE s_axilite port=memoryIn_A bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=M bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=a bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=b bundle=control
	#pragma HLS INTERFACE s_axilite port=incY bundle=control
	#pragma HLS INTERFACE s_axilite port=incOut bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_gemv(static_cast<const size_t>(M), static_cast<const size_t>(N), a, memoryIn_A, memoryIn_X, static_cast<const size_t>(incX),
			b, memoryIn_Y, static_cast<const size_t>(incY), memoryOut, static_cast<const size_t>(incOut));
}

extern "C" void dgemv(
		const int M,
		const int N,
		const double a,
		const Type<double> memoryIn_A[],
		const Type<double> memoryIn_X[],
		int incX,
		const double b,
		const double memoryIn_Y[],
		int incY,
		double memoryOut[],
		int incOut) {
	#pragma HLS INTERFACE m_axi port=memoryIn_A offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem3
	#pragma HLS INTERFACE s_axilite port=memoryIn_A bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=M bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=a bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=b bundle=control
	#pragma HLS INTERFACE s_axilite port=incY bundle=control
	#pragma HLS INTERFACE s_axilite port=incOut bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_gemv(static_cast<const size_t>(M), static_cast<const size_t>(N), a, memoryIn_A, memoryIn_X, static_cast<const size_t>(incX),
	      b, memoryIn_Y, static_cast<const size_t>(incY), memoryOut, static_cast<const size_t>(incOut));
}
