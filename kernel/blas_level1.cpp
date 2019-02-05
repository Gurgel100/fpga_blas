//
// Created by stpascal on 23.08.18.
//

#include <hlslib/Simulation.h>
#include <dot.h>
#include <Scalar.h>
#include <VectorElementOperation.h>
#include <Norm.h>
#include <fblas.h>

using namespace FBLAS;
using namespace hlslib;

template <class T>
using Type = hlslib::DataPack<T, WIDTH>;

/*
 * Scalar
 */
template <class T>
static void _scal(const size_t N, const T alpha, const Type<T> memoryIn_X[], const size_t incX, Type<T> memoryOut[]) {
#pragma HLS INLINE

	Stream<Type<T>> inX("scal_inX"), out("scal_out");
	PackedScalar<T, WIDTH> scalar(N, inX, out);

	HLSLIB_DATAFLOW_INIT();
	scalar.getReaderX().template readFromMemory<true>(memoryIn_X, incX);
	scalar.template calc<true>(alpha);
	scalar.getWriterX().template writeToMemory<true>(memoryOut);
	HLSLIB_DATAFLOW_FINALIZE();
}

#ifndef FBLAS_DISABLE_SINGLE
extern "C"
void sscal(const int N, const float alpha, const Type<float> memoryIn_X[], int incX, Type<float> memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem1
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=alpha bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_scal(static_cast<const size_t>(N), alpha, memoryIn_X, static_cast<const size_t>(incX), memoryOut);
}
#endif

#ifndef FBLAS_DISABLE_DOUBLE
extern "C"
void dscal(const int N, const double alpha, const Type<double> memoryIn_X[], int incX, Type<double> memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem1
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=alpha bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_scal(static_cast<const size_t>(N), alpha, memoryIn_X, static_cast<const size_t>(incX), memoryOut);
}
#endif


/*
 * AXPY
 */
template <class T>
static void _axpy(const size_t N, const T a, const Type<T> memoryIn_X[], const size_t incX, const Type<T> memoryIn_Y[], const size_t incY, Type<T> memoryOut[], const size_t incOut) {
	#pragma HLS INLINE

	Stream<Type<T>> inX("axpy_inX"), intermediateX("aspy_intermediateX"), inY("axpy_inY"), out("axpy_out");
	PackedScalar<T, WIDTH> scalar(N, inX, intermediateX);
	VectorElementOperation<Type<T>, hlslib::op::Add<Type<T>>> vectorOperation(N, intermediateX, inY, out);

	HLSLIB_DATAFLOW_INIT();
	scalar.getReaderX().template readFromMemory<true>(memoryIn_X, incX);
	scalar.template calc<true>(a);
	vectorOperation.getReaderY().template readFromMemory<true>(memoryIn_Y, incY);
	vectorOperation.template calc<true>();
	vectorOperation.getWriter().template writeToMemory<true>(memoryOut, incOut);
	HLSLIB_DATAFLOW_FINALIZE();
}

#ifndef FBLAS_DISABLE_SINGLE
extern "C"
void saxpy(const int N, const float a, const Type<float> memoryIn_X[], int incX, const Type<float> memoryIn_Y[], int incY, Type<float> memoryOut[], int incOut) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=a bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=incY bundle=control
	#pragma HLS INTERFACE s_axilite port=incOut bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_axpy(static_cast<const size_t>(N), a, memoryIn_X, static_cast<const size_t>(incX), memoryIn_Y,
	      static_cast<const size_t>(incY), memoryOut, static_cast<const size_t>(incOut));
}
#endif

#ifndef FBLAS_DISABLE_DOUBLE
extern "C"
void daxpy(const int N, const double a, const Type<double> memoryIn_X[], int incX, const Type<double> memoryIn_Y[], int incY, Type<double> memoryOut[], int incOut) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=a bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=incY bundle=control
	#pragma HLS INTERFACE s_axilite port=incOut bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_axpy(static_cast<const size_t>(N), a, memoryIn_X, static_cast<const size_t>(incX), memoryIn_Y,
	      static_cast<const size_t>(incY), memoryOut, static_cast<const size_t>(incOut));
}
#endif


/*
 * Dot Product
 */
template <class T>
static void _dot(const size_t N, typename DotProduct<T, WIDTH>::Chunk const memoryIn_X[], typename DotProduct<T, WIDTH>::Chunk const memoryIn_Y[], T memoryOut[1]) {
	#pragma HLS INLINE

	Stream<typename DotProduct<T, WIDTH>::Chunk> inX("dot_inX"), inY("dot_inY");
	Stream<T> out("dot_out");
	DotProduct<T, WIDTH> dotProduct(N, inX, inY, out);

	HLSLIB_DATAFLOW_INIT();
	dotProduct.getReaderX().template readFromMemory<true>(memoryIn_X);
	dotProduct.getReaderY().template readFromMemory<true>(memoryIn_Y);
	dotProduct.template calc<true>();
	dotProduct.getWriter().template writeToMemory<true>(memoryOut);
	HLSLIB_DATAFLOW_FINALIZE();
}

#ifndef FBLAS_DISABLE_SINGLE
extern "C"
void sdot(const int N, typename DotProduct<float, WIDTH>::Chunk const memoryIn_X[], typename DotProduct<float, WIDTH>::Chunk const memoryIn_Y[], float memoryOut[1]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_dot(static_cast<const size_t>(N), memoryIn_X, memoryIn_Y, memoryOut);
}
#endif

#ifndef FBLAS_DISABLE_DOUBLE
extern "C"
void ddot(const int N, typename DotProduct<double, WIDTH>::Chunk const memoryIn_X[], typename DotProduct<double, WIDTH>::Chunk const memoryIn_Y[], double memoryOut[1]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_dot(static_cast<const size_t>(N), memoryIn_X, memoryIn_Y, memoryOut);
}
#endif

/* Norm */
template <class T>
static void _nrm2(const size_t N, const Type<T> memoryIn_X[], const size_t incX, T memoryOut[]) {
	#pragma HLS INLINE

	Stream<Type<T>> inX("nrm2_inX");
	Stream<T> out("nrm2_out");
	PackedNorm<T, WIDTH> norm(N, inX, out);

	HLSLIB_DATAFLOW_INIT();
	norm.getReader().template readFromMemory<true>(memoryIn_X, incX);
	norm.template calc<true>();
	norm.getWriter().template writeToMemory<true>(memoryOut);
	HLSLIB_DATAFLOW_FINALIZE();
}

#ifndef FBLAS_DISABLE_SINGLE
extern "C"
void snrm2(const int N, const Type<float> memoryIn_X[], int incX, float memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem1
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_nrm2(static_cast<const size_t>(N), memoryIn_X, static_cast<const size_t>(incX), memoryOut);
}
#endif

#ifndef FBLAS_DISABLE_DOUBLE
extern "C"
void dnrm2(const int N, const Type<double> memoryIn_X[], int incX, double memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem1
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=incX bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	_nrm2(static_cast<const size_t>(N), memoryIn_X, static_cast<const size_t>(incX), memoryOut);
}
#endif
