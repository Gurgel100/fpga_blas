#include <hlslib/DataPack.h>
#include "hlslib/Stream.h"
#include "hlslib/Simulation.h"
#include "hlslib/TreeReduce.h"
#include "hlslib/Operators.h"
#include "DotProduct.h"
#include "dot.h"

using hlslib::Stream;

using DotProduct = FBLAS::DotProduct<Data_t, dot_width>;
using DotProductInterleaved = FBLAS::DotProductInterleaved<Data_t, dot_width>;

void blas_dot(const size_t N, DotProduct::Chunk const memoryIn_X[], DotProduct::Chunk const memoryIn_Y[], Data_t memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	Stream<DotProduct::Chunk> inX("blas_dot_inX"), inY("blas_dot_inY");
	Stream<Data_t> out("blas_dot_out");

	HLSLIB_DATAFLOW_INIT();

	DotProduct dotProduct(N, inX, inY, out);
	dotProduct.getReaderX().readFromMemory<true>(memoryIn_X, 1);
	dotProduct.getReaderY().readFromMemory<true>(memoryIn_Y, 1);
	dotProduct.calc<true>();
	dotProduct.getWriter().writeToMemory<true>(memoryOut);

	HLSLIB_DATAFLOW_FINALIZE();
}

void blas_dot_multiple(const size_t N, const size_t n_prod, DotProductInterleaved::Chunk const memoryIn_X[], DotProductInterleaved::Chunk const memoryIn_Y[], Data_t memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_Y offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_Y bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=n_prod bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	Stream<DotProductInterleaved::Chunk> inX("blas_dot_multiple_inX"), inY("blas_dot_multiple_inY");
	Stream<Data_t> out("blas_dot_multiple_out");

	HLSLIB_DATAFLOW_INIT();

	DotProductInterleaved dotProduct(N, n_prod, inX, inY, out);
	dotProduct.getReaderX().readFromMemory<true>(memoryIn_X);
	dotProduct.getReaderY().readFromMemory<true>(memoryIn_Y);
	dotProduct.calc<true>();
	dotProduct.getWriter().writeToMemory<true>(memoryOut);

	HLSLIB_DATAFLOW_FINALIZE();
}
