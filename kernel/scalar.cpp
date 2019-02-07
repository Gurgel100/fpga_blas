//
// Created by stpascal on 01.02.19.
//

#include <hlslib/DataPack.h>
#include <hlslib/Stream.h>
#include <hlslib/Simulation.h>
#include <Config.h>
#include <Scalar.h>

#ifndef WIDTH
#define WIDTH   16
#endif

using hlslib::Stream;

using Scalar = FBLAS::PackedScalar<Data_t, WIDTH>;

extern "C"
void scalar(const size_t N, const Data_t alpha, const hlslib::DataPack<Data_t, WIDTH> memoryIn_X[], hlslib::DataPack<Data_t, WIDTH> memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_X offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem1
	#pragma HLS INTERFACE s_axilite port=memoryIn_X bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=alpha bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	Stream<hlslib::DataPack<Data_t, WIDTH>> inX("scalar_inX"), out("scalar_out");

	HLSLIB_DATAFLOW_INIT();

	Scalar scalar(N, inX, out);
	scalar.getReaderX().readFromMemory<true>(memoryIn_X);
	scalar.calc<true>(alpha);
	scalar.getWriterX().writeToMemory<true>(memoryOut);

	HLSLIB_DATAFLOW_FINALIZE();
}
