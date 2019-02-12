//
// Created by stpascal on 19.12.18.
//

#include <dot.h>
#include <Matrix.h>
#include <VectorElementOperation.h>
#include <Scalar.h>
#include <DatapackConverter.h>
#include <Norm.h>

constexpr auto width = 1;

template <class T>
using Type = hlslib::DataPack<T, width>;

// (Ax . Bx) / ||(A + 3.1419 * B)x||

template <typename T>
struct Divide {
	static T Apply(T const &a, T const &b) {
		#pragma HLS INLINE
		return a / b;
	}
	static constexpr T identity() { return 1; }
private:
	Divide() = delete;
	~Divide() = delete;
};

extern "C"
void benchmark(const size_t N, const size_t M, const Type<float> memoryIn_A[], const Type<float> memoryIn_B[], const Type<float> memoryIn_x[], Type<float> memoryOut[]) {
	#pragma HLS INTERFACE m_axi port=memoryIn_A offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=memoryIn_B offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=memoryIn_x offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=memoryOut offset=slave bundle=gmem3
	#pragma HLS INTERFACE s_axilite port=memoryIn_A bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_B bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryIn_x bundle=control
	#pragma HLS INTERFACE s_axilite port=memoryOut bundle=control
	#pragma HLS INTERFACE s_axilite port=N bundle=control
	#pragma HLS INTERFACE s_axilite port=M bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW

	using MatrixVectorMultiplication_t = FBLAS::MatrixVectorMultiplication<float, width, width>;

	HLSLIB_DATAFLOW_INIT();

	hlslib::Stream<Type<float>> inA("inA"), inB("inB"), inX("inX"), inX1("inX1"), inX2("inX2"), AxPacked("AxPacked"),
								BxPacked("BxPacked"), Ax1("Ax1"), Ax2("Ax2", 2), Bx1("Bx1"), Bx2("Bx2"), scaledBx("scaledBx"), summedAxBx("summedAxBx");
	hlslib::Stream<float> Ax("Ax"), Bx("Bx"), resDot("resDot"), normOut("normOut"), out("out");

	FBLAS::StreamDuplicator<Type<float>> duplicatorType(M / width);
	FBLAS::StreamDuplicator<Type<float>> duplicatorX(N / MatrixVectorMultiplication_t::colchunk_size * hlslib::CeilDivide(M,  MatrixVectorMultiplication_t::rowchunk_size));
	FBLAS::DatapackConverter<float> converterM(M);

	MatrixVectorMultiplication_t matrixVectorMultiplicationReader(N, M, inA, inX, Ax);
	MatrixVectorMultiplication_t matrixVectorMultiplicationAx(N, M, inA, inX1, Ax);
	MatrixVectorMultiplication_t matrixVectorMultiplicationBx(N, M, inB, inX2, Bx);

	FBLAS::PackedScalar<float, width> scalarBx(M, /*Bx2*/BxPacked, scaledBx);

	FBLAS::DotProduct<float, width> dotProduct(M, Ax1, Bx1, resDot);
	FBLAS::PackedNorm<float, width> norm(M, summedAxBx, normOut);

	FBLAS::VectorElementOperation<Type<float>, hlslib::op::Add<Type<float>>> sumAxBx(M / width, /*Ax2*/AxPacked, scaledBx, summedAxBx);
	FBLAS::VectorElementOperation<float, Divide<float>> finalOp(1, resDot, normOut, out);

	matrixVectorMultiplicationReader.getReaderX().readFromMemory<true>(memoryIn_x);

	duplicatorX.duplicate<true>(inX, inX1, inX2);

	matrixVectorMultiplicationAx.getReaderA().readFromMemory<true>(memoryIn_A);
	matrixVectorMultiplicationBx.getReaderA().readFromMemory<true>(memoryIn_B);

	matrixVectorMultiplicationAx.calc<true>();
	matrixVectorMultiplicationBx.calc<true>();

	converterM.convert<true>(Ax, AxPacked);
	converterM.convert<true>(Bx, BxPacked);

	//duplicatorType.duplicate<true>(AxPacked, Ax1, Ax2);
	//duplicatorType.duplicate<true>(BxPacked, Bx1, Bx2);

	scalarBx.calc<true>(3.1419);

	//dotProduct.calc<true>();

	sumAxBx.calc<true>();
	sumAxBx.getWriter().writeToMemory<true>(memoryOut);

	//norm.calc<true>();

	/*for (int i = 0; i < 1; ++i) {
		#pragma HLS PIPELINE II=1
		auto dot_res = resDot.Pop();
		auto nrm_res = normOut.Pop();
		memoryOut[0] = dot_res;
		memoryOut[1] = nrm_res;
		memoryOut[2] = dot_res / nrm_res;
		memoryOut[3] = nrm_res / dot_res;
	}*/

	//finalOp.calc<true>();
	//finalOp.getWriter().writeToMemory<true>(memoryOut);

	HLSLIB_DATAFLOW_FINALIZE();
}
