//
// Created by stpascal on 31.10.18.
//

#include <hlslib/SDAccel.h>
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <fblas.h>

#define BLAS_WIDTH  16

using namespace hlslib;

static ocl::Context context;

template <class T>
ocl::Program &getProgram();

template <>
ocl::Program &getProgram<float>() {
	static ocl::Program program(context, "kernel_fblas_single.xclbin");
	return program;
}

template <>
ocl::Program &getProgram<double>() {
	static ocl::Program program(context, "kernel_fblas_double.xclbin");
	return program;
}

template <class T, size_t width = BLAS_WIDTH>
class AlignedBuffer {
public:
	AlignedBuffer(size_t N, T *data, bool writeback = !std::is_const<T>()) : realN(N), writeback(writeback), origin(data), data_ptr(data) {
		this->N = roundUp(realN, width);
		if (this->N != realN) {
			_data.resize(this->N);
			std::copy(data_ptr, data_ptr + realN, _data.begin());
			data_ptr = _data.data();
		}
	}

	~AlignedBuffer() {
		if (!std::is_const<T>() && writeback && N != realN) {
			// The const_cast is necessary to eliminate any compilation error.
			// This is secure because the copy is only available when T is not const.
			std::copy(_data.begin(), _data.begin() + realN, const_cast<Type*>(origin));
		}
	}

	constexpr T *begin() const {
		return data_ptr;
	}

	constexpr T *end() const {
		return data_ptr + N;
	}

	constexpr size_t size() const {
		return N;
	}

private:
	using Type = typename std::remove_const<T>::type;

	const size_t realN;
	size_t N;
	const bool writeback;
	std::vector<Type> _data;
	T *origin, *data_ptr;

	static size_t roundUp(size_t size, size_t multiple) {
		auto remainder = size % multiple;
		if (remainder == 0) return size;
		return size + multiple - remainder;
	}
};

// Level 1
template <class T>
static void xaxpy(const std::string &func, int N, T ALPHA, const T *X, int INCX, T *Y, int INCY) {
	AlignedBuffer<const T> bufferX((size_t)N, X);
	AlignedBuffer<T> bufferY((size_t) N, Y, true);

	auto deviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, bufferX.begin(), bufferX.end());
	auto deviceY = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank1, bufferY.begin(), bufferY.end());
	auto deviceOut = context.MakeBuffer<T, ocl::Access::write>(ocl::MemoryBank::bank2, bufferY.begin(), bufferY.end());
	auto kernel = getProgram<T>().MakeKernel(func, static_cast<int>(bufferX.size()), ALPHA, deviceX, INCX, deviceY, INCY, deviceOut, INCY);
	kernel.ExecuteTask();
	deviceOut.CopyToHost(bufferY.begin());
}

void BLAS_FUNCTION(saxpy)(int N, float ALPHA, const float *X, int INCX, float *Y, int INCY) {
	xaxpy("saxpy", N, ALPHA, X, INCX, Y, INCY);
}

void BLAS_FUNCTION(daxpy)(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY) {
	xaxpy("daxpy", N, ALPHA, X, INCX, Y, INCY);
}

template <class T>
static T xdot(const std::string &func, int N, const T *X, int INCX, const T *Y, int INCY) {
	if (INCX != 1 || INCY != 1) {
		throw std::runtime_error("INCX and INCY are currently not supported");
	}

	std::array<T, 1> result = {0};
	AlignedBuffer<const T> bufferX((size_t)N, X), bufferY((size_t)N, Y);

	auto deviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, bufferX.begin(), bufferX.end());
	auto deviceY = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank1, bufferY.begin(), bufferY.end());
	auto deviceOut = context.MakeBuffer<T, ocl::Access::write>(ocl::MemoryBank::bank2, result.begin(), result.end());
	auto kernel = getProgram<T>().MakeKernel(func, static_cast<int>(bufferX.size()), deviceX, deviceY, deviceOut);
	kernel.ExecuteTask();
	deviceOut.CopyToHost(result.data());

	return result[0];
}

float BLAS_FUNCTION(sdot)(int N, const float *X, int INCX, const float *Y, int INCY) {
	return xdot("sdot", N, X, INCX, Y, INCY);
}

double BLAS_FUNCTION(ddot)(int N, const double *X, int INCX, const double *Y, int INCY) {
	return xdot("ddot", N, X, INCX, Y, INCY);
}

template <class T>
static T xnrm2(const std::string &func, int N, const T *X, int INCX) {
	std::array<T, 1> result = {0};
	AlignedBuffer<const T> bufferX((size_t)N, X);

	auto deviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, bufferX.begin(), bufferX.end());
	auto deviceOut = context.MakeBuffer<T, ocl::Access::write>(ocl::MemoryBank::bank1, result.begin(), result.end());
	auto kernel = getProgram<T>().MakeKernel(func, static_cast<int>(bufferX.size()), deviceX, INCX, deviceOut);
	kernel.ExecuteTask();
	deviceOut.CopyToHost(result.data());

	return result[0];
}

float BLAS_FUNCTION(snrm2)(int N, const float *X, int INCX) {
	return xnrm2("snrm2", N, X, INCX);
}

double BLAS_FUNCTION(dnrm2)(int N, const double *X, int INCX) {
	return xnrm2("dnrm2", N, X, INCX);
}

/*
 * Level 2
 */

template <class T>
static void xgemv(std::string func, char TRANS, int M, int N, T ALPHA, const T *A, int LDA, const T *X, int INCX, T BETA, T *Y, int INCY) {
	int lengthX, lengthY;
	char transposed = static_cast<char>(tolower(TRANS));
	if (LDA != N) {
		std::cerr << "LDA != N is not (yet) supported!" << std::endl;
		abort();
	}
	if (transposed == 'n') {
		lengthX = 1 + (N - 1) * abs(INCX);
		lengthY = 1 + (M - 1) * abs(INCY);
	} else if (transposed == 't' || transposed == 'c') {
		lengthX = 1 + (M - 1) * abs(INCX);
		lengthY = 1 + (N - 1) * abs(INCY);
		func += "_transposed";
	} else {
		std::cerr << "Unknown mode: " << TRANS << std::endl;
		abort();
	}

	AlignedBuffer<const T> bufferA((size_t) N * M, A), bufferX((size_t) lengthX, X);
	AlignedBuffer<T, 1> bufferY((size_t) lengthY, Y);

	auto deviceA = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, bufferA.begin(), bufferA.end());
	auto deviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank1, bufferX.begin(), bufferX.end());
	auto deviceY = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank2, bufferY.begin(), bufferY.end());
	auto deviceOut = context.MakeBuffer<T, ocl::Access::write>(ocl::MemoryBank::bank3, bufferY.begin(), bufferY.end());

	auto kernel = getProgram<T>().MakeKernel(func, M, N, ALPHA, deviceA, deviceX, INCX, BETA, deviceY, INCY, deviceOut, INCY);
	kernel.ExecuteTask();
	deviceOut.CopyToHost(bufferY.begin());
}

void BLAS_FUNCTION(sgemv)(char TRANS, int M, int N, float ALPHA, const float *A, int LDA, const float *X, int INCX,
                           float BETA, float *Y, int INCY) {
	xgemv("sgemv", TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
}

void BLAS_FUNCTION(dgemv)(char TRANS, int M, int N, double ALPHA, const double *A, int LDA, const double *X, int INCX,
                           double BETA, double *Y, int INCY) {
	xgemv("dgemv", TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
}
