//
// Created by stpascal on 01.07.18.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <ratio>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <chrono>
#include "hlslib/SDAccel.h"

using namespace std;
using namespace hlslib;

using Type_t = float;

template <class _Period>
struct Frequency {
	typedef _Period period;

	explicit Frequency(double freq) : freq(freq) {}

	template <class periodOther>
	explicit Frequency(const Frequency<periodOther> &other) {
		freq = other.getFreq<period>();
	}

	template <class outPeriod = period>
	double getFreq() const {
		typedef ratio_divide<period, outPeriod> r;
		return freq * r::num / r::den;
	}

private:
	double freq;
};

const auto freq = Frequency<mega>(256.4);

static size_t roundUp(size_t n, size_t multiple) {
	size_t rem = n % multiple;
	if (rem != 0 && multiple != 0)
		return n + multiple - rem;
	else
		return n;
}

int main(int argc, const char *argv[]) {
	const size_t N = roundUp(static_cast<size_t>(argc > 1 ? strtod(argv[1], NULL)  : 1e4), 16);
	const size_t M = roundUp(static_cast<size_t>(argc > 2 ? strtod(argv[2], NULL) : 1e4), 16);
	vector<Type_t> inputHostA(N * M, 1), inputHostX(N, 1), outputHost(M);

	cout << "N = " << N << ", M = " << M << endl;

	ocl::Context context;

	auto inputDeviceA = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank0, inputHostA.cbegin(), inputHostA.cend());
	auto inputDeviceX = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank1, inputHostX.cbegin(), inputHostX.cend());
	auto outputDevice = context.MakeBuffer<Type_t, ocl::Access::write>(ocl::MemoryBank::bank2, outputHost.begin(), outputHost.end());

	auto program = context.MakeProgram("kernel_gemv.xclbin");

	cout << "Actual frequency: " << freq.getFreq<mega>() << " MHz" << endl;

	this_thread::sleep_for(chrono::seconds(1));

	auto kernel = program.MakeKernel("blas_gemv", inputDeviceA, inputDeviceX, outputDevice, N, M);
	cout << "Executing kernel..." << endl;
	auto time = kernel.ExecuteTask();

	cout << "Kernel executed in " << scientific << time.first << " seconds" << endl;

	return 0;
}
