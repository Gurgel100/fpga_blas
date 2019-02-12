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

const auto freq = Frequency<mega>(200.0);

int main(int argc, const char *argv[]) {
	const size_t size = argc > 1 ? strtoul(argv[1], NULL, 0)  : static_cast<size_t>(1e7);
	const size_t numProducts = argc > 2 ? strtoul(argv[2], NULL, 0) : 16;
	vector<Type_t> inputHostX(numProducts * size, 2), inputHostY(numProducts * size, 1);

	cout << "maxSize = " << size << ", numProducts = " << numProducts << endl;

	ocl::Context context;

	auto inputDeviceX = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank0, numProducts * size);
	auto inputDeviceY = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank1, numProducts * size);
	auto outputDevice = context.MakeBuffer<Type_t, ocl::Access::write>(ocl::MemoryBank::bank0, numProducts);

	inputDeviceX.CopyFromHost(inputHostX.data());
	inputDeviceY.CopyFromHost(inputHostY.data());

	auto program = context.MakeProgram("kernel_dot_multiple.xclbin");

	cout << "Actual frequency: " << freq.getFreq<mega>() << " MHz" << endl;

	this_thread::sleep_for(chrono::seconds(1));

	auto kernel = program.MakeKernel("blas_dot_multiple", size, numProducts, inputDeviceX, inputDeviceY, outputDevice);
	this_thread::sleep_for(chrono::milliseconds(1000));
	auto time = kernel.ExecuteTask();

	vector<Type_t> res(numProducts);
	outputDevice.CopyToHost(res.data());

	cout << "Kernel executed in " << scientific << time.first << " seconds" << endl;
	for (const auto &r : res) {
		cout << "\t" << r << endl;
	}

	return 0;
}
