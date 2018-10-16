//
// Created by stpascal on 01.07.18.
//

#include <iostream>
#include <vector>
#include <array>
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
		freq = other.template getFreq<period>();
	}

	template <class outPeriod = period>
	double getFreq() const {
		typedef ratio_divide<period, outPeriod> r;
		return freq * r::num / r::den;
	}

private:
	double freq;
};

//const auto freq = Frequency<mega>(300.0);
const auto freq = Frequency<mega>(206.9);

int main(int argc, const char *argv[]) {
	const size_t size = static_cast<size_t>(argc > 1 ? strtod(argv[1], NULL) : 5e8);
	if (size % 16 != 0) {
		cerr << "Size must be divisible by 16!" << endl;
		return 1;
	}
	if (size % 1000 != 0) {
		cerr << "Size must be divisible by 1000!" << endl;
		return 1;
	}
	const size_t chunkwidth = static_cast<size_t>(argc > 2 ? strtod(argv[2], NULL) : 1) * 16;
	if (size % chunkwidth != 0) {
		cerr << "Size must be divisible by chunkwidth!" << endl;
		return 1;
	}

	vector<Type_t> inputHostX(size, 2), inputHostY(size, 1);
	array<Type_t, 1> outputHost = {0};

	cout << "size = " << size << endl;
	cout << "chunkwidth = " << chunkwidth << endl;

	ocl::Context context;

	auto inputDeviceX = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank0, inputHostX.cbegin(), inputHostX.cend());
	auto inputDeviceY = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank1, inputHostY.cbegin(), inputHostY.cend());
	auto outputDevice = context.MakeBuffer<Type_t, ocl::Access::write>(ocl::MemoryBank::bank0, outputHost.begin(), outputHost.end());

	auto program = context.MakeProgram("kernel_dot.xclbin");

	cout << "Actual frequency: " << freq.getFreq<mega>() << " MHz" << endl;

	this_thread::sleep_for(chrono::seconds(1));

	auto kernel = program.MakeKernel("blas_dot", size, chunkwidth, inputDeviceX, inputDeviceY, outputDevice);
	this_thread::sleep_for(chrono::milliseconds(1000));
	cout << "Executing kernel..." << endl;
	auto time = kernel.ExecuteTask();

	cout << "Kernel of size " << size << " executed in " << scientific << time.first << " seconds" << endl;

	return 0;
}
