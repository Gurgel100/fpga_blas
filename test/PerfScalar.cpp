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
#include <cblas.h>
#include "hlslib/SDAccel.h"

using namespace std;
using namespace hlslib;

using Type_t = float;

int main(int argc, const char *argv[]) {
	const size_t size = static_cast<size_t>(argc > 1 ? strtod(argv[1], NULL) : 1e5);
	if (size % 32 != 0) {
		cerr << "Size must be divisible by 32!" << endl;
		return 1;
	}
	const std::string kernel_file = (argc > 2 ? argv[2] : "kernel_scalar.xclbin");

	vector<Type_t, ocl::AlignedAllocator<Type_t, 4096>> inputHost(size, 0.45), outputHost(size);
	Type_t alpha = 3.1415;

	ocl::Context context;

	auto inputDevice = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank0, inputHost.cbegin(), inputHost.cend());
	auto outputDevice = context.MakeBuffer<Type_t, ocl::Access::write>(ocl::MemoryBank::bank1, outputHost.begin(), outputHost.end());

	auto program = context.MakeProgram(kernel_file);

	auto kernel = program.MakeKernel("scalar", size, alpha, inputDevice, outputDevice);
	cout << "Executing kernel..." << endl;
	auto time = kernel.ExecuteTask().first;

	outputDevice.CopyToHost(outputHost.begin());

	cblas_sscal(size, alpha, inputHost.data(), 1);
	for (size_t i = 0; i < size; ++i) {
		if (abs(inputHost[i] - outputHost[i]) > numeric_limits<Type_t>::epsilon() * (max(inputHost[i], outputHost[i]))) {
			cerr << "Got wrong result: " << outputHost[i] << " instead of " << inputHost[i] << endl;
		}
	}

	cout << "csv_start" << endl << "size, time, read data, written data" << endl;
	cout << size << ',' << time << ',' << size * sizeof(Type_t) << ',' << size * sizeof(Type_t) << endl;

	return 0;
}
