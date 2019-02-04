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
	const size_t size = static_cast<size_t>(argc > 1 ? strtod(argv[1], NULL) : 5e6);
	if (size % 32 != 0) {
		cerr << "Size must be divisible by 32!" << endl;
		return 1;
	}
	const std::string kernel_file = (argc > 2 ? argv[2] : "kernel_dot.xclbin");

	vector<Type_t, ocl::AlignedAllocator<Type_t, 4096>> inputHostX(size, 2), inputHostY(size, 1), outputHost(1);

	ocl::Context context;

	auto inputDeviceX = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank0, inputHostX.cbegin(), inputHostX.cend());
	auto inputDeviceY = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank1, inputHostY.cbegin(), inputHostY.cend());
	auto outputDevice = context.MakeBuffer<Type_t, ocl::Access::write>(ocl::MemoryBank::bank0, outputHost.begin(), outputHost.end());

	auto program = context.MakeProgram(kernel_file);

	auto kernel = program.MakeKernel("blas_dot", size, inputDeviceX, inputDeviceY, outputDevice);
	cout << "Executing kernel..." << endl;
	auto time = kernel.ExecuteTask().first;

	outputDevice.CopyToHost(outputHost.begin());

	auto res = cblas_sdot(size, inputHostX.data(), 1, inputHostY.data(), 1);
	if (abs(res - outputHost[0]) > numeric_limits<Type_t>::epsilon() * (max(res, outputHost[0]))) {
		cerr << "Got wrong result: " << outputHost[0] << " instead of " << res << endl;
	}

	cout << "csv_start" << endl << "size, time, read data, written data" << endl;
	cout << size << ',' << time << ',' << size * sizeof(Type_t) * 2 << ',' << sizeof(Type_t) << endl;

	return 0;
}
