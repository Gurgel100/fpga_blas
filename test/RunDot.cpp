//
// Created by stpascal on 14.06.18.
//

#include "hlslib/SDAccel.h"
#include <iostream>
#include <vector>
#include <random>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using namespace hlslib;

template <class T>
static T reference_dot(const std::vector<T> &x, const std::vector<T> &y, const size_t N)
{
    T res = 0;
    for (size_t i = 0; i < N; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

template <typename T>
static void test_dot(ocl::Context &context, const std::string &kernelFile, const std::vector<size_t> &sizes) {
	const size_t maxSize = sizes.back();

	std::cout << "Initializing device memory..." << std::flush;
	auto inputDeviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, maxSize * sizeof(T));
	auto inputDeviceY = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank1, maxSize * sizeof(T));
	auto outputDevice = context.MakeBuffer<T, ocl::Access::write>(ocl::MemoryBank::bank0, sizeof(T));
	std::cout << " Done." << std::endl;

	std::cout << "Initialising input data..." << std::flush;
	std::vector<T> inputHostX(maxSize);
	std::vector<T> inputHostY(maxSize);
	std::vector<T> outputHost(1);
	std::random_device rd;
	std::default_random_engine re(rd());
	std::uniform_real_distribution<T> dist(-10, 10);

	for (size_t i = 0; i < maxSize; ++i) {
		inputHostX[i] = dist(re);
		inputHostY[i] = dist(re);
	}
	std::cout << " Done." << std::endl;

	std::cout << "Copying input to device..." << std::flush;
	inputDeviceX.CopyFromHost(inputHostX.data());
	inputDeviceY.CopyFromHost(inputHostY.data());
	std::cout << " Done." << std::endl;

	std::cout << "Creating program..." << std::flush;
	auto program = context.MakeProgram(kernelFile);
	std::cout << " Done." << std::endl;

	for (auto &size : sizes) {
		std::cout << "Executing kernel for size " << size << "..." << std::flush;
		auto kernel = program.MakeKernel("blas_dot", inputDeviceX, 1, inputDeviceY, 1, outputDevice, size);
		auto elapsed = kernel.ExecuteTask();
		std::cout << " Done." << std::endl;

		std::cout << "Kernel ran in " << elapsed.first << " seconds.\n";

		std::cout << "Verifying result..." << std::flush;
		outputDevice.CopyToHost(outputHost.data());
		REQUIRE(outputHost[0] == reference_dot(inputHostX, inputHostY, size));
		std::cout << " Done." << std::endl;

		std::cout << "Kernel ran successfully." << std::endl;
	}
}

TEST_CASE("Dot", "[Dot]") {
	std::vector<size_t> sizes;
	for (size_t i = 0; i < 32; ++i) {
		sizes.push_back(i);
	}

	std::cout << "Initializing OpenCL context..." << std::endl;
	ocl::Context context;
	std::cout << "Done." << std::endl;

	SECTION("Double") {
		test_dot<double>(context, "kernel_dot", sizes);
	}
}
