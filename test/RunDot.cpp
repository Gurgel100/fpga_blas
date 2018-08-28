//
// Created by stpascal on 14.06.18.
//

#include "hlslib/SDAccel.h"
#include <iostream>
#include <vector>
#include <random>
#include <strstream>
#include <cblas.h>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using namespace hlslib;

static double reference_dot(const std::vector<double> &x, const std::vector<double> &y, const size_t N)
{
	return cblas_ddot(static_cast<int>(N), x.data(), 1, y.data(), 1);
}

static float reference_dot(const std::vector<float> &x, const std::vector<float> &y, const size_t N)
{
	return cblas_sdot(static_cast<int>(N), x.data(), 1, y.data(), 1);
}

template <typename T>
static void test_dot(ocl::Context &context, const std::string &kernelFile, const std::vector<size_t> &sizes) {
	const size_t maxSize = sizes.back();

	std::vector<T> inputHostX(maxSize);
	std::vector<T> inputHostY(maxSize);
	std::array<T, 1> outputHost;
	std::random_device rd;
	std::default_random_engine re(rd());
	std::uniform_real_distribution<T> dist(-10, 10);

	for (size_t i = 0; i < maxSize; ++i) {
		inputHostX[i] = dist(re);
		inputHostY[i] = dist(re);
	}

	auto inputDeviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, inputHostX.cbegin(), inputHostX.cend());
	auto inputDeviceY = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank1, inputHostY.cbegin(), inputHostY.cend());
	auto outputDevice = context.MakeBuffer<T, ocl::Access::write>(ocl::MemoryBank::bank0, outputHost.begin(), outputHost.end());

	auto program = context.MakeProgram(kernelFile + ".xclbin");

	for (auto &size : sizes) {
		SECTION("Vector of size " + std::to_string(size)) {
			auto kernel = program.MakeKernel("blas_dot", size, inputDeviceX, 1, inputDeviceY, 1, outputDevice);
			kernel.ExecuteTask();

			outputDevice.CopyToHost(outputHost.data());
			auto reference = reference_dot<T>(inputHostX, inputHostY, size);
			REQUIRE(outputHost[0] == Approx(reference));
		}
	}
}

template <typename T>
static void test_interleaved_dot(ocl::Context &context, const std::string &kernelFile, const std::vector<size_t> &sizes, const std::vector<size_t> &num_vectors) {
	const size_t maxSize = sizes.back();
	const size_t maxVectors = num_vectors.back();

	std::vector<T> inputHostX(maxVectors * maxSize);
	std::vector<T> inputHostY(maxVectors * maxSize);
	std::vector<T> outputHost(maxVectors);
	std::random_device rd;
	std::default_random_engine re(rd());
	std::uniform_real_distribution<T> dist(-10, 10);

	for (size_t i = 0; i < maxVectors * maxSize; ++i) {
		inputHostX[i] = dist(re);
		inputHostY[i] = dist(re);
	}

	auto inputDeviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, inputHostX.cbegin(), inputHostX.cend());
	auto inputDeviceY = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank1, inputHostY.cbegin(), inputHostY.cend());
	auto outputDevice = context.MakeBuffer<T, ocl::Access::write>(ocl::MemoryBank::bank0, outputHost.begin(), outputHost.end());

	auto program = context.MakeProgram(kernelFile + ".xclbin");

	for (auto &num_vector : num_vectors) {
		SECTION(std::to_string(num_vector) + " vectors") {
			for (auto &size : sizes) {
				SECTION("Vectors of size " + std::to_string(size)) {
					auto kernel = program.MakeKernel("blas_dot_multiple", size, num_vector, inputDeviceX, inputDeviceY,
					                                 outputDevice);
					kernel.ExecuteTask();

					outputDevice.CopyToHost(outputHost.data());
					for (size_t i = 0; i < num_vector; ++i) {
						INFO("Vector " << i + 1);
						auto beginRefX = inputHostX.begin() + i * size;
						auto beginRefY = inputHostY.begin() + i * size;

						auto refX = std::vector<T>(beginRefX, beginRefX + size);
						auto refY = std::vector<T>(beginRefY, beginRefY + size);
						auto reference = reference_dot(refX, refY, size);
						REQUIRE(outputHost[i] == Approx(reference));
					}
				}
			}
		}
	}
}

TEST_CASE("Dot", "[Dot]") {
	std::vector<size_t> sizes;
	for (size_t i = 1; i <= 32; ++i) {
		sizes.push_back(i);
	}

	ocl::Context context;

	SECTION("Float") {
		test_dot<float>(context, "kernel_dot", sizes);
	}
}

TEST_CASE("Interleaved Dot", "[Dot]") {
	std::vector<size_t> sizes;
	std::vector<size_t> num_vectors;

	for (size_t i = 1; i <= 32; ++i) {
		sizes.push_back(i);
	}
	for (size_t i = 1; i <= 32; ++i) {
		num_vectors.push_back(i);
	}

	ocl::Context context;

	SECTION("Float") {
		test_interleaved_dot<float>(context, "kernel_dot_multiple", sizes, num_vectors);
	}
}
