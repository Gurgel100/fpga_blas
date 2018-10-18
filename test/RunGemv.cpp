//
// Created by stpascal on 14.06.18.
//

#include "hlslib/SDAccel.h"
#include <iostream>
#include <vector>
#include <random>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <cblas.h>

using namespace hlslib;
using namespace std;

template <typename T>
static void run_gemv(ocl::Context &context, ocl::Program &program, const vector<T> &A, const vector<T> &x, vector<T> &y, const size_t N, const size_t M) {
	auto inputDeviceA = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, A.cbegin(), A.cend());
	auto inputDeviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank1, x.cbegin(), x.cend());
	auto outputDevice = context.MakeBuffer<T, ocl::Access::write>(ocl::MemoryBank::bank2, y.begin(), y.end());

	auto kernel = program.MakeKernel("blas_gemv", inputDeviceA, inputDeviceX, outputDevice, N, M);
	kernel.ExecuteTask();

	outputDevice.CopyToHost(y.data());
}

TEST_CASE("kernel_gemv", "[GEMV]") {
	ocl::Context context;
	auto program = context.MakeProgram("kernel_gemv.xclbin");

	for (size_t N = 16; N <= 32; N += 16) {
		for (size_t M = 16; M <= 32; M += 16) {
			vector<float> A(N * M), x(N), y(M), ref_y(M);

			SECTION("N=" + to_string(N) + ", M=" + to_string(M)) {
				SECTION("All 1s") {
					fill(A.begin(), A.end(), 1);
					fill(x.begin(), x.end(), 1);

					run_gemv(context, program, A, x, y, N, M);

					cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < M; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}

				SECTION("Counted") {
					iota(A.begin(), A.end(), 0);
					iota(x.begin(), x.end(), 0);

					run_gemv(context, program, A, x, y, N, M);

					cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < M; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}

				SECTION("Random") {
					mt19937 g;
					uniform_real_distribution<float> dist;
					generate(A.begin(), A.end(), [&g, &dist](){return dist(g);});
					generate(x.begin(), x.end(), [&g, &dist](){return dist(g);});

					run_gemv(context, program, A, x, y, N, M);

					cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < M; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}
			}
		}
	}
}

template <typename T>
static void run_gemv_transposed(ocl::Context &context, ocl::Program &program, const vector<T> &A, const vector<T> &x, vector<T> &y, const size_t N, const size_t M) {
	auto inputDeviceA = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank0, A.cbegin(), A.cend());
	auto inputDeviceX = context.MakeBuffer<T, ocl::Access::read>(ocl::MemoryBank::bank1, x.cbegin(), x.cend());
	auto outputDevice = context.MakeBuffer<T, ocl::Access::readWrite>(ocl::MemoryBank::bank2, y.begin(), y.end());

	auto kernel = program.MakeKernel("blas_gemv_transposed", inputDeviceA, inputDeviceX, outputDevice, N, M);
	kernel.ExecuteTask();

	outputDevice.CopyToHost(y.data());
}

TEST_CASE("kernel_gemv_transposed", "[GEMV]") {
	ocl::Context context;
	auto program = context.MakeProgram("kernel_gemv_transposed.xclbin");

	for (size_t N = 16; N <= 32; N += 16) {
		for (size_t M = 16; M <= 32; M += 16) {
			vector<float> A(N * M), x(M), y(N), ref_y(N);

			SECTION("N=" + to_string(N) + ", M=" + to_string(M)) {
				SECTION("All 1s") {
					fill(A.begin(), A.end(), 1);
					fill(x.begin(), x.end(), 1);

					run_gemv_transposed(context, program, A, x, y, N, M);

					cblas_sgemv(CblasRowMajor, CblasTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < N; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}

				SECTION("Counted") {
					iota(A.begin(), A.end(), 0);
					iota(x.begin(), x.end(), 0);

					run_gemv_transposed(context, program, A, x, y, N, M);

					cblas_sgemv(CblasRowMajor, CblasTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < N; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}

				SECTION("Random") {
					mt19937 g;
					uniform_real_distribution<float> dist;
					generate(A.begin(), A.end(), [&g, &dist](){return dist(g);});
					generate(x.begin(), x.end(), [&g, &dist](){return dist(g);});

					run_gemv_transposed(context, program, A, x, y, N, M);

					cblas_sgemv(CblasRowMajor, CblasTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < N; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}
			}
		}
	}
}
