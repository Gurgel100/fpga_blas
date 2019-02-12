//
// Created by stpascal on 18.12.18.
//

#include <hlslib/SDAccel.h>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cblas.h>
#include <fblas.h>
#include <unistd.h>
#include <wait.h>
#include <utility>
#include <sys/mman.h>

using Data_t = float;

// (A + 3.1419 * B)x * (Ax . Bx)

using namespace std;

template <class Allocator>
static void calc_cblas(size_t N, size_t M, const std::vector<float, Allocator> &A, const std::vector<float, Allocator> &x, const std::vector<float, Allocator> &y, std::vector<float, Allocator> &res) {
	std::vector<float> Ax(N);

	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, 1, A.data(), N, x.data(), 1, 0, Ax.data(), 1);

	auto dot_res = cblas_sdot(M, Ax.data(), 1, y.data(), 1);

	res[0] = dot_res;
}

template <class Allocator>
static void calc_fblas(size_t N, size_t M, const std::vector<float, Allocator> &A, const std::vector<float, Allocator> &x, const std::vector<float, Allocator> &y, std::vector<float, Allocator> &res) {
	std::vector<float, Allocator> Ax(M);

	fblas_sgemv('N', M, N, 1, A.data(), N, x.data(), 1, 0, Ax.data(), 1);

	auto dot_res = fblas_sdot(M, Ax.data(), 1, y.data(), 1);

	res[0] = dot_res;
}

int main(int argc, const char *argv[]) {
	if (argc < 2) {
		cout << "Usage: " << argv[0] << " fblas|kernel [N] [M]" << endl;
		return 1;
	}
	size_t N = argc > 2 ? strtoul(argv[2], NULL, 0) : 256;
	size_t M = argc > 3 ? strtoul(argv[3], NULL, 0) : 256;

	std::vector<Data_t, hlslib::ocl::AlignedAllocator<Data_t, 4096>> A(N * M), x(M), y(N), res(1);

	std::mt19937 gen;
	std::uniform_real_distribution<float> dist(-10, 10);
	auto generator = [&gen, &dist](){
		return dist(gen);
	};
	std::generate(A.begin(), A.end(), generator);
	std::generate(x.begin(), x.end(), generator);
	std::generate(y.begin(), y.end(), generator);


	std::pair<double , double> time_kernel;
	double time_fblas;
	Data_t result_kernel, result_fblas;

	if (strcmp(argv[1], "kernel") == 0) {
		hlslib::ocl::Context context;

		auto BufferA = context.MakeBuffer<Data_t, hlslib::ocl::Access::read>(hlslib::ocl::MemoryBank::bank0, A.begin(),
		                                                                     A.end());
		auto BufferX = context.MakeBuffer<Data_t, hlslib::ocl::Access::read>(hlslib::ocl::MemoryBank::bank1, x.begin(),
		                                                                     x.end());
		auto BufferY = context.MakeBuffer<Data_t, hlslib::ocl::Access::read>(hlslib::ocl::MemoryBank::bank2, y.begin(),
		                                                                     y.end());
		auto BufferRes = context.MakeBuffer<Data_t, hlslib::ocl::Access::write>(hlslib::ocl::MemoryBank::bank3,
		                                                                        res.begin(), res.end());

		auto program = context.MakeProgram("kernel_benchmark2_float.xclbin");
		auto kernel = program.MakeKernel("benchmark2", N, M, BufferA, BufferX, BufferY, BufferRes);
		cout << "Execute kernel" << endl;
		time_kernel = kernel.ExecuteTask();
		BufferRes.CopyToHost(res.data());
		result_kernel = res[0];
	} else {
		prepareFPGAForSingle();
		cout << "Executing fblas" << endl;
		const auto start = std::chrono::high_resolution_clock::now();
		calc_fblas(N, M, A, x, y, res);
		const auto end = std::chrono::high_resolution_clock::now();
		time_fblas = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		result_fblas = res[0];
	}

	calc_cblas(N, M, A, x, y, res);

	cout << "Time kernel: " << time_kernel.first << "s(" << time_kernel.second << "s)" << endl;
	cout << "Time fblas:  " << time_fblas << "s" << endl;

	cout << "Result cblas:  " << res[0] << endl;
	cout << "Result kernel: " << result_kernel << endl;
	cout << "Result fblas:  " << result_fblas << endl;

	return 0;
}