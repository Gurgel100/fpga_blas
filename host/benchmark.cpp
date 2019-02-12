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

template <class T>
class SharedContent {
public:
	SharedContent() {
		mem = static_cast<T*>(mmap(NULL, sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0));
		if (mem == MAP_FAILED) {
			throw runtime_error("Error while creating shared memory: " + to_string(errno));
		}
	}

	explicit SharedContent(const T &init) : SharedContent() {
		*this = init;
	}

	~SharedContent() {
		munmap(mem, sizeof(T));
	}

	SharedContent<T> &operator=(const T &val) {
		*mem = val;
		return *this;
	}

	SharedContent<T> &operator+=(const T &val) {
		*mem += val;
		return *this;
	}

	operator T() {
		return *mem;
	}

	T operator ()() {
		return *mem;
	}

private:
	T *mem;
};

template <class Allocator>
static void calc_cblas(size_t N, size_t M, const std::vector<float, Allocator> &A, const std::vector<float, Allocator> &B, const std::vector<float, Allocator> &x, std::vector<float, Allocator> &res) {
	std::vector<float, Allocator> Ax(M), Bx(M);

	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, 1, A.data(), N, x.data(), 1, 0, Ax.data(), 1);
	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, 1, B.data(), N, x.data(), 1, 0, Bx.data(), 1);

	std::vector<float, Allocator> ABx(Ax);
	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, 3.1419, B.data(), N, x.data(), 1, 1, ABx.data(), 1);

	auto dot_res = cblas_sdot(M, Ax.data(), 1, Bx.data(), 1);
	auto norm_res = cblas_snrm2(M, ABx.data(), 1);

	res[0] = dot_res / norm_res;
}

template <class Allocator>
static void calc_fblas(size_t N, size_t M, const std::vector<float, Allocator> &A, const std::vector<float, Allocator> &B, const std::vector<float, Allocator> &x, std::vector<float, Allocator> &res) {
	std::vector<float, Allocator> Ax(M), Bx(M);

	fblas_sgemv('N', M, N, 1, A.data(), N, x.data(), 1, 0, Ax.data(), 1);
	fblas_sgemv('N', M, N, 1, B.data(), N, x.data(), 1, 0, Bx.data(), 1);

	std::vector<float, Allocator> ABx(Ax);
	fblas_sgemv('N', M, N, 3.1419, B.data(), N, x.data(), 1, 1, ABx.data(), 1);

	auto dot_res = fblas_sdot(M, Ax.data(), 1, Bx.data(), 1);
	auto norm_res = fblas_snrm2(M, ABx.data(), 1);

	res[0] = dot_res / norm_res;
}

int main(int argc, const char *argv[]) {
	size_t N = argc > 1 ? strtoul(argv[1], NULL, 0) : 16;
	size_t M = argc > 2 ? strtoul(argv[2], NULL, 0) : 16;
	const string kernel_binary = argc > 3 ? argv[3] : "kernel_benchmark.xclbin";
	std::vector<Data_t, hlslib::ocl::AlignedAllocator<Data_t, 4096>> A(N * M), B(N * M), x(N), res(1);

	std::mt19937 gen;
	std::uniform_real_distribution<float> dist(-10, 10);
	auto generator = [&gen, &dist](){
		return dist(gen);
	};
	std::generate(A.begin(), A.end(), generator);
	std::generate(B.begin(), B.end(), generator);
	std::generate(x.begin(), x.end(), generator);
	SharedContent<std::pair<double , double>> time_kernel;
	SharedContent<double> time_fblas;
	SharedContent<Data_t> result_kernel, result_fblas;
	bool failed = false;

	for (int i = 0; failed && i < 5; ++i) {
		pid_t child = fork();
		if (child < 0) {
			throw runtime_error("Error while forking: " + to_string(errno));
		} else if (child == 0) {
			auto &context = hlslib::ocl::GlobalContext();

			auto BufferA = context.MakeBuffer<Data_t, hlslib::ocl::Access::read>(hlslib::ocl::MemoryBank::bank0,
			                                                                     A.begin(),
			                                                                     A.end());
			auto BufferB = context.MakeBuffer<Data_t, hlslib::ocl::Access::read>(hlslib::ocl::MemoryBank::bank1,
			                                                                     B.begin(),
			                                                                     B.end());
			auto BufferX = context.MakeBuffer<Data_t, hlslib::ocl::Access::read>(hlslib::ocl::MemoryBank::bank2,
			                                                                     x.begin(),
			                                                                     x.end());
			auto BufferRes = context.MakeBuffer<Data_t, hlslib::ocl::Access::write>(hlslib::ocl::MemoryBank::bank3,
			                                                                        res.begin(), res.end());

			auto program = context.MakeProgram("kernel_benchmark.xclbin");
			auto kernel = program.MakeKernel("benchmark", N, M, BufferA, BufferB, BufferX, BufferRes);
			cout << "Execute kernel" << endl;
			time_kernel = kernel.ExecuteTask();
			BufferRes.CopyToHost(res.data());
			result_kernel = res[0];
		} else {
			int status;
			waitpid(child, &status, 0);
			failed = status != 0;
		}
	}
	if (failed) {
		throw runtime_error("Failed to execute kernel");
	}

	failed = true;
	for (int i = 0; failed && i < 5; ++i) {
		pid_t child = fork();
		if (child < 0) {
			throw runtime_error("Error while forking: " + to_string(errno));
		} else if (child == 0) {
			prepareFPGAForSingle();
			const auto start = std::chrono::high_resolution_clock::now();
			calc_fblas(N, M, A, B, x, res);
			const auto end = std::chrono::high_resolution_clock::now();
			time_fblas = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
			result_fblas = res[0];
		} else {
			int status;
			waitpid(child, &status, 0);
			failed = status != 0;
		}
	}
	if (failed) {
		throw runtime_error("Failed to execute fblas");
	}

	calc_cblas(N, M, A, B, x, res);

	cout << "Time kernel: " << time_kernel().first << "s(" << time_kernel().second << "s)" << endl;
	cout << "Time fblas:  " << time_fblas << "s" << endl;

	cout << "Result cblas:  " << res[0] << endl;
	cout << "Result kernel: " << result_kernel << endl;
	cout << "Result fblas:  " << result_fblas << endl;

	return 0;
}