//
// Created by stpascal on 01.07.18.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <fstream>
#include <unistd.h>
#include <wait.h>
#include <utility>
#include <sys/mman.h>
#include "hlslib/SDAccel.h"

using namespace std;
using namespace hlslib;

using Type_t = float;

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

private:
	T *mem;
};

static pair<double, double> execute(size_t N, size_t M, const vector<Type_t> &inputHostA, const vector<Type_t> &inputHostX, vector<Type_t> &outputHost) {
	SharedContent<double> shared_time(NAN), shared_variance(0);

	pid_t child = fork();
	if (child < 0) {
		throw runtime_error("Error while forking: " + to_string(errno));
	} else if (child == 0) {
		ocl::Context context;
		auto program = context.MakeProgram("kernel_gemv.xclbin");

		try {
			auto inputDeviceA = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank0,
			                                                                  inputHostA.cbegin(),
			                                                                  inputHostA.cend());
			auto inputDeviceX = context.MakeBuffer<Type_t, ocl::Access::read>(ocl::MemoryBank::bank1,
			                                                                  inputHostX.cbegin(),
			                                                                  inputHostX.cend());
			auto outputDevice = context.MakeBuffer<Type_t, ocl::Access::write>(ocl::MemoryBank::bank2,
			                                                                   outputHost.begin(),
			                                                                   outputHost.end());
			auto kernel = program.MakeKernel("blas_gemv", inputDeviceA, inputDeviceX, outputDevice, N, M);

			double time = 0.0;
			double mintime = INFINITY;
			double maxtime = -INFINITY;
			array<double, 10> times{};
			for (double &i : times) {
				auto t = kernel.ExecuteTask();
				time += t.first;
				i = t.first;
				mintime = min(mintime, t.first);
				maxtime = max(maxtime, t.first);
			}
			shared_time = time / times.size();

			for (const double t : times) {
				double tmp = t - shared_time;
				shared_variance += tmp * tmp;
			}

			exit(EXIT_SUCCESS);
		} catch (...) {
			exit(EXIT_FAILURE);
		}
	} else {
		waitpid(child, NULL, 0);
		if (shared_time == NAN) {
			throw runtime_error("Something went wrong while executing kernel");
		}
		return {shared_time, shared_variance};
	}
}

int main(int argc, const char *argv[]) {

	if (argc < 2) {
		cout << "Usage: " << argv[0] << " path/to/out.csv" << endl;
		return 0;
	}

	fstream out(argv[1], ios_base::out | ios_base::trunc);

	out << "N,M,time,variance" << endl;

	for (size_t N = 16; N < 1 << 30; N *= 2) {
		for (size_t M = 16; M < 1 << 30; M *= 2) {
			cout << "N = " << N << ", M = " << M << ": ";
			try {
				vector<Type_t> inputHostA(N * M, 1), inputHostX(N, 1), outputHost(M);

				auto stats = execute(N, M, inputHostA, inputHostX, outputHost);

				out << N << "," << M << "," << stats.first << "," << stats.second << endl;
				cout << stats.first << "s (" << stats.second << ")" << endl;
			} catch (exception &e) {
				cout << "skipped (reason: '" << e.what() << "')" << endl;
			}
		}
	}

	return 0;
}
