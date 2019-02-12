#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cblas.h>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using namespace std;

static mt19937 mt;

template <class T>
static void fill(vector<T> &d) {
	uniform_real_distribution<T> dist(0.0, 1.0);

	for (auto &item : d) {
		item = dist(mt);
	}
}

template <class T>
static void check_diff(const size_t size, const vector<T> &a, const vector<T> &b) {
	for (size_t i = 0; i < size; ++i) {
		INFO("Index " << i);
		CHECK(a[i] == Approx(b[i]));
	}
}

static void calc_cblas(size_t N, size_t M, const std::vector<float> &A, const std::vector<float> &B, const std::vector<float> &x, std::vector<float> &res) {
	std::vector<float> Ax(M), Bx(M);

	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, 1, A.data(), N, x.data(), 1, 0, Ax.data(), 1);
	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, 1, B.data(), N, x.data(), 1, 0, Bx.data(), 1);

	std::vector<float> ABx(Ax);
	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, 3.1419, B.data(), N, x.data(), 1, 1, ABx.data(), 1);

	auto dot_res = cblas_sdot(M, Ax.data(), 1, Bx.data(), 1);
	auto norm_res = cblas_snrm2(M, ABx.data(), 1);

	res[0] = dot_res / norm_res;

	//TMP
	//copy(Ax.begin(), Ax.end(), res.begin());
	copy(ABx.begin(), ABx.end(), res.begin());
	/*res[0] = dot_res;
	res[1] = norm_res;
	res[2] = dot_res / norm_res;
	res[3] = norm_res / dot_res;*/
}

extern "C" void benchmark(size_t N, size_t M, const float memoryIn_A[], const float memoryIn_B[], const float memoryIn_x[], float memoryOut[]);

/*TEST_CASE("Benchmark") {
	const size_t N = 1024, M = 1024;
	vector<float> A(N * M), B(N * M), x(N), res(M), res2(M);
	fill(A);
	fill(B);
	fill(x);

	for (size_t i = 1; i <= N; i <<= 1) {
		for (size_t j = 1; j <= M; j <<= 1) {
			SECTION("n = " + to_string(i) + ", m = " + to_string(j)) {
				benchmark(i, j, A.data(), B.data(), x.data(), res.data());
				calc_cblas(i, j, A, B, x, res2);

				check_diff(j, res, res2);
			}
		}
	}
}*/

static void calc_cblas2(size_t N, size_t M, const std::vector<float> &A, const std::vector<float> &x, const std::vector<float> &y, std::vector<float> &res) {
	std::vector<float> Ax(N);

	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasTrans, M, N, 1, A.data(), N, x.data(), 1, 0, Ax.data(), 1);

	auto dot_res = cblas_sdot(N, Ax.data(), 1, y.data(), 1);

	res[0] = dot_res;
}

extern "C" void benchmark2(size_t N, size_t M, const float memoryIn_A[], const float memoryIn_x[], const float memoryIn_y[], float memoryOut[]);

TEST_CASE("Benchmark2") {
	const size_t N = 8196, M = 8196;
	vector<float> A(N * M), x(M), y(N), res(N), res2(N);
	fill(A);
	fill(x);
	fill(y);

	for (size_t i = 256; i <= N; i <<= 1) {
		for (size_t j = 64; j <= M; j <<= 1) {
			SECTION("n = " + to_string(i) + ", m = " + to_string(j)) {
				benchmark2(i, j, A.data(), x.data(), y.data(), res.data());
				calc_cblas2(i, j, A, x, y, res2);

				check_diff(i, res, res2);
			}
		}
	}
}

