#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cassert>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "DotProduct.h"
#include "GEMV.h"
#include "cblas.h"

using namespace std;

static mt19937 mt;

template <class T>
static void fill(vector<T> &d) {
	uniform_real_distribution<Data_t> dist(0.0, 10.0);

	for (auto &item : d) {
		item = dist(mt);
	}
}

template <class T>
static T reference_dot(const size_t N, const vector<T> &x, const vector<T> &y)
{
	T res = 0;
	for (size_t i = 0; i < N; ++i) {
		res += x[i] * y[i];
	}
	return res;
}

template <>
double reference_dot<double>(const size_t N, const std::vector<double> &x, const std::vector<double> &y)
{
	return cblas_ddot(static_cast<int>(N), x.data(), 1, y.data(), 1);
}

template <>
float reference_dot<float>(const size_t N, const std::vector<float> &x, const std::vector<float> &y)
{
	return cblas_sdot(static_cast<int>(N), x.data(), 1, y.data(), 1);
}

template <class T, int width>
static T reference_dot(size_t N, const std::vector<hlslib::DataPack<T, width>> &x, const std::vector<hlslib::DataPack<T, width>> &y) {
	std::vector<T> tmpX(N * width), tmpY(N * width);
	for (size_t i = 0; i < N / width; ++i) {
		for (int j = 0; j < width; ++j) {
			tmpX[i * width + j] = x[i][j];
			tmpY[i * width + j] = y[i][j];
		}
	}
	return reference_dot(N, tmpX, tmpY);
}

/*template <class T>
class LogarithmicRangeGenerator : public Catch::IGenerator<T>
{
public:
	LogarithmicRangeGenerator(T from, T to, T factor = 1) : from(from), to(to), factor(factor) {}

	virtual T getValue(std::size_t index) const {
		return from + static_cast<T>(pow(factor, index));
	}

	virtual size_t size() const {
		return static_cast<size_t>(log(1 + to - from) / log(factor));
	}

private:
	T from, to, factor;
};

extern "C" void fblas_saxpy(const int N, const float A, const hlslib::DataPack<float, 1> memoryIn_X[], int incX, hlslib::DataPack<float, 1> memory_Y[], int incY);
TEST_CASE("saxpy") {
	Catch::CompositeGenerator<int> generator;
	generator.add(new LogarithmicRangeGenerator<int>(1, 32));
	const int N = GENERATE(generator);
	uniform_real_distribution<float> dist(-1000, 1000);
	vector<hlslib::DataPack<float, 1>> x(N), y(N);
	vector<float> blas_x(N), blas_y(N);
	const float a = dist(mt);

	fill(x);
	fill(y);

	transform(x.begin(), x.end(), blas_x.begin(), [](const hlslib::DataPack<float, 1> &val){return val[0];});
	transform(y.begin(), y.end(), blas_y.begin(), [](const hlslib::DataPack<float, 1> &val){return val[0];});

	cblas_saxpy(N, a, blas_x.data(), 1, blas_y.data(), 1);
	fblas_saxpy(N, a, x.data(), 1, y.data(), 1);

	for (int i = 0; i < N; ++i) {
		INFO("i:" + to_string(i));
		REQUIRE(y[i][0] == Approx(blas_y[i]));
	}
}*/

TEST_CASE("blas_dot") {
	const size_t maxSize = 256;
	vector<hlslib::DataPack<Data_t, DOT_WIDTH>> x(maxSize), y(maxSize);
	Data_t out;
	fill(x);
	fill(y);

	for (size_t n = 1; n <= maxSize; ++n) {
		SECTION("Vectors of size " + to_string(n)) {
			Data_t ref = reference_dot(n * DOT_WIDTH, x, y);
			blas_dot(n * DOT_WIDTH, x.data(), y.data(), &out);
			REQUIRE(out == Approx(ref));
		}
	}
}

TEST_CASE("blas_dot_multiple") {
	const size_t maxSize = 64;
	const size_t maxVectors = 32;

	for (size_t i = 1; i < maxVectors; ++i) {
		SECTION(to_string(i) + " vectors") {
			vector<hlslib::DataPack<Data_t, DOT_WIDTH>> x(maxSize * i), y(maxSize * i);
			vector<Data_t> out(i);
			fill(x);
			fill(y);
			for (size_t n = 1; n < maxSize; ++n) {
				SECTION("Vectors of size " + to_string(n)) {
					blas_dot_multiple(n, i, x.data(), y.data(), out.data());
					for (size_t j = 0; j < i; ++j) {
						auto tmpx = vector<hlslib::DataPack<Data_t, DOT_WIDTH>>(x.begin() + j * n, x.begin() + j * n + n);
						auto tmpy = vector<hlslib::DataPack<Data_t, DOT_WIDTH>>(y.begin() + j * n, y.begin() + j * n + n);
						Data_t reference = reference_dot(n * DOT_WIDTH, tmpx, tmpy);
						INFO("Vector " << j + 1);
						REQUIRE(out[j] == Approx(reference));
					}
				}
			}
		}
	}
}

TEST_CASE("blas_gemv") {

	for (size_t N = 16; N <= 256; N += 16) {
		for (size_t M = 16; M <= 256; M += 16) {
			vector<Data_t> A(N * M);
			vector<Data_t> x(N);
			vector<Data_t> y(M);
			vector<Data_t> ref_y(M);

			SECTION("N=" + to_string(N) + ", M=" + to_string(M)) {
				SECTION("All 1s") {
					fill(A.begin(), A.end(), 1);
					fill(x.begin(), x.end(), 1);

					blas_gemv(reinterpret_cast<MatrixVectorMultiplication_t::Col_t *>(A.data()),
					          reinterpret_cast<MatrixVectorMultiplication_t::Col_t *>(x.data()),
					          y.data(), N, M);

					cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < M; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}

				SECTION("Counted") {
					iota(A.begin(), A.end(), 0);
					iota(x.begin(), x.end(), 0);

					blas_gemv(reinterpret_cast<MatrixVectorMultiplication_t::Col_t *>(A.data()),
					          reinterpret_cast<MatrixVectorMultiplication_t::Col_t *>(x.data()),
					          y.data(), N, M);

					cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < M; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}

				SECTION("Random") {
					mt19937 g;
					uniform_real_distribution<Data_t> dist;
					generate(A.begin(), A.end(), [&g, &dist](){return dist(g);});
					generate(x.begin(), x.end(), [&g, &dist](){return dist(g);});

					blas_gemv(reinterpret_cast<MatrixVectorMultiplication_t::Col_t *>(A.data()),
					          reinterpret_cast<MatrixVectorMultiplication_t::Col_t *>(x.data()),
					          y.data(), N, M);

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

TEST_CASE("blas_gemv_transposed") {

	for (size_t N = 16; N <= 256; N += 16) {
		for (size_t M = 16; M <= 256; M += 16) {
			vector<Data_t> A(N * M);
			vector<Data_t> x(M);
			vector<Data_t> y(N);
			vector<Data_t> ref_y(N);

			SECTION("N=" + to_string(N) + ", M=" + to_string(M)) {
				SECTION("All 1s") {
					fill(A.begin(), A.end(), 1);
					fill(x.begin(), x.end(), 1);

					blas_gemv_transposed(reinterpret_cast<MatrixVectorMultiplicationTransposed_t::Col_t *>(A.data()), x.data(),
					          reinterpret_cast<MatrixVectorMultiplicationTransposed_t::Col_t *>(y.data()),
					          N, M);

					cblas_sgemv(CblasRowMajor, CblasTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < N; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}

				SECTION("Counted") {
					iota(A.begin(), A.end(), 0);
					iota(x.begin(), x.end(), 0);

					blas_gemv_transposed(reinterpret_cast<MatrixVectorMultiplicationTransposed_t::Col_t *>(A.data()), x.data(),
					                     reinterpret_cast<MatrixVectorMultiplicationTransposed_t::Col_t *>(y.data()),
					                     N, M);

					cblas_sgemv(CblasRowMajor, CblasTrans, M, N, 1, A.data(), N, x.data(), 1, 0, ref_y.data(), 1);

					for (size_t i = 0; i < N; ++i) {
						INFO("Y[" << i << "]");
						CHECK(y[i] == Approx(ref_y[i]));
					}
				}

				SECTION("Random") {
					mt19937 g;
					uniform_real_distribution<Data_t> dist;
					generate(A.begin(), A.end(), [&g, &dist](){return dist(g);});
					generate(x.begin(), x.end(), [&g, &dist](){return dist(g);});

					blas_gemv_transposed(reinterpret_cast<MatrixVectorMultiplicationTransposed_t::Col_t *>(A.data()), x.data(),
					                     reinterpret_cast<MatrixVectorMultiplicationTransposed_t::Col_t *>(y.data()),
					                     N, M);

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
