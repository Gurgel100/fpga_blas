//
// Created by stpascal on 22.01.19.
//

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <fblas.h>
#include <cblas.h>
#include <vector>
#include <random>

using namespace std;

template <class T>
static void fill_random(vector<T> &d) {
	static mt19937 mt;
	uniform_real_distribution<T> dist(0.0, 10.0);

	for (auto &item : d) {
		item = dist(mt);
	}
}

template <class T>
static void fill_ones(vector<T> &d) {
	std::fill(d.begin(), d.end(), 1);
}

template <class T>
static void check(const vector<T> &a, const vector<T> &b) {
	assert(a.size() == b.size());
	for (size_t i = 0; i < a.size(); ++i) {
		CHECK(a[i] == Approx(b[i]));
	}
}

template <class T>
static void test_nrm2(T(*const func)(int, const T*, int), T(*const checker)(int, const T*, int)) {
	for (int N = 1; N <= 1024; N += 1) {
		SECTION("N = " + to_string(N) + ", incX = 1") {
			vector<T> X(N);
			fill_ones(X);

			auto res = func(N, X.data(), 1);
			auto check_res = checker(N, X.data(), 1);

			REQUIRE(res == Approx(check_res));
		}
	}
}

TEST_CASE("snrm2") {
	test_nrm2(fblas_snrm2, cblas_snrm2);
}

TEST_CASE("dnrm2") {
	test_nrm2(fblas_dnrm2, cblas_dnrm2);
}

template <class T>
static void test_dot(T(*const func)(int, const T*, int, const T*, int), T(*const checker)(int, const T*, int, const T*, int)) {
	for (int N = 1; N <= (1 << 30); N <<= 1) {
		SECTION("N = " + to_string(N) + ", incX = 1") {
			vector<T> X(N), Y(N);
			fill_random(X);
			fill_random(Y);

			auto res = func(N, X.data(), 1, Y.data(), 1);
			auto check_res = checker(N, X.data(), 1, Y.data(), 1);

			REQUIRE(res == Approx(check_res));
		}
	}
}

TEST_CASE("sdot") {
	test_dot(fblas_sdot, cblas_sdot);
}

TEST_CASE("ddot") {
	test_dot(fblas_ddot, cblas_ddot);
}

template <class T>
static void test_gemv(
		void(*const func)(char, int, int, T, const T*, int, const T*, int, T, T*, int),
		void(*const checker)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, T, const T*, int, const T*, int, T, T*, int)) {
	/*for (int N = 256; N <= (1 << 30); N <<= 1)*/ {
		/*for (int M = 256; M <= (1 << 30); M <<= 1)*/ {
			int N = 256, M = 256;
			SECTION("N = " + to_string(N) + "M = " + to_string(M) + ", incX = 1") {
				vector<T> A(N * M), X(N), Y(M);
				fill_random(A);
				fill_random(X);
				fill_random(Y);
				vector<T> Y2(Y);

				func('N', M, N, 1, A.data(), N, X.data(), 1, 1, Y.data(), 1);
				checker(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, 1, A.data(), N, X.data(), 1, 1, Y2.data(), 1);

				for (int i = 0; i < M; ++i) {
					INFO("index = " + to_string(i));
					REQUIRE(Y[i] == Approx(Y2[i]));
				}
			}
		}
	}
}

TEST_CASE("sgemv") {
	test_gemv(fblas_sgemv, cblas_sgemv);
}

TEST_CASE("dgemv") {
	test_gemv(fblas_dgemv, cblas_dgemv);
}
