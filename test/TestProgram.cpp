#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cassert>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "DotProduct.h"

using namespace std;

template <class T>
static void fill(vector<T> &d) {
	static mt19937 mt;
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

TEST_CASE("blas_dot") {
	const size_t maxSize = 32;
	vector<Data_t> x(maxSize), y(maxSize), out(1);
	fill(x);
	fill(y);

	for (size_t n = 1; n < maxSize; ++n) {
		SECTION("Vectors of size " + to_string(n)) {
			Data_t ref = reference_dot(n, x, y);
			blas_dot(n, x.data(), 1, y.data(), 1, out.data());
			REQUIRE(out[0] == Approx(ref));
		}
	}
}

TEST_CASE("blas_dot_multiple") {
	const size_t maxSize = 32;
	const size_t maxVectors = 32;

	for (size_t i = 1; i < maxVectors; ++i) {
		SECTION(to_string(i) + " vectors") {
			vector<Data_t> x(maxSize * i), y(maxSize * i), out(i);
			fill(x);
			fill(y);
			for (size_t n = 1; n < maxSize; ++n) {
				SECTION("Vectors of size " + to_string(n)) {
					blas_dot_multiple(n, i, x.data(), y.data(), out.data());
					for (size_t j = 0; j < i; ++j) {
						auto tmpx = vector<Data_t>(x.begin() + j * n, x.begin() + j * n + n);
						auto tmpy = vector<Data_t>(y.begin() + j * n, y.begin() + j * n + n);
						Data_t reference = reference_dot(n, tmpx, tmpy);
						INFO("Vector " << j + 1);
						REQUIRE(out[j] == Approx(reference));
					}
				}
			}
		}
	}
}
