//
// Created by stpascal on 10.08.18.
//

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <cblas.h>
#include <Scalar.h>
#include <vector>
#include <random>

template <class T>
static bool compare(const T lhs[], const T rhs[], const size_t size) {
	for (size_t i = 0; i < size; ++i) {
		if (Approx(lhs[i]) != rhs[i])
			return false;
	}
	return true;
}

static bool operator==(const std::vector<float> &lhs, const std::vector<float> &rhs) {
	return compare(lhs.data(), rhs.data(), lhs.size());
};

static bool operator==(const std::vector<double> &lhs, const std::vector<double> &rhs) {
	return compare(lhs.data(), rhs.data(), lhs.size());
};

template <class T, class U>
static void fill(T &container, const U &value) {
	std::fill(container.begin(), container.end(), value);
}

template <class T>
static void exec_scalar(size_t N, const std::vector<T> &x, size_t incX, T alpha, std::vector<T> &res, size_t incRes) {
	FBLAS::Stream<T> inX, out;
	FBLAS::Scalar<T> scalar(N, inX, out);

	HLSLIB_DATAFLOW_INIT()
	scalar.getReaderX().template readFromMemory<true>(x.data(), incX);
	scalar.template calc<true>(alpha);
	scalar.getWriterX().template writeToMemory<true>(res.data(), incRes);
	HLSLIB_DATAFLOW_FINALIZE()
}

template <class T, class Generator>
static void test_scalar(Generator &g, size_t max_size, size_t num_samples = 10) {
	assert(max_size >= 1);
	std::normal_distribution<T> normal_distribution(0, 10);
	std::uniform_int_distribution<size_t> size_dist(0, max_size);
	std::uniform_int_distribution<size_t> unif(1, max_size);
	std::vector<T> x(max_size, 1), res(max_size), res_ref(max_size);

	SECTION("x = 1, Random factor") {
		for (size_t sample = 0; sample < num_samples; ++sample) {
			size_t size = size_dist(g);
			SECTION("N=" + std::to_string(size)) {
				auto factor = normal_distribution(g);
				for (size_t i = 0; i < size; ++i) {
					res_ref[i] = x[i] * factor;
				}
				exec_scalar(size, x, 1, factor, res, 1);
				REQUIRE(res == res_ref);
			}
		}
	}
	SECTION("Random x, Random factor") {
		for (size_t sample = 0; sample < num_samples; ++sample) {
			size_t size = size_dist(g);
			std::generate(x.begin(), x.end(), [&g, &normal_distribution](){return normal_distribution(g);});
			SECTION("N=" + std::to_string(size)) {
				auto factor = normal_distribution(g);
				for (size_t i = 0; i < size; ++i) {
					res_ref[i] = x[i] * factor;
				}
				exec_scalar(size, x, 1, factor, res, 1);
				REQUIRE(res == res_ref);
			}
		}
	}
}

TEST_CASE("Scalar") {

	std::mt19937_64 eng;

	SECTION("float") {
		test_scalar<float>(eng, 64);
	}

	SECTION("double") {
		test_scalar<double>(eng, 64);
	}
}
