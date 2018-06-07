#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cassert>
#include "Kernel.h"

using namespace std;

template <class T>
static void printResult(T expected, T value) {
	if (value != expected) {
		cout << "Mismatch: " << value << " (expected " << expected << ')' << endl;
	} else {
		cout << "Got correct result" << endl;
	}
}

template <class T>
static T reference_dot(const vector<T> &x, const vector<T> &y)
{
	assert(x.size() == y.size());
	T res = 0;
	for (size_t i = 0; i < x.size(); ++i) {
		res += x[i] * y[i];
	}
	return res;
}

int main(int argc, char const *argv[]) {
	uniform_real_distribution<Data_t> dist(0.0, 10.0);
	mt19937 mt;
	for (size_t n = 1; n < 16; ++n) {
		cout << "Testing for n = " << n << ": ";
		vector<Data_t> x(n);
		vector<Data_t> y(n);
		for (size_t i = 0; i < n; ++i) {
			x[i] = dist(mt);
			y[i] = dist(mt);
		}

		Data_t ref = reference_dot(x, y);

		vector<Data_t> out(1, 0);
		blas_dot(x.data(), y.data(), out.data(), n);
		printResult(ref, out[0]);
	}
	return 0;
}
