#pragma once

#include "Config.h"

extern "C" {
	void blas_dot(const size_t N, Data_t const memoryIn_X[], const int incX, Data_t const memoryIn_Y[], const int incY, Data_t memoryOut[]);
	void blas_dot_multiple(const size_t N, const size_t n_prod, Data_t const memoryIn_X[], Data_t const memoryIn_Y[], Data_t memoryOut[]);
}
