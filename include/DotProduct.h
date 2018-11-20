#pragma once

#include "Config.h"
#include <hlslib/DataPack.h>

#ifndef DOT_WIDTH
#define DOT_WIDTH   16
#endif

extern "C" {
	void blas_dot(const size_t N, hlslib::DataPack<Data_t, DOT_WIDTH> const memoryIn_X[], hlslib::DataPack<Data_t, DOT_WIDTH> const memoryIn_Y[], Data_t memoryOut[]);
	void blas_dot_multiple(const size_t N, const size_t n_prod, hlslib::DataPack<Data_t, DOT_WIDTH> const memoryIn_X[], hlslib::DataPack<Data_t, DOT_WIDTH> const memoryIn_Y[], Data_t memoryOut[]);
}
