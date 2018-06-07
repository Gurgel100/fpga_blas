#pragma once

#include "Config.h"

extern "C" void blas_dot(Data_t const memoryIn_X[], Data_t const memoryIn_Y[], Data_t memoryOut[], const size_t N);
extern "C" void blas_dot_multiple(Data_t const memoryIn_X[], Data_t const memoryIn_Y[], Data_t memoryOut[], const size_t N, const size_t n_prod);
