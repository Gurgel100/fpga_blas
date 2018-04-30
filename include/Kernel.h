#pragma once

#include "Config.h"

extern "C" void blas_xdot(Data_t const memoryIn_X[], Data_t const memoryIn_Y[], Data_t memoryOut[], const size_t N = 4);
