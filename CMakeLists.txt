# Author:  Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
# Created: January 2018

cmake_minimum_required(VERSION 2.8)
project(blas_hls C CXX)

# Target options
set(BLAS_PART_NAME "xcku115-flvb2104-2-e" CACHE STRING "Part name for HLS.")
set(BLAS_DSA_NAME "xilinx:xil-accel-rd-ku115:4ddr-xpr:4.0" CACHE STRING
    "DSA string for xocc.")
set(BLAS_TARGET_CLOCK 300 CACHE STRING "Target clock for kernel (<=0 uses DSA default).")
set(BLAS_TARGET_CLOCK_UNCERTAINTY 0.90 CACHE STRING "Clock uncertainty for HLS.")

# Domain options
set(BLAS_DATA_TYPE "float" CACHE STRING "Matrix data type.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(SDAccel REQUIRED)
find_package(Threads REQUIRED)

include_directories(include hlslib/include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -gdwarf-4")

configure_file(include/Config.h.in Config.h)

enable_testing()

set(BLAS_KERNEL_SRC
    # Put HLS source files here
    ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp)
add_library(fpgablas ${BLAS_KERNEL_SRC})
target_link_libraries(fpgablas ${CMAKE_THREAD_LIBS_INIT})

# Host test file
add_executable(TestProgram test/TestProgram.cpp)
target_link_libraries(TestProgram ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} fpgablas)
add_test(TestProgram TestProgram)

# HLS setup
set(BLAS_PROJECT_NAME "BLAS")
string(REPLACE ";" " " BLAS_SRC_SYNTHESIS "${BLAS_KERNEL_SRC}")
set(BLAS_ENTRY_FUNCTION "blas_dot")
if(BLAS_TARGET_CLOCK GREATER 0)
  set(BLAS_TARGET_TIMING "${BLAS_TARGET_CLOCK}MHz")
else()
  set(BLAS_TARGET_TIMING "300MHz")
endif()
configure_file(scripts/Synthesis.tcl.in Synthesis.tcl)
add_custom_target(synthesis
  COMMAND ${SDAccel_VIVADO_HLS} -f Synthesis.tcl)
