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
set(KERNEL_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -gdwarf-4 -Wall -Wextra -Wno-unused-parameter -Wno-unused-label -Wno-unknown-pragmas")

configure_file(include/Config.h.in Config.h)

enable_testing()

set(BLAS_KERNEL_SRC
    # Put HLS source files here
    ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp)
add_library(fpgablas ${BLAS_KERNEL_SRC})
target_link_libraries(fpgablas ${CMAKE_THREAD_LIBS_INIT})

# Host test file
add_executable(TestProgram test/TestProgram.cpp)
target_link_libraries(TestProgram ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} fpgablas blas)
add_test(TestProgram TestProgram)

# HLS setup
set(BLAS_PROJECT_NAME "BLAS")
string(REPLACE ";" " " BLAS_SRC_SYNTHESIS "${BLAS_KERNEL_SRC}")
if(BLAS_TARGET_CLOCK GREATER 0)
  set(BLAS_TARGET_TIMING "${BLAS_TARGET_CLOCK}MHz")
else()
  set(BLAS_TARGET_TIMING "300MHz")
endif()

add_custom_target(delete_hls_project_dir rm -r ${CMAKE_BINARY_DIR}/${BLAS_PROJECT_NAME})

# Synthesis target
function(synthesis_target KERNEL_NAME KERNEL_ENTRY KERNEL_SRC)
	set(BLAS_ENTRY_FUNCTION ${KERNEL_ENTRY})
	set(BLAS_SRC_SYNTHESIS ${KERNEL_SRC})
	configure_file(scripts/Synthesis.tcl.in Synthesis_${KERNEL_NAME}.tcl)
	add_custom_target(synthesize_${KERNEL_NAME}
			COMMAND ${SDAccel_VIVADO_HLS} -f Synthesis_${KERNEL_NAME}.tcl
			DEPENDS delete_hls_project_dir)
endfunction(synthesis_target)

function(kernel_target KERNEL_NAME KERNEL_ENTRY KERNEL_SRC KERNEL_FLAGS)
	set(KERNEL_FLAGS ${KERNEL_CXX_FLAGS} -DHLSLIB_SYNTHESIS ${KERNEL_FLAGS})
	set(XOCC_FLAGS
			--kernel ${KERNEL_ENTRY}
			--platform ${BLAS_DSA_NAME}
			--xp prop:kernel.${KERNEL_ENTRY}.kernel_flags="${KERNEL_FLAGS}"
			${ARGN})

	synthesis_target(${KERNEL_NAME} ${KERNEL_ENTRY} ${KERNEL_SRC})

	if(SDAccel_FOUND)
		if(((${SDAccel_MAJOR_VERSION} LESS 2018) AND (${SDAccel_MINOR_VERSION} LESS 3)) OR ${SDAccel_MAJOR_VERSION} LESS 2017)
			set(CURRENT_GMEM 0)
			foreach(axi IN ITEMS ${BANK_MAPPING})
				set(XOCC_FLAGS ${XOCC_FLAGS}
						--xp misc:map_connect=add.kernel.${KERNEL_ENTRY}_1.M_AXI_GMEM${CURRENT_GMEM}.core.OCL_REGION_0.M0${axi}_AXI)
				MATH(EXPR CURRENT_GMEM ${CURRENT_GMEM}+1)
			endforeach()

			add_custom_target(kernel_${KERNEL_NAME}
					COMMAND ${SDAccel_XOCC}
					-t hw
					-o kernel_${KERNEL_NAME}.xclbin
					-I${CMAKE_SOURCE_DIR}/include
					-I${CMAKE_SOURCE_DIR}/hlslib/include
					-I${CMAKE_BINARY_DIR}
					${KERNEL_SRC}
					${XOCC_FLAGS})
		else()
			set(CURRENT_GMEM 0)
			foreach(axi IN ITEMS ${BANK_MAPPING})
				set(XOCC_FLAGS ${XOCC_FLAGS}
						--sp ${KERNEL_ENTRY}_1.m_axi_gmem${CURRENT_GMEM}:bank${axi})
				MATH(EXPR CURRENT_GMEM ${CURRENT_GMEM}+1)
			endforeach()
			set(XOCC_FLAGS ${XOCC_FLAGS}
					--profile_kernel "data:all:all:all"
					--profile_kernel "stall:all:all"
					--profile_kernel "exec:all:all")

			add_custom_target(compile_kernel_${KERNEL_NAME}
					COMMAND ${SDAccel_XOCC}
					-c
					-t hw
					-o kernel_${KERNEL_NAME}.xo
					-I${CMAKE_SOURCE_DIR}/include
					-I${CMAKE_SOURCE_DIR}/hlslib/include
					-I${CMAKE_BINARY_DIR}
					${KERNEL_SRC}
					${XOCC_FLAGS})
			add_custom_target(kernel_${KERNEL_NAME}
					COMMAND ${SDAccel_XOCC}
					-l
					-t hw
					-o kernel_${KERNEL_NAME}.xclbin
					-I${CMAKE_SOURCE_DIR}/include
					-I${CMAKE_SOURCE_DIR}/hlslib/include
					-I${CMAKE_BINARY_DIR}
					kernel_${KERNEL_NAME}.xo
					${XOCC_FLAGS})
			add_dependencies(kernel_${KERNEL_NAME} compile_kernel_${KERNEL_NAME})
		endif()
	endif()
endfunction(kernel_target)

# Test kernels
kernel_target("dot" "blas_dot" ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp "" "0;1;0")

kernel_target("dot_multiple" "blas_dot_multiple" ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp "" "0;1;0"
		--kernel_frequency 200)


if(SDAccel_FOUND)
	add_executable(RunDot test/RunDot.cpp)
	target_link_libraries(RunDot ${SDAccel_LIBRARIES} blas)
	add_test(RunDot RunDot)

	add_executable(PerfDot test/PerfDot.cpp)
	target_link_libraries(PerfDot ${SDAccel_LIBRARIES})

	# Target for building all kernels
	add_custom_target(kernel_all)
	add_dependencies(kernel_all kernel_dot kernel_dot_multiple)
endif()
