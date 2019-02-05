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

include_directories(include include/fblaslib hlslib/include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
set(KERNEL_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -gdwarf-4 -Wall -Wextra -Wno-unused-parameter -Wno-unused-label -Wno-unknown-pragmas")

configure_file(include/Config.h.in Config.h)

enable_testing()

set(BLAS_KERNEL_SRC
	    # Put HLS source files here
	    ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp
		${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp
		${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp
		${CMAKE_SOURCE_DIR}/kernel/blas_level2.cpp)
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
function(synthesis_target KERNEL_NAME KERNEL_ENTRY KERNEL_SRC KERNEL_FLAGS)
	set(BLAS_ENTRY_FUNCTION ${KERNEL_ENTRY})
	set(BLAS_SRC_SYNTHESIS ${KERNEL_SRC})
	set(KERNEL_CXX_FLAGS "${KERNEL_CXX_FLAGS} ${KERNEL_FLAGS}")
	configure_file(scripts/Synthesis.tcl.in Synthesis_${KERNEL_NAME}.tcl)
	add_custom_target(synthesize_${KERNEL_NAME}
			COMMAND ${SDAccel_VIVADO_HLS} -f Synthesis_${KERNEL_NAME}.tcl
			DEPENDS delete_hls_project_dir)
endfunction(synthesis_target)

function(kernel_target KERNEL_NAME KERNEL_ENTRY KERNEL_SRC KERNEL_FLAGS BANK_MAPPING)
	if (NOT KERNEL_FLAGS)
		set(KERNEL_FLAGS " ")
	endif ()
	synthesis_target(${KERNEL_NAME} ${KERNEL_ENTRY} ${KERNEL_SRC} ${KERNEL_FLAGS})

	separate_arguments(KERNEL_CXX_FLAGS)
	separate_arguments(KERNEL_FLAGS)
	set(KERNEL_FLAGS ${KERNEL_CXX_FLAGS} -DHLSLIB_SYNTHESIS ${KERNEL_FLAGS})
	set(XOCC_FLAGS
			#-O3
			--kernel ${KERNEL_ENTRY}
			--platform ${BLAS_DSA_NAME}
			-I${CMAKE_SOURCE_DIR}/include
			-I${CMAKE_SOURCE_DIR}/include/fblaslib
			-I${CMAKE_SOURCE_DIR}/hlslib/include
			-I${CMAKE_BINARY_DIR}
			--xp prop:kernel.${KERNEL_ENTRY}.kernel_flags="${KERNEL_FLAGS}"
			${ARGN})

	if(SDAccel_FOUND)
		if(NOT (BLAS_DSA_NAME MATCHES "dynamic"))
			set(CURRENT_GMEM 0)
			foreach(axi IN ITEMS ${BANK_MAPPING})
				set(XOCC_FLAGS ${XOCC_FLAGS}
						--xp misc:map_connect=add.kernel.${KERNEL_ENTRY}_1.M_AXI_GMEM${CURRENT_GMEM}.core.OCL_REGION_0.M0${axi}_AXI)
				MATH(EXPR CURRENT_GMEM ${CURRENT_GMEM}+1)
			endforeach()

			add_custom_target(kernel_${KERNEL_NAME}_hw_emu
					COMMAND ${SDAccel_XOCC}
					-g
					-t hw_emu
					-o kernel_${KERNEL_NAME}_hw_emu.xclbin
					${KERNEL_SRC}
					${XOCC_FLAGS})

			add_custom_target(kernel_${KERNEL_NAME}
					COMMAND ${SDAccel_XOCC}
					-t hw
					-o kernel_${KERNEL_NAME}.xclbin
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

			add_custom_target(compile_kernel_${KERNEL_NAME}_hw_emu
					COMMAND ${SDAccel_XOCC}
					-c
					-g
					-t hw_emu
					-o kernel_${KERNEL_NAME}_hw_emu.xo
					${KERNEL_SRC}
					${XOCC_FLAGS})

			add_custom_target(kernel_${KERNEL_NAME}_hw_emu
					COMMAND ${SDAccel_XOCC}
					-l
					-g
					-t hw_emu
					-o kernel_${KERNEL_NAME}_hw_emu.xclbin
					kernel_${KERNEL_NAME}_hw_emu.xo
					${XOCC_FLAGS})
			add_dependencies(kernel_${KERNEL_NAME}_hw_emu compile_kernel_${KERNEL_NAME}_hw_emu)

			add_custom_target(compile_kernel_${KERNEL_NAME}
					COMMAND ${SDAccel_XOCC}
					-c
					-t hw
					-o kernel_${KERNEL_NAME}.xo
					${KERNEL_SRC}
					${XOCC_FLAGS})
			add_custom_target(kernel_${KERNEL_NAME}
					COMMAND ${SDAccel_XOCC}
					-l
					-t hw
					-o kernel_${KERNEL_NAME}.xclbin
					kernel_${KERNEL_NAME}.xo
					${XOCC_FLAGS})
			add_dependencies(kernel_${KERNEL_NAME} compile_kernel_${KERNEL_NAME})
		endif()
	endif()
endfunction(kernel_target)

# functions for multiple kernels in one kernel file
function(multiple_kernel_compile_target KERNEL_ENTRY KERNEL_SRC KERNEL_FLAGS)
	if (NOT KERNEL_FLAGS)
		set(KERNEL_FLAGS " ")
	endif ()
	synthesis_target(${KERNEL_ENTRY} ${KERNEL_ENTRY} ${KERNEL_SRC} ${KERNEL_FLAGS})

	separate_arguments(KERNEL_CXX_FLAGS)
	separate_arguments(KERNEL_FLAGS)
	set(KERNEL_FLAGS ${KERNEL_CXX_FLAGS} -DHLSLIB_SYNTHESIS ${KERNEL_FLAGS})
	set(XOCC_FLAGS
			#-O3
			--kernel ${KERNEL_ENTRY}
			--platform ${BLAS_DSA_NAME}
			--xp prop:kernel.${KERNEL_ENTRY}.kernel_flags="${KERNEL_FLAGS}"
			-I${CMAKE_SOURCE_DIR}/include
			-I${CMAKE_SOURCE_DIR}/include/fblaslib
			-I${CMAKE_SOURCE_DIR}/hlslib/include
			-I${CMAKE_BINARY_DIR}
			${KERNEL_SRC}
			${ARGN})

	if(SDAccel_FOUND)
		if(BLAS_DSA_NAME MATCHES "dynamic")
			set(XOCC_FLAGS ${XOCC_FLAGS}
					--profile_kernel "data:all:all:all"
					--profile_kernel "stall:all:all"
					--profile_kernel "exec:all:all")
		endif()
		add_custom_target(compile_multiple_kernel_${KERNEL_ENTRY}_hw_emu
				COMMAND ${SDAccel_XOCC}
				-c
				-g
				-t hw_emu
				-o kernel_${KERNEL_ENTRY}.xo
				${XOCC_FLAGS})

		add_custom_target(compile_multiple_kernel_${KERNEL_ENTRY}
				COMMAND ${SDAccel_XOCC}
				-c
				-t hw
				-o kernel_${KERNEL_ENTRY}.xo
				${XOCC_FLAGS})
	endif()
endfunction(multiple_kernel_compile_target)

function(multiple_kernel_link_target KERNEL_NAME XOCC_FLAGS)
	set(XOCC_FLAGS ${XOCC_FLAGS}
			#-O3
			--platform ${BLAS_DSA_NAME})

	math(EXPR NUM_KERNELS "(${ARGC} - 2) % 2")
	if(NOT (${NUM_KERNELS} EQUAL 0))
		message(FATAL_ERROR "The number of parameters for multiple_kernel_link_target has to be given in pairs of 2")
	endif ()

	if(SDAccel_FOUND)
		set(DEPENDENCIES "")
		set(DEPENDENCIES_HW_EMU "")
		set(ARG_NUM 2)  # current argument index
		while(ARG_NUM LESS ARGC)
			set(KERNEL_FILE ${ARGV${ARG_NUM}})
			set(XOCC_FLAGS ${XOCC_FLAGS}
					kernel_${KERNEL_FILE}.xo)
			list(APPEND DEPENDENCIES compile_multiple_kernel_${KERNEL_FILE})
			list(APPEND DEPENDENCIES_HW_EMU compile_multiple_kernel_${KERNEL_FILE}_hw_emu)
			math(EXPR ARG_NUM "${ARG_NUM}+1")

			set(BANK_MAPPING ${ARGV${ARG_NUM}})
			math(EXPR ARG_NUM "${ARG_NUM}+1")

			if(NOT (BLAS_DSA_NAME MATCHES "dynamic"))
				set(CURRENT_GMEM 0)
				foreach(axi IN ITEMS ${BANK_MAPPING})
					set(XOCC_FLAGS ${XOCC_FLAGS}
							--xp misc:map_connect=add.kernel.${KERNEL_FILE}_1.M_AXI_GMEM${CURRENT_GMEM}.core.OCL_REGION_0.M0${axi}_AXI)
					MATH(EXPR CURRENT_GMEM ${CURRENT_GMEM}+1)
				endforeach()
			else()
				set(CURRENT_GMEM 0)
				foreach(axi IN ITEMS ${BANK_MAPPING})
					set(XOCC_FLAGS ${XOCC_FLAGS}
							--sp ${KERNEL_FILE}_1.m_axi_gmem${CURRENT_GMEM}:bank${axi})
					MATH(EXPR CURRENT_GMEM ${CURRENT_GMEM}+1)
				endforeach()

				set(XOCC_FLAGS ${XOCC_FLAGS}
						--profile_kernel "data:all:all:all"
						--profile_kernel "stall:all:all"
						--profile_kernel "exec:all:all")
			endif()
		endwhile()

		add_custom_target(kernel_${KERNEL_NAME}_hw_emu
				COMMAND ${SDAccel_XOCC}
				-l
				-g
				-t hw_emu
				-o kernel_${KERNEL_NAME}_hw_emu.xclbin
				${XOCC_FLAGS})

		add_custom_target(kernel_${KERNEL_NAME}
			COMMAND ${SDAccel_XOCC}
			-l
			-t hw
			-o kernel_${KERNEL_NAME}.xclbin
			${XOCC_FLAGS})

		add_dependencies(kernel_${KERNEL_NAME}_hw_emu ${DEPENDENCIES_HW_EMU})
		add_dependencies(kernel_${KERNEL_NAME} ${DEPENDENCIES})
	endif()
endfunction(multiple_kernel_link_target)

# Test kernels
kernel_target("dot" "blas_dot" ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp "" "0;1;0")
kernel_target("dot_32" "blas_dot" ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp "-DDOT_WIDTH=32" "0;1;0")

kernel_target("dot_multiple" "blas_dot_multiple" ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp "" "0;1;0"
		--kernel_frequency 200)
kernel_target("dot_multiple_32" "blas_dot_multiple" ${CMAKE_SOURCE_DIR}/src/DotProduct.cpp "-DDOT_WIDTH=32" "0;1;0"
		--kernel_frequency 200)

kernel_target("gemv" "blas_gemv" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "" "0;1;2")
kernel_target("gemv_32_16_16" "blas_gemv" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "-DGEMV_SIZE_ROWCHUNK=32" "0;1;2")
kernel_target("gemv_16_32_16" "blas_gemv" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "-DGEMV_SIZE_COLCHUNK=32 -DGEMV_SIZE_COLUMN=16" "0;1;2")
kernel_target("gemv_32_32_16" "blas_gemv" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "-DGEMV_SIZE_ROWCHUNK=32 -DGEMV_SIZE_COLCHUNK=32 -DGEMV_SIZE_COLUMN=16" "0;1;2")
kernel_target("gemv_16_64_16" "blas_gemv" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "-DGEMV_SIZE_COLCHUNK=64 -DGEMV_SIZE_COLUMN=16" "0;1;2")

kernel_target("gemv_transposed" "blas_gemv_transposed" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "" "0;1;2")
kernel_target("gemv_transposed_32_16_16" "blas_gemv_transposed" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "-DGEMV_SIZE_ROWCHUNK=32" "0;1;2")
kernel_target("gemv_transposed_16_32_16" "blas_gemv_transposed" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "-DGEMV_SIZE_COLCHUNK=32 -DGEMV_SIZE_COLUMN=16" "0;1;2")
kernel_target("gemv_transposed_32_32_16" "blas_gemv_transposed" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "-DGEMV_SIZE_ROWCHUNK=32 -DGEMV_SIZE_COLCHUNK=32 -DGEMV_SIZE_COLUMN=16" "0;1;2")
kernel_target("gemv_transposed_16_64_16" "blas_gemv_transposed" ${CMAKE_SOURCE_DIR}/kernel/GEMV.cpp "-DGEMV_SIZE_COLCHUNK=64" "0;1;2")

multiple_kernel_compile_target("sscal" ${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp "-DWIDTH=32 -DFBLAS_DISABLE_DOUBLE")
multiple_kernel_compile_target("saxpy" ${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp "-DWIDTH=32 -DFBLAS_DISABLE_DOUBLE")
multiple_kernel_compile_target("sdot" ${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp "-DWIDTH=32 -DFBLAS_DISABLE_DOUBLE")
multiple_kernel_compile_target("snrm2" ${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp "-DWIDTH=32 -DFBLAS_DISABLE_DOUBLE")
multiple_kernel_compile_target("sgemv" ${CMAKE_SOURCE_DIR}/kernel/blas_level2.cpp "-DWIDTH=32 -DGEMV_SIZE_COLCHUNK=256 -DGEMV_SIZE_ROWCHUNK=64 -DFBLAS_DISABLE_DOUBLE")
multiple_kernel_compile_target("sgemv_transposed" ${CMAKE_SOURCE_DIR}/kernel/blas_level2.cpp "-DWIDTH=32 -DGEMV_SIZE_COLCHUNK=256 -DGEMV_SIZE_ROWCHUNK=64 -DFBLAS_DISABLE_DOUBLE")
multiple_kernel_link_target("fblas_single" "" "sdot" "0;1;2" "snrm2" "0;1" "sgemv" "0;1;2;3" "sgemv_transposed" "0;1;2;3")

multiple_kernel_compile_target("dscal" ${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp "-DWIDTH=16 -DFBLAS_DISABLE_SINGLE")
multiple_kernel_compile_target("daxpy" ${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp "-DWIDTH=16 -DFBLAS_DISABLE_SINGLE")
multiple_kernel_compile_target("ddot" ${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp "-DWIDTH=16 -DFBLAS_DISABLE_SINGLE")
multiple_kernel_compile_target("dnrm2" ${CMAKE_SOURCE_DIR}/kernel/blas_level1.cpp "-DWIDTH=16 -DFBLAS_DISABLE_SINGLE")
multiple_kernel_compile_target("dgemv" ${CMAKE_SOURCE_DIR}/kernel/blas_level2.cpp "-DWIDTH=16 -DGEMV_SIZE_COLCHUNK=128 -DGEMV_SIZE_ROWCHUNK=32 -DFBLAS_DISABLE_SINGLE")
multiple_kernel_compile_target("dgemv_transposed" ${CMAKE_SOURCE_DIR}/kernel/blas_level2.cpp "-DWIDTH=16 -DGEMV_SIZE_COLCHUNK=128 -DGEMV_SIZE_ROWCHUNK=32 -DFBLAS_DISABLE_SINGLE")
multiple_kernel_link_target("fblas_double" "" "ddot" "0;1;2" "dnrm2" "0;1" "dgemv" "0;1;2;3" "dgemv_transposed" "0;1;2;3")

if(SDAccel_FOUND)
	add_executable(RunDot test/RunDot.cpp)
	target_link_libraries(RunDot ${SDAccel_LIBRARIES} blas)
	add_test(RunDot RunDot)

	add_executable(PerfDot test/PerfDot.cpp)
	target_link_libraries(PerfDot ${SDAccel_LIBRARIES} blas)

	add_executable(RunGemv test/RunGemv.cpp)
	target_link_libraries(RunGemv ${SDAccel_LIBRARIES} blas)
	add_test(RunGemv RunGemv)

	add_executable(PerfGemv test/PerfGemv.cpp)
	target_link_libraries(PerfGemv ${SDAccel_LIBRARIES})

	add_library(fblas ${CMAKE_SOURCE_DIR}/host/fblas.cpp)
	target_link_libraries(fblas ${SDAccel_LIBRARIES})

	# Target for building all kernels
	add_custom_target(kernel_all)
	add_dependencies(kernel_all kernel_dot kernel_dot_multiple)
endif()
