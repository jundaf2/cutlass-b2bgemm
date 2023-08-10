cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(b2bgemm VERSION 3.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit)

# Find CUDA ARCH VERSION
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET( ${CUDA_ARCH_LIST})
list (GET CUDA_ARCH_LIST 0 ARCH_NUMBER)
SET(CUDA_ARCH sm_${ARCH_NUMBER})
SET(CUDA_COMPUTE compute_${ARCH_NUMBER})


message(--------------------------------------------)
message(CUDA_ARCH = ${CUDA_ARCH})
message(CUDA_COMPUTE = ${CUDA_COMPUTE})
message(--------------------------------------------)
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(CUDA_HEADER_DIRS ${CUDA_PATH}/include)
set(CUDA_LIB_DIRS ${CUDA_PATH}/lib64)


# string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address")
# --compiler-options=-fsanitize=address

set(CUDA_FLAGS "--expt-relaxed-constexpr -v  --compiler-options=-fno-omit-frame-pointer --keep -v -Xcompiler=-mf16c -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing")
set(CUDA_FLAGS "-gencode code=${CUDA_ARCH},arch=${CUDA_COMPUTE} ${CUDA_FLAGS}")
string(REGEX REPLACE "^sm_" "" ARCH_NUMBER "${CUDA_ARCH}")

if ("${CUDA_ARCH}" STREQUAL "sm_${ARCH_NUMBER}")
	set(CMAKE_CUDA_ARCHITECTURES "${ARCH_NUMBER}")
else()
	message(FATAL_ERROR "Unknown CUDA_ARCH: ${CUDA_ARCH}")
endif()

message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

set(CUDA_FLAGS "-ccbin=${CMAKE_CXX_COMPILER} ${CUDA_FLAGS}")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS "${CUDA_FLAGS}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -lpthread")

include_directories(
  .
  ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
  ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include/
)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

foreach(FUSION_GEMM_EXAMPLE
  fused_two_gemms_f16_sm80_rf
  # fused_two_gemms_s8_sm80_rf
)

  add_executable(${FUSION_GEMM_EXAMPLE} ${FUSION_GEMM_EXAMPLE}.cu)
  target_include_directories(${FUSION_GEMM_EXAMPLE} PRIVATE  ${CUDA_HEADER_DIRS} SYSTEM ${CUDA_INCLUDE_DIRS})
  target_link_directories(${FUSION_GEMM_EXAMPLE} PRIVATE ${CUDA_LIB_DIRS})
  target_link_libraries (${FUSION_GEMM_EXAMPLE} PRIVATE ${CUDA_LIBRARIES} )
  set_target_properties(${FUSION_GEMM_EXAMPLE} PROPERTIES POSITION_INDEPENDENT_CODE ON)

endforeach()

