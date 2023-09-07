/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "device/b2b_gemm.h"
#include "b2b_gemm_run.h"


#include <iostream>

// Run tests on GPUs 

int testRun(int arch, std::vector<bool (*)(cutlass::gemm::GemmCoord, cutlass::gemm::GemmCoord)> & test_funcs, const cutlass::gemm::GemmCoord & gemm_param0, const cutlass::gemm::GemmCoord & gemm_param1, const std::string & test_name) {

  bool supported = false;

  int arch_major = arch / 10;
  int arch_minor = arch - arch / 10 * 10;  

  if(arch_major >= 8) {
    // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
    if (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0)) {
      supported = true;
    }
  }
  else if(arch_major >= 7) {
    // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
    //
    // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
    if (__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)) {
      supported = true;
    }
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!(props.major == arch_major /*&& props.minor == arch_minor*/)) {
    supported = false;
  }

  std::cout << "Current Arch: sm" << props.major << props.minor << std::endl;
  std::cout << "Expected Arch: SM" << arch << std::endl;

  if (!supported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    std::cout << "This example isn't supported on current architecture" << std::endl;
    return 0;
  }

  bool pass = true;
 
  std::cout << "Device: " << props.name << std::endl;
  std::cout << "Arch: SM" << arch << std::endl;
  std::cout << "Test: " << test_name << std::endl;
  for(auto func : test_funcs) {
    pass &= func(gemm_param0, gemm_param1);
  }


  if(pass)
    return 0;
  else
    return -1;

}
////////////////////////////////////////////////////////////////////////////////

bool run_nonfused_gemm_f16_sm80(cutlass::gemm::GemmCoord gemm_param0, cutlass::gemm::GemmCoord gemm_param1) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(1); //beta=1 for bias
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using Gemm0 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;
  using Gemm1 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;

  B2bNonFusedGemmRun<Gemm0, Gemm1> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back FP16 TN GEMMs...\n";
  bool pass = nonFusedGemm.run(gemm_param0, gemm_param1, alpha0, beta0, alpha1, beta1);
  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

bool run_fused_gemm_f16_sm80_rf_res(cutlass::gemm::GemmCoord gemm_param0, cutlass::gemm::GemmCoord gemm_param1) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0); 
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  // constexpr int GEMM0_BLK_WRP_N = 256;
  // constexpr int GEMM1_BLK_WRP_N = 128;
  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      InstructionShape::kM * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3
  >;

  B2bFusedGemmRun<B2bGemm> fusedGemm;

  std::cout << "Running Fused back-to-back FP16 TN GEMMs with RF residency...\n";
  bool passed = fusedGemm.run(gemm_param0, gemm_param1, alpha0, beta0, alpha1, beta1);
  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;

}


bool run_fused_gemm_f16_sm80_shmem(cutlass::gemm::GemmCoord gemm_param0, cutlass::gemm::GemmCoord gemm_param1) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0); 
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  // 240 2560 5120
  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 256, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 32, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 512, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      InstructionShape::kM * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;


  const bool SmemAccumulator = true;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    SmemAccumulator
  >;

  B2bFusedGemmRun<B2bGemm> fusedGemm;

  std::cout << "Running Fused back-to-back FP16 TN GEMMs with shared memory staging...\n";
  bool passed = fusedGemm.run(gemm_param0, gemm_param1, alpha0, beta0, alpha1, beta1);
  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;

}

int main(int argc, char * const argv[]) {
  int batch = std::atoi(argv[1]);
  int input_dim = std::atoi(argv[2]); // K
  int seq_len = std::atoi(argv[3]); // M
  int tp_degree = std::atoi(argv[4]); 
  int expert_count = std::atoi(argv[5]);
  int expert_dim = std::atoi(argv[6]); // N
  std::cout << "batch: " << batch << ", input_dim: " << input_dim << ", seq_len: " << seq_len << ", tp_degree: " << tp_degree << ", expert_count: " << expert_count << ", expert_dim: " << expert_dim << std::endl;

  // M N K: (0) M N K (1) M K N
  int M = batch * seq_len / expert_count;
  int K = input_dim;
  int N = expert_dim / tp_degree;
  std::cout << "M: " << M << ", K: " << K << ", N: " << N << std::endl;

  cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_0(M, K, N);
  cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_1(M, N, K);

  // cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_0(128*640, 64, 576);
  // cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_1(128*640, 128, 64);
  // cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_0(128*640, 64, 576);
  // cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_1(128*640, 256, 64);

  std::vector<bool (*)(cutlass::gemm::GemmCoord, cutlass::gemm::GemmCoord)>funcs = {
    &run_nonfused_gemm_f16_sm80
    ,
    &run_fused_gemm_f16_sm80_rf_res
    ,
    &run_fused_gemm_f16_sm80_shmem
  };

  return testRun(80, funcs, gemm_f16_sm80_problem_size_0, gemm_f16_sm80_problem_size_1, "gemm f16");

}




////////////////////////////////////////////////////////////////////////////////
