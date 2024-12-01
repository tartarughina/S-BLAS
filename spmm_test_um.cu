// ------------------------------------------------------------------------
// File: spmm_test.cu
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file tests the SPMM implementation.
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#include "matrix_um.h"
#include "sblas_um.h"
#include "spmm_um.h"
#include <nvtx3/nvToolsExt.h>

bool spmmCsrTest(const char *A_path, int b_width, double alpha, double beta,
                 unsigned n_gpu, bool tuning) {
  cpu_timer load_timer, run_timer, run_cpu_timer;
  load_timer.start_timer();
  CsrSparseMatrix<int, double> A(A_path);
  DenseMatrix<int, double> B(A.width, b_width, col_major);
  DenseMatrix<int, double> C(A.height, b_width, 1, col_major);
  DenseMatrix<int, double> C_cpu(A.height, b_width, 1, col_major);
  // Partition and Distribute
  A.sync2gpu(n_gpu, replicate);
  B.sync2gpu(n_gpu, segment);
  C.sync2gpu(n_gpu, segment);

  if (tuning) {
    A.applyGpuTuning();
    B.applyGpuTuning();
    C.applyGpuTuning(false);
  }

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  load_timer.stop_timer();

  run_timer.start_timer();
  sblas_spmm_csr_v1<int, double>(&A, &B, &C, alpha, beta, n_gpu);
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_timer.stop_timer();

  if (tuning) {
    A.removeGpuTuning();
    B.removeGpuTuning();
    C.removeGpuTuning(false);

    A.applyCpuTuning();
    B.applyCpuTuning();
    C.applyCpuTuning();
  }

  run_cpu_timer.start_timer();
  sblas_spmm_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_cpu_timer.stop_timer();

  // cout << "GPU result" << endl;
  // print_1d_array(C.val, C.get_mtx_num());
  // cout << "CPU result" << endl;
  // print_1d_array(C_cpu.val, C_cpu.get_mtx_num());

  bool correct = check_equal(C_cpu.val, C.val, C.get_mtx_num());
  cout << "Validation = " << (correct ? "True" : "False") << endl;
  cout << "Load Time: " << load_timer.measure() << "ms." << endl;
  cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
  cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;
  return correct;
}

bool spmmCsrTest2(const char *A_path, int b_width, double alpha, double beta,
                  unsigned n_gpu, bool tuning) {
  cpu_timer load_timer, run_timer, run_cpu_timer;
  load_timer.start_timer();
  // CsrSparseMatrix<int, double> A("./ash85.mtx");
  nvtxRangePush("Matrices Allocation");
  CsrSparseMatrix<int, double> A(A_path);
  DenseMatrix<int, double> B(A.width, b_width, col_major);
  DenseMatrix<int, double> C(A.height, b_width, 1, col_major);
  DenseMatrix<int, double> C_cpu(A.height, b_width, 1, col_major);
  nvtxRangePop();
  // Partition and Distribute
  nvtxRangePush("A Distribution");
  A.sync2gpu(n_gpu, segment);
  nvtxRangePop();
  nvtxRangePush("B Distribution");
  B.sync2gpu(n_gpu, replicate);
  nvtxRangePop();
  nvtxRangePush("C Distribution");
  C.sync2gpu(n_gpu, replicate);
  nvtxRangePop();

  if (tuning) {
    nvtxRangePush("GPU Tuning");
    A.applyGpuTuning();
    B.applyGpuTuning();
    C.applyGpuTuning(false);
    nvtxRangePop();
  }

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  load_timer.stop_timer();
  nvtxRangePush("SPMM CSR v2");
  run_timer.start_timer();
  sblas_spmm_csr_v2<int, double>(&A, &B, &C, alpha, beta, n_gpu);
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_timer.stop_timer();
  nvtxRangePop();
  if (tuning) {
    nvtxRangePush("CPU tuning");
    A.removeGpuTuning();
    B.removeGpuTuning();
    C.removeGpuTuning(false);

    A.applyCpuTuning();
    B.applyCpuTuning();
    C.applyCpuTuning();

    C_cpu.applyCpuTuning();
    B.sync2cpu(0, tuning);
    nvtxRangePop();
  }
  nvtxRangePush("SPMM CSR CPU");
  run_cpu_timer.start_timer();
  sblas_spmm_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_cpu_timer.stop_timer();
  nvtxRangePop();
  // get data back to CPU
  nvtxRangePush("C sync back");
  C.sync2cpu(0, tuning);
  nvtxRangePop();
  // cout << "GPU result" << endl;
  // print_1d_array(C.val, C.get_mtx_num());
  // cout << "CPU result" << endl;
  // print_1d_array(C_cpu.val, C_cpu.get_mtx_num());
  nvtxRangePush("C correctness check");
  bool correct = check_equal(C_cpu.val, C.val, C.get_mtx_num());
  nvtxRangePop();
  cout << "Validation = " << (correct ? "True" : "False") << endl;
  cout << "Load Time: " << load_timer.measure() << "ms." << endl;
  cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
  cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;
  return correct;
}

// NCCL is exectuted in the 2 method, where mat A is partitioned

int main(int argc, char *argv[]) {
  if (argc < 7) {
    cerr << "./spmm_test method(1:partition-B, 2:partition-A) \
            A_path B_width alpha beta gpus tuning (optional, false by default)"
         << endl;
    exit(1);
  }
  const int method =
      atoi(argv[1]); // method-1: partition-B, method-2: partition-A
  const char *A_path = argv[2];
  const int B_width = atof(argv[3]);
  const double alpha = atof(argv[4]);
  const double beta = atof(argv[5]);
  const unsigned gpus = atoi(argv[6]);
  const bool tuning = argc > 7 ? atoi(argv[7]) == 1 : false;

  if (method == 1) {
    spmmCsrTest(A_path, B_width, alpha, beta, gpus, tuning);
  } else if (method == 2) {
    spmmCsrTest2(A_path, B_width, alpha, beta, gpus, tuning);
  } else {
    cerr << "Method can be only 1 or 2." << endl;
    exit(1);
  }
  return 0;
}
