// File to test spmv implementation on Unified Memory

#include "matrix_um.h"
#include "sblas_um.h"
#include <nvtx3/nvToolsExt.h>

bool spmvCsrTest(const char *A_path, double alpha, double beta, unsigned n_gpu,
                 bool tuning) {
  cpu_timer load_timer, run_timer, run_cpu_timer;
  load_timer.start_timer();
  // Correct
  nvtxRangePush("Matrix and Vectors Allocation");
  nvtxRangePush("A creation");
  CsrSparseMatrix<int, double> A(A_path);
  nvtxRangePop();
  nvtxRangePush("B creation");
  DenseVector<int, double> B(A.width, 1.);
  nvtxRangePop();
  nvtxRangePush("C creation");
  DenseVector<int, double> C(A.height, 1.);
  nvtxRangePop();
  nvtxRangePush("C_cpu creation");
  DenseVector<int, double> C_cpu(A.height, 1.);
  nvtxRangePop();
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
  nvtxRangePush("SPMV CSR v1");
  run_timer.start_timer();
  sblas_spmv_csr_v1<int, double>(&A, &B, &C, alpha, beta, n_gpu);
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
  nvtxRangePush("SPMV CSR CPU");
  // CPU Baseline
  run_cpu_timer.start_timer();
  sblas_spmv_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_cpu_timer.stop_timer();
  nvtxRangePop();

  // get data back to CPU
  nvtxRangePush("C sync back");
  C.sync2cpu(0, tuning);
  nvtxRangePop();
  // print_1d_array(C.val,C.get_vec_length());
  // print_1d_array(C_cpu.val,C_cpu.get_vec_length());
  nvtxRangePush("C correctness check");
  bool correct = check_equal(C_cpu.val, C.val, C.get_vec_length());
  nvtxRangePop();
  cout << "Validation = " << (correct ? "True" : "False") << endl;
  cout << "Load Time: " << load_timer.measure() << "ms." << endl;
  cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
  cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;

  return correct;
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    cerr << "./spmm_test \
            A_path alpha beta gpus tuning (optional, false by default)"
         << endl;
    exit(1);
  }

  const char *A_path = argv[1];
  const double alpha = atof(argv[2]);
  const double beta = atof(argv[3]);
  const unsigned gpus = atoi(argv[4]);
  const bool tuning = argc > 5 ? atoi(argv[5]) == 1 : false;

  spmvCsrTest(A_path, alpha, beta, gpus, tuning);

  return 0;
}
