// File to test spmv implementation on Unified Memory

#include "matrix_um.h"
#include "sblas_um.h"

bool spmvCsrTest(const char *A_path, double alpha, double beta, unsigned n_gpu,
                 bool tuning) {
  cpu_timer load_timer, run_timer, run_cpu_timer;
  load_timer.start_timer();
  // Correct
  CsrSparseMatrix<int, double> A(A_path);
  DenseVector<int, double> B(A.width, 1.);
  DenseVector<int, double> C(A.height, 1.);
  DenseVector<int, double> C_cpu(A.height, 1.);
  // Partition and Distribute
  A.sync2gpu(n_gpu, segment);
  B.sync2gpu(n_gpu, replicate);
  C.sync2gpu(n_gpu, replicate);

  if (tuning) {
    A.applyGpuTuning();
    B.applyGpuTuning();
    C.applyGpuTuning(false);
  }

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  load_timer.stop_timer();

  run_timer.start_timer();
  sblas_spmv_csr_v1<int, double>(&A, &B, &C, alpha, beta, n_gpu);
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

    C_cpu.applyCpuTuning();
    B.sync2cpu(0);
  }

  // CPU Baseline
  run_cpu_timer.start_timer();
  sblas_spmv_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_cpu_timer.stop_timer();
  // get data back to CPU
  C.sync2cpu(0, tuning);
  // print_1d_array(C.val,C.get_vec_length());
  // print_1d_array(C_cpu.val,C_cpu.get_vec_length());
  bool correct = check_equal(C_cpu.val, C.val, C.get_vec_length());
  cout << "Validation = " << (correct ? "True" : "False") << endl;
  cout << "Load Time: " << load_timer.measure() << "ms." << endl;
  cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;
  cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
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
