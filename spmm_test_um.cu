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

bool spmmCsrTest(const char *A_path, int b_width, double alpha, double beta,
                 unsigned n_gpu) {
  cpu_timer load_timer, run_timer, run_cpu_timer;
  load_timer.start_timer();
  CsrSparseMatrix<int, double> A(A_path);
  cout << "CSR matrix A created" << endl;
  DenseMatrix<int, double> B(A.width, b_width, col_major);
  cout << "Dense matrix B created" << endl;
  DenseMatrix<int, double> C(A.height, b_width, 1, col_major);
  cout << "Dense matrix C created" << endl;
  DenseMatrix<int, double> C_cpu(A.height, b_width, 1, col_major);
  cout << "Dense matrix C_cpu created" << endl;
  // Partition and Distribute
  A.sync2gpu(n_gpu, replicate);
  cout << "A synced to GPU" << endl;
  B.sync2gpu(n_gpu, segment);
  cout << "B synced to GPU" << endl;
  C.sync2gpu(n_gpu, segment);
  cout << "C synced to GPU" << endl;

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  load_timer.stop_timer();

  run_timer.start_timer();
  sblas_spmm_csr_v1<int, double>(&A, &B, &C, alpha, beta, n_gpu);
  cout << "sblas_spmm_csr_v1 completed" << endl;
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_timer.stop_timer();

  run_cpu_timer.start_timer();
  sblas_spmm_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
  cout << "sblas_spmm_csr_cpu completed" << endl;
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_cpu_timer.stop_timer();

  // print_1d_array(C_cpu.val,C_cpu.get_mtx_num());
  // print_1d_array(C.val,C.get_mtx_num());

  bool correct = check_equal(C_cpu.val, C.val_gpu[0], C.get_mtx_num());
  cout << "Validation = " << (correct ? "True" : "False") << endl;
  cout << "Load Time: " << load_timer.measure() << "ms." << endl;
  cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
  cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;
  return correct;
}

bool spmmCsrTest2(const char *A_path, int b_width, double alpha, double beta,
                  unsigned n_gpu) {
  cpu_timer load_timer, run_timer, run_cpu_timer;
  load_timer.start_timer();
  // CsrSparseMatrix<int, double> A("./ash85.mtx");
  CsrSparseMatrix<int, double> A(A_path, n_gpu);
  cout << "CSR matrix A created" << endl;
  DenseMatrix<int, double> B(A.width, b_width, col_major);
  cout << "Dense matrix B created" << endl;
  DenseMatrix<int, double> C(A.height, b_width, 1, col_major);
  cout << "Dense matrix C created" << endl;
  DenseMatrix<int, double> C_cpu(A.height, b_width, 1, col_major);
  cout << "Dense matrix C_cpu created" << endl;

  // Partition and Distribute
  A.sync2gpu(n_gpu, segment);
  cout << "A synced to GPU" << endl;
  B.sync2gpu(n_gpu, replicate);
  cout << "B synced to GPU" << endl;
  C.sync2gpu(n_gpu, replicate);
  cout << "C synced to GPU" << endl;

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  load_timer.stop_timer();
  run_timer.start_timer();
  sblas_spmm_csr_v2<int, double>(&A, &B, &C, alpha, beta, n_gpu);
  cout << "sblas_spmm_csr_v2 completed" << endl;
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_timer.stop_timer();
  run_cpu_timer.start_timer();
  sblas_spmm_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
  cout << "sblas_spmm_csr_cpu completed" << endl;
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  run_cpu_timer.stop_timer();
  // get data back to CPU
  C.sync2cpu(0);
  // print_1d_array(C.val,C.get_mtx_num());
  // print_1d_array(C_cpu.val,C_cpu.get_mtx_num());
  bool correct = check_equal(C_cpu.val, C.val_gpu[0], C.get_mtx_num());
  cout << "Validation = " << (correct ? "True" : "False") << endl;
  cout << "Load Time: " << load_timer.measure() << "ms." << endl;
  cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
  cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;
  return correct;
}

int main(int argc, char *argv[]) {
  if (argc != 7) {
    cerr << "./spmm_test method(1:partition-B, 2:partition-A) \
            A_path B_width alpha beta gpus"
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
  if (method == 1) {
    spmmCsrTest(A_path, B_width, alpha, beta, gpus);
  } else if (method == 2) {
    spmmCsrTest2(A_path, B_width, alpha, beta, gpus);
    // spmmCsrTest2(256,3.0,4.0,4);
  } else {
    cerr << "Method can be only 1 or 2." << endl;
    exit(1);
  }
  return 0;
}
