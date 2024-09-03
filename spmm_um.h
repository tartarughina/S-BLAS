// ------------------------------------------------------------------------
// File: spmm.h
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file implements the Sparse-Matrix-Dense-Matrix multiplication (SPMM).
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#ifndef SPMM_H
#define SPMM_H

#include <assert.h>
#include <cusparse.h>
#include <iostream>
#include <nccl.h>
#include <omp.h>

#include <mpi.h>

#include "matrix.h"
#include "utility.h"

using namespace std;
//========================== CPU baseline version ============================
template <typename IdxType, typename DataType>
void sblas_spmm_csr_cpu(CsrSparseMatrix<IdxType, DataType> *pA,
                        DenseMatrix<IdxType, DataType> *pB,
                        DenseMatrix<IdxType, DataType> *pC, DataType alpha,
                        DataType beta) {
  assert((pA->width) == (pB->height));
  assert((pA->height) == (pC->height));
  assert((pB->width) == (pC->width));
  if (pB->order == row_major) {
    cerr << "SBLAS_SPMM_CSR_CPU: B should be in column major!" << endl;
    exit(-1);
  }
  if (pC->order == row_major) {
    for (IdxType i = 0; i < pA->height; i++) {
      for (IdxType n = 0; n < pB->width; n++) {
        DataType sum = 0;
        for (IdxType j = pA->csrRowPtr[i]; j < pA->csrRowPtr[i + 1]; j++) {
          IdxType col_A = pA->csrColIdx[j];
          DataType val_A = pA->csrVal[j];
          DataType val_B = pB->val[n * (pB->height) + col_A];
          sum += val_A * val_B;
        }
        pC->val[i * (pC->width) + n] =
            beta * (pC->val[n * (pC->height) + i]) + alpha * sum;
      }
    }
  } else {
    for (IdxType i = 0; i < pA->height; i++) {
      for (IdxType n = 0; n < pB->width; n++) {
        DataType sum = 0;
        for (IdxType j = pA->csrRowPtr[i]; j < pA->csrRowPtr[i + 1]; j++) {
          IdxType col_A = pA->csrColIdx[j];
          DataType val_A = pA->csrVal[j];
          DataType val_B = pB->val[n * (pB->height) + col_A];
          sum += val_A * val_B;
        }
        pC->val[n * (pC->height) + i] =
            beta * (pC->val[n * (pC->height) + i]) + alpha * sum;
      }
    }
  }
}

/** Compute Sparse-Matrix-Dense-Matrix Multiplication using multi-GPUs.
  * Since A and B are allocated on unified memory, there is no need for memcpy.
  * The idea is to reuse A on each GPU and parition B, then each GPU calls
  * cuSparse single-GPU spMM to compute its own share. For this method, there is
  * no explicit inter-GPU communication required.

  * ---------  C = A * B -----------
  * A[m*k] in CSR sparse format
  * B[k*n] in column major dense format
  * C[m*n] in column major dense format
  */
template <typename IdxType, typename DataType>
void sblas_spmm_csr_v1(CsrSparseMatrix<IdxType, DataType> *pA,
                       DenseMatrix<IdxType, DataType> *pB,
                       DenseMatrix<IdxType, DataType> *pC, DataType alpha,
                       DataType beta, unsigned n_gpu) {
  assert((pA->width) == (pB->height));
  assert((pA->height) == (pC->height));
  assert((pB->width) == (pC->width));
  if (pB->order == row_major) {
    cerr << "SBLAS_SPMM_CSR_V1: B should be in column major!" << endl;
    exit(-1);
  }
  if (pC->order == row_major) {
    cerr << "SBLAS_SPMM_CSR_V1: C should be in column major!" << endl;
    exit(-1);
  }
  // Start OpenMP

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cusparseHandle_t handle;
  cusparseMatDescr_t mat_A;
  cusparseStatus_t cusparse_status;
  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&mat_A));
  CHECK_CUSPARSE(cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO));
  printf("gpu-%d m:%d,n:%ld,k:%d\n", rank, pA->height,
         pB->get_dim_gpu_num(rank), pA->width);
  CHECK_CUSPARSE(cusparseDcsrmm(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, pA->height,
      pB->get_dim_gpu_num(rank), pA->width, pA->nnz_gpu[rank], &alpha, mat_A,
      pA->csrVal[rank], pA->csrRowPtr[rank], pA->csrColIdx[rank], pB->val[rank],
      pA->width, &beta, pC->val[rank], pC->height));
  pC->sync2cpu(rank);

  CHECK_CUSPARSE(cusparseDestroyMatDescr(mat_A));
  CHECK_CUSPARSE(cusparseDestroy(handle));
}

template <typename IdxType, typename DataType>
void sblas_spmm_csr_v2(CsrSparseMatrix<IdxType, DataType> *pA,
                       DenseMatrix<IdxType, DataType> *pB,
                       DenseMatrix<IdxType, DataType> *pC, DataType alpha,
                       DataType beta, unsigned n_gpu) {
  assert((pA->width) == (pB->height));
  assert((pA->height) == (pC->height));
  assert((pB->width) == (pC->width));
  if (pB->order == row_major) {
    cerr << "SBLAS_SPMM_CSR_V2: B should be in column major!" << endl;
    exit(-1);
  }
  if (pC->order == row_major) {
    cerr << "SBLAS_SPMM_CSR_V2: C should be in col major!" << endl;
    exit(-1);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                         sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < nRanks; p++) {
    if (p == myRank)
      break;
    if (hostHashs[p] == hostHashs[myRank])
      localRank++;
  }

  ncclUniqueId id;
  if (rank == 0) {
    ncclGetUniqueId(&id);
  }
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  CUDACHECK(cudaSetDevice(localRank));

  ncclComm_t comm;
  DenseMatrix<IdxType, DataType> C_copy(pC->height, pC->width, 0., row_major);
  C_copy.sync2gpu(n_gpu, replicate);
  // Start OpenMP

  CHECK_NCCL(ncclCommInitRank(&comm, n_gpu, id, rank));
  cusparseHandle_t handle;
  cusparseMatDescr_t mat_A;
  cusparseStatus_t cusparse_status;
  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&mat_A));
  CHECK_CUSPARSE(cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO));
  DataType dummy_alpha = 1.0;
  DataType dummy_beta = 1.0;
  CHECK_CUSPARSE(cusparseDcsrmm(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      pA->get_gpu_row_ptr_num(rank) - 1, pB->width, pA->width,
      pA->nnz_gpu[rank], &dummy_alpha, mat_A, pA->csrVal, pA->csrRowPtr,
      pA->csrColIdx, pB->val, pB->height, &dummy_beta,
      /*C_copy.val_gpu[rank],*/
      &(C_copy.val)[(pA->starting_row_gpu[rank])],
      /*&(C_copy.val_gpu[i_gpu])[(pA->starting_row_gpu[i_gpu])*(pB->width)],*/
      /*C_copy.val_gpu[i_gpu] + 1,*/
      // pC->width) );
      pC->height));
  MPI_Barrier(MPI_COMM_WORLD);
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  gpu_timer nccl_timer;
  nccl_timer.start_timer();
  CHECK_NCCL(ncclAllReduce(C_copy.val, C_copy.val, C_copy.get_dim_gpu_num(rank),
                           ncclDouble, ncclSum, comm, 0));
  CUDA_SAFE_CALL(cudaThreadSynchronize());
#pragma omp barrier
  nccl_timer.stop_timer();
  cout << "GPU-" << i_gpu << " NCCL Time: " << nccl_timer.measure() << "ms."
       << endl;
  CHECK_CUSPARSE(cusparseDestroyMatDescr(mat_A));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_NCCL(ncclCommDestroy(comm[i_gpu]));
}
CUDA_CHECK_ERROR();
pC->plusDenseMatrixGPU(C_copy, alpha, beta);
}

#endif
