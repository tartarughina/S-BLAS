#ifndef SPMV_H
#define SPMV_H

#include <assert.h>
#include <cusparse.h>
#include <iostream>
#include <nccl.h>
#include <omp.h>
#include <stdio.h>

#include <mpi.h>

#include "matrix.h"
#include "utility.h"

using namespace std;
//========================== CPU baseline version ============================
template <typename IdxType, typename DataType>
void sblas_spmv_csr_cpu(CsrSparseMatrix<IdxType, DataType> *pA,
                        DenseVector<IdxType, DataType> *pB,
                        DenseVector<IdxType, DataType> *pC, DataType alpha,
                        DataType beta) {
  assert((pA->width) == (pB->length));
  assert((pB->length) == (pC->length));
  for (IdxType i = 0; i < pA->height; i++) {
    DataType sum = 0;
    for (IdxType j = pA->csrRowPtr[i]; j < pA->csrRowPtr[i + 1]; j++) {
      IdxType col_A = pA->csrColIdx[j];
      DataType val_A = pA->csrVal[j];
      DataType val_B = pB->val[col_A];
      sum += val_A * val_B;
    }
    pC->val[i] = beta * (pC->val[i]) + alpha * sum;
  }
}

//========================== GPU V1 ============================
template <typename IdxType, typename DataType>
void sblas_spmv_csr_v1(CsrSparseMatrix<IdxType, DataType> *pA,
                       DenseVector<IdxType, DataType> *pB,
                       DenseVector<IdxType, DataType> *pC, DataType alpha,
                       DataType beta, unsigned n_gpu) {
  assert((pA->width == pB->length));
  assert((pA->height) == (pC->length));

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ncclUniqueId id;

  if (rank == 0) {
    ncclGetUniqueId(&id);
  }

  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;

  DenseVector<IdxType, DataType> C_copy(pC->get_vec_length(), 0.);
  C_copy.sync2gpu(n_gpu, replicate);

  cpu_timer nccl_timer;
  nccl_timer.start_timer();

  // CUDA_SAFE_CALL(cudaSetDevice(i_gpu));

  CHECK_NCCL(ncclCommInitRank(&comm, size, id, rank));

  cusparseHandle_t handle;
  cusparseMatDescr_t mat_A;
  cusparseStatus_t cusparse_status;

  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&mat_A));
  CHECK_CUSPARSE(cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO));

  nccl_timer.stop_timer();
  cout << "GPU-" << i_gpu << " NCCL Time: " << nccl_timer.measure() << "ms."
       << endl;

  DataType dummy_alpha = 1.0;
  DataType dummy_beta = 1.0;

  CHECK_CUSPARSE(cusparseDcsrmv(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      pA->get_gpu_row_ptr_num(rank) - 1, pA->width, pA->nnz_gpu[rank],
      &dummy_alpha, mat_A, pA->csrVal, pA->csrRowPtr, pA->csrColIdx, pB->val,
      &dummy_beta, &(C_copy.val)[pA->starting_row_gpu[rank]]));

  // we can add a shift to result matrix to recover its correct row, then
  // perform all reduce, since NCCL allows inplace all-reduce, we won't need
  // extra buffer.

  CHECK_CUSPARSE(cusparseDestroyMatDescr(mat_A));
  CHECK_CUSPARSE(cusparseDestroy(handle));

  CHECK_NCCL(ncclAllReduce(C_copy.val, C_copy.val, C_copy.get_vec_length(),
                           ncclDouble, ncclSum, comm, 0));

  CHECK_NCCL(ncclCommDestroy(comm));

  /*CUDA_CHECK_ERROR();*/

  pC->plusDenseVectorGPU(C_copy, alpha, beta);
}

#endif
