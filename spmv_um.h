#ifndef SPMV_UM_H
#define SPMV_UM_H

#include "matrix_um.h"
#include "utility.h"
#include <assert.h>
#include <cusparse.h>
#include <iostream>
#include <nccl.h>
#include <omp.h>
#include <stdio.h>

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
                       DataType beta, unsigned n_gpu, bool tuning = false) {
  assert((pA->width == pB->length));
  assert((pA->height) == (pC->length));

  ncclUniqueId id;
  ncclGetUniqueId(&id);
  ncclComm_t comm[n_gpu];

  // Create a copy of C for GPU computations
  DenseVector<IdxType, DataType> C_copy(pC->get_vec_length(), 0.);
  C_copy.sync2gpu(n_gpu, replicate);

  if (tuning) {
    C_copy.applyGpuTuning(false);
  }
  // Start OpenMP parallel region
#pragma omp parallel num_threads(n_gpu) shared(comm, id)
  {
    int i_gpu = omp_get_thread_num();
    CUDA_SAFE_CALL(cudaSetDevice(i_gpu));

    // Initialize NCCL communicator
    CHECK_NCCL(ncclCommInitRank(&comm[i_gpu], n_gpu, id, i_gpu));

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Determine data types
    cudaDataType valueType = getCudaDataType<DataType>();

    cusparseIndexType_t indexType = getCusparseIndexType<IdxType>();

    // Create sparse matrix descriptor for A
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        static_cast<int64_t>(pA->get_gpu_row_ptr_num(i_gpu) - 1), // rows
        static_cast<int64_t>(pA->width),                          // cols
        static_cast<int64_t>(pA->nnz_gpu[i_gpu]),                 // nnz
        pA->csrRowPtr_gpu[i_gpu], pA->csrColIdx_gpu[i_gpu],
        pA->csrVal_gpu[i_gpu], indexType, indexType, CUSPARSE_INDEX_BASE_ZERO,
        valueType));

    // Create dense vector descriptors for B and C_copy
    cusparseDnVecDescr_t vecB;
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vecB, static_cast<int64_t>(pB->get_vec_length()),
                            pB->val_gpu[i_gpu], valueType));

    cusparseDnVecDescr_t vecC;
    CHECK_CUSPARSE(cusparseCreateDnVec(
        &vecC, static_cast<int64_t>(pA->get_gpu_row_ptr_num(i_gpu) - 1),
        &(C_copy.val_gpu[i_gpu])[pA->starting_row_gpu[i_gpu]], valueType));

    // Set alpha and beta
    DataType dummy_alpha = 1.0;
    DataType dummy_beta = 1.0;

    // Compute buffer size and allocate buffer
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &dummy_alpha, matA, vecB,
        &dummy_beta, vecC, valueType, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    void *dBuffer = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&dBuffer, bufferSize));

    // Perform SpMV operation
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &dummy_alpha, matA, vecB, &dummy_beta, vecC,
                                valueType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

#pragma omp barrier
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#pragma omp barrier

    gpu_timer nccl_timer;
    nccl_timer.start_timer();
    // Perform NCCL AllReduce to sum results across GPUs
    CHECK_NCCL(ncclAllReduce(C_copy.val_gpu[i_gpu], C_copy.val_gpu[i_gpu],
                             C_copy.get_vec_length(),
                             (valueType == CUDA_R_64F) ? ncclDouble : ncclFloat,
                             ncclSum, comm[i_gpu], 0));
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#pragma omp barrier
    nccl_timer.stop_timer();

#pragma omp critical
    {
      cout << "GPU-" << i_gpu << " NCCL Time: " << nccl_timer.measure()
           << " ms." << std::endl;
    }

    cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_NCCL(ncclCommDestroy(comm[i_gpu]));
  }

  CUDA_CHECK_ERROR();
  pC->plusDenseVectorGPU(C_copy, alpha, beta);
}

#endif
