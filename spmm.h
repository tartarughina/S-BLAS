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

#include "matrix.h"
#include "utility.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <nccl.h>
#include <omp.h>

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
    std::cerr << "SBLAS_SPMM_CSR_V1: B should be in column major!" << std::endl;
    exit(-1);
  }
  if (pC->order == row_major) {
    std::cerr << "SBLAS_SPMM_CSR_V1: C should be in column major!" << std::endl;
    exit(-1);
  }
  cout << "sblas_spmm_csr_v1 ready to start" << endl;
  // Start OpenMP
#pragma omp parallel num_threads(n_gpu)
  {
    int i_gpu = omp_get_thread_num();
    CUDA_SAFE_CALL(cudaSetDevice(i_gpu));
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Determine data types
    cudaDataType valueType = getCudaDataType<DataType>();
    cusparseIndexType_t indexType = getCusparseIndexType<IdxType>();

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, static_cast<int64_t>(pA->height),
        static_cast<int64_t>(pA->width), static_cast<int64_t>(pA->nnz),
        pA->csrRowPtr_gpu[i_gpu], pA->csrColIdx_gpu[i_gpu],
        pA->csrVal_gpu[i_gpu], indexType, indexType, CUSPARSE_INDEX_BASE_ZERO,
        valueType));

    cusparseDnMatDescr_t matB;
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&matB, static_cast<int64_t>(pA->width),
                            static_cast<int64_t>(pB->get_dim_gpu_num(i_gpu)),
                            static_cast<int64_t>(pA->width), pB->val_gpu[i_gpu],
                            valueType, CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t matC;
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&matC, static_cast<int64_t>(pA->height),
                            static_cast<int64_t>(pB->get_dim_gpu_num(i_gpu)),
                            static_cast<int64_t>(pA->height),
                            pC->val_gpu[i_gpu], valueType, CUSPARSE_ORDER_COL));

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        valueType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    void *externalBuffer = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&externalBuffer, bufferSize));

    printf("gpu-%d m:%d, n:%ld, k:%d\n", i_gpu, pA->height,
           pB->get_dim_gpu_num(i_gpu), pA->width);

    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                matB, &beta, matC, valueType,
                                CUSPARSE_SPMM_ALG_DEFAULT, externalBuffer));

    pC->sync2cpu(i_gpu);

#pragma omp barrier

    cudaFree(externalBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
  } // end of omp
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
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  ncclComm_t comm[n_gpu];
  DenseMatrix<IdxType, DataType> C_copy(pC->height, pC->width, 0., row_major);
  C_copy.sync2gpu(n_gpu, replicate);
// Start OpenMP
#pragma omp parallel num_threads(n_gpu) shared(comm, id)
  {
    int i_gpu = omp_get_thread_num();
    CUDA_SAFE_CALL(cudaSetDevice(i_gpu));
    CHECK_NCCL(ncclCommInitRank(&comm[i_gpu], n_gpu, id, i_gpu));

    // Create cuSPARSE handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Determine the data types
    cudaDataType valueType = getCudaDataType<DataType>();
    cusparseIndexType_t indexType = getCusparseIndexType<IdxType>();

    // Create matrix descriptors
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    int64_t rowsA = pA->get_gpu_row_ptr_num(i_gpu) - 1; // m
    int64_t colsA = pA->width;                          // k
    int64_t nnzA = pA->nnz_gpu[i_gpu];

    void *csrRowOffsets = pA->csrRowPtr_gpu[i_gpu];
    void *csrColInd = pA->csrColIdx_gpu[i_gpu];
    void *csrValues = pA->csrVal_gpu[i_gpu];

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, csrRowOffsets,
                                     csrColInd, csrValues, indexType, indexType,
                                     CUSPARSE_INDEX_BASE_ZERO, valueType));

    int64_t rowsB = pB->height; // k
    int64_t colsB = pB->width;  // n
    int64_t ldb = pB->height;   // leading dimension
    void *valuesB = pB->val_gpu[i_gpu];
    cusparseOrder_t orderB = CUSPARSE_ORDER_COL;

    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, rowsB, colsB, ldb, valuesB,
                                       valueType, orderB));

    int64_t rowsC = rowsA;    // m
    int64_t colsC = colsB;    // n
    int64_t ldc = pC->height; // leading dimension
    void *valuesC = &(C_copy.val_gpu[i_gpu])[(pA->starting_row_gpu[i_gpu])];
    cusparseOrder_t orderC = CUSPARSE_ORDER_COL;

    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, rowsC, colsC, ldc, valuesC,
                                       valueType, orderC));

    DataType dummy_alpha = 1.0;
    DataType dummy_beta = 1.0;

    // Allocate buffer
    size_t bufferSize = 0;
    void *dBuffer = NULL;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &dummy_alpha, matA, matB, &dummy_beta,
        matC, valueType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    CUDA_SAFE_CALL(cudaMalloc(&dBuffer, bufferSize));

    // Perform SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &dummy_alpha,
                                matA, matB, &dummy_beta, matC, valueType,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

#pragma omp barrier
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#pragma omp barrier

    // NCCL AllReduce to sum partial results
    gpu_timer nccl_timer;
    nccl_timer.start_timer();
    CHECK_NCCL(ncclAllReduce(C_copy.val_gpu[i_gpu], C_copy.val_gpu[i_gpu],
                             C_copy.get_mtx_num(), ncclDouble, ncclSum,
                             comm[i_gpu], 0));
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#pragma omp barrier
    nccl_timer.stop_timer();

#pragma omp critical
    {
      cout << "GPU-" << i_gpu << " NCCL Time: " << nccl_timer.measure()
           << " ms." << std::endl;
    }

    // Clean up
    cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_NCCL(ncclCommDestroy(comm[i_gpu]));
  }

  CUDA_CHECK_ERROR();
  pC->plusDenseMatrixGPU(C_copy, alpha, beta);
}
#endif
