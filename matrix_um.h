// ------------------------------------------------------------------------
// File: matrix.h
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file define the sparse CSR, COO and CSC formats.
// It also defines the dense matrix and dense vector.
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#ifndef MATRIX_UM_H
#define MATRIX_UM_H

#include "kernel.h"
#include "mmio.h"
#include "mmio_highlevel.h"
#include "utility.h"
#include <assert.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <nvtx3/nvToolsExt.h>

using namespace std;

// Multi-GPU sparse data sharing policy:
//  - none: no gpu allocation, just on cpu
//  - replicate: duplicate the copy across all gpus
//  - segment: scatter across all gpus
enum GpuSharePolicy { none = 0, replicate = 1, segment = 2 };

// Dense data storage format:
enum MajorOrder { row_major = 0, col_major = 1 };

// Conversion from CSR format to CSC format on CPU
template <typename IdxType, typename DataType>
void CsrToCsc(const IdxType m, const IdxType n, const IdxType nnz,
              const IdxType *csrRowPtr, const IdxType *csrColIdx,
              const DataType *csrVal, IdxType *cscRowIdx, IdxType *cscColPtr,
              DataType *cscVal) {
  // histogram in column pointer
  memset(cscColPtr, 0, sizeof(IdxType) * (n + 1));
  for (IdxType i = 0; i < nnz; i++) {
    cscColPtr[csrColIdx[i]]++;
  }
  // prefix-sum scan to get the column pointer
  exclusive_scan<IdxType, IdxType>(cscColPtr, n + 1);
  IdxType *cscColIncr = (IdxType *)malloc(sizeof(IdxType) * (n + 1));
  memcpy(cscColIncr, cscColPtr, sizeof(IdxType) * (n + 1));
  // insert nnz to csc
  for (IdxType row = 0; row < m; row++) {
    for (IdxType j = csrRowPtr[row]; j < csrRowPtr[row + 1]; j++) {
      IdxType col = csrColIdx[j];
      cscRowIdx[cscColIncr[col]] = row;
      cscVal[cscColIncr[col]] = csrVal[j];
      cscColIncr[col]++;
    }
  }
  free(cscColIncr);
}

// Conversion from CSC format to CSR format on CPU
template <typename IdxType, typename DataType>
void CscToCsr(const IdxType n, const IdxType m, const IdxType nnz,
              const IdxType *cscColPtr, const IdxType *cscRowIdx,
              const DataType *cscVal, IdxType *csrColIdx, IdxType *csrRowPtr,
              DataType *csrVal) {
  // histogram in column pointer
  memset(csrRowPtr, 0, sizeof(IdxType) * (m + 1));
  for (IdxType i = 0; i < nnz; i++) {
    csrRowPtr[cscRowIdx[i]]++;
  }
  // prefix-sum scan to get the column pointer
  exclusive_scan<IdxType, IdxType>(csrRowPtr, m + 1);
  IdxType *csrRowIncr = (IdxType *)malloc(sizeof(IdxType) * (m + 1));
  memcpy(csrRowIncr, csrRowPtr, sizeof(IdxType) * (m + 1));
  // insert nnz to csr
  for (IdxType col = 0; col < n; col++) {
    for (IdxType j = cscColPtr[col]; j < cscColPtr[col + 1]; j++) {
      IdxType row = cscRowIdx[j];
      csrColIdx[csrRowIncr[row]] = col;
      csrVal[csrRowIncr[row]] = cscVal[j];
      csrRowIncr[row]++;
    }
  }
  free(csrRowIncr);
}

// ================================== COO Matrix
// ==================================== Matrix per data element
template <typename IdxType, typename DataType> struct CooElement {
  IdxType row;
  IdxType col;
  DataType val;
};
// Function for value comparision
template <typename IdxType, typename DataType>
int cmp_func(const void *aa, const void *bb) {
  struct CooElement<IdxType, DataType> *a =
      (struct CooElement<IdxType, DataType> *)aa;
  struct CooElement<IdxType, DataType> *b =
      (struct CooElement<IdxType, DataType> *)bb;
  if (a->row > b->row)
    return +1;
  if (a->row < b->row)
    return -1;
  if (a->col > b->col)
    return +1;
  if (a->col < b->col)
    return -1;
  return 0;
}
// COO matrix definition
template <typename IdxType, typename DataType> class CooSparseMatrix {
public:
  CooSparseMatrix() : nnz(0), height(0), width(0), n_gpu(0), policy(none) {
    this->cooRowIdx = NULL;
    this->cooColIdx = NULL;
    this->cooVal = NULL;
    this->cooRowIdx_gpu = NULL;
    this->cooColIdx_gpu = NULL;
    this->cooVal_gpu = NULL;
    this->nnz_gpu = NULL;
  }
  ~CooSparseMatrix() {
    if (n_gpu != 0 and policy != none) {
      if (policy == replicate) {
        SAFE_FREE_MULTI_MANAGED(cooRowIdx_gpu, n_gpu);
        SAFE_FREE_MULTI_MANAGED(cooColIdx_gpu, n_gpu);
        SAFE_FREE_MULTI_MANAGED(cooVal_gpu, n_gpu);
      } else {
        SAFE_FREE_MULTI_MANAGED(cooRowIdx_gpu, 1);
        SAFE_FREE_MULTI_MANAGED(cooColIdx_gpu, 1);
        SAFE_FREE_MULTI_MANAGED(cooVal_gpu, 1);
      }

      SAFE_FREE_HOST(nnz_gpu);
    } else {
      SAFE_FREE_GPU(this->cooRowIdx);
      SAFE_FREE_GPU(this->cooColIdx);
      SAFE_FREE_GPU(this->cooVal);
    }
  }
  CooSparseMatrix(const char *filename, unsigned n_gpu = 0,
                  enum GpuSharePolicy policy = none) {
    assert(n_gpu > 0);
    assert(policy != none);

    MM_typecode matcode;
    FILE *f;
    cout << "Loading input matrix from '" << filename << "'." << endl;
    if ((f = fopen(filename, "r")) == NULL) {
      cerr << "Error openning file " << filename << endl;
      exit(-1);
    }
    if (mm_read_banner(f, &matcode) != 0) {
      cerr << "Could not process Matrix Market banner." << endl;
      exit(-1);
    }
    int m, n, nz;
    if ((mm_read_mtx_crd_size(f, &m, &n, &nz)) != 0) {
      cerr << "Error reading matrix crd size." << endl;
      exit(-1);
    }
    this->height = (IdxType)m;
    this->width = (IdxType)n;
    this->nnz = (IdxType)nz;

    cout << "Height: " << height << " Width: " << width << " nnz: " << nnz
         << endl;

    // To handle the different GPUs pointers
    // For segmentation 1 malloc and GPU will contain the offsets to their data
    // For duplication, n_gpu malloc but work with gpu[0] array as if its the
    // CPU one

    cooRowIdx_gpu = new IdxType *[n_gpu];
    cooColIdx_gpu = new IdxType *[n_gpu];
    cooVal_gpu = new DataType *[n_gpu];

    SAFE_ALOC_MANAGED(this->cooRowIdx, get_nnz_idx_size());
    SAFE_ALOC_MANAGED(this->cooColIdx, get_nnz_idx_size());
    SAFE_ALOC_MANAGED(this->cooVal, get_nnz_val_size());

    for (unsigned i = 0; i < nnz; i++) {
      int row, col;
      double val;
      fscanf(f, "%d %d %lg\n", &row, &col, &val);
      this->cooRowIdx[i] = row - 1; // we're using coo, count from 0
      this->cooColIdx[i] = col - 1;
      this->cooVal[i] = val;
    }
    // Sorting to ensure COO format
    sortByRow();
    fclose(f);

    if (n_gpu != 0 and policy != none) {
      // this is to replicate arrays on each GPU
      if (policy == replicate) {
        for (unsigned i = 0; i < n_gpu; i++) {
          if (i > 0) {
            SAFE_ALOC_MANAGED(this->cooRowIdx_gpu[i], get_nnz_idx_size());
            SAFE_ALOC_MANAGED(this->cooColIdx_gpu[i], get_nnz_idx_size());
            SAFE_ALOC_MANAGED(this->cooVal_gpu[i], get_nnz_val_size());

            std::memcpy(cooRowIdx_gpu[i], cooRowIdx, get_nnz_idx_size());
            std::memcpy(cooColIdx_gpu[i], cooColIdx, get_nnz_idx_size());
            std::memcpy(cooVal_gpu[i], cooVal, get_nnz_val_size());
          } else {
            // No need to copy the same elements twice, GPU0 shares the same
            // pointer as the other
            cooRowIdx_gpu[i] = cooRowIdx;
            cooColIdx_gpu[i] = cooColIdx;
            cooVal_gpu[i] = cooVal;
          }
        }
      }

      if (policy == segment) {
        SAFE_ALOC_HOST(nnz_gpu, n_gpu * sizeof(IdxType))
        IdxType avg_nnz = ceil(nnz / n_gpu);
        for (unsigned i = 0; i < n_gpu; i++) {
          nnz_gpu[i] = min((i + 1) * avg_nnz, nnz) - i * avg_nnz;

          cooRowIdx_gpu[i] = &cooRowIdx[i * avg_nnz];
          cooColIdx_gpu[i] = &cooColIdx[i * avg_nnz];
          cooVal_gpu[i] = &cooVal[i * avg_nnz];
        }
      }
    }
  }

  void applyGpuTuning(bool readOnly = true) {
    cudaMemoryAdvise advise = (readOnly) ? cudaMemAdviseSetReadMostly
                                         : cudaMemAdviseSetPreferredLocation;

    if (this->policy == replicate) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(
            cudaMemAdvise(cooRowIdx_gpu[i], get_nnz_idx_size(), advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(cooColIdx_gpu[i], get_nnz_idx_size(), advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(cooVal_gpu[i], get_nnz_val_size(), advise, i));

        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(cooRowIdx_gpu[i], get_nnz_idx_size(), i));
        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(cooColIdx_gpu[i], get_nnz_idx_size(), i));
        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(cooVal_gpu[i], get_nnz_val_size(), i));
      }
    } else if (this->policy == segment) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(cooRowIdx_gpu[i], get_gpu_nnz_idx_size(i),
                                     advise, i));
        CUDA_SAFE_CALL(cudaMemAdvise(cooColIdx_gpu[i], get_gpu_nnz_idx_size(i),
                                     advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(cooVal_gpu[i], get_gpu_nnz_val_size(i), advise, i));

        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(cooRowIdx_gpu[i], get_gpu_nnz_idx_size(i), i));
        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(cooColIdx_gpu[i], get_gpu_nnz_idx_size(i), i));
        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(cooVal_gpu[i], get_gpu_nnz_val_size(i), i));
      }
    }
  };

  void removeGpuTuning(bool readOnly = true) {
    cudaMemoryAdvise advise = (readOnly) ? cudaMemAdviseUnsetReadMostly
                                         : cudaMemAdviseUnsetPreferredLocation;

    if (this->policy == replicate) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(
            cudaMemAdvise(cooRowIdx_gpu[i], get_nnz_idx_size(), advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(cooColIdx_gpu[i], get_nnz_idx_size(), advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(cooVal_gpu[i], get_nnz_val_size(), advise, i));

        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            cooRowIdx_gpu[i], get_nnz_idx_size(), cudaCpuDeviceId));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            cooColIdx_gpu[i], get_nnz_idx_size(), cudaCpuDeviceId));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(cooVal_gpu[i], get_nnz_val_size(),
                                            cudaCpuDeviceId));
      }
    } else if (this->policy == segment) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(cooRowIdx_gpu[i], get_gpu_nnz_idx_size(i),
                                     advise, i));
        CUDA_SAFE_CALL(cudaMemAdvise(cooColIdx_gpu[i], get_gpu_nnz_idx_size(i),
                                     advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(cooVal_gpu[i], get_gpu_nnz_val_size(i), advise, i));

        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            cooRowIdx_gpu[i], get_gpu_nnz_idx_size(i), cudaCpuDeviceId));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            cooColIdx_gpu[i], get_gpu_nnz_idx_size(i), cudaCpuDeviceId));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            cooVal_gpu[i], get_gpu_nnz_val_size(i), cudaCpuDeviceId));
      }
    }
  };

  void applyCpuTuning() {
    CUDA_SAFE_CALL(cudaMemAdvise(cooRowIdx, get_nnz_idx_size(),
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
    CUDA_SAFE_CALL(cudaMemAdvise(cooColIdx, get_nnz_idx_size(),
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
    CUDA_SAFE_CALL(cudaMemAdvise(cooVal, get_nnz_val_size(),
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
  };

  void sortByRow() {
    struct CooElement<IdxType, DataType> *coo_arr =
        new struct CooElement<IdxType, DataType>[nnz];
    unsigned size = sizeof(struct CooElement<IdxType, DataType>);
    for (unsigned i = 0; i < nnz; i++) {
      coo_arr[i].row = this->cooRowIdx[i];
      coo_arr[i].col = this->cooColIdx[i];
      coo_arr[i].val = this->cooVal[i];
    }
    qsort(coo_arr, nnz, size, cmp_func<IdxType, DataType>);
    for (unsigned i = 0; i < nnz; i++) {
      this->cooRowIdx[i] = coo_arr[i].row;
      this->cooColIdx[i] = coo_arr[i].col;
      this->cooVal[i] = coo_arr[i].val;
    }
    delete[] coo_arr;
  }
  size_t get_gpu_nnz_idx_size(unsigned i_gpu) {
    assert(i_gpu < n_gpu);
    if (nnz_gpu != NULL)
      return nnz_gpu[i_gpu] * sizeof(IdxType);
    else
      return 0;
  }
  size_t get_gpu_nnz_val_size(unsigned i_gpu) {
    assert(i_gpu < n_gpu);
    if (nnz_gpu != NULL)
      return nnz_gpu[i_gpu] * sizeof(DataType);
    else
      return 0;
  }
  size_t get_nnz_idx_size() { return nnz * sizeof(IdxType); }
  size_t get_nnz_val_size() { return nnz * sizeof(DataType); }

public:
  IdxType *cooRowIdx;
  IdxType *cooColIdx;
  DataType *cooVal;

  IdxType **cooRowIdx_gpu;
  IdxType **cooColIdx_gpu;
  DataType **cooVal_gpu;
  IdxType *nnz_gpu; // number of nnz per GPU

  IdxType nnz;
  IdxType height;
  IdxType width;
  unsigned n_gpu;
  enum GpuSharePolicy policy;
};

// ================================== CSR Matrix
// ====================================
template <typename IdxType, typename DataType> class CsrSparseMatrix {
public:
  CsrSparseMatrix() : nnz(0), height(0), width(0), n_gpu(0), policy(none) {
    this->csrRowPtr = NULL;
    this->csrColIdx = NULL;
    this->csrVal = NULL;
    this->csrRowPtr_gpu = NULL;
    this->csrColIdx_gpu = NULL;
    this->csrVal_gpu = NULL;
    this->nnz_gpu = NULL;
    this->starting_row_gpu = NULL;
    this->stoping_row_gpu = NULL;
  }
  ~CsrSparseMatrix() {

    if (policy == replicate) {
      SAFE_FREE_MULTI_MANAGED(csrRowPtr_gpu, n_gpu);
      SAFE_FREE_MULTI_MANAGED(csrColIdx_gpu, n_gpu);
      SAFE_FREE_MULTI_MANAGED(csrVal_gpu, n_gpu);
    } else if (policy == segment) {
      SAFE_FREE_GPU(csrRowPtr);
      SAFE_FREE_MULTI_MANAGED(csrRowPtr_gpu, n_gpu);
      SAFE_FREE_MULTI_MANAGED(csrColIdx_gpu, 1);
      SAFE_FREE_MULTI_MANAGED(csrVal_gpu, 1);
    } else {
      SAFE_FREE_GPU(csrRowPtr);
      SAFE_FREE_GPU(csrColIdx);
      SAFE_FREE_GPU(csrVal);
    }

    SAFE_FREE_HOST(nnz_gpu);
    SAFE_FREE_HOST(starting_row_gpu);
    SAFE_FREE_HOST(stoping_row_gpu);
  }
  CsrSparseMatrix(const char *filename) : policy(none), n_gpu(0) {
    int m = 0, n = 0, nnzA = 0, isSymmetricA;
    mmio_info(&m, &n, &nnzA, &isSymmetricA, filename);
    this->height = (IdxType)m;
    this->width = (IdxType)n;
    this->nnz = (IdxType)nnzA;

    SAFE_ALOC_MANAGED(this->csrRowPtr, get_row_ptr_size());
    SAFE_ALOC_MANAGED(this->csrColIdx, get_col_idx_size());
    SAFE_ALOC_MANAGED(this->csrVal, get_val_size());

    int *csrRowPtrA = (int *)malloc((m + 1) * sizeof(int));
    int *csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    double *csrValA = (double *)malloc(nnzA * sizeof(double));
    mmio_data(csrRowPtrA, csrColIdxA, csrValA, filename);
    printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);

    nvtxRangePush("Casting csrRow");
    for (int i = 0; i < m + 1; i++) {
      this->csrRowPtr[i] = (IdxType)csrRowPtrA[i];
    }
    nvtxRangePop();
    nvtxRangePush("Casting csrCol and csrVal");
    for (int i = 0; i < nnzA; i++) {
      this->csrColIdx[i] = (IdxType)csrColIdxA[i];
      this->csrVal[i] = (DataType)csrValA[i];
    }
    nvtxRangePop();

    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);

    this->csrRowPtr_gpu = NULL;
    this->csrColIdx_gpu = NULL;
    this->csrVal_gpu = NULL;
    this->nnz_gpu = NULL;
    this->starting_row_gpu = NULL;
    this->stoping_row_gpu = NULL;
  }
  void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy) {
    this->n_gpu = _n_gpu;
    this->policy = _policy;
    assert(this->n_gpu != 0);
    assert(this->policy != none);
    if (n_gpu != 0 and policy != none) {
      csrRowPtr_gpu = new IdxType *[n_gpu];
      csrColIdx_gpu = new IdxType *[n_gpu];
      csrVal_gpu = new DataType *[n_gpu];
      // this is to replicate arrays on each GPU
      if (policy == replicate) {
        for (unsigned i = 0; i < n_gpu; i++) {
          if (i == 0) {
            csrRowPtr_gpu[i] = csrRowPtr;
            csrColIdx_gpu[i] = csrColIdx;
            csrVal_gpu[i] = csrVal;
          } else {
            SAFE_ALOC_MANAGED(csrRowPtr_gpu[i], get_row_ptr_size());
            SAFE_ALOC_MANAGED(csrColIdx_gpu[i], get_col_idx_size());
            SAFE_ALOC_MANAGED(csrVal_gpu[i], get_val_size());

            std::memcpy(csrRowPtr_gpu[i], csrRowPtr, get_row_ptr_size());
            std::memcpy(csrColIdx_gpu[i], csrColIdx, get_col_idx_size());
            std::memcpy(csrVal_gpu[i], csrVal, get_val_size());
          }
        }
      } else if (policy == segment) {
        nvtxRangePush("Host Allocation");
        SAFE_ALOC_HOST(nnz_gpu, n_gpu * sizeof(IdxType));
        SAFE_ALOC_HOST(starting_row_gpu, n_gpu * sizeof(IdxType));
        SAFE_ALOC_HOST(stoping_row_gpu, n_gpu * sizeof(IdxType));
        nvtxRangePop();

        IdxType avg_nnz = ceil((float)nnz / n_gpu);

        for (unsigned i = 0; i < n_gpu; i++) {
          // CUDA_SAFE_CALL(cudaSetDevice(i));
          IdxType row_starting_nnz = i * avg_nnz;
          IdxType row_stoping_nnz = min((i + 1) * avg_nnz, nnz) - 1;

          nnz_gpu[i] = row_stoping_nnz - row_starting_nnz + 1;

          starting_row_gpu[i] =
              csr_findRowIdxUsingNnzIdx(csrRowPtr, height, row_starting_nnz);

          stoping_row_gpu[i] =
              csr_findRowIdxUsingNnzIdx(csrRowPtr, height, row_stoping_nnz);

          SAFE_ALOC_MANAGED(csrRowPtr_gpu[i], get_gpu_row_ptr_size(i));
          nvtxRangePush("Copying csrRowPtr");
          csrRowPtr_gpu[i][0] = 0;
          for (int k = 1; k < get_gpu_row_ptr_num(i) - 1; k++)
            csrRowPtr_gpu[i][k] =
                csrRowPtr[starting_row_gpu[i] + k] - i * avg_nnz;
          csrRowPtr_gpu[i][get_gpu_row_ptr_num(i) - 1] = nnz_gpu[i];
          nvtxRangePop();

          // Segments the data to the other GPUs
          nvtxRangePush("Segmenting csrColIdx and csrVal");
          csrColIdx_gpu[i] = &csrColIdx[i * avg_nnz];
          csrVal_gpu[i] = &csrVal[i * avg_nnz];
          nvtxRangePop();

          printf("gpu-%d,start-row:%d,stop-row:%d,num-rows:%ld,num-nnz:%d\n", i,
                 starting_row_gpu[i], stoping_row_gpu[i],
                 get_gpu_row_ptr_num(i), nnz_gpu[i]);
        }
      }
    }
  }

  void applyGpuTuning(bool readOnly = true) {
    cudaMemoryAdvise advise = (readOnly) ? cudaMemAdviseSetReadMostly
                                         : cudaMemAdviseSetPreferredLocation;

    if (this->policy == replicate) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(
            cudaMemAdvise(csrRowPtr_gpu[i], get_row_ptr_size(), advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(csrColIdx_gpu[i], get_col_idx_size(), advise, i));
        CUDA_SAFE_CALL(cudaMemAdvise(csrVal_gpu[i], get_val_size(), advise, i));

        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(csrRowPtr_gpu[i], get_row_ptr_size(), i));
        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(csrColIdx_gpu[i], get_col_idx_size(), i));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(csrVal_gpu[i], get_val_size(), i));
      }
    } else if (this->policy == segment) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(csrRowPtr_gpu[i], get_gpu_row_ptr_size(i),
                                     advise, i));
        CUDA_SAFE_CALL(cudaMemAdvise(csrColIdx_gpu[i], get_gpu_col_idx_size(i),
                                     advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(csrVal_gpu[i], get_gpu_nnz_val_size(i), advise, i));

        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(csrRowPtr_gpu[i], get_gpu_row_ptr_size(i), i));
        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(csrColIdx_gpu[i], get_gpu_col_idx_size(i), i));
        CUDA_SAFE_CALL(
            cudaMemPrefetchAsync(csrVal_gpu[i], get_gpu_nnz_val_size(i), i));
      }
    }
  };

  void removeGpuTuning(bool readOnly = true) {
    cudaMemoryAdvise advise = (readOnly) ? cudaMemAdviseUnsetReadMostly
                                         : cudaMemAdviseUnsetPreferredLocation;

    if (this->policy == replicate) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(
            cudaMemAdvise(csrRowPtr_gpu[i], get_row_ptr_size(), advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(csrColIdx_gpu[i], get_col_idx_size(), advise, i));
        CUDA_SAFE_CALL(cudaMemAdvise(csrVal_gpu[i], get_val_size(), advise, i));

        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            csrRowPtr_gpu[i], get_row_ptr_size(), cudaCpuDeviceId));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            csrColIdx_gpu[i], get_col_idx_size(), cudaCpuDeviceId));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(csrVal_gpu[i], get_val_size(),
                                            cudaCpuDeviceId));
      }
    } else if (this->policy == segment) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(csrRowPtr_gpu[i], get_gpu_row_ptr_size(i),
                                     advise, i));
        CUDA_SAFE_CALL(cudaMemAdvise(csrColIdx_gpu[i], get_gpu_col_idx_size(i),
                                     advise, i));
        CUDA_SAFE_CALL(
            cudaMemAdvise(csrVal_gpu[i], get_gpu_nnz_val_size(i), advise, i));

        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            csrRowPtr_gpu[i], get_gpu_row_ptr_size(i), cudaCpuDeviceId));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            csrColIdx_gpu[i], get_gpu_col_idx_size(i), cudaCpuDeviceId));
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            csrVal_gpu[i], get_gpu_nnz_val_size(i), cudaCpuDeviceId));
      }
    }
  };

  void applyCpuTuning() {
    CUDA_SAFE_CALL(cudaMemAdvise(csrRowPtr, get_row_ptr_size(),
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
    CUDA_SAFE_CALL(cudaMemAdvise(csrColIdx, get_col_idx_size(),
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
    CUDA_SAFE_CALL(cudaMemAdvise(csrVal, get_val_size(),
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
  };

  size_t get_gpu_row_ptr_num(unsigned i_gpu) {
    assert(i_gpu < n_gpu);
    if (nnz_gpu != NULL)
      return (stoping_row_gpu[i_gpu] - starting_row_gpu[i_gpu] + 2);
    else
      return 0;
  }
  size_t get_gpu_row_ptr_size(unsigned i_gpu) // how many rows on this gpu
  {
    return get_gpu_row_ptr_num(i_gpu) * sizeof(IdxType);
  }
  size_t get_gpu_col_idx_num(unsigned i_gpu) {
    assert(i_gpu < n_gpu);
    if (nnz_gpu != NULL)
      return nnz_gpu[i_gpu];
    else
      return 0;
  }
  size_t get_gpu_col_idx_size(unsigned i_gpu) {
    return get_gpu_col_idx_num(i_gpu) * sizeof(IdxType);
  }
  size_t get_gpu_nnz_val_num(unsigned i_gpu) {
    assert(i_gpu < n_gpu);
    if (nnz_gpu != NULL)
      return nnz_gpu[i_gpu];
    else
      return 0;
  }
  size_t get_gpu_nnz_val_size(unsigned i_gpu) {
    return get_gpu_nnz_val_num(i_gpu) * sizeof(DataType);
  }
  size_t get_row_ptr_size() { return (height + 1) * sizeof(IdxType); }
  size_t get_col_idx_size() { return nnz * sizeof(IdxType); }
  size_t get_val_size() { return nnz * sizeof(DataType); }

public:
  // Must be used to store all rows and from it segment everything else
  IdxType *csrRowPtr;
  IdxType *csrColIdx;
  DataType *csrVal;

  IdxType **csrRowPtr_gpu;
  IdxType **csrColIdx_gpu;
  DataType **csrVal_gpu;

  IdxType *nnz_gpu; // number of nnzs coverred by this GPU
  IdxType
      *starting_row_gpu; // since we partition on elements, it is possible a row
  IdxType *stoping_row_gpu; // is shared by two GPUs

  IdxType nnz;
  IdxType height;
  IdxType width;

  unsigned n_gpu;
  enum GpuSharePolicy policy;
};

// ================================== CSC Matrix
// ====================================
template <typename IdxType, typename DataType> class CscSparseMatrix {
public:
  CscSparseMatrix() : nnz(0), height(0), width(0) {
    this->cscColPtr = NULL;
    this->cscRowIdx = NULL;
    this->cscVal = NULL;
  }
  ~CscSparseMatrix() {
    SAFE_FREE_GPU(this->cscColPtr);
    SAFE_FREE_GPU(this->cscRowIdx);
    SAFE_FREE_GPU(this->cscVal);
  }
  CscSparseMatrix(const CsrSparseMatrix<IdxType, DataType> *csr) {
    this->height = csr->height;
    this->width = csr->width;
    this->nnz = csr->nnz;

    cout << "Building csc matrix from a csr matrix." << endl;
    cout << "Height: " << height << " Width: " << width << " nnz: " << nnz
         << endl;

    SAFE_ALOC_MANAGED(this->cscColPtr, get_col_ptr_size());
    SAFE_ALOC_MANAGED(this->cscRowIdx, get_row_idx_size());
    SAFE_ALOC_MANAGED(this->cscVal, get_val_size());

    CsrToCsc<IdxType, DataType>(height, width, nnz, csr->csrRowPtr,
                                csr->csrColIdx, csr->csrVal, this->cscRowIdx,
                                this->cscColPtr, this->cscVal);
  }

  size_t get_col_ptr_size() { return (width + 1) * sizeof(IdxType); }
  size_t get_row_idx_size() { return nnz * sizeof(IdxType); }
  size_t get_val_size() { return nnz * sizeof(DataType); }

public:
  IdxType *cscRowIdx;
  IdxType *cscColPtr;
  DataType *cscVal;
  IdxType nnz;
  IdxType height;
  IdxType width;
};

// =============================== Dense Matrix
// =================================
template <typename IdxType, typename DataType> class DenseMatrix {
public:
  DenseMatrix() : height(0), width(0), order(row_major), n_gpu(0) {
    val = NULL;
    val_gpu = NULL;
    dim_gpu = NULL;
    policy = none;
  }

  DenseMatrix(IdxType _height, IdxType _width, enum MajorOrder _order)
      : height(_height), width(_width), order(_order), n_gpu(0) {

    val_gpu = NULL;
    dim_gpu = NULL;
    SAFE_ALOC_MANAGED(val, get_mtx_size());
    srand(RAND_INIT_SEED);

    nvtxRangePush("Randomizing matrix");
    for (IdxType i = 0; i < get_mtx_num(); i++)
      val[i] = (DataType)rand0to1();
    nvtxRangePop();
    policy = none;
  }
  DenseMatrix(IdxType _height, IdxType _width, DataType _val,
              enum MajorOrder _order)
      : height(_height), width(_width), order(_order), n_gpu(0) {

    val_gpu = NULL;
    dim_gpu = NULL;
    SAFE_ALOC_MANAGED(val, get_mtx_size());

    srand(RAND_INIT_SEED);
    nvtxRangePush("Filling matrix");
    for (IdxType i = 0; i < get_mtx_num(); i++)
      val[i] = (DataType)_val;
    nvtxRangePop();
    policy = none;
  }

  void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy) {
    this->n_gpu = _n_gpu;
    this->policy = _policy;
    assert(this->n_gpu != 0);
    assert(this->policy != none);

    val_gpu = new DataType *[n_gpu];

    if (policy == replicate) {

      for (unsigned i = 0; i < n_gpu; i++) {
        nvtxRangePush("Replicating matrix to GPU");
        if (i == 0) {
          val_gpu[i] = val;
        } else {
          SAFE_ALOC_MANAGED(val_gpu[i], get_mtx_size());
          std::memcpy(val_gpu[i], val, get_mtx_size());
        }
        nvtxRangePop();
      }
    } else if (policy == segment) {
      SAFE_ALOC_HOST(dim_gpu, n_gpu * sizeof(IdxType));
      IdxType first_order = (order == row_major) ? height : width;
      IdxType second_order = (order == row_major) ? width : height;

      IdxType avg_val = ceil((double)first_order / n_gpu);
      for (unsigned i = 0; i < n_gpu; i++) {
        dim_gpu[i] = min((i + 1) * avg_val, first_order) - i * avg_val;
        val_gpu[i] = &val[(i * avg_val) * second_order];
      }
    }
  }

  void applyGpuTuning(bool readOnly = true) {
    cudaMemoryAdvise advise = (readOnly) ? cudaMemAdviseSetReadMostly
                                         : cudaMemAdviseSetPreferredLocation;

    if (this->policy == replicate) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(val_gpu[i], get_mtx_size(), advise, i));

        CUDA_SAFE_CALL(cudaMemPrefetchAsync(val_gpu[i], get_mtx_size(), i));
      }
    } else if (this->policy == segment) {
      IdxType second_order = (order == row_major) ? width : height;

      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(
            val_gpu[i], second_order * get_dim_gpu_size(i), advise, i));

        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            val_gpu[i], second_order * get_dim_gpu_size(i), i));
      }
    }
  }

  void removeGpuTuning(bool readOnly = true) {
    cudaMemoryAdvise advise = (readOnly) ? cudaMemAdviseUnsetReadMostly
                                         : cudaMemAdviseUnsetPreferredLocation;

    if (this->policy == replicate) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(val_gpu[i], get_mtx_size(), advise, i));
      }
    } else if (this->policy == segment) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(val_gpu[i],
                                     ((order == row_major) ? width : height) *
                                         get_dim_gpu_size(i),
                                     advise, i));
      }
    }
  }

  void applyCpuTuning() {
    CUDA_SAFE_CALL(cudaMemAdvise(val, get_mtx_size(),
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
  }

  ~DenseMatrix() {
    SAFE_FREE_HOST(dim_gpu);

    if (policy == replicate) {
      SAFE_FREE_MULTI_MANAGED(val_gpu, n_gpu);
    } else if (policy == segment) {
      SAFE_FREE_MULTI_MANAGED(val_gpu, 1);
    } else {
      SAFE_FREE_GPU(val);
    }
  }

  DenseMatrix *transpose() {
    assert(n_gpu == 0); // currently only allow transpose when no GPU copy
    assert(policy == none);
    DenseMatrix *trans_mtx = new DenseMatrix(
        height, width, (order == row_major ? col_major : row_major));

    if (order == row_major) {
      for (IdxType i = 0; i < height; i++)
        for (IdxType j = 0; j < width; j++)
          trans_mtx->val[j * height + i] = this->val[i * width + j];
    } else {
      for (IdxType i = 0; i < height; i++)
        for (IdxType j = 0; j < width; j++)
          trans_mtx->val[i * width + j] = this->val[j * height + i];
    }
    return trans_mtx;
  }

  void sync2cpu(unsigned i_gpu, bool tuning = false) {
    assert(val_gpu != NULL);
    assert(i_gpu < n_gpu);

    if (policy == segment) {
      IdxType first_order = (order == row_major) ? height : width;
      IdxType second_order = (order == row_major) ? width : height;
      IdxType avg_val = ceil((double)first_order / n_gpu);
      if (tuning) {
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(
            val_gpu[i_gpu], second_order * get_dim_gpu_size(i_gpu),
            cudaCpuDeviceId));
      }
    } else if (policy == replicate) {
      if (tuning) {
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(val_gpu[i_gpu], get_mtx_size(),
                                            cudaCpuDeviceId));
      }
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  void plusDenseMatrixGPU(DenseMatrix const &dm, DataType alpha,
                          DataType beta) {
    if (n_gpu != 0 && policy != none) {
      dim3 blockDim(NUM_THREADS_PER_BLK);
      dim3 gridDim(((get_mtx_num() - 1) / NUM_THREADS_PER_BLK) + 1);
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaSetDevice(i));
        denseVector_plusEqual_denseVector<<<gridDim, blockDim>>>(
            val_gpu[i], dm.val_gpu[i], alpha, beta, get_mtx_num());
      }
      CUDA_CHECK_ERROR();
    }
  }
  size_t get_dim_gpu_size(unsigned i_gpu) {
    return get_dim_gpu_num(i_gpu) * sizeof(DataType);
  }
  size_t get_dim_gpu_num(unsigned i_gpu) {
    assert(i_gpu < n_gpu);
    return (size_t)dim_gpu[i_gpu];
  }
  size_t get_row_size() { return width * sizeof(DataType); }
  size_t get_col_size() { return height * sizeof(DataType); }
  size_t get_mtx_size() { return width * height * sizeof(DataType); }
  size_t get_mtx_num() { return width * height; }

public:
  IdxType height;
  IdxType width;
  DataType *val;
  DataType **val_gpu;
  // num of rows or cols (leading dim) per gpu depending on row-major or
  // col-major
  IdxType *dim_gpu;
  unsigned n_gpu;
  enum GpuSharePolicy policy;
  enum MajorOrder order;
};

// =============================== Dense Vector
// =================================
template <typename IdxType, typename DataType> class DenseVector {
public:
  DenseVector() : length(0) {
    this->val = NULL;
    this->val_gpu = NULL;
  }
  DenseVector(IdxType _length) : length(_length), n_gpu(0), policy(none) {
    this->val_gpu = NULL;
    SAFE_ALOC_MANAGED(this->val, get_vec_size());
    srand(RAND_INIT_SEED);
    nvtxRangePush("Randomizing vector");
    for (IdxType i = 0; i < this->get_vec_length(); i++)
      (this->val)[i] = (DataType)rand0to1();
    nvtxRangePop();
  }
  DenseVector(IdxType _length, DataType _val)
      : length(_length), n_gpu(0), policy(none) {
    this->val_gpu = NULL;
    SAFE_ALOC_MANAGED(this->val, get_vec_size());
    srand(RAND_INIT_SEED);
    nvtxRangePush("Filling vector");
    for (IdxType i = 0; i < this->get_vec_length(); i++)
      (this->val)[i] = _val;
    nvtxRangePop();
  }
  DenseVector(const DenseVector &dv)
      : length(dv.length), n_gpu(dv.n_gpu), policy(dv.policy) {
    if (n_gpu != 0 && policy != none) {
      val_gpu = new DataType *[n_gpu];
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaSetDevice(i));
        SAFE_ALOC_MANAGED(val_gpu[i], get_vec_size());
        CUDA_SAFE_CALL(cudaMemcpy(val_gpu[i], dv.val_gpu[i], get_vec_size(),
                                  cudaMemcpyDeviceToDevice));
      }

      val = val_gpu[0];
    } else {
      SAFE_ALOC_MANAGED(val, get_vec_size());
      memcpy(val, dv.val, get_vec_size());
    }
  }
  void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy) {
    this->n_gpu = _n_gpu;
    this->policy = _policy;
    assert(this->n_gpu != 0);
    assert(this->policy != none);
    assert(this->policy != segment); // assume now vector does not need
                                     // partition
    if (policy == replicate) {
      val_gpu = new DataType *[n_gpu];

      for (unsigned i = 0; i < n_gpu; i++) {
        nvtxRangePush("Replicating vector to GPU");
        if (i == 0) {
          val_gpu[i] = val;
        } else {
          SAFE_ALOC_MANAGED(val_gpu[i], get_vec_size());
          std::memcpy(val_gpu[i], val, get_vec_size());
        }
        nvtxRangePop();
      }
    }

    // Remember to call GPU tuning and before CPU operation remove it
  }

  // Prefetch may benefit from different streams being used however, as the
  // original code used standard cudaMemcpy, using the async feature would be
  // cheating, or at least that's what I think
  void applyGpuTuning(bool readOnly = true) {
    cudaMemoryAdvise advise = (readOnly) ? cudaMemAdviseSetReadMostly
                                         : cudaMemAdviseSetPreferredLocation;

    if (this->policy == replicate) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(val_gpu[i], get_vec_size(), advise, i));

        CUDA_SAFE_CALL(cudaMemPrefetchAsync(val_gpu[i], get_vec_size(), i));
      }
    }
  }

  void removeGpuTuning(bool readOnly = true) {
    cudaMemoryAdvise advise = (readOnly) ? cudaMemAdviseUnsetReadMostly
                                         : cudaMemAdviseUnsetPreferredLocation;

    if (this->policy == replicate) {
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaMemAdvise(val_gpu[i], get_vec_size(), advise, i));
      }
    }
  }

  void applyCpuTuning() {
    CUDA_SAFE_CALL(cudaMemAdvise(val, get_vec_size(),
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
  }

  void
  sync2cpu(unsigned i_gpu,
           bool tuning =
               false) // all gpus have the same res vector, pick from any one
  {
    assert(i_gpu < n_gpu);
    assert(val_gpu != NULL);

    if (tuning) {
      CUDA_SAFE_CALL(cudaMemPrefetchAsync(val_gpu[i_gpu], get_vec_size(),
                                          cudaCpuDeviceId));
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  void plusDenseVectorGPU(DenseVector const &dv, DataType alpha,
                          DataType beta) {
    if (n_gpu != 0 && policy != none) {
      dim3 blockDim(NUM_THREADS_PER_BLK);
      dim3 gridDim(((get_vec_length() - 1) / NUM_THREADS_PER_BLK) + 1);
      for (unsigned i = 0; i < n_gpu; i++) {
        CUDA_SAFE_CALL(cudaSetDevice(i));
        denseVector_plusEqual_denseVector<<<gridDim, blockDim>>>(
            val_gpu[i], dv.val_gpu[i], alpha, beta, get_vec_length());
      }
      CUDA_CHECK_ERROR();
    }
  }
  ~DenseVector() {
    if (n_gpu > 0) {
      SAFE_FREE_MULTI_MANAGED(val_gpu, n_gpu);
    } else {
      SAFE_FREE_GPU(this->val);
    }
  }
  size_t get_vec_size() { return (size_t)length * sizeof(DataType); }
  size_t get_vec_length() { return (size_t)length; }

public:
  IdxType length;
  DataType *val;
  DataType **val_gpu;
  unsigned n_gpu;
  enum GpuSharePolicy policy;
};

#endif
