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

#ifndef MATRIX_H
#define MATRIX_H

#include "kernel.h"
#include "mmio.h"
#include "mmio_highlevel.h"
#include "utility.h"
#include <assert.h>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// This policies are going to be used only as a guideline when implementing the
// MPI version of the library Multi-GPU sparse data sharing policy:
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

template <typename T> MPI_Datatype get_mpi_type<T>() {
  switch (sizeof(T)) {
  case 1:
    return MPI_BYTE; // Assuming you might have a single byte type
  case 2:
    return MPI_SHORT; // For 2-byte integers
  case 4:
    return MPI_INT; // Common for 4-byte integers
  case 8:
    return MPI_LONG_LONG; // Common for 8-byte integers (or MPI_LONG on some
                          // platforms)
  default:
    // Handle other cases or throw an error
    throw std::runtime_error("Unsupported data type size for MPI.");
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
    CooSparseMatrix()
        : nnz(0), height(0), width(0), n_gpu(0), rank(-1), policy(none) {
      this->cooRowIdx = NULL;
      this->cooColIdx = NULL;
      this->cooVal = NULL;
      this->nnz_gpu = NULL;
    }
    ~CooSparseMatrix() {
      SAFE_FREE_GPU(this->cooRowIdx);
      SAFE_FREE_GPU(this->cooColIdx);
      SAFE_FREE_GPU(this->cooVal);
      SAFE_FREE_GPU(this->nnz_gpu);
    }
    CooSparseMatrix(const char *filename, unsigned n_gpu = 0,
                    enum GpuSharePolicy policy = none) {
      /* Obtain the device number as in those examples from nccl where the
       * localhost is used to determine which device */
      // the whole matrix is going to be read by rank 0
      MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
      MPI_Comm_size(MPI_COMM_WORLD, &this->n_gpu);
      if (rank == 0) {
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
        SAFE_ALOC_UM(this->cooRowIdx, get_nnz_idx_size());
        SAFE_ALOC_UM(this->cooColIdx, get_nnz_idx_size());
        SAFE_ALOC_UM(this->cooVal, get_nnz_val_size());

        for (unsigned i = 0; i < nnz; i++) {
          int row, col;
          double val;
          fscanf(f, "%d %d %lg\n", &row, &col, &val);
          cooRowIdx[i] = row - 1; // we're using coo, count from 0
          cooColIdx[i] = col - 1;
          cooVal[i] = val;
        }
        // Sorting to ensure COO format
        sortByRow();
        fclose(f);
      }
      /* READ MATRIX END */
      MPI_Barrier(MPI_COMM_WORLD);

      MPI_Bcast(&this->height, 1, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
      MPI_Bcast(&this->width, 1, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
      MPI_Bcast(&this->nnz, 1, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
      // MPI_Bcast(&this->policy, 1, MPI_INT, 0, MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);

      // The number of GPUs must be 1 for each process
      // TODO: Retrieve the right device number based on the localhost name as
      // done in nccl examples
      if (n_gpu != 0 and policy != none) {
        if (policy == replicate) {
          if (rank > 0) {
            SAFE_ALOC_UM(this->cooRowIdx, get_nnz_idx_size());
            SAFE_ALOC_UM(this->cooColIdx, get_nnz_idx_size());
            SAFE_ALOC_UM(this->cooVal, get_nnz_val_size());
          }

          MPI_Bcast(&this->cooColIdx, this->nnz, get_mpi_type<IdxType>(), 0,
                    MPI_COMM_WORLD);
          MPI_Bcast(&this->cooRowIdx, this->nnz, get_mpi_type<IdxType>(), 0,
                    MPI_COMM_WORLD);
          MPI_Bcast(&this->cooVal, this->nnz, get_mpi_type<DataType>(), 0,
                    MPI_COMM_WORLD);
        }

        if (policy == segment) {
          /* Every rank should allocate this array */
          SAFE_ALOC_UM(nnz_gpu, n_gpu * sizeof(IdxType));

          /* The root defines the nnz_gpu array */
          if (rank == 0) {
            displacements = (IdxType *)malloc(n_gpu * sizeof(IdxType));
            IdxType sum = 0;
            IdxType avg_nnz = ceil(nnz / n_gpu);
            for (unsigned i = 0; i < n_gpu; i++) {
              nnz_gpu[i] = min((i + 1) * avg_nnz, nnz) - i * avg_nnz;
              displacements[i] = sum;
              sum += nnz_gpu[i];
            }
          }

          MPI_Bcast(nnz_gpu, n_gpu, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);

          MPI_Barrier(MPI_COMM_WORLD);

          /* Every rank, except root, should allocate this array */
          if (rank > 0) {
            SAFE_ALOC_UM(cooColIdx, nnz_gpu[rank] * sizeof(IdxType));
            SAFE_ALOC_UM(cooRowIdx, nnz_gpu[rank] * sizeof(IdxType));
            SAFE_ALOC_UM(cooVal, nnz_gpu[rank] * sizeof(DataType));
          }

          /* The root scatters the data to the other ranks */
          MPI_Scatterv(cooColIdx, nnz_gpu, displacements,
                       get_mpi_type<IdxType>(), cooColIdx, nnz_gpu[rank],
                       get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
          MPI_Scatterv(cooRowIdx, nnz_gpu, displacements,
                       get_mpi_type<IdxType>(), cooRowIdx, nnz_gpu[rank],
                       get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
          MPI_Scatterv(cooVal, nnz_gpu, displacements, get_mpi_type<DataType>(),
                       cooVal, nnz_gpu[rank], get_mpi_type<DataType>(), 0,
                       MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
    void sortByRow() {
      struct CooElement<IdxType, DataType> *coo_arr =
          new struct CooElement<IdxType, DataType>[nnz];
      unsigned size = sizeof(struct CooElement<IdxType, DataType>);
      for (unsigned i = 0; i < nnz; i++) {
        coo_arr[i].row = cooRowIdx[i];
        coo_arr[i].col = cooColIdx[i];
        coo_arr[i].val = cooVal[i];
      }
      qsort(coo_arr, nnz, size, cmp_func<IdxType, DataType>);
      for (unsigned i = 0; i < nnz; i++) {
        cooRowIdx[i] = coo_arr[i].row;
        cooColIdx[i] = coo_arr[i].col;
        cooVal[i] = coo_arr[i].val;
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
  }

  public : IdxType *cooRowIdx;
  IdxType *cooColIdx;
  DataType *cooVal;
  IdxType *nnz_gpu; // number of nnz per GPU
  IdxType *displacements;

  IdxType nnz;
  IdxType height;
  IdxType width;

  int rank;       // MPI rank within the world
  unsigned n_gpu; // equivalent to to world size
  int device;     // The GPU device number assigned to this rank

  enum GpuSharePolicy policy;
};

// ================================== CSR Matrix
// ====================================
template <typename IdxType, typename DataType> class CsrSparseMatrix {
public:
  CsrSparseMatrix()
      : nnz(0), height(0), width(0), n_gpu(0), rank(-1), policy(none) {
    this->csrRowPtr = NULL;
    this->csrColIdx = NULL;
    this->csrVal = NULL;
    this->nnz_gpu = NULL;
    this->starting_row_gpu = NULL;
    this->stoping_row_gpu = NULL;
  }
  ~CsrSparseMatrix() {
    SAFE_FREE_GPU(csrRowPtr);
    SAFE_FREE_GPU(csrColIdx);
    SAFE_FREE_GPU(csrVal);
    SAFE_FREE_GPU(nnz_gpu);
    SAFE_FREE_GPU(starting_row_gpu);
    SAFE_FREE_GPU(stoping_row_gpu);
  }
  CsrSparseMatrix(const char *filename) : policy(none), n_gpu(0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->n_gpu);

    if (rank == 0) {
      int m = 0, n = 0, nnzA = 0, isSymmetricA;
      mmio_info(&m, &n, &nnzA, &isSymmetricA, filename);
      this->height = (IdxType)m;
      this->width = (IdxType)n;
      this->nnz = (IdxType)nnzA;

      SAFE_ALOC_UM(this->csrRowPtr, get_row_ptr_size());
      SAFE_ALOC_UM(this->csrColIdx, get_col_idx_size());
      SAFE_ALOC_UM(this->csrVal, get_val_size());

      int *csrRowPtrA = (int *)malloc((m + 1) * sizeof(int));
      int *csrColIdxA = (int *)malloc(nnzA * sizeof(int));
      double *csrValA = (double *)malloc(nnzA * sizeof(double));
      // For the input phase the type must be enforced
      mmio_data(csrRowPtrA, csrColIdxA, csrValA, filename);
      printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);

      // Once the read is complete a cast to the actual type is done
      for (int i = 0; i < m + 1; i++) {
        this->csrRowPtr[i] = (IdxType)csrRowPtrA[i];
      }
      for (int i = 0; i < nnzA; i++) {
        this->csrColIdx[i] = (IdxType)csrColIdxA[i];
        this->csrVal[i] = (DataType)csrValA[i];
      }
      free(csrRowPtrA);
      free(csrColIdxA);
      free(csrValA);
    }
    // READ MATRIX END
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&this->height, 1, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&this->width, 1, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&this->nnz, 1, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
    // MPI_Bcast(&this->policy, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    this->nnz_gpu = NULL;
    this->starting_row_gpu = NULL;
    this->stoping_row_gpu = NULL;
  }

  /* The number of gpus is equal to the world size and the policy is hardcoded
   * in the tests */
  void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy) {
    this->n_gpu = _n_gpu;
    this->policy = _policy;
    assert(this->n_gpu != 0);
    assert(this->policy != none);
    if (n_gpu != 0 and policy != none) {
      // this is to replicate arrays on each GPU
      if (policy == replicate) {
        if (rank > 0) {
          SAFE_ALOC_UM(this->csrRowIdx, get_nnz_idx_size());
          SAFE_ALOC_UM(this->csrColIdx, get_nnz_idx_size());
          SAFE_ALOC_UM(this->csrVal, get_nnz_val_size());
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&this->csrColIdx, this->nnz, get_mpi_type<IdxType>(), 0,
                  MPI_COMM_WORLD);
        MPI_Bcast(&this->csrRowIdx, this->nnz, get_mpi_type<IdxType>(), 0,
                  MPI_COMM_WORLD);
        MPI_Bcast(&this->csrVal, this->nnz, get_mpi_type<DataType>(), 0,
                  MPI_COMM_WORLD);
      } else if (policy == segment) {
        SAFE_ALOC_UM(nnz_gpu, n_gpu * sizeof(IdxType));
        SAFE_ALOC_UM(starting_row_gpu, n_gpu * sizeof(IdxType));
        SAFE_ALOC_UM(stoping_row_gpu, n_gpu * sizeof(IdxType));

        /* Root defines the elements to be then shared among ranks */
        if (rank == 0) {
          displacements = (IdxType *)malloc(n_gpu * sizeof(IdxType));
          IdxType avg_nnz = ceil((float)nnz / n_gpu);
          int sum = 0;
          for (unsigned i = 0; i < n_gpu; i++) {
            IdxType row_starting_nnz = i * avg_nnz;
            IdxType row_stoping_nnz = min((i + 1) * avg_nnz, nnz) - 1;
            nnz_gpu[i] = row_stoping_nnz - row_starting_nnz + 1;
            starting_row_gpu[i] =
                csr_findRowIdxUsingNnzIdx(csrRowPtr, height, row_starting_nnz);
            stoping_row_gpu[i] =
                csr_findRowIdxUsingNnzIdx(csrRowPtr, height, row_stoping_nnz);
            displacements[i] = sum;
            sum += nnz_gpu[i];
          }

          MPI_Barrier(MPI_COMM_WORLD);

          MPI_Bcast(nnz_gpu, n_gpu, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);
          MPI_Bcast(starting_row_gpu, n_gpu, get_mpi_type<IdxType>(), 0,
                    MPI_COMM_WORLD);
          MPI_Bcast(stoping_row_gpu, n_gpu, get_mpi_type<IdxType>(), 0,
                    MPI_COMM_WORLD);

          if (rank > 0) {
            SAFE_ALOC_UM(this->csrRowIdx, get_gpu_row_ptr_size(rank));
            SAFE_ALOC_UM(this->csrColIdx, get_gpu_col_idx_size(rank));
            SAFE_ALOC_UM(this->csrVal, get_gpu_nnz_val_size(rank));
          }

          MPI_Barrier(MPI_COMM_WORLD);

          MPI_Scatterv(csrColIdx, nnz_gpu, displacements,
                       get_mpi_type<IdxType>(), this->csrColIdx,
                       get_gpu_col_idx_num(rank), get_mpi_type<IdxType>(), 0,
                       MPI_COMM_WORLD);
          MPI_Scatterv(csrVal, nnz_gpu, displacements, get_mpi_type<DataType>(),
                       this->csrVal, get_gpu_nnz_val_num(rank),
                       get_mpi_type<DataType>(), 0, MPI_COMM_WORLD);

          if (rank == 0) {
            for (i = 0; i < n_gpu; i++) {
              IdxType *fixedRowPtr = NULL;
              SAFE_ALOC_UM(fixedRowPtr, get_gpu_row_ptr_size(i));

              fixedRowPtr[0] = 0;
              for (int k = 1; k < get_gpu_row_ptr_num(i) - 1; k++)
                fixedRowPtr[k] =
                    csrRowPtr[starting_row_gpu[i] + k] - i * avg_nnz;
              fixedRowPtr[get_gpu_row_ptr_num(i) - 1] = nnz_gpu[i];

              /* Send and Recv combo to send each part of the matrix to the
               * appropriate rank */
              if (i > 0) {
                MPI_Send(fixedRowPtr, get_gpu_row_ptr_num(i),
                         get_mpi_type<IdxType>(), i, 0, MPI_COMM_WORLD);
              } else {
                memcpy(csrRowPtr, fixedRowPtr, get_gpu_row_ptr_size(0));
              }

              SAFE_FREE_GPU(fixedRowPtr);
            }
          } else {
            MPI_Recv(csrRowPtr, get_gpu_row_ptr_num(rank),
                     get_mpi_type<IdxType>(), 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
          }
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
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
  IdxType *csrRowPtr;
  IdxType *csrColIdx;
  DataType *csrVal;

  IdxType *displacements;
  IdxType *nnz_gpu; // number of nnzs coverred by this GPU
  IdxType
      *starting_row_gpu; // since we partition on elements, it is possible a row
  IdxType *stoping_row_gpu; // is shared by two GPUs

  IdxType nnz;
  IdxType height;
  IdxType width;

  int rank;       // current rank
  unsigned n_gpu; // number of GPUs|ranks
  int device;     // current gpu id

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
    /* All the data segmentation or duplication has already happened in the Csr
     * definition */
    this->height = csr->height;
    this->width = csr->width;
    this->nnz = csr->nnz;
    this->device = csr->device;
    this->rank = csr->rank;
    this->n_gpu = csr->n_gpu;
    this->policy = csr->policy;

    if (policy == segment) {
      SAFE_ALOC_UM(this->nnz_gpu, n_gpu * sizeof(IdxType));
      memcpy(this->nnz_gpu, csr->nnz_gpu, n_gpu * sizeof(IdxType))
    }

    cout << "Building csc matrix from a csr matrix." << endl;
    cout << "Height: " << height << " Width: " << width << " nnz: " << nnz
         << endl;

    SAFE_ALOC_UM(this->cscColPtr, get_col_ptr_size());
    SAFE_ALOC_UM(this->cscRowIdx, get_row_idx_size());
    SAFE_ALOC_UM(this->cscVal, get_val_size());

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

  IdxType *nnz_gpu; // number of nnz per GPU
  IdxType *displacements;

  int rank;
  unsigned n_gpu;
  int device;
};

// =============================== Dense Matrix
// =================================
template <typename IdxType, typename DataType> class DenseMatrix {
public:
  DenseMatrix() : height(0), width(0), order(row_major), n_gpu(0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->n_gpu);

    val = NULL;
    dim_gpu = NULL;
    policy = none;
  }
  /* Dense matrix with a random uniform value */
  DenseMatrix(IdxType _height, IdxType _width, enum MajorOrder _order)
      : height(_height), width(_width), order(_order), n_gpu(0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->n_gpu);
    if (rank == 0) {
      SAFE_ALOC_UM(val, get_mtx_size());
      srand(RAND_INIT_SEED);
      for (IdxType i = 0; i < get_mtx_num(); i++)
        val[i] = (DataType)rand0to1();
    }

    dim_gpu = NULL;
    policy = none;
  }
  /* Dense matrix with a fixes value _val set to all the coords */
  DenseMatrix(IdxType _height, IdxType _width, DataType _val,
              enum MajorOrder _order)
      : height(_height), width(_width), order(_order), n_gpu(0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->n_gpu);

    if (rank == 0) {
      SAFE_ALOC_UM(val, get_mtx_size());
      srand(RAND_INIT_SEED);
      for (IdxType i = 0; i < get_mtx_num(); i++)
        val[i] = (DataType)_val;
    }

    dim_gpu = NULL;
    policy = none;
  }

  void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy) {
    this->n_gpu = _n_gpu;
    this->policy = _policy;
    assert(this->n_gpu != 0);
    assert(this->policy != none);

    if (policy == replicate) {
      // same reasoning as above
      if (rank > 0) {
        SAFE_ALOC_UM(val, get_mtx_size());
      }
      MPI_Bcast(val, get_mtx_num(), get_mpi_type<DataType>(), 0,
                MPI_COMM_WORLD);
    } else if (policy == segment) {
      /* nnz_gpu is the equivalent */
      IdxType first_order = (order == row_major) ? height : width;
      IdxType second_order = (order == row_major) ? width : height;
      SAFE_ALOC_UM(dim_gpu, n_gpu * sizeof(IdxType));
      IdxType *disp_amount;

      if (rank == 0) {
        displacements = (IdxType *)malloc(n_gpu * sizeof(IdxType));
        disp_amount = (IdxType *)malloc(n_gpu * sizeof(IdxType));
        int sum = 0;
        IdxType avg_val = ceil((double)first_order / n_gpu);
        for (unsigned i = 0; i < n_gpu; i++) {
          dim_gpu[i] = min((i + 1) * avg_val, first_order) - i * avg_val;
          displacements[i] = sum;
          sum += (i * avg_val) * second_order;
          disp_amount[i] = dim_gpu[i] * second_order;
        }
      }

      MPI_Bcast(dim_gpu, n_gpu, get_mpi_type<IdxType>(), 0, MPI_COMM_WORLD);

      if (rank > 0) {
        SAFE_ALOC_UM(val, second_order * get_dim_gpu_size(i));
      }

      /* disp_amount is only required when sending the data, afterwards every
       * single rank will be able to retrieve that value */
      MPI_Scatterv(val, disp_amount, displacements, get_mpi_type<DataType>(),
                   val, second_order * get_dim_gpu_size(rank),
                   get_mpi_type<DataType>(), 0, MPI_COMM_WORLD);

      if (rank == 0) {
        free(disp_amount);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  ~DenseMatrix() {
    SAFE_FREE_GPU(val);
    SAFE_FREE_GPU(dim_gpu);
    if (displacements)
      free(displacements);
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

  void sync2cpu(unsigned i_gpu) {
    assert(val_gpu != NULL);
    assert(i_gpu < n_gpu);

    if (policy == segment) {
      // CUDA_SAFE_CALL(cudaSetDevice(i_gpu));
      IdxType first_order = (order == row_major) ? height : width;
      IdxType second_order = (order == row_major) ? width : height;
      IdxType avg_val = ceil((double)first_order / n_gpu);
      /* A gather should happens? */
      CUDA_SAFE_CALL(cudaMemcpy(
          &val[i_gpu * avg_val * second_order], val_gpu[i_gpu],
          second_order * get_dim_gpu_size(i_gpu), cudaMemcpyDeviceToHost));
    } else if (policy == replicate) {
      /* Here instead what should happen? A sort of reduction with an average */
      // CUDA_SAFE_CALL(cudaSetDevice(i_gpu));
      // CUDA_SAFE_CALL(cudaMemcpy(val, val_gpu[i_gpu], get_mtx_size(),
      //                           cudaMemcpyDeviceToHost));
    }
  }
  void plusDenseMatrixGPU(DenseMatrix const &dm, DataType alpha,
                          DataType beta) {
    if (n_gpu != 0 && policy != none) {
      dim3 blockDim(NUM_THREADS_PER_BLK);
      dim3 gridDim(((get_mtx_num() - 1) / NUM_THREADS_PER_BLK) + 1);

      IdxType second_order = (order == row_major) ? width : height;
      denseVector_plusEqual_denseVector<<<gridDim, blockDim>>>(
          val, dm.val, alpha, beta, second_order * get_dim_gpu_size(rank));

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
  // num of rows or cols (leading dim) per gpu depending on row-major or
  // col-major
  IdxType *dim_gpu;
  IdxType *displacements;
  int rank;
  int device;
  unsigned n_gpu;
  enum GpuSharePolicy policy;
  enum MajorOrder order;
};

// =============================== Dense Vector
// =================================
template <typename IdxType, typename DataType> class DenseVector {
public:
  DenseVector() : length(0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->n_gpu);

    this->val = NULL;
  }
  DenseVector(IdxType _length) : length(_length), n_gpu(0), policy(none) {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->n_gpu);

    if (rank == 0) {
      SAFE_ALOC_UM(this->val, get_vec_size());
      srand(RAND_INIT_SEED);
      for (IdxType i = 0; i < this->get_vec_length(); i++)
        (this->val)[i] = (DataType)rand0to1();
    }

    MPI_Bcast(val, get_vec_length(), get_mpi_type<DataType>(), 0,
              MPI_COMM_WORLD);
  }
  DenseVector(IdxType _length, DataType _val)
      : length(_length), n_gpu(0), policy(none) {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->n_gpu);

    if (rank == 0) {
      SAFE_ALOC_UM(this->val, get_vec_size());
      srand(RAND_INIT_SEED);
      for (IdxType i = 0; i < this->get_vec_length(); i++)
        (this->val)[i] = _val;
    }

    MPI_Bcast(val, get_vec_length(), get_mpi_type<DataType>(), 0,
              MPI_COMM_WORLD);
  }
  DenseVector(const DenseVector &dv)
      : length(dv.length), n_gpu(dv.n_gpu), policy(dv.policy) {
    SAFE_ALOC_UM(val, get_vec_size());
    memcpy(val, dv.val, get_vec_size());

    // if (n_gpu != 0 && policy != none) {
    //   SAFE_ALOC_HOST(val_gpu, n_gpu * sizeof(DataType *));
    //   for (unsigned i = 0; i < n_gpu; i++) {
    //     CUDA_SAFE_CALL(cudaSetDevice(i));
    //     SAFE_ALOC_GPU(val_gpu[i], get_vec_size());
    //     CUDA_SAFE_CALL(cudaMemcpy(val_gpu[i], dv.val_gpu[i], get_vec_size(),
    //                               cudaMemcpyDeviceToDevice));
    //   }
    // }
  }
  void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy) {
    this->n_gpu = _n_gpu;
    this->policy = _policy;
    assert(this->n_gpu != 0);
    assert(this->policy != none);
    assert(this->policy != segment); // assume now vector does not need
                                     // partition
    // prefetch data to the GPU
    // if (policy == replicate) {
    //   SAFE_ALOC_HOST(val_gpu, n_gpu * sizeof(DataType *));
    //   for (unsigned i = 0; i < n_gpu; i++) {
    //     CUDA_SAFE_CALL(cudaSetDevice(i));
    //     SAFE_ALOC_GPU(val_gpu[i], get_vec_size());
    //     CUDA_SAFE_CALL(cudaMemcpy(val_gpu[i], val, get_vec_size(),
    //                               cudaMemcpyHostToDevice));
    //   }
    // }
  }
  void sync2cpu(
      unsigned i_gpu) // all gpus have the same res vector, pick from any one
  {
    assert(i_gpu < n_gpu);
    assert(val_gpu != NULL);
    // Prefetch data to the CPU
    // CUDA_SAFE_CALL(cudaSetDevice(i_gpu));
    // CUDA_SAFE_CALL(cudaMemcpy(val, val_gpu[i_gpu], get_vec_size(),
    //                           cudaMemcpyDeviceToHost));
  }
  void plusDenseVectorGPU(DenseVector const &dv, DataType alpha,
                          DataType beta) {
    if (n_gpu != 0 && policy != none) {
      dim3 blockDim(NUM_THREADS_PER_BLK);
      dim3 gridDim(((get_vec_length() - 1) / NUM_THREADS_PER_BLK) + 1);
      // as we're dealing with just a GPU per thread there's no need for this
      // CUDA_SAFE_CALL(cudaSetDevice(i));
      denseVector_plusEqual_denseVector<<<gridDim, blockDim>>>(
          val, dv.val, alpha, beta, get_vec_length());

      CUDA_CHECK_ERROR();
    }
  }
  ~DenseVector() {
    SAFE_FREE_GPU(this->val);
    // SAFE_FREE_MULTI_GPU(val_gpu, n_gpu);
  }
  size_t get_vec_size() { return (size_t)length * sizeof(DataType); }
  size_t get_vec_length() { return (size_t)length; }

public:
  IdxType length;
  DataType *val;
  int rank;
  unsigned n_gpu;
  enum GpuSharePolicy policy;
};

#endif
