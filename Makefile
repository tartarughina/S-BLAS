# ------------------------------------------------------------------------
# File: Makefile
# S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
# ------------------------------------------------------------------------
# Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
# and Linghao Song
# GitHub repo: http://www.github.com/uuudown/S-BLAS
# PNNL-IPID: 31803-E, IR: PNNL-31803
# MIT Lincese.
# ------------------------------------------------------------------------

# environment parameters
CUDA_INSTALL_PATH ?= ${NVIDIA_PATH}/cuda
CUDA_MATH_PATH ?= ${NVIDIA_PATH}/math_libs
CUDA_COMM_PATH ?= ${NVIDIA_PATH}/comm_libs/nccl
#compiler
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
CC = g++

#nvcc parameters
NVCC_FLAGS = -O3 -w -m64 -gencode=arch=compute_80,code=compute_80

#debugging
#NVCC_FLAGS = -O0 -g -G -m64 -gencode=arch=compute_70,code=compute_70

CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_MATH_PATH)/include -I$(CUDA_COMM_PATH)/include
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_MATH_PATH)/lib64 -L$(CUDA_COMM_PATH)/lib -ldl -lcudart -lcusparse -Xcompiler -fopenmp -lnccl

all: unit_test spmm_test spmm_test_um spmv_test spmv_test_um

unit_test: unit_test.cu matrix.h spmv.h spmm.h mmio_highlevel.h kernel.h utility.h
	$(NVCC) -ccbin $(CC) $(NVCC_FLAGS) unit_test.cu  $(CUDA_INCLUDES) $(CUDA_LIBS) -o $@

spmm_test: spmm_test.cu matrix.h spmv.h spmm.h mmio_highlevel.h kernel.h utility.h
	$(NVCC) -ccbin $(CC) $(NVCC_FLAGS) spmm_test.cu  $(CUDA_INCLUDES) $(CUDA_LIBS) -o $@

spmm_test_um: spmm_test_um.cu matrix_um.h spmv_um.h spmm_um.h mmio_highlevel.h kernel.h utility.h
	$(NVCC) -ccbin $(CC) $(NVCC_FLAGS) spmm_test_um.cu  $(CUDA_INCLUDES) $(CUDA_LIBS) -o $@

spmv_test: spmv_test.cu matrix.h spmv.h spmm.h mmio_highlevel.h kernel.h utility.h
	$(NVCC) -ccbin $(CC) $(NVCC_FLAGS) spmv_test.cu  $(CUDA_INCLUDES) $(CUDA_LIBS) -o $@

spmv_test_um: spmv_test_um.cu matrix_um.h spmv_um.h spmm_um.h mmio_highlevel.h kernel.h utility.h
	$(NVCC) -ccbin $(CC) $(NVCC_FLAGS) spmv_test_um.cu  $(CUDA_INCLUDES) $(CUDA_LIBS) -o $@
clean:
	rm unit_test spmm_test spmm_test_um spmv_test spmv_test_um
