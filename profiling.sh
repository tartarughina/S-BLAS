#!/bin/bash

nsys profile --stats=true --trace=cuda,nvtx \
--cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true \
--output spmm --force-overwrite true \
./spmm_test_um 2 ./matrices/s3dkq4m2.mtx 64 1.0 1.0 4 0

nsys profile --stats=true --trace=cuda,nvtx \
--cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true \
--output spmm-umt --force-overwrite true \
./spmm_test_um 2 ./matrices/s3dkq4m2.mtx 64 1.0 1.0 4 1

nsys profile --stats=true --trace=cuda,nvtx \
--cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true \
--output spmv --force-overwrite true \
./spmv_test_um ./matrices/webbase-1M.mtx 1.0 1.0 4 0

nsys profile --stats=true --trace=cuda,nvtx \
--cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true \
--output spmv-umt --force-overwrite true \
./spmv_test_um ./matrices/webbase-1M.mtx 1.0 1.0 4 1
