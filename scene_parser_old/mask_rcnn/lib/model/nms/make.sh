#!/usr/bin/env bash

CUDA_PATH=/opt/cuda/include

cd src
echo "Compiling stnm kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
