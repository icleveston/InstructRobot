#!/usr/bin/env bash

CUDA_PATH=/opt/cuda-9.0

python setup.py build_ext --inplace
rm -rf build

# Choose cuda arch as you need
CUDA_ARCH="-gencode arch=compute_70,code=compute_70"

# compile NMS
cd model/nms/src
echo "Compiling nms kernels by nvcc..."
/opt/cuda-9.0/bin/nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=0 -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../
python build.py

# compile roi_pooling
cd ../../
cd model/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
/opt/cuda-9.0/bin/nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=0 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# # compile roi_align
# cd ../../
# cd model/roi_align/src
# echo "Compiling roi align kernels by nvcc..."
# nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
# 	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
# cd ../
# python build.py

# compile roi_crop
cd ../../
cd model/roi_crop/src
echo "Compiling roi crop kernels by nvcc..."
/opt/cuda-9.0/bin/nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu \
	 -D GOOGLE_CUDA=0 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_align (based on Caffe2's implementation)
cd ../../
cd modeling/roi_xfrom/roi_align/src
echo "Compiling roi align kernels by nvcc..."
/opt/cuda-9.0/bin/nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
	 -D GOOGLE_CUDA=0 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py
