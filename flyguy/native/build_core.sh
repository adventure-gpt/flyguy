#!/bin/sh
# flyguy: the native network as a library for the Python trainer, on Linux (a rented GPU machine).
#   sh flyguy/native/build_core.sh      -> flyguy/native/libflyguy_core.so   (needs the CUDA toolkit: nvcc)
set -e
cd "$(dirname "$0")"
nvcc -O3 -std=c++17 -DFLYGUY_CORE -shared -Xcompiler -fPIC -o libflyguy_core.so flyguy_train.cu
echo "built $(pwd)/libflyguy_core.so"
