#!/bin/bash
# build libqkern.so (sm_86 = RTX 30xx, sm_120 = RTX 50xx)
set -e
cd "$(dirname "$0")"
nvcc -O3 -shared -Xcompiler -fPIC -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 -o libqkern.so.new qkern.cu && mv -f libqkern.so.new libqkern.so   # atomic: running quant processes keep their mapping
