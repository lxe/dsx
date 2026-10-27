#!/bin/bash
# t5.sh (on the V100 VM, from the repo root): build bench/t5.fatbin from bench/t5_kernels.cu (the expert_fast path
# as dsx.cu compiles it) and the bench/t5_bench harness, then run correctness + perf under the GPU lock.
# Usage: bash bench/t5.sh [correct|perf|all]   (T5_ARCH=70 default; the dual-arch gate compiles dsx.cu separately)
set -e
cd "$(dirname "$0")/.."
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${T5_ARCH:-70}
$NVCC --fatbin -std=c++17 -O3 --prec-div=true --prec-sqrt=true -Wno-deprecated-gpu-targets \
  -gencode=arch=compute_$ARCH,code=sm_$ARCH -Invidia/src bench/t5_kernels.cu -o bench/t5.fatbin
$NVCC -std=c++17 -O3 -Wno-deprecated-gpu-targets -gencode=arch=compute_$ARCH,code=sm_$ARCH \
  -Invidia/src bench/t5_bench.cu -o bench/t5_bench -lcuda
flock /tmp/dsx-gpu.lock env CUDA_VISIBLE_DEVICES=GPU-4e4e3d86-f648-3706-6489-f5e8e17fcab8 \
  bench/t5_bench "${1:-all}" bench/t5.fatbin
