#!/bin/bash
# Quantize 1017 layer by layer on GPU0, one layer behind the calibration chain (GPU1): per-layer tier allocation at a
# bits/weight budget (edge layers richer), then GPTQ at those tiers. Output: ~/ai/ds41/qwork/q/LNN.{bin,json}.
cd ~/ai/ds41/quant
export CUDA_VISIBLE_DEVICES=0 V41_WORK=~/ai/ds41/calib V41_ROWS=8192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=~/ai/ds41/qwork
mkdir -p $Q/alloc
[ -f $Q/h/L00.bin ] || ../venv/bin/python v41_quant.py init --qwork $Q || exit 1
for L in $(seq ${1:-0} 39); do
  LL=$(printf %02d $L)
  [ -f $Q/q/L$LL.json ] && { echo "quant L$L done"; continue; }
  while [ ! -f ~/ai/ds41/calib/stats/L$LL.json ]; do sleep 30; done
  case $L in 0|1|37|38|39) bpw=2.25;; *) bpw=2.0;; esac
  ../venv/bin/python v41_alloc.py --layers $L-$L --budget-bpw $bpw --max-ternary-frac 0.75 --out $Q/alloc/L$LL.json | tail -1
  ../venv/bin/python v41_quant.py layer $L --qwork $Q --map $Q/alloc/L$LL.json --gc 2>&1 | grep --line-buffered -v "UserWarning\|from_numpy"
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "quant L$L FAILED"; exit 1; }
done
echo "quant chain done"
