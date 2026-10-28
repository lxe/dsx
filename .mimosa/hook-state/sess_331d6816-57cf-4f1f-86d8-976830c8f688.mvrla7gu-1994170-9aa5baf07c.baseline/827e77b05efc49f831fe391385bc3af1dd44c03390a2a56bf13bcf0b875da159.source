#!/bin/bash
# v3a: v2's tier maps unchanged (edges from qwork-joint/alloc, layers 2-36 from the joint map), GPTQ redone on a new
# calibration mix (CALIB: tokens/meta/hashes from v41_calib.py prep). Isolates what the calibration text does.
#   quant_chain_v3.sh QWORK GPU CALIB
cd ~/ai/ds41/quant2
Q=$1; GPU=$2; CAL=$3
V2=~/ai/ds41/qwork-joint/alloc; JOINT=~/ai/ds41/alloc-joint-2-36.json
export V41_BATCH_EXPERTS=${V41_BATCH_EXPERTS:-4} V41_BATCH_TOKENS=${V41_BATCH_TOKENS:-80000} CUDA_VISIBLE_DEVICES=$GPU V41_WORK=$CAL V41_ROWS=8192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $Q
[ -f $Q/h/L00.bin ] || ../venv/bin/python v41_quant.py init --qwork $Q --calib $CAL || exit 1
for L in $(seq 0 39); do
  LL=$(printf %02d $L)
  [ -f $Q/q/L$LL.json ] && { echo "quant L$L done"; continue; }
  case $L in
    0|1|37|38|39) MAP=$V2/L$LL.json;;
    *) MAP=$JOINT;;
  esac
  ../venv/bin/python v41_quant.py layer $L --qwork $Q --map $MAP --gc 2>&1 | grep --line-buffered -v "UserWarning\|from_numpy"
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "quant L$L FAILED"; exit 1; }
  [ $L -ge 2 ] && rm -f $Q/h/L$(printf %02d $((L-1))).bin
done
echo "quant chain done"
