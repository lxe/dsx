#!/bin/bash
# v2 joint: layers 2-36 from one cross-layer allocation (JOINT map, 2.0 bpw over the 35 layers together, no ternary
# cap); edge layers 0, 1, 37-39 per layer at 2.25 bpw with v1's 75% cap.  quant_chain_joint.sh QWORK GPU JOINT_MAP
cd ~/ai/ds41/quant2
Q=$1; GPU=$2; JOINT=$3
export V41_BATCH_EXPERTS=${V41_BATCH_EXPERTS:-4} V41_BATCH_TOKENS=${V41_BATCH_TOKENS:-80000} CUDA_VISIBLE_DEVICES=$GPU V41_WORK=~/ai/ds41/calib V41_ROWS=8192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $Q/alloc
[ -f $Q/h/L00.bin ] || ../venv/bin/python v41_quant.py init --qwork $Q || exit 1
for L in $(seq 0 39); do
  LL=$(printf %02d $L)
  [ -f $Q/q/L$LL.json ] && { echo "quant L$L done"; continue; }
  case $L in
    0|1|37|38|39) ../venv/bin/python v41_alloc.py --layers $L-$L --budget-bpw 2.25 --max-ternary-frac 0.75 --out $Q/alloc/L$LL.json | tail -1; MAP=$Q/alloc/L$LL.json;;
    *) MAP=$JOINT;;
  esac
  ../venv/bin/python v41_quant.py layer $L --qwork $Q --map $MAP --gc 2>&1 | grep --line-buffered -v "UserWarning\|from_numpy"
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "quant L$L FAILED"; exit 1; }
  [ $L -ge 2 ] && rm -f $Q/h/L$(printf %02d $((L-1))).bin
done
echo "quant chain done"
