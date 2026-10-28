#!/bin/bash
# v2 quant chain: quant_chain_v2.sh QWORK GPU MAXTERN  (same 2.0/2.25 bpw budget as v1, different ternary cap)
cd ~/ai/ds41/quant2
Q=$1; GPU=$2; MT=$3
export V41_BATCH_EXPERTS=${V41_BATCH_EXPERTS:-4} V41_BATCH_TOKENS=${V41_BATCH_TOKENS:-80000} CUDA_VISIBLE_DEVICES=$GPU V41_WORK=~/ai/ds41/calib V41_ROWS=8192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $Q/alloc
[ -f $Q/h/L00.bin ] || ../venv/bin/python v41_quant.py init --qwork $Q || exit 1
for L in $(seq 0 39); do
  LL=$(printf %02d $L)
  [ -f $Q/q/L$LL.json ] && { echo "quant L$L done"; continue; }
  case $L in 0|1|37|38|39) bpw=2.25;; *) bpw=2.0;; esac
  ../venv/bin/python v41_alloc.py --layers $L-$L --budget-bpw $bpw --max-ternary-frac $MT --out $Q/alloc/L$LL.json | tail -1
  ../venv/bin/python v41_quant.py layer $L --qwork $Q --map $Q/alloc/L$LL.json --gc 2>&1 | grep --line-buffered -v "UserWarning\|from_numpy"
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "quant L$L FAILED"; exit 1; }
  # keep disk bounded: inputs of finished layers are not needed again
  [ $L -ge 2 ] && rm -f $Q/h/L$(printf %02d $((L-1))).bin
done
echo "quant chain done"
