#!/bin/bash
# quant_job.sh NAME MAP [FLAGS] - on a hydrated pod: the GPTQ chain over all 40 layers with the tier map MAP, then the
# GGUF at /work/ds41/out/1017-NAME.gguf. FLAGS go to v41_quant.py (default: --rot --refit).
set -eo pipefail
NAME=$1; MAP=$2; shift 2; FLAGS=${*:---rot --refit}
W=/work/ds41; Q=$W/qwork-$NAME; CAL=$W/calib3
export V41_MODEL=$W/fp8-1017 V41_REF=$W/fp8-1017/inference V41_WORK=$CAL V41_ROWS=32768 \
  V41_BATCH_EXPERTS=${V41_BATCH_EXPERTS:-8} V41_BATCH_TOKENS=${V41_BATCH_TOKENS:-200000} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/quant2
t0=$(date +%s)
[ -f $Q/h/L00.bin ] || python v41_quant.py init --qwork $Q --calib $CAL
for L in $(seq 0 39); do
  LL=$(printf %02d $L)
  [ -f $Q/q/L$LL.json ] && continue
  python v41_quant.py layer $L --qwork $Q --map $MAP $FLAGS --gc 2>&1 | grep --line-buffered -E "done in|quant L$L:|Error|FAILED|time:"
  [ $L -ge 2 ] && rm -f $Q/h/L$(printf %02d $((L-1))).bin
done
echo "CHAIN $(( $(date +%s) - t0 )) s"
mkdir -p $W/out
python v41_gguf.py --out $W/out/1017-$NAME.gguf --experts $Q/q --no-engram --threads 8 2>&1 | tail -2
echo "JOB-DONE $(( $(date +%s) - t0 )) s"
