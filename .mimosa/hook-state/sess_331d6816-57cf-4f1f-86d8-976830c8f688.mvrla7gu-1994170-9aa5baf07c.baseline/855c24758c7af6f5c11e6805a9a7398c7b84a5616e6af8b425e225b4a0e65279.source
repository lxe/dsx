#!/bin/bash
# v3a end to end on the build machine: calibration mix with the full model's reasoning traces (corpus3/think.jsonl),
# GPTQ with v2's tier maps, GGUF, Lucebox eval, then serve it. Each step under the memory guard; resumable.
set -e
cd ~/ai/ds41
CAL=~/ai/ds41/calib3; Q=~/ai/ds41/qwork-v3a; OUT=gguf/1017-v3a.gguf
mkdir -p $CAL
if [ ! -f $CAL/hashes.npy ]; then
  (cd quant2 && V41_WORK=$CAL ../guard.sh prep3 60 ../venv/bin/python v41_calib.py prep --n 160 --seq 2048 \
     --corpus ~/ai/ds41/corpus3 --think-weight 16)
fi
./guard.sh chain3 110 quant2/quant_chain_v3.sh $Q 0 $CAL
[ -f $OUT ] || (cd quant2 && V41_WORK=~/ai/ds41/calib ../guard.sh gguf3 40 ../venv/bin/python v41_gguf.py --out ../$OUT \
     --experts $Q/q --no-engram --threads 4)
echo "=== eval v3a"
DSX_ADAPT=0 ./guard.sh eval3 120 ./engine/target/release/dsx eval --model $OUT --engram fp8-1017 --eval eval \
  --gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 104 --rank calib/rank.json 2>&1 | grep -E "^\{|seq 7"
echo "=== v2 reference: kl 0.395 top1 0.741 ppl 10.93"
tmux new-session -d -s serve "./dsx-serve.sh $OUT 2>&1 | tee serve.out"
echo V3A_DONE
