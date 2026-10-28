#!/bin/bash
# v5 = v4b with one allocation over all 40 layers, errors weighted by measured KL sensitivity (alloc-v5.json), end to end:
# alloc-v5.json: v41_alloc.py --work CALIB --layers 0-39 --budget-bpw 2.03125 --max-ternary-frac 1.0 --layer-weights layer-weights-v5.json
# (same bytes as v4b; layer weights from sens_sweep.sh)
# quant chain (all 40 layers --rot) -> GGUF -> 4 shards with dsx.expert_rot -> Lucebox eval (prefill, positions 512+)
# -> serve -> S01/S02 x3 at temperature 1.0 (run separately from the workstation).
# expert / token batches sized for one 24 GB GPU
set -eo pipefail
cd ~/ai/ds41
Q=~/ai/ds41/qwork-v5; CAL=~/ai/ds41/calib3
V2=~/ai/ds41/qwork-joint/alloc; JOINT=~/ai/ds41/alloc-joint-2-36.json
export V41_BATCH_EXPERTS=1 V41_BATCH_TOKENS=40000 V41_WORK=$CAL V41_ROWS=8192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd quant2
[ -f $Q/h/L00.bin ] || CUDA_VISIBLE_DEVICES=0 ../venv/bin/python v41_quant.py init --qwork $Q --calib $CAL
for L in $(seq 0 39); do
  LL=$(printf %02d $L)
  [ -f $Q/q/L$LL.json ] && { echo "L$L done"; continue; }
  MAP=~/ai/ds41/alloc-v5.json
  CUDA_VISIBLE_DEVICES=0 ../guard.sh v5-L$LL 110 ../venv/bin/python v41_quant.py layer $L --qwork $Q --map $MAP --rot --refit --gc 2>&1 | grep --line-buffered -E "done in|quant L$L:|Error|FAILED"
  [ $L -ge 2 ] && rm -f $Q/h/L$(printf %02d $((L-1))).bin
done
cd ~/ai/ds41
(cd quant2 && V41_WORK=~/ai/ds41/calib ../guard.sh gguf-v5 40 ../venv/bin/python v41_gguf.py --out ../gguf/1017-v5.gguf --experts $Q/q --no-engram --threads 4 2>&1 | tail -2)
mkdir -p hf-v5
./dsx/target/release/dsx split gguf/1017-v5.gguf hf-v5/DeepSeek-V4.1-Flash-1017-dsx-2bit --shards 4 --name "DeepSeek-V4.1-Flash-1017 dsx 2-bit v5" --expert-rot-from $Q/q
rm -f gguf/1017-v5.gguf
ln -f hf-v3a/vision.safetensors hf-v5/vision.safetensors
M=hf-v5/DeepSeek-V4.1-Flash-1017-dsx-2bit-00001-of-00004.gguf
P="--gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
echo "=== v5 eval, all positions";  DSX_ADAPT=0 ./guard.sh ev5 120 ./dsx/target/release/dsx eval --model $M --engram fp8-1017 --eval eval $P 2>&1 | grep -E "^\{|rror"
echo "=== v5 eval, positions 512+"; DSX_ADAPT=0 ./guard.sh ev5b 120 ./dsx/target/release/dsx eval --model $M --engram fp8-1017 --eval eval $P --from 512 2>&1 | grep -E "^\{|rror"
echo "=== reference v4b: all kl 0.372944 top1 0.744 ppl 10.68"
tmux kill-session -t serve 2>/dev/null || true
tmux new-session -d -s serve ./dsx-serve.sh
echo V5-DONE
