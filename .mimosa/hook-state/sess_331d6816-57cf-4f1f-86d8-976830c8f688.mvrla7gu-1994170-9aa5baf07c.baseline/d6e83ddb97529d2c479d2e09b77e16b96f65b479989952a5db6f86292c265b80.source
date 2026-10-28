#!/bin/bash
# per-layer-group KL sensitivity: Gaussian noise (0.173 x RMS = 3% energy) on routed expert outputs of 5 layers at a time
# (DSX_NOISE); the KL rise per group, normalized to mean 1, is layer-weights-v5.json (v41_alloc.py --layer-weights)
cd ~/ai/ds41
P="--gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
M=hf-v4b/DeepSeek-V4.1-Flash-1017-dsx-2bit-00001-of-00004.gguf
B=./dsx-k/target/release/dsx
echo -n "no noise (must be kl 0.372944): "; DSX_ADAPT=0 ./guard.sh sensb 120 $B eval --model $M --engram fp8-1017 --eval eval $P 2>&1 | grep -E "^\{|rror"
for a in 0 5 10 15 20 25 30 35; do
  b=$((a+4))
  echo -n "layers $a-$b: "
  DSX_NOISE=$a-$b:0.173 DSX_ADAPT=0 ./guard.sh sens$a 120 $B eval --model $M --engram fp8-1017 --eval eval $P 2>&1 | grep -E "^\{|rror"
done
echo SENS-DONE
