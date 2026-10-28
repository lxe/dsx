#!/bin/bash
# after both v2 quant chains: assemble each GGUF, then quality-eval each (Lucebox 1017 set, prefill path) and print a table
cd ~/ai/ds41/quant2
export V41_WORK=~/ai/ds41/calib
for v in joint t100; do
  [ -f ../gguf/1017-v2-$v.gguf ] && continue
  ./../guard.sh gguf-$v 40 ../venv/bin/python v41_gguf.py --out ../gguf/1017-v2-$v.gguf --experts ../qwork-$v/q --no-engram --threads 4 2>&1 | tail -2 || { echo "GGUF $v FAILED"; exit 1; }
done
cd ~/ai/ds41
P="--gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
for v in joint t100; do
  echo "=== eval v2-$v"
  DSX_ADAPT=0 ./guard.sh eval 120 ./engine/target/release/dsx eval --model gguf/1017-v2-$v.gguf --engram fp8-1017 --eval eval $P 2>&1 | grep -E "^\{"
done
echo "=== v1 (for reference): kl 0.462284 top1 0.715054 ppl 11.6631"
echo FINISH_V2_DONE
