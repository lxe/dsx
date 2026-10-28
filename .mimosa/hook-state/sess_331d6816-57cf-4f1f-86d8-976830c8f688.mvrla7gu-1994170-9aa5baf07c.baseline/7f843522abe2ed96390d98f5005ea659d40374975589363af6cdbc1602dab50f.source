#!/bin/bash
# DSX_DEFER=1 (all but each token's best-ranked CPU pick deferred) vs 3 (production) vs 2, interleaved, 400 tokens after
# 3000; decode-eval KL for 3 / 2 / 1 (full output kept). Kill for 1 and 2: decode KL up > 0.01 vs defer=3.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: DSX_DEFER 1/2/3 A/B + decode evals, serve stopped (~30 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 5
M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"
for d in 1 3 1 3; do sleep 25
  echo "defer=$d: $(DSX_DEFER=$d ./guard.sh df$RANDOM 118 /tmp/dsx-cur gen $M --ids @prompt3000.ids -n 400 --ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json 2>&1 | grep -E "^decode" | sed -E "s/GPU0 expert swaps.*//") $(sensors | grep Package | awk '{print $4}')"
done
for d in 3 2 1; do sleep 25
  echo "== eval --decode defer=$d"; DSX_DEFER=$d DSX_ADAPT=0 ./guard.sh ev$RANDOM 118 /tmp/dsx-cur eval $M --eval eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json --decode --seqs 2 > /tmp/ev_$d.out 2>&1
  grep -E "^\{" /tmp/ev_$d.out || tail -5 /tmp/ev_$d.out
done
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN20-DONE $(date)"
