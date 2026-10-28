#!/bin/bash
# Budget lever #2: DSX_DEFER=2 vs 3 on the T16 production build. Predicted: CPU wait 4.34 -> ~3.0 ms, ~5% faster.
# Kill: < 2% faster, or decode-eval KL up > 0.01. Interleaved 3 2 2 3 3 2 (400 tokens after the 3000 prompt, 25 s
# cool-downs), then eval --decode --seqs 2 for each. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: DSX_DEFER 2 vs 3 A/B + decode evals, serve stopped (~25 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 5
M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"
for d in 3 2 2 3 3 2; do sleep 25
  echo "defer=$d: $(DSX_DEFER=$d ./guard.sh df$RANDOM 118 /tmp/dsx-cur gen $M --ids @prompt3000.ids -n 400 --ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json 2>&1 | grep -E "^decode" | sed -E "s/GPU0 expert swaps.*//") $(sensors | grep Package | awk '{print $4}')"
done
for d in 3 2; do sleep 25
  echo -n "eval --decode defer=$d: "; DSX_DEFER=$d DSX_ADAPT=0 ./guard.sh df$RANDOM 118 /tmp/dsx-cur eval $M --eval eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json --decode --seqs 2 2>&1 | grep -E "^\{"
done
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN19-DONE $(date)"
